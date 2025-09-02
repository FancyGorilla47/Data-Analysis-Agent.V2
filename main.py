import os
import urllib.parse
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import builtins
import traceback
from contextlib import redirect_stdout
from contextlib import contextmanager
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.vectorstores import AzureSearch
from langchain_core.tools import tool
from langchain_openai import AzureOpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
import operator
from typing import List
import io
import sys
import pandas as pd
import numpy as np
import collections
import datetime
import statistics
import json
import re

#TWILIO IMPORTS############################

#--- added from the implementation of whatsapp 
from typing import Optional  # add Optional
from fastapi import  HTTPException , BackgroundTasks ,Request # add HTTPException
from urllib.parse import urlsplit, urlunsplit  # for Twilio signature normalization

# Twilio
import twilio 
from twilio.rest import Client
from twilio.request_validator import RequestValidator

#--- end 

####################  added  twilio  ##################

# --- 1. Initialization (Runs once on server start) ---
load_dotenv()

#adding twilio env variables 
TWILIO_ACCOUNT_SID= os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN= os.getenv("TWILIO_AUTH_TOKEN", "")          # <-- fixed getenv
TWILIO_WHATSAPP_FROM= os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
PUBLIC_BASE_URL= (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")

ALLOWED_ORIGINS = [
    o.strip() for o in (os.getenv("ALLOWED_ORIGINS", "http://localhost:8080")).split(",")
    if o.strip()
]
print("Environment variables loaded.")


#----------twilio setup--------------- 
_twilio_client: Optional[Client] = None
_validator: Optional[RequestValidator] = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM:
    _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    _validator = RequestValidator(TWILIO_AUTH_TOKEN)
    print("Twilio WhatsApp enabled.")
else:
    print("Twilio WhatsApp not configured (missing env vars).")

####################  added  twilio  ##################

# --- Single App Initialization ---
app = FastAPI(
    title="SQL Agent API",
    description="API to interact with a LangChain SQL Agent (Tool-Calling) and Refiner",
)

# --- Add Middleware ---
ALLOWED_ORIGINS = (os.getenv("ALLOWED_ORIGINS") or "http://localhost:8080").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Define the State for our Graph ---
# This is the central object that will be passed between our agents (nodes).
class AgentState(TypedDict):
    """
    Represents the state of our multi-agent workflow.
    """
    input: str
    plan: List[str]
    # NEW STRUCTURE: Each dict will now be {"task": str, "structured_data": List[List[Any]]}
    # where the first inner list is the headers.
    sql_results: Annotated[List[Dict[str, Any]], operator.add]
    analysis_summary: str
    python_last_failed_code: str | None 
    python_notebook: Annotated[List[str], operator.add]
    python_retry_count: int
    python_error: str | None  
    final_data: List[Dict[str, Any]] | None
    final_report: str
    sql_error: str | None
    sql_retry_count: int

class StreamingStdOut(io.StringIO):
    """A custom StringIO class that writes to an asyncio.Queue and the original stdout."""
    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self.queue = queue
        self.original_stdout = sys.__stdout__ # Keep a reference to the original stdout

    def write(self, text: str):
        """This method is called by print(). We write to both the queue and the original stdout."""
        self.queue.put_nowait(text)
        self.original_stdout.write(text) # Write to the terminal as well

    def flush(self):
        """Flush the original stdout."""
        self.original_stdout.flush()

@contextmanager
def redirect_stdout_to_queue(queue: asyncio.Queue):
    """A context manager to temporarily redirect sys.stdout to our custom class."""
    original_stdout = sys.stdout
    sys.stdout = StreamingStdOut(queue)
    try:
        yield
    finally:
        sys.stdout = original_stdout


# 1a) Azure OpenAI clients
try:
    openai_O1_llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_o1"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_O1"],
        streaming=False
    )
    print("Azure OpenAI Agent LLM (for tool calling) initialized.")

    openai_O3_llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_O3_MINI"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_O3_MINI"],
        streaming=True,
    )
    print("Azure OpenAI Refiner LLM initialized.")
    openai_GPT4_llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_GPT4"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_GPT4"],
        streaming=False
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI clients: {e}")
    raise

from summary import create_summary_agent_and_router, _extract_output_text
summary_agent_executor, summary_router = create_summary_agent_and_router(openai_O3_llm)
app.include_router(summary_router)


# 1b) Build SQLAlchemy engine
raw_conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
if not raw_conn_str:
    raise RuntimeError("AZURE_SQL_CONNECTION_STRING is not set in .env")

quoted_conn_str = urllib.parse.quote_plus(raw_conn_str)
db_uri = f"mssql+pyodbc:///?odbc_connect={quoted_conn_str}"
engine = create_engine(db_uri, echo=False)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION_O3_MINI"],
)
vector_store = AzureSearch(
    azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
    azure_search_key=os.environ["AZURE_AI_SEARCH_ADMIN_KEY"],
    index_name=os.environ["AZURE_AI_SEARCH_INDEX_NAME"],
    embedding_function=embeddings.embed_query
)
retriever = vector_store.as_retriever(
    search_type="hybrid", 
    k=15,
    search_kwargs={
        "query_type": "semantic",
        "semantic_configuration_name": 'semantic-config'
    }
)

@tool(return_direct=True)
def execute_sql_and_get_results(query: str) -> str | List[any]:
    """
    Executes a SQL query against the database and returns the results,
    including headers. This is the ONLY tool for running SELECT queries.
    The output is a list of lists, where the first inner list contains the column headers.
    If the query fails, it returns a string with the detailed error message.
    """
    print(f"--- Executing SQL with custom tool---")
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            headers = list(result.keys())
            # .fetchall() returns a list of Row objects, which behave like tuples.
            # We must convert each Row to a standard list for consistent formatting.
            rows = [list(row) for row in result.fetchall()] # <-- THE FIX
            # On success, return a pure list of lists.
            return [headers] + rows
    except Exception as e:
        # On failure, catch the exception and return it as a string.
        # The agent will see this error message as the tool's output and can use it to self-correct.
        print(f"--- SQL Execution Failed ---\n{e}")
        return f"Query failed with the following error: {e}"
@tool
def ask_database_expert(question: str) -> str:
    """
    Use this tool FIRST to get expert knowledge about the database.
    Ask it questions about table schemas, column meanings...
    """
    print(f"INFO: Consulting the database expert with question: '{question}'")
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])
# 1c) LangChain SQL toolkit
try:
    sql_db = SQLDatabase(engine)
    sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=openai_GPT4_llm)
    print("SQL Database Toolkit initialized.")
except Exception as e:
    print(f"Error initializing SQLDatabase or Toolkit: {e}")
    raise


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    A custom import function that only allows imports from a predefined list.
    """
    ALLOWED_IMPORTS = {
    # 🔢 Core Data Manipulation
    'pandas', 'numpy',

    # 📈 Statistical & Scientific Analysis
    'statistics', 'math', 'scipy', 'statsmodels', 'scipy.stats', 

    # 🤖 Machine Learning
    'sklearn', 'sklearn.preprocessing', 'sklearn.model_selection', 'sklearn.cluster',
    'xgboost', 'xgboost.sklearn', 'lightgbm', 

    # 📊 Time-Series Analysis
    'prophet', 'darts', 'sktime', 'tsfresh',

    # 📋 Tabular/Pretty Formatting (for future use)
    'tabulate', 'prettytable', 'texttable',

    # 📄 Output Formatting & Enhancement
    'json', 'yaml', 're', 'html', 'csv', 'io', 'base64',

    # 🗓️ Date/Time Handling
    'datetime', 'dateutil', 'time', 'calendar',

    # 🧠 Lightweight Utilities
    'collections', 'itertools', 'string', 'difflib', 'functools', 'operator', 'copy',
    
    # 🔐 Sandboxing & Debugging
    'ast', 'traceback', 'contextlib',

    # Specific pandas sub-libraries
    'pandas.tseries.offsets', 'pandas.api.types',
    }
    
    # This block allows imports of submodules from allowed libraries
    parts = name.split('.')
    if parts[0] in ALLOWED_IMPORTS and len(parts) > 1:
        if parts[1] == 'tseries':
            return __import__(name, globals, locals, fromlist, level)
        elif parts[1] == 'api':
            return __import__(name, globals, locals, fromlist, level)
        elif parts[1] in ALLOWED_IMPORTS:
            # Catches nested submodules like sklearn.cluster
            return __import__(name, globals, locals, fromlist, level)
            
    if name in ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
            
    raise ImportError(f"Import of module '{name}' is not allowed.")



class Plan(BaseModel):
    """A plan to answer the user's question, broken down into a list of steps."""
    steps: List[str] = Field(
        description="A list of sequential steps to accomplish the user's goal. Each step must start with 'SQL:', 'PYTHON:', or 'SYNTHESIZE:'."
    )
def parse_plan(llm_output: str) -> str:
    """
    Parses the LLM output to extract only the plan steps, removing any preamble.
    """
    lines = llm_output.strip().split('\n')
    plan_lines = []
    found_plan = False
    for line in lines:
        # A plan step is typically a numbered list or starts with our keywords
        if re.match(r'^\d+\.\s*', line) or line.strip().startswith(("SQL:", "PYTHON:", "SYNTHESIZE:")):
            found_plan = True
        
        if found_plan:
            plan_lines.append(line)
            
    # If no structured plan was found, return the original output to let the router handle it
    if not plan_lines:
        return llm_output
        
    return "\n".join(plan_lines)



structured_llm = openai_GPT4_llm.with_structured_output(Plan)

DB_SCHEMA = "Error: Schema file could not be loaded."
try:
    # This block executes only once when the server starts.
    with open("tourism_schema.json", 'r') as f:
        schema_data = json.load(f)
    
    schema_string = ""
    # Use .get("tables", []) to safely handle cases where the key might be missing
    for table in schema_data.get("tables", []):
        # FIX: Changed table['name'] to table['table_name']
        schema_string += f"Table Name: `{table['table_name']}`\n"
        schema_string += f"Description: {table['description']}\n"
        schema_string += "Columns:\n"
        
        for col in table.get("columns", []):
            schema_string += f"  - `{col['column_name']}` ({col['data_type']}): {col['description']}\n"
        schema_string += "\n"
        
    DB_SCHEMA = schema_string.strip()
    print("Database schema loaded into memory.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load db_schema.json. Planner will not have context. Error: {e}")
    # Optional: re-raise the exception to stop the server if the schema is critical
    # raise

# Replace your existing planner_prompt with this new, more nuanced version.

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class Principal Analyst and Strategic Planner. Your sole output is a high-level execution plan for a team of agents to answer a user's business question.

Think Managerially: Focus on what each agent should accomplish, not how. Demand excellence from SQL and Python agents.

SQL Steps: Define high-level business questions only. No table/column names or functions.

PYTHON Steps: Specify business logic, calculations, or transformations on retrieved data.

SYNTHESIZE Step (Critical): Always end with SYNTHESIZE:. This step must integrate all previous work into a flawless, user-ready answer. Ensure it fully reflects the quality of SQL and Python outputs. Mistakes here diminish the value of your team.

Tourism Focus: Only create tasks for tourism KPIs; non-tourism requests go directly to SYNTHESIZE.

Database context: {db_schema}

if user input asks for data in arabic , the entire plan should be choreographed to give an answer in arabic, if the user input was in english then the entire answer should be in english

Analyze the user request and produce the execution plan immediately.
""",
        ),
        ("human", "{input}"),
    ]
)

planner = planner_prompt | structured_llm



# --- Now, modify the planner_node function ---
def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    The first node in the workflow. Creates a step-by-step plan to answer
    the user's query, using a high-level schema summary for context.
    """
    print("--- 🧠 Planning... ---")
    # STEP 2: Invoke the planner, now providing BOTH required inputs
    plan_object = planner.invoke({
        "input": state["input"],
        "db_schema": DB_SCHEMA 
    })
    
    plan_steps = plan_object.steps
    
    print(f"Generated Plan: {plan_steps}")
    return {"plan": plan_steps}

# --- Add this new prompt definition ---
custom_sql_tools = [ask_database_expert, execute_sql_and_get_results]
custom_sql_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert MSSQL data analyst agent. Your purpose is to interact with a database to fetch data based on a user's request.

            **Workflow:**
            1.  **Analyze the Request**: Understand what the user is asking for.
            2.  **Consult the Expert**: Your first step MUST be to use the `ask_database_expert` tool. This will give you vital context about table schemas, column meanings, and business rules. THIS TOOL IS FOR CONTEXT AND DATABASE UNDERSTANDING ONLY ,ITS AI SEARCH INDEX WHICH CONTAINS INDEXED JSON ABOUT THE DATABASE CONTENTS 
            3.  **Construct the Query**: Based on the expert's information, write an accurate and efficient MSSQL query.
            4.  **Execute the Query**: Use the `execute_sql_and_get_results` tool to run your query.
            5.  **Return the Result**: The raw output from `execute_sql_and_get_results` will be your final answer. Do not add any conversational text or summaries; the tool's direct output is what's required.

                **Query Quality Requirements:**
            1.  **BE UNIQUE:** If the user asks for a "list of" items (e.g., "list of indicators", "what categories are available?"), you MUST use the `DISTINCT` keyword to avoid returning thousands of duplicate rows. Example: `SELECT DISTINCT MainIndicatorNameEN, IndicatorType FROM ...`
            2.  **EFFICIENT QUERYING**: Always construct your query to filter (NOT BY TOP 10 ) as much as you can to retrieve the smallest dataset restricting your querying to a condition that the retreived data would be used to fully asnwer the user input,THIS IS NOT OPTIONAL U MUST FILTER WHERE APPLICABLE.
            3.  **BE PRECISE:** Never use `SELECT *`. Only select the specific columns needed to answer the user's question.
            4.  **Nulls** : never clean the data of missing values , as null values themseleves are informative and should be reported as is. 
            **MANDATES:**
            - **Always Provide Numerical Context**: Never return a number without its unit or format. This is a non-negotiable rule.
            - When querying metrics from `Tourism_Indicator_Details` (e.g., Actual, Target), you **MUST** also `SELECT` the `UnitEN` and `Format` columns.
            - When querying budget columns from `Tourism_Program_Details` (e.g., AllocatedAmount), you **MUST** use a SQL alias to rename the column with a `_QAR` suffix (e.g., `AllocatedAmount AS AllocatedAmount_QAR`).
            """,
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)



async def sql_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the SQL task with early stopping. If the tool returns an error,
    it updates the state to trigger a retry loop via the router.
    """
    # --- MODIFICATION: Track retry attempts ---
    print("--- 🛢️ Executing SQL Task (Attempt " + str(state.get('sql_retry_count', 0) + 1) + ") ---")
    current_plan_step = state['plan'][0]
    task_description = current_plan_step.split(": ", 1)[1]

    # --- MODIFICATION: Add error context to the prompt on retries ---
    if state.get("sql_error"):
        print(f"Retrying with error context: {state['sql_error']}")
        task_description = (
            f"Your previous attempt to execute a SQL query for the task below failed. "
            f"Analyze the error and write a corrected query.\n\n"
            f"PREVIOUS ERROR: {state['sql_error']}\n\n"
            f"ORIGINAL TASK: {task_description}"
        )

    try:
       
        agent_response = await custom_sql_agent_executor.ainvoke(
            {"original_query": state["input"],
             "full_plan": "\n".join(state["plan"]),
             "input": task_description},
        )
        tool_output = agent_response.get("output")

        # --- MODIFICATION: Handle success and failure cases differently ---

        # SUCCESS CASE: The tool returned a list.
        if isinstance(tool_output, list):
            print(f"Data extraction successful (Headers + First 2 rows): {tool_output[:3]}")
            structured_data = tool_output
            # Create the result object for the state
            new_sql_result = {"task": task_description, "structured_data": structured_data}
            # Return a dictionary that advances the plan and clears any errors
            return {
                "plan": state['plan'][1:],
                "sql_results": [new_sql_result],
                "sql_error": None  # Clear the error on success
            }

        # FAILURE CASE: The tool returned an error string.
        elif isinstance(tool_output, str):
            print(f"--- ERROR: SQL tool returned an error message: {tool_output} ---")
            # Return a dictionary that updates the error state for the router to catch
            return {
                "sql_error": tool_output,
                "sql_retry_count": state.get('sql_retry_count', 0) + 1
            }

        # UNEXPECTED FORMAT CASE: Also treat as a failure to retry.
        else:
            print(f"--- WARN: Unexpected agent output format: {agent_response} ---")
            error_msg = f"Unexpected agent output: {str(agent_response)}"
            return {"sql_error": error_msg, "sql_retry_count": state.get('sql_retry_count', 0) + 1}

    # EXCEPTION CASE: Also treat as a failure to retry.
    except Exception as e:
        print(f"--- ERROR: An exception occurred while invoking the SQL agent: {e} ---")
        traceback.print_exc()
        error_msg = f"An unexpected error occurred in the agent node: {e}"
        return {"sql_error": error_msg, "sql_retry_count": state.get('sql_retry_count', 0) + 1}


# 2c. Python Agent Node
# This node executes the Python analysis tasks.
python_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
     """You are an expert Python data engineer. Your sole mission is to write a clean, efficient Python script to process a pandas DataFrame based on the user's goal.

--- CONTEXT ---
- **User's Goal:** "{task_description}"
- **DataFrame Preview (`df.head()`):**
{df_preview}
- **Available Columns:** {column_names}

--- RULES & REQUIREMENTS ---
1.  **Function-Only Output:** Your output MUST be ONLY the Python code for a single function `def analyze(df: pd.DataFrame) -> pd.DataFrame:`. Do not add any explanation, conversational text, or example usage.
2.  **Return, Don't Print:** This function MUST take a pandas DataFrame as its only argument and **return** the final, transformed DataFrame as its output. The final line of your function should be a `return df` statement.
3.  **Encapsulation:** All your logic (imports, helper functions, etc.) must be contained *inside* the `analyze` function. The only code in the global scope should be the `import` statements at the top.
4.  **Robust Date Handling:** If the task requires parsing dates from a column like 'IntervalValue', your script MUST correctly handle multiple common formats (e.g., 'YYYY-Q#', 'YYYY-MM', 'YYYY'). Create a new sortable 'IntervalDate' column with the results.
5.  **Data Precision:** Round all final numeric values to a maximum of two decimal places.
6.  **Forbidden Libraries:** You MUST NOT import any visualization libraries (e.g., matplotlib, seaborn, plotly) or attempt to generate plots.
7.**Nulls** : never clean the data of missing values , as null values themseleves are informative and should be reported as is. 
"""
),
       ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# In main.py, replace the python_agent_node with this version

def python_agent_node(state: AgentState):
    """
    Generates and executes Python code to analyze a DataFrame.
    """
    print(f"--- 🐍 Executing Python Logic in Node (Attempt {state.get('python_retry_count', 0) + 1}) ---")

    # --- 1. PREPARE DATA & PROMPT INPUTS (Unchanged) ---
    task_description = state['plan'][0]
    structured_data = state['sql_results'][-1]['structured_data']
    failed_code = state.get('python_last_failed_code')
    error_message = state.get('python_error')

    try:
        df = pd.DataFrame(structured_data[1:], columns=structured_data[0])
        df_preview = df.head().to_string()
        column_names = ", ".join(df.columns)
    except Exception as e:
        error = f"Failed to create DataFrame from SQL results: {e}"
        print(f"--- ❌ Python Node Failed --- \n Error Details:\n{error}")
        return {
            "python_error": error,
            "python_retry_count": state.get('python_retry_count', 0) + 1,
        }

    # --- 2. INVOKE THE AGENT TO *GENERATE* THE CODE STRING (Unchanged) ---
    try:
        agent_response = custom_python_agent_executor.invoke({
            "input": state["input"],
            "task_description": task_description,
            "df_preview": df_preview,
            "column_names": column_names,
            "failed_code": failed_code if failed_code else "# No previous script failed.",
            "error": error_message if error_message else "No error on the previous attempt."
        })
        
        if isinstance(agent_response, dict) and 'output' in agent_response:
             python_code_string = agent_response['output']
        else:
             python_code_string = str(agent_response)

        python_code_string = re.sub(r'^```python\n|```$', '', python_code_string, flags=re.MULTILINE).strip()

    except Exception as e:
        error_details = f"Failed to invoke the code-generation agent: {traceback.format_exc()}"
        print(f"--- ❌ Python Agent Invocation Failed ---\n{error_details}")
        return {
            "python_error": str(e),
            "python_retry_count": state.get('python_retry_count', 0) + 1,
        }

    # --- 3. SAFELY EXECUTE THE GENERATED CODE (Corrected Logic) ---
    df_copy = df.copy()

    exec_globals = {
    "__builtins__": {
        # Functions we want to allow from the standard library
        "__import__": safe_import, 
        "print": print, 
        "len": len, 
        "range": range,
        "str": str, 
        "int": int, 
        "float": float, 
        "list": list, 
        "dict": dict, 
        "set": set,
        "round": round, 
        "max": max, 
        "min": min, 
        "sum": sum, 
        "abs": abs,
        "enumerate": enumerate, # <-- ADDED
        "zip": zip, # <-- ADDED
        "isinstance": isinstance, # <-- ADDED
        # Error types to catch
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "NameError": NameError, "AttributeError": AttributeError, "KeyError": KeyError,
    },
    # Modules to make directly available
    "pd": pd, 
    "np": np, 
    "re": re, 
    "statistics": statistics, 
    "collections": collections,
}

    exec_locals = {}
    try:
        print(f"--- Attempting to execute generated code ---\n{python_code_string}\n-----------------------------------------")
        
        exec(python_code_string, exec_globals, exec_locals)

        analyze_func = exec_locals.get('analyze')
        if not callable(analyze_func):
            raise ValueError("The generated Python code did not define a callable function named 'analyze'.")

        final_dataframe = analyze_func(df_copy)

        if not isinstance(final_dataframe, pd.DataFrame):
            raise ValueError("The 'analyze' function did not return a valid pandas DataFrame.")

        print("--- ✅ Python Script Executed Successfully ---")
        
        output_summary = f"The Python script executed successfully. Here is a preview of the resulting data:\n{final_dataframe.head().to_string()}"
        print("--- 📊 Final DataFrame Head ---")
        print(final_dataframe.head().to_string())
        print("---------------------------------")
        
        dataframe_result = final_dataframe.to_dict(orient='records')
        return {
            "plan": state['plan'][1:],
            "analysis_summary": output_summary,
            "final_data": dataframe_result,
            "python_error": None,
            "python_retry_count": 0,
            "python_last_failed_code": None
        }

    except Exception:
        error_details = traceback.format_exc()
        print(f"--- ❌ Python Script Failed ---\nError Details:\n{error_details}")
        return {
            "python_error": error_details,
            "python_retry_count": state.get('python_retry_count', 0) + 1,
            "python_last_failed_code": python_code_string
        }


synthesizer_prompt = PromptTemplate.from_template(
"""You are a senior data analyst and insights communicator. Your primary goal is to provide a clear, direct, and appropriately formatted answer to the user's question based on the final data provided.

First, determine the user's intent: are they asking for a simple fact/list, or do they require analysis/comparison? Then, format your response according to the rules below.

**--- CONTEXT ---**
- **User's Original Question:** {input}
- **Final Data Analysis Summary:** {analysis_summary}
- **Supporting Data Table:** {data_table}

**--- UNIVERSAL RULES (APPLY TO ALL RESPONSES) ---**
1.  **EMPTY DATA RULE (Priority 1):** If the `Supporting Data Table` is empty or contains no meaningful results, you **MUST NOT** invent an answer or interpret the absence of data as zeros. Do not show an empty data table. Instead, respond contextually and directly. For example: "Based on the available data, there were no records found for [topic of the user's question]."
2.  **SCOPE GUARDRAIL (Priority 2):** If the data is NOT empty but the user's question is clearly not about tourism, respond ONLY with: `I can only answer questions related to tourism indicators.`
3.  **Supporting Data(priority 3):** :When creating a summary table for your final report, you MUST include the columns that are most critical for understanding the conclusion.
4. **GREETING** : IF the user input is a simple greeting then just greet him back "Hello! How can I assist you today with your tourism data inquiries?"
5. **Null Data Handling (Priority 4):** If the data contains NULL or missing values, acknowledge this and mention explicitly that data on this or that is missing , analyze logically and report why would this data be missing. 
---
**--- RESPONSE FORMATTING ---**
*Select ONE of the following formats based on the user's question.*

**FORMAT 1: For Direct Questions (Facts, Lists, Counts)**
*Use this format if the user asks "What is...", "List all...", "How many...", or for a simple data retrieval.*

-   Start with a brief, direct introductory sentence.
-   Present the data clearly using a bulleted list or a markdown table.
-   Do not write a formal report with sections like "Executive Summary" or "Conclusion." Keep it concise and to the point.

**FORMAT 2: For Analytical Questions (Analysis, Trends, Comparisons)**
*Use this format if the user asks for an "analysis," "comparison," "summary," "trend," or a "why/how" question.*

-   You **MUST** follow this professional report structure precisely using Markdown.
-   Analyze the data fully to generate your own insightful conclusions.

# Report: [A concise, descriptive title based on the user's question]

## Executive Summary
A brief, one-paragraph summary of the most critical findings and your main conclusion. This should be easily digestible for a busy executive.

## Key Findings
- A bulleted list of the most important facts, trends, or figures you discovered in the data.
- Each point must be a complete, insightful sentence.

## Detailed Analysis
A narrative explanation of the data. Discuss the trends, patterns, or comparisons that support your key findings. This is where you elaborate on the "why" behind the numbers and provide a deeper interpretation of what the data means.

## Supporting Data
The final data table you received, presented in a clean markdown format.

## Conclusion
A final paragraph that summarizes the analysis and provides a concluding thought or implication that directly relates to the user's original question.

---
**Begin your response now, selecting the appropriate format based on the user's question.**
"""
)


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    The final node in the workflow. It synthesizes a report from either the
    Python analysis output or, if that's not present, the direct SQL results.
    """
    MAX_SQL_RETRIES = 10 # This value must match the one in the router

    if state.get("sql_error") and state.get("sql_retry_count", 0) >= MAX_SQL_RETRIES:
        print("--- ✍️ Synthesizing Final Report (SQL Failure) ---")
        error_report = "I'm sorry, but I was unable to retrieve the necessary data from the database after several attempts. This could be due to an issue with the query or the database itself. Please try rephrasing your question."
        return {"final_report": error_report}

    if state.get("python_error"):
        print("--- ✍️ Synthesizing Final Report (Python Failure) ---")
        error_report = "I'm sorry, but I encountered an issue while analyzing the data. The data might have been in an unexpected format. Please try rephrasing your question."
        return {"final_report": error_report}
    
    print("--- ✍️ Synthesizing Final Report ---")

    # Safely get the analysis summary. Provide a default if the python_agent was skipped.
    analysis_summary = state.get("analysis_summary",
                                 "No Python analysis was performed. The data below is the direct result of the SQL query.")

    # Determine the source of the data table. Prioritize Python's output.
    final_data_table = state.get("final_data") # This is already a list of dicts

    # If the python_agent didn't run, process the raw SQL results.
    if final_data_table is None:
        print("INFO: No Python data found. Using raw SQL results for synthesis.")
        sql_results_list = state.get("sql_results", [])
        if sql_results_list:
            # Get the structured_data from the most recent SQL task
            structured_data = sql_results_list[-1].get("structured_data")
            if structured_data and len(structured_data) > 1:
                headers = structured_data[0]
                rows = structured_data[1:]
                # Convert list-of-lists to a list-of-dicts for consistency
                final_data_table = [dict(zip(headers, row)) for row in rows]
            else:
                final_data_table = [{"message": "No data was returned from the SQL query."}]
        else:
            final_data_table = [{"message": "No data is available to report."}]

    # Convert the final data table to a nicely formatted JSON string for the prompt
    # Use default=str to handle potential non-serializable types like dates
    data_table_str = json.dumps(final_data_table, indent=2, default=str)

    prompt = synthesizer_prompt.invoke({
        "input": state["input"],
        "full_plan": "\n".join(state.get("plan", [])),
        "analysis_summary": analysis_summary, # This is now safe
        "data_table": data_table_str
    })

    # Use the streaming LLM for the final output
    response = openai_GPT4_llm.invoke(prompt)

    print(f"Final Report: {response.content}")
    return {"final_report": response.content}

def router(state: AgentState) -> str:
    print("--- 🚦 Routing... ---")
    
    MAX_RETRIES = 10

    # 1. Check for a SQL error. If it exists, either retry or route to the synthesizer on failure.
    if state.get("sql_error"):
        if state.get("sql_retry_count", 0) < MAX_RETRIES:
            print(f"SQL error detected. Routing back to SQL agent for retry.")
            return "sql_agent"
        else:
            print(f"Max SQL retries ({MAX_RETRIES}) reached. Routing to synthesizer for error report.")
            return "synthesizer"

    if state.get("python_error"):
        if state.get("python_retry_count", 0) < MAX_RETRIES:
            print(f"Python error detected. Routing back to Python agent for retry.")
            # We need to increment the retry count in the state before returning
            # NOTE: The python_agent_node must be updated to handle this
            return "python_agent" 
        else:
            print(f"Max Python retries ({MAX_RETRIES}) reached. Routing to synthesizer for error report.")
            return "synthesizer"

    # 3. If no errors, follow the execution plan.
    if not state.get('plan'):
        print("Plan complete. END.")
        return END
        
    next_step = state['plan'][0]
    
    if next_step.startswith("SQL:"):
        print("Next Step: SQL")
        return "sql_agent"
    elif next_step.startswith("PYTHON:"):
        print("Next Step: PYTHON")
        return "python_agent"
    elif next_step.startswith("SYNTHESIZE:"):
        print("Next Step: SYNTHESIZE")
        return "synthesizer"
    else:
        print("Unrecognized step. END.")
        return END
    
sql_agent = create_openai_tools_agent(
    llm=openai_GPT4_llm,
    tools=custom_sql_tools,
    prompt=custom_sql_agent_prompt
)

# 4. Create the Agent Executor, which is what we will invoke
custom_sql_agent_executor = AgentExecutor(
    agent=sql_agent,
    tools=custom_sql_tools,
    verbose=True,
)
print("Custom SQL Agent Executor created with smart tool.")

python_agent = create_openai_tools_agent(
    llm=openai_GPT4_llm, # Using the stronger model for code generation
    tools=[],
    prompt=python_agent_prompt
)

custom_python_agent_executor = AgentExecutor(
    agent=python_agent,
    tools=[],
    verbose=True
)
print("Custom Python Agent Executor created.")


# Now, let's build the graph by wiring together our nodes and router
workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("planner", planner_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("python_agent", python_agent_node)
workflow.add_node("synthesizer", synthesizer_node)

# Define the edges
workflow.set_entry_point("planner")

# The router will decide the path after each SQL or Python step
workflow.add_conditional_edges(
    "planner",
    router, # The router will check the first step of the plan
)
workflow.add_conditional_edges(
    "sql_agent",
    router,
)
workflow.add_conditional_edges(
    "python_agent",
    router
)

# After synthesis, the process ends
workflow.add_edge("synthesizer", END)

# Compile the graph into a runnable object
app_graph = workflow.compile()
print("Multi-agent graph compiled successfully.")
class QueryRequest(BaseModel):
    query: str
# --- 3. Define API Endpoint ---

@app.post("/ask_sql")
async def ask_sql_endpoint(request: QueryRequest):
    query = request.query
    print(f"Received stateless query: {query}")

    async def event_stream():
        # Helper function to send structured messages
        async def send_event(event_type: str, data: dict):
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        log_queue = asyncio.Queue()
        sse_queue = asyncio.Queue()
        
        # Task 1: Streams verbatim logs from the stdout queue to the main SSE queue
        async def log_streamer():
            while True:
                message = await log_queue.get()
                if message is None:
                    break
                await sse_queue.put(await send_event("log", {"message": message}))

        # Task 2: Runs the agent, capturing both stdout and structured events
        async def agent_runner():
            final_answer = ""
            last_state = {}
            try:
                # By wrapping the stream in our context manager, all `verbose=True` prints
                # from the agents will be captured and sent to the `log_queue`.
                with redirect_stdout_to_queue(log_queue):
                    initial_state = {"input": query, "sql_results": []}
                    
                    # Stream the state updates from our new graph
                     # Stream the state updates from our new graph
                    async for chunk in app_graph.astream(initial_state, stream_mode="values"):
                        last_state = chunk
                        
                        # --- MODIFICATION: Send user-friendly status updates ---
                        user_friendly_message = ""
                        if "plan" in last_state and last_state["plan"]:
                            current_step = last_state['plan'][0]
                            if current_step.startswith("SQL:"):
                                user_friendly_message = "Gathering the necessary data..."
                            elif current_step.startswith("PYTHON:"):
                                user_friendly_message = "Analyzing and calculating results..."
                            elif current_step.startswith("SYNTHESIZE:"):
                                user_friendly_message = "Preparing your final answer..."
                            
                            # Only send a message if we have a new one
                            if user_friendly_message:
                                await sse_queue.put(await send_event("status", {"message": user_friendly_message}))

                        elif "final_report" in last_state and last_state["final_report"]:
                             await sse_queue.put(await send_event("status", {"message": "Finalizing report..."}))
                
                # After the agent is done, send the final answer from the last state
                if last_state and "final_report" in last_state:
                    final_answer = last_state["final_report"]
                    await sse_queue.put(await send_event("data", {"answer": final_answer}))
                else:
                    await sse_queue.put(await send_event("error", {"message": "Could not find a final report in the agent's response."}))

            except Exception as e:
                # Log the full, detailed error to the server console for debugging
                print("--- CRITICAL ERROR in agent_runner ---")
                traceback.print_exc()
                
                # Create and send a generic, user-friendly error message to the frontend
                user_friendly_error = "I'm sorry, but an unexpected problem occurred while processing your request. Please try rephrasing your question"
                await sse_queue.put(await send_event("error", {"message": user_friendly_error}))

            finally:
                # Tell both queues we are done
                await log_queue.put(None)
                await sse_queue.put(None)

        # Start the two tasks concurrently
        asyncio.create_task(agent_runner())
        asyncio.create_task(log_streamer())

        # Yield events from the main SSE queue as they arrive
        while True:
            event = await sse_queue.get()
            if event is None:
                break
            yield event
        
        # Finally, send the end event
        yield await send_event("end", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")



########################### TWILIO Whatsapp INTEGRATION ########################################

# ADD THIS NEW FUNCTION IN ITS PLACE

async def run_full_agent_graph(question: str) -> str:
    """
    Runs the entire multi-agent graph for a given question and returns the
    final synthesized report.
    """
    try:
        # Define the initial state for the graph
        initial_state = {"input": question, "sql_results": []}
        
        # Use ainvoke to run the graph to completion and get the final state
        final_state = await app_graph.ainvoke(initial_state)
        
        # Extract the final report from the completed state
        final_report = final_state.get("final_report", "")
        
        print(f"Graph finished for WhatsApp. Final Report: {final_report[:100]}...")
        return final_report or "**Error: The agent did not produce a final report.**"

    except Exception as e:
        print(f"--- Error running full agent graph ---\n{e}")
        traceback.print_exc()
        return f"**An error occurred while processing the request:** {e}"

# ALSO ADD THIS FUNCTION (it was correct before but needs to be re-added)
async def summarize_text_once(answer_md: str) -> str:
    """
    Runs the Summary agent one-shot on the Ask-AI markdown and returns the summary.
    """
    try:
        out = await summary_agent_executor.ainvoke({"input": "", "answer_md": answer_md})
        md = _extract_output_text(out)
        return md or ""
    except Exception as e:
        return f"**Failed to summarize:** {e}"


# background worker do both agents then send excat summary to whatsapp 

# MODIFY THIS FUNCTION

async def _process_and_reply_whatsapp(user: str, text: str):
    """
    Background worker that runs the full agent graph, summarizes the result,
    and sends it back to the user via WhatsApp.
    """
    try:
        # This is the line that changes:
        answer_md = await run_full_agent_graph(text)

        # The rest of the function stays the same
        summary_md = await summarize_text_once(answer_md)
        if not summary_md.strip():
            summary_md = "Sorry, I couldn't produce a summary for the result."

        if _twilio_client:
            _twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                to=user,
                body=summary_md
            )
    except Exception as e:
        print(f"--- WhatsApp processing failed ---\n{e}")
        traceback.print_exc()
        if _twilio_client:
            _twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                to=user,
                body="Sorry, a critical error occurred while handling your request."
            )

#-----IMP ------
# whatsapp webhook validate , queue background job , ACK immediately 

@app.post("/twilio/wh")
async def twilio_webhook(request: Request, background_tasks: BackgroundTasks):
    if not (_twilio_client and _validator):
        raise HTTPException(status_code=503, detail="Twilio not configured")

    # Twilio posts form-encoded parameters (From, Body, etc.)
    form = dict(await request.form())

    # Normalize URL for signature behind Azure proxy
    raw_url = str(request.url)
    if PUBLIC_BASE_URL:
        parts = urlsplit(raw_url)
        pub   = urlsplit(PUBLIC_BASE_URL)
        raw_url = urlunsplit((pub.scheme or "https", pub.netloc, parts.path, parts.query, ""))

    sig = request.headers.get("X-Twilio-Signature", "")
    if not _validator.validate(raw_url, form, sig):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    user = (form.get("From") or "").strip()  # e.g., 'whatsapp:+<user>'
    text = (form.get("Body") or "").strip()
    if not user or not text:
        return ""  # 200 OK; nothing to do

    # Kick off Ask-AI → Summary in the background; ACK immediately to avoid timeout
    background_tasks.add_task(_process_and_reply_whatsapp, user, text)
    return ""  # immediate 200 OK



#####################################------##################################################
# --- 4. Mount Static Frontend Files (as the last step before running) ---
# This serves your index.html from the 'frontend' directory at the root URL.
# It's placed here so that the API route above is registered first.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# --- 5. Run the server (when executing the script directly) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)