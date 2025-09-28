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
import datetime
from typing import Literal
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
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional
from decimal import Decimal
from scipy import stats
from sklearn import preprocessing
import io
import sys
import pandas as pd
import numpy as np
import collections
import datetime
import math
import statistics
import json
import re



# --- 1. Initialization (Runs once on server start) ---
load_dotenv()


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
    report_type: str | None
    enhanced_input: str | None
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
        streaming=False,
        temperature=0.0
    )
except Exception as e:
    print(f"Error initializing Azure OpenAI clients: {e}")
    raise

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
def explore_database_schema(query: str) -> str | List[any]:
    """
    Executes a read-only, exploratory SQL query to understand database contents.
    Use this to discover value distributions, date ranges, or sample data.
    Example: "SELECT DISTINCT MainIndicatorNameEN FROM Tourism_Indicator_Details"
    Example: "SELECT COUNT(*) FROM Tourism_Indicator_Details GROUP BY IndicatorType"
    """
    print(f"--- 探索 Exploring with schema tool (Query: {query}) ---")
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            headers = list(result.keys())
            rows = [list(row) for row in result.fetchall()]
            return [headers] + rows
    except Exception as e:
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
    'tabulate', 'prettytable', 'texttable','typing',

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
    report_type: Literal["direct_answer", "full_report"] = Field(
        description="The type of report to generate. Use 'direct_answer' for simple requests (what, list, how many). Use 'full_report' for analytical requests (analysis, trends, comparisons)."
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

prompt_enhancer_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a prompt optimization expert. Your task is to refine a user's raw query into a detailed, unambiguous prompt for a data analysis system.
              **CONVERSATIONAL**: if the user input is conversational then forward it as is. 
            **Context:**
            - The analysis system has access to a database about a companies distribution data .
            
            **Refinement Checklist:**
            1.  **TIMEFRAMES** : NEVER INSTRUCT TO RETREIVE LATEST DATA FOR ANY QUESTION . 
            2.  **Clarify Ambiguity:** If a term is vague (e.g., "visitors"), mention the specific metric available (e.g., "Number of International Visitors").
            3.  **Preserve Intent:** Do not change the user's question. Only reformulate to adhere by prompt enigneering best practices , as your reformulation will be passed on to agents.


            **CRITICAL RULE:** You **MUST NOT** invent new analytical requirements or dimensions. For example, if the user asks for "visitor numbers," do NOT add "broken down by nationality" unless they specifically asked for it. Your job is to clarify, not to expand the scope of the question.
            
            Produce ONLY the refined prompt as a single string, with no preamble or explanation.
            """,
        ),
        ("human", "Here is the raw query: {input}"),
    ]
)

def prompt_enhancer_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes and refines the user's initial input into a more detailed prompt.
    """
    print("--- 🔬 Enhancing User Prompt... ---")
    user_input = state["input"]
    
    # Using your existing GPT-4 model for this task
    enhancer_chain = prompt_enhancer_template | openai_GPT4_llm | StrOutputParser()
    
    enhanced_prompt = enhancer_chain.invoke({
        "input": user_input,
        "current_date": datetime.datetime.now().strftime("%Y-%m-%d")
    })
    
    print(f"Original Prompt: {user_input}")
    print(f"Enhanced Prompt: {enhanced_prompt}")
    
    return {"enhanced_input": enhanced_prompt}

structured_llm = openai_GPT4_llm.with_structured_output(Plan)

DB_SCHEMA = "Error: Schema file could not be loaded."
try:
    # This block executes only once when the server starts.
    # FIX: Corrected the filename to match the one you uploaded.
    with open("table_infoDB_v2.json", 'r') as f:
        schema_data = json.load(f)
    
    schema_string = ""
    # Use .get("tables", []) to safely handle cases where the key might be missing
    for table in schema_data.get("tables", []):
        # FIX 1: The key for the table name in your JSON is 'name', not 'table_name'.
        table_name = table.get('name')
        if not table_name:
            continue # Skip if the table has no name

        schema_string += f"Table Name: `{table_name}`\n"
        schema_string += f"Description: {table.get('description', 'N/A')}\n"
        schema_string += "Columns:\n"
        
        for col in table.get("columns", []):
            # FIX 2: Handle both 'column_name' and 'name' keys for column names.
            column_name = col.get('column_name') or col.get('name')
            if not column_name:
                continue # Skip if the column has no name

            data_type = col.get('data_type', 'unknown')
            description = col.get('description', 'N/A')
            schema_string += f"  - `{column_name}` ({data_type}): {description}\n"
        schema_string += "\n"
        
    DB_SCHEMA = schema_string.strip()
    print("Database schema loaded into memory.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: The file 'table_infoDB_v2.json' was not found.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load or parse the schema file. Planner will not have context. Error: {e}")
    # Optional: re-raise the exception to stop the server if the schema is critical
    # raise

# Replace your existing planner_prompt with this new, more nuanced version.

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class Principal Analyst and Strategic Planner. Your sole output is a high-level execution plan for a team of agents to answer a user's business question.
                Think Managerially: Focus on what each agent should accomplish, not how. Demand excellence from SQL and Python agents.

        **Your First Responsibility:**
            Analyze the user's request to determine the required report format.
            - Choose 'direct_answer' for simple, fact-based questions (e.g., "what is...", "list all...", "how many...").
            - Choose 'full_report' for analytical questions that require interpretation (e.g., "analysis of...", "compare...", "what are the trends...").

            **CRITICAL RULES:**
            1.  **Look Ahead:** Your primary goal is to create a plan where each step enables the next. The `SQL` step's description **MUST** be comprehensive enough to provide all the raw data needed for the `PYTHON` step.
            2.  **Comprehensive SQL Goal:** The `SQL:` step should not just ask for the final aggregated answer. It must instruct the agent to retrieve a detailed, denormalized dataset. If the `PYTHON` step needs to rank items by quantity, the `SQL` step **MUST** include retrieving item names and quantities in its goal.
                - **Bad Plan:**
                    - `SQL: Retrieve total revenue per salesperson for each item category.`
                    - `PYTHON: Find the top salesperson and then find their best-selling item.`
                - **Good Plan:**
                    - `SQL: Retrieve a detailed dataset of all orders, joining to get salesperson names, item category descriptions, individual item descriptions, and the quantity sold for each item.`
                    - `PYTHON: From the detailed dataset, first find the top salesperson by total revenue in each category. Then, for that salesperson, find the best-selling item by quantity.`
            3.   **CONVERSATIONAL PROMPTS:** If the user input is conversational (e.g., "Hello", "How are you?"), your plan should be to respond conversationally without invoking SQL or Python agents.
            4.   **LITERAL INTERPRETATION (NO ASSUMPTIONS):** Your primary duty is to create a plan that answers the user's **exact, literal question**.
                 - **IF NO TIME IS SPECIFIED, GET ALL TIME:** If a user does not specify a date, year, or time-based filter (like "latest" or "in 2023"), the plan **MUST** be to retrieve all available data for that metric across all time periods.
                 - **Example:**
                     - User: "What is the sector contribution to GDP?"
                     - **BAD PLAN:** "SQL: Retrieve the *most recent* sector contribution to GDP."
                     - **GOOD PLAN:** "SQL: Retrieve all records for sector contribution to GDP."
            5.   **High-Level Goals Only:** Each step must be a high-level objective. Define *what* to achieve, not *how* to do it. The specialist agents will determine the specific implementation.
            6.   **SQL Steps: Define high-level business questions only, you should give the agent instruction on extracting on all the columns needed that are needed downstream 
            7.   **PYTHON Steps: Specify business logic, calculations, or transformations on retrieved data, even for simple tasks , you should pass a python steps so he sorts it and format it correctly.
            8.   **SYNTHESIZE Step (Critical): Always end with SYNTHESIZE:. This step must integrate all previous work into a flawless, user-ready answer. Ensure it fully reflects the quality of SQL and Python outputs. Mistakes here diminish the value of your team.
            9.   **DATA Focus: Only create tasks for The data in hand;  for database context to understand if the question can be answered by the data available : {db_schema} non-related requests are sent directly to the synthesizer stating its " out of scope". 
            10.   **if user input asks for data in arabic , the entire plan should be choreographed to give an answer in arabic, if the user input was in english then the entire answer should be in english
            
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
    prompt_to_plan = state.get("enhanced_input") or state["input"] 
    plan_object = planner.invoke({
        "input": prompt_to_plan,
        "db_schema": DB_SCHEMA 
    })
    
    plan_steps = plan_object.steps
    report_type = plan_object.report_type
    print(f"Generated Plan: {plan_steps}")
    print(f"Determined Report Type: {report_type}")
    return {"plan": plan_steps, "report_type": report_type}

# --- Add this new prompt definition ---
custom_sql_tools = [ask_database_expert, explore_database_schema, execute_sql_and_get_results]
custom_sql_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert MSSQL data analyst agent. Your purpose is to fetch data from a database following a strict workflow.

            **--- Core Workflow: EXPLORE then EXECUTE ---**

            **Phase 1: EXPLORE**
            Your first priority is to gather all necessary information before attempting to fetch the final data.
            1.  Use the `ask_database_expert` tool to understand table schemas, column meanings, and relationships.
            2.  If you need to filter by a specific name (e.g., an indicator name), you MUST verify the exact name first. Use the `explore_database_schema` tool with a `SELECT DISTINCT...` query to get a list of correct names.

            **Phase 2: EXECUTE**
            Once, and only once, you are certain you have all the correct information (table, columns, exact filter names):
            1.  **MANDATORY ID ENRICHMENT**
                 - For **every** query that returns an ID, you **MUST** immediately perform a subsequent query to fetch its corresponding name or description.
                 - ALWAYS try your best to filter with a where clause or preaggregate to limit the number of rows returned.
            2.  Construct the complete, final query to answer the user's request, following all quality rules below.
            3.  Execute this query using the `execute_sql_and_get_results` tool.
            4.  **This is your final action.** This tool will terminate your process and return the data directly.

            **--- Tool Guide ---**
            - `ask_database_expert`: Use FIRST for schema/context, this is index data that is static about the database (metadata,joins, rules on how to handle the database). 
            - `explore_database_schema`: Use for iterative discovery (e.g., `SELECT DISTINCT`). Results are returned to you for further thought.
            - `execute_sql_and_get_results`: Use LAST for final data extraction. This action is final.

            **--- Query Quality Requirements & Mandates ---**
            1.  **DATE FILTERING** **THIS IS NON NEGOTIABLE** NEVER USE MAX() FUNCTION TO CAPTURE LATEST DATES  , ALWAYS RETRIEVE ALL DATA , only filter by a specific date when the user and planner explicitly requests it.   
            2.  **Nulls**: Never clean the data of missing values, as null values themselves are informative and should be reported as is.
            3. **OUTPUT** : output should always come from the execute_sql_and_get_results tool, never add conversational bits in your response , only forward the data as is , the synthesis agent will take care of the final answer formatting.
            """,
        ),
        ("human", "Here is the full plan:\n{full_plan}\n\nExecute the SQL portion of this plan now."),
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
             "input": task_description,
             "current_date": datetime.date.today().isoformat()}
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
    **Step A: Numerical Rounding**
    - Before formatting, ensure all key numeric columns  are rounded to a maximum of two decimal places. Preserve integers (e.g., `61.0` should become `61`).
1.  **CRITICAL DATA TYPE HANDLING:** If you encounter a `TypeError` involving 'decimal.Decimal', you MUST explicitly convert the Decimal column to a float *before* performing any calculations. For example: `df['column_name'] = df['column_name'].astype(float)`. This is a mandatory first step in your script if that error occurs.
2.  **Function-Only Output:** Your output MUST be ONLY the Python code for a single function `def analyze(df: pd.DataFrame) -> pd.DataFrame:`. Do not add any explanation, conversational text, or example usage.
3.  **Return, Don't Print:** This function MUST take a pandas DataFrame as its only argument and **return** the final, transformed DataFrame as its output. The final line of your function should be a `return df` statement.
4.  **Encapsulation:** All your logic (imports, helper functions, etc.) must be contained *inside* the `analyze` function. The only code in the global scope should be the `import` statements at the top.
5.  **Forbidden Libraries:** You MUST NOT import any visualization libraries (e.g., matplotlib, seaborn, plotly) or attempt to generate plots.
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
        # --- Core Data Types ---
        "dict": dict, "list": list, "set": set, "str": str, "int": int,
        "float": float, "tuple": tuple, "bool": bool,

        # --- Numeric Operations ---
        "abs": abs, "max": max, "min": min, "sum": sum, "round": round, "pow": pow,

        # --- Iteration & Data Structures ---
        "all": all, "any": any, "enumerate": enumerate, "len": len, "range": range,
        "reversed": reversed, "sorted": sorted, "zip": zip,

        # --- Type Checking & Utility ---
        "isinstance": isinstance, "type": type, "print": print,
        "__import__": safe_import,

        # --- Error Types ---
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "NameError": NameError, "AttributeError": AttributeError, "KeyError": KeyError,

        # --- ADDED FOR ADVANCED TYPING & DATA HANDLING ---
        "List": List, "Dict": Dict, "Any": Any, "Optional": Optional,
        "Decimal": Decimal,
    },
    # --- Allowed Modules ---
    "pd": pd, "np": np, "re": re, "statistics": statistics,
    "collections": collections, "datetime": datetime, "math": math, "json": json,

    # --- ADDED FOR ADVANCED ANALYSIS (SPECIFIC FUNCTIONS) ---
    # Scipy Stats
    "pearsonr": stats.pearsonr,
    "ttest_ind": stats.ttest_ind,
    "chi2_contingency": stats.chi2_contingency,
    
    # Scikit-learn Preprocessing
    "MinMaxScaler": preprocessing.MinMaxScaler,
    "StandardScaler": preprocessing.StandardScaler,
    "OneHotEncoder": preprocessing.OneHotEncoder,
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

You have been instructed to generate a '{report_type}' style response. You MUST follow the formatting rules for that type precisely and adhere to all universal rules.

**--- CONTEXT ---**
- **User's Original Question:** {input}
- **Final Data Analysis Summary:** {analysis_summary}
- **Supporting Data Table:** {data_table}
- **Current Year Context:** We are in 2025. Be mindful of this when interpreting data from future years (which may represent targets or forecasts).

**--- UNIVERSAL RULES (APPLY TO ALL RESPONSES) ---**
1.  **CONVERSATIONAL PROMPTS:** If the user input is conversational (e.g., "Hello", "How are you?"), your plan should be to respond conversationally
2.  **EMPTY DATA RULE (Priority 1):** If the `Supporting Data Table` is empty or contains no meaningful results, you **MUST NOT** invent an answer. Do not show an empty data table. Instead, respond contextually: "Based on the available data, there were no records found for [topic of the user's question]."
3.  **SCOPE GUARDRAIL (Priority 2):** If the user's question is flagged by out of scope in the planner's plan, respond ONLY with: `i do not have data to answer this question` 
4.  **SUPPORTING DATA (Priority 3):** When creating a summary table for your final report, you MUST include the columns that are most critical for understanding the conclusion.
5.  **GREETING (Priority 4):** If the user input is a simple greeting, greet them back with: "Hello! I'm here to help with any questions about your data analysis needs."
6.  **NULL DATA HANDLING (Priority 5):** If the data contains NULL or missing values, acknowledge this in your summary. Do not omit this detail, but do not include them in the supporting data table unless specifically relevant.
7.  **DECIMALS (Priority 6):** Do not omit decimal places. The data you receive is already rounded correctly; use the values as-is.
8.  **LANGUAGE:** If the user's question is in Arabic, respond in Arabic. If it's in English, respond in English.
9. ** UNIT:** money currency should be presented in QAR (Qatari Riyal) if the user input was in english , if the user input was in arabic then the values should be presented in ريال قطري 
---
**--- RESPONSE FORMATTING ---**
*Select the format below that matches the instructed '{report_type}'.*

**FORMAT 1: Use this if report_type is 'direct_answer'**
*Use for facts, lists, or counts.*
-   Start with a brief, direct introductory sentence.
-   Present the data clearly using a bulleted list or a markdown table.
-   Do not write a formal report. Add a small summary of findings only if there is one; otherwise, the table and introduction are sufficient.

**FORMAT 2: Use this if report_type is 'full_report'**
*Use for analysis, trends, or comparisons.*
-   You **MUST** follow this professional report structure precisely using Markdown.
-   Analyze the data fully to generate your own insightful conclusions.

# Report: [A concise, descriptive title based on the user's question]

## Executive Summary
A brief, one-paragraph summary of the most critical findings and your main conclusion.

## Key Findings
- A bulleted list of the most important facts, trends, or figures you discovered.
- Each point must be a complete, insightful sentence.

## Detailed Analysis
A narrative explanation of the data. Discuss the trends, patterns, or comparisons that support your key findings.

## Supporting Data
The final data table you received, presented in a clean markdown format.

## Conclusion
A final paragraph that summarizes the analysis and provides a concluding thought or implication.
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
        "data_table": data_table_str,
        "report_type": state.get("report_type", "direct_answer")
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
workflow.add_node("prompt_enhancer", prompt_enhancer_node)
workflow.add_node("planner", planner_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("python_agent", python_agent_node)
workflow.add_node("synthesizer", synthesizer_node)

# Define the edges
workflow.set_entry_point("prompt_enhancer")
workflow.add_edge("prompt_enhancer", "planner")
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