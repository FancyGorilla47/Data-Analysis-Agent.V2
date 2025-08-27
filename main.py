import os
import urllib.parse
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import traceback
import ast
import asyncio
from langchain_community.agent_toolkits.sql.base import create_sql_agent
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
    python_notebook: Annotated[List[str], operator.add]
    python_error: str | None  
    final_data: List[Dict[str, Any]] | None
    final_report: str

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
# Check essential environment variables
required_vars = [
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
    "AZURE_SQL_CONNECTION_STRING",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT"
]
for var in required_vars:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# 1a) Azure OpenAI clients
try:
    openai_agent_llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        streaming=False,
        temperature=0
    )
    print("Azure OpenAI Agent LLM (for tool calling) initialized.")

    openai_refiner_llm = AzureChatOpenAI(
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        streaming=True,
        temperature=0
    )
    print("Azure OpenAI Refiner LLM initialized.")
except Exception as e:
    print(f"Error initializing Azure OpenAI clients: {e}")
    raise

from summary import create_summary_agent_and_router, _extract_output_text
summary_agent_executor, summary_router = create_summary_agent_and_router(openai_refiner_llm)
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
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
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

@tool
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
    sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=openai_agent_llm)
    print("SQL Database Toolkit initialized.")
except Exception as e:
    print(f"Error initializing SQLDatabase or Toolkit: {e}")
    raise


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    A custom import function that only allows imports from a predefined list.
    """
    # Define the list of modules the agent is allowed to import
    ALLOWED_IMPORTS = {
    # 🔢 Core Data Libraries
    'pandas', 'numpy',

    # 📈 Stats, Math & Aggregation
    'statistics', 'math', 'scipy',

    # 📋 Tabular/Pretty Formatting
    'tabulate', 'prettytable', 'texttable',

    # 📄 Output Formatting & Enhancement
    'json', 'yaml', 're', 'html', 'csv',

    # 🗓️ Date/Time Handling
    'datetime', 'dateutil',

    # 🧠 Lightweight NLP & Strings
    'collections', 'itertools', 'string', 'difflib',

    # 📦 Data Utilities
    'typing', 'functools', 'operator', 'copy',

    # 🧪 Basic Data Validation
    'pydantic', 'cerberus',


    # 🔐 Safe Eval or AST Parsing
    'ast', 'io', 'traceback', 'contextlib'
}

    if name in ALLOWED_IMPORTS:
        # If the module is on the allowlist, perform the actual import
        return __import__(name, globals, locals, fromlist, level)
    
    # If the module is not on the list, raise an error
    raise ImportError(f"Import of module '{name}' is not allowed.")

@tool
def python_code_interpreter(code: str, df: pd.DataFrame) -> str:
    """
    Executes a Python script in a sandboxed environment to perform advanced data analysis.

    **Instructions for the Agent:**
    1. A pandas DataFrame is already available in a variable named `df`.
    2. Your script SHOULD use this `df` variable directly for analysis.
    3. Your script MUST `print()` its final, user-facing result to standard output.
    
    **Allowed Imports:**
    You can import any of the following libraries:
    - pandas, numpy, statistics, math, scipy, tabulate, prettytable, texttable,
    - json, yaml, re, html, csv, datetime, dateutil, collections, itertools,
    - string, difflib, typing, functools, operator, copy, pydantic, cerberus,
    - ast, io, traceback, contextlib
    
    Any attempt to import a library not on this list will fail.
    """
    # Redirect stdout to capture the output of the print statement
    output_buffer = io.StringIO()
    sys.stdout = output_buffer

    # Create the sandboxed execution environment
    exec_globals = {
        "__builtins__": {
            "__import__": safe_import,
            "math": __import__('math'),
            "statistics": __import__('statistics'),
            "tabulate": __import__('tabulate'),
            "print": print, "len": len, "sum": sum,
            "map": map, "filter": filter, "zip": zip, "range": range,
            "isinstance": isinstance, "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "max": max, "min": min, "round": round, "sorted": sorted,
            "any": any, "all": all, "abs": abs, "enumerate": enumerate, "bool": bool, "getattr": getattr,
            "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "Exception": Exception, "SyntaxError": SyntaxError
        },
        "pd": pd,
        "np": np
    }
    
    exec_locals = {"df": df}
    final_dataframe = None

    try:
        exec(code, exec_globals, exec_locals)
        # After execution, find the last created DataFrame in the local scope
        for var in reversed(exec_locals.values()):
            if isinstance(var, pd.DataFrame):
                final_dataframe = var
                break
    except Exception as e:
        sys.stdout = sys.__stdout__
        raise e
    finally:
        sys.stdout = sys.__stdout__

    # Convert the final DataFrame to a list of dictionaries for serialization
    dataframe_result = final_dataframe.to_dict(orient='records') if final_dataframe is not None else None

    return {
        "stdout": output_buffer.getvalue(),
        "dataframe": dataframe_result
    }

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



structured_llm = openai_agent_llm.with_structured_output(Plan)
# --- Replace your existing planner_prompt with this ---
def load_schema_from_json(file_path: str = "db_schema.json") -> str:
    """Reads a JSON schema file and formats it into a readable string for the LLM."""
    try:
        with open(file_path, 'r') as f:
            schema_data = json.load(f)
        
        schema_string = ""
        for table in schema_data.get("tables", []):
            schema_string += f"Table Name: `{table['name']}`\n"
            schema_string += f"Description: {table['description']}\n"
            schema_string += "Columns:\n"
            for col in table.get("columns", []):
                schema_string += f"  - `{col['name']}` ({col['type']}): {col['description']}\n"
            schema_string += "\n"
        return schema_string.strip()
    except FileNotFoundError:
        return "Error: Schema file not found."
    except json.JSONDecodeError:
        return "Error: Could not decode the JSON schema file."
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class Principal Analyst and Strategic Planner. 
Your primary role is to produce an exhaustive, step-by-step execution plan for a team of junior agents. 
You **never execute SQL or Python yourself** — you only instruct other agents precisely on what to pull, calculate, or assemble.  
You must always avoid requesting massive raw data dumps. Instead, focus on **aggregation, filtering, and only the relevant fields**.

=====================================================
STAGE 1: DECONSTRUCT & EXPAND THE REQUEST
=====================================================
Perform a deep analysis of the user's query. Explicitly identify:
1. **Core Objective** - What is the fundamental business question?  
2. **Explicit Requirements** - Which pieces of information or metrics are directly requested?  
3. **Implicit Needs & Context** - What additional insights (comparisons, trends, root causes) will make the analysis stronger?  
4. **Define Ambiguity** - Pin down vague terms (e.g., “poor performance,” “success,” “at risk”) with measurable definitions.
5.**Gain Context** - use this json file to gain context over the entire database Schema and Metadata to gain full context of what the database contains.

----------------------------
{db_schema}
---------------------------- 


=====================================================
STAGE 2: FORMULATE A HIGH-LEVEL STRATEGY
=====================================================
- Translate the analysis into a clear, high-level strategy for extracting and analyzing data.  
- IMPORTANT: NEVER request entire tables or unfiltered data dumps.  
-  design SQL instructions around aggregation WHERE APPLICABLE DO NOT AGGREGATE IF IT DOESNT MAKE SENSE TO THE ANSWER (e.g., averages, counts, sums), filtering (e.g., specific date ranges, sectors, or thresholds), or summarization .  
- If multiple queries are needed, break them down logically.
-
=====================================================
STAGE 3: CREATE THE STEP-BY-STEP EXECUTION PLAN
=====================================================
Write a precise sequence of instructions for junior agents.  
Each step MUST begin with one of the following commands:

- `SQL:` [Use to instruct which *specific, aggregated, or filtered data* should be retrieved. 
   DO NOT request raw tables or row-level dumps.  
   Good Examples:  
   • "SQL: Retrieve the total allocated budget and total remaining budget per project sector for the last 12 months." 
   Bad Example (NOT ALLOWED):  
   • "SQL: Retrieve all projects in the user's sector, including their current progress, target metrics, and deadlines."  
- `PYTHON:` [Use for calculations, transformations, or filtering. Example: "PYTHON: Compute OverrunPercentage = (BudgetUsed / AllocatedBudget). Filter projects with OverrunPercentage > 15%."]  
- `SYNTHESIZE:` [Always the final step. Instruct to combine all processed outputs into a clear, coherent answer.]

Execution Plan Rules:  
- The plan must be step-by-step, with no ambiguity.  
- SQL instructions must always be scoped (aggregated, filtered, or limited).  
- The **very last step must always be**:  
  `SYNTHESIZE:`  

=====================================================
END OF INSTRUCTIONS
=====================================================
""",
        ),
        ("human", "{input}"),
    ]
)

planner = planner_prompt | structured_llm
# --- Add this block right after the planner_prompt definition ---

# --- Now, modify the planner_node function ---
def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    The first node in the workflow. Creates a step-by-step plan to answer
    the user's query, using a high-level schema summary for context.
    """
    print("--- 🧠 Planning... ---")
    db_schema = load_schema_from_json("db_schema.json")
    # STEP 2: Invoke the planner, now providing BOTH required inputs
    plan_object = planner.invoke({
        "input": state["input"],
        "db_schema": db_schema
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
            2.  **Consult the Expert**: Your first step MUST be to use the `ask_database_expert` tool. This will give you vital context about table schemas, column meanings, and business rules. Ask a clear, direct question to this tool.
            3.  **Construct the Query**: Based on the expert's information, write an accurate and efficient MSSQL query.
            4.  **Execute the Query**: Use the `execute_sql_and_get_results` tool to run your query.
            5.  **Return the Result**: The raw output from `execute_sql_and_get_results` will be your final answer. Do not add any conversational text or summaries; the tool's direct output is what's required.

                **Query Quality Requirements:**
            1.  **BE UNIQUE:** If the user asks for a "list of" items (e.g., "list of indicators", "what categories are available?"), you MUST use the `DISTINCT` keyword to avoid returning thousands of duplicate rows. Example: `SELECT DISTINCT MainIndicatorNameEN, IndicatorType FROM ...`
            2.  **BE PRECISE:** Never use `SELECT *`. Only select the specific columns needed to answer the user's question.

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

# 2b. SQL Agent Node
# This node will execute the SQL tasks from the plan. It will use the `sql_agent_executor` which we will create later.
# 2b. SQL Agent Node (Custom Agent Version)
def sql_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes a SQL task using a custom agent. This node is designed to be highly
    resilient by robustly extracting the structured data from the agent's output,
    regardless of whether it's in the intermediate steps, the final output as a
    list, or the final output as a string.
    """
    print("--- 🛢️ Executing SQL Task with Custom Agent ---")
    current_plan_step = state['plan'][0]
    task_description = current_plan_step.split(": ", 1)[1]

    # Invoke our custom agent
    result = custom_sql_agent_executor.invoke({"input": task_description})
    structured_data = []

    # --- New, Simplified Extraction Logic ---

    # 1. First, check if the output IS the list we need. This is the most common success case.
    if isinstance(result.get("output"), list):
        print("SUCCESS: Extracted data directly from 'output' field.")
        structured_data = result["output"]
    
    # 2. If not, check the intermediate_steps, the other potential success case.
    elif result.get("intermediate_steps"):
        # Check the last step's observation for a list
        last_step_obs = result["intermediate_steps"][-1][1]
        if isinstance(last_step_obs, list):
            print("SUCCESS: Extracted data from intermediate_steps.")
            structured_data = last_step_obs

    # 3. If neither of the above, check if the output is a string that needs parsing.
    elif isinstance(result.get("output"), str):
        print("INFO: Output is a string. Attempting to parse.")
        try:
            output_str = result["output"]
            match = re.search(r"(\[\[.*\]\])", output_str, re.DOTALL)
            if match:
                clean_str = match.group(1)
                # --- THIS IS THE FIX ---
                # Replace any database 'null' with Python's 'None' before parsing
                sanitized_str = clean_str.replace('null', 'None')
                structured_data = ast.literal_eval(sanitized_str)
                # --- END OF FIX ---
                print("SUCCESS: Parsed structured data from string output.")
            else:
                 print("WARN: No list-like pattern found in string output.")
        except (ValueError, SyntaxError) as e:
            print(f"WARN: Failed to parse string output: {e}")

    # --- Final Check & Error Handling ---
    if structured_data:
        print(f"Data extraction successful (Headers + First 2 rows): {structured_data[:3]}")
    else:
        print(f"CRITICAL ERROR: Could not extract structured data. Final agent output was: {result.get('output')}")
        structured_data = [["data_extraction_failed"], [f"Output: {result.get('output')}"]]

    new_sql_result = {"task": task_description, "structured_data": structured_data}
    return {"plan": state['plan'][1:], "sql_results": [new_sql_result]}

# 2c. Python Agent Node
# This node executes the Python analysis tasks.
python_agent_prompt = PromptTemplate.from_template(
    """You are an expert Python data analyst. Your ONLY task is to write a Python script that uses a pre-loaded pandas DataFrame named `df`.

**CRITICAL CONTEXT: The DataFrame (`df`) Schema**
The DataFrame `df` is already loaded. Here are its first few rows:
{df_preview}


**RULES:**
1.  You **MUST ONLY USE** the column names provided above: **{column_names}**.
2.  Do NOT hallucinate or assume other column names exist.
3.  Do NOT write code to create the DataFrame; it is already in memory.
4.  Your script **MUST** `print()` its final, user-facing result to standard output.
5. NEVER drop columns containing metadata like units ,format ,counts or intervals ,THESE ARE ESSENTIAL to give the user the complete answer 

---
**User's Goal:**
{original_query}

**Current Task from Plan:**
"{task_description}"

{error}

**Python Code:**
"""
)

def python_agent_node(state: AgentState) -> Dict[str, Any]:
    print("--- 🐍 Executing Python Analysis ---")
    current_plan_step = state['plan'][0]
    task_description = current_plan_step.split(": ", 1)[1]

    # --- DataFrame Creation & Preview Generation ---
    try:
        # Get the structured data from the previous step
        structured_data = state['sql_results'][0]['structured_data']
        if not structured_data or len(structured_data) < 2:
            raise ValueError("SQL data is empty or missing headers.")

        # Programmatically create the DataFrame
        headers = structured_data[0]
        rows = structured_data[1:]
        df = pd.DataFrame(rows, columns=headers)
        
        # Generate a clean string preview of the DataFrame's head
        df_preview = df.head().to_string()
        column_names = ", ".join(df.columns)
    except Exception as e:
        print(f"--- DataFrame Creation Failed ---\n{e}")
        # If we can't even create the DataFrame, we must return an error
        return {"python_error": f"Failed to create DataFrame: {e}"}


    error_message = state.get('python_error')
    error = f"You previously wrote a script that failed with this error:\n{error_message}\nPlease correct the script." if error_message else ""

    # Generate the Python code with the new, richer context
    prompt = python_agent_prompt.invoke({
        "original_query": state["input"],
        "full_plan": "\n".join(state["plan"]),
        "task_description": task_description,
        "notebook_history": "\n".join(state.get("python_notebook", [])),
        "df_preview": df_preview,
        "column_names": column_names,
        "error": error
    })
    response = openai_agent_llm.invoke(prompt)
    raw_code = response.content

    # Extract the code from markdown blocks
    match = re.search(r"```python\n(.*?)```", raw_code, re.DOTALL)
    code_to_execute = match.group(1).strip() if match else raw_code.strip()
    print(f"Cleaned Python Code to Execute:\n{code_to_execute}")

    # Try to execute the code
    try:
        # The tool now returns a dictionary
        tool_result = python_code_interpreter.invoke({
            "code": code_to_execute,
            "df": df
        })

        analysis_result = tool_result["stdout"]
        final_data_table = tool_result["dataframe"]

        print(f"Python Analysis Result: {analysis_result}")
        # On success, clear any previous error and return BOTH the summary and the data
        return {
            "plan": state['plan'][1:],
            "python_notebook": [analysis_result],
            "analysis_summary": analysis_result,
            "final_data": final_data_table,  # <-- SAVE THE DATA
            "python_error": None
        }
    except Exception as e:
        print(f"--- Python Execution Failed ---\n{e}")
        # On failure, return the error so the agent can retry
        return {"python_error": str(e)}

synthesizer_prompt = PromptTemplate.from_template(
"""You are an expert data analyst writing a final report. Your goal is to provide a clear, insightful summary backed by the specific data that was found.

**1. Start with the User's Original Question:**
{input}

**2. Review the Python Analysis Summary:** This is a high-level text summary of the findings.
{analysis_summary}


**3. Review the Supporting Data Table:** This is the raw data that backs up the summary.
{data_table}


**4. Your Task: Generate the Final Report**
- First, write a concise, clear summary of the findings in prose.
- Then, present the key data from the "Supporting Data Table" in a clean format (like a markdown table) to act as evidence for your summary.
- Ensure the report is well-structured and directly answers the user's original question.

**Final Report:**
"""
)
def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    The final node in the workflow. It synthesizes a report from either the
    Python analysis output or, if that's not present, the direct SQL results.
    """
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
        "analysis_summary": analysis_summary, # This is now safe
        "data_table": data_table_str
    })

    # Use the streaming LLM for the final output
    response = openai_refiner_llm.invoke(prompt)

    print(f"Final Report: {response.content}")
    return {"final_report": response.content}

def router(state: AgentState) -> str:
    print("--- 🚦 Routing... ---")
    
    # NEW: First, check for a Python error. If so, loop back to the python agent.
    if state.get("python_error"):
        print("Python error detected, returning to Python agent for retry.")
        return "python_agent"

    # If there's no error, continue with the plan
    if not state['plan']:
        print("Plan complete. END.")
        return END
        
    next_step = state['plan'][0]
    
    if next_step.startswith("SQL:"):
        print("Next Step: SQL")
        return "sql_agent"
    elif next_step.startswith("PYTHON:"):
        # The router will now correctly handle looping through multiple Python steps
        print("Next Step: PYTHON")
        return "python_agent"
    elif next_step.startswith("SYNTHESIZE:"):
        print("Next Step: SYNTHESIZE")
        return "synthesizer"
    else:
        print("Unrecognized step. END.")
        return END
    
agent = create_openai_tools_agent(
    llm=openai_agent_llm,
    tools=custom_sql_tools,
    prompt=custom_sql_agent_prompt
)

# 4. Create the Agent Executor, which is what we will invoke
custom_sql_agent_executor = AgentExecutor(
    agent=agent,
    tools=custom_sql_tools,
    verbose=True
)
print("Custom SQL Agent Executor created with smart tool.")

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
                    async for chunk in app_graph.astream(initial_state, stream_mode="values"):
                        last_state = chunk
                        
                        # Send high-level status updates to the sse_queue
                        if "plan" in last_state and last_state["plan"]:
                            await sse_queue.put(await send_event("status", {"message": f"Executing: {last_state['plan'][0]}"}))
                        elif "final_report" in last_state and last_state["final_report"]:
                             await sse_queue.put(await send_event("status", {"message": "Finalizing report..."}))
                
                # After the agent is done, send the final answer from the last state
                if last_state and "final_report" in last_state:
                    final_answer = last_state["final_report"]
                    await sse_queue.put(await send_event("data", {"answer": final_answer}))
                else:
                    await sse_queue.put(await send_event("error", {"message": "Could not find a final report in the agent's response."}))

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                traceback.print_exc()
                await sse_queue.put(await send_event("error", {"message": error_message}))
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