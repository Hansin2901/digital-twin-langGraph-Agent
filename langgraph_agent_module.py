# %% [markdown]
# # LangGraph Agent for Digitial Twin
# The following project is an implimentation of prototype for a LLM based query system for data analysis paticularly for a digital twin. In the real world this is done at a large scale where we build a digital version of a complex infrastructure. This paticular implementation simplifies the problem to a case where there is just 1 building with 3 floors. Each floor has 3 rooms and each room has 4 sensors.
# The agent is build using langGraph which allows us to structre the whole system. By building such a structure we ensure that we are able to control the outcome of the output and allow for easier scalability and modification. 
# For this project I have used use the openai gpt4.1 nano model. This model choice is just done due to limited availibility of resourses. When building a real system further experimentation is needed to decide a model. 
# This notebook is divided into x major parts.

# %%
# Import required libraries
import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

# LangChain and LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# Neo4j and data handling
from neo4j import GraphDatabase

# %%
# Configuration for the LLM and Neo4j using Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_BASE_URL = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
NEO4J_URI = st.secrets.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = st.secrets.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD", "password")

# Load the UC Berkeley time series data, extracting from ZIP if needed
csv_path = "uc_berkeley_processed_data_graph_compatible.csv"
zip_path = "uc_berkeley_processed_data_graph_compatible.zip"
if not os.path.exists(csv_path) and os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
time_series_df = pd.read_csv(csv_path)

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    temperature=0.1,
    max_tokens=4000
)

print(f"LLM configured: {llm.model_name}")

# %%
# Enums and data classes for our agent
class DatabaseType(Enum):
    GRAPH = "graph"
    TIME_SERIES = "time_series"
    HYBRID = "hybrid"

class QueryOrder(Enum):
    GRAPH_FIRST = "graph_first"
    TIME_SERIES_FIRST = "time_series_first"

@dataclass
class QueryStrategy:
    database_type: DatabaseType
    order: Optional[QueryOrder] = "null"
    reasoning: str = ""

@dataclass
class DatabaseQuery:
    query_type: str  # "cypher" or "pandas"
    query: str
    description: str

@dataclass
class QueryResult:
    query_type: str
    data: Any
    metadata: Dict[str, Any]

# State definition for our LangGraph agent
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    query_strategy: Optional[QueryStrategy]
    generated_queries: List[DatabaseQuery]
    query_results: List[QueryResult]
    final_answer: str
    graph_schema: str
    time_series_schema: str

# %%
# Neo4j Connection Class
class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            print("Connected to Neo4j database!")
        except Exception as e:
            print(f"Failed to connect to Neo4j database: {e}")
    
    def close(self):
        if self.driver is not None:
            self.driver.close()
            print("Connection to Neo4j closed.")
    
    def execute_query(self, query: str, parameters: Dict = None):
        """Execute a Cypher query against the Neo4j database"""
        if self.driver is None:
            raise Exception("Driver not initialized!")
        
        session = None
        response = None
        
        try:
            session = self.driver.session()
            response = list(session.run(query, parameters or {}))
        except Exception as e:
            print(f"Query failed: {e}")
            raise e
        finally:
            if session is not None:
                session.close()
        
        return response

# Initialize Neo4j connection
neo4j_conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# %%
# Define database schemas
GRAPH_SCHEMA = """
Neo4j Graph Database Schema:

Nodes:
- Floor: {floor_number: int}
- Room: {room_number: int}
- Sensor: {sensor_id: string, type: string}

Relationships:
- (Floor)-[:HAS_ROOM]->(Room)
- (Room)-[:HAS_SENSOR]->(Sensor)

Building Structure (based on id_mapping.csv):
- Floor 4: Rooms 413, 415, 417
- Floor 5: Rooms 510, 511, 513  
- Floor 6: Rooms 621, 640, 644

Sensor ID Format: [SensorType_FirstLetter][BaseID]
- Examples: T1 (temperature sensor), L1 (light sensor), C1 (CO2 sensor), M1 (motion sensor)
- Each room has exactly 4 sensors: tempreture, light, co2, motion
"""

TIME_SERIES_SCHEMA = """
Time Series Data (Pandas DataFrame) - UC Berkeley Smart Building System:

Columns:
- timestamp: datetime - When the reading was taken
- sensor_reading: float - The sensor reading value
- sensor_id: string - Unique identifier matching graph database format (e.g., T1, L2, C3, M4)
- sensor_type: string - Type of the sensor: "tempreture", "light", "co2", "motion"

Building Structure (matches graph database):
- Floor 4: Rooms 413, 415, 417 (Base IDs: 1, 2, 3)
- Floor 5: Rooms 510, 511, 513 (Base IDs: 4, 5, 6)
- Floor 6: Rooms 621, 640, 644 (Base IDs: 7, 8, 9)

Sensor ID Examples:
- Room 413: T1 (tempreture), L1 (light), C1 (co2), M1 (motion)
- Room 415: T2 (tempreture), L2 (light), C2 (co2), M2 (motion)

Time periods:
- Morning: 6:00-12:00
- Afternoon: 12:00-18:00
- Evening: 18:00-24:00
- Night: 0:00-6:00
"""

# %%


# Load the UC Berkeley time series data
print("🏢 Loading UC Berkeley Smart Building System Dataset...")
# Load the graph database compatible processed data
time_series_df = pd.read_csv("uc_berkeley_processed_data_graph_compatible.csv")

# Convert timestamp to datetime if needed
if time_series_df['timestamp'].dtype == 'object':
    time_series_df['timestamp'] = pd.to_datetime(time_series_df['timestamp'])

# Ensure sensor_reading is numeric
time_series_df['sensor_reading'] = pd.to_numeric(time_series_df['sensor_reading'], errors='coerce')

# Remove any rows with missing values
time_series_df = time_series_df.dropna()

# Sort by timestamp
time_series_df = time_series_df.sort_values('timestamp').reset_index(drop=True)

print(f"✅ Loaded UC Berkeley time series data: {len(time_series_df)} records")
print(f"✅ Unique sensors: {time_series_df['sensor_id'].nunique()}")
print(f"✅ Time range: {time_series_df['timestamp'].min()} to {time_series_df['timestamp'].max()}")
print(f"✅ Data shape: {time_series_df.shape}")
print(f"✅ Column names: {time_series_df.columns.tolist()}")
print(f"✅ Sensor types: {sorted(time_series_df['sensor_type'].unique())}")
print(f"✅ Sample sensor IDs: {sorted(time_series_df['sensor_id'].unique())[:10]}")
print(f"✅ Sample data:")
print(time_series_df.head())

# %%
# Node 1: Database Strategy Decision
def decide_database_strategy(state: AgentState) -> AgentState:
    """
    Analyze the query and decide which database(s) to use and in what order.
    """
    print("[DEBUG] Starting decide_database_strategy function")
    print(f"    Original query: {state['original_query']}")
    print(f"    Current state keys: {list(state.keys())}")
    
    strategy_prompt = ChatPromptTemplate.from_template("""
    You are an expert database query planner. Analyze the user's query and determine the optimal database strategy.

    User Query: {query}

    Available databases:
    1. GRAPH DATABASE (Neo4j): Contains UC Berkeley building structure with floors 4-6, rooms 413-644, and sensor metadata
    2. TIME SERIES DATABASE (Pandas): Contains historical sensor readings with timestamps from UC Berkeley dataset
    
    Building Structure (UC Berkeley Smart Building System):
    - Floor 4: Rooms 413, 415, 417 (Base IDs: 1, 2, 3)
    - Floor 5: Rooms 510, 511, 513 (Base IDs: 4, 5, 6)  
    - Floor 6: Rooms 621, 640, 644 (Base IDs: 7, 8, 9)
    
    Each room has 4 sensors: tempreture (T), light (L), co2 (C), motion (M)
    Sensor ID format: [Type][BaseID] (e.g., T1=room 413 temperature, L7=room 621 light)

    If you need to connect sensor readings to building structure, you should use the graph database first to identify relevant rooms/floors, then query time series data.
    You should not answer room or floor information from just sensor IDs - always use the graph database for structure queries.

    Database Schemas:
    {graph_schema}

    {time_series_schema}

    You must choose one of these strategies:
    1. "graph" - Only query the graph database
    2. "time_series" - Only query the time series database  
    3. "hybrid" - Query both databases

    If you choose "hybrid", you must also specify the order:
    - "graph_first" - Query graph database first, then time series
    - "time_series_first" - Query time series first, then graph
    
    The graph database is optimized for relationships and structure, while the time series database is optimized for historical data analysis.
    you will not find any sensor reading data in the graph database, only the structure of the building and relationships between floors, rooms, and sensors.
    

    Respond in JSON format:
    {{
        "database_type": "graph|time_series|hybrid",
        "order": "graph_first|time_series_first|null",
        "reasoning": "Explain your decision in detail"
    }}
    """)

    messages = strategy_prompt.format_messages(
        query=state["original_query"],
        graph_schema=GRAPH_SCHEMA,
        time_series_schema=TIME_SERIES_SCHEMA
    )
    
    print("[DEBUG] Sending prompt to LLM for strategy decision")
    print(f"    Prompt length: {len(str(messages))}")
    
    response = llm.invoke(messages)
    
    print("[DEBUG] Received response from LLM")
    print(f"    Response content: {response.content[:200]}...")
    
    try:
        strategy_data = json.loads(response.content)
        print("[DEBUG] Successfully parsed JSON response")
        print(f"    Strategy data: {strategy_data}")
        
        strategy = QueryStrategy(
            database_type=DatabaseType(strategy_data["database_type"]),
            order=QueryOrder(strategy_data["order"]) if strategy_data.get("order") else None,
            reasoning=strategy_data["reasoning"]
        )
        print(f"[DEBUG] Created strategy object: {strategy}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[DEBUG] Error parsing strategy response: {e}")
        print(f"    Raw response: {response.content}")
        # Default fallback strategy
        strategy = QueryStrategy(
            database_type=DatabaseType.HYBRID,
            order=QueryOrder.GRAPH_FIRST,
            reasoning="Fallback strategy due to parsing error"
        )
        print(f"[DEBUG] Using fallback strategy: {strategy}")
    
    state["query_strategy"] = strategy
    print(f"[DEBUG] Saved strategy to state")
    
    state["messages"].append(AIMessage(content=f"Selected strategy: {strategy.database_type.value}" + 
                                                (f" (order: {strategy.order.value})" if strategy.order else "") +
                                                f"\nReasoning: {strategy.reasoning}"))
    
    print(f"[DEBUG] Added message to state. Total messages: {len(state['messages'])}")
    print("[DEBUG] Finished decide_database_strategy function")
    
    return state

# %%
# Node 2: Query Generation
def generate_queries(state: AgentState) -> AgentState:
    """
    Generate the appropriate database queries based on the selected strategy.
    """
    print("[DEBUG] Starting generate_queries function")
    
    strategy = state["query_strategy"]
    print(f"    Strategy: {strategy.database_type.value}")
    if strategy.order:
        print(f"    Order: {strategy.order.value}")
    
    queries = []
    
    if strategy.database_type == DatabaseType.GRAPH or strategy.database_type == DatabaseType.HYBRID:
        print("[DEBUG] Generating Cypher query for graph database")
        
        # Generate Cypher query
        cypher_prompt = ChatPromptTemplate.from_template("""
        You are an expert in Neo4j Cypher queries. Generate a Cypher query to help answer the user's question.

        User Query: {query}
        
        Graph Schema:
        {graph_schema}

        Building Structure (UC Berkeley Smart Building System):
        - Floor 4: Rooms 413, 415, 417
        - Floor 5: Rooms 510, 511, 513
        - Floor 6: Rooms 621, 640, 644

        Each room has exactly 4 sensors with types: "tempreture", "light", "co2", "motion"
        Sensor ID format: [SensorType_FirstLetter][BaseID] (e.g., T1, L1, C1, M1 for room 413)

        Generate a Cypher query that retrieves relevant information from the graph database.
        Focus on relationships between floors, rooms, and sensors that are relevant to the query.
        The graph database contains the building structure and sensor metadata, but NOT sensor readings.
        Do not query for sensor values or readings - those are in the time series database.

        The query should return relevant nodes and their properties to help identify:
        - Which floors to analyze
        - Which rooms to analyze  
        - Which sensor IDs to look for in time series data

        Respond with only the Cypher query, no explanations:
        """)

        messages = cypher_prompt.format_messages(
            query=state["original_query"],
            graph_schema=GRAPH_SCHEMA
        )
        
        print("[DEBUG] Sending Cypher prompt to LLM")
        response = llm.invoke(messages)
        cypher_query = response.content.strip()
        
        print(f"[DEBUG] Generated Cypher query:")
        print(f"    Query: {cypher_query}")
        
        queries.append(DatabaseQuery(
            query_type="cypher",
            query=cypher_query,
            description="Neo4j Cypher query to get building structure information"
        ))
        print("[DEBUG] Added Cypher query to queries list")
    
    if strategy.database_type == DatabaseType.TIME_SERIES or strategy.database_type == DatabaseType.HYBRID:
        print("[DEBUG] Generating Pandas query for time series database")
        
        # Generate Pandas query
        pandas_prompt = ChatPromptTemplate.from_template("""
        You are an expert in Pandas data analysis. Generate Python code using Pandas to analyze time series sensor data.

        User Query: {query}
        
        Time Series Schema:
        {time_series_schema}

        The data is available in a DataFrame called 'time_series_df' with columns: timestamp, sensor_reading, sensor_id, sensor_type

        Building Structure (UC Berkeley Smart Building System):
        - Floor 4: Rooms 413, 415, 417 with sensor IDs T1,L1,C1,M1 | T2,L2,C2,M2 | T3,L3,C3,M3
        - Floor 5: Rooms 510, 511, 513 with sensor IDs T4,L4,C4,M4 | T5,L5,C5,M5 | T6,L6,C6,M6  
        - Floor 6: Rooms 621, 640, 644 with sensor IDs T7,L7,C7,M7 | T8,L8,C8,M8 | T9,L9,C9,M9

        Sensor Types:
        - "tempreture" (T prefix): Temperature readings
        - "light" (L prefix): Light intensity readings
        - "co2" (C prefix): CO2 concentration readings
        - "motion" (M prefix): Motion detection readings

        Generate Python code that:
        1. Filters the data appropriately based on the query (time periods, sensor types, rooms)
        2. Performs necessary aggregations or calculations
        3. Returns the result as a variable called 'result'

        Example code patterns:
        ```python
        # Filter by time period
        afternoon_data = time_series_df[
            (time_series_df['timestamp'].dt.hour >= 12) & 
            (time_series_df['timestamp'].dt.hour < 18)
        ]
        
        # Filter by sensor type
        temperature_data = time_series_df[time_series_df['sensor_type'] == 'tempreture']
        
        # Filter by specific sensor IDs (for specific rooms)
        room_413_temp = time_series_df[time_series_df['sensor_id'] == 'T1']  # Room 413 temperature
        
        # Aggregation example
        result = afternoon_data.groupby('sensor_id')['sensor_reading'].mean()
        ```

        Respond with only the Python code, no explanations:
        """)

        messages = pandas_prompt.format_messages(
            query=state["original_query"],
            time_series_schema=TIME_SERIES_SCHEMA
        )
        
        print("[DEBUG] Sending Pandas prompt to LLM")
        response = llm.invoke(messages)
        pandas_query = response.content.strip()
        
        print(f"[DEBUG] Generated Pandas query (raw):")
        print(f"    Query: {pandas_query[:200]}...")
        
        # Clean up the code (remove markdown formatting if present)
        if pandas_query.startswith("```python"):
            pandas_query = pandas_query[9:]
            print("[DEBUG] Removed ```python prefix")
        if pandas_query.endswith("```"):
            pandas_query = pandas_query[:-3]
            print("[DEBUG] Removed ``` suffix")
        
        print(f"[DEBUG] Cleaned Pandas query:")
        print(f"    Query: {pandas_query.strip()[:200]}...")
        
        queries.append(DatabaseQuery(
            query_type="pandas",
            query=pandas_query.strip(),
            description="Pandas code to analyze time series sensor data"
        ))
        print("[DEBUG] Added Pandas query to queries list")
    
    state["generated_queries"] = queries
    print(f"[DEBUG] Generated {len(queries)} total queries")
    
    # Add message about generated queries
    query_info = []
    for i, query in enumerate(queries):
        query_info.append(f"{query.query_type.upper()}: {query.description}")
        print(f"    Query {i+1}: {query.query_type} - {query.description}")
    
    state["messages"].append(AIMessage(content=f"Generated {len(queries)} queries:\n" + "\n".join(query_info)))
    
    print(f"[DEBUG] Added query info message. Total messages: {len(state['messages'])}")
    print("[DEBUG] Finished generate_queries function")
    
    return state

# %%
# Node 3: Query Execution
def execute_queries(state: AgentState) -> AgentState:
    """
    Execute the generated queries in the appropriate order.
    """
    
    strategy = state["query_strategy"]
    queries = state["generated_queries"]
    results = []
    
    # Determine execution order
    if strategy.database_type == DatabaseType.HYBRID and strategy.order == QueryOrder.TIME_SERIES_FIRST:
        # Execute time series first, then graph
        execution_order = ["pandas", "cypher"]
    else:
        # Default order: graph first, then time series
        execution_order = ["cypher", "pandas"]
    
    for query_type in execution_order:
        for query in queries:
            if query.query_type == query_type:
                try:
                    if query.query_type == "cypher":
                        # Execute Neo4j query
                        raw_result = neo4j_conn.execute_query(query.query)
                        # Convert to more readable format
                        data = []
                        for record in raw_result:
                            data.append(dict(record))
                        
                        result = QueryResult(
                            query_type="cypher",
                            data=data,
                            metadata={"query": query.query, "record_count": len(data)}
                        )
                        
                    elif query.query_type == "pandas":
                        # Execute Pandas query
                        local_vars = {"time_series_df": time_series_df, "pd": pd, "np": np, "datetime": datetime}
                        exec(query.query, globals(), local_vars)
                        
                        result = QueryResult(
                            query_type="pandas",
                            data=local_vars.get("result", "No result variable found"),
                            metadata={"query": query.query}
                        )
                    
                    results.append(result)
                    print(f"Executed {query.query_type} query successfully")
                    
                except Exception as e:
                    error_result = QueryResult(
                        query_type=query.query_type,
                        data=f"Error: {str(e)}",
                        metadata={"query": query.query, "error": True}
                    )
                    results.append(error_result)
                    print(f"Error executing {query.query_type} query: {e}")
    
    state["query_results"] = results
    
    # Add message about execution results
    state["messages"].append(AIMessage(content=f"Executed {len(results)} queries. Ready for reasoning."))
    
    return state

# %%
# Node 4: Final Reasoning and Answer
def generate_final_answer(state: AgentState) -> AgentState:
    """
    Analyze all query results and generate a comprehensive answer to the original query.
    """
    
    reasoning_prompt = ChatPromptTemplate.from_template("""
    You are an expert data analyst with deep knowledge of smart building systems and digital twins. Use the provided query results to answer the user's original question comprehensively.

    Original Query: {query}

    Building Context (UC Berkeley Smart Building System):
    - Floor 4: Rooms 413, 415, 417 (Base IDs: 1, 2, 3)
    - Floor 5: Rooms 510, 511, 513 (Base IDs: 4, 5, 6)
    - Floor 6: Rooms 621, 640, 644 (Base IDs: 7, 8, 9)
    
    Sensor Types: tempreture (T), light (L), co2 (C), motion (M)
    Sensor ID format: [Type][BaseID] (e.g., T1=room 413 temperature)

    Database Schemas:
    {graph_schema}

    {time_series_schema}

    Query Results:
    {results}

    Instructions:
    1. Analyze all the provided data carefully
    2. Combine insights from both graph and time series data if available
    3. Provide a clear, specific answer to the user's question
    4. Include relevant details and evidence from the data
    5. If the data doesn't fully answer the question, explain what's missing
    6. Do not make up any data or facts, only use the provided query results.

    Provide a comprehensive answer:
    """)

    # Format query results for the prompt
    results_text = ""
    for i, result in enumerate(state["query_results"], 1):
        results_text += f"\nResult {i} ({result.query_type}):\n"
        results_text += f"Query: {result.metadata.get('query', 'N/A')}\n"
        
        if isinstance(result.data, list):
            results_text += f"Data: {result.data[:10]}..."  # Show first 10 items
            if len(result.data) > 10:
                results_text += f" (showing 10 of {len(result.data)} records)"
        else:
            results_text += f"Data: {result.data}"
        
        results_text += f"\nMetadata: {result.metadata}\n"

    messages = reasoning_prompt.format_messages(
        query=state["original_query"],
        graph_schema=GRAPH_SCHEMA,
        time_series_schema=TIME_SERIES_SCHEMA,
        results=results_text
    )
    
    response = llm.invoke(messages)
    final_answer = response.content
    
    state["final_answer"] = final_answer
    state["messages"].append(AIMessage(content=final_answer))
    
    return state

# %%
# Create the LangGraph agent
def create_building_query_agent():
    """
    Create and return the complete LangGraph agent for building queries.
    """
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    workflow.add_node("decide_strategy", decide_database_strategy)
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("execute_queries", execute_queries)
    workflow.add_node("generate_answer", generate_final_answer)
    
    # Define the flow
    workflow.add_edge(START, "decide_strategy")
    workflow.add_edge("decide_strategy", "generate_queries")
    workflow.add_edge("generate_queries", "execute_queries")
    workflow.add_edge("execute_queries", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Create the agent
building_agent = create_building_query_agent()
print("Building Query Agent created successfully!")

# Function to return the Mermaid diagram of the LangGraph workflow
def get_mermaid_diagram():
    """
    Return the Mermaid diagram for the LangGraph agent workflow.
    """
    return building_agent.to_mermaid()

# %%
def query_building_agent(query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Query the building agent with a natural language question.
    
    Args:
        query: The natural language query about the building
        verbose: Whether to print intermediate steps
        
    Returns:
        Dictionary containing the final answer and all intermediate results
    """
    
    # Initialize the state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        original_query=query,
        query_strategy=None,
        generated_queries=[],
        query_results=[],
        final_answer="",
        graph_schema=GRAPH_SCHEMA,
        time_series_schema=TIME_SERIES_SCHEMA
    )
    
    if verbose:
        print(f"Processing query: {query}")
        print("=" * 50)
    
    # Run the agent
    try:
        final_state = building_agent.invoke(initial_state)
        
        if verbose:
            print("\nAgent Execution Summary:")
            print(f"Strategy: {final_state['query_strategy'].database_type.value}")
            if final_state['query_strategy'].order:
                print(f"Order: {final_state['query_strategy'].order.value}")
            print(f"Queries Generated: {len(final_state['generated_queries'])}")
            print(f"Queries Executed: {len(final_state['query_results'])}")
            print("\n" + "=" * 50)
            print("Final Answer:")
            print(final_state['final_answer'])
        
        return {
            "success": True,
            "query": query,
            "strategy": final_state['query_strategy'],
            "queries": final_state['generated_queries'],
            "results": final_state['query_results'],
            "answer": final_state['final_answer'],
            "messages": final_state['messages']
        }
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}")
        
        return {
            "success": False,
            "query": query,
            "error": error_msg,
            "answer": f"I apologize, but I encountered an error while processing your query: {str(e)}"
        }

def print_agent_details(result: Dict[str, Any]):
    """Print detailed information about the agent's execution."""
    
    if not result["success"]:
        print(f"Error: {result['error']}")
        return
    
    print("QUERY ANALYSIS")
    print("-" * 30)
    print(f"Original Query: {result['query']}")
    print(f"Strategy: {result['strategy'].database_type.value}")
    if result['strategy'].order:
        print(f"Execution Order: {result['strategy'].order.value}")
    print(f"Reasoning: {result['strategy'].reasoning}")
    
    print("\nGENERATED QUERIES")
    print("-" * 30)
    for i, query in enumerate(result['queries'], 1):
        print(f"Query {i} ({query.query_type}):")
        print(f"Description: {query.description}")
        print(f"Query: {query.query}")
        print()
    
    print("EXECUTION RESULTS")
    print("-" * 30)
    for i, result_item in enumerate(result['results'], 1):
        print(f"Result {i} ({result_item.query_type}):")
        if isinstance(result_item.data, list):
            print(f"Records: {len(result_item.data)}")
            if result_item.data:
                print(f"Sample: {result_item.data[0]}")
        else:
            print(f"Data: {str(result_item.data)[:200]}...")
        print()
    
    print("FINAL ANSWER")
    print("-" * 30)
    print(result['answer'])

# %%
# Test 1: Simple temperature query
# print("TEST 1: Temperature Query")
# print("=" * 60)

# test_query_1 = "What is the room with the highest temperature reading?"
# result_1 = query_building_agent(test_query_1, verbose=True)

# # %%
# # Test 2: Afternoon temperature query  
# print("TEST 2: Afternoon Temperature Query")
# print("=" * 60)

# test_query_2 = "What is the hottest room in the afternoon time (12 PM to 6 PM)?"
# result_2 = query_building_agent(test_query_2, verbose=True)

# # %%
# # Test 3: Multi-sensor analysis query
# print("\nTEST 3: Multi-sensor Analysis")
# print("=" * 60)

# test_query_3 = "Which floor has the highest average temperature and light levels during working hours (9 AM to 5 PM)?"
# result_3 = query_building_agent(test_query_3, verbose=True)

# # %%
# # Test 4: Building structure query
# print("\nTEST 4: Building Structure Query")
# print("=" * 60)

# test_query_4 = "How many sensors are there on each floor and what types are they?"
# result_4 = query_building_agent(test_query_4, verbose=True)

# # %%
# print("\nTEST 5: Room-specific Structure Query")
# print("=" * 60)

# test_query_5 = "Which rooms are present on floor 5 and what are the sensor IDs present in these rooms?"
# result_5 = query_building_agent(test_query_5, verbose=True)

# %%
# Integration functions for Streamlit chatbot
def initialize_agent():
    """
    Initialize the building query agent and verify all components are working.
    Returns True if successful, False otherwise.
    """
    try:
        # Test if the agent can be created successfully
        if building_agent is None:
            print("❌ Building agent is not initialized")
            return False
            
        # Test if time series data is loaded
        if time_series_df is None or time_series_df.empty:
            print("❌ Time series data is not loaded")
            return False
            
        # Test Neo4j connection
        try:
            neo4j_conn.execute_query("MATCH (n) RETURN count(n) LIMIT 1")
            print("✅ Neo4j connection verified")
        except Exception as e:
            print(f"⚠️ Neo4j connection warning: {e}")
            # Continue without Neo4j for now
            
        # Test a simple query to verify the agent works
        try:
            test_result = query_building_agent("How many sensors are in the building?", verbose=False)
            if test_result["success"]:
                print("✅ Agent test query successful")
                return True
            else:
                print(f"❌ Agent test query failed: {test_result.get('error')}")
                return False
        except Exception as e:
            print(f"❌ Agent test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def get_agent_status():
    """
    Get the current status of the agent and data systems.
    Returns a dictionary with status information.
    """
    try:
        status = {
            "agent_initialized": building_agent is not None,
            "time_series_loaded": time_series_df is not None and not time_series_df.empty,
            "neo4j_connected": False,
            "data_info": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check Neo4j connection
        try:
            neo4j_conn.execute_query("MATCH (n) RETURN count(n) LIMIT 1")
            status["neo4j_connected"] = True
        except:
            status["neo4j_connected"] = False
            
        # Get data information
        if time_series_df is not None and not time_series_df.empty:
            status["data_info"] = {
                "total_records": len(time_series_df),
                "unique_sensors": time_series_df['sensor_id'].nunique(),
                "sensor_types": list(time_series_df['sensor_type'].unique()),
                "time_range": {
                    "start": str(time_series_df['timestamp'].min()),
                    "end": str(time_series_df['timestamp'].max())
                },
                "floors": ["4", "5", "6"],
                "rooms": ["413", "415", "417", "510", "511", "513", "621", "640", "644"]
            }
            
        return status
        
    except Exception as e:
        return {
            "agent_initialized": False,
            "time_series_loaded": False,
            "neo4j_connected": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def get_graph():
    """Return the compiled LangGraph agent for use in Streamlit app."""
    return building_agent

import streamlit as st

def show_data_overview_modal():
    """
    Display a modal (popup) with informative diagrams and a schematic about the dataset.
    """
    with st.popover("📊 Data Overview & Building Structure", use_container_width=True):
        st.markdown("## 📊 Data Overview & Building Structure")
        st.markdown("""
        These diagrams help you understand the structure and content of the UC Berkeley Smart Building dataset.
        """)
        # Diagram 1: Sensor Distribution by Type
        st.markdown("**Sensor Distribution by Type**")
        st.caption("This bar chart shows the number of sensors by type (Temperature, Light, CO₂, Motion) across the entire smart building.")
        st.pyplot(sensor_distribution_plot())
        st.divider()
        # Diagram 2: Data Coverage Over Time
        st.markdown("**Data Coverage Over Time**")
        st.caption("This line plot shows the number of sensor readings recorded each hour, illustrating the density and continuity of data collection over time.")
        st.pyplot(data_coverage_plot())
        st.divider()
        # Diagram 3: Sample Sensor Readings (Box Plot)
        st.markdown("**Distribution of Sensor Readings by Type**")
        st.caption("This box plot shows the distribution of sensor readings by type, helping identify typical ranges and outliers.")
        st.pyplot(sensor_boxplot())
        st.divider()
        # Schematic: Building and Sensor Layout (Mermaid.js)
        st.markdown("**Building and Sensor Layout (Schematic)**")
        st.caption("This schematic shows the building's structure, including all floors, rooms, and the sensors installed in each room.")
        st.markdown(
            '''```mermaid\ngraph TD\n  B[Building] --> F4[Floor 4]\n  B --> F5[Floor 5]\n  B --> F6[Floor 6]\n\n  F4 --> R413_4[Room 413]\n  F4 --> R415_4[Room 415]\n  F4 --> R417_4[Room 417]\n  F5 --> R513_5[Room 513]\n  F5 --> R515_5[Room 515]\n  F5 --> R517_5[Room 517]\n  F6 --> R613_6[Room 613]\n  F6 --> R615_6[Room 615]\n  F6 --> R617_6[Room 617]\n\n  classDef sensor fill:#f9f,stroke:#333,stroke-width:1px;\n  class R413_4,R415_4,R417_4,R513_5,R515_5,R517_5,R613_6,R615_6,R617_6 sensor;\n\n  R413_4 -->|T1| Temp1\n  R413_4 -->|L1| Light1\n  R413_4 -->|C1| CO2_1\n  R413_4 -->|M1| Motion1\n  %% Repeat similarly for other rooms...\n```''',
            unsafe_allow_html=True
        )

# Helper plotting functions for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

def sensor_distribution_plot():
    sensor_counts = time_series_df.groupby('sensor_type')['sensor_id'].nunique()
    fig, ax = plt.subplots()
    sensor_counts.plot(kind='bar', ax=ax)
    ax.set_title("Sensor Distribution by Type")
    ax.set_xlabel("Sensor Type")
    ax.set_ylabel("Number of Unique Sensors")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    return fig

def data_coverage_plot():
    ts = pd.to_datetime(time_series_df['timestamp'])
    readings_per_hour = time_series_df.set_index(ts).resample('H').size()
    fig, ax = plt.subplots()
    readings_per_hour.plot(ax=ax)
    ax.set_title("Sensor Data Coverage Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of Readings per Hour")
    plt.tight_layout()
    return fig

def sensor_boxplot():
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='sensor_type', y='sensor_reading', data=time_series_df, ax=ax)
    ax.set_title("Distribution of Sensor Readings by Type")
    ax.set_xlabel("Sensor Type")
    ax.set_ylabel("Sensor Reading")
    plt.tight_layout()
    return fig


