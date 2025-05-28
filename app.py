import streamlit as st
from langgraph_agent_module import get_graph, query_building_agent, time_series_df, get_mermaid_diagram
import time

# Optional imports for data visualization
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    VISUALIZATION_AVAILABLE = True
    try:
        import seaborn as sns
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False
except ImportError:
    VISUALIZATION_AVAILABLE = False
    SEABORN_AVAILABLE = False

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="centered")

# Initialize session state for chat history and debug info
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "debug" not in st.session_state:
    st.session_state["debug"] = []

# Load the LangGraph agent (graph)
graph = get_graph()

def show_data_overview():
    """Display comprehensive data overview in a modal-like expander"""
    with st.expander("📊 UC Berkeley Smart Building Dataset Overview", expanded=True):
        st.markdown("### 🏢 Building Structure")
        
        # Building layout schematic using Mermaid
        st.markdown("""
        ```mermaid
        graph TD
            B[UC Berkeley Smart Building] --> F4[Floor 4]
            B --> F5[Floor 5] 
            B --> F6[Floor 6]
            
            F4 --> R413[Room 413<br/>T1, L1, C1, M1]
            F4 --> R415[Room 415<br/>T2, L2, C2, M2]
            F4 --> R417[Room 417<br/>T3, L3, C3, M3]
            
            F5 --> R510[Room 510<br/>T4, L4, C4, M4]
            F5 --> R511[Room 511<br/>T5, L5, C5, M5]
            F5 --> R513[Room 513<br/>T6, L6, C6, M6]
            
            F6 --> R621[Room 621<br/>T7, L7, C7, M7]
            F6 --> R640[Room 640<br/>T8, L8, C8, M8]
            F6 --> R644[Room 644<br/>T9, L9, C9, M9]
            
            classDef building fill:#e1f5fe,stroke:#01579b,stroke-width:2px
            classDef floor fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
            classDef room fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
            
            class B building
            class F4,F5,F6 floor
            class R413,R415,R417,R510,R511,R513,R621,R640,R644 room
        ```
        """)
        
        st.markdown("### 📈 Data Visualizations")
        
        if VISUALIZATION_AVAILABLE and 'time_series_df' in globals():
            try:
                # Diagram 1: Sensor Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Sensor Distribution by Type**")
                    sensor_counts = time_series_df.groupby('sensor_type')['sensor_id'].nunique()
                    st.bar_chart(sensor_counts)
                    st.caption("Number of unique sensors by type across the building")
                
                with col2:
                    st.markdown("**Data Coverage Timeline**")
                    if 'timestamp' in time_series_df.columns:
                        time_series_df['timestamp'] = pd.to_datetime(time_series_df['timestamp'])
                        daily_counts = time_series_df.groupby(time_series_df['timestamp'].dt.date).size()
                        st.line_chart(daily_counts)
                        st.caption("Number of sensor readings per day")
                
                # Diagram 3: Sensor Reading Distribution
                st.markdown("**Sensor Reading Ranges by Type**")
                
                # Create pivot table for sensor readings
                sensor_stats = time_series_df.groupby('sensor_type')['sensor_reading'].agg(['min', 'max', 'mean', 'std']).round(2)
                st.dataframe(sensor_stats, use_container_width=True)
                st.caption("Statistical summary of sensor readings by type")
                
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")
                
        else:
            st.info("📊 Data visualizations require matplotlib and pandas. Install them for enhanced charts.")
        
        # Dataset Info
        st.markdown("### 📋 Dataset Information")
        
        info_cols = st.columns(4)
        try:
            with info_cols[0]:
                st.metric("Total Records", f"{len(time_series_df):,}")
            with info_cols[1]:
                st.metric("Unique Sensors", f"{time_series_df['sensor_id'].nunique()}")
            with info_cols[2]:
                st.metric("Sensor Types", f"{time_series_df['sensor_type'].nunique()}")
            with info_cols[3]:
                st.metric("Time Span", f"{(pd.to_datetime(time_series_df['timestamp']).max() - pd.to_datetime(time_series_df['timestamp']).min()).days} days")
        except:
            st.info("Dataset metrics will be available once data is loaded.")
        
        # Sensor Type Descriptions
        st.markdown("### 🔧 Sensor Types")
        sensor_info = {
            "🌡️ Temperature (T)": "Measures ambient temperature in each room",
            "💡 Light (L)": "Measures light intensity/illumination levels", 
            "🌬️ CO2 (C)": "Measures carbon dioxide concentration for air quality",
            "🚶 Motion (M)": "Detects movement/occupancy in rooms"
        }
        
        for sensor, description in sensor_info.items():
            st.markdown(f"**{sensor}:** {description}")
        
        # Building Details
        st.markdown("### 🏗️ Building Layout")
        st.markdown("""
        - **3 Floors:** 4, 5, and 6
        - **9 Rooms:** 3 rooms per floor (413, 415, 417 | 510, 511, 513 | 621, 640, 644)
        - **36 Sensors:** 4 sensors per room (Temperature, Light, CO2, Motion)
        - **Sensor IDs:** Format [Type][Room_Base_ID] (e.g., T1 = Temperature in Room 413)
        """)

# Utility for rerun (Streamlit 1.32+ uses st.rerun, fallback to st.experimental_rerun)
def safe_rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    except AttributeError:
        pass  # Hide rerun attribute error

st.title("🤖 LangGraph Chatbot")

# Add user guidance note about query errors
st.info("""
If your response looks like there was an error in the query, try entering the same prompt a few times. Sometimes the agent may need a retry to process your request correctly.
""")

# Data Overview Button
if st.button("📊 View Dataset Overview", use_container_width=True):
    show_data_overview()

# Clear chat button
if st.button("🗑️ Clear chat", use_container_width=True):
    st.session_state["messages"] = []
    st.session_state["debug"] = []
    safe_rerun()

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("latency") is not None:
            st.caption(f"⏱️ Latency: {msg['latency']:.2f} seconds")

# Chat input
if user_input := st.chat_input("Type your message..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("LangGraph is thinking..."):
        start_time = time.perf_counter()
        try:
            result = query_building_agent(user_input, verbose=False)
            latency = time.perf_counter() - start_time
            if isinstance(result, dict):
                reply = result.get("answer") or result.get("output") or str(result)
                debug = result
            else:
                reply = str(result)
                debug = result
        except Exception as e:
            reply = f"❌ Error: {e}"
            debug = str(e)
            latency = None
        st.session_state["messages"].append({"role": "assistant", "content": reply, "latency": latency})
        st.session_state["debug"].append(debug)
        with st.chat_message("assistant"):
            st.markdown(reply)
            if latency is not None:
                st.caption(f"⏱️ Latency: {latency:.2f} seconds")
    safe_rerun()

# Example queries section
st.markdown("### 💡 Example Queries")
example_queries = [
    "What is the hottest room in the afternoon time?",
    "How many sensors are there on each floor and what types are they?",
    "Which floor has the highest average CO2 levels during working hours?",
    "Which rooms have the most motion detection in the morning?",
    "Show me light level patterns across different rooms during working hours.",
    "Which rooms are present on floor 5 and what are the sensor IDs present in these rooms?"
]
cols = st.columns(len(example_queries))
for i, (col, query) in enumerate(zip(cols, example_queries)):
    with col:
        if st.button(query, key=f"example_query_{i}"):
            st.session_state["messages"].append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            with st.spinner("LangGraph is thinking..."):
                start_time = time.perf_counter()
                try:
                    result = query_building_agent(query, verbose=False)
                    latency = time.perf_counter() - start_time
                    if isinstance(result, dict):
                        reply = result.get("answer") or result.get("output") or str(result)
                        debug = result
                    else:
                        reply = str(result)
                        debug = result
                except Exception as e:
                    reply = f"❌ Error: {e}"
                    debug = str(e)
                    latency = None
                st.session_state["messages"].append({"role": "assistant", "content": reply, "latency": latency})
                st.session_state["debug"].append(debug)
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    if latency is not None:
                        st.caption(f"⏱️ Latency: {latency:.2f} seconds")
            safe_rerun()

# Bonus: Mermaid.js graph visualization
with st.expander("🗺️ View LangGraph Structure (Mermaid.js)"):
    try:
        mermaid_diagram = get_mermaid_diagram()
        if mermaid_diagram:
            st.markdown(f"""
            ```mermaid
            {mermaid_diagram}
            ```
            """, unsafe_allow_html=True)
        else:
            raise ValueError("No mermaid diagram available")
    except Exception as e:
        st.info("Mermaid.js visualization is not available for this graph.\n\nIf you want to visualize your LangGraph, ensure your graph object implements a .to_mermaid() method that returns a valid Mermaid diagram string.\n\nExample:")
        st.markdown(
            """
            ```mermaid
            graph TD
                Start --> DecideStrategy[Decide Strategy]
                DecideStrategy --> GenerateQueries[Generate Queries]
                GenerateQueries --> ExecuteQueries[Execute Queries]
                ExecuteQueries --> GenerateAnswer[Generate Answer]
                GenerateAnswer --> End
                
                classDef startEnd fill:#e1f5fe
                classDef process fill:#f3e5f5
                classDef decision fill:#fff3e0
                
                class Start,End startEnd
                class DecideStrategy decision
                class GenerateQueries,ExecuteQueries,GenerateAnswer process
            ```
            """,
            unsafe_allow_html=True
        )
        st.caption("This is the actual LangGraph workflow structure for the UC Berkeley Smart Building Agent.")

# Bonus: Debug/info expander
with st.expander("🪲 Debug / Info"):
    if st.session_state["debug"]:
        for i, dbg in enumerate(st.session_state["debug"]):
            st.markdown(f"**Turn {i+1}:**")
            st.json(dbg)
    else:
        st.info("No debug info yet.")
