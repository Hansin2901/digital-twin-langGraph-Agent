import streamlit as st
from langgraph_agent_module import get_graph, query_building_agent
import time

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="centered")

# Initialize session state for chat history and debug info
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "debug" not in st.session_state:
    st.session_state["debug"] = []

# Load the LangGraph agent (graph)
graph = get_graph()

st.title("🤖 LangGraph Chatbot")

# Utility for rerun (Streamlit 1.32+ uses st.rerun, fallback to st.experimental_rerun)
def safe_rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    except AttributeError:
        pass  # Hide rerun attribute error

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
    mermaid = getattr(graph, "to_mermaid", None)
    if callable(mermaid):
        st.markdown(f"""
        ```mermaid
        {graph.to_mermaid()}
        ```
        """, unsafe_allow_html=True)
    else:
        st.info("Mermaid.js visualization is not available for this graph.\n\nIf you want to visualize your LangGraph, ensure your graph object implements a .to_mermaid() method that returns a valid Mermaid diagram string.\n\nExample:")
        st.markdown(
            """
            ```mermaid
            graph TD
                Start --> DecideStrategy
                DecideStrategy --> GenerateQueries
                GenerateQueries --> ExecuteQueries
                ExecuteQueries --> GenerateAnswer
                GenerateAnswer --> End
            ```
            """,
            unsafe_allow_html=True
        )
        st.caption("This is a sample diagram. Actual graph visualization requires LangGraph's .to_mermaid() support.")

# Bonus: Debug/info expander
with st.expander("🪲 Debug / Info"):
    if st.session_state["debug"]:
        for i, dbg in enumerate(st.session_state["debug"]):
            st.markdown(f"**Turn {i+1}:**")
            st.json(dbg)
    else:
        st.info("No debug info yet.")
