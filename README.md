# LangGraph Agent for Digital Twin

A prototype implementation of an LLM-based query system for data analysis, particularly designed for digital twin applications. This project demonstrates how to build a structured, scalable system using LangGraph to query multiple databases and provide intelligent responses about building sensor data.

## Overview

In the real world, digital twins are implemented at large scale where we build a digital version of complex infrastructure. This particular implementation simplifies the problem to a case where there is just 1 building with 3 floors. Each floor has 3 rooms and each room has 4 sensors.

The agent is built using LangGraph which allows us to structure the whole system. By building such a structure we ensure that we are able to control the outcome of the output and allow for easier scalability and modification.

For this project I have used the OpenAI GPT-4.1 nano model. This model choice is just done due to limited availability of resources. When building a real system further experimentation is needed to decide a model.

## Features

- **Multi-Database Querying**: Intelligently queries both graph databases and time series databases
- **Smart Decision Making**: Determines which databases to query based on user intent
- **Interactive Web Interface**: Streamlit-based chatbot with data visualization capabilities
- **Comprehensive Data Overview**: Built-in dataset exploration with charts and building structure diagrams
- **Structured Architecture**: LangGraph-based workflow for maintainability and scalability

## LangGraph System Architecture

The goal of this system was to be able to query 2 different databases: 1 graph database and another sensor time series database. I could have opted for a simple logic flow and written the flow by myself. I opted to write the flow using LangGraph because it would make my code much more structured and streamlined, easy to interpret, robust, and the biggest advantage is that it allows me to scale the project in the future.

### LangGraph Structure

The core principle behind the agent is as follows:

1. **Receive user query** - Accept natural language questions about the building
2. **Strategy Decision** - Pass the user query with other metadata and prompt to decide what databases will we need to query (Graph or Time Series or mix of both)
3. **Query Generation** - Based on the response we ask the LLM to write one or multiple queries
4. **Query Execution** - Execute the queries against the appropriate databases
5. **Answer Generation** - Using the data, user prompt, metadata, and other instructions we ask the LLM to generate the final answer to the user's query

This structure is fairly robust, it allows the agent to decide if it needs data from both the databases to answer the query, resulting in saving tokens where possible.

The role of each database is defined very clearly:
- **Time Series Database**: Used to obtain the sensor data (temperature, light, CO2, motion)
- **Graph Database**: Used to map structural relationships between floors, rooms and sensors

This saves us from making complicated joins if all the data was on a relational database.

## Dataset

The project uses sensor data from UC Berkeley's smart building dataset, featuring:

- **Building Structure**: 3 floors (4, 5, 6) with 3 rooms each
- **Sensors**: 4 types per room (Temperature, Light, CO2, Motion)
- **Total Sensors**: 36 sensors across 9 rooms
- **Data Types**: Time series sensor readings with timestamps
- **Sensor ID Format**: [Type][BaseID] (e.g., T1 = Temperature in Room 413)

### Room Layout
- **Floor 4**: Rooms 413, 415, 417
- **Floor 5**: Rooms 510, 511, 513  
- **Floor 6**: Rooms 621, 640, 644

## Installation & Setup

1. **Clone or download the project**:
   ```cmd
   cd f:\UCSD\Project
   ```

2. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key**:
   - Create a `.env` file or set environment variable `OPENAI_API_KEY`

4. **Run the application**:
   ```cmd
   streamlit run app.py
   ```
   

## Usage

### Web Interface
Launch the Streamlit app and interact with the chatbot using natural language queries such as:

- "What is the hottest room in the afternoon time?"
- "How many sensors are there on each floor and what types are they?"
- "Which floor has the highest average CO2 levels during working hours?"
- "Which rooms have the most motion detection in the morning?"
- "Show me light level patterns across different rooms during working hours."

### Data Overview
Click the "📊 View Dataset Overview" button to explore:
- Building structure diagram
- Sensor distribution charts
- Data coverage timeline
- Statistical summaries
- Sensor type descriptions

### Programmatic Usage
```python
from langgraph_agent_module import query_building_agent

# Query the agent
result = query_building_agent("What's the temperature in room 413?")
print(result)
```

## Example Queries

The system can handle various types of queries:

- **Structural Questions**: "Which rooms are on floor 5?"
- **Sensor Data Analysis**: "What's the average temperature across all rooms?"
- **Time-based Queries**: "Show me CO2 levels during working hours"
- **Comparative Analysis**: "Which floor has the most motion detection?"
- **Pattern Recognition**: "Are there any correlations between light and motion sensors?"


## Future Scope

This project initially was a prototype to a much more complex system that I was interested in building for the HPC Power and Thermal Characteristics in the Wild Dataset (https://smc-datachallenge.ornl.gov/2022ch7hpc-power/). This dataset has a complex system of a whole building, each floor having multiple racks and each rack having multiple GPUs and CPUs. Each of them are cooled using a central cooling system.

This was just a prototype which I will scale to this data.

### Technical Goals

That was my ambitious goal but talking about more technical achievable goals, I want to add:

- **Error Handling**: If a query fails, understand the error and rewrite the query
- **Human Intervention**: Option for human intervention when needed
- **Visualization**: The ability for the human to ask the agent to plot graphs
- **Model Comparison**: Experiment with different LLM models comparing which are better
- **Prompt Engineering**: Tweak the prompts, increase details, explore prompt caching to save costs
- **Testing Framework**: A robust set of test cases and error metrics to compare results

### Scaling Ambitions

The final thing I would like to add to complete the project would be a robust set of test cases and error metrics to compare the results that all of these changes will bring about.

This is my prototype and I hope I can build it into a fully functional system.

