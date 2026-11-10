import os
from dotenv import load_dotenv

# 1. Load environment variables FIRST before any langchain/openai imports
load_dotenv()

import pandas as pd
import pickle
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

# Initialize LLM (Will now successfully find OPENAI_API_KEY injected from Streamlit secrets)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Analyst Node ---
df = pd.read_csv("datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv")
analyst_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, agent_type="openai-tools")

def data_analyst_node(state):
    messages = state["messages"]
    response = analyst_agent.invoke({"input": messages[-1].content})
    return {"messages": [SystemMessage(content=f"Analyst Result: {response['output']}")]}

# --- Predictor Node (Following your exact Encoders & Mapping) ---
model = pickle.load(open('models/model_2024_05_16.sav', 'rb'))
encoders = pickle.load(open('models/encoder_2024_05_16.sav', 'rb'))

@tool
def predict_churn_probability(
    age_generation: str = 'Millennials', 
    business_travel: str = 'Travel_Rarely', 
    daily_rate: int = 800, 
    department: str = 'Research & Development', 
    distance_from_home_category: str = 'Short Distance', 
    education: str = 'Bachelor', 
    education_field: str = 'Life Sciences', 
    environment_satisfaction: str = 'High', 
    gender: str = 'Male', 
    job_involvement: str = 'High', 
    job_level: int = 2, 
    job_role: str = 'Sales Executive', 
    job_satisfaction: str = 'High', 
    marital_status: str = 'Married', 
    num_companies_worked: int = 1, 
    overtime_preference: int = 0, 
    percent_salary_hike: int = 15, 
    performance_rating: str = 'Excellent', 
    relationship_satisfaction: str = 'High', 
    stock_option_level: int = 1, 
    total_working_years: int = 10, 
    training_times_last_year: int = 2, 
    work_life_balance: str = 'Good', 
    years_at_company: int = 5, 
    years_in_current_role: int = 3, 
    years_since_last_promotion: int = 1, 
    years_with_curr_manager: int = 3
) -> str:
    """Predicts the churn probability for a specific employee using trained encoders."""
    features = [
        'AgeGeneration', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHomeCategory',
        'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement',
        'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'NumCompaniesWorked',
        'Overtime_Preference', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    values = [
        age_generation, business_travel, daily_rate, department, distance_from_home_category,
        education, education_field, environment_satisfaction, gender, job_involvement,
        job_level, job_role, job_satisfaction, marital_status, num_companies_worked,
        overtime_preference, percent_salary_hike, performance_rating, relationship_satisfaction,
        stock_option_level, total_working_years, training_times_last_year, work_life_balance,
        years_at_company, years_in_current_role, years_since_last_promotion, years_with_curr_manager
    ]
    
    df_to_predict = pd.DataFrame(values, index=features).T

    # Apply the exact LabelEncoders saved in pickle
    for col in encoders.keys():
        if (col != 'Attrition') and (col != 'OverTime') and (col in df_to_predict.columns):
            # Handle potential unseen values gracefully by falling back to the first class
            val = df_to_predict[col].iloc[0]
            if val not in encoders[col].classes_:
                val = encoders[col].classes_[0]
                df_to_predict[col] = val
            df_to_predict[col] = encoders[col].transform(df_to_predict[col])

    pred_proba = model.predict_proba(df_to_predict)
    active_proba = float(pred_proba[0][0])
    churn_proba = float(pred_proba[0][1])
    return f"Prediction Breakdown: {active_proba * 100:.2f}% Stay, {churn_proba * 100:.2f}% Churn."

predictor_llm = llm.bind_tools([predict_churn_probability])

def predictor_node(state):
    messages = state["messages"]
    response = predictor_llm.invoke(messages)
    if response.tool_calls:
        prediction_result = predict_churn_probability.invoke(response.tool_calls[0]["args"])
        return {"messages": [AIMessage(content=prediction_result)]}
    return {"messages": [response]}

# --- RAG Node ---
def rag_node(state):
    messages = state["messages"]
    context = "Philippine HR Retention Policy: Employees facing high commute distances or long working years are eligible for flexible hybrid schedules." 
    response = llm.invoke(f"Answer based on context: {context}\nQuestion: {messages[-1].content}")
    return {"messages": [AIMessage(content=f"Policy Recommendation:\n{response.content}")]}

# --- Router ---
# --- Updated Multi-Intent Supervisor Router ---
def supervisor_router(state) -> str:
    last_message = state["messages"][-1].content.lower()
    
    # If the prompt asks for BOTH prediction and retention advice
    has_predict = any(k in last_message for k in ["predict", "probability", "risk", "churn", "years"])
    has_policy = any(k in last_message for k in ["policy", "handbook", "retention", "hike", "promote", "do to retain"])
    
    if has_predict and has_policy:
        return "predictor_then_rag" # Special multi-intent flow
    elif any(keyword in last_message for keyword in ["average", "how many", "compare", "dataset"]):
        return "analyst"
    elif has_predict:
        return "predictor"
    elif has_policy:
        return "rag"
    return "end"


# --- Graph Compilation ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("analyst", data_analyst_node)
    workflow.add_node("predictor", predictor_node)
    workflow.add_node("rag", rag_node)
    
    # Conditional entry point handling multi-intent
    workflow.set_conditional_entry_point(
        supervisor_router, 
        {
            "analyst": "analyst",
            "predictor": "predictor",
            "rag": "rag",
            "predictor_then_rag": "predictor", 
            "end": END
        }
    )
    
    workflow.add_edge("predictor", "rag") 
    workflow.add_edge("analyst", END)
    workflow.add_edge("rag", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

hr_copilot = build_graph()
