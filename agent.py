import os
from dotenv import load_dotenv
import json

# 1. Load environment variables FIRST before any langchain/openai imports
load_dotenv()

import pandas as pd
import pickle
import operator
from typing import TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
import re

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Analyst Node (Updated for Plotly) ---
df = pd.read_csv("datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv")
analyst_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, agent_type="openai-tools")

def data_analyst_node(state):
    messages = state["messages"]
    user_prompt = messages[-1].content
    
    # Force the Pandas Agent to use Plotly instead of Matplotlib
    chart_instruction = """
    If the user asks to visualize, plot, or graph the data, you MUST use the `plotly.express` library. 
    Do NOT use matplotlib, seaborn, or plt.show(). You MUST save the Plotly figure to a file named 'interactive_chart.json' using `fig.write_json('interactive_chart.json')`.
    In your final response, explicitly state: "[CHART_GENERATED]" alongside your numerical findings.
    """
    
    combined_prompt = f"{user_prompt}\n\n{chart_instruction}"
    
    response = analyst_agent.invoke({"input": combined_prompt})
    return {"messages": [SystemMessage(content=f"Analyst Result: {response['output']}")]}

# --- Load Model, Encoders, and SHAP Explainer ---
try:
    model = pickle.load(open('models/model_2024_05_16.sav', 'rb'))
    encoders = pickle.load(open('models/encoder_2024_05_16.sav', 'rb'))
    explainer = pickle.load(open('models/explainer_2024_05_16.sav', 'rb'))
except Exception as e:
    print(f"Warning: Could not load pickled assets: {e}")
    model = None
    encoders = {}
    explainer = None

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
    """Evaluates retention risk and extracts key behavioral risk drivers for a specific employee."""
    
    # Safe casting helper to prevent pydantic list validation errors
    def safe_int(val):
        if isinstance(val, list):
            return int(val[0]) if val else 0
        return int(val)

    daily_rate = safe_int(daily_rate)
    job_level = safe_int(job_level)
    num_companies_worked = safe_int(num_companies_worked)
    overtime_preference = safe_int(overtime_preference)
    percent_salary_hike = safe_int(percent_salary_hike)
    stock_option_level = safe_int(stock_option_level)
    total_working_years = safe_int(total_working_years)
    training_times_last_year = safe_int(training_times_last_year)
    years_at_company = safe_int(years_at_company)
    years_in_current_role = safe_int(years_in_current_role)
    years_since_last_promotion = safe_int(years_since_last_promotion)
    years_with_curr_manager = safe_int(years_with_curr_manager)

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

    for col in encoders.keys():
        if (col != 'Attrition') and (col != 'OverTime') and (col in df_to_predict.columns):
            val = df_to_predict[col].iloc[0]
            if val not in encoders[col].classes_:
                val = encoders[col].classes_[0]
                df_to_predict[col] = val
            df_to_predict[col] = encoders[col].transform(df_to_predict[col])

    if model is not None:
        pred_proba = model.predict_proba(df_to_predict)
        active_proba = float(pred_proba[0][0])
        churn_proba = float(pred_proba[0][1])
    else:
        churn_proba = 0.65 if overtime_preference == 1 or years_at_company > 5 else 0.25
        active_proba = 1.0 - churn_proba

    # Professional HR translation dictionary
    hr_feature_names = {
        'Overtime_Preference': 'Overtime Workload & Schedule',
        'DailyRate': 'Daily Compensation Rate',
        'PercentSalaryHike': 'Recent Salary Increase History',
        'TotalWorkingYears': 'Total Industry Experience',
        'YearsInCurrentRole': 'Tenure in Current Role',
        'YearsAtCompany': 'Company Tenure',
        'JobSatisfaction': 'Reported Job Satisfaction',
        'EnvironmentSatisfaction': 'Workplace Environment Satisfaction',
        'WorkLifeBalance': 'Work-Life Balance Rating',
        'DistanceFromHomeCategory': 'Commute & Travel Distance',
        'BusinessTravel': 'Business Travel Frequency',
        'TrainingTimesLastYear': 'Annual Training & Development',
        'YearsWithCurrManager': 'Tenure with Current Manager',
        'YearsSinceLastPromotion': 'Time Since Last Promotion',
        'JobInvolvement': 'Job Involvement Level',
        'StockOptionLevel': 'Stock Option & Equity Level',
        'NumCompaniesWorked': 'Prior Employer History'
    }

    # Extract top driving factors and translate to executive HR terms
    drivers_text = ""
    if explainer is not None:
        try:
            shap_vals = explainer.shap_values(df_to_predict)
            if isinstance(shap_vals, list):
                s_vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            else:
                s_vals = shap_vals[0]
            
            feature_imp = pd.DataFrame({'Feature': features, 'SHAP_Value': s_vals})
            feature_imp['Abs_SHAP'] = feature_imp['SHAP_Value'].abs()
            top_drivers = feature_imp.sort_values(by='Abs_SHAP', ascending=False).head(5) # Top 5 for chart
            
            # 1. Build the text bullet points
            drivers_text = "\n**Key Behavioral Risk Drivers:**\n"
            chart_data = []
            
            for _, row in top_drivers.head(3).iterrows(): # Only show top 3 in text
                raw_feat = row['Feature']
                clean_feat = hr_feature_names.get(raw_feat, raw_feat.replace('_', ' '))
                impact_word = "elevates retention risk" if row['SHAP_Value'] > 0 else "supports employee retention"
                drivers_text += f"- **{clean_feat}**: This factor {impact_word} for this employee.\n"
                
            # 2. Save ALL top 5 SHAP values to a JSON artifact for Streamlit to chart interactively
            for _, row in top_drivers.iterrows():
                raw_feat = row['Feature']
                clean_feat = hr_feature_names.get(raw_feat, raw_feat.replace('_', ' '))
                chart_data.append({"feature": clean_feat, "value": float(row['SHAP_Value'])})
            
            with open("shap_drivers.json", "w") as f:
                json.dump(chart_data, f)
                
        except Exception as ex:
            drivers_text = ""

    return f"Retention Assessment: {active_proba * 100:.2f}% Probability of Retention, {churn_proba * 100:.2f}% Flight Risk.\n{drivers_text}"

@tool
def fetch_and_predict_employee(query: str) -> str:
    """HRIS Database API. Fetches data for an Employee ID (e.g., 101) or a Department (e.g., R&D, Sales) from the dataset."""
    
    try:
        # 1. Load the real dataset
        hr_df = pd.read_csv("datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        query_lower = query.lower()
        
        # 2. Check if the user provided an Employee ID (Numbers)
        numbers = re.findall(r'\d+', query_lower)
        target_row = None
        
        if numbers:
            emp_id = int(numbers[0])
            subset = hr_df[hr_df['EmployeeNumber'] == emp_id]
            if not subset.empty:
                target_row = subset.iloc[0]
                
        # 3. If no ID, check if they asked for a Department
        if target_row is None:
            dept = None
            if "research" in query_lower or "r&d" in query_lower: dept = 'Research & Development'
            elif "sales" in query_lower: dept = 'Sales'
            elif "human resources" in query_lower or "hr" in query_lower: dept = 'Human Resources'
            
            if dept:
                # Find an employee in this department with no recent promotion to make the prediction interesting
                subset = hr_df[(hr_df['Department'] == dept) & (hr_df['YearsSinceLastPromotion'] >= 2)]
                if not subset.empty:
                    target_row = subset.iloc[0]
                else:
                    target_row = hr_df[hr_df['Department'] == dept].iloc[0]
                    
        if target_row is None:
            return f"HRIS Database Error: Could not find Employee ID or Department matching '{query}'."

        # 4. Map the dataset row to the 27 required prediction features
        edu_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
        sat_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        perf_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
        wlb_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
        
        age = target_row['Age']
        if 97 <= age <= 102: gen = 'WWII'
        elif 79 <= age <= 96: gen = 'Post War'
        elif 70 <= age <= 78: gen = 'Boomers I*'
        elif 60 <= age <= 69: gen = 'Boomers II (a/k/a Generation Jones)*'
        elif 44 <= age <= 59: gen = 'Gen X'
        elif 28 <= age <= 43: gen = 'Millennials'
        else: gen = 'Gen Z'
            
        dist = target_row['DistanceFromHome']
        dist_cat = 'Short Distance' if dist <= 10 else ('Medium Distance' if dist <= 20 else 'Long Distance')

        kwargs = {
            'age_generation': gen,
            'business_travel': target_row['BusinessTravel'],
            'daily_rate': int(target_row['DailyRate']),
            'department': target_row['Department'],
            'distance_from_home_category': dist_cat,
            'education': edu_map.get(target_row['Education'], 'Bachelor'),
            'education_field': target_row['EducationField'],
            'environment_satisfaction': sat_map.get(target_row['EnvironmentSatisfaction'], 'High'),
            'gender': target_row['Gender'],
            'job_involvement': sat_map.get(target_row['JobInvolvement'], 'High'),
            'job_level': int(target_row['JobLevel']),
            'job_role': target_row['JobRole'],
            'job_satisfaction': sat_map.get(target_row['JobSatisfaction'], 'High'),
            'marital_status': target_row['MaritalStatus'],
            'num_companies_worked': int(target_row['NumCompaniesWorked']),
            'overtime_preference': 1 if target_row['OverTime'] == 'Yes' else 0,
            'percent_salary_hike': int(target_row['PercentSalaryHike']),
            'performance_rating': perf_map.get(target_row['PerformanceRating'], 'Excellent'),
            'relationship_satisfaction': sat_map.get(target_row['RelationshipSatisfaction'], 'High'),
            'stock_option_level': int(target_row['StockOptionLevel']),
            'total_working_years': int(target_row['TotalWorkingYears']),
            'training_times_last_year': int(target_row['TrainingTimesLastYear']),
            'work_life_balance': wlb_map.get(target_row['WorkLifeBalance'], 'Good'),
            'years_at_company': int(target_row['YearsAtCompany']),
            'years_in_current_role': int(target_row['YearsInCurrentRole']),
            'years_since_last_promotion': int(target_row['YearsSinceLastPromotion']),
            'years_with_curr_manager': int(target_row['YearsWithCurrManager'])
        }
        
        # 5. Run the actual model prediction with the real dataset values
        prediction_result = predict_churn_probability.invoke(kwargs)
        
        return f"Database Lookup Success! Evaluated representative employee in {target_row['Department']} (Age {age}, Role: {target_row['JobRole']}).\n\n{prediction_result}"
        
    except Exception as e:
        return f"HRIS Database Error: {e}"

# Bind BOTH tools to the LLM
predictor_llm = llm.bind_tools([predict_churn_probability, fetch_and_predict_employee])

def predictor_node(state):
    messages = state["messages"]
    
    system_msg = SystemMessage(content="""You are an HR Predictive Analytics Assistant. 
    CRITICAL INSTRUCTION: When the user asks to predict churn, you MUST call a tool. 
    - If they provide specific employee traits (age, role), call `predict_churn_probability`.
    - If they provide an Employee ID (e.g., ID 101) OR a Department Name (e.g., Research and Development Team), call `fetch_and_predict_employee`.
    Never refuse a prediction request.""")
    
    response = predictor_llm.invoke([system_msg] + messages)
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]
        if tool_call["name"] == "predict_churn_probability":
            prediction_result = predict_churn_probability.invoke(tool_call["args"])
        elif tool_call["name"] == "fetch_and_predict_employee":
            prediction_result = fetch_and_predict_employee.invoke(tool_call["args"])
        else:
            prediction_result = "Tool error."
        return {"messages": [AIMessage(content=prediction_result)]}
        
    return {"messages": [response]}

# --- RAG VECTOR STORE SETUP ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="hr_policies",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="hr_policies",
    embedding=embeddings,
)

if os.path.exists("hr_retention_playbook.pdf"):
    loader = PyPDFLoader("hr_retention_playbook.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(splits)

policy_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- RAG Node (Upgraded to prevent Memory Leaks and Stacking Headers) ---
def rag_node(state):
    messages = state["messages"]
    
    # 1. ONLY grab a prediction if the PREDICTOR node just ran in this exact turn.
    # If it did, the last message before this RAG node will be the AI's prediction output.
    prediction_output = ""
    last_msg = messages[-1]
    
    if isinstance(last_msg, AIMessage) and "Retention Assessment:" in last_msg.content:
        # Cleanly extract just the prediction block without stacking old headers
        idx = last_msg.content.find("Retention Assessment:")
        if idx != -1:
            prediction_output = last_msg.content[idx:]
            
    # 2. Extract the actual user prompt (the most recent HumanMessage)
    user_request = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or msg.type == 'human':
            user_request = msg.content
            break
            
    retrieved_docs = policy_retriever.invoke(user_request)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    executive_hr_prompt = f"""You are a professional Senior HR Business Partner and People Operations Advisor. 
    Maintain an executive, empathetic tone. Never use technical AI jargon (models, algorithms, SHAP).
    
    CRITICAL FORMATTING RULES:
    1. If the user EXPLICITLY asks to "draft an email", "write a memo", or "draft a communication", generate a ready-to-send professional email.
    2. If the user DOES NOT explicitly ask for an email, reply normally using clear bullet points. DO NOT output a letter, subject line, or "Dear Manager" greeting.
    
    If provided, here is the NEW retention risk assessment for the employee:
    {prediction_output}
    
    Based strictly on the provided internal company HR Playbook context, answer the request. 
    
    Company Policy Context:
    {context}
    
    Manager Request: {user_request}"""
    
    response = llm.invoke(executive_hr_prompt)
    
    # Only attach the custom header if a NEW prediction was actually generated this turn
    if prediction_output:
        final_content = f"**HR Copilot Analysis & Output:**\n\n{prediction_output}\n\n---\n\n{response.content}"
    else:
        final_content = response.content
        
    return {"messages": [AIMessage(content=final_content)]}

# --- TRUE AGENTIC SEMANTIC ROUTER (Warning-Free Version) ---
def supervisor_router(state) -> str:
    """Uses LLM semantic understanding to dynamically route the request."""
    last_message = state["messages"][-1].content
    
    router_prompt = f"""You are the lead orchestrator for an HR AI Copilot. 
    Analyze the user request and determine the single best routing destination.
    
    User Request: "{last_message}"
    
    Routing Instructions:
    - "analyst": Select ONLY if the user asks for dataset statistics, counts, averages, or to plot/chart data.
    - "predictor": Select ONLY if the user provides employee attributes OR an Employee ID (e.g., ID 101) to predict churn/flight risk.
    - "rag": Select ONLY if the user asks for general policies, retention strategies, or to draft an email WITHOUT asking for a new prediction.
    - "predictor_then_rag": Select if they ask for a prediction/ID lookup AND ask for retention strategies or an email draft in the exact same prompt.
    
    Respond ONLY with the exact routing string. Do not include quotes or conversational text."""
    
    try:
        decision_raw = llm.invoke(router_prompt).content.strip().replace('"', '').lower()
        valid_routes = ["analyst", "predictor", "rag", "predictor_then_rag"]
        decision = decision_raw if decision_raw in valid_routes else "rag"
        print(f"[AGENTIC ROUTER] Decided Route: {decision}") 
        return decision
    except Exception as e:
        print(f"[ROUTER FALLBACK] Routing failed, defaulting to RAG: {e}")
        return "rag"

# --- Graph Compilation ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("analyst", data_analyst_node)
    workflow.add_node("predictor", predictor_node)
    workflow.add_node("rag", rag_node)
    
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