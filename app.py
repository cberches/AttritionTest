import streamlit as st
import os
import uuid
import json
import pandas as pd
import plotly.io as pio
import plotly.express as px
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agent import hr_copilot
import re
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "HR-Retention-Copilot")


# --- CUSTOM GUARDRAILS (Microsoft Presidio & Regex) ---

@st.cache_resource
def load_presidio():
    """Loads Presidio and injects custom rules to catch FAKE test data during the demo."""
    
    # 1. Load the default library of recognizers
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    
    # 2. Create custom Regex patterns that bypass mathematical checksums (Luhn algorithm)
    # This ensures "fake" demo cards and SSNs are caught instantly
    cc_pattern = Pattern(name="demo_cc", regex=r"\b(?:\d[ -]*?){13,16}\b", score=0.9)
    ssn_pattern = Pattern(name="demo_ssn", regex=r"\b\d{3}-\d{2}-\d{4}\b", score=0.9)
    
    # 3. Map them to the official Presidio entities
    custom_cc = PatternRecognizer(supported_entity="CREDIT_CARD", patterns=[cc_pattern])
    custom_ssn = PatternRecognizer(supported_entity="US_SSN", patterns=[ssn_pattern])
    
    registry.add_recognizer(custom_cc)
    registry.add_recognizer(custom_ssn)
    
    # 4. Initialize the engines
    analyzer = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    
    return analyzer, anonymizer

analyzer, anonymizer = load_presidio()

def secure_prompt(user_input: str) -> str:
    """Validates topics via Regex and redacts PII via Microsoft Presidio."""
    
    # --- STEP 1: Topic Restriction (Regex Blocklist) ---
    user_lower = user_input.lower()
    forbidden_keywords = [
        "crypto", "bitcoin", "ethereum", 
        "bomb", "violence", "kill", "attack",
        "politics", "election", "democrat", "republican",
        "ignore previous", "jailbreak", "bypass", "system prompt"
    ]
    
    if any(keyword in user_lower for keyword in forbidden_keywords):
        return "GUARDRAIL_BLOCKED: Prompt violates safety or topic guidelines."

    # --- STEP 2: PII Redaction (Microsoft Presidio) ---
    try:
        results = analyzer.analyze(
            text=user_input, 
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"], 
            language='en'
        )
        anonymized_result = anonymizer.anonymize(text=user_input, analyzer_results=results)
        return anonymized_result.text
    except Exception as e:
        print(f"Presidio Warning: {e}")
        return user_input

# --- Streamlit UI Setup ---
st.set_page_config(page_title="HR Retention Copilot", layout="wide", initial_sidebar_state="expanded")

# --- SESSION STATE ---
if "page_refreshed" not in st.session_state:
    st.session_state.clear()
    st.session_state.page_refreshed = True

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR MANUAL & EXPORT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=60)
    st.title("User Manual")
    st.markdown("""
    **1. Dataset Analytics 📊** -> Ask for dataset counts or charts.
    **2. Risk Prediction 🎯** -> Provide demographics OR an **Employee ID** to assess flight risk.
    **3. HR Actions ✉️** -> Ask the agent to draft manager emails or retention plans.
    """)
    
    st.divider()
    st.subheader("📄 Session Export")
    st.caption("Download a paper trail of this retention analysis.")
    
    # Compile chat history for export
    export_text = "# HR Retention & Attrition Report\n\n"
    for m in st.session_state.messages:
        role = "👤 **Manager**" if m["role"] == "user" else "🤖 **HR Copilot**"
        export_text += f"{role}:\n{m['content']}\n\n---\n\n"
        
    st.download_button(
        label="Download Markdown Report",
        data=export_text,
        file_name="HR_Retention_Report.md",
        mime="text/markdown",
        use_container_width=True
    )

# --- MAIN CHAT INTERFACE ---
st.title("HR Retention & Attrition Copilot")

if not st.session_state.messages:
    st.info("👋 **Welcome! Try copying and pasting one of these sample prompts:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Action & Communication:**
        > *Draft an email to a department head about retaining a 24 yr old in R&D who is a high flight risk due to no recent promotions.*
        
        **Data Analytics:**
        > *Plot a bar chart showing the average daily rate by department.*
        
        **Agent Capabilities:**
        > *What specific data columns do you need from me to run a churn prediction?*
        """)



    with col2:
        st.markdown("""
        **Mock HRIS Lookup:**
        > *Fetch the file for Research and Development Team, run a churn prediction on them, and recommend a retention strategy.*

        **Multi-Intent (Predict + Policy):**
        > *Can you predict churn for a 24 yr old from Research & Development who does not travel frequently but is 5 years in the company with no promotion or salary hike, and then give me what I can do to retain this employee?*
        """)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "content" in msg and msg["content"]:
            st.markdown(msg["content"])
        if "plotly_json" in msg and msg["plotly_json"]:
            try:
                fig = pio.from_json(msg["plotly_json"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        if "shap_data" in msg and msg["shap_data"]:
            try:
                df_shap = pd.DataFrame(msg["shap_data"])
                fig = px.bar(df_shap, x="value", y="feature", orientation='h', 
                             title="Interactive SHAP Attrition Drivers", 
                             color="value", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

# --- INPUT & ORCHESTRATION ---
if prompt := st.chat_input("Ask about churn risk, HR policies, or dataset trends..."):
    for f in ["interactive_chart.json", "shap_drivers.json"]:
        if os.path.exists(f): os.remove(f)

    # 1. RUN SECURITY CHECK IMMEDIATELY
    sanitized_prompt = secure_prompt(prompt)
    
    # Safely convert Presidio's HTML-style tags to Markdown-safe brackets so they don't turn invisible
    sanitized_prompt = sanitized_prompt.replace("<", "[").replace(">", "]")

    # 2. SAVE AND DISPLAY THE SCRUBBED PROMPT (Proves guardrails work in UI & Export)
    st.session_state.messages.append({"role": "user", "content": sanitized_prompt})
    with st.chat_message("user"):
        st.markdown(sanitized_prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Agent is thinking and routing request..."):
            
            if sanitized_prompt.startswith("GUARDRAIL_BLOCKED"):
                final_response = "I am an HR decision-support assistant. I cannot process this request because it violates safety policies, attempts a jailbreak, or is off-topic."
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            else:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = hr_copilot.invoke({"messages": [HumanMessage(content=sanitized_prompt)]}, config=config)
                final_response = result["messages"][-1].content
                
                display_text = final_response.replace("[CHART_GENERATED]", "").strip()
                st.markdown(display_text)
                new_msg = {"role": "assistant", "content": display_text}
                
                # Check for Analyst Chart
                if os.path.exists("interactive_chart.json") and os.path.getsize("interactive_chart.json") > 0:
                    try:
                        with open("interactive_chart.json", "r") as f:
                            new_msg["plotly_json"] = f.read()
                        fig = pio.from_json(new_msg["plotly_json"])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        pass
                
                # Check for Predictor SHAP Drivers
                if os.path.exists("shap_drivers.json") and os.path.getsize("shap_drivers.json") > 0:
                    try:
                        with open("shap_drivers.json", "r") as f:
                            new_msg["shap_data"] = json.load(f)
                        if new_msg.get("shap_data"):
                            df_shap = pd.DataFrame(new_msg["shap_data"])
                            fig = px.bar(df_shap, x="value", y="feature", orientation='h', 
                                         title="Interactive SHAP Attrition Drivers", 
                                         labels={'value': 'Impact on Churn Risk (Red = Flight Risk, Blue = Retention)', 'feature': 'HR Feature'},
                                         color="value", color_continuous_scale="RdBu_r") 
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        pass
                
                st.session_state.messages.append(new_msg)