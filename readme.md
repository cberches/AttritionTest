# HR Retention & Attrition Copilot

An enterprise-grade, multi-agent AI Copilot designed for People Operations and HR Business Partners. This application autonomously predicts employee flight risk using Machine Learning (Scikit-Learn), extracts behavioral drivers using SHAP, visualizes dataset trends, and drafts DOLE-compliant retention strategies via Retrieval-Augmented Generation (RAG).

## 🏗️ Architecture

The Copilot is orchestrated using a **LangGraph state machine** and a semantic LLM router. All prompts are pre-processed locally for safety and PII redaction before touching any external LLM APIs.

```mermaid
graph TD
    User([User Input]) --> Guard[Local Guardrails <br> Regex + MS Presidio]
    Guard -- Blocked --> End([Reject Request])
    Guard -- Clean Prompt --> Router{Semantic LLM Router}
    
    Router -->|Dataset Analytics| Node1[Data Analyst Node <br> Pandas Agent]
    Router -->|Predictions / HRIS| Node2[ML Predictor Node <br> Scikit-Learn + SHAP]
    Router -->|Policy / Emails| Node3[RAG Node <br> Qdrant Vector Store]
    
    Node2 -->|Passes Prediction Data| Node3
    
    Node1 --> Output1[Interactive Plotly Chart]
    Node3 --> Output2[Grounded HR Strategy / Email Draft]
```

## ✨ Key Features
*   **Agentic Routing:** Dynamically routes user intent to the correct analytical or generative sub-agent.
*   **Predictive Analytics:** Uses a trained Random Forest model mapped to 27 employee features to calculate real-time churn probabilities.
*   **Interactive SHAP Visualizations:** Explains *why* an employee is a flight risk through dynamic Plotly bar charts.
*   **Mock HRIS Integration:** Autonomously fetches representative employee profiles from the dataset by department name or Employee ID.
*   **Zero-Cost PII Redaction:** Utilizes **Microsoft Presidio** and SpaCy NLP to instantly redact emails, SSNs, and phone numbers locally before the prompt is sent to OpenAI.
*   **Ragas Evaluated RAG:** Policy retrieval is strictly grounded in the internal HR Playbook, evaluated for context precision and faithfulness.

## 🛠️ Tech Stack
*   **Frontend:** Streamlit, Plotly
*   **Orchestration:** LangGraph, LangChain
*   **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
*   **Machine Learning:** Scikit-Learn, Pandas, SHAP
*   **Vector Database:** Qdrant (In-Memory)
*   **Security & Guardrails:** Microsoft Presidio, Regular Expressions
*   **Observability:** LangSmith

## 📁 Repository Structure
```text
├── .streamlit/
│   └── secrets.toml           # (Do not commit to version control)
├── datasets/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── hr_testset_improved.csv
├── models/
│   ├── model_2024_05_16.sav
│   ├── encoder_2024_05_16.sav
│   └── explainer_2024_05_16.sav
├── app.py                     # Streamlit frontend & Guardrails
├── agent.py                   # LangGraph architecture & Tools
├── hr_retention_playbook.pdf  # Knowledge base for RAG
├── AgentExperimentation.ipynb # Jupyter Notebook for agent prototyping
├── run_evals.ipynb            # Ragas evaluation pipeline
└── requirements.txt
```

## 🚀 Setup and Installation

### 1. Prerequisites
Ensure you have **Python 3.9** installed. (Python 3.10+ may cause compatibility issues with older pickled Scikit-Learn models).

### 2. Create a Virtual Environment
```bash
python -m venv capstone_env
source capstone_env/bin/activate  # On Windows: capstone_env\Scripts\activate
```

### 3. Install Dependencies
Due to specific binary dependencies for Microsoft Presidio and Scikit-Learn, ensure `numpy` and `spacy` are pinned correctly:
```bash
pip install -r requirements.txt
```
*Note: If installing manually, run:*
`pip install streamlit langchain langchain-openai langgraph pandas langchain_experimental tabulate "scikit-learn==1.4.2" qdrant-client pypdf tiktoken langchain_community langchain_qdrant plotly "numpy==1.26.4" presidio-analyzer presidio-anonymizer "spacy<3.8.0"`

### 4. Download the SpaCy NLP Model
Required for Microsoft Presidio to perform PII redaction:
```bash
python -m spacy download en_core_web_lg
```

## ⚙️ Configuration (Secrets)

Instead of a `.env` file, this Streamlit application uses a `secrets.toml` file.

1. Create a `.streamlit` folder in the root directory.
2. Inside that folder, create a file named `secrets.toml`.
3. Add the following configuration (replace the placeholder strings with your actual keys). **DO NOT commit this file to GitHub.**

```toml
# .streamlit/secrets.toml

# OpenAI Configuration
OPENAI_API_KEY = "sk-proj-your-openai-api-key-here"

# LangSmith Observability (Optional but recommended for evaluation)
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "lsv2_pt_your_langsmith_api_key_here"
LANGCHAIN_PROJECT = "HR-Retention-Copilot"
```

## ▶️ Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

## 🧪 Experimentation and Evaluation
*   **Prototyping:** Run `AgentExperimentation.ipynb` to see step-by-step logic validations of the Semantic Router, Pandas Agent, and Qdrant ingestions without starting the UI.
*   **Ragas Evals:** Run `run_evals.ipynb` using the `hr_testset_improved.csv` dataset to calculate Faithfulness, Answer Relevancy, and Context Precision scores for the RAG node.