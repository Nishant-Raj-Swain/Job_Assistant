import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Helper function to extract text from PDF
def extract_pdf_text(file_path):
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
    return text

# ==========================================
# 1. DEFINE STATE
# ==========================================
class BotState(Dict[str, Any]):
    user_id: str
    text: str
    resume_text: str
    response: str

# ==========================================
# 2. DEFINE NODES (Bot Features)
# ==========================================
def start_node(state: BotState):
    state["response"] = (
        "🚀 *Welcome to CareerGenie!* 🚀\n\n"
        "Here is what I can do for you:\n"
        "1️⃣ Send a PDF file to analyze and score your resume.\n"
        "2️⃣ `/roadmap [topic]` - Generate a learning roadmap.\n"
        "3️⃣ `/tailor [paste job description]` - Tailor your resume.\n"
        "4️⃣ `/apply [job title]` - Get application links.\n"
        "5️⃣ `/coldEmail [company name]` - Write a cold email.\n"
        "6️⃣ `/assistance` - Get career advice.\n"
        "7️⃣ `/prepare` - Generate interview questions based on your resume.\n\n"
        "Send /start to see this menu again."
    )
    return state

def roadmap_node(state: BotState):
    topic = state["text"].replace("/roadmap", "").strip()
    prompt = ChatPromptTemplate.from_template("Create a detailed week-by-week roadmap for learning {topic}.")
    state["response"] = (prompt | llm).invoke({"topic": topic}).content
    return state

def analyze_resume_node(state: BotState):
    resume = state.get("resume_text", "")
    prompt = ChatPromptTemplate.from_template("Analyze this resume. Score out of 100. List 3 strengths and 3 weaknesses.\n\nResume:\n{resume}")
    state["response"] = (prompt | llm).invoke({"resume": resume}).content
    return state

def tailor_node(state: BotState):
    job_desc = state["text"].replace("/tailor", "").strip()
    resume = state.get("resume_text", "")
    prompt = ChatPromptTemplate.from_template("Rewrite this resume to match this job description.\nResume:\n{resume}\n\nJob:\n{job_desc}")
    state["response"] = (prompt | llm).invoke({"resume": resume, "job_desc": job_desc}).content
    return state

def apply_node(state: BotState):
    job = state["text"].replace("/apply", "").strip()
    prompt = ChatPromptTemplate.from_template("Provide top 3 search links and a checklist for applying to: {job}")
    state["response"] = (prompt | llm).invoke({"job": job}).content
    return state

def cold_email_node(state: BotState):
    company = state["text"].replace("/coldEmail", "").strip()
    resume = state.get("resume_text", "")
    prompt = ChatPromptTemplate.from_template("Write a cold email to {company} based on this resume:\n{resume}")
    state["response"] = (prompt | llm).invoke({"company": company, "resume": resume}).content
    return state

def assistance_node(state: BotState):
    prompt = ChatPromptTemplate.from_template("Give concise career advice on salary negotiation, interview follow ups, and LinkedIn optimization.")
    state["response"] = (prompt | llm).invoke({}).content
    return state

def prepare_node(state: BotState):
    resume = state.get("resume_text", "")
    prompt = ChatPromptTemplate.from_template("Generate 5 technical and 2 behavioral interview questions based on this resume:\n{resume}")
    state["response"] = (prompt | llm).invoke({"resume": resume}).content
    return state

# ==========================================
# 3. DEFINE ROUTER
# ==========================================
def route_command(state: BotState) -> str:
    text = state["text"].lower().strip()
    if text.startswith("/start"):
        return "start"
    elif text.startswith("/roadmap"):
        return "roadmap"
    elif text.startswith("/tailor"):
        return "tailor"
    elif text.startswith("/apply"):
        return "apply"
    elif text.startswith("/coldemail"):
        return "cold_email"
    elif text.startswith("/assistance"):
        return "assistance"
    elif text.startswith("/prepare"):
        return "prepare"
    else:
        state["response"] = "I didn't recognize that command. Send /start to see the menu."
        return END

# ==========================================
# 4. BUILD GRAPH
# ==========================================
workflow = StateGraph(BotState)

workflow.add_node("start", start_node)
workflow.add_node("roadmap", roadmap_node)
workflow.add_node("analyze_resume", analyze_resume_node)
workflow.add_node("tailor", tailor_node)
workflow.add_node("apply", apply_node)
workflow.add_node("cold_email", cold_email_node)
workflow.add_node("assistance", assistance_node)
workflow.add_node("prepare", prepare_node)

workflow.set_conditional_entry_point(
    route_command,
    {
        "start": "start",
        "roadmap": "roadmap",
        "tailor": "tailor",
        "apply": "apply",
        "cold_email": "cold_email",
        "assistance": "assistance",
        "prepare": "prepare",
        END: END
    }
)

for node in ["start", "roadmap", "analyze_resume", "tailor", "apply", "cold_email", "assistance", "prepare"]:
    workflow.add_edge(node, END)

app_graph = workflow.compile()