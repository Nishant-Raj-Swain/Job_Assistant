import os
import logging
import traceback
from dotenv import load_dotenv

# 1. LOAD ENVIRONMENT VARIABLES FIRST! (Crucial step)
load_dotenv()

# 2. NOW import FastAPI and your custom modules
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from agent import app_graph, extract_pdf_text, analyze_resume_node
from whatsapp_api import send_text_message, download_media

# ==========================================
# 1. SETUP & LOGGING
# ==========================================
# (Remove the duplicate load_dotenv() from down here)

# This ensures EVERYTHING prints to your Codespace terminal
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("main")

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

# In-memory storage for user resumes (phone_number: resume_text)
# Note: In a real app, use a database. This is perfect for a Codespace demo.
USER_RESUMES = {}

app = FastAPI(title="WhatsApp Career Bot")

# ==========================================
# 2. WEBHOOK ENDPOINTS
# ==========================================
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta calls this GET request to verify your webhook URL."""
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        logger.info("✅ Webhook Verified Successfully by Meta!")
        return int(params.get("hub.challenge", 0))
    logger.error("❌ Webhook Verification Failed. Tokens do not match.")
    return {"detail": "Verification failed"}

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Meta calls this POST request when a user sends a message."""
    try:
        body = await request.json()
        logger.info(f"📥 Received Webhook Payload: {body}")

        if body.get("entry"):
            changes = body["entry"][0].get("changes", [])
            if changes and changes[0].get("value", {}).get("messages"):
                message_data = changes[0]["value"]["messages"][0]
                user_phone = message_data["from"]
                
                # Handle Text Messages
                if message_data["type"] == "text":
                    msg_body = message_data["text"]["body"]
                    logger.info(f"💬 Text Message from {user_phone}: {msg_body}")
                    # Add to background task so we can return 200 OK instantly
                    background_tasks.add_task(process_text_message, user_phone, msg_body)
                
                # Handle PDF Resume Uploads
                elif message_data["type"] == "document":
                    doc_id = message_data["document"]["id"]
                    logger.info(f"📄 Document from {user_phone}: Media ID {doc_id}")
                    background_tasks.add_task(process_resume_upload, user_phone, doc_id)

    except Exception as e:
        logger.error(f"❌ Error reading webhook body: {e}")
    
    # ALWAYS return 200 OK instantly so Meta doesn't time out
    return {"status": "received"}

# ==========================================
# 3. BACKGROUND PROCESSING FUNCTIONS
# ==========================================
async def process_text_message(user_phone: str, text: str):
    """Runs LangGraph and sends the reply."""
    try:
        if not text.startswith("/"):
            text = "/start" # Default to menu if they just type "hi"

        state = {
            "user_id": user_phone,
            "text": text,
            "resume_text": USER_RESUMES.get(user_phone),
            "response": ""
        }
        
        logger.info(f"🧠 Running LangGraph for command: {text}")
        final_state = app_graph.invoke(state)
        
        reply_text = final_state.get("response", "Sorry, an error occurred.")
        send_text_message(user_phone, reply_text)
        
    except Exception as e:
        logger.error(f"❌ LangGraph Execution Error: {traceback.format_exc()}")
        send_text_message(user_phone, "Sorry, an AI processing error occurred. Check server logs.")

async def process_resume_upload(user_phone: str, doc_id: str):
    """Downloads PDF, extracts text, saves to memory, and scores it."""
    try:
        send_text_message(user_phone, "📥 Downloading your resume...")
        
        file_path = download_media(doc_id)
        if not file_path:
            send_text_message(user_phone, "Failed to download resume. Please try again.")
            return
        
        resume_text = extract_pdf_text(file_path)
        if not resume_text:
            send_text_message(user_phone, "Couldn't read text from PDF. Is it a scanned image?")
            return
        
        # Save to memory so other commands (/tailor, /prepare) can use it
        USER_RESUMES[user_phone] = resume_text
        logger.info(f"💾 Saved resume for user {user_phone}")
        
        # Clean up the local PDF file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        send_text_message(user_phone, "✅ Resume saved! Analyzing it now...")
        
        # Manually trigger the analysis node
        state = {
            "user_id": user_phone,
            "text": "analyze_resume",
            "resume_text": resume_text,
            "response": ""
        }
        
        result_state = analyze_resume_node(state)
        send_text_message(user_phone, result_state["response"])
        
    except Exception as e:
        logger.error(f"❌ Resume Processing Error: {traceback.format_exc()}")
        send_text_message(user_phone, "Sorry, an error occurred while reading your resume.")

# ==========================================
# 4. START SERVER
# ==========================================
if __name__ == "__main__":
    import os
    logger.info("🚀 Starting WhatsApp Career Bot...")
    # Render gives us a port via environment variables. Default to 8000.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)