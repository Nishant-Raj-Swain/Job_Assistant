import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ACCESS_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
GRAPH_API_URL = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"

def send_text_message(phone_number, text):
    """Sends a standard text message to a WhatsApp user."""
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": text}
    }
    try:
        response = requests.post(GRAPH_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            logger.info(f"✅ Message sent to {phone_number}")
        else:
            logger.error(f"❌ WhatsApp API Error {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"❌ Network error sending message: {e}")

def download_media(media_id):
    """Downloads a user-uploaded document (PDF) from Meta servers."""
    try:
        headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        
        # Step 1: Get the media URL
        url_response = requests.get(f"https://graph.facebook.com/v18.0/{media_id}", headers=headers)
        media_url = url_response.json().get("url")
        
        if not media_url:
            logger.error("Could not get media URL from Meta.")
            return None
            
        # Step 2: Download the actual file
        file_response = requests.get(media_url, headers=headers)
        file_path = f"temp_{media_id}.pdf"
        
        with open(file_path, "wb") as f:
            f.write(file_response.content)
            
        logger.info(f"📥 Downloaded media to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"❌ Error downloading media: {e}")
        return None