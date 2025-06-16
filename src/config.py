# config.py
import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found. Please set it in your .env file or environment.")
    raise ValueError("GEMINI_API_KEY is not set.")
if not SERPAPI_API_KEY:
    logger.error("SERPAPI_API_KEY not found. Please set it in your .env file or environment.")
    raise ValueError("SERPAPI_API_KEY is not set.")

llm_general = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_retries=2,
    google_api_key=GEMINI_API_KEY,
)

llm_structured = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_retries=2,
    google_api_key=GEMINI_API_KEY,
)