from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

__all__ = ["gemini_client"]
