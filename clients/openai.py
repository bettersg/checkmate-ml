from langfuse.openai import OpenAI
import os
from logger import StructuredLogger
from models import SupportedModelProvider

logger = StructuredLogger("openai_client")


def create_openai_client(provider=SupportedModelProvider.OPENAI):
    if provider == SupportedModelProvider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
    elif provider == SupportedModelProvider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
    elif provider == SupportedModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL")
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


# Default client for backward compatibility
openai_client = create_openai_client()
