from openai import OpenAI
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
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
    return OpenAI(api_key=api_key, base_url=base_url)


# Default client for backward compatibility
openai_client = create_openai_client()
