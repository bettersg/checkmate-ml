from langfuse.openai import openai
from langfuse import Langfuse
from dotenv import load_dotenv
import os
import time
from utils import calculate_openai_api_cost

# Load environment variables from .env file
load_dotenv()

# Langfuse setup
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

langfuse = Langfuse()


def translate_to_chinese(session_id, text, messages_trace, cost_tracker):
    """
    Translates the given text from English to Simplified Chinese using GPT-4o-mini via Langfuse.

    Args:
        text (str): The English text to translate.

    Returns:
        str: The translated text in Simplified Chinese.
    """
    try:
        # Fetch the translation prompt from Langfuse
        translation_prompt = langfuse.get_prompt("translation_prompt", label="production")
        
        # Construct the messages for the GPT-4o-mini model
        messages = [
            {"role": "system", "content": translation_prompt.prompt},
            {"role": "user", "content": text}
        ]

        # Make the GPT-4o-mini API call
        response = openai.chat.completions.create(
            model=translation_prompt.config['model'],
            messages=messages,
            session_id=session_id
        )

        # Extract the translated content
        translated_text = response.choices[0].message.to_dict().get("content", "").strip()
        print(f"Translated text: {translated_text}")

        openai_cost = calculate_openai_api_cost(response, translation_prompt.config['model'])
        cost_tracker["total_cost"] += openai_cost
        cost_tracker["cost_trace"].append({
            "type": "translator_openai_call",
            "model": translation_prompt.config['model'],
            "cost": openai_cost
        })

        messages.append(response.choices[0].message)
        messages_trace.append(messages)

        return translated_text, messages_trace
    except Exception as e:
        print(f"Translation failed: {e}")
        return None
