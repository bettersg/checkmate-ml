from google.genai import types
from langfuse.decorators import observe, langfuse_context
from clients.openai import create_openai_client
from logger import StructuredLogger
from enum import Enum
from langfuse import Langfuse
import os

client = create_openai_client("deepseek")
langfuse = Langfuse()


class SupportedLanguage(Enum):
    CN = "cn"


supported_languages = {SupportedLanguage.CN.value: "Simplified Chinese"}

logger = StructuredLogger("translation")


@observe(name="translate_text")
async def translate_text(text: str, language: str = SupportedLanguage.CN.value):
    """Translates the given text to the specified language."""
    child_logger = logger.child(text=text, language=language)
    if language not in SupportedLanguage._value2member_map_:
        raise ValueError(f"Unsupported language: {language}")
    try:
        language_enum = SupportedLanguage(language)
        language = supported_languages.get(language_enum.value, "Simplified Chinese")
        # get summary_prompt from langfuse
        prompt = langfuse.get_prompt("translation", label=os.getenv("ENVIRONMENT"))
        messages = prompt.compile(language=language, text=text)
        config = prompt.config
        response = client.chat.completions.create(
            model=config.get("model", "deepseek-chat"),
            temperature=config.get("temperature", 0.0),
            messages=messages,
            langfuse_prompt=prompt,
        )
        translated_text = response.choices[0].message.content
        return translated_text
    except Exception as e:
        child_logger.error(f"Error in translation: {e}")
        return None


translation_definition = dict(
    name="translate_text",
    description="Translates a given text into the specified language.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "text": {
                "type": "STRING",
                "description": "The text to translate.",
            },
            "language": {
                "type": "ENUM",
                "values": [lang.value for lang in SupportedLanguage],
                "description": "The language to translate to. Can only be 'cn' for Simplified Chinese for now.",
            },
        },
        "required": ["text", "language"],
    },
)

translation_tool = {"function": translate_text, "definition": translation_definition}
