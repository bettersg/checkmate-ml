from google.genai import types
from clients.gemini import gemini_client
import json
from enum import Enum

translation_system_prompt = """You are a professional translator specializing in English to {language} translations. Your task is to translate the user's text while ensuring:
1. The translation captures the meaning and context of the original text accurately.
2. The tone and style remain consistent with the original message.
3. Avoid direct transliteration where it might make the text awkward or unclear in {language}.
The output should only be the translated text, and should be fluent and grammatically correct.
"""


class SupportedLanguage(Enum):
    CN = "cn"


supported_languages = {SupportedLanguage.CN.value: "Simplified Chinese"}


async def translate_text(text: str, language: str = SupportedLanguage.CN.value):
    """Translates the given text to the specified language."""
    if language not in SupportedLanguage._value2member_map_:
        raise ValueError(f"Unsupported language: {language}")
    try:
        language_enum = SupportedLanguage(language)
        prompt = translation_system_prompt.format(
            language=supported_languages.get(language_enum.value, "Simplified Chinese")
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[types.Part(text=text)],
            config=types.GenerateContentConfig(
                systemInstruction=prompt,
                temperature=0.2,
            ),
        )
        translated_text = response.candidates[0].content.parts[0].text
        return translated_text
    except Exception as e:
        print(f"Error in translation: {e}")
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
        "required": ["reasoning", "intent"],
    },
)

translation_tool = {"function": translate_text, "definition": translation_definition}
