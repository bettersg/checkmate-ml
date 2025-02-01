from google.genai import types

from clients.gemini import gemini_client
from typing import Union, List
from utils.gemini_utils import generate_image_parts, generate_text_parts
from langfuse.decorators import observe, langfuse_context
import json
from logger import StructuredLogger
from langfuse import Langfuse

langfuse = Langfuse()

# get summary_prompt from langfuse
summary_prompt = langfuse.get_prompt("summary_prompt", label="production").prompt

# get summary_response_description from langfuse
summary_response_description_prompt = langfuse.get_prompt("summary_response_description_prompt", label="production").prompt

summary_response_schema = {
    "type": "OBJECT",
    "properties": {
        "community_note": {
            "type": "STRING",
            "description": summary_response_description_prompt,
        }
    },
}

logger = StructuredLogger("summarise_report")

def summarise_report_factory(
    input_text: Union[str, None] = None,
    input_image_url: Union[str, None] = None,
    input_caption: Union[str, None] = None,
):
    """
    Factory function that returns a summarise_report function with input_text, input_image_url, input_caption pre-set.
    """

    @observe()
    async def summarise_report(report: str):
        """
        Summarise the report (with pre-set inputs for text, image URL, or caption).
        """
        child_logger = logger.child(report=report)
        if input_text is not None and input_image_url is not None:
            raise ValueError(
                "Only one of input_text or input_image_url should be provided"
            )
        if input_text:
            parts = generate_text_parts(input_text)
        elif input_image_url:
            parts = generate_image_parts(input_image_url, input_caption)
        parts.append(types.Part.from_text(f"***Report***: {report}\n****End Report***"))
        messages = [types.Content(parts=parts, role="user")]
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=summary_prompt,
                    response_mime_type="application/json",
                    response_schema=summary_response_schema,
                    temperature=0.2,
                ),
            )

        except Exception as e:
            child_logger.error(f"Error in generation")
            return {"error": str(e), "success": False}
        try:
            response_json = json.loads(response.candidates[0].content.parts[0].text)
        except Exception as e:
            child_logger.error(f"Cannot parse response")
            return {"success": False, "error": str(e)}
        if not isinstance(response_json, dict):
            child_logger.error(f"Response from summariser is not a dictionary")
            return {
                "success": False,
                "error": "Response from summariser is not a dictionary",
            }
        if response_json.get("community_note"):
            return {"community_note": response_json["community_note"], "success": True}
        else:
            return {"success": False, "error": "No community note generated"}

    return summarise_report


summarise_report_definition = dict(
    name="summarise_report",
    description="Given a long-form report, and the text or image message the user originally sent in, summarises the report into an X-style community note of around 50-100 words.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "report": {
                "type": "STRING",
                "description": "The long-form report to summarise.",
            },
        },
        "required": ["reasoning", "intent"],
    },
)

summarise_report_tool = {
    "function": None,
    "definition": summarise_report_definition,
}
