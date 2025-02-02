from google.genai import types
import os
from clients.openai import create_openai_client
from typing import Union, List
from utils.gemini_utils import generate_image_parts, generate_text_parts
from langfuse.decorators import observe, langfuse_context
import json
from logger import StructuredLogger
from langfuse import Langfuse

langfuse = Langfuse()
client = create_openai_client("openai")
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
        prompt = langfuse.get_prompt("summarise_report", label=os.getenv("ENVIRONMENT"))
        messages = prompt.compile()
        config = prompt.config
        if input_text:
            content = [
                {
                    "type": "text",
                    "text": f"User sent in: {input_text}",
                },
            ]
        elif input_image_url:
            caption_suffix = (
                "no caption"
                if input_caption is None
                else f"this caption: {input_caption}"
            )
            content = [
                {
                    "type": "text",
                    "text": f"User sent in the following image with {caption_suffix}",
                },
                {"type": "image_url", "image_url": {"url": input_image_url}},
            ]
        content.append(
            {
                "type": "text",
                "text": f"***Report***: {report}\n****End Report***",
            }
        )
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        try:
            response = client.chat.completions.create(
                model=config.get("model", "gpt-4o"),
                messages=messages,
                temperature=config.get("temperature", 0),
                seed=config.get("seed", 11),
                response_format=config["response_format"],
                langfuse_prompt=prompt,
            )

        except Exception as e:
            child_logger.error(f"Error in generation")
            return {"error": str(e), "success": False}
        try:
            response_json = json.loads(response.choices[0].message.content)
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
