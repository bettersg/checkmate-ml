import os
from clients.openai import create_openai_client
from typing import Union
from langfuse.decorators import observe
import json
from logger import StructuredLogger
from tools import get_website_screenshot
from langfuse import Langfuse
from urlextract import URLExtract
import asyncio
from tools.website_screenshot import get_website_screenshot
from utils.gemini_utils import generate_screenshot_parts
from google.genai import types

extractor = URLExtract()
langfuse = Langfuse()
client = create_openai_client("openai")
logger = StructuredLogger("preprocess_inputs")


async def get_screenshots_from_text(text: str) -> list:
    """Extract URLs from text and get screenshots in parallel."""
    results = []
    urls = extractor.find_urls(text, only_unique=True, check_dns=True)

    if urls:
        screenshot_tasks = [get_website_screenshot(url) for url in urls]
        screenshot_responses = await asyncio.gather(*screenshot_tasks)

        for url, response in zip(urls, screenshot_responses):
            if response.get("success") and "result" in response:
                results.append({"url": url, "image_url": response["result"]})
            elif response.get("success") is False:
                results.append({"url": url, "error": response.get("error")})
    return results


def get_openai_content(screenshot_results):
    content = []
    for result in screenshot_results:
        if "image_url" in result:
            content.append(
                {"type": "text", "text": f"Screenshot of {result['url']} below:"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": result["image_url"]},
                }
            )
        else:
            content.append(
                {
                    "type": "text",
                    "text": f"Blocked from/failed at getting screenshot of {result['url']}: {result['error']}",
                }
            )
    return content


def get_gemini_content(screenshot_results):
    parts = []
    for result in screenshot_results:
        if "image_url" in result:
            parts.extend(generate_screenshot_parts(result["image_url"], result["url"]))
        else:
            parts.append(
                types.Part.from_text(
                    f"Blocked from/failed at getting screenshot of {result['url']}: {result['error']}",
                )
            )
    return parts


@observe()
async def preprocess_inputs(
    image_url: Union[str, None], caption: Union[str, None], text: Union[str, None]
):
    try:
        prompt = langfuse.get_prompt(
            "preprocess_inputs", label=os.getenv("ENVIRONMENT")
        )
        config = prompt.config
        messages = prompt.compile()
        content = []

        if text:
            content.append(
                {
                    "type": "text",
                    "text": f"User sent in: {text}",
                }
            )
            screenshot_results = await get_screenshots_from_text(text)
            screenshot_content = get_openai_content(screenshot_results)
            content.extend(screenshot_content)
        elif image_url:
            caption_suffix = (
                "no caption" if caption is None else f"this caption: {caption}"
            )
            content = [
                {
                    "type": "text",
                    "text": f"User sent in the following image with {caption_suffix}",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        response = client.chat.completions.create(
            model=config.get("model", "gpt-4o"),
            messages=messages,
            temperature=config.get("temperature", 0),
            seed=config.get("seed", 11),
            response_format=config["response_format"],
            langfuse_prompt=prompt,
        )
        result = json.loads(response.choices[0].message.content)
        return {"success": True, "result": result, "screenshots": screenshot_results}
    except:
        logger.error("Error in preprocess_inputs")
        return {"success": False}


preprocess_inputs_definition = dict(
    name="submit_report_for_review",
    description="Submits a report, which concludes the task.",
    parameters={
        "type": "OBJECT",
        "properties": dict(
            [
                (
                    "image_url",
                    {
                        "type": ["STRING", "NULL"],
                        "description": "The URL of the image to be checked.",
                    },
                ),
                (
                    "caption",
                    {
                        "type": ["STRING", "NULL"],
                        "description": "The caption that accompanies the image to be checked",
                    },
                ),
                (
                    "text",
                    {
                        "type": ["STRING", "NULL"],
                        "description": "The text of the message to be checked.",
                    },
                ),
            ]
        ),
        "required": [
            "image_url",
            "caption",
            "text",
        ],
    },
)

review_report_tool = {
    "function": preprocess_inputs,
    "definition": preprocess_inputs_definition,
}
