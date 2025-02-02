# tools/review_report.py

from google.genai import types
from collections import OrderedDict
from clients.openai import create_openai_client
from langfuse.decorators import observe
import json
from langfuse import Langfuse
import os

langfuse = Langfuse()

# get system_prompt_review from langfuse
client = create_openai_client("openai")


@observe()
async def submit_report_for_review(
    report, sources, isControversial, isVideo, isAccessBlocked
):
    prompt = langfuse.get_prompt("review_report", label=os.getenv("ENVIRONMENT"))
    config = prompt.config

    formatted_sources = "\n- ".join(sources) if sources else "<None>"
    if sources:
        formatted_sources = (
            "- " + formatted_sources
        )  # Add the initial '- ' if sources are present
    messages = prompt.compile(report=report, formatted_sources=formatted_sources)
    response = client.chat.completions.create(
        model=config.get("model", "o3-mini"),
        reasoning_effort=config.get("reasoning_effort", "medium"),
        messages=messages,
        response_format=config["response_format"],
        langfuse_prompt=prompt,
    )
    result = json.loads(response.choices[0].message.content)
    return {"result": result}


review_report_definition = dict(
    name="submit_report_for_review",
    description="Submits a report, which concludes the task.",
    parameters={
        "type": "OBJECT",
        "properties": OrderedDict(
            [
                (
                    "report",
                    {
                        "type": "STRING",
                        "description": "The content of the report. This should enough context for readers to stay safe and informed. Try and be succinct.",
                    },
                ),
                (
                    "sources",
                    {
                        "type": "ARRAY",
                        "items": {
                            "type": "STRING",
                            "description": "A link from which you sourced content for your report.",
                        },
                        "description": "A list of links from which your report is based. Avoid including the original link sent in for checking as that is obvious.",
                    },
                ),
                (
                    "isControversial",
                    {
                        "type": "BOOLEAN",
                        "description": "True if the content contains political or religious viewpoints that are grounded in opinions rather than provable facts, and are likely to be divisive or polarizing.",
                    },
                ),
                (
                    "isVideo",
                    {
                        "type": "BOOLEAN",
                        "description": "True if the content or URL sent by the user to be checked points to a video (e.g., YouTube, TikTok, Instagram Reels, Facebook videos).",
                    },
                ),
                (
                    "isAccessBlocked",
                    {
                        "type": "BOOLEAN",
                        "description": "True if the content or URL sent by the user to be checked is inaccessible/removed/blocked. An example is being led to a login page instead of post content.",
                    },
                ),
            ]
        ),
        "required": [
            "report",
            "sources",
            "isControversial",
            "isVideo",
            "isAccessBlocked",
        ],
    },
)

review_report_tool = {
    "function": submit_report_for_review,
    "definition": review_report_definition,
}
