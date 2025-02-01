# tools/review_report.py

from google.genai import types
from collections import OrderedDict
from clients.gemini import gemini_client
from langfuse.decorators import observe
import json
from langfuse import Langfuse

langfuse = Langfuse()

# get system_prompt_review from langfuse
system_prompt_review = langfuse.get_prompt("system_prompt_review", label="production").prompt

response_schema = {
    "type": "OBJECT",
    "properties": OrderedDict(
        [
            (
                "feedback",
                {
                    "type": "STRING",
                    "description": "Your feedback on the report, if any",
                },
            ),
            (
                "passedReview",
                {
                    "type": "BOOLEAN",
                    "description": "A boolean indicating whether the item passed the review",
                },
            ),
        ]
    ),
}


@observe()
async def submit_report_for_review(
    report, sources, isControversial, isVideo, isAccessBlocked
):
    formatted_sources = "\n- ".join(sources) if sources else "<None>"
    if sources:
        formatted_sources = (
            "- " + formatted_sources
        )  # Add the initial '- ' if sources are present
    user_prompt = f"Report: {report}\n*****\nSources:{formatted_sources}"
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[types.Part(text=user_prompt)],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt_review,
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.5,
        ),
    )
    return {"result": json.loads(response.candidates[0].content.parts[0].text)}


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
