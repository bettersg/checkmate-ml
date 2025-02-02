from langfuse import Langfuse

# get system_prompt_review from langfuse
review_report_system_prompt = """#Instructions

You are playing the role of an editor for a credibility/fact-checking service.

You will be provided with a report is written for the public, on a piece of information that has been submitted.

Your role is to review the submission for:

- clarity
- presence of logical errors or inconsistencies
- credibility of sources used

Points to note:
- Do not nitpick, work on the assumption that the drafter is competent
- You have no ability to do your own research. Do not attempt to use your own knowledge, assume that the facts within the note are correct."""

examples = []  # can add in future

user_prompt = "Report: {{report}}\n*****\nSources:{{formatted_sources}}"

config = {
    "model": "o3-mini",
    "reasoning_effort": "medium",
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "review_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "feedback": {
                        "type": "string",
                        "description": "Your feedback on the report. Be concise and constructive.",
                    },
                    "passedReview": {
                        "type": "boolean",
                        "description": "A boolean indicating whether the item passed the review",
                    },
                },
                "required": ["feedback", "passedReview"],
                "additionalProperties": False,
            },
        },
    },
}


def compile_messages_array():
    prompt_messages = [{"role": "system", "content": review_report_system_prompt}]
    prompt_messages.append({"role": "user", "content": user_prompt})
    return prompt_messages


if __name__ == "__main__":
    langfuse = Langfuse()
    prompt_messages = compile_messages_array()
    langfuse.create_prompt(
        name="review_report",
        type="chat",
        prompt=prompt_messages,
        labels=["production", "development", "uat"],  # directly promote to production
        config=config,  # optionally, add configs (e.g. model parameters or model tools) or tags
    )
    langfuse.get_prompt("review_report", label="production")
    print("Prompt created successfully.")
