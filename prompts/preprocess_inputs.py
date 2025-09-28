from langfuse import Langfuse

review_report_system_prompt = """# Context

You are an agent behind CheckMate, a product that allows users based in Singapore to send in dubious content they aren't sure whether to trust, and checks such content on their behalf.

Such content can be a text message or an image message. Image messages could, among others, be screenshots of their phone, pictures from their camera, or downloaded images. They could also be accompanied by captions.

# Task

Given these inputs:
- content submitted by the user, which could be an image or a text
- screenshots of any webpages whose links within the content
 
Your task is to:
1. Determine if the screenshots indicate that the content is a video, and/or access to the content is blocked.
2. Infer the intent of whoever sent the message in - what exactly about the message they want checked, and how to go about it. Note the distinction between the sender and the author. For example, if the message contains claims but no source, they are probably interested in the factuality of the claims. If the message doesn't contain verifiable claims, they are probably asking whether it's from a legitimate, trustworthy source. If it's about an offer, they are probably enquiring about the legitimacy of the offer. If it's a message claiming it's from the government, they want to know if it is really from the government."""

config = {
    "model": "gpt-4o",
    "temperature": 0.0,
    "seed": 11,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "summarise_report",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind the intent you inferred from the message.",
                    },
                    "is_access_blocked": {
                        "type": "boolean",
                        "description": "True if the content or URL sent by the user to be checked is inaccessible/removed/blocked. An example is being led to a login page instead of post content.",
                    },
                    "is_video": {
                        "type": "boolean",
                        "description": "True if the content or URL sent by the user to be checked points to a video (e.g., YouTube, TikTok, Instagram Reels, Facebook videos).",
                    },
                    "intent": {
                        "type": "string",
                        "description": "What the user's intent is, e.g. to check whether this is a scam, to check if this is really from the government, to check the facts in this article, etc.",
                    },
                },
                "required": ["is_access_blocked", "is_video", "reasoning", "intent"],
                "additionalProperties": False,
            },
        },
    },
}


def compile_messages_array():
    prompt_messages = [{"role": "system", "content": review_report_system_prompt}]
    return prompt_messages


if __name__ == "__main__":
    langfuse = Langfuse()
    prompt_messages = compile_messages_array()
    langfuse.create_prompt(
        name="preprocess_inputs",
        type="chat",
        prompt=prompt_messages,
        labels=["production", "development", "uat"],  # directly promote to production
        config=config,  # optionally, add configs (e.g. model parameters or model tools) or tags
    )
    langfuse.get_prompt("preprocess_inputs", label="production")
    print("Prompt created successfully.")
