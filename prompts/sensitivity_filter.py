import json
from langfuse import Langfuse

sensitive_filter_system_prompt = """You are the sensitivity checker of a bigger fact checker system. The user will send you a message. Your job is to evaluate whether the message contains political and/or religious viewpoints likely to be divisive vs actual fact checking messages.

You will respond with valid JSON in the following format:

{  
    "reasoning": "<your reasoning about why this message is or isn't a political/religious viewpoint>",
    "is_sensitive": <boolean, whether the message is sensitive>
}

Your reasoning should be clear and concise."""

examples = [
    {
        "message": "The PAP government is implementing new tax policies that affect everyone.",
        "reasoning": "This message is checking a fact about the government's tax policies. It is not a political viewpoint.",
        "is_sensitive": "false",
    },
    {
        "message": "Let us pray for everyone affected as only God can help us now.",
        "reasoning": "This message has religious overtones as it discusses praying.",
        "is_sensitive": "true",
    },
    {
        "message": "WP is so much better than PAP",
        "reasoning": "This message is a political viewpoint involving parties in Singapore.",
        "is_sensitive": "true",
    },
    {
        "message": "The weather is beautiful today!",
        "reasoning": "This message is neither political nor religious.",
        "is_sensitive": "false",
    },
]


def compile_messages_array():
    prompt_messages = [{"role": "system", "content": sensitive_filter_system_prompt}]
    for example in examples:
        prompt_messages.append({"role": "user", "content": example["message"]})
        prompt_messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "reasoning": example["reasoning"],
                        "is_sensitive": example["is_sensitive"],
                    }
                ),
            }
        )
    prompt_messages.append({"role": "user", "content": "{{message}}"})
    return prompt_messages


config = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "seed": 11,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "is_sensitive",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "A detailed explanation of why the message why this message is or isn't a political/religious viewpoint",
                    },
                    "is_sensitive": {
                        "type": "boolean",
                        "description": "A boolean flag indicating whether the message is sensitive",
                    },
                },
                "required": ["reasoning", "is_sensitive"],
                "additionalProperties": False,
            },
        },
    },
}

if __name__ == "__main__":
    langfuse = Langfuse()
    prompt_messages = compile_messages_array()
    langfuse.create_prompt(
        name="sensitivity_filter",
        type="chat",
        prompt=prompt_messages,
        labels=["production", "development", "uat"],  # directly promote to production
        config=config,  # optionally, add configs (e.g. model parameters or model tools) or tags
    )
    print("Prompt created successfully.")
