from langfuse import Langfuse
import json
from dotenv import load_dotenv, find_dotenv

# This will search for .env file starting from current directory and going up
load_dotenv(find_dotenv())

confirm_same_claim_system_prompt = """You are a professional analyst. Your task is to determine if two texts should be treated as variants of the same claim for fact-checking purposes.

Two texts should be considered variants of the same claim if:
- They make the same TYPE of claim or offer the same TYPE of service/product
- A single fact-check about the legitimacy, legality, or truthfulness would apply to both
- They appear to be variations of the same underlying scheme, scam, or advertisement

They should NOT be considered the same if:
- They make contradictory factual assertions about the same subject (e.g., "2nd wife" vs "3rd wife")
- They would require fundamentally different fact-checking approaches

Minor variations that DON'T affect whether texts are the same claim include:
- Different contact information (names, phone numbers, URLs)
- Different specific numbers (prices, rates, quantities)
- Formatting or phrasing differences
- Other details that don't change the core nature of what's being claimed or offered

Respond in JSON with the following syntax:

{
    "reasoning": <string, explain your analysis>,
    "are_variants_of_same_claim": <boolean, true if they are variants of the same claim>
}
"""

user_prompt = "Text 1: {{text1}}\n*****\nText 2: {{text2}}"

examples = [
    {
        "text1": "Melania is Donald Trump's 2nd wife.",
        "text2": "Melania is Donald Trump's 3rd wife.",
        "reasoning": "These texts make fundamentally different factual claims about which number wife Melania is to Donald Trump. One claims she is his 2nd wife, the other claims she is his 3rd wife. These would require different evidence to verify - you would need to check Trump's marriage history to determine which is correct. The core factual assertion differs, so they cannot be treated as variants of the same claim.",
        "are_variants_of_same_claim": False,
    },
    {
        "text1": """Local  SG Lender
5Kx12=450 mth
10Kx36=300 mth
30Kx36=900 mth
No CPF Available,Monthly,Weekly
Contact Us: 80517714 Alvin
https://disckson88.wasap.my""",
        "text2": """Local  SG Lender
5Kx12=450 mth
10Kx36=300 mth
30Kx36=900 mth
No CPF Available,Monthly,Weekly
Contact Us: 91785124 Paul
https://paul88.wasap.my""",
        "reasoning": "Both texts are advertising the same type of service - informal lending in Singapore with various loan amounts and terms, no CPF requirement, and flexible payment schedules. Even if the specific loan amounts or terms differed between the texts, they would still be variants of the same core claim about offering unlicensed moneylending services. A fact-check about the legality or legitimacy of such lending services would apply to both texts regardless of the specific numbers, contact details, or minor variations in terms.",
        "are_variants_of_same_claim": True,
    },
]


def compile_messages_array():
    prompt_messages = [{"role": "system", "content": confirm_same_claim_system_prompt}]
    for example in examples:
        prompt_messages.append(
            {
                "role": "user",
                "content": user_prompt.replace("{{text1}}", example["text1"]).replace(
                    "{{text2}}", example["text2"]
                ),
            }
        )
        prompt_messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "reasoning": example["reasoning"],
                        "are_variants_of_same_claim": example[
                            "are_variants_of_same_claim"
                        ],
                    }
                ),
            }
        )
    prompt_messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )
    return prompt_messages


config = {
    "model": "gpt-4.1-mini",
    "temperature": 0.0,
    "seed": 11,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "confirm_same_claim",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "An explanation of whether the two texts are variants of the same claim for fact-checking purposes.",
                    },
                    "are_variants_of_same_claim": {
                        "type": "boolean",
                        "description": "A flag indicating whether the two texts should be treated as variants of the same claim.",
                    },
                },
                "required": ["reasoning", "are_variants_of_same_claim"],
                "additionalProperties": False,
            },
        },
    },
}

if __name__ == "__main__":
    langfuse = Langfuse()
    prompt_messages = compile_messages_array()
    langfuse.create_prompt(
        name="confirm_same_claim",
        type="chat",
        prompt=prompt_messages,
        labels=["production", "development", "uat"],  # directly promote to production
        config=config,  # optionally, add configs (e.g. model parameters or model tools) or tags
    )
    print("Prompt created successfully.")
