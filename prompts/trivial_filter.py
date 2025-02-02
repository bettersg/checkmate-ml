import json
from langfuse import Langfuse

incorrect_usage_system_prompt = """You are an intelligent assistant for a checking service, that checks any of 1) credibility, 2) legitimacy, 3) safety, and 4) factuality of messages.\
forwarded or sent in to the service. Your role is to filter incoming messages and respond with a structured JSON object containing two fields: reasoning (a string) and needs_checking (a boolean). Follow these guidelines:

Relevance for checking: Examples that are relevant for checking include forwarded WhatsApp messages, emails, or online content that the user finds suspicious, unclear, or worthy of being checked. It may also include factual contents that the user might want us to check, or marketing/promotional content where the user is interested to check their validity. Take an expansive view of this, as forwarded messages that seem innocent could also be a lead in to phishing attempts or scams. If the is some doubt, be conservative and indicate that the message should be checked.

Not Addressed to You: The message should not be a direct inquiry or command meant for you as the chatbot. Examples include questions about your functionality, greetings, or unrelated comments.

Clear Intent: Messages should be explicitly forwarded or referenced content that the user has encountered and wishes to verify.

Response Requirements:
For every message, return a JSON object with the following fields:

reasoning: Provide an explanation of why the message does or does not meet the criteria for checking.
needs_checking: A boolean indicating whether the message should be forwarded for checking.
Set to true if the message requires verification.
Set to false if the message does not meet the criteria. """

examples = [
    {
        "input": """Whats up?""",
        "output": json.dumps(
            {
                "reasoning": "The message is a casual greeting and does not contain any content that requires verification or fact-checking. While it could technically be a lead-in to a social engineering attempt, it is too open-ended to conclude as such. As such, it does not meet the criteria for requiring a check.",
                "needs_checking": False,
            }
        ),
    },
    {
        "input": """Local  SG Lender
5Kx12=450 mth
10Kx36=300 mth
30Kx36=900 mth
No CPF Available,Monthly,Weekly
Contact Us: 80517714 Alvin
https://disckson88.wasap.my""",
        "output": json.dumps(
            {
                "reasoning": "This message appears to advertise moneylending, and provide details. The user could be questioning if this service is a legitimate one.",
                "needs_checking": True,
            }
        ),
    },
    {
        "input": """Hello, I haven't contacted you for a long time. I don't know if you still remember me, so I sent you a photo of me. I really miss you, how are you? My WhatsaAPP account has been stopped, and I hope you can add my Telegram account. You can click the Telegram link below to contact me👇👇👇 https://t.me/L39972?opn=tOD5QJ3x3w""",
        "output": json.dumps(
            {
                "reasoning": "While this appears to be a personal communication, it could very likely be a phishing attempt. It is worth checking.",
                "needs_checking": True,
            }
        ),
    },
    {
        "input": """Show me""",
        "output": json.dumps(
            {
                "reasoning": "The message 'Show me' is a direct command or request likely intended for the assistant, and it does not contain any content that requires checking.",
                "needs_checking": False,
            }
        ),
    },
    {
        "input": """Is this fake news""",
        "output": json.dumps(
            {
                "reasoning": "The message is a direct inquiry asking if something is fake news, but it does not provide any specific content or context that needs verification. Without additional information or a forwarded message, it does not meet the criteria for needing a check.",
                "needs_checking": False,
            }
        ),
    },
    {
        "input": """https://vt.tiktok.com/ZSLpBwVb6/""",
        "output": json.dumps(
            {
                "reasoning": "The message contains a link to a TikTok video. Since we can't be sure what the content is, we should be conservative and indicate that it needs to be checked.",
                "needs_checking": True,
            }
        ),
    },
    {
        "input": """Is lawerence wong the pm of singapore""",
        "output": json.dumps(
            {
                "reasoning": "The message is a factual inquiry about the current Prime Minister of Singapore. The implicit claim is that lawrence wong is the PM of Singapore, which is worth checking.",
                "needs_checking": True,
            }
        ),
    },
    {
        "input": """Hi good morning，Ting Shang Jia are you free now to chat about a job opportunity?""",
        "output": json.dumps(
            {
                "reasoning": "While this message appears to be a personal communication regarding a job opportunity, there is enough within in to suggest it could be a lead-in to a job scam. It is woth checking.",
                "needs_checking": True,
            }
        ),
    },
    {
        "input": """Dear Delegate, a gentle reminder that SMEICC 2022 continues tomorrow, 14 Sept (Wed). Registration starts at 9.30am, Suntec Convention Centre, Lvl 3, Room 324 to 326 and 328 & 329. Business Attire is required. Kindly bring your business card and QR code for entry.""",
        "output": json.dumps(
            {
                "reasoning": "The message seems to be a reminder for an event. The user is likely checking whether there is such an event, and that the sender is legitimate.",
                "needs_checking": True,
            }
        ),
    },
]

config = {
    "model": "gpt-4o",
    "temperature": 0.0,
    "seed": 11,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "needs_checking",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "A detailed explanation of why the message does or does not require checking. This field should clearly articulate the decision-making process.",
                    },
                    "needs_checking": {
                        "type": "boolean",
                        "description": "A flag indicating whether the message contains content that requires checking. Set to true if it needs checking; false otherwise.",
                    },
                },
                "required": ["reasoning", "needs_checking"],
                "additionalProperties": False,
            },
        },
    },
}


def compile_messages_array():
    prompt_messages = [{"role": "system", "content": incorrect_usage_system_prompt}]
    for example in examples:
        prompt_messages.append({"role": "user", "content": example["input"]})
        prompt_messages.append({"role": "assistant", "content": example["output"]})
    prompt_messages.append({"role": "user", "content": "{{message}}"})
    return prompt_messages


if __name__ == "__main__":
    langfuse = Langfuse()
    prompt_messages = compile_messages_array()
    langfuse.create_prompt(
        name="trivial_filter",
        type="chat",
        prompt=prompt_messages,
        labels=["production", "development", "uat"],  # directly promote to production
        config=config,  # optionally, add configs (e.g. model parameters or model tools) or tags
    )
    print("Prompt created successfully.")
