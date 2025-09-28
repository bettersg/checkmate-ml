import json
from langfuse import Langfuse

incorrect_usage_system_prompt = """You are an intelligent assistant for an information checking service that verifies the credibility, legitimacy, safety, and factuality of messages forwarded or sent to the service. Your role is to filter incoming messages and respond with a structured JSON object containing three fields: reasoning (a string), needs_checking (a boolean), and confidence (a number). Follow these guidelines:

1. **Relevance for Checking**: Messages relevant for checking include:
   - Forwarded WhatsApp messages, emails, URLs, or online content that users find suspicious or unclear.
   - Marketing or promotional content with unverifiable claims.
   - Financial offers, job opportunities, or personal messages that could lead to phishing or scams.
   - Ambiguous cases where the intent is unclear or the content is vague, especially with shortened URLs, vague claims, or partial information.

2. **Conversational in nature, with the absence of submitted/forwarded materials for checking**: Messages that are conversational, vague, or do not include any specific content or links to verify — such as general inquiries, direct commands (e.g., “What’s up?”, “Show me”), or comments without any submitted material — should not be flagged for checking.

3. **Conservative Approach**: If uncertain, classify conservatively as needing checking (needs_checking: true). However:
   - Include reasoning for why the case is ambiguous and the rationale for the chosen classification.

4. **Confidence**: Include a confidence score (0–1) reflecting your certainty in the decision:
   - If confidence is below 0.7, classify conservatively as needing checking but explain the uncertainty.
   - If confidence is high (0.9 or above), ensure the classification aligns with the clear criteria outlined here.
   - If there is any uncertainty or ambiguity, prioritize reducing false negatives.

5. **Risk Analysis**:
   - Explicitly evaluate the risk of false negatives (missing something important) versus false positives (flagging irrelevant content). Use this analysis to justify your decision.

Response Format:
Return a JSON object with:
- `reasoning` (string): A detailed, step-by-step explanation of the decision-making process.
- `needs_checking` (boolean): Set to true if the message requires verification; false otherwise."""

examples = [
    {
        "input": """Whats up?""",
        "output": json.dumps(
            {
                "reasoning": "The message is a casual greeting and does not contain any content that requires verification or fact-checking. While it could technically be a lead-in to a social engineering attempt, it is too open-ended to conclude as such. As such, it does not meet the criteria for requiring a check.",
                "needs_checking": False,
                "confidence": 0.7,
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
                "confidence": 0.99,
            }
        ),
    },
    {
        "input": """Hello, I haven't contacted you for a long time. I don't know if you still remember me, so I sent you a photo of me. I really miss you, how are you? My WhatsaAPP account has been stopped, and I hope you can add my Telegram account. You can click the Telegram link below to contact me👇👇👇 https://t.me/L39972?opn=tOD5QJ3x3w""",
        "output": json.dumps(
            {
                "reasoning": "While this appears to be a personal communication, it could very likely be a phishing attempt. It is worth checking.",
                "needs_checking": True,
                "confidence": 0.7,
            }
        ),
    },
    {
        "input": """Show me""",
        "output": json.dumps(
            {
                "reasoning": "The message 'Show me' is a direct command or request likely intended for the assistant, and it does not contain any content that requires checking.",
                "needs_checking": False,
                "confidence": 0.96,
            }
        ),
    },
    {
        "input": """Is this fake news""",
        "output": json.dumps(
            {
                "reasoning": "The message is a direct inquiry asking if something is fake news, but it does not provide any specific content or context that needs verification. Without additional information or a forwarded message, it does not meet the criteria for needing a check.",
                "needs_checking": False,
                "confidence": 0.98,
            }
        ),
    },
    {
        "input": """https://vt.tiktok.com/ZSLpBwVb6/""",
        "output": json.dumps(
            {
                "reasoning": "The message contains a link to a TikTok video. Since the content of the link cannot be verified in isolation and could potentially lead to malicious or misleading content, it should be flagged for checking.",
                "needs_checking": True,
                "confidence": 0.99,
            }
        ),
    },
    {
        "input": """Is lawerence wong the pm of singapore""",
        "output": json.dumps(
            {
                "reasoning": "The message is a factual inquiry about the current Prime Minister of Singapore. The implicit claim is that lawrence wong is the PM of Singapore, which is worth checking.",
                "needs_checking": True,
                "confidence": 0.99,
            }
        ),
    },
    {
        "input": """Hi good morning，Ting Shang Jia are you free now to chat about a job opportunity?""",
        "output": json.dumps(
            {
                "reasoning": "While this message appears to be a personal communication regarding a job opportunity, there is enough within in to suggest it could be a lead-in to a job scam. It is worth checking.",
                "needs_checking": True,
                "confidence": 0.8,
            }
        ),
    },
    {
        "input": """Dear Delegate, a gentle reminder that SMEICC 2022 continues tomorrow, 14 Sept (Wed). Registration starts at 9.30am, Suntec Convention Centre, Lvl 3, Room 324 to 326 and 328 & 329. Business Attire is required. Kindly bring your business card and QR code for entry.""",
        "output": json.dumps(
            {
                "reasoning": "The message seems to be a reminder for an event. The user is likely checking whether there is such an event, and that the sender is legitimate.",
                "needs_checking": True,
                "confidence": 0.9,
            }
        ),
    },
    {
        "input": """Boss said I would get a payrise if I work hard, but it was never realized. Is it a scam?""",
        "output": json.dumps(
            {
                "reasoning": "The message is a personal concern about a promise made by an employer. While it reflects a potential issue of unfulfilled promises, it does not fit the typical criteria of a scam that requires verification. It is more of a workplace issue rather than a scam or phishing attempt.",
                "needs_checking": False,
                "confidence": 0.85,
            }
        ),
    },
    {
        "input": """This message sent by puyong1919@f3w.leioxzxs.shop""",
        "output": json.dumps(
            {
                "reasoning": "The email address provided appears suspicious due to its unusual domain structure, which is often indicative of phishing or scam attempts. The message lacks context, but the sender's email alone warrants a check for legitimacy and safety.",
                "needs_checking": True,
                "confidence": 0.75,
            }
        ),
    },
    {
        "input": """Hello, I just received a call from an unknown number and want to know if it’s a scam""",
        "output": json.dumps(
            {
                "reasoning": "The message seeks to clarify if the call received is part of a scam. However, it does not contain any content that requires verification or fact-checking in itself. While the follow-up message sent from the user could include more information that requires checking and/or verification, this message in itself does not. As such, it does not meet the criteria for requiring a check.",
                "needs_checking": False,
                "confidence": 0.8,
            }
        ),
    },
    {
        "input": """I would like to check if this TikTok video of Rishi Sunak (former British PM) and his wife was manipulated as their exaggerated sad faces look cartoonish and distorted.""",
        "output": json.dumps(
            {
                "reasoning": "This message in itself does not contain any content that requires verification or fact-checking in itself. While the follow-up message sent from the user could include more information that requires checking and/or verification, this message in itself does not. As such, it does not meet the criteria for requiring a check. Nevertheless, the message expresses concern about the authenticity of a TikTok video featuring Rishi Sunak and his wife, suggesting that their faces appear exaggerated and potentially manipulated. This falls under the category of content that may have been altered or misrepresented, which warrants verification to determine if the video has been manipulated or deepfaked. When the user subsequently sends another message with the actual link, then the message would require checking.",
                "needs_checking": False,
                "confidence": 0.95,
            }
        ),
    },
    {
        "input": """https://www.channelnewsasia.com/singapore/ge2025-csa-scams-misinformation-election-period-5067091""",
        "output": json.dumps(
            {
                "reasoning": "The message is a link to a news article. The user is likely checking whether its contents are accurate. It should be checked.",
                "needs_checking": True,
                "confidence": 0.99,
            }
        ),
    },
]

config = {
    "model": "gpt-4.1",
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
        labels=[
            "production",
            "development",
            "uat",
            "cf-production",
            "staging",
        ],  # directly promote to production
        config=config,  # optionally, add configs (e.g. model parameters or model tools) or tags
    )
    print("Prompt created successfully.")
