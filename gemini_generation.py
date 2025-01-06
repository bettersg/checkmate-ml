from tools import (
    search_google_tool,
    get_screenshot_tool,
    check_malicious_url_tool,
    review_report_tool,
    plan_next_step_tool,
    infer_intent_tool,
    translate_text,
)

from agents.gemini_agent import GeminiAgent
from clients.gemini import gemini_client
from datetime import datetime
from typing import Union, List
from pydantic import BaseModel
from context import request_id_var  # Import the context variable

system_prompt = f"""# Context

You are an agent behind CheckMate, a product that allows users based in Singapore to send in dubious content they aren't sure whether to trust, and checks such content on their behalf.

Such content is sent via WhatsApp, and can be a text message or an image message.

# Task
Your task is to:
1. Infer the intent of whoever sent the message in - what exactly about the message they want checked, and how to go about it. Note the distinction between the sender and the author. For example, if the message contains claims but no source, they are probably interested in the factuality of the claims. If the message doesn't contain verifiable claims, they are probably asking whether it's from a legitimate, trustworthy source. If it's about an offer, they are probably enquiring about the legitimacy of the offer. If it's a message claiming it's from the government, they want to know if it is really from the government.
2. Use the supplied tools to help you check the information. Focus primarily on credibility/legitimacy of the source/author and factuality of information/claims, if relevant. If not, rely on contextual clues. When searching, give more weight to reliable, well-known sources. Avoid doing more than 5 searches per message.
3. Submit a clear, concise report to conclude your task. Start with your findings and end with a thoughtful conclusion. Focus on the message and avoid specific third-person references to 'the user' who sent it in. Be helpful and address the intent identified in the first step.

# Other useful information

Date: {datetime.now().strftime("%d %b %Y")}
Popular types of messages:
    - scams
    - illegal moneylending
    - marketing content
    - links to news articles
    - links to social media
    - viral messages designed to be forwarded
    - legitimate government communications
Characteristics of legitimate government communications are:
    - Come via SMS (not Telegram or WhatsApp) from a gov.sg alphanumeric sender ID
    - Contain go.gov.sg links, which is from the official Singapore government link shortener. Do note that in emails or Telegram this could be a fake hyperlink
    - Are in the following format

```Govt SMS Format```
<Full name of agency or service>

---
<Message that does not contain hyperlinks>
---

This is an automated message sent by the Singapore Government.
```End Govt SMS Format```
"""

gemini_agent = GeminiAgent(
    gemini_client,
    tool_list=[
        search_google_tool,
        get_screenshot_tool,
        check_malicious_url_tool,
        review_report_tool,
        plan_next_step_tool,
        infer_intent_tool,
    ],
    system_prompt=system_prompt,
    include_planning_step=False,
    temperature=0.2,
)


class CommunityNoteItem(BaseModel):
    en: str
    cn: str
    links: List[str]
    isControversial: bool = False
    isVideo: bool = False
    isAccessBlocked: bool = False


async def get_outputs(
    data_type: str = "text",
    text: Union[str, None] = None,
    image_url: Union[str, None] = None,
    caption: Union[str, None] = None,
):
    request_id = request_id_var.get()  # Access the request_id from context variable
    outputs = await gemini_agent.generate_note(data_type, text, image_url, caption)
    community_note = outputs["community_note"]
    try:
        chinese_note = await translate_text(community_note, language="cn")
    except Exception as e:
        print(f"Error in translation: {e}")
        chinese_note = community_note

    try:
        return CommunityNoteItem(
            en=community_note,
            cn=chinese_note,
            links=outputs["sources"],
            isControversial=outputs["isControversial"],
            isVideo=outputs["isVideo"],
            isAccessBlocked=outputs["isAccessBlocked"],
        )
    except KeyError:
        print("Error in generating community note")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio

    text = "https://www.msn.com/en-sg/health/other/china-struggles-with-new-virus-outbreak-five-years-after-covid-pandemic/ss-BB1hj9oL?ocid=nl_article_link"
    result = asyncio.run(get_outputs(data_type="text", text=text))
    # prettify the result
    print(result)
