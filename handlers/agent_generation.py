from tools import (
    search_google_tool,
    get_screenshot_tool,
    check_malicious_url_tool,
    review_report_tool,
    plan_next_step_tool,
    infer_intent_tool,
    translate_text,
)

from agents.openai_agent import OpenAIAgent
from agents.gemini_agent import GeminiAgent
from clients.gemini import gemini_client
from clients.openai import create_openai_client
from datetime import datetime
from typing import Union, List
from models import SavedAgentCall, SupportedModelProvider
from context import request_id_var  # Import the context variable
from logger import StructuredLogger
from langfuse.decorators import observe, langfuse_context
from clients.firestore_db import db
import os

system_prompt = """# Context

You are an agent behind CheckMate, a product that allows users based in Singapore to send in dubious content they aren't sure whether to trust, and checks such content on their behalf.

Such content can be a text message or an image message. Image messages could, among others, be screenshots of their phone, pictures from their camera, or downloaded images. They could also be accompanied by captions.

# Task
Your task is to:
1. Infer the intent of whoever sent the message in - what exactly about the message they want checked, and how to go about it. Note the distinction between the sender and the author. For example, if the message contains claims but no source, they are probably interested in the factuality of the claims. If the message doesn't contain verifiable claims, they are probably asking whether it's from a legitimate, trustworthy source. If it's about an offer, they are probably enquiring about the legitimacy of the offer. If it's a message claiming it's from the government, they want to know if it is really from the government.
2. Use the supplied tools to help you check the information. Focus primarily on credibility/legitimacy of the source/author and factuality of information/claims, if relevant. If not, rely on contextual clues. When searching, give more weight to reliable, well-known sources. Use searches and visit sites judiciously, you only get 5 of each.
3. Submit a report to conclude your task. Start with your findings and end with a thoughtful conclusion. Be helpful and address the intent identified in the first step.

# Guidelines for Report:
- Avoid references to the user, like "the user wants to know..." or the "the user sent in...", as these are obvious.
- Avoid self-references like "I found that..." or "I was unable to..."
- Use impersonal phrasing such as "The message contains..." or "The content suggests..."
- Start with a summary of the content, analyse it, then end with a thoughtful conclusion.

# Other useful information

Date: {datetime}
Remaining searches: {{remaining_searches}}
Remaining screenshots: {{remaining_screenshots}}
Popular types of messages:
    - scams
    - illegal moneylending/gambling
    - marketing content from companies, especially Singapore companies. Note, not all marketing content is necessarily bad, but should be checked for validity.
    - links to news articles
    - links to social media
    - viral messages designed to be forwarded
    - legitimate government communications from agencies or educational institutions
    - OTP messages. Note, while requests by others to share OTPs are likely scams, the OTP messages themselves are not.

Signs that hint at legitimacy:
    - The message is clearly from a well-known, official company, or the government
    - The message asks the user to access a link elsewhere, rather than providing a direct hyperlink
    - The screenshot shows an SMS with an alphanumeric sender ID (as opposed to a phone number). In Singapore, if the alphanumeric sender ID is not <Likely Scam>, it means it has been whitelisted by the authorities
    - Any links hyperlinks come from legitimate domains

Signs that hint at illegitimacy:
    - Messages that use Cialdini's principles (reciprocity, commitment, social proof, authority, liking, scarcity) to manipulate the user
    - Domains are purposesly made to look like legitimate domains
    - Too good to be true

Characteristics of legitimate government communications:
    - Come via SMS from a gov.sg alphanumeric sender ID
    - Contain .gov.sg or .edu.sg links
    - Sometimes may contain go.gov.sg links which is from the official Singapore government link shortener. Do note that in emails or Telegram this could be a fake hyperlink
    - Are in the following format

```Govt SMS Format```
<Full name of agency or service>

---
<Message that does not contain hyperlinks>
---

This is an automated message sent by the Singapore Government.
```End Govt SMS Format```"""

logger = StructuredLogger("agent_generation")


@observe(name="agent_generation")
async def get_outputs(
    text: Union[str, None] = None,
    image_url: Union[str, None] = None,
    caption: Union[str, None] = None,
    addPlanning: bool = False,
    provider: SupportedModelProvider = SupportedModelProvider.OPENAI,
    **kwargs,
):
    langfuse_context.update_current_trace(
        tags=[
            os.environ.get("ENVIRONMENT", "missing"),
            "agent_generation",
            "community_note",
        ]
    )
    child_logger = logger.child(
        model=provider.value,
        text=text,
        image_url=image_url,
        caption=caption,
        addPlanning=addPlanning,
    )
    request_id = request_id_var.get()
    model = None

    try:
        current_datetime = datetime.now()
        if provider == SupportedModelProvider.GEMINI:
            model = "gemini"
            agent = GeminiAgent(
                gemini_client,
                tool_list=[
                    search_google_tool,
                    get_screenshot_tool,
                    check_malicious_url_tool,
                    review_report_tool,
                    plan_next_step_tool,
                    infer_intent_tool,
                ],
                system_prompt=system_prompt.format(
                    datetime=current_datetime.strftime("%d %b %Y")
                ),
                include_planning_step=addPlanning,
                temperature=0.2,
            )
        else:
            openai_client = create_openai_client(provider)
            if provider == SupportedModelProvider.OPENAI:
                model = "gpt-4o"
            elif provider == SupportedModelProvider.DEEPSEEK:
                model = "deepseek-chat"

            agent = OpenAIAgent(
                openai_client,
                tool_list=[
                    search_google_tool,
                    get_screenshot_tool,
                    check_malicious_url_tool,
                    review_report_tool,
                    plan_next_step_tool,
                    infer_intent_tool,
                ],
                system_prompt=system_prompt.format(
                    datetime=current_datetime.strftime("%d %b %Y")
                ),
                include_planning_step=addPlanning,
                temperature=0.2,
                model=model,
            )

        outputs = await agent.generate_note(text, image_url, caption)
        community_note = outputs.get("community_note", None)
        chinese_note = community_note

        if community_note is not None:
            try:
                chinese_note = await translate_text(community_note, language="cn")
            except Exception as e:
                child_logger.error(f"Error in translation: {e}")

        response = SavedAgentCall(
            requestId=request_id,
            success=outputs.get("success", False),
            en=community_note,
            cn=chinese_note,
            links=outputs.get("sources", None),
            isControversial=outputs.get("isControversial", False),
            isVideo=outputs.get("isVideo", False),
            isAccessBlocked=outputs.get("isAccessBlocked", False),
            report=outputs.get("report", None),
            totalTimeTaken=outputs.get("total_time_taken", None),
            agentTrace=outputs.get("agent_trace", None),
            text=text,
            image_url=image_url,
            caption=caption,
            timestamp=current_datetime,
            model=provider,
            environment=os.environ.get("ENVIRONMENT", "missing"),
        )

    except Exception as e:
        child_logger.error(f"Error in generating community note: {e}")
        response = SavedAgentCall(
            requestId=request_id,
            success=False,
            errorMessage=str(e),
            agentTrace=(
                outputs.get("agent_trace", None) if "outputs" in locals() else None
            ),
            text=text,
            image_url=image_url,
            caption=caption,
            timestamp=current_datetime,
            model=provider,  # Fallback to provider value if model is None
            environment=os.environ.get("ENVIRONMENT", "missing"),
        )

    finally:
        if response:
            try:
                doc_ref = db.collection("agent_calls").document(request_id)
                doc_ref.set(response.model_dump())
            except Exception as e:
                child_logger.error(f"Error storing response in Firestore: {e}")

        return response  # Always return response, even if it's an error response


if __name__ == "__main__":
    import asyncio

    text = "https://www.msn.com/en-sg/health/other/china-struggles-with-new-virus-outbreak-five-years-after-covid-pandemic/ss-BB1hj9oL?ocid=nl_article_link"
    result = asyncio.run(get_outputs(text=text))
    # prettify the result
    print(result.report)
    print(result.en)
