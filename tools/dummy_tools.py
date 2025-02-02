# tools/dummy_tools.py
# Tools that simply reflect the inputs back to the agent
from collections import OrderedDict
from langfuse.decorators import observe


@observe()
async def plan_next_step(articulation, next_step):
    return {"result": {"reasoning": articulation, "next_step": next_step}}


plan_next_step_definition = dict(
    name="plan_next_step",
    description="Indicate what the next step should be, given past steps.",
    parameters={
        "type": "OBJECT",
        "properties": OrderedDict(
            [
                (
                    "articulation",
                    {
                        "type": "STRING",
                        "description": "Articulate the reasoning why you think the next step should be what it is.",
                    },
                ),
                (
                    "next_step",
                    {
                        "type": "STRING",
                        "description": "What the next step should be. You can either search_google to investigate claims, get_website_screenshot to see the contents of a link, or submit_community_note once you have enough information to complete your task. Avoid performing more than 5 searches or visiting more than 5 sites.",
                        "enum": [
                            "search_google",
                            "get_website_screenshot",
                            "submit_community_note",
                        ],
                    },
                ),
            ]
        ),
        "required": ["articulation", "next_step"],
    },
)


@observe()
async def infer_intent(articulation, intent):
    return {"result": {"reasoning": articulation, "intent": intent}}


infer_intent_definition = dict(
    name="infer_intent",
    description="Infer the user's intent.",
    parameters={
        "type": "OBJECT",
        "properties": OrderedDict(
            [
                (
                    "articulation",
                    {
                        "type": "STRING",
                        "description": "Articulate the reasoning behind your choice",
                    },
                ),
                (
                    "intent",
                    {
                        "type": "STRING",
                        "description": "What the user's intent is, e.g. to check whether this is a scam, to check if this is really from the government, to check the facts in this article, etc.",
                        "example": "The user intends to check whether this is a legitimate message sent from the government.",
                    },
                ),
            ]
        ),
        "required": ["articulation", "intent"],
    },
)

plan_next_step_tool = {
    "function": plan_next_step,
    "definition": plan_next_step_definition,
}

infer_intent_tool = {
    "function": infer_intent,
    "definition": infer_intent_definition,
}
