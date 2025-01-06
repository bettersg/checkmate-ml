# tools/dummy_tools.py
# Tools that simply reflect the inputs back to the agent


async def plan_next_step(reasoning, next_step):
    return {"result": {"reasoning": reasoning, "next_step": next_step}}


plan_next_step_definition = dict(
    name="plan_next_step",
    description="Indicate what the next step should be, given past steps.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "reasoning": {
                "type": "STRING",
                "description": "Why you think the next step should be what it is.",
            },
            "next_step": {
                "type": "STRING",
                "description": "What the next step should be. You can either search_google to investigate claims, get_website_screenshot to see the contents of a link, or submit_community_note once you have enough information to complete your task. Avoid performing more than 5 searches or visiting more than 5 sites.",
                "enum": [
                    "search_google",
                    "get_website_screenshot",
                    "submit_community_note",
                ],
            },
        },
        "required": ["reasoning", "next_step"],
    },
)


async def infer_intent(reasoning, intent):
    return {"result": {"reasoning": reasoning, "intent": intent}}


infer_intent_definition = dict(
    name="infer_intent",
    description="Infer the user's intent.",
    parameters={
        "type": "OBJECT",
        "properties": {
            "reasoning": {
                "type": "STRING",
                "description": "The reasoning behind your choice",
            },
            "intent": {
                "type": "STRING",
                "description": "What the user's intent is, e.g. to check whether this is a scam, to check if this is really from the government, to check the facts in this article, etc.",
                "example": "The user intends to check whether this is a legitimate message sent from the government.",
            },
        },
        "required": ["reasoning", "intent"],
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
