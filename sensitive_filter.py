import json
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import SystemMessage
from langchain_openai.chat_models import ChatOpenAI

# Load prompts for political/religious filter
prompts = json.load(open("files/prompts.json"))
relevant_prompts = prompts.get("sensitive-filter", {})

# Validate keys exist in the JSON structure
system_message = relevant_prompts.get("system", "")
assert system_message
few_shot_examples = relevant_prompts.get("examples", [])
assert len(few_shot_examples) > 0
human_template = relevant_prompts.get("human_template", "")
assert human_template
ai_template = relevant_prompts.get("ai_template", "")
assert ai_template

# Define LLM model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    seed=11
)

# Build prompt messages
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
ai_prompt_template = AIMessagePromptTemplate.from_template(ai_template)
messages = [SystemMessage(content=system_message)]

for example in few_shot_examples:
    messages.append(human_prompt_template.format(message=example.get("message")))
    messages.append(ai_prompt_template.format(
        reasoning=example.get("reasoning"),
        is_sensitive=example.get("is_sensitive")
    ))

messages.append(human_prompt_template)
chat_prompt_template = ChatPromptTemplate.from_messages(messages)
parser = JsonOutputParser()

# Combine into a chain
chain = chat_prompt_template | llm | parser

# Define function to check for sensitive content
def check_is_sensitive(message):
    try:
        response = chain.invoke({"message": message})
        print(response)
        return response.get("is_sensitive", False)
    except Exception as e:
        print("Error occurred in chain", e)
        return False
