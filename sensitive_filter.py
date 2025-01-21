import json
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langfuse.openai import openai
from langfuse import Langfuse


langfuse = Langfuse()

# Retrieve prompt from Langfuse
sensitivity_prompt = langfuse.get_prompt("sensitivity_prompt", label="production")

# convert str to dict
prompt = json.loads(sensitivity_prompt.prompt)


# Validate the retrieved prompt
assert sensitivity_prompt, "Prompt could not be retrieved from Langfuse."
assert "system" in prompt, "System message is missing in the prompt."
assert "examples" in prompt, "Few-shot examples are missing in the prompt."
assert "human_template" in prompt, "Human template is missing in the prompt."
assert "ai_template" in prompt, "AI template is missing in the prompt."

# Extract details from the retrieved prompt
system_message = prompt["system"]
few_shot_examples = prompt["examples"]
human_template = prompt["human_template"]
ai_template = prompt["ai_template"]


# Define LLM model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=11)

# Build prompt messages
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
ai_prompt_template = AIMessagePromptTemplate.from_template(ai_template)
messages = [SystemMessage(content=system_message)]

for example in few_shot_examples:
    messages.append(human_prompt_template.format(message=example.get("message")))
    messages.append(
        ai_prompt_template.format(
            reasoning=example.get("reasoning"), is_sensitive=example.get("is_sensitive")
        )
    )

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
