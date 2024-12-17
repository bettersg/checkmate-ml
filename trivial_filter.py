from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import SystemMessage
from langchain_openai.chat_models import ChatOpenAI
import json

from langfuse.openai import openai
from langfuse import Langfuse

llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0,
    seed = 11
)

langfuse = Langfuse()

# Retrieve prompt from Langfuse
trivial_prompt = langfuse.get_prompt("trivial_prompt", label="production")

# convert str to dict
prompt = json.loads(trivial_prompt.prompt)


# Validate the retrieved prompt
assert prompt, "Prompt could not be retrieved from Langfuse."
assert "system" in prompt, "System message is missing in the prompt."
assert "examples" in prompt, "Few-shot examples are missing in the prompt."
assert "human_template" in prompt, "Human template is missing in the prompt."
assert "ai_template" in prompt, "AI template is missing in the prompt."

# Extract details from the retrieved prompt
system_message = prompt["system"]
few_shot_examples = prompt["examples"]
human_template = prompt["human_template"]
ai_template = prompt["ai_template"]

# Build prompt messages
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
ai_prompt_template = AIMessagePromptTemplate.from_template(ai_template)
messages = [SystemMessage(content=system_message)]

for example in few_shot_examples:
  messages.append(human_prompt_template.format(
      message = example.get("message")
  ))
  messages.append(ai_prompt_template.format(
      reasoning=example.get("reasoning"),
      to_review=example.get("to_review")
  ))

messages.append(human_prompt_template)

chat_prompt_template = ChatPromptTemplate.from_messages(messages)
parser = JsonOutputParser()

chain = chat_prompt_template | llm | parser

def check_should_review(message):
    try:
        response = chain.invoke({"message": message})
        # print("response:", response)
        return response.get("to_review", True)
    except Exception as e:
        print("Error occured in chain", e)
        return False
  
