from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import SystemMessage
from langchain_openai.chat_models import ChatOpenAI
import json

llm = ChatOpenAI(
    model = "gpt-3.5-turbo-0125",
    temperature = 0,
    model_kwargs = {
        "seed": 11
    }
)

prompts = json.load(open("files/prompts.json"))
relevant_prompts = prompts.get("trivial-filter",{})

system_message = relevant_prompts.get("system","")
assert system_message
few_shot_examples = relevant_prompts.get("examples",{})
assert len(few_shot_examples) > 0
human_template = relevant_prompts.get("human_template","")
assert human_template
ai_template = relevant_prompts.get("ai_template","")
assert ai_template

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
        return response.get("to_review", False)
    except Exception as e:
        print("Error occured in chain", e)
        return False
  
