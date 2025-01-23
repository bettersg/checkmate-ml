from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langfuse.decorators import observe, langfuse_context
import json
import os
from context import request_id_var  # Import the context variable
from clients.firestore_db import db
from langfuse import Langfuse
from logger import StructuredLogger

# Initialize ChatOpenAI and Langfuse
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed=11)
langfuse = Langfuse()

# Retrieve prompt from Langfuse
trivial_prompt = langfuse.get_prompt("trivial_prompt", label="production")
assert trivial_prompt, "Failed to retrieve prompt from Langfuse."

# Convert retrieved JSON string to dictionary
try:
    prompt = json.loads(trivial_prompt.prompt)
except json.JSONDecodeError as e:
    raise ValueError(f"Failed to decode the retrieved prompt: {e}")

# Validate the structure of the prompt
required_keys = ["system", "examples", "human_template", "ai_template"]
for key in required_keys:
    if key not in prompt:
        raise ValueError(f"Missing '{key}' in the retrieved prompt.")

# Extract prompt details
system_message = prompt["system"]
few_shot_examples = prompt["examples"]
human_template = prompt["human_template"]
ai_template = prompt["ai_template"]

# Build prompt templates
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
ai_prompt_template = AIMessagePromptTemplate.from_template(ai_template)
messages = [SystemMessage(content=system_message)]

# Add few-shot examples to messages
for example in few_shot_examples:
    if "message" in example and "reasoning" in example and "to_review" in example:
        messages.append(human_prompt_template.format(message=example["message"]))
        messages.append(
            ai_prompt_template.format(
                reasoning=example["reasoning"], to_review=example["to_review"]
            )
        )
    else:
        raise ValueError(f"Invalid example format: {example}")

# Add the final human prompt
messages.append(human_prompt_template)

# Construct the ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_messages(messages)
parser = JsonOutputParser()
chain = chat_prompt_template | llm | parser

logger = StructuredLogger("trivial_filter")


@observe(name="trivial_filter")
def check_should_review(message, **kwargs):
    """
    Checks if a message should be reviewed.
    """
    child_logger = logger.child(message=message)
    langfuse_context.update_current_trace(
        tags=[
            os.getenv("ENVIRONMENT", "missing_environment"),
            "trivial_filter",
            "single_call",
        ]
    )
    request_id = request_id_var.get()
    doc_ref = db.collection("trivial_filters").document(request_id)

    try:
        # Invoke the chain to process the message
        response = chain.invoke({"message": message})
        result = response.get("to_review", True)

        # Attempt to store in Firestore, but don't block on failure
        try:
            doc_ref.set(
                {
                    "messageToCheck": message,
                    "success": True,
                    "response": response,
                }
            )
        except Exception as firestore_error:
            child_logger.error("Error saving to Firestore:", firestore_error)

        return result

    except Exception as e:
        error_message = str(e)
        child_logger.error("Error occurred in the processing chain:", error_message)

        # Attempt to store error in Firestore, but don't block on failure
        try:
            doc_ref.set(
                {
                    "messageToCheck": message,
                    "success": False,
                    "error": error_message,
                }
            )
        except Exception as firestore_error:
            child_logger.error("Error saving to Firestore:", firestore_error)

        return False
