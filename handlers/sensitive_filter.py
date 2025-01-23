import json
import os
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langfuse.decorators import observe, langfuse_context
from clients.firestore_db import db
from langfuse import Langfuse
from logger import StructuredLogger

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
from context import request_id_var  # Import the context variable

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
logger = StructuredLogger("sensitive_filter")


# Define function to check for sensitive content
@observe(name="sensitive_filter")
def check_is_sensitive(message, **kwargs):
    """
    Checks if a message contains sensitive content.
    """
    child_logger = logger.child(message=message)
    langfuse_context.update_current_trace(
        tags=[
            os.environ.get("ENVIRONMENT", "missing"),
            "sensitive_filter",
            "single_call",
        ]
    )
    request_id = request_id_var.get()  # Fetch the current request ID

    # Prepare Firestore document reference
    doc_ref = db.collection("check_controversial").document(request_id)

    try:
        # Invoke the chain to process the message
        response = chain.invoke({"message": message})
        result = response.get("is_sensitive", False)

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
        # Log failure to Firestore
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
