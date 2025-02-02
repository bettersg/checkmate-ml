from langfuse.decorators import observe, langfuse_context
import json
import os
from context import request_id_var  # Import the context variable
from clients.firestore_db import db
from langfuse import Langfuse
from logger import StructuredLogger
from clients.openai import create_openai_client

# Initialize ChatOpenAI and Langfuse
client = create_openai_client("openai")
langfuse = Langfuse()

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
        prompt = langfuse.get_prompt("trivial_filter", label=os.getenv("ENVIRONMENT"))
        compiled_prompt = prompt.compile(message=message)
        config = prompt.config

        response = client.chat.completions.create(
            model=config.get("model", "gpt-4o"),
            messages=compiled_prompt,
            temperature=config.get("temperature", 0),
            seed=config.get("seed", 11),
            response_format=config["response_format"],
            langfuse_prompt=prompt,
        )

        json_output = json.loads(response.choices[0].message.content)

        result = json_output.get("needs_checking", True)

        # Attempt to store in Firestore, but don't block on failure
        try:
            doc_ref.set(
                {
                    "messageToCheck": message,
                    "success": True,
                    "response": json_output,
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

        return True
