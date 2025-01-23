from clients.openai import openai_client
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import os
from context import request_id_var  # Import the context variable
from clients.firestore_db import db
from logger import StructuredLogger

langfuse = Langfuse()

logger = StructuredLogger("pii_masking")


@observe(name="PII Masking")
def redact(text, **kwargs):
    """
    Redacts PII information from the given text.
    """
    child_logger = logger.child(text=text)
    langfuse_context.update_current_trace(
        tags=[
            os.environ.get("ENVIRONMENT", "missing"),
            "single_call",
            "pii_masking",
        ]
    )
    request_id = request_id_var.get()
    doc_ref = db.collection("pii_masks").document(request_id)

    try:
        prompt = langfuse.get_prompt("message_redaction", label="prod")
        system_message = prompt.compile()
        prompt_messages = [{"role": "system", "content": system_message}]

        for example in prompt.config["examples"]:
            prompt_messages.append({"role": "user", "content": example["user"]})
            prompt_messages.append(
                {"role": "assistant", "content": example["assistant"]}
            )

        prompt_messages.append({"role": "user", "content": text})

        response = openai_client.chat.completions.create(
            model=prompt.config["model"],
            messages=prompt_messages,
            temperature=prompt.config["temperature"],
            seed=11,
            langfuse_prompt=prompt,
        )

        result = (response.choices[0].message.content, response.usage.total_tokens)

        # Attempt to store in Firestore, but don't block on failure
        try:
            doc_ref.set(
                {
                    "originalText": text,
                    "success": True,
                    "response": result[0],
                    "tokensUsed": result[1],
                }
            )
        except Exception as firestore_error:
            child_logger.error("Error saving to Firestore:", firestore_error)

        return result

    except Exception as e:
        error_message = str(e)
        child_logger.error("Error in PII masking:", error_message)

        # Attempt to store error in Firestore, but don't block on failure
        try:
            doc_ref.set(
                {"originalText": text, "success": False, "error": error_message}
            )
        except Exception as firestore_error:
            child_logger.error("Error saving to Firestore:", str(firestore_error))

        return "", 0
