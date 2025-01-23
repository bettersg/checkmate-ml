import vertexai
from vertexai import generative_models
import json
import requests
import os
from langfuse.decorators import observe, langfuse_context
from context import request_id_var  # Import the context variable
from clients.firestore_db import db
from logger import StructuredLogger

logger = StructuredLogger("ocr_extraction")


def get_project_id():
    url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(url, headers=headers)
    project_id = response.text
    return project_id


REGION = "asia-southeast1"

if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
    PROJECT_ID = get_project_id()
    vertexai.init(project=PROJECT_ID, location=REGION)
else:
    vertexai.init(location=REGION)

multimodal_model = generative_models.GenerativeModel("gemini-1.5-pro")

# Model config
model_config = {"temperature": 0}

# Safety config
safety_config = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# Prompts
prompts = json.load(open("files/prompts.json"))

prompt = prompts.get("ocr-v2", {}).get("system", None)

if prompt is None:
    raise Exception("Prompt not found in prompts.json!")


@observe(name="ocr_extraction")
def perform_ocr(img_url, **kwargs):
    """
    function to perform OCR on the image
    """
    child_logger = logger.child(img_url=img_url)
    langfuse_context.update_current_trace(
        tags=[os.environ.get("ENVIRONMENT", "missing"), "ocr", "single_call"]
    )
    request_id = request_id_var.get()
    doc_ref = db.collection("ocr_extractions").document(request_id)

    try:
        image = generative_models.Part.from_uri(img_url, mime_type="image/jpeg")
        response = multimodal_model.generate_content(
            [prompt, image],
            generation_config=model_config,
            safety_settings=safety_config,
        )

        generated_text = response.text
        # strip everything before the first '{' and after the last '}'
        generated_text = generated_text[generated_text.find("{") :]
        generated_text = generated_text[: generated_text.rfind("}") + 1]

        return_dict = json.loads(generated_text)
        # Validate response structure
        assert "image_type" in return_dict
        assert "sender" in return_dict
        assert "subject" in return_dict
        assert "extracted_message" in return_dict

        if return_dict["image_type"] not in ["email", "convo", "letter", "others"]:
            return_dict["image_type"] = "others"

        # Attempt to store in Firestore, but don't block on failure
        try:
            doc_ref.set({"imageUrl": img_url, "success": True, "response": return_dict})
        except Exception as firestore_error:
            child_logger.error("Error saving to Firestore:", firestore_error)

        return return_dict

    except Exception as e:
        error_message = str(e)
        child_logger.error("Error in OCR extraction:", error_message)

        # Attempt to store error in Firestore, but don't block on failure
        try:
            doc_ref.set({"imageUrl": img_url, "success": False, "error": error_message})
        except Exception as firestore_error:
            child_logger.error("Error saving to Firestore:", str(firestore_error))

        return {
            "image_type": None,
            "sender": None,
            "subject": None,
            "extracted_message": None,
        }
