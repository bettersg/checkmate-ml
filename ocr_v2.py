import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
import json
import requests
import os

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
    vertexai.init()

multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

# Model config
model_config = {"temperature": 0}

# Safety config
safety_config = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

# Prompts
prompts = json.load(open("files/prompts.json"))

prompt = prompts.get("ocr-v2", {}).get("system",None)

if prompt is None:
    raise Exception("Prompt not found in prompts.json!")

def perform_ocr(img_url):
    """
    function to perform OCR on the image
    """
    image = Part.from_uri(img_url, mime_type="image/jpeg")
    response = multimodal_model.generate_content(
        [prompt, image], 
        generation_config=model_config,
        safety_settings=safety_config, 
    )

    try:
        generated_text = response.text
    except Exception as e:
        print("Error parsing Gemini response:", e)
        return {
            "image_type": None,
            "sender": None,
            "subject": None,
            "extracted_message": None
        }
    
    #strip everything before the first '{' and after the last '}'
    generated_text = generated_text[generated_text.find("{"):]
    generated_text = generated_text[:generated_text.rfind("}")+1]
    try:
        return_dict = json.loads(generated_text)
        assert "image_type" in return_dict
        assert "sender" in return_dict
        assert "subject" in return_dict
        assert "extracted_message" in return_dict
        if return_dict["image_type"] not in ["email", "convo", "letter", "others"]:
            return_dict["image_type"] = "others"
        return return_dict
    except Exception as e:
        print(f"Generated string: {generated_text}")
        print("Error processing JSON data", e)
        return {
            "image_type": None,
            "sender": None,
            "subject": None,
            "extracted_message": None
        }