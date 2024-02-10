import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import json
import requests

def get_project_id():
    url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
    headers = {"Metadata-Flavor": "Google"}
    response = requests.get(url, headers=headers)
    project_id = response.text
    return project_id

PROJECT_ID = get_project_id()
REGION = "asia-southeast1"
vertexai.init(project=PROJECT_ID, location=REGION)

multimodal_model = GenerativeModel("gemini-pro-vision")

prompts = json.load(open("files/prompts.json"))

prompt = prompts.get("ocr-v2", {}).get("system",None)

if prompt is None:
    raise Exception("Prompt not found in prompts.json!")

def perform_ocr(img_url):
    """
    function to perform OCR on the image
    """
    image = Part.from_uri(img_url, mime_type="image/jpeg")
    response = multimodal_model.generate_content([prompt, image])
    generated_text = response.text
    #strip everything before the first '{' and after the last '}'
    generated_text = generated_text[generated_text.find("{"):]
    generated_text = generated_text[:generated_text.rfind("}")+1]
    try:
        return_dict = json.loads(generated_text)
        assert "image_type" in return_dict
        assert "sender" in return_dict
        assert "subject" in return_dict
        assert "extracted_message" in return_dict
        return return_dict
    except Exception as e:
        print("Error:", e)
        return {
            "image_type": None,
            "sender": None,
            "subject": None,
            "extracted_message": None
        }