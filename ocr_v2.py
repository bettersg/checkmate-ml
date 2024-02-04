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

prompt = """You are an OCR bot. You will extract the text content from the provided image. \
If the image is a screenshot of a chat window, such as SMS/Whatsapp/Messenger, then you should only extract the text message in chat bubbles from the sender (not the receipient), and you should not include other system or display texts. Please also extract the sender's information (name, number, or user ID). \
If the image is a screenshot of an email, then you should only extract the text message in the main email body. Please also extract the sender's information (name or email address), and the email subject. \
If the image is a picture of a letter, then you should only extract the text message in the main letter body. Please also extract the sender's information, and the letter subject. \
If the image is none of the above, then you should extract all text content with meaningful paragraphing. \
Provide your output in JSON, with the following keys: \
\n- "image_type" (string): one of ["convo", "email", "letter", "others"]
\n- "sender" (string or None): sender information extracted for "image_type" = "convo", "email" or "letter"
\n- "subject" (string or None): subject  extracted for "image_type" = "email" or "letter"
\n- "extracted_message" (string): your extracted text content
"""

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