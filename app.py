from dotenv import load_dotenv

load_dotenv()

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
# from ocr import end_to_end
from ocr_v2 import perform_ocr
from trivial_filter import check_should_review

app = FastAPI()

embedding_model = SentenceTransformer('files/all-MiniLM-L6-v2')
L1_svc = joblib.load('files/L1_svc.joblib')

class ItemText(BaseModel):
  text: str

class ItemUrl(BaseModel):
  url: str

@app.post("/embed")
def get_embedding(item: ItemText):
  """
  Given a text message, returns its corresponding sentence embedding as a list.

  Args:
    item (ItemText): The text message to be embedded.

  Returns:
    dict: A dictionary with a single key 'embedding' containing the embedding as a list of floats.
  """
  embedding= embedding_model.encode(item.text)
  return {'embedding': embedding.tolist()}

@app.post("/getL1Category")
def get_L1_category(item: ItemText):
  """
  Given a text message, returns its corresponding L1 category.

  Args:
    item (ItemText): The text message to be classified.

  Returns:
    dict: A dictionary with a single key 'prediction' containing the L1 category as a string.
           The strings can be one of "scam", "illicit", "spam", "info", "irrelevant", "unsure".
  """
  embedding = embedding_model.encode(item.text)
  prediction = L1_svc.predict(embedding.reshape(1,-1))[0]
  if prediction == "trivial" or prediction == "irrelevant":
    print(f"Message: {item.text} deemed irrelevant and sent to LLM for review")
    # additional review cos there were quite a number of false positives for irrelevant category, so we added another layer of filtering
    should_review = check_should_review(item.text)
    print(f"Message: LLM determined that should_review = {should_review}")
    if should_review:
      prediction = "unsure"
  return {'prediction': "irrelevant" if prediction == "trivial" else prediction}

@app.post("/ocr-v2")
def get_ocr(item: ItemUrl):
  """
  Given a URL pointing to an image, runs the GenAI OCR pipeline on it, and returns the extracted text message and its corresponding L1 category.

  Args:
    item (ItemUrl): The URL pointing to the image to be processed.

  Returns:
    dict: A dictionary containing the following keys:
      - 'extracted_message': The extracted text message.
      - 'prediction': The L1 category of the text message.
      - 'image_type': The type of the image (email, convo, letter, others)
      - 'sender': The sender of the message.
      - 'subject': The subject of the message, if email.
  """
  print(f"GenAI OCR called on {item.url}")
  results = perform_ocr(item.url)
  if "extracted_message" in results and results["extracted_message"]:
    extracted_message = results["extracted_message"]
    print(f"Extracted message: {extracted_message}")
    prediction = get_L1_category(ItemText(text=extracted_message)).get("prediction","unsure")
    results["prediction"] = prediction
  else:
    print(f"No extracted message in results")
    results["prediction"] = "unsure"
  return results