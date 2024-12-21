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
from pii_mask import redact
import json

app = FastAPI()

embedding_model = SentenceTransformer('files/all-MiniLM-L6-v2')
L1_svc = joblib.load('files/L1_svc.joblib')

class ItemText(BaseModel):
  text: str

class ItemUrl(BaseModel):
  url: str

@app.post("/embed")
def get_embedding(item: ItemText):
  embedding= embedding_model.encode(item.text)
  return {'embedding': embedding.tolist()}

@app.post("/getL1Category")
def get_L1_category(item: ItemText):
  embedding = embedding_model.encode(item.text)
  prediction = L1_svc.predict(embedding.reshape(1,-1))[0]
  if prediction == "trivial" or prediction == "irrelevant":
    print(f"Message: {item.text} deemed irrelevant and sent to LLM for review")
    should_review = check_should_review(item.text)
    print(f"Message: LLM determined that should_review = {should_review}")
    if should_review:
      prediction = "unsure"
  return {'prediction': "irrelevant" if prediction == "trivial" else prediction}

# @app.post("/ocr")
# def getOCR(item: ItemUrl):
#   output, is_convo, extracted_message, sender = end_to_end(item.url)
#   if extracted_message:
#     embedding = embedding_model.encode(extracted_message)
#     prediction = L1_svc.predict(embedding.reshape(1,-1))[0]
#   else:
#     prediction = "unsure"
#   return {
#     'output': output,
#     'is_convo': is_convo,
#     'extracted_message': extracted_message,
#     'sender': sender,
#     'prediction': "irrelevant" if prediction == "trivial" else prediction,
#   }

@app.post("/ocr-v2")
def get_ocr(item: ItemUrl):
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

@app.post("/redact")
def get_redact(item: ItemText):
  response, tokens_used = redact(item.text)
  print(f'Tokens used: {tokens_used}')

  try:
    response_dict = json.loads(response)
    redacted_message = item.text
    for redaction in response_dict["redacted"]:
        redacted_text = redaction["text"]
        replacement = redaction["replaceWith"]
        redacted_message = redacted_message.replace(redacted_text, replacement)
    return {'redacted': redacted_message, 'original': item.text, 'tokens_used': tokens_used, 'reasoning': response_dict['reasoning']}

  except Exception as e:
    print(f'Error: {e}')
    return {'redacted': '', 'original': item.text, 'tokens_used': tokens_used, 'reasoning': ''}
    