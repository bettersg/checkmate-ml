import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ocr import end_to_end

app = FastAPI()

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
representative_embeddings = np.load("files/representative_embeddings.npy")
trivial_svc = joblib.load('files/trivial_svc.joblib')
L1_svc = joblib.load('files/L1_svc.joblib')

class ItemText(BaseModel):
  text: str

class ItemUrl(BaseModel):
  url: str

@app.post("/embed")
async def get_embedding(item: ItemText):
  embedding= embedding_model.encode(item.text)
  return {'embedding': embedding.tolist()}

@app.post("/getL1Category")
async def getL1Category(item: ItemText):
  embedding = embedding_model.encode(item.text)
  prediction = L1_svc.predict(embedding.reshape(1,-1))[0]
  return {'prediction': "irrelevant" if prediction == "trivial" else prediction}

@app.post("/ocr")
async def getOCR(item: ItemUrl):
  output, is_convo, extracted_message, sender = end_to_end(item.url)
  if extracted_message:
    embedding = embedding_model.encode(extracted_message)
    prediction = L1_svc.predict(embedding.reshape(1,-1))[0]
  else:
    prediction = "unsure"
  return {
    'output': output,
    'is_convo': is_convo,
    'extracted_message': extracted_message,
    'sender': sender,
    'prediction': "irrelevant" if prediction == "trivial" else prediction,
  }