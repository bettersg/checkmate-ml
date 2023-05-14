from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = torch.load('models/sentence-transformers-all-MiniLM-L6-v2.pt')

class Item(BaseModel):
    text: str

@app.post("/embed")
async def create_item(item: Item):
    embeddings = model.encode(item.text)
    return {'embeddings': embeddings.tolist()}