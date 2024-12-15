from dotenv import load_dotenv

load_dotenv()

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
# from ocr import end_to_end
from ocr_v2 import perform_ocr
from trivial_filter import check_should_review
from community_note import generate_community_note
from fastapi import HTTPException
import datetime
from sensitive_filter import check_is_sensitive


app = FastAPI()

embedding_model = SentenceTransformer('files/all-MiniLM-L6-v2')
L1_svc = joblib.load('files/L1_svc.joblib')

class ItemText(BaseModel):
  text: str

class ItemUrl(BaseModel):
  url: str

# class NoteText(BaseModel):
#     text: str

# class NoteImage(BaseModel):
#     image_url: str
#     caption: str = None


class CommunityNoteRequest(BaseModel):
    text: str = Field(default=None, description="Text content for generating a community note")
    image_url: str = Field(default=None, description="Image URL for generating a community note")
    caption: str = Field(default=None, description="Caption for the image (optional)")


@app.post("/embed")
def get_embedding(item: ItemText):
  embedding= embedding_model.encode(item.text)
  return {'embedding': embedding.tolist()}

@app.post("/getL1Category")
def get_L1_category(item: ItemText):
  embedding = embedding_model.encode(item.text)
  prediction = L1_svc.predict(embedding.reshape(1,-1))[0]
  print(f"Prediction: {prediction}")
  if prediction == "trivial" or prediction == "irrelevant":
    print(f"Message: {item.text} deemed irrelevant and sent to LLM for review")
    should_review = check_should_review(item.text)
    print(f"Message: LLM determined that should_review = {should_review}")
    return {'needsChecking': should_review}
  else:
     return {'needsChecking': True}

@app.post("/sensitivity-filter")
def get_sensitivity(item: ItemText):
  is_sensitive = check_is_sensitive(item.text)
  return {'is_sensitive': is_sensitive}
   

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

@app.post("/generate-community-note")
async def generate_community_note_endpoint(request: CommunityNoteRequest):
    try:
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if request.text:
            result = await generate_community_note(session_id, data_type="text", text=request.text)
        elif request.image_url:
            result = await generate_community_note(session_id, data_type="image", image_url=request.image_url, caption=request.caption)
        else:
            raise HTTPException(status_code=400, detail="Either 'text' or 'image_url' must be provided.")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
