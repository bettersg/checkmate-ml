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
from typing import Optional
from pii_mask import redact
import json
from gemini_generation import get_outputs
from openai_generation import get_outputs as get_openai_outputs
from models import CommunityNoteRequest, AgentResponse, SupportedModelProvider
from middleware import RequestIDMiddleware  # Import the middleware
from context import request_id_var  # Import the context variable

app = FastAPI()

# Add the middleware to the application
app.add_middleware(RequestIDMiddleware)

embedding_model = SentenceTransformer("files/all-MiniLM-L6-v2")
L1_svc = joblib.load("files/L1_svc.joblib")


class ItemText(BaseModel):
    text: str


class ItemUrl(BaseModel):
    url: str


# class NoteText(BaseModel):
#     text: str

# class NoteImage(BaseModel):
#     image_url: str
#     caption: str = None


@app.post("/embed")
def get_embedding(item: ItemText):
    embedding = embedding_model.encode(item.text)
    return {"embedding": embedding.tolist()}


@app.post("/getL1Category")
def get_L1_category(item: ItemText):
    embedding = embedding_model.encode(item.text)
    prediction = L1_svc.predict(embedding.reshape(1, -1))[0]
    print(f"Prediction: {prediction}")
    return {"prediction": "irrelevant" if prediction == "trivial" else prediction}


@app.post("/sensitivity-filter")
def get_sensitivity(item: ItemText):
    is_sensitive = check_is_sensitive(item.text)
    return {"is_sensitive": is_sensitive}


@app.post("/getNeedsChecking")
def get_needs_checking(item: ItemText):
    should_review = check_should_review(item.text)
    return {"needsChecking": should_review}


@app.post("/ocr-v2")
def get_ocr(item: ItemUrl):
    print(f"GenAI OCR called on {item.url}")
    results = perform_ocr(item.url)
    if "extracted_message" in results and results["extracted_message"]:
        extracted_message = results["extracted_message"]
        print(f"Extracted message: {extracted_message}")
        prediction = get_L1_category(ItemText(text=extracted_message)).get(
            "prediction", "unsure"
        )
        results["prediction"] = prediction
    else:
        print(f"No extracted message in results")
        results["prediction"] = "unsure"
    return results


@app.post("/redact")
def get_redact(item: ItemText):
    response, tokens_used = redact(item.text)
    print(f"Tokens used: {tokens_used}")
    try:
        response_dict = json.loads(response)
        redacted_message = item.text
        for redaction in response_dict["redacted"]:
            redacted_text = redaction["text"]
            replacement = redaction["replaceWith"]
            redacted_message = redacted_message.replace(redacted_text, replacement)
        return {
            "redacted": redacted_message,
            "original": item.text,
            "tokens_used": tokens_used,
            "reasoning": response_dict["reasoning"],
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "redacted": "",
            "original": item.text,
            "tokens_used": tokens_used,
            "reasoning": "Error in redact function",
        }


@app.post("/getCommunityNote")
async def generate_community_note_endpoint(request: CommunityNoteRequest):
    try:
        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if request.text:
            result = await generate_community_note(
                session_id, data_type="text", text=request.text
            )
        elif request.image_url:
            result = await generate_community_note(
                session_id,
                data_type="image",
                image_url=request.image_url,
                caption=request.caption,
            )
        else:
            raise HTTPException(
                status_code=400, detail="Either 'text' or 'image_url' must be provided."
            )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/getCommunityNote")
async def get_gemini_note(
    request: CommunityNoteRequest,
    provider: SupportedModelProvider = SupportedModelProvider.GEMINI,
) -> AgentResponse:
    try:
        if request.text is None and request.image_url is None:
            raise HTTPException(
                status_code=400, detail="Either 'text' or 'image_url' must be provided."
            )
        if request.text is not None and request.image_url is not None:
            raise HTTPException(
                status_code=400,
                detail="Only one of 'text' or 'image_url' should be provided.",
            )
        print(provider)
        if (
            provider == SupportedModelProvider.OPENAI
            or provider == SupportedModelProvider.DEEPSEEK
        ):
            return await get_openai_outputs(
                text=request.text,
                image_url=request.image_url,
                caption=request.caption,
                addPlanning=request.addPlanning,
                provider=provider,
            )
        elif provider == SupportedModelProvider.GEMINI:
            return await get_outputs(
                text=request.text,
                image_url=request.image_url,
                caption=request.caption,
                addPlanning=request.addPlanning,
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported model provider: {provider}"
            )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2a/getCommunityNote")
async def get_openai_note(request: CommunityNoteRequest) -> AgentResponse:
    try:
        if request.text is None and request.image_url is None:
            raise HTTPException(
                status_code=400, detail="Either 'text' or 'image_url' must be provided."
            )
        if request.text is not None and request.image_url is not None:
            raise HTTPException(
                status_code=400,
                detail="Only one of 'text' or 'image_url' should be provided.",
            )
        elif request.text:
            return await get_openai_outputs(
                text=request.text, addPlanning=request.addPlanning
            )
        elif request.image_url:
            return await get_openai_outputs(
                image_url=request.image_url,
                caption=request.caption,
                addPlanning=request.addPlanning,
            )
        else:
            raise HTTPException(
                status_code=400, detail="Either 'text' or 'image_url' must be provided."
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
