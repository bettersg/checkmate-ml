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
from logger import StructuredLogger

logger = StructuredLogger("checkmate-ml-api")

app = FastAPI()

# Add the middleware to the application
app.add_middleware(RequestIDMiddleware)

embedding_model = SentenceTransformer("files/all-MiniLM-L6-v2")
L1_svc = joblib.load("files/L1_svc.joblib")


class ItemText(BaseModel):
    text: str


class ItemUrl(BaseModel):
    url: str


@app.post("/embed")
def get_embedding(item: ItemText):
    logger.info("Processing embedding request", text=item.text[:100])
    embedding = embedding_model.encode(item.text)
    logger.info("Embedding generated successfully")
    return {"embedding": embedding.tolist()}


@app.post("/getL1Category")
def get_L1_category(item: ItemText):
    logger.info("Processing L1 category request", text=item.text[:100])
    embedding = embedding_model.encode(item.text)
    prediction = L1_svc.predict(embedding.reshape(1, -1))[0]
    logger.info("Generated L1 category prediction", prediction=prediction)
    return {"prediction": "irrelevant" if prediction == "trivial" else prediction}


@app.post("/sensitivity-filter")
def get_sensitivity(item: ItemText):
    logger.info("Processing sensitivity filter request", text=item.text[:100])
    is_sensitive = check_is_sensitive(item.text)
    logger.info("Sensitivity check complete", is_sensitive=is_sensitive)
    return {"is_sensitive": is_sensitive}


@app.post("/getNeedsChecking")
def get_needs_checking(item: ItemText):
    logger.info("Processing needs checking request", text=item.text[:100])
    should_review = check_should_review(item.text)
    logger.info("Review check complete", needs_checking=should_review)
    return {"needsChecking": should_review}


@app.post("/ocr-v2")
def get_ocr(item: ItemUrl):
    logger.info("Processing OCR request", url=item.url)
    results = perform_ocr(item.url)
    if "extracted_message" in results and results["extracted_message"]:
        extracted_message = results["extracted_message"]
        logger.info("Message extracted from image", message=extracted_message[:100])
        prediction = get_L1_category(ItemText(text=extracted_message)).get(
            "prediction", "unsure"
        )
        results["prediction"] = prediction
    else:
        logger.info("No message extracted from image", url=item.url)
        results["prediction"] = "unsure"
    logger.info("OCR processing complete", prediction=results["prediction"])
    return results


@app.post("/redact")
def get_redact(item: ItemText):
    logger.info("Processing redaction request", text=item.text[:100])
    response, tokens_used = redact(item.text)
    try:
        response_dict = json.loads(response)
        redacted_message = item.text
        for redaction in response_dict["redacted"]:
            redacted_text = redaction["text"]
            replacement = redaction["replaceWith"]
            redacted_message = redacted_message.replace(redacted_text, replacement)
        result = {
            "redacted": redacted_message,
            "original": item.text,
            "tokens_used": tokens_used,
            "reasoning": response_dict["reasoning"],
        }
        logger.info("Redaction complete", tokens_used=tokens_used)
        return result

    except Exception as e:
        logger.error("Redaction failed", error=str(e))
        return {
            "redacted": "",
            "original": item.text,
            "tokens_used": tokens_used,
            "reasoning": "Error in redact function",
        }


@app.post("/getCommunityNote")
async def generate_community_note_endpoint(request: CommunityNoteRequest):
    logger.info(
        "Processing community note request",
        has_text=bool(request.text),
        has_image=bool(request.image_url),
    )
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
            logger.error("Invalid request - missing content")
            raise HTTPException(
                status_code=400, detail="Either 'text' or 'image_url' must be provided."
            )
        logger.info("Community note generated successfully", session_id=session_id)
        return result
    except Exception as e:
        logger.error("Failed to generate community note", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v2/getCommunityNote")
async def get_gemini_note(
    request: CommunityNoteRequest,
    provider: SupportedModelProvider = SupportedModelProvider.GEMINI,
) -> AgentResponse:
    logger.info(
        "Processing v2 community note request",
        provider=provider.value,
        has_text=bool(request.text),
        has_image=bool(request.image_url),
    )
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
        if (
            provider == SupportedModelProvider.OPENAI
            or provider == SupportedModelProvider.DEEPSEEK
        ):
            result = await get_openai_outputs(
                text=request.text,
                image_url=request.image_url,
                caption=request.caption,
                addPlanning=request.addPlanning,
                provider=provider,
            )
            logger.info(
                "OpenAI/Deepseek note generated successfully", provider=provider.value
            )
            return result
        elif provider == SupportedModelProvider.GEMINI:
            result = await get_outputs(
                text=request.text,
                image_url=request.image_url,
                caption=request.caption,
                addPlanning=request.addPlanning,
            )
            logger.info("Gemini note generated successfully")
            return result
        else:
            logger.error("Unsupported provider specified", provider=provider.value)
            raise HTTPException(
                status_code=400, detail=f"Unsupported model provider: {provider}"
            )

    except HTTPException as e:
        logger.error(
            "HTTP exception in note generation",
            status_code=e.status_code,
            detail=e.detail,
        )
        raise e
    except Exception as e:
        logger.error("Unexpected error in note generation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
