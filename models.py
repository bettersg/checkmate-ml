# models.py
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class PredictionEnum(str, Enum):
    scam = "scam"
    illicit = "illicit"
    spam = "spam"
    info = "info"
    irrelevant = "irrelevant"
    unsure = "unsure"

class ImageTypeEnum(str, Enum):
    email = "email"
    convo = "convo"
    letter = "letter"
    others = "others"

class ItemText(BaseModel):
    text: str = Field(
        ...,
        example="Your account has been suspended due to suspicious activity. Please verify your identity."
    )

class ItemUrl(BaseModel):
    url: str = Field(
        ...,
        example="https://example.com/images/sample-email.png"
    )

class EmbeddingResponse(BaseModel):
    embedding: List[float] = Field(
        ...,
        example=[0.123, -0.456, 0.789, -0.101, 0.112, -0.131]
    )

class L1CategoryResponse(BaseModel):
    prediction: PredictionEnum = Field(
        ...,
        example="scam"
    )

class OCRResponse(BaseModel):
    extracted_message: Optional[str] = Field(
        None,
        example="Dear customer, your account has been locked. Click here to reset your password."
    )
    prediction: PredictionEnum = Field(
        ...,
        example="spam"
    )
    image_type: Optional[ImageTypeEnum] = Field(
        None,
        example="email"
    )
    sender: Optional[str] = Field(
        None,
        example="support@example.com"
    )
    subject: Optional[str] = Field(
        None,
        example="Important: Account Verification Required"
    )