from enum import Enum
from typing import List, Optional
from pydantic import BaseModel
from pydantic.fields import Field
from datetime import datetime


class SupportedModelProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class AgentResponse(BaseModel):
    requestId: str
    success: bool = False
    en: str | None = None
    cn: str | None = None
    links: List[str] | None = None
    isControversial: bool = False
    isVideo: bool = False
    isAccessBlocked: bool = False
    totalTimeTaken: float | None = None
    report: str | None = None
    errorMessage: str | None = None
    agentTrace: List[dict] | None = None


class CommunityNoteRequest(BaseModel):
    text: Optional[str] = Field(
        default=None, description="Text content for generating a community note"
    )
    image_url: Optional[str] = Field(
        default=None, description="Image URL for generating a community note"
    )
    caption: Optional[str] = Field(
        default=None, description="Caption for the image (optional)"
    )
    addPlanning: Optional[bool] = Field(
        default=False,
        description="Whether or not to include zero-shot planning step between each agent step",
    )


class SavedAgentCall(AgentResponse):
    text: Optional[str] = Field(
        default=None, description="Input text content for the agent call"
    )
    image_url: Optional[str] = Field(
        default=None, description="Input image URL for the agent call"
    )
    caption: Optional[str] = Field(
        default=None, description="Caption provided with the input"
    )
    timestamp: datetime = Field(description="Timestamp of when the call was made")
    model: SupportedModelProvider = Field(
        description="Model provider used for the call"
    )
    environment: str = Field(
        default=None, description="Environment in which the call was made"
    )
