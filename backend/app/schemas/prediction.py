from pydantic import BaseModel, Field
from typing import Any


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        max_length=50000,
        description="Text document for topic modelling"
    )


class PredictionResponse(BaseModel):
    status: str
    history_id: int
    predictions: dict[str, Any]