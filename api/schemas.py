from pydantic import BaseModel, Field


class NewsRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="News statement to classify"
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float