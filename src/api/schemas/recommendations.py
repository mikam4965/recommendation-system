"""Recommendation schemas."""

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Single recommendation item."""

    item_id: int = Field(..., description="Item ID")
    score: float = Field(..., description="Recommendation score")
    rank: int = Field(..., description="Position in recommendation list")


class RecommendationResponse(BaseModel):
    """Recommendation response."""

    user_id: int = Field(..., description="User ID")
    model: str = Field(..., description="Model used for recommendations")
    recommendations: list[RecommendationItem] = Field(
        ..., description="List of recommended items"
    )
    n_items: int = Field(..., description="Number of items requested")

    model_config = {"json_schema_extra": {
        "example": {
            "user_id": 257597,
            "model": "popular",
            "n_items": 5,
            "recommendations": [
                {"item_id": 461686, "score": 15234.0, "rank": 1},
                {"item_id": 119736, "score": 12456.0, "rank": 2},
                {"item_id": 213834, "score": 10987.0, "rank": 3},
            ],
        }
    }}


class AvailableModelsResponse(BaseModel):
    """Available models response."""

    models: list[str] = Field(..., description="List of available model names")
    default: str = Field(..., description="Default model name")
