"""Statistics schemas."""

from pydantic import BaseModel, Field


class UserStats(BaseModel):
    """User statistics response."""

    user_id: int = Field(..., description="User ID")
    total_events: int = Field(..., description="Total number of events")
    view_count: int = Field(..., description="Number of view events")
    addtocart_count: int = Field(..., description="Number of addtocart events")
    transaction_count: int = Field(..., description="Number of transaction events")
    unique_items_viewed: int = Field(..., description="Number of unique items viewed")
    unique_items_purchased: int = Field(..., description="Number of unique items purchased")

    model_config = {"json_schema_extra": {
        "example": {
            "user_id": 257597,
            "total_events": 25,
            "view_count": 20,
            "addtocart_count": 4,
            "transaction_count": 1,
            "unique_items_viewed": 15,
            "unique_items_purchased": 1,
        }
    }}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models_loaded: int = Field(..., description="Number of loaded models")
    version: str = Field(..., description="API version")

    model_config = {"json_schema_extra": {
        "example": {
            "status": "healthy",
            "models_loaded": 4,
            "version": "0.1.0",
        }
    }}
