"""Event schemas."""

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event type enum."""

    VIEW = "view"
    ADDTOCART = "addtocart"
    TRANSACTION = "transaction"


class EventCreate(BaseModel):
    """Event creation request."""

    visitor_id: int = Field(..., description="Visitor/User ID")
    item_id: int = Field(..., description="Item ID")
    event: EventType = Field(..., description="Event type")
    transaction_id: int | None = Field(
        None, description="Transaction ID (required for transaction events)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "visitor_id": 257597,
            "item_id": 355908,
            "event": "view",
        }
    }}


class EventResponse(BaseModel):
    """Event response."""

    id: int = Field(..., description="Event ID")
    visitor_id: int = Field(..., description="Visitor/User ID")
    item_id: int = Field(..., description="Item ID")
    event: EventType = Field(..., description="Event type")
    transaction_id: int | None = Field(None, description="Transaction ID")
    timestamp: datetime = Field(..., description="Event timestamp")

    model_config = {"json_schema_extra": {
        "example": {
            "id": 1,
            "visitor_id": 257597,
            "item_id": 355908,
            "event": "view",
            "transaction_id": None,
            "timestamp": "2025-01-17T10:30:00Z",
        }
    }}
