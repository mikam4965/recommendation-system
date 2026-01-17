"""Event endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.api.schemas.events import EventCreate, EventResponse

router = APIRouter(prefix="/events", tags=["events"])

# In-memory event storage (for demo purposes)
# In production, this would use a database
_events: list[dict] = []
_event_id_counter = 0


@router.post("/", response_model=EventResponse, status_code=201)
async def create_event(event: EventCreate) -> EventResponse:
    """Record a new user event.

    This endpoint records user interactions (view, addtocart, transaction)
    for later use in training and real-time recommendations.

    Args:
        event: Event data to record.

    Returns:
        Created event with ID and timestamp.
    """
    global _event_id_counter

    # Validate transaction events have transaction_id
    if event.event.value == "transaction" and event.transaction_id is None:
        raise HTTPException(
            status_code=400,
            detail="transaction_id is required for transaction events",
        )

    _event_id_counter += 1

    event_data = {
        "id": _event_id_counter,
        "visitor_id": event.visitor_id,
        "item_id": event.item_id,
        "event": event.event.value,
        "transaction_id": event.transaction_id,
        "timestamp": datetime.now(timezone.utc),
    }

    _events.append(event_data)

    return EventResponse(
        id=event_data["id"],
        visitor_id=event_data["visitor_id"],
        item_id=event_data["item_id"],
        event=event_data["event"],
        transaction_id=event_data["transaction_id"],
        timestamp=event_data["timestamp"],
    )


@router.get("/count")
async def get_event_count() -> dict[str, int]:
    """Get total count of recorded events.

    Returns:
        Dictionary with event count.
    """
    return {"count": len(_events)}
