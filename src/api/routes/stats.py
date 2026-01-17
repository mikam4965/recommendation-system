"""Statistics endpoints."""

from fastapi import APIRouter, HTTPException

import pandas as pd

from src.api.schemas.stats import UserStats
from src.config import settings

router = APIRouter(prefix="/stats", tags=["statistics"])

# Cached data (loaded on first request)
_train_data: pd.DataFrame | None = None


def _load_train_data() -> pd.DataFrame:
    """Load training data for stats queries."""
    global _train_data

    if _train_data is not None:
        return _train_data

    train_path = settings.data_processed_dir / "train.parquet"
    if not train_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Training data not available. Run 'make process-data' first.",
        )

    _train_data = pd.read_parquet(train_path)
    return _train_data


@router.get("/user/{user_id}", response_model=UserStats)
async def get_user_stats(user_id: int) -> UserStats:
    """Get statistics for a specific user.

    Args:
        user_id: User ID to get stats for.

    Returns:
        User statistics including event counts and unique items.
    """
    data = _load_train_data()

    user_events = data[data["visitor_id"] == user_id]

    if user_events.empty:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found in training data",
        )

    # Calculate stats
    event_counts = user_events["event"].value_counts()

    # Items viewed (from view events)
    view_events = user_events[user_events["event"] == "view"]
    unique_viewed = view_events["item_id"].nunique()

    # Items purchased (from transaction events)
    transaction_events = user_events[user_events["event"] == "transaction"]
    unique_purchased = transaction_events["item_id"].nunique()

    return UserStats(
        user_id=user_id,
        total_events=len(user_events),
        view_count=int(event_counts.get("view", 0)),
        addtocart_count=int(event_counts.get("addtocart", 0)),
        transaction_count=int(event_counts.get("transaction", 0)),
        unique_items_viewed=unique_viewed,
        unique_items_purchased=unique_purchased,
    )


@router.get("/summary")
async def get_data_summary() -> dict:
    """Get summary statistics of the training data.

    Returns:
        Dictionary with overall dataset statistics.
    """
    data = _load_train_data()

    event_counts = data["event"].value_counts().to_dict()

    return {
        "total_events": len(data),
        "unique_users": data["visitor_id"].nunique(),
        "unique_items": data["item_id"].nunique(),
        "event_distribution": event_counts,
        "date_range": {
            "min": str(data["datetime"].min()),
            "max": str(data["datetime"].max()),
        },
    }
