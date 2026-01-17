"""Pytest fixtures for tests."""

import pandas as pd
import pytest
from datetime import datetime


@pytest.fixture
def sample_events() -> pd.DataFrame:
    """Create sample events DataFrame for testing."""
    data = {
        "timestamp": [
            1433221332117,
            1433221332200,
            1433221332300,
            1433221332400,
            1433221332500,
            1433224214164,
            1433224214200,
            1433226394089,
            1433226394100,
            1433226394200,
        ],
        "visitor_id": [1, 1, 1, 2, 2, 1, 2, 1, 2, 3],
        "event": [
            "view",
            "view",
            "addtocart",
            "view",
            "view",
            "view",
            "addtocart",
            "transaction",
            "view",
            "view",
        ],
        "item_id": [100, 101, 100, 200, 201, 102, 200, 100, 202, 300],
        "transaction_id": [None, None, None, None, None, None, None, 1, None, None],
    }

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """Create sample interactions for model testing."""
    # More interactions for better model testing
    data = []

    # User 1: views and purchases
    for item_id in range(100, 110):
        data.append({"visitor_id": 1, "item_id": item_id, "event": "view", "timestamp": 1000000 + item_id})
    data.append({"visitor_id": 1, "item_id": 100, "event": "addtocart", "timestamp": 1000200})
    data.append({"visitor_id": 1, "item_id": 100, "event": "transaction", "timestamp": 1000300})

    # User 2: views similar items
    for item_id in range(100, 108):
        data.append({"visitor_id": 2, "item_id": item_id, "event": "view", "timestamp": 2000000 + item_id})
    data.append({"visitor_id": 2, "item_id": 101, "event": "addtocart", "timestamp": 2000200})

    # User 3: views different items
    for item_id in range(200, 210):
        data.append({"visitor_id": 3, "item_id": item_id, "event": "view", "timestamp": 3000000 + item_id})

    # User 4: views mix of items
    for item_id in [100, 101, 200, 201, 300]:
        data.append({"visitor_id": 4, "item_id": item_id, "event": "view", "timestamp": 4000000 + item_id})

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


@pytest.fixture
def recommendations_list() -> list[int]:
    """Sample recommendation list."""
    return [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]


@pytest.fixture
def relevant_items() -> set[int]:
    """Sample relevant items (ground truth)."""
    return {102, 105, 111, 115}
