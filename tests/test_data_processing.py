"""Tests for data processing modules."""

import pytest
import pandas as pd
import numpy as np

from src.data.processors.cleaner import DataCleaner
from src.data.processors.session_builder import SessionBuilder
from src.data.processors.splitter import TimeBasedSplitter


class TestDataCleaner:
    """Tests for DataCleaner."""

    def test_remove_bots(self, sample_events):
        """Test bot removal."""
        # Create events with a bot user (many events in one day)
        bot_events = []
        for i in range(1500):
            bot_events.append({
                "timestamp": 1433221332117 + i * 1000,
                "visitor_id": 999,
                "event": "view",
                "item_id": 100 + (i % 10),
                "transaction_id": None,
            })

        bot_df = pd.DataFrame(bot_events)
        bot_df["datetime"] = pd.to_datetime(bot_df["timestamp"], unit="ms")

        # Combine with sample events
        combined = pd.concat([sample_events, bot_df], ignore_index=True)

        cleaner = DataCleaner(max_events_per_day=1000)
        cleaned = cleaner.remove_bots(combined)

        # Bot user 999 should be removed
        assert 999 not in cleaned["visitor_id"].values
        # Original users should remain
        assert 1 in cleaned["visitor_id"].values

    def test_filter_rare_items(self, sample_events):
        """Test rare item filtering."""
        cleaner = DataCleaner(min_item_interactions=3)
        cleaned = cleaner.filter_rare_items(sample_events)

        # Only items with >= 3 interactions should remain
        item_counts = cleaned["item_id"].value_counts()
        assert all(count >= 3 for count in item_counts.values)

    def test_remove_single_interaction_users(self):
        """Test single-interaction user removal."""
        events = pd.DataFrame({
            "timestamp": [1, 2, 3, 4, 5],
            "visitor_id": [1, 1, 2, 3, 3],
            "event": ["view"] * 5,
            "item_id": [100, 101, 200, 300, 301],
            "transaction_id": [None] * 5,
        })
        events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

        cleaner = DataCleaner()
        cleaned = cleaner.remove_single_interaction_users(events)

        # User 2 has only 1 interaction, should be removed
        assert 2 not in cleaned["visitor_id"].values
        # Users 1 and 3 should remain
        assert 1 in cleaned["visitor_id"].values
        assert 3 in cleaned["visitor_id"].values

    def test_clean_pipeline(self, sample_events):
        """Test full cleaning pipeline."""
        cleaner = DataCleaner(max_events_per_day=1000, min_item_interactions=1)
        cleaned = cleaner.clean(
            sample_events,
            remove_bots=True,
            filter_rare_items=True,
            remove_single_users=False,
        )

        assert len(cleaned) > 0
        assert "visitor_id" in cleaned.columns
        assert "item_id" in cleaned.columns


class TestSessionBuilder:
    """Tests for SessionBuilder."""

    def test_build_sessions(self, sample_events):
        """Test session building."""
        builder = SessionBuilder(timeout_minutes=30)
        with_sessions = builder.build_sessions(sample_events)

        assert "session_id" in with_sessions.columns
        assert with_sessions["session_id"].nunique() > 0

    def test_session_timeout(self):
        """Test that sessions are split at timeout boundary."""
        # Create events with clear session gap
        events = pd.DataFrame({
            "timestamp": [
                1000000,  # Session 1
                1001000,  # Session 1 (+1 sec)
                1002000,  # Session 1 (+2 sec)
                3000000,  # Session 2 (>30 min gap)
                3001000,  # Session 2
            ],
            "visitor_id": [1, 1, 1, 1, 1],
            "event": ["view"] * 5,
            "item_id": [100, 101, 102, 103, 104],
            "transaction_id": [None] * 5,
        })
        events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

        builder = SessionBuilder(timeout_minutes=30)
        with_sessions = builder.build_sessions(events)

        # Should have 2 sessions
        assert with_sessions["session_id"].nunique() == 2

    def test_get_session_stats(self, sample_events):
        """Test session statistics calculation."""
        builder = SessionBuilder(timeout_minutes=30)
        with_sessions = builder.build_sessions(sample_events)

        stats = builder.get_session_stats(with_sessions)

        assert "total_sessions" in stats
        assert "avg_session_length" in stats
        assert stats["total_sessions"] > 0

    def test_filter_short_sessions(self):
        """Test short session filtering."""
        events = pd.DataFrame({
            "timestamp": [1000, 1001, 1002, 5000000, 5000001, 5000002, 5000003],
            "visitor_id": [1, 1, 1, 1, 1, 1, 1],
            "event": ["view"] * 7,
            "item_id": list(range(100, 107)),
            "transaction_id": [None] * 7,
        })
        events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

        builder = SessionBuilder(timeout_minutes=30)
        with_sessions = builder.build_sessions(events)

        # Filter sessions with < 3 events
        filtered = builder.filter_short_sessions(with_sessions, min_length=3)

        # All remaining events should be in sessions with >= 3 events
        session_sizes = filtered.groupby("session_id").size()
        assert all(size >= 3 for size in session_sizes.values)


class TestTimeBasedSplitter:
    """Tests for TimeBasedSplitter."""

    def test_split_ratios(self):
        """Test that splits respect ratios."""
        # Create events with timestamps
        n = 1000
        events = pd.DataFrame({
            "timestamp": list(range(n)),
            "visitor_id": [i % 100 for i in range(n)],
            "event": ["view"] * n,
            "item_id": [i % 50 for i in range(n)],
        })
        events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

        splitter = TimeBasedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        train, val, test = splitter.split(events)

        # Check approximate ratios
        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - 0.7) < 0.01
        assert abs(len(val) / total - 0.15) < 0.01
        assert abs(len(test) / total - 0.15) < 0.01

    def test_no_time_leakage(self):
        """Test that train data comes before val/test."""
        events = pd.DataFrame({
            "timestamp": list(range(100)),
            "visitor_id": [1] * 100,
            "event": ["view"] * 100,
            "item_id": list(range(100)),
        })
        events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

        splitter = TimeBasedSplitter()
        train, val, test = splitter.split(events)

        # Train timestamps should all be <= val timestamps
        assert train["timestamp"].max() <= val["timestamp"].min()
        # Val timestamps should all be <= test timestamps
        assert val["timestamp"].max() <= test["timestamp"].min()

    def test_invalid_ratios_raises(self):
        """Test that invalid ratios raise error."""
        with pytest.raises(ValueError):
            TimeBasedSplitter(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)

    def test_get_split_stats(self):
        """Test split statistics calculation."""
        events = pd.DataFrame({
            "timestamp": list(range(100)),
            "visitor_id": [i % 10 for i in range(100)],
            "event": ["view"] * 100,
            "item_id": [i % 20 for i in range(100)],
        })
        events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

        splitter = TimeBasedSplitter()
        train, val, test = splitter.split(events)

        stats = splitter.get_split_stats(train, val, test)

        assert "train_events" in stats
        assert "val_events" in stats
        assert "test_events" in stats
        assert "train_users" in stats
        assert "val_cold_start_users" in stats
