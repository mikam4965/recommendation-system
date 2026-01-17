"""Session building utilities."""

import pandas as pd
from loguru import logger

from src.config import settings


class SessionBuilder:
    """Build user sessions from event data."""

    def __init__(self, timeout_minutes: int | None = None):
        """Initialize session builder.

        Args:
            timeout_minutes: Session timeout in minutes. Events more than this
                apart are considered different sessions. Defaults to settings.
        """
        self.timeout_minutes = timeout_minutes or settings.session_timeout_minutes

    def build_sessions(self, events: pd.DataFrame) -> pd.DataFrame:
        """Assign session IDs to events based on time gaps.

        Sessions are user-specific and are split when the gap between
        consecutive events exceeds timeout_minutes.

        Args:
            events: Events DataFrame with visitor_id and timestamp columns.

        Returns:
            DataFrame with added session_id column.
        """
        logger.info(f"Building sessions (timeout: {self.timeout_minutes} minutes)")

        events = events.copy()

        # Sort by user and time
        events = events.sort_values(["visitor_id", "timestamp"])

        # Calculate time difference within each user's events
        events["time_diff"] = events.groupby("visitor_id")["timestamp"].diff()

        # Convert milliseconds to minutes
        timeout_ms = self.timeout_minutes * 60 * 1000

        # Mark new session starts (first event or gap > timeout)
        events["new_session"] = (events["time_diff"].isna()) | (
            events["time_diff"] > timeout_ms
        )

        # Create session IDs by cumulative sum of new_session flags
        events["session_id"] = events["new_session"].cumsum()

        # Clean up temporary columns
        events = events.drop(columns=["time_diff", "new_session"])

        # Calculate session statistics
        session_counts = events.groupby("session_id").size()
        logger.info(f"Created {events['session_id'].nunique():,} sessions")
        logger.info(f"Average session length: {session_counts.mean():.2f} events")
        logger.info(f"Median session length: {session_counts.median():.0f} events")

        return events

    def get_session_stats(self, events: pd.DataFrame) -> dict:
        """Calculate session statistics.

        Args:
            events: Events DataFrame with session_id column.

        Returns:
            Dictionary with session statistics.
        """
        if "session_id" not in events.columns:
            raise ValueError("Events must have session_id column. Run build_sessions first.")

        session_lengths = events.groupby("session_id").size()
        session_durations = events.groupby("session_id").agg(
            duration_ms=("timestamp", lambda x: x.max() - x.min())
        )

        stats = {
            "total_sessions": events["session_id"].nunique(),
            "avg_session_length": float(session_lengths.mean()),
            "median_session_length": float(session_lengths.median()),
            "max_session_length": int(session_lengths.max()),
            "min_session_length": int(session_lengths.min()),
            "avg_duration_minutes": float(session_durations["duration_ms"].mean() / 60000),
            "single_event_sessions": int((session_lengths == 1).sum()),
            "single_event_session_ratio": float((session_lengths == 1).mean()),
        }

        return stats

    def filter_short_sessions(
        self, events: pd.DataFrame, min_length: int = 2
    ) -> pd.DataFrame:
        """Remove sessions with fewer than min_length events.

        Args:
            events: Events DataFrame with session_id column.
            min_length: Minimum session length to keep.

        Returns:
            Filtered DataFrame.
        """
        logger.info(f"Filtering sessions shorter than {min_length} events")

        if "session_id" not in events.columns:
            raise ValueError("Events must have session_id column. Run build_sessions first.")

        initial_sessions = events["session_id"].nunique()
        initial_events = len(events)

        # Get session lengths
        session_lengths = events.groupby("session_id").size()

        # Find sessions to keep
        valid_sessions = session_lengths[session_lengths >= min_length].index

        # Filter events
        events = events[events["session_id"].isin(valid_sessions)]

        removed_sessions = initial_sessions - events["session_id"].nunique()
        removed_events = initial_events - len(events)

        logger.info(f"Removed {removed_sessions:,} short sessions ({removed_events:,} events)")

        return events
