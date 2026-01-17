"""Data cleaning utilities."""

import pandas as pd
from loguru import logger

from src.config import settings


class DataCleaner:
    """Clean and filter event data."""

    def __init__(
        self,
        max_events_per_day: int | None = None,
        min_item_interactions: int | None = None,
    ):
        """Initialize cleaner.

        Args:
            max_events_per_day: Maximum events per user per day (bot threshold).
                Defaults to settings.max_events_per_day.
            min_item_interactions: Minimum interactions for an item to be kept.
                Defaults to settings.min_item_interactions.
        """
        self.max_events_per_day = max_events_per_day or settings.max_events_per_day
        self.min_item_interactions = min_item_interactions or settings.min_item_interactions

    def remove_bots(self, events: pd.DataFrame) -> pd.DataFrame:
        """Remove bot users based on activity threshold.

        Users with more than max_events_per_day events on any single day
        are considered bots and removed.

        Args:
            events: Events DataFrame with visitor_id and datetime columns.

        Returns:
            Filtered DataFrame without bot users.
        """
        logger.info(f"Removing bots (threshold: {self.max_events_per_day} events/day)")

        initial_count = len(events)
        initial_users = events["visitor_id"].nunique()

        # Add date column for grouping
        events = events.copy()
        events["date"] = events["datetime"].dt.date

        # Count events per user per day
        daily_counts = events.groupby(["visitor_id", "date"]).size().reset_index(name="count")

        # Find bot users (any day exceeding threshold)
        bot_users = daily_counts[daily_counts["count"] > self.max_events_per_day][
            "visitor_id"
        ].unique()

        # Filter out bots
        events = events[~events["visitor_id"].isin(bot_users)]

        # Drop temporary date column
        events = events.drop(columns=["date"])

        removed_users = initial_users - events["visitor_id"].nunique()
        removed_events = initial_count - len(events)

        logger.info(f"Removed {removed_users:,} bot users ({removed_events:,} events)")

        return events

    def filter_rare_items(self, events: pd.DataFrame) -> pd.DataFrame:
        """Remove items with too few interactions.

        Args:
            events: Events DataFrame with item_id column.

        Returns:
            Filtered DataFrame with only items having enough interactions.
        """
        logger.info(f"Filtering rare items (min interactions: {self.min_item_interactions})")

        initial_count = len(events)
        initial_items = events["item_id"].nunique()

        # Count interactions per item
        item_counts = events["item_id"].value_counts()

        # Get items with enough interactions
        popular_items = item_counts[item_counts >= self.min_item_interactions].index

        # Filter events
        events = events[events["item_id"].isin(popular_items)]

        removed_items = initial_items - events["item_id"].nunique()
        removed_events = initial_count - len(events)

        logger.info(f"Removed {removed_items:,} rare items ({removed_events:,} events)")

        return events

    def remove_single_interaction_users(self, events: pd.DataFrame) -> pd.DataFrame:
        """Remove users with only one interaction.

        These users cannot be used for training/evaluation properly.

        Args:
            events: Events DataFrame with visitor_id column.

        Returns:
            Filtered DataFrame with only users having multiple interactions.
        """
        logger.info("Removing single-interaction users")

        initial_count = len(events)
        initial_users = events["visitor_id"].nunique()

        # Count interactions per user
        user_counts = events["visitor_id"].value_counts()

        # Get users with multiple interactions
        active_users = user_counts[user_counts > 1].index

        # Filter events
        events = events[events["visitor_id"].isin(active_users)]

        removed_users = initial_users - events["visitor_id"].nunique()
        removed_events = initial_count - len(events)

        logger.info(f"Removed {removed_users:,} single-interaction users ({removed_events:,} events)")

        return events

    def clean(
        self,
        events: pd.DataFrame,
        remove_bots: bool = True,
        filter_rare_items: bool = True,
        remove_single_users: bool = False,
    ) -> pd.DataFrame:
        """Apply all cleaning steps.

        Args:
            events: Raw events DataFrame.
            remove_bots: Whether to remove bot users.
            filter_rare_items: Whether to filter rare items.
            remove_single_users: Whether to remove single-interaction users.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Starting data cleaning pipeline")
        logger.info(f"Initial: {len(events):,} events, {events['visitor_id'].nunique():,} users, {events['item_id'].nunique():,} items")

        if remove_bots:
            events = self.remove_bots(events)

        if filter_rare_items:
            events = self.filter_rare_items(events)

        if remove_single_users:
            events = self.remove_single_interaction_users(events)

        logger.info(f"Final: {len(events):,} events, {events['visitor_id'].nunique():,} users, {events['item_id'].nunique():,} items")

        return events
