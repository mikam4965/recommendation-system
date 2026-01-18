"""Data splitting utilities."""

import pandas as pd
from loguru import logger

from src.config import settings


class TimeBasedSplitter:
    """Split data based on timestamp for train/val/test."""

    def __init__(
        self,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
    ):
        """Initialize splitter.

        Args:
            train_ratio: Ratio of data for training. Defaults to settings.
            val_ratio: Ratio of data for validation. Defaults to settings.
            test_ratio: Ratio of data for testing. Defaults to settings.
        """
        self.train_ratio = train_ratio if train_ratio is not None else settings.train_ratio
        self.val_ratio = val_ratio if val_ratio is not None else settings.val_ratio
        self.test_ratio = test_ratio if test_ratio is not None else settings.test_ratio

        # Validate ratios sum to 1
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

    def split(
        self, events: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split events by timestamp.

        Events are sorted by timestamp and split into train/val/test
        based on the specified ratios.

        Args:
            events: Events DataFrame with timestamp column.

        Returns:
            Tuple of (train, val, test) DataFrames.
        """
        logger.info(
            f"Splitting data: train={self.train_ratio:.0%}, "
            f"val={self.val_ratio:.0%}, test={self.test_ratio:.0%}"
        )

        # Sort by timestamp
        events = events.sort_values("timestamp").reset_index(drop=True)

        # Calculate split points
        n = len(events)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        # Split
        train = events.iloc[:train_end].copy()
        val = events.iloc[train_end:val_end].copy()
        test = events.iloc[val_end:].copy()

        # Log split info
        logger.info(f"Train: {len(train):,} events ({len(train)/n:.1%})")
        logger.info(f"  Date range: {train['datetime'].min()} to {train['datetime'].max()}")
        logger.info(f"  Users: {train['visitor_id'].nunique():,}, Items: {train['item_id'].nunique():,}")

        logger.info(f"Val: {len(val):,} events ({len(val)/n:.1%})")
        logger.info(f"  Date range: {val['datetime'].min()} to {val['datetime'].max()}")
        logger.info(f"  Users: {val['visitor_id'].nunique():,}, Items: {val['item_id'].nunique():,}")

        logger.info(f"Test: {len(test):,} events ({len(test)/n:.1%})")
        logger.info(f"  Date range: {test['datetime'].min()} to {test['datetime'].max()}")
        logger.info(f"  Users: {test['visitor_id'].nunique():,}, Items: {test['item_id'].nunique():,}")

        # Check for data leakage
        train_max = train["timestamp"].max()
        val_min = val["timestamp"].min()
        test_min = test["timestamp"].min()

        if train_max >= val_min:
            logger.warning("Potential data leakage: train overlaps with val")
        if train_max >= test_min:
            logger.warning("Potential data leakage: train overlaps with test")

        return train, val, test

    def get_split_stats(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> dict:
        """Calculate statistics about the split.

        Args:
            train: Training DataFrame.
            val: Validation DataFrame.
            test: Test DataFrame.

        Returns:
            Dictionary with split statistics.
        """
        total = len(train) + len(val) + len(test)

        # User overlap between splits
        train_users = set(train["visitor_id"].unique())
        val_users = set(val["visitor_id"].unique())
        test_users = set(test["visitor_id"].unique())

        # Item overlap between splits
        train_items = set(train["item_id"].unique())
        val_items = set(val["item_id"].unique())
        test_items = set(test["item_id"].unique())

        stats = {
            "train_events": len(train),
            "val_events": len(val),
            "test_events": len(test),
            "train_ratio_actual": len(train) / total,
            "val_ratio_actual": len(val) / total,
            "test_ratio_actual": len(test) / total,
            "train_users": len(train_users),
            "val_users": len(val_users),
            "test_users": len(test_users),
            "train_items": len(train_items),
            "val_items": len(val_items),
            "test_items": len(test_items),
            "val_users_in_train": len(val_users & train_users),
            "test_users_in_train": len(test_users & train_users),
            "val_items_in_train": len(val_items & train_items),
            "test_items_in_train": len(test_items & train_items),
            "val_cold_start_users": len(val_users - train_users),
            "test_cold_start_users": len(test_users - train_users),
            "val_cold_start_items": len(val_items - train_items),
            "test_cold_start_items": len(test_items - train_items),
        }

        return stats
