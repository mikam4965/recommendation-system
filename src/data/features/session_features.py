"""Session feature extraction utilities."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SessionFeatures:
    """Features for a user session.

    Attributes:
        session_id: Unique session identifier.
        user_id: Visitor ID.
        session_length: Number of events in the session.
        unique_items: Number of unique items interacted with.
        unique_categories: Number of unique categories.
        has_addtocart: Whether session contains addtocart event.
        has_transaction: Whether session contains transaction event.
        avg_item_popularity: Average popularity of items in session.
        category_concentration: Herfindahl-Hirschman Index for category concentration.
        duration_minutes: Session duration in minutes.
        view_count: Number of view events.
        addtocart_count: Number of addtocart events.
        transaction_count: Number of transaction events.
    """

    session_id: int
    user_id: int
    session_length: int = 0
    unique_items: int = 0
    unique_categories: int = 0
    has_addtocart: bool = False
    has_transaction: bool = False
    avg_item_popularity: float = 0.0
    category_concentration: float = 0.0
    duration_minutes: float = 0.0
    view_count: int = 0
    addtocart_count: int = 0
    transaction_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "session_length": self.session_length,
            "unique_items": self.unique_items,
            "unique_categories": self.unique_categories,
            "has_addtocart": self.has_addtocart,
            "has_transaction": self.has_transaction,
            "avg_item_popularity": self.avg_item_popularity,
            "category_concentration": self.category_concentration,
            "duration_minutes": self.duration_minutes,
            "view_count": self.view_count,
            "addtocart_count": self.addtocart_count,
            "transaction_count": self.transaction_count,
        }


class SessionFeatureExtractor:
    """Extract features from user sessions.

    Computes session-level features including engagement metrics,
    category concentration, and item popularity.
    """

    def __init__(
        self,
        item_categories: dict[int, int] | None = None,
        item_popularity: dict[int, int] | None = None,
    ):
        """Initialize feature extractor.

        Args:
            item_categories: Mapping of item_id to category_id.
            item_popularity: Mapping of item_id to popularity (interaction count).
        """
        self.item_categories = item_categories or {}
        self.item_popularity = item_popularity or {}
        self.total_interactions = sum(item_popularity.values()) if item_popularity else 0

        # Will be populated during fit
        self._session_features: dict[int, SessionFeatures] = {}
        self._session_items: dict[int, list[int]] = {}
        self._is_fitted = False

    def fit(
        self,
        interactions: pd.DataFrame,
        item_categories: dict[int, int] | None = None,
        item_popularity: dict[int, int] | None = None,
    ) -> "SessionFeatureExtractor":
        """Fit the extractor on interaction data.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type
                - timestamp: Event timestamp
                - session_id: Session ID (required)
            item_categories: Optional mapping of item_id to category_id.
            item_popularity: Optional mapping of item_id to popularity.

        Returns:
            Self for method chaining.
        """
        if "session_id" not in interactions.columns:
            raise ValueError("Interactions must have session_id column")

        logger.info(f"Fitting SessionFeatureExtractor on {len(interactions):,} interactions")

        if item_categories:
            self.item_categories = item_categories
        if item_popularity:
            self.item_popularity = item_popularity
            self.total_interactions = sum(item_popularity.values())

        # Reset state
        self._session_features = {}
        self._session_items = {}

        # Group by session
        session_groups = interactions.groupby("session_id")

        for session_id, session_data in session_groups:
            features = self._extract_session_features(session_id, session_data)
            self._session_features[session_id] = features

            # Store session items in order
            session_sorted = session_data.sort_values("timestamp")
            self._session_items[session_id] = session_sorted["item_id"].tolist()

        self._is_fitted = True

        # Log summary statistics
        df = self.get_all_features()
        if not df.empty:
            logger.info(f"Extracted features for {len(df):,} sessions")
            logger.info(f"Average session length: {df['session_length'].mean():.2f}")
            logger.info(f"Sessions with addtocart: {df['has_addtocart'].mean():.1%}")
            logger.info(f"Sessions with transaction: {df['has_transaction'].mean():.1%}")

        return self

    def _extract_session_features(
        self, session_id: int, session_data: pd.DataFrame
    ) -> SessionFeatures:
        """Extract features for a single session.

        Args:
            session_id: Session ID.
            session_data: DataFrame of session's interactions.

        Returns:
            SessionFeatures object.
        """
        user_id = session_data["visitor_id"].iloc[0]
        session_length = len(session_data)
        unique_items = session_data["item_id"].nunique()

        # Event counts
        event_counts = session_data["event"].value_counts().to_dict()
        view_count = event_counts.get("view", 0)
        addtocart_count = event_counts.get("addtocart", 0)
        transaction_count = event_counts.get("transaction", 0)

        has_addtocart = addtocart_count > 0
        has_transaction = transaction_count > 0

        # Category features
        unique_categories = 0
        category_concentration = 0.0

        if self.item_categories:
            items = session_data["item_id"].unique()
            categories = [
                self.item_categories[item_id]
                for item_id in items
                if item_id in self.item_categories
            ]
            unique_categories = len(set(categories))

            # Calculate HHI (Herfindahl-Hirschman Index) for concentration
            if categories:
                category_concentration = self._calculate_hhi(categories)

        # Item popularity
        avg_item_popularity = 0.0
        if self.item_popularity and self.total_interactions > 0:
            items = session_data["item_id"].tolist()
            popularities = [
                self.item_popularity.get(item_id, 0) / self.total_interactions
                for item_id in items
            ]
            avg_item_popularity = np.mean(popularities) if popularities else 0.0

        # Duration
        duration_minutes = 0.0
        if len(session_data) > 1:
            duration_ms = session_data["timestamp"].max() - session_data["timestamp"].min()
            duration_minutes = duration_ms / 60000  # Convert ms to minutes

        return SessionFeatures(
            session_id=session_id,
            user_id=user_id,
            session_length=session_length,
            unique_items=unique_items,
            unique_categories=unique_categories,
            has_addtocart=has_addtocart,
            has_transaction=has_transaction,
            avg_item_popularity=avg_item_popularity,
            category_concentration=category_concentration,
            duration_minutes=duration_minutes,
            view_count=view_count,
            addtocart_count=addtocart_count,
            transaction_count=transaction_count,
        )

    def _calculate_hhi(self, categories: list[int]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration.

        HHI = sum(share_i^2) where share_i is the proportion of category i.
        HHI ranges from 1/N (perfectly diverse) to 1 (all same category).

        Args:
            categories: List of category IDs.

        Returns:
            HHI value between 0 and 1.
        """
        if not categories:
            return 0.0

        # Count categories
        category_counts = defaultdict(int)
        for cat in categories:
            category_counts[cat] += 1

        total = len(categories)
        shares = [count / total for count in category_counts.values()]
        hhi = sum(share**2 for share in shares)

        return hhi

    def get_session_features(self, session_id: int) -> SessionFeatures | None:
        """Get features for a specific session.

        Args:
            session_id: Session ID.

        Returns:
            SessionFeatures object or None if session not found.
        """
        return self._session_features.get(session_id)

    def get_session_items(self, session_id: int) -> list[int]:
        """Get ordered list of items in a session.

        Args:
            session_id: Session ID.

        Returns:
            List of item IDs in chronological order.
        """
        return self._session_items.get(session_id, [])

    def get_all_features(self) -> pd.DataFrame:
        """Get features for all sessions as a DataFrame.

        Returns:
            DataFrame with session features.
        """
        if not self._session_features:
            return pd.DataFrame()

        records = [features.to_dict() for features in self._session_features.values()]
        return pd.DataFrame(records)

    def get_user_sessions(self, user_id: int) -> list[int]:
        """Get all session IDs for a user.

        Args:
            user_id: User ID.

        Returns:
            List of session IDs.
        """
        return [
            session_id
            for session_id, features in self._session_features.items()
            if features.user_id == user_id
        ]

    def get_converting_sessions(self, event_type: str = "transaction") -> list[int]:
        """Get sessions that have a specific conversion event.

        Args:
            event_type: Event type to filter by (addtocart or transaction).

        Returns:
            List of session IDs.
        """
        if event_type == "transaction":
            return [
                session_id
                for session_id, features in self._session_features.items()
                if features.has_transaction
            ]
        elif event_type == "addtocart":
            return [
                session_id
                for session_id, features in self._session_features.items()
                if features.has_addtocart
            ]
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    @property
    def is_fitted(self) -> bool:
        """Check if extractor has been fitted."""
        return self._is_fitted
