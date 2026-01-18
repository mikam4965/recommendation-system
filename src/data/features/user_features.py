"""User feature extraction and funnel stage detection."""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
from loguru import logger


class FunnelStage(Enum):
    """User funnel stage based on behavior.

    Stages represent user's position in the conversion funnel:
    - NEW_USER: Initial visitors with minimal engagement
    - ACTIVE_BROWSER: Regular browsers exploring the catalog
    - INTENDER: Users showing purchase intent (addtocart)
    - BUYER: Users who have completed transactions
    """

    NEW_USER = "new_user"
    ACTIVE_BROWSER = "active_browser"
    INTENDER = "intender"
    BUYER = "buyer"

    def __str__(self) -> str:
        return self.value


@dataclass
class UserFeatures:
    """Aggregated features for a user.

    Attributes:
        user_id: Visitor ID.
        total_views: Total number of view events.
        total_addtocarts: Total number of addtocart events.
        total_transactions: Total number of transaction events.
        unique_items_viewed: Number of unique items viewed.
        unique_items_carted: Number of unique items added to cart.
        unique_items_purchased: Number of unique items purchased.
        unique_categories: Number of unique categories interacted with.
        session_count: Number of sessions.
        avg_session_length: Average session length in events.
        days_active: Number of unique days with activity.
        funnel_stage: Current funnel stage.
        view_to_cart_rate: Conversion rate from view to addtocart.
        cart_to_purchase_rate: Conversion rate from addtocart to transaction.
    """

    user_id: int
    total_views: int = 0
    total_addtocarts: int = 0
    total_transactions: int = 0
    unique_items_viewed: int = 0
    unique_items_carted: int = 0
    unique_items_purchased: int = 0
    unique_categories: int = 0
    session_count: int = 0
    avg_session_length: float = 0.0
    days_active: int = 0
    funnel_stage: FunnelStage = FunnelStage.NEW_USER
    view_to_cart_rate: float = 0.0
    cart_to_purchase_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "total_views": self.total_views,
            "total_addtocarts": self.total_addtocarts,
            "total_transactions": self.total_transactions,
            "unique_items_viewed": self.unique_items_viewed,
            "unique_items_carted": self.unique_items_carted,
            "unique_items_purchased": self.unique_items_purchased,
            "unique_categories": self.unique_categories,
            "session_count": self.session_count,
            "avg_session_length": self.avg_session_length,
            "days_active": self.days_active,
            "funnel_stage": str(self.funnel_stage),
            "view_to_cart_rate": self.view_to_cart_rate,
            "cart_to_purchase_rate": self.cart_to_purchase_rate,
        }


class UserFeatureExtractor:
    """Extract user features from interaction data.

    Computes aggregated user features including funnel stage,
    engagement metrics, and conversion rates.
    """

    # Thresholds for funnel stage detection
    NEW_USER_MAX_VIEWS = 5

    def __init__(
        self,
        new_user_max_views: int = 5,
        item_categories: dict[int, int] | None = None,
    ):
        """Initialize feature extractor.

        Args:
            new_user_max_views: Maximum views to be considered a new user.
            item_categories: Optional mapping of item_id to category_id.
        """
        self.new_user_max_views = new_user_max_views
        self.item_categories = item_categories or {}

        # Will be populated during fit
        self._user_features: dict[int, UserFeatures] = {}
        self._user_items: dict[int, dict[str, set[int]]] = defaultdict(
            lambda: {"view": set(), "addtocart": set(), "transaction": set()}
        )
        self._is_fitted = False

    def fit(
        self,
        interactions: pd.DataFrame,
        item_categories: dict[int, int] | None = None,
    ) -> "UserFeatureExtractor":
        """Fit the extractor on interaction data.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type (view, addtocart, transaction)
                - timestamp: Event timestamp
                - session_id (optional): Session ID
            item_categories: Optional mapping of item_id to category_id.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting UserFeatureExtractor on {len(interactions):,} interactions")

        if item_categories:
            self.item_categories = item_categories

        # Reset state
        self._user_features = {}
        self._user_items = defaultdict(
            lambda: {"view": set(), "addtocart": set(), "transaction": set()}
        )

        # Group by user
        user_groups = interactions.groupby("visitor_id")

        for user_id, user_data in user_groups:
            features = self._extract_user_features(user_id, user_data)
            self._user_features[user_id] = features

            # Track items per event type
            for _, row in user_data.iterrows():
                self._user_items[user_id][row["event"]].add(row["item_id"])

        self._is_fitted = True

        # Log funnel stage distribution
        stage_counts = defaultdict(int)
        for features in self._user_features.values():
            stage_counts[features.funnel_stage] += 1

        logger.info("Funnel stage distribution:")
        for stage in FunnelStage:
            count = stage_counts[stage]
            pct = count / len(self._user_features) * 100 if self._user_features else 0
            logger.info(f"  {stage}: {count:,} users ({pct:.1f}%)")

        return self

    def _extract_user_features(self, user_id: int, user_data: pd.DataFrame) -> UserFeatures:
        """Extract features for a single user.

        Args:
            user_id: User ID.
            user_data: DataFrame of user's interactions.

        Returns:
            UserFeatures object.
        """
        # Count events by type
        event_counts = user_data["event"].value_counts().to_dict()
        total_views = event_counts.get("view", 0)
        total_addtocarts = event_counts.get("addtocart", 0)
        total_transactions = event_counts.get("transaction", 0)

        # Unique items by event type
        unique_items_viewed = user_data[user_data["event"] == "view"]["item_id"].nunique()
        unique_items_carted = user_data[user_data["event"] == "addtocart"]["item_id"].nunique()
        unique_items_purchased = user_data[user_data["event"] == "transaction"][
            "item_id"
        ].nunique()

        # Unique categories
        unique_categories = 0
        if self.item_categories:
            items = user_data["item_id"].unique()
            categories = set(
                self.item_categories[item_id]
                for item_id in items
                if item_id in self.item_categories
            )
            unique_categories = len(categories)

        # Session statistics
        session_count = 0
        avg_session_length = 0.0
        if "session_id" in user_data.columns:
            session_count = user_data["session_id"].nunique()
            if session_count > 0:
                avg_session_length = len(user_data) / session_count

        # Days active
        days_active = 0
        if "datetime" in user_data.columns:
            days_active = user_data["datetime"].dt.date.nunique()
        elif "timestamp" in user_data.columns:
            # Convert timestamp (ms) to date
            dates = pd.to_datetime(user_data["timestamp"], unit="ms").dt.date
            days_active = dates.nunique()

        # Conversion rates
        view_to_cart_rate = (
            unique_items_carted / unique_items_viewed if unique_items_viewed > 0 else 0.0
        )
        cart_to_purchase_rate = (
            unique_items_purchased / unique_items_carted if unique_items_carted > 0 else 0.0
        )

        # Determine funnel stage
        funnel_stage = self._determine_funnel_stage(
            total_views, total_addtocarts, total_transactions
        )

        return UserFeatures(
            user_id=user_id,
            total_views=total_views,
            total_addtocarts=total_addtocarts,
            total_transactions=total_transactions,
            unique_items_viewed=unique_items_viewed,
            unique_items_carted=unique_items_carted,
            unique_items_purchased=unique_items_purchased,
            unique_categories=unique_categories,
            session_count=session_count,
            avg_session_length=avg_session_length,
            days_active=days_active,
            funnel_stage=funnel_stage,
            view_to_cart_rate=view_to_cart_rate,
            cart_to_purchase_rate=cart_to_purchase_rate,
        )

    def _determine_funnel_stage(
        self,
        total_views: int,
        total_addtocarts: int,
        total_transactions: int,
    ) -> FunnelStage:
        """Determine user's funnel stage based on their behavior.

        Logic:
        - BUYER: Has at least one transaction
        - INTENDER: Has addtocart but no transaction
        - ACTIVE_BROWSER: More than threshold views, no addtocart
        - NEW_USER: Few views (<=5), no addtocart

        Args:
            total_views: Total view events.
            total_addtocarts: Total addtocart events.
            total_transactions: Total transaction events.

        Returns:
            FunnelStage enum value.
        """
        if total_transactions > 0:
            return FunnelStage.BUYER
        elif total_addtocarts > 0:
            return FunnelStage.INTENDER
        elif total_views > self.new_user_max_views:
            return FunnelStage.ACTIVE_BROWSER
        else:
            return FunnelStage.NEW_USER

    def get_funnel_stage(self, user_id: int) -> FunnelStage:
        """Get funnel stage for a user.

        Args:
            user_id: User ID.

        Returns:
            FunnelStage for the user, or NEW_USER for unknown users.
        """
        if not self._is_fitted:
            logger.warning("UserFeatureExtractor not fitted, returning NEW_USER")
            return FunnelStage.NEW_USER

        if user_id in self._user_features:
            return self._user_features[user_id].funnel_stage

        # Unknown user defaults to NEW_USER
        return FunnelStage.NEW_USER

    def get_user_features(self, user_id: int) -> UserFeatures | None:
        """Get features for a specific user.

        Args:
            user_id: User ID.

        Returns:
            UserFeatures object or None if user not found.
        """
        return self._user_features.get(user_id)

    def get_user_items(self, user_id: int, event_type: str | None = None) -> set[int]:
        """Get items a user has interacted with.

        Args:
            user_id: User ID.
            event_type: Optional event type filter (view, addtocart, transaction).

        Returns:
            Set of item IDs.
        """
        if user_id not in self._user_items:
            return set()

        if event_type:
            return self._user_items[user_id].get(event_type, set())

        # Return all items across all event types
        all_items = set()
        for items in self._user_items[user_id].values():
            all_items.update(items)
        return all_items

    def get_all_features(self) -> pd.DataFrame:
        """Get features for all users as a DataFrame.

        Returns:
            DataFrame with user features.
        """
        if not self._user_features:
            return pd.DataFrame()

        records = [features.to_dict() for features in self._user_features.values()]
        return pd.DataFrame(records)

    def get_users_by_stage(self, stage: FunnelStage) -> list[int]:
        """Get list of users at a specific funnel stage.

        Args:
            stage: FunnelStage to filter by.

        Returns:
            List of user IDs.
        """
        return [
            user_id
            for user_id, features in self._user_features.items()
            if features.funnel_stage == stage
        ]

    @property
    def is_fitted(self) -> bool:
        """Check if extractor has been fitted."""
        return self._is_fitted
