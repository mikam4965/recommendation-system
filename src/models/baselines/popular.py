"""Popularity-based recommender."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings
from src.models.base import BaseRecommender


class PopularItemsRecommender(BaseRecommender):
    """Recommender based on item popularity.

    Recommends items based on weighted interaction counts.
    Event weights: transaction > addtocart > view.
    """

    name = "popular"

    def __init__(
        self,
        weight_view: float | None = None,
        weight_addtocart: float | None = None,
        weight_transaction: float | None = None,
    ):
        """Initialize popularity recommender.

        Args:
            weight_view: Weight for view events.
            weight_addtocart: Weight for addtocart events.
            weight_transaction: Weight for transaction events.
        """
        super().__init__()

        self.weight_view = weight_view or settings.event_weight_view
        self.weight_addtocart = weight_addtocart or settings.event_weight_addtocart
        self.weight_transaction = weight_transaction or settings.event_weight_transaction

        self.event_weights = {
            "view": self.weight_view,
            "addtocart": self.weight_addtocart,
            "transaction": self.weight_transaction,
        }

        # Will be populated during fit
        self.item_scores: dict[int, float] = {}
        self.sorted_items: list[int] = []
        self.user_items: dict[int, set[int]] = defaultdict(set)

    def fit(self, interactions: pd.DataFrame) -> "PopularItemsRecommender":
        """Fit popularity model.

        Calculates weighted popularity scores for all items.

        Args:
            interactions: DataFrame with visitor_id, item_id, event columns.

        Returns:
            Self.
        """
        logger.info(f"Fitting {self.name} model on {len(interactions):,} interactions")

        # Calculate weighted scores
        interactions = interactions.copy()
        interactions["weight"] = interactions["event"].map(self.event_weights)

        # Aggregate scores per item
        item_scores = interactions.groupby("item_id")["weight"].sum()
        self.item_scores = item_scores.to_dict()

        # Sort items by score (descending)
        self.sorted_items = list(
            item_scores.sort_values(ascending=False).index
        )

        # Track user-item interactions for excluding seen items
        for _, row in interactions.iterrows():
            self.user_items[row["visitor_id"]].add(row["item_id"])

        self._is_fitted = True

        logger.info(f"Fitted on {len(self.item_scores):,} items")
        logger.info(f"Top item score: {max(self.item_scores.values()):.2f}")

        return self

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get popular item recommendations.

        Args:
            user_id: User ID (used only for excluding seen items).
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items user has interacted with.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        seen_items = self.user_items.get(user_id, set()) if exclude_seen else set()

        recommendations = []
        for item_id in self.sorted_items:
            if item_id not in seen_items:
                recommendations.append((item_id, self.item_scores[item_id]))
                if len(recommendations) >= n_items:
                    break

        return recommendations

    def recommend_batch(
        self,
        user_ids: list[int],
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, list[tuple[int, float]]]:
        """Get recommendations for multiple users efficiently.

        Args:
            user_ids: List of user IDs.
            n_items: Number of items per user.
            exclude_seen: Whether to exclude seen items.

        Returns:
            Dictionary mapping user_id to recommendations.
        """
        self._check_fitted()

        # For popularity model, all users get same items (minus seen)
        # Can optimize by computing once if exclude_seen=False
        if not exclude_seen:
            common_recs = [
                (item_id, self.item_scores[item_id])
                for item_id in self.sorted_items[:n_items]
            ]
            return {user_id: common_recs for user_id in user_ids}

        # Otherwise, call individual recommend
        return super().recommend_batch(user_ids, n_items, exclude_seen)

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "weight_view": self.weight_view,
            "weight_addtocart": self.weight_addtocart,
            "weight_transaction": self.weight_transaction,
        }
