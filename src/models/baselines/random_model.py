"""Random recommender baseline."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Random recommender baseline.

    Recommends random items from the item catalog.
    Useful as a lower-bound baseline for comparison.
    """

    name = "random"

    def __init__(self, seed: int | None = None):
        """Initialize random recommender.

        Args:
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Will be populated during fit
        self.all_items: list[int] = []
        self.user_items: dict[int, set[int]] = defaultdict(set)

    def fit(self, interactions: pd.DataFrame) -> "RandomRecommender":
        """Fit random model.

        Just collects the list of all items and user interactions.

        Args:
            interactions: DataFrame with visitor_id, item_id columns.

        Returns:
            Self.
        """
        logger.info(f"Fitting {self.name} model on {len(interactions):,} interactions")

        # Get all unique items
        self.all_items = list(interactions["item_id"].unique())

        # Track user-item interactions for excluding seen items
        for _, row in interactions.iterrows():
            self.user_items[row["visitor_id"]].add(row["item_id"])

        self._is_fitted = True

        logger.info(f"Fitted on {len(self.all_items):,} items")

        return self

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get random item recommendations.

        Args:
            user_id: User ID (used only for excluding seen items).
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items user has interacted with.

        Returns:
            List of (item_id, score) tuples. Scores are random [0, 1].
        """
        self._check_fitted()

        seen_items = self.user_items.get(user_id, set()) if exclude_seen else set()

        # Get candidate items
        candidates = [item for item in self.all_items if item not in seen_items]

        # Sample random items
        n_to_sample = min(n_items, len(candidates))
        if n_to_sample == 0:
            return []

        sampled_indices = self.rng.choice(
            len(candidates), size=n_to_sample, replace=False
        )
        sampled_items = [candidates[i] for i in sampled_indices]

        # Assign random scores
        scores = self.rng.random(n_to_sample)

        # Sort by score and return
        recommendations = list(zip(sampled_items, scores))
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {"seed": self.seed}
