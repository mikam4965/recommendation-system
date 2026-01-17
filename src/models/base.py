"""Base recommender interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger


class BaseRecommender(ABC):
    """Abstract base class for all recommender models."""

    name: str = "base"

    def __init__(self):
        """Initialize recommender."""
        self._is_fitted = False

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "BaseRecommender":
        """Fit the model on interaction data.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type (view, addtocart, transaction)
                - timestamp: Event timestamp

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations for a single user.

        Args:
            user_id: User ID to get recommendations for.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items the user has interacted with.

        Returns:
            List of (item_id, score) tuples, sorted by score descending.
        """
        pass

    def recommend_batch(
        self,
        user_ids: list[int],
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, list[tuple[int, float]]]:
        """Get recommendations for multiple users.

        Default implementation calls recommend() for each user.
        Override for more efficient batch processing.

        Args:
            user_ids: List of user IDs.
            n_items: Number of items per user.
            exclude_seen: Whether to exclude seen items.

        Returns:
            Dictionary mapping user_id to list of (item_id, score) tuples.
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend(
                user_id, n_items=n_items, exclude_seen=exclude_seen
            )
        return results

    def save(self, path: Path | str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "BaseRecommender":
        """Load model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded model instance.
        """
        path = Path(path)
        model = joblib.load(path)
        logger.info(f"Loaded model from {path}")
        return model

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    def _check_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} model is not fitted. Call fit() first.")

    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters.
        """
        return {}

    def __repr__(self) -> str:
        """String representation."""
        params = self.get_params()
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({params_str})"
