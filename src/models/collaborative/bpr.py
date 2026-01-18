"""Bayesian Personalized Ranking (BPR) recommender using implicit library."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking
from loguru import logger
from scipy import sparse

from src.config import settings
from src.models.base import BaseRecommender


class BPRRecommender(BaseRecommender):
    """BPR-based collaborative filtering recommender.

    Uses Bayesian Personalized Ranking for implicit feedback matrix factorization.
    Optimizes AUC directly using pairwise ranking loss.
    Based on the implicit library's BayesianPersonalizedRanking implementation.
    """

    name = "bpr"

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
        iterations: int = 100,
        use_gpu: bool = False,
        weight_view: float | None = None,
        weight_addtocart: float | None = None,
        weight_transaction: float | None = None,
        random_state: int = 42,
    ):
        """Initialize BPR recommender.

        Args:
            factors: Number of latent factors.
            learning_rate: Learning rate for SGD.
            regularization: L2 regularization coefficient.
            iterations: Number of training iterations.
            use_gpu: Whether to use GPU acceleration.
            weight_view: Weight for view events.
            weight_addtocart: Weight for addtocart events.
            weight_transaction: Weight for transaction events.
            random_state: Random seed for reproducibility.
        """
        super().__init__()

        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.random_state = random_state

        # Event weights
        self.weight_view = weight_view if weight_view is not None else settings.event_weight_view
        self.weight_addtocart = (
            weight_addtocart if weight_addtocart is not None else settings.event_weight_addtocart
        )
        self.weight_transaction = (
            weight_transaction
            if weight_transaction is not None
            else settings.event_weight_transaction
        )

        self.event_weights = {
            "view": self.weight_view,
            "addtocart": self.weight_addtocart,
            "transaction": self.weight_transaction,
        }

        # Model and mappings
        self._model: BayesianPersonalizedRanking | None = None
        self._user_to_idx: dict[int, int] = {}
        self._idx_to_user: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._user_items: dict[int, set[int]] = defaultdict(set)
        self._interaction_matrix: sparse.csr_matrix | None = None

    def fit(self, interactions: pd.DataFrame) -> "BPRRecommender":
        """Fit BPR model on interaction data.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type (view, addtocart, transaction)

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name} model on {len(interactions):,} interactions")

        # Build user and item mappings
        unique_users = interactions["visitor_id"].unique()
        unique_items = interactions["item_id"].unique()

        self._user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self._idx_to_user = {idx: user for user, idx in self._user_to_idx.items()}
        self._item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self._idx_to_item = {idx: item for item, idx in self._item_to_idx.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)

        logger.info(f"Users: {n_users:,}, Items: {n_items:,}")

        # Build interaction matrix with event weights
        interactions = interactions.copy()
        interactions["weight"] = interactions["event"].map(self.event_weights)

        # Aggregate weights for duplicate (user, item) pairs
        aggregated = (
            interactions.groupby(["visitor_id", "item_id"])["weight"].sum().reset_index()
        )

        # Map to indices
        user_indices = aggregated["visitor_id"].map(self._user_to_idx).values
        item_indices = aggregated["item_id"].map(self._item_to_idx).values
        weights = aggregated["weight"].values

        # Create sparse matrix (item x user for implicit library)
        self._interaction_matrix = sparse.csr_matrix(
            (weights, (item_indices, user_indices)),
            shape=(n_items, n_users),
            dtype=np.float32,
        )

        # Track user-item interactions for excluding seen items
        for _, row in interactions.iterrows():
            self._user_items[row["visitor_id"]].add(row["item_id"])

        # Initialize and fit model
        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
        )

        logger.info("Training BPR model...")
        self._model.fit(self._interaction_matrix, show_progress=True)

        self._is_fitted = True
        logger.info(f"BPR model fitted with {self.factors} factors")

        return self

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations for a user.

        Args:
            user_id: User ID to get recommendations for.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items the user has seen.

        Returns:
            List of (item_id, score) tuples, sorted by score descending.
        """
        self._check_fitted()

        # Handle cold-start user
        if user_id not in self._user_to_idx:
            logger.debug(f"Cold-start user {user_id}, returning empty recommendations")
            return []

        user_idx = self._user_to_idx[user_id]

        # Get seen items for filtering
        seen_items = self._user_items.get(user_id, set()) if exclude_seen else set()

        # Request more items to account for filtering
        # Note: filter_already_liked_items has a bug in implicit 0.7.x, so we filter manually
        n_request = min(n_items + len(seen_items), self.n_items)

        # Get recommendations without built-in filtering
        item_indices, scores = self._model.recommend(
            user_idx,
            self._interaction_matrix.T.tocsr()[user_idx],
            N=n_request,
            filter_already_liked_items=False,
        )

        # Convert back to original IDs and filter seen items manually
        recommendations = []
        for idx, score in zip(item_indices, scores):
            item_id = self._idx_to_item[idx]
            if item_id not in seen_items:
                recommendations.append((item_id, float(score)))
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
            Dictionary mapping user_id to list of (item_id, score) tuples.
        """
        self._check_fitted()

        results = {}

        # Use single-user recommend for each user (workaround for implicit bug)
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, n_items, exclude_seen)

        return results

    def get_similar_items(
        self,
        item_id: int,
        n_items: int = 10,
    ) -> list[tuple[int, float]]:
        """Get similar items based on learned embeddings.

        Args:
            item_id: Item ID to find similar items for.
            n_items: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        self._check_fitted()

        if item_id not in self._item_to_idx:
            logger.debug(f"Unknown item {item_id}")
            return []

        item_idx = self._item_to_idx[item_id]

        # Get similar items using implicit library
        similar_indices, scores = self._model.similar_items(item_idx, N=n_items + 1)

        # Convert to original IDs (skip first which is the item itself)
        similar_items = [
            (self._idx_to_item[idx], float(score))
            for idx, score in zip(similar_indices[1:], scores[1:])
        ]

        return similar_items

    def get_user_embedding(self, user_id: int) -> np.ndarray | None:
        """Get learned user embedding.

        Args:
            user_id: User ID.

        Returns:
            User embedding vector or None if user not found.
        """
        self._check_fitted()

        if user_id not in self._user_to_idx:
            return None

        user_idx = self._user_to_idx[user_id]
        return self._model.user_factors[user_idx]

    def get_item_embedding(self, item_id: int) -> np.ndarray | None:
        """Get learned item embedding.

        Args:
            item_id: Item ID.

        Returns:
            Item embedding vector or None if item not found.
        """
        self._check_fitted()

        if item_id not in self._item_to_idx:
            return None

        item_idx = self._item_to_idx[item_id]
        return self._model.item_factors[item_idx]

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "factors": self.factors,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "use_gpu": self.use_gpu,
            "weight_view": self.weight_view,
            "weight_addtocart": self.weight_addtocart,
            "weight_transaction": self.weight_transaction,
        }

    @property
    def n_users(self) -> int:
        """Number of users in the model."""
        return len(self._user_to_idx)

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return len(self._item_to_idx)
