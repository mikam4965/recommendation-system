"""Item-based collaborative filtering."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from src.config import settings
from src.models.base import BaseRecommender


class ItemBasedCF(BaseRecommender):
    """Item-based collaborative filtering recommender.

    Finds similar items based on co-occurrence patterns and recommends
    items similar to those the user has interacted with.
    """

    name = "item_cf"

    def __init__(
        self,
        n_similar: int = 50,
        weight_view: float | None = None,
        weight_addtocart: float | None = None,
        weight_transaction: float | None = None,
        min_similarity: float = 0.0,
    ):
        """Initialize Item-based CF.

        Args:
            n_similar: Number of similar items to consider per item.
            weight_view: Weight for view events.
            weight_addtocart: Weight for addtocart events.
            weight_transaction: Weight for transaction events.
            min_similarity: Minimum similarity threshold.
        """
        super().__init__()

        self.n_similar = n_similar
        self.weight_view = weight_view or settings.event_weight_view
        self.weight_addtocart = weight_addtocart or settings.event_weight_addtocart
        self.weight_transaction = weight_transaction or settings.event_weight_transaction
        self.min_similarity = min_similarity

        self.event_weights = {
            "view": self.weight_view,
            "addtocart": self.weight_addtocart,
            "transaction": self.weight_transaction,
        }

        # Will be populated during fit
        self.item_user_matrix: csr_matrix | None = None
        self.item_similarity: csr_matrix | None = None
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.user_items: dict[int, set[int]] = defaultdict(set)
        self.user_item_weights: dict[int, dict[int, float]] = defaultdict(dict)

    def fit(self, interactions: pd.DataFrame) -> "ItemBasedCF":
        """Fit Item-based CF model.

        Builds item-user matrix and computes item-item similarity.

        Args:
            interactions: DataFrame with visitor_id, item_id, event columns.

        Returns:
            Self.
        """
        logger.info(f"Fitting {self.name} model on {len(interactions):,} interactions")

        # Create mappings
        unique_users = interactions["visitor_id"].unique()
        unique_items = interactions["item_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        logger.info(f"Users: {len(unique_users):,}, Items: {len(unique_items):,}")

        # Build item-user matrix (transposed from user-item)
        interactions = interactions.copy()
        interactions["weight"] = interactions["event"].map(self.event_weights)

        # Aggregate weights per user-item pair
        agg = interactions.groupby(["visitor_id", "item_id"])["weight"].sum().reset_index()

        # Create sparse matrix (item x user)
        row_indices = agg["item_id"].map(self.item_to_idx).values
        col_indices = agg["visitor_id"].map(self.user_to_idx).values
        data = agg["weight"].values

        self.item_user_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(unique_items), len(unique_users)),
        )

        logger.info(f"Matrix shape: {self.item_user_matrix.shape}")
        logger.info(f"Sparsity: {1 - self.item_user_matrix.nnz / np.prod(self.item_user_matrix.shape):.4%}")

        # Compute item similarity
        logger.info("Computing item similarities...")

        # Normalize rows for cosine similarity
        norms = sparse.linalg.norm(self.item_user_matrix, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = self.item_user_matrix.multiply(1 / norms.reshape(-1, 1))

        # Compute similarity matrix
        self.item_similarity = cosine_similarity(normalized, dense_output=False)

        # Track user-item interactions and weights
        for _, row in agg.iterrows():
            user_id = row["visitor_id"]
            item_id = row["item_id"]
            weight = row["weight"]
            self.user_items[user_id].add(item_id)
            self.user_item_weights[user_id][item_id] = weight

        self._is_fitted = True
        logger.info("Fitting complete")

        return self

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations based on similar items.

        Args:
            user_id: User ID to get recommendations for.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items the user has interacted with.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        # Get user's interacted items
        user_item_ids = self.user_items.get(user_id, set())

        if not user_item_ids:
            # Cold-start user
            return []

        # Get item indices for user's items
        user_item_indices = [
            self.item_to_idx[item_id]
            for item_id in user_item_ids
            if item_id in self.item_to_idx
        ]

        if not user_item_indices:
            return []

        # Get user's item weights
        user_weights = self.user_item_weights.get(user_id, {})

        # Compute scores for all items based on similarity to user's items
        item_scores = np.zeros(self.item_similarity.shape[0])

        for item_idx in user_item_indices:
            item_id = self.idx_to_item[item_idx]
            item_weight = user_weights.get(item_id, 1.0)

            # Get similarities to this item
            similarities = self.item_similarity[item_idx].toarray().flatten()

            # Apply minimum threshold
            similarities[similarities < self.min_similarity] = 0

            # Weight by user's interaction strength with this item
            item_scores += item_weight * similarities

        # Exclude seen items if requested
        if exclude_seen:
            for item_idx in user_item_indices:
                item_scores[item_idx] = 0

        # Get top items
        top_item_indices = np.argsort(item_scores)[::-1][:n_items]

        recommendations = []
        for item_idx in top_item_indices:
            if item_scores[item_idx] > 0:
                item_id = self.idx_to_item[item_idx]
                recommendations.append((item_id, float(item_scores[item_idx])))

        return recommendations

    def get_similar_items(
        self, item_id: int, n_items: int = 10
    ) -> list[tuple[int, float]]:
        """Get items similar to a given item.

        Args:
            item_id: Item ID to find similar items for.
            n_items: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        self._check_fitted()

        if item_id not in self.item_to_idx:
            return []

        item_idx = self.item_to_idx[item_id]
        similarities = self.item_similarity[item_idx].toarray().flatten()

        # Zero out self-similarity
        similarities[item_idx] = 0

        # Get top similar items
        top_indices = np.argsort(similarities)[::-1][:n_items]

        results = []
        for idx in top_indices:
            if similarities[idx] > self.min_similarity:
                results.append((self.idx_to_item[idx], float(similarities[idx])))

        return results

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "n_similar": self.n_similar,
            "weight_view": self.weight_view,
            "weight_addtocart": self.weight_addtocart,
            "weight_transaction": self.weight_transaction,
            "min_similarity": self.min_similarity,
        }
