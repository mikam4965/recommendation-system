"""User-based collaborative filtering."""

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


class UserBasedCF(BaseRecommender):
    """User-based collaborative filtering recommender.

    Finds similar users based on interaction patterns and recommends
    items that similar users have interacted with.
    """

    name = "user_cf"

    def __init__(
        self,
        n_neighbors: int = 50,
        weight_view: float | None = None,
        weight_addtocart: float | None = None,
        weight_transaction: float | None = None,
        min_similarity: float = 0.0,
    ):
        """Initialize User-based CF.

        Args:
            n_neighbors: Number of similar users to consider.
            weight_view: Weight for view events.
            weight_addtocart: Weight for addtocart events.
            weight_transaction: Weight for transaction events.
            min_similarity: Minimum similarity threshold.
        """
        super().__init__()

        self.n_neighbors = n_neighbors
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
        self.user_item_matrix: csr_matrix | None = None
        self.user_similarity: np.ndarray | None = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.user_items: dict[int, set[int]] = defaultdict(set)

    def fit(self, interactions: pd.DataFrame) -> "UserBasedCF":
        """Fit User-based CF model.

        Builds user-item matrix and computes user-user similarity.

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
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for item, i in self.item_to_idx.items()}

        logger.info(f"Users: {len(unique_users):,}, Items: {len(unique_items):,}")

        # Build user-item matrix
        interactions = interactions.copy()
        interactions["weight"] = interactions["event"].map(self.event_weights)

        # Aggregate weights per user-item pair
        agg = interactions.groupby(["visitor_id", "item_id"])["weight"].sum().reset_index()

        # Create sparse matrix
        row_indices = agg["visitor_id"].map(self.user_to_idx).values
        col_indices = agg["item_id"].map(self.item_to_idx).values
        data = agg["weight"].values

        self.user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(unique_users), len(unique_items)),
        )

        logger.info(f"Matrix shape: {self.user_item_matrix.shape}")
        logger.info(f"Sparsity: {1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape):.4%}")

        # Compute user similarity (using normalized vectors)
        logger.info("Computing user similarities...")

        # Normalize rows for cosine similarity
        norms = sparse.linalg.norm(self.user_item_matrix, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = self.user_item_matrix.multiply(1 / norms.reshape(-1, 1))

        # Compute similarity matrix (this can be memory intensive for large datasets)
        # For production, consider using approximate methods
        self.user_similarity = cosine_similarity(normalized, dense_output=False)

        # Track user-item interactions
        for _, row in interactions.iterrows():
            self.user_items[row["visitor_id"]].add(row["item_id"])

        self._is_fitted = True
        logger.info("Fitting complete")

        return self

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations based on similar users.

        Args:
            user_id: User ID to get recommendations for.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items the user has interacted with.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        # Handle cold-start users
        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]

        # Get similar users
        similarities = self.user_similarity[user_idx].toarray().flatten()

        # Apply minimum similarity threshold
        similarities[similarities < self.min_similarity] = 0

        # Zero out self-similarity
        similarities[user_idx] = 0

        # Get top k similar users
        top_k_indices = np.argsort(similarities)[::-1][: self.n_neighbors]
        top_k_similarities = similarities[top_k_indices]

        # Filter out users with zero similarity
        valid_mask = top_k_similarities > 0
        top_k_indices = top_k_indices[valid_mask]
        top_k_similarities = top_k_similarities[valid_mask]

        if len(top_k_indices) == 0:
            return []

        # Get items seen by target user
        seen_items = self.user_items.get(user_id, set()) if exclude_seen else set()
        seen_item_indices = {self.item_to_idx[i] for i in seen_items if i in self.item_to_idx}

        # Compute weighted item scores from similar users
        item_scores = np.zeros(self.user_item_matrix.shape[1])

        for neighbor_idx, sim in zip(top_k_indices, top_k_similarities):
            neighbor_items = self.user_item_matrix[neighbor_idx].toarray().flatten()
            item_scores += sim * neighbor_items

        # Normalize by total similarity
        total_sim = top_k_similarities.sum()
        if total_sim > 0:
            item_scores /= total_sim

        # Exclude seen items
        for item_idx in seen_item_indices:
            item_scores[item_idx] = 0

        # Get top items
        top_item_indices = np.argsort(item_scores)[::-1][:n_items]

        recommendations = []
        for item_idx in top_item_indices:
            if item_scores[item_idx] > 0:
                item_id = self.idx_to_item[item_idx]
                recommendations.append((item_id, float(item_scores[item_idx])))

        return recommendations

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "n_neighbors": self.n_neighbors,
            "weight_view": self.weight_view,
            "weight_addtocart": self.weight_addtocart,
            "weight_transaction": self.weight_transaction,
            "min_similarity": self.min_similarity,
        }
