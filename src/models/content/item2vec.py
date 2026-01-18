"""Item2Vec content-based recommender using Word2Vec on sessions."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseRecommender


class Item2VecRecommender(BaseRecommender):
    """Item2Vec recommender using Word2Vec on user sessions.

    Treats sessions as "sentences" and items as "words" to learn
    item embeddings that capture co-occurrence patterns.
    """

    name = "item2vec"

    def __init__(
        self,
        embedding_dim: int = 64,
        window: int = 5,
        min_count: int = 5,
        epochs: int = 10,
        workers: int = 4,
        negative: int = 5,
        sg: int = 1,  # Skip-gram (1) vs CBOW (0)
        random_state: int = 42,
    ):
        """Initialize Item2Vec recommender.

        Args:
            embedding_dim: Dimension of item embeddings.
            window: Context window size.
            min_count: Minimum item frequency to be included.
            epochs: Number of training epochs.
            workers: Number of worker threads.
            negative: Number of negative samples.
            sg: Skip-gram (1) or CBOW (0).
            random_state: Random seed for reproducibility.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.workers = workers
        self.negative = negative
        self.sg = sg
        self.random_state = random_state

        # Model and data
        self._model: Word2Vec | None = None
        self._user_items: dict[int, list[int]] = defaultdict(list)
        self._item_set: set[int] = set()

    def fit(self, interactions: pd.DataFrame) -> "Item2VecRecommender":
        """Fit Item2Vec model on interaction data.

        Builds sessions and trains Word2Vec on session sequences.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - timestamp: Event timestamp
                - session_id (optional): Pre-built session IDs

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name} model on {len(interactions):,} interactions")

        # Build sessions (sequences of items)
        sessions = self._build_sessions(interactions)

        logger.info(f"Built {len(sessions):,} sessions for training")

        # Filter empty sessions and convert items to strings (Word2Vec requirement)
        sessions_str = [[str(item) for item in session] for session in sessions if session]

        if not sessions_str:
            raise ValueError("No valid sessions found for training")

        # Track user items for recommendation
        for _, row in interactions.iterrows():
            self._user_items[row["visitor_id"]].append(row["item_id"])
            self._item_set.add(row["item_id"])

        # Train Word2Vec model
        logger.info("Training Word2Vec model...")
        self._model = Word2Vec(
            sentences=sessions_str,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            workers=self.workers,
            negative=self.negative,
            sg=self.sg,
            seed=self.random_state,
        )

        self._is_fitted = True

        # Log stats
        vocab_size = len(self._model.wv)
        logger.info(f"Item2Vec model fitted: {vocab_size:,} items in vocabulary")
        logger.info(f"Total unique items: {len(self._item_set):,}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        return self

    def _build_sessions(self, interactions: pd.DataFrame) -> list[list[int]]:
        """Build session sequences from interactions.

        Args:
            interactions: Interaction DataFrame.

        Returns:
            List of sessions, where each session is a list of item IDs.
        """
        sessions = []

        if "session_id" in interactions.columns:
            # Use pre-built sessions
            for session_id, session_data in interactions.groupby("session_id"):
                session_sorted = session_data.sort_values("timestamp")
                items = session_sorted["item_id"].tolist()
                if len(items) >= 2:  # Need at least 2 items for context
                    sessions.append(items)
        else:
            # Fall back to user-level sequences
            for user_id, user_data in interactions.groupby("visitor_id"):
                user_sorted = user_data.sort_values("timestamp")
                items = user_sorted["item_id"].tolist()
                if len(items) >= 2:
                    sessions.append(items)

        return sessions

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations based on user's item history.

        Computes the average embedding of user's items and finds similar items.

        Args:
            user_id: User ID to get recommendations for.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items the user has seen.

        Returns:
            List of (item_id, score) tuples, sorted by score descending.
        """
        self._check_fitted()

        # Get user's items
        user_items = self._user_items.get(user_id, [])
        if not user_items:
            logger.debug(f"No items for user {user_id}, returning empty recommendations")
            return []

        # Use most recent items for the query
        recent_items = user_items[-50:]  # Last 50 items

        return self.recommend_from_items(
            items=recent_items,
            n_items=n_items,
            exclude_items=set(user_items) if exclude_seen else None,
        )

    def recommend_from_items(
        self,
        items: list[int],
        n_items: int = 10,
        exclude_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Get recommendations based on a list of items (e.g., current session).

        Args:
            items: List of item IDs to base recommendations on.
            n_items: Number of items to recommend.
            exclude_items: Items to exclude from recommendations.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        if not items:
            return []

        # Get embeddings for known items
        item_embeddings = []
        for item_id in items:
            item_str = str(item_id)
            if item_str in self._model.wv:
                item_embeddings.append(self._model.wv[item_str])

        if not item_embeddings:
            logger.debug("No items in vocabulary, returning empty recommendations")
            return []

        # Compute average embedding
        query_embedding = np.mean(item_embeddings, axis=0)

        # Find most similar items
        exclude_items = exclude_items or set()
        exclude_str = {str(item_id) for item_id in exclude_items}

        # Get all items and their similarities
        all_items = []
        all_similarities = []

        for item_str in self._model.wv.index_to_key:
            if item_str not in exclude_str:
                item_vec = self._model.wv[item_str]
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    item_vec.reshape(1, -1),
                )[0][0]
                all_items.append(int(item_str))
                all_similarities.append(similarity)

        # Sort by similarity and take top n
        sorted_indices = np.argsort(all_similarities)[::-1][:n_items]

        recommendations = [
            (all_items[idx], float(all_similarities[idx]))
            for idx in sorted_indices
        ]

        return recommendations

    def get_similar_items(
        self,
        item_id: int,
        n_items: int = 10,
    ) -> list[tuple[int, float]]:
        """Get similar items based on embeddings.

        Args:
            item_id: Item ID to find similar items for.
            n_items: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        self._check_fitted()

        item_str = str(item_id)
        if item_str not in self._model.wv:
            logger.debug(f"Item {item_id} not in vocabulary")
            return []

        # Use gensim's most_similar
        similar = self._model.wv.most_similar(item_str, topn=n_items)

        return [(int(item_str), float(score)) for item_str, score in similar]

    def get_item_embedding(self, item_id: int) -> np.ndarray | None:
        """Get learned item embedding.

        Args:
            item_id: Item ID.

        Returns:
            Item embedding vector or None if item not in vocabulary.
        """
        self._check_fitted()

        item_str = str(item_id)
        if item_str not in self._model.wv:
            return None

        return self._model.wv[item_str]

    def get_all_embeddings(self) -> tuple[list[int], np.ndarray]:
        """Get all item embeddings.

        Returns:
            Tuple of (item_ids, embedding_matrix).
        """
        self._check_fitted()

        item_ids = [int(item_str) for item_str in self._model.wv.index_to_key]
        embeddings = self._model.wv.vectors

        return item_ids, embeddings

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "embedding_dim": self.embedding_dim,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "negative": self.negative,
            "sg": self.sg,
        }

    @property
    def vocab_size(self) -> int:
        """Number of items in the vocabulary."""
        if self._model is None:
            return 0
        return len(self._model.wv)

    @property
    def n_items(self) -> int:
        """Total number of unique items seen during training."""
        return len(self._item_set)
