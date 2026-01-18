"""Two-Tower Model for efficient recommendation retrieval.

Two-tower (dual encoder) architecture with separate user and item encoders.
Uses FAISS for approximate nearest neighbor search at inference time.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.base import BaseRecommender

# Optional FAISS import
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Install with: pip install faiss-cpu")


class UserTower(nn.Module):
    """User encoder tower.

    Encodes user features into a dense embedding.
    """

    def __init__(
        self,
        n_users: int,
        embedding_dim: int = 64,
        user_feature_dim: int = 0,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        """Initialize User Tower.

        Args:
            n_users: Number of users.
            embedding_dim: Output embedding dimension.
            user_feature_dim: Dimension of additional user features.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.has_features = user_feature_dim > 0

        # User ID embedding
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # Feature processing if features are provided
        if self.has_features:
            hidden_dims = hidden_dims or [128, 64]
            layers = []
            input_dim = embedding_dim + user_feature_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, embedding_dim))
            self.feature_net = nn.Sequential(*layers)
        else:
            # Simple projection
            self.feature_net = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User ID tensor [batch_size].
            user_features: Optional user features [batch_size, feature_dim].

        Returns:
            User embeddings [batch_size, embedding_dim].
        """
        user_emb = self.user_embedding(user_ids)

        if self.has_features and user_features is not None:
            x = torch.cat([user_emb, user_features], dim=-1)
        else:
            x = user_emb

        x = self.feature_net(x)

        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)

        return x


class ItemTower(nn.Module):
    """Item encoder tower.

    Encodes item features into a dense embedding.
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        item_feature_dim: int = 0,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        """Initialize Item Tower.

        Args:
            n_items: Number of items.
            embedding_dim: Output embedding dimension.
            item_feature_dim: Dimension of additional item features.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.has_features = item_feature_dim > 0

        # Item ID embedding (0 reserved for padding)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])

        # Feature processing if features are provided
        if self.has_features:
            hidden_dims = hidden_dims or [128, 64]
            layers = []
            input_dim = embedding_dim + item_feature_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                input_dim = hidden_dim

            layers.append(nn.Linear(input_dim, embedding_dim))
            self.feature_net = nn.Sequential(*layers)
        else:
            # Simple projection
            self.feature_net = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )

    def forward(
        self,
        item_ids: torch.Tensor,
        item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            item_ids: Item ID tensor [batch_size] (1-indexed).
            item_features: Optional item features [batch_size, feature_dim].

        Returns:
            Item embeddings [batch_size, embedding_dim].
        """
        item_emb = self.item_embedding(item_ids)

        if self.has_features and item_features is not None:
            x = torch.cat([item_emb, item_features], dim=-1)
        else:
            x = item_emb

        x = self.feature_net(x)

        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=-1)

        return x


class TwoTowerModule(nn.Module):
    """Two-Tower model combining user and item encoders."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        user_feature_dim: int = 0,
        item_feature_dim: int = 0,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        temperature: float = 0.1,
    ):
        """Initialize Two-Tower model.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            embedding_dim: Embedding dimension.
            user_feature_dim: User feature dimension.
            item_feature_dim: Item feature dimension.
            hidden_dims: Hidden layer dimensions for both towers.
            dropout: Dropout rate.
            temperature: Temperature for softmax (lower = sharper).
        """
        super().__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.user_tower = UserTower(
            n_users=n_users,
            embedding_dim=embedding_dim,
            user_feature_dim=user_feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.item_tower = ItemTower(
            n_items=n_items,
            embedding_dim=embedding_dim,
            item_feature_dim=item_feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
        item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass computing similarity scores.

        Args:
            user_ids: User IDs [batch_size].
            item_ids: Item IDs [batch_size] (1-indexed).
            user_features: Optional user features.
            item_features: Optional item features.

        Returns:
            Similarity scores [batch_size].
        """
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)

        # Dot product (embeddings are already L2-normalized)
        scores = torch.sum(user_emb * item_emb, dim=-1)

        return scores

    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get user embeddings."""
        return self.user_tower(user_ids, user_features)

    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get item embeddings."""
        return self.item_tower(item_ids, item_features)


class TwoTowerDataset(Dataset):
    """Dataset for Two-Tower training with negative sampling."""

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        n_items: int,
        user_items: dict[int, set[int]],
        negative_samples: int = 4,
    ):
        """Initialize dataset.

        Args:
            user_ids: Array of user indices.
            item_ids: Array of positive item indices (1-indexed).
            n_items: Total number of items.
            user_items: Dict mapping user_idx to set of interacted item indices.
            negative_samples: Number of negative samples per positive.
        """
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.n_items = n_items
        self.user_items = user_items
        self.negative_samples = negative_samples

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> tuple[int, np.ndarray]:
        """Get a training sample.

        Returns tuple of (user_id, item_ids) where item_ids[0] is positive
        and the rest are negative samples.
        """
        user_id = self.user_ids[idx]
        pos_item = self.item_ids[idx]

        # Sample negative items
        user_item_set = self.user_items.get(user_id, set())
        neg_items = []

        while len(neg_items) < self.negative_samples:
            neg_item = np.random.randint(1, self.n_items + 1)  # 1-indexed
            if neg_item not in user_item_set:
                neg_items.append(neg_item)

        items = np.array([pos_item] + neg_items)

        return user_id, items


class TwoTowerRecommender(BaseRecommender):
    """Two-Tower recommender with FAISS-based retrieval.

    Uses separate encoders for users and items, enabling efficient
    approximate nearest neighbor search at inference time.
    """

    name = "two_tower"

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        temperature: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 10,
        negative_samples: int = 4,
        use_faiss: bool = True,
        faiss_nlist: int = 100,
        device: str | None = None,
        random_state: int = 42,
    ):
        """Initialize Two-Tower recommender.

        Args:
            embedding_dim: Embedding dimension.
            hidden_dims: Hidden layer dimensions.
            dropout: Dropout rate.
            temperature: Temperature for contrastive loss.
            learning_rate: Learning rate.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            negative_samples: Negative samples per positive in batch.
            use_faiss: Whether to use FAISS for retrieval.
            faiss_nlist: Number of clusters for IVF index.
            device: Device (cuda/cpu).
            random_state: Random seed.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_nlist = faiss_nlist
        self.random_state = random_state

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model and mappings
        self._model: TwoTowerModule | None = None
        self._user_to_idx: dict[int, int] = {}
        self._idx_to_user: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._user_items: dict[int, set[int]] = defaultdict(set)
        self._n_items: int = 0

        # FAISS index
        self._faiss_index = None
        self._item_embeddings: np.ndarray | None = None

    def fit(self, interactions: pd.DataFrame) -> "TwoTowerRecommender":
        """Fit Two-Tower model.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name} model on {len(interactions):,} interactions")

        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build mappings
        unique_users = interactions["visitor_id"].unique()
        unique_items = interactions["item_id"].unique()

        self._user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self._idx_to_user = {idx: user for user, idx in self._user_to_idx.items()}
        self._item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}  # 1-indexed
        self._idx_to_item = {idx: item for item, idx in self._item_to_idx.items()}

        n_users = len(unique_users)
        self._n_items = len(unique_items)

        logger.info(f"Users: {n_users:,}, Items: {self._n_items:,}")
        logger.info(f"Device: {self.device}")

        # Build user-item interaction sets
        for _, row in interactions.iterrows():
            user_idx = self._user_to_idx[row["visitor_id"]]
            item_idx = self._item_to_idx[row["item_id"]]
            self._user_items[user_idx].add(item_idx)

        # Create model
        self._model = TwoTowerModule(
            n_users=n_users,
            n_items=self._n_items,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            temperature=self.temperature,
        )
        self._model.to(self.device)

        # Prepare training data
        user_ids = []
        item_ids = []
        for _, row in interactions.drop_duplicates(["visitor_id", "item_id"]).iterrows():
            user_ids.append(self._user_to_idx[row["visitor_id"]])
            item_ids.append(self._item_to_idx[row["item_id"]])

        dataset = TwoTowerDataset(
            user_ids=np.array(user_ids),
            item_ids=np.array(item_ids),
            n_items=self._n_items,
            user_items=self._user_items,
            negative_samples=self.negative_samples,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Train model
        self._train_model(dataloader)

        # Build FAISS index
        if self.use_faiss:
            self._build_faiss_index()

        self._is_fitted = True
        return self

    def _train_model(self, dataloader: DataLoader) -> None:
        """Train the Two-Tower model using in-batch negatives.

        Args:
            dataloader: Training data loader.
        """
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
        )

        self._model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for user_ids, item_batch in progress:
                user_ids = user_ids.to(self.device)
                item_batch = item_batch.to(self.device)  # [B, 1 + neg_samples]

                batch_size = user_ids.size(0)
                n_items_per_user = item_batch.size(1)

                optimizer.zero_grad()

                # Get user embeddings
                user_emb = self._model.get_user_embeddings(user_ids)  # [B, E]

                # Get item embeddings for all items in batch
                item_ids_flat = item_batch.view(-1)  # [B * (1 + neg)]
                item_emb_flat = self._model.get_item_embeddings(item_ids_flat)  # [B * (1 + neg), E]
                item_emb = item_emb_flat.view(batch_size, n_items_per_user, -1)  # [B, 1 + neg, E]

                # Compute scores: [B, 1 + neg]
                scores = torch.bmm(item_emb, user_emb.unsqueeze(-1)).squeeze(-1)
                scores = scores / self.temperature

                # Contrastive loss: positive is at index 0
                labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(scores, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / n_batches
            logger.info(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

    def _build_faiss_index(self) -> None:
        """Build FAISS index from item embeddings."""
        logger.info("Building FAISS index...")

        self._model.eval()

        # Get all item embeddings
        all_item_ids = torch.arange(1, self._n_items + 1, device=self.device)

        with torch.no_grad():
            # Process in batches
            embeddings = []
            batch_size = 10000

            for i in range(0, len(all_item_ids), batch_size):
                batch_ids = all_item_ids[i:i + batch_size]
                batch_emb = self._model.get_item_embeddings(batch_ids)
                embeddings.append(batch_emb.cpu().numpy())

            self._item_embeddings = np.vstack(embeddings).astype(np.float32)

        # Build FAISS index
        d = self.embedding_dim

        if self._n_items < self.faiss_nlist * 50:
            # Use flat index for small catalogs
            self._faiss_index = faiss.IndexFlatIP(d)
        else:
            # Use IVF index for larger catalogs
            quantizer = faiss.IndexFlatIP(d)
            self._faiss_index = faiss.IndexIVFFlat(
                quantizer, d, min(self.faiss_nlist, self._n_items // 50),
                faiss.METRIC_INNER_PRODUCT
            )
            self._faiss_index.train(self._item_embeddings)

        self._faiss_index.add(self._item_embeddings)

        logger.info(f"FAISS index built with {self._faiss_index.ntotal} items")

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations for a user.

        Args:
            user_id: User ID.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude seen items.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        if user_id not in self._user_to_idx:
            logger.debug(f"Cold-start user {user_id}")
            return []

        user_idx = self._user_to_idx[user_id]

        # Get user embedding
        self._model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            user_emb = self._model.get_user_embeddings(user_tensor)
            user_emb = user_emb.cpu().numpy().astype(np.float32)

        # Get exclusion set
        if exclude_seen:
            exclude_set = self._user_items.get(user_idx, set())
        else:
            exclude_set = set()

        # Use FAISS for retrieval
        if self.use_faiss and self._faiss_index is not None:
            # Request more items to account for exclusions
            k = min(n_items + len(exclude_set) + 10, self._n_items)
            scores, indices = self._faiss_index.search(user_emb, k)

            recommendations = []
            for idx, score in zip(indices[0], scores[0]):
                item_idx = idx + 1  # FAISS uses 0-indexed, our items are 1-indexed
                if item_idx in exclude_set:
                    continue
                if item_idx not in self._idx_to_item:
                    continue

                item_id = self._idx_to_item[item_idx]
                recommendations.append((item_id, float(score)))

                if len(recommendations) >= n_items:
                    break

            return recommendations

        else:
            # Brute force fallback
            with torch.no_grad():
                all_item_ids = torch.arange(1, self._n_items + 1, device=self.device)
                item_emb = self._model.get_item_embeddings(all_item_ids)

                scores = torch.matmul(
                    torch.from_numpy(user_emb).to(self.device),
                    item_emb.T
                ).squeeze(0).cpu().numpy()

            sorted_indices = np.argsort(scores)[::-1]

            recommendations = []
            for idx in sorted_indices:
                item_idx = idx + 1
                if item_idx in exclude_set:
                    continue
                if item_idx not in self._idx_to_item:
                    continue

                item_id = self._idx_to_item[item_idx]
                recommendations.append((item_id, float(scores[idx])))

                if len(recommendations) >= n_items:
                    break

            return recommendations

    def get_user_embedding(self, user_id: int) -> np.ndarray | None:
        """Get embedding for a user.

        Args:
            user_id: User ID.

        Returns:
            User embedding array or None if not found.
        """
        self._check_fitted()

        if user_id not in self._user_to_idx:
            return None

        user_idx = self._user_to_idx[user_id]

        self._model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            user_emb = self._model.get_user_embeddings(user_tensor)
            return user_emb.cpu().numpy().squeeze()

    def get_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings.

        Returns:
            Item embeddings array [n_items, embedding_dim].
        """
        self._check_fitted()

        if self._item_embeddings is not None:
            return self._item_embeddings

        self._model.eval()
        with torch.no_grad():
            all_item_ids = torch.arange(1, self._n_items + 1, device=self.device)
            item_emb = self._model.get_item_embeddings(all_item_ids)
            return item_emb.cpu().numpy()

    def get_similar_items(
        self,
        item_id: int,
        n_items: int = 10,
    ) -> list[tuple[int, float]]:
        """Get similar items based on item embeddings.

        Args:
            item_id: Query item ID.
            n_items: Number of similar items.

        Returns:
            List of (item_id, similarity) tuples.
        """
        self._check_fitted()

        if item_id not in self._item_to_idx:
            return []

        item_idx = self._item_to_idx[item_id]

        # Get query item embedding
        query_emb = self._item_embeddings[item_idx - 1:item_idx].astype(np.float32)

        if self.use_faiss and self._faiss_index is not None:
            # +1 to exclude the query item itself
            scores, indices = self._faiss_index.search(query_emb, n_items + 1)

            recommendations = []
            for idx, score in zip(indices[0], scores[0]):
                result_idx = idx + 1
                if result_idx == item_idx:  # Skip query item
                    continue
                if result_idx not in self._idx_to_item:
                    continue

                result_id = self._idx_to_item[result_idx]
                recommendations.append((result_id, float(score)))

                if len(recommendations) >= n_items:
                    break

            return recommendations

        else:
            # Brute force
            similarities = np.dot(self._item_embeddings, query_emb.T).squeeze()
            sorted_indices = np.argsort(similarities)[::-1]

            recommendations = []
            for idx in sorted_indices:
                result_idx = idx + 1
                if result_idx == item_idx:
                    continue
                if result_idx not in self._idx_to_item:
                    continue

                result_id = self._idx_to_item[result_idx]
                recommendations.append((result_id, float(similarities[idx])))

                if len(recommendations) >= n_items:
                    break

            return recommendations

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "negative_samples": self.negative_samples,
            "use_faiss": self.use_faiss,
            "device": self.device,
        }

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return self._n_items

    @property
    def n_users(self) -> int:
        """Number of users in the model."""
        return len(self._user_to_idx)
