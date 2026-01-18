"""Neural Collaborative Filtering (NCF) recommender using PyTorch."""

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.base import BaseRecommender


class GMF(nn.Module):
    """Generalized Matrix Factorization component.

    GMF uses element-wise product of user and item embeddings.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int):
        """Initialize GMF.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            embedding_dim: Dimension of embeddings.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User ID tensor.
            item_ids: Item ID tensor.

        Returns:
            Element-wise product of embeddings.
        """
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        return user_embed * item_embed


class MLP(nn.Module):
    """Multi-Layer Perceptron component.

    MLP learns non-linear interactions between user and item embeddings.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        layers: list[int],
        dropout: float = 0.2,
    ):
        """Initialize MLP.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            embedding_dim: Dimension of embeddings.
            layers: List of hidden layer sizes.
            dropout: Dropout rate.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        # Build MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2

        for layer_size in layers:
            mlp_layers.append(nn.Linear(input_size, layer_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_size = layer_size

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User ID tensor.
            item_ids: Item ID tensor.

        Returns:
            MLP output features.
        """
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)

        # Concatenate embeddings
        concat = torch.cat([user_embed, item_embed], dim=-1)
        return self.mlp(concat)


class NeuMF(nn.Module):
    """Neural Matrix Factorization model combining GMF and MLP.

    Combines GMF and MLP components for learning both linear and
    non-linear user-item interactions.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_embedding_dim: int = 32,
        mlp_embedding_dim: int = 32,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        """Initialize NeuMF.

        Args:
            n_users: Number of users.
            n_items: Number of items.
            gmf_embedding_dim: GMF embedding dimension.
            mlp_embedding_dim: MLP embedding dimension.
            mlp_layers: MLP hidden layer sizes.
            dropout: Dropout rate.
        """
        super().__init__()

        mlp_layers = mlp_layers or [64, 32, 16]

        self.gmf = GMF(n_users, n_items, gmf_embedding_dim)
        self.mlp = MLP(n_users, n_items, mlp_embedding_dim, mlp_layers, dropout)

        # Final prediction layer
        final_input_size = gmf_embedding_dim + mlp_layers[-1]
        self.output_layer = nn.Linear(final_input_size, 1)

        nn.init.kaiming_normal_(self.output_layer.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            user_ids: User ID tensor.
            item_ids: Item ID tensor.

        Returns:
            Prediction scores.
        """
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)

        # Concatenate GMF and MLP outputs
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.output_layer(concat)

        return torch.sigmoid(output).squeeze(-1)


class NCFDataset(Dataset):
    """PyTorch Dataset for NCF training with negative sampling."""

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        labels: np.ndarray,
    ):
        """Initialize dataset.

        Args:
            user_ids: Array of user indices.
            item_ids: Array of item indices.
            labels: Array of labels (1 for positive, 0 for negative).
        """
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


class NCFRecommender(BaseRecommender):
    """Neural Collaborative Filtering recommender.

    Implements GMF, MLP, and NeuMF (combined) models.
    """

    name = "ncf"

    def __init__(
        self,
        model_type: Literal["gmf", "mlp", "neumf"] = "neumf",
        embedding_dim: int = 32,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 20,
        negative_samples: int = 4,
        device: str | None = None,
        random_state: int = 42,
    ):
        """Initialize NCF recommender.

        Args:
            model_type: Type of model (gmf, mlp, neumf).
            embedding_dim: Embedding dimension for all components.
            mlp_layers: MLP hidden layer sizes.
            dropout: Dropout rate.
            learning_rate: Learning rate for Adam optimizer.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            negative_samples: Number of negative samples per positive.
            device: Device to use (cuda/cpu). Auto-detects if None.
            random_state: Random seed.
        """
        super().__init__()

        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers or [64, 32, 16]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.random_state = random_state

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model and mappings
        self._model: nn.Module | None = None
        self._user_to_idx: dict[int, int] = {}
        self._idx_to_user: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._user_items: dict[int, set[int]] = defaultdict(set)
        self._all_items: set[int] = set()

    def fit(self, interactions: pd.DataFrame) -> "NCFRecommender":
        """Fit NCF model on interaction data.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name} ({self.model_type}) model on {len(interactions):,} interactions")

        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build user and item mappings
        unique_users = interactions["visitor_id"].unique()
        unique_items = interactions["item_id"].unique()

        self._user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self._idx_to_user = {idx: user for user, idx in self._user_to_idx.items()}
        self._item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self._idx_to_item = {idx: item for item, idx in self._item_to_idx.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)
        self._all_items = set(range(n_items))

        logger.info(f"Users: {n_users:,}, Items: {n_items:,}")
        logger.info(f"Device: {self.device}")

        # Track user-item interactions
        for _, row in interactions.iterrows():
            user_idx = self._user_to_idx[row["visitor_id"]]
            item_idx = self._item_to_idx[row["item_id"]]
            self._user_items[user_idx].add(item_idx)

        # Create model
        self._model = self._create_model(n_users, n_items)
        self._model.to(self.device)

        # Create training data with negative sampling
        train_dataset = self._create_training_data(interactions)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Train model
        self._train_model(train_loader)

        self._is_fitted = True
        return self

    def _create_model(self, n_users: int, n_items: int) -> nn.Module:
        """Create the neural network model.

        Args:
            n_users: Number of users.
            n_items: Number of items.

        Returns:
            PyTorch model.
        """
        if self.model_type == "gmf":
            model = nn.Sequential(
                GMF(n_users, n_items, self.embedding_dim),
                nn.Linear(self.embedding_dim, 1),
                nn.Sigmoid(),
            )
            # Need custom forward for GMF
            class GMFWrapper(nn.Module):
                def __init__(self, gmf, fc):
                    super().__init__()
                    self.gmf = gmf
                    self.fc = fc

                def forward(self, user_ids, item_ids):
                    gmf_out = self.gmf(user_ids, item_ids)
                    return torch.sigmoid(self.fc(gmf_out)).squeeze(-1)

            gmf = GMF(n_users, n_items, self.embedding_dim)
            fc = nn.Linear(self.embedding_dim, 1)
            return GMFWrapper(gmf, fc)

        elif self.model_type == "mlp":
            class MLPWrapper(nn.Module):
                def __init__(self, mlp, fc):
                    super().__init__()
                    self.mlp = mlp
                    self.fc = fc

                def forward(self, user_ids, item_ids):
                    mlp_out = self.mlp(user_ids, item_ids)
                    return torch.sigmoid(self.fc(mlp_out)).squeeze(-1)

            mlp = MLP(n_users, n_items, self.embedding_dim, self.mlp_layers, self.dropout)
            fc = nn.Linear(self.mlp_layers[-1], 1)
            return MLPWrapper(mlp, fc)

        else:  # neumf
            return NeuMF(
                n_users,
                n_items,
                gmf_embedding_dim=self.embedding_dim,
                mlp_embedding_dim=self.embedding_dim,
                mlp_layers=self.mlp_layers,
                dropout=self.dropout,
            )

    def _create_training_data(self, interactions: pd.DataFrame) -> NCFDataset:
        """Create training dataset with negative sampling.

        Args:
            interactions: Interaction DataFrame.

        Returns:
            NCFDataset for training.
        """
        # Get positive samples
        positive_pairs = interactions[["visitor_id", "item_id"]].drop_duplicates()

        user_ids = []
        item_ids = []
        labels = []

        # Add positive samples
        for _, row in positive_pairs.iterrows():
            user_idx = self._user_to_idx[row["visitor_id"]]
            item_idx = self._item_to_idx[row["item_id"]]

            user_ids.append(user_idx)
            item_ids.append(item_idx)
            labels.append(1.0)

            # Add negative samples
            user_items = self._user_items[user_idx]
            negative_items = list(self._all_items - user_items)

            if negative_items:
                n_neg = min(self.negative_samples, len(negative_items))
                neg_samples = np.random.choice(negative_items, size=n_neg, replace=False)

                for neg_item in neg_samples:
                    user_ids.append(user_idx)
                    item_ids.append(neg_item)
                    labels.append(0.0)

        return NCFDataset(
            np.array(user_ids),
            np.array(item_ids),
            np.array(labels),
        )

    def _train_model(self, train_loader: DataLoader) -> None:
        """Train the NCF model.

        Args:
            train_loader: Training data loader.
        """
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        self._model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for user_ids, item_ids, labels in progress:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                predictions = self._model(user_ids, item_ids)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / n_batches
            logger.info(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

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
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        if user_id not in self._user_to_idx:
            logger.debug(f"Cold-start user {user_id}, returning empty recommendations")
            return []

        user_idx = self._user_to_idx[user_id]

        # Get items to score
        if exclude_seen:
            candidate_items = list(self._all_items - self._user_items[user_idx])
        else:
            candidate_items = list(self._all_items)

        if not candidate_items:
            return []

        # Score all candidate items
        self._model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx] * len(candidate_items)).to(self.device)
            item_tensor = torch.LongTensor(candidate_items).to(self.device)

            scores = self._model(user_tensor, item_tensor).cpu().numpy()

        # Get top-n items
        top_indices = np.argsort(scores)[::-1][:n_items]

        recommendations = [
            (self._idx_to_item[candidate_items[idx]], float(scores[idx]))
            for idx in top_indices
        ]

        return recommendations

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "model_type": self.model_type,
            "embedding_dim": self.embedding_dim,
            "mlp_layers": self.mlp_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "negative_samples": self.negative_samples,
            "device": self.device,
        }

    @property
    def n_users(self) -> int:
        """Number of users in the model."""
        return len(self._user_to_idx)

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return len(self._item_to_idx)
