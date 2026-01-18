"""GRU4Rec: Session-based Recommendations with Recurrent Neural Networks.

Reference: Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks", ICLR 2016.
https://arxiv.org/abs/1511.06939
"""

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.base import BaseRecommender


class GRU4RecModule(nn.Module):
    """GRU-based neural network for session-based recommendations.

    Takes a sequence of item IDs and predicts the next item.
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.2,
        embedding_dropout: float = 0.25,
    ):
        """Initialize GRU4Rec module.

        Args:
            n_items: Number of items in the catalog.
            embedding_dim: Dimension of item embeddings.
            hidden_dim: Dimension of GRU hidden state.
            n_layers: Number of GRU layers.
            dropout: Dropout rate between GRU layers.
            embedding_dropout: Dropout rate for embeddings.
        """
        super().__init__()

        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Item embeddings (0 reserved for padding)
        self.item_embedding = nn.Embedding(
            n_items + 1, embedding_dim, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # GRU layer(s)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Output layer: hidden_dim -> n_items (excluding padding)
        self.output = nn.Linear(hidden_dim, n_items)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # Skip padding
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        item_sequences: torch.Tensor,
        lengths: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            item_sequences: Item ID sequences [batch_size, seq_len].
                           Item IDs are 1-indexed (0 = padding).
            lengths: Actual sequence lengths [batch_size].
            hidden: Initial hidden state [n_layers, batch_size, hidden_dim].

        Returns:
            Tuple of (logits, hidden):
                - logits: Output scores [batch_size, seq_len, n_items]
                - hidden: Final hidden state [n_layers, batch_size, hidden_dim]
        """
        batch_size = item_sequences.size(0)

        # Get embeddings
        embedded = self.item_embedding(item_sequences)  # [B, L, E]
        embedded = self.embedding_dropout(embedded)

        # Initialize hidden if not provided
        if hidden is None:
            hidden = self._init_hidden(batch_size, item_sequences.device)

        # Pack sequences if lengths provided
        if lengths is not None:
            # Pack for variable length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, hidden = self.gru(packed, hidden)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, hidden = self.gru(embedded, hidden)

        # Project to item space
        logits = self.output(output)  # [B, L, n_items]

        return logits, hidden

    def _init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state with zeros."""
        return torch.zeros(
            self.n_layers, batch_size, self.hidden_dim, device=device
        )

    def predict_next(
        self,
        session_items: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next item probabilities given session history.

        Args:
            session_items: Current session item IDs [batch_size, seq_len].
            hidden: Optional hidden state.

        Returns:
            Tuple of (scores, hidden):
                - scores: Item scores [batch_size, n_items]
                - hidden: Updated hidden state
        """
        logits, hidden = self.forward(session_items, hidden=hidden)
        # Take last position's output
        scores = logits[:, -1, :]  # [batch_size, n_items]
        return scores, hidden


class SessionDataset(Dataset):
    """PyTorch Dataset for session-based training.

    Each sample is (input_sequence, target_items, length).
    For a session [A, B, C, D], creates training pairs:
      [A] -> B
      [A, B] -> C
      [A, B, C] -> D
    """

    def __init__(
        self,
        sessions: list[list[int]],
        max_length: int = 50,
    ):
        """Initialize dataset.

        Args:
            sessions: List of sessions, each session is a list of item indices (1-indexed).
            max_length: Maximum sequence length.
        """
        self.max_length = max_length
        self.samples = []

        for session in sessions:
            if len(session) < 2:
                continue

            # Truncate if too long
            session = session[-max_length - 1:]

            # Create all subsequence -> next item pairs
            for i in range(1, len(session)):
                input_seq = session[:i]
                target = session[i]
                self.samples.append((input_seq, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


def collate_sessions(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for session batches.

    Args:
        batch: List of (input_sequence, target) tuples.

    Returns:
        Tuple of (padded_sequences, targets, lengths).
    """
    sequences, targets = zip(*batch)

    # Convert to tensors
    seq_tensors = [torch.LongTensor(seq) for seq in sequences]
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    targets = torch.LongTensor(targets)

    # Pad sequences
    padded = pad_sequence(seq_tensors, batch_first=True, padding_value=0)

    return padded, targets, lengths


class GRU4RecRecommender(BaseRecommender):
    """GRU4Rec session-based recommender.

    This model is designed for session-based recommendations where
    user history is represented by item sequences within a session.
    """

    name = "gru4rec"

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 1,
        dropout: float = 0.2,
        embedding_dropout: float = 0.25,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        epochs: int = 10,
        max_session_length: int = 50,
        loss_type: Literal["ce", "bpr", "top1"] = "ce",
        device: str | None = None,
        random_state: int = 42,
    ):
        """Initialize GRU4Rec recommender.

        Args:
            embedding_dim: Item embedding dimension.
            hidden_dim: GRU hidden state dimension.
            n_layers: Number of GRU layers.
            dropout: Dropout rate for GRU.
            embedding_dropout: Dropout rate for embeddings.
            learning_rate: Learning rate for Adam optimizer.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            max_session_length: Maximum session length.
            loss_type: Loss function type:
                - "ce": Cross-entropy (softmax)
                - "bpr": Bayesian Personalized Ranking
                - "top1": TOP1 loss from GRU4Rec paper
            device: Device to use (cuda/cpu). Auto-detects if None.
            random_state: Random seed.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_session_length = max_session_length
        self.loss_type = loss_type
        self.random_state = random_state

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model and mappings
        self._model: GRU4RecModule | None = None
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._n_items: int = 0

        # User history for recommend() fallback
        self._user_sessions: dict[int, list[int]] = defaultdict(list)

    def fit(self, interactions: pd.DataFrame) -> "GRU4RecRecommender":
        """Fit GRU4Rec model on session data.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - session_id: Session ID (required)
                - timestamp: Event timestamp

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name} model on {len(interactions):,} interactions")

        # Check required columns
        if "session_id" not in interactions.columns:
            raise ValueError(
                "GRU4Rec requires session_id column. "
                "Use SessionBuilder to add sessions first."
            )

        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build item mappings (1-indexed, 0 reserved for padding)
        unique_items = interactions["item_id"].unique()
        self._item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self._idx_to_item = {idx: item for item, idx in self._item_to_idx.items()}
        self._n_items = len(unique_items)

        logger.info(f"Items: {self._n_items:,}")
        logger.info(f"Device: {self.device}")

        # Build sessions
        sessions = self._build_sessions(interactions)
        logger.info(f"Sessions: {len(sessions):,}")

        # Store user sessions for recommend() fallback
        for _, row in interactions.iterrows():
            user_id = row["visitor_id"]
            item_idx = self._item_to_idx[row["item_id"]]
            self._user_sessions[user_id].append(item_idx)

        # Create model
        self._model = GRU4RecModule(
            n_items=self._n_items,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            embedding_dropout=self.embedding_dropout,
        )
        self._model.to(self.device)

        # Create dataset and dataloader
        dataset = SessionDataset(sessions, max_length=self.max_session_length)
        logger.info(f"Training samples: {len(dataset):,}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_sessions,
            num_workers=0,
        )

        # Train model
        self._train_model(dataloader)

        self._is_fitted = True
        return self

    def _build_sessions(self, interactions: pd.DataFrame) -> list[list[int]]:
        """Build sessions from interaction data.

        Args:
            interactions: Interaction DataFrame with session_id.

        Returns:
            List of sessions, each session is a list of item indices.
        """
        # Sort by session and timestamp
        interactions = interactions.sort_values(["session_id", "timestamp"])

        sessions = []
        for _, group in interactions.groupby("session_id"):
            # Get item sequence for this session
            items = [self._item_to_idx[item] for item in group["item_id"].tolist()]
            if len(items) >= 2:  # Need at least 2 items for training
                sessions.append(items)

        return sessions

    def _train_model(self, dataloader: DataLoader) -> None:
        """Train the GRU4Rec model.

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
            for sequences, targets, lengths in progress:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits, _ = self._model(sequences, lengths)

                # Get prediction at last position for each sequence
                batch_size = sequences.size(0)
                last_positions = lengths - 1
                batch_indices = torch.arange(batch_size, device=self.device)
                last_logits = logits[batch_indices, last_positions]  # [B, n_items]

                # Compute loss
                loss = self._compute_loss(last_logits, targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / n_batches
            logger.info(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on loss_type.

        Args:
            logits: Model output logits [batch_size, n_items].
            targets: Target item indices [batch_size] (1-indexed).

        Returns:
            Loss tensor.
        """
        # Convert targets to 0-indexed for loss computation
        targets_0idx = targets - 1

        if self.loss_type == "ce":
            return F.cross_entropy(logits, targets_0idx)

        elif self.loss_type == "bpr":
            # BPR: maximize difference between positive and negative items
            batch_size = logits.size(0)

            # Get positive scores
            pos_scores = logits[torch.arange(batch_size), targets_0idx]

            # Sample negative items
            neg_indices = torch.randint(
                0, self._n_items, (batch_size,), device=logits.device
            )
            neg_scores = logits[torch.arange(batch_size), neg_indices]

            # BPR loss: -log(sigmoid(pos - neg))
            return -F.logsigmoid(pos_scores - neg_scores).mean()

        elif self.loss_type == "top1":
            # TOP1 loss from GRU4Rec paper
            batch_size = logits.size(0)

            # Get positive scores
            pos_scores = logits[torch.arange(batch_size), targets_0idx].unsqueeze(1)

            # Compare with all items (approximation with sampling for efficiency)
            n_samples = min(100, self._n_items)
            neg_indices = torch.randint(
                0, self._n_items, (batch_size, n_samples), device=logits.device
            )
            neg_scores = torch.gather(logits, 1, neg_indices)

            # TOP1 loss: sigmoid(neg - pos) + sigmoid(neg^2)
            diff = neg_scores - pos_scores
            loss = torch.sigmoid(diff) + torch.sigmoid(neg_scores**2)
            return loss.mean()

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations for a user based on their session history.

        Args:
            user_id: User ID to get recommendations for.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items the user has seen.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        if user_id not in self._user_sessions:
            logger.debug(f"Cold-start user {user_id}, returning empty recommendations")
            return []

        session_items = self._user_sessions[user_id]
        if not session_items:
            return []

        return self.recommend_session(
            session_items=session_items,
            n_items=n_items,
            exclude_seen=exclude_seen,
        )

    def recommend_session(
        self,
        session_items: list[int],
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations based on current session items.

        This is the main recommendation method for GRU4Rec.

        Args:
            session_items: List of item indices in current session (1-indexed internal IDs)
                          or item_ids if map_items=True.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items already in session.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        if not session_items:
            return []

        # Truncate to max length
        session_items = session_items[-self.max_session_length:]

        # Convert to tensor
        session_tensor = torch.LongTensor([session_items]).to(self.device)

        # Get predictions
        self._model.eval()
        with torch.no_grad():
            scores, _ = self._model.predict_next(session_tensor)
            scores = scores.squeeze(0).cpu().numpy()  # [n_items]

        # Build exclusion set
        if exclude_seen:
            exclude_set = set(session_items)
        else:
            exclude_set = set()

        # Get top items
        sorted_indices = np.argsort(scores)[::-1]

        recommendations = []
        for idx in sorted_indices:
            item_idx = idx + 1  # Convert back to 1-indexed
            if item_idx in exclude_set:
                continue
            if item_idx not in self._idx_to_item:
                continue

            item_id = self._idx_to_item[item_idx]
            recommendations.append((item_id, float(scores[idx])))

            if len(recommendations) >= n_items:
                break

        return recommendations

    def recommend_by_item_ids(
        self,
        item_ids: list[int],
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations based on current session item IDs (external).

        Convenience method that handles item ID to index mapping.

        Args:
            item_ids: List of external item IDs in current session.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items already in session.

        Returns:
            List of (item_id, score) tuples.
        """
        # Map item IDs to indices
        session_items = []
        for item_id in item_ids:
            if item_id in self._item_to_idx:
                session_items.append(self._item_to_idx[item_id])

        if not session_items:
            return []

        return self.recommend_session(
            session_items=session_items,
            n_items=n_items,
            exclude_seen=exclude_seen,
        )

    def get_item_embeddings(self) -> np.ndarray:
        """Get learned item embeddings.

        Returns:
            Item embeddings array [n_items, embedding_dim].
        """
        self._check_fitted()

        with torch.no_grad():
            # Skip padding embedding (index 0)
            embeddings = self._model.item_embedding.weight[1:].cpu().numpy()
        return embeddings

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "embedding_dropout": self.embedding_dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "max_session_length": self.max_session_length,
            "loss_type": self.loss_type,
            "device": self.device,
        }

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return self._n_items
