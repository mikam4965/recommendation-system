"""SASRec: Self-Attentive Sequential Recommendation.

Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICDM 2018.
https://arxiv.org/abs/1808.09781
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


class PointWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        """Initialize FFN.

        Args:
            hidden_dim: Hidden dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim].

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim].
        """
        x = self.dropout(F.gelu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SASRecBlock(nn.Module):
    """Single Transformer block for SASRec."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.2,
    ):
        """Initialize SASRec block.

        Args:
            hidden_dim: Hidden dimension (must be divisible by n_heads).
            n_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)

        self.ffn = PointWiseFeedForward(hidden_dim, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim].
            attention_mask: Causal attention mask [seq_len, seq_len].

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim].
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=attention_mask,
            need_weights=False,
        )
        x = self.attention_norm(x + self.attention_dropout(attn_output))

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + self.ffn_dropout(ffn_output))

        return x


class SASRecModule(nn.Module):
    """Self-Attentive Sequential Recommendation model.

    Uses Transformer architecture for modeling user sequences.
    """

    def __init__(
        self,
        n_items: int,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2,
    ):
        """Initialize SASRec module.

        Args:
            n_items: Number of items in the catalog.
            hidden_dim: Hidden dimension (embedding size).
            n_heads: Number of attention heads.
            n_layers: Number of Transformer blocks.
            max_seq_length: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()

        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

        # Item embeddings (0 reserved for padding)
        self.item_embedding = nn.Embedding(
            n_items + 1, hidden_dim, padding_idx=0
        )

        # Positional embeddings (learnable)
        self.position_embedding = nn.Embedding(max_seq_length, hidden_dim)

        # Dropout for embeddings
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output layer (share weights with item embeddings for efficiency)
        # self.output = nn.Linear(hidden_dim, n_items, bias=False)
        # self.output.weight = self.item_embedding.weight[1:]  # Skip padding

        # Initialize weights
        self._init_weights()

        # Register causal mask buffer
        self._register_causal_mask(max_seq_length)

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.position_embedding.weight)

    def _register_causal_mask(self, size: int):
        """Register causal attention mask.

        Args:
            size: Maximum sequence length.
        """
        # Create causal mask (upper triangular = True means masked)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(
        self,
        item_sequences: torch.Tensor,
        return_all_positions: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            item_sequences: Item ID sequences [batch_size, seq_len].
                           Item IDs are 1-indexed (0 = padding).
            return_all_positions: If True, return outputs for all positions.
                                 If False, return only last position.

        Returns:
            If return_all_positions:
                Logits [batch_size, seq_len, n_items]
            Else:
                Logits [batch_size, n_items]
        """
        batch_size, seq_len = item_sequences.shape

        # Get item embeddings
        item_emb = self.item_embedding(item_sequences)  # [B, L, H]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=item_sequences.device)
        pos_emb = self.position_embedding(positions)  # [L, H]
        x = item_emb + pos_emb.unsqueeze(0)  # [B, L, H]

        x = self.embedding_dropout(x)

        # Get causal mask for current sequence length
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=causal_mask)

        x = self.layer_norm(x)  # [B, L, H]

        # Compute logits by dot product with item embeddings
        # Item embeddings: [n_items + 1, H], skip padding
        item_emb_weight = self.item_embedding.weight[1:]  # [n_items, H]

        if return_all_positions:
            # [B, L, H] @ [H, n_items] -> [B, L, n_items]
            logits = torch.matmul(x, item_emb_weight.T)
        else:
            # [B, H] @ [H, n_items] -> [B, n_items]
            last_hidden = x[:, -1, :]
            logits = torch.matmul(last_hidden, item_emb_weight.T)

        return logits

    def predict_next(self, session_items: torch.Tensor) -> torch.Tensor:
        """Predict next item scores given session history.

        Args:
            session_items: Current session item IDs [batch_size, seq_len].

        Returns:
            Item scores [batch_size, n_items].
        """
        return self.forward(session_items, return_all_positions=False)


class SASRecDataset(Dataset):
    """PyTorch Dataset for SASRec training.

    Fixed-length sequence approach: each sample is a fixed-length window.
    """

    def __init__(
        self,
        sessions: list[list[int]],
        max_length: int = 50,
    ):
        """Initialize dataset.

        Args:
            sessions: List of sessions, each is a list of item indices (1-indexed).
            max_length: Maximum sequence length.
        """
        self.max_length = max_length
        self.samples = []

        for session in sessions:
            if len(session) < 2:
                continue

            # For SASRec, we use sliding window approach
            # Input: [1, ..., n-1], Target: [2, ..., n]
            for end_idx in range(2, len(session) + 1):
                start_idx = max(0, end_idx - max_length - 1)
                seq = session[start_idx:end_idx - 1]
                target = session[end_idx - 1]
                self.samples.append((seq, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


def collate_sasrec(
    batch: list[tuple[list[int], int]],
    max_length: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for SASRec batches.

    Pads sequences to max_length from the left (right-aligned).

    Args:
        batch: List of (input_sequence, target) tuples.
        max_length: Maximum sequence length.

    Returns:
        Tuple of (padded_sequences, targets).
    """
    sequences, targets = zip(*batch)

    # Pad sequences from the left to max_length
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded.append(seq[-max_length:])
        else:
            padding = [0] * (max_length - len(seq))
            padded.append(padding + list(seq))

    return torch.LongTensor(padded), torch.LongTensor(targets)


class SASRecRecommender(BaseRecommender):
    """SASRec session-based recommender using self-attention.

    Self-Attentive Sequential Recommendation uses Transformer
    architecture for modeling user sequences.
    """

    name = "sasrec"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        max_seq_length: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        epochs: int = 10,
        l2_reg: float = 0.0,
        device: str | None = None,
        random_state: int = 42,
    ):
        """Initialize SASRec recommender.

        Args:
            hidden_dim: Hidden dimension (embedding size).
            n_heads: Number of attention heads.
            n_layers: Number of Transformer blocks.
            max_seq_length: Maximum sequence length.
            dropout: Dropout rate.
            learning_rate: Learning rate for Adam optimizer.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            l2_reg: L2 regularization weight.
            device: Device to use (cuda/cpu). Auto-detects if None.
            random_state: Random seed.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.random_state = random_state

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Model and mappings
        self._model: SASRecModule | None = None
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: dict[int, int] = {}
        self._n_items: int = 0

        # User history for recommend() fallback
        self._user_sessions: dict[int, list[int]] = defaultdict(list)

    def fit(self, interactions: pd.DataFrame) -> "SASRecRecommender":
        """Fit SASRec model on session data.

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
                "SASRec requires session_id column. "
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
        self._model = SASRecModule(
            n_items=self._n_items,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_seq_length=self.max_seq_length,
            dropout=self.dropout,
        )
        self._model.to(self.device)

        # Create dataset and dataloader
        dataset = SASRecDataset(sessions, max_length=self.max_seq_length)
        logger.info(f"Training samples: {len(dataset):,}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_sasrec(b, self.max_seq_length),
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
            items = [self._item_to_idx[item] for item in group["item_id"].tolist()]
            if len(items) >= 2:
                sessions.append(items)

        return sessions

    def _train_model(self, dataloader: DataLoader) -> None:
        """Train the SASRec model.

        Args:
            dataloader: Training data loader.
        """
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )

        self._model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for sequences, targets in progress:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self._model(sequences, return_all_positions=False)

                # Cross-entropy loss (targets are 1-indexed, convert to 0-indexed)
                loss = F.cross_entropy(logits, targets - 1)

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

        Args:
            session_items: List of item indices in current session (1-indexed).
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items already in session.

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        if not session_items:
            return []

        # Truncate and pad to max_seq_length
        session_items = session_items[-self.max_seq_length:]
        if len(session_items) < self.max_seq_length:
            padding = [0] * (self.max_seq_length - len(session_items))
            session_items = padding + session_items

        # Convert to tensor
        session_tensor = torch.LongTensor([session_items]).to(self.device)

        # Get predictions
        self._model.eval()
        with torch.no_grad():
            scores = self._model.predict_next(session_tensor)
            scores = scores.squeeze(0).cpu().numpy()  # [n_items]

        # Build exclusion set (original items without padding)
        if exclude_seen:
            exclude_set = set(s for s in session_items if s > 0)
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
            Item embeddings array [n_items, hidden_dim].
        """
        self._check_fitted()

        with torch.no_grad():
            # Skip padding embedding (index 0)
            embeddings = self._model.item_embedding.weight[1:].cpu().numpy()
        return embeddings

    def get_attention_weights(
        self,
        session_items: list[int],
    ) -> np.ndarray:
        """Get attention weights for interpretability.

        Args:
            session_items: List of item indices in session.

        Returns:
            Attention weights [n_layers, n_heads, seq_len, seq_len].
        """
        self._check_fitted()

        # Truncate and pad
        session_items = session_items[-self.max_seq_length:]
        if len(session_items) < self.max_seq_length:
            padding = [0] * (self.max_seq_length - len(session_items))
            session_items = padding + session_items

        session_tensor = torch.LongTensor([session_items]).to(self.device)

        self._model.eval()
        attention_weights = []

        # Hook to capture attention weights
        def hook(module, input, output):
            if isinstance(output, tuple):
                attention_weights.append(output[1])

        # Register hooks
        handles = []
        for block in self._model.blocks:
            h = block.attention.register_forward_hook(hook)
            handles.append(h)

        # Forward pass with attention weights
        with torch.no_grad():
            # Need to modify forward to return attention
            self._model(session_tensor)

        # Remove hooks
        for h in handles:
            h.remove()

        if attention_weights:
            return np.stack([w.cpu().numpy() for w in attention_weights])
        return np.array([])

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "l2_reg": self.l2_reg,
            "device": self.device,
        }

    @property
    def n_items(self) -> int:
        """Number of items in the model."""
        return self._n_items
