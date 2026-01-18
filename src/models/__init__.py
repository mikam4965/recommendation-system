"""Recommendation models package."""

from src.models.base import BaseRecommender

# Baselines
from src.models.baselines.popular import PopularItemsRecommender
from src.models.baselines.random_model import RandomRecommender

# Collaborative filtering
from src.models.collaborative.als import ALSRecommender
from src.models.collaborative.bpr import BPRRecommender
from src.models.collaborative.ncf import NCFRecommender

# Content-based
from src.models.content.item2vec import Item2VecRecommender

# Sequential (V2.0)
from src.models.sequential.gru4rec import GRU4RecRecommender
from src.models.sequential.sasrec import SASRecRecommender

# Retrieval (V2.0)
from src.models.retrieval.two_tower import TwoTowerRecommender

# Hybrid
from src.models.hybrid.funnel_aware import FunnelAwareHybridRecommender

# Explainable (V2.0)
from src.models.explainable.explainer import (
    RecommendationExplainer,
    Explanation,
    ExplanationType,
)

__all__ = [
    # Base
    "BaseRecommender",
    # Baselines
    "PopularItemsRecommender",
    "RandomRecommender",
    # Collaborative
    "ALSRecommender",
    "BPRRecommender",
    "NCFRecommender",
    # Content
    "Item2VecRecommender",
    # Sequential
    "GRU4RecRecommender",
    "SASRecRecommender",
    # Retrieval
    "TwoTowerRecommender",
    # Hybrid
    "FunnelAwareHybridRecommender",
    # Explainable
    "RecommendationExplainer",
    "Explanation",
    "ExplanationType",
]
