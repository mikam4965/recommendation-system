"""Hybrid recommendation models."""

from src.models.hybrid.funnel_aware import (
    FunnelAwareHybridRecommender,
    RecommendationExplanation,
)

__all__ = [
    "FunnelAwareHybridRecommender",
    "RecommendationExplanation",
]
