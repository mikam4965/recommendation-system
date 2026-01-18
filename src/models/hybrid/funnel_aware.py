"""Funnel-Aware Hybrid Recommender - Scientific Novelty #1.

Dynamically adjusts model weights based on user's position in the conversion funnel.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.data.features.user_features import FunnelStage, UserFeatureExtractor
from src.models.base import BaseRecommender


@dataclass
class RecommendationExplanation:
    """Explanation for a recommendation.

    Attributes:
        item_id: Recommended item ID.
        score: Final combined score.
        funnel_stage: User's funnel stage.
        component_scores: Scores from each component model.
        component_weights: Weights applied to each component.
        explanation_text: Human-readable explanation.
    """

    item_id: int
    score: float
    funnel_stage: FunnelStage
    component_scores: dict[str, float]
    component_weights: dict[str, float]
    explanation_text: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "score": self.score,
            "funnel_stage": str(self.funnel_stage),
            "component_scores": self.component_scores,
            "component_weights": self.component_weights,
            "explanation_text": self.explanation_text,
        }


class FunnelAwareHybridRecommender(BaseRecommender):
    """Hybrid recommender with dynamic weights based on funnel stage.

    Key innovation: Adjusts the contribution of different recommendation
    strategies based on where the user is in the conversion funnel.

    Funnel Stage Weights:
    - NEW_USER: Heavy on popular items (cold-start), some content-based
    - ACTIVE_BROWSER: Balanced CF and content, reduced popular
    - INTENDER: Strong CF, session-based recommendations important
    - BUYER: Primarily CF, some session-based for cross-sell

    This approach addresses the observation that users at different stages
    respond differently to various recommendation strategies.
    """

    name = "funnel_aware_hybrid"

    # Default weights for each funnel stage
    DEFAULT_STAGE_WEIGHTS = {
        FunnelStage.NEW_USER: {
            "popular": 0.7,
            "content": 0.3,
            "cf": 0.0,
            "session": 0.0,
        },
        FunnelStage.ACTIVE_BROWSER: {
            "popular": 0.2,
            "content": 0.3,
            "cf": 0.5,
            "session": 0.0,
        },
        FunnelStage.INTENDER: {
            "popular": 0.1,
            "content": 0.0,
            "cf": 0.6,
            "session": 0.3,
        },
        FunnelStage.BUYER: {
            "popular": 0.1,
            "content": 0.1,
            "cf": 0.7,
            "session": 0.1,
        },
    }

    def __init__(
        self,
        popular_model: BaseRecommender | None = None,
        content_model: BaseRecommender | None = None,
        cf_model: BaseRecommender | None = None,
        session_model: BaseRecommender | None = None,
        stage_weights: dict[FunnelStage, dict[str, float]] | None = None,
        score_normalization: Literal["minmax", "zscore", "rank"] = "minmax",
        user_feature_extractor: UserFeatureExtractor | None = None,
    ):
        """Initialize Funnel-Aware Hybrid recommender.

        Args:
            popular_model: Popularity-based recommender.
            content_model: Content-based recommender (e.g., Item2Vec).
            cf_model: Collaborative filtering model (e.g., ALS, BPR, NCF).
            session_model: Session-based recommender (optional for v1.0).
            stage_weights: Custom weights per funnel stage.
            score_normalization: Method for normalizing scores.
            user_feature_extractor: Extractor for determining funnel stage.
        """
        super().__init__()

        self.popular_model = popular_model
        self.content_model = content_model
        self.cf_model = cf_model
        self.session_model = session_model

        self.stage_weights = stage_weights or self.DEFAULT_STAGE_WEIGHTS.copy()
        self.score_normalization = score_normalization
        self.user_feature_extractor = user_feature_extractor

        # Track which models are available
        self._models: dict[str, BaseRecommender] = {}

    def fit(self, interactions: pd.DataFrame) -> "FunnelAwareHybridRecommender":
        """Fit the hybrid model.

        Note: Component models should be pre-fitted before passing to this class.
        This method only fits the user feature extractor for funnel stage detection.

        Args:
            interactions: DataFrame with user interactions.

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting {self.name} model")

        # Build model registry
        self._models = {}
        if self.popular_model and self.popular_model.is_fitted:
            self._models["popular"] = self.popular_model
        if self.content_model and self.content_model.is_fitted:
            self._models["content"] = self.content_model
        if self.cf_model and self.cf_model.is_fitted:
            self._models["cf"] = self.cf_model
        if self.session_model and self.session_model.is_fitted:
            self._models["session"] = self.session_model

        logger.info(f"Available models: {list(self._models.keys())}")

        if not self._models:
            raise ValueError("At least one fitted component model is required")

        # Fit or validate user feature extractor
        if self.user_feature_extractor is None:
            self.user_feature_extractor = UserFeatureExtractor()
            self.user_feature_extractor.fit(interactions)
        elif not self.user_feature_extractor.is_fitted:
            self.user_feature_extractor.fit(interactions)

        self._is_fitted = True
        logger.info("Funnel-Aware Hybrid model ready")

        return self

    def recommend(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
        session_items: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Get recommendations for a user.

        Args:
            user_id: User ID to get recommendations for.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude seen items.
            session_items: Current session items (for session-based component).

        Returns:
            List of (item_id, score) tuples.
        """
        self._check_fitted()

        # Determine user's funnel stage
        funnel_stage = self.user_feature_extractor.get_funnel_stage(user_id)

        # Get weights for this stage
        weights = self._get_adjusted_weights(funnel_stage)

        # Get recommendations from each model with non-zero weight
        all_scores: dict[int, dict[str, float]] = {}

        for model_name, weight in weights.items():
            if weight <= 0 or model_name not in self._models:
                continue

            model = self._models[model_name]

            # Get recommendations (more than needed for diversity)
            recs = model.recommend(
                user_id, n_items=n_items * 3, exclude_seen=exclude_seen
            )

            # Store scores
            for item_id, score in recs:
                if item_id not in all_scores:
                    all_scores[item_id] = {}
                all_scores[item_id][model_name] = score

        if not all_scores:
            logger.debug(f"No recommendations for user {user_id}")
            return []

        # Normalize scores per model
        normalized_scores = self._normalize_scores(all_scores)

        # Combine scores using weights
        final_scores: dict[int, float] = {}
        for item_id, model_scores in normalized_scores.items():
            combined_score = 0.0
            for model_name, score in model_scores.items():
                combined_score += score * weights.get(model_name, 0.0)
            final_scores[item_id] = combined_score

        # Sort and return top-n
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_items]

    def recommend_with_explanation(
        self,
        user_id: int,
        n_items: int = 10,
        exclude_seen: bool = True,
        session_items: list[int] | None = None,
    ) -> list[RecommendationExplanation]:
        """Get recommendations with explanations.

        Args:
            user_id: User ID.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude seen items.
            session_items: Current session items.

        Returns:
            List of RecommendationExplanation objects.
        """
        self._check_fitted()

        # Determine user's funnel stage
        funnel_stage = self.user_feature_extractor.get_funnel_stage(user_id)
        weights = self._get_adjusted_weights(funnel_stage)

        # Get recommendations from each model
        all_scores: dict[int, dict[str, float]] = {}

        for model_name, weight in weights.items():
            if weight <= 0 or model_name not in self._models:
                continue

            model = self._models[model_name]
            recs = model.recommend(user_id, n_items=n_items * 3, exclude_seen=exclude_seen)

            for item_id, score in recs:
                if item_id not in all_scores:
                    all_scores[item_id] = {}
                all_scores[item_id][model_name] = score

        if not all_scores:
            return []

        # Normalize and combine
        normalized_scores = self._normalize_scores(all_scores)

        # Build explanations
        explanations = []
        for item_id, model_scores in normalized_scores.items():
            combined_score = 0.0
            for model_name, score in model_scores.items():
                combined_score += score * weights.get(model_name, 0.0)

            # Generate explanation text
            explanation_text = self._generate_explanation(
                funnel_stage, model_scores, weights
            )

            explanations.append(
                RecommendationExplanation(
                    item_id=item_id,
                    score=combined_score,
                    funnel_stage=funnel_stage,
                    component_scores=model_scores,
                    component_weights=weights,
                    explanation_text=explanation_text,
                )
            )

        # Sort by score
        explanations.sort(key=lambda x: x.score, reverse=True)

        return explanations[:n_items]

    def _get_adjusted_weights(self, funnel_stage: FunnelStage) -> dict[str, float]:
        """Get weights adjusted for available models.

        Redistributes weight from unavailable models to available ones.

        Args:
            funnel_stage: User's funnel stage.

        Returns:
            Adjusted weights dict.
        """
        base_weights = self.stage_weights.get(funnel_stage, self.stage_weights[FunnelStage.NEW_USER])

        # Filter to available models
        available_weights = {
            name: weight
            for name, weight in base_weights.items()
            if name in self._models and weight > 0
        }

        if not available_weights:
            # Fallback: use any available model equally
            n_models = len(self._models)
            return {name: 1.0 / n_models for name in self._models}

        # Normalize weights to sum to 1
        total = sum(available_weights.values())
        return {name: weight / total for name, weight in available_weights.items()}

    def _normalize_scores(
        self, all_scores: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        """Normalize scores across models.

        Args:
            all_scores: Dict of item_id -> {model_name: score}.

        Returns:
            Normalized scores.
        """
        if self.score_normalization == "rank":
            return self._normalize_by_rank(all_scores)
        elif self.score_normalization == "zscore":
            return self._normalize_by_zscore(all_scores)
        else:  # minmax
            return self._normalize_by_minmax(all_scores)

    def _normalize_by_minmax(
        self, all_scores: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        """Min-max normalization per model."""
        # Collect all scores per model
        model_scores: dict[str, list[tuple[int, float]]] = {}
        for item_id, scores in all_scores.items():
            for model_name, score in scores.items():
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append((item_id, score))

        # Compute min/max per model
        model_stats: dict[str, tuple[float, float]] = {}
        for model_name, scores in model_scores.items():
            score_values = [s for _, s in scores]
            min_s = min(score_values) if score_values else 0
            max_s = max(score_values) if score_values else 1
            model_stats[model_name] = (min_s, max_s)

        # Normalize
        normalized: dict[int, dict[str, float]] = {}
        for item_id, scores in all_scores.items():
            normalized[item_id] = {}
            for model_name, score in scores.items():
                min_s, max_s = model_stats[model_name]
                if max_s - min_s > 0:
                    normalized[item_id][model_name] = (score - min_s) / (max_s - min_s)
                else:
                    normalized[item_id][model_name] = 0.5

        return normalized

    def _normalize_by_zscore(
        self, all_scores: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        """Z-score normalization per model."""
        # Collect scores per model
        model_scores: dict[str, list[float]] = {}
        for scores in all_scores.values():
            for model_name, score in scores.items():
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(score)

        # Compute mean/std per model
        model_stats: dict[str, tuple[float, float]] = {}
        for model_name, scores in model_scores.items():
            mean = np.mean(scores)
            std = np.std(scores)
            model_stats[model_name] = (mean, std if std > 0 else 1.0)

        # Normalize
        normalized: dict[int, dict[str, float]] = {}
        for item_id, scores in all_scores.items():
            normalized[item_id] = {}
            for model_name, score in scores.items():
                mean, std = model_stats[model_name]
                z_score = (score - mean) / std
                # Map z-score to [0, 1] using sigmoid-like transformation
                normalized[item_id][model_name] = 1 / (1 + np.exp(-z_score))

        return normalized

    def _normalize_by_rank(
        self, all_scores: dict[int, dict[str, float]]
    ) -> dict[int, dict[str, float]]:
        """Rank-based normalization per model."""
        # Collect and rank scores per model
        model_ranks: dict[str, dict[int, float]] = {}

        # Group by model
        model_items: dict[str, list[tuple[int, float]]] = {}
        for item_id, scores in all_scores.items():
            for model_name, score in scores.items():
                if model_name not in model_items:
                    model_items[model_name] = []
                model_items[model_name].append((item_id, score))

        # Compute ranks
        for model_name, items in model_items.items():
            sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
            n_items = len(sorted_items)
            model_ranks[model_name] = {}
            for rank, (item_id, _) in enumerate(sorted_items):
                # Convert rank to score: higher rank = higher score
                model_ranks[model_name][item_id] = 1.0 - (rank / max(1, n_items - 1))

        # Build normalized dict
        normalized: dict[int, dict[str, float]] = {}
        for item_id, scores in all_scores.items():
            normalized[item_id] = {}
            for model_name in scores:
                normalized[item_id][model_name] = model_ranks[model_name][item_id]

        return normalized

    def _generate_explanation(
        self,
        funnel_stage: FunnelStage,
        model_scores: dict[str, float],
        weights: dict[str, float],
    ) -> str:
        """Generate human-readable explanation.

        Args:
            funnel_stage: User's funnel stage.
            model_scores: Scores from each model.
            weights: Weights applied.

        Returns:
            Explanation text.
        """
        # Find dominant model
        contributions = {
            name: score * weights.get(name, 0)
            for name, score in model_scores.items()
        }
        dominant = max(contributions.items(), key=lambda x: x[1])[0]

        # Stage-specific explanation
        stage_explanations = {
            FunnelStage.NEW_USER: "As a new user, we're showing you our most popular items.",
            FunnelStage.ACTIVE_BROWSER: "Based on your browsing history and similar users.",
            FunnelStage.INTENDER: "Since you've shown purchase intent, these items match your interests.",
            FunnelStage.BUYER: "Based on your purchase history and preferences.",
        }

        base_explanation = stage_explanations.get(funnel_stage, "Recommended for you.")

        # Add model-specific detail
        model_details = {
            "popular": "This is a trending item.",
            "content": "Similar to items you've viewed.",
            "cf": "Users like you also liked this.",
            "session": "Matches your current browsing session.",
        }

        detail = model_details.get(dominant, "")

        return f"{base_explanation} {detail}".strip()

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "score_normalization": self.score_normalization,
            "available_models": list(self._models.keys()),
            "stage_weights": {
                str(stage): weights
                for stage, weights in self.stage_weights.items()
            },
        }

    def set_stage_weights(
        self, stage: FunnelStage, weights: dict[str, float]
    ) -> None:
        """Update weights for a specific funnel stage.

        Args:
            stage: FunnelStage to update.
            weights: New weights dict.
        """
        # Validate weights sum to 1 (approximately)
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            # Normalize
            weights = {k: v / total for k, v in weights.items()}

        self.stage_weights[stage] = weights
        logger.info(f"Updated weights for {stage}: {weights}")
