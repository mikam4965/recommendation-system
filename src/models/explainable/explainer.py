"""Explainable AI module for recommendation explanations.

Provides human-readable explanations for why items are recommended.
Supports rule-based explanations, LIME, and attention-based explanations.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger

from src.data.features.user_features import FunnelStage, UserFeatureExtractor


class ExplanationType(Enum):
    """Types of explanations."""

    COLLABORATIVE = "collaborative"  # Similar users liked this
    CONTENT = "content"  # Similar to items you liked
    CATEGORY = "category"  # From your favorite category
    POPULARITY = "popularity"  # Popular item
    TRENDING = "trending"  # Trending recently
    INTENT = "intent"  # Based on cart/purchase intent
    SESSION = "session"  # Based on current session
    PERSONALIZED = "personalized"  # Personalized for you


@dataclass
class ExplanationReason:
    """Single reason in an explanation."""

    type: ExplanationType
    text: str
    weight: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "text": self.text,
            "weight": self.weight,
            "evidence": self.evidence,
        }


@dataclass
class Explanation:
    """Complete explanation for a recommendation."""

    item_id: int
    score: float
    reasons: list[ExplanationReason] = field(default_factory=list)
    confidence: float = 0.0
    model_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "score": self.score,
            "reasons": [r.to_dict() for r in self.reasons],
            "confidence": self.confidence,
            "model_name": self.model_name,
            "summary": self.get_summary(),
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the explanation."""
        if not self.reasons:
            return "Рекомендация на основе анализа ваших предпочтений"

        # Get top reason
        top_reason = max(self.reasons, key=lambda r: r.weight)
        return top_reason.text

    def get_full_explanation(self) -> str:
        """Get full explanation with all reasons."""
        if not self.reasons:
            return "Рекомендация на основе анализа ваших предпочтений"

        sorted_reasons = sorted(self.reasons, key=lambda r: r.weight, reverse=True)
        lines = [f"• {r.text}" for r in sorted_reasons]
        return "\n".join(lines)


class RecommendationExplainer:
    """Generate explanations for recommendations.

    Combines multiple explanation strategies:
    - Collaborative: based on similar users
    - Content-based: based on similar items
    - Category: based on user's category preferences
    - Intent: based on funnel stage
    - Session: based on current session behavior
    """

    def __init__(
        self,
        user_feature_extractor: UserFeatureExtractor | None = None,
        item_categories: dict[int, int] | None = None,
        item_popularity: dict[int, int] | None = None,
        category_names: dict[int, str] | None = None,
    ):
        """Initialize explainer.

        Args:
            user_feature_extractor: Fitted UserFeatureExtractor for funnel stages.
            item_categories: Mapping of item_id to category_id.
            item_popularity: Mapping of item_id to popularity score (interaction count).
            category_names: Mapping of category_id to human-readable name.
        """
        self.user_feature_extractor = user_feature_extractor
        self.item_categories = item_categories or {}
        self.item_popularity = item_popularity or {}
        self.category_names = category_names or {}

        # Will be populated during fit
        self._user_items: dict[int, set[int]] = defaultdict(set)
        self._user_categories: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._item_users: dict[int, set[int]] = defaultdict(set)
        self._user_cart_items: dict[int, set[int]] = defaultdict(set)
        self._user_purchased_items: dict[int, set[int]] = defaultdict(set)
        self._popular_items: list[int] = []
        self._is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> "RecommendationExplainer":
        """Fit the explainer on interaction data.

        Args:
            interactions: DataFrame with columns:
                - visitor_id: User ID
                - item_id: Item ID
                - event: Event type (view, addtocart, transaction)

        Returns:
            Self for method chaining.
        """
        logger.info(f"Fitting RecommendationExplainer on {len(interactions):,} interactions")

        # Reset state
        self._user_items = defaultdict(set)
        self._user_categories = defaultdict(lambda: defaultdict(int))
        self._item_users = defaultdict(set)
        self._user_cart_items = defaultdict(set)
        self._user_purchased_items = defaultdict(set)

        # Build indices
        for _, row in interactions.iterrows():
            user_id = row["visitor_id"]
            item_id = row["item_id"]
            event = row["event"]

            self._user_items[user_id].add(item_id)
            self._item_users[item_id].add(user_id)

            # Track categories
            if item_id in self.item_categories:
                cat_id = self.item_categories[item_id]
                self._user_categories[user_id][cat_id] += 1

            # Track cart and purchase items
            if event == "addtocart":
                self._user_cart_items[user_id].add(item_id)
            elif event == "transaction":
                self._user_purchased_items[user_id].add(item_id)

        # Build popularity ranking
        if not self.item_popularity:
            item_counts = interactions["item_id"].value_counts().to_dict()
            self.item_popularity = item_counts

        sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self._popular_items = [item_id for item_id, _ in sorted_items]

        self._is_fitted = True
        logger.info(f"Explainer fitted with {len(self._user_items):,} users, "
                   f"{len(self._item_users):,} items")

        return self

    def explain(
        self,
        user_id: int,
        item_id: int,
        score: float,
        model_name: str = "",
        session_items: list[int] | None = None,
    ) -> Explanation:
        """Generate explanation for a single recommendation.

        Args:
            user_id: User ID.
            item_id: Recommended item ID.
            score: Recommendation score.
            model_name: Name of the recommender model.
            session_items: Current session items (optional).

        Returns:
            Explanation object.
        """
        reasons = []

        # Check collaborative signal
        collab_reason = self._explain_collaborative(user_id, item_id)
        if collab_reason:
            reasons.append(collab_reason)

        # Check category match
        category_reason = self._explain_category(user_id, item_id)
        if category_reason:
            reasons.append(category_reason)

        # Check intent/funnel stage
        intent_reason = self._explain_intent(user_id, item_id)
        if intent_reason:
            reasons.append(intent_reason)

        # Check popularity
        popularity_reason = self._explain_popularity(item_id)
        if popularity_reason:
            reasons.append(popularity_reason)

        # Check session context
        if session_items:
            session_reason = self._explain_session(session_items, item_id)
            if session_reason:
                reasons.append(session_reason)

        # If no specific reasons, add generic personalized reason
        if not reasons:
            reasons.append(ExplanationReason(
                type=ExplanationType.PERSONALIZED,
                text="Подобрано специально для вас на основе ваших предпочтений",
                weight=0.5,
            ))

        # Normalize weights
        total_weight = sum(r.weight for r in reasons)
        if total_weight > 0:
            for reason in reasons:
                reason.weight /= total_weight

        # Calculate confidence
        confidence = min(1.0, len(reasons) * 0.25 + max(r.weight for r in reasons))

        return Explanation(
            item_id=item_id,
            score=score,
            reasons=reasons,
            confidence=confidence,
            model_name=model_name,
        )

    def explain_batch(
        self,
        user_id: int,
        recommendations: list[tuple[int, float]],
        model_name: str = "",
        session_items: list[int] | None = None,
    ) -> list[Explanation]:
        """Generate explanations for a batch of recommendations.

        Args:
            user_id: User ID.
            recommendations: List of (item_id, score) tuples.
            model_name: Name of the recommender model.
            session_items: Current session items (optional).

        Returns:
            List of Explanation objects.
        """
        return [
            self.explain(
                user_id=user_id,
                item_id=item_id,
                score=score,
                model_name=model_name,
                session_items=session_items,
            )
            for item_id, score in recommendations
        ]

    def _explain_collaborative(
        self,
        user_id: int,
        item_id: int,
    ) -> ExplanationReason | None:
        """Explain based on collaborative filtering signal.

        Checks if similar users (users who liked same items) also liked this item.
        """
        if user_id not in self._user_items:
            return None

        user_items = self._user_items[user_id]
        if not user_items:
            return None

        # Find users who interacted with same items
        similar_users = set()
        for other_item in user_items:
            similar_users.update(self._item_users.get(other_item, set()))

        similar_users.discard(user_id)

        if not similar_users:
            return None

        # Check how many similar users also interacted with recommended item
        item_users = self._item_users.get(item_id, set())
        common_users = similar_users & item_users

        if len(common_users) < 3:  # Need at least 3 similar users
            return None

        ratio = len(common_users) / len(similar_users)

        return ExplanationReason(
            type=ExplanationType.COLLABORATIVE,
            text=f"Пользователи с похожими интересами также выбрали этот товар ({len(common_users)} чел.)",
            weight=0.4 + ratio * 0.2,
            evidence={
                "similar_users_count": len(common_users),
                "total_similar_users": len(similar_users),
                "ratio": ratio,
            },
        )

    def _explain_category(
        self,
        user_id: int,
        item_id: int,
    ) -> ExplanationReason | None:
        """Explain based on category preferences."""
        if item_id not in self.item_categories:
            return None

        if user_id not in self._user_categories:
            return None

        item_category = self.item_categories[item_id]
        user_cat_counts = self._user_categories[user_id]

        if item_category not in user_cat_counts:
            return None

        # Calculate category preference
        total_interactions = sum(user_cat_counts.values())
        category_ratio = user_cat_counts[item_category] / total_interactions

        if category_ratio < 0.1:  # Not a significant preference
            return None

        category_name = self.category_names.get(item_category, f"категория {item_category}")

        return ExplanationReason(
            type=ExplanationType.CATEGORY,
            text=f"Вы часто интересуетесь товарами из категории «{category_name}»",
            weight=0.3 + category_ratio * 0.2,
            evidence={
                "category_id": item_category,
                "category_name": category_name,
                "preference_ratio": category_ratio,
            },
        )

    def _explain_intent(
        self,
        user_id: int,
        item_id: int,
    ) -> ExplanationReason | None:
        """Explain based on user's funnel stage and intent."""
        if self.user_feature_extractor is None:
            return None

        funnel_stage = self.user_feature_extractor.get_funnel_stage(user_id)

        # Check cart similarity
        cart_items = self._user_cart_items.get(user_id, set())
        purchased_items = self._user_purchased_items.get(user_id, set())

        if item_id in self.item_categories and cart_items:
            item_category = self.item_categories[item_id]

            # Check if similar to cart items
            cart_categories = {
                self.item_categories.get(i)
                for i in cart_items
                if i in self.item_categories
            }

            if item_category in cart_categories:
                return ExplanationReason(
                    type=ExplanationType.INTENT,
                    text="Похож на товары в вашей корзине",
                    weight=0.5,
                    evidence={
                        "funnel_stage": str(funnel_stage),
                        "cart_items": len(cart_items),
                    },
                )

        # Check if similar to purchased items
        if item_id in self.item_categories and purchased_items:
            item_category = self.item_categories[item_id]

            purchased_categories = {
                self.item_categories.get(i)
                for i in purchased_items
                if i in self.item_categories
            }

            if item_category in purchased_categories:
                return ExplanationReason(
                    type=ExplanationType.INTENT,
                    text="Дополнит ваши предыдущие покупки",
                    weight=0.4,
                    evidence={
                        "funnel_stage": str(funnel_stage),
                        "purchased_items": len(purchased_items),
                    },
                )

        # Generic funnel-based explanation
        if funnel_stage == FunnelStage.INTENDER:
            return ExplanationReason(
                type=ExplanationType.INTENT,
                text="Рекомендуем на основе вашего интереса к покупкам",
                weight=0.3,
                evidence={"funnel_stage": str(funnel_stage)},
            )

        return None

    def _explain_popularity(self, item_id: int) -> ExplanationReason | None:
        """Explain based on item popularity."""
        if item_id not in self.item_popularity:
            return None

        if not self._popular_items:
            return None

        # Check if item is in top 10%
        popularity_rank = self._popular_items.index(item_id) if item_id in self._popular_items else -1

        if popularity_rank < 0:
            return None

        top_threshold = len(self._popular_items) * 0.1

        if popularity_rank < top_threshold:
            return ExplanationReason(
                type=ExplanationType.POPULARITY,
                text="Популярный товар среди покупателей",
                weight=0.2,
                evidence={
                    "rank": popularity_rank + 1,
                    "popularity_score": self.item_popularity[item_id],
                },
            )

        return None

    def _explain_session(
        self,
        session_items: list[int],
        item_id: int,
    ) -> ExplanationReason | None:
        """Explain based on current session context."""
        if not session_items:
            return None

        if item_id not in self.item_categories:
            return None

        item_category = self.item_categories[item_id]

        # Check if session has items from same category
        session_categories = [
            self.item_categories.get(i)
            for i in session_items
            if i in self.item_categories
        ]

        if item_category in session_categories:
            return ExplanationReason(
                type=ExplanationType.SESSION,
                text="Связан с товарами, которые вы просматриваете сейчас",
                weight=0.4,
                evidence={
                    "session_length": len(session_items),
                    "matching_category": item_category,
                },
            )

        return None

    @property
    def is_fitted(self) -> bool:
        """Check if explainer has been fitted."""
        return self._is_fitted


class LIMEExplainer:
    """LIME-based model-agnostic explainer.

    Uses Local Interpretable Model-agnostic Explanations to explain
    individual recommendations.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        n_samples: int = 1000,
    ):
        """Initialize LIME explainer.

        Args:
            feature_names: Names of input features.
            n_samples: Number of samples for LIME explanation.
        """
        self.feature_names = feature_names or []
        self.n_samples = n_samples

        # Optional LIME import
        try:
            from lime.lime_tabular import LimeTabularExplainer
            self._lime_available = True
        except ImportError:
            self._lime_available = False
            logger.warning("LIME not installed. Install with: pip install lime")

    def explain_prediction(
        self,
        model_predict: Callable[[np.ndarray], np.ndarray],
        user_features: np.ndarray,
        training_data: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Explain a model prediction using LIME.

        Args:
            model_predict: Function that takes features and returns predictions.
            user_features: Features for the user to explain.
            training_data: Training data for LIME (optional).

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self._lime_available:
            logger.warning("LIME not available")
            return {}

        from lime.lime_tabular import LimeTabularExplainer

        if training_data is None:
            # Create synthetic training data
            training_data = np.random.randn(100, len(user_features))

        explainer = LimeTabularExplainer(
            training_data,
            feature_names=self.feature_names or [f"feature_{i}" for i in range(len(user_features))],
            mode="regression",
            random_state=42,
        )

        exp = explainer.explain_instance(
            user_features,
            model_predict,
            num_samples=self.n_samples,
        )

        return dict(exp.as_list())


class AttentionExplainer:
    """Explainer for attention-based models like SASRec.

    Extracts and interprets attention weights to explain
    which items in the sequence influenced the recommendation.
    """

    def __init__(
        self,
        item_names: dict[int, str] | None = None,
    ):
        """Initialize attention explainer.

        Args:
            item_names: Mapping of item_id to human-readable name.
        """
        self.item_names = item_names or {}

    def explain_attention(
        self,
        session_items: list[int],
        attention_weights: np.ndarray,
        recommended_item: int,
        top_k: int = 3,
    ) -> list[ExplanationReason]:
        """Explain recommendation based on attention weights.

        Args:
            session_items: Items in the session (without padding).
            attention_weights: Attention weights [n_layers, n_heads, seq_len, seq_len].
            recommended_item: The recommended item ID.
            top_k: Number of top attended items to include.

        Returns:
            List of ExplanationReason objects.
        """
        if attention_weights.size == 0:
            return []

        # Average attention across layers and heads
        # Focus on attention to the last position (prediction position)
        avg_attention = attention_weights.mean(axis=(0, 1))  # [seq_len, seq_len]

        # Get attention from last position to all previous positions
        seq_len = len(session_items)
        if seq_len > avg_attention.shape[0]:
            seq_len = avg_attention.shape[0]

        # Pad offset if needed
        offset = avg_attention.shape[0] - seq_len
        last_pos_attention = avg_attention[-1, offset:offset + seq_len]

        # Get top-k attended items
        top_indices = np.argsort(last_pos_attention)[-top_k:][::-1]

        reasons = []
        for idx in top_indices:
            if idx >= len(session_items):
                continue

            item_id = session_items[idx]
            attention_score = float(last_pos_attention[idx])

            if attention_score < 0.1:  # Skip low attention
                continue

            item_name = self.item_names.get(item_id, f"товар #{item_id}")

            reasons.append(ExplanationReason(
                type=ExplanationType.SESSION,
                text=f"Рекомендация основана на вашем интересе к «{item_name}»",
                weight=attention_score,
                evidence={
                    "source_item_id": item_id,
                    "attention_score": attention_score,
                    "position_in_session": idx,
                },
            ))

        return reasons
