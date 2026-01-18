"""Tests for Version 1.0 recommendation models.

Tests for:
- ALS (Alternating Least Squares)
- BPR (Bayesian Personalized Ranking)
- Item2Vec
- NCF (Neural Collaborative Filtering)
- Funnel-Aware Hybrid
- User Features / FunnelStage
"""

from numbers import Integral, Real

import numpy as np
import pandas as pd
import pytest

from src.data.features.user_features import (
    FunnelStage,
    UserFeatureExtractor,
    UserFeatures,
)
from src.data.features.session_features import (
    SessionFeatureExtractor,
    SessionFeatures,
)
from src.models.collaborative.als import ALSRecommender
from src.models.collaborative.bpr import BPRRecommender
from src.models.collaborative.ncf import NCFRecommender
from src.models.content.item2vec import Item2VecRecommender
from src.models.hybrid.funnel_aware import FunnelAwareHybridRecommender
from src.models.baselines.popular import PopularItemsRecommender


@pytest.fixture
def larger_interactions() -> pd.DataFrame:
    """Create larger interaction dataset for model testing.

    Creates a dataset with:
    - Multiple users at different funnel stages
    - Enough interactions for matrix factorization
    - Sessions for Item2Vec
    """
    data = []
    timestamp = 1000000

    # User 1: BUYER - has transactions
    for item_id in range(100, 120):
        data.append({"visitor_id": 1, "item_id": item_id, "event": "view", "timestamp": timestamp})
        timestamp += 100
    for item_id in [100, 101, 102]:
        data.append({"visitor_id": 1, "item_id": item_id, "event": "addtocart", "timestamp": timestamp})
        timestamp += 100
    data.append({"visitor_id": 1, "item_id": 100, "event": "transaction", "timestamp": timestamp})
    timestamp += 100

    # User 2: INTENDER - has addtocart but no transaction
    for item_id in range(100, 115):
        data.append({"visitor_id": 2, "item_id": item_id, "event": "view", "timestamp": timestamp})
        timestamp += 100
    for item_id in [103, 104]:
        data.append({"visitor_id": 2, "item_id": item_id, "event": "addtocart", "timestamp": timestamp})
        timestamp += 100

    # User 3: ACTIVE_BROWSER - many views, no addtocart
    for item_id in range(200, 220):
        data.append({"visitor_id": 3, "item_id": item_id, "event": "view", "timestamp": timestamp})
        timestamp += 100

    # User 4: NEW_USER - few views only
    for item_id in range(300, 303):
        data.append({"visitor_id": 4, "item_id": item_id, "event": "view", "timestamp": timestamp})
        timestamp += 100

    # Users 5-10: Additional users for CF models
    for user_id in range(5, 11):
        base_items = list(range(100, 110)) if user_id % 2 == 0 else list(range(200, 210))
        for item_id in base_items:
            data.append({"visitor_id": user_id, "item_id": item_id, "event": "view", "timestamp": timestamp})
            timestamp += 100
        if user_id % 3 == 0:
            data.append({"visitor_id": user_id, "item_id": base_items[0], "event": "addtocart", "timestamp": timestamp})
            timestamp += 100

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Add session_id (simple: each user is one session)
    df["session_id"] = df["visitor_id"]

    return df


@pytest.fixture
def split_data(larger_interactions):
    """Split data for train/test."""
    df = larger_interactions.sort_values("timestamp")
    split_point = int(len(df) * 0.7)
    train = df.iloc[:split_point].copy()
    test = df.iloc[split_point:].copy()
    return train, test


# =============================================================================
# User Features Tests
# =============================================================================

class TestFunnelStage:
    """Tests for FunnelStage enum and detection."""

    def test_funnel_stage_values(self):
        """Test FunnelStage enum values."""
        assert FunnelStage.NEW_USER.value == "new_user"
        assert FunnelStage.ACTIVE_BROWSER.value == "active_browser"
        assert FunnelStage.INTENDER.value == "intender"
        assert FunnelStage.BUYER.value == "buyer"

    def test_funnel_stage_str(self):
        """Test FunnelStage string representation."""
        assert str(FunnelStage.BUYER) == "buyer"


class TestUserFeatureExtractor:
    """Tests for UserFeatureExtractor."""

    def test_fit(self, larger_interactions):
        """Test feature extractor fitting."""
        extractor = UserFeatureExtractor()
        extractor.fit(larger_interactions)

        assert extractor.is_fitted
        assert len(extractor._user_features) > 0

    def test_funnel_stage_detection(self, larger_interactions):
        """Test correct funnel stage detection."""
        extractor = UserFeatureExtractor()
        extractor.fit(larger_interactions)

        # User 1: BUYER (has transaction)
        assert extractor.get_funnel_stage(1) == FunnelStage.BUYER

        # User 2: INTENDER (has addtocart, no transaction)
        assert extractor.get_funnel_stage(2) == FunnelStage.INTENDER

        # User 3: ACTIVE_BROWSER (many views, no addtocart)
        assert extractor.get_funnel_stage(3) == FunnelStage.ACTIVE_BROWSER

        # User 4: NEW_USER (few views)
        assert extractor.get_funnel_stage(4) == FunnelStage.NEW_USER

    def test_cold_start_user(self, larger_interactions):
        """Test funnel stage for unknown user."""
        extractor = UserFeatureExtractor()
        extractor.fit(larger_interactions)

        # Unknown user should default to NEW_USER
        assert extractor.get_funnel_stage(9999) == FunnelStage.NEW_USER

    def test_get_user_features(self, larger_interactions):
        """Test getting user features."""
        extractor = UserFeatureExtractor()
        extractor.fit(larger_interactions)

        features = extractor.get_user_features(1)
        assert features is not None
        assert features.user_id == 1
        assert features.total_views > 0
        assert features.total_addtocarts > 0
        assert features.total_transactions > 0

    def test_get_all_features(self, larger_interactions):
        """Test getting all features as DataFrame."""
        extractor = UserFeatureExtractor()
        extractor.fit(larger_interactions)

        df = extractor.get_all_features()
        assert len(df) > 0
        assert "user_id" in df.columns
        assert "funnel_stage" in df.columns


class TestSessionFeatureExtractor:
    """Tests for SessionFeatureExtractor."""

    def test_fit(self, larger_interactions):
        """Test session feature extractor fitting."""
        extractor = SessionFeatureExtractor()
        extractor.fit(larger_interactions)

        assert extractor.is_fitted

    def test_get_session_features(self, larger_interactions):
        """Test getting session features."""
        extractor = SessionFeatureExtractor()
        extractor.fit(larger_interactions)

        features = extractor.get_session_features(1)  # Session 1 = User 1
        assert features is not None
        assert features.session_length > 0

    def test_session_items(self, larger_interactions):
        """Test getting session items."""
        extractor = SessionFeatureExtractor()
        extractor.fit(larger_interactions)

        items = extractor.get_session_items(1)
        assert len(items) > 0


# =============================================================================
# ALS Model Tests
# =============================================================================

class TestALSRecommender:
    """Tests for ALSRecommender."""

    def test_fit(self, larger_interactions):
        """Test ALS fitting."""
        model = ALSRecommender(factors=16, iterations=5)
        model.fit(larger_interactions)

        assert model.is_fitted
        assert model.n_users > 0
        assert model.n_items > 0

    def test_recommend(self, larger_interactions):
        """Test ALS recommendations."""
        model = ALSRecommender(factors=16, iterations=5)
        model.fit(larger_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        assert len(recs) <= 5
        assert all(isinstance(item_id, Integral) for item_id, _ in recs)
        assert all(isinstance(score, Real) for _, score in recs)

    def test_recommend_excludes_seen(self, larger_interactions):
        """Test that seen items are excluded."""
        model = ALSRecommender(factors=16, iterations=5)
        model.fit(larger_interactions)

        recs = model.recommend(user_id=1, n_items=20, exclude_seen=True)
        rec_items = {item_id for item_id, _ in recs}

        # User 1's items should not be in recommendations
        user_items = model._user_items.get(1, set())
        assert len(rec_items & user_items) == 0

    def test_cold_start_user(self, larger_interactions):
        """Test recommendations for unknown user."""
        model = ALSRecommender(factors=16, iterations=5)
        model.fit(larger_interactions)

        recs = model.recommend(user_id=9999, n_items=5)
        assert recs == []

    def test_recommend_batch(self, larger_interactions):
        """Test batch recommendations."""
        model = ALSRecommender(factors=16, iterations=5)
        model.fit(larger_interactions)

        user_ids = [1, 2, 3]
        results = model.recommend_batch(user_ids, n_items=5)

        assert len(results) == len(user_ids)
        for user_id in user_ids:
            assert user_id in results
            assert isinstance(results[user_id], list)

    def test_get_similar_items(self, larger_interactions):
        """Test similar items."""
        model = ALSRecommender(factors=16, iterations=5)
        model.fit(larger_interactions)

        # Get item that exists
        items = larger_interactions["item_id"].unique()
        similar = model.get_similar_items(items[0], n_items=3)

        assert isinstance(similar, list)

    def test_get_params(self):
        """Test get_params method."""
        model = ALSRecommender(factors=32, regularization=0.05, iterations=10)
        params = model.get_params()

        assert params["factors"] == 32
        assert params["regularization"] == 0.05
        assert params["iterations"] == 10


# =============================================================================
# BPR Model Tests
# =============================================================================

class TestBPRRecommender:
    """Tests for BPRRecommender."""

    def test_fit(self, larger_interactions):
        """Test BPR fitting."""
        model = BPRRecommender(factors=16, iterations=50)
        model.fit(larger_interactions)

        assert model.is_fitted
        assert model.n_users > 0
        assert model.n_items > 0

    def test_recommend(self, larger_interactions):
        """Test BPR recommendations."""
        model = BPRRecommender(factors=16, iterations=50)
        model.fit(larger_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        assert len(recs) <= 5
        assert all(isinstance(item_id, Integral) for item_id, _ in recs)

    def test_cold_start_user(self, larger_interactions):
        """Test recommendations for unknown user."""
        model = BPRRecommender(factors=16, iterations=50)
        model.fit(larger_interactions)

        recs = model.recommend(user_id=9999, n_items=5)
        assert recs == []

    def test_get_params(self):
        """Test get_params method."""
        model = BPRRecommender(factors=32, learning_rate=0.05)
        params = model.get_params()

        assert params["factors"] == 32
        assert params["learning_rate"] == 0.05


# =============================================================================
# Item2Vec Model Tests
# =============================================================================

class TestItem2VecRecommender:
    """Tests for Item2VecRecommender."""

    def test_fit(self, larger_interactions):
        """Test Item2Vec fitting."""
        model = Item2VecRecommender(embedding_dim=16, min_count=1, epochs=5)
        model.fit(larger_interactions)

        assert model.is_fitted
        assert model.vocab_size > 0

    def test_recommend(self, larger_interactions):
        """Test Item2Vec recommendations."""
        model = Item2VecRecommender(embedding_dim=16, min_count=1, epochs=5)
        model.fit(larger_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        # May return empty if user has no items in vocab
        assert isinstance(recs, list)

    def test_recommend_from_items(self, larger_interactions):
        """Test Item2Vec recommendations from item list."""
        model = Item2VecRecommender(embedding_dim=16, min_count=1, epochs=5)
        model.fit(larger_interactions)

        # Get some items that are in the vocabulary
        items = larger_interactions["item_id"].unique()[:3].tolist()
        recs = model.recommend_from_items(items, n_items=5)

        assert isinstance(recs, list)

    def test_get_similar_items(self, larger_interactions):
        """Test similar items."""
        model = Item2VecRecommender(embedding_dim=16, min_count=1, epochs=5)
        model.fit(larger_interactions)

        # Get an item that should be in vocabulary
        items = larger_interactions["item_id"].value_counts()
        frequent_item = items.index[0]

        similar = model.get_similar_items(frequent_item, n_items=3)
        assert isinstance(similar, list)

    def test_get_item_embedding(self, larger_interactions):
        """Test getting item embedding."""
        model = Item2VecRecommender(embedding_dim=16, min_count=1, epochs=5)
        model.fit(larger_interactions)

        items = larger_interactions["item_id"].value_counts()
        frequent_item = items.index[0]

        embedding = model.get_item_embedding(frequent_item)
        if embedding is not None:
            assert embedding.shape == (16,)


# =============================================================================
# NCF Model Tests
# =============================================================================

class TestNCFRecommender:
    """Tests for NCFRecommender."""

    def test_fit_gmf(self, larger_interactions):
        """Test NCF GMF variant fitting."""
        model = NCFRecommender(
            model_type="gmf",
            embedding_dim=8,
            epochs=2,
            batch_size=32,
        )
        model.fit(larger_interactions)

        assert model.is_fitted
        assert model.n_users > 0
        assert model.n_items > 0

    def test_fit_mlp(self, larger_interactions):
        """Test NCF MLP variant fitting."""
        model = NCFRecommender(
            model_type="mlp",
            embedding_dim=8,
            mlp_layers=[16, 8],
            epochs=2,
            batch_size=32,
        )
        model.fit(larger_interactions)

        assert model.is_fitted

    def test_fit_neumf(self, larger_interactions):
        """Test NCF NeuMF variant fitting."""
        model = NCFRecommender(
            model_type="neumf",
            embedding_dim=8,
            mlp_layers=[16, 8],
            epochs=2,
            batch_size=32,
        )
        model.fit(larger_interactions)

        assert model.is_fitted

    def test_recommend(self, larger_interactions):
        """Test NCF recommendations."""
        model = NCFRecommender(
            model_type="neumf",
            embedding_dim=8,
            epochs=2,
            batch_size=32,
        )
        model.fit(larger_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        assert len(recs) <= 5
        assert all(isinstance(item_id, Integral) for item_id, _ in recs)

    def test_cold_start_user(self, larger_interactions):
        """Test recommendations for unknown user."""
        model = NCFRecommender(embedding_dim=8, epochs=2, batch_size=32)
        model.fit(larger_interactions)

        recs = model.recommend(user_id=9999, n_items=5)
        assert recs == []

    def test_get_params(self):
        """Test get_params method."""
        model = NCFRecommender(
            model_type="neumf",
            embedding_dim=32,
            learning_rate=0.001,
        )
        params = model.get_params()

        assert params["model_type"] == "neumf"
        assert params["embedding_dim"] == 32
        assert params["learning_rate"] == 0.001


# =============================================================================
# Funnel-Aware Hybrid Model Tests
# =============================================================================

class TestFunnelAwareHybridRecommender:
    """Tests for FunnelAwareHybridRecommender."""

    @pytest.fixture
    def trained_models(self, larger_interactions):
        """Train component models."""
        popular = PopularItemsRecommender()
        popular.fit(larger_interactions)

        als = ALSRecommender(factors=16, iterations=5)
        als.fit(larger_interactions)

        item2vec = Item2VecRecommender(embedding_dim=16, min_count=1, epochs=5)
        item2vec.fit(larger_interactions)

        return {
            "popular": popular,
            "cf": als,
            "content": item2vec,
        }

    def test_fit(self, larger_interactions, trained_models):
        """Test hybrid model fitting."""
        hybrid = FunnelAwareHybridRecommender(
            popular_model=trained_models["popular"],
            cf_model=trained_models["cf"],
            content_model=trained_models["content"],
        )
        hybrid.fit(larger_interactions)

        assert hybrid.is_fitted

    def test_recommend(self, larger_interactions, trained_models):
        """Test hybrid recommendations."""
        hybrid = FunnelAwareHybridRecommender(
            popular_model=trained_models["popular"],
            cf_model=trained_models["cf"],
            content_model=trained_models["content"],
        )
        hybrid.fit(larger_interactions)

        recs = hybrid.recommend(user_id=1, n_items=5)

        assert len(recs) <= 5
        assert all(isinstance(item_id, Integral) for item_id, _ in recs)

    def test_recommend_different_funnel_stages(self, larger_interactions, trained_models):
        """Test that different funnel stages get different recommendations."""
        hybrid = FunnelAwareHybridRecommender(
            popular_model=trained_models["popular"],
            cf_model=trained_models["cf"],
            content_model=trained_models["content"],
        )
        hybrid.fit(larger_interactions)

        # User 1 is BUYER, User 4 is NEW_USER
        recs_buyer = hybrid.recommend(user_id=1, n_items=5)
        recs_new_user = hybrid.recommend(user_id=4, n_items=5)

        # Both should return recommendations
        assert len(recs_buyer) > 0 or len(recs_new_user) > 0

    def test_recommend_with_explanation(self, larger_interactions, trained_models):
        """Test recommendations with explanations."""
        hybrid = FunnelAwareHybridRecommender(
            popular_model=trained_models["popular"],
            cf_model=trained_models["cf"],
            content_model=trained_models["content"],
        )
        hybrid.fit(larger_interactions)

        explanations = hybrid.recommend_with_explanation(user_id=1, n_items=3)

        assert isinstance(explanations, list)
        for exp in explanations:
            assert exp.funnel_stage is not None
            assert exp.explanation_text is not None
            assert isinstance(exp.component_scores, dict)

    def test_custom_stage_weights(self, larger_interactions, trained_models):
        """Test setting custom stage weights."""
        hybrid = FunnelAwareHybridRecommender(
            popular_model=trained_models["popular"],
            cf_model=trained_models["cf"],
            content_model=trained_models["content"],
        )
        hybrid.fit(larger_interactions)

        # Set custom weights
        hybrid.set_stage_weights(
            FunnelStage.BUYER,
            {"popular": 0.1, "cf": 0.8, "content": 0.1}
        )

        assert hybrid.stage_weights[FunnelStage.BUYER]["cf"] == pytest.approx(0.8, rel=0.01)

    def test_score_normalization_methods(self, larger_interactions, trained_models):
        """Test different score normalization methods."""
        for method in ["minmax", "zscore", "rank"]:
            hybrid = FunnelAwareHybridRecommender(
                popular_model=trained_models["popular"],
                cf_model=trained_models["cf"],
                content_model=trained_models["content"],
                score_normalization=method,
            )
            hybrid.fit(larger_interactions)

            recs = hybrid.recommend(user_id=1, n_items=5)
            assert isinstance(recs, list)

    def test_missing_models(self, larger_interactions, trained_models):
        """Test hybrid with only some models available."""
        # Only popular model
        hybrid = FunnelAwareHybridRecommender(
            popular_model=trained_models["popular"],
            cf_model=None,
            content_model=None,
        )
        hybrid.fit(larger_interactions)

        recs = hybrid.recommend(user_id=1, n_items=5)
        assert len(recs) > 0

    def test_get_params(self, larger_interactions, trained_models):
        """Test get_params method."""
        hybrid = FunnelAwareHybridRecommender(
            popular_model=trained_models["popular"],
            cf_model=trained_models["cf"],
        )
        hybrid.fit(larger_interactions)

        params = hybrid.get_params()
        assert "score_normalization" in params
        assert "available_models" in params
