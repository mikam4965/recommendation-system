"""Tests for recommendation models."""

from numbers import Integral, Real

import pytest
import tempfile
from pathlib import Path

from src.models.baselines.popular import PopularItemsRecommender
from src.models.baselines.random_model import RandomRecommender
from src.models.collaborative.user_cf import UserBasedCF
from src.models.collaborative.item_cf import ItemBasedCF


class TestPopularItemsRecommender:
    """Tests for PopularItemsRecommender."""

    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = PopularItemsRecommender()
        model.fit(sample_interactions)

        assert model.is_fitted
        assert len(model.item_scores) > 0
        assert len(model.sorted_items) > 0

    def test_recommend(self, sample_interactions):
        """Test recommendations."""
        model = PopularItemsRecommender()
        model.fit(sample_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        assert len(recs) <= 5
        assert all(isinstance(item_id, Integral) for item_id, _ in recs)
        assert all(isinstance(score, Real) for _, score in recs)
        # Scores should be descending
        scores = [score for _, score in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_excludes_seen(self, sample_interactions):
        """Test that seen items are excluded."""
        model = PopularItemsRecommender()
        model.fit(sample_interactions)

        recs = model.recommend(user_id=1, n_items=20, exclude_seen=True)
        rec_items = {item_id for item_id, _ in recs}

        # User 1's items should not be in recommendations
        user_items = model.user_items.get(1, set())
        assert len(rec_items & user_items) == 0

    def test_recommend_not_fitted_raises(self):
        """Test that recommend before fit raises error."""
        model = PopularItemsRecommender()
        with pytest.raises(RuntimeError):
            model.recommend(user_id=1)

    def test_save_load(self, sample_interactions, tmp_path):
        """Test model save and load."""
        model = PopularItemsRecommender()
        model.fit(sample_interactions)

        # Save
        model_path = tmp_path / "popular.joblib"
        model.save(model_path)
        assert model_path.exists()

        # Load
        loaded = PopularItemsRecommender.load(model_path)
        assert loaded.is_fitted
        assert loaded.item_scores == model.item_scores

    def test_weights(self, sample_interactions):
        """Test that event weights are applied correctly."""
        model = PopularItemsRecommender(
            weight_view=1.0, weight_addtocart=5.0, weight_transaction=10.0
        )
        model.fit(sample_interactions)

        # Item 100 has view, addtocart, and transaction - should have high score
        assert model.item_scores[100] > model.item_scores.get(102, 0)


class TestRandomRecommender:
    """Tests for RandomRecommender."""

    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = RandomRecommender(seed=42)
        model.fit(sample_interactions)

        assert model.is_fitted
        assert len(model.all_items) > 0

    def test_recommend(self, sample_interactions):
        """Test recommendations."""
        model = RandomRecommender(seed=42)
        model.fit(sample_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        assert len(recs) <= 5
        assert all(isinstance(item_id, Integral) for item_id, _ in recs)

    def test_recommend_reproducible(self, sample_interactions):
        """Test that same seed gives same results."""
        model1 = RandomRecommender(seed=42)
        model1.fit(sample_interactions)

        model2 = RandomRecommender(seed=42)
        model2.fit(sample_interactions)

        # Reset RNG for fair comparison
        model1.rng = __import__("numpy").random.default_rng(42)
        model2.rng = __import__("numpy").random.default_rng(42)

        recs1 = model1.recommend(user_id=999, n_items=5, exclude_seen=False)
        recs2 = model2.recommend(user_id=999, n_items=5, exclude_seen=False)

        items1 = [item for item, _ in recs1]
        items2 = [item for item, _ in recs2]
        assert items1 == items2


class TestUserBasedCF:
    """Tests for UserBasedCF."""

    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = UserBasedCF(n_neighbors=2)
        model.fit(sample_interactions)

        assert model.is_fitted
        assert model.user_item_matrix is not None
        assert model.user_similarity is not None

    def test_recommend(self, sample_interactions):
        """Test recommendations."""
        model = UserBasedCF(n_neighbors=2)
        model.fit(sample_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        # Should return some recommendations
        assert isinstance(recs, list)
        for item in recs:
            assert len(item) == 2
            assert isinstance(item[0], Integral)
            assert isinstance(item[1], Real)

    def test_recommend_cold_start_user(self, sample_interactions):
        """Test recommendations for unknown user."""
        model = UserBasedCF(n_neighbors=2)
        model.fit(sample_interactions)

        # User 999 doesn't exist
        recs = model.recommend(user_id=999, n_items=5)
        assert recs == []

    def test_params(self):
        """Test get_params method."""
        model = UserBasedCF(n_neighbors=100, min_similarity=0.1)
        params = model.get_params()

        assert params["n_neighbors"] == 100
        assert params["min_similarity"] == 0.1


class TestItemBasedCF:
    """Tests for ItemBasedCF."""

    def test_fit(self, sample_interactions):
        """Test model fitting."""
        model = ItemBasedCF(n_similar=2)
        model.fit(sample_interactions)

        assert model.is_fitted
        assert model.item_user_matrix is not None
        assert model.item_similarity is not None

    def test_recommend(self, sample_interactions):
        """Test recommendations."""
        model = ItemBasedCF(n_similar=2)
        model.fit(sample_interactions)

        recs = model.recommend(user_id=1, n_items=5)

        assert isinstance(recs, list)
        for item in recs:
            assert len(item) == 2

    def test_recommend_cold_start_user(self, sample_interactions):
        """Test recommendations for unknown user."""
        model = ItemBasedCF(n_similar=2)
        model.fit(sample_interactions)

        recs = model.recommend(user_id=999, n_items=5)
        assert recs == []

    def test_get_similar_items(self, sample_interactions):
        """Test similar items functionality."""
        model = ItemBasedCF(n_similar=5)
        model.fit(sample_interactions)

        # Item 100 is viewed by multiple users
        similar = model.get_similar_items(item_id=100, n_items=3)

        assert isinstance(similar, list)
        # Should find some similar items
        for item_id, score in similar:
            assert item_id != 100  # Should not include self
            assert score >= 0
