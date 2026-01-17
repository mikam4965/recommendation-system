"""Tests for evaluation metrics."""

import pytest
import numpy as np

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate,
    mrr,
    ndcg_at_k,
    average_precision,
    coverage,
    novelty,
)


class TestPrecisionAtK:
    """Tests for precision@k metric."""

    def test_precision_at_k_basic(self, recommendations_list, relevant_items):
        """Test basic precision calculation."""
        # recommendations_list = [101, 102, 103, 104, 105, ...]
        # relevant_items = {102, 105, 111, 115}
        # At k=5: [101, 102, 103, 104, 105] contains 2 relevant items (102, 105)
        precision = precision_at_k(recommendations_list, relevant_items, k=5)
        assert precision == 2 / 5

    def test_precision_at_k_10(self, recommendations_list, relevant_items):
        """Test precision@10."""
        precision = precision_at_k(recommendations_list, relevant_items, k=10)
        # Only 102 and 105 are in recommendations
        assert precision == 2 / 10

    def test_precision_at_k_empty_recommendations(self, relevant_items):
        """Test precision with empty recommendations."""
        precision = precision_at_k([], relevant_items, k=5)
        assert precision == 0.0

    def test_precision_at_k_empty_relevant(self, recommendations_list):
        """Test precision with no relevant items."""
        precision = precision_at_k(recommendations_list, set(), k=5)
        assert precision == 0.0

    def test_precision_at_k_all_relevant(self):
        """Test precision when all recommended are relevant."""
        recs = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3, 4, 5, 6, 7}
        precision = precision_at_k(recs, relevant, k=5)
        assert precision == 1.0

    def test_precision_at_k_zero(self):
        """Test precision with k=0."""
        precision = precision_at_k([1, 2, 3], {1, 2}, k=0)
        assert precision == 0.0


class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_recall_at_k_basic(self, recommendations_list, relevant_items):
        """Test basic recall calculation."""
        # 2 relevant items found out of 4 total relevant
        recall = recall_at_k(recommendations_list, relevant_items, k=10)
        assert recall == 2 / 4

    def test_recall_at_k_empty_relevant(self, recommendations_list):
        """Test recall with no relevant items."""
        recall = recall_at_k(recommendations_list, set(), k=5)
        assert recall == 0.0

    def test_recall_at_k_all_found(self):
        """Test recall when all relevant items found."""
        recs = [1, 2, 3, 4, 5]
        relevant = {1, 3}
        recall = recall_at_k(recs, relevant, k=5)
        assert recall == 1.0


class TestHitRate:
    """Tests for hit rate metric."""

    def test_hit_rate_with_hit(self, recommendations_list, relevant_items):
        """Test hit rate when there's a hit."""
        hr = hit_rate(recommendations_list, relevant_items)
        assert hr == 1.0

    def test_hit_rate_no_hit(self):
        """Test hit rate when there's no hit."""
        recs = [1, 2, 3]
        relevant = {10, 11, 12}
        hr = hit_rate(recs, relevant)
        assert hr == 0.0

    def test_hit_rate_empty(self):
        """Test hit rate with empty inputs."""
        assert hit_rate([], {1, 2}) == 0.0
        assert hit_rate([1, 2], set()) == 0.0


class TestMRR:
    """Tests for Mean Reciprocal Rank metric."""

    def test_mrr_first_position(self):
        """Test MRR when relevant item is first."""
        recs = [1, 2, 3, 4, 5]
        relevant = {1}
        assert mrr(recs, relevant) == 1.0

    def test_mrr_second_position(self):
        """Test MRR when relevant item is second."""
        recs = [1, 2, 3, 4, 5]
        relevant = {2}
        assert mrr(recs, relevant) == 1 / 2

    def test_mrr_fifth_position(self):
        """Test MRR when relevant item is fifth."""
        recs = [1, 2, 3, 4, 5]
        relevant = {5}
        assert mrr(recs, relevant) == 1 / 5

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant items found."""
        recs = [1, 2, 3]
        relevant = {10}
        assert mrr(recs, relevant) == 0.0

    def test_mrr_multiple_relevant(self):
        """Test MRR returns reciprocal of first relevant."""
        recs = [1, 2, 3, 4, 5]
        relevant = {3, 5}  # First relevant is at position 3
        assert mrr(recs, relevant) == 1 / 3


class TestNDCGAtK:
    """Tests for NDCG@k metric."""

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        recs = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}  # All relevant items at top
        ndcg = ndcg_at_k(recs, relevant, k=5)
        assert ndcg == 1.0

    def test_ndcg_worst_ranking(self):
        """Test NDCG with no relevant items."""
        recs = [1, 2, 3, 4, 5]
        relevant = {10, 11, 12}
        ndcg = ndcg_at_k(recs, relevant, k=5)
        assert ndcg == 0.0

    def test_ndcg_partial_ranking(self):
        """Test NDCG with partial relevance."""
        recs = [1, 2, 3, 4, 5]
        relevant = {2, 4}
        ndcg = ndcg_at_k(recs, relevant, k=5)
        # DCG = 1/log2(3) + 1/log2(5)
        # IDCG = 1/log2(2) + 1/log2(3)
        expected_dcg = 1 / np.log2(3) + 1 / np.log2(5)
        expected_idcg = 1 / np.log2(2) + 1 / np.log2(3)
        expected_ndcg = expected_dcg / expected_idcg
        assert abs(ndcg - expected_ndcg) < 1e-6

    def test_ndcg_empty_relevant(self):
        """Test NDCG with no relevant items."""
        ndcg = ndcg_at_k([1, 2, 3], set(), k=3)
        assert ndcg == 0.0


class TestAveragePrecision:
    """Tests for Average Precision metric."""

    def test_ap_perfect(self):
        """Test AP with perfect ranking."""
        recs = [1, 2, 3, 4, 5]
        relevant = {1, 2, 3}
        ap = average_precision(recs, relevant)
        # P@1=1/1, P@2=2/2, P@3=3/3 -> AP = (1+1+1)/3 = 1.0
        assert ap == 1.0

    def test_ap_partial(self):
        """Test AP with partial relevance."""
        recs = [1, 2, 3, 4, 5]
        relevant = {2, 4}
        ap = average_precision(recs, relevant)
        # P@2=1/2 (rel), P@4=2/4 (rel) -> AP = (0.5 + 0.5) / 2 = 0.5
        assert abs(ap - 0.5) < 1e-6

    def test_ap_no_relevant(self):
        """Test AP with no relevant items."""
        ap = average_precision([1, 2, 3], {10})
        assert ap == 0.0


class TestCoverage:
    """Tests for coverage metric."""

    def test_coverage_full(self):
        """Test coverage when all items covered."""
        all_recs = [[1, 2], [3, 4], [5]]
        cov = coverage(all_recs, catalog_size=5)
        assert cov == 1.0

    def test_coverage_partial(self):
        """Test coverage with partial catalog."""
        all_recs = [[1, 2], [1, 3]]
        cov = coverage(all_recs, catalog_size=10)
        assert cov == 0.3  # 3 unique items / 10

    def test_coverage_empty(self):
        """Test coverage with empty recommendations."""
        cov = coverage([], catalog_size=10)
        assert cov == 0.0


class TestNovelty:
    """Tests for novelty metric."""

    def test_novelty_popular_items(self):
        """Test novelty with popular items (should be low)."""
        recs = [1, 2]
        popularity = {1: 100, 2: 100}
        total = 200
        nov = novelty(recs, popularity, total)
        # Popular items have low novelty
        assert nov < 2.0

    def test_novelty_rare_items(self):
        """Test novelty with rare items (should be high)."""
        recs = [1, 2]
        popularity = {1: 1, 2: 1}
        total = 10000
        nov = novelty(recs, popularity, total)
        # Rare items have high novelty
        assert nov > 10.0

    def test_novelty_empty(self):
        """Test novelty with empty recommendations."""
        nov = novelty([], {1: 10}, 100)
        assert nov == 0.0
