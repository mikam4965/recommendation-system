"""Recommendation evaluation metrics."""

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Precision@K.

    Precision@K = |recommended[:k] ∩ relevant| / k

    Args:
        recommended: List of recommended item IDs (ordered by score).
        relevant: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.

    Returns:
        Precision score in [0, 1].
    """
    if k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    if not recommended_k:
        return 0.0

    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Recall@K.

    Recall@K = |recommended[:k] ∩ relevant| / |relevant|

    Args:
        recommended: List of recommended item IDs (ordered by score).
        relevant: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.

    Returns:
        Recall score in [0, 1].
    """
    if not relevant or k <= 0:
        return 0.0

    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in relevant)
    return hits / len(relevant)


def hit_rate(recommended: list[int], relevant: set[int]) -> float:
    """Calculate Hit Rate (binary).

    Hit Rate = 1 if any recommended item is in relevant, else 0.

    Args:
        recommended: List of recommended item IDs.
        relevant: Set of relevant (ground truth) item IDs.

    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    if not recommended or not relevant:
        return 0.0

    for item in recommended:
        if item in relevant:
            return 1.0
    return 0.0


def mrr(recommended: list[int], relevant: set[int]) -> float:
    """Calculate Mean Reciprocal Rank.

    MRR = 1 / rank of first relevant item (0 if no relevant item found).

    Args:
        recommended: List of recommended item IDs (ordered by score).
        relevant: Set of relevant (ground truth) item IDs.

    Returns:
        Reciprocal rank in [0, 1].
    """
    if not recommended or not relevant:
        return 0.0

    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    Uses binary relevance (1 if in relevant set, 0 otherwise).

    Args:
        recommended: List of recommended item IDs (ordered by score).
        relevant: Set of relevant (ground truth) item IDs.
        k: Number of top recommendations to consider.

    Returns:
        NDCG score in [0, 1].
    """
    if k <= 0 or not relevant:
        return 0.0

    recommended_k = recommended[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            # Using log2(i + 2) because i is 0-indexed
            dcg += 1.0 / np.log2(i + 2)

    # Calculate ideal DCG (all relevant items at top)
    n_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(recommended: list[int], relevant: set[int]) -> float:
    """Calculate Average Precision.

    AP = sum(P@k * rel(k)) / |relevant| for k=1..N

    Args:
        recommended: List of recommended item IDs (ordered by score).
        relevant: Set of relevant (ground truth) item IDs.

    Returns:
        Average precision score in [0, 1].
    """
    if not recommended or not relevant:
        return 0.0

    score = 0.0
    num_hits = 0

    for i, item in enumerate(recommended):
        if item in relevant:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / len(relevant)


def coverage(
    all_recommendations: list[list[int]],
    catalog_size: int,
) -> float:
    """Calculate catalog coverage.

    Coverage = |unique recommended items| / |catalog|

    Args:
        all_recommendations: List of recommendation lists for all users.
        catalog_size: Total number of items in catalog.

    Returns:
        Coverage ratio in [0, 1].
    """
    if catalog_size <= 0:
        return 0.0

    unique_items = set()
    for recs in all_recommendations:
        unique_items.update(recs)

    return len(unique_items) / catalog_size


def diversity(recommended: list[int], item_similarity: dict[tuple[int, int], float]) -> float:
    """Calculate intra-list diversity.

    Diversity = 1 - average pairwise similarity

    Args:
        recommended: List of recommended item IDs.
        item_similarity: Dictionary mapping (item1, item2) to similarity score.

    Returns:
        Diversity score in [0, 1].
    """
    if len(recommended) < 2:
        return 0.0

    total_sim = 0.0
    count = 0

    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            item1, item2 = recommended[i], recommended[j]
            # Get similarity (order-independent)
            sim = item_similarity.get((item1, item2), 0.0)
            if sim == 0.0:
                sim = item_similarity.get((item2, item1), 0.0)
            total_sim += sim
            count += 1

    if count == 0:
        return 0.0

    avg_sim = total_sim / count
    return 1.0 - avg_sim


def novelty(
    recommended: list[int],
    item_popularity: dict[int, int],
    total_interactions: int,
) -> float:
    """Calculate novelty (inverse popularity).

    Novelty = average(-log2(popularity)) for recommended items

    Args:
        recommended: List of recommended item IDs.
        item_popularity: Dictionary mapping item_id to interaction count.
        total_interactions: Total number of interactions in training set.

    Returns:
        Novelty score (higher = more novel).
    """
    if not recommended or total_interactions <= 0:
        return 0.0

    novelty_scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1) / total_interactions
        # Avoid log(0)
        pop = max(pop, 1e-10)
        novelty_scores.append(-np.log2(pop))

    return float(np.mean(novelty_scores))
