"""Model evaluation utilities."""

from collections import defaultdict
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.evaluation.metrics import (
    average_precision,
    coverage,
    hit_rate,
    mrr,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)
from src.models.base import BaseRecommender


class Evaluator:
    """Evaluate recommendation models."""

    def __init__(
        self,
        k_values: list[int] | None = None,
        relevance_events: list[str] | None = None,
    ):
        """Initialize evaluator.

        Args:
            k_values: Values of K for @K metrics. Default: [5, 10, 20].
            relevance_events: Event types considered as "relevant".
                Default: ["addtocart", "transaction"].
        """
        self.k_values = k_values or [5, 10, 20]
        self.relevance_events = relevance_events or ["addtocart", "transaction"]

    def _build_ground_truth(
        self, test_data: pd.DataFrame
    ) -> dict[int, set[int]]:
        """Build ground truth from test data.

        Args:
            test_data: Test DataFrame with visitor_id, item_id, event columns.

        Returns:
            Dictionary mapping user_id to set of relevant item_ids.
        """
        # Filter to relevant events only
        relevant = test_data[test_data["event"].isin(self.relevance_events)]

        ground_truth = defaultdict(set)
        for _, row in relevant.iterrows():
            ground_truth[row["visitor_id"]].add(row["item_id"])

        return dict(ground_truth)

    def evaluate(
        self,
        model: BaseRecommender,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        n_items: int = 20,
        max_users: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, float]:
        """Evaluate a model on test data.

        Args:
            model: Fitted recommender model.
            train_data: Training DataFrame (for item popularity stats).
            test_data: Test DataFrame for evaluation.
            n_items: Number of items to recommend per user.
            max_users: Maximum users to evaluate (for speed). None = all.
            show_progress: Whether to show progress bar.

        Returns:
            Dictionary of metric names to scores.
        """
        logger.info(f"Evaluating {model.name} model")

        # Build ground truth
        ground_truth = self._build_ground_truth(test_data)

        # Get users that appear in both train and test (with relevant events in test)
        train_users = set(train_data["visitor_id"].unique())
        test_users_with_relevant = set(ground_truth.keys())
        eval_users = list(train_users & test_users_with_relevant)

        if not eval_users:
            logger.warning("No users found in both train and test with relevant events")
            return {}

        if max_users and len(eval_users) > max_users:
            eval_users = eval_users[:max_users]

        logger.info(f"Evaluating on {len(eval_users):,} users")

        # Compute item popularity for novelty
        item_popularity = train_data["item_id"].value_counts().to_dict()
        total_interactions = len(train_data)
        catalog_size = train_data["item_id"].nunique()

        # Collect metrics
        metrics: dict[str, list[float]] = defaultdict(list)
        all_recommendations: list[list[int]] = []

        iterator = tqdm(eval_users, desc=f"Evaluating {model.name}") if show_progress else eval_users

        for user_id in iterator:
            # Get recommendations
            recs = model.recommend(user_id, n_items=n_items, exclude_seen=True)
            recommended_items = [item_id for item_id, _ in recs]
            all_recommendations.append(recommended_items)

            relevant_items = ground_truth[user_id]

            # Calculate metrics
            metrics["hit_rate"].append(hit_rate(recommended_items, relevant_items))
            metrics["mrr"].append(mrr(recommended_items, relevant_items))
            metrics["map"].append(average_precision(recommended_items, relevant_items))

            for k in self.k_values:
                metrics[f"precision@{k}"].append(
                    precision_at_k(recommended_items, relevant_items, k)
                )
                metrics[f"recall@{k}"].append(
                    recall_at_k(recommended_items, relevant_items, k)
                )
                metrics[f"ndcg@{k}"].append(
                    ndcg_at_k(recommended_items, relevant_items, k)
                )

            # Novelty
            metrics["novelty"].append(
                novelty(recommended_items, item_popularity, total_interactions)
            )

        # Aggregate results
        results = {
            "model": model.name,
            "n_users": len(eval_users),
        }

        for metric_name, values in metrics.items():
            results[metric_name] = float(pd.Series(values).mean())

        # Coverage (aggregate metric)
        results["coverage"] = coverage(all_recommendations, catalog_size)

        logger.info(f"Evaluation complete for {model.name}")

        return results

    def evaluate_models(
        self,
        models: list[BaseRecommender],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        n_items: int = 20,
        max_users: int | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Evaluate multiple models and return comparison table.

        Args:
            models: List of fitted recommender models.
            train_data: Training DataFrame.
            test_data: Test DataFrame.
            n_items: Number of items to recommend per user.
            max_users: Maximum users to evaluate per model.
            show_progress: Whether to show progress bar.

        Returns:
            DataFrame with metrics for each model.
        """
        results = []

        for model in models:
            model_results = self.evaluate(
                model=model,
                train_data=train_data,
                test_data=test_data,
                n_items=n_items,
                max_users=max_users,
                show_progress=show_progress,
            )
            results.append(model_results)

        df = pd.DataFrame(results)

        # Reorder columns
        cols = ["model", "n_users"]
        metric_cols = [c for c in df.columns if c not in cols]
        metric_cols = sorted(metric_cols)
        df = df[cols + metric_cols]

        return df

    def print_results(self, results: pd.DataFrame) -> None:
        """Print results table in a nice format.

        Args:
            results: DataFrame from evaluate_models().
        """
        # Format floats
        format_cols = [c for c in results.columns if c not in ["model", "n_users"]]

        formatted = results.copy()
        for col in format_cols:
            formatted[col] = formatted[col].apply(lambda x: f"{x:.4f}")

        print("\n" + "=" * 80)
        print("MODEL EVALUATION RESULTS")
        print("=" * 80)
        print(formatted.to_string(index=False))
        print("=" * 80 + "\n")
