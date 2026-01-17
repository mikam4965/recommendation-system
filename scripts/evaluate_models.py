#!/usr/bin/env python
"""Evaluate trained recommendation models."""

import pandas as pd
from loguru import logger

from src.config import settings
from src.evaluation.evaluator import Evaluator
from src.models.base import BaseRecommender


def evaluate_models():
    """Evaluate all trained models on test data."""
    logger.info("=" * 60)
    logger.info("Evaluating Models")
    logger.info("=" * 60)

    # Load train and test data
    train_path = settings.data_processed_dir / "train.parquet"
    test_path = settings.data_processed_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        logger.error("Train/test data not found")
        logger.error("Please run 'make process-data' first")
        return

    logger.info(f"Loading training data from {train_path}")
    train_data = pd.read_parquet(train_path)

    logger.info(f"Loading test data from {test_path}")
    test_data = pd.read_parquet(test_path)

    logger.info(f"Train: {len(train_data):,} interactions")
    logger.info(f"Test: {len(test_data):,} interactions")

    # Load trained models
    model_names = ["random", "popular", "user_cf", "item_cf"]
    models = []

    for name in model_names:
        model_path = settings.models_dir / f"{name}.joblib"
        if model_path.exists():
            model = BaseRecommender.load(model_path)
            models.append(model)
            logger.info(f"Loaded model: {name}")
        else:
            logger.warning(f"Model not found: {model_path}")

    if not models:
        logger.error("No models found. Please run 'make train-baseline' first")
        return

    # Initialize evaluator
    evaluator = Evaluator(
        k_values=[5, 10, 20],
        relevance_events=["addtocart", "transaction"],
    )

    # Evaluate models
    # Limit users for faster evaluation during development
    max_users = 5000  # Set to None for full evaluation

    results = evaluator.evaluate_models(
        models=models,
        train_data=train_data,
        test_data=test_data,
        n_items=20,
        max_users=max_users,
        show_progress=True,
    )

    # Print results
    evaluator.print_results(results)

    # Save results
    results_path = settings.models_dir / "evaluation_results.csv"
    results.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    evaluate_models()
