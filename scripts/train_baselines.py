#!/usr/bin/env python3
"""Train and evaluate baseline recommendation models."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.data.loaders.retailrocket import RetailRocketLoader
from src.data.processors.splitter import TimeBasedSplitter
from src.evaluation.evaluator import Evaluator
from src.models.baselines.popular import PopularItemsRecommender
from src.models.baselines.random_model import RandomRecommender
from src.models.collaborative.item_cf import ItemBasedCF
from src.models.collaborative.user_cf import UserBasedCF


def main():
    """Train and evaluate all baseline models."""
    logger.info("=" * 60)
    logger.info("BASELINE MODELS TRAINING")
    logger.info("=" * 60)

    # Load data
    logger.info("\n[1/4] Loading data...")
    loader = RetailRocketLoader()
    events = loader.load_events()

    # Split data
    logger.info("\n[2/4] Splitting data...")
    splitter = TimeBasedSplitter(train_ratio=0.80, val_ratio=0.00, test_ratio=0.20)
    train, _, test = splitter.split(events)

    # Calculate cold-start stats
    train_users = set(train["visitor_id"].unique())
    test_users = set(test["visitor_id"].unique())
    train_items = set(train["item_id"].unique())
    test_items = set(test["item_id"].unique())

    cold_start_users = len(test_users - train_users)
    cold_start_items = len(test_items - train_items)
    logger.info(f"Cold-start users in test: {cold_start_users:,} ({cold_start_users/len(test_users):.1%})")
    logger.info(f"Cold-start items in test: {cold_start_items:,} ({cold_start_items/len(test_items):.1%})")

    # Initialize models
    logger.info("\n[3/4] Training models...")
    models = [
        RandomRecommender(seed=42),
        PopularItemsRecommender(),
        # UserBasedCF is too slow on full dataset, skip for now
        # UserBasedCF(n_neighbors=50),
        ItemBasedCF(n_similar=50),
    ]

    # Train all models
    for model in models:
        logger.info(f"\nTraining {model.name}...")
        model.fit(train)

    # Evaluate
    logger.info("\n[4/4] Evaluating models...")
    evaluator = Evaluator(k_values=[5, 10, 20])

    # Limit evaluation to max_users for speed
    # Full evaluation can take very long time
    max_users = 10000

    results = evaluator.evaluate_models(
        models=models,
        train_data=train,
        test_data=test,
        n_items=20,
        max_users=max_users,
        show_progress=True,
    )

    # Print results
    evaluator.print_results(results)

    # Save results
    output_path = project_root / "data" / "processed" / "baseline_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
