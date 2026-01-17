#!/usr/bin/env python
"""Train baseline recommendation models."""

import pandas as pd
from loguru import logger

from src.config import settings
from src.models.baselines.popular import PopularItemsRecommender
from src.models.baselines.random_model import RandomRecommender
from src.models.collaborative.item_cf import ItemBasedCF
from src.models.collaborative.user_cf import UserBasedCF


def train_baselines():
    """Train all baseline models."""
    logger.info("=" * 60)
    logger.info("Training Baseline Models")
    logger.info("=" * 60)

    # Create models directory
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_path = settings.data_processed_dir / "train.parquet"
    if not train_path.exists():
        logger.error(f"Training data not found at {train_path}")
        logger.error("Please run 'make process-data' first")
        return

    logger.info(f"Loading training data from {train_path}")
    train_data = pd.read_parquet(train_path)
    logger.info(f"Training data: {len(train_data):,} interactions")

    # Initialize models
    models = [
        RandomRecommender(seed=42),
        PopularItemsRecommender(),
        UserBasedCF(n_neighbors=50),
        ItemBasedCF(n_similar=50),
    ]

    # Train and save each model
    for model in models:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training {model.name}...")

        model.fit(train_data)

        # Save model
        model_path = settings.models_dir / f"{model.name}.joblib"
        model.save(model_path)

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

    # Print summary
    logger.info("\nTrained models:")
    for model in models:
        model_path = settings.models_dir / f"{model.name}.joblib"
        logger.info(f"  - {model.name}: {model_path}")


if __name__ == "__main__":
    train_baselines()
