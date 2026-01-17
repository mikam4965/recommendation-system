#!/usr/bin/env python
"""Process RetailRocket data: clean, build sessions, and split."""

from loguru import logger

from src.config import settings
from src.data.loaders.retailrocket import RetailRocketLoader
from src.data.processors.cleaner import DataCleaner
from src.data.processors.session_builder import SessionBuilder
from src.data.processors.splitter import TimeBasedSplitter


def process_data():
    """Run full data processing pipeline."""
    logger.info("=" * 60)
    logger.info("RetailRocket Data Processing Pipeline")
    logger.info("=" * 60)

    # Create output directories
    settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
    settings.data_interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    logger.info("\n[1/5] Loading raw data...")
    loader = RetailRocketLoader()
    data = loader.load_all()

    events = data["events"]
    item_features = data["item_features"]

    # 2. Clean data
    logger.info("\n[2/5] Cleaning data...")
    cleaner = DataCleaner()
    events_clean = cleaner.clean(events, remove_bots=True, filter_rare_items=True)

    # 3. Build sessions
    logger.info("\n[3/5] Building sessions...")
    session_builder = SessionBuilder()
    events_with_sessions = session_builder.build_sessions(events_clean)

    session_stats = session_builder.get_session_stats(events_with_sessions)
    logger.info(f"Session statistics: {session_stats}")

    # 4. Split data
    logger.info("\n[4/5] Splitting data (time-based)...")
    splitter = TimeBasedSplitter()
    train, val, test = splitter.split(events_with_sessions)

    split_stats = splitter.get_split_stats(train, val, test)
    logger.info(f"Split statistics: {split_stats}")

    # 5. Save processed data
    logger.info("\n[5/5] Saving processed data...")

    # Save clean events with sessions
    events_clean_path = settings.data_processed_dir / "events_clean.parquet"
    events_with_sessions.to_parquet(events_clean_path, index=False)
    logger.info(f"Saved: {events_clean_path}")

    # Save train/val/test splits
    train_path = settings.data_processed_dir / "train.parquet"
    train.to_parquet(train_path, index=False)
    logger.info(f"Saved: {train_path} ({len(train):,} events)")

    val_path = settings.data_processed_dir / "val.parquet"
    val.to_parquet(val_path, index=False)
    logger.info(f"Saved: {val_path} ({len(val):,} events)")

    test_path = settings.data_processed_dir / "test.parquet"
    test.to_parquet(test_path, index=False)
    logger.info(f"Saved: {test_path} ({len(test):,} events)")

    # Save item features
    if not item_features.empty:
        item_features_path = settings.data_processed_dir / "item_features.parquet"
        item_features.to_parquet(item_features_path, index=False)
        logger.info(f"Saved: {item_features_path} ({len(item_features):,} items)")

    logger.info("\n" + "=" * 60)
    logger.info("Processing complete!")
    logger.info("=" * 60)

    # Print summary
    logger.info("\nSummary:")
    logger.info(f"  Total events (clean): {len(events_with_sessions):,}")
    logger.info(f"  Total users: {events_with_sessions['visitor_id'].nunique():,}")
    logger.info(f"  Total items: {events_with_sessions['item_id'].nunique():,}")
    logger.info(f"  Total sessions: {events_with_sessions['session_id'].nunique():,}")
    logger.info(f"  Train: {len(train):,} events")
    logger.info(f"  Val: {len(val):,} events")
    logger.info(f"  Test: {len(test):,} events")


if __name__ == "__main__":
    process_data()
