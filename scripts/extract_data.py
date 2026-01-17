#!/usr/bin/env python
"""Extract RetailRocket dataset from zip archive."""

import zipfile
from pathlib import Path

from loguru import logger

from src.config import settings


def extract_data():
    """Extract RetailRocket dataset from zip file."""
    raw_dir = settings.data_raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Look for zip file
    zip_files = list(raw_dir.glob("*.zip"))

    if not zip_files:
        logger.info("No zip files found in data/raw/")
        logger.info("Please download the RetailRocket dataset from:")
        logger.info("https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset")
        logger.info(f"And place the zip file in: {raw_dir}")
        return

    for zip_path in zip_files:
        logger.info(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Extract only CSV files
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith(".csv"):
                    # Extract to raw_dir, flatten directory structure
                    filename = Path(file_info.filename).name
                    target_path = raw_dir / filename

                    if target_path.exists():
                        logger.info(f"  Skipping {filename} (already exists)")
                        continue

                    logger.info(f"  Extracting {filename}...")
                    with zip_ref.open(file_info) as source:
                        with open(target_path, "wb") as target:
                            target.write(source.read())

    # Check required files
    required_files = ["events.csv"]
    optional_files = [
        "item_properties_part1.csv",
        "item_properties_part2.csv",
        "category_tree.csv",
    ]

    for filename in required_files:
        if not (raw_dir / filename).exists():
            logger.error(f"Required file missing: {filename}")
            return

    for filename in optional_files:
        if (raw_dir / filename).exists():
            logger.info(f"Found: {filename}")
        else:
            logger.warning(f"Optional file missing: {filename}")

    logger.info("Extraction complete!")


if __name__ == "__main__":
    extract_data()
