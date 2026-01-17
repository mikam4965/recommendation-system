"""RetailRocket dataset loader."""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import settings


class RetailRocketLoader:
    """Loader for RetailRocket e-commerce dataset."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize loader.

        Args:
            data_dir: Path to raw data directory. Defaults to settings.data_raw_dir.
        """
        self.data_dir = data_dir or settings.data_raw_dir

    def load_events(self) -> pd.DataFrame:
        """Load events.csv file.

        Returns:
            DataFrame with columns: timestamp, visitor_id, event, item_id, transaction_id
        """
        events_path = self.data_dir / "events.csv"
        logger.info(f"Loading events from {events_path}")

        df = pd.read_csv(
            events_path,
            dtype={
                "visitorid": "int64",
                "event": "category",
                "itemid": "int64",
                "transactionid": "float64",  # Has NaN values
            },
        )

        # Rename columns to snake_case
        df = df.rename(
            columns={
                "visitorid": "visitor_id",
                "itemid": "item_id",
                "transactionid": "transaction_id",
            }
        )

        # Convert timestamp from milliseconds to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Convert event to string for easier handling
        df["event"] = df["event"].astype(str)

        logger.info(f"Loaded {len(df):,} events")
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        logger.info(f"Unique visitors: {df['visitor_id'].nunique():,}")
        logger.info(f"Unique items: {df['item_id'].nunique():,}")

        return df

    def load_item_properties(self) -> pd.DataFrame:
        """Load item properties from both part files.

        Returns:
            DataFrame with columns: timestamp, item_id, property, value
        """
        part1_path = self.data_dir / "item_properties_part1.csv"
        part2_path = self.data_dir / "item_properties_part2.csv"

        logger.info(f"Loading item properties from {part1_path} and {part2_path}")

        dfs = []
        for path in [part1_path, part2_path]:
            if path.exists():
                df = pd.read_csv(
                    path,
                    dtype={
                        "itemid": "int64",
                        "property": "category",
                        "value": "str",
                    },
                )
                dfs.append(df)
                logger.info(f"Loaded {len(df):,} rows from {path.name}")

        if not dfs:
            logger.warning("No item properties files found")
            return pd.DataFrame(columns=["timestamp", "item_id", "property", "value"])

        df = pd.concat(dfs, ignore_index=True)

        # Rename columns
        df = df.rename(columns={"itemid": "item_id"})

        # Convert property to string
        df["property"] = df["property"].astype(str)

        logger.info(f"Total item properties: {len(df):,}")
        logger.info(f"Unique items with properties: {df['item_id'].nunique():,}")
        logger.info(f"Unique properties: {df['property'].nunique()}")

        return df

    def load_category_tree(self) -> pd.DataFrame:
        """Load category hierarchy.

        Returns:
            DataFrame with columns: category_id, parent_id
        """
        category_path = self.data_dir / "category_tree.csv"
        logger.info(f"Loading category tree from {category_path}")

        if not category_path.exists():
            logger.warning("Category tree file not found")
            return pd.DataFrame(columns=["category_id", "parent_id"])

        df = pd.read_csv(
            category_path,
            dtype={
                "categoryid": "int64",
                "parentid": "float64",  # Has NaN for root categories
            },
        )

        # Rename columns
        df = df.rename(columns={"categoryid": "category_id", "parentid": "parent_id"})

        logger.info(f"Loaded {len(df):,} categories")
        logger.info(f"Root categories: {df['parent_id'].isna().sum()}")

        return df

    def extract_item_features(self, item_properties: pd.DataFrame) -> pd.DataFrame:
        """Extract item features from properties.

        Extracts the most recent categoryid for each item.

        Args:
            item_properties: DataFrame from load_item_properties()

        Returns:
            DataFrame with columns: item_id, category_id
        """
        logger.info("Extracting item features from properties")

        # Filter only categoryid property
        categories = item_properties[item_properties["property"] == "categoryid"].copy()

        if categories.empty:
            logger.warning("No category information found in properties")
            return pd.DataFrame(columns=["item_id", "category_id"])

        # Get most recent category for each item
        categories = categories.sort_values("timestamp", ascending=False)
        categories = categories.drop_duplicates(subset=["item_id"], keep="first")

        # Convert value to int (category_id)
        categories["category_id"] = pd.to_numeric(categories["value"], errors="coerce")
        categories = categories.dropna(subset=["category_id"])
        categories["category_id"] = categories["category_id"].astype("int64")

        result = categories[["item_id", "category_id"]].copy()

        logger.info(f"Extracted categories for {len(result):,} items")

        return result

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all RetailRocket data.

        Returns:
            Dictionary with keys: events, item_properties, category_tree, item_features
        """
        logger.info("Loading all RetailRocket data")

        events = self.load_events()
        item_properties = self.load_item_properties()
        category_tree = self.load_category_tree()
        item_features = self.extract_item_features(item_properties)

        return {
            "events": events,
            "item_properties": item_properties,
            "category_tree": category_tree,
            "item_features": item_features,
        }
