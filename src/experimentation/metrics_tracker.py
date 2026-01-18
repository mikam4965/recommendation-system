"""Metrics tracking for A/B experiments.

Tracks experiment metrics with support for multiple storage backends.
"""

import json
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger


@dataclass
class MetricRecord:
    """Single metric record."""

    experiment_name: str
    variant_name: str
    user_id: int
    metric_name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "variant_name": self.variant_name,
            "user_id": self.user_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a variant."""

    variant_name: str
    metric_name: str
    count: int
    sum_value: float
    min_value: float
    max_value: float
    unique_users: int

    @property
    def mean_value(self) -> float:
        """Calculate mean value."""
        return self.sum_value / self.count if self.count > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_name": self.variant_name,
            "metric_name": self.metric_name,
            "count": self.count,
            "sum": self.sum_value,
            "mean": self.mean_value,
            "min": self.min_value,
            "max": self.max_value,
            "unique_users": self.unique_users,
        }


class MetricsBackend(ABC):
    """Abstract base class for metrics storage backends."""

    @abstractmethod
    def store(self, record: MetricRecord) -> None:
        """Store a metric record."""
        pass

    @abstractmethod
    def query(
        self,
        experiment_name: str,
        metric_name: str | None = None,
        variant_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MetricRecord]:
        """Query metric records."""
        pass


class InMemoryMetricsBackend(MetricsBackend):
    """In-memory metrics storage for development/testing."""

    def __init__(self, max_records: int = 100000):
        """Initialize backend.

        Args:
            max_records: Maximum records to keep in memory.
        """
        self.max_records = max_records
        self._records: list[MetricRecord] = []
        self._lock = threading.Lock()

    def store(self, record: MetricRecord) -> None:
        """Store a metric record."""
        with self._lock:
            self._records.append(record)

            # Trim if too many records
            if len(self._records) > self.max_records:
                self._records = self._records[-self.max_records:]

    def query(
        self,
        experiment_name: str,
        metric_name: str | None = None,
        variant_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MetricRecord]:
        """Query metric records."""
        with self._lock:
            results = []
            for record in self._records:
                if record.experiment_name != experiment_name:
                    continue
                if metric_name and record.metric_name != metric_name:
                    continue
                if variant_name and record.variant_name != variant_name:
                    continue
                if start_time and record.timestamp < start_time:
                    continue
                if end_time and record.timestamp > end_time:
                    continue
                results.append(record)
            return results

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()


class ClickHouseMetricsBackend(MetricsBackend):
    """ClickHouse metrics storage for production.

    Requires clickhouse-driver package.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9000,
        database: str = "recsys",
        table: str = "experiment_metrics",
    ):
        """Initialize ClickHouse backend.

        Args:
            host: ClickHouse host.
            port: ClickHouse native port.
            database: Database name.
            table: Table name.
        """
        self.host = host
        self.port = port
        self.database = database
        self.table = table
        self._client = None

        try:
            from clickhouse_driver import Client

            self._client = Client(host=host, port=port)
            self._ensure_table()
            self._available = True
        except ImportError:
            logger.warning(
                "clickhouse-driver not installed. "
                "Install with: pip install clickhouse-driver"
            )
            self._available = False
        except Exception as e:
            logger.warning(f"Failed to connect to ClickHouse: {e}")
            self._available = False

    def _ensure_table(self) -> None:
        """Create table if not exists."""
        if not self._client:
            return

        self._client.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")

        self._client.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.database}.{self.table} (
                experiment_name String,
                variant_name String,
                user_id Int64,
                metric_name String,
                value Float64,
                timestamp DateTime64(3),
                metadata String
            ) ENGINE = MergeTree()
            ORDER BY (experiment_name, variant_name, metric_name, timestamp)
            PARTITION BY toYYYYMM(timestamp)
        """)

    def store(self, record: MetricRecord) -> None:
        """Store a metric record in ClickHouse."""
        if not self._available or not self._client:
            return

        try:
            self._client.execute(
                f"""
                INSERT INTO {self.database}.{self.table}
                (experiment_name, variant_name, user_id, metric_name, value, timestamp, metadata)
                VALUES
                """,
                [
                    (
                        record.experiment_name,
                        record.variant_name,
                        record.user_id,
                        record.metric_name,
                        record.value,
                        record.timestamp,
                        json.dumps(record.metadata),
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Failed to store metric in ClickHouse: {e}")

    def query(
        self,
        experiment_name: str,
        metric_name: str | None = None,
        variant_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MetricRecord]:
        """Query metric records from ClickHouse."""
        if not self._available or not self._client:
            return []

        conditions = [f"experiment_name = '{experiment_name}'"]

        if metric_name:
            conditions.append(f"metric_name = '{metric_name}'")
        if variant_name:
            conditions.append(f"variant_name = '{variant_name}'")
        if start_time:
            conditions.append(f"timestamp >= '{start_time.isoformat()}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time.isoformat()}'")

        where_clause = " AND ".join(conditions)

        try:
            rows = self._client.execute(
                f"""
                SELECT experiment_name, variant_name, user_id, metric_name,
                       value, timestamp, metadata
                FROM {self.database}.{self.table}
                WHERE {where_clause}
                ORDER BY timestamp
                """
            )

            return [
                MetricRecord(
                    experiment_name=row[0],
                    variant_name=row[1],
                    user_id=row[2],
                    metric_name=row[3],
                    value=row[4],
                    timestamp=row[5],
                    metadata=json.loads(row[6]) if row[6] else {},
                )
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to query ClickHouse: {e}")
            return []


class ExperimentMetricsTracker:
    """Tracks and aggregates experiment metrics.

    Features:
    - Multiple storage backends (in-memory, ClickHouse)
    - Real-time aggregation
    - Query interface for analysis
    """

    def __init__(
        self,
        backend: MetricsBackend | None = None,
    ):
        """Initialize metrics tracker.

        Args:
            backend: Storage backend. Defaults to in-memory.
        """
        self.backend = backend or InMemoryMetricsBackend()

        # Real-time aggregation cache
        self._aggregations: dict[
            tuple[str, str, str], dict[str, Any]
        ] = defaultdict(lambda: {
            "count": 0,
            "sum": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "users": set(),
        })
        self._lock = threading.Lock()

    def log_metric(
        self,
        experiment_name: str,
        variant_name: str,
        user_id: int,
        metric_name: str,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a metric value.

        Args:
            experiment_name: Experiment name.
            variant_name: Variant name.
            user_id: User ID.
            metric_name: Metric name.
            value: Metric value.
            metadata: Additional metadata.
        """
        record = MetricRecord(
            experiment_name=experiment_name,
            variant_name=variant_name,
            user_id=user_id,
            metric_name=metric_name,
            value=value,
            metadata=metadata or {},
        )

        # Store in backend
        self.backend.store(record)

        # Update real-time aggregation
        key = (experiment_name, variant_name, metric_name)
        with self._lock:
            agg = self._aggregations[key]
            agg["count"] += 1
            agg["sum"] += value
            agg["min"] = min(agg["min"], value)
            agg["max"] = max(agg["max"], value)
            agg["users"].add(user_id)

    def get_aggregated_metrics(
        self,
        experiment_name: str,
        metric_name: str | None = None,
    ) -> dict[str, list[AggregatedMetrics]]:
        """Get aggregated metrics for an experiment.

        Args:
            experiment_name: Experiment name.
            metric_name: Optional metric name filter.

        Returns:
            Dictionary mapping metric names to list of aggregations per variant.
        """
        results: dict[str, list[AggregatedMetrics]] = defaultdict(list)

        with self._lock:
            for (exp, variant, metric), agg in self._aggregations.items():
                if exp != experiment_name:
                    continue
                if metric_name and metric != metric_name:
                    continue

                if agg["count"] > 0:
                    results[metric].append(
                        AggregatedMetrics(
                            variant_name=variant,
                            metric_name=metric,
                            count=agg["count"],
                            sum_value=agg["sum"],
                            min_value=agg["min"] if agg["min"] != float("inf") else 0,
                            max_value=agg["max"] if agg["max"] != float("-inf") else 0,
                            unique_users=len(agg["users"]),
                        )
                    )

        return dict(results)

    def get_experiment_summary(
        self,
        experiment_name: str,
    ) -> dict[str, Any]:
        """Get summary statistics for an experiment.

        Args:
            experiment_name: Experiment name.

        Returns:
            Summary dictionary.
        """
        aggregations = self.get_aggregated_metrics(experiment_name)

        summary = {
            "experiment_name": experiment_name,
            "metrics": {},
        }

        for metric_name, variants in aggregations.items():
            summary["metrics"][metric_name] = {
                "variants": {v.variant_name: v.to_dict() for v in variants},
                "total_count": sum(v.count for v in variants),
                "total_users": sum(v.unique_users for v in variants),
            }

        return summary

    def query_metrics(
        self,
        experiment_name: str,
        metric_name: str | None = None,
        variant_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MetricRecord]:
        """Query raw metric records.

        Args:
            experiment_name: Experiment name.
            metric_name: Optional metric name filter.
            variant_name: Optional variant name filter.
            start_time: Start time filter.
            end_time: End time filter.

        Returns:
            List of metric records.
        """
        return self.backend.query(
            experiment_name=experiment_name,
            metric_name=metric_name,
            variant_name=variant_name,
            start_time=start_time,
            end_time=end_time,
        )

    def export_metrics(
        self,
        experiment_name: str,
    ) -> list[dict[str, Any]]:
        """Export all metrics for an experiment.

        Args:
            experiment_name: Experiment name.

        Returns:
            List of metric dictionaries.
        """
        records = self.backend.query(experiment_name=experiment_name)
        return [r.to_dict() for r in records]

    def reset_aggregations(self, experiment_name: str | None = None) -> None:
        """Reset real-time aggregations.

        Args:
            experiment_name: Optional experiment to reset. Resets all if None.
        """
        with self._lock:
            if experiment_name:
                keys_to_remove = [
                    k for k in self._aggregations if k[0] == experiment_name
                ]
                for key in keys_to_remove:
                    del self._aggregations[key]
            else:
                self._aggregations.clear()
