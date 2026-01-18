"""A/B Testing Framework for recommendation experiments.

Provides consistent user-to-variant assignment, experiment configuration,
and integration with metrics tracking.
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger


class ExperimentStatus(Enum):
    """Experiment status enum."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class ExperimentVariant:
    """Variant configuration for an experiment."""

    name: str
    percentage: float  # Traffic percentage (0-100)
    model_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "percentage": self.percentage,
            "model_name": self.model_name,
            "config": self.config,
            "description": self.description,
        }


@dataclass
class Experiment:
    """Experiment configuration."""

    name: str
    description: str
    variants: list[ExperimentVariant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_date: datetime | None = None
    end_date: datetime | None = None
    target_metric: str = "ndcg@10"
    min_sample_size: int = 1000
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate experiment configuration."""
        # Ensure percentages sum to 100
        total_percentage = sum(v.percentage for v in self.variants)
        if abs(total_percentage - 100.0) > 0.01:
            raise ValueError(
                f"Variant percentages must sum to 100, got {total_percentage}"
            )

        # Ensure at least 2 variants
        if len(self.variants) < 2:
            raise ValueError("Experiment must have at least 2 variants")

    def get_variant(self, name: str) -> ExperimentVariant | None:
        """Get variant by name."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "target_metric": self.target_metric,
            "min_sample_size": self.min_sample_size,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class VariantAssignment:
    """Result of variant assignment for a user."""

    experiment_name: str
    variant_name: str
    user_id: int
    assigned_at: datetime = field(default_factory=datetime.now)
    model_name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "variant_name": self.variant_name,
            "user_id": self.user_id,
            "assigned_at": self.assigned_at.isoformat(),
            "model_name": self.model_name,
            "config": self.config,
        }


class ABTestManager:
    """Manages A/B test experiments and variant assignments.

    Features:
    - Consistent hashing for stable user-to-variant assignment
    - Multiple concurrent experiments support
    - Experiment lifecycle management
    - Integration with metrics tracking

    Usage:
        manager = ABTestManager()

        # Create experiment
        experiment = Experiment(
            name="hybrid_vs_sasrec",
            description="Compare hybrid model vs SASRec",
            variants=[
                ExperimentVariant("control", 50, model_name="hybrid"),
                ExperimentVariant("treatment", 50, model_name="sasrec"),
            ]
        )
        manager.add_experiment(experiment)
        manager.start_experiment("hybrid_vs_sasrec")

        # Get variant for user
        assignment = manager.get_variant(user_id=12345, experiment_name="hybrid_vs_sasrec")
        model_to_use = assignment.model_name
    """

    def __init__(
        self,
        salt: str = "recsys_ab_test",
        metrics_tracker: "ExperimentMetricsTracker | None" = None,
    ):
        """Initialize A/B test manager.

        Args:
            salt: Salt for consistent hashing.
            metrics_tracker: Optional metrics tracker for logging.
        """
        self.salt = salt
        self.metrics_tracker = metrics_tracker

        self._experiments: dict[str, Experiment] = {}
        self._assignment_cache: dict[tuple[int, str], VariantAssignment] = {}
        self._lock = threading.RLock()

    def add_experiment(self, experiment: Experiment) -> None:
        """Add a new experiment.

        Args:
            experiment: Experiment configuration.
        """
        with self._lock:
            if experiment.name in self._experiments:
                raise ValueError(f"Experiment '{experiment.name}' already exists")
            self._experiments[experiment.name] = experiment
            logger.info(f"Added experiment: {experiment.name}")

    def remove_experiment(self, experiment_name: str) -> None:
        """Remove an experiment.

        Args:
            experiment_name: Name of experiment to remove.
        """
        with self._lock:
            if experiment_name not in self._experiments:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            experiment = self._experiments[experiment_name]
            if experiment.status == ExperimentStatus.RUNNING:
                raise ValueError("Cannot remove running experiment. Stop it first.")

            del self._experiments[experiment_name]

            # Clear cached assignments
            self._assignment_cache = {
                k: v
                for k, v in self._assignment_cache.items()
                if k[1] != experiment_name
            }

            logger.info(f"Removed experiment: {experiment_name}")

    def start_experiment(self, experiment_name: str) -> None:
        """Start an experiment.

        Args:
            experiment_name: Name of experiment to start.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_name)
            experiment.status = ExperimentStatus.RUNNING
            experiment.start_date = datetime.now()
            logger.info(f"Started experiment: {experiment_name}")

    def stop_experiment(self, experiment_name: str) -> None:
        """Stop a running experiment.

        Args:
            experiment_name: Name of experiment to stop.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_name)
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_date = datetime.now()
            logger.info(f"Stopped experiment: {experiment_name}")

    def pause_experiment(self, experiment_name: str) -> None:
        """Pause a running experiment.

        Args:
            experiment_name: Name of experiment to pause.
        """
        with self._lock:
            experiment = self._get_experiment(experiment_name)
            experiment.status = ExperimentStatus.PAUSED
            logger.info(f"Paused experiment: {experiment_name}")

    def get_variant(
        self,
        user_id: int,
        experiment_name: str,
        use_cache: bool = True,
    ) -> VariantAssignment | None:
        """Get variant assignment for a user.

        Uses consistent hashing to ensure stable assignment across calls.

        Args:
            user_id: User ID.
            experiment_name: Experiment name.
            use_cache: Use cached assignment if available.

        Returns:
            Variant assignment or None if experiment not running.
        """
        # Check cache first
        cache_key = (user_id, experiment_name)
        if use_cache and cache_key in self._assignment_cache:
            return self._assignment_cache[cache_key]

        with self._lock:
            experiment = self._experiments.get(experiment_name)
            if not experiment:
                logger.warning(f"Experiment '{experiment_name}' not found")
                return None

            if experiment.status != ExperimentStatus.RUNNING:
                logger.debug(
                    f"Experiment '{experiment_name}' not running "
                    f"(status: {experiment.status.value})"
                )
                return None

            # Compute hash for consistent assignment
            hash_value = self._compute_hash(user_id, experiment_name)

            # Find variant based on hash
            variant = self._select_variant(experiment, hash_value)

            # Create assignment
            assignment = VariantAssignment(
                experiment_name=experiment_name,
                variant_name=variant.name,
                user_id=user_id,
                model_name=variant.model_name,
                config=variant.config.copy(),
            )

            # Cache assignment
            if use_cache:
                self._assignment_cache[cache_key] = assignment

            return assignment

    def get_all_assignments(
        self,
        user_id: int,
    ) -> list[VariantAssignment]:
        """Get all active experiment assignments for a user.

        Args:
            user_id: User ID.

        Returns:
            List of variant assignments.
        """
        assignments = []

        with self._lock:
            for experiment_name, experiment in self._experiments.items():
                if experiment.status == ExperimentStatus.RUNNING:
                    assignment = self.get_variant(user_id, experiment_name)
                    if assignment:
                        assignments.append(assignment)

        return assignments

    def log_metric(
        self,
        user_id: int,
        experiment_name: str,
        metric_name: str,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Log a metric value for an experiment.

        Args:
            user_id: User ID.
            experiment_name: Experiment name.
            metric_name: Metric name.
            value: Metric value.
            metadata: Additional metadata.

        Returns:
            True if metric was logged successfully.
        """
        assignment = self.get_variant(user_id, experiment_name)
        if not assignment:
            logger.warning(
                f"Cannot log metric: no assignment for user {user_id} "
                f"in experiment {experiment_name}"
            )
            return False

        if self.metrics_tracker:
            self.metrics_tracker.log_metric(
                experiment_name=experiment_name,
                variant_name=assignment.variant_name,
                user_id=user_id,
                metric_name=metric_name,
                value=value,
                metadata=metadata,
            )
            return True

        logger.debug(
            f"Metric logged: {experiment_name}/{assignment.variant_name}/"
            f"{metric_name}={value}"
        )
        return True

    def log_conversion(
        self,
        user_id: int,
        experiment_name: str,
        conversion_type: str = "transaction",
        value: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Log a conversion event for an experiment.

        Args:
            user_id: User ID.
            experiment_name: Experiment name.
            conversion_type: Type of conversion.
            value: Conversion value (default 1.0).
            metadata: Additional metadata.

        Returns:
            True if conversion was logged successfully.
        """
        return self.log_metric(
            user_id=user_id,
            experiment_name=experiment_name,
            metric_name=f"conversion_{conversion_type}",
            value=value,
            metadata=metadata,
        )

    def get_experiment(self, experiment_name: str) -> Experiment | None:
        """Get experiment configuration.

        Args:
            experiment_name: Experiment name.

        Returns:
            Experiment configuration or None.
        """
        return self._experiments.get(experiment_name)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
    ) -> list[Experiment]:
        """List all experiments.

        Args:
            status: Filter by status.

        Returns:
            List of experiments.
        """
        with self._lock:
            experiments = list(self._experiments.values())
            if status:
                experiments = [e for e in experiments if e.status == status]
            return experiments

    def _get_experiment(self, experiment_name: str) -> Experiment:
        """Get experiment or raise error."""
        experiment = self._experiments.get(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        return experiment

    def _compute_hash(self, user_id: int, experiment_name: str) -> int:
        """Compute consistent hash for user-experiment pair.

        Args:
            user_id: User ID.
            experiment_name: Experiment name.

        Returns:
            Hash value in range [0, 100).
        """
        hash_input = f"{self.salt}_{user_id}_{experiment_name}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
        return hash_int % 100

    def _select_variant(
        self,
        experiment: Experiment,
        hash_value: int,
    ) -> ExperimentVariant:
        """Select variant based on hash value.

        Args:
            experiment: Experiment configuration.
            hash_value: Hash value in range [0, 100).

        Returns:
            Selected variant.
        """
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.percentage
            if hash_value < cumulative:
                return variant

        # Fallback to last variant (should not happen)
        return experiment.variants[-1]

    def export_config(self) -> dict[str, Any]:
        """Export all experiment configurations.

        Returns:
            Dictionary of experiment configurations.
        """
        with self._lock:
            return {
                name: exp.to_dict() for name, exp in self._experiments.items()
            }

    def import_config(self, config: dict[str, Any]) -> None:
        """Import experiment configurations.

        Args:
            config: Dictionary of experiment configurations.
        """
        for name, exp_config in config.items():
            variants = [
                ExperimentVariant(**v) for v in exp_config.pop("variants")
            ]
            status = ExperimentStatus(exp_config.pop("status"))

            # Parse dates
            start_date = exp_config.pop("start_date", None)
            if start_date:
                start_date = datetime.fromisoformat(start_date)

            end_date = exp_config.pop("end_date", None)
            if end_date:
                end_date = datetime.fromisoformat(end_date)

            created_at = exp_config.pop("created_at", None)
            if created_at:
                created_at = datetime.fromisoformat(created_at)
            else:
                created_at = datetime.now()

            experiment = Experiment(
                name=name,
                variants=variants,
                status=status,
                start_date=start_date,
                end_date=end_date,
                created_at=created_at,
                **exp_config,
            )

            self._experiments[name] = experiment

        logger.info(f"Imported {len(config)} experiments")


# Type hints import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.experimentation.metrics_tracker import ExperimentMetricsTracker
