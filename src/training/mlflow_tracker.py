"""MLflow experiment tracking utilities."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from loguru import logger

from src.config import settings


class MLflowTracker:
    """MLflow experiment tracker for recommendation models."""

    def __init__(
        self,
        experiment_name: str = "recsys-ecommerce",
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment.
            tracking_uri: MLflow tracking server URI. Defaults to local file store.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or f"file://{settings.project_root / 'mlruns'}"
        self._active_run = None

        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"MLflow experiment: {self.experiment_name}")

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run.

        Args:
            run_name: Name for the run.
            tags: Optional tags for the run.
            nested: Whether this is a nested run.

        Returns:
            Active MLflow run.
        """
        self._active_run = mlflow.start_run(run_name=run_name, nested=nested)

        if tags:
            mlflow.set_tags(tags)

        logger.info(f"Started MLflow run: {run_name} (ID: {self._active_run.info.run_id})")
        return self._active_run

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED).
        """
        if self._active_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
            self._active_run = None

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ):
        """Context manager for MLflow runs.

        Args:
            run_name: Name for the run.
            tags: Optional tags for the run.
            nested: Whether this is a nested run.

        Yields:
            Active MLflow run.
        """
        try:
            active_run = self.start_run(run_name=run_name, tags=tags, nested=nested)
            yield active_run
        except Exception as e:
            self.end_run(status="FAILED")
            raise e
        else:
            self.end_run(status="FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the current run.

        Args:
            params: Dictionary of parameter names and values.
        """
        # MLflow has limits on param value length, truncate if needed
        sanitized_params = {}
        for key, value in params.items():
            str_value = str(value)
            if len(str_value) > 250:
                str_value = str_value[:247] + "..."
            sanitized_params[key] = str_value

        mlflow.log_params(sanitized_params)
        logger.debug(f"Logged {len(sanitized_params)} parameters")

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        self.log_params({key: value})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for tracking metric history.
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log a model artifact.

        Args:
            model: Model object to log (uses joblib serialization).
            artifact_path: Path within the artifact store.
            registered_model_name: Optional name to register the model.
        """
        import joblib
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            joblib.dump(model, model_path)

            mlflow.log_artifact(str(model_path), artifact_path)

            if registered_model_name:
                # Get the artifact URI for registration
                artifact_uri = mlflow.get_artifact_uri(artifact_path)
                mlflow.register_model(artifact_uri, registered_model_name)

        logger.info(f"Logged model to {artifact_path}")

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log an artifact file.

        Args:
            local_path: Path to the local file.
            artifact_path: Optional destination path in artifact store.
        """
        mlflow.log_artifact(str(local_path), artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: dict, artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log.
            artifact_file: Filename for the artifact.
        """
        mlflow.log_dict(dictionary, artifact_file)
        logger.debug(f"Logged dict to {artifact_file}")

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib/plotly figure.

        Args:
            figure: Figure object.
            artifact_file: Filename for the artifact.
        """
        mlflow.log_figure(figure, artifact_file)
        logger.debug(f"Logged figure to {artifact_file}")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the current run.

        Args:
            key: Tag name.
            value: Tag value.
        """
        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set multiple tags on the current run.

        Args:
            tags: Dictionary of tag names and values.
        """
        mlflow.set_tags(tags)

    @property
    def active_run(self) -> mlflow.ActiveRun | None:
        """Get the currently active run."""
        return self._active_run

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        if self._active_run:
            return self._active_run.info.run_id
        return None

    def get_experiment_id(self) -> str:
        """Get the experiment ID."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            return experiment.experiment_id
        raise ValueError(f"Experiment '{self.experiment_name}' not found")
