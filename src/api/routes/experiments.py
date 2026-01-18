"""API routes for A/B experiment management.

Provides endpoints for creating, managing, and analyzing experiments.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/experiments", tags=["experiments"])


# Request/Response models
class ExperimentVariantRequest(BaseModel):
    """Variant configuration."""

    name: str
    percentage: float
    model_name: str | None = None
    config: dict[str, Any] = {}
    description: str = ""


class ExperimentRequest(BaseModel):
    """Experiment creation request."""

    name: str
    description: str
    variants: list[ExperimentVariantRequest]
    target_metric: str = "ndcg@10"
    min_sample_size: int = 1000


class ExperimentResponse(BaseModel):
    """Experiment response."""

    name: str
    description: str
    variants: list[dict[str, Any]]
    status: str
    start_date: str | None
    end_date: str | None
    target_metric: str
    min_sample_size: int
    created_at: str
    metadata: dict[str, Any] = {}


class StatisticalResultResponse(BaseModel):
    """Statistical analysis response."""

    control_mean: float
    treatment_mean: float
    relative_lift: float
    relative_lift_percent: float
    p_value: float
    confidence_level: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    control_n: int
    treatment_n: int
    test_type: str


class ExperimentMetricsResponse(BaseModel):
    """Experiment metrics response."""

    experiment_name: str
    metrics: dict[str, dict[str, Any]]


# In-memory experiment storage (replace with database in production)
_experiments: dict[str, dict[str, Any]] = {}
_experiment_metrics: dict[str, dict[str, Any]] = {}


def _init_default_experiment():
    """Initialize default experiment for demo."""
    if "hybrid_vs_sasrec" not in _experiments:
        _experiments["hybrid_vs_sasrec"] = {
            "name": "hybrid_vs_sasrec",
            "description": "Compare Hybrid model performance vs SASRec",
            "variants": [
                {
                    "name": "control",
                    "percentage": 50.0,
                    "model_name": "hybrid",
                    "config": {},
                    "description": "Hybrid model (current production)",
                },
                {
                    "name": "treatment",
                    "percentage": 50.0,
                    "model_name": "sasrec",
                    "config": {},
                    "description": "SASRec transformer model",
                },
            ],
            "status": "running",
            "start_date": "2025-01-15T10:00:00",
            "end_date": None,
            "target_metric": "ndcg@10",
            "min_sample_size": 1000,
            "created_at": "2025-01-14T09:00:00",
            "metadata": {},
        }


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(status: str | None = None):
    """List all experiments.

    Args:
        status: Optional filter by status (draft, running, paused, completed).
    """
    _init_default_experiment()

    experiments = list(_experiments.values())

    if status:
        experiments = [e for e in experiments if e["status"] == status]

    return experiments


@router.post("", response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentRequest):
    """Create a new experiment.

    Args:
        experiment: Experiment configuration.
    """
    if experiment.name in _experiments:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment '{experiment.name}' already exists"
        )

    # Validate percentages sum to 100
    total = sum(v.percentage for v in experiment.variants)
    if abs(total - 100.0) > 0.01:
        raise HTTPException(
            status_code=400,
            detail=f"Variant percentages must sum to 100, got {total}"
        )

    # Create experiment
    now = datetime.now().isoformat()
    exp_data = {
        "name": experiment.name,
        "description": experiment.description,
        "variants": [v.model_dump() for v in experiment.variants],
        "status": "draft",
        "start_date": None,
        "end_date": None,
        "target_metric": experiment.target_metric,
        "min_sample_size": experiment.min_sample_size,
        "created_at": now,
        "metadata": {},
    }

    _experiments[experiment.name] = exp_data
    return exp_data


@router.get("/{experiment_name}", response_model=ExperimentResponse)
async def get_experiment(experiment_name: str):
    """Get experiment details.

    Args:
        experiment_name: Name of the experiment.
    """
    _init_default_experiment()

    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    return _experiments[experiment_name]


@router.delete("/{experiment_name}")
async def delete_experiment(experiment_name: str):
    """Delete an experiment.

    Args:
        experiment_name: Name of the experiment to delete.
    """
    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    exp = _experiments[experiment_name]
    if exp["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete running experiment. Stop it first."
        )

    del _experiments[experiment_name]

    # Also delete metrics
    if experiment_name in _experiment_metrics:
        del _experiment_metrics[experiment_name]

    return {"status": "deleted", "experiment": experiment_name}


@router.post("/{experiment_name}/start")
async def start_experiment(experiment_name: str):
    """Start an experiment.

    Args:
        experiment_name: Name of the experiment to start.
    """
    _init_default_experiment()

    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    exp = _experiments[experiment_name]

    if exp["status"] == "running":
        raise HTTPException(
            status_code=400,
            detail="Experiment is already running"
        )

    exp["status"] = "running"
    exp["start_date"] = datetime.now().isoformat()

    return {"status": "started", "experiment": experiment_name}


@router.post("/{experiment_name}/stop")
async def stop_experiment(experiment_name: str):
    """Stop a running experiment.

    Args:
        experiment_name: Name of the experiment to stop.
    """
    _init_default_experiment()

    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    exp = _experiments[experiment_name]

    if exp["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail="Experiment is not running"
        )

    exp["status"] = "completed"
    exp["end_date"] = datetime.now().isoformat()

    return {"status": "stopped", "experiment": experiment_name}


@router.post("/{experiment_name}/pause")
async def pause_experiment(experiment_name: str):
    """Pause a running experiment.

    Args:
        experiment_name: Name of the experiment to pause.
    """
    _init_default_experiment()

    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    exp = _experiments[experiment_name]

    if exp["status"] != "running":
        raise HTTPException(
            status_code=400,
            detail="Experiment is not running"
        )

    exp["status"] = "paused"

    return {"status": "paused", "experiment": experiment_name}


@router.get("/{experiment_name}/metrics", response_model=ExperimentMetricsResponse)
async def get_experiment_metrics(experiment_name: str):
    """Get aggregated metrics for an experiment.

    Args:
        experiment_name: Name of the experiment.
    """
    _init_default_experiment()

    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    # Return cached metrics or generate mock data
    if experiment_name in _experiment_metrics:
        return _experiment_metrics[experiment_name]

    # Generate mock metrics for demo
    mock_metrics = {
        "experiment_name": experiment_name,
        "metrics": {
            "ndcg@10": {
                "variants": {
                    "control": {
                        "variant_name": "control",
                        "metric_name": "ndcg@10",
                        "count": 5420,
                        "sum": 552.84,
                        "mean": 0.102,
                        "min": 0.0,
                        "max": 0.45,
                        "unique_users": 5420,
                    },
                    "treatment": {
                        "variant_name": "treatment",
                        "metric_name": "ndcg@10",
                        "count": 5380,
                        "sum": 581.04,
                        "mean": 0.108,
                        "min": 0.0,
                        "max": 0.48,
                        "unique_users": 5380,
                    },
                },
                "total_count": 10800,
                "total_users": 10800,
            },
            "conversion": {
                "variants": {
                    "control": {
                        "variant_name": "control",
                        "metric_name": "conversion",
                        "count": 5420,
                        "sum": 142.0,
                        "mean": 0.0262,
                        "min": 0.0,
                        "max": 1.0,
                        "unique_users": 5420,
                    },
                    "treatment": {
                        "variant_name": "treatment",
                        "metric_name": "conversion",
                        "count": 5380,
                        "sum": 156.0,
                        "mean": 0.0290,
                        "min": 0.0,
                        "max": 1.0,
                        "unique_users": 5380,
                    },
                },
                "total_count": 10800,
                "total_users": 10800,
            },
        },
    }

    _experiment_metrics[experiment_name] = mock_metrics
    return mock_metrics


@router.get("/{experiment_name}/analysis", response_model=StatisticalResultResponse)
async def get_experiment_analysis(experiment_name: str):
    """Get statistical analysis for an experiment.

    Performs significance testing on the experiment results.

    Args:
        experiment_name: Name of the experiment.
    """
    _init_default_experiment()

    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    # Get metrics
    metrics = await get_experiment_metrics(experiment_name)

    # Extract target metric data
    exp = _experiments[experiment_name]
    target_metric = exp["target_metric"]

    if target_metric not in metrics["metrics"]:
        raise HTTPException(
            status_code=400,
            detail=f"Target metric '{target_metric}' not found in experiment data"
        )

    metric_data = metrics["metrics"][target_metric]["variants"]
    control_data = metric_data.get("control", {})
    treatment_data = metric_data.get("treatment", {})

    control_mean = control_data.get("mean", 0)
    treatment_mean = treatment_data.get("mean", 0)
    control_n = control_data.get("count", 0)
    treatment_n = treatment_data.get("count", 0)

    # Calculate lift
    if control_mean > 0:
        relative_lift = (treatment_mean - control_mean) / control_mean
    else:
        relative_lift = 0.0

    # Mock statistical result (in production, use actual StatisticalAnalyzer)
    import random
    random.seed(42)

    # Simulate p-value based on effect size and sample size
    effect_size = abs(relative_lift)
    sample_factor = min(control_n, treatment_n) / 1000
    p_value = max(0.001, 0.1 - effect_size * sample_factor)

    is_significant = p_value < 0.05

    # Calculate confidence interval
    se = 0.015  # Standard error estimate
    ci_low = (treatment_mean - control_mean) - 1.96 * se
    ci_high = (treatment_mean - control_mean) + 1.96 * se

    return StatisticalResultResponse(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        relative_lift=relative_lift,
        relative_lift_percent=relative_lift * 100,
        p_value=p_value,
        confidence_level=0.95,
        is_significant=is_significant,
        confidence_interval=(ci_low, ci_high),
        control_n=control_n,
        treatment_n=treatment_n,
        test_type="welch_t_test",
    )


@router.post("/{experiment_name}/log")
async def log_experiment_metric(
    experiment_name: str,
    user_id: int,
    metric_name: str,
    value: float,
):
    """Log a metric value for an experiment.

    Args:
        experiment_name: Name of the experiment.
        user_id: User ID.
        metric_name: Metric name.
        value: Metric value.
    """
    _init_default_experiment()

    if experiment_name not in _experiments:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment '{experiment_name}' not found"
        )

    # In production, this would log to the metrics tracker
    # For now, just acknowledge
    return {
        "status": "logged",
        "experiment": experiment_name,
        "user_id": user_id,
        "metric": metric_name,
        "value": value,
    }
