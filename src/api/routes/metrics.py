"""API routes for dashboard metrics.

Provides endpoints for model metrics, system metrics, and funnel data.
"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/metrics", tags=["metrics"])


# Response models
class ModelMetricsResponse(BaseModel):
    """Model metrics response."""

    model: str
    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float
    hit_rate: float
    mrr: float
    coverage: float
    diversity: float


class FunnelStageResponse(BaseModel):
    """Funnel stage data."""

    name: str
    count: int
    percentage: float


class ConversionRatesResponse(BaseModel):
    """Conversion rates data."""

    view_to_cart: float
    cart_to_transaction: float
    overall: float


class FunnelDataResponse(BaseModel):
    """Funnel data response."""

    stages: list[FunnelStageResponse]
    conversion_rates: ConversionRatesResponse


class SystemMetricsResponse(BaseModel):
    """System metrics response."""

    total_users: int
    total_items: int
    total_events: int
    active_experiments: int
    recommendations_served: int
    avg_latency_ms: float


# In-memory storage for demo (replace with actual database/cache)
_model_metrics_cache: list[dict[str, Any]] = []
_system_metrics_cache: dict[str, Any] = {}
_funnel_data_cache: dict[str, Any] = {}


def get_default_model_metrics() -> list[dict[str, Any]]:
    """Get default model metrics."""
    return [
        {
            "model": "Popular",
            "precision_at_5": 0.042, "precision_at_10": 0.045, "precision_at_20": 0.041,
            "recall_at_5": 0.028, "recall_at_10": 0.032, "recall_at_20": 0.038,
            "ndcg_at_5": 0.048, "ndcg_at_10": 0.051, "ndcg_at_20": 0.054,
            "hit_rate": 0.28, "mrr": 0.15, "coverage": 0.02, "diversity": 0.12,
        },
        {
            "model": "ALS",
            "precision_at_5": 0.068, "precision_at_10": 0.072, "precision_at_20": 0.065,
            "recall_at_5": 0.048, "recall_at_10": 0.054, "recall_at_20": 0.062,
            "ndcg_at_5": 0.082, "ndcg_at_10": 0.089, "ndcg_at_20": 0.094,
            "hit_rate": 0.42, "mrr": 0.28, "coverage": 0.18, "diversity": 0.45,
        },
        {
            "model": "BPR",
            "precision_at_5": 0.065, "precision_at_10": 0.068, "precision_at_20": 0.062,
            "recall_at_5": 0.045, "recall_at_10": 0.051, "recall_at_20": 0.058,
            "ndcg_at_5": 0.078, "ndcg_at_10": 0.083, "ndcg_at_20": 0.088,
            "hit_rate": 0.40, "mrr": 0.26, "coverage": 0.15, "diversity": 0.42,
        },
        {
            "model": "NCF",
            "precision_at_5": 0.071, "precision_at_10": 0.075, "precision_at_20": 0.068,
            "recall_at_5": 0.052, "recall_at_10": 0.057, "recall_at_20": 0.065,
            "ndcg_at_5": 0.088, "ndcg_at_10": 0.094, "ndcg_at_20": 0.098,
            "hit_rate": 0.45, "mrr": 0.31, "coverage": 0.22, "diversity": 0.48,
        },
        {
            "model": "GRU4Rec",
            "precision_at_5": 0.074, "precision_at_10": 0.078, "precision_at_20": 0.071,
            "recall_at_5": 0.055, "recall_at_10": 0.061, "recall_at_20": 0.068,
            "ndcg_at_5": 0.091, "ndcg_at_10": 0.097, "ndcg_at_20": 0.102,
            "hit_rate": 0.48, "mrr": 0.33, "coverage": 0.25, "diversity": 0.52,
        },
        {
            "model": "SASRec",
            "precision_at_5": 0.076, "precision_at_10": 0.081, "precision_at_20": 0.074,
            "recall_at_5": 0.058, "recall_at_10": 0.064, "recall_at_20": 0.071,
            "ndcg_at_5": 0.094, "ndcg_at_10": 0.101, "ndcg_at_20": 0.106,
            "hit_rate": 0.51, "mrr": 0.35, "coverage": 0.28, "diversity": 0.55,
        },
        {
            "model": "Two-Tower",
            "precision_at_5": 0.073, "precision_at_10": 0.077, "precision_at_20": 0.070,
            "recall_at_5": 0.054, "recall_at_10": 0.059, "recall_at_20": 0.066,
            "ndcg_at_5": 0.089, "ndcg_at_10": 0.095, "ndcg_at_20": 0.100,
            "hit_rate": 0.47, "mrr": 0.32, "coverage": 0.35, "diversity": 0.58,
        },
        {
            "model": "Hybrid",
            "precision_at_5": 0.078, "precision_at_10": 0.084, "precision_at_20": 0.077,
            "recall_at_5": 0.059, "recall_at_10": 0.066, "recall_at_20": 0.074,
            "ndcg_at_5": 0.096, "ndcg_at_10": 0.105, "ndcg_at_20": 0.110,
            "hit_rate": 0.54, "mrr": 0.38, "coverage": 0.32, "diversity": 0.58,
        },
    ]


def get_default_system_metrics() -> dict[str, Any]:
    """Get default system metrics."""
    return {
        "total_users": 1407580,
        "total_items": 235061,
        "total_events": 2756101,
        "active_experiments": 1,
        "recommendations_served": 125430,
        "avg_latency_ms": 45.2,
    }


def get_default_funnel_data() -> dict[str, Any]:
    """Get default funnel data."""
    return {
        "stages": [
            {"name": "View", "count": 2664312, "percentage": 100.0},
            {"name": "Add to Cart", "count": 69332, "percentage": 2.60},
            {"name": "Transaction", "count": 22457, "percentage": 0.84},
        ],
        "conversion_rates": {
            "view_to_cart": 0.026,
            "cart_to_transaction": 0.324,
            "overall": 0.0084,
        },
    }


@router.get("/models", response_model=list[ModelMetricsResponse])
async def get_model_metrics():
    """Get metrics for all recommendation models.

    Returns comparison metrics (Precision, Recall, NDCG, etc.) for each model.
    """
    global _model_metrics_cache

    if not _model_metrics_cache:
        _model_metrics_cache = get_default_model_metrics()

    return _model_metrics_cache


@router.post("/models/{model_name}")
async def update_model_metrics(model_name: str, metrics: ModelMetricsResponse):
    """Update metrics for a specific model.

    Used after model evaluation to update stored metrics.
    """
    global _model_metrics_cache

    if not _model_metrics_cache:
        _model_metrics_cache = get_default_model_metrics()

    # Find and update model
    for i, m in enumerate(_model_metrics_cache):
        if m["model"] == model_name:
            _model_metrics_cache[i] = metrics.model_dump()
            return {"status": "updated", "model": model_name}

    # Add new model
    _model_metrics_cache.append(metrics.model_dump())
    return {"status": "created", "model": model_name}


@router.get("/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system-level metrics.

    Returns aggregate statistics about users, items, events, and performance.
    """
    global _system_metrics_cache

    if not _system_metrics_cache:
        _system_metrics_cache = get_default_system_metrics()

    return _system_metrics_cache


@router.post("/system")
async def update_system_metrics(metrics: SystemMetricsResponse):
    """Update system metrics.

    Called periodically to refresh system statistics.
    """
    global _system_metrics_cache
    _system_metrics_cache = metrics.model_dump()
    return {"status": "updated"}


@router.get("/funnel", response_model=FunnelDataResponse)
async def get_funnel_data():
    """Get conversion funnel data.

    Returns event counts and conversion rates for each funnel stage.
    """
    global _funnel_data_cache

    if not _funnel_data_cache:
        _funnel_data_cache = get_default_funnel_data()

    return _funnel_data_cache


@router.post("/funnel")
async def update_funnel_data(funnel: FunnelDataResponse):
    """Update funnel data.

    Called to refresh funnel statistics.
    """
    global _funnel_data_cache
    _funnel_data_cache = funnel.model_dump()
    return {"status": "updated"}
