"""API routes module.

Exports all route handlers for the FastAPI application.
"""

from src.api.routes.health import router as health_router
from src.api.routes.recommendations import router as recommendations_router
from src.api.routes.events import router as events_router
from src.api.routes.stats import router as stats_router
from src.api.routes.metrics import router as metrics_router
from src.api.routes.experiments import router as experiments_router

__all__ = [
    "health_router",
    "recommendations_router",
    "events_router",
    "stats_router",
    "metrics_router",
    "experiments_router",
]
