"""Health check endpoint."""

from fastapi import APIRouter

from src.api.config import get_api_settings
from src.api.schemas.stats import HealthResponse
from src.api.services.recommendation_service import recommendation_service

router = APIRouter(tags=["health"])
api_settings = get_api_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns service status and basic information.
    """
    return HealthResponse(
        status="healthy" if recommendation_service.is_loaded else "degraded",
        models_loaded=len(recommendation_service.models),
        version=api_settings.api_version,
    )
