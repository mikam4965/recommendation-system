"""Recommendation endpoints."""

from fastapi import APIRouter, HTTPException, Query

from src.api.config import get_api_settings
from src.api.schemas.recommendations import (
    AvailableModelsResponse,
    RecommendationItem,
    RecommendationResponse,
)
from src.api.services.recommendation_service import recommendation_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
api_settings = get_api_settings()


@router.get("/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int,
    n: int = Query(
        default=api_settings.default_n_items,
        ge=1,
        le=api_settings.max_n_items,
        description="Number of items to recommend",
    ),
    model: str = Query(
        default=api_settings.default_model,
        description="Model to use for recommendations",
    ),
    exclude_seen: bool = Query(
        default=True,
        description="Exclude items the user has already seen",
    ),
) -> RecommendationResponse:
    """Get personalized recommendations for a user.

    Args:
        user_id: User ID to get recommendations for.
        n: Number of items to recommend (1-100).
        model: Model name to use (popular, random, user_cf, item_cf).
        exclude_seen: Whether to exclude items user has interacted with.

    Returns:
        Recommendation response with list of items.
    """
    # Check if service is ready
    if not recommendation_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run 'make train-baseline' first.",
        )

    # Validate model
    available = recommendation_service.get_available_models()
    if model not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not available. Choose from: {available}",
        )

    try:
        recs = recommendation_service.get_recommendations(
            user_id=user_id,
            model_name=model,
            n_items=n,
            exclude_seen=exclude_seen,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Format response
    recommendations = [
        RecommendationItem(item_id=item_id, score=score, rank=i + 1)
        for i, (item_id, score) in enumerate(recs)
    ]

    return RecommendationResponse(
        user_id=user_id,
        model=model,
        n_items=n,
        recommendations=recommendations,
    )


@router.get("/", response_model=AvailableModelsResponse)
async def get_available_models() -> AvailableModelsResponse:
    """Get list of available recommendation models.

    Returns:
        List of available model names and default model.
    """
    models = recommendation_service.get_available_models()
    return AvailableModelsResponse(
        models=models,
        default=api_settings.default_model,
    )
