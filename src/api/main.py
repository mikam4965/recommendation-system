"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.config import get_api_settings
from src.api.routes import events, health, recommendations, stats
from src.api.services.recommendation_service import recommendation_service

api_settings = get_api_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler.

    Loads models on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting up API server...")
    logger.info("Loading recommendation models...")

    recommendation_service.load_models()

    if recommendation_service.is_loaded:
        models = recommendation_service.get_available_models()
        logger.info(f"Loaded models: {models}")
    else:
        logger.warning("No models loaded. Recommendations will be unavailable.")
        logger.warning("Run 'make train-baseline' to train models.")

    yield

    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title=api_settings.api_title,
    version=api_settings.api_version,
    description=api_settings.api_description,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(recommendations.router)
app.include_router(events.router)
app.include_router(stats.router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": api_settings.api_title,
        "version": api_settings.api_version,
        "docs": "/docs",
        "health": "/health",
    }
