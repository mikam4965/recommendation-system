"""Recommendation service."""

from pathlib import Path

from loguru import logger

from src.config import settings
from src.models.base import BaseRecommender


class RecommendationService:
    """Service for managing and serving recommendations."""

    def __init__(self):
        """Initialize recommendation service."""
        self.models: dict[str, BaseRecommender] = {}
        self._loaded = False

    def load_models(self, models_dir: Path | None = None) -> None:
        """Load all trained models from directory.

        Args:
            models_dir: Directory containing model files. Default: settings.models_dir.
        """
        models_dir = models_dir or settings.models_dir

        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return

        model_files = list(models_dir.glob("*.joblib"))

        if not model_files:
            logger.warning(f"No model files found in {models_dir}")
            return

        for model_path in model_files:
            try:
                model = BaseRecommender.load(model_path)
                self.models[model.name] = model
                logger.info(f"Loaded model: {model.name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_path}: {e}")

        self._loaded = True
        logger.info(f"Loaded {len(self.models)} models")

    def get_recommendations(
        self,
        user_id: int,
        model_name: str,
        n_items: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """Get recommendations for a user.

        Args:
            user_id: User ID.
            model_name: Name of the model to use.
            n_items: Number of items to recommend.
            exclude_seen: Whether to exclude items user has seen.

        Returns:
            List of (item_id, score) tuples.

        Raises:
            ValueError: If model not found.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")

        model = self.models[model_name]
        return model.recommend(user_id, n_items=n_items, exclude_seen=exclude_seen)

    def get_available_models(self) -> list[str]:
        """Get list of available model names.

        Returns:
            List of model names.
        """
        return list(self.models.keys())

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded and len(self.models) > 0


# Global service instance
recommendation_service = RecommendationService()
