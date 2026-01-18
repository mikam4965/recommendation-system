"""Hyperparameter tuning with Optuna."""

from typing import Any, Callable, Literal

import optuna
import pandas as pd
from loguru import logger
from optuna.samplers import TPESampler

from src.evaluation.evaluator import Evaluator
from src.models.base import BaseRecommender
from src.models.collaborative.als import ALSRecommender
from src.models.collaborative.bpr import BPRRecommender
from src.models.collaborative.ncf import NCFRecommender
from src.models.content.item2vec import Item2VecRecommender
from src.training.mlflow_tracker import MLflowTracker


class HyperparameterTuner:
    """Hyperparameter tuning for recommendation models using Optuna."""

    def __init__(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        metric: str = "ndcg@10",
        direction: Literal["maximize", "minimize"] = "maximize",
        n_items: int = 20,
        max_users: int | None = 5000,
        mlflow_tracker: MLflowTracker | None = None,
    ):
        """Initialize tuner.

        Args:
            train_data: Training DataFrame.
            val_data: Validation DataFrame.
            metric: Metric to optimize.
            direction: Optimization direction.
            n_items: Number of items for evaluation.
            max_users: Maximum users for evaluation (for speed).
            mlflow_tracker: Optional MLflow tracker for logging.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.metric = metric
        self.direction = direction
        self.n_items = n_items
        self.max_users = max_users
        self.mlflow_tracker = mlflow_tracker
        self.evaluator = Evaluator()

    def _create_objective(
        self,
        model_class: type[BaseRecommender],
        param_space: Callable[[optuna.Trial], dict[str, Any]],
    ) -> Callable[[optuna.Trial], float]:
        """Create objective function for Optuna.

        Args:
            model_class: Model class to tune.
            param_space: Function that takes a trial and returns parameters.

        Returns:
            Objective function.
        """
        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = param_space(trial)

            try:
                # Create and train model
                model = model_class(**params)
                model.fit(self.train_data)

                # Evaluate
                results = self.evaluator.evaluate(
                    model=model,
                    train_data=self.train_data,
                    test_data=self.val_data,
                    n_items=self.n_items,
                    max_users=self.max_users,
                    show_progress=False,
                )

                score = results.get(self.metric, 0.0)

                # Log to MLflow if available
                if self.mlflow_tracker and self.mlflow_tracker.active_run:
                    self.mlflow_tracker.log_metrics({
                        f"trial_{trial.number}_{self.metric}": score,
                    })

                return score

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                # Return worst possible score
                return 0.0 if self.direction == "maximize" else float("inf")

        return objective

    def tune(
        self,
        model_class: type[BaseRecommender],
        param_space: Callable[[optuna.Trial], dict[str, Any]],
        n_trials: int = 50,
        timeout: int | None = None,
        study_name: str | None = None,
        random_state: int = 42,
    ) -> tuple[dict[str, Any], float, optuna.Study]:
        """Run hyperparameter tuning.

        Args:
            model_class: Model class to tune.
            param_space: Parameter space function.
            n_trials: Number of trials.
            timeout: Optional timeout in seconds.
            study_name: Name for the Optuna study.
            random_state: Random seed.

        Returns:
            Tuple of (best_params, best_score, study).
        """
        study_name = study_name or f"{model_class.name}_tuning"

        logger.info(f"Starting hyperparameter tuning for {model_class.name}")
        logger.info(f"Optimizing {self.metric} ({self.direction})")
        logger.info(f"Max trials: {n_trials}")

        # Create study
        sampler = TPESampler(seed=random_state)
        study = optuna.create_study(
            study_name=study_name,
            direction=self.direction,
            sampler=sampler,
        )

        # Create objective
        objective = self._create_objective(model_class, param_space)

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best {self.metric}: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params, best_score, study

    def tune_als(
        self,
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> tuple[dict[str, Any], float, optuna.Study]:
        """Tune ALS model.

        Search space:
        - factors: [32, 256]
        - regularization: [0.001, 0.1]
        - iterations: [10, 30]

        Args:
            n_trials: Number of trials.
            timeout: Optional timeout.

        Returns:
            Tuple of (best_params, best_score, study).
        """
        def param_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                "factors": trial.suggest_int("factors", 32, 256, step=32),
                "regularization": trial.suggest_float("regularization", 0.001, 0.1, log=True),
                "iterations": trial.suggest_int("iterations", 10, 30),
            }

        return self.tune(
            model_class=ALSRecommender,
            param_space=param_space,
            n_trials=n_trials,
            timeout=timeout,
            study_name="als_tuning",
        )

    def tune_bpr(
        self,
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> tuple[dict[str, Any], float, optuna.Study]:
        """Tune BPR model.

        Search space:
        - factors: [32, 256]
        - learning_rate: [0.001, 0.1]
        - regularization: [0.0001, 0.01]
        - iterations: [50, 200]

        Args:
            n_trials: Number of trials.
            timeout: Optional timeout.

        Returns:
            Tuple of (best_params, best_score, study).
        """
        def param_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                "factors": trial.suggest_int("factors", 32, 256, step=32),
                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                "regularization": trial.suggest_float("regularization", 0.0001, 0.01, log=True),
                "iterations": trial.suggest_int("iterations", 50, 200, step=25),
            }

        return self.tune(
            model_class=BPRRecommender,
            param_space=param_space,
            n_trials=n_trials,
            timeout=timeout,
            study_name="bpr_tuning",
        )

    def tune_item2vec(
        self,
        n_trials: int = 50,
        timeout: int | None = None,
    ) -> tuple[dict[str, Any], float, optuna.Study]:
        """Tune Item2Vec model.

        Search space:
        - embedding_dim: [32, 128]
        - window: [3, 10]
        - min_count: [3, 10]
        - epochs: [5, 20]

        Args:
            n_trials: Number of trials.
            timeout: Optional timeout.

        Returns:
            Tuple of (best_params, best_score, study).
        """
        def param_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                "embedding_dim": trial.suggest_int("embedding_dim", 32, 128, step=32),
                "window": trial.suggest_int("window", 3, 10),
                "min_count": trial.suggest_int("min_count", 3, 10),
                "epochs": trial.suggest_int("epochs", 5, 20),
                "negative": trial.suggest_int("negative", 3, 10),
            }

        return self.tune(
            model_class=Item2VecRecommender,
            param_space=param_space,
            n_trials=n_trials,
            timeout=timeout,
            study_name="item2vec_tuning",
        )

    def tune_ncf(
        self,
        n_trials: int = 30,
        timeout: int | None = None,
    ) -> tuple[dict[str, Any], float, optuna.Study]:
        """Tune NCF model.

        Search space:
        - model_type: [gmf, mlp, neumf]
        - embedding_dim: [16, 64]
        - learning_rate: [0.0001, 0.01]
        - dropout: [0.1, 0.5]
        - epochs: [10, 30]

        Args:
            n_trials: Number of trials.
            timeout: Optional timeout.

        Returns:
            Tuple of (best_params, best_score, study).
        """
        def param_space(trial: optuna.Trial) -> dict[str, Any]:
            model_type = trial.suggest_categorical("model_type", ["gmf", "mlp", "neumf"])

            # MLP layers only matter for mlp and neumf
            mlp_layers = None
            if model_type in ["mlp", "neumf"]:
                n_layers = trial.suggest_int("n_layers", 2, 4)
                first_layer = trial.suggest_int("first_layer_size", 32, 128, step=32)
                mlp_layers = [first_layer // (2 ** i) for i in range(n_layers)]
                mlp_layers = [max(8, size) for size in mlp_layers]  # Minimum 8

            return {
                "model_type": model_type,
                "embedding_dim": trial.suggest_int("embedding_dim", 16, 64, step=16),
                "mlp_layers": mlp_layers,
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01, log=True),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "epochs": trial.suggest_int("epochs", 10, 30),
                "negative_samples": trial.suggest_int("negative_samples", 2, 6),
            }

        return self.tune(
            model_class=NCFRecommender,
            param_space=param_space,
            n_trials=n_trials,
            timeout=timeout,
            study_name="ncf_tuning",
        )

    def tune_all(
        self,
        n_trials_per_model: int = 30,
        timeout_per_model: int | None = None,
    ) -> dict[str, tuple[dict[str, Any], float]]:
        """Tune all models.

        Args:
            n_trials_per_model: Trials per model.
            timeout_per_model: Timeout per model.

        Returns:
            Dictionary mapping model name to (best_params, best_score).
        """
        results = {}

        logger.info("Starting full hyperparameter search")

        # Tune ALS
        logger.info("\n" + "=" * 50)
        logger.info("Tuning ALS")
        logger.info("=" * 50)
        als_params, als_score, _ = self.tune_als(
            n_trials=n_trials_per_model,
            timeout=timeout_per_model,
        )
        results["als"] = (als_params, als_score)

        # Tune BPR
        logger.info("\n" + "=" * 50)
        logger.info("Tuning BPR")
        logger.info("=" * 50)
        bpr_params, bpr_score, _ = self.tune_bpr(
            n_trials=n_trials_per_model,
            timeout=timeout_per_model,
        )
        results["bpr"] = (bpr_params, bpr_score)

        # Tune Item2Vec
        logger.info("\n" + "=" * 50)
        logger.info("Tuning Item2Vec")
        logger.info("=" * 50)
        i2v_params, i2v_score, _ = self.tune_item2vec(
            n_trials=n_trials_per_model,
            timeout=timeout_per_model,
        )
        results["item2vec"] = (i2v_params, i2v_score)

        # Tune NCF (fewer trials due to longer training time)
        logger.info("\n" + "=" * 50)
        logger.info("Tuning NCF")
        logger.info("=" * 50)
        ncf_params, ncf_score, _ = self.tune_ncf(
            n_trials=max(10, n_trials_per_model // 2),
            timeout=timeout_per_model,
        )
        results["ncf"] = (ncf_params, ncf_score)

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("TUNING SUMMARY")
        logger.info("=" * 50)

        for model_name, (params, score) in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Best {self.metric}: {score:.4f}")
            logger.info(f"  Best params: {params}")

        return results


def get_best_params_from_study(study: optuna.Study) -> dict[str, Any]:
    """Extract best parameters from an Optuna study.

    Args:
        study: Completed Optuna study.

    Returns:
        Dictionary of best parameters.
    """
    return study.best_params


def visualize_study(study: optuna.Study, output_dir: str | None = None) -> None:
    """Generate visualization plots for an Optuna study.

    Args:
        study: Completed Optuna study.
        output_dir: Directory to save plots.
    """
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )

        # Optimization history
        fig = plot_optimization_history(study)
        if output_dir:
            fig.write_html(f"{output_dir}/optimization_history.html")

        # Parameter importances
        fig = plot_param_importances(study)
        if output_dir:
            fig.write_html(f"{output_dir}/param_importances.html")

        # Parallel coordinate
        fig = plot_parallel_coordinate(study)
        if output_dir:
            fig.write_html(f"{output_dir}/parallel_coordinate.html")

        logger.info(f"Saved visualizations to {output_dir}")

    except ImportError:
        logger.warning("Plotly not installed, skipping visualizations")
