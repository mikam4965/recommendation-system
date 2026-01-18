"""Model training utilities with MLflow integration."""

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.evaluation.evaluator import Evaluator
from src.models.base import BaseRecommender
from src.training.mlflow_tracker import MLflowTracker


class Trainer:
    """Train and evaluate recommendation models with MLflow tracking."""

    def __init__(
        self,
        experiment_name: str = "recsys-ecommerce",
        tracking_uri: str | None = None,
        evaluator: Evaluator | None = None,
    ):
        """Initialize trainer.

        Args:
            experiment_name: MLflow experiment name.
            tracking_uri: MLflow tracking URI.
            evaluator: Evaluator instance. Creates default if None.
        """
        self.tracker = MLflowTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
        self.evaluator = evaluator or Evaluator()

    def train_and_evaluate(
        self,
        model: BaseRecommender,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        save_model: bool = True,
        model_path: Path | str | None = None,
        n_items: int = 20,
        max_users: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Train model and evaluate with MLflow tracking.

        Args:
            model: Model to train.
            train_data: Training DataFrame.
            test_data: Test DataFrame.
            val_data: Optional validation DataFrame.
            run_name: Name for MLflow run.
            tags: Optional tags for the run.
            save_model: Whether to save model artifact.
            model_path: Path to save model (optional).
            n_items: Number of items for evaluation.
            max_users: Maximum users to evaluate.
            show_progress: Whether to show progress bars.

        Returns:
            Dictionary with training results and metrics.
        """
        run_name = run_name or f"{model.name}_training"

        with self.tracker.run(run_name=run_name, tags=tags):
            # Log model parameters
            params = model.get_params()
            params["model_name"] = model.name
            params["train_size"] = len(train_data)
            params["test_size"] = len(test_data)
            if val_data is not None:
                params["val_size"] = len(val_data)

            self.tracker.log_params(params)

            # Train model
            logger.info(f"Training {model.name}...")
            model.fit(train_data)

            # Evaluate on validation set if provided
            val_results = None
            if val_data is not None:
                logger.info("Evaluating on validation set...")
                val_results = self.evaluator.evaluate(
                    model=model,
                    train_data=train_data,
                    test_data=val_data,
                    n_items=n_items,
                    max_users=max_users,
                    show_progress=show_progress,
                )

                # Log validation metrics with prefix
                val_metrics = {
                    f"val_{k}": v
                    for k, v in val_results.items()
                    if isinstance(v, (int, float)) and k != "n_users"
                }
                self.tracker.log_metrics(val_metrics)

            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_results = self.evaluator.evaluate(
                model=model,
                train_data=train_data,
                test_data=test_data,
                n_items=n_items,
                max_users=max_users,
                show_progress=show_progress,
            )

            # Log test metrics
            test_metrics = {
                f"test_{k}": v
                for k, v in test_results.items()
                if isinstance(v, (int, float)) and k != "n_users"
            }
            self.tracker.log_metrics(test_metrics)

            # Save model artifact
            if save_model:
                self.tracker.log_model(model, artifact_path="model")

                if model_path:
                    model.save(model_path)

            # Compile results
            results = {
                "model_name": model.name,
                "params": params,
                "test_metrics": test_results,
                "val_metrics": val_results,
                "run_id": self.tracker.run_id,
            }

            # Log results summary
            self.tracker.log_dict(results, "results.json")

            logger.info(f"Training complete for {model.name}")
            logger.info(f"Test NDCG@10: {test_results.get('ndcg@10', 'N/A'):.4f}")
            logger.info(f"Test Hit Rate: {test_results.get('hit_rate', 'N/A'):.4f}")

            return results

    def train_multiple_models(
        self,
        models: list[BaseRecommender],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        val_data: pd.DataFrame | None = None,
        parent_run_name: str = "model_comparison",
        n_items: int = 20,
        max_users: int | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Train and evaluate multiple models with nested MLflow runs.

        Args:
            models: List of models to train.
            train_data: Training DataFrame.
            test_data: Test DataFrame.
            val_data: Optional validation DataFrame.
            parent_run_name: Name for parent MLflow run.
            n_items: Number of items for evaluation.
            max_users: Maximum users to evaluate.
            show_progress: Whether to show progress bars.

        Returns:
            DataFrame with comparison of all models.
        """
        all_results = []

        with self.tracker.run(run_name=parent_run_name):
            self.tracker.log_params({
                "n_models": len(models),
                "train_size": len(train_data),
                "test_size": len(test_data),
            })

            for model in models:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training: {model.name}")
                logger.info(f"{'='*50}")

                # Train with nested run
                with self.tracker.run(run_name=model.name, nested=True):
                    # Log model parameters
                    params = model.get_params()
                    params["model_name"] = model.name
                    self.tracker.log_params(params)

                    # Train
                    model.fit(train_data)

                    # Evaluate on test
                    test_results = self.evaluator.evaluate(
                        model=model,
                        train_data=train_data,
                        test_data=test_data,
                        n_items=n_items,
                        max_users=max_users,
                        show_progress=show_progress,
                    )

                    # Log metrics
                    metrics = {
                        k: v
                        for k, v in test_results.items()
                        if isinstance(v, (int, float)) and k != "n_users"
                    }
                    self.tracker.log_metrics(metrics)

                    # Save model
                    self.tracker.log_model(model, artifact_path="model")

                    all_results.append(test_results)

            # Create comparison DataFrame
            results_df = pd.DataFrame(all_results)

            # Sort by NDCG@10
            if "ndcg@10" in results_df.columns:
                results_df = results_df.sort_values("ndcg@10", ascending=False)

            # Log comparison table
            self.tracker.log_dict(
                results_df.to_dict(orient="records"),
                "model_comparison.json",
            )

            # Log best model info
            if not results_df.empty:
                best_model = results_df.iloc[0]["model"]
                best_ndcg = results_df.iloc[0].get("ndcg@10", 0)
                self.tracker.set_tag("best_model", best_model)
                self.tracker.log_metric("best_ndcg@10", best_ndcg)

        return results_df

    def cross_validate(
        self,
        model_class: type[BaseRecommender],
        model_kwargs: dict[str, Any],
        data: pd.DataFrame,
        n_folds: int = 5,
        run_name: str | None = None,
        n_items: int = 20,
        max_users: int | None = None,
    ) -> dict[str, Any]:
        """Perform time-based cross-validation.

        Args:
            model_class: Model class to instantiate.
            model_kwargs: Keyword arguments for model.
            data: Full dataset.
            n_folds: Number of folds.
            run_name: Name for MLflow run.
            n_items: Number of items for evaluation.
            max_users: Maximum users per fold.

        Returns:
            Dictionary with CV results.
        """
        run_name = run_name or f"{model_class.name}_cv"

        # Sort by timestamp
        data = data.sort_values("timestamp").reset_index(drop=True)
        n_samples = len(data)

        fold_results = []

        with self.tracker.run(run_name=run_name):
            self.tracker.log_params({
                "model_class": model_class.name,
                "n_folds": n_folds,
                "total_samples": n_samples,
                **model_kwargs,
            })

            for fold in range(n_folds):
                logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")

                # Time-based split: use first (fold+1)/n_folds for train, next portion for test
                train_end = int(n_samples * (fold + 1) / (n_folds + 1))
                test_end = int(n_samples * (fold + 2) / (n_folds + 1))

                train_data = data.iloc[:train_end]
                test_data = data.iloc[train_end:test_end]

                logger.info(f"Train: {len(train_data):,}, Test: {len(test_data):,}")

                # Create and train model
                model = model_class(**model_kwargs)
                model.fit(train_data)

                # Evaluate
                results = self.evaluator.evaluate(
                    model=model,
                    train_data=train_data,
                    test_data=test_data,
                    n_items=n_items,
                    max_users=max_users,
                    show_progress=False,
                )

                results["fold"] = fold + 1
                fold_results.append(results)

                # Log fold metrics
                fold_metrics = {
                    f"fold_{fold + 1}_{k}": v
                    for k, v in results.items()
                    if isinstance(v, (int, float)) and k not in ["n_users", "fold"]
                }
                self.tracker.log_metrics(fold_metrics)

            # Aggregate results
            results_df = pd.DataFrame(fold_results)
            metric_cols = [c for c in results_df.columns if c not in ["model", "n_users", "fold"]]

            mean_metrics = results_df[metric_cols].mean().to_dict()
            std_metrics = results_df[metric_cols].std().to_dict()

            # Log aggregated metrics
            self.tracker.log_metrics({f"mean_{k}": v for k, v in mean_metrics.items()})
            self.tracker.log_metrics({f"std_{k}": v for k, v in std_metrics.items()})

            cv_results = {
                "fold_results": fold_results,
                "mean_metrics": mean_metrics,
                "std_metrics": std_metrics,
                "n_folds": n_folds,
            }

            self.tracker.log_dict(cv_results, "cv_results.json")

            logger.info(f"\nCV Results (mean ± std):")
            for metric in ["ndcg@10", "hit_rate", "mrr"]:
                if metric in mean_metrics:
                    logger.info(
                        f"  {metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}"
                    )

            return cv_results
