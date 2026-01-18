#!/usr/bin/env python3
"""Train and evaluate Version 1.0 recommendation models.

This script trains all V1.0 models:
- ALS (Alternating Least Squares)
- BPR (Bayesian Personalized Ranking)
- Item2Vec
- NCF (Neural Collaborative Filtering)
- Funnel-Aware Hybrid (Scientific Novelty #1)

All experiments are tracked in MLflow.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from loguru import logger

from src.config import settings
from src.data.features.user_features import UserFeatureExtractor
from src.data.loaders.retailrocket import RetailRocketLoader
from src.data.processors.session_builder import SessionBuilder
from src.data.processors.splitter import TimeBasedSplitter
from src.evaluation.evaluator import Evaluator
from src.models.baselines.popular import PopularItemsRecommender
from src.models.collaborative.als import ALSRecommender
from src.models.collaborative.bpr import BPRRecommender
from src.models.collaborative.ncf import NCFRecommender
from src.models.content.item2vec import Item2VecRecommender
from src.models.hybrid.funnel_aware import FunnelAwareHybridRecommender
from src.training.mlflow_tracker import MLflowTracker
from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train V1.0 recommendation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--max-users",
        type=int,
        default=10000,
        help="Maximum users to evaluate (for speed)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=20,
        help="Number of items to recommend per user",
    )
    parser.add_argument(
        "--skip-ncf",
        action="store_true",
        help="Skip NCF training (slow on CPU)",
    )
    parser.add_argument(
        "--ncf-epochs",
        type=int,
        default=10,
        help="Number of epochs for NCF (reduce for faster training)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="recsys-v1",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "processed"),
        help="Directory to save results",
    )

    return parser.parse_args()


def main():
    """Train and evaluate all V1.0 models."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("VERSION 1.0 MODEL TRAINING")
    logger.info("=" * 70)

    # Initialize MLflow tracker
    tracker = MLflowTracker(experiment_name=args.experiment_name)

    with tracker.run(run_name="v1_full_training"):
        # ============================================================
        # STEP 1: Load and prepare data
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Loading and preparing data")
        logger.info("=" * 70)

        loader = RetailRocketLoader()
        events = loader.load_events()

        # Build sessions for Item2Vec
        logger.info("\nBuilding sessions...")
        session_builder = SessionBuilder(timeout_minutes=30)
        events = session_builder.build_sessions(events)

        # Log data stats
        tracker.log_params({
            "total_events": len(events),
            "unique_users": events["visitor_id"].nunique(),
            "unique_items": events["item_id"].nunique(),
            "unique_sessions": events["session_id"].nunique(),
        })

        # ============================================================
        # STEP 2: Split data
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Splitting data (70/15/15)")
        logger.info("=" * 70)

        splitter = TimeBasedSplitter(
            train_ratio=settings.train_ratio,
            val_ratio=settings.val_ratio,
            test_ratio=settings.test_ratio,
        )
        train, val, test = splitter.split(events)

        # Get split stats
        split_stats = splitter.get_split_stats(train, val, test)
        logger.info(f"Cold-start users in test: {split_stats['test_cold_start_users']:,}")
        logger.info(f"Cold-start items in test: {split_stats['test_cold_start_items']:,}")

        tracker.log_params({
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "test_cold_start_users": split_stats["test_cold_start_users"],
            "test_cold_start_items": split_stats["test_cold_start_items"],
        })

        # ============================================================
        # STEP 3: Extract user features for funnel stage
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Extracting user features")
        logger.info("=" * 70)

        user_feature_extractor = UserFeatureExtractor()
        user_feature_extractor.fit(train)

        # ============================================================
        # STEP 4: Train individual models
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Training individual models")
        logger.info("=" * 70)

        evaluator = Evaluator(k_values=[5, 10, 20])
        all_results = []

        # --- Baseline: Popular ---
        logger.info("\n--- Training Popular baseline ---")
        popular_model = PopularItemsRecommender()
        popular_model.fit(train)
        popular_results = evaluator.evaluate(
            popular_model, train, test,
            n_items=args.n_items, max_users=args.max_users
        )
        all_results.append(popular_results)
        logger.info(f"Popular NDCG@10: {popular_results.get('ndcg@10', 0):.4f}")

        # --- ALS ---
        logger.info("\n--- Training ALS ---")
        with tracker.run(run_name="als", nested=True):
            als_model = ALSRecommender(
                factors=64,
                regularization=0.01,
                iterations=15,
            )
            tracker.log_params(als_model.get_params())
            als_model.fit(train)

            als_results = evaluator.evaluate(
                als_model, train, test,
                n_items=args.n_items, max_users=args.max_users
            )
            all_results.append(als_results)

            tracker.log_metrics({
                k: v for k, v in als_results.items()
                if isinstance(v, (int, float)) and k != "n_users"
            })
            logger.info(f"ALS NDCG@10: {als_results.get('ndcg@10', 0):.4f}")

        # --- BPR ---
        logger.info("\n--- Training BPR ---")
        with tracker.run(run_name="bpr", nested=True):
            bpr_model = BPRRecommender(
                factors=64,
                learning_rate=0.01,
                regularization=0.001,
                iterations=100,
            )
            tracker.log_params(bpr_model.get_params())
            bpr_model.fit(train)

            bpr_results = evaluator.evaluate(
                bpr_model, train, test,
                n_items=args.n_items, max_users=args.max_users
            )
            all_results.append(bpr_results)

            tracker.log_metrics({
                k: v for k, v in bpr_results.items()
                if isinstance(v, (int, float)) and k != "n_users"
            })
            logger.info(f"BPR NDCG@10: {bpr_results.get('ndcg@10', 0):.4f}")

        # --- Item2Vec ---
        logger.info("\n--- Training Item2Vec ---")
        with tracker.run(run_name="item2vec", nested=True):
            item2vec_model = Item2VecRecommender(
                embedding_dim=64,
                window=5,
                min_count=5,
                epochs=10,
            )
            tracker.log_params(item2vec_model.get_params())
            item2vec_model.fit(train)

            item2vec_results = evaluator.evaluate(
                item2vec_model, train, test,
                n_items=args.n_items, max_users=args.max_users
            )
            all_results.append(item2vec_results)

            tracker.log_metrics({
                k: v for k, v in item2vec_results.items()
                if isinstance(v, (int, float)) and k != "n_users"
            })
            logger.info(f"Item2Vec NDCG@10: {item2vec_results.get('ndcg@10', 0):.4f}")

        # --- NCF (optional) ---
        ncf_model = None
        if not args.skip_ncf:
            logger.info("\n--- Training NCF ---")
            with tracker.run(run_name="ncf", nested=True):
                ncf_model = NCFRecommender(
                    model_type="neumf",
                    embedding_dim=32,
                    mlp_layers=[64, 32, 16],
                    epochs=args.ncf_epochs,
                    batch_size=1024,
                    negative_samples=4,
                )
                tracker.log_params(ncf_model.get_params())
                ncf_model.fit(train)

                ncf_results = evaluator.evaluate(
                    ncf_model, train, test,
                    n_items=args.n_items, max_users=args.max_users
                )
                all_results.append(ncf_results)

                tracker.log_metrics({
                    k: v for k, v in ncf_results.items()
                    if isinstance(v, (int, float)) and k != "n_users"
                })
                logger.info(f"NCF NDCG@10: {ncf_results.get('ndcg@10', 0):.4f}")
        else:
            logger.info("\n--- Skipping NCF (--skip-ncf flag set) ---")

        # ============================================================
        # STEP 5: Train Funnel-Aware Hybrid (Scientific Novelty #1)
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Training Funnel-Aware Hybrid (Scientific Novelty #1)")
        logger.info("=" * 70)

        with tracker.run(run_name="funnel_aware_hybrid", nested=True):
            # Use best CF model and Item2Vec as content
            cf_model = bpr_model if bpr_results.get("ndcg@10", 0) > als_results.get("ndcg@10", 0) else als_model
            logger.info(f"Using {cf_model.name} as CF component")

            hybrid_model = FunnelAwareHybridRecommender(
                popular_model=popular_model,
                content_model=item2vec_model,
                cf_model=cf_model,
                session_model=None,  # Will add in v2.0
                score_normalization="minmax",
                user_feature_extractor=user_feature_extractor,
            )
            hybrid_model.fit(train)

            tracker.log_params(hybrid_model.get_params())

            hybrid_results = evaluator.evaluate(
                hybrid_model, train, test,
                n_items=args.n_items, max_users=args.max_users
            )
            all_results.append(hybrid_results)

            tracker.log_metrics({
                k: v for k, v in hybrid_results.items()
                if isinstance(v, (int, float)) and k != "n_users"
            })
            logger.info(f"Funnel-Aware Hybrid NDCG@10: {hybrid_results.get('ndcg@10', 0):.4f}")

        # ============================================================
        # STEP 6: Results comparison
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Results comparison")
        logger.info("=" * 70)

        results_df = pd.DataFrame(all_results)

        # Sort by NDCG@10
        results_df = results_df.sort_values("ndcg@10", ascending=False)

        # Print results table
        evaluator.print_results(results_df)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "v1_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")

        # Log to MLflow
        tracker.log_dict(
            results_df.to_dict(orient="records"),
            "model_comparison.json"
        )

        # Log best model
        best_model = results_df.iloc[0]["model"]
        best_ndcg = results_df.iloc[0]["ndcg@10"]
        tracker.set_tag("best_model", best_model)
        tracker.log_metric("best_ndcg@10", best_ndcg)

        # ============================================================
        # Summary
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 70)

        logger.info(f"\nBest model: {best_model}")
        logger.info(f"Best NDCG@10: {best_ndcg:.4f}")

        # Calculate improvement of hybrid over best single model
        single_model_results = [r for r in all_results if r["model"] != "funnel_aware_hybrid"]
        best_single = max(single_model_results, key=lambda x: x.get("ndcg@10", 0))
        hybrid_ndcg = hybrid_results.get("ndcg@10", 0)
        best_single_ndcg = best_single.get("ndcg@10", 0)

        if best_single_ndcg > 0:
            improvement = (hybrid_ndcg - best_single_ndcg) / best_single_ndcg * 100
            logger.info(
                f"\nFunnel-Aware Hybrid improvement over {best_single['model']}: "
                f"{improvement:+.1f}%"
            )

        logger.info(f"\nMLflow UI: mlflow ui --port 5000")
        logger.info(f"Tracking URI: {tracker.tracking_uri}")

        return results_df


if __name__ == "__main__":
    main()
