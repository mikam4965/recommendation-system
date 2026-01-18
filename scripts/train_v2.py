#!/usr/bin/env python3
"""Train and evaluate Version 2.0 recommendation models.

This script trains all V2.0 models:
- GRU4Rec (Session-based RNN)
- SASRec (Self-Attentive Sequential)
- Two-Tower (Dual Encoder with FAISS)
- Session-History Fusion Hybrid (Scientific Novelty #2)

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
from src.models.content.item2vec import Item2VecRecommender
from src.models.explainable.explainer import RecommendationExplainer
from src.models.hybrid.funnel_aware import FunnelAwareHybridRecommender
from src.models.retrieval.two_tower import TwoTowerRecommender
from src.models.sequential.gru4rec import GRU4RecRecommender
from src.models.sequential.sasrec import SASRecRecommender
from src.training.mlflow_tracker import MLflowTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train V2.0 recommendation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--max-users",
        type=int,
        default=5000,
        help="Maximum users to evaluate (for speed)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=20,
        help="Number of items to recommend per user",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for deep learning models",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--skip-gru4rec",
        action="store_true",
        help="Skip GRU4Rec training",
    )
    parser.add_argument(
        "--skip-sasrec",
        action="store_true",
        help="Skip SASRec training",
    )
    parser.add_argument(
        "--skip-two-tower",
        action="store_true",
        help="Skip Two-Tower training",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="recsys-v2",
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
    """Train and evaluate all V2.0 models."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("VERSION 2.0 MODEL TRAINING - Deep Learning & XAI")
    logger.info("=" * 70)

    # Initialize MLflow tracker
    tracker = MLflowTracker(experiment_name=args.experiment_name)

    with tracker.run(run_name="v2_full_training"):
        # ============================================================
        # STEP 1: Load and prepare data
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: Loading and preparing data")
        logger.info("=" * 70)

        loader = RetailRocketLoader()
        events = loader.load_events()

        # Build sessions (required for sequential models)
        logger.info("\nBuilding sessions...")
        session_builder = SessionBuilder(timeout_minutes=30)
        events = session_builder.build_sessions(events)

        # Filter short sessions for sequential models
        events_filtered = session_builder.filter_short_sessions(events, min_length=2)

        # Log data stats
        tracker.log_params({
            "total_events": len(events),
            "filtered_events": len(events_filtered),
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
        train, val, test = splitter.split(events_filtered)

        # Get split stats
        split_stats = splitter.get_split_stats(train, val, test)
        logger.info(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
        logger.info(f"Cold-start users in test: {split_stats['test_cold_start_users']:,}")

        tracker.log_params({
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "test_cold_start_users": split_stats["test_cold_start_users"],
            "test_cold_start_items": split_stats["test_cold_start_items"],
        })

        # ============================================================
        # STEP 3: Extract user features
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Extracting user features")
        logger.info("=" * 70)

        user_feature_extractor = UserFeatureExtractor()
        user_feature_extractor.fit(train)

        # ============================================================
        # STEP 4: Train baseline models for comparison
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Training baseline models")
        logger.info("=" * 70)

        evaluator = Evaluator(k_values=[5, 10, 20])
        all_results = []

        # --- Popular baseline ---
        logger.info("\n--- Training Popular baseline ---")
        popular_model = PopularItemsRecommender()
        popular_model.fit(train)
        popular_results = evaluator.evaluate(
            popular_model, train, test,
            n_items=args.n_items, max_users=args.max_users
        )
        all_results.append(popular_results)
        logger.info(f"Popular NDCG@10: {popular_results.get('ndcg@10', 0):.4f}")

        # --- ALS for hybrid ---
        logger.info("\n--- Training ALS ---")
        als_model = ALSRecommender(factors=64, regularization=0.01, iterations=15)
        als_model.fit(train)
        als_results = evaluator.evaluate(
            als_model, train, test,
            n_items=args.n_items, max_users=args.max_users
        )
        all_results.append(als_results)
        logger.info(f"ALS NDCG@10: {als_results.get('ndcg@10', 0):.4f}")

        # ============================================================
        # STEP 5: Train GRU4Rec (Session-based)
        # ============================================================
        gru4rec_model = None
        if not args.skip_gru4rec:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 5: Training GRU4Rec (Session-based RNN)")
            logger.info("=" * 70)

            with tracker.run(run_name="gru4rec", nested=True):
                gru4rec_model = GRU4RecRecommender(
                    embedding_dim=args.embedding_dim,
                    hidden_dim=128,
                    n_layers=1,
                    dropout=0.2,
                    learning_rate=0.001,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    loss_type="ce",
                )
                tracker.log_params(gru4rec_model.get_params())

                gru4rec_model.fit(train)

                gru4rec_results = evaluator.evaluate(
                    gru4rec_model, train, test,
                    n_items=args.n_items, max_users=args.max_users
                )
                all_results.append(gru4rec_results)

                tracker.log_metrics({
                    k: v for k, v in gru4rec_results.items()
                    if isinstance(v, (int, float)) and k != "n_users"
                })
                logger.info(f"GRU4Rec NDCG@10: {gru4rec_results.get('ndcg@10', 0):.4f}")
        else:
            logger.info("\n--- Skipping GRU4Rec (--skip-gru4rec flag set) ---")

        # ============================================================
        # STEP 6: Train SASRec (Self-Attentive Sequential)
        # ============================================================
        sasrec_model = None
        if not args.skip_sasrec:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 6: Training SASRec (Self-Attentive Sequential)")
            logger.info("=" * 70)

            with tracker.run(run_name="sasrec", nested=True):
                sasrec_model = SASRecRecommender(
                    hidden_dim=args.embedding_dim,
                    n_heads=2,
                    n_layers=2,
                    max_seq_length=50,
                    dropout=0.2,
                    learning_rate=0.001,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                )
                tracker.log_params(sasrec_model.get_params())

                sasrec_model.fit(train)

                sasrec_results = evaluator.evaluate(
                    sasrec_model, train, test,
                    n_items=args.n_items, max_users=args.max_users
                )
                all_results.append(sasrec_results)

                tracker.log_metrics({
                    k: v for k, v in sasrec_results.items()
                    if isinstance(v, (int, float)) and k != "n_users"
                })
                logger.info(f"SASRec NDCG@10: {sasrec_results.get('ndcg@10', 0):.4f}")
        else:
            logger.info("\n--- Skipping SASRec (--skip-sasrec flag set) ---")

        # ============================================================
        # STEP 7: Train Two-Tower with FAISS
        # ============================================================
        two_tower_model = None
        if not args.skip_two_tower:
            logger.info("\n" + "=" * 70)
            logger.info("STEP 7: Training Two-Tower with FAISS")
            logger.info("=" * 70)

            with tracker.run(run_name="two_tower", nested=True):
                two_tower_model = TwoTowerRecommender(
                    embedding_dim=args.embedding_dim,
                    hidden_dims=[128, 64],
                    dropout=0.2,
                    temperature=0.1,
                    learning_rate=0.001,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    negative_samples=4,
                    use_faiss=True,
                )
                tracker.log_params(two_tower_model.get_params())

                two_tower_model.fit(train)

                two_tower_results = evaluator.evaluate(
                    two_tower_model, train, test,
                    n_items=args.n_items, max_users=args.max_users
                )
                all_results.append(two_tower_results)

                tracker.log_metrics({
                    k: v for k, v in two_tower_results.items()
                    if isinstance(v, (int, float)) and k != "n_users"
                })
                logger.info(f"Two-Tower NDCG@10: {two_tower_results.get('ndcg@10', 0):.4f}")
        else:
            logger.info("\n--- Skipping Two-Tower (--skip-two-tower flag set) ---")

        # ============================================================
        # STEP 8: Train Session + History Fusion Hybrid (Novelty #2)
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 8: Training Session + History Fusion Hybrid (Novelty #2)")
        logger.info("=" * 70)

        # Use best session-based model
        session_model = None
        if sasrec_model is not None:
            session_model = sasrec_model
            logger.info("Using SASRec as session component")
        elif gru4rec_model is not None:
            session_model = gru4rec_model
            logger.info("Using GRU4Rec as session component")

        with tracker.run(run_name="session_history_fusion", nested=True):
            # Create Item2Vec for content
            item2vec_model = Item2VecRecommender(
                embedding_dim=64, window=5, min_count=5, epochs=10
            )
            item2vec_model.fit(train)

            hybrid_model = FunnelAwareHybridRecommender(
                popular_model=popular_model,
                content_model=item2vec_model,
                cf_model=als_model,
                session_model=session_model,
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
            logger.info(f"Session+History Fusion NDCG@10: {hybrid_results.get('ndcg@10', 0):.4f}")

        # ============================================================
        # STEP 9: Initialize Explainer (Novelty #3)
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 9: Initializing Explainable AI (Novelty #3)")
        logger.info("=" * 70)

        # Get item categories if available
        try:
            item_props = loader.load_item_properties()
            item_categories = {}
            if "categoryid" in item_props.columns:
                for _, row in item_props.drop_duplicates("itemid").iterrows():
                    if pd.notna(row.get("categoryid")):
                        item_categories[row["itemid"]] = int(row["categoryid"])
            logger.info(f"Loaded categories for {len(item_categories)} items")
        except Exception as e:
            logger.warning(f"Could not load item categories: {e}")
            item_categories = {}

        # Get item popularity
        item_popularity = train["item_id"].value_counts().to_dict()

        explainer = RecommendationExplainer(
            user_feature_extractor=user_feature_extractor,
            item_categories=item_categories,
            item_popularity=item_popularity,
        )
        explainer.fit(train)

        # Demo explanation
        logger.info("\n--- Demo Explanation ---")
        sample_user = train["visitor_id"].iloc[0]
        sample_recs = hybrid_model.recommend(sample_user, n_items=3)

        if sample_recs:
            explanations = explainer.explain_batch(
                user_id=sample_user,
                recommendations=sample_recs,
                model_name=hybrid_model.name,
            )

            for exp in explanations:
                logger.info(f"Item {exp.item_id}: {exp.get_summary()}")

        # ============================================================
        # STEP 10: Results comparison
        # ============================================================
        logger.info("\n" + "=" * 70)
        logger.info("STEP 10: Results comparison")
        logger.info("=" * 70)

        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("ndcg@10", ascending=False)

        # Print results table
        evaluator.print_results(results_df)

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "v2_results.csv"
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

        # Calculate improvements
        baseline_ndcg = popular_results.get("ndcg@10", 0)
        if baseline_ndcg > 0:
            improvement = (best_ndcg - baseline_ndcg) / baseline_ndcg * 100
            logger.info(f"Improvement over Popular baseline: {improvement:+.1f}%")

        # Session model improvement
        if session_model is not None:
            session_results = gru4rec_results if gru4rec_model else sasrec_results
            session_ndcg = session_results.get("ndcg@10", 0)
            als_ndcg = als_results.get("ndcg@10", 0)

            if als_ndcg > 0 and session_ndcg > als_ndcg:
                session_improvement = (session_ndcg - als_ndcg) / als_ndcg * 100
                logger.info(f"Session model improvement over ALS: {session_improvement:+.1f}%")

        logger.info(f"\nMLflow UI: mlflow ui --port 5000")
        logger.info(f"Tracking URI: {tracker.tracking_uri}")

        return results_df


if __name__ == "__main__":
    main()
