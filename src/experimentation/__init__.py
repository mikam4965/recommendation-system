"""Experimentation module for A/B testing and model experiments.

Components:
- ABTestManager: Manages A/B test configurations and variant assignment
- ExperimentTracker: Tracks experiment metrics and results
- StatisticalAnalyzer: Performs statistical significance analysis
"""

from src.experimentation.ab_testing import (
    ABTestManager,
    Experiment,
    ExperimentVariant,
    VariantAssignment,
)
from src.experimentation.metrics_tracker import ExperimentMetricsTracker
from src.experimentation.statistical import StatisticalAnalyzer

__all__ = [
    "ABTestManager",
    "Experiment",
    "ExperimentVariant",
    "VariantAssignment",
    "ExperimentMetricsTracker",
    "StatisticalAnalyzer",
]
