"""Statistical analysis for A/B test experiments.

Provides statistical significance testing and confidence interval calculation.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

# Optional scipy import for advanced statistics
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning(
        "scipy not installed. Some statistical functions will be limited. "
        "Install with: pip install scipy"
    )


@dataclass
class StatisticalResult:
    """Result of statistical significance test."""

    control_mean: float
    treatment_mean: float
    relative_lift: float  # (treatment - control) / control
    p_value: float
    confidence_level: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    control_std: float
    treatment_std: float
    control_n: int
    treatment_n: int
    test_type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "control_mean": self.control_mean,
            "treatment_mean": self.treatment_mean,
            "relative_lift": self.relative_lift,
            "relative_lift_percent": self.relative_lift * 100,
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "is_significant": self.is_significant,
            "confidence_interval": self.confidence_interval,
            "control_std": self.control_std,
            "treatment_std": self.treatment_std,
            "control_n": self.control_n,
            "treatment_n": self.treatment_n,
            "test_type": self.test_type,
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lift_pct = self.relative_lift * 100
        ci_low, ci_high = self.confidence_interval

        significance = "SIGNIFICANT" if self.is_significant else "NOT significant"

        return (
            f"Treatment vs Control: {lift_pct:+.2f}% lift\n"
            f"Control: {self.control_mean:.4f} (n={self.control_n})\n"
            f"Treatment: {self.treatment_mean:.4f} (n={self.treatment_n})\n"
            f"p-value: {self.p_value:.4f}\n"
            f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]\n"
            f"Result: {significance} at {self.confidence_level*100:.0f}% confidence"
        )


@dataclass
class SampleSizeResult:
    """Result of sample size calculation."""

    required_sample_size: int
    per_variant: int
    baseline_rate: float
    minimum_detectable_effect: float
    power: float
    significance_level: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "required_sample_size": self.required_sample_size,
            "per_variant": self.per_variant,
            "baseline_rate": self.baseline_rate,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "power": self.power,
            "significance_level": self.significance_level,
        }


class StatisticalAnalyzer:
    """Statistical analysis for A/B test experiments.

    Features:
    - T-test for mean comparison
    - Chi-squared test for proportions
    - Bayesian analysis
    - Sample size calculation
    - Confidence interval calculation
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
    ):
        """Initialize analyzer.

        Args:
            confidence_level: Default confidence level for tests.
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compare_means(
        self,
        control_values: list[float] | np.ndarray,
        treatment_values: list[float] | np.ndarray,
        confidence_level: float | None = None,
    ) -> StatisticalResult:
        """Compare means using Welch's t-test.

        Args:
            control_values: Control group values.
            treatment_values: Treatment group values.
            confidence_level: Confidence level for test.

        Returns:
            Statistical result.
        """
        confidence_level = confidence_level or self.confidence_level
        alpha = 1 - confidence_level

        control = np.array(control_values)
        treatment = np.array(treatment_values)

        control_mean = float(np.mean(control))
        treatment_mean = float(np.mean(treatment))
        control_std = float(np.std(control, ddof=1))
        treatment_std = float(np.std(treatment, ddof=1))
        control_n = len(control)
        treatment_n = len(treatment)

        # Calculate relative lift
        if control_mean != 0:
            relative_lift = (treatment_mean - control_mean) / abs(control_mean)
        else:
            relative_lift = 0.0

        # Perform Welch's t-test
        if SCIPY_AVAILABLE:
            t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        else:
            # Manual calculation (simplified)
            pooled_se = math.sqrt(
                control_std**2 / control_n + treatment_std**2 / treatment_n
            )
            if pooled_se > 0:
                t_stat = (treatment_mean - control_mean) / pooled_se
                # Approximate p-value using normal distribution
                p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
            else:
                t_stat = 0
                p_value = 1.0

        # Calculate confidence interval for the difference
        diff_mean = treatment_mean - control_mean
        diff_se = math.sqrt(control_std**2 / control_n + treatment_std**2 / treatment_n)

        if SCIPY_AVAILABLE:
            # Welch-Satterthwaite degrees of freedom
            df = self._welch_df(control_std, treatment_std, control_n, treatment_n)
            t_critical = stats.t.ppf(1 - alpha / 2, df)
        else:
            t_critical = 1.96  # Approximate for large samples

        ci_low = diff_mean - t_critical * diff_se
        ci_high = diff_mean + t_critical * diff_se

        return StatisticalResult(
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            relative_lift=relative_lift,
            p_value=float(p_value),
            confidence_level=confidence_level,
            is_significant=p_value < alpha,
            confidence_interval=(ci_low, ci_high),
            control_std=control_std,
            treatment_std=treatment_std,
            control_n=control_n,
            treatment_n=treatment_n,
            test_type="welch_t_test",
        )

    def compare_proportions(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        confidence_level: float | None = None,
    ) -> StatisticalResult:
        """Compare proportions using chi-squared test.

        Args:
            control_successes: Number of successes in control.
            control_total: Total samples in control.
            treatment_successes: Number of successes in treatment.
            treatment_total: Total samples in treatment.
            confidence_level: Confidence level for test.

        Returns:
            Statistical result.
        """
        confidence_level = confidence_level or self.confidence_level
        alpha = 1 - confidence_level

        control_rate = control_successes / control_total if control_total > 0 else 0
        treatment_rate = treatment_successes / treatment_total if treatment_total > 0 else 0

        # Calculate relative lift
        if control_rate != 0:
            relative_lift = (treatment_rate - control_rate) / control_rate
        else:
            relative_lift = 0.0

        # Chi-squared test
        if SCIPY_AVAILABLE:
            # Create contingency table
            observed = np.array([
                [control_successes, control_total - control_successes],
                [treatment_successes, treatment_total - treatment_successes],
            ])
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        else:
            # Manual calculation (simplified)
            pooled_rate = (control_successes + treatment_successes) / (
                control_total + treatment_total
            )
            se = math.sqrt(
                pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total)
            )
            if se > 0:
                z_stat = (treatment_rate - control_rate) / se
                p_value = 2 * (1 - self._normal_cdf(abs(z_stat)))
            else:
                p_value = 1.0

        # Calculate confidence interval for difference in proportions
        se_diff = math.sqrt(
            control_rate * (1 - control_rate) / control_total
            + treatment_rate * (1 - treatment_rate) / treatment_total
        )

        if SCIPY_AVAILABLE:
            z_critical = stats.norm.ppf(1 - alpha / 2)
        else:
            z_critical = 1.96

        diff = treatment_rate - control_rate
        ci_low = diff - z_critical * se_diff
        ci_high = diff + z_critical * se_diff

        # Calculate standard deviations for proportions
        control_std = math.sqrt(control_rate * (1 - control_rate))
        treatment_std = math.sqrt(treatment_rate * (1 - treatment_rate))

        return StatisticalResult(
            control_mean=control_rate,
            treatment_mean=treatment_rate,
            relative_lift=relative_lift,
            p_value=float(p_value),
            confidence_level=confidence_level,
            is_significant=p_value < alpha,
            confidence_interval=(ci_low, ci_high),
            control_std=control_std,
            treatment_std=treatment_std,
            control_n=control_total,
            treatment_n=treatment_total,
            test_type="chi_squared",
        )

    def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8,
        significance_level: float = 0.05,
        n_variants: int = 2,
    ) -> SampleSizeResult:
        """Calculate required sample size for experiment.

        Args:
            baseline_rate: Expected baseline conversion/metric rate.
            minimum_detectable_effect: Minimum effect to detect (relative).
            power: Statistical power (1 - beta).
            significance_level: Significance level (alpha).
            n_variants: Number of variants (including control).

        Returns:
            Sample size calculation result.
        """
        # Calculate effect size
        treatment_rate = baseline_rate * (1 + minimum_detectable_effect)

        if SCIPY_AVAILABLE:
            z_alpha = stats.norm.ppf(1 - significance_level / 2)
            z_beta = stats.norm.ppf(power)
        else:
            z_alpha = 1.96  # For alpha = 0.05
            z_beta = 0.84  # For power = 0.8

        # Pooled proportion
        pooled_rate = (baseline_rate + treatment_rate) / 2

        # Sample size per variant (for proportion test)
        numerator = (
            2 * pooled_rate * (1 - pooled_rate) * (z_alpha + z_beta) ** 2
        )
        denominator = (treatment_rate - baseline_rate) ** 2

        if denominator > 0:
            n_per_variant = int(math.ceil(numerator / denominator))
        else:
            n_per_variant = float("inf")

        total_sample_size = n_per_variant * n_variants

        return SampleSizeResult(
            required_sample_size=total_sample_size,
            per_variant=n_per_variant,
            baseline_rate=baseline_rate,
            minimum_detectable_effect=minimum_detectable_effect,
            power=power,
            significance_level=significance_level,
        )

    def bayesian_analysis(
        self,
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        n_samples: int = 10000,
    ) -> dict[str, Any]:
        """Perform Bayesian analysis using Beta-Binomial model.

        Args:
            control_successes: Successes in control.
            control_total: Total in control.
            treatment_successes: Successes in treatment.
            treatment_total: Total in treatment.
            prior_alpha: Beta prior alpha parameter.
            prior_beta: Beta prior beta parameter.
            n_samples: Number of Monte Carlo samples.

        Returns:
            Bayesian analysis results.
        """
        # Posterior parameters
        control_alpha = prior_alpha + control_successes
        control_beta = prior_beta + control_total - control_successes
        treatment_alpha = prior_alpha + treatment_successes
        treatment_beta = prior_beta + treatment_total - treatment_successes

        # Sample from posteriors
        np.random.seed(42)
        control_samples = np.random.beta(control_alpha, control_beta, n_samples)
        treatment_samples = np.random.beta(treatment_alpha, treatment_beta, n_samples)

        # Calculate probability that treatment > control
        prob_treatment_better = np.mean(treatment_samples > control_samples)

        # Calculate expected lift
        lift_samples = (treatment_samples - control_samples) / control_samples
        expected_lift = np.mean(lift_samples)
        lift_ci = (np.percentile(lift_samples, 2.5), np.percentile(lift_samples, 97.5))

        # Risk analysis
        risk_of_choosing_treatment = np.mean(
            np.maximum(control_samples - treatment_samples, 0)
        )
        risk_of_choosing_control = np.mean(
            np.maximum(treatment_samples - control_samples, 0)
        )

        return {
            "prob_treatment_better": prob_treatment_better,
            "expected_lift": expected_lift,
            "lift_95_ci": lift_ci,
            "control_posterior_mean": control_alpha / (control_alpha + control_beta),
            "treatment_posterior_mean": treatment_alpha / (treatment_alpha + treatment_beta),
            "risk_of_choosing_treatment": risk_of_choosing_treatment,
            "risk_of_choosing_control": risk_of_choosing_control,
            "recommendation": (
                "treatment" if prob_treatment_better > 0.95 else
                "control" if prob_treatment_better < 0.05 else
                "continue_experiment"
            ),
        }

    def _welch_df(
        self,
        s1: float,
        s2: float,
        n1: int,
        n2: int,
    ) -> float:
        """Calculate Welch-Satterthwaite degrees of freedom."""
        num = (s1**2/n1 + s2**2/n2)**2
        denom = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        return num / denom if denom > 0 else 1

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def analyze_experiment(
        self,
        control_values: list[float] | np.ndarray,
        treatment_values: list[float] | np.ndarray,
        is_proportion: bool = False,
    ) -> dict[str, Any]:
        """Comprehensive analysis of experiment results.

        Args:
            control_values: Control group values.
            treatment_values: Treatment group values.
            is_proportion: Whether values are binary (proportions).

        Returns:
            Comprehensive analysis results.
        """
        if is_proportion:
            # Convert to successes/totals
            control = np.array(control_values)
            treatment = np.array(treatment_values)

            freq_result = self.compare_proportions(
                control_successes=int(np.sum(control)),
                control_total=len(control),
                treatment_successes=int(np.sum(treatment)),
                treatment_total=len(treatment),
            )

            bayesian_result = self.bayesian_analysis(
                control_successes=int(np.sum(control)),
                control_total=len(control),
                treatment_successes=int(np.sum(treatment)),
                treatment_total=len(treatment),
            )
        else:
            freq_result = self.compare_means(control_values, treatment_values)
            bayesian_result = None

        return {
            "frequentist": freq_result.to_dict(),
            "bayesian": bayesian_result,
            "summary": freq_result.summary(),
            "recommendation": self._get_recommendation(freq_result, bayesian_result),
        }

    def _get_recommendation(
        self,
        freq_result: StatisticalResult,
        bayesian_result: dict | None,
    ) -> str:
        """Generate recommendation based on analysis."""
        if freq_result.is_significant:
            if freq_result.relative_lift > 0:
                return "SHIP: Treatment shows significant positive improvement"
            else:
                return "REVERT: Treatment shows significant negative impact"

        if bayesian_result and bayesian_result.get("recommendation") != "continue_experiment":
            return f"CONSIDER: Bayesian analysis suggests {bayesian_result['recommendation']}"

        return "CONTINUE: Not enough evidence to make a decision yet"
