"""
Statistical analysis utilities for all experiments.

Includes:
- Effect size computation (Cohen's d)
- Confidence intervals (bootstrap and parametric)
- Holm-Bonferroni multiple-comparisons correction
- Paired and independent t-tests with effect sizes
- Summary statistics across seeds
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: Optional[float]  # after Holm-Bonferroni
    effect_size: float                   # Cohen's d
    ci_lower: float                      # 95% CI lower bound
    ci_upper: float                      # 95% CI upper bound
    n_samples: int
    significant: bool                    # after correction
    description: str = ""


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d (standardized mean difference).

    Uses pooled standard deviation.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def bootstrap_ci(data: np.ndarray, statistic_fn=np.mean,
                 n_bootstrap: int = 10000, ci: float = 0.95,
                 seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of observations
        statistic_fn: function to compute (default: mean)
        n_bootstrap: number of bootstrap resamples
        ci: confidence level
        seed: random seed

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    boot_stats = np.zeros(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        boot_sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic_fn(boot_sample)

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, 100 * alpha)
    upper = np.percentile(boot_stats, 100 * (1 - alpha))
    return lower, upper


def parametric_ci(data: np.ndarray, ci: float = 0.95) -> Tuple[float, float]:
    """Compute parametric confidence interval for the mean (t-distribution)."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    t_crit = stats.t.ppf((1 + ci) / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def independent_ttest(group1: np.ndarray, group2: np.ndarray,
                       description: str = "") -> TestResult:
    """
    Independent samples t-test with Cohen's d and 95% CI on the difference.
    """
    t_stat, p_val = stats.ttest_ind(group1, group2)
    d = cohens_d(group1, group2)

    diff = group1 - group2 if len(group1) == len(group2) else None
    if diff is not None:
        ci_low, ci_high = parametric_ci(diff)
    else:
        # CI on the difference of means
        se_diff = np.sqrt(stats.sem(group1)**2 + stats.sem(group2)**2)
        mean_diff = np.mean(group1) - np.mean(group2)
        t_crit = stats.t.ppf(0.975, df=len(group1) + len(group2) - 2)
        ci_low = mean_diff - t_crit * se_diff
        ci_high = mean_diff + t_crit * se_diff

    return TestResult(
        test_name="Independent t-test",
        statistic=t_stat,
        p_value=p_val,
        p_value_corrected=None,
        effect_size=d,
        ci_lower=ci_low,
        ci_upper=ci_high,
        n_samples=len(group1) + len(group2),
        significant=False,  # set after correction
        description=description,
    )


def paired_ttest(group1: np.ndarray, group2: np.ndarray,
                  description: str = "") -> TestResult:
    """
    Paired samples t-test with Cohen's d and 95% CI on the paired difference.
    """
    assert len(group1) == len(group2), "Paired test requires equal-length arrays"
    t_stat, p_val = stats.ttest_rel(group1, group2)
    d = cohens_d(group1, group2)
    diff = group1 - group2
    ci_low, ci_high = parametric_ci(diff)

    return TestResult(
        test_name="Paired t-test",
        statistic=t_stat,
        p_value=p_val,
        p_value_corrected=None,
        effect_size=d,
        ci_lower=ci_low,
        ci_upper=ci_high,
        n_samples=len(group1),
        significant=False,
        description=description,
    )


def holm_bonferroni(results: List[TestResult], alpha: float = 0.05) -> List[TestResult]:
    """
    Apply Holm-Bonferroni correction to a list of test results.

    This is the multiple-comparisons correction specified in the v2 methodology
    (Section 7, point 5).

    The procedure:
    1. Sort p-values from smallest to largest
    2. For the i-th smallest p-value (0-indexed), compare against alpha / (m - i)
       where m is the total number of tests
    3. Reject hypotheses in order until the first non-rejection
    """
    m = len(results)
    if m == 0:
        return results

    # Sort by p-value
    indexed_results = sorted(enumerate(results), key=lambda x: x[1].p_value)

    corrected_results = [None] * m
    any_non_rejected = False

    for rank, (orig_idx, result) in enumerate(indexed_results):
        adjusted_alpha = alpha / (m - rank)
        corrected_p = result.p_value * (m - rank)
        corrected_p = min(corrected_p, 1.0)

        if any_non_rejected:
            # Once we fail to reject one, all subsequent are also not rejected
            result.p_value_corrected = corrected_p
            result.significant = False
        else:
            result.p_value_corrected = corrected_p
            if result.p_value <= adjusted_alpha:
                result.significant = True
            else:
                result.significant = False
                any_non_rejected = True

        corrected_results[orig_idx] = result

    return corrected_results


def summarize_across_seeds(seed_results: List[Dict],
                            key: str) -> Dict:
    """
    Summarize a metric across multiple seeds.

    Args:
        seed_results: list of dicts, one per seed, each containing the key
        key: the metric name to summarize

    Returns:
        dict with mean, std, ci_lower, ci_upper, median, min, max
    """
    values = np.array([r[key] for r in seed_results])
    ci_low, ci_high = parametric_ci(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_seeds": len(values),
        "values": values.tolist(),
    }


def format_result(result: TestResult) -> str:
    """Format a test result as a human-readable string."""
    sig_str = "[YES] SIGNIFICANT" if result.significant else "[NO] not significant"
    corr_str = f" (corrected p = {result.p_value_corrected:.6f})" if result.p_value_corrected is not None else ""
    return (
        f"{result.description}\n"
        f"  Test: {result.test_name}\n"
        f"  t = {result.statistic:.4f}, p = {result.p_value:.6f}{corr_str}\n"
        f"  Cohen's d = {result.effect_size:.4f}\n"
        f"  95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]\n"
        f"  n = {result.n_samples}\n"
        f"  {sig_str}"
    )
