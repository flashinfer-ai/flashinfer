"""Statistical analysis for performance testing."""

import warnings
from typing import List, Dict, Any, Tuple, Optional
import statistics


class StatisticalAnalyzer:
    """Statistical analysis tools for performance data."""
    
    @staticmethod
    def compute_statistics(values: List[float]) -> Dict[str, float]:
        """
        Compute comprehensive statistics for a list of values.
        
        Args:
            values: List of numeric values
        
        Returns:
            Dictionary with statistical metrics
        """
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }
        
        sorted_values = sorted(values)
        n = len(values)
        
        return {
            "count": n,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "p95": sorted_values[int(0.95 * n)] if n > 0 else 0.0,
            "p99": sorted_values[int(0.99 * n)] if n > 0 else 0.0,
        }
    
    @staticmethod
    def detect_outliers(
        values: List[float],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Tuple[List[int], List[float]]:
        """
        Detect outliers in performance measurements.
        
        Args:
            values: List of measurements
            method: Detection method ('iqr' or 'zscore')
            threshold: Outlier threshold (1.5 for IQR, 3.0 for z-score)
        
        Returns:
            Tuple of (outlier_indices, filtered_values)
        """
        if len(values) < 4:
            return [], values
        
        if method == "iqr":
            # Interquartile range method
            sorted_values = sorted(values)
            n = len(sorted_values)
            q1 = sorted_values[n // 4]
            q3 = sorted_values[3 * n // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_indices = [
                i for i, v in enumerate(values)
                if v < lower_bound or v > upper_bound
            ]
            filtered_values = [
                v for v in values
                if lower_bound <= v <= upper_bound
            ]
        
        elif method == "zscore":
            # Z-score method
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            
            if std == 0:
                return [], values
            
            z_scores = [(v - mean) / std for v in values]
            outlier_indices = [
                i for i, z in enumerate(z_scores)
                if abs(z) > threshold
            ]
            filtered_values = [
                v for i, v in enumerate(values)
                if abs(z_scores[i]) <= threshold
            ]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outlier_indices, filtered_values
    
    @staticmethod
    def mann_whitney_u_test(
        baseline: List[float],
        current: List[float],
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test to compare two distributions.
        
        This is a non-parametric test that doesn't assume normal distribution,
        making it suitable for performance measurements.
        
        Args:
            baseline: Baseline measurements
            current: Current measurements
            alpha: Significance level
        
        Returns:
            Test result dictionary
        """
        try:
            from scipy import stats
        except ImportError:
            warnings.warn(
                "scipy not available. Install with: pip install scipy"
            )
            return {
                "statistically_significant": False,
                "p_value": None,
                "error": "scipy not installed"
            }
        
        if len(baseline) < 3 or len(current) < 3:
            return {
                "statistically_significant": False,
                "p_value": None,
                "error": "Insufficient samples"
            }
        
        try:
            statistic, p_value = stats.mannwhitneyu(
                baseline,
                current,
                alternative='two-sided'
            )
            
            return {
                "statistically_significant": p_value < alpha,
                "p_value": float(p_value),
                "statistic": float(statistic),
                "interpretation": (
                    "Distributions are significantly different"
                    if p_value < alpha
                    else "No significant difference detected"
                )
            }
        except Exception as e:
            return {
                "statistically_significant": False,
                "p_value": None,
                "error": str(e)
            }
    
    @staticmethod
    def effect_size_cohens_d(
        baseline: List[float],
        current: List[float]
    ) -> Optional[float]:
        """
        Compute Cohen's d effect size.
        
        This measures the standardized difference between two means,
        helping understand the practical significance of performance changes.
        
        |d| < 0.2: Small effect
        0.2 <= |d| < 0.8: Medium effect
        |d| >= 0.8: Large effect
        
        Args:
            baseline: Baseline measurements
            current: Current measurements
        
        Returns:
            Cohen's d value, or None if cannot be computed
        """
        if len(baseline) < 2 or len(current) < 2:
            return None
        
        mean_baseline = statistics.mean(baseline)
        mean_current = statistics.mean(current)
        
        std_baseline = statistics.stdev(baseline)
        std_current = statistics.stdev(current)
        
        # Pooled standard deviation
        n1, n2 = len(baseline), len(current)
        pooled_std = (
            ((n1 - 1) * std_baseline ** 2 + (n2 - 1) * std_current ** 2)
            / (n1 + n2 - 2)
        ) ** 0.5
        
        if pooled_std == 0:
            return None
        
        return (mean_current - mean_baseline) / pooled_std
    
    @staticmethod
    def bootstrap_confidence_interval(
        values: List[float],
        statistic_func: callable = statistics.mean,
        confidence: float = 0.95,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a statistic.
        
        Args:
            values: Sample values
            statistic_func: Function to compute statistic (default: mean)
            confidence: Confidence level (default: 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        import random
        
        if len(values) < 2:
            stat = statistic_func(values) if values else 0.0
            return stat, stat
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = random.choices(values, k=len(values))
            bootstrap_stats.append(statistic_func(sample))
        
        bootstrap_stats.sort()
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_bootstrap)
        upper_idx = int((1 - alpha / 2) * n_bootstrap)
        
        return bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]
