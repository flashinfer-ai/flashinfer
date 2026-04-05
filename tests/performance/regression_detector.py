"""Performance regression detection for FlashInfer."""

from typing import List, Dict, Any, Optional
from .statistics import StatisticalAnalyzer


class RegressionDetector:
    """
    Detect performance regressions by comparing current results
    against baseline measurements.
    """
    
    def __init__(
        self,
        threshold_percent: float = 5.0,
        min_samples: int = 3,
        use_statistical_test: bool = True,
        significance_level: float = 0.05
    ):
        """
        Initialize regression detector.
        
        Args:
            threshold_percent: Regression threshold as percentage
                              (e.g., 5.0 means 5% slower is a regression)
            min_samples: Minimum number of samples required for analysis
            use_statistical_test: Whether to use statistical significance test
            significance_level: Alpha for statistical tests (default: 0.05)
        """
        self.threshold_percent = threshold_percent
        self.min_samples = min_samples
        self.use_statistical_test = use_statistical_test
        self.significance_level = significance_level
        self.analyzer = StatisticalAnalyzer()
    
    def detect_regressions(
        self,
        current_results: List[Dict[str, Any]],
        baseline_results: List[Dict[str, Any]],
        metric: str = "median_latency_ms"
    ) -> List[Dict[str, Any]]:
        """
        Detect performance regressions.
        
        Args:
            current_results: Current benchmark results
            baseline_results: Baseline benchmark results
            metric: Metric to compare (default: median_latency_ms)
        
        Returns:
            List of regression dictionaries
        """
        regressions = []
        
        # Build lookup table for baseline results
        baseline_lookup = {}
        for result in baseline_results:
            key = self._get_result_key(result)
            baseline_lookup[key] = result
        
        # Compare current results against baseline
        for current in current_results:
            key = self._get_result_key(current)
            
            if key not in baseline_lookup:
                continue  # Skip if no matching baseline
            
            baseline = baseline_lookup[key]
            regression = self._compare_results(
                current, baseline, metric
            )
            
            if regression:
                regressions.append(regression)
        
        return regressions
    
    def _get_result_key(self, result: Dict[str, Any]) -> str:
        """Generate a unique key for a benchmark result."""
        return (
            f"{result.get('routine', 'unknown')}_"
            f"{result.get('backend', 'unknown')}_"
            f"{result.get('batch_size', 0)}_"
            f"{result.get('kv_len', 0)}_"
            f"{result.get('num_qo_heads', 0)}_"
            f"{result.get('head_dim', 0)}"
        )
    
    def _compare_results(
        self,
        current: Dict[str, Any],
        baseline: Dict[str, Any],
        metric: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compare current result against baseline.
        
        Returns:
            Regression dictionary if regression detected, None otherwise
        """
        current_value = current.get(metric)
        baseline_value = baseline.get(metric)
        
        if current_value is None or baseline_value is None:
            return None
        
        if baseline_value == 0:
            return None  # Avoid division by zero
        
        # Compute percentage change
        # For latency metrics, higher is worse (regression)
        # For throughput metrics, lower is worse (regression)
        if "latency" in metric.lower() or "time" in metric.lower():
            percent_change = (
                (current_value - baseline_value) / baseline_value * 100
            )
            is_regression = percent_change > self.threshold_percent
        else:  # Throughput or performance metrics
            percent_change = (
                (baseline_value - current_value) / baseline_value * 100
            )
            is_regression = percent_change > self.threshold_percent
        
        if not is_regression:
            return None
        
        # If using statistical test, verify significance
        statistically_significant = True
        if self.use_statistical_test:
            # Check if we have raw measurement data
            current_samples = current.get('raw_samples', [current_value])
            baseline_samples = baseline.get('raw_samples', [baseline_value])
            
            if (len(current_samples) >= self.min_samples and
                len(baseline_samples) >= self.min_samples):
                test_result = self.analyzer.mann_whitney_u_test(
                    baseline_samples,
                    current_samples,
                    alpha=self.significance_level
                )
                statistically_significant = test_result.get(
                    'statistically_significant', True
                )
            
            if not statistically_significant:
                return None  # Not a statistically significant regression
        
        # Calculate effect size
        effect_size = None
        if 'raw_samples' in current and 'raw_samples' in baseline:
            effect_size = self.analyzer.effect_size_cohens_d(
                baseline['raw_samples'],
                current['raw_samples']
            )
        
        return {
            "test_name": self._get_result_key(current),
            "routine": current.get('routine', 'unknown'),
            "backend": current.get('backend', 'unknown'),
            "metric": metric,
            "baseline_value": baseline_value,
            "current_value": current_value,
            "percent_change": percent_change,
            "threshold_percent": self.threshold_percent,
            "statistically_significant": statistically_significant,
            "effect_size": effect_size,
            "severity": self._classify_severity(percent_change, effect_size),
            "config": {
                "batch_size": current.get('batch_size'),
                "kv_len": current.get('kv_len'),
                "num_qo_heads": current.get('num_qo_heads'),
                "num_kv_heads": current.get('num_kv_heads'),
                "head_dim": current.get('head_dim'),
            }
        }
    
    def _classify_severity(
        self,
        percent_change: float,
        effect_size: Optional[float]
    ) -> str:
        """
        Classify regression severity.
        
        Args:
            percent_change: Percentage performance change
            effect_size: Cohen's d effect size
        
        Returns:
            Severity level: 'critical', 'high', 'medium', 'low'
        """
        # Use effect size if available
        if effect_size is not None:
            abs_effect = abs(effect_size)
            if abs_effect >= 1.2:
                return "critical"
            elif abs_effect >= 0.8:
                return "high"
            elif abs_effect >= 0.5:
                return "medium"
            else:
                return "low"
        
        # Fallback to percentage change
        abs_change = abs(percent_change)
        if abs_change >= 20:
            return "critical"
        elif abs_change >= 10:
            return "high"
        elif abs_change >= 5:
            return "medium"
        else:
            return "low"
    
    def generate_report(
        self,
        regressions: List[Dict[str, Any]],
        include_all_results: bool = False
    ) -> str:
        """
        Generate a human-readable regression report.
        
        Args:
            regressions: List of detected regressions
            include_all_results: Include non-regressed tests in report
        
        Returns:
            Report string
        """
        if not regressions:
            return "✓ No performance regressions detected!"
        
        # Sort by severity and percent change
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        regressions.sort(
            key=lambda x: (
                severity_order.get(x['severity'], 99),
                -abs(x['percent_change'])
            )
        )
        
        lines = [
            "⚠️  Performance Regressions Detected",
            "=" * 70,
            ""
        ]
        
        for reg in regressions:
            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }.get(reg['severity'], "⚪")
            
            lines.append(
                f"{severity_emoji} {reg['severity'].upper()}: {reg['test_name']}"
            )
            lines.append(f"   Routine: {reg['routine']}")
            lines.append(f"   Backend: {reg['backend']}")
            lines.append(f"   Metric: {reg['metric']}")
            lines.append(
                f"   Baseline: {reg['baseline_value']:.3f} → "
                f"Current: {reg['current_value']:.3f}"
            )
            lines.append(
                f"   Change: {reg['percent_change']:+.1f}% "
                f"(threshold: {reg['threshold_percent']}%)"
            )
            
            if reg.get('effect_size') is not None:
                lines.append(f"   Effect size: {reg['effect_size']:.3f}")
            
            lines.append(f"   Config: {reg['config']}")
            lines.append("")
        
        lines.append(f"Total regressions found: {len(regressions)}")
        
        return "\n".join(lines)
    
    def export_to_json(
        self,
        regressions: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Export regressions to JSON file.
        
        Args:
            regressions: List of regression dictionaries
            output_path: Path to output JSON file
        """
        import json
        from datetime import datetime
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_regressions": len(regressions),
            "regressions": regressions
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Exported regression report to {output_path}")
