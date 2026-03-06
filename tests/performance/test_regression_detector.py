"""Tests for performance regression detection."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.regression_detector import RegressionDetector


@pytest.fixture
def baseline_results():
    """Sample baseline results."""
    return [
        {
            "routine": "batch_decode",
            "backend": "flashinfer",
            "batch_size": 128,
            "kv_len": 2048,
            "num_qo_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "median_latency_ms": 0.285,
            "raw_samples": [0.280, 0.285, 0.290, 0.283, 0.287]
        },
        {
            "routine": "batch_prefill",
            "backend": "cudnn",
            "batch_size": 16,
            "kv_len": 512,
            "num_qo_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "median_latency_ms": 1.52,
            "raw_samples": [1.50, 1.52, 1.54, 1.51, 1.53]
        }
    ]


class TestRegressionDetector:
    """Test regression detection functionality."""
    
    def test_no_regression(self, baseline_results):
        """Test when there is no regression."""
        detector = RegressionDetector(threshold_percent=5.0)
        
        # Current results are same as baseline
        current_results = baseline_results.copy()
        
        regressions = detector.detect_regressions(
            current_results, baseline_results
        )
        
        assert len(regressions) == 0
    
    def test_latency_regression(self, baseline_results):
        """Test latency regression detection."""
        detector = RegressionDetector(threshold_percent=5.0)
        
        # Make current results 10% slower
        current_results = [
            {
                **baseline_results[0],
                "median_latency_ms": baseline_results[0]["median_latency_ms"] * 1.10,
                "raw_samples": [0.308, 0.313, 0.319, 0.311, 0.315]
            }
        ]
        
        regressions = detector.detect_regressions(
            current_results, baseline_results
        )
        
        assert len(regressions) == 1
        assert regressions[0]["routine"] == "batch_decode"
        assert regressions[0]["backend"] == "flashinfer"
        assert regressions[0]["percent_change"] > 5.0
    
    def test_throughput_improvement(self, baseline_results):
        """Test that throughput improvements are not flagged as regressions."""
        detector = RegressionDetector(threshold_percent=5.0)
        
        # Add throughput metric (higher is better)
        baseline_with_tflops = [
            {**baseline_results[0], "tflops": 10.0}
        ]
        
        # Current has higher TFLOPS (improvement)
        current_with_tflops = [
            {**baseline_with_tflops[0], "tflops": 12.0}
        ]
        
        regressions = detector.detect_regressions(
            current_with_tflops,
            baseline_with_tflops,
            metric="tflops"
        )
        
        # Should not detect regression (it's an improvement)
        assert len(regressions) == 0
    
    def test_severity_classification(self, baseline_results):
        """Test regression severity classification."""
        detector = RegressionDetector(threshold_percent=5.0)
        
        test_cases = [
            (1.06, "medium"),   # 6% slower
            (1.12, "high"),     # 12% slower
            (1.25, "critical"), # 25% slower
        ]
        
        for multiplier, expected_severity in test_cases:
            current = [{
                **baseline_results[0],
                "median_latency_ms": baseline_results[0]["median_latency_ms"] * multiplier
            }]
            
            regressions = detector.detect_regressions(
                current, baseline_results
            )
            
            if regressions:
                assert regressions[0]["severity"] == expected_severity, \
                    f"Expected {expected_severity} for {multiplier}x, got {regressions[0]['severity']}"
    
    def test_statistical_significance(self, baseline_results):
        """Test statistical significance testing."""
        try:
            import scipy
        except ImportError:
            pytest.skip("scipy required for statistical tests")
        
        detector = RegressionDetector(
            threshold_percent=5.0,
            use_statistical_test=True
        )
        
        # Create results with overlapping distributions (not significant)
        current = [{
            **baseline_results[0],
            "median_latency_ms": 0.290,  # Slightly slower
            "raw_samples": [0.285, 0.290, 0.295, 0.288, 0.292]  # Overlap with baseline
        }]
        
        regressions = detector.detect_regressions(
            current, baseline_results
        )
        
        # Should not detect regression due to lack of statistical significance
        assert len(regressions) == 0
    
    def test_generate_report(self, baseline_results):
        """Test regression report generation."""
        detector = RegressionDetector(threshold_percent=5.0)
        
        # Create a regression
        current = [{
            **baseline_results[0],
            "median_latency_ms": baseline_results[0]["median_latency_ms"] * 1.15
        }]
        
        regressions = detector.detect_regressions(
            current, baseline_results
        )
        
        report = detector.generate_report(regressions)
        
        assert "Performance Regressions Detected" in report
        assert "batch_decode" in report
        assert "flashinfer" in report
        assert "15.0%" in report or "15%" in report
    
    def test_export_to_json(self, baseline_results, tmp_path):
        """Test JSON export of regressions."""
        detector = RegressionDetector(threshold_percent=5.0)
        
        current = [{
            **baseline_results[0],
            "median_latency_ms": baseline_results[0]["median_latency_ms"] * 1.10
        }]
        
        regressions = detector.detect_regressions(
            current, baseline_results
        )
        
        output_path = tmp_path / "regressions.json"
        detector.export_to_json(regressions, str(output_path))
        
        assert output_path.exists()
        
        import json
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert "timestamp" in data
        assert "total_regressions" in data
        assert data["total_regressions"] == 1
        assert len(data["regressions"]) == 1
