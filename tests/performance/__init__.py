"""
Performance testing and regression detection for FlashInfer.

This module provides tools for:
- Performance baseline management
- Regression detection across commits
- Statistical analysis of benchmark results
- CI/CD integration for automated performance testing
"""

from .baseline_manager import BaselineManager
from .regression_detector import RegressionDetector
from .benchmark_runner import BenchmarkRunner, BenchmarkResult
from .statistics import StatisticalAnalyzer

__all__ = [
    "BaselineManager",
    "RegressionDetector",
    "BenchmarkRunner",
    "BenchmarkResult",
    "StatisticalAnalyzer",
]
