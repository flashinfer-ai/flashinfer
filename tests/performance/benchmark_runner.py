"""Benchmark runner for FlashInfer performance testing."""

import subprocess
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch


@dataclass
class BenchmarkResult:
    """Container for a single benchmark result."""
    routine: str
    backend: str
    batch_size: int
    median_latency_ms: float
    std_latency_ms: float
    tflops: float
    throughput_gb_per_sec: float
    kv_len: Optional[int] = None
    num_qo_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    dtype: Optional[str] = None
    raw_samples: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkRunner:
    """
    Run FlashInfer benchmarks and collect results.
    
    This class provides utilities to:
    - Run benchmarks programmatically
    - Parse and structure results
    - Generate standard test suites
    - Compare against baselines
    """
    
    def __init__(self, benchmark_script: Optional[Path] = None):
        """
        Initialize benchmark runner.
        
        Args:
            benchmark_script: Path to flashinfer_benchmark.py
                            Defaults to benchmarks/flashinfer_benchmark.py
        """
        if benchmark_script is None:
            # Try to find benchmark script
            current_dir = Path(__file__).parent
            repo_root = current_dir.parent.parent
            benchmark_script = repo_root / "benchmarks" / "flashinfer_benchmark.py"
        
        self.benchmark_script = Path(benchmark_script)
        
        if not self.benchmark_script.exists():
            raise FileNotFoundError(
                f"Benchmark script not found: {self.benchmark_script}"
            )
    
    def run_benchmark(
        self,
        routine: str,
        backends: List[str],
        **kwargs
    ) -> List[BenchmarkResult]:
        """
        Run a single benchmark routine.
        
        Args:
            routine: Benchmark routine name (e.g., 'batch_decode')
            backends: List of backends to test
            **kwargs: Additional arguments for the benchmark
        
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        
        for backend in backends:
            try:
                result = self._run_single_benchmark(
                    routine, backend, **kwargs
                )
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Warning: Benchmark failed for {routine}/{backend}: {e}")
        
        return results
    
    def _run_single_benchmark(
        self,
        routine: str,
        backend: str,
        **kwargs
    ) -> Optional[BenchmarkResult]:
        """Run a single benchmark configuration."""
        # Build command
        cmd = [
            "python",
            str(self.benchmark_script),
            "--routine", routine,
            "--backends", backend,
            "--refcheck",  # Enable reference checking
        ]
        
        # Add kwargs as command-line arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        # Run benchmark
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"Benchmark failed: {result.stderr}")
                return None
            
            # Parse output
            return self._parse_benchmark_output(
                result.stdout,
                routine,
                backend,
                **kwargs
            )
        
        except subprocess.TimeoutExpired:
            print(f"Benchmark timed out: {routine}/{backend}")
            return None
        except Exception as e:
            print(f"Error running benchmark: {e}")
            return None
    
    def _parse_benchmark_output(
        self,
        output: str,
        routine: str,
        backend: str,
        **kwargs
    ) -> Optional[BenchmarkResult]:
        """Parse benchmark output to extract performance metrics."""
        # Look for [PERF] lines in output
        for line in output.split('\n'):
            if f"[PERF] {backend}" in line:
                # Parse performance line
                # Format: [PERF] backend :: median time X.XXX ms; std X.XXX ms; ...
                try:
                    parts = line.split('::')[1].strip()
                    metrics = {}
                    
                    for part in parts.split(';'):
                        if 'median time' in part:
                            metrics['median_latency_ms'] = float(
                                part.split()[2]
                            )
                        elif 'std' in part:
                            metrics['std_latency_ms'] = float(
                                part.split()[1]
                            )
                        elif 'tflops' in part.lower():
                            metrics['tflops'] = float(
                                part.split()[2]
                            )
                        elif 'tb_per_sec' in part.lower():
                            metrics['throughput_gb_per_sec'] = float(
                                part.split()[2]
                            ) * 1000  # Convert TB/s to GB/s
                    
                    return BenchmarkResult(
                        routine=routine,
                        backend=backend,
                        batch_size=kwargs.get('batch_size', 0),
                        median_latency_ms=metrics.get('median_latency_ms', 0.0),
                        std_latency_ms=metrics.get('std_latency_ms', 0.0),
                        tflops=metrics.get('tflops', 0.0),
                        throughput_gb_per_sec=metrics.get('throughput_gb_per_sec', 0.0),
                        kv_len=kwargs.get('kv_len'),
                        num_qo_heads=kwargs.get('num_qo_heads'),
                        num_kv_heads=kwargs.get('num_kv_heads'),
                        head_dim=kwargs.get('head_dim'),
                        dtype=kwargs.get('dtype'),
                    )
                
                except (IndexError, ValueError) as e:
                    print(f"Error parsing benchmark output: {e}")
                    return None
        
        return None
    
    def run_standard_suite(
        self,
        gpu_name: Optional[str] = None,
        precision: str = "fp16"
    ) -> List[Dict[str, Any]]:
        """
        Run a standard suite of benchmarks.
        
        This suite covers common LLM inference scenarios across
        different backends and workload shapes.
        
        Args:
            gpu_name: GPU name for reporting (auto-detected if None)
            precision: Data type precision ('fp16', 'bf16', 'fp8')
        
        Returns:
            List of benchmark result dictionaries
        """
        if gpu_name is None:
            gpu_name = torch.cuda.get_device_name(0)
        
        print(f"Running standard benchmark suite on {gpu_name}...")
        
        # Define test configurations
        test_configs = [
            # Decode scenarios
            {
                "routine": "batch_decode",
                "backends": ["flashinfer", "cudnn"],
                "batch_size": 32,
                "kv_len": 1024,
                "num_qo_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
            {
                "routine": "batch_decode",
                "backends": ["flashinfer", "cudnn"],
                "batch_size": 128,
                "kv_len": 2048,
                "num_qo_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
            # Prefill scenarios
            {
                "routine": "batch_prefill",
                "backends": ["flashinfer", "cudnn"],
                "batch_size": 16,
                "kv_len": 512,
                "num_qo_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
            {
                "routine": "batch_prefill",
                "backends": ["flashinfer", "cudnn"],
                "batch_size": 8,
                "kv_len": 2048,
                "num_qo_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
        ]
        
        all_results = []
        
        for config in test_configs:
            routine = config.pop("routine")
            backends = config.pop("backends")
            
            print(f"\nRunning {routine} with {config}...")
            
            results = self.run_benchmark(
                routine=routine,
                backends=backends,
                **config
            )
            
            for result in results:
                all_results.append(result.to_dict())
        
        print(f"\n✓ Completed {len(all_results)} benchmarks")
        
        return all_results
    
    def detect_regressions(
        self,
        current_results: List[Dict[str, Any]],
        baseline: Dict[str, Any],
        threshold_percent: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Detect regressions by comparing against baseline.
        
        Args:
            current_results: Current benchmark results
            baseline: Baseline dictionary from BaselineManager
            threshold_percent: Regression threshold
        
        Returns:
            List of regression dictionaries
        """
        from .regression_detector import RegressionDetector
        
        detector = RegressionDetector(threshold_percent=threshold_percent)
        
        return detector.detect_regressions(
            current_results,
            baseline.get('results', [])
        )
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU information."""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "error": "CUDA not available"
            }
        
        try:
            device_props = torch.cuda.get_device_properties(0)
            
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "compute_capability": (
                    device_props.major,
                    device_props.minor
                ),
                "total_memory_gb": device_props.total_memory / (1024 ** 3),
                "cuda_version": torch.version.cuda,
            }
        except Exception as e:
            return {
                "available": True,
                "error": str(e)
            }
