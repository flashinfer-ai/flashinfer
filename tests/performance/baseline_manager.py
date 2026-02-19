"""Baseline performance management for FlashInfer."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib


class BaselineManager:
    """
    Manage performance baselines for regression testing.
    
    Baselines are stored as JSON files containing benchmark results
    for specific configurations (GPU, CUDA version, workload parameters).
    """
    
    def __init__(self, baseline_dir: Optional[Path] = None):
        """
        Initialize baseline manager.
        
        Args:
            baseline_dir: Directory to store baseline files.
                         Defaults to tests/performance/baselines/
        """
        if baseline_dir is None:
            baseline_dir = Path(__file__).parent / "baselines"
        
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
    
    def save_baseline(
        self,
        results: List[Dict[str, Any]],
        gpu_name: str,
        cuda_version: str,
        flashinfer_version: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save benchmark results as a baseline.
        
        Args:
            results: List of benchmark result dictionaries
            gpu_name: GPU model name (e.g., "H100", "A100")
            cuda_version: CUDA version (e.g., "12.6")
            flashinfer_version: FlashInfer version
            name: Optional custom baseline name
            metadata: Additional metadata to store
        
        Returns:
            Path to the saved baseline file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if name is None:
            name = f"{gpu_name}_cuda{cuda_version}_{timestamp}"
        
        baseline = {
            "name": name,
            "timestamp": timestamp,
            "gpu_name": gpu_name,
            "cuda_version": cuda_version,
            "flashinfer_version": flashinfer_version,
            "metadata": metadata or {},
            "results": results
        }
        
        baseline_path = self.baseline_dir / f"{name}.json"
        
        with open(baseline_path, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"✓ Saved baseline to {baseline_path}")
        return baseline_path
    
    def load_baseline(self, name: str) -> Dict[str, Any]:
        """
        Load a baseline by name.
        
        Args:
            name: Baseline name (without .json extension)
        
        Returns:
            Baseline dictionary
        """
        baseline_path = self.baseline_dir / f"{name}.json"
        
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline not found: {baseline_path}")
        
        with open(baseline_path, 'r') as f:
            return json.load(f)
    
    def list_baselines(
        self,
        gpu_name: Optional[str] = None,
        cuda_version: Optional[str] = None
    ) -> List[str]:
        """
        List available baselines.
        
        Args:
            gpu_name: Filter by GPU name
            cuda_version: Filter by CUDA version
        
        Returns:
            List of baseline names
        """
        baselines = []
        
        for path in self.baseline_dir.glob("*.json"):
            try:
                with open(path, 'r') as f:
                    baseline = json.load(f)
                
                # Apply filters
                if gpu_name and baseline.get("gpu_name") != gpu_name:
                    continue
                if cuda_version and baseline.get("cuda_version") != cuda_version:
                    continue
                
                baselines.append(path.stem)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        return sorted(baselines)
    
    def get_latest_baseline(
        self,
        gpu_name: str,
        cuda_version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent baseline for a GPU.
        
        Args:
            gpu_name: GPU model name
            cuda_version: Optional CUDA version filter
        
        Returns:
            Latest baseline dictionary, or None if not found
        """
        baselines = self.list_baselines(gpu_name, cuda_version)
        
        if not baselines:
            return None
        
        # Load all baselines and sort by timestamp
        loaded_baselines = []
        for name in baselines:
            try:
                baseline = self.load_baseline(name)
                loaded_baselines.append(baseline)
            except Exception:
                continue
        
        if not loaded_baselines:
            return None
        
        # Sort by timestamp (newest first)
        loaded_baselines.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return loaded_baselines[0]
    
    def compute_baseline_hash(self, test_config: Dict[str, Any]) -> str:
        """
        Compute a hash for a test configuration.
        
        This hash is used to match current results with baseline results
        for the same test configuration.
        
        Args:
            test_config: Dictionary with test parameters
                        (routine, batch_size, kv_len, etc.)
        
        Returns:
            Hash string
        """
        # Sort keys for consistent hashing
        config_str = json.dumps(test_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def export_baseline_report(
        self,
        baseline: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a human-readable report for a baseline.
        
        Args:
            baseline: Baseline dictionary
            output_path: Optional path to save report
        
        Returns:
            Report string
        """
        report_lines = [
            f"# FlashInfer Performance Baseline Report",
            f"",
            f"**Name:** {baseline['name']}",
            f"**Timestamp:** {baseline['timestamp']}",
            f"**GPU:** {baseline['gpu_name']}",
            f"**CUDA Version:** {baseline['cuda_version']}",
            f"**FlashInfer Version:** {baseline['flashinfer_version']}",
            f"",
            f"## Benchmark Results",
            f"",
            f"| Test | Backend | Median Latency (ms) | Throughput (TFLOPS) |",
            f"|------|---------|--------------------|--------------------|"
        ]
        
        for result in baseline['results']:
            test_name = result.get('test_name', 'Unknown')
            backend = result.get('backend', 'N/A')
            latency = result.get('median_latency_ms', 0)
            tflops = result.get('tflops', 0)
            
            report_lines.append(
                f"| {test_name} | {backend} | {latency:.3f} | {tflops:.2f} |"
            )
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"✓ Exported report to {output_path}")
        
        return report
