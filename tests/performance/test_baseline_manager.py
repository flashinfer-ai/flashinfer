"""Tests for performance baseline manager."""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance.baseline_manager import BaselineManager


@pytest.fixture
def temp_baseline_dir():
    """Create a temporary directory for baseline files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_results():
    """Sample benchmark results."""
    return [
        {
            "test_name": "batch_decode_h100",
            "routine": "batch_decode",
            "backend": "flashinfer",
            "batch_size": 128,
            "kv_len": 2048,
            "num_qo_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "median_latency_ms": 0.285,
            "tflops": 13.18,
        },
        {
            "test_name": "batch_prefill_h100",
            "routine": "batch_prefill",
            "backend": "cudnn",
            "batch_size": 16,
            "kv_len": 512,
            "num_qo_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "median_latency_ms": 1.52,
            "tflops": 45.3,
        }
    ]


class TestBaselineManager:
    """Test baseline management functionality."""
    
    def test_save_and_load_baseline(self, temp_baseline_dir, sample_results):
        """Test saving and loading a baseline."""
        manager = BaselineManager(temp_baseline_dir)
        
        # Save baseline
        baseline_path = manager.save_baseline(
            results=sample_results,
            gpu_name="H100",
            cuda_version="12.6",
            flashinfer_version="0.4.0",
            name="test_baseline"
        )
        
        assert baseline_path.exists()
        
        # Load baseline
        loaded = manager.load_baseline("test_baseline")
        
        assert loaded["gpu_name"] == "H100"
        assert loaded["cuda_version"] == "12.6"
        assert loaded["flashinfer_version"] == "0.4.0"
        assert len(loaded["results"]) == 2
        assert loaded["results"][0]["routine"] == "batch_decode"
    
    def test_list_baselines(self, temp_baseline_dir, sample_results):
        """Test listing baselines with filters."""
        manager = BaselineManager(temp_baseline_dir)
        
        # Save multiple baselines
        manager.save_baseline(
            results=sample_results,
            gpu_name="H100",
            cuda_version="12.6",
            flashinfer_version="0.4.0",
            name="h100_baseline_1"
        )
        
        manager.save_baseline(
            results=sample_results,
            gpu_name="A100",
            cuda_version="12.6",
            flashinfer_version="0.4.0",
            name="a100_baseline_1"
        )
        
        # List all
        all_baselines = manager.list_baselines()
        assert len(all_baselines) == 2
        
        # Filter by GPU
        h100_baselines = manager.list_baselines(gpu_name="H100")
        assert len(h100_baselines) == 1
        assert "h100_baseline_1" in h100_baselines
        
        a100_baselines = manager.list_baselines(gpu_name="A100")
        assert len(a100_baselines) == 1
        assert "a100_baseline_1" in a100_baselines
    
    def test_get_latest_baseline(self, temp_baseline_dir, sample_results):
        """Test getting the latest baseline."""
        manager = BaselineManager(temp_baseline_dir)
        
        # Save baselines with different timestamps
        import time
        
        manager.save_baseline(
            results=sample_results,
            gpu_name="H100",
            cuda_version="12.6",
            flashinfer_version="0.4.0",
            name="h100_old"
        )
        
        time.sleep(0.1)  # Ensure different timestamps
        
        manager.save_baseline(
            results=sample_results,
            gpu_name="H100",
            cuda_version="12.6",
            flashinfer_version="0.4.1",
            name="h100_new"
        )
        
        # Get latest
        latest = manager.get_latest_baseline("H100")
        
        assert latest is not None
        assert latest["flashinfer_version"] == "0.4.1"
    
    def test_compute_baseline_hash(self, temp_baseline_dir):
        """Test config hashing for baseline matching."""
        manager = BaselineManager(temp_baseline_dir)
        
        config1 = {
            "routine": "batch_decode",
            "batch_size": 128,
            "kv_len": 2048
        }
        
        config2 = {
            "routine": "batch_decode",
            "batch_size": 128,
            "kv_len": 2048
        }
        
        config3 = {
            "routine": "batch_decode",
            "batch_size": 128,
            "kv_len": 4096  # Different
        }
        
        hash1 = manager.compute_baseline_hash(config1)
        hash2 = manager.compute_baseline_hash(config2)
        hash3 = manager.compute_baseline_hash(config3)
        
        assert hash1 == hash2  # Same config = same hash
        assert hash1 != hash3  # Different config = different hash
    
    def test_export_baseline_report(self, temp_baseline_dir, sample_results):
        """Test baseline report export."""
        manager = BaselineManager(temp_baseline_dir)
        
        manager.save_baseline(
            results=sample_results,
            gpu_name="H100",
            cuda_version="12.6",
            flashinfer_version="0.4.0",
            name="test_report"
        )
        
        baseline = manager.load_baseline("test_report")
        report = manager.export_baseline_report(baseline)
        
        assert "FlashInfer Performance Baseline Report" in report
        assert "H100" in report
        assert "batch_decode" in report
        assert "batch_prefill" in report
