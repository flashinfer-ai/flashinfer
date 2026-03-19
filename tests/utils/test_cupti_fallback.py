"""Test that bench_gpu_time_with_cupti falls back gracefully when CUPTI is
unavailable or the driver does not support it (e.g. CUDA driver < 13.0)."""

import warnings
from unittest.mock import patch, MagicMock

import pytest
import torch

from flashinfer.testing import bench_gpu_time_with_cupti


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cupti_fallback_on_activity_enable_error():
    """When cupti.activity_enable raises (old driver), fall back to CUDA events."""
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")

    # Build a fake cupti module whose activity_enable always raises
    fake_cupti = MagicMock()
    fake_cupti.activity_enable.side_effect = Exception(
        "CUPTI_ERROR_NOT_SUPPORTED: driver too old"
    )
    fake_module = MagicMock()
    fake_module.cupti = fake_cupti

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _patched_import(name, *args, **kwargs):
        if name == "cupti":
            return fake_module
        return real_import(name, *args, **kwargs)

    # Also patch importlib.metadata.version to report a new-enough cupti-python
    with patch("importlib.metadata.version", return_value="13.0.0"), \
         patch("builtins.__import__", side_effect=_patched_import):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            times = bench_gpu_time_with_cupti(
                fn=torch.matmul,
                input_args=(a, b),
                repeat_iters=5,
                dry_run_iters=2,
                cold_l2_cache=False,
            )

    # Should have fallen back successfully
    assert isinstance(times, list)
    assert len(times) == 5
    assert all(t > 0 for t in times)

    # Should have emitted a fallback warning
    fallback_warnings = [
        w for w in caught if issubclass(w.category, UserWarning) and "Falling back" in str(w.message)
    ]
    assert len(fallback_warnings) >= 1
