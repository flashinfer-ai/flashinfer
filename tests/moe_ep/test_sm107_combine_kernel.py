"""Native Rubin correctness for repeated launches, masked routes, and zero input."""

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize("early", [True, False])
@pytest.mark.parametrize("combine_dtype", ["nvfp4", "mxfp8"])
def test_combine_repeated_launches(monkeypatch, kind, early, combine_dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    from tests.moe_ep.test_sm107_kernel_boundaries import _run_case

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run_case(kind, 5, 3, dict(combine_dtype=combine_dtype, apply_topk_at_fc1=early))
