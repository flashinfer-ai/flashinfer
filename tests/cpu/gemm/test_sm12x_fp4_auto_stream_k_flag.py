"""CPU checks: FLASHINFER_FP4_GEMM_DISABLE_STREAM_K narrows mm_fp4 auto on SM12x."""

import pytest
import torch
from packaging.version import Version

import flashinfer.gemm.gemm_base as gemm_base

FLAG = "FLASHINFER_FP4_GEMM_DISABLE_STREAM_K"
ALL = ["cudnn", "cutlass", "b12x"]


def _auto(monkeypatch, cc, cuda="13.0", suitable=ALL, flag=None, use_nvfp4=True):
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    monkeypatch.setattr(gemm_base, "get_cuda_version", lambda *a, **k: Version(cuda))
    monkeypatch.setattr(gemm_base, "get_compute_capability", lambda device: cc)
    # Mock cuDNN as unavailable, so the result does not depend on the test host.
    monkeypatch.setattr(gemm_base, "CUDNN_AVAILABLE", False)
    a = torch.empty(16, 64, dtype=torch.uint8)
    b = torch.empty(64, 128, dtype=torch.uint8)
    return gemm_base._heuristic_func_mm_fp4(
        list(suitable), a, b, None, None, backend="auto", use_nvfp4=use_nvfp4
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
def test_flag_set_restricts_auto_to_cutlass_on_sm12x(monkeypatch, cc):
    # b12x and cuDNN are not covered by the flag, and their results can differ
    # from CUTLASS in the last bits. So auto must offer CUTLASS alone.
    assert _auto(monkeypatch, cc, flag="1") == ["cutlass"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
@pytest.mark.parametrize("flag", ["0", "true"])
def test_flag_not_set_keeps_auto_unchanged(monkeypatch, cc, flag):
    # Only "1" turns the flag on, so "0" and "true" must match an unset flag.
    # Comparing with the unset result avoids hard-coding auto's default order.
    assert _auto(monkeypatch, cc, flag=flag) == _auto(monkeypatch, cc, flag=None)


@pytest.mark.cpu_only
@pytest.mark.parametrize("cc", [(12, 0), (12, 1)])
def test_flag_set_without_cutlass_keeps_auto_unchanged(monkeypatch, cc):
    # With MXFP4, CUTLASS cannot run the call, so it is not in the `suitable`
    # list. The flag must then leave auto as it is, not return an empty list.
    kw = dict(suitable=["cudnn", "b12x"], use_nvfp4=False)
    got = _auto(monkeypatch, cc, flag="1", **kw)
    assert got and got == _auto(monkeypatch, cc, flag=None, **kw)


@pytest.mark.cpu_only
@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_flag_set_does_not_touch_other_archs(monkeypatch, cc):
    with_flag = _auto(monkeypatch, cc, flag="1", suitable=["cudnn", "cutlass"])
    without = _auto(monkeypatch, cc, flag=None, suitable=["cudnn", "cutlass"])
    assert with_flag == without
