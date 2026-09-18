import os

import pytest
import torch

from flashinfer.mamba.musa_ssd_chunk_state import _DT_MAX, _chunk_cumsum_fwd
from flashinfer.mamba.musa_ssd_cumsum_native import (
    musa_ssd_chunk_cumsum_native,
    preload_musa_ssd_chunk_cumsum,
)


_DEVICE = os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skipif(
    _DEVICE != "musa" or getattr(torch.version, "musa", None) is None,
    reason="requires a MUSA runtime",
)


def _inputs(tokens: int):
    torch.manual_seed(17)
    dt = torch.randn(tokens, 64, device=_DEVICE, dtype=torch.float32) * 0.1
    A = torch.randn(64, device=_DEVICE, dtype=torch.float32) * 0.1
    bias = torch.randn(64, device=_DEVICE, dtype=torch.float32) * 0.1
    cu_chunks = torch.arange(
        0, tokens + 1, 128, device=_DEVICE, dtype=torch.int32
    )
    return dt, A, bias, cu_chunks


@pytest.mark.parametrize("tokens", [128, 1024, 4096])
def test_native_cumsum_matches_triton(tokens):
    dt, A, bias, cu_chunks = _inputs(tokens)
    preload_musa_ssd_chunk_cumsum()
    actual = musa_ssd_chunk_cumsum_native(dt, A, bias)
    expected = _chunk_cumsum_fwd(
        dt,
        A,
        128,
        cu_chunks,
        dt_bias=bias,
        dt_softplus=True,
        dt_limit=(0.0, _DT_MAX),
        regular_full_chunks=False,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-6, atol=2e-6)


def test_regular_full_chunk_dispatch_matches_triton(monkeypatch):
    dt, A, bias, cu_chunks = _inputs(4096)
    actual = _chunk_cumsum_fwd(
        dt,
        A,
        128,
        cu_chunks,
        dt_bias=bias,
        dt_softplus=True,
        dt_limit=(0.0, _DT_MAX),
        regular_full_chunks=True,
    )
    monkeypatch.setenv("FLASHINFER_MUSA_SSD_CUMSUM_DISABLE_FAST", "1")
    expected = _chunk_cumsum_fwd(
        dt,
        A,
        128,
        cu_chunks,
        dt_bias=bias,
        dt_softplus=True,
        dt_limit=(0.0, _DT_MAX),
        regular_full_chunks=True,
    )
    torch.musa.synchronize()
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-5, atol=2e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-6, atol=2e-6)


def test_non_softplus_contract_uses_triton(monkeypatch):
    dt, A, bias, cu_chunks = _inputs(128)
    monkeypatch.setattr(
        "flashinfer.mamba.musa_ssd_cumsum_native.musa_ssd_chunk_cumsum_native",
        lambda *args: pytest.fail("native path must not handle dt_softplus=False"),
    )
    result = _chunk_cumsum_fwd(
        dt,
        A,
        128,
        cu_chunks,
        dt_bias=bias,
        dt_softplus=False,
        dt_limit=(0.0, _DT_MAX),
        regular_full_chunks=True,
    )
    assert result[0].shape == (64, 1, 128)
