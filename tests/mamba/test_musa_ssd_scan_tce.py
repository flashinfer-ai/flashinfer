import os

import pytest
import torch

from flashinfer.mamba.musa_ssd_scan_tce import (
    musa_ssd_chunk_scan_tce_native,
    preload_musa_ssd_chunk_scan_tce,
)


_DEVICE = os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skipif(
    _DEVICE != "musa" or getattr(torch.version, "musa", None) is None,
    reason="requires a MUSA runtime",
)


def _reference(state, x, dt, A, B, C, D):
    # The dashboard candidate receives the already transformed positive dt
    # used by the SSD scan stage; INCUM=3 performs its prefix sum in-kernel.
    dtp = dt
    cumsum = torch.cumsum(dtp, dim=0)
    decay = torch.exp(A[None, :] * cumsum)
    weighted_x = x.float() * dtp[:, :, None] / decay[:, :, None]
    cb = torch.einsum("tgn,sgn->gts", C.float(), B.float())
    causal = torch.tril(
        torch.ones(128, 128, device=x.device, dtype=torch.bool)
    )
    cb = cb * causal[None]
    groups = B.shape[1]
    rep = x.shape[1] // groups
    z_grouped = weighted_x.reshape(128, groups, rep * x.shape[2]).transpose(0, 1)
    y = torch.bmm(cb, z_grouped).reshape(groups, 128, rep, x.shape[2])
    y = y.transpose(0, 1).reshape_as(x).float()
    c_expanded = C.float().repeat_interleave(rep, dim=1).transpose(0, 1)
    y = y + torch.bmm(c_expanded, state.transpose(1, 2)).transpose(0, 1)
    y = y * decay[:, :, None] + x.float() * D[None, :, None]
    return y.to(torch.bfloat16)


def test_exact_shape_tce_scan_matches_reference():
    torch.manual_seed(29)
    state = torch.randn(64, 64, 128, device=_DEVICE, dtype=torch.float32) * 0.1
    x = torch.randn(128, 64, 64, device=_DEVICE, dtype=torch.bfloat16) * 0.1
    dt = torch.rand(128, 64, device=_DEVICE, dtype=torch.float32) * 0.1
    A = -torch.rand(64, device=_DEVICE, dtype=torch.float32) * 0.05
    B = torch.randn(128, 8, 128, device=_DEVICE, dtype=torch.bfloat16) * 0.1
    C = torch.randn(128, 8, 128, device=_DEVICE, dtype=torch.bfloat16) * 0.1
    D = torch.randn(64, device=_DEVICE, dtype=torch.float32) * 0.1

    preload_musa_ssd_chunk_scan_tce()
    actual = musa_ssd_chunk_scan_tce_native(state, x, dt, A, B, C, D)
    expected = _reference(state, x, dt, A, B, C, D)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
