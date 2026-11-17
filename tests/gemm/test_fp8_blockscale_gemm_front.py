# SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch / structure tests for the generalized fp8_blockscale_gemm front.

GPU-free: the underlying arch entry is stubbed and only the front's naming,
routing and refusal contracts are exercised. Numerics live with the
underlying entry's own suites (test_fp8bs_gemm_decodem_sm90.py and the
stock-route tests) and need SM90 silicon.
"""

import pytest
import torch

import flashinfer.gemm as fi_gemm
from flashinfer.gemm import FP8_BLOCKSCALE_GEMM_BACKENDS, fp8_blockscale_gemm


def _cpu_args():
    a = torch.zeros(4, 256, dtype=torch.bfloat16)
    b = torch.zeros(128, 256, dtype=torch.bfloat16)
    return a, b


def test_backends_tuple_is_plain_strings():
    assert FP8_BLOCKSCALE_GEMM_BACKENDS == ("auto", "sm90", "specialized")
    assert all(isinstance(b, str) for b in FP8_BLOCKSCALE_GEMM_BACKENDS)


def test_exported_next_to_the_arch_entry():
    assert "fp8_blockscale_gemm" in fi_gemm.__all__
    assert "fp8_blockscale_gemm_sm90" in fi_gemm.__all__
    assert fi_gemm.fp8_blockscale_gemm is fp8_blockscale_gemm


def test_unknown_backend_raises_and_never_swaps():
    a, b = _cpu_args()
    with pytest.raises(ValueError, match="never silently swapped"):
        fp8_blockscale_gemm(a, b, backend="cutlass")


def test_non_sm90_device_refused_with_contract_pointer(monkeypatch):
    import flashinfer.gemm.gemm_base as gemm_base

    monkeypatch.setattr(gemm_base, "_match_sm_version", lambda dev, vers: False)
    a, b = _cpu_args()
    with pytest.raises(ValueError, match="gemm_fp8_nt_blockscaled"):
        fp8_blockscale_gemm(a, b, backend="auto")


@pytest.mark.parametrize(
    "front_backend,inner_backend",
    [("auto", "auto"), ("sm90", "stock"), ("specialized", "specialized")],
)
def test_backend_mapping_to_arch_entry(monkeypatch, front_backend, inner_backend):
    import flashinfer.gemm.gemm_base as gemm_base

    seen = {}

    def fake_sm90(input, weight, input_scale, weight_scale, out, out_dtype, *, backend):
        seen.update(
            input=input,
            weight=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            out=out,
            out_dtype=out_dtype,
            backend=backend,
        )
        return input

    monkeypatch.setattr(gemm_base, "_match_sm_version", lambda dev, vers: True)
    monkeypatch.setattr(gemm_base, "fp8_blockscale_gemm_sm90", fake_sm90)
    a, b = _cpu_args()
    ws = torch.zeros(1, 2, dtype=torch.float32)
    fp8_blockscale_gemm(a, b, None, ws, out_dtype=torch.bfloat16, backend=front_backend)
    assert seen["backend"] == inner_backend
    assert seen["input"] is a
    assert seen["weight"] is b
    assert seen["input_scale"] is None
    assert seen["weight_scale"] is ws
    assert seen["out"] is None
    assert seen["out_dtype"] is torch.bfloat16
