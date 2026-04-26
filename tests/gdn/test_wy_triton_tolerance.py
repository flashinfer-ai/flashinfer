"""Pytest covering the Triton WY kernel against the FP32 reference using
the FlashInfer-standard BF16 tolerance (atol=5e-3, rtol=5e-3 — matches
tests/gdn/test_decode_delta_rule.py). Also runs a tighter 1e-3/1e-3 pass.

Runs on qwen3.5 (HV=64) and qwen3-next (HV=32), several T and BS, and
with state-update on/off.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from reference_delta_rule import verify_delta_rule  # noqa: E402
from flashinfer.gdn_kernels.gdn_decode_wy_triton import (  # noqa: E402
    gated_delta_rule_mtp_wy_triton as triton_wy,
)


# FlashInfer convention for BF16 GDN output (see test_decode_delta_rule.py)
BF16_OUTPUT_ATOL = 5e-3
BF16_OUTPUT_RTOL = 5e-3
BF16_STATE_ATOL = 5e-3
BF16_STATE_RTOL = 5e-3


def _inputs(B, T, H, HV, D=128, seed=42):
    torch.manual_seed(seed)
    dt = torch.bfloat16
    dev = "cuda"
    return dict(
        q=torch.randn(B, T, H, D, dtype=dt, device=dev),
        k=torch.randn(B, T, H, D, dtype=dt, device=dev),
        v=torch.randn(B, T, HV, D, dtype=dt, device=dev),
        a=torch.randn(B, T, HV, dtype=dt, device=dev),
        b=torch.randn(B, T, HV, dtype=dt, device=dev),
        A_log=torch.randn(HV, dtype=torch.float32, device=dev) * 0.1,
        dt_bias=torch.randn(HV, dtype=torch.float32, device=dev) * 0.1,
        state_vk=torch.randn(B, HV, D, D, dtype=dt, device=dev) * 0.01,
        D=D,
    )


@pytest.mark.parametrize("preset", ["qwen3-next", "qwen3.5"])
@pytest.mark.parametrize("T", [2, 4, 8, 16])
@pytest.mark.parametrize("B", [1, 4, 16])
def test_output_tolerance(preset, T, B):
    """Triton WY output must pass torch.testing.assert_close at the
    FlashInfer-standard BF16 gate (5e-3/5e-3) vs the FP32 reference."""
    H = 16
    HV = 32 if preset == "qwen3-next" else 64
    ins = _inputs(B, T, H, HV)
    D = ins["D"]
    scale = 1.0 / math.sqrt(D)
    state_kv = ins["state_vk"].float().permute(0, 1, 3, 2).contiguous()
    idx = torch.arange(B, dtype=torch.int32, device="cuda")

    ref_out, _, _ = verify_delta_rule(
        ins["q"].float(),
        ins["k"].float(),
        ins["v"].float(),
        state_kv,
        ins["A_log"],
        ins["a"].float(),
        ins["dt_bias"],
        ins["b"].float(),
        scale_factor=scale,
        use_l2_norm=True,
        state_dtype=torch.float32,
    )
    tri_out = triton_wy(
        A_log=ins["A_log"],
        a=ins["a"],
        dt_bias=ins["dt_bias"],
        q=ins["q"],
        k=ins["k"],
        v=ins["v"],
        b=ins["b"],
        initial_state_source=ins["state_vk"].clone(),
        initial_state_indices=idx,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )
    torch.testing.assert_close(
        tri_out.float(),
        ref_out.float(),
        atol=BF16_OUTPUT_ATOL,
        rtol=BF16_OUTPUT_RTOL,
    )


@pytest.mark.parametrize("preset", ["qwen3-next", "qwen3.5"])
@pytest.mark.parametrize("T", [4, 8, 16])
@pytest.mark.parametrize("B", [4, 16])
def test_state_update_tolerance(preset, T, B):
    """State update (h_new) must pass the same 5e-3/5e-3 gate."""
    H = 16
    HV = 32 if preset == "qwen3-next" else 64
    ins = _inputs(B, T, H, HV)
    D = ins["D"]
    scale = 1.0 / math.sqrt(D)
    state_kv = ins["state_vk"].float().permute(0, 1, 3, 2).contiguous()
    idx = torch.arange(B, dtype=torch.int32, device="cuda")

    ref_out, ref_state, _ = verify_delta_rule(
        ins["q"].float(),
        ins["k"].float(),
        ins["v"].float(),
        state_kv,
        ins["A_log"],
        ins["a"].float(),
        ins["dt_bias"],
        ins["b"].float(),
        scale_factor=scale,
        use_l2_norm=True,
        state_dtype=torch.float32,
    )
    state_tri = ins["state_vk"].clone()
    tri_out = triton_wy(
        A_log=ins["A_log"],
        a=ins["a"],
        dt_bias=ins["dt_bias"],
        q=ins["q"],
        k=ins["k"],
        v=ins["v"],
        b=ins["b"],
        initial_state_source=state_tri,
        initial_state_indices=idx,
        disable_state_update=False,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )
    torch.testing.assert_close(
        tri_out.float(),
        ref_out.float(),
        atol=BF16_OUTPUT_ATOL,
        rtol=BF16_OUTPUT_RTOL,
    )
    # Triton writes state as [B, HV, V, K]; reference is [B, HV, K, V]
    tri_state_kv = state_tri.float().permute(0, 1, 3, 2).contiguous()
    torch.testing.assert_close(
        tri_state_kv,
        ref_state.float(),
        atol=BF16_STATE_ATOL,
        rtol=BF16_STATE_RTOL,
    )


@pytest.mark.parametrize("T", [4, 8, 16])
def test_tight_tolerance(T):
    """Triton also passes a 5× tighter BF16 gate (1e-3/1e-3) on qwen3.5 BS=16,
    confirming we have substantial headroom over the FlashInfer standard."""
    B, H, HV = 16, 16, 64
    ins = _inputs(B, T, H, HV)
    D = ins["D"]
    scale = 1.0 / math.sqrt(D)
    state_kv = ins["state_vk"].float().permute(0, 1, 3, 2).contiguous()
    idx = torch.arange(B, dtype=torch.int32, device="cuda")

    ref_out, _, _ = verify_delta_rule(
        ins["q"].float(),
        ins["k"].float(),
        ins["v"].float(),
        state_kv,
        ins["A_log"],
        ins["a"].float(),
        ins["dt_bias"],
        ins["b"].float(),
        scale_factor=scale,
        use_l2_norm=True,
        state_dtype=torch.float32,
    )
    tri_out = triton_wy(
        A_log=ins["A_log"],
        a=ins["a"],
        dt_bias=ins["dt_bias"],
        q=ins["q"],
        k=ins["k"],
        v=ins["v"],
        b=ins["b"],
        initial_state_source=ins["state_vk"].clone(),
        initial_state_indices=idx,
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
    )
    torch.testing.assert_close(
        tri_out.float(),
        ref_out.float(),
        atol=1e-3,
        rtol=1e-3,
    )
