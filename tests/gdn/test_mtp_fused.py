"""Correctness test for the fused MTP kernel.

Reference: run SEQ kernel twice — first over the K accepted tokens to update
state h0 → h_K, then over T new tokens with h_K as initial state to compute
outputs.  Fused kernel should produce the same outputs + state.

Tolerance: FlashInfer standard BF16 gate (atol=rtol=5e-3).
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
    gated_delta_rule_mtp as seq_kernel,
)
from flashinfer.gdn_kernels.gdn_decode_wy_triton_mtp_fused import (
    gated_delta_rule_mtp_fused,
)
from flashinfer.gdn_kernels.gdn_decode_wy_triton_mtp_split import (
    gated_delta_rule_mtp_split,
)


def _make_inputs(B, T, K_MAX, H, HV, D=128, seed=42):
    torch.manual_seed(seed)
    dt = torch.bfloat16
    dev = "cuda"
    return dict(
        k_acc=torch.randn(B, K_MAX, H, D, dtype=dt, device=dev),
        v_acc=torch.randn(B, K_MAX, HV, D, dtype=dt, device=dev),
        a_acc=torch.randn(B, K_MAX, HV, dtype=dt, device=dev),
        b_acc=torch.randn(B, K_MAX, HV, dtype=dt, device=dev),
        q_new=torch.randn(B, T, H, D, dtype=dt, device=dev),
        k_new=torch.randn(B, T, H, D, dtype=dt, device=dev),
        v_new=torch.randn(B, T, HV, D, dtype=dt, device=dev),
        a_new=torch.randn(B, T, HV, dtype=dt, device=dev),
        b_new=torch.randn(B, T, HV, dtype=dt, device=dev),
        A_log=torch.randn(HV, dtype=torch.float32, device=dev) * 0.1,
        dt_bias=torch.randn(HV, dtype=torch.float32, device=dev) * 0.1,
        state=torch.randn(B, HV, D, D, dtype=dt, device=dev) * 0.01,
    )


def _reference(ins, num_accepted, scale):
    """Two SEQ calls: first K_i per batch to get h_K, then T new tokens."""
    B, T, H, D = ins["q_new"].shape
    HV = ins["v_new"].shape[2]
    dev = "cuda"
    idx = torch.arange(B, dtype=torch.int32, device=dev)

    # ------- Phase A reference: per-batch SEQ over first K_i accepted tokens -------
    state = ins["state"].clone()
    for bi in range(B):
        K_i = int(num_accepted[bi].item())
        if K_i == 0:
            continue
        # Slice the accepted tokens for batch bi
        q_dummy = torch.zeros(1, K_i, H, D, dtype=torch.bfloat16, device=dev)
        k_bi = ins["k_acc"][bi : bi + 1, :K_i]
        v_bi = ins["v_acc"][bi : bi + 1, :K_i]
        a_bi = ins["a_acc"][bi : bi + 1, :K_i]
        b_bi = ins["b_acc"][bi : bi + 1, :K_i]
        state_bi = state[bi : bi + 1]
        dummy_out = torch.empty(1, K_i, HV, D, dtype=torch.bfloat16, device=dev)
        seq_kernel(
            A_log=ins["A_log"],
            a=a_bi,
            dt_bias=ins["dt_bias"],
            q=q_dummy,
            k=k_bi,
            v=v_bi,
            b=b_bi,
            initial_state_source=state_bi,
            initial_state_indices=torch.zeros(1, dtype=torch.int32, device=dev),
            disable_state_update=False,
            use_qk_l2norm_in_kernel=True,
            scale=scale,
            output=dummy_out,
        )
        state[bi : bi + 1] = (
            state_bi  # already in-place updated but re-assign for clarity
        )

    # ------- Phase B reference: SEQ over T new tokens with h_K initial state -------
    out_ref = torch.empty(B, T, HV, D, dtype=torch.bfloat16, device=dev)
    state_kept = state.clone()  # we want to keep h_K after the call
    seq_kernel(
        A_log=ins["A_log"],
        a=ins["a_new"],
        dt_bias=ins["dt_bias"],
        q=ins["q_new"],
        k=ins["k_new"],
        v=ins["v_new"],
        b=ins["b_new"],
        initial_state_source=state,  # state reads as h_K
        initial_state_indices=idx,
        disable_state_update=True,  # don't overwrite with h_{K+T}
        use_qk_l2norm_in_kernel=True,
        scale=scale,
        output=out_ref,
    )
    return out_ref, state_kept


@pytest.mark.parametrize("impl", ["fused", "split"])
@pytest.mark.parametrize("preset", ["qwen3.5", "qwen3-next"])
@pytest.mark.parametrize("K_MAX", [4, 8, 16])
@pytest.mark.parametrize("T", [4, 8, 16])
@pytest.mark.parametrize("B", [2, 8])
def test_fused_matches_two_seq_calls(impl, preset, K_MAX, T, B):
    H = 16
    HV = 32 if preset == "qwen3-next" else 64
    D = 128
    ins = _make_inputs(B, T, K_MAX, H, HV, D)
    scale = 1.0 / math.sqrt(D)
    dev = "cuda"

    # Random num_accepted per batch, in [0, K_MAX]
    num_accepted = torch.randint(0, K_MAX + 1, (B,), dtype=torch.int32, device=dev)

    out_ref, state_ref = _reference(ins, num_accepted, scale)

    # Run fused/split kernel on a fresh copy of state
    ins_tri = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in ins.items()
    }
    kernel_fn = (
        gated_delta_rule_mtp_fused if impl == "fused" else gated_delta_rule_mtp_split
    )
    out_tri = kernel_fn(
        k_accepted=ins_tri["k_acc"],
        v_accepted=ins_tri["v_acc"],
        a_accepted=ins_tri["a_acc"],
        b_accepted=ins_tri["b_acc"],
        num_accepted=num_accepted,
        q_new=ins_tri["q_new"],
        k_new=ins_tri["k_new"],
        v_new=ins_tri["v_new"],
        a_new=ins_tri["a_new"],
        b_new=ins_tri["b_new"],
        A_log=ins_tri["A_log"],
        dt_bias=ins_tri["dt_bias"],
        initial_state_source=ins_tri["state"],
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    state_tri = ins_tri["state"]

    # Both outputs must match at BF16 precision
    torch.testing.assert_close(out_tri.float(), out_ref.float(), atol=5e-3, rtol=5e-3)

    # State tensor must be h_K (not h_{K+T})
    torch.testing.assert_close(
        state_tri.float(), state_ref.float(), atol=5e-3, rtol=5e-3
    )


def test_zero_accepted_behaves_like_pure_output():
    """K=0 everywhere → state unchanged; outputs match output-only kernel."""
    B, T, K_MAX, H, HV = 4, 8, 8, 16, 64
    ins = _make_inputs(B, T, K_MAX, H, HV)
    scale = 1.0 / math.sqrt(128)
    dev = "cuda"

    num_accepted = torch.zeros(B, dtype=torch.int32, device=dev)

    # Fused kernel
    ins_tri = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in ins.items()
    }
    state_before = ins_tri["state"].clone()
    out_tri = gated_delta_rule_mtp_fused(
        k_accepted=ins_tri["k_acc"],
        v_accepted=ins_tri["v_acc"],
        a_accepted=ins_tri["a_acc"],
        b_accepted=ins_tri["b_acc"],
        num_accepted=num_accepted,
        q_new=ins_tri["q_new"],
        k_new=ins_tri["k_new"],
        v_new=ins_tri["v_new"],
        a_new=ins_tri["a_new"],
        b_new=ins_tri["b_new"],
        A_log=ins_tri["A_log"],
        dt_bias=ins_tri["dt_bias"],
        initial_state_source=ins_tri["state"],
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )
    # State must be unchanged (within bf16 round-trip noise)
    torch.testing.assert_close(
        ins_tri["state"].float(), state_before.float(), atol=5e-3, rtol=5e-3
    )

    # Reference via pure output kernel (no state update)
    out_ref = torch.empty_like(out_tri)
    seq_kernel(
        A_log=ins["A_log"],
        a=ins["a_new"],
        dt_bias=ins["dt_bias"],
        q=ins["q_new"],
        k=ins["k_new"],
        v=ins["v_new"],
        b=ins["b_new"],
        initial_state_source=state_before,
        initial_state_indices=torch.arange(B, dtype=torch.int32, device=dev),
        disable_state_update=True,
        use_qk_l2norm_in_kernel=True,
        scale=scale,
        output=out_ref,
    )
    torch.testing.assert_close(out_tri.float(), out_ref.float(), atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
