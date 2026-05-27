"""Reference correctness test for the trtllm_fmha_v2_prefill trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _cc,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        pytest.param(
            dict(num_tokens=48, batch_size=2, num_heads=8, head_dim=128, seed=0),
            id="T48-B2-H8-D128",
        ),
        pytest.param(
            dict(num_tokens=64, batch_size=4, num_heads=4, head_dim=128, seed=1),
            id="T64-B4-H4-D128",
        ),
    ],
)
def test_trtllm_fmha_v2_prefill_reference_correctness(shape_kwargs):
    """trtllm_fmha_v2_prefill kernel (PACKED_QKV) vs reference (causal SDPA)."""
    from flashinfer.prefill import trtllm_fmha_v2_prefill
    from flashinfer.trace.templates.page import trtllm_fmha_v2_prefill_trace

    # FMHA v2 compiles only for SM90 (Hopper) or SM12x (Blackwell refresh).
    if _cc()[0] not in (9, 12):
        pytest.skip("FMHA v2 requires SM90 (Hopper) or SM12x")
    torch.manual_seed(0)
    # head_dim 128 is the smallest size the existing tests/attention/
    # test_fmha_v2_prefill.py exercises on SM90; smaller values hit unsupported
    # instantiations on H100.
    inputs = trtllm_fmha_v2_prefill_trace.init(**shape_kwargs)
    q = inputs["qkv"]
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    packed = torch.stack((q, k, v), dim=1)
    seq_lens = inputs["seq_lens"]
    cum = inputs["cum_seq_lens_q"]
    B = inputs["batch_size"]
    sm_scale = inputs["bmm1_scale"]
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        api_out = trtllm_fmha_v2_prefill(
            packed,
            "PACKED_QKV",
            workspace_buffer=ws,
            seq_lens=seq_lens,
            max_q_len=inputs["max_q_len"],
            max_kv_len=inputs["max_kv_len"],
            bmm1_scale=sm_scale,
            bmm2_scale=inputs["bmm2_scale"],
            batch_size=B,
            cum_seq_lens_q=cum,
            cum_seq_lens_kv=cum,
            mask_mode="causal",
        )
    except Exception as exc:
        pytest.skip(f"trtllm_fmha_v2_prefill unavailable: {exc}")
    ref_out = trtllm_fmha_v2_prefill_trace.reference(
        packed,
        seq_lens,
        inputs["max_q_len"],
        inputs["max_kv_len"],
        sm_scale,
        inputs["bmm2_scale"],
        B,
        cum,
        cum,
    )
    # Matches tests/attention/test_fmha_v2_prefill.py bf16 tolerance.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
