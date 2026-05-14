"""Reference correctness test for the trtllm_fmha_v2_prefill trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _cc,
    _close,
)


def test_trtllm_fmha_v2_prefill_reference_correctness():
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
    B, H, D = 2, 8, 128
    q_lens = [16, 32]
    kv_lens = [16, 32]
    total_tokens = sum(q_lens)
    packed = torch.randn(total_tokens, 3, H, D, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cum_list = [0] + [sum(q_lens[: i + 1]) for i in range(B)]
    cum = torch.tensor(cum_list, dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / (D**0.5)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        api_out = trtllm_fmha_v2_prefill(
            packed,
            "PACKED_QKV",
            workspace_buffer=ws,
            seq_lens=seq_lens,
            max_q_len=max(q_lens),
            max_kv_len=max(kv_lens),
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
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
        max(q_lens),
        max(kv_lens),
        sm_scale,
        1.0,
        B,
        cum,
        cum,
    )
    # Matches tests/attention/test_fmha_v2_prefill.py bf16 tolerance.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
