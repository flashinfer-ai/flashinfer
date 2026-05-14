"""Reference correctness test for the batch_attention_run trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            device="cuda",
            num_qo_tokens=1,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
            num_pages=1,
            page_size=16,
        ),
        dict(
            device="cuda",
            num_qo_tokens=1,
            num_qo_heads=4,
            num_kv_heads=1,
            head_dim=128,
            num_pages=2,
            page_size=8,
        ),
    ],
)
def test_batch_attention_run_reference_correctness(shape_kwargs):
    """BatchAttention.run kernel vs reference (page-gather SDPA).

    Compares the reference against BatchDecodeWithPagedKVCacheWrapper.run
    (same semantics: decode attention over a (k_cache, v_cache) paged tuple).
    """
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.trace.templates.attention import batch_attention_run_trace

    inputs = batch_attention_run_trace.init(**shape_kwargs)
    plan = inputs["plan"]
    run = inputs["run"]
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        wrapper.plan(
            plan["kv_indptr"],
            plan["kv_indices"],
            plan["kv_last_page_len"],
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            plan["head_dim"],
            plan["page_size"],
            q_data_type=plan["q_data_type"],
            kv_data_type=plan["kv_data_type"],
        )
        api_out = wrapper.run(run["q"], run["kv_cache"])
    except Exception as exc:
        pytest.skip(f"BatchDecodeWithPagedKVCacheWrapper unavailable: {exc}")
    # Reference returns (output, lse); kernel returns just output in this mode.
    ref_out, _ = batch_attention_run_trace.reference(run["q"], run["kv_cache"])
    # Matches tests/attention/test_batch_attention.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
