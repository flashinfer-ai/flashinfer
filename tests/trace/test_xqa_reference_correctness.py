"""Reference correctness test for the xqa trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close_pass_ratio,
    _skip_if_not_sm100,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            device="cuda",
            batch_size=2,
            beam_width=1,
            num_pages=4,
            max_pages_per_seq=2,
            num_heads_qo=16,
            num_kv_heads=2,
            head_dim=128,
            page_size=16,
        ),
        dict(
            device="cuda",
            batch_size=1,
            beam_width=1,
            num_pages=2,
            max_pages_per_seq=2,
            num_heads_qo=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=16,
        ),
    ],
)
def test_xqa_reference_correctness(shape_kwargs):
    """XQA kernel vs reference (page-gather + SDPA)."""
    from flashinfer import xqa
    from flashinfer.trace.templates.page import xqa_trace

    _skip_if_not_sm100()
    inputs = xqa_trace.init(**shape_kwargs)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    xqa(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["page_table"],
        inputs["seq_lens"],
        inputs["output"],
        inputs["workspace_buffer"],
        inputs["semaphores"],
        inputs["num_kv_heads"],
        inputs["page_size"],
        kv_layout=inputs["kv_layout"],
        sm_count=sm_count,
    )
    # Reference uses [num_tokens, Hq, D] layout — squeeze beam dim.
    q_ref = inputs["q"].squeeze(1)
    seq_lens_ref = inputs["seq_lens"].squeeze(1).to(torch.int32)
    ref_out = xqa_trace.reference(
        q_ref,
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["page_table"],
        seq_lens_ref,
    )
    # Matches tests/attention/test_xqa.py: >=98% of elements within
    # (atol=0.05, rtol=0.05).
    _close_pass_ratio(
        inputs["output"].squeeze(1).float(),
        ref_out.float(),
        atol=0.05,
        rtol=0.05,
        pass_ratio=0.98,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
