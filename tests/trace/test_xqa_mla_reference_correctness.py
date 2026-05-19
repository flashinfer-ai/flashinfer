"""Reference correctness test for the xqa_mla trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _cc,
    _close_pass_ratio,
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
            num_heads_qo=128,
            head_dim_ckv=512,
            head_dim_qk=576,
            page_size=32,
        ),
        dict(
            device="cuda",
            batch_size=1,
            beam_width=1,
            num_pages=2,
            max_pages_per_seq=2,
            num_heads_qo=64,
            head_dim_ckv=512,
            head_dim_qk=576,
            page_size=16,
        ),
    ],
)
def test_xqa_mla_reference_correctness(shape_kwargs):
    """XQA MLA kernel vs reference (latent-split page-gather SDPA)."""
    from flashinfer import xqa_mla
    from flashinfer.trace.templates.page import xqa_mla_trace

    if _cc()[0] != 12:
        pytest.skip("XQA MLA kernel only supports SM120/121")
    inputs = xqa_mla_trace.init(**shape_kwargs)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    xqa_mla(
        inputs["q"],
        inputs["k_cache"],
        inputs["v_cache"],
        inputs["page_table"],
        inputs["seq_lens"],
        inputs["output"],
        inputs["workspace_buffer"],
        inputs["semaphores"],
        inputs["page_size"],
        sm_count=sm_count,
    )
    # Reference uses dequantized API inputs for a clean comparison.
    q_ref = inputs["q"].float().squeeze(1)  # [B, Hq, QK]
    k_ref = inputs["k_cache"].float().squeeze(-2)
    v_ref = inputs["v_cache"].float().squeeze(-2)
    seq_lens_ref = inputs["seq_lens"].squeeze(1).to(torch.int32)
    ref_buffer = torch.empty_like(inputs["output"])
    ref_out = xqa_mla_trace.reference(
        q_ref,
        k_ref,
        v_ref,
        inputs["page_table"],
        seq_lens_ref,
        output=ref_buffer,
        output_dtype=torch.bfloat16,
    )
    # XQA MLA quantizes Q and the KV cache to FP8 internally; a few outlier
    # positions land on tied FP8 rounding boundaries. Matches the pass-ratio
    # metric the existing tests/attention/test_xqa.py uses for the same op:
    # >=95% of elements within (atol=0.05, rtol=0.05).
    _close_pass_ratio(
        inputs["output"].squeeze(1).float(),
        ref_out.float(),
        atol=0.05,
        rtol=0.05,
        pass_ratio=0.95,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
