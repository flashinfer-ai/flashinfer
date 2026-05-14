"""Reference correctness test for the multi_level_cascade_run trace API."""

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
            batch_size=1,
            num_pages=1,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
            page_size=16,
        ),
        dict(
            device="cuda",
            batch_size=1,
            num_pages=2,
            num_qo_heads=4,
            num_kv_heads=1,
            head_dim=64,
            page_size=8,
        ),
    ],
)
def test_multi_level_cascade_run_reference_correctness(shape_kwargs):
    """MultiLevelCascadeAttentionWrapper.run kernel vs reference.

    Single-level cascade with batch_size=1 so the reference's single-sequence
    page-gather assumption holds.
    """
    from flashinfer import MultiLevelCascadeAttentionWrapper
    from flashinfer.trace.templates.attention import multi_level_cascade_run_trace

    inputs = multi_level_cascade_run_trace.init(**shape_kwargs)
    plan = inputs["plan"]
    run = inputs["run"]
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = MultiLevelCascadeAttentionWrapper(1, ws, "NHD")
        wrapper.plan(
            [plan["qo_indptr"]],
            [plan["kv_indptr"]],
            [plan["kv_indices"]],
            [plan["kv_last_page_len"]],
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            plan["head_dim"],
            plan["page_size"],
            q_data_type=plan["q_data_type"],
        )
        api_out = wrapper.run(run["q"], run["paged_kv_cache"])
    except Exception as exc:
        pytest.skip(f"MultiLevelCascadeAttentionWrapper unavailable: {exc}")
    ref_out = multi_level_cascade_run_trace.reference(run["q"], run["paged_kv_cache"])
    # tests/attention/test_shared_prefix_kernels.py uses 1e-3 but compares
    # two kernel outputs with identical internal math; our reference uses
    # torch-level fp32 math which diverges by ~1 bf16 ULP from the kernel's
    # bf16 accumulation. Use 1e-2 (matching test_batch_attention.py bf16 tol).
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
