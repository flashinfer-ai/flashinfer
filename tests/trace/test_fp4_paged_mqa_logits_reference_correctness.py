"""Reference correctness test for the fp4_paged_mqa_logits trace API."""

import pytest
import torch

from tests.trace.reference_utils import _check, _skip_if_not_sm100_or_103


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(batch_size=4, next_n=2, max_context_len=4096, block_size=64),
        dict(batch_size=8, next_n=1, max_context_len=2048, block_size=128),
        # max_context_len is a free axis and need not be page-aligned: the last
        # physical page then runs past the output width (257 -> 5*64 = 320).
        dict(batch_size=2, next_n=2, max_context_len=257, block_size=64),
        dict(batch_size=2, next_n=1, max_context_len=513, block_size=128),
    ],
)
def test_fp4_paged_mqa_logits_reference_correctness(shape_kwargs):
    """flashinfer.fp4_paged_mqa_logits vs the trace template reference."""
    _skip_if_not_sm100_or_103()
    import flashinfer
    from flashinfer.trace.templates.attn_scores import fp4_paged_mqa_logits_trace

    inputs = fp4_paged_mqa_logits_trace.init(**shape_kwargs)
    out = flashinfer.fp4_paged_mqa_logits(
        inputs["q"],
        inputs["sf_q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["context_lens"],
        inputs["block_table"],
        inputs["max_context_len"],
    )
    ref = fp4_paged_mqa_logits_trace.reference(
        inputs["q"],
        inputs["sf_q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["context_lens"],
        inputs["block_table"],
        inputs["max_context_len"],
    )
    # The template declares bfloat16 logits and the API defaults to it; assert
    # rather than assume, so a future divergence between the schema, the API and
    # the reference is caught here instead of silently masked by forcing a dtype.
    declared = fp4_paged_mqa_logits_trace.outputs["logits"].dtype
    assert declared == "bfloat16"
    assert out.dtype is torch.bfloat16, f"kernel returned {out.dtype}"
    assert ref.dtype is torch.bfloat16, f"reference returned {ref.dtype}"

    _check(
        fp4_paged_mqa_logits_trace,
        ref,
        out,
        context_lens=inputs["context_lens"],
        next_n=shape_kwargs["next_n"],
    )
    torch.cuda.synchronize()
