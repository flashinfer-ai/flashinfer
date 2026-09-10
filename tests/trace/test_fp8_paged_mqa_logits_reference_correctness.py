"""Reference correctness test for the fp8_paged_mqa_logits trace API."""

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
def test_fp8_paged_mqa_logits_reference_correctness(shape_kwargs):
    """flashinfer.fp8_paged_mqa_logits vs the trace template reference."""
    _skip_if_not_sm100_or_103()
    import flashinfer
    from flashinfer.trace.templates.attn_scores import fp8_paged_mqa_logits_trace

    inputs = fp8_paged_mqa_logits_trace.init(**shape_kwargs)
    out = flashinfer.fp8_paged_mqa_logits(
        inputs["q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["context_lens"],
        inputs["block_table"],
        inputs["max_context_len"],
    )
    ref = fp8_paged_mqa_logits_trace.reference(
        inputs["q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["context_lens"],
        inputs["block_table"],
        inputs["max_context_len"],
    )
    _check(
        fp8_paged_mqa_logits_trace,
        ref,
        out,
        context_lens=inputs["context_lens"],
        next_n=shape_kwargs["next_n"],
    )
    torch.cuda.synchronize()
