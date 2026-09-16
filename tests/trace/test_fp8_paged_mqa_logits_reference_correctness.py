"""Reference correctness test for the fp8_paged_mqa_logits trace API."""

import pytest
import torch

from tests.trace.reference_utils import _cc, _check


def _skip_if_no_paged_mqa_support():
    if _cc() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(batch_size=4, next_n=2, max_seq_len=4096, block_size=64),
        dict(batch_size=8, next_n=1, max_seq_len=2048, block_size=128),
        # max_seq_len is a free axis and need not be block-aligned: the last
        # physical block then runs past the output width (257 -> 5*64 = 320).
        dict(batch_size=2, next_n=2, max_seq_len=257, block_size=64),
        dict(batch_size=2, next_n=1, max_seq_len=513, block_size=128),
    ],
)
def test_fp8_paged_mqa_logits_reference_correctness(shape_kwargs):
    """flashinfer.fp8_paged_mqa_logits vs the trace template reference."""
    _skip_if_no_paged_mqa_support()
    import flashinfer
    from flashinfer.trace.templates.attn_scores import fp8_paged_mqa_logits_trace

    inputs = fp8_paged_mqa_logits_trace.init(**shape_kwargs)
    out = flashinfer.fp8_paged_mqa_logits(
        inputs["q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["block_tables"],
        inputs["seq_lens"],
        inputs["max_seq_len"],
    )
    ref = fp8_paged_mqa_logits_trace.reference(
        inputs["q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["block_tables"],
        inputs["seq_lens"],
        inputs["max_seq_len"],
    )
    _check(
        fp8_paged_mqa_logits_trace,
        ref,
        out,
        seq_lens=inputs["seq_lens"],
        next_n=shape_kwargs["next_n"],
    )
    torch.cuda.synchronize()
