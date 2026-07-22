"""Reference correctness test for the top_k_varlen trace API."""

import pytest
import torch

from tests.trace.reference_utils import _check


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(batch_size=8, max_seq_len=4096, top_k=512),
        dict(batch_size=32, max_seq_len=8192, top_k=1024),
    ],
)
def test_top_k_varlen_reference_correctness(shape_kwargs):
    """flashinfer.top_k_varlen (radix backend) vs reference."""
    import flashinfer
    from flashinfer.trace.templates.topk import top_k_varlen_trace

    inputs = top_k_varlen_trace.init(**shape_kwargs)
    indices = flashinfer.top_k_varlen(
        inputs["logits"],
        inputs["seq_lens"],
        inputs["top_k"],
        backend=inputs["backend"],
    )
    ref = top_k_varlen_trace.reference(
        inputs["logits"], inputs["seq_lens"], inputs["top_k"]
    )
    _check(
        top_k_varlen_trace,
        ref,
        indices,
        logits=inputs["logits"],
        seq_lens=inputs["seq_lens"],
        top_k=inputs["top_k"],
    )
    torch.cuda.synchronize()
