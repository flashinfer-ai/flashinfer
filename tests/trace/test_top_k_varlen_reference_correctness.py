"""Reference correctness test for the top_k_varlen trace API."""

import pytest
import torch

from tests.trace.reference_utils import _check


def _radix_cutlass_supported() -> bool:
    """The trace fixes ``backend="radix_cutlass"``, which is not offered on every
    compute capability (e.g. SM107/Rubin)."""
    if not torch.cuda.is_available():
        return False
    import flashinfer
    from flashinfer.utils import get_compute_capability

    major, minor = get_compute_capability(torch.device("cuda"))
    return flashinfer.top_k_varlen.is_backend_supported(
        "radix_cutlass", major * 10 + minor
    )


@pytest.mark.skipif(
    not _radix_cutlass_supported(),
    reason="top_k_varlen radix_cutlass backend is unsupported on this compute capability",
)
@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(batch_size=8, max_seq_len=4096, top_k=512),
        dict(batch_size=32, max_seq_len=8192, top_k=1024),
    ],
)
def test_top_k_varlen_reference_correctness(shape_kwargs):
    """flashinfer.top_k_varlen (radix_cutlass backend) vs reference."""
    import flashinfer
    from flashinfer.trace.templates.topk import top_k_varlen_trace

    inputs = top_k_varlen_trace.init(**shape_kwargs)
    indices, _ = flashinfer.top_k_varlen(
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
