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
        # fused paged output (walkfirst_primitives): physical slots through a
        # 256-page table of 64 columns
        dict(batch_size=16, max_seq_len=16384, top_k=512, max_pages=256, page_size=64),
    ],
)
def test_top_k_varlen_reference_correctness(shape_kwargs):
    """flashinfer.top_k_varlen (radix_cutlass, or walkfirst_primitives when paged) vs reference."""
    import flashinfer
    from flashinfer.trace.templates.topk import top_k_varlen_trace
    from flashinfer.utils import get_compute_capability

    inputs = top_k_varlen_trace.init(**shape_kwargs)
    major, minor = get_compute_capability(torch.device("cuda"))
    if not flashinfer.top_k_varlen.is_backend_supported(
        inputs["backend"], major * 10 + minor
    ):
        pytest.skip(f"{inputs['backend']} unsupported on this device")
    paged = {k: inputs[k] for k in ("page_table", "page_size") if k in inputs}
    indices, _ = flashinfer.top_k_varlen(
        inputs["logits"],
        inputs["seq_lens"],
        inputs["top_k"],
        backend=inputs["backend"],
        **paged,
    )
    ref = top_k_varlen_trace.reference(
        inputs["logits"], inputs["seq_lens"], inputs["top_k"], **paged
    )
    _check(
        top_k_varlen_trace,
        ref,
        indices,
        logits=inputs["logits"],
        seq_lens=inputs["seq_lens"],
        top_k=inputs["top_k"],
        **paged,
    )
    torch.cuda.synchronize()
