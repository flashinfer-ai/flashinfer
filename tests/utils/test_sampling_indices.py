"""Regression coverage for sampling params indexed by input rows.

This covers the failure mode in flashinfer-ai/flashinfer#5878: the output
batch can be longer than the input batch when `indices` remaps output rows to
input rows, but the per-row sampling parameters are still input-sized.
"""

import torch
import pytest

from flashinfer import sampling


@pytest.mark.parametrize("op", ["top_k", "min_p", "top_p"])
def test_sampling_params_indexed_by_input_rows(op):
    if not torch.cuda.is_available():
        pytest.skip("sampling kernels require CUDA")

    torch.manual_seed(0)
    input_rows, output_rows, vocab = 4, 16, 512
    probs = torch.softmax(torch.randn(input_rows, vocab, device="cuda"), dim=-1)
    indices = torch.arange(output_rows, device="cuda", dtype=torch.int32) % input_rows
    expected = probs.argmax(dim=-1)[indices]

    if op == "top_k":
        params = torch.ones(input_rows, dtype=torch.int32, device="cuda")
        actual = sampling.top_k_sampling_from_probs(
            probs, params, indices=indices, deterministic=True
        )
    elif op == "min_p":
        params = torch.ones(input_rows, dtype=torch.float32, device="cuda")
        actual = sampling.min_p_sampling_from_probs(
            probs, params, indices=indices, deterministic=True
        )
    else:
        params = torch.full(
            (input_rows,), 1e-6, dtype=torch.float32, device="cuda"
        )
        actual = sampling.top_p_sampling_from_probs(
            probs, params, indices=indices, deterministic=True
        )

    assert torch.equal(actual, expected), (
        f"{op}: output rows were not mapped to input-row parameters; "
        f"actual={actual.tolist()} expected={expected.tolist()}"
    )
