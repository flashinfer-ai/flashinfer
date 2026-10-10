"""Regression coverage for logits sampling across vocabulary tiles."""

import pytest
import torch

import flashinfer


@pytest.mark.parametrize(
    "vocab_size", [33, 1023, 1024, 1025, 4095, 4096, 4097, 32001, 151937]
)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("indices_dtype", [None, torch.int32, torch.int64])
def test_logits_sampling_masked_tile_boundaries(
    vocab_size, deterministic, indices_dtype
):
    positions = sorted(
        {
            p
            for p in [0, 1, 3, 31, 32, 1023, 1024, 4095, 4096, vocab_size - 1]
            if p < vocab_size
        }
    )
    expected = torch.tensor(positions, device="cuda", dtype=torch.int64)
    logits = torch.full((len(positions), vocab_size), -torch.inf, device="cuda")
    logits.scatter_(1, expected[:, None], 0.0)
    indices = None
    if indices_dtype is not None:
        # Reordered and repeated rows exercise the output-to-input mapping.
        indices = torch.arange(
            len(positions) - 1, -1, -1, device="cuda", dtype=indices_dtype
        ).repeat(2)
        expected = expected[indices.long()]
    samples = flashinfer.sampling_from_logits(
        logits, indices=indices, deterministic=deterministic, seed=12345, offset=17
    )
    torch.testing.assert_close(samples.long(), expected, rtol=0, atol=0)
    assert samples.dtype == (indices_dtype or torch.int32)


@pytest.mark.parametrize("vocab_size", [4097, 8193, 151937])
@pytest.mark.parametrize("deterministic", [False, True])
def test_logits_sampling_ties_across_tiles(vocab_size, deterministic):
    # At this magnitude, adding Gumbel noise rounds back to the same FP32 value.
    # The existing sampler retains the later vocabulary tile on equal maxima.
    logits = torch.full((8, vocab_size), -torch.inf, device="cuda")
    logits[:, 0] = 1e30
    logits[:, -1] = 1e30
    samples = flashinfer.sampling_from_logits(
        logits, deterministic=deterministic, seed=12345, offset=17
    )
    torch.testing.assert_close(
        samples, torch.full_like(samples, vocab_size - 1), rtol=0, atol=0
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_logits_sampling_graph_replay_updates_input(deterministic):
    logits = torch.full((4, 8193), -torch.inf, device="cuda")
    logits[:, 0] = 0.0

    def sample():
        return flashinfer.sampling_from_logits(
            logits, deterministic=deterministic, seed=12345, offset=17
        )

    sample()  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = sample()
    for token in [8192, 4096, 1023, 0]:
        logits.fill_(-torch.inf)
        logits[:, token] = 0.0
        graph.replay()
        torch.testing.assert_close(
            output, torch.full_like(output, token), rtol=0, atol=0
        )
