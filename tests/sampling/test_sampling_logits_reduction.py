"""Regression coverage for logits sampling across vocabulary tiles."""

import pytest
import torch

import flashinfer

# Odd sizes take vector width 1, sizes that are 2 mod 4 width 2, and multiples of 4
# width 4. Most of them span several vocabulary tiles.
VOCAB_SIZES = [
    33,
    1023,
    1024,
    1025,
    4095,
    4096,
    4097,
    4098,
    8196,
    32000,
    32001,
    151936,
    151937,
]


@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
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


@pytest.mark.parametrize("vocab_size", [4097, 4098, 8196, 151936])
@pytest.mark.parametrize("deterministic", [False, True])
def test_logits_sampling_ties_across_tiles(vocab_size, deterministic):
    # At this magnitude, adding Gumbel noise rounds back to the same FP32 value, and
    # equal maxima resolve to the larger position.
    logits = torch.full((8, vocab_size), -torch.inf, device="cuda")
    logits[:4, [0, vocab_size - 1]] = 1e30
    logits[4:, ::7] = 1e30
    expected = [vocab_size - 1] * 4 + [(vocab_size - 1) // 7 * 7] * 4
    samples = flashinfer.sampling_from_logits(
        logits, deterministic=deterministic, seed=12345, offset=17
    )
    assert samples.tolist() == expected


@pytest.mark.parametrize("vocab_size", [4097, 4098, 8196])
@pytest.mark.parametrize("positions", [(0, 1), (2, 3), (0, 127), (5, 300, 511)])
@pytest.mark.parametrize("deterministic", [False, True])
def test_logits_sampling_ties_within_tile(vocab_size, positions, deterministic):
    # Positions below 512 share the first tile for every block size and vector width,
    # so these ties are settled inside one block reduction: between the lanes of one
    # thread, the threads of one warp, or the warps of one block.
    logits = torch.full((8, vocab_size), -torch.inf, device="cuda")
    logits[:, list(positions)] = 1e30
    samples = flashinfer.sampling_from_logits(
        logits, deterministic=deterministic, seed=12345, offset=17
    )
    assert samples.tolist() == [max(positions)] * 8


# A tile spans 512 or 1024 threads times the vector width. Each size here fills its last
# tile, or leaves it partial, for both block sizes.
@pytest.mark.parametrize(
    ("vocab_size", "expected"),
    [(4096, 4095), (4097, 0), (4098, 0), (8192, 8191), (8196, 0), (32000, 0)],
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_logits_sampling_all_masked_row(vocab_size, expected, deterministic):
    # Every slot ties at -inf, including the padding past the vocabulary, so the row
    # resolves to the last slot: d - 1 when the last tile is full, and 0 otherwise.
    logits = torch.full((4, vocab_size), -torch.inf, device="cuda")
    samples = flashinfer.sampling_from_logits(
        logits, deterministic=deterministic, seed=12345, offset=17
    )
    assert samples.tolist() == [expected] * 4


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
