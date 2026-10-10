"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Per-output RNG tensors must follow requests across batch layouts."""

from functools import partial

import pytest
import torch

from flashinfer import sampling


SAMPLERS = [
    "logits",
    "probs",
    "top_k",
    "top_p",
    "min_p",
    "top_k_top_p",
    "top_k_top_p_logits",
    "top_k_top_p_sequential",
    "top_k_top_p_logits_sequential",
]
RNG_SHAPES = [(4, 4), (1, 4), (4, 1), (1, 1)]


def _sampler(name, indexed, vocab_size=257):
    generator = torch.Generator(device="cuda").manual_seed(42)
    logits = torch.randn(
        3 if indexed else 4, vocab_size, device="cuda", generator=generator
    )
    probs = torch.softmax(logits, dim=-1)
    # RNG arrays describe output rows, including repeated/gathered input rows.
    indices = (
        torch.tensor([2, 0, 2, 1], device="cuda", dtype=torch.int32)
        if indexed
        else None
    )
    common = {"indices": indices, "deterministic": True}
    if name == "logits":
        return partial(sampling.sampling_from_logits, logits, **common)
    if name == "probs":
        return partial(sampling.sampling_from_probs, probs, **common)
    if name == "top_k":
        return partial(sampling.top_k_sampling_from_probs, probs, 100, **common)
    if name == "top_p":
        return partial(sampling.top_p_sampling_from_probs, probs, 0.8, **common)
    if name == "min_p":
        return partial(sampling.min_p_sampling_from_probs, probs, 0.1, **common)
    function = (
        sampling.top_k_top_p_sampling_from_logits
        if "logits" in name
        else sampling.top_k_top_p_sampling_from_probs
    )
    return partial(
        function,
        logits if "logits" in name else probs,
        100,
        0.8,
        filter_apply_order="top_k_first" if name.endswith("sequential") else "joint",
        **common,
    )


def _rng_tensors(seed_size, offset_size, dtype):
    seeds = [17, 324, 907, 1049][:seed_size]
    offsets = [0, 128, 512, 1024][:offset_size]
    return (
        torch.tensor(seeds, dtype=dtype, device="cuda"),
        torch.tensor(offsets, dtype=dtype, device="cuda"),
    )


def _select_rows(sample, rows):
    """Select output rows without changing their input distributions."""
    kwargs = dict(sample.keywords)
    if kwargs.get("indices") is not None:
        kwargs["indices"] = kwargs["indices"][rows].contiguous()
        args = sample.args
    else:
        args = tuple(
            arg[rows].contiguous() if isinstance(arg, torch.Tensor) else arg
            for arg in sample.args
        )
    return partial(sample.func, *args, **kwargs)


def _assert_rng_reference(sample, actual, seed, offset):
    # Tensor keys identify requests even in singleton or broadcast-only calls.
    seeds = seed.cpu().tolist()
    offsets = offset.cpu().tolist()
    outputs = actual if isinstance(actual, tuple) else (actual,)
    for row in range(4):
        seed_row = 0 if len(seeds) == 1 else row
        offset_row = 0 if len(offsets) == 1 else row
        expected = _select_rows(sample, [row])(
            seed=seed[seed_row : seed_row + 1],
            offset=offset[offset_row : offset_row + 1],
        )
        references = expected if isinstance(expected, tuple) else (expected,)
        for output, reference in zip(outputs, references, strict=True):
            torch.testing.assert_close(output[row], reference[0], rtol=0, atol=0)
    for rows in ([3, 1, 0, 2], [3, 1], [2]):
        selected = _select_rows(sample, rows)(
            # PyTorch does not implement CUDA advanced indexing for uint64.
            seed=seed.view(torch.int64)[rows].view(seed.dtype)
            if len(seeds) > 1
            else seed,
            offset=offset.view(torch.int64)[rows].view(offset.dtype)
            if len(offsets) > 1
            else offset,
        )
        selected = selected if isinstance(selected, tuple) else (selected,)
        for output, reordered in zip(outputs, selected, strict=True):
            torch.testing.assert_close(output[rows], reordered, rtol=0, atol=0)


@pytest.mark.parametrize("name", SAMPLERS)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("seed_size,offset_size", RNG_SHAPES)
def test_sampling_row_seed_offset(name, indexed, dtype, seed_size, offset_size):
    sample = _sampler(name, indexed)
    seed, offset = _rng_tensors(seed_size, offset_size, dtype)
    actual = sample(seed=seed, offset=offset)
    _assert_rng_reference(sample, actual, seed, offset)


def _chain_sampler(reject_at):
    # Deterministically accept draft token 0 until reject_at, then sample a
    # replacement. reject_at=3 exercises the all-accepted final-token draw.
    draft = torch.zeros(4, 3, 257, device="cuda")
    draft[:, :, 0] = 1
    target = torch.full((4, 4, 257), 1 / 256, device="cuda")
    target[:, :, 0] = 0
    target[:, :reject_at] = draft[:, :reject_at]
    draft_ids = torch.zeros(4, 3, dtype=torch.int32, device="cuda")
    return partial(
        sampling.chain_speculative_sampling,
        draft,
        draft_ids,
        target,
        deterministic=True,
    )


@pytest.mark.parametrize("reject_at", [0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("seed_size,offset_size", RNG_SHAPES)
def test_chain_sampling_row_seed_offset(reject_at, dtype, seed_size, offset_size):
    sample = _chain_sampler(reject_at)
    seed, offset = _rng_tensors(seed_size, offset_size, dtype)
    actual = sample(seed=seed, offset=offset)
    _assert_rng_reference(sample, actual, seed, offset)


@pytest.mark.parametrize("name", [*SAMPLERS, "chain"])
@pytest.mark.parametrize("seed_size,offset_size", [(4, 4), (1, 1)])
def test_sampling_rng_tensor_cuda_graph(name, seed_size, offset_size):
    sample = _chain_sampler(1) if name == "chain" else _sampler(name, True)
    seed, offset = _rng_tensors(seed_size, offset_size, torch.int64)
    sample(seed=seed, offset=offset)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sample(seed=seed, offset=offset)
    graph.replay()
    _assert_rng_reference(sample, actual, seed, offset)
    seed.add_(1234567)
    offset.add_(256)
    graph.replay()
    _assert_rng_reference(sample, actual, seed, offset)


@pytest.mark.parametrize("name", ["seed", "offset"])
@pytest.mark.parametrize("size", [0, 2, 5])
def test_sampling_rng_tensor_ffi_rejects_invalid_length(name, size):
    # Exercise the FFI boundary directly, without the Python shape validator.
    module = sampling.gen_sampling_module().build_and_load()
    logits = torch.zeros(3, 257, device="cuda")
    indices = torch.tensor([2, 0, 2, 1], dtype=torch.int32, device="cuda")
    output = torch.empty(4, dtype=torch.int32, device="cuda")
    seed, offset = _rng_tensors(4, 4, torch.int64)
    invalid = torch.zeros(size, dtype=torch.int64, device="cuda")
    if name == "seed":
        seed = invalid
    else:
        offset = invalid
    with pytest.raises(RuntimeError, match=f"{name} tensor length must be 1 or 4"):
        module.sampling_from_logits(logits, output, indices, True, seed, 0, offset, 0)


@pytest.mark.parametrize("name", SAMPLERS)
@pytest.mark.parametrize("vocab_size", [256, 1026])
def test_sampling_row_rng_vectorization(name, vocab_size):
    # Cover vector widths 4 and 2 in addition to the width-1 (257) matrix.
    sample = _sampler(name, False, vocab_size)
    seed, offset = _rng_tensors(4, 4, torch.int64)
    _assert_rng_reference(sample, sample(seed=seed, offset=offset), seed, offset)


@pytest.mark.parametrize(
    "name", ["top_k_top_p_sequential", "top_k_top_p_logits_sequential"]
)
def test_sampling_row_rng_top_k_fast_path(name):
    sample = _sampler(name, False, 65536)
    seed, offset = _rng_tensors(4, 4, torch.int64)
    _assert_rng_reference(sample, sample(seed=seed, offset=offset), seed, offset)


@pytest.mark.parametrize("name", [*SAMPLERS, "chain"])
def test_repeated_request_rng_replays_same_draw(name):
    sample = _chain_sampler(1) if name == "chain" else _sampler(name, True)
    # Duplicate a single logical request, including its RNG values. Distinct
    # output slots must not introduce hidden randomness for this request.
    sample = _select_rows(sample, [2, 2, 2, 2])
    seed = torch.full((4,), 2**63 + 17, device="cuda", dtype=torch.uint64)
    offset = torch.full((4,), 2**32, device="cuda", dtype=torch.uint64)
    _assert_rng_reference(sample, sample(seed=seed, offset=offset), seed, offset)


@pytest.mark.parametrize("name", ["logits", "probs", "top_k", "top_p", "min_p"])
def test_adjacent_offsets_do_not_share_draws(name):
    """Neighboring request keys must not be shifted views of one RNG stream."""
    batch_size, vocab_size = 512, 4096
    logits = torch.zeros(batch_size, vocab_size, device="cuda")
    probs = torch.full_like(logits, 1 / vocab_size)
    functions = {
        "logits": partial(sampling.sampling_from_logits, logits),
        "probs": partial(sampling.sampling_from_probs, probs),
        "top_k": partial(sampling.top_k_sampling_from_probs, probs, 50),
        "top_p": partial(sampling.top_p_sampling_from_probs, probs, 0.8),
        "min_p": partial(sampling.min_p_sampling_from_probs, probs, 0.1),
    }
    sample = functions[name]
    seed = torch.tensor([17], device="cuda", dtype=torch.int64)
    offsets = torch.arange(batch_size, device="cuda", dtype=torch.int64)
    actual = sample(seed=seed, offset=offsets, deterministic=True)
    shifted = sample(seed=seed, offset=offsets + 1, deterministic=True)
    # Identical keys still replay, even when their output row changes.
    torch.testing.assert_close(shifted[:-1], actual[1:], rtol=0, atol=0)
    # Uniform top-k has expected duplicate rate 1/50; allow ample margin while
    # rejecting the excessive duplicates reported for shared rejection draws.
    assert (actual[1:] == actual[:-1]).float().mean().item() < 0.1
    if name == "logits":
        # Gumbel's vectorized draws must not slide across token coordinates.
        for delta in (1, 2, 3):
            shifted = sample(seed=seed, offset=offsets + delta, deterministic=True)
            assert (shifted == actual - delta).float().mean().item() < 0.02
