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

"""Per-output RNG tensors must match scalar RNG values at the same output row."""

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
]
RNG_SHAPES = [(4, 4), (1, 4), (4, 1), (1, 1)]


def _sampler(name, indexed):
    generator = torch.Generator(device="cuda").manual_seed(42)
    logits = torch.randn(3 if indexed else 4, 257, device="cuda", generator=generator)
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
    if name == "top_k_top_p":
        return partial(
            sampling.top_k_top_p_sampling_from_probs,
            probs,
            100,
            0.8,
            filter_apply_order="joint",
            **common,
        )
    return partial(
        sampling.top_k_top_p_sampling_from_logits,
        logits,
        100,
        0.8,
        filter_apply_order="joint",
        **common,
    )


def _rng_tensors(seed_size, offset_size, dtype):
    seeds = [17, 324, 907, 1049][:seed_size]
    offsets = [0, 128, 512, 1024][:offset_size]
    return (
        torch.tensor(seeds, dtype=dtype, device="cuda"),
        torch.tensor(offsets, dtype=dtype, device="cuda"),
    )


def _assert_scalar_reference(sample, actual, seed, offset):
    # Keep the full output batch in each reference call: the existing Philox
    # subsequence includes the output row, even when explicit seeds are supplied.
    seeds = seed.cpu().tolist()
    offsets = offset.cpu().tolist()
    outputs = actual if isinstance(actual, tuple) else (actual,)
    for row in range(4):
        expected = sample(
            seed=seeds[0 if len(seeds) == 1 else row],
            offset=offsets[0 if len(offsets) == 1 else row],
        )
        references = expected if isinstance(expected, tuple) else (expected,)
        for output, reference in zip(outputs, references, strict=True):
            torch.testing.assert_close(output[row], reference[row], rtol=0, atol=0)


@pytest.mark.parametrize("name", SAMPLERS)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("seed_size,offset_size", RNG_SHAPES)
def test_sampling_row_seed_offset(name, indexed, dtype, seed_size, offset_size):
    sample = _sampler(name, indexed)
    seed, offset = _rng_tensors(seed_size, offset_size, dtype)
    actual = sample(seed=seed, offset=offset)
    _assert_scalar_reference(sample, actual, seed, offset)


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
    _assert_scalar_reference(sample, actual, seed, offset)


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
    _assert_scalar_reference(sample, actual, seed, offset)
    seed.add_(1234567)
    offset.add_(256)
    graph.replay()
    _assert_scalar_reference(sample, actual, seed, offset)


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
