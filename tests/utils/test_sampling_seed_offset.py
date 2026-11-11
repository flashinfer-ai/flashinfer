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

"""Cover per-output RNG tensors and generator reservations with scalar references."""

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
# 32000 and 151936 reach the SM90 split-row kernel with 4- and 8-CTA clusters.
SPLIT_ROW_SAMPLERS = ["top_k", "top_p", "top_k_top_p"]
SAMPLER_VOCAB_SIZES = [(name, 257) for name in SAMPLERS] + [
    (name, vocab_size) for name in SPLIT_ROW_SAMPLERS for vocab_size in [32000, 151936]
]


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


@pytest.mark.parametrize("name,vocab_size", SAMPLER_VOCAB_SIZES)
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
@pytest.mark.parametrize("seed_size,offset_size", RNG_SHAPES)
def test_sampling_row_seed_offset(
    name, vocab_size, indexed, dtype, seed_size, offset_size
):
    sample = _sampler(name, indexed, vocab_size)
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


@pytest.mark.parametrize("use_default_generator", [False, True])
@pytest.mark.parametrize("seed", [0, 1234, (1 << 63) - 1, 1 << 63, (1 << 64) - 1])
@pytest.mark.parametrize(
    "initial_offset", [0, 128, (1 << 63) - 256, 1 << 63, (1 << 64) - 4]
)
def test_generator_reservation_state(use_default_generator, seed, initial_offset):
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.random.fork_rng(devices=[device.index]):
        generator = (
            torch.cuda.default_generators[device.index]
            if use_default_generator
            else torch.Generator(device=device)
        )
        generator.manual_seed(seed)
        generator.set_offset(initial_offset)
        expected = torch.Generator(device=device)
        expected.set_state(generator.get_state())
        for increment in (0, 1, 3, 4, 5, 17, 128):
            # The state-tensor representation is the pre-optimization contract,
            # including signed seed/offset bits and uint64 offset wraparound.
            expected_seed, expected_offset = map(
                int, expected.get_state().view(torch.int64)
            )
            next_offset = expected_offset + (increment + 3) // 4 * 4
            expected.set_state(
                torch.tensor([expected_seed, next_offset], dtype=torch.int64).view(
                    torch.uint8
                )
            )
            actual = sampling.get_seed_and_offset(
                increment, None if use_default_generator else generator, device
            )
            assert actual == (expected_seed, expected_offset)
            torch.testing.assert_close(
                generator.get_state(), expected.get_state(), rtol=0, atol=0
            )


@pytest.mark.parametrize(
    "initial_offset,increment", [((1 << 63) - 4, 1), ((1 << 63) - 8, 5)]
)
def test_generator_reservation_overflow_preserves_state(initial_offset, increment):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    generator.set_offset(initial_offset)
    state = generator.get_state()
    with pytest.raises((RuntimeError, ValueError), match="[Oo]verflow"):
        sampling.get_seed_and_offset(increment, generator)
    torch.testing.assert_close(generator.get_state(), state, rtol=0, atol=0)


@pytest.mark.parametrize("name", [*SAMPLERS, "chain"])
@pytest.mark.parametrize("use_default_generator", [False, True])
@pytest.mark.parametrize("seed", [1234, (1 << 64) - 1])
def test_sampling_generator_cuda_graph(monkeypatch, name, use_default_generator, seed):
    device = torch.device("cuda", torch.cuda.current_device())
    sample = _chain_sampler(1) if name == "chain" else _sampler(name, True)
    sample(seed=1234, offset=0)  # Compile before capture.
    torch.cuda.synchronize(device)
    reservations = []
    reserve = sampling.get_seed_and_offset

    def record_reservation(increment, generator=None, device=None):
        assert torch.cuda.is_current_stream_capturing()
        result = reserve(increment, generator, device)
        reservations.append(result)
        return result

    monkeypatch.setattr(sampling, "get_seed_and_offset", record_reservation)
    with torch.random.fork_rng(devices=[device.index]):
        generator = (
            torch.cuda.default_generators[device.index]
            if use_default_generator
            else torch.Generator(device=device)
        )
        generator.manual_seed(seed)
        generator.set_offset(128)
        graph = torch.cuda.CUDAGraph()
        if not use_default_generator:
            # PyTorch 2.13 requires explicit generators to be registered before capture.
            graph.register_generator_state(generator)
        try:
            with torch.cuda.graph(graph):
                outputs = [
                    sample(generator=None if use_default_generator else generator)
                    for _ in range(2)
                ]
            assert len(reservations) == 2
            assert reservations[1][1] > reservations[0][1]
            # Check the actual ranges chosen during capture: PyTorch versions
            # differ in their capture-time generator offset bookkeeping.
            for _ in range(3):
                graph.replay()
                for output, (captured_seed, captured_offset) in zip(
                    outputs, reservations, strict=True
                ):
                    expected = sample(seed=captured_seed, offset=captured_offset)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
            assert len(reservations) == 2  # Replay must not reserve on the host.
        finally:
            graph.reset()


@pytest.mark.parametrize("name", [*SAMPLERS, "chain"])
@pytest.mark.parametrize("use_default_generator", [False, True])
@pytest.mark.parametrize("seed", [1234, (1 << 64) - 1])
def test_sampling_generator_matches_explicit_rng(name, use_default_generator, seed):
    device = torch.device("cuda", torch.cuda.current_device())
    sample = _chain_sampler(1) if name == "chain" else _sampler(name, True)
    with torch.random.fork_rng(devices=[device.index]):
        generator = (
            torch.cuda.default_generators[device.index]
            if use_default_generator
            else torch.Generator(device=device)
        )
        generator.manual_seed(seed)
        generator.set_offset(128)
        actual = sample(generator=None if use_default_generator else generator)
        signed_seed = seed if seed < 1 << 63 else seed - (1 << 64)
        expected = sample(seed=signed_seed, offset=128)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
