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

import math
import weakref

import pytest
import torch

from flashinfer import launch_sm110_gqa_decode_prepared, prepare_sm110_gqa_decode

_SHAPES = [
    (1, 64, [1], 95601),
    (4, 256, [64, 127, 191, 256], 95602),
    (1, 1024, [1024], 95603),
    (1, 4096, [3968], 95604),
]
_SHAPE_ARGUMENTS = ("batch", "capacity", "lengths", "seed")
_REQUIRES_SM110 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="requires an exact SM110 GPU",
)


def _inputs(batch, capacity, lengths, seed, q_scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch, 32, 128, dtype=torch.float16, device="cuda", generator=generator
    )
    kv = torch.randn(
        batch,
        2,
        8,
        capacity,
        128,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    return {
        "Q": q,
        "KV": kv,
        "O": torch.empty_like(q),
        "sequence_lengths": torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        "q_scale": q_scale,
    }


def _reference(inputs):
    q, kv, lengths = (inputs[k] for k in ("Q", "KV", "sequence_lengths"))
    batch, _, head_dim = q.shape
    capacity = kv.shape[-2]
    grouped_q = q.float().view(batch, 8, 4, head_dim)
    scores = torch.einsum("bhgd,bhkd->bhgk", grouped_q, kv[:, 0].float())
    positions = torch.arange(capacity, device=q.device)
    valid = positions.view(1, 1, 1, capacity) < lengths.view(batch, 1, 1, 1)
    scores.masked_fill_(~valid, -torch.inf)
    probabilities = torch.softmax(
        scores * inputs["q_scale"] / math.sqrt(head_dim), dim=-1
    )
    return torch.einsum("bhgk,bhkd->bhgd", probabilities, kv[:, 1].float()).reshape_as(
        q
    )


def test_prepared_entry_points_are_experimental():
    assert prepare_sm110_gqa_decode.is_experimental
    assert launch_sm110_gqa_decode_prepared.is_experimental


@_REQUIRES_SM110
@pytest.mark.parametrize(_SHAPE_ARGUMENTS, _SHAPES)
@pytest.mark.parametrize("q_scale", [0.5, 1.0, 1.5])
def test_prepared_matches_fp32_reference_and_owns_output(
    batch, capacity, lengths, seed, q_scale
):
    inputs = _inputs(batch, capacity, lengths, seed, q_scale)
    expected = _reference(inputs)
    before = {k: inputs[k].clone() for k in ("Q", "KV", "sequence_lengths")}
    output_ref = weakref.ref(inputs["O"])
    prepared = prepare_sm110_gqa_decode(inputs)
    # All views, input storage, output, and scratch must survive the caller's
    # input mapping. The returned output is the exact supplied Tensor object.
    del inputs
    actual = launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()

    assert actual is output_ref()
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        prepared["bindings"]["Q"].transpose(1, 2).reshape_as(before["Q"]),
        before["Q"],
        atol=0,
        rtol=0,
    )
    assert torch.equal(prepared["bindings"]["K"], before["KV"][:, 0])
    assert torch.equal(prepared["bindings"]["V"], before["KV"][:, 1])
    assert torch.equal(
        prepared["bindings"]["sequence_lengths"], before["sequence_lengths"]
    )


@_REQUIRES_SM110
@pytest.mark.parametrize(_SHAPE_ARGUMENTS, _SHAPES[1:])
def test_explicit_one_split_uses_original_long(batch, capacity, lengths, seed):
    inputs = _inputs(batch, capacity, lengths, seed)
    expected = _reference(inputs)
    prepared = prepare_sm110_gqa_decode(inputs, num_splits=1)
    actual = launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()

    assert prepared["route"] == "long"
    assert prepared["num_splits"] == 1
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


@_REQUIRES_SM110
@pytest.mark.parametrize(_SHAPE_ARGUMENTS, _SHAPES[2:])
def test_explicit_ten_splits_matches_reference(batch, capacity, lengths, seed):
    inputs = _inputs(batch, capacity, lengths, seed)
    expected = _reference(inputs)
    prepared = prepare_sm110_gqa_decode(inputs, num_splits=10)
    actual = launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()

    assert prepared["num_splits"] == 10
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


@_REQUIRES_SM110
def test_unlisted_shape_uses_original_long():
    inputs = _inputs(2, 65, [1, 65], 95701)
    expected = _reference(inputs)
    prepared = prepare_sm110_gqa_decode(inputs)
    actual = launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()

    assert prepared["route"] == "long"
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


@_REQUIRES_SM110
@pytest.mark.parametrize(
    ("batch", "capacity", "states", "seed"),
    [
        (1, 64, [[1], [31], [64], [63]], 95801),
        (4, 256, [[64, 127, 191, 256], [1, 63, 64, 65], [2, 191, 255, 256]], 95802),
        (1, 1024, [[1], [127], [1023], [1024]], 95803),
        (1, 4096, [[1], [1025], [3968], [4096]], 95804),
    ],
)
def test_graph_replay_with_dynamic_lengths(batch, capacity, states, seed):
    inputs = _inputs(batch, capacity, states[0], seed, q_scale=1.5)
    prepared = prepare_sm110_gqa_decode(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch_sm110_gqa_decode_prepared(prepared)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch_sm110_gqa_decode_prepared(prepared)

    snapshots = []
    with torch.cuda.stream(stream):
        for state in states:
            inputs["sequence_lengths"].copy_(
                torch.tensor(state, dtype=torch.int32, device="cuda")
            )
            # Queued replays reuse the same scratch/counters without a reset
            # launch, and the captured kernel must observe updated lengths.
            for _ in range(16):
                graph.replay()
            snapshots.append((inputs["O"].clone(), _reference(inputs)))
    stream.synchronize()

    for actual, expected in snapshots:
        torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


@_REQUIRES_SM110
@pytest.mark.parametrize(_SHAPE_ARGUMENTS, _SHAPES)
def test_independent_prepared_objects_on_nondefault_streams(
    batch, capacity, lengths, seed
):
    inputs = [_inputs(batch, capacity, lengths, seed + offset) for offset in (0, 10)]
    prepared = [prepare_sm110_gqa_decode(item) for item in inputs]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    snapshots = []
    for stream, item, state in zip(streams, inputs, prepared, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            item["Q"].mul_(0.5)
            item["O"].fill_(float("nan"))
            for _ in range(16):
                actual = launch_sm110_gqa_decode_prepared(state)
            assert actual is item["O"]
            snapshots.append((actual.clone(), _reference(item)))
    for stream in streams:
        stream.synchronize()

    for actual, expected in snapshots:
        torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


@_REQUIRES_SM110
@pytest.mark.parametrize("alias", ["Q", "KV"])
def test_prepared_rejects_output_alias(alias):
    inputs = _inputs(4, 256, [64, 127, 191, 256], 95901)
    inputs["O"] = inputs[alias].view(-1)[: 4 * 32 * 128].view(4, 32, 128)
    with pytest.raises(ValueError, match="O must not alias"):
        prepare_sm110_gqa_decode(inputs)


@_REQUIRES_SM110
@pytest.mark.parametrize("num_splits", [True, 3, 10])
def test_prepared_rejects_unsupported_b4_splits(num_splits):
    inputs = _inputs(4, 256, [64, 127, 191, 256], 95902)
    with pytest.raises(ValueError, match="num_splits|exported split tile"):
        prepare_sm110_gqa_decode(inputs, num_splits=num_splits)
