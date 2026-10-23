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


def _one_split_route(batch, capacity):
    """The size rule above 128 keys: 64-token tiles for one request over at most 256 keys or two
    requests over at most 512, the four-slot 64-token ring for longer caches of one or two
    requests, 32-token tiles for three or more requests."""
    if batch >= 3:
        return "split_n32_direct"
    return "split_n64_direct" if capacity <= 256 * batch else "split_n64_r4_direct"


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
def test_explicit_one_split_is_the_size_rule(batch, capacity, lengths, seed):
    """``num_splits=1`` and ``None`` select the same one-split route: the direct build of the split-KV
    kernel the size rule names above 128 keys (``_one_split_route``)."""
    inputs = _inputs(batch, capacity, lengths, seed)
    expected = _reference(inputs)
    prepared = prepare_sm110_gqa_decode(inputs, num_splits=1)
    actual = launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()

    assert prepared["route"] == _one_split_route(batch, capacity)
    assert prepared["route"] == prepare_sm110_gqa_decode(inputs)["route"]
    assert prepared["num_splits"] == 1
    assert prepared["grid"] == (1, batch * 8, 1)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)


@_REQUIRES_SM110
@pytest.mark.parametrize("num_splits", [2, 4, 8, 10, 16])
@pytest.mark.parametrize(_SHAPE_ARGUMENTS, _SHAPES)
def test_explicit_splits_match_reference(batch, capacity, lengths, seed, num_splits):
    """Every accepted split count above one launches the run-time-split program of the shape's tile
    (64-token tiles for one or two requests over at most 4095 keys, 32-token tiles otherwise) with
    that many splits (splits past the key count merge as zeros)."""
    inputs = _inputs(batch, capacity, lengths, seed)
    expected = _reference(inputs)
    prepared = prepare_sm110_gqa_decode(inputs, num_splits=num_splits)
    actual = launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()

    assert prepared["route"] == (
        "split_n64" if batch <= 2 and capacity <= 4095 else "split_n32"
    )
    assert prepared["num_splits"] == num_splits
    assert prepared["grid"] == (num_splits, batch * 8, 1)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
    completed = prepared["workspace"][-1]
    assert torch.equal(completed, torch.zeros_like(completed))


@_REQUIRES_SM110
@pytest.mark.parametrize(
    ("batch", "capacity", "lengths", "route"),
    [
        (1, 64, [1], "short"),
        (1, 65, [65], "long"),
        (1, 128, [100], "long"),
        (1, 129, [129], "split_n64_direct"),
        (2, 65, [1, 65], "split_n64_direct"),
        (2, 300, [37, 300], "split_n64_direct"),
        (1, 257, [257], "split_n64_r4_direct"),
        (2, 513, [1, 513], "split_n64_r4_direct"),
        (3, 100, [1, 50, 100], "split_n32_direct"),
    ],
)
def test_size_rule_routes(batch, capacity, lengths, route):
    inputs = _inputs(batch, capacity, lengths, 95701)
    expected = _reference(inputs)
    prepared = prepare_sm110_gqa_decode(inputs)
    actual = launch_sm110_gqa_decode_prepared(prepared)
    torch.cuda.synchronize()

    assert prepared["route"] == route
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
@pytest.mark.parametrize("num_splits", [True, 3, 0, 32])
def test_prepared_rejects_unsupported_splits(num_splits):
    inputs = _inputs(4, 256, [64, 127, 191, 256], 95902)
    with pytest.raises(ValueError, match="num_splits must be None or an integer"):
        prepare_sm110_gqa_decode(inputs, num_splits=num_splits)


@_REQUIRES_SM110
@pytest.mark.parametrize(
    ("batch", "capacity", "num_splits"),
    [(1, 32, None), (1, 63, 1), (1, 32, 4), (2, 48, None)],
)
def test_prepared_rejects_caches_below_the_short_tma_box(batch, capacity, num_splits):
    """Unchanged public behaviour: a cache of fewer than 64 tokens raises the previous
    release's error (the short route's 64-token TMA box) and launches nothing."""
    inputs = _inputs(batch, capacity, [capacity] * batch, 95903)
    inputs["O"].fill_(float("nan"))
    with pytest.raises(
        ValueError,
        match=r"TMA box \(64, 64, 1, 1\) exceeds resolved global dims for 'K'",
    ):
        prepare_sm110_gqa_decode(inputs, num_splits=num_splits)
    assert torch.isnan(inputs["O"]).all().item()
