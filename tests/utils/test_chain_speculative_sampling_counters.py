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

"""Caller-owned counter contracts for chain speculative sampling."""

import pytest
import torch

from flashinfer import sampling
from flashinfer.jit.sampling import gen_sampling_module

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_COUNTER_NAMES = (
    "maybe_output_accepted_token_num",
    "maybe_output_emitted_draft_token_num",
)
_BATCH = 4
_DRAFT = 3


@pytest.fixture(scope="module")
def native_module():
    return gen_sampling_module().build_and_load()


def _inputs(batch=_BATCH):
    draft = torch.zeros(batch, _DRAFT, 257, device="cuda")
    draft[:, :, 0] = 1
    ids = torch.zeros(batch, _DRAFT, dtype=torch.int32, device="cuda")
    target = torch.zeros(batch, _DRAFT + 1, 257, device="cuda")
    target[:, :, 0] = 1
    return draft, ids, target


def _set_rejection(target, rejection):
    target.zero_()
    target[:, :, 0] = 1
    if rejection is None:
        return _DRAFT, _DRAFT, [0] * (_DRAFT + 1)
    target[:, rejection, 0] = 0
    target[:, rejection, 1] = 1
    # Later one-hot checks still accept, but only the prefix is emitted.
    return _DRAFT - 1, rejection, [0] * rejection + [1] + [-1] * (_DRAFT - rejection)


def _call(route, native_module, inputs, accepted, emitted):
    if route == "public":
        return sampling.chain_speculative_sampling(
            *inputs, accepted, emitted, seed=17, offset=0
        )
    if route == "custom_op":
        return sampling.get_sampling_module().chain_speculative_sampling(
            *inputs, accepted, emitted, True, None, 17, 0
        )
    output = torch.full((_BATCH, _DRAFT + 1), -99, device="cuda", dtype=torch.int32)
    native_module.chain_speculative_sampling(
        *inputs, output, accepted, emitted, True, None, 17, None, 0
    )
    return output


def _invalid_counter(kind, initial):
    dtype = {
        "int64": torch.int64,
        "float32": torch.float32,
        "int16": torch.int16,
    }.get(kind, torch.int32)
    # Retain padding so regressions can be detected without intentional OOB writes.
    storage = torch.full(
        (_BATCH * 4,), 101, device="cpu" if kind == "cpu" else "cuda", dtype=dtype
    )
    if kind == "column":
        counter = storage[: _BATCH * 2].view(_BATCH, 2)[:, 0]
    elif kind == "expanded":
        storage[0] = initial
        return storage, storage[:1].expand(_BATCH)
    elif kind == "rank2":
        counter = storage[: _BATCH * 2].view(_BATCH, 2)
    elif kind == "scalar":
        counter = storage[0]
    elif kind == "short":
        counter = storage[: _BATCH - 1]
    elif kind == "long":
        counter = storage[: _BATCH + 1]
    else:
        counter = storage[:_BATCH]
    counter.fill_(initial)
    return storage, counter


@pytest.mark.parametrize("route", ["public", "custom_op", "native"])
@pytest.mark.parametrize("counter_index", [0, 1], ids=["accepted", "emitted"])
@pytest.mark.parametrize(
    "kind",
    [
        "int64",
        "float32",
        "int16",
        "column",
        "expanded",
        "rank2",
        "scalar",
        "short",
        "long",
        "cpu",
    ],
)
def test_chain_counter_rejected_without_writes(
    native_module, route, counter_index, kind
):
    inputs = _inputs()
    counters = [
        torch.full((_BATCH,), initial, device="cuda", dtype=torch.int32)
        for initial in (5, 7)
    ]
    storage, counter = _invalid_counter(kind, (5, 7)[counter_index])
    counters[counter_index] = counter
    before_storage = storage.clone()
    other = counters[1 - counter_index]
    before_other = other.clone()
    error = TypeError if kind in ("int64", "float32", "int16") else ValueError
    with pytest.raises(error, match=_COUNTER_NAMES[counter_index]):
        _call(route, native_module, inputs, *counters)
    torch.cuda.synchronize()
    torch.testing.assert_close(storage, before_storage, rtol=0, atol=0)
    torch.testing.assert_close(other, before_other, rtol=0, atol=0)


@pytest.mark.parametrize("counter_index", [0, 1], ids=["accepted", "emitted"])
def test_chain_counter_python_rejects_before_dispatch(monkeypatch, counter_index):
    def unexpected_dispatch():
        pytest.fail(
            "invalid counters must be rejected before loading the sampling module"
        )

    monkeypatch.setattr(sampling, "get_sampling_module", unexpected_dispatch)
    counters = [None, None]
    counters[counter_index] = torch.zeros(_BATCH, device="cuda", dtype=torch.int64)
    with pytest.raises(TypeError, match=_COUNTER_NAMES[counter_index]):
        sampling.chain_speculative_sampling(*_inputs(), *counters)


@pytest.mark.parametrize("batch", [1, _BATCH])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("rejection", [None, 0, 1], ids=["all", "first", "middle"])
@pytest.mark.parametrize(
    "supplied", ["both", "accepted", "emitted", "neither", "offset"]
)
def test_chain_counter_accumulation(batch, deterministic, rejection, supplied):
    inputs = _inputs(batch)
    da, de, tokens = _set_rejection(inputs[2], rejection)
    stores = [
        torch.full((batch + 2,), 101, device="cuda", dtype=torch.int32)
        for _ in range(2)
    ]
    counters = [
        store[1:-1] if supplied == "offset" else store[:batch] for store in stores
    ]
    counters[0].fill_(5)
    counters[1].fill_(7)
    accepted = None if supplied in ("emitted", "neither") else counters[0]
    emitted = None if supplied in ("accepted", "neither") else counters[1]
    expected_stores = [store.clone() for store in stores]
    start = 1 if supplied == "offset" else 0
    for repeat in (1, 2):
        output, a, e = sampling.chain_speculative_sampling(
            *inputs, accepted, emitted, deterministic=deterministic, seed=17, offset=0
        )
        assert output.tolist() == [tokens] * batch
        assert a.tolist() == [da if accepted is None else 5 + repeat * da] * batch
        assert e.tolist() == [de if emitted is None else 7 + repeat * de] * batch
        assert accepted is None or a is accepted
        assert emitted is None or e is emitted
        for i, (supplied_counter, increment) in enumerate(
            ((accepted, da), (emitted, de))
        ):
            if supplied_counter is not None:
                expected_stores[i][start : start + batch] += increment
            torch.testing.assert_close(stores[i], expected_stores[i], rtol=0, atol=0)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("offset_view", [False, True])
def test_chain_counter_graph_replay(deterministic, offset_view):
    inputs = _inputs()
    stores = [
        torch.full((_BATCH + 2,), 101, device="cuda", dtype=torch.int32)
        for _ in range(2)
    ]
    start = int(offset_view)
    accepted, emitted = [store[start : start + _BATCH] for store in stores]

    def run():
        return sampling.chain_speculative_sampling(
            *inputs, accepted, emitted, deterministic=deterministic, seed=17, offset=0
        )

    # Warm the JIT/custom op before capturing; capture reads the same caller storage.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, a, e = run()
    assert a is accepted and e is emitted
    for rejection in (None, 0, 1):
        da, de, tokens = _set_rejection(inputs[2], rejection)
        for store in stores:
            store.fill_(101)
        accepted.fill_(5)
        emitted.fill_(7)
        expected_stores = [store.clone() for store in stores]
        for repeat in (1, 2):
            graph.replay()
            assert output.tolist() == [tokens] * _BATCH
            assert accepted.tolist() == [5 + repeat * da] * _BATCH
            assert emitted.tolist() == [7 + repeat * de] * _BATCH
            expected_stores[0][start : start + _BATCH] += da
            expected_stores[1][start : start + _BATCH] += de
            for store, expected in zip(stores, expected_stores, strict=True):
                torch.testing.assert_close(store, expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("route", ["public", "custom_op", "native"])
@pytest.mark.parametrize("counter_index", [0, 1], ids=["accepted", "emitted"])
def test_chain_counter_wrong_cuda_device(native_module, route, counter_index):
    inputs = _inputs()
    counters = [
        torch.zeros(_BATCH, device=inputs[0].device, dtype=torch.int32)
        for _ in range(2)
    ]
    other_device = (inputs[0].device.index + 1) % torch.cuda.device_count()
    counters[counter_index] = torch.zeros(
        _BATCH, device=f"cuda:{other_device}", dtype=torch.int32
    )
    with pytest.raises(ValueError, match=_COUNTER_NAMES[counter_index]):
        _call(route, native_module, inputs, *counters)
    for counter in counters:
        assert counter.tolist() == [0] * _BATCH
