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
from pathlib import Path

import pytest
import torch

from flashinfer import sm110_gqa_decode
from flashinfer.experimental.sm110_gqa_decode import backend, cake_jit

_PACKAGE = Path(cake_jit.__file__).resolve().parent
_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
_REQUIRES_SM110 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="requires an exact SM110 GPU",
)
_SHAPES = [
    (1, 64, [1], 95601),
    (4, 256, [64, 127, 191, 256], 95602),
    (1, 1024, [1024], 95603),
    (1, 4096, [3968], 95604),
    # Several requests over a short cache: the one-split 32-token-tile build.
    (3, 64, [64, 64, 64], 95605),
    (2, 100, [37, 100], 95606),
]
_TMA_BOX_MESSAGE = r"TMA box \(64, 64, 1, 1\) exceeds resolved global dims for 'K'"


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    q_scale: float,
) -> torch.Tensor:
    batch, _, head_dim = q.shape
    capacity = int(kv.shape[-2])
    grouped_q = q.float().view(batch, 8, 4, head_dim)
    k = kv[:, 0].float()
    v = kv[:, 1].float()
    scores = torch.einsum("bhgd,bhkd->bhgk", grouped_q, k)
    positions = torch.arange(capacity, device=q.device)
    valid = positions.view(1, 1, 1, capacity) < sequence_lengths.view(batch, 1, 1, 1)
    scores.masked_fill_(~valid, -torch.inf)
    probabilities = torch.softmax(scores * q_scale / math.sqrt(head_dim), dim=-1)
    return torch.einsum("bhgk,bhkd->bhgd", probabilities, v).reshape_as(q).half()


def _inputs(batch, capacity, lengths, seed=0):
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
    lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q, kv, lengths


class _RecordingModule:
    """Stands in for the compiled module so the host path runs on any GPU."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, entry):
        if not entry.startswith("run"):
            raise AttributeError(entry)

        def launch(*arguments):
            self.calls.append((entry, arguments))

        return launch


def test_public_entry_point_is_experimental() -> None:
    assert sm110_gqa_decode.is_experimental


def test_registry_names_delivered_sources() -> None:
    manifest = cake_jit.cake_build.manifest()
    libraries = cake_jit.cake_build.libraries_by_name()
    assert manifest["schema"] == "cake-build/1"
    for route, record in cake_jit.MODULES.items():
        assert record["ffi_entry"] == "run", route
        assert record["library"].startswith("cake_sm110_gqa_decode_"), route
        entry = libraries[cake_jit.library_name(route, "sm_110a")]
        for source in entry["sources"]:
            assert (_PACKAGE / source).is_file(), f"{route}: missing {source}"


@_REQUIRES_CUDA
@pytest.mark.parametrize(("batch", "capacity", "lengths", "seed"), _SHAPES)
def test_decode_host_path_launches_without_device_synchronization(
    monkeypatch, batch, capacity, lengths, seed
) -> None:
    """The convenience API enqueues one launch and never reads device values."""

    module = _RecordingModule()
    monkeypatch.setattr(backend, "require_sm110", lambda device=None: None)
    monkeypatch.setattr(backend, "_entry", lambda record, index: module.run)
    q, kv, sequence_lengths = _inputs(batch, capacity, lengths, seed)
    out = torch.empty_like(q)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        result = sm110_gqa_decode(q, kv, sequence_lengths, out=out, q_scale=1.5)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert result is out
    (entry, arguments), *rest = module.calls
    assert not rest
    assert entry == "run"
    record, splits, grid = backend.route(batch, capacity, None)
    assert record == cake_jit.select_record(batch=batch, capacity=capacity)
    assert splits == 1
    if cake_jit.MODULES[record]["grid_rule"]["kind"] == "batch_heads":
        grouped_q, k, v, o, lengths_arg, scale, grid_x, grid_y, grid_z = arguments
        assert (grid_x, grid_y, grid_z) == (batch * 8, 1, 1) == grid
    else:
        # The one-split builds of the split-KV program keep the workspace pointers in their binding.
        (
            grouped_q,
            k,
            v,
            partial_o,
            partial_max,
            partial_sum,
            completed,
            o,
            lengths_arg,
            scale,
            grid_x,
            grid_y,
            grid_z,
        ) = arguments
        assert (grid_x, grid_y, grid_z) == (1, batch * 8, 1) == grid
        assert tuple(partial_o.shape) == (batch, 32, 1, 128)
        assert completed.dtype == torch.uint32 and tuple(completed.shape) == (
            batch * 8,
        )
    assert tuple(grouped_q.shape) == (batch, 4, 8, 128)
    assert grouped_q.data_ptr() == q.data_ptr()
    assert k.data_ptr() == kv[:, 0].data_ptr() and v.data_ptr() == kv[:, 1].data_ptr()
    assert o is out and lengths_arg is sequence_lengths
    assert scale == pytest.approx(1.5 / math.sqrt(128) / math.log(2.0))


def _workspace_bytes(batch: int, splits: int) -> int:
    return 4 * (batch * 32 * splits * 128 + 2 * batch * 32 * splits) + 4 * batch * 8


@_REQUIRES_CUDA
@pytest.mark.parametrize("capacity", [1, 32, 63])
@pytest.mark.parametrize("batch", [1, 2])
def test_caches_below_the_short_tma_box_are_rejected_as_before(
    monkeypatch, batch, capacity
) -> None:
    """A cache of fewer than 64 tokens is refused by both APIs before any launch,
    with the previous release's error type and message (its short route's
    64-token TMA box); the rule itself still answers for every capacity."""

    launches = []
    monkeypatch.setattr(backend, "require_sm110", lambda device=None: None)
    monkeypatch.setattr(
        backend, "_entry", lambda record, index: lambda *a: launches.append(a)
    )
    assert backend.MIN_CAPACITY == 64
    q, kv, lengths = _inputs(batch, capacity, [capacity] * batch, 95800 + capacity)
    with pytest.raises(ValueError, match=_TMA_BOX_MESSAGE):
        sm110_gqa_decode(q, kv, lengths)
    inputs = {"Q": q, "KV": kv, "O": torch.empty_like(q), "sequence_lengths": lengths}
    for num_splits in (None, 1, 4):
        with pytest.raises(ValueError, match=_TMA_BOX_MESSAGE):
            backend.prepare_for_launch(inputs, num_splits=num_splits)
    assert not launches
    assert cake_jit.select_record(batch=batch, capacity=capacity) in cake_jit.MODULES
    # The box itself is accepted.
    q, kv, lengths = _inputs(batch, 64, [64] * batch, 95864)
    sm110_gqa_decode(q, kv, lengths)
    backend.prepare_for_launch(
        {"Q": q, "KV": kv, "O": torch.empty_like(q), "sequence_lengths": lengths}
    )
    assert len(launches) == 1


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("batch", "capacity", "num_splits", "route", "splits", "workspace_bytes"),
    [
        (1, 64, None, "short", 1, 0),
        (1, 128, None, "long", 1, 0),
        (1, 64, 1, "short", 1, 0),
        (4, 256, None, "split_n32_direct", 1, _workspace_bytes(4, 1)),
        (4, 256, 1, "split_n32_direct", 1, _workspace_bytes(4, 1)),
        (2, 65, None, "split_n64_direct", 1, _workspace_bytes(2, 1)),
        (2, 512, 1, "split_n64_direct", 1, _workspace_bytes(2, 1)),
        (1, 1024, None, "split_n64_r4_direct", 1, _workspace_bytes(1, 1)),
        (2, 513, 1, "split_n64_r4_direct", 1, _workspace_bytes(2, 1)),
        (1, 1024, 10, "split_n64", 10, _workspace_bytes(1, 10)),
        (1, 64, 2, "split_n64", 2, _workspace_bytes(1, 2)),
        (1, 4096, 10, "split_n32", 10, _workspace_bytes(1, 10)),
        (4, 256, 10, "split_n32", 10, _workspace_bytes(4, 10)),
        (3, 1024, 16, "split_n32", 16, _workspace_bytes(3, 16)),
    ],
)
def test_prepared_host_path_selects_routes_without_device_synchronization(
    monkeypatch, batch, capacity, num_splits, route, splits, workspace_bytes
) -> None:
    launches = []
    monkeypatch.setattr(backend, "require_sm110", lambda device=None: None)
    monkeypatch.setattr(
        backend,
        "_entry",
        lambda record, index: lambda *arguments: launches.append((record, arguments)),
    )
    q, kv, lengths = _inputs(batch, capacity, [capacity] * batch, 95700)
    inputs = {"Q": q, "KV": kv, "O": torch.empty_like(q), "sequence_lengths": lengths}
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        state = backend.prepare_for_launch(inputs, num_splits=num_splits)
        result = backend.launch_prepared(state)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert result is inputs["O"]
    assert state["route"] == route
    assert state["num_splits"] == splits
    assert state["workspace_bytes"] == workspace_bytes
    ((selected, arguments),) = launches
    assert selected == route
    if cake_jit.MODULES[route]["grid_rule"]["kind"] == "batch_heads":
        assert arguments[-3:] == (batch * 8, 1, 1)
        assert len(arguments) == 9
    else:
        assert arguments[-3:] == (splits, batch * 8, 1)
        assert len(arguments) == 13
    assert state["grid"] == arguments[-3:]


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("num_splits", "message"),
    [
        (True, "num_splits must be None or an integer"),
        (3, "num_splits must be None or an integer"),
        (0, "num_splits must be None or an integer"),
        (32, "num_splits must be None or an integer"),
    ],
)
def test_prepared_rejects_unsupported_splits(monkeypatch, num_splits, message):
    """The public split validation and its message are unchanged: the accepted set is {1,2,4,8,10,16}."""

    monkeypatch.setattr(backend, "require_sm110", lambda device=None: None)
    q, kv, lengths = _inputs(4, 256, [64, 127, 191, 256], 95902)
    inputs = {"Q": q, "KV": kv, "O": torch.empty_like(q), "sequence_lengths": lengths}
    with pytest.raises(ValueError, match=message):
        backend.prepare_for_launch(inputs, num_splits=num_splits)


@_REQUIRES_SM110
@pytest.mark.parametrize(("batch", "capacity", "lengths", "seed"), _SHAPES)
def test_sm110_gqa_decode_matches_reference(
    batch: int,
    capacity: int,
    lengths: list[int],
    seed: int,
) -> None:
    q, kv, sequence_lengths = _inputs(batch, capacity, lengths, seed)
    expected = _reference(q, kv, sequence_lengths, q_scale=1.0)
    q_before = q.clone()
    kv_before = kv.clone()
    lengths_before = sequence_lengths.clone()
    out = torch.empty_like(q)

    actual = sm110_gqa_decode(
        q,
        kv,
        sequence_lengths,
        out=out,
    )
    torch.cuda.synchronize()

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    assert torch.isfinite(actual).all()
    assert torch.equal(q, q_before)
    assert torch.equal(kv, kv_before)
    assert torch.equal(sequence_lengths, lengths_before)
