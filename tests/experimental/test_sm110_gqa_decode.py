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
from flashinfer.experimental.sm110_gqa_decode import backend, jit, prepared

_PACKAGE = Path(jit.__file__).resolve().parent
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
]


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
    assert set(jit.ROUTES) == {
        "short",
        "long",
        "n32_b4_direct",
        "n32_disjoint_s10",
        "n64_kvlast_s10",
    }
    for route, record in jit.ROUTES.items():
        module = jit.MODULES[record["module"]]
        assert record["num_splits"] in (1, 10), route
        assert record["ffi_entry"].isidentifier(), route
        assert record["kernel_symbol"].startswith("kernel_sm110_gqa_decode_"), route
        for source in module["sources"]:
            assert (_PACKAGE / source).is_file(), f"{route}: missing {source}"
    assert set(jit.PREPARED_ROUTES.values()) <= set(jit.ROUTES)
    assert {"manifest.json", "RESULTS.md"}.isdisjoint(
        p.name for p in _PACKAGE.rglob("*")
    )


@_REQUIRES_CUDA
@pytest.mark.parametrize(("batch", "capacity", "lengths", "seed"), _SHAPES)
def test_decode_host_path_launches_without_device_synchronization(
    monkeypatch, batch, capacity, lengths, seed
) -> None:
    """The convenience API enqueues one launch and never reads device values."""

    module = _RecordingModule()
    monkeypatch.setattr(backend, "load_sm110_gqa_decode_module", lambda **_: module)
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
    assert entry == jit.ROUTES["short" if capacity <= 64 else "long"]["ffi_entry"]
    grouped_q, k, v, o, lengths_arg, scale, grid_x, grid_y, grid_z = arguments
    assert tuple(grouped_q.shape) == (batch, 4, 8, 128)
    assert grouped_q.data_ptr() == q.data_ptr()
    assert k.data_ptr() == kv[:, 0].data_ptr() and v.data_ptr() == kv[:, 1].data_ptr()
    assert o is out and lengths_arg is sequence_lengths
    assert scale == pytest.approx(1.5 / math.sqrt(128) / math.log(2.0))
    assert (grid_x, grid_y, grid_z) == (batch * 8, 1, 1)


@_REQUIRES_CUDA
@pytest.mark.parametrize(
    ("batch", "capacity", "num_splits", "route", "workspace_bytes"),
    [
        (1, 64, None, "short", 0),
        (4, 256, None, "n32_b4_direct", 0),
        (4, 256, 1, "long", 0),
        (1, 1024, None, "n64_kvlast_s10", 4 * (32 * 10 * 128 + 2 * 32 * 10) + 4 * 8),
        (1, 1024, 10, "n64_kvlast_s10", 4 * (32 * 10 * 128 + 2 * 32 * 10) + 4 * 8),
        (1, 4096, None, "n32_disjoint_s10", 4 * (32 * 10 * 128 + 2 * 32 * 10) + 4 * 8),
        (2, 65, None, "long", 0),
    ],
)
def test_prepared_host_path_selects_routes_without_device_synchronization(
    monkeypatch, batch, capacity, num_splits, route, workspace_bytes
) -> None:
    launches = []
    monkeypatch.setattr(prepared, "_check_exact_sm110a", lambda device: None)
    monkeypatch.setattr(
        prepared,
        "_launcher",
        lambda selected: lambda *arguments: launches.append((selected, arguments)),
    )
    q, kv, lengths = _inputs(batch, capacity, [capacity] * batch, 95700)
    inputs = {"Q": q, "KV": kv, "O": torch.empty_like(q), "sequence_lengths": lengths}
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        state = prepared.prepare_for_launch(inputs, num_splits=num_splits)
        result = prepared.launch_prepared(state)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert result is inputs["O"]
    assert state["route"] == route
    assert state["num_splits"] == jit.ROUTES[route]["num_splits"]
    assert state["workspace_bytes"] == workspace_bytes
    assert state["launch_names"] == [jit.ROUTES[route]["kernel_symbol"]]
    ((selected, arguments),) = launches
    assert selected == route
    assert arguments[-3:] == (batch * 8 * state["num_splits"], 1, 1)
    assert len(arguments) == (13 if state["num_splits"] > 1 else 9)


@_REQUIRES_CUDA
@pytest.mark.parametrize("num_splits", [3, 10])
def test_prepared_rejects_unsupported_b4_splits(monkeypatch, num_splits) -> None:
    monkeypatch.setattr(prepared, "_check_exact_sm110a", lambda device: None)
    q, kv, lengths = _inputs(4, 256, [64, 127, 191, 256], 95902)
    inputs = {"Q": q, "KV": kv, "O": torch.empty_like(q), "sequence_lengths": lengths}
    with pytest.raises(ValueError, match="exported split tile"):
        prepared.prepare_for_launch(inputs, num_splits=num_splits)


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
