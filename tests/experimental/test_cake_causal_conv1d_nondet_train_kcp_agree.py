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

CPU tests of the rank-agreement check of the non-deterministic conv KCP route (issue #8946: every rank of a call
passes the same partition and calls the same entry, mode and channel counts): the int64[8] fingerprint against an
independent evaluation of its definition, one check per (context, phase), no check with one rank, the same error on
every rank naming the ranks and fields that differ, and a refused first call inside a CUDA-graph capture.  The
collective is replaced by a shim; no kernel runs here.
"""

from __future__ import annotations

import hashlib
import types

import pytest
import torch

from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from flashinfer.experimental.cake.cake_kda_det_train_kcp import cake_kcp_plan

W = 4
CU = (0, 3000, 3001, 9000, 12000)


def _low63(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") & (
        2**63 - 1
    )


def _ctx(world: int = W, layout: str = "zigzag_per_document"):
    part = cake_kcp_plan.cake_kda_det_train_kcp_partition(list(CU), world, layout)
    return types.SimpleNamespace(
        partition=part, world_size=world, conv_kernel_size=4, group=None
    ), part


def _expected(part, phase: str, channels: tuple[int, ...], multi: bool) -> list[int]:
    text = (
        f"{part.world_size}|{part.layout}|"
        + ",".join(map(str, part.cu_seqlens))
        + "|"
        + ";".join(
            f"{f.doc},{f.off},{f.length},{f.rank},{f.slot}"
            for frags in part.fragments
            for f in frags
        )
    )
    return [
        0x4B43504650310001,
        2,
        {"fwd": 1, "bwd": 2}[phase],
        ((2 if multi else 1) << 32) | len(channels),
        _low63(",".join(map(str, channels))) if multi else channels[0],
        (part.world_size << 8) | 4,
        _low63(text),
        0 if phase == "fwd" else 1,
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("layout", ["contiguous", "zigzag", "zigzag_per_document"])
def test_fingerprint_matches_its_definition(layout) -> None:
    ctx, part = _ctx(layout=layout)
    for phase in ("fwd", "bwd"):
        assert conv_kcp._fingerprint(ctx, phase, (10240,), False) == _expected(
            part, phase, (10240,), False
        )
        three = (8192, 8192, 8192)
        assert conv_kcp._fingerprint(ctx, phase, three, True) == _expected(
            part, phase, three, True
        )
    # Every field is a non-negative int64.
    assert all(0 <= v < 2**63 for v in conv_kcp._fingerprint(ctx, "bwd", (8,), False))
    # The partition field tells layouts apart.
    other, _ = _ctx(layout="contiguous" if layout != "contiguous" else "zigzag")
    assert (
        conv_kcp._fingerprint(ctx, "fwd", (8,), False)[6]
        != conv_kcp._fingerprint(other, "fwd", (8,), False)[6]
    )


class _Gather:
    """Replaces ``dist.all_gather_into_tensor``: fills the output with this rank's row and the given rows of the
    other ranks (``None``: the same as this rank's)."""

    def __init__(self, others) -> None:
        self.others = others
        self.calls = 0

    def __call__(self, out, mine, group=None):
        self.calls += 1
        rows = [mine.tolist()] + [
            (mine.tolist() if o is None else o(mine.tolist())) for o in self.others
        ]
        out.copy_(torch.tensor(sum(rows, []), dtype=torch.int64))


def _patch(monkeypatch, gather) -> None:
    monkeypatch.setattr(conv_kcp.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.mark.cpu_only
def test_once_per_context_and_phase(monkeypatch) -> None:
    gather = _Gather([None] * (W - 1))
    _patch(monkeypatch, gather)
    ctx, _ = _ctx()
    cpu = torch.device("cpu")
    conv_kcp._agree(ctx, "fwd", (64,), False, device=cpu)
    # Already checked: no collective.
    conv_kcp._agree(ctx, "fwd", (128,), True, device=cpu)
    assert gather.calls == 1
    conv_kcp._agree(ctx, "bwd", (64,), False, device=cpu)
    conv_kcp._agree(ctx, "bwd", (64,), False, device=cpu)
    assert gather.calls == 2
    fresh, _ = _ctx()
    conv_kcp._agree(fresh, "fwd", (64,), False, device=cpu)
    assert gather.calls == 3
    one, _ = _ctx(world=1)
    conv_kcp._agree(one, "fwd", (64,), False, device=cpu)  # one rank: no check
    assert gather.calls == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "field,edit",
    [
        ("mode", lambda r: r[:1] + [1] + r[2:]),
        ("entry/tensors", lambda r: r[:3] + [(2 << 32) | 3] + r[4:]),
        ("channels", lambda r: r[:4] + [r[4] + 8] + r[5:]),
        ("partition", lambda r: r[:6] + [r[6] ^ 1] + r[7:]),
        ("form", lambda r: r[:7] + [2]),
    ],
)
def test_mismatch_raises_naming_ranks_and_fields(monkeypatch, field, edit) -> None:
    _patch(monkeypatch, _Gather([None, edit, None]))
    ctx, _ = _ctx()
    with pytest.raises(ValueError, match="rank agreement") as info:
        conv_kcp._agree(ctx, "bwd", (64,), False, device=torch.device("cpu"))
    assert f"rank 2 differs from rank 0 in {field}" in str(info.value)
    # A failed check records nothing: the next call checks again.
    gather = _Gather([None] * (W - 1))
    _patch(monkeypatch, gather)
    conv_kcp._agree(ctx, "bwd", (64,), False, device=torch.device("cpu"))
    assert gather.calls == 1


@pytest.mark.cpu_only
def test_first_call_inside_a_capture_is_refused(monkeypatch) -> None:
    gather = _Gather([None] * (W - 1))
    monkeypatch.setattr(conv_kcp.dist, "all_gather_into_tensor", gather)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    ctx, _ = _ctx()
    with pytest.raises(ValueError, match="eager call"):
        conv_kcp._agree(ctx, "fwd", (64,), False, device=torch.device("cpu"))
    assert gather.calls == 0


@pytest.mark.cpu_only
def test_a_multi_rank_context_needs_its_partition() -> None:
    ctx = types.SimpleNamespace(world_size=2, conv_kernel_size=4, group=None)
    with pytest.raises(TypeError, match="partition"):
        conv_kcp._fingerprint(ctx, "fwd", (64,), False)
