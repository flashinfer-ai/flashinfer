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

The KDA ``ContextDetKcp`` is one leaf of FSDP2's input walks (flashinfer issue #7914): the pre-forward hook of every
``fully_shard`` module walks each dataclass, dict, list and tuple of the module's arguments twice
(``collect_grad_tensors`` / ``replace_grad_tensors``), so a context carried in per-layer metadata must not be one.
The gate is structural: the number of nodes the walk visits is the same with a context of 80 fragments, of 2,048
fragments and without a context.  Also checked: the former frozen dataclass's constructor, repr, identity equality
and ``FrozenInstanceError``.  CPU only (gloo, world size 1); no kernel is launched.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch import nn

from tests.test_helpers.cake_package import cake_package

kcp_plan = cake_package("cake_kda_det_train_kcp", "cake_kcp_plan")

try:
    from torch.distributed.fsdp import _common_utils as fsdp_common
except ImportError:  # a torch without FSDP
    fsdp_common = None

pytestmark = pytest.mark.skipif(
    fsdp_common is None or not hasattr(fsdp_common, "collect_grad_tensors"),
    reason="torch.distributed.fsdp._common_utils.collect_grad_tensors not available",
)

HEADS = 64
FIVE_DOCS = [56843, 59310, 72971, 23218, 38408]  # the issue's KDA batch


@dataclasses.dataclass(frozen=True)
class BatchMeta:
    """Per-batch metadata a model hands to every layer (the issue's repro)."""

    cu_seqlens: torch.Tensor
    kcp: object | None


def _context(lengths: list[int], cp: int, fragments: int):
    """Rank 0's context of ``lengths`` cut per document into ``2 * cp`` pieces, without a process group, in its
    state after the first ``chunk_kda_det_kcp`` call of ``HEADS`` heads; ``fragments`` = the group's fragment count."""
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    part = kcp_plan.partition_det_kcp(cu, cp, "zigzag_per_document")
    assert sum(len(f) for f in part.fragments) == fragments
    return cu, kcp_plan.build_context_det_kcp(part, None, 0, "cpu", heads=HEADS)


@pytest.fixture(scope="module")
def contexts():
    return {
        80: _context(FIVE_DOCS, 8, 80),
        2048: _context([4000] * 64, 16, 2048),
    }


def _counting(monkeypatch, name: str) -> list[int]:
    """Wrap ``fsdp_common.<name>`` (the recursive walk: its recursion goes through the module global) to count
    the nodes it visits."""
    count = [0]
    inner = getattr(fsdp_common, name)

    def visit(*args: Any, **kwargs: Any) -> Any:
        count[0] += 1
        return inner(*args, **kwargs)

    monkeypatch.setattr(fsdp_common, name, visit)
    return count


def _walk_counts(monkeypatch, inputs) -> tuple[int, int]:
    collect = _counting(monkeypatch, "_collect_grad_tensors")
    replace = _counting(monkeypatch, "_replace_grad_tensors")
    tensors = fsdp_common.collect_grad_tensors(inputs)
    assert fsdp_common.replace_grad_tensors(inputs, iter(tensors)) is not None
    monkeypatch.undo()
    return collect[0], replace[0]


def _inputs(cu: torch.Tensor, ctx) -> tuple:
    """A layer's ``(args, kwargs)``: the context in a frozen dataclass, a dict and a list."""
    x = torch.randn(4, 8, requires_grad=True)
    meta = BatchMeta(cu, ctx)
    return (x, meta), {"meta": {"kcp": ctx, "cu": cu}, "contexts": [ctx, meta]}


def test_walk_visits_context_once(contexts, monkeypatch) -> None:
    """The walk visits the same number of nodes with an 80- or a 2,048-fragment context as with ``None`` in its
    place; ``replace_grad_tensors`` returns the metadata itself."""
    cu80, ctx80 = contexts[80]
    cu2k, ctx2k = contexts[2048]
    assert ctx2k.n_frag > ctx80.n_frag and len(ctx2k.heads(HEADS).fwd_m_off) > 0
    base = _walk_counts(monkeypatch, _inputs(cu80, None))
    assert _walk_counts(monkeypatch, _inputs(cu80, ctx80)) == base
    assert _walk_counts(monkeypatch, _inputs(cu2k, ctx2k)) == base
    assert fsdp_common.collect_grad_tensors(ctx2k) == ()
    meta = BatchMeta(cu2k, ctx2k)
    assert fsdp_common.replace_grad_tensors(meta, iter(())) is meta
    assert fsdp_common.replace_grad_tensors(ctx2k, iter(())) is ctx2k
    y = torch.randn(3, requires_grad=True)
    out = fsdp_common.replace_grad_tensors(({"kcp": ctx2k, "y": y},), iter([y * 2]))
    assert out[0]["kcp"] is ctx2k


def test_input_cast_keeps_context(contexts) -> None:
    """The input cast of FSDP2's ``MixedPrecisionPolicy`` (``torch.distributed.utils._apply_to_tensors``) rebuilds
    every object with ``__dataclass_fields__``; the context passes through as the same object."""
    apply = getattr(
        importlib.import_module("torch.distributed.utils"), "_apply_to_tensors", None
    )
    if apply is None:
        pytest.skip("torch.distributed.utils._apply_to_tensors not available")
    cu, ctx = contexts[2048]
    meta = BatchMeta(cu, ctx)
    assert apply(lambda t: t, ctx) is ctx
    assert apply(lambda t: t, meta).kcp is ctx
    out = apply(lambda t: t, {"meta": meta, "contexts": [ctx], "kcp": ctx})
    assert out["meta"].kcp is ctx and out["contexts"][0] is ctx and out["kcp"] is ctx


def test_context_keeps_dataclass_behavior(contexts) -> None:
    """Constructor fields in the former order, the former repr, identity equality and hash, frozen attributes."""
    _, ctx = contexts[80]
    K = kcp_plan.ContextDetKcp
    assert not dataclasses.is_dataclass(ctx) and not dataclasses.is_dataclass(K)
    assert tuple(inspect.signature(K).parameters) == K._FIELDS
    # the repr of the former dataclass (sequence and the "_" caches had repr=False)
    former = dataclasses.make_dataclass(
        "ContextDetKcp",
        [
            (f, Any, dataclasses.field(repr=f != "sequence" and not f.startswith("_")))
            for f in K._FIELDS
        ],
        frozen=True,
        eq=False,
    )
    values = {f: getattr(ctx, f) for f in K._FIELDS}
    assert repr(ctx) == repr(former(**values))
    # keyword construction; the default-factory fields are fresh per instance
    public = {
        f: v for f, v in values.items() if not f.startswith("_") and f != "sequence"
    }
    twin = K(**public)
    assert twin.p_rule == ctx.p_rule and twin.upload == ctx.upload
    assert (
        isinstance(twin.sequence, kcp_plan.KcpSequence)
        and twin.sequence is not ctx.sequence
    )
    for f in ("_head_plans", "_table_sets", "_waited", "_lazy"):
        assert getattr(twin, f) == {} and getattr(twin, f) is not getattr(ctx, f)
    assert twin._uploads == []
    assert K(**public, sequence=None).sequence is None  # an explicit value is kept
    assert twin != ctx and twin == twin and hash(twin) == object.__hash__(twin)
    assert len({ctx, twin}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.rank = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        del ctx.rank
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.extra = 1
    assert ctx.rank == 0
    # the caches stay mutable through the context's own methods
    assert twin.heads(4) is twin.heads(4) and 4 in twin._head_plans


class _Layer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
        assert meta.kcp is None or isinstance(meta.kcp, kcp_plan.ContextDetKcp)
        return self.proj(x)


class _Stack(nn.Module):
    def __init__(self, layers: int, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer(width) for _ in range(layers))

    def forward(self, x: torch.Tensor, meta: BatchMeta) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, meta)
        return x


def test_fully_shard_layers_walk_context_once(contexts, monkeypatch) -> None:
    """Three ``fully_shard`` layers in a ``fully_shard`` stack (gloo, world size 1) take the metadata with the
    2,048-fragment context: forward and backward run, the context reaches every layer unchanged and the hooks' walks
    visit as many nodes as without a context."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with gloo not available")
    if dist.is_initialized():
        pytest.skip("a default process group is already initialized")
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        pytest.skip("torch.distributed.fsdp.fully_shard not available")
    cu, ctx = contexts[2048]
    seen: list[object] = []
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        model = _Stack(3, 8)
        mesh = init_device_mesh("cpu", (1,))
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
            layer.register_forward_pre_hook(lambda m, a: seen.append(a[1].kcp))
        fully_shard(model, mesh=mesh)

        def step(meta: BatchMeta) -> tuple[int, int]:
            collect = _counting(monkeypatch, "_collect_grad_tensors")
            replace = _counting(monkeypatch, "_replace_grad_tensors")
            x = torch.randn(2, 8, requires_grad=True)
            model(x, meta).sum().backward()
            monkeypatch.undo()
            return collect[0], replace[0]

        step(BatchMeta(cu, None))  # warm-up
        base = step(BatchMeta(cu, None))
        seen.clear()
        assert step(BatchMeta(cu, ctx)) == base
        assert len(seen) == 3 and all(s is ctx for s in seen)
    finally:
        dist.destroy_process_group()
