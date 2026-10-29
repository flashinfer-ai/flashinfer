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

One KCP workspace shared by several prepared packs (``cake_kda_train.cake_kcp.KcpWorkspace``): ``reserve`` sizes it
for a list of contexts (each pool at its largest need, the local arena at the largest layout, a plan view of it per
other pack), ``kcp_workspace_bytes([ctx, ...], H=)`` predicts its bytes, and the calls of the reserved packs, in any
interleaving, allocate nothing, keep every pointer and give bitwise the results of one workspace per pack.  The
workspace holds no context, its host caches are bounded, a call on another stream is ordered after the last one on
the device, and ``allocate=False`` sizes the first forward for the backward.  The multi-GPU rows are in
``test_cake_kda_train_kcp_dist.py``-style launch forms at the end of this file (``spawn`` or one process per Slurm
task).
"""

import gc
import json
import os
import socket
import tempfile
import time
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_helpers.cake_host_sync import (
    allocator_counts,
    census,
    nonblocking,
    sleep_cycles,
    summary,
)
from tests.test_helpers.cake_kda_train_kcp_saved import written_tensors
from tests.test_helpers.cake_package import cake_package

kp = cake_package("cake_kda_train", "cake_kcp_plan")
cake_kcp = cake_package("cake_kda_train", "cake_kcp")
cake_backend = cake_package("cake_kda_train", "cake_backend")
cake_jit = cake_package("cake_kda_train", "cake_jit")

SM_COUNT = 152
# the document lengths of four packs of one rank: different T_r and n_frag; at H = 4 the one-document and
# two-document packs split into P > 1 segments, at H = 64 every pack runs P = 1
PACKS = ((1000, 577), (50, 640, 77, 1000, 129), (1450,), (300, 64, 700))
HEADS = (4, 64)


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def _part(lengths, W: int = 1, layout: str = "contiguous"):
    return kp.kcp_partition(
        torch.tensor(_cu(lengths), dtype=torch.int64), W, layout=layout
    )


def _new_ws(**kw):
    backend = cake_kcp.get_backend()
    return cake_kcp.KcpWorkspace(
        state_dtype=backend.dtypes["state"], local=backend.local_stages, **kw
    )


def _storages(ws) -> list[torch.Tensor]:
    """Every device storage the workspace holds, once: the pools, the local arena and the scratch table sets of the
    arena and its cached plan views."""
    out = {t.untyped_storage().data_ptr(): t for t in ws._pools.values() if t.numel()}
    if ws.local is not None:
        out[ws.local.arena.data_ptr()] = ws.local.arena
        for arena in (ws.local, *ws._views.values()):
            tables = arena.__dict__.get(cake_kcp.SCRATCH_TABLES_ATTR)
            if tables:
                first = next(iter(tables.values()))
                out[first.untyped_storage().data_ptr()] = first
    return list(out.values())


def _pointers(ws) -> dict:
    ptrs = {name: t.data_ptr() for name, t in ws._pools.items()}
    if ws.local is not None:
        ptrs["arena"] = ws.local.arena.data_ptr()
        ptrs["arena_id"] = id(ws.local)
        ptrs["views"] = {key: id(v) for key, v in ws._views.items()}
    return ptrs


# --------------------------------------------------------------------------- CPU


def test_byte_query_of_a_list_is_the_per_pool_max() -> None:
    """``kcp_workspace_bytes([ctx, ...])`` (host only): a list of one context is that context's bytes; a list gives
    each pool at its largest need over the contexts (a :data:`SHARED_POOLS` pair one storage of the larger), the
    largest local arena and each distinct pack's scratch table set -- computed here independently from the
    single-context queries; a ``reserve`` of the same list on a pools-only workspace allocates exactly that pool
    term, whichever order the contexts come in, and a later ``ensure`` of any of them grows nothing.  Rows: the
    C / Z / D partitions at W = 2 / 4 / 8, every rank's four packs, H = 4 / 64, backward and forward-only."""
    seen = 0
    for W in (2, 4, 8):
        for layout in ("contiguous", "zigzag", "zigzag_per_document"):
            parts = [_part(lengths, W, layout) for lengths in PACKS]
            for r in range(W):
                ctxs = [
                    kp.build_kcp_context(p, None, r, "cpu", 4, sm_count=SM_COUNT)
                    for p in parts
                ]
                for H in HEADS:
                    for backward in (True, False):
                        kw = {"H": H, "backward": backward}
                        single = [kp.kcp_workspace_bytes(c, **kw) for c in ctxs]
                        assert kp.kcp_workspace_bytes([ctxs[0]], **kw) == single[0]
                        lays = [kp.kcp_arena_layout(c, **kw) for c in ctxs]
                        need = {
                            key[5:]: max(lay[key] for lay in lays)
                            for key in lays[0]
                            if key.startswith("pool.")
                        }
                        lazy = kp.LAZY_POOLS + (() if backward else kp.FWD_LAZY_POOLS)
                        paired = {n for pair in kp.SHARED_POOLS for n in pair}
                        pools = sum(
                            v
                            for n, v in need.items()
                            if n not in lazy and n not in paired
                        ) + sum(
                            max(need.get(n, 0) for n in pair)
                            for pair in kp.SHARED_POOLS
                        )
                        local = [
                            s - lay["arena"]
                            for s, lay in zip(single, lays, strict=True)
                        ]
                        arena = [
                            cake_backend.workspace_bytes(
                                c.T_r,
                                c.n_frag,
                                H,
                                backward=backward,
                                sm_count=SM_COUNT,
                                fwd_layout=kp.local_fwd_layout(backward),
                            )
                            if c.T_r
                            else 0
                            for c in ctxs
                        ]
                        scratch = {
                            (c.T_r, c.n_frag): loc - a
                            for c, loc, a in zip(ctxs, local, arena, strict=True)
                        }
                        got = kp.kcp_workspace_bytes(ctxs, **kw)
                        assert got == max(arena) + sum(scratch.values()) + pools, (
                            W,
                            layout,
                            r,
                            H,
                            backward,
                        )
                        assert max(single) <= got < sum(single)
                        if backward:
                            for order in (ctxs, ctxs[::-1]):
                                ws = cake_kcp.KcpWorkspace(local=False)
                                ws.reserve(order, H)
                                assert ws.bytes == pools, (W, layout, r, H)
                                ptrs = _pointers(ws)
                                for c in order:
                                    ws.ensure(c, H, device="cpu", backward=False)
                                assert _pointers(ws) == ptrs and ws.bytes == pools
                            ws_r = cake_kcp.KcpWorkspace(local=False)
                            ws_r.reserve(ctxs, H, ring=True)  # the hop pools too
                            assert ws_r.bytes == pools + sum(
                                max(lay[f"pool.{n}"] for lay in lays)
                                for n in kp.LAZY_POOLS
                            )
                        seen += 1
    assert seen > 0


def test_reserve_floor_and_allocate_false_size_forwards_for_the_backward() -> None:
    """``reserve`` and ``KcpWorkspace(ctx, H, allocate=False)`` keep a backward sizing floor: a later forward ensure
    sizes the backward pools too (no forward-only state to upgrade); ``allocate=False`` allocates nothing until the
    first ensure; a workspace without a context keeps today's forward-only first state."""
    part = _part((1000, 577), 4, "zigzag_per_document")
    ctx = kp.build_kcp_context(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
    full = kp.kcp_arena_layout(ctx, H=4)["arena"]
    fwd = kp.kcp_arena_layout(ctx, H=4, backward=False)["arena"]
    assert fwd < full
    ws = cake_kcp.KcpWorkspace(ctx, 4, device="cpu", local=False, allocate=False)
    assert ws.bytes == 0 and not ws._pools and ws.local is None
    ws.ensure(ctx, 4, device="cpu", backward=False)
    assert ws.bytes == full and ws.has("b1_send")
    plain = cake_kcp.KcpWorkspace(local=False)
    plain.ensure(ctx, 4, device="cpu", backward=False)
    assert plain.bytes == fwd and not plain.has("b1_send")
    reserved = cake_kcp.KcpWorkspace(local=False)
    reserved.reserve([ctx], 4)
    ptrs = _pointers(reserved)
    reserved.ensure(ctx, 4, device="cpu", backward=False)
    assert reserved.bytes == full and _pointers(reserved) == ptrs
    with pytest.raises(ValueError):
        reserved.reserve([], 4)


def test_shared_workspace_holds_no_context_and_bounds_its_caches() -> None:
    """A long-lived workspace keeps no context alive (the trimmed path's ensure records hold the context and its
    table set weakly; the table set lives on the context); its pool-view and ensure-record caches stay bounded,
    and a record per context lets interleaved packs keep the fast path."""
    part = _part((1000, 577))
    ws = cake_kcp.KcpWorkspace(local=False)
    refs = []
    for _ in range(3):
        ctx = kp.build_kcp_context(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
        tb = ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True)
        assert ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True) is tb
        refs.append((weakref.ref(ctx), weakref.ref(tb)))
        del ctx, tb
    gc.collect()
    assert all(a() is None and b() is None for a, b in refs)
    assert ws._fast and all(rec[0]() is None for rec in ws._fast.values())
    # a dead ensure record misses: the next ensure takes the full path
    ctx = kp.build_kcp_context(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
    assert ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True) is not None
    n = ws._pools["c1_send"].numel()
    for i in range(1, cake_kcp.SLICE_CACHE_MAX + 50):
        ws.pool("c1_send", min(i, n), cached=True)
        assert len(ws._slices) <= cake_kcp.SLICE_CACHE_MAX
    # interleaved packs: one record each, every one a hit while no storage changes
    other = kp.build_kcp_context(
        _part((50, 640, 77)), None, 0, "cpu", 4, sm_count=SM_COUNT
    )
    for _ in range(2):  # the first round may grow a pool, which drops the records
        tbs = [
            ws.ensure(c, 64, device="cpu", backward=True, host_trim=True)
            for c in (ctx, other)
        ]
    calls = []
    real = ws.tables
    ws.tables = lambda *a, **k: calls.append(1) or real(*a, **k)
    for _ in range(3):
        for c, tb in zip((ctx, other), tbs, strict=True):
            assert ws.ensure(c, 64, device="cpu", backward=True, host_trim=True) is tb
    assert not calls  # no full-path ensure
    ws.tables = real
    part = _part((50, 640, 77))
    for _ in range(cake_kcp.ENSURE_CACHE_MAX + 10):
        c = kp.build_kcp_context(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
        ws.ensure(c, 64, device="cpu", backward=True, host_trim=True)
        assert len(ws._fast) <= cake_kcp.ENSURE_CACHE_MAX


# --------------------------------------------------------------------------- GPU


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {tuple(cake_jit.ARCHES)}"
        )
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _operands(device, T: int, H: int, seed: int) -> dict:
    """Seeded operands of the delta rule's contract (q / k L2-normalized per head, natural-log decay in (-5, 0)) and
    a ``dout``, in the backend's dtypes."""
    dt = cake_kcp.get_backend().dtypes
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)

    def unit(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    D = cake_kcp.D
    return {
        "q": unit(randn(T, H, D)).to(dt["q"]),
        "k": unit(randn(T, H, D)).to(dt["k"]),
        "v": (0.5 * randn(T, H, D)).to(dt["v"]),
        "g": (-5.0 * torch.sigmoid(randn(T, H, D))).to(dt["g"]),
        "beta": torch.sigmoid(randn(T, H)).to(dt["beta"]),
        "dout": (0.5 * randn(T, H, D)).to(dt["dout"]),
    }


def _packs(device, H: int, packs=PACKS, W: int = 1, rank: int = 0, layout="contiguous"):
    ctxs = [
        kp.build_kcp_context(_part(lengths, W, layout), None, rank, device)
        for lengths in packs
    ]
    ops = [_operands(device, int(c.T_r), H, 0x7815 + i) for i, c in enumerate(ctxs)]
    return ctxs, ops


def _fwd(ctx, o, ws, **kw):
    return cake_kcp.chunk_kda_kcp_fwd(
        o["q"], o["k"], o["v"], o["g"], o["beta"], ctx, workspace=ws, **kw
    )


def _bwd(ctx, o, ws, saved, **kw):
    return cake_kcp.chunk_kda_kcp_bwd(
        o["q"],
        o["k"],
        o["v"],
        o["g"],
        o["beta"],
        ctx,
        o["dout"],
        saved=saved,
        workspace=ws,
        **kw,
    )


def _schedule(ctxs, ops, wss, *, poison=None, **kw) -> list[dict]:
    """One interleaved pass over the packs (``wss[i]``: the workspace of pack ``i``; one object for a shared one):
    the saving forwards in pack order, the replay-mode forwards in reverse order, the saved-mode backwards in pack
    order, the replay-mode backwards in reverse order, the forwards without a saved set, then the checkpointed
    autograd forwards of every pack before one backward through all of them.  ``poison(ws)`` runs before every
    forward (the arena filled with NaN).  Returns per pack its outputs, written saved sets and gradients."""
    n = len(ctxs)
    if not isinstance(wss, list):  # one shared workspace
        wss = [wss] * n
    res: list[dict] = [{} for _ in range(n)]
    saved = [None] * n

    def pre(i):
        if poison is not None:
            poison(wss[i])

    for i in range(n):
        pre(i)
        out, s = _fwd(ctxs[i], ops[i], wss[i], **kw)
        saved[i] = s
        res[i]["out"] = out
        res[i]["saved"] = written_tensors(s)
    for i in reversed(range(n)):
        pre(i)
        out, none = _fwd(ctxs[i], ops[i], wss[i], boundary_mode="replay", **kw)
        assert none is None
        res[i]["out_replay"] = out
    for i in range(n):
        res[i]["grads"] = _bwd(ctxs[i], ops[i], wss[i], saved[i], **kw)
    for i in reversed(range(n)):
        res[i]["grads_replay"] = _bwd(ctxs[i], ops[i], wss[i], None, **kw)
    for i in range(n):
        pre(i)
        out, none = _fwd(ctxs[i], ops[i], wss[i], save_for_backward=False, **kw)
        res[i]["out_nosave"] = out
    leaves, outs = [], []
    for i in range(n):
        pre(i)
        lf = [
            ops[i][name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        ]
        out = torch.utils.checkpoint.checkpoint(
            lambda *a, i=i: cake_kcp.chunk_kda_kcp(
                *a, ctxs[i], None, "light", "saved", wss[i], **kw
            ),
            *lf,
            use_reentrant=False,
        )
        leaves.append(lf)
        outs.append(out)
    torch.autograd.backward(outs, [o["dout"] for o in ops])
    for i in range(n):
        res[i]["out_ckpt"] = outs[i].detach()
        res[i]["grads_ckpt"] = tuple(t.grad for t in leaves[i])
    return res


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))


def _poison(ws) -> None:
    if ws.local is not None:
        ws.local.arena.fill_(0xFF)


def test_gpu_reserved_shared_workspace_is_bitwise_per_pack(device) -> None:
    """Four packs of one rank (different ``T_r`` / ``n_frag``; P > 1 and P = 1) at H = 4 and 64 on ONE reserved
    workspace, in an interleaved schedule (saving and replay-mode forwards, saved and replay-mode backwards, forwards
    without a saved set, checkpointed autograd through all packs; the arena NaN-filled before every forward): out,
    the written saved sets and every gradient are bitwise those of one workspace per pack.  ``reserve`` allocates
    exactly the workspace's storages (``memory_allocated`` delta = their allocator-rounded bytes) and its ``bytes``
    equal ``kcp_workspace_bytes(ctxs, H=H)``, below the sum of the per-pack workspaces; the local arena is the
    largest per-pack arena; after two schedules no pointer, arena, view or byte count has changed."""
    seen_p1 = seen_pn = 0
    for H in HEADS:
        ctxs, ops = _packs(device, H)
        Ps = [cake_kcp.KcpWorkspace(local=False).tables(c, H).P for c in ctxs]
        seen_p1 += sum(P == 1 for P in Ps)
        seen_pn += sum(P > 1 for P in Ps)
        per = [_new_ws() for _ in ctxs]
        for c, w in zip(ctxs, per, strict=True):
            w.ensure(c, H, device=device, backward=True)
        per_bytes = [w.bytes for w in per]
        assert per_bytes == [kp.kcp_workspace_bytes(c, H=H) for c in ctxs]
        per_arena = max(w.local.bytes for w in per)
        ref = _schedule(ctxs, ops, per)
        del per
        gc.collect()
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
        ws = _new_ws()
        got = ws.reserve(ctxs, H)
        torch.cuda.synchronize(device)
        delta = torch.cuda.memory_allocated(device) - base
        rounded = sum(
            -(-t.untyped_storage().nbytes() // 512) * 512 for t in _storages(ws)
        )
        pred = kp.kcp_workspace_bytes(ctxs, H=H)
        print(
            f"SHARED_WS H={H} P={Ps} per_pack={per_bytes} sum={sum(per_bytes)} max={max(per_bytes)} "
            f"shared={ws.bytes} pred={pred} delta={delta} rounded={rounded}"
        )
        assert got == ws.bytes == pred, (H, got, pred)
        # the caching allocator may hand out a cached block up to 1 MiB larger than a request without splitting it
        assert rounded <= delta < rounded + len(_storages(ws)) * (1 << 20), (
            H,
            delta,
            rounded,
        )
        assert ws.local.bytes == per_arena and ws.local.backward
        assert ws.bytes < sum(per_bytes)
        ptrs = _pointers(ws)
        for rnd in range(2):
            res = _schedule(ctxs, ops, ws, poison=_poison if rnd == 0 else None)
            for i in range(len(ctxs)):
                for key, val in ref[i].items():
                    assert _same(res[i][key], val), (H, rnd, i, Ps[i], key)
            assert _pointers(ws) == ptrs and ws.bytes == pred, (H, rnd)
        del ws, res, ref
    assert seen_p1 > 0 and seen_pn > 0, (seen_p1, seen_pn)


def test_gpu_unreserved_shared_workspace_grows_to_the_max_only(device) -> None:
    """Without ``reserve``: a workspace that first serves the smallest pack grows its arena only when a pack does not
    fit (a larger pack's arena replaces it, released first), never for a smaller one, and a forward never turns its
    backward arena into a forward-only one; after one pass over the packs it holds the bytes of the reserved form
    and the results are bitwise one workspace per pack's."""
    H = 4
    ctxs, ops = _packs(device, H)
    order = sorted(range(len(ctxs)), key=lambda i: ctxs[i].T_r)
    ctxs = [ctxs[i] for i in order]
    ops = [ops[i] for i in order]
    per = [_new_ws() for _ in ctxs]
    ref = _schedule(ctxs, ops, per)
    del per
    ws = _new_ws()
    replaced = 0
    for j, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
        before = ws.local
        out, saved = _fwd(c, o, ws)
        if j > 0:  # a forward never turns the backward arena into a forward-only one
            assert ws.local.backward, j
            replaced += ws.local is not before
        _bwd(c, o, ws, saved)
        assert ws.local.backward, j
        cur = ws.local
        for k in range(
            j
        ):  # every pack seen before fits: it runs on a view, no replacement
            out_k, _ = _fwd(ctxs[k], ops[k], ws)
            assert ws.local is cur, (j, k)
            del out_k
        del out, saved
    res = _schedule(ctxs, ops, ws)
    for i in range(len(ctxs)):
        for key, val in ref[i].items():
            assert _same(res[i][key], val), (i, key)
    assert ws.bytes == kp.kcp_workspace_bytes(ctxs, H=H)
    print(f"UNRESERVED replaced={replaced} bytes={ws.bytes}")


def test_gpu_allocate_false_sizes_the_first_forward_for_the_backward(device) -> None:
    """B T3 with the sizing floor (compared after the first backward): ``KcpWorkspace(ctx, H, allocate=False)``
    allocates nothing; its first forward allocates the backward workspace (``kcp_workspace_bytes(ctx, H=H)``, the
    backward arena) and the first backward allocates nothing more and moves no pointer; the results are bitwise
    those of ``KcpWorkspace(ctx, H)``."""
    for H in HEADS:
        ctxs, ops = _packs(device, H, packs=PACKS[:2])
        ctx, o = ctxs[0], ops[0]
        ref_ws = _new_ws()
        ref_ws.ensure(ctx, H, device=device)
        out_r, saved_r = _fwd(ctx, o, ref_ws)
        grads_r = _bwd(ctx, o, ref_ws, saved_r)
        gc.collect()
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
        ws = _new_ws(ctx=ctx, H=H, device=device, allocate=False)
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_allocated(device) == base and ws.bytes == 0
        out, saved = _fwd(ctx, o, ws)
        assert ws.local.backward and ws.bytes == kp.kcp_workspace_bytes(ctx, H=H)
        ptrs = _pointers(ws)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        grads = _bwd(ctx, o, ws, saved)
        assert _pointers(ws) == ptrs and ws.bytes == kp.kcp_workspace_bytes(ctx, H=H)
        assert torch.equal(out, out_r) and _same(grads, grads_r), H
        assert _same(written_tensors(saved), written_tensors(saved_r)), H
        del grads
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_allocated(device) <= before, H
        del ws, ref_ws, out, saved, out_r, saved_r, grads_r


def test_gpu_four_pack_step_peak_drops_by_sum_minus_max(device) -> None:
    """The in-repo memory proxy (four prepared packs, one checkpointed KDA layer per pack, forwards of all packs then
    one backward through all of them, at H = 64): the step's ``max_memory_allocated`` with one workspace per pack
    minus the step's with one reserved shared workspace equals the per-pack workspaces' bytes minus the shared one's
    (within allocator rounding), with bitwise equal gradients."""
    H = 64
    ctxs, ops = _packs(device, H)

    def step(wss) -> tuple[int, list]:
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        leaves, outs = [], []
        for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
            lf = [
                o[name].detach().clone().requires_grad_(True)
                for name in ("q", "k", "v", "g", "beta")
            ]
            outs.append(
                torch.utils.checkpoint.checkpoint(
                    lambda *a, c=c, i=i: cake_kcp.chunk_kda_kcp(
                        *a, c, None, "light", "saved", wss[i]
                    ),
                    *lf,
                    use_reentrant=False,
                )
            )
            leaves.append(lf)
        torch.autograd.backward(outs, [o["dout"] for o in ops])
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - base
        grads = [
            [outs[i].detach()] + [t.grad for t in leaves[i]] for i in range(len(ctxs))
        ]
        return peak, grads

    def arm(shared: bool) -> tuple[int, int, list]:
        gc.collect()
        torch.cuda.synchronize(device)
        base = torch.cuda.memory_allocated(device)
        if shared:
            ws = _new_ws()
            ws.reserve(ctxs, H)
            wss = [ws] * len(ctxs)
            held = ws.bytes
        else:
            wss = [_new_ws() for _ in ctxs]
            for c, w in zip(ctxs, wss, strict=True):
                w.ensure(c, H, device=device)
            held = sum(w.bytes for w in wss)
        step(wss)  # warm-up: the caching allocator's blocks of the step's own tensors
        gc.collect()
        torch.cuda.synchronize(device)
        mid = torch.cuda.memory_allocated(device)
        peak, grads = step(wss)
        peak_total = mid - base + peak
        del wss
        return held, peak_total, grads

    warm = [_new_ws() for _ in ctxs]
    step(
        warm
    )  # the contexts' tables are copied by their first calls: before either arm
    del warm
    held_p, peak_p, grads_p = arm(False)
    held_s, peak_s, grads_s = arm(True)
    print(
        f"PROXY4 H={H} per_pack_ws={held_p} shared_ws={held_s} peak_per_pack={peak_p} peak_shared={peak_s} "
        f"drop={peak_p - peak_s} expected={held_p - held_s}"
    )
    assert _same(grads_p, grads_s)
    assert abs((peak_p - peak_s) - (held_p - held_s)) <= (1 << 20), (
        peak_p,
        peak_s,
        held_p,
        held_s,
    )


def test_gpu_shared_workspace_steady_calls_do_not_block_or_allocate(device) -> None:
    """Census (``tests/test_helpers/cake_host_sync.py``) of a steady interleaved round of the four packs (saving
    forwards, then saved-mode backwards in reverse order) on one reserved workspace: no pageable or pinned copy, no
    synchronizing operation, the queued probe still runs when the host returns, and no device allocation or free;
    bitwise the first round."""
    alloc_ok = allocator_counts(device)
    H = 4
    ctxs, ops = _packs(device, H)
    ws = _new_ws()
    ws.reserve(ctxs, H)

    def rnd():
        saved, res = [], []
        for c, o in zip(ctxs, ops, strict=True):
            out, s = _fwd(c, o, ws)
            saved.append(s)
            res.append(out)
        for i in reversed(range(len(ctxs))):
            res.extend(_bwd(ctxs[i], ops[i], ws, saved[i]))
        return res

    first = rnd()
    rnd()
    c = census(rnd, device)
    print(f"CENSUS_SHARED {summary(c)}")
    assert nonblocking(c), summary(c)
    assert c["pinned"] == 0, summary(c)
    if alloc_ok:
        assert (c["device_alloc"], c["device_free"], c["alloc_retries"]) == (0, 0, 0), (
            summary(c)
        )
        assert c["prof_malloc"] in (None, 0), summary(c)
    assert _same(first, c["out"])
    if not alloc_ok:
        pytest.skip(
            "the caching allocator's statistics do not count device allocations here"
        )


def test_gpu_fresh_contexts_on_a_long_lived_shared_workspace_release_everything(
    device,
) -> None:
    """A training loop that prepares NEW contexts every step and runs them on one long-lived reserved workspace keeps
    no context: with the cycle collector disabled every step's contexts and table sets are freed by reference
    counting alone, ``memory_allocated`` returns to its level after the first step, and the results are bitwise the
    first step's."""
    H = 4

    def step(ws):
        ctxs, ops = _packs(device, H)
        ws.reserve(ctxs, H)
        res = []
        for c, o in zip(ctxs, ops, strict=True):
            out, s = _fwd(c, o, ws)
            res += [out.detach().clone(), *(t.clone() for t in _bwd(c, o, ws, s))]
        refs = [weakref.ref(c) for c in ctxs] + [
            weakref.ref(ws.tables(c, H)) for c in ctxs
        ]
        return res, refs

    ws = _new_ws()
    first, refs = step(ws)
    gc.collect()
    torch.cuda.synchronize(device)
    assert all(r() is None for r in refs)
    base = torch.cuda.memory_allocated(device)
    enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(3):
            res, refs = step(ws)
            assert _same(first, res), i
            del res
            torch.cuda.synchronize(device)
            assert all(r() is None for r in refs), (
                i,
                "a context or table set is retained",
            )
            assert torch.cuda.memory_allocated(device) == base, i
    finally:
        if enabled:
            gc.enable()


def test_gpu_call_on_another_stream_waits_for_the_last_one(device) -> None:
    """Stream rule: a call on another stream is ordered after the work the workspace's last stream queued (one
    ``wait_stream``, no host wait, no error), and its results are bitwise those of the same call on one stream."""
    H = 4
    ctxs, ops = _packs(device, H, packs=PACKS[:2])
    ws = _new_ws()
    ws.reserve(ctxs, H)
    ref_out, ref_saved = _fwd(ctxs[1], ops[1], ws)
    ref_grads = _bwd(ctxs[1], ops[1], ws, ref_saved)
    torch.cuda.synchronize(device)
    s1, s2 = torch.cuda.Stream(device), torch.cuda.Stream(device)
    cycles = sleep_cycles(device, 1500.0)
    for s in (s1, s2):
        s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s1):
        torch.cuda._sleep(cycles)
        out0, _ = _fwd(ctxs[0], ops[0], ws)
    t0 = time.perf_counter()
    with torch.cuda.stream(s2):
        out, saved = _fwd(ctxs[1], ops[1], ws)
        grads = _bwd(ctxs[1], ops[1], ws, saved)
        done = torch.cuda.Event()
        done.record()
    host = time.perf_counter() - t0
    time.sleep(0.3)
    waited = not done.query()  # still queued behind s1's sleep
    torch.cuda.synchronize(device)
    assert host < 1.0, host
    assert waited, (
        "the call on the second stream did not wait for the first stream's queued work"
    )
    assert ws._stream is not None and ws._stream[0] == s2.cuda_stream
    assert torch.equal(out, ref_out) and _same(grads, ref_grads)
    del out0


def test_gpu_graph_replay_on_a_reserved_shared_workspace(device) -> None:
    """``graph="replay"`` with two packs on one reserved workspace, interleaved over four steps: each pack's keys
    capture once and replay (the pointers never move), bitwise the eager per-pack results; a pack that does not fit
    the arena then replaces it and drops every cached graph that pins the old arena or one of its plan views."""
    H = 4
    ctxs, ops = _packs(device, H, packs=(PACKS[0], PACKS[2]))
    per = [_new_ws() for _ in ctxs]
    refs = []
    for c, o, w in zip(ctxs, ops, per, strict=True):
        out, s = _fwd(c, o, w)
        refs.append((out, _bwd(c, o, w, s)))
    cake_kcp.graph_cache_clear()
    ws = _new_ws()
    ws.reserve(ctxs, H)
    ptrs = _pointers(ws)
    try:
        events = []
        for _step in range(4):
            row = []
            for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
                out, s = _fwd(c, o, ws, graph="replay")
                row.append(cake_kcp.LAST_CALL["fwd"]["graph_event"])
                grads = _bwd(c, o, ws, s, graph="replay")
                row.append(cake_kcp.LAST_CALL["bwd"]["graph_event"])
                assert torch.equal(out, refs[i][0]) and _same(grads, refs[i][1]), (
                    _step,
                    i,
                )
                del out, s, grads
            events.append(row)
        print(f"GRAPH_EVENTS {events}")
        assert events[-1] == ["replay"] * 4, events
        assert all(e.disabled is None for e in cake_kcp._GRAPHS.values())
        assert _pointers(ws) == ptrs
        old = ws.local
        old_views = list(ws._views.values())
        assert old_views, "the reserved packs share no plan view"
        pinned = [
            e
            for e in cake_kcp._GRAPHS.values()
            if any(p is old or p in old_views for p in e.pins)
        ]
        assert pinned
        big, big_ops = _packs(device, H, packs=((4000, 1700),))
        _fwd(big[0], big_ops[0], ws)
        assert ws.local is not old
        for e in cake_kcp._GRAPHS.values():
            assert not any(
                isinstance(p, cake_backend.KdaWorkspace) and p.arena is old.arena
                for p in e.pins
            )
    finally:
        cake_kcp.graph_cache_clear()


# --------------------------------------------------------------------------- multi-GPU

# (packs, H, W, layout, exchange): four prepared packs per rank on one reserved workspace per rank; a2a and ring
# (every document of the ring rows spans at most three ranks), P = 1 and P > 1 (H = 4), forward-only and training
DIST_ROWS = [
    (
        ((3000,), (2048, 5, 3000, 1), (700, 1, 1300, 999), (1200, 900, 900)),
        4,
        4,
        "zigzag_per_document",
        "a2a",
    ),
    (
        ((100, 640, 77, 1000, 129), (2000, 1000), (3000,), (500, 2500)),
        64,
        4,
        "contiguous",
        "a2a",
    ),
    (
        ((100, 640, 77, 1000, 129), (2000, 1000), (1500, 1500), (1000, 2000)),
        4,
        4,
        "contiguous",
        "ring",
    ),
    (
        ((5000, 1003), (700, 1, 1300, 999, 3000), (3000,), (2500, 2500, 1000)),
        4,
        8,
        "zigzag_per_document",
        "a2a",
    ),
    (
        ((5000, 1003), (700, 1, 1300, 999, 3000), (6000,), (2500, 2500, 1000)),
        64,
        8,
        "zigzag",
        "a2a",
    ),
    (
        (
            (1000, 1000, 1000, 1000),
            (500,) * 8,
            (800, 700, 900, 600, 1000),
            (300, 900, 1000, 800, 1000),
        ),
        4,
        8,
        "contiguous",
        "ring",
    ),
]


def _dist_row_id(row) -> str:
    packs, H, W, layout, exchange = row
    return f"p{len(packs)}-h{H}-w{W}-{layout}-{exchange}"


def _run_dist_rank(rank: int, W: int, row, port: int, result_dir: str | None) -> dict:
    """One rank: the four packs on one workspace per pack (reference) and on one reserved shared workspace (the
    interleaved schedule, the arena NaN-filled before every forward), bitwise compared on this rank; the shared
    workspace's bytes against ``kcp_workspace_bytes(ctxs)`` and the per-pack sum, pointer stability over two rounds.
    The verdicts are gathered to every rank."""
    packs, H, _w, layout, exchange = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD if W > 1 else None
    ctxs = [
        kp.build_kcp_context(_part(lengths, W, layout), group, rank, device)
        for lengths in packs
    ]
    ops = [
        _operands(device, int(c.T_r), H, 0x7815 + 97 * i + rank)
        for i, c in enumerate(ctxs)
    ]
    kw = {"exchange": exchange}
    if exchange == "ring":
        assert all(c.ring_eligible for c in ctxs), (
            "a ring row holds a document over more than three ranks"
        )
    ring = exchange == "ring"
    report: dict = {"rank": rank, "row": _dist_row_id(row), "pass": True, "fail": []}
    per = [_new_ws() for _ in ctxs]
    ref = _schedule(ctxs, ops, per, **kw)
    report["per_pack_bytes"] = [w.bytes for w in per]
    report["P"] = [cake_kcp.KcpWorkspace(local=False).tables(c, H).P for c in ctxs]
    del per
    gc.collect()
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    ws = _new_ws()
    ws.reserve(ctxs, H, ring=ring)
    torch.cuda.synchronize(device)
    report["reserve_delta"] = torch.cuda.memory_allocated(device) - base
    report["rounded"] = sum(
        -(-t.untyped_storage().nbytes() // 512) * 512 for t in _storages(ws)
    )
    report["shared_bytes"] = ws.bytes
    # the ring rows reserve the hop pools too: each at its largest need over the packs
    report["pred_bytes"] = kp.kcp_workspace_bytes(ctxs, H=H) + (
        sum(
            max(kp.kcp_arena_layout(c, H=H)[f"pool.{name}"] for c in ctxs)
            for name in kp.LAZY_POOLS
        )
        if ring
        else 0
    )
    report["storages"] = len(_storages(ws))
    ptrs = _pointers(ws)
    poison = _poison if ws.local is not None else None
    for rnd in range(2):
        res = _schedule(ctxs, ops, ws, poison=poison if rnd == 0 else None, **kw)
        for i in range(len(ctxs)):
            for key, val in ref[i].items():
                if not _same(res[i][key], val):
                    report["fail"].append(f"bitwise r{rnd} pack{i} {key}")
        if _pointers(ws) != ptrs:
            report["fail"].append(f"pointers moved r{rnd}")
    if report["shared_bytes"] != report["pred_bytes"]:
        report["fail"].append("bytes != kcp_workspace_bytes(ctxs)")
    # the caching allocator may hand out a cached block up to 1 MiB larger than a request without splitting it
    if not (
        report["rounded"]
        <= report["reserve_delta"]
        < report["rounded"] + report["storages"] * (1 << 20)
    ):
        report["fail"].append("reserve delta outside the rounded storages")
    if sum(1 for b in report["per_pack_bytes"] if b) > 1 and not (
        report["shared_bytes"] < sum(report["per_pack_bytes"])
    ):
        report["fail"].append("shared bytes not below the per-pack sum")
    report["pass"] = not report["fail"]
    if W > 1:
        verdicts = [None] * W
        dist.all_gather_object(verdicts, {k: report[k] for k in report if k != "row"})
        report["ranks"] = verdicts
        report["pass"] = all(v["pass"] for v in verdicts)
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "shared-" + _dist_row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_dist_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        report = _run_dist_rank(rank, W, row, port, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("row", DIST_ROWS, ids=_dist_row_id)
def test_multi_gpu_shared_workspace_rows_are_bitwise_per_pack(row) -> None:
    """The rows of :func:`_run_dist_rank`: one process per GPU (``spawn``, rows up to the visible GPU count) or one
    process per Slurm task (a step with ``SLURM_NTASKS`` > 1 runs the rows of its own world size and skips the
    others)."""
    _packs_, _H, W, _layout, _exchange = row
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"this Slurm step runs the W = {ntasks} rows")
        report = _run_dist_rank(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
        )
        assert report["pass"], report
        return
    _device_or_skip()
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_kcp_shared_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_dist_entry,
            args=(W, row, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(
            os.path.join(result_dir, "shared-" + _dist_row_id(row) + ".json")
        ) as fh:
            report = json.load(fh)
    assert report["pass"], report
