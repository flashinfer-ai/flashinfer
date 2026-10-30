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

One KCP workspace shared by several prepared packs (``cake_kda_det_train_kcp.cake_kcp.WorkspaceDetKcp``): ``reserve`` sizes it
for a list of contexts (each pool at its largest need, the local arena at the largest layout, a plan view of it per
other pack), ``kda_det_workspace_bytes_kcp([ctx, ...], H=)`` predicts its bytes, and the calls of the reserved packs, in any
interleaving, allocate nothing, keep every pointer and give bitwise the results of one workspace per pack.  The
workspace holds no context, its host caches are bounded, a call on another stream is ordered after the last one on
the device and a storage it releases is reused only after that stream's queued work (``record_stream`` on a switch
only), and ``allocate=False`` sizes the first forward for the backward.  Checkpointed layers with one recompute
handle per pack and layer on a shared workspace are bitwise and exchange-free in their recompute, and the tensors a
handle stores are never in the workspace (the other packs' calls cannot overwrite them).  The multi-GPU rows are in
``test_cake_kda_det_train_kcp_dist.py``-style launch forms at the end of this file (``spawn`` or one process per Slurm
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

kp = cake_package("cake_kda_det_train_kcp", "cake_kcp_plan")
cake_kcp = cake_package("cake_kda_det_train_kcp", "cake_kcp")
cake_backend = cake_package("cake_kda_det_train", "cake_backend")
cake_jit = cake_package("cake_kda_det_train", "cake_jit")

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
    return kp.partition_det_kcp(
        torch.tensor(_cu(lengths), dtype=torch.int64), W, layout=layout
    )


def _new_ws(**kw):
    backend = cake_kcp.get_backend()
    return cake_kcp.WorkspaceDetKcp(
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
    """``kda_det_workspace_bytes_kcp([ctx, ...])`` (host only): a list of one context is that context's bytes; a list gives
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
                    kp.build_context_det_kcp(p, None, r, "cpu", 4, sm_count=SM_COUNT)
                    for p in parts
                ]
                for H in HEADS:
                    for backward in (True, False):
                        kw = {"H": H, "backward": backward}
                        single = [kp.kda_det_workspace_bytes_kcp(c, **kw) for c in ctxs]
                        assert (
                            kp.kda_det_workspace_bytes_kcp([ctxs[0]], **kw) == single[0]
                        )
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
                        got = kp.kda_det_workspace_bytes_kcp(ctxs, **kw)
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
                                ws = cake_kcp.WorkspaceDetKcp(local=False)
                                ws.reserve(order, H)
                                assert ws.bytes == pools, (W, layout, r, H)
                                ptrs = _pointers(ws)
                                for c in order:
                                    ws.ensure(c, H, device="cpu", backward=False)
                                assert _pointers(ws) == ptrs and ws.bytes == pools
                            ws_r = cake_kcp.WorkspaceDetKcp(local=False)
                            ws_r.reserve(ctxs, H, ring=True)  # the hop pools too
                            assert ws_r.bytes == pools + sum(
                                max(lay[f"pool.{n}"] for lay in lays)
                                for n in kp.LAZY_POOLS
                            )
                        seen += 1
    assert seen > 0


def test_reserve_floor_and_allocate_false_size_forwards_for_the_backward() -> None:
    """``reserve`` and ``WorkspaceDetKcp(ctx, H, allocate=False)`` keep a backward sizing floor: a later forward ensure
    sizes the backward pools too (no forward-only state to upgrade); ``allocate=False`` allocates nothing until the
    first ensure; a workspace without a context keeps today's forward-only first state."""
    part = _part((1000, 577), 4, "zigzag_per_document")
    ctx = kp.build_context_det_kcp(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
    full = kp.kcp_arena_layout(ctx, H=4)["arena"]
    fwd = kp.kcp_arena_layout(ctx, H=4, backward=False)["arena"]
    assert fwd < full
    ws = cake_kcp.WorkspaceDetKcp(ctx, 4, device="cpu", local=False, allocate=False)
    assert ws.bytes == 0 and not ws._pools and ws.local is None
    ws.ensure(ctx, 4, device="cpu", backward=False)
    assert ws.bytes == full and ws.has("b1_send")
    plain = cake_kcp.WorkspaceDetKcp(local=False)
    plain.ensure(ctx, 4, device="cpu", backward=False)
    assert plain.bytes == fwd and not plain.has("b1_send")
    reserved = cake_kcp.WorkspaceDetKcp(local=False)
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
    ws = cake_kcp.WorkspaceDetKcp(local=False)
    refs = []
    for _ in range(3):
        ctx = kp.build_context_det_kcp(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
        tb = ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True)
        assert ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True) is tb
        refs.append((weakref.ref(ctx), weakref.ref(tb)))
        del ctx, tb
    gc.collect()
    assert all(a() is None and b() is None for a, b in refs)
    assert ws._fast and all(rec[0]() is None for rec in ws._fast.values())
    # a dead ensure record misses: the next ensure takes the full path
    ctx = kp.build_context_det_kcp(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
    assert ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True) is not None
    n = ws._pools["c1_send"].numel()
    for i in range(1, cake_kcp.SLICE_CACHE_MAX + 50):
        ws.pool("c1_send", min(i, n), cached=True)
        assert len(ws._slices) <= cake_kcp.SLICE_CACHE_MAX
    # interleaved packs: one record each, every one a hit while no storage changes
    other = kp.build_context_det_kcp(
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
        c = kp.build_context_det_kcp(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
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
        kp.build_context_det_kcp(_part(lengths, W, layout), None, rank, device)
        for lengths in packs
    ]
    ops = [_operands(device, int(c.T_r), H, 0x7815 + i) for i, c in enumerate(ctxs)]
    return ctxs, ops


def _fwd(ctx, o, ws, **kw):
    return cake_kcp.chunk_kda_det_fwd_kcp(
        o["q"], o["k"], o["v"], o["g"], o["beta"], ctx, workspace=ws, **kw
    )


def _bwd(ctx, o, ws, saved, **kw):
    return cake_kcp.chunk_kda_det_bwd_kcp(
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
            lambda *a, i=i: cake_kcp.chunk_kda_det_kcp(
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
    equal ``kda_det_workspace_bytes_kcp(ctxs, H=H)``, below the sum of the per-pack workspaces; the local arena is the
    largest per-pack arena; after two schedules no pointer, arena, view or byte count has changed."""
    seen_p1 = seen_pn = 0
    for H in HEADS:
        ctxs, ops = _packs(device, H)
        Ps = [cake_kcp.WorkspaceDetKcp(local=False).tables(c, H).P for c in ctxs]
        seen_p1 += sum(P == 1 for P in Ps)
        seen_pn += sum(P > 1 for P in Ps)
        per = [_new_ws() for _ in ctxs]
        for c, w in zip(ctxs, per, strict=True):
            w.ensure(c, H, device=device, backward=True)
        per_bytes = [w.bytes for w in per]
        assert per_bytes == [kp.kda_det_workspace_bytes_kcp(c, H=H) for c in ctxs]
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
        pred = kp.kda_det_workspace_bytes_kcp(ctxs, H=H)
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
    assert ws.bytes == kp.kda_det_workspace_bytes_kcp(ctxs, H=H)
    print(f"UNRESERVED replaced={replaced} bytes={ws.bytes}")


def test_gpu_allocate_false_sizes_the_first_forward_for_the_backward(device) -> None:
    """B T3 with the sizing floor (compared after the first backward): ``WorkspaceDetKcp(ctx, H, allocate=False)``
    allocates nothing; its first forward allocates the backward workspace (``kda_det_workspace_bytes_kcp(ctx, H=H)``, the
    backward arena) and the first backward allocates nothing more and moves no pointer; the results are bitwise
    those of ``WorkspaceDetKcp(ctx, H)``."""
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
        assert ws.local.backward and ws.bytes == kp.kda_det_workspace_bytes_kcp(
            ctx, H=H
        )
        ptrs = _pointers(ws)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        grads = _bwd(ctx, o, ws, saved)
        assert _pointers(ws) == ptrs and ws.bytes == kp.kda_det_workspace_bytes_kcp(
            ctx, H=H
        )
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
                    lambda *a, c=c, i=i: cake_kcp.chunk_kda_det_kcp(
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
    ``wait_stream``, no host wait, no error), and its results are bitwise those of the same call on one stream.  The
    order is checked by events, with no timing assumption: once the second stream's work is complete, an event
    recorded after the first stream's queued call (behind ~1.5 s of device sleep) is complete too.  Without the wait
    the second stream's work completes while the first stream still sleeps."""
    H = 4
    ctxs, ops = _packs(device, H, packs=PACKS[:2])
    ws = _new_ws()
    ws.reserve(ctxs, H)
    ref_out, ref_saved = _fwd(ctxs[1], ops[1], ws)
    ref_grads = _bwd(ctxs[1], ops[1], ws, ref_saved)
    torch.cuda.synchronize(device)
    s1, s2 = torch.cuda.Stream(device), torch.cuda.Stream(device)
    assert s1.cuda_stream != s2.cuda_stream
    cycles = sleep_cycles(device, 1500.0)
    for s in (s1, s2):
        s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s1):
        torch.cuda._sleep(cycles)
        out0, _ = _fwd(ctxs[0], ops[0], ws)
        first = torch.cuda.Event()
        first.record()
    t0 = time.perf_counter()
    with torch.cuda.stream(s2):
        out, saved = _fwd(ctxs[1], ops[1], ws)
        grads = _bwd(ctxs[1], ops[1], ws, saved)
        done = torch.cuda.Event()
        done.record()
    host = time.perf_counter() - t0
    done.synchronize()
    # complete whenever the second stream's work is, if that work waited for it
    ordered = first.query()
    torch.cuda.synchronize(device)
    assert host < 1.0, host
    assert ordered, (
        "the call on the second stream did not wait for the first stream's queued work"
    )
    assert ws._stream is not None and ws._stream[0] == s2.cuda_stream
    assert torch.equal(out, ref_out) and _same(grads, ref_grads)
    del out0


def _live_ranges(ws) -> dict[int, int]:
    """``data_ptr -> nbytes`` of every device storage the workspace holds (:func:`_storages`)."""
    return {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in _storages(ws)
    }


def _hits(tensors, ranges: dict[int, int]) -> list[tuple[int, int]]:
    """The ``(data_ptr, nbytes)`` of ``ranges`` that one of ``tensors`` overlaps."""
    out = []
    for t in tensors:
        a = t.untyped_storage().data_ptr()
        b = a + t.untyped_storage().nbytes()
        out += [(p, n) for p, n in ranges.items() if a < p + n and p < b]
    return out


def _switch_setup(device, H: int):
    """A small pack and a pack whose arena and pools are larger (their tables already copied by a first call), and a
    workspace reserved for the small pack on a fresh stream ``sa``, with the large pack's tables built; plus a second
    fresh stream ``sb``."""
    small, small_ops = _packs(device, H, packs=(PACKS[0],))
    big, big_ops = _packs(device, H, packs=((4000, 1700),))
    warm = _new_ws()
    _fwd(small[0], small_ops[0], warm)
    _fwd(big[0], big_ops[0], warm)
    del warm
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    sa, sb = torch.cuda.Stream(device), torch.cuda.Stream(device)
    with torch.cuda.stream(sa):
        ws = _new_ws()
        ws.reserve(small, H)
        ws.tables(big[0], H)
    torch.cuda.synchronize(device)
    return (small[0], small_ops[0]), (big[0], big_ops[0]), ws, sa, sb


# device sleep of the stream-switch rows, in ms: a later attempt runs only when stream B finished before the host's
# check (an inconclusive attempt, never a pass); each attempt's results are checked whatever its timing
SWITCH_SLEEPS_MS = (1000.0, 4000.0, 16000.0)


def test_gpu_storage_a_switched_stream_used_is_not_reused_while_it_runs(device) -> None:
    """Allocator rule of a stream switch (``record_stream``): a workspace allocated on stream A serves a call on
    stream B queued behind device sleep, and an ``ensure`` on B for a larger pack then replaces its pools and its
    arena.  While B is still busy (checked after the allocations; an attempt where B already finished is repeated
    with a longer sleep), allocations on A of every released storage's size (twice each) never overlap a released
    storage: the caching allocator returns a block to the stream that allocated it, so without the record it would
    hand A the block B's queued work still reads and writes (the negative control: the previous head)."""
    H = 4
    for ms in SWITCH_SLEEPS_MS:
        (sc, so), (bc, _bo), ws, sa, sb = _switch_setup(device, H)
        old = _live_ranges(ws)
        cycles = sleep_cycles(device, ms)
        with torch.cuda.stream(sb):
            torch.cuda._sleep(cycles)
            out, _ = _fwd(sc, so, ws, save_for_backward=False)
            ws.ensure(bc, H, device=device, backward=True)
        live = _live_ranges(ws)
        freed = {p: n for p, n in old.items() if p not in live}
        held = []
        with torch.cuda.stream(sa):
            for n in freed.values():
                for _ in range(2):
                    held.append(torch.empty((n,), dtype=torch.uint8, device=device))
        busy = not sb.query()  # B had not finished when the allocations were made
        hits = _hits(held, freed)
        torch.cuda.synchronize(device)
        print(
            f"STREAM_SWITCH_REUSE sleep_ms={ms} freed={sorted(freed.values())} held={len(held)} busy={busy} "
            f"hits={hits}"
        )
        assert ws.local is not None and ws.local.arena.data_ptr() not in old
        assert len(freed) >= 2, (old, live)  # the arena and at least one pool
        del out, held, ws
        if busy:
            assert not hits, (
                f"a storage stream B's queued work uses was handed to stream A: {hits}"
            )
            return
    pytest.fail(f"stream B finished before the check at every sleep {SWITCH_SLEEPS_MS}")


def test_gpu_stream_switch_with_a_replacement_and_poisoned_reuse_is_bitwise(
    device,
) -> None:
    """A workspace allocated on stream A serves, on stream B behind device sleep, a forward and backward of a small
    pack and then of a larger pack that replaces its pools and arena; once B's sleep ends, stream A fills blocks of
    the released sizes allocated after the switch with NaN, over and over, while B's calls run.  The outputs and
    gradients are bitwise those of one workspace per pack on one stream (checked at every attempt; an attempt whose
    poison was queued only after B finished is repeated with a longer sleep)."""
    H = 4
    for ms in SWITCH_SLEEPS_MS:
        (sc, so), (bc, bo), ws, sa, sb = _switch_setup(device, H)
        refs = []
        for c, o in ((sc, so), (bc, bo)):
            w = _new_ws()
            out, s = _fwd(c, o, w)
            refs.append((out, _bwd(c, o, w, s)))
            del w
        torch.cuda.synchronize(device)
        old = _live_ranges(ws)
        cycles = sleep_cycles(device, ms)
        got = []
        with torch.cuda.stream(sb):
            torch.cuda._sleep(cycles)
            woke = torch.cuda.Event()
            woke.record()
            for c, o in ((sc, so), (bc, bo)):
                out, s = _fwd(c, o, ws)
                got.append((out, _bwd(c, o, ws, s)))
            done = torch.cuda.Event()
            done.record()
        freed = {p: n for p, n in old.items() if p not in _live_ranges(ws)}
        assert len(freed) >= 2, freed
        with torch.cuda.stream(sa):
            held = [
                torch.empty((n // 4,), dtype=torch.float32, device=device)
                for n in freed.values()
            ]
            sa.wait_event(woke)
            for _ in range(64):
                for t in held:
                    t.fill_(float("nan"))
                torch.cuda._sleep(max(cycles // 20000, 1))
        busy = not done.query()  # the poison was queued before B's calls ran
        torch.cuda.synchronize(device)
        print(f"STREAM_SWITCH_POISON sleep_ms={ms} busy={busy}")
        for i, ((out, grads), (rout, rgrads)) in enumerate(zip(got, refs, strict=True)):
            assert torch.equal(out, rout) and _same(grads, rgrads), (ms, i)
        del held, got, refs, ws
        if busy:
            return
    pytest.fail(
        f"stream B finished before the poison was queued at every sleep {SWITCH_SLEEPS_MS}"
    )


def test_gpu_record_stream_only_on_a_stream_switch(device, monkeypatch) -> None:
    """Census of ``record_stream``: steady rounds of two packs on one stream record nothing; the first call on
    another stream records each live storage of the workspace once (every pool, the arena and the scratch table
    sets among them), and the later calls on that stream record nothing again; bitwise the first round."""
    H = 4
    ctxs, ops = _packs(device, H, packs=PACKS[:2])
    ws = _new_ws()
    ws.reserve(ctxs, H)

    def rnd():
        res = []
        for c, o in zip(ctxs, ops, strict=True):
            out, s = _fwd(c, o, ws)
            res += [out, *_bwd(c, o, ws, s)]
        return res

    first = rnd()
    rnd()
    seen: list[tuple[int, int]] = []
    record = torch.Tensor.record_stream

    def counting(self, stream):
        seen.append((self.untyped_storage().data_ptr(), stream.cuda_stream))
        return record(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", counting)
    for _ in range(2):
        assert _same(rnd(), first)
    steady = list(seen)
    s2 = torch.cuda.Stream(device)
    s2.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s2):
        res = rnd()
        switch = list(seen[len(steady) :])
        res2 = rnd()
        after = list(seen[len(steady) + len(switch) :])
    torch.cuda.current_stream(device).wait_stream(s2)
    recorded = {p for p, s in switch if s == s2.cuda_stream}
    print(
        f"RECORD_STREAM steady={len(steady)} switch={len(switch)} storages={len(_live_ranges(ws))} "
        f"recorded={len(recorded & set(_live_ranges(ws)))} after={len(after)}"
    )
    assert steady == [], steady
    assert set(_live_ranges(ws)) <= recorded, (
        "a storage of the workspace was not recorded on the new stream"
    )
    live = {t.untyped_storage().data_ptr() for t in ws._live_storages()}
    mine = sorted(p for p, s in switch if p in live and s == s2.cuda_stream)
    assert mine == sorted(live), (mine, sorted(live))
    assert after == [], after
    assert _same(res, first) and _same(res2, first)


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


# --------------------------------------------------------------------------- recompute handles on a shared workspace

# checkpointed KDA layers per pack: y1 = KDA(q, k, v, g, beta), y2 = KDA(q, k, y1, g, beta) on the pack's context
HANDLE_LAYERS = 2
# "interleaved": every pack's forward, then the backwards in pack order (A fwd, B fwd, A bwd, B bwd);
# "sequential": pack by pack, forward then backward (A fwd, A bwd, B fwd, B bwd)
HANDLE_ORDERS = ("interleaved", "sequential")
RECOMPUTE_EXCHANGES = (
    "a2a:",
    "ring:",
)  # sequence entries of the state exchange (c1 / c2 or the ring hops)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Byte view: -0.0 vs 0.0 and NaN payloads count as differences."""
    return t.detach().contiguous().view(torch.uint8)


def _same_bits(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(_bits(a), _bits(b))
        )
    return len(a) == len(b) and all(_same_bits(x, y) for x, y in zip(a, b, strict=True))


def _poison_all(ws) -> None:
    """The arena and every pool but ``eye`` filled with all-ones bytes (NaN in BF16 / FP32): a stored tensor that
    lived in the workspace would read NaN afterwards.  ``eye`` is the identity the workspace writes once when it
    sizes that pool, a constant every call reads; the other pools carry no contents across calls."""
    _poison(ws)
    eye = ws._pools.get("eye")
    keep = None if eye is None else eye.untyped_storage().data_ptr()
    for t in ws._pools.values():
        if t.numel() and t.untyped_storage().data_ptr() != keep:
            t.view(torch.uint8).fill_(0xFF)


def _overlaps(t: torch.Tensor, storages) -> bool:
    """Whether ``t``'s storage shares a byte with one of ``storages``."""
    if t.numel() == 0:
        return False
    a0 = t.untyped_storage().data_ptr()
    a1 = a0 + t.untyped_storage().nbytes()
    for s in storages:
        b0 = s.untyped_storage().data_ptr()
        if b0 < a1 and a0 < b0 + s.untyped_storage().nbytes():
            return True
    return False


class _Records:
    """Every forward / backward record (``cake_kcp.LAST_CALL`` as ``_record`` writes it) while active."""

    def __enter__(self):
        self.recs: list[tuple[str, dict]] = []
        self._orig = cake_kcp._record

        def record(kind, call):
            self._orig(kind, call)
            self.recs.append((kind, dict(cake_kcp.LAST_CALL[kind])))

        cake_kcp._record = record
        return self

    def __exit__(self, *exc):
        cake_kcp._record = self._orig
        return False

    def phases(self, n_calls: int, exchange: str) -> list[str]:
        """The violations of the handle phases over ``n_calls`` checkpointed calls: one "capture" (the original
        forward, which exchanges) and one "replay" per call; a replay issues no collective and launches no exchange."""
        bad = []
        fwd = [r for kind, r in self.recs if kind == "fwd"]
        capture = [r for r in fwd if r["recompute_handle"] == "capture"]
        replay = [r for r in fwd if r["recompute_handle"] == "replay"]
        if len(capture) != n_calls or len(replay) != n_calls or len(fwd) != 2 * n_calls:
            bad.append(
                f"phases fwd {len(fwd)} capture {len(capture)} replay {len(replay)} != {n_calls}"
            )
        for r in replay:
            if r["n_collectives"] != 0:
                bad.append(f"replay n_collectives {r['n_collectives']}")
            if any(s.startswith(RECOMPUTE_EXCHANGES) for s in r["sequence"]):
                bad.append("replay launches an exchange")
        if exchange == "a2a" and any(r["n_collectives"] != 2 for r in capture):
            bad.append("an original forward without its c1 + c2")
        return bad


def _handle_pass(ctxs, ops, wss, *, keep=None, order="interleaved", poison=None, **kw):
    """Per pack :data:`HANDLE_LAYERS` KDA layers on its context (layer ``l + 1`` reads layer ``l``'s ``out`` as
    ``v``; leaves ``q, k, v, g, beta``; ``backward(dout)`` from the last layer).  ``keep=None``: direct autograd, no
    checkpoint, no handle (the reference).  Otherwise every layer runs under ``torch.utils.checkpoint`` (non-reentrant)
    with its own :class:`RecomputeHandleDetKcp` and ``checkpoint_keep=keep``.  ``wss[i]``: pack ``i``'s workspace (one
    object: shared).  ``poison(ws)`` before every forward and backward.  Right after each checkpointed forward the
    handle's stored tensors are copied; before every backward and at the end every stored tensor is compared bitwise
    with its copy (a stored tensor in the shared workspace would be overwritten by the other packs' calls and
    poisoned), and each must lie outside the workspace's storages.  Returns per pack ``out`` and the five gradients,
    and a log of the violations and of the stored element counts."""
    n = len(ctxs)
    if not isinstance(wss, list):
        wss = [wss] * n
    res: list[dict] = [{} for _ in range(n)]
    handles: list[list] = [[] for _ in range(n)]
    snaps: list[list] = [[] for _ in range(n)]
    seen: list[list] = [[[] for _ in range(HANDLE_LAYERS)] for _ in range(n)]
    log: dict = {
        "changed": [],
        "overlap": [],
        "recompute_out": [],
        "S_in": 0,
        "c1_recv": 0,
    }
    live: dict = {}

    def check(tag):
        for i in range(n):
            for layer, (h, snap) in enumerate(zip(handles[i], snaps[i], strict=True)):
                entry = h.entries.get(cake_kcp.HANDLE_KEY) or {}
                for name, copy in snap.items():
                    t = entry.get(name)
                    if not isinstance(t, torch.Tensor) or not _same_bits(t, copy):
                        log["changed"].append(f"{tag}: pack {i} layer {layer} {name}")

    def forward(i):
        ctx, ws = ctxs[i], wss[i]
        leaves = [
            ops[i][name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        ]
        q, k, y, g, b = leaves
        for layer in range(HANDLE_LAYERS):
            if poison is not None:
                poison(ws)
            if keep is None:
                y = cake_kcp.chunk_kda_det_kcp(
                    q, k, y, g, b, ctx, None, "light", "saved", ws, **kw
                )
                continue

            def fn(q_, k_, v_, g_, b_, h_, w_, _i=i, _layer=layer):
                out = cake_kcp.chunk_kda_det_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctxs[_i],
                    None,
                    "light",
                    "saved",
                    wss[_i],
                    recompute_handle=h_,
                    checkpoint_keep=keep,
                    **kw,
                )
                seen[_i][_layer].append(out.detach().clone())
                return out * w_

            h = cake_kcp.RecomputeHandleDetKcp()
            # ``out * w`` with ``w`` a leaf of ones changes no bit of ``out`` or of its gradient, but the product's
            # backward reads the recompute's ``out``: the recompute runs past the record above (torch stops a
            # recompute once the tensors it needs are back)
            w = torch.ones_like(y).requires_grad_(True)
            y = torch.utils.checkpoint.checkpoint(
                fn, q, k, y, g, b, h, w, use_reentrant=False
            )
            stored = {
                name: t
                for name, t in h.entries[cake_kcp.HANDLE_KEY].items()
                if isinstance(t, torch.Tensor)
            }
            handles[i].append(h)
            snaps[i].append({name: t.clone() for name, t in stored.items()})
            for name, t in stored.items():
                log[name] += t.numel()
                if _overlaps(t, _storages(ws)):
                    log["overlap"].append(f"pack {i} layer {layer} {name}")
        live[i] = (leaves, y)

    def backward(i):
        check(f"before the backward of pack {i}")
        if poison is not None:
            poison(wss[i])
        leaves, y = live.pop(i)
        y.backward(ops[i]["dout"])
        res[i]["out"] = y.detach()
        res[i]["grads"] = tuple(t.grad for t in leaves)

    if order == "interleaved":
        for i in range(n):
            forward(i)
        for i in range(n):
            backward(i)
    else:
        for i in range(n):
            forward(i)
            backward(i)
    check("after the last backward")
    if keep is not None:
        for i in range(n):
            for layer in range(HANDLE_LAYERS):
                outs = seen[i][layer]
                if len(outs) != 2 or not _same_bits(outs[0], outs[1]):
                    log["recompute_out"].append(
                        f"pack {i} layer {layer}: {len(outs)} outs"
                    )
    return res, log


def _handle_violations(ref, got, log, recs, n_calls, exchange) -> list[str]:
    bad = [
        f"pack {i} {key}"
        for i in range(len(ref))
        for key in ("out", "grads")
        if not _same_bits(got[i][key], ref[i][key])
    ]
    bad += [f"stored tensor changed: {c}" for c in log["changed"]]
    bad += [f"stored tensor in the workspace: {c}" for c in log["overlap"]]
    bad += [f"recompute out differs: {c}" for c in log["recompute_out"]]
    return bad + recs.phases(n_calls, exchange)


@pytest.mark.parametrize("order", HANDLE_ORDERS)
@pytest.mark.parametrize("keep", ("states", "transfer"))
@pytest.mark.parametrize("H", HEADS)
def test_gpu_recompute_handles_on_a_reserved_shared_workspace(
    device, H, keep, order
) -> None:
    """Four packs on ONE reserved workspace, two checkpointed KDA layers per pack, one recompute handle per pack and
    layer (``checkpoint_keep`` ``"states"`` / ``"transfer"``), in the interleaved (A fwd, B fwd, A bwd, B bwd) and the
    sequential order, the arena and the pools poisoned before every forward and backward: out and the five gradients
    of every pack are bitwise those of one workspace per pack without handles; each recompute binds its entry
    ("replay": no collective, no exchange launch) and gives bitwise its original ``out``; no stored tensor changes
    before its backward and none lies in the workspace; no pointer moves.  One rank: no fragment crosses a rank, so
    the stored tensors are empty here; the W = 4 rows at the end of this file store and check real ones."""
    ctxs, ops = _packs(device, H)
    ref, _ = _handle_pass(ctxs, ops, [_new_ws() for _ in ctxs])
    ws = _new_ws()
    ws.reserve(ctxs, H)
    ptrs = _pointers(ws)
    with _Records() as recs:
        got, log = _handle_pass(
            ctxs, ops, ws, keep=keep, order=order, poison=_poison_all
        )
    bad = _handle_violations(ref, got, log, recs, len(ctxs) * HANDLE_LAYERS, "a2a")
    assert not bad, bad
    assert _pointers(ws) == ptrs
    assert all(c.n_in == 0 for c in ctxs) and log["S_in"] == log["c1_recv"] == 0


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


def _rank_packs(rank: int, W: int, row, port: int):
    """This rank's device (the process group initialized for ``W > 1``), the row's packs prepared on it and seeded
    operands per pack."""
    packs, H, _w, layout, exchange = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD if W > 1 else None
    ctxs = [
        kp.build_context_det_kcp(_part(lengths, W, layout), group, rank, device)
        for lengths in packs
    ]
    ops = [
        _operands(device, int(c.T_r), H, 0x7815 + 97 * i + rank)
        for i, c in enumerate(ctxs)
    ]
    if exchange == "ring":
        assert all(c.ring_eligible for c in ctxs), (
            "a ring row holds a document over more than three ranks"
        )
    return device, ctxs, ops


def _run_dist_rank(rank: int, W: int, row, port: int, result_dir: str | None) -> dict:
    """One rank: the four packs on one workspace per pack (reference) and on one reserved shared workspace (the
    interleaved schedule, the arena NaN-filled before every forward), bitwise compared on this rank; the shared
    workspace's bytes against ``kda_det_workspace_bytes_kcp(ctxs)`` and the per-pack sum, pointer stability over two rounds.
    The verdicts are gathered to every rank."""
    packs, H, _w, layout, exchange = row
    device, ctxs, ops = _rank_packs(rank, W, row, port)
    kw = {"exchange": exchange}
    ring = exchange == "ring"
    report: dict = {"rank": rank, "row": _dist_row_id(row), "pass": True, "fail": []}
    per = [_new_ws() for _ in ctxs]
    ref = _schedule(ctxs, ops, per, **kw)
    report["per_pack_bytes"] = [w.bytes for w in per]
    report["P"] = [cake_kcp.WorkspaceDetKcp(local=False).tables(c, H).P for c in ctxs]
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
    report["pred_bytes"] = kp.kda_det_workspace_bytes_kcp(ctxs, H=H) + (
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
        report["fail"].append("bytes != kda_det_workspace_bytes_kcp(ctxs)")
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


def _spawn_dist_entry(
    rank: int, W: int, row, port: int, result_dir: str, kind: str = "shared"
) -> None:
    try:
        report = RANK_RUNNERS[kind](rank, W, row, port, result_dir)
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
    _launch(row, "shared")


def _launch(row, kind: str) -> None:
    """Run ``RANK_RUNNERS[kind]`` for ``row``: in this process as one Slurm task (a step with ``SLURM_NTASKS`` > 1
    runs the rows of its own world size and skips the others) or one spawned process per GPU (rows up to the
    visible GPU count); assert the gathered verdict."""
    _packs_, _H, W, _layout, _exchange = row
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"this Slurm step runs the W = {ntasks} rows")
        report = RANK_RUNNERS[kind](
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
            args=(W, row, _free_port(), result_dir, kind),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, RESULT_NAMES[kind](row))) as fh:
            report = json.load(fh)
    assert report["pass"], report


# (packs, H, W, layout, exchange): the W = 4 rows above with recompute handles -- a2a P > 1 (H = 4, both tiers),
# a2a P = 1 (H = 64, both tiers), ring (H = 4, "states": the ring receives no c1 tensor)
HANDLE_DIST_ROWS = [row for row in DIST_ROWS if row[2] == 4]


def _handle_row_id(row) -> str:
    return "handles-" + _dist_row_id(row)


def _run_handle_rank(rank: int, W: int, row, port: int, result_dir: str | None) -> dict:
    """One rank: the row's packs on one workspace per pack without handles (the reference), then on ONE reserved
    shared workspace with two checkpointed layers per pack and one recompute handle per pack and layer, for every
    tier (``"transfer"`` with a2a only) and both orders (:func:`_handle_pass`; the arena and the pools poisoned
    before every forward and backward): bitwise out and gradients, no stored ``S_in`` / c1 receive tensor changed
    before its backward or placed in the workspace, every recompute exchange-free and bitwise its original ``out``,
    no pointer moved.  The stored element counts are gathered: the row must store non-empty tensors on some rank (of
    each tier's members).  The verdicts are gathered to every rank."""
    _packs_, H, _w, _layout, exchange = row
    _device, ctxs, ops = _rank_packs(rank, W, row, port)
    kw = {"exchange": exchange}
    report: dict = {"rank": rank, "row": _handle_row_id(row), "pass": True, "fail": []}
    ref, _ = _handle_pass(ctxs, ops, [_new_ws() for _ in ctxs], **kw)
    gc.collect()
    ws = _new_ws()
    ws.reserve(ctxs, H, ring=exchange == "ring")
    ptrs = _pointers(ws)
    keeps = ("states",) if exchange == "ring" else ("states", "transfer")
    stored: dict = {}
    for keep in keeps:
        for order in HANDLE_ORDERS:
            with _Records() as recs:
                got, log = _handle_pass(
                    ctxs, ops, ws, keep=keep, order=order, poison=_poison_all, **kw
                )
            bad = _handle_violations(
                ref, got, log, recs, len(ctxs) * HANDLE_LAYERS, exchange
            )
            report["fail"] += [f"{keep} {order}: {b}" for b in bad]
            if _pointers(ws) != ptrs:
                report["fail"].append(f"{keep} {order}: pointers moved")
            stored[f"{keep}/{order}"] = {"S_in": log["S_in"], "c1_recv": log["c1_recv"]}
            del got, log
    report["stored"] = stored
    report["pass"] = not report["fail"]
    if W > 1:
        verdicts = [None] * W
        dist.all_gather_object(verdicts, {k: report[k] for k in report if k != "row"})
        report["ranks"] = verdicts
        for key in stored:
            members = ("S_in", "c1_recv") if key.startswith("transfer") else ("S_in",)
            for m in members:
                if not any(v["stored"][key][m] for v in verdicts):
                    report["fail"].append(f"{key}: no rank stored a non-empty {m}")
        report["pass"] = all(v["pass"] for v in verdicts) and not report["fail"]
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "shared-" + _handle_row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


RANK_RUNNERS = {"shared": _run_dist_rank, "handles": _run_handle_rank}
RESULT_NAMES = {
    "shared": lambda row: "shared-" + _dist_row_id(row) + ".json",
    "handles": lambda row: "shared-" + _handle_row_id(row) + ".json",
}


@pytest.mark.parametrize("row", HANDLE_DIST_ROWS, ids=_dist_row_id)
def test_multi_gpu_recompute_handles_on_a_shared_workspace(row) -> None:
    """The rows of :func:`_run_handle_rank`, launched like the shared-workspace rows (``spawn`` or one process per
    Slurm task)."""
    _launch(row, "handles")
