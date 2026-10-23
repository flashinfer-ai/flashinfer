"""FP64 fake backend of the context-parallel (KCP) route of the Cake packed KDA training backend.

It emulates ``W`` ranks in ONE CPU process on the real :mod:`cake_kcp_plan` tables: every rank's exchange buffers
are plain FP64 tensors laid out exactly as the device buffers (``KcpHeadPlan`` tile tables and chain offsets), the
collectives are dictionary ``all_to_all_single`` over the published split sizes, and the per-fragment compute is the
FP64 token recurrence of the step-0 reference (``reference_kcp.recur_interval`` /
``recur_interval_adjoint``).  The launch sequence follows FINAL_DESIGN §3.5 (forward: offset + transfer summaries ->
c1 -> owner chain prefix -> c2 -> local forward from ``S_in``) and §6.5 (backward: adjoint offsets ``d_f`` [+ re-sent
maps] -> b1 -> owner reverse chain -> b2 -> local adjoint from ``dS_out``) at fragment granularity (``P = 1``
semantics: fragment = segment).  Saved mode reuses the forward's ``S_in`` (and the owner's c1 receive tensor with
``save_transfer=True``); replay mode re-runs the forward exchange.  Everything is deterministic FP64, so saved and
replay are bitwise equal and both agree with the unpartitioned per-document FP64 reference to rounding (~1e-12).

The transfer map of a fragment is the final state of the recurrence started from the identity with ``v = 0``
(the recurrence is affine in the state; its linear part does not depend on ``v``), one FP64 recurrence per
crossing-out fragment and no cancellation; ``_transfer_matches_lane_p`` checks it against lane P's basis-vector
``transfer`` helper on a short interval.

Inputs come from the reference harness (``kcp_shapes`` rows, ``kcp_inputs`` operands) and lane P's
``reference_kcp`` (``$W/harness``, ``$W/lanes/P`` on ``PYTHONPATH``); they are imported lazily so that this module
imports without them.
"""

from __future__ import annotations

import dataclasses
import time

import torch

from tests.test_helpers.cake_package import cake_package

kp = cake_package("cake_kda_train", "cake_kcp_plan")

F64 = torch.float64
D = kp.HEAD_DIM
TILE, SUMMARY_TILE, TILE_B, TILE_M = kp.TILE, kp.SUMMARY_TILE, kp.TILE_B, kp.TILE_M
TILE_D = kp.TILE_D
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
OPERANDS = ("q", "k", "v", "g", "beta", "dout")


def _harness():
    import kcp_inputs  # noqa: PLC0415  (the reference harness, on PYTHONPATH in the lane environment)
    import kcp_shapes  # noqa: PLC0415

    return kcp_shapes, kcp_inputs


def _reference():
    import reference_kcp  # noqa: PLC0415  (lane P's FP64 helpers, on PYTHONPATH in the lane environment)

    return reference_kcp


def _log(progress, msg: str) -> None:
    if progress is not None:
        progress(msg)


# ----------------------------------------------------------------------------------------------------------------
# Emulated group and buffers


def all_to_all(
    sends: list[torch.Tensor],
    send_splits: list[tuple[int, ...]],
    recv_splits: list[tuple[int, ...]],
) -> list[torch.Tensor]:
    """Dictionary ``all_to_all_single`` over ``W`` ranks: ``recv[j]`` is the concatenation over ``i`` of the ``j``-th
    segment of ``sends[i]``.  Checks the split tables of every pair (``send_splits[i][j] == recv_splits[j][i]``) and the
    buffer sizes."""
    W = len(sends)
    for i in range(W):
        if sends[i].numel() != sum(send_splits[i]):
            raise AssertionError(
                f"rank {i}: send buffer {sends[i].numel()} != sum of its splits {sum(send_splits[i])}"
            )
        for j in range(W):
            if send_splits[i][j] != recv_splits[j][i]:
                raise AssertionError(
                    f"split mismatch {i}->{j}: sender says {send_splits[i][j]}, receiver says {recv_splits[j][i]}"
                )
    offsets = [[0] for _ in range(W)]
    for i in range(W):
        for j in range(W):
            offsets[i].append(offsets[i][-1] + send_splits[i][j])
    out = []
    for j in range(W):
        parts = [sends[i][offsets[i][j] : offsets[i][j + 1]] for i in range(W)]
        buf = torch.cat(parts) if parts else sends[0].new_zeros(0)
        if buf.numel() != sum(recv_splits[j]):
            raise AssertionError(
                f"rank {j}: receive buffer {buf.numel()} != sum of its splits {sum(recv_splits[j])}"
            )
        out.append(buf)
    return out


def tile(buf: torch.Tensor, offset: int) -> torch.Tensor:
    """FP64 ``[128, 128]`` view of one state tile at element ``offset`` of a flat buffer."""
    return buf[offset : offset + TILE].view(D, D)


@dataclasses.dataclass
class FakeRanks:
    """The emulated group: the partition, one context per rank and every rank's FP64 operands (local order)."""

    row: object
    H: int
    W: int
    layout: str
    part: object
    ctxs: list
    ops: list[dict[str, torch.Tensor]]
    cu: tuple[int, ...]  # cu_seqlens of the (sub-)pack the partition cuts
    docs: list[int]  # global document indices of the (sub-)pack's documents

    def frag_ops(self, rank: int, slot: int) -> dict[str, torch.Tensor]:
        cu = self.ctxs[rank].cu_seqlens_local_cpu
        a, b = cu[slot], cu[slot + 1]
        return {name: t[a:b] for name, t in self.ops[rank].items()}

    def rows(self, rank: int, slot: int) -> slice:
        cu = self.ctxs[rank].cu_seqlens_local_cpu
        return slice(cu[slot], cu[slot + 1])


def build_ranks(
    row,
    W: int,
    layout: str,
    *,
    docs=None,
    fragments=None,
    cut_points="harness",
    sm_count: int = 152,
    conv_kernel_size: int = 4,
    progress=None,
) -> FakeRanks:
    """Partition ``row`` (or its documents ``docs``) over ``W`` ranks and generate every rank's FP64 operands.

    ``layout``: ``"C"`` / ``"Z"`` / ``"D"`` (harness letters) or a :mod:`cake_kcp_plan` layout name; ``"R"`` takes the
    harness ragged table of ``row`` (``fragments`` overrides).  Layout D uses the harness jitter (``kcp_shapes.doc_cuts_D``)
    as the caller's cut points unless ``cut_points`` is given explicitly (``None`` = the module default)."""
    kcp_shapes, kcp_inputs = _harness()
    H = int(row.h)
    all_docs = list(range(len(row.lengths)))
    docs = all_docs if docs is None else [int(d) for d in docs]
    lengths = [row.lengths[d] for d in docs]
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    cu = tuple(cu)
    names = {"C": "contiguous", "Z": "zigzag", "D": "zigzag_per_document"}
    if fragments is not None or layout == "R":
        if fragments is None:
            if docs != all_docs:
                raise ValueError(
                    "explicit ragged tables are defined for the whole pack"
                )
            harness_frags = kcp_shapes.partition(row, W, "R")
            fragments = [
                [
                    (f.doc, f.off, f.length)
                    for f in sorted(
                        (f for f in harness_frags if f.rank == r), key=lambda f: f.slot
                    )
                ]
                for r in range(W)
            ]
        part = kp.kcp_partition(
            torch.tensor(cu, dtype=torch.int64), W, fragments=fragments
        )
    else:
        name = names.get(layout, layout)
        cuts = None
        if name == "zigzag_per_document":
            if isinstance(cut_points, str) and cut_points == "harness":
                sub = (
                    row
                    if docs == all_docs
                    else dataclasses.replace(row, lengths=tuple(lengths))
                )
                # the harness seeds its jitter by (row name, W, document index of the pack it cuts)
                cuts = [kcp_shapes.doc_cuts_D(sub, W, i) for i in range(len(docs))]
            else:
                cuts = cut_points
        part = kp.kcp_partition(
            torch.tensor(cu, dtype=torch.int64), W, layout=name, cut_points=cuts
        )
    ctxs = [
        kp.build_kcp_context(part, None, r, "cpu", conv_kernel_size, sm_count=sm_count)
        for r in range(W)
    ]
    params = kcp_inputs.head_params(row, H)
    ops = []
    for r in range(W):
        parts: dict[str, list[torch.Tensor]] = {name: [] for name in OPERANDS}
        for f in part.fragments[r]:
            local = kcp_inputs.doc_range(row, H, docs[f.doc], f.off, f.length, params)
            for name in OPERANDS:
                parts[name].append(local[name].to(F64))
        ops.append(
            {
                name: (
                    torch.cat(ts, 0)
                    if ts
                    else kcp_inputs.doc_range(row, H, docs[0], 0, 0, params)[name].to(
                        F64
                    )
                )
                for name, ts in parts.items()
            }
        )
    _log(
        progress,
        f"{row.name} W={W} {layout}: {sum(len(frs) for frs in part.fragments)} fragments, T_r={[c.T_r for c in ctxs]}",
    )
    return FakeRanks(
        row=row,
        H=H,
        W=W,
        layout=layout,
        part=part,
        ctxs=ctxs,
        ops=ops,
        cu=cu,
        docs=docs,
    )


# ----------------------------------------------------------------------------------------------------------------
# Per-fragment compute (lane P's FP64 helpers)


def fragment_offset_and_transfer(
    ops: dict[str, torch.Tensor], H: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(B_f, M_f)`` of one fragment: ``B_f`` = the final state from zero (the offset), ``M_f`` = the final state from
    the identity with ``v = 0`` (the linear part); both ``[H, 128, 128]`` FP64 with ``S_out = M_f @ S_in + B_f`` per head."""
    rk = _reference()
    zero = torch.zeros(H, D, D, dtype=F64)
    eye = torch.eye(D, dtype=F64).expand(H, D, D).contiguous()
    with torch.no_grad():
        _o, B = rk.recur_interval(
            zero, ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"]
        )
        _o, M = rk.recur_interval(
            eye, ops["q"], ops["k"], torch.zeros_like(ops["v"]), ops["g"], ops["beta"]
        )
    return B, M


def _transfer_matches_lane_p(ops: dict[str, torch.Tensor], H: int) -> float:
    """max |M (identity, v = 0) - M (lane P's basis-vector transfer)| on one short interval (a unit check)."""
    rk = _reference()
    _B, M = fragment_offset_and_transfer(ops, H)
    M_ref, _B_ref = rk.transfer(
        torch.zeros(H, D, D, dtype=F64),
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
    )
    return float((M - M_ref).abs().max())


# ----------------------------------------------------------------------------------------------------------------
# Forward (§3.5) and backward (§6.5) at fragment granularity


@dataclasses.dataclass
class FakeSaved:
    s_in: list[torch.Tensor]  # per rank the c2 receive buffer (S_in, head-group major)
    c1_recv: (
        list[torch.Tensor] | None
    )  # per owner the c1 receive tensor (only save_transfer)
    maps: dict  # (rank, slot) -> M_f of crossing-out fragments (the forward's K3 result; reused as the "recompute")


def fake_forward(
    ranks: FakeRanks,
    *,
    save_transfer: bool = False,
    want_out: bool = True,
    log=None,
    progress=None,
):
    """Returns ``(out per rank or None, FakeSaved)``.  ``log`` collects the collective issue order."""
    rk = _reference()
    H = ranks.H
    plans = [c.heads(H) for c in ranks.ctxs]
    maps: dict = {}
    c1_send = [torch.zeros(hp.numel["c1_send"], dtype=F64) for hp in plans]
    t0 = time.time()
    for r, ctx in enumerate(ranks.ctxs):
        hp = plans[r]
        for f in ctx.fragments:
            if not f.crossing_out:
                continue
            B, M = fragment_offset_and_transfer(ranks.frag_ops(r, f.slot), H)
            maps[(r, f.slot)] = M
            s = ctx.out_slot[f.slot]
            for h in range(H):
                base = hp.out_tile[s][h] * SUMMARY_TILE
                c1_send[r][base + TILE_B * TILE : base + TILE_B * TILE + TILE] = B[
                    h
                ].reshape(-1)
                c1_send[r][base + TILE_M * TILE : base + TILE_M * TILE + TILE] = (
                    M[h].transpose(0, 1).reshape(-1)
                )
    _log(
        progress,
        f"  summaries ({len(maps)} crossing-out fragments) {time.time() - t0:.1f}s",
    )
    if log is not None:
        log.append("c1")
    c1_recv = all_to_all(
        c1_send,
        [hp.splits["c1_send"] for hp in plans],
        [hp.splits["c1_recv"] for hp in plans],
    )
    c2_send = [torch.zeros(hp.numel["c2_send"], dtype=F64) for hp in plans]
    for r, hp in enumerate(plans):
        for c in range(len(hp.fwd_chain_doc)):
            for hg in range(hp.Hg_r):
                X = torch.zeros(D, D, dtype=F64)
                for e in range(hp.fwd_chain_start[c], hp.fwd_chain_start[c + 1]):
                    Mt = tile(
                        c1_recv[r], hp.fwd_m_off[e] + hg * hp.hg_stride["fwd_m_off"]
                    )
                    B = tile(
                        c1_recv[r], hp.fwd_b_off[e] + hg * hp.hg_stride["fwd_b_off"]
                    )
                    X = Mt.transpose(0, 1) @ X + B
                    o = hp.fwd_out_off[e] + hg * hp.hg_stride["fwd_out_off"]
                    c2_send[r][o : o + TILE] = X.reshape(-1)
    if log is not None:
        log.append("c2")
    s_in = all_to_all(
        c2_send,
        [hp.splits["c2_send"] for hp in plans],
        [hp.splits["c2_recv"] for hp in plans],
    )
    out = None
    if want_out:
        out = []
        t0 = time.time()
        for r, ctx in enumerate(ranks.ctxs):
            o_r = torch.zeros(ctx.T_r, H, D, dtype=F64)
            for f in ctx.fragments:
                if f.length == 0:
                    continue
                S0 = initial_state(ranks, r, f, s_in)
                ops = ranks.frag_ops(r, f.slot)
                with torch.no_grad():
                    o, _S_T = rk.recur_interval(
                        S0, ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"]
                    )
                o_r[ranks.rows(r, f.slot)] = o
            out.append(o_r)
        _log(progress, f"  local forward {time.time() - t0:.1f}s")
    return out, FakeSaved(
        s_in=s_in, c1_recv=c1_recv if save_transfer else None, maps=maps
    )


def initial_state(
    ranks: FakeRanks, rank: int, f, s_in: list[torch.Tensor]
) -> torch.Tensor:
    """``S_in`` of fragment ``f`` on ``rank`` from the c2 receive buffer (zero for a document start)."""
    H = ranks.H
    S0 = torch.zeros(H, D, D, dtype=F64)
    if f.crossing_in:
        hp = ranks.ctxs[rank].heads(H)
        s = ranks.ctxs[rank].in_slot[f.slot]
        for h in range(H):
            S0[h] = tile(s_in[rank], hp.h0_tile[s][h] * TILE)
    return S0


def fake_backward(
    ranks: FakeRanks,
    saved: FakeSaved,
    *,
    mode: str = "saved",
    save_transfer: bool = False,
    log=None,
    progress=None,
):
    """Returns ``(grads per rank, checks)``; ``grads[r]`` holds ``dq, dk, dv, dg, dbeta`` of the rank's tokens.
    ``mode="saved"`` uses the forward's ``S_in`` (and the owner's c1 tensor when ``save_transfer``), ``"replay"``
    re-runs the forward exchange first.  ``checks`` reports the max difference between the owner-chain ``dS_in`` of
    every crossing-in fragment and the local adjoint's ``dS0`` (an internal consistency figure)."""
    rk = _reference()
    W, H = ranks.W, ranks.H
    plans = [c.heads(H) for c in ranks.ctxs]
    if mode == "replay":
        _out, replayed = fake_forward(
            ranks, save_transfer=True, want_out=False, log=log, progress=progress
        )
        for r in range(W):
            if not torch.equal(replayed.s_in[r], saved.s_in[r]):
                raise AssertionError(
                    f"replay: rank {r}'s recomputed S_in differs from the saved one"
                )
        s_in, c1_recv, maps = replayed.s_in, replayed.c1_recv, replayed.maps
        with_maps = False
    elif mode == "saved":
        s_in, maps = saved.s_in, saved.maps
        with_maps = not save_transfer
        c1_recv = saved.c1_recv
        if save_transfer and c1_recv is None:
            raise ValueError(
                "saved mode with save_transfer needs the forward's c1 receive tensors"
            )
    else:
        raise ValueError(f"mode must be 'saved' or 'replay', got {mode!r}")
    key = "dm" if with_maps else "d"
    b1_send = [torch.zeros(hp.numel[f"b1_send_{key}"], dtype=F64) for hp in plans]
    t0 = time.time()
    for r, ctx in enumerate(ranks.ctxs):
        hp = plans[r]
        for f in ctx.fragments:
            if f.crossing_in:
                ops = ranks.frag_ops(r, f.slot)
                S0 = initial_state(ranks, r, f, s_in)
                adj = rk.recur_interval_adjoint(
                    S0,
                    ops["q"],
                    ops["k"],
                    ops["v"],
                    ops["g"],
                    ops["beta"],
                    ops["dout"],
                    dS_T=None,
                )
                d_f = adj["dS0"]
                s = ctx.in_slot[f.slot]
                table = hp.b1_d_tile_dm if with_maps else hp.b1_d_tile_d
                for h in range(H):
                    b1_send[r][table[s][h] * TILE : (table[s][h] + 1) * TILE] = d_f[
                        h
                    ].reshape(-1)
            # the M half of the b1 pair [d_f | M_f^T] (P1-R3) exists only for crossing-in fragments: a document's
            # first fragment has no pair (its map travels in c1 only); a document's last fragment leaves its half unused
            if with_maps and f.crossing_in and f.crossing_out:
                M = maps[
                    (r, f.slot)
                ]  # the K3 recompute: same inputs, same function -> the forward's bits
                s = ctx.in_slot[
                    f.slot
                ]  # the M half of the fragment's b1 pair [d_f | M_f^T] (P1-R3)
                for h in range(H):
                    t = hp.b1_m_tile_dm[s][h]
                    assert t == hp.b1_d_tile_dm[s][h] + 1
                    b1_send[r][t * TILE : (t + 1) * TILE] = (
                        M[h].transpose(0, 1).reshape(-1)
                    )
    _log(progress, f"  adjoint offsets {time.time() - t0:.1f}s")
    if log is not None:
        log.append("b1")
    b1_recv = all_to_all(
        b1_send,
        [hp.splits[f"b1_send_{key}"] for hp in plans],
        [hp.splits[f"b1_recv_{key}"] for hp in plans],
    )
    b2_send = [torch.zeros(hp.numel["b2_send"], dtype=F64) for hp in plans]
    for r, hp in enumerate(plans):
        d_name = "adj_d_off_dm" if with_maps else "adj_d_off_d"
        d_off, d_stride = getattr(hp, d_name), hp.hg_stride[d_name]
        for c in range(len(hp.adj_chain_doc)):
            for hg in range(hp.Hg_r):
                X = torch.zeros(D, D, dtype=F64)
                for e in range(hp.adj_chain_start[c], hp.adj_chain_start[c + 1]):
                    d = tile(b1_recv[r], d_off[e] + hg * d_stride)
                    if with_maps:
                        m_off, src, stride = (
                            hp.adj_m_off_b1[e],
                            b1_recv[r],
                            hp.hg_stride["adj_m_off_b1"],
                        )
                    else:
                        m_off, src, stride = (
                            hp.adj_m_off_c1[e],
                            c1_recv[r],
                            hp.hg_stride["adj_m_off_c1"],
                        )
                    if m_off >= 0:
                        X = (
                            tile(src, m_off + hg * stride) @ X + d
                        )  # the stored tile is M^T: M^T X + d
                    else:
                        if e != hp.adj_chain_start[c]:
                            raise AssertionError(
                                "a map-less element must be the document's last fragment (first of the reverse chain)"
                            )
                        X = d.clone()
                    o = hp.adj_out_off[e] + hg * hp.hg_stride["adj_out_off"]
                    b2_send[r][o : o + TILE] = X.reshape(-1)
    if log is not None:
        log.append("b2")
    ds_out = all_to_all(
        b2_send,
        [hp.splits["b2_send"] for hp in plans],
        [hp.splits["b2_recv"] for hp in plans],
    )
    grads = []
    chain_vs_local = 0.0
    t0 = time.time()
    for r, ctx in enumerate(ranks.ctxs):
        hp = plans[r]
        g_r = {
            "dq": torch.zeros(ctx.T_r, H, D, dtype=F64),
            "dk": torch.zeros(ctx.T_r, H, D, dtype=F64),
            "dv": torch.zeros(ctx.T_r, H, D, dtype=F64),
            "dg": torch.zeros(ctx.T_r, H, D, dtype=F64),
            "dbeta": torch.zeros(ctx.T_r, H, dtype=F64),
        }
        for f in ctx.fragments:
            if f.length == 0:
                continue
            ops = ranks.frag_ops(r, f.slot)
            S0 = initial_state(ranks, r, f, s_in)
            dS_T = None
            if f.crossing_out:
                dS_T = torch.zeros(H, D, D, dtype=F64)
                s = ctx.out_slot[f.slot]
                for h in range(H):
                    dS_T[h] = tile(ds_out[r], hp.dht_tile[s][h] * TILE)
            adj = rk.recur_interval_adjoint(
                S0,
                ops["q"],
                ops["k"],
                ops["v"],
                ops["g"],
                ops["beta"],
                ops["dout"],
                dS_T=dS_T,
            )
            rows = ranks.rows(r, f.slot)
            for name in GRAD_NAMES:
                g_r[name][rows] = adj[name]
            if f.crossing_in:
                # the owner published dS_in(f) as dS_out of f's predecessor: compare with the local adjoint's dS0
                pr, ps = f.prev
                pctx, php = ranks.ctxs[pr], plans[pr]
                po = pctx.out_slot[ps]
                chain = torch.stack(
                    [tile(ds_out[pr], php.dht_tile[po][h] * TILE) for h in range(H)]
                )
                chain_vs_local = max(
                    chain_vs_local, float((chain - adj["dS0"]).abs().max())
                )
        grads.append(g_r)
    _log(
        progress,
        f"  local adjoint {time.time() - t0:.1f}s; chain dS_in vs local dS0 max {chain_vs_local:.3e}",
    )
    return grads, {"chain_vs_local_dS_in": chain_vs_local}


# ----------------------------------------------------------------------------------------------------------------
# Global order, reference and metrics


def gather_global(ranks: FakeRanks, per_rank: list[torch.Tensor]) -> torch.Tensor:
    """Per-rank ``[T_r, ...]`` tensors -> the (sub-)pack's global token order ``[T, ...]`` (the partition inverse)."""
    T = ranks.cu[-1]
    any_t = per_rank[0]
    out = torch.zeros(T, *any_t.shape[1:], dtype=any_t.dtype)
    for r, t in enumerate(per_rank):
        for f in ranks.part.fragments[r]:
            if f.length:
                g0 = ranks.cu[f.doc] + f.off
                out[g0 : g0 + f.length] = t[ranks.rows(r, f.slot)]
    return out


def metrics(cand: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """``rel_rms`` = ||cand - ref|| / ||ref||, ``max_abs`` = max |cand - ref|, ``max_abs_rel`` = max_abs / max |ref|."""
    diff = (cand.to(F64) - ref.to(F64)).reshape(-1)
    ref_f = ref.to(F64).reshape(-1)
    ref_norm = float(ref_f.norm())
    ref_max = float(ref_f.abs().max()) if ref_f.numel() else 0.0
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    return {
        "rel_rms": float(diff.norm()) / ref_norm if ref_norm > 0 else max_abs,
        "max_abs": max_abs,
        "max_abs_rel": max_abs / ref_max if ref_max > 0 else max_abs,
        "ref_max": ref_max,
    }


def load_or_compute_reference(
    row, ref_dir: str | None, progress=None
) -> dict[str, torch.Tensor]:
    """Lane P's cached FP64 reference of the whole pack (``<ref_dir>/<row>.pt`` when complete) or a fresh
    ``reference_pack`` on the CPU (the same helpers, deterministic).  Returns ``out, dq, dk, dv, dg, dbeta`` FP64 CPU."""
    rk = _reference()
    if ref_dir:
        ref = rk.load_reference(ref_dir, row.name)
        if ref is not None:
            meta = ref.pop("_meta")
            _log(
                progress,
                f"{row.name}: cached reference {meta.get('path')} ({meta.get('bytes')} B, sha256 {meta.get('sha256')})",
            )
            return {k: v.to(F64) for k, v in ref.items()}
    t0 = time.time()
    out, grads = rk.reference_pack(row, torch.device("cpu"), progress=progress)
    _log(
        progress,
        f"{row.name}: reference computed on the CPU in {time.time() - t0:.1f}s",
    )
    return {"out": out, **grads}


def run_fake(
    row,
    W: int,
    layout: str,
    *,
    docs=None,
    fragments=None,
    save_transfer: bool = False,
    sm_count: int = 152,
    progress=None,
) -> dict:
    """Forward + both backward modes of ``row`` at ``W`` ranks / ``layout``.  Returns a dict with the global-order
    FP64 tensors of the saved-mode run (``out, dq, dk, dv, dg, dbeta``), the replay-mode gradients, the collective
    logs, the consistency checks and the ``FakeRanks``."""
    ranks = build_ranks(
        row,
        W,
        layout,
        docs=docs,
        fragments=fragments,
        sm_count=sm_count,
        progress=progress,
    )
    log_fwd: list[str] = []
    out, saved = fake_forward(
        ranks, save_transfer=save_transfer, log=log_fwd, progress=progress
    )
    log_saved: list[str] = []
    grads_saved, checks_saved = fake_backward(
        ranks,
        saved,
        mode="saved",
        save_transfer=save_transfer,
        log=log_saved,
        progress=progress,
    )
    log_replay: list[str] = []
    grads_replay, checks_replay = fake_backward(
        ranks,
        saved,
        mode="replay",
        save_transfer=save_transfer,
        log=log_replay,
        progress=progress,
    )
    result = {"out": gather_global(ranks, out)}
    for name in GRAD_NAMES:
        result[name] = gather_global(ranks, [g[name] for g in grads_saved])
    replay = {
        name: gather_global(ranks, [g[name] for g in grads_replay])
        for name in GRAD_NAMES
    }
    return {
        "ranks": ranks,
        "saved": result,
        "replay": replay,
        "logs": {
            "forward": log_fwd,
            "backward_saved": log_saved,
            "backward_replay": log_replay,
        },
        "checks": {"saved": checks_saved, "replay": checks_replay},
        "save_transfer": save_transfer,
    }


def compare_to_reference(
    result: dict, ref: dict[str, torch.Tensor], cu_global: list[int], docs: list[int]
) -> dict[str, dict]:
    """Per-tensor metrics of the saved-mode result against the reference restricted to ``docs`` (global pack order)."""
    sel = (
        torch.cat([torch.arange(cu_global[d], cu_global[d + 1]) for d in docs])
        if docs
        else torch.arange(0)
    )
    out = {}
    for name in ("out",) + GRAD_NAMES:
        out[name] = metrics(result["saved"][name], ref[name][sel])
    return out


def replay_equals_saved(result: dict) -> bool:
    return all(
        torch.equal(result["saved"][name], result["replay"][name])
        for name in GRAD_NAMES
    )


__all__ = [
    "FakeRanks",
    "FakeSaved",
    "all_to_all",
    "build_ranks",
    "compare_to_reference",
    "fake_backward",
    "fake_forward",
    "fragment_offset_and_transfer",
    "gather_global",
    "load_or_compute_reference",
    "metrics",
    "replay_equals_saved",
    "run_fake",
]
