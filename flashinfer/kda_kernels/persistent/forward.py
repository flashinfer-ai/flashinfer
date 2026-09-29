# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# Adapted from humanfia/kda-for-kda-release; see licenses/LICENSE.kda-for-kda.

"""Persistent KDA scheduling and compilation, with exact FP32 state handoffs."""

import heapq
import hashlib
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from . import kernel as _k
from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

D = 128
C32 = 32
LOG2E = 1.4426950408889634

# Device compilation controls and caches.
_TPROBE = 0
_COMPILED: dict = {}
_DUMMY_EXP: dict = {}
_DUMMY_MID: dict = {}
_DUMMY_TPR: dict = {}
LAST_TPROBE: list = [None]

_SMS: dict = {}


def _sm_count(dev: torch.device) -> int:
    n = _SMS.get(dev.index)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _SMS[dev.index] = n
    return n


def _get_compiled(
    H: int,
    dev: torch.device,
    gate2: int = 0,
    mid_bf16: bool = False,
    has_split: bool = False,
    do_export: bool = False,
    have_state: bool = True,
    beta_tma: bool = False,
    bf16_dv: bool = False,
    beta_bf16: bool = False,
    qk_rowpair: bool = False,
    rcp_decor: bool = True,
    rf_hoist: bool = True,
    restore_tail: bool = False,
    gram_w9: int = 0,
    reg_mode: int = 0,
    store_final_state: bool = False,
    full_chunks: bool = False,
    lower_bound_m5: bool = False,
    approximate_split: bool = False,
    prep_peel: bool = False,
    gate_roll: bool = False,
    cluster_size: int = 1,
    partial_tma: bool = True,
):
    key = (
        dev.index,
        torch.cuda.get_device_capability(dev),
        H,
        gate2,
        mid_bf16,
        store_final_state,
        has_split,
        do_export,
        have_state,
        beta_tma,
        bf16_dv,
        beta_bf16,
        qk_rowpair,
        rcp_decor,
        rf_hoist,
        restore_tail,
        gram_w9,
        reg_mode,
        full_chunks,
        lower_bound_m5,
        approximate_split,
        prep_peel,
        cluster_size,
        gate_roll,
        partial_tma,
    )
    entry = _COMPILED.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm the persistent KDA kernel before CUDA graph capture"
            )
        N = 2
        T = 8 * C32
        bf = torch.bfloat16
        mk = lambda t, a=16: from_dlpack(t, assumed_align=a).mark_compact_shape_dynamic(
            mode=0
        )
        q = torch.empty(T, H, D, dtype=bf, device=dev)
        k = torch.empty(T, H, D, dtype=bf, device=dev)
        v = torch.empty(T, H, D, dtype=bf, device=dev)
        g = torch.empty(T, H, D, dtype=bf, device=dev)
        beta = torch.empty(T, H, dtype=bf, device=dev)
        alog = torch.empty(H, dtype=torch.float32, device=dev)
        dtb = torch.empty(H, D, dtype=torch.float32, device=dev)
        st = torch.empty(N * H, D, D, dtype=torch.float32, device=dev)
        ns = torch.empty(N * H, D, D, dtype=torch.float32, device=dev)
        cu_t = torch.zeros(N + 1, dtype=torch.int32, device=dev)
        cu_t[1] = T // 2
        cu_t[2] = T
        G = 2 * H
        soff = torch.arange(G + 1, dtype=torch.int32, device=dev)
        schain = torch.arange(G, dtype=torch.int32, device=dev)
        spt0 = torch.zeros(G, dtype=torch.int32, device=dev)
        sptn = torch.full((G,), T // 2, dtype=torch.int32, device=dev)
        ssrc = torch.full((G,), -1, dtype=torch.int32, device=dev)
        sdst = torch.full((G,), -1, dtype=torch.int32, device=dev)
        mid = torch.empty(
            2, D, D, dtype=torch.bfloat16 if mid_bf16 else torch.float32, device=dev
        )
        mfl = torch.zeros(2, dtype=torch.int32, device=dev)
        tpr = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        out = torch.empty(T, H, D, dtype=bf, device=dev)
        nc2 = (T // 2 + C32 - 1) // C32
        exp_ws = torch.empty(H * nc2 * 64, 1, 256, dtype=bf, device=dev)
        expt = torch.empty(H * nc2, 160, dtype=torch.float32, device=dev)
        stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)

        def compile_kernel():
            return cute.compile(
                _k._launch_pkd,
                mk(q),
                mk(k),
                mk(v),
                mk(g),
                mk(beta, 4),
                from_dlpack(alog, assumed_align=16),
                from_dlpack(dtb, assumed_align=16),
                mk(st),
                mk(ns),
                mk(out),
                mk(cu_t, 4),
                mk(soff, 4),
                mk(schain, 4),
                mk(spt0, 4),
                mk(sptn, 4),
                mk(ssrc, 4),
                mk(sdst, 4),
                mk(mid),
                mk(mfl, 4),
                mk(exp_ws),
                mk(expt),
                mk(tpr, 8),
                cutlass.Int64(0),
                cutlass.Int32(1),
                cutlass.Int32(G),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(nc2),
                cutlass.Int32(1),
                cutlass.Float32(1.0),
                cutlass.Float32(-1.0),
                H_=H,
                stream=stream,
                TPROBE_=_TPROBE,
                GATE2_=gate2,
                FINAL_=1 if store_final_state else 0,
                SPLIT_=1 if has_split and not approximate_split else 0,
                WARM_SPLIT_=1 if approximate_split else 0,
                EXPORT_=1 if do_export else 0,
                STATE_=1 if have_state else 0,
                BETA_TMA_=1 if beta_tma else 0,
                BF16_DV_=1 if bf16_dv else 0,
                BETA_BF16_=1 if beta_bf16 else 0,
                QK_ROWPAIR_=int(qk_rowpair),
                RCP_DECOR_=1 if rcp_decor else 0,
                RF_HOIST_=1 if rf_hoist else 0,
                RESTORE_TAIL_=1 if restore_tail else 0,
                GRAM_W9_=int(gram_w9),
                REG_MODE_=int(reg_mode),
                FULL_CHUNKS_=1 if full_chunks else 0,
                LOWER_BOUND_M5_=1 if lower_bound_m5 else 0,
                CLUSTER_=int(cluster_size),
                PREP_PEEL_=1 if prep_peel else 0,
                PARTIAL_TMA_=1 if partial_tma else 0,
                GATE_ROLL_=1 if gate_roll else 0,
                options="--enable-tvm-ffi --opt-level 3",
            )

        tfn = build_and_load_cute_dsl_kernel(
            "kda_persistent",
            "kda_" + hashlib.sha256(repr(key).encode()).hexdigest()[:24],
            compile_kernel,
            extra_key_files=[str(Path(__file__)), str(Path(_k.__file__))],
        )
        _COMPILED[key] = tfn
        entry = tfn
    return entry


def _dummy_exp(dev: torch.device):
    key = dev.index
    tensors = _DUMMY_EXP.get(key)
    if tensors is None:
        tensors = (
            torch.empty(64, 1, 256, dtype=torch.bfloat16, device=dev),
            torch.empty(1, 160, dtype=torch.float32, device=dev),
        )
        _DUMMY_EXP[key] = tensors
    return tensors


def _dummy_mid(dev: torch.device):
    key = dev.index
    tensors = _DUMMY_MID.get(key)
    if tensors is None:
        tensors = (
            torch.empty(1, D, D, dtype=torch.float32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
        )
        _DUMMY_MID[key] = tensors
    return tensors


def _tprobe_buf(G: int, dev: torch.device):
    if _TPROBE:
        tensor = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        LAST_TPROBE[0] = tensor
        return tensor
    key = dev.index
    tensor = _DUMMY_TPR.get(key)
    if tensor is None:
        tensor = torch.zeros(64, dtype=torch.int64, device=dev)
        _DUMMY_TPR[key] = tensor
    return tensor


@torch.no_grad()
def _launch_forward(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    sched=None,
    export=None,
    gate2=0,
    has_split=False,
    beta_tma=False,
    bf16_dv=True,
    beta_bf16=True,
    qk_rowpair=False,
    rcp_decor=True,
    rf_hoist=True,
    restore_tail=False,
    gram_w9=0,
    reg_mode=0,
    full_chunks=False,
    lower_bound_m5=False,
    approximate_split=False,
    prep_peel=False,
    cluster_size=1,
    gate_roll=False,
    partial_tma=True,
):
    dev = q.device
    B, T, H, K = q.shape
    assert K == D
    qf = q.view(T * B, H, D)
    kf = k.view(T * B, H, D)
    vf = v.view(T * B, H, D)
    gf = g.view(T * B, H, D)
    bf_ = beta.view(T * B, H)
    of = out.view(T * B, H, D)

    if cu_seqlens is None:
        cu_t = torch.tensor([0, B * T], dtype=torch.int32, device=dev)
        nseqs = 1
    else:
        cu_t = cu_seqlens.to(torch.int32)
        nseqs = int(cu_seqlens.numel()) - 1

    store_final_state = final_state is not None
    if final_state is None:
        final_state = torch.empty(nseqs, H, D, D, dtype=torch.float32, device=dev)
    have_state = 1 if initial_state is not None else 0
    st = initial_state if initial_state is not None else final_state
    if st.dtype != torch.float32:
        st = st.float()

    if sched is None:
        chains = nseqs * H
        soff = torch.arange(chains + 1, dtype=torch.int32, device=dev)
        schain = torch.arange(chains, dtype=torch.int32, device=dev)
        spt0 = torch.zeros(chains, dtype=torch.int32, device=dev)
        sptn = cu_t.diff().to(torch.int32).repeat_interleave(H)
        ssrc = torch.full((chains,), -1, dtype=torch.int32, device=dev)
        sdst = ssrc
        mid, mfl = _dummy_mid(dev)
        fepoch = 1
        G = chains
    else:
        (soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, fepoch) = sched
    if export is None:
        exp_ws, expt = _dummy_exp(dev)
        do_export, export_seq, nc2 = 0, 0, 1
    else:
        exp_ws, expt, export_seq, nc2 = export
        do_export = 1

    tfn = _get_compiled(
        H,
        dev,
        gate2,
        mid.dtype == torch.bfloat16,
        has_split=has_split,
        do_export=bool(do_export),
        have_state=bool(have_state),
        beta_tma=beta_tma,
        bf16_dv=bf16_dv,
        beta_bf16=beta_bf16,
        qk_rowpair=qk_rowpair,
        rcp_decor=rcp_decor,
        rf_hoist=rf_hoist,
        restore_tail=restore_tail,
        gram_w9=gram_w9,
        reg_mode=reg_mode,
        store_final_state=store_final_state,
        full_chunks=full_chunks,
        lower_bound_m5=lower_bound_m5,
        approximate_split=approximate_split,
        prep_peel=prep_peel,
        cluster_size=cluster_size,
        gate_roll=gate_roll,
        partial_tma=partial_tma,
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(dev).cuda_stream)
    tfn(
        qf,
        kf,
        vf,
        gf,
        bf_,
        A_log,
        dt_bias.view(H, D),
        st.view(nseqs * H, D, D),
        final_state.view(nseqs * H, D, D),
        of,
        cu_t,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        mid,
        mfl,
        exp_ws,
        expt,
        _tprobe_buf(G, dev),
        0,
        have_state,
        G,
        do_export,
        export_seq,
        nc2,
        fepoch,
        float(scale),
        float(lower_bound) * LOG2E,
        stream,
    )
    return out, final_state


C_TOK = 32  # chunk tokens; split points are chunk-aligned
# Modelled per-piece cost (state seed/export and pipeline refill), in chunks;
# measured B300 chain starts cost about three chunks.
PIECE_OVERHEAD = 3
# Smallest producer or consumer piece, in chunks.
MIN_PIECE = 2
# Split only for this modelled gain: the handoff-capable kernel variant costs
# up to 3% on H96 routes.
MIN_SPLIT_GAIN = 0.05


def _lpt(costs, nslots):
    """Longest-processing-time packing of whole chains; returns (slots, loads)."""
    heap = [(0, slot) for slot in range(nslots)]
    heapq.heapify(heap)
    slots: list[list[int]] = [[] for _ in range(nslots)]
    loads = [0] * nslots
    for chain in sorted(range(len(costs)), key=lambda c: (-costs[c], c)):
        load, slot = heapq.heappop(heap)
        slots[slot].append(chain)
        loads[slot] = load + costs[chain]
        heapq.heappush(heap, (loads[slot], slot))
    return slots, loads


def _wrap_around(chunks, full, span, nslots):
    """McNaughton wrap-around packing at makespan ``span`` (chunks).

    Chains fill slots in order. The chain crossing a slot boundary is cut at a
    chunk boundary: its first part (a producer, full chunks only) starts the
    next slot and its remainder (the consumer) ends the current one. Producers
    lead their slots and never wait, so every consumer's producer runs on a
    resident CTA and the handoff graph cannot deadlock. Returns None when the
    chains do not fit in ``nslots`` slots.
    """
    slots = [[]]
    load = 0
    # Alternate long and short chains so every slot gets a similar mix of
    # chain starts; short chains cost more per chunk than the model assumes.
    ranked = sorted(range(len(chunks)), key=lambda c: (-chunks[c], c))
    order = [
        ranked[i // 2] if i % 2 == 0 else ranked[-1 - i // 2]
        for i in range(len(ranked))
    ]
    for chain in order:
        count = chunks[chain]
        if load + count + PIECE_OVERHEAD <= span:
            slots[-1].append((chain, 0, count))
            load += count + PIECE_OVERHEAD
            continue
        tail = span - load - 2 * PIECE_OVERHEAD
        head = count - tail
        if tail >= MIN_PIECE and MIN_PIECE <= head <= full[chain]:
            slots[-1].append((chain, head, count))
            slots.append([(chain, 0, head)])
            load = head + PIECE_OVERHEAD
        else:
            slots.append([(chain, 0, count)])
            load = count + PIECE_OVERHEAD
        if len(slots) > nslots:
            return None
    return slots


def _split_heads(chunks, full, nslots, span_limit):
    """LPT after cutting a short head off the longest chains.

    Each head is a producer placed before every other piece of its slot, so
    producers never wait; the rest of the chain is a consumer that ends its
    slot. Searches a coarse grid of split counts and head lengths with a
    waiting-aware simulation and returns the best (span, slots) below
    ``span_limit``, where slots hold (chain, first chunk, end chunk), or None.
    """
    order = sorted(range(len(chunks)), key=lambda c: (-chunks[c], c))
    best = None
    for n_split in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        if n_split > len(order):
            break
        cut = order[:n_split]
        room = min(full[c] for c in cut)
        for head in (2, 4, 6, 8, 12, 16, 24, 32, 48, 64):
            if head > room or chunks[cut[-1]] - head < MIN_PIECE:
                break
            # (cost, kind, chain, first chunk, end chunk); kind 0 = producer,
            # 1 = whole chain, 2 = consumer.
            jobs = [
                (chunks[c] + PIECE_OVERHEAD, 1, c, 0, chunks[c])
                for c in order[n_split:]
            ]
            for c in cut:
                jobs.append((head + PIECE_OVERHEAD, 0, c, 0, head))
                jobs.append((chunks[c] - head + PIECE_OVERHEAD, 2, c, head, chunks[c]))
            heap = [(0, slot) for slot in range(nslots)]
            slots = [[] for _ in range(nslots)]
            for job in sorted(jobs, key=lambda j: (-j[0], j[1], j[2])):
                load, slot = heapq.heappop(heap)
                slots[slot].append(job)
                heapq.heappush(heap, (load + job[0], slot))
            ready = {}
            for slot in slots:
                slot.sort(key=lambda j: (j[1], -j[0]))
                t = 0
                for cost, kind, c, _, _ in slot:
                    if kind == 0:
                        t += cost
                        ready[c] = t
            span = 0
            for slot in slots:
                t = 0
                for cost, kind, c, _, _ in slot:
                    if kind == 2:
                        t = max(t, ready[c])
                    t += cost
                span = max(span, t)
            if span < span_limit and (best is None or span < best[0]):
                best = (span, [[j[2:] for j in slot] for slot in slots])
    return best


def _piece_schedule(lens, H: int, nslots: int, dev: torch.device):
    """Pack the (sequence, head) chains onto at most ``nslots`` persistent CTAs.

    Whole-chain LPT is used unless cutting chains lowers the modelled
    makespan. Split schedules use wrap-around packing, with exact FP32 state
    handoffs between the pieces of a chain.
    """
    nchains = len(lens) * H
    chunks = [(lens[c // H] + C_TOK - 1) // C_TOK for c in range(nchains)]
    full = [lens[c // H] // C_TOK for c in range(nchains)]
    G = min(nchains, nslots)
    costs = [count + PIECE_OVERHEAD for count in chunks]
    lpt_slots, loads = _lpt(costs, G)
    items = [[(c, 0, lens[c // H], -1, -1) for c in slot] for slot in lpt_slots]
    nbuf = 0
    span = max(max(costs), -(-sum(costs) // G))
    limit = max(loads) * (1 - MIN_SPLIT_GAIN)
    if nchains > G and span <= limit:
        packed = None
        while packed is None and span <= limit:
            packed = _wrap_around(chunks, full, span, G)
            span += 1
        heads = _split_heads(chunks, full, G, span if packed else limit + 1)
        if heads is not None:
            packed = heads[1]
        if packed is not None:
            split = sorted({chain for slot in packed for chain, c0, _ in slot if c0})
            buffer_of = {chain: index for index, chain in enumerate(split)}
            items = [
                [
                    (
                        chain,
                        c0 * C_TOK,
                        min(c1 * C_TOK, lens[chain // H]) - c0 * C_TOK,
                        buffer_of[chain] if c0 else -1,
                        buffer_of[chain] if c1 < chunks[chain] else -1,
                    )
                    for chain, c0, c1 in slot
                ]
                for slot in packed
            ]
            nbuf = len(buffer_of)
    fc, ft0, ftn, fsrc, fdst, off = [], [], [], [], [], [0]
    for slot in items:
        for c, t0, tn, src, dst in slot:
            fc.append(c)
            ft0.append(t0)
            ftn.append(tn)
            fsrc.append(src)
            fdst.append(dst)
        off.append(len(fc))
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return (
        i32(off),
        i32(fc),
        i32(ft0),
        i32(ftn),
        i32(fsrc),
        i32(fdst),
        len(items),
        nbuf,
    )


def make_plan(offsets, H, dev):
    """Build private scheduling and handoff storage from validated host offsets."""
    lens = [end - start for start, end in zip(offsets, offsets[1:], strict=False)]
    cu32 = torch.tensor(offsets, dtype=torch.int32, device=dev)
    soff, schain, spt0, sptn, ssrc, sdst, G, nbuf = _piece_schedule(
        lens, H, _sm_count(dev), dev
    )
    # Keep every inter-CTA handoff in FP32, including shapes outside INT21.
    mid = torch.empty(max(nbuf, 1), D, D, dtype=torch.float32, device=dev)
    flags = torch.zeros(max(nbuf, 1), dtype=torch.int32, device=dev)
    return (
        cu32,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        G,
        mid,
        flags,
        nbuf > 0,
        all(length % C_TOK == 0 for length in lens),
    )


@torch.no_grad()
def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    plan=None,
):
    H = q.shape[2]
    nseq = 1 if cu_seqlens is None else int(cu_seqlens.numel()) - 1
    fixed = nseq == 1
    store_final_state = final_state is not None

    (
        cu32,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        G,
        mid,
        mfl,
        has_split,
        full_chunks,
    ) = plan
    # Reset before every launch, including captured launches and graph replays.
    if has_split:
        mfl.zero_()
    fepoch = 1
    # v98 fused-gate kernel variant: routed to single-sequence (fixed)
    # shapes only — it wins ~1-3% there (long chains, steady-state L1
    # relief) but costs ~2% on short varlen chains (prep-latency exposure
    # at chain starts; profiles/NOTES.md session 2).
    common = {
        "initial_state": initial_state,
        "final_state": final_state,
        "cu_seqlens": cu32,
        "sched": (soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, fepoch),
        "gate2": 1 if fixed else 0,
    }
    # NCU v119 source counters identify the prep warpgroup's raw-Q/K indexing
    # as a compiler-spill trigger.  Mode 4 is algebraically just a four-row
    # half-warp pairing through the combined prep-thread coordinate.  Mode 8
    # keeps that row map and enables the fused per-channel gate scan while
    # retaining varlen's measured-faster raw V/residual layout.
    qk_rowpair = 4 if fixed else 8
    # Central, preaddressed Gram issue relieves generic-tail prep scheduling.
    # Full-chunk varlen routes retain per-instance self-issue.
    generic = not fixed and not full_chunks
    gram_w9 = 2 if generic else 0
    # Full-chunk varlen favors 160/88/24/48. Packed normalization lets generic
    # mixed use 152/96/24/48 without H96's old 56-reg prep. Both fixed shapes
    # use 160/88/24/48 after the strict final-state retune.
    reg_mode = 1 if fixed else 0
    if not fixed:
        reg_mode = 1 if full_chunks or (H == 96 and store_final_state) else 2
    # Split schedules need every producer CTA resident, which clusters would
    # not guarantee. Unsplit single sequences keep cluster-4 launches.
    cluster_size = 1 if has_split else (4 if fixed else 1)
    while G % cluster_size:
        cluster_size //= 2
    return _launch_forward(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        has_split=has_split,
        beta_tma=True,
        qk_rowpair=qk_rowpair,
        gram_w9=gram_w9,
        restore_tail=False,
        reg_mode=reg_mode,
        full_chunks=full_chunks,
        # H64 generic also benefits from deleting the runtime lower-bound
        # scalar; the same specialization regresses generic H96.
        lower_bound_m5=(full_chunks or H == 64) and lower_bound == -5.0,
        approximate_split=False,
        prep_peel=H == 96 and generic,
        cluster_size=cluster_size,
        # Rolling eight-row gate batches also shorten generic H64 by about
        # 0.17% in matched 5x100 bookends; full-chunk H64 still prefers the
        # scalar schedule.
        gate_roll=generic,
        # Varlen routes load partial last chunks by TMA and mask them; the
        # single-sequence route keeps its per-element tail (2% faster).
        partial_tma=not fixed,
        **common,
    )
