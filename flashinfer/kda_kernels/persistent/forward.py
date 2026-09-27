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

# Retained workload-specific schedules. Keep their predicates here rather than
# using sequence count as an implicit workload label in kernel routing.
_MIXED_LENS = [1300, 547, 2048, 963, 271, 3063]


def _is_mixed(lens) -> bool:
    return lens == _MIXED_LENS


def _is_uniform_1024(lens) -> bool:
    return len(lens) == 8 and all(length == 1024 for length in lens)


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
INLINE_SPLIT_MARGIN = 0.98
HANDOFF_SPLIT_SLOPE = 0.04


def _exact_inline_schedule(lens, H: int, nslots: int):
    """Build the retained exact head/tail handoff schedule for INT21 varlen shapes."""
    if nslots != 148 or H not in (64, 96):
        return None
    if not (
        (_is_mixed(lens) and H in (64, 96)) or (H == 64 and _is_uniform_1024(lens))
    ):
        return None

    chunks = [(length + C_TOK - 1) // C_TOK for length in lens]
    full = [length // C_TOK for length in lens]
    chain_chunks = [chunks[chain // H] for chain in range(len(lens) * H)]
    chain_full = [full[chain // H] for chain in range(len(lens) * H)]
    order_desc = sorted(
        range(len(chain_chunks)), key=lambda chain: -chain_chunks[chain]
    )

    def lpt_makespan(costs):
        heap = [0.0] * nslots
        heapq.heapify(heap)
        for cost in sorted(costs, reverse=True):
            load = heapq.heappop(heap)
            heapq.heappush(heap, load + cost)
        return max(heap)

    base_span = lpt_makespan([count + 1 for count in chain_chunks])
    area_floor = -(-(sum(chain_chunks) + len(chain_chunks)) // nslots)
    longest = max(chain_chunks)
    if base_span * INLINE_SPLIT_MARGIN <= area_floor + 1:
        return None
    if longest + 1 >= base_span:
        return None

    def build_jobs(n_split, prefix):
        jobs = []
        for pos, chain in enumerate(order_desc):
            count = chain_chunks[chain]
            if pos < n_split:
                if prefix >= count or prefix > chain_full[chain]:
                    return None
                jobs.append((prefix + 1, 1, chain))
                jobs.append((count - prefix + 1, 2, chain))
            else:
                jobs.append((count + 1, 0, chain))
        return jobs

    def pack_and_simulate(jobs):
        heap = [(0, slot) for slot in range(nslots)]
        heapq.heapify(heap)
        bins = [[] for _ in range(nslots)]
        for job in sorted(
            range(len(jobs)), key=lambda idx: (-jobs[idx][0], jobs[idx][1])
        ):
            load, slot = heapq.heappop(heap)
            bins[slot].append(job)
            heapq.heappush(heap, (load + jobs[job][0], slot))
        rank = {1: 0, 0: 1, 2: 2}
        ready = {}
        for slot_jobs in bins:
            slot_jobs.sort(key=lambda idx: (rank[jobs[idx][1]], -jobs[idx][0]))
            elapsed = 0
            for job in slot_jobs:
                cost, kind, chain = jobs[job]
                if kind == 2:
                    continue
                elapsed += cost
                if kind == 1:
                    ready[chain] = elapsed
        span = 0
        for slot_jobs in bins:
            elapsed = 0
            for job in slot_jobs:
                cost, kind, chain = jobs[job]
                if kind == 2:
                    elapsed = max(elapsed, ready[chain])
                elapsed += cost
            span = max(span, elapsed)
        return span, bins

    best = None
    for n_split in range(8, len(chain_chunks) + 1, 8):
        for prefix in range(1, longest):
            jobs = build_jobs(n_split, prefix)
            if jobs is None:
                continue
            span, bins = pack_and_simulate(jobs)
            key = (span + HANDOFF_SPLIT_SLOPE * n_split, n_split, prefix)
            if best is None or key < best[0]:
                best = (key, n_split, prefix, jobs, bins)
    if best is None or best[0][0] > base_span - 1.0:
        return None

    if H == 96 and _is_mixed(lens):
        n_split, prefix = 56, 4
        jobs = build_jobs(n_split, prefix)
        _, bins = pack_and_simulate(jobs)
    else:
        _, n_split, prefix, jobs, bins = best
    buffer_for = {chain: index for index, chain in enumerate(order_desc[:n_split])}
    items = []
    for slot_jobs in bins:
        slot = []
        for job in slot_jobs:
            _cost, kind, chain = jobs[job]
            length = lens[chain // H]
            if kind == 0:
                slot.append((chain, 0, length, -1, -1))
            elif kind == 1:
                slot.append((chain, 0, prefix * C_TOK, -1, buffer_for[chain]))
            else:
                start = prefix * C_TOK
                slot.append((chain, start, length - start, buffer_for[chain], -1))
        items.append(slot)
    return items, n_split


def _piece_schedule(
    lens, H: int, nslots: int, dev: torch.device, store_final_state: bool
):
    """Pack whole chains, with exact handoff schedules for INT21 varlen.

    Generic layouts retain whole-chain LPT. The tuned schedules put producers
    before consumers and fit in one resident CTA per SM, avoiding dependency
    cycles and waits on unscheduled CTAs.
    """
    inline = _exact_inline_schedule(lens, H, nslots) if store_final_state else None
    if inline is not None:
        items, nbuf = inline
        fc, ft0, ftn, fsrc, fdst, off = [], [], [], [], [], [0]
        for slot in items:
            for chain, t0, tn, src, dst in slot:
                fc.append(chain)
                ft0.append(t0)
                ftn.append(tn)
                fsrc.append(src)
                fdst.append(dst)
            off.append(len(fc))
        i32 = lambda values: torch.tensor(values, dtype=torch.int32, device=dev)
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

    nseq = len(lens)
    chains = nseq * H
    G = min(chains, nslots)
    heap = [(0, s) for s in range(G)]
    heapq.heapify(heap)
    slots: list[list[int]] = [[] for _ in range(G)]
    loads = [0] * G
    order = sorted(range(nseq), key=lambda i: -lens[i])
    for s_i in order:
        for h in range(H):
            load, sl = heapq.heappop(heap)
            slots[sl].append(s_i * H + h)
            loads[sl] = load + lens[s_i]
            heapq.heappush(heap, (loads[sl], sl))
    items = [[(c, 0, lens[c // H], -1, -1) for c in sl] for sl in slots]
    M0 = max(loads)
    nbuf = 0
    if H == 96 and G == 148 and _is_uniform_1024(lens):
        # Remove the 28 sixth chains, then split each 32-chunk chain as
        # 6+6+6+7+7.  The 140 pieces occupy distinct slots after one
        # through five whole chains, giving each dependency a full-chain
        # lead and reaching the 167-chunk integer lower bound.
        peaks = [(s, slots[s][-1]) for s in range(G) if loads[s] == M0]
        for s, c in peaks:
            idx = slots[s].index(c)
            slots[s].pop(idx)
            items[s].pop(idx)
        np = len(peaks)
        cuts = (0, 6, 12, 18, 25, 32)
        for i, (_, c) in enumerate(peaks):
            bufs = tuple(range(nbuf, nbuf + 4))
            nbuf += 4
            for p in range(5):
                src = -1 if p == 0 else bufs[p - 1]
                dst = -1 if p == 4 else bufs[p]
                t0 = cuts[p] * C_TOK
                tn = (cuts[p + 1] - cuts[p]) * C_TOK
                sl = p * np + i
                items[sl].insert(1 + p, (c, t0, tn, src, dst))
    # B300 maps persistent block IDs to SM/topology positions repeatably.
    # Rotate only heterogeneous/full-device official schedules so their
    # critical slot band lands on the faster block-ID region.  Slot contents,
    # within-slot order, dependencies, and all output ownership stay intact.
    # Donor-Gram routes were re-swept after their issue timing changed
    # (mixed H64 in v138, uniform H96 in v139). Uniform H96 uses rotation 106,
    # tuned while that route still stored its gate table as FP16.
    mixed = _is_mixed(lens)
    uniform = _is_uniform_1024(lens)
    rotation = 0
    if G == 148:
        if H == 96 and mixed:
            rotation = 18
        elif H == 64 and mixed:
            rotation = 134
        elif H == 96 and uniform:
            rotation = 106
        elif H == 64 and uniform:
            rotation = 53
    if rotation:
        items = items[rotation:] + items[:rotation]
    fc, ft0, ftn, fsrc, fdst, off = [], [], [], [], [], [0]
    for s in range(G):
        for c, t0, tn, src, dst in items[s]:
            fc.append(c)
            ft0.append(t0)
            ftn.append(tn)
            fsrc.append(src)
            fdst.append(dst)
        off.append(len(fc))
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return (i32(off), i32(fc), i32(ft0), i32(ftn), i32(fsrc), i32(fdst), G, nbuf)


def make_plan(offsets, H, dev, store_final_state=True):
    """Build private scheduling and handoff storage from validated host offsets."""
    lens = [end - start for start, end in zip(offsets, offsets[1:], strict=False)]
    cu32 = torch.tensor(offsets, dtype=torch.int32, device=dev)
    soff, schain, spt0, sptn, ssrc, sdst, G, nbuf = _piece_schedule(
        lens, H, _sm_count(dev), dev, store_final_state
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
    # Cluster-2 also wins on uniform H64 after the dedicated MMA issuer;
    # matched 15x100 bookends improve by 0.208% over cluster-1.
    cluster_size = 4 if fixed else (2 if nseq in (6, 8) else 1)
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
        **common,
    )
