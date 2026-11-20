"""Cross-NUMA (xnuma) transport for the p2p_split backend.

Productized form of the qualified V10 experiment. Same-NUMA peers use the
unchanged p2p_direct path. A cross-NUMA peer is represented on every rank by a
local *shadow segment* (see ``transport/shadow.py``); this module keeps the
shadows in sync with GIN:

* ``bump``      (caller stream, before compute): epoch += 1, clear the K2
                 per-expert progress counters.
* ``sender``    (caller stream, after ``bump`` and before the compute graph):
                 per cross owner, one span put of this rank's activation |
                 scale | topk-weight buffers into the owner's shadow-of-me, a
                 record (per-expert counts + ``token * topk + slot`` lists,
                 built in O(routes)) into the owner's inbox, then a
                 non-blocking strong commit (value = epoch); its local flush
                 is deferred to the next call. K1 on the owner waits for the
                 commit and installs the record (``token_comm.xnuma``). The
                 sender must precede K1 in stream order: K1 spins on the
                 commits, and persistent K1/K2 CTAs can otherwise occupy every
                 SM before an unordered sender is scheduled.
* ``forwarder`` (forked after K1 inside the compute graph, beside K2): per
                 cross source, ``fsplit`` CTAs (experts split by parity) poll the
                 K2 per-expert progress counters; each newly completed expert's
                 rows are packed in completion order (16 B header + row) and
                 sent with one contiguous put to the source's return region;
                 a non-blocking strong commit counts completion. The local
                 flush of the previous epoch is deferred to the next call.
* ``scatter``   (after compute, before K3): wait until every cross owner has
                 committed ``fsplit`` times this epoch, then copy each returned
                 row to ``combine_output[token, slot]``.

Cross-call reuse: a source's next send happens after its previous call
completed, which needed every owner's completion commit, which an owner only
issues after its K2 consumed the inputs. An owner's next forward needs the
source's next dispatch, i.e. after the source finished its previous K3.

Requirements: 8-warp K2 (``Sm120Fc2CombineKernel``), non-streaming K3,
``experts_per_rank <= 64``, inputs allocated contiguously in segment 0.
"""

from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cutlass_dsl import Int32, Int64

from .xnuma_layout import XnumaLayout


@dataclass(frozen=True)
class XnumaK1Config:
    """Compile-time constants for the K1 hooks in ``token_comm``."""

    epoch_addr: int
    dsig_addr: int
    inbox_addr: int
    record_bytes: int
    cross_sources: Tuple[int, ...]
    experts_per_rank: int
    tokens_per_rank: int
    k2tot_addr: int
    k2meta_addr: int


_TABLE_FIELDS = 8  # per cross peer j: peer, send_span_dst, inbox_dst, dsig_dst, shadow_comb, ret_dst, done_dst, -


class XnumaTransport:
    def __init__(self, tester, session, layout: XnumaLayout, *, region: XnumaLayout = None,
                 epoch: torch.Tensor = None, input_regions=None, zero_reserved: bool = True):
        """``layout`` is this call shape. Optional FlashInfer arguments:

        * ``region``: layout whose offsets/strides place the reserved control
          region (default ``layout``). Transports of different token buckets
          sharing one workspace pass the same, largest region so every signal
          word and commit-source word keeps a fixed address.
        * ``epoch``: shared device epoch (default: a private one). GIN commits
          are signal-adds keyed by source word, so transports sharing a region
          must also share the epoch.
        * ``input_regions``: input tensors sent as separate puts (default: one
          span from the first to the last input buffer).
        * ``zero_reserved``: clear the control region (only before first use).
        """
        self.tester = tester
        self.session = session
        self.layout = layout
        region = layout if region is None else region
        if (region.world, region.n_cross, region.experts_per_rank, region.topk, region.hidden,
                region.fsplit) != (layout.world, layout.n_cross, layout.experts_per_rank,
                                   layout.topk, layout.hidden, layout.fsplit) or (
                region.tokens_per_rank < layout.tokens_per_rank):
            raise ValueError("xnuma region layout does not cover the call layout")
        self.region = region
        shadow = session.shadow
        me = tester.rank
        self.rank = me
        self.cross = shadow.cross_peers(me)
        if len(self.cross) != layout.n_cross:
            raise ValueError("layout n_cross differs from the topology")
        k2 = tester._kernel_k2
        from ..kernel_fc2_combine import Sm120Fc2CombineKernel

        if type(k2) is not Sm120Fc2CombineKernel:
            raise RuntimeError("p2p_split needs the 8-warp K2 (Sm120Fc2CombineKernel)")
        if layout.hidden % int(k2.mma_tiler[0]):
            raise RuntimeError("hidden must be a multiple of the K2 M tile")
        self.hidden_tiles = layout.hidden // int(k2.mma_tiler[0])
        self.compute_warps = len(k2.compute_warp_id)

        arena = session.p2p.base
        self.reserved = arena + session.reserved_offset
        if region.reserved_bytes > session.reserved_bytes:
            raise RuntimeError("p2p_split reserved region is smaller than the xnuma layout")
        if zero_reserved:
            session.reserved_view(0, region.reserved_bytes).zero_()
        off = region.offsets
        self.addr = {name: self.reserved + value for name, value in off.items()}

        inputs = (tester.my_activation, tester.my_activation_sf, tester.my_topk_idx, tester.my_topk_weights)
        span_lo = min(int(t.data_ptr()) for t in inputs)
        span_hi = max(int(t.data_ptr()) + t.numel() * t.element_size() for t in inputs)
        combine = tester.combine_output
        co_lo = int(combine.data_ptr())
        co_hi = co_lo + combine.numel() * combine.element_size()
        # Shadow segments may cover only a prefix of segment 0.
        shadow_end = arena + int(getattr(session, "shadow_limit", session.reserved_offset))
        if not (arena <= span_lo and span_hi <= self.reserved and (co_hi <= span_lo or co_lo >= span_hi)):
            raise RuntimeError("xnuma needs the input buffers contiguous in segment 0, disjoint from combine_output")
        if span_hi > shadow_end or co_hi > shadow_end:
            raise RuntimeError("xnuma inputs/combine_output lie outside the shadow segment")
        if tuple(combine.shape) != (layout.tokens_per_rank, layout.topk, layout.hidden) or combine.dtype != torch.bfloat16:
            raise RuntimeError("xnuma expects a per-slot BF16 combine_output")
        if input_regions is None:
            regions = ((span_lo, span_hi - span_lo),)
        else:
            regions = []
            for t in input_regions:
                if not t.is_contiguous():
                    raise RuntimeError("xnuma input regions must be contiguous")
                lo = int(t.data_ptr())
                regions.append((lo, t.numel() * t.element_size()))
            if any(not (span_lo <= lo and lo + n <= span_hi) for lo, n in regions):
                raise RuntimeError("xnuma input regions must lie inside the input span")
        # (source address, offset inside segment 0, bytes).
        self.input_regions = tuple((lo, lo - arena, n) for lo, n in regions if n > 0)
        self.span_lo, self.span_bytes = span_lo, span_hi - span_lo
        self.combine_addr = co_lo

        stride = region.max_routes * region.slot_bytes
        rows = []
        for j, peer in enumerate(self.cross):
            mine_on_peer = shadow.segment_of(peer, me) - 1  # my slot index on that peer
            rows.append([
                peer,
                arena + session.segment_offset(mine_on_peer + 1),  # owner's shadow-of-me (segment base)
                self.addr["inbox"] + mine_on_peer * region.record_bytes,
                self.addr["dsig"] + mine_on_peer * 8,
                arena + session.segment_offset(j + 1) + (co_lo - arena),  # local shadow of source peer
                self.addr["ret"] + mine_on_peer * region.fsplit * stride,  # source's return region for me
                self.addr["done"] + mine_on_peer * 8,
                0,
            ])
        self.table = torch.tensor([v for row in rows for v in row], dtype=torch.int64, device="cuda")
        self.epoch = torch.zeros((1,), dtype=torch.int64, device="cuda") if epoch is None else epoch
        self.forward_stream = torch.cuda.Stream()
        self._compile()

    # -- configs for the K1/K2 hooks ------------------------------------------
    def k1_config(self) -> XnumaK1Config:
        L = self.layout
        return XnumaK1Config(
            epoch_addr=int(self.epoch.data_ptr()), dsig_addr=self.addr["dsig"], inbox_addr=self.addr["inbox"],
            record_bytes=self.region.record_bytes, cross_sources=tuple(self.cross), experts_per_rank=L.experts_per_rank,
            tokens_per_rank=L.tokens_per_rank, k2tot_addr=self.addr["k2tot"], k2meta_addr=self.addr["k2meta"],
        )

    @property
    def k2_progress_addr(self) -> int:
        return self.addr["k2cnt"]

    # -- kernels ----------------------------------------------------------------
    def _compile(self) -> None:
        from . import gin_bridge as gin

        L = self.layout
        G = self.region  # control-region placement (== L unless shared)
        E, T, TOPK = L.experts_per_rank, L.tokens_per_rank, L.topk
        N, FS = L.n_cross, L.fsplit
        REC_I32, REC_B = L.record_i32, G.record_bytes  # content / slot stride
        REC_PUT = L.record_bytes
        ROW, SLOT = L.row_bytes, L.slot_bytes
        STRIDE = G.max_routes * SLOT
        REGIONS = self.input_regions
        HTNW = self.hidden_tiles * self.compute_warps
        A = dict(self.addr)
        TABLE = int(self.table.data_ptr())
        EPOCH = int(self.epoch.data_ptr())
        TOPK_ADDR = int(self.tester.my_topk_idx.data_ptr())
        CO = self.combine_addr
        CH = (T * TOPK + 127) // 128  # sender: routes per thread
        ROW8 = ROW // 8
        UNROLL = ROW8 // 128 if ROW8 % 128 == 0 else 0
        half_masks = [sum(1 << e for e in range(E) if e % FS != h) for h in range(FS)]
        cross_mask = sum(1 << p for p in self.cross)

        class _Obj:
            pass

        tx, fw, rx = _Obj(), _Obj(), _Obj()
        # FlashInfer sessions own their GIN state; the standalone runner uses
        # the gin_bridge module session.
        bind = getattr(self.session, "bind_network_kernel", None) or gin.bind
        for obj, role in ((tx, "dispatch_tx"), (fw, "return_tx"), (rx, "return_rx")):
            obj.hybrid_force_ibgda_mask = cross_mask
            bind(obj, role)

        def ptr(dtype, addr, align):
            return cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=align)

        def u8(addr, n):
            return cute.make_tensor(ptr(cutlass.Uint8, addr, 16), cute.make_layout(Int32(n)))

        @cute.jit
        def tbl(j, field: cutlass.Constexpr):
            t = cute.make_tensor(ptr(Int64, Int64(TABLE), 8), cute.make_layout(N * _TABLE_FIELDS))
            return t[j * Int32(_TABLE_FIELDS) + Int32(field)]

        @cute.jit
        def copy_row(dst, src, tid):
            d = cute.make_tensor(ptr(Int64, dst, 16), cute.make_layout(ROW8))
            s = cute.make_tensor(ptr(Int64, src, 16), cute.make_layout(ROW8))
            if cutlass.const_expr(UNROLL > 0):
                vals = [s[tid + Int32(128 * k)] for k in range(UNROLL)]
                for k in cutlass.range_constexpr(UNROLL):
                    d[tid + Int32(128 * k)] = vals[k]
            else:
                x = tid
                while x < Int32(ROW8):
                    d[x] = s[x]
                    x = x + Int32(128)

        @cute.kernel
        def bump_kernel():
            tid, _, _ = cute.arch.thread_idx()
            if tid == Int32(0):
                p = ptr(Int64, Int64(EPOCH), 8)
                v = cute.arch.load(p, Int64, sem="relaxed", scope="gpu")
                cute.arch.store(p, v + Int64(1), sem="release", scope="gpu")
            cnt = cute.make_tensor(ptr(Int32, Int64(A["k2cnt"]), 4), cute.make_layout(E))
            e = tid
            while e < Int32(E):
                cnt[e] = Int32(0)
                e = e + Int32(128)
            cute.arch.fence_acq_rel_gpu()

        @cute.jit
        def bump(stream: cuda.CUstream):
            bump_kernel().launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)

        @cute.kernel
        def sender_kernel():
            j, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            j = Int32(j)
            owner = Int32(tbl(j, 0))
            epoch = cute.arch.load(ptr(Int64, Int64(EPOCH), 8), Int64, sem="acquire", scope="gpu")
            # Retire the previous call's puts from this CTA's GIN slot (deferred
            # flush: by the time a call returns, every owner consumed them).
            gin.quiet(tx)
            # Data first; the record build overlaps the transfer.
            for r in cutlass.range_constexpr(len(REGIONS)):
                gin.put_nbi_block(tx, u8(tbl(j, 1) + Int64(REGIONS[r][1]), REGIONS[r][2]),
                                  u8(Int64(REGIONS[r][0]), REGIONS[r][2]), owner)
            rec_addr = Int64(A["txrec"]) + Int64(j) * Int64(REC_B)
            rec = cute.make_tensor(ptr(Int32, rec_addr, 4), cute.make_layout(REC_I32))
            topk = cute.make_tensor(ptr(Int64, Int64(TOPK_ADDR), 8), cute.make_layout(T * TOPK))
            # O(routes) record build. Thread t owns the contiguous route chunk
            # [t*CH, (t+1)*CH): per-thread counts per expert, an exclusive scan
            # over threads per expert, then in-order writes, so every expert
            # list is ascending in token*topk+slot (same as a serial build).
            scr = cute.make_tensor(ptr(Int32, Int64(A["sscr"]) + Int64(j) * Int64(E * 128 * 4), 4),
                                   cute.make_layout(E * 128))
            for e in cutlass.range_constexpr(E):
                scr[Int32(e * 128) + tid] = Int32(0)
            lo = tid * Int32(CH)
            hi = lo + Int32(CH)
            if hi > Int32(T * TOPK):
                hi = Int32(T * TOPK)
            i = lo
            while i < hi:
                ex = topk[i]
                if ex >= Int64(0):
                    if Int32(ex // Int64(E)) == owner:
                        slot_i = Int32(ex % Int64(E)) * Int32(128) + tid
                        scr[slot_i] = scr[slot_i] + Int32(1)
                i = i + Int32(1)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            if tid < Int32(E):
                running = Int32(0)
                base = tid * Int32(128)
                t = Int32(0)
                while t < Int32(128):
                    v = scr[base + t]
                    scr[base + t] = running
                    running = running + v
                    t = t + Int32(1)
                rec[tid] = running
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            i = lo
            while i < hi:
                ex = topk[i]
                if ex >= Int64(0):
                    if Int32(ex // Int64(E)) == owner:
                        le = Int32(ex % Int64(E))
                        slot_i = le * Int32(128) + tid
                        pos = scr[slot_i]
                        rec[Int32(E) + le * Int32(T) + pos] = i
                        scr[slot_i] = pos + Int32(1)
                i = i + Int32(1)
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            gin.put_nbi_block(tx, u8(tbl(j, 2), REC_PUT), u8(rec_addr, REC_PUT), owner)
            gin.fence()
            word = Int64(A["txd"]) + Int64(j) * Int64(8)
            if tid == Int32(0):
                cute.arch.store(ptr(Int64, word, 8), epoch, sem="relaxed", scope="gpu")
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            gin.put_nbi_block(  # 8-byte put = strong VA signal-add commit record
                tx,
                cute.make_tensor(ptr(Int64, tbl(j, 3), 8), cute.make_layout(1)),
                cute.make_tensor(ptr(Int64, word, 8), cute.make_layout(1)),
                owner,
            )

        @cute.jit
        def sender(stream: cuda.CUstream):
            sender_kernel().launch(grid=[N, 1, 1], block=[128, 1, 1], stream=stream)

        @cute.kernel
        def forwarder_kernel():
            b, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            b = Int32(b)
            j = b // Int32(FS)
            h = b % Int32(FS)
            source = Int32(tbl(j, 0))
            epoch = cute.arch.load(ptr(Int64, Int64(EPOCH), 8), Int64, sem="acquire", scope="gpu")
            gin.quiet(fw)  # previous call's puts from this CTA's stage are retired before reuse
            if tid == Int32(0):
                mp = ptr(Int64, Int64(A["k2meta"]), 8)
                m = cute.arch.load(mp, Int64, sem="acquire", scope="gpu")
                while m < epoch:
                    m = cute.arch.load(mp, Int64, sem="acquire", scope="gpu")
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            rec = cute.make_tensor(ptr(Int32, Int64(A["inbox"]) + Int64(j) * Int64(REC_B), 4), cute.make_layout(REC_I32))
            tot = cute.make_tensor(ptr(Int32, Int64(A["k2tot"]), 4), cute.make_layout(E))
            scr = cute.make_tensor(ptr(Int32, Int64(A["fscr"]) + Int64(b) * Int64(L.scratch_bytes), 4), cute.make_layout(E + 2))
            shadow_comb = tbl(j, 4)
            stage = Int64(A["stage"]) + Int64(b) * Int64(STRIDE)
            ret_base = tbl(j, 5) + Int64(h) * Int64(STRIDE)
            sent = Int64(half_masks[0])
            if cutlass.const_expr(FS == 2):
                if h == Int32(1):
                    sent = Int64(half_masks[1])
            go = Int32(1)
            wpos = Int32(0)
            while go == Int32(1):
                if tid == Int32(0):
                    nnew = Int32(0)
                    for e in cutlass.range_constexpr(E):
                        f = Int32(0)
                        if (sent >> Int64(e)) & Int64(1) == Int64(0):
                            c = cute.arch.load(ptr(Int32, Int64(A["k2cnt"] + 4 * e), 4), Int32, sem="acquire", scope="gpu")
                            if c >= tot[e] * Int32(HTNW):
                                f = Int32(1)
                                sent = sent | Int64(1 << e)
                                nnew = nnew + Int32(1)
                        scr[e] = f
                    scr[E] = nnew
                    done_all = Int32(0)
                    if sent == Int64((1 << E) - 1):
                        done_all = Int32(1)
                    scr[E + 1] = done_all
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                nnew_all = scr[E]
                done_all = scr[E + 1]
                if nnew_all > Int32(0):
                    w0 = wpos
                    for e in cutlass.range_constexpr(E):
                        if scr[e] == Int32(1):
                            ne = rec[e]
                            pos = Int32(0)
                            while pos < ne:
                                v = Int64(rec[Int32(E + e * T) + pos])
                                slot = stage + Int64(wpos) * Int64(SLOT)
                                if tid == Int32(0):
                                    cute.arch.store(ptr(Int64, slot, 8), v, sem="relaxed", scope="gpu")
                                copy_row(slot + Int64(16), shadow_comb + v * Int64(ROW), tid)
                                wpos = wpos + Int32(1)
                                pos = pos + Int32(1)
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.barrier(barrier_id=0, number_of_threads=128)
                    if wpos > w0:
                        nbytes = Int64(wpos - w0) * Int64(SLOT)
                        gin.put_nbi_block(fw, u8(ret_base + Int64(w0) * Int64(SLOT), nbytes),
                                          u8(stage + Int64(w0) * Int64(SLOT), nbytes), source)
                cute.arch.barrier(barrier_id=0, number_of_threads=128)
                go = Int32(1) - done_all
            gin.fence()
            word = Int64(A["txf"]) + Int64(b) * Int64(8)
            if tid == Int32(0):
                cute.arch.store(ptr(Int64, word, 8), epoch, sem="relaxed", scope="gpu")
            cute.arch.fence_acq_rel_sys()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            gin.put_nbi_block(  # 8-byte put = strong VA signal-add commit record
                fw,
                cute.make_tensor(ptr(Int64, tbl(j, 6), 8), cute.make_layout(1)),
                cute.make_tensor(ptr(Int64, word, 8), cute.make_layout(1)),
                source,
            )

        @cute.jit
        def forwarder(stream: cuda.CUstream):
            forwarder_kernel().launch(grid=[N * FS, 1, 1], block=[128, 1, 1], stream=stream)

        @cute.kernel
        def scatter_kernel():
            b, _, _ = cute.arch.block_idx()
            tid, _, _ = cute.arch.thread_idx()
            j = Int32(b) // Int32(4)
            q4 = Int32(b) % Int32(4)
            epoch = cute.arch.load(ptr(Int64, Int64(EPOCH), 8), Int64, sem="acquire", scope="gpu")
            if tid == Int32(0):
                gin.wait_commit64(rx, Int32(tbl(j, 0)), ptr(Int64, Int64(A["done"]) + Int64(j) * Int64(8), 8),
                                  Int64(FS) * epoch)
            cute.arch.barrier(barrier_id=0, number_of_threads=128)
            cute.arch.fence_acq_rel_sys()
            rec = cute.make_tensor(ptr(Int32, Int64(A["txrec"]) + Int64(j) * Int64(REC_B), 4), cute.make_layout(REC_I32))
            for hh in cutlass.range_constexpr(FS):
                n = Int32(0)
                for e in cutlass.range_constexpr(E):
                    if cutlass.const_expr(e % FS == hh):
                        n = n + rec[e]
                ret = Int64(A["ret"]) + (Int64(j) * Int64(FS) + Int64(hh)) * Int64(STRIDE)
                k = q4
                while k < n:
                    rslot = ret + Int64(k) * Int64(SLOT)
                    vv = cute.arch.load(ptr(Int64, rslot, 8), Int64, sem="relaxed", scope="gpu")
                    copy_row(Int64(CO) + vv * Int64(ROW), rslot + Int64(16), tid)
                    k = k + Int32(4)
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier(barrier_id=0, number_of_threads=128)

        @cute.jit
        def scatter(stream: cuda.CUstream):
            scatter_kernel().launch(grid=[N * 4, 1, 1], block=[128, 1, 1], stream=stream)

        stream0 = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        link = f"--link-libraries={gin.device_bitcode()}"
        self._bump = cute.compile(bump, stream0)
        self._sender = cute.compile(sender, stream0, options=link)
        self._forwarder = cute.compile(forwarder, stream0, options=link)
        self._scatter = cute.compile(scatter, stream0, options=link)
        _preload_kernels((self._bump, self._sender, self._forwarder, self._scatter))

    # -- debugging ------------------------------------------------------------------
    def debug_snapshot(self) -> dict:
        """Read the control region on a separate non-blocking stream (works
        while a transport wait is stuck): epoch, signals, per-expert K2
        progress versus the K1 totals the forwarder waits for."""
        L = self.region
        device = self.epoch.device
        # Copy-engine transfers only: a stuck transport keeps the SMs busy, so
        # nothing here may launch a kernel.
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            src = self.session.reserved_view(0, L.offsets["ret"])
            raw = torch.empty(src.numel(), dtype=src.dtype, pin_memory=True)
            epoch_host = torch.empty(1, dtype=torch.int64, pin_memory=True)
            with torch.cuda.stream(stream):
                raw.copy_(src, non_blocking=True)
                epoch_host.copy_(self.epoch, non_blocking=True)
            stream.synchronize()
        epoch = epoch_host
        off = L.offsets

        def i64(name, n):
            return raw[off[name]:off[name] + 8 * n].view(torch.int64).tolist()

        def i32(name, n):
            return raw[off[name]:off[name] + 4 * n].view(torch.int32).tolist()

        E = L.experts_per_rank
        cnt, tot = i32("k2cnt", E), i32("k2tot", E)
        need = [t * self.hidden_tiles * self.compute_warps for t in tot]
        return dict(
            epoch=int(epoch[0]), dsig=i64("dsig", L.world), done=i64("done", L.world),
            txd=i64("txd", L.world), txf=i64("txf", L.n_cross * L.fsplit), k2meta=i64("k2meta", 1)[0],
            k2tot=tot, k2cnt=cnt, k2cnt_needed=need,
            unfinished_experts=[e for e in range(E) if cnt[e] < need[e]],
        )

    def start_debug_watchdog(self, seconds: float) -> None:
        """Debug only (MEGAMOE_XNUMA_WATCHDOG=seconds): if the process is still
        running after ``seconds``, print a snapshot every 10 s."""
        import json
        import sys
        import threading
        import time

        def watch():
            time.sleep(seconds)
            while True:
                try:
                    snap = self.debug_snapshot()
                except Exception as exc:  # noqa: BLE001 - debugging aid
                    snap = dict(error=repr(exc))
                print(f"[xnuma watchdog rank {self.rank}] {json.dumps(snap)}", file=sys.stderr, flush=True)
                time.sleep(10)

        threading.Thread(target=watch, daemon=True).start()

    # -- orchestration ------------------------------------------------------------
    @staticmethod
    def _raw(stream):
        return cuda.CUstream(stream.cuda_stream)

    def before_compute(self, stream: torch.cuda.Stream) -> None:
        """Epoch bump, then the sender, both in ``stream`` order *before* the
        compute graph: K1 spins until every cross source's record arrived,
        and persistent K1/K2 CTAs could otherwise occupy every SM before an
        unordered sender is scheduled (observed as a deadlock)."""
        self._bump(self._raw(stream))
        self._sender(self._raw(stream))

    def launch_forwarder_after(self, k1_stream: torch.cuda.Stream) -> torch.cuda.Event:
        """Fork the forwarder from the K1 stream (call right after K1's launch)."""
        k1_launched = torch.cuda.Event()
        k1_launched.record(k1_stream)
        self.forward_stream.wait_event(k1_launched)
        self._forwarder(self._raw(self.forward_stream))
        done = torch.cuda.Event()
        done.record(self.forward_stream)
        return done

    def before_k3(self, stream: torch.cuda.Stream) -> None:
        """Scatter the returned rows into combine_output (the sender, whose
        record the scatter reads, already ran in stream order)."""
        self._scatter(self._raw(stream))


def _preload_kernels(compiled_kernels) -> None:
    """Load every function of each compiled library before persistent
    launches: a first-use module load may synchronize the context behind a
    receiver that waits for this very sender."""
    def checked(result):
        if int(result[0]):
            raise RuntimeError(f"kernel preload failed: {result}")
        return result[1] if len(result) == 2 else result[1:]

    loaded = set()
    for compiled in compiled_kernels:
        lib = int(compiled.library)
        if lib in loaded:
            continue
        loaded.add(lib)
        module = checked(cuda.cuLibraryGetModule(cuda.CUlibrary(lib)))
        count = checked(cuda.cuModuleGetFunctionCount(module))
        for fun in checked(cuda.cuModuleEnumerateFunctions(count, module)):
            checked(cuda.cuFuncLoad(fun))


__all__ = ["XnumaK1Config", "XnumaTransport"]
