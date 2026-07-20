# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/distributed/pcie_dma.py @ fb628056 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""CE-driven PCIe ring allreduce for prefill-size tensors.

NCCL's SM-copy transport sustains ~34 GB/s bus bandwidth on this fabric
while CE peer copies run at ~56 GB/s on every ring hop concurrently
(including the two root-complex crossings, which each own a partition
uplink per direction). This runtime drives a classic reduce-scatter +
all-gather ring where the data plane is CE copies and the SM only
synchronizes (monotonic flag kernels) and reduces, so captured graphs
replay without host patching.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import PCIeOneshotAllReduce

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}
FLAG_STRIDE = 128
FLAG_SLOTS = 256
MAX_PIECES = 8
SCRATCH_ALIGN = 256
FP8_QUANT_BLOCK = 128


def _fp8_mode() -> str:
    """Opt-in E4M3 wire transport mode.

    "ag" (also "1"): keep the saturated bf16 reduce-scatter ring and
    quantize only the allgather phase. Final values quantize exactly once
    at their owner and are forwarded verbatim around the ring, so the
    error cost is a single rounding while AG wire bytes halve.

    "ring": quantize every reduce-scatter hop and the allgather payload,
    keeping the saturated neighbor-only topology while halving both phases.

    "a2a": quantize-once all-to-all (two roundings, half the wire in both
    phases).

    Every FP8 mode materializes the locally owned reduced shard through the
    same FP8 payload as its peers.  An all-reduce result must be rank-identical;
    retaining a pre-wire BF16 owner shard while peers dequantize that shard
    gives every TP rank a different replicated activation.
    """

    return _normalize_fp8_mode(os.getenv("FLASHINFER_EXP_SM12X_PCIE_DMA_FP8", "0"))


def _normalize_fp8_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return ""
    if raw in ("a2a", "ring"):
        return raw
    return "ag"


@lru_cache(maxsize=1)
def _load_extension():
    source = Path(__file__).with_name("pcie_dma.cu")
    verbose = os.getenv("FLASHINFER_EXP_SM12X_PCIE_DMA_VERBOSE_BUILD", "0") == "1"
    return load(
        name="sm12x_pcie_dma_ext",
        sources=[str(source)],
        extra_cuda_cflags=["-O2"],
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class PCIeDmaAllReduce:
    """Single-channel ring allreduce over IPC scratch buffers.

    A channel is a single ordered stream context; concurrent use from
    multiple CUDA streams needs separate channels (same contract as the
    oneshot runtime).
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_bytes: int,
        ext_module=None,
        fp8: Optional[str] = None,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = (
            device
            if isinstance(device, torch.device)
            else torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        )
        if self.device.type != "cuda":
            raise ValueError("PCIe ring allreduce requires a CUDA device")
        if self.world_size < 2:
            raise ValueError("ring allreduce needs at least 2 ranks")
        self.max_bytes = int(max_bytes)
        self._ext = ext_module or _load_extension()
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._closed = False

        self.shard_capacity = _align_up(
            (self.max_bytes + self.world_size - 1) // self.world_size, SCRATCH_ALIGN
        )
        steps = 2 * (self.world_size - 1)
        flags_bytes = FLAG_SLOTS * FLAG_STRIDE
        slab_bytes = flags_bytes + steps * self.shard_capacity
        self._slab = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group, slab_bytes, zero_fill=True, ipc=self._ipc
        )
        self._flags_base = list(self._slab.peer_ptrs)
        self._scratch_base = [ptr + flags_bytes for ptr in self._slab.peer_ptrs]
        # Device-resident monotonic counters: one per flag slot for the
        # publisher role and one for the waiter role.
        self._send_counters = torch.zeros(
            FLAG_SLOTS, dtype=torch.int32, device=self.device
        )
        self._wait_counters = torch.zeros(
            FLAG_SLOTS, dtype=torch.int32, device=self.device
        )
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._flag_stream = torch.cuda.Stream(device=self.device)
        # Separate CE/flag streams for the a2a broadcast phase so allgather
        # traffic overlaps reduce-scatter traffic instead of queueing
        # behind it.
        self._ag_copy_stream = torch.cuda.Stream(device=self.device)
        self._ag_flag_stream = torch.cuda.Stream(device=self.device)
        # Persistent cross-stream events: captured graphs keep references to
        # recorded events, so per-call temporaries must not be destroyed.
        self._piece_events = [torch.cuda.Event() for _ in range(MAX_PIECES)]
        self._copied_events = [
            torch.cuda.Event() for _ in range(2 * (self.world_size - 1) * MAX_PIECES)
        ]
        self._input_ready = torch.cuda.Event()
        self._ag_ready = torch.cuda.Event()
        self._a2a_qdone = [torch.cuda.Event() for _ in range(MAX_PIECES)]
        self._a2a_ownq = [torch.cuda.Event() for _ in range(MAX_PIECES)]
        # Explicit argument wins over the environment so integrations can
        # plumb the mode through their own configuration.
        self._fp8 = _normalize_fp8_mode(fp8) if fp8 is not None else _fp8_mode()
        self._fp8_stage = None
        self._fp8_stage_stride = 0
        if self._fp8:
            max_shard_elems = self.max_bytes // 2 // self.world_size
            stride = _align_up(
                max_shard_elems + max_shard_elems // FP8_QUANT_BLOCK * 4,
                SCRATCH_ALIGN,
            )
            self._fp8_stage = torch.empty(
                self.world_size * stride, dtype=torch.uint8, device=self.device
            )
            self._fp8_stage_stride = stride
        self.min_bytes = 0
        self.wire_mode = f"fp8-{self._fp8}" if self._fp8 else "bf16"
        logger.debug("[PCIe DMA allreduce] wire mode: %s", self.wire_mode)
        if logger.isEnabledFor(logging.DEBUG):
            self._log_peer_copy_bandwidth()

    def _log_peer_copy_bandwidth(self, iters: int = 20) -> None:
        """One-time raw cudaMemcpyAsync bandwidth check, bypassing the ring
        schedule and flag sync entirely, so a slow deployment environment
        shows up here (bandwidth) rather than only in the full ring's
        latency (which would also be sensitive to sync/launch overhead).

        Every rank concurrently writes step 1 of its successor's scratch
        from step 0 of its own; no rank's step 0 (read) or step 1 (write)
        is touched by anyone else, so this measures true full-ring-style
        concurrent peer bandwidth with no self-inflicted read/write race.
        """

        if self.world_size < 2 or 2 * (self.world_size - 1) < 2:
            return
        nxt = (self.rank + 1) % self.world_size
        probe_bytes = min(self.shard_capacity, 4 << 20)
        probe_bytes -= probe_bytes % 16
        if probe_bytes <= 0:
            return
        stream = torch.cuda.Stream(device=self.device)
        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        dist.barrier(group=self.group, device_ids=[device_index])
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._ext.dma_copy(
                    self._scratch_ptr(nxt, 1),
                    self._scratch_ptr(self.rank, 0),
                    probe_bytes,
                )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for _ in range(iters):
                self._ext.dma_copy(
                    self._scratch_ptr(nxt, 1),
                    self._scratch_ptr(self.rank, 0),
                    probe_bytes,
                )
            end.record(stream)
        stream.synchronize()
        ms = start.elapsed_time(end)
        gbps = probe_bytes * iters / (ms * 1e-3) / 1e9
        logger.debug(
            "[PCIe DMA allreduce] rank %d -> %d raw peer copy: %.1f GB/s "
            "(%d bytes x %d iters)",
            self.rank,
            nxt,
            gbps,
            probe_bytes,
            iters,
        )

    def _flag_ptr(self, rank: int, slot: int) -> int:
        return self._flags_base[rank] + slot * FLAG_STRIDE

    def _counter_ptr(self, counters: torch.Tensor, slot: int) -> int:
        return counters.data_ptr() + slot * 4

    def _scratch_ptr(self, rank: int, step: int) -> int:
        return self._scratch_base[rank] + step * self.shard_capacity

    @staticmethod
    def _pick_pieces(shard_elems: int, shard_bytes: int) -> int:
        override = int(os.getenv("FLASHINFER_EXP_SM12X_PCIE_DMA_PIECES", "0"))
        # pieces=2 measured best at every size (deeper chunking pays an
        # extra wait+add launch chain per piece on the main stream).
        candidates = (override,) if 1 <= override <= MAX_PIECES else (2,)
        for pieces in candidates:
            if shard_elems % (pieces * 8) == 0 and shard_bytes // pieces >= 512 << 10:
                return pieces
        return 1

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed or inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        numel = inp.numel()
        if numel <= 0 or numel % (self.world_size * 8) != 0:
            return False
        size_bytes = numel * inp.element_size()
        if size_bytes < self.min_bytes:
            return False
        return inp.is_contiguous() and size_bytes <= self.max_bytes

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy ring allreduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if out is None:
            out = torch.empty_like(inp)
        elif (
            out.shape != inp.shape or out.dtype != inp.dtype or not out.is_contiguous()
        ):
            raise ValueError("output must match input shape/dtype and be contiguous")
        ext = self._ext
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        dtype_code = SUPPORTED_DTYPES[inp.dtype]
        elem = inp.element_size()
        shard_elems = inp.numel() // world
        shard_bytes = shard_elems * elem

        fp8_eligible = (
            bool(self._fp8)
            and inp.dtype == torch.bfloat16
            and shard_elems % FP8_QUANT_BLOCK == 0
        )
        if fp8_eligible and self._fp8 == "a2a":
            return self._all_reduce_fp8(inp, out, shard_elems)
        fp8_ring = fp8_eligible and self._fp8 == "ring"
        fp8_ag = fp8_eligible and self._fp8 in ("ag", "ring")

        base = out.data_ptr()

        # Sub-chunking with a dedicated copy stream keeps the copy engine
        # busy: the CE never waits for a flag round trip or an add because
        # sub-chunk c+1's copy overlaps sub-chunk c's wait+reduce. Deeper
        # chunking amortizes the flag round trip further as long as each
        # piece's copy time dominates the ~5us sub-step overhead.
        pieces = self._pick_pieces(shard_elems, shard_bytes)
        if fp8_ag and (shard_elems // pieces) % FP8_QUANT_BLOCK != 0:
            pieces = 1
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        # fp8 AG slices are piece-contiguous: [payload][scales] per piece.
        piece_slice_bytes = piece_elems + piece_elems // FP8_QUANT_BLOCK * 4
        steps = 2 * (world - 1)

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream

        # No upfront out.copy_(inp): the first send of each chunk reads the
        # caller's input directly and every reduce-scatter add is a first
        # touch (out = inp + scratch), so the accumulation base folds into
        # the add instead of a full-size copy on the critical path.
        in_base = inp.data_ptr()
        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)

        def piece_ptr(chunk: int, piece: int) -> int:
            return base + chunk * shard_bytes + piece * piece_bytes

        def in_piece_ptr(chunk: int, piece: int) -> int:
            return in_base + chunk * shard_bytes + piece * piece_bytes

        def scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_bytes

        def slot(step: int, piece: int) -> int:
            return step * pieces + piece

        # Events gating each step's send on the previous step's reduce of
        # the same payload piece (persistent; re-recorded per step). Flag
        # kernels run on their own stream, gated per copy by copied[] events,
        # so the copy stream is pure back-to-back CE work: an SM kernel
        # between CE ops stalls the engine for the launch round trip, which
        # is what made deeper sub-chunking regress.
        add_done = self._piece_events
        flag_stream = self._flag_stream
        copied = self._copied_events
        flag_stream.wait_event(self._input_ready)

        def fp8_scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_slice_bytes

        stage = self._fp8_stage.data_ptr() if fp8_ag else 0

        def fp8_stage_piece(chunk: int, piece: int) -> int:
            return stage + chunk * self._fp8_stage_stride + piece * piece_slice_bytes

        for k in range(steps):
            reduce_phase = k < world - 1
            if reduce_phase:
                send_chunk = (rank - k) % world
                recv_chunk = (rank - k - 1) % world
            else:
                send_chunk = (rank + 1 - (k - (world - 1))) % world
                recv_chunk = (rank - (k - (world - 1))) % world
            fp8_reduce = fp8_ring and reduce_phase
            fp8_step = fp8_reduce or (fp8_ag and not reduce_phase)
            if fp8_step and k == world - 1:
                # The AG-only mode quantizes the fully reduced owner chunk
                # here.  The FP8 ring's fused final reduce hop already
                # emitted the same payload.  Both modes forward those bytes
                # verbatim, with no additional all-gather rounding.
                if not fp8_ring:
                    for p in range(pieces):
                        ag_stage = fp8_stage_piece(send_chunk, p)
                        ext.dma_quant(
                            piece_ptr(send_chunk, p),
                            ag_stage,
                            ag_stage + piece_elems,
                            piece_elems,
                        )
                # Publish the payload before the local materialization so the
                # CE broadcast can overlap this read-only dequant kernel.
                self._ag_ready.record(main)
                # The owner used to retain its pre-wire BF16 shard while the
                # other ranks materialized this same shard from FP8.  That
                # violates the replicated-output contract of all-reduce and
                # lets the next TP layer consume rank-dependent activations.
                # Round-trip the owner through the exact forwarded payload so
                # all ranks receive bit-identical BF16 values for every shard.
                for p in range(pieces):
                    owner_stage = fp8_stage_piece(send_chunk, p)
                    ext.dma_dequant_store(
                        piece_ptr(send_chunk, p),
                        owner_stage,
                        owner_stage + piece_elems,
                        piece_elems,
                    )
            for p in range(pieces):
                if fp8_reduce:
                    send_src = fp8_stage_piece(send_chunk, p)
                    if k == 0:
                        ext.dma_quant(
                            in_piece_ptr(send_chunk, p),
                            send_src,
                            send_src + piece_elems,
                            piece_elems,
                        )
                        self._a2a_qdone[p].record(main)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                elif not fp8_step:
                    send_src = (
                        in_piece_ptr(send_chunk, p)
                        if k == 0
                        else piece_ptr(send_chunk, p)
                    )
                    send_bytes = piece_bytes
                    send_dst = scratch_piece(nxt, k, p)
                elif k == world - 1:
                    send_src = fp8_stage_piece(send_chunk, p)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                else:
                    send_src = fp8_scratch_piece(rank, k - 1, p)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                with torch.cuda.stream(copy_stream):
                    if fp8_reduce:
                        copy_stream.wait_event(
                            self._a2a_qdone[p] if k == 0 else add_done[p]
                        )
                    elif fp8_step and k == world - 1:
                        copy_stream.wait_event(self._ag_ready)
                    elif k > 0:
                        copy_stream.wait_event(add_done[p])
                    ext.dma_copy(send_dst, send_src, send_bytes)
                    copied[slot(k, p)].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied[slot(k, p)])
                    ext.dma_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                ext.dma_wait_flag(
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                )
                if reduce_phase:
                    if fp8_reduce:
                        payload = fp8_scratch_piece(rank, k, p)
                        reduced = fp8_stage_piece(recv_chunk, p)
                        ext.dma_dequant_add_quant(
                            piece_ptr(recv_chunk, p),
                            in_piece_ptr(recv_chunk, p),
                            payload,
                            payload + piece_elems,
                            reduced,
                            reduced + piece_elems,
                            piece_elems,
                            k == world - 2,
                        )
                    else:
                        ext.dma_add(
                            piece_ptr(recv_chunk, p),
                            in_piece_ptr(recv_chunk, p),
                            scratch_piece(rank, k, p),
                            piece_elems,
                            dtype_code,
                        )
                elif fp8_step:
                    payload = fp8_scratch_piece(rank, k, p)
                    # Forwarding reads the received FP8 payload verbatim, so
                    # it only depends on the receive flag, not on the local
                    # BF16 materialization below.  Publish readiness before
                    # dequantization to overlap the next hop's CE copy with
                    # this rank's read-only dequant/store.
                    add_done[p].record(main)
                    ext.dma_dequant_store(
                        piece_ptr(recv_chunk, p),
                        payload,
                        payload + piece_elems,
                        piece_elems,
                    )
                else:
                    ext.dma_copy(
                        piece_ptr(recv_chunk, p),
                        scratch_piece(rank, k, p),
                        piece_bytes,
                    )
                if reduce_phase or not fp8_step:
                    add_done[p].record(main)

        # Neighbor handshake so the next call (or graph replay) cannot
        # overwrite scratch a lagging neighbor still reads. The main stream
        # must also drain the copy and flag streams before the op is done.
        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = steps * pieces
        ext.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        ext.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )
        return out

    def _pick_a2a_chunks(self, shard_elems: int) -> int:
        override = int(os.getenv("FLASHINFER_EXP_SM12X_PCIE_DMA_A2A_CHUNKS", "0"))
        candidates = (override,) if 1 <= override <= MAX_PIECES else (4, 3, 2)
        for chunks in candidates:
            if (
                shard_elems % (chunks * FP8_QUANT_BLOCK) == 0
                and shard_elems // chunks >= 384 << 10
            ):
                return chunks
        return 1

    def _all_reduce_fp8(
        self, inp: torch.Tensor, out: torch.Tensor, shard_elems: int
    ) -> torch.Tensor:
        """Pipelined quantize-once E4M3 all-to-all.

        Slices are split into chunks; each chunk's quantize -> scatter ->
        fp32 dequant-accumulate -> quantize-once broadcast wave overlaps the
        next chunk's, with broadcast copies on their own CE stream so the
        two phases' wire time overlaps rather than queues.

        No end handshake is needed: a rank re-enters the op only after its
        stream finished, which required every peer's broadcast of every
        chunk and therefore every peer's accumulate and placement; peers'
        next-call writes are stream-ordered after that.
        """

        ext = self._ext
        world = self.world_size
        rank = self.rank
        shard_bytes = shard_elems * 2
        chunks = self._pick_a2a_chunks(shard_elems)
        chunk_elems = shard_elems // chunks
        chunk_bytes = chunk_elems * 2
        chunk_payload = chunk_elems
        chunk_slice = chunk_payload + chunk_elems // FP8_QUANT_BLOCK * 4
        in_base = inp.data_ptr()
        out_base = out.data_ptr()
        stage_base = self._fp8_stage.data_ptr()
        stride = self._fp8_stage_stride

        def stage_chunk(shard: int, c: int) -> int:
            return stage_base + shard * stride + c * chunk_slice

        def rs_chunk(owner: int, srcpos: int, c: int) -> int:
            return self._scratch_ptr(owner, srcpos) + c * chunk_slice

        def ag_chunk(owner: int, srcpos: int, c: int) -> int:
            return self._scratch_ptr(owner, (world - 1) + srcpos) + c * chunk_slice

        def rs_slot(srcpos: int, c: int) -> int:
            return srcpos * chunks + c

        def ag_slot(srcpos: int, c: int) -> int:
            return (world - 1) * chunks + srcpos * chunks + c

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        ag_copy = self._ag_copy_stream
        ag_flag = self._ag_flag_stream
        copied = self._copied_events
        half = len(copied) // 2
        peers = [(rank + 1 + i) % world for i in range(world - 1)]
        pos_at = [(rank - j - 1) % world for j in peers]

        # Quantize all outgoing chunks up front (cheap kernels on main);
        # per-chunk events let the scatter start as soon as its chunk is
        # ready while later quants still run.
        for c in range(chunks):
            for j in peers:
                ext.dma_quant(
                    in_base + j * shard_bytes + c * chunk_bytes,
                    stage_chunk(j, c),
                    stage_chunk(j, c) + chunk_payload,
                    chunk_elems,
                )
            self._a2a_qdone[c].record(main)

        # Scatter: reduce-scatter slices, chunk-pipelined.
        for c in range(chunks):
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(self._a2a_qdone[c])
                for i, j in enumerate(peers):
                    ext.dma_copy(
                        rs_chunk(j, pos_at[i], c), stage_chunk(j, c), chunk_slice
                    )
                    copied[i * chunks + c].record(copy_stream)
            with torch.cuda.stream(flag_stream):
                for i, j in enumerate(peers):
                    flag_stream.wait_event(copied[i * chunks + c])
                    slot = rs_slot(pos_at[i], c)
                    ext.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Accumulate own shard chunk by chunk; broadcast each chunk as soon
        # as it is reduced and quantized (once).
        for c in range(chunks):
            for i in range(world - 1):
                slot = rs_slot(i, c)
                ext.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
            payloads = [rs_chunk(rank, i, c) for i in range(world - 1)]
            scales = [ptr + chunk_payload for ptr in payloads]
            own = rank * shard_bytes + c * chunk_bytes
            ext.dma_dequant_accum(
                out_base + own, in_base + own, payloads, scales, chunk_elems
            )
            ext.dma_quant(
                out_base + own,
                stage_chunk(rank, c),
                stage_chunk(rank, c) + chunk_payload,
                chunk_elems,
            )
            # Publish the payload first so its CE broadcast can overlap the
            # local read-only materialization below.
            self._a2a_ownq[c].record(main)
            # Peers materialize this reduced shard from the broadcast FP8
            # payload.  The owner must do the same or every rank enters the
            # next TP layer with a different replicated activation.
            ext.dma_dequant_store(
                out_base + own,
                stage_chunk(rank, c),
                stage_chunk(rank, c) + chunk_payload,
                chunk_elems,
            )
            with torch.cuda.stream(ag_copy):
                ag_copy.wait_event(self._a2a_ownq[c])
                for i, j in enumerate(peers):
                    ext.dma_copy(
                        ag_chunk(j, pos_at[i], c), stage_chunk(rank, c), chunk_slice
                    )
                    copied[half + i * chunks + c].record(ag_copy)
            with torch.cuda.stream(ag_flag):
                for i, j in enumerate(peers):
                    ag_flag.wait_event(copied[half + i * chunks + c])
                    slot = ag_slot(pos_at[i], c)
                    ext.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Place incoming reduced shards.
        for c in range(chunks):
            for i in range(world - 1):
                src = peers[i]
                slot = ag_slot(i, c)
                ext.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
                payload = ag_chunk(rank, i, c)
                ext.dma_dequant_store(
                    out_base + src * shard_bytes + c * chunk_bytes,
                    payload,
                    payload + chunk_payload,
                    chunk_elems,
                )
        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        main.wait_stream(ag_copy)
        main.wait_stream(ag_flag)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for ptr in self._slab.remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(ptr)
        with suppress(Exception):
            self._ipc.cudaFree(self._slab.local_ptr)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def autotune_crossovers(
    oneshot,
    dma: Optional[PCIeDmaAllReduce],
    nccl_group: ProcessGroup,
    *,
    hidden_size: int,
    max_rows: int,
    rms_norm_op=None,
    epsilon: float = 1e-6,
    warmup: int = 5,
    iters: int = 50,
    samples: int = 5,
    win_margin: float = 0.02,
) -> tuple[int, int]:
    """Single sweep from 1 row to the prefill chunk size with the real
    kernels: the oneshot channel (fused AR+RMSNorm when ``rms_norm_op`` is
    given, plain otherwise), the CE ring, and NCCL (plus ``rms_norm_op``)
    as the fallback. Returns (oneshot_max_bytes, dma_min_bytes) and sets
    ``dma.min_bytes``. Each timing is the median of multiple CUDA-event
    samples after MAX-reducing every sample across ranks. A backend must win
    by ``win_margin`` and DMA must do so at two consecutive sizes before its
    crossover is committed.
    """

    device = oneshot.device if oneshot is not None else dma.device
    stream = torch.cuda.Stream(device=device)
    dtype = torch.bfloat16
    weight = torch.ones(hidden_size, dtype=dtype, device=device)
    inf = float("inf")
    oneshot_max = 0
    dma_min = 0
    if dma is not None:
        original_dma_min = dma.min_bytes
        dma.min_bytes = 0
    wire = "bf16" if dma is None else dma.wire_mode
    lines = [
        f"[PCIe allreduce] Crossover sweep (dma wire={wire}, "
        f"hidden={hidden_size}, fused={rms_norm_op is not None}):"
    ]

    def bench(build) -> float:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            replay = build()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            replay()
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        dist.barrier(group=nccl_group, device_ids=[device_index])
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                graph.replay()
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        rank_max = torch.empty((), dtype=torch.float64, device=device)
        timings = []
        for _ in range(samples):
            dist.barrier(group=nccl_group, device_ids=[device_index])
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(iters):
                    graph.replay()
                end.record(stream)
            end.synchronize()
            rank_max.fill_(start.elapsed_time(end) * 1e3 / iters)
            dist.all_reduce(rank_max, op=dist.ReduceOp.MAX, group=nccl_group)
            timings.append(float(rank_max.item()))
        return float(median(timings))

    try:
        # Fully dense through 8 rows (the decode regime, where every row
        # count occurs), quarter steps through 32-128 rows (where the
        # NCCL/DMA boundary lives), and powers of two with midpoints
        # elsewhere. The sweep stops once the DMA allreduce has won twice
        # in a row: the curves are monotone above the boundary and the
        # large probes are the expensive ones.
        ladder = list(range(1, min(8, max_rows) + 1))
        step = 8
        while step <= max_rows:
            if step not in ladder:
                ladder.append(step)
            if 32 <= step <= 64:
                extra = (step + step // 4, step + step // 2, step + 3 * step // 4)
            else:
                extra = (step + step // 2,)
            ladder.extend(rows for rows in extra if rows <= max_rows)
            step *= 2
        oneshot_losses = 0
        dma_wins = 0
        dma_candidate = 0
        rank0 = dist.get_rank(group=nccl_group) == 0
        if rank0:
            logger.debug(lines[0])
        for rows in ladder:
            point_start = time.perf_counter()
            shape = (rows, hidden_size)
            size_bytes = rows * hidden_size * dtype.itemsize

            def build_nccl():
                inp = torch.randn(shape, dtype=dtype, device=device) * 0.01
                residual = torch.randn(shape, dtype=dtype, device=device)
                if rms_norm_op is None:
                    return lambda: dist.all_reduce(inp, group=nccl_group)
                return lambda: (
                    dist.all_reduce(inp, group=nccl_group),
                    rms_norm_op(inp, residual, weight, epsilon),
                )

            nccl_us = bench(build_nccl)

            # Stop probing the oneshot after it has clearly lost (its curve
            # is monotone against NCCL); a probe the kernel refuses (row or
            # capacity limits) counts as a loss. Every rank takes the same
            # branch because verdicts come from MAX-reduced timings.
            oneshot_us = inf
            if (
                oneshot is not None
                and oneshot_losses < 2
                and size_bytes <= oneshot.max_size
            ):

                def build_oneshot():
                    inp = torch.randn(shape, dtype=dtype, device=device) * 0.01
                    residual = torch.randn(shape, dtype=dtype, device=device)
                    out = torch.empty_like(inp)
                    residual_out = torch.empty_like(inp)
                    if rms_norm_op is None:
                        return lambda: oneshot.all_reduce(inp, out=out)
                    return lambda: oneshot.all_reduce_fused_add_rms_norm(
                        inp,
                        residual,
                        weight,
                        epsilon,
                        out=out,
                        residual_out=residual_out,
                    )

                try:
                    oneshot_us = bench(build_oneshot)
                except Exception:
                    oneshot_us = inf

            dma_us = inf
            probe = torch.empty(shape, dtype=dtype, device=device)
            if dma is not None and dma.should_allreduce(probe):

                def build_dma():
                    inp = torch.randn(shape, dtype=dtype, device=device) * 0.01
                    out = torch.empty_like(inp)
                    return lambda: dma.all_reduce(inp, out=out)

                dma_us = bench(build_dma)
            del probe

            stats = torch.tensor(
                [nccl_us, oneshot_us, dma_us], dtype=torch.float64, device=device
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=nccl_group)
            nccl_us, oneshot_us, dma_us = (float(v) for v in stats.tolist())
            oneshot_limit = (1.0 + win_margin) * min(nccl_us, dma_us)
            if oneshot_us < oneshot_limit:
                oneshot_max = size_bytes
                oneshot_losses = 0
            else:
                oneshot_losses += 1
            dma_limit = (1.0 - win_margin) * min(nccl_us, oneshot_us)
            if dma_us < dma_limit:
                if dma_wins == 0:
                    dma_candidate = size_bytes
                dma_wins += 1
            else:
                dma_wins = 0
                dma_candidate = 0
            line = (
                f"  rows={rows:5d} ({size_bytes >> 10:6d}KB): "
                f"oneshot {oneshot_us:9.1f}  dma {dma_us:9.1f}  "
                f"nccl {nccl_us:9.1f} us"
                f"  [{time.perf_counter() - point_start:.2f}s]"
            )
            lines.append(line)
            if rank0:
                logger.debug(line)
            if dma_wins >= 2:
                dma_min = dma_candidate
                break
    except Exception:
        if dma is not None:
            dma.min_bytes = original_dma_min
        raise

    if dma is not None:
        dma.min_bytes = dma_min if dma_min > 0 else dma.max_bytes + 1
    if dist.get_rank(group=nccl_group) == 0:
        logger.debug("  oneshot_max_bytes=%d dma_min_bytes=%d", oneshot_max, dma_min)
    return oneshot_max, dma_min


__all__ = ["PCIeDmaAllReduce", "autotune_crossovers"]
