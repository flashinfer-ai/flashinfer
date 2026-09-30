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
from contextlib import suppress
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._dma_kernels import DmaKernels
from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import PCIeOneshotAllReduce, _normalize_device

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}
SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)
FLAG_STRIDE = 128
FLAG_SLOTS = 256
MAX_PIECES = 8
SCRATCH_ALIGN = 256
FP8_QUANT_BLOCK = 128


def _eager_replay_capacities(
    max_bytes: int,
    min_bytes: int,
    itemsize: int,
    world_size: int,
    *,
    max_elements: int | None = None,
) -> tuple[int, ...]:
    """Fixed element capacities with eight-value shards on every supported ring."""
    multiple = world_size * 8
    elements = max_bytes // itemsize
    if max_elements is not None:
        if max_elements <= 0 or max_elements > elements:
            raise ValueError("DMA dtype capacity must fit the positive ring byte bound")
        elements = max_elements
    elements -= elements % multiple
    if elements <= 0:
        raise ValueError("DMA replay capacity is too small")
    size_bytes = 1 << (max(1 << 20, min_bytes) - 1).bit_length()
    capacities = []
    while size_bytes < elements * itemsize:
        capacity = size_bytes // itemsize
        capacity -= capacity % multiple
        if capacity > 0:
            capacities.append(capacity)
        size_bytes *= 2
    return (*capacities, elements)


def _fp8_mode() -> str:
    """Opt-in compressed wire transport mode.

    "ag" (also "1"): keep the saturated bf16 reduce-scatter ring and
    quantize only the allgather phase. Final values quantize exactly once
    at their owner and are forwarded verbatim around the ring, so the
    error cost is a single rounding while AG wire bytes halve.

    "ring": quantize every reduce-scatter hop and the allgather payload,
    keeping the saturated neighbor-only topology while halving both phases.

    "a2a": quantize-once all-to-all (two roundings, half the wire in both
    phases).

    "i8" / "i8_ring" / "i8_a2a": the matching topology with a symmetric
    signed-INT8 payload. Both codecs use the same layout: one payload byte
    per value plus one fp32 scale per 128 values.

    "mx" / "mx_ring" / "mx_a2a": standard MXFP8, with E4M3 payload values
    and one E8M0 scale per 32 values. Four scale bytes per 128 values retain
    the same wire footprint as the other compressed codecs.

    Every compressed mode materializes the locally owned reduced shard through
    the same wire payload as its peers. An all-reduce result must be
    rank-identical; retaining a pre-wire BF16 owner shard while peers dequantize
    that shard gives every TP rank a different replicated activation.
    """

    return _normalize_fp8_mode(os.getenv("B12X_PCIE_DMA_FP8", "0"))


def _normalize_fp8_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return ""
    if raw in ("a2a", "ring"):
        return raw
    if raw in (
        "i8",
        "int8",
        "i8_ag",
        "i8-ag",
        "ag_i8",
        "int8_ag",
        "int8-ag",
    ):
        return "i8"
    if raw in ("i8_ring", "i8-ring", "int8_ring", "int8-ring", "ring_i8"):
        return "i8_ring"
    if raw in ("i8_a2a", "i8-a2a", "int8_a2a", "int8-a2a", "a2a_i8"):
        return "i8_a2a"
    if raw in (
        "mx",
        "mxfp8",
        "mx_ag",
        "mx-ag",
        "mxfp8_ag",
        "mxfp8-ag",
        "ag_mx",
    ):
        return "mx"
    if raw in (
        "mx_ring",
        "mx-ring",
        "mxfp8_ring",
        "mxfp8-ring",
        "ring_mx",
    ):
        return "mx_ring"
    if raw in (
        "mx_a2a",
        "mx-a2a",
        "mxfp8_a2a",
        "mxfp8-a2a",
        "a2a_mx",
    ):
        return "mx_a2a"
    if raw in ("ag", "1"):
        return "ag"
    raise ValueError(f"unrecognized PCIe DMA wire mode: {value!r}")


def _load_kernels(ipc: CudaRTLibrary) -> DmaKernels:
    """Construct an unprepared transport façade for this IPC channel."""

    return DmaKernels(ipc)


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
        # Kept only as a source-compatible keyword for callers that used to
        # inject the removed C++ extension.  Device work is always CuTe DSL.
        del ext_module
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "PCIe DMA all-reduce supports only the reviewed world sizes "
                f"{SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("PCIe ring allreduce requires a CUDA device")
        self.max_bytes = int(max_bytes)
        self._ipc = CudaRTLibrary()
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._kernels = _load_kernels(self._ipc)
        self._closed = False
        self._eager_replays: dict[
            torch.dtype,
            tuple[tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph], ...],
        ] = {}

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
        self._pieces_override = int(os.getenv("B12X_PCIE_DMA_PIECES", "0"))
        self._a2a_chunks_override = int(os.getenv("B12X_PCIE_DMA_A2A_CHUNKS", "0"))
        self.min_bytes = 0
        wire_modes = {
            "i8": "int8-ag",
            "i8_ring": "int8-ring",
            "i8_a2a": "int8-a2a",
            "mx": "mxfp8-ag",
            "mx_ring": "mxfp8-ring",
            "mx_a2a": "mxfp8-a2a",
        }
        self.wire_mode = wire_modes.get(
            self._fp8, f"fp8-{self._fp8}" if self._fp8 else "bf16"
        )
        logger.debug("[PCIe DMA allreduce] wire mode: %s", self.wire_mode)

    def _flag_ptr(self, rank: int, slot: int) -> int:
        return self._flags_base[rank] + slot * FLAG_STRIDE

    def _counter_ptr(self, counters: torch.Tensor, slot: int) -> int:
        return counters.data_ptr() + slot * 4

    def _scratch_ptr(self, rank: int, step: int) -> int:
        return self._scratch_base[rank] + step * self.shard_capacity

    def _pick_pieces(self, shard_elems: int, shard_bytes: int) -> int:
        override = self._pieces_override
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

    def prepare_eager_replay(
        self, dtype: torch.dtype, *, max_elements: int | None = None
    ) -> None:
        """Precapture bounded-capacity lossless rings for later eager calls.

        Call collectively before serving. Powers-of-two capacities, capped at
        max_bytes, bound padding traffic without capturing on a request path.
        Graphs borrow one private input/output allocation per dtype. Live sizes
        select a covering graph, not a compilation or allocation cache entry.
        The private output cannot alias a retained result.
        max_elements bounds this dtype independently of the ring byte capacity,
        so preparing FP32 reductions does not double BF16 replay storage.
        """
        if self._closed or self._fp8:
            raise ValueError("eager replay requires an open lossless DMA ring")
        if dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"unsupported DMA replay dtype: {dtype}")
        if max_elements is None:
            elements = self.max_bytes // dtype.itemsize
        else:
            if isinstance(max_elements, bool) or not isinstance(max_elements, int):
                raise TypeError("DMA replay max_elements must be an integer")
            if max_elements * dtype.itemsize > self.max_bytes:
                raise ValueError(
                    f"DMA replay max_elements={max_elements} for {dtype} "
                    f"exceeds max_bytes={self.max_bytes}"
                )
            elements = max_elements
        alignment = self.world_size * 8
        elements -= elements % alignment
        if elements <= 0:
            raise ValueError(
                f"DMA replay capacity must contain at least {alignment} elements"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare DMA eager replay before CUDA graph capture")
        capacities = _eager_replay_capacities(
            self.max_bytes,
            self.min_bytes,
            dtype.itemsize,
            self.world_size,
            max_elements=max_elements,
        )
        if dtype in self._eager_replays:
            if capacities[-1] != self._eager_replays[dtype][-1][0].numel():
                raise ValueError(
                    "DMA dtype replay capacity cannot change after preparation"
                )
            return
        elements = capacities[-1]
        with torch.cuda.device(self.device):
            source = torch.zeros(elements, dtype=dtype, device=self.device)
            result = torch.empty_like(source)
            replays = []
            for capacity in capacities:
                source_view, result_view = source[:capacity], result[:capacity]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._all_reduce_on_device(source_view, out=result_view)
                replays.append((source_view, result_view, graph))
        self._eager_replays[dtype] = tuple(replays)

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run only the already-prepared fixed DMA channel."""

        state = require_prepared(plan, "comm.pcie", self.device)
        if not hasattr(state, "require_runtime") or not hasattr(state, "run"):
            raise TypeError("plan does not contain a prepared DMA channel")
        state.require_runtime(self)
        if out is None:
            out = torch.empty_like(inp)
        return state.run(inp, out=out)
    def _run_prepared(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor], state
    ) -> torch.Tensor:
        state.require_runtime(self)
        torch.cuda.set_device(self.device)
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy ring allreduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if out is None:
            raise ValueError("prepared DMA all_reduce requires a caller-owned output tensor")
        if (
            out.shape != inp.shape
            or out.dtype != inp.dtype
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "output must match input shape/dtype/device and be contiguous"
            )
        replays = self._eager_replays.get(inp.dtype)
        if replays is not None and not torch.cuda.is_current_stream_capturing():
            elements = inp.numel()
            replay = next(
                (item for item in replays if item[0].numel() >= elements), None
            )
            if replay is not None:
                source, result, graph = replay
                source[:elements].copy_(inp.view(-1))
                graph.replay()
                out.view(-1).copy_(result[:elements])
                return out
        kernels = self._kernels
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        dtype_code = SUPPORTED_DTYPES[inp.dtype]
        elem = inp.element_size()
        shard_elems = inp.numel() // world
        shard_bytes = shard_elems * elem

        compressed_eligible = (
            bool(self._fp8)
            and inp.dtype == torch.bfloat16
            and shard_elems % FP8_QUANT_BLOCK == 0
        )
        int8_wire = compressed_eligible and self._fp8.startswith("i8")
        mxfp8_wire = compressed_eligible and self._fp8.startswith("mx")
        wire_codec = "i8" if int8_wire else "mx" if mxfp8_wire else "e4m3"
        if compressed_eligible and self._fp8 in ("a2a", "i8_a2a", "mx_a2a"):
            return self._all_reduce_fp8(inp, out, shard_elems, wire_codec=wire_codec)
        compressed_ring = compressed_eligible and self._fp8 in (
            "ring",
            "i8_ring",
            "mx_ring",
        )
        compressed_ag = compressed_eligible and self._fp8 in (
            "ag",
            "ring",
            "i8",
            "i8_ring",
            "mx",
            "mx_ring",
        )
        if wire_codec == "i8":
            quantize = kernels.dma_quant_i8
            dequantize_store = kernels.dma_dequant_store_i8
            dequantize_add_quant = kernels.dma_dequant_add_quant_i8
        elif wire_codec == "mx":
            quantize = kernels.dma_quant_mx
            dequantize_store = kernels.dma_dequant_store_mx
            dequantize_add_quant = kernels.dma_dequant_add_quant_mx
        else:
            quantize = kernels.dma_quant
            dequantize_store = kernels.dma_dequant_store
            dequantize_add_quant = kernels.dma_dequant_add_quant

        base = out.data_ptr()

        # Sub-chunking with a dedicated copy stream keeps the copy engine
        # busy: the CE never waits for a flag round trip or an add because
        # sub-chunk c+1's copy overlaps sub-chunk c's wait+reduce. Deeper
        # chunking amortizes the flag round trip further as long as each
        # piece's copy time dominates the ~5us sub-step overhead.
        pieces = self._pick_pieces(shard_elems, shard_bytes)
        if compressed_ag and (shard_elems // pieces) % FP8_QUANT_BLOCK != 0:
            pieces = 1
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        # Compressed slices are piece-contiguous: [payload][scales] per piece.
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

        stage = self._fp8_stage.data_ptr() if compressed_ag else 0

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
            compressed_reduce = compressed_ring and reduce_phase
            compressed_step = compressed_reduce or (compressed_ag and not reduce_phase)
            if compressed_step and k == world - 1:
                # The AG-only mode quantizes the fully reduced owner chunk
                # here.  The FP8 ring's fused final reduce hop already
                # emitted the same payload.  Both modes forward those bytes
                # verbatim, with no additional all-gather rounding.
                if not compressed_ring:
                    for p in range(pieces):
                        ag_stage = fp8_stage_piece(send_chunk, p)
                        quantize(
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
                    dequantize_store(
                        piece_ptr(send_chunk, p),
                        owner_stage,
                        owner_stage + piece_elems,
                        piece_elems,
                    )
            for p in range(pieces):
                if compressed_reduce:
                    send_src = fp8_stage_piece(send_chunk, p)
                    if k == 0:
                        quantize(
                            in_piece_ptr(send_chunk, p),
                            send_src,
                            send_src + piece_elems,
                            piece_elems,
                        )
                        self._a2a_qdone[p].record(main)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                elif not compressed_step:
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
                    if compressed_reduce:
                        copy_stream.wait_event(
                            self._a2a_qdone[p] if k == 0 else add_done[p]
                        )
                    elif compressed_step and k == world - 1:
                        copy_stream.wait_event(self._ag_ready)
                    elif k > 0:
                        copy_stream.wait_event(add_done[p])
                    kernels.dma_copy(send_dst, send_src, send_bytes)
                    copied[slot(k, p)].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied[slot(k, p)])
                    kernels.dma_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                )
                if reduce_phase:
                    if compressed_reduce:
                        payload = fp8_scratch_piece(rank, k, p)
                        reduced = fp8_stage_piece(recv_chunk, p)
                        dequantize_add_quant(
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
                        kernels.dma_add(
                            piece_ptr(recv_chunk, p),
                            in_piece_ptr(recv_chunk, p),
                            scratch_piece(rank, k, p),
                            piece_elems,
                            dtype_code,
                        )
                elif compressed_step:
                    payload = fp8_scratch_piece(rank, k, p)
                    # Forwarding reads the received FP8 payload verbatim, so
                    # it only depends on the receive flag, not on the local
                    # BF16 materialization below.  Publish readiness before
                    # dequantization to overlap the next hop's CE copy with
                    # this rank's read-only dequant/store.
                    add_done[p].record(main)
                    dequantize_store(
                        piece_ptr(recv_chunk, p),
                        payload,
                        payload + piece_elems,
                        piece_elems,
                    )
                else:
                    kernels.dma_copy(
                        piece_ptr(recv_chunk, p),
                        scratch_piece(rank, k, p),
                        piece_bytes,
                    )
                if reduce_phase or not compressed_step:
                    add_done[p].record(main)

        # Neighbor handshake so the next call (or graph replay) cannot
        # overwrite scratch a lagging neighbor still reads. The main stream
        # must also drain the copy and flag streams before the op is done.
        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = steps * pieces
        kernels.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        kernels.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )
        return out

    def _prime_prepared(self, state) -> None:
        """Prime selected code using this channel's fixed IPC buffers only."""

        state.require_runtime(self)
        self._kernels.install(state.launchers)
        with torch.cuda.device(self.device):
            # Local probes must not overwrite an arriving peer flag or payload.
            # The final slot is beyond both transport layouts for supported worlds.
            slot = FLAG_SLOTS - 1
            flag = self._flag_ptr(self.rank, slot)
            self._kernels.dma_set_flag(flag, self._counter_ptr(self._send_counters, slot))
            self._kernels.dma_wait_flag(flag, self._counter_ptr(self._wait_counters, slot))
            scratch = flag + 16
            self._kernels.dma_copy(scratch, scratch, 16)
            for dtype_code, elems in ((0, 8), (1, 8), (2, 4)):
                self._kernels.dma_add(scratch, scratch, scratch, elems, dtype_code)
            if self._fp8:
                if self._fp8_stage is None or self._fp8_stage.numel() < 512:
                    raise RuntimeError("compressed DMA stage buffer is too small to prime")
                codec = (
                    "i8" if self._fp8.startswith("i8")
                    else "mx" if self._fp8.startswith("mx") else "e4m3"
                )
                stage = self._fp8_stage.data_ptr()
                source, payload, scales = stage, stage + 256, stage + 384
                self._kernels._quant(codec, source, payload, scales, FP8_QUANT_BLOCK)
                self._kernels._dequant_store(codec, source, payload, scales, FP8_QUANT_BLOCK)
                self._kernels._dequant_add_quant(
                    codec, source, source, payload, scales, payload, scales,
                    FP8_QUANT_BLOCK, False,
                )
                self._kernels._dequant_add_quant(
                    codec, source, source, payload, scales, payload, scales,
                    FP8_QUANT_BLOCK, True,
                )
                if self._fp8 == "a2a" or self._fp8.endswith("a2a"):
                    sources = [payload] * (self.world_size - 1)
                    self._kernels._dequant_accum(
                        codec, source, source, sources, [scales] * len(sources),
                        FP8_QUANT_BLOCK,
                    )
            torch.cuda.current_stream(self.device).synchronize()

    def _pick_a2a_chunks(self, shard_elems: int) -> int:
        override = self._a2a_chunks_override
        candidates = (override,) if 1 <= override <= MAX_PIECES else (4, 3, 2)
        for chunks in candidates:
            if (
                shard_elems % (chunks * FP8_QUANT_BLOCK) == 0
                and shard_elems // chunks >= 384 << 10
            ):
                return chunks
        return 1

    def _all_reduce_fp8(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        shard_elems: int,
        *,
        wire_codec: str = "e4m3",
    ) -> torch.Tensor:
        """Pipelined quantize-once compressed all-to-all.

        Slices are split into chunks; each chunk's quantize -> scatter ->
        fp32 dequant-accumulate -> quantize-once broadcast wave overlaps the
        next chunk's, with broadcast copies on their own CE stream so the
        two phases' wire time overlaps rather than queues.

        No end handshake is needed: a rank re-enters the op only after its
        stream finished, which required every peer's broadcast of every
        chunk and therefore every peer's accumulate and placement; peers'
        next-call writes are stream-ordered after that.
        """

        kernels = self._kernels
        if wire_codec == "i8":
            quantize = kernels.dma_quant_i8
            dequantize_accum = kernels.dma_dequant_accum_i8
            dequantize_store = kernels.dma_dequant_store_i8
        elif wire_codec == "mx":
            quantize = kernels.dma_quant_mx
            dequantize_accum = kernels.dma_dequant_accum_mx
            dequantize_store = kernels.dma_dequant_store_mx
        else:
            quantize = kernels.dma_quant
            dequantize_accum = kernels.dma_dequant_accum
            dequantize_store = kernels.dma_dequant_store
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
                quantize(
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
                    kernels.dma_copy(
                        rs_chunk(j, pos_at[i], c), stage_chunk(j, c), chunk_slice
                    )
                    copied[i * chunks + c].record(copy_stream)
            with torch.cuda.stream(flag_stream):
                for i, j in enumerate(peers):
                    flag_stream.wait_event(copied[i * chunks + c])
                    slot = rs_slot(pos_at[i], c)
                    kernels.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Accumulate own shard chunk by chunk; broadcast each chunk as soon
        # as it is reduced and quantized (once).
        for c in range(chunks):
            for i in range(world - 1):
                slot = rs_slot(i, c)
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
            payloads = [rs_chunk(rank, i, c) for i in range(world - 1)]
            scales = [ptr + chunk_payload for ptr in payloads]
            own = rank * shard_bytes + c * chunk_bytes
            dequantize_accum(
                out_base + own, in_base + own, payloads, scales, chunk_elems
            )
            quantize(
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
            dequantize_store(
                out_base + own,
                stage_chunk(rank, c),
                stage_chunk(rank, c) + chunk_payload,
                chunk_elems,
            )
            with torch.cuda.stream(ag_copy):
                ag_copy.wait_event(self._a2a_ownq[c])
                for i, j in enumerate(peers):
                    kernels.dma_copy(
                        ag_chunk(j, pos_at[i], c), stage_chunk(rank, c), chunk_slice
                    )
                    copied[half + i * chunks + c].record(ag_copy)
            with torch.cuda.stream(ag_flag):
                for i, j in enumerate(peers):
                    ag_flag.wait_event(copied[half + i * chunks + c])
                    slot = ag_slot(pos_at[i], c)
                    kernels.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Place incoming reduced shards.
        for c in range(chunks):
            for i in range(world - 1):
                src = peers[i]
                slot = ag_slot(i, c)
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
                payload = ag_chunk(rank, i, c)
                dequantize_store(
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
        # Drain the main and four auxiliary streams before any rank unmaps a
        # peer allocation.  Every importer must unmap before its owner frees
        # the exported slab.
        torch.cuda.synchronize(self.device)
        self._eager_replays.clear()
        dist.barrier(group=self.group)
        for ptr in self._slab.remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(ptr)
        dist.barrier(group=self.group)
        with suppress(Exception):
            self._ipc.cudaFree(self._slab.local_ptr)
        dist.barrier(group=self.group)

    def __enter__(self) -> "PCIeDmaAllReduce":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        # Distributed barriers are unsafe during asymmetric interpreter
        # teardown. Explicit/context-manager close owns coordinated release.
        return None

__all__ = ["PCIeDmaAllReduce"]
