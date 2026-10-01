"""Kernel 2, clustered register-resident top-k: one row across a cluster of CTAs (16K to 128K).

Kernel 1 with the row split into slices that each fit a CTA's registers.  Every CTA loads its
slice once, counts its coarse histogram, and after one cluster barrier every CTA merges all
peers' histograms over DSMEM (redundantly, so the crossing needs no broadcast).  Each CTA then
classifies its own slice from registers; winners go straight to the output and ties to rank
0's tie stage, at the offsets given by the lower-ranked peers' histograms (the same remote
loads as the crossing; v0.1.23, no remote atomic).  After a second cluster barrier rank 0
selects the ties.  This is gvr_2's reg_clus family: a batch with SMs to spare
gives each row a cluster's worth of registers and reads the row exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

try:
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # pragma: no cover
    from cutlass.utils.smem_allocator import SmemAllocator
try:
    from cuda.bindings import driver as cuda_driver
except ImportError:  # pragma: no cover
    from cuda import cuda as cuda_driver

from ...device.cluster import (
    cluster_rank,
    cluster_sync,
    peer_max_u32,
    peer_min_u32,
    peer_shared_address,
)
from ...device.launch import release_dependent_grid, wait_for_prior_grid
from ...device.timers import read_clock64

from ..phases.census import COARSE_BINS
from ..phases.cluster_crossing import cluster_crossing, summarize_groups_256
from ..phases.elements import Elements
from ..phases.fallback import radix_select_in_range
from ..phases.register_row import (
    classify_from_registers_cluster,
    count_coarse_bins,
    key_range_in_bin,
    load_row_words,
    zero_bins,
)
from ..phases.resolve import _select_ties
from ..phases.varlen import effective_length, gather_values
from .layout import arena_view, check_layout
from .register_resident import _no_values

__all__ = [
    "RegisterClusterConfig",
    "RegisterClusterTopK",
    "topk_register_cluster",
    "register_cluster_config_for",
    "STATUS_WORDS",
]

_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
STATUS_WORDS = 8  # [0] fallback taken, [1] arm (0 identity, 2 cluster, 3 radix), [2] ties, [3..6] phase clocks


@dataclass(frozen=True)
class RegisterClusterConfig:
    threads: int = 1024
    words_per_thread: int = 16
    splits: int = 2  # CTAs per row: 2, 4 or 8
    bins: int = 4096
    tie_capacity: int = 2048
    ballot_limit: int = 128
    pdl: bool = False
    telemetry: bool = False

    def slice_capacity(self, elems: Elements) -> int:
        return self.threads * self.words_per_thread * elems.per_word

    @property
    def bin_shift(self) -> int:
        """Right shift from a census bin (4096) to this histogram's bin."""
        return {1024: 2, 2048: 1, 4096: 0}[self.bins]

    def validate(self, k: int, elems: Elements, shared_memory_limit: int) -> None:
        if self.threads not in (256, 512, 1024) or self.words_per_thread not in (
            4,
            8,
            16,
        ):
            raise ValueError("threads must be 256/512/1024 and words_per_thread 4/8/16")
        if self.splits not in (2, 4, 8):
            raise ValueError(f"splits must be 2, 4 or 8, got {self.splits}")
        if self.bins not in (1024, 2048, 4096):
            raise ValueError(f"bins must be 1024, 2048 or 4096, got {self.bins}")
        if self.threads < 256:
            raise ValueError(
                "the two-level cluster crossing needs at least 256 threads"
            )
        # the tie stage need not hold k: an overflowing crossing bin takes the radix select over
        # the bin's key range on rank 0, exact for any k
        if (
            self.tie_capacity < 2 * self.threads
            or self.tie_capacity > 8 * self.threads
            or self.tie_capacity % self.threads
        ):
            raise ValueError(
                "tie_capacity must be a multiple of threads in [2 * threads, 8 * threads]"
            )
        if self.shared_memory_bytes() > shared_memory_limit:
            raise ValueError(
                f"{self.shared_memory_bytes()} B exceeds the {shared_memory_limit} B budget"
            )

    def shared_memory_bytes(self) -> int:
        b = max(self.bins, COARSE_BINS) + 4
        return (
            b * 4
            + (256 + 256 + 16) * 4
            + 2 * self.tie_capacity * 4
            + (self.threads // 32 + 32) * 4
            + 512
        )

    def name(self) -> str:
        return "_".join(f"{f.name}{getattr(self, f.name)}" for f in fields(self))


class RegisterClusterTopK:
    def __init__(
        self,
        dtype,
        k: int,
        config: RegisterClusterConfig,
        shared_memory_limit: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        return_values: bool = False,
        row_length: int | None = None,
        col_offset: int = 0,
    ):
        self.dtype = dtype
        self.elems = Elements.of(dtype)
        self.k = k
        self.config = config
        config.validate(k, self.elems, shared_memory_limit)
        self.threads = config.threads
        self.next_n = next_n
        self.compress_ratio = compress_ratio
        self.return_values = return_values
        self.row_length = row_length
        self.col_offset = col_offset

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
        status: cute.Tensor,
    ):
        cfg = self.config
        elems = self.elems
        threads = cutlass.const_expr(self.threads)
        words = cutlass.const_expr(cfg.words_per_thread)
        bins = cutlass.const_expr(cfg.bins)
        splits = cutlass.const_expr(cfg.splits)
        k = cutlass.const_expr(self.k)
        row_stride = cutlass.const_expr(x.shape[1])
        n_cols = cutlass.const_expr(
            self.row_length if self.row_length is not None else x.shape[1]
        )
        telemetry = cutlass.const_expr(cfg.telemetry)
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        rank = cluster_rank()

        smem = SmemAllocator()
        s_bins_all = smem.allocate_array(
            cutlass.Int32, max(bins, COARSE_BINS) + 4, byte_alignment=128
        )
        s_bins = s_bins_all + 4  # slot -1 takes out-of-range elements
        s_groups = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_merged_groups = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_fine = smem.allocate_array(cutlass.Int32, 16, byte_alignment=128)
        s_tie_keys = smem.allocate_array(
            cutlass.Uint32, cfg.tie_capacity, byte_alignment=128
        )
        s_tie_idx = smem.allocate_array(
            cutlass.Int32, cfg.tie_capacity, byte_alignment=128
        )
        s_slots = smem.allocate_array(cutlass.Int32, threads // 32, byte_alignment=128)
        s_slots_u32 = smem.allocate_array(
            cutlass.Uint32, 2 * (threads // 32), byte_alignment=128
        )
        s_result = smem.allocate_array(cutlass.Int32, 16, byte_alignment=128)

        if cutlass.const_expr(cfg.pdl):
            wait_for_prior_grid()
        mark = cutlass.Int64(0)
        if telemetry:
            mark = read_clock64()
        length = effective_length(
            lengths, row, n_cols, self.next_n, self.compress_ratio
        )
        row_ptr = x.iterator + cutlass.Int64(row) * row_stride + self.col_offset
        out_row = out.iterator + cutlass.Int64(row) * k
        status_row = status.iterator + cutlass.Int64(row) * STATUS_WORDS

        # the direct arm is rank 0's alone; no cluster barrier or DSMEM access happens on it
        if cutlass.Int32(k) >= length:
            if rank == 0:
                for i in range(tidx, k, threads):
                    v = cutlass.Int32(-1)
                    if i < length:
                        v = cutlass.Int32(i)
                    out_row[i] = v
                if tidx == 0:
                    status_row[0] = cutlass.Int32(0)
                    status_row[1] = cutlass.Int32(0)
        else:
            # this CTA's slice: equal vector-aligned chunks, the last one shorter or empty
            chunk = (
                (length + cutlass.Int32(splits - 1)) // cutlass.Int32(splits)
                + cutlass.Int32(elems.per_vector - 1)
            ) & ~cutlass.Int32(elems.per_vector - 1)
            start = rank * chunk
            count = length - start
            if count > chunk:
                count = chunk
            if count < 0:
                count = cutlass.Int32(0)
            zero_bins(s_bins_all, bins + 4, tidx, threads)
            if tidx == 0:
                s_result[12] = cutlass.Int32(
                    -1
                )  # key range of the crossing bin: min (0xFFFFFFFF) ...
                s_result[13] = cutlass.Int32(0)  # ... and max, folded in by every CTA
            cute.arch.barrier()
            load_len = count
            if load_len < 1:
                load_len = cutlass.Int32(
                    1
                )  # an empty slice loads one vector and masks everything
            wordvals = load_row_words(
                row_ptr + start, load_len, tidx, threads, words, elems.log2_per_vector
            )
            packed_bins = count_coarse_bins(
                elems, wordvals, count, s_bins, tidx, threads, words, bins
            )
            cute.arch.barrier()
            summarize_groups_256(s_bins, s_groups, bins, tidx)
            cluster_sync()  # every CTA's histogram and group sums are complete
            if telemetry:
                if tidx == 0:
                    status_row[3] = (read_clock64() - mark).to(cutlass.Int32)
                mark = read_clock64()
            cluster_crossing(
                s_bins,
                s_groups,
                s_merged_groups,
                s_fine,
                s_result,
                k,
                bins,
                splits,
                tidx,
                rank,
            )
            cut_bin = s_result[0]
            above = s_result[1]
            ties = s_result[6]  # the row's, from every peer's crossing bin
            overflow = cutlass.Int32(
                s_result[2] > cutlass.Int32(cfg.tie_capacity)
            )  # cluster-uniform
            if telemetry:
                if tidx == 0:
                    status_row[4] = (read_clock64() - mark).to(cutlass.Int32)
                mark = read_clock64()
            root = cutlass.Int32(0)
            classify_from_registers_cluster(
                elems,
                wordvals,
                packed_bins,
                cut_bin,
                start,
                out_row,
                peer_shared_address(s_tie_keys.toint(), root),
                peer_shared_address(s_tie_idx.toint(), root),
                s_result[4],
                s_result[5],
                cfg.tie_capacity,
                s_slots,
                tidx,
                threads,
                words,
            )
            if overflow == 1:
                # the bin will not fit rank 0's tie stage: fold this slice's key range in the
                # bin into rank 0's before the barrier, after which peers may exit
                kmin, kmax = key_range_in_bin(
                    elems,
                    wordvals,
                    packed_bins,
                    cut_bin,
                    s_slots_u32,
                    tidx,
                    threads,
                    words,
                )
                if tidx == 0:
                    peer_min_u32(
                        peer_shared_address((s_result + 12).toint(), root), kmin
                    )
                    peer_max_u32(
                        peer_shared_address((s_result + 13).toint(), root), kmax
                    )
            cluster_sync()  # winners written, ties (or ranges) landed on rank 0; peers stop touching rank 0
            if telemetry:
                if tidx == 0:
                    status_row[5] = (read_clock64() - mark).to(cutlass.Int32)
                mark = read_clock64()
            if rank == 0:
                ok = _select_ties(
                    elems,
                    k,
                    above,
                    ties,
                    out_row,
                    s_bins,
                    s_tie_keys,
                    s_tie_idx,
                    cfg.tie_capacity,
                    cfg.ballot_limit,
                    s_slots,
                    s_result,
                    tidx,
                    threads,
                )
                arm = cutlass.Int32(2)
                if ok == 0:
                    # the crossing bin overflowed the tie stage; everything above it is already in
                    # the output, so rank 0 runs the exact select on that bin over the whole row,
                    # by the key range merged from every slice
                    kmin_row = cutlass.Uint32(s_result[12])
                    kmax_row = cutlass.Uint32(s_result[13])
                    cute.arch.barrier()
                    radix_select_in_range(
                        elems,
                        row_ptr,
                        length,
                        cutlass.Int32(k) - above,
                        out_row + above,
                        kmin_row,
                        kmax_row,
                        s_bins,
                        s_slots,
                        s_result,
                        tidx,
                        threads,
                    )
                    arm = cutlass.Int32(3)
                if telemetry:
                    if tidx == 0:
                        status_row[6] = (read_clock64() - mark).to(cutlass.Int32)
                if tidx == 0:
                    status_row[0] = cutlass.Int32(1) - ok
                    status_row[1] = arm
                    status_row[2] = ties
        if cutlass.const_expr(self.return_values):
            # rank 0 wrote the ties and the fallback; the peers' winners preceded the cluster
            # barrier above, so after this block barrier every index of the row is visible
            if rank == 0:
                cute.arch.barrier()
                gather_values(
                    self.dtype,
                    row_ptr,
                    out_row,
                    values.iterator + cutlass.Int64(row) * k,
                    k,
                    tidx,
                    threads,
                )
        if cutlass.const_expr(cfg.pdl):
            release_dependent_grid()

    @cute.jit
    def launch(
        self,
        x: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
        status: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        splits = self.config.splits
        self.kernel(x, lengths, out, values, status).launch(
            grid=(x.shape[0], splits, 1),
            block=(self.threads, 1, 1),
            cluster=(1, splits, 1),
            use_pdl=self.config.pdl,
            stream=stream,
        )


_compiled: dict = {}


SLICE_ELEMENTS = (
    16384  # per CTA: 16 elements per thread; the per-element work sets the CTA's time
)


def register_cluster_config_for(
    facts, dtype: torch.dtype, k: int, n: int, rows: int = 0
) -> RegisterClusterConfig:
    """The shape for a batch of ``rows`` rows of ``n`` elements.

    Slices hold at most 16K elements per CTA (16 words per thread for fp32, 8 for 16-bit: a
    32-element slice per thread measured 12.0 us against 8.9 for the streaming kernel at bf16
    64K b=8), and the cluster is the *largest* of 2, 4, 8 whose clusters for the batch fit one
    wave (``cluster_capacity``), with the fewest words per thread that hold the slice (at least
    4, one vector).  The per-CTA phases (load, count, classify) scale with the slice, so more
    CTAs of smaller slices win while the SMs are free: B200 k=1024 64K b=8 7.89 (4 x 16 words)
    -> 6.94 us (8 x 8), 32K b=8 7.63 (2 x 16) -> 6.00 (8 x 4), 32K b=16 7.61 -> 6.44 (4 x 8);
    gvr_2's reg_clus, which always splits eight ways, measures 6.43 and 6.23 there.  One
    cluster past the capacity costs a second wave (32K b=16 at 8 x 4: 10.7; 64K b=32 at 8 x 8:
    17.8), hence the bound.  With ``rows`` unknown (0) the smallest cluster is chosen.  N / 4
    bins.

    Re-swept 2026-09-18 (v0.1.30, B200 and Rubin, b=8/16/32, k=512/1024): the rule is at its
    bound everywhere.  B200 32K b=16 stays at 4 x 8 (6.32 us; 8 x 4 is 9.8, a second wave, as
    the part fits 15 clusters of 8 and gvr_2 vetoes 8 above 15 rows as well), 64K b=16 at 4 x
    16 (7.26-7.71; 8 x 8 is 11.2-12.0); Rubin fits 22 clusters of 8 and runs 8 x 4 at 32K b=16
    (4.33 against 4.88 for 4 x 8).  The 0.88-0.94x against gvr_2 on B200 32K b=16 is therefore
    per-CTA cost at the same shape, not the shape.
    """
    from ...dispatch.device import cluster_capacity

    per_word = 1 if dtype == torch.float32 else 2
    if n > 8 * SLICE_ELEMENTS:
        raise ValueError(
            f"row length {n} exceeds the clustered register capacity {8 * SLICE_ELEMENTS}"
        )
    candidates = [s for s in (2, 4, 8) if n <= s * SLICE_ELEMENTS]
    splits = candidates[0]
    if rows > 0:
        for s in reversed(candidates):
            if rows <= cluster_capacity(facts.index, s):
                splits = s
                break
    slice_elements = -(-n // splits)
    words = 4
    while words * 1024 * per_word < slice_elements:
        words *= 2
    bins = 1024
    while bins < 4096 and bins * 4 < n:
        bins *= 2
    return RegisterClusterConfig(
        words_per_thread=words,
        splits=splits,
        bins=bins,
        tie_capacity=min(8192, max(2048, -(-k // 1024) * 1024)),
        pdl=facts.supports_pdl,
    )


__all__ = list(__all__) + ["SLICE_ELEMENTS"]


def topk_register_cluster(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    config: RegisterClusterConfig | None = None,
    out: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    next_n: int = 1,
    compress_ratio: int = 1,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Indices of the k largest of each row of ``x`` (rows, N <= 128K), same contract as
    ``topk_streaming`` (``workspace`` holds the status words and a misaligned copy's arena).
    Needs SM90 or newer (clusters)."""
    from ...dispatch.device import device_facts

    from ..dispatch.workspace import carve, workspace_layout
    from .layout import arena_bytes

    assert x.dtype in _DTYPES
    check_layout(x)
    rows, n = x.shape
    facts = device_facts(x.device)
    if not facts.supports_clusters:
        raise ValueError(
            "the clustered register kernel needs thread-block clusters (SM90+)"
        )
    if config is None:
        config = register_cluster_config_for(facts, x.dtype, k, n, rows)
    elems = Elements.of(_DTYPES[x.dtype])
    if n > config.splits * config.slice_capacity(elems):
        raise ValueError(
            f"row length {n} exceeds this configuration's capacity {config.splits * config.slice_capacity(elems)}"
        )
    arena = None
    if workspace is not None:
        ws = carve(
            workspace,
            workspace_layout("register_cluster", config, rows, arena_bytes(x)),
            x.device,
        )
        arena = ws.arena
        if status is None:
            status = ws.status
    xa, col_offset = arena_view(x, arena)
    if lengths is None:
        lengths = torch.full(
            (rows // next_n,), n * compress_ratio, device=x.device, dtype=torch.int32
        )
    if out is None:
        out = torch.empty(rows, k, device=x.device, dtype=torch.int32)
    if status is None:
        status = torch.empty(rows * STATUS_WORDS, device=x.device, dtype=torch.int32)
    vals = values if values is not None else _no_values(x.device, x.dtype)
    key = (
        x.dtype,
        k,
        rows,
        n,
        xa.shape[1],
        col_offset,
        config,
        facts.capability,
        next_n,
        compress_ratio,
        values is not None,
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    args = (
        from_dlpack(xa),
        from_dlpack(lengths),
        from_dlpack(out),
        from_dlpack(vals),
        from_dlpack(status),
        stream,
    )
    if key not in _compiled:
        kern = RegisterClusterTopK(
            _DTYPES[x.dtype],
            k,
            config,
            facts.shared_memory_optin,
            next_n,
            compress_ratio,
            values is not None,
            n,
            col_offset,
        )
        _compiled[key] = cute.compile(kern.launch, *args)
    _compiled[key](*args)
    return out
