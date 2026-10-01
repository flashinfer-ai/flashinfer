"""Kernel 4: census split.  Exact top-k for any k, the row split across CTAs, two launches.

The sampled kernels stage about 1.5k survivors and hand the whole answer to one CTA; past k
of a few percent of the row that hand-off is the time (1M k=200000 on B200: 187 us against a
plain radix select's 67).  A radix select streams the row once per key byte with every CTA
of the row busy, so its cost is a few row passes whatever k is; FlashInfer's ``radix``
backends pay four to five.  This kernel pays two, using the census idea of the exact fallback:

* **Census** (launch A, every CTA over its slice): a 4096-bin histogram of the fp16 ordered
  key's top 12 bits (64 bins per octave, the same bins as ``phases/census.py``), published as
  a whole to the row's slab.  The last arriver merges the ``splits`` histograms, finds the
  rank-k bin and the count above it (``crossing_wide_pair``), and, because it holds every
  CTA's histogram, computes each CTA's count above the bin and in it: exclusive prefixes over
  the CTAs are the output offsets of launch B.  It writes a 16-word header and resets the
  arrival counter.
* **Emit** (launch B, every CTA over its slice): elements in bins above the cut go straight to
  the output at the CTA's offset, in parallel across CTAs; members of the cut bin are staged
  (key, index) into the row's tie slab at the CTA's tie offset, and their key range is
  reduced per CTA.  The last arriver copies the staged ties into shared memory and runs the
  tie select (``k - above`` of them, deterministic); if the bin holds more than the shared
  stage, it runs the radix select over the bin's key range instead (``_radix_in_range``, the
  row streamed once per differing key byte), and gathers the values if asked.

Two launches instead of a grid barrier: the CTAs of a row need each other's histograms
before they can emit, and a grid may run in several waves, so no CTA may wait for another.
The kernel boundary orders the header and slab for launch B, and programmatic dependent
launch lets B's prologue overlap A's tail.  The crossing bin of a 12-bit fp16 census is small
on real data (a 1M randn row puts about 3K elements in the bin at any rank), so the tie
select is a few microseconds and the two row passes are the cost: on B200 at 1M b=8 the
sampled kernel's 26.6 us at k=16384 and 187 us at k=200000 both become about the same
twenty-something microseconds here.

Same contract as the other kernels: ``lengths`` with ``next_n`` and ``compress_ratio``,
rows shorter than k padded with -1 (rank 0 writes the identity in launch A), values on
request, NaN above +inf, order within a row unspecified.  Slab per row (Int32 words):
``HEADER_WORDS`` header, ``splits x 4096`` histograms, ``4 x splits`` per-CTA words (winner
offset, tie offset, key min, key max), ``2 x tie_slab`` staged ties.  Two arrival counters
per row, zero between launches.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

try:
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # older DSL releases
    from cutlass.utils.smem_allocator import SmemAllocator

try:
    from cuda.bindings import driver as cuda_driver
except ImportError:
    from cuda import cuda as cuda_driver

from ...block.crossing import crossing_wide_pair
from ...block.reduce import block_max_min_u32
from ...device.atomics import global_add_acq_rel, shared_add, shared_count
from ...device.launch import release_dependent_grid, wait_for_prior_grid
from ...device.memory import load_global_l2_i32
from ...device.warp import warp_inclusive_scan_add, warp_sum

from ..phases.census import COARSE_BINS, element_bits, pair_coarse_bins
from ..phases.elements import Elements
from ..phases.fallback import _radix_in_range
from ..phases.resolve import _select_ties
from ..phases.row_scan import index_of_element, load_quad, quad_stride
from ..phases.varlen import effective_length, gather_values
from .layout import arena_bytes, arena_view, check_layout

__all__ = [
    "STATUS_WORDS",
    "HEADER_WORDS",
    "CensusSplitConfig",
    "CensusSplitTopK",
    "census_split_config_for",
    "slab_words_per_row",
    "topk_census_split",
]

_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

STATUS_WORDS = 8  # per row: [0] radix taken (the cut bin overflowed the shared tie stage), [1] arm (0 identity, 4 census split, 3 radix), [2] members of the cut bin, [3] count above it
HEADER_WORDS = 16  # per row: [0] cut bin, [1] above, [2] in bin, [3] staged (1: ties staged in the slab, 0: beyond tie_slab)
# per-CTA words after the histograms: [0..splits) winner offsets, [splits..2s) tie offsets, [2s..3s) key min, [3s..4s) key max


def slab_words_per_row(splits: int, tie_slab: int) -> int:
    return HEADER_WORDS + splits * COARSE_BINS + 4 * splits + 2 * tie_slab


@dataclass(frozen=True)
class CensusSplitConfig:
    """Every knob of the census split kernel; all compile-time, all in the artifact name."""

    threads: int = 1024
    splits: int = 16  # CTAs per row (1 .. 32)
    tie_capacity: int = 8192  # shared tie stage (key, index) pairs: a multiple of threads in [2 x threads, 8 x threads]
    tie_slab: int = 16384  # staged ties per row in the slab; a bin beyond this takes the radix select over its key range
    ballot_limit: int = 128
    pdl: bool = False
    telemetry: bool = False

    def validate(self, k: int, elems: Elements, shared_memory_limit: int) -> None:
        if self.threads not in (512, 1024):
            raise ValueError(f"threads must be 512 or 1024, got {self.threads}")
        if not 1 <= self.splits <= 32:
            raise ValueError(f"splits must be in 1..32, got {self.splits}")
        if (
            self.tie_capacity < 2 * self.threads
            or self.tie_capacity > 8 * self.threads
            or self.tie_capacity % self.threads
        ):
            raise ValueError(
                "tie_capacity must be a multiple of threads in [2 * threads, 8 * threads]"
            )
        if self.tie_slab < self.tie_capacity:
            raise ValueError("tie_slab must hold at least the shared tie stage")
        if self.shared_memory_bytes() > shared_memory_limit:
            raise ValueError(
                f"{self.shared_memory_bytes()} B exceeds the {shared_memory_limit} B shared-memory budget"
            )

    def shared_memory_bytes(self) -> int:
        return (
            COARSE_BINS * 4
            + 2 * self.tie_capacity * 4
            + (3 * (self.threads // 32) + 16 + 64) * 4
            + 1024
        )

    def name(self) -> str:
        return "_".join(f"{f.name}{getattr(self, f.name)}" for f in fields(self))


class CensusSplitTopK:
    """A compiled census split top-k for one (dtype, k, config); call it on (rows, N) inputs."""

    def __init__(
        self,
        dtype,
        k: int,
        config: CensusSplitConfig,
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
        self.threads = config.threads
        self.warps = config.threads // 32
        self.next_n = next_n
        self.compress_ratio = compress_ratio
        self.return_values = return_values
        self.row_length = row_length
        self.col_offset = col_offset
        config.validate(k, self.elems, shared_memory_limit)

    # ------------------------------------------------------------------ shared pieces
    @cute.jit
    def slice_of(self, length, rank):
        """(start, count) of this CTA's slice: equal vector-aligned chunks, the last one shorter."""
        splits = cutlass.const_expr(self.config.splits)
        per_vector = cutlass.const_expr(self.elems.per_vector)
        chunk = (
            (length + cutlass.Int32(splits - 1)) // cutlass.Int32(splits)
            + cutlass.Int32(per_vector - 1)
        ) & ~cutlass.Int32(per_vector - 1)
        start = rank * chunk
        count = length - start
        if count > chunk:
            count = chunk
        if count < 0:
            count = cutlass.Int32(0)
        return start, count

    @cute.jit
    def row_slab(self, slab, row):
        return slab.iterator + cutlass.Int64(row) * slab_words_per_row(
            self.config.splits, self.config.tie_slab
        )

    # ------------------------------------------------------------------ launch A: census
    @cute.kernel
    def census_kernel(
        self,
        x: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        status: cute.Tensor,
        slab: cute.Tensor,
        counters: cute.Tensor,
    ):
        cfg = self.config
        elems = self.elems
        threads = cutlass.const_expr(self.threads)
        warps = cutlass.const_expr(self.warps)
        k = cutlass.const_expr(self.k)
        splits = cutlass.const_expr(cfg.splits)
        row_stride = cutlass.const_expr(x.shape[1])
        n_cols = cutlass.const_expr(
            self.row_length if self.row_length is not None else x.shape[1]
        )
        tidx, _, _ = cute.arch.thread_idx()
        row, rank, _ = cute.arch.block_idx()

        smem = SmemAllocator()
        s_bins = smem.allocate_array(cutlass.Int32, COARSE_BINS, byte_alignment=128)
        s_slots = smem.allocate_array(cutlass.Int32, warps, byte_alignment=128)
        s_result = smem.allocate_array(cutlass.Int32, 16, byte_alignment=128)
        s_counts = smem.allocate_array(
            cutlass.Int32, 64, byte_alignment=128
        )  # per-CTA above (0..32) and in-bin (32..64) counts

        if cutlass.const_expr(cfg.pdl):
            wait_for_prior_grid()
        row_ptr = x.iterator + cutlass.Int64(row) * row_stride + self.col_offset
        out_row = out.iterator + cutlass.Int64(row) * k
        status_row = status.iterator + cutlass.Int64(row) * STATUS_WORDS
        length = effective_length(
            lengths, row, n_cols, self.next_n, self.compress_ratio
        )
        header = self.row_slab(slab, row)
        hists = header + HEADER_WORDS
        per_cta = hists + splits * COARSE_BINS

        if cutlass.Int32(k) >= length:
            if rank == 0:  # the identity, padded; launch B leaves the row alone
                for i in range(tidx, k, threads):
                    v = cutlass.Int32(-1)
                    if i < length:
                        v = cutlass.Int32(i)
                    out_row[i] = v
                if tidx == 0:
                    status_row[0] = cutlass.Int32(0)
                    status_row[1] = cutlass.Int32(0)
                    status_row[2] = cutlass.Int32(0)
                    status_row[3] = cutlass.Int32(0)
        else:
            start, count = self.slice_of(length, rank)
            for i in range(tidx, COARSE_BINS, threads):
                s_bins[i] = cutlass.Int32(0)
            cute.arch.barrier()
            # the census of this slice (phases/census.py's first pass): bins of two elements
            # at a time, four vectors in flight, plain shared increments
            pairs = cutlass.const_expr(8 * elems.per_word)
            n_vectors = (count + cutlass.Int32(elems.per_vector - 1)) >> cutlass.Int32(
                elems.log2_per_vector
            )
            last_vector = n_vectors - 1
            stride = cutlass.const_expr(quad_stride(threads))
            base = (row_ptr + start).toint()
            bins_base = s_bins.toint()
            for v in range(tidx, n_vectors, stride):
                words = load_quad(base, v, threads, last_vector)
                for p in cutlass.range_constexpr(pairs):
                    lo, hi = pair_coarse_bins(elems, words, p)
                    if index_of_element(elems, v, threads, 2 * p) < count:
                        shared_count(bins_base + lo * 4)
                    if index_of_element(elems, v, threads, 2 * p + 1) < count:
                        shared_count(bins_base + hi * 4)
            cute.arch.barrier()
            # publish the histogram whole (coalesced plain stores; the arrival's release
            # lifts them, after the barrier, to GPU scope)
            mine = hists + rank * COARSE_BINS
            for i in range(tidx, COARSE_BINS, threads):
                mine[i] = s_bins[i]
            cute.arch.barrier()
            if tidx == 0:
                arrived = global_add_acq_rel(counters.iterator + row * 2, 1)
                s_result[10] = cutlass.Int32(0)
                if arrived == cutlass.Int32(splits - 1):
                    s_result[10] = cutlass.Int32(1)
            cute.arch.barrier()
            if s_result[10] == 1:
                # merged histogram: thread t sums bins t, t + threads, .. over the CTAs
                for i in range(tidx, COARSE_BINS, threads):
                    total = cutlass.Int32(0)
                    for r in cutlass.range_constexpr(splits):
                        total = total + load_global_l2_i32(
                            (hists + r * COARSE_BINS + i).toint()
                        )
                    s_bins[i] = total
                cute.arch.barrier()
                crossing_wide_pair(
                    s_bins,
                    COARSE_BINS,
                    cutlass.Int32(k),
                    cutlass.Int32(k),
                    s_slots,
                    s_result,
                    tidx,
                    threads,
                )
                cut = s_result[0]
                above = s_result[1]
                in_bin = s_result[2]
                # each CTA's count above the cut and in it: warp w takes CTAs w, w + warps, ..;
                # lane l sums that CTA's bins cut + 1 + l, + 32, .. (independent L2 loads)
                warp = tidx // 32
                lane = tidx % 32
                for r in range(warp, splits, warps):
                    seg = hists + r * COARSE_BINS
                    part = cutlass.Int32(0)
                    for b in range(cut + 1 + lane, COARSE_BINS, 32):
                        part = part + load_global_l2_i32((seg + b).toint())
                    total = warp_sum(part)
                    if lane == 0:
                        s_counts[r] = total
                        s_counts[32 + r] = load_global_l2_i32((seg + cut).toint())
                cute.arch.barrier()
                if (
                    tidx < 32
                ):  # exclusive prefixes over the CTAs: launch B's output and tie offsets
                    a = cutlass.Int32(0)
                    t = cutlass.Int32(0)
                    if tidx < cutlass.Int32(splits):
                        a = s_counts[tidx]
                        t = s_counts[32 + tidx]
                    a_incl = warp_inclusive_scan_add(a, tidx)
                    t_incl = warp_inclusive_scan_add(t, tidx)
                    if tidx < cutlass.Int32(splits):
                        per_cta[tidx] = a_incl - a
                        per_cta[splits + tidx] = t_incl - t
                    if tidx == 0:
                        header[0] = cut
                        header[1] = above
                        header[2] = in_bin
                        staged = cutlass.Int32(0)
                        if in_bin <= cutlass.Int32(cfg.tie_slab):
                            staged = cutlass.Int32(1)
                        header[3] = staged
                        # plain store: launch B starts after this grid completes
                        counter_a = counters.iterator + row * 2
                        counter_a[0] = cutlass.Int32(0)
        if cutlass.const_expr(cfg.pdl):
            release_dependent_grid()

    # ------------------------------------------------------------------ launch B: emit
    @cute.kernel
    def emit_kernel(
        self,
        x: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
        status: cute.Tensor,
        slab: cute.Tensor,
        counters: cute.Tensor,
    ):
        cfg = self.config
        elems = self.elems
        threads = cutlass.const_expr(self.threads)
        warps = cutlass.const_expr(self.warps)
        k = cutlass.const_expr(self.k)
        splits = cutlass.const_expr(cfg.splits)
        row_stride = cutlass.const_expr(x.shape[1])
        n_cols = cutlass.const_expr(
            self.row_length if self.row_length is not None else x.shape[1]
        )
        tidx, _, _ = cute.arch.thread_idx()
        row, rank, _ = cute.arch.block_idx()

        smem = SmemAllocator()
        s_bins = smem.allocate_array(
            cutlass.Int32, COARSE_BINS, byte_alignment=128
        )  # tie select scratch, radix histogram
        s_tie_keys = smem.allocate_array(
            cutlass.Uint32, cfg.tie_capacity, byte_alignment=128
        )
        s_tie_idx = smem.allocate_array(
            cutlass.Int32, cfg.tie_capacity, byte_alignment=128
        )
        s_slots = smem.allocate_array(cutlass.Int32, warps, byte_alignment=128)
        s_slots_u32 = smem.allocate_array(cutlass.Uint32, 2 * warps, byte_alignment=128)
        s_result = smem.allocate_array(cutlass.Int32, 16, byte_alignment=128)

        if cutlass.const_expr(cfg.pdl):
            wait_for_prior_grid()  # launch A complete: header, histograms and identity rows visible
        row_ptr = x.iterator + cutlass.Int64(row) * row_stride + self.col_offset
        out_row = out.iterator + cutlass.Int64(row) * k
        status_row = status.iterator + cutlass.Int64(row) * STATUS_WORDS
        length = effective_length(
            lengths, row, n_cols, self.next_n, self.compress_ratio
        )
        header = self.row_slab(slab, row)
        per_cta = header + HEADER_WORDS + splits * COARSE_BINS
        tie_keys = per_cta + 4 * splits
        tie_idx = tie_keys + cfg.tie_slab

        if cutlass.Int32(k) >= length:
            if cutlass.const_expr(self.return_values):
                if rank == 0:
                    gather_values(
                        self.dtype,
                        row_ptr,
                        out_row,
                        values.iterator + cutlass.Int64(row) * k,
                        k,
                        tidx,
                        threads,
                    )
        else:
            cut = load_global_l2_i32(header.toint())
            above = load_global_l2_i32((header + 1).toint())
            in_bin = load_global_l2_i32((header + 2).toint())
            staged = load_global_l2_i32((header + 3).toint())
            win_off = load_global_l2_i32((per_cta + rank).toint())
            tie_off = load_global_l2_i32((per_cta + splits + rank).toint())
            start, count = self.slice_of(length, rank)
            if tidx == 0:
                s_result[6] = cutlass.Int32(0)  # winner cursor within this CTA's range
                s_result[7] = cutlass.Int32(0)  # tie cursor within this CTA's range
            cute.arch.barrier()
            pairs = cutlass.const_expr(8 * elems.per_word)
            n_vectors = (count + cutlass.Int32(elems.per_vector - 1)) >> cutlass.Int32(
                elems.log2_per_vector
            )
            last_vector = n_vectors - 1
            stride = cutlass.const_expr(quad_stride(threads))
            base = (row_ptr + start).toint()
            kmax = cutlass.Uint32(0)
            kmin = cutlass.Uint32(0xFFFFFFFF)
            for v in range(tidx, n_vectors, stride):
                words = load_quad(base, v, threads, last_vector)
                for p in cutlass.range_constexpr(pairs):
                    lo, hi = pair_coarse_bins(elems, words, p)
                    for side in cutlass.range_constexpr(2):
                        b = hi
                        if cutlass.const_expr(side == 0):
                            b = lo
                        e = cutlass.const_expr(2 * p + side)
                        idx = index_of_element(elems, v, threads, e)
                        if idx < count:
                            gidx = start + idx
                            if b > cut:
                                out_row[win_off + shared_add(s_result + 6, 1)] = gidx
                            else:
                                if b == cut:
                                    key = elems.key(element_bits(elems, words, e))
                                    if key > kmax:
                                        kmax = key
                                    if key < kmin:
                                        kmin = key
                                    if staged == 1:
                                        t = tie_off + shared_add(s_result + 7, 1)
                                        tie_keys[t] = key.bitcast(cutlass.Int32)
                                        tie_idx[t] = gidx
            cute.arch.barrier()
            kmax, kmin = block_max_min_u32(kmax, kmin, s_slots_u32, tidx, threads)
            if tidx == 0:
                per_cta[2 * splits + rank] = kmin.bitcast(cutlass.Int32)
                per_cta[3 * splits + rank] = kmax.bitcast(cutlass.Int32)
            cute.arch.barrier()
            if tidx == 0:
                arrived = global_add_acq_rel(counters.iterator + row * 2 + 1, 1)
                s_result[10] = cutlass.Int32(0)
                if arrived == cutlass.Int32(splits - 1):
                    s_result[10] = cutlass.Int32(1)
            cute.arch.barrier()
            if s_result[10] == 1:
                ok = cutlass.Int32(0)
                if (staged == 1) & (in_bin <= cutlass.Int32(cfg.tie_capacity)):
                    for t in range(tidx, in_bin, threads):
                        s_tie_keys[t] = cutlass.Uint32(
                            load_global_l2_i32((tie_keys + t).toint())
                        )
                        s_tie_idx[t] = load_global_l2_i32((tie_idx + t).toint())
                    cute.arch.barrier()
                    ok = _select_ties(
                        elems,
                        k,
                        above,
                        in_bin,
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
                arm = cutlass.Int32(4)
                if ok == 0:
                    # the bin's exact key range over every CTA, then the radix select over
                    # the row restricted to it (streamed once per differing key byte)
                    lo = cutlass.Uint32(0xFFFFFFFF)
                    hi = cutlass.Uint32(0)
                    for r in cutlass.range_constexpr(splits):
                        c_in = load_global_l2_i32(
                            (per_cta + splits + r).toint()
                        )  # tie offset r
                        rmin = cutlass.Uint32(
                            load_global_l2_i32((per_cta + 2 * splits + r).toint())
                        )
                        rmax = cutlass.Uint32(
                            load_global_l2_i32((per_cta + 3 * splits + r).toint())
                        )
                        if rmin < lo:
                            lo = rmin
                        if rmax > hi:
                            hi = rmax
                        c_in = c_in  # offsets are informational here; a CTA with no members left min > max
                    cute.arch.barrier()
                    _radix_in_range(
                        elems,
                        row_ptr,
                        length,
                        cutlass.Int32(k) - above,
                        out_row + above,
                        lo,
                        hi,
                        s_bins,
                        s_slots,
                        s_result,
                        tidx,
                        threads,
                    )
                    arm = cutlass.Int32(3)
                if tidx == 0:
                    status_row[0] = cutlass.Int32(1) - ok
                    status_row[1] = arm
                    status_row[2] = in_bin
                    status_row[3] = above
                    counter_b = counters.iterator + row * 2 + 1
                    counter_b[0] = cutlass.Int32(
                        0
                    )  # plain store: the next launch is ordered after this grid
                if cutlass.const_expr(self.return_values):
                    cute.arch.barrier()  # every index of the row is written (the others' before their release)
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
        slab: cute.Tensor,
        counters: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        """Both launches on the caller's stream, launch B programmatically dependent on A."""
        splits = self.config.splits
        self.census_kernel(x, lengths, out, status, slab, counters).launch(
            grid=(x.shape[0], splits, 1),
            block=(self.threads, 1, 1),
            use_pdl=self.config.pdl,
            stream=stream,
        )
        self.emit_kernel(x, lengths, out, values, status, slab, counters).launch(
            grid=(x.shape[0], splits, 1),
            block=(self.threads, 1, 1),
            use_pdl=self.config.pdl,
            stream=stream,
        )


# ---------------------------------------------------------------------------- policy and glue


def census_split_config_for(
    facts, dtype: torch.dtype, k: int, n: int, rows: int = 0
) -> CensusSplitConfig:
    """CTAs per row: the widest of 32, 16, 8, 4, 2 whose slices hold at least 4096 elements and
    whose grid fits one wave; one CTA per row for batches at or above the SM count (the two
    launches still overlap the tail).  Eight tie slots per thread, a 16K tie slab."""
    splits = 1
    if rows and rows < facts.sm_count:
        for s in (32, 16, 8, 4, 2):
            if n // s >= 4096 and rows * s <= facts.sm_count:
                splits = s
                break
    return CensusSplitConfig(splits=splits, pdl=facts.supports_pdl)


_slabs: dict = {}
_placeholders: dict = {}
_compiled: dict = {}


def _no_values(device, dtype):
    key = (device, dtype)
    if key not in _placeholders:
        _placeholders[key] = torch.empty(1, 1, dtype=dtype, device=device)
    return _placeholders[key]


def _slab_workspace(device, rows: int, config: CensusSplitConfig):
    """Per-(device, stream, rows, splits, tie_slab) slab and zeroed arrival counters (rows, 2),
    kept alive so the counters' self-reset carries from one launch to the next; keyed by the
    stream so concurrent streams never share one (see ``dispatch/workspace.py``)."""
    key = (
        device,
        torch.cuda.current_stream(device).cuda_stream,
        rows,
        config.splits,
        config.tie_slab,
    )
    if key not in _slabs:
        words = rows * slab_words_per_row(config.splits, config.tie_slab)
        _slabs[key] = (
            torch.empty(words, device=device, dtype=torch.int32),
            torch.zeros(rows, 2, device=device, dtype=torch.int32),
        )
    return _slabs[key]


def topk_census_split(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    config: CensusSplitConfig | None = None,
    out: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    next_n: int = 1,
    compress_ratio: int = 1,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Indices of the k largest of each row of ``x`` (rows, N), any k; same contract as
    ``topk_streaming`` (``status``: rows * STATUS_WORDS Int32)."""
    from ...dispatch.device import device_facts

    from ..dispatch.workspace import carve, workspace_layout

    assert x.dtype in _DTYPES
    check_layout(x)
    rows, n = x.shape
    facts = device_facts(x.device)
    if config is None:
        config = census_split_config_for(facts, x.dtype, k, n, rows)
    arena = None
    if workspace is not None:
        ws = carve(
            workspace,
            workspace_layout("census_split", config, rows, arena_bytes(x)),
            x.device,
        )
        assert ws.slab is not None and ws.counters is not None
        slab, counters, arena = ws.slab, ws.counters.view(rows, 2), ws.arena
        counters.zero_()  # the kernel needs zero arrivals at launch; the caller's memory holds anything
        if status is None:
            status = ws.status
    else:
        slab, counters = _slab_workspace(x.device, rows, config)
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
        from_dlpack(slab),
        from_dlpack(counters),
        stream,
    )
    if key not in _compiled:
        kern = CensusSplitTopK(
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
