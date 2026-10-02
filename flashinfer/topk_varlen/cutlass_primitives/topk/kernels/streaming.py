"""Kernel 3, streaming sample-filter top-k: the composition (see docs/kernels/streaming.md).

One CTA per row.  Phases in order: sample (three barriers) -> filter pass -> verdict and
repair -> resolution, with the exact radix select as the fallback for anything the sampled
path cannot certify.  Short rows go straight to the radix select.  This file names the
phases and owns the shared-memory plan and the configuration; the phases hold the logic.

With ``splits > 1`` the row is shared by a cluster of CTAs launched as ``cluster=(1, splits,
1)``: every CTA samples the whole row (identical threshold), filters its own slice, and the
verdict and resolution merge over distributed shared memory.  The slab merge for parts
without clusters is the next increment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

try:  # the allocator moved between DSL releases; both spellings are in use on our parts
    from cutlass.memory.smem import SmemAllocator
except ImportError:  # pragma: no cover - older nvidia-cutlass-dsl
    from cutlass.utils.smem_allocator import SmemAllocator
try:
    from cuda.bindings import driver as cuda_driver
except ImportError:  # pragma: no cover - older cuda-python
    from cuda import cuda as cuda_driver

from ...device.cluster import cluster_sync
from ...device.launch import release_dependent_grid, wait_for_prior_grid
from ...device.warp import ballot_count, ballot_rank
from ...device.timers import read_clock64

from ...block.crossing import crossing_wide_pair

from ..phases.aim import aim_tight, aim_wide
from ..phases.elements import Elements
from ..phases.fallback import exact_select_row, radix_select_in_range
from ..phases.filter_pass import filter_pass
from ..phases.register_row import (
    classify_from_registers,
    count_coarse_bins,
    key_range_in_bin,
    load_row_words,
    words_after_barrier,
    zero_bins,
)
from ..phases.repair import verdict_and_repair, verdict_and_repair_cluster
from ..phases.resolve import _select_ties, emit_and_select, emit_and_select_cluster
from ..phases.sample import Threshold, sample_probe, sample_threshold, valid_samples
from ..phases.slab import merge_slab, publish_and_arrive, slab_words_per_row
from ..phases.varlen import effective_length, gather_values
from .layout import arena_view, check_layout
from .register_resident import _no_values
from ...device.cluster import cluster_rank

__all__ = ["StreamingConfig", "StreamingTopK", "topk_streaming"]

_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
EXACT_ONLY = (
    1 << 30
)  # a short_cutoff no row length reaches: every row takes the exact select
STATUS_WORDS = 12  # per row: [0] sampled path failed, [1] arm (0 identity, 1 census, 2 sampled, 3 radix), [2] survivors, [3..7] phase clocks, [8..11] slab-merge sub-phase clocks


@dataclass(frozen=True)
class StreamingConfig:
    """Every performance knob of the streaming kernel.  All fields are compile-time constants
    and all appear in the artifact name."""

    # shape
    threads: int = 1024
    ctas_per_sm: int = 1
    unroll: int = 4
    walk_width: int = 1  # survivor reloads issued together in the filter's bit-walk (1, 2, 4): phases/filter_pass.py
    # split
    splits: int = 1  # CTAs per row
    merge: str = "cluster"  # "cluster": DSMEM merge (SM90+, splits <= 8); "slab": global-memory last arriver
    # sample
    aim: str = "tight"  # "tight" | "wide"
    aim_margin: float = 0.125  # tight aim, grids with the sigma floor: k + max(aim_margin * k, length / 256), raised to the floor (phases/aim.py)
    aim_margin_small: float = 0.5  # tight aim, grids below aim_floor_rows (no floor): the fixed margin alone must cover the spread
    aim_z: float = 3.5  # tight aim: sigma of the statistical undershoot floor
    aim_floor_rows: int = (
        32  # tight aim: the floor applies to grids of at least this many rows
    )
    floor_multiple: int = 2
    span_ext: float = 1.5
    sample_vectors: int = 1  # adjacent 16-byte vectors per thread in the sample (1, 2 or 4): the survivor spread shrinks with the square root
    sample_bins: int = 256  # equal-width bins of the sample histogram (256, 512, 1024, 2048); above 256 they live in the dead stage
    reprobe_below: float = 0.0  # a row shorter than this fraction of the buffer re-samples at length-derived positions (full sample count) instead of masking the buffer probe; 0: never
    # stage
    stage: int = 8192
    tie_capacity: int = 2048
    # repair
    overflow_offset: int = 64
    # epilogue
    ballot_limit: int = 128
    scan_emit: bool = False  # output positions from a block scan instead of a shared cursor per candidate (phases/resolve.py)
    bin_cursors: bool = True  # emit through a cursor per histogram bin (v0.1.29); False: one shared cursor for all winners, a device fact (SM80)
    # scheduling
    register_arm: bool = False  # rows that fit register_words per thread take the register-resident phases (whole-row configurations)
    register_words: int = (
        16  # 32-bit words per thread held in registers by that arm (4, 8 or 16)
    )
    count_after_barrier: bool = True  # the arm pins its count below the zeroing barrier (see register_row.words_after_barrier); a device fact
    short_cutoff: int = 16384  # rows at or below take the exact select directly; EXACT_ONLY: every row does
    lpt_order: bool = False  # block b processes the row of rank b by length (longest first); wide batches, splits == 1
    row_order: bool = False  # block b processes the row named by a permutation from a one-CTA prepare launch (phases/row_order.py) or the caller; wide batches
    pdl: bool = False
    packed_compare: bool = False  # 16-bit rows: setp.le.{f16x2,bf16x2} classify
    # instrumentation
    telemetry: bool = False

    @property
    def exact_only(self) -> bool:
        """Every row takes the exact select (census, then the radix select by key range): the
        configuration for a k that no stage x splits holds.  One CTA per row, no sample."""
        return self.short_cutoff >= EXACT_ONLY

    def validate(self, k: int, elems: Elements, shared_memory_limit: int) -> None:
        if self.threads not in (256, 512, 1024):
            raise ValueError(f"threads must be 256, 512 or 1024, got {self.threads}")
        if self.unroll * elems.per_vector > 32:
            raise ValueError(
                "unroll * elements per vector must fit the 32-bit dead mask"
            )
        if self.walk_width not in (1, 2, 4):
            raise ValueError("walk_width must be 1, 2 or 4")
        if self.aim not in ("tight", "wide"):
            raise ValueError(f"unknown aim policy {self.aim!r}")
        if self.sample_vectors not in (1, 2, 4):
            raise ValueError("sample_vectors must be 1, 2 or 4")
        if self.sample_bins not in (256, 512, 1024, 2048):
            raise ValueError("sample_bins must be 256, 512, 1024 or 2048")
        if self.sample_bins > 256 and (
            self.sample_bins > self.stage or self.sample_bins % self.threads
        ):
            raise ValueError(
                "a wide sample histogram must fit the stage and be a multiple of threads (the block-wide crossing)"
            )
        if (
            not 0.0 <= self.aim_margin <= 2.0
            or not 0.0 <= self.aim_margin_small <= 2.0
            or not 1.0 <= self.aim_z <= 6.0
        ):
            raise ValueError("aim margins in [0, 2] and aim_z in [1, 6]")
        if not 0.0 <= self.reprobe_below <= 1.0:
            raise ValueError("reprobe_below is a fraction of the row buffer in [0, 1]")
        if self.stage < 2048 or self.stage % 256:
            raise ValueError(
                "stage must be a multiple of 256 and at least 2048 (the census histogram aliases both stage halves)"
            )
        if (
            self.merge == "slab"
            and self.splits > 1
            and self.stage < max(768, 256 * self.splits)
        ):
            raise ValueError(
                "the slab merge's scratch needs a stage of at least max(768, 256 x splits)"
            )
        if self.exact_only and self.splits != 1:
            raise ValueError("the exact-only configuration runs one CTA per row")
        if self.lpt_order and self.splits != 1:
            raise ValueError("lpt_order permutes whole rows: one CTA per row")
        if self.register_arm and self.splits != 1:
            raise ValueError(
                "the register arm holds a whole row in one CTA's registers"
            )
        if self.register_words not in (4, 8, 16):
            raise ValueError("register_words must be 4, 8 or 16")
        if self.register_arm and 2 * (self.stage + 4) < self.register_bins() + 4:
            raise ValueError("the register arm's histogram does not fit the stage")
        if k >= 3 * self.stage * self.splits // 4 and not self.exact_only:
            raise ValueError(
                f"stage {self.stage} x {self.splits} too small for k={k}: k must sit below 3/4 of it (the balanced aim needs room on both sides)"
            )
        if self.merge not in ("cluster", "slab"):
            raise ValueError(f"unknown merge {self.merge!r}")
        if self.merge == "cluster" and self.splits not in (1, 2, 3, 4, 6, 8):
            raise ValueError(
                f"cluster splits must be one of 1, 2, 3, 4, 6, 8 (portable cluster sizes), got {self.splits}"
            )
        if self.merge == "slab" and self.splits not in (1, 2, 4, 8, 16, 32):
            raise ValueError(
                f"slab splits must be one of 1, 2, 4, 8, 16, 32, got {self.splits}"
            )
        # the tie stage need not hold k: a crossing bin wider than it sends the row to the exact
        # select (census, then the radix select by key range), which is exact for any k
        if self.tie_capacity < 2 * self.threads:
            raise ValueError(
                f"tie_capacity must be at least 2 * threads = {2 * self.threads}"
            )
        if self.tie_capacity > 8 * self.threads or self.tie_capacity % self.threads:
            raise ValueError(
                "tie_capacity must be a multiple of threads, at most 8 * threads (the radix select's candidate slots per thread)"
            )
        if self.packed_compare and elems.is_f32:
            raise ValueError("packed_compare applies to 16-bit rows only")
        need = self.shared_memory_bytes()
        if need * self.ctas_per_sm > shared_memory_limit:
            raise ValueError(
                f"{need} B x {self.ctas_per_sm} CTAs exceeds the {shared_memory_limit} B shared-memory budget"
            )

    def register_bins(self, words: int | None = None) -> int:
        """Coarse bins of the register arm's ``words`` tier: about a quarter of its row capacity
        in [1024, 4096] and at least the thread count (``register_config_for``'s rule)."""
        capacity = self.threads * (
            words or self.register_words
        )  # fp32 elements; 16-bit rows hold twice as many
        bins = 1024
        while bins < 4096 and bins * 4 < capacity:
            bins *= 2
        return max(bins, self.threads)

    def register_tiers(self) -> tuple:
        """Words per thread of the register arm's tiers, smallest first: a row takes the smallest
        tier that holds it (fewer words load and count fewer clamped duplicates: 1K rows at 512
        threads 5.5 us with 16 words, 3.8 with 4)."""
        return tuple(w for w in (4, 8, 16) if w <= self.register_words)

    def shared_memory_bytes(self) -> int:
        stage = 2 * (self.stage + 4) * 4
        ties = 2 * self.tie_capacity * 4
        return (
            stage + ties + 2 * 256 * 4 + (3 * (self.threads // 32) + 64) * 4 + 1024
        )  # + alignment slack

    def name(self) -> str:
        return "_".join(f"{f.name}{getattr(self, f.name)}" for f in fields(self))


class StreamingTopK:
    """A compiled streaming top-k for one (dtype, k, config); call it on (rows, N) inputs."""

    def __init__(
        self,
        dtype,
        k: int,
        config: StreamingConfig,
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
        self.aim_policy = aim_tight if config.aim == "tight" else aim_wide
        self.threads = config.threads
        self.warps = config.threads // 32
        self.next_n = next_n
        self.compress_ratio = compress_ratio
        self.return_values = return_values
        self.row_length = row_length  # None: the tensor's width; else the row is row_length columns from col_offset (paged arenas)
        self.col_offset = col_offset

    @cute.jit
    def exact_select(
        self,
        row_ptr,
        length,
        out_row,
        s_scratch,
        s_tie_keys,
        s_tie_idx,
        s_slots,
        s_slots_u32,
        s_result,
        tidx,
        check_constant: cutlass.Constexpr,
    ):
        """The exact path for one whole row (census, then the radix select over the rank-k
        bin's key range if it overflowed the tie stage; with ``check_constant`` a range pass
        first).  ``s_scratch``: at least 4096 Int32 (the dead stage, both halves).  Returns the
        arm taken (1 census, 3 radix) for the status word."""
        cfg = self.config
        return exact_select_row(
            self.elems,
            row_ptr,
            length,
            self.k,
            out_row,
            s_scratch,
            s_tie_keys,
            s_tie_idx,
            cfg.tie_capacity,
            cfg.ballot_limit,
            s_slots,
            s_slots_u32,
            s_result,
            tidx,
            self.threads,
            check_constant,
            cfg.telemetry,
        )

    @cute.jit
    def register_select(
        self,
        row_ptr,
        length,
        out_row,
        s_bins_all,
        s_tie_keys,
        s_tie_idx,
        s_slots,
        s_slots_u32,
        s_result,
        tidx,
        words: cutlass.Constexpr,
    ):
        """The register-resident kernel's phases on one whole row that fits ``words`` per
        thread: one load, the coarse census from registers, the wide crossing, the classify with
        a block scan, the tie select, and the radix refine by key range if the crossing bin
        overflows the tie stage.  ``s_bins_all``: ``register_bins(words) + 4`` Int32 (the dead
        stage).  Returns 1 when the tie select resolved the row, 0 when the radix refine did.
        """
        cfg = self.config
        elems = self.elems
        threads = cutlass.const_expr(self.threads)
        bins = cutlass.const_expr(cfg.register_bins(words))
        k = cutlass.const_expr(self.k)
        s_bins = s_bins_all + 4  # slot -1 takes out-of-range elements
        zero_bins(s_bins_all, bins + 4, tidx, threads)
        wordvals = load_row_words(
            row_ptr, length, tidx, threads, words, elems.log2_per_vector
        )
        cute.arch.barrier()
        if cutlass.const_expr(cfg.count_after_barrier):
            wordvals = words_after_barrier(wordvals, s_bins_all)
        packed_bins = count_coarse_bins(
            elems, wordvals, length, s_bins, tidx, threads, words, bins
        )
        cute.arch.barrier()
        crossing_wide_pair(
            s_bins,
            bins,
            cutlass.Int32(k),
            cutlass.Int32(k),
            s_slots,
            s_result,
            tidx,
            threads,
        )
        cut_bin = s_result[0]
        above = s_result[1]
        _winners, ties = classify_from_registers(
            elems,
            wordvals,
            packed_bins,
            cut_bin,
            out_row,
            s_tie_keys,
            s_tie_idx,
            cfg.tie_capacity,
            s_slots,
            tidx,
            threads,
            words,
            bins,
        )
        cute.arch.barrier()
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
        if ok == 0:
            kmin, kmax = key_range_in_bin(
                elems, wordvals, packed_bins, cut_bin, s_slots_u32, tidx, threads, words
            )
            cute.arch.barrier()
            radix_select_in_range(
                elems,
                row_ptr,
                length,
                cutlass.Int32(k) - above,
                out_row + above,
                kmin,
                kmax,
                s_bins,
                s_slots,
                s_result,
                tidx,
                threads,
            )
        return ok

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        lengths: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
        status: cute.Tensor,
        slab: cute.Tensor,
        counters: cute.Tensor,
        order: cute.Tensor,
    ):
        cfg = self.config
        elems = self.elems
        threads = cutlass.const_expr(self.threads)
        k = cutlass.const_expr(self.k)
        row_stride = cutlass.const_expr(x.shape[1])
        n_cols = cutlass.const_expr(
            self.row_length if self.row_length is not None else x.shape[1]
        )
        splits = cutlass.const_expr(cfg.splits)
        clustered = cutlass.const_expr(splits > 1 and cfg.merge == "cluster")
        slabbed = cutlass.const_expr(splits > 1 and cfg.merge == "slab")
        tidx, _, _ = cute.arch.thread_idx()
        row, rank_y, _ = cute.arch.block_idx()
        rows, _, _ = cute.arch.grid_dim()
        rank = cutlass.Int32(0)
        if cutlass.const_expr(clustered):
            rank = cluster_rank()
        if cutlass.const_expr(slabbed):
            rank = rank_y
        telemetry = cutlass.const_expr(cfg.telemetry)

        smem = SmemAllocator()
        # the survivor stage: (bits, index) pairs, 8 bytes per slot (filter_pass.stage_pair;
        # v0.1.31, one 64-bit shared access per survivor instead of two 32-bit ones into a key
        # array and an index array).  The exact fallback's 4096-bin census, the register arm's
        # bins, the lpt classes and the slab merge's scratch alias the dead stage as Int32:
        # s_keys is its first half, s_idx its second (the names survive from the split layout)
        s_stage = smem.allocate_array(
            cutlass.Int32, 2 * (cfg.stage + 4), byte_alignment=128
        )
        s_keys = s_stage
        s_idx = s_stage + (cfg.stage + 4)
        s_hist = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_merged = smem.allocate_array(cutlass.Int32, 256, byte_alignment=128)
        s_tie_keys = smem.allocate_array(
            cutlass.Uint32, cfg.tie_capacity, byte_alignment=128
        )
        s_tie_idx = smem.allocate_array(
            cutlass.Int32, cfg.tie_capacity, byte_alignment=128
        )
        s_slots_u32 = smem.allocate_array(
            cutlass.Uint32, 2 * self.warps, byte_alignment=128
        )
        s_slots = smem.allocate_array(cutlass.Int32, self.warps, byte_alignment=128)
        s_result = smem.allocate_array(cutlass.Int32, 32, byte_alignment=128)
        s_count = smem.allocate_array(cutlass.Int32, 8, byte_alignment=128)

        if cutlass.const_expr(cfg.pdl):
            wait_for_prior_grid()
        mark = cutlass.Int64(
            0
        )  # telemetry: after the PDL wait, so phases exclude the previous kernel's tail
        if telemetry:
            mark = read_clock64()
        # the sample probe for the launch-order row goes out first, in flight while the length
        # (and, with lpt_order, the whole ranking) loads; a permuted CTA reloads it
        if cutlass.const_expr(cfg.row_order):
            # the permutation from the prepare launch (or the caller): block b takes row
            # order[b].  Read before the probe so the probe goes to the real row (one dependent
            # load; probing the launch-order row and re-probing after cost a wasted round trip)
            row = order[row]
        launch_row = row
        probe = sample_probe(
            elems,
            x.iterator + cutlass.Int64(launch_row) * row_stride + self.col_offset,
            n_cols,
            tidx,
            threads,
            cfg.sample_vectors,
        )
        lpt = cutlass.Int32(0)
        if cutlass.const_expr(cfg.lpt_order):
            if rows <= cutlass.Int32(
                threads
            ):  # one row per ranking thread; wider grids keep launch order
                lpt = cutlass.Int32(1)
        if lpt == 1:
            # Longest rows first: the hardware scheduler's first wave then holds the long rows,
            # and the short rows it pairs beside them leave early so the long ones finish alone
            # (B200 64K b=256 k=2048 ragged, rows sorted on the host: 16.9 -> 14.6 us).  Every
            # CTA derives the same permutation from the lengths array, no prepare launch:
            # rows fall into three classes by length (at least half the row, at least a
            # quarter, the rest), one packed block scan gives each row its position within its
            # class and the class sizes, and block b takes position b of the concatenated
            # classes.  Rows within a class keep launch order.  A full rank by length (a
            # 128-bucket count table with per-chunk prefixes and warp shuffles) cost 2 us per
            # CTA, most of what the order saved; this is one global load, one scan, three
            # barriers.  rows <= threads (one row per thread; the policy caps rows at 512).
            block = row
            warp = tidx // 32
            lane_lt = cute.arch.lanemask_lt()
            cls = cutlass.Int32(3)  # sentinel past the last row
            if tidx < rows:
                my_len = effective_length(
                    lengths, tidx, n_cols, self.next_n, self.compress_ratio
                )
                cls = cutlass.Int32(2)
                if my_len * cutlass.Int32(4) >= cutlass.Int32(n_cols):
                    cls = cutlass.Int32(1)
                if my_len * cutlass.Int32(2) >= cutlass.Int32(n_cols):
                    cls = cutlass.Int32(0)
            # per-warp class counts by ballot (two barriers in all: publish, then the answer)
            in0 = cls == 0
            in1 = cls == 1
            in2 = cls == 2
            r0 = ballot_rank(in0, lane_lt)
            r1 = ballot_rank(in1, lane_lt)
            r2 = ballot_rank(in2, lane_lt)
            c0 = ballot_count(in0)
            c1 = ballot_count(in1)
            c2 = ballot_count(in2)
            s_cls = s_stage  # warps x 3 counts (the stage is dead here)
            if tidx % 32 == 0:
                s_cls[warp * 3] = c0
                s_cls[warp * 3 + 1] = c1
                s_cls[warp * 3 + 2] = c2
            cute.arch.barrier()
            p0 = cutlass.Int32(
                0
            )  # this warp's offset within each class, and the class totals
            p1 = cutlass.Int32(0)
            p2 = cutlass.Int32(0)
            n0 = cutlass.Int32(0)
            n1 = cutlass.Int32(0)
            for w in cutlass.range_constexpr(threads // 32):
                w0 = s_cls[w * 3]
                w1 = s_cls[w * 3 + 1]
                if cutlass.Int32(w) < warp:
                    p0 = p0 + w0
                    p1 = p1 + w1
                    p2 = p2 + s_cls[w * 3 + 2]
                n0 = n0 + w0
                n1 = n1 + w1
            want_cls = cutlass.Int32(0)
            want_pos = block
            if block >= n0:
                want_cls = cutlass.Int32(1)
                want_pos = block - n0
            if block >= n0 + n1:
                want_cls = cutlass.Int32(2)
                want_pos = block - n0 - n1
            my_pos = p0 + r0
            if cls == 1:
                my_pos = p1 + r1
            if cls == 2:
                my_pos = p2 + r2
            if cls < 3:
                if (cls == want_cls) & (my_pos == want_pos):
                    s_result[16] = tidx
            cute.arch.barrier()
            row = s_result[16]

        row_ptr = x.iterator + cutlass.Int64(row) * row_stride + self.col_offset
        out_row = out.iterator + cutlass.Int64(row) * k
        status_row = status.iterator + cutlass.Int64(row) * STATUS_WORDS
        probe_stale = cutlass.Int32(row != launch_row)
        length = effective_length(
            lengths, row, n_cols, self.next_n, self.compress_ratio
        )
        if cutlass.const_expr(cfg.reprobe_below > 0.0):
            # a row much shorter than its buffer keeps too few of the buffer probe's vectors
            # (the masked sample shrinks with the length: v0.1.24); below the threshold it pays
            # the dependent round trip of a fresh probe over its own length instead
            if length < cutlass.Int32(int(cfg.reprobe_below * n_cols)):
                probe_stale = cutlass.Int32(1)
        emitter = cutlass.Int32(
            0
        )  # 1 on the CTA that wrote this row's indices (for the values gather)

        # the two direct arms are rank 0's alone; peers of a cluster idle through them (no
        # cluster barrier or DSMEM access happens on these arms)
        if cutlass.Int32(k) >= length:
            if rank == 0:
                emitter = cutlass.Int32(1)
                for i in range(tidx, k, threads):
                    v = cutlass.Int32(-1)
                    if i < length:
                        v = cutlass.Int32(i)
                    out_row[i] = v
                if tidx == 0:
                    status_row[0] = cutlass.Int32(0)
                    status_row[1] = cutlass.Int32(0)
        else:
            reg_arm = cutlass.Int32(0)
            if cutlass.const_expr(cfg.register_arm):
                if length <= cutlass.Int32(
                    threads * cfg.register_words * elems.per_word
                ):
                    reg_arm = cutlass.Int32(1)
            if reg_arm == 1:
                # a row that fits the CTA's registers: one read of the row, the census from
                # registers, no sample and no survivor stage (walk-first's short-row path;
                # the census arm below paid two passes and a 4096-bin crossing for it).  The
                # smallest tier of words per thread that holds the row.
                emitter = cutlass.Int32(1)
                ok_reg = cutlass.Int32(1)
                tiers = cfg.register_tiers()
                taken = cutlass.Int32(0)
                for w in cutlass.range_constexpr(len(tiers)):
                    tier_words = cutlass.const_expr(tiers[w])
                    fits = length <= cutlass.Int32(
                        threads * tier_words * elems.per_word
                    )
                    if fits & (taken == 0):
                        taken = cutlass.Int32(1)
                        ok_reg = self.register_select(
                            row_ptr,
                            length,
                            out_row,
                            s_stage,
                            s_tie_keys,
                            s_tie_idx,
                            s_slots,
                            s_slots_u32,
                            s_result,
                            tidx,
                            tier_words,
                        )
                if tidx == 0:
                    status_row[0] = cutlass.Int32(1) - ok_reg
                    status_row[1] = cutlass.Int32(5)
            elif length <= cutlass.Int32(cfg.short_cutoff):
                if rank == 0:
                    emitter = cutlass.Int32(1)
                    arm = self.exact_select(
                        row_ptr,
                        length,
                        out_row,
                        s_keys,
                        s_tie_keys,
                        s_tie_idx,
                        s_slots,
                        s_slots_u32,
                        s_result,
                        tidx,
                        False,
                    )
                    if tidx == 0:
                        status_row[0] = cutlass.Int32(0)
                        status_row[1] = arm
            else:
                # this CTA's slice: equal vector-aligned chunks, the last one shorter
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
                # the sample: the launch probe's vectors inside this row (a shorter row masks
                # the rest rather than re-sampling: one dependent round trip per ragged row,
                # v0.1.24), or the full probe of a re-sampled permuted row
                samples = cutlass.Int32(threads * elems.per_vector * cfg.sample_vectors)
                if length != cutlass.Int32(n_cols):
                    if (
                        probe_stale == 0
                    ):  # a permuted CTA re-samples its real row in full
                        samples = valid_samples(
                            n_cols,
                            length,
                            threads,
                            cfg.sample_vectors,
                            elems.log2_per_vector,
                        )
                aim = self.aim_policy(
                    k,
                    length,
                    rows,
                    samples,
                    3 * cfg.stage * splits // 4,
                    cfg.aim_margin,
                    cfg.aim_z,
                    cfg.aim_floor_rows,
                    cfg.aim_margin_small,
                )
                th = Threshold(
                    *sample_threshold(
                        elems,
                        row_ptr,
                        length,
                        n_cols,
                        probe,
                        aim,
                        cfg.floor_multiple,
                        cfg.span_ext,
                        s_hist,
                        s_slots_u32,
                        s_result,
                        tidx,
                        threads,
                        cfg.telemetry,
                        cfg.sample_vectors,
                        cfg.sample_bins,
                        s_keys,
                        s_slots,
                        probe_stale,
                        samples,
                    )
                )
                if telemetry:
                    if tidx == 0:
                        status_row[3] = (read_clock64() - mark).to(cutlass.Int32)
                        if cutlass.const_expr(not slabbed):
                            for w in cutlass.range_constexpr(3):
                                status_row[8 + w] = s_result[12 + w]
                    mark = read_clock64()
                bar = th.bar
                scale = th.scale
                if tidx == 0:
                    s_count[0] = cutlass.Int32(0)  # a degenerate sample stages nothing
                    s_result[6] = cutlass.Int32(
                        0
                    )  # winner cursor (rank 0's is the row's)
                    s_result[7] = cutlass.Int32(0)  # tie cursor
                if th.degenerate == 0:
                    filter_pass(
                        elems,
                        row_ptr,
                        start,
                        count,
                        bar,
                        scale,
                        cfg.packed_compare,
                        cfg.stage,
                        s_count,
                        s_hist,
                        s_stage,
                        tidx,
                        threads,
                        cfg.unroll,
                        cfg.walk_width,
                    )
                else:
                    cute.arch.barrier()
                if telemetry:
                    if tidx == 0:
                        status_row[4] = (read_clock64() - mark).to(cutlass.Int32)
                    mark = read_clock64()
                survivors = s_count[0]
                ok = cutlass.Int32(0)
                finisher = cutlass.Int32(
                    1
                )  # the CTA that resolves the row and writes its status
                passes = cutlass.Int32(
                    1
                )  # filter passes over this slice (the slab merge repairs nothing)
                if cutlass.const_expr(slabbed):
                    # the row's verdict happens on the last arriver, after the others have left
                    slab_row = slab.iterator + cutlass.Int64(row) * slab_words_per_row(
                        splits, cfg.stage
                    )
                    slab_keys = slab_row
                    slab_idx = slab_row + splits * cfg.stage
                    tables = slab_row + 2 * splits * cfg.stage
                    finisher = publish_and_arrive(
                        elems,
                        rank,
                        splits,
                        bar,
                        scale,
                        cfg.stage,
                        s_count,
                        s_stage,
                        s_hist,
                        s_merged,
                        s_result,
                        slab_keys,
                        slab_idx,
                        tables,
                        counters.iterator + row,
                        tidx,
                        threads,
                    )
                    if telemetry:
                        if tidx == 0:
                            status_row[5] = (read_clock64() - mark).to(cutlass.Int32)
                        mark = read_clock64()
                    if finisher == 1:
                        ok = merge_slab(
                            elems,
                            k,
                            splits,
                            cfg.stage,
                            out_row,
                            s_keys,
                            s_idx,
                            s_merged,
                            s_tie_keys,
                            s_tie_idx,
                            cfg.tie_capacity,
                            cfg.ballot_limit,
                            s_slots,
                            s_result,
                            slab_keys,
                            slab_idx,
                            tables,
                            counters.iterator + row,
                            tidx,
                            threads,
                            cfg.telemetry,
                        )
                        survivors = s_result[8]
                        if telemetry:
                            if tidx == 0:
                                for w in cutlass.range_constexpr(4):
                                    status_row[8 + w] = s_result[12 + w]
                else:
                    if cutlass.const_expr(clustered):
                        bar, scale, survivors, ok, passes = verdict_and_repair_cluster(
                            elems,
                            row_ptr,
                            start,
                            count,
                            k,
                            bar,
                            scale,
                            th.floor_bar,
                            th.floor_scale,
                            th.degenerate,
                            cfg.packed_compare,
                            cfg.stage,
                            splits,
                            s_count,
                            s_hist,
                            s_stage,
                            s_result,
                            tidx,
                            threads,
                            cfg.unroll,
                            cfg.walk_width,
                        )
                    else:
                        bar, scale, survivors, ok, passes = verdict_and_repair(
                            elems,
                            row_ptr,
                            start,
                            count,
                            k,
                            bar,
                            scale,
                            th.floor_bar,
                            th.floor_scale,
                            th.degenerate,
                            cfg.packed_compare,
                            cfg.stage,
                            cfg.overflow_offset,
                            s_count,
                            s_hist,
                            s_stage,
                            s_result,
                            tidx,
                            threads,
                            cfg.unroll,
                            cfg.walk_width,
                        )
                    if telemetry:
                        if tidx == 0:
                            status_row[5] = (read_clock64() - mark).to(cutlass.Int32)
                        mark = read_clock64()
                    if ok == 1:
                        if cutlass.const_expr(clustered):
                            ok = emit_and_select_cluster(
                                elems,
                                k,
                                rank,
                                splits,
                                bar,
                                scale,
                                out_row,
                                s_stage,
                                s_hist,
                                s_merged,
                                s_tie_keys,
                                s_tie_idx,
                                cfg.tie_capacity,
                                cfg.ballot_limit,
                                s_count,
                                s_slots,
                                s_result,
                                tidx,
                                threads,
                            )
                        else:
                            ok = emit_and_select(
                                elems,
                                k,
                                survivors,
                                bar,
                                scale,
                                out_row,
                                s_stage,
                                s_hist,
                                s_tie_keys,
                                s_tie_idx,
                                cfg.tie_capacity,
                                cfg.ballot_limit,
                                s_slots,
                                s_result,
                                tidx,
                                threads,
                                cfg.scan_emit,
                                s_merged,
                                cfg.bin_cursors,  # s_merged: idle without a cluster
                            )
                    else:
                        if cutlass.const_expr(clustered):
                            cluster_sync()  # match the resolution's barrier so no CTA exits early
                    if rank != 0:
                        finisher = cutlass.Int32(0)
                if telemetry:
                    if tidx == 0:
                        status_row[6] = (read_clock64() - mark).to(cutlass.Int32)
                    mark = read_clock64()
                arm = cutlass.Int32(2)
                if finisher == 1:
                    emitter = cutlass.Int32(1)
                    if ok == 0:
                        arm = self.exact_select(
                            row_ptr,
                            length,
                            out_row,
                            s_keys,
                            s_tie_keys,
                            s_tie_idx,
                            s_slots,
                            s_slots_u32,
                            s_result,
                            tidx,
                            True,
                        )
                    if telemetry:
                        if tidx == 0:
                            status_row[7] = (read_clock64() - mark).to(cutlass.Int32)
                            if (
                                ok == 0
                            ):  # the exact select's sub-phases replace the sample's
                                for w in cutlass.range_constexpr(3):
                                    status_row[8 + w] = s_result[12 + w]
                    if tidx == 0:
                        status_row[0] = cutlass.Int32(1) - ok
                        status_row[1] = arm
                        status_row[2] = survivors
                        if cutlass.const_expr(not telemetry):
                            status_row[3] = (
                                passes  # filter passes over the slice (1, or 2 after an undershoot / overflow repair)
                            )
        if cutlass.const_expr(self.return_values):
            # the emitter wrote the ties and the fallback itself; in a cluster the peers' winners
            # preceded the cluster barrier, in a slab the last arriver copied everything: after
            # this block barrier every index of the row is visible to the emitter
            if emitter == 1:
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
        slab: cute.Tensor,
        counters: cute.Tensor,
        order: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        """Launch on the caller's stream (torch's current stream, so CUDA-graph capture and
        stream ordering with the surrounding torch work both hold)."""
        splits = self.config.splits
        if cutlass.const_expr(splits > 1 and self.config.merge == "cluster"):
            self.kernel(x, lengths, out, values, status, slab, counters, order).launch(
                grid=(x.shape[0], splits, 1),
                block=(self.threads, 1, 1),
                cluster=(1, splits, 1),
                min_blocks_per_mp=self.config.ctas_per_sm,
                use_pdl=self.config.pdl,
                stream=stream,
            )
        else:
            self.kernel(x, lengths, out, values, status, slab, counters, order).launch(
                grid=(x.shape[0], splits, 1),
                block=(self.threads, 1, 1),
                min_blocks_per_mp=self.config.ctas_per_sm,
                use_pdl=self.config.pdl,
                stream=stream,
            )


_slabs: dict = {}


def _slab_workspace(device, rows: int, config: StreamingConfig):
    """Per-(device, stream, rows, splits, stage) slab and zeroed arrival counters, kept alive so
    the counters' self-reset carries from one launch to the next.  Keyed by the current stream
    because launches on one stream are ordered and launches on two are not: two streams running
    the same shape at once would otherwise count each other's arrivals and merge each other's
    slab segments (see ``dispatch/workspace.py``)."""
    if config.merge != "slab" or config.splits == 1:
        return _placeholder_slab(device)
    key = (
        device,
        torch.cuda.current_stream(device).cuda_stream,
        rows,
        config.splits,
        config.stage,
    )
    if key not in _slabs:
        words = rows * slab_words_per_row(config.splits, config.stage)
        _slabs[key] = (
            torch.empty(words, device=device, dtype=torch.int32),
            torch.zeros(rows, device=device, dtype=torch.int32),
        )
    return _slabs[key]


def _placeholder_slab(device):
    """One-element slab and counters for configurations that do not merge through the slab; the
    kernel never touches them, so sharing them across streams is safe."""
    key = (device, "none")
    if key not in _slabs:
        _slabs[key] = (
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.zeros(1, device=device, dtype=torch.int32),
        )
    return _slabs[key]


def _caller_workspace(
    workspace: torch.Tensor, x: torch.Tensor, rows: int, config: StreamingConfig
):
    """Buffers carved from a caller-owned workspace: status, slab and counters (placeholders when
    the configuration does not merge through the slab) and the arena buffer for a copy."""
    from ..dispatch.workspace import carve, workspace_layout
    from .layout import arena_bytes

    ws = carve(
        workspace, workspace_layout("streaming", config, rows, arena_bytes(x)), x.device
    )
    if ws.slab is None:
        slab, counters = _placeholder_slab(x.device)
    else:
        slab, counters = ws.slab, ws.counters
        assert counters is not None
        counters.zero_()  # the kernel needs zero arrivals at launch; the caller's memory holds anything
    return ws.status, slab, counters, ws.arena, ws.order


_orders: dict = {}


def _order_buffer(device, rows: int) -> torch.Tensor:
    """Per-(device, stream, rows) buffer for the prepare launch's permutation (stream-keyed for
    the same reason as the slab: two streams must not share it)."""
    key = (device, torch.cuda.current_stream(device).cuda_stream, rows)
    if key not in _orders:
        _orders[key] = torch.empty(rows, device=device, dtype=torch.int32)
    return _orders[key]


_compiled: dict = {}


def topk_streaming(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    config: StreamingConfig | None = None,
    out: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    next_n: int = 1,
    compress_ratio: int = 1,
    workspace: torch.Tensor | None = None,
    row_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Indices of the k largest elements of each row of ``x`` (rows, N), as an int32 (rows, k)
    tensor.  ``lengths`` (rows // next_n, int32) limits each row to its first elements: row r
    sees ``(lengths[r // next_n] - next_n + r % next_n + 1) // compress_ratio`` of them (the
    speculative-decode stride and the KV-index compression; both default to 1, when it is
    ``lengths[r]``).  Rows shorter than k are padded with -1.  NaN ranks above +inf, as in
    torch.  Order within a row is unspecified.  ``out``, ``status`` (rows * STATUS_WORDS,
    int32) and ``values`` (rows, k, x's dtype: the selected elements, -inf in padding) are
    written in place when given, so a caller in a hot loop allocates nothing; values are only
    produced when ``values`` is passed.  ``config`` defaults to the dispatcher's choice for the
    device and problem.  ``workspace`` (a CUDA byte tensor of at least
    ``dispatch.workspace.workspace_bytes(x, k)`` bytes) supplies the status words, the slab
    merge's buffers and the arena for a misaligned copy, so the call allocates nothing; without
    it those come from per-(device, stream, shape) caches (see ``dispatch/workspace.py``).
    ``row_order`` (rows, int32, a permutation of the row indices): the order in which CTAs
    take rows, for configurations with ``config.row_order``; without it the prepare launch
    (``phases/row_order.py``) ranks the rows longest-first on the device first.  The result
    does not depend on the order.
    """
    from ...dispatch.device import device_facts

    from ..dispatch.streaming_policy import streaming_config_for
    from ..phases.row_order import rank_rows

    assert x.dtype in _DTYPES
    check_layout(x)
    rows, n = x.shape
    facts = device_facts(x.device)
    if config is None:
        config = streaming_config_for(facts, x.dtype, k, n, rows)
    if row_order is not None and not config.row_order and config.splits == 1:
        config = StreamingConfig(
            **{**config.__dict__, "row_order": True}
        )  # a supplied order selects the ordered kernel
    arena = None
    ws_order = None
    if workspace is not None:
        ws_status, slab, counters, arena, ws_order = _caller_workspace(
            workspace, x, rows, config
        )
        if status is None:
            status = ws_status
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
    if config.row_order:
        if row_order is not None:
            if (
                row_order.dtype != torch.int32
                or row_order.shape != (rows,)
                or not row_order.is_contiguous()
                or row_order.device != x.device
            ):
                raise ValueError(
                    f"row_order must be a contiguous int32 tensor of shape ({rows},) on {x.device}"
                )
            order = row_order
        else:
            order = ws_order if ws_order is not None else _order_buffer(x.device, rows)
            rank_rows(lengths, order, n, next_n, compress_ratio, config.pdl)
    else:
        order = lengths  # unused by the kernel; any int32 tensor keeps the argument list fixed
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
        from_dlpack(order),
        stream,
    )
    if key not in _compiled:
        kern = StreamingTopK(
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
