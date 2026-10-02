"""Kernel 1, register-resident top-k: rows that fit one CTA's registers (up to 16K elements).

Composition: load the row once into registers, coarse histogram from registers, rank-k
crossing, classify from registers into the output and the tie stage, exact tie select.  A
crossing bin holding more ties than the stage (constant rows, one-bin rows) hands the row to
the exact fallback, which re-reads the row from L2.

Shape rules (all measured on B200, see ``register_config_for``): 1024 threads at one CTA per
SM for batches up to the SM count; 512 threads at two per SM when the batch is wider and the
row fits; the histogram has N / 4 bins (1024 to 4096) so its zeroing and crossing scale with
the row.
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

from ...block.crossing import crossing_wide_pair
from ...device.launch import release_dependent_grid, wait_for_prior_grid
from ...device.timers import read_clock64

from ..phases.census import COARSE_BINS
from ..phases.elements import Elements
from ..phases.fallback import radix_select_in_range
from ..phases.register_row import (
    classify_from_registers,
    count_coarse_bins,
    key_range_in_bin,
    load_row_words,
    words_after_barrier,
    zero_bins,
)
from ..phases.resolve import _select_ties
from ..phases.varlen import effective_length, gather_values
from .layout import arena_view, check_layout

__all__ = [
    "RegisterConfig",
    "RegisterTopK",
    "topk_register",
    "register_config_for",
    "STATUS_WORDS",
]

_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
STATUS_WORDS = 8  # [0] fallback taken, [1] arm (0 identity, 1 registers, 3 radix), [2] ties, [3..6] phase clocks


@dataclass(frozen=True)
class RegisterConfig:
    threads: int = 1024
    words_per_thread: int = 16  # 4, 8 or 16 (one to four 16-byte vectors)
    ctas_per_sm: int = 1
    bins: int = 4096  # coarse histogram bins: 1024, 2048 or 4096
    tie_capacity: int = 2048
    ballot_limit: int = 128
    pdl: bool = False
    count_after_barrier: bool = True  # pin the count below the zeroing barrier (words_after_barrier); a device fact
    telemetry: bool = False

    def row_capacity(self, elems: Elements) -> int:
        """Longest row this configuration holds in registers."""
        return self.threads * self.words_per_thread * elems.per_word

    @property
    def bin_shift(self) -> int:
        """Right shift from a census bin (4096) to this histogram's bin."""
        return {1024: 2, 2048: 1, 4096: 0}[self.bins]

    def validate(self, k: int, elems: Elements, shared_memory_limit: int) -> None:
        if self.threads not in (256, 512, 1024):
            raise ValueError(f"threads must be 256, 512 or 1024, got {self.threads}")
        if self.words_per_thread not in (4, 8, 16):
            raise ValueError(
                f"words_per_thread must be 4, 8 or 16, got {self.words_per_thread}"
            )
        if self.bins not in (1024, 2048, 4096) or self.bins < self.threads:
            raise ValueError(
                f"bins must be 1024, 2048 or 4096 and at least the thread count, got {self.bins}"
            )
        # the tie stage need not hold k: a crossing bin wider than it sends the row to the radix
        # select over the bin's key range, which streams the row and is exact for any k
        if (
            self.tie_capacity < 2 * self.threads
            or self.tie_capacity > 8 * self.threads
            or self.tie_capacity % self.threads
        ):
            raise ValueError(
                f"tie_capacity must be a multiple of threads in [2 * threads, 8 * threads] = [{2 * self.threads}, {8 * self.threads}]"
            )
        if k >= self.row_capacity(elems):
            raise ValueError(
                f"k={k} must be below the row capacity {self.row_capacity(elems)}"
            )
        if self.shared_memory_bytes() * self.ctas_per_sm > shared_memory_limit:
            raise ValueError(
                f"{self.shared_memory_bytes()} B x {self.ctas_per_sm} CTAs exceeds the {shared_memory_limit} B budget"
            )

    def shared_memory_bytes(self) -> int:
        # the fallback's census needs 4096 bins whatever the kernel's own bin count
        return (
            (max(self.bins, COARSE_BINS) + 4) * 4
            + 2 * self.tie_capacity * 4
            + (self.threads // 32 + 32) * 4
            + 512
        )

    def name(self) -> str:
        return "_".join(f"{f.name}{getattr(self, f.name)}" for f in fields(self))


class RegisterTopK:
    def __init__(
        self,
        dtype,
        k: int,
        config: RegisterConfig,
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
        self.row_length = row_length  # None: the tensor's width; else the row is row_length columns from col_offset (paged arenas)
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
        k = cutlass.const_expr(self.k)
        row_stride = cutlass.const_expr(x.shape[1])
        n_cols = cutlass.const_expr(
            self.row_length if self.row_length is not None else x.shape[1]
        )
        telemetry = cutlass.const_expr(cfg.telemetry)
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        smem = SmemAllocator()
        s_bins_all = smem.allocate_array(
            cutlass.Int32, max(bins, COARSE_BINS) + 4, byte_alignment=128
        )
        s_bins = (
            s_bins_all + 4
        )  # slot -1 (index 3 of the allocation) takes out-of-range elements
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
        row_ptr = x.iterator + cutlass.Int64(row) * row_stride + self.col_offset
        out_row = out.iterator + cutlass.Int64(row) * k
        status_row = status.iterator + cutlass.Int64(row) * STATUS_WORDS
        # the row's words are loaded before the length is known: every vector of the row buffer
        # is in-bounds memory whatever the length, and only the count of valid elements depends
        # on it, so the two global latencies overlap instead of adding (the length load alone
        # is a dependent ~1 us round trip at the head of the kernel)
        wordvals = load_row_words(
            row_ptr, cutlass.Int32(n_cols), tidx, threads, words, elems.log2_per_vector
        )
        length = effective_length(
            lengths, row, n_cols, self.next_n, self.compress_ratio
        )

        if cutlass.Int32(k) >= length:
            for i in range(tidx, k, threads):
                v = cutlass.Int32(-1)
                if i < length:
                    v = cutlass.Int32(i)
                out_row[i] = v
            if tidx == 0:
                status_row[0] = cutlass.Int32(0)
                status_row[1] = cutlass.Int32(0)
        else:
            zero_bins(s_bins_all, bins + 4, tidx, threads)
            cute.arch.barrier()
            if cutlass.const_expr(cfg.count_after_barrier):
                wordvals = words_after_barrier(wordvals, s_bins_all)
            packed_bins = count_coarse_bins(
                elems, wordvals, length, s_bins, tidx, threads, words, bins
            )
            cute.arch.barrier()
            if telemetry:
                if tidx == 0:
                    status_row[3] = (read_clock64() - mark).to(cutlass.Int32)
                mark = read_clock64()
            # (a one-warp two-level crossing behind one barrier measured 0.2-0.9 us SLOWER
            # than this block form on every register cell: docs/measured-worse.md, 2026-09-18)
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
            if telemetry:
                if tidx == 0:
                    status_row[4] = (read_clock64() - mark).to(cutlass.Int32)
                mark = read_clock64()
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
            if telemetry:
                if tidx == 0:
                    status_row[5] = (read_clock64() - mark).to(cutlass.Int32)
                mark = read_clock64()
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
            arm = cutlass.Int32(1)
            if ok == 0:
                # the crossing bin overflowed the tie stage; everything above it is already in
                # the output, so the exact select runs on that bin alone, by its key range
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
                cute.arch.barrier()  # s_bins and s_result are reused below
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
                arm = cutlass.Int32(3)
            if telemetry:
                if tidx == 0:
                    status_row[6] = (read_clock64() - mark).to(cutlass.Int32)
            if tidx == 0:
                status_row[0] = cutlass.Int32(1) - ok
                status_row[1] = arm
                status_row[2] = ties
        if cutlass.const_expr(self.return_values):
            cute.arch.barrier()  # every output index of the row is written
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
        self.kernel(x, lengths, out, values, status).launch(
            grid=(x.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            min_blocks_per_mp=self.config.ctas_per_sm,
            use_pdl=self.config.pdl,
            stream=stream,
        )


_compiled: dict = {}


def _words_for(threads: int, n: int, per_word: int) -> int | None:
    for words in (4, 8, 16):
        if n <= threads * words * per_word:
            return words
    return None


def register_config_for(
    facts, dtype: torch.dtype, k: int, n: int, rows: int = 0
) -> RegisterConfig:
    """The dispatcher's shape for a batch of ``rows`` rows of ``n`` elements.

    * 1024 threads, one CTA per SM, the fewest words per thread that hold the row.  Two CTAs
      per SM at 1024 threads cap registers at 32 and spilled (8.7 vs 7.3 us at 16K b=64).
    * Batches wider than the SM count: 512 threads at two per SM when the row fits 512 x 16
      words (64 registers each, no spill), so the batch runs in one wave per two rows per SM:
      4K b=256 6.8 -> 6.0 us, 8K b=256 8.7 -> 8.3, 16K bf16 b=256 11.6 -> 10.7.
    * Histogram bins N / 4 clamped to [1024, 4096]: expected ties in the crossing bin scale
      with N / bins, and fewer bins cost less to zero and to scan.
    * Tiny rows (one 16-byte vector per thread at 256 or 512 threads: up to 1K / 2K fp32,
      2K / 4K 16-bit) run on the fewest threads that hold the row at four words.  At 1024
      threads three quarters of the warps loaded duplicates of the tail vector and every
      barrier, block scan and the tie select ran over 32 warps: 1K b=1 on B200 2.71 -> 2.32 us,
      H100 2.68 -> 2.26, Rubin 2.03 -> 1.77, RTX 5080 2.83 -> 2.27 (v0.1.22; gvr_2's register
      kernel runs 256 threads here).  More words on fewer threads does not pay above that:
      4K rows at 512 x 8 or 256 x 16 words measured 1-20% slower than 1024 x 4 everywhere.
      Wide batches of tiny rows keep two CTAs per SM, four where the batch is more than two
      waves of SMs (RTX 5080 1K b=256: 5.48 -> 4.06).
    """
    per_word = 1 if dtype == torch.float32 else 2
    # k >= n rows are the identity (the kernel's direct arm writes 0..length-1 and -1 padding);
    # the configuration only needs its register capacity above k, so size it for max(n, k + 1)
    need = max(n, k + 1)
    threads = 1024
    for t in (256, 512):  # the fewest threads that hold the row at one vector each
        if need <= t * 4 * per_word:
            threads = t
            break
    if (
        threads == 1024
        and rows > facts.sm_count
        and k <= 2048
        and _words_for(512, need, per_word) is not None
    ):
        threads = 512
    words = _words_for(threads, need, per_word)
    if words is None:
        raise ValueError(f"row length {n} exceeds the register-resident capacity")
    # a multiple of threads (candidate slots per thread), holding k where it can; beyond eight
    # slots per thread the crossing bin takes the radix refine when it does not fit
    tie_capacity = min(8 * threads, max(2 * threads, -(-k // threads) * threads))
    bins = 1024
    while bins < 4096 and bins * 4 < n:
        bins *= 2
    bins = max(bins, threads)
    ctas_per_sm = 1
    if threads == 512:
        ctas_per_sm = 2
    elif threads == 256:
        ctas_per_sm = 4 if rows > 2 * facts.sm_count else 2
    cfg = RegisterConfig(
        threads=threads,
        words_per_thread=words,
        ctas_per_sm=ctas_per_sm,
        bins=bins,
        tie_capacity=tie_capacity,
        pdl=facts.supports_pdl,
        count_after_barrier=facts.staggered_count,
    )
    while (
        cfg.ctas_per_sm > 1
        and cfg.ctas_per_sm * cfg.shared_memory_bytes() > facts.shared_memory_optin
    ):
        cfg = RegisterConfig(**{**cfg.__dict__, "ctas_per_sm": cfg.ctas_per_sm // 2})
    return cfg


def topk_register(
    x: torch.Tensor,
    k: int,
    lengths: torch.Tensor | None = None,
    config: RegisterConfig | None = None,
    out: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    next_n: int = 1,
    compress_ratio: int = 1,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Indices of the k largest of each row of ``x`` (rows, N <= 16K), same contract as
    ``topk_streaming`` (``workspace`` holds the status words and a misaligned copy's arena)."""
    from ...dispatch.device import device_facts

    from ..dispatch.workspace import carve, workspace_layout
    from .layout import arena_bytes

    assert x.dtype in _DTYPES
    check_layout(x)
    rows, n = x.shape
    facts = device_facts(x.device)
    if config is None:
        config = register_config_for(facts, x.dtype, k, n, rows)
    capacity = config.row_capacity(Elements.of(_DTYPES[x.dtype]))
    if n > capacity:
        raise ValueError(
            f"row length {n} exceeds this configuration's register capacity {capacity}"
        )
    arena = None
    if workspace is not None:
        ws = carve(
            workspace,
            workspace_layout("register", config, rows, arena_bytes(x)),
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
        kern = RegisterTopK(
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


_placeholders: dict = {}


def _no_values(device, dtype) -> torch.Tensor:
    """A one-element stand-in for the values tensor when none is requested (never written)."""
    key = (device, dtype)
    if key not in _placeholders:
        _placeholders[key] = torch.empty(1, 1, device=device, dtype=dtype)
    return _placeholders[key]
