# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Exact BF16 selection from per-warp sparse candidate banks."""

import functools
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.nvgpu import cpasync
from ... import memory as indexer_memory
from ...pdl import pdl_prologue
from ...api import DTYPES
from ..selection import score_key, candidate_score, candidate_index
from ..small_crossing import refine_quantized
from ..candidate_streaming import select_stream


@cute.jit
def numeric_bucket(score, lower, inverse_span, bins: cutlass.Constexpr = 1024):
    bucket = cutlass.Int32(0)
    if inverse_span >= 0.0:
        normalized = (score - lower) * inverse_span
        normalized = min(
            cutlass.Float32(bins - 1), max(cutlass.Float32(0.0), normalized)
        )
        bucket = bins - 1 - cutlass.Int32(normalized)
    if score != score:
        bucket = cutlass.Int32(bins - 1)
    return bucket


class SparseSelector:
    def __init__(self, logical_only=False, pdl=False, cache_entries=128, banks=16):
        if banks != 16:
            raise ValueError("sparse selection requires 16 banks (128 threads)")
        self.banks = banks
        self.threads = banks * 8
        self.cache_entries = cache_entries
        self.logical_only = logical_only
        self.pdl = pdl

    @cute.jit
    def __call__(
        self,
        candidates,
        counts,
        starts,
        ends,
        out,
        thresholds,
        samples,
        recovery_queue,
        blocks,
        logical,
        stream: cuda.CUstream,
    ):
        self.kernel(
            candidates,
            counts,
            starts,
            ends,
            out,
            thresholds,
            samples,
            recovery_queue,
            blocks,
            logical,
        ).launch(
            grid=(out.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=9,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        candidates,
        counts,
        starts,
        ends,
        out,
        thresholds,
        samples,
        recovery_queue,
        blocks,
        logical,
    ):
        pdl_prologue(self.pdl)
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        lane = cute.arch.lane_idx()
        warp = tid // 32
        amount = cutlass.Int32(candidates[row, lane])
        overflow = cute.arch.vote_ballot_sync(amount > (candidates.shape[1] - 32) // 32)
        amount = cute.arch.warp_redux_sync(amount, "add")
        if overflow != 0:
            amount = cutlass.Int32(-1)
        if tid == 0:
            counts[row] = amount
        early_n = amount
        early_required = min(512, ends[row] - starts[row])
        early_tau = thresholds[row]
        early_upper = cutlass.Float32(samples[row, samples.shape[1] - 30])
        early_count = cutlass.Int32(candidates[row, tid // 8])
        # Invalid proposal metadata falls back to exact K4 scoring.
        if early_upper != early_upper:
            early_n = cutlass.Int32(-1)
            if tid == 0:
                counts[row] = early_n
        if (
            tid == 0
            and early_required == 512
            and ends[row] - starts[row] > 512
            and early_n < early_required
        ):
            position = cute.arch.atomic_add(
                recovery_queue.iterator.llvm_ptr,
                cutlass.Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            recovery_queue[position + 1] = row
        mem = indexer_memory.SmemAllocator()
        cached = mem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout(self.banks * (self.cache_entries + 8)),
            byte_alignment=128,
        )
        hist = mem.allocate_tensor(
            cutlass.Int32, cute.make_layout(2048), byte_alignment=128
        )
        ck = cute.make_tensor(
            cute.recast_ptr(hist.iterator + 1024, dtype=cutlass.Uint32),
            cute.make_layout(128),
        )
        ci = cute.make_tensor(hist.iterator + 1152, cute.make_layout(128))
        state = cute.make_tensor(hist.iterator + 1280, cute.make_layout(8))
        staged = mem.allocate_tensor(
            cutlass.Int32, cute.make_layout(512), byte_alignment=128
        )
        n = early_n
        required = early_required
        if ends[row] - starts[row] <= 512:
            for i in cutlass.range(tid, 512, self.threads):
                value = cutlass.Int32(-1)
                if i < required:
                    value = starts[row] + i
                if cutlass.const_expr(not self.logical_only):
                    out[row, i] = value
                mapped = cutlass.Int32(-1)
                if value >= 0:
                    mapped = blocks[row, value // 8] * 8 + value % 8
                logical[row, i] = mapped
        elif n >= required:
            bank = tid // 8
            local = tid % 8
            bank_count = early_count
            threshold_value = early_tau
            span = (candidates.shape[1] - 32) // 32
            copy_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(), cutlass.Int32, num_bits_per_copy=128
            )
            copy_entries = 4
            for group in cutlass.range_constexpr(
                self.cache_entries // 8 // copy_entries
            ):
                offset = group * (8 * copy_entries) + local * copy_entries
                if offset < bank_count:
                    src = cute.make_tensor(
                        (
                            candidates.iterator
                            + cutlass.Int32(row) * cutlass.Int32(candidates.stride[0])
                            + 32
                            + bank * span
                            + offset
                        ).align(16),
                        cute.make_layout(copy_entries),
                    )
                    dst = cute.make_tensor(
                        (
                            cached.iterator + bank * (self.cache_entries + 8) + offset
                        ).align(16),
                        cute.make_layout(copy_entries),
                    )
                    cute.copy(copy_atom, src, dst)
            cute.arch.cp_async_commit_group()
            upper = early_upper
            sample_max = cutlass.Float32(samples[row, samples.shape[1] - 32])
            prefix_meta = cute.make_rmem_tensor(
                (self.cache_entries // 8,), cutlass.Uint32
            )
            cute.arch.cp_async_wait_group(0)
            crossing = cutlass.Int32(0)
            above = cutlass.Int32(0)
            m = cutlass.Int32(0)
            inverse_span = cutlass.Float32(-1.0)
            retry = cutlass.Boolean(True)
            while retry:
                span_value = upper - threshold_value
                inverse_span = cutlass.Float32(-1.0)
                if (span_value > 0.0) & (span_value < float("inf")):
                    inverse_span = cutlass.Float32((self.threads * 2) - 1) / span_value
                hist[tid * 2] = 0
                hist[tid * 2 + 1] = 0
                cute.arch.barrier(barrier_id=0, number_of_threads=self.threads)
                for j in cutlass.range_constexpr(self.cache_entries // 8):
                    offset = j * 8 + local
                    prefix_meta[j] = cutlass.Uint32(0)
                    if offset < bank_count:
                        entry = cached[bank * (self.cache_entries + 8) + offset]
                        bucket = numeric_bucket(
                            candidate_score(entry, cutlass.BFloat16, True),
                            threshold_value,
                            inverse_span,
                            (self.threads * 2),
                        )
                        prefix_meta[j] = cutlass.Uint32(bucket) << 20 | cutlass.Uint32(
                            candidate_index(entry, True)
                        )
                        cute.arch.atomic_add(
                            (hist.iterator + bucket).llvm_ptr,
                            cutlass.Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                entry_buffer = cute.make_rmem_tensor((4,), cutlass.Int32)
                for base in cutlass.range(self.cache_entries, bank_count, 8 * 4):
                    for j in cutlass.range_constexpr(4):
                        offset = base + j * 8 + local
                        entry_buffer[j] = cutlass.Int32(0)
                        if offset < bank_count:
                            if offset < self.cache_entries:
                                entry_buffer[j] = cached[
                                    bank * (self.cache_entries + 8) + offset
                                ]
                            else:
                                entry_buffer[j] = candidates[
                                    row, 32 + bank * span + offset
                                ]
                    for j in cutlass.range_constexpr(4):
                        offset = base + j * 8 + local
                        if offset < bank_count:
                            entry = entry_buffer[j]
                            bucket = numeric_bucket(
                                candidate_score(entry, cutlass.BFloat16, True),
                                threshold_value,
                                inverse_span,
                                (self.threads * 2),
                            )
                            cute.arch.atomic_add(
                                (hist.iterator + bucket).llvm_ptr,
                                cutlass.Int32(1),
                                sem="relaxed",
                                scope="cta",
                            )
                cute.arch.barrier(barrier_id=0, number_of_threads=self.threads)
                count0 = hist[tid * 2]
                count1 = hist[tid * 2 + 1]
                count = count0 + count1
                prefix = count
                for step in cutlass.range_constexpr(5):
                    delta = 1 << step
                    other = cute.arch.shuffle_sync_up(
                        prefix, delta, mask=-1, mask_and_clamp=0
                    )
                    if lane >= delta:
                        prefix += other
                if lane == 31:
                    hist[1312 + warp] = prefix
                cute.arch.barrier(barrier_id=0, number_of_threads=self.threads)
                prior = cutlass.Int32(0)
                if lane < warp:
                    prior = hist[1312 + lane]
                prior = cute.arch.warp_redux_sync(prior, "add")
                cumulative = prior + prefix
                hist[tid * 2] = cumulative - count
                hist[tid * 2 + 1] = cumulative - count1
                if (cumulative - count1 >= required) & (cumulative - count < required):
                    state[0] = tid * 2
                    state[1] = cumulative - count
                    state[2] = count0
                if (cumulative >= required) & (cumulative - count1 < required):
                    state[0] = tid * 2 + 1
                    state[1] = cumulative - count1
                    state[2] = count1
                if tid == 0:
                    state[3] = 0
                cute.arch.barrier(barrier_id=0, number_of_threads=self.threads)
                crossing = state[0]
                above = state[1]
                m = state[2]
                retry = (
                    (crossing == 0) & (m > 128) & (m > required) & (upper < sample_max)
                )
                if retry:
                    upper = sample_max
            need = required - above
            whole = m == need
            if (m <= 128) | whole:
                for i in cutlass.range(required + tid, 512, self.threads):
                    if cutlass.const_expr(not self.logical_only):
                        out[row, i] = -1
                    logical[row, i] = -1
                for j in cutlass.range_constexpr(self.cache_entries // 8):
                    offset = j * 8 + local
                    if offset < bank_count:
                        meta = prefix_meta[j]
                        index = cutlass.Int32(meta & cutlass.Uint32(1048575))
                        bucket = cutlass.Int32(meta >> 20)
                        if (bucket < crossing) | whole & (bucket == crossing):
                            pos = cute.arch.atomic_add(
                                (hist.iterator + bucket).llvm_ptr,
                                cutlass.Int32(1),
                                sem="relaxed",
                                scope="cta",
                            )
                            staged[pos] = index
                        elif bucket == crossing:
                            entry = cached[bank * (self.cache_entries + 8) + offset]
                            key = score_key(
                                candidate_score(entry, cutlass.BFloat16, True)
                            )
                            pos = cute.arch.atomic_add(
                                (state.iterator + 3).llvm_ptr,
                                cutlass.Int32(1),
                                sem="relaxed",
                                scope="cta",
                            )
                            ck[pos] = key
                            ci[pos] = index
                entry_buffer = cute.make_rmem_tensor((4,), cutlass.Int32)
                for base in cutlass.range(self.cache_entries, bank_count, 8 * 4):
                    for j in cutlass.range_constexpr(4):
                        offset = base + j * 8 + local
                        entry_buffer[j] = cutlass.Int32(0)
                        if offset < bank_count:
                            if offset < self.cache_entries:
                                entry_buffer[j] = cached[
                                    bank * (self.cache_entries + 8) + offset
                                ]
                            else:
                                entry_buffer[j] = candidates[
                                    row, 32 + bank * span + offset
                                ]
                    for j in cutlass.range_constexpr(4):
                        offset = base + j * 8 + local
                        if offset < bank_count:
                            entry = entry_buffer[j]
                            key = score_key(
                                candidate_score(entry, cutlass.BFloat16, True)
                            )
                            bucket = numeric_bucket(
                                candidate_score(entry, cutlass.BFloat16, True),
                                threshold_value,
                                inverse_span,
                                (self.threads * 2),
                            )
                            if (bucket < crossing) | whole & (bucket == crossing):
                                pos = cute.arch.atomic_add(
                                    (hist.iterator + bucket).llvm_ptr,
                                    cutlass.Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                                staged[pos] = candidate_index(entry, True)
                            elif bucket == crossing:
                                pos = cute.arch.atomic_add(
                                    (state.iterator + 3).llvm_ptr,
                                    cutlass.Int32(1),
                                    sem="relaxed",
                                    scope="cta",
                                )
                                ck[pos] = key
                                ci[pos] = candidate_index(entry, True)
                cute.arch.barrier(barrier_id=0, number_of_threads=self.threads)
                if not whole:
                    refine_quantized(ck, ci, m, need, above, tid, staged, self.threads)
                cute.arch.barrier(barrier_id=0, number_of_threads=self.threads)
                write_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
                )
                for col in cutlass.range(tid * 4, required, self.threads * 4):
                    if col + 3 < required:
                        values = cute.make_rmem_tensor((4,), cutlass.Int32)
                        mapped = cute.make_rmem_tensor((4,), cutlass.Int32)
                        src = cute.make_tensor(
                            (staged.iterator + col).align(16), cute.make_layout(4)
                        )
                        dst = cute.make_tensor(
                            (
                                out.iterator + cutlass.Int64(row) * out.stride[0] + col
                            ).align(16),
                            cute.make_layout(4),
                        )
                        logical_dst = cute.make_tensor(
                            (
                                logical.iterator
                                + cutlass.Int64(row) * logical.stride[0]
                                + col
                            ).align(16),
                            cute.make_layout(4),
                        )
                        cute.copy(write_atom, src, values)
                        if cutlass.const_expr(not self.logical_only):
                            cute.copy(write_atom, values, dst)
                        for j in cutlass.range_constexpr(4):
                            pick = values[j]
                            mapped[j] = blocks[row, pick // 8] * 8 + pick % 8
                        cute.copy(write_atom, mapped, logical_dst)
                    else:
                        for j in cutlass.range_constexpr(4):
                            if col + j < required:
                                pick = staged[col + j]
                                if cutlass.const_expr(not self.logical_only):
                                    out[row, col + j] = pick
                                logical[row, col + j] = (
                                    blocks[row, pick // 8] * 8 + pick % 8
                                )

            else:
                select_stream(
                    row,
                    tid,
                    candidates,
                    n,
                    required,
                    hist,
                    out,
                    512,
                    cutlass.BFloat16,
                    self.threads,
                    0,
                    banks=self.banks,
                    header_slots=32,
                    storage_banks=32,
                    warp_uniform_counts=True,
                )
                cute.arch.barrier(barrier_id=0, number_of_threads=self.threads)
                for col in cutlass.range(tid, 512, self.threads):
                    pick = out[row, col]
                    mapped = cutlass.Int32(-1)
                    if pick >= 0:
                        mapped = blocks[row, pick // 8] * 8 + pick % 8
                    logical[row, col] = mapped


@functools.cache
def compile_selector(
    signature, logical_only=False, pdl=False, cache_entries=128, banks=16
):
    tensors = []
    for i, (dtype, ndim) in enumerate(signature):
        dims = tuple(cute.sym_int() for _ in range(ndim))
        if i == 0:
            tensor = cute.runtime.make_fake_tensor(
                DTYPES[dtype], (dims[0], 8224), stride=(8224, 1), assumed_align=16
            )
        else:
            tensor = cute.runtime.make_fake_compact_tensor(
                DTYPES[dtype],
                dims,
                stride_order=tuple(reversed(range(ndim))),
                assumed_align=16,
            )
        tensors.append(tensor)
    return cute.compile(
        SparseSelector(logical_only, pdl, cache_entries, banks),
        *tensors,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
