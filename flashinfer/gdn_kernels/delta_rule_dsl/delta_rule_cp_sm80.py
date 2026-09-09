"""Chunk-parallel GDN prefill for sm_80: stage 1, the T precompute.

The chunk-parallel backend is four stages -- precompute T, precompute each
chunk's local transfer and state, fix those into chunk-boundary states, and run
the prefill recurrence inside each chunk. Only the first three are ported here.
The fourth computes a prefill over a token range from a given state, which
`_FullyFusedDeltaRuleSm80` already does correctly, so chunks are handed to it as
independent sequences; the composition is proven bit-exact in
`logs/cp_compose/RESULT.md` in the measurement harness, and its contract is
written down in
`CP_LAYOUTS.md`.

Ported from `delta_rule_cp_sm120.py`, which is the closer base: it uses the same
`warp.MmaF16BF16Op((16, 8, 16))` this target has, where the sm_90 file uses
`warpgroup.MmaF16BF16Op` and twenty-one wgmma fence/commit/wait calls that have
no equivalent before sm_90.

What had to be replaced, and it is the same list as the non-CP port:

    TMA loads                 -> cp.async through `_load_tile`
    PipelineTmaAsync          -> PipelineCpAsyncSm80, which counts async groups
    PipelineAsync (the beta   -> a block barrier; with one warp group there is
    vector)                      no other group to hand the tile to
    mbarrier_init_fence       -> nothing to fence
    prefetch_descriptor       -> no descriptor to warm
    stmatrix                  -> an ordinary register-to-shared copy

`wgmma`, clusters, distributed shared memory and `elect_one` do not appear in
the sm120 file at all, so nothing here budgets for them.

The two stages do not share a block width, and that is forced rather than
chosen. Stage 1's MMA has BLK = 64 as its M mode, which four warps span exactly.
Stage 2's have D = 128, and `make_acc_into_op` -- which hands an accumulator on
as the next GEMM's A operand, four times in its inner loop -- is only valid at
one M repeat per warp. Eight warps, so 256 threads. The same constraint that
pinned `D_v` to `NUM_MMA_WARPS * 16` in the fused kernel.
"""

from __future__ import annotations

import functools
import math
from enum import IntEnum

import torch
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import warp
from cutlass.cute.nvgpu import warpgroup

from ...utils import (
    _get_cache_buf,
    get_compute_capability,
    get_device_name,
    get_device_sm_count,
)
from .alpha import AlphaProcessor
from .collective_inverse_hmma import CollectiveInverse
from .custom_compile_cache import (
    KeyedCompileMixin,
    cached_compile,
    get_cached_compile,
    sm8x_compile_options,
)
from .delta_rule_sm80 import NamedBarrier as FusedNamedBarrier
from .delta_rule_sm80 import NUM_MMA_WARPS as _FUSED_MMA_WARPS
from .delta_rule_sm80 import LOAD_THREADS as _FUSED_LOAD_THREADS
from .delta_rule_sm80 import THREADS_PER_WARP_GROUP as _FUSED_WG_THREADS
from .delta_rule_sm80 import _BLK_V as _FUSED_BLK_V
from .delta_rule_sm80 import _FullyFusedDeltaRuleSm80
from .delta_rule_sm80 import _check_load_alignment
from .helpers import (
    SM80,
    load_tensor_as_a,
    load_tensor_as_b,
    select_tensor_10,
)
from .pipeline_sm80 import PipelineCpAsyncSm80
from .schedule import WorkDesc
from .varlen_helper import (
    CP_CHUNK_LEN_GRANULARITY,
    choose_cp_chunk_len_host,
    chunks_for_len,
    integer_dtype_to_cutlass,
    is_integer_dtype,
    max_num_chunks_host,
    varlen_chunk_idx,
    varlen_chunk_valid_len,
    workspace_num_chunks_host,
)


# One warp group. The sm120 kernel gives the K tile and the beta vector to warps
# 1 and 2 and lets the rest wait on an mbarrier; that hand-off needs a pipeline
# this DSL will not emit for sm_80 -- see `pipeline_sm80` for which mbarrier
# operations it refuses -- so every thread issues its own share of the loads and
# a block barrier publishes them.


class NamedBarrier(IntEnum):
    """Barrier ids for the CP stages.

    Three, not two. The ordered pair between the math warp groups and the
    block-wide rendezvous inside a block's math must not share an id: reusing
    MATH_WG0 for both hung the kernel with the GPU at 100% and one thread
    sleeping, which is what a barrier half the block is waiting on looks like.
    """

    MATH_WG0 = 4
    MATH_WG1 = 5
    MATH_SYNC = 6
    # Stage 1 borrows the fused kernel's, since it runs the same inverse.
    KK_SYNC = FusedNamedBarrier.KK_SYNC


THREADS = 128
# Stage 2 needs eight warps: its MMAs carry D = 128 as the M mode and
# `make_acc_into_op` is only valid at one M repeat per warp.
THREADS_MN = 256
_LOAD_ALIGN_BYTES = 16


@functools.cache
def _sm80_cp_compile_options(device):
    """The compile options every CP stage shares: the device's SM8x arch."""
    return sm8x_compile_options(device)


class _CPInvocation:
    """What the four entries of one composition call can share.

    Built only by `cp_delta_rule_dsl_sm80`, lives for one call, and is passed
    down as a private argument. Nothing here is a cache across calls, a global,
    or a patched descriptor: every field is derived from this call's own
    arguments and dropped when it returns.

    The reason it exists is measured. On `1x8192 h2` the pointer path spends
    366 us idle between its four kernels -- 185 us after T and 171 us after
    the fixup, both short kernels the GPU drains before the host has issued the
    next entry -- against 391 us of GPU work
    (`logs/cp_timeline_ptr/RESULT.md` in the measurement harness). Argument
    generation is only 99 us of that, so what is left is each entry
    re-deriving what the entry before it already had.

    What it does *not* do is elide any of the four workspace acquisitions.
    Those are four different buffers with four different shapes; sharing the
    lookup would be sharing nothing.
    """

    __slots__ = (
        "device",
        "stream",
        "compile_options",
        "total_t_blocks",
        "total_cp_chunks",
        "max_t_blocks_per_seq",
        "max_cp_chunks_per_seq",
        "cp_chunk_len",
        "_ptrs",
    )

    def __init__(
        self,
        *,
        device,
        stream,
        compile_options,
        total_t_blocks,
        total_cp_chunks,
        max_t_blocks_per_seq,
        max_cp_chunks_per_seq,
        cp_chunk_len,
    ):
        self.device = device
        # The composition took this from `torch.cuda.current_stream`, so the
        # four entries below bind to one stream without asking again.
        self.stream = stream
        self.compile_options = compile_options
        self.total_t_blocks = total_t_blocks
        self.total_cp_chunks = total_cp_chunks
        self.max_t_blocks_per_seq = max_t_blocks_per_seq
        self.max_cp_chunks_per_seq = max_cp_chunks_per_seq
        self.cp_chunk_len = cp_chunk_len
        # `(address, dtype, alignment) -> pointer`. q, k, v and cu_seqlens are
        # each handed to more than one entry; this builds one wrapper for each
        # and checks its alignment once. Emptied with the object.
        self._ptrs: dict = {}

    def ptr(self, dtype, tensor, align, name):
        """A pointer for `tensor`, alignment checked once per distinct buffer.

        `assumed_align` is a promise the compiler cannot verify, so it is
        checked here -- and checked at the point the pointer is made, which is
        the only place that knows which promise is being given.
        """
        addr = tensor.data_ptr()
        if addr % align:
            raise RuntimeError(
                f"the pointer entry promises {align}-byte alignment for "
                f"{name}, and its data_ptr() is {addr % align} bytes past it; "
                f"pass an aligned buffer or use the tensor entry, which lets "
                f"the DSL derive the alignment"
            )
        key = (addr, dtype, align)
        got = self._ptrs.get(key)
        if got is None:
            got = cute.runtime.make_ptr(
                dtype, addr, cute.AddressSpace.gmem, assumed_align=align
            )
            self._ptrs[key] = got
        return got


def _ptr_factory(_ctx, device, _stream, promises):
    """How the pointer entries make their pointers, with or without a context.

    Standalone: `_check_ptr_abi` up front, exactly as before -- the alignment
    promises and the current-stream rule, neither behind `_skip_check`.

    Inside a composition: the context checks each promise where the pointer is
    made, and reuses one wrapper per distinct `(address, dtype, alignment)`.
    q, k, v and `cu_seqlens` reach more than one entry, so the check and the
    wrapper happen once for each buffer rather than once per entry. The
    composition validated the stream when it built the context.
    """
    if _ctx is not None:
        return _ctx.ptr
    _check_ptr_abi(device, _stream, *promises)

    def mk(dtype, tensor, align, name):
        """Make one gmem pointer for a tensor, at the alignment the layout claims.

        Closed over the context so repeated entries share one checked pointer
        per (address, dtype, alignment) instead of rebuilding it per stage."""
        return cute.runtime.make_ptr(
            dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
        )

    return mk


def _check_ptr_abi(device, stream, *promises):
    """Preconditions the pointer entries need, checked unconditionally.

    Deliberately not behind `_skip_check`. That flag turns off shape and dtype
    validation, which the composition already did once for the whole call --
    but these are not shape checks. `assumed_align` is a promise to the
    compiler that the caller cannot see, and a raw pointer carries no owner, so
    a caller passing `_skip_check=True` would be turning off exactly the two
    things the pointer path adds. The composition does pass it, per stage.

    `promises` are `(name, tensor, bytes)`. A contiguous tensor is not an
    aligned one: an offset slice of a larger buffer is contiguous at any
    address.
    """
    for name, tensor, want in promises:
        if tensor is None:
            continue
        addr = tensor.data_ptr()
        if addr % want:
            raise RuntimeError(
                f"the pointer entry promises {want}-byte alignment for {name}, "
                f"and its data_ptr() is {addr % want} bytes past it; pass an "
                f"aligned buffer or use the tensor entry, which lets the DSL "
                f"derive the alignment"
            )
    if stream is None:
        return
    current = torch.cuda.current_stream(device).cuda_stream
    given = getattr(stream, "value", stream)
    if int(given) != int(current):
        # The pointers are raw addresses with no reference to the tensors that
        # own them, and nothing here keeps those alive past the launch. On the
        # current stream the caller's own references cover it; on another
        # stream the caller would have to guarantee that too, and no test
        # covers it. The composition passes the current stream's handle, so
        # equality is the check rather than refusing a handle outright.
        raise RuntimeError(
            "the pointer entry runs on the current stream only: it passes raw "
            f"addresses and keeps nothing alive, and the handle given "
            f"({int(given):#x}) is not the current stream's ({int(current):#x})"
        )


def _cp_workspace(name, shape, dtype, device):
    """A cached device buffer of this shape, viewed as the requested dtype."""
    nbytes = math.prod(shape) * dtype.itemsize
    return _get_cache_buf(name, nbytes, device)[:nbytes].view(dtype).view(shape)


class CPDeltaRuleTPrecomputeSm80(KeyedCompileMixin):
    """T = -beta * (I - tril(diag(beta) K K^T, -1))^-1, per 64-token block.

    One block of the grid per (sequence, head, 64-token chunk). Nothing here is
    sequential across chunks -- that is stage 3 -- so the grid is as wide as the
    token count divided by 64, which is the point of the whole exercise.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        cu_seqlens_dtype: type[cutlass.Numeric] = cutlass.Int64,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.inverse_dtype = cutlass.Float16
        self.BLK = 64
        self.D = 128
        # bf16 or fp16 elements in one 16 B cp.async access.
        self.elems_per_lane = 128 // self.dtype.width
        self.manual_cache_key(
            "dtype", "acc_dtype", "cu_seqlens_dtype", "inverse_dtype", "BLK", "D"
        )

    # ─── loads ────────────────────────────────────────────────────────────────

    @cute.jit
    def _load_k_tile(
        self,
        gK_full: cute.Tensor,
        sK_DS: cute.Tensor,
        tok_offset,
        head_idx: cutlass.Int32,
        valid_len: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """One (D, BLK) tile of K, zero-filling the rows past the chunk's end.

        TMA clamped at the tensor bound; here the bound is a predicate, and a
        row the chunk does not own has to be *zeroed* rather than skipped --
        K K^T reads the whole tile and the masking below assumes the padding
        contributes nothing.
        """
        mK = cute.domain_offset(
            (cutlass.Int32(0), tok_offset), gK_full[None, None, head_idx]
        )
        gK = cute.zipped_divide(mK, (self.D, self.BLK))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]

        lanes_along_d = cutlass.const_expr(self.D // self.elems_per_lane)
        tv_layout = cute.make_layout(
            (lanes_along_d, THREADS // lanes_along_d), stride=(1, lanes_along_d)
        )
        val_layout = cute.make_layout((self.elems_per_lane, 1))
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=128
        )
        tiled = cute.make_tiled_copy_tv(atom, tv_layout, val_layout)
        thr = tiled.get_slice(tid)
        tSrc = thr.partition_S(gK)
        tDst = thr.partition_D(sK_DS)

        rows_at_a_time = cutlass.Int32(THREADS) // cutlass.Int32(lanes_along_d)
        lane_row = tid // cutlass.Int32(lanes_along_d)
        for rep in cutlass.range_constexpr(cute.size(tSrc, mode=[2])):
            row = cutlass.Int32(rep) * rows_at_a_time + lane_row
            if row < valid_len:
                source = tSrc[None, 0, rep]
                # The strides are runtime values, so the layout algebra falls
                # back to element alignment and the 128-bit atom is rejected for
                # wanting more than it is told the pointer holds. Rounding up is
                # identity on a pointer already there; `_check_load_alignment`
                # rejects inputs that break the premise before any of this runs.
                cute.copy(
                    atom,
                    cute.make_tensor(
                        source.iterator.align(_LOAD_ALIGN_BYTES), source.layout
                    ),
                    tDst[None, 0, rep],
                )
            else:
                tDst[None, 0, rep].fill(self.dtype(0.0))

    @cute.jit
    def _load_beta(
        self,
        g_beta: cute.Tensor,
        sBeta: cute.Tensor,
        tok_offset,
        head_idx: cutlass.Int32,
        valid_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """The chunk's beta, one lane a row, zero past the end.

        The sm120 kernel gives this to a warp of its own behind a
        `PipelineAsync`; with a single warp group there is nobody to hand it to,
        so the whole block writes its own share and the barrier in the caller
        publishes it.
        """
        for i in cutlass.range_constexpr(self.BLK // THREADS + 1):
            row = tid + i * THREADS
            if row < cutlass.Int32(self.BLK):
                beta = cutlass.Float32(0.0)
                if row < valid_len:
                    beta = cutlass.Float32(
                        g_beta[(tok_offset + row) * num_heads + head_idx]
                    )
                sBeta[row] = beta

    # ─── epilogues ────────────────────────────────────────────────────────────

    @cute.jit
    def _kk_epi(self, tKKrKK: cute.Tensor, tKKcMkk: cute.Tensor, sBeta: cute.Tensor):
        """Strictly-lower triangle, scaled by the row's beta; zero elsewhere."""
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            row, col = tKKcMkk[i]
            value = cutlass.Float32(0.0)
            if row > col:
                value = cutlass.Float32(tKKrKK[i]) * cutlass.Float32(sBeta[row])
            tKKrKK[i] = value

    @cute.jit
    def _store_ikk_to_smem(
        self,
        tKKrKK: cute.Tensor,
        kk_tiled_mma,
        thread_idx: cutlass.Int32,
        sKK_inv: cute.Tensor,
        tKKcMkk: cute.Tensor,
    ):
        """The matrix the inverse runs on, into shared.

        sm120 writes it with `stmatrix`, which starts at sm_90. An ordinary
        register-to-shared copy through the same tiled layout puts the same
        values in the same places; `CollectiveInverse(has_stmatrix=False)` is
        the matching read.
        """
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.inverse_dtype)
        tiled_store = cute.make_tiled_copy_C(r2s_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(thread_idx)
        tKKsKK = thr_store.partition_D(sKK_inv)
        tKKrKK_cv = thr_store.retile(tKKrKK)
        tKKcMkk_cv = thr_store.retile(tKKcMkk)
        tKKrIKK = cute.make_fragment_like(tKKrKK_cv, self.inverse_dtype)
        for i in cutlass.range_constexpr(cute.size(tKKrKK_cv)):
            row, col = tKKcMkk_cv[i]
            value = cutlass.Float32(0.0)
            if row > col:
                value = cutlass.Float32(tKKrKK_cv[i])
            tKKrIKK[i] = self.inverse_dtype(value)
        cute.copy(tiled_store, tKKrIKK, tKKsKK)

    @cute.jit
    def _store_t(
        self,
        sKK_inv: cute.Tensor,
        sBeta: cute.Tensor,
        gT: cute.Tensor,
        valid_len: cutlass.Int32,
        kk_tiled_mma,
        tid: cutlass.Int32,
    ):
        """T = -beta * inverse, transposed on the way out.

        The coordinate is read as (col, row), which is where the transpose
        lives -- `CP_LAYOUTS.md` for why the orientation is worth naming.
        """
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.inverse_dtype
        )
        tiled_load = cute.make_tiled_copy_C(ldsm_atom, kk_tiled_mma)
        thr_load = tiled_load.get_slice(tid)
        tTsT = thr_load.partition_S(sKK_inv)
        tTrInv = cute.make_rmem_tensor(
            kk_tiled_mma.partition_shape_C((self.BLK, self.BLK)), self.inverse_dtype
        )
        cute.copy(tiled_load, tTsT, thr_load.retile(tTrInv))

        store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        tiled_store = cute.make_tiled_copy_C(store_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(tid)
        tTgT = thr_store.partition_D(gT)
        tTcT = thr_store.partition_D(cute.make_identity_tensor((self.BLK, self.BLK)))
        tTrInv_store = thr_store.retile(tTrInv)
        tTrT = cute.make_rmem_tensor(
            kk_tiled_mma.partition_shape_C((self.BLK, self.BLK)), self.dtype
        )
        tTrT_store = thr_store.retile(tTrT)
        for i in cutlass.range_constexpr(cute.size(tTrT_store)):
            col, row = tTcT[i]
            value = cutlass.Float32(0.0)
            if row < valid_len and col < valid_len:
                value = -cutlass.Float32(sBeta[row]) * cutlass.Float32(tTrInv_store[i])
            tTrT_store[i] = self.dtype(value)
        cute.autovec_copy(tTrT_store, tTgT)

    # ─── entry ────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        g_k: cute.Tensor,
        g_beta: cute.Tensor,
        g_t: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_k_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        max_t_blocks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        k_smem_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, self.dtype
        )
        k_layout_sd = cute.tile_to_shape(k_smem_atom, (self.BLK, self.D), order=(0, 1))
        kk_layout = cute.tile_to_shape(
            cute.make_layout((8, 8), stride=(8, 1)), (self.BLK, self.BLK), order=(0, 1)
        )
        beta_layout = cute.make_layout(self.BLK)

        @cute.struct
        class SharedStorage:
            # No mbarrier storage: the sm_80 pipeline counts cp.async groups,
            # so the two the sm120 kernel reserves here are gone.
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_layout_sd)], 128
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[self.inverse_dtype, cute.cosize(kk_layout)], 16
            ]
            smem_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_layout)], 16
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            g_k,
            g_beta,
            g_t,
            cu_seqlens,
            num_k_heads,
            num_sab_heads,
            total_t_blocks,
            num_seqs,
        ).launch(
            grid=(num_sab_heads * max_t_blocks_per_seq, num_seqs, 1),
            block=(THREADS, 1, 1),
            max_number_threads=(THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gK_full: cute.Tensor,
        g_beta: cute.Tensor,
        g_t: cute.Tensor,
        cu_seqlens: cute.Tensor,
        num_k_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, seq_idx, _ = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        block_idx_in_seq = bx // num_sab_heads
        seq_start = cu_seqlens[seq_idx]
        seq_len = cutlass.Int32(cu_seqlens[seq_idx + 1] - seq_start)
        num_blocks = chunks_for_len(seq_len, self.BLK)

        if block_idx_in_seq < num_blocks:
            tok_offset = seq_start + block_idx_in_seq * self.BLK
            t_block_idx = varlen_chunk_idx(
                seq_idx, seq_start, block_idx_in_seq, self.BLK
            )
            valid_len = varlen_chunk_valid_len(seq_len, block_idx_in_seq, self.BLK)
            mT = cute.make_tensor(
                g_t.iterator,
                cute.make_ordered_layout(
                    (self.BLK, self.BLK, num_sab_heads, total_t_blocks),
                    order=(0, 1, 2, 3),
                ),
            )
            gT = mT[None, None, sab_head_idx, t_block_idx]

            k_smem_atom = warpgroup.make_smem_layout_atom(
                warpgroup.SmemLayoutAtomKind.K_SW128, self.dtype
            )
            k_layout_sd = cute.tile_to_shape(
                k_smem_atom, (self.BLK, self.D), order=(0, 1)
            )
            k_layout_ds = cute.select(k_layout_sd, [1, 0])
            kk_layout = cute.tile_to_shape(
                cute.make_layout((8, 8), stride=(8, 1)),
                (self.BLK, self.BLK),
                order=(0, 1),
            )
            beta_layout = cute.make_layout(self.BLK)

            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate(self.shared_storage)
            sK_DS = storage.smem_k.get_tensor(
                k_layout_ds.outer, swizzle=k_layout_ds.inner
            )
            sK_SD = select_tensor_10(sK_DS)
            sKK_inv = storage.smem_kk.get_tensor(kk_layout)
            sBeta = storage.smem_beta.get_tensor(beta_layout)

            k_pipeline = PipelineCpAsyncSm80(1)
            k_producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            k_consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )

            # Every thread loads its share of both, so nothing is handed between
            # warps and one barrier publishes the pair.
            k_pipeline.producer_acquire(k_producer)
            self._load_k_tile(gK_full, sK_DS, tok_offset, k_head_idx, valid_len, tidx)
            cute.arch.cp_async_commit_group()
            k_pipeline.producer_commit(k_producer)
            self._load_beta(
                g_beta, sBeta, tok_offset, sab_head_idx, valid_len, num_sab_heads, tidx
            )
            k_pipeline.consumer_wait(k_consumer)

            kk_tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
                cute.make_layout((THREADS // 32, 1, 1)),
                permutation_mnk=(self.BLK, self.BLK, self.D),
            )
            kk_thr_mma = kk_tiled_mma.get_slice(tidx)
            ldsm_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
            )
            kk_tiled_copy_A = cute.make_tiled_copy_A(ldsm_atom, kk_tiled_mma)
            kk_tiled_copy_B = cute.make_tiled_copy_B(ldsm_atom, kk_tiled_mma)
            kk_thr_copy_A = kk_tiled_copy_A.get_slice(tidx)
            kk_thr_copy_B = kk_tiled_copy_B.get_slice(tidx)
            tKKrA = kk_thr_mma.make_fragment_A(kk_thr_mma.partition_A(sK_SD))
            tKKrB = kk_thr_mma.make_fragment_B(kk_thr_mma.partition_B(sK_SD))
            tKKrKK = kk_thr_mma.make_fragment_C(
                kk_thr_mma.partition_shape_C((self.BLK, self.BLK))
            )
            tKKcMkk = kk_thr_mma.partition_C(
                cute.make_identity_tensor((self.BLK, self.BLK))
            )

            cute.copy(
                kk_tiled_copy_A,
                kk_thr_copy_A.partition_S(sK_SD),
                kk_thr_copy_A.retile(tKKrA),
            )
            cute.copy(
                kk_tiled_copy_B,
                kk_thr_copy_B.partition_S(sK_SD),
                kk_thr_copy_B.retile(tKKrB),
            )
            tKKrKK.fill(self.acc_dtype(0.0))
            cute.gemm(kk_tiled_mma, tKKrKK, tKKrA, tKKrB, tKKrKK)

            self._kk_epi(tKKrKK, tKKcMkk, sBeta)
            self._store_ikk_to_smem(tKKrKK, kk_tiled_mma, tidx, sKK_inv, tKKcMkk)
            cute.arch.barrier(
                barrier_id=NamedBarrier.KK_SYNC, number_of_threads=THREADS
            )
            CollectiveInverse(has_stmatrix=False).run(sKK_inv, NamedBarrier.KK_SYNC)
            cute.arch.barrier(
                barrier_id=NamedBarrier.KK_SYNC, number_of_threads=THREADS
            )
            self._store_t(sKK_inv, sBeta, gT, valid_len, kk_tiled_mma, tidx)


class CPDeltaRuleMNPrecomputeSm80(KeyedCompileMixin):
    """One chunk's local affine map: state <- transfer @ state + local.

    Each block owns one (sequence, head, chunk) and walks that chunk's 64-token
    blocks, which is where the parallelism is -- nothing here is sequential
    across chunks. Stage 3 composes the results.

    256 threads, and not by choice. The four GEMMs in the inner loop hand
    accumulators on as the next one's A operand through `make_acc_into_op`, and
    that reinterpretation is only valid at one M repeat per warp. M is D = 128
    here, so eight warps. Stage 1's M is BLK = 64 and runs on four; the fused
    kernel pins `D_v` to `NUM_MMA_WARPS * 16` for exactly the same reason.

    The sm120 kernel spends a third warp group on TMA loads and lets the math
    groups wait on an mbarrier. That hand-off is what this target cannot build,
    so the loads are cp.async issued by every thread and the pipelines count
    async groups. The two math warp groups and their ordered barriers stay:
    removing that ordering from the fused kernel cost 2.3% and 22% on two cells.

    Both outputs are (V, K) -- `CP_LAYOUTS.md` -- and every name here says so.
    """

    def __init__(
        self,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        cu_seqlens_dtype: type[cutlass.Numeric] = cutlass.Int64,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.BLK = 64
        self.D = 128
        # Two stages each, and they advance -- but nothing overlaps yet. The
        # consumer waits on `cp_async_wait_group(0)` right after the loads are
        # issued, which drains every group before the MMAs start, so the second
        # buffer costs shared memory and buys nothing.
        #
        # One stage would be correct as this stands. Two is what a moved wait
        # would need, and moving it is the point of keeping them; collapsing to
        # one and moving the wait are the two halves of the same A/B, to run
        # once correctness is frozen.
        self.k_stage = 2
        self.v_stage = 2
        self.t_stage = 2
        self.alpha_stage = 2
        self.elems_per_lane = 128 // self.dtype.width
        self.manual_cache_key(
            "dtype",
            "acc_dtype",
            "cu_seqlens_dtype",
            "BLK",
            "D",
            "k_stage",
            "v_stage",
            "t_stage",
            "alpha_stage",
        )

    # ─── layouts, computed once ───────────────────────────────────────────────

    def _layouts(self):
        """Every shared layout, from one place.

        `__call__` sizes `SharedStorage` from these and `kernel` builds the
        tensors from them. Computing them twice means one edit can leave the
        allocation and the view disagreeing, which is a silent overrun rather
        than an error.
        """
        atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_INTER, self.dtype
        )
        k_sd = cute.tile_to_shape(
            atom, (self.BLK, self.D, self.k_stage), order=(0, 1, 2)
        )
        v_sd = cute.tile_to_shape(
            atom, (self.BLK, self.D, self.v_stage), order=(0, 1, 2)
        )
        t = cute.tile_to_shape(
            atom, (self.BLK, self.BLK, self.t_stage), order=(0, 1, 2)
        )
        x_ds = cute.tile_to_shape(
            cute.make_layout((8, 8), stride=(8, 1)), (self.D, self.BLK), order=(0, 1)
        )
        alpha = cute.make_layout(
            (self.BLK, AlphaProcessor.NUM_CHANNELS, self.alpha_stage)
        )
        return k_sd, v_sd, t, x_ds, alpha

    # ─── loads ────────────────────────────────────────────────────────────────

    @cute.jit
    def _load_dn_tile(
        self,
        gFull: cute.Tensor,
        sDest: cute.Tensor,
        stage: cutlass.Int32,
        row_offset,
        col_offset,
        head_idx: cutlass.Int32,
        rows: cutlass.Constexpr,
        valid_len: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """A (rows, BLK) tile with `rows` contiguous, into one stage.

        The whole block issues it. sm120 gives each of K, V and T a warp of its
        own and hands the tile over on an mbarrier; that hand-off is the thing
        this target cannot build.
        """
        mTensor = cute.domain_offset(
            (row_offset, col_offset), gFull[None, None, head_idx]
        )
        gTile = cute.zipped_divide(mTensor, (rows, self.BLK))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        lanes = cutlass.const_expr(rows // self.elems_per_lane)
        tiled = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=128),
            cute.make_layout((lanes, THREADS_MN // lanes), stride=(1, lanes)),
            cute.make_layout((self.elems_per_lane, 1)),
        )
        thr = tiled.get_slice(tid)
        tSrc = thr.partition_S(gTile)
        tDst = thr.partition_D(sDest[None, None, stage])
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=128
        )
        rows_at_a_time = cutlass.Int32(THREADS_MN) // cutlass.Int32(lanes)
        lane_row = tid // cutlass.Int32(lanes)
        for rep in cutlass.range_constexpr(cute.size(tSrc, mode=[2])):
            source = tSrc[None, 0, rep]
            dest = tDst[None, 0, rep]
            # A column past the chunk's last token is zeroed rather than
            # skipped. TMA clamped at the tensor bound for free and cp.async
            # does not, so without this the tail block reads whatever follows
            # the chunk in global memory -- and for V nothing downstream kills
            # it, since only the alpha channel masks that tail and it masks the
            # scale, not the value.
            #
            # Both ends are told their alignment: the source's strides are
            # runtime values so the algebra falls back to element alignment, and
            # the destination's swizzle hides its own.
            if cutlass.Int32(rep) * rows_at_a_time + lane_row < valid_len:
                cute.copy(
                    atom,
                    cute.make_tensor(
                        source.iterator.align(_LOAD_ALIGN_BYTES), source.layout
                    ),
                    cute.make_tensor(
                        dest.iterator.align(_LOAD_ALIGN_BYTES), dest.layout
                    ),
                )
            else:
                dest.fill(self.dtype(0.0))

    @cute.jit
    def _load_alpha_block(
        self,
        g_alpha: cute.Tensor,
        sAlpha: cute.Tensor,
        stage: cutlass.Int32,
        tok_offset,
        head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        chunk_len: cutlass.Int32,
        num_heads: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """The block's alpha, one lane a row, 1.0 past the chunk's end.

        One is the identity for the cumulative product the processor builds, so
        a token the chunk does not own contributes nothing to the decay.
        """
        lane = tid % 32
        if tid < cutlass.Int32(32):
            sAlpha_stage = sAlpha[None, None, stage]
            for i in cutlass.range_constexpr(self.BLK // 32):
                row = lane + i * 32
                tok = blk * self.BLK + row
                alpha = cutlass.Float32(1.0)
                if tok < chunk_len:
                    alpha = cutlass.Float32(
                        g_alpha[(tok_offset + tok) * num_heads + head_idx]
                    )
                sAlpha_stage[row, AlphaProcessor.CUMSUM_LOG] = alpha

    @cute.jit
    def _mask_alpha_oob(
        self,
        sAlpha: cute.Tensor,
        stage: cutlass.Int32,
        blk: cutlass.Int32,
        chunk_len: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """Zero the alpha channels a block does not own, so the tile reads whole."""
        lane = tid % 32
        if tid < cutlass.Int32(32):
            sAlpha_stage = sAlpha[None, None, stage]
            for i in cutlass.range_constexpr(self.BLK // 32):
                row = lane + i * 32
                if blk * self.BLK + row >= chunk_len:
                    sAlpha_stage[row, AlphaProcessor.CUMPROD_NEG_END_RCP] = (
                        cutlass.Float32(0.0)
                    )

    # ─── epilogues ────────────────────────────────────────────────────────────

    @cute.jit
    def _store_acc_to_smem(
        self,
        tCrC: cute.Tensor,
        sC: cute.Tensor,
        tiled_mma,
        thread_idx: cutlass.Int32,
    ):
        """sm120 uses `stmatrix`, which starts at sm_90.

        An ordinary register-to-shared copy through the same tiled C layout puts
        the same values in the same places. The transpose the sm_90 atom could
        do in the instruction is not needed here: this caller asks for
        `transpose=False`.
        """
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        tiled_copy = cute.make_tiled_copy_C(atom, tiled_mma)
        thr_copy = tiled_copy.get_slice(thread_idx)
        tCsC = thr_copy.partition_D(sC)
        converted = cute.make_fragment_like(tCrC, self.dtype)
        for i in cutlass.range_constexpr(cute.size(tCrC)):
            converted[i] = self.dtype(tCrC[i])
        cute.copy(tiled_copy, thr_copy.retile(converted), tCsC)

    @cute.jit
    def _store_acc_VK(
        self,
        tCrC_VK: cute.Tensor,
        gC_VK: cute.Tensor,
        tiled_mma,
        thread_idx: cutlass.Int32,
    ):
        """The accumulator out to global, V down the rows and K across.

        Named because nothing checks it: D is 128 on both axes, so a transposed
        write is the same shape and the same byte count, and only the numbers
        come out wrong.
        """
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(), self.acc_dtype, num_bits_per_copy=64
        )
        tiled_copy = cute.make_tiled_copy_C(atom, tiled_mma)
        thr_copy = tiled_copy.get_slice(thread_idx)
        tCgC = thr_copy.partition_D(gC_VK)
        tCgC = cute.make_tensor(tCgC.iterator.align(8), tCgC.layout)
        cute.copy(tiled_copy, thr_copy.retile(tCrC_VK), tCgC)

    @cute.jit
    def _load_v_as_fragment_C(self, sV, tiled_mma, thr_mma, thread_idx, c_shape):
        """V into the accumulator layout the next GEMM will reinterpret.

        `load_tensor_as_c` builds its destination with `make_rmem_tensor(shape)`,
        which is a plain compact layout. `make_acc_into_op` reads a
        `make_fragment_C` layout. The two have the same size -- 32 slots here --
        so `basic_copy` between them passes its static check and walks the two
        in different orders. That pins the state's rank at 8, one MMA atom's N,
        however many blocks run.

        So the destination is the fragment from the start, with an fp16 staging
        buffer of the same layout for the ldmatrix to land in.
        """
        tAcc = thr_mma.make_fragment_C(thr_mma.partition_shape_C(c_shape))
        tStage = cute.make_fragment_like(tAcc, self.dtype)
        ldsm_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        tiled_copy = cute.make_tiled_copy_C(ldsm_atom, tiled_mma)
        thr_copy = tiled_copy.get_slice(thread_idx)
        cute.copy(tiled_copy, thr_copy.partition_S(sV), thr_copy.retile(tStage))
        for i in cutlass.range_constexpr(cute.size(tAcc)):
            tAcc[i] = self.acc_dtype(tStage[i])
        return tAcc

    @cute.jit
    def _scale_acc(self, tCrC: cute.Tensor, coeff: cutlass.Float32):
        """Scale an accumulator in place by one runtime coefficient."""
        for i in cutlass.range(cute.size(tCrC), unroll_full=True):
            tCrC[i] = cutlass.Float32(tCrC[i]) * coeff

    # ─── the inner loop ───────────────────────────────────────────────────────

    @cute.jit
    def _compute_block(
        self,
        sV_DS: cute.Tensor,
        sK_DS: cute.Tensor,
        sT: cute.Tensor,
        sX_DS: cute.Tensor,
        sAlpha: cute.Tensor,
        stage: cutlass.Int32,
        tTransfer_VK: cute.Tensor,
        tState_VK: cute.Tensor,
        valid_len: cutlass.Int32,
        thread_idx: cutlass.Int32,
        wg_idx: cutlass.Int32,
        is_first_block: cutlass.Constexpr,
    ):
        """One 64-token block's contribution to the chunk's affine map.

        Four GEMMs, and each hands its accumulator on as the next one's A
        operand, which is why this needs eight warps -- see the class docstring.
        """
        x_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((THREADS_MN // 32, 1, 1)),
            permutation_mnk=(self.D, self.BLK, self.BLK),
        )
        z_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((THREADS_MN // 32, 1, 1)),
            permutation_mnk=(self.D, self.BLK, self.D),
        )
        y_mma = z_mma
        trans_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((THREADS_MN // 32, 1, 1)),
            permutation_mnk=(self.D, self.D, self.BLK),
        )
        state_mma = trans_mma

        x_thr = x_mma.get_slice(thread_idx)
        z_thr = z_mma.get_slice(thread_idx)
        y_thr = z_thr

        sK = sK_DS[None, None, stage]
        sV = sV_DS[None, None, stage]
        sT_blk = sT[None, None, stage]
        sAlpha_blk = sAlpha[None, None, stage]
        sK_SD = select_tensor_10(sK)
        cK = cute.make_identity_tensor((self.D, self.BLK))
        block_coeff = cutlass.Float32(sAlpha_blk[valid_len - 1, AlphaProcessor.CUMPROD])

        # X = T_fused K, kept transposed for both the state and the transfer.
        tXrK = load_tensor_as_a(
            sK, x_mma, thread_idx, (self.D, self.BLK), self.dtype, False
        )
        tXrT = load_tensor_as_b(
            sT_blk, x_mma, thread_idx, (self.BLK, self.BLK), self.dtype, True
        )
        tXrX = x_thr.make_fragment_C(x_thr.partition_shape_C((self.D, self.BLK)))
        tXrX.fill(self.acc_dtype(0.0))
        cute.gemm(x_mma, tXrX, tXrK, tXrT, tXrX)
        self._store_acc_to_smem(tXrX, sX_DS, x_mma, thread_idx)
        cute.arch.barrier(
            barrier_id=NamedBarrier.MATH_SYNC, number_of_threads=THREADS_MN
        )

        # Transfer: Z = M K^T, then M <- gamma_end (M + Z X).
        ldsm_n4 = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        z_copy_B = cute.make_tiled_copy_B(ldsm_n4, z_mma)
        z_thr_copy_B = z_copy_B.get_slice(thread_idx)
        tZrK = cute.make_rmem_tensor(
            z_thr.partition_shape_B(
                cute.slice_((self.D, self.BLK, self.D), (0, None, None))
            ),
            self.dtype,
        )
        tZrK_cv = z_thr_copy_B.retile(tZrK)
        tZsK = z_thr_copy_B.partition_S(sK_SD)
        if cutlass.const_expr(is_first_block):
            # The first block's transfer matrix is the identity, so the GEMM
            # that would apply it is skipped rather than run against I.
            tTRANSrZ = tXrK
        else:
            tZrTrans = SM80.make_acc_into_op(tTransfer_VK, z_mma, self.dtype)
            tZrZ = z_thr.make_fragment_C(z_thr.partition_shape_C((self.D, self.BLK)))
            cute.copy(z_copy_B, tZsK, tZrK_cv)
            tZrZ.fill(self.acc_dtype(0.0))
            cute.gemm(z_mma, tZrZ, tZrTrans, tZrK, tZrZ)
            tTRANSrZ = SM80.make_acc_into_op(tZrZ, trans_mma, self.dtype)

        tTRANSrX = load_tensor_as_b(
            sX_DS, trans_mma, thread_idx, (self.D, self.BLK), self.dtype, True
        )
        cute.gemm(trans_mma, tTransfer_VK, tTRANSrZ, tTRANSrX, tTransfer_VK)
        self._scale_acc(tTransfer_VK, block_coeff)

        # State: Y = gamma_end S K^T - gamma_end V^T Gamma^-1, S <- gamma_end S + Y X.
        tYrY = y_thr.make_fragment_C(y_thr.partition_shape_C((self.D, self.BLK)))
        tYcK = y_thr.partition_C(cK)
        tYrV = self._load_v_as_fragment_C(
            sV, y_mma, y_thr, thread_idx, (self.D, self.BLK)
        )

        tYrK = tZrK
        if cutlass.const_expr(not is_first_block):
            tYrS = SM80.make_acc_into_op(tState_VK, y_mma, self.dtype)
            self._math_order_wait(wg_idx)
            # K is already in tZrK from the transfer above and sK_SD has not
            # changed since; re-reading it inside the ordered section serialised
            # an ldmatrix across both warp groups for nothing.
            tYrY.fill(self.acc_dtype(0.0))
            cute.gemm(y_mma, tYrY, tYrS, tYrK, tYrY)
            self._math_order_notify(wg_idx)
            for i in cutlass.range(cute.size(tYrY), unroll_full=True):
                _, token = tYcK[i]
                scaled = tYrV[i] * sAlpha_blk[token, AlphaProcessor.CUMPROD_NEG_END_RCP]
                tYrY[i] = block_coeff * tYrY[i] + scaled
        else:
            cute.basic_copy(tYrV, tYrY)
            # The scale reads its coordinate from `tYcK`, which is
            # `y_thr.partition_C`. Applying it to `tYrY`, which is
            # `y_thr.make_fragment_C`, keeps value and coordinate on one
            # partition; applying it to `tYrV` did not.
            for i in cutlass.range(cute.size(tYrY), unroll_full=True):
                _, token = tYcK[i]
                tYrY[i] = (
                    tYrY[i] * sAlpha_blk[token, AlphaProcessor.CUMPROD_NEG_END_RCP]
                )

        tSTATErY = SM80.make_acc_into_op(tYrY, state_mma, self.dtype)
        tSTATErX = load_tensor_as_b(
            sX_DS, state_mma, thread_idx, (self.D, self.BLK), self.dtype, True
        )
        self._scale_acc(tState_VK, block_coeff)
        self._math_order_wait(wg_idx)
        cute.gemm(state_mma, tState_VK, tSTATErY, tSTATErX, tState_VK)
        self._math_order_notify(wg_idx)
        cute.arch.barrier(
            barrier_id=NamedBarrier.MATH_SYNC, number_of_threads=THREADS_MN
        )

    # ─── ordered barriers between the two math warp groups ────────────────────

    @cute.jit
    def _math_order_init(self, wg_idx: cutlass.Int32):
        """Open the two-warp-group handshake: WG1 lets WG0 start."""
        if wg_idx == cutlass.Int32(1):
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=THREADS_MN
            )

    @cute.jit
    def _math_order_wait(self, wg_idx: cutlass.Int32):
        """Wait for the other warp group's turn on the shared buffer."""
        if wg_idx == cutlass.Int32(0):
            cute.arch.barrier(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=THREADS_MN
            )
        else:
            cute.arch.barrier(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=THREADS_MN
            )

    @cute.jit
    def _math_order_notify(self, wg_idx: cutlass.Int32):
        """Hand the shared buffer to the other warp group."""
        if wg_idx == cutlass.Int32(0):
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG1, number_of_threads=THREADS_MN
            )
        else:
            cute.arch.barrier_arrive(
                barrier_id=NamedBarrier.MATH_WG0, number_of_threads=THREADS_MN
            )

    # ─── entry ────────────────────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        g_t: cute.Tensor,
        g_alpha: cute.Tensor,
        g_transfer_VK: cute.Tensor,
        g_state_VK: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        k_sd, v_sd, t_layout, x_ds, alpha_layout = self._layouts()

        @cute.struct
        class SharedStorage:
            # No mbarrier storage: the pipelines here count cp.async groups.
            #
            # Order and alignment follow sm120's exactly. X is written to shared
            # inside a block while V is still being read from it, so a layout
            # that puts them differently is a corruption of V and of nothing
            # else -- which is the shape of a failure where the transfer is
            # right to ten decimal places and only the state is wrong.
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_sd)], 128
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_sd)], 128
            ]
            smem_x: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(x_ds)], 128
            ]
            smem_t: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(t_layout)], 128
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(alpha_layout)], 128
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            g_alpha,
            g_k,
            g_v,
            g_t,
            g_transfer_VK,
            g_state_VK,
            cu_seqlens,
            chunk_len,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            total_cp_chunks,
            total_t_blocks,
            num_seqs,
        ).launch(
            grid=(num_sab_heads * max_cp_chunks_per_seq, num_seqs, 1),
            block=(THREADS_MN, 1, 1),
            max_number_threads=(THREADS_MN, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        g_alpha: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        gT_full: cute.Tensor,
        g_transfer_VK: cute.Tensor,
        g_state_VK: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        num_seqs: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        wg_idx = tidx // cutlass.Int32(128)
        bx, seq_idx, _ = cute.arch.block_idx()
        sab_head_idx = bx % num_sab_heads
        k_head_idx = sab_head_idx * num_k_heads // num_sab_heads
        v_head_idx = sab_head_idx * num_v_heads // num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        seq_start = cu_seqlens[seq_idx]
        seq_len = cutlass.Int32(cu_seqlens[seq_idx + 1] - seq_start)
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)

        if chunk_idx_in_seq < num_cp_chunks:
            tok_offset = seq_start + chunk_idx_in_seq * chunk_len
            cp_chunk_idx = varlen_chunk_idx(
                seq_idx, seq_start, chunk_idx_in_seq, chunk_len
            )
            valid_chunk_len = varlen_chunk_valid_len(
                seq_len, chunk_idx_in_seq, chunk_len
            )
            num_blocks = (valid_chunk_len + self.BLK - 1) // self.BLK
            t_blocks_per_cp_chunk = (chunk_len + self.BLK - 1) // self.BLK
            t_block_start = varlen_chunk_idx(
                seq_idx,
                seq_start,
                chunk_idx_in_seq * t_blocks_per_cp_chunk,
                self.BLK,
            )

            k_sd, v_sd, t_layout, x_ds, alpha_layout = self._layouts()
            k_ds = cute.select(k_sd, [1, 0, 2])
            v_ds = cute.select(v_sd, [1, 0, 2])

            allocator = cutlass.utils.SmemAllocator()
            storage = allocator.allocate(self.shared_storage)
            sK_DS = storage.smem_k.get_tensor(k_ds.outer, swizzle=k_ds.inner)
            sV_DS = storage.smem_v.get_tensor(v_ds.outer, swizzle=v_ds.inner)
            sT = storage.smem_t.get_tensor(t_layout.outer, swizzle=t_layout.inner)
            sX_DS = storage.smem_x.get_tensor(x_ds)
            sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

            # T is stored per 64-token block, so its tile is addressed by block
            # index rather than by token offset.
            mT = cute.make_tensor(
                gT_full.iterator,
                cute.make_ordered_layout(
                    (self.BLK, self.BLK, num_sab_heads, total_t_blocks),
                    order=(0, 1, 2, 3),
                ),
            )

            self._math_order_init(wg_idx)

            # The transfer starts as the identity and the state at zero: an
            # empty chunk's affine map is `state -> state`.
            acc_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
                cute.make_layout((THREADS_MN // 32, 1, 1)),
                permutation_mnk=(self.D, self.D, self.D),
            )
            acc_thr = acc_mma.get_slice(tidx)
            tTransfer_VK = acc_thr.make_fragment_C(
                acc_thr.partition_shape_C((self.D, self.D))
            )
            tState_VK = acc_thr.make_fragment_C(
                acc_thr.partition_shape_C((self.D, self.D))
            )
            tTransferC = acc_thr.partition_C(
                cute.make_identity_tensor((self.D, self.D))
            )
            tState_VK.fill(self.acc_dtype(0.0))
            for i in cutlass.range(cute.size(tTransfer_VK), unroll_full=True):
                v_row, k_col = tTransferC[i]
                value = cutlass.Float32(0.0)
                if v_row == k_col:
                    value = cutlass.Float32(1.0)
                tTransfer_VK[i] = value

            for blk in cutlass.range(0, num_blocks, 1, unroll=1):
                stage = blk % cutlass.Int32(self.k_stage)
                block_start = blk * self.BLK
                valid_len = valid_chunk_len - block_start
                if valid_len > cutlass.Int32(self.BLK):
                    valid_len = cutlass.Int32(self.BLK)

                self._load_dn_tile(
                    gK_full,
                    sK_DS,
                    stage,
                    cutlass.Int32(0),
                    tok_offset + block_start,
                    k_head_idx,
                    self.D,
                    valid_len,
                    tidx,
                )
                self._load_dn_tile(
                    gV_full,
                    sV_DS,
                    stage,
                    cutlass.Int32(0),
                    tok_offset + block_start,
                    v_head_idx,
                    self.D,
                    valid_len,
                    tidx,
                )
                self._load_t_tile(
                    mT, sT, stage, sab_head_idx, t_block_start + blk, tidx
                )
                cute.arch.cp_async_commit_group()
                self._load_alpha_block(
                    g_alpha,
                    sAlpha,
                    stage,
                    tok_offset,
                    sab_head_idx,
                    blk,
                    valid_chunk_len,
                    num_sab_heads,
                    tidx,
                )
                cute.arch.cp_async_wait_group(0)
                cute.arch.barrier()

                # The scan reads and writes one channel, so exactly one warp
                # may run it; a second races the first between its load and its
                # store.
                if tidx < cutlass.Int32(32):
                    AlphaProcessor().run(
                        sAlpha[None, None, stage], cutlass.Float32(1.0), True
                    )
                self._mask_alpha_oob(sAlpha, stage, blk, valid_chunk_len, tidx)
                cute.arch.barrier()

                if blk == cutlass.Int32(0):
                    self._compute_block(
                        sV_DS,
                        sK_DS,
                        sT,
                        sX_DS,
                        sAlpha,
                        stage,
                        tTransfer_VK,
                        tState_VK,
                        valid_len,
                        tidx,
                        wg_idx,
                        True,
                    )
                else:
                    self._compute_block(
                        sV_DS,
                        sK_DS,
                        sT,
                        sX_DS,
                        sAlpha,
                        stage,
                        tTransfer_VK,
                        tState_VK,
                        valid_len,
                        tidx,
                        wg_idx,
                        False,
                    )

            out_layout = cute.make_layout(
                (self.D, self.D, num_sab_heads, total_cp_chunks),
                stride=(
                    self.D,
                    1,
                    self.D * self.D,
                    self.D * self.D * num_sab_heads,
                ),
            )
            mTransfer_VK = cute.make_tensor(g_transfer_VK.iterator, out_layout)
            mState_VK = cute.make_tensor(g_state_VK.iterator, out_layout)
            self._store_acc_VK(
                tTransfer_VK,
                mTransfer_VK[None, None, sab_head_idx, cp_chunk_idx],
                acc_mma,
                tidx,
            )
            self._store_acc_VK(
                tState_VK,
                mState_VK[None, None, sab_head_idx, cp_chunk_idx],
                acc_mma,
                tidx,
            )

    @cute.jit
    def _load_t_tile(
        self,
        mT: cute.Tensor,
        sT: cute.Tensor,
        stage: cutlass.Int32,
        head_idx: cutlass.Int32,
        t_block: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """One (BLK, BLK) T tile, indexed by block rather than by token."""
        gT = mT[None, None, head_idx, t_block]
        # T is transposed on the way in, and that transpose is the whole reason
        # this is a scalar copy rather than cp.async.
        #
        # The MMA computes C = A . B^T. The transfer's A is K as (D, BLK) and
        # its B is `X = K . T`, so it forms K^T . T^T . K, while the reference
        # forms K^T . T . K -- `block_transfer = gamma I + gamma (K^T T K)` in
        # `tests/gdn/reference_delta_rule.py`. T is upper triangular, not
        # symmetric, so the two differ. Transposing here is what makes them
        # agree, and stage 2 goes from 60 failing to 60 passing.
        #
        # Slow. Restoring cp.async means folding this transpose into its TV
        # pattern -- and checking the result against a rank, since the load that
        # was here before delivered T at rank 16 and a shape check saw nothing.
        _dst = sT[None, None, stage]
        for _i in cutlass.range(self.BLK * self.BLK // THREADS_MN, unroll_full=True):
            _f = _i * THREADS_MN + tid
            _dst[_f % self.BLK, _f // self.BLK] = gT[_f // self.BLK, _f % self.BLK]


class CPDeltaRuleTPrecomputePtrSm80(CPDeltaRuleTPrecomputeSm80):
    """The T precompute, entered through pointers instead of tensors.

    `generate_execution_args` is 684 us of the CP path's 719 us of host time
    per call, against 35 us for the four launches themselves -- see
    `cp_host_split.py` in the measurement harness. The cost is per tensor
    argument: a DLPack capsule from the torch tensor, a CuTe runtime tensor
    around it, a dynamic-layout marking and a set of C pointers, all in Python,
    ten tensors deep and four entries wide.

    So this entry takes what the kernel actually needs -- a typed pointer per
    buffer and the extents that are not compile-time constants -- and rebuilds
    the same layouts inside the JIT, where they cost nothing. The kernel body
    is inherited unchanged; only the way its arguments arrive differs.

    A separate class rather than a flag, so the two entries cannot share a
    compile-cache entry: `manual_cache_key` puts `type(self).__mro__` in the
    key, and a flag named only in a base class's call is invisible to a
    subclass that calls it again.
    """

    @cute.jit
    def __call__(
        self,
        k_ptr: cute.Pointer,
        beta_ptr: cute.Pointer,
        t_ptr: cute.Pointer,
        cu_seqlens_ptr: cute.Pointer,
        total_seqlen: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        max_t_blocks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        # The same three views the tensor entry is handed, rebuilt here.
        # `k` arrives as (D, token, head) over the caller's (token, head, D)
        # storage: D contiguous is what the 128-bit cp.async needs on its fast
        # axis, and it is a view, not a copy.
        g_k = cute.make_tensor(
            k_ptr,
            cute.make_layout(
                (self.D, total_seqlen, num_k_heads),
                stride=(1, num_k_heads * self.D, self.D),
            ),
        )
        g_beta = cute.make_tensor(
            beta_ptr, cute.make_layout(total_seqlen * num_sab_heads)
        )
        g_t = cute.make_tensor(
            t_ptr,
            cute.make_layout(total_t_blocks * num_sab_heads * self.BLK * self.BLK),
        )
        g_cu = cute.make_tensor(cu_seqlens_ptr, cute.make_layout(num_seqs + 1))
        super().__call__(
            g_k,
            g_beta,
            g_t,
            g_cu,
            num_k_heads,
            num_sab_heads,
            total_t_blocks,
            max_t_blocks_per_seq,
            num_seqs,
            stream,
        )


@functools.cache
def _get_t_precompute_ptr_kernel(kernel_dtype, cu_seqlens_dtype):
    return CPDeltaRuleTPrecomputePtrSm80(
        kernel_dtype, cu_seqlens_dtype=integer_dtype_to_cutlass(cu_seqlens_dtype)
    )


@functools.cache
def _get_t_precompute_kernel(kernel_dtype, cu_seqlens_dtype):
    return CPDeltaRuleTPrecomputeSm80(
        kernel_dtype, cu_seqlens_dtype=integer_dtype_to_cutlass(cu_seqlens_dtype)
    )


def cp_delta_rule_t_precompute_dsl_sm80(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    max_seqlen: int | None = None,
    *,
    # Opt-in, and off by default until the four entries have all moved: this
    # entry's arguments arrive as pointers and extents rather than as tensors.
    _ptr_abi: bool = False,
    # One composition call's shared state, or None for a standalone
    # call. Never built here: see `_CPInvocation`.
    _ctx: "_CPInvocation | None" = None,
    _skip_check: bool = False,
    # When given, the arguments this entry would hand the compiled kernel
    # are appended to it and nothing is launched. That is how a prepared
    # execution is built: explicitly, at this layer, rather than by patching
    # the DSL's own classes from outside. The tuple is (compiled, args), and
    # the caller owns whatever it keeps alive.
    _plan_sink: list | None = None,
    _device=None,
    _stream=None,
):
    """Signed, beta-folded T tiles: `T = -(inv(IKK) beta)^T`.

    Returns `(total_t_blocks, num_sab_heads, 64, 64)` over flat varlen input,
    where `total_t_blocks` counts every 64-token chunk of every sequence.

    Inputs must already be contiguous. This validates that contract rather than
    materialising copies of anything.
    """
    import cuda.bindings.driver as cuda_driver

    device = k.device if _device is None else _device
    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if beta.ndim != 2 or beta.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"beta must have shape (total_seqlen, num_sab_heads), got "
                f"{tuple(beta.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and "
                f"{k.shape[0]}"
            )
        if not is_integer_dtype(cu_seqlens.dtype):
            raise RuntimeError(
                f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_k_heads, d = k.shape
    num_sab_heads = beta.shape[1]
    if not _skip_check:
        if num_sab_heads < num_k_heads or num_sab_heads % num_k_heads != 0:
            raise RuntimeError(
                "beta heads must be a positive multiple of k heads, "
                f"got beta heads={num_sab_heads} and k heads={num_k_heads}"
            )
        if d != 128:
            raise RuntimeError(f"the SM80 CP T precompute needs D=128, got {d}")
        if k.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"the SM80 CP T precompute needs fp16 or bf16 k, got {k.dtype}"
            )
        if beta.dtype != torch.float32:
            raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
        for name, tensor in (("k", k), ("beta", beta)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
        # cp.async wants 16 B, and the layout algebra cannot prove it of a
        # runtime stride, so the premise is checked here instead.
        _check_load_alignment(k=k)

    if _ctx is not None:
        total_t_blocks = _ctx.total_t_blocks
        max_t_blocks_per_seq = _ctx.max_t_blocks_per_seq
    else:
        total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
        max_t_blocks_per_seq = max_num_chunks_host(max_seqlen, 64)
    t = _cp_workspace(
        "gdn_cp_sm80_t", (total_t_blocks, num_sab_heads, 64, 64), k.dtype, device
    )
    if total_t_blocks == 0:
        return t

    # (D, token, head) over the same storage: d is contiguous, which is what the
    # 128-bit cp.async needs on its fast axis.
    k_dhs = k.as_strided((d, total_seqlen, num_k_heads), (1, num_k_heads * d, d))

    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[k.dtype]
    stream = (
        _ctx.stream
        if _ctx is not None
        else _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )
    compile_options = (
        _ctx.compile_options if _ctx is not None else _sm80_cp_compile_options(device)
    )
    if _ptr_abi:
        # Pointers and extents instead of tensors. The kernel body is the same
        # one; only the argument conversion the DSL does per call is gone. See
        # `CPDeltaRuleTPrecomputePtrSm80`.
        #
        mk = _ptr_factory(
            _ctx,
            device,
            _stream,
            (
                ("k", k, 16),
                ("beta", beta, 16),
                ("cu_seqlens", cu_seqlens, 8),
                ("t", t, 128),
            ),
        )
        kernel = _get_t_precompute_ptr_kernel(kernel_dtype, cu_seqlens.dtype)
        args = (
            mk(kernel_dtype, k, 16, "k"),
            mk(cutlass.Float32, beta, 16, "beta"),
            mk(kernel_dtype, t, 128, "t"),
            mk(integer_dtype_to_cutlass(cu_seqlens.dtype), cu_seqlens, 8, "cu_seqlens"),
            cutlass.Int32(total_seqlen),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(total_t_blocks),
            cutlass.Int32(max_t_blocks_per_seq),
            cutlass.Int32(num_seqs),
            stream,
        )
        compiled = get_cached_compile(kernel, compile_options)
        if compiled is None:
            compiled = cached_compile(kernel, *args, compile_options=compile_options)
        if _plan_sink is not None:
            _plan_sink.append((compiled, args))
        else:
            compiled(*args)
        return t
    kernel = _get_t_precompute_kernel(kernel_dtype, cu_seqlens.dtype)
    compiled = get_cached_compile(kernel, compile_options)
    if compiled is None:
        from_dlpack = cute.runtime.from_dlpack
        kernel_args = (
            from_dlpack(k_dhs, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(t.view(-1), assumed_align=128).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(total_t_blocks),
            cutlass.Int32(max_t_blocks_per_seq),
            cutlass.Int32(num_seqs),
            stream,
        )
        compiled = cached_compile(kernel, *kernel_args, compile_options=compile_options)
    if _plan_sink is not None:
        _plan_sink.append(
            (
                compiled,
                (
                    k_dhs,
                    beta.reshape(-1),
                    t.view(-1),
                    cu_seqlens,
                    num_k_heads,
                    num_sab_heads,
                    total_t_blocks,
                    max_t_blocks_per_seq,
                    num_seqs,
                    stream,
                ),
            )
        )
    else:
        compiled(
            k_dhs,
            beta.reshape(-1),
            t.view(-1),
            cu_seqlens,
            num_k_heads,
            num_sab_heads,
            total_t_blocks,
            max_t_blocks_per_seq,
            num_seqs,
            stream,
        )
    return t


class CPDeltaRuleMNPrecomputePtrSm80(CPDeltaRuleMNPrecomputeSm80):
    """The MN precompute, entered through pointers instead of tensors.

    Same trade as `CPDeltaRuleTPrecomputePtrSm80`: seven tensor arguments cost
    seven DLPack conversions and seven CuTe runtime tensors per call, in
    Python, and the extents they carry are already known here as integers. The
    kernel body is inherited unchanged.
    """

    @cute.jit
    def __call__(
        self,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        t_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        transfer_ptr: cute.Pointer,
        state_ptr: cute.Pointer,
        cu_seqlens_ptr: cute.Pointer,
        total_seqlen: cutlass.Int32,
        chunk_len: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        stream,
    ):
        # (D, token, head) over the caller's (token, head, D) storage, as the
        # tensor entry builds with `as_strided`: D contiguous is what the
        # 128-bit cp.async needs on its fast axis.
        g_k = cute.make_tensor(
            k_ptr,
            cute.make_layout(
                (self.D, total_seqlen, num_k_heads),
                stride=(1, num_k_heads * self.D, self.D),
            ),
        )
        g_v = cute.make_tensor(
            v_ptr,
            cute.make_layout(
                (self.D, total_seqlen, num_v_heads),
                stride=(1, num_v_heads * self.D, self.D),
            ),
        )
        g_t = cute.make_tensor(
            t_ptr, cute.make_layout(total_t_blocks * num_sab_heads * 64 * 64)
        )
        g_alpha = cute.make_tensor(
            alpha_ptr, cute.make_layout(total_seqlen * num_sab_heads)
        )
        chunk_elems = total_cp_chunks * num_sab_heads * self.D * self.D
        g_transfer = cute.make_tensor(transfer_ptr, cute.make_layout(chunk_elems))
        g_state = cute.make_tensor(state_ptr, cute.make_layout(chunk_elems))
        g_cu = cute.make_tensor(cu_seqlens_ptr, cute.make_layout(num_seqs + 1))
        super().__call__(
            g_k,
            g_v,
            g_t,
            g_alpha,
            g_transfer,
            g_state,
            g_cu,
            chunk_len,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            total_cp_chunks,
            total_t_blocks,
            max_cp_chunks_per_seq,
            num_seqs,
            stream,
        )


@functools.cache
def _get_mn_precompute_ptr_kernel(kernel_dtype, cu_seqlens_dtype):
    return CPDeltaRuleMNPrecomputePtrSm80(
        kernel_dtype, cu_seqlens_dtype=integer_dtype_to_cutlass(cu_seqlens_dtype)
    )


@functools.cache
def _get_mn_precompute_kernel(kernel_dtype, cu_seqlens_dtype):
    return CPDeltaRuleMNPrecomputeSm80(
        kernel_dtype, cu_seqlens_dtype=integer_dtype_to_cutlass(cu_seqlens_dtype)
    )


def cp_delta_rule_mn_precompute_dsl_sm80(
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    max_seqlen: int | None = None,
    *,
    # Opt-in; see `CPDeltaRuleMNPrecomputePtrSm80`.
    _ptr_abi: bool = False,
    # One composition call's shared state, or None for a standalone
    # call. Never built here: see `_CPInvocation`.
    _ctx: "_CPInvocation | None" = None,
    _skip_check: bool = False,
    # When given, the arguments this entry would hand the compiled kernel
    # are appended to it and nothing is launched. That is how a prepared
    # execution is built: explicitly, at this layer, rather than by patching
    # the DSL's own classes from outside. The tuple is (compiled, args), and
    # the caller owns whatever it keeps alive.
    _plan_sink: list | None = None,
    _device=None,
    _stream=None,
):
    """Each chunk's local affine map, in the workspace layout stage 3 reads.

    Returns `(transfer_VK, state_VK)`, both
    `(total_cp_chunks, num_sab_heads, DimV, DimK)`. The names carry the
    orientation because nothing else can: DimV and DimK are both 128, so a
    transposed write is the same shape and only the numbers come out wrong.
    See `CP_LAYOUTS.md`.
    """
    import cuda.bindings.driver as cuda_driver

    device = k.device if _device is None else _device
    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if v.ndim != 3 or v.shape[0] != k.shape[0] or v.shape[2] != k.shape[2]:
            raise RuntimeError(
                f"v must have shape (total_seqlen, num_v_heads, D), got {tuple(v.shape)}"
            )
        if alpha.ndim != 2 or alpha.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"alpha must have shape (total_seqlen, num_sab_heads), got "
                f"{tuple(alpha.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and "
                f"{k.shape[0]}"
            )
        if not is_integer_dtype(cu_seqlens.dtype):
            raise RuntimeError(
                f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
            )
        if cp_chunk_len % 64 != 0 or cp_chunk_len <= 0:
            raise RuntimeError(
                f"cp_chunk_len must be a positive multiple of 64, got {cp_chunk_len}"
            )
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    _, num_k_heads, d = k.shape
    num_v_heads = v.shape[1]
    num_sab_heads = alpha.shape[1]
    if not _skip_check:
        if d != 128:
            raise RuntimeError(f"the SM80 CP MN precompute needs D=128, got {d}")
        if k.dtype not in (torch.float16, torch.bfloat16) or v.dtype != k.dtype:
            raise RuntimeError(
                f"k and v must share an fp16/bf16 dtype, got {k.dtype} and {v.dtype}"
            )
        if alpha.dtype != torch.float32:
            raise RuntimeError(
                f"alpha must have dtype torch.float32, got {alpha.dtype}"
            )
        for name, tensor in (("k", k), ("v", v), ("alpha", alpha), ("t", t)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
        _check_load_alignment(k=k, v=v)

    total_t_blocks = (
        _ctx.total_t_blocks
        if _ctx is not None
        else workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    )
    if not _skip_check and tuple(t.shape) != (total_t_blocks, num_sab_heads, 64, 64):
        raise RuntimeError(
            f"t must have shape {(total_t_blocks, num_sab_heads, 64, 64)}, got "
            f"{tuple(t.shape)}"
        )
    if _ctx is not None:
        total_cp_chunks = _ctx.total_cp_chunks
        max_cp_chunks_per_seq = _ctx.max_cp_chunks_per_seq
    else:
        total_cp_chunks = workspace_num_chunks_host(
            cu_seqlens, cp_chunk_len, total_seqlen
        )
        max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)

    workspace_shape = (total_cp_chunks, num_sab_heads, d, d)
    transfer_VK = _cp_workspace(
        "gdn_cp_sm80_local_transfer", workspace_shape, torch.float32, device
    )
    state_VK = _cp_workspace(
        "gdn_cp_sm80_local_state", workspace_shape, torch.float32, device
    )
    if total_cp_chunks == 0:
        return transfer_VK, state_VK

    # (D, token, head): d contiguous, which is what the 128-bit cp.async needs
    # on its fast axis.
    k_dhs = k.as_strided((d, total_seqlen, num_k_heads), (1, num_k_heads * d, d))
    v_dhs = v.as_strided((d, total_seqlen, num_v_heads), (1, num_v_heads * d, d))

    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[k.dtype]
    stream = (
        _ctx.stream
        if _ctx is not None
        else _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )
    compile_options = (
        _ctx.compile_options if _ctx is not None else _sm80_cp_compile_options(device)
    )
    if _ptr_abi:
        mk = _ptr_factory(
            _ctx,
            device,
            _stream,
            (
                ("k", k, 16),
                ("v", v, 16),
                ("alpha", alpha, 16),
                ("t", t, 16),
                ("transfer_VK", transfer_VK, 16),
                ("state_VK", state_VK, 16),
                ("cu_seqlens", cu_seqlens, 8),
            ),
        )
        kernel = _get_mn_precompute_ptr_kernel(kernel_dtype, cu_seqlens.dtype)
        args = (
            mk(kernel_dtype, k, 16, "k"),
            mk(kernel_dtype, v, 16, "v"),
            mk(kernel_dtype, t, 16, "t"),
            mk(cutlass.Float32, alpha, 16, "alpha"),
            mk(cutlass.Float32, transfer_VK, 16, "transfer_VK"),
            mk(cutlass.Float32, state_VK, 16, "state_VK"),
            mk(integer_dtype_to_cutlass(cu_seqlens.dtype), cu_seqlens, 8, "cu_seqlens"),
            cutlass.Int32(total_seqlen),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(total_cp_chunks),
            cutlass.Int32(total_t_blocks),
            cutlass.Int32(max_cp_chunks_per_seq),
            cutlass.Int32(num_seqs),
            stream,
        )
        compiled = get_cached_compile(kernel, compile_options)
        if compiled is None:
            compiled = cached_compile(kernel, *args, compile_options=compile_options)
        if _plan_sink is not None:
            _plan_sink.append((compiled, args))
        else:
            compiled(*args)
        return transfer_VK, state_VK
    kernel = _get_mn_precompute_kernel(kernel_dtype, cu_seqlens.dtype)
    compiled = get_cached_compile(kernel, compile_options)
    if compiled is None:
        from_dlpack = cute.runtime.from_dlpack
        kernel_args = (
            from_dlpack(k_dhs, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(v_dhs, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(t.view(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(alpha.view(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(transfer_VK.view(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(state_VK.view(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(total_cp_chunks),
            cutlass.Int32(total_t_blocks),
            cutlass.Int32(max_cp_chunks_per_seq),
            cutlass.Int32(num_seqs),
            stream,
        )
        compiled = cached_compile(kernel, *kernel_args, compile_options=compile_options)
    if _plan_sink is not None:
        _plan_sink.append(
            (
                compiled,
                (
                    k_dhs,
                    v_dhs,
                    t.view(-1),
                    alpha.view(-1),
                    transfer_VK.view(-1),
                    state_VK.view(-1),
                    cu_seqlens,
                    cp_chunk_len,
                    num_k_heads,
                    num_v_heads,
                    num_sab_heads,
                    total_cp_chunks,
                    total_t_blocks,
                    max_cp_chunks_per_seq,
                    num_seqs,
                    stream,
                ),
            )
        )
    else:
        compiled(
            k_dhs,
            v_dhs,
            t.view(-1),
            alpha.view(-1),
            transfer_VK.view(-1),
            state_VK.view(-1),
            cu_seqlens,
            cp_chunk_len,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            total_cp_chunks,
            total_t_blocks,
            max_cp_chunks_per_seq,
            num_seqs,
            stream,
        )
    return transfer_VK, state_VK


__all__ = [
    "CPDeltaRuleTPrecomputeSm80",
    "CPDeltaRuleMNPrecomputeSm80",
    "cp_delta_rule_t_precompute_dsl_sm80",
    "cp_delta_rule_mn_precompute_dsl_sm80",
]


def _get_cp_workspace(name, shape, dtype, device):
    """A cached device buffer of this shape, viewed as the requested dtype."""
    nbytes = math.prod(shape) * dtype.itemsize
    return _get_cache_buf(name, nbytes, device)[:nbytes].view(dtype).view(shape)


# Only the one symbol the rest of this file needs and the header does not
# already import. The block used to re-import six names that were either
# already bound at the top or never used, which ruff reads as F811/F401.
from .helpers import state_dtype_to_cutlass


class CPDeltaRuleFixupSimtSm80(KeyedCompileMixin):
    def __init__(
        self,
        needs_initial_state: bool = False,
        initial_state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        rows_per_cta: int = 4,
        use_state_indices: bool = False,
        cu_seqlens_dtype: torch.dtype = torch.int64,
        state_indices_dtype: torch.dtype | None = None,
        initial_state_inner_strides: tuple[int, ...] | None = None,
    ):
        self.needs_initial_state = needs_initial_state
        self.initial_state_dtype = initial_state_dtype
        self.use_state_indices = use_state_indices
        self.cu_seqlens_dtype = cu_seqlens_dtype
        self.state_indices_dtype = state_indices_dtype
        self.initial_state_inner_strides = initial_state_inner_strides
        self.D = 128
        self.rows_per_cta = rows_per_cta
        self.row_ctas = self.D // self.rows_per_cta
        self.threads_per_cta = 128
        self.num_warps = 4
        self.min_blocks_per_mp = 2
        self.registers_per_thread = 256
        self.manual_cache_key(
            "needs_initial_state",
            "initial_state_dtype",
            "use_state_indices",
            "cu_seqlens_dtype",
            "state_indices_dtype",
            "initial_state_inner_strides",
            "D",
            "rows_per_cta",
            "row_ctas",
            "threads_per_cta",
            "num_warps",
            "min_blocks_per_mp",
            "registers_per_thread",
        )

    @cute.jit
    def init_state_tile(
        self,
        sState: cute.Tensor,
        gFixedState: cute.Tensor,
        gLocalState: cute.Tensor,
        gInitialState: cute.Tensor,
        col: cutlass.Int32,
    ) -> cutlass.Int32:
        """Seed the running state tile from the caller's state, or from zero."""
        start = cutlass.Int32(0)
        if cutlass.const_expr(self.needs_initial_state):
            for i in cutlass.range_constexpr(self.rows_per_cta):
                sState[i, col] = gInitialState[i, col].to(cutlass.Float32)
        else:
            start = cutlass.Int32(1)
            for i in cutlass.range_constexpr(self.rows_per_cta):
                value = gLocalState[i, col, 0]
                sState[i, col] = value
                gFixedState[i, col, 0] = value
        return start

    @cute.jit
    def load_transfer_fragment(
        self,
        rM: cute.Tensor,
        gTransferTile: cute.Tensor,
        chunk_idx: cutlass.Int32,
        k_tile: cutlass.Int32,
    ):
        """Read one chunk's transfer matrix into registers."""
        for j in cutlass.range_constexpr(16):
            rM[j] = gTransferTile[(j, chunk_idx), (k_tile, 0)]

    @cute.jit
    def load_local_state_fragment(
        self,
        rAcc: cute.Tensor,
        gLocalState: cute.Tensor,
        chunk_idx: cutlass.Int32,
        col: cutlass.Int32,
    ):
        """Read one chunk's local state contribution into registers."""
        for i in cutlass.range_constexpr(self.rows_per_cta):
            rAcc[i] = gLocalState[i, col, chunk_idx]

    @cute.jit
    def accumulate_state_fragment(
        self,
        rAcc: cute.Tensor,
        sState: cute.Tensor,
        rM: cute.Tensor,
        k: cutlass.Int32,
    ):
        """Fold one chunk into the running state: `acc + state @ M`."""
        for i in cutlass.range_constexpr(self.rows_per_cta):
            for j in cutlass.range_constexpr(16):
                rAcc[i] = rAcc[i] + sState[i, k + cutlass.Int32(j)] * rM[j]

    @cute.jit
    def run_simt_fixup_loop(
        self,
        sState: cute.Tensor,
        gTransfer: cute.Tensor,
        gLocalState: cute.Tensor,
        gFixedState: cute.Tensor,
        num_chunks: cutlass.Int32,
        col: cutlass.Int32,
        start: cutlass.Int32,
    ):
        """Scan a sequence's chunks in order, carrying the state forward.

        The one serial stage, and serial over chunks rather than tokens: each
        step folds a chunk's local contribution into the running state and
        writes the absolute state that chunk's prefill will read."""
        rAcc = cute.make_rmem_tensor(self.rows_per_cta, cutlass.Float32)
        rM = cute.make_rmem_tensor(16, cutlass.Float32)
        rM_next = cute.make_rmem_tensor(16, cutlass.Float32)
        # ((k_in_tile, chunk_idx), (k_tile, _)); column is fixed by this thread.
        gTransferTile = cute.zipped_divide(gTransfer[None, col, None], (16, num_chunks))
        k_tiles = cute.size(gTransferTile, mode=[1, 0])
        last_k_tile = k_tiles - 1
        rAccNext = cute.make_rmem_tensor(self.rows_per_cta, cutlass.Float32)
        if start < num_chunks:
            self.load_local_state_fragment(rAcc, gLocalState, start, col)
            self.load_transfer_fragment(rM, gTransferTile, start, cutlass.Int32(0))

        for chunk_idx in cutlass.range(start, num_chunks, unroll=1):
            next_chunk_idx = chunk_idx + cutlass.Int32(1)
            for iter_k in cutlass.range_constexpr(last_k_tile):
                self.load_transfer_fragment(
                    rM_next,
                    gTransferTile,
                    chunk_idx,
                    cutlass.Int32(iter_k + 1),
                )
                self.accumulate_state_fragment(
                    rAcc, sState, rM, cutlass.Int32(iter_k * 16)
                )
                for j in cutlass.range_constexpr(16):
                    rM[j] = rM_next[j]

            # Last K tile owns the inter-chunk handoff: preload next chunk
            # before the final accumulation, then publish the new state.
            if next_chunk_idx < num_chunks:
                self.load_local_state_fragment(
                    rAccNext,
                    gLocalState,
                    next_chunk_idx,
                    col,
                )
                self.load_transfer_fragment(
                    rM_next,
                    gTransferTile,
                    next_chunk_idx,
                    cutlass.Int32(0),
                )
            self.accumulate_state_fragment(
                rAcc, sState, rM, cutlass.Int32(last_k_tile * 16)
            )
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(self.rows_per_cta):
                value = rAcc[i]
                sState[i, col] = value
                gFixedState[i, col, chunk_idx] = value
            cute.arch.sync_threads()
            if next_chunk_idx < num_chunks:
                for i in cutlass.range_constexpr(self.rows_per_cta):
                    rAcc[i] = rAccNext[i]
                for j in cutlass.range_constexpr(16):
                    rM[j] = rM_next[j]

    @cute.jit
    def zero_invalid_slots(
        self,
        gFixedState: cute.Tensor,
        gap_len: cutlass.Int32,
        col: cutlass.Int32,
    ):
        """Zero the state rows no sequence owns.

        A zero-length sequence and the gap a ragged pack leaves both produce
        slots nothing writes, and a reader cannot tell an unwritten slot from
        a written zero."""
        for slot in cutlass.range(0, gap_len, unroll=1):
            for i in cutlass.range_constexpr(self.rows_per_cta):
                gFixedState[i, col, slot] = cutlass.Float32(0.0)

    @cute.jit
    def __call__(
        self,
        g_transfer_t: cute.Tensor,
        g_local_state_t: cute.Tensor,
        g_initial_state_t: cute.Tensor,
        g_state_indices_t: cute.Tensor,
        g_fixed_state_t: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
        num_heads: cutlass.Int32,
        stream,
    ):
        state_layout = cute.make_layout((self.rows_per_cta, self.D), stride=(self.D, 1))

        @cute.struct
        class SharedStorage:
            smem_state: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(state_layout)], 128
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            g_transfer_t,
            g_local_state_t,
            g_initial_state_t,
            g_state_indices_t,
            g_fixed_state_t,
            g_cu_seqlens,
            chunk_len,
            total_cp_chunks,
            num_seqs,
            num_heads,
        ).launch(
            grid=(num_seqs * num_heads * self.row_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            max_number_threads=(self.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        g_transfer_t: cute.Tensor,
        g_local_state_t: cute.Tensor,
        g_initial_state_t: cute.Tensor,
        g_state_indices_t: cute.Tensor,
        g_fixed_state_t: cute.Tensor,
        g_cu_seqlens: cute.Tensor,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
        num_heads: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        row_cta_idx = bx % self.row_ctas
        head_seq_idx = bx // self.row_ctas
        head_idx = head_seq_idx % num_heads
        seq_idx = head_seq_idx // num_heads
        seq_start = g_cu_seqlens[seq_idx]
        seq_end = g_cu_seqlens[seq_idx + 1]
        seq_len = cutlass.Int32(seq_end - seq_start)
        num_chunks = chunks_for_len(seq_len, chunk_len)
        chunk_start = varlen_chunk_idx(seq_idx, seq_start, 0, chunk_len)
        gap_start = chunk_start + num_chunks
        gap_end = total_cp_chunks
        if seq_idx + cutlass.Int32(1) < num_seqs:
            gap_end = varlen_chunk_idx(
                seq_idx + cutlass.Int32(1), seq_end, 0, chunk_len
            )

        state_layout = cute.make_layout((self.rows_per_cta, self.D), stride=(self.D, 1))
        out_layout = cute.make_layout(
            (self.D, self.D, num_heads, total_cp_chunks),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads),
        )
        workspace_layout = out_layout
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        gTransfer = cute.make_tensor(g_transfer_t.iterator.align(128), workspace_layout)
        gLocalState = cute.make_tensor(
            g_local_state_t.iterator.align(128), workspace_layout
        )
        gFixedState = cute.make_tensor(g_fixed_state_t.iterator.align(128), out_layout)
        fixed_state_layout = cute.make_layout(
            (self.D, self.D, num_heads, num_seqs),
            stride=(self.D, 1, self.D * self.D, self.D * self.D * num_heads),
        )
        state_idx = seq_idx
        if cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices_t[seq_idx])
        if cutlass.const_expr(self.needs_initial_state):
            initial_state_ref_layout = cute.make_layout(
                (
                    g_initial_state_t.shape[0],
                    g_initial_state_t.shape[1],
                    self.D,
                    self.D,
                ),
                stride=g_initial_state_t.stride,
            )
            indexed_initial_state_layout = cute.select(
                initial_state_ref_layout, mode=[2, 3, 1, 0]
            )
            if cutlass.const_expr(self.use_state_indices):
                initial_state_layout = indexed_initial_state_layout
            else:
                initial_state_layout = fixed_state_layout
            gInitialState = cute.make_tensor(
                g_initial_state_t.iterator, initial_state_layout
            )
        else:
            gInitialState = gFixedState

        sState = storage.smem_state.get_tensor(state_layout)
        gTransfer_head = gTransfer[None, None, head_idx, None]
        gLocalState_cta = cute.local_tile(
            gLocalState[None, None, head_idx, None],
            (self.rows_per_cta, self.D, total_cp_chunks),
            (row_cta_idx, 0, 0),
        )
        gFixedState_cta = cute.local_tile(
            gFixedState[None, None, head_idx, None],
            (self.rows_per_cta, self.D, total_cp_chunks),
            (row_cta_idx, 0, 0),
        )
        gTransfer_seq = cute.domain_offset((0, 0, chunk_start), gTransfer_head)
        gLocalState_seq = cute.domain_offset((0, 0, chunk_start), gLocalState_cta)
        gFixedState_seq = cute.domain_offset((0, 0, chunk_start), gFixedState_cta)
        if cutlass.const_expr(self.needs_initial_state):
            gInitialState_cta = cute.local_tile(
                gInitialState[None, None, head_idx, state_idx],
                (self.rows_per_cta, self.D),
                (row_cta_idx, 0),
            )
        else:
            gInitialState_cta = gFixedState_seq
        col = tidx

        cute.arch.sync_threads()
        if num_chunks > 0:
            start = self.init_state_tile(
                sState,
                gFixedState_seq,
                gLocalState_seq,
                gInitialState_cta,
                col,
            )
            cute.arch.sync_threads()
            self.run_simt_fixup_loop(
                sState,
                gTransfer_seq,
                gLocalState_seq,
                gFixedState_seq,
                num_chunks,
                col,
                start,
            )
        gFixedState_gap = cute.domain_offset((0, 0, gap_start), gFixedState_cta)
        self.zero_invalid_slots(gFixedState_gap, gap_end - gap_start, col)


class CPDeltaRuleFixupPtrSm80(CPDeltaRuleFixupSimtSm80):
    """The SIMT fixup, entered through pointers instead of tensors.

    Every compile-time specialization the tensor entry has is kept on the
    instance -- `needs_initial_state`, `use_state_indices`,
    `initial_state_dtype`, `initial_state_inner_strides`, `rows_per_cta`, the
    index dtypes -- so nothing here is a runtime branch on a nullable pointer.
    The inactive optional slots still receive a valid pointer, and the base
    kernel only reads them under `cutlass.const_expr`, so they are never
    dereferenced.

    What the kernel body actually takes from each argument decides what has to
    arrive:

      transfer, local state and fixed state are used for their iterator alone
      -- the body rebuilds their layouts from `total_cp_chunks` and
      `num_heads` -- so a pointer and a placeholder layout suffice;

      `state_indices` is indexed by sequence, so it needs an extent;

      `initial_state` is the only one whose *layout* is read: `shape[0]`,
      `shape[1]` and `stride`, because the pool is the caller's and its rows
      are addressed through `state_indices`. Those arrive as scalars, with the
      inner strides taken from the specialization when it fixes them, so the
      layout is as static here as it is on the tensor path.
    """

    @cute.jit
    def __call__(
        self,
        transfer_ptr: cute.Pointer,
        local_state_ptr: cute.Pointer,
        initial_state_ptr: cute.Pointer,
        state_indices_ptr: cute.Pointer,
        fixed_state_ptr: cute.Pointer,
        cu_seqlens_ptr: cute.Pointer,
        init_pool_rows: cutlass.Int32,
        init_pool_heads: cutlass.Int32,
        init_stride_0: cutlass.Int32,
        init_stride_1: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
        num_heads: cutlass.Int32,
        stream,
    ):
        one = cute.make_layout(1)
        g_transfer_t = cute.make_tensor(transfer_ptr, one)
        g_local_state_t = cute.make_tensor(local_state_ptr, one)
        g_fixed_state_t = cute.make_tensor(fixed_state_ptr, one)
        g_cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout(num_seqs + 1))
        if cutlass.const_expr(self.use_state_indices):
            g_state_indices_t = cute.make_tensor(
                state_indices_ptr, cute.make_layout(num_seqs)
            )
        else:
            g_state_indices_t = cute.make_tensor(state_indices_ptr, one)
        if cutlass.const_expr(self.needs_initial_state):
            inner = cutlass.const_expr(self.initial_state_inner_strides)
            if cutlass.const_expr(inner is not None):
                # The specialization fixes the pool's inner strides, so they
                # stay compile-time constants here exactly as they do when the
                # DSL derives them from the caller's tensor.
                strides = (init_stride_0, inner[0], inner[1], inner[2])
            else:
                strides = (
                    init_stride_0,
                    init_stride_1,
                    self.D,
                    1,
                )
            g_initial_state_t = cute.make_tensor(
                initial_state_ptr,
                cute.make_layout(
                    (init_pool_rows, init_pool_heads, self.D, self.D),
                    stride=strides,
                ),
            )
        else:
            g_initial_state_t = cute.make_tensor(initial_state_ptr, one)
        super().__call__(
            g_transfer_t,
            g_local_state_t,
            g_initial_state_t,
            g_state_indices_t,
            g_fixed_state_t,
            g_cu_seqlens,
            chunk_len,
            total_cp_chunks,
            num_seqs,
            num_heads,
            stream,
        )


@functools.cache
def _get_fixup_ptr_kernel(
    needs_initial_state,
    initial_state_dtype,
    kernel_kind,
    use_state_indices,
    cu_seqlens_dtype,
    state_indices_dtype,
    initial_state_inner_strides,
):
    """The pointer twin of `_get_fixup_kernel`, with the same specializations."""
    if kernel_kind not in ("simt_row4", "simt_row8"):
        raise ValueError(
            f"the SM80 CP fixup builds simt_row4 and simt_row8, got {kernel_kind}"
        )
    return CPDeltaRuleFixupPtrSm80(
        needs_initial_state=needs_initial_state,
        initial_state_dtype=state_dtype_to_cutlass(initial_state_dtype),
        rows_per_cta=4 if kernel_kind == "simt_row4" else 8,
        use_state_indices=use_state_indices,
        cu_seqlens_dtype=integer_dtype_to_cutlass(cu_seqlens_dtype),
        state_indices_dtype=(
            integer_dtype_to_cutlass(state_indices_dtype)
            if state_indices_dtype is not None
            else None
        ),
        initial_state_inner_strides=initial_state_inner_strides,
    )


@functools.cache
def _get_fixup_kernel(
    needs_initial_state,
    initial_state_dtype,
    kernel_kind,
    use_state_indices,
    cu_seqlens_dtype,
    state_indices_dtype,
    initial_state_inner_strides,
):
    """Only the SIMT kernels. The HMMA variant is not ported.

    `sm8x` has no `setmaxnreg`, which the HMMA fixup leans on to hand its
    math warps the registers its accumulators need. Rather than gate on it,
    the kind simply is not built here and asking for it raises.
    """
    if kernel_kind not in ("simt_row4", "simt_row8"):
        raise ValueError(
            f"the SM80 CP fixup builds simt_row4 and simt_row8, got {kernel_kind}"
        )
    return CPDeltaRuleFixupSimtSm80(
        needs_initial_state=needs_initial_state,
        initial_state_dtype=state_dtype_to_cutlass(initial_state_dtype),
        rows_per_cta=4 if kernel_kind == "simt_row4" else 8,
        use_state_indices=use_state_indices,
        cu_seqlens_dtype=cu_seqlens_dtype,
        state_indices_dtype=state_indices_dtype,
        initial_state_inner_strides=initial_state_inner_strides,
    )


def cp_delta_rule_fixup_dsl_sm80(
    local_transfer: torch.Tensor,
    local_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    initial_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    *,
    # Opt-in; see `CPDeltaRuleFixupPtrSm80`.
    _ptr_abi: bool = False,
    # One composition call's shared state, or None for a standalone
    # call. Never built here: see `_CPInvocation`.
    _ctx: "_CPInvocation | None" = None,
    _skip_check: bool = False,
    # When given, the arguments this entry would hand the compiled kernel
    # are appended to it and nothing is launched. That is how a prepared
    # execution is built: explicitly, at this layer, rather than by patching
    # the DSL's own classes from outside. The tuple is (compiled, args), and
    # the caller owns whatever it keeps alive.
    _plan_sink: list | None = None,
    _kernel_kind: str | None = None,
    _device=None,
    _stream=None,
):
    """Fix CP precompute chunk artifacts into global chunk-boundary states.

    Flat varlen workspaces use the native
    `(total_cp_chunks, num_heads, DimV, DimK)` layout produced by
    `cp_delta_rule_mn_precompute_dsl_sm80`.
    """
    import cuda.bindings.driver as cuda_driver

    device = local_transfer.device if _device is None else _device
    if not _skip_check:
        if local_transfer.ndim != 4:
            raise RuntimeError(
                "local_transfer must have shape (num_chunks, num_heads, DimV, DimK), "
                f"got {tuple(local_transfer.shape)}"
            )
        if local_state.shape != local_transfer.shape:
            raise RuntimeError(
                "local_state shape must match local_transfer shape, "
                f"got {tuple(local_state.shape)} and {tuple(local_transfer.shape)}"
            )
        if local_transfer.shape[-2:] != (128, 128):
            raise RuntimeError(
                f"CPDeltaRuleFixupSm80 only supports D=128, got {tuple(local_transfer.shape[-2:])}"
            )
        if local_transfer.dtype != torch.float32 or local_state.dtype != torch.float32:
            raise RuntimeError(
                "CPDeltaRuleFixupSm80 only supports float32 inputs, "
                f"got {local_transfer.dtype} and {local_state.dtype}"
            )
        use_state_indices = state_indices is not None
        if initial_state is not None:
            expected_tail = (local_transfer.shape[1], 128, 128)
            if (
                not use_state_indices
                and initial_state.shape != (cu_seqlens.shape[0] - 1, *expected_tail)
            ) or (
                use_state_indices and tuple(initial_state.shape[1:]) != expected_tail
            ):
                raise RuntimeError(
                    "initial_state must have shape "
                    f"{('[N_pool]' if use_state_indices else f'[{cu_seqlens.shape[0] - 1}]')} "
                    f"+ {expected_tail}, got {tuple(initial_state.shape)}"
                )
            state_dtype_to_cutlass(initial_state.dtype)
        if use_state_indices and (
            not is_integer_dtype(state_indices.dtype)
            or state_indices.shape != (cu_seqlens.shape[0] - 1,)
        ):
            raise RuntimeError(
                f"state_indices must have shape {(cu_seqlens.shape[0] - 1,)} and an integer dtype"
            )
        for name, tensor in (
            ("local_transfer", local_transfer),
            ("local_state", local_state),
            ("state_indices", state_indices),
        ):
            if tensor is None:
                continue
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
        if initial_state is not None:
            if not use_state_indices and not initial_state.is_contiguous():
                raise RuntimeError("initial_state must be contiguous")
    total_cp_chunks, num_heads, _, _ = local_transfer.shape
    if _ctx is not None and _ctx.total_cp_chunks != total_cp_chunks:
        # The composition sized the MN workspace from the same arithmetic, so a
        # disagreement here means the tensor handed in is not this call's.
        raise RuntimeError(
            f"local_transfer has {total_cp_chunks} chunks and this invocation "
            f"was built for {_ctx.total_cp_chunks}"
        )
    if not _skip_check:
        if cp_chunk_len <= 0:
            raise RuntimeError(f"cp_chunk_len must be positive, got {cp_chunk_len}")
        if not is_integer_dtype(cu_seqlens.dtype):
            raise RuntimeError(
                f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
            )
        expected_chunks = workspace_num_chunks_host(
            cu_seqlens, cp_chunk_len, total_seqlen
        )
        if expected_chunks != total_cp_chunks:
            raise RuntimeError(
                "local_transfer/local_state first dim must equal "
                f"chunk_bound(num_seqs, total_seqlen, cp_chunk_len)={expected_chunks}, got {total_cp_chunks}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1

    fixed_state = _get_cp_workspace(
        "gdn_cp_sm80_fixed_state", local_state.shape, local_state.dtype, device
    )
    if total_cp_chunks == 0:
        return fixed_state

    stream = (
        _ctx.stream
        if _ctx is not None
        else _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )
    d = local_transfer.shape[-1]
    local_transfer_tma = local_transfer.as_strided(
        (d, d, num_heads, total_cp_chunks),
        (d, 1, d * d, num_heads * d * d),
    )
    local_state_tma = local_state.as_strided(
        (d, d, num_heads, total_cp_chunks),
        (d, 1, d * d, num_heads * d * d),
    )

    needs_initial_state = initial_state is not None
    use_state_indices = state_indices is not None
    if _kernel_kind is None:
        if num_heads <= 8:
            _kernel_kind = "simt_row4"
        elif num_heads <= 16:
            _kernel_kind = "simt_row8"
        else:
            _kernel_kind = "simt_row8"
    spec = (
        needs_initial_state,
        initial_state.dtype if needs_initial_state else torch.float32,
        _kernel_kind,
        use_state_indices,
        cu_seqlens.dtype,
        state_indices.dtype if use_state_indices else None,
        (
            tuple(initial_state.stride()[1:])
            if use_state_indices and needs_initial_state
            else None
        ),
    )
    if _ptr_abi:
        mk = _ptr_factory(
            _ctx,
            device,
            _stream,
            (
                ("local_transfer", local_transfer, 128),
                ("local_state", local_state, 128),
                ("fixed_state", fixed_state, 128),
                ("cu_seqlens", cu_seqlens, 8),
                ("initial_state", initial_state, 16),
                (
                    "state_indices",
                    state_indices,
                    state_indices.element_size() if use_state_indices else 1,
                ),
            ),
        )
        kernel = _get_fixup_ptr_kernel(*spec)
        state_cutlass = state_dtype_to_cutlass(
            initial_state.dtype if needs_initial_state else torch.float32
        )
        # Inactive optional slots still get a valid address -- the workspace's
        # -- and the kernel only reads them under `const_expr`.
        placeholder = local_transfer
        args = (
            mk(cutlass.Float32, local_transfer, 128, "local_transfer"),
            mk(cutlass.Float32, local_state, 128, "local_state"),
            mk(
                state_cutlass,
                initial_state if needs_initial_state else placeholder,
                16,
                "initial_state",
            ),
            mk(
                integer_dtype_to_cutlass(
                    state_indices.dtype if use_state_indices else torch.int32
                ),
                state_indices if use_state_indices else placeholder,
                4,
                "state_indices",
            ),
            mk(cutlass.Float32, fixed_state, 128, "fixed_state"),
            mk(integer_dtype_to_cutlass(cu_seqlens.dtype), cu_seqlens, 8, "cu_seqlens"),
            cutlass.Int32(initial_state.shape[0] if needs_initial_state else 1),
            cutlass.Int32(initial_state.shape[1] if needs_initial_state else 1),
            cutlass.Int32(initial_state.stride(0) if needs_initial_state else 1),
            cutlass.Int32(initial_state.stride(1) if needs_initial_state else 1),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(total_cp_chunks),
            cutlass.Int32(num_seqs),
            cutlass.Int32(num_heads),
            stream,
        )
        opts = (
            _ctx.compile_options
            if _ctx is not None
            else _sm80_cp_compile_options(device)
        )
        compiled = get_cached_compile(kernel, opts)
        if compiled is None:
            compiled = cached_compile(kernel, *args, compile_options=opts)
        if _plan_sink is not None:
            _plan_sink.append((compiled, args))
        else:
            compiled(*args)
        return fixed_state
    kernel = _get_fixup_kernel(*spec)
    compiled = get_cached_compile(
        kernel,
        _ctx.compile_options if _ctx is not None else _sm80_cp_compile_options(device),
    )
    if compiled is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        initial_state_cute = (
            from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic()
            if needs_initial_state
            else None
        )
        state_indices_cute = (
            from_dlpack(state_indices, assumed_align=4).mark_layout_dynamic()
            if use_state_indices
            else None
        )
        kernel_args = (
            from_dlpack(local_transfer_tma, assumed_align=128).mark_layout_dynamic(
                leading_dim=1
            ),
            from_dlpack(local_state_tma, assumed_align=128).mark_layout_dynamic(
                leading_dim=1
            ),
            initial_state_cute,
            state_indices_cute,
            from_dlpack(
                fixed_state.reshape(-1), assumed_align=128
            ).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(total_cp_chunks),
            cutlass.Int32(num_seqs),
            cutlass.Int32(num_heads),
            stream,
        )
        compiled = cached_compile(
            kernel,
            *kernel_args,
            compile_options=(
                _ctx.compile_options
                if _ctx is not None
                else _sm80_cp_compile_options(device)
            ),
        )
    if _plan_sink is not None:
        _plan_sink.append(
            (
                compiled,
                (
                    local_transfer_tma,
                    local_state_tma,
                    initial_state if needs_initial_state else None,
                    state_indices if use_state_indices else None,
                    fixed_state.reshape(-1),
                    cu_seqlens,
                    cp_chunk_len,
                    total_cp_chunks,
                    num_seqs,
                    num_heads,
                    stream,
                ),
            )
        )
    else:
        compiled(
            local_transfer_tma,
            local_state_tma,
            initial_state if needs_initial_state else None,
            state_indices if use_state_indices else None,
            fixed_state.reshape(-1),
            cu_seqlens,
            cp_chunk_len,
            total_cp_chunks,
            num_seqs,
            num_heads,
            stream,
        )
    return fixed_state


# Stage 4, the chunk prefill. It subclasses the fused kernel the same way the
# sm90 CP prefill subclasses its own, and for the same reason: the recurrence
# inside a chunk is the fused recurrence, with the chunk's starting state coming
# from stage 3 rather than from the caller.
#
# Ported in pieces. `run_cp_state_math_role` carries no
# architecture-specific code at all and is taken as it stands.
# `run_aux_loop_body` and `run_aux_math_role` are not: they read `qk_stage`
# and `kk_stage`, which this class does not stage, and they call the wgmma
# helper, which is an sm90 instruction. Nothing here calls them -- the aux
# role's work is folded into `kernel` -- so they are not carried over rather
# than carried over broken. `load_t_tma`, `run_load_t_role` and `cp_qk_and_t_epi`
# reach for TMA and are rewritten against `cp.async`. `__call__` and `kernel`
# are built from the fused sm80 versions with the CP differences folded in --
# taking the sm90 ones would drag in TMA descriptors and a pipeline this target
# cannot build.
class CPDeltaRulePrefillSm80(_FullyFusedDeltaRuleSm80):
    def __init__(
        self,
        dtype: type[cutlass.Numeric] = cutlass.Float16,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        needs_initial_state: bool = False,
        store_final_state: bool = True,
        initial_state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        checkpoint_state_dtype: type[cutlass.Numeric] = cutlass.Float32,
        use_state_indices: bool = False,
        needs_checkpointing: bool = False,
        cu_seqlens_dtype: torch.dtype = torch.int64,
        state_indices_dtype: torch.dtype | None = None,
        checkpoint_cu_starts_dtype: torch.dtype | None = None,
        state_inner_strides: tuple[int, ...] | None = None,
        initial_state_inner_strides: tuple[int, ...] | None = None,
    ):
        super().__init__(
            True,
            False,
            True,
            needs_checkpointing,
            dtype,
            acc_dtype,
            # The fused kernel splits the value dimension across the block, and
            # every V-dependent GEMM carries that split as its M mode. It is not
            # optional there -- `make_acc_into_op` is only valid at one M repeat
            # per warp -- so the chunk prefill inherits it rather than choosing.
            blk_v=_FUSED_BLK_V,
            use_state_indices=use_state_indices,
            cu_seqlens_dtype=cu_seqlens_dtype,
            state_indices_dtype=state_indices_dtype,
            checkpoint_cu_starts_dtype=checkpoint_cu_starts_dtype,
            state_inner_strides=state_inner_strides,
            init_state_inner_strides=initial_state_inner_strides,
        )
        # T rides the beta stage's pipeline, so the consumer has to advance with
        # the producer. `needs_beta` is what gates that advance, and every other
        # place it gates is overridden here -- the epilogue, the store and the
        # load issuer all belong to this class.
        self.needs_beta = True
        self.needs_initial_state = needs_initial_state
        self.store_final_state = store_final_state
        self.initial_state_dtype = initial_state_dtype
        self.state_dtype = state_dtype
        self.checkpoint_state_dtype = checkpoint_state_dtype
        self.t_stage = 1
        self.manual_cache_key(
            "needs_alpha",
            "needs_beta",
            "needs_init_state",
            "needs_checkpointing",
            "needs_initial_state",
            "store_final_state",
            "initial_state_dtype",
            "state_dtype",
            "checkpoint_state_dtype",
            "use_state_indices",
            "cu_seqlens_dtype",
            "state_indices_dtype",
            "checkpoint_cu_starts_dtype",
            "state_inner_strides",
            "init_state_inner_strides",
            "dtype",
            "acc_dtype",
            "inverse_dtype",
            "BLK_Q",
            "BLK_KV",
            "D",
            "q_stage",
            "k_stage",
            "v_stage",
            "o_stage",
            # No qk_stage or kk_stage here: those buffers are staged on this
            # target only because sm90 hands them between roles, and sm80
            # collapsed the roles.
            "alpha_beta_stage",
            "t_stage",
            "D_v",
        )

    @cute.jit
    def __call__(
        self,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_v: cute.Tensor,
        # T sits between V and O, which is the order the entry builds its
        # arguments in. The fused kernel has beta further down and taking its
        # slot put T where O was expected.
        g_t: cute.Tensor,
        g_o: cute.Tensor,
        g_alpha: cute.Tensor,
        g_state: cute.Tensor,
        g_fixed_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        # Host ints: these are launch dimensions, and a dynamic value is not
        # a grid. The fused kernel takes its width the same way.
        grid_x: int,
        grid_y: int,
        stream,
    ):
        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_storage_layout = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_storage_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D_v, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_storage_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D_v, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )

        # One cp.async atom serves every load: the tile geometry lives in the
        # thread-value layout below rather than in a descriptor, so Q, K and V
        # differ only in how the loader partitions them.
        #
        # 128 bits a lane is the widest cp.async does, and it is what keeps a
        # tile down to one access per lane per row group. The value layout has
        # to run along the contiguous axis or the access is a gather and the
        # atom rejects it.
        # The atom itself is built inside the trace, not here: making it in
        # __init__ produces an IR value belonging to no kernel region, and the
        # tiled copy that consumes it is then rejected for using a value from
        # outside. Only the shape it implies is host state.
        elems_per_lane = 128 // self.dtype.width
        self.elems_per_lane = elems_per_lane
        # Q is (token, d) with d contiguous; K and V are (d, token) with d
        # contiguous. Both put d on the fast axis, so one tiled copy shape
        # covers all three: lanes split d, and successive thread rows walk the
        # other axis.
        # A lane's 16 B has to sit on whichever mode is contiguous, and the two
        # orientations disagree: Q is (token, d) so d is mode 1, while K and V
        # are (d, token) so d is mode 0. One layout pair each.
        lanes_along_d = self.D // elems_per_lane
        rows_at_a_time = _FUSED_LOAD_THREADS // lanes_along_d
        self.lanes_along_d = lanes_along_d
        self.load_rows_at_a_time = rows_at_a_time
        # Shapes only -- the layouts themselves are built inside the trace, for
        # the same reason the atom is.
        # Q: lanes split d within a row, thread rows walk tokens.
        self.q_load_tv_shape = ((rows_at_a_time, lanes_along_d), (lanes_along_d, 1))
        self.q_load_val_shape = (1, elems_per_lane)
        # K and V take the same split with the modes swapped, so the vector
        # still runs along d -- built per width by `load_tv_shape_dn`, because a
        # split value dimension makes V narrower than K.

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        kk_storage_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
        )
        alpha_storage_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        # T is staged where the fused kernel stages beta: the chunk prefill is
        # handed T rather than recomputing it from beta, which is the whole
        # point of stage 1 running ahead of it.
        qk_layout_atom_t = cute.make_layout((8, 8), stride=(8, 1))
        t_storage_layout = cute.tile_to_shape(
            qk_layout_atom_t, (self.BLK_KV, self.BLK_KV, self.t_stage), order=(0, 1, 2)
        )

        @cute.struct
        class SharedStorage:
            # The sm_90 path reserved two mbarriers a stage here, for the full
            # and empty phases. The sm_80 pipelines count async groups instead,
            # so that storage is gone and the stages have it back.
            smem_q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_storage_layout)],
                128,
            ]
            smem_k: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_storage_layout_sd)],
                128,
            ]
            smem_v: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(v_storage_layout_sd)],
                128,
            ]
            smem_qk: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(qk_storage_layout)],
                16,
            ]
            smem_kk: cute.struct.Align[
                cute.struct.MemRange[
                    self.inverse_dtype, cute.cosize(kk_storage_layout)
                ],
                16,
            ]
            smem_o: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(o_storage_layout)],
                128,
            ]
            smem_alpha: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(alpha_storage_layout)
                ],
                16,
            ]
            smem_t: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(t_storage_layout)],
                16,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            g_q,
            g_k,
            g_v,
            g_t,
            g_o,
            g_alpha,
            g_state,
            g_fixed_state,
            g_init_state,
            g_state_indices,
            g_state_checkpoints,
            checkpoint_cu_starts,
            cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            chunk_len,
            total_cp_chunks,
            num_seqs,
            total_checkpoints,
            checkpoint_every_n_tokens,
        ).launch(
            # One block per (value slice, head, chunk), one row per sequence.
            # The slice varies fastest. A chunk
            # is what this kernel runs the recurrence over, so it is what the
            # grid is cut by.
            grid=(grid_x, grid_y, 1),
            block=(_FUSED_LOAD_THREADS, 1, 1),
            max_number_threads=(_FUSED_LOAD_THREADS, 1, 1),
            stream=stream,
            # One. Asking for two costs 30% (1.751 -> 2.270 ms): the bound
            # forces ptxas to 128 registers a thread, half what this kernel
            # wants, and it reaches that by spilling. On the shapes where the
            # extra residency would matter the grid is also smaller than the
            # SM count -- num_seqs * num_sab_heads * the value splits -- so
            # there is no second block to co-reside with in the first place.
            min_blocks_per_mp=1,
        )

    @cute.jit
    def load_t_tile_into_stage(
        self,
        sT: cute.Tensor,
        gT_full: cute.Tensor,
        stage: cutlass.Int32,
        t_block: cutlass.Int32,
        head_idx: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """One square T tile into one stage, through the fused `_copy_tile`.

        T is indexed by block rather than by token and a tile is always whole,
        so there is no tail to predicate.
        """
        # A scalar coordinate copy, not `_copy_tile`. The T tile's shared layout
        # is the (8, 8) operand atom, which puts its rows 16 bytes apart, and
        # the 128-bit cp.async that `_copy_tile` issues wants 16-byte alignment
        # on the destination. This is the same copy stage 2 uses for the same
        # tile and for the same reason.
        mT = gT_full[None, None, head_idx, t_block]
        dst = sT[None, None, stage]
        for i in cutlass.range(
            self.BLK_KV * self.BLK_KV // _FUSED_LOAD_THREADS, unroll_full=True
        ):
            flat = i * _FUSED_LOAD_THREADS + tid
            # Destination subscripts swapped against the source. The shared
            # layout is the (8, 8) operand atom, and writing `dst[a, b] = T[a, b]`
            # through it reads back as `sT[b, a]` -- measured, exactly, against
            # stage 1's T. The consumer wants `sT[t, s]` to be `T[t, s]`, so the
            # transpose is undone here rather than at every read.
            dst[flat // self.BLK_KV, flat % self.BLK_KV] = mT[
                flat % self.BLK_KV, flat // self.BLK_KV
            ]

    @cute.jit
    def issue_block_loads(
        self,
        sQ_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sAlpha: cute.Tensor,
        sBeta: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        g_alpha: cute.Tensor,
        g_beta: cute.Tensor,
        q_pipeline,
        q_producer_state,
        k_pipeline,
        k_producer_state,
        v_pipeline,
        v_producer_state,
        alpha_pipeline,
        alpha_producer_state,
        beta_pipeline,
        beta_producer_state,
        blk: cutlass.Int32,
        t_block_start: cutlass.Int32,
        tok_start,
        tok_end: cutlass.Int32,
        scale: cutlass.Float32,
        q_head_idx: cutlass.Int32,
        k_head_idx: cutlass.Int32,
        v_head_idx: cutlass.Int32,
        v_row_offset: cutlass.Int32,
        sab_head_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        tid: cutlass.Int32,
        warp_idx: cutlass.Int32,
    ):
        """Fetch everything one block needs, with the whole block issuing it.

        The sm_90 kernel gives this to a warp group of its own and lets the
        math warps wait on an mbarrier, which this DSL will not emit for an
        sm_80 target -- and without one a thread can only wait on cp.async it
        issued itself. So the block does its own fetching and the math follows
        behind a barrier.

        Q, K and V go through cp.async and land when the consumer drains the
        group. Alpha and beta are scalar streams read with ordinary loads, so
        they are already in shared memory when this returns; the same barrier
        publishes them.
        """
        (
            q_producer_state,
            k_producer_state,
            v_producer_state,
        ) = self.load_qkv_cpasync(
            sQ_SD,
            sK_DS,
            sV_DS,
            gQ_full,
            gK_full,
            gV_full,
            q_pipeline,
            q_producer_state,
            k_pipeline,
            k_producer_state,
            v_pipeline,
            v_producer_state,
            blk,
            tok_start,
            tok_end,
            q_head_idx,
            k_head_idx,
            v_head_idx,
            v_row_offset,
            tid,
        )

        # Alpha's scan reads and writes the same channel, so exactly one warp
        # may run it; a second would race the first between its load and its
        # store. Beta only writes, but it is kept to one warp for the same
        # reason, and there is nothing to gain from the rest doing it too.
        #
        # Neither branch contains a barrier, so restricting them does not make
        # the block's barrier participation uneven.
        if cutlass.const_expr(self.needs_alpha):
            alpha_pipeline.producer_acquire(alpha_producer_state)
            if warp_idx == cutlass.Int32(0):
                blk_tok = tok_start + blk * cutlass.Int32(self.BLK_Q)
                self.load_alpha(
                    sAlpha,
                    g_alpha,
                    blk_tok,
                    tok_end,
                    sab_head_idx,
                    num_sab_heads,
                    alpha_producer_state.index,
                )
                AlphaProcessor().run(
                    sAlpha[None, None, alpha_producer_state.index], scale
                )
            alpha_pipeline.producer_commit(alpha_producer_state)
            alpha_producer_state.advance()

        # A T tile where the fused kernel loads beta. `needs_beta` is off on this
        # path -- stage 1 already folded beta into T -- so the beta stage's
        # pipeline and its producer state carry T instead, and nothing else in
        # the issuer changes.
        beta_pipeline.producer_acquire(beta_producer_state)
        self.load_t_tile_into_stage(
            sBeta,
            g_beta,
            beta_producer_state.index,
            t_block_start + blk,
            sab_head_idx,
            tid,
        )
        cute.arch.cp_async_commit_group()
        beta_pipeline.producer_commit(beta_producer_state)
        beta_producer_state.advance()

        return (
            q_producer_state,
            k_producer_state,
            v_producer_state,
            alpha_producer_state,
            beta_producer_state,
        )

    @cute.jit
    def kk_epi(
        self,
        tKKrKK: cute.Tensor,
        tKKcMkk: cute.Tensor,
        sAlpha: cute.Tensor,
        sT: cute.Tensor,
        alpha_stage: cutlass.Int32,
        t_stage: cutlass.Int32,
    ):
        """The KK accumulator is discarded and T is put in its place.

        The fused kernel builds T across this epilogue and the store that
        follows: the epilogue scales the KK accumulator by the decay, the store
        inverts it and scales by beta. Stage 1 produced that already, so both
        halves collapse -- the accumulator is overwritten here with
        `-gamma * T[t, s]` for `s >= t` and zero above the diagonal, which is
        what the sm90 path writes in `cp_qk_and_t_epi`, and the store that
        follows only stores.

        The GEMM that filled the accumulator still runs, and removing it is
        not the one-line change it looks like. Guarding its block in
        `compute_loop_body` with a compile-time flag took the two operand
        loads out -- PTX `ldmatrix` 636 to 516 across this kernel -- and left
        `mma.sync` at 960, so the GEMM still issued, now over fragments
        nothing had loaded. That attempt was reverted. The evidence, and the
        two-process proof that the variants are two binaries at all, is the
        bench harness's `cp_kk_variant_proof.py`; note also that
        `manual_cache_key` replaces the key rather than extending it, so a
        flag added to the base class is invisible here unless this class names
        it too.
        """
        alpha_log = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            s, t = tKKcMkk[i]
            value = cutlass.Float32(0.0)
            if s >= t:
                gamma = cute.math.exp2(
                    cutlass.Float32(alpha_log[s]) - cutlass.Float32(alpha_log[t]),
                    fastmath=True,
                )
                value = -gamma * cutlass.Float32(sT[t, s, t_stage])
            tKKrKK[i] = self.acc_dtype(value)

    @cute.jit
    def _kk_store_and_inv(
        self,
        tKKrKK: cute.Tensor,
        kk_tiled_mma,
        kk_thread_idx: cutlass.Int32,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sT: cute.Tensor,
        t_pipe_idx: cutlass.Int32,
        tKKcMkk: cute.Tensor,
    ):
        """Store the operand. There is nothing left to invert.

        `kk_epi` above already put T in the accumulator, so this writes it out
        as the MMA operand and stops -- no inverse, no reload, no beta scale.
        """
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyR2SOp(), self.dtype)
        tiled_store = cute.make_tiled_copy_C(r2s_atom, kk_tiled_mma)
        thr_store = tiled_store.get_slice(kk_thread_idx)
        tKKsKK = thr_store.partition_D(sKK_opd)
        tKKrT = cute.make_fragment_like(tKKrKK, self.dtype)
        for i in cutlass.range_constexpr(cute.size(tKKrKK)):
            tKKrT[i] = self.dtype(tKKrKK[i])
        cute.copy(tiled_store, thr_store.retile(tKKrT), tKKsKK)
        # The inverse this replaced ended on a barrier, and that barrier was
        # also what stood between writing the operand and the GEMM that reads
        # it. Dropping the inverse dropped it, and the GEMM read whatever was
        # there.
        cute.arch.barrier(
            barrier_id=FusedNamedBarrier.KK_SYNC,
            number_of_threads=_FUSED_WG_THREADS,
        )

    @cute.jit
    def cp_qk_and_t_epi(
        self,
        tQKrQK: cute.Tensor,
        tQKcMqk: cute.Tensor,
        sT: cute.Tensor,
        sQK: cute.Tensor,
        sKK_opd: cute.Tensor,
        sAlpha: cute.Tensor,
        alpha_stage: cutlass.Int32,
        is_final_block: bool,
        B: cutlass.Int32,
        scale: cutlass.Float32,
        qk_tiled_mma,
        kk_tiled_mma,
        aux_tidx: cutlass.Int32,
    ):
        alpha_log = sAlpha[None, AlphaProcessor.CUMSUM_LOG, alpha_stage]
        # `stmatrix` is sm90's. The fused kernel's `qk_store` already writes
        # this same accumulator to shared through `CopyR2SOp` and passes, so
        # that is the atom here too.
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyR2SOp(), self.dtype)
        qk_tiled_store = cute.make_tiled_copy_C(r2s_atom, qk_tiled_mma)
        kk_tiled_store = cute.make_tiled_copy_C(r2s_atom, kk_tiled_mma)
        qk_thr_store = qk_tiled_store.get_slice(aux_tidx)
        kk_thr_store = kk_tiled_store.get_slice(aux_tidx)
        tQKsQK = qk_thr_store.partition_D(sQK)
        tKKsKK = kk_thr_store.partition_D(sKK_opd)
        tQKcMqk_cv = kk_thr_store.retile(tQKcMqk)
        tQKrQK_cv = kk_thr_store.retile(tQKrQK)
        tQKrQK_cvt = cute.make_fragment_like(tQKrQK, self.dtype)
        tQKrQK_cvt_cv = kk_thr_store.retile(tQKrQK_cvt)
        tKKrT = cute.make_fragment_like(tKKsKK, self.dtype)

        for i in cutlass.range_constexpr(cute.size(tKKrT)):
            s, t = tQKcMqk_cv[i]
            gamma = cutlass.Float32(0.0)
            qk_value = cutlass.Float32(0.0)
            t_value = cutlass.Float32(0.0)
            pred = s >= t
            if cutlass.const_expr(is_final_block):
                pred = pred and s < B and t < B
            if pred:
                gamma = cute.math.exp2(
                    cutlass.Float32(alpha_log[s]) - cutlass.Float32(alpha_log[t]),
                    fastmath=True,
                )
            qk_value = tQKrQK_cv[i] * gamma * scale
            t_value = -gamma * cutlass.Float32(sT[t, s])
            if cutlass.const_expr(is_final_block):
                qk_value = qk_value if pred else cutlass.Float32(0.0)
                t_value = t_value if pred else cutlass.Float32(0.0)

            tQKrQK_cvt_cv[i] = self.dtype(qk_value)
            tKKrT[i] = self.dtype(t_value)
        cute.copy(qk_tiled_store, qk_thr_store.retile(tQKrQK_cvt), tQKsQK)
        cute.copy(kk_tiled_store, tKKrT, tKKsKK)

    @cute.jit
    def load_t_cpasync(
        self,
        sT: cute.Tensor,
        gT_full: cute.Tensor,
        t_pipeline,
        t_producer_state,
        blk: cutlass.Int32,
        t_block_start: cutlass.Int32,
        head_idx: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        """One T tile into a stage, in place of the sm90 TMA load.

        The fused kernel's own `_copy_tile` does the work, so this stays the
        same shape as the K, Q and V loads beside it: acquire, copy, commit the
        cp.async group, release. A T tile is square and always whole -- it is
        indexed by block, not by token -- so `rows_live` is the full block and
        there is no tail to predicate.
        """
        sT_stage = sT[None, None, t_producer_state.index]
        mT = gT_full[None, None, head_idx, t_block_start + blk]
        gT = cute.zipped_divide(mT, (self.BLK_KV, self.BLK_KV))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        t_pipeline.producer_acquire(t_producer_state)
        self._copy_tile(gT, sT_stage, self.BLK_KV, tid, self.BLK_KV, False)
        cute.arch.cp_async_commit_group()
        t_pipeline.producer_commit(t_producer_state)
        t_producer_state.advance()
        return t_producer_state

    @cute.jit
    def run_load_t_role(
        self,
        sT: cute.Tensor,
        gT_full: cute.Tensor,
        t_pipeline,
        num_blocks: cutlass.Int32,
        t_block_start: cutlass.Int32,
        head_idx: cutlass.Int32,
        tid: cutlass.Int32,
    ):
        t_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.t_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            t_producer_state = self.load_t_cpasync(
                sT,
                gT_full,
                t_pipeline,
                t_producer_state,
                blk,
                t_block_start,
                head_idx,
                tid,
            )

    @cute.jit
    def run_cp_state_math_role(
        self,
        sQ_SD: cute.Tensor,
        sK_SD: cute.Tensor,
        sK_DS: cute.Tensor,
        sV_DS: cute.Tensor,
        sQK: cute.Tensor,
        sKK_inv: cute.Tensor,
        sKK_opd: cute.Tensor,
        sO: cute.Tensor,
        sAlpha: cute.Tensor,
        sT: cute.Tensor,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        gO_full: cute.Tensor,
        g_alpha: cute.Tensor,
        g_t: cute.Tensor,
        q_pipeline,
        k_pipeline,
        v_pipeline,
        alpha_pipeline,
        t_pipeline,
        g_state: cute.Tensor,
        g_fixed_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        work_desc: WorkDesc,
        public_seq_idx: cutlass.Int32,
        fixed_state_idx: cutlass.Int32,
        load_fixed_state,
        store_state,
        t_block_start: cutlass.Int32,
        seq_block_offset: cutlass.Int32,
        full_seq_len: cutlass.Int32,
        scale: cutlass.Float32,
        wg_idx: cutlass.Int32,
        math_tidx: cutlass.Int32,
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tok_end: cutlass.Int32,
        num_blocks: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
    ):
        self._math_order_init(wg_idx)
        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.q_stage
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.k_stage
        )
        v_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.v_stage
        )
        q_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.q_stage
        )
        k_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.k_stage
        )
        v_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.v_stage
        )
        alpha_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        t_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.alpha_beta_stage
        )
        alpha_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )
        beta_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.alpha_beta_stage
        )

        kv_tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((_FUSED_MMA_WARPS, 1, 1)),
            permutation_mnk=(self.D_v, self.D, self.BLK_KV),
        )
        kv_thr_mma = kv_tiled_mma.get_slice(math_tidx)
        tKVrKV = cute.make_rmem_tensor(
            kv_thr_mma.partition_shape_C((self.D_v, self.D)), self.acc_dtype
        )
        tKVrKV.fill(self.acc_dtype(0.0))

        packed_state_layout = cute.make_ordered_layout(
            (self.D, self.D, num_sab_heads, num_seqs), order=(0, 1, 2, 3)
        )
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        # (k, v): every K row of the band of V rows this block owns. Blocks of
        # one sequence and head write disjoint bands, so the pool needs no
        # coordination between them.
        v_row_offset = work_desc.v_row_offset(self.D_v)
        # Addressed only when there is something to address. With no final state
        # asked for, `g_state` is absent, and reading its shape and stride to
        # build an address the guarded store never uses is an illegal access on
        # its own.
        state_idx = public_seq_idx
        gStateKV = None
        if cutlass.const_expr(self.store_final_state):
            state_ref_shape = (g_state.shape[0], g_state.shape[1], self.D, self.D)
            state_ref_layout = cute.make_layout(state_ref_shape, stride=g_state.stride)
            indexed_state_layout = cute.select(state_ref_layout, mode=[3, 2, 1, 0])
            if cutlass.const_expr(self.use_state_indices):
                state_idx = cutlass.Int32(g_state_indices[public_seq_idx])
                state_layout = indexed_state_layout
            else:
                state_layout = packed_state_layout
            mState = cute.make_tensor(g_state.iterator, state_layout)
            gStateKV = cute.domain_offset(
                (cutlass.Int32(0), v_row_offset),
                mState[None, None, o_head_idx, state_idx],
            )
        elif cutlass.const_expr(self.use_state_indices):
            state_idx = cutlass.Int32(g_state_indices[public_seq_idx])
        # Where a chunk starts. Its sequence's first chunk starts from what the
        # caller passed; every other chunk starts from the state stage 3 fixed
        # for the chunk before it. `fixed_state_idx` is already `cp_chunk_idx - 1`
        # for those, so both reads are the same shape and differ only in tensor
        # and row.
        # `needs_init_state` is always on here -- a chunk always starts from
        # some state -- but the caller's own initial state is optional, and
        # `needs_initial_state` is what says whether one was passed. Reading
        # `g_init_state.stride` when it was not is how that difference showed
        # up.
        if cutlass.const_expr(self.needs_init_state and self.needs_initial_state):
            # Built from `g_init_state`'s own shape and stride. It borrowed the
            # output state's, which is wrong twice: `state_ref_shape` only
            # exists when a final state was asked for, so
            # `initial_state` with `output_final_state=False` did not compile at
            # all, and the two pools need not have the same leading dimension
            # when they are separate tensors.
            init_state_ref_shape = (
                g_init_state.shape[0],
                g_init_state.shape[1],
                self.D,
                self.D,
            )
            init_state_ref_layout = cute.make_layout(
                init_state_ref_shape, stride=g_init_state.stride
            )
            indexed_init_state_layout = cute.select(
                init_state_ref_layout, mode=[3, 2, 1, 0]
            )
            if cutlass.const_expr(self.use_state_indices):
                init_state_layout = indexed_init_state_layout
            else:
                init_state_layout = packed_state_layout
            mInitState = cute.make_tensor(g_init_state.iterator, init_state_layout)
            mFixedState = cute.make_tensor(g_fixed_state.iterator, packed_state_layout)
            # The load happens inside each branch rather than after them. The
            # two tensors do not share a layout when `use_state_indices` is on,
            # so a view assigned in one branch and used after both changes
            # structure across the `if` and the DSL rejects it.
            if load_fixed_state:
                self.kv_load(
                    tKVrKV,
                    cute.domain_offset(
                        (cutlass.Int32(0), v_row_offset),
                        mFixedState[None, None, o_head_idx, fixed_state_idx],
                    ),
                    kv_thr_mma,
                )
            else:
                self.kv_load(
                    tKVrKV,
                    cute.domain_offset(
                        (cutlass.Int32(0), v_row_offset),
                        mInitState[None, None, o_head_idx, state_idx],
                    ),
                    kv_thr_mma,
                )
        elif load_fixed_state:
            mFixedState = cute.make_tensor(g_fixed_state.iterator, packed_state_layout)
            self.kv_load(
                tKVrKV,
                cute.domain_offset(
                    (cutlass.Int32(0), v_row_offset),
                    mFixedState[None, None, o_head_idx, fixed_state_idx],
                ),
                kv_thr_mma,
            )

        first_B = work_desc.seq_len
        if first_B > cutlass.Int32(self.BLK_KV):
            first_B = cutlass.Int32(self.BLK_KV)
        if cutlass.const_expr(self.needs_init_state):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                t_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sT,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_t,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                t_pipeline,
                t_producer_state,
                cutlass.Int32(0),
                t_block_start,
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.v_row_offset(self.D_v),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sT,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                t_pipeline,
                beta_consumer_state,
                False,
                True,
                first_B,
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                cutlass.Int32(0),
                num_q_heads,
                num_v_heads,
                tidx,
            )
        else:
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                t_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sT,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_t,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                t_pipeline,
                t_producer_state,
                cutlass.Int32(0),
                t_block_start,
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.v_row_offset(self.D_v),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sT,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                t_pipeline,
                beta_consumer_state,
                True,
                True,
                first_B,
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                cutlass.Int32(0),
                num_q_heads,
                num_v_heads,
                tidx,
            )
        self.maybe_store_checkpoint(
            tKVrKV,
            g_state_checkpoints,
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
            kv_thr_mma,
            public_seq_idx,
            o_head_idx,
            v_row_offset,
            num_sab_heads,
            total_checkpoints,
            seq_block_offset + cutlass.Int32(self.BLK_KV),
            full_seq_len,
        )

        for blk in cutlass.range(
            cutlass.Int32(1), num_blocks - cutlass.Int32(1), cutlass.Int32(1), unroll=1
        ):
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                t_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sT,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_t,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                t_pipeline,
                t_producer_state,
                blk,
                t_block_start,
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.v_row_offset(self.D_v),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sT,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                t_pipeline,
                beta_consumer_state,
                False,
                False,
                cutlass.Int32(self.BLK_KV),
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                blk,
                num_q_heads,
                num_v_heads,
                tidx,
            )
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_thr_mma,
                public_seq_idx,
                o_head_idx,
                v_row_offset,
                num_sab_heads,
                total_checkpoints,
                seq_block_offset
                + (blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                full_seq_len,
            )

        if num_blocks != cutlass.Int32(1):
            last_blk = num_blocks - cutlass.Int32(1)
            last_B = work_desc.seq_len - last_blk * cutlass.Int32(self.BLK_KV)
            (
                q_producer_state,
                k_producer_state,
                v_producer_state,
                alpha_producer_state,
                t_producer_state,
            ) = self.issue_block_loads(
                sQ_SD,
                sK_DS,
                sV_DS,
                sAlpha,
                sT,
                gQ_full,
                gK_full,
                gV_full,
                g_alpha,
                g_t,
                q_pipeline,
                q_producer_state,
                k_pipeline,
                k_producer_state,
                v_pipeline,
                v_producer_state,
                alpha_pipeline,
                alpha_producer_state,
                t_pipeline,
                t_producer_state,
                last_blk,
                t_block_start,
                work_desc.tok_offset,
                tok_end,
                scale,
                work_desc.q_head_idx(),
                work_desc.k_head_idx(num_q_heads, num_v_heads),
                work_desc.v_head_idx(),
                work_desc.v_row_offset(self.D_v),
                work_desc.o_head_idx(num_q_heads, num_v_heads),
                num_sab_heads,
                tidx,
                warp_idx,
            )
            (
                q_consumer_state,
                k_consumer_state,
                v_consumer_state,
                alpha_consumer_state,
                beta_consumer_state,
            ) = self.compute_loop_body(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sT,
                kv_tiled_mma,
                q_pipeline,
                q_consumer_state,
                k_pipeline,
                k_consumer_state,
                v_pipeline,
                v_consumer_state,
                alpha_pipeline,
                alpha_consumer_state,
                t_pipeline,
                beta_consumer_state,
                False,
                True,
                last_B,
                tKVrKV,
                scale,
                wg_idx,
            )
            self.o_writer.run(
                sO[None, None, 0],
                gO_full,
                work_desc,
                last_blk,
                num_q_heads,
                num_v_heads,
                tidx,
            )
            self.maybe_store_checkpoint(
                tKVrKV,
                g_state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
                kv_thr_mma,
                public_seq_idx,
                o_head_idx,
                v_row_offset,
                num_sab_heads,
                total_checkpoints,
                seq_block_offset
                + (last_blk + cutlass.Int32(1)) * cutlass.Int32(self.BLK_KV),
                full_seq_len,
            )
        # Only a sequence's last chunk writes that sequence's state, and only
        # when a final state was asked for. Several chunks writing it would be
        # blocks racing for one row; writing it when it was not asked for is
        # worse, because `gStateKV` then points into the fixup workspace and the
        # write lands on a chunk state another block still has to read.
        if cutlass.const_expr(self.store_final_state):
            if store_state:
                self.kv_store(tKVrKV, gStateKV, kv_thr_mma)

    @cute.kernel
    def kernel(
        # Declared in the order `__call__` takes its own arguments, so the
        # forwarding below is a straight list. They were not, and every
        # scalar after `cu_seqlens` sat two places off: `scale` received
        # `chunk_len` and `chunk_len` received `num_v_heads`.
        self,
        gQ_full: cute.Tensor,
        gK_full: cute.Tensor,
        gV_full: cute.Tensor,
        g_t: cute.Tensor,
        gO_full: cute.Tensor,
        g_alpha: cute.Tensor,
        g_state: cute.Tensor,
        g_fixed_state: cute.Tensor,
        g_init_state: cute.Tensor,
        g_state_indices: cute.Tensor,
        g_state_checkpoints: cute.Tensor,
        checkpoint_cu_starts: cute.Tensor,
        cu_seqlens: cute.Tensor,
        scale: cutlass.Float32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # The sm_90 path prefetched the four TMA descriptors here. cp.async has
        # no descriptor to warm, so the block goes straight to work.

        # A block owns one (value slice, head, chunk) and the y axis is the
        # sequence, so
        # the work comes off block_idx rather than out of a scheduler. A chunk
        # past the end of its sequence has nothing to do and falls out below.
        bx, seq_idx, _ = cute.arch.block_idx()
        # The value slice varies fastest, the way the fused scheduler orders it:
        # blocks sharing a sequence and a head sit next to each other and read
        # the same Q, K and alpha. Without this axis the grid ran slice 0 only --
        # every block took `v_slice_idx = 0`, so the other bands were never
        # computed at all. Fixing it did not change the 512 the output comes
        # back with, so this is a defect on its own and not that one.
        v_slice_idx = cutlass.Int32(0)
        if cutlass.const_expr(self.num_v_splits > 1):
            v_slice_idx = bx % cutlass.Int32(self.num_v_splits)
            bx = bx // cutlass.Int32(self.num_v_splits)
        o_head_idx = bx % num_sab_heads
        q_head_idx = o_head_idx * num_q_heads // num_sab_heads
        v_head_idx = o_head_idx * num_v_heads // num_sab_heads
        chunk_idx_in_seq = bx // num_sab_heads
        seq_start = cutlass.Int32(cu_seqlens[seq_idx])
        seq_len = cutlass.Int32(cu_seqlens[seq_idx + 1]) - seq_start
        num_cp_chunks = chunks_for_len(seq_len, chunk_len)
        is_valid_chunk = chunk_idx_in_seq < num_cp_chunks

        tok_offset = seq_start + chunk_idx_in_seq * chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx_in_seq, chunk_len)
        valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx_in_seq, chunk_len)
        t_blocks_per_cp_chunk = (
            chunk_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)
        t_block_start = varlen_chunk_idx(
            seq_idx,
            seq_start,
            chunk_idx_in_seq * t_blocks_per_cp_chunk,
            self.BLK_KV,
        )
        work_desc = WorkDesc(
            seq_idx=cp_chunk_idx,
            private_q_head_idx=q_head_idx,
            private_v_head_idx=v_head_idx,
            tok_offset=tok_offset,
            seq_len=valid_chunk_len,
            tile_idx=cutlass.Int32(0),
            v_slice_idx=v_slice_idx,
        )
        tok_end = tok_offset + valid_chunk_len
        num_blocks = (
            valid_chunk_len + cutlass.Int32(self.BLK_KV) - cutlass.Int32(1)
        ) // cutlass.Int32(self.BLK_KV)

        # `fixed_state_idx` is the chunk before this one, which is the state
        # this one starts from. A sequence's first chunk starts from the
        # caller's state instead, and only its last chunk writes its row.
        load_fixed_state = chunk_idx_in_seq != cutlass.Int32(0)
        fixed_state_idx = cp_chunk_idx
        if load_fixed_state:
            fixed_state_idx = cp_chunk_idx - cutlass.Int32(1)
        store_state = chunk_idx_in_seq == num_cp_chunks - cutlass.Int32(1)

        # T arrives flat, the way stage 1 wrote it. The same ordered layout the
        # MN precompute reads it through gives it back its (BLK, BLK, head,
        # block) shape -- writer and reader have to agree on this, and nothing
        # in a shape check would catch it if they did not.
        total_t_blocks = total_cp_chunks * t_blocks_per_cp_chunk
        mT_full = cute.make_tensor(
            g_t.iterator,
            cute.make_ordered_layout(
                (self.BLK_KV, self.BLK_KV, num_sab_heads, total_t_blocks),
                order=(0, 1, 2, 3),
            ),
        )

        math_tidx = tidx
        wg_idx = tidx // cutlass.Int32(_FUSED_WG_THREADS)

        # ── Smem allocation ───────────────────────────────────────────────────
        allocator = cutlass.utils.SmemAllocator()
        storage = allocator.allocate(self.shared_storage)

        qkv_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128,
            self.dtype,
        )
        q_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_Q, self.D, self.q_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        sQ_SD = storage.smem_q.get_tensor(q_layout_sd.outer, swizzle=q_layout_sd.inner)

        k_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D, self.k_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        k_layout_ds = cute.select(k_layout_sd, [1, 0, 2])
        sK_SD = storage.smem_k.get_tensor(k_layout_sd.outer, swizzle=k_layout_sd.inner)
        sK_DS = storage.smem_k.get_tensor(k_layout_ds.outer, swizzle=k_layout_ds.inner)

        v_layout_sd = cute.coalesce(
            cute.tile_to_shape(
                qkv_smem_layout_atom,
                (self.BLK_KV, self.D_v, self.v_stage),
                order=(0, 1, 2),
            ),
            target_profile=(1, 1, 1),
        )
        v_layout_ds = cute.select(v_layout_sd, [1, 0, 2])
        sV_DS = storage.smem_v.get_tensor(v_layout_ds.outer, swizzle=v_layout_ds.inner)

        qk_layout_atom = cute.make_layout((8, 8), stride=(8, 1))
        qk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_Q, self.BLK_KV), order=(1, 0)
        )
        sQK = storage.smem_qk.get_tensor(qk_layout)

        kk_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV), order=(1, 0)
        )
        sKK_inv = storage.smem_kk.get_tensor(kk_layout)
        kk_opd_ptr = cute.recast_ptr(storage.smem_kk.data_ptr(), dtype=self.dtype)
        sKK_opd = cute.make_tensor(kk_opd_ptr, kk_layout)

        o_smem_layout_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.MN_SW32,
            self.dtype,
        )
        o_layout = cute.tile_to_shape(
            o_smem_layout_atom,
            (self.D_v, self.BLK_Q, self.o_stage),
            order=(1, 0, 2),
        )
        sO = storage.smem_o.get_tensor(o_layout.outer, swizzle=o_layout.inner)
        alpha_layout = cute.make_layout(
            (self.BLK_Q, AlphaProcessor.NUM_CHANNELS, self.alpha_beta_stage)
        )
        sAlpha = storage.smem_alpha.get_tensor(alpha_layout)

        t_layout = cute.tile_to_shape(
            qk_layout_atom, (self.BLK_KV, self.BLK_KV, self.t_stage), order=(0, 1, 2)
        )
        sT = storage.smem_t.get_tensor(t_layout)

        q_pipeline = PipelineCpAsyncSm80(self.q_stage)
        k_pipeline = PipelineCpAsyncSm80(self.k_stage)
        v_pipeline = PipelineCpAsyncSm80(self.v_stage)
        alpha_pipeline = PipelineCpAsyncSm80(self.alpha_beta_stage)
        t_pipeline = PipelineCpAsyncSm80(self.t_stage)
        # No mbarrier storage to fence: the stage pipelines are group
        # counters. One barrier still has to line the block up before the
        # first load.
        cute.arch.sync_threads()

        if is_valid_chunk and valid_chunk_len != cutlass.Int32(0):
            self.run_cp_state_math_role(
                sQ_SD,
                sK_SD,
                sK_DS,
                sV_DS,
                sQK,
                sKK_inv,
                sKK_opd,
                sO,
                sAlpha,
                sT,
                gQ_full,
                gK_full,
                gV_full,
                gO_full,
                g_alpha,
                mT_full,
                q_pipeline,
                k_pipeline,
                v_pipeline,
                alpha_pipeline,
                t_pipeline,
                g_state,
                g_fixed_state,
                g_init_state,
                g_state_indices,
                g_state_checkpoints,
                checkpoint_cu_starts,
                work_desc,
                seq_idx,
                fixed_state_idx,
                load_fixed_state,
                store_state,
                t_block_start,
                chunk_idx_in_seq * chunk_len,
                seq_len,
                scale,
                wg_idx,
                math_tidx,
                tidx,
                warp_idx,
                tok_end,
                num_blocks,
                num_q_heads,
                num_v_heads,
                num_sab_heads,
                num_seqs,
                total_checkpoints,
                checkpoint_every_n_tokens,
            )


class CPDeltaRulePrefillPtrSm80(CPDeltaRulePrefillSm80):
    """The chunk prefill, entered through pointers instead of tensors.

    The last of the four, and the one with the most arguments: twelve tensors,
    which is why its `generate_execution_args` was the most expensive of the
    four. Every specialization axis of `_get_prefill_kernel` stays on the
    instance, so no case here is a runtime branch on a nullable pointer, and
    the inactive optional slots receive a valid aligned address that the
    kernel body only reads under `cutlass.const_expr`.

    What has to arrive is decided by what the kernel body reads:

      q, k, v, o and t are used through their layouts, so the extents and the
      strides of the `as_strided` views the tensor entry builds are rebuilt
      here, with the same leading dimensions;

      `state` and `initial_state` are the two whose *pool* shape and stride
      the body reads -- separately, because they need not be the same tensor
      and need not share a leading dimension. They are passed as two
      independent sets of scalars;

      alpha, fixed_state, checkpoints, checkpoint starts and cu_seqlens are
      flat or read for their iterator, so a pointer and an extent suffice.

    `store_final_state=False` keeps the tensor entry's contract that
    `state_arg` aliases `fixed_state`: the same pointer is passed, and the body
    never addresses it, because `gStateKV` is only built under
    `const_expr(self.store_final_state)`.
    """

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        k_ptr: cute.Pointer,
        v_ptr: cute.Pointer,
        t_ptr: cute.Pointer,
        o_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        state_ptr: cute.Pointer,
        fixed_state_ptr: cute.Pointer,
        init_state_ptr: cute.Pointer,
        state_indices_ptr: cute.Pointer,
        checkpoints_ptr: cute.Pointer,
        checkpoint_starts_ptr: cute.Pointer,
        cu_seqlens_ptr: cute.Pointer,
        scale: cutlass.Float32,
        total_seqlen: cutlass.Int32,
        total_t_blocks: cutlass.Int32,
        state_rows: cutlass.Int32,
        state_heads: cutlass.Int32,
        state_stride_0: cutlass.Int32,
        state_stride_1: cutlass.Int32,
        init_rows: cutlass.Int32,
        init_heads: cutlass.Int32,
        init_stride_0: cutlass.Int32,
        init_stride_1: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_k_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        total_checkpoints: cutlass.Int32,
        checkpoint_every_n_tokens: cutlass.Int32,
        chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        grid_x: cutlass.Int32,
        grid_y: cutlass.Int32,
        stream,
    ):
        D = cutlass.const_expr(self.D)
        one = cute.make_layout(1)
        # The five `as_strided` views the tensor entry builds, with the same
        # leading dimension in each: q is (token, d, head) with d contiguous
        # on mode 1, the rest put d on mode 0.
        g_q = cute.make_tensor(
            q_ptr,
            cute.make_layout(
                (total_seqlen, D, num_q_heads),
                stride=(num_q_heads * D, 1, D),
            ),
        )
        g_k = cute.make_tensor(
            k_ptr,
            cute.make_layout(
                (D, total_seqlen, num_k_heads),
                stride=(1, num_k_heads * D, D),
            ),
        )
        g_v = cute.make_tensor(
            v_ptr,
            cute.make_layout(
                (D, total_seqlen, num_v_heads),
                stride=(1, num_v_heads * D, D),
            ),
        )
        g_o = cute.make_tensor(
            o_ptr,
            cute.make_layout(
                (D, total_seqlen, num_sab_heads),
                stride=(1, num_sab_heads * D, D),
            ),
        )
        g_t = cute.make_tensor(
            t_ptr,
            cute.make_layout(
                (64, 64, num_sab_heads, total_t_blocks),
                stride=(64, 1, 64 * 64, num_sab_heads * 64 * 64),
            ),
        )
        g_alpha = cute.make_tensor(
            alpha_ptr, cute.make_layout(total_seqlen * num_sab_heads)
        )
        g_fixed_state = cute.make_tensor(
            fixed_state_ptr,
            cute.make_layout(total_cp_chunks * num_sab_heads * D * D),
        )
        g_cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout(num_seqs + 1))
        # The two pools, each from its own scalars. The body reads shape[0],
        # shape[1] and stride off whichever it is allowed to address.
        if cutlass.const_expr(self.store_final_state):
            inner = cutlass.const_expr(self.state_inner_strides)
            strides = (
                (state_stride_0, inner[0], inner[1], inner[2])
                if cutlass.const_expr(inner is not None)
                else (state_stride_0, state_stride_1, D, 1)
            )
            g_state = cute.make_tensor(
                state_ptr,
                cute.make_layout((state_rows, state_heads, D, D), stride=strides),
            )
        else:
            # Aliases `fixed_state`, as on the tensor path, and is never
            # addressed: `gStateKV` is built only under `store_final_state`.
            g_state = cute.make_tensor(state_ptr, one)
        if cutlass.const_expr(self.needs_init_state and self.needs_initial_state):
            inner = cutlass.const_expr(self.init_state_inner_strides)
            strides = (
                (init_stride_0, inner[0], inner[1], inner[2])
                if cutlass.const_expr(inner is not None)
                else (init_stride_0, init_stride_1, D, 1)
            )
            g_init_state = cute.make_tensor(
                init_state_ptr,
                cute.make_layout((init_rows, init_heads, D, D), stride=strides),
            )
        else:
            g_init_state = cute.make_tensor(init_state_ptr, one)
        if cutlass.const_expr(self.use_state_indices):
            g_state_indices = cute.make_tensor(
                state_indices_ptr, cute.make_layout(num_seqs)
            )
        else:
            g_state_indices = cute.make_tensor(state_indices_ptr, one)
        if cutlass.const_expr(self.needs_checkpointing):
            g_state_checkpoints = cute.make_tensor(
                checkpoints_ptr,
                cute.make_layout(total_checkpoints * num_sab_heads * D * D),
            )
            g_checkpoint_cu_starts = cute.make_tensor(
                checkpoint_starts_ptr, cute.make_layout(num_seqs + 1)
            )
        else:
            g_state_checkpoints = cute.make_tensor(checkpoints_ptr, one)
            g_checkpoint_cu_starts = cute.make_tensor(checkpoint_starts_ptr, one)
        super().__call__(
            g_q,
            g_k,
            g_v,
            g_t,
            g_o,
            g_alpha,
            g_state,
            g_fixed_state,
            g_init_state,
            g_state_indices,
            g_state_checkpoints,
            g_checkpoint_cu_starts,
            g_cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
            total_checkpoints,
            checkpoint_every_n_tokens,
            chunk_len,
            total_cp_chunks,
            grid_x,
            grid_y,
            stream,
        )


@functools.cache
def _get_prefill_ptr_kernel(
    kernel_dtype,
    needs_initial_state,
    store_final_state,
    initial_state_dtype,
    state_dtype,
    checkpoint_state_dtype,
    use_state_indices,
    needs_checkpointing,
    cu_seqlens_dtype,
    state_indices_dtype,
    checkpoint_cu_starts_dtype,
    state_inner_strides,
    initial_state_inner_strides,
):
    """The pointer twin of `_get_prefill_kernel`, same thirteen axes."""
    return CPDeltaRulePrefillPtrSm80(
        kernel_dtype,
        needs_initial_state=needs_initial_state,
        store_final_state=store_final_state,
        initial_state_dtype=state_dtype_to_cutlass(initial_state_dtype),
        state_dtype=state_dtype_to_cutlass(state_dtype),
        checkpoint_state_dtype=state_dtype_to_cutlass(checkpoint_state_dtype),
        use_state_indices=use_state_indices,
        needs_checkpointing=needs_checkpointing,
        cu_seqlens_dtype=cu_seqlens_dtype,
        state_indices_dtype=state_indices_dtype,
        checkpoint_cu_starts_dtype=checkpoint_cu_starts_dtype,
        state_inner_strides=state_inner_strides,
        initial_state_inner_strides=initial_state_inner_strides,
    )


@functools.cache
def _get_prefill_kernel(
    kernel_dtype,
    needs_initial_state,
    store_final_state,
    initial_state_dtype,
    state_dtype,
    checkpoint_state_dtype,
    use_state_indices,
    needs_checkpointing,
    cu_seqlens_dtype,
    state_indices_dtype,
    checkpoint_cu_starts_dtype,
    state_inner_strides,
    initial_state_inner_strides,
):
    return CPDeltaRulePrefillSm80(
        kernel_dtype,
        needs_initial_state=needs_initial_state,
        store_final_state=store_final_state,
        initial_state_dtype=state_dtype_to_cutlass(initial_state_dtype),
        state_dtype=state_dtype_to_cutlass(state_dtype),
        checkpoint_state_dtype=state_dtype_to_cutlass(checkpoint_state_dtype),
        use_state_indices=use_state_indices,
        needs_checkpointing=needs_checkpointing,
        cu_seqlens_dtype=cu_seqlens_dtype,
        state_indices_dtype=state_indices_dtype,
        checkpoint_cu_starts_dtype=checkpoint_cu_starts_dtype,
        state_inner_strides=state_inner_strides,
        initial_state_inner_strides=initial_state_inner_strides,
    )


def cp_delta_rule_prefill_dsl_sm80(
    o: torch.Tensor,
    state: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    fixed_state: torch.Tensor,
    alpha: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    max_seqlen: int | None = None,
    initial_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    *,
    # Opt-in; see `CPDeltaRulePrefillPtrSm80`.
    _ptr_abi: bool = False,
    # One composition call's shared state, or None for a standalone
    # call. Never built here: see `_CPInvocation`.
    _ctx: "_CPInvocation | None" = None,
    _skip_check: bool = False,
    # When given, the arguments this entry would hand the compiled kernel
    # are appended to it and nothing is launched. That is how a prepared
    # execution is built: explicitly, at this layer, rather than by patching
    # the DSL's own classes from outside. The tuple is (compiled, args), and
    # the caller owns whatever it keeps alive.
    _plan_sink: list | None = None,
    _device=None,
    _stream=None,
):
    """Run CP main prefill with precomputed T and fixed-up chunk states.

    Flat varlen input consumes flat Q/K/V/O/alpha tensors and varlen T/fixed-state
    workspaces. `state` is the public per-sequence final state in native
    `(DimV, DimK)` layout.
    """
    import cuda.bindings.driver as cuda_driver

    device = q.device if _device is None else _device
    needs_checkpointing = checkpoint_every_n_tokens > 0
    if not _skip_check:
        if q.ndim != 3:
            raise RuntimeError(
                f"q must have shape (total_seqlen, num_q_heads, D), got {tuple(q.shape)}"
            )
        if k.ndim != 3 or v.ndim != 3 or o.ndim != 3:
            raise RuntimeError(
                "k, v, and o must have shape (total_seqlen, num_heads, D), "
                f"got k={tuple(k.shape)}, v={tuple(v.shape)}, o={tuple(o.shape)}"
            )
        if (
            k.shape[0] != q.shape[0]
            or v.shape[0] != q.shape[0]
            or o.shape[0] != q.shape[0]
        ):
            raise RuntimeError("q, k, v, and o must have the same total_seqlen")
        if (
            k.shape[2] != q.shape[2]
            or v.shape[2] != q.shape[2]
            or o.shape[2] != q.shape[2]
        ):
            raise RuntimeError("q, k, v, and o must have the same D")
        if alpha.ndim != 2 or alpha.shape[0] != q.shape[0]:
            raise RuntimeError(
                f"alpha must have shape (total_seqlen, num_sab_heads), got {tuple(alpha.shape)}"
            )
        if total_seqlen != q.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match q.shape[0], got {total_seqlen} and {q.shape[0]}"
            )
        if cp_chunk_len % 64 != 0:
            raise RuntimeError(
                f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}"
            )
        if checkpoint_every_n_tokens < 0 or checkpoint_every_n_tokens % 64 != 0:
            raise RuntimeError(
                "checkpoint_every_n_tokens must be a non-negative multiple of 64, "
                f"got {checkpoint_every_n_tokens}"
            )
        if needs_checkpointing and (
            state_checkpoints is None or checkpoint_cu_starts is None
        ):
            raise RuntimeError(
                "state_checkpoints and checkpoint_cu_starts are required when checkpointing is enabled"
            )
        if not is_integer_dtype(cu_seqlens.dtype):
            raise RuntimeError(
                f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_q_heads, d = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    if _ctx is not None:
        total_t_blocks = _ctx.total_t_blocks
        total_cp_chunks = _ctx.total_cp_chunks
    else:
        total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
        total_cp_chunks = workspace_num_chunks_host(
            cu_seqlens, cp_chunk_len, total_seqlen
        )
    if not _skip_check:
        if o.shape[1] != num_sab_heads:
            raise RuntimeError(
                f"o heads must equal max(q heads, v heads)={num_sab_heads}, got {o.shape[1]}"
            )
        if alpha.shape[1] != num_sab_heads:
            raise RuntimeError(
                f"alpha heads must equal max(q heads, v heads)={num_sab_heads}, got {alpha.shape[1]}"
            )
        if not _FullyFusedDeltaRuleSm80.can_implement(
            num_q_heads, num_k_heads, num_v_heads, d, q.element_size()
        ):
            raise RuntimeError(
                "CPDeltaRulePrefillSm80 only supports head counts where q/v heads are positive multiples "
                f"of k heads, got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
            )
        if t.shape != (total_t_blocks, num_sab_heads, 64, 64):
            raise RuntimeError(
                "t must have shape "
                f"{(total_t_blocks, num_sab_heads, 64, 64)}, got {tuple(t.shape)}"
            )
        expected_fixed_state_shape = (total_cp_chunks, num_sab_heads, d, d)
        expected_state_shape = (num_seqs, num_sab_heads, d, d)
        use_state_indices = state_indices is not None
        if fixed_state.shape != expected_fixed_state_shape or (
            state is not None
            and (
                (not use_state_indices and state.shape != expected_state_shape)
                or (
                    use_state_indices
                    and tuple(state.shape[1:]) != expected_state_shape[1:]
                )
            )
        ):
            raise RuntimeError(
                "fixed_state/state must have shapes "
                f"{expected_fixed_state_shape} and {expected_state_shape}, "
                f"got {tuple(fixed_state.shape)} and "
                f"{None if state is None else tuple(state.shape)}"
            )
        if initial_state is not None and (
            (not use_state_indices and initial_state.shape != expected_state_shape)
            or (
                use_state_indices
                and tuple(initial_state.shape[1:]) != expected_state_shape[1:]
            )
        ):
            raise RuntimeError(
                f"initial_state must have shape "
                f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_shape[1:]}, "
                f"got {tuple(initial_state.shape)}"
            )
        if use_state_indices and (
            not is_integer_dtype(state_indices.dtype)
            or state_indices.shape != (num_seqs,)
        ):
            raise RuntimeError(
                f"state_indices must have shape {(num_seqs,)} and an integer dtype"
            )
        if q.shape[-1] != 128:
            raise RuntimeError(
                f"CPDeltaRulePrefillSm80 only supports D=128, got {q.shape[-1]}"
            )
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRulePrefillSm80 only supports fp16/bf16 inputs, got {q.dtype}"
            )
        if (
            k.dtype != q.dtype
            or v.dtype != q.dtype
            or o.dtype != q.dtype
            or t.dtype != q.dtype
        ):
            raise RuntimeError(
                "q/k/v/o/t dtypes must match, "
                f"got q={q.dtype}, k={k.dtype}, v={v.dtype}, o={o.dtype}, t={t.dtype}"
            )
        if alpha.dtype != torch.float32:
            raise RuntimeError(
                f"alpha must have dtype torch.float32, got {alpha.dtype}"
            )
        if initial_state is not None:
            state_dtype_to_cutlass(initial_state.dtype)
        if state is not None:
            state_dtype_to_cutlass(state.dtype)
        if needs_checkpointing:
            state_dtype_to_cutlass(state_checkpoints.dtype)
            if tuple(state_checkpoints.shape[1:]) != (num_sab_heads, d, d):
                raise RuntimeError(
                    "state_checkpoints must have shape "
                    f"[*, {num_sab_heads}, {d}, {d}], got {tuple(state_checkpoints.shape)}"
                )
            if not is_integer_dtype(
                checkpoint_cu_starts.dtype
            ) or checkpoint_cu_starts.shape != (num_seqs + 1,):
                raise RuntimeError(
                    f"checkpoint_cu_starts must have shape {(num_seqs + 1,)} and an integer dtype"
                )
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("t", t),
            ("fixed_state", fixed_state),
            ("alpha", alpha),
            ("o", o),
            ("state_indices", state_indices),
            ("state_checkpoints", state_checkpoints),
            ("checkpoint_cu_starts", checkpoint_cu_starts),
        ):
            if tensor is None:
                continue
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
        for name, tensor in (("state", state), ("initial_state", initial_state)):
            if tensor is None:
                continue
            if not use_state_indices and not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")
    max_cp_chunks_per_seq = (
        _ctx.max_cp_chunks_per_seq
        if _ctx is not None
        else max_num_chunks_host(max_seqlen, cp_chunk_len)
    )
    if total_cp_chunks == 0:
        return
    q_tma = q.as_strided(
        (total_seqlen, d, num_q_heads),
        (num_q_heads * d, 1, d),
    )
    k_tma = k.as_strided(
        (d, total_seqlen, num_k_heads),
        (1, num_k_heads * d, d),
    )
    v_tma = v.as_strided(
        (d, total_seqlen, num_v_heads),
        (1, num_v_heads * d, d),
    )
    o_tma = o.as_strided(
        (d, total_seqlen, num_sab_heads),
        (1, num_sab_heads * d, d),
    )
    t_tma = t.as_strided(
        (64, 64, num_sab_heads, total_t_blocks),
        (
            64,
            1,
            64 * 64,
            num_sab_heads * 64 * 64,
        ),
    )

    kernel_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }[q.dtype]

    stream = (
        _ctx.stream
        if _ctx is not None
        else _stream
        if _stream is not None
        else cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)
    )

    needs_initial_state = initial_state is not None
    store_final_state = state is not None
    use_state_indices = state_indices is not None
    spec = dict(
        needs_initial_state=needs_initial_state,
        store_final_state=store_final_state,
        initial_state_dtype=(
            initial_state.dtype if needs_initial_state else torch.float32
        ),
        state_dtype=state.dtype if store_final_state else torch.float32,
        checkpoint_state_dtype=(
            state_checkpoints.dtype if needs_checkpointing else torch.float32
        ),
        use_state_indices=use_state_indices,
        needs_checkpointing=needs_checkpointing,
        cu_seqlens_dtype=cu_seqlens.dtype,
        state_indices_dtype=state_indices.dtype if use_state_indices else None,
        checkpoint_cu_starts_dtype=(
            checkpoint_cu_starts.dtype if needs_checkpointing else None
        ),
        state_inner_strides=(
            tuple(state.stride()[1:])
            if use_state_indices and store_final_state
            else None
        ),
        initial_state_inner_strides=(
            tuple(initial_state.stride()[1:])
            if use_state_indices and needs_initial_state
            else None
        ),
    )
    state_arg = state if store_final_state else fixed_state
    if _ptr_abi:
        mk = _ptr_factory(
            _ctx,
            device,
            _stream,
            (
                ("q", q, 16),
                ("k", k, 16),
                ("v", v, 16),
                ("o", o, 16),
                ("t", t, 16),
                ("alpha", alpha, 16),
                ("state_arg", state_arg, 16),
                ("fixed_state", fixed_state, 16),
                ("cu_seqlens", cu_seqlens, 8),
                ("initial_state", initial_state if needs_initial_state else None, 16),
                (
                    "state_indices",
                    state_indices if use_state_indices else None,
                    state_indices.element_size() if use_state_indices else 1,
                ),
                (
                    "state_checkpoints",
                    state_checkpoints if needs_checkpointing else None,
                    16,
                ),
                (
                    "checkpoint_cu_starts",
                    checkpoint_cu_starts if needs_checkpointing else None,
                    8,
                ),
            ),
        )
        kernel = _get_prefill_ptr_kernel(kernel_dtype, **spec)
        state_cutlass = state_dtype_to_cutlass(
            state.dtype if store_final_state else torch.float32
        )
        init_cutlass = state_dtype_to_cutlass(
            initial_state.dtype if needs_initial_state else torch.float32
        )
        ckpt_cutlass = state_dtype_to_cutlass(
            state_checkpoints.dtype if needs_checkpointing else torch.float32
        )
        # Inactive optional slots get an address that is valid and aligned --
        # the workspace's -- and the kernel body reads them only under
        # `const_expr`.
        placeholder = fixed_state
        args = (
            mk(kernel_dtype, q, 16, "q"),
            mk(kernel_dtype, k, 16, "k"),
            mk(kernel_dtype, v, 16, "v"),
            mk(kernel_dtype, t, 16, "t"),
            mk(kernel_dtype, o, 16, "o"),
            mk(cutlass.Float32, alpha, 16, "alpha"),
            mk(state_cutlass, state_arg, 16, "state_arg"),
            mk(cutlass.Float32, fixed_state, 16, "fixed_state"),
            mk(
                init_cutlass,
                initial_state if needs_initial_state else placeholder,
                16,
                "initial_state",
            ),
            mk(
                integer_dtype_to_cutlass(
                    state_indices.dtype if use_state_indices else torch.int32
                ),
                state_indices if use_state_indices else placeholder,
                4,
                "state_indices",
            ),
            mk(
                ckpt_cutlass,
                state_checkpoints if needs_checkpointing else placeholder,
                16,
                "state_checkpoints",
            ),
            mk(
                integer_dtype_to_cutlass(
                    checkpoint_cu_starts.dtype if needs_checkpointing else torch.int32
                ),
                checkpoint_cu_starts if needs_checkpointing else placeholder,
                8,
                "checkpoint_cu_starts",
            ),
            mk(integer_dtype_to_cutlass(cu_seqlens.dtype), cu_seqlens, 8, "cu_seqlens"),
            cutlass.Float32(scale),
            cutlass.Int32(total_seqlen),
            cutlass.Int32(total_t_blocks),
            cutlass.Int32(state.shape[0] if store_final_state else 1),
            cutlass.Int32(state.shape[1] if store_final_state else 1),
            cutlass.Int32(state.stride(0) if store_final_state else 1),
            cutlass.Int32(state.stride(1) if store_final_state else 1),
            cutlass.Int32(initial_state.shape[0] if needs_initial_state else 1),
            cutlass.Int32(initial_state.shape[1] if needs_initial_state else 1),
            cutlass.Int32(initial_state.stride(0) if needs_initial_state else 1),
            cutlass.Int32(initial_state.stride(1) if needs_initial_state else 1),
            cutlass.Int32(num_q_heads),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(num_seqs),
            cutlass.Int32(state_checkpoints.shape[0] if needs_checkpointing else 1),
            cutlass.Int32(checkpoint_every_n_tokens),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(total_cp_chunks),
            int(kernel.num_v_splits * num_sab_heads * max_cp_chunks_per_seq),
            int(num_seqs),
            stream,
        )
        opts = (
            _ctx.compile_options
            if _ctx is not None
            else _sm80_cp_compile_options(device)
        )
        compiled = get_cached_compile(kernel, opts)
        if compiled is None:
            compiled = cached_compile(kernel, *args, compile_options=opts)
        if _plan_sink is not None:
            _plan_sink.append((compiled, args))
        else:
            compiled(*args)
        return
    kernel = _get_prefill_kernel(kernel_dtype, **spec)
    compiled = get_cached_compile(
        kernel,
        _ctx.compile_options if _ctx is not None else _sm80_cp_compile_options(device),
    )
    if compiled is None:
        from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
            *args, **{**kwargs, "enable_tvm_ffi": True}
        )
        initial_state_cute = (
            from_dlpack(initial_state, assumed_align=16).mark_layout_dynamic()
            if needs_initial_state
            else None
        )
        state_indices_cute = (
            from_dlpack(state_indices, assumed_align=4).mark_layout_dynamic()
            if use_state_indices
            else None
        )
        total_checkpoints = state_checkpoints.shape[0] if needs_checkpointing else 1
        state_checkpoints_cute = (
            from_dlpack(
                state_checkpoints.reshape(-1), assumed_align=16
            ).mark_layout_dynamic()
            if needs_checkpointing
            else None
        )
        checkpoint_cu_cute = (
            from_dlpack(checkpoint_cu_starts, assumed_align=8).mark_layout_dynamic()
            if needs_checkpointing
            else None
        )
        kernel_args = (
            from_dlpack(q_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(t_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(o_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
            from_dlpack(alpha.reshape(-1), assumed_align=16).mark_layout_dynamic(),
            from_dlpack(state_arg, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(
                fixed_state.reshape(-1), assumed_align=16
            ).mark_layout_dynamic(),
            initial_state_cute,
            state_indices_cute,
            state_checkpoints_cute,
            checkpoint_cu_cute,
            from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
            cutlass.Float32(scale),
            cutlass.Int32(num_q_heads),
            cutlass.Int32(num_k_heads),
            cutlass.Int32(num_v_heads),
            cutlass.Int32(num_sab_heads),
            cutlass.Int32(num_seqs),
            cutlass.Int32(total_checkpoints),
            cutlass.Int32(checkpoint_every_n_tokens),
            cutlass.Int32(cp_chunk_len),
            cutlass.Int32(total_cp_chunks),
            int(kernel.num_v_splits * num_sab_heads * max_cp_chunks_per_seq),
            int(num_seqs),
            stream,
        )
        compiled = cached_compile(
            kernel,
            *kernel_args,
            compile_options=(
                _ctx.compile_options
                if _ctx is not None
                else _sm80_cp_compile_options(device)
            ),
        )
    if _plan_sink is not None:
        _plan_sink.append(
            (
                compiled,
                (
                    q_tma,
                    k_tma,
                    v_tma,
                    t_tma,
                    o_tma,
                    alpha.reshape(-1),
                    state_arg,
                    fixed_state.reshape(-1),
                    initial_state if needs_initial_state else None,
                    state_indices if use_state_indices else None,
                    state_checkpoints.reshape(-1) if needs_checkpointing else None,
                    checkpoint_cu_starts if needs_checkpointing else None,
                    cu_seqlens,
                    scale,
                    num_q_heads,
                    num_k_heads,
                    num_v_heads,
                    num_sab_heads,
                    num_seqs,
                    state_checkpoints.shape[0] if needs_checkpointing else 1,
                    checkpoint_every_n_tokens,
                    cp_chunk_len,
                    total_cp_chunks,
                    int(kernel.num_v_splits * num_sab_heads * max_cp_chunks_per_seq),
                    int(num_seqs),
                    stream,
                ),
            )
        )
    else:
        compiled(
            q_tma,
            k_tma,
            v_tma,
            t_tma,
            o_tma,
            alpha.reshape(-1),
            state_arg,
            fixed_state.reshape(-1),
            initial_state if needs_initial_state else None,
            state_indices if use_state_indices else None,
            state_checkpoints.reshape(-1) if needs_checkpointing else None,
            checkpoint_cu_starts if needs_checkpointing else None,
            cu_seqlens,
            scale,
            num_q_heads,
            num_k_heads,
            num_v_heads,
            num_sab_heads,
            num_seqs,
            state_checkpoints.shape[0] if needs_checkpointing else 1,
            checkpoint_every_n_tokens,
            cp_chunk_len,
            total_cp_chunks,
            int(kernel.num_v_splits * num_sab_heads * max_cp_chunks_per_seq),
            int(num_seqs),
            stream,
        )


def cp_delta_rule_dsl_sm80(
    o: torch.Tensor,
    state: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    initial_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    max_seqlen: int | None = None,
    cp_chunk_len: int | None = None,
    cp_chunk_len_granularity: int = CP_CHUNK_LEN_GRANULARITY,
    # Threaded to all four stages. With a sink the stages record what they
    # would launch and launch nothing, so the workspaces come back allocated
    # and unfilled -- which is all a plan needs, since it keeps the argument
    # descriptors rather than their contents.
    # Opt-in for all four stages at once. Off by default: the entries
    # are proven one at a time and nothing public reaches this yet.
    _ptr_abi: bool = False,
    _plan_sink: list | None = None,
):
    """Run the CP SM80 delta-rule prefill pipeline on flat varlen tensors.

    Q/K/V/alpha/beta/O/state follow the same public layout as the non-CP
    prefill path. Internal T, M, N, and fixed-state tensors use the varlen
    workspace layout.
    """
    import cuda.bindings.driver as cuda_driver

    device = q.device
    device_name = get_device_name(device)
    device_capability = get_compute_capability(device)
    num_sms = get_device_sm_count(device)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(device).cuda_stream)

    total_seqlen = q.shape[0]
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None and num_seqs == 1:
        max_seqlen = total_seqlen
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided when num_seqs != 1")
    if max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    if cp_chunk_len is None:
        num_heads = max(q.shape[1], v.shape[1])
        cp_chunk_len = choose_cp_chunk_len_host(
            max_seqlen,
            num_heads,
            num_sms,
            chunk_len_granularity=cp_chunk_len_granularity,
            device_capability=device_capability,
            total_seqlen=total_seqlen,
            num_seqs=num_seqs,
            device_name=device_name,
        )
    if q.ndim != 3:
        raise RuntimeError(
            f"q must have shape (total_seqlen, num_q_heads, D), got {tuple(q.shape)}"
        )
    if k.ndim != 3 or v.ndim != 3 or o.ndim != 3:
        raise RuntimeError(
            "k, v, and o must have shape (total_seqlen, num_heads, D), "
            f"got k={tuple(k.shape)}, v={tuple(v.shape)}, o={tuple(o.shape)}"
        )
    if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0] or o.shape[0] != q.shape[0]:
        raise RuntimeError("q, k, v, and o must have the same total_seqlen")
    if k.shape[2] != q.shape[2] or v.shape[2] != q.shape[2] or o.shape[2] != q.shape[2]:
        raise RuntimeError("q, k, v, and o must have the same D")
    if alpha.ndim != 2 or alpha.shape[0] != q.shape[0]:
        raise RuntimeError(
            f"alpha must have shape (total_seqlen, num_sab_heads), got {tuple(alpha.shape)}"
        )
    if beta.ndim != 2 or beta.shape[0] != q.shape[0]:
        raise RuntimeError(
            f"beta must have shape (total_seqlen, num_sab_heads), got {tuple(beta.shape)}"
        )
    if cp_chunk_len % 64 != 0:
        raise RuntimeError(f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}")
    needs_checkpointing = checkpoint_every_n_tokens > 0
    if checkpoint_every_n_tokens < 0 or checkpoint_every_n_tokens % 64 != 0:
        raise RuntimeError(
            "checkpoint_every_n_tokens must be a non-negative multiple of 64, "
            f"got {checkpoint_every_n_tokens}"
        )
    if needs_checkpointing and (
        state_checkpoints is None or checkpoint_cu_starts is None
    ):
        raise RuntimeError(
            "state_checkpoints and checkpoint_cu_starts are required when checkpointing is enabled"
        )
    if not is_integer_dtype(cu_seqlens.dtype):
        raise RuntimeError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )
    if not cu_seqlens.is_contiguous():
        raise RuntimeError("cu_seqlens must be contiguous")
    _, num_q_heads, d = q.shape
    num_k_heads = k.shape[1]
    num_v_heads = v.shape[1]
    num_sab_heads = max(num_q_heads, num_v_heads)
    if o.shape[1] != num_sab_heads:
        raise RuntimeError(
            f"o heads must equal max(q heads, v heads)={num_sab_heads}, got {o.shape[1]}"
        )
    if alpha.shape[1] != num_sab_heads:
        raise RuntimeError(
            f"alpha heads must equal max(q heads, v heads)={num_sab_heads}, got {alpha.shape[1]}"
        )
    if beta.shape[1] != num_sab_heads:
        raise RuntimeError(
            f"beta heads must equal max(q heads, v heads)={num_sab_heads}, got {beta.shape[1]}"
        )
    if not _FullyFusedDeltaRuleSm80.can_implement(
        num_q_heads, num_k_heads, num_v_heads, d, q.element_size()
    ):
        raise RuntimeError(
            "CPDeltaRuleSm80 only supports head counts where q/v heads are positive multiples "
            f"of k heads, got q={num_q_heads}, k={num_k_heads}, v={num_v_heads}"
        )
    expected_state_shape = (num_seqs, num_sab_heads, d, d)
    use_state_indices = state_indices is not None
    if state is not None and (
        (not use_state_indices and state.shape != expected_state_shape)
        or (use_state_indices and tuple(state.shape[1:]) != expected_state_shape[1:])
    ):
        raise RuntimeError(
            f"state must have shape "
            f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_shape[1:]}, "
            f"got {tuple(state.shape)}"
        )
    if initial_state is not None and (
        (not use_state_indices and initial_state.shape != expected_state_shape)
        or (
            use_state_indices
            and tuple(initial_state.shape[1:]) != expected_state_shape[1:]
        )
    ):
        raise RuntimeError(
            f"initial_state must have shape "
            f"{('[N_pool]' if use_state_indices else f'[{num_seqs}]')} + {expected_state_shape[1:]}, "
            f"got {tuple(initial_state.shape)}"
        )
    if use_state_indices and (
        not is_integer_dtype(state_indices.dtype) or state_indices.shape != (num_seqs,)
    ):
        raise RuntimeError(
            f"state_indices must have shape {(num_seqs,)} and an integer dtype"
        )
    if q.shape[-1] != 128:
        raise RuntimeError(f"CPDeltaRuleSm80 only supports D=128, got {q.shape[-1]}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"CPDeltaRuleSm80 only supports fp16/bf16 inputs, got {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype or o.dtype != q.dtype:
        raise RuntimeError(
            "q/k/v/o dtypes must match, "
            f"got q={q.dtype}, k={k.dtype}, v={v.dtype}, o={o.dtype}"
        )
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        raise RuntimeError(
            f"alpha/beta must have dtype torch.float32, got {alpha.dtype} and {beta.dtype}"
        )
    if initial_state is not None:
        state_dtype_to_cutlass(initial_state.dtype)
    if state is not None:
        state_dtype_to_cutlass(state.dtype)
    if needs_checkpointing:
        state_dtype_to_cutlass(state_checkpoints.dtype)
        if tuple(state_checkpoints.shape[1:]) != (num_sab_heads, d, d):
            raise RuntimeError(
                "state_checkpoints must have shape "
                f"[*, {num_sab_heads}, {d}, {d}], got {tuple(state_checkpoints.shape)}"
            )
        if not is_integer_dtype(
            checkpoint_cu_starts.dtype
        ) or checkpoint_cu_starts.shape != (num_seqs + 1,):
            raise RuntimeError(
                f"checkpoint_cu_starts must have shape {(num_seqs + 1,)} and an integer dtype"
            )
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("alpha", alpha),
        ("beta", beta),
        ("o", o),
        ("state_indices", state_indices),
        ("state_checkpoints", state_checkpoints),
        ("checkpoint_cu_starts", checkpoint_cu_starts),
    ):
        if tensor is None:
            continue
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    for name, tensor in (("state", state), ("initial_state", initial_state)):
        if tensor is None:
            continue
        if not use_state_indices and not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    # One context per composition call, and only here: the four entries below
    # then share this call's device and stream binding, its compile options,
    # its chunk geometry, and one alignment-checked pointer per distinct
    # buffer. A stage called on its own gets no context and keeps its own
    # `_check_ptr_abi`.
    ctx = (
        _CPInvocation(
            device=device,
            stream=stream,
            compile_options=_sm80_cp_compile_options(device),
            total_t_blocks=workspace_num_chunks_host(cu_seqlens, 64, total_seqlen),
            total_cp_chunks=workspace_num_chunks_host(
                cu_seqlens, cp_chunk_len, total_seqlen
            ),
            max_t_blocks_per_seq=max_num_chunks_host(max_seqlen, 64),
            max_cp_chunks_per_seq=max_num_chunks_host(max_seqlen, cp_chunk_len),
            cp_chunk_len=cp_chunk_len,
        )
        if _ptr_abi
        else None
    )

    t = cp_delta_rule_t_precompute_dsl_sm80(
        k,
        beta,
        cu_seqlens,
        total_seqlen,
        max_seqlen=max_seqlen,
        _skip_check=True,
        _device=device,
        _stream=stream,
        _ptr_abi=_ptr_abi,
        _ctx=ctx,
        _plan_sink=_plan_sink,
    )
    local_transfer, local_state = cp_delta_rule_mn_precompute_dsl_sm80(
        k,
        v,
        t,
        alpha,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        _skip_check=True,
        _device=device,
        _stream=stream,
        _ptr_abi=_ptr_abi,
        _ctx=ctx,
        _plan_sink=_plan_sink,
    )
    fixed_state = cp_delta_rule_fixup_dsl_sm80(
        local_transfer,
        local_state,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        initial_state=initial_state,
        state_indices=state_indices,
        _skip_check=True,
        _device=device,
        _stream=stream,
        _ptr_abi=_ptr_abi,
        _ctx=ctx,
        _plan_sink=_plan_sink,
    )

    cp_delta_rule_prefill_dsl_sm80(
        o,
        state,
        q,
        k,
        v,
        t,
        fixed_state,
        alpha,
        scale,
        cu_seqlens,
        total_seqlen,
        cp_chunk_len=cp_chunk_len,
        max_seqlen=max_seqlen,
        initial_state=initial_state,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        _skip_check=True,
        _device=device,
        _stream=stream,
        _ptr_abi=_ptr_abi,
        _ctx=ctx,
        _plan_sink=_plan_sink,
    )
