# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-through of per-token head rows into PAGED pools at ``slot_mapping`` -- the serving cache write.

A serving stack keeps the post-norm, post-RoPE K and the V of every token in a paged KV cache: pools
``[num_pages, H, page_size, D]`` whose per-token row is addressed by a FLAT slot ``slot = page * page_size + offset``
(the ``slot_mapping [T]`` the stack hands every attention layer).  This kernel copies one ``[T, H, D]`` operand's
rows into such a pool, bitwise -- it is the block's own ``_CacheWrite`` stage, launched once for K AND V (two
operands in ONE launch: rows ``[0, T*H)`` are the first operand's, ``[T*H, 2*T*H)`` the second's) and once more for
the raw indexer key when the block declares the indexer band.

Layout contract (the SDPA adapter's own paged contract): the pool is a 4-D view ``[num_pages, H, page_size, D]`` with
``D`` contiguous and the first three strides FREE -- HND compact (``strides = (H*page_size*D, page_size*D, D, 1)``) or
NHD storage (``(page_size*H*D, D, H*D, 1)``) bind as views, nothing is repacked.  ``slot_mapping`` is int32 or int64
(the dtype is a compile key); a NEGATIVE slot is the serving stacks' padding convention and writes nothing; a slot at
or past ``num_pages * page_size`` is a caller-contract violation and ALSO writes nothing (never a write past the pool).
Slot VALUES are device data, never read on the host.

Shaped like ``elementwise.py``: one head row is ``D`` elements, ``LANES = D // 8`` lanes move 16 bytes each (a
32-lane warp per 512-B row at ``D = 256``), the loads of a row group are issued before any store, and the copy is a
bit pattern (no unpack, no cast) so a pool row IS the source row.  No shared memory and no mbarrier anywhere, so the
two pre-development tables of a FROST kernel (barriers, SMEM buffers) are empty by construction.

Traffic per launch: ``n_ops * T * H * D * elem`` read + the same written, plus ``T`` slot reads.  The roofline term is
HBM bytes; no perf claim is made here until one is measured on an exclusive GPU.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v2, ld_global_v4, st_global_v4

from ..cache_write_contract import _SLOT_DTYPES, check_pool, check_slot_mapping
from .elementwise import validate_shape
from .qk_norm_rope import ACCESS_BYTES, fake_rowmajor_dynamic_token_stride, lanes_per_row, vec_chunks

DEFAULT_THREADS_PER_CTA = 128
DEFAULT_ROWS_PER_GROUP = 2
DEFAULT_CONST_HEAD_COUNT = True
"""The elementwise kernel's three occupancy / address-math knobs, same defaults, same reason (``elementwise.py``);
re-measure per arch before changing them."""

_IO_DTYPES = (torch.bfloat16, torch.float16)
_FAKE_STREAM = None
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with a compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def validate_cache_write_shape(d: int, page_size: int, threads_per_cta: int) -> None:
    """Raise on any geometry this kernel cannot address.  The row geometry is the elementwise kernel's
    (``validate_shape``); the page size only has to be positive here -- the block's INPUT contract (a multiple of 16,
    so a 4-token block never straddles a page) is the API's rule, checked where the attribute is declared."""
    validate_shape(d, threads_per_cta)
    if int(page_size) < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")


@cute.kernel
def frost_cache_write(
    mSrcA: cute.Tensor,  # [T, H, D] operand A (K): token stride symbolic -- a column slice of the projection slab binds as is
    mSrcB: Optional[cute.Tensor],  # [T, H, D] operand B (V), or None (one operand per launch)
    mPoolA: cute.Tensor,  # [P, H, page_size, D]: the first three strides symbolic (HND compact or NHD), D contiguous
    mPoolB: Optional[cute.Tensor],  # same, operand B's pool
    mSlot: cute.Tensor,  # [T] int32 or int64, contiguous
    n_rows: cutlass.Int32,  # T * H: the rows of ONE operand
    n_slots: cutlass.Int64,  # P * page_size: a slot outside [0, n_slots) writes nothing
    h: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """``pool[slot // page_size, head, slot % page_size, :] = src[token, head, :]`` for every ``(token, head)`` of
    every operand, with ``slot = slot_mapping[token]``; rows whose slot is negative or past the pool's capacity are
    skipped (the store is predicated, the row's loads still issue from a clamped in-range source row).

    The operand of a row is a RUNTIME select (``row >= n_rows``): base pointers and strides are Int64 scalars chosen
    per row, so one artifact writes K and V in one launch without a second grid axis.  Both operands share ``H``,
    ``D``, ``slot_mapping`` and the compile-time ``page_size``; their pools may differ in strides.
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    has_b = cutlass.const_expr(mSrcB is not None)
    bpe = cutlass.Int64(2)  # bf16 / f16: the only element widths this kernel serves (the runner refuses the rest)

    # `_h` is the head count the address math uses: a Python constant under `const_head_count` (shift + mask at the
    # block's power-of-two head counts), else a runtime software divide per row -- `elementwise.py` says why.
    _h = cutlass.Int32(h_ct) if cutlass.const_expr(const_head_count) else h
    n_total = (n_rows + n_rows) if cutlass.const_expr(has_b) else n_rows

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    row0 = (cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp) * cutlass.Int32(rows_per_group)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)

    # PASS 1: resolve every row's source / destination and issue its loads -- pure memory-level parallelism, nothing
    # consumes a loaded value before the next load can issue.
    lives = []
    dsts = []
    srcs = []
    for r in cutlass.range_constexpr(rows_per_group):
        row = row0 + cutlass.Int32(r)
        valid = row < n_total
        is_b = row >= n_rows
        row_op = (row - n_rows) if is_b else row  # the row within its operand
        row_c = row_op if valid else cutlass.Int32(0)  # a tail row past the grid reads row 0 and stores nothing
        token = row_c // _h
        head = row_c % _h
        token64 = token.to(cutlass.Int64)
        head64 = head.to(cutlass.Int64)
        # The slot of this token.  int64: two 32-bit halves recombined (the 64-bit value is what the range check
        # sees, so a slot >= 2^31 can never alias an in-range one); int32: one word, sign-extended.
        if cutlass.const_expr(slot_i64):
            lo, hi = ld_global_v2(mSlot.iterator.toint() + token64 * cutlass.Int64(8), cutlass.Int32)
            slot = (hi.to(cutlass.Int64) * cutlass.Int64(1 << 32)) + (lo.to(cutlass.Int64) & cutlass.Int64(0xFFFFFFFF))
        else:
            slot = ld_global(mSlot.iterator.toint() + token64 * cutlass.Int64(4), cutlass.Int32).to(cutlass.Int64)
        live = valid & (slot >= cutlass.Int64(0)) & (slot < n_slots)
        # In-range slots fit 32 bits (a pool of 2^31 rows of >= 16 B exceeds any device); the Int32 divide by the
        # compile-time page size strength-reduces.  A dead row's page / offset are garbage its predicated store ignores.
        slot32 = slot.to(cutlass.Int32)
        page = slot32 // cutlass.Int32(page_size)
        off = slot32 - page * cutlass.Int32(page_size)
        # Operand select: Int64 scalars chosen per row (no second grid axis, no divergent branch).
        if cutlass.const_expr(has_b):
            src_base = mSrcB.iterator.toint() if is_b else mSrcA.iterator.toint()
            src_tok = cutlass.Int64(mSrcB.stride[0]) if is_b else cutlass.Int64(mSrcA.stride[0])
            pool_base = mPoolB.iterator.toint() if is_b else mPoolA.iterator.toint()
            p0 = cutlass.Int64(mPoolB.stride[0]) if is_b else cutlass.Int64(mPoolA.stride[0])
            p1 = cutlass.Int64(mPoolB.stride[1]) if is_b else cutlass.Int64(mPoolA.stride[1])
            p2 = cutlass.Int64(mPoolB.stride[2]) if is_b else cutlass.Int64(mPoolA.stride[2])
        else:
            src_base = mSrcA.iterator.toint()
            src_tok = cutlass.Int64(mSrcA.stride[0])
            pool_base = mPoolA.iterator.toint()
            p0 = cutlass.Int64(mPoolA.stride[0])
            p1 = cutlass.Int64(mPoolA.stride[1])
            p2 = cutlass.Int64(mPoolA.stride[2])
        src_addr = src_base + (token64 * src_tok + head64 * cutlass.Int64(d)) * bpe
        dst_addr = pool_base + (page.to(cutlass.Int64) * p0 + head64 * p1 + off.to(cutlass.Int64) * p2) * bpe
        row_src = []
        for c in cutlass.range_constexpr(chunks):
            coff = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
            row_src.append(ld_global_v4(src_addr + coff, cutlass.Int32))
        lives.append(live)
        dsts.append(dst_addr)
        srcs.append(row_src)

    # PASS 2: the stores, predicated on the row's slot being live.  A bit-pattern copy: the pool row IS the source row.
    for r in cutlass.range_constexpr(rows_per_group):
        if lives[r]:
            for c in cutlass.range_constexpr(chunks):
                coff = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                st_global_v4(dsts[r] + coff, srcs[r][c], cutlass.Int32)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def cache_write_launch(
    src_a: cute.Tensor,
    src_b: Optional[cute.Tensor],
    pool_a: cute.Tensor,
    pool_b: Optional[cute.Tensor],
    slot: cute.Tensor,
    n_rows: cutlass.Int32,
    n_slots: cutlass.Int64,
    h: cutlass.Int32,
    n_blocks: cutlass.Int32,
    h_ct: cutlass.Constexpr[int],
    const_head_count: cutlass.Constexpr[bool],
    d: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    rows_per_group: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_cache_write(
        src_a, src_b, pool_a, pool_b, slot, n_rows, n_slots, h, h_ct, const_head_count, d, page_size, slot_i64, threads_per_cta, rows_per_group, use_pdl
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


class CacheWriteRecipe(NamedTuple):
    """Build-time facts of one cache-write launch.  Every field is plan-time derivable; the token count, the pool
    capacity and the pool strides ride in at runtime (Rule 4: one artifact per (dtype, H, D, page_size, slot dtype,
    operand count))."""

    compiled: object
    h: int
    d: int
    page_size: int
    rows_per_cta: int
    two_operands: bool
    dtype: object  # the io torch dtype the artifact was traced for (a compile key; the runner refuses any other)
    slot_dtype: object  # torch.int32 or torch.int64 (a compile key)


def _fake_pool(dtype, h: int, page_size: int, d: int):
    """``[P, H, page_size, D]`` with P and the first three strides SYMBOLIC and D contiguous: one artifact binds an HND
    compact pool and an NHD one (the layout is nothing but those strides, bound at execute).  ``page_size`` is concrete:
    it is a compile key (the in-kernel divide), so a pool of another page size is refused at the tvm-ffi boundary."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type

    return cute.runtime.make_fake_tensor(
        dtype=_convert_to_cutlass_data_type(dtype),
        shape=(cute.sym_int(), h, page_size, d),
        stride=(cute.sym_int(width=64), cute.sym_int(width=64), cute.sym_int(width=64), 1),
        assumed_align=16,
    )


def _fake_slot(slot_dtype):
    """``[T]`` contiguous int32 / int64 with a symbolic length; 4- / 8-byte aligned (the natural alignment of its element)."""
    i64 = slot_dtype == torch.int64
    return cute.runtime.make_fake_compact_tensor(cutlass.Int64 if i64 else cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=8 if i64 else 4)


def compile_cache_write(
    *,
    dtype,
    h: int,
    d: int,
    page_size: int,
    slot_dtype=torch.int32,
    two_operands: bool = True,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
    const_head_count: bool = DEFAULT_CONST_HEAD_COUNT,
    use_pdl: bool = False,
) -> CacheWriteRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch."""
    global _FAKE_STREAM
    validate_cache_write_shape(d, page_size, threads_per_cta)
    if dtype not in _IO_DTYPES:
        raise ValueError(f"cache write serves bf16 / f16 pools only, got {dtype}")
    if slot_dtype not in _SLOT_DTYPES:
        raise ValueError(f"slot_mapping must be int32 or int64, got {slot_dtype}")
    if h < 1:
        raise ValueError(f"h must be >= 1, got {h}")
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)

    key = (
        str(dtype),
        int(h),
        int(d),
        int(page_size),
        str(slot_dtype),
        bool(two_operands),
        int(threads_per_cta),
        int(rows_per_group),
        bool(const_head_count),
        bool(use_pdl),
        current_device(),
    )
    if key not in compiled_cache:
        # The dict stays the in-process memo; behind it sits the persistent compiled-plan cache (``cudnn.frost.compiled_cache``),
        # keyed by this file's digest + the dict key's plain values: a second process reloads the object instead of re-tracing.
        cache_key = template_key(globals(), {"key": key}, "compile_cache_write")
        tok = cute.sym_int()
        src_a = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d)
        src_b = fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) if two_operands else None
        pool_a = _fake_pool(dtype, h, page_size, d)
        pool_b = _fake_pool(dtype, h, page_size, d) if two_operands else None
        compiled_cache[key] = compile_cached(
            cache_write_launch,
            src_a,
            src_b,
            pool_a,
            pool_b,
            _fake_slot(slot_dtype),
            cutlass.Int32(0),  # n_rows   ) runtime; the zeros pin the TYPE only
            cutlass.Int64(0),  # n_slots  )
            cutlass.Int32(h),  # h        )
            cutlass.Int32(0),  # n_blocks )
            int(h),
            bool(const_head_count),
            int(d),
            int(page_size),
            slot_dtype == torch.int64,
            int(threads_per_cta),
            int(rows_per_group),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
            cache_key=cache_key,
            symbol="cudnn_block_cache_write",
        )
    return CacheWriteRecipe(
        compiled=compiled_cache[key],
        h=int(h),
        d=int(d),
        page_size=int(page_size),
        rows_per_cta=(threads_per_cta // lanes_per_row(d)) * rows_per_group,
        two_operands=bool(two_operands),
        dtype=dtype,
        slot_dtype=slot_dtype,
    )


def run_cache_write(r: CacheWriteRecipe, src_a, src_b, pool_a, pool_b, slot, *, stream) -> None:
    """Launch.  ``src_*`` are ``[T, H, D]`` (any token stride), ``pool_*`` are ``[P, H, page_size, D]`` with D
    contiguous, ``slot`` is ``[T]``; ``src_b`` / ``pool_b`` are both given (two-operand artifact) or both ``None``."""
    if r.two_operands != (src_b is not None) or (src_b is None) != (pool_b is None):
        raise ValueError(
            f"this artifact was compiled for {'two operands' if r.two_operands else 'one operand'}; src_b / pool_b must be "
            f"{'both given' if r.two_operands else 'both None'} (Rule 1: no silent fallback)"
        )
    for name, ten in (("src_a", src_a), ("src_b", src_b)):
        if ten is None:
            continue
        if ten.dtype != r.dtype:
            raise ValueError(f"{name} is {ten.dtype} but this artifact was compiled for {r.dtype}; dtype is fixed per artifact")
        if ten.dim() != 3 or int(ten.shape[1]) != r.h or int(ten.shape[2]) != r.d:
            raise ValueError(f"{name} must be [T, H={r.h}, D={r.d}], got {tuple(ten.shape)}; H and D are fixed per artifact")
        if int(ten.stride(2)) != 1 or int(ten.stride(1)) != r.d:
            raise ValueError(f"{name} must have contiguous heads (strides (*, {r.d}, 1)), got {tuple(ten.stride())}")
        if (int(ten.stride(0)) * ten.element_size()) % 16 or ten.data_ptr() % 16:
            raise ValueError(f"{name} rows must be 16-byte aligned (token stride {ten.stride(0)} elements, data_ptr {ten.data_ptr():#x})")
    t = int(src_a.shape[0])
    if src_b is not None and int(src_b.shape[0]) != t:
        raise ValueError(f"src_a and src_b must carry the same T, got {t} and {int(src_b.shape[0])}")
    # Every operand on ONE CUDA device, src_a's (the kernel reads them through device pointers): a CPU slot_mapping -- a serving
    # stack's `torch.tensor(slots, dtype=torch.int32)` with no device= -- passed the dtype / rank / length checks and the kernel
    # dereferenced a host address, an illegal-address fault that is sticky for the process; `is_cuda` alone cannot tell a tensor on
    # ANOTHER GPU from a local one, so the pools and the slot mapping are compared against src_a's device by name.
    if not src_a.is_cuda:
        raise ValueError(f"src_a must be a CUDA tensor (the kernel reads every operand through a device pointer), got device {src_a.device}")
    if src_b is not None and src_b.device != src_a.device:
        raise ValueError(f"src_b must be on {src_a.device} with src_a, got {src_b.device}")
    check_pool(pool_a, "pool_a", h=r.h, page_size=r.page_size, d=r.d, dtype=r.dtype, device=src_a.device)
    if pool_b is not None:
        check_pool(pool_b, "pool_b", h=r.h, page_size=r.page_size, d=r.d, dtype=r.dtype, device=src_a.device)
    check_slot_mapping(slot, "slot_mapping", t=t, device=src_a.device)
    if slot.dtype != r.slot_dtype:
        raise ValueError(f"slot_mapping is {slot.dtype} but this artifact was compiled for {r.slot_dtype}; the slot dtype is fixed per artifact")
    n_rows = t * r.h
    n_total = 2 * n_rows if r.two_operands else n_rows
    n_blocks = (n_total + r.rows_per_cta - 1) // r.rows_per_cta
    # The capacity the kernel range-checks every slot against: the SMALLER pool's when two are written (a slot past
    # either pool would be a write past that pool).
    n_slots = int(pool_a.shape[0]) * r.page_size
    if pool_b is not None:
        n_slots = min(n_slots, int(pool_b.shape[0]) * r.page_size)
    r.compiled(
        src_a,
        src_b,
        pool_a,
        pool_b,
        slot,
        cutlass.Int32(n_rows),
        cutlass.Int64(n_slots),
        cutlass.Int32(r.h),
        cutlass.Int32(n_blocks),
        cuda.CUstream(int(stream)),
    )


def moved_bytes(t: int, h: int, d: int, *, elem_bytes: int = 2, n_ops: int = 2, slot_bytes: int = 4) -> int:
    """HBM traffic of one launch -- the denominator for an SOL number: every operand's rows read once and written once,
    plus the slot per token."""
    return 2 * n_ops * t * h * d * elem_bytes + t * slot_bytes


frost_cache_write.set_name_prefix("cudnn", remove_cutlass_symbol=True)
