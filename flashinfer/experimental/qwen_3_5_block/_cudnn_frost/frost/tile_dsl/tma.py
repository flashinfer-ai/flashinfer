# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

from .swizzle import swizzle_xor_128b
from .requirements import tma_gather4_requirement_error  # noqa: F401  re-export; the HOST entry is tile_dsl.requirements


def _advance(x, n):
    if hasattr(x, "subview"):
        return x.subview(n)
    return x + n


def _bulk_copy_ptr(x):
    if hasattr(x, "llvm_ptr"):
        return x.llvm_ptr
    if hasattr(x, "ir_value") and callable(x.ir_value):
        return x.ir_value()
    return x


@cute.jit
def cp_async_bulk_shared_cluster_shared_cta(dst_mem, src_mem, mbar, size, *, pred=None):
    if cutlass.const_expr(pred is None):
        nvvm.cp_async_bulk_shared_cluster_shared_cta(dst_mem, src_mem, mbar, size)
    else:
        nvvm.inline_ptx(
            "cp.async.bulk.shared::cluster.shared::cta" ".mbarrier::complete_tx::bytes " "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
            read_only_args=[
                _bulk_copy_ptr(dst_mem),
                _bulk_copy_ptr(src_mem),
                cutlass.Int32(size),
                _bulk_copy_ptr(mbar),
            ],
            predicate=pred,
        )


@cute.jit
def tma_load_tile(
    smem_tile,
    gmem_slice,
    mbar,
    *,
    cta_group: int = 1,
    mcast_mask=None,
    acquire: cutlass.Constexpr[bool] = True,
    l2_cache_hint=None,
):
    num_iters = smem_tile.tma_loads_per_tile
    granu_elems = smem_tile.tma_granu_elems
    sub_stride = smem_tile.tma_subtile_stride_elems
    if cutlass.const_expr(gmem_slice.desc_ptr is not None):
        tma_desc_ptr = gmem_slice.desc_ptr
        if cutlass.const_expr(acquire):
            nvvm.fence_proxy_acquire(
                nvvm.MemScope.GPU,
                tma_desc_ptr,
                128,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    else:
        tma_desc_ptr = gmem_slice.tma_desc.get_ptr()
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    for i in cutlass.range_constexpr(num_iters):
        d = coord_d + cutlass.Int32(i * granu_elems)
        smem_chunk = smem_tile.base.subview(i * sub_stride)
        if nvvm.elect_sync():
            coords = [d] + list(outer_coords)
            if cutlass.const_expr(cta_group == 1):
                nvvm.cp_async_bulk_tensor_shared_cta_global(
                    smem_chunk,
                    tma_desc_ptr,
                    coords,
                    mbar,
                    l2_cache_hint=l2_cache_hint,
                )
            else:
                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                    smem_chunk,
                    tma_desc_ptr,
                    coords,
                    mbar,
                    [],
                    multicast_mask=mcast_mask,
                    group=nvvm.CTAGroup.CTA_2,
                    l2_cache_hint=l2_cache_hint,
                )


@cute.jit
def tma_load_subtiles(
    smem_tile,
    gmem_slice,
    mbar,
    first: cutlass.Constexpr[int],
    count: cutlass.Constexpr[int],
    *,
    cta_group: int = 1,
    mcast_mask=None,
    acquire: cutlass.Constexpr[bool] = True,
    l2_cache_hint=None,
):
    """Issue subtiles ``first .. first + count`` of the tile load :func:`tma_load_tile` issues whole.

    Same op, same operands: subtile ``i`` lands at ``smem_tile.base + i * tma_subtile_stride_elems``
    from GMEM inner coordinate ``coord_d + i * tma_granu_elems``, one elected lane per subtile,
    completing on ``mbar``.  The caller splits a tile's subtiles across ISSUERS -- the d512
    2x2-datapath kernel's twin CTAs each issue half of every K/V sub-chunk with a two-CTA
    multicast mask, so that every CTA receives the whole sub-chunk while the L2 is read once per
    pair of pairs -- and the mbarrier accounting stays the receiver's: under ``cta_group=2`` every
    byte that lands in CTA d completes on the barrier at the same offset in d's PAIR LEADER, whoever
    issued it (probe mcast_twin, 2026-10-01), so the leader's ``expect_tx`` is the bytes landing in
    its pair, not the bytes this CTA issues.  ``first + count <= smem_tile.tma_loads_per_tile``.
    """
    num_iters = smem_tile.tma_loads_per_tile
    granu_elems = smem_tile.tma_granu_elems
    sub_stride = smem_tile.tma_subtile_stride_elems
    if cutlass.const_expr(first < 0 or count < 1 or first + count > num_iters):
        raise ValueError(f"tma_load_subtiles: subtiles [{first}, {first + count}) outside the tile's {num_iters}")
    if cutlass.const_expr(gmem_slice.desc_ptr is not None):
        tma_desc_ptr = gmem_slice.desc_ptr
        if cutlass.const_expr(acquire):
            nvvm.fence_proxy_acquire(
                nvvm.MemScope.GPU,
                tma_desc_ptr,
                128,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    else:
        tma_desc_ptr = gmem_slice.tma_desc.get_ptr()
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    for j in cutlass.range_constexpr(count):
        i = first + j
        d = coord_d + cutlass.Int32(i * granu_elems)
        smem_chunk = smem_tile.base.subview(i * sub_stride)
        if nvvm.elect_sync():
            coords = [d] + list(outer_coords)
            if cutlass.const_expr(cta_group == 1):
                nvvm.cp_async_bulk_tensor_shared_cta_global(
                    smem_chunk,
                    tma_desc_ptr,
                    coords,
                    mbar,
                    l2_cache_hint=l2_cache_hint,
                )
            else:
                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                    smem_chunk,
                    tma_desc_ptr,
                    coords,
                    mbar,
                    [],
                    multicast_mask=mcast_mask,
                    group=nvvm.CTAGroup.CTA_2,
                    l2_cache_hint=l2_cache_hint,
                )


@cute.jit
def tma_store_tile(smem_tile, gmem_slice, *, acquire: cutlass.Constexpr[bool] = True):
    num_iters = smem_tile.tma_loads_per_tile
    granu_elems = smem_tile.tma_granu_elems
    sub_stride = smem_tile.tma_subtile_stride_elems
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    if cutlass.const_expr(gmem_slice.desc_ptr is not None):
        tma_desc_ptr = gmem_slice.desc_ptr
        if cutlass.const_expr(acquire):
            nvvm.fence_proxy_acquire(
                nvvm.MemScope.GPU,
                tma_desc_ptr,
                128,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    else:
        tma_desc_ptr = gmem_slice.tma_desc.get_ptr()
    for i in cutlass.range_constexpr(num_iters):
        d = coord_d + cutlass.Int32(i * granu_elems)
        smem_chunk = smem_tile.base.subview(i * sub_stride)
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            tma_desc_ptr,
            smem_chunk,
            tuple([d] + list(outer_coords)),
        )


@cute.jit
def tma_store_subtile(smem_tile, gmem_slice, subtile: cutlass.Constexpr[int], *, acquire: cutlass.Constexpr[bool] = False):
    """Issue ONE subtile of a TMA tile store -- subtile ``subtile`` of the ``smem_tile.tma_loads_per_tile`` that
    :func:`tma_store_tile` issues back to back -- so a producer can STREAM a tile's store behind the epilogue that fills
    it: wait the mbarrier that publishes SMEM chunk ``c``, store subtile ``c``, and move on while the later chunks are
    still being written, instead of parking on every chunk and then queueing the whole tile as one burst in the SM's
    in-order TMA engine (ahead of the next work item's operand loads).  Same op, same operands as the whole-tile store:
    subtile ``i`` of ``tma_store_tile(smem_tile, gmem_slice)`` == ``tma_store_subtile(smem_tile, gmem_slice, i)`` -- the
    GMEM inner coordinate is ``coord_d + i * tma_granu_elems`` and the SMEM source ``base + i * tma_subtile_stride_elems``.

    Gotchas the whole-tile store hides from its caller:

    * ``tma_subtile_stride_elems`` and ``tma_granu_elems`` are in ELEMENTS OF THE ALLOC DTYPE of ``smem_tile`` (the
      ``SmemTile``'s dtype), not of the tensor the descriptor stores.  A buffer allocated as one dtype and viewed as a
      narrower one (an aliased Q u O slab written as fp8 O) needs ``BPE_load / BPE_alloc`` folded into the stride when the
      tile is declared -- the SAME rule as ``tma_store_tile`` / ``tma_load_tile`` (frost-tile-dsl.md s5); the subtile form
      makes a stride error visible as ONE garbled subtile, not a garbled second half.
    * ``acquire`` is the runtime-descriptor (THD) GENERIC->TENSORMAP fence: ``tma_store_tile`` issues it ONCE before its
      subtile loop, so a streamed tile passes ``acquire=True`` on its FIRST subtile only (default False); a
      ``gmem_slice`` without ``desc_ptr`` (a compile-time ``GmemTileTma``) ignores it.
    * No elect inside, exactly like ``tma_store_tile``: the caller is one warp (``UTMASTG`` is a warp-uniform op).
    * Completion rides the bulk group: after the LAST subtile of the tile, ``tma_store_commit()`` +
      ``tma_store_wait(0)`` ONCE -- ``cp.async.bulk.commit_group`` batches every prior uncommitted bulk op of the thread,
      so N streamed subtiles form ONE bulk group per tile, the same group count as the whole-tile store.
    * The producer side is unchanged: the SMEM chunk must be published to the async proxy (``fence_proxy`` + the mbarrier
      arrive the caller waits) BEFORE its subtile is issued -- the streamed form only moves each subtile's issue point
      from "after the last chunk" to "after ITS chunk".

    Shipped by the sm107 d512 prefill kernels (``O_STORE_STREAM``): MEASURED there 2026-09-23, mxfp8 bf16-O dense B=1 H_Q=64
    H_KV=1 S_Q=16K, +16.3 / +16.5 / +5.9 / +0.5 % of time at S_KV = 512 / 1024 / 2048 / 8192 on a 212-SM Rubin part,
    O / LSE / Amax_O bitwise identical -- the store's TMA-ENGINE occupancy, not its HBM bytes, is the per-work-item cost
    (deleting the store: +11.2 % @8K; redirecting it to one L2-resident tile: -0.1 %)."""
    granu_elems = smem_tile.tma_granu_elems
    sub_stride = smem_tile.tma_subtile_stride_elems
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    if cutlass.const_expr(gmem_slice.desc_ptr is not None):
        tma_desc_ptr = gmem_slice.desc_ptr
        if cutlass.const_expr(acquire):
            nvvm.fence_proxy_acquire(
                nvvm.MemScope.GPU,
                tma_desc_ptr,
                128,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    else:
        tma_desc_ptr = gmem_slice.tma_desc.get_ptr()
    d = coord_d + cutlass.Int32(subtile * granu_elems)
    smem_chunk = smem_tile.base.subview(subtile * sub_stride)
    nvvm.cp_async_bulk_tensor_global_shared_cta(
        tma_desc_ptr,
        smem_chunk,
        tuple([d] + list(outer_coords)),
    )


@cute.jit
def bulk_copy(smem_dst, gmem_src, n_bytes, mbar):
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_shared_cluster_global(
            smem_dst,
            gmem_src,
            mbar,
            n_bytes,
        )


@cute.jit
def bulk_copy_multicast(smem_dst, gmem_src, n_bytes, mbar, mcast_mask):
    if nvvm.elect_sync():
        cluster_dst = nvvm.mapa(smem_dst, cutlass.Int32(0), addrspace=7)
        nvvm.cp_async_bulk_shared_cluster_global(
            cluster_dst,
            gmem_src,
            mbar,
            n_bytes,
            multicast_mask=mcast_mask,
        )


@cute.jit
def tma_store_commit():
    nvvm.cp_async_bulk_commit_group()


@cute.jit
def tma_store_wait(num_remaining: int = 0):
    nvvm.cp_async_bulk_wait_group(num_remaining, read=True)


@cute.jit
def cp_async_commit():
    nvvm.cp_async_commit_group()


@cute.jit
def cp_async_wait(num_remaining: cutlass.Constexpr[int] = 0):
    nvvm.cp_async_wait_group(num_remaining)


@cute.jit
def load_tile(
    smem_dst,
    gmem_src,
    total_elems: cutlass.Constexpr[int],
    tidx,
    *,
    num_threads: cutlass.Constexpr[int],
    elems_per_copy: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
    cache: nvvm.LoadCacheModifier = nvvm.LoadCacheModifier.CG,
):
    bytes_per_copy = elems_per_copy * elem_bytes
    if cutlass.const_expr(bytes_per_copy not in (4, 8, 16)):
        raise ValueError(f"load_tile: elems_per_copy*elem_bytes must be 4/8/16, got " f"{elems_per_copy}*{elem_bytes}={bytes_per_copy}")
    chunk_elems = num_threads * elems_per_copy
    if cutlass.const_expr(total_elems % chunk_elems != 0):
        raise ValueError(
            f"load_tile: total_elems ({total_elems}) must be a multiple of " f"num_threads*elems_per_copy ({num_threads}*{elems_per_copy}=" f"{chunk_elems})"
        )
    n_iters = total_elems // chunk_elems
    base_off = tidx * elems_per_copy
    for i in cutlass.range_constexpr(n_iters):
        off = base_off + i * chunk_elems
        nvvm.cp_async_shared_global(
            _advance(smem_dst, off),
            _advance(gmem_src, off),
            bytes_per_copy,
            cache,
        )


@cute.jit
def load_tile_2d(
    smem_dst,
    gmem_src,
    rows: cutlass.Constexpr[int],
    elems_per_row: cutlass.Constexpr[int],
    gmem_row_stride_elems,
    tidx,
    *,
    num_threads: cutlass.Constexpr[int],
    elems_per_copy: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
    cache: nvvm.LoadCacheModifier = nvvm.LoadCacheModifier.CG,
    swizzle: cutlass.Constexpr[bool] = False,
    cp_size_bytes=None,
    valid_rows=None,
    valid_cols=None,
    row_base=None,
    col_base=None,
):
    bytes_per_copy = elems_per_copy * elem_bytes
    if cutlass.const_expr(bytes_per_copy not in (4, 8, 16)):
        raise ValueError(f"load_tile_2d: elems_per_copy*elem_bytes must be 4/8/16, got " f"{elems_per_copy}*{elem_bytes}={bytes_per_copy}")
    if cutlass.const_expr(elems_per_row % elems_per_copy != 0):
        raise ValueError(f"load_tile_2d: elems_per_row ({elems_per_row}) must be a multiple " f"of elems_per_copy ({elems_per_copy})")
    chunks_per_row = elems_per_row // elems_per_copy
    total_chunks = rows * chunks_per_row
    if cutlass.const_expr(total_chunks % num_threads != 0):
        raise ValueError(f"load_tile_2d: total_chunks ({rows}*{chunks_per_row}={total_chunks}) " f"must be a multiple of num_threads ({num_threads})")
    if cutlass.const_expr(cp_size_bytes is None):
        cp_size_bytes = bytes_per_copy
    predicate_active = cutlass.const_expr((valid_rows is not None) or (valid_cols is not None))
    n_iters = total_chunks // num_threads
    for i in cutlass.range_constexpr(n_iters):
        chunk_idx = i * num_threads + tidx
        row = chunk_idx // chunks_per_row
        col_elem = (chunk_idx % chunks_per_row) * elems_per_copy
        src = _advance(gmem_src, row * gmem_row_stride_elems + col_elem)
        if cutlass.const_expr(swizzle):
            smem_col = swizzle_xor_128b(row, col_elem, elem_bytes=elem_bytes)
        else:
            smem_col = col_elem
        dst = smem_dst.subview(row * elems_per_row + smem_col)
        if cutlass.const_expr(predicate_active):
            pred = cutlass.Int32(1)
            if cutlass.const_expr(valid_rows is not None):
                row_abs = row if row_base is None else row + row_base
                pred = pred * cutlass.Int32(row_abs < valid_rows)
            if cutlass.const_expr(valid_cols is not None):
                col_abs_end = col_elem + cutlass.Int32(elems_per_copy) if col_base is None else col_elem + col_base + cutlass.Int32(elems_per_copy)
                pred = pred * cutlass.Int32(col_abs_end <= valid_cols)
            cp_size_final = cp_size_bytes * pred
        else:
            cp_size_final = cp_size_bytes
        nvvm.cp_async_shared_global(dst, src, bytes_per_copy, cache, cp_size=cp_size_final)


@cute.jit
def tma_tensormap_acquire(desc_ptr):
    """Issue a single ``fence.proxy.tensormap::generic.acquire.gpu`` over a
    runtime TMA descriptor.

    A runtime descriptor written on the host by a descriptor-builder kernel
    (via ``tensormap_replace``) is visible to the TMA proxy only after this
    GENERIC->TENSORMAP acquire.  Such descriptors are built once (not rewritten
    per work-tile), so a single acquire per consumer CTA (or once per
    persistent-loop tile) suffices; call this once and pass ``acquire=False`` to
    the per-tile ``tma_load_tile`` / ``tma_store_tile`` wrappers to skip the
    redundant per-call fences.
    """
    nvvm.fence_proxy_acquire(
        nvvm.MemScope.GPU,
        desc_ptr,
        128,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )


def ptx_type_suffix(dtype) -> str:
    """PTX type suffix for a 32-bit global ld/st: ``f32`` for floats, ``b32`` for bit patterns."""
    return "f32" if dtype == cutlass.Float32 else "b32"


def ld_global(addr, dtype):
    """32-bit global load: one register of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.global.{ptx_type_suffix(dtype)} $0, [$1];",
        write_only_types=[dtype],
        read_only_args=[addr],
    )


def ld_global_v2(addr, dtype):
    """64-bit global load: two 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.global.v2.{ptx_type_suffix(dtype)} {{$0, $1}}, [$2];",
        write_only_types=[dtype] * 2,
        read_only_args=[addr],
    )


def ld_global_v4(addr, dtype):
    """128-bit global load: four 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.global.v4.{ptx_type_suffix(dtype)} {{$0, $1, $2, $3}}, [$4];",
        write_only_types=[dtype] * 4,
        read_only_args=[addr],
    )


def st_global(addr, value, dtype):
    """32-bit global store: one register of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.global.{ptx_type_suffix(dtype)} [$0], $1;",
        read_only_args=[addr, value],
    )


def st_global_v2(addr, values, dtype):
    """64-bit global store: two 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.global.v2.{ptx_type_suffix(dtype)} [$0], {{$1, $2}};",
        read_only_args=[addr, values[0], values[1]],
    )


def st_global_v4(addr, values, dtype):
    """128-bit global store: four 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.global.v4.{ptx_type_suffix(dtype)} [$0], {{$1, $2, $3, $4}};",
        read_only_args=[addr, values[0], values[1], values[2], values[3]],
    )


def ld_shared_v2(addr, dtype):
    """64-bit shared load: two 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.shared.v2.{ptx_type_suffix(dtype)} {{$0, $1}}, [$2];",
        write_only_types=[dtype] * 2,
        read_only_args=[addr],
    )


def ld_shared_v4(addr, dtype):
    """128-bit shared load: four 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.shared.v4.{ptx_type_suffix(dtype)} {{$0, $1, $2, $3}}, [$4];",
        write_only_types=[dtype] * 4,
        read_only_args=[addr],
    )


def st_shared_v2(addr, values, dtype):
    """64-bit shared store: two 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.shared.v2.{ptx_type_suffix(dtype)} [$0], {{$1, $2}};",
        read_only_args=[addr, values[0], values[1]],
    )


def st_shared_v4(addr, values, dtype):
    """128-bit shared store: four 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.shared.v4.{ptx_type_suffix(dtype)} [$0], {{$1, $2, $3, $4}};",
        read_only_args=[addr, values[0], values[1], values[2], values[3]],
    )


# ---------------------------------------------------------------------------
# TMA gather4: four indirectly-addressed rows of a 2-D tensor map per issue.
# ---------------------------------------------------------------------------
#
# `cp.async.bulk.tensor.2d...tile::gather4` copies four ROWS of a 2-D tensor map
# (one box-row of `box_dims[1]` contiguous elements each) into four consecutive
# box-rows of SMEM.  It is the row gather every index-list sparse-attention
# kernel needs: K/V rows named by a block / token list, landed in the SAME
# 128-B-swizzled layout a tiled box load would produce.  The swizzle is a
# function of the SMEM address bits (bits [4:7) of the 16-B chunk index XOR
# bits [7:10) of the 128-B row index inside a 1024-B span), so a quad at a
# 512-B-aligned offset inside a 1024-B-aligned sub-box is byte-identical to rows
# 4g..4g+3 of a tiled load of the same sub-box.  The MMA descriptors that read
# the tile do not change.
#
# Why raw ``llvm.inline_asm`` and not ``nvvm.inline_ptx``: the ``L2::cache_hint``
# operand must be a 64-bit REGISTER, and the ``nvvm.inline_ptx`` lowering gives a
# constant operand the ``n`` immediate constraint (frost-tile-dsl.md section 7;
# the ``tcgen05.commit`` mask twin in barrier.py).  The constraint string below
# (``r,l,r,r,r,r,r,r,l``) is the one the in-tree DSA bridge runs on SM100
# (``deepseek_sparse_attention/sparse_attention_forward/_tma_gather4.py``); the
# hint additionally goes through an opaque ``mov.b64`` so it can never fold.
# This is the third spelling of the instruction in the tree (the DSA bridge
# above, the MoE grouped GEMM's ``gemm/frost/tile_helpers.tma_gather4`` -- no
# L2 hint, a multicast arm -- and the pre-upstream d512 index-list fork); a
# fourth belongs here, not in a kernel.
#
# Why not the public wrappers: ``nvvm.cp_async_bulk_tensor_shared_cta_global(
# mode=TILE_GATHER4)`` asserts exactly TWO coordinates (``nvvm_wrapper.py``
# ``_assert_coords``: "gather4/scatter4 mode requires exactly 2 coordinate
# elements") and offers no way to hand it the four row coordinates, and the
# ``CopyBulkTensor2DGather4G2SOp`` atom route needs a ``gmem_coord_tensor`` +
# ``tma_partition`` + ``cute.copy`` per issue on top of a compiler-owned atom --
# neither can be driven from a FROST ``TensorMap`` pointer with per-lane
# register coordinates, which is what a gather warp holds.
#
# ``cta_group=2`` routes the transaction bytes to the 2-SM PAIR LEADER's mbarrier
# by clearing bit 24 of the local mbar address (the LSB of ``%cluster_ctarank``;
# what ``nvvm.cp_async_bulk_tensor_shared_cluster_global(group="cta_2")`` does
# for the tiled loads this kernel family already issues).  The DESTINATION stays
# CTA-local (``shared::cta``): each CTA gathers its own rows, one elected lane on
# the leader arms ``expect_tx`` for BOTH CTAs' bytes.  Under a (4,1,1) cluster the
# even-rank CTA of each pair is the leader (P9).
#
# Issue model: the instruction lowers to ``UTMALDG`` on the uniform datapath, so
# a per-lane operand makes ptxas emit an ``R2UR`` + one ``UTMALDG`` per active
# lane -- issue from ONE elected lane, in an explicit loop, with the four row
# coordinates broadcast to it (``cute.arch.shuffle_sync``) BEFORE the elect
# branch (a shuffle inside a divergent branch is undefined).

#: TMA ``CacheHintSm90::EVICT_LAST`` encoding (the DSA bridge's constant).
TMA_L2_EVICT_LAST = 0x14F0000000000000

_GATHER4_TEMPLATE = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
    "mbarrier::complete_tx::bytes.cta_group::{cta_group}.L2::cache_hint "
    "[$0], [$1, {{$2, $3, $4, $5, $6}}], [$7], $8;"
)
_GATHER4_CONSTRAINTS = "r,l,r,r,r,r,r,r,l"
# int32 spelling of 0xFEFFFFFF: clears bit 24 only (the 2-SM pair-rank bit).
_PAIR_LEADER_MBAR_MASK = -0x1000001


# The HOST half of the Rule-7 gate, ``tma_gather4_requirement_error``, lives in ``tile_dsl/requirements.py`` -- a module
# with NO cutlass import, so an adapter's ``check_support`` can import it on a DSL below the floor, where THIS module's
# ``cutlass.experimental`` import (line 5) fails before any function here could run -- and is re-exported above for
# callers that already hold the DSL.


def _require_gather4_dsl():
    """Trace-time half of the Rule-7 gate: raise (naming the version) before any atom import -- the host half is
    :func:`tile_dsl.requirements.tma_gather4_requirement_error`."""
    msg = tma_gather4_requirement_error()
    if msg is not None:
        raise RuntimeError(msg)


def _smem_addr_i32(x):
    """32-bit shared-window address of an SMEM ``cutlass.Array`` / ``Pointer`` / ``Int32``."""
    if hasattr(x, "data_ptr") and not hasattr(x, "toint"):
        x = x.data_ptr()
    if hasattr(x, "toint"):
        x = x.toint()
    return cutlass.Int32(x)


def _desc_addr_i64(x):
    """64-bit generic address of a ``TensorMap`` / its ``Pointer`` / an ``Int64``."""
    if hasattr(x, "get_ptr"):
        x = x.get_ptr()
    if hasattr(x, "toint"):
        x = x.toint()
    return cutlass.Int64(x)


def opaque_i64(value: int):
    """A 64-bit immediate materialised in a REGISTER the optimizer cannot re-fold
    (``mov.b64``) -- the Int64 twin of ``pointwise.opaque_f32_zero``, for asm
    operands whose constraint demands a register (the TMA L2 cache-hint: a folded
    constant would reach the ``l`` operand as an immediate ptxas rejects)."""
    return nvvm.inline_ptx(f"mov.b64 $0, 0x{value & 0xFFFFFFFFFFFFFFFF:016X};", write_only_types=[cutlass.Int64])


def tma_gather4(desc, dst_smem, mbar, col, r0, r1, r2, r3, *, cta_group: int = 1, l2_hint=None):
    """Issue ONE ``cp.async.bulk.tensor.2d ... tile::gather4`` (four rows x one box-row of columns).

    ``desc``     -- the 2-D ``TensorMap`` (or its ``get_ptr()`` / ``Int64`` address); box ``(1, cols)``
                    in the tensor's mode order, i.e. one row of ``cols`` contiguous elements.  Under
                    ``SWIZZLE_128B`` the box-row is at most 128 B (``cols = 64`` bf16), so a 256-wide
                    bf16 row takes four issues at ``col = 0, 64, 128, 192``.
    ``dst_smem`` -- SMEM destination of the FIRST gathered row (``Array``/``Pointer``/``Int32``);
                    rows ``r1..r3`` land at ``+1, +2, +3`` box-rows (128 B each under SWIZZLE_128B).
                    ALIGNMENT RULE: keep every quad at a 512-B-aligned offset inside a 1024-B-aligned
                    swizzle sub-box (``box * sub_box_bytes + quad * 512``), so the swizzle the hardware
                    applies from the address bits is the one a TILED load of that sub-box applies to its
                    rows ``4 quad .. 4 quad + 3`` -- the gathered tile then equals ``kv[ids]`` loaded
                    tiled, bitwise, and the MMA descriptors over it are the dense kernel's.  A quad at a
                    128-B- or 256-B-aligned offset lands rows in the wrong XOR pattern: finite, plausible,
                    wrong, and no test on the raw bytes sees it -- compare through a tiled store.
    ``mbar``     -- the LOCAL ``mbarrier`` object; under ``cta_group=2`` bit 24 is cleared so the
                    ``complete_tx`` credits the 2-SM pair leader's copy (only the leader arms ``expect_tx``).
    ``col``      -- the contiguous (innermost) coordinate, in elements.
    ``r0..r3``   -- the four row coordinates (signed).  A ``-1`` (or any out-of-range) row is TMA-OOB:
                    the box-row is ZERO-FILLED and its bytes ARE STILL COUNTED on the mbarrier -- so a
                    block id of ``-1`` spelled as rows ``-4..-1`` yields four zero rows, an ``expect_tx``
                    is ALWAYS the full box (``4 * cols * BPE`` per issue, never the in-range row count:
                    an arm for the in-range bytes only hangs), and OOB rows cost fabric bytes like real
                    ones (zero-fill included).
    ``l2_hint``  -- an ``Int64`` cache policy; ``None`` = ``EVICT_LAST`` via :func:`opaque_i64` PER CALL (one
                    ``mov.b64`` per issue in the PTX: 16 per 16-issue warp body of the roundtrip probe).  A gather
                    loop hoists it -- ``hint = opaque_i64(TMA_L2_EVICT_LAST)`` once per tile, ``l2_hint=hint`` per
                    issue -- the way the tiled loads of this family already do.

    One elected lane per CTA issues; the caller elects (this is a trace-time macro, like
    ``tma_load_tile``'s inner op).  Bytes per issue = ``4 * cols * BPE``.  Gate: a DSL below the
    library floor raises here, naming the version, before the inline-asm atom is imported
    (the host half is :func:`tile_dsl.requirements.tma_gather4_requirement_error`, importable below the
    floor; AGENTS.md Rule 7).
    """
    if cutlass.const_expr(cta_group not in (1, 2)):
        raise ValueError(f"tma_gather4: cta_group must be 1 or 2, got {cta_group}")
    _require_gather4_dsl()
    from cutlass._mlir.dialects import llvm

    if not hasattr(llvm, "inline_asm"):
        raise RuntimeError("tma_gather4: this cutlass-dsl exposes no llvm.inline_asm atom; the gather4 instruction cannot be emitted -- upgrade the DSL")

    mbar_i32 = _smem_addr_i32(mbar)
    if cutlass.const_expr(cta_group == 2):
        mbar_i32 = mbar_i32 & cutlass.Int32(_PAIR_LEADER_MBAR_MASK)
    hint = opaque_i64(TMA_L2_EVICT_LAST) if l2_hint is None else cutlass.Int64(l2_hint)
    llvm.inline_asm(
        None,
        [
            _smem_addr_i32(dst_smem).ir_value(),
            _desc_addr_i64(desc).ir_value(),
            cutlass.Int32(col).ir_value(),
            cutlass.Int32(r0).ir_value(),
            cutlass.Int32(r1).ir_value(),
            cutlass.Int32(r2).ir_value(),
            cutlass.Int32(r3).ir_value(),
            mbar_i32.ir_value(),
            hint.ir_value(),
        ],
        _GATHER4_TEMPLATE.format(cta_group=cta_group),
        _GATHER4_CONSTRAINTS,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def ldg_int32x4(addr):
    """One 16-B read-only global load: four ``Int32`` from the 16-B-ALIGNED byte address ``addr``
    (``Int64`` or a generic ``Pointer``).  ``ld.global.nc`` -- the index / membership feeds are
    read-only for the kernel's lifetime.  The per-lane spelling of the DSA kernel's
    ``_ldg_indices_128``: lanes ``0..15`` each fetch their quad's four row ids BEFORE the ring wait,
    then shuffle them to the elected issuing lane (shuffles outside the elect branch).  An id row whose
    pitch is a multiple of 16 B keeps every quad's load aligned -- a ``top_k`` that is a multiple of 4
    int32 is that rule."""
    if hasattr(addr, "toint"):
        addr = addr.toint()
    return nvvm.inline_ptx(
        "ld.global.nc.v4.s32 {$0, $1, $2, $3}, [$4];",
        write_only_types=[cutlass.Int32] * 4,
        read_only_args=[cutlass.Int64(addr)],
    )
