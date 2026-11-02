# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import write_thd_work_prefix
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.swizzle import swizzle_xor
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import (
    TENSOR_MAP_QWORDS,
    THD_CTR_OFF,
    THD_LIVE_OFF,
    THD_META_WORDS,
    THD_REMAP_OFF,
    THD_SETUP_THREADS,
    THD_BATCH_OF_WORK_META_WORDS,
    THD_WORK_META_WORDS,
    THD_WORK_PREFIX_OFF,
    emit_clamped_desc,
    emit_seq_descs,
    thd_claim_next,
    thd_decode_unit,
    thd_decode_unit_prefix,
    write_thd_batch_remap,
    write_thd_live_and_ctr,
    write_thd_meta,
    write_thd_prefix_warp,
    write_thd_small_batch_meta,
)

# The THD metadata layout, the batch ranking, the unit decode, the claim counter
# and the per-sequence descriptor patchers moved to ``frost/tile_dsl/thd.py``
# when the SM100 backward needed the same pieces -- a second copy of a metadata
# LAYOUT is a silent-wrong-answer waiting to happen, since a reader offset and a
# writer offset that disagree by one word decode garbage batches rather than
# failing.  Re-exported here so the kernels importing them from this module keep
# working.  What stays below is genuinely forward-specific: the SM120 sV tail
# sanitizer (its SMEM order and swizzle are that kernel's), and the per-flavor
# setup LAUNCHES.
__all__ = [
    "TENSOR_MAP_QWORDS",
    "THD_CTR_OFF",
    "THD_LIVE_OFF",
    "THD_META_WORDS",
    "THD_REMAP_OFF",
    "THD_SETUP_THREADS",
    "THD_BATCH_OF_WORK_META_WORDS",
    "THD_WORK_META_WORDS",
    "THD_WORK_PREFIX_OFF",
    "build_thd_meta_kernel",
    "build_thd_meta_o_descs_kernel",
    "build_thd_meta_o_kv_descs_kernel",
    "sanitize_v_tail",
    "thd_claim_next",
    "thd_decode_unit",
    "thd_decode_unit_prefix",
    "write_thd_batch_remap",
    "write_thd_live_and_ctr",
    "write_thd_meta",
]


@cute.jit
def sanitize_v_tail(
    sV,
    lane: cutlass.Int32,
    seqlen_k: cutlass.Int32,
    kv_seq_idx: cutlass.Int32,
    in_dtype,
    head_tile_v,
    kv_tile,
    v_swizzle_chunk_elems,
) -> None:
    """Zero ``sV`` rows at/past this tile's valid KV extent before P @ V.

    Shared by the SM120 f16 and FP8 flavors (their sV tiles use the same
    ``[I][kv_tile][C]`` TMA order and per-row swizzle; the element size rides
    ``in_dtype``). The K/V TMA descriptors span the bound buffers' CAPACITY
    (under THD the packed totals are device values, so the views bind
    whole-buffer extents), so rows between the valid KV length and the tile
    end can carry UNINITIALIZED storage — including NaN bit patterns. The
    S-side mask overwrites those columns with -inf (a select, NaN-safe),
    but P @ V still multiplies P = 0 against the NaN V row and
    ``0 * NaN = NaN`` poisons the whole accumulator column-free. Reached
    only from THD specializations' first masked step (see the call
    sites): dense descriptors carry the declared S_kv, so their overhang
    loads zero-fill in hardware — a dense PADDED graph's pad rows are
    user memory and deliberately NOT sanitized here (whether the
    contract requires tolerating NaN bit patterns there is an open
    question for the sibling kernels too).

    Every compute warp redundantly zeroes the full overhang (idempotent
    zero stores race benignly), so a warp-level sync is enough for each
    warp's own ``ldmatrix`` lanes to observe the zeros.
    """
    seg_elems = 16 // in_dtype.bytes  # elements per 16-byte store
    segs_per_row = head_tile_v // seg_elems
    row_lo = cute.math.max(cutlass.Int32(0), seqlen_k - kv_seq_idx)
    for r_it in cutlass.range_constexpr(kv_tile // 32):
        row = cutlass.Int32(r_it * 32) + lane
        for seg in cutlass.range_constexpr(segs_per_row):
            col = seg * seg_elems
            phys_row = (col // v_swizzle_chunk_elems) * kv_tile + row
            sv_ptr = (
                sV.data_ptr()
                + phys_row * v_swizzle_chunk_elems
                + swizzle_xor(
                    phys_row,
                    col % v_swizzle_chunk_elems,
                    v_swizzle_chunk_elems,
                    in_dtype.bytes,
                )
            )
            zero16 = cutlass.Vector.from_elements(
                tuple(in_dtype(0.0) for _ in range(seg_elems)),
                in_dtype,
            )
            if row >= row_lo:
                sv_ptr.store(zero16, alignment=16)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.kernel
def build_thd_meta_kernel(
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_batch: cutlass.Int32,
    n_qh: cutlass.Int32,
    q_tile: cutlass.Int32,
    n_ctas: cutlass.Int32,
    work_prefix: cutlass.Constexpr[bool] = False,
) -> None:
    """Meta-only THD setup (SM120: no per-batch O TMA descriptors — O stores
    are raw pointer writes predicated per row). Two warps build Q/KV prefixes
    for batched inputs; B <= 1 stays serial. The remap and live-unit total are
    whole-block. The main kernel launched after it on the same stream sees the
    writes by kernel boundary ordering.

    ``work_prefix``: also publish the remap-ordered unit prefix
    (``THD_WORK_PREFIX_OFF``, ``B + 1`` words past the 4B+4 layout) for a
    consumer decoding through :func:`thd_decode_unit_prefix`; the buffer must
    then hold ``THD_WORK_META_WORDS(B)`` words. Off by default: the sm107
    sparse path sizes its metadata at ``THD_META_WORDS(B)``."""
    meta = cutlass.make_array_view(meta_t)
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    if n_batch <= cutlass.Int32(1):
        if nvvm.elect_sync() and tidx < cutlass.Int32(32):
            write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
    else:
        warp = cutlass.Int32(tidx) // cutlass.Int32(32)
        lane = cutlass.Int32(tidx) % cutlass.Int32(32)
        if warp == cutlass.Int32(0):
            write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens_t), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
        if warp == cutlass.Int32(1):
            write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens_t), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
    # Barrier first: the ranking reads the cu_seqlens_q written above.
    cute.arch.barrier()
    write_thd_batch_remap(meta, n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    # Live unit total + claim counter, as on SM100 — a SM120 unit is q_tile
    # rows of one head, so the same count applies with cga_tile_m := q_tile.
    cute.arch.barrier()
    if cutlass.const_expr(work_prefix):
        write_thd_work_prefix(meta, n_batch, n_qh, q_tile, n_ctas, cutlass.Int32(tidx), 1)
    else:
        write_thd_live_and_ctr(meta, n_batch, n_qh, q_tile, n_ctas, cutlass.Int32(tidx))


build_thd_meta_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def build_thd_meta_o_kv_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    o_row_stride: cutlass.Int64,
    cga_tile_m: cutlass.Int32,
    n_clusters: cutlass.Int32,
    k_seq_dim: cutlass.Constexpr[int] = 2,
    v_seq_dim: cutlass.Constexpr[int] = 2,
    splits: cutlass.Constexpr[int] = 1,
) -> None:
    """THD setup for the FP8/MXFP8 flavors: metadata, per-batch O descriptors
    and the packed-total-clamped K/V descriptors.

    A single request keeps the elected-thread path. Larger batches use two
    warps for Q/KV prefixes and distribute descriptors across warp leaders.
    Callers must launch at least two full warps.

    Like ``build_thd_meta_o_descs_kernel``, this publishes the persistent
    scheduler's live-unit total and claim counter, and the remap-ordered unit
    prefix the shared decode binary-searches (``THD_WORK_PREFIX_OFF``; the
    metadata buffer holds ``THD_WORK_META_WORDS(B)`` words). Both kernels also
    clamp K/V (issue #624).

    The K/V loads tile in TILE_N rows, so the LAST sequence's tile steps past
    the packed KV total into the buffer's capacity tail — caller-owned bytes
    that may be NaN (test_mhas_v2 poisons them deliberately). Masked S columns
    are NaN-safe (the mask is a select), but BMM2's ``P·V`` is not
    (``0 · NaN == NaN`` wipes every valid row of the tile), and on cc10.3 the
    fused-LDTM row-max reduces S BEFORE the mask. So the setup kernel also
    copies the K and V base descriptors into ``o_desc_words`` slots
    ``n_batch+1`` / ``n_batch+2`` with each descriptor's sequence extent
    patched to the packed total ``cu_k[B]`` — tile-tail loads past it become
    TMA-OOB and land as EXACT ZEROS in SMEM (zero V nulls the masked P·V
    terms; zero K keeps the pre-mask row-max finite). Slot ``n_batch`` stays
    the never-built dead-unit pad slot."""
    tidx, _, _ = cute.arch.thread_idx()
    nthreads, _, _ = cute.arch.block_dim()
    meta = cutlass.make_array_view(meta_t)
    if n_batch <= cutlass.Int32(1):
        if nvvm.elect_sync() and tidx < cutlass.Int32(32):
            write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
            emit_seq_descs(base_o_desc, o_desc_words, meta, n_batch, o_tensor, n_batch, o_row_stride, seq_ord=2)
            t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])
            emit_clamped_desc(base_k_desc, o_desc_words, n_batch + cutlass.Int32(1), t_kv, seq_ord=k_seq_dim)
            emit_clamped_desc(base_v_desc, o_desc_words, n_batch + cutlass.Int32(2), t_kv, seq_ord=v_seq_dim)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    else:
        warp = cutlass.Int32(tidx) // cutlass.Int32(32)
        lane = cutlass.Int32(tidx) % cutlass.Int32(32)
        nwarps = cutlass.Int32(nthreads) // cutlass.Int32(32)
        if warp == cutlass.Int32(0):
            write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens_t), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
        if warp == cutlass.Int32(1):
            write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens_t), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
        cute.arch.barrier()
        if nvvm.elect_sync():
            emit_seq_descs(base_o_desc, o_desc_words, meta, n_batch, o_tensor, n_batch, o_row_stride, seq_ord=2, first_batch=warp, batch_step=nwarps)
            # Capacity tails may contain NaNs: make the final partial K/V
            # tile TMA-OOB so masked P @ V cannot read 0 * NaN (#624).
            t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])
            if warp == nwarps - cutlass.Int32(2):
                emit_clamped_desc(base_k_desc, o_desc_words, n_batch + cutlass.Int32(1), t_kv, seq_ord=k_seq_dim)
            if warp == nwarps - cutlass.Int32(1):
                emit_clamped_desc(base_v_desc, o_desc_words, n_batch + cutlass.Int32(2), t_kv, seq_ord=v_seq_dim)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    # Outside the elect: every thread helps rank the batches. The barrier makes
    # the cu_seqlens_q written above visible to the whole block first. The
    # decode walks this permutation on every THD flavor, so a setup that skips
    # it leaves the region uninitialized and units decode garbage batches.
    cute.arch.barrier()
    write_thd_batch_remap(cutlass.make_array_view(meta_t), n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads))
    # Unit prefix + live unit total + claim counter for the persistent
    # scheduler, exactly as build_thd_meta_o_descs_kernel writes them. The
    # scheduler reads meta[4B+2] as its bound, so leaving these words unwritten
    # hands out units off uninitialized workspace (an illegal-instruction
    # fault, not a silent wrong answer). The counter starts at n_clusters:
    # cluster c takes unit c from its blockIdx, then pulls from the counter.
    cute.arch.barrier()
    write_thd_work_prefix(cutlass.make_array_view(meta_t), n_batch, n_qh, cga_tile_m, n_clusters, cutlass.Int32(tidx), splits)


build_thd_meta_o_kv_descs_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.kernel
def build_thd_meta_o_descs_kernel(
    o_tensor: cute.Tensor,
    base_o_desc: cutlass.GridConstant[tmap.TensorMap],
    base_k_desc: cutlass.GridConstant[tmap.TensorMap],
    base_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_desc_words: cute.Tensor,
    meta_t: cute.Tensor,
    q_lens_t: cute.Tensor,
    kv_lens_t: cute.Tensor,
    lens_form: cutlass.Int32,
    n_qh: cutlass.Int32,
    n_batch: cutlass.Int32,
    o_row_stride: cutlass.Int64,
    cga_tile_m: cutlass.Int32,
    n_clusters: cutlass.Int32,
    # False under paged KV: K/V are page pools addressed through block tables,
    # so no packed-total clamp exists (the descriptors are pool-shaped).
    clamp_kv: cutlass.Constexpr[bool] = True,
    splits: cutlass.Constexpr[int] = 1,
    rank_by_kv: cutlass.Constexpr[bool] = False,
    small_batch_meta: cutlass.Constexpr[bool] = False,
    release_dependents: cutlass.Constexpr[bool] = False,
) -> None:
    """Per-execute f16/bf16 THD metadata, O descriptors, and scheduler bounds.

    Opted-in batches of one through 32 requests build metadata in one warp.
    Other callers keep the elected-thread path for a single request. Larger
    batches use one warp per prefix. Elected leaders patch disjoint O
    descriptors. The final two warps also patch the packed K/V extents; paged
    pools retain their pool-shaped descriptors. All work stays in the existing
    setup launch, before the attention kernel on the same stream.

    ``lens_form`` bit 0/1 chooses cumulative Q/KV inputs; cumulative prefixes
    are normalized by subtracting their first entry. This preserves sliced
    prefixes without host validation or device-to-host reads.

    Forward callers may rank by KV traversal length for bottom-right attention
    without a sliding window. Top-left and windowed work retain Q ordering:
    their actual traversal is not described by the full KV length.

    The remap-ordered unit prefix (``THD_WORK_PREFIX_OFF``) is always
    published: the shared decode binary-searches it instead of walking every
    sequence, so the metadata buffer holds ``THD_WORK_META_WORDS(B)`` words.
    """
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(release_dependents):
        # This setup launch retains ordinary stream ordering with its input
        # producers. Only its consumer opts into PDL: allow that consumer's
        # independent prologue to start, then require griddepcontrol.wait
        # before it reads any metadata or runtime tensor-map descriptor.
        if tidx == 0:
            launch_dependent_grids()
    nthreads, _, _ = cute.arch.block_dim()
    meta = cutlass.make_array_view(meta_t)
    fused_meta = cutlass.Boolean(False)
    if cutlass.const_expr(small_batch_meta):
        fused_meta = (n_batch >= 1) & (n_batch <= 32)
    # Cumulative inputs already provide the prefix needed by every descriptor.
    # Keep the metadata warp independent of descriptor writers instead of
    # publishing that prefix to GMEM only to read it back after a CTA barrier.
    direct_prefix = fused_meta & ((lens_form & 1) != 0)
    direct_prefix = direct_prefix & (cutlass.Int32(nthreads) >= cutlass.Int32(128 if clamp_kv else 64))
    if cutlass.const_expr(clamp_kv):
        direct_prefix = direct_prefix & ((lens_form & 2) != 0)
    if direct_prefix:
        warp = cutlass.Int32(tidx) // cutlass.Int32(32)
        lane = cutlass.Int32(tidx) % cutlass.Int32(32)
        nwarps = cutlass.Int32(nthreads) // cutlass.Int32(32)
        if warp == 0:
            write_thd_small_batch_meta(
                meta,
                cutlass.make_array_view(q_lens_t),
                cutlass.make_array_view(kv_lens_t),
                lens_form,
                n_batch,
                n_qh,
                cga_tile_m,
                n_clusters,
                lane,
                splits,
                rank_by_kv,
            )
        else:
            if nvvm.elect_sync():
                descriptor_warps = nwarps - cutlass.Int32(3 if clamp_kv else 1)
                if warp <= descriptor_warps:
                    q_prefix = cutlass.make_array_view(q_lens_t)
                    emit_seq_descs(
                        base_o_desc,
                        o_desc_words,
                        q_prefix,
                        cutlass.Int32(0),
                        o_tensor,
                        n_batch,
                        o_row_stride,
                        seq_ord=2,
                        first_batch=warp - cutlass.Int32(1),
                        batch_step=descriptor_warps,
                        compact_origin=cutlass.Int32(q_prefix[0]),
                    )
                if cutlass.const_expr(clamp_kv):
                    kv_prefix = cutlass.make_array_view(kv_lens_t)
                    t_kv = cutlass.Int32(kv_prefix[n_batch]) - cutlass.Int32(kv_prefix[0])
                    if warp == nwarps - cutlass.Int32(2):
                        emit_clamped_desc(base_k_desc, o_desc_words, n_batch + cutlass.Int32(1), t_kv, seq_ord=2)
                    if warp == nwarps - cutlass.Int32(1):
                        emit_clamped_desc(base_v_desc, o_desc_words, n_batch + cutlass.Int32(2), t_kv, seq_ord=2)
                nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    elif (n_batch <= cutlass.Int32(1)) & ~fused_meta:
        if nvvm.elect_sync() and tidx < cutlass.Int32(32):
            write_thd_meta(meta, cutlass.make_array_view(q_lens_t), cutlass.make_array_view(kv_lens_t), lens_form, n_batch)
            emit_seq_descs(base_o_desc, o_desc_words, meta, n_batch, o_tensor, n_batch, o_row_stride, seq_ord=2)
            if cutlass.const_expr(clamp_kv):
                t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])
                emit_clamped_desc(base_k_desc, o_desc_words, n_batch + cutlass.Int32(1), t_kv, seq_ord=2)
                emit_clamped_desc(base_v_desc, o_desc_words, n_batch + cutlass.Int32(2), t_kv, seq_ord=2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    else:
        warp = cutlass.Int32(tidx) // cutlass.Int32(32)
        lane = cutlass.Int32(tidx) % cutlass.Int32(32)
        nwarps = cutlass.Int32(nthreads) // cutlass.Int32(32)
        if fused_meta:
            if warp == cutlass.Int32(0):
                write_thd_small_batch_meta(
                    meta,
                    cutlass.make_array_view(q_lens_t),
                    cutlass.make_array_view(kv_lens_t),
                    lens_form,
                    n_batch,
                    n_qh,
                    cga_tile_m,
                    n_clusters,
                    lane,
                    splits,
                    rank_by_kv,
                )
        else:
            if warp == cutlass.Int32(0):
                write_thd_prefix_warp(meta, cutlass.make_array_view(q_lens_t), n_batch, n_batch, (lens_form & 1) != 0, lane, store_lengths=False)
            if warp == cutlass.Int32(1):
                write_thd_prefix_warp(meta, cutlass.make_array_view(kv_lens_t), n_batch, 2 * n_batch + 1, (lens_form & 2) != 0, lane, store_lengths=True)
        cute.arch.barrier()
        if nvvm.elect_sync():
            emit_seq_descs(base_o_desc, o_desc_words, meta, n_batch, o_tensor, n_batch, o_row_stride, seq_ord=2, first_batch=warp, batch_step=nwarps)
            if cutlass.const_expr(clamp_kv):
                # Capacity tails may contain NaNs: make the final partial K/V
                # tile TMA-OOB so masked P @ V cannot read 0 * NaN (#624).
                t_kv = cutlass.Int32(meta[cutlass.Int32(3) * n_batch + cutlass.Int32(1)])
                if warp == nwarps - cutlass.Int32(2):
                    emit_clamped_desc(base_k_desc, o_desc_words, n_batch + cutlass.Int32(1), t_kv, seq_ord=2)
                if warp == nwarps - cutlass.Int32(1):
                    emit_clamped_desc(base_v_desc, o_desc_words, n_batch + cutlass.Int32(2), t_kv, seq_ord=2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if not fused_meta:
        cute.arch.barrier()
        write_thd_batch_remap(cutlass.make_array_view(meta_t), n_batch, cutlass.Int32(tidx), cutlass.Int32(nthreads), rank_by_kv=rank_by_kv)
        # Live unit total + claim counter for the persistent scheduler. The host
        # cannot know Sigma_b ceil(s_b/tile)*QH without a D2H (issue #552), so the
        # kernel reads its own bound from here. The counter starts at n_clusters:
        # cluster c takes unit c from its blockIdx, then pulls from the counter.
        cute.arch.barrier()
        write_thd_work_prefix(cutlass.make_array_view(meta_t), n_batch, n_qh, cga_tile_m, n_clusters, cutlass.Int32(tidx), splits)


build_thd_meta_o_descs_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)
