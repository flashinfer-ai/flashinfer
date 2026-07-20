# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/io.py @ e130f195 (2026-07-17) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Sparse-MLA decode IO-warp PRODUCER (P6 warp-specialized pipeline).

The generic SM120 decode CTA uses 8 math warps plus one IO warp; native GLM H8
uses 4 math warps plus one IO warp. This module is the producer body: per chunk
it waits on the ``empty`` mbarrier for the target KV buffer, gathers the BI=64
candidate rows from the paged FP8 KV cache into the double-buffered smem KV
regions via ``cp.async.bulk``, gathers DSV4 UE8M0 footer scales when present,
and signals ``full`` via
``mbarrier_arrive_and_expect_tx`` so the bulk completion drives the transaction
count to zero.

Protocol (raw mbarrier ops; modeled EXACTLY on FlashInfer
``decode_dsv4_kernel.cuh`` :243-323 ``issue_gather`` + the ``is_io`` producer
loop, and ``common/kv_cache_io.cuh`` ``io_bulk_gather_tile``/``io_gather_scales``):

  * ``mbar_full[s]``  : IO leader (io_lane 0) arrives + expect_tx(BULK_TX_BYTES);
                        the cp.async.bulk completions decrement the transaction
                        count. The full barrier flips its phase only when BOTH
                        the arrival (1, by the leader) AND tx==0 are met -- i.e.
                        when every byte of buffer ``s`` has landed.
  * ``mbar_empty[s]`` : the math consumer arrives (one math thread) when it is
                        done reading buffer ``s``; the IO warp waits on it
                        before overwriting ``s``.

The IO warp NEVER touches a math-only named barrier (``bar.sync 3, 256``) --
that would deadlock (256-count waiter that the IO warp can't satisfy and
shouldn't join). It only touches mbarriers + its own loads.

FOOTER caveat (matches FlashInfer): the 7+1 UE8M0 footer bytes per token live
in gmem in a BLOCK-STRUCTURED (FlashMLA footer ABI) region -- the BI tokens are
NOT contiguous (different ``local_idx`` / ``block_idx``), so there is no single
16B-aligned ``BI*8`` gmem region to cp.async.bulk. Per-token 8B cp.async.bulk
is also illegal (must be 16B-aligned). So the footer is gathered with SCALAR
``ld.global.nc`` (one v2.u32 = the 8 footer bytes/token) into the CONTIGUOUS
smem ``kv_sc`` buffer, then a CTA-scope fence (``fence_acq_rel_cta`` ==
``__threadfence_block``) orders those scalar stores before the leader's
expect_tx -- exactly as ``issue_gather`` does (:268-285). The grouped footer
thus lands in smem as a contiguous BI*8=512B region; the math reads it FLAT.
The bulk transaction count (``BULK_TX_BYTES``) covers ONLY the nope+rope data
(BI*(448+128)); the footer is NOT part of the mbarrier tx.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from flashinfer.experimental.sm12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    get_ptr_as_int64,
    ld_global_nc_v2_u32,
    shared_ptr_to_u32,
    st_shared_u32,
)

# DSV4 KV gmem IO stride: DATA portion only (448 nope + 64 rope * 2B = 576),
# 16B aligned for cp.async.bulk. == FlashInfer KVIOTraits<DSV4>::IO_STRIDE.
_DSV4_IO_STRIDE = 576
_DSV4_NOPE_BYTES = 448  # FP8 nope; == D_NOPE; bulk #1 per entry.
_DSV4_ROPE_BYTES = 128  # BF16 rope; == D_ROPE * 2; bulk #2 per entry.
_DSV4_FOOTER_BYTES = 8  # 7 UE8M0 + 1 pad; == SCALE_BYTES_PER_TOKEN.

# GLM (ARBITRARY_FP32) KV gmem layout: per-token 656B contiguous record
# (reference.pack_mla_kv_cache_reference): 512 e4m3 nope + 16 inline fp32 scales
# (4 groups x 4B) + 128 bf16 rope. There is NO grouped footer -- the inline
# scales travel WITH the nope (bulk #1), so the nope+scales region (528B) and the
# rope (128B) are both 16B-aligned cp.async.bulk copies. == GLM_KV_GMEM_STRIDE.
_GLM_GMEM_STRIDE = 656
_GLM_NOPE_SCALE_BYTES = 528  # 512 e4m3 + 16 inline fp32; bulk #1 (-> kv_fp8 row).
_GLM_ROPE_BYTES = 128  # 64 bf16; bulk #2 (-> kv_rope).

# NVFP4 MLA latent layout: 256B E2M1 NoPE + 32B E4M3 scales + 16B pad + 128B
# BF16 RoPE. The NoPE+scale+pad region is staged as a 288B row; decode math
# dequants the E2M1 data and E4M3 group-16 scales in registers.
_NVFP4_GMEM_STRIDE = 432
_NVFP4_NOPE_SCALE_BYTES = 288
_NVFP4_ROPE_BYTES = 128
_NVFP4_ROPE_SRC = 304
# GLM-only KV_FP8_ROPE tail.  The latent bytes [0,288) are identical to the
# stock record; [288,304) holds fp32 scale + 12B zero pad and [304,368) is E4M3.
_NVFP4_FP8_ROPE_GMEM_STRIDE = 368
_NVFP4_FP8_ROPE_TAIL_BYTES = 80
_NVFP4_FP8_ROPE_TAIL_SRC = 288

# IO warp width (one warp = 32 threads; FlashInfer DSV4_IO_THREADS).
_IO_THREADS = 32


@cute.jit
def io_issue_gather(
    kv_cache_u8: cute.Tensor,  # flat 1-D u8 view of the paged DSV4 KV cache
    topk_indices: cute.Tensor,  # 1-D int32 topk slice for this query token
    kv_fp8_dst_addr: Int32,  # u32 smem addr of kv_fp8[buf] (BI x KV_SMEM_STRIDE)
    kv_rope_dst_addr: Int32,  # u32 smem addr of kv_rope[buf] (BI x D_ROPE bf16)
    kv_sc_dst_addr: Int32,  # u32 smem addr of kv_sc[buf] (BI x 8 footer)
    token_idx_view: cute.Tensor,  # smem int32 validity buffer (BI,) for THIS buf
    full_mbar_ptr,  # cute.Pointer (u64) of mbar_full[buf]
    g_start: Int32,  # absolute candidate offset of entry 0 (chunk_in_section*CAND_WINDOW)
    g_end: Int32,  # min(g_start + CAND_WINDOW, section_len)
    page_block_size: Int32,  # pbs: tokens per paged block (THIS section)
    stride_kv_block: Int64,  # per-block byte stride in gmem (THIS section)
    io_lane: Int32,  # lane within the IO warp [0, 32)
    *,
    bi: cutlass.Constexpr,  # 64
    kv_smem_stride: cutlass.Constexpr,  # 464 DSV4 / 528 GLM (smem nope row stride)
    rope_smem_stride: cutlass.Constexpr,  # 64 (D_ROPE bf16 elems)
    scale_bytes_per_token: cutlass.Constexpr,  # 8 (DSV4 footer); unused for GLM
    bulk_tx_bytes: cutlass.Constexpr,  # BI*(448+128)=36864 DSV4 / BI*(528+128)=41984 GLM
    scale_format: cutlass.Constexpr = 0,  # UE8M0_BYTE (0) / ARBITRARY_FP32 (1)
    fp8_rope: cutlass.Constexpr = False,
    io_threads: cutlass.Constexpr = _IO_THREADS,  # 32 (decode 1 IO warp) / 128 (prefill 4 IO warps)
    packed_glm: cutlass.Constexpr = False,
    packed_dsv4: cutlass.Constexpr = False,
    split_mbar_arrival: cutlass.Constexpr = False,
    overlap_footer_gather: cutlass.Constexpr = False,
):
    """Producer body for ONE chunk into buffer ``buf`` (caller selects the dst
    addrs + full_mbar_ptr for ``buf``). Mirrors FlashInfer ``issue_gather``:

    ``io_threads`` (const_expr) is the number of IO threads sharing this gather
    (32 for the decode single IO warp, 128 for the prefill 4 IO warps). The entry
    stride + unroll count both fold to ``io_threads`` so the BI=64 entries are
    distributed across the available IO threads (prefill: 1 pass of 64 over 128
    threads; decode: 2 passes of 32). ``io_lane`` is the lane within the whole IO
    group ([0, io_threads)). Defaulting to 32 keeps the decode PTX byte-identical.

      1. SCALAR footer gather (32 IO threads stride over BI entries; v2.u32 read
         of the 8 footer bytes, clamped-invalid -> 0) into the contiguous smem
         ``kv_sc`` buffer; in the SAME pass stage the raw topk index (incl the
         -1 sentinel) into the ``token_idx`` validity buffer (gap #9 single
         source of truth for the S3 consumer mask).
      2. CTA-scope fence so those footer/index stores are visible before the
         leader's expect_tx (mbarrier.try_wait.parity has NO implicit fence).
      3. IO leader (io_lane 0) arrive + expect_tx(BULK_TX_BYTES) on full[buf].
      4. cp.async.bulk each entry; DSV4 issues separate NoPE (448B) and RoPE
         (128B) copies, while packed GLM H8 issues one contiguous 656B copy.
         Bulk completion decrements the full[buf] transaction count.

    DUAL-CACHE (DSV4 only; FlashInfer ``issue_gather`` :243-306): this function
    gathers ONE chunk from ONE section. The dual-cache section dispatch lives in
    the KERNEL IO loop (launch.py): for a MAIN chunk it calls this with the main
    cache / indices / page_block_size / stride_kv_block; for an EXTRA chunk it
    calls this with the extra cache / extra indices / pbs_extra /
    stride_extra_kv_block. The smem dst layout + per-entry byte geometry are
    IDENTICAL across sections, so this body is section-agnostic and the no-extra
    DSV4 / GLM PTX is unchanged (the caller never emits the extra branch).
    """
    # Section-agnostic gather: ``kv_cache_u8`` / ``topk_indices`` /
    # ``page_block_size`` / ``stride_kv_block`` are THIS section's pool (the caller
    # selects main vs extra). Aliased to the historical ``_section_*`` names used
    # by the per-entry addressing below.
    _section_kv = kv_cache_u8
    _section_idx = topk_indices
    _section_pbs = page_block_size
    _section_stride = stride_kv_block

    # Per-model gmem geometry (const_expr). DSV4: 576B data + grouped 8B footer;
    # GLM: 656B contiguous (528 nope+inline-scales + 128 rope), NO footer.
    # NVFP4: 432B contiguous (288 nope+E4M3-scales+pad + 128 rope), NO footer.
    if cutlass.const_expr(scale_format == 0):
        _IOS = Int64(_DSV4_IO_STRIDE)  # 576 per-token data stride
        _NOPE = Int32(_DSV4_NOPE_BYTES)  # 448 -> kv_fp8 (e4m3 nope)
        _ROPE = Int32(_DSV4_ROPE_BYTES)  # 128 -> kv_rope
        _ROPE_SRC = Int64(_DSV4_NOPE_BYTES)  # rope follows nope in the record
    elif cutlass.const_expr(scale_format == 2):
        _NOPE = Int32(_NVFP4_NOPE_SCALE_BYTES)
        if cutlass.const_expr(fp8_rope):
            _IOS = Int64(_NVFP4_FP8_ROPE_GMEM_STRIDE)
            # Stage scale+pad+FP8 payload contiguously in the existing 128-byte
            # BF16-rope smem row. decode_math interprets scale at +0 and E4M3 at +16.
            _ROPE = Int32(_NVFP4_FP8_ROPE_TAIL_BYTES)
            _ROPE_SRC = Int64(_NVFP4_FP8_ROPE_TAIL_SRC)
        else:
            _IOS = Int64(_NVFP4_GMEM_STRIDE)
            _ROPE = Int32(_NVFP4_ROPE_BYTES)
            _ROPE_SRC = Int64(_NVFP4_ROPE_SRC)
    else:
        _IOS = Int64(_GLM_GMEM_STRIDE)  # 656 per-token contiguous record
        _NOPE = Int32(_GLM_NOPE_SCALE_BYTES)  # 528 nope+inline-fp32 -> kv_fp8
        _ROPE = Int32(_GLM_ROPE_BYTES)  # 128 -> kv_rope
        _ROPE_SRC = Int64(_GLM_NOPE_SCALE_BYTES)  # rope follows nope+scales
    _FOOT = Int32(scale_bytes_per_token)

    def _issue_payload_entry(entry: Int32, idx_raw: Int32):
        full_mbar_u32 = shared_ptr_to_u32(full_mbar_ptr)
        idx = idx_raw
        if idx < Int32(0):
            idx = Int32(0)
        block_idx = idx // _section_pbs
        local_idx = idx - block_idx * _section_pbs
        data_base_off = Int64(block_idx) * _section_stride + Int64(local_idx) * _IOS
        data_base_i64 = get_ptr_as_int64(_section_kv, data_base_off)

        if cutlass.const_expr(packed_glm):
            cp_async_bulk_g2s_mbar(
                kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                data_base_i64,
                Int32(_GLM_GMEM_STRIDE),
                full_mbar_u32,
            )
        elif cutlass.const_expr(packed_dsv4):
            cp_async_bulk_g2s_mbar(
                kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                data_base_i64,
                Int32(_DSV4_IO_STRIDE),
                full_mbar_u32,
            )
        else:
            cp_async_bulk_g2s_mbar(
                kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                data_base_i64,
                _NOPE,
                full_mbar_u32,
            )
            cp_async_bulk_g2s_mbar(
                kv_rope_dst_addr + entry * Int32(rope_smem_stride * 2),
                data_base_i64 + _ROPE_SRC,
                _ROPE,
                full_mbar_u32,
            )

    def _issue_payload():
        eo = Int32(0)
        for _ in cutlass.range_constexpr((bi + io_threads - 1) // io_threads):
            entry = eo + io_lane
            if entry < Int32(bi):
                cand_pos = g_start + entry
                idx_raw = Int32(-1)
                if cand_pos < g_end:
                    idx_raw = Int32(_section_idx[cand_pos])
                idx = idx_raw
                if idx < Int32(0):
                    idx = Int32(0)
                block_idx = idx // _section_pbs
                local_idx = idx - block_idx * _section_pbs
                data_base_off = (
                    Int64(block_idx) * _section_stride + Int64(local_idx) * _IOS
                )
                data_base_i64 = get_ptr_as_int64(_section_kv, data_base_off)
                full_mbar_u32 = shared_ptr_to_u32(full_mbar_ptr)

                if cutlass.const_expr(packed_glm):
                    cp_async_bulk_g2s_mbar(
                        kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                        data_base_i64,
                        Int32(_GLM_GMEM_STRIDE),
                        full_mbar_u32,
                    )
                elif cutlass.const_expr(packed_dsv4):
                    cp_async_bulk_g2s_mbar(
                        kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                        data_base_i64,
                        Int32(_DSV4_IO_STRIDE),
                        full_mbar_u32,
                    )
                else:
                    cp_async_bulk_g2s_mbar(
                        kv_fp8_dst_addr + entry * Int32(kv_smem_stride),
                        data_base_i64,
                        _NOPE,
                        full_mbar_u32,
                    )
                    cp_async_bulk_g2s_mbar(
                        kv_rope_dst_addr + entry * Int32(rope_smem_stride * 2),
                        data_base_i64 + _ROPE_SRC,
                        _ROPE,
                        full_mbar_u32,
                    )
            eo += Int32(io_threads)

    # Native H16 can overlap the random scalar footer load with the much larger
    # payload transfer. Its producer leaders set the transaction count without
    # arriving; their post-footer arrivals remain the release condition.
    if cutlass.const_expr(overlap_footer_gather):
        if (io_lane & Int32(31)) == Int32(0):
            cute.arch.mbarrier_expect_tx(full_mbar_ptr, Int32(bulk_tx_bytes // 2))
        # H16 has exactly one row per IO thread. Start the random footer load,
        # launch the independent payload copy, and only then consume the footer
        # result in shared stores so the two memory operations overlap.
        overlap_entry = io_lane
        overlap_cand_pos = g_start + overlap_entry
        overlap_idx_raw = Int32(-1)
        if overlap_cand_pos < g_end:
            overlap_idx_raw = Int32(_section_idx[overlap_cand_pos])
        token_idx_view[overlap_entry] = overlap_idx_raw

        overlap_f0 = Uint32(0)
        overlap_f1 = Uint32(0)
        if overlap_idx_raw >= Int32(0):
            overlap_block_idx = overlap_idx_raw // _section_pbs
            overlap_local_idx = overlap_idx_raw - overlap_block_idx * _section_pbs
            overlap_scale_base_off = (
                Int64(overlap_block_idx) * _section_stride
                + Int64(_section_pbs) * _IOS
                + Int64(overlap_local_idx) * Int64(_FOOT)
            )
            overlap_f0, overlap_f1 = ld_global_nc_v2_u32(
                get_ptr_as_int64(_section_kv, overlap_scale_base_off)
            )

        _issue_payload_entry(overlap_entry, overlap_idx_raw)
        overlap_s_byte = overlap_entry * _FOOT
        st_shared_u32(kv_sc_dst_addr + overlap_s_byte, overlap_f0)
        st_shared_u32(kv_sc_dst_addr + overlap_s_byte + Int32(4), overlap_f1)
        cute.arch.fence_acq_rel_cta()
        if (io_lane & Int32(31)) == Int32(0):
            cute.arch.mbarrier_arrive(full_mbar_ptr)
        return

    # --- (1) per-entry validity index staging + (DSV4 only) scalar footer gather. ---
    eo = Int32(0)
    for _ in cutlass.range_constexpr((bi + io_threads - 1) // io_threads):
        entry = eo + io_lane
        if entry < Int32(bi):
            cand_pos = g_start + entry
            idx_raw = Int32(-1)
            if cand_pos < g_end:
                idx_raw = Int32(_section_idx[cand_pos])
            # gap #9: stage the raw index (incl -1) for the S3 consumer mask.
            token_idx_view[entry] = idx_raw
            if cutlass.const_expr(scale_format == 0):
                # DSV4 grouped UE8M0 footer -> contiguous smem kv_sc. GLM has no
                # footer (inline scales travel in the kv_fp8 nope bulk).
                f0 = Uint32(0)
                f1 = Uint32(0)
                if idx_raw >= Int32(0):
                    block_idx = idx_raw // _section_pbs
                    local_idx = idx_raw - block_idx * _section_pbs
                    scale_base_off = (
                        Int64(block_idx) * _section_stride
                        + Int64(_section_pbs) * _IOS
                        + Int64(local_idx) * Int64(_FOOT)
                    )
                    f0, f1 = ld_global_nc_v2_u32(
                        get_ptr_as_int64(_section_kv, scale_base_off)
                    )
                s_byte = entry * _FOOT
                st_shared_u32(kv_sc_dst_addr + s_byte, f0)
                st_shared_u32(kv_sc_dst_addr + s_byte + Int32(4), f1)
        eo += Int32(io_threads)

    # --- (2) CTA-scope fence: footer/index stores visible before release. ---
    # == FlashInfer __threadfence_block() (issue_gather :281). try_wait.parity
    # has no implicit memory fence so this acq-rel is load-bearing.
    cute.arch.fence_acq_rel_cta()

    # --- (3) Publish the footer stores, or start the conventional payload. ---
    if cutlass.const_expr(overlap_footer_gather):
        if (io_lane & Int32(31)) == Int32(0):
            cute.arch.mbarrier_arrive(full_mbar_ptr)
    else:
        if cutlass.const_expr(split_mbar_arrival):
            # Native H16 has two producer warps, each responsible for 32 of the
            # 64 packed rows. Each leader contributes half the transaction bytes.
            if (io_lane & Int32(31)) == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_mbar_ptr, Int32(bulk_tx_bytes // 2)
                )
        else:
            if io_lane == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_mbar_ptr, Int32(bulk_tx_bytes)
                )
        _issue_payload()
