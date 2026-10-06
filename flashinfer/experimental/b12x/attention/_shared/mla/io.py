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
from cutlass import Boolean, Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    cvt_e4m3_to_f32_via_f16,
    get_ptr_as_int64,
    ld_global_nc_v4_u32,
    ld_global_nc_v2_u32,
    pack_f32x2_to_f16x2,
    pow2_ceil_ue8m0,
    shared_ptr_to_u32,
    st_shared_u32,
    ue8m0_to_output_scale,
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
_GLM_NEXT_GMEM_STRIDE = 528  # Same latent+scales row, with no RoPE suffix.

# NVFP4 MLA latent layout: 256B E2M1 NoPE + 32B E4M3 scales + 16B pad + 128B
# BF16 RoPE. The NoPE+scale+pad region is staged as a 288B row; decode math
# dequants the E2M1 data and E4M3 group-16 scales in registers.
_NVFP4_GMEM_STRIDE = 432
_GLM_NEXT_NVFP4_GMEM_STRIDE = 304
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
def stage_dsv41_fp8_scales(
    source: cute.Tensor,
    record_offset: Int64,
    ratio_addr: Int32,
    scales_addr: Int32,
    valid: Boolean,
    *,
    swa: cutlass.Constexpr,
) -> None:
    """Stage DSV4.1's canonical FP8 metadata before publishing a KV stage.

    ``ratio_addr`` names the 64-byte FP16 sidecar at row byte 544 and
    ``scales_addr`` names its eight canonical UE8M0 bytes in ``kv_sc``.  Invalid
    entries deliberately issue no global load and publish a zero conversion
    sidecar with unit (byte 127) canonical scales.
    """
    if valid:
        if cutlass.const_expr(swa):
            # The 16 source bytes are two UE8M0 scales per 64-dimensional group.
            # A single aligned vector load avoids scalar metadata traffic.
            m0, m1, m2, m3 = ld_global_nc_v4_u32(
                get_ptr_as_int64(source, record_offset + Int64(512))
            )
            metadata = [m0, m1, m2, m3]
            scale_word_lo = Uint32(0)
            scale_word_hi = Uint32(0)
            for group in cutlass.range_constexpr(8):
                packed = metadata[group // 2]
                pair = (packed >> Uint32((group & 1) * 16)) & Uint32(0xFFFF)
                lo = pair & Uint32(0xFF)
                hi = pair >> Uint32(8)
                canonical = cutlass.max(cutlass.max(lo, hi), Uint32(1))
                ratio_lo = cutlass.Float32(0.0)
                ratio_hi = cutlass.Float32(0.0)
                # UE8M0 exponents make source/canonical an exact power of two;
                # avoid a general floating-point divide on the producer path.
                if canonical - lo <= Uint32(18):
                    ratio_lo = ue8m0_to_output_scale(
                        Uint32(127) + canonical - lo
                    )
                if canonical - hi <= Uint32(18):
                    ratio_hi = ue8m0_to_output_scale(
                        Uint32(127) + canonical - hi
                    )
                ratio_pair = pack_f32x2_to_f16x2(ratio_lo, ratio_lo)
                st_shared_u32(ratio_addr + Int32(group * 8), ratio_pair)
                ratio_pair = pack_f32x2_to_f16x2(ratio_hi, ratio_hi)
                st_shared_u32(ratio_addr + Int32(group * 8 + 4), ratio_pair)
                # The consumer's four original 16-wide groups consist of the
                # two native 32-wide scale regions, duplicated into each half.
                if group < 4:
                    scale_word_lo |= canonical << Uint32(group * 8)
                else:
                    scale_word_hi |= canonical << Uint32((group - 4) * 8)
            st_shared_u32(scales_addr, scale_word_lo)
            st_shared_u32(scales_addr + Int32(4), scale_word_hi)
        else:
            # Indexed records have 32 E4M3 scales at byte 256: four native
            # 16-wide scales per output 64-wide group.
            m0, m1, m2, m3 = ld_global_nc_v4_u32(
                get_ptr_as_int64(source, record_offset + Int64(256))
            )
            m4, m5, m6, m7 = ld_global_nc_v4_u32(
                get_ptr_as_int64(source, record_offset + Int64(272))
            )
            metadata = [m0, m1, m2, m3, m4, m5, m6, m7]
            scale_word_lo = Uint32(0)
            scale_word_hi = Uint32(0)
            for group in cutlass.range_constexpr(8):
                packed = metadata[group]
                s0 = cvt_e4m3_to_f32_via_f16(packed & Uint32(0xFF))
                s1 = cvt_e4m3_to_f32_via_f16((packed >> Uint32(8)) & Uint32(0xFF))
                s2 = cvt_e4m3_to_f32_via_f16((packed >> Uint32(16)) & Uint32(0xFF))
                s3 = cvt_e4m3_to_f32_via_f16(packed >> Uint32(24))
                # Metadata is a positive scale format; treat zero/non-positive
                # encodings as zero when deriving the canonical containment bound.
                s0 = cutlass.max(s0, cutlass.Float32(0.0))
                s1 = cutlass.max(s1, cutlass.Float32(0.0))
                s2 = cutlass.max(s2, cutlass.Float32(0.0))
                s3 = cutlass.max(s3, cutlass.Float32(0.0))
                largest = cutlass.max(cutlass.max(s0, s1), cutlass.max(s2, s3))
                _, canonical = pow2_ceil_ue8m0(
                    cutlass.max(
                        largest * cutlass.Float32(6.0 / 448.0),
                        cutlass.Float32(2.0**-126),
                    )
                )
                inv_canonical = ue8m0_to_output_scale(canonical)
                st_shared_u32(
                    ratio_addr + Int32(group * 8),
                    pack_f32x2_to_f16x2(s0 * inv_canonical, s1 * inv_canonical),
                )
                st_shared_u32(
                    ratio_addr + Int32(group * 8 + 4),
                    pack_f32x2_to_f16x2(s2 * inv_canonical, s3 * inv_canonical),
                )
                if group < 4:
                    scale_word_lo |= canonical << Uint32(group * 8)
                else:
                    scale_word_hi |= canonical << Uint32((group - 4) * 8)
            st_shared_u32(scales_addr, scale_word_lo)
            st_shared_u32(scales_addr + Int32(4), scale_word_hi)
    else:
        # Invalid tags must not read their clamped row-0 source.  Zero sidecar
        # ratios convert every payload lane to zero; byte 127 denotes unit scale.
        for group in cutlass.range_constexpr(8):
            st_shared_u32(ratio_addr + Int32(group * 8), Uint32(0))
            st_shared_u32(ratio_addr + Int32(group * 8 + 4), Uint32(0))
        st_shared_u32(scales_addr, Uint32(0x7F7F7F7F))
        st_shared_u32(scales_addr + Int32(4), Uint32(0x7F7F7F7F))


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
    per_token_latent_scale: cutlass.Constexpr = False,
    dsv41: cutlass.Constexpr = False,
    dsv41_swa: cutlass.Constexpr = True,
    dsv41_fp8: cutlass.Constexpr = False,
    ratio_base_addr: Int32 = Int32(0),
    ratio_stride: cutlass.Constexpr = 0,
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
    if cutlass.const_expr(dsv41):
        _IOS = Int64(528 if dsv41_swa else 288)
        _NOPE = Int32(528 if dsv41_swa else 288)
        _ROPE = Int32(0)
        _ROPE_SRC = Int64(0)
    elif cutlass.const_expr(scale_format == 0):
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
        elif cutlass.const_expr(rope_smem_stride):
            _IOS = Int64(_NVFP4_GMEM_STRIDE)
            _ROPE = Int32(_NVFP4_ROPE_BYTES)
            _ROPE_SRC = Int64(_NVFP4_ROPE_SRC)
        else:
            _IOS = Int64(_GLM_NEXT_NVFP4_GMEM_STRIDE)
            _ROPE = Int32(0)
            _ROPE_SRC = Int64(_GLM_NEXT_NVFP4_GMEM_STRIDE)
    else:
        _IOS = Int64(_GLM_GMEM_STRIDE if rope_smem_stride else _GLM_NEXT_GMEM_STRIDE)
        _NOPE = Int32(_GLM_NOPE_SCALE_BYTES)  # 528 nope+inline-fp32 -> kv_fp8
        _ROPE = Int32(_GLM_ROPE_BYTES if rope_smem_stride else 0)
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
            if cutlass.const_expr(rope_smem_stride > 0):
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
                    if cutlass.const_expr(rope_smem_stride > 0):
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
            if cutlass.const_expr(dsv41):
                source_tag = Uint32(1 if dsv41_swa else 0)
                if idx_raw < Int32(0):
                    source_tag = Uint32(2)
                st_shared_u32(
                    kv_fp8_dst_addr + entry * Int32(kv_smem_stride) + Int32(528),
                    source_tag,
                )
            if cutlass.const_expr(dsv41_fp8):
                # DSV4.1 records are truly 528B (SWA) or 288B (indexed), not
                # the padded shared-row stride.  Keep every pool product Int64.
                metadata_idx = idx_raw
                if metadata_idx < Int32(0):
                    metadata_idx = Int32(0)
                metadata_block = metadata_idx // _section_pbs
                metadata_local = metadata_idx - metadata_block * _section_pbs
                metadata_record_offset = (
                    Int64(metadata_block) * _section_stride
                    + Int64(metadata_local)
                    * Int64(528 if dsv41_swa else 288)
                )
                ratio_addr = kv_fp8_dst_addr + entry * Int32(kv_smem_stride) + Int32(544)
                if cutlass.const_expr(ratio_stride):
                    ratio_addr = ratio_base_addr + entry * Int32(ratio_stride)
                stage_dsv41_fp8_scales(
                    _section_kv,
                    metadata_record_offset,
                    ratio_addr,
                    kv_sc_dst_addr + entry * Int32(8),
                    idx_raw >= Int32(0),
                    swa=dsv41_swa,
                )
            elif cutlass.const_expr(scale_format == 0):
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
            elif cutlass.const_expr(scale_format == 2 and per_token_latent_scale):
                # NVFP4 two-level record: the per-token fp32 latent scale sits
                # at [292, 296).  292 is not 8B-aligned, so load the 8-aligned
                # pair at [288, 296) (rope scale, latent scale) and keep the
                # second word -- same nc.v2 pattern as the DSV4 footer.
                f1 = Uint32(0)
                if idx_raw >= Int32(0):
                    block_idx = idx_raw // _section_pbs
                    local_idx = idx_raw - block_idx * _section_pbs
                    scale_base_off = (
                        Int64(block_idx) * _section_stride
                        + Int64(local_idx) * _IOS
                        + Int64(_NVFP4_FP8_ROPE_TAIL_SRC)
                    )
                    _, f1 = ld_global_nc_v2_u32(
                        get_ptr_as_int64(_section_kv, scale_base_off)
                    )
                st_shared_u32(kv_sc_dst_addr + entry * Int32(4), f1)
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
                    full_mbar_ptr,
                    Int32(bi // 2) * _NOPE
                    if cutlass.const_expr(dsv41)
                    else Int32(bulk_tx_bytes // 2),
                )
        else:
            if io_lane == Int32(0):
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full_mbar_ptr,
                    Int32(bi) * _NOPE if cutlass.const_expr(dsv41) else Int32(bulk_tx_bytes),
                )
        _issue_payload()


@cute.jit
def io_issue_packed_payload(
    kv_cache_u8: cute.Tensor,
    kv_fp8_dst_addr: Int32,
    full_mbar_ptr,
    idx0: Int32,
    idx1: Int32,
    page_block_size: Int32,
    stride_kv_block: Int64,
    io_lane: Int32,
    *,
    kv_smem_stride: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    bulk_tx_bytes: cutlass.Constexpr,
):
    """Register the chunk's transaction bytes and issue its two record copies.

    Entry ``io_lane`` copies candidate ``idx0`` and entry ``io_lane + 32``
    copies ``idx1``; a negative index copies row 0 so the byte count is
    always met. Callable before the chunk's validity is known: the copies
    only fill the shared stage, and the phase cannot complete before the
    producer's later arrival.
    """
    full_mbar_u32 = shared_ptr_to_u32(full_mbar_ptr)
    if io_lane == Int32(0):
        cute.arch.mbarrier_expect_tx(full_mbar_ptr, Int32(bulk_tx_bytes))
    clamped0 = idx0
    if clamped0 < Int32(0):
        clamped0 = Int32(0)
    clamped1 = idx1
    if clamped1 < Int32(0):
        clamped1 = Int32(0)
    blk0 = clamped0 // page_block_size
    loc0 = clamped0 - blk0 * page_block_size
    blk1 = clamped1 // page_block_size
    loc1 = clamped1 - blk1 * page_block_size
    data0 = Int64(blk0) * stride_kv_block + Int64(loc0) * Int64(record_bytes)
    data1 = Int64(blk1) * stride_kv_block + Int64(loc1) * Int64(record_bytes)
    cp_async_bulk_g2s_mbar(
        kv_fp8_dst_addr + io_lane * Int32(kv_smem_stride),
        get_ptr_as_int64(kv_cache_u8, data0),
        Int32(record_bytes),
        full_mbar_u32,
    )
    cp_async_bulk_g2s_mbar(
        kv_fp8_dst_addr + (io_lane + Int32(32)) * Int32(kv_smem_stride),
        get_ptr_as_int64(kv_cache_u8, data1),
        Int32(record_bytes),
        full_mbar_u32,
    )


@cute.jit
def io_issue_gather_packed(
    kv_cache_u8: cute.Tensor,  # flat 1-D u8 view of THIS section's paged cache
    topk_indices: cute.Tensor,  # 1-D int32 index row for this token and section
    kv_fp8_dst_addr: Int32,  # u32 smem addr of the packed KV stage for this buffer
    kv_sc_dst_addr: Int32,  # u32 smem addr of the DSV4 footer stage (unused for GLM)
    token_idx_view: cute.Tensor,  # smem int32 validity buffer (BI,) for this buffer
    full_mbar_ptr,  # cute.Pointer (u64) of mbar_full[buf]
    g_start: Int32,  # absolute candidate offset of entry 0 within the section
    g_end: Int32,  # min(g_start + BI, section_len)
    page_block_size: Int32,  # tokens per paged block in this section
    stride_kv_block: Int64,  # per-block byte stride in this section
    io_lane: Int32,  # lane within the single IO warp [0, 32)
    pre_valid: Int32,  # 1: pre_idx0/pre_idx1 hold the raw indices; 2: payload issued too
    pre_idx0: Int32,
    pre_idx1: Int32,
    *,
    kv_smem_stride: cutlass.Constexpr,  # 656 GLM / 592 DSV4 packed smem row
    record_bytes: cutlass.Constexpr,  # 656 GLM / 576 DSV4 contiguous data record
    bulk_tx_bytes: cutlass.Constexpr,  # BI * record_bytes
    index_capacity: cutlass.Constexpr,  # entries in topk_indices
    dsv4_footer: cutlass.Constexpr,  # gather the 8-byte UE8M0 footer per token
    scale_bytes_per_token: cutlass.Constexpr = 8,
):
    """Producer body for one chunk of contiguous-record decode paths.

    Serves the eight-head GLM and DSV4 arms and the generic GLM_NEXT arm; all
    stage one contiguous record per candidate. One IO warp gathers BI=64
    records, two per lane. The order is
    chosen for latency: the raw indices are consumed first (either preloaded
    by the caller before the kernel's length-dependent prologue or loaded
    here), the bulk copies are issued immediately, and only then are the
    DSV4 footer scalar loads and the validity-index stores performed, so the
    footer round trip overlaps the payload transfer instead of preceding it.
    The transaction count is registered before the copies with a bare
    expect-tx; the single arrival that releases the phase follows the CTA
    fence that publishes the footer and index stores.

    Every entry copies exactly ``record_bytes`` (an invalid entry copies row 0)
    so the transaction count always reaches ``bulk_tx_bytes``; validity is
    carried by ``token_idx_view`` alone.
    """
    entry0 = io_lane
    entry1 = io_lane + Int32(32)
    cand0 = g_start + entry0
    cand1 = g_start + entry1
    idx0 = Int32(-1)
    idx1 = Int32(-1)
    if pre_valid != Int32(0):
        idx0 = pre_idx0
        idx1 = pre_idx1
    else:
        if cand0 < Int32(index_capacity):
            idx0 = Int32(topk_indices[cand0])
        if cand1 < Int32(index_capacity):
            idx1 = Int32(topk_indices[cand1])
    if cand0 >= g_end:
        idx0 = Int32(-1)
    if cand1 >= g_end:
        idx1 = Int32(-1)

    if pre_valid != Int32(2):
        io_issue_packed_payload(
            kv_cache_u8,
            kv_fp8_dst_addr,
            full_mbar_ptr,
            idx0,
            idx1,
            page_block_size,
            stride_kv_block,
            io_lane,
            kv_smem_stride=kv_smem_stride,
            record_bytes=record_bytes,
            bulk_tx_bytes=bulk_tx_bytes,
        )

    clamped0 = idx0
    if clamped0 < Int32(0):
        clamped0 = Int32(0)
    clamped1 = idx1
    if clamped1 < Int32(0):
        clamped1 = Int32(0)
    blk0 = clamped0 // page_block_size
    loc0 = clamped0 - blk0 * page_block_size
    blk1 = clamped1 // page_block_size
    loc1 = clamped1 - blk1 * page_block_size

    if cutlass.const_expr(dsv4_footer):
        f00 = Uint32(0)
        f01 = Uint32(0)
        f10 = Uint32(0)
        f11 = Uint32(0)
        footer_base = Int64(page_block_size) * Int64(record_bytes)
        if idx0 >= Int32(0):
            f00, f01 = ld_global_nc_v2_u32(
                get_ptr_as_int64(
                    kv_cache_u8,
                    Int64(blk0) * stride_kv_block
                    + footer_base
                    + Int64(loc0) * Int64(scale_bytes_per_token),
                )
            )
        if idx1 >= Int32(0):
            f10, f11 = ld_global_nc_v2_u32(
                get_ptr_as_int64(
                    kv_cache_u8,
                    Int64(blk1) * stride_kv_block
                    + footer_base
                    + Int64(loc1) * Int64(scale_bytes_per_token),
                )
            )
        s0 = entry0 * Int32(scale_bytes_per_token)
        s1 = entry1 * Int32(scale_bytes_per_token)
        st_shared_u32(kv_sc_dst_addr + s0, f00)
        st_shared_u32(kv_sc_dst_addr + s0 + Int32(4), f01)
        st_shared_u32(kv_sc_dst_addr + s1, f10)
        st_shared_u32(kv_sc_dst_addr + s1 + Int32(4), f11)

    token_idx_view[entry0] = idx0
    token_idx_view[entry1] = idx1
    cute.arch.fence_acq_rel_cta()
    if io_lane == Int32(0):
        cute.arch.mbarrier_arrive(full_mbar_ptr)
