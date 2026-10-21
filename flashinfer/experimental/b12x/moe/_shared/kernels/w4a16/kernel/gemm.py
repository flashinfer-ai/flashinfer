"""W4A16 routed GEMM device implementation."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS
from cutlass.cutlass_dsl import Int32
from cutlass.cutlass_dsl import Int64
from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from b12x.moe._shared.trellis_codebooks import LUT_FP16
from cutlass.cutlass_dsl import Uint32
from b12x.moe._shared.kernels.trellis_ring import (
    trellis256_lane_geom_bits as _trellis_ring_lane_geom_bits,
)
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.intrinsics.mma import bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32
from b12x._lib.intrinsics.packed import bfloat2_broadcast_lane
from b12x._lib.intrinsics.packed import bfloat2_mul
from b12x._lib.intrinsics.packed import bfloat2_to_float2_scaled
from b12x._lib.quant.block_codec import block_codec
from b12x._lib.intrinsics.packed import broadcast_f32_to_bfloat2
from b12x._lib.intrinsics.packed import broadcast_f32_to_half2
from b12x._lib.intrinsics.memory import cp_async4_shared_global
from b12x._lib.intrinsics.memory import cp_async4_shared_global_pred
from b12x._lib.intrinsics.memory import cp_async_u64_shared_global
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass
from b12x._lib.intrinsics.mma import f16_mma_m16n8k16_f32
from b12x._lib.intrinsics.mma import f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32
from b12x._lib.intrinsics.packed import fp8x4_e4m3_to_bfloat2x2_native_sm120
from b12x._lib.intrinsics.packed import fp8x4_e4m3_to_half2x2
from b12x._lib.intrinsics.memory import get_ptr_as_int64
from b12x._lib.intrinsics.packed import half2_mul
from b12x._lib.intrinsics.packed import half2_to_float2_scaled
from b12x._lib.intrinsics.trellis import iq2_xxs_descriptor_pair
from b12x._lib.intrinsics.sync import ld_global_acquire_i32
from b12x._lib.intrinsics.memory import ld_global_nc_u32
from b12x._lib.intrinsics.memory import ld_global_v4_f32
from b12x._lib.intrinsics.atomics import ld_shared_f32
from b12x._lib.intrinsics.atomics import ld_shared_i32_relaxed
from b12x._lib.intrinsics.memory import ld_shared_u16_offset
from b12x._lib.intrinsics.atomics import ld_shared_u32
from b12x._lib.intrinsics.memory import ld_shared_v2_u32
from b12x._lib.intrinsics.atomics import ld_shared_v4_f32
from b12x._lib.intrinsics.memory import ld_shared_v4_u32
from b12x._lib.intrinsics.memory import ldmatrix_m8n8x2_b16
from b12x._lib.intrinsics.memory import ldmatrix_m8n8x4_b16
import os
from b12x._lib.intrinsics.packed import pack_f32x2_to_bfloat2
from b12x._lib.intrinsics.conversion import pack_f32x2_to_f16x2
from b12x._lib.intrinsics.trellis import packed_decode_iq2_xs_to_bfloat2x4
from b12x._lib.intrinsics.trellis import packed_decode_lut_e4m3_direct_to_e4m3x8
from b12x._lib.intrinsics.trellis import packed_decode_lut_e4m3_to_e4m3x8
from b12x._lib.intrinsics.trellis import packed_decode_lut_fp16_to_bfloat2x4
from b12x._lib.intrinsics.trellis import packed_decode_lut_fp16_to_half2x4
from b12x._lib.intrinsics.packed import packed_dequant_e2m1x4_to_bfloat2x2
from b12x._lib.intrinsics.packed import packed_dequant_e2m1x4_to_half2x2
from b12x._lib.intrinsics.packed import packed_dequant_e4m3x4_to_bfloat2x2
from b12x._lib.intrinsics.packed import packed_dequant_e4m3x4_to_half2x2
from b12x._lib.intrinsics.packed import packed_dequant_e8m0x4_to_bfloat2x2
from b12x._lib.intrinsics.packed import packed_dequant_e8m0x4_to_half2x2
from b12x._lib.intrinsics.trellis import packed_dequant_trellis_stream_to_bfloat2x4
from b12x._lib.intrinsics.trellis import packed_dequant_trellis_stream_to_half2x4
from b12x._lib.intrinsics.trellis import packed_dequant_trellis_to_bfloat2x4
from b12x._lib.intrinsics.trellis import packed_dequant_trellis_to_half2x4
from b12x._lib.intrinsics.trellis import q8_0_pair_to_bf16x2
from b12x._lib.intrinsics.atomics import red_add_global_bf16x2
from b12x._lib.intrinsics.atomics import red_add_global_release_i32
from b12x._lib.intrinsics.atomics import red_add_global_v4_f32
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.intrinsics.atomics import st_global_i32
from b12x._lib.intrinsics.memory import st_global_v4_f32
from b12x._lib.intrinsics.memory import st_global_v4_u32
from b12x._lib.intrinsics.atomics import st_shared_bf16_from_f32
from b12x._lib.intrinsics.atomics import st_shared_f16_from_f32
from b12x._lib.intrinsics.atomics import st_shared_f32
from b12x._lib.intrinsics.atomics import st_shared_i32
from b12x._lib.intrinsics.atomics import st_shared_u32
from b12x._lib.intrinsics.atomics import st_shared_v4_f32
from b12x._lib.intrinsics.memory import st_shared_v4_u32
import torch
from b12x._lib.intrinsics.trellis import trellis_align_stream_u32x2
from b12x.moe._shared.trellis_codebooks import validate_codebook_bits
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _ALLOWED_ROUTED_SIZES,
    _DEFAULT_MAX_SHARED_MEM,
    _E8M0_K32_BF16_GLOBAL_COMPENSATION,
    _E8M0_K32_FP16_GLOBAL_COMPENSATION,
    _FP32_TOPK_WEIGHTS,
    _MODEL_OPT_W13_LAYOUTS,
    _PACK_FACTOR,
    _SCALAR_ACC_FRAGMENT_WIDTH,
    _TRELLIS256_BITS,
    _TRELLIS256_CODEBOOKS,
    _TRELLIS256_W13_LAYOUTS,
    _WEIGHT_LAYOUTS,
    _covering_count,
    _dequant_native_fp4_word,
    _determine_blocks_per_sm,
    _device_shared_mem_per_sm,
    _e8m0_logical_tail_scale_n,
    _gather_native_scale_bytes,
    _iq2_xs_stage_bytes,
    _normalize_scale_format,
    _pack_modelopt_words,
    _reduction_shared_bytes,
    _resident_cta_shared_bytes,
    _scale_group_size,
    _w4a16_pipeline_stages,
    _w4a16_small_m_splitk_enabled,
)


class W4A16GemmKernel:
    def __init__(
        self,
        *,
        size_m: int,
        size_n: int,
        size_k: int,
        num_experts: int,
        top_k: int,
        mul_topk_weights: bool,
        tile_n: int,
        tile_k: int,
        moe_block_size: int,
        max_m_blocks: int,
        element_dtype: str = "bf16",
        epilogue_activation: str | None = None,
        weight_layout: str = "packed",
        scale_format: str = "e4m3_k16",
        w13_layout: str = "w13",
        trellis_bits: int = 3,
        trellis_codebook: str = LUT_E4M3,
        trellis_pair_kind: str | None = None,
        trellis_rate_axis: str | None = None,
        source_n_rotation: int = 0,
        single_token_route_fast_path: bool = False,
        direct_topk_routes: bool = False,
        dense_route_fast_path: bool = False,
        dual_a: bool = False,
        route_major_a: bool = False,
        fused_topk_sum: bool = False,
        fused_sum_fp32: bool = False,
        fused_sum_topk: int = 1,
        schedule_whole_tiles: bool = False,
        dynamic_num_experts: bool = False,
        schedule_route_block_factor: int = 1,
        pipeline_stages: int | None = None,
        skip_empty_m_blocks: bool | None = None,
        small_m_occupancy: int | None = None,
        csf_inline_scales: bool = False,
    ):
        if element_dtype not in {"bf16", "fp16"}:
            raise ValueError(f"unsupported element_dtype {element_dtype!r}")
        if weight_layout not in _WEIGHT_LAYOUTS:
            raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
        trellis_bits = int(trellis_bits)
        trellis_codebook = str(trellis_codebook).lower()
        scale_format = _normalize_scale_format(scale_format)
        if weight_layout == "modelopt":
            if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
                raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
        elif weight_layout == "trellis_t256":
            if w13_layout not in _TRELLIS256_W13_LAYOUTS:
                raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
        else:
            w13_layout = "packed"
            source_n_rotation = 0
        if weight_layout == "trellis_t256":
            if trellis_codebook not in _TRELLIS256_CODEBOOKS:
                raise ValueError(
                    "trellis_t256 codebook must be one of "
                    f"{sorted(_TRELLIS256_CODEBOOKS)}, got {trellis_codebook!r}"
                )
            if trellis_bits not in _TRELLIS256_BITS:
                raise ValueError(
                    "trellis_t256 bits must be one of "
                    f"{_TRELLIS256_BITS}, got {trellis_bits}"
                )
            validate_codebook_bits(trellis_codebook, trellis_bits)
            if scale_format != "e4m3_k32":
                raise ValueError(
                    "trellis_t256 W4A16 weights require scale_format='e4m3_k32'"
                )
        elif weight_layout in BLOCK_CODECS:
            if element_dtype != "bf16":
                raise ValueError("IQ2_XS W4A16 requires BF16 activations")
            if scale_format != weight_layout:
                raise ValueError("IQ2_XS W4A16 weights require scale_format='iq2_xs'")
            if (
                size_k % max(128, block_codec(weight_layout).block_weights) != 0
                or size_n % 16 != 0
            ):
                raise ValueError(
                    "IQ2_XS W4A16 requires K % 256 == 0 and N % 16 == 0; "
                    f"got N={size_n} K={size_k}"
                )
        trellis_pair_kind = (
            None if trellis_pair_kind is None else str(trellis_pair_kind).upper()
        )
        trellis_rate_axis = (
            None if trellis_rate_axis is None else str(trellis_rate_axis).lower()
        )
        if (trellis_pair_kind is None) != (trellis_rate_axis is None):
            raise ValueError(
                "trellis_pair_kind and trellis_rate_axis must be supplied together"
            )
        if trellis_pair_kind is not None:
            if weight_layout != "trellis_t256":
                raise ValueError("trellis pairs require trellis_t256 weights")
            if trellis_pair_kind not in {
                "P24",
                "P33",
                "P43",
                "P44",
                "PDYNAMIC",
                "P33_P43",
            }:
                raise ValueError(
                    "trellis_pair_kind must be P24, P33, P43, P44, "
                    "PDYNAMIC, or P33_P43, got "
                    f"{trellis_pair_kind!r}"
                )
            if trellis_rate_axis not in {"k", "n"}:
                raise ValueError(
                    f"trellis_rate_axis must be 'k' or 'n', got {trellis_rate_axis!r}"
                )
            if trellis_bits != 3:
                raise ValueError(
                    "Trellis pair decoding requires the trellis_bits=3 base "
                    "specialization"
                )
        if epilogue_activation not in (None, "relu2"):
            raise ValueError(
                "W4A16 GEMM epilogue activation currently supports only relu2"
            )
        if tile_n % 16 != 0 or tile_k % 16 != 0:
            raise ValueError("tile_n/tile_k must be multiples of 16")
        if weight_layout in BLOCK_CODECS and tile_k > 128:
            raise ValueError("IQ2_XS warp reduction requires tile_k <= 128")
        if trellis_rate_axis == "n" and (size_n % 256 or tile_n != 256):
            raise ValueError(
                "N-axis trellis pairs require size_n % 256 == 0 and "
                "tile_n=256 so every CTA consumes one complete fixed-rate pair"
            )
        if trellis_rate_axis == "k" and (size_k != 256 or tile_k > 128 or 128 % tile_k):
            raise ValueError(
                "K-axis trellis pairs require size_k=256 and a tile_k that "
                "divides one 128-channel record"
            )
        scale_group_size = _scale_group_size(scale_format)
        has_n_tile_tail = int(size_n) % int(tile_n) != 0
        has_k_tile_tail = int(size_k) % int(tile_k) != 0
        has_scale_k_tail = int(size_k) % scale_group_size != 0
        has_logical_tail = has_n_tile_tail or has_k_tile_tail or has_scale_k_tail
        if has_logical_tail:
            if weight_layout != "modelopt" or scale_format != "e8m0_k32":
                raise ValueError(
                    "W4A16 logical tail tiles are only supported for native "
                    "E8M0 K/32 weights"
                )
            if int(size_n) % 16 != 0:
                raise ValueError(
                    "native E8M0 W4A16 tail tiles require size_n % 16 == 0"
                )
            if int(size_k) % 8 != 0:
                raise ValueError("native E8M0 W4A16 tail tiles require size_k % 8 == 0")
            if int(tile_k) % scale_group_size != 0:
                raise ValueError(
                    "native E8M0 W4A16 tail tiles require tile_k multiples of 32"
                )
        else:
            if size_n % tile_n != 0:
                raise ValueError("size_n must be divisible by tile_n")
            if size_k % tile_k != 0:
                raise ValueError("size_k must be divisible by tile_k")
            if scale_format == "e8m0_k32" and (size_k % 32 != 0 or tile_k % 32 != 0):
                raise ValueError(
                    "E8M0 K/32 W4A16 scales require size_k/tile_k multiples of 32"
                )
            if scale_format == "e4m3_k32" and (size_k % 32 != 0 or tile_k % 32 != 0):
                raise ValueError(
                    "E4M3 K/32 W4A16 scales require size_k/tile_k multiples of 32"
                )
        if moe_block_size not in _ALLOWED_ROUTED_SIZES:
            raise ValueError(f"unsupported moe_block_size {moe_block_size}")
        if moe_block_size != 8 and moe_block_size % 16 != 0:
            raise ValueError("moe_block_size must be 8 or a multiple of 16")
        cta_threads = tile_n * tile_k // 64
        if cta_threads not in (128, 256):
            raise ValueError("W4A16 GEMM expects 128 or 256 CTA threads")
        self.size_m = int(size_m)
        self.size_n = int(size_n)
        self.size_k = int(size_k)
        self.num_experts = int(num_experts)
        self.dynamic_num_experts = bool(dynamic_num_experts)
        if self.dynamic_num_experts and weight_layout not in {
            "packed",
            "trellis_t256",
            "iq2_xs",
            "iq2_xxs",
            "q8_0",
        }:
            raise ValueError(
                "dynamic_num_experts is only supported for packed and "
                "native compressed-codebook weights"
            )
        self.top_k = int(top_k)
        self.mul_topk_weights = bool(mul_topk_weights)
        self.tile_n = int(tile_n)
        self.tile_k = int(tile_k)
        self.cta_n_blocks = int(tile_n // 16)
        self.cta_k_blocks = int(tile_k // 16)
        self.cta_threads = int(cta_threads)
        self.moe_block_size = int(moe_block_size)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.epilogue_relu2 = epilogue_activation == "relu2"
        self.weight_layout = weight_layout
        self.trellis_bits = trellis_bits
        self.trellis_codebook = trellis_codebook
        self.trellis_pair_kind = trellis_pair_kind
        self.trellis_rate_axis = trellis_rate_axis
        self.weight_layout_trellis256_pair = trellis_pair_kind is not None
        self.trellis_pair_dynamic = trellis_pair_kind in {"PDYNAMIC", "P33_P43"}
        self.trellis_pair_compact_offsets = trellis_pair_kind == "P33_P43"
        static_pair_rates = {
            "P24": (2, 4),
            "P33": (3, 3),
            "P43": (4, 3),
            "P44": (4, 4),
        }
        self.trellis_pair_low_bits, self.trellis_pair_high_bits = static_pair_rates.get(
            trellis_pair_kind, (3, 3)
        )
        self.lut_e4m3_smem = False
        self.trellis_direct_lut = False
        # Large-M route blocks only issue the MMAs of 16-row blocks that hold
        # live routes; padded rows of a partial block are never read back.
        self.skip_empty_m_blocks = (
            os.environ.get("B12X_W4A16_SKIP_EMPTY_M_BLOCKS", "1") == "1"
            if skip_empty_m_blocks is None
            else skip_empty_m_blocks
        )
        # Small-M stripe split-K: opt out of the one-tile-per-CTA fast path
        # so decode-heavy small-M phases spread each mn-tile's K range across
        # multiple CTAs (existing tail scheduling plus cross-CTA finalize).
        self.small_m_splitk = (
            weight_layout in BLOCK_CODECS and moe_block_size == 8
        ) or _w4a16_small_m_splitk_enabled()
        self.weight_layout_trellis256 = weight_layout == "trellis_t256"
        self.weight_layout_block = weight_layout in BLOCK_CODECS
        self.iq2_xxs = weight_layout == "iq2_xxs"
        self.q8 = weight_layout == "q8_0"
        self.block_pack_factor = (
            block_codec(weight_layout).pack_factor if self.weight_layout_block else 1
        )
        self.block_k = (
            block_codec(weight_layout).block_weights
            if self.weight_layout_block
            else 256
        )
        self.block_metadata_bytes = (
            block_codec(weight_layout).metadata_bytes if self.weight_layout_block else 0
        )
        self.block_tile_bytes = 256 // self.block_pack_factor
        self.iq2_xs_smem_lut = False
        self.weight_layout_trellis256_proj = (
            self.weight_layout_trellis256 and w13_layout == "trellis_t256_proj"
        )
        if self.weight_layout_trellis256_proj and (
            self.size_n % 2 != 0 or (self.size_n // 2) % self.tile_n != 0
        ):
            raise ValueError(
                "trellis_t256_proj requires each FC1 projection to contain "
                "an integral number of CTA N tiles"
            )
        self.b_region_variable = (
            self.weight_layout_trellis256 or self.weight_layout_block
        )
        self.b_bundle_rows = 2
        self.scale_format = scale_format
        self.csf_block_inline = bool(csf_inline_scales)
        if self.csf_block_inline:
            from b12x._lib.quant.csf_inline import CsfInlineReader

            bits = 1 if scale_format == "e8m0_k32" else 4
            columns = (size_k + scale_group_size - 1) // scale_group_size
            self.csf_block_reader = CsfInlineReader(
                (size_n, columns, bits, 9, size_n, columns, num_experts, 8, 1, 1, 1)
            )
        self.native_nvfp4_scales = (
            weight_layout == "modelopt" and scale_format == "e4m3_k16"
        )
        self.scale_format_e8m0_k32 = scale_format == "e8m0_k32"
        # k32 scale cadence (two K16 rows share one scale group); the scale
        # DECODE stays keyed on scale_format_e8m0_k32 (e4m3_k32 uses the e4m3
        # decode arm, its 2**116 compensation lives in the global_scale tensor).
        self.scale_k32 = scale_format in ("e8m0_k32", "e4m3_k32")
        self.scale_group_size = int(scale_group_size)
        self.scale_k_groups = _covering_count(self.size_k, self.scale_group_size)
        self.n_tiles = _covering_count(self.size_n, self.tile_n)
        self.k_tiles = _covering_count(self.size_k, self.tile_k)
        self.covered_size_n = self.n_tiles * self.tile_n
        self.covered_size_k = self.k_tiles * self.tile_k
        self.has_n_tile_tail = bool(has_n_tile_tail)
        self.has_k_tile_tail = bool(has_k_tile_tail)
        self.has_scale_k_tail = bool(has_scale_k_tail)
        self.has_logical_tail = bool(has_logical_tail)
        self.scale_size_n = (
            _e8m0_logical_tail_scale_n(self.size_n)
            if self.scale_format_e8m0_k32 and self.has_n_tile_tail
            else self.size_n
        )
        self.scale_n_groups = self.scale_size_n // 16
        self.w13_layout = w13_layout
        self.source_n_rotation = int(source_n_rotation)
        self.single_token_route_fast_path = bool(single_token_route_fast_path)
        self.direct_topk_routes = bool(direct_topk_routes)
        self.dense_route_fast_path = bool(dense_route_fast_path)
        if self.dense_route_fast_path and (
            self.direct_topk_routes
            or self.single_token_route_fast_path
            or self.num_experts != 1
            or self.top_k != 1
            or self.mul_topk_weights
        ):
            raise ValueError(
                "dense_route_fast_path requires E=1, top_k=1, no top-k weights, "
                "and no other route fast path"
            )
        self.dual_a = bool(dual_a)
        if self.dual_a and not self.weight_layout_trellis256_proj:
            raise ValueError(
                "dual_a is only valid for projection-major trellis_t256 FC1"
            )
        self.route_major_a = bool(route_major_a)
        if self.route_major_a and not self.dual_a:
            raise ValueError("route_major_a requires the exact dual-A FC1 path")
        self.fused_topk_sum = bool(fused_topk_sum)
        self.fused_sum_fp32 = bool(fused_sum_fp32)
        self.fused_sum_router_weights_fp32 = bool(fused_sum_fp32 and _FP32_TOPK_WEIGHTS)
        self.fused_sum_topk = int(fused_sum_topk)
        # Whole-tile persistent scheduling: every mn-tile is computed by one
        # CTA over the full K (grid-strided waves, ragged last wave), skipping
        # the split-K tail machinery entirely. Requires the host to bound the
        # wave count; used by the exact-geometry hybrid decode schedule.
        self.schedule_whole_tiles = bool(schedule_whole_tiles)
        self.schedule_route_block_factor = int(schedule_route_block_factor)
        if self.schedule_route_block_factor < 1:
            raise ValueError("schedule_route_block_factor must be >= 1")
        if self.schedule_route_block_factor != 1 and (
            not self.schedule_whole_tiles
            or self.direct_topk_routes
            or self.dense_route_fast_path
        ):
            raise ValueError(
                "grouped route-block scheduling requires route-packed "
                "whole-tile execution"
            )
        if (
            self.schedule_whole_tiles
            and not self.direct_topk_routes
            and not self.weight_layout_trellis256
        ):
            raise ValueError(
                "schedule_whole_tiles requires direct_topk_routes or trellis_t256"
            )
        if self.fused_topk_sum and self.fused_sum_topk < 1:
            raise ValueError("fused_sum_topk must be >= 1")
        if self.fused_sum_fp32 and not self.fused_topk_sum:
            raise ValueError("fused_sum_fp32 requires fused_topk_sum")
        self.cta_m_blocks = int(_covering_count(moe_block_size, 16))
        self.uses_m_block_8 = moe_block_size == 8
        self.stages = _w4a16_pipeline_stages(
            weight_layout=weight_layout,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            pipeline_stages=pipeline_stages,
            scale_format=scale_format,
            small_m_occupancy=small_m_occupancy,
        )
        self.max_m_blocks = int(max_m_blocks)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", _DEFAULT_MAX_SHARED_MEM)
            )
        else:
            self.sms = 120
            max_shared_mem = _DEFAULT_MAX_SHARED_MEM
        self.max_shared_mem = max_shared_mem
        self.blocks_per_sm = _determine_blocks_per_sm(
            problem_m=self.size_m,
            problem_n=self.covered_size_n,
            top_k=self.top_k,
            cta_threads=self.cta_threads,
            cta_m_blocks=self.cta_m_blocks,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            uses_m_block_8=self.uses_m_block_8,
            sms=self.sms,
            max_shared_mem=max_shared_mem,
            scale_format=scale_format,
            weight_layout=weight_layout,
            weight_bits=(
                max(4, self.trellis_bits) if self.weight_layout_trellis256 else 4
            ),
            pipeline_stages=self.stages,
            small_m_occupancy=small_m_occupancy,
            problem_k=self.size_k,
        )

        # W4A16 shared-memory geometry, in int4 units unless noted.
        self.a_sh_stride = 16 * self.cta_k_blocks // 8
        self.a_sh_stage = self.a_sh_stride * (
            8 if self.uses_m_block_8 and self.stages == 3 else 16 * self.cta_m_blocks
        )
        self.a_gl_rd_delta_o = 16 * self.cta_k_blocks // 8
        self.a_sh_wr_delta = self.a_sh_stride * (
            self.cta_threads // self.a_gl_rd_delta_o
        )
        self.a_sh_wr_iters = _covering_count(self.a_sh_stage, self.a_sh_wr_delta)
        self.a_sh_rd_delta_i = self.a_sh_stride * 16

        self.b_sh_stride = ((self.cta_n_blocks * 16) * 16 // _PACK_FACTOR) // 4
        self.b_thread_vecs = 1
        self.b_sh_stride_threads = self.b_sh_stride
        self.b_sh_stage = self.b_sh_stride * self.cta_k_blocks
        self.b_sh_wr_iters = self.b_sh_stage // self.cta_threads
        # Native t256 uses 4*bits bytes per 32 codes;
        # packed/modelopt use 16 bytes for the same logical unit.
        self.b_unit_bytes = 16
        if self.weight_layout_trellis256:
            self.b_unit_bytes = 4 * self.trellis_bits
            if self.weight_layout_trellis256_pair:
                if self.trellis_rate_axis == "k":
                    self.b_unit_bytes = 16
                elif self.trellis_pair_compact_offsets:
                    self.b_unit_bytes = 14
                elif not self.trellis_pair_dynamic:
                    self.b_unit_bytes = 2 * (
                        self.trellis_pair_low_bits + self.trellis_pair_high_bits
                    )
        self.b_sh_stage_bytes = self.b_sh_stage * self.b_unit_bytes
        if self.weight_layout_block:
            self.b_sh_stage_bytes = _iq2_xs_stage_bytes(
                self.tile_k, self.tile_n, weight_layout
            )
        if self.b_region_variable:
            if self.b_sh_stage_bytes % 16 != 0:
                raise ValueError(
                    "trellis B stage bytes must be a multiple of 16; "
                    f"got {self.b_sh_stage_bytes}"
                )
            self.b_sh_chunks = self.b_sh_stage_bytes // 16
            self.b_sh_wr_iters_var = _covering_count(self.b_sh_chunks, self.cta_threads)
        else:
            self.b_sh_chunks = self.b_sh_stage
            self.b_sh_wr_iters_var = self.b_sh_wr_iters

        self.s_sh_stride = 16 * self.cta_n_blocks // 16
        self.s_tb_groups = (
            self.cta_k_blocks // 2 if self.scale_k32 else self.cta_k_blocks
        )
        self.s_sh_stage = (
            0 if self.weight_layout_block else self.s_tb_groups * self.s_sh_stride
        )
        self.tb_n_warps = self.cta_n_blocks // 4

        sh_block_route_indices = self.moe_block_size // 4
        sh_rd_block_route_indices = self.moe_block_size // 4
        sh_block_topk_weights = self.moe_block_size // 2
        self.sh_valid_count_off = (
            sh_block_route_indices + sh_rd_block_route_indices + sh_block_topk_weights
        )
        self.sh_route_off = 0
        self.sh_rd_route_off = sh_block_route_indices
        self.sh_topk_off = sh_block_route_indices + sh_rd_block_route_indices

        sh_red_size = (2 * self.cta_n_blocks + 1) * 16 * self.cta_m_blocks
        # B region size in int4 (16-byte) units, rounded up to a 16-byte
        # multiple so the following SMEM regions keep their 16-byte alignment
        # (exact for all supported tiles since b_sh_stage is a multiple of 4).
        sh_b_size = _covering_count(self.stages * self.b_sh_stage_bytes, 16)
        sh_size_min = min(sh_red_size, sh_b_size)
        sh_size_max = max(sh_red_size, sh_b_size)
        sh_bias_size = self.cta_n_blocks * 16 // 8
        sh_b_red_bias_size = max(sh_size_max, sh_size_min + sh_bias_size)
        self.sh_b_off = self.sh_valid_count_off
        self.sh_red_off = self.sh_valid_count_off
        self.sh_s_off = self.sh_valid_count_off + sh_b_red_bias_size
        self.sh_a_off = self.sh_s_off + self.stages * self.s_sh_stage
        self.shared_int4 = self.sh_a_off + self.stages * self.a_sh_stage
        self.shared_int4 = max(
            self.shared_int4,
            self.sh_red_off
            + _reduction_shared_bytes(self.tile_n, self.tile_k, self.uses_m_block_8)
            // 16,
        )
        if self.csf_block_inline:
            self.csf_stage_groups = ((self.tile_n + 127) // 128) * (
                (self.s_tb_groups + 3) // 4
            )
            self.csf_inplace_stage = (
                self.tile_n == 128
                and self.size_n % 128 == 0
                and self.s_tb_groups % 4 == 0
            )
            if self.csf_inplace_stage:
                self.sh_csf_block_off = self.sh_s_off * 16
            else:
                self.sh_csf_block_off = self.shared_int4 * 16
                self.shared_int4 += self.stages * self.csf_stage_groups * 32
            self.sh_csf_index_off = self.shared_int4 * 16
            self.csf_index_count = (
                (self.tile_n + 127) // 128
            ) * self.csf_block_reader.plane.storage.column_groups
            self.shared_int4 += (self.csf_index_count * 8 + 15) // 16
        self.shared_words = self.shared_int4 * 4
        if self.csf_block_inline:
            self.blocks_per_sm = min(
                self.blocks_per_sm,
                _device_shared_mem_per_sm()
                // _resident_cta_shared_bytes(self.shared_words * 4),
            )
        if self.shared_words * 4 > int(max_shared_mem):
            raise ValueError(
                "W4A16 shared-memory footprint exceeds device opt-in limit: "
                f"{self.shared_words * 4} > {int(max_shared_mem)} bytes "
                f"(layout={self.weight_layout})"
            )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.size_n,
            self.size_k,
            self.covered_size_n,
            self.covered_size_k,
            self.scale_k_groups,
            self.has_logical_tail,
            None if self.dynamic_num_experts else self.num_experts,
            self.dynamic_num_experts,
            self.top_k,
            self.mul_topk_weights,
            self.tile_n,
            self.tile_k,
            self.stages,
            self.cta_threads,
            self.moe_block_size,
            self.element_dtype,
            self.epilogue_relu2,
            self.weight_layout,
            self.trellis_bits,
            self.trellis_codebook,
            self.trellis_pair_kind,
            self.trellis_rate_axis,
            self.scale_format,
            self.csf_block_inline,
            self.num_experts if self.csf_block_inline else None,
            self.native_nvfp4_scales,
            self.w13_layout,
            self.source_n_rotation,
            self.single_token_route_fast_path,
            self.direct_topk_routes,
            self.dense_route_fast_path,
            self.dual_a,
            self.route_major_a,
            self.fused_topk_sum,
            self.fused_sum_fp32,
            self.fused_sum_router_weights_fp32,
            self.fused_sum_topk,
            self.size_m if self.fused_sum_fp32 else None,
            self.cta_m_blocks,
            self.uses_m_block_8,
            self.shared_words,
            # Launch bounds are part of the compiled kernel.  Keep binaries
            # planned for different residency targets out of the same cache
            # entry even when their arithmetic geometry otherwise matches.
            self.blocks_per_sm,
            self.schedule_whole_tiles,
            self.schedule_route_block_factor,
            self.lut_e4m3_smem,
            self.trellis_direct_lut,
            self.small_m_splitk,
            self.skip_empty_m_blocks,
        )

    @cute.jit
    def _activation_smem_permuted_offset(self, i: Int32) -> Int32:
        row = i // Int32(self.a_gl_rd_delta_o)
        if cutlass.const_expr(self.a_gl_rd_delta_o < 8):
            # K32 rows hold four 16-byte vectors: an XOR with row & 7 would
            # leave the row (colliding rows 7 and 8 and overrunning the
            # stage). Pairs of rows share one 128-byte line, so XOR the
            # vector with the pair index to keep ldmatrix conflict free.
            return Int32(self.a_gl_rd_delta_o) * row + (
                (i - row * Int32(self.a_gl_rd_delta_o))
                ^ ((row >> Int32(1)) & Int32(self.a_gl_rd_delta_o - 1))
            )
        return Int32(self.a_gl_rd_delta_o) * row + (
            (i - row * Int32(self.a_gl_rd_delta_o)) ^ (row & Int32(7))
        )

    @cute.jit
    def _int4_addr(self, smem_base: Int32, int4_off: Int32) -> Int32:
        return smem_base + int4_off * Int32(16)

    @cute.jit
    def _iq2_xs_descriptor_offset(
        self, expert_idx: Int32, global_k16: Int32, global_n16: Int32
    ):
        tile = (
            Int64(expert_idx) * Int64(self.size_n // 64) + Int64(global_n16 // Int32(4))
        ) * Int64(self.size_k // 128) + Int64(global_k16 // Int32(8))
        return tile * Int64(32 * self.block_tile_bytes // 4) + (
            Int64(global_k16 % Int32(8)) * Int64(4) + Int64(global_n16 % Int32(4))
        ) * Int64(self.block_tile_bytes // 4)

    @cute.jit
    def _dequant_e2m1x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e2m1x4_to_half2x2(packed)
        return packed_dequant_e2m1x4_to_bfloat2x2(packed)

    @cute.jit
    def _dequant_e4m3x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e4m3x4_to_half2x2(packed)
        return packed_dequant_e4m3x4_to_bfloat2x2(packed)

    @cute.jit
    def _dequant_scale_x4_to_elem2x2(self, packed: Uint32):
        if cutlass.const_expr(self.native_nvfp4_scales):
            if cutlass.const_expr(self.is_fp16):
                return fp8x4_e4m3_to_half2x2(packed)
            s0, s1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(packed)
            # The bitwise FP4 decoder produces values scaled by 2**-126.
            # All finite E4M3 scales remain representable after a 2**119 lift;
            # the remaining factor of 128 is applied with the weight global.
            lift = Uint32(0x7B007B00)
            return bfloat2_mul(s0, lift), bfloat2_mul(s1, lift)
        if cutlass.const_expr(self.scale_format_e8m0_k32):
            if cutlass.const_expr(self.is_fp16):
                return packed_dequant_e8m0x4_to_half2x2(packed)
            return packed_dequant_e8m0x4_to_bfloat2x2(packed)
        if cutlass.const_expr(self.is_fp16):
            return packed_dequant_e4m3x4_to_half2x2(packed)
        return packed_dequant_e4m3x4_to_bfloat2x2(packed)

    @cute.jit
    def _elem2_mul(self, a: Uint32, b: Uint32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return half2_mul(a, b)
        return bfloat2_mul(a, b)

    @cute.jit
    def _broadcast_f32_to_elem2(self, x: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return broadcast_f32_to_half2(x)
        return broadcast_f32_to_bfloat2(x)

    @cute.jit
    def _pack_f32x2_to_elem2(self, x0: cutlass.Float32, x1: cutlass.Float32) -> Uint32:
        if cutlass.const_expr(self.is_fp16):
            return pack_f32x2_to_f16x2(x0, x1)
        return pack_f32x2_to_bfloat2(x0, x1)

    @cute.jit
    def _elem2_to_f32x2(self, packed: Uint32):
        if cutlass.const_expr(self.is_fp16):
            return half2_to_float2_scaled(packed, cutlass.Float32(1.0))
        return bfloat2_to_float2_scaled(packed, cutlass.Float32(1.0))

    @cute.jit
    def _relu2_elem2(self, packed: Uint32) -> Uint32:
        x0, x1 = self._elem2_to_f32x2(packed)
        if x0 < cutlass.Float32(0.0):
            x0 = cutlass.Float32(0.0)
        if x1 < cutlass.Float32(0.0):
            x1 = cutlass.Float32(0.0)
        return self._pack_f32x2_to_elem2(x0 * x0, x1 * x1)

    @cute.jit
    def _st_shared_elem_from_f32(self, addr: Int32, val: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            st_shared_f16_from_f32(addr, val)
        else:
            st_shared_bf16_from_f32(addr, val)

    @cute.jit
    def _mma_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        a0: Uint32,
        a1: Uint32,
        a2: Uint32,
        a3: Uint32,
        b0: Uint32,
        b1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)
        return bf16_mma_m16n8k16_f32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)

    @cute.jit
    def _mma_rhs_fragments_as_mma_a_m16n8k16_f32(
        self,
        d0: cutlass.Float32,
        d1: cutlass.Float32,
        d2: cutlass.Float32,
        d3: cutlass.Float32,
        b0_0: Uint32,
        b1_0: Uint32,
        b0_1: Uint32,
        b1_1: Uint32,
        a0: Uint32,
        a1: Uint32,
    ):
        if cutlass.const_expr(self.is_fp16):
            return f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
            )
        return bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            d0, d1, d2, d3, b0_0, b1_0, b0_1, b1_1, a0, a1
        )

    @cute.jit
    def __call__(
        self,
        a_bf16_ptr: cute.Pointer,
        a_alt_bf16_ptr: cute.Pointer,
        b_i32_flat: cute.Tensor,
        c_bf16_ptr: cute.Pointer,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_flat: cute.Tensor,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        a_bf16_flat = cute.make_tensor(
            a_bf16_ptr,
            layout=cute.make_layout((active_m * Int32(self.size_k),), stride=(1,)),
        )
        a_alt_bf16_flat = cute.make_tensor(
            a_alt_bf16_ptr,
            layout=cute.make_layout((active_m * Int32(self.size_k),), stride=(1,)),
        )
        c_bf16_flat = cute.make_tensor(
            c_bf16_ptr,
            layout=cute.make_layout(
                (active_m * Int32(self.top_k) * Int32(self.size_n),), stride=(1,)
            ),
        )
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            trellis_lut_flat,
            active_m,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_flat: cute.Tensor,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        grid_x, _, _ = cute.arch.grid_dim()
        trellis_lut_addr = get_ptr_as_int64(trellis_lut_flat, Int32(0))
        self._run_persistent_gemm(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            trellis_lut_addr,
            smem_base,
            tid,
            cta,
            Int32(grid_x),
            Int32(active_m),
        )

    @cute.jit
    def _run_persistent_gemm(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_size_m: Int32,
        emit_tile: cutlass.Constexpr = None,
    ):
        n_tiles = Int32(self.n_tiles)
        route_blocks = active_size_m * Int32(self.top_k)
        if cutlass.const_expr(self.dense_route_fast_path):
            route_blocks = (
                active_size_m + Int32(self.moe_block_size) - Int32(1)
            ) // Int32(self.moe_block_size)
        elif cutlass.const_expr(not self.direct_topk_routes):
            route_blocks = packed_route_count[Int32(0)].to(Int32) // Int32(
                self.moe_block_size * self.schedule_route_block_factor
            )
        k_tiles = Int32(self.k_tiles)
        global_mn_tiles = route_blocks * n_tiles

        tail_mn_tiles = global_mn_tiles
        full_grid_mn_iters = Int32(0)
        force_one_tile_per_cta = Int32(0)
        if cutlass.const_expr(self.schedule_whole_tiles):
            # Whole-tile waves: one CTA computes each mn-tile over the full K,
            # task = cta + wave * grid_x, ragged last wave skipped through the
            # route_block_idx bound below. No split-K tail, no lock traffic.
            tail_mn_tiles = Int32(0)
            full_grid_mn_iters = (global_mn_tiles + grid_x - Int32(1)) // grid_x
        if cutlass.const_expr(not self.schedule_whole_tiles):
            if cutlass.const_expr(self.uses_m_block_8 and not self.small_m_splitk):
                # TC-decode small-M: when every mn-tile fits inside the launched grid
                # (FC1 has only route_blocks*n_tiles tiles, far fewer than grid_x),
                # the default tail path fans each mn-tile across multiple CTAs along
                # K and pays a lock-serialized cross-CTA split-K finalize plus the
                # reduction-turn handshake. Instead give the first global_mn_tiles
                # CTAs exactly one full mn-tile (all k_tiles, reduce_slice_count==1,
                # no finalize, no lock traffic) and idle the rest. grid_x and the
                # grid-barrier participant count are unchanged (so FC2 coverage is
                # untouched); only FC1's intra-GEMM work partition changes. Numerically
                # identical: a single CTA computes the whole K-reduction per tile.
                # Decode-heavy codecs opt back into the stripe split-K partition
                # via ``B12X_W4A16_SMALL_M_SPLITK`` (see
                # ``_w4a16_small_m_splitk_enabled``): idling 5/6 of the grid
                # through an expensive FC1 costs far more than the finalize.
                if global_mn_tiles <= grid_x:
                    force_one_tile_per_cta = Int32(1)
            whole_k_waves = Int32(0)
            if cutlass.const_expr(self.weight_layout_block and self.direct_topk_routes):
                if global_mn_tiles >= grid_x:
                    whole_k_waves = Int32(1)
            if whole_k_waves != Int32(0):
                tail_mn_tiles = Int32(0)
                full_grid_mn_iters = (global_mn_tiles + grid_x - Int32(1)) // grid_x
            elif force_one_tile_per_cta != Int32(0):
                tail_mn_tiles = Int32(0)
                full_grid_mn_iters = Int32(1)
            elif global_mn_tiles > grid_x:
                tail_mn_tiles = global_mn_tiles - (global_mn_tiles // grid_x) * grid_x
                if tail_mn_tiles * Int32(3) <= grid_x:
                    tail_mn_tiles += grid_x
                full_grid_mn_iters = (global_mn_tiles - tail_mn_tiles) // grid_x

        iters = (k_tiles * tail_mn_tiles + grid_x - Int32(1)) // grid_x

        lock_slot = Int32(0)
        if tail_mn_tiles >= grid_x:
            lock_slot = cta
        else:
            lock_slot = (iters * cta) // k_tiles - Int32(1)

        in_tail_region = Int32(0)
        has_work = Int32(1)
        work_mn_tile = cta
        reduce_k_tile = Int32(0)
        route_block_idx = Int32(0)
        output_n_tile = Int32(0)
        if iters == Int32(0) and full_grid_mn_iters == Int32(0):
            has_work = Int32(0)

        while has_work != Int32(0):
            reduce_tile_count = Int32(0)
            reduce_slice_count = Int32(1)
            reduce_slice_idx = Int32(0)

            if in_tail_region == Int32(0) and full_grid_mn_iters > Int32(0):
                route_block_idx = work_mn_tile // n_tiles
                output_n_tile = work_mn_tile - route_block_idx * n_tiles
                reduce_k_tile = Int32(0)
                reduce_tile_count = k_tiles
                full_grid_mn_iters -= Int32(1)
            else:
                if in_tail_region == Int32(0):
                    in_tail_region = Int32(1)
                    tail_mn_base = global_mn_tiles - tail_mn_tiles
                    cta_iter_start = iters * cta
                    work_mn_tile = cta_iter_start // k_tiles
                    reduce_k_tile = cta_iter_start - work_mn_tile * k_tiles
                    global_mn_tile = work_mn_tile + tail_mn_base
                    route_block_idx = global_mn_tile // n_tiles
                    output_n_tile = global_mn_tile - route_block_idx * n_tiles

                if work_mn_tile < tail_mn_tiles and iters > Int32(0):
                    reduce_tile_count = iters * (cta + Int32(1)) - (
                        k_tiles * work_mn_tile + reduce_k_tile
                    )
                    if reduce_tile_count < Int32(0):
                        reduce_tile_count = Int32(0)
                    if reduce_k_tile + reduce_tile_count > k_tiles:
                        reduce_tile_count = k_tiles - reduce_k_tile

                    if reduce_tile_count > Int32(0):
                        first_reduce_boundary = iters * (
                            (k_tiles * work_mn_tile + iters - Int32(1)) // iters
                        )
                        if first_reduce_boundary <= k_tiles * (work_mn_tile + Int32(1)):
                            reduce_boundary_offset = (
                                first_reduce_boundary - k_tiles * work_mn_tile
                            )
                            reduce_slice_count = (
                                k_tiles - reduce_boundary_offset + iters - Int32(1)
                            ) // iters
                            if reduce_boundary_offset > Int32(0):
                                reduce_slice_count += Int32(1)
                            reduce_boundary_delta = iters * cta - first_reduce_boundary
                            if reduce_boundary_delta < Int32(0):
                                reduce_slice_idx = reduce_slice_count - Int32(1)
                            else:
                                if reduce_boundary_offset == Int32(
                                    0
                                ) and reduce_boundary_delta == Int32(0):
                                    reduce_slice_idx = reduce_slice_count - Int32(1)
                                else:
                                    reduce_slice_idx = (
                                        reduce_slice_count
                                        - Int32(1)
                                        - reduce_boundary_delta // iters
                                    )
                                    if reduce_boundary_offset > Int32(0):
                                        reduce_slice_idx -= Int32(1)

                        if tail_mn_tiles >= grid_x:
                            if reduce_slice_count > Int32(
                                1
                            ) and reduce_slice_idx == reduce_slice_count - Int32(1):
                                lock_slot += Int32(1)
                        else:
                            lock_slot += Int32(1)
                    else:
                        has_work = Int32(0)
                else:
                    has_work = Int32(0)

            if (
                has_work != Int32(0)
                and reduce_tile_count > Int32(0)
                and route_block_idx < route_blocks
            ):
                if cutlass.const_expr(emit_tile is not None):
                    # Trace-time tile emission hook: the caller owns expert
                    # resolution and the _run_tile dispatch (e.g. the hybrid
                    # multi-tier route map). The scheduling state machine above
                    # is unchanged; only tile emission is delegated.
                    emit_tile(
                        route_block_idx,
                        output_n_tile,
                        reduce_k_tile,
                        reduce_tile_count,
                        reduce_slice_count,
                        reduce_slice_idx,
                        lock_slot,
                    )
                else:
                    if cutlass.const_expr(self.dense_route_fast_path):
                        expert_idx = Int32(0)
                    elif cutlass.const_expr(self.direct_topk_routes):
                        expert_idx = packed_route_indices[route_block_idx].to(Int32)
                    else:
                        expert_idx = block_expert_ids[route_block_idx].to(Int32)
                    if expert_idx >= Int32(0):
                        self._run_tile(
                            a_bf16_flat,
                            a_alt_bf16_flat,
                            b_i32_flat,
                            c_bf16_flat,
                            scales_i32_flat,
                            global_scale,
                            packed_route_indices,
                            topk_weights_flat,
                            c_tmp_f32_flat,
                            locks_i32_flat,
                            trellis_lut_addr,
                            smem_base,
                            tid,
                            route_block_idx,
                            expert_idx,
                            output_n_tile,
                            reduce_k_tile,
                            reduce_tile_count,
                            reduce_slice_count,
                            reduce_slice_idx,
                            lock_slot,
                            active_size_m,
                        )

            if has_work != Int32(0):
                if in_tail_region == Int32(0):
                    work_mn_tile += grid_x
                else:
                    reduce_k_tile = Int32(0)
                    work_mn_tile += Int32(1)
                    output_n_tile += Int32(1)
                    if output_n_tile == n_tiles:
                        output_n_tile = Int32(0)
                        route_block_idx += Int32(1)

    @cute.jit
    def _read_moe_block_data(
        self,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        global_scale_f32: cutlass.Float32,
        active_size_m: Int32,
    ) -> Int32:
        if cutlass.const_expr(self.dense_route_fast_path):
            block_row = route_block_idx * Int32(self.moe_block_size)
            valid_rows = active_size_m - block_row
            if valid_rows > Int32(self.moe_block_size):
                valid_rows = Int32(self.moe_block_size)
            if valid_rows < Int32(0):
                valid_rows = Int32(0)
            if tid < Int32(self.moe_block_size):
                row = block_row + tid
                st_shared_i32(
                    smem_base + Int32(self.sh_route_off * 16) + tid * Int32(4),
                    row,
                )
                st_shared_i32(
                    smem_base + Int32(self.sh_rd_route_off * 16) + tid * Int32(4),
                    row,
                )
            cute.arch.sync_threads()
            return valid_rows

        if cutlass.const_expr(self.direct_topk_routes):
            if tid == Int32(0):
                idx = route_block_idx
                st_shared_i32(smem_base + Int32(self.sh_route_off * 16), idx)
                rd_row = idx // Int32(self.top_k)
                if cutlass.const_expr(self.route_major_a):
                    rd_row = idx
                st_shared_i32(smem_base + Int32(self.sh_rd_route_off * 16), rd_row)
                if cutlass.const_expr(self.mul_topk_weights):
                    topk = topk_weights_flat[idx].to(cutlass.Float32) * global_scale_f32
                    st_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16),
                        self._broadcast_f32_to_elem2(topk),
                    )
                elif cutlass.const_expr(self.fused_sum_router_weights_fp32):
                    st_shared_f32(
                        smem_base + Int32(self.sh_topk_off * 16),
                        topk_weights_flat[idx].to(cutlass.Float32),
                    )
            cute.arch.sync_threads()
            return Int32(1)

        if cutlass.const_expr(self.single_token_route_fast_path):
            if tid == Int32(0):
                idx = packed_route_indices[
                    route_block_idx * Int32(self.moe_block_size)
                ].to(Int32)
                st_shared_i32(smem_base + Int32(self.sh_route_off * 16), idx)
                rd_row = idx // Int32(self.top_k)
                if cutlass.const_expr(self.route_major_a):
                    rd_row = idx
                st_shared_i32(smem_base + Int32(self.sh_rd_route_off * 16), rd_row)
                if cutlass.const_expr(self.mul_topk_weights):
                    safe_idx = idx
                    if idx >= active_size_m * Int32(self.top_k):
                        safe_idx = Int32(0)
                    topk = (
                        topk_weights_flat[safe_idx].to(cutlass.Float32)
                        * global_scale_f32
                    )
                    st_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16),
                        self._broadcast_f32_to_elem2(topk),
                    )
                elif cutlass.const_expr(self.fused_sum_router_weights_fp32):
                    safe_idx = idx
                    if idx >= active_size_m * Int32(self.top_k):
                        safe_idx = Int32(0)
                    st_shared_f32(
                        smem_base + Int32(self.sh_topk_off * 16),
                        topk_weights_flat[safe_idx].to(cutlass.Float32),
                    )
            cute.arch.sync_threads()
            return Int32(1)

        route_indices_int4_addr = self._int4_addr(
            smem_base, Int32(self.sh_route_off) + tid
        )
        route_indices_gmem = get_ptr_as_int64(
            packed_route_indices,
            route_block_idx * Int32(self.moe_block_size) + tid * Int32(4),
        )
        cp_async4_shared_global_pred(
            route_indices_int4_addr,
            route_indices_gmem,
            (tid < Int32(self.moe_block_size // 4)).to(Int32),
        )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if tid >= Int32(self.cta_threads - 32):
            size_per_thread = _covering_count(self.moe_block_size, 32)
            lane = tid - Int32(self.cta_threads - 32)
            local_count = Int32(0)
            for i in cutlass.range_constexpr(size_per_thread):
                j = lane * Int32(size_per_thread) + Int32(i)
                if j < Int32(self.moe_block_size):
                    idx = ld_shared_i32_relaxed(
                        smem_base + Int32(self.sh_route_off * 16) + j * Int32(4)
                    )
                    if idx < active_size_m * Int32(self.top_k):
                        local_count += Int32(1)
            valid = cute.arch.warp_redux_sync(local_count, "add")
            if lane == Int32(0):
                st_shared_i32(smem_base + Int32(self.sh_valid_count_off * 16), valid)

        if tid < Int32(self.moe_block_size):
            idx = ld_shared_i32_relaxed(
                smem_base + Int32(self.sh_route_off * 16) + tid * Int32(4)
            )
            rd_row = idx // Int32(self.top_k)
            if cutlass.const_expr(self.route_major_a):
                rd_row = idx
            st_shared_i32(
                smem_base + Int32(self.sh_rd_route_off * 16) + tid * Int32(4),
                rd_row,
            )
            if cutlass.const_expr(self.mul_topk_weights):
                safe_idx = idx
                if idx >= active_size_m * Int32(self.top_k):
                    safe_idx = Int32(0)
                topk = (
                    topk_weights_flat[safe_idx].to(cutlass.Float32) * global_scale_f32
                )
                packed_topk = self._broadcast_f32_to_elem2(topk)
                # top-k weights are cached as packed element pairs.
                topk_word_addr = (
                    smem_base + Int32(self.sh_topk_off * 16) + tid * Int32(4)
                )
                st_shared_u32(topk_word_addr, packed_topk)
            elif cutlass.const_expr(self.fused_sum_router_weights_fp32):
                safe_idx = idx
                if idx >= active_size_m * Int32(self.top_k):
                    safe_idx = Int32(0)
                st_shared_f32(
                    smem_base + Int32(self.sh_topk_off * 16) + tid * Int32(4),
                    topk_weights_flat[safe_idx].to(cutlass.Float32),
                )

        cute.arch.sync_threads()
        valid_count = ld_shared_i32_relaxed(
            smem_base + Int32(self.sh_valid_count_off * 16)
        )
        cute.arch.sync_threads()
        return valid_count

    @cute.jit
    def _run_tile(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
    ):
        if cutlass.const_expr(self.trellis_pair_dynamic):
            # One expert-static, CTA-uniform branch chooses a fully specialized
            # pair path.  The mode never changes during this tile, so carrying
            # it as a constexpr keeps all inner staging/decode loops branchless.
            pair_mode = scales_i32_flat[expert_idx].to(Int64)
            if cutlass.const_expr(self.trellis_pair_compact_offsets):
                pair_mode = pair_mode & Int64(1)
            if pair_mode != Int64(0):
                self._run_tile_with_pair_override(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    packed_route_indices,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
                    smem_base,
                    tid,
                    route_block_idx,
                    expert_idx,
                    output_n_tile,
                    reduce_k_tile,
                    reduce_tile_count,
                    reduce_slice_count,
                    reduce_slice_idx,
                    lock_slot,
                    active_size_m,
                    2 if cutlass.const_expr(self.trellis_pair_compact_offsets) else 1,
                )
            else:
                self._run_tile_with_pair_override(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    packed_route_indices,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
                    smem_base,
                    tid,
                    route_block_idx,
                    expert_idx,
                    output_n_tile,
                    reduce_k_tile,
                    reduce_tile_count,
                    reduce_slice_count,
                    reduce_slice_idx,
                    lock_slot,
                    active_size_m,
                    0,
                )
            return
        self._run_tile_with_pair_override(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_i32_flat,
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            trellis_lut_addr,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
            reduce_k_tile,
            reduce_tile_count,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            active_size_m,
            -1,
        )

    @cute.jit
    def _run_tile_with_pair_override(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.uses_m_block_8):
            self._run_tile_m8(
                a_bf16_flat,
                a_alt_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                trellis_lut_addr,
                smem_base,
                tid,
                route_block_idx,
                expert_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
                active_size_m,
                dynamic_pair_override,
            )
        else:
            self._run_tile_large_m(
                a_bf16_flat,
                a_alt_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_i32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                c_tmp_f32_flat,
                locks_i32_flat,
                trellis_lut_addr,
                smem_base,
                tid,
                route_block_idx,
                expert_idx,
                output_n_tile,
                reduce_k_tile,
                reduce_tile_count,
                reduce_slice_count,
                reduce_slice_idx,
                lock_slot,
                active_size_m,
                dynamic_pair_override,
            )

    @cute.jit
    def _tile_common_prologue(
        self,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        active_size_m: Int32,
    ):
        global_scale_f32 = global_scale[expert_idx].to(cutlass.Float32)
        if cutlass.const_expr(self.native_nvfp4_scales and not self.is_fp16):
            global_scale_f32 *= cutlass.Float32(128.0)
        if cutlass.const_expr(self.scale_format_e8m0_k32):
            if cutlass.const_expr(self.is_fp16):
                global_scale_f32 *= cutlass.Float32(_E8M0_K32_FP16_GLOBAL_COMPENSATION)
            else:
                global_scale_f32 *= cutlass.Float32(_E8M0_K32_BF16_GLOBAL_COMPENSATION)
        block_valid_rows = self._read_moe_block_data(
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            global_scale_f32,
            active_size_m,
        )
        (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_stream_offsets(tid, expert_idx, output_n_tile)
        return (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _tile_stream_offsets(self, tid: Int32, expert_idx: Int32, output_n_tile: Int32):
        a_gl_stride = Int32(self.size_k // 8)
        b_gl_stride = Int32(16 * self.size_n // (_PACK_FACTOR * 4))
        s_gl_stride = Int32(self.scale_n_groups)
        if cutlass.const_expr(self.scale_k32):
            if cutlass.const_expr(self.scale_format_e8m0_k32 and self.has_logical_tail):
                scales_expert_stride = Int32(self.scale_k_groups * self.scale_n_groups)
            else:
                scales_expert_stride = Int32((self.size_n * self.size_k) // (32 * 16))
        else:
            scales_expert_stride = Int32((self.size_n * self.size_k) // (16 * 16))
        b_expert_off = (
            Int32((self.size_n * self.size_k) // (_PACK_FACTOR * 4)) * expert_idx
        )
        scales_expert_off = scales_expert_stride * expert_idx

        a_gl_rd_row = tid // Int32(self.a_gl_rd_delta_o)
        a_gl_rd_col0 = tid - a_gl_rd_row * Int32(self.a_gl_rd_delta_o)
        a_sh_wr = Int32(self.a_sh_stride) * (tid // Int32(self.a_gl_rd_delta_o)) + (
            tid - (tid // Int32(self.a_gl_rd_delta_o)) * Int32(self.a_gl_rd_delta_o)
        )
        a_rows_per_iter = Int32(self.cta_threads // self.a_gl_rd_delta_o)

        if cutlass.const_expr(self.cta_threads <= self.b_sh_stride):
            b_gl_rd_base = tid
        else:
            b_gl_rd_base = b_gl_stride * (tid // Int32(self.b_sh_stride)) + (
                tid % Int32(self.b_sh_stride)
            )
        b_gl_rd_base += b_expert_off + Int32(self.b_sh_stride) * output_n_tile
        b_sh_rd = tid
        b_sh_rd += (b_sh_rd // Int32(self.b_sh_stride)) * Int32(
            self.b_sh_stride * (self.b_sh_wr_iters - 1)
        )

        s_sh_rd = Int32(8) * ((tid // Int32(32)) % Int32(self.tb_n_warps)) + (
            tid & Int32(31)
        ) // Int32(4)
        return (
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        )

    @cute.jit
    def _a_shared_read_offset(self, tid: Int32, lanes_per_row: cutlass.Constexpr[int]):
        a_sh_rd = Int32(self.a_sh_stride) * (
            (tid & Int32(31)) % Int32(lanes_per_row)
        ) + (tid & Int32(31)) // Int32(lanes_per_row)
        a_sh_rd += (
            Int32(2)
            * ((tid // Int32(32)) // Int32(self.tb_n_warps))
            * Int32(self.b_sh_wr_iters)
        )
        return a_sh_rd

    @cute.jit
    def _run_tile_m8(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
            active_size_m,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 8)
        # LLVM 23 also promotes this 16-f32 accumulator across the pipelined
        # control-flow joins, repeatedly packing adjacent values through i64
        # temporaries.  Keep the values in independent scalar fragments, as in
        # the large-M path below, so those PHIs remain scalar.
        acc = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
            for _ in range(16 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(16 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc[frag].fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
            dynamic_pair_override,
        )

        b_scale_cur = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
            reduce_k_tile,
            dynamic_pair_override,
        )
        a_regs_cur = cute.make_rmem_tensor((2,), Uint32)
        a_regs_next = cute.make_rmem_tensor((2,), Uint32)
        self._load_a_register_bundle(
            a_regs_cur,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            True,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            trellis_lut_addr,
            smem_base,
            tid,
            acc,
            acc,
            acc,
            acc,
            b_scale_cur,
            b_scale_next,
            a_regs_cur,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
            dynamic_pair_override,
            True,
        )

        self._finish_tile(
            acc,
            acc,
            acc,
            acc,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            True,
        )

    @cute.jit
    def _run_tile_large_m(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
            active_size_m,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 16)
        # Keep each accumulator element in its own scalar rmem tensor. LLVM 23
        # otherwise promotes the 128-f32 accumulator to wide vector PHIs and
        # repeatedly packs/unpacks adjacent f32 values through i64 temporaries.
        acc0 = [
            cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
            for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
        ]
        for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
            acc0[frag].fill(0.0)
        acc1 = acc0
        acc2 = acc0
        acc3 = acc0
        if cutlass.const_expr(self.cta_m_blocks > 1):
            acc1 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc1[frag].fill(0.0)
        if cutlass.const_expr(self.cta_m_blocks > 2):
            acc2 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc2[frag].fill(0.0)
        if cutlass.const_expr(self.cta_m_blocks > 3):
            acc3 = [
                cute.make_rmem_tensor((_SCALAR_ACC_FRAGMENT_WIDTH,), cutlass.Float32)
                for _ in range(32 // _SCALAR_ACC_FRAGMENT_WIDTH)
            ]
            for frag in cutlass.range_constexpr(32 // _SCALAR_ACC_FRAGMENT_WIDTH):
                acc3[frag].fill(0.0)

        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
            dynamic_pair_override,
        )

        b_scale_cur = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        b_scale_next = cute.make_rmem_tensor((self.b_bundle_rows, 4), Uint32)
        self._load_b_scale_register_bundle(
            b_scale_cur,
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            Int32(0),
            Int32(0),
            reduce_k_tile,
            dynamic_pair_override,
        )
        a_regs = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        a_regs_next = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        self._load_a_register_bundle(
            a_regs,
            smem_base,
            a_sh_rd,
            Int32(0),
            Int32(0),
            False,
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            a_alt_bf16_flat,
            b_i32_flat,
            scales_i32_flat,
            trellis_lut_addr,
            smem_base,
            tid,
            acc0,
            acc1,
            acc2,
            acc3,
            b_scale_cur,
            b_scale_next,
            a_regs,
            a_regs_next,
            b_sh_rd,
            s_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
            dynamic_pair_override,
            False,
        )

        self._finish_tile(
            acc0,
            acc1,
            acc2,
            acc3,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            False,
        )

    @cute.jit
    def _run_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        acc0,
        acc1,
        acc2,
        acc3,
        b_scale_cur: cute.Tensor,
        b_scale_next: cute.Tensor,
        a_regs_cur: cute.Tensor,
        a_regs_next: cute.Tensor,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        b_frag = cute.make_rmem_tensor((2, 2), Uint32)
        live_m_blocks = (block_valid_rows + Int32(15)) // Int32(16)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range(
                self.stages,
                unroll=1 if self.weight_layout_block else 0,
                unroll_full=not self.weight_layout_block,
            ):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        if cutlass.const_expr(
                            not self.weight_layout_block or kk + 1 < self.b_sh_wr_iters
                        ):
                            self._load_next_fragment_bundle(
                                b_scale_next,
                                a_regs_next,
                                smem_base,
                                tid,
                                b_sh_rd,
                                s_sh_rd,
                                a_sh_rd,
                                pipe,
                                kk,
                                tile_idx,
                                k_tiles,
                                reduce_k_tile,
                                dynamic_pair_override,
                                uses_m_block_8,
                            )

                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            a_alt_bf16_flat,
                            b_i32_flat,
                            scales_i32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            reduce_k_tile,
                            block_valid_rows,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                            expert_idx,
                            dynamic_pair_override,
                        )

                        if cutlass.const_expr(self.trellis_pair_dynamic):
                            if cutlass.const_expr(int(dynamic_pair_override) != 0):
                                self._dequant_and_accumulate_bundle(
                                    acc0,
                                    acc1,
                                    acc2,
                                    acc3,
                                    b_frag,
                                    b_scale_cur,
                                    a_regs_cur,
                                    tid,
                                    smem_base,
                                    Int32(pipe),
                                    reduce_k_tile + tile_idx,
                                    kk,
                                    trellis_lut_addr,
                                    uses_m_block_8,
                                    dynamic_pair_override,
                                    live_m_blocks,
                                )
                            else:
                                self._dequant_and_accumulate_bundle(
                                    acc0,
                                    acc1,
                                    acc2,
                                    acc3,
                                    b_frag,
                                    b_scale_cur,
                                    a_regs_cur,
                                    tid,
                                    smem_base,
                                    Int32(pipe),
                                    reduce_k_tile + tile_idx,
                                    kk,
                                    trellis_lut_addr,
                                    uses_m_block_8,
                                    0,
                                    live_m_blocks,
                                )
                        else:
                            self._dequant_and_accumulate_bundle(
                                acc0,
                                acc1,
                                acc2,
                                acc3,
                                b_frag,
                                b_scale_cur,
                                a_regs_cur,
                                tid,
                                smem_base,
                                Int32(pipe),
                                reduce_k_tile + tile_idx,
                                kk,
                                trellis_lut_addr,
                                uses_m_block_8,
                                -1,
                                live_m_blocks,
                            )
                        if cutlass.const_expr(
                            self.weight_layout_block and kk + 1 == self.b_sh_wr_iters
                        ):
                            # All inline weight reads finish before a stage is reused.
                            cute.arch.cp_async_wait_group(self.stages - 2)
                            cute.arch.sync_threads()
                            self._load_next_fragment_bundle(
                                b_scale_next,
                                a_regs_next,
                                smem_base,
                                tid,
                                b_sh_rd,
                                s_sh_rd,
                                a_sh_rd,
                                pipe,
                                kk,
                                tile_idx,
                                k_tiles,
                                reduce_k_tile,
                                dynamic_pair_override,
                                uses_m_block_8,
                            )

                        if cutlass.const_expr(uses_m_block_8):
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                        else:
                            self._copy_b_scale_register_bundle(
                                b_scale_cur, b_scale_next
                            )
                            self._copy_a_register_bundle(
                                a_regs_cur,
                                a_regs_next,
                                uses_m_block_8,
                            )
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_cur,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(0),
                    Int32(0),
                    reduce_k_tile + tile_idx,
                    dynamic_pair_override,
                )
                self._load_a_register_bundle(
                    a_regs_cur,
                    smem_base,
                    a_sh_rd,
                    Int32(0),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _dequant_and_accumulate_bundle(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        b_frag: cute.Tensor,
        b_scale_cur: cute.Tensor,
        a_regs_cur: cute.Tensor,
        tid: Int32,
        smem_base: Int32,
        pipe: Int32,
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
        trellis_lut_addr: Int64,
        uses_m_block_8: cutlass.Constexpr[bool],
        dynamic_pair_override: cutlass.Constexpr[int],
        live_m_blocks: Int32,
    ):
        if cutlass.const_expr(
            uses_m_block_8
            and self.weight_layout_trellis256_pair
            and self.trellis_rate_axis == "k"
            and (
                self.trellis_pair_kind in {"P24", "P43", "P44"}
                or (self.trellis_pair_dynamic and int(dynamic_pair_override) in (1, 2))
            )
        ):
            # FC2 assigns an entire warp/kk fragment to one side of an
            # asymmetric pair.  Select the record bitrate once, outside the
            # four unrolled N16 fragments.
            low_bits = self.trellis_pair_low_bits
            high_bits = self.trellis_pair_high_bits
            if cutlass.const_expr(
                self.trellis_pair_dynamic and int(dynamic_pair_override) == 1
            ):
                low_bits = 2
                high_bits = 4
            elif cutlass.const_expr(
                self.trellis_pair_dynamic and int(dynamic_pair_override) == 2
            ):
                low_bits = 4
                high_bits = 3
            warp_id = tid >> Int32(5)
            warp_row = warp_id // Int32(self.tb_n_warps)
            kt_local = Int32(self.b_sh_wr_iters) * warp_row + Int32(kk)
            logical_k16 = tile_idx * Int32(self.cta_k_blocks) + kt_local
            if logical_k16 < Int32(8):
                for jj in cutlass.range_constexpr(4):
                    self._scaled_dequant_b_fragment_trellis256_bits(
                        b_frag,
                        b_scale_cur[0, jj],
                        b_scale_cur[1, jj],
                        trellis_lut_addr,
                        low_bits,
                    )
                    self._mma_accumulate_m8(acc0, jj, a_regs_cur, b_frag)
            else:
                for jj in cutlass.range_constexpr(4):
                    self._scaled_dequant_b_fragment_trellis256_bits(
                        b_frag,
                        b_scale_cur[0, jj],
                        b_scale_cur[1, jj],
                        trellis_lut_addr,
                        high_bits,
                    )
                    self._mma_accumulate_m8(acc0, jj, a_regs_cur, b_frag)
            return

        for jj in cutlass.range_constexpr(4):
            if cutlass.const_expr(self.weight_layout_block):
                q0, q1, base_pair, subscale_pair = self._load_iq2_xs_fragment(
                    smem_base, tid, pipe, Int32(kk), jj
                )
                self._scaled_dequant_b_fragment_iq2_xs(
                    b_frag,
                    q0,
                    q1,
                    base_pair,
                    subscale_pair,
                    trellis_lut_addr,
                    tid,
                )
            elif cutlass.const_expr(self.weight_layout_trellis256):
                if cutlass.const_expr(self.weight_layout_trellis256_pair):
                    if cutlass.const_expr(int(dynamic_pair_override) == 0):
                        self._scaled_dequant_b_fragment_trellis256_bits(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            3,
                        )
                    elif cutlass.const_expr(int(dynamic_pair_override) == 1):
                        self._scaled_dequant_b_fragment_trellis256_p24(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            tid,
                            jj,
                            tile_idx,
                            kk,
                        )
                    elif cutlass.const_expr(int(dynamic_pair_override) == 2):
                        self._scaled_dequant_b_fragment_trellis256_p43(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            tid,
                            jj,
                            tile_idx,
                            kk,
                        )
                    else:
                        self._scaled_dequant_b_fragment_trellis256_pair(
                            b_frag,
                            b_scale_cur[0, jj],
                            b_scale_cur[1, jj],
                            trellis_lut_addr,
                            tid,
                            jj,
                            tile_idx,
                            kk,
                        )
                else:
                    self._scaled_dequant_b_fragment_trellis256(
                        b_frag,
                        b_scale_cur[0, jj],
                        b_scale_cur[1, jj],
                        trellis_lut_addr,
                    )
            else:
                q, s = self._select_b_scale_register(jj, b_scale_cur)
                self._scaled_dequant_b_fragment(b_frag, q, s)
            if cutlass.const_expr(uses_m_block_8):
                self._mma_accumulate_m8(acc0, jj, a_regs_cur, b_frag)
            elif cutlass.const_expr(self.skip_empty_m_blocks and self.cta_m_blocks > 1):
                self._mma_accumulate_large_m(acc0, a_regs_cur, 0, jj, b_frag)
                if live_m_blocks > Int32(1):
                    self._mma_accumulate_large_m(acc1, a_regs_cur, 1, jj, b_frag)
                if cutlass.const_expr(self.cta_m_blocks > 2):
                    if live_m_blocks > Int32(2):
                        self._mma_accumulate_large_m(acc2, a_regs_cur, 2, jj, b_frag)
                if cutlass.const_expr(self.cta_m_blocks > 3):
                    if live_m_blocks > Int32(3):
                        self._mma_accumulate_large_m(acc3, a_regs_cur, 3, jj, b_frag)
            else:
                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                    if cutlass.const_expr(mb == 0):
                        self._mma_accumulate_large_m(acc0, a_regs_cur, mb, jj, b_frag)
                    elif cutlass.const_expr(mb == 1):
                        self._mma_accumulate_large_m(acc1, a_regs_cur, mb, jj, b_frag)
                    elif cutlass.const_expr(mb == 2):
                        self._mma_accumulate_large_m(acc2, a_regs_cur, mb, jj, b_frag)
                    else:
                        self._mma_accumulate_large_m(acc3, a_regs_cur, mb, jj, b_frag)

    @cute.jit
    def _finish_tile(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        c_bf16_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._fold_cta_partials_m8(acc0, smem_base, tid)
        else:
            self._fold_cta_partials_large_m(
                acc0,
                acc1,
                acc2,
                acc3,
                smem_base,
                tid,
            )

        if reduce_slice_count > Int32(1):
            if cutlass.const_expr(self.weight_layout_block and uses_m_block_8):
                self._combine_iq2_xs_splitk_partials(
                    acc0,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    lock_slot,
                    reduce_slice_idx,
                    reduce_slice_count,
                    tid,
                )
            else:
                self._wait_for_reduction_turn(
                    locks_i32_flat, lock_slot, reduce_slice_idx, tid
                )
                self._combine_splitk_accumulators(
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    c_tmp_f32_flat,
                    block_valid_rows,
                    lock_slot,
                    reduce_slice_idx,
                    reduce_slice_count,
                    tid,
                    uses_m_block_8,
                )
                self._publish_reduction_turn(
                    locks_i32_flat,
                    lock_slot,
                    reduce_slice_idx == reduce_slice_count - Int32(1),
                    tid,
                )

        if reduce_slice_idx == reduce_slice_count - Int32(1):
            if cutlass.const_expr(uses_m_block_8):
                self._store_tile_m8(
                    acc0,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )
            else:
                self._store_tile_large_m(
                    acc0,
                    acc1,
                    acc2,
                    acc3,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )

    @cute.jit
    def _combine_iq2_xs_splitk_partials(
        self,
        acc,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        tid: Int32,
    ):
        cta, _, _ = cute.arch.block_idx()
        active_threads = Int32(32 * self.tb_n_warps)
        tile_elements = Int64(8 * self.tile_n)
        lock_addr = get_ptr_as_int64(locks_i32_flat, Int64(lock_slot))
        # A stripe publishes at most its first partial tile. Its final partial
        # is the reducer, so one scratch slot per CTA remains live until read.
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    offset = Int64(cta) * tile_elements + (
                        Int64(active_threads) * Int64(jj) + Int64(tid)
                    ) * Int64(4)
                    st_global_v4_f32(
                        get_ptr_as_int64(c_tmp_f32_flat, offset),
                        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    )
            cute.arch.sync_threads()
            if tid == Int32(0):
                red_add_global_release_i32(lock_addr, Int32(1))
        else:
            if tid == Int32(0):
                arrived = Int32(-1)
                while arrived != reduce_slice_count - Int32(1):
                    arrived = ld_global_acquire_i32(lock_addr)
            cute.arch.sync_threads()
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    s0 = cutlass.Float32(0.0)
                    s1 = cutlass.Float32(0.0)
                    s2 = cutlass.Float32(0.0)
                    s3 = cutlass.Float32(0.0)
                    for part in cutlass.range(reduce_slice_count - Int32(1), unroll=1):
                        source_cta = Int64(cta) + Int64(
                            reduce_slice_count - Int32(1) - part
                        )
                        offset = source_cta * tile_elements + (
                            Int64(active_threads) * Int64(jj) + Int64(tid)
                        ) * Int64(4)
                        v0, v1, v2, v3 = ld_global_v4_f32(
                            get_ptr_as_int64(c_tmp_f32_flat, offset)
                        )
                        s0 = v0 + s0
                        s1 = v1 + s1
                        s2 = v2 + s2
                        s3 = v3 + s3
                    acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s0
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s1
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s2
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] += s3
            cute.arch.sync_threads()
            if tid == Int32(0):
                st_global_i32(lock_addr, Int32(0))

    @cute.jit
    def _wait_for_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        count: Int32,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        if tid == Int32(0):
            state = Int32(-1)
            while state != count:
                state = ld_global_acquire_i32(lock_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _publish_reduction_turn(
        self,
        locks_i32_flat: cute.Tensor,
        lock_slot: Int32,
        reset,
        tid: Int32,
    ):
        lock_addr = get_ptr_as_int64(locks_i32_flat, lock_slot)
        cute.arch.sync_threads()
        if tid == Int32(0):
            if reset:
                st_global_i32(lock_addr, Int32(0))
            else:
                red_add_global_release_i32(lock_addr, Int32(1))

    @cute.jit
    def _merge_splitk_vec4(
        self,
        c_tmp_f32_flat: cute.Tensor,
        f32_off: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        if reduce_slice_idx != Int32(0):
            r0, r1, r2, r3 = ld_global_v4_f32(get_ptr_as_int64(c_tmp_f32_flat, f32_off))
            c0 = c0 + r0
            c1 = c1 + r1
            c2 = c2 + r2
            c3 = c3 + r3
        if reduce_slice_idx != reduce_slice_count - Int32(1):
            st_global_v4_f32(
                get_ptr_as_int64(c_tmp_f32_flat, f32_off),
                c0,
                c1,
                c2,
                c3,
            )
        return c0, c1, c2, c3

    @cute.jit
    def _combine_splitk_accumulators(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        c_tmp_f32_flat: cute.Tensor,
        block_valid_rows: Int32,
        lock_slot: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        tid: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        active_threads = Int32(32 * self.tb_n_warps)
        c_size_int4 = Int32((self.cta_m_blocks * 16 * self.cta_n_blocks * 16) // 4)
        c_cur_offset = lock_slot * c_size_int4
        if cutlass.const_expr(uses_m_block_8):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    k = jj * 2
                    (
                        acc0[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    ) = self._merge_splitk_slot(
                        c_tmp_f32_flat,
                        c_cur_offset,
                        active_threads,
                        Int32(k),
                        tid,
                        reduce_slice_idx,
                        reduce_slice_count,
                        acc0[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc0[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    )
        else:
            lane_row = (tid & Int32(31)) // Int32(4)
            if tid < active_threads:
                for mb in cutlass.range_constexpr(self.cta_m_blocks):
                    if cutlass.const_expr(mb == 0):
                        self._combine_splitk_accumulator_block(
                            acc0,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )
                    elif cutlass.const_expr(mb == 1):
                        self._combine_splitk_accumulator_block(
                            acc1,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )
                    elif cutlass.const_expr(mb == 2):
                        self._combine_splitk_accumulator_block(
                            acc2,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )
                    else:
                        self._combine_splitk_accumulator_block(
                            acc3,
                            mb,
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            block_valid_rows,
                            lane_row,
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                        )

    @cute.jit
    def _combine_splitk_accumulator_block(
        self,
        acc,
        mb: cutlass.Constexpr[int],
        c_tmp_f32_flat: cute.Tensor,
        c_cur_offset: Int32,
        active_threads: Int32,
        block_valid_rows: Int32,
        lane_row: Int32,
        tid: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
    ):
        for flat_j in cutlass.range_constexpr(8):
            row_valid = Int32(mb * 16) + lane_row < block_valid_rows
            if row_valid:
                (
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                ) = self._merge_splitk_slot(
                    c_tmp_f32_flat,
                    c_cur_offset,
                    active_threads,
                    Int32(mb * 8 + flat_j),
                    tid,
                    reduce_slice_idx,
                    reduce_slice_count,
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                )

    @cute.jit
    def _merge_splitk_slot(
        self,
        c_tmp_f32_flat: cute.Tensor,
        c_cur_offset: Int32,
        active_threads: Int32,
        slot: Int32,
        tid: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        c2: cutlass.Float32,
        c3: cutlass.Float32,
    ):
        int4_off = c_cur_offset + active_threads * slot + tid
        return self._merge_splitk_vec4(
            c_tmp_f32_flat,
            int4_off * Int32(4),
            reduce_slice_idx,
            reduce_slice_count,
            c0,
            c1,
            c2,
            c3,
        )

    @cute.jit
    def _load_a_registers_large_m(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        m_block: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(
                Int32(2) * kk + m_block * Int32(self.a_sh_rd_delta_i) + a_sh_rd
            ),
        )
        return ldmatrix_m8n8x4_b16(a_addr)

    @cute.jit
    def _load_a_registers_m8(
        self,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_a_off)
            + pipe * Int32(self.a_sh_stage)
            + self._activation_smem_permuted_offset(Int32(2) * kk + a_sh_rd),
        )
        return ldmatrix_m8n8x2_b16(a_addr)

    @cute.jit
    def _load_a_registers_large_m_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            a0, a1, a2, a3 = self._load_a_registers_large_m(
                smem_base,
                a_sh_rd,
                pipe,
                kk,
                Int32(mb),
            )
            regs[mb, 0] = a0
            regs[mb, 1] = a1
            regs[mb, 2] = a2
            regs[mb, 3] = a3

    @cute.jit
    def _load_a_registers_m8_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        a0, a1 = self._load_a_registers_m8(smem_base, a_sh_rd, pipe, kk)
        regs[0] = a0
        regs[1] = a1

    @cute.jit
    def _clear_a_register_bundle_large_m(self, regs: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                regs[mb, reg] = Uint32(0)

    @cute.jit
    def _clear_a_register_bundle_m8(self, regs: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            regs[reg] = Uint32(0)

    @cute.jit
    def _copy_a_register_bundle_large_m(self, dst: cute.Tensor, src: cute.Tensor):
        for mb in cutlass.range_constexpr(self.cta_m_blocks):
            for reg in cutlass.range_constexpr(4):
                dst[mb, reg] = src[mb, reg]

    @cute.jit
    def _copy_a_register_bundle_m8(self, dst: cute.Tensor, src: cute.Tensor):
        for reg in cutlass.range_constexpr(2):
            dst[reg] = src[reg]

    @cute.jit
    def _load_a_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._load_a_registers_m8_bundle(regs, smem_base, a_sh_rd, pipe, kk)
        else:
            self._load_a_registers_large_m_bundle(regs, smem_base, a_sh_rd, pipe, kk)

    @cute.jit
    def _clear_a_register_bundle(
        self,
        regs: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._clear_a_register_bundle_m8(regs)
        else:
            self._clear_a_register_bundle_large_m(regs)

    @cute.jit
    def _copy_a_register_bundle(
        self,
        dst: cute.Tensor,
        src: cute.Tensor,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._copy_a_register_bundle_m8(dst, src)
        else:
            self._copy_a_register_bundle_large_m(dst, src)

    @cute.jit
    def _load_iq2_xs_fragment(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        jj: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id // Int32(self.tb_n_warps)
        warp_n = warp_id % Int32(self.tb_n_warps)
        kt_local = Int32(self.b_sh_wr_iters) * warp_row + kk
        tc_col = lane // Int32(4)
        b_region = (
            smem_base + Int32(self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
        )
        base_region = b_region + Int32(
            self.cta_k_blocks * self.cta_n_blocks * self.block_tile_bytes
        )
        scale_region = base_region + Int32(
            _covering_count(self.cta_k_blocks * 16, self.block_k)
            * self.cta_n_blocks
            * 32
        )
        local_n16 = Int32(4) * warp_n + Int32(jj)
        base_addr = base_region + (
            (
                kt_local // Int32(self.block_k // 16) * Int32(self.cta_n_blocks)
                + local_n16
            )
            * Int32(8)
            + tc_col
        ) * Int32(4)
        base_pair = ld_shared_u32(base_addr)
        if cutlass.const_expr(self.q8):
            tile_base = (kt_local * Int32(self.cta_n_blocks) + local_n16) * Int32(16)
            addr = (
                b_region
                + (tile_base + tc_col * Int32(2)) * Int32(16)
                + (tid & Int32(3)) * Int32(2)
            )
            q0 = Uint32(ld_shared_u16_offset(addr, 0)) | (
                Uint32(ld_shared_u16_offset(addr, 8)) << 16
            )
            q1 = Uint32(ld_shared_u16_offset(addr, 16)) | (
                Uint32(ld_shared_u16_offset(addr, 24)) << 16
            )
            subscale_pair = Uint32(0)
        elif cutlass.const_expr(self.iq2_xxs):
            tile_base = (
                (kt_local // Int32(2) * Int32(2)) * Int32(self.cta_n_blocks) + local_n16
            ) * Int32(16)
            record_addr = b_region + (tile_base + tc_col * Int32(2)) * Int32(4)
            grid0, grid1 = ld_shared_v2_u32(record_addr)
            signs0, signs1 = ld_shared_v2_u32(
                record_addr + Int32(self.cta_n_blocks * self.block_tile_bytes)
            )
            q0 = iq2_xxs_descriptor_pair(grid0, signs0, kt_local % Int32(2))
            q1 = iq2_xxs_descriptor_pair(grid1, signs1, kt_local % Int32(2))
            subscale_pair = (signs0 >> 28) | ((signs1 >> 28) << 8)
        else:
            scale_addr = scale_region + (
                (kt_local // Int32(2) * Int32(self.cta_n_blocks) + local_n16)
                * Int32(16)
                + tc_col // Int32(2) * Int32(4)
            )
            shift = (tc_col % Int32(2)) * Int32(16) + (kt_local % Int32(2)) * Int32(4)
            scale_pair = ld_shared_u32(scale_addr)
            tile_base = (kt_local * Int32(self.cta_n_blocks) + local_n16) * Int32(16)
            q0, q1 = ld_shared_v2_u32(
                b_region + (tile_base + tc_col * Int32(2)) * Int32(4)
            )
            subscale_pair = scale_pair >> shift
        return q0, q1, base_pair, subscale_pair

    @cute.jit
    def _load_b_scale_registers(
        self,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.weight_layout_trellis256):
            # TRELLIS-256: no per-weight scale; the (2,4) bundle carries the 4
            # jj-tiles' pre-funnelled windows -> regs[0,jj]=win_a, regs[1,jj]=win_b.
            return self._load_b_registers_trellis256(
                smem_base, tid, pipe, kk, tile_idx, dynamic_pair_override
            )
        if cutlass.const_expr(self.weight_layout == "modelopt"):
            q0, q1, q2, q3 = self._load_b_registers_modelopt_shared(
                smem_base,
                b_sh_rd,
                pipe,
                kk,
            )
        else:
            b_addr = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + Int32(self.b_sh_stride) * kk
                + b_sh_rd,
            )
            q0, q1, q2, q3 = ld_shared_v4_u32(b_addr)

        warp_id = tid // Int32(32)
        warp_row = warp_id // Int32(self.tb_n_warps)
        cur_group_id = Int32(self.b_sh_wr_iters) * warp_row + kk
        if cutlass.const_expr(self.scale_k32):
            scale_group_id = cur_group_id // Int32(2)
        else:
            scale_group_id = cur_group_id
        s_addr = (
            smem_base
            + Int32(self.sh_s_off * 16)
            + pipe * Int32(self.s_sh_stage * 16)
            + (s_sh_rd + scale_group_id * Int32(2 * self.s_sh_stride)) * Int32(8)
        )
        if cutlass.const_expr(self.native_nvfp4_scales):
            words = cute.make_rmem_tensor((2, 4), Uint32)
            local_n = (warp_id % Int32(self.tb_n_warps)) * Int32(64) + (
                (tid & Int32(31)) // Int32(4)
            )
            stage_base = (
                smem_base
                + Int32(self.sh_s_off * 16)
                + pipe * Int32(self.s_sh_stage * 16)
            )
            vector_n = (
                128
                if self.tile_n % 128 == 0 and self.source_n_rotation % 128 == 0
                else 64
            )
            vector_words = vector_n // 32
            for i in cutlass.range_constexpr(4):
                n0 = local_n + Int32(i * 8)
                word0, word1 = ld_shared_v2_u32(
                    stage_base
                    + (
                        (scale_group_id // Int32(4)) * Int32(self.tile_n)
                        + (n0 // Int32(vector_n) * Int32(32) + n0 % Int32(32))
                        * Int32(vector_words)
                        + n0 % Int32(vector_n) // Int32(32)
                    )
                    * Int32(4)
                )
                words[0, i] = word0
                words[1, i] = word1
            byte = Uint32(scale_group_id % Int32(4))
            s_pack0 = _gather_native_scale_bytes(
                words[0, 0], words[0, 1], words[0, 2], words[0, 3], byte
            )
            s_pack1 = _gather_native_scale_bytes(
                words[1, 0], words[1, 1], words[1, 2], words[1, 3], byte
            )
        else:
            s_pack0, s_pack1 = ld_shared_v2_u32(s_addr)
        s0, s1 = self._dequant_scale_x4_to_elem2x2(s_pack0)
        s2, s3 = self._dequant_scale_x4_to_elem2x2(s_pack1)
        return q0, q1, q2, q3, s0, s1, s2, s3

    @cute.jit
    def _load_b_scale_register_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.weight_layout_block):
            return
        q0, q1, q2, q3, s0, s1, s2, s3 = self._load_b_scale_registers(
            smem_base,
            tid,
            b_sh_rd,
            s_sh_rd,
            pipe,
            kk,
            tile_idx,
            dynamic_pair_override,
        )
        regs[0, 0] = q0
        regs[0, 1] = q1
        regs[0, 2] = q2
        regs[0, 3] = q3
        regs[1, 0] = s0
        regs[1, 1] = s1
        regs[1, 2] = s2
        regs[1, 3] = s3

    @cute.jit
    def _clear_b_scale_register_bundle(self, regs: cute.Tensor):
        for row in cutlass.range_constexpr(self.b_bundle_rows):
            for col in cutlass.range_constexpr(4):
                regs[row, col] = Uint32(0)

    @cute.jit
    def _copy_b_scale_register_bundle(self, dst: cute.Tensor, src: cute.Tensor):
        for row in cutlass.range_constexpr(self.b_bundle_rows):
            for col in cutlass.range_constexpr(4):
                dst[row, col] = src[row, col]

    @cute.jit
    def _select_b_scale_register(self, jj: cutlass.Constexpr[int], regs: cute.Tensor):
        return regs[0, jj], regs[1, jj]

    @cute.jit
    def _load_next_fragment_bundle(
        self,
        b_scale_next: cute.Tensor,
        a_regs_next: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        s_sh_rd: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        self._clear_b_scale_register_bundle(b_scale_next)
        self._clear_a_register_bundle(a_regs_next, uses_m_block_8)

        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    reduce_k_tile + tile_idx,
                    dynamic_pair_override,
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    uses_m_block_8,
                )
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                self._load_b_scale_register_bundle(
                    b_scale_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    s_sh_rd,
                    Int32((pipe + 1) % self.stages),
                    Int32(0),
                    reduce_k_tile + next_tile,
                    dynamic_pair_override,
                )
                self._load_a_register_bundle(
                    a_regs_next,
                    smem_base,
                    a_sh_rd,
                    Int32((pipe + 1) % self.stages),
                    Int32(0),
                    uses_m_block_8,
                )

    @cute.jit
    def _scaled_dequant_b_fragment(self, frag: cute.Tensor, q: Uint32, s: Uint32):
        if cutlass.const_expr(self.native_nvfp4_scales and self.is_fp16):
            b0_0, b0_1, b1_0, b1_1 = _dequant_native_fp4_word(q, self.is_fp16)
        else:
            bq1 = q
            bq0 = bq1 << Uint32(8)
            b0_0, b0_1 = self._dequant_e2m1x4_to_elem2x2(bq0)
            b1_0, b1_1 = self._dequant_e2m1x4_to_elem2x2(bq1)
        s_lane0 = bfloat2_broadcast_lane(s, Int32(0))
        s_lane1 = bfloat2_broadcast_lane(s, Int32(1))
        b0_0 = self._elem2_mul(b0_0, s_lane0)
        b0_1 = self._elem2_mul(b0_1, s_lane0)
        b1_0 = self._elem2_mul(b1_0, s_lane1)
        b1_1 = self._elem2_mul(b1_1, s_lane1)
        frag[0, 0] = b0_0
        frag[0, 1] = b0_1
        frag[1, 0] = b1_0
        frag[1, 1] = b1_1

    @cute.jit
    def _scaled_dequant_b_fragment_iq2_xs(
        self,
        frag: cute.Tensor,
        q_row0: Uint32,
        q_row1: Uint32,
        base_pair: Uint32,
        subscale_pair: Uint32,
        execution_lut_addr: Int64,
        tid: Int32,
    ):
        if cutlass.const_expr(self.q8):
            b0_0 = q8_0_pair_to_bf16x2(q_row0, base_pair)
            b0_1 = q8_0_pair_to_bf16x2(q_row0 >> 16, base_pair)
            b1_0 = q8_0_pair_to_bf16x2(q_row1, base_pair >> 16)
            b1_1 = q8_0_pair_to_bf16x2(q_row1 >> 16, base_pair >> 16)
        else:
            pair_byte_offset = (tid & Int32(3)) * Int32(2)
            b0_0, b0_1, b1_0, b1_1 = packed_decode_iq2_xs_to_bfloat2x4(
                q_row0,
                q_row1,
                base_pair,
                subscale_pair,
                execution_lut_addr,
                pair_byte_offset,
                shared_lut=self.iq2_xs_smem_lut,
                selector_lut=True,
            )
        frag[0, 0] = b0_0
        frag[0, 1] = b0_1
        frag[1, 0] = b1_0
        frag[1, 1] = b1_1

    @cute.jit
    def _trellis256_lane_geom_bits(
        self,
        lane: Int32,
        weight_offset: cutlass.Constexpr[int],
        weight_count: cutlass.Constexpr[int],
        bits: cutlass.Constexpr[int],
    ):
        return _trellis_ring_lane_geom_bits(lane, weight_offset, weight_count, bits)

    @cute.jit
    def _trellis256_lane_geom(
        self,
        lane: Int32,
        weight_offset: cutlass.Constexpr[int],
        weight_count: cutlass.Constexpr[int],
    ):
        return self._trellis256_lane_geom_bits(
            lane, weight_offset, weight_count, self.trellis_bits
        )

    @cute.jit
    def _trellis_funnel256(self, a: Uint32, b: Uint32, s2: Int32):
        merged = (Int64(a) << Int64(32)) | Int64(b)
        return Uint32(merged >> Int64(s2))

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_bits(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        bits: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.trellis_codebook == "mcg"):
            if cutlass.const_expr(int(bits) == 6):
                if cutlass.const_expr(self.is_fp16):
                    o0, o1, o2, o3 = packed_dequant_trellis_stream_to_half2x4(
                        win_a, win_b, int(bits)
                    )
                else:
                    o0, o1, o2, o3 = packed_dequant_trellis_stream_to_bfloat2x4(
                        win_a, win_b, int(bits)
                    )
            elif cutlass.const_expr(self.is_fp16):
                o0, o1, o2, o3 = packed_dequant_trellis_to_half2x4(
                    win_a, win_b, int(bits)
                )
            else:
                o0, o1, o2, o3 = packed_dequant_trellis_to_bfloat2x4(
                    win_a, win_b, int(bits)
                )
        elif cutlass.const_expr(self.trellis_codebook == LUT_FP16):
            if cutlass.const_expr(self.is_fp16):
                o0, o1, o2, o3 = packed_decode_lut_fp16_to_half2x4(
                    win_a, win_b, trellis_lut_addr, int(bits)
                )
            else:
                o0, o1, o2, o3 = packed_decode_lut_fp16_to_bfloat2x4(
                    win_a, win_b, trellis_lut_addr, int(bits)
                )
        else:
            if cutlass.const_expr(self.trellis_direct_lut):
                e_lo, e_hi = packed_decode_lut_e4m3_direct_to_e4m3x8(
                    win_a,
                    win_b,
                    trellis_lut_addr,
                    int(bits),
                    in_shared=self.lut_e4m3_smem,
                )
            else:
                e_lo, e_hi = packed_decode_lut_e4m3_to_e4m3x8(
                    win_a,
                    win_b,
                    trellis_lut_addr,
                    int(bits),
                    value_table_in_shared=self.lut_e4m3_smem,
                )
            if cutlass.const_expr(self.is_fp16):
                o0, o1 = fp8x4_e4m3_to_half2x2(e_lo)
                o2, o3 = fp8x4_e4m3_to_half2x2(e_hi)
            else:
                o0, o1 = fp8x4_e4m3_to_bfloat2x2_native_sm120(e_lo)
                o2, o3 = fp8x4_e4m3_to_bfloat2x2_native_sm120(e_hi)
        frag[0, 0] = o0
        frag[0, 1] = o1
        frag[1, 0] = o2
        frag[1, 1] = o3

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
    ):
        self._scaled_dequant_b_fragment_trellis256_bits(
            frag, win_a, win_b, trellis_lut_addr, self.trellis_bits
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_pair(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
    ):
        # Dynamic pair kernels dispatch to bitrate-specific arms before entering
        # the MMA loop. This fallback serves compile-time static pair rates.
        if cutlass.const_expr(self.trellis_pair_kind == "P33"):
            self._scaled_dequant_b_fragment_trellis256_bits(
                frag, win_a, win_b, trellis_lut_addr, 3
            )
            return
        self._scaled_dequant_b_fragment_trellis256_pair_rates(
            frag,
            win_a,
            win_b,
            trellis_lut_addr,
            tid,
            jj,
            tile_idx,
            kk,
            self.trellis_pair_low_bits,
            self.trellis_pair_high_bits,
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_p24(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
    ):
        self._scaled_dequant_b_fragment_trellis256_pair_rates(
            frag, win_a, win_b, trellis_lut_addr, tid, jj, tile_idx, kk, 2, 4
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_p43(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
    ):
        self._scaled_dequant_b_fragment_trellis256_pair_rates(
            frag, win_a, win_b, trellis_lut_addr, tid, jj, tile_idx, kk, 4, 3
        )

    @cute.jit
    def _scaled_dequant_b_fragment_trellis256_pair_rates(
        self,
        frag: cute.Tensor,
        win_a: Uint32,
        win_b: Uint32,
        trellis_lut_addr: Int64,
        tid: Int32,
        jj: cutlass.Constexpr[int],
        tile_idx: Int32,
        kk: cutlass.Constexpr[int],
        low_bits: cutlass.Constexpr[int],
        high_bits: cutlass.Constexpr[int],
    ):
        warp_id = tid >> Int32(5)
        if cutlass.const_expr(self.trellis_rate_axis == "n"):
            # The MMA work is striped LLHH inside each four-N16 warp slab so
            # every warp executes two K2 and two K4 fragments.  The epilogue
            # scatters these accumulator fragments back to the reference
            # record0 || record1 channel order before the outer Hadamard.
            if cutlass.const_expr(jj < 2):
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, low_bits
                )
            else:
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, high_bits
                )
        else:
            warp_row = warp_id // Int32(self.tb_n_warps)
            kt_local = Int32(self.b_sh_wr_iters) * warp_row + Int32(kk)
            logical_k16 = tile_idx * Int32(self.cta_k_blocks) + kt_local
            if logical_k16 < Int32(8):
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, low_bits
                )
            else:
                self._scaled_dequant_b_fragment_trellis256_bits(
                    frag, win_a, win_b, trellis_lut_addr, high_bits
                )

    @cute.jit
    def _load_trellis256_pair_tile_windows(
        self,
        b_region: Int32,
        tile_base_u32: Int32,
        lane: Int32,
        bits: cutlass.Constexpr[int],
    ):
        ia, ib, s2, _ = self._trellis256_lane_geom_bits(lane, 0, 8, bits)
        a = ld_shared_u32(b_region + (tile_base_u32 + ia) * Int32(4))
        b = ld_shared_u32(b_region + (tile_base_u32 + ib) * Int32(4))
        return (
            self._trellis_funnel256(a, b, s2),
            self._trellis_funnel256(a, b, s2 + Int32(4 * int(bits))),
        )

    @cute.jit
    def _load_b_registers_trellis256_pair(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        regs = cute.make_rmem_tensor((2, 4), Uint32)
        if cutlass.const_expr(
            self.trellis_pair_kind == "P24"
            or (self.trellis_pair_dynamic and int(dynamic_pair_override) == 1)
        ):
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 2, 4
            )
        elif cutlass.const_expr(
            self.trellis_pair_kind == "P43"
            or (self.trellis_pair_dynamic and int(dynamic_pair_override) == 2)
        ):
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 4, 3
            )
        elif cutlass.const_expr(self.trellis_pair_kind == "P44"):
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 4, 4
            )
        else:
            self._load_b_registers_trellis256_pair_bits(
                regs, smem_base, tid, pipe, kk, tile_idx, 3, 3
            )
        return (
            regs[0, 0],
            regs[0, 1],
            regs[0, 2],
            regs[0, 3],
            regs[1, 0],
            regs[1, 1],
            regs[1, 2],
            regs[1, 3],
        )

    @cute.jit
    def _load_b_registers_trellis256_pair_bits(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        low_bits: cutlass.Constexpr[int],
        high_bits: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id // Int32(self.tb_n_warps)
        w_n = warp_id % Int32(self.tb_n_warps)
        kt_local = Int32(self.b_sh_wr_iters) * warp_row + kk
        b_region = (
            smem_base + Int32(self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
        )
        wa = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        wb = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        if cutlass.const_expr(self.trellis_rate_axis == "n"):
            pair_span_u32 = 8 * 8 * (low_bits + high_bits)
            high_base_u32 = 8 * 8 * low_bits
            for jj in cutlass.range_constexpr(4):
                # Reference storage remains record0 || record1.  MMA register
                # assignment is LLHH within a warp; the epilogue restores the
                # original contiguous output-channel order.
                if cutlass.const_expr(jj < 2):
                    record_n16 = Int32(2) * w_n + Int32(jj)
                    tile_base = kt_local * Int32(pair_span_u32) + record_n16 * Int32(
                        8 * low_bits
                    )
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, low_bits
                    )
                else:
                    record_n16 = Int32(2) * w_n + Int32(jj - 2)
                    tile_base = (
                        kt_local * Int32(pair_span_u32)
                        + Int32(high_base_u32)
                        + record_n16 * Int32(8 * high_bits)
                    )
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, high_bits
                    )
        else:
            # Each K16 row owns a max-K4 shared slot.  K2/K3 rows occupy only
            # the compact prefix; the remainder is intentionally untouched.
            logical_k16 = tile_idx * Int32(self.cta_k_blocks) + kt_local
            tile_slot_u32 = self.cta_n_blocks * 8 * 4
            kt_base_u32 = kt_local * Int32(tile_slot_u32)
            for jj in cutlass.range_constexpr(4):
                local_n16 = Int32(4) * w_n + Int32(jj)
                tile_base = Int32(0)
                # CuTe must specialize the payload width before evaluating the
                # device-side row predicate, so these equal bodies cannot be
                # combined into one Python boolean expression.
                if cutlass.const_expr(int(low_bits) == int(high_bits)):  # noqa: SIM114
                    tile_base = kt_base_u32 + local_n16 * Int32(8 * low_bits)
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, low_bits
                    )
                elif logical_k16 < Int32(8):
                    tile_base = kt_base_u32 + local_n16 * Int32(8 * low_bits)
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, low_bits
                    )
                else:
                    tile_base = kt_base_u32 + local_n16 * Int32(8 * high_bits)
                    wa[jj], wb[jj] = self._load_trellis256_pair_tile_windows(
                        b_region, tile_base, lane, high_bits
                    )
        for jj in cutlass.range_constexpr(4):
            regs[0, jj] = wa[jj]
            regs[1, jj] = wb[jj]

    @cute.jit
    def _load_b_registers_trellis256(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: Int32,
        tile_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.weight_layout_trellis256_pair):
            return self._load_b_registers_trellis256_pair(
                smem_base,
                tid,
                pipe,
                kk,
                tile_idx,
                dynamic_pair_override,
            )
        lane = tid & Int32(31)
        warp_id = tid >> Int32(5)
        warp_row = warp_id // Int32(self.tb_n_warps)
        w_n = warp_id % Int32(self.tb_n_warps)
        kt_local = Int32(self.b_sh_wr_iters) * warp_row + kk
        b_region = (
            smem_base + Int32(self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage_bytes)
        )
        tile_u32 = 8 * self.trellis_bits
        base_u32 = (kt_local * Int32(self.cta_n_blocks) + Int32(4) * w_n) * Int32(
            tile_u32
        )
        wa = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        wb = [Uint32(0), Uint32(0), Uint32(0), Uint32(0)]
        if cutlass.const_expr(self.trellis_bits <= 4):
            ia, ib, s2, _ = self._trellis256_lane_geom(lane, 0, 8)
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                a = ld_shared_u32(b_region + (tbase + ia) * Int32(4))
                b = ld_shared_u32(b_region + (tbase + ib) * Int32(4))
                wa[jj] = self._trellis_funnel256(a, b, s2)
                wb[jj] = self._trellis_funnel256(
                    a, b, s2 + Int32(4 * self.trellis_bits)
                )
        elif cutlass.const_expr(self.trellis_bits == 5):
            ib0, ib1, sb, _ = self._trellis256_lane_geom(lane, 0, 4)
            ia0, ia1, sa, _ = self._trellis256_lane_geom(lane, 4, 4)
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                b0 = ld_shared_u32(b_region + (tbase + ib0) * Int32(4))
                b1 = ld_shared_u32(b_region + (tbase + ib1) * Int32(4))
                a0 = ld_shared_u32(b_region + (tbase + ia0) * Int32(4))
                a1 = ld_shared_u32(b_region + (tbase + ia1) * Int32(4))
                wb[jj] = self._trellis_funnel256(b0, b1, sb)
                wa[jj] = self._trellis_funnel256(a0, a1, sa)
        else:
            i0, i2, s2, delta = self._trellis256_lane_geom(lane, 0, 8)
            i1 = i0 + Int32(1)
            i1 = i1 - Int32(tile_u32) * (i1 >= Int32(tile_u32)).to(Int32)
            for jj in cutlass.range_constexpr(4):
                tbase = base_u32 + Int32(jj * tile_u32)
                z0 = ld_shared_u32(b_region + (tbase + i0) * Int32(4))
                z1 = ld_shared_u32(b_region + (tbase + i1) * Int32(4))
                z2 = ld_shared_u32(b_region + (tbase + i2) * Int32(4))
                wa[jj], wb[jj] = trellis_align_stream_u32x2(z0, z1, z2, s2, delta)
        return wa[0], wa[1], wa[2], wa[3], wb[0], wb[1], wb[2], wb[3]

    @cute.jit
    def _mma_accumulate_m8(
        self,
        acc,
        jj: cutlass.Constexpr[int],
        a_regs: cute.Tensor,
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_rhs_fragments_as_mma_a_m16n8k16_f32(
            acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            b_frag[0, 0],
            b_frag[1, 0],
            b_frag[0, 1],
            b_frag[1, 1],
            a_regs[0],
            a_regs[1],
        )
        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d0
        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d1
        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d2
        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d3

    @cute.jit
    def _mma_accumulate_large_m(
        self,
        acc,
        a_regs: cute.Tensor,
        mb: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
        b_frag: cute.Tensor,
    ):
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[(jj * 8) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[0, 0],
            b_frag[0, 1],
        )
        acc[(jj * 8) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d0
        acc[(jj * 8 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d1
        acc[(jj * 8 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d2
        acc[(jj * 8 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d3
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[(jj * 8 + 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 4) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 5) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 5) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 6) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 6) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            acc[(jj * 8 + 7) // _SCALAR_ACC_FRAGMENT_WIDTH][
                (jj * 8 + 7) % _SCALAR_ACC_FRAGMENT_WIDTH
            ],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[1, 0],
            b_frag[1, 1],
        )
        acc[(jj * 8 + 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 4) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d0
        acc[(jj * 8 + 5) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 5) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d1
        acc[(jj * 8 + 6) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 6) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d2
        acc[(jj * 8 + 7) // _SCALAR_ACC_FRAGMENT_WIDTH][
            (jj * 8 + 7) % _SCALAR_ACC_FRAGMENT_WIDTH
        ] = d3

    @cute.jit
    def _source_n_from_logical(self, logical_n: Int32) -> Int32:
        source_n = logical_n
        if cutlass.const_expr(self.source_n_rotation != 0):
            source_n += Int32(self.source_n_rotation)
            if source_n >= Int32(self.size_n):
                source_n -= Int32(self.size_n)
        return source_n

    @cute.jit
    def _modelopt_smem_permuted_offset(self, byte_offset: Int32) -> Int32:
        # Distribute row gathers across banks while preserving 16-byte alignment.
        return byte_offset ^ ((byte_offset >> Int32(3)) & Int32(0x70))

    @cute.jit
    def _stage_modelopt_scales(
        self,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        tile_idx: Int32,
    ):
        words_per_row = self.s_tb_groups // 4
        vector_n = (
            128 if self.tile_n % 128 == 0 and self.source_n_rotation % 128 == 0 else 64
        )
        vector_words = vector_n // 32
        vectors_per_k_word = self.tile_n // vector_words
        scale_cols = _covering_count(self.size_k // 16, 4) * 4
        expert_words = _covering_count(self.size_n, 128) * 128 * scale_cols // 4
        stage_base = (
            smem_base + Int32(self.sh_s_off * 16) + pipe * Int32(self.s_sh_stage * 16)
        )
        for i in cutlass.range_constexpr(
            _covering_count(self.s_sh_stage * 4 // vector_words, self.cta_threads)
        ):
            vector_idx = tid + Int32(i * self.cta_threads)
            if vector_idx < Int32(self.s_sh_stage * 4 // vector_words):
                n_group = vector_idx % Int32(vectors_per_k_word)
                local_n = (n_group // Int32(32)) * Int32(vector_n) + n_group % Int32(32)
                local_k_word = vector_idx // Int32(vectors_per_k_word)
                k_word = tile_idx * Int32(words_per_row) + local_k_word
                logical_n = output_n_tile * Int32(self.tile_n) + local_n
                source_n = self._source_n_from_logical(logical_n)
                offset = (
                    Int64(expert_idx) * Int64(expert_words)
                    + Int64(source_n // Int32(128)) * Int64(scale_cols * 32)
                    + Int64(k_word) * Int64(128)
                    + Int64(source_n % Int32(32)) * Int64(4)
                    + Int64((source_n % Int32(128)) // Int32(32))
                )
                if logical_n < Int32(self.size_n) and k_word < Int32(scale_cols // 4):
                    if cutlass.const_expr(vector_words == 4):
                        cp_async4_shared_global(
                            stage_base + vector_idx * Int32(16),
                            get_ptr_as_int64(scales_i32_flat, offset),
                        )
                    else:
                        cp_async_u64_shared_global(
                            stage_base + vector_idx * Int32(8),
                            get_ptr_as_int64(scales_i32_flat, offset),
                        )
                else:
                    for word in cutlass.range_constexpr(vector_words):
                        st_shared_u32(
                            stage_base
                            + (vector_idx * Int32(vector_words) + Int32(word))
                            * Int32(4),
                            Uint32(0),
                        )

    @cute.jit
    def _stage_b_tile_modelopt_native(
        self,
        b_u8_flat: cute.Tensor,
        smem_addr: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        tile_idx: Int32,
        local_int4: Int32,
    ):
        chunks_per_row = Int32(self.tile_k // 32)
        local_n = local_int4 // chunks_per_row
        local_k_vec = local_int4 - local_n * chunks_per_row
        logical_n = output_n_tile * Int32(self.tile_n) + local_n
        source_n = self._source_n_from_logical(logical_n)
        packed_cols = Int32(self.size_k // 2)
        tile_byte = tile_idx * Int32(self.tile_k // 2) + local_k_vec * Int32(16)
        byte_offset = (
            Int64(expert_idx) * Int64(self.size_n * (self.size_k // 2))
            + Int64(source_n) * Int64(packed_cols)
            + Int64(tile_byte)
        )
        if cutlass.const_expr(self.has_logical_tail):
            valid_n = logical_n < Int32(self.size_n)
            valid_bytes = packed_cols - tile_byte
            if valid_bytes > Int32(16):
                valid_bytes = Int32(16)
            if (
                valid_n
                and valid_bytes >= Int32(16)
                and cutlass.const_expr(not self.has_scale_k_tail)
            ):
                cp_async4_shared_global(
                    smem_addr,
                    get_ptr_as_int64(b_u8_flat, byte_offset),
                )
            else:
                v0 = Uint32(0)
                v1 = Uint32(0)
                v2 = Uint32(0)
                v3 = Uint32(0)
                if valid_n and valid_bytes > Int32(0):
                    src = get_ptr_as_int64(b_u8_flat, byte_offset)
                    if valid_bytes >= Int32(4):
                        v0 = ld_global_nc_u32(src)
                    if valid_bytes >= Int32(8):
                        v1 = ld_global_nc_u32(src + Int64(4))
                    if valid_bytes >= Int32(12):
                        v2 = ld_global_nc_u32(src + Int64(8))
                    if valid_bytes >= Int32(16):
                        v3 = ld_global_nc_u32(src + Int64(12))
                st_shared_v4_u32(smem_addr, v0, v1, v2, v3)
        else:
            cp_async4_shared_global(
                smem_addr,
                get_ptr_as_int64(b_u8_flat, byte_offset),
            )

    @cute.jit
    def _load_modelopt_shared_packed_word_for_lane(
        self,
        smem_base: Int32,
        pipe: Int32,
        n_tile: Int32,
        k_tile: Int32,
        warp_id: Int32,
        tc_col: Int32,
        tc_row: Int32,
    ) -> Uint32:
        local_n = n_tile * Int32(64) + warp_id * Int32(16) + tc_col
        byte_offset = local_n * Int32(self.tile_k // 2) + k_tile * Int32(8)
        stage_base = (
            smem_base + Int32(self.sh_b_off * 16) + pipe * Int32(self.b_sh_stage * 16)
        )
        a0, a1 = ld_shared_v2_u32(
            stage_base + self._modelopt_smem_permuted_offset(byte_offset)
        )
        b0, b1 = ld_shared_v2_u32(
            stage_base
            + self._modelopt_smem_permuted_offset(
                byte_offset + Int32(8 * (self.tile_k // 2))
            )
        )
        return _pack_modelopt_words(a0, a1, b0, b1, Uint32(tc_row // Int32(2)))

    @cute.jit
    def _load_b_registers_modelopt_shared(
        self,
        smem_base: Int32,
        b_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        packed_word_index = (Int32(self.b_sh_stride) * kk + b_sh_rd) * Int32(4)
        words_per_k_tile = Int32((self.tile_n // 64) * 128)
        k_tile = packed_word_index // words_per_k_tile
        pos_in_k_tile = packed_word_index - k_tile * words_per_k_tile
        n_tile = pos_in_k_tile // Int32(128)
        pos = pos_in_k_tile - n_tile * Int32(128)
        th_id = pos // Int32(4)
        tc_col = th_id // Int32(4)
        tc_row = (th_id - tc_col * Int32(4)) * Int32(2)
        q0 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(0), tc_col, tc_row
        )
        q1 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(1), tc_col, tc_row
        )
        q2 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(2), tc_col, tc_row
        )
        q3 = self._load_modelopt_shared_packed_word_for_lane(
            smem_base, pipe, n_tile, k_tile, Int32(3), tc_col, tc_row
        )
        return q0, q1, q2, q3

    @cute.jit
    def _stage_k_tile_async(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        tile_idx: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        for i in cutlass.range_constexpr(self.a_sh_wr_iters):
            row = a_rows_per_iter * Int32(i) + a_gl_rd_row
            route_index = Int32(0)
            if row < Int32(self.moe_block_size):
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_rd_route_off * 16) + row * Int32(4)
                )
            a_int4 = (
                Int64(route_index) * Int64(a_gl_stride)
                + Int64(tile_idx) * Int64(self.a_gl_rd_delta_o)
                + Int64(a_gl_rd_col0)
            )
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off)
                + pipe * Int32(self.a_sh_stage)
                + self._activation_smem_permuted_offset(
                    Int32(i * self.a_sh_wr_delta) + a_sh_wr
                ),
            )
            a_src = get_ptr_as_int64(a_bf16_flat, a_int4 * Int64(8))
            if cutlass.const_expr(self.dual_a):
                # Projection-major FC1 validates that each CTA N tile is wholly
                # gate or wholly up.  Select the matching pre-rotated A operand
                # with the same logical projection boundary used by B staging.
                # The selected source still lands in the one existing A SMEM
                # stage, so this adds neither shared memory nor MMA work.
                if output_n_tile >= Int32(self.n_tiles // 2):
                    a_src = get_ptr_as_int64(a_alt_bf16_flat, a_int4 * Int32(8))
            if cutlass.const_expr(self.has_k_tile_tail):
                a_k_int4 = tile_idx * Int32(self.a_gl_rd_delta_o) + a_gl_rd_col0
                if row < block_valid_rows and a_k_int4 < a_gl_stride:
                    cp_async4_shared_global(
                        a_dst,
                        a_src,
                    )
                else:
                    st_shared_v4_u32(a_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0))
            else:
                cp_async4_shared_global_pred(
                    a_dst,
                    a_src,
                    (row < block_valid_rows).to(Int32),
                )

        if cutlass.const_expr(self.weight_layout_block):
            n16_total = self.size_n // 16
            chunks_per_k16 = self.cta_n_blocks * (self.block_tile_bytes // 16)
            total_chunks = self.cta_k_blocks * chunks_per_k16
            b_region = (
                smem_base
                + Int32(self.sh_b_off * 16)
                + pipe * Int32(self.b_sh_stage_bytes)
            )
            packed_tile = Int64(0)
            if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
                packed_tile = (
                    Int64(expert_idx) * Int64(self.size_n // 64) + Int64(output_n_tile)
                ) * Int64(self.size_k // 128) + Int64(tile_idx)
            for i in cutlass.range_constexpr(
                _covering_count(total_chunks, self.cta_threads)
            ):
                chunk = Int32(i * self.cta_threads) + tid
                local_tile = chunk // Int32(self.block_tile_bytes // 16)
                local_k16 = local_tile // Int32(self.cta_n_blocks)
                local_n16 = local_tile % Int32(self.cta_n_blocks)
                tile_chunk = chunk % Int32(self.block_tile_bytes // 16)
                global_k16 = tile_idx * Int32(self.cta_k_blocks) + local_k16
                global_n16 = output_n_tile * Int32(self.cta_n_blocks) + local_n16
                if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
                    source_u32 = packed_tile * Int64(
                        32 * self.block_tile_bytes // 4
                    ) + Int64(chunk) * Int64(4)
                else:
                    source_u32 = self._iq2_xs_descriptor_offset(
                        expert_idx, global_k16, global_n16
                    ) + Int64(tile_chunk) * Int64(4)
                cp_async4_shared_global_pred(
                    b_region + chunk * Int32(16),
                    get_ptr_as_int64(b_i32_flat, source_u32),
                    (chunk < Int32(total_chunks)).to(Int32),
                )
            metadata_addr = get_ptr_as_int64(scales_i32_flat, Int64(0))
            expert_count = Int64(cute.size(b_i32_flat)) // Int64(
                self.size_k * self.size_n // (4 * self.block_pack_factor)
            )
            base_plane_bytes = expert_count * Int64(
                self.size_k // self.block_k * self.size_n * 2
            )
            base_chunks = (
                _covering_count(self.cta_k_blocks * 16, self.block_k)
                * self.cta_n_blocks
                * 2
            )
            base_region = b_region + Int32(
                self.cta_k_blocks * self.cta_n_blocks * self.block_tile_bytes
            )
            scale_region = base_region + Int32(base_chunks * 16)
            for i in cutlass.range_constexpr(
                _covering_count(base_chunks, self.cta_threads)
            ):
                chunk = Int32(i * self.cta_threads) + tid
                if chunk < Int32(base_chunks):
                    if cutlass.const_expr(
                        not self.q8 and self.tile_n == 64 and self.tile_k == 128
                    ):
                        base_tile = (
                            Int64(expert_idx) * Int64(self.size_k // self.block_k)
                            + Int64(tile_idx // Int32(2))
                        ) * Int64(self.size_n // 64) + Int64(output_n_tile)
                        base_offset = base_tile * Int64(128) + Int64(chunk) * Int64(16)
                    else:
                        local_tile = chunk // Int32(2)
                        global_k256 = tile_idx * Int32(self.cta_k_blocks) // Int32(
                            self.block_k // 16
                        ) + local_tile // Int32(self.cta_n_blocks)
                        global_n16 = output_n_tile * Int32(
                            self.cta_n_blocks
                        ) + local_tile % Int32(self.cta_n_blocks)
                        block = (
                            Int64(expert_idx) * Int64(self.size_k // self.block_k)
                            + Int64(global_k256)
                        ) * Int64(n16_total) + Int64(global_n16)
                        base_offset = block * Int64(32) + Int64(
                            chunk % Int32(2)
                        ) * Int64(16)
                    cp_async4_shared_global(
                        base_region + chunk * Int32(16),
                        metadata_addr + base_offset,
                    )
            scale_chunks = (
                0
                if self.iq2_xxs or self.q8
                else self.cta_k_blocks // 2 * self.cta_n_blocks
            )
            for i in cutlass.range_constexpr(
                _covering_count(scale_chunks, self.cta_threads)
            ):
                chunk = Int32(i * self.cta_threads) + tid
                if chunk < Int32(scale_chunks):
                    if cutlass.const_expr(self.tile_n == 64 and self.tile_k == 128):
                        scale_chunk = packed_tile * Int64(16) + Int64(chunk)
                    else:
                        global_k32 = tile_idx * Int32(
                            self.cta_k_blocks // 2
                        ) + chunk // Int32(self.cta_n_blocks)
                        global_n16 = output_n_tile * Int32(
                            self.cta_n_blocks
                        ) + chunk % Int32(self.cta_n_blocks)
                        scale_tile = (
                            Int64(expert_idx) * Int64(self.size_n // 64)
                            + Int64(global_n16 // Int32(4))
                        ) * Int64(self.size_k // 128) + Int64(global_k32 // Int32(4))
                        scale_chunk = (
                            scale_tile * Int64(16)
                            + Int64(global_k32 % Int32(4)) * Int64(4)
                            + Int64(global_n16 % Int32(4))
                        )
                    cp_async4_shared_global(
                        scale_region + chunk * Int32(16),
                        metadata_addr + base_plane_bytes + scale_chunk * Int64(16),
                    )

        if cutlass.const_expr(self.weight_layout_trellis256):
            t256_n16 = self.size_n // 16
            if cutlass.const_expr(self.weight_layout_trellis256_pair):
                low_bits = 3
                high_bits = 3
                if cutlass.const_expr(
                    self.trellis_pair_kind == "P24"
                    or (self.trellis_pair_dynamic and int(dynamic_pair_override) == 1)
                ):
                    low_bits = 2
                    high_bits = 4
                elif cutlass.const_expr(
                    self.trellis_pair_kind == "P43"
                    or (self.trellis_pair_dynamic and int(dynamic_pair_override) == 2)
                ):
                    low_bits = 4
                    high_bits = 3
                elif cutlass.const_expr(self.trellis_pair_kind == "P44"):
                    low_bits = 4
                    high_bits = 4
                if cutlass.const_expr(self.trellis_rate_axis == "n"):
                    # Preparation swizzles the reference record-major payload
                    # into one fixed-size compact pair span per K16 row.
                    pair_u32_per_k16 = Int32(8 * 8 * (low_bits + high_bits))
                    t256_chunks_per_kt = self.cta_n_blocks * (low_bits + high_bits)
                    t256_total_chunks = self.cta_k_blocks * t256_chunks_per_kt
                    t256_pair_u32 = (self.size_k // 16) * pair_u32_per_k16
                    for i in cutlass.range_constexpr(self.b_sh_wr_iters_var):
                        t256_chunk = Int32(i * self.cta_threads) + tid
                        t256_kt = t256_chunk // Int32(t256_chunks_per_kt)
                        t256_in_kt = t256_chunk - t256_kt * Int32(t256_chunks_per_kt)
                        b_dst = (
                            smem_base
                            + Int32(self.sh_b_off * 16)
                            + pipe * Int32(self.b_sh_stage_bytes)
                            + t256_chunk * Int32(16)
                        )
                        pair_base_i64 = Int64(0)
                        if cutlass.const_expr(self.weight_layout_trellis256_proj):
                            # Projection-major FC1 payload: [projection, E,
                            # K16, compact-pair-row].  TP12 has exactly one
                            # 256-channel pair per gate/up projection.
                            pair_plane_u32 = Int64(cute.size(b_i32_flat)) // Int64(2)
                            if cutlass.const_expr(self.trellis_pair_compact_offsets):
                                pair_descriptor = scales_i32_flat[expert_idx].to(Int64)
                                pair_base_i64 = Int64(
                                    output_n_tile
                                ) * pair_plane_u32 + (pair_descriptor >> Int64(1))
                            else:
                                pair_base_i64 = Int64(
                                    output_n_tile
                                ) * pair_plane_u32 + Int64(expert_idx) * Int64(
                                    t256_pair_u32
                                )
                        else:
                            # Dense/expert-major payload: [E, pair, K16,
                            # compact-pair-row].
                            if cutlass.const_expr(self.trellis_pair_compact_offsets):
                                pair_descriptor = scales_i32_flat[expert_idx].to(Int64)
                                pair_base_i64 = (pair_descriptor >> Int64(1)) + Int64(
                                    output_n_tile
                                ) * Int64(t256_pair_u32)
                            else:
                                pair_base_i64 = (
                                    Int64(expert_idx) * Int64(self.size_n // 256)
                                    + Int64(output_n_tile)
                                ) * Int64(t256_pair_u32)
                        b_src_i64 = (
                            pair_base_i64
                            + (
                                Int64(tile_idx) * Int64(self.cta_k_blocks)
                                + Int64(t256_kt)
                            )
                            * Int64(pair_u32_per_k16)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                        cp_async4_shared_global_pred(
                            b_dst,
                            get_ptr_as_int64(b_i32_flat, b_src_i64),
                            (t256_chunk < Int32(t256_total_chunks)).to(Int32),
                        )
                else:
                    # The two source records remain contiguous in both payload
                    # and physical K.  Reserve K4-sized shared slots, then copy
                    # each compact K2/K3/K4 source tile into its K16 slot's
                    # prefix.
                    max_chunks_per_kt = self.cta_n_blocks * 8
                    t256_expert_u32 = (
                        (self.size_k // 16) * t256_n16 * 4 * (low_bits + high_bits)
                    )
                    low_record_u32 = Int32(8 * t256_n16 * 8 * low_bits)
                    for i in cutlass.range_constexpr(self.b_sh_wr_iters_var):
                        t256_chunk = Int32(i * self.cta_threads) + tid
                        t256_kt = t256_chunk // Int32(max_chunks_per_kt)
                        t256_in_kt = t256_chunk - t256_kt * Int32(max_chunks_per_kt)
                        logical_k16 = tile_idx * Int32(self.cta_k_blocks) + t256_kt
                        high_record = (logical_k16 >= Int32(8)).to(Int32)
                        local_k16 = logical_k16 - high_record * Int32(8)
                        record_bits = Int32(low_bits)
                        if high_record != Int32(0):
                            record_bits = Int32(high_bits)
                        record_base_u32 = high_record * low_record_u32
                        actual_chunks = Int32(self.cta_n_blocks * 2) * record_bits
                        b_dst = (
                            smem_base
                            + Int32(self.sh_b_off * 16)
                            + pipe * Int32(self.b_sh_stage_bytes)
                            + t256_chunk * Int32(16)
                        )
                        expert_base_u32 = Int64(t256_expert_u32) * Int64(expert_idx)
                        if cutlass.const_expr(self.trellis_pair_compact_offsets):
                            pair_descriptor = scales_i32_flat[expert_idx].to(Int64)
                            expert_base_u32 = pair_descriptor >> Int64(1)
                        b_src_i64 = (
                            expert_base_u32
                            + Int64(record_base_u32)
                            + Int64(local_k16)
                            * Int64(t256_n16 * 8)
                            * Int64(record_bits)
                            + Int64(output_n_tile)
                            * Int64(self.cta_n_blocks * 8)
                            * Int64(record_bits)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                        cp_async4_shared_global_pred(
                            b_dst,
                            get_ptr_as_int64(b_i32_flat, b_src_i64),
                            (t256_in_kt < actual_chunks).to(Int32),
                        )
            else:
                t256_tile_u32 = 8 * self.trellis_bits
                t256_expert_u32 = (self.size_k // 16) * t256_n16 * t256_tile_u32
                t256_chunks_per_kt = self.cta_n_blocks * (2 * self.trellis_bits)
                t256_total_chunks = self.cta_k_blocks * t256_chunks_per_kt
                for i in cutlass.range_constexpr(self.b_sh_wr_iters_var):
                    t256_chunk = Int32(i * self.cta_threads) + tid
                    t256_kt = t256_chunk // Int32(t256_chunks_per_kt)
                    t256_in_kt = t256_chunk - t256_kt * Int32(t256_chunks_per_kt)
                    b_dst = (
                        smem_base
                        + Int32(self.sh_b_off * 16)
                        + pipe * Int32(self.b_sh_stage_bytes)
                        + t256_chunk * Int32(16)
                    )
                    if cutlass.const_expr(self.weight_layout_trellis256_proj):
                        t256_half_n16 = t256_n16 // 2
                        t256_out_n16 = output_n_tile * Int32(self.cta_n_blocks)
                        t256_proj = (t256_out_n16 >= Int32(t256_half_n16)).to(Int32)
                        t256_local_n16 = t256_out_n16 - t256_proj * Int32(t256_half_n16)
                        t256_proj_expert_u32 = (
                            (self.size_k // 16) * t256_half_n16 * t256_tile_u32
                        )
                        # Projection-major W13 is physically [2, E, ...].
                        t256_plane_u32 = Int64(cute.size(b_i32_flat)) // Int64(2)
                        b_src_i64 = (
                            Int64(t256_proj) * t256_plane_u32
                            + Int64(expert_idx) * Int64(t256_proj_expert_u32)
                            + (
                                Int64(tile_idx) * Int64(self.cta_k_blocks)
                                + Int64(t256_kt)
                            )
                            * Int64(t256_half_n16 * t256_tile_u32)
                            + Int64(t256_local_n16) * Int64(t256_tile_u32)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                    else:
                        b_src_i64 = (
                            Int64(t256_expert_u32) * Int64(expert_idx)
                            + (
                                Int64(tile_idx) * Int64(self.cta_k_blocks)
                                + Int64(t256_kt)
                            )
                            * Int64(t256_n16 * t256_tile_u32)
                            + Int64(output_n_tile)
                            * Int64(self.cta_n_blocks * t256_tile_u32)
                            + Int64(t256_in_kt) * Int64(4)
                        )
                    cp_async4_shared_global_pred(
                        b_dst,
                        get_ptr_as_int64(b_i32_flat, b_src_i64),
                        (t256_chunk < Int32(t256_total_chunks)).to(Int32),
                    )

        for i in cutlass.range_constexpr(
            0 if self.b_region_variable else self.b_sh_wr_iters
        ):
            b_src_int4 = (
                b_gl_rd_base
                + tile_idx * Int32(self.cta_k_blocks) * b_gl_stride
                + Int32(i * (self.cta_threads // self.b_sh_stride)) * b_gl_stride
            )
            b_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + Int32(i * self.cta_threads)
                + tid,
            )
            if cutlass.const_expr(self.weight_layout == "packed"):
                cp_async4_shared_global(
                    b_dst,
                    get_ptr_as_int64(b_i32_flat, b_src_int4 * Int32(4)),
                )
            else:
                b_dst = (
                    smem_base
                    + Int32(self.sh_b_off * 16)
                    + pipe * Int32(self.b_sh_stage * 16)
                    + self._modelopt_smem_permuted_offset(
                        (Int32(i * self.cta_threads) + tid) * Int32(16)
                    )
                )
                self._stage_b_tile_modelopt_native(
                    b_i32_flat,
                    b_dst,
                    expert_idx,
                    output_n_tile,
                    tile_idx,
                    Int32(i * self.cta_threads) + tid,
                )

        if cutlass.const_expr(self.csf_block_inline):
            self._csf_block_stage(
                scales_i32_flat,
                smem_base,
                tid,
                pipe,
                tile_idx,
                expert_idx,
                output_n_tile,
            )
        elif cutlass.const_expr(self.native_nvfp4_scales):
            self._stage_modelopt_scales(
                scales_i32_flat,
                smem_base,
                tid,
                pipe,
                expert_idx,
                output_n_tile,
                tile_idx,
            )
        elif cutlass.const_expr(
            not self.weight_layout_trellis256 and not self.weight_layout_block
        ):
            if tid < Int32(self.s_sh_stage):
                s_k_group = tile_idx * Int32(self.s_tb_groups) + tid // Int32(
                    self.s_sh_stride
                )
                s_n_group = Int32(self.s_sh_stride) * output_n_tile + (
                    tid % Int32(self.s_sh_stride)
                )
                s_src_int4 = scales_expert_off + s_gl_stride * s_k_group + s_n_group
                s_dst = self._int4_addr(
                    smem_base,
                    Int32(self.sh_s_off) + pipe * Int32(self.s_sh_stage) + tid,
                )
                if cutlass.const_expr(self.has_logical_tail):
                    if s_k_group < Int32(self.scale_k_groups) and s_n_group < Int32(
                        self.scale_n_groups
                    ):
                        cp_async4_shared_global(
                            s_dst,
                            get_ptr_as_int64(scales_i32_flat, s_src_int4 * Int32(4)),
                        )
                    else:
                        st_shared_v4_u32(
                            s_dst, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        )
                else:
                    cp_async4_shared_global(
                        s_dst,
                        get_ptr_as_int64(scales_i32_flat, s_src_int4 * Int32(4)),
                    )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _csf_block_stage(
        self, scales, smem_base, tid, pipe, tile_idx, expert, output_tile
    ):
        warp, lane = tid // Int32(32), tid % Int32(32)
        columns = (self.s_tb_groups + 3) // 4
        for chunk in cutlass.range_constexpr(
            (self.csf_stage_groups + self.cta_threads // 32 - 1)
            // (self.cta_threads // 32)
        ):
            local = warp + Int32(chunk * (self.cta_threads // 32))
            if local < Int32(self.csf_stage_groups):
                group = (
                    (
                        output_tile * Int32(self.tile_n) // Int32(128)
                        + local // Int32(columns)
                    )
                    * Int32(self.csf_block_reader.plane.storage.column_groups)
                    + tile_idx * Int32(self.s_tb_groups) // Int32(4)
                    + local % Int32(columns)
                )
                target = cute.make_ptr(
                    cutlass.Uint8,
                    smem_base
                    + Int32(self.sh_csf_block_off)
                    + (pipe * Int32(self.csf_stage_groups) + local) * Int32(512),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                index = (
                    local
                    // Int32(columns)
                    * Int32(self.csf_block_reader.plane.storage.column_groups)
                    + tile_idx * Int32(self.s_tb_groups) // Int32(4)
                    + local % Int32(columns)
                )
                index_addr = smem_base + Int32(self.sh_csf_index_off) + index * Int32(8)
                bounds = (
                    ld_shared_u32(index_addr),
                    ld_shared_u32(index_addr + Int32(4)),
                )
                self.csf_block_reader.stage_compressed(
                    scales.iterator, expert, group, target, lane, bounds
                )

    @cute.jit
    def _csf_block_expand(self, scales, smem_base, tid, pipe, tile_idx, output_tile):
        warp, lane = tid // Int32(32), tid % Int32(32)
        columns = (self.s_tb_groups + 3) // 4
        for chunk in cutlass.range_constexpr(
            (self.csf_stage_groups + self.cta_threads // 32 - 1)
            // (self.cta_threads // 32)
        ):
            local = warp + Int32(chunk * (self.cta_threads // 32))
            if local < Int32(self.csf_stage_groups):
                row_start = output_tile * Int32(self.tile_n)
                row_group = row_start // Int32(128) + local // Int32(columns)
                column_group = tile_idx * Int32(self.s_tb_groups) // Int32(
                    4
                ) + local % Int32(columns)
                group = (
                    row_group * Int32(self.csf_block_reader.plane.storage.column_groups)
                    + column_group
                )
                target = cute.make_ptr(
                    cutlass.Uint8,
                    smem_base
                    + Int32(self.sh_csf_block_off)
                    + (pipe * Int32(self.csf_stage_groups) + local) * Int32(512),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                self.csf_block_reader.expand_compressed(
                    scales.iterator, group, target, lane
                )
                if cutlass.const_expr(not self.csf_inplace_stage):
                    valid_rows = Int32(self.size_n) - row_group * Int32(128)
                    nr = cutlass.max(Int32(1), cutlass.min(Int32(128), valid_rows))
                    for quarter in cutlass.range_constexpr(4):
                        position = lane * Int32(16) + Int32(quarter * 4)
                        row = row_group * Int32(128) + position % nr - row_start
                        column = (
                            column_group * Int32(4)
                            + position // nr
                            - tile_idx * Int32(self.s_tb_groups)
                        )
                        if (
                            valid_rows > 0
                            and row >= 0
                            and row < Int32(self.tile_n)
                            and column >= 0
                            and column < Int32(self.s_tb_groups)
                        ):
                            value = ld_shared_u32(target.toint().to(Int32) + position)
                            st_shared_u32(
                                smem_base
                                + Int32(self.sh_s_off * 16)
                                + pipe * Int32(self.s_sh_stage * 16)
                                + column * Int32(self.tile_n)
                                + row,
                                value,
                            )
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_pipeline_step(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(kk == self.b_sh_wr_iters - 2):
            self._prefetch_lookahead_tile(
                a_bf16_flat,
                a_alt_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                pipe,
                tile_idx,
                k_tiles,
                reduce_k_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
                expert_idx,
                dynamic_pair_override,
            )

    @cute.jit
    def _prefetch_initial_tiles(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        if cutlass.const_expr(self.csf_block_inline):
            source = cute.recast_ptr(scales_i32_flat.iterator, dtype=cutlass.Uint8)
            for chunk in cutlass.range_constexpr(
                (self.csf_index_count + self.cta_threads - 1) // self.cta_threads
            ):
                index = tid + Int32(chunk * self.cta_threads)
                if index < Int32(self.csf_index_count):
                    group = (
                        output_n_tile
                        * Int32(self.tile_n)
                        // Int32(128)
                        * Int32(self.csf_block_reader.plane.storage.column_groups)
                        + index
                    )
                    first, end = Uint32(0), Uint32(0)
                    if group < Int32(self.csf_block_reader.plane.storage.groups):
                        first, end = self.csf_block_reader.plane.group_record_bounds(
                            source, expert_idx, group
                        )
                    addr = smem_base + Int32(self.sh_csf_index_off) + index * Int32(8)
                    st_shared_u32(addr, first)
                    st_shared_u32(addr + Int32(4), end)
            cute.arch.sync_threads()
        for pipe in cutlass.range_constexpr(self.stages - 1):
            if Int32(pipe) < k_tiles:
                self._stage_k_tile_async(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    scales_i32_flat,
                    smem_base,
                    tid,
                    Int32(pipe),
                    reduce_k_tile + Int32(pipe),
                    block_valid_rows,
                    a_gl_stride,
                    b_gl_stride,
                    s_gl_stride,
                    scales_expert_off,
                    b_gl_rd_base,
                    a_gl_rd_row,
                    a_gl_rd_col0,
                    a_sh_wr,
                    a_rows_per_iter,
                    output_n_tile,
                    expert_idx,
                    dynamic_pair_override,
                )
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()
        if cutlass.const_expr(self.csf_block_inline):
            if k_tiles > Int32(0):
                self._csf_block_expand(
                    scales_i32_flat,
                    smem_base,
                    tid,
                    Int32(0),
                    reduce_k_tile,
                    output_n_tile,
                )

    @cute.jit
    def _prefetch_lookahead_tile(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
        dynamic_pair_override: cutlass.Constexpr[int],
    ):
        fetch_tile = tile_idx + Int32(self.stages - 1)
        if fetch_tile < k_tiles:
            self._stage_k_tile_async(
                a_bf16_flat,
                a_alt_bf16_flat,
                b_i32_flat,
                scales_i32_flat,
                smem_base,
                tid,
                Int32((pipe + self.stages - 1) % self.stages),
                reduce_k_tile + fetch_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
                expert_idx,
                dynamic_pair_override,
            )
        else:
            cute.arch.cp_async_commit_group()
        if cutlass.const_expr(not self.weight_layout_block):
            cute.arch.cp_async_wait_group(self.stages - 2)
            cute.arch.sync_threads()
        if cutlass.const_expr(self.csf_block_inline):
            if tile_idx + Int32(1) < k_tiles:
                self._csf_block_expand(
                    scales_i32_flat,
                    smem_base,
                    tid,
                    Int32((pipe + 1) % self.stages),
                    reduce_k_tile + tile_idx + Int32(1),
                    output_n_tile,
                )

    @cute.jit
    def _reduction_offsets(self, tid: Int32):
        red_idx = tid // Int32(self.b_sh_stride_threads)
        red_sh_stride = Int32(self.b_sh_stride_threads * 4 * 2)
        red_sh_delta = Int32(self.b_sh_stride_threads)
        red_sh_rd = red_sh_stride * (tid // Int32(self.b_sh_stride_threads)) + (
            tid % Int32(self.b_sh_stride_threads)
        )
        return red_idx, red_sh_stride, red_sh_delta, red_sh_rd

    @cute.jit
    def _fold_cta_partials_m8(self, acc, smem_base: Int32, tid: Int32):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )
            if cutlass.const_expr(red_off == 2):
                if Int32(2) <= red_idx and red_idx < Int32(4):
                    for jj in cutlass.range_constexpr(4):
                        red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                            red_sh_rd - red_sh_stride * Int32(2)
                        )
                        st_shared_v4_f32(
                            self._int4_addr(
                                smem_base, Int32(self.sh_red_off) + red_sh_wr
                            ),
                            acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                            acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                            acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                            acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ],
                        )
                cute.arch.sync_threads()

            if Int32(1) <= red_idx and red_idx < Int32(2):
                for jj in cutlass.range_constexpr(4):
                    red_sh_wr = red_sh_delta * Int32(jj * 2) + (
                        red_sh_rd - red_sh_stride
                    )
                    if cutlass.const_expr(red_off > 1):
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off)
                            + red_sh_delta * Int32(jj * 2)
                            + red_sh_rd,
                        )
                        wr_addr = self._int4_addr(
                            smem_base, Int32(self.sh_red_off) + red_sh_wr
                        )
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r0
                            + w0
                        )
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r1
                            + w1
                        )
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r2
                            + w2
                        )
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ] = (
                            acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                                (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                            ]
                            + r3
                            + w3
                        )
                    st_shared_v4_f32(
                        self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    )
            cute.arch.sync_threads()

            if red_idx == Int32(0):
                for jj in cutlass.range_constexpr(4):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off)
                        + red_sh_delta * Int32(jj * 2)
                        + red_sh_rd,
                    )
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r0
                    )
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r1
                    )
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r2
                    )
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r3
                    )
            cute.arch.sync_threads()

    @cute.jit
    def _output_store_cursor(self, tid: Int32, output_n_tile: Int32):
        c_gl_stride = Int32(self.size_n // 8)
        c_sh_stride = Int32(2 * self.cta_n_blocks + 1)
        c_gl_wr_delta = c_gl_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_sh_rd_delta = c_sh_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_gl_wr = (
            c_gl_stride * (tid // Int32(2 * self.cta_n_blocks))
            + (tid % Int32(2 * self.cta_n_blocks))
            + Int32(2 * self.cta_n_blocks) * output_n_tile
        )
        c_sh_rd = c_sh_stride * (tid // Int32(2 * self.cta_n_blocks)) + (
            tid % Int32(2 * self.cta_n_blocks)
        )
        return c_gl_stride, c_sh_stride, c_gl_wr_delta, c_sh_rd_delta, c_gl_wr, c_sh_rd

    @cute.jit
    def _output_store_cursor_tail(self, tid: Int32, output_n_tile: Int32):
        c_gl_stride = Int32(self.size_n // 8)
        c_gl_stride_covered = Int32(self.covered_size_n // 8)
        c_sh_stride = Int32(2 * self.cta_n_blocks + 1)
        c_gl_wr_delta = c_gl_stride_covered * Int32(
            self.cta_threads // (2 * self.cta_n_blocks)
        )
        c_sh_rd_delta = c_sh_stride * Int32(self.cta_threads // (2 * self.cta_n_blocks))
        c_gl_wr = (
            c_gl_stride_covered * (tid // Int32(2 * self.cta_n_blocks))
            + (tid % Int32(2 * self.cta_n_blocks))
            + Int32(2 * self.cta_n_blocks) * output_n_tile
        )
        c_sh_rd = c_sh_stride * (tid // Int32(2 * self.cta_n_blocks)) + (
            tid % Int32(2 * self.cta_n_blocks)
        )
        return (
            c_gl_stride,
            c_gl_stride_covered,
            c_sh_stride,
            c_gl_wr_delta,
            c_sh_rd_delta,
            c_gl_wr,
            c_sh_rd,
        )

    @cute.jit
    def _drain_output_smem(
        self,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        c_gl_stride: Int32,
        c_gl_wr: Int32,
        c_gl_wr_delta: Int32,
        c_sh_rd: Int32,
        c_sh_rd_delta: Int32,
        block_valid_rows: Int32,
        store_iters: cutlass.Constexpr[int],
    ):
        for _ in cutlass.range_constexpr(store_iters):
            row = c_gl_wr // c_gl_stride
            if row < block_valid_rows:
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_route_off * 16) + row * Int32(4)
                )
                true_idx = Int64(route_index) * Int64(c_gl_stride) + Int64(
                    c_gl_wr % c_gl_stride
                )
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16) + row * Int32(4)
                    )
                    q0 = self._elem2_mul(q0, scale_bf2)
                    q1 = self._elem2_mul(q1, scale_bf2)
                    q2 = self._elem2_mul(q2, scale_bf2)
                    q3 = self._elem2_mul(q3, scale_bf2)
                if cutlass.const_expr(self.epilogue_relu2):
                    q0 = self._relu2_elem2(q0)
                    q1 = self._relu2_elem2(q1)
                    q2 = self._relu2_elem2(q2)
                    q3 = self._relu2_elem2(q3)
                if cutlass.const_expr(self.fused_topk_sum):
                    # Fold the per-route partials into the per-token output in
                    # the epilogue (drops the separate top-k-sum launch). The
                    # output must be zeroed before launch; each route slot maps
                    # to token = route_index // top_k.  bf16x2 add lands two
                    # consecutive hidden lanes per word.
                    token_idx = route_index // Int32(self.fused_sum_topk)
                    col_word = c_gl_wr % c_gl_stride
                    if cutlass.const_expr(self.fused_sum_fp32):
                        out_elem = Int64(token_idx) * Int64(self.size_n) + Int64(
                            col_word
                        ) * Int64(8)
                        out_addr = get_ptr_as_int64(c_bf16_flat, out_elem)
                        q00, q01 = self._elem2_to_f32x2(q0)
                        q10, q11 = self._elem2_to_f32x2(q1)
                        q20, q21 = self._elem2_to_f32x2(q2)
                        q30, q31 = self._elem2_to_f32x2(q3)
                        if cutlass.const_expr(self.fused_sum_router_weights_fp32):
                            scale = ld_shared_f32(
                                smem_base
                                + Int32(self.sh_topk_off * 16)
                                + row * Int32(4)
                            )
                            q00, q01 = q00 * scale, q01 * scale
                            q10, q11 = q10 * scale, q11 * scale
                            q20, q21 = q20 * scale, q21 * scale
                            q30, q31 = q30 * scale, q31 * scale
                        red_add_global_v4_f32(out_addr, q00, q01, q10, q11)
                        red_add_global_v4_f32(out_addr + Int64(16), q20, q21, q30, q31)
                    else:
                        out_idx = Int64(token_idx) * Int64(c_gl_stride) + Int64(
                            col_word
                        )
                        out_addr = get_ptr_as_int64(c_bf16_flat, out_idx * Int64(8))
                        red_add_global_bf16x2(out_addr, q0)
                        red_add_global_bf16x2(out_addr + Int64(4), q1)
                        red_add_global_bf16x2(out_addr + Int64(8), q2)
                        red_add_global_bf16x2(out_addr + Int64(12), q3)
                else:
                    st_global_v4_u32(
                        get_ptr_as_int64(c_bf16_flat, true_idx * Int64(8)),
                        q0,
                        q1,
                        q2,
                        q3,
                    )
            c_gl_wr += c_gl_wr_delta
            c_sh_rd += c_sh_rd_delta
        cute.arch.sync_threads()

    @cute.jit
    def _drain_output_smem_tail(
        self,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        c_gl_stride: Int32,
        c_gl_stride_covered: Int32,
        c_gl_wr: Int32,
        c_gl_wr_delta: Int32,
        c_sh_rd: Int32,
        c_sh_rd_delta: Int32,
        block_valid_rows: Int32,
        store_iters: cutlass.Constexpr[int],
    ):
        for _ in cutlass.range_constexpr(store_iters):
            row = c_gl_wr // c_gl_stride_covered
            col_word = c_gl_wr - row * c_gl_stride_covered
            if row < block_valid_rows and col_word < c_gl_stride:
                route_index = ld_shared_i32_relaxed(
                    smem_base + Int32(self.sh_route_off * 16) + row * Int32(4)
                )
                true_idx = Int64(route_index) * Int64(c_gl_stride) + Int64(col_word)
                q0, q1, q2, q3 = ld_shared_v4_u32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + c_sh_rd)
                )
                if cutlass.const_expr(self.mul_topk_weights):
                    scale_bf2 = ld_shared_u32(
                        smem_base + Int32(self.sh_topk_off * 16) + row * Int32(4)
                    )
                    q0 = self._elem2_mul(q0, scale_bf2)
                    q1 = self._elem2_mul(q1, scale_bf2)
                    q2 = self._elem2_mul(q2, scale_bf2)
                    q3 = self._elem2_mul(q3, scale_bf2)
                if cutlass.const_expr(self.epilogue_relu2):
                    q0 = self._relu2_elem2(q0)
                    q1 = self._relu2_elem2(q1)
                    q2 = self._relu2_elem2(q2)
                    q3 = self._relu2_elem2(q3)
                if cutlass.const_expr(self.fused_topk_sum):
                    token_idx = route_index // Int32(self.fused_sum_topk)
                    if cutlass.const_expr(self.fused_sum_fp32):
                        out_elem = Int64(token_idx) * Int64(self.size_n) + Int64(
                            col_word
                        ) * Int64(8)
                        out_addr = get_ptr_as_int64(c_bf16_flat, out_elem)
                        q00, q01 = self._elem2_to_f32x2(q0)
                        q10, q11 = self._elem2_to_f32x2(q1)
                        q20, q21 = self._elem2_to_f32x2(q2)
                        q30, q31 = self._elem2_to_f32x2(q3)
                        if cutlass.const_expr(self.fused_sum_router_weights_fp32):
                            scale = ld_shared_f32(
                                smem_base
                                + Int32(self.sh_topk_off * 16)
                                + row * Int32(4)
                            )
                            q00, q01 = q00 * scale, q01 * scale
                            q10, q11 = q10 * scale, q11 * scale
                            q20, q21 = q20 * scale, q21 * scale
                            q30, q31 = q30 * scale, q31 * scale
                        red_add_global_v4_f32(out_addr, q00, q01, q10, q11)
                        red_add_global_v4_f32(out_addr + Int64(16), q20, q21, q30, q31)
                    else:
                        out_idx = Int64(token_idx) * Int64(c_gl_stride) + Int64(
                            col_word
                        )
                        out_addr = get_ptr_as_int64(c_bf16_flat, out_idx * Int64(8))
                        red_add_global_bf16x2(out_addr, q0)
                        red_add_global_bf16x2(out_addr + Int64(4), q1)
                        red_add_global_bf16x2(out_addr + Int64(8), q2)
                        red_add_global_bf16x2(out_addr + Int64(12), q3)
                else:
                    st_global_v4_u32(
                        get_ptr_as_int64(c_bf16_flat, true_idx * Int64(8)),
                        q0,
                        q1,
                        q2,
                        q3,
                    )
            c_gl_wr += c_gl_wr_delta
            c_sh_rd += c_sh_rd_delta
        cute.arch.sync_threads()

    @cute.jit
    def _store_tile_m8(
        self,
        acc,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_n_tile_tail):
            (
                c_gl_stride,
                c_gl_stride_covered,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor_tail(tid, output_n_tile)
        else:
            (
                c_gl_stride,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor(tid, output_n_tile)
            c_gl_stride_covered = c_gl_stride
        c_sh_wr = (
            Int32(8) * c_sh_stride * (((tid & Int32(31)) % Int32(4)) * Int32(2))
            + (tid & Int32(31)) // Int32(4)
            + Int32(64) * (tid // Int32(32))
        )
        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for jj in cutlass.range_constexpr(4):
                wr = c_sh_wr + Int32(16 * jj)
                if cutlass.const_expr(
                    self.weight_layout_trellis256_pair and self.trellis_rate_axis == "n"
                ):
                    # MMA register assignment is [L0,L1,H0,H1] per warp for
                    # balanced P24 work.  Scatter the four N16 fragments back
                    # to contiguous [L(0..7),H(0..7)] record order in output
                    # SMEM.  This is a bijection and occurs before the outer
                    # H128, so the public channel/transform semantics do not
                    # change.
                    warp_n = tid // Int32(32)
                    semantic_n16 = Int32(2) * warp_n + Int32(jj)
                    if cutlass.const_expr(jj >= 2):
                        semantic_n16 = Int32(8) + Int32(2) * warp_n + Int32(jj - 2)
                    compute_n16 = Int32(4) * warp_n + Int32(jj)
                    wr += Int32(16) * (semantic_n16 - compute_n16)
                self._st_shared_elem_from_f32(
                    smem_base + Int32(self.sh_red_off * 16) + (wr * Int32(2)),
                    acc[(jj * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[(jj * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8)) * Int32(2)),
                    acc[(jj * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
                self._st_shared_elem_from_f32(
                    smem_base
                    + Int32(self.sh_red_off * 16)
                    + ((wr + Int32(8) + Int32(8) * c_sh_stride) * Int32(2)),
                    acc[(jj * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (jj * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    * write_scale,
                )
        cute.arch.sync_threads()

        store_iters = _covering_count(16, self.cta_threads // (2 * self.cta_n_blocks))
        if cutlass.const_expr(self.has_n_tile_tail):
            self._drain_output_smem_tail(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_stride_covered,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )
        else:
            self._drain_output_smem(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )

    @cute.jit
    def _fold_cta_partials_large_m(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        smem_base: Int32,
        tid: Int32,
    ):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )

            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(mb == 0):
                    self._fold_cta_partials_large_m_block(
                        acc0,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )
                elif cutlass.const_expr(mb == 1):
                    self._fold_cta_partials_large_m_block(
                        acc1,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )
                elif cutlass.const_expr(mb == 2):
                    self._fold_cta_partials_large_m_block(
                        acc2,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )
                else:
                    self._fold_cta_partials_large_m_block(
                        acc3,
                        smem_base,
                        red_off,
                        red_idx,
                        red_sh_stride,
                        red_sh_delta,
                        red_sh_rd,
                    )

    @cute.jit
    def _fold_cta_partials_large_m_block(
        self,
        acc,
        smem_base: Int32,
        red_off: cutlass.Constexpr[int],
        red_idx: Int32,
        red_sh_stride: Int32,
        red_sh_delta: Int32,
        red_sh_rd: Int32,
    ):
        if cutlass.const_expr(red_off == 2):
            if Int32(2) <= red_idx and red_idx < Int32(4):
                for flat_j in cutlass.range_constexpr(8):
                    red_sh_wr = red_sh_delta * Int32(flat_j) + (
                        red_sh_rd - red_sh_stride * Int32(2)
                    )
                    st_shared_v4_f32(
                        self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                        acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                        acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ],
                    )
            cute.arch.sync_threads()

        if Int32(1) <= red_idx and red_idx < Int32(2):
            for flat_j in cutlass.range_constexpr(8):
                red_sh_wr = red_sh_delta * Int32(flat_j) + (red_sh_rd - red_sh_stride)
                if cutlass.const_expr(red_off > 1):
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off)
                        + red_sh_delta * Int32(flat_j)
                        + red_sh_rd,
                    )
                    wr_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off) + red_sh_wr,
                    )
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r0
                        + w0
                    )
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r1
                        + w1
                    )
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r2
                        + w2
                    )
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ] = (
                        acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                            (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                        ]
                        + r3
                        + w3
                    )
                st_shared_v4_f32(
                    self._int4_addr(smem_base, Int32(self.sh_red_off) + red_sh_wr),
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ],
                )
        cute.arch.sync_threads()

        if red_idx == Int32(0):
            for flat_j in cutlass.range_constexpr(8):
                rd_addr = self._int4_addr(
                    smem_base,
                    Int32(self.sh_red_off) + red_sh_delta * Int32(flat_j) + red_sh_rd,
                )
                r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r0
                )
                acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r1
                )
                acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r2
                )
                acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                ] = (
                    acc[(flat_j * 4 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                        (flat_j * 4 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                    ]
                    + r3
                )
        cute.arch.sync_threads()

    @cute.jit
    def _write_bf16x2_shared(
        self,
        smem_base: Int32,
        half2_idx: Int32,
        c0: cutlass.Float32,
        c1: cutlass.Float32,
        write_scale: cutlass.Float32,
    ):
        packed = self._pack_f32x2_to_elem2(c0 * write_scale, c1 * write_scale)
        st_shared_u32(
            smem_base + Int32(self.sh_red_off * 16) + half2_idx * Int32(4),
            packed,
        )

    @cute.jit
    def _store_tile_large_m(
        self,
        acc0,
        acc1,
        acc2,
        acc3,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_n_tile_tail):
            (
                c_gl_stride,
                c_gl_stride_covered,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor_tail(tid, output_n_tile)
        else:
            (
                c_gl_stride,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor(tid, output_n_tile)
            c_gl_stride_covered = c_gl_stride
        c_sh_wr = (
            Int32(4) * c_sh_stride * ((tid & Int32(31)) // Int32(4))
            + (tid & Int32(31)) % Int32(4)
            + Int32(32) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(mb == 0):
                    self._store_tile_large_m_block(
                        acc0,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
                    )
                elif cutlass.const_expr(mb == 1):
                    self._store_tile_large_m_block(
                        acc1,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
                    )
                elif cutlass.const_expr(mb == 2):
                    self._store_tile_large_m_block(
                        acc2,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
                    )
                else:
                    self._store_tile_large_m_block(
                        acc3,
                        smem_base,
                        c_sh_wr,
                        c_sh_stride,
                        tid // Int32(32),
                        write_scale,
                    )
                c_sh_wr += Int32(16 * (4 * (2 * self.cta_n_blocks + 1)))
        cute.arch.sync_threads()

        store_iters = _covering_count(
            16 * self.cta_m_blocks,
            self.cta_threads // (2 * self.cta_n_blocks),
        )
        if cutlass.const_expr(self.has_n_tile_tail):
            self._drain_output_smem_tail(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_stride_covered,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )
        else:
            self._drain_output_smem(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )

    @cute.jit
    def _store_tile_large_m_block(
        self,
        acc,
        smem_base: Int32,
        c_sh_wr: Int32,
        c_sh_stride: Int32,
        warp_n: Int32,
        write_scale: cutlass.Float32,
    ):
        for jj in cutlass.range_constexpr(4):
            wr = c_sh_wr + Int32(8 * jj)
            if cutlass.const_expr(
                self.weight_layout_trellis256_pair and self.trellis_rate_axis == "n"
            ):
                # Match the M<=8 epilogue: restore the reference record order
                # from the balanced LLHH per-warp MMA assignment before H128.
                semantic_n16 = Int32(2) * warp_n + Int32(jj)
                if cutlass.const_expr(jj >= 2):
                    semantic_n16 = Int32(8) + Int32(2) * warp_n + Int32(jj - 2)
                compute_n16 = Int32(4) * warp_n + Int32(jj)
                wr += Int32(8) * (semantic_n16 - compute_n16)
            self._write_bf16x2_shared(
                smem_base,
                wr,
                acc[(jj * 8) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 1) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 1) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )
            self._write_bf16x2_shared(
                smem_base,
                wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(0),
                acc[(jj * 8 + 2) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 2) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 3) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 3) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )
            self._write_bf16x2_shared(
                smem_base,
                wr + Int32(4),
                acc[(jj * 8 + 4) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 4) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 5) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 5) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )
            self._write_bf16x2_shared(
                smem_base,
                wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(4),
                acc[(jj * 8 + 6) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 6) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                acc[(jj * 8 + 7) // _SCALAR_ACC_FRAGMENT_WIDTH][
                    (jj * 8 + 7) % _SCALAR_ACC_FRAGMENT_WIDTH
                ],
                write_scale,
            )
