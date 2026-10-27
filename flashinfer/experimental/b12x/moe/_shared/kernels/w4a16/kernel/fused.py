"""Persistent fused W4A16 MoE kernel."""

from __future__ import annotations

from b12x._lib.quant.block_codec import BLOCK_CODECS
from b12x._lib.quant.block_codec import IQ2_CODECS
from cutlass.cutlass_dsl import Int32
from cutlass.cutlass_dsl import Int64
from b12x.moe._shared.trellis_codebooks import LUT_E4M3
from b12x.moe._shared.kernels.activations import SITU
from b12x.moe._shared.kernels.activations import SITU_DEFAULT_BETA
from b12x.moe._shared.kernels.activations import SITU_DEFAULT_LINEAR_BETA
from b12x.moe._shared.kernels.activations import SWIGLUOAI_UNINTERLEAVE
from cutlass.cutlass_dsl import Uint32
from b12x._lib.intrinsics.atomics import atomic_add_global_i32
from b12x._lib.quant.block_codec import block_codec
from b12x._lib.intrinsics.memory import cp_async4_shared_global
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass
from b12x._lib.intrinsics.math import fabs_f32
from b12x._lib.intrinsics.math import fmax_f32
from b12x._lib.intrinsics.memory import get_ptr_as_int64
from b12x._lib.intrinsics.sync import ld_global_acquire_i32
from b12x._lib.intrinsics.memory import ld_global_nc_v4_u32
from b12x._lib.intrinsics.memory import ld_global_v4_u32
from b12x._lib.intrinsics.atomics import ld_shared_f32
from b12x.moe._shared.kernels.activations import normalize_moe_activation
from functools import partial
from b12x._lib.intrinsics.atomics import red_add_global_release_i32
from b12x._lib.intrinsics.atomics import red_max_global_f32_nonnegative
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.intrinsics.atomics import st_global_i32
from b12x._lib.intrinsics.memory import st_global_v4_u32
from b12x._lib.intrinsics.atomics import st_shared_f32
from b12x._lib.intrinsics.memory import st_shared_v4_u32
from b12x._lib.intrinsics.sync import threadfence
from b12x._lib.intrinsics.memory import u32_as_f32
from b12x.moe._shared.kernels.w4a16.host import validate_activation
from b12x._lib.intrinsics.reduction import warp_reduce
from b12x.moe._shared.kernels.w4a16.kernel.config import (
    _ALLOWED_ROUTED_SIZES,
    _FP32_TOPK_WEIGHTS,
    _LUT_E4M3_SMEM_REGION_BYTES,
    _LUT_E4M3_VALUE_TABLE_ENTRIES,
    _MODEL_OPT_W13_LAYOUTS,
    _TC_DECODE_MAX_M,
    _TRELLIS256_CODEBOOKS,
    _TRELLIS256_W13_LAYOUTS,
    _WEIGHT_LAYOUTS,
    _cooperative_smem_carveout,
    _covering_count,
    _device_shared_mem_per_sm,
    _lut_e4m3_smem_enabled,
    _normalize_activation_swiglu_params,
    _normalize_scale_format,
    _pack_f32x2_to_bf16x2_rn,
    _resident_cta_shared_bytes,
    _w4a16_small_m_splitk_enabled,
)
from b12x.moe._shared.kernels.w4a16.kernel.gemm import W4A16GemmKernel


class W4A16FusedMoeKernel:
    def __init__(
        self,
        *,
        size_m: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        zero_fc2_output: bool,
        fc1_tile_n: int,
        fc1_tile_k: int,
        fc2_tile_n: int,
        fc2_tile_k: int,
        moe_block_size: int,
        max_m_blocks: int,
        fc2_moe_block_size: int | None = None,
        fc2_schedule_route_block_factor: int = 1,
        element_dtype: str = "bf16",
        fast_math: bool = True,
        swiglu_limit: float | None = None,
        swiglu_clamp_after_silu: bool = False,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        weight_layout: str = "packed",
        scale_format: str = "e4m3_k16",
        w13_layout: str = "w13",
        trellis_bits: int = 3,
        trellis_codebook: str = LUT_E4M3,
        fc1_trellis_pair_kind: str | None = None,
        fc2_trellis_pair_kind: str | None = None,
        direct_topk_routes: bool = False,
        use_expert_map: bool = False,
        tc_decode_fused_sum: bool = False,
        prefill_fused_sum_fp32: bool = False,
        tc_zero_output: bool = True,
        collect_activation_amax: bool = False,
        schedule_whole_tiles: bool = False,
        intermediate_rotation: bool = False,
        full_rotation: bool = False,
        intermediate_hadamard: bool = False,
        rotation_input_dtype: str = "fp16",
        broadcast_suh: bool = False,
        pipeline_stages: int | None = None,
        skip_empty_m_blocks: bool | None = None,
        trellis_decode_table: str = "auto",
        small_m_occupancy: int | None = None,
        csf_inline_scales: bool = False,
    ):
        activation = normalize_moe_activation(activation)
        is_gated = validate_activation(activation)
        swiglu_limit, swiglu_alpha, swiglu_beta = _normalize_activation_swiglu_params(
            activation,
            swiglu_limit,
            swiglu_alpha,
            swiglu_beta,
        )
        if weight_layout not in _WEIGHT_LAYOUTS:
            raise ValueError(f"unsupported W4A16 weight_layout {weight_layout!r}")
        scale_format = _normalize_scale_format(scale_format)
        if weight_layout == "modelopt":
            if w13_layout not in _MODEL_OPT_W13_LAYOUTS:
                raise ValueError(f"unsupported W4A16 w13_layout {w13_layout!r}")
        elif weight_layout == "trellis_t256":
            if w13_layout not in _TRELLIS256_W13_LAYOUTS:
                raise ValueError(f"unsupported trellis_t256 w13_layout {w13_layout!r}")
        else:
            w13_layout = "packed"
        self.tc_decode_fused_sum = bool(tc_decode_fused_sum)
        self.prefill_fused_sum_fp32 = bool(prefill_fused_sum_fp32)
        if self.tc_decode_fused_sum and self.prefill_fused_sum_fp32:
            raise ValueError(
                "TC-decode and large-M FP32 route reduction are mutually exclusive"
            )
        # When two TC-decode launches share one pre-zeroed output, only the
        # first must zero it. Default True preserves single-launch behavior.
        self.tc_zero_output = bool(tc_zero_output)
        self.collect_activation_amax = bool(collect_activation_amax)
        if self.collect_activation_amax and bool(direct_topk_routes):
            raise ValueError("activation amax collection requires route-packed W4A16")
        if self.collect_activation_amax and (
            self.tc_decode_fused_sum or self.prefill_fused_sum_fp32
        ):
            raise ValueError(
                "activation amax collection is incompatible with fused route reduction"
            )
        if self.tc_decode_fused_sum and not bool(direct_topk_routes):
            raise ValueError("tc_decode_fused_sum requires direct_topk_routes")
        if self.tc_decode_fused_sum and element_dtype != "bf16":
            raise ValueError("tc_decode_fused_sum currently requires bf16 activations")
        if self.prefill_fused_sum_fp32 and element_dtype != "bf16":
            raise ValueError("prefill_fused_sum_fp32 requires bf16 activations")
        if self.prefill_fused_sum_fp32 and int(size_m) <= _TC_DECODE_MAX_M:
            raise ValueError(
                "prefill_fused_sum_fp32 requires a token capacity above the decode range"
            )
        fc1_cols = int(intermediate_size) * (2 if is_gated else 1)
        routed_rows = int(size_m) * int(top_k)
        self.size_m = int(size_m)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.fc1_cols = int(fc1_cols)
        self.num_experts = int(num_experts)
        # Compact tier expert counts are artifact data, not launch geometry.
        # Keep the exact count in the plan/result metadata, but exclude it from
        # code specialization for the packed MXFP4 and EXL3 serving layouts.
        # All E-sized tensor views are reconstructed from a runtime scalar at
        # the launch boundary.
        self.dynamic_num_experts = weight_layout in {
            "packed",
            "trellis_t256",
            "iq2_xs",
            "iq2_xxs",
            "q8_0",
        }
        self.top_k = int(top_k)
        self.moe_block_size = int(moe_block_size)
        # Stripe split-K spreads each mn-tile's K range across many CTAs for
        # decode-heavy small-M phases. It is incompatible with whole-tile
        # scheduling and grouped FC2 route subtiles.
        self.small_m_splitk = (
            weight_layout in BLOCK_CODECS and moe_block_size == 8
        ) or _w4a16_small_m_splitk_enabled()
        if self.small_m_splitk:
            schedule_whole_tiles = False
            fc2_schedule_route_block_factor = 1
        self.fc2_moe_block_size = int(
            moe_block_size if fc2_moe_block_size is None else fc2_moe_block_size
        )
        self.fc2_schedule_route_block_factor = int(fc2_schedule_route_block_factor)
        if (
            self.fc2_moe_block_size not in _ALLOWED_ROUTED_SIZES
            or self.moe_block_size % self.fc2_moe_block_size != 0
        ):
            raise ValueError(
                "FC2 route subtile must be an allowed divisor of the packed "
                f"route block: packed={self.moe_block_size}, "
                f"fc2={self.fc2_moe_block_size}"
            )
        expected_fc2_schedule_factor = self.moe_block_size // self.fc2_moe_block_size
        if (
            self.fc2_schedule_route_block_factor < 1
            or expected_fc2_schedule_factor % self.fc2_schedule_route_block_factor != 0
        ):
            raise ValueError(
                "FC2 schedule factor must divide one packed route block: "
                f"factor={self.fc2_schedule_route_block_factor}, "
                f"maximum={expected_fc2_schedule_factor}"
            )
        self.activation = activation
        self.activation_is_gated = is_gated
        self.activation_is_situ = activation == SITU
        self.activation_is_swigluoai = activation == SWIGLUOAI_UNINTERLEAVE
        self.swiglu_clamp_after_silu = bool(swiglu_clamp_after_silu)
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else swiglu_limit
        self.swiglu_alpha = float(swiglu_alpha)
        self.swiglu_beta = float(swiglu_beta)
        self.weight_layout = weight_layout
        self.trellis_bits = int(trellis_bits)
        self.trellis_codebook = str(trellis_codebook).lower()
        if self.weight_layout == "trellis_t256":
            if self.trellis_codebook not in _TRELLIS256_CODEBOOKS:
                raise ValueError(
                    "trellis_t256 codebook must be one of "
                    f"{sorted(_TRELLIS256_CODEBOOKS)}, got {self.trellis_codebook!r}"
                )
        self.fc1_trellis_pair_kind = (
            None
            if fc1_trellis_pair_kind is None
            else str(fc1_trellis_pair_kind).upper()
        )
        self.fc2_trellis_pair_kind = (
            None
            if fc2_trellis_pair_kind is None
            else str(fc2_trellis_pair_kind).upper()
        )
        if (self.fc1_trellis_pair_kind is None) != (self.fc2_trellis_pair_kind is None):
            raise ValueError(
                "fused trellis pair weights require both FC1 and FC2 pair kinds"
            )
        if self.fc1_trellis_pair_kind is not None:
            if weight_layout != "trellis_t256":
                raise ValueError("fused trellis pairs require trellis_t256 weights")
            if self.trellis_bits != 3:
                raise ValueError(
                    "fused trellis pairs require the trellis_bits=3 base specialization"
                )
            dynamic_kinds = {"PDYNAMIC", "P33_P43"}
            static_kinds = {"P24", "P33", "P43", "P44"}
            if (
                self.fc1_trellis_pair_kind in dynamic_kinds
                or self.fc2_trellis_pair_kind in dynamic_kinds
            ):
                if self.fc1_trellis_pair_kind != self.fc2_trellis_pair_kind:
                    raise ValueError("dynamic fused trellis pair kinds must match")
            elif (
                self.fc1_trellis_pair_kind not in static_kinds
                or self.fc2_trellis_pair_kind not in static_kinds
            ):
                raise ValueError("unsupported static fused trellis pair kind")
        self.scale_format = scale_format
        self.w13_layout = w13_layout
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.zero_fc2_output = bool(zero_fc2_output)
        self.element_dtype = element_dtype
        self.is_fp16 = element_dtype == "fp16"
        self.fast_math = bool(fast_math)
        self.direct_topk_routes = bool(direct_topk_routes)
        self.use_expert_map = bool(use_expert_map)
        if self.use_expert_map and not self.direct_topk_routes:
            raise ValueError("use_expert_map requires direct_topk_routes")
        self.schedule_whole_tiles = bool(
            (schedule_whole_tiles or weight_layout == "trellis_t256")
            and not self.small_m_splitk
        )
        self.intermediate_rotation = bool(intermediate_rotation)
        if self.intermediate_rotation:
            if weight_layout != "trellis_t256":
                raise ValueError(
                    "intermediate_rotation is only supported for trellis_t256"
                )
            if not is_gated or self.activation_is_swigluoai or self.has_swiglu_limit:
                raise ValueError(
                    "intermediate_rotation requires unclamped gated silu or situ "
                    "(no swiglu limit/oai)"
                )
            if int(intermediate_size) % 128 != 0:
                raise ValueError(
                    "intermediate_rotation requires intermediate_size % 128 == 0"
                )
        self.full_rotation = bool(full_rotation)
        self.intermediate_hadamard = bool(intermediate_hadamard)
        # suh tables hold one row shared by every expert: index them with a
        # zero expert stride.
        self.broadcast_suh = bool(broadcast_suh)
        self.rotation_input_dtype = str(rotation_input_dtype)
        self.rotation_input_is_fp16 = self.rotation_input_dtype == "fp16"
        if self.full_rotation:
            if not self.intermediate_rotation:
                raise ValueError("full_rotation requires intermediate_rotation")
            if element_dtype != "fp16":
                raise ValueError("full_rotation requires fp16 GEMM operands")
            if self.rotation_input_dtype not in {"bf16", "fp16"}:
                raise ValueError("full_rotation input dtype must be 'bf16' or 'fp16'")
            if self.tc_decode_fused_sum or self.prefill_fused_sum_fp32:
                raise ValueError(
                    "full_rotation is incompatible with fused route reduction"
                )
            if self.apply_router_weight_on_input:
                raise ValueError(
                    "full_rotation applies router weights only in the fp32 top-k sum"
                )
            if int(hidden_size) % 128 != 0:
                raise ValueError("full_rotation requires hidden_size % 128 == 0")
        if self.intermediate_hadamard:
            if not self.full_rotation:
                raise ValueError("intermediate Hadamard requires full rotation")
            if int(hidden_size) % 512 != 0:
                raise ValueError(
                    "intermediate Hadamard requires hidden_size % 512 == 0"
                )
            if int(intermediate_size) % 128 != 0:
                raise ValueError(
                    "intermediate Hadamard requires intermediate_size % 128 == 0"
                )
        self.dual_a = bool(
            self.intermediate_rotation
            and weight_layout == "trellis_t256"
            and w13_layout == "trellis_t256_proj"
        )
        fc1_source_n_rotation = (
            int(intermediate_size)
            if (weight_layout == "modelopt" and w13_layout == "w13" and is_gated)
            else 0
        )
        self.fc1 = W4A16GemmKernel(
            size_m=size_m,
            size_n=fc1_cols,
            size_k=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            mul_topk_weights=bool(apply_router_weight_on_input),
            tile_n=fc1_tile_n,
            tile_k=fc1_tile_k,
            moe_block_size=moe_block_size,
            max_m_blocks=max_m_blocks,
            element_dtype=element_dtype,
            epilogue_activation=None if is_gated else "relu2",
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=w13_layout,
            trellis_bits=self.trellis_bits,
            trellis_codebook=self.trellis_codebook,
            trellis_pair_kind=self.fc1_trellis_pair_kind,
            trellis_rate_axis=("n" if self.fc1_trellis_pair_kind is not None else None),
            source_n_rotation=fc1_source_n_rotation,
            single_token_route_fast_path=size_m == 1 and not self.direct_topk_routes,
            direct_topk_routes=self.direct_topk_routes,
            dual_a=self.dual_a,
            route_major_a=self.full_rotation,
            schedule_whole_tiles=self.schedule_whole_tiles,
            dynamic_num_experts=self.dynamic_num_experts,
            pipeline_stages=pipeline_stages,
            skip_empty_m_blocks=skip_empty_m_blocks,
            small_m_occupancy=small_m_occupancy,
            csf_inline_scales=csf_inline_scales,
        )
        if (
            _FP32_TOPK_WEIGHTS
            and bool(apply_router_weight_on_input)
            and not self.full_rotation
        ):
            raise ValueError(
                "B12X_W4A16_FP32_TOPK_WEIGHTS does not support "
                "apply_router_weight_on_input"
            )
        self.fc2 = W4A16GemmKernel(
            size_m=routed_rows,
            size_n=hidden_size,
            size_k=intermediate_size,
            num_experts=num_experts,
            top_k=1,
            mul_topk_weights=(
                not bool(apply_router_weight_on_input)
                and not self.full_rotation
                # The top-k sum applies FP32 weights instead (the fused-sum
                # decode epilogue has no separate sum and keeps its multiply).
                and not (_FP32_TOPK_WEIGHTS and not self.tc_decode_fused_sum)
            ),
            tile_n=fc2_tile_n,
            tile_k=fc2_tile_k,
            moe_block_size=self.fc2_moe_block_size,
            max_m_blocks=(
                max_m_blocks * self.moe_block_size // self.fc2_moe_block_size
            ),
            element_dtype=element_dtype,
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=("packed" if weight_layout == "trellis_t256" else w13_layout),
            trellis_bits=self.trellis_bits,
            trellis_codebook=self.trellis_codebook,
            trellis_pair_kind=self.fc2_trellis_pair_kind,
            trellis_rate_axis=("k" if self.fc2_trellis_pair_kind is not None else None),
            single_token_route_fast_path=size_m == 1 and not self.direct_topk_routes,
            direct_topk_routes=self.direct_topk_routes,
            fused_topk_sum=(
                (self.tc_decode_fused_sum and weight_layout not in BLOCK_CODECS)
                or self.prefill_fused_sum_fp32
            ),
            fused_sum_fp32=self.prefill_fused_sum_fp32,
            fused_sum_topk=int(top_k),
            schedule_whole_tiles=self.schedule_whole_tiles,
            dynamic_num_experts=self.dynamic_num_experts,
            schedule_route_block_factor=self.fc2_schedule_route_block_factor,
            pipeline_stages=pipeline_stages,
            skip_empty_m_blocks=skip_empty_m_blocks,
            small_m_occupancy=small_m_occupancy,
            csf_inline_scales=csf_inline_scales,
        )
        self.cta_threads = max(self.fc1.cta_threads, self.fc2.cta_threads)
        if self.fc1.cta_threads != self.fc2.cta_threads:
            raise ValueError(
                "fused W4A16 kernel expects matching FC1/FC2 thread counts"
            )
        self.sms = self.fc1.sms
        self.blocks_per_sm = min(self.fc1.blocks_per_sm, self.fc2.blocks_per_sm)
        self.shared_words = max(self.fc1.shared_words, self.fc2.shared_words)
        self.iq2_lut_bytes = (
            block_codec(weight_layout).lut_bytes(selectors=True)
            if weight_layout in BLOCK_CODECS
            else 0
        )
        self.iq2_xs_lut_off = self.shared_words * 4
        if self.weight_layout in IQ2_CODECS:
            self.shared_words += self.iq2_lut_bytes // 4
            self.fc1.iq2_xs_smem_lut = True
            self.fc2.iq2_xs_smem_lut = True
        if trellis_decode_table not in {"auto", "compact", "full"}:
            raise ValueError("trellis decode table must be auto, compact, or full")
        # Non-Trellis layouts inherit the codebook argument's default, but do
        # not consume its table. Allocating it for block codecs adds 4 KiB
        # outside their occupancy estimate and can reject cooperative launch.
        compact_smem = (
            self.weight_layout == "trellis_t256"
            and self.trellis_codebook == LUT_E4M3
            and _lut_e4m3_smem_enabled()
        )
        lut_offset = (self.shared_words * 4 + 15) // 16 * 16
        # The cooperative grid must retain its occupancy contract. A direct
        # table is eligible only when it fits without reducing resident CTAs.
        direct_eligible = (
            self.weight_layout == "trellis_t256"
            and self.trellis_codebook == LUT_E4M3
            and self.intermediate_hadamard
            and self.trellis_bits == 2
            and self.fc1_trellis_pair_kind is None
            and self.fc2_trellis_pair_kind is None
            and self.blocks_per_sm == 1
            and lut_offset + 65536 <= self.fc1.max_shared_mem
        )
        if trellis_decode_table == "full" and not direct_eligible:
            raise ValueError(
                "full trellis decode table is incompatible with the planned kernel geometry"
            )
        self.trellis_direct_lut = direct_eligible and (
            trellis_decode_table == "full"
            or (trellis_decode_table == "auto" and compact_smem and size_m <= 16)
        )
        self.fc1.trellis_direct_lut = self.trellis_direct_lut
        self.fc2.trellis_direct_lut = self.trellis_direct_lut
        self.lut_e4m3_smem = compact_smem or self.trellis_direct_lut
        self.trellis_lut_smem_bytes = (
            65536 if self.trellis_direct_lut else _LUT_E4M3_SMEM_REGION_BYTES
        )
        self.lut_e4m3_smem_off = 0
        if self.lut_e4m3_smem:
            self.lut_e4m3_smem_off = (self.shared_words * 4 + 15) // 16 * 16
            self.shared_words = (
                self.lut_e4m3_smem_off + self.trellis_lut_smem_bytes
            ) // 4
            if self.shared_words * 4 > self.fc1.max_shared_mem:
                raise ValueError(
                    f"full trellis decode table requires {self.shared_words * 4} bytes"
                )
            self.fc1.lut_e4m3_smem = True
            self.fc2.lut_e4m3_smem = True
        self.barrier_count_off = self.sms * 4
        self.barrier_sense_off = self.sms * 4 + 1
        # The cooperative launch must admit blocks_per_sm CTAs on every SM.
        if self.fc1.weight_layout_block:
            self.smem_carveout = 100
        elif self.blocks_per_sm > 1:
            shared_mem_per_sm = _device_shared_mem_per_sm()
            resident = _resident_cta_shared_bytes(self.shared_words * 4)
            if self.blocks_per_sm * resident > shared_mem_per_sm:
                raise ValueError(
                    "fused W4A16 plan does not fit its residency: "
                    f"{self.blocks_per_sm} CTAs of {self.shared_words * 4} shared bytes"
                )
            self.smem_carveout = _cooperative_smem_carveout(
                blocks_per_sm=self.blocks_per_sm,
                shared_bytes=self.shared_words * 4,
                shared_mem_per_sm=shared_mem_per_sm,
            )
        else:
            self.smem_carveout = None

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.hidden_size,
            self.intermediate_size,
            self.fc1_cols,
            None if self.dynamic_num_experts else self.num_experts,
            self.dynamic_num_experts,
            self.top_k,
            self.activation,
            self.activation_is_gated,
            self.activation_is_situ,
            self.activation_is_swigluoai,
            self.has_swiglu_limit,
            self.swiglu_limit,
            self.swiglu_clamp_after_silu,
            self.swiglu_alpha,
            self.swiglu_beta,
            self.weight_layout,
            self.trellis_bits,
            self.trellis_codebook,
            self.fc1_trellis_pair_kind,
            self.fc2_trellis_pair_kind,
            self.scale_format,
            self.apply_router_weight_on_input,
            self.zero_fc2_output,
            self.element_dtype,
            self.fast_math,
            self.direct_topk_routes,
            self.use_expert_map,
            self.tc_zero_output,
            self.collect_activation_amax,
            self.intermediate_rotation,
            self.dual_a,
            self.full_rotation,
            self.intermediate_hadamard,
            self.broadcast_suh,
            self.rotation_input_dtype,
            self.lut_e4m3_smem,
            self.trellis_direct_lut,
            self.small_m_splitk,
            self.fc1.__cache_key__,
            self.fc2.__cache_key__,
            self.cta_threads,
            self.sms,
            self.shared_words,
            self.blocks_per_sm,
        )

    @cute.jit
    def _cast_elem(self, x: cutlass.Float32):
        if cutlass.const_expr(self.is_fp16):
            return cutlass.Float16(x)
        return cutlass.BFloat16(x)

    @cute.jit
    def _clamp_swiglu_inputs(
        self,
        gate: cutlass.Float32,
        up: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = cutlass.Float32(self.swiglu_limit)
            neg_limit = cutlass.Float32(-self.swiglu_limit)
            if cutlass.const_expr(not self.swiglu_clamp_after_silu):
                if gate > limit:
                    gate = limit
            if up > limit:
                up = limit
            if up < neg_limit:
                up = neg_limit
        return gate, up

    @cute.jit
    def _emit_expert_map_tile(
        self,
        is_fc1: cutlass.Constexpr,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_i32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        global_topk_ids_i32_flat: cute.Tensor,
        expert_map_i32_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        # Direct decode keeps the router's global ids in route-major order.
        # Resolve the compact weight row at the point of use, and reject both
        # invalid global ids and unmapped (-1) entries before any weight access.
        global_expert = global_topk_ids_i32_flat[route_block_idx].to(Int32)
        local_expert = Int32(-1)
        if global_expert >= Int32(0) and global_expert < route_num_experts:
            local_expert = expert_map_i32_flat[global_expert].to(Int32)
        if local_expert >= Int32(0) and local_expert < weight_num_experts:
            if cutlass.const_expr(is_fc1):
                self.fc1._run_tile(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    global_topk_ids_i32_flat,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
                    smem_base,
                    tid,
                    route_block_idx,
                    local_expert,
                    output_n_tile,
                    reduce_k_tile,
                    reduce_tile_count,
                    reduce_slice_count,
                    reduce_slice_idx,
                    lock_slot,
                    active_size_m,
                )
            else:
                self.fc2._run_tile(
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    b_i32_flat,
                    c_bf16_flat,
                    scales_i32_flat,
                    global_scale,
                    global_topk_ids_i32_flat,
                    topk_weights_flat,
                    c_tmp_f32_flat,
                    locks_i32_flat,
                    trellis_lut_addr,
                    smem_base,
                    tid,
                    route_block_idx,
                    local_expert,
                    output_n_tile,
                    reduce_k_tile,
                    reduce_tile_count,
                    reduce_slice_count,
                    reduce_slice_idx,
                    lock_slot,
                    active_size_m,
                )

    @cute.jit
    def __call__(
        self,
        a_bf16_ptr: cute.Pointer,
        a_alt_bf16_ptr: cute.Pointer,
        rotation_input_ptr: cute.Pointer,
        w13_ptr: cute.Pointer,
        w2_ptr: cute.Pointer,
        w13_elements: cutlass.Int64,
        w2_elements: cutlass.Int64,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_ptr: cute.Pointer,
        w2_scales_ptr: cute.Pointer,
        w13_global_scale_ptr: cute.Pointer,
        w2_global_scale_ptr: cute.Pointer,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        topk_weights_ptr: cute.Pointer,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        rot_scales_ptr: cute.Pointer,
        suh_gate_ptr: cute.Pointer,
        suh_up_ptr: cute.Pointer,
        expert_map_ptr: cute.Pointer,
        fc1_trellis_lut_ptr: cute.Pointer,
        fc2_trellis_lut_ptr: cute.Pointer,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        expert_count = Int64(weight_num_experts)
        w13_i32_flat = cute.make_tensor(
            w13_ptr,
            layout=cute.make_layout((Int64(w13_elements),), stride=(1,)),
        )
        w2_i32_flat = cute.make_tensor(
            w2_ptr,
            layout=cute.make_layout((Int64(w2_elements),), stride=(1,)),
        )
        w13_metadata_elements = expert_count * Int64(
            self.fc1.size_k
            // self.fc1.block_k
            * self.fc1.size_n
            * self.fc1.block_metadata_bytes
            // 4
        )
        w2_metadata_elements = expert_count * Int64(
            self.fc2.size_k
            // self.fc2.block_k
            * self.fc2.size_n
            * self.fc2.block_metadata_bytes
            // 4
        )
        if cutlass.const_expr(not self.fc1.weight_layout_block):
            w13_metadata_elements = (
                expert_count
                if cutlass.const_expr(self.fc1.trellis_pair_compact_offsets)
                else expert_count
                * Int64(self.fc1.scale_k_groups)
                * Int64(self.fc1.scale_size_n // 4)
            )
        if cutlass.const_expr(not self.fc2.weight_layout_block):
            w2_metadata_elements = (
                expert_count
                if cutlass.const_expr(self.fc2.trellis_pair_compact_offsets)
                else expert_count
                * Int64(self.fc2.scale_k_groups)
                * Int64(self.fc2.scale_size_n // 4)
            )
        if cutlass.const_expr(self.fc1.native_nvfp4_scales):
            w13_metadata_elements = expert_count * Int64(
                _covering_count(self.fc1.size_n, 128)
                * 128
                * _covering_count(self.fc1.size_k // 16, 4)
            )
            w2_metadata_elements = expert_count * Int64(
                _covering_count(self.fc2.size_n, 128)
                * 128
                * _covering_count(self.fc2.size_k // 16, 4)
            )
        w13_scales_i32_flat = cute.make_tensor(
            w13_scales_ptr,
            layout=cute.make_layout((w13_metadata_elements,), stride=(1,)),
        )
        w2_scales_i32_flat = cute.make_tensor(
            w2_scales_ptr,
            layout=cute.make_layout((w2_metadata_elements,), stride=(1,)),
        )
        w13_global_scale = cute.make_tensor(
            w13_global_scale_ptr,
            layout=cute.make_layout((expert_count,), stride=(1,)),
        )
        w2_global_scale = cute.make_tensor(
            w2_global_scale_ptr,
            layout=cute.make_layout((expert_count,), stride=(1,)),
        )
        rot_rows = Int64(active_m) * Int64(self.top_k)
        if cutlass.const_expr(self.full_rotation):
            rot_rows = expert_count
        rot_width = 3 * self.intermediate_size
        if cutlass.const_expr(self.intermediate_hadamard):
            rot_width = 6 * self.intermediate_size
        rot_scales_flat = cute.make_tensor(
            rot_scales_ptr,
            layout=cute.make_layout(
                (rot_rows * Int64(rot_width),),
                stride=(1,),
            ),
        )
        suh_rows = expert_count
        if cutlass.const_expr(self.broadcast_suh):
            suh_rows = Int64(1)
        suh_gate_flat = cute.make_tensor(
            suh_gate_ptr,
            layout=cute.make_layout(
                (suh_rows * Int64(self.hidden_size),),
                stride=(1,),
            ),
        )
        suh_up_flat = cute.make_tensor(
            suh_up_ptr,
            layout=cute.make_layout(
                (suh_rows * Int64(self.hidden_size),),
                stride=(1,),
            ),
        )
        a_rows = Int64(active_m)
        if cutlass.const_expr(self.full_rotation):
            a_rows = Int64(active_m) * Int64(self.top_k)
        a_bf16_flat = cute.make_tensor(
            a_bf16_ptr,
            layout=cute.make_layout((a_rows * Int64(self.hidden_size),), stride=(1,)),
        )
        a_alt_bf16_flat = cute.make_tensor(
            a_alt_bf16_ptr,
            layout=cute.make_layout((a_rows * Int64(self.hidden_size),), stride=(1,)),
        )
        rotation_input_flat = cute.make_tensor(
            rotation_input_ptr,
            layout=cute.make_layout(
                (Int64(active_m) * Int64(self.hidden_size),), stride=(1,)
            ),
        )
        topk_weights_flat = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout(
                (Int64(active_m) * Int64(self.top_k),), stride=(1,)
            ),
        )
        expert_map_flat = cute.make_tensor(
            expert_map_ptr,
            layout=cute.make_layout(
                (Int64(route_num_experts) + Int64(1),), stride=(1,)
            ),
        )
        fc1_trellis_lut_flat = cute.make_tensor(
            fc1_trellis_lut_ptr,
            layout=cute.make_layout(
                (Int64(_LUT_E4M3_VALUE_TABLE_ENTRIES),), stride=(1,)
            ),
        )
        fc2_trellis_lut_flat = cute.make_tensor(
            fc2_trellis_lut_ptr,
            layout=cute.make_layout(
                (Int64(_LUT_E4M3_VALUE_TABLE_ENTRIES),), stride=(1,)
            ),
        )
        grid = (grid_x, 1, 1)
        self.kernel(
            a_bf16_flat,
            a_alt_bf16_flat,
            rotation_input_flat,
            w13_i32_flat,
            w2_i32_flat,
            fc1_bf16_flat,
            activated_bf16_flat,
            fc2_bf16_flat,
            w13_scales_i32_flat,
            w2_scales_i32_flat,
            w13_global_scale,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            activation_amax_flat,
            layer_idx,
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            rot_scales_flat,
            suh_gate_flat,
            suh_up_flat,
            expert_map_flat,
            fc1_trellis_lut_flat,
            fc2_trellis_lut_flat,
            weight_num_experts,
            route_num_experts,
            active_m,
        ).launch(
            grid=grid,
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            # The fused body crosses software all-CTA barriers between FC1,
            # activation, and FC2. Require whole-grid admission so unrelated
            # work cannot occupy an SM while resident CTAs wait for peers that
            # have not been scheduled yet.
            cooperative=True,
            # Occupancy uses the full SMEM budget. CUTLASS's automatic carveout
            # omits the per-CTA reservation: two 32 KiB Q8_0 CTAs request a
            # 64 KiB carveout but cannot both reside there, rejecting the
            # cooperative launch. Block codecs take the whole budget; other
            # layouts the smallest carveout that admits their planned CTAs.
            preferred_smem_carveout=self.smem_carveout,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        rotation_input_flat: cute.Tensor,
        w13_i32_flat: cute.Tensor,
        w2_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_i32_flat: cute.Tensor,
        w2_scales_i32_flat: cute.Tensor,
        w13_global_scale: cute.Tensor,
        w2_global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        topk_weights_flat: cute.Tensor,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        rot_scales_flat: cute.Tensor,
        suh_gate_flat: cute.Tensor,
        suh_up_flat: cute.Tensor,
        expert_map_flat: cute.Tensor,
        fc1_trellis_lut_flat: cute.Tensor,
        fc2_trellis_lut_flat: cute.Tensor,
        weight_num_experts: cutlass.Int32,
        route_num_experts: cutlass.Int32,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_x_raw)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        fc1_trellis_lut_addr = get_ptr_as_int64(fc1_trellis_lut_flat, Int32(0))
        fc2_trellis_lut_addr = get_ptr_as_int64(fc2_trellis_lut_flat, Int32(0))

        # The LUT ABI slot carries the shared byte offset for either the
        # compact value table or the complete 2-bit direct table.
        fc1_phase_lut_addr = fc1_trellis_lut_addr
        fc2_phase_lut_addr = fc2_trellis_lut_addr
        if cutlass.const_expr(self.weight_layout in IQ2_CODECS):
            for i in cutlass.range_constexpr(
                _covering_count(self.iq2_lut_bytes // 16, self.cta_threads)
            ):
                chunk = Int32(i * self.cta_threads) + tid
                if chunk < Int32(self.iq2_lut_bytes // 16):
                    cp_async4_shared_global(
                        smem_base + Int32(self.iq2_xs_lut_off) + chunk * Int32(16),
                        fc1_trellis_lut_addr + Int64(chunk) * Int64(16),
                    )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            table_addr = Int64(smem_base + Int32(self.iq2_xs_lut_off))
            fc1_phase_lut_addr = table_addr
            fc2_phase_lut_addr = table_addr
        if cutlass.const_expr(self.lut_e4m3_smem):
            self._lut_smem_copy(
                fc1_trellis_lut_addr,
                smem_base + Int32(self.lut_e4m3_smem_off),
                self.trellis_lut_smem_bytes,
                tid,
            )
            cute.arch.sync_threads()
            table_addr = Int64(smem_base + Int32(self.lut_e4m3_smem_off))
            fc1_phase_lut_addr = table_addr
            fc2_phase_lut_addr = table_addr

        fc1_emit_tile = None
        fc2_emit_tile = None
        if cutlass.const_expr(self.use_expert_map):
            fc1_emit_tile = partial(
                self._emit_expert_map_tile,
                True,
                a_bf16_flat,
                a_alt_bf16_flat,
                w13_i32_flat,
                fc1_bf16_flat
                if cutlass.const_expr(self.activation_is_gated)
                else activated_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                expert_map_flat,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                fc1_phase_lut_addr,
                smem_base,
                tid,
                active_m,
                weight_num_experts,
                route_num_experts,
            )
            fc2_emit_tile = partial(
                self._emit_expert_map_tile,
                False,
                activated_bf16_flat,
                activated_bf16_flat,
                w2_i32_flat,
                (
                    fc1_bf16_flat
                    if cutlass.const_expr(
                        self.tc_decode_fused_sum and self.weight_layout in BLOCK_CODECS
                    )
                    else fc2_bf16_flat
                ),
                w2_scales_i32_flat,
                w2_global_scale,
                packed_route_indices,
                expert_map_flat,
                topk_weights_flat,
                fc2_c_tmp_f32_flat,
                locks_i32_flat,
                fc2_phase_lut_addr,
                smem_base,
                tid,
                active_m * Int32(self.top_k),
                weight_num_experts,
                route_num_experts,
            )
        self._moe_body(
            a_bf16_flat,
            a_alt_bf16_flat,
            rotation_input_flat,
            w13_i32_flat,
            w2_i32_flat,
            fc1_bf16_flat,
            activated_bf16_flat,
            fc2_bf16_flat,
            w13_scales_i32_flat,
            w2_scales_i32_flat,
            w13_global_scale,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            activation_amax_flat,
            layer_idx,
            topk_weights_flat,
            fc1_c_tmp_f32_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            rot_scales_flat,
            suh_gate_flat,
            suh_up_flat,
            expert_map_flat,
            fc1_trellis_lut_addr,
            fc2_trellis_lut_addr,
            weight_num_experts,
            route_num_experts,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m,
            fc1_emit_tile,
            fc2_emit_tile,
        )

    @cute.jit
    def _moe_body(
        self,
        a_bf16_flat: cute.Tensor,
        a_alt_bf16_flat: cute.Tensor,
        rotation_input_flat: cute.Tensor,
        w13_i32_flat: cute.Tensor,
        w2_i32_flat: cute.Tensor,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        fc2_bf16_flat: cute.Tensor,
        w13_scales_i32_flat: cute.Tensor,
        w2_scales_i32_flat: cute.Tensor,
        w13_global_scale: cute.Tensor,
        w2_global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        topk_weights_flat: cute.Tensor,
        fc1_c_tmp_f32_flat: cute.Tensor,
        fc2_c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        rot_scales_flat: cute.Tensor,
        suh_gate_flat: cute.Tensor,
        suh_up_flat: cute.Tensor,
        expert_map_flat: cute.Tensor,
        fc1_trellis_lut_addr: Int64,
        fc2_trellis_lut_addr: Int64,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
        fc1_emit_tile: cutlass.Constexpr = None,
        fc2_emit_tile: cutlass.Constexpr = None,
    ):
        # Phase assembly shared by the single-tier fused kernel and the hybrid
        # multi-tier entry: FC1, activation, FC2, and optional output reduction,
        # with grid barriers between dependent phases. The emit hooks delegate per-tile expert resolution and
        # dispatch (used by the hybrid route map); None keeps the single-tier
        # resolution inside _run_persistent_gemm.
        # The trellis LUT parameters carry the raw global address unless the
        # single-tier entry staged the value table in shared memory.
        fc1_phase_lut = fc1_trellis_lut_addr
        fc2_phase_lut = fc2_trellis_lut_addr
        if cutlass.const_expr(self.weight_layout in IQ2_CODECS):
            table_addr = Int64(smem_base + Int32(self.iq2_xs_lut_off))
            fc1_phase_lut = table_addr
            fc2_phase_lut = table_addr
        if cutlass.const_expr(self.lut_e4m3_smem):
            table_addr = Int64(smem_base + Int32(self.lut_e4m3_smem_off))
            fc1_phase_lut = table_addr
            fc2_phase_lut = table_addr
        if cutlass.const_expr(self.full_rotation):
            if cutlass.const_expr(self.intermediate_hadamard):
                self._run_input_rotation_intermediate_hadamard(
                    rotation_input_flat,
                    a_bf16_flat,
                    suh_gate_flat,
                    packed_route_indices,
                    block_expert_ids,
                    packed_route_count,
                    expert_map_flat,
                    weight_num_experts,
                    route_num_experts,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
            else:
                self._run_input_rotation(
                    rotation_input_flat,
                    a_bf16_flat,
                    a_alt_bf16_flat,
                    suh_gate_flat,
                    suh_up_flat,
                    packed_route_indices,
                    block_expert_ids,
                    packed_route_count,
                    expert_map_flat,
                    weight_num_experts,
                    route_num_experts,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
            self._grid_barrier(locks_i32_flat, tid, grid_x)
        if cutlass.const_expr(
            (self.tc_decode_fused_sum and self.weight_layout not in BLOCK_CODECS)
            or self.prefill_fused_sum_fp32
        ):
            # FC2 route reduction atomically accumulates into one row per token.
            # Every CTA zeroes a grid-strided slice before FC1. The mandatory
            # post-FC1 grid barrier orders these stores before every FC2 atomic.
            # ``tc_zero_output=False`` is valid only when a paired launch has
            # already zeroed the same output and participates in that barrier.
            if cutlass.const_expr(self.tc_zero_output):
                zidx = cta * Int32(self.cta_threads) + tid
                zstride = grid_x * Int32(self.cta_threads)
                zzero = (
                    cutlass.Float32(0.0)
                    if cutlass.const_expr(self.prefill_fused_sum_fp32)
                    else self._cast_elem(cutlass.Float32(0.0))
                )
                if cutlass.const_expr(self.prefill_fused_sum_fp32):
                    zidx_i64 = Int64(zidx)
                    zstride_i64 = Int64(zstride)
                    ztotal_i64 = Int64(active_m) * Int64(self.hidden_size)
                    while zidx_i64 < ztotal_i64:
                        fc2_bf16_flat[zidx_i64] = zzero
                        zidx_i64 += zstride_i64
                else:
                    ztotal = active_m * Int32(self.hidden_size)
                    while zidx < ztotal:
                        fc2_bf16_flat[zidx] = zzero
                        zidx += zstride

        if cutlass.const_expr(self.activation_is_gated):
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                a_alt_bf16_flat,
                w13_i32_flat,
                fc1_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                fc1_phase_lut,
                smem_base,
                tid,
                cta,
                grid_x,
                active_m,
                fc1_emit_tile,
            )
            self._grid_barrier(locks_i32_flat, tid, grid_x)
            if cutlass.const_expr(self.full_rotation):
                if cutlass.const_expr(self.intermediate_hadamard):
                    self._run_activation_intermediate_hadamard(
                        fc1_bf16_flat,
                        activated_bf16_flat,
                        rot_scales_flat,
                        packed_route_indices,
                        block_expert_ids,
                        packed_route_count,
                        expert_map_flat,
                        weight_num_experts,
                        route_num_experts,
                        tid,
                        cta,
                        grid_x,
                        active_m,
                    )
                else:
                    self._run_activation_compact(
                        fc1_bf16_flat,
                        activated_bf16_flat,
                        rot_scales_flat,
                        packed_route_indices,
                        block_expert_ids,
                        packed_route_count,
                        expert_map_flat,
                        weight_num_experts,
                        route_num_experts,
                        tid,
                        cta,
                        grid_x,
                        active_m,
                    )
            else:
                self._run_activation(
                    fc1_bf16_flat,
                    activated_bf16_flat,
                    rot_scales_flat,
                    tid,
                    cta,
                    grid_x,
                    active_m,
                )
        else:
            self.fc1._run_persistent_gemm(
                a_bf16_flat,
                a_alt_bf16_flat,
                w13_i32_flat,
                activated_bf16_flat,
                w13_scales_i32_flat,
                w13_global_scale,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                topk_weights_flat,
                fc1_c_tmp_f32_flat,
                locks_i32_flat,
                fc1_phase_lut,
                smem_base,
                tid,
                cta,
                grid_x,
                active_m,
                fc1_emit_tile,
            )
        self._grid_barrier(locks_i32_flat, tid, grid_x)
        if cutlass.const_expr(self.collect_activation_amax):
            self._collect_activation_amax_epilogue(
                a_bf16_flat,
                activated_bf16_flat,
                packed_route_indices,
                block_expert_ids,
                packed_route_count,
                activation_amax_flat,
                layer_idx,
                smem_base,
                tid,
                cta,
                grid_x,
                active_m,
            )
        if cutlass.const_expr(self.zero_fc2_output and not self.prefill_fused_sum_fp32):
            self._zero_fc2_output(fc2_bf16_flat, tid, cta, grid_x, active_m)
            self._grid_barrier(locks_i32_flat, tid, grid_x)
        fc2_store = fc2_bf16_flat
        if cutlass.const_expr(
            self.tc_decode_fused_sum and self.weight_layout in BLOCK_CODECS
        ):
            # FC1 storage is reusable after the activation handoff.
            fc2_store = fc1_bf16_flat
        self.fc2._run_persistent_gemm(
            activated_bf16_flat,
            activated_bf16_flat,
            w2_i32_flat,
            fc2_store,
            w2_scales_i32_flat,
            w2_global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            fc2_c_tmp_f32_flat,
            locks_i32_flat,
            fc2_phase_lut,
            smem_base,
            tid,
            cta,
            grid_x,
            active_m * Int32(self.top_k),
            fc2_emit_tile,
        )
        if cutlass.const_expr(
            self.tc_decode_fused_sum and self.weight_layout in BLOCK_CODECS
        ):
            self._grid_barrier(locks_i32_flat, tid, grid_x)
            self._sum_iq2_xs_routes(
                fc1_bf16_flat,
                fc2_bf16_flat,
                packed_route_indices,
                expert_map_flat,
                weight_num_experts,
                route_num_experts,
                tid,
                cta,
                grid_x,
                active_m,
            )

    @cute.jit
    def _sum_iq2_xs_routes(
        self,
        route_output: cute.Tensor,
        output: cute.Tensor,
        topk_ids: cute.Tensor,
        expert_map: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        idx = Int64(cta) * Int64(self.cta_threads) + Int64(tid)
        stride = Int64(grid_x) * Int64(self.cta_threads)
        total = Int64(active_m) * Int64(self.hidden_size)
        while idx < total:
            token = idx // Int64(self.hidden_size)
            col = idx % Int64(self.hidden_size)
            value = cutlass.Float32(0.0)
            partials = cute.make_rmem_tensor((4,), cutlass.Float32)
            for group in cutlass.range(self.top_k // 4, unroll=1):
                partials.fill(0.0)
                for j in cutlass.range_constexpr(4):
                    slot = group * Int32(4) + Int32(j)
                    route = token * Int64(self.top_k) + Int64(slot)
                    expert = topk_ids[route].to(Int32)
                    if cutlass.const_expr(self.use_expert_map):
                        global_expert = expert
                        expert = Int32(-1)
                        if (
                            global_expert >= Int32(0)
                            and global_expert < route_num_experts
                        ):
                            expert = expert_map[global_expert].to(Int32)
                    if expert >= Int32(0) and expert < weight_num_experts:
                        offset = route * Int64(self.hidden_size) + col
                        partials[j] = route_output[offset].to(cutlass.Float32)
                for j in cutlass.range_constexpr(4):
                    value += partials[j]
            for slot in cutlass.range_constexpr(self.top_k // 4 * 4, self.top_k):
                route = token * Int64(self.top_k) + Int64(slot)
                expert = topk_ids[route].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map[global_expert].to(Int32)
                if expert >= Int32(0) and expert < weight_num_experts:
                    offset = route * Int64(self.hidden_size) + col
                    value += route_output[offset].to(cutlass.Float32)
            if cutlass.const_expr(not self.tc_zero_output):
                value += output[idx].to(cutlass.Float32)
            output[idx] = cutlass.BFloat16(value)
            idx += stride

    @cute.jit
    def _lut_smem_copy(
        self,
        src_addr: Int64,
        dst_off: Int32,
        nbytes: cutlass.Constexpr[int],
        tid: Int32,
    ):
        chunks = Int32(int(nbytes) // 16)
        i = tid
        while i < chunks:
            v0, v1, v2, v3 = ld_global_nc_v4_u32(src_addr + Int64(i) * Int64(16))
            st_shared_v4_u32(dst_off + i * Int32(16), v0, v1, v2, v3)
            i += Int32(self.cta_threads)

    @cute.jit
    def _grid_barrier(
        self,
        locks_i32_flat: cute.Tensor,
        tid: Int32,
        grid_x: Int32,
    ):
        cute.arch.sync_threads()
        if tid == Int32(0):
            count_addr = get_ptr_as_int64(locks_i32_flat, Int32(self.barrier_count_off))
            sense_addr = get_ptr_as_int64(locks_i32_flat, Int32(self.barrier_sense_off))
            old_sense = ld_global_acquire_i32(sense_addr)
            # The arrival atomic is relaxed: release this CTA's global writes, which
            # sync_threads made visible to thread 0, before other CTAs can pass.
            threadfence()
            old_count = atomic_add_global_i32(count_addr, Int32(1))
            if old_count == grid_x - Int32(1):
                st_global_i32(count_addr, Int32(0))
                threadfence()
                red_add_global_release_i32(sense_addr, Int32(1))
            else:
                sense = old_sense
                while sense == old_sense:
                    sense = ld_global_acquire_i32(sense_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _run_input_rotation(
        self,
        x_input_flat: cute.Tensor,
        a_gate_flat: cute.Tensor,
        a_up_flat: cute.Tensor,
        suh_gate_flat: cute.Tensor,
        suh_up_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        # One warp owns one (packed route, H128 block).  The packed route id is
        # the logical flattened top-k route, so route//top_k is the source token
        # and route is the row in each projection-specific A scratch.  Padding
        # slots carry the live-route sentinel and are skipped.  Both scale
        # multiplies are rounded to fp16 before the existing fp32-register
        # H128 butterfly, and each H128 result is rounded to fp16 on store.
        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.hidden_size // 128)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        elem = lane * Int32(4)
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            blk = unit - route_pos * nblk
            route = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(Int32)
            if cutlass.const_expr(self.direct_topk_routes):
                route = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                route >= Int32(0)
                and route < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                token = route // Int32(self.top_k)
                col0 = blk * Int32(128) + elem
                x_base = token * Int32(self.hidden_size) + col0
                if cutlass.const_expr(self.broadcast_suh):
                    s_base = col0
                else:
                    s_base = expert * Int32(self.hidden_size) + col0
                out_base = route * Int32(self.hidden_size) + col0

                x0 = cutlass.Float16(
                    x_input_flat[x_base + Int32(0)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x1 = cutlass.Float16(
                    x_input_flat[x_base + Int32(1)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x2 = cutlass.Float16(
                    x_input_flat[x_base + Int32(2)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x3 = cutlass.Float16(
                    x_input_flat[x_base + Int32(3)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                sg0 = suh_gate_flat[s_base + Int32(0)].to(cutlass.Float32)
                sg1 = suh_gate_flat[s_base + Int32(1)].to(cutlass.Float32)
                sg2 = suh_gate_flat[s_base + Int32(2)].to(cutlass.Float32)
                sg3 = suh_gate_flat[s_base + Int32(3)].to(cutlass.Float32)
                su0 = suh_up_flat[s_base + Int32(0)].to(cutlass.Float32)
                su1 = suh_up_flat[s_base + Int32(1)].to(cutlass.Float32)
                su2 = suh_up_flat[s_base + Int32(2)].to(cutlass.Float32)
                su3 = suh_up_flat[s_base + Int32(3)].to(cutlass.Float32)

                g0 = cutlass.Float16(x0 * sg0).to(cutlass.Float32)
                g1 = cutlass.Float16(x1 * sg1).to(cutlass.Float32)
                g2 = cutlass.Float16(x2 * sg2).to(cutlass.Float32)
                g3 = cutlass.Float16(x3 * sg3).to(cutlass.Float32)
                u0 = cutlass.Float16(x0 * su0).to(cutlass.Float32)
                u1 = cutlass.Float16(x1 * su1).to(cutlass.Float32)
                u2 = cutlass.Float16(x2 * su2).to(cutlass.Float32)
                u3 = cutlass.Float16(x3 * su3).to(cutlass.Float32)
                gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
                uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
                a_gate_flat[out_base + Int32(0)] = cutlass.Float16(gh0)
                a_gate_flat[out_base + Int32(1)] = cutlass.Float16(gh1)
                a_gate_flat[out_base + Int32(2)] = cutlass.Float16(gh2)
                a_gate_flat[out_base + Int32(3)] = cutlass.Float16(gh3)
                a_up_flat[out_base + Int32(0)] = cutlass.Float16(uh0)
                a_up_flat[out_base + Int32(1)] = cutlass.Float16(uh1)
                a_up_flat[out_base + Int32(2)] = cutlass.Float16(uh2)
                a_up_flat[out_base + Int32(3)] = cutlass.Float16(uh3)
            unit += gw_stride

    @cute.jit
    def _run_input_rotation_intermediate_hadamard(
        self,
        x_input_flat: cute.Tensor,
        a_shared_flat: cute.Tensor,
        suh_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        """Apply the shared H512 boundary followed by ordinary H128 inputs."""

        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.hidden_size // 512)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        elem = lane * Int32(4)
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            blk = unit - route_pos * nblk
            route = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(Int32)
            if cutlass.const_expr(self.direct_topk_routes):
                route = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                route >= Int32(0)
                and route < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                token = route // Int32(self.top_k)
                col0 = blk * Int32(512) + elem
                x_base = token * Int32(self.hidden_size) + col0

                x00 = cutlass.Float16(
                    x_input_flat[x_base + Int32(0)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x01 = cutlass.Float16(
                    x_input_flat[x_base + Int32(1)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x02 = cutlass.Float16(
                    x_input_flat[x_base + Int32(2)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x03 = cutlass.Float16(
                    x_input_flat[x_base + Int32(3)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x10 = cutlass.Float16(
                    x_input_flat[x_base + Int32(128)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x11 = cutlass.Float16(
                    x_input_flat[x_base + Int32(129)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x12 = cutlass.Float16(
                    x_input_flat[x_base + Int32(130)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x13 = cutlass.Float16(
                    x_input_flat[x_base + Int32(131)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x20 = cutlass.Float16(
                    x_input_flat[x_base + Int32(256)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x21 = cutlass.Float16(
                    x_input_flat[x_base + Int32(257)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x22 = cutlass.Float16(
                    x_input_flat[x_base + Int32(258)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x23 = cutlass.Float16(
                    x_input_flat[x_base + Int32(259)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x30 = cutlass.Float16(
                    x_input_flat[x_base + Int32(384)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x31 = cutlass.Float16(
                    x_input_flat[x_base + Int32(385)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x32 = cutlass.Float16(
                    x_input_flat[x_base + Int32(386)].to(cutlass.Float32)
                ).to(cutlass.Float32)
                x33 = cutlass.Float16(
                    x_input_flat[x_base + Int32(387)].to(cutlass.Float32)
                ).to(cutlass.Float32)

                h00, h01, h02, h03 = self._had128_quad(x00, x01, x02, x03, lane)
                h10, h11, h12, h13 = self._had128_quad(x10, x11, x12, x13, lane)
                h20, h21, h22, h23 = self._had128_quad(x20, x21, x22, x23, lane)
                h30, h31, h32, h33 = self._had128_quad(x30, x31, x32, x33, lane)
                c00, c10, c20, c30 = self._had4_normalized(h00, h10, h20, h30)
                c01, c11, c21, c31 = self._had4_normalized(h01, h11, h21, h31)
                c02, c12, c22, c32 = self._had4_normalized(h02, h12, h22, h32)
                c03, c13, c23, c33 = self._had4_normalized(h03, h13, h23, h33)

                if cutlass.const_expr(self.broadcast_suh):
                    s_base = col0
                else:
                    s_base = expert * Int32(self.hidden_size) + col0
                out_base = route * Int32(self.hidden_size) + col0
                for group in cutlass.range_constexpr(4):
                    offset = Int32(group * 128)
                    if cutlass.const_expr(group == 0):
                        c0, c1, c2, c3 = c00, c01, c02, c03
                    elif cutlass.const_expr(group == 1):
                        c0, c1, c2, c3 = c10, c11, c12, c13
                    elif cutlass.const_expr(group == 2):
                        c0, c1, c2, c3 = c20, c21, c22, c23
                    else:
                        c0, c1, c2, c3 = c30, c31, c32, c33
                    s0 = suh_flat[s_base + offset + Int32(0)].to(cutlass.Float32)
                    s1 = suh_flat[s_base + offset + Int32(1)].to(cutlass.Float32)
                    s2 = suh_flat[s_base + offset + Int32(2)].to(cutlass.Float32)
                    s3 = suh_flat[s_base + offset + Int32(3)].to(cutlass.Float32)
                    c0 = cutlass.Float16(c0 * s0).to(cutlass.Float32)
                    c1 = cutlass.Float16(c1 * s1).to(cutlass.Float32)
                    c2 = cutlass.Float16(c2 * s2).to(cutlass.Float32)
                    c3 = cutlass.Float16(c3 * s3).to(cutlass.Float32)
                    o0, o1, o2, o3 = self._had128_quad(c0, c1, c2, c3, lane)
                    a_shared_flat[out_base + offset + Int32(0)] = cutlass.Float16(o0)
                    a_shared_flat[out_base + offset + Int32(1)] = cutlass.Float16(o1)
                    a_shared_flat[out_base + offset + Int32(2)] = cutlass.Float16(o2)
                    a_shared_flat[out_base + offset + Int32(3)] = cutlass.Float16(o3)
            unit += gw_stride

    @cute.jit
    def _load_intermediate_hadamard_pre_quad(
        self,
        fc1_flat: cute.Tensor,
        rotations_flat: cute.Tensor,
        row: Int32,
        expert: Int32,
        pre_block: Int32,
        lane: Int32,
    ):
        """Decode one logical 128-coordinate intermediate-Hadamard preactivation record."""

        if cutlass.const_expr(self.fc1_trellis_pair_kind is None):
            # The uniform-rate layout divides one interleaved intermediate-Hadamard window
            # across the two physical FC1 slots.
            chunk = lane >> Int32(3)
            chunk_lane = lane & Int32(7)
            channel_group = pre_block * Int32(2) + (chunk >> Int32(1))
            slot = chunk & Int32(1)
            coord = channel_group * Int32(32) + chunk_lane * Int32(4)
        else:
            # A fixed-rate pair slot contains two complete 128-coordinate
            # records.  Apply the record-local output Hadamard before the
            # intermediate Hadamard by selecting one physical slot and half.
            slot = pre_block & Int32(1)
            half = pre_block >> Int32(1)
            if cutlass.const_expr(self.fc1_trellis_pair_kind == "P43"):
                half = half ^ Int32(1)
            coord = half * Int32(128) + lane * Int32(4)
        isz = Int32(self.intermediate_size)
        fc1_base = row * Int32(2 * self.intermediate_size) + slot * isz + coord
        rot_base = expert * Int32(6 * self.intermediate_size)
        scale_base = rot_base + slot * isz + coord
        v0 = fc1_flat[fc1_base + Int32(0)].to(cutlass.Float32)
        v1 = fc1_flat[fc1_base + Int32(1)].to(cutlass.Float32)
        v2 = fc1_flat[fc1_base + Int32(2)].to(cutlass.Float32)
        v3 = fc1_flat[fc1_base + Int32(3)].to(cutlass.Float32)
        h0, h1, h2, h3 = self._had128_quad(v0, v1, v2, v3, lane)
        h0 *= rotations_flat[scale_base + Int32(0)].to(cutlass.Float32)
        h1 *= rotations_flat[scale_base + Int32(1)].to(cutlass.Float32)
        h2 *= rotations_flat[scale_base + Int32(2)].to(cutlass.Float32)
        h3 *= rotations_flat[scale_base + Int32(3)].to(cutlass.Float32)
        h0, h1, h2, h3 = self._had128_quad(h0, h1, h2, h3, lane)
        sign_base = (
            rot_base
            + Int32(3 * self.intermediate_size)
            + pre_block * Int32(128)
            + lane * Int32(4)
        )
        h0 *= rotations_flat[sign_base + Int32(0)].to(cutlass.Float32)
        h1 *= rotations_flat[sign_base + Int32(1)].to(cutlass.Float32)
        h2 *= rotations_flat[sign_base + Int32(2)].to(cutlass.Float32)
        h3 *= rotations_flat[sign_base + Int32(3)].to(cutlass.Float32)
        return h0, h1, h2, h3

    @cute.jit
    def _run_activation_intermediate_hadamard(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        rotations_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        """Recover interleaved gate/up, evaluate SiTU, and prepare FC2."""

        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.intermediate_size // 128)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            post_block = unit - route_pos * nblk
            row = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(Int32)
            if cutlass.const_expr(self.direct_topk_routes):
                row = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                row >= Int32(0)
                and row < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                p0 = post_block * Int32(2)
                a0, a1, a2, a3 = self._load_intermediate_hadamard_pre_quad(
                    fc1_flat, rotations_flat, row, expert, p0, lane
                )
                b0, b1, b2, b3 = self._load_intermediate_hadamard_pre_quad(
                    fc1_flat, rotations_flat, row, expert, p0 + Int32(1), lane
                )
                if cutlass.const_expr(self.activation_is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    aa0 = (
                        beta
                        * cute.math.tanh(a0 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(a0)
                        * linear_beta
                        * cute.math.tanh(a1 / linear_beta, fastmath=self.fast_math)
                    )
                    aa1 = (
                        beta
                        * cute.math.tanh(a2 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(a2)
                        * linear_beta
                        * cute.math.tanh(a3 / linear_beta, fastmath=self.fast_math)
                    )
                    bb0 = (
                        beta
                        * cute.math.tanh(b0 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(b0)
                        * linear_beta
                        * cute.math.tanh(b1 / linear_beta, fastmath=self.fast_math)
                    )
                    bb1 = (
                        beta
                        * cute.math.tanh(b2 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(b2)
                        * linear_beta
                        * cute.math.tanh(b3 / linear_beta, fastmath=self.fast_math)
                    )
                else:
                    aa0 = self._silu_f32(a0) * a1
                    aa1 = self._silu_f32(a2) * a3
                    bb0 = self._silu_f32(b0) * b1
                    bb1 = self._silu_f32(b2) * b3

                source0 = (lane & Int32(15)) << Int32(1)
                source1 = source0 + Int32(1)
                av0 = cute.arch.shuffle_sync(aa0, source0)
                av1 = cute.arch.shuffle_sync(aa1, source0)
                av2 = cute.arch.shuffle_sync(aa0, source1)
                av3 = cute.arch.shuffle_sync(aa1, source1)
                bv0 = cute.arch.shuffle_sync(bb0, source0)
                bv1 = cute.arch.shuffle_sync(bb1, source0)
                bv2 = cute.arch.shuffle_sync(bb0, source1)
                bv3 = cute.arch.shuffle_sync(bb1, source1)
                v0, v1, v2, v3 = av0, av1, av2, av3
                if lane >= Int32(16):
                    v0, v1, v2, v3 = bv0, bv1, bv2, bv3

                isz = Int32(self.intermediate_size)
                col0 = post_block * Int32(128) + lane * Int32(4)
                rot_base = expert * Int32(6 * self.intermediate_size)
                sign_base = rot_base + Int32(5 * self.intermediate_size) + col0
                v0 *= rotations_flat[sign_base + Int32(0)].to(cutlass.Float32)
                v1 *= rotations_flat[sign_base + Int32(1)].to(cutlass.Float32)
                v2 *= rotations_flat[sign_base + Int32(2)].to(cutlass.Float32)
                v3 *= rotations_flat[sign_base + Int32(3)].to(cutlass.Float32)
                v0, v1, v2, v3 = self._had128_quad(v0, v1, v2, v3, lane)

                down_base = rot_base + Int32(2 * self.intermediate_size) + col0
                v0 *= rotations_flat[down_base + Int32(0)].to(cutlass.Float32)
                v1 *= rotations_flat[down_base + Int32(1)].to(cutlass.Float32)
                v2 *= rotations_flat[down_base + Int32(2)].to(cutlass.Float32)
                v3 *= rotations_flat[down_base + Int32(3)].to(cutlass.Float32)
                v0, v1, v2, v3 = self._had128_quad(v0, v1, v2, v3, lane)
                out_base = row * isz + col0
                activated_flat[out_base + Int32(0)] = self._cast_elem(v0)
                activated_flat[out_base + Int32(1)] = self._cast_elem(v1)
                activated_flat[out_base + Int32(2)] = self._cast_elem(v2)
                activated_flat[out_base + Int32(3)] = self._cast_elem(v3)
            unit += gw_stride

    @cute.jit
    def _run_activation_compact(
        self,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        rot_scales_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        expert_map_flat: cute.Tensor,
        weight_num_experts: Int32,
        route_num_experts: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        # A9: traverse the already-packed expert blocks and index the persistent
        # [E,3I] table directly.  FC1/activation rows stay at their logical
        # flattened route ids, so packing order cannot change arithmetic or the
        # per-row output location.
        lane = tid & Int32(31)
        warp_in_cta = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        nblk = Int32(self.intermediate_size // 128)
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        if cutlass.const_expr(self.direct_topk_routes):
            route_count = live_routes
        fc1_cols = Int32(self.fc1_cols)
        isz = Int32(self.intermediate_size)
        rot_row = Int32(3 * self.intermediate_size)
        gwarp = cta * warps_per_cta + warp_in_cta
        gw_stride = grid_x * warps_per_cta
        total_units = route_count * nblk
        elem = lane * Int32(4)
        unit = gwarp
        while unit < total_units:
            route_pos = unit // nblk
            blk = unit - route_pos * nblk
            row = packed_route_indices[route_pos].to(Int32)
            expert = block_expert_ids[route_pos // Int32(self.moe_block_size)].to(Int32)
            if cutlass.const_expr(self.direct_topk_routes):
                row = route_pos
                expert = packed_route_indices[route_pos].to(Int32)
                if cutlass.const_expr(self.use_expert_map):
                    global_expert = expert
                    expert = Int32(-1)
                    if global_expert >= Int32(0) and global_expert < route_num_experts:
                        expert = expert_map_flat[global_expert].to(Int32)
            if (
                row >= Int32(0)
                and row < live_routes
                and expert >= Int32(0)
                and expert < weight_num_experts
            ):
                col0 = blk * Int32(128) + elem
                g_base = row * fc1_cols + col0
                u_base = g_base + isz
                s_base = expert * rot_row + col0
                g0 = fc1_bf16_flat[g_base + Int32(0)].to(cutlass.Float32)
                g1 = fc1_bf16_flat[g_base + Int32(1)].to(cutlass.Float32)
                g2 = fc1_bf16_flat[g_base + Int32(2)].to(cutlass.Float32)
                g3 = fc1_bf16_flat[g_base + Int32(3)].to(cutlass.Float32)
                u0 = fc1_bf16_flat[u_base + Int32(0)].to(cutlass.Float32)
                u1 = fc1_bf16_flat[u_base + Int32(1)].to(cutlass.Float32)
                u2 = fc1_bf16_flat[u_base + Int32(2)].to(cutlass.Float32)
                u3 = fc1_bf16_flat[u_base + Int32(3)].to(cutlass.Float32)
                gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
                uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
                svg0 = rot_scales_flat[s_base + Int32(0)].to(cutlass.Float32)
                svg1 = rot_scales_flat[s_base + Int32(1)].to(cutlass.Float32)
                svg2 = rot_scales_flat[s_base + Int32(2)].to(cutlass.Float32)
                svg3 = rot_scales_flat[s_base + Int32(3)].to(cutlass.Float32)
                svu0 = rot_scales_flat[s_base + isz + Int32(0)].to(cutlass.Float32)
                svu1 = rot_scales_flat[s_base + isz + Int32(1)].to(cutlass.Float32)
                svu2 = rot_scales_flat[s_base + isz + Int32(2)].to(cutlass.Float32)
                svu3 = rot_scales_flat[s_base + isz + Int32(3)].to(cutlass.Float32)
                ig0 = gh0 * svg0
                ig1 = gh1 * svg1
                ig2 = gh2 * svg2
                ig3 = gh3 * svg3
                iu0 = uh0 * svu0
                iu1 = uh1 * svu1
                iu2 = uh2 * svu2
                iu3 = uh3 * svu3
                down = isz + isz
                sd0 = rot_scales_flat[s_base + down + Int32(0)].to(cutlass.Float32)
                sd1 = rot_scales_flat[s_base + down + Int32(1)].to(cutlass.Float32)
                sd2 = rot_scales_flat[s_base + down + Int32(2)].to(cutlass.Float32)
                sd3 = rot_scales_flat[s_base + down + Int32(3)].to(cutlass.Float32)
                if cutlass.const_expr(self.activation_is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    a0 = (
                        beta
                        * cute.math.tanh(ig0 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig0)
                        * linear_beta
                        * cute.math.tanh(iu0 / linear_beta, fastmath=self.fast_math)
                        * sd0
                    )
                    a1 = (
                        beta
                        * cute.math.tanh(ig1 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig1)
                        * linear_beta
                        * cute.math.tanh(iu1 / linear_beta, fastmath=self.fast_math)
                        * sd1
                    )
                    a2 = (
                        beta
                        * cute.math.tanh(ig2 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig2)
                        * linear_beta
                        * cute.math.tanh(iu2 / linear_beta, fastmath=self.fast_math)
                        * sd2
                    )
                    a3 = (
                        beta
                        * cute.math.tanh(ig3 / beta, fastmath=self.fast_math)
                        * self._sigmoid_f32(ig3)
                        * linear_beta
                        * cute.math.tanh(iu3 / linear_beta, fastmath=self.fast_math)
                        * sd3
                    )
                else:
                    a0 = self._silu_f32(ig0) * iu0 * sd0
                    a1 = self._silu_f32(ig1) * iu1 * sd1
                    a2 = self._silu_f32(ig2) * iu2 * sd2
                    a3 = self._silu_f32(ig3) * iu3 * sd3
                o0, o1, o2, o3 = self._had128_quad(a0, a1, a2, a3, lane)
                out_base = row * isz + col0
                activated_bf16_flat[out_base + Int32(0)] = self._cast_elem(o0)
                activated_bf16_flat[out_base + Int32(1)] = self._cast_elem(o1)
                activated_bf16_flat[out_base + Int32(2)] = self._cast_elem(o2)
                activated_bf16_flat[out_base + Int32(3)] = self._cast_elem(o3)
            unit += gw_stride

    @cute.jit
    def _zero_fc2_output(
        self,
        fc2_bf16_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        idx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = active_m * Int32(self.top_k * self.hidden_size)
        zero = self._cast_elem(cutlass.Float32(0.0))
        while idx < total:
            fc2_bf16_flat[idx] = zero
            idx += stride

    @cute.jit
    def _reduce_and_red_activation_amax(
        self,
        local_max: cutlass.Float32,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        expert_idx: Int32,
        slot: Int32,
        smem_base: Int32,
        tid: Int32,
    ):
        lane = tid & Int32(31)
        warp_idx = tid // Int32(32)
        warp_amax = warp_reduce(local_max, fmax_f32)
        if lane == Int32(0):
            st_shared_f32(smem_base + warp_idx * Int32(4), warp_amax)
        cute.arch.sync_threads()

        if warp_idx == Int32(0):
            block_amax = cutlass.Float32(0.0)
            if lane < Int32(self.cta_threads // 32):
                block_amax = ld_shared_f32(smem_base + lane * Int32(4))
            block_amax = warp_reduce(block_amax, fmax_f32)
            if lane == Int32(0) and block_amax > cutlass.Float32(0.0):
                out_idx = (layer_idx * Int32(self.num_experts) + expert_idx) * Int32(
                    2
                ) + slot
                red_max_global_f32_nonnegative(
                    get_ptr_as_int64(activation_amax_flat, out_idx),
                    block_amax,
                )
        cute.arch.sync_threads()

    @cute.jit
    def _collect_activation_amax_epilogue(
        self,
        a_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        activation_amax_flat: cute.Tensor,
        layer_idx: cutlass.Int32,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        live_routes = active_m * Int32(self.top_k)
        route_count = packed_route_count[Int32(0)].to(Int32)
        route_blocks = (route_count + Int32(self.moe_block_size) - Int32(1)) // Int32(
            self.moe_block_size
        )
        expert_idx = cta
        while expert_idx < Int32(self.num_experts):
            local_fc1 = cutlass.Float32(0.0)
            local_fc2 = cutlass.Float32(0.0)
            has_route_block = Int32(0)
            block_idx = Int32(0)
            while block_idx < route_blocks:
                block_expert = block_expert_ids[block_idx].to(Int32)
                if block_expert == expert_idx:
                    has_route_block = Int32(1)
                    route_pos = block_idx * Int32(self.moe_block_size)
                    route_stop = route_pos + Int32(self.moe_block_size)
                    if route_stop > route_count:
                        route_stop = route_count
                    while route_pos < route_stop:
                        route_idx = packed_route_indices[route_pos].to(Int32)
                        if route_idx >= Int32(0) and route_idx < live_routes:
                            token_idx = route_idx // Int32(self.top_k)
                            fc1_col = tid
                            while fc1_col < Int32(self.hidden_size):
                                v1 = a_bf16_flat[
                                    token_idx * Int32(self.hidden_size) + fc1_col
                                ].to(cutlass.Float32)
                                local_fc1 = fmax_f32(local_fc1, fabs_f32(v1))
                                fc1_col += Int32(self.cta_threads)

                            fc2_col = tid
                            while fc2_col < Int32(self.intermediate_size):
                                v2 = activated_bf16_flat[
                                    route_idx * Int32(self.intermediate_size) + fc2_col
                                ].to(cutlass.Float32)
                                local_fc2 = fmax_f32(local_fc2, fabs_f32(v2))
                                fc2_col += Int32(self.cta_threads)
                        route_pos += Int32(1)
                block_idx += Int32(1)

            if has_route_block != Int32(0):
                self._reduce_and_red_activation_amax(
                    local_fc1,
                    activation_amax_flat,
                    layer_idx,
                    expert_idx,
                    Int32(0),
                    smem_base,
                    tid,
                )
                self._reduce_and_red_activation_amax(
                    local_fc2,
                    activation_amax_flat,
                    layer_idx,
                    expert_idx,
                    Int32(1),
                    smem_base,
                    tid,
                )
            expert_idx += grid_x

    @cute.jit
    def _had128_quad(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
        lane: Int32,
    ):
        # Blockwise-128 Walsh-Hadamard (Sylvester/natural order) across ONE warp,
        # bit-identical in structure to exllamav3 had_hf_r_128_inner: lane `t`
        # owns the 4 consecutive elements [4t..4t+3], so element index within the
        # 128-block is 4*lane + reg.  H_128 = H_4(reg, in-register) (x) H_32(lane,
        # warp butterfly), then * r_scale = 1/sqrt(128).
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        h0 = s0 + s1
        h1 = d0 + d1
        h2 = s0 - s1
        h3 = d0 - d1
        for i in cutlass.range_constexpr(5):
            st = 1 << i
            p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
            p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
            p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
            p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
            if (lane & Int32(st)) != Int32(0):
                h0 = p0 - h0
                h1 = p1 - h1
                h2 = p2 - h2
                h3 = p3 - h3
            else:
                h0 = p0 + h0
                h1 = p1 + h1
                h2 = p2 + h2
                h3 = p3 + h3
        rs = cutlass.Float32(0.088388347648)  # scale / sqrt(128), scale = 1
        return h0 * rs, h1 * rs, h2 * rs, h3 * rs

    @cute.jit
    def _had4_normalized(
        self,
        v0: cutlass.Float32,
        v1: cutlass.Float32,
        v2: cutlass.Float32,
        v3: cutlass.Float32,
    ):
        """Normalized natural-order H4 over four H128 subblocks."""
        s0 = v0 + v1
        d0 = v0 - v1
        s1 = v2 + v3
        d1 = v2 - v3
        rs = cutlass.Float32(0.5)
        return (
            (s0 + s1) * rs,
            (d0 + d1) * rs,
            (s0 - s1) * rs,
            (d0 - d1) * rs,
        )

    @cute.jit
    def _sigmoid_f32(self, x: cutlass.Float32) -> cutlass.Float32:
        if cutlass.const_expr(self.fast_math):
            e = cute.math.exp(-x, fastmath=True)
        else:
            e = cute.math.exp(-x, fastmath=False)
        return cutlass.Float32(1.0) / (cutlass.Float32(1.0) + e)

    @cute.jit
    def _silu_f32(self, x: cutlass.Float32) -> cutlass.Float32:
        return x * self._sigmoid_f32(x)

    @cute.jit
    def _gated_activation_elem(self, gate: cutlass.Float32, up: cutlass.Float32):
        # Per-element math of the scalar gated loop in _run_activation.
        gate, up = self._clamp_swiglu_inputs(gate, up)
        sigmoid_arg = gate
        up_term = up
        if cutlass.const_expr(self.activation_is_swigluoai):
            sigmoid_arg = cutlass.Float32(self.swiglu_alpha) * gate
            up_term = up + cutlass.Float32(self.swiglu_beta)
        if cutlass.const_expr(self.fast_math):
            exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=True)
        else:
            exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=False)
        sigmoid = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + exp_neg_gate)
        silu = gate * sigmoid
        if cutlass.const_expr(self.has_swiglu_limit and self.swiglu_clamp_after_silu):
            if silu > cutlass.Float32(self.swiglu_limit):
                silu = cutlass.Float32(self.swiglu_limit)
        if cutlass.const_expr(self.activation_is_situ):
            beta = cutlass.Float32(SITU_DEFAULT_BETA)
            linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
            situ_gate = (
                beta * cute.math.tanh(gate / beta, fastmath=self.fast_math) * sigmoid
            )
            situ_up = linear_beta * cute.math.tanh(
                up / linear_beta,
                fastmath=self.fast_math,
            )
            result = self._cast_elem(situ_gate * situ_up)
        elif cutlass.const_expr(self.activation_is_swigluoai):
            result = self._cast_elem(silu * up_term)
        else:
            result = self._cast_elem(self._cast_elem(silu) * self._cast_elem(up_term))
        return result

    @cute.jit
    def _gated_activation_bf16x2(self, gate2: Uint32, up2: Uint32) -> Uint32:
        hi_mask = Uint32(0xFFFF0000)
        lo = self._gated_activation_elem(
            u32_as_f32(gate2 << Uint32(16)), u32_as_f32(up2 << Uint32(16))
        )
        hi = self._gated_activation_elem(
            u32_as_f32(gate2 & hi_mask), u32_as_f32(up2 & hi_mask)
        )
        return _pack_f32x2_to_bf16x2_rn(lo.to(cutlass.Float32), hi.to(cutlass.Float32))

    @cute.jit
    def _run_activation_gated_x8(
        self,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        # The scalar gated loop with 16-byte gate/up loads and output stores:
        # each thread handles 8 consecutive intermediate columns of one row.
        vec_per_row = Int32(self.intermediate_size // 8)
        vidx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = active_m * Int32(self.top_k) * vec_per_row
        while vidx < total:
            row = vidx // vec_per_row
            col = (vidx - row * vec_per_row) * Int32(8)
            gate_off = row * Int32(self.fc1_cols) + col
            g0, g1, g2, g3 = ld_global_v4_u32(get_ptr_as_int64(fc1_bf16_flat, gate_off))
            u0, u1, u2, u3 = ld_global_v4_u32(
                get_ptr_as_int64(
                    fc1_bf16_flat, gate_off + Int32(self.intermediate_size)
                )
            )
            o0 = self._gated_activation_bf16x2(g0, u0)
            o1 = self._gated_activation_bf16x2(g1, u1)
            o2 = self._gated_activation_bf16x2(g2, u2)
            o3 = self._gated_activation_bf16x2(g3, u3)
            st_global_v4_u32(
                get_ptr_as_int64(
                    activated_bf16_flat, row * Int32(self.intermediate_size) + col
                ),
                o0,
                o1,
                o2,
                o3,
            )
            vidx += stride

    @cute.jit
    def _run_activation(
        self,
        fc1_bf16_flat: cute.Tensor,
        activated_bf16_flat: cute.Tensor,
        rot_scales_flat: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: cutlass.Int32,
    ):
        if cutlass.const_expr(self.intermediate_rotation):
            # Rotation-aware epilogue (trellis_t256 tail).  Warp-cooperative over
            # (routed-row, 128-block) units; each warp owns one 128-wide block of
            # a row's intermediate.  Per row r the FC1 output is [gate(I) | up(I)]
            # (fc1_cols = 2I) and rot_scales_flat[r] = [svh_gate(I)|svh_up(I)|
            # suh_down(I)] (3I fp16).  Applies:
            #   ig2 = had128(gate) * svh_gate ;  iu2 = had128(up) * svh_up
            #   ia  = silu(ig2) * iu2
            #   ia2 = had128(ia * suh_down)           -> activated[r, block]
            lane = tid & Int32(31)
            warp_in_cta = tid >> Int32(5)
            warps_per_cta = Int32(self.cta_threads // 32)
            nblk = Int32(self.intermediate_size // 128)
            fc1_cols = Int32(self.fc1_cols)
            isz = Int32(self.intermediate_size)
            rot_row = Int32(3 * self.intermediate_size)
            gwarp = cta * warps_per_cta + warp_in_cta
            gw_stride = grid_x * warps_per_cta
            total_units = active_m * Int32(self.top_k) * nblk
            e = lane * Int32(4)
            unit = gwarp
            while unit < total_units:
                row = unit // nblk
                blk = unit - row * nblk
                col0 = blk * Int32(128) + e
                g_base = row * fc1_cols + col0
                u_base = g_base + isz
                s_base = row * rot_row + col0
                # load 4 gate + 4 up
                g0 = fc1_bf16_flat[g_base + Int32(0)].to(cutlass.Float32)
                g1 = fc1_bf16_flat[g_base + Int32(1)].to(cutlass.Float32)
                g2 = fc1_bf16_flat[g_base + Int32(2)].to(cutlass.Float32)
                g3 = fc1_bf16_flat[g_base + Int32(3)].to(cutlass.Float32)
                u0 = fc1_bf16_flat[u_base + Int32(0)].to(cutlass.Float32)
                u1 = fc1_bf16_flat[u_base + Int32(1)].to(cutlass.Float32)
                u2 = fc1_bf16_flat[u_base + Int32(2)].to(cutlass.Float32)
                u3 = fc1_bf16_flat[u_base + Int32(3)].to(cutlass.Float32)
                # blockwise-128 Hadamard on gate and up (pre-silu, output side)
                gh0, gh1, gh2, gh3 = self._had128_quad(g0, g1, g2, g3, lane)
                uh0, uh1, uh2, uh3 = self._had128_quad(u0, u1, u2, u3, lane)
                # post-scale svh_gate / svh_up
                svg0 = rot_scales_flat[s_base + Int32(0)].to(cutlass.Float32)
                svg1 = rot_scales_flat[s_base + Int32(1)].to(cutlass.Float32)
                svg2 = rot_scales_flat[s_base + Int32(2)].to(cutlass.Float32)
                svg3 = rot_scales_flat[s_base + Int32(3)].to(cutlass.Float32)
                svu0 = rot_scales_flat[s_base + isz + Int32(0)].to(cutlass.Float32)
                svu1 = rot_scales_flat[s_base + isz + Int32(1)].to(cutlass.Float32)
                svu2 = rot_scales_flat[s_base + isz + Int32(2)].to(cutlass.Float32)
                svu3 = rot_scales_flat[s_base + isz + Int32(3)].to(cutlass.Float32)
                ig2_0 = gh0 * svg0
                ig2_1 = gh1 * svg1
                ig2_2 = gh2 * svg2
                ig2_3 = gh3 * svg3
                iu2_0 = uh0 * svu0
                iu2_1 = uh1 * svu1
                iu2_2 = uh2 * svu2
                iu2_3 = uh3 * svu3
                # silu(gate) * up, then pre-scale suh_down
                d2 = isz + isz
                sd0 = rot_scales_flat[s_base + d2 + Int32(0)].to(cutlass.Float32)
                sd1 = rot_scales_flat[s_base + d2 + Int32(1)].to(cutlass.Float32)
                sd2 = rot_scales_flat[s_base + d2 + Int32(2)].to(cutlass.Float32)
                sd3 = rot_scales_flat[s_base + d2 + Int32(3)].to(cutlass.Float32)
                a0 = self._silu_f32(ig2_0) * iu2_0 * sd0
                a1 = self._silu_f32(ig2_1) * iu2_1 * sd1
                a2 = self._silu_f32(ig2_2) * iu2_2 * sd2
                a3 = self._silu_f32(ig2_3) * iu2_3 * sd3
                # blockwise-128 Hadamard on the activation (post-silu, input side)
                o0, o1, o2, o3 = self._had128_quad(a0, a1, a2, a3, lane)
                out_base = row * isz + col0
                activated_bf16_flat[out_base + Int32(0)] = self._cast_elem(o0)
                activated_bf16_flat[out_base + Int32(1)] = self._cast_elem(o1)
                activated_bf16_flat[out_base + Int32(2)] = self._cast_elem(o2)
                activated_bf16_flat[out_base + Int32(3)] = self._cast_elem(o3)
                unit += gw_stride
            return
        if cutlass.const_expr(
            self.activation_is_gated
            and not self.is_fp16
            and self.intermediate_size % 8 == 0
            and self.fc1_cols % 8 == 0
        ):
            self._run_activation_gated_x8(
                fc1_bf16_flat, activated_bf16_flat, tid, cta, grid_x, active_m
            )
            return
        idx = cta * Int32(self.cta_threads) + tid
        stride = grid_x * Int32(self.cta_threads)
        total = active_m * Int32(self.top_k * self.intermediate_size)
        while idx < total:
            if cutlass.const_expr(self.activation_is_gated):
                row = idx // Int32(self.intermediate_size)
                col = idx - row * Int32(self.intermediate_size)
                base = row * Int32(self.fc1_cols)
                gate = fc1_bf16_flat[base + col].to(cutlass.Float32)
                up = fc1_bf16_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                gate, up = self._clamp_swiglu_inputs(gate, up)
                sigmoid_arg = gate
                up_term = up
                if cutlass.const_expr(self.activation_is_swigluoai):
                    sigmoid_arg = cutlass.Float32(self.swiglu_alpha) * gate
                    up_term = up + cutlass.Float32(self.swiglu_beta)
                if cutlass.const_expr(self.fast_math):
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=True)
                else:
                    exp_neg_gate = cute.math.exp(-sigmoid_arg, fastmath=False)
                sigmoid = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + exp_neg_gate)
                silu = gate * sigmoid
                if cutlass.const_expr(
                    self.has_swiglu_limit and self.swiglu_clamp_after_silu
                ):
                    if silu > cutlass.Float32(self.swiglu_limit):
                        silu = cutlass.Float32(self.swiglu_limit)
                if cutlass.const_expr(self.activation_is_situ):
                    beta = cutlass.Float32(SITU_DEFAULT_BETA)
                    linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
                    situ_gate = (
                        beta
                        * cute.math.tanh(gate / beta, fastmath=self.fast_math)
                        * sigmoid
                    )
                    situ_up = linear_beta * cute.math.tanh(
                        up / linear_beta,
                        fastmath=self.fast_math,
                    )
                    activated_bf16_flat[idx] = self._cast_elem(situ_gate * situ_up)
                elif cutlass.const_expr(self.activation_is_swigluoai):
                    activated_bf16_flat[idx] = self._cast_elem(silu * up_term)
                else:
                    activated_bf16_flat[idx] = self._cast_elem(
                        self._cast_elem(silu) * self._cast_elem(up_term)
                    )
            else:
                x = fc1_bf16_flat[idx].to(cutlass.Float32)
                if x < cutlass.Float32(0.0):
                    x = cutlass.Float32(0.0)
                activated_bf16_flat[idx] = self._cast_elem(x * x)
            idx += stride
