"""First-class weight-preparation helpers for the unified MoE API.

Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Backends consume different native weight layouts (quantization + swizzle +
MMA reorder).  These helpers turn canonical or checkpoint expert weights into
backend-native ``MoEWeightPack`` views, so that preparation lives in the
implementation surface rather than being copy-pasted into tests and benchmarks
(design doc CR2/CR7; reviewer comments C6, C7, C31, C32).

The canonical TRTLLM FP4 and BF16 helpers are exposed as
``TrtllmFp4Config.prepare_weights(...)`` /
``CuteDslConfig.prepare_weights(...)`` /
``TrtllmBf16Config.prepare_weights(...)`` / ... static helpers (see ``api.py``).
CUTLASS fused-MoE paths (BF16, NVFP4, per-tensor FP8, DeepSeek block FP8,
MXFP8, W4A16, W4A8, Humming) have matching ``Cutlass*Config.prepare_weights``
helpers in this module.
"""

from __future__ import annotations

import functools
import struct
import warnings
from typing import Dict, Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..tllm_enums import ActivationType
from ..trace.templates.moe import (
    sm90_mixed_gemm_humming_weight_preprocess_trace_dispatch,
    sm90_mixed_gemm_scale_interleave_trace,
    sm90_mixed_gemm_weight_interleave_trace,
)
from ..utils import get_compute_capability, round_up

# Module-level permute-index caches. Permute indices depend on weight geometry
# and layout parameters, so matching keys are safe to reuse across calls.
_TRTLLM_PERMUTE_CACHE: dict = {}
_TRTLLM_FP8_PERMUTE_CACHE: dict = {}
_TRTLLM_FP8_PER_TENSOR_PERMUTE_CACHE: dict = {}
_TRTLLM_MXINT4_PERMUTE_CACHE: dict = {}


def _normalize_activation(activation=None):
    from .api import ActivationConfig, SwiGLU

    activation = SwiGLU() if activation is None else activation
    if not isinstance(activation, ActivationConfig):
        raise TypeError(
            f"activation must be an ActivationConfig value, got {type(activation).__name__}."
        )
    return activation


def _gemm1_rows(intermediate_size: int, activation=None) -> int:
    activation = _normalize_activation(activation)
    return intermediate_size * (2 if activation.is_gated else 1)


def _activation_param_view(
    activation, num_expert_rows: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Expand typed scalar semantics into the existing per-expert launcher ABI."""
    from .api import SiTU, SwiGLU

    activation = _normalize_activation(activation)
    values: Dict[str, torch.Tensor] = {}
    params: Tuple[Tuple[str, Optional[float]], ...]
    if isinstance(activation, SwiGLU) and activation != SwiGLU():
        params = (
            ("gemm1_alpha", activation.alpha),
            ("gemm1_beta", activation.beta),
            ("gemm1_clamp_limit", activation.limit),
        )
    elif isinstance(activation, SiTU):
        params = (
            ("gemm1_alpha", activation.gate_scale),
            ("gemm1_beta", activation.linear_scale),
            ("gemm1_clamp_limit", activation.clamp_limit),
        )
    else:
        params = ()
    for name, value in params:
        if value is not None:
            values[name] = torch.full(
                (num_expert_rows,), value, dtype=torch.float32, device=device
            )
    return values


# The E8M0 range clamp and residual-scale factorization are adapted from
# Humming's HummingLayer.may_process_fused_e8m0_scale:
# https://github.com/inclusionAI/humming/blob/f6241bba8d507c19ca9ce4e5958a5d0641fc8eb4/humming/layer.py#L322-L362
def _preprocess_humming_e8m0_weight_scale(
    raw_scale: torch.Tensor,
    max_range: int = 11,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp Humming fused-E8M0 scales into offset, residual, and FP4 delta.

    The Humming layer computes this range clamp independently per expert.  The
    offset tensor is consumed by the pre-MMA FP4->E4M3 conversion; residual is
    one FP32 scale per expert; delta rewrites clamped FP4 payload values.
    """
    if raw_scale.dim() != 3:
        raise ValueError(
            "raw_scale must be 3D (num_experts, rows, K/32); "
            f"got shape {tuple(raw_scale.shape)}"
        )
    if raw_scale.dtype != torch.uint8:
        raise ValueError(f"raw_scale must be uint8 E8M0 bytes; got {raw_scale.dtype}")
    if not raw_scale.is_cuda:
        raise ValueError("raw_scale must live on CUDA")
    # The fused conversion adds max_range + 1 to FP4 exponent code 3;
    # E4M3 exponent code 15 therefore limits max_range to 11.
    if max_range < 0 or max_range > 11:
        raise ValueError(f"max_range must be in [0, 11]; got {max_range}")

    num_experts = raw_scale.shape[0]
    scale_view = raw_scale.contiguous().view(num_experts, -1)
    scale_max = scale_view.max(dim=1, keepdim=True).values
    scale_min = scale_view.min(dim=1, keepdim=True).values
    scale_range = scale_max - scale_min
    max_range_tensor = torch.tensor(
        max_range, dtype=torch.uint8, device=raw_scale.device
    )
    scale_range = torch.minimum(scale_range, max_range_tensor)
    scale_min_new = scale_max - scale_range

    clamped_scale = scale_view.maximum(scale_min_new)
    delta_scale_offsets = (clamped_scale - scale_view).to(torch.uint8)
    offset = torch.bitwise_and(clamped_scale - scale_min_new + 1, 0x0F).to(torch.uint8)
    residual = torch.exp2(scale_min_new.squeeze(1).to(torch.float32) - 127.0) * 0.5
    return (
        offset.view_as(raw_scale).contiguous(),
        residual.contiguous(),
        delta_scale_offsets.view_as(raw_scale).contiguous(),
    )


# The delta-scale FP4 payload rewrite semantics are adapted from Humming's
# process_mxfp4_w4a8 implementation:
# https://github.com/inclusionAI/humming/blob/f6241bba8d507c19ca9ce4e5958a5d0641fc8eb4/humming/include/humming/kernel/process_mxfp4.cuh#L6-L69
@functools.cache
def _humming_mxfp4_w4a8_rewrite_lut_cpu() -> torch.Tensor:
    def float_from_bits(bits: int) -> float:
        return struct.unpack("f", struct.pack("I", bits & 0xFFFFFFFF))[0]

    def bits_from_float(value: float) -> int:
        return struct.unpack("I", struct.pack("f", value))[0]

    def dequant_fp4_val(code: int) -> float:
        sign = (code & 0x8) << 28
        other = (code & 0x7) << 22
        return float_from_bits(sign | other)

    def quant_to_fp4_val(value: float) -> int:
        value_bits = bits_from_float(value)
        mask = 0x81C00000
        rz_bits = value_bits & mask
        ru_bits = (value_bits + 0x00200000) & mask
        rz_value = float_from_bits(rz_bits)
        ru_value = float_from_bits(ru_bits)
        rounded_bits = (
            ru_bits if abs(value - rz_value) >= abs(value - ru_value) else rz_bits
        )
        return ((rounded_bits & 0x80000000) >> 28) | ((rounded_bits & 0x01C00000) >> 22)

    lut = torch.empty((256, 16), dtype=torch.uint8)
    for delta in range(256):
        scale_factor = float_from_bits(0x3F800000 - (delta << 23))
        for code in range(16):
            normalized_code = 0 if code == 8 else code
            if delta:
                normalized_code = quant_to_fp4_val(
                    dequant_fp4_val(normalized_code) * scale_factor
                )
            lut[delta, code] = normalized_code
    return lut


def _process_humming_mxfp4_w4a8_payload(
    weight: torch.Tensor,
    delta_scale_offsets: torch.Tensor,
) -> torch.Tensor:
    if weight.dim() != 3:
        raise ValueError(
            "weight must be 3D (num_experts, rows, K/2); "
            f"got shape {tuple(weight.shape)}"
        )
    if weight.dtype != torch.uint8:
        raise ValueError(f"weight must be packed uint8 FP4 payload; got {weight.dtype}")
    if not weight.is_cuda:
        raise ValueError("weight must live on CUDA")
    if delta_scale_offsets.shape[0] != weight.shape[0]:
        raise ValueError(
            "delta_scale_offsets and weight must have the same num_experts; "
            f"got {delta_scale_offsets.shape[0]} and {weight.shape[0]}"
        )
    expected_delta_shape = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2] * 2 // 32,
    )
    if tuple(delta_scale_offsets.shape) != expected_delta_shape:
        raise ValueError(
            "delta_scale_offsets must have shape "
            f"{expected_delta_shape}; got {tuple(delta_scale_offsets.shape)}"
        )
    if delta_scale_offsets.dtype != torch.uint8:
        raise ValueError(
            f"delta_scale_offsets must be uint8; got {delta_scale_offsets.dtype}"
        )

    lut = _humming_mxfp4_w4a8_rewrite_lut_cpu().to(weight.device)
    lo = weight & 0x0F
    hi = (weight >> 4) & 0x0F
    fp4_codes = torch.stack([lo, hi], dim=-1).reshape(*weight.shape[:-1], -1)
    delta = delta_scale_offsets.repeat_interleave(32, dim=-1).to(torch.long)
    rewritten = lut[delta, fp4_codes.to(torch.long)]
    processed = rewritten[..., 0::2] | (rewritten[..., 1::2] << 4)
    return processed.contiguous()


@flashinfer_api(trace=sm90_mixed_gemm_humming_weight_preprocess_trace_dispatch)
def preprocess_moe_weights_for_sm90_mixed_gemm_humming(
    weight: torch.Tensor,
    raw_scale: torch.Tensor,
    max_range: int = 11,
    *,
    interleave: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare MXFP4 weights for the SM90 Humming-style FP8 activation path.

    Parameters
    ----------
    weight : torch.Tensor
        ``[num_experts, rows, K // 2]`` CUDA uint8 tensor containing packed
        MXFP4 payload values.
    raw_scale : torch.Tensor
        ``[num_experts, rows, K // 32]`` CUDA uint8 tensor containing original
        E8M0 MXFP4 weight scales.
    max_range : int
        Maximum per-expert E8M0 exponent range kept in the pre-MMA FP4->E4M3
        offset.  Humming uses 11 for FP8 activation.
    interleave : bool
        If true, return tensors ready for ``cutlass_fused_moe``.  If false,
        return the logical processed weight and logical offset scale; this is
        useful for validation against a dequantized or Humming reference.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(weight_out, scale_out, residual)``.  With ``interleave=True``,
        ``weight_out`` is the SM90 mixed-input weight layout and ``scale_out``
        is the folded scale layout.  With ``interleave=False``, they are the
        logical processed packed weight and logical offset scale.  ``residual``
        is one FP32 value per local expert.  For ``cutlass_fused_moe``, multiply
        it by Humming's fixed ``2^6`` compensation and pass the resulting
        ``[num_local_experts]`` tensor in quant-scale slot 1 (FC1) or 4 (FC2).
        The runtime maps each routed row to its local expert and folds the
        residual into that row's dynamic activation dequantization scale.

    Notes
    -----
    The E8M0 range clamp, residual-scale factorization, and FP4 payload-rewrite
    scheme are adapted from `Humming <https://github.com/inclusionAI/humming>`_.
    """
    if weight.dim() != 3:
        raise ValueError(
            "weight must be 3D (num_experts, rows, K/2); "
            f"got shape {tuple(weight.shape)}"
        )
    k = weight.shape[2] * 2
    if k % 32 != 0:
        raise ValueError(f"weight K dimension must be divisible by 32; got K={k}")
    expected_scale_shape = (
        weight.shape[0],
        weight.shape[1],
        k // 32,
    )
    if tuple(raw_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"raw_scale must have shape {expected_scale_shape}; "
            f"got {tuple(raw_scale.shape)}"
        )
    if raw_scale.device != weight.device:
        raise ValueError(
            "raw_scale and weight must be on the same device; "
            f"got {raw_scale.device} and {weight.device}"
        )

    offset, residual, delta_scale_offsets = _preprocess_humming_e8m0_weight_scale(
        raw_scale, max_range
    )
    processed_weight = _process_humming_mxfp4_w4a8_payload(
        weight.contiguous(), delta_scale_offsets
    )
    if not interleave:
        return processed_weight, offset, residual

    return (
        interleave_moe_weights_for_sm90_mixed_gemm(processed_weight, "fp4_fp8"),
        interleave_moe_scales_for_sm90_mixed_gemm(offset),
        residual,
    )


@flashinfer_api(trace=sm90_mixed_gemm_scale_interleave_trace)
def interleave_moe_scales_for_sm90_mixed_gemm(
    scales: torch.Tensor,
    group_size: int = 32,
) -> torch.Tensor:
    """Fold weight scales for the SM90 mixed-input MoE GEMM.

    Parameters
    ----------
    scales : torch.Tensor
        ``[num_experts, rows, K // group_size]`` tensor of scalar weight scales.
        MXFP4 uses uint8 E8M0 scales with ``group_size=32``; W4A8 uses bf16
        bit-pattern scales with ``group_size=128``.
    group_size : int
        Weight quantization group size.

    Returns
    -------
    torch.Tensor
        Contiguous tensor with shape
        ``[num_experts, rows // 64, K // 128, folded_m, physical_cols]``.
        ``physical_cols`` is the number of scale elements in 16B and
        ``folded_m`` is derived so each 64x128 logical scale block is stored as
        a 16B-contiguous folded block.
    """
    if scales.dim() != 3:
        raise ValueError(
            f"scales must be 3D (num_experts, rows, K/group_size); got {tuple(scales.shape)}"
        )

    if group_size <= 0 or 128 % group_size != 0:
        raise ValueError(f"group_size={group_size} must be positive and divide 128")
    scale_groups_per_k128 = 128 // group_size
    element_bits = scales.element_size() * 8
    physical_cols = 128 // element_bits
    if physical_cols < 1 or 128 % element_bits != 0:
        raise ValueError(
            f"scale dtype {scales.dtype} has unsupported element size {element_bits} bits"
        )
    if physical_cols % scale_groups_per_k128 != 0:
        raise ValueError(
            f"scale dtype {scales.dtype} and group_size={group_size} do not form "
            "an integer folded M slice"
        )
    m_slices_per_m64 = physical_cols // scale_groups_per_k128
    if 64 % m_slices_per_m64 != 0:
        raise ValueError(
            f"folded M slices {m_slices_per_m64} must divide the logical M64 block"
        )
    folded_m = 64 // m_slices_per_m64

    e, rows, kgs = scales.shape
    if rows % 64 != 0:
        raise ValueError(f"scale rows={rows} must be divisible by 64")
    if kgs % scale_groups_per_k128 != 0:
        raise ValueError(
            f"K/group_size={kgs} must be divisible by scale groups per K128 block "
            f"{scale_groups_per_k128}"
        )
    k128_blocks = kgs // scale_groups_per_k128
    return (
        scales.reshape(
            e,
            rows // 64,
            m_slices_per_m64,
            folded_m,
            k128_blocks,
            scale_groups_per_k128,
        )
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .reshape(e, rows // 64, k128_blocks, folded_m, physical_cols)
    )


@flashinfer_api(trace=sm90_mixed_gemm_weight_interleave_trace)
def interleave_moe_weights_for_sm90_mixed_gemm(
    weight: torch.Tensor,
    quant_type: str = "fp4",
) -> torch.Tensor:
    """Interleave 4-bit packed MoE weights for the SM90 mixed-input GEMM.

    The SM90 mixed-dtype MoE GEMM (used by ``cutlass_fused_moe`` with
    ``use_w4_group_scaling=True``) expects weights in a specific interleaved
    layout; without preprocessing, the LUT-based FP4->BF16 conversion reads
    bytes from the wrong positions and the output diverges from a dequantized
    reference for any K > 128. TensorRT-LLM's W4A16 MoE runs the equivalent
    preprocessing at weight-load time (see
    ``interleave_4bit_weights_for_Hopper_mixed_gemm`` in TRT-LLM PR #12451).

    Parameters
    ----------
    weight : torch.Tensor
        ``[num_experts, n, k // 2]`` uint8 CUDA tensor (4-bit values packed
        two-per-byte).
    quant_type : str
        ``"fp4"`` for MXFP4 (the W4A16 path), ``"fp4_fp8"`` for MXFP4 consumed
        by the FP8/Humming-style pre-MMA-scale path, or ``"int4"`` for INT4
        (the W4A8 path).

    Returns
    -------
    torch.Tensor
        A new uint8 tensor with the same shape as ``weight`` holding the
        interleaved layout. Feed this directly as ``fc1_expert_weights`` /
        ``fc2_expert_weights`` to :func:`cutlass_fused_moe`.
    """
    if weight.dim() != 3:
        raise ValueError(
            f"weight must be 3D (num_experts, n, k/2); got shape {tuple(weight.shape)}"
        )
    if weight.dtype != torch.uint8:
        raise ValueError(f"weight must be uint8 (packed 4-bit); got {weight.dtype}")
    if not weight.is_cuda:
        raise ValueError("weight must live on CUDA")

    qtype_map = {"fp4": 1, "fp4_fp8": 2, "int4": 0}
    if quant_type not in qtype_map:
        raise ValueError(
            f"quant_type must be one of {list(qtype_map)}; got {quant_type!r}"
        )

    weight = weight.contiguous()
    out = torch.empty_like(weight)

    from .core import get_cutlass_fused_moe_module

    major, minor = get_compute_capability(weight.device)
    device_arch = f"{major * 10 + minor}"
    module = get_cutlass_fused_moe_module(device_arch)
    module.interleave_moe_weights_for_sm90_mixed_gemm(
        weight, out, qtype_map[quant_type]
    )
    return out


def prepare_trtllm_fp4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    variant=None,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
    permute_cache: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Build a TRTLLM FP4 ``trtllm_fp4_routed`` weight view.

    ``NVFP4`` uses 16-element E4M3 scale blocks. ``MXFP4`` (W4A8) and
    ``W4A16`` use the same MXFP4 weights with 32-element UE8M0 scale blocks.
    All variants use per-expert gated-act reorder + MMA shuffle on the packed
    weights and ``block_scale_interleave`` on the block scales.

    Parameters
    ----------
    w1_bf16 : Tensor
        Expert weights for GEMM1. Gated activations use
        ``[num_local_experts, 2*intermediate_size, hidden_size]`` in ``[up, gate]``
        order; non-gated activations (ReLU2) use
        ``[num_local_experts, intermediate_size, hidden_size]``.
    w2_bf16 : Tensor
        Down-projection expert weights ``[num_local_experts, hidden_size, intermediate_size]``.
    num_local_experts, hidden_size, intermediate_size : int
        Expert geometry.
    device : torch.device, optional
        Target device; defaults to ``w1_bf16.device``.
    permute_cache : dict, optional
        Shape-keyed permute-index cache; defaults to a module-level cache.

    Returns
    -------
    dict
        Keys expected by ``TrtllmFp4RoutedRunner.pack_inputs``: ``gemm1_weights``,
        ``gemm1_weights_scale``, ``gemm2_weights``, ``gemm2_weights_scale``,
        ``output1_scale_scalar``, ``output1_scale_gate_scalar``,
        ``output2_scale_scalar``, and, for NVFP4 only, ``gemm1_alpha``.
    """
    from ..fp4_quantization import fp4_quantize
    from ..quantization.fp4_quantization import block_scale_interleave
    from .api import QuantVariant
    from .core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    if variant is None:
        variant = QuantVariant.NVFP4
    if variant not in (
        QuantVariant.NVFP4,
        QuantVariant.MXFP4,
        QuantVariant.W4A16,
    ):
        raise ValueError(
            "TRTLLM FP4 weight preparation requires QuantVariant.NVFP4, "
            f"QuantVariant.MXFP4, or QuantVariant.W4A16; got {variant!r}."
        )
    is_mxfp4 = variant in (QuantVariant.MXFP4, QuantVariant.W4A16)
    sf_vec_size = 32 if is_mxfp4 else 16
    required_alignment = 128 if is_mxfp4 else sf_vec_size
    if (
        hidden_size % required_alignment != 0
        or intermediate_size % required_alignment != 0
    ):
        raise ValueError(
            f"{variant.name} requires hidden_size and intermediate_size divisible "
            f"by {required_alignment}."
        )

    if device is None:
        device = w1_bf16.device
    # Honor the documented device target: move canonical weights there (no-op if
    # already resident). Otherwise CPU weights + device="cuda" hit mixed-device
    # ops inside quantization.
    w1_bf16 = w1_bf16.to(device)
    w2_bf16 = w2_bf16.to(device)
    if permute_cache is None:
        permute_cache = _TRTLLM_PERMUTE_CACHE

    epilogue_tile_m = 128  # TRTLLM kernel-internal constant

    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    # The alignment check above bounds intermediate_size, but the scale
    # permutation tiles GEMM1 by epilogue_tile_m rows. A gated activation
    # doubles the row count, so intermediate_size=64 is fine for SwiGLU (128
    # rows) and short for ReLU2 (64). Without this the shortfall surfaces as a
    # bare AssertionError from inside the permutation.
    if gemm1_rows % epilogue_tile_m != 0:
        raise ValueError(
            f"{variant.name} requires GEMM1 rows divisible by {epilogue_tile_m}; "
            f"{type(activation).__name__} gives {gemm1_rows} rows for "
            f"intermediate_size={intermediate_size}."
        )
    w1_flat = w1_bf16.view(num_local_experts * gemm1_rows, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat,
        global_scale=w1_gs,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=is_mxfp4,
        is_sf_swizzled_layout=False,
    )
    g1_w = w1_q_flat.view(num_local_experts, gemm1_rows, hidden_size // 2).view(
        torch.uint8
    )
    g1_s = w1_sf_flat.view(torch.float8_e4m3fn).reshape(
        num_local_experts, gemm1_rows, hidden_size // sf_vec_size
    )

    w2_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w2_flat = w2_bf16.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat,
        global_scale=w2_gs,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=is_mxfp4,
        is_sf_swizzled_layout=False,
    )
    g2_w = w2_q_flat.view(num_local_experts, hidden_size, intermediate_size // 2).view(
        torch.uint8
    )
    g2_s = w2_sf_flat.view(torch.float8_e4m3fn).reshape(
        num_local_experts, hidden_size, intermediate_size // sf_vec_size
    )

    g1_w_sh, g1_s_sh, g2_w_sh, g2_s_sh = [], [], [], []
    for i in range(num_local_experts):
        p = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            g1_w[i],
            epilogue_tile_m,
            is_gated_act_gemm=activation.is_gated,
        )
        g1_w_sh.append(g1_w[i][p.to(device)].contiguous())

        p_sf = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            g1_s[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=activation.is_gated,
        )
        g1_s_sh.append(
            block_scale_interleave(
                g1_s[i].view(torch.uint8)[p_sf.to(device)].contiguous()
            )
        )

        p = get_w2_permute_indices_with_cache(
            permute_cache,
            g2_w[i],
            epilogue_tile_m,
            is_gated_act_gemm=activation.is_gated,
        )
        g2_w_sh.append(g2_w[i][p.to(device)].contiguous())

        p_sf = get_w2_permute_indices_with_cache(
            permute_cache,
            g2_s[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=activation.is_gated,
        )
        g2_s_sh.append(
            block_scale_interleave(
                g2_s[i].view(torch.uint8)[p_sf.to(device)].contiguous()
            )
        )

    ones = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    gemm1_scale = torch.stack(g1_s_sh).reshape(
        num_local_experts, gemm1_rows, hidden_size // sf_vec_size
    )
    gemm2_scale = torch.stack(g2_s_sh).reshape(
        num_local_experts, hidden_size, intermediate_size // sf_vec_size
    )
    # The FP4 launcher requires float8 tensor metadata for both E4M3 and
    # UE8M0 scale bytes; the dtype pair/scale length selects the interpretation.
    gemm1_scale = gemm1_scale.view(torch.float8_e4m3fn)
    gemm2_scale = gemm2_scale.view(torch.float8_e4m3fn)
    result = {
        "gemm1_weights": torch.stack(g1_w_sh),
        "gemm1_weights_scale": gemm1_scale,
        "gemm2_weights": torch.stack(g2_w_sh),
        "gemm2_weights_scale": gemm2_scale,
        "output1_scale_scalar": ones,
        "output1_scale_gate_scalar": ones,
        "output2_scale_scalar": ones,
    }
    if not is_mxfp4 and activation.is_gated:
        # NVFP4 gated kernels consume a per-expert gate alpha; non-gated
        # activations do not, so ReLU2 must not receive this placeholder.
        result["gemm1_alpha"] = ones
    result.update(_activation_param_view(activation, num_local_experts, device))
    return result


def prepare_trtllm_fp4_activations(
    hidden_states_bf16: torch.Tensor,
    *,
    variant,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Prepare activations for a unified TRTLLM FP4 quantization variant."""
    from .api import QuantVariant

    if hidden_states_bf16.ndim != 2:
        raise ValueError(
            "hidden_states_bf16 must be 2D [num_tokens, hidden_size], got "
            f"{tuple(hidden_states_bf16.shape)}."
        )
    if hidden_states_bf16.dtype != torch.bfloat16:
        raise TypeError(
            f"hidden_states_bf16 must be torch.bfloat16, got {hidden_states_bf16.dtype}."
        )

    if variant is QuantVariant.W4A16:
        return hidden_states_bf16, None
    if variant is QuantVariant.MXFP4:
        from ..quantization.fp8_quantization import mxfp8_quantize

        if hidden_states_bf16.shape[1] % 32 != 0:
            raise ValueError("MXFP4 requires hidden_size divisible by 32.")
        q, sf = mxfp8_quantize(hidden_states_bf16, is_sf_swizzled_layout=False)
        return q, sf.view(torch.float8_e4m3fn).reshape(hidden_states_bf16.shape[0], -1)
    if variant is QuantVariant.NVFP4:
        from ..fp4_quantization import fp4_quantize

        if hidden_states_bf16.shape[1] % 16 != 0:
            raise ValueError("NVFP4 requires hidden_size divisible by 16.")
        # The unified NVFP4 weight view currently fixes its epilogue scalars at
        # one, so activation preparation must use the matching global scale.
        global_scale = torch.ones(1, device=hidden_states_bf16.device)
        q, sf = fp4_quantize(
            hidden_states_bf16,
            global_scale=global_scale,
            sf_vec_size=16,
            sf_use_ue8m0=False,
            is_sf_swizzled_layout=False,
        )
        return q, sf.view(torch.float8_e4m3fn).reshape(hidden_states_bf16.shape[0], -1)
    raise ValueError(
        "TRTLLM FP4 activation preparation requires QuantVariant.NVFP4, "
        f"QuantVariant.MXFP4, or QuantVariant.W4A16; got {variant!r}."
    )


def _deepseek_fp8_quantize_activations(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` BF16 per 1x128 block.

    TRTLLM's DeepSeek path consumes scales transposed as ``[K // 128, M]``.
    """
    block = 128
    m, k = x.shape
    blocks = x.float().reshape(m, k // block, block)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scales = (blocks.abs().amax(dim=-1, keepdim=True) / fp8_max).clamp(min=1e-12)
    quantized = (blocks / scales).clamp(-fp8_max, fp8_max)
    return (
        quantized.reshape(m, k).to(torch.float8_e4m3fn),
        scales.squeeze(-1).transpose(0, 1).contiguous(),
    )


def _deepseek_fp8_quantize_weights(
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[E, N, K]`` BF16 per 128x128 block."""
    block = 128
    e, n, k = weights.shape
    blocks = (
        weights.float()
        .reshape(e, n // block, block, k // block, block)
        .permute(0, 1, 3, 2, 4)
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scales = (blocks.abs().amax(dim=(-1, -2), keepdim=True) / fp8_max).clamp(min=1e-12)
    quantized = (blocks / scales).clamp(-fp8_max, fp8_max)
    quantized = quantized.permute(0, 1, 3, 2, 4).reshape(e, n, k)
    return quantized.to(torch.float8_e4m3fn), scales[..., 0, 0].contiguous()


def _validate_trtllm_fp8_block_inputs(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
) -> None:
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise ValueError(
            "prepare_trtllm_fp8_block_weights expects BF16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    expected_w1 = (
        num_local_experts,
        _gemm1_rows(intermediate_size, activation),
        hidden_size,
    )
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )


def prepare_trtllm_fp8_block_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    variant,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Prepare canonical BF16 expert weights for TRTLLM block-FP8 MoE.

    DeepSeek FP8 uses E4M3 payloads with FP32 128x128 block scales. MXFP8
    uses E4M3 payloads with linear UE8M0 scales over 32-element K blocks.
    Both native views remain in ``MajorK`` layout; the unified runner records
    the exact variant and passes the corresponding kernel enum. Shuffled
    MXFP8 preparation requires ``hidden_size`` and ``intermediate_size`` to be
    divisible by 128 because the returned scales use TRTLLM's unpadded 128x4
    physical layout.
    """
    from .api import QuantVariant

    if variant not in (QuantVariant.DeepSeekFp8, QuantVariant.MxFp8):
        raise ValueError(
            "variant must be QuantVariant.DeepSeekFp8 or QuantVariant.MxFp8, "
            f"got {variant!r}."
        )
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    _validate_trtllm_fp8_block_inputs(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    if device is None:
        device = w1_bf16.device
    w1_bf16 = w1_bf16.to(device).contiguous()
    w2_bf16 = w2_bf16.to(device).contiguous()

    if variant is QuantVariant.DeepSeekFp8:
        for name, dim in (
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("gemm1_rows", gemm1_rows),
        ):
            if dim % 128 != 0:
                raise ValueError(f"DeepSeek FP8 requires {name} divisible by 128.")
        w1_q, w1_sf = _deepseek_fp8_quantize_weights(w1_bf16)
        w2_q, w2_sf = _deepseek_fp8_quantize_weights(w2_bf16)
    else:
        if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
            raise ValueError(
                "MXFP8 shuffled MajorK preparation requires hidden_size and "
                "intermediate_size divisible by 128."
            )
        from ..quantization.fp4_quantization import block_scale_interleave
        from ..quantization.fp8_quantization import mxfp8_quantize
        from .core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        w1_q, w1_sf, w2_q, w2_sf = [], [], [], []
        for expert in range(num_local_experts):
            q, sf = mxfp8_quantize(w1_bf16[expert], is_sf_swizzled_layout=False)
            sf = sf.view(torch.uint8).reshape(gemm1_rows, hidden_size // 32)
            permute = _maybe_get_cached_w3_w1_permute_indices(
                _TRTLLM_FP8_PERMUTE_CACHE,
                q.view(torch.uint8),
                128,
                is_gated_act_gemm=activation.is_gated,
            )
            permute_sf = _maybe_get_cached_w3_w1_permute_indices(
                _TRTLLM_FP8_PERMUTE_CACHE,
                sf,
                128,
                num_elts_per_sf=32,
                is_gated_act_gemm=activation.is_gated,
            )
            w1_q.append(q.view(torch.uint8)[permute.to(device)].view(q.dtype))
            w1_sf.append(
                block_scale_interleave(
                    sf[permute_sf.to(device)].contiguous()
                ).reshape_as(sf)
            )

            q, sf = mxfp8_quantize(w2_bf16[expert], is_sf_swizzled_layout=False)
            sf = sf.view(torch.uint8).reshape(hidden_size, intermediate_size // 32)
            permute = get_w2_permute_indices_with_cache(
                _TRTLLM_FP8_PERMUTE_CACHE,
                q.view(torch.uint8),
                128,
                is_gated_act_gemm=activation.is_gated,
            )
            permute_sf = get_w2_permute_indices_with_cache(
                _TRTLLM_FP8_PERMUTE_CACHE,
                sf,
                128,
                num_elts_per_sf=32,
                is_gated_act_gemm=activation.is_gated,
            )
            w2_q.append(q.view(torch.uint8)[permute.to(device)].view(q.dtype))
            w2_sf.append(
                block_scale_interleave(
                    sf[permute_sf.to(device)].contiguous()
                ).reshape_as(sf)
            )
        w1_q, w1_sf = torch.stack(w1_q), torch.stack(w1_sf)
        w2_q, w2_sf = torch.stack(w2_q), torch.stack(w2_sf)

    result = {
        "gemm1_weights": w1_q,
        "gemm1_weights_scale": w1_sf,
        "gemm2_weights": w2_q,
        "gemm2_weights_scale": w2_sf,
    }
    result.update(_activation_param_view(activation, num_local_experts, device))
    return result


def prepare_trtllm_fp8_block_activations(
    hidden_states_bf16: torch.Tensor,
    *,
    variant,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, H]`` BF16 activations for TRTLLM block-FP8 MoE."""
    from .api import QuantVariant

    if hidden_states_bf16.dtype != torch.bfloat16 or hidden_states_bf16.dim() != 2:
        raise ValueError(
            "prepare_trtllm_fp8_block_activations expects a 2D BF16 tensor, "
            f"got shape={tuple(hidden_states_bf16.shape)}, "
            f"dtype={hidden_states_bf16.dtype}."
        )
    hidden_states_bf16 = hidden_states_bf16.contiguous()
    if variant is QuantVariant.DeepSeekFp8:
        if hidden_states_bf16.shape[1] % 128 != 0:
            raise ValueError("DeepSeek FP8 hidden_size must be divisible by 128.")
        return _deepseek_fp8_quantize_activations(hidden_states_bf16)
    if variant is QuantVariant.MxFp8:
        from ..quantization.fp8_quantization import mxfp8_quantize

        q, sf = mxfp8_quantize(hidden_states_bf16, is_sf_swizzled_layout=False)
        return q, sf.view(torch.uint8).reshape(hidden_states_bf16.shape[0], -1)
    raise ValueError(
        "variant must be QuantVariant.DeepSeekFp8 or QuantVariant.MxFp8, "
        f"got {variant!r}."
    )


def _fp8_per_tensor_scale(
    scale: Union[float, torch.Tensor], *, name: str, device: torch.device
) -> torch.Tensor:
    value = torch.as_tensor(scale, dtype=torch.float32, device=device)
    if value.numel() != 1 or not bool(torch.isfinite(value).all()) or value.item() <= 0:
        raise ValueError(
            f"{name} must be one finite positive FP32 value, got {scale!r}."
        )
    return value.reshape(())


def _quantize_fp8_per_expert(
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize each expert tensor with one E4M3 multiplier."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weights_f32 = weights.float()
    amax = weights_f32.abs().amax(dim=(-1, -2))
    # Weight preparation runs once at model load. Fail here instead of silently
    # sanitizing a corrupted checkpoint or surfacing NaNs much later in inference.
    # Reuse the required per-expert reduction to avoid another full tensor scan.
    if not bool(torch.isfinite(amax).all()):
        raise ValueError(
            "FP8 per-tensor weight preparation requires finite checkpoint weights."
        )
    scales = torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))
    quantized = (weights_f32 * scales[:, None, None]).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn), scales.to(torch.float32)


def prepare_trtllm_fp8_per_tensor_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    hidden_states_scale_global: Union[float, torch.Tensor],
    intermediate_scale_global: Union[float, torch.Tensor],
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the TRTLLM per-tensor-FP8 MajorK weight view.

    ``hidden_states_scale_global`` and ``intermediate_scale_global`` are the
    E4M3 quantization multipliers obtained during PTQ/QAT calibration. Weight
    multipliers are computed independently for each local expert.
    """
    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise ValueError(
            "prepare_trtllm_fp8_per_tensor_weights expects BF16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    expected_w1 = (num_local_experts, gemm1_rows, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )

    input_scale = _fp8_per_tensor_scale(
        hidden_states_scale_global,
        name="hidden_states_scale_global",
        device=device,
    )
    intermediate_scale = _fp8_per_tensor_scale(
        intermediate_scale_global,
        name="intermediate_scale_global",
        device=device,
    )
    w1_q, w1_scale = _quantize_fp8_per_expert(w1_bf16.to(device).contiguous())
    w2_q, w2_scale = _quantize_fp8_per_expert(w2_bf16.to(device).contiguous())

    from .core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    w1_shuffled, w2_shuffled = [], []
    for expert in range(num_local_experts):
        permute = _maybe_get_cached_w3_w1_permute_indices(
            _TRTLLM_FP8_PER_TENSOR_PERMUTE_CACHE,
            w1_q[expert].view(torch.uint8),
            128,
            is_gated_act_gemm=activation.is_gated,
        )
        w1_shuffled.append(
            w1_q[expert]
            .view(torch.uint8)[permute.to(device)]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        permute = get_w2_permute_indices_with_cache(
            _TRTLLM_FP8_PER_TENSOR_PERMUTE_CACHE,
            w2_q[expert].view(torch.uint8),
            128,
            is_gated_act_gemm=activation.is_gated,
        )
        w2_shuffled.append(
            w2_q[expert]
            .view(torch.uint8)[permute.to(device)]
            .contiguous()
            .view(torch.float8_e4m3fn)
        )

    output1_scale = (
        intermediate_scale / (w1_scale * input_scale)
        if activation.is_gated
        else torch.ones_like(w1_scale) * intermediate_scale
    )
    return {
        "gemm1_weights": torch.stack(w1_shuffled),
        "gemm2_weights": torch.stack(w2_shuffled),
        "output1_scales_scalar": output1_scale.contiguous(),
        "output1_scales_gate_scalar": (1.0 / (w1_scale * input_scale)).contiguous(),
        "output2_scales_scalar": (1.0 / (intermediate_scale * w2_scale)).contiguous(),
        # Calibration metadata is retained for callers preparing activations or
        # constructing an independent reference; the runner ignores these keys.
        "hidden_states_scale_global": input_scale,
        "intermediate_scale_global": intermediate_scale,
    }


def prepare_trtllm_fp8_per_tensor_activations(
    hidden_states_bf16: torch.Tensor,
    *,
    hidden_states_scale_global: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, None]:
    """Quantize ``[M, H]`` BF16 activations with one calibrated E4M3 scale."""
    if hidden_states_bf16.dtype != torch.bfloat16 or hidden_states_bf16.dim() != 2:
        raise ValueError(
            "prepare_trtllm_fp8_per_tensor_activations expects a 2D BF16 tensor, "
            f"got shape={tuple(hidden_states_bf16.shape)}, "
            f"dtype={hidden_states_bf16.dtype}."
        )
    scale = _fp8_per_tensor_scale(
        hidden_states_scale_global,
        name="hidden_states_scale_global",
        device=hidden_states_bf16.device,
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    quantized = (hidden_states_bf16.float() * scale).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn), None


def _mxint4_quantize(
    weights: torch.Tensor, sf_vec_size: int = 32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the last dimension to signed packed INT4 with BF16 block scales."""
    blocks = weights.reshape(-1, sf_vec_size)
    block_max = blocks.amax(dim=-1, keepdim=True).to(torch.float32)
    block_min = blocks.amin(dim=-1, keepdim=True).to(torch.float32)
    block_max = block_max * (8.0 / 7.0)
    amax = torch.where(block_max > -block_min, block_max, -block_min)
    scales = amax / 8.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = (
        (blocks * scales.reciprocal())
        .round()
        .clamp(-8, 7)
        .to(torch.int8)
        .reshape(-1, sf_vec_size // 2, 2)
    )
    nibbles = (quantized & 0x0F).to(torch.uint8)
    packed = nibbles[..., 0] | (nibbles[..., 1] << 4)
    return (
        packed.reshape(*weights.shape[:-1], weights.shape[-1] // 2),
        scales.to(torch.bfloat16),
    )


def prepare_trtllm_mxint4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
    permute_cache: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Build the TRTLLM MxInt4 ``trtllm_mxint4_routed`` weight view.

    Canonical BF16 expert weights are quantized in 32-element K blocks, then
    shuffled for fused SwiGLU / transposed-MMA output. Packed INT4 payloads use
    BlockMajorK while BF16 scale tensors use TRTLLM's block-scale interleave.
    """
    from ..quantization.fp4_quantization import block_scale_interleave
    from .core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise ValueError(
            "prepare_trtllm_mxint4_weights expects BF16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    expected_w1 = (num_local_experts, gemm1_rows, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )
    if hidden_size % 256 != 0 or intermediate_size % 256 != 0:
        raise ValueError(
            "TRTLLM MxInt4 requires hidden_size and intermediate_size divisible by 256."
        )

    w1 = w1_bf16.to(device).contiguous()
    w2 = w2_bf16.to(device).contiguous()
    w1_q, w1_sf = _mxint4_quantize(w1)
    w2_q, w2_sf = _mxint4_quantize(w2)
    w1_sf = w1_sf.reshape(num_local_experts, gemm1_rows, hidden_size // 32)
    w2_sf = w2_sf.reshape(num_local_experts, hidden_size, intermediate_size // 32)

    if permute_cache is None:
        permute_cache = _TRTLLM_MXINT4_PERMUTE_CACHE
    epilogue_tile_m = 128
    block_k = 128
    w1_views, w1_scale_views, w2_views, w2_scale_views = [], [], [], []
    for expert in range(num_local_experts):
        w1_permute = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            w1_q[expert],
            epilogue_tile_m,
            is_gated_act_gemm=activation.is_gated,
        )
        w1_scale_permute = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            w1_sf[expert],
            epilogue_tile_m,
            num_elts_per_sf=32,
            is_gated_act_gemm=activation.is_gated,
        )
        w2_permute = get_w2_permute_indices_with_cache(
            permute_cache,
            w2_q[expert],
            epilogue_tile_m,
            is_gated_act_gemm=activation.is_gated,
        )
        # Keep the established flat-test MxInt4 scale permutation contract;
        # preparation parity tests cover this asymmetric GEMM1/GEMM2 setting.
        w2_scale_permute = get_w2_permute_indices_with_cache(
            permute_cache,
            w2_sf[expert],
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=activation.is_gated,
        )

        w1_views.append(
            convert_to_block_layout(
                w1_q[expert][w1_permute.to(device)].contiguous(), block_k
            )
        )
        w1_scale_views.append(
            block_scale_interleave(
                w1_sf[expert][w1_scale_permute.to(device)].contiguous()
            )
        )
        w2_views.append(
            convert_to_block_layout(
                w2_q[expert][w2_permute.to(device)].contiguous(), block_k
            )
        )
        w2_scale_views.append(
            block_scale_interleave(
                w2_sf[expert][w2_scale_permute.to(device)].contiguous()
            )
        )

    result = {
        "gemm1_weights": torch.stack(w1_views),
        "gemm1_weights_scale": torch.stack(w1_scale_views).view(torch.bfloat16),
        "gemm2_weights": torch.stack(w2_views),
        "gemm2_weights_scale": torch.stack(w2_scale_views).view(torch.bfloat16),
    }
    result.update(_activation_param_view(activation, num_local_experts, device))
    return result


def prepare_trtllm_bf16_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
    permute_cache: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Build the TRTLLM BF16 ``trtllm_bf16_routed`` weight view.

    Layout is ``BlockMajorK`` — the only layout the bf16 trtllm-gen entry points
    accept: per-expert fused-gated-activation row reorder chained with the
    ``epilogue_tile_m=128`` MMA shuffle for gemm1 (the w3_w1 permute), plain
    shuffle for gemm2, then ``block_k=128`` K-blocking on the uint8 view.  No
    quantization — weights stay bf16.  The gated-act reorder on gemm1 is
    required for SwiGLU: the kernel pairs gate/linear rows interleaved, and a
    shuffle-only layout mis-pairs them (systematically wrong output that still
    passes kernel-vs-same-kernel parity checks).

    Parameters
    ----------
    w1_bf16 : Tensor
        Expert weights for GEMM1. Gated activations use
        ``[num_local_experts, 2*intermediate_size, hidden_size]`` in ``[up, gate]``
        order; non-gated activations (ReLU2) use
        ``[num_local_experts, intermediate_size, hidden_size]``.
    w2_bf16 : Tensor
        Down-projection expert weights ``[num_local_experts, hidden_size, intermediate_size]``.
    num_local_experts, hidden_size, intermediate_size : int
        Expert geometry.
    device : torch.device, optional
        Target device; defaults to ``w1_bf16.device``.
    permute_cache : dict, optional
        Shape-keyed permute-index cache; defaults to a module-level cache.

    Returns
    -------
    dict
        Keys expected by ``TrtllmBf16RoutedRunner.pack_inputs``:
        ``gemm1_weights``, ``gemm2_weights`` (both bf16, BlockMajorK).
    """
    from .core import (
        _maybe_get_cached_w3_w1_permute_indices,
        convert_to_block_layout,
        get_w2_permute_indices_with_cache,
    )

    if device is None:
        device = w1_bf16.device
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise ValueError(
            f"prepare_trtllm_bf16_weights expects bf16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype} (the uint8 byte-view below "
            f"would silently reinterpret other dtypes)"
        )
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    expect_w1 = (num_local_experts, gemm1_rows, hidden_size)
    expect_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expect_w1 or tuple(w2_bf16.shape) != expect_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expect_w1}/{expect_w2}"
        )
    # Honor the documented device target (no-op if already resident); contiguity
    # is required for the uint8 view below.
    w1_bf16 = w1_bf16.to(device).contiguous()
    w2_bf16 = w2_bf16.to(device).contiguous()
    if permute_cache is None:
        permute_cache = _TRTLLM_PERMUTE_CACHE

    epilogue_tile_m = 128  # TRTLLM kernel-internal constant
    block_k = 128

    w1_views, w2_views = [], []
    for i in range(num_local_experts):
        w1_u8 = w1_bf16[i].view(torch.uint8)
        p1 = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            w1_u8,
            epilogue_tile_m,
            is_gated_act_gemm=activation.is_gated,
        )
        w1_views.append(
            convert_to_block_layout(w1_u8[p1.to(device)].contiguous(), block_k)
        )

        w2_u8 = w2_bf16[i].view(torch.uint8)
        p2 = get_w2_permute_indices_with_cache(
            permute_cache,
            w2_u8,
            epilogue_tile_m,
            is_gated_act_gemm=activation.is_gated,
        )
        w2_views.append(
            convert_to_block_layout(w2_u8[p2.to(device)].contiguous(), block_k)
        )

    result = {
        "gemm1_weights": torch.stack(w1_views).view(torch.bfloat16),
        "gemm2_weights": torch.stack(w2_views).view(torch.bfloat16),
    }
    result.update(_activation_param_view(activation, num_local_experts, device))
    return result


def prepare_cutlass_bf16_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the canonical BF16 view consumed by ``CutlassBf16Runner``.

    ``w1_bf16`` is ``[E, 2*I, H]`` in semantic ``[up, gate]`` order for gated
    activations, or ``[E, I, H]`` for non-gated ones (ReLU2); the row
    count follows ``activation.is_gated``. ``w2_bf16`` is ``[E, H, I]``.
    CUTLASS BF16 paths consume these dense tensors directly; preparation
    validates the source contract and materializes contiguous tensors on the
    requested device.
    """
    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise TypeError(
            "prepare_cutlass_bf16_weights expects BF16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    activation = _normalize_activation(activation)
    expected_w1 = (
        num_local_experts,
        _gemm1_rows(intermediate_size, activation),
        hidden_size,
    )
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )
    return {
        "fc1_expert_weights": w1_bf16.to(device).contiguous(),
        "fc2_expert_weights": w2_bf16.to(device).contiguous(),
    }


def _swizzle_cutile_nvfp4_scales(scale: torch.Tensor) -> torch.Tensor:
    """Convert ``[E, N, K/16]`` scales to the layout used by scaled MMA."""
    num_experts, n, k_groups = scale.shape
    if n % 64 != 0 or k_groups % 4 != 0:
        raise ValueError("cuTile W4A4 scales require N and K divisible by 64.")
    padded_n = (n + 127) // 128 * 128
    if padded_n != n:
        padded = scale.new_zeros((num_experts, padded_n, k_groups))
        padded[:, :n] = scale
        scale = padded
    reshaped = scale.reshape(num_experts * padded_n, k_groups)
    reshaped = reshaped.reshape(num_experts * padded_n // 128, 4, 32, k_groups // 4, 4)
    return (
        reshaped.permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(num_experts, padded_n // 128, k_groups // 4, 32, 16)
    )


def prepare_cutile_nvfp4_weights(
    w1_fp4: torch.Tensor,
    w1_block_scale: torch.Tensor,
    w1_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_block_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation_type: ActivationType = ActivationType.Swiglu,
    source_format: str = "modelopt",
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build checkpoint-native NVFP4 views for the cuTile MoE runners.

    Packed values use two E2M1 elements per byte along K. Block scales are
    E4M3 with a 16-element K group, and global scales are per expert or, for a
    gated GEMM1, optionally per ``[up, gate]`` shard. W4A4 weights retain their
    logical dimensions while their scaled-MMA layout pads outer scale rows to
    a multiple of 128.
    """
    activation_type = ActivationType(activation_type)
    if activation_type not in (ActivationType.Swiglu, ActivationType.Relu2):
        raise ValueError(
            f"unsupported cuTile NVFP4 activation {activation_type!r}; expected "
            "Swiglu or Relu2."
        )
    if hidden_size % 64 != 0 or intermediate_size % 64 != 0:
        raise ValueError(
            "cuTile W4A4 requires hidden_size and intermediate_size divisible by 64."
        )
    if device is None:
        device = w1_fp4.device
    device = torch.device(device)
    tensors = (
        w1_fp4,
        w1_block_scale,
        w1_global_scale,
        w2_fp4,
        w2_block_scale,
        w2_global_scale,
    )
    if any(t.device != tensors[0].device for t in tensors):
        raise ValueError("cuTile NVFP4 checkpoint tensors must share one device.")
    if w1_fp4.dtype != torch.uint8 or w2_fp4.dtype != torch.uint8:
        raise TypeError("cuTile NVFP4 packed weights must use torch.uint8.")
    if (
        w1_block_scale.dtype != torch.float8_e4m3fn
        or w2_block_scale.dtype != torch.float8_e4m3fn
    ):
        raise TypeError("cuTile NVFP4 block scales must use torch.float8_e4m3fn.")
    if w1_global_scale.dtype != torch.float32 or w2_global_scale.dtype != torch.float32:
        raise TypeError("cuTile NVFP4 global scales must use torch.float32.")

    w1_rows = intermediate_size * (2 if activation_type.is_gated else 1)
    expected_w1 = (num_local_experts, w1_rows, hidden_size // 2)
    expected_w1_scale = (num_local_experts, w1_rows, hidden_size // 16)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size // 2)
    expected_w2_scale = (num_local_experts, hidden_size, intermediate_size // 16)
    if tuple(w1_fp4.shape) != expected_w1 or tuple(w2_fp4.shape) != expected_w2:
        raise ValueError(
            f"cuTile NVFP4 weight shapes {tuple(w1_fp4.shape)}/{tuple(w2_fp4.shape)} "
            f"!= expected {expected_w1}/{expected_w2}."
        )
    if (
        tuple(w1_block_scale.shape) != expected_w1_scale
        or tuple(w2_block_scale.shape) != expected_w2_scale
    ):
        raise ValueError(
            "cuTile NVFP4 block-scale shapes "
            f"{tuple(w1_block_scale.shape)}/{tuple(w2_block_scale.shape)} != "
            f"expected {expected_w1_scale}/{expected_w2_scale}."
        )
    allowed_w1_global_shapes: set[tuple[int, ...]] = {(num_local_experts,)}
    if activation_type.is_gated:
        allowed_w1_global_shapes.add((num_local_experts, 2))
    if tuple(w1_global_scale.shape) not in allowed_w1_global_shapes:
        raise ValueError(
            "cuTile NVFP4 GEMM1 global scale must be [E] or [E, 2] for a gated activation."
        )
    if tuple(w2_global_scale.shape) != (num_local_experts,):
        raise ValueError("cuTile NVFP4 GEMM2 global scale must have shape [E].")

    source = source_format.lower().replace("-", "_")
    if source not in ("modelopt", "modelopt_nvfp4", "compressed_tensors"):
        raise ValueError(
            "source_format must be 'modelopt' or 'compressed_tensors', "
            f"got {source_format!r}."
        )
    if source == "compressed_tensors":
        w1_global_scale = 1.0 / w1_global_scale
        w2_global_scale = 1.0 / w2_global_scale

    w1_fp4 = w1_fp4.to(device)
    w1_block_scale = w1_block_scale.to(device)
    w1_global_scale = w1_global_scale.to(device)
    w2_fp4 = w2_fp4.to(device)
    w2_block_scale = w2_block_scale.to(device)
    w2_global_scale = w2_global_scale.to(device)
    if activation_type.is_gated:
        up, gate = w1_fp4.chunk(2, dim=1)
        w1_fp4 = torch.cat((gate, up), dim=1)
        up_scale, gate_scale = w1_block_scale.chunk(2, dim=1)
        w1_block_scale = torch.cat((gate_scale, up_scale), dim=1)
        if w1_global_scale.ndim == 2:
            w1_global_scale = w1_global_scale[:, [1, 0]]

    result = {
        "w1": w1_fp4.contiguous(),
        "w1_scale": w1_block_scale.contiguous(),
        "w1_global_scale": w1_global_scale.contiguous(),
        "w2": w2_fp4.contiguous(),
        "w2_scale": w2_block_scale.contiguous(),
        "w2_global_scale": w2_global_scale.contiguous(),
    }
    result["w1_scale"] = _swizzle_cutile_nvfp4_scales(result["w1_scale"])
    result["w2_scale"] = _swizzle_cutile_nvfp4_scales(result["w2_scale"])
    return result


def prepare_cutile_bf16_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation_type: ActivationType = ActivationType.Swiglu,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the native BF16 weight view for ``CuTileBf16Runner``.

    Gated canonical GEMM1 weights use ``[E, 2I, H]`` in semantic
    ``[up, gate]`` order; preparation swaps the halves before transposing.
    Non-gated weights use ``[E, I, H]`` and need only the transpose. GEMM2
    changes from ``[E, H, I]`` to ``[E, I, H]`` for both activation families.
    """
    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise TypeError(
            "prepare_cutile_bf16_weights expects BF16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    activation_type = ActivationType(activation_type)
    if activation_type not in (ActivationType.Swiglu, ActivationType.Relu2):
        raise ValueError(
            f"unsupported cuTile BF16 activation {activation_type!r}; expected "
            "Swiglu or Relu2."
        )
    expected_w1 = (
        num_local_experts,
        intermediate_size * (2 if activation_type.is_gated else 1),
        hidden_size,
    )
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )

    w1 = w1_bf16.to(device)
    if activation_type.is_gated:
        up, gate = w1.chunk(2, dim=1)
        w1 = torch.cat((gate, up), dim=1)
    return {
        "w1": w1.transpose(1, 2).contiguous(),
        "w2": w2_bf16.to(device).transpose(1, 2).contiguous(),
    }


@torch.no_grad()
def _quantize_mxfp4_linear(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows to packed E2M1 with linear per-32 UE8M0 scales.

    The generic CUDA ``fp4_quantize`` kernel currently produces incorrect
    MXFP4 output on SM90, so the Hopper W4A16 preparation path uses this
    architecture-independent Torch implementation. Work is chunked by rows to
    keep temporary FP32 storage bounded for model-sized expert tensors.
    """
    if weight.ndim != 2 or weight.shape[1] % 32 != 0:
        raise ValueError(
            "MXFP4 linear quantization requires a 2D tensor with K divisible by 32."
        )
    rows, columns = weight.shape
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, columns // 32), dtype=torch.uint8, device=weight.device)
    # Midpoints between the positive E2M1 values
    # [0, .5, 1, 1.5, 2, 3, 4, 6]. ``right=False`` keeps midpoint ties on the
    # lower value, matching argmin over the ordered code-point table.
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=weight.device,
    )
    max_chunk_elements = 8 * 1024 * 1024
    chunk_rows = max(1, max_chunk_elements // columns)
    for begin in range(0, rows, chunk_rows):
        end = min(begin + chunk_rows, rows)
        blocks = weight[begin:end].to(torch.float32).reshape(-1, columns // 32, 32)
        block_scale = blocks.abs().amax(dim=-1) / 6.0
        nonzero = block_scale > 0
        safe_scale = torch.where(nonzero, block_scale, torch.ones_like(block_scale))
        # MX block scales round upward so every finite value remains within the
        # E2M1 magnitude range. E8M0 byte 255 is NaN; finite exponents stop at
        # 127 (byte 254). All-zero blocks use the minimum scale byte 0.
        exponent = torch.ceil(torch.log2(safe_scale)).to(torch.int64)
        exponent = exponent.clamp(-127, 127)
        exponent = torch.where(nonzero, exponent, -127)
        scales[begin:end].copy_((exponent + 127).to(torch.uint8))
        actual_scale = torch.exp2(exponent.to(torch.float32)).unsqueeze(-1)
        scaled = blocks / actual_scale
        magnitude_code = torch.bucketize(scaled.abs(), boundaries, right=False)
        nibbles = magnitude_code | ((scaled < 0).to(torch.int64) << 3)
        nibbles = nibbles.reshape(end - begin, columns)
        packed[begin:end].copy_(
            (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8)
        )
    return packed, scales


def prepare_cutlass_w4a16_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the SM90 mixed-input MXFP4 view for ``CutlassW4A16Runner``.

    An SM90-safe Torch quantizer first produces logical packed E2M1 weights and
    linear UE8M0 scales. Both are then folded into the byte layouts consumed by
    the Hopper mixed-input GEMM.
    """
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise TypeError(
            "prepare_cutlass_w4a16_weights expects BF16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(
            "Cutlass W4A16 requires hidden_size and intermediate_size divisible by 128."
        )
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    expected_w1 = (num_local_experts, gemm1_rows, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )
    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"Cutlass W4A16 preparation requires CUDA, got {device}.")

    def quantize(weight: torch.Tensor, rows: int, cols: int):
        weight = weight.to(device).contiguous().view(num_local_experts * rows, cols)
        packed, scales = _quantize_mxfp4_linear(weight)
        packed = packed.view(num_local_experts, rows, cols // 2)
        scales = scales.view(num_local_experts, rows, cols // 32)
        return (
            interleave_moe_weights_for_sm90_mixed_gemm(packed, "fp4"),
            interleave_moe_scales_for_sm90_mixed_gemm(scales),
        )

    w1, w1_scale = quantize(w1_bf16, gemm1_rows, hidden_size)
    w2, w2_scale = quantize(w2_bf16, hidden_size, intermediate_size)
    return {
        "fc1_expert_weights": w1,
        "fc1_expert_scales": w1_scale,
        "fc2_expert_weights": w2,
        "fc2_expert_scales": w2_scale,
    }


_NVFP4_SF_VEC_SIZE = 16
_NVFP4_SF_SWIZZLE_ROWS = 128


def _nvfp4_swizzled_scale_shape(rows: int, cols: int) -> Tuple[int, int]:
    """CUTLASS 128x4 swizzled NVFP4 scale layout for an ``[N, K]`` matrix."""
    return (
        round_up(rows, _NVFP4_SF_SWIZZLE_ROWS),
        round_up(cols // _NVFP4_SF_VEC_SIZE, 4),
    )


def prepare_cutlass_nvfp4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the CUTLASS NVFP4 view consumed by ``CutlassNvfp4Runner``.

    Canonical source is BF16 ``w1_bf16 [E, rows, H]``, where ``rows`` is
    ``2*I`` in semantic ``[up, gate]`` order for gated activations and ``I``
    for non-gated activations, plus ``w2_bf16 [E, H, I]``. Each expert is quantized
    independently with ``fp4_quantize`` (``sf_vec_size=16``, swizzled scales)
    so 128-row swizzle tiles never cross expert boundaries. Global scales are
    fixed at 1.0 to match the other unified NVFP4 prepares; the kernel still
    receives the six-tensor CUTLASS ``quant_scales`` contract.

    Do not reuse TRTLLM shuffled or BlockMajorK tensors: CUTLASS consumes the
    packed uint8 payload plus the swizzled ``fp4_quantize`` scale buffer.
    """
    from ..quantization.fp4_quantization import fp4_quantize

    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise TypeError(
            "prepare_cutlass_nvfp4_weights expects BF16 weights, got "
            f"w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    if (
        hidden_size % _NVFP4_SF_VEC_SIZE != 0
        or intermediate_size % _NVFP4_SF_VEC_SIZE != 0
    ):
        raise ValueError(
            "Cutlass NVFP4 requires hidden_size and intermediate_size "
            f"divisible by {_NVFP4_SF_VEC_SIZE}."
        )
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    expected_w1 = (num_local_experts, gemm1_rows, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )
    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"Cutlass NVFP4 preparation requires CUDA, got {device}.")

    w1_bf16 = w1_bf16.to(device).contiguous()
    w2_bf16 = w2_bf16.to(device).contiguous()
    # Unit global scale keeps prepare aligned with the other unified NVFP4
    # backends. Per-expert amax scales remain a flat-API concern.
    global_scale = torch.ones(1, device=device, dtype=torch.float32)

    def quantize_experts(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        packed_rows = []
        scale_rows = []
        scale_shape = _nvfp4_swizzled_scale_shape(weight.shape[1], weight.shape[2])
        for expert in range(num_local_experts):
            packed, scale = fp4_quantize(
                weight[expert],
                global_scale=global_scale,
                sf_vec_size=_NVFP4_SF_VEC_SIZE,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=True,
            )
            packed_rows.append(packed)
            scale_rows.append(scale.view(*scale_shape))
        return torch.stack(packed_rows), torch.stack(scale_rows)

    w1_q, w1_block_scale = quantize_experts(w1_bf16)
    w2_q, w2_block_scale = quantize_experts(w2_bf16)
    ones = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    act_global = torch.ones((), device=device, dtype=torch.float32)
    return {
        "fc1_expert_weights": w1_q,
        "fc2_expert_weights": w2_q,
        "fc1_act_global_scale": act_global,
        "fc1_weight_block_scale": w1_block_scale,
        "fc1_dequant_scale": ones,
        "fc2_act_global_scale": act_global.clone(),
        "fc2_weight_block_scale": w2_block_scale,
        "fc2_dequant_scale": ones.clone(),
    }


def _require_canonical_cutlass_bf16_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    name: str,
    activation=None,
    alignment: Optional[int] = None,
    require_cuda: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.device]:
    """Validate canonical BF16 expert weights and move them onto ``device``."""
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise TypeError(
            f"{name} expects BF16 weights, got w1={w1_bf16.dtype}, w2={w2_bf16.dtype}."
        )
    if alignment is not None and (
        hidden_size % alignment != 0 or intermediate_size % alignment != 0
    ):
        raise ValueError(
            f"{name} requires hidden_size and intermediate_size divisible by "
            f"{alignment}."
        )
    activation = _normalize_activation(activation)
    expected_w1 = (
        num_local_experts,
        _gemm1_rows(intermediate_size, activation),
        hidden_size,
    )
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"weight shapes {tuple(w1_bf16.shape)}/{tuple(w2_bf16.shape)} != "
            f"expected {expected_w1}/{expected_w2}."
        )
    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    if require_cuda and device.type != "cuda":
        raise ValueError(f"{name} requires CUDA, got {device}.")
    return w1_bf16.to(device).contiguous(), w2_bf16.to(device).contiguous(), device


def prepare_cutlass_fp8_per_tensor_activations(
    hidden_states_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, H]`` BF16 activations to E4M3 plus a scalar dequant scale."""
    if hidden_states_bf16.dtype != torch.bfloat16 or hidden_states_bf16.dim() != 2:
        raise ValueError(
            "prepare_cutlass_fp8_per_tensor_activations expects a 2D BF16 tensor, "
            f"got shape={tuple(hidden_states_bf16.shape)}, "
            f"dtype={hidden_states_bf16.dtype}."
        )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = hidden_states_bf16.float().abs().amax()
    dequant = torch.where(
        amax > 0, amax / fp8_max, torch.ones_like(amax, dtype=torch.float32)
    ).to(torch.float32)
    quantized = (hidden_states_bf16.float() / dequant).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn), dequant.reshape(())


def prepare_cutlass_fp8_per_tensor_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the unshuffled per-tensor FP8 view for ``CutlassFp8PerTensorRunner``.

    Each expert uses one E4M3 multiplier. The returned ``fc1_dequant`` /
    ``fc2_dequant`` tensors are the CUTLASS dequant scales (``amax / fp8_max``),
    not TRTLLM's inverted calibration multipliers.
    """
    w1_bf16, w2_bf16, device = _require_canonical_cutlass_bf16_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        name="prepare_cutlass_fp8_per_tensor_weights",
        activation=activation,
        device=device,
    )
    w1_q, w1_mult = _quantize_fp8_per_expert(w1_bf16)
    w2_q, w2_mult = _quantize_fp8_per_expert(w2_bf16)
    return {
        "fc1_expert_weights": w1_q,
        "fc2_expert_weights": w2_q,
        "fc1_dequant": (1.0 / w1_mult).contiguous(),
        "fc2_dequant": (1.0 / w2_mult).contiguous(),
    }


def prepare_cutlass_fp8_block_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the DeepSeek 128x128 FP8 block-scale view for CUTLASS.

    Reuses the unified DeepSeek weight quantizer and leaves tensors unshuffled.
    ``hidden_size`` and ``intermediate_size`` must be divisible by 128.
    """
    w1_bf16, w2_bf16, _device = _require_canonical_cutlass_bf16_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        name="prepare_cutlass_fp8_block_weights",
        activation=activation,
        alignment=128,
        device=device,
    )
    w1_q, w1_scale = _deepseek_fp8_quantize_weights(w1_bf16)
    w2_q, w2_scale = _deepseek_fp8_quantize_weights(w2_bf16)
    return {
        "fc1_expert_weights": w1_q.contiguous(),
        "fc2_expert_weights": w2_q.contiguous(),
        "fc1_block_scale": w1_scale.contiguous(),
        "fc2_block_scale": w2_scale.contiguous(),
    }


def prepare_cutlass_mxfp8_activations(
    hidden_states_bf16: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, H]`` BF16 activations to MXFP8 with swizzled scales."""
    from ..quantization.fp8_quantization import mxfp8_quantize

    if hidden_states_bf16.dtype != torch.bfloat16 or hidden_states_bf16.dim() != 2:
        raise ValueError(
            "prepare_cutlass_mxfp8_activations expects a 2D BF16 tensor, "
            f"got shape={tuple(hidden_states_bf16.shape)}, "
            f"dtype={hidden_states_bf16.dtype}."
        )
    if hidden_states_bf16.shape[1] % 32 != 0:
        raise ValueError(
            "MXFP8 activations require hidden_size divisible by 32, got "
            f"{hidden_states_bf16.shape[1]}."
        )
    return mxfp8_quantize(hidden_states_bf16, is_sf_swizzled_layout=True, alignment=32)


def _quantize_mxfp4_experts(
    weight: torch.Tensor, num_local_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    from ..quantization.fp4_quantization import mxfp4_quantize

    packed_rows = []
    scale_rows = []
    for expert in range(num_local_experts):
        packed, scale = mxfp4_quantize(weight[expert])
        packed_rows.append(packed)
        scale_rows.append(scale)
    return torch.stack(packed_rows), torch.stack(scale_rows)


def prepare_cutlass_mxfp8_mxfp4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the CUTLASS MXFP4 weight view consumed with MXFP8 activations.

    MXFP4 block scales are 32-wide, but the fused-MoE binding still requires
    ``hidden_size`` and ``intermediate_size`` divisible by 128.
    """
    w1_bf16, w2_bf16, device = _require_canonical_cutlass_bf16_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        name="prepare_cutlass_mxfp8_mxfp4_weights",
        activation=activation,
        alignment=128,
        require_cuda=True,
        device=device,
    )
    w1_q, w1_scale = _quantize_mxfp4_experts(w1_bf16, num_local_experts)
    w2_q, w2_scale = _quantize_mxfp4_experts(w2_bf16, num_local_experts)
    fake_input_scale = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    return {
        "fc1_expert_weights": w1_q.contiguous(),
        "fc2_expert_weights": w2_q.contiguous(),
        "fc1_expert_scales": w1_scale.contiguous(),
        "fc2_expert_scales": w2_scale.contiguous(),
        "fc1_input_scale": fake_input_scale,
        "fc2_input_scale": fake_input_scale.clone(),
    }


def _pack_mxfp8_weight_scales(
    scale_u8: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    # CUTLASS MXFP8 SF N-dim is alignToSfDim(N, 128). Callers must pass the
    # already-gated row count (2 * round_up(I, 128) for SwiGLU fc1); combining
    # as round_up(2*I, 128) is smaller when I % 128 != 0.
    num_experts = scale_u8.size(0)
    aligned_rows = round_up(rows, 128)
    aligned_k_scales = round_up(cols // 32, 4)
    return (
        scale_u8.contiguous()
        .view(num_experts, aligned_rows, aligned_k_scales)
        .view(torch.int32)
        .contiguous()
    )


def _quantize_mxfp8_experts(
    weight: torch.Tensor, num_local_experts: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    from ..quantization.fp8_quantization import mxfp8_quantize

    packed_rows = []
    scale_rows = []
    for expert in range(num_local_experts):
        packed, scale = mxfp8_quantize(
            weight[expert], is_sf_swizzled_layout=True, alignment=32
        )
        packed_rows.append(packed)
        scale_rows.append(scale)
    return torch.stack(packed_rows), torch.stack(scale_rows)


def prepare_cutlass_mxfp8_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the CUTLASS MXFP8 weight view consumed with MXFP8 activations.

    MXFP8 block scales are 32-wide, but the fused-MoE binding requires
    ``hidden_size`` and ``intermediate_size`` divisible by 128. The gated fc1
    SF N-dim is ``2 * round_up(I, 128)``, which only matches
    ``mxfp8_quantize``'s ``round_up(2*I, 128)`` output when ``I % 128 == 0``.
    """
    w1_bf16, w2_bf16, device = _require_canonical_cutlass_bf16_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        name="prepare_cutlass_mxfp8_weights",
        activation=activation,
        alignment=128,
        require_cuda=True,
        device=device,
    )
    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    w1_q, w1_scale = _quantize_mxfp8_experts(w1_bf16, num_local_experts)
    w2_q, w2_scale = _quantize_mxfp8_experts(w2_bf16, num_local_experts)
    fake_input_scale = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    return {
        "fc1_expert_weights": w1_q.contiguous(),
        "fc2_expert_weights": w2_q.contiguous(),
        "fc1_expert_scales": _pack_mxfp8_weight_scales(
            w1_scale, gemm1_rows, hidden_size
        ),
        "fc2_expert_scales": _pack_mxfp8_weight_scales(
            w2_scale, hidden_size, intermediate_size
        ),
        "fc1_input_scale": fake_input_scale,
        "fc2_input_scale": fake_input_scale.clone(),
    }


def _quantize_int4_grouped(
    weight: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4 pack with per-group BF16 scales along the last dimension."""
    if weight.ndim != 3 or weight.shape[-1] % group_size != 0:
        raise ValueError(
            "INT4 grouped quantization requires a 3D tensor whose last dim is "
            f"divisible by {group_size}; got {tuple(weight.shape)}."
        )
    experts, rows, cols = weight.shape
    blocks = weight.to(torch.float32).reshape(
        experts, rows, cols // group_size, group_size
    )
    amax = blocks.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / 7.0, torch.ones_like(amax))
    quantized = (blocks / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    quantized = quantized.reshape(experts, rows, cols)
    even = (quantized[..., 0::2] & 0xF).to(torch.uint8)
    odd = (quantized[..., 1::2] & 0xF).to(torch.uint8)
    packed = even | (odd << 4)
    return packed.contiguous(), scale.to(torch.bfloat16).contiguous()


def prepare_cutlass_w4a8_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the SM90 packed INT4 view for ``CutlassW4A8Runner``.

    Canonical BF16 weights are quantized in groups of 128, then folded with the
    mixed-input INT4 interleave. Activation prequant scales are identity so the
    kernel can quantize BF16 activations internally.
    """
    group_size = 128
    w1_bf16, w2_bf16, device = _require_canonical_cutlass_bf16_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        name="prepare_cutlass_w4a8_weights",
        activation=activation,
        alignment=group_size,
        require_cuda=True,
        device=device,
    )
    w1_packed, w1_scale = _quantize_int4_grouped(w1_bf16, group_size)
    w2_packed, w2_scale = _quantize_int4_grouped(w2_bf16, group_size)
    w1_il = interleave_moe_weights_for_sm90_mixed_gemm(w1_packed, "int4")
    w2_il = interleave_moe_weights_for_sm90_mixed_gemm(w2_packed, "int4")
    w1_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(w1_scale, group_size)
    w2_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(w2_scale, group_size)
    ones_h = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
    ones_i = torch.ones(intermediate_size, device=device, dtype=torch.bfloat16)
    ones_e = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    empty = torch.empty(0, device=device, dtype=torch.bfloat16)
    return {
        "fc1_expert_weights": w1_il,
        "fc2_expert_weights": w2_il,
        "fc1_expert_scales": w1_scale_il,
        "fc2_expert_scales": w2_scale_il,
        "fc1_act_scale": ones_h,
        "fc2_act_scale": ones_i,
        "fc1_zero": empty,
        "fc2_zero": empty.clone(),
        "fc1_alpha": ones_e,
        "fc2_alpha": ones_e.clone(),
    }


def prepare_cutlass_humming_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the Humming MXFP4 x FP8 mixed-input view for SM90 CUTLASS.

    Logical MXFP4 is produced with the architecture-independent Torch
    quantizer, then rewritten and interleaved by
    :func:`preprocess_moe_weights_for_sm90_mixed_gemm_humming`.
    """
    humming_epilogue_compensation = 64.0
    w1_bf16, w2_bf16, device = _require_canonical_cutlass_bf16_weights(
        w1_bf16,
        w2_bf16,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        name="prepare_cutlass_humming_weights",
        activation=activation,
        alignment=128,
        require_cuda=True,
        device=device,
    )

    def quantize(weight: torch.Tensor, rows: int, cols: int):
        packed, scales = _quantize_mxfp4_linear(
            weight.view(num_local_experts * rows, cols)
        )
        return (
            packed.view(num_local_experts, rows, cols // 2),
            scales.view(num_local_experts, rows, cols // 32),
        )

    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    w1_packed, w1_scale = quantize(w1_bf16, gemm1_rows, hidden_size)
    w2_packed, w2_scale = quantize(w2_bf16, hidden_size, intermediate_size)
    w1_il, w1_scale_il, w1_residual = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(w1_packed, w1_scale)
    )
    w2_il, w2_scale_il, w2_residual = (
        preprocess_moe_weights_for_sm90_mixed_gemm_humming(w2_packed, w2_scale)
    )
    reserved = torch.ones((), device=device, dtype=torch.float32)
    return {
        "fc1_expert_weights": w1_il,
        "fc2_expert_weights": w2_il,
        "fc1_expert_scales": w1_scale_il,
        "fc2_expert_scales": w2_scale_il,
        "fc1_residual_scale": (
            w1_residual * humming_epilogue_compensation
        ).contiguous(),
        "fc2_residual_scale": (
            w2_residual * humming_epilogue_compensation
        ).contiguous(),
        "fc2_act_global": reserved,
    }


def _interleave_linear_and_gate(
    x: torch.Tensor, group_size: int = 64, dim: int = -1
) -> torch.Tensor:
    """Interleave the linear and gate halves of a SwiGLU gemm1 weight."""
    sizes = x.size()
    dim = dim % x.dim()
    assert sizes[dim] % (group_size * 2) == 0
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1 :]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size, *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


def prepare_cute_dsl_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    variant=None,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the CuteDSL FP4 ``cute_dsl`` weight view.

    Gemm1 weights get the linear/gate interleave only for gated activations;
    non-gated ones (ReLU2) skip it and keep their ``[E, I, H]`` rows as-is.
    ``variant`` selects NVFP4/W4A4, MXFP4/W4A8, or W4A16 weights.
    Starts from the same canonical bf16 expert weights as
    :func:`prepare_trtllm_fp4_weights`, so a single weight set can feed both
    backends and a shared reference.

    Returns
    -------
    dict
        Keys expected by ``CuteDslRunner.pack_inputs``: ``w1_weight``,
        ``w1_weight_sf``, ``w1_alpha``, ``fc2_input_scale``, ``w2_weight``,
        ``w2_weight_sf``, ``w2_alpha``.
    """
    from ..cute_dsl.utils import convert_sf_to_mma_layout
    from ..fp4_quantization import fp4_quantize
    from .api import QuantVariant

    if variant is None:
        variant = QuantVariant.NVFP4
    if variant not in (QuantVariant.NVFP4, QuantVariant.MXFP4, QuantVariant.W4A16):
        raise ValueError(
            f"CuTe-DSL FP4 weight preparation does not support {variant!r}"
        )

    if device is None:
        device = w1_bf16.device
    # Honor the documented device target (no-op if already resident); avoids
    # mixed-device ops when canonical weights are on CPU.
    w1_bf16 = w1_bf16.to(device)
    w2_bf16 = w2_bf16.to(device)

    is_mxfp4 = variant is QuantVariant.MXFP4
    if is_mxfp4 and (hidden_size % 128 or intermediate_size % 128):
        raise ValueError(
            "CuTe-DSL MXFP4 requires hidden and intermediate sizes divisible by 128"
        )
    sf_vec_size = 32 if is_mxfp4 else 16
    gs = torch.tensor([1.0], device=device, dtype=torch.float32)

    activation = _normalize_activation(activation)
    gemm1_rows = _gemm1_rows(intermediate_size, activation)
    w1_interleaved = (
        _interleave_linear_and_gate(w1_bf16, group_size=64, dim=1)
        if activation.is_gated
        else w1_bf16
    )
    w1_flat = w1_interleaved.view(num_local_experts * gemm1_rows, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat,
        global_scale=gs,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=is_mxfp4,
        is_sf_swizzled_layout=True,
    )
    w1_weight = w1_q_flat.view(num_local_experts, gemm1_rows, hidden_size // 2)
    w1_weight_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=gemm1_rows,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )

    w2_flat = w2_bf16.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat,
        global_scale=gs,
        sf_vec_size=sf_vec_size,
        sf_use_ue8m0=is_mxfp4,
        is_sf_swizzled_layout=True,
    )
    w2_weight = w2_q_flat.view(num_local_experts, hidden_size, intermediate_size // 2)
    w2_weight_sf = convert_sf_to_mma_layout(
        w2_sf_flat,
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )

    ones = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    view = {
        "w1_weight": w1_weight,
        "w1_weight_sf": w1_weight_sf,
        "w1_alpha": ones,
        "w2_weight": w2_weight,
        "w2_weight_sf": w2_weight_sf,
        "w2_alpha": ones,
    }
    if not is_mxfp4:
        view["fc2_input_scale"] = gs
    return view


def _quantize_b12x_expert_weights(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize canonical expert weights in the b12x non-interleaved layout."""
    from ..cute_dsl.utils import convert_sf_to_mma_layout
    from ..fp4_quantization import fp4_quantize

    num_experts, rows, columns = weights.shape
    weight_q, weight_sf = fp4_quantize(
        weights.reshape(num_experts * rows, columns),
        global_scale=torch.ones(1, device=weights.device, dtype=torch.float32),
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    weight_q = weight_q.reshape(num_experts, rows, columns // 2)
    weight_sf = convert_sf_to_mma_layout(
        weight_sf,
        m=rows,
        k=columns,
        num_groups=num_experts,
        sf_vec_size=16,
    )
    return weight_q, weight_sf


def prepare_b12x_nvfp4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: str = "silu",
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the SM12x NVFP4 ``b12x_nvfp4`` weight view.

    Both gemms are NVFP4 block-quantized in the b12x non-interleaved layout.
    ``B12xMoEWrapper`` handles ragged padding and kernel weight-view caching.

    Parameters
    ----------
    w1_bf16 : Tensor
        Up+gate expert weights ``[E, 2*I, H]`` stored as ``[up, gate]``, or
        ReLU2 weights ``[E, I, H]``.
    w2_bf16 : Tensor
        Down-projection expert weights ``[E, H, I]``.
    num_local_experts, hidden_size, intermediate_size : int
        Expert geometry.
    activation : str
        Kernel activation name: ``"silu"``, ``"gelu_tanh"``, or ``"relu2"``.
    device : torch.device, optional
        Target device; defaults to ``w1_bf16.device``.

    Returns
    -------
    dict
        Keys expected by ``B12xNvfp4Runner.pack_inputs``: ``w1_weight``,
        ``w1_weight_sf``, ``w1_alpha``, ``fc2_input_scale``, ``w2_weight``,
        ``w2_weight_sf``, ``w2_alpha``.
    """
    from .cute_dsl.blackwell_sm12x.moe_activation import is_gated_activation

    supported_activations = {"silu", "gelu_tanh", "relu2"}
    if activation not in supported_activations:
        raise ValueError(
            f"unsupported b12x NVFP4 activation {activation!r}; expected one of "
            f"{sorted(supported_activations)}."
        )

    if device is None:
        device = w1_bf16.device
    device = torch.device(device)
    w1_bf16 = w1_bf16.to(device)
    w2_bf16 = w2_bf16.to(device)
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise TypeError("b12x canonical weights must use torch.bfloat16.")

    is_gated = is_gated_activation(activation)
    w1_rows = intermediate_size * (2 if is_gated else 1)
    expected_w1 = (num_local_experts, w1_rows, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1:
        raise ValueError(
            f"expected w1_bf16 shape {expected_w1}, got {tuple(w1_bf16.shape)}"
        )
    if tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            f"expected w2_bf16 shape {expected_w2}, got {tuple(w2_bf16.shape)}"
        )
    if hidden_size % 16 != 0 or intermediate_size % 16 != 0:
        raise ValueError("b12x NVFP4 dimensions must be multiples of 16.")

    w1_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    w2_alpha = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    fc2_input_scale = torch.ones(1, device=device, dtype=torch.float32)
    w1_weight, w1_weight_sf = _quantize_b12x_expert_weights(w1_bf16)
    w2_weight, w2_weight_sf = _quantize_b12x_expert_weights(w2_bf16)

    return {
        "w1_weight": w1_weight,
        "w1_weight_sf": w1_weight_sf,
        "w1_alpha": w1_alpha,
        "fc2_input_scale": fc2_input_scale.contiguous(),
        "w2_weight": w2_weight,
        "w2_weight_sf": w2_weight_sf,
        "w2_alpha": w2_alpha,
    }


def prepare_b12x_w4a16_weights(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    source_format: str = "modelopt",
) -> Dict[str, torch.Tensor]:
    """Build the SM12x W4A16 ``b12x_w4a16`` weight view.

    The existing b12x packed-weight cache is populated for bf16 activations.
    Returned tensors retain the ``B12xMoEWrapper.run`` input layout.

    Parameters
    ----------
    w1_fp4, w2_fp4 : Tensor
        Packed checkpoint expert weights.
    w1_blockscale, w2_blockscale : Tensor
        Per-block checkpoint scales.
    w1_global_scale, w2_global_scale : Tensor
        Per-expert checkpoint scales.
    activation : str
        Kernel activation name: ``"silu"`` or ``"relu2"``.
    source_format : str
        Checkpoint scale convention: ``"modelopt"`` or
        ``"compressed_tensors"``.

    Returns
    -------
    dict
        Keys expected by ``B12xMoEWrapper.run``: ``w1_weight``,
        ``w1_weight_sf``, ``w1_alpha``, ``w2_weight``, ``w2_weight_sf``,
        ``w2_alpha``.
    """
    from .cute_dsl.blackwell_sm12x.moe_dispatch import (
        _get_w4a16_packed_weights,
    )
    from .cute_dsl.blackwell_sm12x.moe_w4a16_prepare import (
        _normalize_source_format,
        _source_global_scale,
    )

    source_format = _normalize_source_format(source_format)
    w1_global_scale = _source_global_scale(w1_global_scale, source_format=source_format)
    w2_global_scale = _source_global_scale(w2_global_scale, source_format=source_format)
    _get_w4a16_packed_weights(
        w1_weight=w1_fp4,
        w1_weight_sf=w1_blockscale,
        w1_alpha=w1_global_scale,
        w2_weight=w2_fp4,
        w2_weight_sf=w2_blockscale,
        w2_alpha=w2_global_scale,
        activation=activation,
        params_dtype=torch.bfloat16,
        source_format="modelopt",
    )
    return {
        "w1_weight": w1_fp4,
        "w1_weight_sf": w1_blockscale,
        "w1_alpha": w1_global_scale,
        "w2_weight": w2_fp4,
        "w2_weight_sf": w2_blockscale,
        "w2_alpha": w2_global_scale,
    }


def __getattr__(name: str):
    if name == "prepare_cute_dsl_nvfp4_weights":
        warnings.warn(
            "prepare_cute_dsl_nvfp4_weights is deprecated; use "
            "prepare_cute_dsl_weights with variant=QuantVariant.NVFP4 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return prepare_cute_dsl_weights
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
