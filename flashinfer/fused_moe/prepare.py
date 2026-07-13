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
MMA reorder).  These helpers turn canonical bf16 expert weights into the
backend-native ``MoEWeightPack`` views, so that preparation lives in the
implementation surface rather than being copy-pasted into tests and benchmarks
(design doc CR2/CR7; reviewer comments C6, C7, C31, C32).

The canonical NVFP4 helpers are exposed as
``TrtllmFp4Config.prepare_weights(...)`` / ``CuteDslConfig.prepare_weights(...)``
static helpers (see ``api.py``).  The SM90 Humming-style MXFP4 x FP8 helper is
currently exposed as a flat helper for the CUTLASS fused-MoE path.
"""

from __future__ import annotations

import functools
import struct
from typing import Dict, Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.moe import (
    sm90_mixed_gemm_humming_weight_preprocess_trace_dispatch,
    sm90_mixed_gemm_scale_interleave_trace,
    sm90_mixed_gemm_weight_interleave_trace,
)
from ..utils import get_compute_capability

# Module-level permute-index cache.  Permute indices depend only on weight
# dims, so the cache is safe to reuse across shapes and calls.
_TRTLLM_PERMUTE_CACHE: dict = {}


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
        is one FP32 value per expert and should be folded into the routed-token
        activation scale together with Humming's fixed ``2^6`` compensation.

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
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: Optional[torch.device] = None,
    permute_cache: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """Build the TRTLLM NVFP4 ``trtllm_fp4_routed`` weight view.

    Layout is ``Shuffled_MajorK`` — the only NVFP4-compatible trtllm-gen combo
    today: per-expert gated-act reorder + MMA shuffle on the packed weights and
    ``block_scale_interleave`` on the block scales.

    Parameters
    ----------
    w1_bf16 : Tensor
        Gate+up expert weights ``[num_local_experts, 2*intermediate_size, hidden_size]``.
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
        ``gemm1_weights_scale``, ``gemm1_alpha``, ``gemm2_weights``,
        ``gemm2_weights_scale``, ``output1_scale_scalar``,
        ``output1_scale_gate_scalar``, ``output2_scale_scalar``.
    """
    from ..fp4_quantization import fp4_quantize
    from ..quantization.fp4_quantization import block_scale_interleave
    from .core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
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

    sf_vec_size = 16
    epilogue_tile_m = 128  # TRTLLM kernel-internal constant

    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_flat = w1_bf16.view(num_local_experts * 2 * intermediate_size, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat,
        global_scale=w1_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    g1_w = w1_q_flat.view(
        num_local_experts, 2 * intermediate_size, hidden_size // 2
    ).view(torch.uint8)
    g1_s = w1_sf_flat.view(torch.float8_e4m3fn).reshape(
        num_local_experts, 2 * intermediate_size, hidden_size // sf_vec_size
    )

    w2_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w2_flat = w2_bf16.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat,
        global_scale=w2_gs,
        sf_vec_size=sf_vec_size,
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
            permute_cache, g1_w[i], epilogue_tile_m, is_gated_act_gemm=True
        )
        g1_w_sh.append(g1_w[i][p.to(device)].contiguous())

        p_sf = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            g1_s[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        g1_s_sh.append(
            block_scale_interleave(
                g1_s[i].view(torch.uint8)[p_sf.to(device)].contiguous()
            )
        )

        p = get_w2_permute_indices_with_cache(permute_cache, g2_w[i], epilogue_tile_m)
        g2_w_sh.append(g2_w[i][p.to(device)].contiguous())

        p_sf = get_w2_permute_indices_with_cache(
            permute_cache,
            g2_s[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        g2_s_sh.append(
            block_scale_interleave(
                g2_s[i].view(torch.uint8)[p_sf.to(device)].contiguous()
            )
        )

    ones = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    return {
        "gemm1_weights": torch.stack(g1_w_sh),
        "gemm1_weights_scale": torch.stack(g1_s_sh)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, 2 * intermediate_size, hidden_size // sf_vec_size),
        "gemm1_alpha": ones,
        "gemm2_weights": torch.stack(g2_w_sh),
        "gemm2_weights_scale": torch.stack(g2_s_sh)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, hidden_size, intermediate_size // sf_vec_size),
        "output1_scale_scalar": ones,
        "output1_scale_gate_scalar": ones,
        "output2_scale_scalar": ones,
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


def prepare_cute_dsl_nvfp4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the CuteDSL NVFP4 ``cute_dsl_nvfp4`` weight view.

    Gemm1 weights get the SwiGLU linear/gate interleave; both gemms are NVFP4
    block-quantized (swizzled) with scales converted to the CuteDSL MMA layout.
    Starts from the same canonical bf16 expert weights as
    :func:`prepare_trtllm_fp4_weights`, so a single weight set can feed both
    backends and a shared reference.

    Returns
    -------
    dict
        Keys expected by ``CuteDslNvfp4Runner.pack_inputs``: ``w1_weight``,
        ``w1_weight_sf``, ``w1_alpha``, ``fc2_input_scale``, ``w2_weight``,
        ``w2_weight_sf``, ``w2_alpha``.
    """
    from ..cute_dsl.utils import convert_sf_to_mma_layout
    from ..fp4_quantization import fp4_quantize

    if device is None:
        device = w1_bf16.device
    # Honor the documented device target (no-op if already resident); avoids
    # mixed-device ops when canonical weights are on CPU.
    w1_bf16 = w1_bf16.to(device)
    w2_bf16 = w2_bf16.to(device)

    sf_vec_size = 16
    gs = torch.tensor([1.0], device=device, dtype=torch.float32)

    w1_interleaved = _interleave_linear_and_gate(w1_bf16, group_size=64, dim=1)
    w1_flat = w1_interleaved.view(
        num_local_experts * 2 * intermediate_size, hidden_size
    )
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat, global_scale=gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
    )
    w1_weight = w1_q_flat.view(
        num_local_experts, 2 * intermediate_size, hidden_size // 2
    )
    w1_weight_sf = convert_sf_to_mma_layout(
        w1_sf_flat,
        m=2 * intermediate_size,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )

    w2_flat = w2_bf16.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat, global_scale=gs, sf_vec_size=sf_vec_size, is_sf_swizzled_layout=True
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
    return {
        "w1_weight": w1_weight,
        "w1_weight_sf": w1_weight_sf,
        "w1_alpha": ones,
        "fc2_input_scale": torch.tensor([1.0], device=device, dtype=torch.float32),
        "w2_weight": w2_weight,
        "w2_weight_sf": w2_weight_sf,
        "w2_alpha": ones,
    }
