"""First-class NVFP4 weight-preparation helpers for the unified MoE API.

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

They are exposed as ``TrtllmFp4Config.prepare_weights(...)`` /
``CuteDslConfig.prepare_weights(...)`` static helpers (see ``api.py``).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch

# Module-level permute-index cache.  Permute indices depend only on weight
# dims, so the cache is safe to reuse across shapes and calls.
_TRTLLM_PERMUTE_CACHE: dict = {}


def _resolve_expert_global_scales(
    scales: Optional[torch.Tensor],
    weights_bf16: torch.Tensor,
    num_local_experts: int,
    device: torch.device,
) -> torch.Tensor:
    """Per-expert float32 ``[num_local_experts]`` NVFP4 global scales.

    ``None`` computes the calibrated scale ``(448 * 6) / amax`` per expert,
    mirroring the canonical trtllm-gen quantize path
    (``calculate_fp4_global_scale_factor`` / ``quant_fp4_batches`` in
    ``tests/moe/test_trtllm_gen_fused_moe.py``).  A scalar broadcasts to all
    local experts.
    """
    if scales is None:
        amax = weights_bf16.float().abs().nan_to_num().amax(dim=(1, 2))
        return (448.0 * 6.0) / amax
    scales = scales.to(device=device, dtype=torch.float32).reshape(-1)
    if scales.numel() == 1:
        scales = scales.expand(num_local_experts)
    return scales.contiguous()


def _resolve_intermediate_global_scale(
    scale: Optional[Union[float, torch.Tensor]], device: torch.device
) -> torch.Tensor:
    """Scalar float32 ``[1]`` global scale for the gemm1→gemm2 activation
    requantization (``c_global_sf`` / ``fc2_input_scale``).  ``None`` → 1.0."""
    if scale is None:
        return torch.ones(1, device=device, dtype=torch.float32)
    if not isinstance(scale, torch.Tensor):
        return torch.tensor([float(scale)], device=device, dtype=torch.float32)
    return scale.to(device=device, dtype=torch.float32).reshape(1)


def prepare_trtllm_fp4_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: Optional[torch.device] = None,
    permute_cache: Optional[dict] = None,
    gemm1_scales_global: Optional[torch.Tensor] = None,
    gemm2_scales_global: Optional[torch.Tensor] = None,
    intermediate_scale_global: Optional[Union[float, torch.Tensor]] = None,
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
    gemm1_scales_global, gemm2_scales_global : Tensor, optional
        Per-expert float32 ``[num_local_experts]`` (or scalar) NVFP4 global
        scales used to quantize the weights — e.g. the calibrated values from
        a ModelOpt checkpoint.  ``None`` computes the canonical
        ``(448 * 6) / amax`` per expert.
    intermediate_scale_global : float or Tensor, optional
        Global scale (``c_global_sf``) for requantizing the gemm1 activation
        output that feeds gemm2.  ``None`` → 1.0.

    Returns
    -------
    dict
        Keys expected by ``TrtllmFp4RoutedRunner.pack_inputs``: ``gemm1_weights``,
        ``gemm1_weights_scale``, ``gemm1_alpha``, ``gemm2_weights``,
        ``gemm2_weights_scale``, ``output1_scale_scalar``,
        ``output1_scale_gate_scalar``, ``output2_scale_scalar`` (the
        weight-side dequant factors, canonical formulas from
        ``tests/moe/test_trtllm_gen_fused_moe.py`` at activation global scale
        1.0), plus the raw ``gemm1_scales_global`` / ``gemm2_scales_global`` /
        ``intermediate_scale_global`` so the runner can fold the activation
        global scale in at pack time (gh #3548).
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

    g1_gs = _resolve_expert_global_scales(
        gemm1_scales_global, w1_bf16, num_local_experts, device
    )
    g2_gs = _resolve_expert_global_scales(
        gemm2_scales_global, w2_bf16, num_local_experts, device
    )
    c_gs = _resolve_intermediate_global_scale(intermediate_scale_global, device)

    # Quantize per expert — the global scale is per-expert (mirrors
    # quant_fp4_batches in the canonical trtllm-gen test path).
    w1_q_list, w1_sf_list, w2_q_list, w2_sf_list = [], [], [], []
    for i in range(num_local_experts):
        q, sf = fp4_quantize(
            w1_bf16[i],
            global_scale=g1_gs[i : i + 1],
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=False,
        )
        w1_q_list.append(q)
        w1_sf_list.append(sf)
        q, sf = fp4_quantize(
            w2_bf16[i],
            global_scale=g2_gs[i : i + 1],
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=False,
        )
        w2_q_list.append(q)
        w2_sf_list.append(sf)

    g1_w = torch.stack(w1_q_list).view(torch.uint8)
    g1_s = (
        torch.stack(w1_sf_list)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, 2 * intermediate_size, hidden_size // sf_vec_size)
    )
    g2_w = torch.stack(w2_q_list).view(torch.uint8)
    g2_s = (
        torch.stack(w2_sf_list)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, hidden_size, intermediate_size // sf_vec_size)
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
    # Weight-side dequant scalars (activation global scale folded in by the
    # runner) — canonical formulas, gated activation, from
    # tests/moe/test_trtllm_gen_fused_moe.py (scale_c_fc1 / scale_gate_fc1 /
    # scale_c_fc2 with hidden_states_scale_global = 1).
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
        "output1_scale_scalar": c_gs * (1.0 / g1_gs),
        "output1_scale_gate_scalar": 1.0 / g1_gs,
        "output2_scale_scalar": (1.0 / c_gs) * (1.0 / g2_gs),
        "gemm1_scales_global": g1_gs,
        "gemm2_scales_global": g2_gs,
        "intermediate_scale_global": c_gs,
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
    gemm1_scales_global: Optional[torch.Tensor] = None,
    gemm2_scales_global: Optional[torch.Tensor] = None,
    intermediate_scale_global: Optional[Union[float, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Build the CuteDSL NVFP4 ``cute_dsl_nvfp4`` weight view.

    Gemm1 weights get the SwiGLU linear/gate interleave; both gemms are NVFP4
    block-quantized (swizzled) with scales converted to the CuteDSL MMA layout.
    Starts from the same canonical bf16 expert weights as
    :func:`prepare_trtllm_fp4_weights`, so a single weight set can feed both
    backends and a shared reference.

    ``gemm1_scales_global`` / ``gemm2_scales_global`` /
    ``intermediate_scale_global`` follow the same semantics as in
    :func:`prepare_trtllm_fp4_weights`.

    Returns
    -------
    dict
        Keys expected by ``CuteDslNvfp4Runner.pack_inputs``: ``w1_weight``,
        ``w1_weight_sf``, ``w1_alpha``, ``fc2_input_scale``, ``w2_weight``,
        ``w2_weight_sf``, ``w2_alpha``, plus the raw global scales as in
        :func:`prepare_trtllm_fp4_weights`.  The kernel computes
        ``alpha * (A @ B)`` on a dequantized NVFP4 product that carries
        ``act_gs * weight_gs``, so ``w1_alpha = 1 / gemm1_scales_global``
        (``act_gs`` folded in by the runner, gh #3548), ``fc2_input_scale =
        intermediate_scale_global`` (the requant global scale itself), and
        ``w2_alpha = 1 / (intermediate_scale_global * gemm2_scales_global)``.
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
    g1_gs = _resolve_expert_global_scales(
        gemm1_scales_global, w1_bf16, num_local_experts, device
    )
    g2_gs = _resolve_expert_global_scales(
        gemm2_scales_global, w2_bf16, num_local_experts, device
    )
    c_gs = _resolve_intermediate_global_scale(intermediate_scale_global, device)

    # Quantize per expert (per-expert global scales).  The swizzled-sf layout
    # is tiled in 128-row blocks, and each expert's rows (2*intermediate_size
    # / hidden_size) are 128-aligned, so per-expert sf blocks concatenate to
    # the same bytes the flat quantization produced.
    w1_interleaved = _interleave_linear_and_gate(w1_bf16, group_size=64, dim=1)
    w1_q_list, w1_sf_list, w2_q_list, w2_sf_list = [], [], [], []
    for i in range(num_local_experts):
        q, sf = fp4_quantize(
            w1_interleaved[i],
            global_scale=g1_gs[i : i + 1],
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w1_q_list.append(q)
        w1_sf_list.append(sf)
        q, sf = fp4_quantize(
            w2_bf16[i],
            global_scale=g2_gs[i : i + 1],
            sf_vec_size=sf_vec_size,
            is_sf_swizzled_layout=True,
        )
        w2_q_list.append(q)
        w2_sf_list.append(sf)

    w1_weight = torch.stack(w1_q_list).view(
        num_local_experts, 2 * intermediate_size, hidden_size // 2
    )
    w1_weight_sf = convert_sf_to_mma_layout(
        torch.cat([sf.reshape(-1) for sf in w1_sf_list]),
        m=2 * intermediate_size,
        k=hidden_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )

    w2_weight = torch.stack(w2_q_list).view(
        num_local_experts, hidden_size, intermediate_size // 2
    )
    w2_weight_sf = convert_sf_to_mma_layout(
        torch.cat([sf.reshape(-1) for sf in w2_sf_list]),
        m=hidden_size,
        k=intermediate_size,
        num_groups=num_local_experts,
        sf_vec_size=sf_vec_size,
    )

    return {
        "w1_weight": w1_weight,
        "w1_weight_sf": w1_weight_sf,
        "w1_alpha": 1.0 / g1_gs,
        "fc2_input_scale": c_gs,
        "w2_weight": w2_weight,
        "w2_weight_sf": w2_weight_sf,
        "w2_alpha": (1.0 / c_gs) * (1.0 / g2_gs),
        "gemm1_scales_global": g1_gs,
        "gemm2_scales_global": g2_gs,
        "intermediate_scale_global": c_gs,
    }
