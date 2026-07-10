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

They are exposed as ``TrtllmFp4Config.prepare_weights(...)`` /
``CuteDslConfig.prepare_weights(...)`` / ``TrtllmBf16Config.prepare_weights(...)``
static helpers (see ``api.py``).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

# Module-level permute-index cache.  Permute indices depend only on weight
# dims, so the cache is safe to reuse across shapes and calls.
_TRTLLM_PERMUTE_CACHE: dict = {}


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


def prepare_trtllm_bf16_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
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
    # Honor the documented device target (no-op if already resident).
    w1_bf16 = w1_bf16.to(device)
    w2_bf16 = w2_bf16.to(device)
    if permute_cache is None:
        permute_cache = _TRTLLM_PERMUTE_CACHE

    assert w1_bf16.shape == (num_local_experts, 2 * intermediate_size, hidden_size)
    assert w2_bf16.shape == (num_local_experts, hidden_size, intermediate_size)

    epilogue_tile_m = 128  # TRTLLM kernel-internal constant
    block_k = 128

    w1_views, w2_views = [], []
    for i in range(num_local_experts):
        w1_u8 = w1_bf16[i].view(torch.uint8)
        p1 = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache, w1_u8, epilogue_tile_m, is_gated_act_gemm=True
        )
        w1_views.append(
            convert_to_block_layout(w1_u8[p1.to(device)].contiguous(), block_k)
        )

        w2_u8 = w2_bf16[i].view(torch.uint8)
        p2 = get_w2_permute_indices_with_cache(permute_cache, w2_u8, epilogue_tile_m)
        w2_views.append(
            convert_to_block_layout(w2_u8[p2.to(device)].contiguous(), block_k)
        )

    return {
        "gemm1_weights": torch.stack(w1_views).view(torch.bfloat16),
        "gemm2_weights": torch.stack(w2_views).view(torch.bfloat16),
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
