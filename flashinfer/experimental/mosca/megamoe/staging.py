"""
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

Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek, MIT licensed;
see LICENSE.DeepGEMM at the root of this package. Source:
``deep_gemm/utils/math.py`` of the upstream revision (../VENDOR.md).
"""

# Input staging of the experimental MOSCA MegaMoE backend: BF16 tokens ->
# per-32 UE8M0 FP8 (E4M3) in the symmetric buffer, plus the routing.
#
# Same bytes as the stable DeepGEMM backend's staging: the fused single-launch
# CuTe DSL ``mxfp8_e4m3`` stage of the ``cutedsl_megamoe`` drop when it accepts
# the batch, else a torch port of DeepGEMM's
# ``per_token_cast_to_fp8(use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)``.
# Unlike the stable backend, the fallback never imports ``deep_gemm``.
#
# One known difference: the fused stage has no ``amax >= 1e-4`` floor and no
# exponent clamp, which DeepGEMM's cast and the fallback below both apply. For a
# 32-element block whose amax is below 1e-4 the fused bytes therefore differ
# from DeepGEMM's; they still equal the stable FlashInfer backend's, which runs
# the same fused stage.

from __future__ import annotations

import torch

from .weights import ceil_to_ue8m0

GRAN_K = 32


def _per_token_cast_to_fp8_packed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    if n % (GRAN_K * 4) != 0:
        raise ValueError("hidden_size must be a multiple of 128")
    x_view = x.view(m, n // GRAN_K, GRAN_K)
    x_amax = x_view.abs().float().amax(dim=2).view(m, n // GRAN_K).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_fp8 = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n)
    packed_sf = (sf.view(torch.int) >> 23).to(torch.uint8).view(torch.int)
    return x_fp8, packed_sf


def stage_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    """Quantize and copy ``n`` live rows into the buffer views (``x[:n]`` etc. are written)."""
    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return
    from ....moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        fused_quant_stage,
        fused_quant_stage_supported,
    )

    if fused_quant_stage_supported(hidden_states, quant_type="mxfp8_e4m3"):
        fused_quant_stage(
            hidden_states,
            topk_ids,
            topk_weights,
            x.view(torch.float8_e4m3fn),
            x_sf.view(torch.float8_e8m0fnu),
            topk_idx,
            topk_weights_out,
            quant_type="mxfp8_e4m3",
        )
        return
    x_q, x_sf_q = _per_token_cast_to_fp8_packed(hidden_states)
    x[:num_tokens].copy_(x_q)
    x_sf[:num_tokens].copy_(x_sf_q)
    topk_idx[:num_tokens].copy_(topk_ids.to(torch.int64))
    topk_weights_out[:num_tokens].copy_(topk_weights)


__all__ = ["stage_inputs"]
