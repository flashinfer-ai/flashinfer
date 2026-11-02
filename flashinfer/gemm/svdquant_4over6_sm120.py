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

SVDQuant linear with dynamic NVFP4 4over6 activation quantization on SM12x:
``y = alpha * (x_q @ wᵀ) + (x @ lora_downᵀ) @ lora_upᵀ [+ bias]``.
"""

from typing import Literal, Optional

import torch

from ..api_logging import flashinfer_api
from ..quantization.nvfp4_4over6_sm120 import (
    DEFAULT_NVFP4_4OVER6_SM120,
    nvfp4_4over6_dynamic_quantize_lora_down_sm120,
)
from ..quantization.nvfp4_quantization_utils import NVFP44Over6Config
from .gemm_svdquant import mm_nvfp4_svdquant


@flashinfer_api
def nvfp4_4over6_svdquant_linear_sm120(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_sf: torch.Tensor,
    weight_global_scale: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    nvfp4_4over6: NVFP44Over6Config = DEFAULT_NVFP4_4OVER6_SM120,
    rcp_quant_range: Optional[float] = None,
    backend: Literal["cute-dsl", "cute-dsl-unfused", "cutlass-sm120", "auto"] = "auto",
    enable_pdl: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SVDQuant linear on SM12x with a dynamic per-tensor NVFP4 4over6 activation scale.

    Computes ``y = alpha * (x_q @ wᵀ) + (x @ lora_downᵀ) @ lora_upᵀ [+ bias]``, where ``x_q`` is ``x``
    quantized with the 4over6 recipe under the global scale ``1 / deq``, ``deq = max(amax(x), 1e-12) / (6 *
    e4m3_max)``, and ``alpha = deq * weight_global_scale``. Two launches (three when the LoRA-down does not
    fit the quantization launch):

    1. :func:`flashinfer.quantization.nvfp4_4over6_dynamic_quantize_lora_down_sm120`: ``x_q``, its scales,
       ``alpha`` and ``down = bf16((x @ lora_downᵀ) / alpha)``; ``x_q`` and the scales are bit-identical to
       ``fp4_quantize(x, 1 / deq, nvfp4_4over6=nvfp4_4over6)``.
    2. :func:`mm_nvfp4_svdquant`: ``alpha * (x_q @ wᵀ + down @ lora_upᵀ) [+ bias]``.

    Unlike :func:`svdquant_linear`, the activation scale is computed on the device every call, so the
    LoRA-down is divided by the dynamic ``alpha`` instead of pre-dividing ``lora_up`` offline, and there is no
    smoothing scale (fold one into ``lora_down`` and the input if needed).

    Parameters
    ----------
    x : torch.Tensor
        ``[..., K]`` bfloat16 activation, ``K`` a multiple of 32.
    weight_fp4 : torch.Tensor
        ``[N, K / 2]`` uint8 NVFP4 residual weight (two e2m1 values per byte), ``N`` a multiple of 32.
    weight_sf : torch.Tensor
        uint8 UE4M3 weight block scales in the 128x4 swizzled layout.
    weight_global_scale : torch.Tensor
        ``[1]`` float32 weight decode scale (the reciprocal of the weight's encode scale); it is folded into
        ``alpha``.
    lora_down : torch.Tensor
        ``[rank, K]`` bfloat16 down projection, ``rank`` a multiple of 32.
    lora_up : torch.Tensor
        ``[N, rank]`` bfloat16 up projection, not divided by ``alpha``.
    bias : torch.Tensor, optional
        ``[N]`` bfloat16.
    nvfp4_4over6 : NVFP44Over6Config
        Activation recipe; ``err_mode="MSE"`` with exact error. Defaults to
        ``NVFP44Over6Config(e4m3_max=256, err_mode="MSE")``.
    rcp_quant_range : float, optional
        FP32 ``1 / (6 * e4m3_max)``; see :func:`flashinfer.quantization.nvfp4_4over6_dynamic_quantize_sm120`.
    backend : str
        Backend of :func:`mm_nvfp4_svdquant` (``"auto"`` by default).
    enable_pdl : bool, optional
        Programmatic dependent launch. Defaults to the device default.
    out : torch.Tensor, optional
        ``[M, N]`` bfloat16 output buffer.

    Returns
    -------
    torch.Tensor
        ``[..., N]`` bfloat16.
    """
    lead, k = x.shape[:-1], x.shape[-1]
    n = weight_fp4.shape[0]
    x_q, sf, down, _, alpha = nvfp4_4over6_dynamic_quantize_lora_down_sm120(
        x.reshape(-1, k),
        lora_down,
        weight_global_scale,
        nvfp4_4over6=nvfp4_4over6,
        rcp_quant_range=rcp_quant_range,
        enable_pdl=enable_pdl,
    )
    y = mm_nvfp4_svdquant(
        x_q,
        weight_fp4,
        sf,
        weight_sf,
        alpha,
        down,
        lora_up.contiguous(),
        bias=bias,
        out=out,
        backend=backend,
        enable_pdl=enable_pdl,
    )
    return y.reshape(*lead, n)
