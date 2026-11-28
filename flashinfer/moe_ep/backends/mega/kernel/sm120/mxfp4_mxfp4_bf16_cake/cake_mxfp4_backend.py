# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake SM120 MXFP4 x MXFP4 (W4A4) persistent MegaMoE backend.

The same parametric device source, runtime, weight pack and transport as
``sm120_mxfp4_mxfp8_bf16_cake`` built with ``MOE_ACT_CODEC=2``: MXFP4 dispatch records
(``H / 2 + H / 32`` bytes per token instead of ``H + H / 32``), packed E2M1 pool rows, both GEMMs
on ``mma.sync kind::mxf4nvf4`` with UE8M0 block-32 scales against the unchanged MXFP4 weights,
and the gated-activation output requantized to MXFP4.  No global scales, so no per-call device
writes ahead of the launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import FleetParams
from ......core.kernel.registry import register_mega_kernel
from ..mxfp4_mxfp8_bf16_cake.cake_backend import Sm120Mxfp4Mxfp8CakeMegaKernelBackend
from ..nvfp4_nvfp4_bf16_cake.cake_nvfp4_staging import (
    FUSED_QUANT,
    Fp4ActivationCodec,
    stage_fp4_inputs,
    validate_fp4_forward_inputs,
)
from .cake_mxfp4_config import Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig
from .cake_mxfp4_quant import (
    MXFP4_BLOCK,
    UE8M0_SCALE_DTYPES,
    mxfp4_fused_supported,
    mxfp4_quantize_fused,
    mxfp4_quantize_torch,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

KERNEL_NAME = "sm120_mxfp4_mxfp4_bf16_cake"

MXFP4_CODEC = Fp4ActivationCodec(
    name="MXFP4",
    block=MXFP4_BLOCK,
    scale_dtypes=UE8M0_SCALE_DTYPES,
    quantize_torch=mxfp4_quantize_torch,
    quantize_fused=mxfp4_quantize_fused,
)


def stage_mxfp4_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    quantize_input: bool,
    scales: torch.Tensor | None,
    fused: bool | None = None,
) -> int:
    """Stage one step's inputs into the workspace's MXFP4 rows; returns the row count.

    With ``quantize_input`` and a BF16/FP16/FP32 CUDA activation this is ONE launch of the fused
    quantizer: the activation is read once, the e2m1 codes and UE8M0 bytes land in ``x`` /
    ``x_sf`` and the routing rows in ``topk_idx_out`` / ``topk_weights_out``.  ``fused=False``
    (or no Triton) takes the torch reference and ``copy_``."""

    use_fused = FUSED_QUANT if fused is None else fused
    if quantize_input and use_fused and mxfp4_fused_supported(hidden_states):
        mxfp4_quantize_fused(
            hidden_states,
            codes=x,
            sf=x_sf,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            topk_ids_out=topk_idx_out,
            topk_weights_out=topk_weights_out,
        )
        return int(hidden_states.shape[0])
    if quantize_input:
        codes, sf = mxfp4_quantize_torch(hidden_states)
        return stage_fp4_inputs(
            MXFP4_CODEC,
            codes,
            topk_ids,
            topk_weights,
            x=x,
            x_sf=x_sf,
            topk_idx_out=topk_idx_out,
            topk_weights_out=topk_weights_out,
            quantize_input=False,
            scales=sf,
        )
    return stage_fp4_inputs(
        MXFP4_CODEC,
        hidden_states,
        topk_ids,
        topk_weights,
        x=x,
        x_sf=x_sf,
        topk_idx_out=topk_idx_out,
        topk_weights_out=topk_weights_out,
        quantize_input=False,
        scales=scales,
    )


@register_mega_kernel(KERNEL_NAME)
class Sm120Mxfp4Mxfp4CakeMegaKernelBackend(Sm120Mxfp4Mxfp8CakeMegaKernelBackend):
    ACT_CODEC = 2
    CONFIG_TYPE = Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig
    _kernel_config: Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    @staticmethod
    def _activation_shapes(
        max_rows: int, hidden: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (max_rows, hidden // 2), (max_rows, hidden // MXFP4_BLOCK)

    def validate_forward(
        self, t: MoEEpTensors, fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        validate_fp4_forward_inputs(
            MXFP4_CODEC,
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def _stage_activations(
        self, t: MoEEpTensors, ws: Any, *, quantize_input: bool
    ) -> int:
        return stage_mxfp4_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            x=ws.inputs["x_fp8"],
            x_sf=ws.inputs["x_sf"],
            topk_idx_out=ws.inputs["topk_idx"],
            topk_weights_out=ws.inputs["topk_weights"],
            quantize_input=quantize_input,
            scales=t.scales,
        )


__all__ = [
    "KERNEL_NAME",
    "MXFP4_CODEC",
    "Sm120Mxfp4Mxfp4CakeMegaKernelBackend",
    "stage_mxfp4_inputs",
]
