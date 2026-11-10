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

"""Cake SM120 NVFP4 x NVFP4 (W4A4) persistent MegaMoE backend.

The same parametric device source and runtime as ``sm120_mxfp4_mxfp8_bf16_cake`` built with
``MOE_ACT_CODEC=1``: NVFP4 dispatch records (``H / 2 + H / 16`` bytes per token instead of
``H + H / 32``), both GEMMs on ``mma.sync kind::mxf4nvf4`` with E4M3 block-16 scales, the
SwiGLU output requantized to NVFP4, and the per-expert ``fc1_alpha`` / ``fc2_alpha`` /
``fc1_norm_const`` of FlashInfer's NVFP4 MegaMoE backends.  The per-expert scales sit in a
kernel scratch tail that every call rewrites (stream-ordered, CUDA-graph capturable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.registry import register_mega_kernel
from ......weights import MoEWeightPack
from ..mxfp4_mxfp8_bf16_cake.cake_backend import Sm120Mxfp4Mxfp8CakeMegaKernelBackend
from .cake_nvfp4_config import (
    Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig,
    check_expert_scale,
)
from .cake_nvfp4_staging import stage_nvfp4_inputs, validate_nvfp4_forward_inputs
from .cake_nvfp4_weights import (
    NVFP4_BLOCK,
    TransformedCakeNvfp4MegaWeights,
    preprocess_nvfp4_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

KERNEL_NAME = "sm120_nvfp4_nvfp4_bf16_cake"
_SCALE_FIELDS = ("fc1_alpha", "fc2_alpha", "fc1_norm_const")


@register_mega_kernel(KERNEL_NAME)
class Sm120Nvfp4Nvfp4CakeMegaKernelBackend(Sm120Mxfp4Mxfp8CakeMegaKernelBackend):
    ACT_CODEC = 1
    CONFIG_TYPE = Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig
    _kernel_config: Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig

    def __init__(self, config: Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig) -> None:
        super().__init__(config)
        # per-call overrides from MoEEpTensors (set while staging, consumed by the next launch)
        self._call_scales: dict[str, torch.Tensor | None] = {}

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        super().validate_init(bootstrap, fleet_params)
        local = fleet_params.num_experts // bootstrap.world_size
        for name in _SCALE_FIELDS:
            value = getattr(self._kernel_config, name)
            check_expert_scale(name, value, local)
            if value is not None and torch.cuda.is_available():
                device = torch.device("cuda", torch.cuda.current_device())
                if value.device != device:
                    raise ValueError(
                        f"{name} is on {value.device}, the layer runs on {device}"
                    )

    # ------------------------------------------------------------------ weights

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedCakeNvfp4MegaWeights:
        transformed = preprocess_nvfp4_mega_weights(
            weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            shared_expert=self._kernel_config.shared_expert,
        )
        self._transformed = transformed
        return transformed

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedCakeNvfp4MegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        if not isinstance(transformed_weights, TransformedCakeNvfp4MegaWeights):
            raise TypeError(
                f"{KERNEL_NAME} needs TransformedCakeNvfp4MegaWeights (preprocess_weights output)"
            )
        H = fleet_params.token_hidden_size
        I = self._kernel_config.intermediate_size
        L = fleet_params.num_experts // bootstrap.world_size
        expected = (
            ("w1_fp4", transformed_weights.w1_fp4, (L, 2 * I, H // 2), torch.uint8),
            (
                "w1_sfb",
                transformed_weights.w1_sfb,
                ((L + 1) * (H // 64) * 2 * I // 16, 16),
                torch.uint32,
            ),
            ("w2_fp4", transformed_weights.w2_fp4, (L, H, I // 2), torch.uint8),
            (
                "w2_sfb",
                transformed_weights.w2_sfb,
                ((L + 1) * (I // 64) * H // 16, 16),
                torch.uint32,
            ),
            ("w1_b8", transformed_weights.w1_b8, (2 * I, H), torch.uint8),
            ("w2_b8", transformed_weights.w2_b8, (H, I), torch.uint8),
        )
        for name, tensor, shape, dtype in expected:
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or not tensor.is_cuda
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous CUDA {dtype} tensor of shape {shape}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}"
                )
        if (
            self._kernel_config.shared_expert is not None
        ) != transformed_weights.has_shared_expert:
            raise ValueError(
                "the transformed weights and shared_expert disagree on the in-kernel shared expert"
            )
        self._transformed = transformed_weights

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _activation_shapes(
        max_rows: int, hidden: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (max_rows, hidden // 2), (max_rows, hidden // NVFP4_BLOCK)

    def validate_forward(
        self, t: MoEEpTensors, fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        validate_nvfp4_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )
        local = fleet_params.num_experts // self.ep_world_size
        for name in _SCALE_FIELDS:
            check_expert_scale(name, getattr(t, name, None), local)

    def _stage_activations(
        self, t: MoEEpTensors, ws: Any, *, quantize_input: bool
    ) -> int:
        self._call_scales = {name: getattr(t, name, None) for name in _SCALE_FIELDS}
        return stage_nvfp4_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            x=ws.inputs["x_fp8"],
            x_sf=ws.inputs["x_sf"],
            topk_idx_out=ws.inputs["topk_idx"],
            topk_weights_out=ws.inputs["topk_weights"],
            quantize_input=quantize_input,
            scales=t.scales,
            input_norm_const=self._kernel_config.input_norm_const,
        )

    # ------------------------------------------------------------------ launch

    def _before_launch(self, ws: Any, inst: Any, transformed_weights: Any) -> None:
        """Write [fc1_alpha | fc2_alpha | fc1_norm_const] of this layer into the kernel's scale tail.

        The tail lives in the workspace (shared by the layers on it), so every launch rewrites it on
        the stream: one device copy or fill per row, no host transfer (the tensors are CUDA, checked
        by the config and ``validate_forward``)."""

        L = ws.local_experts
        base = int(inst.constants["EXPERT_SCALE_BASE"])
        tail = inst.buffers["routing_weight_pool"][base : base + 3 * L].view(3, L)
        for row, name in enumerate(_SCALE_FIELDS):
            value = self._call_scales.get(name)
            if value is None:
                value = getattr(self._kernel_config, name)
            if value is None:
                tail[row].fill_(1.0)
            else:
                tail[row].copy_(value)


__all__ = ["KERNEL_NAME", "Sm120Nvfp4Nvfp4CakeMegaKernelBackend"]
