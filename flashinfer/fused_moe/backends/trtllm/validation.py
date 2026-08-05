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

"""Validation helpers for TRT-LLM BF16 MoE."""

from __future__ import annotations

from typing import Optional

import torch

from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import check_shape_dtype_device


def validate_bf16_gemm1_activation_params(
    activation_type: int,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    local_num_experts: int,
    device: torch.device,
) -> None:
    if gemm1_alpha is None and gemm1_beta is None and gemm1_clamp_limit is None:
        return
    if int(activation_type) != int(ActivationType.Swiglu):
        raise ValueError(
            "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported "
            "for ActivationType.Swiglu."
        )
    for name, tensor in (
        ("gemm1_alpha", gemm1_alpha),
        ("gemm1_beta", gemm1_beta),
        ("gemm1_clamp_limit", gemm1_clamp_limit),
    ):
        if tensor is not None:
            check_shape_dtype_device(
                tensor,
                (local_num_experts,),
                torch.float32,
                device,
                name,
            )
