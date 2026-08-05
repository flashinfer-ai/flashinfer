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

"""Shared MoE input containers and output helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import torch


# Routing input modes for FusedMoE launcher
# Please keep this in sync with the counterpart defined in csrc/trtllm_fused_moe_kernel_launcher.cu
class RoutingInputMode(IntEnum):
    # Mode 1: Compute routing from logits
    # - Input: routing_logits tensor provided
    # - topk_ids: OUTPUT buffer for computed expert indices
    # - topk_weights: OUTPUT buffer for computed weights
    FromLogits = 0
    # Mode 2: Pre-computed routing with packed format
    # - Input: topk_ids contains packed ``(expert_id << 16) | weight`` (high
    #   16 bits = int16 expert id, low 16 bits = float16/bfloat16 weight, see
    #   PackedScoreIdx in include/flashinfer/trtllm/fused_moe/RoutingKernel.h)
    # - topk_ids: INPUT with packed values
    # - topk_weights: OUTPUT buffer for extracted weights
    PackedPrecomputed = 1
    # Mode 3: Pre-computed routing with separate tensors
    # - Input: separate topk_ids (expert indices) and topk_weights (routing weights)
    # - topk_ids: INPUT - pre-computed expert indices
    # - topk_weights: INPUT - pre-computed routing weights
    UnpackedPrecomputed = 2


@dataclass
class MoeRunnerInputs:
    """MoERunner inputs.

    Field order defines the flat-list index used by the autotuner.
    """

    output: torch.Tensor
    routing_logits: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    expert_weights: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    gemm1_lora_delta: Optional[torch.Tensor]
    per_token_scale: Optional[torch.Tensor]

    _FIELDS = (
        "output",
        "routing_logits",
        "topk_ids",
        "expert_weights",
        "hidden_states",
        "hidden_states_scale",
        "gemm1_lora_delta",
        "per_token_scale",
    )

    # Index of the dynamic dimension for each field.
    # hidden_states_scale is excluded: its layout differs by op (fp8 DeepSeekFp8
    # uses [hidden_size//128, num_tokens] while fp4/MxFp8 uses [num_tokens, ...]),
    # so make_moe_tuning_config infers it from the actual tensor at runtime.
    _DYNAMIC_DIM = {
        "output": 0,
        "routing_logits": 0,
        "topk_ids": 0,
        "expert_weights": 0,
        "hidden_states": 0,
        "gemm1_lora_delta": 0,
        "per_token_scale": 0,
    }

    def to_list(self) -> List[Optional[torch.Tensor]]:
        return [getattr(self, name) for name in MoeRunnerInputs._FIELDS]

    @classmethod
    def from_list(cls, lst: List) -> "MoeRunnerInputs":
        return cls(**{name: lst[i] for i, name in enumerate(cls._FIELDS)})

    @classmethod
    def idx(cls, name: str) -> int:
        return cls._FIELDS.index(name)


# Backward-compatible alias: this class was previously named ``MoEInputs``.
MoEInputs = MoeRunnerInputs


def alloc_trtllm_moe_output(
    num_tokens: int,
    hidden_size: int,
    do_finalize: bool,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Allocate the finalized-output buffer for a trtllm-gen MoE op."""
    return torch.empty(
        num_tokens, hidden_size if do_finalize else 0, dtype=dtype, device=device
    )


def unpack_trtllm_moe_output(
    intermediate_output,
    output: torch.Tensor,
    do_finalize: bool,
    gemm1_lora_delta: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """Translate the ``Array<Tensor>`` returned by ``FusedMoeLauncher::run``."""
    if do_finalize and gemm1_lora_delta is None:
        return [output]
    elif do_finalize and gemm1_lora_delta is not None:
        return [
            output,
            torch.from_dlpack(intermediate_output[1]),
            torch.from_dlpack(intermediate_output[2]),
        ]
    elif not do_finalize and gemm1_lora_delta is None:
        return [
            torch.from_dlpack(intermediate_output[0]),
            torch.from_dlpack(intermediate_output[1]),
            torch.from_dlpack(intermediate_output[2]),
        ]
    else:
        return [
            torch.from_dlpack(intermediate_output[0]),
            torch.from_dlpack(intermediate_output[1]),
            torch.from_dlpack(intermediate_output[2]),
            torch.from_dlpack(intermediate_output[3]),
        ]
