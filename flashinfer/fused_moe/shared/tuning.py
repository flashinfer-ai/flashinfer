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

"""Shared autotuner tuning-config helpers for MoE runners."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from ...autotuner import DynamicTensorSpec, TuningConfig
from ...tllm_enums import Fp8QuantizationType
from ..utils import get_hybrid_num_tokens_buckets, map_to_hybrid_bucket
from .inputs import MoeRunnerInputs


def _has_payload(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is not None and tensor.numel() > 0


def make_moe_tuning_config(
    moe_inputs: MoeRunnerInputs,
    *,
    num_experts: int,
    hidden_size: int,
    fp8_quantization_type: Fp8QuantizationType,
    init_packed_topk_ids: Callable,
    tune_max_num_tokens: int = 8192,
    **kwargs: Any,
) -> TuningConfig:
    """Build a TuningConfig for a MoE runner instance."""

    spec = {
        "output": lambda shapes, dtype, device: torch.empty(
            shapes, dtype=dtype, device=device
        ),
        "hidden_states": lambda shapes, dtype, device: torch.randn(
            shapes, device=device
        ).to(dtype),
    }
    if moe_inputs.routing_logits is not None:
        spec["routing_logits"] = lambda shapes, dtype, device: torch.rand(
            shapes, dtype=dtype, device=device
        )
    if _has_payload(moe_inputs.topk_ids):
        spec["topk_ids"] = init_packed_topk_ids
    if _has_payload(moe_inputs.expert_weights):
        spec["expert_weights"] = lambda shapes, dtype, device: torch.ones(
            shapes, dtype=dtype, device=device
        )
    if moe_inputs.hidden_states_scale is not None:
        spec["hidden_states_scale"] = lambda shapes, dtype, device: torch.ones(
            shapes, device=device
        ).to(dtype)
    if moe_inputs.gemm1_lora_delta is not None:
        spec["gemm1_lora_delta"] = lambda shapes, dtype, device: torch.zeros(
            shapes, dtype=dtype, device=device
        )
    if moe_inputs.per_token_scale is not None:
        spec["per_token_scale"] = lambda shapes, dtype, device: torch.ones(
            shapes, device=device
        ).to(dtype)

    sorted_inputs = sorted(
        (MoeRunnerInputs.idx(name), name, init) for name, init in spec.items()
    )
    input_idx = tuple(i for i, _, _ in sorted_inputs)

    num_tokens = moe_inputs.hidden_states.shape[0]

    def _dynamic_dim(name: str) -> int:
        if name == "hidden_states_scale":
            t = moe_inputs.hidden_states_scale
            if fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                assert t.shape == (hidden_size // 128, num_tokens), (
                    f"hidden_states_scale shape {tuple(t.shape)} does not match "
                    f"expected DeepSeekFp8 layout "
                    f"(hidden_size//128={hidden_size // 128}, num_tokens={num_tokens})"
                )
                return 1
            assert t.shape[0] == num_tokens, (
                f"hidden_states_scale shape {tuple(t.shape)} does not match "
                f"expected layout (num_tokens={num_tokens}, ...)"
            )
            return 0
        return MoeRunnerInputs._DYNAMIC_DIM[name]

    dim_idx = tuple(_dynamic_dim(name) for _, name, _ in sorted_inputs)
    initializers = [init for _, _, init in sorted_inputs]

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx,
                dim_idx,
                get_hybrid_num_tokens_buckets(tune_max_num_tokens, 1),
                lambda x: map_to_hybrid_bucket(x, tune_max_num_tokens),
                initializers,
            ),
        ),
        **kwargs,
    )
