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
from ...autotuner.initializers import (
    autotuner_initializer_empty,
    autotuner_initializer_ones,
    autotuner_initializer_rand,
    autotuner_initializer_randn,
    autotuner_initializer_zeros,
)
from ...tllm_enums import Fp8QuantizationType
from ..utils import get_hybrid_num_tokens_buckets, make_hybrid_bucket_mapper
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
        "output": autotuner_initializer_empty,
        "hidden_states": autotuner_initializer_randn,
    }
    if moe_inputs.routing_logits is not None:
        spec["routing_logits"] = autotuner_initializer_rand
    if _has_payload(moe_inputs.topk_ids):
        spec["topk_ids"] = init_packed_topk_ids
    if _has_payload(moe_inputs.expert_weights):
        spec["expert_weights"] = autotuner_initializer_ones
    if moe_inputs.hidden_states_scale is not None:
        spec["hidden_states_scale"] = autotuner_initializer_ones
    if moe_inputs.gemm1_lora_delta is not None:
        spec["gemm1_lora_delta"] = autotuner_initializer_zeros
    if moe_inputs.per_token_scale is not None:
        spec["per_token_scale"] = autotuner_initializer_ones

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
                make_hybrid_bucket_mapper(tune_max_num_tokens),
                initializers,
            ),
        ),
        **kwargs,
    )
