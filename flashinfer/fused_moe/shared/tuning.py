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

import functools
import math
from typing import Any, Callable

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
from ..utils import (
    get_hybrid_num_tokens_buckets,
    make_hybrid_bucket_mapper,
    make_random_topk_ids,
)
from .inputs import MoeRunnerInputs


@functools.cache
def moe_topk_ids_init(num_experts: int, *, packed: bool = True):
    """Return a top-k-id initializer for a given expert count.

    ``PackedPrecomputed`` profiling needs ``(expert_id << 16) | bf16(weight)``,
    while ``UnpackedPrecomputed`` profiling needs plain expert IDs. Cache the
    closure for object identity preservation in rebuilt tuning configs.
    """

    def _init(
        shapes: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        expert_ids = make_random_topk_ids(
            num_experts=num_experts,
            num_tokens=math.prod(shapes[:-1]),
            top_k=shapes[-1],
            device=device,
        ).view(shapes)
        if not packed:
            return expert_ids
        expert_weights = torch.ones(shapes, dtype=torch.bfloat16, device=device).view(
            torch.int16
        )
        return (expert_ids << 16) | expert_weights

    return _init


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
    if moe_inputs.topk_ids is not None:
        spec["topk_ids"] = init_packed_topk_ids
    if moe_inputs.expert_weights is not None:
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
    tensor_initializers = tuple((idx, init) for idx, _, init in sorted_inputs)

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx,
                dim_idx,
                get_hybrid_num_tokens_buckets(tune_max_num_tokens, 1),
                make_hybrid_bucket_mapper(tune_max_num_tokens),
            ),
        ),
        tensor_initializers=tensor_initializers,
        **kwargs,
    )
