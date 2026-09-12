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
from typing import Any, Callable, Sequence, Tuple

import torch

from ...autotuner import ConstraintSpec, DynamicTensorSpec, TuningConfig
from ...autotuner.initializers import (
    autotuner_initializer_empty,
    autotuner_initializer_ones,
    autotuner_initializer_rand,
    autotuner_initializer_randn,
    autotuner_initializer_zeros,
)
from ...tllm_enums import Fp8QuantizationType, SfLayout
from ..utils import (
    get_hybrid_num_tokens_buckets,
    make_hybrid_bucket_mapper,
    make_random_topk_ids,
)
from .inputs import MoeRunnerInputs
from .validation import SUPPORTED_MOE_ACT_SF_LAYOUT


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
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        expert_ids = make_random_topk_ids(
            num_experts=num_experts,
            num_tokens=math.prod(shapes[:-1]),
            top_k=shapes[-1],
            device=device,
            generator=generator,
        ).view(shapes)
        if not packed:
            return expert_ids
        expert_weights = torch.ones(shapes, dtype=torch.bfloat16, device=device).view(
            torch.int16
        )
        return (expert_ids << 16) | expert_weights

    return _init


def make_repeating_tensor_initializer(
    source: torch.Tensor,
    *,
    num_experts: int | None = None,
    packed: bool = True,
) -> Callable:
    """Initialize tuning routes from runtime data without amplifying imbalance.

    Preserve the caller's exact route rows when the profile is no larger than
    the runtime source. If a dynamic profile is larger, literal repetition can
    turn a one-token route into a pathological all-tokens-to-four-experts
    workload. When ``num_experts`` is provided, expanded profiles instead use
    the deterministic realistic sampler shared by all backends.
    """
    if source.ndim < 1 or source.numel() == 0:
        raise ValueError("source must be a non-empty tensor")

    source = source.detach()
    source_width = source.shape[-1]
    source_rows = source.reshape(-1, source_width)

    def _initializer(shapes, dtype, device):
        if len(shapes) < 1 or shapes[-1] != source_width:
            raise ValueError(
                f"Cannot initialize shape {tuple(shapes)} from source shape "
                f"{tuple(source.shape)}"
            )
        target_rows = math.prod(shapes[:-1])
        if target_rows == 0:
            return torch.empty(shapes, dtype=dtype, device=device)
        if num_experts is not None and target_rows > source_rows.shape[0]:
            return moe_topk_ids_init(num_experts, packed=packed)(shapes, dtype, device)
        repeats = (target_rows + source_rows.shape[0] - 1) // source_rows.shape[0]
        return (
            source_rows.to(device=device, dtype=dtype)
            .repeat(repeats, 1)[:target_rows]
            .reshape(shapes)
        )

    return _initializer


@functools.cache
def _make_flat_act_sf_numel_inferrer(
    hidden_states_idx: int, sf_per_token: int
) -> Callable[[Sequence[Sequence[int]]], int]:
    """ConstraintSpec callback for a flat (1-D) activation scale buffer.

    A flat scale holds ``num_tokens * sf_per_token`` elements, so its single
    dim is not ``num_tokens``.  The autotuner writes a bucket's raw token
    count verbatim into a dynamic dim (see ``_generate_optimization_profiles``),
    which would under-allocate the buffer by a factor of ``sf_per_token`` and
    make the kernel read out of bounds while profiling.  A ConstraintSpec
    instead derives the extent from the profiled ``hidden_states`` dim 0, the
    same approach ``runners.py`` uses for the swizzled MXFP8 activation scale.

    Cached so equal arguments yield the same callable: ConstraintSpec is hashed
    into AutoTuner._find_nearest_profile's lru_cache key, and a fresh closure
    per inference call would cause unbounded cache growth.
    """

    def infer_shape(shapes: Sequence[Sequence[int]]) -> int:
        return shapes[hidden_states_idx][0] * sf_per_token

    return infer_shape


def make_moe_tuning_config(
    moe_inputs: MoeRunnerInputs,
    *,
    num_experts: int,
    hidden_size: int,
    fp8_quantization_type: Fp8QuantizationType,
    init_packed_topk_ids: Callable | None,
    tune_max_num_tokens: int = 8192,
    act_sf_layout: SfLayout = SUPPORTED_MOE_ACT_SF_LAYOUT,
    **kwargs: Any,
) -> TuningConfig:
    """Build a TuningConfig for a MoE runner instance.

    ``act_sf_layout`` is the *already resolved* layout of a block-scale
    ``hidden_states_scale``; resolve it in the caller with
    :func:`~flashinfer.fused_moe.shared.validation.resolve_moe_act_sf_layout`
    so its DeprecationWarning is attributed to the user's call site.
    """

    spec = {
        "output": autotuner_initializer_empty,
        "hidden_states": autotuner_initializer_randn,
    }
    if moe_inputs.routing_logits is not None:
        spec["routing_logits"] = autotuner_initializer_rand
    if moe_inputs.topk_ids is not None:
        # Empty routed placeholders remain dynamic inputs so their historical
        # bucketed cache-key shape is preserved. They carry no route payload.
        spec["topk_ids"] = init_packed_topk_ids or autotuner_initializer_empty
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

    num_tokens = moe_inputs.hidden_states.shape[0]

    # A flat (1-D) activation scale — the linear layout produced by
    # mxfp8_quantize(..., is_sf_swizzled_layout=False), and what
    # TensorRT-LLM passes through — packs num_tokens * sf_per_token
    # elements into a single dimension, so no dim of it equals
    # num_tokens.  Resizing it as a dynamic dim would shrink it to the
    # raw bucket count; drive it with a ConstraintSpec that scales by
    # sf_per_token instead.  The C++ launcher accepts any rank here (it
    # derives the SF vector size from numel alone).
    constraint_specs: Tuple[ConstraintSpec, ...] = ()
    scale = moe_inputs.hidden_states_scale
    flat_scale = (
        scale is not None
        and fp8_quantization_type != Fp8QuantizationType.DeepSeekFp8
        and scale.dim() == 1
    )
    if flat_scale:
        # Hoisted: this runs on every op call (_make_tuning_config is not
        # memoized), so avoid repeating the numel() FFI hop three times.
        _sf_numel = scale.numel()
        if num_tokens <= 0 or _sf_numel % num_tokens != 0:
            # Not an assert: these validate caller input, and `python -O`
            # strips asserts -- which would let a malformed flat scale
            # through and let the floor division below derive an
            # undersized profiling extent.
            raise ValueError(
                f"flat hidden_states_scale numel {_sf_numel} is not a "
                f"multiple of num_tokens={num_tokens}"
            )
        # Validate the buffer against the DECLARED layout.
        # ``act_sf_layout`` is linear here — the caller either said so or
        # the deprecated implicit path inferred it, and
        # resolve_moe_act_sf_layout() rejects every other value up
        # front — so the expected extent is exact: a linear scale holds
        # num_tokens * hidden_size // sf_vec_size elements, and the C++
        # launcher recovers sf_vec_size as
        # num_tokens * hidden_size / numel, accepting only 16 (NvFp4) or
        # 32 (Mx*).  A buffer that implies anything else is malformed
        # for the declared layout and fails here with a clear message
        # instead of deriving a bogus profiling stride.
        #
        # This validates against the declaration; it is not a layout
        # *detector*, and it does not need to be.  As measured in #3455,
        # a 128x4-swizzled buffer has exactly the linear numel whenever
        # num_tokens % 128 == 0, so it would slip past any numel-based
        # test — which is precisely why the layout is declared rather
        # than guessed.  A caller that declares swizzled never reaches
        # this code (NotImplementedError in the resolver); one that stays
        # on the deprecated implicit path gets the DeprecationWarning
        # telling it to declare.  Either way sf_per_token is only used to
        # SIZE a profiling buffer, never to interpret data, so an
        # indistinguishable swizzled buffer sizes correctly and a
        # distinguishable one (row/column padded, hence strictly larger)
        # either trips this check or over-sizes the buffer, which is
        # safe.
        _sf_per_token = _sf_numel // num_tokens
        if not (
            _sf_per_token > 0
            and hidden_size % _sf_per_token == 0
            and hidden_size // _sf_per_token in (16, 32)
        ):
            raise ValueError(
                f"flat hidden_states_scale numel {_sf_numel} implies "
                f"{_sf_per_token} scales/token for hidden_size="
                f"{hidden_size}, i.e. an SF vector size of "
                f"{hidden_size / _sf_per_token if _sf_per_token else 'inf'}; "
                "which is not a supported SF vector size (16 for NvFp4, 32 "
                f"for Mx*). {act_sf_layout!r} expects "
                "num_tokens * hidden_size // sf_vec_size elements, e.g. from "
                "mxfp8_quantize(..., is_sf_swizzled_layout=False)."
            )
        constraint_specs = (
            ConstraintSpec(
                MoeRunnerInputs.idx("hidden_states_scale"),
                0,
                _make_flat_act_sf_numel_inferrer(
                    MoeRunnerInputs.idx("hidden_states"),
                    _sf_per_token,
                ),
            ),
        )

    def _dynamic_dim(name: str) -> int:
        if name == "hidden_states_scale":
            # DeepSeekFp8 uses [hidden_size//128, num_tokens]; all others
            # (MxFp8, fp4, …) use [num_tokens, ...] when 2-D.  The flat
            # 1-D layout never reaches here — it is filtered out above
            # and handled by a ConstraintSpec.
            t = moe_inputs.hidden_states_scale
            if fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                assert t.shape == (hidden_size // 128, num_tokens), (
                    f"hidden_states_scale shape {tuple(t.shape)} does not match "
                    f"expected DeepSeekFp8 layout "
                    f"(hidden_size//128={hidden_size // 128}, num_tokens={num_tokens})"
                )
                return 1
            assert t.dim() >= 1 and t.shape[0] == num_tokens, (
                f"hidden_states_scale shape {tuple(t.shape)} does not match "
                f"expected layout (num_tokens={num_tokens}, ...) or flat "
                f"(num_tokens * sf_per_token,)"
            )
            return 0
        return MoeRunnerInputs._DYNAMIC_DIM[name]

    # The constrained flat scale is excluded from the dynamic spec but keeps
    # its initializer: a ConstraintSpec also marks the dim dynamic, so the
    # profiler still synthesizes the tensor at the derived size.
    dynamic_inputs = tuple(
        (idx, name)
        for idx, name, _ in sorted_inputs
        if not (flat_scale and name == "hidden_states_scale")
    )
    dynamic_input_idx = tuple(idx for idx, _ in dynamic_inputs)
    dim_idx = tuple(_dynamic_dim(name) for _, name in dynamic_inputs)
    tensor_initializers = tuple((idx, init) for idx, _, init in sorted_inputs)

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                dynamic_input_idx,
                dim_idx,
                get_hybrid_num_tokens_buckets(tune_max_num_tokens, 1),
                make_hybrid_bucket_mapper(tune_max_num_tokens),
            ),
        ),
        constraint_specs=constraint_specs,
        tensor_initializers=tensor_initializers,
        **kwargs,
    )
