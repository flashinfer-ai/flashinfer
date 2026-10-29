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

"""
Python orchestrator for the Prims-TS BF16 MoE path.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import cuda.bindings.driver as cuda_drv
import torch

from flashinfer.autotuner import (
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from flashinfer.cuda_utils import checkCudaErrors
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode
from flashinfer.fused_moe.shared.tuning import (
    make_moe_tuning_config,
    make_repeating_tensor_initializer,
    moe_topk_ids_init,
)
from flashinfer.jit.core import logger
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingMethodType,
    SfLayout,
    WeightLayout,
)
from flashinfer.utils import get_compute_capability, round_up

from .compile_cache import get_compiled_gemm, stable_config_hash
from .config_mapper import (
    map_trtllm_bf16_moe_tactic,
    map_trtllm_deepseek_fp8_moe_tactic,
    map_trtllm_fp8_per_tensor_moe_tactic,
    map_trtllm_mxfp4_bf16_moe_tactic,
    map_trtllm_mxfp4_mxfp8_moe_tactic,
    map_trtllm_mxfp8_mxfp8_moe_tactic,
    map_trtllm_nvfp4_moe_tactic,
    valid_prims_ts_bf16_moe_tactics,
    valid_prims_ts_deepseek_fp8_moe_tactics,
    valid_prims_ts_fp8_per_tensor_moe_tactics,
    valid_prims_ts_mxfp4_bf16_moe_tactics,
    valid_prims_ts_mxfp4_mxfp8_moe_tactics,
    valid_prims_ts_mxfp8_mxfp8_moe_tactics,
    valid_prims_ts_nvfp4_moe_tactics,
)
from .support import (
    is_prims_ts_bf16_supported,
    is_prims_ts_fp8_block_scale_supported,
    is_prims_ts_fp8_per_tensor_supported,
    is_prims_ts_mxfp4_bf16_supported,
    is_prims_ts_mxfp4_mxfp8_supported,
    is_prims_ts_nvfp4_supported,
)
from .tensor_adapter import (
    _block_scaled_sf_required_span,
    _fc1_out_hidden,
    _is_gated_activation,
    _validate_swap_ab_cluster_span,
    build_bf16_launch_io,
    build_fp8_block_scale_launch_io,
    build_fp8_per_tensor_launch_io,
    build_mxfp4_bf16_launch_io,
    build_mxfp4_mxfp8_launch_io,
    build_nvfp4_launch_io,
)
from .partition import (
    MoePartitionResources,
    PartitionStrategy,
    PreparedWeightLayout,
    validate_partition_shape,
    validate_partition_tactic,
)


def _sm_arch_for_tensor(tensor: torch.Tensor) -> int:
    """Return the Prims-TS architecture key for ``tensor``'s device."""
    if tensor.device.type != "cuda":
        return 100
    major, minor = get_compute_capability(tensor.device)
    return major * 10 + minor


def _partition_stream_topology(
    resources: MoePartitionResources | None,
) -> tuple[tuple[str, int, int, int, int], ...]:
    """Return stable resource-sharing and stream-alias topology for cache keys."""

    if resources is None:
        return ()
    if resources._stream_topology is not None:
        return resources._stream_topology
    topology: list[tuple[str, int, int, int, int]] = []
    stream_classes: dict[int, int] = {}
    green_context_classes: dict[int, int] = {}
    for torch_stream in resources.streams:
        raw_stream = int(torch_stream.cuda_stream)
        stream_class = stream_classes.setdefault(raw_stream, len(stream_classes))
        priority = int(getattr(torch_stream, "priority", 0))
        stream = cuda_drv.CUstream(raw_stream)
        green_ctx = checkCudaErrors(cuda_drv.cuStreamGetGreenCtx(stream))
        resource = checkCudaErrors(
            cuda_drv.cuGreenCtxGetDevResource(
                green_ctx,
                cuda_drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM,
            )
        )
        green_ctx_key = int(green_ctx)
        context_class = green_context_classes.setdefault(
            green_ctx_key, len(green_context_classes)
        )
        topology.append(
            (
                "green",
                int(resource.sm.smCount),
                context_class,
                stream_class,
                priority,
            )
        )
    resources._stream_topology = tuple(topology)
    return resources._stream_topology


def _moe_topk_ids_init_for_routing(
    num_experts: int, routing_input_mode: RoutingInputMode
):
    return moe_topk_ids_init(
        num_experts,
        packed=(routing_input_mode != RoutingInputMode.UnpackedPrecomputed),
    )


def _per_token_sf_dtype_value(tensor: torch.Tensor) -> int:
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    if tensor.dtype == torch.bfloat16:
        return int(DType.BF16)
    if tensor.dtype == torch.float16:
        return int(DType.FP16)
    if tensor.dtype == torch.float32:
        return int(DType.FP32)
    raise ValueError(f"Unsupported per-token scale dtype {tensor.dtype}")


def _merge_per_token_sf_dtype(
    current: int | None, candidate: int, *, current_name: str, candidate_name: str
) -> int:
    if current is None:
        return int(candidate)
    if int(current) != int(candidate):
        raise ValueError(f"{current_name} and {candidate_name} must use the same dtype")
    return int(current)


def _split_per_channel_weight_scale_from_kwargs(
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return (
        kwargs.get("fc1_per_channel_weight_scale"),
        kwargs.get("fc2_per_channel_weight_scale"),
    )


def _fp8_per_tensor_scale_dtype(
    *,
    fc1_per_channel_weight_scale_dtype: int | None,
    fc2_per_channel_weight_scale_dtype: int | None,
    use_routing_scales_on_input: bool,
    routing_logits: torch.Tensor | None,
) -> int:
    dtype_value = None
    if fc1_per_channel_weight_scale_dtype is not None:
        dtype_value = _merge_per_token_sf_dtype(
            dtype_value,
            int(fc1_per_channel_weight_scale_dtype),
            current_name="fc1_per_channel_weight_scale",
            candidate_name="fc1_per_channel_weight_scale",
        )
    if fc2_per_channel_weight_scale_dtype is not None:
        dtype_value = _merge_per_token_sf_dtype(
            dtype_value,
            int(fc2_per_channel_weight_scale_dtype),
            current_name="fc1_per_channel_weight_scale",
            candidate_name="fc2_per_channel_weight_scale",
        )
    if use_routing_scales_on_input:
        if routing_logits is None:
            raise ValueError(
                "routing logits are required when use_routing_scales_on_input is enabled"
            )
        dtype_value = _merge_per_token_sf_dtype(
            dtype_value,
            _per_token_sf_dtype_value(routing_logits),
            current_name="fc1_per_channel_weight_scale",
            candidate_name="routing_logits",
        )
    return int(dtype_value or 1)


def _staged_routing_mode(
    moe_inputs: MoeRunnerInputs, kwargs: dict[str, Any]
) -> RoutingInputMode:
    """Resolve staged-routing mode, preferring the explicit kwarg.

    The C++ launcher infers packed vs unpacked vs logits from tensor rank,
    not from ``routing_input_mode``. The unified runner and the FP4 / FP8
    flat ops pass the kwarg; the BF16 flat op and direct inner-runner
    callers may omit it, so reconstruct the same rule: 2-D nonempty
    ids+weights are unpacked, 2-D nonempty ids alone are packed, otherwise
    logits.
    """
    mode = kwargs.get("routing_input_mode")
    if mode is not None:
        return RoutingInputMode(int(mode))
    ids = moe_inputs.topk_ids
    weights = moe_inputs.expert_weights
    has_ids = ids is not None and ids.ndim == 2 and ids.size(0) > 0
    has_weights = weights is not None and weights.ndim == 2 and weights.size(0) > 0
    if has_ids and has_weights:
        return RoutingInputMode.UnpackedPrecomputed
    if has_ids:
        return RoutingInputMode.PackedPrecomputed
    return RoutingInputMode.FromLogits


def _staged_routing_io(
    moe_inputs: MoeRunnerInputs, kwargs: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ids/weights buffers the staged C++ launcher needs for ``mode``.

    A 2-D nonempty ``expert_weights`` tensor selects unpacked expert ids.
    Packed and FromLogits therefore pass a 1-D empty weights placeholder.
    """
    mode = _staged_routing_mode(moe_inputs, kwargs)
    hidden = moe_inputs.hidden_states
    empty_ids = hidden.new_empty((0,), dtype=torch.int32)
    empty_weights = hidden.new_empty((0,), dtype=torch.bfloat16)
    if mode is RoutingInputMode.FromLogits:
        return empty_ids, empty_weights
    if mode is RoutingInputMode.PackedPrecomputed:
        topk_ids = moe_inputs.topk_ids
        if topk_ids is None:
            raise RuntimeError("PackedPrecomputed routing requires topk_ids")
        return topk_ids, empty_weights
    topk_ids = moe_inputs.topk_ids
    expert_weights = moe_inputs.expert_weights
    if topk_ids is None or expert_weights is None:
        raise RuntimeError(
            "UnpackedPrecomputed routing requires topk_ids and expert_weights"
        )
    return topk_ids, expert_weights


def _select_expert_weights(
    moe_inputs: MoeRunnerInputs,
    routed_expert_weights: torch.Tensor | None,
    kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Pick finalize weights. Match C++ ``ndim()==2 && size(0)>0``.

    Autotune may synthesize a 1-D nonempty placeholder from a 1-D empty
    buffer; that must not be treated as caller-owned unpacked weights.
    """
    mode = _staged_routing_mode(moe_inputs, kwargs or {})
    if mode is RoutingInputMode.UnpackedPrecomputed:
        weights = moe_inputs.expert_weights
        if weights is None or weights.ndim != 2 or weights.size(0) <= 0:
            raise RuntimeError(
                "UnpackedPrecomputed routing did not provide 2-D expert weights"
            )
        return weights
    if routed_expert_weights is None:
        raise RuntimeError("routing did not return expert weights")
    return routed_expert_weights


def _torch_views_of_ffi_tensors(tensors: Any) -> list[Any]:
    """Return zero-copy Torch views for tensors nested in a TVM-FFI container.

    TVM-FFI 0.1.11+ recursively converts container elements to framework tensors.
    Preserve those objects instead of exporting DLPack again: a cached raw DLPack
    capsule is one-shot and cannot safely be consumed by repeated runner calls.
    """
    return [
        (
            None
            if tensor is None
            else (
                tensor
                if isinstance(tensor, torch.Tensor)
                else torch.from_dlpack(tensor)
            )
        )
        for tensor in tensors
    ]


def _gemm_config_flags_from_static_extras(runner) -> dict[str, bool]:
    static_extras = dict(getattr(runner, "_cache_key_static_extras", ()))
    return {
        "fc1_has_bias": bool(static_extras.get("gemm1_bias", False)),
        "fc2_has_bias": bool(static_extras.get("gemm2_bias", False)),
        "has_gemm1_alpha": bool(static_extras.get("gemm1_alpha", False)),
        "has_gemm1_beta": bool(static_extras.get("gemm1_beta", False)),
        "has_gemm1_clamp_limit": bool(static_extras.get("gemm1_clamp_limit", False)),
    }


def _gemm_config_flags_cache_key(flags: dict[str, bool]) -> tuple:
    return tuple(sorted(flags.items()))


def _gemm1_oa_io_kwargs(kwargs: dict) -> dict[str, torch.Tensor | None]:
    return {
        "gemm1_alpha": kwargs.get("gemm1_alpha"),
        "gemm1_beta": kwargs.get("gemm1_beta"),
        "gemm1_clamp_limit": kwargs.get("gemm1_clamp_limit"),
    }


def _filter_valid_moe_tactics(valid_tactics: List[Any], map_tactic) -> List[Any]:
    filtered_tactics = []
    for tactic in valid_tactics:
        try:
            pair = map_tactic(tactic)
            pair.fc1.cfg.build()
            pair.fc2.cfg.build()
        except Exception as exc:
            logger.debug(f"[Prims-TS MoE] Skipping unsupported tactic {tactic}: {exc}")
            continue
        filtered_tactics.append(tactic)
    return filtered_tactics


def _with_default_moe_tactic(valid_tactics: List[Any]) -> List[Any]:
    return [-1, *[tactic for tactic in valid_tactics if tactic != -1]]


def _concrete_tactic(pair) -> list[int]:
    return [int(pair.tile_n), int(pair.moe_config_index)]


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _routed_token_capacity(
    runner: Any,
    moe_inputs: MoeRunnerInputs,
    resolved_tactic: list[int],
    total_num_padded_tokens: torch.Tensor,
    kwargs: dict[str, Any],
) -> int:
    del total_num_padded_tokens
    if not resolved_tactic:
        raise ValueError("resolved Prims-TS MoE tactic is empty")
    tile_n = int(resolved_tactic[0])
    if tile_n <= 0:
        raise ValueError(f"Prims-TS MoE tile_N must be positive, got {tile_n}")
    num_tokens = int(moe_inputs.hidden_states.shape[0])
    num_experts = int(
        kwargs.get("local_num_experts", getattr(runner, "num_local_experts", 0))
    )
    top_k = int(getattr(runner, "top_k", 0))

    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_max_num_ctas_in_token_dim_for_moe,
    )

    token_ctas = compute_max_num_ctas_in_token_dim_for_moe(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        token_tile_size=tile_n,
        cluster_dim_in_token=1,
    )
    capacity = token_ctas * tile_n
    if capacity <= 0:
        raise ValueError("routed token capacity is empty")
    return capacity


def _nvfp4_per_token_global_scale_inv() -> float:
    # Keep this in sync with the native TRT-LLM Gen per-token NVFP4 MoE path.
    if _env_flag_enabled("FLASHINFER_NVFP4_4OVER6") and _env_flag_enabled(
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"
    ):
        return 1.0 / (256.0 * 6.0)
    return 1.0 / (448.0 * 6.0)


def _pad_mxfp8_linear_scale_for_prims(
    scale: torch.Tensor, *, num_tokens: int, hidden_size: int
) -> torch.Tensor:
    """Pad compact MXFP8 token scales to the routed LINEAR SF layout."""

    sf_cols = int(hidden_size) // 32
    padded_cols = round_up(sf_cols, 16)
    scale_u8 = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
    src = scale_u8.reshape(int(num_tokens), -1)
    if src.shape[1] < sf_cols:
        raise ValueError(
            "MXFP8 hidden_states_scale is too small: "
            f"need at least {sf_cols} scale bytes per token, got {src.shape[1]}"
        )
    if src.shape[1] == padded_cols and src.is_contiguous():
        return src

    padded = torch.empty(
        (int(num_tokens), padded_cols), dtype=torch.uint8, device=scale.device
    )
    padded.fill_(0x7F)
    padded[:, :sf_cols].copy_(src[:, :sf_cols])
    return padded


def _quantize_nvfp4_fc1_output_for_fc2(
    *,
    gemm1_output: torch.Tensor,
    gemm1_output_scale: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_tokens: int,
    top_k: int,
    intermediate_size: int,
    tile_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize routed BF16 FC1 output into the NVFP4 activation used by FC2."""

    if gemm1_output.dtype != torch.bfloat16:
        raise ValueError("per-token NVFP4 FC1 output must be bfloat16")
    if gemm1_output.shape[1] != intermediate_size:
        raise ValueError(
            "per-token NVFP4 FC1 output has unexpected hidden dimension "
            f"{gemm1_output.shape[1]}, expected {intermediate_size}"
        )

    max_padded_tokens = int(gemm1_output.shape[0])
    expanded_tokens = int(num_tokens) * int(top_k)
    if expanded_tokens > max_padded_tokens:
        raise ValueError(
            f"expanded token count {expanded_tokens} exceeds padded capacity "
            f"{max_padded_tokens}"
        )

    activation_output = torch.empty(
        (max_padded_tokens, intermediate_size // 2),
        dtype=torch.uint8,
        device=gemm1_output.device,
    )
    per_token_scale_fc2 = torch.empty(
        (max_padded_tokens,), dtype=torch.float32, device=gemm1_output.device
    )

    sf_row_tile = 128 if tile_n >= 128 else 8
    sf_rows = round_up(max_padded_tokens, sf_row_tile)
    sf_cols = round_up(intermediate_size // 16, 4)
    required_sf_values = sf_rows * sf_cols
    if gemm1_output_scale.numel() < required_sf_values:
        raise ValueError(
            "NVFP4 activation scale buffer is too small: "
            f"need {required_sf_values}, got {gemm1_output_scale.numel()}"
        )
    activation_output_scale = gemm1_output_scale[:required_sf_values].view(
        sf_rows, sf_cols
    )

    input_view = torch.as_strided(
        gemm1_output,
        (expanded_tokens, intermediate_size),
        gemm1_output.stride(),
    )
    sf_layout = (
        SfLayout.layout_128x4.value if tile_n >= 128 else SfLayout.layout_8x4.value
    )
    major, minor = get_compute_capability(gemm1_output.device)
    from flashinfer.quantization.fp4_quantization import get_fp4_quantization_module

    get_fp4_quantization_module(
        f"{major * 10 + minor}"
    ).nvfp4_quant_and_per_token_scale_out_sm100(
        input_view,
        _nvfp4_per_token_global_scale_inv(),
        activation_output,
        activation_output_scale,
        per_token_scale_fc2,
        expanded_idx_to_permuted_idx,
        sf_layout,
    )
    return activation_output, activation_output_scale, per_token_scale_fc2


@dataclass(frozen=True)
class _PartitionExecution:
    """Concrete buffers and GEMM geometry for one partition strategy.

    Tensor tuples are indexed by partition. Shared TP-N/TP-N inputs appear
    twice by identity, while TP-N/TP-K entries point at distinct local
    buffers. ``fc1_sf_offsets`` locates each partition in a shared SF-C
    layout. ``second_fc2_output`` is set only when finalization must reduce two
    full-width FC2 partials.
    """

    fc1_outputs: tuple[torch.Tensor, ...]
    fc1_scales: tuple[torch.Tensor | None, ...]
    fc1_sf_offsets: tuple[int, ...]
    fc2_outputs: tuple[torch.Tensor, ...]
    fc2_activations: tuple[torch.Tensor, ...]
    fc2_activation_scales: tuple[torch.Tensor | None, ...]
    local_fc2_width: int
    fc1_row_stride: int | None
    fc2_intermediate_size: int
    fc2_row_stride: int | None
    rejoin_after_fc1: bool
    second_fc2_output: torch.Tensor | None


class _PrimsTsMoERunnerMixin:
    _cache_key_static_extras: tuple = ()
    _partition_strategy: PartitionStrategy | None = None
    _partition_fc1_layout: PreparedWeightLayout | None = None
    _partition_fc2_layout: PreparedWeightLayout | None = None
    _partition_topology: tuple = ()
    _geometry_tactics_cache: dict = {}

    def set_cache_key_static_extras(self, **kwargs) -> None:
        fc1_per_channel_weight_scale, fc2_per_channel_weight_scale = (
            _split_per_channel_weight_scale_from_kwargs(kwargs)
        )
        fc1_scale_dtype = (
            _per_token_sf_dtype_value(fc1_per_channel_weight_scale)
            if fc1_per_channel_weight_scale is not None
            else None
        )
        fc2_scale_dtype = (
            _per_token_sf_dtype_value(fc2_per_channel_weight_scale)
            if fc2_per_channel_weight_scale is not None
            else None
        )
        per_token_sf_dtype = _fp8_per_tensor_scale_dtype(
            fc1_per_channel_weight_scale_dtype=fc1_scale_dtype,
            fc2_per_channel_weight_scale_dtype=fc2_scale_dtype,
            use_routing_scales_on_input=False,
            routing_logits=None,
        )
        enable_pdl = bool(kwargs.get("enable_pdl", False))
        self._cache_key_static_extras = (
            ("enable_pdl", enable_pdl),
            ("gemm1_bias", kwargs.get("gemm1_bias") is not None),
            ("gemm2_bias", kwargs.get("gemm2_bias") is not None),
            ("gemm1_alpha", kwargs.get("gemm1_alpha") is not None),
            ("gemm1_beta", kwargs.get("gemm1_beta") is not None),
            ("gemm1_clamp_limit", kwargs.get("gemm1_clamp_limit") is not None),
            (
                "num_fused_shared_experts",
                int(kwargs.get("num_fused_shared_experts") or 0),
            ),
            ("routing_input_mode", int(kwargs.get("routing_input_mode", 0))),
            (
                "use_routing_scales_on_input",
                bool(kwargs.get("use_routing_scales_on_input", False)),
            ),
            ("fc1_per_channel_weight_scale", fc1_per_channel_weight_scale is not None),
            ("fc2_per_channel_weight_scale", fc2_per_channel_weight_scale is not None),
            ("per_token_sf_dtype", per_token_sf_dtype),
        )

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        return (
            ("prims_ts_moe_config_version", 8),
            ("top_k", int(self.top_k)),
            ("num_local_experts", int(self.num_local_experts)),
            ("num_experts", int(self.num_experts)),
            ("hidden_size", int(self.hidden_size)),
            ("intermediate_size", int(self.intermediate_size)),
            ("dtype_act", int(self.dtype_act)),
            ("dtype_weights", int(self.dtype_weights)),
            ("fp8_quantization_type", int(self.fp8_quantization_type)),
            ("activation_type", int(self.activation_type)),
            ("weight_layout", int(self.weight_layout)),
            ("use_per_token_scaling", bool(self.use_per_token_scaling)),
            ("partition_strategy", self._partition_strategy or ""),
            ("partition_topology", self._partition_topology),
            ("partition_fc1_layout", self._partition_fc1_layout),
            ("partition_fc2_layout", self._partition_fc2_layout),
            ("per_token_scale", moe_inputs.per_token_scale is not None),
            ("gemm1_lora_delta", moe_inputs.gemm1_lora_delta is not None),
            (
                "routing_logits",
                moe_inputs.routing_logits is not None,
            ),
            (
                "expert_weights",
                moe_inputs.expert_weights is not None,
            ),
            *getattr(self, "_cache_key_static_extras", ()),
        )

    @property
    def _is_partitioned(self) -> bool:
        return self._partition_strategy is not None

    def _staged_tensor(self, kwargs: dict[str, Any], name: str) -> torch.Tensor | None:
        """Return a monolithic weight tensor, or None when shards are passed separately."""
        if self._is_partitioned:
            return None
        return kwargs[name]

    def _partition_shards(
        self, kwargs: dict[str, Any]
    ) -> tuple[
        MoePartitionResources,
        Sequence[torch.Tensor],
        Sequence[torch.Tensor],
        Sequence[torch.Tensor] | None,
        Sequence[torch.Tensor] | None,
    ]:
        if not self._is_partitioned:
            raise RuntimeError(
                "Partitioned execution requested on unpartitioned runner"
            )
        required = (
            "partition_resources",
            "partition_fc1_weights",
            "partition_fc2_weights",
        )
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise RuntimeError(
                "Partitioned execution is missing call payload: " + ", ".join(missing)
            )
        return (
            kwargs["partition_resources"],
            kwargs["partition_fc1_weights"],
            kwargs["partition_fc2_weights"],
            kwargs.get("partition_fc1_scales"),
            kwargs.get("partition_fc2_scales"),
        )

    def _validate_partition_mode(self) -> None:
        if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
            raise NotImplementedError(
                "Partitioned FP8 block-scale MoE does not support the DeepSeek "
                "activation/scale layout"
            )

    def _tactic_common(
        self, moe_inputs: MoeRunnerInputs, kwargs: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        extras = kwargs if kwargs is not None else dict(self._cache_key_static_extras)
        common = dict(
            sm_arch=_sm_arch_for_tensor(moe_inputs.hidden_states),
            activation_type=int(self.activation_type),
            num_tokens=int(moe_inputs.hidden_states.shape[0]),
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            num_experts=self.num_experts,
            weight_layout=int(extras.get("weight_layout", self.weight_layout)),
            enable_pdl=bool(extras.get("enable_pdl", False)),
        )

        def is_present(name: str) -> bool:
            value = extras.get(name)
            return value if isinstance(value, bool) else value is not None

        bias_and_oa = dict(
            fc1_has_bias=is_present("gemm1_bias"),
            fc2_has_bias=is_present("gemm2_bias"),
            has_gemm1_alpha=is_present("gemm1_alpha"),
            has_gemm1_beta=is_present("gemm1_beta"),
            has_gemm1_clamp_limit=is_present("gemm1_clamp_limit"),
        )
        return common, bias_and_oa, extras

    def _map_tactic(
        self,
        tactic,
        moe_inputs: MoeRunnerInputs,
        kwargs: dict[str, Any] | None = None,
    ):
        raise NotImplementedError

    @staticmethod
    def _partition_storage_width(logical_width: int, output: torch.Tensor, cfg) -> int:
        logical_bits = int(logical_width) * int(cfg.dtype_c_bits)
        storage_bits = int(output.element_size()) * 8
        if logical_bits % storage_bits:
            raise ValueError(
                f"logical C width {logical_width} at {cfg.dtype_c_bits} bits "
                f"does not fill {storage_bits}-bit storage elements"
            )
        return logical_bits // storage_bits

    @classmethod
    def _partition_output_view(
        cls,
        output: torch.Tensor,
        *,
        partition_id: int,
        logical_width: int,
        cfg,
    ) -> torch.Tensor:
        storage_width = cls._partition_storage_width(logical_width, output, cfg)
        start = partition_id * storage_width
        stop = start + storage_width
        if output.ndim != 2 or output.shape[1] < stop:
            raise ValueError(
                f"partitioned C storage needs at least {stop} columns, "
                f"got shape {tuple(output.shape)}"
            )
        return output[:, start:stop]

    @classmethod
    def _compact_partition_output(
        cls,
        output: torch.Tensor,
        *,
        partition_id: int,
        logical_width: int,
        cfg,
    ) -> torch.Tensor:
        storage_width = cls._partition_storage_width(logical_width, output, cfg)
        rows = int(output.shape[0])
        partition_numel = rows * storage_width
        required = 2 * partition_numel
        if output.numel() < required:
            raise ValueError(
                f"compact FC1 partitions require {required} storage elements, "
                f"got {output.numel()}"
            )
        start = partition_id * partition_numel
        return output.reshape(-1)[start : start + partition_numel].view(
            rows, storage_width
        )

    def _materialize_partition_execution(
        self,
        *,
        partition: MoePartitionResources,
        fc1_cfg,
        fc2_cfg,
        gemm1_output: torch.Tensor,
        gemm2_output: torch.Tensor,
        gemm1_output_scale: torch.Tensor | None,
        local_fc1_width: int,
        full_fc1_pitch: int,
    ) -> _PartitionExecution:
        """Resolve one strategy into the exact buffers consumed by both GEMMs."""

        partition_ids = range(2)
        if self._partition_strategy == "tp_n_tp_n":
            # FC1 writes two strided N slices. After the join, both FC2 launches
            # consume the assembled intermediate and write disjoint N slices.
            fc1_outputs = tuple(
                self._partition_output_view(
                    gemm1_output,
                    partition_id=partition_id,
                    logical_width=local_fc1_width,
                    cfg=fc1_cfg,
                )
                for partition_id in partition_ids
            )
            if gemm1_output_scale is not None and fc1_cfg.has_output_block_scaling:
                fc1_scales = (gemm1_output_scale,) * 2
                fc1_sf_offsets = (0, local_fc1_width)
            else:
                fc1_scales = (gemm1_output_scale,) * 2
                fc1_sf_offsets = (0, 0)
            fc2_outputs = tuple(
                self._partition_output_view(
                    gemm2_output,
                    partition_id=partition_id,
                    logical_width=self.hidden_size // 2,
                    cfg=fc2_cfg,
                )
                for partition_id in partition_ids
            )
            return _PartitionExecution(
                fc1_outputs=fc1_outputs,
                fc1_scales=fc1_scales,
                fc1_sf_offsets=fc1_sf_offsets,
                fc2_outputs=fc2_outputs,
                fc2_activations=(gemm1_output,) * 2,
                fc2_activation_scales=(gemm1_output_scale,) * 2,
                local_fc2_width=self.hidden_size // 2,
                fc1_row_stride=full_fc1_pitch,
                fc2_intermediate_size=self.intermediate_size,
                fc2_row_stride=self.hidden_size,
                rejoin_after_fc1=True,
                second_fc2_output=None,
            )

        if self._partition_strategy == "tp_n_tp_k":
            # Each compact FC1 N shard feeds the matching FC2 K shard on the
            # same stream. FC2 writes full-width partials for final reduction.
            fc1_outputs = tuple(
                self._compact_partition_output(
                    gemm1_output,
                    partition_id=partition_id,
                    logical_width=local_fc1_width,
                    cfg=fc1_cfg,
                )
                for partition_id in partition_ids
            )
            if gemm1_output_scale is not None and fc1_cfg.has_output_block_scaling:
                scale_span = _block_scaled_sf_required_span(
                    output_m=local_fc1_width,
                    output_n=int(gemm1_output.shape[0]),
                    pitch=local_fc1_width,
                    sf_block_size=int(fc1_cfg.output_sf_block_size_c),
                    sf_layout=int(fc1_cfg.sf_layout_c),
                )
                if gemm1_output_scale.numel() < 2 * scale_span:
                    raise ValueError(
                        "compact FC1 scale partitions do not fit in the routed "
                        "scale workspace"
                    )
                fc1_scales = tuple(
                    gemm1_output_scale.reshape(-1)[
                        partition_id * scale_span : (partition_id + 1) * scale_span
                    ]
                    for partition_id in partition_ids
                )
            else:
                fc1_scales = (None,) * 2
            # Allocation and final consumption both belong to the caller stream.
            # The join before finalize orders partition-stream writes, so the
            # caching allocator cannot recycle this buffer too early.
            fc2_outputs = (
                gemm2_output,
                torch.empty_like(gemm2_output),
            )
            return _PartitionExecution(
                fc1_outputs=fc1_outputs,
                fc1_scales=fc1_scales,
                fc1_sf_offsets=(0, 0),
                fc2_outputs=fc2_outputs,
                fc2_activations=fc1_outputs,
                fc2_activation_scales=fc1_scales,
                local_fc2_width=self.hidden_size,
                fc1_row_stride=None,
                fc2_intermediate_size=self.intermediate_size // 2,
                fc2_row_stride=None,
                rejoin_after_fc1=False,
                second_fc2_output=fc2_outputs[1],
            )

        raise ValueError(
            f"Unsupported MoE partition strategy {self._partition_strategy!r}"
        )

    def _run_partitioned_gemm_pair(
        self,
        *,
        partition: MoePartitionResources,
        partition_fc1_weights: Sequence[torch.Tensor],
        partition_fc2_weights: Sequence[torch.Tensor],
        partition_fc1_scales: Sequence[torch.Tensor] | None,
        partition_fc2_scales: Sequence[torch.Tensor] | None,
        builder,
        fc1_cfg,
        fc2_cfg,
        common_io_kwargs: dict[str, Any],
        fc1_label: str,
        fc2_label: str,
        gemm1_output: torch.Tensor,
        gemm2_output: torch.Tensor,
        gemm1_output_scale: torch.Tensor | None = None,
        fc1_extra_kwargs: dict[str, Any] | None = None,
        fc2_extra_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Launch a strategy-planned two-way tensor-parallel FC1/FC2 pair."""

        self._validate_partition_mode()
        if not _is_gated_activation(int(self.activation_type)):
            raise NotImplementedError(
                "MoE partitioning currently requires a gated activation"
            )
        if (
            common_io_kwargs.get("gemm1_bias") is not None
            or common_io_kwargs.get("gemm2_bias") is not None
        ):
            raise NotImplementedError(
                "MoE partitioning does not yet support row-sharded bias"
            )
        if not fc1_cfg.is_swap_ab or not fc2_cfg.is_swap_ab:
            raise ValueError("MoE partitioning requires swapAB Prims-TS tactics")
        if (
            self._partition_strategy is None
            or self._partition_fc1_layout is None
            or self._partition_fc2_layout is None
        ):
            raise RuntimeError("Partitioned runner is missing its strategy or layouts")
        if (partition_fc1_scales is None) != (partition_fc2_scales is None):
            raise ValueError(
                "partitioned block-scaled weights require both scale shard lists"
            )
        local_intermediate = self.intermediate_size // 2
        validate_partition_shape(
            partition,
            strategy=self._partition_strategy,
            fc1_weights=partition_fc1_weights,
            fc2_weights=partition_fc2_weights,
            fc1_scales=partition_fc1_scales,
            fc2_scales=partition_fc2_scales,
            fc1_layout=self._partition_fc1_layout,
            fc2_layout=self._partition_fc2_layout,
            num_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        )
        local_fc1_width = self._check_partition_tactic(fc1_cfg, fc2_cfg)

        caller_stream = torch.cuda.current_stream(
            common_io_kwargs["hidden_states"].device
        )
        fork_join = partition.fork_join
        if fork_join is None:
            raise ValueError("MoE partition streams must be CUDA streams")
        full_fc1_pitch = _fc1_out_hidden(
            self.intermediate_size, int(self.activation_type)
        )
        if fc1_cfg.has_gated_epilogue:
            full_fc1_pitch //= 2
        execution = self._materialize_partition_execution(
            partition=partition,
            fc1_cfg=fc1_cfg,
            fc2_cfg=fc2_cfg,
            gemm1_output=gemm1_output,
            gemm2_output=gemm2_output,
            gemm1_output_scale=gemm1_output_scale,
            local_fc1_width=local_fc1_width,
            full_fc1_pitch=full_fc1_pitch,
        )
        fc1_ios: list[dict[str, Any]] = []
        fc2_ios: list[dict[str, Any]] = []

        def partition_weights(partition_id: int) -> dict[str, torch.Tensor]:
            weights = {
                "gemm1_weights": partition_fc1_weights[partition_id],
                "gemm2_weights": partition_fc2_weights[partition_id],
            }
            if partition_fc1_scales is not None and partition_fc2_scales is not None:
                weights["gemm1_weights_scale"] = partition_fc1_scales[partition_id]
                weights["gemm2_weights_scale"] = partition_fc2_scales[partition_id]
            return weights

        for partition_id in range(2):
            fc1_output = execution.fc1_outputs[partition_id]
            fc1_scale = execution.fc1_scales[partition_id]

            stage_common = {
                **common_io_kwargs,
                **partition_weights(partition_id),
                "gemm1_output": fc1_output,
                # The FC1 builders validate both workspace tensors even though
                # FC2 output is unused at this stage. Keep the full contiguous
                # allocation here; the FC2 call below receives its real slice.
                "gemm2_output": gemm2_output,
            }
            if fc1_scale is not None:
                stage_common["gemm1_output_scale"] = fc1_scale
                if "activation_output" in stage_common:
                    stage_common["activation_output"] = fc1_output
                if "activation_output_scale" in stage_common:
                    stage_common["activation_output_scale"] = fc1_scale

            fc1_kwargs = {
                **stage_common,
                "intermediate_size": local_intermediate,
                "hidden_size": self.hidden_size,
                "c_row_stride": execution.fc1_row_stride,
                "sf_c_m_offset": execution.fc1_sf_offsets[partition_id],
                **(fc1_extra_kwargs or {}),
            }
            fc1_ios.append(builder(fc="fc1", cfg=fc1_cfg, **fc1_kwargs))

        for partition_id in range(2):
            fc2_activation = execution.fc2_activations[partition_id]
            fc2_activation_scale = execution.fc2_activation_scales[partition_id]
            stage_common = {
                **common_io_kwargs,
                **partition_weights(partition_id),
                "gemm1_output": fc2_activation,
                "gemm2_output": execution.fc2_outputs[partition_id],
            }
            if fc2_activation_scale is not None:
                stage_common["gemm1_output_scale"] = fc2_activation_scale
                if "activation_output" in stage_common:
                    stage_common["activation_output"] = fc2_activation
                if "activation_output_scale" in stage_common:
                    stage_common["activation_output_scale"] = fc2_activation_scale
            fc2_kwargs = {
                **stage_common,
                "intermediate_size": execution.fc2_intermediate_size,
                "hidden_size": execution.local_fc2_width,
                "c_row_stride": execution.fc2_row_stride,
                **(fc2_extra_kwargs or {}),
            }
            fc2_ios.append(builder(fc="fc2", cfg=fc2_cfg, **fc2_kwargs))

        compiled_fc1 = []
        compiled_fc2 = []
        for torch_stream, fc1_io, fc2_io in zip(
            partition.streams, fc1_ios, fc2_ios, strict=True
        ):
            stream = self._cuda_driver_stream(torch_stream)
            compiled_fc1.append(
                (
                    get_compiled_gemm(
                        stable_config_hash(fc1_io["cfg"]),
                        fc1_label,
                        fc1_io,
                        stream,
                    ),
                    stream,
                )
            )
            compiled_fc2.append(
                (
                    get_compiled_gemm(
                        stable_config_hash(fc2_io["cfg"]),
                        fc2_label,
                        fc2_io,
                        stream,
                    ),
                    stream,
                )
            )

        # Fork/join uses ordinary CUDA events. They are full dependencies in
        # eager execution. During graph capture, however, a PDL-attributed
        # consumer launch turns incoming kernel dependencies carried through
        # these event waits into programmatic edges. FC1 therefore retains its
        # grid-dependency wait before reading routing outputs.
        fork_join.fork(caller_stream)
        try:
            for torch_stream, io, (fn, stream) in zip(
                partition.streams, fc1_ios, compiled_fc1, strict=True
            ):
                with torch.cuda.stream(torch_stream):
                    fn(*self._launch_args(io, stream))
            if execution.rejoin_after_fc1:
                # TP-N/TP-N makes both FC2 kernels depend on both FC1 shards.
                # Captured PDL launches form all-to-all programmatic edges, so
                # each FC2 waits for both FC1 grids before reading the combined
                # activation.
                fork_join.join(caller_stream)
                fork_join.fork(caller_stream)
            # Without the rejoin, TP-N/TP-K keeps each FC1 output (and optional
            # scale) on its owner stream. Each FC2 waits only for its own FC1.
            # The final join orders output/partial reuse eagerly and supplies
            # both programmatic parents to a captured PDL finalize launch.
            for torch_stream, io, (fn, stream) in zip(
                partition.streams, fc2_ios, compiled_fc2, strict=True
            ):
                with torch.cuda.stream(torch_stream):
                    fn(*self._launch_args(io, stream))
        finally:
            fork_join.join(caller_stream)
        return gemm2_output, execution.second_fc2_output

    def _run_partitioned(
        self,
        *,
        builder,
        fc1_cfg,
        fc2_cfg,
        common_io_kwargs: dict[str, Any],
        fc1_label: str,
        fc2_label: str,
        gemm1_output: torch.Tensor,
        gemm2_output: torch.Tensor,
        output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        total_num_padded_tokens: torch.Tensor,
        num_tokens: int,
        kwargs: dict[str, Any],
        gemm1_output_scale: torch.Tensor | None = None,
        fc1_extra_kwargs: dict[str, Any] | None = None,
        fc2_extra_kwargs: dict[str, Any] | None = None,
        use_routing_scales_on_input: bool = False,
    ):
        (
            partition,
            partition_fc1_weights,
            partition_fc2_weights,
            partition_fc1_scales,
            partition_fc2_scales,
        ) = self._partition_shards(kwargs)
        assert partition.fork_join is not None
        with partition.launch_guard():
            gemm2_output, second_gemm2_output = self._run_partitioned_gemm_pair(
                partition=partition,
                partition_fc1_weights=partition_fc1_weights,
                partition_fc2_weights=partition_fc2_weights,
                partition_fc1_scales=partition_fc1_scales,
                partition_fc2_scales=partition_fc2_scales,
                builder=builder,
                fc1_cfg=fc1_cfg,
                fc2_cfg=fc2_cfg,
                common_io_kwargs=common_io_kwargs,
                fc1_label=fc1_label,
                fc2_label=fc2_label,
                gemm1_output=gemm1_output,
                gemm1_output_scale=gemm1_output_scale,
                gemm2_output=gemm2_output,
                fc1_extra_kwargs=fc1_extra_kwargs,
                fc2_extra_kwargs=fc2_extra_kwargs,
            )
            return self._finish_partitioned(
                gemm2_output=gemm2_output,
                second_gemm2_output=second_gemm2_output,
                output=output,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                total_num_padded_tokens=total_num_padded_tokens,
                num_tokens=num_tokens,
                num_experts=kwargs["num_experts"],
                do_finalize=kwargs["do_finalize"],
                enable_pdl=kwargs["enable_pdl"],
                use_routing_scales_on_input=use_routing_scales_on_input,
            )

    @staticmethod
    def _cuda_driver_stream(torch_stream):
        return cuda_drv.CUstream(torch_stream.cuda_stream)

    def _finish_partitioned(
        self,
        *,
        gemm2_output: torch.Tensor,
        second_gemm2_output: torch.Tensor | None,
        output: torch.Tensor,
        expert_weights: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        total_num_padded_tokens: torch.Tensor,
        num_tokens: int,
        num_experts: int,
        do_finalize: bool,
        enable_pdl: bool,
        use_routing_scales_on_input: bool = False,
    ):
        if do_finalize:
            # The join above fully orders both partition outputs eagerly.
            # Under graph capture, a PDL finalize launch instead receives one
            # programmatic parent from each FC2 grid and waits before reading
            # either output.
            self.moe_op.trtllm_moe_run_finalize(
                gemm2_output,
                output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                num_experts,
                self.top_k,
                self.hidden_size,
                enable_pdl,
                use_routing_scales_on_input,
                second_gemm2_output,
            )
            return []
        if second_gemm2_output is not None:
            torch.add(gemm2_output, second_gemm2_output, out=gemm2_output)
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    def _check_partition_tactic(self, fc1_cfg, fc2_cfg) -> int:
        """Validate built FC1/FC2 configs against the partition; return local FC1 width."""
        local_fc1_width = _fc1_out_hidden(
            self.intermediate_size // 2, int(self.activation_type)
        )
        if fc1_cfg.has_gated_epilogue:
            local_fc1_width //= 2
        validate_partition_tactic(
            strategy=self._partition_strategy,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            fc1_tile_m=int(getattr(fc1_cfg, "tile_m", 1)),
            fc2_tile_m=int(getattr(fc2_cfg, "tile_m", 1)),
            fc1_epilogue_tile_m=int(getattr(fc1_cfg, "epi_tile_m", 128)),
            fc2_epilogue_tile_m=int(getattr(fc2_cfg, "epi_tile_m", 128)),
            fc1_is_swap_ab=bool(fc1_cfg.is_swap_ab),
            fc2_is_swap_ab=bool(fc2_cfg.is_swap_ab),
            fc1_partition_width=local_fc1_width,
            fc1_output_sf_block_size=(
                int(fc1_cfg.output_sf_block_size_c)
                if fc1_cfg.has_output_block_scaling
                else None
            ),
        )
        return local_fc1_width

    def _check_cga_tactic(self, fc1_cfg, fc2_cfg) -> None:
        local_intermediate = (
            self.intermediate_size // 2
            if self._is_partitioned
            else self.intermediate_size
        )
        fc1_width = _fc1_out_hidden(local_intermediate, int(self.activation_type))
        fc2_width = (
            self.hidden_size // 2
            if self._partition_strategy == "tp_n_tp_n"
            else self.hidden_size
        )
        _validate_swap_ab_cluster_span(fc1_cfg, fc1_width)
        _validate_swap_ab_cluster_span(fc2_cfg, fc2_width)

    def _filter_partition_geometry_tactics(
        self, tactics: List[Any], inputs: List[torch.Tensor], cache_key: tuple
    ) -> List[Any]:
        if self._is_partitioned:
            self._validate_partition_mode()
        key = (
            type(self),
            cache_key,
            self._partition_strategy,
            self._cache_key_static_extras,
        )
        cached = _PrimsTsMoERunnerMixin._geometry_tactics_cache.get(key)
        if cached is not None:
            return cached
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        filtered = []
        dropped = 0
        for tactic in tactics:
            try:
                pair = self._map_tactic(tactic, moe_inputs)
                fc1_cfg = pair.fc1.cfg.build()
                fc2_cfg = pair.fc2.cfg.build()
                self._check_cga_tactic(fc1_cfg, fc2_cfg)
                if self._is_partitioned:
                    self._check_partition_tactic(fc1_cfg, fc2_cfg)
            except ValueError as exc:
                dropped += 1
                logger.debug(
                    "[Prims-TS MoE] Skipping geometry-incompatible tactic "
                    f"{tactic}: {exc}"
                )
                continue
            filtered.append(tactic)
        if dropped:
            logger.warning(
                "[Prims-TS MoE] Dropped "
                f"{dropped}/{len(tactics)} geometry-incompatible tactic(s)"
            )
        if not filtered:
            raise RuntimeError(
                "No Prims-TS tactics support the requested geometry for "
                f"I={self.intermediate_size}, H={self.hidden_size}"
            )
        _PrimsTsMoERunnerMixin._geometry_tactics_cache[key] = filtered
        return filtered

    def precompile_tactics(
        self,
        inputs: List[torch.Tensor],
        tactics: List[Any],
        profile: OptimizationProfile,
        **kwargs,
    ) -> bool:
        del profile
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        hidden_states = moe_inputs.hidden_states

        def _precompile_one(tactic: Any) -> None:
            try:
                compile_only = getattr(self, "_precompile_tactic_compile_only", None)
                if compile_only is not None:
                    compile_only(inputs, tactic, **kwargs)
                    return
                self.forward(inputs, tactic=tactic, **kwargs)
                torch.cuda.current_stream(device=hidden_states.device).synchronize()
            except Exception as exc:
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize(hidden_states.device)
                with contextlib.suppress(Exception):
                    torch.cuda.cudart().cudaGetLastError()
                logger.debug(
                    "[Prims-TS MoE] Skipping precompile for "
                    f"{self.__class__.__name__} tactic {tactic}: {exc}"
                )

        # Compilation is mostly host-side; fan out tactics when there are several.
        max_workers = min(4, max(1, len(tactics)))
        if max_workers == 1 or len(tactics) <= 1:
            for tactic in tactics:
                _precompile_one(tactic)
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(_precompile_one, tactics))
        return True


class PrimsTsBf16MoERunner(_PrimsTsMoERunnerMixin, TunableRunner):
    """Autotuned Prims-TS BF16 MoE runner using shared TRT-LLM routing/finalize."""

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op,
        *,
        top_k: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: int = ActivationType.Swiglu.value,
        use_shuffled_weight: bool = True,
        weight_layout: int = WeightLayout.MajorK,
        use_per_token_scaling: bool = False,
        num_experts: Optional[int] = None,
    ):
        self.moe_op = moe_op
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.dtype_act = DtypeTrtllmGen.Bfloat16
        self.dtype_weights = DtypeTrtllmGen.Bfloat16
        self.fp8_quantization_type = Fp8QuantizationType.NoneFp8
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.use_shuffled_weight = use_shuffled_weight
        self.weight_layout = WeightLayout(weight_layout)
        self.use_per_token_scaling = use_per_token_scaling
        self.num_experts = num_experts if num_experts is not None else num_local_experts
        self._topk_initializer_source = None
        self._topk_initializer = None

    def _map_tactic(self, tactic, moe_inputs, kwargs=None):
        common, bias_and_oa, _ = self._tactic_common(moe_inputs, kwargs)
        return map_trtllm_bf16_moe_tactic(tactic, **common, **bias_and_oa)

    def _make_tuning_config(
        self,
        moe_inputs: MoeRunnerInputs,
        tune_max_num_tokens: int = 8192,
        routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
        **kwargs,
    ) -> TuningConfig:
        return make_moe_tuning_config(
            moe_inputs,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            fp8_quantization_type=self.fp8_quantization_type,
            init_packed_topk_ids=_moe_topk_ids_init_for_routing(
                self.num_experts, routing_input_mode
            ),
            tune_max_num_tokens=tune_max_num_tokens,
            keep_empty_expert_weights_static=True,
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        sm_arch = _sm_arch_for_tensor(moe_inputs.hidden_states)
        has_gemm1_lora_delta = moe_inputs.gemm1_lora_delta is not None
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.num_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            sm_arch,
            has_gemm1_lora_delta,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsBf16MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_bf16_moe_tactics(
                    sm_arch=sm_arch,
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.num_experts,
                    weight_layout=int(self.weight_layout),
                    **gemm_config_flags,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to enumerate Prims-TS BF16 MoE tactics"
                ) from exc
            PrimsTsBf16MoERunner.valid_tactics_dict[instance_key] = (
                _with_default_moe_tactic(valid_tactics)
            )
        return self._filter_partition_geometry_tactics(
            PrimsTsBf16MoERunner.valid_tactics_dict[instance_key],
            inputs,
            instance_key,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        del do_preparation
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        requested_tactic = [-1, -1] if tactic == -1 else tactic
        hidden_states = moe_inputs.hidden_states
        output = moe_inputs.output
        num_tokens = hidden_states.shape[0]
        pair = self._map_tactic(requested_tactic, moe_inputs, kwargs)
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_bf16_supported(
            self,
            moe_inputs,
            resolved_tactic,
            pair=pair,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS BF16 kernel ({reason})"
            )

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kwargs)
        routing_out = self.moe_op.trtllm_moe_run_routing(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            topk_ids,
            expert_weights_in,
            hidden_states,
            self._staged_tensor(kwargs, "gemm1_weights"),
            self._staged_tensor(kwargs, "gemm2_weights"),
            kwargs["num_experts"],
            self.top_k,
            kwargs["n_group"],
            kwargs["topk_group"],
            self.intermediate_size,
            kwargs["local_expert_offset"],
            self.num_local_experts,
            kwargs["routed_scaling_factor"],
            kwargs["routing_method_type"],
            kwargs["use_shuffled_weight"],
            kwargs["weight_layout"],
            kwargs["enable_pdl"],
            resolved_tactic,
            int(self.activation_type),
            kwargs.get("norm_topk_prob", True),
            kwargs.get("routing_replay_out"),
        )

        (
            expert_weights,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            tile_idx,
            mn_limit,
            num_non_exiting_ctas,
            total_num_padded_tokens,
            gemm1_output,
            gemm2_output,
        ) = _torch_views_of_ffi_tensors(routing_out)
        expert_weights = _select_expert_weights(moe_inputs, expert_weights, kwargs)
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
        )

        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()

        common_io_kwargs = dict(
            hidden_states=hidden_states,
            gemm1_weights=kwargs["gemm1_weights"],
            gemm2_weights=kwargs["gemm2_weights"],
            gemm1_bias=kwargs.get("gemm1_bias"),
            gemm2_bias=kwargs.get("gemm2_bias"),
            **_gemm1_oa_io_kwargs(kwargs),
            gemm1_output=gemm1_output,
            gemm2_output=gemm2_output,
            tile_idx=tile_idx,
            mn_limit=mn_limit,
            route_map=permuted_idx_to_token_idx,
            num_non_exiting_ctas=num_non_exiting_ctas,
            total_num_padded_tokens=total_num_padded_tokens,
            routed_token_capacity=routed_token_capacity,
            activation_type=int(self.activation_type),
            num_experts=self.num_local_experts,
            num_tokens=num_tokens,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
        )
        if self._is_partitioned:
            return self._run_partitioned(
                builder=build_bf16_launch_io,
                fc1_cfg=fc1_cfg,
                fc2_cfg=fc2_cfg,
                common_io_kwargs=common_io_kwargs,
                fc1_label="fc1",
                fc2_label="fc2",
                gemm1_output=gemm1_output,
                gemm2_output=gemm2_output,
                output=output,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                total_num_padded_tokens=total_num_padded_tokens,
                num_tokens=num_tokens,
                kwargs=kwargs,
            )

        fc1_io = build_bf16_launch_io(
            fc="fc1",
            cfg=fc1_cfg,
            **common_io_kwargs,
        )
        fc1_hash = stable_config_hash(fc1_io["cfg"])
        fc1_fn = get_compiled_gemm(fc1_hash, "fc1", fc1_io, stream)
        fc1_fn(*self._launch_args(fc1_io, stream))

        fc2_io = build_bf16_launch_io(fc="fc2", cfg=fc2_cfg, **common_io_kwargs)
        fc2_hash = stable_config_hash(fc2_io["cfg"])
        fc2_fn = get_compiled_gemm(fc2_hash, "fc2", fc2_io, stream)
        fc2_fn(*self._launch_args(fc2_io, stream))

        if kwargs["do_finalize"]:
            self.moe_op.trtllm_moe_run_finalize(
                gemm2_output,
                output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                kwargs["num_experts"],
                self.top_k,
                self.hidden_size,
                kwargs["enable_pdl"],
                False,
                None,
            )
            return []
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    @staticmethod
    def _launch_args(io: dict, stream) -> tuple:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

        return _launch_arg_tuple(io, stream)


class PrimsTsNvfp4MoERunner(_PrimsTsMoERunnerMixin, TunableRunner):
    """Autotuned Prims-TS NVFP4xNVFP4 MoE runner."""

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op,
        *,
        top_k: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: int = ActivationType.Swiglu.value,
        use_shuffled_weight: bool = True,
        weight_layout: int = WeightLayout.MajorK,
        use_per_token_scaling: bool = False,
        num_experts: Optional[int] = None,
    ):
        self.moe_op = moe_op
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.dtype_act = DtypeTrtllmGen.E2m1
        self.dtype_weights = DtypeTrtllmGen.E2m1
        self.fp8_quantization_type = Fp8QuantizationType.NoneFp8
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.use_shuffled_weight = use_shuffled_weight
        self.weight_layout = WeightLayout(weight_layout)
        self.use_per_token_scaling = use_per_token_scaling
        self.num_experts = num_experts if num_experts is not None else num_local_experts

    def _map_tactic(self, tactic, moe_inputs, kwargs=None):
        common, bias_and_oa, extras = self._tactic_common(moe_inputs, kwargs)
        uses_per_token_scaling = moe_inputs.per_token_scale is not None
        return map_trtllm_nvfp4_moe_tactic(
            tactic,
            **common,
            **bias_and_oa,
            use_per_token_sf_b=uses_per_token_scaling,
            per_token_sf_dtype=(
                _per_token_sf_dtype_value(moe_inputs.per_token_scale)
                if uses_per_token_scaling
                else int(extras.get("per_token_sf_dtype", 1))
            ),
        )

    def _make_tuning_config(
        self,
        moe_inputs: MoeRunnerInputs,
        tune_max_num_tokens: int = 8192,
        routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
        **kwargs,
    ) -> TuningConfig:
        return make_moe_tuning_config(
            moe_inputs,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            fp8_quantization_type=self.fp8_quantization_type,
            init_packed_topk_ids=_moe_topk_ids_init_for_routing(
                self.num_experts, routing_input_mode
            ),
            tune_max_num_tokens=tune_max_num_tokens,
            keep_empty_expert_weights_static=True,
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        sm_arch = _sm_arch_for_tensor(moe_inputs.hidden_states)
        uses_per_token_scaling = moe_inputs.per_token_scale is not None
        per_token_sf_dtype = (
            _per_token_sf_dtype_value(moe_inputs.per_token_scale)
            if uses_per_token_scaling
            else 1
        )
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.num_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            sm_arch,
            False,
            uses_per_token_scaling,
            per_token_sf_dtype,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsNvfp4MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_nvfp4_moe_tactics(
                    sm_arch=sm_arch,
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.num_experts,
                    weight_layout=int(self.weight_layout),
                    use_per_token_sf_b=uses_per_token_scaling,
                    per_token_sf_dtype=per_token_sf_dtype,
                    **gemm_config_flags,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to enumerate Prims-TS NVFP4 MoE tactics"
                ) from exc
            PrimsTsNvfp4MoERunner.valid_tactics_dict[instance_key] = (
                _with_default_moe_tactic(valid_tactics)
            )
        return self._filter_partition_geometry_tactics(
            PrimsTsNvfp4MoERunner.valid_tactics_dict[instance_key],
            inputs,
            instance_key,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        del do_preparation
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        requested_tactic = [-1, -1] if tactic == -1 else tactic
        hidden_states = moe_inputs.hidden_states
        output = moe_inputs.output
        num_tokens = hidden_states.shape[0]
        uses_per_token_scaling = moe_inputs.per_token_scale is not None
        pair = self._map_tactic(requested_tactic, moe_inputs, kwargs)
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_nvfp4_supported(
            self,
            moe_inputs,
            resolved_tactic,
            pair=pair,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS NVFP4 kernel ({reason})"
            )

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)
        topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kwargs)
        routing_out = self.moe_op.trtllm_moe_run_routing_fp4_nvfp4(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            topk_ids,
            expert_weights_in,
            hidden_states,
            moe_inputs.hidden_states_scale,
            self._staged_tensor(kwargs, "gemm1_weights"),
            self._staged_tensor(kwargs, "gemm1_weights_scale"),
            self._staged_tensor(kwargs, "gemm2_weights"),
            self._staged_tensor(kwargs, "gemm2_weights_scale"),
            kwargs["output1_scale_scalar"],
            kwargs["output1_scale_gate_scalar"],
            kwargs["output2_scale_scalar"],
            kwargs["num_experts"],
            self.top_k,
            kwargs["n_group"],
            kwargs["topk_group"],
            self.intermediate_size,
            kwargs["local_expert_offset"],
            self.num_local_experts,
            kwargs["routed_scaling_factor"],
            kwargs["routing_method_type"],
            kwargs["enable_pdl"],
            resolved_tactic,
            int(kwargs.get("weight_layout", self.weight_layout)),
            int(self.activation_type),
            kwargs.get("norm_topk_prob", True),
            kwargs.get("routing_replay_out"),
        )

        (
            expert_weights,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            tile_idx,
            mn_limit,
            num_non_exiting_ctas,
            total_num_padded_tokens,
            gemm1_output,
            gemm1_output_scale,
            gemm2_output,
        ) = _torch_views_of_ffi_tensors(routing_out)
        expert_weights = _select_expert_weights(moe_inputs, expert_weights, kwargs)
        if (
            moe_inputs.routing_logits is not None
            and moe_inputs.routing_logits.numel() > 0
            and int(kwargs["routing_method_type"]) == int(RoutingMethodType.TopK)
        ):
            expert_weights = torch.topk(
                moe_inputs.routing_logits.to(torch.float32),
                self.top_k,
                dim=-1,
            ).values.to(torch.bfloat16)
        elif expert_weights.dtype != torch.bfloat16:
            expert_weights = expert_weights.to(torch.bfloat16)
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
        )
        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()
        if uses_per_token_scaling and not fc1_cfg.has_output_block_scaling:
            gemm1_output = torch.empty(
                (int(gemm1_output.shape[0]), self.intermediate_size),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

        common_io_kwargs = dict(
            hidden_states=hidden_states,
            hidden_states_scale=moe_inputs.hidden_states_scale,
            gemm1_weights=kwargs["gemm1_weights"],
            gemm1_weights_scale=kwargs["gemm1_weights_scale"],
            gemm1_bias=kwargs.get("gemm1_bias"),
            gemm2_weights=kwargs["gemm2_weights"],
            gemm2_weights_scale=kwargs["gemm2_weights_scale"],
            gemm2_bias=kwargs.get("gemm2_bias"),
            **_gemm1_oa_io_kwargs(kwargs),
            gemm1_output=gemm1_output,
            gemm1_output_scale=gemm1_output_scale,
            gemm2_output=gemm2_output,
            output1_scale_scalar=kwargs["output1_scale_scalar"],
            output1_scale_gate_scalar=kwargs["output1_scale_gate_scalar"],
            output2_scale_scalar=kwargs["output2_scale_scalar"],
            tile_idx=tile_idx,
            mn_limit=mn_limit,
            route_map=permuted_idx_to_token_idx,
            num_non_exiting_ctas=num_non_exiting_ctas,
            total_num_padded_tokens=total_num_padded_tokens,
            routed_token_capacity=routed_token_capacity,
            activation_type=int(self.activation_type),
            num_experts=self.num_local_experts,
            num_tokens=num_tokens,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            per_token_sf_b=moe_inputs.per_token_scale,
        )
        if self._is_partitioned:
            return self._run_partitioned(
                builder=build_nvfp4_launch_io,
                fc1_cfg=fc1_cfg,
                fc2_cfg=fc2_cfg,
                common_io_kwargs=common_io_kwargs,
                fc1_label="nvfp4_fc1",
                fc2_label="nvfp4_fc2",
                gemm1_output=gemm1_output,
                gemm1_output_scale=gemm1_output_scale,
                gemm2_output=gemm2_output,
                output=output,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                total_num_padded_tokens=total_num_padded_tokens,
                num_tokens=num_tokens,
                kwargs=kwargs,
            )
        fc1_io = build_nvfp4_launch_io(
            fc="fc1",
            cfg=fc1_cfg,
            **common_io_kwargs,
        )
        fc1_hash = stable_config_hash(fc1_io["cfg"])
        fc1_fn = get_compiled_gemm(fc1_hash, "nvfp4_fc1", fc1_io, stream)
        fc1_fn(*self._launch_args(fc1_io, stream))

        fc2_io_kwargs = common_io_kwargs
        if uses_per_token_scaling and fc2_cfg.has_per_token_sf_b:
            (
                activation_output,
                activation_output_scale,
                per_token_scale_fc2,
            ) = _quantize_nvfp4_fc1_output_for_fc2(
                gemm1_output=gemm1_output,
                gemm1_output_scale=gemm1_output_scale,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                num_tokens=num_tokens,
                top_k=self.top_k,
                intermediate_size=self.intermediate_size,
                tile_n=pair.tile_n,
            )
            fc2_io_kwargs = {
                **common_io_kwargs,
                "gemm1_output": activation_output,
                "gemm1_output_scale": activation_output_scale,
                "per_token_sf_b": per_token_scale_fc2,
            }

        fc2_io = build_nvfp4_launch_io(fc="fc2", cfg=fc2_cfg, **fc2_io_kwargs)
        fc2_hash = stable_config_hash(fc2_io["cfg"])
        fc2_fn = get_compiled_gemm(fc2_hash, "nvfp4_fc2", fc2_io, stream)
        fc2_fn(*self._launch_args(fc2_io, stream))

        if kwargs["do_finalize"]:
            self.moe_op.trtllm_moe_run_finalize(
                gemm2_output,
                output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                kwargs["num_experts"],
                self.top_k,
                self.hidden_size,
                kwargs["enable_pdl"],
                False,
                None,
            )
            return []
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    @staticmethod
    def _launch_args(io: dict, stream) -> tuple:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

        return _launch_arg_tuple(io, stream)


class PrimsTsMxfp4Mxfp8MoERunner(_PrimsTsMoERunnerMixin, TunableRunner):
    """Autotuned Prims-TS MXFP4xMXFP8 MoE runner."""

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op,
        *,
        top_k: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: int = ActivationType.Swiglu.value,
        use_shuffled_weight: bool = True,
        weight_layout: int = WeightLayout.MajorK,
        use_per_token_scaling: bool = False,
        num_experts: Optional[int] = None,
    ):
        self.moe_op = moe_op
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.dtype_act = DtypeTrtllmGen.MxE4m3
        self.dtype_weights = DtypeTrtllmGen.MxE2m1
        self.fp8_quantization_type = Fp8QuantizationType.NoneFp8
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.use_shuffled_weight = use_shuffled_weight
        self.weight_layout = WeightLayout(weight_layout)
        self.use_per_token_scaling = use_per_token_scaling
        self.num_experts = num_experts if num_experts is not None else num_local_experts
        # Excluded from TunableRunner.__hash__: the tensor and closure carry
        # per-call identity but do not change which tactics are valid.
        self._topk_initializer_cache = None

    def _map_tactic(self, tactic, moe_inputs, kwargs=None):
        common, bias_and_oa, _ = self._tactic_common(moe_inputs, kwargs)
        return map_trtllm_mxfp4_mxfp8_moe_tactic(tactic, **common, **bias_and_oa)

    def _make_tuning_config(
        self,
        moe_inputs: MoeRunnerInputs,
        tune_max_num_tokens: int = 8192,
        routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
        **kwargs,
    ) -> TuningConfig:
        if moe_inputs.topk_ids is not None and moe_inputs.topk_ids.numel() > 0:
            if (
                self._topk_initializer_cache is None
                or self._topk_initializer_cache[0] is not moe_inputs.topk_ids
            ):
                self._topk_initializer_cache = (
                    moe_inputs.topk_ids,
                    make_repeating_tensor_initializer(
                        moe_inputs.topk_ids,
                        num_experts=self.num_experts,
                        packed=(
                            routing_input_mode != RoutingInputMode.UnpackedPrecomputed
                        ),
                    ),
                )
            init_packed_topk_ids = self._topk_initializer_cache[1]
        else:
            init_packed_topk_ids = _moe_topk_ids_init_for_routing(
                self.num_experts, routing_input_mode
            )

        return make_moe_tuning_config(
            moe_inputs,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            fp8_quantization_type=self.fp8_quantization_type,
            init_packed_topk_ids=init_packed_topk_ids,
            tune_max_num_tokens=tune_max_num_tokens,
            keep_empty_expert_weights_static=True,
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        sm_arch = _sm_arch_for_tensor(moe_inputs.hidden_states)
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.num_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            sm_arch,
            False,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsMxfp4Mxfp8MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_mxfp4_mxfp8_moe_tactics(
                    sm_arch=sm_arch,
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.num_experts,
                    weight_layout=int(self.weight_layout),
                    **gemm_config_flags,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to enumerate Prims-TS MXFP4xMXFP8 MoE tactics"
                ) from exc
            PrimsTsMxfp4Mxfp8MoERunner.valid_tactics_dict[instance_key] = (
                _with_default_moe_tactic(valid_tactics)
            )
        return self._filter_partition_geometry_tactics(
            PrimsTsMxfp4Mxfp8MoERunner.valid_tactics_dict[instance_key],
            inputs,
            instance_key,
        )

    def precompile_tactics(
        self,
        inputs: List[torch.Tensor],
        tactics: List[Any],
        profile: OptimizationProfile,
        **kwargs,
    ) -> bool:
        del profile
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        hidden_states = moe_inputs.hidden_states
        num_tokens = hidden_states.shape[0]

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        for tactic in tactics:
            try:
                requested_tactic = [-1, -1] if tactic == -1 else tactic
                pair = self._map_tactic(requested_tactic, moe_inputs, kwargs)
                resolved_tactic = _concrete_tactic(pair)
                ok, reason = is_prims_ts_mxfp4_mxfp8_supported(
                    self,
                    moe_inputs,
                    resolved_tactic,
                    pair=pair,
                    **kwargs,
                )
                if not ok:
                    logger.debug(
                        "[Prims-TS MoE] Skipping MXFP4xMXFP8 precompile for "
                        f"unsupported tactic {tactic}: {reason}"
                    )
                    continue

                topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kwargs)
                routing_out = self.moe_op.trtllm_moe_run_routing_fp4_mxfp4_mxfp8(
                    moe_inputs.routing_logits,
                    kwargs["routing_bias"],
                    topk_ids,
                    expert_weights_in,
                    hidden_states,
                    moe_inputs.hidden_states_scale,
                    self._staged_tensor(kwargs, "gemm1_weights"),
                    self._staged_tensor(kwargs, "gemm1_weights_scale"),
                    self._staged_tensor(kwargs, "gemm2_weights"),
                    self._staged_tensor(kwargs, "gemm2_weights_scale"),
                    kwargs["output1_scale_scalar"],
                    kwargs["output1_scale_gate_scalar"],
                    kwargs["output2_scale_scalar"],
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["enable_pdl"],
                    resolved_tactic,
                    int(kwargs.get("weight_layout", self.weight_layout)),
                    int(self.activation_type),
                    kwargs.get("norm_topk_prob", True),
                    kwargs.get("routing_replay_out"),
                )

                (
                    _expert_weights,
                    _expanded_idx_to_permuted_idx,
                    permuted_idx_to_token_idx,
                    tile_idx,
                    mn_limit,
                    num_non_exiting_ctas,
                    total_num_padded_tokens,
                    gemm1_output,
                    gemm1_output_scale,
                    gemm2_output,
                ) = _torch_views_of_ffi_tensors(routing_out)

                fc1_cfg = pair.fc1.cfg.build()
                fc2_cfg = pair.fc2.cfg.build()

                common_io_kwargs = dict(
                    hidden_states=hidden_states,
                    hidden_states_scale=moe_inputs.hidden_states_scale,
                    gemm1_weights=kwargs["gemm1_weights"],
                    gemm1_weights_scale=kwargs["gemm1_weights_scale"],
                    gemm1_bias=kwargs.get("gemm1_bias"),
                    gemm2_weights=kwargs["gemm2_weights"],
                    gemm2_weights_scale=kwargs["gemm2_weights_scale"],
                    gemm2_bias=kwargs.get("gemm2_bias"),
                    **_gemm1_oa_io_kwargs(kwargs),
                    gemm1_output=gemm1_output,
                    gemm1_output_scale=gemm1_output_scale,
                    gemm2_output=gemm2_output,
                    output1_scale_scalar=kwargs["output1_scale_scalar"],
                    output1_scale_gate_scalar=kwargs["output1_scale_gate_scalar"],
                    output2_scale_scalar=kwargs["output2_scale_scalar"],
                    tile_idx=tile_idx,
                    mn_limit=mn_limit,
                    route_map=permuted_idx_to_token_idx,
                    num_non_exiting_ctas=num_non_exiting_ctas,
                    total_num_padded_tokens=total_num_padded_tokens,
                    activation_type=int(self.activation_type),
                    num_experts=self.num_local_experts,
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    intermediate_size=self.intermediate_size,
                    hidden_size=self.hidden_size,
                )

                fc1_io = build_mxfp4_mxfp8_launch_io(
                    fc="fc1", cfg=fc1_cfg, **common_io_kwargs
                )
                fc1_hash = stable_config_hash(fc1_io["cfg"])
                get_compiled_gemm(
                    fc1_hash,
                    "mxfp4_mxfp8_fc1",
                    fc1_io,
                    stream,
                )

                fc2_io = build_mxfp4_mxfp8_launch_io(
                    fc="fc2", cfg=fc2_cfg, **common_io_kwargs
                )
                fc2_hash = stable_config_hash(fc2_io["cfg"])
                get_compiled_gemm(
                    fc2_hash,
                    "mxfp4_mxfp8_fc2",
                    fc2_io,
                    stream,
                )
                torch_stream.synchronize()
            except Exception as exc:
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize(hidden_states.device)
                with contextlib.suppress(Exception):
                    torch.cuda.cudart().cudaGetLastError()
                logger.debug(
                    "[Prims-TS MoE] Skipping MXFP4xMXFP8 precompile for tactic "
                    f"{tactic}: {exc}"
                )

        return True

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        if do_preparation:
            return None
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        requested_tactic = [-1, -1] if tactic == -1 else tactic
        hidden_states = moe_inputs.hidden_states
        output = moe_inputs.output
        num_tokens = hidden_states.shape[0]
        pair = self._map_tactic(requested_tactic, moe_inputs, kwargs)
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_mxfp4_mxfp8_supported(
            self,
            moe_inputs,
            resolved_tactic,
            pair=pair,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS MXFP4xMXFP8 kernel ({reason})"
            )

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kwargs)
        routing_out = self.moe_op.trtllm_moe_run_routing_fp4_mxfp4_mxfp8(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            topk_ids,
            expert_weights_in,
            hidden_states,
            moe_inputs.hidden_states_scale,
            self._staged_tensor(kwargs, "gemm1_weights"),
            self._staged_tensor(kwargs, "gemm1_weights_scale"),
            self._staged_tensor(kwargs, "gemm2_weights"),
            self._staged_tensor(kwargs, "gemm2_weights_scale"),
            kwargs["output1_scale_scalar"],
            kwargs["output1_scale_gate_scalar"],
            kwargs["output2_scale_scalar"],
            kwargs["num_experts"],
            self.top_k,
            kwargs["n_group"],
            kwargs["topk_group"],
            self.intermediate_size,
            kwargs["local_expert_offset"],
            self.num_local_experts,
            kwargs["routed_scaling_factor"],
            kwargs["routing_method_type"],
            kwargs["enable_pdl"],
            resolved_tactic,
            int(kwargs.get("weight_layout", self.weight_layout)),
            int(self.activation_type),
            kwargs.get("norm_topk_prob", True),
            kwargs.get("routing_replay_out"),
        )

        (
            expert_weights,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            tile_idx,
            mn_limit,
            num_non_exiting_ctas,
            total_num_padded_tokens,
            gemm1_output,
            gemm1_output_scale,
            gemm2_output,
        ) = _torch_views_of_ffi_tensors(routing_out)
        expert_weights = _select_expert_weights(moe_inputs, expert_weights, kwargs)
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
        )

        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()

        common_io_kwargs = dict(
            hidden_states=hidden_states,
            hidden_states_scale=moe_inputs.hidden_states_scale,
            gemm1_weights=kwargs["gemm1_weights"],
            gemm1_weights_scale=kwargs["gemm1_weights_scale"],
            gemm1_bias=kwargs.get("gemm1_bias"),
            gemm2_weights=kwargs["gemm2_weights"],
            gemm2_weights_scale=kwargs["gemm2_weights_scale"],
            gemm2_bias=kwargs.get("gemm2_bias"),
            **_gemm1_oa_io_kwargs(kwargs),
            gemm1_output=gemm1_output,
            gemm1_output_scale=gemm1_output_scale,
            gemm2_output=gemm2_output,
            output1_scale_scalar=kwargs["output1_scale_scalar"],
            output1_scale_gate_scalar=kwargs["output1_scale_gate_scalar"],
            output2_scale_scalar=kwargs["output2_scale_scalar"],
            tile_idx=tile_idx,
            mn_limit=mn_limit,
            route_map=permuted_idx_to_token_idx,
            num_non_exiting_ctas=num_non_exiting_ctas,
            total_num_padded_tokens=total_num_padded_tokens,
            routed_token_capacity=routed_token_capacity,
            activation_type=int(self.activation_type),
            num_experts=self.num_local_experts,
            num_tokens=num_tokens,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
        )
        if self._is_partitioned:
            return self._run_partitioned(
                builder=build_mxfp4_mxfp8_launch_io,
                fc1_cfg=fc1_cfg,
                fc2_cfg=fc2_cfg,
                common_io_kwargs=common_io_kwargs,
                fc1_label="mxfp4_mxfp8_fc1",
                fc2_label="mxfp4_mxfp8_fc2",
                gemm1_output=gemm1_output,
                gemm1_output_scale=gemm1_output_scale,
                gemm2_output=gemm2_output,
                output=output,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                total_num_padded_tokens=total_num_padded_tokens,
                num_tokens=num_tokens,
                kwargs=kwargs,
            )
        fc1_io = build_mxfp4_mxfp8_launch_io(fc="fc1", cfg=fc1_cfg, **common_io_kwargs)
        fc1_hash = stable_config_hash(fc1_io["cfg"])
        fc1_fn = get_compiled_gemm(fc1_hash, "mxfp4_mxfp8_fc1", fc1_io, stream)
        fc1_fn(*self._launch_args(fc1_io, stream))

        fc2_io = build_mxfp4_mxfp8_launch_io(fc="fc2", cfg=fc2_cfg, **common_io_kwargs)
        fc2_hash = stable_config_hash(fc2_io["cfg"])
        fc2_fn = get_compiled_gemm(fc2_hash, "mxfp4_mxfp8_fc2", fc2_io, stream)
        fc2_fn(*self._launch_args(fc2_io, stream))

        if kwargs["do_finalize"]:
            self.moe_op.trtllm_moe_run_finalize(
                gemm2_output,
                output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                kwargs["num_experts"],
                self.top_k,
                self.hidden_size,
                kwargs["enable_pdl"],
                False,
                None,
            )
            return []
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    @staticmethod
    def _launch_args(io: dict, stream) -> tuple:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

        return _launch_arg_tuple(io, stream)


class PrimsTsMxfp4Bf16MoERunner(_PrimsTsMoERunnerMixin, TunableRunner):
    """Autotuned Prims-TS MXFP4xBF16 MoE runner."""

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op,
        *,
        top_k: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: int = ActivationType.Swiglu.value,
        use_shuffled_weight: bool = True,
        weight_layout: int = WeightLayout.MajorK,
        use_per_token_scaling: bool = False,
        num_experts: Optional[int] = None,
    ):
        self.moe_op = moe_op
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.dtype_act = DtypeTrtllmGen.Bfloat16
        self.dtype_weights = DtypeTrtllmGen.MxE2m1
        self.fp8_quantization_type = Fp8QuantizationType.NoneFp8
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.use_shuffled_weight = use_shuffled_weight
        self.weight_layout = WeightLayout(weight_layout)
        self.use_per_token_scaling = use_per_token_scaling
        self.num_experts = num_experts if num_experts is not None else num_local_experts

    def _map_tactic(self, tactic, moe_inputs, kwargs=None):
        common, bias_and_oa, _ = self._tactic_common(moe_inputs, kwargs)
        return map_trtllm_mxfp4_bf16_moe_tactic(tactic, **common, **bias_and_oa)

    def _make_tuning_config(
        self,
        moe_inputs: MoeRunnerInputs,
        tune_max_num_tokens: int = 8192,
        routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
        **kwargs,
    ) -> TuningConfig:
        return make_moe_tuning_config(
            moe_inputs,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            fp8_quantization_type=self.fp8_quantization_type,
            init_packed_topk_ids=_moe_topk_ids_init_for_routing(
                self.num_experts, routing_input_mode
            ),
            tune_max_num_tokens=tune_max_num_tokens,
            keep_empty_expert_weights_static=True,
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        sm_arch = _sm_arch_for_tensor(moe_inputs.hidden_states)
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.num_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            sm_arch,
            False,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsMxfp4Bf16MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_mxfp4_bf16_moe_tactics(
                    sm_arch=sm_arch,
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.num_experts,
                    weight_layout=int(self.weight_layout),
                    **gemm_config_flags,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to enumerate Prims-TS MXFP4xBF16 MoE tactics"
                ) from exc
            PrimsTsMxfp4Bf16MoERunner.valid_tactics_dict[instance_key] = (
                _with_default_moe_tactic(valid_tactics)
            )
        return self._filter_partition_geometry_tactics(
            PrimsTsMxfp4Bf16MoERunner.valid_tactics_dict[instance_key],
            inputs,
            instance_key,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        del do_preparation
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        requested_tactic = [-1, -1] if tactic == -1 else tactic
        hidden_states = moe_inputs.hidden_states
        output = moe_inputs.output
        num_tokens = hidden_states.shape[0]
        pair = self._map_tactic(requested_tactic, moe_inputs, kwargs)
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_mxfp4_bf16_supported(
            self,
            moe_inputs,
            resolved_tactic,
            pair=pair,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS MXFP4xBF16 kernel ({reason})"
            )

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kwargs)
        routing_out = self.moe_op.trtllm_moe_run_routing_fp4_mxfp4_bf16(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            topk_ids,
            expert_weights_in,
            hidden_states,
            self._staged_tensor(kwargs, "gemm1_weights"),
            self._staged_tensor(kwargs, "gemm1_weights_scale"),
            self._staged_tensor(kwargs, "gemm2_weights"),
            self._staged_tensor(kwargs, "gemm2_weights_scale"),
            kwargs["output1_scale_scalar"],
            kwargs["output1_scale_gate_scalar"],
            kwargs["output2_scale_scalar"],
            kwargs["num_experts"],
            self.top_k,
            kwargs["n_group"],
            kwargs["topk_group"],
            self.intermediate_size,
            kwargs["local_expert_offset"],
            self.num_local_experts,
            kwargs["routed_scaling_factor"],
            kwargs["routing_method_type"],
            kwargs["enable_pdl"],
            resolved_tactic,
            int(kwargs.get("weight_layout", self.weight_layout)),
            int(self.activation_type),
            kwargs.get("norm_topk_prob", True),
            kwargs.get("routing_replay_out"),
        )

        (
            expert_weights,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            tile_idx,
            mn_limit,
            num_non_exiting_ctas,
            total_num_padded_tokens,
            gemm1_output,
            gemm2_output,
        ) = _torch_views_of_ffi_tensors(routing_out)
        expert_weights = _select_expert_weights(moe_inputs, expert_weights, kwargs)
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
        )

        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()

        common_io_kwargs = dict(
            hidden_states=hidden_states,
            gemm1_weights=kwargs["gemm1_weights"],
            gemm1_weights_scale=kwargs["gemm1_weights_scale"],
            gemm1_bias=kwargs.get("gemm1_bias"),
            gemm2_weights=kwargs["gemm2_weights"],
            gemm2_weights_scale=kwargs["gemm2_weights_scale"],
            gemm2_bias=kwargs.get("gemm2_bias"),
            **_gemm1_oa_io_kwargs(kwargs),
            gemm1_output=gemm1_output,
            gemm2_output=gemm2_output,
            output1_scale_scalar=kwargs["output1_scale_scalar"],
            output1_scale_gate_scalar=kwargs["output1_scale_gate_scalar"],
            output2_scale_scalar=kwargs["output2_scale_scalar"],
            tile_idx=tile_idx,
            mn_limit=mn_limit,
            route_map=permuted_idx_to_token_idx,
            num_non_exiting_ctas=num_non_exiting_ctas,
            total_num_padded_tokens=total_num_padded_tokens,
            routed_token_capacity=routed_token_capacity,
            activation_type=int(self.activation_type),
            num_experts=self.num_local_experts,
            num_tokens=num_tokens,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
        )
        if self._is_partitioned:
            return self._run_partitioned(
                builder=build_mxfp4_bf16_launch_io,
                fc1_cfg=fc1_cfg,
                fc2_cfg=fc2_cfg,
                common_io_kwargs=common_io_kwargs,
                fc1_label="mxfp4_bf16_fc1",
                fc2_label="mxfp4_bf16_fc2",
                gemm1_output=gemm1_output,
                gemm2_output=gemm2_output,
                output=output,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                total_num_padded_tokens=total_num_padded_tokens,
                num_tokens=num_tokens,
                kwargs=kwargs,
            )
        fc1_io = build_mxfp4_bf16_launch_io(fc="fc1", cfg=fc1_cfg, **common_io_kwargs)
        fc1_hash = stable_config_hash(fc1_io["cfg"])
        fc1_fn = get_compiled_gemm(fc1_hash, "mxfp4_bf16_fc1", fc1_io, stream)
        fc1_fn(*self._launch_args(fc1_io, stream))

        fc2_io = build_mxfp4_bf16_launch_io(fc="fc2", cfg=fc2_cfg, **common_io_kwargs)
        fc2_hash = stable_config_hash(fc2_io["cfg"])
        fc2_fn = get_compiled_gemm(fc2_hash, "mxfp4_bf16_fc2", fc2_io, stream)
        fc2_fn(*self._launch_args(fc2_io, stream))

        if kwargs["do_finalize"]:
            self.moe_op.trtllm_moe_run_finalize(
                gemm2_output,
                output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                kwargs["num_experts"],
                self.top_k,
                self.hidden_size,
                kwargs["enable_pdl"],
                False,
                None,
            )
            return []
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    @staticmethod
    def _launch_args(io: dict, stream) -> tuple:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

        return _launch_arg_tuple(io, stream)


class PrimsTsFp8PerTensorMoERunner(_PrimsTsMoERunnerMixin, TunableRunner):
    """Autotuned Prims-TS FP8 per-tensor MoE runner."""

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op,
        *,
        top_k: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: int = ActivationType.Swiglu.value,
        use_shuffled_weight: bool = True,
        weight_layout: int = WeightLayout.MajorK,
        num_experts: Optional[int] = None,
    ):
        self.moe_op = moe_op
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.dtype_act = DtypeTrtllmGen.E4m3
        self.dtype_weights = DtypeTrtllmGen.E4m3
        self.fp8_quantization_type = Fp8QuantizationType.NoneFp8
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.use_shuffled_weight = use_shuffled_weight
        self.weight_layout = WeightLayout(weight_layout)
        self.use_per_token_scaling = False
        self.num_experts = num_experts if num_experts is not None else num_local_experts

    def _map_tactic(self, tactic, moe_inputs, kwargs=None):
        common, bias_and_oa, extras = self._tactic_common(moe_inputs, kwargs)
        fc1_scale = extras.get("fc1_per_channel_weight_scale")
        fc2_scale = extras.get("fc2_per_channel_weight_scale")
        use_fc1_scale = (
            fc1_scale is not None
            if isinstance(fc1_scale, torch.Tensor)
            else bool(fc1_scale)
        )
        use_fc2_scale = (
            fc2_scale is not None
            if isinstance(fc2_scale, torch.Tensor)
            else bool(fc2_scale)
        )
        use_routing_scales = bool(extras.get("use_routing_scales_on_input", False))
        per_token_sf_dtype = _fp8_per_tensor_scale_dtype(
            fc1_per_channel_weight_scale_dtype=(
                _per_token_sf_dtype_value(fc1_scale)
                if isinstance(fc1_scale, torch.Tensor)
                else (
                    int(extras.get("per_token_sf_dtype", 1)) if use_fc1_scale else None
                )
            ),
            fc2_per_channel_weight_scale_dtype=(
                _per_token_sf_dtype_value(fc2_scale)
                if isinstance(fc2_scale, torch.Tensor)
                else (
                    int(extras.get("per_token_sf_dtype", 1)) if use_fc2_scale else None
                )
            ),
            use_routing_scales_on_input=use_routing_scales,
            routing_logits=moe_inputs.routing_logits,
        )
        return map_trtllm_fp8_per_tensor_moe_tactic(
            tactic,
            **common,
            **bias_and_oa,
            fc1_use_per_token_sf_a=use_fc1_scale,
            fc2_use_per_token_sf_a=use_fc2_scale,
            use_per_token_sf_b=use_routing_scales,
            per_token_sf_dtype=per_token_sf_dtype,
        )

    def _make_tuning_config(
        self,
        moe_inputs: MoeRunnerInputs,
        tune_max_num_tokens: int = 8192,
        routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
        **kwargs,
    ) -> TuningConfig:
        return make_moe_tuning_config(
            moe_inputs,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            fp8_quantization_type=self.fp8_quantization_type,
            init_packed_topk_ids=_moe_topk_ids_init_for_routing(
                self.num_experts, routing_input_mode
            ),
            tune_max_num_tokens=tune_max_num_tokens,
            keep_empty_expert_weights_static=True,
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        sm_arch = _sm_arch_for_tensor(moe_inputs.hidden_states)
        static_extras = dict(getattr(self, "_cache_key_static_extras", ()))
        use_fc1_per_channel_weight_scale = bool(
            static_extras.get("fc1_per_channel_weight_scale", False)
        )
        use_fc2_per_channel_weight_scale = bool(
            static_extras.get("fc2_per_channel_weight_scale", False)
        )
        use_routing_scales_on_input = bool(
            static_extras.get("use_routing_scales_on_input", False)
        )
        static_sf_dtype = int(static_extras.get("per_token_sf_dtype", 1))
        per_token_sf_dtype = _fp8_per_tensor_scale_dtype(
            fc1_per_channel_weight_scale_dtype=(
                static_sf_dtype if use_fc1_per_channel_weight_scale else None
            ),
            fc2_per_channel_weight_scale_dtype=(
                static_sf_dtype if use_fc2_per_channel_weight_scale else None
            ),
            use_routing_scales_on_input=use_routing_scales_on_input,
            routing_logits=moe_inputs.routing_logits,
        )
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.num_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            sm_arch,
            False,
            use_fc1_per_channel_weight_scale,
            use_fc2_per_channel_weight_scale,
            use_routing_scales_on_input,
            per_token_sf_dtype,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsFp8PerTensorMoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_fp8_per_tensor_moe_tactics(
                    sm_arch=sm_arch,
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.num_experts,
                    weight_layout=int(self.weight_layout),
                    fc1_use_per_token_sf_a=use_fc1_per_channel_weight_scale,
                    fc2_use_per_token_sf_a=use_fc2_per_channel_weight_scale,
                    use_per_token_sf_b=use_routing_scales_on_input,
                    per_token_sf_dtype=per_token_sf_dtype,
                    **gemm_config_flags,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to enumerate Prims-TS FP8 per-tensor MoE tactics"
                ) from exc
            PrimsTsFp8PerTensorMoERunner.valid_tactics_dict[instance_key] = (
                _with_default_moe_tactic(valid_tactics)
            )
        return self._filter_partition_geometry_tactics(
            PrimsTsFp8PerTensorMoERunner.valid_tactics_dict[instance_key],
            inputs,
            instance_key,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        del do_preparation
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        requested_tactic = [-1, -1] if tactic == -1 else tactic
        pair = self._map_tactic(requested_tactic, moe_inputs, kwargs)
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_fp8_per_tensor_supported(
            self,
            moe_inputs,
            resolved_tactic,
            pair=pair,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS FP8 per-tensor kernel ({reason})"
            )

        hidden_states = moe_inputs.hidden_states
        output = moe_inputs.output
        num_tokens = hidden_states.shape[0]
        fc1_per_channel_weight_scale, fc2_per_channel_weight_scale = (
            _split_per_channel_weight_scale_from_kwargs(kwargs)
        )
        use_routing_scales_on_input = bool(
            kwargs.get("use_routing_scales_on_input", False)
        )

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        routing_logits_for_routing = moe_inputs.routing_logits
        if (
            routing_logits_for_routing is not None
            and routing_logits_for_routing.dtype == torch.float32
            and int(kwargs["routing_method_type"]) == int(RoutingMethodType.DeepSeekV3)
        ):
            routing_logits_for_routing = routing_logits_for_routing.to(torch.bfloat16)

        topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kwargs)
        routing_out = self.moe_op.trtllm_moe_run_routing_fp8_per_tensor(
            routing_logits_for_routing,
            kwargs["routing_bias"],
            topk_ids,
            expert_weights_in,
            hidden_states,
            self._staged_tensor(kwargs, "gemm1_weights"),
            kwargs["output1_scale_scalar"],
            kwargs["output1_scale_gate_scalar"],
            self._staged_tensor(kwargs, "gemm2_weights"),
            kwargs["output2_scale_scalar"],
            kwargs["num_experts"],
            self.top_k,
            kwargs["n_group"],
            kwargs["topk_group"],
            self.intermediate_size,
            kwargs["local_expert_offset"],
            self.num_local_experts,
            kwargs["routed_scaling_factor"],
            kwargs["routing_method_type"],
            kwargs.get("use_routing_scales_on_input", False),
            kwargs["enable_pdl"],
            resolved_tactic,
            int(kwargs.get("weight_layout", self.weight_layout)),
            int(self.activation_type),
            kwargs.get("norm_topk_prob", True),
            kwargs.get("routing_replay_out"),
        )

        (
            expert_weights,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            tile_idx,
            mn_limit,
            num_non_exiting_ctas,
            total_num_padded_tokens,
            gemm1_output,
            gemm2_output,
        ) = _torch_views_of_ffi_tensors(routing_out)
        expert_weights = _select_expert_weights(moe_inputs, expert_weights, kwargs)
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
        )

        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()
        routing_input_scales = expert_weights if use_routing_scales_on_input else None

        common_io_kwargs = dict(
            hidden_states=hidden_states,
            gemm1_weights=kwargs["gemm1_weights"],
            gemm1_bias=kwargs.get("gemm1_bias"),
            gemm2_weights=kwargs["gemm2_weights"],
            gemm2_bias=kwargs.get("gemm2_bias"),
            **_gemm1_oa_io_kwargs(kwargs),
            gemm1_output=gemm1_output,
            gemm2_output=gemm2_output,
            output1_scale_scalar=kwargs["output1_scale_scalar"],
            output1_scale_gate_scalar=kwargs["output1_scale_gate_scalar"],
            output2_scale_scalar=kwargs["output2_scale_scalar"],
            tile_idx=tile_idx,
            mn_limit=mn_limit,
            route_map=permuted_idx_to_token_idx,
            num_non_exiting_ctas=num_non_exiting_ctas,
            total_num_padded_tokens=total_num_padded_tokens,
            routed_token_capacity=routed_token_capacity,
            activation_type=int(self.activation_type),
            num_experts=self.num_local_experts,
            num_tokens=num_tokens,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
            per_token_sf_b=routing_input_scales,
        )
        if self._is_partitioned:
            return self._run_partitioned(
                builder=build_fp8_per_tensor_launch_io,
                fc1_cfg=fc1_cfg,
                fc2_cfg=fc2_cfg,
                common_io_kwargs=common_io_kwargs,
                fc1_label="fp8_per_tensor_fc1",
                fc2_label="fp8_per_tensor_fc2",
                gemm1_output=gemm1_output,
                gemm2_output=gemm2_output,
                output=output,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                total_num_padded_tokens=total_num_padded_tokens,
                num_tokens=num_tokens,
                kwargs=kwargs,
                fc1_extra_kwargs={"per_token_sf_a": None},
                fc2_extra_kwargs={"per_token_sf_a": None},
                use_routing_scales_on_input=use_routing_scales_on_input,
            )
        fc1_io = build_fp8_per_tensor_launch_io(
            fc="fc1",
            cfg=fc1_cfg,
            per_token_sf_a=fc1_per_channel_weight_scale,
            **common_io_kwargs,
        )
        fc1_hash = stable_config_hash(fc1_io["cfg"])
        fc1_fn = get_compiled_gemm(fc1_hash, "fp8_per_tensor_fc1", fc1_io, stream)
        fc1_fn(*self._launch_args(fc1_io, stream))

        fc2_io = build_fp8_per_tensor_launch_io(
            fc="fc2",
            cfg=fc2_cfg,
            per_token_sf_a=fc2_per_channel_weight_scale,
            **common_io_kwargs,
        )
        fc2_hash = stable_config_hash(fc2_io["cfg"])
        fc2_fn = get_compiled_gemm(fc2_hash, "fp8_per_tensor_fc2", fc2_io, stream)
        fc2_fn(*self._launch_args(fc2_io, stream))

        if kwargs["do_finalize"]:
            self.moe_op.trtllm_moe_run_finalize(
                gemm2_output,
                output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                kwargs["num_experts"],
                self.top_k,
                self.hidden_size,
                kwargs["enable_pdl"],
                kwargs.get("use_routing_scales_on_input", False),
                None,
            )
            return []
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    @staticmethod
    def _launch_args(io: dict, stream) -> tuple:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

        return _launch_arg_tuple(io, stream)


class PrimsTsFp8BlockScaleMoERunner(_PrimsTsMoERunnerMixin, TunableRunner):
    """Autotuned Prims-TS FP8 block-scale MoE runner."""

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op,
        *,
        top_k: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        fp8_quantization_type: int = Fp8QuantizationType.DeepSeekFp8,
        activation_type: int = ActivationType.Swiglu.value,
        use_shuffled_weight: bool = True,
        weight_layout: int = WeightLayout.MajorK,
        num_experts: Optional[int] = None,
    ):
        self.moe_op = moe_op
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.fp8_quantization_type = Fp8QuantizationType(fp8_quantization_type)
        if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
            self.dtype_act = DtypeTrtllmGen.E4m3
            self.dtype_weights = DtypeTrtllmGen.E4m3
        elif self.fp8_quantization_type == Fp8QuantizationType.MxFp8:
            self.dtype_act = DtypeTrtllmGen.MxE4m3
            self.dtype_weights = DtypeTrtllmGen.MxE4m3
        else:
            raise ValueError(
                f"Unsupported FP8 block-scale quantization: {fp8_quantization_type}"
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.use_shuffled_weight = use_shuffled_weight
        self.weight_layout = WeightLayout(weight_layout)
        self.use_per_token_scaling = False
        self.num_experts = num_experts if num_experts is not None else num_local_experts

    def _map_tactic(self, tactic, moe_inputs, kwargs=None):
        common, bias_and_oa, _ = self._tactic_common(moe_inputs, kwargs)
        if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
            return map_trtllm_deepseek_fp8_moe_tactic(
                tactic,
                sm_arch=common["sm_arch"],
                num_tokens=common["num_tokens"],
                top_k=common["top_k"],
                num_local_experts=common["num_local_experts"],
                num_experts=common["num_experts"],
                weight_layout=common["weight_layout"],
                enable_pdl=common["enable_pdl"],
            )
        return map_trtllm_mxfp8_mxfp8_moe_tactic(tactic, **common, **bias_and_oa)

    def _make_tuning_config(
        self,
        moe_inputs: MoeRunnerInputs,
        tune_max_num_tokens: int = 8192,
        routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
        **kwargs,
    ) -> TuningConfig:
        return make_moe_tuning_config(
            moe_inputs,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            fp8_quantization_type=self.fp8_quantization_type,
            init_packed_topk_ids=_moe_topk_ids_init_for_routing(
                self.num_experts, routing_input_mode
            ),
            tune_max_num_tokens=tune_max_num_tokens,
            keep_empty_expert_weights_static=True,
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        sm_arch = _sm_arch_for_tensor(moe_inputs.hidden_states)
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.num_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            sm_arch,
            False,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsFp8BlockScaleMoERunner.valid_tactics_dict:
            try:
                if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                    valid_tactics = valid_prims_ts_deepseek_fp8_moe_tactics(
                        sm_arch=sm_arch,
                        num_tokens=num_tokens,
                        top_k=self.top_k,
                        num_local_experts=self.num_local_experts,
                        num_experts=self.num_experts,
                        weight_layout=int(self.weight_layout),
                    )
                else:
                    valid_tactics = valid_prims_ts_mxfp8_mxfp8_moe_tactics(
                        sm_arch=sm_arch,
                        activation_type=int(self.activation_type),
                        num_tokens=num_tokens,
                        top_k=self.top_k,
                        num_local_experts=self.num_local_experts,
                        num_experts=self.num_experts,
                        weight_layout=int(self.weight_layout),
                        **gemm_config_flags,
                    )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to enumerate Prims-TS FP8 block-scale MoE tactics"
                ) from exc
            PrimsTsFp8BlockScaleMoERunner.valid_tactics_dict[instance_key] = (
                _with_default_moe_tactic(valid_tactics)
            )
        return self._filter_partition_geometry_tactics(
            PrimsTsFp8BlockScaleMoERunner.valid_tactics_dict[instance_key],
            inputs,
            instance_key,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        del do_preparation
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        requested_tactic = [-1, -1] if tactic == -1 else tactic
        hidden_states = moe_inputs.hidden_states
        output = moe_inputs.output
        num_tokens = hidden_states.shape[0]

        pair = self._map_tactic(requested_tactic, moe_inputs, kwargs)
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_fp8_block_scale_supported(
            self,
            moe_inputs,
            resolved_tactic,
            pair=pair,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS FP8 block-scale kernel ({reason})"
            )

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)
        topk_ids, expert_weights_in = _staged_routing_io(moe_inputs, kwargs)
        routing_out = self.moe_op.trtllm_moe_run_routing_fp8_block_scale(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            topk_ids,
            expert_weights_in,
            hidden_states,
            moe_inputs.hidden_states_scale,
            self._staged_tensor(kwargs, "gemm1_weights"),
            self._staged_tensor(kwargs, "gemm1_weights_scale"),
            self._staged_tensor(kwargs, "gemm2_weights"),
            self._staged_tensor(kwargs, "gemm2_weights_scale"),
            kwargs["num_experts"],
            self.top_k,
            kwargs["n_group"],
            kwargs["topk_group"],
            self.intermediate_size,
            kwargs["local_expert_offset"],
            self.num_local_experts,
            kwargs["routed_scaling_factor"],
            kwargs["routing_method_type"],
            kwargs["enable_pdl"],
            resolved_tactic,
            int(kwargs.get("weight_layout", self.weight_layout)),
            int(self.activation_type),
            int(self.fp8_quantization_type),
            kwargs.get("norm_topk_prob", True),
            kwargs.get("routing_replay_out"),
        )

        (
            expert_weights,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            tile_idx,
            mn_limit,
            num_non_exiting_ctas,
            total_num_padded_tokens,
            gemm1_output,
            gemm1_output_scale,
            activation_output,
            activation_output_scale,
            gemm2_output,
        ) = _torch_views_of_ffi_tensors(routing_out)
        expert_weights = _select_expert_weights(moe_inputs, expert_weights, kwargs)
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
        )

        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()
        hidden_states_scale_for_gemm = moe_inputs.hidden_states_scale
        if self.fp8_quantization_type == Fp8QuantizationType.MxFp8:
            hidden_states_scale_for_gemm = _pad_mxfp8_linear_scale_for_prims(
                moe_inputs.hidden_states_scale,
                num_tokens=num_tokens,
                hidden_size=self.hidden_size,
            )
            activation_output = gemm1_output
            activation_output_scale = gemm1_output_scale

        common_io_kwargs = dict(
            fp8_quantization_type=int(self.fp8_quantization_type),
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale_for_gemm,
            gemm1_weights=kwargs["gemm1_weights"],
            gemm1_weights_scale=kwargs["gemm1_weights_scale"],
            gemm2_weights=kwargs["gemm2_weights"],
            gemm2_weights_scale=kwargs["gemm2_weights_scale"],
            gemm1_bias=kwargs.get("gemm1_bias"),
            gemm2_bias=kwargs.get("gemm2_bias"),
            **_gemm1_oa_io_kwargs(kwargs),
            gemm1_output=gemm1_output,
            gemm1_output_scale=gemm1_output_scale,
            activation_output=activation_output,
            activation_output_scale=activation_output_scale,
            gemm2_output=gemm2_output,
            tile_idx=tile_idx,
            mn_limit=mn_limit,
            route_map=permuted_idx_to_token_idx,
            num_non_exiting_ctas=num_non_exiting_ctas,
            total_num_padded_tokens=total_num_padded_tokens,
            routed_token_capacity=routed_token_capacity,
            activation_type=int(self.activation_type),
            num_experts=self.num_local_experts,
            num_tokens=num_tokens,
            top_k=self.top_k,
            intermediate_size=self.intermediate_size,
            hidden_size=self.hidden_size,
        )
        if self._is_partitioned:
            return self._run_partitioned(
                builder=build_fp8_block_scale_launch_io,
                fc1_cfg=fc1_cfg,
                fc2_cfg=fc2_cfg,
                common_io_kwargs=common_io_kwargs,
                fc1_label="fp8_block_scale_fc1",
                fc2_label="fp8_block_scale_fc2",
                gemm1_output=gemm1_output,
                gemm1_output_scale=gemm1_output_scale,
                gemm2_output=gemm2_output,
                output=output,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                total_num_padded_tokens=total_num_padded_tokens,
                num_tokens=num_tokens,
                kwargs=kwargs,
            )

        fc1_io = build_fp8_block_scale_launch_io(
            fc="fc1", cfg=fc1_cfg, **common_io_kwargs
        )
        fc1_hash = stable_config_hash(fc1_io["cfg"])
        fc1_fn = get_compiled_gemm(fc1_hash, "fp8_block_scale_fc1", fc1_io, stream)
        fc1_fn(*self._launch_args(fc1_io, stream))

        if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
            self.moe_op.trtllm_moe_run_deepseek_fp8_activation(
                gemm1_output,
                gemm1_output_scale,
                activation_output,
                activation_output_scale,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                self.top_k,
                self.intermediate_size,
                int(self.activation_type),
                kwargs["enable_pdl"],
            )

        fc2_io = build_fp8_block_scale_launch_io(
            fc="fc2", cfg=fc2_cfg, **common_io_kwargs
        )
        fc2_hash = stable_config_hash(fc2_io["cfg"])
        fc2_fn = get_compiled_gemm(fc2_hash, "fp8_block_scale_fc2", fc2_io, stream)
        fc2_fn(*self._launch_args(fc2_io, stream))

        if kwargs["do_finalize"]:
            self.moe_op.trtllm_moe_run_finalize(
                gemm2_output,
                output,
                expert_weights,
                expanded_idx_to_permuted_idx,
                total_num_padded_tokens,
                num_tokens,
                kwargs["num_experts"],
                self.top_k,
                self.hidden_size,
                kwargs["enable_pdl"],
                False,
                None,
            )
            return []
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    @staticmethod
    def _launch_args(io: dict, stream) -> tuple:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

        return _launch_arg_tuple(io, stream)


class _PartitionedPrimsTsRunnerMixin:
    """Bind hashable partition configuration while constructing a runner."""

    def __init__(
        self,
        *args,
        partition_strategy: PartitionStrategy,
        partition_fc1_layout: PreparedWeightLayout,
        partition_fc2_layout: PreparedWeightLayout,
        partition_resources: MoePartitionResources | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not _is_gated_activation(int(self.activation_type)):
            raise NotImplementedError(
                "MoE partitioning currently requires a gated activation, got "
                f"{self.activation_type.name}"
            )
        if partition_strategy not in ("tp_n_tp_n", "tp_n_tp_k"):
            raise ValueError(
                f"Unsupported MoE partition strategy {partition_strategy!r}"
            )
        self._partition_strategy = partition_strategy
        self._partition_fc1_layout = partition_fc1_layout
        self._partition_fc2_layout = partition_fc2_layout
        self._partition_topology = _partition_stream_topology(partition_resources)

    def forward(self, inputs: List[torch.Tensor], tactic: Any = -1, **kwargs):
        # The autotuner falls back to -1 without consulting get_valid_tactics, and the
        # unpartitioned default pair may violate the partition geometry.
        if tactic == -1:
            tactic = self.get_valid_tactics(inputs, None)[0]
        return super().forward(inputs, tactic=tactic, **kwargs)


class PrimsTsPartitionedBf16MoERunner(
    _PartitionedPrimsTsRunnerMixin, PrimsTsBf16MoERunner
):
    pass


class PrimsTsPartitionedNvfp4MoERunner(
    _PartitionedPrimsTsRunnerMixin, PrimsTsNvfp4MoERunner
):
    pass


class PrimsTsPartitionedMxfp4Mxfp8MoERunner(
    _PartitionedPrimsTsRunnerMixin, PrimsTsMxfp4Mxfp8MoERunner
):
    pass


class PrimsTsPartitionedMxfp4Bf16MoERunner(
    _PartitionedPrimsTsRunnerMixin, PrimsTsMxfp4Bf16MoERunner
):
    pass


class PrimsTsPartitionedFp8PerTensorMoERunner(
    _PartitionedPrimsTsRunnerMixin, PrimsTsFp8PerTensorMoERunner
):
    pass


class PrimsTsPartitionedFp8BlockScaleMoERunner(
    _PartitionedPrimsTsRunnerMixin, PrimsTsFp8BlockScaleMoERunner
):
    pass
