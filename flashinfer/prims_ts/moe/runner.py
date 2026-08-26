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
from typing import Any, List, Optional

import torch

from flashinfer.autotuner import (
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode
from flashinfer.fused_moe.shared.tuning import (
    make_moe_tuning_config,
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
    build_bf16_launch_io,
    build_fp8_block_scale_launch_io,
    build_fp8_per_tensor_launch_io,
    build_mxfp4_bf16_launch_io,
    build_mxfp4_mxfp8_launch_io,
    build_nvfp4_launch_io,
)


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


def _select_expert_weights(
    moe_inputs: MoeRunnerInputs,
    routed_expert_weights: torch.Tensor | None,
) -> torch.Tensor:
    if moe_inputs.expert_weights is not None and moe_inputs.expert_weights.numel() > 0:
        return moe_inputs.expert_weights
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


def _gemm1_oa_flags_from_kwargs(kwargs: dict) -> dict[str, bool]:
    return {
        "has_gemm1_alpha": kwargs.get("gemm1_alpha") is not None,
        "has_gemm1_beta": kwargs.get("gemm1_beta") is not None,
        "has_gemm1_clamp_limit": kwargs.get("gemm1_clamp_limit") is not None,
    }


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


class _PrimsTsMoERunnerMixin:
    _cache_key_static_extras: tuple = ()

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
        self._cache_key_static_extras = (
            ("enable_pdl", bool(kwargs.get("enable_pdl", False))),
            ("gemm1_bias", kwargs.get("gemm1_bias") is not None),
            ("gemm2_bias", kwargs.get("gemm2_bias") is not None),
            ("gemm1_alpha", kwargs.get("gemm1_alpha") is not None),
            ("gemm1_beta", kwargs.get("gemm1_beta") is not None),
            ("gemm1_clamp_limit", kwargs.get("gemm1_clamp_limit") is not None),
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
            ("prims_ts_moe_config_version", 1),
            ("dtype_act", int(self.dtype_act)),
            ("dtype_weights", int(self.dtype_weights)),
            ("fp8_quantization_type", int(self.fp8_quantization_type)),
            ("activation_type", int(self.activation_type)),
            ("use_per_token_scaling", bool(self.use_per_token_scaling)),
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
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
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
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            has_gemm1_lora_delta,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsBf16MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_bf16_moe_tactics(
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
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
        return PrimsTsBf16MoERunner.valid_tactics_dict[instance_key]

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
        pair = map_trtllm_bf16_moe_tactic(
            requested_tactic,
            activation_type=int(self.activation_type),
            num_tokens=num_tokens,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags_from_kwargs(kwargs),
        )
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_bf16_supported(
            self,
            moe_inputs,
            resolved_tactic,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS BF16 kernel ({reason})"
            )

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        routing_out = self.moe_op.trtllm_moe_run_routing(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            moe_inputs.topk_ids,
            moe_inputs.expert_weights,
            hidden_states,
            kwargs["gemm1_weights"],
            kwargs["gemm2_weights"],
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
        expert_weights = _select_expert_weights(moe_inputs, expert_weights)
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
        )

        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()

        fc1_io = build_bf16_launch_io(
            fc="fc1",
            cfg=fc1_cfg,
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
        fc1_hash = stable_config_hash(fc1_io["cfg"])
        fc1_fn = get_compiled_gemm(fc1_hash, "fc1", fc1_io, stream)
        fc1_fn(*self._launch_args(fc1_io, stream))

        fc2_io = build_bf16_launch_io(
            fc="fc2",
            cfg=fc2_cfg,
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
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
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
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            False,
            uses_per_token_scaling,
            per_token_sf_dtype,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsNvfp4MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_nvfp4_moe_tactics(
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
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
        return PrimsTsNvfp4MoERunner.valid_tactics_dict[instance_key]

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
        pair = map_trtllm_nvfp4_moe_tactic(
            requested_tactic,
            activation_type=int(self.activation_type),
            num_tokens=num_tokens,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            use_per_token_sf_b=uses_per_token_scaling,
            per_token_sf_dtype=(
                _per_token_sf_dtype_value(moe_inputs.per_token_scale)
                if uses_per_token_scaling
                else 1
            ),
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags_from_kwargs(kwargs),
        )
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_nvfp4_supported(
            self,
            moe_inputs,
            resolved_tactic,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS NVFP4 kernel ({reason})"
            )

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)
        routing_logits_for_routing = moe_inputs.routing_logits
        if (
            routing_logits_for_routing is not None
            and routing_logits_for_routing.dtype == torch.float32
            and int(kwargs["routing_method_type"]) == int(RoutingMethodType.DeepSeekV3)
        ):
            routing_logits_for_routing = routing_logits_for_routing.to(torch.bfloat16)

        routing_out = self.moe_op.trtllm_moe_run_routing_fp4_nvfp4(
            routing_logits_for_routing,
            kwargs["routing_bias"],
            moe_inputs.topk_ids,
            moe_inputs.expert_weights,
            hidden_states,
            moe_inputs.hidden_states_scale,
            kwargs["gemm1_weights"],
            kwargs["gemm1_weights_scale"],
            kwargs["gemm2_weights"],
            kwargs["gemm2_weights_scale"],
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
        expert_weights = _select_expert_weights(moe_inputs, expert_weights)
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

        if uses_per_token_scaling and not fc1_cfg.has_epilogue_quant:
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
        fc1_io = build_nvfp4_launch_io(fc="fc1", cfg=fc1_cfg, **common_io_kwargs)
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
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            False,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsMxfp4Mxfp8MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_mxfp4_mxfp8_moe_tactics(
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
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
        return PrimsTsMxfp4Mxfp8MoERunner.valid_tactics_dict[instance_key]

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

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        for tactic in tactics:
            try:
                requested_tactic = [-1, -1] if tactic == -1 else tactic
                pair = map_trtllm_mxfp4_mxfp8_moe_tactic(
                    requested_tactic,
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                    weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
                    fc1_has_bias=kwargs.get("gemm1_bias") is not None,
                    fc2_has_bias=kwargs.get("gemm2_bias") is not None,
                    enable_pdl=bool(kwargs.get("enable_pdl", False)),
                    **_gemm1_oa_flags_from_kwargs(kwargs),
                )
                resolved_tactic = _concrete_tactic(pair)
                ok, reason = is_prims_ts_mxfp4_mxfp8_supported(
                    self,
                    moe_inputs,
                    resolved_tactic,
                    **kwargs,
                )
                if not ok:
                    logger.debug(
                        "[Prims-TS MoE] Skipping MXFP4xMXFP8 precompile for "
                        f"unsupported tactic {tactic}: {reason}"
                    )
                    continue

                routing_out = self.moe_op.trtllm_moe_run_routing_fp4_mxfp4_mxfp8(
                    moe_inputs.routing_logits,
                    kwargs["routing_bias"],
                    moe_inputs.topk_ids,
                    moe_inputs.expert_weights,
                    hidden_states,
                    moe_inputs.hidden_states_scale,
                    kwargs["gemm1_weights"],
                    kwargs["gemm1_weights_scale"],
                    kwargs["gemm2_weights"],
                    kwargs["gemm2_weights_scale"],
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
        pair = map_trtllm_mxfp4_mxfp8_moe_tactic(
            requested_tactic,
            activation_type=int(self.activation_type),
            num_tokens=num_tokens,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags_from_kwargs(kwargs),
        )
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_mxfp4_mxfp8_supported(
            self,
            moe_inputs,
            resolved_tactic,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS MXFP4xMXFP8 kernel ({reason})"
            )

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        routing_out = self.moe_op.trtllm_moe_run_routing_fp4_mxfp4_mxfp8(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            moe_inputs.topk_ids,
            moe_inputs.expert_weights,
            hidden_states,
            moe_inputs.hidden_states_scale,
            kwargs["gemm1_weights"],
            kwargs["gemm1_weights_scale"],
            kwargs["gemm2_weights"],
            kwargs["gemm2_weights_scale"],
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
        expert_weights = _select_expert_weights(moe_inputs, expert_weights)
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
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            False,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsMxfp4Bf16MoERunner.valid_tactics_dict:
            try:
                valid_tactics = valid_prims_ts_mxfp4_bf16_moe_tactics(
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
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
        return PrimsTsMxfp4Bf16MoERunner.valid_tactics_dict[instance_key]

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
        pair = map_trtllm_mxfp4_bf16_moe_tactic(
            requested_tactic,
            activation_type=int(self.activation_type),
            num_tokens=num_tokens,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags_from_kwargs(kwargs),
        )
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_mxfp4_bf16_supported(
            self,
            moe_inputs,
            resolved_tactic,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS MXFP4xBF16 kernel ({reason})"
            )

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        routing_out = self.moe_op.trtllm_moe_run_routing_fp4_mxfp4_bf16(
            moe_inputs.routing_logits,
            kwargs["routing_bias"],
            moe_inputs.topk_ids,
            moe_inputs.expert_weights,
            hidden_states,
            kwargs["gemm1_weights"],
            kwargs["gemm1_weights_scale"],
            kwargs["gemm2_weights"],
            kwargs["gemm2_weights_scale"],
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
        expert_weights = _select_expert_weights(moe_inputs, expert_weights)
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
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
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
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
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
                    activation_type=int(self.activation_type),
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
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
        return PrimsTsFp8PerTensorMoERunner.valid_tactics_dict[instance_key]

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
        ok, reason = is_prims_ts_fp8_per_tensor_supported(
            self,
            moe_inputs,
            requested_tactic,
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
        use_fc1_per_channel_weight_scale = fc1_per_channel_weight_scale is not None
        use_fc2_per_channel_weight_scale = fc2_per_channel_weight_scale is not None
        use_routing_scales_on_input = bool(
            kwargs.get("use_routing_scales_on_input", False)
        )
        per_token_sf_dtype = _fp8_per_tensor_scale_dtype(
            fc1_per_channel_weight_scale_dtype=(
                _per_token_sf_dtype_value(fc1_per_channel_weight_scale)
                if use_fc1_per_channel_weight_scale
                else None
            ),
            fc2_per_channel_weight_scale_dtype=(
                _per_token_sf_dtype_value(fc2_per_channel_weight_scale)
                if use_fc2_per_channel_weight_scale
                else None
            ),
            use_routing_scales_on_input=use_routing_scales_on_input,
            routing_logits=moe_inputs.routing_logits,
        )
        pair = map_trtllm_fp8_per_tensor_moe_tactic(
            requested_tactic,
            activation_type=int(self.activation_type),
            num_tokens=num_tokens,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
            fc1_has_bias=kwargs.get("gemm1_bias") is not None,
            fc2_has_bias=kwargs.get("gemm2_bias") is not None,
            fc1_use_per_token_sf_a=use_fc1_per_channel_weight_scale,
            fc2_use_per_token_sf_a=use_fc2_per_channel_weight_scale,
            use_per_token_sf_b=use_routing_scales_on_input,
            per_token_sf_dtype=per_token_sf_dtype,
            enable_pdl=bool(kwargs.get("enable_pdl", False)),
            **_gemm1_oa_flags_from_kwargs(kwargs),
        )
        resolved_tactic = _concrete_tactic(pair)

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        routing_logits_for_routing = moe_inputs.routing_logits
        if (
            routing_logits_for_routing is not None
            and routing_logits_for_routing.dtype == torch.float32
            and int(kwargs["routing_method_type"]) == int(RoutingMethodType.DeepSeekV3)
        ):
            routing_logits_for_routing = routing_logits_for_routing.to(torch.bfloat16)

        routing_out = self.moe_op.trtllm_moe_run_routing_fp8_per_tensor(
            routing_logits_for_routing,
            kwargs["routing_bias"],
            moe_inputs.topk_ids,
            moe_inputs.expert_weights,
            hidden_states,
            kwargs["gemm1_weights"],
            kwargs["output1_scale_scalar"],
            kwargs["output1_scale_gate_scalar"],
            kwargs["gemm2_weights"],
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
        expert_weights = _select_expert_weights(moe_inputs, expert_weights)
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
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[int]:
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)
        instance_key = (
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            num_tokens,
            False,
            _gemm_config_flags_cache_key(gemm_config_flags),
        )
        if instance_key not in PrimsTsFp8BlockScaleMoERunner.valid_tactics_dict:
            try:
                if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                    valid_tactics = valid_prims_ts_deepseek_fp8_moe_tactics(
                        num_tokens=num_tokens,
                        top_k=self.top_k,
                        num_local_experts=self.num_local_experts,
                        weight_layout=int(self.weight_layout),
                    )
                else:
                    valid_tactics = valid_prims_ts_mxfp8_mxfp8_moe_tactics(
                        activation_type=int(self.activation_type),
                        num_tokens=num_tokens,
                        top_k=self.top_k,
                        num_local_experts=self.num_local_experts,
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
        return PrimsTsFp8BlockScaleMoERunner.valid_tactics_dict[instance_key]

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

        if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
            pair = map_trtllm_deepseek_fp8_moe_tactic(
                requested_tactic,
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
                enable_pdl=bool(kwargs.get("enable_pdl", False)),
            )
        else:
            pair = map_trtllm_mxfp8_mxfp8_moe_tactic(
                requested_tactic,
                activation_type=int(self.activation_type),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(kwargs.get("weight_layout", self.weight_layout)),
                fc1_has_bias=kwargs.get("gemm1_bias") is not None,
                fc2_has_bias=kwargs.get("gemm2_bias") is not None,
                enable_pdl=bool(kwargs.get("enable_pdl", False)),
                **_gemm1_oa_flags_from_kwargs(kwargs),
            )
        resolved_tactic = _concrete_tactic(pair)
        ok, reason = is_prims_ts_fp8_block_scale_supported(
            self,
            moe_inputs,
            resolved_tactic,
            **kwargs,
        )
        if not ok:
            raise RuntimeError(
                f"Config not supported by Prims-TS FP8 block-scale kernel ({reason})"
            )

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)
        routing_logits_for_routing = moe_inputs.routing_logits
        if (
            routing_logits_for_routing is not None
            and routing_logits_for_routing.dtype == torch.float32
            and int(kwargs["routing_method_type"]) == int(RoutingMethodType.DeepSeekV3)
        ):
            routing_logits_for_routing = routing_logits_for_routing.to(torch.bfloat16)

        routing_out = self.moe_op.trtllm_moe_run_routing_fp8_block_scale(
            routing_logits_for_routing,
            kwargs["routing_bias"],
            moe_inputs.topk_ids,
            moe_inputs.expert_weights,
            hidden_states,
            moe_inputs.hidden_states_scale,
            kwargs["gemm1_weights"],
            kwargs["gemm1_weights_scale"],
            kwargs["gemm2_weights"],
            kwargs["gemm2_weights_scale"],
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
        expert_weights = _select_expert_weights(moe_inputs, expert_weights)
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
            )
            return []
        return [gemm2_output, expert_weights, expanded_idx_to_permuted_idx]

    @staticmethod
    def _launch_args(io: dict, stream) -> tuple:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

        return _launch_arg_tuple(io, stream)
