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
from collections.abc import Callable, Sequence
from typing import Any, Generic, List, Optional, TypeVar

import torch

from flashinfer.autotuner import (
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode
from flashinfer.fused_moe.tactic_search import FactorizedTacticSpace, MoeTactic
from flashinfer.fused_moe.shared.tuning import (
    MoeTensorInitializer,
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
    PrimsTsGemmPair,
    valid_prims_ts_bf16_moe_tactics,
    valid_prims_ts_deepseek_fp8_moe_tactics,
    valid_prims_ts_fp8_per_tensor_moe_tactics,
    valid_prims_ts_mxfp4_bf16_moe_tactics,
    valid_prims_ts_mxfp4_mxfp8_moe_tactics,
    valid_prims_ts_mxfp8_mxfp8_moe_tactics,
    valid_prims_ts_nvfp4_moe_tactics,
)
from .da_body import (
    PrimsTsBf16BodyWorkspace,
    PrimsTsBodyExecution,
    PrimsTsBodyRouting,
    PrimsTsBodySource,
    PrimsTsBodyWorkspace,
    PrimsTsDeepSeekFp8BodyWorkspace,
    PrimsTsFp8PerTensorBodyWorkspace,
    PrimsTsMxfp4Bf16BodyWorkspace,
    PrimsTsMxfp4Mxfp8BodyWorkspace,
    PrimsTsMxfp8BodyWorkspace,
    PrimsTsNvfp4BodyWorkspace,
    PrimsTsOrdinaryBodySource,
    PrimsTsPreparedBodySource,
    PrimsTsRoutingMetadata,
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
    has_precomputed_routing = (
        moe_inputs.routing_logits is None or moe_inputs.routing_logits.numel() == 0
    )
    if (
        has_precomputed_routing
        and moe_inputs.expert_weights is not None
        and moe_inputs.expert_weights.numel() > 0
    ):
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


def _decode_routing_outputs(
    moe_inputs: MoeRunnerInputs, routing_out: Sequence[Any]
) -> tuple[torch.Tensor, ...]:
    """Decode reusable native outputs while preserving caller-owned expert weights."""
    routing_views = _torch_views_of_ffi_tensors(routing_out)
    expert_weights = _select_expert_weights(moe_inputs, routing_views[0])
    return (
        expert_weights,
        *routing_views[1:],
    )


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


def _filter_valid_moe_tactics(
    valid_tactics: Sequence[MoeTactic],
    map_tactic: Callable[[MoeTactic], PrimsTsGemmPair],
) -> List[MoeTactic]:
    filtered_tactics: List[MoeTactic] = []
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


def _with_default_moe_tactic(
    valid_tactics: Sequence[MoeTactic],
) -> List[MoeTactic]:
    return [-1, *[tactic for tactic in valid_tactics if tactic != -1]]


def _concrete_tactic(pair: PrimsTsGemmPair) -> list[int]:
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
    scale: torch.Tensor,
    *,
    num_tokens: int,
    hidden_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pad compact MXFP8 token scales into caller-owned or new LINEAR storage."""

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

    padded = output
    if padded is None:
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
    activation_output: torch.Tensor | None = None,
    activation_output_scale: torch.Tensor | None = None,
    per_token_scale_fc2: torch.Tensor | None = None,
    launch: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize routed BF16 FC1 output into caller-owned or newly allocated FC2 inputs."""

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

    if activation_output is None:
        activation_output = torch.empty(
            (max_padded_tokens, intermediate_size // 2),
            dtype=torch.uint8,
            device=gemm1_output.device,
        )
    else:
        required_activation_values = max_padded_tokens * (intermediate_size // 2)
        activation_output = activation_output.flatten()[
            :required_activation_values
        ].view(max_padded_tokens, intermediate_size // 2)
    if per_token_scale_fc2 is None:
        per_token_scale_fc2 = torch.empty(
            (max_padded_tokens,), dtype=torch.float32, device=gemm1_output.device
        )
    else:
        per_token_scale_fc2 = per_token_scale_fc2.flatten()[:max_padded_tokens]

    sf_row_tile = 128 if tile_n >= 128 else 8
    sf_rows = round_up(max_padded_tokens, sf_row_tile)
    sf_cols = round_up(intermediate_size // 16, 4)
    required_sf_values = sf_rows * sf_cols
    if gemm1_output_scale.numel() < required_sf_values:
        raise ValueError(
            "NVFP4 activation scale buffer is too small: "
            f"need {required_sf_values}, got {gemm1_output_scale.numel()}"
        )
    if activation_output_scale is None:
        activation_output_scale = gemm1_output_scale[:required_sf_values].view(
            sf_rows, sf_cols
        )
    else:
        activation_output_scale = activation_output_scale.flatten()[
            :required_sf_values
        ].view(sf_rows, sf_cols)

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

    if launch:
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


BodyWorkspaceT = TypeVar("BodyWorkspaceT", bound=PrimsTsBodyWorkspace)


class _PrimsTsMoERunnerMixin(Generic[BodyWorkspaceT]):
    # PrimsTS exposes complete legal FC1/FC2 pairs for ordinary coordinate search.
    use_factorized_moe_tactic_search = True
    # Exact dtype-specific record used to decode prepared body workspace tensors.
    body_workspace_type: type[BodyWorkspaceT]
    # Immutable cache-key dimensions supplied by the active public MoE wrapper.
    _cache_key_static_extras: tuple = ()

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Run the ordinary routed PrimsTS MoE path through its exact dtype ABI."""
        return self._execute_body_source(
            inputs,
            tactic=tactic,
            do_preparation=do_preparation,
            body_source=PrimsTsOrdinaryBodySource(),
            **kwargs,
        )

    def prepare_body_workspace(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Allocate one tactic's exact body workspace outside CUDA Graph capture."""
        return self._execute_body_source(
            inputs,
            tactic=tactic,
            do_preparation=True,
            body_source=PrimsTsOrdinaryBodySource(),
            **kwargs,
        )

    def _body_workspace_from_sequence(
        self, tensors: Sequence[torch.Tensor]
    ) -> BodyWorkspaceT:
        """Decode prepared tensors through this runner's exact body ABI record."""
        return self.body_workspace_type.from_sequence(tensors)

    def run_prepared_body(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic,
        routing_metadata: Sequence[torch.Tensor],
        body_workspace: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Launch one body from graph-live routing and preallocated typed storage."""
        routing = PrimsTsRoutingMetadata.from_sequence(routing_metadata).body_view()
        workspace = self._body_workspace_from_sequence(body_workspace)
        source = PrimsTsPreparedBodySource(
            PrimsTsBodyExecution(routing=routing, workspace=workspace)
        )
        return self._execute_body_source(
            inputs,
            tactic=tactic,
            do_preparation=False,
            body_source=source,
            **kwargs,
        )

    def _topk_ids_initializer(
        self, moe_inputs: MoeRunnerInputs
    ) -> MoeTensorInitializer:
        """Return the cached initializer matching the PrimsTS routed body ABI."""
        expert_weights = moe_inputs.expert_weights
        uses_unpacked_routing = (
            expert_weights is not None and expert_weights.numel() > 0
        )
        return moe_topk_ids_init(
            self.num_experts,
            packed=not uses_unpacked_routing,
        )

    def set_cache_key_static_extras(self, **kwargs: Any) -> None:
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
        """Return cache dimensions that are invariant under input synthesis."""
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        static_extras = getattr(self, "_cache_key_static_extras", ())
        routing_input_mode = RoutingInputMode(
            dict(static_extras).get(
                "routing_input_mode", int(RoutingInputMode.FromLogits)
            )
        )
        return (
            ("prims_ts_moe_config_version", 4),
            ("dtype_act", int(self.dtype_act)),
            ("dtype_weights", int(self.dtype_weights)),
            ("fp8_quantization_type", int(self.fp8_quantization_type)),
            ("activation_type", int(self.activation_type)),
            ("use_per_token_scaling", bool(self.use_per_token_scaling)),
            ("per_token_scale", moe_inputs.per_token_scale is not None),
            ("gemm1_lora_delta", moe_inputs.gemm1_lora_delta is not None),
            (
                "routing_logits",
                routing_input_mode == RoutingInputMode.FromLogits,
            ),
            (
                "expert_weights",
                routing_input_mode == RoutingInputMode.UnpackedPrecomputed,
            ),
            *static_extras,
        )

    def _factorized_tactic_space(
        self,
        inputs: List[torch.Tensor],
        resolve_pair: Callable[[MoeTactic], PrimsTsGemmPair],
    ) -> FactorizedTacticSpace:
        """Build legal fused-MoE coordinates from complete PrimsTS config rows."""
        from flashinfer.fused_moe.tactic_search import (
            FactorizedTactic,
            FactorizedTacticSpace,
        )

        tactics = []
        anchors = {}
        # Each raw tactic remains the public `[tile_n, moe_config_index]` identity. Factor
        # coordinates only expose its resolved FC1/FC2 rows to the shared search algorithm.
        for raw_tactic in self.get_valid_tactics(inputs, None):  # type: ignore[arg-type]
            if raw_tactic == -1:
                continue
            pair = resolve_pair(raw_tactic)
            identity = (int(pair.tile_n), int(pair.moe_config_index))
            tactics.append(
                FactorizedTactic(
                    tactic=identity,
                    tile_n=pair.tile_n,
                    fc1=pair.fc1.prims_ts_gemm_config_index,
                    fc2=pair.fc2.prims_ts_gemm_config_index,
                    public_tactic=(
                        tuple(raw_tactic)
                        if isinstance(raw_tactic, list)
                        else raw_tactic
                    ),
                )
            )
            anchors.setdefault(pair.tile_n, identity)
        return FactorizedTacticSpace(tactics, anchors)

    def precompile_tactics(
        self,
        inputs: List[torch.Tensor],
        tactics: List[MoeTactic],
        profile: OptimizationProfile,
        **kwargs: Any,
    ) -> bool:
        del profile
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        hidden_states = moe_inputs.hidden_states

        def _precompile_one(tactic: MoeTactic) -> None:
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

        # CUTLASS DSL owns an MLIR location/context on the invoking thread. Compiling in worker
        # threads can escape Python exception handling and abort the process, so keep this phase
        # deterministic on the autotuner's caller thread; GPU profiling remains unchanged.
        for tactic in tactics:
            _precompile_one(tactic)
        return True


class PrimsTsBf16MoERunner(
    _PrimsTsMoERunnerMixin[PrimsTsBf16BodyWorkspace], TunableRunner
):
    """Autotuned Prims-TS BF16 MoE runner using shared TRT-LLM routing/finalize."""

    # Exact intermediate ABI used by BF16 prepared bodies.
    body_workspace_type = PrimsTsBf16BodyWorkspace

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op: Any,
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
    ) -> None:
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
    ) -> List[MoeTactic]:
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

    def get_factorized_tactic_space(
        self, inputs: List[torch.Tensor]
    ) -> FactorizedTacticSpace:
        """Return legal BF16 FC1/FC2 factors and deterministic tile anchors."""
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        flags = _gemm_config_flags_from_static_extras(self)

        def resolve_pair(raw_tactic: MoeTactic) -> PrimsTsGemmPair:
            """Resolve one complete public BF16 tactic into its paired configs."""
            return map_trtllm_bf16_moe_tactic(
                raw_tactic,
                activation_type=int(self.activation_type),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(self.weight_layout),
                fc1_has_bias=flags["fc1_has_bias"],
                fc2_has_bias=flags["fc2_has_bias"],
                enable_pdl=dict(self._cache_key_static_extras).get("enable_pdl", False),
                **{
                    name: flags[name]
                    for name in (
                        "has_gemm1_alpha",
                        "has_gemm1_beta",
                        "has_gemm1_clamp_limit",
                    )
                },
            )

        return self._factorized_tactic_space(inputs, resolve_pair)

    def _execute_body_source(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic = -1,
        do_preparation: bool = False,
        *,
        body_source: PrimsTsBodySource[PrimsTsBf16BodyWorkspace],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Execute BF16 from ordinary routing or one prepared typed body source."""
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

        def route_and_allocate() -> PrimsTsBodyExecution[PrimsTsBf16BodyWorkspace]:
            """Route ordinarily and package the resulting BF16 body ABI."""
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
            ) = _decode_routing_outputs(moe_inputs, routing_out)
            return PrimsTsBodyExecution(
                routing=PrimsTsBodyRouting(
                    expert_weights=expert_weights,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_token_idx=permuted_idx_to_token_idx,
                    tile_idx=tile_idx,
                    mn_limit=mn_limit,
                    num_non_exiting_ctas=num_non_exiting_ctas,
                    total_num_padded_tokens=total_num_padded_tokens,
                ),
                workspace=PrimsTsBf16BodyWorkspace(gemm1_output, gemm2_output),
            )

        # The source either performs ordinary routing now or supplies a graph-stable DA record.
        execution = body_source.resolve(route_and_allocate)
        routing_metadata = execution.routing
        body_workspace = execution.workspace
        expert_weights = routing_metadata.expert_weights
        expanded_idx_to_permuted_idx = routing_metadata.expanded_idx_to_permuted_idx
        permuted_idx_to_token_idx = routing_metadata.permuted_idx_to_token_idx
        tile_idx = routing_metadata.tile_idx
        mn_limit = routing_metadata.mn_limit
        num_non_exiting_ctas = routing_metadata.num_non_exiting_ctas
        total_num_padded_tokens = routing_metadata.total_num_padded_tokens
        gemm1_output = body_workspace.gemm1_output
        gemm2_output = body_workspace.gemm2_output
        if do_preparation:
            return list(body_workspace.tensors())
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


class PrimsTsNvfp4MoERunner(
    _PrimsTsMoERunnerMixin[PrimsTsNvfp4BodyWorkspace], TunableRunner
):
    """Autotuned Prims-TS NVFP4xNVFP4 MoE runner."""

    # Exact intermediate ABI used by NVFP4 prepared bodies.
    body_workspace_type = PrimsTsNvfp4BodyWorkspace

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op: Any,
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
    ) -> None:
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
    ) -> List[MoeTactic]:
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

    def get_factorized_tactic_space(
        self, inputs: List[torch.Tensor]
    ) -> FactorizedTacticSpace:
        """Return legal NVFP4 FC1/FC2 factors and deterministic tile anchors."""
        from flashinfer.fused_moe.tactic_search import (
            FactorizedTactic,
            FactorizedTacticSpace,
        )

        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        uses_per_token_scaling = moe_inputs.per_token_scale is not None
        per_token_sf_dtype = (
            _per_token_sf_dtype_value(moe_inputs.per_token_scale)
            if uses_per_token_scaling
            else 1
        )
        gemm_config_flags = _gemm_config_flags_from_static_extras(self)

        # Resolve only enumerated complete tactics. The factorized search may compose FC1/FC2
        # coordinates only when that composition maps back to one of these legal config rows.
        tactics = []
        anchors = {}
        for raw_tactic in self.get_valid_tactics(inputs, None):  # type: ignore[arg-type]
            if raw_tactic == -1:
                continue
            pair = map_trtllm_nvfp4_moe_tactic(
                raw_tactic,
                activation_type=int(self.activation_type),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(self.weight_layout),
                fc1_has_bias=gemm_config_flags["fc1_has_bias"],
                fc2_has_bias=gemm_config_flags["fc2_has_bias"],
                has_gemm1_alpha=gemm_config_flags["has_gemm1_alpha"],
                has_gemm1_beta=gemm_config_flags["has_gemm1_beta"],
                has_gemm1_clamp_limit=gemm_config_flags["has_gemm1_clamp_limit"],
                use_per_token_sf_b=uses_per_token_scaling,
                per_token_sf_dtype=per_token_sf_dtype,
                enable_pdl=dict(self._cache_key_static_extras).get("enable_pdl", False),
            )
            identity = (int(pair.tile_n), int(pair.moe_config_index))
            tactics.append(
                FactorizedTactic(
                    tactic=identity,
                    tile_n=pair.tile_n,
                    fc1=pair.fc1.prims_ts_gemm_config_index,
                    fc2=pair.fc2.prims_ts_gemm_config_index,
                    public_tactic=(
                        tuple(raw_tactic)
                        if isinstance(raw_tactic, list)
                        else raw_tactic
                    ),
                )
            )
            anchors.setdefault(pair.tile_n, identity)
        return FactorizedTacticSpace(tactics, anchors)

    def _execute_body_source(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic = -1,
        do_preparation: bool = False,
        *,
        body_source: PrimsTsBodySource[PrimsTsNvfp4BodyWorkspace],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Execute NVFP4 from ordinary routing or one prepared typed body source."""
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
        fc1_cfg = pair.fc1.cfg.build()
        fc2_cfg = pair.fc2.cfg.build()

        import cuda.bindings.driver as cuda_drv

        torch_stream = torch.cuda.current_stream(device=hidden_states.device)
        stream = cuda_drv.CUstream(torch_stream.cuda_stream)

        def route_and_allocate() -> PrimsTsBodyExecution[PrimsTsNvfp4BodyWorkspace]:
            """Route ordinarily and package the tactic-dependent NVFP4 body ABI."""
            # Ordinary execution and out-of-capture preparation use the established router to
            # allocate the exact NVFP4 body ABI. Preparation returns before launching either GEMM.
            routing_out = self.moe_op.trtllm_moe_run_routing_fp4_nvfp4(
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
            ) = _decode_routing_outputs(moe_inputs, routing_out)
            empty_quantized_field = torch.empty(
                0, dtype=torch.uint8, device=hidden_states.device
            )
            empty_bf16_field = torch.empty(
                0, dtype=torch.bfloat16, device=hidden_states.device
            )
            if gemm1_output.dtype == torch.bfloat16:
                gemm1_output_quantized = empty_quantized_field
                gemm1_output_bf16 = gemm1_output
            else:
                gemm1_output_quantized = gemm1_output
                gemm1_output_bf16 = empty_bf16_field
            return PrimsTsBodyExecution(
                routing=PrimsTsBodyRouting(
                    expert_weights=expert_weights,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_token_idx=permuted_idx_to_token_idx,
                    tile_idx=tile_idx,
                    mn_limit=mn_limit,
                    num_non_exiting_ctas=num_non_exiting_ctas,
                    total_num_padded_tokens=total_num_padded_tokens,
                ),
                workspace=PrimsTsNvfp4BodyWorkspace(
                    gemm1_output_quantized,
                    gemm1_output_bf16,
                    gemm1_output_scale,
                    gemm2_output,
                    empty_quantized_field,
                    empty_quantized_field,
                    empty_quantized_field,
                    empty_quantized_field,
                ),
            )

        # The prepared source bypasses routing and preserves every lane-owned pointer.
        execution = body_source.resolve(route_and_allocate)
        routing_metadata = execution.routing
        body_workspace = execution.workspace
        expert_weights = routing_metadata.expert_weights
        expanded_idx_to_permuted_idx = routing_metadata.expanded_idx_to_permuted_idx
        permuted_idx_to_token_idx = routing_metadata.permuted_idx_to_token_idx
        tile_idx = routing_metadata.tile_idx
        mn_limit = routing_metadata.mn_limit
        num_non_exiting_ctas = routing_metadata.num_non_exiting_ctas
        total_num_padded_tokens = routing_metadata.total_num_padded_tokens
        gemm1_output = (
            body_workspace.gemm1_output_bf16
            if uses_per_token_scaling and not fc1_cfg.has_epilogue_quant
            else body_workspace.gemm1_output_quantized
        )
        gemm1_output_scale = body_workspace.gemm1_output_scale
        gemm2_output = body_workspace.gemm2_output

        # Some per-token configurations retain BF16 FC1 output instead of the router's packed
        # allocation. Materialize that exact tactic ABI before publishing a preparation workspace.
        if (
            not body_source.preallocated
            and uses_per_token_scaling
            and not fc1_cfg.has_epilogue_quant
        ):
            gemm1_output = torch.empty(
                (int(gemm1_output.shape[0]), self.intermediate_size),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            body_workspace = PrimsTsNvfp4BodyWorkspace(
                body_workspace.gemm1_output_quantized,
                gemm1_output,
                gemm1_output_scale,
                gemm2_output,
                body_workspace.activation_output,
                body_workspace.activation_output_scale,
                body_workspace.per_token_scale_fc2,
                body_workspace.expert_weights_bf16,
            )
        # Per-token FC2 quantization owns three additional graph-stable destinations. Allocate
        # them during preparation, then reuse those exact lane pointers during child capture.
        if uses_per_token_scaling and fc2_cfg.has_per_token_sf_b:
            preallocated = body_source.preallocated
            activation_output, activation_output_scale, per_token_scale_fc2 = (
                _quantize_nvfp4_fc1_output_for_fc2(
                    gemm1_output=gemm1_output,
                    gemm1_output_scale=gemm1_output_scale,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    intermediate_size=self.intermediate_size,
                    tile_n=pair.tile_n,
                    activation_output=(
                        body_workspace.activation_output if preallocated else None
                    ),
                    activation_output_scale=(
                        body_workspace.activation_output_scale if preallocated else None
                    ),
                    per_token_scale_fc2=(
                        body_workspace.per_token_scale_fc2 if preallocated else None
                    ),
                    launch=False,
                )
            )
            body_workspace = PrimsTsNvfp4BodyWorkspace(
                body_workspace.gemm1_output_quantized,
                body_workspace.gemm1_output_bf16,
                gemm1_output_scale,
                gemm2_output,
                activation_output,
                activation_output_scale,
                per_token_scale_fc2,
                body_workspace.expert_weights_bf16,
            )
        expert_weights = _select_expert_weights(moe_inputs, expert_weights)
        if expert_weights.dtype != torch.bfloat16 and not body_source.preallocated:
            expert_weights_bf16 = torch.empty_like(expert_weights, dtype=torch.bfloat16)
            body_workspace = PrimsTsNvfp4BodyWorkspace(
                body_workspace.gemm1_output_quantized,
                body_workspace.gemm1_output_bf16,
                body_workspace.gemm1_output_scale,
                body_workspace.gemm2_output,
                body_workspace.activation_output,
                body_workspace.activation_output_scale,
                body_workspace.per_token_scale_fc2,
                expert_weights_bf16,
            )
        if do_preparation:
            return list(body_workspace.tensors())

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
            body_workspace.expert_weights_bf16.copy_(expert_weights)
            expert_weights = body_workspace.expert_weights_bf16
        routed_token_capacity = _routed_token_capacity(
            self,
            moe_inputs,
            resolved_tactic,
            total_num_padded_tokens,
            kwargs,
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
            _quantize_nvfp4_fc1_output_for_fc2(
                gemm1_output=gemm1_output,
                gemm1_output_scale=gemm1_output_scale,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                num_tokens=num_tokens,
                top_k=self.top_k,
                intermediate_size=self.intermediate_size,
                tile_n=pair.tile_n,
                activation_output=activation_output,
                activation_output_scale=activation_output_scale,
                per_token_scale_fc2=per_token_scale_fc2,
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


class PrimsTsMxfp4Mxfp8MoERunner(
    _PrimsTsMoERunnerMixin[PrimsTsMxfp4Mxfp8BodyWorkspace], TunableRunner
):
    """Autotuned Prims-TS MXFP4xMXFP8 MoE runner."""

    # Exact intermediate ABI used by MXFP4xMXFP8 prepared bodies.
    body_workspace_type = PrimsTsMxfp4Mxfp8BodyWorkspace

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op: Any,
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
    ) -> None:
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
            **kwargs,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[MoeTactic]:
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

    def get_factorized_tactic_space(
        self, inputs: List[torch.Tensor]
    ) -> FactorizedTacticSpace:
        """Return legal MXFP4xMXFP8 FC1/FC2 factors and tile anchors."""
        num_tokens = MoeRunnerInputs.from_list(inputs).hidden_states.shape[0]
        flags = _gemm_config_flags_from_static_extras(self)

        def resolve_pair(raw_tactic: MoeTactic) -> PrimsTsGemmPair:
            """Resolve one complete public MXFP4xMXFP8 tactic row."""
            return map_trtllm_mxfp4_mxfp8_moe_tactic(
                raw_tactic,
                activation_type=int(self.activation_type),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(self.weight_layout),
                fc1_has_bias=flags["fc1_has_bias"],
                fc2_has_bias=flags["fc2_has_bias"],
                enable_pdl=dict(self._cache_key_static_extras).get("enable_pdl", False),
                **{
                    name: flags[name]
                    for name in (
                        "has_gemm1_alpha",
                        "has_gemm1_beta",
                        "has_gemm1_clamp_limit",
                    )
                },
            )

        return self._factorized_tactic_space(inputs, resolve_pair)

    def precompile_tactics(
        self,
        inputs: List[torch.Tensor],
        tactics: List[MoeTactic],
        profile: OptimizationProfile,
        **kwargs: Any,
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

    def _execute_body_source(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic = -1,
        do_preparation: bool = False,
        *,
        body_source: PrimsTsBodySource[PrimsTsMxfp4Mxfp8BodyWorkspace],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Execute MXFP4xMXFP8 from ordinary or prepared typed body inputs."""
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

        def route_and_allocate() -> PrimsTsBodyExecution[
            PrimsTsMxfp4Mxfp8BodyWorkspace
        ]:
            """Route ordinarily and package the MXFP4xMXFP8 body ABI."""
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
            ) = _decode_routing_outputs(moe_inputs, routing_out)
            return PrimsTsBodyExecution(
                routing=PrimsTsBodyRouting(
                    expert_weights=expert_weights,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_token_idx=permuted_idx_to_token_idx,
                    tile_idx=tile_idx,
                    mn_limit=mn_limit,
                    num_non_exiting_ctas=num_non_exiting_ctas,
                    total_num_padded_tokens=total_num_padded_tokens,
                ),
                workspace=PrimsTsMxfp4Mxfp8BodyWorkspace(
                    gemm1_output, gemm1_output_scale, gemm2_output
                ),
            )

        # Resolve ordinary routing or reuse the graph-stable record supplied by DA.
        execution = body_source.resolve(route_and_allocate)
        routing_metadata = execution.routing
        body_workspace = execution.workspace
        expert_weights = routing_metadata.expert_weights
        expanded_idx_to_permuted_idx = routing_metadata.expanded_idx_to_permuted_idx
        permuted_idx_to_token_idx = routing_metadata.permuted_idx_to_token_idx
        tile_idx = routing_metadata.tile_idx
        mn_limit = routing_metadata.mn_limit
        num_non_exiting_ctas = routing_metadata.num_non_exiting_ctas
        total_num_padded_tokens = routing_metadata.total_num_padded_tokens
        gemm1_output = body_workspace.gemm1_output
        gemm1_output_scale = body_workspace.gemm1_output_scale
        gemm2_output = body_workspace.gemm2_output
        if do_preparation:
            return list(body_workspace.tensors())
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


class PrimsTsMxfp4Bf16MoERunner(
    _PrimsTsMoERunnerMixin[PrimsTsMxfp4Bf16BodyWorkspace], TunableRunner
):
    """Autotuned Prims-TS MXFP4xBF16 MoE runner."""

    # Exact intermediate ABI used by MXFP4xBF16 prepared bodies.
    body_workspace_type = PrimsTsMxfp4Bf16BodyWorkspace

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op: Any,
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
    ) -> None:
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
    ) -> List[MoeTactic]:
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

    def get_factorized_tactic_space(
        self, inputs: List[torch.Tensor]
    ) -> FactorizedTacticSpace:
        """Return legal MXFP4xBF16 FC1/FC2 factors and tile anchors."""
        num_tokens = MoeRunnerInputs.from_list(inputs).hidden_states.shape[0]
        flags = _gemm_config_flags_from_static_extras(self)

        def resolve_pair(raw_tactic: MoeTactic) -> PrimsTsGemmPair:
            """Resolve one complete public MXFP4xBF16 tactic row."""
            return map_trtllm_mxfp4_bf16_moe_tactic(
                raw_tactic,
                activation_type=int(self.activation_type),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(self.weight_layout),
                fc1_has_bias=flags["fc1_has_bias"],
                fc2_has_bias=flags["fc2_has_bias"],
                enable_pdl=dict(self._cache_key_static_extras).get("enable_pdl", False),
                **{
                    name: flags[name]
                    for name in (
                        "has_gemm1_alpha",
                        "has_gemm1_beta",
                        "has_gemm1_clamp_limit",
                    )
                },
            )

        return self._factorized_tactic_space(inputs, resolve_pair)

    def _execute_body_source(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic = -1,
        do_preparation: bool = False,
        *,
        body_source: PrimsTsBodySource[PrimsTsMxfp4Bf16BodyWorkspace],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Execute MXFP4xBF16 from ordinary or prepared typed body inputs."""
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

        def route_and_allocate() -> PrimsTsBodyExecution[PrimsTsMxfp4Bf16BodyWorkspace]:
            """Route ordinarily and package the MXFP4xBF16 body ABI."""
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
            ) = _decode_routing_outputs(moe_inputs, routing_out)
            return PrimsTsBodyExecution(
                routing=PrimsTsBodyRouting(
                    expert_weights=expert_weights,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_token_idx=permuted_idx_to_token_idx,
                    tile_idx=tile_idx,
                    mn_limit=mn_limit,
                    num_non_exiting_ctas=num_non_exiting_ctas,
                    total_num_padded_tokens=total_num_padded_tokens,
                ),
                workspace=PrimsTsMxfp4Bf16BodyWorkspace(gemm1_output, gemm2_output),
            )

        # Resolve ordinary routing or reuse the graph-stable record supplied by DA.
        execution = body_source.resolve(route_and_allocate)
        routing_metadata = execution.routing
        body_workspace = execution.workspace
        expert_weights = routing_metadata.expert_weights
        expanded_idx_to_permuted_idx = routing_metadata.expanded_idx_to_permuted_idx
        permuted_idx_to_token_idx = routing_metadata.permuted_idx_to_token_idx
        tile_idx = routing_metadata.tile_idx
        mn_limit = routing_metadata.mn_limit
        num_non_exiting_ctas = routing_metadata.num_non_exiting_ctas
        total_num_padded_tokens = routing_metadata.total_num_padded_tokens
        gemm1_output = body_workspace.gemm1_output
        gemm2_output = body_workspace.gemm2_output
        if do_preparation:
            return list(body_workspace.tensors())
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


class PrimsTsFp8PerTensorMoERunner(
    _PrimsTsMoERunnerMixin[PrimsTsFp8PerTensorBodyWorkspace], TunableRunner
):
    """Autotuned Prims-TS FP8 per-tensor MoE runner."""

    # Exact intermediate ABI used by FP8 per-tensor prepared bodies.
    body_workspace_type = PrimsTsFp8PerTensorBodyWorkspace

    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op: Any,
        *,
        top_k: int,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation_type: int = ActivationType.Swiglu.value,
        use_shuffled_weight: bool = True,
        weight_layout: int = WeightLayout.MajorK,
        num_experts: Optional[int] = None,
    ) -> None:
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
    ) -> List[MoeTactic]:
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

    def get_factorized_tactic_space(
        self, inputs: List[torch.Tensor]
    ) -> FactorizedTacticSpace:
        """Return legal FP8 per-tensor FC1/FC2 factors and tile anchors."""
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        num_tokens = moe_inputs.hidden_states.shape[0]
        extras = dict(self._cache_key_static_extras)
        flags = _gemm_config_flags_from_static_extras(self)
        use_fc1_scale = bool(extras.get("fc1_per_channel_weight_scale", False))
        use_fc2_scale = bool(extras.get("fc2_per_channel_weight_scale", False))
        use_routing_scale = bool(extras.get("use_routing_scales_on_input", False))
        scale_dtype = int(extras.get("per_token_sf_dtype", 1))

        def resolve_pair(raw_tactic: MoeTactic) -> PrimsTsGemmPair:
            """Resolve one complete public FP8 per-tensor tactic row."""
            return map_trtllm_fp8_per_tensor_moe_tactic(
                raw_tactic,
                activation_type=int(self.activation_type),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(self.weight_layout),
                fc1_has_bias=flags["fc1_has_bias"],
                fc2_has_bias=flags["fc2_has_bias"],
                fc1_use_per_token_sf_a=use_fc1_scale,
                fc2_use_per_token_sf_a=use_fc2_scale,
                use_per_token_sf_b=use_routing_scale,
                per_token_sf_dtype=scale_dtype,
                enable_pdl=bool(extras.get("enable_pdl", False)),
                **{
                    name: flags[name]
                    for name in (
                        "has_gemm1_alpha",
                        "has_gemm1_beta",
                        "has_gemm1_clamp_limit",
                    )
                },
            )

        return self._factorized_tactic_space(inputs, resolve_pair)

    def _execute_body_source(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic = -1,
        do_preparation: bool = False,
        *,
        body_source: PrimsTsBodySource[PrimsTsFp8PerTensorBodyWorkspace],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Execute FP8 per-tensor from ordinary or prepared typed body inputs."""
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
        if use_routing_scales_on_input and moe_inputs.routing_logits is None:
            # Prepared FromLogits bodies consume canonical expert weights rather than logits, but
            # retain the public logits dtype in the runner's immutable cache-key extras.
            per_token_sf_dtype = int(
                dict(self._cache_key_static_extras).get("per_token_sf_dtype", 1)
            )
        else:
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

        def route_and_allocate() -> PrimsTsBodyExecution[
            PrimsTsFp8PerTensorBodyWorkspace
        ]:
            """Route ordinarily and package the FP8 per-tensor body ABI."""
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
            ) = _decode_routing_outputs(moe_inputs, routing_out)
            return PrimsTsBodyExecution(
                routing=PrimsTsBodyRouting(
                    expert_weights=expert_weights,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_token_idx=permuted_idx_to_token_idx,
                    tile_idx=tile_idx,
                    mn_limit=mn_limit,
                    num_non_exiting_ctas=num_non_exiting_ctas,
                    total_num_padded_tokens=total_num_padded_tokens,
                ),
                workspace=PrimsTsFp8PerTensorBodyWorkspace(gemm1_output, gemm2_output),
            )

        # Resolve ordinary routing or reuse the graph-stable record supplied by DA.
        execution = body_source.resolve(route_and_allocate)
        routing_metadata = execution.routing
        body_workspace = execution.workspace
        expert_weights = routing_metadata.expert_weights
        expanded_idx_to_permuted_idx = routing_metadata.expanded_idx_to_permuted_idx
        permuted_idx_to_token_idx = routing_metadata.permuted_idx_to_token_idx
        tile_idx = routing_metadata.tile_idx
        mn_limit = routing_metadata.mn_limit
        num_non_exiting_ctas = routing_metadata.num_non_exiting_ctas
        total_num_padded_tokens = routing_metadata.total_num_padded_tokens
        gemm1_output = body_workspace.gemm1_output
        gemm2_output = body_workspace.gemm2_output
        if do_preparation:
            return list(body_workspace.tensors())
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


class PrimsTsFp8BlockScaleMoERunner(
    _PrimsTsMoERunnerMixin[PrimsTsDeepSeekFp8BodyWorkspace | PrimsTsMxfp8BodyWorkspace],
    TunableRunner,
):
    """Autotuned Prims-TS FP8 block-scale MoE runner."""

    # Cached valid tactics partitioned by the complete static problem identity.
    valid_tactics_dict: dict = {}

    def __init__(
        self,
        moe_op: Any,
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
    ) -> None:
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

    def _body_workspace_from_sequence(
        self, tensors: Sequence[torch.Tensor]
    ) -> PrimsTsDeepSeekFp8BodyWorkspace | PrimsTsMxfp8BodyWorkspace:
        """Decode the distinct DeepSeek or MXFP8 workspace selected by this runner."""
        workspace_type = (
            PrimsTsDeepSeekFp8BodyWorkspace
            if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8
            else PrimsTsMxfp8BodyWorkspace
        )
        return workspace_type.from_sequence(tensors)

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
    ) -> List[MoeTactic]:
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

    def get_factorized_tactic_space(
        self, inputs: List[torch.Tensor]
    ) -> FactorizedTacticSpace:
        """Return legal FP8 block-scale FC1/FC2 factors and tile anchors."""
        num_tokens = MoeRunnerInputs.from_list(inputs).hidden_states.shape[0]
        flags = _gemm_config_flags_from_static_extras(self)
        extras = dict(self._cache_key_static_extras)

        def resolve_pair(raw_tactic: MoeTactic) -> PrimsTsGemmPair:
            """Resolve one complete public FP8 block-scale tactic row."""
            if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                return map_trtllm_deepseek_fp8_moe_tactic(
                    raw_tactic,
                    num_tokens=num_tokens,
                    top_k=self.top_k,
                    num_local_experts=self.num_local_experts,
                    weight_layout=int(self.weight_layout),
                    enable_pdl=bool(extras.get("enable_pdl", False)),
                )
            return map_trtllm_mxfp8_mxfp8_moe_tactic(
                raw_tactic,
                activation_type=int(self.activation_type),
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                weight_layout=int(self.weight_layout),
                fc1_has_bias=flags["fc1_has_bias"],
                fc2_has_bias=flags["fc2_has_bias"],
                enable_pdl=bool(extras.get("enable_pdl", False)),
                **{
                    name: flags[name]
                    for name in (
                        "has_gemm1_alpha",
                        "has_gemm1_beta",
                        "has_gemm1_clamp_limit",
                    )
                },
            )

        return self._factorized_tactic_space(inputs, resolve_pair)

    def _execute_body_source(
        self,
        inputs: List[torch.Tensor],
        tactic: MoeTactic = -1,
        do_preparation: bool = False,
        *,
        body_source: PrimsTsBodySource[
            PrimsTsDeepSeekFp8BodyWorkspace | PrimsTsMxfp8BodyWorkspace
        ],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Execute block FP8 from ordinary routing or a prepared typed body source."""
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

        def route_and_allocate() -> PrimsTsBodyExecution[Any]:
            """Route ordinarily and package the selected block-scale body ABI."""
            routing_out = self.moe_op.trtllm_moe_run_routing_fp8_block_scale(
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
            ) = _decode_routing_outputs(moe_inputs, routing_out)
            if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                workspace = PrimsTsDeepSeekFp8BodyWorkspace(
                    gemm1_output,
                    gemm1_output_scale,
                    activation_output,
                    activation_output_scale,
                    gemm2_output,
                )
            else:
                padded_scale = _pad_mxfp8_linear_scale_for_prims(
                    moe_inputs.hidden_states_scale,
                    num_tokens=num_tokens,
                    hidden_size=self.hidden_size,
                )
                workspace = PrimsTsMxfp8BodyWorkspace(
                    gemm1_output,
                    gemm1_output_scale,
                    activation_output,
                    activation_output_scale,
                    gemm2_output,
                    padded_scale,
                )
            return PrimsTsBodyExecution(
                routing=PrimsTsBodyRouting(
                    expert_weights=expert_weights,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    permuted_idx_to_token_idx=permuted_idx_to_token_idx,
                    tile_idx=tile_idx,
                    mn_limit=mn_limit,
                    num_non_exiting_ctas=num_non_exiting_ctas,
                    total_num_padded_tokens=total_num_padded_tokens,
                ),
                workspace=workspace,
            )

        # Resolve ordinary routing or reuse the graph-stable record supplied by DA.
        execution = body_source.resolve(route_and_allocate)
        routing_metadata = execution.routing
        body_workspace = execution.workspace
        expert_weights = routing_metadata.expert_weights
        expanded_idx_to_permuted_idx = routing_metadata.expanded_idx_to_permuted_idx
        permuted_idx_to_token_idx = routing_metadata.permuted_idx_to_token_idx
        tile_idx = routing_metadata.tile_idx
        mn_limit = routing_metadata.mn_limit
        num_non_exiting_ctas = routing_metadata.num_non_exiting_ctas
        total_num_padded_tokens = routing_metadata.total_num_padded_tokens
        gemm1_output = body_workspace.gemm1_output
        gemm1_output_scale = body_workspace.gemm1_output_scale
        activation_output = body_workspace.activation_output
        activation_output_scale = body_workspace.activation_output_scale
        gemm2_output = body_workspace.gemm2_output
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
                output=body_workspace.hidden_states_scale_padded,
            )
            activation_output = gemm1_output
            activation_output_scale = gemm1_output_scale
        if do_preparation:
            return list(body_workspace.tensors())

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
