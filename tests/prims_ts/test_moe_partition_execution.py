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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    prims_ts_bf16_moe,
    prims_ts_bf16_moe_partitioned,
    prims_ts_fp4_block_scale_moe,
    prims_ts_fp4_block_scale_moe_partitioned,
    prims_ts_fp8_block_scale_moe,
    prims_ts_fp8_block_scale_moe_partitioned,
    prims_ts_fp8_per_tensor_scale_moe,
    prims_ts_fp8_per_tensor_scale_moe_partitioned,
)
from flashinfer.green_ctx import split_device_green_ctx_by_sm_count
from flashinfer.prims_ts.moe.partition import (
    MoePartitionResources,
    MoeShardedWeights,
    PreparedWeightLayout,
    shard_prepared_moe_weights,
)
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.tllm_enums import (
    ActivationType,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
)
from flashinfer.utils import get_compute_capability
from tests.moe.trtllm_gen_fused_moe_utils import (
    BF16Moe,
    FP4Moe,
    FP8BlockScaleMoe,
    FP8PerTensorMoe,
    moe_args,
    quant_fp4,
    routing_reference_renormalize,
)
from tests.moe.utils import QuantMode


_DEVICE = torch.device("cuda:0")
_NUM_EXPERTS = 8
_TOP_K = 2
_HIDDEN_SIZE = 512
_INTERMEDIATE_SIZE = 256
_PADDING = 8


@dataclass
class _Workload:
    op: Callable[..., Any]
    partitioned_op: Callable[..., Any]
    kwargs: dict[str, Any]
    fc1_layout: PreparedWeightLayout
    fc2_layout: PreparedWeightLayout
    fc1_scale: torch.Tensor | None = None
    fc2_scale: torch.Tensor | None = None
    hidden_size: int = _HIDDEN_SIZE
    intermediate_size: int = _INTERMEDIATE_SIZE

    @property
    def routing_logits(self) -> torch.Tensor:
        return self.kwargs["routing_logits"]

    def run(
        self,
        *,
        partition: tuple[MoePartitionResources, str, MoeShardedWeights] | None,
        do_finalize: bool,
        output: torch.Tensor | None = None,
    ) -> Any:
        if partition is None:
            return self.op(
                **self.kwargs,
                do_finalize=do_finalize,
                output=output,
            )
        resources, strategy, shards = partition
        kwargs = {
            **self.kwargs,
            "gemm1_weights": shards.fc1,
            "gemm2_weights": shards.fc2,
            "partition_resources": resources,
            "partition_strategy": strategy,
        }
        if shards.fc1_scale is not None:
            kwargs["gemm1_weights_scale"] = shards.fc1_scale
        if shards.fc2_scale is not None:
            kwargs["gemm2_weights_scale"] = shards.fc2_scale
        for name in (
            "gemm1_bias",
            "gemm2_bias",
            "num_fused_shared_experts",
            "fc1_per_channel_weight_scale",
            "fc2_per_channel_weight_scale",
        ):
            kwargs.pop(name, None)
        if self.partitioned_op is prims_ts_fp8_per_tensor_scale_moe_partitioned:
            for name in (
                "gemm1_alpha",
                "gemm1_beta",
                "gemm1_clamp_limit",
            ):
                kwargs.pop(name, None)
        if self.partitioned_op is prims_ts_fp4_block_scale_moe_partitioned:
            kwargs.update(topk_ids=None, topk_weights=None)
        else:
            kwargs.update(topk_ids=None, expert_weights=None)
        return self.partitioned_op(
            **kwargs,
            do_finalize=do_finalize,
            output=output,
        )

    def partition(
        self,
        strategy: str,
        streams: tuple[torch.Stream, torch.Stream],
        alloc: Callable[[int, tuple[int, ...], torch.dtype], torch.Tensor]
        | None = None,
    ) -> tuple[MoePartitionResources, str, MoeShardedWeights]:
        shards = shard_prepared_moe_weights(
            self.kwargs["gemm1_weights"],
            self.kwargs["gemm2_weights"],
            strategy=strategy,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            fc1_layout=self.fc1_layout,
            fc2_layout=self.fc2_layout,
            fc1_scale=self.fc1_scale,
            fc2_scale=self.fc2_scale,
            alloc=alloc,
        )
        return MoePartitionResources(streams=streams), strategy, shards


def _require_prims_ts(kind: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = get_compute_capability(_DEVICE)
    if capability != (10, 7):
        pytest.skip("Prims-TS MoE partitioning requires SM107")
    if not is_prims_ts_available():
        pytest.skip("Prims-TS dependencies are unavailable")


def _green_partition_streams() -> tuple[
    tuple[torch.Stream, torch.Stream], tuple[Any, ...]
]:
    properties = torch.cuda.get_device_properties(_DEVICE)
    try:
        green_streams, resources = split_device_green_ctx_by_sm_count(
            _DEVICE, [properties.multi_processor_count // 2]
        )
    except (RuntimeError, ValueError) as exc:
        pytest.skip(f"green contexts are unavailable: {exc}")
    if len(green_streams) < 2:
        pytest.skip("green-context split did not return two streams")
    return (green_streams[0], green_streams[1]), tuple(resources)


def _localized_partition_setup() -> tuple[
    tuple[torch.Stream, torch.Stream],
    Callable[[int, tuple[int, ...], torch.dtype], torch.Tensor],
    tuple[Any, ...],
    Callable[[torch.Tensor], int],
]:
    """Create matched compute and memory resources for two locality domains."""

    from flashinfer.locality_domain import (
        get_localized_domains,
        locality_domain_unsupported_reason,
    )

    reason = locality_domain_unsupported_reason(_DEVICE)
    if reason is not None:
        pytest.skip(f"locality domains are unavailable: {reason}")
    domains = get_localized_domains(_DEVICE)
    return domains.streams, domains.empty, (domains,), domains.pointer_domain


def _quantized_inputs(
    kind: str, num_tokens: int, intermediate_size: int = _INTERMEDIATE_SIZE
) -> tuple[Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device=_DEVICE).manual_seed(2026 + num_tokens)
    hidden_states = torch.randn(
        (num_tokens, _HIDDEN_SIZE),
        generator=generator,
        device=_DEVICE,
        dtype=torch.bfloat16,
    )
    fc1 = (
        torch.randn(
            (_NUM_EXPERTS, 2 * intermediate_size, _HIDDEN_SIZE),
            generator=generator,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        / _HIDDEN_SIZE**0.5
    )
    fc2 = (
        torch.randn(
            (_NUM_EXPERTS, _HIDDEN_SIZE, intermediate_size),
            generator=generator,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        / intermediate_size**0.5
    )
    routing_logits = torch.randn(
        (num_tokens, _NUM_EXPERTS),
        generator=generator,
        device=_DEVICE,
        dtype=torch.bfloat16,
    )
    impl = {
        "bf16": BF16Moe(),
        "nvfp4": FP4Moe(QuantMode.FP4_NVFP4_NVFP4),
        "mxfp4_mxfp8": FP4Moe(QuantMode.FP4_MXFP4_MXFP8),
        "mxfp4_bf16": FP4Moe(QuantMode.FP4_MXFP4_Bf16),
        "fp8_per_tensor": FP8PerTensorMoe(),
        "fp8_block_scale": FP8BlockScaleMoe(QuantMode.FP8_BLOCK_SCALE_MXFP8),
    }[kind]
    impl._cache_permute_indices = {}
    weights = impl.quantize_weights(fc1, fc2, hidden_states)
    if kind in ("nvfp4", "mxfp4_mxfp8"):
        # Prims-TS routes token rows and therefore requires linear activation
        # scales; the 128x4-swizzled layout is only valid for other backends and
        # for the shared FP4 reference dequantizer.
        inputs = impl.quantize_inputs(
            hidden_states,
            weights["hidden_states_scale_global"],
            is_swizzling=False,
        )
    else:
        inputs = impl.quantize_inputs(
            hidden_states, weights["hidden_states_scale_global"]
        )
    return (
        impl,
        inputs,
        hidden_states,
        fc1,
        fc2,
        {
            **weights,
            "routing_logits": routing_logits,
        },
    )


def _prepare_workload(
    kind: str,
    num_tokens: int,
    activation_type: ActivationType = ActivationType.Swiglu,
    intermediate_size: int = _INTERMEDIATE_SIZE,
) -> _Workload:
    _require_prims_ts(kind)
    impl, inputs, hidden_states, fc1, fc2, quantized = _quantized_inputs(
        kind, num_tokens, intermediate_size
    )
    routing_logits = quantized["routing_logits"]
    permute_info, scores = routing_reference_renormalize(
        routing_logits, _TOP_K, _NUM_EXPERTS, _PADDING
    )
    reference_inputs = inputs
    if kind == "nvfp4":
        # The shared FP4 reference dequantizer currently hardcodes r128c4
        # scales, while Prims-TS requires linear scales for routed activations.
        reference_hidden_states, reference_scale, _ = quant_fp4(
            hidden_states,
            quantized["hidden_states_scale_global"],
            use_ue8m0=False,
            is_sf_swizzled_layout=True,
        )
        reference_inputs = {
            "hidden_states": reference_hidden_states,
            "hidden_states_scale": reference_scale.view(torch.float8_e4m3fn),
        }
    elif kind == "mxfp4_mxfp8":
        # Keep the reference's scale layout independent from the linear layout
        # consumed by Prims-TS; run_moe_reference_fp4 expects r128c4 here.
        reference_inputs = impl.quantize_inputs(
            hidden_states,
            quantized["hidden_states_scale_global"],
            is_swizzling=True,
        )
    args = moe_args(
        num_tokens,
        _NUM_EXPERTS,
        _HIDDEN_SIZE,
        intermediate_size,
        _TOP_K,
        _PADDING,
        reference_inputs["hidden_states"],
        reference_inputs["hidden_states_scale"],
        quantized["hidden_states_scale_global"],
        scores,
        quantized["gemm1_weights"],
        quantized["gemm1_scales"],
        quantized["gemm1_scales_global"],
        quantized["gemm2_weights"],
        quantized["gemm2_scales"],
        quantized["gemm2_scales_global"],
        permute_info,
        False,
        activation_type,
    )
    args_dequant = None
    if kind in (
        "nvfp4",
        "mxfp4_mxfp8",
        "mxfp4_bf16",
        "fp8_per_tensor",
    ):
        _, args_dequant = impl.compute_reference(args)
    static = impl.prepare_static_weights_for_kernel(
        args_dequant,
        args,
        fc1,
        fc2,
        _HIDDEN_SIZE,
        intermediate_size,
        _NUM_EXPERTS,
        {"use_shuffled_weight": True, "layout": WeightLayout.MajorK},
    )
    common = dict(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=inputs["hidden_states"],
        num_experts=_NUM_EXPERTS,
        top_k=_TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=_NUM_EXPERTS,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        enable_pdl=False,
        tune_max_num_tokens=max(1024, num_tokens),
        activation_type=activation_type.value,
    )

    if kind == "bf16":
        return _Workload(
            prims_ts_bf16_moe,
            prims_ts_bf16_moe_partitioned,
            {
                **common,
                "gemm1_weights": static["gemm1_weights"],
                "gemm2_weights": static["gemm2_weights"],
                "use_shuffled_weight": True,
                "weight_layout": WeightLayout.MajorK,
            },
            PreparedWeightLayout("major_k", storage_bits=16),
            PreparedWeightLayout("major_k", storage_bits=16),
        )
    if kind in ("nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"):
        fc1_scale = static["gemm1_scales_fp4_shuffled"]
        fc2_scale = static["gemm2_scales_fp4_shuffled"]
        scale_k_group_size = 16 if kind == "nvfp4" else 32
        layout = PreparedWeightLayout(
            "major_k",
            storage_bits=4,
            scale_layout="r128c4",
            scale_k_group_size=scale_k_group_size,
        )
        return _Workload(
            prims_ts_fp4_block_scale_moe,
            prims_ts_fp4_block_scale_moe_partitioned,
            {
                **common,
                "hidden_states_scale": inputs["hidden_states_scale"],
                "gemm1_weights": static["gemm1_weights_fp4_shuffled"],
                "gemm1_weights_scale": fc1_scale,
                "gemm1_bias": None,
                "gemm1_alpha": None,
                "gemm1_beta": None,
                "gemm1_clamp_limit": None,
                "gemm2_weights": static["gemm2_weights_fp4_shuffled"],
                "gemm2_weights_scale": fc2_scale,
                "gemm2_bias": None,
                "output1_scale_scalar": static["scale_c_fc1"],
                "output1_scale_gate_scalar": static["scale_gate_fc1"],
                "output2_scale_scalar": static["scale_c_fc2"],
                "weight_layout": WeightLayout.MajorK,
                "per_token_scale": None,
            },
            layout,
            layout,
            fc1_scale,
            fc2_scale,
            intermediate_size=intermediate_size,
        )
    if kind == "fp8_per_tensor":
        return _Workload(
            prims_ts_fp8_per_tensor_scale_moe,
            prims_ts_fp8_per_tensor_scale_moe_partitioned,
            {
                **common,
                "gemm1_weights": static["gemm1_weights"],
                "output1_scales_scalar": static["scale_c_fc1"],
                "output1_scales_gate_scalar": static["scale_gate_fc1"],
                "gemm2_weights": static["gemm2_weights"],
                "output2_scales_scalar": static["scale_c_fc2"],
                "use_routing_scales_on_input": False,
                "weight_layout": WeightLayout.MajorK,
            },
            PreparedWeightLayout("major_k", storage_bits=8),
            PreparedWeightLayout("major_k", storage_bits=8),
        )

    fc1_scale = static["gemm1_scales"]
    fc2_scale = static["gemm2_scales"]
    layout = PreparedWeightLayout(
        "major_k",
        storage_bits=8,
        scale_layout="r128c4",
        scale_k_group_size=32,
    )
    return _Workload(
        prims_ts_fp8_block_scale_moe,
        prims_ts_fp8_block_scale_moe_partitioned,
        {
            **common,
            "hidden_states_scale": inputs["hidden_states_scale"],
            "gemm1_weights": static["gemm1_weights"],
            "gemm1_weights_scale": fc1_scale,
            "gemm2_weights": static["gemm2_weights"],
            "gemm2_weights_scale": fc2_scale,
            "use_shuffled_weight": True,
            "weight_layout": WeightLayout.MajorK,
            "fp8_quantization_type": Fp8QuantizationType.MxFp8,
        },
        layout,
        layout,
        fc1_scale,
        fc2_scale,
    )


def _prepare_block_major_k_bf16_workload(num_tokens: int) -> _Workload:
    workload = _prepare_workload("bf16", num_tokens)
    block_bytes = 128
    for name in ("gemm1_weights", "gemm2_weights"):
        weight = workload.kwargs[name]
        storage_elements = block_bytes // weight.element_size()
        experts, rows, _ = weight.shape
        workload.kwargs[name] = (
            weight.view(experts, rows, -1, storage_elements)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
    workload.kwargs["weight_layout"] = WeightLayout.BlockMajorK
    layout = PreparedWeightLayout(
        "block_major_k",
        storage_bits=16,
        block_major_k_bytes=block_bytes,
    )
    workload.fc1_layout = layout
    workload.fc2_layout = layout
    return workload


def _resolved_mxfp4_block_major_k_bytes(num_tokens: int) -> tuple[int, int]:
    """Return the FC1/FC2 K-block widths the resolved MXFP4xMXFP8 tactic wants.

    ``block_major_k_bytes`` is derived from the tactic's ``tile_k`` and the
    weight operand's TMA/SMEM widths, not selected by the caller, and FC1 and
    FC2 may differ.  Weights must therefore be prepared to match the tactic,
    the same way ``tests.moe.trtllm_gen_fused_moe_utils`` does for the main MoE
    suite; a mismatch is reported as an unsupported config.
    """

    from flashinfer.prims_ts.moe.config_mapper import (
        map_trtllm_mxfp4_mxfp8_moe_tactic,
    )

    pair = map_trtllm_mxfp4_mxfp8_moe_tactic(
        [-1, -1],
        num_tokens=num_tokens,
        top_k=_TOP_K,
        num_local_experts=_NUM_EXPERTS,
        activation_type=int(ActivationType.Swiglu),
        weight_layout=int(WeightLayout.BlockMajorK),
    )
    return (
        int(pair.fc1.cfg.build().block_major_k_bytes),
        int(pair.fc2.cfg.build().block_major_k_bytes),
    )


def _prepare_block_major_k_mxfp4_workload(num_tokens: int) -> _Workload:
    workload = _prepare_workload("mxfp4_mxfp8", num_tokens)
    layouts = []
    for name, block_bytes, scale in zip(
        ("gemm1_weights", "gemm2_weights"),
        _resolved_mxfp4_block_major_k_bytes(num_tokens),
        (workload.fc1_scale, workload.fc2_scale),
        strict=True,
    ):
        weight = workload.kwargs[name]
        storage_elements = block_bytes // weight.element_size()
        experts, rows, _ = weight.shape
        workload.kwargs[name] = (
            weight.view(experts, rows, -1, storage_elements)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        layouts.append(
            PreparedWeightLayout(
                "block_major_k",
                storage_bits=4,
                weight_dtype=weight.dtype,
                block_major_k_bytes=block_bytes,
                scale_layout="r128c4",
                scale_k_group_size=32,
                scale_dtype=scale.dtype,
            )
        )
    workload.kwargs["weight_layout"] = WeightLayout.BlockMajorK
    workload.fc1_layout, workload.fc2_layout = layouts
    return workload


def _set_activation_params(workload: _Workload, case: str) -> None:
    """Attach expert-varying OA parameters to an execution workload."""

    expert_offset = torch.linspace(
        -0.5,
        0.5,
        _NUM_EXPERTS,
        dtype=torch.float32,
        device=_DEVICE,
    )
    workload.kwargs.update(
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
    )
    if case == "situ":
        workload.kwargs.update(
            activation_type=ActivationType.Situ.value,
            gemm1_alpha=(4.0 + 0.25 * expert_offset).contiguous(),
            gemm1_beta=(25.0 + 2.0 * expert_offset).contiguous(),
        )
        return

    workload.kwargs["activation_type"] = ActivationType.Swiglu.value
    clamp_limit = (0.5 + 0.5 * (expert_offset + 0.5)).contiguous()
    if case == "nvfp4-swiglu-clamp":
        scale_gate = workload.kwargs["output1_scale_gate_scalar"].float()
        assert scale_gate.shape == (_NUM_EXPERTS,)
        assert torch.all(scale_gate > 0)
        clamp_limit = (clamp_limit / scale_gate).contiguous()
    elif case not in ("mxfp8-swiglu-clamp", "bf16-swiglu-clamp"):
        raise ValueError(f"unknown activation-parameter test case {case!r}")
    workload.kwargs["gemm1_clamp_limit"] = clamp_limit


@dataclass(frozen=True)
class _ExecutionCase:
    kind: str
    strategy: str
    do_finalize: bool
    num_tokens: int
    enable_pdl: bool = True


def _execution_case(
    *,
    id: str,
    kind: str,
    strategy: str,
    do_finalize: bool,
    num_tokens: int,
    enable_pdl: bool = True,
):
    return pytest.param(
        _ExecutionCase(
            kind=kind,
            strategy=strategy,
            do_finalize=do_finalize,
            num_tokens=num_tokens,
            enable_pdl=enable_pdl,
        ),
        id=id,
    )


_EXECUTION_CASES = (
    _execution_case(
        id="bf16-nn-finalize-t7-pdl",
        kind="bf16",
        strategy="tp_n_tp_n",
        do_finalize=True,
        num_tokens=7,
    ),
    _execution_case(
        id="bf16-nk-raw-t7-pdl",
        kind="bf16",
        strategy="tp_n_tp_k",
        do_finalize=False,
        num_tokens=7,
    ),
    _execution_case(
        id="bf16-nn-raw-t1024-pdl",
        kind="bf16",
        strategy="tp_n_tp_n",
        do_finalize=False,
        num_tokens=1024,
    ),
    _execution_case(
        id="bf16-nk-finalize-t1024-pdl",
        kind="bf16",
        strategy="tp_n_tp_k",
        do_finalize=True,
        num_tokens=1024,
    ),
    _execution_case(
        id="nvfp4-nn-raw-pdl",
        kind="nvfp4",
        strategy="tp_n_tp_n",
        do_finalize=False,
        num_tokens=7,
    ),
    _execution_case(
        id="nvfp4-nk-finalize-pdl",
        kind="nvfp4",
        strategy="tp_n_tp_k",
        do_finalize=True,
        num_tokens=7,
    ),
    _execution_case(
        id="mxfp4-mxfp8-nn-finalize-pdl",
        kind="mxfp4_mxfp8",
        strategy="tp_n_tp_n",
        do_finalize=True,
        num_tokens=7,
    ),
    _execution_case(
        id="mxfp4-bf16-nk-finalize-pdl",
        kind="mxfp4_bf16",
        strategy="tp_n_tp_k",
        do_finalize=True,
        num_tokens=7,
    ),
    _execution_case(
        id="fp8-tensor-nn-raw-pdl",
        kind="fp8_per_tensor",
        strategy="tp_n_tp_n",
        do_finalize=False,
        num_tokens=7,
    ),
    _execution_case(
        id="fp8-tensor-nk-finalize-pdl",
        kind="fp8_per_tensor",
        strategy="tp_n_tp_k",
        do_finalize=True,
        num_tokens=7,
    ),
    _execution_case(
        id="fp8-block-nn-finalize-pdl",
        kind="fp8_block_scale",
        strategy="tp_n_tp_n",
        do_finalize=True,
        num_tokens=7,
    ),
    _execution_case(
        id="fp8-block-nk-raw-pdl",
        kind="fp8_block_scale",
        strategy="tp_n_tp_k",
        do_finalize=False,
        num_tokens=7,
    ),
    # PDL-off only needs a smoke case per partition strategy; the matrix above
    # exercises the newly enabled path across every dtype and output mode.
    _execution_case(
        id="bf16-nn-finalize-t7-no-pdl",
        kind="bf16",
        strategy="tp_n_tp_n",
        do_finalize=True,
        num_tokens=7,
        enable_pdl=False,
    ),
    _execution_case(
        id="bf16-nk-raw-t7-no-pdl",
        kind="bf16",
        strategy="tp_n_tp_k",
        do_finalize=False,
        num_tokens=7,
        enable_pdl=False,
    ),
)

_ACTIVATION_PARAM_CASES = (
    pytest.param("nvfp4", "situ", "tp_n_tp_n", id="nvfp4-situ-nn"),
    pytest.param(
        "nvfp4", "nvfp4-swiglu-clamp", "tp_n_tp_k", id="nvfp4-swiglu-clamp-nk"
    ),
    pytest.param(
        "fp8_block_scale",
        "mxfp8-swiglu-clamp",
        "tp_n_tp_n",
        id="mxfp8-swiglu-clamp-nn",
    ),
    pytest.param("bf16", "bf16-swiglu-clamp", "tp_n_tp_k", id="bf16-swiglu-clamp-nk"),
)


def _result_tensors(result: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(result, torch.Tensor):
        return (result,)
    assert isinstance(result, (list, tuple)), (
        f"unexpected MoE result type: {type(result).__name__}"
    )
    assert result, "MoE returned an empty result"
    assert all(isinstance(tensor, torch.Tensor) for tensor in result)
    return tuple(result)


def _comparison_output(
    result_tensors: tuple[torch.Tensor, ...], num_tokens: int
) -> torch.Tensor:
    if len(result_tensors) == 1:
        return result_tensors[0]
    assert len(result_tensors) >= 3, (
        f"raw MoE result must contain output and routing metadata, got "
        f"{len(result_tensors)} tensors"
    )
    gemm2_output, expert_weights, expanded_idx_to_permuted_idx = result_tensors[:3]
    permutation = expanded_idx_to_permuted_idx.reshape(num_tokens, _TOP_K).long()
    assert (permutation >= 0).all()
    return (
        gemm2_output[permutation]
        .float()
        .mul(expert_weights.float().reshape(num_tokens, _TOP_K, 1))
        .sum(dim=1)
    )


def test_partitioned_call_rejects_non_sm107() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if get_compute_capability(_DEVICE) == (10, 7):
        pytest.skip("requires a non-SM107 device")
    streams, green_resources = _green_partition_streams()
    assert green_resources  # Keep resources alive through the call.
    resources = MoePartitionResources(streams=streams)
    num_tokens = 4
    hidden_states = torch.zeros(
        (num_tokens, _HIDDEN_SIZE), dtype=torch.bfloat16, device=_DEVICE
    )
    routing_logits = torch.zeros(
        (num_tokens, _NUM_EXPERTS), dtype=torch.bfloat16, device=_DEVICE
    )
    fc1 = [
        torch.empty(
            (_NUM_EXPERTS, _INTERMEDIATE_SIZE, _HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=_DEVICE,
        )
        for _ in range(2)
    ]
    fc2 = [
        torch.empty(
            (_NUM_EXPERTS, _HIDDEN_SIZE // 2, _INTERMEDIATE_SIZE),
            dtype=torch.bfloat16,
            device=_DEVICE,
        )
        for _ in range(2)
    ]

    with pytest.raises(NotImplementedError, match="only supported on SM107"):
        prims_ts_bf16_moe_partitioned(
            routing_logits=routing_logits,
            routing_bias=None,
            topk_ids=None,
            expert_weights=None,
            hidden_states=hidden_states,
            gemm1_weights=fc1,
            gemm2_weights=fc2,
            num_experts=_NUM_EXPERTS,
            top_k=_TOP_K,
            n_group=None,
            topk_group=None,
            intermediate_size=_INTERMEDIATE_SIZE,
            local_expert_offset=0,
            local_num_experts=_NUM_EXPERTS,
            partition_resources=resources,
            partition_strategy="tp_n_tp_n",
            routing_method_type=RoutingMethodType.Renormalize.value,
            enable_pdl=False,
        )


@pytest.mark.parametrize("case", _EXECUTION_CASES)
def test_partitioned_execution_matches_unpartitioned(case: _ExecutionCase) -> None:
    workload = _prepare_workload(case.kind, case.num_tokens)
    workload.kwargs["enable_pdl"] = case.enable_pdl
    streams, green_resources = _green_partition_streams()
    assert green_resources  # Keep resources alive through execution.
    partition = workload.partition(case.strategy, streams)

    with autotune(False):
        expected_result = workload.run(partition=None, do_finalize=case.do_finalize)
        expected_tensors = _result_tensors(expected_result)
        expected = _comparison_output(expected_tensors, case.num_tokens).clone()
        expected_expert_weights = (
            expected_tensors[1].clone() if len(expected_tensors) >= 3 else None
        )
        actual_result = workload.run(partition=partition, do_finalize=case.do_finalize)
        actual_tensors = _result_tensors(actual_result)
        actual = _comparison_output(actual_tensors, case.num_tokens).clone()
    torch.cuda.synchronize()

    if expected_expert_weights is not None:
        # Each routing launch may choose a different valid permutation within an
        # expert, so compare values gathered through each result's own map above.
        assert len(actual_tensors) >= 3
        assert torch.equal(actual_tensors[1], expected_expert_weights)
    if case.strategy == "tp_n_tp_n":
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("kind", "case", "strategy"),
    _ACTIVATION_PARAM_CASES,
)
def test_partitioned_activation_params_match_unpartitioned(
    kind: str,
    case: str,
    strategy: str,
) -> None:
    activation_type = ActivationType.Situ if case == "situ" else ActivationType.Swiglu
    workload = _prepare_workload(
        kind,
        num_tokens=7,
        activation_type=activation_type,
    )
    _set_activation_params(workload, case)
    streams, green_resources = _green_partition_streams()
    assert green_resources  # Keep resources alive through execution.
    partition = workload.partition(strategy, streams)

    with autotune(False):
        expected = _comparison_output(
            _result_tensors(workload.run(partition=None, do_finalize=True)),
            num_tokens=7,
        ).clone()
        actual = _comparison_output(
            _result_tensors(workload.run(partition=partition, do_finalize=True)),
            num_tokens=7,
        ).clone()

        activation_params = {
            name: workload.kwargs[name]
            for name in (
                "gemm1_alpha",
                "gemm1_beta",
                "gemm1_clamp_limit",
            )
        }
        workload.kwargs.update(
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
        )
        without_params = _comparison_output(
            _result_tensors(workload.run(partition=None, do_finalize=True)),
            num_tokens=7,
        ).clone()
        workload.kwargs.update(activation_params)
    torch.cuda.synchronize()

    if strategy == "tp_n_tp_n":
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(
        expected.float(),
        without_params.float(),
        rtol=1e-3,
        atol=1e-3,
    ), f"{case} parameters did not change the unpartitioned output"


@pytest.mark.parametrize("kind", ("nvfp4", "mxfp4_mxfp8"))
def test_partial_sf_group_partition_matches_unpartitioned(kind: str) -> None:
    num_tokens = 7
    workload = _prepare_workload(
        kind,
        num_tokens=num_tokens,
        activation_type=ActivationType.Situ,
        intermediate_size=384,
    )
    _set_activation_params(workload, "situ")
    streams, green_resources = _green_partition_streams()
    assert green_resources  # Keep resources alive through execution.
    partition = workload.partition("tp_n_tp_n", streams)

    with autotune(False):
        expected = _comparison_output(
            _result_tensors(workload.run(partition=None, do_finalize=True)),
            num_tokens,
        ).clone()
        actual = _comparison_output(
            _result_tensors(workload.run(partition=partition, do_finalize=True)),
            num_tokens,
        ).clone()
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_block_major_k_bf16_partitioned_execution_matches_unpartitioned() -> None:
    num_tokens = 7
    workload = _prepare_block_major_k_bf16_workload(num_tokens)
    streams, green_resources = _green_partition_streams()
    assert green_resources  # Keep resources alive through execution.
    partition = workload.partition("tp_n_tp_n", streams)

    with autotune(False):
        expected = workload.run(partition=None, do_finalize=True).clone()
        actual = workload.run(partition=partition, do_finalize=True).clone()
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_block_major_k_mxfp4_partitioned_execution_matches_unpartitioned() -> None:
    num_tokens = 7
    workload = _prepare_block_major_k_mxfp4_workload(num_tokens)
    streams, green_resources = _green_partition_streams()
    assert green_resources  # Keep resources alive through execution.
    partition = workload.partition("tp_n_tp_n", streams)

    with autotune(False):
        expected = _comparison_output(
            _result_tensors(workload.run(partition=None, do_finalize=True)),
            num_tokens,
        ).clone()
        actual = _comparison_output(
            _result_tensors(workload.run(partition=partition, do_finalize=True)),
            num_tokens,
        ).clone()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("kind", "strategy"),
    (
        pytest.param("nvfp4", "tp_n_tp_n", id="nvfp4-nn"),
        pytest.param("nvfp4", "tp_n_tp_k", id="nvfp4-nk"),
        pytest.param("mxfp4_mxfp8", "tp_n_tp_n", id="mxfp4-mxfp8-nn"),
        pytest.param("mxfp4_mxfp8", "tp_n_tp_k", id="mxfp4-mxfp8-nk"),
        pytest.param("bf16", "tp_n_tp_n", id="bf16-nn"),
        pytest.param("fp8_block_scale", "tp_n_tp_n", id="mxfp8-nn"),
    ),
)
def test_partitioned_execution_with_localized_weights(kind: str, strategy: str) -> None:
    """Exercise weight shards and compute placed in matching locality domains."""

    num_tokens = 7
    workload = _prepare_workload(kind, num_tokens)
    streams, alloc, keepalive, pointer_domain = _localized_partition_setup()
    partition = workload.partition(strategy, streams, alloc=alloc)
    shards = partition[2]

    assert keepalive
    for partition_id in range(2):
        assert pointer_domain(shards.fc1[partition_id]) == partition_id
        assert pointer_domain(shards.fc2[partition_id]) == partition_id
        if shards.fc1_scale is not None:
            assert pointer_domain(shards.fc1_scale[partition_id]) == partition_id
        if shards.fc2_scale is not None:
            assert pointer_domain(shards.fc2_scale[partition_id]) == partition_id

    with autotune(False):
        expected = _comparison_output(
            _result_tensors(workload.run(partition=None, do_finalize=True)), num_tokens
        ).clone()
        actual = _comparison_output(
            _result_tensors(workload.run(partition=partition, do_finalize=True)),
            num_tokens,
        ).clone()
    torch.cuda.synchronize()

    if strategy == "tp_n_tp_n":
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("strategy", "localized", "enable_pdl"),
    (
        pytest.param("tp_n_tp_n", False, False, id="green-nn-no-pdl"),
        pytest.param("tp_n_tp_k", False, False, id="green-nk-no-pdl"),
        pytest.param("tp_n_tp_k", False, True, id="green-nk-pdl"),
        pytest.param("tp_n_tp_n", True, True, id="localized-nn-pdl"),
    ),
)
def test_partitioned_cuda_graph_replay_observes_changed_routing(
    strategy: str, localized: bool, enable_pdl: bool
) -> None:
    workload = _prepare_workload("bf16", 7)
    workload.kwargs["enable_pdl"] = enable_pdl
    if localized:
        streams, alloc, keepalive, _ = _localized_partition_setup()
    else:
        streams, green_resources = _green_partition_streams()
        alloc = None
        keepalive = green_resources
    assert keepalive  # Keep contexts and pools alive through graph replay.
    partition = workload.partition(strategy, streams, alloc=alloc)
    partition_resources = partition[0]
    fork_join = partition_resources.fork_join
    graph_output = torch.empty((7, _HIDDEN_SIZE), dtype=torch.bfloat16, device=_DEVICE)
    expected_output = torch.empty_like(graph_output)

    with autotune(False):
        workload.run(partition=partition, do_finalize=True, output=graph_output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = workload.run(
                partition=partition, do_finalize=True, output=graph_output
            )

        workload.routing_logits.copy_(torch.flip(workload.routing_logits, dims=(1,)))
        expected = workload.run(
            partition=None, do_finalize=True, output=expected_output
        ).clone()
        if strategy == "tp_n_tp_k":
            different_shape = _prepare_workload("bf16", 11)
            different_partition = different_shape.partition(
                strategy, streams, alloc=alloc
            )
            different_partition = (
                partition_resources,
                *different_partition[1:],
            )
            different_shape.run(
                partition=different_partition,
                do_finalize=True,
            )
        graph.replay()
    torch.cuda.synchronize()
    assert partition_resources.fork_join is fork_join

    if strategy == "tp_n_tp_n":
        assert torch.equal(captured, expected)
    else:
        torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "strategy",
    (
        pytest.param("tp_n_tp_n", id="nn"),
        pytest.param("tp_n_tp_k", id="nk"),
    ),
)
def test_partitioned_pdl_graph_replay_stress_changed_and_empty_local_routing(
    strategy: str,
) -> None:
    """Stress PDL graph replay while alternating local and remote-only routes."""

    num_tokens = 1
    workload = _prepare_workload("bf16", num_tokens)
    workload.kwargs["enable_pdl"] = True
    workload.kwargs["local_num_experts"] = _NUM_EXPERTS // 2
    for name in ("gemm1_weights", "gemm2_weights"):
        workload.kwargs[name] = workload.kwargs[name][
            : workload.kwargs["local_num_experts"]
        ].contiguous()

    streams, green_resources = _green_partition_streams()
    assert green_resources  # Keep contexts alive through every replay.
    partition = workload.partition(strategy, streams)
    graph_output = torch.empty(
        (num_tokens, _HIDDEN_SIZE), dtype=torch.bfloat16, device=_DEVICE
    )
    expected_output = torch.empty_like(graph_output)

    def set_routes(*expert_ids: int) -> None:
        assert len(expert_ids) == _TOP_K
        workload.routing_logits.fill_(-100.0)
        for score, expert_id in enumerate(expert_ids, start=1):
            workload.routing_logits[:, expert_id] = float(score)

    with autotune(False):
        set_routes(0, 1)
        workload.run(partition=partition, do_finalize=True, output=graph_output)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = workload.run(
                partition=partition,
                do_finalize=True,
                output=graph_output,
            )

        for replay_id in range(200):
            # Experts 4 and 5 are outside this rank's [0, 4) local range.
            set_routes(4, 5) if replay_id % 2 else set_routes(0, 1)
            expected = workload.run(
                partition=None,
                do_finalize=True,
                output=expected_output,
            ).clone()
            graph.replay()
            torch.cuda.synchronize()
            if strategy == "tp_n_tp_n":
                assert torch.equal(captured, expected), f"replay {replay_id}"
            else:
                torch.testing.assert_close(
                    captured,
                    expected,
                    rtol=2e-2,
                    atol=2e-2,
                    msg=lambda message: f"replay {replay_id}: {message}",
                )
