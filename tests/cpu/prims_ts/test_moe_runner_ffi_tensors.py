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

import ast
import contextlib
import inspect
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tvm_ffi

from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.backends.prims_ts import bf16_op, fp4_op, fp8_op
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode
from flashinfer.fused_moe.shared.tuning import moe_topk_ids_init
from flashinfer.jit import env as jit_env
from flashinfer.prims_ts.moe import runner as runner_module
from flashinfer.prims_ts.moe.config_mapper import (
    map_trtllm_bf16_moe_tactic,
    map_trtllm_deepseek_fp8_moe_tactic,
)
from flashinfer.prims_ts.moe.partition import (
    MoePartitionResources,
    PreparedWeightLayout,
    validate_partition_arch,
)
from flashinfer.prims_ts.moe.tensor_adapter import (
    _get_expert_scale_ones,
    _validate_block_scaled_sf_output,
    _validate_c_output_layout,
    _validate_sf_c_m_offset,
)
from flashinfer.prims_ts.moe.runner import (
    PrimsTsBf16MoERunner,
    PrimsTsMxfp4Mxfp8MoERunner,
    PrimsTsPartitionedBf16MoERunner,
    PrimsTsPartitionedFp8BlockScaleMoERunner,
    _moe_topk_ids_init_for_routing,
    _routed_token_capacity,
    _select_expert_weights,
    _staged_routing_io,
    _torch_views_of_ffi_tensors,
)
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.tllm_enums import (
    ActivationType,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
)
from flashinfer.utils import get_compute_capability


_STAGED_WEIGHT_PARAMS = (
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("name", "expected_params"),
    (
        (
            "trtllm_moe_run_routing",
            ("hidden_states", "gemm1_weights", "gemm2_weights", "num_experts"),
        ),
        (
            "trtllm_moe_run_routing_fp4_nvfp4",
            ("hidden_states", "hidden_states_scale", *_STAGED_WEIGHT_PARAMS),
        ),
        (
            "trtllm_moe_run_routing_fp4_mxfp4_mxfp8",
            ("hidden_states", "hidden_states_scale", *_STAGED_WEIGHT_PARAMS),
        ),
        (
            "trtllm_moe_run_routing_fp4_mxfp4_bf16",
            ("hidden_states", *_STAGED_WEIGHT_PARAMS, "output1_scale_scalar"),
        ),
        (
            "trtllm_moe_run_routing_fp8_per_tensor",
            (
                "hidden_states",
                "gemm1_weights",
                "output1_scale_scalar",
                "output1_scale_gate_scalar",
                "gemm2_weights",
                "output2_scale_scalar",
            ),
        ),
        (
            "trtllm_moe_run_routing_fp8_block_scale",
            ("hidden_states", "hidden_states_scale", *_STAGED_WEIGHT_PARAMS),
        ),
    ),
)
def test_staged_routing_ffi_signatures_take_optional_weights(name, expected_params):
    source_path = (
        Path(jit_env.FLASHINFER_CSRC_DIR) / "trtllm_fused_moe_kernel_launcher.cu"
    )
    assert source_path.exists(), f"JIT MoE source does not exist: {source_path}"
    source = source_path.read_text()
    begin = source.index(f"Array<Tensor> {name}(")
    signature = source[source.index("(", begin) + 1 : source.index(")", begin)]
    params = [param.strip() for param in signature.split(",")]
    names = [param.split()[-1] for param in params]

    start = names.index(expected_params[0])
    assert tuple(names[start : start + len(expected_params)]) == expected_params
    assert names[-1] == "routing_replay_out"
    for param, param_name in zip(params, names, strict=True):
        if param_name in _STAGED_WEIGHT_PARAMS:
            assert param.startswith("Optional<TensorView> const&"), param


def _partition_runner_kwargs(strategy: str = "tp_n_tp_k"):
    layout = PreparedWeightLayout("major_k", storage_bits=16)
    return dict(
        partition_strategy=strategy,
        partition_fc1_layout=layout,
        partition_fc2_layout=layout,
    )


def _partition_call_kwargs():
    resources = MoePartitionResources(
        streams=tuple(SimpleNamespace(device=torch.device("cpu")) for _ in range(2)),
    )
    weights = (torch.empty(0), torch.empty(0))
    return dict(
        partition_resources=resources,
        partition_fc1_weights=weights,
        partition_fc2_weights=weights,
        partition_fc1_scales=None,
        partition_fc2_scales=None,
    )


def _partition_pair_kwargs():
    kwargs = _partition_call_kwargs()
    return dict(
        partition=kwargs["partition_resources"],
        partition_fc1_weights=kwargs["partition_fc1_weights"],
        partition_fc2_weights=kwargs["partition_fc2_weights"],
        partition_fc1_scales=kwargs["partition_fc1_scales"],
        partition_fc2_scales=kwargs["partition_fc2_scales"],
    )


@pytest.mark.cpu_only
def test_cache_key_extras_are_invariant_to_synthesized_placeholders():
    runner = PrimsTsBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=1024,
        intermediate_size=512,
        num_experts=8,
    )
    runtime_inputs = MoeRunnerInputs(
        output=torch.empty(32, 1024),
        routing_logits=torch.empty(0),
        topk_ids=torch.empty(32, dtype=torch.int32),
        expert_weights=torch.empty(0, dtype=torch.bfloat16),
        hidden_states=torch.empty(32, 1024),
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    synthesized_inputs = MoeRunnerInputs(
        output=runtime_inputs.output,
        routing_logits=torch.empty(32),
        topk_ids=runtime_inputs.topk_ids,
        expert_weights=torch.empty(32, dtype=torch.bfloat16),
        hidden_states=runtime_inputs.hidden_states,
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )

    assert runner.get_cache_key_extras(
        runtime_inputs.to_list()
    ) == runner.get_cache_key_extras(synthesized_inputs.to_list())


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("runner_cls", "runner_kwargs"),
    (
        pytest.param(PrimsTsBf16MoERunner, {}, id="unpartitioned"),
        pytest.param(
            PrimsTsPartitionedBf16MoERunner,
            _partition_runner_kwargs(),
            id="partitioned",
        ),
    ),
)
def test_prims_ts_packed_tuning_keeps_expert_weights_sentinel_empty(
    runner_cls, runner_kwargs
):
    runner = runner_cls(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        **runner_kwargs,
    )
    moe_inputs = MoeRunnerInputs(
        output=torch.empty(4, 128),
        routing_logits=torch.empty(0),
        topk_ids=torch.zeros(4, 2, dtype=torch.int32),
        expert_weights=torch.empty(0, dtype=torch.bfloat16),
        hidden_states=torch.empty(4, 128),
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )

    config = runner._make_tuning_config(
        moe_inputs,
        tune_max_num_tokens=8,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
    )
    expert_weights_idx = MoeRunnerInputs.idx("expert_weights")
    assert all(
        expert_weights_idx not in spec.input_idx for spec in config.dynamic_tensor_specs
    )

    inputs = moe_inputs.to_list()
    profiles = AutoTuner.get()._generate_optimization_profiles(config, inputs)
    prepared = AutoTuner.get()._prepare_input_tensors(profiles[-1], inputs)
    assert prepared[expert_weights_idx] is moe_inputs.expert_weights
    assert tuple(prepared[expert_weights_idx].shape) == (0,)
    assert tuple(prepared[MoeRunnerInputs.idx("topk_ids")].shape) == (8, runner.top_k)


@pytest.mark.cpu_only
def test_partitioned_runner_hash_uses_strategy_and_stream_topology(monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "_partition_stream_topology",
        lambda resources: resources.topology if resources is not None else (),
    )

    def make_runner(strategy, topology=(60, 60)):
        runner = PrimsTsPartitionedBf16MoERunner(
            None,
            top_k=2,
            num_local_experts=8,
            hidden_size=1024,
            intermediate_size=512,
            num_experts=8,
            partition_resources=SimpleNamespace(topology=topology),
            **_partition_runner_kwargs(strategy),
        )
        runner.set_cache_key_static_extras(
            enable_pdl=False,
            weight_layout=WeightLayout.MajorK,
        )
        return runner

    inputs = MoeRunnerInputs(
        output=torch.empty(32, 1024),
        routing_logits=torch.empty(0),
        topk_ids=torch.empty(32, dtype=torch.int32),
        expert_weights=torch.empty(0, dtype=torch.bfloat16),
        hidden_states=torch.empty(32, 1024),
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    ).to_list()
    first = make_runner("tp_n_tp_n")
    second = make_runner("tp_n_tp_n")
    other_strategy = make_runner("tp_n_tp_k")
    other_topology = make_runner("tp_n_tp_n", topology=(120, 120))
    first_payload = _partition_call_kwargs()
    second_payload = _partition_call_kwargs()

    assert (
        first_payload["partition_fc1_weights"]
        is not second_payload["partition_fc1_weights"]
    )
    assert "partition_fc1_weights" not in first.__dict__
    assert "partition_resources" not in first.__dict__
    assert (
        first._partition_shards(first_payload)[0]
        is first_payload["partition_resources"]
    )
    assert hash(first) == hash(second)
    assert first.get_cache_key_extras(inputs) == second.get_cache_key_extras(inputs)
    assert hash(first) != hash(other_strategy)
    assert first.get_cache_key_extras(inputs) != other_strategy.get_cache_key_extras(
        inputs
    )
    assert hash(first) != hash(other_topology)
    assert first.get_cache_key_extras(inputs) != other_topology.get_cache_key_extras(
        inputs
    )


@pytest.mark.cpu_only
def test_partition_stream_topology_tracks_context_and_stream_aliasing(monkeypatch):
    streams = (
        SimpleNamespace(cuda_stream=11),
        SimpleNamespace(cuda_stream=12),
    )

    def topology(resource_streams=streams):
        return runner_module._partition_stream_topology(
            SimpleNamespace(streams=resource_streams, _stream_topology=None)
        )

    contexts = {11: 101, 12: 102}
    monkeypatch.setattr(runner_module.cuda_drv, "CUstream", lambda stream: stream)
    monkeypatch.setattr(
        runner_module.cuda_drv,
        "cuStreamGetGreenCtx",
        lambda stream: contexts[stream],
    )
    monkeypatch.setattr(
        runner_module.cuda_drv,
        "cuGreenCtxGetDevResource",
        lambda context, resource_type: SimpleNamespace(sm=SimpleNamespace(smCount=60)),
    )
    monkeypatch.setattr(runner_module, "checkCudaErrors", lambda value: value)

    assert topology() == (
        ("green", 60, 0, 0, 0),
        ("green", 60, 1, 1, 0),
    )
    contexts[12] = 101
    assert topology() == (
        ("green", 60, 0, 0, 0),
        ("green", 60, 0, 1, 0),
    )
    streams[1].priority = -1
    assert topology()[1][-1] == -1
    streams[1].priority = 0
    assert topology((streams[0], streams[0])) == (
        ("green", 60, 0, 0, 0),
        ("green", 60, 0, 0, 0),
    )


@pytest.mark.cpu_only
def test_partition_stream_topology_is_cached_on_resources(monkeypatch):
    resources = SimpleNamespace(
        streams=(SimpleNamespace(cuda_stream=11), SimpleNamespace(cuda_stream=12)),
        _stream_topology=None,
    )
    queries = []
    monkeypatch.setattr(runner_module.cuda_drv, "CUstream", lambda stream: stream)
    monkeypatch.setattr(
        runner_module.cuda_drv,
        "cuStreamGetGreenCtx",
        lambda stream: queries.append(stream) or stream,
    )
    monkeypatch.setattr(
        runner_module.cuda_drv,
        "cuGreenCtxGetDevResource",
        lambda context, resource_type: SimpleNamespace(sm=SimpleNamespace(smCount=60)),
    )
    monkeypatch.setattr(runner_module, "checkCudaErrors", lambda value: value)

    first = runner_module._partition_stream_topology(resources)
    assert runner_module._partition_stream_topology(resources) is first
    assert queries == [11, 12]


@pytest.mark.cpu_only
def test_partitioned_runner_rejects_deepseek_fp8_mode():
    runner = PrimsTsPartitionedFp8BlockScaleMoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=1024,
        intermediate_size=512,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        num_experts=8,
        **_partition_runner_kwargs(),
    )
    inputs = MoeRunnerInputs(
        output=torch.empty(1, 1024),
        routing_logits=torch.empty(1, 8),
        topk_ids=torch.empty(1, dtype=torch.int32),
        expert_weights=torch.empty(1, dtype=torch.bfloat16),
        hidden_states=torch.empty(1, 1024),
        hidden_states_scale=torch.empty(1),
        gemm1_lora_delta=None,
        per_token_scale=None,
    ).to_list()

    with pytest.raises(NotImplementedError, match="DeepSeek"):
        runner._filter_partition_geometry_tactics([1], inputs, cache_key=object())


@pytest.mark.cpu_only
def test_partitioned_fp8_block_scale_defaults_to_mxfp8():
    default = (
        inspect.signature(fp8_op.prims_ts_fp8_block_scale_moe_partitioned)
        .parameters["fp8_quantization_type"]
        .default
    )

    assert default == Fp8QuantizationType.MxFp8


@pytest.mark.cpu_only
def test_partitioned_fp4_validates_caller_output(monkeypatch):
    monkeypatch.setattr(fp4_op, "validate_partition_arch", lambda tensor: None)
    partitioned_fp4 = inspect.unwrap(fp4_op.prims_ts_fp4_block_scale_moe_partitioned)
    with pytest.raises(ValueError, match="dtype of output"):
        partitioned_fp4(
            routing_logits=torch.empty(1, 2, dtype=torch.bfloat16),
            routing_bias=None,
            topk_ids=None,
            topk_weights=None,
            hidden_states=torch.empty(1, 32, dtype=torch.bfloat16),
            hidden_states_scale=None,
            gemm1_weights=(torch.empty(0, dtype=torch.uint8),) * 2,
            gemm1_weights_scale=(torch.empty(0, dtype=torch.uint8),) * 2,
            gemm2_weights=(torch.empty(0, dtype=torch.uint8),) * 2,
            gemm2_weights_scale=(torch.empty(0, dtype=torch.uint8),) * 2,
            output1_scale_scalar=torch.empty(2),
            output1_scale_gate_scalar=torch.empty(2),
            output2_scale_scalar=torch.empty(2),
            num_experts=2,
            top_k=1,
            n_group=None,
            topk_group=None,
            intermediate_size=64,
            local_expert_offset=0,
            local_num_experts=2,
            partition_resources=SimpleNamespace(),
            partition_strategy="tp_n_tp_n",
            output=torch.empty(1, 32, dtype=torch.float32),
        )


@pytest.mark.cpu_only
def test_staged_tensor_passes_only_unpartitioned_full_weights():
    runner = PrimsTsBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=1024,
        intermediate_size=512,
        num_experts=8,
    )
    gemm1 = torch.empty(0)
    gemm2 = torch.empty(0)
    kwargs = {"gemm1_weights": gemm1, "gemm2_weights": gemm2}

    assert runner._staged_tensor(kwargs, "gemm1_weights") is gemm1
    assert runner._staged_tensor(kwargs, "gemm2_weights") is gemm2
    partitioned_runner = PrimsTsPartitionedBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=1024,
        intermediate_size=512,
        num_experts=8,
        **_partition_runner_kwargs(),
    )
    assert partitioned_runner._staged_tensor(kwargs, "gemm1_weights") is None
    assert partitioned_runner._staged_tensor(kwargs, "gemm2_weights") is None


def _partition_geometry_runner(
    monkeypatch, tile_m_by_tactic, mapped=None, use_pdl=False
):
    runner = PrimsTsPartitionedBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=256,
        intermediate_size=128,
        num_experts=8,
        **_partition_runner_kwargs("tp_n_tp_n"),
    )

    class Config:
        is_swap_ab = True
        epi_tile_m = 128
        has_gated_epilogue = False
        has_output_block_scaling = False
        preferred_cluster_shape = (1, 1, 1)
        umma_cta_count = 1

        def __init__(self, tile_m):
            self.tile_m = tile_m
            self.use_pdl = use_pdl

        def build(self):
            return self

    def pair(tile_m):
        return type(
            "Pair",
            (),
            {
                "fc1": type("Gemm", (), {"cfg": Config(tile_m)})(),
                "fc2": type("Gemm", (), {"cfg": Config(tile_m)})(),
            },
        )()

    def map_tactic(tactic, moe_inputs, kwargs=None):
        if mapped is not None:
            mapped.append(tactic)
        return pair(tile_m_by_tactic[tactic])

    monkeypatch.setattr(runner, "_map_tactic", map_tactic)

    inputs = MoeRunnerInputs(
        output=torch.empty(1, 256),
        routing_logits=torch.empty(1, 8),
        topk_ids=torch.empty(1, dtype=torch.int32),
        expert_weights=torch.empty(1, dtype=torch.bfloat16),
        hidden_states=torch.empty(1, 256),
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    ).to_list()
    return runner, inputs


@pytest.mark.cpu_only
@pytest.mark.parametrize("use_pdl", (False, True))
def test_partition_geometry_filters_tactics(monkeypatch, use_pdl):
    mapped = []
    runner, inputs = _partition_geometry_runner(
        monkeypatch, {1: 64, 2: 96}, mapped, use_pdl=use_pdl
    )
    cache_key = object()

    assert runner._filter_partition_geometry_tactics([1, 2], inputs, cache_key) == [1]
    assert runner._filter_partition_geometry_tactics([1, 2], inputs, cache_key) == [1]
    assert mapped == [1, 2]


@pytest.mark.cpu_only
def test_unpartitioned_geometry_filters_incomplete_wider_cga(monkeypatch):
    runner = PrimsTsBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=256,
        intermediate_size=128,
        num_experts=8,
    )

    class Config:
        is_swap_ab = True
        tile_m = 128
        umma_cta_count = 2

        def __init__(self, cluster_dim_x):
            self.preferred_cluster_shape = (cluster_dim_x, 1, 1)

        def build(self):
            return self

    def pair(cluster_dim_x):
        return SimpleNamespace(
            fc1=SimpleNamespace(cfg=Config(cluster_dim_x)),
            fc2=SimpleNamespace(cfg=Config(cluster_dim_x)),
        )

    monkeypatch.setattr(
        runner,
        "_map_tactic",
        lambda tactic, moe_inputs, kwargs=None: pair({1: 2, 2: 4}[tactic]),
    )
    inputs = MoeRunnerInputs(
        output=torch.empty(1, 256),
        routing_logits=torch.empty(1, 8),
        topk_ids=torch.empty(1, dtype=torch.int32),
        expert_weights=torch.empty(1, dtype=torch.bfloat16),
        hidden_states=torch.empty(1, 256),
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    ).to_list()

    assert runner._filter_partition_geometry_tactics(
        [1, 2], inputs, cache_key=object()
    ) == [1]


@pytest.mark.cpu_only
def test_partitioned_default_tactic_resolves_to_first_valid(monkeypatch):
    runner, inputs = _partition_geometry_runner(monkeypatch, {-1: 96, 1: 64, 2: 64})
    cache_key = object()
    monkeypatch.setattr(
        runner,
        "get_valid_tactics",
        lambda inputs, profile: runner._filter_partition_geometry_tactics(
            [-1, 1, 2], inputs, cache_key
        ),
    )
    forwarded = []
    monkeypatch.setattr(
        PrimsTsBf16MoERunner,
        "forward",
        lambda self, inputs, tactic=-1, **kwargs: forwarded.append(tactic),
    )

    runner.forward(inputs)
    runner.forward(inputs, tactic=2)
    assert forwarded == [1, 2]


@pytest.mark.cpu_only
def test_tp_n_tp_k_without_finalize_reduces_partials():
    runner = PrimsTsBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=8,
        intermediate_size=8,
        num_experts=8,
    )
    first = torch.ones(4, 8)
    second = torch.full_like(first, 2)

    result = runner._finish_partitioned(
        gemm2_output=first,
        second_gemm2_output=second,
        output=torch.empty_like(first),
        expert_weights=torch.empty(0),
        expanded_idx_to_permuted_idx=torch.empty(0, dtype=torch.int32),
        total_num_padded_tokens=torch.empty(0, dtype=torch.int32),
        num_tokens=4,
        num_experts=8,
        do_finalize=False,
        enable_pdl=False,
    )

    torch.testing.assert_close(result[0], torch.full_like(first, 3))


@pytest.mark.cpu_only
@pytest.mark.parametrize("enable_pdl", (False, True))
def test_partitioned_finalize_forwards_enable_pdl(enable_pdl):
    class Module:
        def trtllm_moe_run_finalize(self, *args):
            self.args = args

    module = Module()
    runner = PrimsTsBf16MoERunner(
        module,
        top_k=2,
        num_local_experts=8,
        hidden_size=8,
        intermediate_size=8,
        num_experts=8,
    )
    tensor = torch.empty(0)

    assert (
        runner._finish_partitioned(
            gemm2_output=tensor,
            second_gemm2_output=None,
            output=tensor,
            expert_weights=tensor,
            expanded_idx_to_permuted_idx=torch.empty(0, dtype=torch.int32),
            total_num_padded_tokens=torch.empty(0, dtype=torch.int32),
            num_tokens=4,
            num_experts=8,
            do_finalize=True,
            enable_pdl=enable_pdl,
        )
        == []
    )
    assert module.args[9] is enable_pdl


def _materialize_bf16_partition_execution(strategy: str):
    runner = PrimsTsPartitionedBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        **_partition_runner_kwargs(strategy),
    )
    partition = _partition_call_kwargs()["partition_resources"]
    cfg = SimpleNamespace(dtype_c_bits=16, has_output_block_scaling=False)
    gemm1_output = torch.empty((4, 128), dtype=torch.bfloat16)
    gemm2_output = torch.empty((4, 128), dtype=torch.bfloat16)
    execution = runner._materialize_partition_execution(
        partition=partition,
        fc1_cfg=cfg,
        fc2_cfg=cfg,
        gemm1_output=gemm1_output,
        gemm2_output=gemm2_output,
        gemm1_output_scale=None,
        local_fc1_width=64,
        full_fc1_pitch=128,
    )
    return execution, partition, gemm1_output, gemm2_output


@pytest.mark.cpu_only
def test_tp_n_tp_n_materializes_shared_strided_execution():
    execution, _, gemm1_output, gemm2_output = _materialize_bf16_partition_execution(
        "tp_n_tp_n"
    )

    assert tuple(output.shape for output in execution.fc1_outputs) == (
        (4, 64),
        (4, 64),
    )
    assert tuple(output.storage_offset() for output in execution.fc1_outputs) == (
        0,
        64,
    )
    assert tuple(output.storage_offset() for output in execution.fc2_outputs) == (
        0,
        64,
    )
    assert all(activation is gemm1_output for activation in execution.fc2_activations)
    assert execution.fc1_sf_offsets == (0, 0)
    assert execution.fc2_activation_scales == (None, None)
    assert execution.fc1_row_stride == 128
    assert execution.fc2_row_stride == 128
    assert execution.fc2_intermediate_size == 128
    assert execution.local_fc2_width == 64
    assert execution.rejoin_after_fc1
    assert execution.second_fc2_output is None
    assert execution.fc2_outputs[0].untyped_storage().data_ptr() == (
        gemm2_output.untyped_storage().data_ptr()
    )


@pytest.mark.cpu_only
def test_tp_n_tp_n_uses_unshifted_sf_buffer_with_logical_offsets():
    runner = PrimsTsPartitionedBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=384,
        num_experts=8,
        **_partition_runner_kwargs("tp_n_tp_n"),
    )
    partition = _partition_call_kwargs()["partition_resources"]
    fc1_cfg = SimpleNamespace(dtype_c_bits=8, has_output_block_scaling=True)
    fc2_cfg = SimpleNamespace(dtype_c_bits=16, has_output_block_scaling=False)
    gemm1_output = torch.empty((4, 384), dtype=torch.uint8)
    gemm1_output_scale = torch.empty(4096, dtype=torch.uint8)
    gemm2_output = torch.empty((4, 128), dtype=torch.bfloat16)

    execution = runner._materialize_partition_execution(
        partition=partition,
        fc1_cfg=fc1_cfg,
        fc2_cfg=fc2_cfg,
        gemm1_output=gemm1_output,
        gemm2_output=gemm2_output,
        gemm1_output_scale=gemm1_output_scale,
        local_fc1_width=192,
        full_fc1_pitch=384,
    )

    assert all(scale is gemm1_output_scale for scale in execution.fc1_scales)
    assert execution.fc1_sf_offsets == (0, 192)


@pytest.mark.cpu_only
def test_tp_n_tp_k_materializes_local_compact_execution():
    execution, _partition, gemm1_output, gemm2_output = (
        _materialize_bf16_partition_execution("tp_n_tp_k")
    )

    assert tuple(output.shape for output in execution.fc1_outputs) == (
        (4, 64),
        (4, 64),
    )
    assert all(output.is_contiguous() for output in execution.fc1_outputs)
    assert all(
        output.untyped_storage().data_ptr() == gemm1_output.untyped_storage().data_ptr()
        for output in execution.fc1_outputs
    )
    assert tuple(output.storage_offset() for output in execution.fc1_outputs) == (
        0,
        4 * 64,
    )
    assert all(
        activation is fc1_output
        for activation, fc1_output in zip(
            execution.fc2_activations, execution.fc1_outputs, strict=True
        )
    )
    assert execution.fc2_activation_scales == execution.fc1_scales == (None, None)
    assert execution.fc1_sf_offsets == (0, 0)
    assert execution.fc2_outputs[0] is gemm2_output
    assert execution.fc2_outputs[1] is not gemm2_output
    assert execution.fc2_outputs[1].shape == gemm2_output.shape
    assert execution.fc2_outputs[1].dtype == gemm2_output.dtype
    assert execution.fc2_outputs[1].device == gemm2_output.device
    assert execution.second_fc2_output is execution.fc2_outputs[1]
    assert execution.fc1_row_stride is None
    assert execution.fc2_row_stride is None
    assert execution.fc2_intermediate_size == 64
    assert execution.local_fc2_width == 128
    assert not execution.rejoin_after_fc1


@pytest.mark.cpu_only
def test_partitioned_runner_rejects_non_swap_ab_tactic():
    runner = PrimsTsBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
    )
    config = type("Config", (), {"is_swap_ab": False})()

    with pytest.raises(ValueError, match="requires swapAB"):
        runner._run_partitioned_gemm_pair(
            **_partition_pair_kwargs(),
            builder=None,
            fc1_cfg=config,
            fc2_cfg=config,
            common_io_kwargs={},
            fc1_label="fc1",
            fc2_label="fc2",
            gemm1_output=torch.empty(0),
            gemm2_output=torch.empty(0),
        )


@pytest.mark.cpu_only
def test_partitioned_runner_rejects_non_gated_activation():
    runner = PrimsTsBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        activation_type=ActivationType.Relu,
    )
    config = type("Config", (), {"is_swap_ab": True})()

    with pytest.raises(NotImplementedError, match="requires a gated"):
        runner._run_partitioned_gemm_pair(
            **_partition_pair_kwargs(),
            builder=None,
            fc1_cfg=config,
            fc2_cfg=config,
            common_io_kwargs={},
            fc1_label="fc1",
            fc2_label="fc2",
            gemm1_output=torch.empty(0),
            gemm2_output=torch.empty(0),
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "runner_name",
    [
        "PrimsTsPartitionedBf16MoERunner",
        "PrimsTsPartitionedNvfp4MoERunner",
        "PrimsTsPartitionedMxfp4Mxfp8MoERunner",
        "PrimsTsPartitionedMxfp4Bf16MoERunner",
        "PrimsTsPartitionedFp8PerTensorMoERunner",
        "PrimsTsPartitionedFp8BlockScaleMoERunner",
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [ActivationType.Identity, ActivationType.Silu, ActivationType.Relu2],
)
def test_partitioned_runner_rejects_non_gated_activation_at_construction(
    runner_name, activation_type
):
    with pytest.raises(
        NotImplementedError,
        match=f"requires a gated activation, got {activation_type.name}",
    ):
        getattr(runner_module, runner_name)(
            None,
            top_k=2,
            num_local_experts=8,
            hidden_size=1024,
            intermediate_size=512,
            num_experts=8,
            activation_type=activation_type.value,
            **_partition_runner_kwargs(),
        )


@pytest.mark.cpu_only
def test_partitioned_runner_rejects_bias_before_launch():
    runner = PrimsTsBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
    )
    config = SimpleNamespace(is_swap_ab=True)

    with pytest.raises(NotImplementedError, match="row-sharded bias"):
        runner._run_partitioned_gemm_pair(
            **_partition_pair_kwargs(),
            builder=None,
            fc1_cfg=config,
            fc2_cfg=config,
            common_io_kwargs={"gemm1_bias": torch.empty(0), "gemm2_bias": None},
            fc1_label="fc1",
            fc2_label="fc2",
            gemm1_output=torch.empty(0),
            gemm2_output=torch.empty(0),
        )


@pytest.mark.cpu_only
def test_partition_launch_failure_joins_and_next_invocation_succeeds(monkeypatch):
    runner = PrimsTsPartitionedBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        **_partition_runner_kwargs("tp_n_tp_k"),
    )
    events = []

    class ForkJoin:
        def fork(self, stream):
            events.append(("fork", stream))

        def join(self, stream):
            events.append(("join", stream))

    streams = (SimpleNamespace(name="p0"), SimpleNamespace(name="p1"))
    partition = SimpleNamespace(streams=streams, fork_join=ForkJoin())
    execution = SimpleNamespace(
        fc1_outputs=(torch.empty(4, 64), torch.empty(4, 64)),
        fc1_scales=(None, None),
        fc1_sf_offsets=(0, 0),
        fc2_activations=(torch.empty(4, 64), torch.empty(4, 64)),
        fc2_activation_scales=(None, None),
        fc2_outputs=(torch.empty(4, 128), torch.empty(4, 128)),
        fc1_row_stride=None,
        fc2_row_stride=None,
        fc2_intermediate_size=64,
        local_fc2_width=128,
        rejoin_after_fc1=False,
        second_fc2_output=torch.empty(4, 128),
    )
    cfg = SimpleNamespace(
        is_swap_ab=True,
        has_gated_epilogue=False,
        has_output_block_scaling=False,
        use_pdl=False,
    )
    weights = (torch.tensor(0), torch.tensor(1))
    monkeypatch.setattr(runner_module, "validate_partition_shape", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module, "validate_partition_tactic", lambda *a, **k: None
    )
    monkeypatch.setattr(
        runner, "_materialize_partition_execution", lambda **kwargs: execution
    )
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "current_stream",
        lambda device=None: "caller",
    )
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "stream",
        lambda stream: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runner, "_cuda_driver_stream", lambda stream: stream.name)
    monkeypatch.setattr(runner, "_launch_args", lambda io, stream: ())
    monkeypatch.setattr(runner_module, "stable_config_hash", lambda cfg: "hash")

    should_fail = True

    def get_compiled(_hash, label, io, stream):
        events.append(("compile", label, io["partition_id"]))

        def launch():
            nonlocal should_fail
            events.append(("launch", label, io["partition_id"]))
            if label == "fc2" and io["partition_id"] == 1 and should_fail:
                should_fail = False
                raise RuntimeError("injected second-partition failure")

        return launch

    monkeypatch.setattr(runner_module, "get_compiled_gemm", get_compiled)

    def builder(*, fc, cfg, **kwargs):
        return {
            "cfg": cfg,
            "partition_id": int(kwargs["gemm1_weights"].item()),
        }

    call = dict(
        partition=partition,
        partition_fc1_weights=weights,
        partition_fc2_weights=weights,
        partition_fc1_scales=None,
        partition_fc2_scales=None,
        builder=builder,
        fc1_cfg=cfg,
        fc2_cfg=cfg,
        common_io_kwargs={
            "hidden_states": torch.empty(4, 128),
            "gemm1_bias": None,
            "gemm2_bias": None,
        },
        fc1_label="fc1",
        fc2_label="fc2",
        gemm1_output=torch.empty(4, 128),
        gemm2_output=torch.empty(4, 128),
    )
    with pytest.raises(RuntimeError, match="injected second-partition failure"):
        runner._run_partitioned_gemm_pair(**call)
    assert events.index(("fork", "caller")) > events.index(("compile", "fc2", 1))
    assert events[-1] == ("join", "caller")

    runner._run_partitioned_gemm_pair(**call)
    assert events[-1] == ("join", "caller")


@pytest.mark.cpu_only
def test_partition_pair_failure_exits_launch_guard(monkeypatch):
    runner = PrimsTsPartitionedBf16MoERunner(
        None,
        top_k=2,
        num_local_experts=8,
        hidden_size=128,
        intermediate_size=128,
        num_experts=8,
        **_partition_runner_kwargs("tp_n_tp_k"),
    )
    guard_events = []

    @contextlib.contextmanager
    def launch_guard():
        guard_events.append("enter")
        try:
            yield
        finally:
            guard_events.append("exit")

    partition = SimpleNamespace(fork_join=SimpleNamespace(), launch_guard=launch_guard)
    monkeypatch.setattr(
        runner,
        "_partition_shards",
        lambda kwargs: (partition, (), (), None, None),
    )
    monkeypatch.setattr(
        runner,
        "_run_partitioned_gemm_pair",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )

    with pytest.raises(RuntimeError, match="injected"):
        runner._run_partitioned(
            builder=None,
            fc1_cfg=None,
            fc2_cfg=None,
            common_io_kwargs={},
            fc1_label="fc1",
            fc2_label="fc2",
            gemm1_output=torch.empty(0),
            gemm2_output=torch.empty(0),
            output=torch.empty(0),
            expert_weights=torch.empty(0),
            expanded_idx_to_permuted_idx=torch.empty(0, dtype=torch.int32),
            total_num_padded_tokens=torch.empty(0, dtype=torch.int32),
            num_tokens=0,
            kwargs={"num_experts": 8, "do_finalize": False},
        )
    assert guard_events == ["enter", "exit"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (12, 0)])
def test_partitioned_runner_rejects_unsupported_architecture(monkeypatch, capability):
    tensor = SimpleNamespace(device=torch.device("cuda"))
    monkeypatch.setattr(
        "flashinfer.prims_ts.moe.partition.get_compute_capability",
        lambda device: capability,
    )

    major, minor = capability
    with pytest.raises(
        NotImplementedError, match=f"only supported on SM107 .*got SM{major}{minor}"
    ):
        validate_partition_arch(tensor)


@pytest.mark.cpu_only
def test_partitioned_runner_accepts_sm107(monkeypatch):
    tensor = SimpleNamespace(device=torch.device("cuda"))
    monkeypatch.setattr(
        "flashinfer.prims_ts.moe.partition.get_compute_capability",
        lambda device: (10, 7),
    )

    validate_partition_arch(tensor)


@pytest.mark.cpu_only
def test_torch_tensor_is_preserved():
    tensor = torch.arange(4)

    (view,) = _torch_views_of_ffi_tensors([tensor])

    assert view is tensor


@pytest.mark.cpu_only
def test_optional_ffi_tensor_is_preserved():
    assert _torch_views_of_ffi_tensors([None]) == [None]


@pytest.mark.cpu_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_deepseek_staged_buffers_share_token_axis():
    """DeepSeek fp8: all staged buffers must use the same padded token extent.

    The 128-KiB allocation minimum can pad narrow FP8 FC1 buffers to more rows
    than the wider BF16 FC2 output. Prims-TS also uses the FP32 scale width as
    its token extent, so mismatched allocations make FC2 write past its buffer.
    """
    major, minor = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("Prims-TS MoE requires an SM100-family GPU")
    if not is_prims_ts_available():
        pytest.skip("Prims-TS dependencies are unavailable")

    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    device = torch.device("cuda")
    num_tokens = 32
    num_experts = 8
    top_k = 2
    hidden_size = 1024
    intermediate_size = 512
    pair = map_trtllm_deepseek_fp8_moe_tactic(
        [-1, -1],
        sm_arch=major * 10 + minor,
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_experts,
        weight_layout=int(WeightLayout.MajorK),
        enable_pdl=False,
    )
    empty_indices = torch.empty(0, dtype=torch.int32, device=device)
    empty_weights = torch.empty(0, dtype=torch.bfloat16, device=device)

    routing_out = (
        get_trtllm_moe_sm100_module().moe_op.trtllm_moe_run_routing_fp8_block_scale(
            torch.zeros((num_tokens, num_experts), dtype=torch.bfloat16, device=device),
            None,
            empty_indices,
            empty_weights,
            torch.zeros(
                (num_tokens, hidden_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.ones(
                (hidden_size // 128, num_tokens),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (num_experts, 2 * intermediate_size, hidden_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.ones(
                (num_experts, 2 * intermediate_size // 128, hidden_size // 128),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (num_experts, hidden_size, intermediate_size),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            torch.ones(
                (num_experts, hidden_size // 128, intermediate_size // 128),
                dtype=torch.float32,
                device=device,
            ),
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            0,
            num_experts,
            None,
            int(RoutingMethodType.Renormalize),
            False,
            [int(pair.tile_n), int(pair.moe_config_index)],
            int(WeightLayout.MajorK),
            int(ActivationType.Swiglu),
            int(Fp8QuantizationType.DeepSeekFp8),
            True,
            None,
        )
    )
    (
        _expert_weights,
        _expanded_idx_to_permuted_idx,
        _permuted_idx_to_token_idx,
        _tile_idx,
        _mn_limit,
        _num_non_exiting_ctas,
        _total_num_padded_tokens,
        gemm1_output,
        gemm1_output_scale,
        activation_output,
        activation_output_scale,
        gemm2_output,
    ) = _torch_views_of_ffi_tensors(routing_out)

    token_extents = {
        int(gemm1_output.shape[0]),
        int(gemm1_output_scale.shape[1]),
        int(activation_output.shape[0]),
        int(activation_output_scale.shape[1]),
        int(gemm2_output.shape[0]),
    }
    assert len(token_extents) == 1
    assert token_extents.pop() >= 256


@pytest.mark.cpu_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_unpartitioned_staged_routing_checks_full_weight_shape():
    major, minor = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("Prims-TS MoE requires an SM100-family GPU")
    if not is_prims_ts_available():
        pytest.skip("Prims-TS dependencies are unavailable")

    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    device = torch.device("cuda")
    num_tokens, num_experts, top_k = 1, 8, 2
    hidden_size, intermediate_size = 512, 256
    pair = map_trtllm_bf16_moe_tactic(
        [-1, -1],
        sm_arch=major * 10 + minor,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=num_tokens,
        top_k=top_k,
        num_local_experts=num_experts,
        weight_layout=int(WeightLayout.MajorK),
        fc1_has_bias=False,
        fc2_has_bias=False,
        enable_pdl=False,
        has_gemm1_alpha=False,
        has_gemm1_beta=False,
        has_gemm1_clamp_limit=False,
    )
    empty_indices = torch.empty(0, dtype=torch.int32, device=device)
    empty_weights = torch.empty(0, dtype=torch.bfloat16, device=device)
    wrong_fc1 = torch.empty(
        (num_experts, 2 * intermediate_size - 32, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    fc2 = torch.empty(
        (num_experts, hidden_size, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )

    with pytest.raises(RuntimeError, match="intermediate_size has incorrect shape"):
        get_trtllm_moe_sm100_module().moe_op.trtllm_moe_run_routing(
            torch.zeros((num_tokens, num_experts), dtype=torch.bfloat16, device=device),
            None,
            empty_indices,
            empty_weights,
            torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device),
            wrong_fc1,
            fc2,
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            0,
            num_experts,
            None,
            int(RoutingMethodType.Renormalize),
            True,
            int(WeightLayout.MajorK),
            False,
            [int(pair.tile_n), int(pair.moe_config_index)],
            int(ActivationType.Swiglu),
            True,
            None,
        )


@pytest.mark.cpu_only
def test_expert_unit_scales_are_reused():
    first = _get_expert_scale_ones(7, torch.device("cpu"))
    second = _get_expert_scale_ones(7, torch.device("cpu"))

    assert first is second
    torch.testing.assert_close(first, torch.ones(7))


@pytest.mark.cpu_only
def test_packed_c_row_stride_uses_storage_element_pitch():
    storage = torch.empty((4, 128), dtype=torch.uint8)
    partition = storage[:, 32:64]

    assert (
        _validate_c_output_layout(
            fc="fc1",
            output_buf=partition,
            logical_output_m=64,
            output_n=4,
            c_row_stride=256,
            logical_elements_per_storage_element=2,
        )
        == 256
    )


@pytest.mark.cpu_only
def test_packed_c_row_stride_rejects_wrong_storage_pitch():
    partition = torch.empty((4, 32), dtype=torch.uint8)

    with pytest.raises(ValueError, match="expected storage row stride 128"):
        _validate_c_output_layout(
            fc="fc1",
            output_buf=partition,
            logical_output_m=64,
            output_n=4,
            c_row_stride=256,
            logical_elements_per_storage_element=2,
        )


@pytest.mark.cpu_only
def test_packed_c_row_stride_rejects_narrow_partition():
    storage = torch.empty((4, 128), dtype=torch.uint8)
    partition = storage[:, :31]

    with pytest.raises(ValueError, match="output buffer width too small"):
        _validate_c_output_layout(
            fc="fc1",
            output_buf=partition,
            logical_output_m=64,
            output_n=4,
            c_row_stride=256,
            logical_elements_per_storage_element=2,
        )


@pytest.mark.cpu_only
def test_packed_c_row_stride_rejects_misaligned_partition():
    storage = torch.empty((4, 128), dtype=torch.uint8)
    partition = storage[:, 1:33]

    with pytest.raises(ValueError, match="pointer must be 16-byte aligned"):
        _validate_c_output_layout(
            fc="fc1",
            output_buf=partition,
            logical_output_m=64,
            output_n=4,
            c_row_stride=256,
            logical_elements_per_storage_element=2,
        )


@pytest.mark.cpu_only
def test_partition_output_view_preserves_global_row_pitch():
    cfg = type("Cfg", (), {"dtype_c_bits": 4})()
    storage = torch.empty((3, 128), dtype=torch.uint8)

    partition = PrimsTsBf16MoERunner._partition_output_view(
        storage,
        partition_id=1,
        logical_width=128,
        cfg=cfg,
    )

    assert partition.shape == (3, 64)
    assert partition.stride() == (128, 1)
    assert partition.data_ptr() == storage.data_ptr() + 64


@pytest.mark.cpu_only
def test_sf_c_m_offset_enforces_epilogue_alignment():
    cfg = SimpleNamespace(
        is_swap_ab=True,
        has_output_block_scaling=True,
        has_gated_epilogue=False,
        output_sf_block_size_c=16,
    )

    assert (
        _validate_sf_c_m_offset(
            fc="fc1",
            cfg=cfg,
            sf_c_m_offset=32,
            logical_output_m=64,
            pitch=128,
        )
        == 32
    )
    with pytest.raises(ValueError, match="must be a multiple of 32"):
        _validate_sf_c_m_offset(
            fc="fc1",
            cfg=cfg,
            sf_c_m_offset=16,
            logical_output_m=64,
            pitch=128,
        )
    with pytest.raises(ValueError, match="non-gated block-scaled output width"):
        _validate_sf_c_m_offset(
            fc="fc1",
            cfg=cfg,
            sf_c_m_offset=0,
            logical_output_m=40,
            pitch=128,
        )


@pytest.mark.cpu_only
def test_block_scaled_sf_output_checks_span_from_unshifted_base():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import SfLayout

    scale_storage = torch.empty(256, dtype=torch.uint8)
    _validate_block_scaled_sf_output(
        fc="fc1",
        output_scale=scale_storage,
        logical_output_m=128,
        output_n=16,
        pitch=256,
        sf_block_size=16,
        sf_layout=int(SfLayout.R8c4),
        m_offset=128,
    )

    with pytest.raises(ValueError, match="SF-C buffer too small"):
        _validate_block_scaled_sf_output(
            fc="fc1",
            output_scale=torch.empty(255, dtype=torch.uint8),
            logical_output_m=128,
            output_n=16,
            pitch=256,
            sf_block_size=16,
            sf_layout=int(SfLayout.R8c4),
            m_offset=128,
        )


@pytest.mark.cpu_only
def test_r128c4_sf_span_accounts_for_completed_n_subtile():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import SfLayout

    scale_storage = torch.empty(1012, dtype=torch.uint8)
    _validate_block_scaled_sf_output(
        fc="fc1",
        output_scale=scale_storage,
        logical_output_m=128,
        output_n=33,
        pitch=256,
        sf_block_size=16,
        sf_layout=int(SfLayout.R128c4),
    )

    with pytest.raises(ValueError, match="SF-C buffer too small"):
        _validate_block_scaled_sf_output(
            fc="fc1",
            output_scale=torch.empty(1011, dtype=torch.uint8),
            logical_output_m=128,
            output_n=33,
            pitch=256,
            sf_block_size=16,
            sf_layout=int(SfLayout.R128c4),
        )


@pytest.mark.cpu_only
def test_block_scaled_sf_output_rejects_misaligned_slice():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import SfLayout

    scale_storage = torch.empty(257, dtype=torch.uint8)
    with pytest.raises(ValueError, match="pointer must be 16-byte aligned"):
        _validate_block_scaled_sf_output(
            fc="fc1",
            output_scale=scale_storage[1:],
            logical_output_m=128,
            output_n=16,
            pitch=256,
            sf_block_size=16,
            sf_layout=int(SfLayout.R8c4),
        )


@pytest.mark.cpu_only
def test_ffi_tensor_can_be_converted_repeatedly():
    tensor = torch.arange(4)
    ffi_tensor = tvm_ffi.from_dlpack(tensor)

    (first_view,) = _torch_views_of_ffi_tensors([ffi_tensor])
    (second_view,) = _torch_views_of_ffi_tensors([ffi_tensor])

    assert first_view.data_ptr() == tensor.data_ptr()
    assert second_view.data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(first_view, tensor)
    torch.testing.assert_close(second_view, tensor)


@pytest.mark.cpu_only
def test_topk_initializer_matches_routing_representation():
    num_experts = 8

    assert _moe_topk_ids_init_for_routing(
        num_experts, RoutingInputMode.PackedPrecomputed
    ) is moe_topk_ids_init(num_experts, packed=True)
    assert _moe_topk_ids_init_for_routing(
        num_experts, RoutingInputMode.FromLogits
    ) is moe_topk_ids_init(num_experts, packed=True)
    assert _moe_topk_ids_init_for_routing(
        num_experts, RoutingInputMode.UnpackedPrecomputed
    ) is moe_topk_ids_init(num_experts, packed=False)


@pytest.mark.cpu_only
def test_mxfp4_mxfp8_runtime_routing_cache_preserves_runner_hash():
    moe_op = object()

    def make_runner():
        return PrimsTsMxfp4Mxfp8MoERunner(
            moe_op,
            top_k=4,
            num_local_experts=8,
            hidden_size=128,
            intermediate_size=128,
        )

    def make_inputs():
        return MoeRunnerInputs(
            output=torch.empty(4, 128),
            routing_logits=None,
            topk_ids=torch.zeros(4, 4, dtype=torch.int32),
            expert_weights=torch.empty(4, 4),
            hidden_states=torch.empty(4, 128),
            hidden_states_scale=torch.empty(4, 4, dtype=torch.uint8),
            gemm1_lora_delta=None,
            per_token_scale=None,
        )

    first, second = make_runner(), make_runner()
    first_inputs, second_inputs = make_inputs(), make_inputs()
    assert hash(first) == hash(second)

    first._make_tuning_config(first_inputs, tune_max_num_tokens=4)
    second._make_tuning_config(second_inputs, tune_max_num_tokens=4)

    assert first._topk_initializer_cache[0] is first_inputs.topk_ids
    assert second._topk_initializer_cache[0] is second_inputs.topk_ids
    assert hash(first) == hash(second)


@pytest.mark.cpu_only
def test_gemm_launches_use_local_expert_count():
    tree = ast.parse(inspect.getsource(runner_module))
    launch_builders = {
        "build_bf16_launch_io",
        "build_fp8_block_scale_launch_io",
        "build_fp8_per_tensor_launch_io",
        "build_mxfp4_bf16_launch_io",
        "build_mxfp4_mxfp8_launch_io",
        "build_nvfp4_launch_io",
    }
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in launch_builders
    ]

    assert Counter(call.func.id for call in launch_calls) == {
        "build_bf16_launch_io": 2,
        "build_fp8_block_scale_launch_io": 2,
        "build_fp8_per_tensor_launch_io": 2,
        "build_mxfp4_bf16_launch_io": 2,
        "build_mxfp4_mxfp8_launch_io": 4,
        "build_nvfp4_launch_io": 2,
    }
    direct_values = [
        keyword.value
        for call in launch_calls
        for keyword in call.keywords
        if keyword.arg == "num_experts"
    ]
    shared_values = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "common_io_kwargs"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "dict"
        for keyword in node.value.keywords
        if keyword.arg == "num_experts"
    ]

    expert_count_values = [*direct_values, *shared_values]
    assert len(expert_count_values) == 7
    for value in expert_count_values:
        assert ast.unparse(value) == "self.num_local_experts"

    # Tactic selection sizes tiles from the global expert count.
    (tactic_common,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_tactic_common"
    ]
    tactic_values = [
        keyword.value
        for node in ast.walk(tactic_common)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        for keyword in node.keywords
        if keyword.arg == "num_experts"
    ]
    assert [ast.unparse(value) for value in tactic_values] == ["self.num_experts"]


@pytest.mark.cpu_only
def test_prims_ts_ops_forward_routing_mode_to_tuning():
    calls = []
    for module in (bf16_op, fp4_op, fp8_op):
        tree = ast.parse(inspect.getsource(module))
        calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_make_tuning_config"
        )

    assert len(calls) == 5
    assert all(
        any(keyword.arg == "routing_input_mode" for keyword in call.keywords)
        for call in calls
    )


@pytest.mark.cpu_only
def test_routed_token_capacity_uses_local_expert_count(monkeypatch):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_config

    captured = []

    def fake_compute(**kwargs):
        captured.append(kwargs)
        return 1

    monkeypatch.setattr(
        batched_gemm_config,
        "compute_max_num_ctas_in_token_dim_for_moe",
        fake_compute,
    )
    runner = type("Runner", (), {"num_local_experts": 4, "top_k": 2})()
    inputs = type("Inputs", (), {"hidden_states": torch.empty((3, 8))})()

    assert (
        _routed_token_capacity(
            runner,
            inputs,
            [8, 0],
            torch.tensor([0], dtype=torch.int32),
            {"num_experts": 16},
        )
        == 8
    )
    assert captured[0]["num_experts"] == 4


def _routing_inputs(*, ids, weights, hidden=None):
    hidden = hidden if hidden is not None else torch.empty(4, 8)
    return MoeRunnerInputs(
        output=hidden,
        routing_logits=None,
        topk_ids=ids,
        expert_weights=weights,
        hidden_states=hidden,
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )


@pytest.mark.cpu_only
def test_staged_routing_io_rewrites_from_logits_2d_placeholders():
    hidden = torch.empty(4, 8)
    ids = torch.empty(4, 2, dtype=torch.int32)
    weights = torch.empty(4, 2, dtype=torch.bfloat16)
    topk_ids, expert_weights = _staged_routing_io(
        _routing_inputs(ids=ids, weights=weights, hidden=hidden),
        {"routing_input_mode": RoutingInputMode.FromLogits},
    )
    assert topk_ids.ndim == 1 and topk_ids.numel() == 0
    assert expert_weights.ndim == 1 and expert_weights.numel() == 0


@pytest.mark.cpu_only
def test_staged_routing_io_rewrites_packed_2d_weight_placeholders():
    hidden = torch.empty(4, 8)
    ids = torch.zeros(4, 2, dtype=torch.int32)
    weights = torch.empty(4, 2, dtype=torch.bfloat16)
    topk_ids, expert_weights = _staged_routing_io(
        _routing_inputs(ids=ids, weights=weights, hidden=hidden),
        {"routing_input_mode": RoutingInputMode.PackedPrecomputed},
    )
    assert topk_ids is ids
    assert expert_weights.ndim == 1
    assert expert_weights.numel() == 0


@pytest.mark.cpu_only
def test_select_expert_weights_ignores_1d_autotune_placeholder():
    routed = torch.ones(4, 2, dtype=torch.bfloat16)
    synthesized = torch.ones(8, dtype=torch.bfloat16)
    picked = _select_expert_weights(
        _routing_inputs(
            ids=torch.zeros(4, 2, dtype=torch.int32),
            weights=synthesized,
        ),
        routed,
        {"routing_input_mode": RoutingInputMode.PackedPrecomputed},
    )
    assert picked is routed
