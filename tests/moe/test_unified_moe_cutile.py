"""Unified cuTile MoE adapter tests."""

from __future__ import annotations

import pytest
import torch

from flashinfer.autotuner import AutoTuner
from flashinfer.cutile import is_cuda_tile_available
from flashinfer.fused_moe import (
    ActivationConfig,
    BackendOptions,
    CuTileBf16Config,
    CuTileBf16Runner,
    CuTileNvfp4Config,
    CuTileNvfp4Runner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe.cutile.fp4 import (
    _activation_quantize_config,
    _input_quantize_config,
)
from flashinfer.fused_moe.cutile.moe import _combine_tile_h, _permute_shape
from flashinfer.fused_moe.runners import (
    _CUTILE_BF16_DEFAULT_GEMM_CONFIGS,
    _CUTILE_BF16_DIVERSITY_GEMM_CONFIGS,
    _CUTILE_W4A4_DEFAULT_GEMM_CONFIGS,
    _CuTileBf16GemmProblem,
    _CuTileW4A4GemmProblem,
    _CuTileW4A4StageRunner,
    _cutile_bf16_config_rejection_reason,
    _cutile_w4a4_config_rejection_reason,
    _rank_cutile_bf16_gemm_configs,
    _rank_cutile_w4a4_gemm_configs,
)
from flashinfer.tllm_enums import ActivationType, RoutingMethodType

from .utils import RecordingTuner, compute_reference_moe


def _config(
    *,
    num_experts: int = 4,
    top_k: int = 2,
    intermediate_size: int = 256,
    tune_max_num_tokens: int = 128,
    **overrides,
) -> MoEConfig:
    values = dict(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions((CuTileBf16Config(),)),
        finalize=MoEFinalizeConfig(do_finalize=True),
        execution=ExecutionConfig(
            enable_pdl=False,
            tune_max_num_tokens=tune_max_num_tokens,
        ),
    )
    values.update(overrides)
    return MoEConfig(**values)


def test_cutile_bf16_config_architectures_and_registration():
    assert [arch for arch in range(140) if CuTileBf16Config.supported(arch)] == [
        89,
        90,
        120,
        121,
    ]
    assert not CuTileBf16Config.supported(100)
    assert _BACKEND_RUNNERS[CuTileBf16Config] is CuTileBf16Runner
    assert repr(CuTileBf16Config()) == "CuTileBf16Config()"


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "expected"),
    (
        (1, 64, 64),
        (1, 2048, 128),
        (64, 2048, 128),
        (65, 2048, 512),
        (256, 2048, 512),
        (257, 512, 512),
        (257, 2048, 1024),
    ),
)
def test_cutile_bf16_combine_tile_heuristic(num_tokens, hidden_size, expected):
    assert _combine_tile_h(num_tokens, hidden_size) == expected


@pytest.mark.parametrize(
    ("num_assignments", "num_experts", "expected_chunk"),
    (
        (48, 128, 8),
        (96, 128, 8),
        (384, 128, 16),
        (1536, 128, 64),
        (6144, 128, 128),
        (8192, 128, 128),
        (12288, 128, 128),
        (16384, 128, 32),
        (2048, 256, 64),
        (4096, 256, 64),
        (8192, 256, 32),
    ),
)
def test_cutile_permute_chunk_heuristic(num_assignments, num_experts, expected_chunk):
    assert _permute_shape(num_assignments, num_experts)[1] == expected_chunk


@pytest.mark.parametrize(
    ("config", "match"),
    (
        (
            _config(quant=QuantConfig(variant=QuantVariant.NVFP4)),
            "QuantVariant.NVFP4",
        ),
        (_config(activation=ActivationConfig.geglu), "Swiglu or Relu2"),
        (_config(finalize=MoEFinalizeConfig(do_finalize=False)), "do_finalize=True"),
        (_config(execution=ExecutionConfig(enable_pdl=True)), "PDL"),
        (
            _config(
                experts=ExpertConfig(
                    intermediate_size=256,
                    local_expert_offset=2,
                    local_num_experts=2,
                )
            ),
            "expert parallelism",
        ),
        (
            _config(
                routing=RoutingConfig(
                    num_experts=4,
                    top_k=2,
                    method=RoutingMethodType.DeepSeekV3,
                ),
                experts=ExpertConfig(
                    intermediate_size=256,
                    num_fused_shared_experts=1,
                ),
            ),
            "fused shared experts",
        ),
    ),
)
def test_cutile_bf16_runner_rejects_out_of_scope_configs(config, match):
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner.config = config
    runner.device = torch.device("cuda:0")
    runner._device_arch = 90

    with pytest.raises(NotImplementedError, match=match):
        runner.check_support()


def test_cutile_bf16_runner_rejects_unsupported_architecture():
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner.config = _config()
    runner.device = torch.device("cuda:0")
    runner._device_arch = 100

    with pytest.raises(RuntimeError, match="does not support SM100"):
        runner.check_support()


def test_cutile_bf16_fused_epilogue_rejects_unsupported_activation():
    from flashinfer.fused_moe.cutile import moe

    with pytest.raises(NotImplementedError, match="does not support activation"):
        moe._grouped_gemm(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            top_k=1,
            block_size=32,
            config=moe.GemmConfig(128, 32, 4),
            activation_type=ActivationType.Geglu,
        )


@pytest.mark.parametrize("num_tokens", (1, 1024))
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_bf16_tunes_stages_to_two_configs(monkeypatch, num_tokens, activation):
    tuner = RecordingTuner()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: tuner))
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._built = True
    runner._device_arch = 120
    runner._num_sms = 128
    runner.config = _config(
        num_experts=64,
        top_k=8,
        intermediate_size=768,
        tune_max_num_tokens=1024,
        activation=activation,
    )
    inputs = [
        torch.empty(num_tokens, 2048),
        torch.empty(num_tokens, 2048),
        torch.empty(num_tokens, 8, dtype=torch.int32),
        torch.empty(num_tokens, 8),
        torch.empty(0),
        torch.empty(0),
    ]

    tactics = runner.get_valid_tactics(inputs, None)

    expected_block = runner._fallback_tactic(inputs)[0]
    assert len(tactics) == 4
    assert {tactic[0] for tactic in tactics} == {expected_block}
    assert all(len(tactic) == 7 for tactic in tactics)
    assert len(tuner.calls) == 2
    assert all(k == 2 for _, k in tuner.calls)


@pytest.mark.parametrize(
    ("arch", "rows_per_expert"),
    ((89, 256), (90, 32), (120, 128), (121, 128)),
)
def test_cutile_bf16_non_gated_block_size_crossovers(arch, rows_per_expert):
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._device_arch = arch
    runner.config = _config(num_experts=128)

    assert runner._candidate_non_gated_block_sizes((rows_per_expert - 1) * 128) == (
        32,
        64,
    )
    assert runner._candidate_non_gated_block_sizes(rows_per_expert * 128) == (
        64,
        128,
    )


@pytest.mark.parametrize(
    ("arch", "rows_per_expert"),
    ((89, 64), (90, 64), (120, 64), (121, 64)),
)
def test_cutile_bf16_gated_block_size_crossovers(arch, rows_per_expert):
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._device_arch = arch
    runner.config = _config(num_experts=128, activation=ActivationConfig.swiglu)

    assert runner._gated_block_size((rows_per_expert - 1) * 128) == 32
    assert runner._gated_block_size(rows_per_expert * 128) == 64
    assert runner._gated_block_size(512 * 128) == 128


def test_cutile_bf16_gemm_heuristic_keeps_defaults_and_rejects_small_shapes():
    problem = _CuTileBf16GemmProblem(
        stage=1,
        arch=90,
        num_sms=132,
        num_assignments=4096,
        num_experts=128,
        block_size=32,
        n=1856,
        k=2688,
    )
    configs = _rank_cutile_bf16_gemm_configs(problem)

    assert len(configs) == 8
    assert set(_CUTILE_BF16_DEFAULT_GEMM_CONFIGS).issubset(configs)
    assert all(
        _cutile_bf16_config_rejection_reason(problem, config) is None
        for config in configs
    )

    sm120_problem = _CuTileBf16GemmProblem(
        stage=1,
        arch=120,
        num_sms=188,
        num_assignments=4096,
        num_experts=128,
        block_size=64,
        n=1856,
        k=2688,
    )
    sm120_configs = _rank_cutile_bf16_gemm_configs(sm120_problem)
    assert len(sm120_configs) == 10
    assert set(_CUTILE_BF16_DIVERSITY_GEMM_CONFIGS).issubset(sm120_configs)

    unsupported = _CuTileBf16GemmProblem(
        stage=2,
        arch=90,
        num_sms=132,
        num_assignments=1,
        num_experts=1,
        block_size=32,
        n=31,
        k=15,
    )
    with pytest.raises(NotImplementedError, match="no supported grouped GEMM"):
        _rank_cutile_bf16_gemm_configs(unsupported)


def test_cutile_bf16_gated_gemm1_problem_uses_preactivation_width():
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._device_arch = 90
    runner._num_sms = 132
    runner.config = _config(
        num_experts=256,
        top_k=8,
        intermediate_size=512,
        activation=ActivationConfig.swiglu,
    )
    inputs = [
        torch.empty(16, 2048),
        torch.empty(16, 2048),
        torch.empty(16, 8, dtype=torch.int32),
        torch.empty(16, 8),
        torch.empty(0),
        torch.empty(0),
    ]

    gemm1 = runner._gemm_problem(inputs, stage=1, block_size=32)
    gemm2 = runner._gemm_problem(inputs, stage=2, block_size=32)

    assert (gemm1.n, gemm1.k) == (1024, 2048)
    assert (gemm2.n, gemm2.k) == (2048, 512)


def test_cutile_bf16_direct_runner_rejects_tokens_above_tuning_ceiling():
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner.config = _config(tune_max_num_tokens=64)
    runner._built = True

    with pytest.raises(
        ValueError, match="num_tokens=65 exceeds tune_max_num_tokens=64"
    ):
        runner._ensure_workspace(65, 128)


def test_cutile_bf16_tuning_pre_hook_initializes_routing_and_workspace():
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner.config = _config(num_experts=4, top_k=2)
    activated = []
    runner._ensure_workspace = lambda tokens, hidden: activated.append((tokens, hidden))
    inputs = [
        torch.empty(7, 128),
        torch.empty(7, 128),
        torch.empty(7, 2, dtype=torch.int32),
        torch.empty(7, 2),
        torch.empty(1),
        torch.empty(1),
    ]

    runner._prepare_tuning_inputs(inputs)

    assert activated == [(7, 128)]
    torch.testing.assert_close(
        inputs[2], torch.arange(14, dtype=torch.int32).reshape(7, 2) % 4
    )
    torch.testing.assert_close(inputs[3], torch.full((7, 2), 0.5))


@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_bf16_workspace_cache_uses_token_buckets(activation):
    class KernelModule:
        def __init__(self):
            self.calls = []

        def allocate_workspace(self, **kwargs):
            self.calls.append(kwargs)
            return object()

    module = KernelModule()
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner.config = _config(activation=activation)
    runner.device = torch.device("cpu")
    runner._built = True
    runner._kernel_module = module
    runner._workspace_cache = {}
    runner._workspace = None

    runner._ensure_workspace(17, 128)
    first = runner._workspace
    runner._ensure_workspace(31, 128)
    same_bucket = runner._workspace
    runner._ensure_workspace(33, 128)
    second = runner._workspace
    runner._ensure_workspace(17, 128)

    assert runner._workspace is first
    assert same_bucket is first
    assert first is not second
    assert len(module.calls) == 2
    assert [call["num_tokens"] for call in module.calls] == [32, 64]
    assert module.calls[0]["block_sizes"] == (32, 64, 128)
    assert module.calls[0]["is_gated"] is activation.is_gated
    assert set(runner._workspace_cache) == {(32, 128), (64, 128)}


@pytest.mark.parametrize(
    ("num_experts", "hidden_size", "intermediate_size"),
    ((1, 16, 8), (3, 24, 12)),
)
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_prepare_cutile_bf16_weights_converts_native_layout(
    activation, num_experts, hidden_size, intermediate_size
):
    w1_rows = intermediate_size * (2 if activation.is_gated else 1)
    w1 = torch.arange(
        num_experts * w1_rows * hidden_size,
        dtype=torch.float32,
    ).reshape(num_experts, w1_rows, hidden_size)
    w2 = torch.arange(
        num_experts * hidden_size * intermediate_size,
        dtype=torch.float32,
    ).reshape(num_experts, hidden_size, intermediate_size)
    w1 = w1.to(torch.bfloat16)
    w2 = w2.to(torch.bfloat16)

    view = CuTileBf16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )

    assert set(view) == {"w1", "w2"}
    assert view["w1"].is_contiguous() and view["w2"].is_contiguous()
    if activation.is_gated:
        up, gate = w1.chunk(2, dim=1)
        expected_w1 = torch.cat((gate, up), dim=1).transpose(1, 2)
    else:
        expected_w1 = w1.transpose(1, 2)
    torch.testing.assert_close(view["w1"], expected_w1)
    torch.testing.assert_close(view["w2"], w2.transpose(1, 2))


def test_prepare_cutile_bf16_weights_rejects_invalid_source_contract():
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CuTileBf16Config.prepare_weights(
            torch.empty(2, 32, 16, dtype=torch.float16),
            torch.empty(2, 16, 16, dtype=torch.bfloat16),
            num_local_experts=2,
            hidden_size=16,
            intermediate_size=16,
        )

    with pytest.raises(ValueError, match="weight shapes"):
        CuTileBf16Config.prepare_weights(
            torch.empty(2, 31, 16, dtype=torch.bfloat16),
            torch.empty(2, 16, 16, dtype=torch.bfloat16),
            num_local_experts=2,
            hidden_size=16,
            intermediate_size=16,
        )

    with pytest.raises(ValueError, match="weight shapes"):
        CuTileBf16Config.prepare_weights(
            torch.empty(2, 32, 16, dtype=torch.bfloat16),
            torch.empty(2, 16, 16, dtype=torch.bfloat16),
            num_local_experts=2,
            hidden_size=16,
            intermediate_size=16,
            activation=ActivationConfig.relu2,
        )


def _cutile_device_is_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return CuTileBf16Config.supported(major * 10 + minor) and is_cuda_tile_available()


cutile_bf16_required = pytest.mark.skipif(
    not _cutile_device_is_supported(),
    reason="requires a working cuTile toolchain on SM89/SM90/SM120/SM121",
)


@cutile_bf16_required
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
@pytest.mark.parametrize(
    ("num_tokens", "num_experts", "top_k", "hidden_size", "intermediate_size"),
    (
        pytest.param(1, 2, 1, 64, 96, id="decode"),
        pytest.param(7, 3, 2, 96, 160, id="non-power-of-two-experts"),
        pytest.param(33, 8, 4, 256, 128, id="top-k-4"),
        pytest.param(4, 4, 2, 2048, 768, id="model-dimensions"),
    ),
)
def test_cutile_bf16_runner_matches_reference(
    activation, num_tokens, num_experts, top_k, hidden_size, intermediate_size
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    config = _config(
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        tune_max_num_tokens=max(128, num_tokens),
        activation=activation,
    )
    hidden_states = (
        torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) / 2
    )
    canonical_w1 = (
        torch.randn(
            num_experts,
            intermediate_size * (2 if activation.is_gated else 1),
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8
    )
    canonical_w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8
    )
    topk_ids = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    native = CuTileBf16Config.prepare_weights(
        canonical_w1,
        canonical_w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_bf16", native)
    activations = MoEActivationPack(
        hidden_states,
        None,
        topk_ids,
        topk_weights,
    )
    runner = CuTileBf16Runner(config, device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(activations, weights)

    actual = runner.forward(inputs, tactic=-1)
    expected = compute_reference_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        canonical_w1,
        canonical_w2,
        activation.type,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-1)


@cutile_bf16_required
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_bf16_runner_supports_tuned_block_sizes(activation):
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_experts, top_k = 17, 4, 2
    hidden_size, intermediate_size = 128, 256
    config = _config(
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    canonical_w1 = torch.randn(
        num_experts,
        intermediate_size * (2 if activation.is_gated else 1),
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    canonical_w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    native = CuTileBf16Config.prepare_weights(
        canonical_w1,
        canonical_w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_bf16", native)
    runner = CuTileBf16Runner(config, device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(
        MoEActivationPack(hidden_states, None, topk_ids, topk_weights), weights
    )
    expected = compute_reference_moe(
        hidden_states,
        topk_ids,
        topk_weights,
        canonical_w1,
        canonical_w2,
        activation.type,
    )
    tuned_gemms = {
        89: (256, 32, 2, 256, 32, 1),
        90: (256, 64, 2, 256, 64, 2),
        120: (256, 32, 2, 256, 32, 2),
        121: (256, 32, 2, 256, 32, 2),
    }[runner._device_arch]
    for block_size in runner._block_sizes:
        tuned_actual = runner.forward(inputs, tactic=(block_size, *tuned_gemms))
        torch.testing.assert_close(tuned_actual, expected, rtol=3e-2, atol=5e-1)


@cutile_bf16_required
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_bf16_runner_is_cuda_graph_capturable(activation):
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size = 4, 128, 256
    config = _config(
        intermediate_size=intermediate_size,
        tune_max_num_tokens=4,
        activation=activation,
    )
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    canonical_w1 = torch.randn(
        4,
        intermediate_size * (2 if activation.is_gated else 1),
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    canonical_w2 = torch.randn(
        4,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    ids = torch.arange(8, dtype=torch.int32, device=device).reshape(4, 2) % 4
    routing_weights = torch.full((4, 2), 0.5, device=device)
    weights = MoEWeightPack()
    weights.prepare_for(
        "cutile_bf16",
        CuTileBf16Config.prepare_weights(
            canonical_w1,
            canonical_w2,
            num_local_experts=4,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
        ),
    )
    runner = CuTileBf16Runner(config, device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(
        MoEActivationPack(hidden_states, None, ids, routing_weights), weights
    )
    tuned_gemms = {
        89: (256, 32, 2, 256, 32, 1),
        90: (256, 64, 2, 256, 64, 2),
        120: (256, 32, 2, 256, 32, 2),
        121: (256, 32, 2, 256, 32, 2),
    }[runner._device_arch]
    tactic = (128, *tuned_gemms)
    runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic=tactic)
    graph.replay()
    torch.cuda.synchronize()

    expected = compute_reference_moe(
        hidden_states,
        ids,
        routing_weights,
        canonical_w1,
        canonical_w2,
        activation.type,
    )
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=5e-1)


@cutile_bf16_required
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_bf16_runs_through_unified_layer(activation):
    torch.manual_seed(1)
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size = 4, 128, 256
    config = _config(
        intermediate_size=intermediate_size,
        tune_max_num_tokens=4,
        activation=activation,
    )
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    canonical_w1 = torch.randn(
        4,
        intermediate_size * (2 if activation.is_gated else 1),
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    canonical_w2 = torch.randn(
        4,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    ids = torch.arange(8, dtype=torch.int32, device=device).reshape(4, 2) % 4
    routing_weights = torch.full((4, 2), 0.5, device=device)
    weights = MoEWeightPack()
    weights.prepare_for(
        "cutile_bf16",
        CuTileBf16Config.prepare_weights(
            canonical_w1,
            canonical_w2,
            num_local_experts=4,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
        ),
    )

    layer = MoELayer(config, device)
    actual = layer(
        MoEActivationPack(hidden_states, None, ids, routing_weights), weights
    )
    expected = compute_reference_moe(
        hidden_states,
        ids,
        routing_weights,
        canonical_w1,
        canonical_w2,
        activation.type,
    )

    assert layer.winner_backend == "cutile_bf16"
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-1)


def _nvfp4_config(
    *,
    num_experts: int = 4,
    top_k: int = 2,
    intermediate_size: int = 128,
    activation: ActivationConfig = ActivationConfig.swiglu,
    max_num_tokens: int = 128,
) -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CuTileNvfp4Config(),)),
        finalize=MoEFinalizeConfig(do_finalize=True),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=max_num_tokens),
    )


def test_cutile_nvfp4_config_is_registered_and_arch_gated():
    assert [arch for arch in range(140) if CuTileNvfp4Config.supported(arch)] == [
        120,
        121,
    ]
    assert _BACKEND_RUNNERS[CuTileNvfp4Config] is CuTileNvfp4Runner


def test_cutile_nvfp4_swiglu_stage_search_is_curated():
    runner = object.__new__(CuTileNvfp4Runner)
    runner._built = True
    runner._device_arch = 120
    runner._num_sms = 188
    runner.config = _nvfp4_config(
        num_experts=64,
        top_k=8,
        intermediate_size=768,
    )
    inputs = [torch.empty(0) for _ in range(10)]
    inputs[1] = torch.empty(1, 2048)
    inputs[2] = torch.empty(8, dtype=torch.int32)
    inputs[6] = torch.empty(64, 2)

    gemm1_runner = _CuTileW4A4StageRunner(runner, 1, block_size=16)
    gemm2_runner = _CuTileW4A4StageRunner(runner, 2, block_size=16)
    assert runner._candidate_block_sizes(inputs[2].numel()) == (16, 32)
    small_gemm1 = gemm1_runner.get_valid_tactics(inputs, None)
    assert len(small_gemm1) == 9
    assert {tactic[0] for tactic in small_gemm1} == {0}

    inputs[2] = torch.empty(64, dtype=torch.int32)
    gemm1 = gemm1_runner.get_valid_tactics(inputs, None)
    gemm2 = gemm2_runner.get_valid_tactics(inputs, None)
    assert len(gemm1) == 9
    assert {tactic[0] for tactic in gemm1} == {0, 1}
    assert all(len(tactic) == 4 for tactic in gemm1)
    assert len(gemm2) == 9
    assert all(len(tactic) == 3 for tactic in gemm2)

    assert runner._candidate_block_sizes(64 * 9) == (32, 64, 128)


def test_cutile_nvfp4_tactics_include_partial_128_tiles():
    problem = _CuTileW4A4GemmProblem(
        stage=1,
        arch=120,
        num_sms=188,
        num_assignments=8,
        num_experts=4,
        block_size=32,
        n=192,
        k=192,
        fused_epilogue=True,
        is_gated=False,
        input_sorted=False,
    )
    configs = _rank_cutile_w4a4_gemm_configs(problem)

    assert (128, 128, 2) in configs
    assert (256, 128, 1) in configs
    assert set(_CUTILE_W4A4_DEFAULT_GEMM_CONFIGS).issubset(configs)
    assert all(
        _cutile_w4a4_config_rejection_reason(problem, config) is None
        for config in configs
    )


def test_cutile_nvfp4_gated_fusion_requires_wide_gemm1_tile():
    problem = _CuTileW4A4GemmProblem(
        stage=1,
        arch=120,
        num_sms=188,
        num_assignments=64,
        num_experts=256,
        block_size=16,
        n=512,
        k=2048,
        fused_epilogue=True,
        is_gated=True,
        input_sorted=False,
    )

    assert _cutile_w4a4_config_rejection_reason(problem, (256, 64, 2)) is None
    assert "effective tile_n" in str(
        _cutile_w4a4_config_rejection_reason(problem, (128, 64, 2))
    )


@pytest.mark.parametrize("num_tokens", (1, 1024))
def test_cutile_nvfp4_relu2_tunes_stages_to_two_configs(monkeypatch, num_tokens):
    tuner = RecordingTuner()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: tuner))
    runner = CuTileNvfp4Runner.__new__(CuTileNvfp4Runner)
    runner._built = True
    runner._device_arch = 120
    runner._num_sms = 188
    runner.config = _nvfp4_config(
        num_experts=128,
        top_k=6,
        intermediate_size=1856,
        activation=ActivationConfig.relu2,
        max_num_tokens=1024,
    )
    inputs = [torch.empty(0) for _ in range(10)]
    inputs[0] = torch.empty(num_tokens, 2688)
    inputs[1] = torch.empty(num_tokens, 2688)
    inputs[2] = torch.empty(num_tokens, 6, dtype=torch.int32)
    inputs[3] = torch.empty(num_tokens, 6)

    tactics = runner.get_valid_tactics(inputs, None)

    expected_block = runner._w4a4_fallback_tactic(inputs)[0]
    assert len(tactics) == 4
    assert {tactic[0] for tactic in tactics} == {expected_block}
    assert {tactic[1] for tactic in tactics} == {1}
    assert all(len(tactic) == 8 for tactic in tactics)
    assert len(tuner.calls) == 2
    assert all(k == 2 for _, k in tuner.calls)


@pytest.mark.parametrize(
    ("num_tokens", "expected_fusion"), ((1, 0), (2, 0), (8, 1), (512, 0))
)
def test_cutile_nvfp4_swiglu_tunes_producer_stage_to_two_configs(
    monkeypatch, num_tokens, expected_fusion
):
    tuner = RecordingTuner()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: tuner))
    runner = CuTileNvfp4Runner.__new__(CuTileNvfp4Runner)
    runner._built = True
    runner._device_arch = 120
    runner._num_sms = 188
    runner.config = _nvfp4_config(
        num_experts=256,
        top_k=8,
        intermediate_size=512,
        activation=ActivationConfig.swiglu,
        max_num_tokens=512,
    )
    inputs = [torch.empty(0) for _ in range(10)]
    inputs[0] = torch.empty(num_tokens, 2048)
    inputs[1] = torch.empty(num_tokens, 2048)
    inputs[2] = torch.empty(num_tokens, 8, dtype=torch.int32)
    inputs[3] = torch.empty(num_tokens, 8)

    tactics = runner.get_valid_tactics(inputs, None)

    expected_block = runner._w4a4_fallback_tactic(inputs)[0]
    assert len(tactics) == 4
    assert {tactic[0] for tactic in tactics} == {expected_block}
    assert {tactic[1] for tactic in tactics} == {expected_fusion}
    assert all(len(tactic) == 8 for tactic in tactics)
    if num_tokens == 1:
        assert any(tactic[2:5] == (128, 256, 2) for tactic in tactics)
        assert any(tactic[5:8] == (128, 128, 2) for tactic in tactics)
    elif num_tokens == 2:
        assert any(tactic[2:5] == (256, 256, 1) for tactic in tactics)
        assert any(tactic[5:8] == (256, 256, 1) for tactic in tactics)
    assert len(tuner.calls) == 2
    assert all(k == 2 for _, k in tuner.calls)


@pytest.mark.parametrize(
    ("num_tokens", "expected_block", "expected_fusion"),
    (
        (256, 16, 1),
        (512, 32, 0),
        (1024, 32, 1),
        (2048, 64, 1),
        (4096, 128, 0),
        (8192, 64, 1),
    ),
)
def test_cutile_nvfp4_swiglu_fallback_heuristic(
    num_tokens, expected_block, expected_fusion
):
    runner = CuTileNvfp4Runner.__new__(CuTileNvfp4Runner)
    runner._device_arch = 120
    runner._num_sms = 188
    runner.config = _nvfp4_config(
        num_experts=256,
        top_k=8,
        intermediate_size=512,
        activation=ActivationConfig.swiglu,
        max_num_tokens=8192,
    )
    inputs = [torch.empty(0) for _ in range(10)]
    inputs[1] = torch.empty(num_tokens, 2048)
    inputs[2] = torch.empty(num_tokens, 8, dtype=torch.int32)

    tactic = runner._w4a4_fallback_tactic(inputs)
    assert tactic[:2] == (expected_block, expected_fusion)


@pytest.mark.parametrize("num_tokens,expected_block", ((512, 32), (1024, 64)))
def test_cutile_nvfp4_relu2_block_size_heuristic(num_tokens, expected_block):
    runner = CuTileNvfp4Runner.__new__(CuTileNvfp4Runner)
    runner._device_arch = 120
    runner._num_sms = 188
    runner.config = _nvfp4_config(
        num_experts=128,
        top_k=6,
        intermediate_size=1856,
        activation=ActivationConfig.relu2,
        max_num_tokens=1024,
    )
    inputs = [torch.empty(0) for _ in range(10)]
    inputs[1] = torch.empty(num_tokens, 2688)
    inputs[2] = torch.empty(num_tokens, 6, dtype=torch.int32)

    assert runner._w4a4_fallback_tactic(inputs)[0] == expected_block


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "num_sms", "scale_row_major", "expected"),
    (
        (8, 4096, 188, False, (2, 128, 0)),
        (16, 2688, 188, False, (4, 128, 0)),
        (32, 2048, 188, False, (2, 256, 0)),
        (32, 4096, 188, False, (2, 256, 4)),
        (64, 2688, 188, False, (16, 64, 4)),
        (128, 4096, 188, False, (8, 128, 4)),
        (256, 2048, 188, False, (8, 128, 4)),
        (512, 4096, 188, False, (32, 64, 4)),
        (512, 2688, 188, True, (16, 128, 4)),
        (1024, 2688, 188, True, (64, 64, 4)),
        (1024, 4096, 188, True, (8, 256, 4)),
    ),
)
def test_cutile_nvfp4_input_quantize_heuristic(
    num_tokens, hidden_size, num_sms, scale_row_major, expected
):
    assert (
        _input_quantize_config(num_tokens, hidden_size, num_sms, scale_row_major)
        == expected
    )


@pytest.mark.parametrize(
    ("rows", "intermediate_size", "num_sms", "scale_row_major", "expected"),
    (
        (64, 512, 188, False, (64, 2)),
        (2048, 512, 188, False, (64, 4)),
        (64, 512, 188, True, (64, 4)),
    ),
)
def test_cutile_nvfp4_activation_quantize_heuristic(
    rows, intermediate_size, num_sms, scale_row_major, expected
):
    assert (
        _activation_quantize_config(rows, intermediate_size, num_sms, scale_row_major)
        == expected
    )


@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_prepare_cutile_nvfp4_weights(activation):
    num_experts, hidden_size, intermediate_size = 2, 128, 128
    w1_rows = intermediate_size * (2 if activation.is_gated else 1)
    w1 = torch.arange(num_experts * w1_rows * hidden_size // 2, dtype=torch.uint8)
    w1 = w1.reshape(num_experts, w1_rows, hidden_size // 2)
    s1 = torch.arange(
        num_experts * w1_rows * hidden_size // 16, dtype=torch.uint8
    ).reshape(num_experts, w1_rows, hidden_size // 16)
    s1 = s1.view(torch.float8_e4m3fn)
    w2 = torch.arange(
        num_experts * hidden_size * intermediate_size // 2, dtype=torch.uint8
    ).reshape(num_experts, hidden_size, intermediate_size // 2)
    s2 = torch.arange(
        num_experts * hidden_size * intermediate_size // 16, dtype=torch.uint8
    ).reshape(num_experts, hidden_size, intermediate_size // 16)
    s2 = s2.view(torch.float8_e4m3fn)
    g1_shape = (num_experts, 2) if activation.is_gated else (num_experts,)
    g1 = torch.arange(1, 1 + torch.tensor(g1_shape).prod().item(), dtype=torch.float32)
    g1 = g1.reshape(g1_shape)
    g2 = torch.arange(1, num_experts + 1, dtype=torch.float32)

    view = CuTileNvfp4Config.prepare_weights(
        w1,
        s1,
        g1,
        w2,
        s2,
        g2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )

    assert set(view) == {
        "w1",
        "w1_scale",
        "w1_global_scale",
        "w2",
        "w2_scale",
        "w2_global_scale",
    }
    if activation.is_gated:
        up, gate = w1.chunk(2, dim=1)
        torch.testing.assert_close(view["w1"], torch.cat((gate, up), dim=1))
        torch.testing.assert_close(view["w1_global_scale"], g1[:, [1, 0]])
    assert view["w1_scale"].shape == (
        num_experts,
        w1_rows // 128,
        hidden_size // 64,
        32,
        16,
    )
    assert view["w2_scale"].shape == (
        num_experts,
        hidden_size // 128,
        intermediate_size // 64,
        32,
        16,
    )


def test_prepare_cutile_nvfp4_weights_rejects_misaligned_dimensions():
    tensors = (
        torch.empty(2, 256, 64, dtype=torch.uint8),
        torch.empty(2, 256, 8, dtype=torch.float8_e4m3fn),
        torch.ones(2, dtype=torch.float32),
        torch.empty(2, 128, 64, dtype=torch.uint8),
        torch.empty(2, 128, 8, dtype=torch.float8_e4m3fn),
        torch.ones(2, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="divisible by 64"):
        CuTileNvfp4Config.prepare_weights(
            *tensors,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=96,
        )


def _unswizzle_cutile_nvfp4_scales(
    scale: torch.Tensor, padded_n: int, k_groups: int
) -> torch.Tensor:
    num_experts = scale.shape[0]
    return (
        scale.reshape(num_experts * padded_n // 128, k_groups // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(num_experts, padded_n, k_groups)
    )


def test_prepare_cutile_nvfp4_weights_pads_only_scale_layout():
    num_experts, hidden_size, intermediate_size = 2, 192, 192
    w1 = torch.zeros(
        num_experts, intermediate_size, hidden_size // 2, dtype=torch.uint8
    )
    s1 = torch.arange(
        num_experts * intermediate_size * hidden_size // 16, dtype=torch.uint8
    ).reshape(num_experts, intermediate_size, hidden_size // 16)
    s1 = s1.view(torch.float8_e4m3fn)
    w2 = torch.zeros(
        num_experts, hidden_size, intermediate_size // 2, dtype=torch.uint8
    )
    s2 = torch.arange(
        num_experts * hidden_size * intermediate_size // 16, dtype=torch.uint8
    ).reshape(num_experts, hidden_size, intermediate_size // 16)
    s2 = s2.view(torch.float8_e4m3fn)

    view = CuTileNvfp4Config.prepare_weights(
        w1,
        s1,
        torch.ones(num_experts),
        w2,
        s2,
        torch.ones(num_experts),
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=ActivationConfig.relu2,
    )

    assert view["w1"].shape == w1.shape
    assert view["w2"].shape == w2.shape
    assert view["w1_scale"].shape == (num_experts, 2, 3, 32, 16)
    assert view["w2_scale"].shape == (num_experts, 2, 3, 32, 16)
    for actual, expected in ((view["w1_scale"], s1), (view["w2_scale"], s2)):
        unswizzled = _unswizzle_cutile_nvfp4_scales(actual, 256, 12)
        torch.testing.assert_close(
            unswizzled[:, :192].view(torch.uint8),
            expected.view(torch.uint8),
        )
        assert torch.count_nonzero(unswizzled[:, 192:].view(torch.uint8)) == 0


def _quantize_weights(weight: torch.Tensor):
    shape = weight.shape
    groups = weight.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    scale = (groups.abs().amax(dim=-1) / 6.0).to(torch.float8_e4m3fn)
    safe_scale = scale.float().clamp_min(2.0**-9)
    values = groups / safe_scale.unsqueeze(-1)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=weight.device,
    )
    codes = torch.bucketize(values.abs(), boundaries, right=False)
    codes = codes | ((values < 0).to(torch.int64) << 3)
    codes = codes.reshape(*shape)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    dequantized = values.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[codes & 7]
    dequantized = torch.where((codes & 8).bool(), -dequantized, dequantized)
    dequantized = (
        dequantized.reshape(*shape[:-1], shape[-1] // 16, 16)
        * scale.float().unsqueeze(-1)
    ).reshape(shape)
    return packed.contiguous(), scale.contiguous(), dequantized.to(torch.bfloat16)


def _cutile_nvfp4_is_supported() -> bool:
    if not torch.cuda.is_available() or not is_cuda_tile_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return CuTileNvfp4Config.supported(major * 10 + minor)


cutile_nvfp4_required = pytest.mark.skipif(
    not _cutile_nvfp4_is_supported(),
    reason="requires a working cuTile toolchain on SM120/SM121",
)


@cutile_nvfp4_required
@pytest.mark.parametrize(
    ("activation", "intermediate_size", "scale_row_major"),
    (
        (ActivationConfig.swiglu, 128, False),
        (ActivationConfig.relu2, 128, True),
        (ActivationConfig.swiglu, 768, True),
        (ActivationConfig.relu2, 768, False),
    ),
)
def test_cutile_fused_activation_quantize_matches_unfused(
    activation, intermediate_size, scale_row_major
):
    from flashinfer.fused_moe.cutile import activation as activation_kernels
    from flashinfer.fused_moe.cutile import fp4

    torch.manual_seed(0)
    rows = 17
    padded_rows = 128
    input_size = intermediate_size * (2 if activation.is_gated else 1)
    x = torch.randn(rows, input_size, dtype=torch.bfloat16, device="cuda")
    activation_out = torch.empty(
        rows, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    expected_q = torch.empty(
        padded_rows, intermediate_size // 2, dtype=torch.uint8, device="cuda"
    )
    actual_q = torch.empty_like(expected_q)
    expected_q.fill_(0xA5)
    actual_q.copy_(expected_q)
    scale_shape = (
        (padded_rows, intermediate_size // 16)
        if scale_row_major
        else (intermediate_size // 16, padded_rows)
    )
    expected_scale = torch.empty(scale_shape, dtype=torch.float8_e4m3fn, device="cuda")
    actual_scale = torch.empty_like(expected_scale)
    expected_scale.view(torch.uint8).fill_(0xA5)
    actual_scale.view(torch.uint8).copy_(expected_scale.view(torch.uint8))

    activation_kernels.launch_activation(x, activation_out, activation.type)
    fp4._quantize(
        activation_out,
        expected_q,
        expected_scale,
        scale_row_major=scale_row_major,
    )
    fp4._launch_activation_quantize(
        x,
        actual_q,
        actual_scale,
        activation.type,
        scale_row_major=scale_row_major,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_q[: rows + 1], expected_q[: rows + 1], rtol=0, atol=0
    )
    actual_valid_scale = (
        actual_scale[: rows + 1] if scale_row_major else actual_scale[:, : rows + 1]
    )
    expected_valid_scale = (
        expected_scale[: rows + 1] if scale_row_major else expected_scale[:, : rows + 1]
    )
    torch.testing.assert_close(
        actual_valid_scale.view(torch.uint8),
        expected_valid_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_cutile_nvfp4_scale_layout_heuristic():
    from flashinfer.fused_moe.cutile.fp4 import _use_row_major_scale_layout

    assert _use_row_major_scale_layout(4096, 1856)
    assert not _use_row_major_scale_layout(4095, 1856)
    assert not _use_row_major_scale_layout(65535, 512)
    assert _use_row_major_scale_layout(65536, 512)


def _make_nvfp4_case(
    activation: ActivationConfig,
    *,
    num_tokens: int = 4,
    num_experts: int = 4,
    top_k: int = 2,
    hidden_size: int = 128,
    intermediate_size: int = 128,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    w1_rows = intermediate_size * (2 if activation.is_gated else 1)
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    w1 = (
        torch.randn(
            num_experts,
            w1_rows,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 8
    )
    w1_q, w1_scale, w1_dequant = _quantize_weights(w1)
    w2_q, w2_scale, w2_dequant = _quantize_weights(w2)
    ids = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    routing_weights = torch.rand(num_tokens, top_k, device=device)
    routing_weights /= routing_weights.sum(dim=1, keepdim=True)
    native = CuTileNvfp4Config.prepare_weights(
        w1_q,
        w1_scale,
        torch.ones(num_experts, device=device),
        w2_q,
        w2_scale,
        torch.ones(num_experts, device=device),
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_nvfp4", native)
    config = _nvfp4_config(
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        activation=activation,
        max_num_tokens=max(128, num_tokens),
    )
    activations = MoEActivationPack(hidden_states, None, ids, routing_weights)
    expected = compute_reference_moe(
        hidden_states,
        ids,
        routing_weights,
        w1_dequant,
        w2_dequant,
        activation.type,
    )
    return config, activations, weights, expected


@cutile_nvfp4_required
@pytest.mark.parametrize(
    ("activation", "num_tokens", "top_k"),
    (
        pytest.param(ActivationConfig.swiglu, 4, 2, id="swiglu"),
        pytest.param(ActivationConfig.relu2, 4, 2, id="relu2"),
        pytest.param(
            ActivationConfig.swiglu,
            21,
            3,
            id="bucketed-small-gated-activation",
        ),
    ),
)
def test_cutile_nvfp4_runner_matches_reference(activation, num_tokens, top_k):
    config, activations, weights, expected = _make_nvfp4_case(
        activation, num_tokens=num_tokens, top_k=top_k
    )
    runner = CuTileNvfp4Runner(config, torch.device("cuda"))
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(activations, weights)

    gemm1_tile_n = 256 if activation.is_gated else 128
    tactic = (32, 1, gemm1_tile_n, 64, 2, 128, 64, 2)
    actual = runner.forward(inputs, tactic=tactic).clone()
    unfused = runner.forward(inputs, tactic=(32, 0, 128, 64, 2, 128, 64, 2)).clone()
    torch.testing.assert_close(actual, unfused, rtol=0, atol=0)
    if activation.type is ActivationType.Relu2:
        fallback = runner.forward(inputs, tactic=-1).clone()
        torch.testing.assert_close(fallback, actual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0.25, atol=1.0)


@cutile_nvfp4_required
@pytest.mark.parametrize(
    ("activation", "fuse_gemm1"),
    (
        (ActivationConfig.relu2, 1),
        (ActivationConfig.swiglu, 0),
    ),
)
def test_cutile_nvfp4_supports_dimensions_divisible_by_64(activation, fuse_gemm1):
    torch.manual_seed(1)
    device = torch.device("cuda")
    num_tokens, num_experts, top_k = 4, 4, 2
    hidden_size = intermediate_size = 192
    w1_rows = intermediate_size * (2 if activation.is_gated else 1)
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    w1 = (
        torch.randn(
            num_experts,
            w1_rows,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 16
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 16
    )
    w1_q, w1_scale, w1_dequant = _quantize_weights(w1)
    w2_q, w2_scale, w2_dequant = _quantize_weights(w2)
    if activation.is_gated:
        w1_global_scale = torch.tensor(
            [[0.75, 1.25]] * num_experts,
            dtype=torch.float32,
            device=device,
        )
        reference_w1 = w1_dequant.clone()
        reference_w1[:, :intermediate_size] *= 0.75
        reference_w1[:, intermediate_size:] *= 1.25
    else:
        w1_global_scale = torch.ones(num_experts, device=device)
        reference_w1 = w1_dequant
    ids = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    routing_weights = torch.rand(num_tokens, top_k, device=device)
    routing_weights /= routing_weights.sum(dim=1, keepdim=True)
    view = CuTileNvfp4Config.prepare_weights(
        w1_q,
        w1_scale,
        w1_global_scale,
        w2_q,
        w2_scale,
        torch.ones(num_experts, device=device),
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_nvfp4", view)
    runner = CuTileNvfp4Runner(
        _nvfp4_config(
            intermediate_size=intermediate_size,
            activation=activation,
        ),
        device,
    )
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(
        MoEActivationPack(hidden_states, None, ids, routing_weights), weights
    )

    actual = runner.forward(
        inputs, tactic=(32, fuse_gemm1, 128, 128, 2, 128, 128, 2)
    ).clone()
    exact_k = runner.forward(
        inputs, tactic=(32, fuse_gemm1, 128, 64, 2, 128, 64, 2)
    ).clone()
    wide_partial = runner.forward(
        inputs, tactic=(32, fuse_gemm1, 256, 256, 1, 256, 256, 1)
    ).clone()
    expected = compute_reference_moe(
        hidden_states,
        ids,
        routing_weights,
        reference_w1,
        w2_dequant,
        activation.type,
    )
    if fuse_gemm1:
        unfused = runner.forward(
            inputs, tactic=(32, 0, 128, 128, 2, 128, 128, 2)
        ).clone()
        torch.testing.assert_close(actual, unfused, rtol=0, atol=0)
    torch.testing.assert_close(actual, exact_k, rtol=1e-2, atol=0.1)
    torch.testing.assert_close(wide_partial, exact_k, rtol=1e-2, atol=0.1)
    torch.testing.assert_close(actual, expected, rtol=0.25, atol=1.0)


@cutile_nvfp4_required
def test_cutile_nvfp4_sorted_io_matches_reference(monkeypatch):
    torch.manual_seed(2)
    from flashinfer.fused_moe.cutile import fp4

    monkeypatch.setattr(fp4, "_SORT_INPUT_MIN_ASSIGNMENTS", 1024)
    device = torch.device("cuda")
    num_tokens, num_experts, top_k = 512, 4, 2
    hidden_size, intermediate_size = 128, 1024
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    w1 = (
        torch.randn(
            num_experts,
            intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 16
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / 16
    )
    w1_q, w1_scale, w1_dequant = _quantize_weights(w1)
    w2_q, w2_scale, w2_dequant = _quantize_weights(w2)
    ids = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    routing_weights = torch.rand(num_tokens, top_k, device=device)
    routing_weights /= routing_weights.sum(dim=1, keepdim=True)
    view = CuTileNvfp4Config.prepare_weights(
        w1_q,
        w1_scale,
        torch.ones(num_experts, device=device),
        w2_q,
        w2_scale,
        torch.ones(num_experts, device=device),
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=ActivationConfig.relu2,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_nvfp4", view)
    runner = CuTileNvfp4Runner(
        _nvfp4_config(
            intermediate_size=intermediate_size,
            activation=ActivationConfig.relu2,
            max_num_tokens=num_tokens,
        ),
        device,
    )
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(
        MoEActivationPack(hidden_states, None, ids, routing_weights), weights
    )
    assert runner._w4a4_gemm_problem(
        inputs, stage=1, block_size=64, fuse_gemm1=True
    ).input_sorted

    fused = runner.forward(inputs, tactic=(64, 1, 128, 128, 2, 128, 128, 2)).clone()
    unfused = runner.forward(inputs, tactic=(64, 0, 128, 128, 2, 128, 128, 2)).clone()
    expected = compute_reference_moe(
        hidden_states,
        ids,
        routing_weights,
        w1_dequant,
        w2_dequant,
        ActivationType.Relu2,
    )
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)
    torch.testing.assert_close(fused, expected, rtol=0.25, atol=1.0)


@cutile_nvfp4_required
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_nvfp4_runner_is_cuda_graph_capturable(activation):
    config, activations, weights, expected = _make_nvfp4_case(activation)
    runner = CuTileNvfp4Runner(config, torch.device("cuda"))
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(activations, weights)
    gemm1_tile_n = 256 if activation.is_gated else 128
    tactic = (32, 1, gemm1_tile_n, 64, 2, 128, 64, 2)
    runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic=tactic)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=0.25, atol=1.0)


@cutile_nvfp4_required
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_nvfp4_runs_through_unified_layer(activation):
    config, activations, weights, expected = _make_nvfp4_case(activation)

    layer = MoELayer(config, torch.device("cuda"))
    actual = layer(activations, weights)

    assert layer.winner_backend == "cutile_nvfp4"
    torch.testing.assert_close(actual, expected, rtol=0.25, atol=1.0)
