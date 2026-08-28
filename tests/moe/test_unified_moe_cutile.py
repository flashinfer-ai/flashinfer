"""Unified cuTile BF16 MoE adapter tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import AutoTuner
from flashinfer.cutile import is_cuda_tile_available
from flashinfer.fused_moe import (
    ActivationConfig,
    BackendOptions,
    CuTileBf16Config,
    CuTileBf16Runner,
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
from flashinfer.fused_moe.cutile.moe import _combine_tile_h, _permute_shape
from flashinfer.fused_moe.runners import (
    _CUTILE_BF16_DEFAULT_GEMM_CONFIGS,
    _CUTILE_BF16_DIVERSITY_GEMM_CONFIGS,
    _CuTileBf16GemmProblem,
    _cutile_bf16_config_rejection_reason,
    _rank_cutile_bf16_gemm_configs,
)
from flashinfer.tllm_enums import ActivationType, RoutingMethodType


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


@pytest.mark.parametrize("arch", (89, 90, 120, 121))
@pytest.mark.parametrize("num_tokens", (1, 1024))
@pytest.mark.parametrize(
    "activation", (ActivationConfig.swiglu, ActivationConfig.relu2)
)
def test_cutile_bf16_tunes_stages_to_two_configs(
    monkeypatch, arch, num_tokens, activation
):
    class RecordingTuner:
        def __init__(self):
            self.calls = []

        def rank_tactics(
            self, custom_op, runners, tuning_config, inputs, k=1, **kwargs
        ):
            del tuning_config, kwargs
            self.calls.append((custom_op, k))
            return runners[0].get_valid_tactics(inputs, None)[:k]

    tuner = RecordingTuner()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: tuner))
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._built = True
    runner._device_arch = arch
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


@pytest.mark.parametrize(("k_in", "n"), ((15, 64), (64, 31)))
def test_cutile_bf16_gemm_configs_reject_unsupported_small_shapes(k_in, n):
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._device_arch = 90

    with pytest.raises(ValueError, match=rf"no cuTile GEMM tile fits n={n}, k={k_in}"):
        runner._gemm_configs(k_in, n)


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
    ((1, 16, 8), (3, 24, 12), (8, 64, 96)),
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


def _reference_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_type: ActivationType = ActivationType.Swiglu,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = w2.shape[2]
    result = torch.zeros(
        num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device
    )
    for expert_id in range(w1.shape[0]):
        token_ids, slots = torch.where(topk_ids == expert_id)
        if token_ids.numel() == 0:
            continue
        gemm1 = (hidden_states[token_ids].float() @ w1[expert_id].float().T).to(
            torch.bfloat16
        )
        if activation_type is ActivationType.Swiglu:
            up, gate = gemm1.split(intermediate_size, dim=-1)
            intermediate = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        elif activation_type is ActivationType.Relu2:
            intermediate = F.relu(gemm1.float()).square().to(torch.bfloat16)
        else:
            raise ValueError(f"unsupported reference activation {activation_type!r}")
        expert_output = (intermediate @ w2[expert_id].T).float()
        result.index_add_(
            0,
            token_ids,
            expert_output * topk_weights[token_ids, slots, None],
        )
    return result.to(torch.bfloat16)


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
        pytest.param(17, 4, 2, 128, 256, id="baseline"),
        pytest.param(33, 8, 4, 256, 128, id="top-k-4"),
        pytest.param(9, 64, 8, 128, 256, id="many-experts"),
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
    expected = _reference_moe(
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
    expected = _reference_moe(
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

    expected = _reference_moe(
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
    expected = _reference_moe(
        hidden_states,
        ids,
        routing_weights,
        canonical_w1,
        canonical_w2,
        activation.type,
    )

    assert layer.winner_backend == "cutile_bf16"
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-1)
