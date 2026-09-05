"""Unified cuTile MoE adapter tests."""

from __future__ import annotations

import pytest
import torch

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
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    ReLU,
    ReLU2,
    RoutingConfig,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.fused_moe.runners import _validate_cutile_int32_routing
from flashinfer.tllm_enums import ActivationType

from .utils import compute_reference_activation, compute_reference_moe


# (num_experts, top_k, hidden_size, intermediate_size) from Qwen3.6-35B-A3B
# (SwiGLU) and NVIDIA-Nemotron-3.5-Lightning-30B-A3B (ReLU2), respectively.
_QWEN_MOE_SHAPE = (256, 8, 2048, 512)
_NEMOTRON_MOE_SHAPE = (128, 6, 2688, 1856)

_CUTILE_ACTIVATIONS = (
    SwiGLU(),
    SwiGLUStep(),
    GeGLU(),
    GeGLUTanh(),
    SiTU(),
    ReLU2(),
    Identity(),
    GELU(),
    ReLU(),
    SiLU(),
)


def test_cutile_activation_capabilities_and_scalar_lowering():
    from flashinfer.fused_moe.cutile.activation import _activation_kernel_args

    expected_classes = {type(activation) for activation in _CUTILE_ACTIVATIONS}
    assert set(CuTileBf16Runner.supported_activation_classes) == expected_classes
    assert set(CuTileNvfp4Runner.supported_activation_classes) == expected_classes
    assert _activation_kernel_args(SwiGLU(alpha=1.7, beta=0.25, limit=6.0)) == (
        int(ActivationType.Swiglu),
        1.7,
        0.25,
        6.0,
    )
    assert _activation_kernel_args(SwiGLUStep(limit=5.0)) == (
        int(ActivationType.SwigluStep),
        5.0,
        0.0,
        0.0,
    )
    assert _activation_kernel_args(
        SiTU(gate_scale=2.0, linear_scale=None, clamp_limit=4.0)
    ) == (int(ActivationType.Situ), 2.0, 0.0, 4.0)


@pytest.mark.parametrize(
    "activation", (SwiGLU(alpha=1.7, beta=0.25, limit=6.0), GELU())
)
def test_batched_reference_matches_per_expert_reference(activation):
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 5, 4, 3
    hidden_size, intermediate_size = 8, 6
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    w1 = torch.randn(
        num_experts,
        intermediate_size * (2 if activation.is_gated else 1),
        hidden_size,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 1, 3], [1, 2, 3], [0, 1, 2], [1, 2, 3], [0, 2, 3]],
        dtype=torch.int32,
    )
    topk_weights = torch.rand(num_tokens, top_k)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    expected = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    for expert_id in torch.unique(topk_ids).tolist():
        token_ids, slots = torch.where(topk_ids == expert_id)
        gemm1 = (hidden_states[token_ids].float() @ w1[expert_id].float().T).to(
            torch.bfloat16
        )
        intermediate = compute_reference_activation(
            gemm1, activation, intermediate_size
        )
        expert_output = (intermediate @ w2[expert_id].T).float()
        expected.index_add_(
            0,
            token_ids,
            expert_output * topk_weights[token_ids, slots, None],
        )

    actual = compute_reference_moe(
        hidden_states, topk_ids, topk_weights, w1, w2, activation
    )
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)


def test_cutile_rejects_int64_routing_ranges():
    _validate_cutile_int32_routing("CuTileTestRunner", (1 << 31) - 2, (1 << 31) - 1)
    with pytest.raises(
        NotImplementedError, match=r"fewer than 2\^31 routed and padded assignments"
    ):
        _validate_cutile_int32_routing("CuTileTestRunner", 1 << 31, 1 << 31)


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
        activation=SwiGLU(),
        backend=BackendOptions((CuTileBf16Config(),)),
        finalize=MoEFinalizeConfig(do_finalize=True),
        execution=ExecutionConfig(
            enable_pdl=False,
            tune_max_num_tokens=tune_max_num_tokens,
        ),
    )
    values.update(overrides)
    return MoEConfig(**values)


@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
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
            activation=ReLU2(),
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


def _force_cutile_int64(monkeypatch):
    from flashinfer.fused_moe.cutile import activation, fp4, moe

    for module in (activation, fp4, moe):
        monkeypatch.setattr(module, "needs_int64_indexing", lambda *_args: True)


@cutile_bf16_required
@pytest.mark.parametrize(
    "activation",
    _CUTILE_ACTIVATIONS
    + (
        SwiGLU(alpha=1.7, beta=0.25, limit=6.0),
        SwiGLUStep(limit=5.0),
        SiTU(gate_scale=2.0, linear_scale=None, clamp_limit=4.0),
    ),
    ids=repr,
)
def test_cutile_activation_matches_torch(activation):
    from flashinfer.fused_moe.cutile.activation import launch_activation

    rows, intermediate_size = 3, 257
    input_size = intermediate_size * (2 if activation.is_gated else 1)
    canonical = torch.linspace(
        -8.0,
        8.0,
        rows * input_size,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(rows, input_size)
    kernel_input = canonical
    if activation.is_gated:
        up, gate = canonical.split(intermediate_size, dim=-1)
        kernel_input = torch.cat((gate, up), dim=-1)
    actual = torch.empty(rows, intermediate_size, dtype=torch.bfloat16, device="cuda")

    launch_activation(kernel_input, actual, activation)
    expected = compute_reference_activation(canonical, activation, intermediate_size)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)


@cutile_bf16_required
@pytest.mark.parametrize(
    ("override", "error"),
    (
        (
            {"execution": ExecutionConfig(enable_pdl=True, tune_max_num_tokens=8)},
            "does not support PDL",
        ),
        (
            {"finalize": MoEFinalizeConfig(do_finalize=False)},
            "requires do_finalize=True",
        ),
    ),
)
def test_cutile_check_support_rejects_unsupported_options(override, error):
    runner = CuTileBf16Runner(_config(**override), torch.device("cuda"))
    with pytest.raises(NotImplementedError, match=error):
        runner.check_support()


@cutile_bf16_required
@pytest.mark.parametrize("invalid_input", ("fp16", "non-contiguous"))
def test_cutile_pack_inputs_rejects_invalid_hidden_states(invalid_input):
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size = 2, 64, 64
    config = _config(
        intermediate_size=intermediate_size,
        tune_max_num_tokens=num_tokens,
    )
    w1 = torch.randn(
        4, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    )
    w2 = torch.randn(
        4, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device
    )
    weights = MoEWeightPack()
    weights.prepare_for(
        "cutile_bf16",
        CuTileBf16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=4,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ),
    )
    if invalid_input == "fp16":
        hidden_states = torch.randn(
            num_tokens, hidden_size, dtype=torch.float16, device=device
        )
        error = "requires 2D BF16"
    else:
        hidden_states = torch.randn(
            num_tokens, hidden_size * 2, dtype=torch.bfloat16, device=device
        )[:, ::2]
        error = "requires contiguous hidden states"
    topk_ids = torch.zeros(num_tokens, 2, dtype=torch.int32, device=device)
    topk_weights = torch.full((num_tokens, 2), 0.5, device=device)
    runner = CuTileBf16Runner(config, device)
    runner.check_support()
    runner.build()
    with pytest.raises((TypeError, ValueError), match=error):
        runner.pack_inputs(
            MoEActivationPack(hidden_states, None, topk_ids, topk_weights), weights
        )


@cutile_bf16_required
@pytest.mark.parametrize(
    ("num_experts", "top_k", "capacity_tokens"),
    (
        (16, 8, 4096),
        (256, 8, 256),
        (512, 8, 128),
    ),
)
def test_cutile_workspace_covers_all_permute_shapes_in_bucket(
    num_experts, top_k, capacity_tokens
):
    from flashinfer.fused_moe.cutile import moe

    max_num_assignments = capacity_tokens * top_k
    epow2, max_ncp, max_num_slabs = moe._permute_workspace_shape(
        max_num_assignments, num_experts
    )
    for num_assignments in range(1, max_num_assignments + 1):
        actual_epow2, _, ncp = moe._permute_shape(num_assignments, num_experts)
        chunks_per_slab = max(1, min(ncp, moe._PERMUTE_TILE_CAP // actual_epow2))
        assert actual_epow2 == epow2
        assert ncp <= max_ncp
        assert ncp // chunks_per_slab <= max_num_slabs

    workspace = moe.allocate_workspace(
        num_tokens=capacity_tokens,
        hidden_size=64,
        intermediate_size=64,
        num_experts=num_experts,
        top_k=top_k,
        is_gated=True,
        block_sizes=(32, 64, 128),
        device=torch.device("cuda"),
    )
    assert workspace.hist.numel() == max_ncp * epow2
    assert workspace.base.numel() == max_ncp * epow2
    assert workspace.slab_tot.numel() == max_num_slabs * epow2


@cutile_bf16_required
@pytest.mark.parametrize(
    (
        "activation",
        "num_tokens",
        "num_experts",
        "top_k",
        "hidden_size",
        "intermediate_size",
    ),
    (
        pytest.param(SwiGLU(), 1, 2, 1, 64, 96, id="decode"),
        pytest.param(ReLU2(), 7, 3, 2, 96, 160, id="non-power-of-two-experts"),
        pytest.param(
            SwiGLU(alpha=1.7, beta=0.25, limit=6.0),
            5,
            4,
            2,
            64,
            64,
            id="swiglu-parameters",
        ),
        pytest.param(GeGLU(), 5, 4, 2, 64, 64, id="geglu"),
        pytest.param(
            SiTU(gate_scale=2.0, linear_scale=None, clamp_limit=4.0),
            5,
            4,
            2,
            64,
            64,
            id="situ-optional-parameters",
        ),
        pytest.param(GELU(), 5, 4, 2, 64, 64, id="gelu"),
        pytest.param(SwiGLU(), 33, 8, 4, 256, 128, id="top-k-4"),
        pytest.param(
            SwiGLU(),
            255,
            256,
            8,
            64,
            64,
            id="non-monotonic-workspace-bucket",
        ),
        pytest.param(SwiGLU(), 1, *_QWEN_MOE_SHAPE, id="qwen-tokens-1"),
        pytest.param(SwiGLU(), 128, *_QWEN_MOE_SHAPE, id="qwen-tokens-128"),
        pytest.param(ReLU2(), 1, *_NEMOTRON_MOE_SHAPE, id="nemotron-tokens-1"),
        pytest.param(ReLU2(), 128, *_NEMOTRON_MOE_SHAPE, id="nemotron-tokens-128"),
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
        / hidden_size**0.5
    )
    canonical_w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / intermediate_size**0.5
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
        activation,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-1)


@cutile_bf16_required
@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
def test_cutile_bf16_int64_specialization_matches_int32(activation, monkeypatch):
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size = 3, 64, 64
    config = _config(
        intermediate_size=intermediate_size,
        tune_max_num_tokens=num_tokens,
        activation=activation,
    )
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    w1 = torch.randn(
        4,
        intermediate_size * (2 if activation.is_gated else 1),
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.randn(
        4, hidden_size, intermediate_size, dtype=torch.bfloat16, device=device
    )
    ids = torch.arange(6, dtype=torch.int32, device=device).reshape(3, 2) % 4
    routing_weights = torch.rand(num_tokens, 2, device=device)
    weights = MoEWeightPack()
    weights.prepare_for(
        "cutile_bf16",
        CuTileBf16Config.prepare_weights(
            w1,
            w2,
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
    tactic = runner._fallback_tactic(inputs)
    expected = runner.forward(inputs, tactic=tactic).clone()
    _force_cutile_int64(monkeypatch)
    actual = runner.forward(inputs, tactic=tactic).clone()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cutile_bf16_required
@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
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
    tactic = runner._fallback_tactic(inputs)
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
        activation,
    )
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=2e-1)


@cutile_bf16_required
@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
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
        activation,
    )

    assert layer.winner_backend == "cutile_bf16"
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-1)


def _nvfp4_config(
    *,
    num_experts: int = 4,
    top_k: int = 2,
    intermediate_size: int = 128,
    activation: ActivationConfig | None = None,
    max_num_tokens: int = 128,
) -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation or SwiGLU(),
        backend=BackendOptions((CuTileNvfp4Config(),)),
        finalize=MoEFinalizeConfig(do_finalize=True),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=max_num_tokens),
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
        activation=ReLU2(),
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


@torch.no_grad()
def _quantize_weights(weight: torch.Tensor):
    shape = weight.shape
    columns = shape[-1]
    rows = weight.numel() // columns
    flat_weight = weight.reshape(rows, columns)
    packed = torch.empty(rows, columns // 2, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(
        rows, columns // 16, dtype=torch.float8_e4m3fn, device=weight.device
    )
    dequantized = torch.empty_like(flat_weight)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=weight.device,
    )
    code_points = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device,
    )
    # Bound the FP32 and int64 quantization temporaries for model-sized expert
    # banks; the unchunked reference path peaks at tens of GiB.
    max_chunk_elements = 8 * 1024 * 1024
    chunk_rows = max(1, max_chunk_elements // columns)
    for begin in range(0, rows, chunk_rows):
        end = min(begin + chunk_rows, rows)
        groups = flat_weight[begin:end].float().reshape(-1, columns // 16, 16)
        scale = (groups.abs().amax(dim=-1) / 6.0).to(torch.float8_e4m3fn)
        scales[begin:end].copy_(scale)
        values = groups / scale.float().clamp_min(2.0**-9).unsqueeze(-1)
        codes = torch.bucketize(values.abs(), boundaries, right=False)
        codes |= (values < 0).to(torch.int64) << 3
        codes = codes.reshape(end - begin, columns)
        packed[begin:end].copy_(
            (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
        )
        decoded = code_points[codes & 7]
        decoded = torch.where((codes & 8).bool(), -decoded, decoded)
        dequantized[begin:end].copy_(
            (
                decoded.reshape(end - begin, columns // 16, 16)
                * scale.float().unsqueeze(-1)
            ).reshape(end - begin, columns)
        )
    return (
        packed.reshape(*shape[:-1], columns // 2),
        scales.reshape(*shape[:-1], columns // 16),
        dequantized.reshape(shape),
    )


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
        (SwiGLU(), 128, False),
        (ReLU2(), 128, True),
        (SwiGLUStep(limit=5.0), 128, False),
        (GeGLU(), 128, True),
        (GeGLUTanh(), 128, False),
        (SiTU(), 128, True),
        (SiTU(gate_scale=2.0, linear_scale=None, clamp_limit=4.0), 128, False),
        (Identity(), 128, True),
        (GELU(), 128, False),
        (ReLU(), 128, True),
        (SiLU(), 128, False),
        (SwiGLU(alpha=1.7, beta=0.25, limit=6.0), 768, True),
        (ReLU2(), 768, False),
    ),
    ids=lambda case: repr(case) if isinstance(case, ActivationConfig) else None,
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

    activation_kernels.launch_activation(x, activation_out, activation)
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
        activation,
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
        / hidden_size**0.5
    )
    w1_q, w1_scale, w1_dequant = _quantize_weights(w1)
    del w1
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        / intermediate_size**0.5
    )
    w2_q, w2_scale, w2_dequant = _quantize_weights(w2)
    del w2
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
        activation,
    )
    return config, activations, weights, expected


@cutile_nvfp4_required
@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
def test_cutile_nvfp4_int64_specialization_matches_int32(activation, monkeypatch):
    config, activations, weights, _ = _make_nvfp4_case(activation)
    runner = CuTileNvfp4Runner(config, torch.device("cuda"))
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(activations, weights)
    tactic = runner._w4a4_fallback_tactic(inputs)
    expected = runner.forward(inputs, tactic=tactic).clone()
    _force_cutile_int64(monkeypatch)
    actual = runner.forward(inputs, tactic=tactic).clone()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@cutile_nvfp4_required
@pytest.mark.parametrize(
    (
        "activation",
        "num_tokens",
        "num_experts",
        "top_k",
        "hidden_size",
        "intermediate_size",
    ),
    (
        pytest.param(
            SwiGLU(),
            21,
            4,
            3,
            128,
            128,
            id="bucketed-small-gated-activation",
        ),
        pytest.param(
            SwiGLU(alpha=1.7, beta=0.25, limit=6.0),
            4,
            4,
            2,
            128,
            128,
            id="swiglu-parameters",
        ),
        pytest.param(GeGLU(), 4, 4, 2, 128, 128, id="geglu"),
        pytest.param(
            SiTU(gate_scale=2.0, linear_scale=None, clamp_limit=4.0),
            4,
            4,
            2,
            128,
            128,
            id="situ",
        ),
        pytest.param(GELU(), 4, 4, 2, 128, 128, id="gelu"),
        pytest.param(SwiGLU(), 1, *_QWEN_MOE_SHAPE, id="qwen-tokens-1"),
        pytest.param(SwiGLU(), 128, *_QWEN_MOE_SHAPE, id="qwen-tokens-128"),
        pytest.param(ReLU2(), 1, *_NEMOTRON_MOE_SHAPE, id="nemotron-tokens-1"),
        pytest.param(ReLU2(), 128, *_NEMOTRON_MOE_SHAPE, id="nemotron-tokens-128"),
    ),
)
def test_cutile_nvfp4_runner_matches_reference(
    activation,
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
):
    config, activations, weights, expected = _make_nvfp4_case(
        activation,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
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
def test_cutile_nvfp4_runner_rejects_stale_tactics():
    config, activations, weights, _ = _make_nvfp4_case(ReLU2())
    runner = CuTileNvfp4Runner(config, torch.device("cuda"))
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(activations, weights)

    with pytest.raises(NotImplementedError, match="GEMM1 tactic is unsupported"):
        runner.forward(inputs, tactic=(32, 1, 64, 64, 2, 128, 64, 2))
    with pytest.raises(NotImplementedError, match="GEMM2 tactic is unsupported"):
        runner.forward(inputs, tactic=(32, 1, 128, 64, 2, 128, 64, 3))


@cutile_nvfp4_required
@pytest.mark.parametrize(
    ("activation", "fuse_gemm1"),
    (
        (ReLU2(), 1),
        (SwiGLU(), 0),
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
        activation,
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
        activation=ReLU2(),
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_nvfp4", view)
    runner = CuTileNvfp4Runner(
        _nvfp4_config(
            intermediate_size=intermediate_size,
            activation=ReLU2(),
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
        ReLU2(),
    )
    torch.testing.assert_close(fused, unfused, rtol=0, atol=0)
    torch.testing.assert_close(fused, expected, rtol=0.25, atol=1.0)


@cutile_nvfp4_required
@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
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
@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
def test_cutile_nvfp4_runs_through_unified_layer(activation):
    config, activations, weights, expected = _make_nvfp4_case(activation)

    layer = MoELayer(config, torch.device("cuda"))
    actual = layer(activations, weights)

    assert layer.winner_backend == "cutile_nvfp4"
    torch.testing.assert_close(actual, expected, rtol=0.25, atol=1.0)
