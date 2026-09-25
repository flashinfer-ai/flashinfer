"""Unified CUTLASS MoE adapter tests covering every quant-specific runner."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import AutoTuner, TuningConfig, autotune
from flashinfer.fused_moe import (
    # Typed activation values
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    # Unified configs, packs, and runners
    BackendOptions,
    CutlassBf16Config,
    CutlassBf16Runner,
    CutlassFp8BlockConfig,
    CutlassFp8BlockRunner,
    CutlassFp8PerTensorConfig,
    CutlassFp8PerTensorRunner,
    CutlassHummingConfig,
    CutlassHummingRunner,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    CutlassMxfp8Mxfp4Runner,
    CutlassMxfp8Runner,
    CutlassNvfp4Config,
    CutlassNvfp4Runner,
    CutlassW4A16Config,
    CutlassW4A16Runner,
    CutlassW4A8Config,
    CutlassW4A8Runner,
    cutlass_fused_moe,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe.runners import MoERunner, _mxfp8_swizzled_act_sf_numel
from flashinfer.fused_moe.prepare import _quantize_mxfp4_linear
from flashinfer.fused_moe.utils import map_to_hybrid_bucket
from tests.moe.utils import fp8_per_tensor_global_scale, fp8_per_tensor_requant_hook
from flashinfer.utils import (
    get_compute_capability,
    is_sm100a_supported,
    is_sm100f_supported,
    is_sm110a_supported,
    is_sm120a_supported,
    is_sm121a_supported,
    is_sm12x_supported,
    is_sm90a_supported,
    round_up,
)


_CUTLASS_ACTIVATIONS = (
    SwiGLU(),
    SwiGLUStep(),
    GeGLU(),
    GeGLUTanh(),
    ReLU2(),
    SiTU(),
    Identity(),
    GELU(),
    ReLU(),
    SiLU(),
)


def _config(**overrides) -> MoEConfig:
    values = dict(
        routing=RoutingConfig(num_experts=4, top_k=2),
        quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=256),
        activation=SwiGLU(),
        backend=BackendOptions((CutlassBf16Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=64),
    )
    values.update(overrides)
    return MoEConfig(**values)


def test_cutlass_bf16_config_architectures_and_registration():
    for arch in (89, 90, 100, 103, 107, 110, 120, 121):
        assert CutlassBf16Config.supported(arch)
        assert CutlassFp8PerTensorConfig.supported(arch)
    assert CutlassW4A16Config.supported(90)
    assert CutlassFp8BlockConfig.supported(90)
    assert CutlassW4A8Config.supported(90)
    assert CutlassHummingConfig.supported(90)
    for arch in (100, 103, 107, 110, 120, 121):
        assert CutlassNvfp4Config.supported(arch)
        assert CutlassMxfp8Mxfp4Config.supported(arch)
    for arch in (100, 103, 107):
        assert CutlassMxfp8Config.supported(arch)
    assert not CutlassBf16Config.supported(80)
    assert not CutlassW4A16Config.supported(100)
    assert not CutlassNvfp4Config.supported(90)
    assert not CutlassFp8BlockConfig.supported(100)
    assert not CutlassMxfp8Config.supported(90)
    assert not CutlassMxfp8Config.supported(110)
    assert not CutlassW4A8Config.supported(100)
    assert not CutlassHummingConfig.supported(100)
    assert not CutlassBf16Config.supported(130)
    assert _BACKEND_RUNNERS[CutlassBf16Config] is CutlassBf16Runner
    assert _BACKEND_RUNNERS[CutlassNvfp4Config] is CutlassNvfp4Runner
    assert _BACKEND_RUNNERS[CutlassW4A16Config] is CutlassW4A16Runner
    assert _BACKEND_RUNNERS[CutlassFp8PerTensorConfig] is CutlassFp8PerTensorRunner
    assert _BACKEND_RUNNERS[CutlassFp8BlockConfig] is CutlassFp8BlockRunner
    assert _BACKEND_RUNNERS[CutlassMxfp8Mxfp4Config] is CutlassMxfp8Mxfp4Runner
    assert _BACKEND_RUNNERS[CutlassMxfp8Config] is CutlassMxfp8Runner
    assert _BACKEND_RUNNERS[CutlassW4A8Config] is CutlassW4A8Runner
    assert _BACKEND_RUNNERS[CutlassHummingConfig] is CutlassHummingRunner


def test_cutlass_quant_runner_activation_capabilities():
    expanded = tuple(type(activation) for activation in _CUTLASS_ACTIVATIONS)
    for runner_cls in (
        CutlassBf16Runner,
        CutlassW4A16Runner,
        CutlassNvfp4Runner,
        CutlassFp8PerTensorRunner,
        CutlassFp8BlockRunner,
        CutlassMxfp8Mxfp4Runner,
        CutlassMxfp8Runner,
        CutlassW4A8Runner,
        CutlassHummingRunner,
    ):
        assert runner_cls.__dict__["supported_activation_classes"] == expanded


def test_all_registered_runners_use_enforced_lifecycle():
    for runner_type in _BACKEND_RUNNERS.values():
        assert issubclass(runner_type, MoERunner)
        assert runner_type.check_support is MoERunner.check_support
        assert runner_type.build is MoERunner.build


@pytest.mark.parametrize("runner_type", tuple(_BACKEND_RUNNERS.values()))
@pytest.mark.parametrize(
    "method,args",
    (
        ("pack_inputs", (None, None)),
        ("get_valid_tactics", ([], None)),
        ("forward", ([],)),
    ),
)
def test_registered_runner_execution_requires_build(runner_type, method, args):
    runner = runner_type.__new__(runner_type)

    with pytest.raises(RuntimeError, match=r"build\(\).*before execution"):
        getattr(runner, method)(*args)


def test_moe_runner_enforces_lifecycle_order():
    events = []

    class Runner(MoERunner):
        supported_quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
        supported_activation_classes = (SwiGLU,)

        def _check_support(self):
            events.append("check_support")
            super()._check_support()

        def _build(self):
            events.append("build")

        def get_valid_tactics(self, inputs, profile):
            self._require_built()
            return [-1]

        def forward(self, inputs, **kwargs):
            self._require_built()
            events.append("execution")

    runner = Runner()
    runner.config = _config()

    with pytest.raises(RuntimeError, match=r"check_support\(\).*build\(\)"):
        runner.build()
    with pytest.raises(RuntimeError, match=r"build\(\).*before execution"):
        runner.forward([])

    runner.check_support()
    runner.build()
    runner.build()
    runner.forward([])

    assert events == ["check_support", "build", "execution"]


def test_failed_support_check_does_not_authorize_build():
    class Runner(MoERunner):
        supported_quant_variants = ((QuantFormat.NVFP4, QuantFormat.NVFP4),)

        def get_valid_tactics(self, inputs, profile):
            return [-1]

        def forward(self, inputs, **kwargs):
            return None

    runner = Runner()
    runner.config = _config()
    runner._support_checked = True

    with pytest.raises(NotImplementedError, match=r"weight=BF16, activation=BF16"):
        runner.check_support()
    with pytest.raises(RuntimeError, match=r"check_support\(\).*build\(\)"):
        runner.build()


def test_prepare_cutlass_bf16_weights_preserves_canonical_layout():
    w1 = torch.randn(2, 64, 64, dtype=torch.bfloat16)[..., ::2]
    w2 = torch.randn(2, 32, 64, dtype=torch.bfloat16)[..., ::2]
    assert not w1.is_contiguous()
    assert not w2.is_contiguous()
    view = CutlassBf16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=32,
    )
    assert set(view) == {"fc1_expert_weights", "fc2_expert_weights"}
    assert view["fc1_expert_weights"].is_contiguous()
    assert view["fc2_expert_weights"].is_contiguous()
    torch.testing.assert_close(view["fc1_expert_weights"], w1)
    torch.testing.assert_close(view["fc2_expert_weights"], w2)


def test_prepare_cutlass_w4a16_weights_rejects_invalid_source_contract():
    w1 = torch.empty(2, 512, 128, dtype=torch.float16)
    w2 = torch.empty(2, 128, 256, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CutlassW4A16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=256,
        )

    w1 = torch.empty(2, 510, 128, dtype=torch.bfloat16)
    w2 = torch.empty(2, 128, 255, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 128"):
        CutlassW4A16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=255,
        )


def test_prepare_cutlass_nvfp4_weights_rejects_invalid_source_contract():
    w1 = torch.empty(2, 512, 128, dtype=torch.float16)
    w2 = torch.empty(2, 128, 256, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CutlassNvfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=256,
        )

    w1 = torch.empty(2, 510, 128, dtype=torch.bfloat16)
    w2 = torch.empty(2, 128, 255, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="intermediate_size divisible by 16"):
        CutlassNvfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=255,
        )

    # H=80 is a multiple of 16 but not of the kernel's 64-element stride
    # alignment; reject at load time, not on the first forward.
    w1 = torch.empty(2, 512, 80, dtype=torch.bfloat16)
    w2 = torch.empty(2, 80, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="hidden_size divisible by 64"):
        CutlassNvfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=80,
            intermediate_size=256,
        )

    w1 = torch.empty(2, 32, 64, dtype=torch.bfloat16)
    w2 = torch.empty(2, 64, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        CutlassNvfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=64,
            intermediate_size=16,
            device=torch.device("cpu"),
        )


def test_prepare_cutlass_fp8_per_tensor_weights_rejects_invalid_source_contract():
    w1 = torch.empty(2, 64, 32, dtype=torch.float16)
    w2 = torch.empty(2, 32, 32, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CutlassFp8PerTensorConfig.prepare_weights(
            w1,
            w2,
            hidden_states_scale_global=1.0,
            intermediate_scale_global=1.0,
            num_local_experts=2,
            hidden_size=32,
            intermediate_size=32,
        )
    w1 = torch.zeros(2, 64, 32, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 32, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="hidden_states_scale_global"):
        CutlassFp8PerTensorConfig.prepare_weights(
            w1,
            w2,
            hidden_states_scale_global=0.0,
            intermediate_scale_global=1.0,
            num_local_experts=2,
            hidden_size=32,
            intermediate_size=32,
        )


def test_prepare_cutlass_fp8_block_weights_rejects_invalid_source_contract():
    w1 = torch.empty(2, 256, 128, dtype=torch.bfloat16)
    w2 = torch.empty(2, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 128"):
        CutlassFp8BlockConfig.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=127,
        )


def test_prepare_cutlass_mxfp8_mxfp4_weights_rejects_invalid_source_contract():
    w1 = torch.empty(2, 64, 32, dtype=torch.float16)
    w2 = torch.empty(2, 32, 32, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CutlassMxfp8Mxfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=32,
            intermediate_size=32,
        )

    w1 = torch.empty(2, 128, 64, dtype=torch.bfloat16)
    w2 = torch.empty(2, 64, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 128"):
        CutlassMxfp8Mxfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=64,
            intermediate_size=64,
        )

    w1 = torch.empty(2, 256, 128, dtype=torch.bfloat16)
    w2 = torch.empty(2, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        CutlassMxfp8Mxfp4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=128,
            device=torch.device("cpu"),
        )


def test_prepare_cutlass_mxfp8_weights_rejects_invalid_source_contract():
    w1 = torch.empty(2, 64, 32, dtype=torch.float16)
    w2 = torch.empty(2, 32, 32, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CutlassMxfp8Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=32,
            intermediate_size=32,
        )

    w1 = torch.empty(2, 128, 64, dtype=torch.bfloat16)
    w2 = torch.empty(2, 64, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 128"):
        CutlassMxfp8Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=64,
            intermediate_size=64,
        )

    w1 = torch.empty(2, 384, 128, dtype=torch.bfloat16)
    w2 = torch.empty(2, 128, 192, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 128"):
        CutlassMxfp8Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=192,
        )

    w1 = torch.empty(2, 256, 128, dtype=torch.bfloat16)
    w2 = torch.empty(2, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires CUDA"):
        CutlassMxfp8Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=128,
            device=torch.device("cpu"),
        )


def test_prepare_cutlass_w4a8_and_humming_weights_reject_invalid_source_contract():
    w1 = torch.empty(2, 256, 128, dtype=torch.float16)
    w2 = torch.empty(2, 128, 128, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CutlassW4A8Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=128,
        )
    with pytest.raises(TypeError, match="expects BF16 weights"):
        CutlassHummingConfig.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size=128,
        )


def test_cutlass_mxfp4_linear_quantizer_code_points():
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            0.0,
        ],
        dtype=torch.bfloat16,
    ).repeat(2, 2)
    packed, scales = _quantize_mxfp4_linear(values)

    expected_bytes = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0xA9, 0xCB, 0xED, 0x0F] * 2,
        dtype=torch.uint8,
    ).repeat(2, 1)
    torch.testing.assert_close(packed, expected_bytes)
    torch.testing.assert_close(scales, torch.full((2, 1), 127, dtype=torch.uint8))


def _dequantize_mxfp4_linear(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
        packed.shape[0], -1
    )
    scale = torch.exp2(scales.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    return lut[nibbles.long()] * scale


def test_cutlass_mxfp4_linear_quantizer_rounds_scale_up():
    values = torch.zeros(2, 32, dtype=torch.bfloat16)
    values[0, 0] = 5.0
    values[1, 0] = 7.0

    packed, scales = _quantize_mxfp4_linear(values)
    dequantized = _dequantize_mxfp4_linear(packed, scales)

    torch.testing.assert_close(
        scales[:, 0], torch.tensor([127, 128], dtype=torch.uint8)
    )
    torch.testing.assert_close(dequantized[:, 0], torch.tensor([4.0, 6.0]))
    actual_scale = torch.exp2(scales[:, 0].to(torch.float32) - 127.0)
    assert torch.all(values[:, 0].abs().float() <= 6.0 * actual_scale)
    assert not torch.any(scales == 255)


def test_cutlass_mxfp4_linear_quantizer_zero_block_uses_minimum_scale():
    packed, scales = _quantize_mxfp4_linear(torch.zeros(2, 64, dtype=torch.bfloat16))

    assert torch.count_nonzero(packed) == 0
    assert torch.count_nonzero(scales) == 0
    assert torch.count_nonzero(_dequantize_mxfp4_linear(packed, scales)) == 0


def test_cutlass_mxfp4_linear_quantizer_clamps_finite_extremes():
    values = torch.zeros(2, 32, dtype=torch.bfloat16)
    values[0, 0] = torch.finfo(torch.bfloat16).tiny
    values[1, 0] = torch.finfo(torch.bfloat16).max

    packed, scales = _quantize_mxfp4_linear(values)

    assert int(scales[0, 0]) == 0
    assert int(scales[1, 0]) <= 254
    assert not torch.any(scales == 255)
    assert int(packed[0, 0] & 0x0F) != 0
    assert int(packed[1, 0] & 0x0F) != 0


@pytest.mark.parametrize(
    "activation,rows",
    (
        (SwiGLU(), 512),
        (SwiGLUStep(), 512),
        (GeGLUTanh(), 512),
        (SiTU(), 512),
        (ReLU2(), 256),
        (Identity(), 256),
        (GELU(), 256),
        (ReLU(), 256),
        (SiLU(), 256),
    ),
)
def test_cutlass_bf16_preparation_uses_activation_gating(activation, rows):
    w1 = torch.empty(4, rows, 128, dtype=torch.bfloat16)
    w2 = torch.empty(4, 128, 256, dtype=torch.bfloat16)
    view = CutlassBf16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=256,
        activation=activation,
        device="cpu",
    )
    assert view["fc1_expert_weights"].shape == (4, rows, 128)
    assert view["fc2_expert_weights"].shape == (4, 128, 256)


@pytest.mark.parametrize(
    "config_cls",
    (
        CutlassBf16Config,
        CutlassW4A16Config,
        CutlassNvfp4Config,
        CutlassFp8PerTensorConfig,
        CutlassFp8BlockConfig,
        CutlassMxfp8Mxfp4Config,
        CutlassMxfp8Config,
        CutlassW4A8Config,
        CutlassHummingConfig,
    ),
)
@pytest.mark.parametrize("activation", (SwiGLU(), Identity()))
def test_all_cutlass_preparers_reject_opposite_activation_geometry(
    config_cls, activation
):
    """Every quant path validates I versus 2I before device-specific work."""
    experts = 2
    hidden = intermediate = 128
    wrong_rows = intermediate if activation.is_gated else 2 * intermediate
    w1 = torch.empty(experts, wrong_rows, hidden, dtype=torch.bfloat16)
    w2 = torch.empty(experts, hidden, intermediate, dtype=torch.bfloat16)
    extra = (
        dict(hidden_states_scale_global=1.0, intermediate_scale_global=1.0)
        if config_cls is CutlassFp8PerTensorConfig
        else {}
    )

    with pytest.raises(ValueError, match="weight shapes"):
        config_cls.prepare_weights(
            w1,
            w2,
            num_local_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
            activation=activation,
            device="cpu",
            **extra,
        )


def test_cutlass_integer_scalars_materialize_float32():
    from flashinfer.fused_moe.runners import _cutlass_activation_params

    params = _cutlass_activation_params(
        SwiGLU(alpha=2, beta=1, limit=7), 4, torch.device("cpu")
    )
    assert all(t is not None and t.dtype is torch.float32 for t in params.values())


def test_cutlass_situ_materializes_only_non_default_scalars():
    """SituAdaptor's compile-time defaults are the typed defaults.

    So the default needs no tensor, while any other value must be materialized.
    This is the opposite of the TRTLLM path, whose null default is 1.0/1.0
    rather than the canonical scales, and where the tensors are therefore
    required even at the default.
    """
    from flashinfer.fused_moe.runners import (
        _cutlass_activation_params,
        _cutlass_activation_required_keys,
    )

    default = _cutlass_activation_params(SiTU(), 4, torch.device("cpu"))
    assert default["situ_beta"] is None
    assert default["situ_linear_beta"] is None
    assert _cutlass_activation_required_keys(SiTU()) == frozenset()

    tuned = SiTU(gate_scale=2.0, linear_scale=10.0)
    params = _cutlass_activation_params(tuned, 4, torch.device("cpu"))
    torch.testing.assert_close(params["situ_beta"], torch.full((4,), 2.0))
    torch.testing.assert_close(params["situ_linear_beta"], torch.full((4,), 10.0))
    # The key set drives cache *validity* -- a cached mapping missing one of
    # these is treated as uninitialized. Cache *identity* comes from
    # repr(config.activation), which already separates a tuned SiTU from a
    # default one.
    assert _cutlass_activation_required_keys(tuned) == frozenset(
        ("situ_beta", "situ_linear_beta")
    )


@pytest.mark.parametrize(
    ("activation", "match"),
    (
        (SiTU(linear_scale=None), "unclamped"),
        (SiTU(clamp_limit=4.0), "clamp channel"),
    ),
)
def test_cutlass_rejects_situ_shapes_its_abi_cannot_express(activation, match):
    """The CUTLASS SiTU ABI carries situ_beta and situ_linear_beta only."""
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=activation)
    runner.device = torch.device("cpu")
    runner._device_arch = 100
    with pytest.raises(NotImplementedError, match=match):
        runner.check_support()


def test_cutlass_situ_unclamped_linear_fails_legibly():
    """Calling the helper directly bypasses check_support's rejection.

    Without the guard torch.full(None) raises an opaque argument-combination
    TypeError naming neither the activation nor the backend.
    """
    from flashinfer.fused_moe.runners import _cutlass_activation_params

    with pytest.raises(NotImplementedError, match="unclamped linear-branch"):
        _cutlass_activation_params(SiTU(linear_scale=None), 4, torch.device("cpu"))


def test_cutlass_situ_per_expert_overrides_are_read():
    """A supplied situ_* tensor must reach the kernel params.

    SiTU carries CUTLASS's native keys rather than the gemm1_* spelling, so an
    override map keyed only on gemm1_* would drop situ_beta silently while
    rejecting names the caller never used.
    """
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=SiTU(gate_scale=2.0, linear_scale=10.0))
    runner.device = torch.device("cpu")

    resolved = runner._resolve_activation_params(
        {
            "situ_beta": torch.full((4,), 9.0),
            "situ_linear_beta": torch.full((4,), 99.0),
        }
    )
    torch.testing.assert_close(resolved["situ_beta"], torch.full((4,), 9.0))
    torch.testing.assert_close(resolved["situ_linear_beta"], torch.full((4,), 99.0))

    # Without an override the configured scalars stand.
    resolved = runner._resolve_activation_params({})
    torch.testing.assert_close(resolved["situ_beta"], torch.full((4,), 2.0))
    torch.testing.assert_close(resolved["situ_linear_beta"], torch.full((4,), 10.0))


@pytest.mark.parametrize(
    "key", ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"), ids=lambda k: k
)
def test_cutlass_situ_rejects_gemm1_spelled_overrides(key):
    """The gemm1_* names belong to the TRTLLM path and are not SiTU's."""
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=SiTU())
    runner.device = torch.device("cpu")
    with pytest.raises(ValueError, match="situ_beta / situ_linear_beta"):
        runner._resolve_activation_params({key: torch.full((4,), 1.0)})


@pytest.mark.parametrize(
    "activation", (SwiGLU(alpha=1.7), SwiGLUStep(), GeGLUTanh(), ReLU2())
)
@pytest.mark.parametrize("key", ("situ_beta", "situ_linear_beta"), ids=lambda k: k)
def test_cutlass_non_situ_rejects_situ_spelled_overrides(activation, key):
    """The mirror of the SiTU case: a situ_* key is foreign to every other one.

    Reading only the alias map that matches the configured activation would let
    the other spelling through untouched, which looks like a working override
    while the kernel never sees it.
    """
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=activation)
    runner.device = torch.device("cpu")
    with pytest.raises(ValueError, match="gemm1_alpha / gemm1_beta"):
        runner._resolve_activation_params({key: torch.full((4,), 9.0)})


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda t: t.to(torch.float16), "float32"),
        (lambda t: t[:1], "shape"),
    ),
    ids=("dtype", "shape"),
)
def test_cutlass_situ_override_boundary_is_validated(mutation, match):
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=SiTU())
    runner.device = torch.device("cpu")
    with pytest.raises((ValueError, TypeError), match=match):
        runner._resolve_activation_params(
            {"situ_beta": mutation(torch.full((4,), 9.0))}
        )


def test_cutlass_activation_params_derived_when_build_did_not_cache_them():
    # _build() normally caches the config-derived scalars. A runner that never
    # built (or whose _build() was overridden) must still resolve the typed
    # activation rather than raising, or silently dropping non-default scalars.
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=SwiGLU(alpha=2.0, beta=1.0, limit=7.0))
    runner.device = torch.device("cpu")
    assert not hasattr(runner, "_config_activation_params")

    resolved = runner._resolve_activation_params({})
    for name, value in (
        ("swiglu_alpha", 2.0),
        ("swiglu_beta", 1.0),
        ("swiglu_limit", 7.0),
    ):
        assert resolved[name].dtype is torch.float32
        torch.testing.assert_close(resolved[name], torch.full((4,), value))

    # Per-expert overrides still win over the derived scalars.
    alpha = torch.arange(4, dtype=torch.float32)
    assert (
        runner._resolve_activation_params({"gemm1_alpha": alpha})["swiglu_alpha"]
        is alpha
    )

    # SwiGLUStep carries only a limit; the other slots stay unset.
    step = CutlassBf16Runner.__new__(CutlassBf16Runner)
    step.config = _config(activation=SwiGLUStep(limit=5.0))
    step.device = torch.device("cpu")
    step_resolved = step._resolve_activation_params({})
    torch.testing.assert_close(step_resolved["swiglu_limit"], torch.full((4,), 5.0))
    assert step_resolved["swiglu_alpha"] is None
    assert step_resolved["swiglu_beta"] is None


def test_cutlass_derived_activation_params_are_cached():
    # Repeated packing must reuse the same tensors: reallocating them would
    # invalidate the raw pointers captured by an existing CUDA graph.
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=SwiGLU(alpha=2.0))
    runner.device = torch.device("cpu")

    first = runner._resolve_activation_params({})
    assert runner._config_activation_params is not None
    second = runner._resolve_activation_params({})
    for name in ("swiglu_alpha", "swiglu_beta", "swiglu_limit"):
        assert first[name] is second[name]


def test_cutlass_incomplete_activation_param_cache_is_rederived():
    # An empty or partial mapping is uninitialized state, not a request for
    # kernel defaults; resolving it must re-derive rather than drop the scalars.
    all_none = {"swiglu_alpha": None, "swiglu_beta": None, "swiglu_limit": None}
    for stale in ({}, {"swiglu_alpha": torch.full((4,), 9.0)}, all_none):
        runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
        runner.config = _config(activation=SwiGLU(alpha=2.0, beta=1.0, limit=7.0))
        runner.device = torch.device("cpu")
        runner._config_activation_params = stale

        resolved = runner._resolve_activation_params({})
        torch.testing.assert_close(resolved["swiglu_alpha"], torch.full((4,), 2.0))
        torch.testing.assert_close(resolved["swiglu_beta"], torch.full((4,), 1.0))
        torch.testing.assert_close(resolved["swiglu_limit"], torch.full((4,), 7.0))


def test_cutlass_per_expert_activation_overrides():
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config(activation=SwiGLU(alpha=2.0))
    runner.device = torch.device("cpu")
    runner._config_activation_params = {
        "swiglu_alpha": torch.full((4,), 2.0),
        "swiglu_beta": torch.zeros(4),
        "swiglu_limit": torch.full((4,), 7.0),
    }
    alpha = torch.arange(4, dtype=torch.float32)
    resolved = runner._resolve_activation_params({"gemm1_alpha": alpha})
    assert resolved["swiglu_alpha"] is alpha
    torch.testing.assert_close(resolved["swiglu_beta"], torch.zeros(4))

    with pytest.raises(TypeError, match=r"torch\.float32"):
        runner._resolve_activation_params(
            {"gemm1_alpha": torch.ones(4, dtype=torch.int64)}
        )
    with pytest.raises(ValueError, match="shape"):
        runner._resolve_activation_params({"gemm1_alpha": torch.ones(3)})

    runner.config = _config(activation=SwiGLUStep())
    runner._config_activation_params = {
        "swiglu_alpha": None,
        "swiglu_beta": None,
        "swiglu_limit": torch.full((4,), 7.0),
    }
    step_limit = torch.arange(4, dtype=torch.float32)
    resolved = runner._resolve_activation_params({"gemm1_clamp_limit": step_limit})
    assert resolved["swiglu_limit"] is step_limit
    for invalid in ("gemm1_alpha", "gemm1_beta"):
        with pytest.raises(ValueError, match="does not consume"):
            runner._resolve_activation_params({invalid: torch.ones(4)})


@pytest.mark.parametrize(
    "config,match",
    (
        (
            _config(
                quant=QuantConfig(
                    weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4
                )
            ),
            "weight=NVFP4, activation=NVFP4",
        ),
        (
            _config(finalize=MoEFinalizeConfig(do_finalize=False)),
            "do_finalize=True",
        ),
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
    ),
)
def test_cutlass_runner_rejects_out_of_scope_configs(config, match):
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = config
    with pytest.raises(NotImplementedError, match=match):
        runner.check_support()


@pytest.mark.parametrize(
    "config,match",
    (
        (
            _config(
                quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16)
            ),
            "weight=BF16, activation=BF16",
        ),
        (
            _config(
                quant=QuantConfig(
                    weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4
                ),
                finalize=MoEFinalizeConfig(do_finalize=False),
            ),
            "do_finalize=True",
        ),
        (
            _config(
                quant=QuantConfig(
                    weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4
                ),
                experts=ExpertConfig(
                    intermediate_size=256,
                    local_expert_offset=2,
                    local_num_experts=2,
                ),
            ),
            "expert parallelism",
        ),
    ),
)
def test_cutlass_nvfp4_runner_rejects_out_of_scope_configs(config, match):
    runner = CutlassNvfp4Runner.__new__(CutlassNvfp4Runner)
    runner.config = config
    with pytest.raises(NotImplementedError, match=match):
        runner.check_support()


@pytest.mark.parametrize(
    "runner_cls,quant,match",
    (
        (
            CutlassFp8PerTensorRunner,
            QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
            "weight=BF16, activation=BF16",
        ),
        (
            CutlassFp8BlockRunner,
            QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
            "weight=BF16, activation=BF16",
        ),
        (
            CutlassMxfp8Mxfp4Runner,
            QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
            "weight=BF16, activation=BF16",
        ),
        (
            CutlassMxfp8Runner,
            QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
            "weight=BF16, activation=BF16",
        ),
        (
            CutlassW4A8Runner,
            QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
            "weight=BF16, activation=BF16",
        ),
        (
            CutlassHummingRunner,
            QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
            "weight=BF16, activation=BF16",
        ),
        (
            CutlassFp8PerTensorRunner,
            QuantConfig(
                weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
            ),
            "do_finalize=True",
        ),
    ),
)
def test_cutlass_quant_runners_reject_out_of_scope_configs(runner_cls, quant, match):
    if match == "do_finalize=True":
        config = _config(
            quant=quant,
            finalize=MoEFinalizeConfig(do_finalize=False),
        )
    else:
        config = _config(quant=quant)
    runner = runner_cls.__new__(runner_cls)
    runner.config = config
    with pytest.raises(NotImplementedError, match=match):
        runner.check_support()


def test_cutlass_nvfp4_rejects_swizzled_scale_factors():
    # NVFP4 has no swizzled input_sf path; an explicit True must not be read
    # as the linear layout it would otherwise silently fall into.
    runner = CutlassNvfp4Runner.__new__(CutlassNvfp4Runner)
    runner.config = _config(
        quant=QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.NVFP4,
            swizzled_scale_factors=True,
        )
    )
    runner._device_arch = 100
    with pytest.raises(NotImplementedError, match="linear activation"):
        runner.check_support()
    runner.config = _config(
        quant=QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.NVFP4,
            swizzled_scale_factors=False,
        )
    )
    runner.check_support()


@pytest.mark.parametrize(
    "runner_cls, quant",
    (
        (CutlassMxfp8Runner, (QuantFormat.MXFP8, QuantFormat.MXFP8)),
        (CutlassMxfp8Mxfp4Runner, (QuantFormat.MXFP4, QuantFormat.MXFP8)),
    ),
)
@pytest.mark.parametrize("swizzled", (None, False, True))
def test_cutlass_mxfp8_scale_layout_follows_swizzled_scale_factors(
    runner_cls, quant, swizzled
):
    # None/False: canonical TRTLLM linear [M, H // 32] pack. True: flat
    # CUTLASS swizzled 1-D input_sf. Both pass check_support.
    runner = runner_cls.__new__(runner_cls)
    runner.config = _config(
        quant=QuantConfig(
            weight=quant[0], activation=quant[1], swizzled_scale_factors=swizzled
        )
    )
    runner._device_arch = 100
    runner.check_support()
    assert runner._swizzled_act_sf is (swizzled is True)
    assert runner._linear_act_sf is (swizzled is not True)


def test_cutlass_mxfp8_prepare_activations_reads_quant_config():
    """The pack layout follows the same QuantConfig the layer declares."""
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires MXFP8×MXFP8"):
        CutlassMxfp8Config.prepare_activations(
            x, quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
        )
    with pytest.raises(ValueError, match="requires MXFP4×MXFP8"):
        CutlassMxfp8Mxfp4Config.prepare_activations(
            x, quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
        )


def test_cutlass_fp8_block_rejects_cuda_below_12_8(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least",
        lambda _version: False,
    )
    runner = CutlassFp8BlockRunner.__new__(CutlassFp8BlockRunner)
    runner.config = _config(
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
        )
    )
    runner._device_arch = 90
    with pytest.raises(NotImplementedError, match="requires CUDA 12.8 or newer"):
        runner.check_support()


def test_cutlass_mxfp8_swizzled_layer_rejects_linear_activation_scales():
    runner = CutlassMxfp8Runner.__new__(CutlassMxfp8Runner)
    runner.config = _config(
        quant=QuantConfig(
            weight=QuantFormat.MXFP8,
            activation=QuantFormat.MXFP8,
            swizzled_scale_factors=True,
        )
    )
    hidden = torch.empty(16, 128, dtype=torch.float8_e4m3fn)
    linear_sf = torch.empty(16, 4, dtype=torch.uint8)
    act = MoEActivationPack(
        hidden,
        linear_sf,
        torch.zeros(16, 2, dtype=torch.int32),
        torch.ones(16, 2, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="swizzled"):
        runner._validate_activation_scale(act)
    # M % 128 == 0 and (H // 32) % 4 == 0: the canonical 2-D pack has exactly
    # the swizzled numel, so only the rank tells them apart.
    act.hidden_states_q = torch.empty(128, 128, dtype=torch.float8_e4m3fn)
    act.topk_ids = torch.zeros(128, 2, dtype=torch.int32)
    act.topk_weights = torch.ones(128, 2, dtype=torch.float32)
    act.hidden_states_scale = torch.empty(128, 4, dtype=torch.uint8)
    assert act.hidden_states_scale.numel() == _mxfp8_swizzled_act_sf_numel(128, 128)
    with pytest.raises(ValueError, match="1-D"):
        runner._validate_activation_scale(act)
    act.hidden_states_scale = act.hidden_states_scale.reshape(-1)
    runner._validate_activation_scale(act)


def test_cutlass_nvfp4_pack_rejects_hidden_size_not_multiple_of_64():
    # The kernel reads the linear input_sf with a padded (64-aligned) row
    # stride, so a compact [M, H // 16] pack is only correct for H % 64 == 0.
    runner = CutlassNvfp4Runner.__new__(CutlassNvfp4Runner)
    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        routing=RoutingConfig(num_experts=2, top_k=2),
        experts=ExpertConfig(intermediate_size=64),
    )
    runner.device = torch.device("cpu")
    view = {key: torch.empty(1) for key in runner._required_weight_keys}
    with pytest.raises(ValueError, match="divisible by 64"):
        runner._pack_weight_inputs(view, hidden_size=80)


def test_cutlass_rejects_per_token_scale():
    """The flat CUTLASS ABI has no per-token activation scale; fail loud."""
    runner = CutlassNvfp4Runner.__new__(CutlassNvfp4Runner)
    runner.config = _config(
        quant=QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.NVFP4,
            per_token_scale=True,
        )
    )
    runner.device = torch.device("cpu")
    runner._device_arch = 100
    with pytest.raises(NotImplementedError, match="per_token_scale=True"):
        runner.check_support()

    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
    )
    act = MoEActivationPack(
        torch.empty(16, 64, dtype=torch.uint8),
        torch.empty(16, 8, dtype=torch.float8_e4m3fn),
        torch.zeros(16, 2, dtype=torch.int32),
        torch.ones(16, 2, dtype=torch.float32),
        per_token_scale=torch.ones(16, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="per_token_scale"):
        runner._validate_activation_scale(act)


def test_cutlass_act_sf_unit_byte_follows_scale_format():
    # Autotune fills synthesized block scales with a unit value; E8M0 1.0 is
    # 127, E4M3 1.0 is 0x38. Derived from the format, not per-runner.
    for runner_cls in (CutlassMxfp8Runner, CutlassMxfp8Mxfp4Runner):
        assert runner_cls.__new__(runner_cls)._act_sf_unit_byte == 127
    assert CutlassNvfp4Runner.__new__(CutlassNvfp4Runner)._act_sf_unit_byte == 0x38


def test_cutlass_mxfp8_linear_layer_rejects_swizzled_activation_scales():
    runner = CutlassMxfp8Runner.__new__(CutlassMxfp8Runner)
    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
    )
    hidden = torch.empty(16, 128, dtype=torch.float8_e4m3fn)
    swizzled_sf = torch.empty(_mxfp8_swizzled_act_sf_numel(16, 128), dtype=torch.uint8)
    act = MoEActivationPack(
        hidden,
        swizzled_sf,
        torch.zeros(16, 2, dtype=torch.int32),
        torch.ones(16, 2, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="linear hidden_states_scale"):
        runner._validate_activation_scale(act)
    # The canonical [M, H // 32] layout is accepted with either storage dtype.
    for dtype in (torch.uint8, torch.float8_e4m3fn):
        act.hidden_states_scale = torch.empty(16, 4, dtype=dtype)
        runner._validate_activation_scale(act)


def test_cutlass_fp8_per_tensor_rejects_activation_scale():
    # The static per-tensor scale lives in the weight view (TRTLLM pack); a
    # per-call hidden_states_scale would silently be ignored.
    runner = CutlassFp8PerTensorRunner.__new__(CutlassFp8PerTensorRunner)
    hidden = torch.empty(1, 128, dtype=torch.float8_e4m3fn)
    act = MoEActivationPack(
        hidden,
        torch.ones((), dtype=torch.float32),
        torch.zeros(1, 2, dtype=torch.int32),
        torch.ones(1, 2, dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="do not use hidden_states_scale"):
        runner._validate_activation_scale(act)
    act.hidden_states_scale = None
    runner._validate_activation_scale(act)


def _fp8_per_tensor_cpu_view():
    return {
        "fc1_expert_weights": torch.empty(4, 512, 128, dtype=torch.float8_e4m3fn),
        "fc2_expert_weights": torch.empty(4, 128, 256, dtype=torch.float8_e4m3fn),
        "fc1_dequant_scale": torch.full((4,), 0.125),
        "fc2_act_quant_scale": torch.tensor(8.0),
        "fc2_dequant_scale": torch.full((4,), 0.03125),
        "fc1_act_dequant_scale": torch.tensor(0.25),
    }


def test_cutlass_fp8_per_tensor_pack_passes_folded_scales_through():
    """The view carries the flat ABI already folded; the runner adds no arithmetic."""
    runner = CutlassFp8PerTensorRunner.__new__(CutlassFp8PerTensorRunner)
    runner.config = _config(
        quant=QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        )
    )
    runner.device = torch.device("cpu")
    runner._inner = object()
    runner._built = True
    runner._ensure_workspace = lambda *_args, **_kwargs: None
    act = MoEActivationPack(
        torch.empty(1, 128, dtype=torch.float8_e4m3fn),
        None,
        torch.zeros(1, 2, dtype=torch.int32),
        torch.full((1, 2), 0.5, dtype=torch.float32),
    )
    view = _fp8_per_tensor_cpu_view()
    weights = MoEWeightPack()
    weights.prepare_for(runner.backend_key, view)
    inputs = runner.pack_inputs(act, weights)
    assert len(inputs) == runner._expected_num_inputs == 10
    assert all(
        packed is view[key]
        for packed, key in zip(
            runner._quant_scales(inputs),
            runner._required_weight_keys[2:],
            strict=True,
        )
    )
    # Static scales are not token-dynamic.
    spec = runner.tuning_config.dynamic_tensor_specs[0]
    assert spec.input_idx == (0, 1, 2, 3)

    view["fc2_act_quant_scale"] = torch.tensor(8.0, dtype=torch.float64)
    with pytest.raises(TypeError, match="fc2_act_quant_scale"):
        runner.pack_inputs(act, weights)
    view["fc2_act_quant_scale"] = torch.full((4,), 8.0)
    with pytest.raises(ValueError, match="fc2_act_quant_scale"):
        runner.pack_inputs(act, weights)


def test_cutlass_mxfp8_mxfp4_pack_rejects_unaligned_hidden_size():
    runner = CutlassMxfp8Mxfp4Runner.__new__(CutlassMxfp8Mxfp4Runner)
    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        routing=RoutingConfig(num_experts=2, top_k=2),
        experts=ExpertConfig(intermediate_size=64),
    )
    runner.device = torch.device("cpu")
    view = {key: torch.empty(1) for key in runner._required_weight_keys}
    with pytest.raises(ValueError, match="divisible by 128"):
        runner._pack_weight_inputs(view, hidden_size=64)


def test_cutlass_mxfp8_pack_rejects_unaligned_hidden_size():
    runner = CutlassMxfp8Runner.__new__(CutlassMxfp8Runner)
    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
        routing=RoutingConfig(num_experts=2, top_k=2),
        experts=ExpertConfig(intermediate_size=64),
    )
    runner.device = torch.device("cpu")
    view = {key: torch.empty(1) for key in runner._required_weight_keys}
    with pytest.raises(ValueError, match="divisible by 128"):
        runner._pack_weight_inputs(view, hidden_size=64)

    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
        routing=RoutingConfig(num_experts=2, top_k=2),
        experts=ExpertConfig(intermediate_size=192),
    )
    with pytest.raises(ValueError, match="divisible by 128"):
        runner._pack_weight_inputs(view, hidden_size=128)


def test_cutlass_mxfp8_pack_rejects_malformed_weight_scales():
    runner = CutlassMxfp8Runner.__new__(CutlassMxfp8Runner)
    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
        routing=RoutingConfig(num_experts=2, top_k=2),
        experts=ExpertConfig(intermediate_size=256),
    )
    runner.device = torch.device("cpu")
    view = {
        "fc1_expert_weights": torch.empty(2, 512, 128, dtype=torch.float8_e4m3fn),
        "fc2_expert_weights": torch.empty(2, 128, 256, dtype=torch.float8_e4m3fn),
        "fc1_expert_scales": torch.empty(2, 4, dtype=torch.int32),
        "fc2_expert_scales": torch.empty(2, 4, dtype=torch.int32),
        "fc1_input_scale": torch.ones(2, dtype=torch.float32),
        "fc2_input_scale": torch.ones(2, dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="fc1_expert_scales"):
        runner._pack_weight_inputs(view, hidden_size=128)


def test_cutlass_w4a8_pack_rejects_malformed_weight_scales():
    runner = CutlassW4A8Runner.__new__(CutlassW4A8Runner)
    runner.config = _config(
        quant=QuantConfig(weight=QuantFormat.INT4, activation=QuantFormat.FP8PerTensor),
        routing=RoutingConfig(num_experts=2, top_k=2),
        experts=ExpertConfig(intermediate_size=256),
    )
    runner.device = torch.device("cpu")
    view = {
        "fc1_expert_weights": torch.empty(2, 512, 64, dtype=torch.uint8),
        "fc2_expert_weights": torch.empty(2, 128, 128, dtype=torch.uint8),
        "fc1_expert_scales": torch.empty(2, 4, dtype=torch.bfloat16),
        "fc2_expert_scales": torch.empty(2, 4, dtype=torch.bfloat16),
        "fc1_act_scale": torch.ones(128, dtype=torch.bfloat16),
        "fc2_act_scale": torch.ones(256, dtype=torch.bfloat16),
        "fc1_zero": torch.empty(0, dtype=torch.bfloat16),
        "fc2_zero": torch.empty(0, dtype=torch.bfloat16),
        "fc1_alpha": torch.ones(2, dtype=torch.float32),
        "fc2_alpha": torch.ones(2, dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="fc1_expert_scales"):
        runner._pack_weight_inputs(view, hidden_size=128)


def test_cutlass_humming_pack_rejects_malformed_weight_scales():
    runner = CutlassHummingRunner.__new__(CutlassHummingRunner)
    runner.config = _config(
        quant=QuantConfig(
            weight=QuantFormat.MXFP4, activation=QuantFormat.FP8PerTensor
        ),
        routing=RoutingConfig(num_experts=2, top_k=2),
        experts=ExpertConfig(intermediate_size=256),
    )
    runner.device = torch.device("cpu")
    view = {
        "fc1_expert_weights": torch.empty(2, 512, 64, dtype=torch.uint8),
        "fc2_expert_weights": torch.empty(2, 128, 128, dtype=torch.uint8),
        "fc1_expert_scales": torch.empty(2, 4, dtype=torch.uint8),
        "fc2_expert_scales": torch.empty(2, 4, dtype=torch.uint8),
        "fc1_residual_scale": torch.ones(2, dtype=torch.float32),
        "fc2_residual_scale": torch.ones(2, dtype=torch.float32),
        "fc2_act_global": torch.ones((), dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="fc1_expert_scales"):
        runner._pack_weight_inputs(view, hidden_size=128)


def test_moe_layer_checks_support_before_build_and_execution(monkeypatch):
    from flashinfer.fused_moe import layer as layer_module

    events = []

    class RecordingRunner:
        supported_quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
        supported_output_formats = (QuantFormat.BF16,)
        supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
        backend_key = "recording"

        @classmethod
        def supports_quant(cls, quant):
            return (
                quant.pair in cls.supported_quant_variants
                and quant.output in cls.supported_output_formats
            )

        def __init__(self, config, device):
            events.append("init")

        def check_support(self):
            events.append("check_support")

        def build(self):
            events.append("build")

        def pack_inputs(self, act_pack, weight_pack):
            events.append("pack_inputs")
            return []

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, tactic=-1):
            events.append("forward")
            return torch.empty(0)

    monkeypatch.setattr(layer_module, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setitem(
        layer_module._BACKEND_RUNNERS, CutlassBf16Config, RecordingRunner
    )
    monkeypatch.setattr(
        MoELayer,
        "_select_winner",
        lambda self, act_pack, weight_pack, runners: (runners[0], -1),
    )

    layer = MoELayer(_config(), device=torch.device("cuda"))
    act = MoEActivationPack(
        torch.empty(1, 1, dtype=torch.bfloat16),
        None,
        torch.zeros(1, 2, dtype=torch.int32),
        torch.full((1, 2), 0.5),
    )
    layer(act, MoEWeightPack())

    assert len(layer.runners) == 1
    assert events == [
        "init",
        "check_support",
        "build",
        "pack_inputs",
        "forward",
    ]


@pytest.mark.parametrize(
    "config",
    (
        _config(
            quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
        ),
        _config(finalize=MoEFinalizeConfig(do_finalize=False)),
        _config(
            experts=ExpertConfig(
                intermediate_size=256,
                local_expert_offset=2,
                local_num_experts=2,
            )
        ),
        _config(
            experts=ExpertConfig(
                intermediate_size=256,
                local_expert_offset=0,
                local_num_experts=2,
            )
        ),
    ),
)
def test_moe_layer_rejects_cutlass_config_before_build(monkeypatch, config):
    import flashinfer.utils as utils_module
    from flashinfer.fused_moe import layer as layer_module

    def must_not_build(self):
        raise AssertionError("incompatible runner was built")

    monkeypatch.setattr(layer_module, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setattr(utils_module, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setattr(utils_module, "device_support_pdl", lambda device: False)
    monkeypatch.setattr(CutlassBf16Runner, "build", must_not_build)

    with pytest.raises(RuntimeError, match="none of the configured backends"):
        MoELayer(config, device=torch.device("cuda:0"))


def test_cutlass_constructor_does_not_load_module(monkeypatch):
    import flashinfer.utils as utils_module
    from flashinfer.fused_moe import core

    monkeypatch.setattr(utils_module, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setattr(utils_module, "device_support_pdl", lambda device: False)

    def must_not_load(*args, **kwargs):
        raise AssertionError("CUTLASS module was loaded during construction")

    monkeypatch.setattr(core, "get_cutlass_fused_moe_module", must_not_load)
    runner = CutlassBf16Runner(_config(), torch.device("cuda:0"))

    assert runner._inner is None


@pytest.mark.parametrize(
    "execute",
    (
        lambda runner: runner.pack_inputs(None, None),
        lambda runner: runner.get_valid_tactics([], None),
        lambda runner: runner.forward([]),
    ),
    ids=("pack_inputs", "get_valid_tactics", "forward"),
)
def test_cutlass_direct_execution_requires_explicit_build(monkeypatch, execute):
    from flashinfer.fused_moe import core

    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner._inner = None
    backend_calls = []

    monkeypatch.setattr(
        core,
        "get_cutlass_fused_moe_module",
        lambda *args, **kwargs: backend_calls.append("module"),
    )
    monkeypatch.setattr(
        core,
        "cutlass_fused_moe_workspace_size",
        lambda *args, **kwargs: backend_calls.append("workspace"),
    )

    with pytest.raises(RuntimeError, match=r"build\(\).*before execution"):
        execute(runner)

    assert backend_calls == []


def test_cutlass_autotuner_preparation_initializes_both_gemms():
    class RecordingInner:
        def __init__(self):
            self.calls = []

        def forward(self, inputs, **kwargs):
            self.calls.append((inputs, kwargs))

    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner._inner = RecordingInner()
    runner._built = True
    runner._workspace = torch.empty(1, dtype=torch.uint8)
    inputs = [torch.empty(1) for _ in range(6)]

    # AutoTuner performs one fallback-tactic preparation call before profiling
    # the valid tactics returned by get_valid_tactics.
    output = runner.forward(inputs, tactic=-1, do_preparation=True)

    assert output is inputs[0]
    assert [call[1]["gemm_idx"] for call in runner._inner.calls] == [1, 2]
    assert [call[1]["tactic"] for call in runner._inner.calls] == [-1, -1]
    assert all(call[1]["do_preparation"] for call in runner._inner.calls)


def test_cutlass_tunes_gemm_stages_independently(monkeypatch):
    class RecordingTuner:
        def __init__(self):
            self.calls = []

        def rank_tactics(
            self, custom_op, runners, tuning_config, inputs, k=1, **kwargs
        ):
            self.calls.append((custom_op, kwargs["gemm_idx"], k))
            if kwargs["gemm_idx"] == 1:
                return [3, 5][:k]
            return [9, 7][:k]

    class Inner:
        gemm_idx_for_tuning = None

    tuner = RecordingTuner()
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: tuner))
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner._inner = Inner()
    runner._built = True
    runner._device_arch = 100
    runner._num_top_tactics_per_stage = 2
    inputs = [torch.empty(1) for _ in range(6)]

    tactics = runner.get_valid_tactics(inputs, None)

    assert tactics == [(3, 9), (3, 7), (5, 9), (5, 7)]
    assert tuner.calls == [
        ("moe_cutlass_bf16_sm100_gemm1", 1, 2),
        ("moe_cutlass_bf16_sm100_gemm2", 2, 2),
    ]
    assert runner._inner.gemm_idx_for_tuning is None


def test_cutlass_one_top_tactic_preserves_single_compound_pair(monkeypatch):
    class RecordingTuner:
        def rank_tactics(
            self, custom_op, runners, tuning_config, inputs, k=1, **kwargs
        ):
            return [3] if kwargs["gemm_idx"] == 1 else [9]

    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: RecordingTuner()))
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner._inner = type("Inner", (), {"gemm_idx_for_tuning": None})()
    runner._built = True
    runner._device_arch = 100
    runner._num_top_tactics_per_stage = 1
    inputs = [torch.empty(1) for _ in range(6)]

    assert runner.get_valid_tactics(inputs, None) == [(3, 9)]


def test_cutlass_outer_cache_key_includes_enable_pdl():
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config()
    runner._device_arch = 90
    runner._enable_pdl = False
    without_pdl = runner.get_cache_key_extras([])

    runner._enable_pdl = True
    with_pdl = runner.get_cache_key_extras([])

    assert without_pdl[-2:] == (90, False)
    assert with_pdl[-2:] == (90, True)
    assert without_pdl[:-2] == with_pdl[:-2]


def test_cutlass_direct_runner_rejects_tokens_above_tuning_ceiling():
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config()
    runner.device = torch.device("cpu")
    runner._inner = object()
    runner._built = True
    num_tokens, hidden_size, top_k = 65, 128, 2
    act = MoEActivationPack(
        torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16),
        None,
        torch.zeros(num_tokens, top_k, dtype=torch.int32),
        torch.full((num_tokens, top_k), 1.0 / top_k, dtype=torch.float32),
    )

    with pytest.raises(
        ValueError, match="num_tokens=65 exceeds tune_max_num_tokens=64"
    ):
        runner.pack_inputs(act, MoEWeightPack())


def test_cutlass_direct_pack_succeeds_after_explicit_build():
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config()
    runner.device = torch.device("cpu")
    runner._inner = None
    events = []

    def check_support():
        events.append("check_support")

    def build():
        events.append("build")
        runner._inner = object()

    def ensure_workspace(num_tokens, hidden_size):
        events.append("workspace")

    runner._check_support = check_support
    runner._build = build
    runner._ensure_workspace = ensure_workspace
    runner._pack_weight_inputs = lambda view, hidden_size: [
        torch.empty(1),
        torch.empty(1),
    ]

    act = MoEActivationPack(
        torch.empty(1, 128, dtype=torch.bfloat16),
        None,
        torch.zeros(1, 2, dtype=torch.int32),
        torch.full((1, 2), 0.5, dtype=torch.float32),
    )
    weights = MoEWeightPack()
    weights.prepare_for(
        runner.backend_key,
        {
            "fc1_expert_weights": torch.empty(1),
            "fc2_expert_weights": torch.empty(1),
        },
    )

    runner.check_support()
    runner.build()
    runner.pack_inputs(act, weights)

    assert events == ["check_support", "build", "workspace"]


def test_cutlass_tuning_pre_hook_activates_synthesized_bucket_workspace():
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config()
    activated = []
    runner._ensure_workspace = lambda tokens, hidden: activated.append((tokens, hidden))
    inputs = [
        torch.empty(64, 128, dtype=torch.bfloat16),
        torch.empty(64, 128, dtype=torch.bfloat16),
        torch.empty(64, 2, dtype=torch.int32),
        torch.empty(64, 2),
        torch.empty(1),
        torch.empty(1),
    ]

    runner._prepare_tuning_inputs(inputs)

    assert activated == [(64, 128)]


def test_cutlass_reuses_geometric_workspace_capacities(monkeypatch):
    from flashinfer.fused_moe import core

    requested_tokens = []

    def fake_workspace_size(num_tokens, *args, **kwargs):
        requested_tokens.append(num_tokens)
        return 8

    monkeypatch.setattr(core, "cutlass_fused_moe_workspace_size", fake_workspace_size)
    runner = CutlassBf16Runner.__new__(CutlassBf16Runner)
    runner.config = _config()
    runner.device = torch.device("cpu")
    runner._use_fused_finalize = True
    runner._workspace_cache = {}
    runner._workspace = None
    runner._workspace_num_tokens = 0
    runner._workspace_hidden_size = None

    runner._ensure_workspace(17, 128)
    workspace_32 = runner._workspace
    runner._ensure_workspace(33, 128)
    workspace_64 = runner._workspace
    runner._ensure_workspace(17, 128)

    assert runner._workspace is workspace_32
    assert runner._workspace_num_tokens == 32
    assert requested_tokens == [32, 64]
    assert runner._workspace_cache[(32, 128)] is workspace_32
    assert runner._workspace_cache[(64, 128)] is workspace_64
    with pytest.raises(ValueError, match="hidden_size changed"):
        runner._ensure_workspace(32, 256)


def _is_cutlass_bf16_runtime_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    arch = major * 10 + minor
    if not CutlassBf16Config.supported(arch):
        return False
    if arch == 90:
        return is_sm90a_supported(device)
    if arch in (100, 103):
        return is_sm100a_supported(device)
    if arch == 107:
        return is_sm100f_supported(device)
    if arch == 110:
        return is_sm110a_supported(device)
    if arch == 120:
        return is_sm120a_supported(device)
    if arch == 121:
        return is_sm121a_supported(device)
    return arch == 89


cutlass_bf16_required = pytest.mark.skipif(
    not _is_cutlass_bf16_runtime_supported(),
    reason="requires a supported CUTLASS BF16 GPU and CUDA toolkit",
)


cutlass_w4a16_required = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")),
    reason="requires SM90a with CUDA 12.3+",
)


def _make_case(num_tokens: int = 16, activation=None):
    torch.manual_seed(42)
    device = torch.device("cuda", torch.cuda.current_device())
    num_experts, top_k = 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    activation = activation or SwiGLU()
    gemm1_rows = intermediate_size * (2 if activation.is_gated else 1)
    w1 = (
        torch.randn(
            num_experts,
            gemm1_rows,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)

    config = _config(
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        execution=ExecutionConfig(
            enable_pdl=False,
            tune_max_num_tokens=max(64, num_tokens),
        ),
    )
    act = MoEActivationPack(x, None, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for(
        "cutlass_bf16",
        CutlassBf16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        ),
    )
    return config, act, weights, w1, w2


def _make_w4a16_case(num_tokens: int = 16, activation=None):
    torch.manual_seed(43)
    device = torch.device("cuda", torch.cuda.current_device())
    num_experts, top_k = 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    activation = activation or SwiGLU()
    gemm1_rows = intermediate_size * (2 if activation.is_gated else 1)
    w1 = (
        torch.randn(
            num_experts,
            gemm1_rows,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        backend=BackendOptions((CutlassW4A16Config(),)),
        activation=activation,
        execution=ExecutionConfig(
            enable_pdl=False,
            tune_max_num_tokens=max(64, num_tokens),
        ),
    )
    act = MoEActivationPack(x, None, topk_ids, topk_weights)
    weights = MoEWeightPack()
    view = CutlassW4A16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    weights.prepare_for("cutlass_w4a16", view)
    w1_packed, w1_scale = _quantize_mxfp4_linear(
        w1.view(num_experts * gemm1_rows, hidden_size)
    )
    w2_packed, w2_scale = _quantize_mxfp4_linear(
        w2.view(num_experts * hidden_size, intermediate_size)
    )
    w1_quantized = _dequantize_mxfp4_linear(w1_packed, w1_scale).view_as(w1)
    w2_quantized = _dequantize_mxfp4_linear(w2_packed, w2_scale).view_as(w2)
    return config, act, weights, w1_quantized, w2_quantized, view


def _reference(
    act: MoEActivationPack,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation=None,
    *,
    intermediate_hook=None,
):
    """``intermediate_hook`` models a backend's GEMM2-input requantization."""
    activation = activation or SwiGLU()
    x = act.hidden_states_q.float()
    result = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(act.topk_ids.shape[1]):
            expert = int(act.topk_ids[token, slot])
            fc1 = x[token] @ w1[expert].float().T
            if isinstance(activation, ReLU2):
                intermediate = F.relu(fc1) ** 2
            elif isinstance(activation, Identity):
                intermediate = fc1
            elif isinstance(activation, GELU):
                intermediate = F.gelu(fc1, approximate="none")
            elif isinstance(activation, ReLU):
                intermediate = F.relu(fc1)
            elif isinstance(activation, SiLU):
                intermediate = F.silu(fc1)
            else:
                up, gate = fc1.chunk(2)
                if isinstance(activation, SwiGLU):
                    gate = gate.clamp(max=activation.limit)
                    up = up.clamp(min=-activation.limit, max=activation.limit)
                    intermediate = (
                        gate
                        * torch.sigmoid(activation.alpha * gate)
                        * (up + activation.beta)
                    )
                elif isinstance(activation, SwiGLUStep):
                    intermediate = F.silu(gate).clamp(max=activation.limit) * up.clamp(
                        min=-activation.limit, max=activation.limit
                    )
                elif isinstance(activation, GeGLUTanh):
                    intermediate = F.gelu(gate, approximate="tanh") * up
                elif isinstance(activation, GeGLU):
                    intermediate = F.gelu(gate, approximate="none") * up
                elif isinstance(activation, SiTU):
                    # SituAdaptor approximates tanh as 2*sigmoid(2z)-1;
                    # torch.tanh remains within the test tolerance.
                    beta = activation.gate_scale
                    linear_beta = activation.linear_scale
                    intermediate = (
                        beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
                    ) * (linear_beta * torch.tanh(up / linear_beta))
                else:
                    raise AssertionError(
                        f"unsupported CUTLASS activation {activation!r}"
                    )
            if intermediate_hook is not None:
                intermediate = intermediate_hook(intermediate)
            expert_out = intermediate @ w2[expert].float().T
            result[token] += act.topk_weights[token, slot] * expert_out
    return result.to(torch.bfloat16)


def _dequant_linear_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequant packed E2M1 + linear UE8M0 scales without Humming preprocessing."""
    low = packed & 0xF
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).reshape(
        packed.shape[0], packed.shape[1], packed.shape[2] * 2
    )
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
        dtype=torch.float32,
    )
    values = magnitudes[codes.to(torch.long) & 0x7]
    values = torch.where((codes & 0x8) != 0, -values, values)
    scale = torch.exp2(scales.to(torch.int16).to(torch.float32) - 127)
    return values * scale.repeat_interleave(32, dim=-1)


def _assert_numerically_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            torch.zeros_like(expected),
            expected,
            rtol=rtol,
            atol=atol,
        )
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _pin_fallback_winner(layer: MoELayer, act: MoEActivationPack):
    runner = layer.runners[0]
    bucket = map_to_hybrid_bucket(
        act.num_tokens, layer.config.execution.tune_max_num_tokens
    )
    layer._winners[(bucket, RoutingInputMode.PackedPrecomputed)] = (runner, -1)
    return runner


def _independent_activation_params(activation, num_experts, device):
    """Lower a typed activation to CUTLASS scalar tensors, from the value alone.

    Deliberately does not call runners._cutlass_activation_params: that is the
    lowering under test, so reusing it would let a wrong mapping agree with
    itself. CUTLASS compile-time defaults cover SwiGLU() and SiTU(), which is
    why only non-default values materialize tensors.
    """
    full = lambda v: torch.full((num_experts,), v, dtype=torch.float32, device=device)
    if isinstance(activation, SiTU):
        if activation == SiTU():
            return {}
        return {
            "situ_beta": full(activation.gate_scale),
            "situ_linear_beta": full(activation.linear_scale),
        }
    if isinstance(activation, SwiGLU) and activation != SwiGLU():
        return {
            "swiglu_alpha": full(activation.alpha),
            "swiglu_beta": full(activation.beta),
            "swiglu_limit": full(activation.limit),
        }
    if isinstance(activation, SwiGLUStep):
        return {"swiglu_limit": full(activation.limit)}
    return {}


# Per-backend flat contract, written out from the prepared view's named keys
# rather than read off the runner. quant_scales is ordered as the flat binding
# consumes it; a runner that reorders or mis-selects a scale disagrees here.
def _independent_quant_scales(backend_key, view, act):
    v = view
    if backend_key == "cutlass_bf16":
        return []
    if backend_key == "cutlass_w4a16":
        return [
            v["fc1_expert_scales"].view(torch.int32),
            v["fc2_expert_scales"].view(torch.int32),
        ]
    if backend_key == "cutlass_nvfp4":
        return [
            v["fc1_act_global_scale"],
            v["fc1_weight_block_scale"].view(torch.int32),
            v["fc1_dequant_scale"],
            v["fc2_act_global_scale"],
            v["fc2_weight_block_scale"].view(torch.int32),
            v["fc2_dequant_scale"],
        ]
    if backend_key == "cutlass_fp8_per_tensor":
        return [
            v["fc1_dequant_scale"],
            v["fc2_act_quant_scale"],
            v["fc2_dequant_scale"],
            v["fc1_act_dequant_scale"],
        ]
    if backend_key == "cutlass_fp8_block":
        return [v["fc1_block_scale"], v["fc2_block_scale"]]
    if backend_key in ("cutlass_mxfp8_mxfp4", "cutlass_mxfp8"):
        return [
            v["fc1_expert_scales"].view(torch.int32),
            v["fc1_input_scale"],
            v["fc2_expert_scales"].view(torch.int32),
            v["fc2_input_scale"],
        ]
    if backend_key == "cutlass_w4a8":
        return [
            v["fc1_expert_scales"],
            v["fc2_expert_scales"],
            v["fc1_act_scale"],
            v["fc2_act_scale"],
            v["fc1_zero"],
            v["fc2_zero"],
            v["fc1_alpha"],
            v["fc2_alpha"],
        ]
    if backend_key == "cutlass_humming":
        # The gemm2 activation scale sits between the two gemm groups here,
        # not after them -- see the Humming quant_scales list in the
        # cutlass_fused_moe docstring.
        return [
            v["fc1_expert_scales"].view(torch.int32),
            v["fc1_residual_scale"],
            v["fc2_act_global"],
            v["fc2_expert_scales"].view(torch.int32),
            v["fc2_residual_scale"],
        ]
    raise AssertionError(f"no independent flat contract for {backend_key}")


# Expected quant_scales ordering per backend, by view key. Pinned here because
# four CUTLASS backends are SM90-only and skip on SM100, so a wrong order in
# _independent_quant_scales would reach H100 CI unchecked -- which is how the
# W4A8 and Humming orderings were originally wrong. This locks the contract in
# one reviewable place; the orderings themselves are ground-truthed by the
# cutlass_fused_moe docstring and by running the suite on a supported arch.
_EXPECTED_SCALE_ORDER = {
    "cutlass_bf16": (),
    "cutlass_w4a16": ("fc1_expert_scales", "fc2_expert_scales"),
    "cutlass_fp8_block": ("fc1_block_scale", "fc2_block_scale"),
    "cutlass_fp8_per_tensor": (
        "fc1_dequant_scale",
        "fc2_act_quant_scale",
        "fc2_dequant_scale",
        "fc1_act_dequant_scale",
    ),
    "cutlass_nvfp4": (
        "fc1_act_global_scale",
        "fc1_weight_block_scale",
        "fc1_dequant_scale",
        "fc2_act_global_scale",
        "fc2_weight_block_scale",
        "fc2_dequant_scale",
    ),
    "cutlass_mxfp8_mxfp4": (
        "fc1_expert_scales",
        "fc1_input_scale",
        "fc2_expert_scales",
        "fc2_input_scale",
    ),
    "cutlass_mxfp8": (
        "fc1_expert_scales",
        "fc1_input_scale",
        "fc2_expert_scales",
        "fc2_input_scale",
    ),
    "cutlass_w4a8": (
        "fc1_expert_scales",
        "fc2_expert_scales",
        "fc1_act_scale",
        "fc2_act_scale",
        "fc1_zero",
        "fc2_zero",
        "fc1_alpha",
        "fc2_alpha",
    ),
    "cutlass_humming": (
        "fc1_expert_scales",
        "fc1_residual_scale",
        "fc2_act_global",
        "fc2_expert_scales",
        "fc2_residual_scale",
    ),
}


@pytest.mark.parametrize("backend_key", sorted(_EXPECTED_SCALE_ORDER))
def test_scale_orders_match_flat_abi(backend_key):
    """CPU-only, so the SM90 backends are covered on any arch.

    Sentinels are int32 so the ``.view(torch.int32)`` calls stay no-ops.
    """
    expected = _EXPECTED_SCALE_ORDER[backend_key]
    keys = sorted(set(expected))
    view = {key: torch.tensor(i + 1, dtype=torch.int32) for i, key in enumerate(keys)}
    expected_values = [view[key].item() for key in expected]

    independent = _independent_quant_scales(backend_key, view, None)
    assert [tensor.item() for tensor in independent] == expected_values

    runner_cls = next(
        cls for cls in _BACKEND_RUNNERS.values() if cls.backend_key == backend_key
    )
    weight_inputs = [view[key] for key in runner_cls._required_weight_keys[2:]]
    runner_inputs = [
        torch.empty(0, dtype=torch.int32) for _ in range(6)
    ] + weight_inputs
    production = runner_cls.__new__(runner_cls)._quant_scales(runner_inputs)
    assert [tensor.item() for tensor in production] == expected_values


def _run_flat_cutlass_independently(
    config_cls, backend_key, act: MoEActivationPack, weights: MoEWeightPack, config
) -> torch.Tensor:
    """Launch the flat API with arguments built from the view, not the runner.

    Routing every argument through runner.pack_inputs()/_quant_scales() would
    make this comparison circular: _CutlassRunnerBase.forward() calls
    cutlass_fused_moe with those same expressions, so a mis-ordered scale or a
    dropped activation scalar would reach both sides identically and still
    compare bit-exact. Rebuilding from the prepared view's named keys is what
    gives the assertion its teeth.
    """
    view = weights.get_view(backend_key)
    x = act.hidden_states_q
    num_experts = config.routing.num_experts
    output = torch.empty(
        (x.shape[0], x.shape[1]), dtype=torch.bfloat16, device=x.device
    )
    input_sf = None
    swizzled_input_sf = True
    if config_cls in (CutlassMxfp8Mxfp4Config, CutlassMxfp8Config):
        # A 2-D [M, H // 32] scale is the canonical linear pack; the flat
        # swizzled buffer is 1-D.
        input_sf = act.hidden_states_scale
        swizzled_input_sf = input_sf.ndim == 1
    elif config_cls is CutlassNvfp4Config:
        # Canonical TRTLLM pack: packed uint8 [M, H // 2] with a linear
        # [M, H // 16] block scale, so the output is twice as wide as x.
        input_sf = act.hidden_states_scale
        swizzled_input_sf = False
        output = torch.empty(
            (x.shape[0], x.shape[1] * 2), dtype=torch.bfloat16, device=x.device
        )
    # The flat NVFP4/MXFP4 ABI takes packed weights viewed as int64 -- that is
    # how the binding selects the FP4 kernel, and it rejects raw uint8.
    packed_fp4 = config_cls in (CutlassNvfp4Config, CutlassMxfp8Mxfp4Config)
    fc1 = view["fc1_expert_weights"]
    fc2 = view["fc2_expert_weights"]
    if packed_fp4:
        fc1, fc2 = fc1.view(torch.int64), fc2.view(torch.int64)
    cutlass_fused_moe(
        x,
        act.topk_ids,
        act.topk_weights,
        fc1,
        fc2,
        output_dtype=torch.bfloat16,
        quant_scales=_independent_quant_scales(backend_key, view, act),
        input_sf=input_sf,
        output=output,
        tune_max_num_tokens=config.execution.tune_max_num_tokens,
        enable_pdl=config.execution.enable_pdl,
        activation_type=config.activation.type,
        use_deepseek_fp8_block_scale=(config_cls is CutlassFp8BlockConfig),
        use_w4_group_scaling=config_cls
        in (CutlassW4A16Config, CutlassW4A8Config, CutlassHummingConfig),
        use_mxfp8_act_scaling=config_cls
        in (CutlassMxfp8Mxfp4Config, CutlassMxfp8Config),
        use_packed_weights=(config_cls is CutlassW4A8Config),
        use_wfp4afp8_humming=(config_cls is CutlassHummingConfig),
        swizzled_input_sf=swizzled_input_sf,
        # Written out rather than read off the runner: if a runner stops using
        # the fused finalize, this comparison should surface that as a
        # difference instead of silently following it.
        use_fused_finalize=True,
        profile_ids=[-1, -1],
        **_independent_activation_params(config.activation, num_experts, x.device),
    )
    return output


@cutlass_bf16_required
@pytest.mark.parametrize(
    "activation",
    _CUTLASS_ACTIVATIONS
    + (
        SwiGLU(alpha=1.7, beta=1.0, limit=7.0),
        SiTU(gate_scale=2.0, linear_scale=10.0),
    ),
)
def test_cutlass_bf16_moe_layer_matches_independent_reference(activation):
    config, act, weights, w1, w2 = _make_case(activation=activation)
    layer = MoELayer(config)
    runner = _pin_fallback_winner(layer, act)

    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassBf16Config, "cutlass_bf16", act, weights, config
    )
    expected = _reference(act, w1, w2, activation)

    assert layer.winner_backend == "cutlass_bf16"
    assert runner._workspace is not None
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    _assert_numerically_close(actual, expected, rtol=2e-2, atol=2e-2)


@cutlass_w4a16_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_w4a16_moe_layer_matches_quantized_reference(activation):
    config, act, weights, w1, w2, view = _make_w4a16_case(activation=activation)
    assert view["fc1_expert_weights"].dtype is torch.uint8
    assert view["fc1_expert_scales"].ndim == 5
    layer = MoELayer(config)
    runner = _pin_fallback_winner(layer, act)

    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassW4A16Config, "cutlass_w4a16", act, weights, config
    )
    expected = _reference(act, w1, w2, activation)

    assert layer.winner_backend == "cutlass_w4a16"
    assert runner._workspace is not None
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    assert torch.isfinite(actual).all(), "CUTLASS W4A16 produced non-finite output"
    _assert_numerically_close(actual, expected, rtol=5e-2, atol=2e-2)


@cutlass_bf16_required
def test_cutlass_stage_file_cache_key_includes_top_k():
    config, act, weights, _, _ = _make_case()
    runner = MoELayer(config).runners[0]
    inputs = runner.pack_inputs(act, weights)
    stage_inputs = [inputs[1], inputs[4], None, inputs[5], None]
    input_shapes = AutoTuner.get()._get_input_sizes(stage_inputs)
    original_top_k = runner._inner.top_k
    original_extras = runner._inner.get_cache_key_extras(stage_inputs)
    try:
        runner._inner.top_k = original_top_k + 1
        other_extras = runner._inner.get_cache_key_extras(stage_inputs)
    finally:
        runner._inner.top_k = original_top_k
    original_key = AutoTuner._get_cache_key(
        f"moe_cutlass_bf16_sm{runner._device_arch}_gemm1",
        runner._inner,
        input_shapes,
        TuningConfig(),
        original_extras,
    )
    other_key = AutoTuner._get_cache_key(
        f"moe_cutlass_bf16_sm{runner._device_arch}_gemm1",
        runner._inner,
        input_shapes,
        TuningConfig(),
        other_extras,
    )
    assert original_key.nearest_profile == other_key.nearest_profile
    assert original_key.file_key != other_key.file_key


@cutlass_bf16_required
def test_cutlass_autotuned_compound_tactic_numerics_and_cuda_graph():
    config, act, weights, w1, w2 = _make_case(num_tokens=17)
    runner = MoELayer(config).runners[0]
    inputs = runner.pack_inputs(act, weights)
    spec = runner.tuning_config.dynamic_tensor_specs[0]
    assert spec.input_idx == (0, 1, 2, 3)
    assert spec.gen_tuning_buckets == (32,)
    assert runner._workspace_num_tokens == 32

    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            "test_moe_cutlass_bf16_compound",
            [runner],
            runner.tuning_config,
            inputs,
        )
    assert isinstance(tactic, tuple) and len(tactic) == 2
    assert all(stage_tactic >= 0 for stage_tactic in tactic)

    actual = runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()
    expected = _reference(act, w1, w2)
    _assert_numerically_close(actual, expected, rtol=2e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.forward(inputs, tactic=tactic)
    captured_workspace = runner._workspace
    runner._ensure_workspace(64, inputs[1].shape[1])
    assert runner._workspace is not captured_workspace
    assert runner._workspace_cache[(32, inputs[1].shape[1])] is captured_workspace
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_numerically_close(captured, expected, rtol=2e-2, atol=2e-2)


@cutlass_bf16_required
def test_cutlass_autotune_override_uses_geometric_workspace():
    config, act, weights, _, _ = _make_case(num_tokens=17)
    runner = MoELayer(config).runners[0]
    inputs = runner.pack_inputs(act, weights)
    workspace_32 = runner._workspace

    with autotune(True, tuning_buckets=(64,)):
        AutoTuner.get().choose_one(
            "test_moe_cutlass_bf16_override_workspace",
            [runner],
            runner.tuning_config,
            inputs,
        )

    assert runner._workspace_num_tokens == 64
    assert runner._workspace is runner._workspace_cache[(64, 128)]
    assert runner._workspace_cache[(32, 128)] is workspace_32


@cutlass_bf16_required
def test_cutlass_forward_reselects_runtime_workspace_after_smaller_override():
    config, act, weights, w1, w2 = _make_case(num_tokens=50)
    runner = MoELayer(config).runners[0]
    inputs = runner.pack_inputs(act, weights)
    workspace_64 = runner._workspace

    with autotune(True, tuning_buckets=(32,)):
        _, tactic = AutoTuner.get().choose_one(
            "test_moe_cutlass_bf16_smaller_override_workspace",
            [runner],
            runner.tuning_config,
            inputs,
        )

    assert runner._workspace_num_tokens == 32
    assert runner._workspace is runner._workspace_cache[(32, 128)]
    actual = runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()

    assert runner._workspace_num_tokens == 64
    assert runner._workspace is workspace_64
    _assert_numerically_close(actual, _reference(act, w1, w2), rtol=2e-2, atol=2e-2)


@cutlass_w4a16_required
def test_cutlass_w4a16_autotuned_compound_tactic_and_cuda_graph():
    config, act, weights, w1, w2, _ = _make_w4a16_case(num_tokens=17)
    runner = MoELayer(config).runners[0]
    inputs = runner.pack_inputs(act, weights)

    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            "test_moe_cutlass_w4a16_compound",
            [runner],
            runner.tuning_config,
            inputs,
        )
    assert isinstance(tactic, tuple) and len(tactic) == 2
    assert all(stage_tactic >= 0 for stage_tactic in tactic)

    actual = runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()
    expected = _reference(act, w1, w2)
    assert torch.isfinite(actual).all(), "CUTLASS W4A16 produced non-finite output"
    _assert_numerically_close(actual, expected, rtol=5e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.forward(inputs, tactic=tactic)
    captured_workspace = runner._workspace
    runner._ensure_workspace(64, inputs[1].shape[1])
    assert runner._workspace is not captured_workspace
    assert runner._workspace_cache[(32, inputs[1].shape[1])] is captured_workspace
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_numerically_close(captured, expected, rtol=5e-2, atol=2e-2)


def _is_cutlass_nvfp4_runtime_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    arch = major * 10 + minor
    if not CutlassNvfp4Config.supported(arch):
        return False
    if arch in (100, 103):
        return is_sm100a_supported(device)
    if arch == 107:
        return is_sm100f_supported(device)
    if arch == 110:
        return is_sm110a_supported(device)
    if arch in (120, 121):
        return is_sm12x_supported(device)
    return False


cutlass_nvfp4_required = pytest.mark.skipif(
    not _is_cutlass_nvfp4_runtime_supported(),
    reason="requires SM100/SM110/SM12x CUTLASS NVFP4 GPU and CUDA toolkit",
)


def _make_nvfp4_case(
    num_tokens: int = 16, activation=None, intermediate_size: int = 256
):
    torch.manual_seed(44)
    device = torch.device("cuda", torch.cuda.current_device())
    num_experts, top_k = 4, 2
    hidden_size = 128
    activation = activation or SwiGLU()
    gemm1_rows = intermediate_size * (2 if activation.is_gated else 1)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1 = (
        torch.randn(
            num_experts,
            gemm1_rows,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    config = _config(
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CutlassNvfp4Config(),)),
        execution=ExecutionConfig(
            enable_pdl=False,
            tune_max_num_tokens=max(64, num_tokens),
        ),
    )
    # The same pack TRTLLM / CuTe-DSL / Cake consume: packed uint8 [M, H // 2]
    # plus a linear E4M3 [M, H // 16] block scale with unit global scale.
    x_q, x_sf = CutlassNvfp4Config.prepare_activations(x)
    act = MoEActivationPack(x_q, x_sf, topk_ids, topk_weights)
    view = CutlassNvfp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_nvfp4", view)
    return config, act, weights, view


def _dequantize_cutlass_nvfp4_matrix(
    packed: torch.Tensor, scale: torch.Tensor, *, swizzled: bool = True
) -> torch.Tensor:
    from flashinfer.fp4_quantization import e2m1_and_ufp8sf_scale_to_float

    rows, packed_cols = packed.shape
    global_scale = torch.ones(1, dtype=torch.float32)
    return e2m1_and_ufp8sf_scale_to_float(
        packed,
        scale.view(torch.uint8).reshape(-1),
        global_scale,
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=swizzled,
    ).view(rows, packed_cols * 2)


def _nvfp4_quantized_reference(
    act: MoEActivationPack,
    view: dict[str, torch.Tensor],
    activation=None,
):
    # Dequantize the pack itself (linear scale layout) rather than
    # re-quantizing: the reference must see exactly what the kernel sees.
    x = act.hidden_states_q
    x_dq = _dequantize_cutlass_nvfp4_matrix(
        x, act.hidden_states_scale, swizzled=False
    ).to(device=x.device, dtype=torch.bfloat16)
    w1_q = view["fc1_expert_weights"]
    w2_q = view["fc2_expert_weights"]
    w1_sf = view["fc1_weight_block_scale"]
    w2_sf = view["fc2_weight_block_scale"]
    w1 = torch.stack(
        [
            _dequantize_cutlass_nvfp4_matrix(w1_q[i], w1_sf[i]).to(
                device=x.device, dtype=torch.bfloat16
            )
            for i in range(w1_q.shape[0])
        ]
    )
    w2 = torch.stack(
        [
            _dequantize_cutlass_nvfp4_matrix(w2_q[i], w2_sf[i]).to(
                device=x.device, dtype=torch.bfloat16
            )
            for i in range(w2_q.shape[0])
        ]
    )
    ref_act = MoEActivationPack(x_dq, None, act.topk_ids, act.topk_weights)
    return _reference(
        ref_act, w1, w2, activation, intermediate_hook=_nvfp4_requant_hook(x.device)
    )


def _nvfp4_requant_hook(device):
    """Model the kernels' NVFP4 requantization of the GEMM1 output (unit
    global scale, 16-element E4M3 block scales) before GEMM2. Every NVFP4
    backend does this, so a reference without it carries ~2x the error."""
    from flashinfer import fp4_quantize

    one = torch.ones(1, device=device)

    def hook(intermediate: torch.Tensor) -> torch.Tensor:
        q, sf = fp4_quantize(
            intermediate.to(torch.bfloat16).reshape(1, -1),
            global_scale=one,
            sf_vec_size=16,
            is_sf_swizzled_layout=False,
        )
        return (
            _dequantize_cutlass_nvfp4_matrix(q, sf, swizzled=False)
            .to(device=device, dtype=torch.float32)
            .reshape(-1)
        )

    return hook


@cutlass_nvfp4_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_nvfp4_moe_layer_matches_quantized_reference(activation):
    config, act, weights, view = _make_nvfp4_case(activation=activation)
    assert view["fc1_expert_weights"].dtype is torch.uint8
    assert view["fc1_weight_block_scale"].ndim == 3
    assert act.hidden_states_q.dtype is torch.uint8
    assert act.hidden_states_scale.dtype is torch.float8_e4m3fn
    assert tuple(act.hidden_states_scale.shape) == (
        act.hidden_states_q.shape[0],
        act.hidden_states_q.shape[1] * 2 // 16,
    )
    layer = MoELayer(config)
    runner = _pin_fallback_winner(layer, act)

    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassNvfp4Config, "cutlass_nvfp4", act, weights, config
    )
    expected = _nvfp4_quantized_reference(act, view, activation)

    assert layer.winner_backend == "cutlass_nvfp4"
    assert runner._workspace is not None
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    assert torch.isfinite(actual).all(), "CUTLASS NVFP4 produced non-finite output"
    # Identity is the only pure pass-through here, so GEMM1 output reaches the
    # NVFP4 requantization with its full dynamic range instead of being
    # compressed by a nonlinearity first. The dequantized reference tracks that
    # noise less well: one element of 2048 lands ~11% past the shared bound,
    # while ReLU/GELU/SiLU -- non-gated but still compressive -- stay inside it.
    # The exact flat comparison above is what pins the plumbing; this bound only
    # needs to catch a wrong magnitude.
    ref_rtol = ref_atol = 3e-1 if isinstance(activation, Identity) else 2e-1
    _assert_numerically_close(actual, expected, rtol=ref_rtol, atol=ref_atol)


@cutlass_nvfp4_required
@pytest.mark.parametrize("activation", (ReLU2(), SwiGLU()))
@pytest.mark.parametrize("intermediate_size", (96, 192))
def test_cutlass_nvfp4_accepts_intermediate_size_not_multiple_of_128(
    activation, intermediate_size
):
    # The fc1 block scale is padded to the 128-row swizzle tile and fc2's
    # K-dimension scale groups to 64, so these shapes hand the binding scale
    # tensors with more rows than inter_size (2 * inter_size when gated).
    config, act, weights, view = _make_nvfp4_case(
        activation=activation, intermediate_size=intermediate_size
    )
    expected_rows = intermediate_size * (2 if activation.is_gated else 1)
    assert view["fc1_weight_block_scale"].shape[1] == round_up(expected_rows, 128)
    layer = MoELayer(config)
    _pin_fallback_winner(layer, act)

    actual = layer(act, weights)
    expected = _nvfp4_quantized_reference(act, view, activation)

    assert layer.winner_backend == "cutlass_nvfp4"
    assert torch.isfinite(actual).all(), "CUTLASS NVFP4 produced non-finite output"
    _assert_numerically_close(actual, expected, rtol=2e-1, atol=2e-1)


@cutlass_nvfp4_required
def test_cutlass_nvfp4_autotuned_compound_tactic_and_cuda_graph():
    config, act, weights, view = _make_nvfp4_case(num_tokens=17)
    runner = MoELayer(config).runners[0]
    inputs = runner.pack_inputs(act, weights)
    assert inputs[4].dtype is torch.int64
    assert len(inputs) == 13
    assert inputs[-1] is act.hidden_states_scale
    # The linear block scale is token-dynamic and must be resized with the
    # token bucket alongside output/hidden/topk_ids/topk_weights.
    spec = runner.tuning_config.dynamic_tensor_specs[0]
    assert spec.input_idx == (0, 1, 2, 3, 12)
    assert spec.dim_idx == (0, 0, 0, 0, 0)

    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            "test_moe_cutlass_nvfp4_compound",
            [runner],
            runner.tuning_config,
            inputs,
        )
    assert isinstance(tactic, tuple) and len(tactic) == 2
    assert all(stage_tactic >= 0 for stage_tactic in tactic)

    actual = runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()
    expected = _nvfp4_quantized_reference(act, view)
    assert torch.isfinite(actual).all(), "CUTLASS NVFP4 produced non-finite output"
    _assert_numerically_close(actual, expected, rtol=2e-1, atol=2e-1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.forward(inputs, tactic=tactic)
    captured_workspace = runner._workspace
    hidden_size = inputs[1].shape[1] * 2
    runner._ensure_workspace(64, hidden_size)
    assert runner._workspace is not captured_workspace
    assert runner._workspace_cache[(32, hidden_size)] is captured_workspace
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_numerically_close(captured, expected, rtol=2e-1, atol=2e-1)


def _is_cutlass_fp8_runtime_supported() -> bool:
    return _is_cutlass_bf16_runtime_supported()


cutlass_fp8_required = pytest.mark.skipif(
    not _is_cutlass_fp8_runtime_supported(),
    reason="requires a supported CUTLASS FP8 GPU and CUDA toolkit",
)

cutlass_fp8_block_required = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")),
    reason="requires SM90a CUTLASS DeepSeek FP8 block scaling",
)


def _is_cutlass_mxfp8_runtime_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda")
    major, minor = get_compute_capability(device)
    arch = major * 10 + minor
    if not CutlassMxfp8Config.supported(arch):
        return False
    if arch in (100, 103):
        return is_sm100a_supported(device)
    if arch == 107:
        return is_sm100f_supported(device)
    return False


cutlass_mxfp8_mxfp4_required = pytest.mark.skipif(
    not _is_cutlass_nvfp4_runtime_supported(),
    reason="requires SM100/SM110/SM12x CUTLASS MXFP8xMXFP4 GPU and CUDA toolkit",
)

cutlass_mxfp8_required = pytest.mark.skipif(
    not _is_cutlass_mxfp8_runtime_supported(),
    reason="requires SM100/SM103/SM107 CUTLASS MXFP8xMXFP8 GPU and CUDA toolkit",
)

cutlass_w4a8_required = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")),
    reason="requires SM90a CUTLASS W4A8",
)

cutlass_humming_required = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")),
    reason="requires SM90a CUTLASS Humming",
)


def _make_routing(num_tokens, num_experts, top_k, device):
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    return topk_ids, topk_weights


def _make_bf16_experts(
    num_experts, hidden_size, intermediate_size, device, activation=None
):
    activation = activation or SwiGLU()
    gemm1_rows = intermediate_size * (2 if activation.is_gated else 1)
    w1 = (
        torch.randn(
            num_experts,
            gemm1_rows,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    return w1, w2


def _autotune_and_graph(runner, act, weights, expected, *, rtol, atol, cache_name):
    inputs = runner.pack_inputs(act, weights)
    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            cache_name,
            [runner],
            runner.tuning_config,
            inputs,
        )
    assert isinstance(tactic, tuple) and len(tactic) == 2
    assert all(stage_tactic >= 0 for stage_tactic in tactic)
    actual = runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    _assert_numerically_close(actual, expected, rtol=rtol, atol=atol)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.forward(inputs, tactic=tactic)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_numerically_close(captured, expected, rtol=rtol, atol=atol)


def _fp8_per_tensor_static_scales(x: torch.Tensor):
    """Calibration multipliers a framework would carry: amax-derived for the
    activations, a fixed value for the GEMM1 output (the fuzz does the same
    with 64.0; 32.0 here keeps the non-gated activations' larger GEMM1
    outputs representable)."""
    return fp8_per_tensor_global_scale(x), torch.tensor(32.0, device=x.device)


def _assert_fp8_per_tensor_view_folded(view):
    """The four flat-ABI slots are the fold of the calibration metadata."""
    a1, a2 = view["hidden_states_scale_global"], view["intermediate_scale_global"]
    torch.testing.assert_close(view["fc1_dequant_scale"], view["fc1_dequant"] / a1)
    torch.testing.assert_close(view["fc2_act_quant_scale"], a2)
    torch.testing.assert_close(view["fc2_dequant_scale"], view["fc2_dequant"] / a2)
    torch.testing.assert_close(view["fc1_act_dequant_scale"], 1.0 / a1)


@cutlass_fp8_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_fp8_per_tensor_moe_layer_matches_quantized_reference(activation):
    torch.manual_seed(45)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, num_experts, top_k = 16, 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    # Static calibration multipliers, as a framework would supply them.
    hs_scale, inter_scale = _fp8_per_tensor_static_scales(x)
    x_q, x_scale = CutlassFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=hs_scale
    )
    assert x_q.dtype is torch.float8_e4m3fn
    assert x_scale is None
    view = CutlassFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        hidden_states_scale_global=hs_scale,
        intermediate_scale_global=inter_scale,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    x_dq = x_q.float() / view["hidden_states_scale_global"]
    w1_dq = view["fc1_expert_weights"].float() * view["fc1_dequant"][:, None, None]
    w2_dq = view["fc2_expert_weights"].float() * view["fc2_dequant"][:, None, None]
    config = _config(
        quant=QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CutlassFp8PerTensorConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=64),
    )
    act = MoEActivationPack(x_q, x_scale, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_fp8_per_tensor", view)
    layer = MoELayer(config)
    # Keep the public weight-view contract coupled to the actual runner
    # boundary, rather than only checking the numerical launch below.
    _assert_fp8_per_tensor_view_folded(view)
    packed_inputs = layer.runners[0].pack_inputs(act, weights)
    assert packed_inputs[7] is view["fc2_act_quant_scale"]
    _pin_fallback_winner(layer, act)
    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassFp8PerTensorConfig, "cutlass_fp8_per_tensor", act, weights, config
    )
    # Dequantized operands stay fp32: rounding fc1_dequant-scaled weights to
    # bf16 adds error that ReLU2's square amplifies past the tolerance.
    expected = _reference(
        MoEActivationPack(x_dq, None, topk_ids, topk_weights),
        w1_dq,
        w2_dq,
        activation,
        intermediate_hook=fp8_per_tensor_requant_hook(
            view["intermediate_scale_global"]
        ),
    )
    assert layer.winner_backend == "cutlass_fp8_per_tensor"
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    _assert_numerically_close(actual, expected, rtol=1e-1, atol=1e-1)
    _autotune_and_graph(
        layer.runners[0],
        act,
        weights,
        expected,
        rtol=1e-1,
        atol=1e-1,
        cache_name="test_moe_cutlass_fp8_per_tensor",
    )


@cutlass_fp8_block_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_fp8_block_moe_layer_matches_quantized_reference(activation):
    torch.manual_seed(46)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, num_experts, top_k = 16, 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    view = CutlassFp8BlockConfig.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    w1_dq = view["fc1_expert_weights"].float() * view[
        "fc1_block_scale"
    ].repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    w2_dq = view["fc2_expert_weights"].float() * view[
        "fc2_block_scale"
    ].repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)
    config = _config(
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8, activation=QuantFormat.DeepSeekFp8
        ),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CutlassFp8BlockConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=64),
    )
    act = MoEActivationPack(x, None, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_fp8_block", view)
    layer = MoELayer(config)
    _pin_fallback_winner(layer, act)
    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassFp8BlockConfig, "cutlass_fp8_block", act, weights, config
    )
    expected = _reference(
        act, w1_dq.to(torch.bfloat16), w2_dq.to(torch.bfloat16), activation
    )
    assert layer.winner_backend == "cutlass_fp8_block"
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    _assert_numerically_close(actual, expected, rtol=1e-1, atol=1e-1)
    _autotune_and_graph(
        layer.runners[0],
        act,
        weights,
        expected,
        rtol=1e-1,
        atol=1e-1,
        cache_name="test_moe_cutlass_fp8_block",
    )


@cutlass_mxfp8_mxfp4_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_mxfp8_mxfp4_moe_layer_matches_quantized_reference(activation):
    from flashinfer import mxfp4_dequantize, mxfp8_dequantize_host

    torch.manual_seed(47)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, num_experts, top_k = 16, 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    x_q, x_sf = CutlassMxfp8Mxfp4Config.prepare_activations(x)
    assert x_sf.shape == (num_tokens, hidden_size // 32)
    view = CutlassMxfp8Mxfp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    x_dq = mxfp8_dequantize_host(
        x_q.cpu().view(torch.uint8),
        x_sf.cpu().view(torch.uint8).reshape(-1),
        False,
    ).to(device=device, dtype=torch.bfloat16)
    w1_dq = torch.stack(
        [
            mxfp4_dequantize(
                view["fc1_expert_weights"][i].cpu(),
                view["fc1_expert_scales"][i].cpu().view(torch.uint8).reshape(-1),
            )
            for i in range(num_experts)
        ]
    ).to(device=device, dtype=torch.bfloat16)
    w2_dq = torch.stack(
        [
            mxfp4_dequantize(
                view["fc2_expert_weights"][i].cpu(),
                view["fc2_expert_scales"][i].cpu().view(torch.uint8).reshape(-1),
            )
            for i in range(num_experts)
        ]
    ).to(device=device, dtype=torch.bfloat16)
    config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CutlassMxfp8Mxfp4Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=16),
    )
    act = MoEActivationPack(x_q, x_sf, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_mxfp8_mxfp4", view)
    layer = MoELayer(config)
    _pin_fallback_winner(layer, act)
    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassMxfp8Mxfp4Config, "cutlass_mxfp8_mxfp4", act, weights, config
    )
    expected = _reference(
        MoEActivationPack(x_dq, None, topk_ids, topk_weights),
        w1_dq,
        w2_dq,
        activation,
    )
    assert layer.winner_backend == "cutlass_mxfp8_mxfp4"
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    _assert_numerically_close(actual, expected, rtol=1e-1, atol=1e-1)
    _autotune_and_graph(
        layer.runners[0],
        act,
        weights,
        expected,
        rtol=1e-1,
        atol=1e-1,
        cache_name="test_moe_cutlass_mxfp8_mxfp4",
    )


@cutlass_mxfp8_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_mxfp8_moe_layer_matches_quantized_reference(activation):
    from flashinfer import mxfp8_dequantize_host

    torch.manual_seed(48)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, num_experts, top_k = 16, 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    x_q, x_sf = CutlassMxfp8Config.prepare_activations(x)
    assert x_sf.shape == (num_tokens, hidden_size // 32)
    view = CutlassMxfp8Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    x_dq = mxfp8_dequantize_host(
        x_q.cpu().view(torch.uint8),
        x_sf.cpu().view(torch.uint8).reshape(-1),
        False,
    ).to(device=device, dtype=torch.bfloat16)
    w1_dq = torch.stack(
        [
            mxfp8_dequantize_host(
                view["fc1_expert_weights"][i].cpu().view(torch.uint8),
                view["fc1_expert_scales"][i].cpu().view(torch.uint8).reshape(-1),
                True,
            )
            for i in range(num_experts)
        ]
    ).to(device=device, dtype=torch.bfloat16)
    w2_dq = torch.stack(
        [
            mxfp8_dequantize_host(
                view["fc2_expert_weights"][i].cpu().view(torch.uint8),
                view["fc2_expert_scales"][i].cpu().view(torch.uint8).reshape(-1),
                True,
            )
            for i in range(num_experts)
        ]
    ).to(device=device, dtype=torch.bfloat16)
    config = _config(
        quant=QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CutlassMxfp8Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=16),
    )
    act = MoEActivationPack(x_q, x_sf, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_mxfp8", view)
    layer = MoELayer(config)
    _pin_fallback_winner(layer, act)
    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassMxfp8Config, "cutlass_mxfp8", act, weights, config
    )
    expected = _reference(
        MoEActivationPack(x_dq, None, topk_ids, topk_weights),
        w1_dq,
        w2_dq,
        activation,
    )
    assert layer.winner_backend == "cutlass_mxfp8"
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    _assert_numerically_close(actual, expected, rtol=2e-1, atol=2e-1)
    _autotune_and_graph(
        layer.runners[0],
        act,
        weights,
        expected,
        rtol=2e-1,
        atol=2e-1,
        cache_name="test_moe_cutlass_mxfp8",
    )


@cutlass_mxfp8_required
def test_cutlass_mxfp8_autotune_regenerates_swizzled_input_sf_across_bucket():
    torch.manual_seed(51)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, num_experts, top_k = 257, 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(num_experts, hidden_size, intermediate_size, device)
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    # The swizzled flat layout stays available behind the QuantConfig knob; the
    # same quant drives both the pack layout and the layer declaration.
    quant = QuantConfig(
        weight=QuantFormat.MXFP8,
        activation=QuantFormat.MXFP8,
        swizzled_scale_factors=True,
    )
    x_q, x_sf = CutlassMxfp8Config.prepare_activations(x, quant=quant)
    view = CutlassMxfp8Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    config = _config(
        quant=quant,
        experts=ExpertConfig(intermediate_size=intermediate_size),
        backend=BackendOptions((CutlassMxfp8Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=8192),
    )
    act = MoEActivationPack(x_q, x_sf, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_mxfp8", view)
    runner = MoELayer(config).runners[0]
    inputs = runner.pack_inputs(act, weights)
    assert x_sf.numel() == _mxfp8_swizzled_act_sf_numel(num_tokens, hidden_size)
    assert inputs[-1].numel() == x_sf.numel()
    bucket = map_to_hybrid_bucket(num_tokens, 8192)
    assert bucket == 512
    assert runner.tuning_config.constraint_specs
    infer_numel = runner.tuning_config.constraint_specs[0].infer_shape
    assert infer_numel([None, (bucket, hidden_size)]) == _mxfp8_swizzled_act_sf_numel(
        bucket, hidden_size
    )

    synthesized = list(inputs)
    synthesized[0] = torch.empty(
        bucket, hidden_size, dtype=torch.bfloat16, device=device
    )
    synthesized[1] = torch.empty(
        bucket, hidden_size, dtype=torch.float8_e4m3fn, device=device
    )
    synthesized[2] = torch.empty(bucket, top_k, dtype=torch.int32, device=device)
    synthesized[3] = torch.empty(bucket, top_k, dtype=torch.float32, device=device)
    tuned = runner._prepare_tuning_inputs(synthesized)
    assert tuned[-1].numel() == _mxfp8_swizzled_act_sf_numel(bucket, hidden_size)

    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            "test_moe_cutlass_mxfp8_bucket_boundary",
            [runner],
            runner.tuning_config,
            inputs,
        )
    assert isinstance(tactic, tuple) and len(tactic) == 2
    assert all(stage_tactic >= 0 for stage_tactic in tactic)
    actual = runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()


def _dequant_int4(packed, scale, group_size=128):
    even = packed.to(torch.int16) & 0xF
    odd = packed.to(torch.int16) >> 4
    even = torch.where(even >= 8, even - 16, even)
    odd = torch.where(odd >= 8, odd - 16, odd)
    unpacked = torch.stack((even, odd), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    expanded = scale.float().repeat_interleave(group_size, dim=-1)
    return unpacked.float() * expanded


@cutlass_w4a8_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_w4a8_moe_layer_matches_quantized_reference(activation):
    torch.manual_seed(49)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, num_experts, top_k = 16, 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    from flashinfer.fused_moe.prepare import _quantize_int4_grouped

    packed_w1, scale_w1 = _quantize_int4_grouped(w1)
    packed_w2, scale_w2 = _quantize_int4_grouped(w2)
    view = CutlassW4A8Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    config = _config(
        quant=QuantConfig(weight=QuantFormat.INT4, activation=QuantFormat.FP8PerTensor),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CutlassW4A8Config(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=64),
    )
    act = MoEActivationPack(x, None, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_w4a8", view)
    layer = MoELayer(config)
    _pin_fallback_winner(layer, act)
    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassW4A8Config, "cutlass_w4a8", act, weights, config
    )
    expected = _reference(
        act,
        _dequant_int4(packed_w1, scale_w1).to(torch.bfloat16),
        _dequant_int4(packed_w2, scale_w2).to(torch.bfloat16),
        activation,
    )
    assert layer.winner_backend == "cutlass_w4a8"
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    _assert_numerically_close(actual, expected, rtol=1e-1, atol=1e-1)
    _autotune_and_graph(
        layer.runners[0],
        act,
        weights,
        expected,
        rtol=1e-1,
        atol=1e-1,
        cache_name="test_moe_cutlass_w4a8",
    )


@cutlass_humming_required
@pytest.mark.parametrize("activation", _CUTLASS_ACTIVATIONS)
def test_cutlass_humming_moe_layer_matches_quantized_reference(activation):
    torch.manual_seed(50)
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens, num_experts, top_k = 16, 4, 2
    hidden_size, intermediate_size = 128, 256
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    view = CutlassHummingConfig.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
    )
    w1_lin, w1_sf = _quantize_mxfp4_linear(
        w1.view(num_experts * w1.shape[1], hidden_size)
    )
    w2_lin, w2_sf = _quantize_mxfp4_linear(
        w2.view(num_experts * hidden_size, intermediate_size)
    )
    config = _config(
        quant=QuantConfig(
            weight=QuantFormat.MXFP4, activation=QuantFormat.FP8PerTensor
        ),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions((CutlassHummingConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=64),
    )
    act = MoEActivationPack(x, None, topk_ids, topk_weights)
    weights = MoEWeightPack()
    weights.prepare_for("cutlass_humming", view)
    layer = MoELayer(config)
    _pin_fallback_winner(layer, act)
    actual = layer(act, weights)
    flat = _run_flat_cutlass_independently(
        CutlassHummingConfig, "cutlass_humming", act, weights, config
    )
    expected = _reference(
        act,
        _dequant_linear_mxfp4(
            w1_lin.view(num_experts, w1.shape[1], hidden_size // 2),
            w1_sf.view(num_experts, w1.shape[1], hidden_size // 32),
        ).to(torch.bfloat16),
        _dequant_linear_mxfp4(
            w2_lin.view(num_experts, hidden_size, intermediate_size // 2),
            w2_sf.view(num_experts, hidden_size, intermediate_size // 32),
        ).to(torch.bfloat16),
        activation,
    )
    assert layer.winner_backend == "cutlass_humming"
    torch.testing.assert_close(actual, flat, rtol=0, atol=0)
    _assert_numerically_close(actual, expected, rtol=2e-1, atol=2e-1)
    _autotune_and_graph(
        layer.runners[0],
        act,
        weights,
        expected,
        rtol=2e-1,
        atol=2e-1,
        cache_name="test_moe_cutlass_humming",
    )


@cutlass_fp8_required
@pytest.mark.parametrize("activation", (SwiGLU(), ReLU2()))
def test_cutlass_fp8_per_tensor_shares_canonical_pack_with_trtllm(activation):
    """The per-tensor FP8 pack from ``TrtllmFp8PerTensorConfig
    .prepare_activations`` (no pack scale) feeds TRTLLM and CUTLASS runners
    built from one ``MoELayer`` whose views carry the same static scales.

    ReLU2 covers the TRT-LLM non-gated ``output1_scales_scalar`` branch."""
    from flashinfer.fused_moe import TrtllmFp8PerTensorConfig

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = get_compute_capability(device)
    if not TrtllmFp8PerTensorConfig.supported(major * 10 + minor):
        pytest.skip("requires an arch with TRTLLM and CUTLASS per-tensor FP8")

    torch.manual_seed(54)
    num_tokens, num_experts, top_k = 32, 8, 2
    hidden_size, intermediate_size = 256, 512
    quant = QuantConfig(
        weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
    )
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    hs_scale, inter_scale = _fp8_per_tensor_static_scales(x)
    scales = dict(
        hidden_states_scale_global=hs_scale, intermediate_scale_global=inter_scale
    )
    x_q, x_scale = TrtllmFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=hs_scale
    )
    assert x_scale is None
    act = MoEActivationPack(x_q, None, topk_ids, topk_weights)

    def reference(view):
        x_dq = x_q.float() / view["hidden_states_scale_global"]
        w1_dq = view["fc1_expert_weights"].float() * view["fc1_dequant"][:, None, None]
        w2_dq = view["fc2_expert_weights"].float() * view["fc2_dequant"][:, None, None]
        return _reference(
            # fp32 operands, as in the single-runner test above.
            MoEActivationPack(x_dq, None, topk_ids, topk_weights),
            w1_dq,
            w2_dq,
            activation,
            intermediate_hook=fp8_per_tensor_requant_hook(
                view["intermediate_scale_global"]
            ),
        )

    candidates = (
        (TrtllmFp8PerTensorConfig(), "trtllm_fp8_per_tensor"),
        (CutlassFp8PerTensorConfig(), "cutlass_fp8_per_tensor"),
    )
    config = _config(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=quant,
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions(tuple(cfg for cfg, _ in candidates)),
    )
    weights = MoEWeightPack()
    for cfg, key in candidates:
        weights.prepare_for(
            key,
            cfg.prepare_weights(
                w1,
                w2,
                num_local_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                device=device,
                **scales,
            ),
        )

    layer = MoELayer(config)
    assert {r.backend_key for r in layer.runners} == {key for _, key in candidates}
    outputs = {}
    for runner in layer.runners:
        inputs = runner.pack_inputs(act, weights)
        out = runner.forward(inputs, tactic=-1, **runner.launch_kwargs_for(inputs))
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"{runner.backend_key} produced non-finite"
        outputs[runner.backend_key] = out.clone().to(torch.bfloat16)
    # Both backends quantize the same weights with the same static scales, so
    # they agree tightly; pin CUTLASS to its dequantized reference and TRT-LLM
    # to CUTLASS.
    expected = reference(weights.get_view("cutlass_fp8_per_tensor"))
    _assert_numerically_close(
        outputs["cutlass_fp8_per_tensor"], expected, rtol=1e-1, atol=1e-1
    )
    _assert_numerically_close(
        outputs["trtllm_fp8_per_tensor"],
        outputs["cutlass_fp8_per_tensor"],
        rtol=1e-1,
        atol=1e-1,
    )
    actual = layer(act, weights)
    assert layer.winner_backend in outputs
    torch.testing.assert_close(
        actual.to(torch.bfloat16), outputs[layer.winner_backend], rtol=5e-2, atol=5e-2
    )


@cutlass_fp8_required
def test_fp8_per_tensor_prepare_activations_is_graph_capturable():
    """A device-tensor scale (the one stored in the view) launches no host sync."""
    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.randn(8, 128, device=device, dtype=torch.bfloat16)
    scale = fp8_per_tensor_global_scale(x)
    eager, _ = CutlassFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=scale
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        CutlassFp8PerTensorConfig.prepare_activations(
            x, hidden_states_scale_global=scale
        )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, _ = CutlassFp8PerTensorConfig.prepare_activations(
            x, hidden_states_scale_global=scale
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured.float(), eager.float())


def _assert_backends_share_activation_pack(
    *,
    quant: QuantConfig,
    candidates,
    act: MoEActivationPack,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation,
    cutlass_key: str,
    cutlass_reference,
    ref_tol: float,
    cross_tol: float,
):
    """One activation pack must feed every backend in a mixed candidate set.

    Builds a single ``MoELayer`` over ``candidates`` (``(config, backend_key)``
    pairs), prepares each backend's weight view from the same BF16 experts,
    runs every runner standalone on ``act``, pins the CUTLASS runner to its
    dequantized reference and the remaining consumers of the pack to the
    CUTLASS output (all backends quantize the same weights, so they agree far
    more tightly than any of them agrees with a BF16-weight reference), then
    checks the autotuned layer reproduces its winner's standalone output.
    """
    device = act.hidden_states_q.device
    num_experts, hidden_size, intermediate_size = w2.shape
    top_k = act.topk_ids.shape[1]
    config = _config(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=quant,
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=activation,
        backend=BackendOptions(tuple(cfg for cfg, _ in candidates)),
    )
    # A candidate may decline this pair on this device (CuTe-DSL W4A8 on
    # SM107); the test is about the survivors sharing one pack, so drop those
    # rather than fail, but keep at least the CUTLASS runner and one peer.
    supported = []
    for cfg, key in candidates:
        runner = _BACKEND_RUNNERS[type(cfg)](config, device)
        try:
            runner.check_support()
        except NotImplementedError as exc:
            if key == cutlass_key:
                pytest.skip(f"{key} unsupported here: {exc}")
            continue
        supported.append((cfg, key))
    if len(supported) < 2:
        pytest.skip("needs at least two backends that support this pair here")
    candidates = tuple(supported)
    config = dataclasses.replace(
        config, backend=BackendOptions(tuple(cfg for cfg, _ in candidates))
    )
    weights = MoEWeightPack()
    for cfg, key in candidates:
        kwargs = dict(
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )
        if not key.startswith("cutlass_"):
            kwargs["quant"] = quant
        weights.prepare_for(key, cfg.prepare_weights(w1, w2, **kwargs))

    layer = MoELayer(config)
    assert {runner.backend_key for runner in layer.runners} == {
        key for _, key in candidates
    }
    outputs = {}
    for runner in layer.runners:
        inputs = runner.pack_inputs(act, weights)
        out = runner.forward(inputs, tactic=-1, **runner.launch_kwargs_for(inputs))
        torch.cuda.synchronize()
        assert torch.isfinite(out).all(), f"{runner.backend_key} produced non-finite"
        outputs[runner.backend_key] = out.clone().to(torch.bfloat16)
    expected = cutlass_reference(weights.get_view(cutlass_key))
    _assert_numerically_close(
        outputs[cutlass_key], expected, rtol=ref_tol, atol=ref_tol
    )
    for key in outputs:
        if key != cutlass_key:
            _assert_numerically_close(
                outputs[key], outputs[cutlass_key], rtol=cross_tol, atol=cross_tol
            )

    actual = layer(act, weights)
    assert layer.winner_backend in outputs
    torch.testing.assert_close(
        actual.to(torch.bfloat16), outputs[layer.winner_backend], rtol=5e-2, atol=5e-2
    )


@cutlass_nvfp4_required
def test_cutlass_nvfp4_shares_canonical_pack_with_trtllm_and_cute_dsl():
    """The NVFP4 pack from ``TrtllmFp4Config.prepare_activations`` feeds
    TRTLLM, CuTe-DSL, and CUTLASS runners built from one ``MoELayer``."""
    from flashinfer.fused_moe import CuteDslConfig, TrtllmFp4Config

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = get_compute_capability(device)
    arch = major * 10 + minor
    if not (TrtllmFp4Config.supported(arch) and CuteDslConfig.supported(arch)):
        pytest.skip("requires an arch with TRTLLM, CuTe-DSL, and CUTLASS NVFP4")

    torch.manual_seed(46)
    num_tokens, num_experts, top_k = 32, 8, 2
    hidden_size, intermediate_size = 256, 512
    activation = SwiGLU()
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    x_q, x_sf = TrtllmFp4Config.prepare_activations(x)
    act = MoEActivationPack(x_q, x_sf, topk_ids, topk_weights)
    _assert_backends_share_activation_pack(
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        candidates=(
            (TrtllmFp4Config(), "trtllm_fp4_routed"),
            (CuteDslConfig(), "cute_dsl"),
            (CutlassNvfp4Config(), "cutlass_nvfp4"),
        ),
        act=act,
        w1=w1,
        w2=w2,
        activation=activation,
        cutlass_key="cutlass_nvfp4",
        cutlass_reference=lambda view: _nvfp4_quantized_reference(
            act, view, activation
        ),
        # H=256 / I=512 accumulate more NVFP4 requantization noise than the
        # H=128 single-backend case; the bound only needs to catch a wrong
        # magnitude.
        ref_tol=3e-1,
        cross_tol=2e-1,
    )


def _mxfp8_pack_reference(act: MoEActivationPack, w1_dq, w2_dq, activation):
    from flashinfer import mxfp8_dequantize_host

    x_dq = mxfp8_dequantize_host(
        act.hidden_states_q.cpu().view(torch.uint8),
        act.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
        False,
    ).to(device=act.hidden_states_q.device, dtype=torch.bfloat16)
    return _reference(
        MoEActivationPack(x_dq, None, act.topk_ids, act.topk_weights),
        w1_dq,
        w2_dq,
        activation,
    )


@cutlass_mxfp8_required
def test_cutlass_mxfp8_shares_canonical_pack_with_trtllm():
    """The MXFP8 pack from ``TrtllmFp8BlockConfig.prepare_activations`` feeds
    TRTLLM and CUTLASS MXFP8xMXFP8 runners built from one ``MoELayer``."""
    from flashinfer import mxfp8_dequantize_host
    from flashinfer.fused_moe import TrtllmFp8BlockConfig

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = get_compute_capability(device)
    if not TrtllmFp8BlockConfig.supported(major * 10 + minor):
        pytest.skip("requires an arch with TRTLLM and CUTLASS MXFP8")

    torch.manual_seed(52)
    num_tokens, num_experts, top_k = 32, 8, 2
    hidden_size, intermediate_size = 256, 512
    activation = SwiGLU()
    quant = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    x_q, x_sf = TrtllmFp8BlockConfig.prepare_activations(x, quant=quant)
    act = MoEActivationPack(x_q, x_sf, topk_ids, topk_weights)

    def reference(view):
        dq = lambda w, sf: torch.stack(
            [
                mxfp8_dequantize_host(
                    w[i].cpu().view(torch.uint8),
                    sf[i].cpu().view(torch.uint8).reshape(-1),
                    True,
                )
                for i in range(num_experts)
            ]
        ).to(device=device, dtype=torch.bfloat16)
        return _mxfp8_pack_reference(
            act,
            dq(view["fc1_expert_weights"], view["fc1_expert_scales"]),
            dq(view["fc2_expert_weights"], view["fc2_expert_scales"]),
            activation,
        )

    _assert_backends_share_activation_pack(
        quant=quant,
        candidates=(
            (TrtllmFp8BlockConfig(), "trtllm_fp8_block"),
            (CutlassMxfp8Config(), "cutlass_mxfp8"),
        ),
        act=act,
        w1=w1,
        w2=w2,
        activation=activation,
        cutlass_key="cutlass_mxfp8",
        cutlass_reference=reference,
        ref_tol=2e-1,
        cross_tol=2e-1,
    )


@cutlass_mxfp8_mxfp4_required
def test_cutlass_mxfp8_mxfp4_shares_canonical_pack_with_trtllm_and_cute_dsl():
    """The MXFP4xMXFP8 pack from ``TrtllmFp4Config.prepare_activations`` feeds
    TRTLLM, CuTe-DSL, and CUTLASS runners built from one ``MoELayer``."""
    from flashinfer import mxfp4_dequantize
    from flashinfer.fused_moe import CuteDslConfig, TrtllmFp4Config

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = get_compute_capability(device)
    arch = major * 10 + minor
    if not (TrtllmFp4Config.supported(arch) and CuteDslConfig.supported(arch)):
        pytest.skip("requires an arch with TRTLLM, CuTe-DSL, and CUTLASS MXFP4")

    torch.manual_seed(53)
    num_tokens, num_experts, top_k = 32, 8, 2
    hidden_size, intermediate_size = 256, 512
    activation = SwiGLU()
    quant = QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8)
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 2
    w1, w2 = _make_bf16_experts(
        num_experts, hidden_size, intermediate_size, device, activation
    )
    topk_ids, topk_weights = _make_routing(num_tokens, num_experts, top_k, device)
    x_q, x_sf = TrtllmFp4Config.prepare_activations(x, quant=quant)
    act = MoEActivationPack(x_q, x_sf, topk_ids, topk_weights)

    def reference(view):
        dq = lambda w, sf: torch.stack(
            [
                mxfp4_dequantize(w[i].cpu(), sf[i].cpu().view(torch.uint8).reshape(-1))
                for i in range(num_experts)
            ]
        ).to(device=device, dtype=torch.bfloat16)
        return _mxfp8_pack_reference(
            act,
            dq(view["fc1_expert_weights"], view["fc1_expert_scales"]),
            dq(view["fc2_expert_weights"], view["fc2_expert_scales"]),
            activation,
        )

    _assert_backends_share_activation_pack(
        quant=quant,
        candidates=(
            (TrtllmFp4Config(), "trtllm_fp4_routed"),
            (CuteDslConfig(), "cute_dsl"),
            (CutlassMxfp8Mxfp4Config(), "cutlass_mxfp8_mxfp4"),
        ),
        act=act,
        w1=w1,
        w2=w2,
        activation=activation,
        cutlass_key="cutlass_mxfp8_mxfp4",
        cutlass_reference=reference,
        ref_tol=2e-1,
        cross_tol=2e-1,
    )
