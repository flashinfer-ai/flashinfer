"""CPU-only tests for the unified GEMM API configuration surface.

No GPU, no JIT: config dataclasses, validation, ``repr`` round-trip, the
``**config`` protocol, backend-config capability sets, the weight-pack
protocol and the ``torch.nn.Linear`` inheritance contract of ``Linear``.
GPU behaviour lives in ``tests/gemm/test_unified_gemm_api.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from flashinfer.fused_moe import SwiGLU
from flashinfer.gemm import (
    DEFAULT_GEMM_BACKENDS,
    BackendUnsupported,
    CublasGemmConfig,
    CudnnGemmConfig,
    CuteDslGemmConfig,
    CutlassGemmConfig,
    EpilogueConfig,
    ExecutionConfig,
    GemmActivationPack,
    GemmConfig,
    GemmProblem,
    GemmWeightPack,
    Linear,
    TrtllmGemmConfig,
)
from flashinfer.gemm.api import (
    BackendOptions,
    Identity,
    QuantConfig,
    QuantFormat,
    output_torch_dtype,
)
from flashinfer.gemm.runners import _BACKEND_RUNNERS, runner_class_for

pytestmark = pytest.mark.cpu_only

NVFP4 = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
FP8 = QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor)
ALL_BACKENDS = (
    CutlassGemmConfig(),
    CuteDslGemmConfig(),
    TrtllmGemmConfig(),
    CudnnGemmConfig(),
    CublasGemmConfig(),
)


# ---------------------------------------------------------------------------
# GemmConfig
# ---------------------------------------------------------------------------


def test_default_config_is_bf16_with_default_backends():
    cfg = GemmConfig()
    assert cfg.quant == QuantConfig()
    assert cfg.quant.pair == (QuantFormat.BF16, QuantFormat.BF16)
    assert cfg.backend is DEFAULT_GEMM_BACKENDS
    assert cfg.epilogue == EpilogueConfig()
    assert cfg.execution == ExecutionConfig()


def test_config_is_hashable_and_repr_round_trips():
    cfg = GemmConfig(
        quant=NVFP4,
        epilogue=EpilogueConfig(bias=True),
        backend=BackendOptions(
            (CutlassGemmConfig(), TrtllmGemmConfig(use_8x4_sf_layout=True))
        ),
        execution=ExecutionConfig(enable_pdl=False, tune_max_m=4096),
    )
    hash(cfg)
    text = repr(cfg)
    # Enum members use the MoE API's qualified repr, so the string is constructor syntax.
    assert (
        "QuantFormat.NVFP4" in text
        and "TrtllmGemmConfig(use_8x4_sf_layout=True)" in text
    )
    rebuilt = eval(  # noqa: S307 - same-version repr round-trip, as the MoE API tests do
        text,
        {
            "GemmConfig": GemmConfig,
            "QuantConfig": QuantConfig,
            "QuantFormat": QuantFormat,
            "EpilogueConfig": EpilogueConfig,
            "ExecutionConfig": ExecutionConfig,
            "BackendOptions": BackendOptions,
            "CutlassGemmConfig": CutlassGemmConfig,
            "TrtllmGemmConfig": TrtllmGemmConfig,
            "Identity": Identity,
        },
    )
    assert rebuilt == cfg


def test_config_unpacks_like_a_mapping():
    cfg = GemmConfig(quant=FP8)
    assert list(cfg.keys()) == ["quant", "epilogue", "backend", "execution"]
    assert cfg["quant"] is cfg.quant
    assert dict(**cfg)["backend"] is cfg.backend


def test_replace_makes_variants_without_mutation():
    cfg = GemmConfig(quant=NVFP4)
    fp8 = dataclasses.replace(cfg, quant=FP8)
    assert cfg.quant is NVFP4 and fp8.quant is FP8
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.quant = FP8  # type: ignore[misc]


@pytest.mark.parametrize(
    "make, exc",
    [
        (lambda: GemmConfig(quant=torch.bfloat16), TypeError),
        # Backend configs not wrapped in BackendOptions.
        (lambda: GemmConfig(backend=(CutlassGemmConfig(),)), TypeError),
        (lambda: GemmConfig(backend=BackendOptions(())), ValueError),
        (lambda: GemmConfig(backend=BackendOptions((object(),))), TypeError),
        (lambda: ExecutionConfig(tune_max_m=0), ValueError),
        (lambda: EpilogueConfig(activation="swiglu"), TypeError),
        (lambda: GemmProblem(n=0, k=512), ValueError),
        (lambda: GemmProblem(n=256, k=512, m=-1), ValueError),
    ],
)
def test_config_validation_names_the_mistake(make, exc):
    with pytest.raises(exc):
        make()


def test_epilogue_accepts_typed_activation_values():
    ep = EpilogueConfig(activation=SwiGLU(alpha=1.0, beta=0.0))
    assert ep.activation.is_gated
    assert "SwiGLU" in repr(ep)


def test_output_dtype_mapping():
    assert output_torch_dtype(QuantConfig()) is torch.bfloat16
    assert output_torch_dtype(QuantConfig(output=QuantFormat.FP16)) is torch.float16
    with pytest.raises(BackendUnsupported):
        output_torch_dtype(QuantConfig(output=QuantFormat.NVFP4))


def test_gemm_problem_defaults():
    p = GemmProblem(n=256, k=512)
    assert (p.m, p.form) == (None, "dense")


# ---------------------------------------------------------------------------
# Backend configs: capability records
# ---------------------------------------------------------------------------


def test_backend_keys_are_the_legacy_backend_strings():
    assert [c.backend_key for c in ALL_BACKENDS] == [
        "cutlass",
        "cute-dsl",
        "trtllm",
        "cudnn",
        "cublas",
    ]


@pytest.mark.parametrize("cand", ALL_BACKENDS)
def test_backend_configs_are_frozen_hashable_and_repr_round_trip(cand):
    hash(cand)
    assert eval(repr(cand), {type(cand).__name__: type(cand)}) == cand  # noqa: S307
    with pytest.raises(dataclasses.FrozenInstanceError):
        cand.backend_key = "x"  # type: ignore[misc]


def test_supports_quant_matches_the_mvp_matrix():
    nvfp4 = {c.backend_key for c in ALL_BACKENDS if c.supports_quant(NVFP4)}
    fp8 = {c.backend_key for c in ALL_BACKENDS if c.supports_quant(FP8)}
    assert nvfp4 == {"cutlass", "cute-dsl", "trtllm", "cudnn"}
    assert fp8 == {"cutlass", "cudnn", "cublas"}
    assert not any(
        c.supports_quant(QuantConfig()) for c in ALL_BACKENDS
    )  # BF16: post-MVP


def test_every_supported_pair_has_a_runner_adapter():
    for cand in ALL_BACKENDS:
        for pair in cand.supported_quant_pairs:
            cfg = GemmConfig(quant=QuantConfig(weight=pair[0], activation=pair[1]))
            assert runner_class_for(cand, cfg) is not None, (cand, pair)
    assert len(_BACKEND_RUNNERS) == 7


def test_default_backends_filter_by_arch():
    assert [c.backend_key for c in DEFAULT_GEMM_BACKENDS.valid_for(100)] == [
        "cutlass",
        "cublas",
        "cudnn",
    ]
    # Hopper / Ada: no CUTLASS block-scaled GEMM, cuBLAS and cuDNN FP8 remain.
    assert [c.backend_key for c in DEFAULT_GEMM_BACKENDS.valid_for(90)] == [
        "cublas",
        "cudnn",
    ]
    assert [c.backend_key for c in DEFAULT_GEMM_BACKENDS.valid_for(89)] == [
        "cublas",
        "cudnn",
    ]
    assert DEFAULT_GEMM_BACKENDS.valid_for(80) == []


def test_weight_recipes_share_views_between_backends():
    assert CutlassGemmConfig.weight_recipe(NVFP4) == CudnnGemmConfig.weight_recipe(
        NVFP4
    )
    assert CutlassGemmConfig.weight_recipe(NVFP4) == CuteDslGemmConfig.weight_recipe(
        NVFP4
    )
    assert TrtllmGemmConfig.weight_recipe(NVFP4) == "nvfp4_128x4_shuffled"
    assert CublasGemmConfig.weight_recipe(FP8) == CutlassGemmConfig.weight_recipe(FP8)
    with pytest.raises(BackendUnsupported):
        CublasGemmConfig.weight_recipe(NVFP4)


# ---------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------


def test_weight_pack_protocol():
    pack = GemmWeightPack((256, 512))
    assert pack.out_features == 256 and pack.in_features == 512
    view = {"recipe": "fp8_per_tensor", "b": torch.zeros(256, 512, dtype=torch.uint8)}
    pack.prepare_for("cublas", view)
    pack.prepare_for("cudnn", view)
    assert pack.get_view("cublas") is pack.get_view("cudnn")
    assert pack.view_for_recipe("fp8_per_tensor") is view
    assert pack.shapes("cublas") == {"b": (256, 512)}
    with pytest.raises(KeyError):
        pack.get_view("cutlass")
    with pytest.raises(ValueError):
        pack.prepare_for("x", {"b": torch.zeros(1)})
    assert "fp8_per_tensor" in repr(pack)


def test_activation_pack_shape_and_repr_do_not_touch_data():
    act = GemmActivationPack(
        a=torch.zeros(8, 64, dtype=torch.uint8), a_scale=torch.zeros(8, 8)
    )
    assert act.num_rows == 8 and act.form == "dense"
    assert "Tensor(shape=(8, 64)" in repr(act)
    assert GemmActivationPack(a=torch.zeros(2, 8, 64)).form == "batched"


# ---------------------------------------------------------------------------
# Linear is a torch.nn.Linear
# ---------------------------------------------------------------------------


def test_linear_is_an_nn_linear_with_the_same_state_dict():
    ref = torch.nn.Linear(64, 32, bias=True)
    lin = Linear(64, 32, bias=True, config=GemmConfig(quant=NVFP4))
    assert isinstance(lin, torch.nn.Linear)
    assert list(lin.state_dict()) == list(ref.state_dict()) == ["weight", "bias"]
    lin.load_state_dict(ref.state_dict())
    assert torch.equal(lin.weight, ref.weight) and torch.equal(lin.bias, ref.bias)
    assert lin.config.epilogue.bias is True
    assert (
        Linear(64, 32, bias=False, config=GemmConfig(quant=NVFP4)).config.epilogue.bias
        is False
    )
    assert "prepared=False" in repr(lin) and "GemmConfig(" in repr(lin)


def test_linear_before_prepare_runs_f_linear():
    torch.manual_seed(0)
    lin = Linear(64, 32, config=GemmConfig(quant=NVFP4))
    x = torch.randn(3, 5, 64)
    torch.testing.assert_close(
        lin(x), torch.nn.functional.linear(x, lin.weight, lin.bias)
    )
    assert lin.winner_backend is None and lin.winner_report() is None


def test_linear_prepare_requires_cuda():
    lin = Linear(64, 32, config=GemmConfig(quant=NVFP4))
    with pytest.raises(RuntimeError, match="CUDA"):
        lin.prepare_weights()
