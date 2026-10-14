"""CPU contract tests for the exact-SM100/SM103 Cake warp-decode MoE runner."""

from __future__ import annotations

import functools
import gc
import weakref
from collections import OrderedDict
from dataclasses import dataclass, replace
from types import SimpleNamespace

import flashinfer.fused_moe.runners as moe_runners
import flashinfer.jit.cake_fused_moe_warp_decode as cake_warp_decode_jit
import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CakeWarpDecodeConfig,
    CakeWarpDecodeRunner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    SiLU,
    SiTU,
    SwiGLU,
    TrtllmFp4Config,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.api import _DEFAULT_BACKEND
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS


@dataclass
class _TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device = torch.device("cpu")
    contiguous: bool = True

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self.contiguous


class _Module:
    def __init__(self) -> None:
        self.size_calls = []
        self.prepare_calls = []
        self.release_calls = []
        self.run_calls = []
        self.parameter_run_calls = []
        self.next_receipt = 1
        self.live_receipts = set()
        self.release_error: Exception | None = None
        self.run_error: Exception | None = None

    def cake_fused_moe_warp_decode_workspace_size(self, *geometry):
        self.size_calls.append(geometry)
        return 64

    def cake_fused_moe_warp_decode_prepare_workspace(self, workspace, *geometry) -> int:
        self.prepare_calls.append((workspace, geometry))
        receipt = self.next_receipt
        self.next_receipt += 1
        self.live_receipts.add(receipt)
        return receipt

    def cake_fused_moe_warp_decode_release_workspace(self, receipt) -> None:
        self.release_calls.append(receipt)
        if receipt <= 0 or receipt not in self.live_receipts:
            raise RuntimeError("unknown or already released receipt")
        if self.release_error is not None:
            raise self.release_error
        self.live_receipts.remove(receipt)

    def cake_fused_moe_warp_decode(self, *args) -> None:
        self.run_calls.append(args)
        if self.run_error is not None:
            raise self.run_error

    def cake_fused_moe_warp_decode_with_activation_params(self, *args) -> None:
        self.parameter_run_calls.append(args)
        if self.run_error is not None:
            raise self.run_error


@pytest.mark.parametrize("target", ["sm100a", "sm103a"])
def test_jit_device_sources_resolve_and_share_common_kernels(target):
    csrc_dir = cake_warp_decode_jit._get_cake_fused_moe_warp_decode_csrc_dir()
    sources = cake_warp_decode_jit._device_sources(csrc_dir, target)
    names = [source.name for source in sources]
    assert sources and len(set(names)) == len(names)
    assert set(cake_warp_decode_jit._COMMON_SOURCES) <= set(names)
    assert set(cake_warp_decode_jit._NO_FAST_MATH_SOURCES) <= set(names)
    with pytest.raises(FileNotFoundError, match="device sources not found"):
        cake_warp_decode_jit._device_sources(csrc_dir / "missing", target)


def test_jit_prefers_checkout_sources_when_editable_data_is_staged(
    tmp_path, monkeypatch
):
    checkout_module = tmp_path / "flashinfer" / "jit" / "cake_warp_decode.py"
    checkout_module.parent.mkdir(parents=True)
    checkout_module.touch()
    checkout_csrc = tmp_path / "csrc" / "fused_moe" / "warp_decode"
    checkout_csrc.mkdir(parents=True)
    staged_csrc = (
        tmp_path / "flashinfer" / "data" / "csrc" / "fused_moe" / "warp_decode"
    )
    staged_csrc.mkdir(parents=True)

    monkeypatch.setattr(cake_warp_decode_jit, "__file__", str(checkout_module))
    monkeypatch.setattr(
        cake_warp_decode_jit.jit_env,
        "FLASHINFER_CSRC_DIR",
        tmp_path / "flashinfer" / "data" / "csrc",
    )

    assert (
        cake_warp_decode_jit._get_cake_fused_moe_warp_decode_csrc_dir() == checkout_csrc
    )


def _config(
    *,
    intermediate_size: int = 1536,
    num_experts: int = 60,
    top_k: int = 4,
    enable_pdl: bool | None = True,
    activation=None,
) -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=SwiGLU() if activation is None else activation,
        backend=BackendOptions((CakeWarpDecodeConfig(backend="cake"),)),
        execution=ExecutionConfig(enable_pdl=enable_pdl),
    )


def _runner(
    config: MoEConfig | None = None, *, device_arch: int = 103
) -> tuple[CakeWarpDecodeRunner, _Module]:
    module = _Module()
    runner = object.__new__(CakeWarpDecodeRunner)
    runner.config = config or _config()
    runner.device = torch.device("cpu")
    runner._device_arch = device_arch
    runner._module = module
    runner._support_checked = True
    runner._built = True
    runner._workspace_cache = OrderedDict()
    runner._prepared_workspaces = {}
    runner._workspace_receipt_finalizers = {}
    runner._workspace_stream_claims = {}
    runner._topk_validation_receipts = OrderedDict()
    return runner, module


def _activation_pack(
    *,
    num_tokens: int = 7,
    top_k: int = 4,
    mode: RoutingInputMode = RoutingInputMode.UnpackedPrecomputed,
    weights_dtype: torch.dtype = torch.bfloat16,
    topk_ids: torch.Tensor | None = None,
    hidden_size: int = 2048,
) -> MoEActivationPack:
    return MoEActivationPack(
        _TensorSpec((num_tokens, hidden_size // 2), torch.uint8),
        _TensorSpec((num_tokens, hidden_size // 16), torch.uint8),
        topk_ids
        if topk_ids is not None
        else _TensorSpec((num_tokens, top_k), torch.int32),
        _TensorSpec((num_tokens, top_k), weights_dtype),
        routing_input_mode=mode,
    )


def _weight_pack(
    *,
    hidden_size: int = 2048,
    intermediate_size: int = 1536,
    num_experts: int = 60,
    activation=None,
    extra: dict | None = None,
) -> tuple[MoEWeightPack, dict]:
    activation = SwiGLU() if activation is None else activation
    gemm1_rows = intermediate_size * (2 if activation.is_gated else 1)
    view = {
        "gemm1_weights": _TensorSpec(
            (num_experts, gemm1_rows, hidden_size // 2), torch.uint8
        ),
        "gemm1_weights_scale": _TensorSpec(
            (num_experts, gemm1_rows, hidden_size // 16), torch.uint8
        ),
        "gemm2_weights": _TensorSpec(
            (num_experts, hidden_size, intermediate_size // 2), torch.uint8
        ),
        "gemm2_weights_scale": _TensorSpec(
            (num_experts, hidden_size, intermediate_size // 16), torch.uint8
        ),
        "output1_scale_scalar": _TensorSpec((num_experts,), torch.float32),
        "output1_scale_gate_scalar": _TensorSpec((num_experts,), torch.float32),
        "output2_scale_scalar": _TensorSpec((num_experts,), torch.float32),
    }
    if activation.is_gated:
        view["gemm1_alpha"] = _TensorSpec((num_experts,), torch.float32)
    if isinstance(activation, SiTU) or (
        isinstance(activation, SwiGLU) and activation != SwiGLU()
    ):
        view["gemm1_beta"] = _TensorSpec((num_experts,), torch.float32)
    if isinstance(activation, SwiGLU) and activation != SwiGLU():
        view["gemm1_clamp_limit"] = _TensorSpec((num_experts,), torch.float32)
    if extra:
        view.update(extra)
    weights = MoEWeightPack()
    weights.prepare_for("cake", view)
    return weights, view


def test_config_is_explicit_exact_sm100_sm103_and_not_default():
    config = CakeWarpDecodeConfig(backend="cake")
    assert repr(config) == "CakeWarpDecodeConfig(backend='cake')"
    assert config.supported(100)
    assert config.supported(103)
    assert not config.supported(101)
    assert not config.supported(107)
    assert _BACKEND_RUNNERS[CakeWarpDecodeConfig] is CakeWarpDecodeRunner
    assert not any(isinstance(item, CakeWarpDecodeConfig) for item in _DEFAULT_BACKEND)
    with pytest.raises(ValueError, match="must be 'cake'"):
        CakeWarpDecodeConfig(backend="auto")


@pytest.mark.parametrize("device_arch,target", [(100, "sm100a"), (103, "sm103a")])
def test_runner_build_passes_its_explicit_target_and_device(
    monkeypatch, device_arch, target
):
    runner, _ = _runner(device_arch=device_arch)
    runner.device = torch.device("cuda:1")
    sentinel = object()
    calls = []

    def load(requested_target, *, device):
        calls.append((requested_target, device))
        return sentinel

    monkeypatch.setattr(
        "flashinfer.jit.cake_fused_moe_warp_decode."
        "get_cake_fused_moe_warp_decode_module",
        load,
    )
    runner._build()

    assert runner._module is sentinel
    assert calls == [(target, torch.device("cuda:1"))]


@pytest.mark.parametrize(
    "activation,expected_activation,hidden_size,intermediate_size,num_experts",
    [
        (None, SwiGLU(), 2048, 512, 512),
        (SwiGLU(), SwiGLU(), 2048, 1536, 60),
        (None, SwiGLU(), 2560, 768, 384),
        (SwiGLU(), SwiGLU(), 2560, 768, 384),
        (SiLU(), SiLU(), 6144, 1536, 192),
        (None, SwiGLU(), 2048, 768, 128),
        (SwiGLU(), SwiGLU(), 4096, 1536, 128),
        (SwiGLU(), SwiGLU(), 2048, 512, 256),
        (SwiGLU(), SwiGLU(), 4096, 1024, 512),
        (SwiGLU(), SwiGLU(), 3072, 1536, 256),
        (SwiGLU(), SwiGLU(), 4096, 512, 512),
        (None, SwiGLU(), 4096, 256, 512),
        (SwiGLU(), SwiGLU(), 3072, 768, 256),
        (None, SwiGLU(), 3072, 384, 256),
        (
            SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
            SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
            6144,
            3072,
            128,
        ),
        (SiTU(), SiTU(), 3584, 3072, 896),
    ],
)
def test_config_preparation_delegates_to_trtllm_physical_view(
    monkeypatch,
    activation,
    expected_activation,
    hidden_size,
    intermediate_size,
    num_experts,
):
    expected = object()
    calls = []

    def prepare_weights(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(TrtllmFp4Config, "prepare_weights", prepare_weights)
    result = CakeWarpDecodeConfig.prepare_weights(
        object(),
        object(),
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
    )
    assert result is expected
    assert calls[0][1]["quant"] == QuantConfig(
        weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4
    )
    assert calls[0][1]["activation"] == expected_activation
    with pytest.raises(ValueError, match=r"NVFP4×NVFP4"):
        CakeWarpDecodeConfig.prepare_weights(
            object(),
            object(),
            quant=QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
        )


@pytest.mark.parametrize(
    "activation,hidden_size,intermediate_size,num_experts",
    [
        (SiLU(), 2048, 1536, 60),
        (SwiGLU(), 6144, 1536, 192),
        (SwiGLU(alpha=2.0), 2048, 1536, 60),
        (SiLU(), 2560, 768, 384),
        (SwiGLU(alpha=2.0), 2560, 768, 384),
        (SiLU(), 2048, 768, 128),
        (SwiGLU(alpha=2.0), 4096, 1536, 128),
        (SiLU(), 2048, 512, 256),
        (SiLU(), 4096, 1024, 512),
        (SiLU(), 3072, 1536, 256),
        (SiLU(), 4096, 512, 512),
        (SwiGLU(alpha=2.0), 4096, 256, 512),
        (SiLU(), 3072, 768, 256),
        (SwiGLU(alpha=2.0), 3072, 384, 256),
        (SwiGLU(), 6144, 3072, 128),
        (SwiGLU(), 3584, 3072, 896),
        (SwiGLU(alpha=1.702, beta=0.0, limit=7.0), 6144, 3072, 128),
        (SiTU(gate_scale=1.0), 3584, 3072, 896),
    ],
)
def test_config_preparation_rejects_activation_geometry_cross_product(
    activation, hidden_size, intermediate_size, num_experts
):
    with pytest.raises(ValueError, match="supports only default SwiGLU"):
        CakeWarpDecodeConfig.prepare_weights(
            object(),
            object(),
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
        )


@pytest.mark.parametrize("enable_pdl", [None, False])
def test_support_requires_explicit_pdl(enable_pdl):
    runner, _ = _runner(_config(enable_pdl=enable_pdl))
    with pytest.raises(NotImplementedError, match="enable_pdl=True"):
        runner._check_support()


def test_support_rejects_semantic_and_geometry_expansion():
    runner, _ = _runner()
    runner._check_support()

    runner.config = replace(runner.config, activation=SwiGLU(alpha=2.0))
    with pytest.raises(NotImplementedError, match="default SwiGLU"):
        runner._check_support()

    runner.config = _config(
        intermediate_size=1536, num_experts=192, top_k=4, activation=SiLU()
    )
    runner._check_support()

    runner.config = _config(
        intermediate_size=1536, num_experts=60, top_k=4, activation=SiLU()
    )
    with pytest.raises(NotImplementedError, match="default SwiGLU"):
        runner._check_support()

    runner.config = replace(_config(), finalize=MoEFinalizeConfig(do_finalize=False))
    with pytest.raises(NotImplementedError, match="do_finalize=True"):
        runner._check_support()

    runner.config = _config(intermediate_size=1024)
    with pytest.raises(NotImplementedError, match="supports only"):
        runner._check_support()

    runner.config = replace(
        _config(),
        experts=ExpertConfig(
            intermediate_size=1536,
            local_expert_offset=1,
            local_num_experts=60,
        ),
    )
    with pytest.raises(NotImplementedError, match="expert parallelism"):
        runner._check_support()

    runner.config = replace(
        _config(),
        routing=RoutingConfig(
            num_experts=60, top_k=4, method=RoutingMethodType.DeepSeekV3
        ),
        experts=ExpertConfig(intermediate_size=1536, num_fused_shared_experts=1),
    )
    with pytest.raises(NotImplementedError, match="fused shared experts"):
        runner._check_support()


@pytest.mark.parametrize("device_arch", [90, 101, 107, 120])
def test_support_rejects_nonexact_architectures(device_arch):
    runner, _ = _runner(device_arch=device_arch)
    with pytest.raises(NotImplementedError, match="exact SM100 or SM103"):
        runner._check_support()


@pytest.mark.parametrize("device_arch", [100, 103])
@pytest.mark.parametrize(
    "intermediate_size,num_experts,top_k",
    [
        (1536, 60, 4),
        (768, 384, 4),
        (768, 128, 8),
        (1536, 128, 8),
        (512, 256, 8),
        (1024, 512, 10),
        (1536, 256, 8),
    ],
)
def test_support_accepts_exact_architectures(
    device_arch, intermediate_size, num_experts, top_k
):
    runner, _ = _runner(
        _config(
            intermediate_size=intermediate_size, num_experts=num_experts, top_k=top_k
        ),
        device_arch=device_arch,
    )
    runner._check_support()


@pytest.mark.parametrize("device_arch", [100, 103])
@pytest.mark.parametrize(
    "hidden_size,intermediate_size,num_experts,top_k",
    [
        (2048, 1536, 60, 4),
        (2560, 768, 384, 4),
        (2048, 768, 128, 8),
        (4096, 1536, 128, 8),
        (2048, 512, 256, 8),
        (4096, 1024, 512, 10),
        (3072, 1536, 256, 8),
        (4096, 512, 512, 10),
        (4096, 256, 512, 10),
        (3072, 768, 256, 8),
        (3072, 384, 256, 8),
    ],
)
def test_pack_reuses_prepared_workspace_and_preserves_ffi_order(
    device_arch, hidden_size, intermediate_size, num_experts, top_k
):
    runner, module = _runner(
        _config(
            intermediate_size=intermediate_size, num_experts=num_experts, top_k=top_k
        ),
        device_arch=device_arch,
    )
    act = _activation_pack(hidden_size=hidden_size, top_k=top_k)
    weights, view = _weight_pack(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )

    first = runner.pack_inputs(act, weights)
    second = runner.pack_inputs(act, weights)

    assert module.size_calls == [
        (7, hidden_size, intermediate_size, num_experts, top_k)
    ]
    assert len(module.prepare_calls) == 1
    assert first[1] is second[1]
    assert first[0] is not second[0]
    assert first[2:6] == [
        act.hidden_states_q,
        act.hidden_states_scale,
        act.topk_ids,
        act.topk_weights,
    ]
    assert first[6:] == [
        view["gemm1_weights"],
        view["gemm1_weights_scale"],
        view["gemm2_weights"],
        view["gemm2_weights_scale"],
        view["output1_scale_scalar"],
        view["output1_scale_gate_scalar"],
        view["output2_scale_scalar"],
    ]

    assert runner.forward(first, do_preparation=True) is first[0]
    assert module.run_calls == []
    assert len(module.prepare_calls) == 2
    assert runner.forward(first) is first[0]
    assert module.run_calls == [tuple([*first, 2, True])]


@pytest.mark.parametrize("device_arch", [100, 103])
@pytest.mark.parametrize(
    "activation,hidden_size,num_experts,top_k",
    [
        (SwiGLU(alpha=1.702, beta=1.0, limit=7.0), 6144, 128, 4),
        (SiTU(gate_scale=4.0, linear_scale=25.0), 3584, 896, 16),
    ],
)
def test_parameterized_pack_forwards_live_activation_tensors(
    device_arch, activation, hidden_size, num_experts, top_k
):
    runner, module = _runner(
        _config(
            intermediate_size=3072,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
        ),
        device_arch=device_arch,
    )
    runner._check_support()
    act = _activation_pack(hidden_size=hidden_size, top_k=top_k)
    weights, view = _weight_pack(
        hidden_size=hidden_size,
        intermediate_size=3072,
        num_experts=num_experts,
        activation=activation,
    )
    inputs = runner.pack_inputs(act, weights)
    assert len(inputs) == 16
    assert inputs[13] is view["gemm1_alpha"]
    assert inputs[14] is view["gemm1_beta"]
    assert inputs[15] is view.get("gemm1_clamp_limit")
    assert module.size_calls == [(7, hidden_size, 3072, num_experts, top_k)]
    assert runner.forward(inputs) is inputs[0]
    assert module.run_calls == []
    assert module.parameter_run_calls == [tuple([*inputs, 1, True])]


@pytest.mark.parametrize(
    "activation,hidden_size,num_experts,top_k,missing_key",
    [
        (SwiGLU(alpha=1.702, beta=1.0, limit=7.0), 6144, 128, 4, "gemm1_alpha"),
        (SwiGLU(alpha=1.702, beta=1.0, limit=7.0), 6144, 128, 4, "gemm1_beta"),
        (SwiGLU(alpha=1.702, beta=1.0, limit=7.0), 6144, 128, 4, "gemm1_clamp_limit"),
        (SiTU(), 3584, 896, 16, "gemm1_alpha"),
        (SiTU(), 3584, 896, 16, "gemm1_beta"),
    ],
)
def test_parameterized_pack_requires_prepared_activation_tensors(
    activation, hidden_size, num_experts, top_k, missing_key
):
    runner, module = _runner(
        _config(
            intermediate_size=3072,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
        )
    )
    act = _activation_pack(hidden_size=hidden_size, top_k=top_k)
    weights, view = _weight_pack(
        hidden_size=hidden_size,
        intermediate_size=3072,
        num_experts=num_experts,
        activation=activation,
    )
    del view[missing_key]
    with pytest.raises(KeyError, match=missing_key):
        runner.pack_inputs(act, weights)
    assert module.prepare_calls == []


def test_situ_pack_rejects_unconsumed_clamp_override():
    runner, _ = _runner(
        _config(intermediate_size=3072, num_experts=896, top_k=16, activation=SiTU())
    )
    weights, _ = _weight_pack(
        hidden_size=3584,
        intermediate_size=3072,
        num_experts=896,
        activation=SiTU(),
        extra={"gemm1_clamp_limit": _TensorSpec((896,), torch.float32)},
    )
    with pytest.raises(ValueError, match="gemm1_clamp_limit"):
        runner.pack_inputs(_activation_pack(hidden_size=3584, top_k=16), weights)


def test_silu_pack_uses_nongated_rows_and_preserves_ffi_order():
    config = _config(
        intermediate_size=1536, num_experts=192, top_k=4, activation=SiLU()
    )
    runner, module = _runner(config)
    act = _activation_pack(hidden_size=6144)
    weights, view = _weight_pack(
        hidden_size=6144,
        intermediate_size=1536,
        num_experts=192,
        activation=SiLU(),
    )

    packed = runner.pack_inputs(act, weights)

    assert module.size_calls == [(7, 6144, 1536, 192, 4)]
    assert "gemm1_alpha" not in view
    assert view["gemm1_weights"].shape == (192, 1536, 3072)
    assert view["gemm1_weights_scale"].shape == (192, 1536, 384)
    assert view["gemm2_weights"].shape == (192, 6144, 768)
    assert view["gemm2_weights_scale"].shape == (192, 6144, 96)
    assert len(packed) == 13
    assert packed[2:6] == [
        act.hidden_states_q,
        act.hidden_states_scale,
        act.topk_ids,
        act.topk_weights,
    ]
    assert packed[6:] == [
        view["gemm1_weights"],
        view["gemm1_weights_scale"],
        view["gemm2_weights"],
        view["gemm2_weights_scale"],
        view["output1_scale_scalar"],
        view["output1_scale_gate_scalar"],
        view["output2_scale_scalar"],
    ]


def test_silu_pack_rejects_gated_rows_and_alpha():
    config = _config(
        intermediate_size=1536, num_experts=192, top_k=4, activation=SiLU()
    )
    runner, _ = _runner(config)
    act = _activation_pack(hidden_size=6144)
    gated_rows, _ = _weight_pack(
        hidden_size=6144,
        intermediate_size=1536,
        num_experts=192,
        activation=SiLU(),
        extra={
            "gemm1_weights": _TensorSpec((192, 3072, 3072), torch.uint8),
        },
    )
    with pytest.raises(ValueError, match="gemm1_weights must have shape"):
        runner.pack_inputs(act, gated_rows)

    with_alpha, _ = _weight_pack(
        hidden_size=6144,
        intermediate_size=1536,
        num_experts=192,
        activation=SiLU(),
        extra={"gemm1_alpha": _TensorSpec((192,), torch.float32)},
    )
    with pytest.raises(ValueError, match="must not provide gemm1_alpha"):
        runner.pack_inputs(act, with_alpha)


@pytest.mark.parametrize(
    "config,hidden_size",
    [
        (
            _config(
                intermediate_size=1536,
                num_experts=192,
                top_k=4,
                activation=SiLU(),
            ),
            2048,
        ),
        (_config(), 6144),
        (_config(intermediate_size=768, num_experts=384), 2048),
        (
            _config(intermediate_size=768, num_experts=384, activation=SiLU()),
            2560,
        ),
    ],
)
def test_pack_rejects_activation_geometry_cross_product(config, hidden_size):
    runner, _ = _runner(config)
    with pytest.raises(ValueError, match="supports only default SwiGLU"):
        runner.pack_inputs(
            _activation_pack(hidden_size=hidden_size),
            _weight_pack()[0],
        )


def test_pack_uses_a_distinct_workspace_per_cuda_stream(monkeypatch):
    runner, module = _runner()
    act = _activation_pack()
    weights, _ = _weight_pack()
    streams = iter((SimpleNamespace(cuda_stream=101), SimpleNamespace(cuda_stream=202)))
    monkeypatch.setattr(runner, "_current_stream", lambda: next(streams))

    first = runner.pack_inputs(act, weights)
    second = runner.pack_inputs(act, weights)

    assert first[1] is not second[1]
    assert len(module.size_calls) == 2
    assert len(module.prepare_calls) == 2
    assert [entry[0].cuda_stream for entry in runner._workspace_cache.values()] == [
        101,
        202,
    ]


def test_stream_workspace_cache_fails_closed_at_limit_and_retains_streams():
    runner, _ = _runner()
    geometry = (7, 2048, 1536, 60, 4)
    streams = [SimpleNamespace(cuda_stream=index + 1) for index in range(65)]

    for stream in streams[:64]:
        runner._cache_workspace_for_stream(
            stream, geometry, torch.empty(64, dtype=torch.uint8)
        )
    with pytest.raises(RuntimeError, match="at most 64"):
        runner._cache_workspace_for_stream(
            streams[-1], geometry, torch.empty(64, dtype=torch.uint8)
        )

    assert len(runner._workspace_cache) == 64
    assert runner._workspace_cache[(1, geometry)][0] is streams[0]
    assert (65, geometry) not in runner._workspace_cache


def test_topk_validation_receipt_reuses_one_tensor_version_during_capture(
    monkeypatch,
):
    runner, _ = _runner()
    topk_ids = torch.zeros((7, 4), dtype=torch.int32)
    act = _activation_pack(topk_ids=topk_ids)
    weights, _ = _weight_pack()

    runner.pack_inputs(act, weights)
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: True)
    runner.pack_inputs(act, weights)

    topk_ids[0, 0] = 1
    with pytest.raises(RuntimeError, match="exact tensor version"):
        runner.pack_inputs(act, weights)


def test_capture_rejects_unvalidated_topk_ids(monkeypatch):
    runner, _ = _runner()
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="exact tensor version"):
        runner.pack_inputs(
            _activation_pack(topk_ids=torch.zeros((7, 4), dtype=torch.int32)),
            _weight_pack()[0],
        )


def test_inference_tensor_receipt_is_reusable_during_capture(monkeypatch):
    runner, _ = _runner()
    with torch.inference_mode():
        topk_ids = torch.zeros((7, 4), dtype=torch.int32)
    act = _activation_pack(topk_ids=topk_ids)
    weights, _ = _weight_pack()

    runner.pack_inputs(act, weights)
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: True)
    runner.pack_inputs(act, weights)


def test_topk_validation_receipt_limit_fails_without_eviction(monkeypatch):
    runner, _ = _runner()
    monkeypatch.setattr(runner, "_MAX_TOPK_VALIDATION_RECEIPTS", 2)
    first = torch.zeros((7, 4), dtype=torch.int32)
    second = torch.ones((7, 4), dtype=torch.int32)
    third = torch.full((7, 4), 2, dtype=torch.int32)

    runner._validate_expert_id_range(first, 60)
    runner._validate_expert_id_range(second, 60)
    with pytest.raises(RuntimeError, match="at most 64"):
        runner._validate_expert_id_range(third, 60)

    assert list(runner._topk_validation_receipts) == [id(first), id(second)]


@pytest.mark.parametrize("invalid_id", [-1, 60])
def test_pack_rejects_out_of_range_expert_ids(invalid_id):
    runner, _ = _runner()
    topk_ids = torch.zeros((7, 4), dtype=torch.int32)
    topk_ids[0, 0] = invalid_id
    with pytest.raises(ValueError, match="0 <= id < 60"):
        runner.pack_inputs(
            _activation_pack(topk_ids=topk_ids),
            _weight_pack()[0],
        )


def test_forward_selects_workspace_for_its_current_stream(monkeypatch):
    runner, module = _runner()
    act = _activation_pack()
    weights, _ = _weight_pack()
    stream_a = SimpleNamespace(cuda_stream=101)
    stream_b = SimpleNamespace(cuda_stream=202)
    current_stream = stream_a
    monkeypatch.setattr(runner, "_current_stream", lambda: current_stream)

    inputs_a = runner.pack_inputs(act, weights)
    current_stream = stream_b
    inputs_b = runner.pack_inputs(act, weights)

    runner.forward(inputs_a)
    assert module.run_calls[-1][1] is inputs_b[1]
    current_stream = stream_a
    runner.forward(inputs_b)
    assert module.run_calls[-1][1] is inputs_a[1]


def test_forward_allocates_for_a_new_stream_after_fallback_is_claimed(
    monkeypatch,
):
    runner, module = _runner()
    stream_a = SimpleNamespace(cuda_stream=101)
    stream_b = SimpleNamespace(cuda_stream=202)
    current_stream = stream_a
    monkeypatch.setattr(runner, "_current_stream", lambda: current_stream)

    inputs = runner.pack_inputs(_activation_pack(), _weight_pack()[0])
    runner.forward(inputs)
    current_stream = stream_b
    runner.forward(inputs)

    assert len(module.size_calls) == 2
    assert len(module.prepare_calls) == 2
    assert module.run_calls[-1][1] is not inputs[1]


def test_capture_stream_reuses_packed_workspace_claimed_by_warmup(monkeypatch):
    runner, module = _runner()
    stream_a = SimpleNamespace(cuda_stream=101)
    stream_b = SimpleNamespace(cuda_stream=202)
    current_stream = stream_a
    capturing = False
    monkeypatch.setattr(runner, "_current_stream", lambda: current_stream)
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: capturing)

    inputs = runner.pack_inputs(_activation_pack(), _weight_pack()[0])
    runner.forward(inputs)
    current_stream = stream_b
    capturing = True
    runner.forward(inputs)

    assert len(module.run_calls) == 2
    assert module.run_calls[-1][1] is inputs[1]
    assert list(runner._workspace_cache) == [(202, (7, 2048, 1536, 60, 4))]


def test_autotune_profile_keeps_prepared_workspace_for_capture(monkeypatch):
    from flashinfer.autotuner import AutoTuner

    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: (10, 0))
    config = CakeWarpDecodeRunner(_config(), torch.device("cuda:0")).tuning_config
    runner, module = _runner()
    current_stream = SimpleNamespace(cuda_stream=101)
    capturing = False
    monkeypatch.setattr(runner, "_current_stream", lambda: current_stream)
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: capturing)
    inputs = runner.pack_inputs(_activation_pack(), _weight_pack()[0])
    _, batches = AutoTuner()._prepare_input_tensors_with_batches(inputs, config)
    profile_inputs, _ = batches[0]

    runner.forward(profile_inputs, do_preparation=True)
    runner.forward(profile_inputs)
    current_stream = SimpleNamespace(cuda_stream=202)
    capturing = True
    runner.forward(profile_inputs)

    assert profile_inputs[0] is inputs[0]
    assert profile_inputs[1] is inputs[1]
    assert module.run_calls[-1][1] is inputs[1]


def test_capture_pack_and_forward_reuse_warmed_geometry(monkeypatch):
    runner, module = _runner()
    stream_a = SimpleNamespace(cuda_stream=101)
    stream_b = SimpleNamespace(cuda_stream=202)
    current_stream = stream_a
    capturing = False
    monkeypatch.setattr(runner, "_current_stream", lambda: current_stream)
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: capturing)
    activations = _activation_pack()
    weights = _weight_pack()[0]

    warm_inputs = runner.pack_inputs(activations, weights)
    runner.forward(warm_inputs)
    current_stream = stream_b
    capturing = True
    capture_inputs = runner.pack_inputs(activations, weights)
    runner.forward(capture_inputs)

    assert capture_inputs[1] is warm_inputs[1]
    assert len(module.prepare_calls) == 1
    assert len(module.run_calls) == 2
    assert list(runner._workspace_cache) == [(202, (7, 2048, 1536, 60, 4))]


def test_forward_fails_closed_until_workspace_is_prepared():
    runner, module = _runner()
    inputs = runner.pack_inputs(_activation_pack(), _weight_pack()[0])
    runner._prepared_workspaces.clear()
    with pytest.raises(RuntimeError, match="not prepared"):
        runner.forward(inputs)
    runner.forward(inputs, do_preparation=True)
    assert len(module.prepare_calls) == 2
    assert module.run_calls == []


def test_capture_rejects_unprepared_workspace(monkeypatch):
    runner, _ = _runner()
    runner.device = torch.device("cuda:0")
    monkeypatch.setattr(runner, "_is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
        runner._ensure_workspace_prepared(
            torch.empty(64, dtype=torch.uint8), (7, 2048, 1536, 60, 4)
        )


def test_workspace_receipt_does_not_match_a_reused_address(monkeypatch):
    runner, module = _runner()
    geometry = (7, 2048, 1536, 60, 4)
    first = torch.empty(64, dtype=torch.uint8)
    second = torch.empty(64, dtype=torch.uint8)
    monkeypatch.setattr(
        runner,
        "_workspace_identity",
        lambda workspace, shape: (1234, workspace.numel(), shape),
    )

    assert runner._ensure_workspace_prepared(first, geometry) == 1
    assert runner._ensure_workspace_prepared(second, geometry) == 2
    assert len(module.prepare_calls) == 2


def test_workspace_receipt_is_released_with_its_runner():
    runner, module = _runner()
    geometry = (7, 2048, 1536, 60, 4)
    workspace = torch.empty(64, dtype=torch.uint8)

    assert runner._ensure_workspace_prepared(workspace, geometry) == 1
    del workspace
    gc.collect()
    assert module.release_calls == []

    del runner
    gc.collect()

    assert module.release_calls == [1]


def test_workspace_release_rejects_nonpositive_unknown_and_double_receipts():
    module = _Module()
    workspace = torch.empty(64, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="unknown or already released"):
        module.cake_fused_moe_warp_decode_release_workspace(0)
    with pytest.raises(RuntimeError, match="unknown or already released"):
        module.cake_fused_moe_warp_decode_release_workspace(99)
    receipt = module.cake_fused_moe_warp_decode_prepare_workspace(
        workspace, 7, 2048, 1536, 60, 4
    )
    module.cake_fused_moe_warp_decode_release_workspace(receipt)
    with pytest.raises(RuntimeError, match="unknown or already released"):
        module.cake_fused_moe_warp_decode_release_workspace(receipt)

    assert module.live_receipts == set()
    assert module.release_calls == [0, 99, receipt, receipt]


def test_failed_finalizer_release_quarantines_workspace():
    runner, module = _runner()
    geometry = (7, 2048, 1536, 60, 4)
    workspace = torch.empty(64, dtype=torch.uint8)
    workspace_ref = weakref.ref(workspace)
    key = (id(module), 1)

    assert runner._ensure_workspace_prepared(workspace, geometry) == 1
    module.release_error = RuntimeError("injected release failure")
    del workspace
    del runner
    gc.collect()

    try:
        assert module.release_calls == [1]
        assert workspace_ref() is not None
        assert (
            "injected release failure"
            in (moe_runners._CAKE_QUARANTINED_WORKSPACES[key][2])
        )
    finally:
        moe_runners._CAKE_QUARANTINED_WORKSPACES.pop(key, None)


def test_forced_workspace_prepare_retires_previous_finalizer():
    runner, module = _runner()
    geometry = (7, 2048, 1536, 60, 4)
    workspace = torch.empty(64, dtype=torch.uint8)

    assert runner._ensure_workspace_prepared(workspace, geometry) == 1
    assert runner._ensure_workspace_prepared(workspace, geometry, force=True) == 2

    assert module.release_calls == [1]
    assert len(runner._workspace_receipt_finalizers) == 1


def test_forced_workspace_prepare_release_failure_keeps_old_state_quarantined():
    runner, module = _runner()
    geometry = (7, 2048, 1536, 60, 4)
    workspace = torch.empty(64, dtype=torch.uint8)
    identity = runner._workspace_identity(workspace, geometry)
    quarantine_key = (id(module), 1)

    assert runner._ensure_workspace_prepared(workspace, geometry) == 1
    module.release_error = RuntimeError("injected reprepare release failure")
    try:
        with pytest.raises(RuntimeError, match="injected reprepare release failure"):
            runner._ensure_workspace_prepared(workspace, geometry, force=True)

        assert len(module.prepare_calls) == 1
        assert module.release_calls == [1]
        assert module.live_receipts == {1}
        assert runner._prepared_workspaces[identity] == (workspace, 1)
        assert identity not in runner._workspace_receipt_finalizers
        assert moe_runners._CAKE_QUARANTINED_WORKSPACES[quarantine_key][1] is workspace
    finally:
        module.release_error = None
        moe_runners._CAKE_QUARANTINED_WORKSPACES.pop(quarantine_key, None)
        runner._ensure_workspace_prepared(workspace, geometry, force=True)


def test_launch_failure_retires_workspace_and_preserves_launch_error():
    runner, module = _runner()
    inputs = runner.pack_inputs(_activation_pack(), _weight_pack()[0])
    identity = runner._workspace_identity(inputs[1], (7, 2048, 1536, 60, 4))
    module.run_error = RuntimeError("injected launch failure")

    with pytest.raises(RuntimeError, match="injected launch failure"):
        runner.forward(inputs)

    assert module.release_calls == [1]
    assert module.live_receipts == set()
    assert identity not in runner._prepared_workspaces
    assert identity not in runner._workspace_stream_claims
    assert identity not in runner._workspace_receipt_finalizers


def test_launch_and_release_failure_preserves_launch_error_and_quarantines():
    runner, module = _runner()
    inputs = runner.pack_inputs(_activation_pack(), _weight_pack()[0])
    workspace = inputs[1]
    identity = runner._workspace_identity(workspace, (7, 2048, 1536, 60, 4))
    quarantine_key = (id(module), 1)
    module.run_error = RuntimeError("injected launch failure")
    module.release_error = RuntimeError("injected launch cleanup failure")
    try:
        with pytest.raises(RuntimeError, match="injected launch failure"):
            runner.forward(inputs)

        assert module.release_calls == [1]
        assert module.live_receipts == {1}
        assert runner._prepared_workspaces[identity] == (workspace, 1)
        assert runner._workspace_stream_claims[identity][0] is workspace
        assert identity not in runner._workspace_receipt_finalizers
        assert moe_runners._CAKE_QUARANTINED_WORKSPACES[quarantine_key][1] is workspace
    finally:
        module.run_error = None
        module.release_error = None
        moe_runners._CAKE_QUARANTINED_WORKSPACES.pop(quarantine_key, None)
        runner._ensure_workspace_prepared(workspace, identity[2], force=True)


def test_pack_rejects_mode_tokens_weights_and_extra_fields():
    runner, _ = _runner()
    weights, _ = _weight_pack()
    with pytest.raises(NotImplementedError, match="UnpackedPrecomputed"):
        runner.pack_inputs(
            _activation_pack(mode=RoutingInputMode.PackedPrecomputed), weights
        )
    with pytest.raises(ValueError, match="1 <= num_tokens <= 32"):
        runner.pack_inputs(_activation_pack(num_tokens=33), weights)
    with pytest.raises(TypeError, match="topk_weights"):
        runner.pack_inputs(_activation_pack(weights_dtype=torch.float32), weights)

    weights_with_bias, _ = _weight_pack(
        extra={"gemm1_bias": _TensorSpec((60, 3072), torch.bfloat16)}
    )
    with pytest.raises(ValueError, match="does not accept bias"):
        runner.pack_inputs(_activation_pack(), weights_with_bias)


# ---------------------------------------------------------------------------
# GPU numerical parity on exact SM100 / SM103 devices.
#
# Ported from the correctness matrix of benchmarks/cake_warp_decode.py. The
# TRT-LLM NVFP4 routed MoE is the public reference for the SwiGLU and SiTU
# geometries. The standalone SiLU geometry has no public NVFP4 peer, so its rows
# check finiteness and launch-to-launch repeatability only.
# ---------------------------------------------------------------------------

_GPU_MAX_TOKENS = 32
_GPU_SEED = 20260205
_GPU_ATOL = 1e-2
_GPU_RTOL = 1e-2
_GPU_QUANT = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)


@dataclass
class _GpuGeometry:
    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    token_counts: tuple[int, ...]
    activation: object = SwiGLU()

    @property
    def has_public_reference(self) -> bool:
        return isinstance(self.activation, (SwiGLU, SiTU))

    @property
    def uses_activation_params(self) -> bool:
        return self.activation.is_gated and self.activation != SwiGLU()


# Token counts are the schedule / route-packer boundaries of each geometry.
_GPU_GEOMETRIES = (
    _GpuGeometry("e512_i512_k10", 2048, 512, 512, 10, (1, 2, 22, 23, 32)),
    _GpuGeometry("e60_i1536_k4", 2048, 1536, 60, 4, (1, 7, 8, 10, 11, 12, 16, 17, 32)),
    _GpuGeometry("e192_i1536_k4_silu", 6144, 1536, 192, 4, (1, 2, 32), SiLU()),
    _GpuGeometry("e384_i768_k4", 2560, 768, 384, 4, (1, 2, 32)),
    _GpuGeometry("e128_i768_k8", 2048, 768, 128, 8, (1, 2, 8, 9, 10, 16, 32)),
    _GpuGeometry("e128_i1536_k8", 4096, 1536, 128, 8, (1, 2, 32)),
    _GpuGeometry("e256_i512_k8", 2048, 512, 256, 8, (1, 2, 32)),
    _GpuGeometry("e512_i1024_k10", 4096, 1024, 512, 10, (1, 2, 32)),
    _GpuGeometry("e256_i1536_k8", 3072, 1536, 256, 8, (1, 2, 32)),
    _GpuGeometry(
        "e128_i3072_k4_swiglu_oa",
        6144,
        3072,
        128,
        4,
        (1, 2, 32),
        SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
    ),
    _GpuGeometry(
        "e896_i3072_k16_situ",
        3584,
        3072,
        896,
        16,
        (1, 2, 32),
        SiTU(gate_scale=4.0, linear_scale=25.0),
    ),
    # Sharded per-partition slices (fused route packing on every token count).
    _GpuGeometry("h4096_e512_i512_k10", 4096, 512, 512, 10, (1, 2, 16, 32)),
    _GpuGeometry("h4096_e512_i256_k10", 4096, 256, 512, 10, (1, 2, 16, 32)),
    _GpuGeometry("h3072_e256_i768_k8", 3072, 768, 256, 8, (1, 2, 16, 32)),
    _GpuGeometry("h3072_e256_i384_k8", 3072, 384, 256, 8, (1, 2, 16, 32)),
)
_GPU_ROWS = [
    pytest.param(geometry, num_tokens, id=f"{geometry.name}-t{num_tokens:02d}")
    for geometry in _GPU_GEOMETRIES
    for num_tokens in geometry.token_counts
]


def _gpu_target() -> str:
    if not torch.cuda.is_available():
        pytest.skip("Cake warp decode GPU parity requires a CUDA device")
    capability = torch.cuda.get_device_capability(torch.device("cuda"))
    targets = {(10, 0): "sm100a", (10, 3): "sm103a"}
    if capability not in targets:
        pytest.skip(
            "Cake warp decode GPU parity requires exact SM100 or SM103, "
            f"got SM{capability[0]}{capability[1]}"
        )
    return targets[capability]


def _gpu_routing(geometry: _GpuGeometry, device: torch.device):
    tokens = torch.arange(_GPU_MAX_TOKENS, device=device, dtype=torch.int64)[:, None]
    ranks = torch.arange(geometry.top_k, device=device, dtype=torch.int64)[None, :]
    ids = ((tokens * 17 + ranks * 29) % (geometry.num_experts - 1) + 1).to(torch.int32)
    ids[:, 0] = 0
    raw_weights = (geometry.top_k + 1 - ranks).expand(_GPU_MAX_TOKENS, -1).float()
    raw_weights = raw_weights + (tokens % 5).float() * 0.03125
    weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    return ids.contiguous(), weights.contiguous()


@functools.lru_cache(maxsize=1)
def _gpu_fixture(geometry_name: str) -> SimpleNamespace:
    """Quantize one geometry's activations and expert weights with the public API.

    One geometry is cached at a time: the largest geometry holds tens of GB of
    expert weights, and the parity rows are ordered geometry-major.
    """
    index, geometry = next(
        (index, geometry)
        for index, geometry in enumerate(_GPU_GEOMETRIES)
        if geometry.name == geometry_name
    )
    device = torch.device("cuda")
    torch.manual_seed(_GPU_SEED + index)
    hidden = torch.randn(
        _GPU_MAX_TOKENS, geometry.hidden_size, device=device, dtype=torch.bfloat16
    )
    gate_rows = 2 if geometry.activation.is_gated else 1
    w1 = torch.randn(
        geometry.num_experts,
        geometry.intermediate_size * gate_rows,
        geometry.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w1 *= 0.02
    w2 = torch.randn(
        geometry.num_experts,
        geometry.hidden_size,
        geometry.intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w2 *= 0.02
    hidden_q, hidden_scale = TrtllmFp4Config.prepare_activations(
        hidden, quant=_GPU_QUANT
    )
    weight_view = CakeWarpDecodeConfig.prepare_weights(
        w1,
        w2,
        quant=_GPU_QUANT,
        num_local_experts=geometry.num_experts,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        activation=geometry.activation,
        device=device,
    )
    del w1, w2
    topk_ids, topk_weights = _gpu_routing(geometry, device)
    return SimpleNamespace(
        geometry=geometry,
        hidden_states_q=hidden_q,
        hidden_states_scale=hidden_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        weight_view=weight_view,
    )


def _gpu_reference(fixture: SimpleNamespace, num_tokens: int) -> torch.Tensor:
    """Run the public TRT-LLM NVFP4 routed MoE on the same physical tensors."""
    geometry = fixture.geometry
    view = fixture.weight_view
    device = fixture.hidden_states_q.device
    official_beta = view.get("gemm1_beta")
    official_clamp_limit = view.get("gemm1_clamp_limit")
    if geometry.name == "e128_i3072_k4_swiglu_oa":
        # The public baseline consumes beta and clamp in raw-accumulator units.
        official_beta = view["gemm1_beta"] / view["output1_scale_gate_scalar"]
        official_clamp_limit = (
            view["gemm1_clamp_limit"] / view["output1_scale_gate_scalar"]
        )
    output = torch.empty(
        num_tokens, geometry.hidden_size, dtype=torch.bfloat16, device=device
    )
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(fixture.topk_ids[:num_tokens], fixture.topk_weights[:num_tokens]),
        routing_bias=None,
        hidden_states=fixture.hidden_states_q[:num_tokens],
        hidden_states_scale=fixture.hidden_states_scale[:num_tokens],
        gemm1_weights=view["gemm1_weights"],
        gemm1_weights_scale=view["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=view.get("gemm1_alpha"),
        gemm1_beta=official_beta,
        gemm1_clamp_limit=official_clamp_limit,
        gemm2_weights=view["gemm2_weights"],
        gemm2_weights_scale=view["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=view["output1_scale_scalar"],
        output1_scale_gate_scalar=view["output1_scale_gate_scalar"],
        output2_scale_scalar=view["output2_scale_scalar"],
        num_experts=geometry.num_experts,
        top_k=geometry.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=geometry.intermediate_size,
        local_expert_offset=0,
        local_num_experts=geometry.num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value,
        do_finalize=True,
        enable_pdl=True,
        activation_type=geometry.activation.type.value,
        per_token_scale=None,
        output=output,
        tune_max_num_tokens=_GPU_MAX_TOKENS,
    )
    if isinstance(result, (list, tuple)):
        result = result[0]
    assert result.data_ptr() == output.data_ptr()
    return output


def _gpu_run_cake(
    module,
    fixture: SimpleNamespace,
    num_tokens: int,
    output: torch.Tensor,
    workspace: torch.Tensor,
    receipt: int,
) -> torch.Tensor:
    geometry = fixture.geometry
    view = fixture.weight_view
    inputs = (
        output,
        workspace,
        fixture.hidden_states_q[:num_tokens],
        fixture.hidden_states_scale[:num_tokens],
        fixture.topk_ids[:num_tokens],
        fixture.topk_weights[:num_tokens],
        view["gemm1_weights"],
        view["gemm1_weights_scale"],
        view["gemm2_weights"],
        view["gemm2_weights_scale"],
        view["output1_scale_scalar"],
        view["output1_scale_gate_scalar"],
        view["output2_scale_scalar"],
    )
    if geometry.uses_activation_params:
        module.cake_fused_moe_warp_decode_with_activation_params(
            *inputs,
            view.get("gemm1_alpha"),
            view.get("gemm1_beta"),
            view.get("gemm1_clamp_limit"),
            receipt,
            True,
        )
    else:
        module.cake_fused_moe_warp_decode(*inputs, receipt, True)
    return output


@pytest.mark.parametrize("geometry,num_tokens", _GPU_ROWS)
def test_gpu_output_matches_public_nvfp4_reference(geometry, num_tokens):
    target = _gpu_target()
    device = torch.device("cuda")
    fixture = _gpu_fixture(geometry.name)
    module = cake_warp_decode_jit.get_cake_fused_moe_warp_decode_module(
        target, device=device
    )
    shape = (
        num_tokens,
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.num_experts,
        geometry.top_k,
    )
    workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
    assert workspace_size > 0
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    output = torch.empty(
        num_tokens, geometry.hidden_size, dtype=torch.bfloat16, device=device
    )
    receipt = int(
        module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
    )
    assert receipt > 0
    try:
        first = _gpu_run_cake(
            module, fixture, num_tokens, output, workspace, receipt
        ).clone()
        output.fill_(float("nan"))
        second = _gpu_run_cake(
            module, fixture, num_tokens, output, workspace, receipt
        ).clone()
        torch.cuda.synchronize(device)
    finally:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)

    assert torch.isfinite(first.float()).all()
    assert torch.equal(second, first), "repeated launch changed the output"
    if geometry.has_public_reference:
        expected = _gpu_reference(fixture, num_tokens)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(first, expected, atol=_GPU_ATOL, rtol=_GPU_RTOL)
