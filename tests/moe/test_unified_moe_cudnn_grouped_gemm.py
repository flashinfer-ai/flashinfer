"""Unified cuDNN grouped-GEMM MoE adapter tests for the BF16, FP8, MXFP8 and NVFP4 backends."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    GELU,
    BackendOptions,
    CudnnGroupedGemmBf16Config,
    CudnnGroupedGemmBf16Runner,
    CudnnGroupedGemmFp8PerTensorConfig,
    CudnnGroupedGemmFp8PerTensorRunner,
    CudnnGroupedGemmMxfp8Config,
    CudnnGroupedGemmMxfp8Runner,
    CudnnGroupedGemmNvfp4Config,
    CudnnGroupedGemmNvfp4Runner,
    ExecutionConfig,
    ExpertConfig,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    ReLU,
    ReLU2,
    RoutingConfig,
    RoutingInputMode,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.fused_moe.api import (
    _CUDNN_GROUPED_GEMM_BF16_ARCHS,
    _CUDNN_GROUPED_GEMM_FP4_ARCHS,
    _CUDNN_GROUPED_GEMM_FP8_ARCHS,
    _CUDNN_GROUPED_GEMM_MXFP8_ARCHS,
    ALL_BACKEND_CONFIGS,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe.runners import _MOE_UTILS_ARCHS, MoERunner
from flashinfer.grouped_mm.core import (
    _check_grouped_mm_bf16,
    _check_grouped_mm_fp4,
    _check_grouped_mm_fp8,
    _check_grouped_mm_mxfp8,
)
from flashinfer.grouped_mm.cudnn import _CUDNN_MOE_MIN_VERSION
from flashinfer.quantization.fp4_quantization import e2m1_and_ufp8sf_scale_to_float
from flashinfer.quantization.fp8_quantization import mxfp8_dequantize_host
from flashinfer.utils import get_compute_capability

from .utils import compute_reference_moe

# The gated activations backed by fused flashinfer.activation kernels.
_ACTIVATIONS = (SwiGLU(), GeGLU(), GeGLUTanh())
# Everything else the unified API can express is rejected by check_support.
_UNSUPPORTED_ACTIVATIONS = (
    SwiGLU(alpha=1.7, beta=0.25, limit=6.0),
    SwiGLUStep(),
    SiTU(),
    ReLU2(),
    Identity(),
    GELU(),
    ReLU(),
    SiLU(),
)
_BF16_KEY = "cudnn_grouped_gemm_bf16"
_FP8_KEY = "cudnn_grouped_gemm_fp8_per_tensor"
_MXFP8_KEY = "cudnn_grouped_gemm_mxfp8"
_NVFP4_KEY = "cudnn_grouped_gemm_nvfp4"
_BLOCK_SCALE_KEYS = (_MXFP8_KEY, _NVFP4_KEY)
_BF16_TOL = dict(rtol=2e-2, atol=2e-2)


def _fp8_tol(expected):
    """FP8 tolerance: 10% relative plus 10% of the reference's largest magnitude.

    Two E4M3 quantizations (activations and the dynamically scaled intermediate)
    leave single-element outliers that a fixed absolute tolerance cannot bound.
    """
    return dict(rtol=1e-1, atol=1e-1 * expected.abs().max().item())


def _mxfp8_tol(expected):
    return dict(rtol=1e-1, atol=1e-1 * expected.abs().max().item())


def _nvfp4_tol(expected):
    """NVFP4's 1-mantissa-bit values need a wider envelope than MXFP8."""
    return dict(rtol=1e-1, atol=2.5e-1 * expected.abs().max().item())


@dataclasses.dataclass(frozen=True)
class _Family:
    """One cuDNN grouped-GEMM backend: its config, quant pair, archs and tolerance."""

    key: str
    config_cls: type
    quant: QuantConfig
    archs: tuple[int, ...]
    activations: tuple
    size_multiple: int  # hidden_size and intermediate_size alignment of the view
    tol: Callable[[torch.Tensor], dict]


_FAMILIES = {
    _BF16_KEY: _Family(
        _BF16_KEY,
        CudnnGroupedGemmBf16Config,
        QuantConfig(),
        _CUDNN_GROUPED_GEMM_BF16_ARCHS,
        _ACTIVATIONS,
        16,
        lambda expected: _BF16_TOL,
    ),
    _FP8_KEY: _Family(
        _FP8_KEY,
        CudnnGroupedGemmFp8PerTensorConfig,
        QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
        _CUDNN_GROUPED_GEMM_FP8_ARCHS,
        (SwiGLU(),),  # the fused requantizing activation kernel is SwiGLU only
        16,
        _fp8_tol,
    ),
    _MXFP8_KEY: _Family(
        _MXFP8_KEY,
        CudnnGroupedGemmMxfp8Config,
        QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8),
        _CUDNN_GROUPED_GEMM_MXFP8_ARCHS,
        _ACTIVATIONS,
        128,
        _mxfp8_tol,
    ),
    _NVFP4_KEY: _Family(
        _NVFP4_KEY,
        CudnnGroupedGemmNvfp4Config,
        QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        _CUDNN_GROUPED_GEMM_FP4_ARCHS,
        _ACTIVATIONS,
        128,
        _nvfp4_tol,
    ),
}
_FAMILY_KEYS = tuple(_FAMILIES)
_RUNNERS = {
    _BF16_KEY: CudnnGroupedGemmBf16Runner,
    _FP8_KEY: CudnnGroupedGemmFp8PerTensorRunner,
    _MXFP8_KEY: CudnnGroupedGemmMxfp8Runner,
    _NVFP4_KEY: CudnnGroupedGemmNvfp4Runner,
}


def _cudnn_moe_available() -> bool:
    try:
        import cudnn
    except (ImportError, OSError):
        return False
    return cudnn.backend_version() >= _CUDNN_MOE_MIN_VERSION and hasattr(
        cudnn, "moe_grouped_matmul_mode"
    )


def _device_arch() -> int:
    if not torch.cuda.is_available():
        return 0
    major, minor = get_compute_capability(torch.device("cuda"))
    return major * 10 + minor


# cuDNN below 9.22 has no SM120 / SM121 engine for the block-scaled MoE grouped
# GEMM although the flat ``grouped_mm_mxfp8`` / ``grouped_mm_fp4`` list those
# architectures; the block-scaled backends are skipped there until the flat API
# states the per-architecture minimum.
_SM12X_BLOCK_SCALE_MIN_CUDNN = 92200
_SM12X_BLOCK_SCALE_SKIP = (
    "cuDNN < 9.22 has no SM120 / SM121 engine for the block-scaled MoE grouped GEMM"
)


def _sm12x_block_scale_unsupported() -> bool:
    if not _cudnn_moe_available() or _device_arch() not in (120, 121):
        return False
    import cudnn

    return cudnn.backend_version() < _SM12X_BLOCK_SCALE_MIN_CUDNN


def _cudnn_backend_is_supported(key: str) -> bool:
    if not (_cudnn_moe_available() and _device_arch() in _FAMILIES[key].archs):
        return False
    return key not in _BLOCK_SCALE_KEYS or not _sm12x_block_scale_unsupported()


cudnn_bf16_required = pytest.mark.skipif(
    not _cudnn_backend_is_supported(_BF16_KEY),
    reason="requires cuDNN >= 9.21 with moe_grouped_matmul on a grouped_mm_bf16 arch",
)
cudnn_fp8_required = pytest.mark.skipif(
    not _cudnn_backend_is_supported(_FP8_KEY),
    reason="requires cuDNN >= 9.21 with moe_grouped_matmul on a grouped_mm_fp8 arch",
)
cudnn_mxfp8_required = pytest.mark.skipif(
    not _cudnn_backend_is_supported(_MXFP8_KEY),
    reason=(
        _SM12X_BLOCK_SCALE_SKIP
        if _sm12x_block_scale_unsupported()
        else "requires cuDNN >= 9.21 with moe_grouped_matmul on a grouped_mm_mxfp8 arch"
    ),
)
cudnn_nvfp4_required = pytest.mark.skipif(
    not _cudnn_backend_is_supported(_NVFP4_KEY),
    reason=(
        _SM12X_BLOCK_SCALE_SKIP
        if _sm12x_block_scale_unsupported()
        else "requires cuDNN >= 9.21 with moe_grouped_matmul on a grouped_mm_fp4 arch"
    ),
)
moe_utils_required = pytest.mark.skipif(
    _device_arch() not in _MOE_UTILS_ARCHS,
    reason="the moe_utils kernels sort, permute and finalize on SM90/SM100/SM103",
)
_REQUIRED = {
    _BF16_KEY: cudnn_bf16_required,
    _FP8_KEY: cudnn_fp8_required,
    _MXFP8_KEY: cudnn_mxfp8_required,
    _NVFP4_KEY: cudnn_nvfp4_required,
}
_FAMILY_PARAMS = tuple(pytest.param(key, marks=_REQUIRED[key]) for key in _FAMILY_KEYS)
_PLAIN_FAMILY_PARAMS = _FAMILY_PARAMS[:2]  # BF16, FP8: 16-row tiles
_BLOCK_SCALE_FAMILY_PARAMS = _FAMILY_PARAMS[2:]  # MXFP8, NVFP4: 128-row tiles
_DYNAMIC_SCALE_FAMILY_PARAMS = (_FAMILY_PARAMS[1], _FAMILY_PARAMS[3])  # FP8, NVFP4
# (family, activation) for every activation the family runs.
_FAMILY_ACTIVATION_PARAMS = tuple(
    pytest.param(key, activation, marks=_REQUIRED[key])
    for key in _FAMILY_KEYS
    for activation in _FAMILIES[key].activations
)
_PERMUTE_PATHS = (
    pytest.param(True, marks=moe_utils_required, id="moe_utils"),
    pytest.param(False, id="torch"),
)


def _config(
    key: str, *, num_experts: int = 4, top_k: int = 2, **overrides
) -> MoEConfig:
    family = _FAMILIES[key]
    values = dict(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=family.quant,
        experts=ExpertConfig(intermediate_size=max(64, family.size_multiple)),
        activation=SwiGLU(),
        backend=BackendOptions((family.config_cls(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=64),
        finalize=MoEFinalizeConfig(do_finalize=False),
    )
    values.update(overrides)
    return MoEConfig(**values)


def _swap_halves(w1: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    """``[up, gate]`` <-> ``[gate, up]`` fc1 row order (the views store ``[gate, up]``)."""
    return torch.cat((w1[:, intermediate_size:], w1[:, :intermediate_size]), dim=1)


def _experts(num_experts, hidden_size, intermediate_size, activation, device):
    gemm1_rows = intermediate_size * (2 if activation.is_gated else 1)
    w1 = (
        torch.randn(
            num_experts, gemm1_rows, hidden_size, dtype=torch.bfloat16, device=device
        )
        / hidden_size**0.5
    )
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
    return w1, w2


def _routing(num_tokens, num_experts, top_k, device):
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1)
    return topk_ids, topk_weights


def _topk_ids_with_counts(counts, device):
    """``[sum(counts), 1]`` int32 assignments giving expert ``e`` exactly ``counts[e]`` tokens."""
    ids = torch.cat(
        [torch.full((n,), e, dtype=torch.int32) for e, n in enumerate(counts)]
    )
    return ids[torch.randperm(ids.numel())].view(-1, 1).to(device)


# ---------------------------------------------------------------------------
# CPU: registration, capabilities and preparation contracts
# ---------------------------------------------------------------------------


def test_cudnn_config_registration_and_architecture_lists():
    for key, flat_check in (
        (_BF16_KEY, _check_grouped_mm_bf16),
        (_FP8_KEY, _check_grouped_mm_fp8),
        (_MXFP8_KEY, _check_grouped_mm_mxfp8),
        (_NVFP4_KEY, _check_grouped_mm_fp4),
    ):
        family, runner_cls = _FAMILIES[key], _RUNNERS[key]
        assert family.config_cls in ALL_BACKEND_CONFIGS
        assert _BACKEND_RUNNERS[family.config_cls] is runner_cls
        assert issubclass(runner_cls, MoERunner)
        assert runner_cls.backend_key == key
        # The unified arch lists mirror the flat grouped_mm capability lists.
        assert set(family.archs) == set(flat_check._supported_ccs)
        for arch in family.archs:
            assert family.config_cls.supported(arch)
        assert not family.config_cls.supported(75)
        assert not family.config_cls.supported(130)
    assert CudnnGroupedGemmBf16Config.supported(80)
    assert not CudnnGroupedGemmFp8PerTensorConfig.supported(80)  # FP8 starts at SM89
    # Explicit opt-in only: never part of the default candidate list.
    default = MoEConfig(
        routing=RoutingConfig(num_experts=4, top_k=2),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=64),
    )
    cudnn_configs = tuple(f.config_cls for f in _FAMILIES.values())
    assert not any(isinstance(c, cudnn_configs) for c in default.backend)
    for family in _FAMILIES.values():
        cfg = family.config_cls()
        assert eval(repr(cfg)) == cfg and hash(cfg) == hash(family.config_cls())
        assert _config(family.key).backend.valid_for(100) == [cfg]


def test_cudnn_runner_capability_declarations():
    for key, family in _FAMILIES.items():
        runner_cls = _RUNNERS[key]
        assert runner_cls.supported_activation_classes == tuple(
            type(a) for a in family.activations
        )
        assert runner_cls.supported_routing_modes == (
            RoutingInputMode.PackedPrecomputed,
            RoutingInputMode.UnpackedPrecomputed,
        )
        assert runner_cls.supports_expert_parallelism
        assert not runner_cls.supports_fused_shared_experts
        assert runner_cls.supported_quant_variants == (family.quant.pair,)
        assert runner_cls.supports_quant(family.quant)
        for other in _FAMILIES.values():
            if other is not family:
                assert not runner_cls.supports_quant(other.quant)


def _detached_runner(key: str, config: MoEConfig, *, use_moe_utils: bool = True):
    """A runner object for ``_check_support`` without a device."""
    runner_cls = _RUNNERS[key]
    runner = runner_cls.__new__(runner_cls)
    runner.config = config
    runner._device_arch = 100
    runner._use_moe_utils = use_moe_utils
    return runner


def test_cudnn_check_support_rejects_unsupported_options():
    if _cudnn_moe_available():
        # Finalize runs on every arch: torch ops where the moe_utils kernels are missing.
        for key in _FAMILY_KEYS:
            _detached_runner(
                key, _config(key, finalize=MoEFinalizeConfig()), use_moe_utils=False
            )._check_support()
    with pytest.raises(NotImplementedError, match="does not support PDL"):
        _detached_runner(
            _BF16_KEY,
            _config(
                _BF16_KEY,
                execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=8),
            ),
        )._check_support()
    for key in _BLOCK_SCALE_KEYS:
        with pytest.raises(NotImplementedError, match="intermediate_size divisible"):
            _detached_runner(
                key, _config(key, experts=ExpertConfig(intermediate_size=96))
            )._check_support()
    for activation in _UNSUPPORTED_ACTIVATIONS:
        with pytest.raises(NotImplementedError, match="SwiGLU|does not support"):
            _detached_runner(
                _BF16_KEY, _config(_BF16_KEY, activation=activation)
            )._check_support()
    for activation in (GeGLU(), GeGLUTanh()):
        with pytest.raises(NotImplementedError, match="does not support"):
            _detached_runner(
                _FP8_KEY, _config(_FP8_KEY, activation=activation)
            )._check_support()
    with pytest.raises(
        NotImplementedError, match="does not support weight=FP8PerTensor"
    ):
        _detached_runner(
            _BF16_KEY, _config(_BF16_KEY, quant=_FAMILIES[_FP8_KEY].quant)
        )._check_support()
    with pytest.raises(NotImplementedError, match="does not support weight=BF16"):
        _detached_runner(
            _FP8_KEY, _config(_FP8_KEY, quant=QuantConfig())
        )._check_support()
    for key, arch in (
        (_BF16_KEY, 75),
        (_FP8_KEY, 80),
        (_MXFP8_KEY, 90),
        (_NVFP4_KEY, 89),
    ):
        runner = _detached_runner(key, _config(key))
        runner._device_arch = arch
        with pytest.raises(RuntimeError, match=f"does not support SM{arch}"):
            runner._check_support()


@pytest.mark.parametrize("key", _FAMILY_KEYS)
def test_cudnn_prepare_weights_rejects_invalid_inputs(key):
    family = _FAMILIES[key]
    hidden_size = intermediate_size = family.size_multiple
    kwargs = dict(
        num_local_experts=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    w1 = torch.randn(2, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16)
    w2 = torch.randn(2, hidden_size, intermediate_size, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="expects BF16 weights"):
        family.config_cls.prepare_weights(w1.half(), w2, **kwargs)
    with pytest.raises(ValueError, match="weight shapes"):
        family.config_cls.prepare_weights(w1[:, :-1], w2, **kwargs)
    with pytest.raises(ValueError, match="weight shapes"):
        family.config_cls.prepare_weights(w1, w2, activation=ReLU2(), **kwargs)
    with pytest.raises(ValueError, match="3D"):
        family.config_cls.prepare_weights(w1[0], w2, **kwargs)
    if key in _BLOCK_SCALE_KEYS:
        for sizes in ((128, 96), (96, 128)):
            with pytest.raises(ValueError, match="divisible by 128"):
                family.config_cls.prepare_weights(
                    torch.randn(2, 2 * sizes[1], sizes[0], dtype=torch.bfloat16),
                    torch.randn(2, sizes[0], sizes[1], dtype=torch.bfloat16),
                    num_local_experts=2,
                    hidden_size=sizes[0],
                    intermediate_size=sizes[1],
                )


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _dequant_experts(family: _Family, view, prefix: str) -> torch.Tensor:
    """Dequantize a weight view entry back to BF16 ``[E_local, n, k]``."""
    weights = view[f"{prefix}_expert_weights"]
    if family.key == _BF16_KEY:
        return weights
    if family.key == _FP8_KEY:
        return (weights.float() * view[f"{prefix}_dequant"][:, None, None]).to(
            torch.bfloat16
        )
    scale = view[f"{prefix}_weight_scale"]
    if family.key == _MXFP8_KEY:
        experts = [
            mxfp8_dequantize_host(
                weights[e].cpu().view(torch.uint8),
                scale[e].cpu().view(torch.uint8).reshape(-1),
                True,
            )
            for e in range(weights.shape[0])
        ]
    else:
        rows, packed_cols = weights.shape[1], weights.shape[2]
        dequant = view[f"{prefix}_dequant"]
        experts = [
            e2m1_and_ufp8sf_scale_to_float(
                weights[e].cpu(),
                scale[e].cpu().view(torch.uint8).reshape(-1),
                torch.ones(1, dtype=torch.float32),
                16,
                1,
                True,
            ).view(rows, packed_cols * 2)
            * dequant[e].cpu()
            for e in range(weights.shape[0])
        ]
    return torch.stack(experts).to(device=weights.device, dtype=torch.bfloat16)


def _dequant_activations(family: _Family, q, scale) -> torch.Tensor:
    """Dequantize an activation pack back to BF16 ``[M, H]``."""
    if family.key == _BF16_KEY:
        return q
    if family.key == _FP8_KEY:
        return (q.float() * scale).to(torch.bfloat16)
    if family.key == _MXFP8_KEY:
        values = mxfp8_dequantize_host(
            q.cpu().view(torch.uint8), scale.cpu().view(torch.uint8).reshape(-1), False
        )
    else:
        values = e2m1_and_ufp8sf_scale_to_float(
            q.cpu(),
            scale.cpu().view(torch.uint8).reshape(-1),
            torch.ones(1, dtype=torch.float32),
            16,
            1,
            False,
        ).view(q.shape[0], q.shape[1] * 2)
    return values.to(device=q.device, dtype=torch.bfloat16)


@dataclasses.dataclass
class _Case:
    """A layer problem with the pack the runner reads and the reference it must match."""

    family: _Family
    config: MoEConfig
    act: MoEActivationPack
    weights: MoEWeightPack
    expected: torch.Tensor  # finalized [T, H] BF16 reference
    tol: dict
    w1_ref: torch.Tensor  # canonical [up, gate] BF16 weights the reference consumed
    w2_ref: torch.Tensor
    local: tuple[int, int]  # (local_expert_offset, local_num_experts)

    @property
    def num_tokens(self) -> int:
        return self.act.topk_ids.shape[0]

    @property
    def top_k(self) -> int:
        return self.act.topk_ids.shape[1]


def _make_case(
    family: _Family,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    *,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    hidden_states: Optional[torch.Tensor] = None,
    experts: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    local: Optional[tuple[int, int]] = None,
    shared: Optional[_Case] = None,
) -> _Case:
    """Random inputs for one family; ``shared`` reuses another case's weights.

    ``local=(offset, count)`` prepares only that expert shard and masks the
    other assignments in the reference. Quantized families feed the runner
    the pre-quantized pack and the reference the dequantized tensors.
    """
    device = torch.device("cuda")
    activation = SwiGLU() if activation is None else activation
    offset, count = local or (0, num_experts)
    if hidden_states is None:
        hidden_states = (
            torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
            / 2
        )
    routed_ids, routed_weights = _routing(num_tokens, num_experts, top_k, device)
    topk_ids = routed_ids if topk_ids is None else topk_ids
    topk_weights = routed_weights if topk_weights is None else topk_weights
    if shared is None:
        w1, w2 = experts or _experts(
            num_experts, hidden_size, intermediate_size, activation, device
        )
        view = family.config_cls.prepare_weights(
            w1[offset : offset + count],
            w2[offset : offset + count],
            num_local_experts=count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
        )
        weights = MoEWeightPack()
        weights.prepare_for(family.key, view)
        # The reference sees what the runner sees: the dequantized local shard
        # (non-local experts carry zero routing weight).
        w1_ref, w2_ref = w1.clone(), w2.clone()
        fc1 = _dequant_experts(family, view, "fc1")
        w1_ref[offset : offset + count] = (
            _swap_halves(fc1, intermediate_size) if activation.is_gated else fc1
        )
        w2_ref[offset : offset + count] = _dequant_experts(family, view, "fc2")
    else:
        weights, w1_ref, w2_ref = shared.weights, shared.w1_ref, shared.w2_ref
    if family.key == _BF16_KEY:
        act = MoEActivationPack(hidden_states, None, topk_ids, topk_weights)
    else:
        q, scale = family.config_cls.prepare_activations(hidden_states)
        act = MoEActivationPack(q, scale, topk_ids, topk_weights)
    is_local = ((topk_ids >= offset) & (topk_ids < offset + count)).to(
        topk_weights.dtype
    )
    expected = compute_reference_moe(
        _dequant_activations(family, act.hidden_states_q, act.hidden_states_scale),
        topk_ids,
        topk_weights * is_local,
        w1_ref,
        w2_ref,
        activation,
    )
    config = _config(
        family.key,
        num_experts=num_experts,
        top_k=top_k,
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=offset,
            local_num_experts=count,
        ),
        activation=activation,
        execution=ExecutionConfig(
            enable_pdl=False, tune_max_num_tokens=max(64, num_tokens)
        ),
    )
    return _Case(
        family,
        config,
        act,
        weights,
        expected,
        family.tol(expected),
        w1_ref,
        w2_ref,
        (offset, count),
    )


def _combine(result, num_tokens: int, top_k: int) -> torch.Tensor:
    """Caller-side finalize of the unfinalized triple: weighted sum of each token's rows."""
    gemm2_out, expert_weights, token_to_row = result
    rows = token_to_row.to(torch.int64).view(num_tokens, top_k)
    gathered = gemm2_out[rows.clamp(min=0)].float()
    weights = expert_weights.float() * (rows >= 0)
    return (gathered * weights[..., None]).sum(1).to(torch.bfloat16)


def _assert_matches_reference(actual: torch.Tensor, case: _Case) -> None:
    """``assert_close`` against the reference, with a non-vacuous tolerance.

    The guard is exception-free on purpose: a caught exception would pin the
    caller frames (and any CUDA graph they hold) in a traceback cycle, and the
    cyclic collector may then destroy that graph during a later capture.
    """
    expected, tol = case.expected, case.tol
    assert not torch.allclose(torch.zeros_like(expected), expected, **tol), (
        "tolerance is vacuous for this reference"
    )
    torch.testing.assert_close(actual, expected, **tol)
    cosine = F.cosine_similarity(
        actual.float().reshape(-1), expected.float().reshape(-1), dim=0
    )
    assert cosine.item() > 0.98, f"cosine similarity {cosine.item():.4f}"


def _layer(
    case: _Case, *, finalize: bool = False, use_moe_utils: Optional[bool] = None
):
    config = case.config
    if finalize:
        config = dataclasses.replace(
            config, finalize=MoEFinalizeConfig(do_finalize=True)
        )
    layer = MoELayer(config, torch.device("cuda"))
    assert [r.backend_key for r in layer.runners] == [case.family.key]
    if use_moe_utils is not None:
        layer.runners[0]._use_moe_utils = use_moe_utils
    return layer


def _forward(case: _Case, **layer_kwargs):
    """Run the layer on the case and return its output."""
    return _layer(case, **layer_kwargs)(case.act, case.weights)


def _check_unfinalized(result, case: _Case) -> torch.Tensor:
    """Validate the unfinalized triple's contract; return the caller-side combine."""
    assert isinstance(result, list) and len(result) == 3
    gemm2_out, expert_weights, token_to_row = result
    num_tokens, top_k = case.num_tokens, case.top_k
    hidden_size = case.w2_ref.shape[1]
    assert gemm2_out.dtype is torch.bfloat16 and gemm2_out.shape[1] == hidden_size
    assert token_to_row.dtype is torch.int32 and token_to_row.shape == (
        num_tokens * top_k,
    )
    assert expert_weights.shape == (num_tokens, top_k)
    offset, count = case.local
    is_local = (case.act.topk_ids >= offset) & (case.act.topk_ids < offset + count)
    rows = token_to_row.to(torch.int64)
    assert torch.equal(rows < 0, ~is_local.reshape(-1))
    local_rows = rows[rows >= 0]
    assert local_rows.unique().numel() == local_rows.numel()  # one row per assignment
    assert local_rows.numel() == 0 or local_rows.max().item() < gemm2_out.shape[0]
    return _combine(result, num_tokens, top_k)


# --- reference matrix ---------------------------------------------------------


# (num_tokens, num_experts, top_k, hidden_size, intermediate_size); sizes are
# multiples of 128 so the block-scaled families run them too.
_SHAPES = (
    (1, 2, 1, 128, 128),  # one token
    (7, 5, 3, 384, 128),  # odd counts, top_k does not divide the experts
    (33, 16, 4, 256, 512),
    (64, 1, 1, 256, 256),  # one expert
    (200, 4, 2, 128, 256),
    (1024, 32, 8, 512, 256),
    (1536, 16, 4, 512, 384),
)
# Sizes off the 128-row tile: BF16 and FP8 only (16-byte aligned rows).
_PLAIN_SHAPES = (
    (5, 3, 2, 64, 64),
    (48, 8, 2, 320, 192),
    (300, 6, 3, 208, 96),
)


def _shape_id(shape) -> str:
    return "T{}_E{}_k{}_H{}_I{}".format(*shape)


@pytest.mark.parametrize("key, activation", _FAMILY_ACTIVATION_PARAMS)
@pytest.mark.parametrize("shape", _SHAPES, ids=_shape_id)
def test_cudnn_problem_sizes(key, activation, shape):
    """Every family and fused activation over the shape matrix, against the reference."""
    family = _FAMILIES[key]
    torch.manual_seed(hash(shape) % 2**31)
    case = _make_case(family, *shape, activation)
    _assert_matches_reference(_check_unfinalized(_forward(case), case), case)


@pytest.mark.parametrize("key", _PLAIN_FAMILY_PARAMS)
@pytest.mark.parametrize("shape", _PLAIN_SHAPES, ids=_shape_id)
def test_cudnn_problem_sizes_off_the_block_scale_tile(key, shape):
    family = _FAMILIES[key]
    torch.manual_seed(hash(shape) % 2**31)
    case = _make_case(family, *shape)
    _assert_matches_reference(_check_unfinalized(_forward(case), case), case)


# Expert load patterns, as token counts per expert (top_k = 1).
_DISTRIBUTIONS = {
    "non_uniform": (64, 32, 128, 96),
    "empty_experts": (64, 0, 128, 0),
    "single_expert": (256,),
    "one_hot_expert": (0, 0, 320, 0),
    "one_token_each": (1, 1, 1, 1, 1, 1, 1, 1),
}


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
@pytest.mark.parametrize("distribution", sorted(_DISTRIBUTIONS))
def test_cudnn_expert_load_distributions(key, distribution):
    """Imbalanced, empty, one-hot and one-token expert segments."""
    family = _FAMILIES[key]
    torch.manual_seed(2)
    counts = _DISTRIBUTIONS[distribution]
    case = _make_case(
        family,
        sum(counts),
        len(counts),
        1,
        256,
        256,
        topk_ids=_topk_ids_with_counts(counts, torch.device("cuda")),
    )
    _assert_matches_reference(_check_unfinalized(_forward(case), case), case)


_ROUTING_WEIGHTS = {
    # mode, dtype handed to the runner, dtype it hands back
    "packed_fp32": (RoutingInputMode.PackedPrecomputed, torch.float32, torch.bfloat16),
    "unpacked_bf16": (
        RoutingInputMode.UnpackedPrecomputed,
        torch.bfloat16,
        torch.bfloat16,
    ),
    "unpacked_fp32": (
        RoutingInputMode.UnpackedPrecomputed,
        torch.float32,
        torch.float32,
    ),
}


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
@pytest.mark.parametrize("mode", sorted(_ROUTING_WEIGHTS))
def test_cudnn_routing_weight_modes(key, mode):
    """Packed routing narrows the weights to BF16; unpacked returns them as given."""
    family = _FAMILIES[key]
    routing_mode, in_dtype, out_dtype = _ROUTING_WEIGHTS[mode]
    torch.manual_seed(5)
    device = torch.device("cuda")
    topk_ids, topk_weights = _routing(12, 8, 3, device)
    # The reference consumes exactly the values the runner receives.
    exact = topk_weights.to(in_dtype).float()
    case = _make_case(family, 12, 8, 3, 256, 128, topk_ids=topk_ids, topk_weights=exact)
    act = MoEActivationPack(
        case.act.hidden_states_q,
        case.act.hidden_states_scale,
        topk_ids,
        exact.to(in_dtype),
        routing_input_mode=routing_mode,
    )
    result = _layer(case)(act, case.weights)
    assert result[1].dtype is out_dtype
    torch.testing.assert_close(result[1], exact.to(out_dtype), rtol=0, atol=0)
    _assert_matches_reference(_combine(result, 12, 3), case)


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
@pytest.mark.parametrize("local", ((0, 8), (2, 4), (6, 2)), ids=lambda l: f"experts{l}")
def test_cudnn_expert_parallel_shard(key, local):
    """Only the local shard's assignments are computed; the rest map to row -1."""
    family = _FAMILIES[key]
    torch.manual_seed(6)
    case = _make_case(family, 24, 8, 2, 256, 256, local=local)
    result = _forward(case)
    _assert_matches_reference(_check_unfinalized(result, case), case)
    finalized = _forward(case, finalize=True)
    assert finalized.shape == case.expected.shape and finalized.dtype is torch.bfloat16
    _assert_matches_reference(finalized, case)


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
@pytest.mark.parametrize("use_moe_utils", _PERMUTE_PATHS)
def test_cudnn_permute_paths_match_reference(key, use_moe_utils):
    """The kernel and torch permute paths both reproduce the reference."""
    family = _FAMILIES[key]
    torch.manual_seed(30)
    case = _make_case(family, 100, 8, 3, 256, 256)
    result = _forward(case, use_moe_utils=use_moe_utils)
    _assert_matches_reference(_check_unfinalized(result, case), case)


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
@pytest.mark.parametrize("use_moe_utils", _PERMUTE_PATHS)
def test_cudnn_finalize_matches_reference_and_replays(key, use_moe_utils):
    """``do_finalize=True`` returns the combined ``[T, H]`` rows on both finalize paths.

    The finalized output agrees with the reference and with the caller-side
    combine of the unfinalized triple up to BF16 rounding, and a CUDA graph of
    the finalized forward replays it bit for bit.
    """
    family = _FAMILIES[key]
    torch.manual_seed(26)
    case = _make_case(family, 24, 8, 2, 256, 128)
    unpacked = MoEActivationPack(
        case.act.hidden_states_q,
        case.act.hidden_states_scale,
        case.act.topk_ids,
        case.act.topk_weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    finalized = _layer(case, finalize=True, use_moe_utils=use_moe_utils)
    out = finalized(unpacked, case.weights)
    assert isinstance(out, torch.Tensor)
    assert out.shape == case.expected.shape and out.dtype is torch.bfloat16
    _assert_matches_reference(out, case)
    unfinalized = _layer(case, use_moe_utils=use_moe_utils)(unpacked, case.weights)
    torch.testing.assert_close(
        out, _combine(unfinalized, case.num_tokens, case.top_k), rtol=1e-2, atol=1e-2
    )
    runner = finalized.runners[0]
    inputs = runner.pack_inputs(unpacked, case.weights)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.forward(inputs)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, out, rtol=0, atol=0)


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
def test_cudnn_layer_reuse_across_token_counts(key):
    """One layer serves changing token counts (and buckets) and is deterministic."""
    family = _FAMILIES[key]
    torch.manual_seed(9)
    # The layer's token ceiling comes from the first case: 200 tokens.
    first = _make_case(family, 200, 4, 2, 256, 128)
    layer = _layer(first)
    for num_tokens in (40, 3, 64, 4, 200, 40):
        case = _make_case(family, num_tokens, 4, 2, 256, 128, shared=first)
        result = layer(case.act, case.weights)
        combined = _check_unfinalized(result, case)
        _assert_matches_reference(combined, case)
        # Repeating a call reproduces every assignment's rows bit for bit
        # (padding rows of the output are unspecified).
        again = layer(case.act, case.weights)
        torch.testing.assert_close(
            _combine(again, case.num_tokens, case.top_k), combined, rtol=0, atol=0
        )


@pytest.mark.parametrize("key", _DYNAMIC_SCALE_FAMILY_PARAMS)
def test_cudnn_padding_rows_do_not_skew_dynamic_scales(key):
    """Padding rows never enter the intermediate's dynamic scale.

    Padding rows are computed with an expert that never sees their token. With
    that expert's weights blown up, a padding row entering the per-tensor
    (FP8) or global (NVFP4) scale of the intermediate would quantize every
    real row to zero. A preceding call that routes tokens to that expert
    leaves its outputs in exactly the rows the real routing pads.
    """
    family = _FAMILIES[key]
    torch.manual_seed(23)
    device = torch.device("cuda")
    num_tokens, num_experts, hidden_size, intermediate_size = 20, 3, 128, 128
    w1, w2 = _experts(num_experts, hidden_size, intermediate_size, SwiGLU(), device)
    w1[2] *= 1000.0
    w2[2] *= 1000.0
    case = _make_case(
        family,
        num_tokens,
        num_experts,
        1,
        hidden_size,
        intermediate_size,
        topk_ids=torch.zeros(num_tokens, 1, dtype=torch.int32, device=device),
        topk_weights=torch.ones(num_tokens, 1, device=device),
        experts=(w1, w2),
    )
    layer = _layer(case)
    _assert_matches_reference(
        _check_unfinalized(layer(case.act, case.weights), case), case
    )
    stale = MoEActivationPack(
        case.act.hidden_states_q,
        case.act.hidden_states_scale,
        _topk_ids_with_counts((7, 7, 6), device),
        case.act.topk_weights,
    )
    layer(stale, case.weights)  # fills the blown-up expert's rows
    _assert_matches_reference(
        _check_unfinalized(layer(case.act, case.weights), case), case
    )


def _autotune_and_graph(runner, case: _Case, *, cache_name: str):
    """Tune the compound tactic, check every advertised pair, then replay a CUDA graph."""
    inputs = runner.pack_inputs(case.act, case.weights)
    num_tokens, top_k = case.num_tokens, case.top_k
    routing_before = (case.act.topk_ids.clone(), case.act.topk_weights.clone())
    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            cache_name, [runner], runner.tuning_config_for(inputs), inputs
        )
    # Profiling synthesizes its own routing; the caller's tensors are untouched.
    torch.testing.assert_close(case.act.topk_ids, routing_before[0], rtol=0, atol=0)
    torch.testing.assert_close(case.act.topk_weights, routing_before[1], rtol=0, atol=0)
    assert isinstance(tactic, tuple) and len(tactic) == 2
    valid = runner.get_valid_tactics(inputs, None)
    assert (-1, -1) in valid
    assert all(isinstance(t, tuple) and len(t) == 2 for t in valid)
    for candidate in valid + [tactic]:
        _assert_matches_reference(
            _combine(runner.forward(inputs, tactic=candidate), num_tokens, top_k), case
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.forward(inputs, tactic=tactic)
    captured[0].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(_combine(captured, num_tokens, top_k), case)
    return tactic


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
def test_cudnn_autotune_and_cuda_graph_replay(key):
    """Every advertised (gemm1, gemm2) tactic pair matches; the tuned one replays."""
    family = _FAMILIES[key]
    torch.manual_seed(8)
    case = _make_case(family, 16, 8, 2, 256, 256)
    runner = _layer(case).runners[0]
    _autotune_and_graph(runner, case, cache_name=f"test_unified_moe_{key}")


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
def test_cudnn_moe_layer_autotunes_and_caches_a_winner_per_bucket(key):
    family = _FAMILIES[key]
    torch.manual_seed(10)
    first = _make_case(family, 16, 4, 2, 256, 128)
    layer = _layer(first)
    with autotune(True):
        layer(first.act, first.weights)
    _assert_matches_reference(
        _check_unfinalized(layer(first.act, first.weights), first), first
    )
    for num_tokens in (3, 40):
        case = _make_case(family, num_tokens, 4, 2, 256, 128, shared=first)
        _assert_matches_reference(
            _check_unfinalized(layer(case.act, case.weights), case), case
        )
    assert layer.winner_backend == key
    layer.reset_winner()
    assert layer.winner_backend is None


# --- preparation contracts and fail-fast validation --------------------------


@pytest.mark.parametrize("key", _FAMILY_PARAMS)
def test_cudnn_prepare_weights_and_activations_contract(key):
    """View keys, shapes and dtypes; dequantizing the view recovers the weights."""
    family = _FAMILIES[key]
    torch.manual_seed(31)
    device = torch.device("cuda")
    num_experts, hidden_size, intermediate_size = 3, 256, 128
    w1, w2 = _experts(num_experts, hidden_size, intermediate_size, SwiGLU(), device)
    view = family.config_cls.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    keys = {"fc1_expert_weights", "fc2_expert_weights"}
    if key == _FP8_KEY:
        keys |= {"fc1_dequant", "fc2_dequant"}
    elif key == _MXFP8_KEY:
        keys |= {"fc1_weight_scale", "fc2_weight_scale"}
    elif key == _NVFP4_KEY:
        keys |= {"fc1_weight_scale", "fc2_weight_scale", "fc1_dequant", "fc2_dequant"}
    assert set(view) == keys
    k_pack = 2 if key == _NVFP4_KEY else 1
    assert view["fc1_expert_weights"].shape == (
        num_experts,
        2 * intermediate_size,
        hidden_size // k_pack,
    )
    assert view["fc2_expert_weights"].shape == (
        num_experts,
        hidden_size,
        intermediate_size // k_pack,
    )
    for tensor in view.values():
        assert tensor.is_contiguous() and tensor.device.type == "cuda"
    for suffix in ("dequant",):
        for prefix in ("fc1", "fc2"):
            if f"{prefix}_{suffix}" in view:
                dequant = view[f"{prefix}_{suffix}"]
                assert (
                    dequant.shape == (num_experts,) and dequant.dtype is torch.float32
                )
                assert (dequant > 0).all()
    # fc1 rows come back as [gate, up], the fused activation kernels' operand order.
    fc1 = _swap_halves(_dequant_experts(family, view, "fc1"), intermediate_size)
    fc2 = _dequant_experts(family, view, "fc2")
    frac = {_BF16_KEY: 0.0, _FP8_KEY: 0.1, _MXFP8_KEY: 0.1, _NVFP4_KEY: 0.3}[key]
    for actual, source in ((fc1, w1), (fc2, w2)):
        torch.testing.assert_close(
            actual, source, rtol=frac, atol=frac * source.abs().max().item()
        )
    # Non-gated activations halve GEMM1's rows.
    relu2 = family.config_cls.prepare_weights(
        w1[:, :intermediate_size].contiguous(),
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=ReLU2(),
    )
    assert relu2["fc1_expert_weights"].shape[1] == intermediate_size
    if key == _BF16_KEY:
        return
    x = torch.randn(6, hidden_size, dtype=torch.bfloat16, device=device)
    q, scale = family.config_cls.prepare_activations(x)
    if key == _FP8_KEY:
        assert q.dtype is torch.float8_e4m3fn and q.shape == x.shape
        assert scale.dtype is torch.float32 and scale.ndim == 0
    elif key == _MXFP8_KEY:
        assert q.dtype is torch.float8_e4m3fn and q.shape == x.shape
        assert scale.dtype is torch.uint8 and scale.shape == (6, hidden_size // 32)
    else:
        assert q.dtype is torch.uint8 and q.shape == (6, hidden_size // 2)
        assert scale.dtype is torch.float8_e4m3fn and scale.shape == (
            6,
            hidden_size // 16,
        )
    torch.testing.assert_close(
        _dequant_activations(family, q, scale),
        x,
        rtol=frac,
        atol=frac * x.abs().max().item(),
    )
    with pytest.raises(ValueError, match="2D BF16"):
        family.config_cls.prepare_activations(x.float())
    if key in _BLOCK_SCALE_KEYS:
        for width in (40, 160):
            with pytest.raises(ValueError, match="hidden_size divisible by 128"):
                family.config_cls.prepare_activations(
                    torch.randn(6, width, dtype=torch.bfloat16, device=device)
                )


@cudnn_bf16_required
def test_cudnn_bf16_pack_inputs_fail_fast():
    family = _FAMILIES[_BF16_KEY]
    torch.manual_seed(7)
    device = torch.device("cuda")
    case = _make_case(family, 4, 4, 2, 64, 64)
    act, weights = case.act, case.weights
    runner = _layer(case).runners[0]
    with pytest.raises(TypeError, match="2D torch.bfloat16"):
        runner.pack_inputs(
            MoEActivationPack(
                act.hidden_states_q.float(), None, act.topk_ids, act.topk_weights
            ),
            weights,
        )
    with pytest.raises(ValueError, match="contiguous hidden states"):
        runner.pack_inputs(
            MoEActivationPack(
                torch.randn(4, 128, dtype=torch.bfloat16, device=device)[:, ::2],
                None,
                act.topk_ids,
                act.topk_weights,
            ),
            weights,
        )
    with pytest.raises(ValueError, match="do not use hidden_states_scale"):
        runner.pack_inputs(
            MoEActivationPack(
                act.hidden_states_q,
                torch.ones((), device=device),
                act.topk_ids,
                act.topk_weights,
            ),
            weights,
        )
    with pytest.raises(NotImplementedError, match="zero tokens"):
        runner.pack_inputs(
            MoEActivationPack(
                act.hidden_states_q[:0], None, act.topk_ids[:0], act.topk_weights[:0]
            ),
            weights,
        )
    with pytest.raises(NotImplementedError, match="pre-routed"):
        runner.pack_inputs(
            MoEActivationPack(
                act.hidden_states_q,
                None,
                routing_input_mode=RoutingInputMode.FromLogits,
                routing_logits=torch.randn(4, 4, device=device),
            ),
            weights,
        )
    view = weights.get_view(_BF16_KEY)
    for bad_view, error, match in (
        ({"fc1_expert_weights": view["fc1_expert_weights"]}, KeyError, "missing"),
        (
            {**view, "fc1_expert_weights": view["fc1_expert_weights"][:, :-1]},
            ValueError,
            "weight shapes",
        ),
        (
            {**view, "fc1_expert_weights": view["fc1_expert_weights"].half()},
            TypeError,
            "must be torch.bfloat16",
        ),
    ):
        bad = MoEWeightPack()
        bad.prepare_for(_BF16_KEY, bad_view)
        with pytest.raises(error, match=match):
            runner.pack_inputs(act, bad)
    small = _layer(
        dataclasses.replace(
            case,
            config=dataclasses.replace(
                case.config,
                execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=2),
            ),
        )
    ).runners[0]
    with pytest.raises(ValueError, match="exceeds tune_max_num_tokens"):
        small.pack_inputs(act, weights)


@cudnn_fp8_required
def test_cudnn_fp8_pack_inputs_fail_fast():
    family = _FAMILIES[_FP8_KEY]
    torch.manual_seed(14)
    case = _make_case(family, 16, 4, 2, 128, 256)
    act, weights = case.act, case.weights
    runner = _layer(case).runners[0]
    with pytest.raises(ValueError, match="0-dim float32"):
        runner.pack_inputs(
            MoEActivationPack(
                act.hidden_states_q,
                act.hidden_states_scale.reshape(1),
                act.topk_ids,
                act.topk_weights,
            ),
            weights,
        )
    with pytest.raises(TypeError, match="2D torch.float8_e4m3fn"):
        runner.pack_inputs(
            MoEActivationPack(
                act.hidden_states_q.view(torch.float8_e5m2),
                act.hidden_states_scale,
                act.topk_ids,
                act.topk_weights,
            ),
            weights,
        )
    view = weights.get_view(_FP8_KEY)
    for bad_view, error, match in (
        (
            {
                **view,
                "fc1_expert_weights": view["fc1_expert_weights"].view(
                    torch.float8_e5m2
                ),
            },
            TypeError,
            "must be torch.float8_e4m3fn",
        ),
        (
            {**view, "fc1_dequant": view["fc1_dequant"].half()},
            ValueError,
            "fc1_dequant must be",
        ),
        (
            {**view, "fc2_dequant": view["fc2_dequant"][:-1]},
            ValueError,
            "fc2_dequant must be",
        ),
    ):
        bad = MoEWeightPack()
        bad.prepare_for(_FP8_KEY, bad_view)
        with pytest.raises(error, match=match):
            runner.pack_inputs(act, bad)


@pytest.mark.parametrize("key", _BLOCK_SCALE_FAMILY_PARAMS)
def test_cudnn_block_scale_pack_inputs_fail_fast(key):
    family = _FAMILIES[key]
    torch.manual_seed(20)
    device = torch.device("cuda")
    case = _make_case(family, 8, 4, 2, 128, 128)
    act, weights = case.act, case.weights
    runner = _layer(case).runners[0]
    for bad_scale in (None, torch.ones((), device=device), act.hidden_states_scale[1:]):
        with pytest.raises(ValueError, match="requires hidden_states_scale"):
            runner.pack_inputs(
                MoEActivationPack(
                    act.hidden_states_q, bad_scale, act.topk_ids, act.topk_weights
                ),
                weights,
            )
    with pytest.raises(TypeError, match="requires 2D"):
        runner.pack_inputs(
            MoEActivationPack(
                torch.randn(8, 128, dtype=torch.bfloat16, device=device),
                act.hidden_states_scale,
                act.topk_ids,
                act.topk_weights,
            ),
            weights,
        )
    narrow = MoEActivationPack(
        act.hidden_states_q[:, : act.hidden_states_q.shape[1] // 2].contiguous(),
        act.hidden_states_scale[
            :, : act.hidden_states_scale.shape[1] // 2
        ].contiguous(),
        act.topk_ids,
        act.topk_weights,
    )
    with pytest.raises(ValueError, match="hidden_size divisible by 128"):
        runner.pack_inputs(narrow, weights)
    view = weights.get_view(key)
    for bad_view, error, match in (
        (
            {**view, "fc1_weight_scale": view["fc1_weight_scale"][:, :-1]},
            ValueError,
            "fc1_weight_scale must be",
        ),
        (
            {**view, "fc2_weight_scale": view["fc2_weight_scale"].to(torch.float32)},
            ValueError,
            "fc2_weight_scale must be",
        ),
        (
            {k: v for k, v in view.items() if k != "fc2_weight_scale"},
            KeyError,
            "missing",
        ),
    ):
        bad = MoEWeightPack()
        bad.prepare_for(key, bad_view)
        with pytest.raises(error, match=match):
            runner.pack_inputs(act, bad)
