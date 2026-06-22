"""flashinfer.moe_ep_v2 — MoE Expert-Parallel (split + mega kernels).

Package layout::

    moe_ep_v2/
      core/                 shared comm + kernel abstractions and validation
      backends/
        split/
          comm/             NCCL-EP, NIXL-EP transport
          kernel/           post-dispatch inner kernels
        mega/
          kernel/           fused comm + local MoE kernels
      modes/                split and mega orchestration layers
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import MoEEpNotBuiltError
from .algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobNumChannelsPerRank,
    FleetAlgoKnobNumQpsPerRank,
    FleetAlgoKnobQuantization,
    FleetAlgoKnobRdmaBufferSize,
    FleetAlgoKnobTopologyCapacity,
    HandleAlgoKnobNumReceivedTokens,
    HandleAlgoKnobSplitOperation,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
)
from .backends.mega.kernel.deep_gemm_mega import (
    DeepGemmMegaMoeConfig,
    preprocess_mega_weights,
)
from .config import (
    BootstrapConfig,
    CombineInputParams,
    CombineOutput,
    DispatchInputParams,
    DispatchOutput,
    EpAlgorithm,
    EpLayout,
    FleetParams,
    HandleParams,
    QuantType,
)
from .core.comm.fleet import Fleet, create_fleet
from .core.comm.handle import Handle
from .core.validation import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_arch_for_backend,
    validate_bootstrap_world_size,
    validate_fleet_params,
    validate_mega_arch,
    validate_mega_fleet_params,
    validate_mega_forward_inputs,
    validate_split_forward_inputs,
)
from .layer import MoEEpLayer
from .modes import (
    FusedMoeKernelConfig,
    IdentityConfig,
    MegaConfig,
    MoEEpMegaLayer,
    MoEEpSplitLayer,
    NCCLEPConfig,
    NcclEpConfig,
    NvepConfig,
    SplitConfig,
    SplitKernelContext,
    kernel_requires_weights,
    run_split_kernel,
)
from .tensors import MoEEpTensors
from .weights import MoEWeightPack

__all__ = [
    "AlgoKnob",
    "BootstrapConfig",
    "CombineInputParams",
    "CombineOutput",
    "DeepGemmMegaMoeConfig",
    "DispatchInputParams",
    "DispatchOutput",
    "EpAlgorithm",
    "EpLayout",
    "Fleet",
    "FleetAlgoKnobNumChannelsPerRank",
    "FleetAlgoKnobNumQpsPerRank",
    "FleetAlgoKnobQuantization",
    "FleetAlgoKnobRdmaBufferSize",
    "FleetAlgoKnobTopologyCapacity",
    "FleetParams",
    "FusedMoeKernelConfig",
    "Handle",
    "HandleAlgoKnobNumReceivedTokens",
    "HandleAlgoKnobSplitOperation",
    "HandleAlgoKnobTopKWeights",
    "HandleAlgoKnobUserStream",
    "HandleParams",
    "IdentityConfig",
    "MegaConfig",
    "MoEEpArchError",
    "MoEEpConfigError",
    "MoEEpLayer",
    "MoEEpMegaLayer",
    "MoEEpNotBuiltError",
    "MoEEpSplitLayer",
    "MoEEpTensors",
    "MoEWeightPack",
    "NCCLEPConfig",
    "NcclEpConfig",
    "NvepConfig",
    "QuantType",
    "SplitConfig",
    "SplitKernelContext",
    "available_backends",
    "create_fleet",
    "have_nccl_ep",
    "have_nixl_ep",
    "kernel_requires_weights",
    "preprocess_mega_weights",
    "run_split_kernel",
    "validate_arch_for_backend",
    "validate_bootstrap_world_size",
    "validate_fleet_params",
    "validate_mega_arch",
    "validate_mega_fleet_params",
    "validate_mega_forward_inputs",
    "validate_split_forward_inputs",
]


_pkg_dir = Path(__file__).parent
_moe_ep_libs_dir = _pkg_dir.parent / "moe_ep"
_REBUILD_HINT = (
    "flashinfer.moe_ep_v2 transport libs are not built. Rebuild with:\n"
    '    BUILD_NVEP=1 pip install -e ".[nvep]"\n'
    "from the FlashInfer source tree (libs stage under flashinfer/moe_ep/)."
)


def _nccl_libs_dir() -> Path:
    for root in (_pkg_dir, _moe_ep_libs_dir):
        staged = root / "backends" / "split" / "comm" / "nccl_ep" / "_libs"
        if (staged / "libnccl_ep.so").exists():
            return staged
        legacy = root / "split" / "nccl_ep" / "_libs"
        if (legacy / "libnccl_ep.so").exists():
            return legacy
        legacy = root / "nccl_ep" / "_libs"
        if (legacy / "libnccl_ep.so").exists():
            return legacy
    return _pkg_dir / "backends" / "split" / "comm" / "nccl_ep" / "_libs"


def _nixl_libs_dir() -> Path:
    for root in (_pkg_dir, _moe_ep_libs_dir):
        staged = root / "backends" / "split" / "comm" / "nixl_ep" / "_libs"
        if staged.is_dir() and any(staged.glob("nixl_ep_cpp*.so")):
            return staged
        legacy = root / "split" / "nixl_ep" / "_libs"
        if legacy.is_dir() and any(legacy.glob("nixl_ep_cpp*.so")):
            return legacy
        legacy = root / "nixl_ep" / "_libs"
        if legacy.is_dir() and any(legacy.glob("nixl_ep_cpp*.so")):
            return legacy
    return _pkg_dir / "backends" / "split" / "comm" / "nixl_ep" / "_libs"


def _probe_nccl_ep() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("nccl.ep") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _probe_nixl_ep() -> bool:
    libs = _nixl_libs_dir()
    return libs.is_dir() and any(libs.glob("nixl_ep_cpp*.so"))


def have_nccl_ep() -> bool:
    return _probe_nccl_ep()


def have_nixl_ep() -> bool:
    return _probe_nixl_ep()


def available_backends() -> list[str]:
    out: list[str] = []
    if have_nccl_ep():
        out.append("nccl_ep")
    if have_nixl_ep():
        out.append("nixl_ep")
    return out


def _require_built(backend: str) -> None:
    probe = {"nccl_ep": _probe_nccl_ep, "nixl_ep": _probe_nixl_ep}.get(backend)
    if probe is None:
        raise ValueError(
            f"unknown moe_ep_v2 backend {backend!r}; expected one of nccl_ep, nixl_ep"
        )
    if not probe():
        raise MoEEpNotBuiltError(
            f"moe_ep_v2 backend {backend!r} is not built.\n\n{_REBUILD_HINT}"
        )


_set_build_flags = [
    name
    for name in ("BUILD_NVEP", "BUILD_NCCL_EP", "BUILD_NIXL_EP")
    if os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
]
if _set_build_flags and not available_backends():
    import warnings

    warnings.warn(
        f"{'/'.join(_set_build_flags)} was set, but no moe_ep_v2 backend "
        f"libraries were found. Check the build log "
        "for pre-flight probe misses (meson/make/nvcc/git on PATH, "
        "ucx/libibverbs via pkg-config, nixl-cu13 / nvidia-nccl-cu13 "
        "wheels importable) or meson/make compile failures.",
        RuntimeWarning,
        stacklevel=2,
    )

from . import backends as _backends  # noqa: E402,F401
from .backends.split.comm.nccl_ep import fleet as _nccl_ep_fleet  # noqa: E402,F401
from .backends.split.comm.nixl_ep import fleet as _nixl_ep_fleet  # noqa: E402,F401
