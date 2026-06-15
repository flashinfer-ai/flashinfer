"""flashinfer.moe_ep_v2 — MoE Expert-Parallel (split + mega kernels).

Package layout::

    moe_ep_v2/
      split/     dispatch/combine transport (NCCL-EP, NIXL-EP) + inner kernels
      mega/      fused DeepGEMM mega-MoE (symmetric memory)

Native transport libraries are reused from :mod:`flashinfer.moe_ep` until
v2 ships its own staged ``_libs/`` trees under ``split/``.
"""

from __future__ import annotations

import os
from pathlib import Path

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
from .config import (
    BootstrapConfig,
    CombineInputParams,
    CombineOutput,
    DispatchInputParams,
    DispatchOutput,
    EpAlgorithm,
    FleetParams,
    HandleParams,
    QuantType,
)
from ._validators import (
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
from .fleet import Fleet, create_fleet
from .handle import Handle
from .layer import MoEEpLayer
from .mega import (
    DeepGemmMegaMoeConfig,
    MegaConfig,
    MoEEpMegaLayer,
    preprocess_mega_weights,
)
from .split import (
    FusedMoeKernelConfig,
    IdentityConfig,
    MoEEpSplitLayer,
    NCCLEPConfig,
    NcclEpConfig,
    NvepConfig,
    SplitConfig,
    SplitKernelContext,
)
from .tensors import MoEEpTensors
from .weights import MoEWeightPack

__all__ = [
    "AlgoKnob",
    "BootstrapConfig",
    "CombineInputParams",
    "CombineOutput",
    "DispatchInputParams",
    "DispatchOutput",
    "EpAlgorithm",
    "Fleet",
    "FleetAlgoKnobNumChannelsPerRank",
    "FleetAlgoKnobNumQpsPerRank",
    "FleetAlgoKnobQuantization",
    "FleetAlgoKnobRdmaBufferSize",
    "FleetAlgoKnobTopologyCapacity",
    "FleetParams",
    "Handle",
    "HandleAlgoKnobNumReceivedTokens",
    "HandleAlgoKnobSplitOperation",
    "HandleAlgoKnobTopKWeights",
    "HandleAlgoKnobUserStream",
    "HandleParams",
    "MoEEpArchError",
    "MoEEpConfigError",
    "DeepGemmMegaMoeConfig",
    "FusedMoeKernelConfig",
    "IdentityConfig",
    "MegaConfig",
    "MoEEpLayer",
    "MoEEpMegaLayer",
    "MoEEpNotBuiltError",
    "MoEEpSplitLayer",
    "MoEEpTensors",
    "MoEWeightPack",
    "NCCLEPConfig",
    "NcclEpConfig",
    "NvepConfig",
    "SplitConfig",
    "SplitKernelContext",
    "preprocess_mega_weights",
    "QuantType",
    "available_backends",
    "create_fleet",
    "have_nccl_ep",
    "have_nixl_ep",
    "validate_mega_arch",
    "validate_mega_forward_inputs",
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
        staged = root / "split" / "nccl_ep" / "_libs"
        if (staged / "libnccl_ep.so").exists():
            return staged
        legacy = root / "nccl_ep" / "_libs"
        if (legacy / "libnccl_ep.so").exists():
            return legacy
    return _pkg_dir / "split" / "nccl_ep" / "_libs"


def _nixl_libs_dir() -> Path:
    for root in (_pkg_dir, _moe_ep_libs_dir):
        staged = root / "split" / "nixl_ep" / "_libs"
        if staged.is_dir() and any(staged.glob("nixl_ep_cpp*.so")):
            return staged
        legacy = root / "nixl_ep" / "_libs"
        if legacy.is_dir() and any(legacy.glob("nixl_ep_cpp*.so")):
            return legacy
    return _pkg_dir / "split" / "nixl_ep" / "_libs"


class MoEEpNotBuiltError(RuntimeError):
    """Raised when an EP backend is invoked but its native libs are missing."""


def _probe_nccl_ep() -> bool:
    """True if the NCCL-EP plugin .so was staged by the build."""
    libs = _nccl_libs_dir()
    return (libs / "libnccl_ep.so").exists()


def _probe_nixl_ep() -> bool:
    """True if the NIXL-EP plugin .so was staged by the build."""
    libs = _nixl_libs_dir()
    return libs.is_dir() and any(libs.glob("nixl_ep_cpp*.so"))


def have_nccl_ep() -> bool:
    """Return True if the NCCL-EP backend native libs are present."""
    return _probe_nccl_ep()


def have_nixl_ep() -> bool:
    """Return True if the NIXL-EP backend native libs are present."""
    return _probe_nixl_ep()


def available_backends() -> list[str]:
    """Names of EP backends with both native libs and python wrappers present."""
    out: list[str] = []
    if have_nccl_ep():
        out.append("nccl_ep")
    if have_nixl_ep():
        out.append("nixl_ep")
    return out


def _require_built(backend: str) -> None:
    """Raise MoEEpNotBuiltError if `backend` is missing its native libs."""
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

from .split.nccl_ep import fleet as _nccl_ep_fleet  # noqa: E402,F401
from .split.nixl_ep import fleet as _nixl_ep_fleet  # noqa: E402,F401
