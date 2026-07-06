"""flashinfer.moe_ep — MoE Expert-Parallel (split + mega kernels).

Package layout::

<<<<<<< HEAD
    moe_ep/
      core/                 shared comm + kernel abstractions and validation
      backends/
        split/
          comm/             NCCL-EP, NIXL-EP transport
          kernel/           post-dispatch inner kernels
        mega/
          kernel/           fused comm + local MoE kernels
      modes/                split and mega orchestration layers
=======
- ``flashinfer.moe_ep.nccl_ep``  — primary backend, wraps NVIDIA's nccl4py
  ``nccl.ep`` API (nccl-ep-v0.1.0; built from ``3rdparty/nccl/bindings/nccl4py``).
- ``flashinfer.moe_ep.nixl_ep``  — alternate backend, wraps ai-dynamo's
  ``nixl_ep`` (built in-tree from ``3rdparty/nixl/examples/device/ep``).

NCCL-EP availability is the importability of ``nccl.ep`` (no in-tree
``libnccl_ep.so`` as of v0.1.0); NIXL-EP still ships a staged ``nixl_ep_cpp*.so``.
Both are produced by the FlashInfer build only when ``BUILD_NVEP=1`` (or the
per-backend ``BUILD_NCCL_EP`` / ``BUILD_NIXL_EP``) is set at install time:

    BUILD_NVEP=1 pip install -e ".[nvep]"

Without a built backend the package imports succeed but calling
:func:`create_fleet` raises :class:`MoEEpNotBuiltError` with rebuild
instructions.
>>>>>>> upstream/main
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import MoEEpNotBuiltError
from .algo_knobs import (
    AlgoKnob,
    FleetAlgoKnobAllocator,
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
from .backends.mega.kernel.mxfp8_cutedsl import (
    Mxfp8CutedslMegaMoeConfig,
    preprocess_mega_weights as preprocess_mxfp8_cutedsl_mega_weights,
)
from .backends.mega.kernel.nvfp4_cutedsl import (
    Nvfp4CutedslMegaMoeConfig,
    preprocess_mega_weights as preprocess_nvfp4_cutedsl_mega_weights,
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
from .core.bootstrap_utils import (
    bootstrap_comm_group,
    bootstrap_ep_rank_world,
    bootstrap_ep_world_size,
)
from .core.comm.fleet import Fleet, create_fleet
from .core.comm.handle import Handle
from .core.runtime import (
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from .core.validation import (
    MoEEpArchError,
    MoEEpConfigError,
    ensure_bootstrap_dist_validated,
    validate_arch_for_backend,
    validate_bootstrap_process_group_ready,
    validate_bootstrap_world_size,
    validate_fleet_params,
    validate_fleet_weights,
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
from .weights import MoEWeightPack, dummy_moe_weights

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
    "FleetAlgoKnobAllocator",
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
    "Mxfp8CutedslMegaMoeConfig",
    "NCCLEPConfig",
    "NcclEpConfig",
    "Nvfp4CutedslMegaMoeConfig",
    "NvepConfig",
    "QuantType",
    "SplitConfig",
    "SplitKernelContext",
    "available_backends",
    "bootstrap_comm_group",
    "bootstrap_ep_rank_world",
    "bootstrap_ep_world_size",
    "bootstrap_moe_ep_runtime",
    "create_fleet",
    "dummy_moe_weights",
    "ensure_bootstrap_dist_validated",
    "ensure_moe_ep_cuda_device",
    "finalize_moe_ep_runtime",
    "have_nccl_ep",
    "have_nixl_ep",
    "kernel_requires_weights",
    "preprocess_mega_weights",
    "preprocess_mxfp8_cutedsl_mega_weights",
    "preprocess_nvfp4_cutedsl_mega_weights",
    "run_split_kernel",
    "validate_arch_for_backend",
    "validate_bootstrap_process_group_ready",
    "validate_bootstrap_world_size",
    "validate_fleet_params",
    "validate_fleet_weights",
    "validate_mega_arch",
    "validate_mega_fleet_params",
    "validate_mega_forward_inputs",
    "validate_split_forward_inputs",
]


_pkg_dir = Path(__file__).parent
_REBUILD_HINT = (
    "flashinfer.moe_ep transport libs are not built. Rebuild with:\n"
    '    BUILD_NVEP=1 pip install -e ".[nvep]"\n'
    "from the FlashInfer source tree (libs stage under "
    "flashinfer/moe_ep/backends/split/comm/*/_libs/)."
)


def _nccl_libs_dir() -> Path:
    return _pkg_dir / "backends" / "split" / "comm" / "nccl_ep" / "_libs"


def _nixl_libs_dir() -> Path:
    return _pkg_dir / "backends" / "split" / "comm" / "nixl_ep" / "_libs"


def _probe_nccl_ep() -> bool:
<<<<<<< HEAD
    import importlib.util

=======
    """True if the NCCL-EP backend is available.

    As of nccl-ep-v0.1.0 the backend is the nccl4py ``nccl.ep`` package (no
    in-tree libnccl_ep.so).  Probe with ``find_spec`` so we don't import/execute
    ``nccl.ep`` (and pull in its native lib) just to answer availability.
    """
    import importlib.util

>>>>>>> upstream/main
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
            f"unknown moe_ep backend {backend!r}; expected one of nccl_ep, nixl_ep"
        )
    if not probe():
        raise MoEEpNotBuiltError(
            f"moe_ep backend {backend!r} is not built.\n\n{_REBUILD_HINT}"
        )


_set_build_flags = [
    name
    for name in ("BUILD_NVEP", "BUILD_NCCL_EP", "BUILD_NIXL_EP")
    if os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
]
if _set_build_flags and not available_backends():
    import warnings

    warnings.warn(
        f"{'/'.join(_set_build_flags)} was set, but no moe_ep backend "
        f"libraries were found. Check the build log "
        "for pre-flight probe misses (meson/make/nvcc/git on PATH, "
        "ucx/libibverbs via pkg-config, nixl-cu13 / nvidia-nccl-cu13 "
        "wheels importable) or meson/make compile failures.",
        RuntimeWarning,
        stacklevel=2,
    )

<<<<<<< HEAD
from . import backends as _backends  # noqa: E402,F401
from .backends.split.comm.nccl_ep import fleet as _nccl_ep_fleet  # noqa: E402,F401
from .backends.split.comm.nixl_ep import fleet as _nixl_ep_fleet  # noqa: E402,F401
=======
# Trigger backend registration. Importing these modules populates
# ``_BACKEND_REGISTRY`` via module-level assignments. Both imports are
# pure-Python and don't touch the nccl.ep native lib / nixl_ep_cpp.so — those
# only load when a Fleet is actually instantiated. Must happen AFTER
# MoEEpNotBuiltError / _require_built are defined above (the backend
# modules `from .. import` them).
from .nccl_ep import fleet as _nccl_ep_fleet  # noqa: E402,F401
from .nixl_ep import fleet as _nixl_ep_fleet  # noqa: E402,F401
>>>>>>> upstream/main
