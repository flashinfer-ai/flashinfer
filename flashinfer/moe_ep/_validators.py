"""Backend-specific FleetParams / quant validators.

Backend Fleet __init__ calls into these before touching the C ABI so a
config error surfaces immediately rather than as a cryptic kernel-launch
failure later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import BootstrapConfig, FleetParams, QuantType

if TYPE_CHECKING:
    from ..fused_moe.api import MoEConfig
    from .algo_knobs import FleetAlgoKnobQuantization


# NIXL EP's kernels are JIT-compiled only for these hidden sizes; the wrapper
# rounds up the layer's hidden dimension to the nearest supported value. The
# canonical set is from vLLM's NixlEPPrepareAndFinalize.
_NIXL_EP_SUPPORTED_HIDDEN_SIZES = frozenset(
    {2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}
)

# NIXL EP's `FINISHED_SUM_TAG` is hard-coded to 1024 in the kernel.
_NIXL_EP_MAX_TOKENS_PER_RANK = 1024

# NCCL-EP group-create fails on Blackwell (B200) with older NCCL
# (2.27.x/2.29.x, at nccl_ep.cc:1438); >=2.30.7 carries the B200 EP support.
# This floor is enforced HERE rather than as a base-dependency pin because
# torch's cu13 wheels pin nvidia-nccl-cu13 exactly (e.g. ==2.29.7) — a
# metadata floor makes pip evict torch (see requirements.txt). The build hook
# upgrades the wheel --no-deps on source installs (build_backend.py).
_NCCL_EP_BLACKWELL_MIN_NCCL = (2, 30, 7)


class MoEEpConfigError(ValueError):
    """Raised when an EP config field is out-of-range for the chosen backend."""


class MoEEpArchError(MoEEpConfigError):
    """Raised when the GPU arch doesn't support the chosen backend."""


def _installed_nccl_version() -> "tuple[int, int, int] | None":
    """Best-effort probe of the NCCL version the EP backend will load.

    Prefers the nvidia-nccl-cu13 pip wheel's metadata (cuda-pathfinder loads
    that wheel's libnccl first when present); falls back to ncclGetVersion on
    the dynamic linker's default search path (covers NGC-style images with a
    system NCCL and no pip wheel). Returns None when undeterminable — callers
    must not block in that case.
    """
    try:
        from importlib.metadata import version

        parts = version("nvidia-nccl-cu13").split(".")[:3]
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except Exception:
        pass
    try:
        import ctypes

        lib = ctypes.CDLL("libnccl.so.2")
        out = ctypes.c_int()
        if lib.ncclGetVersion(ctypes.byref(out)) == 0:
            # NCCL_VERSION_CODE encoding: major*10000 + minor*100 + patch
            # (e.g. 2.30.7 -> 23007).
            code = out.value
            return (code // 10000, (code // 100) % 100, code % 100)
    except Exception:
        pass
    return None


def validate_arch_for_backend(backend: str) -> None:
    """Check the GPU arch and CUDA version are supported by `backend`."""
    import torch

    # The EP runtime wheels (nccl4py, nvidia-nccl-cu13, nixl-cu13) are
    # CUDA-13-only, so a torch built for CUDA 12 can't drive either backend —
    # fail here with a clear message instead of a cryptic dlopen error later.
    # Parse defensively: custom/nightly torch builds can carry version
    # strings this check shouldn't crash on; skip it when unparseable.
    cuda_ver = torch.version.cuda
    try:
        cuda_major = int(cuda_ver.split(".")[0]) if cuda_ver else None
    except ValueError:
        cuda_major = None
    if cuda_major is not None and cuda_major < 13:
        raise MoEEpConfigError(
            f"{backend} requires CUDA 13: the EP runtime wheels (nccl4py, "
            f"nvidia-nccl-cu13, nixl-cu13) ship CUDA-13 binaries only, but "
            f"the installed torch was built for CUDA {cuda_ver}. Install a "
            "CUDA-13 torch build to use flashinfer.moe_ep."
        )

    if not torch.cuda.is_available():
        return  # Mock/test path — let backend probes catch missing libs instead.
    cc = torch.cuda.get_device_capability(0)
    # Both nccl_ep and nixl_ep require sm_90+.
    if cc < (9, 0):
        raise MoEEpArchError(f"{backend} requires sm_90+, host has sm_{cc[0]}{cc[1]}")

    # NCCL-EP group-create fails on Blackwell with NCCL < 2.30.7 — catch it
    # here (Fleet construction) with an actionable message instead of the
    # cryptic nccl_ep.cc:1438 failure. Skipped when the version can't be
    # determined (no pip wheel + no loadable libnccl.so.2).
    if backend == "nccl_ep" and cc >= (10, 0):
        nccl_ver = _installed_nccl_version()
        if nccl_ver is not None and nccl_ver < _NCCL_EP_BLACKWELL_MIN_NCCL:
            floor = ".".join(map(str, _NCCL_EP_BLACKWELL_MIN_NCCL))
            found = ".".join(map(str, nccl_ver))
            raise MoEEpConfigError(
                f"nccl_ep on Blackwell (sm_{cc[0]}{cc[1]}) requires NCCL >= "
                f"{floor} (group-create fails with older releases); found "
                f"{found}. Upgrade with:\n"
                f"    pip install --no-deps 'nvidia-nccl-cu13>={floor}'\n"
                "and ensure that wheel's libnccl is the one loaded (first on "
                "LD_LIBRARY_PATH) rather than a base-image system NCCL."
            )


def validate_fleet_params(
    params: FleetParams,
    backend: str,
    world_size: int,
    quant: "FleetAlgoKnobQuantization | None" = None,
) -> None:
    """Validate ``params`` against backend-specific constraints.

    * ``num_experts % world_size == 0`` is required by both backends.
    * NIXL EP further requires ``max_tokens_per_rank ≤ 1024`` and
      ``token_hidden_size`` in the SUPPORTED_HIDDEN_SIZES set.
    * UE8M0 quant on NIXL EP requires sm_100+ (Blackwell); rejected on sm_90.
    """
    import torch

    if params.num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({params.num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )

    if backend == "nixl_ep":
        if params.max_tokens_per_rank > _NIXL_EP_MAX_TOKENS_PER_RANK:
            raise MoEEpConfigError(
                f"nixl_ep: max_tokens_per_rank ({params.max_tokens_per_rank}) "
                f"must be ≤ {_NIXL_EP_MAX_TOKENS_PER_RANK}"
            )
        if params.token_hidden_size not in _NIXL_EP_SUPPORTED_HIDDEN_SIZES:
            raise MoEEpConfigError(
                f"nixl_ep: token_hidden_size ({params.token_hidden_size}) not "
                f"in supported set {sorted(_NIXL_EP_SUPPORTED_HIDDEN_SIZES)}"
            )
        if quant is not None and QuantType.UE8M0 in quant.quants:
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability(0)
                if cc < (10, 0):
                    raise MoEEpConfigError(
                        f"nixl_ep: UE8M0 quantization requires sm_100+ (Blackwell); "
                        f"host has sm_{cc[0]}{cc[1]}"
                    )


def validate_compute_consistency(
    fleet_params: FleetParams,
    bootstrap: BootstrapConfig,
    compute_config: "MoEConfig",
) -> None:
    """Check the EP comm config and the unified-compute ``MoEConfig`` agree.

    The two configs are authored separately (one sizes the transport, one drives
    the per-expert FFN), so a mismatch is easy to introduce and produces a wrong
    result rather than an error.  Enforce the shared invariants up front:

    * global expert count: ``RoutingConfig.num_experts == FleetParams.num_experts``
    * hidden size: compute is inferred from tensors, but the EP buffer is sized by
      ``FleetParams.token_hidden_size`` — they must match (checked at forward via
      tensor shapes; here we only sanity-check the static fields).
    * per-rank sharding: ``ExpertConfig.local_num_experts`` and
      ``local_expert_offset`` must match this rank's slice of the global experts.
    """
    world_size = bootstrap.world_size
    rank = bootstrap.rank
    routing = compute_config.routing
    experts = compute_config.experts

    if routing.num_experts != fleet_params.num_experts:
        raise MoEEpConfigError(
            f"RoutingConfig.num_experts ({routing.num_experts}) != "
            f"FleetParams.num_experts ({fleet_params.num_experts}); the global "
            "expert count must be a single source of truth."
        )

    expected_local = fleet_params.num_experts // world_size
    local = experts.local_num_experts or routing.num_experts
    if local != expected_local:
        raise MoEEpConfigError(
            f"ExpertConfig.local_num_experts ({local}) != "
            f"num_experts // world_size ({expected_local}); each rank owns an "
            "equal shard of the global experts."
        )

    expected_offset = rank * expected_local
    if experts.local_expert_offset != expected_offset:
        raise MoEEpConfigError(
            f"ExpertConfig.local_expert_offset ({experts.local_expert_offset}) != "
            f"rank * local_num_experts ({expected_offset}) for rank {rank}."
        )

    # The EP bridge reshapes a finalized [M, hidden] tensor back for combine, and
    # RANK_MAJOR/HT rely on the runner's weighted local pre-reduce. do_finalize=False
    # would pass the un-reduced expert outputs through, which MoEEpLayer then
    # consumes as if finalized.
    if not compute_config.execution.do_finalize:
        raise MoEEpConfigError(
            "compute_config.execution.do_finalize must be True for MoE-EP: the "
            "bridge consumes a finalized [M, hidden] output and RANK_MAJOR/HT need "
            "the runner's weighted local pre-reduce."
        )
