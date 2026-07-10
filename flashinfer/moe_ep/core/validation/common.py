"""Shared validation helpers for moe_ep layers and comm backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import BootstrapConfig, EpAlgorithm, EpLayout, FleetParams, QuantType
from ...weights import MoEWeightPack

if TYPE_CHECKING:
    import torch

    from ...algo_knobs import FleetAlgoKnobQuantization


_NIXL_EP_SUPPORTED_HIDDEN_SIZES = frozenset(
    {2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}
)
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


def _device_capability() -> tuple[int, int]:
    import torch

    return torch.cuda.get_device_capability(torch.cuda.current_device())


def validate_bootstrap_process_group_ready(bootstrap: BootstrapConfig) -> None:
    """Fail fast when ``process_group`` is set but ``torch.distributed`` is not up."""
    import torch.distributed as dist

    if bootstrap.process_group is not None and not dist.is_initialized():
        raise MoEEpConfigError(
            "BootstrapConfig.process_group is set but torch.distributed is not "
            "initialized; initialize torch.distributed before layer construction, "
            "or set auto_bootstrap=True"
        )


def validate_bootstrap_world_size(bootstrap: BootstrapConfig) -> None:
    import torch.distributed as dist

    from ..bootstrap_utils import bootstrap_comm_group, bootstrap_ep_rank_world

    validate_bootstrap_process_group_ready(bootstrap)

    if not dist.is_initialized():
        return

    pg = bootstrap_comm_group(bootstrap)
    pg_ws = dist.get_world_size(pg)
    pg_rank = dist.get_rank(pg)
    if bootstrap.world_size != pg_ws:
        pg_label = (
            "BootstrapConfig.process_group"
            if bootstrap.process_group is not None
            else "torch.distributed world"
        )
        raise MoEEpConfigError(
            f"BootstrapConfig.world_size ({bootstrap.world_size}) must match "
            f"{pg_label} size ({pg_ws})"
        )
    if bootstrap.rank != pg_rank:
        pg_label = (
            "BootstrapConfig.process_group"
            if bootstrap.process_group is not None
            else "torch.distributed world"
        )
        raise MoEEpConfigError(
            f"BootstrapConfig.rank ({bootstrap.rank}) must match "
            f"{pg_label} rank ({pg_rank})"
        )
    # Sanity: resolved EP sizing matches the configured bootstrap fields.
    resolved_rank, resolved_ws = bootstrap_ep_rank_world(bootstrap)
    if resolved_rank != bootstrap.rank or resolved_ws != bootstrap.world_size:
        raise MoEEpConfigError(
            f"BootstrapConfig rank/world_size ({bootstrap.rank}, "
            f"{bootstrap.world_size}) does not match resolved EP comm "
            f"({resolved_rank}, {resolved_ws})"
        )


def ensure_bootstrap_dist_validated(bootstrap: BootstrapConfig) -> None:
    """Validate bootstrap against the active process group once dist is available.

    Safe to call repeatedly. When ``torch.distributed`` is not initialized yet
    (e.g. ``auto_bootstrap=False`` and the host framework has not inited dist),
    only the ``process_group`` readiness check runs; full rank/world_size
    cross-check runs on the first call after dist comes up.
    """
    validate_bootstrap_process_group_ready(bootstrap)
    validate_bootstrap_world_size(bootstrap)


def validate_split_forward_inputs(
    hidden_states: "torch.Tensor",
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    fleet_params: FleetParams,
) -> None:
    num_tokens = hidden_states.shape[0]
    if num_tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError(
            f"token count {num_tokens} exceeds "
            f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}"
        )
    if hidden_states.ndim != 2:
        raise MoEEpConfigError(
            f"hidden_states must be 2D [num_tokens, hidden], got shape "
            f"{tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[1] != fleet_params.token_hidden_size:
        raise MoEEpConfigError(
            f"hidden_states.shape[1] ({hidden_states.shape[1]}) does not match "
            f"FleetParams.token_hidden_size ({fleet_params.token_hidden_size})"
        )
    if topk_ids.shape[0] != num_tokens:
        raise MoEEpConfigError(
            f"topk_ids batch dim ({topk_ids.shape[0]}) does not match "
            f"hidden_states batch dim ({num_tokens})"
        )
    if topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("topk_weights and topk_ids must have the same shape")


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
        return
    cc = _device_capability()
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


def validate_mega_arch() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    cc = _device_capability()
    if cc < (10, 0):
        raise MoEEpArchError(
            f"mega_moe requires sm_100+ (Blackwell); host has sm_{cc[0]}{cc[1]}"
        )


def validate_fleet_weights(
    weights: MoEWeightPack, params: FleetParams, world_size: int
) -> None:
    """Check canonical weight layout matches EP sizing for this rank."""
    if world_size <= 0:
        raise MoEEpConfigError(f"world_size must be positive, got {world_size}")
    if params.num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({params.num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )
    local = params.num_experts // world_size
    pack = weights
    if not isinstance(pack, MoEWeightPack):
        raise MoEEpConfigError(
            f"layer weights must be MoEWeightPack, got {type(pack).__name__}"
        )
    hidden = params.token_hidden_size
    for name in ("w13", "w2"):
        t = getattr(pack, name)
        if t.ndim < 2:
            raise MoEEpConfigError(
                f"MoEWeightPack.{name} must be at least 2D, got shape {tuple(t.shape)}"
            )
        if t.shape[0] != local:
            raise MoEEpConfigError(
                f"MoEWeightPack.{name}.shape[0] ({t.shape[0]}) != "
                f"num_experts // world_size ({local})"
            )
    w13_hidden = pack.w13.shape[-1]
    if w13_hidden not in (hidden, hidden // 2):
        raise MoEEpConfigError(
            f"MoEWeightPack.w13 hidden dim ({w13_hidden}) does not match "
            f"token_hidden_size ({hidden})"
        )
    w2_hidden = pack.w2.shape[1] if pack.w2.ndim >= 2 else None
    if w2_hidden is not None and w2_hidden not in (hidden, hidden // 2):
        raise MoEEpConfigError(
            f"MoEWeightPack.w2 hidden dim ({w2_hidden}) does not match "
            f"token_hidden_size ({hidden})"
        )


def validate_mega_fleet_params(
    params: FleetParams,
    world_size: int,
    *,
    intermediate_size: int,
    top_k: int,
) -> None:
    if world_size <= 0:
        raise MoEEpConfigError(f"world_size must be positive, got {world_size}")
    if params.num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({params.num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )
    if params.token_hidden_size % 128 != 0:
        raise MoEEpConfigError(
            f"token_hidden_size ({params.token_hidden_size}) must be a multiple of 128"
        )
    if intermediate_size % 128 != 0:
        raise MoEEpConfigError(
            f"intermediate_size ({intermediate_size}) must be a multiple of 128"
        )
    if top_k <= 0:
        raise MoEEpConfigError(f"top_k must be positive, got {top_k}")


def validate_mega_forward_inputs(
    hidden_states: "torch.Tensor",
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    fleet_params: FleetParams,
    *,
    top_k: int,
    quantize_input: bool,
    scales: "torch.Tensor | None" = None,
) -> None:
    num_tokens = hidden_states.shape[0]
    if num_tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError(
            f"token count {num_tokens} exceeds "
            f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}"
        )
    if hidden_states.ndim != 2:
        raise MoEEpConfigError(
            f"hidden_states must be 2D [num_tokens, hidden], got shape "
            f"{tuple(hidden_states.shape)}"
        )
    if hidden_states.shape[1] != fleet_params.token_hidden_size:
        raise MoEEpConfigError(
            f"hidden_states.shape[1] ({hidden_states.shape[1]}) does not match "
            f"FleetParams.token_hidden_size ({fleet_params.token_hidden_size})"
        )
    if topk_ids.shape[0] != num_tokens:
        raise MoEEpConfigError(
            f"topk_ids batch dim ({topk_ids.shape[0]}) does not match "
            f"hidden_states batch dim ({num_tokens})"
        )
    if topk_ids.shape[1] != top_k:
        raise MoEEpConfigError(
            f"topk_ids.shape[1] ({topk_ids.shape[1]}) does not match "
            f"kernel top_k ({top_k})"
        )
    if topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("topk_weights and topk_ids must have the same shape")
    if not quantize_input and scales is None:
        raise MoEEpConfigError(
            "MoEEpTensors.scales is required when MegaConfig.quantize_input=False"
        )


def validate_fleet_params(
    params: FleetParams,
    backend: str,
    world_size: int,
    quant: "FleetAlgoKnobQuantization | None" = None,
    topology_capacity: int | None = None,
) -> None:
    import torch

    if backend == "nixl_ep":
        if params.algorithm is not EpAlgorithm.LOW_LATENCY:
            raise MoEEpConfigError(
                "nixl_ep: only algorithm=LOW_LATENCY is supported "
                "(HIGH_THROUGHPUT requires nccl_ep)."
            )
        if params.layout is not EpLayout.EXPERT_MAJOR:
            raise MoEEpConfigError(
                "nixl_ep: only layout=EXPERT_MAJOR is supported "
                "(RANK_MAJOR requires nccl_ep)."
            )
        cap = topology_capacity if topology_capacity is not None else world_size
        if cap <= 0:
            raise MoEEpConfigError(
                f"nixl_ep: topology capacity ({cap}) must be positive"
            )
        if params.num_experts % cap != 0:
            raise MoEEpConfigError(
                f"nixl_ep: num_experts ({params.num_experts}) must be a positive "
                f"multiple of topology capacity ({cap})"
            )
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
                cc = _device_capability()
                if cc < (10, 0):
                    raise MoEEpConfigError(
                        f"nixl_ep: UE8M0 quantization requires sm_100+ (Blackwell); "
                        f"host has sm_{cc[0]}{cc[1]}"
                    )
