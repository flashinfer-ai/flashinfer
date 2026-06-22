"""Shared validation helpers for moe_ep_v2 layers and comm backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...config import BootstrapConfig, EpAlgorithm, EpLayout, FleetParams, QuantType

if TYPE_CHECKING:
    import torch

    from ...algo_knobs import FleetAlgoKnobQuantization


_NIXL_EP_SUPPORTED_HIDDEN_SIZES = frozenset(
    {2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}
)
_NIXL_EP_MAX_TOKENS_PER_RANK = 1024


class MoEEpConfigError(ValueError):
    """Raised when an EP config field is out-of-range for the chosen backend."""


class MoEEpArchError(MoEEpConfigError):
    """Raised when the GPU arch doesn't support the chosen backend."""


def _device_capability() -> tuple[int, int]:
    import torch

    return torch.cuda.get_device_capability(torch.cuda.current_device())


def validate_bootstrap_world_size(bootstrap: BootstrapConfig) -> None:
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    dist_ws = dist.get_world_size()
    if bootstrap.world_size != dist_ws:
        raise MoEEpConfigError(
            f"BootstrapConfig.world_size ({bootstrap.world_size}) must match "
            f"torch.distributed world size ({dist_ws})"
        )


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


def validate_arch_for_backend(backend: str) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    cc = _device_capability()
    if cc < (9, 0):
        raise MoEEpArchError(f"{backend} requires sm_90+, host has sm_{cc[0]}{cc[1]}")


def validate_mega_arch() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    cc = _device_capability()
    if cc < (10, 0):
        raise MoEEpArchError(
            f"mega_moe requires sm_100+ (Blackwell); host has sm_{cc[0]}{cc[1]}"
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
    stage_inputs: bool,
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
    if not stage_inputs and scales is None:
        raise MoEEpConfigError(
            "MoEEpTensors.scales is required when MegaConfig.stage_inputs=False"
        )


def validate_fleet_params(
    params: FleetParams,
    backend: str,
    world_size: int,
    quant: "FleetAlgoKnobQuantization | None" = None,
    topology_capacity: int | None = None,
) -> None:
    import torch

    if params.num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({params.num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )

    if (
        params.layout is EpLayout.RANK_MAJOR
        and params.algorithm is not EpAlgorithm.LOW_LATENCY
    ):
        raise MoEEpConfigError(
            "FleetParams.layout=RANK_MAJOR is only valid with "
            "algorithm=LOW_LATENCY (HT uses the FLAT layout)."
        )

    if backend == "nixl_ep":
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
