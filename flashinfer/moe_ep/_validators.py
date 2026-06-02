"""Backend-specific FleetParams / quant validators.

Backend Fleet __init__ calls into these before touching the C ABI so a
config error surfaces immediately rather than as a cryptic kernel-launch
failure later.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import FleetParams, QuantType

if TYPE_CHECKING:
    from .algo_knobs import FleetAlgoKnobQuantization


# NIXL EP's kernels are JIT-compiled only for these hidden sizes; the wrapper
# rounds up the layer's hidden dimension to the nearest supported value. The
# canonical set is from vLLM's NixlEPPrepareAndFinalize.
_NIXL_EP_SUPPORTED_HIDDEN_SIZES = frozenset(
    {2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192}
)

# NIXL EP's `FINISHED_SUM_TAG` is hard-coded to 1024 in the kernel.
_NIXL_EP_MAX_TOKENS_PER_RANK = 1024


class MoEEpConfigError(ValueError):
    """Raised when an EP config field is out-of-range for the chosen backend."""


class MoEEpArchError(MoEEpConfigError):
    """Raised when the GPU arch doesn't support the chosen backend."""


def validate_arch_for_backend(backend: str) -> None:
    """Check ``torch.cuda.get_device_capability(0)`` is supported by `backend`."""
    import torch

    if not torch.cuda.is_available():
        return  # Mock/test path — let backend probes catch missing libs instead.
    cc = torch.cuda.get_device_capability(0)
    # Both nccl_ep and nixl_ep require sm_90+.
    if cc < (9, 0):
        raise MoEEpArchError(f"{backend} requires sm_90+, host has sm_{cc[0]}{cc[1]}")


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
