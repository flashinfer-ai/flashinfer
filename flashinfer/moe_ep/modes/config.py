"""Mode-level config: comm + kernel composition for split and mega paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backends.split.comm.nccl_ep.config import NcclEpConfig
from ..backends.split.kernel.identity.config import IdentityConfig

__all__ = [
    "IdentityConfig",
    "MegaConfig",
    "SplitConfig",
]


@dataclass
class SplitConfig:
    """Dispatch → inner kernel → combine over NCCL-EP / NIXL-EP."""

    comm: object = field(default_factory=NcclEpConfig)
    kernel: object = field(default_factory=IdentityConfig)


@dataclass
class MegaConfig:
    """Fused expert-parallel mega kernel (symmetric memory).

    ``megakernel`` is the registered kernel config (backend, geometry, knobs).

    ``quantize_input`` selects the activation format. ``True``: ``hidden_states``
    are BF16 and the backend quantizes while staging. ``False``: they are
    already quantized and the quantized weights and ``MoEEpTensors.scales`` will be provided. BF16-activation kernels ignore this flag and always copy BF16.

    ``preprocess_weights`` selects the weight format. ``True``: preprocess the ``MoEWeightPack`` into kernel's required layout at init. ``False``: the
    caller supplies kernel-ready ``transformed_weights`` instead.

    ``transformed_weights`` is that kernel-ready FC1/FC2 payload. It is
    required when ``preprocess_weights`` is False and unused otherwise.
    """

    megakernel: object
    quantize_input: bool = True
    preprocess_weights: bool = True
    # TODO: This should probably just be passed via a top level parameter
    # when preprocess_weights is False, instead of the regular weights
    transformed_weights: object | None = None
