"""Mode-level config: comm + kernel composition for split and mega paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..backends.split.comm.nccl_ep.config import NcclEpConfig
from ..backends.split.kernel.identity.config import IdentityConfig

__all__ = [
    "AutoConfig",
    "IdentityConfig",
    "MegaConfig",
    "SplitConfig",
]


@dataclass
class SplitConfig:
    """Dispatch → inner kernel → combine.

    ``comm`` selects a Fleet transport (``NcclEpConfig``, ``NvepConfig``) or a
    :class:`~flashinfer.moe_ep.core.comm.communication.MoEEpCommunication`
    backend (``NVLinkOneSidedConfig``, ``CakeAlltoAllConfig``,
    ``NVLinkTwoSidedConfig``).
    """

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


@dataclass
class AutoConfig:
    """Select a complete EP path from explicitly configured candidates.

    Each split candidate keeps its configured transport. ``candidate`` pins one
    entry while still allowing tuning within it. An unset value compares all
    eligible entries during an enabled ``autotune_v2()`` warmup.
    """

    candidates: tuple[MegaConfig | SplitConfig, ...]
    candidate: int | None = None

    def __post_init__(self):
        self.candidates = tuple(self.candidates)
        if not self.candidates or any(
            not isinstance(c, (MegaConfig, SplitConfig)) for c in self.candidates
        ):
            raise ValueError(
                "AutoConfig requires explicit MegaConfig/SplitConfig candidates"
            )
        if self.candidate is not None and (
            type(self.candidate) is not int
            or not 0 <= self.candidate < len(self.candidates)
        ):
            raise ValueError("candidate must index AutoConfig.candidates")
