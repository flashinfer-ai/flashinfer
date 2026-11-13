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


def _batch_invariant_numerical_contract(config: MegaConfig | SplitConfig):
    """Identify the supported batch-invariant calculation before backend setup."""
    if not isinstance(config, MegaConfig) or not getattr(
        config.megakernel, "batch_invariant", False
    ):
        return None

    from ..backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.config import (
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
    )
    from ..backends.mega.kernel.sm100.common.batch_invariant import (
        BF16_BATCH_INVARIANT_VERSION,
    )
    from ..batch_invariant.spec import f32_bits

    kernel = config.megakernel
    if type(kernel) is not Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig or (
        kernel.kernel_name not in ("sm100_bf16_bf16_bf16_cutedsl", "bf16_cutedsl")
    ):
        raise ValueError(
            "Batch-invariant EP selection only recognizes the SM100 BF16 CuTeDSL backend"
        )
    clamp = (
        kernel.gate_up_clamp
        if kernel.gate_up_clamp is not None
        else kernel.activation_clamp
    )
    return {
        "family": "sm100_bf16_cutedsl",
        "version": BF16_BATCH_INVARIANT_VERSION,
        "intermediate": kernel.intermediate_size,
        "top_k": kernel.top_k,
        "activation": "swiglu",
        "gate_up_clamp_bits": None if clamp is None else f32_bits(abs(clamp)),
        "apply_routing_weights_before_fc2": kernel.apply_routing_weights_before_fc2,
    }


def _validate_batch_invariant_candidates(candidates):
    contracts = [_batch_invariant_numerical_contract(c) for c in candidates]
    if not any(contract is not None for contract in contracts):
        return
    if any(contract is None for contract in contracts):
        raise ValueError(
            "AutoConfig cannot mix batch-invariant candidates with ordinary or "
            "unsupported candidates"
        )
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError(
            "Batch-invariant AutoConfig candidates must use the same calculation: "
            "expert geometry, activation clamp and routing-weight placement"
        )


@dataclass
class AutoConfig:
    """Select a complete EP path from explicitly configured candidates.

    Each split candidate keeps its configured transport. ``candidate`` pins one
    entry while still allowing tuning within it. An unset value compares all
    eligible entries during an enabled ``autotune_v2()`` warmup.

    Batch-invariant candidates cannot be mixed with ordinary candidates or
    different calculations, including when ``candidate`` pins one entry.
    Backends must separately implement complete-EP selection.
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
        _validate_batch_invariant_candidates(self.candidates)
