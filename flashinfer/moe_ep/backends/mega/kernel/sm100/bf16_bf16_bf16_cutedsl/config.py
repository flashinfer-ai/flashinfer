"""CuTeDSL BF16 MegaMoE kernel configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashinfer.moe_ep._compat import legacy_finalize
from flashinfer.moe_ep.config import FinalizeMode, validate_finalize_mode

from ..common.bf16_config import Sm100_Bf16_Cutedsl_MegaMoeConfigBase


@legacy_finalize
@dataclass
class Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(Sm100_Bf16_Cutedsl_MegaMoeConfigBase):
    """Parameters for the SM100 BF16 CuTeDSL MegaMoE kernel.

    ``batch_invariant=True`` restricts execution to the fixed BF16 MMA fold
    and slot-order FP32 finalizer. Routing ids and weights are supplied by
    the caller; their computation is outside this guarantee. The routing
    weight placement is a numerical parameter, never an autotuning axis.
    """

    kernel_name: str = "sm100_bf16_bf16_bf16_cutedsl"
    # NONE makes MoEEpMegaLayer.forward return per-route outputs.
    finalize_mode: FinalizeMode = FinalizeMode.AUTO
    # False weights the rounded BF16 FC2 rows in the FP32 top-k finalizer.
    apply_routing_weights_before_fc2: bool = True
    batch_invariant: bool = field(default=False, kw_only=True)
    activation: str = field(default="swiglu", kw_only=True)
    swiglu_alpha: float | None = field(default=None, kw_only=True)
    swiglu_beta: float | None = field(default=None, kw_only=True)
    situ_beta: float | None = field(default=None, kw_only=True)
    situ_linear_beta: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.batch_invariant) is not bool:
            raise ValueError("batch_invariant must be a bool.")
        if not self.batch_invariant:
            from ..common.activation_batch_invariant import (
                batch_invariant_activation_fields,
            )

            if batch_invariant_activation_fields(self):
                raise ValueError(
                    "BF16 activation variants require batch_invariant=True"
                )
        if self.batch_invariant:
            if self.kernel_name not in (
                "sm100_bf16_bf16_bf16_cutedsl",
                "bf16_cutedsl",
            ):
                raise ValueError(
                    "batch_invariant=True only supports the SM100 BF16 "
                    "CuTeDSL MegaMoE backend."
                )
            from ..common.batch_invariant import (
                validate_bf16_batch_invariant,
            )

            if self.finalize_mode not in (
                FinalizeMode.AUTO,
                FinalizeMode.DEFAULT,
                FinalizeMode.NONE,
            ):
                raise ValueError(
                    "batch_invariant=True requires the slot-order finalizer "
                    "(finalize_mode=AUTO, DEFAULT, or NONE)."
                )
            if (
                self.knobs is not None
                and self.knobs != "auto"
                and not isinstance(self.knobs, dict)
            ):
                raise ValueError("knobs must be a dictionary, 'auto', or None.")
            validate_bf16_batch_invariant(
                self, self.knobs if isinstance(self.knobs, dict) else None
            )
        validate_finalize_mode(
            self.finalize_mode,
            (FinalizeMode.AUTO, FinalizeMode.DEFAULT, FinalizeMode.NONE),
        )
        if self.finalize_mode is FinalizeMode.NONE:
            if self.enable_in_kernel_fc2_reduce:
                raise ValueError(
                    "finalize_mode=NONE requires enable_in_kernel_fc2_reduce=False."
                )
            if self.knobs == "auto":
                raise ValueError(
                    "finalize_mode=NONE requires fixed/offline-tuned knobs."
                )
