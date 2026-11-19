"""Optional Rubin shared-expert geometry and format."""

from __future__ import annotations

from flashinfer.moe_ep.config import Sm107SharedExpertConfig
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.activation import (
    fc1_rows_per_intermediate,
)


def validate_feature_config(config):
    fc1_rows_per_intermediate(config.activation)
    if getattr(config, "activation", None) == "clamped_relu2":
        kind = getattr(config, "quant_kind", getattr(config, "kind", None))
        if (
            kind != "mxfp8_e4m3"
            or config.kernel_variant != "genphase"
            or getattr(config, "batch_invariant", False)
            or config.shared_expert is not None
            or config.combine_dedup
            or config.epilogue_warpgroups != 1
            or config.combine_dtype != "bf16"
        ):
            raise ValueError(
                "clamped_relu2 requires MXFP8 E4M3 GenPhase, BF16 combine, "
                "one epilogue warpgroup, no shared expert/dedup/BI"
            )
        if (
            getattr(config, "gate_up_clamp", None) is not None
            or getattr(config, "activation_clamp", None) is not None
            or config.situ_beta is not None
            or config.situ_linear_beta is not None
            or config.use_tanh_sigmoid
        ):
            raise ValueError("clamped_relu2 uses a fixed limit of 16 and hardware tanh")
    shared = getattr(config, "shared_expert", None)
    if shared is not None and not isinstance(shared, Sm107SharedExpertConfig):
        raise ValueError("shared_expert must be Sm107SharedExpertConfig or None")
    if type(config.combine_dedup) is not bool:
        raise ValueError("combine_dedup must be a bool")
    if getattr(config, "batch_invariant", False) and (
        shared is not None or config.combine_dedup
    ):
        raise ValueError(
            "shared fusion and dedup do not implement the batch-invariant contract"
        )
