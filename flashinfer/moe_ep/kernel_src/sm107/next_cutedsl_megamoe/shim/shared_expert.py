"""Optional Rubin shared-expert geometry and format."""

from __future__ import annotations

from flashinfer.moe_ep.config import Sm107SharedExpertConfig


def validate_feature_config(config):
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
