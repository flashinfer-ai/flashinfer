"""Shape-qualified launch tuning for the fused dynamic MoE kernel."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicLaunchTuning:
    """Compile-time scheduler and logical resident-cluster selection."""

    work_source: str
    max_active_clusters: int


# The logical cluster count is doubled by the two-resident-CTA W4A8 launcher.
# Every entry therefore stays at or below 94 on an SM120 device with 188 SMs.
_SM120_E256_K4096_N512_TOP6_SILU_LADDER = (
    (96, 94),
    (144, 72),
    (192, 84),
    (768, 64),
)

# TP2 retains 1024 intermediate columns per rank.  CUDA-event measurements of
# the complete fused kernel select the full 94-cluster resident domain at C8,
# C16, and C32.  Rows above C32 remain unqualified and preserve the generic
# scheduler.
_SM120_E256_K4096_N1024_TOP6_SILU_LADDER = ((192, 94),)


def select_dynamic_launch_tuning(
    *,
    compute_capability: tuple[int, int] | None,
    quant_mode: str,
    activation: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    routed_rows: int,
    qmma_repacked: bool,
    deterministic_output: bool,
) -> DynamicLaunchTuning | None:
    """Return measured tuning only for its complete execution contract.

    The selected SM120 W4A8 kernel has two resident CTAs per SM.  Above 188
    physical CTAs, its resident-grid synchronization and task geometry become
    slower for this expert shape.  Persistent arithmetic task assignment and
    the bounded logical cluster counts below avoid that cliff.  Smaller decode
    batches retain the materialized queue because it is faster for their short
    work domain.
    """

    if (
        compute_capability != (12, 0)
        or quant_mode != "w4a8_mx"
        or activation != "silu"
        or num_experts != 256
        or hidden_size != 4096
        or top_k != 6
        or not qmma_repacked
        or deterministic_output
        or routed_rows <= 24
    ):
        return None

    ladders = {
        512: _SM120_E256_K4096_N512_TOP6_SILU_LADDER,
        1024: _SM120_E256_K4096_N1024_TOP6_SILU_LADDER,
    }
    ladder = ladders.get(intermediate_size)
    if ladder is None:
        return None

    for end_routed_rows, max_active_clusters in ladder:
        if routed_rows <= end_routed_rows:
            return DynamicLaunchTuning(
                work_source="persistent_grid",
                max_active_clusters=max_active_clusters,
            )
    return None


__all__ = ["DynamicLaunchTuning", "select_dynamic_launch_tuning"]
