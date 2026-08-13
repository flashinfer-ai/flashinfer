"""CuTeDSL W4A8 split kernel config — MXFP8 activations x MXFP4 weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig:
    """Kernel params for the ``cute_dsl_fused_moe_mxfp8_mxfp4`` inner compute.

    Expert geometry is derived at runtime: the local expert count and the
    hidden/intermediate sizes come from the canonical bf16 ``MoEWeightPack``
    at ``preprocess_weights()``, the global expert count from ``FleetParams``,
    and the rank offset from ``BootstrapConfig`` at ``validate_init()``.

    ``tactic`` pins a kernel tactic tuple (see
    ``flashinfer.fused_moe.cute_dsl.mixed_tuner``); ``None`` defers to the
    AutoTuner (default tactic outside a tuning context).
    """

    kernel_name: str = "sm100_mxfp8_mxfp4_bf16_cutedsl"
    enable_pdl: bool = True
    tactic: Optional[Tuple[Any, ...]] = None
