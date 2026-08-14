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

    ``mxfp8_dispatch`` quantizes tokens BEFORE EP dispatch and sends one
    packed row per token (``[H]`` fp8 payload + ``[H/32]`` UE8M0 scale bytes,
    zero-padded to the nearest transport-supported width and viewed as bf16
    — see ``packed_dispatch_width``); ``compute()`` unpacks instead of
    re-quantizing. Wire bytes vs plain BF16 dispatch: 0.57x at H=7168,
    0.625x at H=4096/8192, no saving at H<=2048. Numerically identical to
    the default post-dispatch quantization (the same per-token rows produce
    the same MXFP8 codes).
    """

    kernel_name: str = "sm100_mxfp8_mxfp4_bf16_cutedsl"
    enable_pdl: bool = True
    tactic: Optional[Tuple[Any, ...]] = None
    mxfp8_dispatch: bool = False
