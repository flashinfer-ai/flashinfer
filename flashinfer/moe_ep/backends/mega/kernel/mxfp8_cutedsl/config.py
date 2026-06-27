"""CuTeDSL MXFP8 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Mxfp8CutedslMegaMoeConfig:
    """Kernel params for ``cutedsl_backend_kernels.frontend.mxfp8_mega_moe``.

    Expert weights must be MXFP8 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    fp8 weights with ``w13_scale`` / ``w2_scale``.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "mxfp8_cutedsl"
    kind: Literal["mxfp8_e4m3", "mxfp8_e5m2"] = "mxfp8_e4m3"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    in_kernel_fc2_reduce: bool = False
    token_back_by_dispatch: bool = False
