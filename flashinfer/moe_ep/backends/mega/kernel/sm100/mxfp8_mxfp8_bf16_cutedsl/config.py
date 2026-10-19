"""CuTeDSL MXFP8 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm100.cutedsl_megamoe.mxfp8_mega_moe``.

    ``intermediate_size`` is the post-SwiGLU width, matching NVFP4 and SGLang.
    The MXFP8 kernel's full FC1 gate+up width is derived internally as
    ``2 * intermediate_size``.

    Expert weights must be MXFP8 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    fp8 weights with ``w13_scale`` / ``w2_scale``.

    ``kernel_variant="genphase"`` opts into the Blackwell decoding pipeline.
    Its ``knobs=None`` selects an exact cached profile or a host heuristic;
    dictionaries and explicit tile/cluster/schedule fields constrain selection.
    ``knobs="auto"`` tunes collectively during warmup before graph capture.
    Auto layers own a single capacity and cannot call ``create_workspace()``;
    use fixed or offline-tuned knobs for additional capacity profiles.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_mxfp8_mxfp8_bf16_cutedsl"
    kind: Literal["mxfp8_e4m3", "mxfp8_e5m2"] = "mxfp8_e4m3"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    # Enables in_kernel_fc2_reduce, autotune may still disable this if it is faster
    enable_in_kernel_fc2_reduce: bool = False
    # GenPhase: None resolves the cache or geometry heuristic; dictionaries
    # constrain tile/cluster/schedule_policy; "auto" tunes at collective warmup.
    # Inference tuning knobs (see kernel_src.sm100.cutedsl_megamoe.shim.tuner); overrides
    # the token-count default heuristic entirely when set, e.g. a winner from the
    # kernel repo's tester sweep. None -> tuner.default_knobs(..., dtype="mxfp8").
    # "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    knobs: dict | str | None = None
    # Opt-in decoding pipeline. Existing callers keep the legacy inference
    # path, including its knob cache and online autotuner.
    kernel_variant: Literal["inference", "genphase"] = "inference"
    # GenPhase only: None selects cached knobs or geometry-based heuristics.
    # Explicit fields constrain the heuristic and automatic candidate sweep.
    mma_tiler_mnk: tuple[int, int, int] | None = None
    cluster_shape_mn: tuple[int, int] | None = None
    # GenPhase only: None uses the geometry-based scheduling heuristic. An explicit
    # ("grouped", None) preserves the kernel's original launch-cluster hint;
    # ("grouped", positive_int) overrides it for offline tuning.
    schedule_policy: tuple[str, int | None] | None = None
