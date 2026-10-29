"""CuTeDSL MXFP8 mega-MoE kernel config."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from flashinfer.moe_ep.config import FinalizeMode, validate_finalize_mode

if TYPE_CHECKING:
    import torch


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
    It supports E4M3, E5M2 and ``kind="mxfp4_mxfp8"`` (packed MXFP4 weights
    with E4M3 activations). The latter requires 128-aligned intermediate size.
    Its ``knobs=None`` selects an exact cached profile or a host heuristic;
    dictionaries and explicit tile/cluster/schedule fields constrain selection.
    GenPhase and the standard ``_cutedsl_next`` selector use ``knobs="auto"``
    with ``autotune_v2()`` for collective warmup before graph capture.
    These selectors support independent automatic handles through
    ``create_workspace()``. The legacy inference selector retains its
    single-capacity automatic workspace.

    ``kernel_name="sm100_mxfp8_mxfp8_bf16_cutedsl_next"`` selects the standard
    provider export for E4M3 or E5M2. ``kernel_variant="genphase"`` selects
    decoding, including mixed MXFP4/MXFP8. With ``kernel_variant="inference"``,
    use default, fixed or automatic ``knobs``. Activation controls and
    late routing weights are available through this selector.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_mxfp8_mxfp8_bf16_cutedsl"
    kind: Literal["mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"] = "mxfp8_e4m3"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    # Enables in_kernel_fc2_reduce, autotune may still disable this if it is faster
    enable_in_kernel_fc2_reduce: bool = False
    # NONE exposes per-route outputs through forward_unfinalized(). The legacy
    # selector requires fixed/offline-tuned knobs for this mode.
    finalize_mode: FinalizeMode = field(default=FinalizeMode.AUTO, kw_only=True)
    # Caller-owned CUDA int32 (1,) live-token count the PERSISTENT finalize kernel
    # reads; the workspace borrows it and never writes it. Update before every
    # forward.
    num_valid_tokens_tensor: Optional["torch.Tensor"] = field(
        default=None, kw_only=True
    )
    # GenPhase: None resolves the cache or geometry heuristic; dictionaries
    # constrain tile/cluster/schedule_policy; "auto" searches within autotune_v2().
    # Inference tuning knobs (see kernel_src.sm100.cutedsl_megamoe.shim.tuner); overrides
    # the token-count default heuristic entirely when set, e.g. a winner from the
    # kernel repo's tester sweep. None -> tuner.default_knobs(..., dtype="mxfp8").
    # Legacy "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    # The standard _cutedsl_next selector defers searches to autotune_v2().
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

    activation: Literal["swiglu", "situ"] = field(default="swiglu", kw_only=True)
    situ_beta: float | None = field(default=None, kw_only=True)
    situ_linear_beta: float | None = field(default=None, kw_only=True)
    swiglu_alpha: float | None = field(default=None, kw_only=True)
    swiglu_beta: float | None = field(default=None, kw_only=True)
    apply_routing_weights_before_fc2: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Reject incompatible dtype, activation and finalization options."""
        if self.kind == "mxfp4_mxfp8" and self.kernel_variant != "genphase":
            raise ValueError("Mixed MXFP4/MXFP8 requires kernel_variant='genphase'.")
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be set together.")
        if self.swiglu_alpha is not None and self.swiglu_beta is not None:
            if not all(math.isfinite(x) for x in (self.swiglu_alpha, self.swiglu_beta)):
                raise ValueError("SwiGLU constants must be finite.")
        if self.activation not in ("swiglu", "situ"):
            raise ValueError("activation must be 'swiglu' or 'situ'.")
        if self.activation == "situ":
            if self.swiglu_alpha is not None:
                raise ValueError("SwiGLU constants are not supported with SiTU.")
            if (
                self.situ_beta is None
                or not math.isfinite(self.situ_beta)
                or self.situ_beta <= 0
            ):
                raise ValueError("SiTU requires a positive, finite situ_beta.")
            if self.situ_linear_beta is not None and (
                not math.isfinite(self.situ_linear_beta) or self.situ_linear_beta <= 0
            ):
                raise ValueError(
                    "situ_linear_beta must be positive and finite when set."
                )
            if self.gate_up_clamp is not None or self.activation_clamp is not None:
                raise ValueError("Activation clamps are not supported with SiTU.")
        elif self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError("SiTU parameters require activation='situ'.")
        validate_finalize_mode(
            self.finalize_mode,
            (
                FinalizeMode.AUTO,
                FinalizeMode.DEFAULT,
                FinalizeMode.PERSISTENT,
                FinalizeMode.NONE,
            ),
            self.num_valid_tokens_tensor,
        )
        if self.finalize_mode is FinalizeMode.NONE:
            if self.enable_in_kernel_fc2_reduce:
                raise ValueError(
                    "finalize_mode=NONE requires enable_in_kernel_fc2_reduce=False."
                )
            if (
                self.knobs == "auto"
                and self.kernel_name != "sm100_mxfp8_mxfp8_bf16_cutedsl_next"
            ):
                raise ValueError(
                    "finalize_mode=NONE requires fixed/offline-tuned knobs."
                )

    @property
    def resolved_finalize_mode(self) -> FinalizeMode:
        if self.finalize_mode is FinalizeMode.AUTO:
            return (
                FinalizeMode.DEFAULT
                if self.num_valid_tokens_tensor is None
                else FinalizeMode.PERSISTENT
            )
        return self.finalize_mode
