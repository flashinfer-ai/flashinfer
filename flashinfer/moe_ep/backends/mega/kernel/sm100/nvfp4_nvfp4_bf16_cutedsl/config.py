"""CuTeDSL NVFP4 mega-MoE kernel config."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

from flashinfer.moe_ep._compat import deprecated_aliases, legacy_finalize
from flashinfer.moe_ep.config import FinalizeMode, validate_finalize_mode

if TYPE_CHECKING:
    import torch


@legacy_finalize
@deprecated_aliases(apply_topk_in_fc1="apply_routing_weights_before_fc2")
@dataclass
class Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm100.cutedsl_megamoe.nvfp4_mega_moe``.

    Expert weights must be NVFP4 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    NVFP4 weights with ``w13_scale`` / ``w2_scale``.

    Set ``swiglu_alpha=1.702`` and ``swiglu_beta=1.0`` for MiniMax-M3:
    ``(up + beta) * gate * sigmoid(alpha * gate)``, after ``gate_up_clamp``.
    These are scalar activation constants, distinct from ``fc1_alpha``.

    ``kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next"`` selects the standard
    provider export. It accepts default or fixed ``knobs``, activation controls
    and the finalization modes below. With ``knobs="auto"``, warm up inside
    ``autotune_v2()`` before capture to select and prepare a configuration.
    ``kernel_variant="genphase"`` selects decoding with separate BF16
    finalization and a K256/N128/CGA4 Gather4 tile. It supports standard SwiGLU
    and early or late routing weights, without per-token FC1 scales.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_nvfp4_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: float | None = None
    situ_linear_beta: float | None = None
    fast_math: bool = True
    apply_routing_weights_before_fc2: bool = True
    # Enables in_kernel_fc2_reduce, knobs may still disable this if it is faster
    # In-flight top-k combine: cross-rank REDG atomic-add collapses the combine
    # as peer data arrives (no per-topk staging / explicit tail reduce).
    # ~1-2% faster and removes the multi-GB combine staging from the symmetric
    # workspace.  Requires apply_routing_weights_before_fc2=True and
    # combine_dtype="bf16"; accumulation order is nondeterministic
    # (tolerance-compare outputs).
    enable_in_kernel_fc2_reduce: bool = False
    # Cross-rank combine wire format: "bf16" (exact), "mxfp8" (2x less combine
    # traffic), "nvfp4" (4x less).  Quantized wires trade accuracy for NVLink
    # bandwidth and require enable_in_kernel_fc2_reduce=False.
    combine_dtype: Literal["bf16", "mxfp8", "nvfp4"] = "bf16"
    # Caller-owned CUDA int32 (1,) live-token count the PERSISTENT finalize kernel
    # reads; the workspace borrows it and never writes it. Update before every
    # forward.
    num_valid_tokens_tensor: Optional["torch.Tensor"] = field(
        default=None, kw_only=True
    )
    input_norm_const: float = 1.0
    fc1_alpha: Optional["torch.Tensor"] = None
    fc2_alpha: Optional["torch.Tensor"] = None
    fc1_norm_const: Optional["torch.Tensor"] = None
    # Per-token fc1 activation scale (compile-time switch).  When True the
    # workspace carries a ``(max_tokens_per_rank,)`` fp32 symmetric-heap tensor
    # and every forward must supply ``MoEEpTensors.fc1_activation_per_token_scale``
    # (``(num_tokens,)`` fp32).  The fc1 epilogue multiplies it into the
    # dequantized fc1 output before the clamp / gated activation, so callers
    # that quantize ``x_t / s_t`` to NVFP4 recover the per-token range in-kernel.
    enable_fc1_activation_per_token_scale: bool = False
    # Kernel tuning knobs (see kernel_src.sm100.cutedsl_megamoe.shim.tuner); overrides
    # the token-count default heuristic entirely when set, e.g. a winner from the
    # kernel repo's tester sweep. None -> tuner.default_knobs(num_max_tokens).
    # Legacy "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    # The standard _cutedsl_next selector defers searches to autotune_v2().
    knobs: dict | str | None = None
    # GenPhase is available through the _cutedsl_next selector.
    kernel_variant: Literal["inference", "genphase"] = field(
        default="inference", kw_only=True
    )
    mma_tiler_mnk: tuple[int, int, int] | None = field(default=None, kw_only=True)
    cluster_shape_mn: tuple[int, int] | None = field(default=None, kw_only=True)
    schedule_policy: tuple[str, int | None] | None = field(default=None, kw_only=True)
    # Uniform per-layer SwiGLU constants; None/None keeps standard SwiGLU.
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    # NONE exposes per-route outputs through forward_unfinalized(). The legacy
    # selector requires fixed/offline-tuned knobs for this mode.
    finalize_mode: FinalizeMode = FinalizeMode.AUTO

    def __post_init__(self) -> None:
        validate_finalize_mode(
            self.finalize_mode, tuple(FinalizeMode), self.num_valid_tokens_tensor
        )
        if self.finalize_mode is FinalizeMode.NONE:
            if self.enable_in_kernel_fc2_reduce or self.combine_dtype != "bf16":
                raise ValueError(
                    "finalize_mode=NONE requires enable_in_kernel_fc2_reduce=False "
                    "and combine_dtype='bf16'."
                )
            if (
                self.knobs == "auto"
                and self.kernel_name != "sm100_nvfp4_nvfp4_bf16_cutedsl_next"
            ):
                raise ValueError(
                    "finalize_mode=NONE requires fixed/offline-tuned knobs."
                )
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be set together.")
        if self.activation not in ("swiglu", "situ"):
            raise ValueError(
                f"activation must be 'swiglu' or 'situ', got {self.activation!r}."
            )
        if self.activation == "situ":
            if self.swiglu_alpha is not None:
                raise ValueError("SwiGLU parameters are not supported with SiTU.")
            if self.situ_beta is None:
                raise ValueError("activation='situ' requires situ_beta.")
            if not math.isfinite(self.situ_beta) or self.situ_beta <= 0:
                raise ValueError("situ_beta must be positive and finite.")
            if self.situ_linear_beta is not None and (
                not math.isfinite(self.situ_linear_beta) or self.situ_linear_beta <= 0
            ):
                raise ValueError(
                    "situ_linear_beta must be positive and finite when set."
                )
            if self.gate_up_clamp is not None or self.activation_clamp is not None:
                raise ValueError("activation clamps are not supported with SiTU.")
        elif self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError("SiTU parameters require activation='situ'.")
