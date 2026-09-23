"""CuTeDSL NVFP4 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import torch


@dataclass
class Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.sm100.cutedsl_megamoe.nvfp4_mega_moe``.

    Expert weights must be NVFP4 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    NVFP4 weights with ``w13_scale`` / ``w2_scale``.

    Set ``swiglu_alpha=1.702`` and ``swiglu_beta=1.0`` for MiniMax-M3:
    ``(up + beta) * gate * sigmoid(alpha * gate)``, after ``gate_up_clamp``.
    These are scalar activation constants, distinct from ``fc1_alpha``.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_nvfp4_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    fast_math: bool = True
    apply_topk_in_fc1: bool = True
    # Enables in_kernel_fc2_reduce, knobs may still disable this if it is faster
    # In-flight top-k combine: cross-rank REDG atomic-add collapses the combine
    # as peer data arrives (no per-topk staging / explicit tail reduce).
    # ~1-2% faster and removes the multi-GB combine staging from the symmetric
    # workspace.  Requires apply_topk_in_fc1=True and combine_dtype="bf16";
    # accumulation order is nondeterministic (tolerance-compare outputs).
    enable_in_kernel_fc2_reduce: bool = False
    # Cross-rank combine wire format: "bf16" (exact), "mxfp8" (2x less combine
    # traffic), "nvfp4" (4x less).  Quantized wires trade accuracy for NVLink
    # bandwidth and require enable_in_kernel_fc2_reduce=False.
    combine_dtype: Literal["bf16", "mxfp8", "nvfp4"] = "bf16"
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
    # "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    knobs: dict | str | None = None
    # Uniform per-layer SwiGLU constants; None/None keeps standard SwiGLU.
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None

    def __post_init__(self) -> None:
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be set together.")
