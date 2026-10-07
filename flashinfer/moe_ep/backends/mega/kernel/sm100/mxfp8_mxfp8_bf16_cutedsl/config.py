"""CuTeDSL MXFP8 mega-MoE kernel config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

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
    # Forwards the actual token count to the top-k function to prevent wasted work
    # when max_num_tokens is larger than the actual token count.
    use_persistent_finalize_kernel: bool = False
    # Caller-owned CUDA int32 (1,) live-token count the reducer reads; the
    # workspace borrows it and never writes it. Update before every forward.
    num_valid_tokens_tensor: Optional["torch.Tensor"] = None
    # Kernel tuning knobs (see kernel_src.sm100.cutedsl_megamoe.shim.tuner); overrides
    # the token-count default heuristic entirely when set, e.g. a winner from the
    # kernel repo's tester sweep. None -> tuner.default_knobs(..., dtype="mxfp8").
    # "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    knobs: dict | str | None = None

    def __post_init__(self) -> None:
        if self.use_persistent_finalize_kernel and self.num_valid_tokens_tensor is None:
            raise ValueError(
                "use_persistent_finalize_kernel=True requires "
                "num_valid_tokens_tensor: a caller-owned CUDA int32 tensor of "
                "shape (1,) holding the live token count."
            )
