"""CuTeDSL mixed MXFP8-weight/BF16-activation MegaMoE config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from ..common.bf16_config import Sm100_Bf16_Cutedsl_MegaMoeConfigBase

if TYPE_CHECKING:
    import torch


@dataclass
class Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(Sm100_Bf16_Cutedsl_MegaMoeConfigBase):
    """Parameters for the mixed SwapAB SM100 CuTeDSL MegaMoE kernel.

    ``intermediate_size`` is the post-SwiGLU width, matching NVFP4 and SGLang.
    Activations and output stay BF16. Expert weights are MXFP8 with one E8M0
    scale per K32 block; canonical BF16 weights are quantized by preprocessing.
    """

    kernel_name: str = "sm100_bf16_mxfp8_bf16_cutedsl"
    kind: Literal["bf16_mxfp8_e4m3", "bf16_mxfp8_e5m2"] = "bf16_mxfp8_e4m3"
    # Forwards the actual token count to the top-k function to prevent wasted work
    # when max_num_tokens is larger than the actual token count.
    use_persistent_finalize_kernel: bool = False
    # Caller-owned CUDA int32 (1,) live-token count the reducer reads; the
    # workspace borrows it and never writes it. Update before every forward.
    num_valid_tokens_tensor: Optional["torch.Tensor"] = None

    def __post_init__(self) -> None:
        if self.use_persistent_finalize_kernel and self.num_valid_tokens_tensor is None:
            raise ValueError(
                "use_persistent_finalize_kernel=True requires "
                "num_valid_tokens_tensor: a caller-owned CUDA int32 tensor of "
                "shape (1,) holding the live token count."
            )
