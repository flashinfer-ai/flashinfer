"""CuTeDSL mixed MXFP8-weight/BF16-activation MegaMoE config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from flashinfer.moe_ep.config import FinalizeMode, validate_finalize_mode

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
    # NONE makes MoEEpMegaLayer.forward return per-route outputs for caller
    # finalization; it requires fixed/offline-tuned knobs.
    finalize_mode: FinalizeMode = FinalizeMode.AUTO
    # Caller-owned CUDA int32 (1,) live-token count the PERSISTENT finalize kernel
    # reads; the workspace borrows it and never writes it. Update before every
    # forward.
    num_valid_tokens_tensor: Optional["torch.Tensor"] = None

    def __post_init__(self) -> None:
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
            if self.knobs == "auto":
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
