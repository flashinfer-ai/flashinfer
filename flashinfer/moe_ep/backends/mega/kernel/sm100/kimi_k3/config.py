"""Configuration for the isolated Kimi K3 MegaMoE backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from ......config import FinalizeMode

if TYPE_CHECKING:
    import torch


_KNOB_KEYS = frozenset(
    (
        "mma_tiler_mnk",
        "cluster_shape_mn",
        "fallback_cluster_shape_mn",
        "schedule_policy",
        "work_id_mode",
        "token_back_mode",
        "fc2_use_bulk",
        "fc2_tma_stages",
        "token_in_flag_batch",
        "epi_flag_batches",
    )
)


@dataclass(frozen=True)
class Sm100_KimiK3_MegaMoeConfig:
    """K3/K3.1 Pro routed SiTU inference on SM100/SM103.

    Uses 896 experts, routed width 3584, intermediate size 3072 and top-16.
    Read the routed width from ``routed_expert_hidden_size``, not ``hidden_size``.
    Routing, latent projections, post-MoE norm and shared experts stay outside.
    Both SiTU beta values must be supplied, positive and finite.
    ``fast_situ`` selects the source's faster SiTU arithmetic; it can change
    output bits. ``knobs`` accepts fixed native settings, not online tuning.
    NVFP4 alpha/norm tensors hold one FP32 value per local expert.
    """

    intermediate_size: int = 3072
    top_k: int = 16
    quant_kind: Literal["nvfp4", "mxfp4_mxfp8"] = "nvfp4"
    kernel_name: str = field(default="sm100_kimi_k3_megamoe", init=False)
    combine_dtype: Literal["bf16", "mxfp8"] = field(default="bf16", kw_only=True)
    activation: Literal["situ"] = "situ"
    situ_beta: float | None = None
    situ_linear_beta: float | None = None
    fast_situ: bool = field(default=False, kw_only=True)
    apply_routing_weights_before_fc2: bool = False
    input_norm_const: float = 1.0
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None
    knobs: dict | None = None
    finalize_mode: FinalizeMode = FinalizeMode.DEFAULT
    kernel_variant: Literal["inference"] = field(default="inference", kw_only=True)
    enable_in_kernel_fc2_reduce: bool = field(default=False, kw_only=True)
    enable_fc1_activation_per_token_scale: bool = field(default=False, kw_only=True)
    num_valid_tokens_tensor: torch.Tensor | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        for name, expected in (("intermediate_size", 3072), ("top_k", 16)):
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise ValueError(f"Kimi K3 requires {name}={expected}")
        if self.quant_kind not in ("nvfp4", "mxfp4_mxfp8"):
            raise ValueError("quant_kind must be 'nvfp4' or 'mxfp4_mxfp8'")
        if self.combine_dtype not in ("bf16", "mxfp8"):
            raise ValueError("combine_dtype must be 'bf16' or 'mxfp8'")
        if self.activation != "situ":
            raise ValueError("Kimi K3 requires activation='situ'")
        for name in ("situ_beta", "situ_linear_beta", "input_norm_const"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "fast_situ",
            "apply_routing_weights_before_fc2",
            "enable_in_kernel_fc2_reduce",
            "enable_fc1_activation_per_token_scale",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if self.apply_routing_weights_before_fc2:
            raise ValueError("Kimi K3 requires late routing weights")
        if self.finalize_mode is not FinalizeMode.DEFAULT:
            raise ValueError("Kimi K3 requires finalize_mode=FinalizeMode.DEFAULT")
        if self.kernel_variant != "inference":
            raise ValueError("Kimi K3 requires kernel_variant='inference'")
        if self.enable_in_kernel_fc2_reduce:
            raise ValueError("Kimi K3 does not support in-kernel FC2 reduction")
        if self.enable_fc1_activation_per_token_scale:
            raise ValueError("Kimi K3 does not support per-token FC1 scales")
        if self.num_valid_tokens_tensor is not None:
            raise ValueError("Kimi K3 does not support num_valid_tokens_tensor")
        if self.knobs is not None and not isinstance(self.knobs, dict):
            raise ValueError("knobs must be a fixed dictionary or None")
        if self.knobs is not None and self.knobs.keys() - _KNOB_KEYS:
            raise ValueError("knobs contains unsupported Kimi settings")
        if self.quant_kind == "mxfp4_mxfp8" and (
            self.input_norm_const != 1.0
            or any(
                value is not None
                for value in (self.fc1_alpha, self.fc2_alpha, self.fc1_norm_const)
            )
        ):
            raise ValueError("MXFP4/MXFP8 uses unit alpha and normalization values")
