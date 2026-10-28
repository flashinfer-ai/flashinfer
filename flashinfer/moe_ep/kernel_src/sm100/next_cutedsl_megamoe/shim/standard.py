# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Native Blackwell standard inference over the exported provider kernels."""

from __future__ import annotations

import dataclasses
import math
import os

import torch

from flashinfer.moe_ep._compat import deprecated_aliases

from .block_scaled import (
    BlockScaledMoeConfig,
    BlockScaledSymmBuffer,
    _expected_weight_shapes,
    _validate_weight_leg,
)


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
@dataclasses.dataclass(frozen=True)
class Sm100BlockScaledMoeConfig(BlockScaledMoeConfig):
    """Standard block-scaled inference with native Blackwell instruction K."""

    enable_fc1_activation_per_token_scale: bool = False
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    use_custom_finalize: bool = False
    use_persistent_finalize_kernel: bool = False

    @property
    def instruction_k(self) -> int:
        return 64 if self.quant_kind == "nvfp4" else 32

    @property
    def hidden_alignment(self) -> int:
        return 64

    @property
    def resolved_mma_tiler_mnk(self) -> tuple[int, int, int]:
        return self.mma_tiler_mnk or (256, 128, 128)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be set together.")
        if self.swiglu_alpha is not None and self.swiglu_beta is not None:
            if self.activation != "swiglu":
                raise ValueError("SwiGLU constants require activation='swiglu'.")
            if not all(math.isfinite(x) for x in (self.swiglu_alpha, self.swiglu_beta)):
                raise ValueError("SwiGLU constants must be finite.")
        for name in ("use_custom_finalize", "use_persistent_finalize_kernel"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool.")
        if self.use_custom_finalize and (
            self.combine_dtype != "bf16"
            or self.reduce_topk_in_kernel
            or self.use_persistent_finalize_kernel
        ):
            raise ValueError(
                "Custom finalization requires BF16 combine without in-kernel or persistent reduction."
            )
        if self.use_persistent_finalize_kernel and self.reduce_topk_in_kernel:
            raise ValueError("Persistent finalization requires separate reduction.")
        if self.num_topk > 256 or self.world_size > 256:
            raise ValueError(
                "Routing metadata supports at most 256 ranks and routes per token."
            )
        if self.kernel_variant != "inference":
            raise ValueError("Use Sm100GenphaseMoeConfig for GenPhase.")
        if self.quant_kind not in ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
            raise ValueError(
                "This adapter supports NVFP4 and MXFP8 standard inference."
            )


class Sm100BlockScaledSymmBuffer(BlockScaledSymmBuffer):
    """Standard-inference storage, compilation and launch on SM100/SM103."""

    _architecture = "sm_100"
    config: Sm100BlockScaledMoeConfig

    @staticmethod
    def _require_device() -> None:
        cc = torch.cuda.get_device_capability()
        if cc not in ((10, 0), (10, 3)):
            raise RuntimeError(f"Blackwell MegaMoE requires SM100/SM103; got {cc}.")

    @staticmethod
    def _configure_dsl() -> None:
        cc = torch.cuda.get_device_capability()
        target = f"sm_{cc[0]}{cc[1]}a"
        configured = os.environ.get("CUTE_DSL_ARCH")
        if configured not in (None, target, target.removesuffix("a")):
            raise RuntimeError(f"Blackwell MegaMoE requires CUTE_DSL_ARCH={target}.")
        os.environ["CUTE_DSL_ARCH"] = target
        from flashinfer.cute_dsl.availability import _dsl_captured_arch

        if _dsl_captured_arch() not in (target, target.removesuffix("a")):
            raise RuntimeError(f"Restart Python with CUTE_DSL_ARCH={target}.")

    def __init__(
        self,
        config: Sm100BlockScaledMoeConfig,
        *,
        num_valid_tokens_tensor: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(config, Sm100BlockScaledMoeConfig):
            raise TypeError("Standard inference requires Sm100BlockScaledMoeConfig.")
        if config.use_persistent_finalize_kernel:
            if (
                num_valid_tokens_tensor is None
                or not num_valid_tokens_tensor.is_cuda
                or num_valid_tokens_tensor.dtype != torch.int32
                or tuple(num_valid_tokens_tensor.shape) != (1,)
                or num_valid_tokens_tensor.device.index != torch.cuda.current_device()
            ):
                raise ValueError(
                    "Persistent finalization requires a CUDA int32 (1,) live count on the workspace device."
                )
        elif num_valid_tokens_tensor is not None:
            raise ValueError("A live count requires persistent finalization.")
        self.num_valid_tokens = num_valid_tokens_tensor
        super().__init__(config)

    def _runtime_kwargs(self, transformed_l1, transformed_l2) -> dict:
        from .block_scaled import _to_cute

        kwargs = super()._runtime_kwargs(transformed_l1, transformed_l2)
        if self.num_valid_tokens is not None:
            kwargs["num_valid_tokens"] = _to_cute(
                self.num_valid_tokens, assumed_align=4, dynamic_leading=False
            )
        if self.fc1_activation_per_token_scale is not None:
            kwargs["fc1_activation_per_token_scale"] = _to_cute(
                self.fc1_activation_per_token_scale, assumed_align=4
            )
        return kwargs

    def custom_finalize_workspace(self) -> torch.Tensor:
        """Borrow BF16 [capacity, top-k, hidden] partials until workspace reuse."""
        if self._destroyed or not self.config.use_custom_finalize:
            raise RuntimeError(
                "Custom finalization is not enabled on a live workspace."
            )
        return self._pre_reduced_activation.view(torch.bfloat16)

    def destroy(self) -> None:
        from .comm import free_sym_tensor

        if self._destroyed:
            return
        super().destroy()
        free_sym_tensor(self.fc1_activation_per_token_scale)
        self.fc1_activation_per_token_scale = None
        self.num_valid_tokens = None


def validate_standard_weights(transformed, config, device: torch.device) -> None:
    """Check external weights against the standard adapter's physical layout."""
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise ValueError("Transformed weights must be a (fc1, fc2) tuple.")
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(config)
    for name, leg, shape, sf in (
        ("fc1", transformed[0], fc1_shape, fc1_sf),
        ("fc2", transformed[1], fc2_shape, fc2_sf),
    ):
        _validate_weight_leg(name, leg, shape, sf, config, device)
