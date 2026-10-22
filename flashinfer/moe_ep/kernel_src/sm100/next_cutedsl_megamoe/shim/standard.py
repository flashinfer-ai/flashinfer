# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Native Blackwell standard inference over the exported provider kernels."""

from __future__ import annotations

import dataclasses
import os

import torch

from .block_scaled import (
    BlockScaledMoeConfig,
    BlockScaledSymmBuffer,
    _expected_weight_shapes,
    _validate_weight_leg,
)


@dataclasses.dataclass(frozen=True)
class Sm100BlockScaledMoeConfig(BlockScaledMoeConfig):
    """Standard block-scaled inference with native Blackwell instruction K."""

    enable_fc1_activation_per_token_scale: bool = False

    @property
    def instruction_k(self) -> int:
        return 64 if self.quant_kind == "nvfp4" else 32

    @property
    def resolved_mma_tiler_mnk(self) -> tuple[int, int, int]:
        return self.mma_tiler_mnk or (256, 128, 128)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kernel_variant != "inference":
            raise ValueError("Use Sm100GenphaseMoeConfig for GenPhase.")
        if self.quant_kind not in ("nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
            raise ValueError(
                "This adapter supports NVFP4 and MXFP8 standard inference."
            )


class Sm100BlockScaledSymmBuffer(BlockScaledSymmBuffer):
    """Standard-inference storage, compilation and launch on SM100/SM103."""

    _architecture = "sm_100"

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

    def __init__(self, config: Sm100BlockScaledMoeConfig) -> None:
        if not isinstance(config, Sm100BlockScaledMoeConfig):
            raise TypeError("Standard inference requires Sm100BlockScaledMoeConfig.")
        super().__init__(config)

    def _runtime_kwargs(self, transformed_l1, transformed_l2) -> dict:
        from .block_scaled import _to_cute

        kwargs = super()._runtime_kwargs(transformed_l1, transformed_l2)
        if self.fc1_activation_per_token_scale is not None:
            kwargs["fc1_activation_per_token_scale"] = _to_cute(
                self.fc1_activation_per_token_scale, assumed_align=4
            )
        return kwargs

    def destroy(self) -> None:
        from .comm import free_sym_tensor

        if self._destroyed:
            return
        super().destroy()
        free_sym_tensor(self.fc1_activation_per_token_scale)
        self.fc1_activation_per_token_scale = None


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
