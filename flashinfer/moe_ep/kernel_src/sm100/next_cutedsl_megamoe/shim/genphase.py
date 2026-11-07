# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit Blackwell GenPhase frontend over its independently owned source fork.

The prototype selects native Blackwell MMA and a 512-column TMEM plan.
It does not import the Rubin frontend or source package.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

import torch

from flashinfer.moe_ep._compat import deprecated_aliases

from .block_scaled import (
    BlockScaledMoeConfig,
    BlockScaledSymmBuffer,
    _expected_weight_shapes,
    _validate_weight_leg,
    block_scaled_mega_launch_thunk as sm100_genphase_mega_launch_thunk,
    block_scaled_mega_moe as sm100_genphase_mega_moe,
)

from .kernel_helpers import (
    concatenate_block_scaled_weights as concatenate_genphase_weights,
    preprocess_block_scaled_weights as preprocess_genphase_weights,
    preprocess_prequantized_block_scaled_weights as preprocess_genphase_prequantized_weights,
    quantize_mxfp8_block32 as quantize_genphase_mxfp8,
)


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
@dataclasses.dataclass(frozen=True)
class Sm100GenphaseMoeConfig(BlockScaledMoeConfig):
    """Block-scaled weights, BF16 combine, grouped GenPhase on SM100/103.

    NVFP4 uses K256/N128/CGA4. MXFP8 E4M3/E5M2 and mixed MXFP4/MXFP8
    use K128/N128/CGA4, with N64 alternatives for sparse expert batches.
    """

    kernel_variant: Literal["inference", "genphase"] = "genphase"
    cluster_shape_mn: tuple[int, int] = (4, 1)
    fc2_use_bulk: bool = True

    @property
    def instruction_k(self) -> int:
        """Return the native block-scaled MMA K extent for the quantization kind."""
        return 64 if self.quant_kind == "nvfp4" else 32

    @property
    def gather_tile_k(self) -> int:
        """Pack each activation row into one 128-byte Gather4 swizzle atom."""
        return 256 if self.quant_kind == "nvfp4" else 128

    @property
    def resolved_mma_tiler_mnk(self) -> tuple[int, int, int]:
        """Use explicit geometry or the dtype's N128 Gather4 default."""
        return self.mma_tiler_mnk or (256, 128, self.gather_tile_k)

    @property
    def admitted_geometries(self):
        """Enumerate the dtype's supported Gather4 tile and cluster pairs."""
        geometries = (((256, 128, self.gather_tile_k), (4, 1)),)
        if self.quant_kind != "nvfp4":
            geometries += (((256, 64, 128), (4, 1)),)
        if self.quant_kind in ("mxfp8_e4m3", "mxfp8_e5m2"):
            geometries += (((256, 64, 128), (2, 1)),)
        return geometries

    def __post_init__(self) -> None:
        """Restrict GenPhase to admitted Gather4 geometries and BF16 reduction."""
        super().__post_init__()
        geometry = (self.resolved_mma_tiler_mnk, self.cluster_shape_mn)
        if (
            self.kernel_variant != "genphase"
            or geometry not in self.admitted_geometries
            or self.schedule_policy[0] != "grouped"
            or self.reduce_topk_in_kernel
            or self.activation != "swiglu"
        ):
            raise ValueError(
                "Blackwell GenPhase requires SwiGLU, N128/CGA4 or an admitted "
                "N64 geometry, K256 for NVFP4 or K128 for MXFP8 and mixed "
                "MXFP4/MXFP8, grouped scheduling and "
                "separate BF16 reduction."
            )


class Sm100GenphaseSymmBuffer(BlockScaledSymmBuffer):
    """Blackwell workspace; staging and launch follow the shared block-scaled API."""

    # SM100 and SM103 have the same block-scaled atom and TMEM capacity.
    # The compiler target below remains native to the actual device.
    _architecture = "sm_100"

    @staticmethod
    def _require_device() -> None:
        cc = torch.cuda.get_device_capability()
        if cc not in ((10, 0), (10, 3)):
            raise RuntimeError(f"Blackwell GenPhase requires SM100/103; got {cc}.")

    @staticmethod
    def _configure_dsl() -> None:
        cc = torch.cuda.get_device_capability()
        target = f"sm_{cc[0]}{cc[1]}a"
        configured = os.environ.get("CUTE_DSL_ARCH")
        if configured not in (None, target, target.removesuffix("a")):
            raise RuntimeError(
                f"Blackwell GenPhase requires CUTE_DSL_ARCH={target}; got {configured!r}."
            )
        os.environ["CUTE_DSL_ARCH"] = target
        from flashinfer.cute_dsl.availability import _dsl_captured_arch

        if _dsl_captured_arch() not in (target, target.removesuffix("a")):
            raise RuntimeError(
                f"Restart Python with CUTE_DSL_ARCH={target} before importing CuTe DSL."
            )

    def __init__(self, config: Sm100GenphaseMoeConfig) -> None:
        if not isinstance(config, Sm100GenphaseMoeConfig):
            raise TypeError("Blackwell GenPhase requires Sm100GenphaseMoeConfig.")
        super().__init__(config)

    def retune(self, knobs) -> None:
        """Collectively replace kernel storage before capture, preserving I/O views.

        Only performance knobs may change. Automatic selection instead uses
        ``install_tuned_knobs()`` to prepare the winner with its weights.
        """
        from . import comm
        from .tuning import is_valid_sm100_genphase_knobs

        comm.ensure_not_capturing("Blackwell GenPhase retune")
        if self._destroyed:
            raise RuntimeError("cannot retune a destroyed workspace")
        if not is_valid_sm100_genphase_knobs(knobs, self.config):
            raise ValueError("invalid Blackwell GenPhase tuning knobs")
        config = dataclasses.replace(self.config, **knobs)
        if config == self.config:
            return
        kernel = self._build_kernel(config)
        torch.cuda.synchronize()
        self._compiled = self._launch_key = self._launch_kwargs = None
        self._launch_weights = None
        self._pre_reduced_activation = self._pre_reduced_activation_sf = None
        comm.free_sym_tensor(self.shared_workspace)
        self.shared_workspace = self.local_workspace = None
        self.config, self.kernel = config, kernel
        self._allocate_kernel_workspace(kernel)


def validate_genphase_weights(transformed, config, device: torch.device) -> None:
    """Validate external GenPhase weights without allocating a workspace.

    Legacy SM100 transformed weights have a different physical layout and
    cannot be reused as a GenPhase payload.
    """
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise ValueError("GenPhase transformed weights must be a (fc1, fc2) tuple.")
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(config)
    for name, leg, shape, sf in (
        ("fc1", transformed[0], fc1_shape, fc1_sf),
        ("fc2", transformed[1], fc2_shape, fc2_sf),
    ):
        _validate_weight_leg(name, leg, shape, sf, config, device)


__all__ = [
    "concatenate_genphase_weights",
    "Sm100GenphaseMoeConfig",
    "Sm100GenphaseSymmBuffer",
    "preprocess_genphase_weights",
    "preprocess_genphase_prequantized_weights",
    "validate_genphase_weights",
    "quantize_genphase_mxfp8",
    "sm100_genphase_mega_launch_thunk",
    "sm100_genphase_mega_moe",
]
