# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-visible configuration descriptors for SM120 NVFP4 MegaMoE."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass
class ImplDesc:
    """Kernel-instantiation-side configuration."""

    mma_tiler_mnk: Tuple[int, int, int] = (64, 128, 128)
    enable_static_expert_shape: bool = False
    group_hint: Optional[int] = None
    token_back_mode: Literal[
        "epi_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    flag_batch: int = 4
    epi_flag_batch: Optional[Tuple[int, int]] = (1, 1)

    def __post_init__(self) -> None:
        m, n, _k = self.mma_tiler_mnk

        if m != 64:
            raise ValueError(f"SM120 swap-AB requires mma_tiler_m=64, got {m}.")
        if n not in (16, 32, 64, 128):
            raise ValueError(
                f"mma_tiler_n must be 16, 32, 64, or 128, got {n}."
            )
        if self.token_back_mode not in (
            "epi_warps", "reuse_dispatch_warps"
        ):
            raise ValueError(
                "token_back_mode must be 'epi_warps' or "
                f"'reuse_dispatch_warps'; got {self.token_back_mode!r}."
            )
        if self.group_hint is not None and self.group_hint <= 0:
            raise ValueError(
                f"group_hint must be positive when set, got {self.group_hint}."
            )
        if self.flag_batch < 1:
            raise ValueError(f"flag_batch must be >= 1, got {self.flag_batch}.")
        eb = self.epi_flag_batch if self.epi_flag_batch is not None else (1, 1)
        if len(eb) != 2:
            raise ValueError(
                f"epi_flag_batch must be a (fc1, fc2) pair, got "
                f"{self.epi_flag_batch}."
            )
        for leg, value in (("fc1", eb[0]), ("fc2", eb[1])):
            if value < 1 or value > 32:
                raise ValueError(
                    f"epi_flag_batch[{leg}] must be in [1, 32], got {value}."
                )

    @property
    def token_back_by_dispatch(self) -> bool:
        return self.token_back_mode != "epi_warps"

    def __str__(self) -> str:
        tile = ",".join(map(str, self.mma_tiler_mnk))
        static_shape = "static" if self.enable_static_expert_shape else "dynamic"
        group_hint = (
            str(self.group_hint)
            if self.group_hint is not None
            else "max_active_clusters"
        )
        return (
            f"ImplDesc: tile={tile} cluster=1,1,1 expert_shape={static_shape} "
            f"sched=static group_hint={group_hint} "
            f"token_back_mode={self.token_back_mode}"
        )


@dataclass
class MiscDesc:
    """Runtime, validation and profiling switches used by the host runner."""

    perf_run: bool = False
    skip_ref_check: bool = False
    run_target_kernel_only: bool = False
    enable_debug_checks: bool = False
    ref_compute_graph: Literal["transformers", "deepgemm"] = "deepgemm"
    seed: int = 1234
    enable_iket: bool = False
    verbose: bool = False

    @property
    def profile_friendly(self) -> bool:
        return self.run_target_kernel_only

    def __post_init__(self) -> None:
        if self.ref_compute_graph not in ("transformers", "deepgemm"):
            raise ValueError(
                "ref_compute_graph must be 'transformers' or 'deepgemm', "
                f"got {self.ref_compute_graph!r}."
            )

    def __str__(self) -> str:
        return (
            f"MiscDesc: perf={self.perf_run} skip_ref={self.skip_ref_check} "
            f"target_only={self.run_target_kernel_only} "
            f"debug_checks={'on' if self.enable_debug_checks else 'off'} "
            f"ref_graph={self.ref_compute_graph} "
            f"iket={'on' if self.enable_iket else 'off'} seed={self.seed}"
        )
