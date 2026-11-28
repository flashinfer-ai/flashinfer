# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warp roles of the decode kernel: active roles, padding and finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .device.constants import MAX_WARP_GROUPS

if TYPE_CHECKING:
    from .config import FmhaDecodeConfig


@dataclass(frozen=True)
class WarpRole:
    """One active task's configured warp placement."""

    name: str
    index_field: str
    preferred_warp_idx: int
    num_warps: int
    is_padding: bool = False


def active_warp_roles(cfg: FmhaDecodeConfig) -> list[WarpRole]:
    """Return the warp roles instantiated by the current kernel profile."""
    roles = [
        WarpRole(
            "softmax0",
            "softmax0_warp_idx",
            cfg.softmax0_warp_idx,
            cfg.softmax0_num_warps,
        ),
        WarpRole(
            "correction",
            "correction_warp_idx",
            cfg.correction_warp_idx,
            cfg.correction_num_warps,
        ),
        WarpRole("mma", "mma_warp_idx", cfg.mma_warp_idx, cfg.mma_num_warps),
    ]
    if cfg.num_insts_kv != 1:
        roles.append(
            WarpRole(
                "softmax1",
                "softmax1_warp_idx",
                cfg.softmax1_warp_idx,
                cfg.softmax1_num_warps,
            )
        )
    if cfg.use_transform_kv:
        roles.append(
            WarpRole(
                "transform_kv",
                "transform_kv_warp_idx",
                cfg.transform_kv_warp_idx,
                cfg.transform_kv_num_warps,
            )
        )
        if cfg.uses_dual_kv_transform:
            roles.append(
                WarpRole(
                    "transform_v",
                    "transform_v_warp_idx",
                    cfg.transform_v_warp_idx,
                    cfg.transform_v_num_warps,
                )
            )
        if cfg.uses_second_smem_converter:
            roles.append(
                WarpRole(
                    "transform_kv1",
                    "transform_kv1_warp_idx",
                    cfg.transform_kv1_warp_idx,
                    cfg.transform_kv1_num_warps,
                )
            )

    if cfg.use_persistent_scheduler:
        roles.extend(
            (
                WarpRole(
                    "scheduler",
                    "scheduler_warp_idx",
                    cfg.scheduler_warp_idx,
                    cfg.scheduler_num_warps,
                ),
                WarpRole(
                    "load",
                    "clc_load_warp_idx",
                    cfg.clc_load_warp_idx,
                    cfg.load_num_warps,
                ),
            )
        )
    else:
        roles.append(
            WarpRole(
                "load",
                "load_warp_idx",
                cfg.load_warp_idx,
                cfg.load_num_warps,
            )
        )

    if cfg.uses_page_offsets_task:
        roles.append(
            WarpRole(
                "page_offsets",
                "page_offsets_warp_idx",
                cfg.page_offsets_warp_idx,
                cfg.page_offsets_num_warps,
            )
        )
    append_padding_warp_roles(cfg, roles)
    return roles


def append_padding_warp_roles(cfg: FmhaDecodeConfig, roles: list[WarpRole]) -> None:
    """Append padding warps to the tail of each warp group."""
    num_warps_per_wg = [0] * MAX_WARP_GROUPS
    total_num_wgs = 0
    for role in roles:
        if role.num_warps <= 0:
            raise ValueError(f"{role.name} must use at least one warp")
        preferred_wg = role.preferred_warp_idx // 4
        preferred_wg_end = (role.preferred_warp_idx + role.num_warps - 1) // 4
        if preferred_wg != preferred_wg_end:
            # A producer-only load or K/V transform task may occupy complete
            # consecutive warp groups. Keeping each group full preserves
            # warpgroup register reallocation participation and needs no
            # cross-group padding.
            spans_complete_wgs = (
                role.name in ("load", "transform_kv")
                and role.preferred_warp_idx % 4 == 0
                and role.num_warps % 4 == 0
            )
            if not spans_complete_wgs:
                raise ValueError(
                    f"{role.name} spans warp groups {preferred_wg} and "
                    f"{preferred_wg_end}; only complete-group producer tasks "
                    "may span warp groups"
                )
        if preferred_wg < 0 or preferred_wg_end >= MAX_WARP_GROUPS:
            raise ValueError(
                f"{role.name} is assigned through unsupported warp group "
                f"{preferred_wg_end}"
            )
        role_warps_remaining = role.num_warps
        role_wg = preferred_wg
        while role_warps_remaining > 0:
            warps_in_group = min(role_warps_remaining, 4)
            num_warps_per_wg[role_wg] += warps_in_group
            role_warps_remaining -= warps_in_group
            role_wg += 1
        total_num_wgs = max(total_num_wgs, preferred_wg_end + 1)
    for wg_idx, num_warps in enumerate(num_warps_per_wg):
        if num_warps > 4:
            raise ValueError(
                f"warp group {wg_idx} has {num_warps} active warps; at most 4 are allowed"
            )

    # When using persistent scheduling, force 4 warp groups
    if cfg.use_persistent_scheduler:
        # Ordinary persistent decode retains its established four-group
        # contract. A profile that explicitly places a role in WG4 keeps that
        # fifth group instead of truncating the task from the CTA.
        total_num_wgs = max(total_num_wgs, 4)

    for wg_idx, active_warps in enumerate(num_warps_per_wg):
        padding_index_field = f"wg{wg_idx}_padding_warp_idx"
        padding_count_field = f"wg{wg_idx}_padding_num_warps"
        padding_warp_idx = wg_idx * 4 + active_warps
        padding_num_warps = 4 - active_warps if wg_idx < total_num_wgs else 0
        setattr(cfg, padding_index_field, padding_warp_idx)
        setattr(cfg, padding_count_field, padding_num_warps)
        if padding_num_warps > 0:
            roles.append(
                WarpRole(
                    f"wg{wg_idx}_padding",
                    padding_index_field,
                    padding_warp_idx,
                    padding_num_warps,
                    is_padding=True,
                )
            )


def finalize_warp_roles(cfg: FmhaDecodeConfig) -> None:
    """Compact active warp indices and add paddings."""
    warp_roles = sorted(
        active_warp_roles(cfg),
        key=lambda role: (
            role.preferred_warp_idx // 4,
            role.is_padding,
            role.preferred_warp_idx,
        ),
    )

    next_warp_idx = 0
    for role in warp_roles:
        setattr(cfg, role.index_field, next_warp_idx)
        next_warp_idx += role.num_warps
