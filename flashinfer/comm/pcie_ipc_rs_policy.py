"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Launch policy for the PCIe IPC reduce-scatter.

The launch seed is deterministic and makes no device-name assumptions.
``blocks`` is the actual grid size; device-specific choices belong in tuning.
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Optional


_PACK_BYTES = 16
_MAX_THREADS = 512
_MAX_BLOCKS = 64
_TOPOLOGY_CYCLIC_BYTES = 128 * 1024
_FLAT_CYCLIC_BYTES = 4 * 1024 * 1024


class PcieIpcReduceScatterVariant(IntEnum):
    """Reduce-scatter kernel selected across the FFI boundary."""

    FLAT_CYCLIC = 0
    FLAT_ONE_PACK = 1
    TOPOLOGY_CYCLIC = 2
    TOPOLOGY_ONE_PACK = 3


@dataclass(frozen=True)
class PcieIpcReduceScatterLaunchConfig:
    blocks: int
    threads: int
    variant: PcieIpcReduceScatterVariant


def _threads_for(num_packs: int) -> int:
    if num_packs <= 1024:
        return 64
    if num_packs <= 4096:
        return 128
    if num_packs <= 16384:
        return 256
    return _MAX_THREADS


def _is_launchable(
    world_size: int,
    config: PcieIpcReduceScatterLaunchConfig,
    max_blocks: int,
    ordered_4plus4: bool = False,
) -> bool:
    if (
        type(config.blocks) is not int
        or type(config.threads) is not int
        or not isinstance(config.variant, PcieIpcReduceScatterVariant)
    ):
        return False
    if not 0 < config.blocks <= max_blocks:
        return False
    if not 32 <= config.threads <= _MAX_THREADS or config.threads % 32 != 0:
        return False
    if config.variant in (
        PcieIpcReduceScatterVariant.TOPOLOGY_CYCLIC,
        PcieIpcReduceScatterVariant.TOPOLOGY_ONE_PACK,
    ):
        return world_size == 8 and ordered_4plus4
    return world_size in (2, 4) and config.variant in (
        PcieIpcReduceScatterVariant.FLAT_CYCLIC,
        PcieIpcReduceScatterVariant.FLAT_ONE_PACK,
    )


@lru_cache(maxsize=None)
def get_pcie_ipc_reduce_scatter_launch_config(
    world_size: int,
    shard_numel: int,
    max_blocks: int = _MAX_BLOCKS,
    element_size: int = 2,
    ordered_4plus4: bool = False,
) -> Optional[PcieIpcReduceScatterLaunchConfig]:
    """Return a deterministic launch seed, or ``None`` if unsupported."""
    if world_size not in (2, 4, 8):
        return None
    if element_size <= 0 or _PACK_BYTES % element_size != 0:
        return None
    payload_bytes = shard_numel * element_size
    if shard_numel <= 0 or payload_bytes % _PACK_BYTES != 0:
        return None
    if max_blocks <= 0 or max_blocks > _MAX_BLOCKS:
        return None
    if world_size == 8 and not ordered_4plus4:
        return None
    if world_size == 8:
        variant = (
            PcieIpcReduceScatterVariant.TOPOLOGY_CYCLIC
            if payload_bytes >= _TOPOLOGY_CYCLIC_BYTES
            else PcieIpcReduceScatterVariant.TOPOLOGY_ONE_PACK
        )
    else:
        variant = (
            PcieIpcReduceScatterVariant.FLAT_CYCLIC
            if world_size == 2 and payload_bytes >= _FLAT_CYCLIC_BYTES
            else PcieIpcReduceScatterVariant.FLAT_ONE_PACK
        )

    num_packs = payload_bytes // _PACK_BYTES
    threads = _threads_for(num_packs)
    useful_blocks = (num_packs + threads - 1) // threads
    grid_cap = max_blocks
    if variant == PcieIpcReduceScatterVariant.TOPOLOGY_CYCLIC:
        grid_cap = min(grid_cap, 4 if payload_bytes >= 1024 * 1024 else 8)
    blocks = min(grid_cap, useful_blocks)
    config = PcieIpcReduceScatterLaunchConfig(blocks, threads, variant)
    return (
        config
        if _is_launchable(world_size, config, max_blocks, ordered_4plus4)
        else None
    )
