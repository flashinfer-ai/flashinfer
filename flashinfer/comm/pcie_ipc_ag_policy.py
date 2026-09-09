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

Launch policy for the PCIe IPC all-gather.

The policy is deliberately a pure, conservative seed. Launch geometry contains
the actual CUDA grid size, not an upper bound that the launcher silently
reduces. Device-specific choices belong in the tuner.
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Optional


_PACK_BYTES = 16
_MAX_THREADS = 512
_MAX_BLOCKS = 64
_FLAT_TO_RECURSIVE_BYTES = 256 * 1024
_COPY_ENGINE_BYTES = 512 * 1024


class PcieIpcAllGatherVariant(IntEnum):
    """All-gather kernel selected across the FFI boundary."""

    FLAT_PUSH = 0
    RECURSIVE_DOUBLING = 1
    COPY_ENGINE = 2


@dataclass(frozen=True)
class PcieIpcAllGatherLaunchConfig:
    blocks: int
    threads: int
    variant: PcieIpcAllGatherVariant


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
    config: PcieIpcAllGatherLaunchConfig,
    max_blocks: int,
    ordered_4plus4: bool = False,
) -> bool:
    if (
        type(config.blocks) is not int
        or type(config.threads) is not int
        or not isinstance(config.variant, PcieIpcAllGatherVariant)
    ):
        return False
    if not 0 < config.blocks <= max_blocks:
        return False
    if not 32 <= config.threads <= _MAX_THREADS or config.threads % 32 != 0:
        return False
    if config.variant == PcieIpcAllGatherVariant.COPY_ENGINE:
        return (
            world_size == 8
            and ordered_4plus4
            and config.blocks == 1
            and config.threads == 32
        )
    if config.variant == PcieIpcAllGatherVariant.FLAT_PUSH:
        return world_size == 4
    return config.variant == PcieIpcAllGatherVariant.RECURSIVE_DOUBLING


@lru_cache(maxsize=None)
def get_pcie_ipc_all_gather_launch_config(
    world_size: int,
    shard_numel: int,
    max_blocks: int = _MAX_BLOCKS,
    element_size: int = 2,
    ordered_4plus4: bool = False,
) -> Optional[PcieIpcAllGatherLaunchConfig]:
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

    if world_size == 8 and ordered_4plus4:
        if payload_bytes >= _COPY_ENGINE_BYTES:
            return PcieIpcAllGatherLaunchConfig(
                blocks=1,
                threads=32,
                variant=PcieIpcAllGatherVariant.COPY_ENGINE,
            )
        variant = PcieIpcAllGatherVariant.RECURSIVE_DOUBLING
    elif world_size == 4 and payload_bytes < _FLAT_TO_RECURSIVE_BYTES:
        variant = PcieIpcAllGatherVariant.FLAT_PUSH
    else:
        variant = PcieIpcAllGatherVariant.RECURSIVE_DOUBLING

    num_packs = payload_bytes // _PACK_BYTES
    threads = _threads_for(num_packs)
    blocks = min(max_blocks, (num_packs + threads - 1) // threads)
    config = PcieIpcAllGatherLaunchConfig(blocks, threads, variant)
    return (
        config
        if _is_launchable(world_size, config, max_blocks, ordered_4plus4)
        else None
    )
