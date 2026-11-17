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
"""

# Host plan of the Cake NVFP4 MegaMoE route on sm_107a: its geometry, token
# tiers, buffers, launch grids and the names of the program arguments the host
# supplies. Pure host constants and rules: no device, no NVSHMEM.
#
# A forward is four programs of the generated loader
# ``flashinfer.jit.cake_mega_moe_nvfp4``: STAGE quantizes the BF16 activations
# into this rank's symmetric source rows, ROUTER_T<n> pushes every route's
# record and score into its owner's pool, MEGA_ROUTE (the 1024 tier) or
# MEGA_RANK (the dedup tiers) pulls the routed rows and runs FC1, SwiGLU,
# NVFP4 requantization and FC2 into the source ranks' slots, and
# REDUCE_ROUTE / REDUCE_RANK sums each token's live slots to BF16. The router
# bakes the token count per rank: the route runs the smallest tier of
# :data:`TIERS` that holds ``max_tokens_per_rank``, and the routes of the
# tokens past the live count are -1 (dropped).
#
# The router hands its records and counts to the mega program as Lamport
# words: ``token_src_metadata`` (and on the dedup tiers ``fc1_topk_scores``)
# and ``sizes_by_rank`` hold two copies that alternate by the mega program's
# launch count, which it keeps in ``metadata_ready``; every word of them starts
# as the sentinel ``0x80000000`` (:func:`symmetric_fills`), and the programs
# arm each copy again after reading it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

#: The geometry the programs are generated for (DeepSeek-V4-Pro's EP4 slice).
HIDDEN = 7168
INTERMEDIATE = 3072
NUM_EXPERTS = 48
TOP_K = 6
WORLD_SIZE = 4
EXPERTS_LOCAL = NUM_EXPERTS // WORLD_SIZE
GATE_UP_CLAMP = 10.0
#: ``round_up(NUM_EXPERTS, 4)``: the router's padded expert count.
E_PAD = -(-NUM_EXPERTS // 4) * 4
#: Values per E4M3 block scale; the activation scale row is padded to 16 bytes.
SF_VEC = 16
SF_ROW_BYTES = -(-(HIDDEN // SF_VEC) // 16) * 16
#: Peer tables are by-value arrays of this many addresses; a process group's
#: flag pad holds two epoch words and ``2 phases x 2 banks x capacity``
#: mailbox words.
PEER_TABLE_CAPACITY = 8
PG_FLAG_WORDS = 2 + 2 * 2 * PEER_TABLE_CAPACITY
#: The mega program's arrival and ticket words.
SYNC_WORDS = 16
#: Rank slots per token of the dedup combine's output.
RANK_SLOTS = 8
#: Every program launches one CTA per SM at most; the mega program's CTAs
#: come in clusters of two, which the rank-major program prefers to pair into
#: clusters of four.
MEGA_CLUSTER_CTAS = 2
#: nvcc has to know ``%locality_domain_id`` (PTX ISA 9.5) for the dedup tiers.
MIN_CUDA_DEDUP = "13.5"

GRID_ARGS = ("grid_x", "grid_y", "grid_z")
STAGE = "STAGE"
REDUCE_ROUTE = "REDUCE_ROUTE"
REDUCE_RANK = "REDUCE_RANK"
#: The mega program by route (the 1024 tier) and by rank (the dedup tiers).
MEGA_ROUTE = "MEGA_ROUTE"
MEGA_RANK = "MEGA_RANK"


@dataclass(frozen=True)
class Tier:
    """The constants of one token tier at :data:`WORLD_SIZE` ranks, as the
    Cake kernels compute them (the router's ``pool_capacity``,
    ``launch_configuration``, ``grid_ctas`` and ``push_ctas``; the mega
    kernel's ``workspace``)."""

    tokens: int
    dedup: bool
    pool_rows: int  # the router's pool_capacity: pool and FC1-output rows
    scale_rows: int  # rows of the pool's and FC1 output's scale atoms
    block_words: int  # words of fc1_ready / fc1_done (one per token block)
    data_ctas: int  # the router's sorting CTAs
    router_ctas: int  # the router's grid
    push_ctas: int  # the router's PUSH CTAs, a mega kernel argument

    @property
    def router(self) -> str:
        return f"ROUTER_T{self.tokens}"

    @property
    def mega(self) -> str:
        return MEGA_RANK if self.dedup else MEGA_ROUTE

    @property
    def reduce(self) -> str:
        return REDUCE_RANK if self.dedup else REDUCE_ROUTE

    @property
    def multi(self) -> bool:
        return self.data_ctas > 1

    @property
    def slots(self) -> int:
        """Slot rows of ``pre_reduced`` per token."""
        return RANK_SLOTS if self.dedup else TOP_K

    @property
    def mega_grid_multiple(self) -> int:
        """CTAs the mega grid is a multiple of: its preferred cluster shape,
        four for the rank-major program, two for the route-major one."""
        return 2 * MEGA_CLUSTER_CTAS if self.dedup else MEGA_CLUSTER_CTAS


TIERS: Tuple[Tier, ...] = (
    Tier(1024, False, 25280, 25984, 109, 1, 3, 3),
    Tier(4096, True, 99008, 99712, 397, 2, 48, 48),
    Tier(8192, True, 197312, 198016, 781, 3, 24, 24),
)
PROGRAM_IDENTIFIERS: Tuple[str, ...] = (
    STAGE,
    *(tier.router for tier in TIERS),
    MEGA_ROUTE,
    MEGA_RANK,
    REDUCE_ROUTE,
    REDUCE_RANK,
)


def select_tier(max_tokens_per_rank: int) -> Tier:
    """The smallest tier holding ``max_tokens_per_rank``."""
    for tier in TIERS:
        if max_tokens_per_rank <= tier.tokens:
            return tier
    raise ValueError(
        f"max_tokens_per_rank={max_tokens_per_rank} exceeds the largest Cake NVFP4 "
        f"tier ({TIERS[-1].tokens})"
    )


def validate_geometry(
    *, hidden: int, intermediate: int, num_experts: int, top_k: int, world_size: int
) -> None:
    """Refuse a geometry the generated programs were not built for."""
    expected = {
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "world_size": WORLD_SIZE,
    }
    got = {
        "hidden": hidden,
        "intermediate": intermediate,
        "num_experts": num_experts,
        "top_k": top_k,
        "world_size": world_size,
    }
    wrong = {name: value for name, value in got.items() if value != expected[name]}
    if wrong:
        raise ValueError(
            f"the Cake NVFP4 MegaMoE programs are generated for {expected}; got {wrong}"
        )


#: Symmetric buffers: name -> (shape, dtype); ``cap`` is the tier's tokens.
#: ``metadata_ready`` holds the mega program's launch count and the rows each
#: copy of the records filled; ``x_ready[s]`` the launch count source ``s``'s
#: router released its activations at.
def symmetric_buffers(tier: Tier) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
    cap = tier.tokens
    return {
        "activation": ((cap, HIDDEN // 2), torch.uint8),
        "activation_sf": ((cap, SF_ROW_BYTES), torch.uint8),
        "sizes_by_rank": ((2 * PEER_TABLE_CAPACITY, E_PAD), torch.int32),
        "sizes": ((E_PAD,), torch.int32),
        "sizes_ready": ((1,), torch.int32),
        "metadata_ready": ((3,), torch.int32),
        "x_ready": ((PEER_TABLE_CAPACITY,), torch.int32),
        "token_src_metadata": ((tier.pool_rows, 2), torch.int64),
        "fc1_topk_scores": (
            (tier.pool_rows, 2) if tier.dedup else (tier.pool_rows,),
            torch.float32,
        ),
        "pre_reduced": ((cap, tier.slots, HIDDEN), torch.bfloat16),
    }


#: The Lamport sentinel every 32-bit word of a Lamport buffer starts as.
LAMPORT_SENTINEL = -0x80000000


def symmetric_fills(tier: Tier) -> Tuple[str, ...]:
    """The symmetric buffers whose every 32-bit word starts as
    :data:`LAMPORT_SENTINEL` instead of zero."""
    names = ("token_src_metadata", "sizes_by_rank")
    return names + (("fc1_topk_scores",) if tier.dedup else ())


#: Local buffers: name -> (shape, dtype), zeroed once; the programs leave
#: every counter as they found it.
def local_buffers(tier: Tier) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
    cap = tier.tokens
    buffers = {
        "topk_indices": ((cap, TOP_K), torch.int32),
        "topk_scores": ((cap, TOP_K), torch.float32),
        "sorted_metadata": ((cap * TOP_K,), torch.int64),
        "sorted_scores": ((cap * TOP_K,), torch.float32),
        "pool_expert_base": ((EXPERTS_LOCAL,), torch.int32),
        "expert_counts": ((EXPERTS_LOCAL,), torch.int32),
        "source_expert_base": ((E_PAD,), torch.int32),
        "push_destination_base": ((E_PAD,), torch.int32),
        "sorted_metadata_ready": ((1,), torch.int32),
        "push_table_ready": ((1,), torch.int32),
        "router_size_counter": ((E_PAD + 1,), torch.int32),
        "router_histogram_done": ((1,), torch.int32),
        "source_base_ready": ((1,), torch.int32),
        "pool": ((tier.pool_rows, HIDDEN // 2), torch.uint8),
        "pool_sf": ((tier.scale_rows // 128, HIDDEN // (4 * SF_VEC), 512), torch.uint8),
        "fc1_out": ((tier.pool_rows, INTERMEDIATE // 2), torch.uint8),
        "fc1_out_sf": (
            (tier.scale_rows // 128, INTERMEDIATE // (4 * SF_VEC), 512),
            torch.uint8,
        ),
        "fc1_ready": ((tier.block_words,), torch.uint32),
        "fc1_done": ((tier.block_words,), torch.uint32),
        "sync_words": ((SYNC_WORDS,), torch.uint32),
    }
    if tier.dedup:
        buffers["dedup_rows"] = (
            (WORLD_SIZE * cap * min(TOP_K, EXPERTS_LOCAL) * HIDDEN * 2,),
            torch.uint8,
        )
        buffers["fc2_done"] = ((max(EXPERTS_LOCAL, tier.block_words),), torch.uint32)
    return buffers


def mega_grid(sm_count: int, tier: Tier) -> int:
    """One CTA per SM, rounded down to whole clusters of the tier's preferred
    shape (:attr:`Tier.mega_grid_multiple`): the CTAs wait on one another, so
    the grid is what the device holds at once beside nothing else."""
    return tier.mega_grid_multiple * (sm_count // tier.mega_grid_multiple)


def scale_multiplier(input_norm_const: float) -> float:
    """``input_norm_const / 6`` rounded to FP32, as the cutedsl staging passes
    its ``scale_multiplier``."""
    return float(torch.tensor(input_norm_const / 6.0, dtype=torch.float32))


__all__ = [
    "E_PAD",
    "EXPERTS_LOCAL",
    "GATE_UP_CLAMP",
    "GRID_ARGS",
    "HIDDEN",
    "INTERMEDIATE",
    "MEGA_RANK",
    "MEGA_ROUTE",
    "LAMPORT_SENTINEL",
    "MEGA_CLUSTER_CTAS",
    "MIN_CUDA_DEDUP",
    "NUM_EXPERTS",
    "PEER_TABLE_CAPACITY",
    "PG_FLAG_WORDS",
    "PROGRAM_IDENTIFIERS",
    "RANK_SLOTS",
    "REDUCE_RANK",
    "REDUCE_ROUTE",
    "SF_ROW_BYTES",
    "STAGE",
    "SYNC_WORDS",
    "TIERS",
    "TOP_K",
    "Tier",
    "WORLD_SIZE",
    "local_buffers",
    "mega_grid",
    "scale_multiplier",
    "select_tier",
    "symmetric_buffers",
    "symmetric_fills",
    "validate_geometry",
]
