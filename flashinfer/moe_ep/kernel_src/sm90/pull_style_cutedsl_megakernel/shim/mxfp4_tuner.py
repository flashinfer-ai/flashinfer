# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Hopper MXFP4 offline winners and compact online-autotune candidates.

This module is deliberately data-only: it does not import torch, initialize a
process group, compile kernels, or touch the persistent knob cache.  The tables
are the exact result of the formal, fresh-process offline sweeps documented by
``MXFP4_TUNING_PROVENANCE``. Backend integration therefore keeps the frozen
profile-specific heuristic/manifest data distinct from the bounded live
candidate union assembled below.

The per-token heuristic tables remain the H200 formal-sweep result. Fused
online autotuning additionally carries a very small set of cross-device
anchors that won a formal H20 workload. Those anchors are candidates only:
they never replace an H200-derived heuristic without being timed on the live
device. Split candidates remain tied to their certified 132-SM partition.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)


Mxfp4ExecutionMode = Literal["fused", "split"]
Mxfp4RoutingProfile = Literal["block_permutation_v1", "published_exact_balanced_v1"]

MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: Mxfp4RoutingProfile = (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
)
MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: Mxfp4RoutingProfile = (
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
)
MXFP4_TUNING_ROUTING_PROFILES = (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
)

MXFP4_TUNING_TOKEN_BUCKETS = (8, 32, 64, 128, 256, 512, 1024, 2048)

# These two tactics are supplemental runtime candidates, not an H20 heuristic
# table. They were the two close formal finalists for the one measured H20
# workload. Keeping their provenance separate preserves both frozen H200
# routing-profile manifests byte-for-byte while allowing live autotune to
# decide whether either tactic helps its actual device and shape.
MXFP4_FUSED_RUNTIME_ANCHOR_PROVENANCE = MappingProxyType(
    {
        "device": "NVIDIA H20-3e",
        "compute_capability": (9, 0),
        "sm_count": 78,
        "world_size": 8,
        "tokens_per_rank": 1,
        "hidden": 3072,
        "intermediate": 1280,
        "num_experts": 384,
        "topk": 8,
        "routing_profile": MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
        "routing_seed": 1234,
        "route_ids_sha256": (
            "d95ea5e18e4bb5010dd9cdadf928c6a43e085537615c5dbd166027a64de844eb"
        ),
        "winner_manifest_sha256": (
            "ffb3f8df0edef5e6a07d8685b35e9759e4da691e6010d4251c0df9b37ca40ce7"
        ),
        "formal_manifest_sha256": (
            "fe5cca19a9bc8e30a74f28a47561fd08755ba3f5ab22f90e50b63ebaceabafd6"
        ),
        "artifact_files_sha256": (
            "ae23042d2d4505794867cbdd3523e5469be785506211d38a6266821fe45514f7"
        ),
    }
)


def _current_hopper_device_identity() -> tuple[str, tuple[int, int], int]:
    """Return ``(product name, compute capability, SM count)`` for CUDA."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hopper MXFP4 cache/heuristic/autotune requires a CUDA device"
        )
    device = torch.cuda.current_device()
    name = str(torch.cuda.get_device_name(device))
    raw_capability = torch.cuda.get_device_capability(device)
    capability = (int(raw_capability[0]), int(raw_capability[1]))
    properties = torch.cuda.get_device_properties(device)
    sm_count = int(properties.multi_processor_count)
    return name, capability, sm_count


def require_hopper_mxfp4_fused_tuning_device() -> None:
    """Require a live SM90 device for fused cache/heuristic/autotune.

    Fused tactics have no fixed SM partition. Their residency is resolved
    from the live device, while ``group_hint`` remains a performance hint, so
    a product-name or SM-count restriction would reject otherwise legal
    Hopper configurations such as H20.
    """

    actual = _current_hopper_device_identity()
    if actual[1] != (9, 0):
        raise RuntimeError(
            "Hopper MXFP4 fused cache/heuristic/autotune requires SM90 "
            f"(CC 9.0); got {actual!r}"
        )


def require_hopper_mxfp4_tuning_device() -> None:
    """Fail closed unless the split manifest's H200 domain is exact.

    This legacy public name is intentionally retained for compatibility. It
    now guards only paths that consume the frozen 132-SM split partition;
    fused paths use :func:`require_hopper_mxfp4_fused_tuning_device`.
    """

    actual = _current_hopper_device_identity()
    expected = ("NVIDIA H200", (9, 0), 132)
    if actual != expected:
        raise RuntimeError(
            "Hopper MXFP4 split cache/heuristic/autotune candidates are "
            "certified only for standard NVIDIA H200, CC 9.0, 132 SM; "
            f"got {actual!r}"
        )


_IMPLEMENTATION = {
    "fused": "mxfp4_fused",
    "split": "mxfp4_split",
}

# The block-profile artifact hash binds the reviewed combined `discovery_winners_and_unions.json`
# formal aggregate. Candidate, policy, source, and runtime-manifest hashes below are
# regenerated from that exact byte sequence; the exact-balanced/H20 provenance stays separate.
_PROVENANCE = {
    "fused": {
        "artifact_manifest_sha256": "852cbb019a17a9e76991b4f82b5ae64fb3df3245f6e72ec43a20bb407cbbd477",
        "candidate_union_sha256": "dce1c6e5f22ac482928ebc2b07ba682665a41464ae44f367c306d2065c3b1926",
        "domain_sha256": "bff57fb6a658b968e40a8f0c0ec7b4d3e52392070905dae574de808cdeccc6ba",
        "external_schema_version": 1,
        "input_recipe_sha256": "ca6258b91b1b64a7953d9c4e1376f6503085df0af856845b0844f6e4a7829de8",
        "policy_sha256": "d3349c7ca1976b02ab076dcd87617de7552164d640481d7d614256d70da9a4cf",
        "routing_identity_sha256": "bedd230fa8afa768cc841438a91c0156de8fa6954f1af4ebbcb471478805190b",
        "runtime_manifest_sha256": "94ae92677c24630f28192d5df57704ff3ca7d6d94e9a24ebf46b761c198d9bf8",
        "source_manifest_sha256": "1081ba172754b107a9ab2360a6fac8d2e854fd5c1fc45305b571c935885d5277",
        "workload_recipe_sha256": "f4381c345df95b7da21bbe27766427aa5723aefa64f842c04c95fafae7f53352",
    },
    "split": {
        "artifact_manifest_sha256": "852cbb019a17a9e76991b4f82b5ae64fb3df3245f6e72ec43a20bb407cbbd477",
        "candidate_union_sha256": "c673529405013fb4d763190527bd8c27a8911cb4e6efb68a3e0fce25df38f690",
        "domain_sha256": "bff57fb6a658b968e40a8f0c0ec7b4d3e52392070905dae574de808cdeccc6ba",
        "external_schema_version": 1,
        "input_recipe_sha256": "ca6258b91b1b64a7953d9c4e1376f6503085df0af856845b0844f6e4a7829de8",
        "policy_sha256": "d3349c7ca1976b02ab076dcd87617de7552164d640481d7d614256d70da9a4cf",
        "routing_identity_sha256": "bedd230fa8afa768cc841438a91c0156de8fa6954f1af4ebbcb471478805190b",
        "runtime_manifest_sha256": "2f7d34abaaae8cdb3a1f8b8cc9f7f2c25c09e4b3afa50bd9b906a0e4455bf5ed",
        "source_manifest_sha256": "1081ba172754b107a9ab2360a6fac8d2e854fd5c1fc45305b571c935885d5277",
        "workload_recipe_sha256": "f4381c345df95b7da21bbe27766427aa5723aefa64f842c04c95fafae7f53352",
    },
}

MXFP4_TUNING_PROVENANCE = MappingProxyType(
    {mode: MappingProxyType(values) for mode, values in _PROVENANCE.items()}
)

_FROZEN_FUSED_FIELDS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "load_balance_mode",
        "token_back_mode",
        "group_hint",
        "num_sched_stages",
        "in_kernel_fc2_reduce",
    }
)

_FUSED_LAYOUT_FIELDS = frozenset(
    {
        "dedup_dispatch",
        "grouped_token_back",
        "combine_format",
        "active_dispatch_warps",
        "fc1_store_offload",
        "fc1_early_done_publish",
        "fold_producer_warps",
    }
)
_FUSED_FIELDS = _FROZEN_FUSED_FIELDS | _FUSED_LAYOUT_FIELDS

# Historical H200/H20 results predate the latest PR4688 warp-layout knobs.
# Materialize them as the old physical layout instead of silently inheriting
# the latest folded defaults. This preserves their execution semantics while
# giving runtime/cache/candidate identities all constructor-relevant fields.
_FROZEN_FUSED_LAYOUT = MappingProxyType(
    {
        "dedup_dispatch": False,
        "grouped_token_back": False,
        "combine_format": "bf16",
        "active_dispatch_warps": 4,
        "fc1_store_offload": False,
        "fc1_early_done_publish": False,
        "fold_producer_warps": False,
    }
)

_SPLIT_FIELDS = frozenset(
    {
        "k1_mma_tiler_mnk",
        "k2_mma_tiler_mnk",
        "k1_cluster_shape_mnk",
        "k2_cluster_shape_mnk",
        "k1_group_hint",
        "k2_group_hint",
        "k1_num_sched_stages",
        "k2_num_sched_stages",
        "k1_sm_count",
        "k2_sm_count",
        "counter_epoch_banks",
        "graph_variant",
        "enable_iket",
    }
)


def _fused_tactic(
    *,
    tile: tuple[int, int, int],
    cluster: tuple[int, int, int] = (2, 1, 1),
    group_hint: int,
    stages: int,
    pingpong: bool = False,
    load_balance: str = "atomic_counter",
    token_back: str = "epi_warps",
) -> dict[str, Any]:
    return {
        "cluster_shape_mnk": cluster,
        "fp8_accum_mode": "1xacc",
        "group_hint": group_hint,
        "in_kernel_fc2_reduce": False,
        "load_balance_mode": load_balance,
        "mma_tiler_mnk": tile,
        "num_sched_stages": stages,
        "pingpong": pingpong,
        "swap_ab": True,
        "token_back_mode": token_back,
    }


_FUSED_RUNTIME_ANCHORS: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": (
            "26f9703c730e6cf11527e73f8bb8fd4c9adc2e811d8d406c17ad5aa3f79101bd"
        ),
        "tactic": _fused_tactic(
            tile=(128, 16, 256),
            cluster=(2, 1, 1),
            group_hint=78,
            stages=2,
            pingpong=True,
            load_balance="static",
            token_back="epi_warps",
        ),
        "source": "h20_ws8_t1_formal_f2",
    },
    {
        "candidate_id": (
            "d46673da5f8606544a3cfe83b9a150b9e2254ce5cd3173a0c079702b20ae773b"
        ),
        "tactic": _fused_tactic(
            tile=(128, 16, 256),
            cluster=(2, 1, 1),
            group_hint=78,
            stages=1,
            pingpong=True,
            load_balance="static",
            token_back="epi_warps",
        ),
        "source": "h20_ws8_t1_formal_f1",
    },
)


def _split_tactic(
    *,
    banks: int,
    k1_group: int,
    k1_tile: tuple[int, int, int],
    k1_stages: int,
    k2_group: int,
    k2_tile: tuple[int, int, int],
    k2_stages: int,
    k1_sm_count: int = 80,
    k2_sm_count: int = 52,
) -> dict[str, Any]:
    return {
        "counter_epoch_banks": banks,
        "enable_iket": False,
        "graph_variant": "steady_k3_reset",
        "k1_cluster_shape_mnk": (1, 1, 1),
        "k1_group_hint": k1_group,
        "k1_mma_tiler_mnk": k1_tile,
        "k1_num_sched_stages": k1_stages,
        "k1_sm_count": k1_sm_count,
        "k2_cluster_shape_mnk": (1, 1, 1),
        "k2_group_hint": k2_group,
        "k2_mma_tiler_mnk": k2_tile,
        "k2_num_sched_stages": k2_stages,
        "k2_sm_count": k2_sm_count,
    }


# Candidate order is the lexicographically sorted candidate_id order from each
# formal manifest.  Keeping source and parent ids makes every online candidate
# traceable to its offline discovery stage.
_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "fused": (
        {
            "candidate_id": "0a635083f8a826f49f3bc5bbdcbcab108d93e880ffc86335d7d490668d01d8e8",
            "parent_candidate_ids": (
                "4e76d705aea3363bd8aea786e93b0e78acdd634a011d78089628aae5129c5466",
            ),
            "source": "fused_stages",
            "tactic": {
                "active_dispatch_warps": 2,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": True,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": False,
                "fp8_accum_mode": "1xacc",
                "group_hint": 512,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "static",
                "mma_tiler_mnk": (256, 16, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (128,),
        },
        {
            "candidate_id": "0fc1560a23803ffe743e1a6e3f0ecbe5dad93f544066cbe74e4a81b5d21e730a",
            "parent_candidate_ids": (
                "addb2ef34dd3683d188510fc978458be27bca9e9790a8910c494ab250e906074",
            ),
            "source": "fused_stages",
            "tactic": {
                "active_dispatch_warps": 2,
                "cluster_shape_mnk": (1, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": False,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": False,
                "fp8_accum_mode": "1xacc",
                "group_hint": 330,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 32, 128),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
            "parent_candidate_ids": (
                "dc71c194960719fca8b8874aa4472d85cb32aa1ecd4d1881402975ac6746adbb",
            ),
            "requested_tactic": {
                "active_dispatch_warps": 1,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": False,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": True,
                "fp8_accum_mode": "1xacc",
                "group_hint": 512,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 64, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "source": "fused_stages",
            "tactic": {
                "active_dispatch_warps": 1,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": False,
                "fc1_early_done_publish": True,
                "fc1_store_offload": False,
                "fold_producer_warps": True,
                "fp8_accum_mode": "1xacc",
                "group_hint": 512,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 64, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "851b0aa827a934c5e94fee613460b35b2ff19c38c0edf6884ef0f1ba9aa63eb7",
            "parent_candidate_ids": (
                "1369f47bc165ec6e18dc7c1684a291c99fa472b3d138ba6c5efbac0a49d9cd11",
            ),
            "requested_tactic": {
                "active_dispatch_warps": 1,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": False,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": True,
                "fp8_accum_mode": "1xacc",
                "group_hint": 528,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 64, 256),
                "num_sched_stages": 2,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "source": "fused_group",
            "tactic": {
                "active_dispatch_warps": 1,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": False,
                "fc1_early_done_publish": True,
                "fc1_store_offload": False,
                "fold_producer_warps": True,
                "fp8_accum_mode": "1xacc",
                "group_hint": 528,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 64, 256),
                "num_sched_stages": 2,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (1024,),
        },
        {
            "candidate_id": "c8ce6c465c70a34b5c1f5923319d07753efb199bdd164e352239bf53925c61a7",
            "parent_candidate_ids": (
                "54c89ca6198cea1d9a63e073804743b79b09e933719b5eb32059ba1a3dc21f0f",
            ),
            "source": "fused_stages",
            "tactic": {
                "active_dispatch_warps": 1,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": False,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": False,
                "fp8_accum_mode": "1xacc",
                "group_hint": 132,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "static",
                "mma_tiler_mnk": (256, 16, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (64,),
        },
        {
            "candidate_id": "e66dbeaf780401025f44e6543492740a0a0471a0edea7548ada45b7f42a06420",
            "parent_candidate_ids": (
                "05ade503d2a3401a1c2fd6eca5ed889a071c2653ed706870968762f9c096f837",
            ),
            "source": "fused_stages",
            "tactic": {
                "active_dispatch_warps": 4,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": False,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": False,
                "fp8_accum_mode": "1xacc",
                "group_hint": 132,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "static",
                "mma_tiler_mnk": (256, 16, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (8,),
        },
        {
            "candidate_id": "f3f218e0009b41b3f0a2aef60cde244c1faf1d1988e6cd71e7d56020006ca50e",
            "parent_candidate_ids": (
                "0b2e706e6a98bd402ce838673712e28edbcef1c0e6e6884bd25724bff688a612",
            ),
            "source": "fused_stages",
            "tactic": {
                "active_dispatch_warps": 4,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": True,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": False,
                "fp8_accum_mode": "1xacc",
                "group_hint": 132,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "static",
                "mma_tiler_mnk": (256, 16, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (32,),
        },
        {
            "candidate_id": "f573e32fe8c9c8d94754fd7002be6544925c063e77e461290d83505dad8d9c8b",
            "parent_candidate_ids": (
                "d22b11b33850233223f4de0d5991a753da9642fe5b71c2f8987f9dfbef01c3cb",
            ),
            "requested_tactic": {
                "active_dispatch_warps": 1,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": True,
                "fc1_early_done_publish": False,
                "fc1_store_offload": True,
                "fold_producer_warps": True,
                "fp8_accum_mode": "1xacc",
                "group_hint": 512,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 64, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "source": "fused_stages",
            "tactic": {
                "active_dispatch_warps": 1,
                "cluster_shape_mnk": (2, 1, 1),
                "combine_format": "bf16",
                "dedup_dispatch": True,
                "fc1_early_done_publish": True,
                "fc1_store_offload": False,
                "fold_producer_warps": True,
                "fp8_accum_mode": "1xacc",
                "group_hint": 512,
                "grouped_token_back": False,
                "in_kernel_fc2_reduce": False,
                "load_balance_mode": "atomic_counter",
                "mma_tiler_mnk": (256, 64, 256),
                "num_sched_stages": 1,
                "pingpong": False,
                "swap_ab": True,
                "token_back_mode": "epi_warps",
            },
            "winner_for_tokens": (512,),
        },
    ),
    "split": (
        {
            "candidate_id": "3a66b96690ddd716e259e4a2db988755cfc25a3e057d0bb7a78ba18c62f112a2",
            "parent_candidate_ids": (
                "3ddbb3505a5101b495474ca3a806b313112021de2776f97c41d32ed48453b61a",
            ),
            "source": "split_graph_bank",
            "tactic": {
                "counter_epoch_banks": 1,
                "enable_iket": False,
                "graph_variant": "cold_k0",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 512,
                "k1_mma_tiler_mnk": (256, 64, 256),
                "k1_num_sched_stages": 1,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 396,
                "k2_mma_tiler_mnk": (256, 64, 256),
                "k2_num_sched_stages": 2,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "47727d195a5a8068aabc1e638bdcc47ad1ea9bcfe1a469c3c41e539181a45ee7",
            "parent_candidate_ids": (
                "e98d1d5c645df69e9dc2538e8920792f7141cfe5014a1612864317e9f433c0b8",
            ),
            "source": "split_graph_bank",
            "tactic": {
                "counter_epoch_banks": 2,
                "enable_iket": False,
                "graph_variant": "steady_k3_reset",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 128,
                "k1_mma_tiler_mnk": (256, 32, 128),
                "k1_num_sched_stages": 1,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 528,
                "k2_mma_tiler_mnk": (256, 32, 256),
                "k2_num_sched_stages": 2,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "49b828034913e102790cf5bbe9768e6b17661ea26af15ef0ff6f051b2a36e8c6",
            "parent_candidate_ids": (
                "aac2d63a19211c69830eb934762a4d62513d545b42e5a51118187b47add49b8c",
            ),
            "source": "split_graph_bank",
            "tactic": {
                "counter_epoch_banks": 2,
                "enable_iket": False,
                "graph_variant": "steady_k3_reset",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 396,
                "k1_mma_tiler_mnk": (256, 64, 128),
                "k1_num_sched_stages": 3,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 512,
                "k2_mma_tiler_mnk": (256, 64, 256),
                "k2_num_sched_stages": 2,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (1024,),
        },
        {
            "candidate_id": "5e2563323984c3bf2f9686029a5809d7240a8e1dc56cbd9ead4975481b1695ef",
            "parent_candidate_ids": (
                "59aaa4eafcbf279f8949bd1125ed1788ad80f7a793be94bdc083742b2d5e91eb",
            ),
            "source": "split_graph_bank",
            "tactic": {
                "counter_epoch_banks": 2,
                "enable_iket": False,
                "graph_variant": "steady_k3_reset",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 256,
                "k1_mma_tiler_mnk": (256, 16, 128),
                "k1_num_sched_stages": 1,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 330,
                "k2_mma_tiler_mnk": (256, 16, 128),
                "k2_num_sched_stages": 2,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (32,),
        },
        {
            "candidate_id": "60faf6bafe1c0f21ca79b241f3dbb82e328cca3769749f1061a513a1cf7f9c1b",
            "parent_candidate_ids": (
                "99ac967e38c756f33bbd3291d54a3642bc73b2d06154662c0aefd477571dd242",
            ),
            "source": "split_stages",
            "tactic": {
                "counter_epoch_banks": 1,
                "enable_iket": False,
                "graph_variant": "steady_k3_reset",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 528,
                "k1_mma_tiler_mnk": (256, 16, 128),
                "k1_num_sched_stages": 1,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 128,
                "k2_mma_tiler_mnk": (256, 16, 128),
                "k2_num_sched_stages": 1,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (128,),
        },
        {
            "candidate_id": "748ef68a8b715647dbf20cb8e9055bfde13170ef24981c4b22d0deca6e727efe",
            "parent_candidate_ids": (
                "42fae5e4c7f0a78cd121abbf404989cf451b011608ece64e82065136c454a876",
            ),
            "source": "split_graph_bank",
            "tactic": {
                "counter_epoch_banks": 2,
                "enable_iket": False,
                "graph_variant": "steady_k3_reset",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 528,
                "k1_mma_tiler_mnk": (256, 64, 256),
                "k1_num_sched_stages": 3,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 132,
                "k2_mma_tiler_mnk": (256, 32, 256),
                "k2_num_sched_stages": 1,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (512,),
        },
        {
            "candidate_id": "7b623d13c42368a6ab0ba227dc29b0eef0858fcfc81db6b54fbd08742e306ab8",
            "parent_candidate_ids": (
                "06dd0d6dd664f322725b3cb926b59f56b758b966aeabf00cb6cc7080c072766f",
            ),
            "source": "split_stages",
            "tactic": {
                "counter_epoch_banks": 1,
                "enable_iket": False,
                "graph_variant": "steady_k3_reset",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 64,
                "k1_mma_tiler_mnk": (256, 16, 128),
                "k1_num_sched_stages": 1,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 264,
                "k2_mma_tiler_mnk": (256, 16, 128),
                "k2_num_sched_stages": 1,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (64,),
        },
        {
            "candidate_id": "ce1da82f795a4c8f4f61200037bc4b94a36d7a16de2dc6ad7e9aca3c999508bb",
            "parent_candidate_ids": (
                "c4129f0b00dc299cfd2e92aeb830b5a9a0bb8a8f23ffcd9247fdc6ff63f098f8",
            ),
            "source": "split_graph_bank",
            "tactic": {
                "counter_epoch_banks": 2,
                "enable_iket": False,
                "graph_variant": "steady_k3_reset",
                "k1_cluster_shape_mnk": (1, 1, 1),
                "k1_group_hint": 330,
                "k1_mma_tiler_mnk": (256, 16, 256),
                "k1_num_sched_stages": 2,
                "k1_sm_count": 80,
                "k2_cluster_shape_mnk": (1, 1, 1),
                "k2_group_hint": 264,
                "k2_mma_tiler_mnk": (256, 16, 128),
                "k2_num_sched_stages": 3,
                "k2_sm_count": 52,
            },
            "winner_for_tokens": (8,),
        },
    ),
}

_WINNERS: dict[str, dict[int, dict[str, Any]]] = {
    "fused": {
        8: {
            "candidate_id": "e66dbeaf780401025f44e6543492740a0a0471a0edea7548ada45b7f42a06420",
            "median_score_us": 480.464011,
            "relative_spread": 0.004229236640993683,
            "scores_us": (479.328007, 481.360003, 480.464011),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "f3f218e0009b41b3f0a2aef60cde244c1faf1d1988e6cd71e7d56020006ca50e",
            "median_score_us": 837.408006,
            "relative_spread": 0.0032290042376308093,
            "scores_us": (837.408006, 835.696012, 838.400006),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "c8ce6c465c70a34b5c1f5923319d07753efb199bdd164e352239bf53925c61a7",
            "median_score_us": 919.631988,
            "relative_spread": 0.00508028326652775,
            "scores_us": (923.584014, 919.631988, 918.912023),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "0a635083f8a826f49f3bc5bbdcbcab108d93e880ffc86335d7d490668d01d8e8",
            "median_score_us": 954.30398,
            "relative_spread": 0.0020790021225731837,
            "scores_us": (953.247994, 955.231994, 954.30398),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "0fc1560a23803ffe743e1a6e3f0ecbe5dad93f544066cbe74e4a81b5d21e730a",
            "median_score_us": 1037.935972,
            "relative_spread": 0.0019114859235268426,
            "scores_us": (1038.800001, 1036.816001, 1037.935972),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "f573e32fe8c9c8d94754fd7002be6544925c063e77e461290d83505dad8d9c8b",
            "median_score_us": 1332.896054,
            "relative_spread": 0.0017045665287866,
            "scores_us": (1331.856012, 1334.128022, 1332.896054),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "851b0aa827a934c5e94fee613460b35b2ff19c38c0edf6884ef0f1ba9aa63eb7",
            "median_score_us": 1772.911966,
            "relative_spread": 0.0015703058320945815,
            "scores_us": (1771.200001, 1772.911966, 1773.984015),
            "telemetry_warning": False,
        },
        2048: {
            "candidate_id": "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
            "median_score_us": 3000.0,
            "relative_spread": 0.00256534433333339,
            "scores_us": (3007.568002, 2999.871969, 3000.0),
            "telemetry_warning": True,
        },
    },
    "split": {
        8: {
            "candidate_id": "ce1da82f795a4c8f4f61200037bc4b94a36d7a16de2dc6ad7e9aca3c999508bb",
            "median_score_us": 526.015997,
            "relative_spread": 0.0053534892780076,
            "scores_us": (526.015997, 526.639998, 523.823977),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "5e2563323984c3bf2f9686029a5809d7240a8e1dc56cbd9ead4975481b1695ef",
            "median_score_us": 882.671982,
            "relative_spread": 0.007141946417870987,
            "scores_us": (888.303995, 881.999999, 882.671982),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "7b623d13c42368a6ab0ba227dc29b0eef0858fcfc81db6b54fbd08742e306ab8",
            "median_score_us": 978.015989,
            "relative_spread": 0.003713649920706895,
            "scores_us": (976.495981, 978.015989, 980.12799),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "60faf6bafe1c0f21ca79b241f3dbb82e328cca3769749f1061a513a1cf7f9c1b",
            "median_score_us": 1015.663981,
            "relative_spread": 0.00890052041729341,
            "scores_us": (1012.912035, 1021.951973, 1015.663981),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "47727d195a5a8068aabc1e638bdcc47ad1ea9bcfe1a469c3c41e539181a45ee7",
            "median_score_us": 1095.02399,
            "relative_spread": 0.0027031608686490702,
            "scores_us": (1095.02399, 1097.263992, 1094.303966),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "748ef68a8b715647dbf20cb8e9055bfde13170ef24981c4b22d0deca6e727efe",
            "median_score_us": 1451.312006,
            "relative_spread": 0.0012898694369377227,
            "scores_us": (1449.855983, 1451.312006, 1451.727986),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "49b828034913e102790cf5bbe9768e6b17661ea26af15ef0ff6f051b2a36e8c6",
            "median_score_us": 1960.431993,
            "relative_spread": 0.0004407339826554612,
            "scores_us": (1960.431993, 1961.120009, 1960.25598),
            "telemetry_warning": True,
        },
        2048: {
            "candidate_id": "3a66b96690ddd716e259e4a2db988755cfc25a3e057d0bb7a78ba18c62f112a2",
            "median_score_us": 3310.720086,
            "relative_spread": 0.0018171321778122792,
            "scores_us": (3310.096025, 3310.720086, 3316.112041),
            "telemetry_warning": True,
        },
    },
}

_EXACT_PROVENANCE = {
    "fused": {
        "artifact_manifest_sha256": (
            "62733c7605f7233ac81c341084e0d589f4a91ca3f1aaaf1fac0660f7d1842a61"
        ),
        "domain_sha256": (
            "86b846926d510fb013132462cf88122e36479db5b374645cfa3951021becf1a6"
        ),
        "external_schema_version": 1,
        "policy_sha256": (
            "88503f1be444226eed1cf59d1083de5ffa92491d9bb5c626dcf7f96d22013137"
        ),
        "runtime_manifest_sha256": (
            "f4112c7d0d7ead640239c1df3d7f4af74e2a1fb35cf5e821edd1beba9bba3e99"
        ),
        "source_manifest_sha256": (
            "bcc0448df03348eb82addb5513152f7c7db42673769a40f9d67600d72c08a689"
        ),
        "workload_recipe_sha256": (
            "e53d38f2f6fc708fb93a6405ad2129b49a3ca5499e34083e1aa2c706c9eacbfd"
        ),
    },
    "split": {
        "artifact_manifest_sha256": (
            "094d840c579a7331439d1acd50690909ad2c88e6085253326c6f2d98ddad248a"
        ),
        "candidate_union_sha256": (
            "210adb840e66c0f44949ea866a8bbfa9f5b2b3835ee8992e57b5edb75f8b9321"
        ),
        "domain_sha256": (
            "86b846926d510fb013132462cf88122e36479db5b374645cfa3951021becf1a6"
        ),
        "external_schema_version": 2,
        "high_policy_sha256": (
            "1753d198bfaf7335a0998fe660642a3603b7448535f93435da2dcc2fd5d8f747"
        ),
        "low_policy_sha256": (
            "8069be3056f7a9235a1e67bcf3cd3da89fdec441e3735b8b6259f74895a8735d"
        ),
        "provenance_model": (
            "per-token source/policy/root bindings; no synthetic unified policy_sha256"
        ),
        "runtime_manifest_sha256": (
            "97a4a40bffeb062b9cc916959186e308bb8e3ff52150dab8a01b3273af22261f"
        ),
        "source_manifest_sha256": (
            "bcc0448df03348eb82addb5513152f7c7db42673769a40f9d67600d72c08a689"
        ),
        "workload_recipe_sha256": (
            "e53d38f2f6fc708fb93a6405ad2129b49a3ca5499e34083e1aa2c706c9eacbfd"
        ),
    },
}

_EXACT_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "fused": (
        {
            "candidate_id": "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
            "tactic": _fused_tactic(tile=(256, 32, 128), group_hint=396, stages=2),
            "source": "fused_group",
            "parent_candidate_ids": (
                "8ef6a914cd598b8e1ed7753396d282020d907fc10a9ea9eaf46a56214eee73dd",
            ),
            "winner_for_tokens": (512,),
        },
        {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=512, stages=1),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
            ),
            "winner_for_tokens": (32, 64),
        },
        {
            "candidate_id": "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
            "tactic": _fused_tactic(
                tile=(128, 64, 256),
                group_hint=330,
                stages=2,
                token_back="reuse_dispatch_warps",
            ),
            "source": "fused_load",
            "parent_candidate_ids": (
                "f6cc04937e556078a2f039a414a4131da8beead85c37a13a84ea3b4769b2438d",
            ),
            "winner_for_tokens": (1024,),
        },
        {
            "candidate_id": "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
            "tactic": _fused_tactic(
                tile=(256, 16, 256),
                group_hint=396,
                stages=2,
                load_balance="static",
            ),
            "source": "fused_group",
            "parent_candidate_ids": (
                "08387b9c3c7e64639cf97a3fa92804fe5615850fa9d2105f733b97ad96321196",
            ),
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=396, stages=1),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "f3bded653245824c8cd8a1b67605608757b5df569da3dca409302e94e7baf7a4",
            ),
            "winner_for_tokens": (8,),
        },
        {
            "candidate_id": "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
            "tactic": _fused_tactic(
                tile=(128, 64, 256),
                cluster=(1, 1, 1),
                group_hint=528,
                stages=2,
                token_back="reuse_dispatch_warps",
            ),
            "source": "fused_load",
            "parent_candidate_ids": (
                "2b24fef280ba7168a0041f96990bfa93bee2570b31f3a0365a006b7f0f201813",
            ),
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=512, stages=2),
            "source": "fused_group",
            "parent_candidate_ids": (
                "6764dee4fcd21556a13c148e2eca6bd30c642d3e7a44e9e5c0a1a66c8ea75f5f",
            ),
            "winner_for_tokens": (128,),
        },
    ),
    "split": (
        {
            "candidate_id": "302ce3733b64947c20ec514e6107d2c0b7e6f305b078d5fe3b9a12421a276aee",
            "tactic": _split_tactic(
                banks=2,
                k1_group=256,
                k1_tile=(256, 16, 256),
                k1_stages=3,
                k2_group=330,
                k2_tile=(256, 16, 256),
                k2_stages=3,
                k1_sm_count=88,
                k2_sm_count=44,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "4efd035fd9bbfacbef6787ccd8b311530b4e21245ff5f433721f94eb08234d47",
            ),
            "winner_for_tokens": (8,),
        },
        {
            "candidate_id": "43ab071b691a2153a14e9173f5a99b4c167ad49f4e4160f25de5385cd4f5b634",
            "tactic": _split_tactic(
                banks=2,
                k1_group=396,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=512,
                k2_tile=(256, 16, 128),
                k2_stages=1,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "72e080ec57447dc33c70154a81e85105e13c1cb11760bfc2f5528653e97ff005",
            ),
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "47254cc9fcd2fae00d7ee5236e1b7e19fbe7c0b14da2b2ca5f6572af75f49a0c",
            "tactic": _split_tactic(
                banks=2,
                k1_group=528,
                k1_tile=(256, 64, 256),
                k1_stages=2,
                k2_group=528,
                k2_tile=(256, 64, 256),
                k2_stages=2,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "462d763bc23efc9777611f99a1b87d327aa8958e4e71bd406756ea315a9bd7f4",
            ),
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "845359c757ce881e2d98077f209c02721dc95b66771dc7149b5f1e900631f355",
            "tactic": _split_tactic(
                banks=2,
                k1_group=512,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=512,
                k2_tile=(256, 16, 128),
                k2_stages=1,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "dd27c6d35a943f035151c28964193336247bec4d576b16303cab9a937ac3d5c9",
            ),
            "winner_for_tokens": (128,),
        },
        {
            "candidate_id": "9a16b958e9f3acb42df1ba4ae66175a3a5021c0d05fffa606d9a113d403b6b1b",
            "tactic": _split_tactic(
                banks=1,
                k1_group=512,
                k1_tile=(256, 64, 256),
                k1_stages=2,
                k2_group=528,
                k2_tile=(256, 64, 256),
                k2_stages=1,
            ),
            "source": "split_stages",
            "parent_candidate_ids": (
                "686c6f58c90e8ec5124794157c3e121f04a84cda560ff893dab0c09cd918fa8d",
            ),
            "winner_for_tokens": (1024,),
        },
        {
            "candidate_id": "9e7fd2c153bdcc2cf8477b913431e39be2040a6d49f31fffddcfafd94af0494f",
            "tactic": _split_tactic(
                banks=1,
                k1_group=128,
                k1_tile=(256, 16, 128),
                k1_stages=2,
                k2_group=132,
                k2_tile=(256, 32, 256),
                k2_stages=2,
            ),
            "source": "split_k1_group",
            "parent_candidate_ids": (
                "7695f9dd75e66a74aaf91ed75224689b163dffcba43797321f66cced78e39871",
            ),
            "winner_for_tokens": (64,),
        },
        {
            "candidate_id": "c052d631fb1e1c9b0a2a3890b789d9f48007955530aa38366e87860f25028e94",
            "tactic": _split_tactic(
                banks=2,
                k1_group=256,
                k1_tile=(256, 16, 128),
                k1_stages=2,
                k2_group=132,
                k2_tile=(256, 32, 256),
                k2_stages=2,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "5c2ec0a2c6828bfc0581a2b51fdc7774c019a4d3d63b00e7c2cf00ea7c09e5a9",
            ),
            "winner_for_tokens": (32,),
        },
        {
            "candidate_id": "f6ad358a9b115bcaaf2c6222ec782ccfc98a8c6a9d393a78ed187713c621795a",
            "tactic": _split_tactic(
                banks=2,
                k1_group=512,
                k1_tile=(256, 32, 128),
                k1_stages=2,
                k2_group=330,
                k2_tile=(256, 32, 256),
                k2_stages=3,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "8533cb0e0b3ccfc144a498ab1c3a51031c00baa61b12ddbcd3026684febe9f6c",
            ),
            "winner_for_tokens": (512,),
        },
    ),
}

_EXACT_WINNERS: dict[str, dict[int, dict[str, Any]]] = {
    "fused": {
        8: {
            "candidate_id": "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
            "median_score_us": 519.807994,
            "relative_spread": 0.0015082684549864128,
            "scores_us": (520.352006, 519.807994, 519.567996),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "median_score_us": 937.472016,
            "relative_spread": 0.005359110367300778,
            "scores_us": (937.472016, 934.816003, 939.840019),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "median_score_us": 948.416024,
            "relative_spread": 0.003525878849976127,
            "scores_us": (948.336005, 948.416024, 951.680005),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
            "median_score_us": 963.279992,
            "relative_spread": 0.006228731054137874,
            "scores_us": (963.279992, 963.904023, 957.904011),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
            "median_score_us": 994.527996,
            "relative_spread": 0.0006917713757350173,
            "scores_us": (994.800001, 994.527996, 994.112015),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
            "median_score_us": 1097.21601,
            "relative_spread": 0.002697760489295057,
            "scores_us": (1096.127987, 1097.21601, 1099.088013),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
            "median_score_us": 1783.792019,
            "relative_spread": 0.0023948890646987332,
            "scores_us": (1783.13601, 1783.792019, 1787.407994),
            "telemetry_warning": False,
        },
        2048: {
            "candidate_id": "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
            "median_score_us": 3483.951926,
            "relative_spread": 0.001786487624456375,
            "scores_us": (3483.951926, 3488.576055, 3482.352018),
            "telemetry_warning": True,
        },
    },
    "split": {
        8: {
            "candidate_id": "302ce3733b64947c20ec514e6107d2c0b7e6f305b078d5fe3b9a12421a276aee",
            "median_score_us": 544.384003,
            "relative_spread": 0.0032036246296532397,
            "scores_us": (545.231998, 543.487996, 544.384003),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "c052d631fb1e1c9b0a2a3890b789d9f48007955530aa38366e87860f25028e94",
            "median_score_us": 972.81599,
            "relative_spread": 0.009325479939942136,
            "scores_us": (972.81599, 970.080018, 979.151994),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "9e7fd2c153bdcc2cf8477b913431e39be2040a6d49f31fffddcfafd94af0494f",
            "median_score_us": 978.991985,
            "relative_spread": 0.00400413799097653,
            "scores_us": (978.991985, 981.263995, 977.343976),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "845359c757ce881e2d98077f209c02721dc95b66771dc7149b5f1e900631f355",
            "median_score_us": 985.95202,
            "relative_spread": 0.00149296920148304,
            "scores_us": (985.359997, 985.95202, 986.831993),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "43ab071b691a2153a14e9173f5a99b4c167ad49f4e4160f25de5385cd4f5b634",
            "median_score_us": 1019.151986,
            "relative_spread": 0.003673648338453037,
            "scores_us": (1017.215967, 1020.959973, 1019.151986),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "f6ad358a9b115bcaaf2c6222ec782ccfc98a8c6a9d393a78ed187713c621795a",
            "median_score_us": 1144.97602,
            "relative_spread": 0.0018725099587674724,
            "scores_us": (1144.97602, 1145.26397, 1143.119991),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "9a16b958e9f3acb42df1ba4ae66175a3a5021c0d05fffa606d9a113d403b6b1b",
            "median_score_us": 1569.82404,
            "relative_spread": 0.0009376821621358214,
            "scores_us": (1568.751991, 1570.223987, 1569.82404),
            "telemetry_warning": False,
        },
        2048: {
            "candidate_id": "47254cc9fcd2fae00d7ee5236e1b7e19fbe7c0b14da2b2ca5f6572af75f49a0c",
            "median_score_us": 2875.983953,
            "relative_spread": 0.0028539922802552553,
            "scores_us": (2874.992013, 2875.983953, 2883.200049),
            "telemetry_warning": True,
        },
    },
}

_PROFILE_PROVENANCE = {
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: _PROVENANCE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: _EXACT_PROVENANCE,
}
_PROFILE_CANDIDATES = {
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: _CANDIDATES,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: _EXACT_CANDIDATES,
}
_PROFILE_WINNERS = {
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: _WINNERS,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: _EXACT_WINNERS,
}

MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE = MappingProxyType(
    {
        profile: MappingProxyType(
            {mode: MappingProxyType(values) for mode, values in per_mode.items()}
        )
        for profile, per_mode in _PROFILE_PROVENANCE.items()
    }
)

# Filled from the canonical JSON serialization of the compact runtime
# projections returned by ``hopper_mxfp4_tuning_manifest``.  These hashes are
# independent of the larger external artifact hashes above.
_EXACT_RUNTIME_MANIFEST_SHA256 = {
    mode: values["runtime_manifest_sha256"]
    for mode, values in _EXACT_PROVENANCE.items()
}


def _mode(execution_mode: str) -> Mxfp4ExecutionMode:
    if execution_mode not in _IMPLEMENTATION:
        raise ValueError(
            f"execution_mode must be exactly 'fused' or 'split', got {execution_mode!r}"
        )
    return execution_mode  # type: ignore[return-value]


def normalize_hopper_mxfp4_routing_profile(
    routing_profile: str,
) -> Mxfp4RoutingProfile:
    """Compatibility wrapper around the canonical SM90 profile normalizer."""

    return normalize_sm90_routing_profile(routing_profile)  # type: ignore[return-value]


def hopper_mxfp4_tuning_provenance(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> Mapping[str, Any]:
    """Return immutable provenance for one execution/routing domain."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE[profile][mode]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-bool integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an M,N,K triple")
    return tuple(_positive_int(item, name) for item in value)  # type: ignore[return-value]


def _require_exact_fields(
    tactic: Mapping[str, Any], expected: frozenset[str], implementation: str
) -> None:
    actual = set(tactic)
    if actual != expected:
        raise ValueError(
            f"{implementation} tactic fields differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _materialize_frozen_fused_tactic(
    tactic: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the explicit pre-PR4688-layout semantics to a frozen tactic."""

    _require_exact_fields(tactic, _FROZEN_FUSED_FIELDS, _IMPLEMENTATION["fused"])
    return {**dict(tactic), **dict(_FROZEN_FUSED_LAYOUT)}


def validate_hopper_mxfp4_tactic(
    tactic: Mapping[str, Any], *, execution_mode: str, total_sms: int = 132
) -> dict[str, Any]:
    """Validate and normalize one tactic from the bounded Hopper search domain.

    The returned mapping is a fresh copy whose tile and cluster fields are
    tuples. No defaults are inserted: callers must preserve the complete
    current fused or split tactic identity. Frozen pre-layout manifest rows
    are normalized only by private provenance helpers below.
    """

    mode = _mode(execution_mode)
    if not isinstance(tactic, Mapping):
        raise TypeError(f"tactic must be a mapping, got {type(tactic).__name__}")

    if mode == "fused":
        _require_exact_fields(tactic, _FUSED_FIELDS, _IMPLEMENTATION[mode])
        if tactic["swap_ab"] is not True:
            raise ValueError("MXFP4 fused requires swap_ab=true")
        if not isinstance(tactic["pingpong"], bool):
            raise ValueError("MXFP4 fused pingpong must be bool")
        tile = _triple(tactic["mma_tiler_mnk"], "mma_tiler_mnk")
        m, n, k = tile
        if (
            m not in (128, 256)
            or n not in (16, 32, 64, 128)
            or k
            not in (
                128,
                256,
            )
        ):
            raise ValueError("illegal MXFP4 fused tile")
        if tactic["pingpong"] and m != 128:
            raise ValueError("MXFP4 fused pingpong requires M128")
        cluster = _triple(tactic["cluster_shape_mnk"], "cluster_shape_mnk")
        if cluster not in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)):
            raise ValueError("illegal MXFP4 fused cluster")
        if tactic["fp8_accum_mode"] != "1xacc":
            raise ValueError("MXFP4 fused fixes 1xacc")
        if tactic["load_balance_mode"] not in ("static", "atomic_counter"):
            raise ValueError("illegal MXFP4 fused load balance")
        if tactic["token_back_mode"] not in (
            "epi_warps",
            "reuse_dispatch_warps",
            "standalone_warps",
        ):
            raise ValueError("illegal MXFP4 fused token-back")
        group_hint = _optional_positive_int(tactic["group_hint"], "group_hint")
        stages = _optional_positive_int(tactic["num_sched_stages"], "num_sched_stages")
        if tactic["in_kernel_fc2_reduce"] is not False:
            raise ValueError("MXFP4 fused fixes in-kernel reduce false")
        for name in (
            "dedup_dispatch",
            "grouped_token_back",
            "fc1_store_offload",
            "fc1_early_done_publish",
            "fold_producer_warps",
        ):
            if not isinstance(tactic[name], bool):
                raise ValueError(f"MXFP4 fused {name} must be bool")
        active_dispatch_warps = tactic["active_dispatch_warps"]
        if isinstance(active_dispatch_warps, bool) or active_dispatch_warps not in (
            1,
            2,
            4,
        ):
            raise ValueError("MXFP4 fused active_dispatch_warps must be 1, 2, or 4")
        if tactic["fold_producer_warps"] and active_dispatch_warps != 1:
            raise ValueError(
                "MXFP4 fused fold_producer_warps=True requires active_dispatch_warps=1"
            )
        if tactic["grouped_token_back"] is not False:
            raise ValueError(
                "MXFP4 fused production BF16-combine tactics fix "
                "grouped_token_back=false"
            )
        if tactic["combine_format"] != "bf16":
            raise ValueError("MXFP4 fused production tactics fix combine_format='bf16'")
        return {
            "active_dispatch_warps": active_dispatch_warps,
            "cluster_shape_mnk": cluster,
            "combine_format": "bf16",
            "dedup_dispatch": tactic["dedup_dispatch"],
            "fc1_early_done_publish": tactic["fc1_early_done_publish"],
            "fc1_store_offload": tactic["fc1_store_offload"],
            "fold_producer_warps": tactic["fold_producer_warps"],
            "fp8_accum_mode": "1xacc",
            "group_hint": group_hint,
            "grouped_token_back": False,
            "in_kernel_fc2_reduce": False,
            "load_balance_mode": tactic["load_balance_mode"],
            "mma_tiler_mnk": tile,
            "num_sched_stages": stages,
            "pingpong": tactic["pingpong"],
            "swap_ab": True,
            "token_back_mode": tactic["token_back_mode"],
        }

    _require_exact_fields(tactic, _SPLIT_FIELDS, _IMPLEMENTATION[mode])
    k1 = _triple(tactic["k1_mma_tiler_mnk"], "k1_mma_tiler_mnk")
    k2 = _triple(tactic["k2_mma_tiler_mnk"], "k2_mma_tiler_mnk")
    for role, tile in (("K1", k1), ("K2", k2)):
        if (
            tile[0] not in (128, 256)
            or tile[1] not in (16, 32, 64, 128)
            or tile[2] not in (128, 256)
        ):
            raise ValueError(f"illegal split {role} tile")
    c1 = _triple(tactic["k1_cluster_shape_mnk"], "k1_cluster_shape_mnk")
    c2 = _triple(tactic["k2_cluster_shape_mnk"], "k2_cluster_shape_mnk")
    if c1 != c2:
        raise ValueError("split K1/K2 clusters must match")
    if c1 != (1, 1, 1):
        raise ValueError(
            "non-1x1x1 split clusters are quarantined correctness failures"
        )
    k1_group = _optional_positive_int(tactic["k1_group_hint"], "k1_group_hint")
    k2_group = _optional_positive_int(tactic["k2_group_hint"], "k2_group_hint")
    k1_stages = _optional_positive_int(
        tactic["k1_num_sched_stages"], "k1_num_sched_stages"
    )
    k2_stages = _optional_positive_int(
        tactic["k2_num_sched_stages"], "k2_num_sched_stages"
    )
    k1_sms = _positive_int(tactic["k1_sm_count"], "k1_sm_count")
    k2_sms = _positive_int(tactic["k2_sm_count"], "k2_sm_count")
    expected_sms = _positive_int(total_sms, "total_sms")
    if k1_sms + k2_sms != expected_sms or k1_sms % 8:
        raise ValueError(
            f"split partition must sum to {expected_sms} SMs; "
            "only K1 is alignment-constrained"
        )
    banks = tactic["counter_epoch_banks"]
    if isinstance(banks, bool) or banks not in (1, 2):
        raise ValueError("split counter banks must be 1 or 2")
    graph = tactic["graph_variant"]
    if graph not in ("cold_k0", "steady_k3_reset"):
        raise ValueError("illegal split graph variant")
    if banks == 2 and graph != "steady_k3_reset":
        raise ValueError("two split counter banks require steady_k3_reset")
    if tactic["enable_iket"] is not False:
        raise ValueError("first formal split domain fixes IKET false")
    return {
        "counter_epoch_banks": banks,
        "enable_iket": False,
        "graph_variant": graph,
        "k1_cluster_shape_mnk": c1,
        "k1_group_hint": k1_group,
        "k1_mma_tiler_mnk": k1,
        "k1_num_sched_stages": k1_stages,
        "k1_sm_count": k1_sms,
        "k2_cluster_shape_mnk": c2,
        "k2_group_hint": k2_group,
        "k2_mma_tiler_mnk": k2,
        "k2_num_sched_stages": k2_stages,
        "k2_sm_count": k2_sms,
    }


def _validate_frozen_hopper_mxfp4_tactic(
    tactic: Mapping[str, Any], *, execution_mode: str
) -> dict[str, Any]:
    """Validate one immutable historical manifest tactic without rewriting it."""

    mode = _mode(execution_mode)
    if mode == "split":
        return validate_hopper_mxfp4_tactic(tactic, execution_mode=mode)
    complete = validate_hopper_mxfp4_tactic(
        _materialize_frozen_fused_tactic(tactic), execution_mode=mode
    )
    return {name: complete[name] for name in _FROZEN_FUSED_FIELDS}


def _validate_profile_manifest_tactic(
    tactic: Mapping[str, Any],
    *,
    execution_mode: Mxfp4ExecutionMode,
    routing_profile: Mxfp4RoutingProfile,
) -> dict[str, Any]:
    """Validate the tactic identity stored by one profile manifest.

    Only the published-exact fused manifest predates the PR4688 layout fields.
    Its ten-field identity must stay byte-for-byte frozen; the refreshed block
    manifest stores complete modern tactics.
    """

    if (
        routing_profile == MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
        and execution_mode == "fused"
    ):
        return _validate_frozen_hopper_mxfp4_tactic(
            tactic, execution_mode=execution_mode
        )
    return validate_hopper_mxfp4_tactic(tactic, execution_mode=execution_mode)


def _materialize_profile_tactic(
    tactic: Mapping[str, Any],
    *,
    execution_mode: Mxfp4ExecutionMode,
    routing_profile: Mxfp4RoutingProfile,
) -> dict[str, Any]:
    """Return the complete tactic executed by a profile candidate."""

    stored = _validate_profile_manifest_tactic(
        tactic,
        execution_mode=execution_mode,
        routing_profile=routing_profile,
    )
    if (
        routing_profile == MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
        and execution_mode == "fused"
    ):
        return validate_hopper_mxfp4_tactic(
            _materialize_frozen_fused_tactic(stored), execution_mode=execution_mode
        )
    return stored


def is_valid_hopper_mxfp4_tactic(
    tactic: Mapping[str, Any], *, execution_mode: str
) -> bool:
    """Return whether ``tactic`` is a complete legal fused/split tactic."""

    try:
        validate_hopper_mxfp4_tactic(tactic, execution_mode=execution_mode)
    except (TypeError, ValueError):
        return False
    return True


def _candidate_id(implementation: str, tactic: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"implementation": implementation, "tactic": tactic},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _fused_runtime_candidate_records() -> tuple[dict[str, Any], ...]:
    """Merge modern block, frozen exact, and H20 fused candidates by tactic."""

    by_id: dict[str, dict[str, Any]] = {}
    sources = (
        (_CANDIDATES["fused"], False),
        (_EXACT_CANDIDATES["fused"], True),
        (_FUSED_RUNTIME_ANCHORS, True),
    )
    for records, is_frozen in sources:
        for record in records:
            if is_frozen:
                source_tactic = _validate_frozen_hopper_mxfp4_tactic(
                    record["tactic"], execution_mode="fused"
                )
            else:
                source_tactic = validate_hopper_mxfp4_tactic(
                    record["tactic"], execution_mode="fused"
                )
            source_candidate_id = str(record["candidate_id"])
            expected_source_id = _candidate_id(_IMPLEMENTATION["fused"], source_tactic)
            if source_candidate_id != expected_source_id:
                raise RuntimeError(
                    f"corrupt fused candidate id {source_candidate_id}; "
                    f"expected {expected_source_id}"
                )
            if is_frozen:
                tactic = validate_hopper_mxfp4_tactic(
                    _materialize_frozen_fused_tactic(source_tactic),
                    execution_mode="fused",
                )
            else:
                tactic = source_tactic
            candidate_id = _candidate_id(_IMPLEMENTATION["fused"], tactic)
            previous = by_id.get(candidate_id)
            if previous is not None:
                if previous["tactic"] != tactic:
                    raise RuntimeError(
                        f"fused runtime candidate id collision {candidate_id}"
                    )
                continue
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "source_candidate_id": source_candidate_id,
                "tactic": tactic,
            }
    return tuple(by_id[candidate_id] for candidate_id in sorted(by_id))


def _candidate_union_sha256(records: tuple[dict[str, Any], ...]) -> str:
    canonical = json.dumps(
        [
            {
                "candidate_id": record["candidate_id"],
                "tactic": record["tactic"],
            }
            for record in records
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


MXFP4_FUSED_RUNTIME_CANDIDATE_UNION_SHA256 = _candidate_union_sha256(
    _fused_runtime_candidate_records()
)


def _candidate_by_id(
    mode: Mxfp4ExecutionMode, routing_profile: Mxfp4RoutingProfile
) -> dict[str, dict[str, Any]]:
    return {
        record["candidate_id"]: record
        for record in _PROFILE_CANDIDATES[routing_profile][mode]
    }


def _manifest_candidate(
    profile: Mxfp4RoutingProfile,
    mode: Mxfp4ExecutionMode,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    effective = _validate_profile_manifest_tactic(
        record["tactic"], execution_mode=mode, routing_profile=profile
    )
    candidate = {
        "candidate_id": record["candidate_id"],
        "effective_tactic": copy.deepcopy(effective),
        "implementation": _IMPLEMENTATION[mode],
        "parent_candidate_ids": list(record["parent_candidate_ids"]),
        "requested_tactic": copy.deepcopy(effective),
        "source": record["source"],
    }
    if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
        requested = validate_hopper_mxfp4_tactic(
            record.get("requested_tactic", effective), execution_mode=mode
        )
        candidate["requested_tactic"] = copy.deepcopy(requested)
        candidate["requested_tactic_aliases"] = [copy.deepcopy(requested)]
    return candidate


def hopper_mxfp4_candidate_records(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the deduplicated union with offline discovery provenance."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return [
        {
            "candidate": _manifest_candidate(profile, mode, record),
            "winner_for_tokens": list(record["winner_for_tokens"]),
        }
        for record in _PROFILE_CANDIDATES[profile][mode]
    ]


def hopper_mxfp4_candidates(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return fresh executable tactics from one routing-profile manifest."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return [
        _materialize_profile_tactic(
            record["tactic"], execution_mode=mode, routing_profile=profile
        )
        for record in _PROFILE_CANDIDATES[profile][mode]
    ]


def hopper_mxfp4_runtime_candidates(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return fresh tactics for live online/offline tuning.

    Fused follows the ordinary FP8 tuner model: every routing profile sees
    the same compact, deduplicated union of modern block-profile winners,
    frozen published-exact winners, and the two measured H20 anchors. The routing-specific
    default remains only an ordering hint and is never replaced without live
    timing. Split remains profile-specific because each tactic embeds its
    certified 132-SM partition.
    """

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    if mode == "split":
        return hopper_mxfp4_candidates(
            execution_mode=mode,
            routing_profile=profile,
        )
    return [
        validate_hopper_mxfp4_tactic(record["tactic"], execution_mode="fused")
        for record in _fused_runtime_candidate_records()
    ]


def hopper_mxfp4_cache_provenance_sha256(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> str:
    """Return the exact cache identity for the shipped tuning domain."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    if mode == "fused":
        candidate_union_sha256 = MXFP4_FUSED_RUNTIME_CANDIDATE_UNION_SHA256
    else:
        records: tuple[dict[str, Any], ...] = tuple(
            {
                "candidate_id": _candidate_id(_IMPLEMENTATION[mode], tactic),
                "tactic": tactic,
            }
            for tactic in hopper_mxfp4_runtime_candidates(
                execution_mode=mode,
                routing_profile=profile,
            )
        )
        candidate_union_sha256 = _candidate_union_sha256(
            tuple(sorted(records, key=lambda record: str(record["candidate_id"])))
        )

    canonical = json.dumps(
        {
            "execution_mode": mode,
            "routing_profile": profile,
            "runtime_candidate_union_sha256": candidate_union_sha256,
            "tuning_provenance": dict(
                hopper_mxfp4_tuning_provenance(
                    execution_mode=mode,
                    routing_profile=profile,
                )
            ),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def is_hopper_mxfp4_tactic_shape_compatible(
    tactic: Mapping[str, Any],
    *,
    execution_mode: str,
    hidden: int,
    intermediate: int,
) -> bool:
    """Whether a validated tactic's GEMM K axes divide the model shape."""

    mode = _mode(execution_mode)
    _positive_int(hidden, "hidden")
    _positive_int(intermediate, "intermediate")
    normalized = validate_hopper_mxfp4_tactic(tactic, execution_mode=mode)
    if mode == "fused":
        tile_k = int(normalized["mma_tiler_mnk"][2])
        return hidden % tile_k == 0 and intermediate % tile_k == 0
    k1_tile_k = int(normalized["k1_mma_tiler_mnk"][2])
    k2_tile_k = int(normalized["k2_mma_tiler_mnk"][2])
    return hidden % k1_tile_k == 0 and intermediate % k2_tile_k == 0


def hopper_mxfp4_candidates_for_shape(
    *,
    execution_mode: str,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the stable manifest union filtered to tactics legal for a shape."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = [
        tactic
        for tactic in hopper_mxfp4_candidates(
            execution_mode=mode, routing_profile=profile
        )
        if is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            execution_mode=mode,
            hidden=hidden,
            intermediate=intermediate,
        )
    ]
    if not legal:
        raise ValueError(
            f"no manifest-derived MXFP4 {mode} tactic supports "
            f"hidden={hidden}, intermediate={intermediate}"
        )
    return legal


def hopper_mxfp4_runtime_candidates_for_shape(
    *,
    execution_mode: str,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the live-tuning union filtered to tactics legal for a shape."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = [
        tactic
        for tactic in hopper_mxfp4_runtime_candidates(
            execution_mode=mode,
            routing_profile=profile,
        )
        if is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            execution_mode=mode,
            hidden=hidden,
            intermediate=intermediate,
        )
    ]
    if not legal:
        raise ValueError(
            f"no runtime MXFP4 {mode} tactic supports "
            f"hidden={hidden}, intermediate={intermediate}"
        )
    return legal


def _token_bucket(max_tokens: int) -> int:
    _positive_int(max_tokens, "max_tokens")
    for bucket in MXFP4_TUNING_TOKEN_BUCKETS:
        if max_tokens <= bucket:
            return bucket
    return MXFP4_TUNING_TOKEN_BUCKETS[-1]


def hopper_mxfp4_default_tactic(
    max_tokens: int,
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> dict[str, Any]:
    """Return the ceil-bucket offline winner, clamping above 2048 tokens."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    bucket = _token_bucket(max_tokens)
    winner_id = _PROFILE_WINNERS[profile][mode][bucket]["candidate_id"]
    record = _candidate_by_id(mode, profile)[winner_id]
    return _materialize_profile_tactic(
        record["tactic"], execution_mode=mode, routing_profile=profile
    )


def hopper_mxfp4_ordered_candidates(
    max_tokens: int,
    *,
    execution_mode: str,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Put a legal bucket winner first, otherwise keep stable union order."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = hopper_mxfp4_runtime_candidates_for_shape(
        execution_mode=mode,
        hidden=hidden,
        intermediate=intermediate,
        routing_profile=profile,
    )
    default = hopper_mxfp4_default_tactic(
        max_tokens, execution_mode=mode, routing_profile=profile
    )
    if default not in legal:
        return legal
    return [default, *(candidate for candidate in legal if candidate != default)]


def hopper_mxfp4_tuning_manifest(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> dict[str, Any]:
    """Return the compact runtime projection for one routing profile.

    Both profiles retain their external artifact hash and schema version. The
    published-exact candidate payload stays byte-for-byte frozen even though
    the refreshed block profile now stores complete modern fused tactics.
    """

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    provenance = _PROFILE_PROVENANCE[profile][mode]
    records = _candidate_by_id(mode, profile)
    per_token_winners: dict[str, dict[str, Any]] = {}
    for token in MXFP4_TUNING_TOKEN_BUCKETS:
        winner = _PROFILE_WINNERS[profile][mode][token]
        candidate_id = winner["candidate_id"]
        per_token_winners[str(token)] = {
            "candidate": _manifest_candidate(profile, mode, records[candidate_id]),
            "candidate_id": candidate_id,
            "median_score_us": winner["median_score_us"],
            "relative_spread": winner["relative_spread"],
            "scores_us": list(winner["scores_us"]),
            "telemetry_warning": winner["telemetry_warning"],
        }
    return {
        "artifact_manifest_sha256": provenance["artifact_manifest_sha256"],
        "candidate_union": hopper_mxfp4_candidate_records(
            execution_mode=mode, routing_profile=profile
        ),
        "external_schema_version": provenance["external_schema_version"],
        "implementation": _IMPLEMENTATION[mode],
        "per_token_winners": per_token_winners,
        "routing_profile": profile,
        "runtime_schema_version": 1,
    }


def _validate_embedded_tables() -> None:
    for profile in MXFP4_TUNING_ROUTING_PROFILES:
        for mode in ("fused", "split"):
            candidate_ids = []
            winning_tokens = []
            candidate_union_records = []
            for record in _PROFILE_CANDIDATES[profile][mode]:
                tactic = _validate_profile_manifest_tactic(
                    record["tactic"],
                    execution_mode=mode,
                    routing_profile=profile,
                )
                expected_id = _candidate_id(_IMPLEMENTATION[mode], tactic)
                if record["candidate_id"] != expected_id:
                    raise RuntimeError(
                        f"corrupt {profile}/{mode} MXFP4 candidate id "
                        f"{record['candidate_id']}"
                    )
                if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
                    validate_hopper_mxfp4_tactic(
                        record.get("requested_tactic", tactic), execution_mode=mode
                    )
                candidate_ids.append(record["candidate_id"])
                candidate_union_records.append(
                    {"candidate_id": record["candidate_id"], "tactic": tactic}
                )
                winning_tokens.extend(record["winner_for_tokens"])
            if candidate_ids != sorted(set(candidate_ids)):
                raise RuntimeError(
                    f"{profile}/{mode} MXFP4 candidate union is not sorted/unique"
                )
            if sorted(winning_tokens) != list(MXFP4_TUNING_TOKEN_BUCKETS):
                raise RuntimeError(
                    f"{profile}/{mode} MXFP4 winners do not cover every token bucket"
                )
            if set(_PROFILE_WINNERS[profile][mode]) != set(MXFP4_TUNING_TOKEN_BUCKETS):
                raise RuntimeError(
                    f"{profile}/{mode} MXFP4 heuristic table has wrong token buckets"
                )
            if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
                actual_union_sha256 = _candidate_union_sha256(
                    tuple(candidate_union_records)
                )
                expected_union_sha256 = _PROVENANCE[mode]["candidate_union_sha256"]
                if actual_union_sha256 != expected_union_sha256:
                    raise RuntimeError(
                        f"embedded {profile}/{mode} MXFP4 candidate union "
                        f"sha256={actual_union_sha256} does not match "
                        f"{expected_union_sha256}"
                    )
            manifest = hopper_mxfp4_tuning_manifest(
                execution_mode=mode, routing_profile=profile
            )
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
                expected_sha256 = _PROVENANCE[mode]["runtime_manifest_sha256"]
            else:
                expected_sha256 = _EXACT_RUNTIME_MANIFEST_SHA256[mode]
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"embedded {profile}/{mode} MXFP4 runtime manifest "
                    f"sha256={actual_sha256} does not match {expected_sha256}"
                )


_validate_embedded_tables()


__all__ = [
    "MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE",
    "MXFP4_FUSED_RUNTIME_ANCHOR_PROVENANCE",
    "MXFP4_FUSED_RUNTIME_CANDIDATE_UNION_SHA256",
    "MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE",
    "MXFP4_TUNING_PROVENANCE",
    "MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE",
    "MXFP4_TUNING_ROUTING_PROFILES",
    "MXFP4_TUNING_TOKEN_BUCKETS",
    "Mxfp4ExecutionMode",
    "Mxfp4RoutingProfile",
    "hopper_mxfp4_cache_provenance_sha256",
    "hopper_mxfp4_candidate_records",
    "hopper_mxfp4_candidates",
    "hopper_mxfp4_candidates_for_shape",
    "hopper_mxfp4_default_tactic",
    "hopper_mxfp4_ordered_candidates",
    "hopper_mxfp4_runtime_candidates",
    "hopper_mxfp4_runtime_candidates_for_shape",
    "hopper_mxfp4_tuning_manifest",
    "hopper_mxfp4_tuning_provenance",
    "is_hopper_mxfp4_tactic_shape_compatible",
    "is_valid_hopper_mxfp4_tactic",
    "normalize_hopper_mxfp4_routing_profile",
    "require_hopper_mxfp4_fused_tuning_device",
    "require_hopper_mxfp4_tuning_device",
    "validate_hopper_mxfp4_tactic",
]
