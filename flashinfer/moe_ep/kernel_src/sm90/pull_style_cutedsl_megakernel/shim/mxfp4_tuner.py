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
device.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal, cast

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)


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


_IMPLEMENTATION = "mxfp4_fused"

# The block-profile artifact hash binds the reviewed combined `discovery_winners_and_unions.json`
# formal aggregate. Candidate, policy, source, and runtime-manifest hashes below are
# regenerated from that exact byte sequence; the exact-balanced/H20 provenance stays separate.
_PROVENANCE = {
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
}

MXFP4_TUNING_PROVENANCE = MappingProxyType(_PROVENANCE)

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


# Candidate order is the lexicographically sorted candidate_id order from each
# formal manifest.  Keeping source and parent ids makes every online candidate
# traceable to its offline discovery stage.
_CANDIDATES = (
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
)

_WINNERS = {
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
}

_EXACT_PROVENANCE = {
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
}

_EXACT_CANDIDATES = (
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
)

_EXACT_WINNERS = {
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
        profile: MappingProxyType(values)
        for profile, values in _PROFILE_PROVENANCE.items()
    }
)

# Filled from the canonical JSON serialization of the compact runtime
# projections returned by ``hopper_mxfp4_tuning_manifest``.  These hashes are
# independent of the larger external artifact hashes above.
_EXACT_RUNTIME_MANIFEST_SHA256 = _EXACT_PROVENANCE["runtime_manifest_sha256"]


def normalize_hopper_mxfp4_routing_profile(
    routing_profile: str,
) -> Mxfp4RoutingProfile:
    """Compatibility wrapper around the canonical SM90 profile normalizer."""

    return normalize_sm90_routing_profile(routing_profile)  # type: ignore[return-value]


def hopper_mxfp4_tuning_provenance(
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> Mapping[str, Any]:
    """Return immutable provenance for one execution/routing domain."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE[profile]


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

    _require_exact_fields(tactic, _FROZEN_FUSED_FIELDS, _IMPLEMENTATION)
    return {**dict(tactic), **dict(_FROZEN_FUSED_LAYOUT)}


def validate_hopper_mxfp4_tactic(tactic: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one tactic from the bounded Hopper search domain.

    The returned mapping is a fresh copy whose tile and cluster fields are
    tuples. No defaults are inserted: callers must preserve the complete
    current fused tactic identity. Frozen pre-layout manifest rows
    are normalized only by private provenance helpers below.
    """

    if not isinstance(tactic, Mapping):
        raise TypeError(f"tactic must be a mapping, got {type(tactic).__name__}")

    if any(
        name in tactic for name in ("fc2_tail_n8", "fc1_ready_mode", "tail_split_pairs")
    ):
        from .mxfp4_optimization import normalize_mxfp4_optimization_tactic

        return normalize_mxfp4_optimization_tactic(tactic)
    _require_exact_fields(tactic, _FUSED_FIELDS, _IMPLEMENTATION)
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
            "MXFP4 fused production BF16-combine tactics fix grouped_token_back=false"
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


def _validate_frozen_hopper_mxfp4_tactic(tactic: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable historical manifest tactic without rewriting it."""

    complete = validate_hopper_mxfp4_tactic(_materialize_frozen_fused_tactic(tactic))
    return {name: complete[name] for name in _FROZEN_FUSED_FIELDS}


def _validate_profile_manifest_tactic(
    tactic: Mapping[str, Any],
    *,
    routing_profile: Mxfp4RoutingProfile,
) -> dict[str, Any]:
    """Validate the tactic identity stored by one profile manifest.

    Only the published-exact fused manifest predates the PR4688 layout fields.
    Its ten-field identity must stay byte-for-byte frozen; the refreshed block
    manifest stores complete modern tactics.
    """

    if routing_profile == MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE:
        return _validate_frozen_hopper_mxfp4_tactic(tactic)
    return validate_hopper_mxfp4_tactic(tactic)


def _materialize_profile_tactic(
    tactic: Mapping[str, Any],
    *,
    routing_profile: Mxfp4RoutingProfile,
) -> dict[str, Any]:
    """Return the complete tactic executed by a profile candidate."""

    stored = _validate_profile_manifest_tactic(
        tactic,
        routing_profile=routing_profile,
    )
    if routing_profile == MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE:
        return validate_hopper_mxfp4_tactic(_materialize_frozen_fused_tactic(stored))
    return stored


def is_valid_hopper_mxfp4_tactic(tactic: Mapping[str, Any]) -> bool:
    """Return whether ``tactic`` is a complete legal fused tactic."""

    try:
        validate_hopper_mxfp4_tactic(tactic)
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
        (_CANDIDATES, False),
        (_EXACT_CANDIDATES, True),
        (_FUSED_RUNTIME_ANCHORS, True),
    )
    for records, is_frozen in sources:
        for record in records:
            if is_frozen:
                source_tactic = _validate_frozen_hopper_mxfp4_tactic(
                    cast(Mapping[str, object], record["tactic"])
                )
            else:
                source_tactic = validate_hopper_mxfp4_tactic(
                    cast(Mapping[str, object], record["tactic"])
                )
            source_candidate_id = str(record["candidate_id"])
            expected_source_id = _candidate_id(_IMPLEMENTATION, source_tactic)
            if source_candidate_id != expected_source_id:
                raise RuntimeError(
                    f"corrupt fused candidate id {source_candidate_id}; "
                    f"expected {expected_source_id}"
                )
            if is_frozen:
                tactic = validate_hopper_mxfp4_tactic(
                    _materialize_frozen_fused_tactic(source_tactic),
                )
            else:
                tactic = source_tactic
            candidate_id = _candidate_id(_IMPLEMENTATION, tactic)
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


def _candidate_by_id(routing_profile: Mxfp4RoutingProfile) -> dict[str, dict[str, Any]]:
    return {
        cast(str, record["candidate_id"]): record
        for record in _PROFILE_CANDIDATES[routing_profile]
    }


def _manifest_candidate(
    profile: Mxfp4RoutingProfile,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    effective = _validate_profile_manifest_tactic(
        record["tactic"], routing_profile=profile
    )
    candidate = {
        "candidate_id": record["candidate_id"],
        "effective_tactic": copy.deepcopy(effective),
        "implementation": _IMPLEMENTATION,
        "parent_candidate_ids": list(record["parent_candidate_ids"]),
        "requested_tactic": copy.deepcopy(effective),
        "source": record["source"],
    }
    if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
        requested = validate_hopper_mxfp4_tactic(
            record.get("requested_tactic", effective)
        )
        candidate["requested_tactic"] = copy.deepcopy(requested)
        candidate["requested_tactic_aliases"] = [copy.deepcopy(requested)]
    return candidate


def hopper_mxfp4_candidate_records(
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the deduplicated union with offline discovery provenance."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return [
        {
            "candidate": _manifest_candidate(profile, record),
            "winner_for_tokens": list(record["winner_for_tokens"]),
        }
        for record in _PROFILE_CANDIDATES[profile]
    ]


def hopper_mxfp4_candidates(
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return fresh executable tactics from one routing-profile manifest."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return [
        _materialize_profile_tactic(
            cast(Mapping[str, object], record["tactic"]), routing_profile=profile
        )
        for record in _PROFILE_CANDIDATES[profile]
    ]


def hopper_mxfp4_runtime_candidates(
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return fresh tactics for live online/offline tuning.

    Fused follows the ordinary FP8 tuner model: every routing profile sees
    the same compact, deduplicated union of modern block-profile winners,
    frozen published-exact winners, and the two measured H20 anchors. The routing-specific
    default remains only an ordering hint and is never replaced without live
    timing.
    """

    normalize_hopper_mxfp4_routing_profile(routing_profile)
    return [
        validate_hopper_mxfp4_tactic(record["tactic"])
        for record in _fused_runtime_candidate_records()
    ]


def hopper_mxfp4_cache_provenance_sha256(
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> str:
    """Return the exact cache identity for the shipped tuning domain."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    candidate_union_sha256 = MXFP4_FUSED_RUNTIME_CANDIDATE_UNION_SHA256

    payload = {
        "execution_mode": "fused",
        "routing_profile": profile,
        "runtime_candidate_union_sha256": candidate_union_sha256,
        "tuning_provenance": dict(
            hopper_mxfp4_tuning_provenance(
                routing_profile=profile,
            )
        ),
    }
    from ..src.moe_hopper_fp8.mxfp4_policy import MXFP4_OPTIMIZATION_VERSION
    from .mxfp4_optimization import mxfp4_tail_candidate_provenance

    # Shape and world size are already separate knob-cache key fields.
    # This domain version binds eligibility/code changes without rewriting
    # frozen manifests.
    payload["optimization_domain"] = {
        "implementation": MXFP4_OPTIMIZATION_VERSION,
        "fc2_tail_n8": [False, True],
        "fc1_ready_mode": ["tile", "k256"],
        "tail_split_pairs": [False, True],
        "tail_candidate_extension": mxfp4_tail_candidate_provenance(),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def is_hopper_mxfp4_tactic_shape_compatible(
    tactic: Mapping[str, Any],
    *,
    hidden: int,
    intermediate: int,
) -> bool:
    """Whether a validated tactic's GEMM K axes divide the model shape."""

    _positive_int(hidden, "hidden")
    _positive_int(intermediate, "intermediate")
    normalized = validate_hopper_mxfp4_tactic(tactic)
    tile_k = int(normalized["mma_tiler_mnk"][2])
    return hidden % tile_k == 0 and intermediate % tile_k == 0


def hopper_mxfp4_candidates_for_shape(
    *,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the stable manifest union filtered to tactics legal for a shape."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = [
        tactic
        for tactic in hopper_mxfp4_candidates(routing_profile=profile)
        if is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            hidden=hidden,
            intermediate=intermediate,
        )
    ]
    if not legal:
        raise ValueError(
            f"no manifest-derived MXFP4 fused tactic supports "
            f"hidden={hidden}, intermediate={intermediate}"
        )
    return legal


def hopper_mxfp4_runtime_candidates_for_shape(
    *,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the live-tuning union filtered to tactics legal for a shape."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = [
        tactic
        for tactic in hopper_mxfp4_runtime_candidates(
            routing_profile=profile,
        )
        if is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            hidden=hidden,
            intermediate=intermediate,
        )
    ]
    if not legal:
        raise ValueError(
            f"no runtime MXFP4 fused tactic supports "
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
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> dict[str, Any]:
    """Return the ceil-bucket offline winner, clamping above 2048 tokens."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    bucket = _token_bucket(max_tokens)
    winner_id = cast(str, _PROFILE_WINNERS[profile][bucket]["candidate_id"])
    record = _candidate_by_id(profile)[winner_id]
    return _materialize_profile_tactic(record["tactic"], routing_profile=profile)


def hopper_mxfp4_ordered_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Put a legal bucket winner first, otherwise keep stable union order."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = hopper_mxfp4_runtime_candidates_for_shape(
        hidden=hidden,
        intermediate=intermediate,
        routing_profile=profile,
    )
    default = hopper_mxfp4_default_tactic(max_tokens, routing_profile=profile)
    if default not in legal:
        return legal
    return [default, *(candidate for candidate in legal if candidate != default)]


def hopper_mxfp4_tuning_manifest(
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> dict[str, Any]:
    """Return the compact runtime projection for one routing profile.

    Both profiles retain their external artifact hash and schema version. The
    published-exact candidate payload stays byte-for-byte frozen even though
    the refreshed block profile now stores complete modern fused tactics.
    """

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    provenance = _PROFILE_PROVENANCE[profile]
    records = _candidate_by_id(profile)
    per_token_winners: dict[str, dict[str, Any]] = {}
    for token in MXFP4_TUNING_TOKEN_BUCKETS:
        winner = _PROFILE_WINNERS[profile][token]
        candidate_id = cast(str, winner["candidate_id"])
        per_token_winners[str(token)] = {
            "candidate": _manifest_candidate(profile, records[candidate_id]),
            "candidate_id": candidate_id,
            "median_score_us": winner["median_score_us"],
            "relative_spread": winner["relative_spread"],
            "scores_us": list(cast(tuple[float, ...], winner["scores_us"])),
            "telemetry_warning": winner["telemetry_warning"],
        }
    return {
        "artifact_manifest_sha256": provenance["artifact_manifest_sha256"],
        "candidate_union": hopper_mxfp4_candidate_records(routing_profile=profile),
        "external_schema_version": provenance["external_schema_version"],
        "implementation": _IMPLEMENTATION,
        "per_token_winners": per_token_winners,
        "routing_profile": profile,
        "runtime_schema_version": 1,
    }


def _validate_embedded_tables() -> None:
    for profile in MXFP4_TUNING_ROUTING_PROFILES:
        candidate_ids: list[str] = []
        winning_tokens: list[int] = []
        candidate_union_records = []
        for record in _PROFILE_CANDIDATES[profile]:
            tactic = _validate_profile_manifest_tactic(
                cast(Mapping[str, object], record["tactic"]),
                routing_profile=profile,
            )
            expected_id = _candidate_id(_IMPLEMENTATION, tactic)
            if record["candidate_id"] != expected_id:
                raise RuntimeError(
                    f"corrupt {profile}/fused MXFP4 candidate id "
                    f"{record['candidate_id']}"
                )
            if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
                validate_hopper_mxfp4_tactic(
                    cast(Mapping[str, object], record.get("requested_tactic", tactic))
                )
            candidate_ids.append(cast(str, record["candidate_id"]))
            candidate_union_records.append(
                {"candidate_id": record["candidate_id"], "tactic": tactic}
            )
            winning_tokens.extend(cast(tuple[int, ...], record["winner_for_tokens"]))
        if candidate_ids != sorted(set(candidate_ids)):
            raise RuntimeError(
                f"{profile}/fused MXFP4 candidate union is not sorted/unique"
            )
        if sorted(winning_tokens) != list(MXFP4_TUNING_TOKEN_BUCKETS):
            raise RuntimeError(
                f"{profile}/fused MXFP4 winners do not cover every token bucket"
            )
        if set(_PROFILE_WINNERS[profile]) != set(MXFP4_TUNING_TOKEN_BUCKETS):
            raise RuntimeError(
                f"{profile}/fused MXFP4 heuristic table has wrong token buckets"
            )
        if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
            actual_union_sha256 = _candidate_union_sha256(
                tuple(candidate_union_records)
            )
            expected_union_sha256 = _PROVENANCE["candidate_union_sha256"]
            if actual_union_sha256 != expected_union_sha256:
                raise RuntimeError(
                    f"embedded {profile}/fused MXFP4 candidate union "
                    f"sha256={actual_union_sha256} does not match "
                    f"{expected_union_sha256}"
                )
        manifest = hopper_mxfp4_tuning_manifest(routing_profile=profile)
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
            expected_sha256 = _PROVENANCE["runtime_manifest_sha256"]
        else:
            expected_sha256 = _EXACT_RUNTIME_MANIFEST_SHA256
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"embedded {profile}/fused MXFP4 runtime manifest "
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
    "validate_hopper_mxfp4_tactic",
]
