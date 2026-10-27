# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVFP4 candidate seeds and default capacity buckets.

The defaults were measured on H200 with the reference model below. They are
heuristics on other SM90 devices and unmeasured capacities, not performance
claims. Explicit caller settings take precedence.
"""


def _tactic(**overrides):
    return dict(
        dict(
            mma_tiler_mnk=(256, 16, 256),
            cluster_shape_mnk=(2, 1, 1),
            group_hint=512,
            num_sched_stages=1,
            pingpong=False,
            swap_ab=True,
            fp8_accum_mode="1xacc",
            load_balance_mode="atomic_counter",
            token_back_mode="epi_warps",
            in_kernel_fc2_reduce=False,
            combine_format="bf16",
            grouped_token_back=False,
            dedup_dispatch=False,
            active_dispatch_warps=4,
            fc1_store_offload=False,
            fc1_early_done_publish=False,
            fold_producer_warps=False,
            fc2_tail_n8=False,
            fc1_ready_mode="tile",
            tail_split_pairs=False,
            compact_pull_buffer=False,
            nvfp4_lut_replicas=8,
            nvfp4_lut_map="lane",
        ),
        **overrides,
    )


NVFP4_SF0_CANDIDATE_SEEDS = (
    _tactic(
        group_hint=132,
        load_balance_mode="static",
        fc1_store_offload=True,
        nvfp4_lut_replicas=32,
    ),
    _tactic(nvfp4_lut_replicas=32),
    _tactic(
        group_hint=132,
        load_balance_mode="static",
        active_dispatch_warps=1,
        dedup_dispatch=True,
        fc1_store_offload=True,
        num_sched_stages=2,
        compact_pull_buffer=True,
    ),
    _tactic(cluster_shape_mnk=(1, 1, 1), group_hint=396, fc1_store_offload=True),
    _tactic(
        cluster_shape_mnk=(1, 1, 1),
        group_hint=330,
        active_dispatch_warps=1,
        dedup_dispatch=True,
        fc1_store_offload=True,
        compact_pull_buffer=True,
    ),
    _tactic(
        mma_tiler_mnk=(256, 32, 256),
        cluster_shape_mnk=(1, 1, 1),
        group_hint=330,
        active_dispatch_warps=2,
        fc1_store_offload=True,
        num_sched_stages=2,
        compact_pull_buffer=True,
    ),
    _tactic(
        mma_tiler_mnk=(256, 64, 256),
        active_dispatch_warps=1,
        fc1_early_done_publish=True,
        fold_producer_warps=True,
        fc1_ready_mode="k256",
    ),
    _tactic(
        mma_tiler_mnk=(256, 64, 256),
        group_hint=528,
        active_dispatch_warps=1,
        fc1_early_done_publish=True,
        fold_producer_warps=True,
        fc2_tail_n8=True,
        nvfp4_lut_replicas=16,
    ),
    *(
        _tactic(
            mma_tiler_mnk=(256, 64, 256),
            cluster_shape_mnk=(1, 2, 1),
            group_hint=group,
            num_sched_stages=stages,
            active_dispatch_warps=1,
            fc1_early_done_publish=True,
            fold_producer_warps=True,
            fc2_tail_n8=tail,
            tail_split_pairs=True,
            compact_pull_buffer=True,
            nvfp4_lut_replicas=32,
        )
        for group, stages, tail in (
            (528, 3, True),
            (512, 1, True),
            (256, 1, True),
            (512, 1, False),
        )
    ),
)

_H200_REFERENCE = dict(
    hidden=7168,
    intermediate=3072,
    num_experts=384,
    world_size=4,
    num_topk=6,
    routing_profile="block_permutation_v1",
    kperm=64,
)

# Capacity: (seed index, FC1 overlap default).
_DEFAULTS = {
    8: (0, False),
    16: (1, True),
    32: (2, True),
    64: (3, False),
    128: (4, True),
    256: (5, False),
    512: (6, True),
    1024: (7, True),
    2048: (8, True),
    4096: (9, True),
    8192: (10, True),
    16384: (9, True),
    32768: (11, True),
}

NVFP4_DEFAULT_TACTICS_BY_TOKENS = {
    tokens: dict(
        requested_tactic=dict(NVFP4_SF0_CANDIDATE_SEEDS[seed]),
        requested_config=dict(kperm=64, nvfp4_fc1_k256_overlap=overlap),
    )
    for tokens, (seed, overlap) in _DEFAULTS.items()
}
