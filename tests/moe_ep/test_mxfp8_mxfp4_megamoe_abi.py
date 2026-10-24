# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side scheduler and workspace ABI tests for MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

import pytest

from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
    resolve_tactic,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.scheduler import (
    AtomicTileClaimer,
    PersistentFc1Fc2Scheduler,
    Phase,
    SchedulerConfig,
    StaticTileClaimer,
    WORK_RECORD_WORDS,
    WorkRecord,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.workspace import (
    WorkspaceConfig,
    make_workspace_plan,
    sf_uint32_per_token,
)


def test_mxfp8_mxfp4_token_back_tactic_contract():
    assert resolve_tactic(129, {"token_back_mode": "epi_warps"})["token_back_mode"] == (
        "epi_warps"
    )


@pytest.mark.parametrize("mode", ["reuse_dispatch_warps", "standalone_warps"])
def test_mxfp8_mxfp4_rejects_other_token_back_modes(mode):
    with pytest.raises(ValueError, match="token_back_mode must be epi_warps"):
        resolve_tactic(129, {"token_back_mode": mode})


def _scheduler() -> PersistentFc1Fc2Scheduler:
    return PersistentFc1Fc2Scheduler(
        SchedulerConfig(
            expert_token_counts=(0, 1, 129, 0),
            gate_up_size=256,
            hidden_size=128,
            cta_tile_m=64,
            cta_tile_n=128,
            cluster_shape_mn=(2, 1),
            group_hint=4,
            token_padding_block=64,
            sf_padding_block=128,
            num_persistent_clusters=3,
            fc2_ready_threshold=2,
        )
    )


def test_mxfp8_mxfp4_megamoe_work_record_wire_abi_roundtrips() -> None:
    scheduler = _scheduler()
    records = [
        scheduler.record_for(index) for index in range(scheduler.cluster_tile_count)
    ]
    assert records
    assert {record.expert_idx for record in records} == {1, 2}
    assert {record.phase for record in records} == {Phase.FC1, Phase.FC2}
    for record in records:
        assert len(record.to_words()) == WORK_RECORD_WORDS == 8
        assert WorkRecord.from_words(record.to_words()) == record
        assert record.cumulative_data_physical_row % 64 == 0
        assert record.cumulative_sf_physical_row % 128 == 0
        assert 0 < record.valid_tokens_in_cta_tile <= 64


def test_mxfp8_mxfp4_megamoe_static_and_atomic_claim_each_tile_once() -> None:
    scheduler = _scheduler()
    claimed: list[int] = []
    for cluster_id in range(3):
        claimer = StaticTileClaimer(cluster_id=cluster_id, cluster_count=3)
        while True:
            index = claimer.claim()
            if index >= scheduler.cluster_tile_count:
                break
            claimed.append(index)
    claimed.sort()
    assert claimed == list(range(scheduler.cluster_tile_count))

    atomic = AtomicTileClaimer()
    atomic_records = list(scheduler.records_for_claimer(atomic))
    assert len(atomic_records) == scheduler.cluster_tile_count
    assert atomic.value == scheduler.cluster_tile_count + 1  # final DONE claim


def test_mxfp8_mxfp4_megamoe_readiness_peek_uses_phase_specific_threshold() -> None:
    scheduler = _scheduler()
    fc1 = next(
        record
        for record in (
            scheduler.record_for(i) for i in range(scheduler.cluster_tile_count)
        )
        if record.phase is Phase.FC1
    )
    fc2 = next(
        record
        for record in (
            scheduler.record_for(i) for i in range(scheduler.cluster_tile_count)
        )
        if record.phase is Phase.FC2
    )
    size = max(fc1.counter_slot, fc2.counter_slot) + 1
    ready = [0] * size
    done = [0] * size
    ready[fc1.counter_slot] = fc1.valid_tokens_in_cta_tile
    done[fc2.counter_slot] = scheduler.config.fc2_ready_threshold
    assert scheduler.enrich_readiness_peek(fc1, fc1_ready_counters=ready).peek_ready
    assert scheduler.enrich_readiness_peek(fc2, fc1_done_counters=done).peek_ready


def test_mxfp8_mxfp4_megamoe_workspace_scale_sideband_and_alignment() -> None:
    assert sf_uint32_per_token(32) == 1
    assert sf_uint32_per_token(128) == 1
    assert sf_uint32_per_token(160) == 2
    config = WorkspaceConfig(
        world_size=1,
        num_topk=4,
        num_experts_per_rank=8,
        max_tokens_per_rank=129,
        hidden=128,
        intermediate=256,
        token_padding_block=64,
        sf_padding_block=128,
        cluster_tile_tokens=128,
        load_balance_mode="atomic_counter",
    )
    plan = make_workspace_plan(config)
    for layout in (plan.local, plan.shared):
        previous_end = 0
        for region in layout.regions:
            assert region.offset >= previous_end
            assert region.offset % region.spec.align == 0
            previous_end = region.end
        assert layout.total_bytes % 16 == 0
    assert plan.local.zero_prefix_bytes == plan.local.region("l1_token_buffer").offset
    assert (
        plan.shared.zero_prefix_bytes == plan.shared.region("src_token_topk_idx").offset
    )
    assert plan.local.region("l1_sf_buffer").spec.shape == (
        config.pool_sf_capacity * config.sf_uint32_per_token,
    )
    assert plan.local.region("fc1_output_sf").spec.shape[1] == 8
