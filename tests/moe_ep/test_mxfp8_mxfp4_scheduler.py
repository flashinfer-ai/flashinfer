# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for the MXFP8 x MXFP4 persistent scheduler contract."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import pytest


# Load the sibling by path under a collision-resistant module name.  A bare
# ``import scheduler`` can resolve TeKit's top-level scheduler.py when these
# tests run inside the development container.
_MODULE_NAME = "flashinfer_mxfp8_mxfp4_scheduler"
_SPEC = spec_from_file_location(
    _MODULE_NAME,
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/cute_dsl/megamoe/mxfp8_mxfp4/scheduler.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_SCHEDULER = module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _SCHEDULER
_SPEC.loader.exec_module(_SCHEDULER)

AtomicTileClaimer = _SCHEDULER.AtomicTileClaimer
PEEK_READY_BIT = _SCHEDULER.PEEK_READY_BIT
PersistentFc1Fc2Scheduler = _SCHEDULER.PersistentFc1Fc2Scheduler
Phase = _SCHEDULER.Phase
if TYPE_CHECKING:
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.scheduler import SchedulerConfig
else:
    SchedulerConfig = _SCHEDULER.SchedulerConfig
StaticTileClaimer = _SCHEDULER.StaticTileClaimer
WorkRecord = _SCHEDULER.WorkRecord


def _config(**overrides) -> SchedulerConfig:
    config = SchedulerConfig(
        expert_token_counts=(17, 0, 33),
        gate_up_size=256,
        hidden_size=128,
        cta_tile_m=16,
        cta_tile_n=64,
        cluster_shape_mn=(1, 2),
        group_hint=3,
        token_padding_block=64,
        sf_padding_block=128,
        num_persistent_clusters=3,
        fc2_ready_threshold=2,
    )
    return replace(config, **overrides)


def test_work_record_wire_order_and_packed_phase() -> None:
    words = (3, 4, 5, 64, 128, 7, 9, int(Phase.FC2) | PEEK_READY_BIT)
    record = WorkRecord.from_words(words)
    assert record.to_words() == words
    assert record.phase is Phase.FC2
    assert record.peek_ready
    assert record.counter_slot == 12
    assert not WorkRecord.done().is_valid


def test_group_phase_order_and_independent_physical_rows() -> None:
    scheduler = PersistentFc1Fc2Scheduler(_config())
    records = [scheduler.record_for(i) for i in range(scheduler.cluster_tile_count)]

    # group_hint=3: expert 0 contributes 4 FC1 cluster tiles, so it forms the
    # first group.  Its FC2 work must precede the next group's FC1 work.
    phases_experts = [(r.phase, r.expert_idx) for r in records]
    first_expert_2 = next(i for i, item in enumerate(phases_experts) if item[1] == 2)
    assert all(expert == 0 for _, expert in phases_experts[:first_expert_2])
    assert Phase.FC2 in [phase for phase, _ in phases_experts[:first_expert_2]]
    assert phases_experts[first_expert_2][0] is Phase.FC1

    expert_2 = records[first_expert_2]
    # Expert 1 has zero tokens but still consumes its configured padded rows
    # as zero; expert 0's 17 rows round differently in the two planes.
    assert expert_2.cumulative_data_physical_row == 64
    assert expert_2.cumulative_sf_physical_row == 128
    assert expert_2.cumulative_token_block_count == 2


def test_cta_decode_clips_tail_tokens_and_expands_feature_axis() -> None:
    scheduler = PersistentFc1Fc2Scheduler(
        _config(expert_token_counts=(17,), group_hint=1)
    )
    leader = scheduler.record_for(0, cta_coord_mn=(0, 0))
    feature_peer = scheduler.record_for(0, cta_coord_mn=(0, 1))
    assert leader.tile_m_idx == 0
    assert feature_peer.tile_m_idx == 1
    assert leader.tile_n_idx == feature_peer.tile_n_idx == 0
    assert leader.valid_tokens_in_cta_tile == 16

    tail = scheduler.record_for(2, cta_coord_mn=(0, 0))
    assert tail.phase is Phase.FC1
    assert tail.tile_n_idx == 1
    assert tail.valid_tokens_in_cta_tile == 1


@pytest.mark.parametrize("clusters", [3, 64])
def test_static_claimers_partition_work_without_overlap(clusters: int) -> None:
    scheduler = PersistentFc1Fc2Scheduler(_config(num_persistent_clusters=clusters))
    claimed: list[int] = []
    for cluster_id in range(scheduler.config.num_persistent_clusters):
        claimer = StaticTileClaimer(
            cluster_id=cluster_id,
            cluster_count=scheduler.config.num_persistent_clusters,
        )
        while True:
            index = claimer.claim()
            if index >= scheduler.cluster_tile_count:
                break
            claimed.append(index)
    assert sorted(claimed) == list(range(scheduler.cluster_tile_count))


def test_atomic_claims_are_dense_and_readiness_peeks_are_phase_specific() -> None:
    scheduler = PersistentFc1Fc2Scheduler(
        _config(expert_token_counts=(17,), group_hint=1)
    )
    claimer = AtomicTileClaimer()
    indices = [claimer.claim() for _ in range(scheduler.cluster_tile_count)]
    assert indices == list(range(scheduler.cluster_tile_count))

    fc1 = scheduler.record_for(0)
    fc1_counters = {fc1.counter_slot: fc1.valid_tokens_in_cta_tile}
    assert scheduler.enrich_readiness_peek(
        fc1, fc1_ready_counters=fc1_counters
    ).peek_ready
    assert not scheduler.enrich_readiness_peek(
        fc1, fc1_ready_counters={fc1.counter_slot: fc1.valid_tokens_in_cta_tile - 1}
    ).peek_ready

    fc2_index = next(
        i
        for i in range(scheduler.cluster_tile_count)
        if scheduler.record_for(i).phase is Phase.FC2
    )
    fc2 = scheduler.record_for(fc2_index)
    assert scheduler.enrich_readiness_peek(
        fc2, fc1_done_counters={fc2.counter_slot: 2}
    ).peek_ready
    assert not scheduler.enrich_readiness_peek(
        fc2, fc1_done_counters={fc2.counter_slot: 1}
    ).peek_ready


def test_zero_token_suffix_terminates_without_work() -> None:
    scheduler = PersistentFc1Fc2Scheduler(
        _config(expert_token_counts=(0, 0), group_hint=1)
    )
    assert scheduler.cluster_tile_count == 0
    assert not scheduler.record_for(0).is_valid


@pytest.mark.parametrize("hidden,intermediate", [(128, 128), (512, 256), (1024, 768)])
def test_lean_explicit_rows_with_empty_and_skewed_experts(hidden, intermediate):
    config = _config(
        expert_token_counts=(0, 257, 0, 3, 0),
        hidden_size=hidden,
        gate_up_size=2 * intermediate,
        expert_data_row_offsets=(0, 64, 384, 448, 512, 512),
        expert_scale_row_offsets=(0, 128, 512, 768, 896, 896),
        wait_for_dispatch=False,
    )
    scheduler = PersistentFc1Fc2Scheduler(config)
    records = list(scheduler.records_for_claimer(AtomicTileClaimer()))
    assert {r.expert_idx for r in records} == {1, 3}
    for record in records:
        assert (
            record.cumulative_data_physical_row
            == config.expert_data_row_offsets[record.expert_idx]
        )
        assert (
            record.cumulative_sf_physical_row
            == config.expert_scale_row_offsets[record.expert_idx]
        )
        assert record.cumulative_token_block_count == (
            0 if record.expert_idx == 1 else 17
        )
        if record.phase is Phase.FC1:
            assert scheduler.enrich_readiness_peek(record).peek_ready
        else:
            assert not scheduler.enrich_readiness_peek(record).peek_ready
            assert scheduler.enrich_readiness_peek(
                record,
                fc1_done_counters={record.counter_slot: config.fc2_ready_threshold},
            ).peek_ready
    # The K extents change phase feature counts, not physical token offsets.
    for expert_idx, token_count in ((1, 257), (3, 3)):
        token_blocks = (
            token_count + config.cluster_tile_m - 1
        ) // config.cluster_tile_m
        for phase, features in (
            (Phase.FC1, config.fc1_feature_blocks),
            (Phase.FC2, config.fc2_feature_blocks),
        ):
            assert (
                sum(r.expert_idx == expert_idx and r.phase == phase for r in records)
                == token_blocks * features
            )


def test_explicit_rows_match_mega_when_padding_matches():
    mega = PersistentFc1Fc2Scheduler(_config())
    lean = PersistentFc1Fc2Scheduler(
        _config(
            expert_data_row_offsets=(0, 64, 64, 128),
            expert_scale_row_offsets=(0, 128, 128, 256),
            wait_for_dispatch=False,
        )
    )
    assert list(mega.records_for_claimer(AtomicTileClaimer())) == list(
        lean.records_for_claimer(AtomicTileClaimer())
    )


@pytest.mark.parametrize(
    "data,scales",
    [
        ((0, 64, 64, 128), None),
        ((0, 64), (0, 128)),
        ((64, 128, 128, 192), (0, 128, 128, 256)),
        ((0, 32, 32, 128), (0, 128, 128, 256)),
        ((0, 64, 0, 128), (0, 128, 128, 256)),
        ((0, 0, 0, 64), (0, 128, 128, 256)),
    ],
)
def test_explicit_row_validation(data, scales):
    with pytest.raises(ValueError):
        _config(expert_data_row_offsets=data, expert_scale_row_offsets=scales)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("features", [(52, 16), (1, 8), (8, 1), (1, 1)])
@pytest.mark.parametrize("counts", [(0, 0), (1,), (0, 17, 0, 257, 1, 0)])
def test_alternating_random_cluster_durations(seed, features, counts):
    import random

    rng = random.Random(seed)
    clusters = rng.choice([1, 2, 3, 8, 74])
    config = _config(
        expert_token_counts=counts,
        gate_up_size=128 * features[0],
        hidden_size=128 * features[1],
        num_persistent_clusters=clusters,
    )
    scheduler = PersistentFc1Fc2Scheduler(config)
    fc1, fc2 = AtomicTileClaimer(), AtomicTileClaimer()
    claimers = [
        _SCHEDULER.AlternatingTileClaimer(config, fc1, fc2) for _ in range(clusters)
    ]
    active = set(range(clusters))
    ready_at = [0.0] * clusters
    seen = set()
    completed_fc1 = set()
    inflight = {}
    now = 0.0
    iterations = 0
    while active:
        runnable = [c for c in active if ready_at[c] <= now]
        rng.shuffle(runnable)
        progressed = False
        for c in runnable:
            if c in inflight:
                record = inflight[c]
                key = (record.counter_slot, record.tile_m_idx)
                if record.phase == Phase.FC1:
                    completed_fc1.add(key)
                else:
                    required = {
                        (record.counter_slot, f * 2) for f in range(features[0])
                    }
                    if not required <= completed_fc1:
                        continue
                del inflight[c]
                progressed = True
            claim = claimers[c].claim()
            if claim is None:
                continue
            progressed = True
            record = scheduler.record_for_alternating(claim)
            if not record.is_valid:
                active.remove(c)
                continue
            key = record.to_words()
            assert key not in seen
            seen.add(key)
            if record.phase == Phase.FC2:
                assert fc1.value >= (record.counter_slot + 1) * features[0]
            # FC1 durations are deliberately skewed across clusters.
            ready_at[c] = now + rng.uniform(0.1, 100 if c % 3 == 0 else 3)
            inflight[c] = record
        future = [ready_at[c] for c in active if ready_at[c] > now]
        if future:
            now = min(future)
        else:
            assert progressed or not active, "all clusters deadlocked"
        iterations += 1
        assert iterations < 100000
    expected = {
        scheduler.record_for(i).to_words() for i in range(scheduler.cluster_tile_count)
    }
    assert seen == expected


def test_alternating_fixed_claim_decode_ignores_groups():
    scheduler = PersistentFc1Fc2Scheduler(_config())
    # A signed FC2 index is phase-local even though the static order has
    # multiple groups. Decode every CTA, including the clipped token tail.
    for phase in [Phase.FC1, Phase.FC2]:
        expected = [
            scheduler.record_for(i)
            for i in range(scheduler.cluster_tile_count)
            if scheduler.record_for(i).phase == phase
        ]
        for index, record in enumerate(expected):
            claim = index if phase == Phase.FC1 else -index - 1
            assert scheduler.record_for_alternating(claim) == record
            peer = scheduler.record_for_alternating(claim, cta_coord_mn=(0, 1))
            assert peer.tile_m_idx == record.tile_m_idx + 1
            assert peer.counter_slot == record.counter_slot


@pytest.mark.parametrize("experts", [1, 7, 32, 33, 64, 65])
@pytest.mark.parametrize("group_hint", [1, 5, 4096])
@pytest.mark.parametrize("explicit_offsets", [False, True])
def test_longest_first_preserves_tiles_and_pool_addresses(
    experts, group_hint, explicit_offsets
):
    import random

    rng = random.Random(20261018 + experts)
    counts = tuple(rng.choice([0, 0, 17, 33, 65, 257]) for _ in range(experts))
    offsets = {}
    if explicit_offsets:
        offsets = dict(
            expert_data_row_offsets=tuple(320 * e for e in range(experts + 1)),
            expert_scale_row_offsets=tuple(384 * e for e in range(experts + 1)),
        )
    config = _config(expert_token_counts=counts, group_hint=group_hint, **offsets)
    original = PersistentFc1Fc2Scheduler(config)
    reordered = PersistentFc1Fc2Scheduler(replace(config, claim_order="longest_first"))
    assert reordered.order == tuple(
        sorted(range(experts), key=lambda e: (-counts[e], e))
    )
    for cta in [(0, 0), (0, 1)]:
        before = [
            original.record_for(i, cta_coord_mn=cta).to_words()
            for i in range(original.cluster_tile_count)
        ]
        after = [
            reordered.record_for(i, cta_coord_mn=cta).to_words()
            for i in range(reordered.cluster_tile_count)
        ]
        # Full records include the counter prefix, both physical row offsets,
        # valid-token mask and phase as well as the tile coordinates.
        assert sorted(before) == sorted(after)

    def warmup(cfg):
        return _SCHEDULER.AlternatingTileClaimer(
            cfg, AtomicTileClaimer(), AtomicTileClaimer()
        ).pending

    assert warmup(config) == warmup(reordered.config)
    for phase in (Phase.FC1, Phase.FC2):
        records = [w for w in reordered._cluster_work if w.phase == phase]
        claims = [i if phase == Phase.FC1 else -i - 1 for i in range(len(records))]
        assert [reordered.record_for_alternating(i).expert_idx for i in claims] == [
            w.expert_idx for w in records
        ]


def test_longest_first_ties_zeros_and_invalid_order():
    assert _SCHEDULER.longest_first_order((0, 9, 9, 0, 4)) == (1, 2, 4, 0, 3)
    scheduler = PersistentFc1Fc2Scheduler(
        _config(expert_token_counts=(0,) * 33, claim_order="longest_first")
    )
    assert scheduler.cluster_tile_count == 0
    assert not scheduler.record_for(0).is_valid
    with pytest.raises(ValueError, match="claim_order"):
        _config(claim_order="random")
