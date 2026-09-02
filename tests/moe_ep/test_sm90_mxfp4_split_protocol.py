# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent synchronization models and shipped-source audits for split MoE."""

from __future__ import annotations

from collections import Counter
import inspect
import math
from pathlib import Path

import pytest

import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as sm90_mega
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    SplitEpochResetBarrier,
    SplitK2GlobalJoin,
)


@pytest.mark.parametrize("world_size,rank", [(1, 0), (4, 0), (4, 3), (32, 31)])
def test_epoch_reset_and_global_join_accept_one_warp_topologies(
    world_size: int, rank: int
) -> None:
    reset = SplitEpochResetBarrier(world_size, rank)
    join = SplitK2GlobalJoin(world_size, rank)
    assert (reset.world_size, reset.local_rank, reset._threads) == (
        world_size,
        rank,
        32,
    )
    assert (join.world_size, join.local_rank, join._threads) == (
        world_size,
        rank,
        32,
    )


@pytest.mark.parametrize("world_size", [-1, 0, 33, 1.5, None])
def test_epoch_reset_and_join_reject_invalid_world_size(world_size) -> None:
    for cls in (SplitEpochResetBarrier, SplitK2GlobalJoin):
        with pytest.raises(ValueError, match="world_size"):
            cls(world_size, 0)


@pytest.mark.parametrize("rank", [-1, 4, 4.5, None])
def test_epoch_reset_and_join_reject_invalid_rank(rank) -> None:
    for cls in (SplitEpochResetBarrier, SplitK2GlobalJoin):
        with pytest.raises(ValueError, match="local_rank"):
            cls(4, rank)


@pytest.mark.parametrize("world_size", [1, 4, 32])
def test_four_stage_epoch_reset_model_returns_every_signal_to_zero(
    world_size: int,
) -> None:
    peer_signals = [[0, 0] for _ in range(world_size)]
    expected_stages = (
        (0, +1, world_size),
        (1, +1, world_size),
        (0, -1, 0),
        (1, -1, 0),
    )

    # Three complete cycles catch wraparound/stale-epoch bugs.
    for epoch in range(12):
        signal_phase, signal_delta, target = expected_stages[epoch & 3]
        for _source_rank in range(world_size):
            for destination_rank in range(world_size):
                peer_signals[destination_rank][signal_phase] += signal_delta
        assert all(signal[signal_phase] == target for signal in peer_signals)

    assert peer_signals == [[0, 0] for _ in range(world_size)]


def _canonical_slot(
    physical_row: int,
    tile_idx: int,
    token_n: int,
    handoff_n: int,
) -> int:
    return physical_row // handoff_n + tile_idx * token_n // handoff_n


def _batched_counter_values(slots: list[int], batch: int) -> Counter[int]:
    counters: Counter[int] = Counter()
    pending: list[int] = []
    for slot in slots:
        pending.append(slot)
        if len(pending) == batch:
            counters.update(pending)
            pending.clear()
    counters.update(pending)
    return counters


@pytest.mark.parametrize(
    "fc1_n,fc2_n", [(32, 64), (64, 32), (32, 128), (128, 32), (64, 128), (128, 64)]
)
@pytest.mark.parametrize("epi_flag_batch", [1, 2])
def test_per_tile_counter_threshold_covers_independent_n_and_tails(
    fc1_n: int,
    fc2_n: int,
    epi_flag_batch: int,
) -> None:
    handoff_n = max(fc1_n, fc2_n)
    tail_cases = sorted(
        {
            1,
            min(fc1_n, fc2_n) - 1,
            min(fc1_n, fc2_n),
            min(fc1_n, fc2_n) + 1,
            handoff_n - 1,
            handoff_n,
            handoff_n + 1,
            2 * handoff_n - 1,
            2 * handoff_n,
            2 * handoff_n + 1,
        }
    )
    physical_row = 3 * handoff_n

    for valid_tokens in tail_cases:
        published = [
            _canonical_slot(physical_row, tile, fc1_n, handoff_n)
            for tile in range(math.ceil(valid_tokens / fc1_n))
        ]
        counters = _batched_counter_values(published, epi_flag_batch)
        for tile in range(math.ceil(valid_tokens / fc2_n)):
            valid_in_tile = min(fc2_n, valid_tokens - tile * fc2_n)
            slot = _canonical_slot(physical_row, tile, fc2_n, handoff_n)
            required_fc1_tiles = math.ceil(valid_in_tile / fc1_n)
            assert counters[slot] >= required_fc1_tiles


def _vendor_source(name: str) -> str:
    package_root = Path(sm90_mega.__file__).resolve().parent
    path = package_root / "src" / name
    assert path.is_file(), f"missing vendored split source: {path}"
    return path.read_text(encoding="utf-8")


def _method_block(source: str, signature: str, next_signature: str) -> str:
    begin = source.index(signature)
    end = source.find(next_signature, begin + len(signature))
    if end == -1:
        end = len(source)
    return source[begin:end]


def test_dispatch_ready_is_released_after_global_counts_before_payload_pull() -> None:
    source = _vendor_source("src/token_comm.py")
    body = _method_block(
        source,
        "    def dispatch_warp_body(",
        "    def ",
    )
    barrier = body.index("self.dispatch_barrier(")
    release = body.index("red_add_release_sys_s32_raw(")
    pull = body.index("self.dispatch_pull(")
    assert barrier < release < pull
    assert body.count("red_add_release_sys_s32_raw(") == 1
    assert "cta_linear_id == Int32(0)" in body
    assert "local_warp_idx == Int32(0)" in body
    assert "lane_idx == Int32(0)" in body


def test_k2_acquires_dispatch_ready_before_scheduler_publication() -> None:
    source = _vendor_source("moe_hopper_fp8/megamoe_kernel_fp8.py")
    wait = _method_block(
        source,
        "    def token_comm_hook_sched_warp_pre_init_wait(",
        "    def ",
    )
    assert wait.index("spin_wait(") < wait.index('sem="acquire"')
    assert wait.index('sem="acquire"') < wait.index("fence_acq_rel_sys()")

    body_source = _vendor_source("moe_hopper_fp8/kernel_fp8_glu_fc12_swapab.py")
    body = _method_block(
        body_source,
        "    def _kernel_body(",
        "    def ",
    )
    scheduler = body[body.index("token_comm_hook_sched_warp_pre_init_wait") :]
    assert scheduler.index(
        "token_comm_hook_sched_warp_pre_init_wait"
    ) < scheduler.index("scheduler.internal_init(")
    assert scheduler.index("scheduler.internal_init(") < scheduler.index(
        "scheduler.gen_next_work()"
    )
    assert scheduler.index("scheduler.gen_next_work()") < scheduler.index(
        "scheduler.publish_work()"
    )


def test_each_k2_handoff_tile_acquires_fc1_done_before_reading_payload_and_scale() -> (
    None
):
    source = _vendor_source("moe_hopper_fp8/kernel_fp8_glu_fc12_swapab.py")
    body = _method_block(
        source,
        "    def _issue_fc2_handoff_tma_tile(",
        "    def ",
    )
    spin = body.index("spin_wait(")
    acquire = body.index('sem="acquire"', spin)
    async_fence = body.index('fence_proxy("async")', acquire)
    global_fence = body.index('fence_proxy("async.global")', async_fence)
    payload = body.index('ext.get_gmem_tensor(\n            "b"', global_fence)
    scale = body.index("_tma_load_b_with_activation_sf_task_tile(", payload)
    assert spin < acquire < async_fence < global_fence < payload < scale
    assert "valid_tokens_in_cta_tile" in body
    assert "split_fc1_token_n" in body
    assert "wait_threshold =" in body


def test_global_join_uses_system_release_then_acquire_before_k3_visibility() -> None:
    source = inspect.getsource(SplitK2GlobalJoin)
    release = source.index("red_add_release_sys_s32_raw")
    sync = source.index("sync_warp", release)
    acquire = source.index('sem="acquire", scope="sys"', sync)
    final_fence = source.rindex("fence_acq_rel_sys")
    assert release < sync < acquire < final_fence


def test_epoch_reset_source_keeps_release_acquire_four_phase_protocol() -> None:
    source = inspect.getsource(SplitEpochResetBarrier)
    for fragment in (
        "status = phase_counter[0] & Int32(3)",
        "signal_phase = status & Int32(1)",
        "signal_sign = status >> Int32(1)",
        "signal_delta = Int32(-1)",
        "target = Int32(0)",
        "red_add_release_sys_s32_raw",
        'sem="acquire", scope="sys"',
        'sem="relaxed"',
    ):
        assert fragment in source
    release = source.index("red_add_release_sys_s32_raw")
    warp_sync = source.index("sync_warp", release)
    phase_advance = source.index("atomic_add", warp_sync)
    acquire = source.index('sem="acquire", scope="sys"', phase_advance)
    assert release < warp_sync < phase_advance < acquire
