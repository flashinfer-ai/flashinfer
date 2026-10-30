"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""The bin table of a direct M128 body built with work bins (CPU-only).

The table holds the bin offsets as absolute table indices, then the task ids
``sequence * num_heads + head``; tasks go longest first to the least-loaded
bin (the planner's LPT rule), one bin per SM up to the task count. A launch
asks for the work-binned program only where a bin runs tasks back to back on
the critical path (its chunk load exceeds the longest task), and the loader
resolves it through the work-bin routes.
"""

import types

import pytest

import flashinfer.cake_kda_tf32_runtime as runtime
import flashinfer.jit.cake_kda_tf32 as jit
import flashinfer.kda_prefill as planner

CASES = (
    ((64,) * 4, 12, 148),
    ((64,) * 64, 16, 148),
    ((128,) * 16, 12, 152),
    ((1024, 64) * 8, 12, 148),
    ((4096,) * 4, 16, 152),
    (tuple(17 + (index * 113) % 239 for index in range(16)), 16, 148),
    ((1,), 1, 148),
)


@pytest.mark.parametrize(("lengths", "num_heads", "workers"), CASES)
def test_direct_work_bins_layout_and_lpt(lengths, num_heads, workers):
    table, num_bins = runtime._make_direct_work_bins(
        lengths,
        num_heads=num_heads,
        worker_count=workers,
        chunk_tokens=runtime.BF16_M128_CHUNK,
    )
    tasks = len(lengths) * num_heads
    assert num_bins == min(tasks, workers)
    offsets, ids = table[: num_bins + 1], table[num_bins + 1 :]
    assert offsets[0] == num_bins + 1 and offsets[-1] == len(table)
    assert all(left < right for left, right in zip(offsets, offsets[1:], strict=False))
    assert sorted(ids) == list(range(tasks))
    order = sorted(range(len(lengths)), key=lambda index: (-lengths[index], index))
    ref_ids, ref_offsets, _loads = planner._make_lpt_task_bins(
        tuple(lengths[index] for index in order),
        num_heads=num_heads,
        sm_count=num_bins,
    )
    assert list(ids) == [
        order[task // num_heads] * num_heads + task % num_heads for task in ref_ids
    ]
    assert [offset - (num_bins + 1) for offset in offsets] == list(ref_offsets)


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(num_heads=0, worker_count=148, chunk_tokens=32),
        dict(num_heads=12, worker_count=0, chunk_tokens=32),
        dict(num_heads=12, worker_count=148, chunk_tokens=0),
    ),
)
def test_direct_work_bins_reject_empty_inputs(kwargs):
    with pytest.raises(ValueError):
        runtime._make_direct_work_bins((64,), **kwargs)
    with pytest.raises(ValueError):
        runtime._make_direct_work_bins(
            (), num_heads=12, worker_count=148, chunk_tokens=32
        )


MIXED = (17, 64, 65, 127, 128, 200, 255, 96) * 2


@pytest.mark.parametrize(
    ("lengths", "num_heads", "sm_count", "binned"),
    (
        # At most one task per SM.
        ((64,) * 4, 12, 148, False),
        ((64,) * 4, 16, 152, False),
        ((4096,) * 4, 16, 148, False),
        ((37,) * 37, 4, 148, False),
        # Equal tasks beyond one per SM: some bin runs two or more back to back.
        ((64,) * 16, 12, 148, True),
        ((128,) * 16, 16, 148, True),
        ((256,) * 16, 12, 152, True),
        ((64,) * 64, 16, 152, True),
        ((37,) * 37, 4, 147, True),
        # The longest task alone sets the makespan: short tasks fill the other bins.
        ((1024, 64) * 8, 12, 148, False),
        ((1024, 64) * 8, 16, 148, False),
        (MIXED, 12, 148, False),
        (MIXED, 16, 148, False),
        (MIXED, 16, 64, True),
        # An empty sequence keeps the one-task-per-CTA launch.
        ((0,) + (64,) * 16, 12, 148, False),
    ),
)
def test_direct_work_bins_only_where_a_bin_chains_tasks(
    lengths, num_heads, sm_count, binned
):
    assert (
        runtime._use_direct_work_bins(
            lengths,
            num_heads=num_heads,
            sm_count=sm_count,
            chunk_tokens=runtime.BF16_M128_CHUNK,
        )
        is binned
    )


def test_n16_work_bins_count_16_token_chunks():
    # Three tasks on two SMs. In 32-token chunks the 40-token task (2) sets the
    # makespan beside the bin of both 32-token tasks (1 + 1); in 16-token chunks
    # that bin (2 + 2) exceeds it (3).
    lengths = (40, 32, 32)
    assert not runtime._use_direct_work_bins(
        lengths, num_heads=1, sm_count=2, chunk_tokens=runtime.BF16_M128_CHUNK
    )
    assert runtime._use_direct_work_bins(
        lengths, num_heads=1, sm_count=2, chunk_tokens=runtime.BF16_N16_M128_CHUNK
    )


def test_factory_takes_the_work_bin_program_only_when_asked(monkeypatch):
    arch = "sm_100a"
    monkeypatch.setattr(jit, "device_arch", lambda device=None: arch)
    monkeypatch.setattr(
        jit,
        "FACTORIES",
        {
            arch: {
                "family": {
                    jit._key((32,), {"a": 1}): "binned_module",
                    jit._key((16,), {"a": 1}): "plain_module",
                }
            }
        },
    )
    monkeypatch.setattr(
        jit,
        "_RENDERED_ROUTES",
        {arch: {"binned_module": "row_program", "plain_module": "plain_program"}},
    )
    monkeypatch.setattr(jit, "_RENDERED_HEAD_ROUTES", {arch: {}})
    monkeypatch.setattr(
        jit, "_RENDERED_WORK_BIN_ROUTES", {arch: {"binned_module": "bins_program"}}
    )
    monkeypatch.setattr(jit, "PreparedKernel", lambda name, arch: (name, arch))
    assert jit._factory("family", 32, a=1) == ("row_program", arch)
    assert jit._factory("family", 32, a=1, route_work_bins=False) == (
        "row_program",
        arch,
    )
    assert jit._factory("family", 32, a=1, route_work_bins=True) == (
        "bins_program",
        arch,
    )
    # A module without a work-binned program keeps its route.
    assert jit._factory("family", 16, a=1, route_work_bins=True) == (
        "plain_program",
        arch,
    )


def test_a_program_the_work_bin_routes_name_runs_work_bins(monkeypatch):
    arch = "sm_100a"
    record = {"ffi_entry": "run", "arg_plan": [["buffer", "q"], ["grid", "grid_x"]]}
    monkeypatch.setattr(jit, "MODULES", {"bins_program": record, "row_program": record})
    monkeypatch.setattr(jit, "load", lambda name, arch: types.SimpleNamespace(run=None))
    monkeypatch.setattr(
        jit, "_RENDERED_WORK_BIN_ROUTES", {arch: {"module": "bins_program"}}
    )
    assert jit.PreparedKernel("bins_program", arch).work_bins is True
    assert jit.PreparedKernel("row_program", arch).work_bins is False
    assert jit.PreparedKernel("bins_program", "sm_103a").work_bins is False
