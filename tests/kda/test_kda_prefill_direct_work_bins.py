"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""The bin table of a direct M128 body built with work bins (CPU-only).

The table holds the bin offsets as absolute table indices, then the task ids
``sequence * num_heads + head``; tasks go longest first to the least-loaded
bin (the planner's LPT rule), one bin per SM up to the task count.
"""

import pytest

import flashinfer.cake_kda_tf32_runtime as runtime
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
