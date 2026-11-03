"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Block-run split of the affine apply / pair-map grids (CPU-only).

Whatever run count ``_affine_apply_runs`` picks, the items must cover every
block of every window-head exactly once (a block's arithmetic does not depend
on its run), and the pick never costs more waves x blocks than the base
``ceil(sm_count / window_heads)`` split.
"""

import itertools

import pytest

import flashinfer.cake_kda_tf32_runtime as runtime

SM_COUNTS = (148, 152)
HEADS = (8, 12, 16, 32)
WINDOWS = (1, 2, 5, 8, 9, 10, 11, 12, 13, 18, 20)
TOKENS = (64, 300, 683, 704, 1024, 2048, 3641)


def _blocks(length):
    return ((length + runtime.CHUNK - 1) // runtime.CHUNK + 1) // 2


def _items(tail_lengths, heads, sm_count, pairmap):
    cu = [0, 0]
    for length in tail_lengths:
        cu.append(cu[-1] + (length + runtime.CHUNK - 1) // runtime.CHUNK)
    return runtime.build_affine_apply_items(
        tail_lengths=tail_lengths,
        cu_chunk_offsets=cu,
        snap_starts=list(range(len(tail_lengths))),
        tail_offsets=[0] * len(tail_lengths),
        num_heads=heads,
        rows_enabled=not pairmap,
        sm_count=sm_count,
        pairmap=pairmap,
        owns_final=False,
    )


def _cost(window_heads, sm_count, max_blocks, runs):
    waves = -(-(runs * window_heads) // sm_count)
    return waves * -(-max_blocks // min(runs, max_blocks))


@pytest.mark.parametrize("pairmap", [False, True])
@pytest.mark.parametrize("sm_count", SM_COUNTS)
def test_items_cover_every_block_once(sm_count, pairmap):
    for windows, heads, length in itertools.product(WINDOWS, HEADS, TOKENS):
        tails = [length] * (windows - 1) + [max(32, length // 3)]
        items = _items(tails, heads, sm_count, pairmap)
        runs = {}
        for item in items:
            runs.setdefault(item[0], []).append((item[3], item[4]))
        assert len(runs) == windows * heads
        for key, spans in runs.items():
            blocks = _blocks(tails[key // heads])
            spans.sort()
            assert spans[0][0] == 0 and spans[-1][1] == blocks
            assert all(a[1] == b[0] for a, b in zip(spans, spans[1:], strict=False))


@pytest.mark.parametrize("pairmap", [False, True])
@pytest.mark.parametrize("sm_count", SM_COUNTS)
def test_runs_never_cost_more_than_the_base_split(sm_count, pairmap):
    for windows, heads, length in itertools.product(WINDOWS, HEADS, TOKENS):
        window_heads = windows * heads
        blocks = _blocks(length)
        base = max(1, -(-sm_count // window_heads))
        runs = runtime._affine_apply_runs(window_heads, sm_count, blocks, pairmap)
        assert runs >= 1
        assert _cost(window_heads, sm_count, blocks, runs) <= _cost(
            window_heads, sm_count, blocks, base
        )
        if runs != base:
            assert window_heads > sm_count or (pairmap and base == 2)


@pytest.mark.parametrize("sm_count", SM_COUNTS)
def test_split_of_the_layer_windows(sm_count):
    # 11 tail windows x 12 heads, 11 blocks each: the pair map runs one wave of
    # whole windows, the apply keeps two runs per window-head.
    assert len(_items([683] * 11, 12, sm_count, True)) == 132
    assert len(_items([683] * 11, 12, sm_count, False)) == 264
    # 22 blocks: one run is no cheaper than two, so both keep the base split.
    assert len(_items([1408] * 11, 12, sm_count, True)) == 264
    # 13 x 12 window-heads exceed the SMs: two runs replace the sub-wave tail.
    assert len(_items([1408] * 13, 12, sm_count, True)) == 312
    assert len(_items([1408] * 13, 12, sm_count, False)) == 312
