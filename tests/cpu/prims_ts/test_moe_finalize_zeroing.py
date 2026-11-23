# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""FC2 initialization must not race FC1 reads of shared activation storage."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.prims_ts.moe import finalize

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "input_start,overlap", [(0, False), (64, True), (80, True), (128, False)]
)
def test_output_overlap_uses_byte_ranges(input_start, overlap):
    storage = torch.empty(128, dtype=torch.bfloat16)
    output = storage[32:64].view(4, 8)
    packed_input = storage.view(torch.uint8)[input_start : input_start + 32]
    assert finalize._overlapping_bytes(output, packed_input) is overlap
    assert finalize._overlapping_bytes(packed_input, output) is overlap
    assert not finalize._overlapping_bytes(output, packed_input[:0])
    assert not finalize._overlapping_bytes(output, packed_input.clone())


@pytest.mark.parametrize("overlap", [False, True])
def test_zeroing_orders_reads_and_preserves_disjoint_overlap(monkeypatch, overlap):
    storage = torch.empty(128, dtype=torch.bfloat16)
    output = storage[:64]
    packed_input = (
        storage.view(torch.uint8)[0:32]
        if overlap
        else storage.view(torch.uint8)[128:160]
    )
    calls = []
    main = SimpleNamespace(wait_event=lambda event: calls.append("main waits done"))
    auxiliary = SimpleNamespace(
        wait_event=lambda event: calls.append("aux waits ready")
    )
    events = iter(
        (
            SimpleNamespace(
                record=lambda stream: calls.append("ready on main")
                if stream is main
                else pytest.fail("wrong ready stream")
            ),
            SimpleNamespace(
                record=lambda stream: calls.append("done on aux")
                if stream is auxiliary
                else pytest.fail("wrong done stream")
            ),
        )
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: main)
    monkeypatch.setattr(torch.cuda, "Event", lambda: next(events))
    monkeypatch.setattr(finalize, "_zero_stream", lambda device: auxiliary)
    monkeypatch.setattr(
        finalize,
        "_zero_output",
        lambda output, stream: calls.append(
            "zero on main" if stream is main else "zero on aux"
        ),
    )
    with finalize.zero_output_during_fc1(output, read_input=packed_input):
        calls.append("FC1")
    assert calls == (
        ["FC1", "zero on main"]
        if overlap
        else [
            "ready on main",
            "FC1",
            "aux waits ready",
            "zero on aux",
            "done on aux",
            "main waits done",
        ]
    )
