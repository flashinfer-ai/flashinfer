# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""FC2 initialization must not race FC1 reads of shared activation storage."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.prims_ts.moe import finalize

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "view",
    [
        "shape",
        "dtype",
        "device",
        "strided",
        "misaligned",
        "scale_alias",
        "valid",
        "one_row",
    ],
)
def test_inner_runner_validates_output_before_cuda_work(monkeypatch, view):
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.prims_ts.moe import runner

    tokens, hidden = (1 if view == "one_row" else 8), 128
    backing = torch.full((tokens, hidden * 2), 7, dtype=torch.bfloat16)
    output = backing[:, :hidden] if view == "one_row" else backing[:, ::2]
    if view == "shape":
        output = backing
    elif view == "dtype":
        output = torch.empty((tokens, hidden), dtype=torch.float32)
    elif view == "device":
        output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device="meta")
    elif view == "misaligned":
        output = backing.flatten()[1 : 1 + tokens * hidden].view(tokens, hidden)
    elif view in ("valid", "scale_alias"):
        output = backing.flatten()[: tokens * hidden].view(tokens, hidden)

    inner = object.__new__(runner.PrimsTsNvfp4MoERunner)
    inner.hidden_size = hidden
    config = SimpleNamespace(kwargs={}, build=lambda: SimpleNamespace(use_pdl=False))
    pair = SimpleNamespace(
        fc1=SimpleNamespace(cfg=config),
        fc2=SimpleNamespace(cfg=config),
        tile_n=512,
        moe_config_index=0,
    )
    monkeypatch.setattr(inner, "_map_tactic", lambda *args: pair)
    monkeypatch.setattr(
        runner, "is_prims_ts_nvfp4_supported", lambda *a, **kw: (True, "")
    )
    monkeypatch.setattr(runner, "fused_finalize_requested", lambda *args: True)
    monkeypatch.setattr(runner, "supports_fused_finalize", lambda *args: True)

    monkeypatch.setattr(
        runner,
        "_fused_finalize_config",
        lambda cfg: SimpleNamespace(fuse_moe_finalize=True),
    )

    def reached_cuda(**kwargs):
        raise RuntimeError("output accepted before CUDA work")

    monkeypatch.setattr(torch.cuda, "current_stream", reached_cuda)
    inputs = MoeRunnerInputs(
        output=output,
        hidden_states=torch.empty((tokens, hidden // 2), dtype=torch.uint8),
        routing_logits=None,
        topk_ids=None,
        expert_weights=None,
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    if view in ("valid", "one_row"):
        error, message = RuntimeError, "output accepted before CUDA work"
    elif view == "scale_alias":
        error, message = ValueError, "overlaps output2_scale_scalar"
    else:
        error = ValueError
        message = "16-byte aligned" if view == "misaligned" else "contiguous BF16"
    with pytest.raises(error, match=message):
        scale = (
            output.view(torch.float32).flatten()[1:2] if view == "scale_alias" else None
        )
        inner.forward(inputs.to_list(), output2_scale_scalar=scale)
    assert torch.all(backing == 7)


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


@pytest.mark.parametrize("overlap", ["neither", "input", "scales", "both"])
def test_zeroing_orders_reads_and_preserves_disjoint_overlap(monkeypatch, overlap):
    storage = torch.empty(128, dtype=torch.bfloat16)
    output = storage[:64]
    packed_input = (
        storage.view(torch.uint8)[0:32]
        if overlap in ("input", "both")
        else storage.view(torch.uint8)[128:160]
    )
    input_scales = (
        storage.view(torch.uint8)[32:64]
        if overlap in ("scales", "both")
        else storage.view(torch.uint8)[160:192]
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
    with finalize.zero_output_during_fc1(
        output, read_input=packed_input, read_input_scale=input_scales
    ):
        calls.append("FC1")
    assert calls == (
        ["FC1", "zero on main"]
        if overlap != "neither"
        else [
            "ready on main",
            "FC1",
            "aux waits ready",
            "zero on aux",
            "done on aux",
            "main waits done",
        ]
    )
