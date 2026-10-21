"""Compressed FC2 return payloads, scale resets, and numerical checks."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg
from tests.moe_ep.test_sm107_block_scaled_contracts import _config, _workspace


@pytest.mark.parametrize("combine_dtype", ["nvfp4", "mxfp8"])
def test_quantized_combine_rejects_atomic_topk_reduce(combine_dtype):
    with pytest.raises(ValueError, match="separate top-k"):
        _config(combine_dtype=combine_dtype, reduce_topk_in_kernel=True)


@pytest.mark.parametrize(
    "combine_dtype,sf_byte,slot_bytes", [("nvfp4", 0, 64), ("mxfp8", 127, 128)]
)
def test_masked_combine_clears_payload_and_scale_on_each_launch(
    monkeypatch, combine_dtype, sf_byte, slot_bytes
):
    ws = _workspace()
    ws.config = dataclasses.replace(ws.config, combine_dtype=combine_dtype)
    ws._pre_reduced_activation = torch.full((4, 2, slot_bytes), 255, dtype=torch.uint8)
    ws._pre_reduced_activation_sf = torch.full((4, 2, 16), 255, dtype=torch.uint8)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=11)
    )
    weights = (torch.empty(4), torch.empty(4))
    for route in (-1, ws.config.num_total_experts):
        ws.topk_idx[0, 0] = route
        ws._pre_reduced_activation_sf[0, 0].fill_(255)
        ws.launch(weights, weights)
        invalid = (ws.topk_idx < 0) | (ws.topk_idx >= ws.config.num_total_experts)
        assert (ws._pre_reduced_activation[invalid] == 0).all()
        assert (ws._pre_reduced_activation_sf[invalid] == sf_byte).all()
        assert (ws._pre_reduced_activation[~invalid] == 255).all()
        assert (ws._pre_reduced_activation_sf[~invalid] == 255).all()


def test_nvfp4_combine_uses_bf16_amax_per_16_values():
    block = torch.tensor(
        [0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.75, -2.5, 0, 0, 0, 0, 0]
    )
    rounded = torch.tensor([0, 0, 1, 1, 2, 2, 4, 4, 6, -1, -2, 0, 0, 0, 0, 0])
    value = torch.cat([block, block / 2, torch.zeros(16)]).reshape(1, -1)
    expected = torch.cat([rounded, rounded / 2, torch.zeros(16)]).reshape(1, -1)
    torch.testing.assert_close(
        pkg.round_trip_combine(value, "nvfp4"), expected, rtol=0, atol=0
    )


def test_nvfp4_combine_bulk_path_and_default_cache_fallback(monkeypatch):
    with pytest.raises(ValueError, match="token-back data path"):
        _config(combine_dtype="nvfp4", fc2_use_bulk=True)
    for mode in ("standalone_warps", "reuse_dispatch_warps"):
        _config(combine_dtype="nvfp4", fc2_use_bulk=True, token_back_mode=mode)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    knobs, source = pkg.resolve_knobs(
        dtype="mxfp8_e4m3",
        world_size=1,
        hidden=128,
        intermediate=64,
        num_experts=8,
        topk=2,
        max_tokens=3,
        combine_dtype="nvfp4",
    )
    assert source == "heuristic"
    assert pkg.is_valid_sm107(knobs, _config(combine_dtype="nvfp4"))


def test_mxfp8_combine_uses_e4m3_with_per_32_e8m0_scale():
    block = torch.zeros(32)
    block[:4] = torch.tensor([448, 1.0625, 1.1875, -1.0625])
    rounded = torch.zeros(32)
    rounded[:4] = torch.tensor([448, 1, 1.25, -1])
    value = torch.cat([block, block * 2, torch.zeros(32)]).reshape(1, -1)
    expected = torch.cat([rounded, rounded * 2, torch.zeros(32)]).reshape(1, -1)
    torch.testing.assert_close(
        pkg.round_trip_combine(value, "mxfp8"), expected, rtol=0, atol=0
    )
