"""Rubin host contracts and regressions; no Rubin kernel compilation."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import FleetParams, MoEEpTensors
from flashinfer.moe_ep.backends.mega.kernel.sm107.validation import (
    validate_forward_metadata,
    validate_routing_values,
    validate_unit_scalars,
    validate_weight_layout,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    Sm107BlockScaledMoeConfig,
    Sm107BlockScaledSymmBuffer,
    lookup_knobs,
    record_knobs,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    block_scaled,
    comm,
)


def _config(**kw):
    return Sm107BlockScaledMoeConfig(
        **(
            dict(
                num_total_experts=8,
                max_tokens_per_rank=3,
                num_topk=2,
                hidden=128,
                intermediate=64,
                rank=0,
                world_size=1,
            )
            | kw
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden": 192},
        {"hidden": 0},
        {"intermediate": -64},
        {"num_total_experts": 0},
        {"num_total_experts": 16385},
        {"num_topk": 0},
        {"num_topk": 9},
        {"world_size": 0},
        {"max_tokens_per_rank": 0},
        {"max_tokens_per_rank": 1048577},
        {"cluster_shape_mn": (2, 2)},
        {"cluster_shape_mn": (6, 1)},
        {"cluster_shape_mn": (0, 1)},
        {"cluster_shape_mn": (32, 1)},
        {"epi_flag_batches": (0, 2)},
        {"epi_flag_batches": (-1, 2)},
        {"token_in_flag_batch": 0},
        {"token_in_flag_batch": 33},
        {"token_padding_block": 0},
        {"sf_padding_block": 64},
        {"token_back_mode": "invalid"},
        {"work_id_mode": "invalid"},
        {"max_sm_count": 1},
        {"gate_up_clamp": float("nan")},
        {"schedule_policy": ("grouped", True)},
        {"mma_tiler_mnk": (128.0, 128, 128)},
    ],
)
def test_reject_unsafe_config_before_workspace_allocation(overrides):
    with pytest.raises((ValueError, TypeError)):
        _config(**overrides)


@pytest.mark.parametrize("capacity", [1, 2, 3, 5, 127, 128, 129])
@pytest.mark.parametrize("topk", [1, 2, 3, 4, 6, 8])
def test_router_vector_tail_has_allocated_storage(capacity, topk):
    cfg = _config(max_tokens_per_rank=capacity, num_topk=topk)
    count = cfg.padded_tokens_per_rank * topk
    last_load = ((count - 1) // 4) * 4
    assert last_load + 4 <= count
    assert cfg.max_tokens_per_rank == capacity
    assert capacity <= cfg.padded_tokens_per_rank <= capacity + 3


def _workspace():
    ws = object.__new__(Sm107BlockScaledSymmBuffer)
    ws.config = _config()
    ws.device = torch.device("cpu")
    ws._destroyed = False
    ws._compiled = mock.Mock()
    ws._launch_key = None
    ws._launch_kwargs = None
    ws._runtime_kwargs = mock.Mock(side_effect=lambda a, b: {"weights": (a, b)})
    ws.topk_idx = torch.tensor([[0, 1], [2, -1], [-1, -1], [-1, -1]])
    ws._pre_reduced_activation = torch.full((4, 2, 128), 7.0)
    ws.output_activation = torch.zeros(4, 128)
    return ws


def test_warmed_binding_switches_weights_and_stream_during_capture(monkeypatch):
    ws = _workspace()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    weights = [tuple(torch.empty(4) for _ in range(2)) for _ in range(4)]
    for stream, a, b in [(11, *weights[:2]), (22, *weights[:2]), (22, *weights[2:])]:
        monkeypatch.setattr(
            torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=stream)
        )
        ws.launch(a, b)
        assert ws._launch_weights == (a, b)
    assert ws._compiled.call_count == 3
    assert ws._runtime_kwargs.call_count == 3
    # Binding the same addresses and stream again reuses metadata.
    ws.launch(*weights[2:])
    assert ws._runtime_kwargs.call_count == 3


def test_first_compile_still_rejected_during_capture(monkeypatch):
    ws = _workspace()
    ws._compiled = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=11)
    )
    weights = (torch.empty(4), torch.empty(4))
    with pytest.raises(RuntimeError, match="cannot run"):
        ws.launch(weights, weights)
    ws._runtime_kwargs.assert_not_called()


def test_masked_routes_clear_stale_combine_slots(monkeypatch):
    ws = _workspace()
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=11)
    )
    weights = (torch.empty(4), torch.empty(4))
    ws.launch(weights, weights)
    assert (ws._pre_reduced_activation[ws.topk_idx < 0] == 0).all()
    assert (ws._pre_reduced_activation[ws.topk_idx >= 0] == 7).all()
    ws.topk_idx[0, 0] = -1
    ws.launch(weights, weights)
    assert (ws._pre_reduced_activation[0, 0] == 0).all()


def test_rejected_capture_destroy_can_retry(monkeypatch):
    ws = _workspace()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="cannot run"):
        ws.destroy()
    assert not ws._destroyed
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with (
        mock.patch.object(torch.cuda, "synchronize"),
        mock.patch.object(comm, "free_sym_tensor") as free,
    ):
        ws.destroy()
        ws.destroy()
        assert free.call_count == 4
    assert ws._destroyed


def test_occupancy_cache_is_per_device(monkeypatch):
    import cutlass.utils

    monkeypatch.setattr(block_scaled, "_MAX_ACTIVE_CLUSTERS_CACHE", {})
    hardware = mock.Mock()
    hardware.get_max_active_clusters.side_effect = [40, 28]
    monkeypatch.setattr(cutlass.utils, "HardwareInfo", lambda: hardware)
    for device, expected in [(0, 40), (1, 28), (0, 40)]:
        monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
        assert block_scaled._max_active_clusters(4) == expected
    assert hardware.get_max_active_clusters.call_count == 2


def test_cache_keeps_reduction_and_weighting_policy_separate(tmp_path, monkeypatch):
    path = tmp_path / "knobs.json"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    key = dict(
        dtype="nvfp4",
        world_size=4,
        hidden=128,
        intermediate=64,
        num_experts=8,
        topk=2,
        max_tokens=3,
        device="rubin-test",
    )
    knobs = {"reduce_topk_in_kernel": True}
    with pytest.raises(ValueError, match="allow_nondeterministic"):
        record_knobs(knobs, **key)
    record_knobs(knobs, allow_nondeterministic=True, **key)
    assert lookup_knobs(**key) is None
    assert lookup_knobs(**key, allow_nondeterministic=True) == knobs
    assert (
        lookup_knobs(**key, allow_nondeterministic=True, apply_topk_at_fc1=False)
        is None
    )
    data = json.loads(path.read_text())
    # The new upstream drop must not reuse tuning results from the old kernel.
    data["entries"][0]["backend_revision"] = "sm107-block-scaled-v2"
    path.write_text(json.dumps(data))
    assert lookup_knobs(**key, allow_nondeterministic=True) is None
    del data["entries"][0]["backend_revision"]
    path.write_text(json.dumps(data))
    assert lookup_knobs(**key, allow_nondeterministic=True) is None


def test_sm100_and_sm107_cache_entries_cannot_collide(tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe as sm100

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "shared.json"))
    key = dict(
        dtype="nvfp4",
        world_size=4,
        hidden=128,
        intermediate=64,
        num_experts=8,
        topk=2,
        max_tokens=128,
        device="NVIDIA Graphics Device",
    )
    old, new = {"flag_batch": 16}, {"reduce_topk_in_kernel": False}
    sm100.record_knobs(old, **key)
    record_knobs(new, **key)
    assert sm100.lookup_knobs(**key) == old
    assert lookup_knobs(**key) == new
    sm100.record_knobs({"flag_batch": 8}, **key)
    assert lookup_knobs(**key) == new


@pytest.mark.parametrize("ids", [[[0, 0]], [[-2, 0]], [[0, 8]], [[0, 2**32]]])
def test_invalid_routes_rejected_before_int32_conversion(ids):
    with pytest.raises(RuntimeError, match="unique expert"):
        validate_routing_values(torch.tensor(ids), torch.ones(1, 2), 8)


def test_masked_routes_and_finite_score_contract():
    validate_routing_values(torch.tensor([[-1, -1], [0, 7]]), torch.ones(2, 2), 8)
    with pytest.raises(RuntimeError, match="finite scores"):
        validate_routing_values(
            torch.tensor([[0, 1]]), torch.tensor([[1.0, float("nan")]]), 8
        )


@pytest.mark.parametrize("name", ["fc1_alpha", "fc2_alpha", "fc1_norm_const"])
def test_unsupported_scalars_are_never_silently_ignored(name):
    t = MoEEpTensors(torch.empty(0), torch.empty(0), torch.empty(0))
    setattr(t, name, torch.ones(1))
    with pytest.raises(ValueError, match=name):
        validate_unit_scalars(t)


def test_transformed_layout_and_scale_encoding_contract():
    w = torch.empty(2, 128, 128, dtype=torch.float8_e4m3fn)
    sf = torch.empty(2, 512, dtype=torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="K stride 1"):
        validate_weight_layout(w, sf, scale_dtype=sf.dtype)
    w = w.permute(0, 2, 1)
    validate_weight_layout(w, sf, scale_dtype=sf.dtype)
    with pytest.raises(ValueError, match="reinterpret raw bytes"):
        validate_weight_layout(w, sf.view(torch.uint8), scale_dtype=sf.dtype)


def test_scalar_input_shape_is_rejected_cleanly():
    with pytest.raises(ValueError, match="must be 2D"):
        validate_forward_metadata(
            torch.tensor(0),
            torch.empty(0),
            torch.empty(0),
            FleetParams(num_experts=8, max_tokens_per_rank=3, token_hidden_size=128),
            top_k=2,
            quantize_input=False,
            quant_kind="nvfp4",
        )


def test_no_dist_rejects_multirank(monkeypatch):
    from flashinfer.moe_ep import BootstrapConfig
    from flashinfer.moe_ep.core.runtime import sm107_block_scaled_runtime_requirements

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    with pytest.raises(ValueError, match="world_size=1"):
        sm107_block_scaled_runtime_requirements(BootstrapConfig(rank=0, world_size=2))


def test_ikr_requires_early_weighting():
    with pytest.raises(ValueError, match="apply_topk_at_fc1"):
        dataclasses.replace(
            _config(), reduce_topk_in_kernel=True, apply_topk_at_fc1=False
        )
