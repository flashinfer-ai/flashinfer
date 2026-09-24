"""Graph buffer validation without requiring a transport or multiple GPUs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpTensors
from flashinfer.moe_ep.modes import split_layer


@pytest.fixture
def layer(monkeypatch):
    # Exercise the real graph-state factory with a stub transport; these
    # checks must reject invalid buffers before native handle construction.
    result = split_layer.MoEEpSplitLayer.__new__(split_layer.MoEEpSplitLayer)
    torch.nn.Module.__init__(result)
    result._destroyed = False
    result._graph_state = None
    result._runtime = None
    result._bootstrap = BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    result._fleet_params = FleetParams(
        num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
    )
    result._fleet = Mock()
    monkeypatch.setattr(split_layer, "_is_capturing", lambda: False)
    monkeypatch.setattr(split_layer, "ensure_bootstrap_dist_validated", lambda _: None)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0)
    )
    yield result
    result.destroy()


@pytest.fixture
def tensors():
    return MoEEpTensors(
        hidden_states=torch.zeros(4, 8, dtype=torch.bfloat16),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.full((4, 2), 0.5),
    )


def test_graph_state_accepts_in_place_updates_and_equivalent_views(layer, tensors):
    state = layer.create_graph_state(tensors)
    tensors.hidden_states.fill_(2)
    tensors.topk_ids.fill_(1)
    tensors.topk_weights.fill_(0.25)
    state._check(layer, tensors)
    aliases = MoEEpTensors(
        hidden_states=tensors.hidden_states.view_as(tensors.hidden_states),
        topk_ids=tensors.topk_ids.view_as(tensors.topk_ids),
        topk_weights=tensors.topk_weights.view_as(tensors.topk_weights),
    )
    state._check(layer, aliases)


@pytest.mark.parametrize("name", ["hidden_states", "topk_ids", "topk_weights"])
@pytest.mark.parametrize("change", ["dtype", "strides"])
def test_graph_state_rejects_same_address_reinterpretation(
    layer, tensors, name, change
):
    state = layer.create_graph_state(tensors)
    original = getattr(tensors, name)
    if change == "dtype":
        dtype = {
            torch.bfloat16: torch.float16,
            torch.int64: torch.float64,
            torch.float32: torch.int32,
        }[original.dtype]
        replacement = original.view(dtype)
    else:
        replacement = original.as_strided(original.shape, (1, original.shape[0]))
    assert replacement.data_ptr() == original.data_ptr()
    assert replacement.shape == original.shape
    setattr(tensors, name, replacement)
    with pytest.raises(ValueError, match=name):
        state._check(layer, tensors)


@pytest.mark.parametrize("name", ["hidden_states", "topk_ids", "topk_weights", "out"])
def test_graph_state_rejects_in_place_storage_rebinding(layer, tensors, name):
    state = layer.create_graph_state(tensors)
    bound = state.out if name == "out" else getattr(tensors, name)
    original_ptr = bound.data_ptr()
    bound.set_(bound.clone())
    assert bound.data_ptr() != original_ptr
    with pytest.raises(ValueError, match=name):
        state._check(layer, tensors)


def test_graph_state_rejects_output_layout_mutation(layer, tensors):
    state = layer.create_graph_state(tensors)
    state.out.as_strided_(state.out.shape, (1, state.out.shape[0]))
    with pytest.raises(ValueError, match="out"):
        state._check(layer, tensors)


def test_graph_state_alias_cannot_hide_rebound_handle_weights(layer, tensors):
    state = layer.create_graph_state(tensors)
    bound = tensors.topk_weights
    tensors.topk_weights = bound.view_as(bound)
    bound.set_(bound.clone())
    with pytest.raises(ValueError, match="topk_weights"):
        state._check(layer, tensors)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "strides"])
def test_create_graph_state_rejects_invalid_output_before_handle(
    layer, tensors, invalid
):
    if invalid == "shape":
        out = torch.empty(3, 8, dtype=torch.bfloat16)
    elif invalid == "dtype":
        out = torch.empty_like(tensors.hidden_states, dtype=torch.float16)
    elif invalid == "device":
        # A distinct device without requiring a physical GPU.
        out = torch.empty_like(tensors.hidden_states, device="meta")
    else:
        out = torch.empty(8, 4, dtype=torch.bfloat16).t()
    with pytest.raises(
        ValueError, match="out must be contiguous and match hidden_states"
    ):
        layer.create_graph_state(tensors, out=out)
    layer._fleet.create_handle.assert_not_called()


def test_create_graph_state_uses_supplied_output(layer, tensors):
    out = torch.empty_like(tensors.hidden_states)
    state = layer.create_graph_state(tensors, out=out)
    assert state.out is out
    state._check(layer, tensors)
