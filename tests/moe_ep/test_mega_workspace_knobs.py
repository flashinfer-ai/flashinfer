"""Fixed knob selection for shared-weight MegaMoE workspace profiles."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from .test_mega_workspace_api import _inputs


@pytest.fixture
def make_layer(monkeypatch):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl import (
        backend,
    )
    from flashinfer.moe_ep.core.kernel import workspace_pool

    cls = backend.Nvfp4CutedslMegaKernelBackend
    monkeypatch.setattr(backend, "validate_mega_arch", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(cls, "_uses_native_topk_reduce", lambda *args: False)
    monkeypatch.setattr(cls, "validate_transformed_weights", mock.Mock())
    monkeypatch.setattr(workspace_pool, "_POOL", {})
    monkeypatch.setattr(workspace_pool, "_KEY_BY_ID", {})

    def allocate(self, fleet):
        return mock.MagicMock(
            name=f"workspace_{fleet.max_tokens_per_rank}", topk_idx=None
        )

    monkeypatch.setattr(cls, "_allocate_workspace", allocate)
    layers = []

    def make(knobs=None):
        layer = MoEEpMegaLayer(
            BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            FleetParams(num_experts=4, max_tokens_per_rank=64, token_hidden_size=128),
            MoEWeightPack(w13=torch.empty(0), w2=torch.empty(0)),
            MegaConfig(
                megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                    intermediate_size=128, top_k=2, knobs=knobs
                ),
                preprocess_weights=False,
                transformed_weights=(mock.sentinel.w1, mock.sentinel.w2),
            ),
        )
        layers.append(layer)
        return layer

    yield make
    for layer in reversed(layers):
        layer.destroy()
    assert workspace_pool.pooled_workspace_count() == 0


def test_explicit_knobs_are_copied_and_pool_by_configuration(make_layer):
    from flashinfer.moe_ep.core.kernel.workspace_pool import pooled_workspace_refcount

    owner = make_layer({"flag_batch": 8})
    other = make_layer()
    different = make_layer()
    knobs = {"flag_batch": 4, "mma_tiler_mnk": [256, 128, 256]}
    first = owner.create_workspace(128, knobs=knobs)
    shared = other.create_workspace(128, knobs=knobs)
    separate = different.create_workspace(128, knobs={"flag_batch": 8})

    raw = first._backend_workspace
    assert shared._backend_workspace is raw
    assert separate._backend_workspace is not raw
    assert pooled_workspace_refcount(raw) == 2
    knobs["mma_tiler_mnk"][0] = 1
    assert first._kernel._kernel_config.knobs["mma_tiler_mnk"] == [256, 128, 256]
    assert owner._megakernel_config.knobs == {"flag_batch": 8}
    first.destroy()
    raw.destroy.assert_not_called()
    shared.destroy()
    raw.destroy.assert_called_once()


def test_execution_uses_selected_backend_and_clears_destroyed_state(
    make_layer, monkeypatch
):
    layer = make_layer()
    default = layer.create_workspace(64, knobs={"flag_batch": 4})
    other = layer.create_workspace(64, live_tokens=8, knobs={"flag_batch": 8})
    assert other.live_tokens == 8
    methods = (
        "validate_forward",
        "validate_capture_ready",
        "stage_inputs",
        "compute",
        "compute_unfinalized",
        "set_compile_tokens_per_rank",
        "workspace_output",
    )
    for name in methods:
        monkeypatch.setattr(
            layer._kernel, name, mock.Mock(side_effect=AssertionError(name))
        )
        for handle in (default, other):
            monkeypatch.setattr(handle._kernel, name, mock.Mock())

    tensors = _inputs(8)
    output = torch.empty(8, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    for selected in (None, other, None):
        handle = default if selected is None else selected
        kernel, raw = handle._kernel, handle._backend_workspace
        kernel.compute.reset_mock()
        layer.forward(tensors, workspace=selected)
        kernel.validate_forward.assert_called_with(
            tensors, handle._fleet_params, quantize_input=True
        )
        kernel.validate_capture_ready.assert_called_with(raw, layer._transformed)
        kernel.stage_inputs.assert_called_with(tensors, raw, quantize_input=True)
        assert kernel.compute.call_args.args == (raw, layer._transformed)
        layer.stage_inputs(tensors, workspace=selected, compile_tokens_per_rank=16)
        kernel.set_compile_tokens_per_rank.assert_called_with(raw, 16)
        layer.compute_staged(output=output)
        kernel.compute.assert_called_with(raw, layer._transformed, output=output)
    assert layer.output_buffer is default._kernel.workspace_output.return_value
    default._kernel.workspace_output.assert_called_once_with(default._backend_workspace)

    monkeypatch.setattr(layer._kernel, "supports_unfinalized_output", True)
    for handle in (default, other):
        kernel, raw = handle._kernel, handle._backend_workspace
        assert (
            layer.forward_unfinalized(tensors, workspace=handle)
            is kernel.compute_unfinalized.return_value
        )
        kernel.stage_inputs.assert_called_with(tensors, raw, quantize_input=True)
        kernel.compute_unfinalized.assert_called_once_with(raw, layer._transformed)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    other.destroy()
    layer.compute_staged(output=output)
    raw = default._backend_workspace
    default._kernel._thunk_states[(id(raw),)] = mock.sentinel.thunk
    default.destroy()
    assert not default._kernel._thunk_states
    assert id(raw) not in layer._forward_signatures
    with pytest.raises(ValueError, match="requires a prior"):
        layer.compute_staged(output=output)


def test_override_does_not_replace_allocated_default(make_layer):
    layer = make_layer()
    raw = layer._ensure_workspace()
    with pytest.raises(ValueError, match="already allocated default"):
        layer.create_workspace(64, knobs={"flag_batch": 4})
    assert layer._ensure_workspace() is raw
    raw.destroy.assert_not_called()
    handle = layer.create_workspace(64)
    assert handle._backend_workspace is raw
