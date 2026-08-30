"""Host-only contract tests for reusable MegaMoE workspace handles."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest import mock

import pytest


def _fake_deep_gemm_transformed(
    *,
    num_experts: int = 1,
    intermediate: int = 128,
    hidden: int = 128,
):
    import torch

    fc1_out = 2 * intermediate
    w1 = torch.zeros(num_experts, fc1_out, hidden // 2, dtype=torch.int8)
    sf1 = torch.zeros(num_experts, fc1_out, hidden // 32)
    w2 = torch.zeros(num_experts, hidden, intermediate // 2, dtype=torch.int8)
    sf2 = torch.zeros(num_experts, hidden, intermediate // 32)
    return ((w1, sf1), (w2, sf2))


def _make_layer(*, preprocess_weights: bool = False):
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
    )

    transformed = _fake_deep_gemm_transformed()
    transformed_arg = None if preprocess_weights else transformed
    preprocess_target = (
        "flashinfer.moe_ep.backends.mega.kernel.sm100."
        "fp8_fp4_bf16_deepgemm.backend.DeepGemmMegaKernelBackend."
        "preprocess_weights"
    )
    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "fp8_fp4_bf16_deepgemm.backend.validate_mega_arch"
        ),
        mock.patch(preprocess_target, return_value=transformed) as preprocess_mock,
    ):
        layer = MoEEpMegaLayer(
            bootstrap=BootstrapConfig(
                world_size=1,
                rank=0,
                auto_bootstrap=False,
            ),
            fleet_params=FleetParams(
                num_experts=1,
                max_tokens_per_rank=64,
                token_hidden_size=128,
            ),
            weights=MoEWeightPack(
                w13=torch.zeros(1, 256, 128),
                w2=torch.zeros(1, 128, 128),
            ),
            backend=MegaConfig(
                megakernel=Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
                    intermediate_size=128,
                    top_k=2,
                ),
                quantize_input=True,
                preprocess_weights=preprocess_weights,
                transformed_weights=transformed_arg,
            ),
        )
    return layer, transformed, preprocess_mock


def _inputs(num_tokens: int):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    return MoEEpTensors(
        hidden_states=torch.zeros(num_tokens, 128, dtype=torch.bfloat16),
        topk_ids=torch.zeros(num_tokens, 2, dtype=torch.int64),
        topk_weights=torch.zeros(num_tokens, 2, dtype=torch.float32),
    )


def _create_workspace(layer, capacity: int, raw_workspace=None):
    raw_workspace = raw_workspace or mock.MagicMock(name=f"workspace_{capacity}")
    with (
        mock.patch.object(layer._kernel, "validate_init") as validate_init,
        mock.patch.object(
            layer._kernel,
            "prepare_workspace",
            return_value=raw_workspace,
        ) as prepare_workspace,
    ):
        handle = layer.create_workspace(capacity)
    return handle, raw_workspace, validate_init, prepare_workspace


def test_create_workspace_derives_only_capacity_from_layer_fleet_params():
    layer, _, _ = _make_layer()
    handle, raw_workspace, validate_init, prepare_workspace = _create_workspace(
        layer,
        256,
    )

    expected = dataclasses.replace(
        layer._fleet_params,
        max_tokens_per_rank=256,
    )
    assert handle.max_tokens_per_rank == 256
    assert handle._fleet_params == expected
    assert handle._backend_workspace is raw_workspace
    for field in dataclasses.fields(expected):
        if field.name != "max_tokens_per_rank":
            assert getattr(handle._fleet_params, field.name) == getattr(
                layer._fleet_params,
                field.name,
            )
    validate_init.assert_called_once_with(layer._bootstrap, expected)
    prepare_workspace.assert_called_once_with(layer._bootstrap, expected)

    handle.destroy()
    layer.destroy()


def test_same_capacity_shares_pool_entry_and_different_capacity_does_not(
    monkeypatch,
):
    from flashinfer.moe_ep.core.kernel import workspace_pool

    monkeypatch.setattr(workspace_pool, "_POOL", {})
    monkeypatch.setattr(workspace_pool, "_KEY_BY_ID", {})
    layer, _, _ = _make_layer()
    allocated: list[tuple[int, mock.MagicMock]] = []

    def allocate(fleet_params):
        workspace = mock.MagicMock(name=f"workspace_{fleet_params.max_tokens_per_rank}")
        allocated.append((fleet_params.max_tokens_per_rank, workspace))
        return workspace

    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(
            layer._kernel,
            "_workspace_pool_key",
            side_effect=lambda fp: ("workspace-api", fp.max_tokens_per_rank),
        ),
        mock.patch.object(
            layer._kernel,
            "_allocate_workspace",
            side_effect=allocate,
        ),
    ):
        first = layer.create_workspace(128)
        second = layer.create_workspace(128)
        large = layer.create_workspace(256)

    assert first._backend_workspace is second._backend_workspace
    assert first._backend_workspace is not large._backend_workspace
    assert [capacity for capacity, _ in allocated] == [128, 256]
    assert workspace_pool.pooled_workspace_refcount(first._backend_workspace) == 2
    assert workspace_pool.pooled_workspace_refcount(large._backend_workspace) == 1

    shared_raw = first._backend_workspace
    large_raw = large._backend_workspace
    with (
        mock.patch.object(
            layer._kernel,
            "_forget_local_workspace_state",
            wraps=layer._kernel._forget_local_workspace_state,
        ) as forget_local,
        mock.patch.object(
            layer._kernel,
            "_forget_workspace_state",
            wraps=layer._kernel._forget_workspace_state,
        ) as forget_global,
    ):
        first.destroy()
        forget_local.assert_called_once_with(shared_raw)
        forget_global.assert_not_called()
        shared_raw.destroy.assert_not_called()
        assert workspace_pool.pooled_workspace_refcount(shared_raw) == 1
        second.destroy()
        assert forget_local.call_count == 2
        forget_global.assert_called_once_with(shared_raw)
        shared_raw.destroy.assert_called_once()
        large.destroy()
        assert forget_local.call_count == 3
        assert forget_global.call_count == 2
        forget_global.assert_called_with(large_raw)
        large_raw.destroy.assert_called_once()
    assert workspace_pool.pooled_workspace_count() == 0
    layer.destroy()


def test_forward_uses_selected_workspace_and_capacity_profile():
    layer, transformed, _ = _make_layer()
    handle, raw_workspace, _, _ = _create_workspace(layer, 128)
    tensors = _inputs(8)

    with (
        mock.patch.object(
            layer._kernel,
            "validate_forward",
            wraps=layer._kernel.validate_forward,
        ) as validate_forward,
        mock.patch.object(layer._kernel, "stage_inputs") as stage_inputs,
        mock.patch.object(
            layer._kernel,
            "compute",
            side_effect=lambda workspace, weights, *, output: output,
        ) as compute,
    ):
        output = layer.forward(tensors, workspace=handle)

    validate_forward.assert_called_once_with(
        tensors,
        handle._fleet_params,
        quantize_input=True,
    )
    stage_inputs.assert_called_once_with(
        tensors,
        raw_workspace,
        quantize_input=True,
    )
    assert compute.call_args.args == (raw_workspace, transformed)
    assert compute.call_args.kwargs["output"] is output
    assert output.shape == (8, 128)

    handle.destroy()
    layer.destroy()


def test_workspace_over_capacity_rejected_before_allocation_stage_or_compute():
    from flashinfer.moe_ep import MoEEpConfigError

    layer, _, _ = _make_layer()
    handle, _, _, _ = _create_workspace(layer, 128)
    tensors = _inputs(129)

    with (
        mock.patch("flashinfer.moe_ep.modes.mega_layer.torch.empty") as allocate_output,
        mock.patch.object(layer._kernel, "stage_inputs") as stage_inputs,
        mock.patch.object(layer._kernel, "compute") as compute,
        pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"),
    ):
        layer.forward(tensors, workspace=handle)

    allocate_output.assert_not_called()
    stage_inputs.assert_not_called()
    compute.assert_not_called()
    handle.destroy()
    layer.destroy()


def test_forward_rejects_cross_layer_and_closed_workspace_before_staging():
    from flashinfer.moe_ep import MoEEpConfigError

    first_layer, _, _ = _make_layer()
    other_layer, _, _ = _make_layer()
    handle, _, _, _ = _create_workspace(first_layer, 128)
    tensors = _inputs(8)

    with (
        mock.patch.object(other_layer._kernel, "stage_inputs") as other_stage,
        mock.patch.object(other_layer._kernel, "compute") as other_compute,
        pytest.raises(MoEEpConfigError, match="different layer"),
    ):
        other_layer.forward(tensors, workspace=handle)
    other_stage.assert_not_called()
    other_compute.assert_not_called()

    with mock.patch.object(first_layer._kernel, "destroy") as destroy:
        handle.close()
        destroy.assert_called_once()
    with (
        mock.patch.object(first_layer._kernel, "stage_inputs") as first_stage,
        mock.patch.object(first_layer._kernel, "compute") as first_compute,
        pytest.raises(MoEEpConfigError, match="destroyed"),
    ):
        first_layer.forward(tensors, workspace=handle)
    first_stage.assert_not_called()
    first_compute.assert_not_called()

    first_layer.destroy()
    other_layer.destroy()


def test_workspace_destroy_is_idempotent_and_layer_destroy_cleans_all_handles():
    layer, _, _ = _make_layer()
    first_raw = mock.MagicMock(name="first_raw")
    second_raw = mock.MagicMock(name="second_raw")
    default_raw = mock.MagicMock(name="default_raw")
    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(
            layer._kernel,
            "prepare_workspace",
            side_effect=[first_raw, second_raw],
        ),
    ):
        first = layer.create_workspace(128)
        second = layer.create_workspace(256)
    layer._workspace = default_raw

    with mock.patch.object(layer._kernel, "destroy") as destroy:
        first.destroy()
        first.destroy()
        destroy.assert_called_once_with(first_raw)

        layer.destroy()
        assert second.is_destroyed
        assert second._backend_workspace is None
        assert first.is_destroyed
        assert layer._workspace is None
        assert layer._destroyed
        assert destroy.call_count == 3
        destroy.assert_any_call(second_raw)
        destroy.assert_any_call(default_raw)

        layer.destroy()
        assert destroy.call_count == 3


def test_creating_workspaces_does_not_repeat_weight_preprocessing():
    layer, transformed, preprocess_weights = _make_layer(preprocess_weights=True)
    assert layer.preprocessing_count == 1
    preprocess_weights.assert_called_once()

    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(
            layer._kernel,
            "prepare_workspace",
            side_effect=[
                mock.MagicMock(name="small_workspace"),
                mock.MagicMock(name="large_workspace"),
            ],
        ),
    ):
        small = layer.create_workspace(128)
        large = layer.create_workspace(256)

    assert layer.transformed_weights is transformed
    assert layer.preprocessing_count == 1
    preprocess_weights.assert_called_once()
    small.destroy()
    large.destroy()
    layer.destroy()


def test_create_workspace_rejects_cuda_graph_capture_before_backend_calls():
    from flashinfer.moe_ep import MoEEpConfigError

    layer, _, _ = _make_layer()
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        mock.patch.object(layer._kernel, "validate_init") as validate_init,
        mock.patch.object(layer._kernel, "prepare_workspace") as prepare_workspace,
        pytest.raises(MoEEpConfigError, match="CUDA graph capture"),
    ):
        layer.create_workspace(128)

    validate_init.assert_not_called()
    prepare_workspace.assert_not_called()
    layer.destroy()


def test_workspace_and_layer_destruction_reject_cuda_graph_capture():
    from flashinfer.moe_ep import MoEEpConfigError

    layer, _, _ = _make_layer()
    handle, raw_workspace, _, _ = _create_workspace(layer, 128)
    default_workspace = mock.MagicMock(name="default_workspace")
    layer._workspace = default_workspace

    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        mock.patch.object(layer._kernel, "destroy") as destroy,
    ):
        with pytest.raises(MoEEpConfigError, match="CUDA graph capture"):
            handle.destroy()
        assert not handle.is_destroyed
        assert handle._backend_workspace is raw_workspace
        destroy.assert_not_called()

        with pytest.raises(MoEEpConfigError, match="CUDA graph capture"):
            layer.destroy()
        assert not layer._destroyed
        assert layer._workspace is default_workspace
        destroy.assert_not_called()

    layer.destroy()
    assert handle.is_destroyed


def test_create_workspace_rejects_mutable_auto_tuning_state():
    from flashinfer.moe_ep import MoEEpConfigError

    layer, _, _ = _make_layer()
    layer._megakernel_config = SimpleNamespace(knobs="auto")
    with (
        mock.patch.object(layer._kernel, "validate_init") as validate_init,
        mock.patch.object(layer._kernel, "prepare_workspace") as prepare_workspace,
        pytest.raises(MoEEpConfigError, match="knobs='auto'"),
    ):
        layer.create_workspace(128)

    validate_init.assert_not_called()
    prepare_workspace.assert_not_called()
    layer.destroy()
