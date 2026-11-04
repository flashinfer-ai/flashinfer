"""Host-only contract tests for reusable MegaMoE workspace handles."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from unittest import mock

import pytest

from flashinfer.moe_ep import FinalizeMode


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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_live_profiles_preserve_capacity_and_handle_lifetime():
    layer, _, _ = _make_layer()
    layer._kernel.supports_live_token_tuning = True
    # These fake workspaces exercise handle bookkeeping on any host GPU.
    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(
            layer._kernel, "prepare_workspace", side_effect=[object(), object()]
        ),
    ):
        small = layer.create_workspace(64, live_tokens=8)
        large = layer.create_workspace(64, live_tokens=32)
    assert small.max_tokens_per_rank == large.max_tokens_per_rank == 64
    assert (small.live_tokens, large.live_tokens) == (8, 32)
    assert layer._default_workspace_handle is None
    assert layer._resolve_workspace(small)[1] is small._backend_workspace
    with mock.patch.object(layer._kernel, "destroy"):
        small.destroy()
        assert not large.is_destroyed
        assert layer._resolve_workspace(large)[1] is large._backend_workspace
        layer.destroy()
    assert large.is_destroyed


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "config_name",
    [
        "Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig",
        "Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
        "Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
        "Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig",
        "Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig",
        "Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig",
        "Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
        "Sm107_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig",
        "Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig",
    ],
)
def test_live_tactics_change_pool_key_for_all_cached_backends(monkeypatch, config_name):
    import torch
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        "flashinfer.jit.cake_megamoe_topk_reduce.supports_device", lambda: False
    )
    options = {"knobs": "cache"} if config_name.startswith("Sm107") else {}
    backend = create_mega_kernel(
        getattr(moe_ep, config_name)(intermediate_size=256, top_k=2, **options)
    )
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    cache = SimpleNamespace(
        live_token_buckets=mock.Mock(return_value=[8, 16, 32]),
        resolve_knobs=mock.Mock(
            side_effect=lambda **key: (
                {"token_in_flag_batch": 4 if key["live_tokens"] <= 16 else 8},
                "cache",
            )
        ),
    )
    backend._ep_comm_group = object()
    monkeypatch.setattr(backend, "_tuning_cache", lambda: cache)
    if config_name.startswith("Sm90"):
        import sys

        # SM90 now pools by the resolved frontend config. Keep this host-only
        # test independent of the mutually exclusive SM90/SM100 kernel imports.
        monkeypatch.setitem(
            sys.modules,
            "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel",
            SimpleNamespace(
                resolve_hopper_fp8_mega_moe_config=lambda **kwargs: tuple(
                    sorted((key, repr(value)) for key, value in kwargs.items())
                ),
                _get_symm_buffer_for_hopper_fp8_mega_moe_from_resolved_config=mock.Mock(
                    side_effect=AssertionError
                ),
            ),
        )
    fleet = moe_ep.FleetParams(2, 64, 128)
    assert backend.live_token_buckets(fleet) == [8, 16, 32]
    keys = [
        backend._workspace_pool_key(dataclasses.replace(fleet, tuning_live_tokens=n))
        for n in (8, 16, 32)
    ]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    backend._workspace_knobs(dataclasses.replace(fleet, tuning_live_tokens=8))
    assert cache.resolve_knobs.call_count == 3
    assert all(c.kwargs["max_tokens"] == 64 for c in cache.resolve_knobs.call_args_list)


@pytest.mark.cpu_only
def test_live_profile_pool_shares_only_equal_tactics(monkeypatch, tmp_path):
    import torch
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.knob_cache import (
        record_knobs,
    )

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        "flashinfer.jit.cake_megamoe_topk_reduce.supports_device", lambda: False
    )
    config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        finalize_mode=FinalizeMode.NONE,
        apply_routing_weights_before_fc2=False,
    )
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    fleet = moe_ep.FleetParams(2, 64, 128)
    for live, flag in ((8, 4), (16, 4), (32, 8)):
        record_knobs(
            {"flag_batch": flag},
            dtype="nvfp4",
            world_size=1,
            hidden=128,
            intermediate=256,
            num_experts=2,
            topk=2,
            max_tokens=64,
            live_tokens=live,
            use_custom_finalize=True,
        )
    assert backend.live_token_buckets(fleet) == [8, 16, 32]
    keys = [
        backend._workspace_pool_key(dataclasses.replace(fleet, tuning_live_tokens=n))
        for n in (8, 16, 32)
    ]
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]


@pytest.mark.cpu_only
def test_sm90_explicit_geometry_bypasses_live_cache(monkeypatch):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    backend = create_mega_kernel(
        moe_ep.Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=256, top_k=2, swap_ab=True
        )
    )
    monkeypatch.setattr(backend, "_tuning_cache", mock.Mock(side_effect=AssertionError))
    fleet = moe_ep.FleetParams(2, 64, 128)
    assert backend.live_token_buckets(fleet) == []
    assert backend._workspace_knobs(fleet) is None


@pytest.mark.cpu_only
def test_duplicate_capacity_is_rejected_and_different_capacity_does_not_alias(
    monkeypatch,
):
    from flashinfer.moe_ep import MoEEpConfigError
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
        with pytest.raises(
            MoEEpConfigError,
            match="workspace profile.*already exists",
        ):
            layer.create_workspace(128)
        large = layer.create_workspace(256)

    assert first._backend_workspace is not large._backend_workspace
    assert [capacity for capacity, _ in allocated] == [128, 256]
    assert workspace_pool.pooled_workspace_refcount(first._backend_workspace) == 1
    assert workspace_pool.pooled_workspace_refcount(large._backend_workspace) == 1

    first_raw = first._backend_workspace
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
        forget_local.assert_called_once_with(first_raw)
        forget_global.assert_called_once_with(first_raw)
        first_raw.destroy.assert_called_once()
        large.destroy()
        assert forget_local.call_count == 2
        assert forget_global.call_count == 2
        forget_global.assert_called_with(large_raw)
        large_raw.destroy.assert_called_once()
    assert workspace_pool.pooled_workspace_count() == 0
    layer.destroy()


@pytest.mark.cpu_only
@pytest.mark.parametrize("live_tokens", [None, 8])
def test_default_profile_handle_adopts_the_lazy_default_workspace(live_tokens):
    layer, _, _ = _make_layer()
    layer._kernel.supports_live_token_tuning = True
    layer._fleet_params = dataclasses.replace(
        layer._fleet_params, tuning_live_tokens=live_tokens
    )
    default_workspace = mock.MagicMock(name="default_workspace")
    layer._workspace = default_workspace
    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(layer._kernel, "prepare_workspace") as prepare_workspace,
    ):
        handle = layer.create_workspace(
            layer._fleet_params.max_tokens_per_rank, live_tokens=live_tokens
        )

    prepare_workspace.assert_not_called()
    assert handle._backend_workspace is default_workspace
    assert layer._workspace is None
    assert layer._default_workspace_handle is handle
    assert layer._ensure_workspace() is default_workspace

    with mock.patch.object(layer._kernel, "destroy") as destroy:
        handle.destroy()
    destroy.assert_called_once_with(default_workspace)
    assert layer._default_workspace_handle is None
    layer.destroy()


@pytest.mark.cpu_only
@pytest.mark.parametrize("initialize_default_first", [False, True])
def test_full_capacity_handle_preserves_the_default_live_profile(
    initialize_default_first,
):
    layer, _, _ = _make_layer()
    layer._fleet_params = dataclasses.replace(layer._fleet_params, tuning_live_tokens=8)
    default_workspace, full_workspace = object(), object()
    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(
            layer._kernel,
            "prepare_workspace",
            side_effect=lambda bootstrap, fleet: (
                default_workspace if fleet.tuning_live_tokens == 8 else full_workspace
            ),
        ),
        mock.patch.object(layer._kernel, "destroy"),
    ):
        if initialize_default_first:
            assert layer._ensure_workspace() is default_workspace
        handle = layer.create_workspace(layer._fleet_params.max_tokens_per_rank)
        assert handle.live_tokens is None
        assert handle._backend_workspace is full_workspace
        assert layer._default_workspace_handle is None
        assert layer._ensure_workspace() is default_workspace
        layer.destroy()


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_unfinalized_output_requires_explicit_backend_opt_in():
    from flashinfer.moe_ep import MoEEpConfigError

    layer, _, _ = _make_layer()
    with pytest.raises(
        MoEEpConfigError, match="requires finalize_mode=FinalizeMode.NONE"
    ):
        layer.forward_unfinalized(_inputs(8))
    layer._kernel.supports_unfinalized_output = True
    with pytest.raises(MoEEpConfigError, match="requires forward_unfinalized"):
        layer.forward(_inputs(8))
    layer.destroy()


@pytest.mark.cpu_only
@pytest.mark.parametrize("apply_routing_weights_before_fc2", [False, True])
@pytest.mark.parametrize("backend", ["nvfp4", "bf16", "w4a16"])
def test_unfinalized_output_borrows_route_metadata_and_launches_empty_batches(
    monkeypatch, backend, apply_routing_weights_before_fc2
):
    """Remaining weights and identity indices reuse storage across live sizes."""
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.cute_dsl.megamoe import bf16_nvfp4
    from flashinfer.moe_ep.kernel_src.sm100 import cutedsl_megamoe

    capacity, top_k, hidden = 7, 2, 128
    partials = torch.randn(capacity, top_k, hidden, dtype=torch.bfloat16)
    raw = SimpleNamespace(
        x=torch.empty(capacity, hidden, dtype=torch.bfloat16),
        hidden=hidden,
        output_activation=None,
        topk_idx=torch.empty(capacity, top_k, dtype=torch.int32),
        topk_weights=torch.randn(capacity, top_k, dtype=torch.float32),
        _frontend=mock.Mock(),
        _staging_inputs=None,
    )
    raw._frontend.config.apply_routing_weights_before_fc2 = (
        apply_routing_weights_before_fc2
    )
    options = {"apply_routing_weights_before_fc2": apply_routing_weights_before_fc2}
    allocate = mock.Mock(return_value=raw)
    if backend == "nvfp4":
        config_cls = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        monkeypatch.setattr(cutedsl_megamoe, "get_symm_buffer_for_mega_moe", allocate)
        raw._frontend.custom_finalize_workspace.return_value = (
            partials,
            partials,
            None,
        )
        launch = mock.Mock()
    elif backend == "bf16":
        config_cls = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig
        monkeypatch.setattr(
            cutedsl_megamoe, "get_symm_buffer_for_bf16_mega_moe", allocate
        )
        launch = raw._frontend.run
        launch.side_effect = lambda inputs, *, num_tokens: None
        raw._frontend.custom_finalize_workspace.return_value = (
            partials,
            partials,
            None,
        )
    else:
        config_cls = Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        monkeypatch.setattr(
            bf16_nvfp4, "get_symm_buffer_for_bf16_nvfp4_mega_moe", allocate
        )
        launch = mock.Mock(
            side_effect=lambda *a, num_tokens, **k: partials[:num_tokens]
        )
        monkeypatch.setattr(bf16_nvfp4, "bf16_nvfp4_mega_moe", launch)
    kernel = create_mega_kernel(
        config_cls(
            intermediate_size=128,
            top_k=top_k,
            finalize_mode=FinalizeMode.NONE,
            **options,
        )
    )
    kernel.bind_ep_bootstrap(
        BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    workspace = kernel._allocate_workspace(FleetParams(2, capacity, hidden))
    assert workspace is raw
    if backend != "bf16":
        assert allocate.call_args.kwargs["use_custom_finalize"]
    assert kernel.supports_unfinalized_output
    with pytest.raises(ValueError, match="requires compute_unfinalized"):
        kernel.compute(workspace, (), output=None)
    monkeypatch.setattr(cutedsl_megamoe, "staged_tokens", lambda _: None)
    with pytest.raises(ValueError, match="requires stage_inputs"):
        kernel.compute_unfinalized(workspace, ())
    weights_storage = raw._unfinalized_weights.untyped_storage().data_ptr()
    map_storage = raw._unfinalized_route_map.untyped_storage().data_ptr()
    if backend == "nvfp4":
        monkeypatch.setattr(
            kernel, "_prepared_thunk_state", lambda *args: ((), launch, None)
        )
    monkeypatch.setattr(torch, "arange", mock.Mock(side_effect=AssertionError))
    monkeypatch.setattr(torch, "ones_like", mock.Mock(side_effect=AssertionError))

    for live_tokens in (3, 0, 5):
        monkeypatch.setattr(cutedsl_megamoe, "staged_tokens", lambda _: live_tokens)
        raw._staging_inputs = (
            raw.x[:live_tokens],
            raw.topk_idx[:live_tokens],
            raw.topk_weights[:live_tokens],
        )
        rows, weights, indices = kernel.compute_unfinalized(
            workspace, ((None,), (None,))
        )
        assert rows.shape == (live_tokens * top_k, hidden)
        assert rows.dtype == torch.bfloat16 and rows.is_contiguous()
        assert weights.shape == indices.shape == (live_tokens, top_k)
        assert weights.dtype == torch.float32 and indices.dtype == torch.int32
        assert (
            rows.untyped_storage().data_ptr() == partials.untyped_storage().data_ptr()
        )
        assert weights.untyped_storage().data_ptr() == weights_storage
        assert indices.untyped_storage().data_ptr() == map_storage
        assert indices.flatten().tolist() == list(range(live_tokens * top_k))
        expected_weights = (
            torch.full((live_tokens, top_k), 1.0)
            if apply_routing_weights_before_fc2
            else raw.topk_weights[:live_tokens]
        )
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(rows[indices.long()], partials[:live_tokens])
        if backend == "w4a16":
            assert raw._staging_inputs is None
    assert launch.call_count == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "knobs",
    [
        {"apply_routing_weights_before_fc2": False},
        {"use_custom_finalize": False},
        {"use_persistent_finalize_kernel": True},
    ],
)
def test_nvfp4_knobs_reject_explicit_config_fields(monkeypatch, knobs):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim import nvfp4

    allocate = mock.Mock(
        side_effect=AssertionError("allocation before knob validation")
    )
    monkeypatch.setattr(nvfp4, "_sym_zeros_byte_view", allocate)
    kernel = create_mega_kernel(
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            finalize_mode=FinalizeMode.NONE,
            knobs=knobs,
        )
    )
    kernel.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    with pytest.raises(ValueError, match="explicit config fields"):
        kernel._allocate_workspace(moe_ep.FleetParams(2, 1, 128))
    allocate.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "config_name",
    [
        "Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
        "Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig",
        "Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
    ],
)
def test_finalized_and_unfinalized_workspaces_have_distinct_pool_keys(
    monkeypatch, config_name
):
    import torch

    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        "flashinfer.jit.cake_megamoe_topk_reduce.supports_device", lambda: False
    )
    config = getattr(moe_ep, config_name)(intermediate_size=128, top_k=2)
    finalized = create_mega_kernel(config)
    unfinalized = create_mega_kernel(
        dataclasses.replace(config, finalize_mode=FinalizeMode.NONE)
    )
    bootstrap = moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    finalized.bind_ep_bootstrap(bootstrap)
    unfinalized.bind_ep_bootstrap(bootstrap)
    finalized._ep_comm_group = unfinalized._ep_comm_group = object()
    fleet = moe_ep.FleetParams(2, 7, 128)
    assert finalized._workspace_pool_key(fleet) != unfinalized._workspace_pool_key(
        fleet
    )
    with pytest.raises(ValueError, match="requires finalize_mode=NONE"):
        finalized.compute_unfinalized(None, ())


@pytest.mark.cpu_only
def test_w4a16_unfinalized_launch_failure_releases_staging_sources(monkeypatch):
    import torch

    from flashinfer.moe_ep import Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.cute_dsl.megamoe import bf16_nvfp4

    kernel = create_mega_kernel(
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128, top_k=2, finalize_mode=FinalizeMode.NONE
        )
    )
    workspace = SimpleNamespace(_staging_inputs=(torch.empty(3, 128),))
    monkeypatch.setattr(
        bf16_nvfp4,
        "bf16_nvfp4_mega_moe",
        mock.Mock(side_effect=RuntimeError("launch failed")),
    )
    with pytest.raises(RuntimeError, match="launch failed"):
        kernel.compute_unfinalized(workspace, ((), ()))
    assert workspace._staging_inputs is None


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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
        handle.destroy()
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_layer_destroy_releases_profiles_in_reverse_creation_order():
    layer, _, _ = _make_layer()
    first_raw = mock.MagicMock(name="first_raw")
    second_raw = mock.MagicMock(name="second_raw")
    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(
            layer._kernel,
            "prepare_workspace",
            side_effect=[first_raw, second_raw],
        ),
        mock.patch.object(layer._kernel, "destroy") as destroy,
    ):
        first = layer.create_workspace(128)
        second = layer.create_workspace(256)
        layer.destroy()

    assert first.is_destroyed
    assert second.is_destroyed
    assert [call.args[0] for call in destroy.call_args_list] == [
        second_raw,
        first_raw,
    ]


@pytest.mark.cpu_only
def test_workspace_handle_is_not_a_context_manager():
    layer, _, _ = _make_layer()
    handle, _, _, _ = _create_workspace(layer, 128)

    assert not hasattr(handle, "__enter__")
    assert not hasattr(handle, "__exit__")

    handle.destroy()
    layer.destroy()


@pytest.mark.cpu_only
def test_workspace_destroy_fails_loudly_if_owning_layer_is_gone():
    from flashinfer.moe_ep import MoEEpConfigError

    layer, _, _ = _make_layer()
    handle, raw_workspace, _, _ = _create_workspace(layer, 128)
    handle._layer_ref = lambda: None

    with pytest.raises(MoEEpConfigError, match="owning MegaMoE layer no longer exists"):
        handle.destroy()

    assert not handle.is_destroyed
    assert handle._backend_workspace is raw_workspace
    layer.destroy()


@pytest.mark.cpu_only
def test_layer_finalizer_does_not_run_collective_cleanup():
    layer, _, _ = _make_layer()
    workspace = mock.MagicMock(name="default_workspace")
    layer._workspace = workspace

    with (
        mock.patch.object(layer._kernel, "destroy") as destroy,
        pytest.warns(ResourceWarning, match="collective resources"),
    ):
        layer.__del__()

    destroy.assert_not_called()
    assert layer._workspace is workspace
    layer.destroy()


@pytest.mark.cpu_only
def test_forward_rejects_unregistered_workspace_handle():
    from flashinfer.moe_ep import MoEEpConfigError, MoEEpMegaWorkspace

    layer, _, _ = _make_layer()
    fleet_params = dataclasses.replace(layer._fleet_params, max_tokens_per_rank=128)
    forged_raw = mock.MagicMock()
    forged = MoEEpMegaWorkspace(layer, fleet_params, forged_raw)

    with (
        mock.patch.object(layer._kernel, "stage_inputs") as stage_inputs,
        mock.patch.object(layer._kernel, "compute") as compute,
        pytest.raises(MoEEpConfigError, match="not created by this layer"),
    ):
        layer.forward(_inputs(8), workspace=forged)

    stage_inputs.assert_not_called()
    compute.assert_not_called()
    with pytest.raises(MoEEpConfigError, match="not created by this layer"):
        forged.destroy()
    forged_raw.destroy.assert_not_called()
    layer.destroy()


@pytest.mark.cpu_only
def test_explicit_workspace_preserves_owned_output_by_default():
    layer, _, _ = _make_layer()
    handle, raw_workspace, _, _ = _create_workspace(layer, 128)
    tensors = _inputs(8)
    borrowed = mock.MagicMock(name="borrowed")
    layer._kernel.supports_output_view = True

    with (
        mock.patch.object(layer._kernel, "validate_forward"),
        mock.patch.object(layer._kernel, "stage_inputs"),
        mock.patch.object(
            layer._kernel,
            "compute",
            side_effect=lambda workspace, weights, *, output: (
                borrowed if output is None else output
            ),
        ) as compute,
    ):
        owned = layer.forward(tensors, workspace=handle)
        view = layer.forward(
            tensors,
            workspace=handle,
            return_workspace_view=True,
        )

    assert owned.shape == (8, 128)
    assert compute.call_args_list[0].kwargs["output"] is owned
    assert compute.call_args_list[1].kwargs["output"] is None
    assert view is borrowed
    assert all(call.args[0] is raw_workspace for call in compute.call_args_list)

    handle.destroy()
    layer.destroy()


@pytest.mark.cpu_only
def test_creating_workspaces_does_not_repeat_weight_preprocessing():
    layer, transformed, preprocess_weights = _make_layer(preprocess_weights=True)
    assert layer._preprocessing_count == 1
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

    assert layer._transformed is transformed
    assert layer._preprocessing_count == 1
    preprocess_weights.assert_called_once()
    small.destroy()
    large.destroy()
    layer.destroy()


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("supports_live_profiles", [False, True])
def test_auto_handles_and_default_replacement_retain_old_storage(
    supports_live_profiles,
):
    from flashinfer.moe_ep import MoEEpConfigError

    layer, transformed, _ = _make_layer()
    layer._megakernel_config = SimpleNamespace(knobs="auto")
    layer._kernel.supports_auto_workspaces = True
    layer._kernel.supports_live_token_tuning = supports_live_profiles
    if not supports_live_profiles:
        with pytest.raises(MoEEpConfigError, match="live-token tuning"):
            layer.create_workspace(64, live_tokens=8, knobs={})
    first_raw = object()
    with (
        mock.patch.object(layer._kernel, "validate_init"),
        mock.patch.object(
            layer._kernel, "prepare_workspace", return_value=first_raw
        ) as allocate,
        pytest.warns(UserWarning, match="live_tokens is ignored for automatic tuning"),
    ):
        first = layer.create_workspace(64, live_tokens=8)
    allocate.assert_called_once()
    assert first.live_tokens == 8
    second, second_raw, _, _ = _create_workspace(layer, 64)
    assert first_raw is not second_raw
    assert layer._resolve_workspace(first)[1] is first_raw
    assert layer._resolve_workspace(second)[1] is second_raw
    layer._bootstrap_validated = True
    replacement = object()
    with (
        mock.patch.object(layer._kernel, "validate_forward"),
        mock.patch.object(
            layer._kernel, "coordinate_workspace_replacement", return_value=True
        ),
        mock.patch.object(layer._kernel, "prepare_workspace", return_value=replacement),
        mock.patch.object(layer._kernel, "destroy") as destroy,
    ):
        _, selected, _, _, weights = layer._prepare_inputs(_inputs(8))
        assert selected is replacement and weights is transformed
        assert first._backend_workspace is first_raw and not first.is_destroyed
        assert second._backend_workspace is second_raw and not second.is_destroyed
        destroy.assert_not_called()
        first.destroy()
        destroy.assert_called_once_with(first_raw)
        assert layer._ensure_workspace() is replacement
        layer.destroy()
        assert destroy.call_count == 3


@pytest.mark.cpu_only
@pytest.mark.parametrize("dummy_warmup", [False, True])
def test_default_replacement_keeps_implicit_workspace_until_layer_destroy(dummy_warmup):
    import torch

    layer, _, _ = _make_layer()
    layer._bootstrap_validated = True
    original, replacement = object(), object()
    layer._workspace = original
    with (
        mock.patch.object(layer._kernel, "validate_forward"),
        mock.patch.object(
            layer._kernel, "coordinate_workspace_replacement", return_value=True
        ),
        mock.patch.object(layer._kernel, "prepare_workspace", return_value=replacement),
        mock.patch.object(layer._kernel, "destroy") as destroy,
    ):
        if dummy_warmup:
            with (
                mock.patch("torch.cuda.current_device", return_value=0),
                mock.patch("torch.device", return_value=torch.device("cpu")),
                mock.patch("torch.cuda.synchronize"),
                mock.patch.object(
                    layer,
                    "forward",
                    side_effect=lambda t, **kwargs: layer._prepare_inputs(t),
                ),
            ):
                layer.warmup()
            assert id(replacement) not in layer._forward_signatures
        else:
            assert layer._prepare_inputs(_inputs(8))[1] is replacement
        assert layer._workspace is replacement
        assert layer._retired_tuning_workspaces == [original]
        destroy.assert_not_called()
        layer.destroy()
        assert destroy.call_args_list == [mock.call(replacement), mock.call(original)]


@pytest.mark.cpu_only
def test_nvfp4_recompiled_workspace_releases_obsolete_launch_scratch(monkeypatch):
    """Recompiling one profile releases its old scratch and preserves its peers."""
    import weakref

    import torch

    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.backend import (
        Nvfp4CutedslMegaKernelBackend,
    )

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=1)
    )
    kernel = Nvfp4CutedslMegaKernelBackend(
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    monkeypatch.setattr(kernel, "_mega_inputs", lambda workspace, weights: workspace)

    def session(value):
        return SimpleNamespace(compiled=object(), scratch=torch.full((1, 128), value))

    def workspace(value):
        frontend = SimpleNamespace(
            _mega=session(value),
            config=SimpleNamespace(use_custom_finalize=False),
            set_swiglu_params=lambda *args: None,
        )
        raw = SimpleNamespace(_frontend=frontend, output_activation=torch.empty(1, 128))

        def prepare(inputs):
            mega = frontend._mega
            return lambda: inputs.output_activation.copy_(mega.scratch)

        frontend.make_launch_thunk = prepare
        return raw

    first, second = workspace(1.0), workspace(2.0)
    output = torch.empty(1, 128)
    kernel.compute(first, (), output=output)
    old_scratch = weakref.ref(first._frontend._mega.scratch)
    kernel.compute(second, (), output=output)

    first._frontend._mega = session(3.0)
    assert old_scratch() is not None  # The old launch still owns its scratch.
    kernel.compute(first, (), output=output)
    assert torch.equal(output, torch.full_like(output, 3.0))
    assert old_scratch() is None

    # The other profile's prepared launch remains valid and cached.
    second._frontend.make_launch_thunk = mock.Mock(side_effect=AssertionError)
    kernel.compute(second, (), output=output)
    assert torch.equal(output, torch.full_like(output, 2.0))
