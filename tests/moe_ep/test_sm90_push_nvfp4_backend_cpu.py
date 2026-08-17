"""CPU-only contracts for the SM90 push NVFP4 mega backend."""

import weakref
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest


def _make_backend(config, *, process_group=None, rank=1, world_size=2):
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.backend import (
        Sm90PushNvFp4MegaKernelBackend,
    )

    backend = Sm90PushNvFp4MegaKernelBackend(config)
    backend._ep_bootstrap = object()
    backend._ep_rank = rank
    backend._ep_world_size = world_size
    backend._ep_comm_group = process_group if process_group is not None else object()
    return backend


def _make_w4a8_bundle():
    from flashinfer.fused_moe.sm90_nvfp4_repack import NVFP4SM90WeightViewV3
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_weights import (
        Sm90PushNvFp4Weights,
    )

    w13 = object.__new__(NVFP4SM90WeightViewV3)
    w2 = object.__new__(NVFP4SM90WeightViewV3)
    object.__setattr__(w13, "manifest", SimpleNamespace(expert_mapping=(0, 1)))
    object.__setattr__(w2, "manifest", SimpleNamespace(expert_mapping=(0, 1)))
    bundle = object.__new__(Sm90PushNvFp4Weights)
    object.__setattr__(bundle, "nvfp4_mode", "w4a8")
    object.__setattr__(bundle, "w13", w13)
    object.__setattr__(bundle, "w2", w2)
    return bundle


def _make_w4a8_runner(weights):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_runner import (
        Sm90PushNvFp4MoERunner,
        _LegacyW4A8PairEngine,
    )
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.runner import (
        _RunnerState,
    )

    runner = object.__new__(Sm90PushNvFp4MoERunner)
    runner._state = _RunnerState.IDLE
    runner._bound_weights = weights
    runner._validated_weights = {id(weights): weakref.ref(weights)}
    runner.nvfp4_mode = "w4a8"
    runner.weights = weights
    engine = object.__new__(_LegacyW4A8PairEngine)
    engine.total_experts = 2
    engine.fc1 = mock.Mock()
    engine.fc2 = mock.Mock()
    runner._w4a8_engine = engine
    return runner


def test_nvfp4_production_rs_runner_uses_frozen_kernel_variant(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import (
        nvfp4_runner,
    )

    calls = []

    class _FakeRsRunner:
        def get_workspace_size(self, *_args):
            return 0

        def configure_workspace(self, _workspace):
            return None

    def _create(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeRsRunner()

    monkeypatch.setattr(
        nvfp4_runner,
        "create_sm90_push_nvfp4_rs_gemm_runner",
        _create,
    )
    runner = object.__new__(nvfp4_runner.Sm90PushNvFp4MoERunner)
    runner._rs_n_tactic = 64
    runner._rs_stages = 3
    runner._rs_stage_k = 64
    runner._padded_max_rows = 128
    runner.pipe = SimpleNamespace(E=2, device="cpu")

    runner._new_rs_runner(64, 128)

    assert calls == [(("rs_wgmma", 64, 3, 64), {"use_environment": False})]


def test_nvfp4_staging_rebinds_layer_weights_and_records_lease():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.backend import (
        Sm90PushNvFp4MegaKernelBackend,
        _Sm90PushNvFp4Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.config import (
        Sm90PushNvFp4MegaMoeConfig,
    )

    backend = Sm90PushNvFp4MegaKernelBackend(
        Sm90PushNvFp4MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock()
    workspace = _Sm90PushNvFp4Workspace(
        pipe=object(),
        runner=runner,
        active_weights=object(),
    )
    inputs = SimpleNamespace(
        hidden_states=object(),
        topk_ids=object(),
        topk_weights=object(),
        num_tokens=3,
    )

    backend.stage_inputs(inputs, workspace, quantize_input=True)

    runner.bind_weights.assert_called_once_with(transformed)
    runner.stage_inputs.assert_called_once_with(
        inputs.hidden_states,
        inputs.topk_ids,
        inputs.topk_weights,
    )
    assert workspace.active_weights is transformed
    assert workspace.staged_weights is transformed
    assert workspace.staged_tokens == 3


def test_nvfp4_compute_finishes_round_before_rejecting_different_weights():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.backend import (
        Sm90PushNvFp4MegaKernelBackend,
        _Sm90PushNvFp4Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.config import (
        Sm90PushNvFp4MegaMoeConfig,
    )

    backend = Sm90PushNvFp4MegaKernelBackend(
        Sm90PushNvFp4MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    output = object()
    runner = mock.Mock(state="idle")
    runner.compute.return_value = output
    workspace = _Sm90PushNvFp4Workspace(
        pipe=object(),
        runner=runner,
        active_weights=transformed,
        staged_weights=transformed,
        staged_tokens=3,
    )

    with pytest.raises(RuntimeError, match="different weight bundle"):
        backend.compute(workspace, object(), output=output)

    runner.compute.assert_called_once_with(output=output)
    runner.abort.assert_not_called()
    assert workspace.staged_tokens is None
    assert workspace.staged_weights is None
    assert workspace.poisoned is False


@pytest.mark.parametrize(
    ("runner_state", "expected_poisoned"),
    [("idle", False), ("poisoned", True)],
)
def test_nvfp4_compute_mirrors_runner_poison_state(runner_state, expected_poisoned):
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.backend import (
        Sm90PushNvFp4MegaKernelBackend,
        _Sm90PushNvFp4Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.config import (
        Sm90PushNvFp4MegaMoeConfig,
    )

    backend = Sm90PushNvFp4MegaKernelBackend(
        Sm90PushNvFp4MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock(state=runner_state)
    runner.compute.side_effect = RuntimeError("compute failed")
    workspace = _Sm90PushNvFp4Workspace(
        pipe=object(),
        runner=runner,
        active_weights=transformed,
        staged_weights=transformed,
        staged_tokens=3,
    )

    with pytest.raises(RuntimeError, match="compute failed"):
        backend.compute(workspace, transformed, output=object())

    assert workspace.poisoned is expected_poisoned
    assert workspace.staged_tokens is None
    assert workspace.staged_weights is None


def test_nvfp4_workspace_pool_key_covers_construction_state():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.config import (
        Sm90PushNvFp4MegaMoeConfig,
    )

    config = Sm90PushNvFp4MegaMoeConfig(intermediate_size=256, top_k=2)
    fleet = SimpleNamespace(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=512,
    )
    group = object()

    with mock.patch("torch.cuda.current_device", return_value=3):
        baseline = _make_backend(config, process_group=group)._workspace_pool_key(fleet)
        assert (
            _make_backend(config, process_group=group)._workspace_pool_key(fleet)
            == baseline
        )
        variants = (
            replace(config, intermediate_size=384),
            replace(config, top_k=4),
            replace(config, nvfp4_mode="w4a16_rs"),
            replace(config, group_size=64),
            replace(config, residual_scheme="pow2"),
            replace(config, capacity_factor=0.5),
            replace(config, dedup_dispatch=False),
            replace(config, grouped_combine=False),
            replace(config, fuse_act=False),
            replace(config, payload_dtype="bf16"),
            replace(config, combine_dtype="bf16", grouped_combine=False),
            replace(config, rs_n_tactic=96),
            replace(config, rs_stages=4),
            replace(config, rs_stage_k=128),
            replace(config, allow_unverified_p2p=True),
            replace(config, init_timeout_s=30.0),
            replace(config, tma_cache_capacity=1),
            replace(config, n64_expected_m_per_sm=2.0),
            replace(config, payload_layout=3, allow_legacy_layout=True),
            replace(config, allow_legacy_layout=True),
            replace(
                config,
                weight_policy="dual",
                acknowledge_dual_residency=True,
            ),
        )
        assert all(
            _make_backend(value, process_group=group)._workspace_pool_key(fleet)
            != baseline
            for value in variants
        )
        assert _make_backend(config)._workspace_pool_key(fleet) != baseline
        assert (
            _make_backend(config, process_group=group, rank=0)._workspace_pool_key(
                fleet
            )
            != baseline
        )
        assert (
            _make_backend(
                config, process_group=group, world_size=4
            )._workspace_pool_key(fleet)
            != baseline
        )
        for changed_fleet in (
            SimpleNamespace(
                num_experts=16,
                max_tokens_per_rank=64,
                token_hidden_size=512,
            ),
            SimpleNamespace(
                num_experts=8,
                max_tokens_per_rank=128,
                token_hidden_size=512,
            ),
            SimpleNamespace(
                num_experts=8,
                max_tokens_per_rank=64,
                token_hidden_size=1024,
            ),
        ):
            assert (
                _make_backend(config, process_group=group)._workspace_pool_key(
                    changed_fleet
                )
                != baseline
            )
    with mock.patch("torch.cuda.current_device", return_value=4):
        assert (
            _make_backend(config, process_group=group)._workspace_pool_key(fleet)
            != baseline
        )


def test_nvfp4_dual_policy_requires_explicit_residency_acknowledgement():
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        Sm90PushNvFp4MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        backend as backend_module,
    )

    fleet = SimpleNamespace(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=128,
    )
    bootstrap = SimpleNamespace(world_size=2, stream=0)
    config = Sm90PushNvFp4MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        weight_policy="dual",
    )
    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        mock.patch.object(backend_module, "validate_mega_fleet_params"),
    ):
        with pytest.raises(MoEEpConfigError, match="acknowledge_dual_residency"):
            _make_backend(config).validate_init(bootstrap, fleet)
        _make_backend(replace(config, acknowledge_dual_residency=True)).validate_init(
            bootstrap, fleet
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"tma_cache_capacity": 0}, "tma_cache_capacity"),
        ({"tma_cache_capacity": 129}, "tma_cache_capacity"),
        ({"n64_expected_m_per_sm": 0.0}, "n64_expected_m_per_sm"),
        ({"n64_expected_m_per_sm": float("nan")}, "n64_expected_m_per_sm"),
        ({"payload_layout": 2}, "payload_layout"),
        ({"allow_legacy_layout": 1}, "allow_legacy_layout"),
    ),
)
def test_nvfp4_rejects_invalid_w4a8_tuning_config(changes, message):
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        Sm90PushNvFp4MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        backend as backend_module,
    )

    fleet = SimpleNamespace(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=128,
    )
    bootstrap = SimpleNamespace(world_size=2, stream=0)
    config = replace(
        Sm90PushNvFp4MegaMoeConfig(intermediate_size=128, top_k=2),
        **changes,
    )
    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        mock.patch.object(backend_module, "validate_mega_fleet_params"),
        pytest.raises(MoEEpConfigError, match=message),
    ):
        _make_backend(config).validate_init(bootstrap, fleet)


def test_nvfp4_rs_ignores_the_w4a8_payload_layout_selector():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        Sm90PushNvFp4MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        backend as backend_module,
    )

    fleet = SimpleNamespace(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=128,
    )
    bootstrap = SimpleNamespace(world_size=2, stream=0)
    config = Sm90PushNvFp4MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        nvfp4_mode="w4a16_rs",
        payload_layout=4,
        combine_dtype="bf16",
        grouped_combine=False,
        fuse_act=False,
    )
    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        mock.patch.object(backend_module, "validate_mega_fleet_params"),
    ):
        _make_backend(config).validate_init(bootstrap, fleet)


def test_nvfp4_w4a8_legacy_layout_requires_explicit_opt_in():
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        Sm90PushNvFp4MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        backend as backend_module,
    )

    fleet = SimpleNamespace(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=128,
    )
    bootstrap = SimpleNamespace(world_size=2, stream=0)
    config = Sm90PushNvFp4MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        payload_layout=3,
    )
    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        mock.patch.object(backend_module, "validate_mega_fleet_params"),
        pytest.raises(MoEEpConfigError, match="allow_legacy_layout"),
    ):
        _make_backend(config).validate_init(bootstrap, fleet)

    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        mock.patch.object(backend_module, "validate_mega_fleet_params"),
    ):
        _make_backend(replace(config, allow_legacy_layout=True)).validate_init(
            bootstrap, fleet
        )


def test_nvfp4_residency_estimate_separates_policy_components():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4 import (
        estimate_residency,
    )

    arguments = dict(
        local_experts=8,
        hidden_size=256,
        intermediate_size=128,
        group_size=128,
        residual_scheme="generic",
    )
    packed = estimate_residency(policy="packed", **arguments)
    folded = estimate_residency(policy="folded", **arguments)
    hybrid = estimate_residency(
        policy="hot_folded",
        hot_expert_count=3,
        **arguments,
    )
    dual = estimate_residency(policy="dual", **arguments)

    assert packed.hot_experts == 0
    assert packed.folded_bytes == 0
    assert folded.hot_experts == 8
    assert folded.packed_bytes == 0
    assert hybrid.hot_experts == 3
    assert hybrid.packed_bytes == packed.packed_bytes * 5 // 8
    assert hybrid.folded_bytes == folded.folded_bytes * 3 // 8
    assert dual.packed_bytes == packed.packed_bytes
    assert dual.folded_bytes == folded.folded_bytes
    assert dual.total_bytes == packed.total_bytes + folded.total_bytes

    model_shape = dict(
        local_experts=8,
        hidden_size=7168,
        intermediate_size=2048,
        group_size=128,
        residual_scheme="generic",
    )
    model_packed = estimate_residency(policy="packed", **model_shape)
    model_hybrid = estimate_residency(
        policy="hot_folded",
        hot_expert_count=1,
        **model_shape,
    )
    incremental_mib = (model_hybrid.total_bytes - model_packed.total_bytes) / float(
        1 << 20
    )
    assert 11.8 < incremental_mib < 11.9


def test_nvfp4_destroy_uses_workspace_pool_refcount(monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.backend import (
        Sm90PushNvFp4MegaKernelBackend,
        _Sm90PushNvFp4Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4.config import (
        Sm90PushNvFp4MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel import workspace_pool

    monkeypatch.setattr(workspace_pool, "_POOL", {})
    monkeypatch.setattr(workspace_pool, "_KEY_BY_ID", {})
    runner = mock.Mock()
    workspace = _Sm90PushNvFp4Workspace(pipe=object(), runner=runner)
    first = workspace_pool.acquire_workspace(("sm90_push_nvfp4",), lambda: workspace)
    second = workspace_pool.acquire_workspace(("sm90_push_nvfp4",), lambda: workspace)
    backend = Sm90PushNvFp4MegaKernelBackend(
        Sm90PushNvFp4MegaMoeConfig(intermediate_size=128, top_k=2)
    )

    backend.destroy(first)
    runner.destroy.assert_not_called()
    assert not workspace.destroyed

    backend.destroy(second)
    runner.destroy.assert_called_once_with()
    assert workspace.destroyed
    assert workspace.active_weights is None
    assert workspace.staged_weights is None


def test_w4a8_bind_validates_both_views_before_mutating_either_layer():
    old_weights = _make_w4a8_bundle()
    new_weights = _make_w4a8_bundle()
    runner = _make_w4a8_runner(old_weights)
    events = []
    engine = runner._w4a8_engine
    engine.fc1._validate_weight_view.side_effect = lambda view: events.append(
        ("validate_fc1", view)
    )
    engine.fc2._validate_weight_view.side_effect = lambda view: events.append(
        ("validate_fc2", view)
    )
    engine.fc1._bind_weight_view.side_effect = lambda view: events.append(
        ("bind_fc1", view)
    )
    engine.fc2._bind_weight_view.side_effect = lambda view: events.append(
        ("bind_fc2", view)
    )

    runner.bind_weights(new_weights)

    assert events == [
        ("validate_fc1", new_weights.w13),
        ("validate_fc2", new_weights.w2),
        ("bind_fc1", new_weights.w13),
        ("bind_fc2", new_weights.w2),
    ]
    assert runner.weights is new_weights
    assert runner._bound_weights is new_weights


def test_w4a8_fc2_validation_failure_leaves_fc1_and_bundle_unchanged():
    old_weights = _make_w4a8_bundle()
    new_weights = _make_w4a8_bundle()
    runner = _make_w4a8_runner(old_weights)
    engine = runner._w4a8_engine
    engine.fc2._validate_weight_view.side_effect = ValueError("invalid FC2")

    with pytest.raises(ValueError, match="invalid FC2"):
        runner.bind_weights(new_weights)

    engine.fc1._validate_weight_view.assert_called_once_with(new_weights.w13)
    engine.fc1._bind_weight_view.assert_not_called()
    engine.fc2._bind_weight_view.assert_not_called()
    assert runner.weights is old_weights
    assert runner._bound_weights is old_weights
    assert id(new_weights) not in runner._validated_weights


def test_w4a8_bind_rejects_staged_runner_before_validation():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.runner import (
        _RunnerState,
    )

    old_weights = _make_w4a8_bundle()
    new_weights = _make_w4a8_bundle()
    runner = _make_w4a8_runner(old_weights)
    runner._state = _RunnerState.STAGED

    with pytest.raises(RuntimeError, match="only be rebound while idle"):
        runner.bind_weights(new_weights)

    runner._w4a8_engine.fc1._validate_weight_view.assert_not_called()
    runner._w4a8_engine.fc2._validate_weight_view.assert_not_called()
    assert runner.weights is old_weights
    assert runner._bound_weights is old_weights
