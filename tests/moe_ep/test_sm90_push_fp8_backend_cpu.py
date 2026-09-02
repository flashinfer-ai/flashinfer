"""CPU-only contract tests for the SM90 push FP8 mega backend."""

import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest


def test_sm90_push_backend_import_defers_kernel_package():
    code = textwrap.dedent(
        """
        import importlib
        import sys
        import typing

        kernel_name = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
        backend_package = importlib.import_module(
            "flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda"
        )
        weights_module = importlib.import_module(
            "flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.weights"
        )
        assert kernel_name not in sys.modules
        hints = typing.get_type_hints(weights_module.preprocess_mega_weights)
        assert hints["return"] is typing.Any
        assert kernel_name not in sys.modules
        transformed_type = backend_package.TransformedMegaWeights
        kernel_package = importlib.import_module(kernel_name)
        assert transformed_type is kernel_package.Sm90PushWeights
        """
    )
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_sm90_push_timeout_requires_supported_setter():
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.backend import (
        _set_process_group_timeout,
    )

    with (
        mock.patch.object(dist, "set_timeout", None, create=True),
        mock.patch.object(dist, "distributed_c10d", SimpleNamespace(), create=True),
        pytest.raises(RuntimeError, match="exposes neither set_timeout"),
    ):
        _set_process_group_timeout(object(), 1.0)


def test_sm90_push_unfused_intermediate_size_limit():
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda import (
        backend as backend_module,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.backend import (
        Sm90PushFp8MegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.config import (
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig(
            intermediate_size=16384 + 128,
            top_k=2,
            fuse_fc1_epilogue=False,
        )
    )
    bootstrap = BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    fleet = FleetParams(
        num_experts=1,
        max_tokens_per_rank=8,
        token_hidden_size=128,
    )

    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        pytest.raises(
            MoEEpConfigError,
            match=r"silu_mul_quant.*<= 16384.*fuse_fc1_epilogue=True",
        ),
    ):
        backend.validate_init(bootstrap, fleet)


def test_sm90_push_fp8_config_defaults_to_unfused_fc1():
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.config import (
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig,
    )

    config = Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig(intermediate_size=128, top_k=2)

    assert config.fuse_fc1_epilogue is False


def test_sm90_push_staging_binds_layer_weights_and_records_lease():
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.backend import (
        Sm90PushFp8MegaKernelBackend,
        _Sm90PushFp8Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.config import (
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock()
    workspace = _Sm90PushFp8Workspace(
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
    backend.stage_inputs(
        inputs,
        workspace,
        quantize_input=True,
    )
    runner.bind_weights.assert_called_once_with(transformed)
    runner.stage_inputs.assert_called_once_with(
        inputs.hidden_states,
        inputs.topk_ids,
        inputs.topk_weights,
    )
    assert workspace.active_weights is transformed
    assert workspace.staged_weights is transformed
    assert workspace.staged_tokens == 3


def test_sm90_push_compute_finishes_round_before_rejecting_different_weights():
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.backend import (
        Sm90PushFp8MegaKernelBackend,
        _Sm90PushFp8Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.config import (
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    output = object()
    runner = mock.Mock(state="idle")
    runner.compute.return_value = output
    workspace = _Sm90PushFp8Workspace(
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
def test_sm90_push_compute_mirrors_runner_poison_state(runner_state, expected_poisoned):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.backend import (
        Sm90PushFp8MegaKernelBackend,
        _Sm90PushFp8Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.config import (
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock(state=runner_state)
    runner.compute.side_effect = RuntimeError("compute failed")
    workspace = _Sm90PushFp8Workspace(
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


def test_sm90_push_workspace_pool_key_covers_construction_state():
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.backend import (
        Sm90PushFp8MegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.config import (
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig,
    )

    config = Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig(intermediate_size=256, top_k=2)
    fleet = SimpleNamespace(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=512,
    )
    group = object()

    def make_backend(candidate, *, process_group=group, rank=1, world_size=2):
        backend = Sm90PushFp8MegaKernelBackend(candidate)
        backend._ep_bootstrap = object()
        backend._ep_rank = rank
        backend._ep_world_size = world_size
        backend._ep_comm_group = process_group
        return backend

    with mock.patch("torch.cuda.current_device", return_value=3):
        baseline = make_backend(config)._workspace_pool_key(fleet)
        assert make_backend(config)._workspace_pool_key(fleet) == baseline
        variants = (
            replace(config, intermediate_size=384),
            replace(config, top_k=4),
            replace(config, capacity_factor=0.5),
            replace(config, dedup_dispatch=False),
            replace(config, grouped_combine=False),
            replace(config, fuse_fc1_epilogue=True),
            replace(config, payload_dtype="bf16"),
            replace(config, combine_dtype="bf16", grouped_combine=False),
            replace(config, allow_unverified_p2p=True),
            replace(config, init_timeout_s=30.0),
        )
        assert all(
            make_backend(value)._workspace_pool_key(fleet) != baseline
            for value in variants
        )
        assert (
            make_backend(config, process_group=object())._workspace_pool_key(fleet)
            != baseline
        )
        assert make_backend(config, rank=0)._workspace_pool_key(fleet) != baseline
        assert make_backend(config, world_size=4)._workspace_pool_key(fleet) != baseline
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
            assert make_backend(config)._workspace_pool_key(changed_fleet) != baseline
    with mock.patch("torch.cuda.current_device", return_value=4):
        assert make_backend(config)._workspace_pool_key(fleet) != baseline


def test_sm90_push_destroy_uses_workspace_pool_refcount(monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.backend import (
        Sm90PushFp8MegaKernelBackend,
        _Sm90PushFp8Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda.config import (
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel import workspace_pool

    monkeypatch.setattr(workspace_pool, "_POOL", {})
    monkeypatch.setattr(workspace_pool, "_KEY_BY_ID", {})
    runner = mock.Mock()
    workspace = _Sm90PushFp8Workspace(pipe=object(), runner=runner)
    first = workspace_pool.acquire_workspace(
        ("sm90_fp8_fp8_bf16_push_cuda",), lambda: workspace
    )
    second = workspace_pool.acquire_workspace(
        ("sm90_fp8_fp8_bf16_push_cuda",), lambda: workspace
    )
    backend = Sm90PushFp8MegaKernelBackend(
        Sm90_Fp8_Fp8_Bf16_PushCuda_MegaMoeConfig(intermediate_size=128, top_k=2)
    )

    backend.destroy(first)
    runner.destroy.assert_not_called()
    assert not workspace.destroyed

    backend.destroy(second)
    runner.destroy.assert_called_once_with()
    assert workspace.destroyed
    assert workspace.active_weights is None
    assert workspace.staged_weights is None


def test_sm90_push_destroy_quiesces_locally_before_releasing_views(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.protocol import (
        Sm90PushPipe,
    )

    calls = []

    pipe = object.__new__(Sm90PushPipe)
    pipe._destroyed = False
    pipe.device = object()
    pipe.symm = object()
    pipe._release_window_views = lambda: calls.append(("release", None))
    monkeypatch.setattr(
        "torch.cuda.synchronize", lambda device: calls.append(("sync", device))
    )

    pipe.destroy()

    assert [name for name, _ in calls] == ["sync", "release"]
    assert pipe.symm is None
    assert pipe._destroyed


def test_sm90_push_destroy_preserves_peer_views_after_local_sync_failure(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.protocol import (
        Sm90PushPipe,
    )

    pipe = object.__new__(Sm90PushPipe)
    pipe._destroyed = False
    pipe.device = object()
    pipe.symm = object()
    released = []
    pipe._release_window_views = lambda: released.append(True)

    def _fail_sync(_device):
        raise RuntimeError("sync failed")

    monkeypatch.setattr("torch.cuda.synchronize", _fail_sync)

    with pytest.raises(RuntimeError, match="sync failed"):
        pipe.destroy()

    assert not released
    assert pipe.symm is not None
    assert not pipe._destroyed
