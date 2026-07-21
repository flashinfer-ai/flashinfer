"""CPU-only contract tests for the SM90 push FP8 mega backend."""

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest import mock

import pytest


def test_sm90_push_backend_import_defers_kernel_package():
    code = textwrap.dedent(
        """
        import importlib
        import sys
        import typing

        kernel_name = "flashinfer.moe_ep.kernel_src.sm90_push_megamoe"
        backend_package = importlib.import_module(
            "flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8"
        )
        weights_module = importlib.import_module(
            "flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.weights"
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
    )
    assert result.returncode == 0, result.stderr


def test_torch_dist_timeout_requires_supported_setter():
    try:
        from flashinfer.comm.mnnvl import TorchDistBackend
    except ImportError as exc:
        pytest.skip(f"requires cuda-python to import mnnvl: {exc}")

    backend = object.__new__(TorchDistBackend)
    backend._dist = SimpleNamespace(distributed_c10d=SimpleNamespace())
    backend._group = object()

    with pytest.raises(RuntimeError, match="exposes neither set_timeout"):
        backend.set_timeout(1.0)


def test_sm90_push_unfused_intermediate_size_limit():
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8 import (
        backend as backend_module,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.backend import (
        Sm90PushFp8MegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.config import (
        Sm90PushFp8MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90PushFp8MegaMoeConfig(
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
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.config import (
        Sm90PushFp8MegaMoeConfig,
    )

    config = Sm90PushFp8MegaMoeConfig(intermediate_size=128, top_k=2)

    assert config.fuse_fc1_epilogue is False


def test_sm90_push_staging_validates_context_before_runner():
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.backend import (
        Sm90PushFp8MegaKernelBackend,
        _Sm90PushFp8Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.config import (
        Sm90PushFp8MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90PushFp8MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock()
    workspace = _Sm90PushFp8Workspace(
        pipe=object(),
        runner=runner,
        transformed_weights=transformed,
    )
    inputs = SimpleNamespace(
        hidden_states=object(),
        topk_ids=object(),
        topk_weights=object(),
        num_tokens=3,
    )
    output = object()

    backend._transformed_weights = object()
    with pytest.raises(RuntimeError, match="workspace bundle"):
        backend.stage_inputs(
            inputs,
            workspace,
            quantize_input=True,
            output=output,
        )
    runner.stage_inputs.assert_not_called()
    backend._transformed_weights = transformed

    with pytest.raises(MoEEpConfigError, match="destination output"):
        backend.stage_inputs(
            inputs,
            workspace,
            quantize_input=True,
            output=None,
        )
    runner.stage_inputs.assert_not_called()

    backend.stage_inputs(
        inputs,
        workspace,
        quantize_input=True,
        output=output,
    )
    runner.stage_inputs.assert_called_once_with(
        inputs.hidden_states,
        inputs.topk_ids,
        inputs.topk_weights,
        output=output,
    )
    assert workspace.staged_tokens == 3


def test_sm90_push_compute_finishes_round_before_rejecting_different_weights():
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.backend import (
        Sm90PushFp8MegaKernelBackend,
        _Sm90PushFp8Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.config import (
        Sm90PushFp8MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90PushFp8MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    output = object()
    runner = mock.Mock(state="idle")
    runner.compute.return_value = output
    workspace = _Sm90PushFp8Workspace(
        pipe=object(),
        runner=runner,
        transformed_weights=transformed,
        staged_tokens=3,
    )

    with pytest.raises(RuntimeError, match="different weight bundle"):
        backend.compute(workspace, object(), output=output)

    runner.compute.assert_called_once_with(output=output)
    runner.abort.assert_not_called()
    assert workspace.staged_tokens is None
    assert workspace.poisoned is False


@pytest.mark.parametrize(
    ("runner_state", "expected_poisoned"),
    [("idle", False), ("poisoned", True)],
)
def test_sm90_push_compute_mirrors_runner_poison_state(runner_state, expected_poisoned):
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.backend import (
        Sm90PushFp8MegaKernelBackend,
        _Sm90PushFp8Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.config import (
        Sm90PushFp8MegaMoeConfig,
    )

    backend = Sm90PushFp8MegaKernelBackend(
        Sm90PushFp8MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock(state=runner_state)
    runner.compute.side_effect = RuntimeError("compute failed")
    workspace = _Sm90PushFp8Workspace(
        pipe=object(),
        runner=runner,
        transformed_weights=transformed,
        staged_tokens=3,
    )

    with pytest.raises(RuntimeError, match="compute failed"):
        backend.compute(workspace, transformed, output=object())

    assert workspace.poisoned is expected_poisoned
    assert workspace.staged_tokens is None


def test_sm90_push_destroy_quiesces_locally_before_releasing_views(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.protocol import (
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
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim.protocol import (
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
