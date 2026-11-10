"""Host checks for Kimi staged destinations and epilogue batching."""

from unittest import mock

import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.backends.mega.kernel.sm100.kimi_k3.backend import (
    KimiK3MegaKernelBackend,
)

pytestmark = pytest.mark.cpu_only
HIDDEN = 3584


@pytest.fixture
def runtime(monkeypatch):
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("cuda.bindings.driver")
    if not all(hasattr(torch, name) for name in ("float4_e2m1fn_x2", "float8_e8m0fnu")):
        pytest.skip("Kimi runtime requires FP4 and E8M0 Torch dtypes")
    from flashinfer.moe_ep.kernel_src.sm100._kimi_k3_megamoe.shim import runtime

    monkeypatch.setattr(runtime, "_launch_counts", lambda *a: (32, 32, 0))
    return runtime


class LaunchSetupReached(Exception):
    pass


@pytest.fixture
def staged_layer(runtime, monkeypatch):
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(situ_beta=4.0, situ_linear_beta=25.0)
    workspace = object.__new__(runtime.KimiWorkspace)
    workspace._check_alive = mock.Mock()
    workspace._inputs = object()
    workspace._live = 2
    workspace.device = torch.device("cpu")
    workspace.fleet_params = moe_ep.FleetParams(896, 256, HIDDEN)
    workspace.output = torch.empty(2, HIDDEN, dtype=torch.bfloat16)
    layer = object.__new__(moe_ep.MoEEpMegaLayer)
    torch.nn.Module.__init__(layer)
    layer._staged_kernel = KimiK3MegaKernelBackend(config)
    layer._staged_workspace = workspace
    layer._staged_transformed = object()

    def stop_before_launch():
        raise LaunchSetupReached

    # Exercise the public staged path without reaching a CUDA launch.
    monkeypatch.setattr(torch.cuda, "current_stream", stop_before_launch)
    return layer


@pytest.mark.parametrize(
    "output",
    [
        torch.empty(2, 8, dtype=torch.bfloat16),
        torch.empty(1, HIDDEN, dtype=torch.bfloat16),
        torch.empty((), dtype=torch.bfloat16),
        torch.empty(2 * HIDDEN, dtype=torch.bfloat16),
        torch.empty(2, HIDDEN, dtype=torch.float32),
        torch.empty(2, HIDDEN, dtype=torch.bfloat16, device="meta"),
        torch.empty(2, 2 * HIDDEN, dtype=torch.bfloat16)[:, ::2],
        torch.empty(2 * HIDDEN + 1, dtype=torch.bfloat16)[1:].reshape(2, HIDDEN),
    ],
    ids=["width", "rows", "scalar", "rank", "dtype", "device", "stride", "alignment"],
)
def test_staged_compute_rejects_invalid_output(staged_layer, output):
    with pytest.raises(ValueError, match="Kimi output"):
        staged_layer.compute_staged(output=output)


@pytest.mark.parametrize("rows", [None, 0, 2, 4])
def test_staged_compute_accepts_valid_output(staged_layer, rows):
    if rows == 0:
        staged_layer._staged_workspace._live = 0
    output = None if rows is None else torch.empty(rows, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(LaunchSetupReached):
        staged_layer.compute_staged(output=output)


@pytest.mark.parametrize("caller_output", [False, True])
def test_forward_does_not_recheck_prepared_output(
    staged_layer, monkeypatch, caller_output
):
    layer = staged_layer
    workspace, backend = layer._staged_workspace, layer._staged_kernel
    layer._kernel = backend
    layer._fleet_params = workspace.fleet_params
    layer._destroyed = False
    tensors = moe_ep.MoEEpTensors(
        torch.empty(2, HIDDEN, dtype=torch.bfloat16),
        torch.zeros(2, 16, dtype=torch.int32),
        torch.ones(2, 16),
        output=torch.empty(2, HIDDEN, dtype=torch.bfloat16) if caller_output else None,
    )
    monkeypatch.setattr(
        layer,
        "_prepare_inputs",
        lambda *a, **kw: (
            workspace.fleet_params,
            workspace,
            backend,
            True,
            layer._staged_transformed,
        ),
    )
    monkeypatch.setattr(backend, "stage_inputs", mock.Mock())
    checked = mock.Mock(side_effect=AssertionError("output was checked again"))
    monkeypatch.setattr(workspace, "compute", checked)
    monkeypatch.setattr(workspace, "_compute", lambda weights, *, output: output)
    output = layer(tensors)
    assert output.shape == (2, HIDDEN) and output.dtype == torch.bfloat16
    if caller_output:
        assert output is tensors.output
    backend.stage_inputs.assert_called_once()
    checked.assert_not_called()


@pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
@pytest.mark.parametrize("batches", [(5, 1), (1, 5), (32, 1)])
def test_mx_rejects_oversized_async_batches(runtime, wire, batches):
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
        situ_beta=4.0,
        situ_linear_beta=25.0,
        quant_kind="mxfp4_mxfp8",
        combine_dtype=wire,
        knobs={"epi_flag_batches": batches, "schedule_policy": ("grouped", 32)},
    )
    with pytest.raises(ValueError, match="async epilogue flag batches.*at most 4"):
        runtime._kernel(config, moe_ep.FleetParams(896, 8192, HIDDEN), 8)


@pytest.mark.parametrize("kind,batches", [("nvfp4", (32, 32)), ("mxfp4_mxfp8", (4, 4))])
@pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
def test_flag_batches_preserve_each_tracker_limit(runtime, kind, batches, wire):
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
        situ_beta=4.0,
        situ_linear_beta=25.0,
        quant_kind=kind,
        combine_dtype=wire,
        knobs={"epi_flag_batches": batches},
    )
    kernel = runtime._kernel(config, moe_ep.FleetParams(896, 8192, HIDDEN), 8)
    assert kernel.epi_flag_batches == batches


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4_mxfp8"])
@pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
def test_default_profiles_pass_batch_validation(runtime, kind, wire):
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
        situ_beta=4.0, situ_linear_beta=25.0, quant_kind=kind, combine_dtype=wire
    )
    for capacity in runtime.PROD1_TOKEN_BUCKETS:
        if capacity * config.top_k * HIDDEN < 2**31:
            runtime._kernel(config, moe_ep.FleetParams(896, capacity, HIDDEN), 8)
