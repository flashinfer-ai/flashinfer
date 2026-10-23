"""Automatic MegaMoE tuning results exposed without cache-file access."""

from __future__ import annotations

import importlib
import subprocess
import sys
from unittest import mock

import pytest
import torch

from .test_mega_workspace_api import _make_layer


@pytest.mark.parametrize(
    "config_name,package,dtype",
    [
        (
            "Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
            "kernel_src.sm100.cutedsl_megamoe",
            "nvfp4",
        ),
        (
            "Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig",
            "kernel_src.sm100.cutedsl_megamoe",
            "mxfp8",
        ),
        (
            "Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig",
            "kernel_src.sm100.cutedsl_megamoe",
            "bf16",
        ),
        (
            "Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig",
            "kernel_src.sm100.cutedsl_megamoe",
            "bf16_mxfp8",
        ),
        (
            "Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig",
            "cute_dsl.megamoe.bf16_nvfp4",
            "bf16_nvfp4",
        ),
        (
            "Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig",
            "kernel_src.sm90.pull_style_cutedsl_megakernel",
            "hopper_fp8",
        ),
        (
            "Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig",
            "kernel_src.sm90.pull_style_cutedsl_megakernel",
            "hopper_bf16",
        ),
        (
            "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig",
            "kernel_src.sm90.pull_style_cutedsl_megakernel",
            "hopper_mxfp4",
        ),
    ],
)
def test_backend_retains_successful_tuning_result(config_name, package, dtype):
    # SM90 and SM100 use conflicting top-level kernel module names.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pytest; "
            "from tests.moe_ep.test_mega_tuned_knobs import _check_tuning_result; "
            "_check_tuning_result(pytest.MonkeyPatch(), *sys.argv[1:])",
            config_name,
            package,
            dtype,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_tuning_result(monkeypatch, config_name, package, dtype):
    import flashinfer.moe_ep as moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    config = getattr(moe_ep, config_name)(intermediate_size=128, top_k=2, knobs="auto")
    backend = create_mega_kernel(config)
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    shim = importlib.import_module(f"flashinfer.moe_ep.{package}")
    winner = {"flag_batch": 4, "mma_tiler_mnk": [256, 128, 256]}
    tune = mock.Mock(return_value=winner)
    monkeypatch.setattr(shim, f"autotune_{dtype}_mega_moe", tune)
    monkeypatch.setattr(shim, f"{dtype}_mega_moe", mock.Mock())
    workspace = mock.MagicMock()
    workspace._frontend.config.use_custom_finalize = False
    output = torch.zeros(2, 128, dtype=torch.bfloat16)
    if hasattr(backend, "_prepared_thunk_state"):
        monkeypatch.setattr(
            backend, "_prepared_thunk_state", lambda *args: ((), lambda: None, output)
        )

    backend.compute(workspace, (None, None), output=output)
    assert backend.get_tuned_knobs() == winner


def test_layer_getter_returns_independent_result_and_checks_lifecycle():
    from flashinfer.moe_ep import MoEEpConfigError

    layer, _, _ = _make_layer()
    try:
        with pytest.raises(RuntimeError, match="no completed automatic tuning result"):
            layer.get_tuned_knobs()
        layer._kernel._autotune_winner = {"mma_tiler_mnk": [256, 128, 256]}
        result = layer.get_tuned_knobs()
        result["mma_tiler_mnk"][0] = 1
        assert layer.get_tuned_knobs() == {"mma_tiler_mnk": [256, 128, 256]}
        assert layer._workspace is None
    finally:
        layer.destroy()
    with pytest.raises(MoEEpConfigError, match="destroyed"):
        layer.get_tuned_knobs()
