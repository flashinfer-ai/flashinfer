"""Scratch synthesis must not allocate FP32 random temporaries."""

import importlib

import pytest
import torch

from flashinfer.autotuner import AutoTuner, TuningConfig
from flashinfer.gemm import svdquant_sm120_cutlass


@pytest.mark.parametrize(
    ("config", "workspace_index"),
    (
        (svdquant_sm120_cutlass._NVFP4_SVDQUANT_GEMM_TUNING_CONFIG_EXACT, 9),
        (svdquant_sm120_cutlass._SM120_LINEAR_TUNING_CONFIG_COLD, 13),
    ),
    ids=("gemm", "linear"),
)
def test_scratch_synthesis_does_not_randomize_workspace(
    monkeypatch: pytest.MonkeyPatch,
    config: TuningConfig,
    workspace_index: int,
) -> None:
    """Exercise the actual profiles while rejecting random scratch synthesis."""
    tuner_module = importlib.import_module("flashinfer.autotuner.autotuner")
    initialize = tuner_module.autotuner_initializer_rand_scaled
    workspace_shape = (4096,)

    def reject_scratch_randomization(
        shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        assert shapes != workspace_shape, (
            "scratch workspace used random FP32 temporaries"
        )
        return initialize(shapes, dtype, device)

    monkeypatch.setattr(
        tuner_module, "autotuner_initializer_rand_scaled", reject_scratch_randomization
    )
    inputs = [torch.ones((2, 256), dtype=torch.uint8) for _ in range(workspace_index)]
    inputs.append(torch.empty(workspace_shape, dtype=torch.uint8))
    tuner = AutoTuner.get()
    profiles = tuner._generate_optimization_profiles(config, inputs)
    assert len(profiles) == 1
    prepared = tuner._prepare_input_tensors(profiles[0], inputs)
    workspace = prepared[workspace_index]
    assert workspace is not None
    assert workspace.shape == workspace_shape
    assert workspace.dtype == torch.uint8
    assert workspace.data_ptr() != inputs[workspace_index].data_ptr()
