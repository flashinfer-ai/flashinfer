# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""The standalone grouped-GEMM API remains experimental."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("gate", [None, "0", "1"])
def test_bf16_grouped_api_preserves_explicit_tactic(explicit, gate, monkeypatch):
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.backends import cudnn_frost as package
    from flashinfer.fused_moe.cudnn_frost_selected import (
        cudnn_frost_grouped_gemm1_swiglu,
    )

    if gate is None:
        monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    else:
        monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", gate)
    calls, choices = [], []
    requested = ("manual", "artifact", "digest")
    selected = ("autotuned", "artifact", "digest")
    output = torch.empty(2, 3, dtype=torch.bfloat16)

    def run(*, inputs, tactic):
        calls.append(tactic)
        return inputs[5]

    def choose(*args):
        choices.append(args)
        return run, selected

    monkeypatch.setattr(package, "CudnnFrostGroupedGemm1SwiGLURunner", lambda: run)
    monkeypatch.setattr(package, "workspace_size", lambda *args: 0)
    monkeypatch.setattr(AutoTuner, "get", lambda: SimpleNamespace(choose_one=choose))
    result = cudnn_frost_grouped_gemm1_swiglu(
        torch.empty(2, 4, dtype=torch.bfloat16),
        torch.empty(1, 3, 4, dtype=torch.bfloat16),
        torch.empty(1, 3, 4, dtype=torch.bfloat16),
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1),
        torch.empty(0, dtype=torch.uint8),
        out=output,
        tactic=requested if explicit else -1,
    )
    assert result is output
    assert calls == [requested if explicit else selected]
    assert len(choices) == (0 if explicit else 1)
