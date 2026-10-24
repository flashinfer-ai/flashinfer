# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sorted FP8/MX paths must compile when weight offsets need 64-bit indexing."""

import pytest
import torch

from flashinfer.fused_moe import ReLU, SwiGLU
from tests.moe.test_unified_moe_cutile import (
    _CUTILE_FP8_MODES,
    _make_cutile_fp8_case,
)


@pytest.mark.parametrize(
    "mode_index,execution_mode",
    [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (3, 5), (4, 5)],
)
def test_sorted_fp8_wide_indexing_graph(monkeypatch, mode_index, execution_mode):
    # The helper applies the native architecture/compiler guards. For fused
    # GEMM1 quantization, MXFP8 supports ungated activation; MXFP4 supports gated.
    activation = ReLU() if (mode_index, execution_mode) == (3, 5) else SwiGLU()
    runner, inputs, _, _, _ = _make_cutile_fp8_case(
        _CUTILE_FP8_MODES[mode_index],
        activation,
        tokens=4,
        hidden=256,
        inter=256,
    )
    module = runner._kernel_module
    tactic = (execution_mode, *runner._fallback_tactic(inputs))
    if execution_mode == 5:
        tactic = (*tactic[:2], 128, 128, 2, *tactic[5:])
    narrow = runner.forward(inputs, tactic).clone()
    # This regression checks index-width equivalence. The existing cuTile
    # precision tests cover reference accuracy; an extra elementwise FP8
    # reference check also fails for some data on the unchanged base path.
    assert torch.isfinite(narrow).all()

    # Exercise the same dispatch used by multi-GiB expert tensors without
    # requiring a large allocation in every CI worker.
    monkeypatch.setattr(module, "needs_int64_indexing", lambda *_: True)
    wide = runner.forward(inputs, tactic).clone()
    torch.testing.assert_close(wide, narrow, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner.forward(inputs, tactic)
    inputs[3].mul_(0.5)
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, wide * 0.5, atol=0, rtol=0)
