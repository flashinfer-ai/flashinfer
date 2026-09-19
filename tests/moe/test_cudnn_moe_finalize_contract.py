# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real runner dispatch while keeping routing-scale precision independent."""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from flashinfer.fused_moe import cudnn_backend
from flashinfer.fused_moe.cute_dsl import moe_utils


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("hidden", [2048, 4096])
@pytest.mark.parametrize("top_k", [4, 8])
def test_finalize_dispatch_preserves_scale_mode(
    monkeypatch, native, packed, tokens, hidden, top_k
):
    runner = object.__new__(cudnn_backend.CudnnMoeRunner)
    runner._require_built = lambda: None
    runner._native_finalize_pdl = native
    runner.backend_config = SimpleNamespace(use_native_routing=False)
    runner.config = SimpleNamespace(routing=SimpleNamespace(top_k=top_k))
    runner._stage_indices = lambda *_: None
    runner._run_stages = Mock()
    rows = tokens * top_k
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    ids = (torch.arange(rows, dtype=torch.int32) % 4).view(tokens, top_k)
    scales = torch.full((tokens, top_k), 0.5009765625, dtype=torch.float32)
    output = torch.empty_like(x)
    state = {
        "sorted_ids": torch.empty(rows, dtype=torch.int32),
        "order": torch.empty(rows, dtype=torch.int64),
        "source_rows": torch.empty(rows, dtype=torch.int64),
        "expert_range": torch.arange(4, dtype=torch.int32),
        "offsets": torch.empty(4, dtype=torch.int32),
        "inverse64": torch.empty(rows, dtype=torch.int64),
        "arange": torch.arange(rows, dtype=torch.int64),
        "inverse": torch.empty(rows, dtype=torch.int32),
        "routed": torch.empty(rows, hidden, dtype=torch.bfloat16),
        "projected": torch.empty(rows, hidden, dtype=torch.bfloat16),
    }
    runner._resources = lambda _: state
    finalizer = Mock()
    monkeypatch.setattr(moe_utils, "moe_unpermute", finalizer)
    inputs = [x, ids, scales, None, None, None, None, output, packed]
    assert runner.forward(inputs) is output
    runner._run_stages.assert_called_once_with(state, inputs, -1)
    finalizer.assert_called_once()
    args = finalizer.call_args.args
    kwargs = finalizer.call_args.kwargs
    assert args[0] is state["projected"] and args[1] is output
    assert args[3] is scales and scales.dtype == torch.float32
    assert args[4:] == (tokens, top_k)
    assert kwargs["round_scales_to_bf16"] is packed
    assert kwargs["enable_pdl"] is (
        native and tokens == 1 and hidden == 2048 and top_k == 8
    )
    assert kwargs["use_native_finalize"] is (hidden == 2048 and top_k == 8)
