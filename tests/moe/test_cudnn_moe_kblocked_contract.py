# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit K64 preparation keeps parent aliases and never changes FC2."""

import pytest
import torch
from flashinfer.fused_moe import CudnnMoeConfig
from flashinfer.fused_moe.cudnn_backend import (
    _has_k64_fc1_parent,
    _has_canonical_fc1_parent,
    _prepare_fused_fc1,
    _ParentGraphUnsupported,
)


@pytest.mark.parametrize(
    "e,h,n", [(3, 128, 64), (3, 64, 64), (1, 64, 64), (1, 128, 192)]
)
def test_k64_weight_preparation_and_aliases(e, h, n):
    w1 = torch.arange(e * 2 * n * h).reshape(e, 2 * n, h).to(torch.bfloat16)
    w2 = torch.randn(e, h, n, dtype=torch.bfloat16)
    args = dict(num_local_experts=e, hidden_size=h, intermediate_size=n)
    old = CudnnMoeConfig.prepare_weights(w1, w2, **args)
    new = CudnnMoeConfig.prepare_weights(
        w1, w2, **args, fc1_weight_layout="k_blocked_64_v1"
    )
    assert _has_canonical_fc1_parent([old["up"], old["gate"]], old["gate_up"])
    assert _has_k64_fc1_parent([new["up"], new["gate"]], new["gate_up"])
    assert not _has_canonical_fc1_parent([new["up"], new["gate"]], new["gate_up"])
    assert torch.equal(
        new["gate_up"].permute(0, 2, 1, 3).reshape(e, 2 * n, h), old["gate_up"]
    )
    assert new["down"].data_ptr() == old["down"].data_ptr() == w2.data_ptr()
    assert new["gate_up"].numel() == old["gate_up"].numel()
    assert not _has_k64_fc1_parent([new["up"].clone(), new["gate"]], new["gate_up"])
    assert not _has_k64_fc1_parent([new["gate"], new["up"]], new["gate_up"])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(fc1_weight_layout="unknown"),
        dict(fc1_weight_layout="k_blocked_64_v1", fc1_fusion=False),
    ],
)
def test_invalid_k64_config_declines(kwargs):
    with pytest.raises(ValueError):
        CudnnMoeConfig(**kwargs)


def test_k64_decline_never_retries_canonical(monkeypatch):
    from flashinfer.fused_moe import cudnn_backend as backend

    calls = []

    def decline(*args, **kwargs):
        calls.append(kwargs)
        raise _ParentGraphUnsupported("unsupported")

    monkeypatch.setattr(backend, "_Stage", decline)
    with pytest.raises(_ParentGraphUnsupported):
        _prepare_fused_fc1(weights_parent=object(), weight_layout="k_blocked_64_v1")
    assert len(calls) == 1
