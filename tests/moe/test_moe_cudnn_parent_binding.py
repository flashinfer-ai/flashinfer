# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A shared FC1 parent is an alias contract, not merely a stride signature."""

import pytest
import torch

from flashinfer.fused_moe import CudnnMoeConfig
from flashinfer.fused_moe.cudnn_backend import (
    CudnnMoeRunner,
    _has_canonical_fc1_parent,
)


def _pack():
    return CudnnMoeConfig.prepare_weights(
        torch.randn(4, 128, 128, dtype=torch.bfloat16),
        torch.randn(4, 128, 64, dtype=torch.bfloat16),
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=64,
    )


def _inputs(pack):
    return [None, None, None, *(pack[key] for key in ("up", "gate", "down", "gate_up"))]


def test_same_strides_do_not_declare_a_shared_fc1_parent():
    pack = _pack()
    weights = [pack["up"], pack["gate"]]
    assert _has_canonical_fc1_parent(weights, pack["gate_up"])
    independent = [
        torch.empty_strided(weight.shape, weight.stride(), dtype=weight.dtype)
        for weight in weights
    ]
    for dst, src in zip(independent, weights, strict=True):
        dst.copy_(src)
    assert not _has_canonical_fc1_parent(independent, pack["gate_up"])
    runner = object.__new__(CudnnMoeRunner)
    unrelated = dict(pack, up=independent[0], gate=independent[1])
    assert runner._weight_layout_key(_inputs(pack)) != runner._weight_layout_key(
        _inputs(unrelated)
    )


def test_distinct_shared_parents_reuse_the_plan_geometry():
    first, second = _pack(), _pack()
    assert first["gate_up"].data_ptr() != second["gate_up"].data_ptr()
    runner = object.__new__(CudnnMoeRunner)
    assert runner._weight_layout_key(_inputs(first)) == runner._weight_layout_key(
        _inputs(second)
    )
    assert _has_canonical_fc1_parent([second["up"], second["gate"]], second["gate_up"])
    assert not _has_canonical_fc1_parent(
        [first["up"], first["gate"]], second["gate_up"]
    )
    assert not _has_canonical_fc1_parent([first["gate"], first["up"]], first["gate_up"])


@pytest.mark.parametrize("failure", [RuntimeError, NotImplementedError])
def test_parent_retry_does_not_hide_compile_errors(monkeypatch, failure):
    from flashinfer.fused_moe import cudnn_backend as backend

    calls = []
    error = failure("kernel compilation failed")

    def stage(*args, **kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(backend, "_Stage", stage)
    with pytest.raises(failure) as caught:
        backend._prepare_fused_fc1(object(), weights_parent=object(), fused=True)
    assert caught.value is error
    assert len(calls) == 1


def test_parent_decline_keeps_fusion_weights_and_requested_tactic(monkeypatch):
    from flashinfer.fused_moe import cudnn_backend as backend

    calls = []
    weight_views, parent, tactic, prepared = object(), object(), object(), object()

    def stage(*args, **kwargs):
        calls.append((args, kwargs))
        if kwargs["weights_parent"] is not None:
            raise backend._ParentGraphUnsupported("older FE has no weight slices")
        return prepared

    monkeypatch.setattr(backend, "_Stage", stage)
    assert (
        backend._prepare_fused_fc1(
            weight_views, weights_parent=parent, fused=True, tactic=tactic
        )
        is prepared
    )
    assert len(calls) == 2
    for args, kwargs in calls:
        assert args == (weight_views,)
        assert kwargs["fused"] is True and kwargs["tactic"] is tactic
    assert calls[0][1]["weights_parent"] is parent
    assert calls[1][1]["weights_parent"] is None
