# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered dispatch-window ownership and admission, without GPUs."""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def modern_ep(fake_nccl_ep):
    fake_nccl_ep.get_lib_version = lambda: "0.3.0"
    return fake_nccl_ep


def make_handle(ep, *, algorithm="LOW_LATENCY", layout="RANK_MAJOR", **options):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetAlgoKnobNcclEpConfig,
        FleetParams,
        HandleAlgoKnobTopKWeights,
        HandleParams,
    )
    from flashinfer.moe_ep.backends.split.comm.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        FleetParams(
            num_experts=8,
            max_tokens_per_rank=4,
            token_hidden_size=2048,
            algorithm=EpAlgorithm[algorithm],
            layout=EpLayout[layout],
        ),
        [FleetAlgoKnobNcclEpConfig(**options)],
    )
    handle = fleet.create_handle(
        HandleParams(topk_ids=torch.zeros(3, 2, dtype=torch.int64)),
        [HandleAlgoKnobTopKWeights(torch.ones(3, 2))],
    )
    return fleet, handle


def test_window_reuse_capture_geometry_and_cleanup(
    modern_ep, bypass_build_checks, monkeypatch
):
    from flashinfer.moe_ep import DispatchInputParams

    fleet, handle = make_handle(modern_ep, dispatch_zero_copy=True)
    modern_ep._core.WindowFlag = SimpleNamespace(COLL_SYMMETRIC=1)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0)
    )
    windows = []

    class Window:
        handle = 1
        closed = 0

        def close(self):
            self.closed += 1

    def register(tensor, flags):
        window = Window()
        windows.append(window)
        return window

    fleet._comm = SimpleNamespace(register_window=register, destroy=lambda: None)
    monkeypatch.setattr(
        fleet,
        "_allocate_window_tensor",
        lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
    )
    x = torch.zeros(3, 2048, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="warm up"):
        handle.dispatch(DispatchInputParams(x=[x]))
    assert not windows
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    first = handle.dispatch(DispatchInputParams(x=[x]))
    assert len(windows) == 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    second = handle.dispatch(DispatchInputParams(x=[x]))
    assert first.expert_tensors.data_ptr() == second.expert_tensors.data_ptr()
    assert len(windows) == 1
    with pytest.raises(ValueError, match="geometry"):
        fleet._dispatch_window("tokens", (4, 4, 4096), x.dtype, x.device)
    assert all(w.closed == 0 for w in windows)
    handle.destroy()
    fleet.destroy()
    fleet.destroy()
    assert all(w.closed == 1 for w in windows)
    assert not fleet._dispatch_windows


@pytest.mark.parametrize(
    "algorithm,layout,version,match",
    [
        ("LOW_LATENCY", "RANK_MAJOR", "0.2.0", "EP 0.3"),
        ("HIGH_THROUGHPUT", "EXPERT_MAJOR", "0.3.0", "LOW_LATENCY"),
        ("LOW_LATENCY", "EXPERT_MAJOR", "0.3.0", "RANK_MAJOR"),
    ],
)
def test_window_admission(
    modern_ep, bypass_build_checks, algorithm, layout, version, match
):
    modern_ep.get_lib_version = lambda: version
    with pytest.raises(ValueError, match=match):
        make_handle(
            modern_ep, algorithm=algorithm, layout=layout, dispatch_zero_copy=True
        )


@pytest.mark.parametrize("null_handle", [False, True])
def test_window_registration_failure_allows_retry(
    modern_ep, bypass_build_checks, monkeypatch, null_handle
):
    fleet, handle = make_handle(modern_ep, dispatch_zero_copy=True)
    modern_ep._core.WindowFlag = SimpleNamespace(COLL_SYMMETRIC=1)
    closed = []
    rejected = SimpleNamespace(handle=0, close=lambda: closed.append("rejected"))
    accepted = SimpleNamespace(handle=1, close=lambda: closed.append("accepted"))
    registrations = iter([None if null_handle else rejected, accepted])
    fleet._comm = SimpleNamespace(
        register_window=lambda tensor, flags: next(registrations), destroy=lambda: None
    )
    monkeypatch.setattr(
        fleet,
        "_allocate_window_tensor",
        lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
    )
    shape = (4, 4, 2048)
    with pytest.raises(ValueError, match="did not register"):
        fleet._dispatch_window("tokens", shape, torch.bfloat16, torch.device("cpu"))
    assert not fleet._dispatch_windows
    assert closed == ([] if null_handle else ["rejected"])
    fleet._dispatch_window("tokens", shape, torch.bfloat16, torch.device("cpu"))
    handle.destroy()
    fleet.destroy()
    assert closed == (["accepted"] if null_handle else ["rejected", "accepted"])


def test_window_descriptor_failure_releases_registration(
    modern_ep, bypass_build_checks, monkeypatch
):
    fleet, handle = make_handle(modern_ep, dispatch_zero_copy=True)
    modern_ep._core.WindowFlag = SimpleNamespace(COLL_SYMMETRIC=1)
    closed = []
    window = SimpleNamespace(handle=1, close=lambda: closed.append(True))
    fleet._comm = SimpleNamespace(
        register_window=lambda tensor, flags: window, destroy=lambda: None
    )
    monkeypatch.setattr(
        fleet,
        "_allocate_window_tensor",
        lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
    )

    def reject_descriptor(*args, **kwargs):
        raise RuntimeError("descriptor construction failed")

    monkeypatch.setattr(modern_ep, "Tensor", reject_descriptor)
    with pytest.raises(RuntimeError, match="descriptor"):
        fleet._dispatch_window(
            "tokens", (4, 4, 2048), torch.bfloat16, torch.device("cpu")
        )
    assert closed == [True] and not fleet._dispatch_windows
    handle.destroy()
    fleet.destroy()
