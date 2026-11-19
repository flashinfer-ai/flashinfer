# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registered dispatch-window ownership and admission, without GPUs."""

from types import SimpleNamespace

import pytest
import torch


def test_window_reuse_capture_geometry_and_cleanup(
    modern_ep, make_native_handle, monkeypatch
):
    from flashinfer.moe_ep import DispatchInputParams

    fleet, handle = make_native_handle(dispatch_zero_copy=True)
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
    calls = [call for call in handle._handle.calls if call[0] == "dispatch"]
    assert len(calls) == 2
    assert all(call[2].tokens is fleet._dispatch_windows["tokens"][3] for call in calls)
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
    "algorithm,layout,match",
    [
        ("HIGH_THROUGHPUT", "EXPERT_MAJOR", "LOW_LATENCY"),
        ("LOW_LATENCY", "EXPERT_MAJOR", "RANK_MAJOR"),
    ],
)
def test_window_admission(make_native_handle, algorithm, layout, match):
    with pytest.raises(ValueError, match=match):
        make_native_handle(algorithm=algorithm, layout=layout, dispatch_zero_copy=True)


@pytest.mark.parametrize("null_handle", [False, True])
def test_window_registration_failure_allows_retry(
    modern_ep, make_native_handle, monkeypatch, null_handle
):
    fleet, handle = make_native_handle(dispatch_zero_copy=True)
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
    modern_ep, make_native_handle, monkeypatch
):
    fleet, handle = make_native_handle(dispatch_zero_copy=True)
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
