"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""Tensor exports of a rebuilt KDA prefill launch record (CPU; no GPU needed).

A record rebuild exports to tvm-ffi only the tensors that moved since the
site's previous record (``_ffi_arguments_reusing``).  A moved tensor bound
again at the next argument position (``out`` / ``out_tma``, the FP32 final
state and its tensor map) takes the export of that position instead of a
second one.  The first launch of a binding passes the bound torch tensors
and builds no exported record; a second launch builds it and later launches
reuse it until a rebind drops it.  ``_ffi_arguments`` is replaced by a
recording stub, so no tvm-ffi conversion runs.
"""

from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_kda_tf32_runtime as runtime


@pytest.fixture
def exports(monkeypatch):
    calls = []

    def record(arguments):
        values = tuple(arguments)
        calls.append(values)
        return tuple(("ffi", id(value)) for value in values)

    monkeypatch.setattr(runtime, "_ffi_arguments", record)
    return calls


def _tensor():
    return torch.zeros(4)


def test_adjacent_repeat_takes_the_previous_export(exports):
    owner = SimpleNamespace()
    a, b = _tensor(), _tensor()
    out = runtime._ffi_arguments_reusing(owner, None, (a, a, 7, b))
    assert len(exports) == 1 and len(exports[0]) == 2
    assert exports[0][0] is a and exports[0][1] is b
    assert out[0] == ("ffi", id(a)) and out[1] is out[0]
    assert out[2] == 7 and out[3] == ("ffi", id(b))


def test_unmoved_tensors_keep_their_export(exports):
    owner = SimpleNamespace()
    a, b = _tensor(), _tensor()
    first = runtime._ffi_arguments_reusing(owner, None, (a, a, b))
    again = runtime._ffi_arguments_reusing(owner, None, (a, a, b))
    assert len(exports) == 1
    assert again == first


def test_moved_repeat_is_exported_once(exports):
    owner = SimpleNamespace()
    a, b, c = _tensor(), _tensor(), _tensor()
    runtime._ffi_arguments_reusing(owner, None, (a, a, b))
    out = runtime._ffi_arguments_reusing(owner, None, (c, c, b))
    assert len(exports) == 2 and exports[1] == (c,)
    assert out[0] == ("ffi", id(c)) and out[1] is out[0]
    assert out[2] == ("ffi", id(b))


def test_a_repeat_at_a_later_position_is_exported_again(exports):
    owner = SimpleNamespace()
    a, b = _tensor(), _tensor()
    out = runtime._ffi_arguments_reusing(owner, None, (a, b, a))
    assert len(exports) == 1 and len(exports[0]) == 3
    assert out[0] == out[2] == ("ffi", id(a))


def test_a_raw_record_passes_the_bound_tensors(exports):
    t = _tensor()
    module = SimpleNamespace(
        _ready=True,
        _call=object(),
        _arguments=lambda grid, bindings: [bindings["x"], grid],
    )
    call, arguments = runtime._prepared_kernel_record(module, 3, {"x": t}, export=False)
    assert call is module._call and arguments[0] is t and arguments[1] == 3
    assert exports == []
    _, arguments = runtime._prepared_kernel_record(module, 3, {"x": t})
    assert exports == [(t,)] and arguments[0] == ("ffi", id(t))


class _Launch:
    """The launch-record state of ``FlashKDABlackwellBF16FusedLaunch``."""

    _launch_records = None
    _descriptors_stale = False
    _launch_in_stream = runtime.FlashKDABlackwellBF16FusedLaunch._launch_in_stream

    def __init__(self):
        self.builds = []
        self.launched = []

    def _build_launch_records(self, export=True):
        self.builds.append(export)
        kind = "exported" if export else "raw"
        return [(self.launched.append, (kind,))]


def test_the_first_launch_of_a_binding_is_raw():
    launch = _Launch()
    for _ in range(3):
        launch._launch_in_stream()
    assert launch.builds == [False, True]
    assert launch.launched == ["raw", "exported", "exported"]


def test_a_rebind_makes_the_next_launch_raw():
    launch = _Launch()
    launch._launch_in_stream()
    launch._launch_in_stream()
    runtime._invalidate_launch_records(launch)
    launch._launch_in_stream()
    launch._launch_in_stream()
    assert launch.builds == [False, True, False, True]
    assert launch.launched == ["raw", "exported", "raw", "exported"]
