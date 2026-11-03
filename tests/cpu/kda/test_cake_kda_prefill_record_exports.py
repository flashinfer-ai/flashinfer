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
second one.  ``_ffi_arguments`` is replaced by a recording stub, so no
tvm-ffi conversion runs.
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
