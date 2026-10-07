# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Support decisions must not retain storage or poison distinct calls."""

import gc
from contextlib import nullcontext
import weakref

import pytest
import torch

from flashinfer.cudnn import linear_attention as la


@pytest.fixture(autouse=True)
def isolated_cache():
    la._la_auto_declines.clear()
    la._la_auto_decline_scopes.clear()
    yield
    la._la_auto_declines.clear()
    la._la_auto_decline_scopes.clear()


def test_cache_does_not_retain_tensors_or_exception_tracebacks():
    calls = []

    def decline(q, **kwargs):
        calls.append(1)
        error = NotImplementedError("unsupported")
        error._fi_la_build_unsupported = True
        raise error

    q = torch.empty(2, 3)
    ref = weakref.ref(q)
    assert la._try_cudnn_auto(decline, q) is None
    del q
    gc.collect()
    assert ref() is None
    assert la._try_cudnn_auto(decline, torch.ones(2, 3)) is None
    assert calls == [1]


@pytest.mark.parametrize(
    "change",
    [
        "shape",
        "stride",
        "dtype",
        "device",
        "grad",
        "scale",
        "normalization",
        "gate",
        "output",
        "state",
        "frontend",
        "builder",
    ],
)
def test_distinct_descriptor_or_runtime_retries(monkeypatch, change):
    calls = []

    def decline(*args, **kwargs):
        calls.append(1)
        error = NotImplementedError("unsupported")
        error._fi_la_build_unsupported = True
        raise error

    q = torch.empty(2, 3)
    options = dict(
        scale=1.0,
        additive_epsilon=1e-6,
        gate_domain="log",
        output=None,
        initial_state=None,
    )
    la._try_cudnn_auto(decline, q, **options)
    la._try_cudnn_auto(decline, q.clone(), **options)
    assert len(calls) == 1
    if change == "shape":
        q = torch.empty(3, 3)
    elif change == "stride":
        q = torch.empty(3, 2).t()
    elif change == "dtype":
        q = q.double()
    elif change == "device":
        q = q.to("meta")
    elif change == "grad":
        q.requires_grad_()
    elif change == "scale":
        options["scale"] = 0.5
    elif change == "normalization":
        options["additive_epsilon"] = None
    elif change == "gate":
        options["gate_domain"] = "linear"
    elif change == "output":
        options["output"] = torch.empty_like(q)
    elif change == "state":
        options["initial_state"] = torch.zeros(1, 3, 3)
    elif change == "frontend":
        monkeypatch.setattr(la.cudnn, "__version__", "1.32.0")
    else:
        monkeypatch.setattr(la, "_build_la_graph", lambda *args, **kwargs: None)
    la._try_cudnn_auto(decline, q, **options)
    assert len(calls) == 2


def test_decline_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(la, "_LA_AUTO_DECLINE_LIMIT", 2)
    calls = []

    def decline(q):
        calls.append(q.numel())
        error = NotImplementedError("unsupported")
        error._fi_la_build_unsupported = True
        raise error

    for size in (1, 2, 3, 1):
        la._try_cudnn_auto(decline, torch.empty(size))
    assert calls == [1, 2, 3, 1]
    assert len(la._la_auto_declines) == 2


@pytest.mark.parametrize(
    "error_type", [RuntimeError, MemoryError, TypeError, NotImplementedError]
)
def test_unmarked_failures_are_never_cached(error_type):
    calls = []
    error = error_type("transient or execution failure")

    def fail(q):
        calls.append(1)
        raise error

    for _ in range(2):
        with pytest.raises(error_type) as caught:
            la._try_cudnn_auto(fail, torch.empty(3))
        assert caught.value is error
    assert calls == [1, 1]
    assert not la._la_auto_declines


def test_successful_calls_do_not_compute_decline_keys(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("no negative-cache key work is needed before any decline")

    monkeypatch.setattr(la, "_la_auto_decline_key", unexpected)
    q = torch.ones(1)
    for _ in range(3):
        assert la._try_cudnn_auto(lambda value: value, q) is q


def test_capture_build_decline_does_not_poison_eager(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    capturing = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    calls = []

    def candidate(q):
        calls.append(1)
        if capturing:
            error = NotImplementedError("cannot build inside capture")
            error._fi_la_build_unsupported = True
            raise error
        return q

    q = torch.ones(1)
    assert la._try_cudnn_auto(candidate, q) is None
    capturing = False
    assert la._try_cudnn_auto(candidate, q) is q
    assert calls == [1, 1]
    assert not la._la_auto_declines


@pytest.mark.parametrize("other_family", [False, True])
def test_other_shapes_and_families_skip_full_keys_after_decline(
    monkeypatch, other_family
):
    def candidate(q):
        if q.numel() == 1:
            error = NotImplementedError("unsupported shape")
            error._fi_la_build_unsupported = True
            raise error
        return q

    la._try_cudnn_auto(candidate, torch.ones(1))

    def unexpected(*args, **kwargs):
        pytest.fail("unrelated successful calls must not build full decline keys")

    monkeypatch.setattr(la, "_la_auto_decline_key", unexpected)
    q = torch.ones(1 if other_family else 2)
    call = (lambda value: value) if other_family else candidate
    assert la._try_cudnn_auto(call, q) is q
