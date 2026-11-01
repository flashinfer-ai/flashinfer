"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Public-API tests of the det-free chunked LM-head entries ``flashinfer.chunked_lm_head.chunked_lm_head_{loss,
logprob,preload,support_logprob,support_kl}``: experimental dispatchers with a keyword-only
``deterministic: bool = True``.  ``True`` runs the deterministic entry (``chunked_lm_head_det_*``) with the same
arguments -- the same bits, no added allocation or synchronization, one identity test of host work --;
``False`` runs ``flashinfer.experimental.cake.cake_lm_head_loss`` with the same argument checks and environment
fallbacks, or the deterministic entry with one warning per entry point while torch's deterministic algorithms are
on; any other value is a ``TypeError``.  The deterministic entries are unchanged and still refuse
``deterministic=False``.  The host-layer tests replace the packages by recorders; the device tests compare the
bits of real calls.
"""

import importlib
import inspect
import json
import os
import sys
import warnings

import pytest
import torch
import torch.utils.checkpoint

import flashinfer.api_logging as api_logging
import flashinfer.chunked_lm_head as public
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs

SEED = 20261031
H_HOST, V_HOST = 256, 512
# det-free entry -> its deterministic entry
ENTRIES = {
    "chunked_lm_head_loss": "chunked_lm_head_det_loss",
    "chunked_lm_head_logprob": "chunked_lm_head_det_logprob",
    "chunked_lm_head_preload": "chunked_lm_head_det_preload",
    "chunked_lm_head_support_logprob": "chunked_lm_head_det_support_logprob",
    "chunked_lm_head_support_kl": "chunked_lm_head_det_support_kl",
}
# The deterministic entry's body each dispatcher calls (below the experimental wrapper).
BODIES = {
    "chunked_lm_head_loss": "_det_loss",
    "chunked_lm_head_logprob": "_det_logprob",
    "chunked_lm_head_preload": "_det_preload",
    "chunked_lm_head_support_logprob": "_det_support_logprob",
    "chunked_lm_head_support_kl": "_det_support_kl",
}
SUPPORT = ("chunked_lm_head_support_logprob", "chunked_lm_head_support_kl")


def _host_call():
    X = torch.zeros(4, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    labels = torch.zeros(4, dtype=torch.int64)
    return X, W, labels


def _support_args():
    X, W, _ = _host_call()
    ids = torch.zeros(4, dtype=torch.int64)
    offsets = torch.arange(5, dtype=torch.int64)
    return X, W, ids, offsets


def _call_args(entry):
    """Positional and keyword arguments of one call of ``entry`` (every keyword given a value that is not its
    default where the type allows, so a dropped or reordered keyword shows)."""
    X, W, labels = _host_call()
    if entry == "chunked_lm_head_loss":
        return (X, W, labels), dict(
            objective="policy",
            loss_div=3.0,
            infer_logp=torch.zeros(4),
            loss_weights=torch.ones(4),
            chunk_size=1024,
            return_logp=True,
            grad_weight_dtype=torch.float32,
            backend="cake",
            compact_rows=False,
            fuse_dw_cast=False,
            dx_finalize=False,
            dw_stream="auto",
            hidden_count=False,
            gemm_tuning={"dx": {"k_slices": 2}},
            memory_mode="budget",
            num_valid_rows=4,
            check_num_valid_rows=True,
        )
    if entry == "chunked_lm_head_logprob":
        return (X, W, labels), dict(
            chunk_size=1024,
            backend="cake",
            compact_rows=False,
            fuse_dw_cast=False,
            dx_finalize=False,
            dw_stream=True,
            hidden_count=False,
            gemm_tuning={"logits": {"group_m": None}},
            memory_mode="fast",
            num_valid_rows=3,
            check_num_valid_rows=True,
        )
    if entry == "chunked_lm_head_preload":
        return ("cuda:0",), {}
    X, W, ids, offsets = _support_args()
    if entry == "chunked_lm_head_support_logprob":
        return (X, W, ids, offsets, labels), dict(
            k_cap=3, temperature=0.5, chunk_size=1024, check_inputs=True, backend="cake"
        )
    return (X, W, ids, offsets, torch.zeros(4)), dict(
        k_cap=3,
        temperature=0.5,
        return_logprobs=True,
        chunk_size=1024,
        check_inputs=True,
        backend="cake",
    )


class _Recorder:
    """A stand-in for a deterministic body or a package entry: records its calls, returns a marker."""

    def __init__(self, name):
        self.name, self.calls = name, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("result-of", self.name)


class _FakePackage:
    """A stand-in for a module of the non-deterministic package: every attribute is a recorder."""

    def __init__(self):
        self.recorders = {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.recorders.setdefault(name, _Recorder(name))


def _fail(*_args, **_kwargs):
    raise AssertionError("the deterministic=True path must not reach this")


@pytest.fixture
def recorders(monkeypatch):
    """The deterministic bodies and the non-deterministic package replaced by recorders."""
    bodies = {entry: _Recorder(body) for entry, body in BODIES.items()}
    for entry, body in BODIES.items():
        monkeypatch.setattr(public, body, bodies[entry])
    fake = {
        "backend": _FakePackage(),
        "opt_in": _FakePackage(),
        "support": _FakePackage(),
    }
    monkeypatch.setattr(public, "_nd_backend", lambda: fake["backend"])
    monkeypatch.setattr(public, "_nd_backend_opt_in", lambda: fake["opt_in"])
    monkeypatch.setattr(public, "_nd_support", lambda: fake["support"])
    monkeypatch.setattr(public, "_DET_FALLBACK_WARNED", set())
    monkeypatch.setattr(public, "_DET_FALLBACK_PIN_WARNED", set())
    for name in (
        "FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING",
    ):
        monkeypatch.delenv(name, raising=False)
    return bodies, fake


# ---------------------------------------------------------------------------
# Host layer
# ---------------------------------------------------------------------------


def test_entries_are_experimental_dispatchers():
    """Each det-free entry is its own experimental function (no longer the deterministic object) with the
    deterministic entry's feature, and every deterministic entry keeps its marking."""
    for entry, det in ENTRIES.items():
        fn, det_fn = getattr(public, entry), getattr(public, det)
        assert fn is not det_fn, entry
        assert fn.is_experimental is True and det_fn.is_experimental is True, entry
        assert fn.experimental_feature == det_fn.experimental_feature, entry
        assert fn.__name__ == entry and "SM100" in fn.__doc__, entry
        assert getattr(public, BODIES[entry]) is det_fn.__wrapped__, entry


def test_signatures_mirror_the_deterministic_entries():
    """The loss / log-probability entries have the deterministic signature (``deterministic`` keyword-only,
    default ``True``); preload and the support entries add that keyword (the deterministic ones have none)."""
    for entry, det in ENTRIES.items():
        params = dict(inspect.signature(getattr(public, entry)).parameters)
        det_params = dict(inspect.signature(getattr(public, det)).parameters)
        mode = params["deterministic"]
        assert mode.kind is inspect.Parameter.KEYWORD_ONLY and mode.default is True
        assert mode.annotation in (bool, "bool"), entry
        if "deterministic" in det_params:
            assert list(params.items()) == list(det_params.items()), entry
        else:
            del params["deterministic"]
            assert list(params.items()) == list(det_params.items()), entry


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("explicit", [False, True])
def test_default_runs_the_deterministic_body_with_the_same_arguments(
    recorders, monkeypatch, entry, explicit
):
    """``deterministic=True`` (default or explicit) calls the deterministic body once with exactly the call's
    arguments -- the same objects, positional and keyword as given, ``deterministic`` not passed on (the body's
    default is ``True``) -- and does nothing else: no knob resolution, no torch-mode query, no package import."""
    bodies, fake = recorders
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", _fail)
    monkeypatch.setattr(public, "_resolve_knobs", _fail)
    monkeypatch.setattr(public, "_runs_det", _fail)
    monkeypatch.setattr(public, "_det_fallback_kwargs", _fail)
    args, kwargs = _call_args(entry)
    given = dict(kwargs, deterministic=True) if explicit else kwargs
    assert getattr(public, entry)(*args, **given) == ("result-of", BODIES[entry])
    assert len(bodies[entry].calls) == 1
    got_args, got_kwargs = bodies[entry].calls[0]
    assert len(got_args) == len(args) and all(
        a is b for a, b in zip(got_args, args, strict=True)
    )
    assert list(got_kwargs) == list(kwargs)
    assert all(got_kwargs[name] is value for name, value in kwargs.items())
    # positional arguments the caller gives by keyword stay keywords
    if entry in ("chunked_lm_head_loss", "chunked_lm_head_logprob"):
        X, W, labels = args
        getattr(public, entry)(X, W=W, labels=labels, **given)
        assert bodies[entry].calls[-1] == ((X,), dict(W=W, labels=labels, **kwargs))
    assert not any(f.recorders for f in fake.values())


@pytest.mark.parametrize("entry", list(ENTRIES))
def test_default_path_runs_one_python_frame_beyond_the_deterministic_entry(
    recorders, entry
):
    """Structurally, a ``deterministic=True`` call runs the experimental wrapper, the dispatcher and the
    deterministic body and no other Python function (a profile of the call's Python frames), so the default adds one
    frame and its ``is True`` test to the deterministic entry's call.  At API log level 0 only (a higher level adds
    the logging layer to both)."""
    if api_logging._API_LOG_LEVEL != 0:
        pytest.skip("API logging adds its own frames")
    bodies, _ = recorders
    fn = getattr(public, entry)
    args, kwargs = _call_args(entry)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn(*args, **kwargs)  # the experimental warning of the first call
    frames: list = []

    def profile(frame, event, _arg):
        if event == "call":
            frames.append(frame.f_code)

    sys.setprofile(profile)
    try:
        fn(*args, **kwargs)
    finally:
        sys.setprofile(None)
    want = [fn.__code__, fn.__wrapped__.__code__, _Recorder.__call__.__code__]
    assert frames == want, [f"{c.co_filename}:{c.co_name}" for c in frames]
    assert len(bodies[entry].calls) == 2


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("value", [1, 0, None, "True", "False", 1.0])
def test_a_mode_that_is_not_a_bool_is_a_type_error(recorders, entry, value):
    bodies, fake = recorders
    args, kwargs = _call_args(entry)
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        getattr(public, entry)(*args, **kwargs, deterministic=value)
    assert not bodies[entry].calls and not any(f.recorders for f in fake.values())


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e != "chunked_lm_head_preload"]
)
def test_argument_errors_name_the_public_entry(entry, deterministic):
    """An unknown keyword or a missing argument names the public entry, not the deterministic body it forwards to or
    the private non-deterministic helper (both raise before any work: no device is needed)."""
    fn = getattr(public, entry)
    args = _support_args() if entry in SUPPORT else _host_call()
    with pytest.raises(
        TypeError, match=rf"^{entry}\(\) got an unexpected keyword argument 'bogus'$"
    ):
        fn(*args, bogus=1, deterministic=deterministic)
    with pytest.raises(TypeError, match=rf"^{entry}\(\) missing "):
        fn(deterministic=deterministic)


def test_false_reaches_the_non_deterministic_package(recorders, monkeypatch):
    """``deterministic=False`` calls the package's entry of the same name with ``deterministic=False``, the
    switches resolved from the environment like the deterministic entries do, and the opt-in module for
    ``memory_mode`` / ``num_valid_rows``; the deterministic bodies are not called."""
    bodies, fake = recorders
    monkeypatch.setenv("FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS", "0")
    monkeypatch.setenv("FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM", "on")
    X, W, labels = _host_call()
    public.chunked_lm_head_loss(X, W, labels, loss_div=2.0, deterministic=False)
    ((args, kw),) = fake["backend"].recorders["chunked_lm_head_loss"].calls
    assert args == (X, W, labels)
    assert kw["deterministic"] is False and kw["backend"] == "cake"
    assert kw["compact_rows"] is False and kw["dw_stream"] is True
    assert kw["loss_div"] == 2.0 and "memory_mode" not in kw
    public.chunked_lm_head_loss(
        X, W, labels, loss_div=2.0, deterministic=False, memory_mode="budget"
    )
    ((_, kw),) = fake["opt_in"].recorders["chunked_lm_head_loss"].calls
    assert kw["memory_mode"] == "budget" and kw["deterministic"] is False
    public.chunked_lm_head_logprob(X, W, labels, deterministic=False, num_valid_rows=4)
    ((_, kw),) = fake["opt_in"].recorders["chunked_lm_head_logprob"].calls
    assert kw["num_valid_rows"] == 4 and kw["deterministic"] is False
    public.chunked_lm_head_logprob(X, W, labels, deterministic=False)
    ((_, kw),) = fake["backend"].recorders["chunked_lm_head_logprob"].calls
    assert kw["deterministic"] is False
    public.chunked_lm_head_preload("cuda:0", deterministic=False)
    assert fake["opt_in"].recorders["preload"].calls == [(("cuda:0",), {})]
    for entry in SUPPORT:
        args, kwargs = _call_args(entry)
        getattr(public, entry)(*args, **kwargs, deterministic=False)
        ((got_args, got_kw),) = fake["support"].recorders[entry].calls
        assert all(a is b for a, b in zip(got_args, args, strict=True))
        assert got_kw == dict(kwargs, deterministic=False)
    assert not any(b.calls for b in bodies.values())


def test_false_checks_the_arguments_like_the_deterministic_entries(recorders):
    bodies, fake = recorders
    X, W, labels = _host_call()
    with pytest.raises(ValueError, match="backend"):
        public.chunked_lm_head_loss(
            X, W, labels, loss_div=1.0, backend="reference", deterministic=False
        )
    with pytest.raises(ValueError, match="backend"):
        public.chunked_lm_head_logprob(
            X, W, labels, backend="reference", deterministic=False
        )
    with pytest.raises(ValueError, match="num_valid_rows"):
        public.chunked_lm_head_loss(
            X, W, labels, loss_div=1.0, check_num_valid_rows=True, deterministic=False
        )
    for entry in SUPPORT:
        args, kwargs = _call_args(entry)
        with pytest.raises(ValueError, match="backend"):
            getattr(public, entry)(
                *args, **dict(kwargs, backend="reference"), deterministic=False
            )
    assert not any(f.recorders for f in fake.values())


@pytest.mark.parametrize("warn_only", [False, True])
def test_torch_deterministic_mode_runs_the_deterministic_entry_and_warns_once(
    recorders, warn_only
):
    """With torch's deterministic algorithms on (also ``warn_only``), ``deterministic=False`` runs the deterministic
    body with the call's arguments and warns once per entry point, naming the caller's line."""
    bodies, fake = recorders
    was, was_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        for entry in ENTRIES:
            args, kwargs = _call_args(entry)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                for _ in range(3):
                    getattr(public, entry)(*args, **kwargs, deterministic=False)
            mine = [
                w
                for w in rec
                if "runs the deterministic implementation" in str(w.message)
            ]
            assert len(mine) == 1, (entry, [str(w.message) for w in rec])
            assert mine[0].category is UserWarning and entry in str(mine[0].message)
            assert os.path.abspath(mine[0].filename) == os.path.abspath(__file__), mine[
                0
            ].filename
            assert len(bodies[entry].calls) == 3, entry
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)
    assert not any(f.recorders for f in fake.values())


@pytest.mark.parametrize("entry", ["chunked_lm_head_loss", "chunked_lm_head_logprob"])
@pytest.mark.parametrize("source", ["keyword", "environment"])
def test_det_fallback_drops_a_k_slices_pin_it_does_not_serve(
    recorders, monkeypatch, entry, source
):
    """Under torch's deterministic algorithms ``deterministic=False`` runs the deterministic body; a dX ``k_slices``
    pin above its four slices (``gemm_tuning`` or ``FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING``) is dropped with one
    warning per entry point -- the other knobs pass on --, and a pin it serves passes on unchanged without a
    warning."""
    bodies, fake = recorders
    X, W, labels = _host_call()
    extra = dict(loss_div=1.0) if entry == "chunked_lm_head_loss" else {}
    pinned = {"dx": {"k_slices": 8, "tile_n": 256}, "logits": {"group_m": 8}}
    served = {"dx": {"k_slices": 4}}
    was, was_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            for _ in range(3):
                if source == "keyword":
                    getattr(public, entry)(
                        X, W, labels, gemm_tuning=pinned, deterministic=False, **extra
                    )
                else:
                    monkeypatch.setenv(
                        "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING", json.dumps(pinned)
                    )
                    getattr(public, entry)(X, W, labels, deterministic=False, **extra)
            monkeypatch.delenv(
                "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING", raising=False
            )
            getattr(public, entry)(
                X, W, labels, gemm_tuning=served, deterministic=False, **extra
            )
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)
    dropped = [w for w in rec if "k_slices=8 is dropped" in str(w.message)]
    assert len(dropped) == 1 and entry in str(dropped[0].message), [
        str(w.message) for w in rec
    ]
    assert os.path.abspath(dropped[0].filename) == os.path.abspath(__file__)
    calls = bodies[entry].calls
    assert len(calls) == 4
    for _, kw in calls[:3]:
        assert kw["gemm_tuning"] == {"dx": {"tile_n": 256}, "logits": {"group_m": 8}}
    assert calls[3][1]["gemm_tuning"] is served
    assert not any(f.recorders for f in fake.values())


def test_det_fallback_drops_a_lone_k_slices_pin_to_no_dx_knobs(recorders):
    """A pin of ``k_slices`` alone leaves no dX knob: the deterministic body gets no ``dx`` entry (its own rule), and
    the other GEMMs' knobs pass on."""
    bodies, _ = recorders
    X, W, labels = _host_call()
    was, was_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            public.chunked_lm_head_logprob(
                X, W, labels, gemm_tuning={"dx": {"k_slices": 6}}, deterministic=False
            )
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)
    ((_, kw),) = bodies["chunked_lm_head_logprob"].calls
    assert kw["gemm_tuning"] == {}


def test_deterministic_entries_still_refuse_false():
    """``chunked_lm_head_det_loss`` / ``_logprob`` keep their contract: ``deterministic=False`` is refused by the
    deterministic backend's validation (host tensors suffice: it runs before any device query)."""
    X, W, labels = _host_call()
    with pytest.raises(NotImplementedError, match="only deterministic=True"):
        public.chunked_lm_head_det_loss(X, W, labels, loss_div=1.0, deterministic=False)
    with pytest.raises(NotImplementedError, match="only deterministic=True"):
        public.chunked_lm_head_det_logprob(X, W, labels, deterministic=False)
    for det in (
        "chunked_lm_head_det_support_logprob",
        "chunked_lm_head_det_support_kl",
    ):
        assert "deterministic" not in inspect.signature(getattr(public, det)).parameters


def test_remaining_aliases_are_the_canonical_objects():
    """Every alias that remains after the dispatchers is an identity: the legacy package path
    ``flashinfer.experimental.cake_lm_head_loss`` is the deterministic package (module for module), never the
    non-deterministic one, and the deterministic modules' former names are their canonical objects."""
    nd_root = "flashinfer.experimental.cake.cake_lm_head_loss"
    for name in (
        "cake_backend",
        "cake_backend_opt_in",
        "cake_build",
        "cake_jit",
        "cake_support",
    ):
        legacy = importlib.import_module(
            f"flashinfer.experimental.cake_lm_head_loss.{name}"
        )
        det = importlib.import_module(
            f"flashinfer.experimental.cake_lm_head_det_loss.{name}"
        )
        nd = importlib.import_module(f"{nd_root}.{name}")
        assert legacy is det and legacy is not nd, name
    det_backend = importlib.import_module(
        "flashinfer.experimental.cake_lm_head_det_loss.cake_backend"
    )
    for old, new in (
        ("chunked_lm_head_loss", "chunked_lm_head_det_loss"),
        ("chunked_lm_head_logprob", "chunked_lm_head_det_logprob"),
        ("prepare_lm_head_loss", "prepare_lm_head_det_loss"),
        ("LmHeadLossRunner", "LmHeadDetLossRunner"),
        ("load_cake_lm_head_loss_module", "load_cake_lm_head_det_loss_module"),
    ):
        assert getattr(det_backend, old) is getattr(det_backend, new), old


def test_non_deterministic_binding_caches_take_the_import_settings():
    """The package's binding caches are configured from the variables read at import, like the deterministic
    package's."""
    settings = public._BINDING_CACHE_SETTINGS
    for loader in (public._nd_backend, public._nd_backend_opt_in):
        module = loader()
        assert module.BINDING_CACHE.enabled == settings["enabled"]
        want = (
            module.BINDING_CACHE_DEFAULT_CAPACITY
            if settings["capacity"] is None
            else settings["capacity"]
        )
        assert module.BINDING_CACHE.capacity == want
    assert (
        public._nd_backend().__name__
        == "flashinfer.experimental.cake.cake_lm_head_loss.cake_backend"
    )


# ---------------------------------------------------------------------------
# Device: the bits and the host cost of the default path
# ---------------------------------------------------------------------------


def _require_device():
    det_backend = importlib.import_module(
        "flashinfer.experimental.cake_lm_head_det_loss.cake_backend"
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in (
        det_backend.SUPPORTED_COMPUTE_CAPABILITIES
    ):
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    if not det_backend.generated_program_available(torch.device("cuda"), entry="loss"):
        pytest.skip(
            "the deterministic package's programs are not registered for this device"
        )


def _bits(t):
    t = t.detach().reshape(-1).contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _loss_step(fn, inp, **kw):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    loss, logp = fn(
        X,
        W,
        inp.labels,
        objective=inp.objective,
        loss_div=inp.loss_div if inp.objective == "ce" else None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        return_logp=True,
        **kw,
    )
    loss.backward()
    return dict(loss=loss.detach(), logp=logp, dX=X.grad, dW=W.grad)


def _logprob_step(fn, inp, **kw):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    logp = fn(X, W, inp.labels, **kw)
    logp.backward(inp.dlogp)
    return dict(logp=logp.detach(), dX=X.grad, dW=W.grad)


def _support_step(fn, inp, **kw):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    out = fn(
        X, W, inp.support_ids, inp.support_offsets, inp.labels, **inp.kwargs(), **kw
    )
    out.backward(inp.d)
    return dict(out=out.detach(), dX=X.grad, dW=W.grad)


# Measured default calls per path.
_MEASURED_CALLS = 4


def _allocations() -> int:
    return int(torch.cuda.memory_stats()["allocation.all.allocated"])


def _syncs(records) -> int:
    return sum("synchroniz" in str(w.message) for w in records)


def _measure_dispatch(monkeypatch, entry, step, inp, kw):
    """The outputs of ``_MEASURED_CALLS`` default calls of ``entry`` and, per call, ``(allocations, synchronizations)``
    the call made OUTSIDE the deterministic body it reached: the caching allocator's allocation count and PyTorch's
    synchronization debug warnings, each taken around the whole call and around the body (a recorder in place of the
    body), so the difference is the dispatcher's own -- whatever the body's binding-cache state (a miss in one call
    and a hit in another changes the body's count, not the difference)."""
    body_name = BODIES[entry]
    body = getattr(public, body_name)
    records: list = []
    inside: list = []

    def probed_body(*args, **kwargs):
        a0, s0 = _allocations(), len(records)
        out = body(*args, **kwargs)
        inside.append((_allocations() - a0, _syncs(records[s0:])))
        return out

    entry_fn = getattr(public, entry)
    outside: list = []

    def measured_entry(*args, **kwargs):
        a0, s0 = _allocations(), len(records)
        out = entry_fn(*args, **kwargs)
        total = (_allocations() - a0, _syncs(records[s0:]))
        body_cost = inside[-1]
        outside.append((total[0] - body_cost[0], total[1] - body_cost[1]))
        return out

    monkeypatch.setattr(public, body_name, probed_body)
    outs = []
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            records = rec
            for _ in range(_MEASURED_CALLS):
                outs.append(step(measured_entry, inp, **kw))
    finally:
        torch.cuda.set_sync_debug_mode("default")
        monkeypatch.setattr(public, body_name, body)
    torch.cuda.synchronize()
    assert len(inside) == len(outside) == _MEASURED_CALLS
    return outs, outside


def _cases():
    from tests.test_helpers.cake_lm_head_support_reference import make_support_inputs

    cuda = torch.device("cuda")
    W = make_inputs(1, seed=SEED, device=cuda, ignore_frac=0.0).W
    ce = make_inputs(4097, seed=SEED + 1, device=cuda, W=W)
    policy = make_inputs(4097, objective="policy", seed=SEED + 2, device=cuda, W=W)
    sup = make_support_inputs(
        257, seed=SEED + 3, H=256, V=512, k_cap=8, device=cuda, labels=True
    )
    return [
        ("loss-ce", _loss_step, ce, {}),
        ("loss-policy", _loss_step, policy, {}),
        ("loss-budget", _loss_step, ce, dict(memory_mode="budget")),
        ("loss-count", _loss_step, ce, dict(num_valid_rows=int(ce.valid.sum()))),
        ("logprob", _logprob_step, ce, {}),
        ("support-logprob", _support_step, sup, {}),
    ]


_DET_OF = {
    _loss_step: "chunked_lm_head_det_loss",
    _logprob_step: "chunked_lm_head_det_logprob",
    _support_step: "chunked_lm_head_det_support_logprob",
}
_NEW_OF = {
    _loss_step: "chunked_lm_head_loss",
    _logprob_step: "chunked_lm_head_logprob",
    _support_step: "chunked_lm_head_support_logprob",
}


def test_device_default_is_the_deterministic_call_bit_for_bit_with_no_extra_allocation_or_sync(
    monkeypatch,
):
    """For each entry and form, every default call gives the deterministic entry's bits, and the dispatcher itself --
    the call measured around the deterministic body it runs -- allocates nothing and synchronizes nothing (the
    caching allocator's count and PyTorch's synchronization debug mode), in every one of the measured calls."""
    _require_device()
    for name, step, inp, kw in _cases():
        det_out = step(getattr(public, _DET_OF[step]), inp, **kw)
        outs, outside = _measure_dispatch(monkeypatch, _NEW_OF[step], step, inp, kw)
        for out in outs:
            for key, value in det_out.items():
                assert torch.equal(_bits(value), _bits(out[key])), (name, key)
        print(f"\n{name}: dispatcher (allocations, syncs) per call {outside}")
        assert all(cost == (0, 0) for cost in outside), (name, outside)


@pytest.mark.parametrize("step", [_loss_step, _logprob_step], ids=["loss", "logprob"])
def test_device_det_fallback_runs_a_k_slices_pin_above_four_without_it(
    monkeypatch, step
):
    """On the device: under ``torch.use_deterministic_algorithms(True)`` a ``deterministic=False`` call pinning
    ``{"dx": {"k_slices": 8, "tile_n": 256}}`` runs the real deterministic entry without the pin -- one warning names
    it -- instead of raising: the deterministic call with ``{"dx": {"tile_n": 256}}``, bit for bit."""
    _require_device()
    monkeypatch.setattr(public, "_DET_FALLBACK_WARNED", set())
    monkeypatch.setattr(public, "_DET_FALLBACK_PIN_WARNED", set())
    inp = make_inputs(4097, seed=SEED + 9, device=torch.device("cuda"))
    want = step(
        getattr(public, _DET_OF[step]), inp, gemm_tuning={"dx": {"tile_n": 256}}
    )
    was, was_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            got = step(
                getattr(public, _NEW_OF[step]),
                inp,
                gemm_tuning={"dx": {"k_slices": 8, "tile_n": 256}},
                deterministic=False,
            )
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)
    dropped = [w for w in rec if "k_slices=8 is dropped" in str(w.message)]
    assert len(dropped) == 1, [str(w.message) for w in rec]
    for key, value in want.items():
        assert torch.equal(_bits(value), _bits(got[key])), key


def test_device_support_kl_default_is_the_deterministic_call():
    _require_device()
    from tests.test_helpers.cake_lm_head_support_reference import make_support_inputs

    inp = make_support_inputs(
        257,
        seed=SEED + 5,
        H=256,
        V=512,
        k_cap=8,
        device=torch.device("cuda"),
        teacher=True,
    )
    outs = []
    for fn in (
        public.chunked_lm_head_det_support_kl,
        public.chunked_lm_head_support_kl,
    ):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        kl = fn(X, W, inp.support_ids, inp.support_offsets, inp.teacher, **inp.kwargs())
        kl.backward(inp.e)
        outs.append((kl.detach(), X.grad, W.grad))
    for a, b in zip(*outs, strict=True):
        assert torch.equal(_bits(a), _bits(b))


def test_device_public_entry_places_the_gradients(monkeypatch):
    """Through the public entry with ``deterministic=False`` on the generated program, the non-deterministic package's
    gradient placement ``"auto"`` (its default): a call under ``torch.no_grad`` launches no dX / dW GEMM and gives the
    plain call's ``loss`` / ``logp`` bits; a non-reentrant checkpointed step launches the plain step's dX / dW GEMMs
    once and gives its ``loss`` / ``logp`` / ``dX`` / ``dW`` bits (one dX slice pinned: the same bits run to run)."""
    _require_device()
    nd_backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_loss.cake_backend"
    )
    if not nd_backend.generated_program_available(torch.device("cuda"), entry="loss"):
        pytest.skip(
            "the non-deterministic package's programs are not registered for this device"
        )
    inp = make_inputs(
        8700, seed=SEED + 8700, device=torch.device("cuda"), ignore_frac=0.05
    )
    launched: list = []
    real_run_keys = nd_backend._run_keys

    def spy_run_keys(plan, device_index, keys, run_key, *args, **kwargs):
        def run(key):
            launched.append(key[0])
            return run_key(key)

        return real_run_keys(plan, device_index, keys, run, *args, **kwargs)

    monkeypatch.setattr(nd_backend, "_run_keys", spy_run_keys)
    assert nd_backend.gradient_placement() == "auto"

    def gradient_gemms():
        return sum(1 for s in launched if str(s).startswith(("gemm_dx", "gemm_dw")))

    def step(kind):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        logps = []

        def head(a, b):
            loss, logp = public.chunked_lm_head_loss(
                a,
                b,
                inp.labels,
                objective="ce",
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
                gemm_tuning={"dx": {"k_slices": 1}},
                deterministic=False,
            )
            logps.append(logp)
            return loss

        launched.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", api_logging.ExperimentalWarning)
            if kind == "no_grad":
                with torch.no_grad():
                    loss = head(X, W)
                torch.cuda.synchronize()
                return dict(loss=loss, logp=logps[0]), gradient_gemms()
            if kind == "plain":
                loss = head(X, W)
            else:
                loss = torch.utils.checkpoint.checkpoint(
                    head, X, W, use_reentrant=False
                )
            loss.backward()
        torch.cuda.synchronize()
        out = dict(loss=loss.detach(), logp=logps[0], dX=X.grad, dW=W.grad)
        return out, gradient_gemms()

    step("plain")  # binds
    plain, count = step("plain")
    assert count >= 5, launched
    nograd, none = step("no_grad")
    assert none == 0, launched
    ckpt, once = step("nonreentrant")
    assert once == count
    for key in ("loss", "logp"):
        assert torch.equal(_bits(nograd[key]), _bits(plain[key])), key
    for key in ("loss", "logp", "dX", "dW"):
        assert torch.equal(_bits(ckpt[key]), _bits(plain[key])), key


def test_keep_logits_mode_reaches_both_routes(recorders):
    """``memory_mode="keep_logits"`` passes through the log-probability dispatcher unchanged on both routes:
    ``deterministic=True`` hands it to the deterministic entry with the call's other arguments, ``deterministic=False``
    to this package's opt-in module (with ``deterministic=False``), and while torch's deterministic algorithms are on
    the ``False`` route's fallback hands it to the deterministic entry."""
    bodies, fake = recorders
    X, W, labels = _host_call()
    public.chunked_lm_head_logprob(
        X, W, labels, chunk_size=16, memory_mode="keep_logits"
    )
    ((args, kw),) = bodies["chunked_lm_head_logprob"].calls
    assert args == (X, W, labels)
    assert kw == dict(chunk_size=16, memory_mode="keep_logits")
    public.chunked_lm_head_logprob(
        X, W, labels, chunk_size=16, memory_mode="keep_logits", deterministic=False
    )
    ((args, kw),) = fake["opt_in"].recorders["chunked_lm_head_logprob"].calls
    assert args == (X, W, labels)
    assert kw["memory_mode"] == "keep_logits" and kw["deterministic"] is False
    assert kw["chunk_size"] == 16 and kw["num_valid_rows"] is None
    assert "chunked_lm_head_logprob" not in fake["backend"].recorders
    was, was_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            public.chunked_lm_head_logprob(
                X,
                W,
                labels,
                chunk_size=16,
                memory_mode="keep_logits",
                deterministic=False,
            )
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)
    assert len(bodies["chunked_lm_head_logprob"].calls) == 2
    assert bodies["chunked_lm_head_logprob"].calls[1][1]["memory_mode"] == "keep_logits"
    assert len(fake["opt_in"].recorders["chunked_lm_head_logprob"].calls) == 1


def test_device_keep_logits_on_both_routes_is_the_plain_call(monkeypatch):
    """``deterministic=True`` and ``deterministic=False`` both accept ``memory_mode="keep_logits"`` through the public
    log-probability entry and run their package's kept entry (``chunked_lm_head_det_logprob_keep`` /
    ``chunked_lm_head_logprob_keep``, counted); each gives its plain call's ``logp`` / ``dW`` bits and the bits of
    the loss ``sum(dlogp * logp)`` -- and ``dX``: the deterministic route's always, the non-deterministic route's
    with one dX slice pinned (its plain store; the default slice rule's reduction adds the same products in an
    unordered order)."""
    _require_device()
    nd_backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_loss.cake_backend"
    )
    det_backend = importlib.import_module(
        "flashinfer.experimental.cake_lm_head_det_loss.cake_backend"
    )
    if not nd_backend.generated_program_available(
        torch.device("cuda"), entry="logprob"
    ) or not det_backend.generated_program_available(
        torch.device("cuda"), entry="logprob"
    ):
        pytest.skip("the log-probability programs are not registered for this device")
    counts = {"det": 0, "nd": 0}
    for name, module, attr in (
        ("det", det_backend, "chunked_lm_head_det_logprob_keep"),
        ("nd", nd_backend, "chunked_lm_head_logprob_keep"),
    ):
        real = getattr(module, attr)

        def counted(*args, _real=real, _name=name, **kwargs):
            counts[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, attr, counted)
    inp = make_inputs(
        8700, seed=SEED + 8701, device=torch.device("cuda"), ignore_frac=0.05
    )

    def loss_of(out):
        return (out["logp"].double() * inp.dlogp.double())[inp.valid].sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", api_logging.ExperimentalWarning)
        for deterministic, tuning in (
            (True, None),
            (False, None),
            (False, {"dx": {"k_slices": 1}}),
        ):
            kw = dict(chunk_size=4096, deterministic=deterministic, gemm_tuning=tuning)
            plain = _logprob_step(public.chunked_lm_head_logprob, inp, **kw)
            before = dict(counts)
            kept = _logprob_step(
                public.chunked_lm_head_logprob, inp, memory_mode="keep_logits", **kw
            )
            route = "det" if deterministic else "nd"
            assert counts[route] == before[route] + 1, (deterministic, counts)
            assert (
                counts["det" if route == "nd" else "nd"]
                == before["det" if route == "nd" else "nd"]
            ), (deterministic, counts)
            keys = ["logp", "dW"] + (["dX"] if deterministic or tuning else [])
            for key in keys:
                assert torch.equal(_bits(kept[key]), _bits(plain[key])), (
                    deterministic,
                    tuning,
                    key,
                )
            assert torch.equal(
                _bits(loss_of(kept).float()), _bits(loss_of(plain).float())
            ), (deterministic, tuning)
            assert bool(torch.isfinite(kept["dX"].float()).all())
            assert torch.all(kept["dX"][~inp.valid] == 0)
