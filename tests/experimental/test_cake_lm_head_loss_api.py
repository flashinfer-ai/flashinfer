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
logprob,preload,support_logprob,support_kl}``: experimental dispatchers of the non-deterministic package
``flashinfer.experimental.cake.cake_lm_head_loss`` with a keyword-only ``deterministic: bool = True``.  Only
``deterministic=False`` is accepted: it runs that package with the same argument checks and environment fallbacks,
also while torch's deterministic algorithms are on (the package's own alert decides; no fallback, no warning from the
dispatcher).  Any other value, the default ``True`` included, is a ``ValueError`` naming the deterministic entry
(``flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_*``) before any package import or torch query.  That the deterministic entries take no
``deterministic`` keyword is checked by ``test_cake_lm_head_det_train_names.py``.  The host-layer tests replace the package by recorders; the device tests
compare the bits of real calls.
"""

import importlib
import inspect
import json
import re
import warnings

import pytest
import torch
import torch.utils.checkpoint

import flashinfer.api_logging as api_logging
import flashinfer.cake_lm_head_det_train as det_public
import flashinfer.chunked_lm_head as public
from tests.experimental.test_helpers.cake_lm_head_loss_reference import make_inputs

SEED = 20261031
H_HOST, V_HOST = 256, 512
# det-free entry -> its deterministic entry
ENTRIES = {
    "chunked_lm_head_loss": "cake_lm_head_det_train_loss",
    "chunked_lm_head_logprob": "cake_lm_head_det_train_logprob",
    "chunked_lm_head_preload": "cake_lm_head_det_train_preload",
    "chunked_lm_head_support_logprob": "cake_lm_head_det_train_support_logprob",
    "chunked_lm_head_support_kl": "cake_lm_head_det_train_support_kl",
}
# The mode-free helper each dispatcher calls after its mode check (preload loads the opt-in module directly).
HELPERS = {
    "chunked_lm_head_loss": "_nd_loss",
    "chunked_lm_head_logprob": "_nd_logprob",
    "chunked_lm_head_support_logprob": "_nd_support_logprob",
    "chunked_lm_head_support_kl": "_nd_support_kl",
}
SUPPORT = ("chunked_lm_head_support_logprob", "chunked_lm_head_support_kl")
# A call that leaves ``deterministic`` at its default.
DEFAULT = object()


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
    raise AssertionError("a call refused by the mode check must not reach this")


@pytest.fixture
def recorders(monkeypatch):
    """The non-deterministic package replaced by recorders."""
    fake = {
        "backend": _FakePackage(),
        "opt_in": _FakePackage(),
        "support": _FakePackage(),
    }
    monkeypatch.setattr(public, "_nd_backend", lambda: fake["backend"])
    monkeypatch.setattr(public, "_nd_backend_opt_in", lambda: fake["opt_in"])
    monkeypatch.setattr(public, "_nd_support", lambda: fake["support"])
    for name in (
        "FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT",
        "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING",
    ):
        monkeypatch.delenv(name, raising=False)
    return fake


def _torch_deterministic(warn_only):
    """Turn torch's deterministic algorithms on; returns the previous ``(mode, warn_only)``."""
    was = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    return was


def _nd_calls(fake, entry):
    """The recorded calls of the package entry that ``_call_args(entry)`` reaches with ``deterministic=False``."""
    if entry == "chunked_lm_head_preload":
        return fake["opt_in"].recorders["preload"].calls
    if entry in SUPPORT:
        return fake["support"].recorders[entry].calls
    # ``_call_args`` gives the loss call ``memory_mode="budget"`` and the log-probability call ``memory_mode="fast"``
    return fake["opt_in"].recorders[entry].calls


# ---------------------------------------------------------------------------
# Host layer
# ---------------------------------------------------------------------------


def test_entries_are_experimental_dispatchers():
    """Each det-free entry is its own experimental function (not the deterministic object) with its module's
    feature; every deterministic entry is a stable API (no experimental marking)."""
    for entry, det in ENTRIES.items():
        fn, det_fn = getattr(public, entry), getattr(det_public, det)
        assert fn is not det_fn, entry
        assert fn.is_experimental is True, entry
        assert getattr(det_fn, "is_experimental", False) is False, det
        assert fn.experimental_feature == public._FEATURE, entry
        assert fn.__name__ == entry and "SM100" in fn.__doc__, entry
    for name in (
        "_det_loss",
        "_det_logprob",
        "_det_preload",
        "_det_support_logprob",
        "_det_support_kl",
        "_runs_det",
        "_det_fallback_kwargs",
        "_DET_FALLBACK_WARNED",
        "_DET_FALLBACK_PIN_WARNED",
        "_DET_DX_K_SLICES_MAX",
        "_caller_stacklevel",
    ):
        assert not hasattr(public, name), name


def test_signatures_mirror_the_deterministic_entries():
    """Each dispatcher's signature is its deterministic entry's plus the keyword-only ``deterministic: bool = True``
    (displayed as before; only ``False`` is accepted); no deterministic entry has that keyword."""
    for entry, det in ENTRIES.items():
        params = dict(inspect.signature(getattr(public, entry)).parameters)
        det_params = dict(inspect.signature(getattr(det_public, det)).parameters)
        mode = params.pop("deterministic")
        assert mode.kind is inspect.Parameter.KEYWORD_ONLY and mode.default is True
        assert mode.annotation in (bool, "bool"), entry
        assert "deterministic" not in det_params, det
        assert list(params.items()) == list(det_params.items()), entry


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize(
    "value",
    [DEFAULT, True, None, 0, 1, "False", "True", 1.0],
    ids=["default", "True", "None", "0", "1", "str-False", "str-True", "1.0"],
)
def test_any_mode_but_false_is_a_value_error_before_any_work(
    recorders, monkeypatch, entry, value
):
    """The default ``deterministic=True`` and every other value but ``False`` raise ``ValueError`` naming the
    deterministic entry and ``deterministic=False`` -- before any knob resolution, torch-mode query or package
    import (each patched to fail), and before the non-deterministic helper binds the call's other arguments."""
    fake = recorders
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", _fail)
    monkeypatch.setattr(public._knobs, "resolve_knobs", _fail)
    for loader in ("_nd_backend", "_nd_backend_opt_in", "_nd_support"):
        monkeypatch.setattr(public, loader, _fail)
    for helper in HELPERS.values():
        monkeypatch.setattr(public, helper, _fail)
    args, kwargs = _call_args(entry)
    given = kwargs if value is DEFAULT else dict(kwargs, deterministic=value)
    det = ENTRIES[entry]
    got = re.escape(repr(True if value is DEFAULT else value))
    pattern = (
        rf"^{entry}\(\) runs the non-deterministic package "
        r"flashinfer\.experimental\.cake\.cake_lm_head_loss only; "
        r"pass deterministic=False for the non-deterministic path "
        rf"\(got deterministic={got}\); "
        rf"the deterministic entry is flashinfer\.cake_lm_head_det_train\.{det}$"
    )
    with pytest.raises(ValueError, match=pattern):
        getattr(public, entry)(*args, **given)
    if entry != "chunked_lm_head_preload":  # preload binds its own signature first
        # the mode check comes before the helper binds the call: an unknown keyword is not reported
        with pytest.raises(ValueError, match=re.escape(det)):
            getattr(public, entry)(*args, bogus=1, **given)
    assert not any(f.recorders for f in fake.values())


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e != "chunked_lm_head_preload"]
)
def test_argument_errors_name_the_public_entry(entry):
    """With ``deterministic=False``, an unknown keyword or a missing argument names the public entry, not the private
    non-deterministic helper (both raise before any work: no device is needed)."""
    fn = getattr(public, entry)
    args = _support_args() if entry in SUPPORT else _host_call()
    with pytest.raises(
        TypeError, match=rf"^{entry}\(\) got an unexpected keyword argument 'bogus'$"
    ):
        fn(*args, bogus=1, deterministic=False)
    with pytest.raises(TypeError, match=rf"^{entry}\(\) missing "):
        fn(deterministic=False)


def test_false_reaches_the_non_deterministic_package(recorders, monkeypatch):
    """``deterministic=False`` calls the package's entry of the same name with ``deterministic=False``, the
    switches resolved from the environment like the deterministic entries do, and the opt-in module for
    ``memory_mode`` / ``num_valid_rows``."""
    fake = recorders
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


def test_false_checks_the_arguments_like_the_deterministic_entries(recorders):
    fake = recorders
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
def test_torch_deterministic_mode_still_runs_the_non_deterministic_package(
    recorders, warn_only
):
    """With torch's deterministic algorithms on (strict or ``warn_only``), ``deterministic=False`` still reaches the
    non-deterministic package with the call's arguments on every call, and the dispatcher itself warns nothing (the
    package's own alert decides; here the package is a recorder, so no warning at all but the experimental one)."""
    fake = recorders
    was = _torch_deterministic(warn_only)
    try:
        for entry in ENTRIES:
            args, kwargs = _call_args(entry)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                for _ in range(3):
                    getattr(public, entry)(*args, **kwargs, deterministic=False)
            other = [
                str(w.message)
                for w in rec
                if not issubclass(w.category, api_logging.ExperimentalWarning)
            ]
            assert not other, (entry, other)
            calls = _nd_calls(fake, entry)
            assert len(calls) == 3, entry
            for got_args, _ in calls:
                assert all(a is b for a, b in zip(got_args, args, strict=True)), entry
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])


@pytest.mark.parametrize("entry", ["chunked_lm_head_loss", "chunked_lm_head_logprob"])
@pytest.mark.parametrize("source", ["keyword", "environment"])
def test_torch_deterministic_mode_passes_a_k_slices_pin_unchanged(
    recorders, monkeypatch, entry, source
):
    """Under torch's deterministic algorithms a dX ``k_slices`` pin of any size (``gemm_tuning`` or
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING``) reaches the non-deterministic package unchanged, without a
    warning."""
    fake = recorders
    X, W, labels = _host_call()
    extra = dict(loss_div=1.0) if entry == "chunked_lm_head_loss" else {}
    pinned = {"dx": {"k_slices": 8, "tile_n": 256}, "logits": {"group_m": 8}}
    if source == "environment":
        monkeypatch.setenv(
            "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING", json.dumps(pinned)
        )
    else:
        extra = dict(extra, gemm_tuning=pinned)
    was = _torch_deterministic(False)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            getattr(public, entry)(X, W, labels, deterministic=False, **extra)
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])
    assert not [
        w for w in rec if not issubclass(w.category, api_logging.ExperimentalWarning)
    ]
    ((_, kw),) = fake["backend"].recorders[entry].calls
    assert kw["gemm_tuning"] == pinned
    if source == "keyword":
        assert kw["gemm_tuning"] is pinned


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
# Device
# ---------------------------------------------------------------------------


def _require_device():
    det_backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_det_train.cake_backend"
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in (
        det_backend.SUPPORTED_COMPUTE_CAPABILITIES
    ):
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not det_backend.generated_program_available(torch.device("cuda"), entry="loss"):
        pytest.skip(
            "the deterministic package's programs are not registered for this device"
        )


def _bits(t):
    t = t.detach().reshape(-1).contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _logprob_step(fn, inp, **kw):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    logp = fn(X, W, inp.labels, **kw)
    logp.backward(inp.dlogp)
    return dict(logp=logp.detach(), dX=X.grad, dW=W.grad)


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


def test_keep_logits_mode_reaches_the_non_deterministic_route(recorders):
    """``memory_mode="keep_logits"`` passes through the log-probability dispatcher unchanged: ``deterministic=False``
    hands it to this package's opt-in module (with ``deterministic=False``), also while torch's deterministic
    algorithms are on (no fallback); the default is a ``ValueError`` naming ``cake_lm_head_det_train_logprob``, which
    takes the mode itself."""
    fake = recorders
    X, W, labels = _host_call()
    with pytest.raises(ValueError, match="cake_lm_head_det_train_logprob"):
        public.chunked_lm_head_logprob(
            X, W, labels, chunk_size=16, memory_mode="keep_logits"
        )
    public.chunked_lm_head_logprob(
        X, W, labels, chunk_size=16, memory_mode="keep_logits", deterministic=False
    )
    ((args, kw),) = fake["opt_in"].recorders["chunked_lm_head_logprob"].calls
    assert args == (X, W, labels)
    assert kw["memory_mode"] == "keep_logits" and kw["deterministic"] is False
    assert kw["chunk_size"] == 16 and kw["num_valid_rows"] is None
    assert "chunked_lm_head_logprob" not in fake["backend"].recorders
    was = _torch_deterministic(True)
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
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])
    calls = fake["opt_in"].recorders["chunked_lm_head_logprob"].calls
    assert len(calls) == 2 and calls[1][1]["memory_mode"] == "keep_logits"


def test_device_keep_logits_on_both_routes_is_the_plain_call(monkeypatch):
    """``cake_lm_head_det_train_logprob`` and ``chunked_lm_head_logprob(deterministic=False)`` both accept
    ``memory_mode="keep_logits"`` and run their package's kept entry (``cake_lm_head_det_train_logprob_keep`` /
    ``chunked_lm_head_logprob_keep``, counted); each gives its plain call's ``logp`` / ``dW`` bits and the bits of
    the loss ``sum(dlogp * logp)`` -- and ``dX``: the deterministic route's always, the non-deterministic route's
    with one dX slice pinned (its plain store; the default slice rule's reduction adds the same products in an
    unordered order)."""
    _require_device()
    nd_backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_loss.cake_backend"
    )
    det_backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_det_train.cake_backend"
    )
    if not nd_backend.generated_program_available(
        torch.device("cuda"), entry="logprob"
    ) or not det_backend.generated_program_available(
        torch.device("cuda"), entry="logprob"
    ):
        pytest.skip("the log-probability programs are not registered for this device")
    counts = {"det": 0, "nd": 0}
    for name, module, attr in (
        ("det", det_backend, "cake_lm_head_det_train_logprob_keep"),
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
        for route, fn, mode, tuning in (
            ("det", det_public.cake_lm_head_det_train_logprob, {}, None),
            ("nd", public.chunked_lm_head_logprob, dict(deterministic=False), None),
            (
                "nd",
                public.chunked_lm_head_logprob,
                dict(deterministic=False),
                {"dx": {"k_slices": 1}},
            ),
        ):
            kw = dict(chunk_size=4096, gemm_tuning=tuning, **mode)
            plain = _logprob_step(fn, inp, **kw)
            before = dict(counts)
            kept = _logprob_step(fn, inp, memory_mode="keep_logits", **kw)
            other = "det" if route == "nd" else "nd"
            assert counts[route] == before[route] + 1, (route, counts)
            assert counts[other] == before[other], (route, counts)
            keys = ["logp", "dW"] + (["dX"] if route == "det" or tuning else [])
            for key in keys:
                assert torch.equal(_bits(kept[key]), _bits(plain[key])), (
                    route,
                    tuning,
                    key,
                )
            assert torch.equal(
                _bits(loss_of(kept).float()), _bits(loss_of(plain).float())
            ), (route, tuning)
            assert bool(torch.isfinite(kept["dX"].float()).all())
            assert torch.all(kept["dX"][~inp.valid] == 0)


# ---------------------------------------------------------------------------
# Temperature of the log-probability entry with ``deterministic=False``
# ---------------------------------------------------------------------------


def test_false_temperature_reaches_the_non_deterministic_temperature_entry(recorders):
    """``deterministic=False`` with a temperature (a per-token tensor, a scalar) calls the package's
    ``chunked_lm_head_logprob_temperature`` -- the opt-in module's for ``memory_mode`` / ``num_valid_rows`` -- with
    ``temperature`` / ``check_temperature`` unchanged (a tensor as the same object) and every other keyword of the
    call without one; ``None`` and the scalars whose FP32 value is 1.0 call the entry without a temperature, with
    neither keyword; a bool is a ``ValueError``; ``chunked_lm_head_preload(temperature=True)`` passes it to the
    opt-in module's ``preload``."""
    fake = recorders
    X, W, labels = _host_call()
    tau = torch.full((4,), 0.7)
    public.chunked_lm_head_logprob(X, W, labels, deterministic=False)
    ((_, plain),) = fake["backend"].recorders["chunked_lm_head_logprob"].calls
    calls = fake["backend"].chunked_lm_head_logprob_temperature.calls
    for value, check in ((tau, True), (0.7, False), (2, False)):
        calls.clear()
        public.chunked_lm_head_logprob(
            X,
            W,
            labels,
            deterministic=False,
            temperature=value,
            check_temperature=check,
        )
        ((args, kw),) = calls
        assert args == (X, W, labels)
        assert kw["temperature"] is value and kw["check_temperature"] is check
        assert {k: v for k, v in kw.items() if "temperature" not in k} == plain
    for opt in (
        dict(memory_mode="budget"),
        dict(memory_mode="fast"),
        dict(num_valid_rows=3),
    ):
        recorder = fake["opt_in"].chunked_lm_head_logprob_temperature
        recorder.calls.clear()
        public.chunked_lm_head_logprob(
            X,
            W,
            labels,
            deterministic=False,
            temperature=tau,
            check_temperature=True,
            **opt,
        )
        ((_, kw),) = recorder.calls
        assert kw["temperature"] is tau and kw["check_temperature"] is True, opt
        assert kw["deterministic"] is False, opt
        assert all(kw[k] == v for k, v in opt.items()), opt
    for value in (None, 1.0, 1, 1.0 + 1e-9):
        fake["backend"].recorders["chunked_lm_head_logprob"].calls.clear()
        calls.clear()
        public.chunked_lm_head_logprob(
            X, W, labels, deterministic=False, temperature=value, check_temperature=True
        )
        ((_, kw),) = fake["backend"].recorders["chunked_lm_head_logprob"].calls
        assert kw == plain and not calls, value
    with pytest.raises(ValueError, match="not a bool"):
        public.chunked_lm_head_logprob(
            X, W, labels, deterministic=False, temperature=True
        )
    preload = fake["opt_in"].preload
    preload.calls.clear()
    public.chunked_lm_head_preload("cuda:0", deterministic=False, temperature=True)
    public.chunked_lm_head_preload("cuda:0", deterministic=False)
    assert preload.calls == [
        (("cuda:0",), {"temperature": True}),
        (("cuda:0",), {}),
    ]


def test_false_temperature_refuses_keep_logits():
    """With ``deterministic=False`` and ``memory_mode="keep_logits"`` the public entry passes a temperature to the
    package's opt-in temperature entry, which rejects a scalar other than 1.0 and a per-token tensor with a
    ``ValueError`` before any launch."""
    inp = make_inputs(
        37,
        seed=SEED + 37,
        H=H_HOST,
        V=V_HOST,
        device=torch.device("cpu"),
        ignore_frac=0.2,
        temperature="hetero",
    )
    for temperature in (0.5, inp.temperature):
        with pytest.raises(ValueError, match="takes no temperature"):
            public.chunked_lm_head_logprob(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=16,
                memory_mode="keep_logits",
                temperature=temperature,
                deterministic=False,
            )


def test_device_false_temperature_through_the_public_entry():
    """Through the public entry on the generated program with ``deterministic=False``: a heterogeneous per-token
    temperature and a scalar give the bits (integer views; one dX slice pinned) of the package's temperature entry
    with the same argument, ``temperature=1.0`` the bits of the call without the keyword; with the default dX slices
    ``logp`` / ``dW`` are the deterministic entry's bits with the same temperature;
    ``chunked_lm_head_preload(deterministic=False, temperature=True)`` loads the temperature forms and a call after it
    gives the same bits."""
    _require_device()
    nd = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_loss.cake_backend"
    )
    name, _ = nd.record_for(torch.device("cuda"))
    if not set(nd.TEMPERATURE_STAGES) <= set(
        nd.registered_stages_with_temperature(name)
    ):
        pytest.skip("the registered program has no temperature forms")
    inp = make_inputs(
        4097, seed=SEED + 4097, device="cuda", ignore_frac=0.05, temperature="hetero"
    )
    one = dict(gemm_tuning={"dx": {"k_slices": 1}})

    def run(fn, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = _logprob_step(fn, inp, chunk_size=4096, backend="cake", **kw)
        torch.cuda.synchronize()
        return out

    def same(a, b, case, keys=("logp", "dX", "dW")):
        for key in keys:
            assert torch.equal(_bits(a[key]), _bits(b[key])), (case, key)

    nd_kw = dict(deterministic=False)
    for value in (inp.temperature, 0.7):
        same(
            run(public.chunked_lm_head_logprob, temperature=value, **one, **nd_kw),
            run(
                nd.chunked_lm_head_logprob_temperature,
                temperature=value,
                **one,
                **nd_kw,
            ),
            type(value).__name__,
        )
        same(
            run(public.chunked_lm_head_logprob, temperature=value, **nd_kw),
            run(det_public.cake_lm_head_det_train_logprob, temperature=value),
            ("vs deterministic", type(value).__name__),
            keys=("logp", "dW"),
        )
    plain = run(public.chunked_lm_head_logprob, **one, **nd_kw)
    same(
        plain,
        run(public.chunked_lm_head_logprob, temperature=1.0, **one, **nd_kw),
        "one",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        public.chunked_lm_head_preload(deterministic=False, temperature=True)
    same(
        run(
            nd.chunked_lm_head_logprob_temperature,
            temperature=inp.temperature,
            **one,
            **nd_kw,
        ),
        run(
            public.chunked_lm_head_logprob, temperature=inp.temperature, **one, **nd_kw
        ),
        "after preload",
    )
