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

Public-API tests of the non-deterministic LM-head module ``flashinfer.cake_lm_head_nondet_train``: the entries
``cake_lm_head_nondet_train_{loss,logprob,preload,support_logprob,support_kl}`` are stable thin wrappers of the
backend ``flashinfer.experimental.cake.cake_lm_head_nondet_train`` with the deterministic entries' signatures
(``flashinfer.cake_lm_head_det_train``) and no ``deterministic`` keyword.  They run the backend with the same
argument checks and environment fallbacks, also while torch's deterministic algorithms are on (the backend's own
alert decides; no fallback, no warning from the facade, no deterministic module imported).  The names of the module
are checked by ``test_cake_lm_head_nondet_train_names.py``.  The host-layer tests replace the backend by recorders;
the device tests compare the bits of real calls.
"""

import importlib
import inspect
import itertools
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

import flashinfer.cake_lm_head_det_train as det_public
import flashinfer.cake_lm_head_nondet_train as public
from tests.experimental.test_helpers.cake_lm_head_loss_reference import (
    error_report,
    make_inputs,
    reference_fp64,
    reference_unchunked,
)

SEED = 20261031
H_HOST, V_HOST = 256, 512
# entry -> its deterministic entry
ENTRIES = {
    "cake_lm_head_nondet_train_loss": "cake_lm_head_det_train_loss",
    "cake_lm_head_nondet_train_logprob": "cake_lm_head_det_train_logprob",
    "cake_lm_head_nondet_train_preload": "cake_lm_head_det_train_preload",
    "cake_lm_head_nondet_train_support_logprob": "cake_lm_head_det_train_support_logprob",
    "cake_lm_head_nondet_train_support_kl": "cake_lm_head_det_train_support_kl",
}
SUPPORT = (
    "cake_lm_head_nondet_train_support_logprob",
    "cake_lm_head_nondet_train_support_kl",
)
LOADERS = ("_cake_backend", "_cake_backend_opt_in", "_cake_support")


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
    if entry == "cake_lm_head_nondet_train_loss":
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
    if entry == "cake_lm_head_nondet_train_logprob":
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
    if entry == "cake_lm_head_nondet_train_preload":
        return ("cuda:0",), {}
    X, W, ids, offsets = _support_args()
    if entry == "cake_lm_head_nondet_train_support_logprob":
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
    """A stand-in for a backend entry: records its calls, returns a marker."""

    def __init__(self, name):
        self.name, self.calls = name, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("result-of", self.name)


class _FakePackage:
    """A stand-in for a module of the backend: every attribute is a recorder."""

    def __init__(self):
        self.recorders = {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self.recorders.setdefault(name, _Recorder(name))


def _fail(*_args, **_kwargs):
    raise AssertionError("a call refused by argument binding must not reach this")


@pytest.fixture
def recorders(monkeypatch):
    """The backend replaced by recorders."""
    fake = {
        "backend": _FakePackage(),
        "opt_in": _FakePackage(),
        "support": _FakePackage(),
    }
    monkeypatch.setattr(public, "_cake_backend", lambda: fake["backend"])
    monkeypatch.setattr(public, "_cake_backend_opt_in", lambda: fake["opt_in"])
    monkeypatch.setattr(public, "_cake_support", lambda: fake["support"])
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


def _backend_calls(fake, entry):
    """The recorded calls of the backend entry that ``_call_args(entry)`` reaches."""
    if entry == "cake_lm_head_nondet_train_preload":
        return fake["opt_in"].recorders[entry].calls
    if entry in SUPPORT:
        return fake["support"].recorders[entry].calls
    # ``_call_args`` gives the loss call ``memory_mode="budget"`` and the log-probability call ``memory_mode="fast"``
    return fake["opt_in"].recorders[entry].calls


# ---------------------------------------------------------------------------
# Host layer
# ---------------------------------------------------------------------------


def test_entries_are_stable_apis():
    """Each entry is its own function (not the deterministic object), a stable API (no experimental marking), and
    the former dispatcher plumbing is gone."""
    for entry, det in ENTRIES.items():
        fn, det_fn = getattr(public, entry), getattr(det_public, det)
        assert fn is not det_fn, entry
        assert getattr(fn, "is_experimental", False) is False, entry
        assert fn.__name__ == entry and "SM100" in fn.__doc__, entry
    for name in (
        "_FEATURE",
        "_require_nd",
        "_raise_named",
        "_with_mode",
        "_nd_backend",
        "_nd_backend_opt_in",
        "_nd_support",
        "_nd_loss",
        "_nd_logprob",
        "_nd_support_logprob",
        "_nd_support_kl",
    ):
        assert not hasattr(public, name), name


def test_signatures_equal_the_deterministic_entries():
    """Each entry's signature is its deterministic entry's (names, kinds, defaults and annotations); neither has a
    ``deterministic`` parameter."""
    for entry, det in ENTRIES.items():
        params = list(inspect.signature(getattr(public, entry)).parameters.items())
        det_params = list(
            inspect.signature(getattr(det_public, det)).parameters.items()
        )
        assert params == det_params, entry
        assert "deterministic" not in dict(params), entry


@pytest.mark.parametrize("entry", list(ENTRIES))
@pytest.mark.parametrize("value", [True, False])
def test_deterministic_keyword_is_a_type_error_before_any_work(
    recorders, monkeypatch, entry, value
):
    """``deterministic`` (any value) is Python's own ``TypeError`` from argument binding, naming the entry -- before
    any knob resolution, torch-mode query or backend import (each patched to fail)."""
    fake = recorders
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", _fail)
    monkeypatch.setattr(public._knobs, "resolve_knobs", _fail)
    for loader in LOADERS:
        monkeypatch.setattr(public, loader, _fail)
    args, kwargs = _call_args(entry)
    with pytest.raises(
        TypeError,
        match=rf"^{entry}\(\) got an unexpected keyword argument 'deterministic'$",
    ):
        getattr(public, entry)(*args, **kwargs, deterministic=value)
    assert not any(f.recorders for f in fake.values())


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e != "cake_lm_head_nondet_train_preload"]
)
def test_argument_errors_name_the_public_entry(entry):
    """An unknown keyword or a missing argument names the public entry (both raise before any work: no device is
    needed)."""
    fn = getattr(public, entry)
    args = _support_args() if entry in SUPPORT else _host_call()
    with pytest.raises(
        TypeError, match=rf"^{entry}\(\) got an unexpected keyword argument 'bogus'$"
    ):
        fn(*args, bogus=1)
    with pytest.raises(TypeError, match=rf"^{entry}\(\) missing "):
        fn()


def test_entries_reach_the_backend(recorders, monkeypatch):
    """Each entry calls the backend's entry of the same name without a mode keyword, the switches resolved from the
    environment like the deterministic entries do, and the opt-in module for ``memory_mode`` /
    ``num_valid_rows``."""
    fake = recorders
    monkeypatch.setenv("FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS", "0")
    monkeypatch.setenv("FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM", "on")
    X, W, labels = _host_call()
    loss = "cake_lm_head_nondet_train_loss"
    logprob = "cake_lm_head_nondet_train_logprob"
    public.cake_lm_head_nondet_train_loss(X, W, labels, loss_div=2.0)
    ((args, kw),) = fake["backend"].recorders[loss].calls
    assert args == (X, W, labels)
    assert "deterministic" not in kw and kw["backend"] == "cake"
    assert kw["compact_rows"] is False and kw["dw_stream"] is True
    assert kw["loss_div"] == 2.0 and "memory_mode" not in kw
    public.cake_lm_head_nondet_train_loss(
        X, W, labels, loss_div=2.0, memory_mode="budget"
    )
    ((_, kw),) = fake["opt_in"].recorders[loss].calls
    assert kw["memory_mode"] == "budget" and "deterministic" not in kw
    public.cake_lm_head_nondet_train_logprob(X, W, labels, num_valid_rows=4)
    ((_, kw),) = fake["opt_in"].recorders[logprob].calls
    assert kw["num_valid_rows"] == 4 and "deterministic" not in kw
    public.cake_lm_head_nondet_train_logprob(X, W, labels)
    ((_, kw),) = fake["backend"].recorders[logprob].calls
    assert "deterministic" not in kw
    public.cake_lm_head_nondet_train_preload("cuda:0")
    assert fake["opt_in"].recorders["cake_lm_head_nondet_train_preload"].calls == [
        (("cuda:0",), {})
    ]
    for entry in SUPPORT:
        args, kwargs = _call_args(entry)
        getattr(public, entry)(*args, **kwargs)
        ((got_args, got_kw),) = fake["support"].recorders[entry].calls
        assert all(a is b for a, b in zip(got_args, args, strict=True))
        assert got_kw == kwargs


def test_entries_check_the_arguments_like_the_deterministic_entries(recorders):
    fake = recorders
    X, W, labels = _host_call()
    with pytest.raises(ValueError, match="backend"):
        public.cake_lm_head_nondet_train_loss(
            X, W, labels, loss_div=1.0, backend="reference"
        )
    with pytest.raises(ValueError, match="backend"):
        public.cake_lm_head_nondet_train_logprob(X, W, labels, backend="reference")
    with pytest.raises(ValueError, match="num_valid_rows"):
        public.cake_lm_head_nondet_train_loss(
            X, W, labels, loss_div=1.0, check_num_valid_rows=True
        )
    for entry in SUPPORT:
        args, kwargs = _call_args(entry)
        with pytest.raises(ValueError, match="backend"):
            getattr(public, entry)(*args, **dict(kwargs, backend="reference"))
    assert not any(f.recorders for f in fake.values())


@pytest.mark.parametrize("warn_only", [False, True])
def test_torch_deterministic_mode_still_runs_the_backend(recorders, warn_only):
    """With torch's deterministic algorithms on (strict or ``warn_only``), every entry still reaches the backend
    with the call's positional arguments and every keyword unchanged on every call, and the facade itself warns
    nothing (the backend's own alert decides; here the backend is a recorder, so no warning at all)."""
    fake = recorders
    was = _torch_deterministic(warn_only)
    try:
        for entry in ENTRIES:
            args, kwargs = _call_args(entry)
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                for _ in range(3):
                    getattr(public, entry)(*args, **kwargs)
            assert not [str(w.message) for w in rec], entry
            calls = _backend_calls(fake, entry)
            assert len(calls) == 3, entry
            for got_args, got_kw in calls:
                assert all(a is b for a, b in zip(got_args, args, strict=True)), entry
                assert got_kw == kwargs, entry
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])


@pytest.mark.parametrize(
    "entry", ["cake_lm_head_nondet_train_loss", "cake_lm_head_nondet_train_logprob"]
)
@pytest.mark.parametrize("source", ["keyword", "environment"])
def test_torch_deterministic_mode_passes_a_k_slices_pin_unchanged(
    recorders, monkeypatch, entry, source
):
    """Under torch's deterministic algorithms a dX ``k_slices`` pin of any size (``gemm_tuning`` or
    ``FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING``) reaches the backend unchanged, without a warning."""
    fake = recorders
    X, W, labels = _host_call()
    extra = dict(loss_div=1.0) if entry == "cake_lm_head_nondet_train_loss" else {}
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
            getattr(public, entry)(X, W, labels, **extra)
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])
    assert not rec
    ((_, kw),) = fake["backend"].recorders[entry].calls
    assert kw["gemm_tuning"] == pinned
    if source == "keyword":
        assert kw["gemm_tuning"] is pinned


def test_facade_with_fakes_under_strict_mode_imports_no_deterministic_module():
    """In a fresh interpreter with torch's strict deterministic algorithms on, every entry of the facade, with the
    backend replaced by recorders, reaches the backend, warns nothing and imports no deterministic LM-head module;
    the facade's source never reads torch's deterministic mode."""
    assert "are_deterministic_algorithms_enabled" not in inspect.getsource(public)
    repo = str(Path(__file__).resolve().parents[2])
    code = """
import json, sys, warnings
import torch
import flashinfer.cake_lm_head_nondet_train as public

class Fake:
    def __init__(self):
        self.calls = []
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append(name)

fake = Fake()
public._cake_backend = public._cake_backend_opt_in = public._cake_support = lambda: fake
X = torch.zeros(4, 256, dtype=torch.bfloat16)
W = torch.zeros(512, 256, dtype=torch.bfloat16)
labels = torch.zeros(4, dtype=torch.int64)
ids, offsets = torch.zeros(4, dtype=torch.int64), torch.arange(5, dtype=torch.int64)
torch.use_deterministic_algorithms(True)
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    public.cake_lm_head_nondet_train_loss(X, W, labels, loss_div=1.0)
    public.cake_lm_head_nondet_train_loss(X, W, labels, loss_div=1.0, memory_mode="budget")
    public.cake_lm_head_nondet_train_logprob(X, W, labels)
    public.cake_lm_head_nondet_train_logprob(X, W, labels, temperature=0.5)
    public.cake_lm_head_nondet_train_preload("cuda:0")
    public.cake_lm_head_nondet_train_support_logprob(X, W, ids, offsets, labels, k_cap=3)
    public.cake_lm_head_nondet_train_support_kl(X, W, ids, offsets, torch.zeros(4), k_cap=3)
det = sorted(m for m in sys.modules if "lm_head_det" in m)
print(json.dumps(dict(calls=fake.calls, warnings=[str(w.message) for w in rec], det=det)))
"""
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == dict(
        calls=[
            "cake_lm_head_nondet_train_loss",
            "cake_lm_head_nondet_train_loss",
            "cake_lm_head_nondet_train_logprob",
            "cake_lm_head_nondet_train_logprob_temperature",
            "cake_lm_head_nondet_train_preload",
            "cake_lm_head_nondet_train_support_logprob",
            "cake_lm_head_nondet_train_support_kl",
        ],
        warnings=[],
        det=[],
    ), result


def test_binding_caches_take_the_import_settings():
    """The backend's binding caches are configured from the variables read at import, like the deterministic
    backend's."""
    settings = public._BINDING_CACHE_SETTINGS
    for loader in (public._cake_backend, public._cake_backend_opt_in):
        module = loader()
        assert module.BINDING_CACHE.enabled == settings["enabled"]
        want = (
            module.BINDING_CACHE_DEFAULT_CAPACITY
            if settings["capacity"] is None
            else settings["capacity"]
        )
        assert module.BINDING_CACHE.capacity == want
    assert (
        public._cake_backend().__name__
        == "flashinfer.experimental.cake.cake_lm_head_nondet_train.cake_backend"
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


# test_cake_lm_head_nondet_train's accuracy gate, unchanged: an error at most GATE_MARGIN times the unchunked path's
# error against the FP64 oracle plus GATE_TINY
_GATE_MARGIN = 1.05
_GATE_TINY = dict(dX_rel_l2=1e-3, dW_rel_l2=1e-3)


def _logprob_step(fn, inp, **kw):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    logp = fn(X, W, inp.labels, **kw)
    logp.backward(inp.dlogp)
    return dict(logp=logp.detach(), dX=X.grad, dW=W.grad)


def test_device_public_entry_places_the_gradients(monkeypatch):
    """Through the public entry on the generated program, the backend's
    gradient placement ``"auto"`` (its default): a call under ``torch.no_grad`` launches no dX / dW GEMM and gives the
    plain call's ``loss`` / ``logp`` bits; a non-reentrant checkpointed step launches the plain step's dX / dW GEMMs
    once and gives its ``loss`` / ``logp`` / ``dX`` / ``dW`` bits (one dX slice pinned: the same bits run to run)."""
    _require_device()
    nd_backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_nondet_train.cake_backend"
    )
    if not nd_backend.generated_program_available(torch.device("cuda"), entry="loss"):
        pytest.skip("the backend's programs are not registered for this device")
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
            loss, logp = public.cake_lm_head_nondet_train_loss(
                a,
                b,
                inp.labels,
                objective="ce",
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
                gemm_tuning={"dx": {"k_slices": 1}},
            )
            logps.append(logp)
            return loss

        launched.clear()
        with warnings.catch_warnings():
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


def test_keep_logits_mode_reaches_the_opt_in_route(recorders):
    """``memory_mode="keep_logits"`` passes through the log-probability entry unchanged to the backend's opt-in
    module, also while torch's deterministic algorithms are on (no fallback)."""
    fake = recorders
    X, W, labels = _host_call()
    public.cake_lm_head_nondet_train_logprob(
        X, W, labels, chunk_size=16, memory_mode="keep_logits"
    )
    ((args, kw),) = fake["opt_in"].recorders["cake_lm_head_nondet_train_logprob"].calls
    assert args == (X, W, labels)
    assert kw["memory_mode"] == "keep_logits" and "deterministic" not in kw
    assert kw["chunk_size"] == 16 and kw["num_valid_rows"] is None
    assert "cake_lm_head_nondet_train_logprob" not in fake["backend"].recorders
    was = _torch_deterministic(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            public.cake_lm_head_nondet_train_logprob(
                X,
                W,
                labels,
                chunk_size=16,
                memory_mode="keep_logits",
            )
    finally:
        torch.use_deterministic_algorithms(was[0], warn_only=was[1])
    calls = fake["opt_in"].recorders["cake_lm_head_nondet_train_logprob"].calls
    assert len(calls) == 2 and calls[1][1]["memory_mode"] == "keep_logits"


def test_keep_logits_mode_leaves_the_hidden_count_to_the_kept_form(
    recorders, monkeypatch
):
    """With ``memory_mode="keep_logits"`` a ``hidden_count`` left at ``None`` reaches the opt-in module as ``None`` (the
    backend's kept-logits form decides) -- ``False`` when ``FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT`` is ``"0"`` -- and
    an explicit value as given; every other call resolves it from the variable as before (on unless ``"0"``)."""
    fake = recorders
    X, W, labels = _host_call()
    logprob = "cake_lm_head_nondet_train_logprob"

    def last_hidden_count(module, **kw):
        public.cake_lm_head_nondet_train_logprob(X, W, labels, chunk_size=16, **kw)
        return fake[module].recorders[logprob].calls[-1][1]["hidden_count"]

    assert last_hidden_count("opt_in", memory_mode="keep_logits") is None
    for value in (True, False):
        got = last_hidden_count("opt_in", memory_mode="keep_logits", hidden_count=value)
        assert got is value
    assert last_hidden_count("opt_in", memory_mode="fast") is True
    assert last_hidden_count("backend") is True
    monkeypatch.setenv("FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT", "0")
    assert last_hidden_count("opt_in", memory_mode="keep_logits") is False
    assert last_hidden_count("opt_in", memory_mode="fast") is False
    assert last_hidden_count("backend") is False
    monkeypatch.setenv("FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT", "1")
    assert last_hidden_count("opt_in", memory_mode="keep_logits") is None
    assert (
        last_hidden_count("opt_in", memory_mode="keep_logits", hidden_count=True)
        is True
    )


def test_device_keep_logits_on_both_routes_is_the_plain_call(monkeypatch):
    """``cake_lm_head_det_train_logprob`` and ``cake_lm_head_nondet_train_logprob`` both accept
    ``memory_mode="keep_logits"`` and run their package's kept entry (``cake_lm_head_det_train_logprob_keep`` /
    ``cake_lm_head_nondet_train_logprob_keep``, counted); each gives its plain call's ``logp`` bits and the bits of
    the loss ``sum(dlogp * logp)``, and ``dW`` / ``dX`` bits where both calls run the chunk loop: the deterministic
    route's always, the non-deterministic route's with one dX slice pinned (an explicit ``gemm_tuning``: its plain
    store) -- the non-deterministic route under its chunked kept form (``set_kept_logits_form(tail_fold=False,
    one_pass=False)``).  The non-deterministic plain call of more than one chunk otherwise runs the vocabulary-major
    backward and the kept call its own chunk loop (another summation order): the kept ``dX`` / ``dW`` meet the suite's
    unchanged gates against the FP64 oracle (``test_cake_lm_head_nondet_train``: ``GATE_MARGIN`` times the
    unchunked path's error plus ``GATE_TINY``).  On a three-chunk call the non-deterministic public route keeps one
    contiguous buffer -- with ``hidden_count=False`` (the valid rows counted on the host) on the default plan's chunks
    (``loop_chunks``: no fold at three chunks of ``T``), with ``hidden_count`` left at ``None`` (the kept form's
    default: the hidden count where eligible) on the shipped chunks -- and gives the plain call's ``logp`` and loss
    bits, its ``dX`` bits with one dX slice pinned (dX runs in the forward's chunks), and a ``dW`` (one GEMM over the
    kept rows) that repeats bit for bit; on a call of fewer than three chunks the default keeps one buffer on the
    folded chunks."""
    _require_device()
    nd_backend = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_nondet_train.cake_backend"
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
        ("nd", nd_backend, "cake_lm_head_nondet_train_logprob_keep"),
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
        for route, fn, tuning in (
            ("det", det_public.cake_lm_head_det_train_logprob, None),
            ("nd", public.cake_lm_head_nondet_train_logprob, None),
            ("nd", public.cake_lm_head_nondet_train_logprob, {"dx": {"k_slices": 1}}),
        ):
            kw = dict(chunk_size=4096, gemm_tuning=tuning)
            plain = _logprob_step(fn, inp, **kw)
            before = dict(counts)
            # the chunked kept form: the plain call's dW bits
            previous = nd_backend.set_kept_logits_form(tail_fold=False, one_pass=False)
            try:
                kept = _logprob_step(fn, inp, memory_mode="keep_logits", **kw)
            finally:
                nd_backend.set_kept_logits_form(**previous)
            other = "det" if route == "nd" else "nd"
            assert counts[route] == before[route] + 1, (route, counts)
            assert counts[other] == before[other], (route, counts)
            # the non-deterministic plain call of three chunks without a tuning runs the vocabulary-major backward
            vm = route == "nd" and tuning is None
            keys = (
                ["logp"]
                + ([] if vm else ["dW"])
                + (["dX"] if route == "det" or tuning else [])
            )
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
            if vm:
                oracle = reference_fp64(inp, entry="logprob")
                b0 = error_report(
                    reference_unchunked(inp, entry="logprob"), oracle, inp.labels
                )
                errors = error_report(kept, oracle, inp.labels)
                for key in ("dX_rel_l2", "dW_rel_l2"):
                    assert errors[key] <= _GATE_MARGIN * b0[key] + _GATE_TINY[key], (
                        key,
                        errors[key],
                        b0[key],
                    )
        # the kept form through the public route on a three-chunk call: the host count (hidden_count=False, the chunks
        # of the default plan's fold rule, ``loop_chunks``: the shipped chunks at three chunks of T) and the default
        # (hidden_count left at None: the hidden count, the shipped chunks)
        fn = public.cake_lm_head_nondet_train_logprob
        T_v = int(inp.valid.sum())
        shipped = nd_backend.plan_chunks(T_v, 4096)
        assert len(shipped) >= nd_backend.HIDDEN_COUNT_MIN_CHUNKS
        folded = tuple(r for _, r in nd_backend.loop_chunks(8700, T_v, 4096))
        for hidden_count, tuning in itertools.product(
            (False, None), (None, {"dx": {"k_slices": 1}})
        ):
            kw = dict(chunk_size=4096, gemm_tuning=tuning, hidden_count=hidden_count)
            plain = _logprob_step(fn, inp, chunk_size=4096, gemm_tuning=tuning)
            X = inp.X.detach().requires_grad_(True)
            W = inp.W.detach().requires_grad_(True)
            logp = fn(X, W, inp.labels, memory_mode="keep_logits", **kw)
            assert logp.grad_fn.kept_chunks == (
                folded if hidden_count is False else tuple(r for _, r in shipped)
            ), (hidden_count, tuning)
            logp.backward(inp.dlogp)
            kept = dict(logp=logp.detach(), dX=X.grad, dW=W.grad)
            del logp, X, W
            again = _logprob_step(fn, inp, memory_mode="keep_logits", **kw)
            assert torch.equal(_bits(kept["logp"]), _bits(plain["logp"])), tuning
            assert torch.equal(
                _bits(loss_of(kept).float()), _bits(loss_of(plain).float())
            ), tuning
            assert torch.equal(_bits(again["dW"]), _bits(kept["dW"])), tuning
            if tuning:
                assert torch.equal(_bits(kept["dX"]), _bits(plain["dX"]))
                assert torch.equal(_bits(again["dX"]), _bits(kept["dX"]))
            assert bool(torch.isfinite(kept["dW"].float()).all())
            assert bool(torch.isfinite(kept["dX"].float()).all())
            assert torch.all(kept["dX"][~inp.valid] == 0)
        # below three chunks the hidden count is not eligible: the default keeps one buffer on the folded chunks
        small = make_inputs(
            4600, seed=SEED + 4601, device=torch.device("cuda"), ignore_frac=0.05
        )
        T_s = int(small.valid.sum())
        X = small.X.detach().requires_grad_(True)
        W = small.W.detach().requires_grad_(True)
        logp = fn(X, W, small.labels, chunk_size=4096, memory_mode="keep_logits")
        assert logp.grad_fn.kept_chunks == tuple(
            r for _, r in nd_backend.loop_chunks(4600, T_s, 4096)
        )
        assert len(logp.grad_fn.kept_chunks) == 1


# ---------------------------------------------------------------------------
# Temperature of the log-probability entry
# ---------------------------------------------------------------------------


def test_temperature_reaches_the_backend_temperature_entry(recorders):
    """A temperature (a per-token tensor, a scalar) calls the backend's
    ``cake_lm_head_nondet_train_logprob_temperature`` -- the opt-in module's for ``memory_mode`` / ``num_valid_rows`` -- with
    ``temperature`` / ``check_temperature`` unchanged (a tensor as the same object) and every other keyword of the
    call without one; ``None`` and the scalars whose FP32 value is 1.0 call the entry without a temperature, with
    neither keyword; a bool is a ``ValueError``; ``cake_lm_head_nondet_train_preload(temperature=True)`` passes it
    to the opt-in module's ``cake_lm_head_nondet_train_preload``."""
    fake = recorders
    X, W, labels = _host_call()
    tau = torch.full((4,), 0.7)
    public.cake_lm_head_nondet_train_logprob(X, W, labels)
    ((_, plain),) = fake["backend"].recorders["cake_lm_head_nondet_train_logprob"].calls
    calls = fake["backend"].cake_lm_head_nondet_train_logprob_temperature.calls
    for value, check in ((tau, True), (0.7, False), (2, False)):
        calls.clear()
        public.cake_lm_head_nondet_train_logprob(
            X,
            W,
            labels,
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
        recorder = fake["opt_in"].cake_lm_head_nondet_train_logprob_temperature
        recorder.calls.clear()
        public.cake_lm_head_nondet_train_logprob(
            X,
            W,
            labels,
            temperature=tau,
            check_temperature=True,
            **opt,
        )
        ((_, kw),) = recorder.calls
        assert kw["temperature"] is tau and kw["check_temperature"] is True, opt
        assert "deterministic" not in kw, opt
        assert all(kw[k] == v for k, v in opt.items()), opt
    for value in (None, 1.0, 1, 1.0 + 1e-9):
        fake["backend"].recorders["cake_lm_head_nondet_train_logprob"].calls.clear()
        calls.clear()
        public.cake_lm_head_nondet_train_logprob(
            X, W, labels, temperature=value, check_temperature=True
        )
        ((_, kw),) = (
            fake["backend"].recorders["cake_lm_head_nondet_train_logprob"].calls
        )
        assert kw == plain and not calls, value
    with pytest.raises(ValueError, match="not a bool"):
        public.cake_lm_head_nondet_train_logprob(X, W, labels, temperature=True)
    preload = fake["opt_in"].cake_lm_head_nondet_train_preload
    preload.calls.clear()
    public.cake_lm_head_nondet_train_preload("cuda:0", temperature=True)
    public.cake_lm_head_nondet_train_preload("cuda:0")
    assert preload.calls == [
        (("cuda:0",), {"temperature": True}),
        (("cuda:0",), {}),
    ]


def test_temperature_refuses_keep_logits():
    """With ``memory_mode="keep_logits"`` the public entry passes a temperature to the
    backend's opt-in temperature entry, which rejects a scalar other than 1.0 and a per-token tensor with a
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
            public.cake_lm_head_nondet_train_logprob(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=16,
                memory_mode="keep_logits",
                temperature=temperature,
            )


def test_device_temperature_through_the_public_entry():
    """Through the public entry on the generated program: a heterogeneous per-token
    temperature and a scalar give the bits (integer views; one dX slice pinned) of the backend's temperature entry
    with the same argument, ``temperature=1.0`` the bits of the call without the keyword; with the default dX slices
    ``logp`` / ``dW`` are the deterministic entry's bits with the same temperature;
    ``cake_lm_head_nondet_train_preload(temperature=True)`` loads the temperature forms and a call after it
    gives the same bits."""
    _require_device()
    nd = importlib.import_module(
        "flashinfer.experimental.cake.cake_lm_head_nondet_train.cake_backend"
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

    for value in (inp.temperature, 0.7):
        same(
            run(public.cake_lm_head_nondet_train_logprob, temperature=value, **one),
            run(
                nd.cake_lm_head_nondet_train_logprob_temperature,
                temperature=value,
                **one,
            ),
            type(value).__name__,
        )
        same(
            run(public.cake_lm_head_nondet_train_logprob, temperature=value),
            run(det_public.cake_lm_head_det_train_logprob, temperature=value),
            ("vs deterministic", type(value).__name__),
            keys=("logp", "dW"),
        )
    plain = run(public.cake_lm_head_nondet_train_logprob, **one)
    same(
        plain,
        run(public.cake_lm_head_nondet_train_logprob, temperature=1.0, **one),
        "one",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        public.cake_lm_head_nondet_train_preload(temperature=True)
    same(
        run(
            nd.cake_lm_head_nondet_train_logprob_temperature,
            temperature=inp.temperature,
            **one,
        ),
        run(
            public.cake_lm_head_nondet_train_logprob, temperature=inp.temperature, **one
        ),
        "after preload",
    )
