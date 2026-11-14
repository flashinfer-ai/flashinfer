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

flashinfer public-API tests of the Cake chunked LM-head + loss package.

The stable entry points ``flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss`` and
``cake_lm_head_det_train_logprob`` are thin wrappers over
``flashinfer.experimental.cake.cake_lm_head_det_train.cake_backend``: they carry no experimental
marking and emit no ``ExperimentalWarning``, accept ``backend="cake"`` only and own FlashInfer's A/B environment variables
``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` -- a switch keyword left at ``None`` falls back to its
variable and the resolved value is passed down; the two binding-cache variables are
read when the module is imported and applied when it first reaches the backend.  The
tests here cover that wrapper: the (absent) marking, each variable's fallback and parsing, the
cache configuration and, on a supported device, the pass-down through the public
entry.  The backend tests live in ``test_cake_lm_head_det_train.py``, which imports the
package only through ``tests.cake.test_helpers.cake_package`` and reads no environment
variable.
"""

import importlib
import inspect
import warnings

import pytest
import torch
import torch.utils.checkpoint

import flashinfer._cake_lm_head_train_knobs as knobs
import flashinfer.cake_lm_head_det_train as public
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.cake_lm_head_det_train import (
    cake_lm_head_det_train_logprob,
    cake_lm_head_det_train_loss,
)
from flashinfer.experimental.cake.cake_lm_head_det_train import cake_backend
from flashinfer.experimental.cake.cake_lm_head_det_train.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    BindingCache,
    forward_binding_key,
    generated_program_available,
    stages_for_entry,
)
from tests.experimental.test_helpers.cake_lm_head_loss_reference import make_inputs

SEED = 20260930
# Host-layer shapes (the wrapper's own checks never reach a kernel).
H_HOST, V_HOST = 256, 512
# The switch keywords of both public entries and the variable each one falls back to.
ENV_OF = {
    "compact_rows": knobs.COMPACT_ROWS_ENV,
    "fuse_dw_cast": knobs.FUSE_DW_CAST_ENV,
    "dx_finalize": knobs.DX_FINALIZE_ENV,
    "dw_stream": knobs.DW_STREAM_ENV,
    "hidden_count": knobs.HIDDEN_COUNT_ENV,
    "gemm_tuning": knobs.GEMM_TUNING_ENV,
}
SWITCHES = tuple(ENV_OF)
CACHE_ENVS = (knobs.BINDING_CACHE_ENV, knobs.BINDING_CACHE_CAPACITY_ENV)


def _knobs(**explicit):
    """The keywords the wrapper passes down for a call whose switches are ``explicit`` (every other one ``None``)."""
    given = {name: None for name in SWITCHES}
    given.update(explicit)
    return knobs.resolve_knobs(**given)


def _clear(monkeypatch):
    for name in (*ENV_OF.values(), *CACHE_ENVS):
        monkeypatch.delenv(name, raising=False)


def _host_call():
    X = torch.zeros(4, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    labels = torch.zeros(4, dtype=torch.int64)
    return X, W, labels


def test_public_api_is_stable():
    """The entries are a stable API: no experimental marking and no ``ExperimentalWarning`` on a call."""
    assert getattr(cake_lm_head_det_train_loss, "is_experimental", False) is False
    assert getattr(cake_lm_head_det_train_logprob, "is_experimental", False) is False
    assert "SM100" in cake_lm_head_det_train_loss.__doc__
    assert "SM100" in cake_lm_head_det_train_logprob.__doc__
    X, W, labels = _host_call()
    # the public module serves the cake backend only
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        with pytest.raises(ValueError, match="backend"):
            cake_lm_head_det_train_loss(X, W, labels, loss_div=1.0, backend="reference")
    with pytest.raises(ValueError, match="backend"):
        cake_lm_head_det_train_logprob(X, W, labels, backend="reference")


def test_public_api_switch_keywords():
    """Both public entries take every switch as a keyword-only argument defaulting to ``None``; the loss entry's
    docstring states each default and the variable it falls back to; the backend entries take the same names (the
    wrapper passes them by keyword); the variables the wrapper names are exactly the ones these tests cover."""
    for fn in (cake_lm_head_det_train_loss, cake_lm_head_det_train_logprob):
        params = inspect.signature(fn).parameters
        for name in SWITCHES:
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
                fn.__name__,
                name,
            )
            assert params[name].default is None, (fn.__name__, name)
        assert "FLASHINFER_CAKE_LM_HEAD_LOSS_" in fn.__doc__
    for variable in (*ENV_OF.values(), *CACHE_ENVS):
        assert variable in cake_lm_head_det_train_loss.__doc__, variable
    for fn in (
        cake_backend.cake_lm_head_det_train_loss,
        cake_backend.cake_lm_head_det_train_logprob,
    ):
        params = inspect.signature(fn).parameters
        assert all(params[name].default is None for name in SWITCHES), fn.__name__
    named = {v for k, v in vars(knobs).items() if k.endswith("_ENV")}
    assert named == set(ENV_OF.values()) | set(CACHE_ENVS)
    assert all(v.startswith("FLASHINFER_CAKE_LM_HEAD_LOSS_") for v in named)


def test_public_api_memory_and_count_keywords(monkeypatch):
    """Both public entries take ``memory_mode`` (default ``None``, no environment variable), ``num_valid_rows``
    (``None``) and ``check_num_valid_rows`` (``False``) as keyword-only arguments and document them; a call with
    ``memory_mode`` or ``num_valid_rows`` passes all three to the opt-in backend module unchanged, and so does a call
    with neither whose ``W`` requires a gradient (with grad mode on its automatic memory form can pick a budget form;
    with it off its hidden valid-row count is checked against the step's bound); any other call with neither calls
    the default backend entry without them;
    ``cake_lm_head_det_train_preload`` is experimental and needs a CUDA device."""
    _clear(monkeypatch)
    for fn in (cake_lm_head_det_train_loss, cake_lm_head_det_train_logprob):
        params = inspect.signature(fn).parameters
        for name, default in (
            ("memory_mode", None),
            ("num_valid_rows", None),
            ("check_num_valid_rows", False),
        ):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
                fn.__name__,
                name,
            )
            assert params[name].default is default, (fn.__name__, name)
            assert name in fn.__doc__, (fn.__name__, name)
    seen = {}

    def spy(name):
        def call(*args, **kw):
            seen[name] = kw
            return torch.zeros(())

        return call

    backend, opt_in = public._cake_backend(), public._cake_backend_opt_in()
    for module, tag in ((backend, "default"), (opt_in, "opt_in")):
        monkeypatch.setattr(module, "cake_lm_head_det_train_loss", spy(f"{tag}_loss"))
        monkeypatch.setattr(
            module, "cake_lm_head_det_train_logprob", spy(f"{tag}_logprob")
        )
    X, W, labels = _host_call()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        cake_lm_head_det_train_loss(
            X,
            W,
            labels,
            loss_div=1.0,
            memory_mode="budget",
            num_valid_rows=3,
            check_num_valid_rows=True,
        )
        cake_lm_head_det_train_logprob(X, W, labels, num_valid_rows=3)
        cake_lm_head_det_train_loss(X, W, labels, loss_div=1.0)
        cake_lm_head_det_train_logprob(X, W, labels)
    # either keyword: the opt-in module, every keyword passed unchanged
    assert seen["opt_in_loss"]["memory_mode"] == "budget"
    assert seen["opt_in_loss"]["num_valid_rows"] == 3
    assert seen["opt_in_loss"]["check_num_valid_rows"] is True
    assert seen["opt_in_logprob"]["memory_mode"] is None
    assert seen["opt_in_logprob"]["num_valid_rows"] == 3
    assert seen["opt_in_logprob"]["check_num_valid_rows"] is False
    # neither: the default backend entry, called as without the keywords
    for name in ("default_loss", "default_logprob"):
        assert (
            not {"memory_mode", "num_valid_rows", "check_num_valid_rows"}
            & seen[name].keys()
        ), name
    # neither, with a ``W`` that requires a gradient: the opt-in module's automatic form (memory_mode None), also
    # without grad mode (a reentrant checkpoint's first pass: its hidden valid-row count is checked there)
    trained = W.detach().requires_grad_(True)
    for grad_mode in (True, False):
        seen.clear()
        with warnings.catch_warnings(), torch.set_grad_enabled(grad_mode):
            warnings.simplefilter("ignore", ExperimentalWarning)
            cake_lm_head_det_train_loss(X, trained, labels, loss_div=1.0)
            cake_lm_head_det_train_logprob(X, trained, labels)
        assert set(seen) == {"opt_in_loss", "opt_in_logprob"}, (grad_mode, seen)
        for name in ("opt_in_loss", "opt_in_logprob"):
            assert seen[name]["memory_mode"] is None, name
            assert seen[name]["num_valid_rows"] is None, name
            assert seen[name]["check_num_valid_rows"] is False, name
    # check_num_valid_rows=True without a count: ValueError before either backend entry runs
    seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for fn, kw in (
            (cake_lm_head_det_train_loss, dict(loss_div=1.0)),
            (cake_lm_head_det_train_logprob, {}),
        ):
            for mode in (None, "fast", "budget"):
                with pytest.raises(ValueError, match="needs num_valid_rows"):
                    fn(
                        X,
                        W,
                        labels,
                        memory_mode=mode,
                        check_num_valid_rows=True,
                        **kw,
                    )
    assert not seen, seen
    assert (
        getattr(public.cake_lm_head_det_train_preload, "is_experimental", False)
        is False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ExperimentalWarning)
        with pytest.raises(ValueError, match="CUDA device"):
            public.cake_lm_head_det_train_preload("cpu")


def test_public_api_temperature_keywords(monkeypatch):
    """The log-probability entry takes ``temperature`` (default ``None``, the same as ``1.0``: no temperature) and
    ``check_temperature`` (default ``False``) as documented keyword-only arguments, with no environment variable.
    Without a temperature it calls the backend entry (or, with ``memory_mode`` or ``num_valid_rows``, the opt-in
    module's entry) exactly as it does without these keywords; with one it calls that module's
    ``cake_lm_head_det_train_logprob_temperature`` and passes both unchanged (a tensor as the same object).  The backend
    entries keep their signatures; the loss entry has neither keyword; ``cake_lm_head_det_train_preload`` takes
    ``temperature`` (default ``False``: the temperature forms are not loaded) and passes it down."""
    _clear(monkeypatch)
    params = inspect.signature(cake_lm_head_det_train_logprob).parameters
    for name, default in (("temperature", None), ("check_temperature", False)):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params[name].default == default and type(params[name].default) is type(
            default
        ), name
        assert name in cake_lm_head_det_train_logprob.__doc__, name
    loss_params = inspect.signature(cake_lm_head_det_train_loss).parameters
    assert "temperature" not in loss_params and "check_temperature" not in loss_params
    for module in (cake_backend, public._cake_backend_opt_in()):
        assert (
            "temperature"
            not in inspect.signature(module.cake_lm_head_det_train_logprob).parameters
        )
        tau_params = inspect.signature(
            module.cake_lm_head_det_train_logprob_temperature
        ).parameters
        assert tau_params["check_temperature"].default is False
    preload = inspect.signature(public.cake_lm_head_det_train_preload).parameters
    assert preload["temperature"].kind is inspect.Parameter.KEYWORD_ONLY
    assert preload["temperature"].default is False
    assert "temperature" in public.cake_lm_head_det_train_preload.__doc__
    assert not any("TEMPERATURE" in k for k in vars(knobs) if k.endswith("_ENV"))
    seen = {}

    def spy(name):
        def call(*args, **kw):
            seen[name] = (args, kw)
            return torch.zeros(())

        return call

    backend = public._cake_backend()
    monkeypatch.setattr(backend, "cake_lm_head_det_train_logprob", spy("logprob"))
    monkeypatch.setattr(
        backend, "cake_lm_head_det_train_logprob_temperature", spy("logprob_tau")
    )
    opt_in = public._cake_backend_opt_in()
    monkeypatch.setattr(opt_in, "cake_lm_head_det_train_logprob", spy("opt_in_logprob"))
    monkeypatch.setattr(
        opt_in, "cake_lm_head_det_train_logprob_temperature", spy("opt_in_logprob_tau")
    )
    monkeypatch.setattr(opt_in, "cake_lm_head_det_train_preload", spy("preload"))
    X, W, labels = _host_call()
    tau = torch.full((4,), 0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        cake_lm_head_det_train_logprob(X, W, labels)
        assert "temperature" not in seen["logprob"][1]
        assert "check_temperature" not in seen["logprob"][1]
        assert "logprob_tau" not in seen
        plain = seen.pop("logprob")
        cake_lm_head_det_train_logprob(X, W, labels, temperature=None)
        assert seen.pop("logprob") == plain
        cake_lm_head_det_train_logprob(X, W, labels, temperature=1.0)
        assert seen["logprob_tau"][1]["temperature"] == 1.0
        assert {
            k: v for k, v in seen["logprob_tau"][1].items() if "temperature" not in k
        } == plain[1]
        cake_lm_head_det_train_logprob(
            X, W, labels, temperature=tau, check_temperature=True
        )
        assert seen["logprob_tau"][1]["temperature"] is tau
        assert seen["logprob_tau"][1]["check_temperature"] is True
        cake_lm_head_det_train_logprob(X, W, labels, temperature=0.7)
        assert seen["logprob_tau"][1]["temperature"] == 0.7
        assert "logprob" not in seen
        for kw in (
            dict(memory_mode="budget"),
            dict(memory_mode="fast"),
            dict(num_valid_rows=3),
        ):
            seen.pop("opt_in_logprob_tau", None)
            cake_lm_head_det_train_logprob(
                X, W, labels, temperature=tau, check_temperature=True, **kw
            )
            assert seen["opt_in_logprob_tau"][1]["temperature"] is tau, kw
            assert seen["opt_in_logprob_tau"][1]["check_temperature"] is True, kw
            seen.pop("opt_in_logprob_tau")
            cake_lm_head_det_train_logprob(X, W, labels, temperature=0.7, **kw)
            assert seen["opt_in_logprob_tau"][1]["temperature"] == 0.7, kw
            assert seen["opt_in_logprob_tau"][1]["check_temperature"] is False, kw
            seen.pop("opt_in_logprob_tau")
            seen.pop("opt_in_logprob", None)
            cake_lm_head_det_train_logprob(X, W, labels, **kw)
            assert "temperature" not in seen["opt_in_logprob"][1], kw
            assert "opt_in_logprob_tau" not in seen, kw
        public.cake_lm_head_det_train_preload(0, temperature=True)
        args, kw = seen["preload"]
        assert kw.get("temperature") is True and 0 in (*args, kw.get("device"))
        public.cake_lm_head_det_train_preload(0)
        assert not seen["preload"][1].get("temperature", False)


def test_public_api_temperature_refuses_keep_logits(monkeypatch):
    """With ``memory_mode="keep_logits"`` the public entry passes a temperature to the opt-in module's temperature
    entry, which rejects a scalar other than 1.0 and a per-token tensor with a ``ValueError`` before any launch."""
    _clear(monkeypatch)
    inp = make_inputs(
        37,
        seed=SEED + 37,
        H=256,
        V=512,
        device=torch.device("cpu"),
        ignore_frac=0.2,
        temperature="hetero",
    )
    for temperature in (0.5, inp.temperature):
        with pytest.raises(ValueError, match="takes no temperature"):
            cake_lm_head_det_train_logprob(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=16,
                memory_mode="keep_logits",
                temperature=temperature,
            )


def test_compact_rows_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["compact_rows"] is True  # unset: the backend's default
    monkeypatch.setenv(knobs.COMPACT_ROWS_ENV, "0")
    assert _knobs()["compact_rows"] is False
    assert _knobs(compact_rows=True)["compact_rows"] is True  # the keyword wins
    monkeypatch.setenv(knobs.COMPACT_ROWS_ENV, "1")
    assert _knobs()["compact_rows"] is True
    assert _knobs(compact_rows=False)["compact_rows"] is False
    monkeypatch.setenv(knobs.COMPACT_ROWS_ENV, "off")
    assert _knobs()["compact_rows"] is True  # "0" is the only off value


def test_fuse_dw_cast_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["fuse_dw_cast"] is True
    assert "gemm_dw_cast_bf16" in stages_for_entry(
        "loss", fuse_dw_cast=_knobs()["fuse_dw_cast"]
    )
    monkeypatch.setenv(knobs.FUSE_DW_CAST_ENV, "0")
    assert _knobs()["fuse_dw_cast"] is False
    assert "gemm_dw_cast_bf16" not in stages_for_entry(
        "loss", fuse_dw_cast=_knobs()["fuse_dw_cast"]
    )
    assert _knobs(fuse_dw_cast=True)["fuse_dw_cast"] is True  # the keyword wins
    monkeypatch.setenv(knobs.FUSE_DW_CAST_ENV, "1")
    assert _knobs()["fuse_dw_cast"] is True
    assert _knobs(fuse_dw_cast=False)["fuse_dw_cast"] is False


def test_dx_finalize_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["dx_finalize"] is True
    monkeypatch.setenv(knobs.DX_FINALIZE_ENV, "0")
    assert _knobs()["dx_finalize"] is False
    assert _knobs(dx_finalize=True)["dx_finalize"] is True  # the keyword wins
    monkeypatch.setenv(knobs.DX_FINALIZE_ENV, "1")
    assert _knobs()["dx_finalize"] is True
    assert _knobs(dx_finalize=False)["dx_finalize"] is False
    assert cake_backend._resolve_dx_finalize(_knobs()["dx_finalize"]) is True


def test_hidden_count_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["hidden_count"] is True
    monkeypatch.setenv(knobs.HIDDEN_COUNT_ENV, "0")
    assert _knobs()["hidden_count"] is False
    assert _knobs(hidden_count=True)["hidden_count"] is True  # the keyword wins
    monkeypatch.setenv(knobs.HIDDEN_COUNT_ENV, "1")
    assert _knobs()["hidden_count"] is True
    assert _knobs(hidden_count=False)["hidden_count"] is False
    assert cake_backend._resolve_hidden_count(_knobs()["hidden_count"]) is True


def test_dw_stream_default_env(monkeypatch):
    """The dW side-stream variable: unset / ``auto`` / anything else -> ``"auto"`` (calls of
    :data:`DW_STREAM_MIN_CHUNKS` chunks or more), ``1`` / ``true`` / ``on`` / ``yes`` -> ``True`` (every multi-chunk
    call), ``0`` / ``false`` / ``off`` / ``no`` -> ``False`` (never); the value reaches the backend's rule."""
    _clear(monkeypatch)
    assert _knobs()["dw_stream"] == "auto"
    assert cake_backend.dw_stream_mode(_knobs()["dw_stream"]) == "auto"
    assert [
        cake_backend.dw_side_stream(n, _knobs()["dw_stream"]) for n in (1, 2, 3, 8)
    ] == [False, False, True, True]
    for value in ("0", "false", "off", "no", " OFF "):
        monkeypatch.setenv(knobs.DW_STREAM_ENV, value)
        assert _knobs()["dw_stream"] is False
        assert not any(
            cake_backend.dw_side_stream(n, _knobs()["dw_stream"]) for n in (1, 2, 3, 8)
        )
    for value in ("1", "true", "on", "yes", " On "):
        monkeypatch.setenv(knobs.DW_STREAM_ENV, value)
        assert _knobs()["dw_stream"] is True
        assert [
            cake_backend.dw_side_stream(n, _knobs()["dw_stream"]) for n in (1, 2, 3)
        ] == [False, True, True]
    for value in ("auto", "AUTO", "", "anything-else"):
        monkeypatch.setenv(knobs.DW_STREAM_ENV, value)
        assert _knobs()["dw_stream"] == "auto"
        assert [
            cake_backend.dw_side_stream(n, _knobs()["dw_stream"]) for n in (2, 3)
        ] == [False, True]
    monkeypatch.setenv(knobs.DW_STREAM_ENV, "0")
    assert _knobs(dw_stream=True)["dw_stream"] is True  # the keyword wins
    assert _knobs(dw_stream="auto")["dw_stream"] == "auto"


def test_gemm_tuning_default_env(monkeypatch):
    """The GEMM-tuning variable: unset / blank -> no knob, a JSON object -> the knob dict (validated by the backend and
    part of the binding key), invalid JSON or a non-object -> a ``ValueError`` naming the variable, raised by the
    public entry before any backend work; the keyword wins over the variable."""
    _clear(monkeypatch)
    assert _knobs()["gemm_tuning"] is None
    assert cake_backend._resolve_gemm_tuning(_knobs()["gemm_tuning"]) == {}
    monkeypatch.setenv(knobs.GEMM_TUNING_ENV, "  ")
    assert _knobs()["gemm_tuning"] is None
    monkeypatch.setenv(
        knobs.GEMM_TUNING_ENV, '{"logits": {"group_m": 16}, "dx": {"tile_n": 256}}'
    )
    assert _knobs()["gemm_tuning"] == {"logits": {"group_m": 16}, "dx": {"tile_n": 256}}
    assert cake_backend._resolve_gemm_tuning(_knobs()["gemm_tuning"]) == {
        "logits": {"group_m": 16},
        "dx": {"tile_n": 256},
    }
    X, W, labels = _host_call()
    monkeypatch.setenv(knobs.GEMM_TUNING_ENV, "not json")
    with pytest.raises(ValueError, match="JSON") as info:
        _knobs()
    assert knobs.GEMM_TUNING_ENV in str(info.value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        with pytest.raises(ValueError, match=knobs.GEMM_TUNING_ENV):
            cake_lm_head_det_train_loss(X, W, labels, loss_div=1.0)
        with pytest.raises(ValueError, match=knobs.GEMM_TUNING_ENV):
            cake_lm_head_det_train_logprob(X, W, labels)
    monkeypatch.setenv(knobs.GEMM_TUNING_ENV, '["dx"]')
    with pytest.raises(ValueError, match="JSON object"):
        _knobs()
    monkeypatch.setenv(knobs.GEMM_TUNING_ENV, '{"dx": {"tile_n": 256}}')
    assert _knobs(gemm_tuning={})["gemm_tuning"] == {}  # the keyword wins
    common = dict(
        objective="ce",
        loss_div=1.0,
        infer_logp=None,
        loss_weights=None,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.bfloat16,
        entry="loss",
    )
    env_key = forward_binding_key(
        X, W, labels, gemm_tuning=_knobs()["gemm_tuning"], **common
    )  # the variable's knobs are part of the binding
    assert env_key == forward_binding_key(
        X, W, labels, gemm_tuning={"dx": {"tile_n": 256}}, **common
    )
    assert env_key != forward_binding_key(X, W, labels, gemm_tuning={}, **common)


def test_binding_cache_capacity_from_environment(monkeypatch):
    """The two binding-cache variables are read when ``flashinfer.cake_lm_head_det_train`` is imported (here: reloaded) and
    applied through ``BindingCache.configure`` on the backend's live cache when the reloaded module first reaches
    the backend (``_cake_backend()``, the first call of either entry point): unset -> enabled, 64 bindings; a
    positive integer sets the capacity; ``..._BINDING_CACHE=0`` disables it; anything but a positive integer is a
    ``ValueError`` at import that leaves the cache as it was.  The backend itself never reads them."""
    cache = cake_backend.BINDING_CACHE
    was = (cache.enabled, cache.capacity)
    try:
        _clear(monkeypatch)
        importlib.reload(public)
        public._cake_backend()
        # configured in place, never replaced
        assert cake_backend.BINDING_CACHE is cache
        assert cache.enabled and cache.capacity == 64
        assert cache.capacity == cake_backend.BINDING_CACHE_DEFAULT_CAPACITY
        monkeypatch.setenv(knobs.BINDING_CACHE_CAPACITY_ENV, "5")
        assert BindingCache().capacity == 64  # the backend does not read the variable
        importlib.reload(public)
        assert cache.capacity == 64  # read at import, applied on first use
        public._cake_backend()
        assert cache.capacity == 5 and cache.enabled
        monkeypatch.setenv(knobs.BINDING_CACHE_CAPACITY_ENV, " ")
        importlib.reload(public)
        public._cake_backend()
        assert cache.capacity == 64
        monkeypatch.setenv(knobs.BINDING_CACHE_ENV, "0")
        importlib.reload(public)
        public._cake_backend()
        assert not cache.enabled and len(cache) == 0 and cache.capacity == 64
        monkeypatch.setenv(knobs.BINDING_CACHE_ENV, "1")
        monkeypatch.setenv(knobs.BINDING_CACHE_CAPACITY_ENV, "3")
        importlib.reload(public)
        public._cake_backend()
        assert cache.enabled and cache.capacity == 3
        assert knobs.binding_cache_settings() == dict(enabled=True, capacity=3)
        for bad in ("0", "-1", "many", "2.5"):
            monkeypatch.setenv(knobs.BINDING_CACHE_CAPACITY_ENV, bad)
            with pytest.raises(ValueError, match="positive integer"):
                importlib.reload(public)
            # a failed import reads nothing to apply: the cache is as it was
            assert cache.enabled and cache.capacity == 3
    finally:
        monkeypatch.undo()
        importlib.reload(public)  # the process's real environment again
        public._cake_backend()
        cache.configure(enabled=was[0], capacity=was[1])


# ---------------------------------------------------------------------------
# Device layer (the generated program through the public entry)
# ---------------------------------------------------------------------------

CUDA = torch.device("cuda")


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The bit pattern of a float tensor as integers (bf16 -> int16, fp32 -> int32): equality of these views is
    bit-level equality (``torch.equal`` on floats equates +0 and -0)."""
    t = t.detach().reshape(-1).contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _require_program():
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    ):
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    for entry in ("loss", "logprob"):
        if not generated_program_available(CUDA, entry=entry):
            pytest.skip(
                f"generated chunked LM-head program ({entry} entry) not registered for this device"
            )


def test_device_env_switches_reach_the_backend(monkeypatch):
    """Through the public entry on the generated program: ``FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM=0`` moves the
    per-chunk weight-gradient accumulates back to the caller's stream (one stream entered instead of two on a
    three-chunk call) and ``FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST=0`` binds the accumulate + flat cast instead of
    the fused cast GEMM (its TMA-store form where :func:`dw_cast_tma_rule` routes the short last chunk); an explicit
    keyword wins over the variable; ``loss`` / ``logp`` / ``dX`` / ``dW`` are bitwise (integer views) the same under
    every setting.  The calls pin ``memory_mode="fast"`` (the default call of this shape takes a budget form, whose
    stages differ)."""
    _require_program()
    _clear(monkeypatch)
    inp = make_inputs(
        8700, seed=SEED + 8700, device=CUDA, ignore_frac=0.05
    )  # 8265 valid rows: three compact chunks, so the auto rule turns the side stream on
    assert int(inp.valid.sum()) == 8265
    streams: list = []
    bound: list = []
    real_stream = cake_backend._ffi_stream_context
    real_bind_all = cake_backend._bind_all

    def spy_stream(index):
        streams.append(torch.cuda.current_stream(index).cuda_stream)
        return real_stream(index)

    def spy_bind_all(record, module_name, keys, values, device, geometry):
        bound.extend(k[0] for k in keys)
        return real_bind_all(record, module_name, keys, values, device, geometry)

    monkeypatch.setattr(cake_backend, "_ffi_stream_context", spy_stream)
    monkeypatch.setattr(cake_backend, "_bind_all", spy_bind_all)
    public._cake_backend()  # the first use applies the import-time cache settings: before the patch
    monkeypatch.setattr(
        cake_backend.BINDING_CACHE, "enabled", False
    )  # every call binds, so the bound stages show the switch

    def run(**kw):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        streams.clear()
        bound.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            loss, logp = cake_lm_head_det_train_loss(
                X,
                W,
                inp.labels,
                objective="ce",
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
                memory_mode="fast",
                **kw,
            )
        loss.backward()
        torch.cuda.synchronize()
        out = dict(loss=loss.detach(), logp=logp, dX=X.grad, dW=W.grad)
        return out, len(set(streams)), list(bound)

    # the fused cast's form on the deferred last chunk (73 rows): the TMA store where dw_cast_tma_rule routes it
    _, record = cake_backend.record_for(CUDA)
    cast = (
        "gemm_dw_cast_bf16_tma"
        if cake_backend.dw_cast_tma_rule(3, 8265 - 2 * 4096, record["arch"])
        else "gemm_dw_cast_bf16"
    )
    casts = {"gemm_dw_cast_bf16", "gemm_dw_cast_bf16_tma"}
    base, base_streams, base_bound = run()
    assert base_streams == 2 and casts & set(base_bound) == {cast}  # the defaults
    monkeypatch.setenv(knobs.DW_STREAM_ENV, "0")
    off, off_streams, _ = run()
    assert off_streams == 1
    on, on_streams, _ = run(dw_stream=True)  # the keyword wins over the variable
    assert on_streams == 2
    monkeypatch.delenv(knobs.DW_STREAM_ENV)
    monkeypatch.setenv(knobs.FUSE_DW_CAST_ENV, "0")
    unfused, _, unfused_bound = run()
    assert not casts & set(unfused_bound)
    assert "gemm_dw_acc" in unfused_bound and "scale_cast_bf16" in unfused_bound
    fused, _, fused_bound = run(fuse_dw_cast=True)  # the keyword wins over the variable
    assert casts & set(fused_bound) == {cast}
    for name, res in (("off", off), ("on", on), ("unfused", unfused), ("fused", fused)):
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(_bits(res[key]), _bits(base[key])), (name, key)


def test_device_temperature_through_the_public_entry(monkeypatch):
    """Through the public entry on the generated program: a heterogeneous per-token temperature and a scalar give the
    bits (integer views) of the backend entry with the same argument, ``temperature=1.0`` the bits of the call
    without the keyword (default ``None``); ``cake_lm_head_det_train_preload(temperature=True)`` loads the temperature forms and a call after it
    gives the same bits.  The public calls pin ``memory_mode="fast"``, the backend entry's form (the default call of
    this shape takes a budget form, tested in ``test_cake_lm_head_det_train.py``)."""
    _require_program()
    _clear(monkeypatch)
    name, _ = cake_backend.record_for(CUDA)
    tau_stages = cake_backend.TEMPERATURE_STAGES
    if not set(tau_stages) <= set(
        cake_backend.registered_stages_with_temperature(name)
    ):
        pytest.skip("the registered program has no temperature forms")
    inp = make_inputs(
        4097, seed=SEED + 4097, device=CUDA, ignore_frac=0.05, temperature="hetero"
    )

    def run(fn, **kw):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            kw.setdefault("backend", "cake")
            if fn is cake_lm_head_det_train_logprob:
                kw.setdefault("memory_mode", "fast")
            logp = fn(X, W, inp.labels, chunk_size=4096, **kw)
        logp.backward(inp.dlogp)
        torch.cuda.synchronize()
        return dict(logp=logp.detach(), dX=X.grad, dW=W.grad)

    def same(a, b, case):
        for key in ("logp", "dX", "dW"):
            assert torch.equal(_bits(a[key]), _bits(b[key])), (case, key)

    for value in (inp.temperature, 0.7):
        same(
            run(cake_lm_head_det_train_logprob, temperature=value),
            run(
                cake_backend.cake_lm_head_det_train_logprob_temperature,
                temperature=value,
            ),
            type(value).__name__,
        )
    plain = run(cake_lm_head_det_train_logprob)
    same(plain, run(cake_lm_head_det_train_logprob, temperature=1.0), "one")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        public.cake_lm_head_det_train_preload(temperature=True)
    same(
        run(
            cake_backend.cake_lm_head_det_train_logprob_temperature,
            temperature=inp.temperature,
        ),
        run(cake_lm_head_det_train_logprob, temperature=inp.temperature),
        "after preload",
    )


def test_device_public_entry_places_the_gradients(monkeypatch):
    """Through the public entry on the generated program, gradient placement ``"auto"`` (the backend default): a call
    under ``torch.no_grad`` launches no dX / dW GEMM and gives the plain call's ``loss`` / ``logp`` bits; a
    non-reentrant checkpointed step launches the plain step's dX / dW GEMMs once and gives its ``loss`` / ``logp`` /
    ``dX`` / ``dW`` bits.  The calls pin ``memory_mode="fast"`` (the default call of this shape takes a budget form;
    its checkpointed steps are tested in ``test_cake_lm_head_det_train.py``)."""
    _require_program()
    _clear(monkeypatch)
    inp = make_inputs(8700, seed=SEED + 8700, device=CUDA, ignore_frac=0.05)
    launched: list = []
    real_run_keys = cake_backend._run_keys

    def spy_run_keys(plan, device_index, keys, run_key, *args, **kwargs):
        def run(key):
            launched.append(key[0])
            return run_key(key)

        return real_run_keys(plan, device_index, keys, run, *args, **kwargs)

    monkeypatch.setattr(cake_backend, "_run_keys", spy_run_keys)
    assert cake_backend.gradient_placement() == "auto"

    def gradient_gemms():
        return sum(1 for s in launched if str(s).startswith(("gemm_dx", "gemm_dw")))

    def step(kind):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        logps = []

        def head(a, b):
            loss, logp = cake_lm_head_det_train_loss(
                a,
                b,
                inp.labels,
                objective="ce",
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
                memory_mode="fast",
            )
            logps.append(logp)
            return loss

        launched.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
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


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_public_default_step_peak_within_the_head_bytes(monkeypatch, entry):
    """Through the public entry on the generated program, the default call (no ``memory_mode``) at flashinfer#7545's
    shape (H 6144, V 154880, T 16384, a third of the labels ignored, chunk 4096): its training step -- the call that
    binds and one from the remembered binding, forward through a backward that frees the graph, outputs and gradients
    included -- peaks within the bytes of a head that materializes the BF16 logits of the valid rows
    (``2 * T_v * V + 2 * V * H + 2 * T * H``) and gives the bits (integer views) of the explicit fast form."""
    _require_program()
    _clear(monkeypatch)
    name, _ = cake_backend.record_for(CUDA)
    if cake_backend.LOGPROB_DZ_STAGE not in cake_backend.registered_stages(name):
        pytest.skip(
            "the registered program has no dz arm of the logits GEMM (no recompute forms)"
        )
    T = 16384
    inp = make_inputs(T, seed=SEED + 7545, device=CUDA, ignore_frac=0.33)
    R, H, V = int(inp.valid.sum()), inp.H, inp.V
    head = 2 * R * V + 2 * V * H + 2 * T * H
    keys = ("loss", "logp", "dX", "dW") if entry == "loss" else ("logp", "dX", "dW")

    def step(**kw):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            if entry == "loss":
                loss, logp = cake_lm_head_det_train_loss(
                    X,
                    W,
                    inp.labels,
                    objective="ce",
                    loss_div=inp.loss_div,
                    chunk_size=4096,
                    return_logp=True,
                    **kw,
                )
                loss.backward()
                loss = loss.detach()
            else:
                logp = cake_lm_head_det_train_logprob(
                    X, W, inp.labels, chunk_size=4096, **kw
                )
                logp.backward(inp.dlogp)
                loss = None
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        return dict(loss=loss, logp=logp.detach(), dX=X.grad, dW=W.grad), peak

    fast, _ = step(memory_mode="fast")
    for _ in range(2):
        torch.cuda.empty_cache()
        result, peak = step()
        for key in keys:
            assert torch.equal(_bits(result[key]), _bits(fast[key])), key
        del result
        assert peak <= head, (peak, head)


def test_device_public_logprob_keeps_one_chunk_with_a_frozen_weight(monkeypatch):
    """Through the public entry on the generated program, a default log-probability call whose valid rows fit one
    chunk (T 1500, 5 % of the labels ignored: below every architecture's zero-gradient row skip floor) keeps its
    logits with a trainable ``W`` and with a frozen one (``X`` trainable: the facade hands the call to the opt-in
    module's rule) and gives the ``logp`` / ``dX`` (/ ``dW``) bits (integer views) of ``memory_mode="fast"``; under
    ``torch.no_grad`` it runs the default forward."""
    _require_program()
    _clear(monkeypatch)
    inp = make_inputs(1500, seed=SEED + 1500, device=CUDA, ignore_frac=0.05)
    applied: list = []
    bound = cake_backend._forward_keep_bound

    def spy(*args, **kw):
        applied.append(len(args))
        return bound(*args, **kw)

    def step(train_w, **kw):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(train_w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            logp = cake_lm_head_det_train_logprob(
                X, W, inp.labels, chunk_size=4096, **kw
            )
            logp.backward(inp.dlogp)
        torch.cuda.synchronize()
        return dict(logp=logp.detach(), dX=X.grad, dW=W.grad)

    for train_w in (True, False):
        fast = step(train_w, memory_mode="fast")
        with monkeypatch.context() as m:
            m.setattr(cake_backend, "_forward_keep_bound", spy)
            applied.clear()
            out = step(train_w)
            assert applied, train_w
            applied.clear()
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore", ExperimentalWarning)
                lp = cake_lm_head_det_train_logprob(
                    inp.X, inp.W, inp.labels, chunk_size=4096
                )
            assert not applied, train_w
        assert torch.equal(_bits(lp), _bits(fast["logp"])), train_w
        for key in ("logp", "dX", "dW") if train_w else ("logp", "dX"):
            assert torch.equal(_bits(out[key]), _bits(fast[key])), (train_w, key)
        assert train_w or out["dW"] is None
