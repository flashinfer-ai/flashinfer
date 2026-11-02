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

The experimental entry points ``flashinfer.chunked_lm_head.chunked_lm_head_det_loss`` and
``chunked_lm_head_det_logprob`` are thin wrappers over
``flashinfer.experimental.cake_lm_head_det_loss.cake_backend``: they carry the experimental
marking, accept ``backend="cake"`` only and own FlashInfer's A/B environment variables
``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` -- a switch keyword left at ``None`` falls back to its
variable and the resolved value is passed down; the two binding-cache variables are
read when the module is imported and applied when it first reaches the backend.  The
tests here cover that wrapper: the marking, each variable's fallback and parsing, the
cache configuration and, on a supported device, the pass-down through the public
entry.  The backend tests live in ``test_cake_lm_head_det_loss.py``, which imports the
package only through ``tests.test_helpers.cake_package`` and reads no environment
variable.
"""

import importlib
import inspect
import warnings

import pytest
import torch
import torch.utils.checkpoint

import flashinfer.chunked_lm_head as public
from flashinfer.api_logging import ExperimentalWarning
from flashinfer.chunked_lm_head import (
    chunked_lm_head_det_logprob,
    chunked_lm_head_det_loss,
)
from flashinfer.experimental.cake_lm_head_det_loss import cake_backend
from flashinfer.experimental.cake_lm_head_det_loss.cake_backend import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    BindingCache,
    forward_binding_key,
    generated_program_available,
    stages_for_entry,
)
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs

SEED = 20260930
# Host-layer shapes (the wrapper's own checks never reach a kernel).
H_HOST, V_HOST = 256, 512
# The switch keywords of both public entries and the variable each one falls back to.
ENV_OF = {
    "compact_rows": public.COMPACT_ROWS_ENV,
    "fuse_dw_cast": public.FUSE_DW_CAST_ENV,
    "dx_finalize": public.DX_FINALIZE_ENV,
    "dw_stream": public.DW_STREAM_ENV,
    "hidden_count": public.HIDDEN_COUNT_ENV,
    "gemm_tuning": public.GEMM_TUNING_ENV,
}
SWITCHES = tuple(ENV_OF)
CACHE_ENVS = (public.BINDING_CACHE_ENV, public.BINDING_CACHE_CAPACITY_ENV)


def _knobs(**explicit):
    """The keywords the wrapper passes down for a call whose switches are ``explicit`` (every other one ``None``)."""
    given = {name: None for name in SWITCHES}
    given.update(explicit)
    return public._resolve_knobs(**given)


def _clear(monkeypatch):
    for name in (*ENV_OF.values(), *CACHE_ENVS):
        monkeypatch.delenv(name, raising=False)


def _host_call():
    X = torch.zeros(4, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    labels = torch.zeros(4, dtype=torch.int64)
    return X, W, labels


def test_public_api_is_marked_experimental():
    assert chunked_lm_head_det_loss.is_experimental is True
    assert chunked_lm_head_det_logprob.is_experimental is True
    assert "SM100" in chunked_lm_head_det_loss.__doc__
    assert "SM100" in chunked_lm_head_det_logprob.__doc__
    X, W, labels = _host_call()
    # the public module serves the cake backend only
    with pytest.raises(ValueError, match="backend"):
        chunked_lm_head_det_loss(X, W, labels, loss_div=1.0, backend="reference")
    with pytest.raises(ValueError, match="backend"):
        chunked_lm_head_det_logprob(X, W, labels, backend="reference")


# The public names before the deterministic (``det``) naming: since the non-deterministic package they are dispatchers
# whose default (``deterministic=True``) runs the det entry with the same arguments, not aliases
# (tests/experimental/test_cake_lm_head_loss_api.py checks the dispatch).
OLD_PUBLIC_NAMES = (
    "chunked_lm_head_loss",
    "chunked_lm_head_logprob",
    "chunked_lm_head_preload",
    "chunked_lm_head_support_logprob",
    "chunked_lm_head_support_kl",
)
# The names before the deterministic (``det``) naming and the canonical object each one is, by module.
OLD_NAMES = {
    "flashinfer.experimental.cake_lm_head_det_loss.cake_backend": {
        "chunked_lm_head_loss": "chunked_lm_head_det_loss",
        "chunked_lm_head_logprob": "chunked_lm_head_det_logprob",
        "prepare_lm_head_loss": "prepare_lm_head_det_loss",
        "validate_lm_head_inputs": "validate_lm_head_det_inputs",
        "lm_head_loss_workspace_size": "lm_head_det_loss_workspace_size",
        "LmHeadLossRunner": "LmHeadDetLossRunner",
        "ChunkedLmHeadLossFunction": "ChunkedLmHeadDetLossFunction",
        "ChunkedLmHeadLogprobFunction": "ChunkedLmHeadDetLogprobFunction",
        "load_cake_lm_head_loss_module": "load_cake_lm_head_det_loss_module",
    },
    "flashinfer.experimental.cake_lm_head_det_loss.cake_backend_opt_in": {
        "chunked_lm_head_loss": "chunked_lm_head_det_loss",
        "chunked_lm_head_logprob": "chunked_lm_head_det_logprob",
        "prepare_lm_head_loss": "prepare_lm_head_det_loss",
        "validate_lm_head_inputs": "validate_lm_head_det_inputs",
        "LmHeadLossRunner": "LmHeadDetLossRunner",
        "ChunkedLmHeadLossFunction": "ChunkedLmHeadDetLossFunction",
        "ChunkedLmHeadLogprobFunction": "ChunkedLmHeadDetLogprobFunction",
        "load_cake_lm_head_loss_module": "load_cake_lm_head_det_loss_module",
    },
    "flashinfer.experimental.cake_lm_head_det_loss.cake_support": {
        "chunked_lm_head_support_logprob": "chunked_lm_head_det_support_logprob",
        "chunked_lm_head_support_kl": "chunked_lm_head_det_support_kl",
        "SupportLogprobFunction": "SupportDetLogprobFunction",
    },
    "flashinfer.experimental.cake_lm_head_det_loss.cake_jit": {
        "load_cake_lm_head_loss_module": "load_cake_lm_head_det_loss_module",
    },
}
PACKAGE_MODULES = (
    "cake_backend",
    "cake_backend_opt_in",
    "cake_build",
    "cake_jit",
    "cake_support",
)


def test_old_names_are_the_canonical_objects():
    """Every name before the ``det`` naming is an alias of the canonical object (the same object, not a copy), the
    canonical object carries its own name, and the canonical names are the ones the modules define."""
    for module_name, names in OLD_NAMES.items():
        module = importlib.import_module(module_name)
        for old, new in names.items():
            canonical = getattr(module, new)
            assert getattr(module, old) is canonical, (module_name, old)
            assert canonical.__name__ == new, (module_name, new, canonical.__name__)


def test_backend_loader_aliases_are_the_jit_loader():
    """The stage loader's name before the ``det`` naming, read from either backend module (by the new or the old
    package path), is the package loader itself, ``cake_jit.load_cake_lm_head_det_loss_module``."""
    jit = importlib.import_module(
        "flashinfer.experimental.cake_lm_head_det_loss.cake_jit"
    )
    loader = jit.load_cake_lm_head_det_loss_module
    for name in ("cake_backend", "cake_backend_opt_in"):
        module = importlib.import_module(
            f"flashinfer.experimental.cake_lm_head_det_loss.{name}"
        )
        old = importlib.import_module(
            f"flashinfer.experimental.cake_lm_head_loss.{name}"
        )
        assert module.load_cake_lm_head_det_loss_module is loader, name
        assert module.load_cake_lm_head_loss_module is loader, name
        assert old.load_cake_lm_head_loss_module is loader, name


def test_old_package_path_is_the_canonical_package():
    """``flashinfer.experimental.cake_lm_head_loss.<module>`` is the module of the same name of
    ``flashinfer.experimental.cake_lm_head_det_loss`` (the same module object: one binding cache, one registry),
    whether reached by ``import`` or by ``from ... import``, and its old names resolve through it."""
    for name in PACKAGE_MODULES:
        canonical = importlib.import_module(
            f"flashinfer.experimental.cake_lm_head_det_loss.{name}"
        )
        old = importlib.import_module(
            f"flashinfer.experimental.cake_lm_head_loss.{name}"
        )
        assert old is canonical, name
        assert (
            canonical.__name__
            == f"flashinfer.experimental.cake_lm_head_det_loss.{name}"
        )
    from flashinfer.experimental.cake_lm_head_loss import cake_backend as old_backend
    from flashinfer.experimental.cake_lm_head_loss.cake_backend import (
        chunked_lm_head_loss as old_entry,
    )
    from flashinfer.experimental.cake_lm_head_loss.cake_support import (
        SupportLogprobFunction as old_support,
    )

    assert old_backend is cake_backend
    assert old_entry is cake_backend.chunked_lm_head_det_loss
    support = importlib.import_module(
        "flashinfer.experimental.cake_lm_head_det_loss.cake_support"
    )
    assert old_support is support.SupportDetLogprobFunction
    assert set(support.__all__) >= {
        "SupportDetLogprobFunction",
        "SupportLogprobFunction",
        "chunked_lm_head_det_support_logprob",
        "chunked_lm_head_support_logprob",
        "chunked_lm_head_det_support_kl",
        "chunked_lm_head_support_kl",
    }


def test_old_public_names_are_marked_experimental():
    """The old public names are decorated experimental entries."""
    for old in OLD_PUBLIC_NAMES:
        assert getattr(public, old).is_experimental is True, old


def test_public_api_switch_keywords():
    """Both public entries take every switch as a keyword-only argument defaulting to ``None``; the loss entry's
    docstring states each default and the variable it falls back to; the backend entries take the same names (the
    wrapper passes them by keyword); the variables the wrapper names are exactly the ones these tests cover."""
    for fn in (chunked_lm_head_det_loss, chunked_lm_head_det_logprob):
        params = inspect.signature(fn).parameters
        for name in SWITCHES:
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
                fn.__name__,
                name,
            )
            assert params[name].default is None, (fn.__name__, name)
        assert "FLASHINFER_CAKE_LM_HEAD_LOSS_" in fn.__doc__
    for variable in (*ENV_OF.values(), *CACHE_ENVS):
        assert variable in chunked_lm_head_det_loss.__doc__, variable
    for fn in (
        cake_backend.chunked_lm_head_det_loss,
        cake_backend.chunked_lm_head_det_logprob,
    ):
        params = inspect.signature(fn).parameters
        assert all(params[name].default is None for name in SWITCHES), fn.__name__
    named = {v for k, v in vars(public).items() if k.endswith("_ENV")}
    assert named == set(ENV_OF.values()) | set(CACHE_ENVS)
    assert all(v.startswith("FLASHINFER_CAKE_LM_HEAD_LOSS_") for v in named)


def test_public_api_memory_and_count_keywords(monkeypatch):
    """Both public entries take ``memory_mode`` (default ``None``, no environment variable), ``num_valid_rows``
    (``None``) and ``check_num_valid_rows`` (``False``) as keyword-only arguments and document them; a call with
    ``memory_mode`` or ``num_valid_rows`` passes all three to the opt-in backend module unchanged, a call with neither
    calls the default backend entry without them; ``chunked_lm_head_det_preload`` is experimental and needs a CUDA
    device."""
    _clear(monkeypatch)
    for fn in (chunked_lm_head_det_loss, chunked_lm_head_det_logprob):
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
        monkeypatch.setattr(module, "chunked_lm_head_det_loss", spy(f"{tag}_loss"))
        monkeypatch.setattr(
            module, "chunked_lm_head_det_logprob", spy(f"{tag}_logprob")
        )
    X, W, labels = _host_call()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        chunked_lm_head_det_loss(
            X,
            W,
            labels,
            loss_div=1.0,
            memory_mode="budget",
            num_valid_rows=3,
            check_num_valid_rows=True,
        )
        chunked_lm_head_det_logprob(X, W, labels, num_valid_rows=3)
        chunked_lm_head_det_loss(X, W, labels, loss_div=1.0)
        chunked_lm_head_det_logprob(X, W, labels)
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
    # check_num_valid_rows=True without a count: ValueError before either backend entry runs
    seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        for fn, kw in (
            (chunked_lm_head_det_loss, dict(loss_div=1.0)),
            (chunked_lm_head_det_logprob, {}),
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
    assert public.chunked_lm_head_det_preload.is_experimental is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        with pytest.raises(ValueError, match="CUDA device"):
            public.chunked_lm_head_det_preload("cpu")


def test_compact_rows_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["compact_rows"] is True  # unset: the backend's default
    monkeypatch.setenv(public.COMPACT_ROWS_ENV, "0")
    assert _knobs()["compact_rows"] is False
    assert _knobs(compact_rows=True)["compact_rows"] is True  # the keyword wins
    monkeypatch.setenv(public.COMPACT_ROWS_ENV, "1")
    assert _knobs()["compact_rows"] is True
    assert _knobs(compact_rows=False)["compact_rows"] is False
    monkeypatch.setenv(public.COMPACT_ROWS_ENV, "off")
    assert _knobs()["compact_rows"] is True  # "0" is the only off value


def test_fuse_dw_cast_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["fuse_dw_cast"] is True
    assert "gemm_dw_cast_bf16" in stages_for_entry(
        "loss", fuse_dw_cast=_knobs()["fuse_dw_cast"]
    )
    monkeypatch.setenv(public.FUSE_DW_CAST_ENV, "0")
    assert _knobs()["fuse_dw_cast"] is False
    assert "gemm_dw_cast_bf16" not in stages_for_entry(
        "loss", fuse_dw_cast=_knobs()["fuse_dw_cast"]
    )
    assert _knobs(fuse_dw_cast=True)["fuse_dw_cast"] is True  # the keyword wins
    monkeypatch.setenv(public.FUSE_DW_CAST_ENV, "1")
    assert _knobs()["fuse_dw_cast"] is True
    assert _knobs(fuse_dw_cast=False)["fuse_dw_cast"] is False


def test_dx_finalize_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["dx_finalize"] is True
    monkeypatch.setenv(public.DX_FINALIZE_ENV, "0")
    assert _knobs()["dx_finalize"] is False
    assert _knobs(dx_finalize=True)["dx_finalize"] is True  # the keyword wins
    monkeypatch.setenv(public.DX_FINALIZE_ENV, "1")
    assert _knobs()["dx_finalize"] is True
    assert _knobs(dx_finalize=False)["dx_finalize"] is False
    assert cake_backend._resolve_dx_finalize(_knobs()["dx_finalize"]) is True


def test_hidden_count_default_env(monkeypatch):
    _clear(monkeypatch)
    assert _knobs()["hidden_count"] is True
    monkeypatch.setenv(public.HIDDEN_COUNT_ENV, "0")
    assert _knobs()["hidden_count"] is False
    assert _knobs(hidden_count=True)["hidden_count"] is True  # the keyword wins
    monkeypatch.setenv(public.HIDDEN_COUNT_ENV, "1")
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
        monkeypatch.setenv(public.DW_STREAM_ENV, value)
        assert _knobs()["dw_stream"] is False
        assert not any(
            cake_backend.dw_side_stream(n, _knobs()["dw_stream"]) for n in (1, 2, 3, 8)
        )
    for value in ("1", "true", "on", "yes", " On "):
        monkeypatch.setenv(public.DW_STREAM_ENV, value)
        assert _knobs()["dw_stream"] is True
        assert [
            cake_backend.dw_side_stream(n, _knobs()["dw_stream"]) for n in (1, 2, 3)
        ] == [False, True, True]
    for value in ("auto", "AUTO", "", "anything-else"):
        monkeypatch.setenv(public.DW_STREAM_ENV, value)
        assert _knobs()["dw_stream"] == "auto"
        assert [
            cake_backend.dw_side_stream(n, _knobs()["dw_stream"]) for n in (2, 3)
        ] == [False, True]
    monkeypatch.setenv(public.DW_STREAM_ENV, "0")
    assert _knobs(dw_stream=True)["dw_stream"] is True  # the keyword wins
    assert _knobs(dw_stream="auto")["dw_stream"] == "auto"


def test_gemm_tuning_default_env(monkeypatch):
    """The GEMM-tuning variable: unset / blank -> no knob, a JSON object -> the knob dict (validated by the backend and
    part of the binding key), invalid JSON or a non-object -> a ``ValueError`` naming the variable, raised by the
    public entry before any backend work; the keyword wins over the variable."""
    _clear(monkeypatch)
    assert _knobs()["gemm_tuning"] is None
    assert cake_backend._resolve_gemm_tuning(_knobs()["gemm_tuning"]) == {}
    monkeypatch.setenv(public.GEMM_TUNING_ENV, "  ")
    assert _knobs()["gemm_tuning"] is None
    monkeypatch.setenv(
        public.GEMM_TUNING_ENV, '{"logits": {"group_m": 16}, "dx": {"tile_n": 256}}'
    )
    assert _knobs()["gemm_tuning"] == {"logits": {"group_m": 16}, "dx": {"tile_n": 256}}
    assert cake_backend._resolve_gemm_tuning(_knobs()["gemm_tuning"]) == {
        "logits": {"group_m": 16},
        "dx": {"tile_n": 256},
    }
    X, W, labels = _host_call()
    monkeypatch.setenv(public.GEMM_TUNING_ENV, "not json")
    with pytest.raises(ValueError, match="JSON") as info:
        _knobs()
    assert public.GEMM_TUNING_ENV in str(info.value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        with pytest.raises(ValueError, match=public.GEMM_TUNING_ENV):
            chunked_lm_head_det_loss(X, W, labels, loss_div=1.0)
        with pytest.raises(ValueError, match=public.GEMM_TUNING_ENV):
            chunked_lm_head_det_logprob(X, W, labels)
    monkeypatch.setenv(public.GEMM_TUNING_ENV, '["dx"]')
    with pytest.raises(ValueError, match="JSON object"):
        _knobs()
    monkeypatch.setenv(public.GEMM_TUNING_ENV, '{"dx": {"tile_n": 256}}')
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
    """The two binding-cache variables are read when ``flashinfer.chunked_lm_head`` is imported (here: reloaded) and
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
        monkeypatch.setenv(public.BINDING_CACHE_CAPACITY_ENV, "5")
        assert BindingCache().capacity == 64  # the backend does not read the variable
        importlib.reload(public)
        assert cache.capacity == 64  # read at import, applied on first use
        public._cake_backend()
        assert cache.capacity == 5 and cache.enabled
        monkeypatch.setenv(public.BINDING_CACHE_CAPACITY_ENV, " ")
        importlib.reload(public)
        public._cake_backend()
        assert cache.capacity == 64
        monkeypatch.setenv(public.BINDING_CACHE_ENV, "0")
        importlib.reload(public)
        public._cake_backend()
        assert not cache.enabled and len(cache) == 0 and cache.capacity == 64
        monkeypatch.setenv(public.BINDING_CACHE_ENV, "1")
        monkeypatch.setenv(public.BINDING_CACHE_CAPACITY_ENV, "3")
        importlib.reload(public)
        public._cake_backend()
        assert cache.enabled and cache.capacity == 3
        assert public._binding_cache_settings() == dict(enabled=True, capacity=3)
        for bad in ("0", "-1", "many", "2.5"):
            monkeypatch.setenv(public.BINDING_CACHE_CAPACITY_ENV, bad)
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
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
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
    every setting."""
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
            loss, logp = chunked_lm_head_det_loss(
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
    monkeypatch.setenv(public.DW_STREAM_ENV, "0")
    off, off_streams, _ = run()
    assert off_streams == 1
    on, on_streams, _ = run(dw_stream=True)  # the keyword wins over the variable
    assert on_streams == 2
    monkeypatch.delenv(public.DW_STREAM_ENV)
    monkeypatch.setenv(public.FUSE_DW_CAST_ENV, "0")
    unfused, _, unfused_bound = run()
    assert not casts & set(unfused_bound)
    assert "gemm_dw_acc" in unfused_bound and "scale_cast_bf16" in unfused_bound
    fused, _, fused_bound = run(fuse_dw_cast=True)  # the keyword wins over the variable
    assert casts & set(fused_bound) == {cast}
    for name, res in (("off", off), ("on", on), ("unfused", unfused), ("fused", fused)):
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(_bits(res[key]), _bits(base[key])), (name, key)


def test_device_public_entry_places_the_gradients(monkeypatch):
    """Through the public entry on the generated program, gradient placement ``"auto"`` (the backend default): a call
    under ``torch.no_grad`` launches no dX / dW GEMM and gives the plain call's ``loss`` / ``logp`` bits; a
    non-reentrant checkpointed step launches the plain step's dX / dW GEMMs once and gives its ``loss`` / ``logp`` /
    ``dX`` / ``dW`` bits."""
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
            loss, logp = chunked_lm_head_det_loss(
                a,
                b,
                inp.labels,
                objective="ce",
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
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
