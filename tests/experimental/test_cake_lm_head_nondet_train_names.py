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

Names of the non-deterministic LM-head training module ``flashinfer.cake_lm_head_nondet_train``.

The public names are the facade's ``__all__`` and the backend objects it forwards to
(``flashinfer.experimental.cake.cake_lm_head_nondet_train``): each carries the mode (``nondet``), the
former names and package paths are gone (no alias), no entry point takes a ``deterministic`` keyword
and the facade does not read torch's deterministic mode.  Internal helpers and build identities
(program and library stems ``cake_lm_head_loss_<hex>``) keep their names.
"""

import ast
import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.cake_lm_head_nondet_train as public

pytestmark = pytest.mark.cpu_only

BACKEND = "flashinfer.experimental.cake.cake_lm_head_nondet_train"
ENTRIES = (
    "cake_lm_head_nondet_train_logprob",
    "cake_lm_head_nondet_train_loss",
    "cake_lm_head_nondet_train_preload",
    "cake_lm_head_nondet_train_support_kl",
    "cake_lm_head_nondet_train_support_logprob",
)
# The backend objects the facade forwards to, by module (each defined in that module).
FORWARDED = {
    "cake_backend": (
        "cake_lm_head_nondet_train_loss",
        "cake_lm_head_nondet_train_logprob",
        "cake_lm_head_nondet_train_logprob_temperature",
        "cake_lm_head_nondet_train_logprob_keep",
        "CakeLMHeadNondetLossFunction",
        "CakeLMHeadNondetLogprobFunction",
        "CakeLMHeadNondetLogprobTemperatureFunction",
        "CakeLMHeadNondetLogprobKeepFunction",
    ),
    "cake_backend_opt_in": (
        "cake_lm_head_nondet_train_loss",
        "cake_lm_head_nondet_train_logprob",
        "cake_lm_head_nondet_train_logprob_temperature",
        "cake_lm_head_nondet_train_preload",
        "CakeLMHeadNondetLossFunction",
        "CakeLMHeadNondetLogprobFunction",
        "CakeLMHeadNondetLogprobTemperatureFunction",
    ),
    "cake_support": (
        "cake_lm_head_nondet_train_support_logprob",
        "cake_lm_head_nondet_train_support_kl",
        "CakeLMHeadNondetSupportFunction",
    ),
}
# Module paths that no longer import: the former entry module, a module of the backend package before this
# module's name, and of the alias package removed with the deterministic rename.
REMOVED_MODULES = (
    "flashinfer.chunked_lm_head",
    "flashinfer.experimental.cake.cake_lm_head_loss.cake_backend",
    "flashinfer.experimental.cake.cake_lm_head_loss.cake_support",
    "flashinfer.experimental.cake_lm_head_loss.cake_backend",
)
# Former names that the backend modules no longer define or export (no aliases).
REMOVED_NAMES = {
    "cake_backend": (
        "chunked_lm_head_loss",
        "chunked_lm_head_logprob",
        "chunked_lm_head_logprob_temperature",
        "chunked_lm_head_logprob_keep",
        "ChunkedLmHeadLossFunction",
        "ChunkedLmHeadLogprobFunction",
        "ChunkedLmHeadLogprobTemperatureFunction",
        "ChunkedLmHeadLogprobKeepFunction",
        "_require_nondeterministic",
    ),
    "cake_backend_opt_in": (
        "chunked_lm_head_loss",
        "chunked_lm_head_logprob",
        "chunked_lm_head_logprob_temperature",
        "ChunkedLmHeadLossFunction",
        "ChunkedLmHeadLogprobFunction",
        "ChunkedLmHeadLogprobTemperatureFunction",
        "preload",
    ),
    "cake_support": (
        "chunked_lm_head_support_logprob",
        "chunked_lm_head_support_kl",
        "SupportLogprobFunction",
    ),
}
H_HOST, V_HOST = 256, 512


def _tokens(name: str) -> list:
    """The lowercase words of ``name``: split on ``_``, then on CamelCase (``CakeLMHeadNondetSupportFunction`` ->
    cake, lm, head, nondet, support, function)."""
    words = []
    for part in name.split("_"):
        words += re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+", part)
    return [w.lower() for w in words]


def _autograd_functions(module) -> list:
    """The ``torch.autograd.Function`` classes defined in ``module``."""
    return [
        (name, value)
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, torch.autograd.Function)
        and value.__module__ == module.__name__
    ]


def _backend(module_name):
    return importlib.import_module(f"{BACKEND}.{module_name}")


def _fresh(code: str):
    """``code`` run in a fresh interpreter (no module of this process imported); returns the JSON it prints."""
    repo = str(Path(__file__).resolve().parents[2])
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


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


def test_removed_module_paths_do_not_import():
    """In a fresh interpreter, no former module path imports (``ModuleNotFoundError``; a stale ``__pycache__`` or
    ``aot`` directory can still make a bare former package path import as a namespace package, so a module of it
    is imported)."""
    code = (
        "import importlib, json\n"
        "result = {}\n"
        f"for name in {REMOVED_MODULES!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        result[name] = 'imported'\n"
        "    except ModuleNotFoundError as exc:\n"
        "        result[name] = type(exc).__name__\n"
        "print(json.dumps(result))\n"
    )
    assert _fresh(code) == dict.fromkeys(REMOVED_MODULES, "ModuleNotFoundError")


def test_former_names_are_gone_from_the_backend():
    """No backend module defines or exports a former name (no aliases)."""
    assert sum(len(names) for names in REMOVED_NAMES.values()) == 19
    for module_name, names in REMOVED_NAMES.items():
        module = _backend(module_name)
        for name in names:
            assert not hasattr(module, name), (module_name, name)
            assert name not in getattr(module, "__all__", ()), (module_name, name)


def test_facade_all_is_the_five_entries():
    """The facade's ``__all__`` is exactly its five entries, each named after the module and each a stable API
    (no experimental marking)."""
    assert sorted(public.__all__) == list(ENTRIES)
    for name in public.__all__:
        fn = getattr(public, name)
        assert name.startswith("cake_lm_head_nondet_train_"), name
        assert getattr(fn, "is_experimental", False) is False, name
        assert fn.__name__ == name, name


def test_mode_word_is_a_token():
    """The token split finds the mode word only as a whole word: ``nondet`` is one token, never ``det``."""
    assert _tokens("CakeLMHeadNondetSupportFunction") == [
        "cake",
        "lm",
        "head",
        "nondet",
        "support",
        "function",
    ]
    assert _tokens("cake_lm_head_nondet_train_loss") == [
        "cake",
        "lm",
        "head",
        "nondet",
        "train",
        "loss",
    ]
    assert "det" not in _tokens("cake_lm_head_nondet_train_loss")
    assert "nondet" not in _tokens("cake_lm_head_det_train_loss")


def test_public_names_carry_the_mode():
    """Every name of the facade's ``__all__`` and every backend object the facade forwards to carries the mode
    word ``nondet`` as a token (split on ``_`` and CamelCase) and not ``det``; ``cake_support.__all__`` lists only
    those objects and ALL_CAPS constants."""
    count = 0
    for name in public.__all__:
        assert "nondet" in _tokens(name) and "det" not in _tokens(name), name
        count += 1
    for module_name, names in FORWARDED.items():
        module = _backend(module_name)
        for name in names:
            obj = getattr(module, name)
            assert obj.__module__ == module.__name__, (module_name, name)
            assert "nondet" in _tokens(name) and "det" not in _tokens(name), (
                module_name,
                name,
            )
            count += 1
    assert count == 23
    support = _backend("cake_support")
    for name in support.__all__:
        assert name.isupper() or "nondet" in _tokens(name), name
    assert set(FORWARDED["cake_support"]) <= set(support.__all__)


def test_no_entry_takes_a_mode_keyword():
    """No facade entry, forwarded backend entry or autograd ``Function.forward`` of the backend accepts a
    ``deterministic`` keyword: ``inspect.signature(fn).bind_partial(deterministic=False)`` is a ``TypeError``
    (no such parameter, no ``**kwargs``); passing ``deterministic`` (``True`` or ``False``) to a facade entry is a
    ``TypeError`` from argument binding (host tensors suffice: it raises before any work)."""
    fns = {name: getattr(public, name) for name in ENTRIES}
    for module_name, names in FORWARDED.items():
        module = _backend(module_name)
        for name in names:
            obj = getattr(module, name)
            fns[f"{module_name}.{name}"] = obj.forward if inspect.isclass(obj) else obj
    assert len(fns) == 23
    classes = []
    for module_name in ("cake_backend", "cake_backend_opt_in", "cake_support"):
        for name, cls in _autograd_functions(_backend(module_name)):
            fns[f"{module_name}.{name}.forward"] = cls.forward
            classes.append(f"{module_name}.{name}")
    assert sorted(classes) == sorted(
        [
            "cake_backend.CakeLMHeadNondetLogprobFunction",
            "cake_backend.CakeLMHeadNondetLogprobKeepFunction",
            "cake_backend.CakeLMHeadNondetLogprobTemperatureFunction",
            "cake_backend.CakeLMHeadNondetLossFunction",
            "cake_backend._SmallCallStage",
            "cake_backend_opt_in.CakeLMHeadNondetLogprobFunction",
            "cake_backend_opt_in.CakeLMHeadNondetLogprobTemperatureFunction",
            "cake_backend_opt_in.CakeLMHeadNondetLossFunction",
            "cake_support.CakeLMHeadNondetSupportFunction",
        ]
    ), sorted(classes)
    for label, fn in fns.items():
        try:
            inspect.signature(fn).bind_partial(deterministic=False)
        except TypeError:
            pass
        else:
            pytest.fail(f"{label} accepts a deterministic keyword")
        params = inspect.signature(fn).parameters
        assert "deterministic" not in params, label
        assert all(p.kind is not p.VAR_KEYWORD for p in params.values()), label
    X, W, labels = _host_call()
    sup = _support_args()
    calls = {
        "cake_lm_head_nondet_train_loss": ((X, W, labels), dict(loss_div=1.0)),
        "cake_lm_head_nondet_train_logprob": ((X, W, labels), {}),
        "cake_lm_head_nondet_train_preload": ((), {}),
        "cake_lm_head_nondet_train_support_logprob": (sup, dict(k_cap=3)),
        "cake_lm_head_nondet_train_support_kl": (
            sup + (torch.zeros(4),),
            dict(k_cap=3),
        ),
    }
    for name, (args, kwargs) in calls.items():
        for value in (True, False):
            with pytest.raises(
                TypeError, match="unexpected keyword argument 'deterministic'"
            ):
                getattr(public, name)(*args, **kwargs, deterministic=value)


def test_facade_reads_no_torch_mode_and_imports_no_deterministic_module():
    """The facade's source never reads torch's deterministic mode, and its only imports of the LM-head backend are
    the function-local imports of ``flashinfer.experimental.cake.cake_lm_head_nondet_train`` (no module-level
    backend import, no deterministic module)."""
    source = inspect.getsource(public)
    assert "are_deterministic_algorithms_enabled" not in source
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
    assert not [m for m in imported if "det_train" in m and "nondet" not in m]
    top = {
        ("." * n.level) + (n.module or "")
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
    }
    assert not [m for m in top if "experimental" in m], top
    local = [m for m in imported if "experimental" in m]
    assert local == [".experimental.cake.cake_lm_head_nondet_train"] * 3, local


def test_package_identity_and_build_identity():
    """``build.json`` names the moved package; the program and library stems keep their build identity."""
    path = Path(_backend("cake_build").__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    assert json.loads(path.read_text())["package"] == BACKEND
    assert _backend("cake_jit").library_stem("x") == "cake_lm_head_loss_x"


def test_flashinfer_exports_the_module_lazily_and_no_former_name():
    """In a fresh interpreter, ``import flashinfer`` binds no name of the former entry module or package and does
    not import this module; ``flashinfer.cake_lm_head_nondet_train`` is listed by ``dir`` and binds this module on
    first access."""
    code = (
        "import json, sys\n"
        "import flashinfer\n"
        "names = set(dir(flashinfer)) | set(getattr(flashinfer, '__all__', ()))\n"
        "former = sorted(n for n in names if 'chunked_lm_head' in n or 'cake_lm_head_loss' in n)\n"
        "loaded = 'flashinfer.cake_lm_head_nondet_train' in sys.modules\n"
        "listed = 'cake_lm_head_nondet_train' in dir(flashinfer)\n"
        "module = flashinfer.cake_lm_head_nondet_train\n"
        "print(json.dumps(dict(former=former, loaded=loaded, listed=listed, name=module.__name__)))\n"
    )
    assert _fresh(code) == dict(
        former=[],
        loaded=False,
        listed=True,
        name="flashinfer.cake_lm_head_nondet_train",
    )
