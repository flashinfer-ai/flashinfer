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

Naming rules of flashinfer-ai/flashinfer#8928 for the non-deterministic Cake DSA indexer.

The public module is ``flashinfer.cake_dsa_indexer_nondet_train`` and its backend package
``flashinfer.experimental.cake.cake_dsa_indexer_nondet_train``.  Every public name of the module (except ALL_CAPS
module constants) carries the mode word ``nondet`` as a name token, no public callable takes ``deterministic=``, the
backend binds no name of the deterministic package, only the alert reads torch's deterministic-algorithms setting,
importing the module loads no backend (checked in a fresh interpreter), and the build manifest names the package.
The module is new: there is no former name to retire.  CPU only.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer.cake_dsa_indexer_nondet_train as facade
from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

pytestmark = pytest.mark.cpu_only

MODE = "nondet"
FACADE = "flashinfer.cake_dsa_indexer_nondet_train"
BACKEND = "flashinfer.experimental.cake.cake_dsa_indexer_nondet_train"
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
# the backend objects the module forwards to carry the module's names
FORWARDED = (
    "cake_dsa_indexer_nondet_train_topk",
    "cake_dsa_indexer_nondet_train_topk_workspace_size",
)
# the backend-internal names of this package: the deterministic package's with the mode word swapped
INTERNAL = (
    "DSAIndexerNondetRunner",
    "plan_dsa_indexer_nondet_topk",
    "prepare_dsa_indexer_nondet_topk",
    "validate_dsa_indexer_nondet_inputs",
)
# a read of torch's deterministic-algorithms setting
MODE_READ = re.compile(r"torch\.(?:are|is)_deterministic_algorithms_\w*enabled\(")


def _tokens(name: str) -> list[str]:
    """Name tokens: split on ``_`` and on CamelCase boundaries (``DSAIndexerNondetRunner`` -> dsa, indexer, nondet,
    runner)."""
    out = []
    for chunk in name.split("_"):
        out += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", chunk)
    return [t.lower() for t in out]


def _public_names() -> list[str]:
    return [n for n in facade.__all__ if not CONSTANT.match(n)]


def _backend(module: str = "cake_backend"):
    return importlib.import_module(f"{BACKEND}.{module}")


def _fresh(code: str) -> dict:
    """Run ``code`` in a fresh interpreter; it prints one JSON document as its last line."""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_tokenizer_separates_the_mode_words() -> None:
    assert _tokens("cake_dsa_indexer_nondet_train_topk")[3] == "nondet"
    assert _tokens("DSAIndexerNondetRunner") == ["dsa", "indexer", "nondet", "runner"]
    assert "det" not in _tokens("cake_dsa_indexer_nondet_train")
    assert "nondet" not in _tokens("cake_dsa_indexer_det_train")
    assert "det" not in _tokens("_alert_nondeterministic")


def test_every_public_name_carries_the_mode_word() -> None:
    names = _public_names()
    assert sorted(names) == sorted(FORWARDED)
    for name in names:
        assert hasattr(facade, name), name
        tokens = _tokens(name)
        assert MODE in tokens and "det" not in tokens, name


@pytest.mark.parametrize("name", FORWARDED)
def test_forwarded_backend_objects_take_the_module_names(name: str) -> None:
    backend = _backend()
    assert name in backend.__all__
    assert getattr(backend, name).__name__ == name
    assert getattr(facade, name).__name__ == name


def _callables() -> list[tuple[str, object]]:
    backend = _backend()
    objs = [(f"{FACADE}.{n}", getattr(facade, n)) for n in _public_names()]
    objs += [(f"{BACKEND}.cake_backend.{n}", getattr(backend, n)) for n in FORWARDED]
    objs += [(f"{BACKEND}.cake_backend.{n}", getattr(backend, n)) for n in INTERNAL]
    return [(n, o) for n, o in objs if callable(o)]


def test_no_public_callable_takes_deterministic() -> None:
    callables = _callables()
    assert len(callables) == 2 * len(FORWARDED) + len(INTERNAL)
    for name, obj in callables:
        signature = inspect.signature(obj)
        assert "deterministic" not in signature.parameters, name
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)


def test_public_entry_points_are_stable_apis() -> None:
    for name in _public_names():
        assert not getattr(getattr(facade, name), "is_experimental", False), name


@pytest.mark.parametrize("module", ["cake_backend", "cake_policy", "cake_jit"])
def test_backend_binds_no_name_of_the_deterministic_package(module: str) -> None:
    """No attribute of the backend modules has ``det`` as a name token (``DSAIndexerDetRunner``,
    ``prepare_dsa_indexer_det_topk``, ...): the package swaps the mode word in every internal name and imports
    nothing from the deterministic package.  The swapped internal names are bound."""
    mod = _backend(module)
    assert not [n for n in vars(mod) if "det" in _tokens(n)], module
    assert not re.search(
        r"^\s*(?:from|import)\s+\S*cake_dsa_indexer_det_train",
        inspect.getsource(mod),
        re.M,
    ), module
    if module == "cake_backend":
        for name in INTERNAL:
            assert hasattr(mod, name), name


def _defined_functions(module) -> list[tuple[str, object]]:
    """``(qualified name, function)`` of every function and method defined in ``module`` (methods that a decorator
    generated, without source, are left out)."""
    found = {}
    for value in vars(module).values():
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            found[value.__qualname__] = value
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for member in vars(value).values():
                if inspect.isfunction(member):
                    found[member.__qualname__] = member
    out = []
    for name, fn in sorted(found.items()):
        try:
            inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        out.append((name, fn))
    return out


def test_only_the_alert_reads_torch_mode() -> None:
    """The facade never reads torch's deterministic mode; the backend reads it only in ``_alert_nondeterministic``,
    which only the runner preparation (the eager entry's alert, once per call) and ``DSAIndexerNondetRunner.run``
    (every run of a prepared runner) call: preparing a runner and the workspace bound never alert."""
    assert not MODE_READ.search(inspect.getsource(facade))
    backend = _backend()
    functions = _defined_functions(backend)
    readers = [n for n, fn in functions if MODE_READ.search(inspect.getsource(fn))]
    assert readers == ["_alert_nondeterministic"]
    callers = [
        n
        for n, fn in functions
        if "_alert_nondeterministic(" in inspect.getsource(fn)
        and n != "_alert_nondeterministic"
    ]
    assert callers == ["DSAIndexerNondetRunner.run", "_prepare_runner"]
    for name in (
        "prepare_dsa_indexer_nondet_topk",
        "cake_dsa_indexer_nondet_train_topk_workspace_size",
        "plan_dsa_indexer_nondet_topk",
    ):
        assert "_alert_nondeterministic(" not in inspect.getsource(
            getattr(backend, name)
        ), name


def test_import_loads_no_backend() -> None:
    code = f"""
import json, sys
import flashinfer
after_flashinfer = sorted(m for m in sys.modules if m in ({FACADE!r}, {BACKEND!r}))
import {FACADE}
after_facade = sorted(m for m in sys.modules if m.startswith({BACKEND!r}))
print(json.dumps({{"after_flashinfer": after_flashinfer, "after_facade": after_facade}}))
"""
    assert _fresh(code) == {"after_flashinfer": [], "after_facade": []}


def test_the_manifest_names_the_package() -> None:
    backend = _backend()
    document = json.loads(
        (Path(backend.__file__).resolve().parent / "build.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["package"] == BACKEND


def test_shared_mode_name_checks() -> None:
    """The shared #8928 checks (``tests/cake/test_helpers/cake_mode_names.py``) over the entry-point module: the mode word
    and its place in every public name, no ``deterministic`` keyword, and each object bound only under its own name in
    the entry-point module, ``flashinfer``, the backend package and its ``cake_backend``.  The module is new, so there
    are no former modules or names to check."""
    import flashinfer

    backend = _backend()
    assert_cake_mode_module(
        facade,
        namespaces=[flashinfer, importlib.import_module(BACKEND), backend],
        forwards=[(name, getattr(backend, name)) for name in FORWARDED],
    )
