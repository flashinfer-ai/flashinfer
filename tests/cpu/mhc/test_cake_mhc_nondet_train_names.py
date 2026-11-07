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

Names of the non-deterministic mHC training module ``flashinfer.cake_mhc_nondet_train``
(FlashInfer issue #8928): every exported name but the ALL_CAPS constants carries
the mode token ``nondet`` (``Nondet`` in a class name), no public callable takes
a ``deterministic`` keyword (the ``forward`` / ``backward`` of the autograd
``Function`` classes included), and the former mHC training modules and names
are gone (no module, no alias, no top-level name).
"""

import inspect
import json
import re
import subprocess
import sys

import pytest

import flashinfer
import flashinfer.cake_mhc_nondet_train as module

MODE = "nondet"
# The former modules of the mHC training modes (FlashInfer 0.7.3's package names, the backend paths before
# #8928, the former non-deterministic module): importing them fails.
REMOVED_MODULES = (
    "flashinfer.cake_mhc_train",
    "flashinfer.cake_mhc_train_backend",
    "flashinfer.cake_mhc_train_head_backend",
    "flashinfer.cake_mhc_det_train_backend",
    "flashinfer.cake_mhc_det_train_head_backend",
    "flashinfer.experimental.cake.cake_mhc_train",
    "flashinfer.experimental.cake.cake_mhc_train_backend",
)
# The former names: their prefixes (0.7.3's names without the mode word, the classes and head names before
# #8928) and the generic route helpers.
REMOVED_PREFIXES = (
    "cake_mhc_train_",
    "is_cake_mhc_train_",
    "CakeMHCTrain",
    "CakeMHCDetTrain",
    "cake_mhc_det_train_head",
    "is_cake_mhc_det_train_head",
)
REMOVED_NAMES = ("route_key", "route_modules")

_CHILD = r"""
import importlib, json, sys

out = {}
for name in sys.argv[1:]:
    try:
        imported = importlib.import_module(name)
    except ImportError:
        out[name] = None
    else:
        # a namespace package left by a stale __pycache__ directory has no origin: it holds no code
        spec = getattr(imported, "__spec__", None)
        out[name] = getattr(spec, "origin", None)
print(json.dumps(out))
"""


def _mode_tokens(name: str) -> set[str]:
    """The mode words of ``name``: ``det`` / ``nondet`` as ``_``-separated tokens of a snake_case name, ``Det`` /
    ``Nondet`` as words of a CamelCase one (a substring test would read ``nondet`` as ``det`` too)."""
    if name[:1].isupper():
        return {word.lower() for word in re.findall(r"(Nondet|Det)(?=[A-Z]|$)", name)}
    return set(re.findall(r"(?:^|_)(det|nondet)(?=_|$)", name))


@pytest.mark.cpu_only
def test_every_name_carries_the_mode() -> None:
    assert module.__all__
    without = [
        n for n in module.__all__ if not n.isupper() and _mode_tokens(n) != {MODE}
    ]
    assert not without, without


@pytest.mark.cpu_only
def test_no_public_callable_takes_a_mode_keyword() -> None:
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isclass(obj):
            functions = (obj.forward, obj.backward)
        elif callable(obj):
            functions = (obj,)
        else:
            assert name.isupper(), name  # only the constants are not callable
            continue
        for function in functions:
            parameters = inspect.signature(function).parameters.values()
            assert all(p.name != "deterministic" for p in parameters), name
            assert all(p.kind is not p.VAR_KEYWORD for p in parameters), name


@pytest.mark.cpu_only
def test_former_modules_fail_at_import() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, *REMOVED_MODULES],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == dict.fromkeys(REMOVED_MODULES), out


@pytest.mark.cpu_only
def test_former_names_are_gone() -> None:
    for owner in (module, flashinfer):
        leftovers = [
            n
            for n in dir(owner)
            if n.startswith(REMOVED_PREFIXES) or n in REMOVED_NAMES
        ]
        assert not leftovers, (owner.__name__, leftovers)
    # the mode has no top-level name: its module is imported by its full name (importing it binds the submodule
    # attribute ``flashinfer.cake_mhc_nondet_train``, which is the module, not a re-exported name)
    leftovers = [
        n
        for n in dir(flashinfer)
        if "mhc" in n.lower()
        and MODE in n.lower()
        and not inspect.ismodule(getattr(flashinfer, n))
    ]
    assert not leftovers, leftovers
