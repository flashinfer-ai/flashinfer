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

Naming rules of flashinfer-ai/flashinfer#8928 for the deterministic Cake DSA indexer.

The public module is ``flashinfer.cake_dsa_indexer_det_train`` and its backend package
``flashinfer.experimental.cake.cake_dsa_indexer_det_train``.  Every public name of the module (except ALL_CAPS
module constants) carries the mode word ``det`` as a name token, no public callable takes ``deterministic=``, the
previous names and package paths fail at import (no aliases), the SM120 serving entry
``flashinfer.dsa_indexer.dsa_indexer_det_topk`` no longer routes ``backend="cake"``, and importing the module loads
no backend (checked in a fresh interpreter).  CPU only.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys

import pytest
import torch

import flashinfer.cake_dsa_indexer_det_train as facade
import flashinfer.dsa_indexer as sm120_entry

pytestmark = pytest.mark.cpu_only

MODE = "det"
FACADE = "flashinfer.cake_dsa_indexer_det_train"
BACKEND = "flashinfer.experimental.cake.cake_dsa_indexer_det_train"
CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
# the backend objects the module forwards to carry the module's names
FORWARDED = (
    "cake_dsa_indexer_det_train_topk",
    "cake_dsa_indexer_det_train_topk_workspace_size",
)
# package paths before #8928: they fail at import
OLD_MODULES = (
    "flashinfer.experimental.cake_dsa_indexer_det",
    "flashinfer.experimental.cake_dsa_indexer_det.cake_backend",
    "flashinfer.experimental.cake_dsa_indexer",
    "flashinfer.experimental.cake_dsa_indexer.cake_backend",
)
# (module, name) pairs before #8928: absent
OLD_NAMES = (
    ("flashinfer.dsa_indexer", "dsa_indexer_det_topk_workspace_size"),
    ("flashinfer.dsa_indexer", "dsa_indexer_topk"),
    ("flashinfer.dsa_indexer", "dsa_indexer_topk_workspace_size"),
    (f"{BACKEND}.cake_backend", "dsa_indexer_det_topk"),
    (f"{BACKEND}.cake_backend", "dsa_indexer_det_workspace_size"),
    (f"{BACKEND}.cake_backend", "DSAIndexerRunner"),
    (f"{BACKEND}.cake_backend", "dsa_indexer_topk"),
    (f"{BACKEND}.cake_backend", "dsa_indexer_workspace_size"),
    (f"{BACKEND}.cake_backend", "plan_dsa_indexer_topk"),
    (f"{BACKEND}.cake_backend", "prepare_dsa_indexer_topk"),
    (f"{BACKEND}.cake_backend", "validate_dsa_indexer_inputs"),
)


def _tokens(name: str) -> list[str]:
    """Name tokens: split on ``_`` and on CamelCase boundaries (``CakeDSAIndexerDet`` -> cake, dsa, indexer, det)."""
    out = []
    for chunk in name.split("_"):
        out += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", chunk)
    return [t.lower() for t in out]


def _public_names() -> list[str]:
    return [n for n in facade.__all__ if not CONSTANT.match(n)]


def _backend():
    import importlib

    return importlib.import_module(f"{BACKEND}.cake_backend")


def _fresh(code: str) -> dict:
    """Run ``code`` in a fresh interpreter; it prints one JSON document as its last line."""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_tokenizer_separates_the_mode_words() -> None:
    assert _tokens("cake_dsa_indexer_det_train_topk")[3] == "det"
    assert _tokens("CakeDSAIndexerDetRunner") == [
        "cake",
        "dsa",
        "indexer",
        "det",
        "runner",
    ]
    assert "det" not in _tokens("cake_dsa_indexer_nondet_train")


def test_every_public_name_carries_the_mode_word() -> None:
    names = _public_names()
    assert sorted(names) == sorted(FORWARDED)
    for name in names:
        assert hasattr(facade, name), name
        tokens = _tokens(name)
        assert MODE in tokens and "nondet" not in tokens, name


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
    return [(n, o) for n, o in objs if callable(o)]


def test_no_public_callable_takes_deterministic() -> None:
    callables = _callables()
    assert len(callables) == 2 * len(FORWARDED)
    for name, obj in callables:
        signature = inspect.signature(obj)
        assert "deterministic" not in signature.parameters, name
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)


def test_public_entry_points_are_stable_apis() -> None:
    for name in _public_names():
        assert not getattr(getattr(facade, name), "is_experimental", False), name


def test_old_package_paths_fail_at_import() -> None:
    code = f"""
import importlib, json
present = []
for name in {list(OLD_MODULES)!r}:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        continue
    if getattr(module, "__file__", None) is None:  # a namespace left by a stale __pycache__ counts as absent
        continue
    present.append(name)
print(json.dumps({{"present": present}}))
"""
    assert _fresh(code) == {"present": []}


@pytest.mark.parametrize("module,name", OLD_NAMES)
def test_old_names_are_absent(module: str, name: str) -> None:
    import importlib

    mod = importlib.import_module(module)
    assert not hasattr(mod, name)
    assert name not in getattr(mod, "__all__", ())
    with pytest.raises(ImportError):
        exec(f"from {module} import {name}", {})


def test_sm120_entry_no_longer_routes_cake() -> None:
    q = torch.zeros((6, 32, 128), dtype=torch.bfloat16)
    k = torch.zeros((10, 128), dtype=torch.bfloat16)
    w = torch.zeros((6, 32), dtype=torch.float32)
    cu_q = torch.tensor((0, 2, 6), dtype=torch.int32)
    cu_k = torch.tensor((0, 3, 10), dtype=torch.int32)
    with pytest.raises(ValueError, match=r"flashinfer\.cake_dsa_indexer_det_train"):
        sm120_entry.dsa_indexer_det_topk(q, k, w, cu_q, cu_k, backend="cake")
    with pytest.raises(TypeError, match="backend"):
        sm120_entry.dsa_indexer_det_topk(q, k, w, cu_q, cu_k)
    with pytest.raises(ValueError, match="sm120"):
        sm120_entry.dsa_indexer_det_topk(q, k, w, cu_q, cu_k, backend="other")


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
    from pathlib import Path

    backend = _backend()
    document = json.loads(
        (Path(backend.__file__).resolve().parent / "build.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["package"] == BACKEND
