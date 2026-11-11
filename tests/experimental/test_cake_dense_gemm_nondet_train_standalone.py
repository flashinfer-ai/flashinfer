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

Standalone-build tests of the non-deterministic Cake dense GEMM backend
(flashinfer-ai/flashinfer#6553, #8928).

The backend directory declares one dependency: the deterministic backend
directory ``cake_dense_gemm_det_train``, shipped next to it (``build.json``:
``common.requires.packages``).  In a ``python -I``
subprocess with ``flashinfer`` blocked, the directory loads as a top-level
package under its own name, and its ``cake_backend``, ``cake_jit`` and
``cake_build`` import without any ``flashinfer`` module.  With the deterministic
directory loaded as a top-level package beside it, the first use resolves that
sibling; without it, importing still succeeds and the first use raises an
``ImportError`` that names the missing directory.  With both directories, the
planners of both packages run under the blocker, and loading a program that is
not prebuilt raises the error that names the missing FlashInfer JIT.  CPU only:
no GPU, no generated program and no JIT build.
"""

import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

tvm_ffi = pytest.importorskip("tvm_ffi")  # cake_backend imports it at module level

ND_DIR = Path(cake_package("cake_dense_gemm_nondet_train").__file__).resolve().parent
DET_DIR = Path(cake_package("cake_dense_gemm_det_train").__file__).resolve().parent


def _dependency_paths() -> list[str]:
    """Where ``torch`` and ``tvm_ffi`` live, plus the interpreter's site-packages (``python -I`` drops
    ``PYTHONPATH`` and the user site)."""
    paths = list(site.getsitepackages())
    for module in (torch, tvm_ffi):
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(module.__file__))))
    return list(dict.fromkeys(paths))


# The child's preamble: ``flashinfer`` blocked, the two directories loadable as top-level packages.
_BLOCKED = r"""
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys

ND_DIR, DET_DIR = sys.argv[1], sys.argv[2]
for p in sys.argv[3:]:
    if p not in sys.path:
        sys.path.append(p)


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError(f"flashinfer is blocked in this test: {name}")
        return None


sys.meta_path.insert(0, _Block())
sys.modules["flashinfer"] = None


def load_top_level(directory):
    name = os.path.basename(directory)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(directory, "__init__.py"), submodule_search_locations=[directory]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def flashinfer_modules():
    return sorted(
        n for n, m in sys.modules.items() if (n == "flashinfer" or n.startswith("flashinfer.")) and m is not None
    )
"""

_STANDALONE = (
    _BLOCKED
    + r"""
nd = load_top_level(ND_DIR)
det = load_top_level(DET_DIR) if DET_DIR else None
backend, jit, build = (importlib.import_module(f"{nd.__name__}.{sub}") for sub in ("cake_backend", "cake_jit", "cake_build"))
facts = {
    "package": nd.__name__,
    "submodules": sorted(m.__name__ for m in (backend, jit, build)),
    "det_package": jit.DETERMINISTIC_PACKAGE,
    "det_loaded_before_use": sorted(n for n in sys.modules if n.startswith(jit.DETERMINISTIC_PACKAGE + ".")),
    "flashinfer_after_import": flashinfer_modules(),
}
try:
    resolved = jit.deterministic_module("cake_backend")
except Exception as exc:  # the case without the sibling: report the error
    facts["error"] = [type(exc).__name__, str(exc)]
    try:
        backend.deterministic_backend()
    except Exception as again:
        facts["backend_error"] = [type(again).__name__, str(again)]
else:
    facts["resolved"] = resolved.__name__
    facts["resolved_file"] = os.path.realpath(resolved.__file__)
    facts["resolved_is_loaded_sibling"] = resolved is sys.modules.get(f"{det.__name__}.cake_backend")
    facts["backend_resolves_same"] = backend.deterministic_backend() is resolved
facts["flashinfer_after_use"] = flashinfer_modules()
print(json.dumps(facts))
"""
)


# Under the blocker, with the deterministic directory beside it: the planners of both packages on CPU views (no
# device: the architecture and SM count given), then the load of each plan's program -- no library is prebuilt in the
# checkout, and the fallback to FlashInfer's JIT is blocked.
_PLANS = (
    _BLOCKED
    + r"""
import torch

nd, det = load_top_level(ND_DIR), load_top_level(DET_DIR)
backend, jit, build = (importlib.import_module(f"{nd.__name__}.{sub}") for sub in ("cake_backend", "cake_jit", "cake_build"))
det_jit = importlib.import_module(f"{det.__name__}.cake_jit")
det_build = importlib.import_module(f"{det.__name__}.cake_build")
arch, sm_count, l2_bytes = "sm_103a", 152, 126 << 20
bf16, T, K, N, E = torch.bfloat16, 1000, 4096, 128, 256
X, W, G = torch.empty(T, K, dtype=bf16), torch.empty(N, K, dtype=bf16), torch.empty(T, N, dtype=bf16)
facts = {}


def load(name, record):
    # the program's load: the prebuilt library when present, else FlashInfer's JIT (blocked)
    library = record[0].library_name(name, arch)
    if record[1].prebuilt_path(library).is_file():
        return ["prebuilt", library]
    try:
        record[2](name, arch)
    except Exception as exc:
        return [type(exc).__name__, str(exc), isinstance(exc.__cause__, ImportError), library]
    return ["loaded", library]


# this package's weight-gradient program at an explicit split count: the transposed-store GEMM X.T @ G
plan = backend.cake_dense_gemm_nondet_train_projection_plan(
    X.t(), G, torch.empty(N, K), transposed_out=True, splits=2, arch=arch, sm_count=sm_count
)
facts["wgrad_plan"] = [type(plan).__name__, plan.template, plan.splits, plan.reproducible]
facts["wgrad_load"] = load(jit.select_module(arch, plan.template), (jit, build, jit.load_cake_dense_projection_gemm_module))
# a forward: the deterministic package's plan (its planner, resolved as the sibling)
dplan = backend.cake_dense_gemm_nondet_train_projection_plan(
    X, W.t(), torch.empty(T, N, dtype=bf16), arch=arch, sm_count=sm_count, l2_bytes=l2_bytes
)
dplan = dplan[0] if isinstance(dplan, tuple) else dplan
facts["fwd_plan"] = [type(dplan).__name__, dplan.template]
facts["fwd_load"] = load(
    det_jit.select_module(arch, dplan.template), (det_jit, det_build, det_jit.load_cake_dense_projection_det_gemm_module)
)
# the router's weight gradient at an explicit split count (the FP32 GEMM G.T @ X)
rplan, reason = backend.plan_atomic_router(
    torch.empty(T, E).t(), torch.empty(T, 5120), torch.empty(E, 5120), arch=arch, sm_count=sm_count, splits=2
)
facts["router_plan"] = [type(rplan).__name__, getattr(rplan, "template", None), getattr(rplan, "units", None), reason]
try:
    build.jit_specs(arch)
except Exception as exc:
    facts["jit_specs_error"] = [type(exc).__name__, str(exc)]
facts["flashinfer_after_use"] = flashinfer_modules()
print(json.dumps(facts))
"""
)


def _standalone(det_dir: str, script: str = _STANDALONE) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(ND_DIR),
            det_dir,
            *_dependency_paths(),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_build_json_declares_the_deterministic_directory():
    """The one dependency outside the directory is declared for vendoring builds: ``common.requires.packages`` holds
    the sibling directory names."""
    manifest = json.loads((ND_DIR / "build.json").read_text())
    assert (
        manifest["common"]["requires"]["packages"]
        == [DET_DIR.name]
        == ["cake_dense_gemm_det_train"]
    )


def test_the_backend_loads_as_a_top_level_package_and_resolves_its_top_level_sibling():
    facts = _standalone(str(DET_DIR))
    name = ND_DIR.name
    assert facts["package"] == name
    assert facts["submodules"] == sorted(
        f"{name}.{sub}" for sub in ("cake_backend", "cake_build", "cake_jit")
    )
    # the dependency is imported lazily: loading the package and its submodules imports none of its modules
    assert facts["det_loaded_before_use"] == []
    assert facts["flashinfer_after_import"] == []
    # first use: the top-level sibling cake_dense_gemm_det_train, from the directory shipped next to it
    assert facts["det_package"] == DET_DIR.name == "cake_dense_gemm_det_train"
    assert facts["resolved"] == f"{DET_DIR.name}.cake_backend"
    assert facts["resolved_file"].startswith(os.path.realpath(DET_DIR) + os.sep)
    assert facts["resolved_is_loaded_sibling"] is True
    assert facts["backend_resolves_same"] is True
    assert facts["flashinfer_after_use"] == []


def test_without_the_deterministic_directory_import_succeeds_and_first_use_names_it():
    facts = _standalone("")
    assert facts["package"] == ND_DIR.name
    assert facts["flashinfer_after_import"] == []
    assert "resolved" not in facts
    for key in ("error", "backend_error"):
        kind, message = facts[key]
        assert kind == "ImportError", facts[key]
        assert "cake_dense_gemm_det_train" in message
        assert "use_deterministic_backend" in message
    assert facts["flashinfer_after_use"] == []


def test_the_backend_plans_under_the_blocker_and_names_the_missing_jit():
    """With ``flashinfer`` blocked the planners of both packages run on CPU views: this package's weight-gradient
    plan at two splits (bitwise reproducible: one add of two values), the deterministic package's forward plan through
    the sibling, the router planner.  Loading a planned program whose library is not prebuilt under ``aot/`` takes the
    fallback to FlashInfer's JIT, which the blocker refuses: a ``RuntimeError`` that names the library and the missing
    JIT, raised from the ``ImportError`` -- in both packages -- and no ``flashinfer`` module is loaded."""
    facts = _standalone(str(DET_DIR), _PLANS)
    kind, template, splits, reproducible = facts["wgrad_plan"]
    assert kind == "CakeDenseGemmNondetProjectionPlan"
    assert template.startswith("dense_proj_gemm_atomic_nn_n")
    assert template.endswith("_f32_t")
    assert (splits, reproducible) == (2, True)
    kind, template = facts["fwd_plan"]
    assert kind == "CakeDenseGemmDetProjectionPlan" and template
    kind, template, units, reason = facts["router_plan"]
    assert kind == "CakeDenseGemmNondetRouterPlan", reason
    assert template.startswith("router_fp32_gemm_") and units
    for key in ("wgrad_load", "fwd_load"):
        load = facts[key]
        # a tree with prebuilt libraries: nothing to check here
        if load[0] == "prebuilt":
            continue
        kind, message, from_import_error, library = load
        assert kind == "RuntimeError", load
        assert library in message and "is not prebuilt" in message, load
        assert "flashinfer's JIT is not available" in message, load
        assert from_import_error is True, load
    if "jit_specs_error" in facts:  # some library of the architecture is not prebuilt
        kind, message = facts["jit_specs_error"]
        assert kind == "RuntimeError" and "flashinfer's JIT is not available" in message
    assert facts["flashinfer_after_use"] == []
