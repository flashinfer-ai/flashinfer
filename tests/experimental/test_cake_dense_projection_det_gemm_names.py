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
"""

# The dense projection GEMM package's names: the canonical (det) names, the former names kept as aliases of the
# same objects (functions, classes), the former import path (now the package with an explicit numerical mode,
# whose default runs these canonical objects), and no former spelling left in the generated registry, manifest or
# sources.  CPU only: nothing here launches a kernel.

import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

PACKAGE = "flashinfer.experimental.dense_projection_det_gemm"
FORMER_PACKAGE = "flashinfer.experimental.dense_projection_gemm"
# former name -> canonical name of the public entry points of ``cake_backend``
API = {
    "GemmPlan": "DetGemmPlan",
    "PreparedGemm": "PreparedDetGemm",
    "PreparedRouterGemm": "PreparedRouterDetGemm",
    "RouterPlan": "RouterDetPlan",
    "dense_projection_gemm": "dense_projection_det_gemm",
    "plan_dense_projection_gemm": "plan_dense_projection_det_gemm",
    "plan_router_fp32_gemm": "plan_router_fp32_det_gemm",
    "prepare_dense_projection_gemm": "prepare_dense_projection_det_gemm",
    "prepare_projection_wgrad": "prepare_projection_det_wgrad",
    "prepare_router_fp32_gemm": "prepare_router_fp32_det_gemm",
    "projection_dgrad": "projection_det_dgrad",
    "projection_forward": "projection_det_forward",
    "projection_wgrad": "projection_det_wgrad",
    "router_dgrad": "router_det_dgrad",
    "router_forward": "router_det_forward",
    "router_fp32_gemm": "router_fp32_det_gemm",
    "router_wgrad": "router_det_wgrad",
    "split_fp32_to_bf16x3_": "split_fp32_to_bf16x3_det_",
}
LOADER = (
    "load_cake_dense_projection_gemm_module",
    "load_cake_dense_projection_det_gemm_module",
)
FORMER_TOKENS = (
    "dense_projection_gemm",
    "dense_proj_gemm",
    "router_fp32_gemm",
    "split_fp32_bf16x3",
    "DENSE_PROJ_GEMM",
)
MODULES = ("cake_backend", "cake_jit", "cake_build")


def _module(package: str, name: str):
    return importlib.import_module(f"{package}.{name}")


@pytest.mark.parametrize("former, canonical", sorted(API.items()))
def test_former_names_are_the_canonical_objects(former, canonical):
    backend = _module(PACKAGE, "cake_backend")
    obj = getattr(backend, canonical)
    assert getattr(backend, former) is obj
    assert obj.__name__ == canonical
    assert inspect.unwrap(obj).__name__ == canonical
    assert obj.__module__ == f"{PACKAGE}.cake_backend"
    former_backend = _module(FORMER_PACKAGE, "cake_backend")
    assert former_backend.DETERMINISTIC_NAMES[former] == canonical
    if inspect.isclass(obj):
        assert getattr(former_backend, former) is obj
    else:
        entry = getattr(former_backend, former)
        assert entry is not obj
        assert inspect.signature(entry).parameters["deterministic"].default is True


# former entry point -> its positional arguments (sentinels)
ARITY = {
    "dense_projection_gemm": 3,
    "plan_dense_projection_gemm": 3,
    "plan_router_fp32_gemm": 3,
    "prepare_dense_projection_gemm": 3,
    "prepare_projection_wgrad": 3,
    "prepare_router_fp32_gemm": 3,
    "projection_dgrad": 2,
    "projection_forward": 2,
    "projection_wgrad": 2,
    "router_dgrad": 2,
    "router_forward": 2,
    "router_fp32_gemm": 3,
    "router_wgrad": 2,
    "split_fp32_to_bf16x3_": 3,
}


@pytest.mark.parametrize("former", sorted(ARITY))
def test_the_former_path_defaults_to_the_canonical_object(monkeypatch, former):
    """The default of the former path's entry point is the canonical object with the caller's arguments: the same
    program, so the same bits (the numerical mode keeps the deterministic default)."""
    calls = []
    result = object()

    def canonical(*args, **kwargs):
        calls.append(args)
        return result

    monkeypatch.setattr(_module(PACKAGE, "cake_backend"), API[former], canonical)
    args = tuple(object() for _ in range(ARITY[former]))
    assert getattr(_module(FORMER_PACKAGE, "cake_backend"), former)(*args) is result
    assert len(calls) == 1 and calls[0][: len(args)] == args


def test_the_loader_keeps_its_former_name():
    jit = _module(PACKAGE, "cake_jit")
    loader = getattr(jit, LOADER[1])
    assert getattr(jit, LOADER[0]) is loader
    assert inspect.unwrap(loader).__name__ == LOADER[1]


@pytest.mark.parametrize("name", MODULES)
def test_the_former_package_path_is_its_own_package(name):
    """The former import path is the package with the numerical mode (its own modules and programs); the
    canonical package never imports it."""
    assert _module(FORMER_PACKAGE, name) is not _module(PACKAGE, name)
    former = importlib.import_module(FORMER_PACKAGE)
    assert Path(former.__file__).parent.name == "dense_projection_gemm"


def _former_lines(text: str) -> list[str]:
    """The lines of ``text`` spelling a former token, an alias assignment of the former names aside."""
    alias = re.compile(r"^(?P<former>\w+) = (?P<canonical>\w+)$")
    out = []
    for line in text.split("\n"):
        if not any(token in line for token in FORMER_TOKENS):
            continue
        match = alias.match(line)
        known = dict(API, **{LOADER[0]: LOADER[1]})
        if match and known.get(match["former"]) == match["canonical"]:
            continue
        out.append(line)
    return out


def test_the_registry_spells_only_canonical_names():
    jit = _module(PACKAGE, "cake_jit")
    backend = _module(PACKAGE, "cake_backend")
    assert jit.MODULES and jit.KERNELS
    for name, record in jit.MODULES.items():
        assert (
            name.startswith("cake_dense_projection_det_gemm_")
            and record["library"] == name
        )
        assert record["kernel"] in (f"kernel_{name}",) or record["kernel"].startswith(
            "cake::dense_projection_det_gemm::"
        )
        assert record["template"].startswith(
            ("dense_proj_det_gemm_", "router_fp32_det_gemm_", "split_fp32_det_bf16x3")
        )
        assert (
            "build" not in record
            or ".dense_projection_gemm" not in record["build"]["module"]
        )
    templates = {t for table in jit.KERNELS.values() for t in table}
    assert backend.SPLIT_TEMPLATE in templates
    assert {
        backend.router_symbol(a, b, t)
        for a, b, t in ((False, False, False), (False, True, False), (True, True, True))
    } <= templates
    assert backend._TEMPLATE_PREFIX == "dense_proj_det_gemm_"
    assert all(
        t.startswith(backend._TEMPLATE_PREFIX)
        for t in templates
        if t.startswith("dense_proj")
    )
    assert _former_lines(json.dumps([jit.MODULES, jit.KERNELS])) == []


def test_no_former_spelling_in_the_package():
    root = Path(_module(PACKAGE, "cake_jit").__file__).parent
    manifest = json.loads((root / "build.json").read_text())
    assert manifest["package"] == PACKAGE
    for entry in manifest["libraries"]:
        assert entry["name"].startswith("cake_dense_projection_det_gemm_")
        assert all(
            s.startswith(
                ("csrc/cake_dense_projection_det_gemm/", "csrc/cake_device_helpers/")
            )
            for s in entry["sources"]
        )
    files = [
        root / "build.json",
        *sorted(root.glob("*.py")),
        *sorted((root / "csrc").rglob("*")),
    ]
    found = {
        str(path.relative_to(root)): lines
        for path in files
        if path.is_file() and (lines := _former_lines(path.read_text(errors="replace")))
    }
    assert found == {}
    assert not [
        p for p in (root / "csrc").rglob("*") if any(t in p.name for t in FORMER_TOKENS)
    ]
