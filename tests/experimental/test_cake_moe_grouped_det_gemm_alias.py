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

The names of the Cake MoE grouped GEMM before the deterministic-family rename.

The package was ``flashinfer.experimental.cake_moe_grouped_gemm``; it is now
``flashinfer.experimental.cake_moe_grouped_det_gemm`` and its functions carry
``det`` names.  The old package path (with its submodules) and the old names of
the package and of ``flashinfer.grouped_mm`` stay as deprecated aliases: they
resolve, on first use, to the very same objects, they stay out of ``__all__``,
and importing the old package path imports nothing of the new package.  CPU
only.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("tvm_ffi")  # cake_backend imports it at module level

import flashinfer.experimental.cake_moe_grouped_det_gemm as pkg
import flashinfer.grouped_mm as grouped_mm
from tests.test_helpers.cake_package import cake_package

REPO_ROOT = Path(__file__).resolve().parents[2]
OLD = "flashinfer.experimental.cake_moe_grouped_gemm"
NEW = "flashinfer.experimental.cake_moe_grouped_det_gemm"

# module -> {old name: det name}
RENAMED = {
    NEW: {
        "CakeGroupedMm": "CakeGroupedDetMm",
        "GroupedGemmLaunch": "GroupedDetGemmLaunch",
        "cake_grouped_mm": "cake_grouped_det_mm",
        "grouped_gemm_dgrad": "grouped_det_gemm_dgrad",
        "grouped_gemm_fwd": "grouped_det_gemm_fwd",
        "grouped_gemm_wgrad": "grouped_det_gemm_wgrad",
        "prepare_grouped_gemm_dgrad": "prepare_grouped_det_gemm_dgrad",
        "prepare_grouped_gemm_fwd": "prepare_grouped_det_gemm_fwd",
        "prepare_grouped_gemm_wgrad": "prepare_grouped_det_gemm_wgrad",
    },
    f"{NEW}.cake_backend": {
        "GroupedGemmLaunch": "GroupedDetGemmLaunch",
        "grouped_mm_bf16_cake": "grouped_det_mm_bf16_cake",
        "prepare_grouped_gemm_dgrad": "prepare_grouped_det_gemm_dgrad",
        "prepare_grouped_gemm_fwd": "prepare_grouped_det_gemm_fwd",
        "prepare_grouped_gemm_wgrad": "prepare_grouped_det_gemm_wgrad",
    },
    f"{NEW}.cake_autograd": {
        "CakeGroupedMm": "CakeGroupedDetMm",
        "cake_grouped_mm": "cake_grouped_det_mm",
    },
    "flashinfer.grouped_mm": {
        "cake_grouped_mm": "cake_grouped_det_mm",
        "grouped_gemm_dgrad": "grouped_det_gemm_dgrad",
        "grouped_gemm_fwd": "grouped_det_gemm_fwd",
        "grouped_gemm_wgrad": "grouped_det_gemm_wgrad",
    },
    "flashinfer.grouped_mm.cake": {
        "cake_grouped_mm": "cake_grouped_det_mm",
        "grouped_gemm_dgrad": "grouped_det_gemm_dgrad",
        "grouped_gemm_fwd": "grouped_det_gemm_fwd",
        "grouped_gemm_wgrad": "grouped_det_gemm_wgrad",
    },
}
SUBMODULES = ("cake_autograd", "cake_backend", "cake_build", "cake_jit")


@pytest.mark.parametrize("module_name", sorted(RENAMED))
def test_old_names_are_the_det_objects(module_name):
    module = importlib.import_module(module_name)
    for old, new in RENAMED[module_name].items():
        assert getattr(module, old) is getattr(module, new), (module_name, old)
        assert old not in getattr(module, "__all__", ()), (module_name, old)
    exported = getattr(module, "__all__", None)
    if exported is not None:
        assert set(RENAMED[module_name].values()) <= set(exported), module_name
    with pytest.raises(AttributeError):
        module.no_such_name  # noqa: B018


def test_old_package_path_is_the_det_package():
    old = importlib.import_module(OLD)
    for name in pkg.__all__:
        assert getattr(old, name) is getattr(pkg, name), name
    for name, new in RENAMED[NEW].items():
        assert getattr(old, name) is getattr(pkg, new), name
    for sub in SUBMODULES:
        det = importlib.import_module(f"{NEW}.{sub}")
        assert importlib.import_module(f"{OLD}.{sub}") is det, sub
        assert getattr(old, sub) is det, sub
        # the det module keeps its own identity
        assert det.__name__ == f"{NEW}.{sub}" and det.__spec__.name == det.__name__
    from flashinfer.experimental.cake_moe_grouped_gemm.cake_jit import record_name

    assert record_name("fwd") == "cake_moe_grouped_det_gemm_fwd"
    with pytest.raises(AttributeError):
        old.no_such_name  # noqa: B018


def test_old_suite_name_is_the_det_package():
    assert cake_package("cake_moe_grouped_gemm") is pkg
    for sub in SUBMODULES:
        assert cake_package("cake_moe_grouped_gemm", sub) is cake_package(
            "cake_moe_grouped_det_gemm", sub
        )


def test_grouped_mm_old_names_are_the_det_entry_points():
    for old, new in RENAMED["flashinfer.grouped_mm"].items():
        entry = getattr(grouped_mm, old)
        assert entry is getattr(grouped_mm, new)
        assert entry.experimental_feature == f"cake_moe_grouped_det_gemm.{new}"


_LAZY = """
import json, sys

def loaded(prefix):
    return sorted(m for m in sys.modules if m == prefix or m.startswith(prefix + "."))

import flashinfer
import flashinfer.grouped_mm
first = {"old": loaded("{OLD}"), "new": loaded("{NEW}")}
import {OLD} as old
second = {"old": loaded("{OLD}"), "new": loaded("{NEW}")}
fn = old.grouped_gemm_fwd
third = {"new": loaded("{NEW}")}
print(json.dumps([first, second, third, fn.__module__]))
"""


def test_the_aliases_import_lazily():
    """``import flashinfer`` / ``flashinfer.grouped_mm`` load neither package;
    importing the old package path loads only itself; its first attribute access
    loads the det package root (whose one-shot functions need no submodule)."""
    code = _LAZY.replace("{OLD}", OLD).replace("{NEW}", NEW)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    first, second, third, module = json.loads(proc.stdout.strip().splitlines()[-1])
    assert first == {"old": [], "new": []}
    assert second == {"old": [OLD], "new": []}
    assert third == {"new": [NEW]}
    assert module == NEW
