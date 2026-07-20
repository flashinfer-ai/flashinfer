# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Import hygiene for flashinfer.experimental (runtime side of isolation).

Each case runs in a fresh subprocess so module caches can't leak between
assertions:

1. ``import flashinfer`` alone never touches experimental and never warns.
2. ``import flashinfer.experimental.sm12x`` works without a GPU, emits
   exactly one FutureWarning, and is side-effect free: no cutlass/triton
   import, no sm12x compiler import, no ``flashinfer_sm12x`` torch ops.
3. The FutureWarning fires once per process (re-import stays silent), and
   the freeze controls are callable from the namespace root.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def _run(code: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"subprocess failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


CORE_ONLY = """
import json, sys, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import flashinfer
print(json.dumps({
    "experimental_modules": sorted(
        m for m in sys.modules if m.startswith("flashinfer.experimental")
    ),
    "experimental_warnings": [
        str(w.message) for w in caught
        if w.category is FutureWarning and "flashinfer.experimental" in str(w.message)
    ],
}))
"""


def test_core_import_never_touches_experimental():
    data = _run(CORE_ONLY)
    assert data["experimental_modules"] == []
    assert data["experimental_warnings"] == []


NAMESPACE_IMPORT = """
import json, sys, warnings
import flashinfer
import torch

before = set(sys.modules)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import flashinfer.experimental.sm12x as sm12x
    ops = sm12x.list_ops()
newly_loaded = sorted(set(sys.modules) - before)

future_warnings = [
    str(w.message) for w in caught
    if w.category is FutureWarning and "flashinfer.experimental" in str(w.message)
]

with warnings.catch_warnings(record=True) as caught_again:
    warnings.simplefilter("always")
    import importlib
    importlib.import_module("flashinfer.experimental")
    importlib.import_module("flashinfer.experimental.sm12x")
second_warnings = [
    str(w.message) for w in caught_again
    if w.category is FutureWarning and "flashinfer.experimental" in str(w.message)
]

sm12x.freeze_kernel_resolution("hygiene-test")
frozen = sm12x.kernel_resolution_frozen()
sm12x.unfreeze_kernel_resolution()

torch_namespaces = {name.split("::")[0] for name in torch._C._dispatch_get_all_op_names()}

print(json.dumps({
    "n_future_warnings": len(future_warnings),
    "n_second_warnings": len(second_warnings),
    "newly_loaded_cutlass": [m for m in newly_loaded if m == "cutlass" or m.startswith("cutlass.")],
    "newly_loaded_triton": [m for m in newly_loaded if m == "triton" or m.startswith("triton.")],
    "compiler_loaded": "flashinfer.experimental.sm12x._lib.compiler" in sys.modules,
    "intrinsics_loaded": "flashinfer.experimental.sm12x._lib.intrinsics" in sys.modules,
    "n_ops": len(ops),
    "torch_ops_registered": "flashinfer_sm12x" in torch_namespaces,
    "freeze_roundtrip": frozen,
}))
"""


def test_namespace_import_is_side_effect_free():
    data = _run(NAMESPACE_IMPORT)
    assert data["n_future_warnings"] == 1, "exactly one FutureWarning on first import"
    assert data["n_second_warnings"] == 0, "re-import must not warn again"
    assert data["newly_loaded_cutlass"] == [], "namespace import must not pull cutlass"
    assert data["newly_loaded_triton"] == [], "namespace import must not pull triton"
    assert not data["compiler_loaded"], "sm12x compiler must load on first op use only"
    assert not data["intrinsics_loaded"], (
        "kernel intrinsics must load on first op use only"
    )
    assert not data["torch_ops_registered"], "torch ops must register on op import only"
    assert data["freeze_roundtrip"] is True
    assert isinstance(data["n_ops"], int)


QUIET_IMPORT = """
import json, os, warnings
os.environ["FLASHINFER_EXP_QUIET"] = "1"
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import flashinfer.experimental.sm12x
print(json.dumps({
    "warnings": [
        str(w.message) for w in caught
        if w.category is FutureWarning and "flashinfer.experimental" in str(w.message)
    ],
}))
"""


def test_quiet_env_silences_future_warning():
    data = _run(QUIET_IMPORT)
    assert data["warnings"] == []
