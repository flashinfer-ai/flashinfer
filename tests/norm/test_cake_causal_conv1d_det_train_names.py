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

Public names of the Cake deterministic causal conv1d + SiLU training packages.

Every module, package and entry point of the deterministic packages carries ``det`` in its name, and the names
these had before the det rename are gone: the modules ``flashinfer.cake_causal_conv1d_train`` and
``flashinfer.cake_causal_conv1d_train_backend`` (and its submodules) do not import, and no namespace (the
``flashinfer`` top level, ``flashinfer.cake_causal_conv1d_det_train``, its backend package, the KCP facade
``flashinfer.cake_causal_conv1d_det_train_kcp`` and the KCP backend module) binds an old name.  Importing a
single-rank facade or package loads no KCP host module (checked in fresh interpreters by module file).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer
import flashinfer.cake_causal_conv1d_det_train as facade
import flashinfer.cake_causal_conv1d_det_train_backend as backend
import flashinfer.cake_causal_conv1d_det_train_kcp as kcp_facade

pytestmark = pytest.mark.cpu_only

FACADE = "flashinfer.cake_causal_conv1d_det_train"
PACKAGE = "flashinfer.cake_causal_conv1d_det_train_backend"
KCP_FACADE = "flashinfer.cake_causal_conv1d_det_train_kcp"
SUBMODULES = ("cake_build", "cake_jit", "kcp", "kcp_jit")
REMOVED_MODULES = (
    "flashinfer.cake_causal_conv1d_train",
    "flashinfer.cake_causal_conv1d_train_backend",
    *(f"flashinfer.cake_causal_conv1d_train_backend.{m}" for m in SUBMODULES),
)

# (removed name, det name) of the single-rank entry points: the facade and the backend package.
SINGLE_RANK = (
    ("CakeCausalConv1dFunction", "CakeCausalConv1dDetFunction"),
    ("cake_causal_conv1d", "cake_causal_conv1d_det"),
    ("cake_causal_conv1d_train_forward", "cake_causal_conv1d_det_train_forward"),
    ("cake_causal_conv1d_train_backward", "cake_causal_conv1d_det_train_backward"),
    (
        "cake_causal_conv1d_train_forward_workspace",
        "cake_causal_conv1d_det_train_forward_workspace",
    ),
    (
        "cake_causal_conv1d_train_forward_workspace_bytes",
        "cake_causal_conv1d_det_train_forward_workspace_bytes",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace",
        "cake_causal_conv1d_det_train_backward_workspace",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace_bytes",
        "cake_causal_conv1d_det_train_backward_workspace_bytes",
    ),
    (
        "is_cake_causal_conv1d_train_supported",
        "is_cake_causal_conv1d_det_train_supported",
    ),
)
# The single-rank names the ``flashinfer`` top level exports.
TOP_LEVEL = (
    ("CakeCausalConv1dFunction", "CakeCausalConv1dDetFunction"),
    ("cake_causal_conv1d", "cake_causal_conv1d_det"),
    ("cake_causal_conv1d_train_forward", "cake_causal_conv1d_det_train_forward"),
    ("cake_causal_conv1d_train_backward", "cake_causal_conv1d_det_train_backward"),
    (
        "cake_causal_conv1d_train_forward_workspace_bytes",
        "cake_causal_conv1d_det_train_forward_workspace_bytes",
    ),
    (
        "cake_causal_conv1d_train_backward_workspace_bytes",
        "cake_causal_conv1d_det_train_backward_workspace_bytes",
    ),
)
# (removed name, det name) of the KCP entry points: the KCP facade and the KCP backend module.
KCP_FACADE_NAMES = (
    ("cake_causal_conv1d_kcp", "cake_causal_conv1d_det_kcp"),
    ("cake_causal_conv1d_kcp_multi", "cake_causal_conv1d_det_multi_kcp"),
    ("KcpRecomputeHandle", "RecomputeHandleDetKcp"),
)
KCP_BACKEND_NAMES = (
    *KCP_FACADE_NAMES,
    ("CakeCausalConv1dKcpFunction", "CakeCausalConv1dDetFunctionKcp"),
    ("CakeCausalConv1dKcpMultiFunction", "CakeCausalConv1dDetMultiFunctionKcp"),
    ("cake_causal_conv1d_kcp_forward", "cake_causal_conv1d_det_forward_kcp"),
    ("cake_causal_conv1d_kcp_backward", "cake_causal_conv1d_det_backward_kcp"),
    (
        "cake_causal_conv1d_kcp_multi_forward",
        "cake_causal_conv1d_det_multi_forward_kcp",
    ),
    (
        "cake_causal_conv1d_kcp_multi_backward",
        "cake_causal_conv1d_det_multi_backward_kcp",
    ),
)
NAMESPACES = {"facade": facade, "backend": backend}


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_removed_module_does_not_import(name: str) -> None:
    sys.modules.pop(name, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)
    leaf = name.removeprefix("flashinfer.")
    if "." not in leaf:
        assert not hasattr(flashinfer, leaf), leaf


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("pair", SINGLE_RANK, ids=lambda p: p[0])
def test_single_rank_names(namespace: str, pair: tuple[str, str]) -> None:
    module = NAMESPACES[namespace]
    old, new = pair
    assert not hasattr(module, old), (namespace, old)
    assert old not in module.__all__, (namespace, old)
    assert new in module.__all__ and callable(getattr(module, new)), (namespace, new)


@pytest.mark.parametrize("pair", TOP_LEVEL, ids=lambda p: p[0])
def test_top_level_names(pair: tuple[str, str]) -> None:
    old, new = pair
    assert not hasattr(flashinfer, old), old
    assert getattr(flashinfer, new) is getattr(facade, new), new


@pytest.mark.parametrize("pair", KCP_FACADE_NAMES, ids=lambda p: p[0])
def test_kcp_names_not_on_the_single_rank_facade(pair: tuple[str, str]) -> None:
    old, new = pair
    for module in (facade, backend, flashinfer):
        assert not hasattr(module, old), (module.__name__, old)
    assert new in kcp_facade.__all__, new


def test_det_entry_points_are_the_decorated_defs() -> None:
    # One ``flashinfer_api`` wrapper per entry point, named as the det def.
    for _, new in SINGLE_RANK[1:4]:
        assert getattr(facade, new).__name__ == new, new
    assert backend.CakeCausalConv1dDetFunction.__name__ == "CakeCausalConv1dDetFunction"


# The KCP names, checked in a fresh interpreter after the single-rank imports (which must not load the KCP host
# module), then after importing the KCP facade and backend module.
_KCP_CHECK = """
import sys
from pathlib import Path

{imports}
import {package} as package

kcp_files = {{(Path(package.__file__).parent / f).resolve() for f in ("kcp.py", "kcp_jit.py")}}
loaded = sorted(
    name
    for name, module in list(sys.modules.items())
    if getattr(module, "__file__", None) and Path(module.__file__).resolve() in kcp_files
)
assert not loaded, loaded
print("KCP_NOT_LOADED")
import {kcp_facade} as kcp_facade
from {package} import kcp as kcp_backend
for old, new in {facade_names!r}:
    assert not hasattr(kcp_facade, old), old
    assert new in kcp_facade.__all__ and hasattr(kcp_backend, new), new
assert kcp_facade.RecomputeHandleDetKcp is kcp_backend.RecomputeHandleDetKcp
for old, new in {backend_names!r}:
    assert not hasattr(kcp_backend, old), old
    assert old not in kcp_backend.__all__, old
    assert new in kcp_backend.__all__, new
print("KCP_DET_NAMES")
"""

IMPORTS = {
    "flashinfer": "import flashinfer",
    "facade": f"import {FACADE}",
    "package": f"import {PACKAGE}",
    "package_names": f"from {PACKAGE} import cake_causal_conv1d_det_train_forward, cake_jit",
}


@pytest.mark.parametrize("form", sorted(IMPORTS))
def test_imports_do_not_load_kcp_and_kcp_names_carry_det(form: str) -> None:
    code = _KCP_CHECK.format(
        imports=IMPORTS[form],
        package=PACKAGE,
        kcp_facade=KCP_FACADE,
        facade_names=KCP_FACADE_NAMES,
        backend_names=KCP_BACKEND_NAMES,
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, (form, r.stdout, r.stderr)
    assert "KCP_NOT_LOADED" in r.stdout and "KCP_DET_NAMES" in r.stdout, (
        form,
        r.stdout,
        r.stderr,
    )


def test_old_package_directory_is_gone() -> None:
    root = Path(flashinfer.__file__).parent
    assert not (root / "cake_causal_conv1d_train_backend").exists()
    assert not (root / "cake_causal_conv1d_train.py").exists()
