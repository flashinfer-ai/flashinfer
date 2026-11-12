# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The top-level lazy-export table of ``flashinfer`` (``flashinfer._LAZY_EXPORTS``)."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

_REPO_ROOT = Path(__file__).parents[3]


def _run_isolated(code: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO_ROOT), env.get("PYTHONPATH")))
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.cpu_only
def test_every_row_is_bound_on_first_access_only():
    _run_isolated(
        """
import importlib
import types
import flashinfer

rows = flashinfer._LAZY_EXPORTS
assert rows, "the table is empty"
# Check every row before the first access: importing a row's module binds that
# submodule on flashinfer, which also serves a module-valued row for it.
for name in rows:
    assert name not in vars(flashinfer), f"{name} is bound by import flashinfer"
    assert name in dir(flashinfer), name
for name, (module_name, attribute) in sorted(rows.items()):
    value = getattr(flashinfer, name)
    module = importlib.import_module(module_name, "flashinfer")
    if attribute is None:
        assert isinstance(value, types.ModuleType) and value is module, name
    else:
        assert value is getattr(module, attribute), name
    assert vars(flashinfer)[name] is value, f"{name} is not cached"
"""
    )


@pytest.mark.cpu_only
def test_row_values_are_module_paths_and_optional_attributes():
    import flashinfer

    for name, row in flashinfer._LAZY_EXPORTS.items():
        assert isinstance(name, str) and name.isidentifier(), name
        module_name, attribute = row
        assert isinstance(module_name, str) and module_name, name
        assert attribute is None or attribute.isidentifier(), name


@pytest.mark.cpu_only
def test_unknown_name_raises_attribute_error():
    import flashinfer

    with pytest.raises(AttributeError, match="no attribute 'no_such_lazy_export'"):
        flashinfer.no_such_lazy_export  # noqa: B018
    assert not hasattr(flashinfer, "no_such_lazy_export")


@pytest.mark.cpu_only
def test_module_valued_and_absolute_rows(monkeypatch):
    import flashinfer

    version = importlib.import_module("flashinfer.version")
    rows = dict(flashinfer._LAZY_EXPORTS)
    rows["_lazy_probe_module"] = (".version", None)
    rows["_lazy_probe_absolute"] = ("flashinfer.version", "__version__")
    monkeypatch.setattr(flashinfer, "_LAZY_EXPORTS", rows)
    try:
        assert flashinfer._lazy_probe_module is version
        assert isinstance(flashinfer._lazy_probe_module, types.ModuleType)
        assert flashinfer._lazy_probe_absolute == version.__version__
        assert {"_lazy_probe_module", "_lazy_probe_absolute"} <= set(dir(flashinfer))
    finally:
        for name in ("_lazy_probe_module", "_lazy_probe_absolute"):
            vars(flashinfer).pop(name, None)
