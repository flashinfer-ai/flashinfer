"""CPU-only checks for the Rubin vendored-package import guard."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


_PATHS_FILE = (
    Path(__file__).resolve().parents[2]
    / "flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/shim/_paths.py"
)


def _load_paths():
    spec = importlib.util.spec_from_file_location("sm107_paths_under_test", _PATHS_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reject_preimported_namespace_sources(monkeypatch, tmp_path):
    (tmp_path / "sources").mkdir()
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.delitem(sys.modules, "sources", raising=False)
    namespace = importlib.import_module("sources")
    assert namespace.__file__ is None

    with pytest.raises(RuntimeError, match="already imported without a file"):
        _load_paths().bootstrap_paths()


def test_reject_preimported_regular_external_sources(monkeypatch, tmp_path):
    external = ModuleType("sources")
    external.__file__ = str(tmp_path / "sources" / "__init__.py")
    monkeypatch.setitem(sys.modules, "sources", external)

    with pytest.raises(RuntimeError, match="already imported from"):
        _load_paths().bootstrap_paths()


def test_allow_vendored_sources_and_idempotent_bootstrap(monkeypatch):
    paths = _load_paths()
    src_dir = os.path.join(os.path.dirname(os.path.dirname(_PATHS_FILE)), "src")
    vendored = ModuleType("sources")
    vendored.__file__ = os.path.join(src_dir, "sources", "__init__.py")
    monkeypatch.setitem(sys.modules, "sources", vendored)
    monkeypatch.setattr(sys, "path", list(sys.path))

    paths.bootstrap_paths()
    assert sys.path[0] == src_dir
    paths.bootstrap_paths()
    assert sys.path.count(src_dir) == 1
