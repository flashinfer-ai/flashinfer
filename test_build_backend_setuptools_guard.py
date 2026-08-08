"""Standalone test for build_backend.py's setuptools-version guard.

Not under tests/: that suite's conftest.py imports torch and the fully built
flashinfer package, which this pure PEP 517 build-hook check has no need of.
Run directly: pytest test_build_backend_setuptools_guard.py
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
build_backend = importlib.import_module("build_backend")


def _set_setuptools_version(monkeypatch, version):
    # _check_setuptools_version() does `import setuptools` locally, so patching
    # sys.modules is what that local import will actually see.
    fake = types.SimpleNamespace(__version__=version)
    monkeypatch.setitem(sys.modules, "setuptools", fake)


def test_old_setuptools_raises_clear_error(monkeypatch):
    _set_setuptools_version(monkeypatch, "65.5.0")
    with pytest.raises(RuntimeError, match="setuptools>=77"):
        build_backend._check_setuptools_version()


def test_pinned_setuptools_passes(monkeypatch):
    _set_setuptools_version(monkeypatch, "77.0.0")
    build_backend._check_setuptools_version()  # must not raise


def test_newer_setuptools_passes(monkeypatch):
    _set_setuptools_version(monkeypatch, "79.0.1")
    build_backend._check_setuptools_version()  # must not raise
