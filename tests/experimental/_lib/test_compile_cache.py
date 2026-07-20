# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Compile-cache re-rooting: the highest-risk mechanical edit of the port.

Wrong fingerprint root ⇒ silent stale disk-cache hits (running old kernels),
so these assertions are load-bearing.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

compiler = importlib.import_module("flashinfer.experimental.sm12x._lib.compiler")
env = importlib.import_module("flashinfer.experimental.sm12x._lib.env")


def test_package_root_is_the_sm12x_subtree():
    root = compiler._PACKAGE_ROOT
    assert root.parts[-3:] == ("flashinfer", "experimental", "sm12x"), root
    assert (root / "_lib" / "compiler.py").is_file()


def test_fingerprint_tracks_source_edits(tmp_path, monkeypatch):
    (tmp_path / "kernel.py").write_text("x = 1\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    monkeypatch.setattr(compiler, "_PACKAGE_ROOT", tmp_path)

    before = compiler._compute_sm12x_package_fingerprint()

    (pycache / "kernel.cpython-312.pyc").write_bytes(b"ignored")
    assert compiler._compute_sm12x_package_fingerprint() == before, (
        "__pycache__ must not affect the fingerprint"
    )

    (tmp_path / "kernel.py").write_text("x = 2\n")
    after = compiler._compute_sm12x_package_fingerprint()
    assert after != before, "editing any source must change the fingerprint"


def test_cache_dir_resolution_order(monkeypatch):
    for name in (
        "FLASHINFER_EXP_SM12X_COMPILE_CACHE_DIR",
        "FLASHINFER_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)

    assert compiler._cute_compile_cache_dir() == (
        Path.home() / ".cache" / "flashinfer" / "experimental" / "sm12x" / "compile"
    )

    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")
    assert compiler._cute_compile_cache_dir() == Path(
        "/xdg/flashinfer/experimental/sm12x/compile"
    )

    monkeypatch.setenv("FLASHINFER_CACHE_DIR", "/fi-cache")
    assert compiler._cute_compile_cache_dir() == Path(
        "/fi-cache/experimental/sm12x/compile"
    )

    monkeypatch.setenv("FLASHINFER_EXP_SM12X_COMPILE_CACHE_DIR", "/explicit")
    assert compiler._cute_compile_cache_dir() == Path("/explicit")


def test_legacy_env_read_through(monkeypatch):
    monkeypatch.setattr(env, "_synced", False)
    created = []
    try:
        monkeypatch.setenv("B12X_MLA_FORCE_SPLIT", "3")
        monkeypatch.setenv("B12X_CUTE_COMPILE_DISK_CACHE", "0")
        monkeypatch.setenv("B12X_VLLM_ENGINE_STARTED", "1")
        monkeypatch.setenv("B12X_DENSE_ATOM_24", "legacy")
        monkeypatch.setenv("FLASHINFER_EXP_SM12X_DENSE_ATOM_24", "explicit")
        for name in (
            "FLASHINFER_EXP_SM12X_MLA_FORCE_SPLIT",
            "FLASHINFER_EXP_SM12X_COMPILE_DISK_CACHE",
            "FLASHINFER_EXP_SM12X_ENGINE_STARTED",
        ):
            monkeypatch.delenv(name, raising=False)
            created.append(name)

        import pytest

        with pytest.warns(DeprecationWarning, match="legacy b12x environment"):
            env.sync_legacy_env()

        assert os.environ["FLASHINFER_EXP_SM12X_MLA_FORCE_SPLIT"] == "3"
        assert os.environ["FLASHINFER_EXP_SM12X_COMPILE_DISK_CACHE"] == "0"
        assert os.environ["FLASHINFER_EXP_SM12X_ENGINE_STARTED"] == "1"
        # explicitly-set new-style names always win over legacy values
        assert os.environ["FLASHINFER_EXP_SM12X_DENSE_ATOM_24"] == "explicit"

        assert env.env_raw("MLA_FORCE_SPLIT") == "3"
        assert env.env_flag("COMPILE_DISK_CACHE", default=True) is False
    finally:
        for name in created:
            os.environ.pop(name, None)
        monkeypatch.setattr(env, "_synced", False)
