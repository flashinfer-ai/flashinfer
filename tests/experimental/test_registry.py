# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Registry contract: sm12x._OPS and the on-disk op directories stay in
lockstep, and every op honors the META/__all__ shape (invariant #3)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
SM12X_DIR = REPO / "flashinfer" / "experimental" / "sm12x"


def _on_disk_ops() -> list[str]:
    ops = []
    for group_dir in sorted(SM12X_DIR.iterdir()):
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        for op_dir in sorted(group_dir.iterdir()):
            if not op_dir.is_dir() or op_dir.name.startswith("_"):
                continue
            ops.append(f"{group_dir.name}.{op_dir.name}")
    return ops


def _sm12x():
    return importlib.import_module("flashinfer.experimental.sm12x")


def test_registry_matches_disk():
    sm12x = _sm12x()
    assert sorted(sm12x._OPS) == _on_disk_ops(), (
        "sm12x._OPS and the op directories under flashinfer/experimental/sm12x/ "
        "must be in bijection; register new ops in _OPS"
    )


def test_list_ops_and_find_op():
    sm12x = _sm12x()
    metas = sm12x.list_ops()
    assert len(metas) == len(sm12x._OPS)
    for meta in metas:
        assert sm12x.find_op(meta.qualname) is meta
    with pytest.raises(KeyError):
        sm12x.find_op("no_such.op")


def test_every_op_meta_contract():
    sm12x = _sm12x()
    for meta in sm12x.list_ops():
        module = importlib.import_module(
            f"flashinfer.experimental.sm12x.{meta.qualname}"
        )
        assert isinstance(module.META, sm12x.OpMeta)
        assert set(module.__all__) == set(meta.entry_points) | {"META"}, meta.qualname
        assert "is_supported" in meta.entry_points, meta.qualname
        assert meta.archs and set(meta.archs) <= {"sm120a", "sm121a"}, meta.qualname
        assert meta.provenance.commit, f"{meta.qualname} missing provenance commit"
        assert meta.test_path and (REPO / meta.test_path).is_file(), (
            f"{meta.qualname} META.test_path {meta.test_path!r} does not exist"
        )


def test_clear_all_caches_never_forces_imports():
    sm12x = _sm12x()
    sm12x.clear_all_caches()  # must be a no-op / safe with nothing imported


def test_every_op_api_resolves():
    """Force-load every op's api and resolve every declared entry point.

    Catches facade/alias typos without a GPU; needs cutlass (kernel modules
    import it), so the CPU-only CI job skips this one.
    """
    pytest.importorskip("cutlass")
    sm12x = _sm12x()
    for meta in sm12x.list_ops():
        module = importlib.import_module(
            f"flashinfer.experimental.sm12x.{meta.qualname}"
        )
        for name in meta.entry_points:
            assert getattr(module, name) is not None, f"{meta.qualname}.{name}"
