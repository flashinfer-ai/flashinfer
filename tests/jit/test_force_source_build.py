"""FLASHINFER_FORCE_JIT: build from source even when a prebuilt artifact exists.

No GPU or compiler needed — these cover the artifact-selection predicate, not
a build.
"""

from __future__ import annotations

import pathlib

import pytest

from flashinfer.jit import core
from flashinfer.jit import env as jit_env


def _spec(name: str = "force_jit_dummy") -> core.JitSpecNvcc:
    return core.JitSpecNvcc(
        name=name,
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )


@pytest.fixture
def prebuilt(monkeypatch, tmp_path: pathlib.Path) -> pathlib.Path:
    """A module whose prebuilt .so exists, as an installed jit-cache wheel."""
    artifact = tmp_path / "force_jit_dummy.so"
    artifact.touch()
    monkeypatch.setattr(jit_env, "get_aot_path", lambda name: artifact)
    monkeypatch.delenv("FLASHINFER_FORCE_JIT", raising=False)
    return artifact


def test_prebuilt_artifact_wins_by_default(prebuilt: pathlib.Path) -> None:
    spec = _spec()
    assert spec.is_aot
    assert spec.get_library_path() == prebuilt


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_force_jit_ignores_prebuilt_artifact(
    monkeypatch, prebuilt: pathlib.Path, value: str
) -> None:
    """The prebuilt artifact is bypassed, so build_and_load() reaches build()."""
    monkeypatch.setenv("FLASHINFER_FORCE_JIT", value)
    spec = _spec()
    assert not spec.is_aot
    assert spec.get_library_path() == spec.jit_library_path
    # try_load() only ever short-circuits on a prebuilt artifact; returning
    # None is what routes build_and_load() through a real ninja build.
    assert spec.try_load() is None


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_force_jit_off_values_keep_prebuilt(
    monkeypatch, prebuilt: pathlib.Path, value: str
) -> None:
    """A set-but-negative value must not silently turn the flag on."""
    monkeypatch.setenv("FLASHINFER_FORCE_JIT", value)
    assert not core.force_source_build()
    assert _spec().is_aot


def test_force_jit_without_prebuilt_is_a_no_op(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(jit_env, "get_aot_path", lambda name: tmp_path / "absent.so")
    monkeypatch.setenv("FLASHINFER_FORCE_JIT", "1")
    spec = _spec()
    assert not spec.is_aot
    assert spec.get_library_path() == spec.jit_library_path
