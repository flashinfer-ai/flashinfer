# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

import pytest

from flashinfer.attn_scores import prefill_indexer as pi


@pytest.fixture
def compiler_lookup(monkeypatch):
    pi._resolve_nvcc.cache_clear()
    original = shutil.which
    calls = []

    def which(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(pi, "_probe_nvcc", lambda path: path)
    yield calls
    pi._resolve_nvcc.cache_clear()


def executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


def test_absolute_compiler_warm_lookup_and_override(
    tmp_path, monkeypatch, compiler_lookup
):
    first = executable(tmp_path / "first" / "nvcc")
    second = executable(tmp_path / "second" / "nvcc")
    monkeypatch.setattr(
        os, "getcwd", lambda: pytest.fail("absolute lookup must not need CWD")
    )
    monkeypatch.setenv("FLASHINFER_NVCC", first)
    assert pi._jit_nvcc_release() == first
    assert pi._jit_nvcc_release() == first
    monkeypatch.setenv("PATH", "/changed")
    assert pi._jit_nvcc_release() == first
    assert len(compiler_lookup) == 1
    monkeypatch.setenv("FLASHINFER_NVCC", second)
    assert pi._jit_nvcc_release() == second
    assert len(compiler_lookup) == 2


def test_bare_compiler_tracks_path_and_relative_path_cwd(
    tmp_path, monkeypatch, compiler_lookup
):
    first = executable(tmp_path / "one" / "bin" / "nvcc")
    second = executable(tmp_path / "two" / "bin" / "nvcc")
    monkeypatch.setenv("FLASHINFER_NVCC", "nvcc")
    monkeypatch.setenv("PATH", str(tmp_path / "one" / "bin"))
    assert pi._jit_nvcc_release() == first
    assert pi._jit_nvcc_release() == first
    monkeypatch.setenv("PATH", str(tmp_path / "two" / "bin"))
    assert pi._jit_nvcc_release() == second
    monkeypatch.setenv("PATH", "bin")
    monkeypatch.chdir(tmp_path / "one")
    assert pi._jit_nvcc_release() == first
    monkeypatch.chdir(tmp_path / "two")
    assert pi._jit_nvcc_release() == second
    # An empty PATH component means CWD.
    monkeypatch.setenv("PATH", ":/absent")
    monkeypatch.chdir(tmp_path / "one" / "bin")
    assert pi._jit_nvcc_release() == first
    monkeypatch.setenv("PATH", "")
    assert pi._jit_nvcc_release() == ""


def test_relative_compiler_tracks_cwd(tmp_path, monkeypatch, compiler_lookup):
    first = executable(tmp_path / "one" / "bin" / "nvcc")
    second = executable(tmp_path / "two" / "bin" / "nvcc")
    monkeypatch.setenv("FLASHINFER_NVCC", "bin/nvcc")
    monkeypatch.chdir(tmp_path / "one")
    assert pi._jit_nvcc_release() == first
    assert pi._jit_nvcc_release() == first
    monkeypatch.chdir(tmp_path / "two")
    assert pi._jit_nvcc_release() == second


def test_missing_compiler_recovers_without_config_change(
    tmp_path, monkeypatch, compiler_lookup
):
    path = tmp_path / "nvcc"
    monkeypatch.setenv("FLASHINFER_NVCC", str(path))
    assert pi._jit_nvcc_release() == ""
    executable(path)
    assert pi._jit_nvcc_release() == str(path)
    assert pi._jit_nvcc_release() == str(path)
    assert len(compiler_lookup) == 2


def test_bare_compiler_preserves_search_precedence(
    tmp_path, monkeypatch, compiler_lookup
):
    first = tmp_path / "one" / "nvcc"
    second = executable(tmp_path / "two" / "nvcc")
    monkeypatch.setenv("FLASHINFER_NVCC", "nvcc")
    monkeypatch.setenv("PATH", f"{first.parent}:{tmp_path / 'two'}")
    monkeypatch.setattr(
        os, "getcwd", lambda: pytest.fail("absolute PATH must not need CWD")
    )
    assert pi._jit_nvcc_release() == second
    executable(first)
    assert pi._jit_nvcc_release() == str(first)
    first.unlink()
    assert pi._jit_nvcc_release() == second
