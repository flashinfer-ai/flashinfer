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

Build-locking contract for the CuTe-DSL disk cache.

Which lock a build holds decides whether concurrent processes can fill the cache
in parallel: a lock spanning the compile makes workers building different
specializations of one op family queue behind each other, which measured 1.25x
against 5.3x on an 8-way sharded GDN decode run. So the build lock is
per-kernel, and only the state siblings genuinely share -- the stale-module
wipe and the export -- is serialized module-wide.

No GPU: ``CUTE_DSL_ARCH`` supplies the target and a stub stands in for the
``cute.compile`` result.
"""

import fcntl
import json
from pathlib import Path

import pytest

from flashinfer.jit import env as jit_env
from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl

MODULE = "build_lock_probe"


class _StubCompiled:
    """A ``cute.compile`` result stand-in; only ``export_to_c`` is exercised."""

    def __init__(self, marker: str):
        self.marker = marker

    def export_to_c(self, path: str, function_name: str) -> None:
        Path(path).write_text(f"{function_name}:{self.marker}")


def _module_lock_is_free(path: Path) -> bool:
    """Whether the module lock can be taken right now.

    ``flock`` grants per open file description, not per process, so this answers
    honestly even from inside the build that holds it.
    """
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
        return True


@pytest.fixture
def jit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm100a")
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.setattr(
        JitSpecCuteDsl, "_load_from_disk", lambda self: self.object_path.read_text()
    )
    return tmp_path


def _spec(kernel_name, compile_fn=None, source_sha256="sha256"):
    return JitSpecCuteDsl(
        MODULE,
        kernel_name,
        compile_fn or (lambda: _StubCompiled(kernel_name)),
        source_sha256,
    )


def test_specializations_lock_apart_but_share_the_module_lock(jit_dir):
    first, second = _spec("kernel_a"), _spec("kernel_b")

    assert first.lock_path != second.lock_path
    assert first.module_lock_path == second.module_lock_path


def test_lock_filename_fits_a_path_component(jit_dir):
    """Kernel names run to ~210 characters; a filename gets 255 bytes."""
    spec = _spec("k" * 210)

    assert len(spec.lock_path.name) < 255


def test_build_leaves_the_module_lock_free_while_compiling(jit_dir):
    """The property the parallel speedup rests on: a sibling specialization can
    start while this one is still inside ``cute.compile``."""
    free_during_compile = []

    def compile_fn():
        free_during_compile.append(_module_lock_is_free(spec.module_lock_path))
        return _StubCompiled("kernel_a")

    spec = _spec("kernel_a", compile_fn)
    spec.build()

    assert free_during_compile == [True], (
        "the module lock spans cute.compile, so specializations of one op "
        "family cannot be built concurrently"
    )


def test_export_runs_under_the_module_lock(jit_dir, monkeypatch):
    """The export writes into a directory siblings share, and a stale-module
    wipe can delete what it just wrote, so it must not interleave with one."""
    free_during_export = []
    original_export = JitSpecCuteDsl._export

    def spying_export(self):
        free_during_export.append(_module_lock_is_free(self.module_lock_path))
        original_export(self)

    monkeypatch.setattr(JitSpecCuteDsl, "_export", spying_export)
    _spec("kernel_a").build()

    assert free_during_export == [False]


def test_one_specialization_is_compiled_once(jit_dir):
    """Narrowing the lock must not cost the dedup: a second call for the same
    specialization loads the artifact instead of recompiling."""
    compiles = []

    def compile_fn():
        compiles.append(1)
        return _StubCompiled("kernel_a")

    expected = f"{MODULE}_kernel_a:kernel_a"
    assert _spec("kernel_a", compile_fn).build_and_load() == expected
    assert _spec("kernel_a", compile_fn).build_and_load() == expected
    assert len(compiles) == 1


def test_stale_module_is_wiped_before_the_new_kernel_lands(jit_dir):
    """A source change invalidates the whole module directory, as before."""
    _spec("kernel_a", source_sha256="old").build()
    stale_artifact = jit_dir / f"{MODULE}_sm100a_cute_dsl" / "kernel_a.o"
    assert stale_artifact.exists()

    _spec("kernel_b", source_sha256="new").build()

    assert not stale_artifact.exists()
    meta = json.loads((stale_artifact.parent / "meta.json").read_text())
    assert meta["source_sha256"] == "new"
