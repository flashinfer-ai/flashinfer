# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``JitSpecCuteDsl`` with an export-only compile closure, without a device: the disk-cache path compiles the kernel
without the DSL's execution engine and hands back the exported object; a failed export compiles again with the engine;
``compile_and_persist`` exports the engine-less kernel; a disabled cache or a spec without the closure keep today's path;
a key hash the caller passes is used as given. The compiles are stubs: what is tested is which closure runs and what
``load()`` returns."""

import json
from pathlib import Path

import pytest

from flashinfer.jit import cute_dsl_core
from flashinfer.jit import env as jit_env

pytestmark = pytest.mark.cpu_only


class _Kernel:
    """A compiled kernel as the spec sees it: exportable (or not), tagged by the closure that made it."""

    def __init__(self, tag, fail_export=False):
        self.tag, self.fail_export = tag, fail_export

    def export_to_c(self, path, function_name):
        if self.fail_export:
            raise OSError("export refused")
        Path(path).write_bytes(f"{function_name}:{self.tag}".encode())


class _Closures:
    def __init__(self, fail_export=False):
        self.calls = []
        self.fail_export = fail_export

    def engine(self):
        self.calls.append("engine")
        return _Kernel("engine")

    def no_engine(self):
        self.calls.append("no_engine")
        return _Kernel("no_engine", self.fail_export)


@pytest.fixture
def jit_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "cached_ops")
    monkeypatch.setattr(cute_dsl_core, "get_tmpdir", lambda: tmp_path)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    # loading the exported object = reading it back (no DSL runtime here)
    monkeypatch.setattr(
        cute_dsl_core.JitSpecCuteDsl,
        "_load_from_disk",
        lambda self: self.object_path.read_bytes(),
    )
    return tmp_path / "cached_ops"


def _spec(closures, export=True):
    return cute_dsl_core.JitSpecCuteDsl(
        "mod",
        "k",
        closures.engine,
        "sha",
        export_compile_fn=closures.no_engine if export else None,
    )


def test_cache_path_compiles_without_the_engine_and_loads_the_object(jit_dir):
    c = _Closures()
    spec = _spec(c)
    assert spec.build_and_load() == b"mod_k:no_engine"
    assert c.calls == ["no_engine"]  # the engine closure never ran
    assert spec._compiled_kernel is None and spec.is_compiled
    # a hit loads the object: no compile at all
    assert _spec(_Closures()).build_and_load() == b"mod_k:no_engine"


def test_failed_export_compiles_again_with_the_engine(jit_dir):
    c = _Closures(fail_export=True)
    spec = _spec(c)
    kernel = spec.build_and_load()
    assert c.calls == ["no_engine", "engine"]
    assert isinstance(kernel, _Kernel) and kernel.tag == "engine"
    assert not spec.object_path.exists()


def test_spec_without_the_closure_keeps_the_engine(jit_dir):
    c = _Closures()
    spec = _spec(c, export=False)
    assert spec.build_and_load() == b"mod_k:engine"
    assert c.calls == ["engine"]


def test_compile_and_persist_exports_the_engine_less_kernel(jit_dir, monkeypatch):
    c = _Closures()
    spec = _spec(c)
    spec.compile_and_persist()
    assert c.calls == ["no_engine"] and spec._compiled_kernel is None
    assert spec.is_compiled and spec.object_path.read_bytes() == b"mod_k:no_engine"
    assert json.loads(spec.meta_path.read_text())["source_sha256"] == "sha"

    sibling = cute_dsl_core.JitSpecCuteDsl(
        "mod", "other", _Closures().engine, "new source"
    )
    sibling.compile_and_persist()
    assert not spec.is_compiled
    with monkeypatch.context() as disabled:
        disabled.setenv("FLASHINFER_DISABLE_JIT", "1")
        with pytest.raises(cute_dsl_core.MissingJITCacheError):
            spec.load()
    kernel = spec.load()
    assert kernel.tag == "engine" and spec.load() is kernel
    assert c.calls == ["no_engine", "engine"]


def test_compile_and_persist_failed_export_compiles_again_with_the_engine(jit_dir):
    c = _Closures(fail_export=True)
    spec = _spec(c)
    spec.compile_and_persist()
    assert c.calls == ["no_engine", "engine"] and not spec.object_path.exists()
    kernel = spec.load()
    assert isinstance(kernel, _Kernel) and kernel.tag == "engine"


def test_build_and_load_cute_dsl_kernel_passes_the_closure(jit_dir, monkeypatch):
    monkeypatch.setattr(cute_dsl_core, "_hash_source_files", lambda paths: "sha")
    c = _Closures()
    kernel = cute_dsl_core.build_and_load_cute_dsl_kernel(
        "mod", "k", c.engine, export_compile_fn=c.no_engine
    )
    assert kernel == b"mod_k:no_engine" and c.calls == ["no_engine"]


def test_disabled_cache_runs_the_engine_compile(jit_dir, monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    c = _Closures()
    kernel = cute_dsl_core.build_and_load_cute_dsl_kernel(
        "mod", "k", c.engine, export_compile_fn=c.no_engine
    )
    assert isinstance(kernel, _Kernel) and kernel.tag == "engine"
    assert c.calls == ["engine"] and not jit_dir.exists()


def test_given_source_hash_is_used_without_hashing(jit_dir, monkeypatch):
    def refuse(paths):
        raise AssertionError("the key files were hashed although the hash was given")

    monkeypatch.setattr(cute_dsl_core, "_hash_source_files", refuse)
    c = _Closures()
    kernel = cute_dsl_core.build_and_load_cute_dsl_kernel(
        "mod", "k", c.engine, extra_key_files=["/nowhere.py"], source_sha256="given"
    )
    assert kernel == b"mod_k:engine"
    meta = json.loads((jit_dir / "mod_sm100a_cute_dsl" / "meta.json").read_text())
    assert meta["source_sha256"] == "given"
