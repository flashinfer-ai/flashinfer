# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Concurrent builds and the argument-keyed decorator of the CuTe-DSL disk cache, without a device.

Different kernels of one module compile concurrently (only the commit takes the module lock), and every one is
committed; a kernel missed by several builders at once is compiled and written once, the others load it. The
decorator names a kernel after the compile function's arguments. Threads stand in for processes: each FileLock
opens its own descriptor, so they exclude each other as processes do. The compiles are stubs."""

import threading
import time
from pathlib import Path

import pytest

from flashinfer.jit import cute_dsl_core
from flashinfer.jit import env as jit_env

pytestmark = pytest.mark.cpu_only


class _Kernel:
    def __init__(self, tag, log):
        self.tag, self.log = tag, log

    def export_to_c(self, path, function_name):
        self.log.append(("export", self.tag))
        Path(path).write_bytes(f"{function_name}:{self.tag}".encode())


@pytest.fixture
def jit_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "cached_ops")
    monkeypatch.setattr(cute_dsl_core, "get_tmpdir", lambda: tmp_path)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(
        cute_dsl_core.JitSpecCuteDsl,
        "_load_from_disk",
        lambda self: self.object_path.read_bytes(),
    )
    return tmp_path / "cached_ops"


def _slow_compile(tag, log, seconds=0.5):
    def compile_fn():
        start = time.monotonic()
        time.sleep(seconds)
        log.append(("compile", tag, start, time.monotonic()))
        return _Kernel(tag, log)

    return compile_fn


def _build_concurrently(specs):
    results = [None] * len(specs)

    def run(i):
        results[i] = specs[i].build_and_load()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(specs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_kernels_of_one_module_compile_concurrently_and_all_commit(jit_dir):
    log = []
    specs = [
        cute_dsl_core.JitSpecCuteDsl("mod", tag, _slow_compile(tag, log), "sha")
        for tag in ("a", "b", "c")
    ]
    assert _build_concurrently(specs) == [b"mod_a:a", b"mod_b:b", b"mod_c:c"]
    compiles = [e for e in log if e[0] == "compile"]
    # every compile started before any other ended: none waited on the module lock
    assert max(e[2] for e in compiles) < min(e[3] for e in compiles)
    for tag in ("a", "b", "c"):
        spec = cute_dsl_core.JitSpecCuteDsl("mod", tag, None, "sha")
        assert spec.is_compiled and spec.try_load() == f"mod_{tag}:{tag}".encode()


def test_a_kernel_missed_at_once_is_compiled_and_written_once(jit_dir):
    log = []
    specs = [
        cute_dsl_core.JitSpecCuteDsl("mod", "k", _slow_compile("k", log), "sha")
        for _ in range(4)
    ]
    assert _build_concurrently(specs) == [b"mod_k:k"] * 4
    assert [e[0] for e in log] == ["compile", "export"]


def test_stale_module_is_replaced_on_commit(jit_dir):
    log = []
    old = cute_dsl_core.JitSpecCuteDsl("mod", "a", _slow_compile("old", log, 0), "v1")
    old.build_and_load()
    new = cute_dsl_core.JitSpecCuteDsl("mod", "b", _slow_compile("new", log, 0), "v2")
    assert new.build_and_load() == b"mod_b:new"
    assert not old.is_compiled and new.is_compiled
    assert not old.object_path.exists()


def _decorated(log, **decorator_kwargs):
    @cute_dsl_core.cute_dsl_disk_cached("mod", (__file__,), **decorator_kwargs)
    def compile_kernel(dtype_name, n, tactic=(1, 2)):
        log.append((dtype_name, n, tactic))
        return _Kernel(f"{dtype_name}-{n}-{tactic}", [])

    return compile_kernel


def test_decorated_compile_loads_from_disk_in_a_new_process(jit_dir):
    log = []
    assert _decorated(log)("bf16", 128) == _decorated(log)("bf16", n=128)
    assert log == [("bf16", 128, (1, 2))]  # the second "process" loaded the artifact


@pytest.mark.parametrize("args", [("fp16", 128), ("bf16", 256), ("bf16", 128, (1, 3))])
def test_decorated_compile_names_every_argument(jit_dir, args):
    log = []
    first = _decorated(log)("bf16", 128)
    assert _decorated(log)(*args) != first
    assert len(log) == 2


def test_decorated_compile_bypasses_the_cache_for_unrenderable_arguments(jit_dir):
    log = []
    kernel = _decorated(log)("bf16", object())
    assert isinstance(kernel, _Kernel)  # the in-process kernel, not a loaded artifact
    assert not jit_dir.exists() or not any(jit_dir.rglob("*.o"))


def test_kernel_name_renders_dataclasses_enums_and_types():
    import dataclasses
    import enum

    import torch

    class Mode(enum.Enum):
        A = 1
        B = 2

    @dataclasses.dataclass(frozen=True)
    class Tactic:
        tile: int
        mode: Mode

    key = (Tactic(16, Mode.A), torch.bfloat16, int, None, 1.5)
    name = cute_dsl_core.cute_dsl_kernel_name("k", key)
    assert name == cute_dsl_core.cute_dsl_kernel_name("k", key)
    assert name != cute_dsl_core.cute_dsl_kernel_name(
        "k", (Tactic(16, Mode.B), *key[1:])
    )
    assert name != cute_dsl_core.cute_dsl_kernel_name(
        "k", (key[0], torch.float16, *key[2:])
    )
    assert cute_dsl_core.sanitize_symbol_name(name) == name
