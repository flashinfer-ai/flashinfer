# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL cache lifecycle and real file-lock concurrency without a GPU."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest
from filelock import FileLock, Timeout

from flashinfer.jit import cute_dsl_core as cache
from flashinfer.jit.core import MissingJITCacheError

pytestmark = pytest.mark.cpu_only


class _Kernel:
    def export_to_c(self, path, function_name):
        Path(path).write_bytes(b"kernel")

    def dump_to_object(self, symbol):
        return b"kernel"


@pytest.fixture
def disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "ops")
    monkeypatch.setattr(cache, "get_tmpdir", lambda: tmp_path)
    monkeypatch.setattr(cache, "_get_cute_dsl_version", lambda: "test")
    monkeypatch.setattr(
        cache.JitSpecCuteDsl,
        "_load_from_disk",
        lambda spec: spec.object_path.read_bytes(),
    )
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_80")
    for name in (
        "FLASHINFER_DISABLE_JIT",
        "FLASHINFER_CUTE_DSL_DISABLE_CACHE",
        "FLASHINFER_JIT_BUILD_LOG",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("enable_tvm_ffi", [True, False])
def test_build_load_cache_hits_and_invalidation(
    disk_cache, monkeypatch, enable_tvm_ffi
):
    builds = []

    def compile_kernel():
        builds.append(1)
        return _Kernel()

    def load():
        return cache.build_and_load_cute_dsl_kernel(
            "test", "variant", compile_kernel, enable_tvm_ffi=enable_tvm_ffi
        )

    assert load() == b"kernel"
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    assert load() == b"kernel"
    assert len(builds) == 1
    monkeypatch.setattr(cache, "_get_cute_dsl_version", lambda: "new version")
    with pytest.raises(MissingJITCacheError):
        load()
    assert len(builds) == 1
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT")
    assert load() == b"kernel"
    assert len(builds) == 2


def test_different_kernels_compile_concurrently(disk_cache):
    compiling = Barrier(2, timeout=5)

    def compile_kernel():
        compiling.wait()
        return _Kernel()

    specs = [
        cache.JitSpecCuteDsl("shared", name, compile_kernel, "source")
        for name in ("a", "b")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(spec.build_and_load) for spec in specs]
        assert [f.result(timeout=10) for f in futures] == [b"kernel", b"kernel"]
    assert all(spec.is_compiled for spec in specs)


def test_same_kernel_lock_is_held_until_publication(disk_cache, monkeypatch):
    exporting, release = Event(), Event()
    builds = []

    def compile_kernel():
        builds.append(1)
        return _Kernel()

    def spec():
        return cache.JitSpecCuteDsl("shared", "same", compile_kernel, "source")

    first_spec = spec()
    export = first_spec._export_object

    def blocked_export(path):
        exporting.set()
        assert release.wait(timeout=5)
        export(path)

    monkeypatch.setattr(first_spec, "_export_object", blocked_export)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_spec.build_and_load)
        try:
            assert exporting.wait(timeout=5)
            with pytest.raises(Timeout), FileLock(spec().kernel_lock_path, timeout=0):
                pytest.fail("the publishing worker must still hold the kernel lock")
            second = pool.submit(spec().build_and_load)
        finally:
            release.set()
        assert first.result(timeout=10) == second.result(timeout=10) == b"kernel"
    assert len(builds) == 1


def test_bad_artifact_recovers_and_export_failure_keeps_memory_kernel(
    disk_cache, monkeypatch
):
    builds = []

    def compile_kernel():
        builds.append(1)
        return _Kernel()

    spec = cache.JitSpecCuteDsl("test", "variant", compile_kernel, "source")
    assert spec.build_and_load() == b"kernel"
    spec.object_path.write_bytes(b"corrupt")

    def load_checked():
        value = spec.object_path.read_bytes()
        if value != b"kernel":
            raise RuntimeError("corrupt cached object")
        return value

    monkeypatch.setattr(spec, "_load_from_disk", load_checked)
    assert spec.build_and_load() == b"kernel" and len(builds) == 2
    # Existing callers can explicitly rebuild after a failed cache load.
    spec.compile_and_persist()
    assert len(builds) == 3

    fresh = cache.JitSpecCuteDsl("test", "no_disk", compile_kernel, "source")

    def fail_export(path):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(fresh, "_export_object", fail_export)
        assert isinstance(fresh.build_and_load(), _Kernel)
    assert not fresh.is_compiled
    with FileLock(fresh.kernel_lock_path, timeout=0):
        pass
    assert fresh.build_and_load() == b"kernel"


def test_disabled_cache_compiles_in_memory(disk_cache, monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    assert isinstance(
        cache.build_and_load_cute_dsl_kernel("test", "variant", _Kernel), _Kernel
    )
