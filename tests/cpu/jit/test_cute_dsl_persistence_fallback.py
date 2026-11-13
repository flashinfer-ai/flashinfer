# Copyright (c) 2026 by FlashInfer team.
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

"""Cache persistence failures must preserve the newly compiled kernel."""

import errno
import json
from pathlib import Path

import pytest

from flashinfer.jit import cute_dsl_core as cache

pytestmark = pytest.mark.cpu_only


class _Kernel:
    fail_export = False

    def dump_to_object(self, symbol):
        if self.fail_export:
            raise OSError(errno.ENOSPC, "controlled export failure")
        return b"new-object"

    def export_to_c(self, path, function_name):
        Path(path).write_bytes(self.dump_to_object(function_name))


@pytest.fixture
def disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.setattr(cache, "get_tmpdir", lambda: tmp_path)
    monkeypatch.setattr(cache, "_get_cute_dsl_version", lambda: "cpu-test")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.delenv("FLASHINFER_JIT_BUILD_LOG", raising=False)

    def load_disk(spec):
        payload = spec.object_path.read_bytes()
        if payload != b"new-object":
            raise RuntimeError("unloadable cached object")
        return payload

    monkeypatch.setattr(cache.JitSpecCuteDsl, "_load_from_disk", load_disk)


@pytest.mark.parametrize("enable_tvm_ffi", [True, False])
@pytest.mark.parametrize("existing_object", [False, True])
@pytest.mark.parametrize("entrypoint", ["build_and_load", "compile_and_persist"])
def test_export_failure_returns_new_kernel(
    disk_cache, enable_tvm_ffi, existing_object, entrypoint
):
    kernel = _Kernel()
    kernel.fail_export = True
    spec = cache.JitSpecCuteDsl(
        "fallback", "kernel", lambda: kernel, "source", enable_tvm_ffi=enable_tvm_ffi
    )
    if existing_object:
        spec.module_dir.mkdir()
        spec.object_path.write_bytes(b"old-corrupt-object")
        spec.meta_path.write_text(json.dumps(spec.expected_meta))

    if entrypoint == "build_and_load":
        result = spec.build_and_load()
    else:
        spec.compile_and_persist()
        result = spec.load()
    assert result is kernel


@pytest.mark.parametrize("enable_tvm_ffi", [True, False])
def test_successful_export_still_validates_new_artifact(
    disk_cache, monkeypatch, enable_tvm_ffi
):
    spec = cache.JitSpecCuteDsl(
        "validation", "kernel", _Kernel, "source", enable_tvm_ffi=enable_tvm_ffi
    )

    def reject_artifact(spec):
        assert spec.object_path.read_bytes() == b"new-object"
        raise RuntimeError("new artifact is incompatible")

    monkeypatch.setattr(cache.JitSpecCuteDsl, "_load_from_disk", reject_artifact)
    with pytest.raises(RuntimeError, match="new artifact is incompatible"):
        spec.build_and_load()


@pytest.mark.parametrize("enable_tvm_ffi", [True, False])
def test_failed_reexport_does_not_reuse_previous_artifact(disk_cache, enable_tvm_ffi):
    kernel = _Kernel()
    spec = cache.JitSpecCuteDsl(
        "reexport", "kernel", lambda: kernel, "source", enable_tvm_ffi=enable_tvm_ffi
    )
    assert spec.build_and_load() == b"new-object"
    # Recompile explicitly: a previous successful export must not mask this failure.
    kernel.fail_export = True
    spec.compile_and_persist()
    assert spec.load() is kernel
    kernel.fail_export = False
    spec.compile_and_persist()
    assert spec.load() == b"new-object"
