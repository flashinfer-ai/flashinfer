# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from types import SimpleNamespace
import subprocess
import sys

from filelock import FileLock
import pytest
import torch

from flashinfer.attention.prims_ts import _compile_cache as cc
from flashinfer.attention.prims_ts import mla_decode

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("as_tuple", [False, True])
def test_precompile_modes_bypass_loaded_memo(as_tuple):
    calls = []

    @cc.cache_compiled
    def compile_kernel(key):
        calls.append(cc._PRECOMPILE_MODE.get())
        result = cc._unloaded_kernel if calls[-1] else object()
        return (result, None) if as_tuple else result

    with cc.probe_cached():
        compile_kernel("kernel")
    with cc.precompile_only():
        compile_kernel("kernel")
    real = compile_kernel("kernel")
    assert compile_kernel("kernel") is real
    with cc.probe_cached():
        compile_kernel("kernel")
    assert compile_kernel("kernel") is real
    assert calls == ["probe", "persist", None, "probe"]
    compile_kernel.cache_clear()
    assert compile_kernel("kernel") is not real


def test_precompile_context_restores_after_nested_failure():
    with cc.precompile_only(skip_busy=True):
        with pytest.raises(ValueError), cc.probe_cached():
            assert cc._PRECOMPILE_MODE.get() == "probe"
            raise ValueError("plan failed")
        assert cc._PRECOMPILE_MODE.get() == "persist"
        assert cc._SKIP_BUSY.get()
    assert cc._PRECOMPILE_MODE.get() is None
    assert not cc._SKIP_BUSY.get()


def test_precompile_lock_probe_and_failed_export(monkeypatch, tmp_path):
    from flashinfer.jit import core

    calls = []
    spec = SimpleNamespace(
        module_dir_name="prims_ts_test",
        kernel_name="test",
        name="test",
        is_compiled=False,
    )

    def persist():
        calls.append("persist")
        spec.is_compiled = True

    spec.compile_and_persist = persist
    monkeypatch.setattr(cc, "_make_spec", lambda *a: spec)
    monkeypatch.setattr(core, "get_tmpdir", lambda: tmp_path)
    args = ("family", "kernel", lambda: None, ())
    with pytest.raises(cc.KernelNotCached):
        cc._probe_kernel(*args)
    with (
        FileLock(tmp_path / "prims_ts_test.test.lock"),
        cc.precompile_only(skip_busy=True),
        pytest.raises(cc.KernelBusy),
    ):
        cc._precompile_kernel(*args)
    cc._precompile_kernel(*args)
    cc._precompile_kernel(*args)
    cc._probe_kernel(*args)
    assert calls == ["persist"]
    spec.is_compiled = False
    spec.compile_and_persist = lambda: None
    with pytest.raises(RuntimeError, match="not persisted"):
        cc._precompile_kernel(*args)


def test_build_modes_do_not_load_and_preserve_normal_cache_path(monkeypatch):
    from flashinfer.jit import cute_dsl_core

    calls = []
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    monkeypatch.setattr(cc, "_kernel_source_files", lambda: ())
    monkeypatch.setattr(
        cc, "_precompile_kernel", lambda *a: calls.append(("persist", a[1]))
    )
    monkeypatch.setattr(cc, "_probe_kernel", lambda *a: calls.append(("probe", a[1])))
    monkeypatch.setattr(
        cute_dsl_core,
        "build_and_load_cute_dsl_kernel",
        lambda *a, **k: calls.append(("load", a[1])),
    )

    def build():
        return cc.build_and_load_attention_kernel(
            "mla_decode", "main", (1, 2), "", lambda: pytest.fail("unexpected compile")
        )

    with cc.precompile_only():
        placeholder = build()
        assert cc.is_placeholder(placeholder)
        with pytest.raises(RuntimeError, match="never loaded"):
            placeholder()
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    with cc.probe_cached():
        assert cc.is_placeholder(build())
    with cc.precompile_only(), pytest.raises(RuntimeError, match="DISABLE_JIT"):
        build()
    build()
    assert [c[0] for c in calls] == ["persist", "probe", "load"]
    assert len({c[1] for c in calls}) == 1
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT")
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    with cc.precompile_only(), pytest.raises(RuntimeError, match="on-disk"):
        build()


def test_sparse_plan_normalizes_current_device_before_cache(monkeypatch):
    device = [0]
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device[0])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a: SimpleNamespace(multi_processor_count=148),
    )
    monkeypatch.setattr(
        mla_decode, "_cached_sparse_mla_plan", lambda *a: calls.append(a)
    )
    for current in (0, 1):
        device[0] = current
        for value in (None, "cuda", torch.device("cuda")):
            mla_decode._resolve_sparse_mla_plan(value, 2, 16, max_topk=128)
    assert [c[0].index for c in calls] == [0, 0, 0, 1, 1, 1]
    for batch in (True, 2.0):
        with pytest.raises(ValueError, match="positive integer"):
            mla_decode._resolve_sparse_mla_plan("cuda", batch, 16, max_topk=128)
    assert len(calls) == 6


def test_sparse_tensor_keys_include_metadata_and_reject_non_tensors():
    storage = torch.empty(16, dtype=torch.float32)
    key = mla_decode._sparse_tensor_key(storage)
    assert key != mla_decode._sparse_tensor_key(storage.view(4, 4))
    assert key != mla_decode._sparse_tensor_key(storage.view(torch.int32))
    with pytest.raises(TypeError):
        mla_decode._sparse_tensor_key("not a tensor")


@pytest.mark.parametrize("first", ["helpers_paged", "fmha_resources"])
def test_paged_helpers_import_in_either_order(first):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"from flashinfer.attention.prims_ts.kernels.fmha_context import {first}; "
            "from flashinfer.attention.prims_ts.kernels.fmha_context import fmha_resources, helpers_paged; "
            "assert fmha_resources._load_runtime_seq_len_kv is helpers_paged._load_runtime_seq_len_kv",
        ],
        check=True,
    )
