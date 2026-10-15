# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile identity and device-local occupancy regression coverage."""

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import enum
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from flashinfer.attention.prims_ts._compile_cache import (
    _compile_cache_key,
    _kernel_source_files,
    build_and_load_attention_kernel,
)
from flashinfer.attention.prims_ts import _hardware


@dataclass(frozen=True)
class _Spec:
    device_index: int = 0
    shape: tuple = (128, 256)
    kernel_signature: tuple = ("kernel", 4)
    kernel: object = field(default=None, compare=False)


def test_compile_identity_tracks_codegen_but_not_device_placement():
    spec = _Spec()
    key = _compile_cache_key(spec, "--opt-level 3")
    assert key is not None
    assert key == _compile_cache_key(
        replace(spec, device_index=1, kernel=object()), "--opt-level 3"
    )
    assert key != _compile_cache_key(replace(spec, shape=(256, 256)), "--opt-level 3")
    assert key != _compile_cache_key(
        replace(spec, kernel_signature=("kernel", 8)), "--opt-level 3"
    )
    assert key != _compile_cache_key(spec, "--opt-level 2")
    assert _compile_cache_key(spec, "", (132,)) != _compile_cache_key(spec, "", (148,))


def test_keys_are_stable_across_processes():
    script = """
from flashinfer.attention.prims_ts._compile_cache import _compile_cache_key
from flashinfer.attention.prims_ts.kernels.stage import FmhaStage
print(_compile_cache_key((128, FmhaStage.Loop, None, True), '--opt-level 3'))
"""
    keys = [
        subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        for _ in range(2)
    ]
    assert keys[0] == keys[1]
    assert keys[0] != "None"


def test_enum_identity_does_not_alias_an_integer():
    class Mode(enum.IntEnum):
        FIRST = 1

    assert _compile_cache_key((Mode.FIRST,), "") != _compile_cache_key((1,), "")


def test_unknown_compile_values_bypass_persistence(monkeypatch):
    from flashinfer.jit import cute_dsl_core

    monkeypatch.setattr(
        cute_dsl_core,
        "build_and_load_cute_dsl_kernel",
        lambda *a, **kw: pytest.fail("unexpected persistence"),
    )
    sentinel = object()
    assert (
        build_and_load_attention_kernel(
            "decode", "main", (object(),), "", lambda: sentinel
        )
        is sentinel
    )


def test_source_dependencies_include_shared_attention_code():
    sources = {
        Path(path).relative_to(Path(_hardware.__file__).parent).as_posix()
        for path in _kernel_source_files()
    }
    assert {
        "decode.py",
        "context.py",
        "mla_decode.py",
        "sage.py",
        "_compile_cache.py",
        "_block_sparse/compiler.py",
        "_block_sparse/prepared.py",
        "kernels/mask.py",
        "kernels/separate_reduction.py",
        "kernels/task_compat.py",
        "kernels/fmha_decode/block_sparse_prepare.py",
        "kernels/fmha_context/fmha_kernel.py",
        "kernels/mla_decode/throughput_2cta/kernel.py",
    } <= sources


def test_disabled_cache_compiles_directly(monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    sentinel = object()
    assert (
        build_and_load_attention_kernel("decode", "main", _Spec(), "", lambda: sentinel)
        is sentinel
    )


def test_occupancy_queries_are_device_local_and_failures_are_not_cached(monkeypatch):
    utils = pytest.importorskip("cutlass.utils")
    device = [0]
    calls = []
    fail = [False]

    def query(cluster_size, stream):
        calls.append((device[0], cluster_size))
        if fail[0]:
            raise RuntimeError("probe failed")
        return (148 - device[0]) // cluster_size

    monkeypatch.setattr(torch.cuda, "current_device", lambda: device[0])
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=0)
    )
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: None)
    monkeypatch.setattr(
        utils, "HardwareInfo", lambda *_: SimpleNamespace(get_max_active_clusters=query)
    )
    _hardware._query_max_active_clusters.cache_clear()
    try:
        assert _hardware.get_max_active_clusters(1) == 148
        assert _hardware.get_max_active_clusters(1) == 148
        assert _hardware.get_max_active_clusters(2) == 74
        device[0] = 1
        assert _hardware.get_max_active_clusters(1) == 147
        fail[0] = True
        with pytest.raises(RuntimeError, match="probe failed"):
            _hardware.get_max_active_clusters(2)
        fail[0] = False
        assert _hardware.get_max_active_clusters(2) == 73
        assert calls == [(0, 1), (0, 2), (1, 1), (1, 2), (1, 2)]
    finally:
        _hardware._query_max_active_clusters.cache_clear()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="PrimTS attention requires Blackwell",
)
def test_public_attention_disk_cache_in_fresh_processes(tmp_path):
    pytest.importorskip("cutlass", minversion="4.7.0")
    # Reuse the public numerical oracles for each distinct compiler entry.
    targets = [
        "test_attention_ts_decode.py::test_attention_ts_decode_compact_variable_k_acceptance[F7-fp8-sq4-q128-separate]",
        "test_attention_ts_context.py::test_attention_ts_context_bounded_public_correctness_matrix[fixed-fp16-dense-mha-one-shot]",
        "test_attention_ts_context.py::test_attention_ts_context_paged_supported_page_sizes_accuracy[d128-bf16-32]",
        "test_attention_ts_mla_decode.py::test_attention_ts_mla_decode_compact_variable_k_acceptance[bf16-1cta-direct]",
        "test_attention_ts_mla_decode.py::test_attention_ts_mla_decode_compact_variable_k_acceptance[bf16-short-k-2cta-fallback]",
        "test_attention_ts_block_sparse.py::test_public_block_sparse_correctness[q16_kv16_fp16_noncausal_holey_static]",
        "test_attention_ts_block_sparse.py::test_public_paged_one_shot_q64_kv256_gqa_matches_reference[live96-two-pages]",
        "test_attention_ts_sage.py::test_sage_matches_dequantized_reference[kv256_fp8_k1_one_shot]",
    ]
    script = """
import sys
import pytest
import torch
from flashinfer.attention.prims_ts import _compile_cache
from flashinfer.jit.cute_dsl_core import JitSpecCuteDsl

original_key = _compile_cache._compile_cache_key
def checked_key(*args, **kwargs):
    key = original_key(*args, **kwargs)
    assert key is not None, 'compile specification bypassed disk cache'
    return key
_compile_cache._compile_cache_key = checked_key

if sys.argv[1] == 'warm':
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)
    original_build = JitSpecCuteDsl.build
    def no_attention_recompile(self):
        assert not self.module_name.startswith('prims_ts_'), self.name
        return original_build(self)
    JitSpecCuteDsl.build = no_attention_recompile

raise SystemExit(pytest.main(['-q', '--maxfail=1', *sys.argv[2:]]))
"""
    env = dict(os.environ)
    env["FLASHINFER_WORKSPACE_BASE"] = str(tmp_path)
    env["CUTE_DSL_CACHE_DIR"] = str(tmp_path / "cute_dsl")
    env.pop("FLASHINFER_CUTE_DSL_DISABLE_CACHE", None)
    root = Path(__file__).resolve().parents[2]
    for phase in ("cold", "warm"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                phase,
                *[f"tests/attention/{target}" for target in targets],
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    artifacts = {path.parent.name for path in tmp_path.rglob("*.o")}
    for family in ("decode", "context", "mla_decode", "block_sparse"):
        assert any(name.startswith(f"prims_ts_{family}_") for name in artifacts)
