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

Regression test for https://github.com/flashinfer-ai/flashinfer/issues/4782

The JIT path delegates .so freshness entirely to ninja's dependency scan
(JitSpecNvcc.try_load returns None for non-AOT modules), so staging must not
touch the mtime of unchanged sources: if gen_bgmv_moe_module() re-stages an
unchanged source with a fresh timestamp, every new process makes the staged
sources newer than the already-built objects and ninja rebuilds from scratch.
"""

import os

import pytest

import flashinfer.jit.bgmv_moe as bgmv_moe_jit
from flashinfer.jit import core as jit_core
from flashinfer.jit import env as jit_env

# Filenames gen_bgmv_moe_module() expects in the bgmv_moe csrc directory.
_SOURCE_FILES = [
    "moe_bgmv_binding.cu",
    "moe_bgmv_bf16_bf16_bf16.cu",
    "moe_bgmv_bf16_fp32_bf16.cu",
    "moe_bgmv_fp16_fp16_fp16.cu",
    "moe_bgmv_fp16_fp32_fp16.cu",
    "moe_bgmv_fp32_bf16_bf16.cu",
    "moe_bgmv_fp32_fp16_fp16.cu",
]
_HEADER_FILES = [
    "moe_bgmv_impl.cuh",
    "moe_bgmv_config.h",
    "moe_bgmv_ops.h",
    "moe_bgmv_ops.cu",
    "kernel_config.h",
]

# A fixed timestamp far in the past (2020-09-13T12:26:40Z), so that any staging
# step that stamps "now" onto a copy is unambiguously detectable without sleeps.
_OLD_MTIME_NS = 1_600_000_000_000_000_000


class _StubCompilationContext:
    """No-GPU stand-in so the generation pass needs no CUDA device."""

    TARGET_CUDA_ARCHS = [(9, "0a")]

    def get_nvcc_flags_list(self, supported_major_versions=None):
        return []


@pytest.fixture
def bgmv_moe_workspace(tmp_path, monkeypatch):
    """Fake csrc dir with old mtimes plus a redirected JIT workspace."""
    csrc_dir = tmp_path / "csrc" / "bgmv_moe"
    csrc_dir.mkdir(parents=True)
    for fname in _SOURCE_FILES + _HEADER_FILES:
        path = csrc_dir / fname
        path.write_text(f"// fake {fname} for issue #4782 regression test\n")
        os.utime(path, ns=(_OLD_MTIME_NS, _OLD_MTIME_NS))

    monkeypatch.setattr(bgmv_moe_jit, "_get_bgmv_moe_csrc_dir", lambda: csrc_dir)
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path / "cached_ops")
    stub_ctx = _StubCompilationContext()
    monkeypatch.setattr(bgmv_moe_jit, "current_compilation_context", stub_ctx)
    monkeypatch.setattr(jit_core, "current_compilation_context", stub_ctx)

    bgmv_moe_jit.gen_bgmv_moe_module.cache_clear()
    yield tmp_path
    # Drop specs whose paths point into the (now gone) tmp workspace.
    bgmv_moe_jit.gen_bgmv_moe_module.cache_clear()


def test_restaging_unchanged_sources_keeps_built_artifact_fresh(
    bgmv_moe_workspace,
):
    # First generation pass: a fresh process stages sources for the build.
    spec = bgmv_moe_jit.gen_bgmv_moe_module()
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / bgmv_moe_jit.get_bgmv_moe_uri()
    staged_files = [gen_directory / f for f in _SOURCE_FILES + _HEADER_FILES]
    for staged in staged_files:
        assert staged.exists(), f"staging did not produce {staged}"

    # Simulate the .so a completed build would leave behind: newer than the
    # staged sources it was built from, but still in the past relative to now.
    so_path = spec.jit_library_path
    so_path.parent.mkdir(parents=True, exist_ok=True)
    so_path.write_bytes(b"")
    artifact_mtime_ns = _OLD_MTIME_NS + 3600 * 10**9
    os.utime(so_path, ns=(artifact_mtime_ns, artifact_mtime_ns))

    # A new process re-runs module generation over the unchanged sources.
    bgmv_moe_jit.gen_bgmv_moe_module.cache_clear()
    bgmv_moe_jit.gen_bgmv_moe_module()

    # Freshness invariant: re-staging unchanged inputs must not make any
    # staged source/header newer than the existing artifact, or ninja's
    # dependency scan sees stale objects and rebuilds every kernel.
    stale = [
        staged.name
        for staged in staged_files
        if staged.stat().st_mtime_ns > artifact_mtime_ns
    ]
    assert not stale, (
        "re-staging unchanged bgmv_moe inputs made these staged files newer "
        f"than the already-built artifact, forcing a full JIT rebuild: {stale}"
    )
