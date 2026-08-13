import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.jit import core, cpp_ext
from flashinfer.jit.attention import modules as attention_modules
from flashinfer.utils import (
    PosEncodingMode,
    determine_attention_backend,
    is_fa3_prefill_head_dim_supported,
)
from tests.test_helpers import jit_utils


def test_nvcc_parallelism_flags_use_flashinfer_nvcc_threads(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


def test_nvcc_parallelism_flags_ignore_sccache_launcher(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVCC_THREADS", "4")
    monkeypatch.setenv("FLASHINFER_NVCC_LAUNCHER", "sccache")

    assert cpp_ext.get_nvcc_parallelism_flags() == ["--threads=4"]


def test_generate_ninja_uses_sccache_compatible_nvcc_depfile_flag(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    monkeypatch.setattr(cpp_ext.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "7.5")

    ninja = cpp_ext.generate_ninja_build_for_op(
        name="test_module",
        sources=[tmp_path / "generated" / "kernel.cu"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    assert "--generate-dependencies-with-compile -MF $out.d" in ninja
    assert "--dependency-output" not in ninja


def test_debug_jit_uses_sccache_compatible_nvcc_device_debug_flag(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "--device-debug" in spec.extra_cuda_cflags
    assert "-G" not in spec.extra_cuda_cflags


def test_release_jit_propagates_ndebug_to_host_cflags(monkeypatch):
    monkeypatch.delenv("FLASHINFER_JIT_DEBUG", raising=False)
    monkeypatch.delenv("FLASHINFER_JIT_VERBOSE", raising=False)
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" in spec.extra_cflags
    assert "-DNDEBUG" in spec.extra_cuda_cflags


def test_debug_jit_does_not_propagate_ndebug(monkeypatch):
    monkeypatch.setenv("FLASHINFER_JIT_DEBUG", "1")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "get_nvcc_parallelism_flags", lambda: ["--threads=1"])

    spec = core.gen_jit_spec(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_paths=None,
    )

    assert "-DNDEBUG" not in spec.extra_cflags
    assert "-DNDEBUG" not in spec.extra_cuda_cflags


def test_run_ninja_uses_max_jobs(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setenv("MAX_JOBS", "8")
    monkeypatch.setattr(cpp_ext.subprocess, "run", fake_run)

    cpp_ext.run_ninja(tmp_path, tmp_path / "build.ninja", verbose=False)

    assert commands == [
        [
            "ninja",
            "-v",
            "-C",
            str(tmp_path.resolve()),
            "-f",
            str((tmp_path / "build.ninja").resolve()),
            "-j",
            "8",
        ]
    ]


def test_jit_spec_build_rewrites_ninja_before_build(monkeypatch):
    writes = []
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)

    spec = core.JitSpecNvcc(
        name="test_module",
        sources=[],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    monkeypatch.setattr(spec, "write_ninja", lambda _content=None: writes.append(True))
    monkeypatch.setattr(core, "run_ninja", lambda *_args, **_kwargs: None)

    spec.build(verbose=False, need_lock=False)

    assert writes == [True]


def test_customize_batch_prefill_nvfp4_large_head_uses_prefill_flags(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(
        attention_modules.current_compilation_context, "TARGET_CUDA_ARCHS", {(8, 6)}
    )
    monkeypatch.setattr(
        attention_modules.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "gen"
    )

    spec = attention_modules.gen_customize_batch_prefill_module(
        "fa2",
        "test_batch_prefill_nvfp4_large_head",
        torch.float16,
        torch.uint8,
        torch.float16,
        torch.int32,
        512,
        512,
        # NVFP4 (uint8) KV paged prefill now requires the scale-factor tensors as
        # additional inputs (maybe_k_cache_sf / maybe_v_cache_sf), matching the
        # generator contract; pass them so generation reaches the flag assertions.
        ["maybe_k_cache_sf", "maybe_v_cache_sf"],
        ["uint8_t", "uint8_t"],
        ["sm_scale"],
        ["double"],
        "DefaultAttention<false, false, false, false>",
        "#include <flashinfer/attention/variants.cuh>",
    )

    assert any("sm_86" in flag for flag in spec.extra_cuda_cflags)
    with pytest.raises(RuntimeError, match="No supported CUDA architectures"):
        attention_modules._fa2_head_dim_nvcc_flags(512, 512, torch.uint8)


@pytest.mark.parametrize(
    ("head_dim_qk", "head_dim_vo", "supported"),
    [
        (64, 64, True),
        (128, 128, True),
        (256, 256, True),
        (192, 128, True),
        (512, 512, False),
        (256, 128, False),
        (128, 192, False),
    ],
)
def test_fa3_prefill_head_dim_supported(head_dim_qk, head_dim_vo, supported):
    assert is_fa3_prefill_head_dim_supported(head_dim_qk, head_dim_vo) is supported


@pytest.mark.parametrize(
    ("head_dim_qk", "head_dim_vo", "expected_backend"),
    [
        (256, 256, "fa3"),
        (192, 128, "fa3"),
        (512, 512, "fa2"),
    ],
)
def test_determine_attention_backend_respects_fa3_prefill_head_dim(
    monkeypatch, head_dim_qk, head_dim_vo, expected_backend
):
    monkeypatch.setattr(flashinfer.utils, "is_sm90a_supported", lambda device: True)

    backend = determine_attention_backend(
        torch.device("cuda"),
        PosEncodingMode.NONE.value,
        use_fp16_qk_reductions=False,
        use_custom_mask=False,
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
    )

    assert backend == expected_backend


def test_prefill_jit_helper_skips_fa3_unsupported_large_head(monkeypatch):
    calls = []

    def fake_single_prefill_module(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        head_dim_qk,
        head_dim_vo,
        *_args,
    ):
        calls.append(("single", backend, head_dim_qk, head_dim_vo))
        return SimpleNamespace(name=f"{backend}_single_{head_dim_qk}_{head_dim_vo}")

    def fake_batch_prefill_module(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim_qk,
        head_dim_vo,
        *_args,
    ):
        calls.append(("batch", backend, head_dim_qk, head_dim_vo))
        return SimpleNamespace(name=f"{backend}_batch_{head_dim_qk}_{head_dim_vo}")

    monkeypatch.setattr(jit_utils, "is_sm90a_supported", lambda device: True)
    monkeypatch.setattr(
        flashinfer.prefill, "gen_single_prefill_module", fake_single_prefill_module
    )
    monkeypatch.setattr(
        flashinfer.prefill, "gen_batch_prefill_module", fake_batch_prefill_module
    )
    monkeypatch.setattr(
        flashinfer.quantization,
        "gen_quantization_module",
        lambda: SimpleNamespace(name="quantization"),
    )
    monkeypatch.setattr(
        flashinfer.page,
        "gen_page_module",
        lambda: SimpleNamespace(name="page"),
    )

    jit_utils.gen_prefill_attention_modules(
        q_dtypes=[torch.float16],
        kv_dtypes=[torch.float16],
        head_dims=[512],
        pos_encoding_modes=[PosEncodingMode.NONE.value],
        use_sliding_window_options=[False],
        use_logits_soft_cap_options=[False],
        use_fp16_qk_reduction_options=[False],
    )

    assert ("single", "fa3", 512, 512) not in calls
    assert ("batch", "fa3", 512, 512) not in calls
    assert ("single", "fa2", 512, 512) in calls
    assert ("batch", "fa2", 512, 512) in calls


# ---------------------------------------------------------------------------
# NVCC JIT cache build-fingerprint invalidation (meta.json)
# ---------------------------------------------------------------------------


def _make_nvcc_spec(monkeypatch, tmp_path, name="test_fp_module", mtime_shift=None):
    """Build a JitSpecNvcc isolated in tmp_path, with source files in a
    separate (later-hashed) tree whose mtimes can be backdated."""
    monkeypatch.setattr(core.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "_get_compiler_identity", lambda: "nvcc=mock;cxx=mock")
    # Deterministic fingerprints regardless of the host toolchain.
    monkeypatch.setattr(core, "_get_wheel_record_hash", lambda: "mock-record")
    monkeypatch.setattr(core, "_get_tvm_ffi_version", lambda: "mock-tvmffi")

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    src = src_dir / "kernel.cu"
    src.write_text("__global__ void k() {}")
    if mtime_shift is not None:
        os.utime(src, (mtime_shift, mtime_shift))

    spec = core.JitSpecNvcc(
        name=name,
        sources=[src],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        extra_include_dirs=None,
    )

    calls = {}

    def fake_render_ninja():
        return (
            "ninja_required_version = 1.3\n"
            f"cflags = {spec.extra_cflags}\n"
            f"cuda_cflags = {spec.extra_cuda_cflags}\n"
            f"ldflags = {spec.extra_ldflags}\n"
        )

    def fake_write_ninja(content=None):
        spec.build_dir.mkdir(parents=True, exist_ok=True)
        content = fake_render_ninja() if content is None else content
        (spec.build_dir / "build.ninja").write_text(content)

    def fake_run_ninja(*_args, **_kwargs):
        spec.build_dir.mkdir(parents=True, exist_ok=True)
        (spec.build_dir / f"{spec.name}.so").write_bytes(b"\x7fELF")
        # Sentinel proving the directory was freshly rebuilt from scratch.
        (spec.build_dir / "built.marker").write_text("1")
        calls["builds"] = calls.get("builds", 0) + 1

    monkeypatch.setattr(spec, "_render_ninja", fake_render_ninja)
    monkeypatch.setattr(spec, "write_ninja", fake_write_ninja)
    monkeypatch.setattr(core, "run_ninja", fake_run_ninja)
    monkeypatch.setattr(core.jit_env, "FLASHINFER_CSRC_DIR", tmp_path / "csrc")
    return spec, calls


def test_meta_schema_invariants(monkeypatch, tmp_path):
    spec, _ = _make_nvcc_spec(monkeypatch, tmp_path)
    meta = spec.expected_meta
    assert meta["schema"] == core._META_SCHEMA_VERSION
    assert "flashinfer_version" in meta
    assert "python_soabi" in meta
    assert "torch_build_identity" in meta
    assert "source_sha256" in meta
    assert "cflags" in meta
    assert "cuda_cflags" in meta
    # Deterministic: computing twice yields identical metadata.
    assert meta == spec.expected_meta


def test_meta_committed_after_successful_build(monkeypatch, tmp_path):
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    assert not spec.meta_path.exists()
    spec.build(verbose=False, need_lock=False)
    assert spec.meta_path.exists()
    assert core._read_meta(spec.meta_path) == spec.expected_meta
    assert calls["builds"] == 1


def test_stale_source_rewrites_same_mtime_still_invalidates(monkeypatch, tmp_path):
    """Source bytes change while the mtime stays *older* than the committed
    artifacts: the fingerprint must change and wipe the stale build dir."""
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path, mtime_shift=1_700_000_000)
    # First build: mtime is old already; a naive ninja timestamp scan would
    # think everything is fresh. Meta still commits.
    spec.build(verbose=False, need_lock=False)
    assert (spec.build_dir / f"{spec.name}.so").exists()
    assert (spec.build_dir / "built.marker").exists()
    stale_artifact = spec.build_dir / "stale-artifact"
    stale_artifact.write_text("must be removed")
    old_meta = core._read_meta(spec.meta_path)

    # Change the source content, keep its mtime unchanged.
    src = spec.sources[0]
    old_mtime = src.stat().st_mtime
    src.write_text("__global__ void k() {}\n// changed")
    os.utime(src, (old_mtime, old_mtime))
    assert src.stat().st_mtime == old_mtime

    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    # The stale directory was actually wiped before the fake build recreated
    # its normal outputs.
    assert not stale_artifact.exists()
    assert (
        core._read_meta(spec.meta_path)["sources_sha256"] != old_meta["sources_sha256"]
    )
    assert (spec.build_dir / "built.marker").exists()
    assert (spec.build_dir / "built.marker").read_text() == "1"
    assert (spec.build_dir / f"{spec.name}.so").exists()
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_compile_flags_change_invalidates(monkeypatch, tmp_path):
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1

    spec.extra_cuda_cflags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    assert (spec.build_dir / "built.marker").exists()
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_matching_meta_keeps_build_dir_and_ninja_incremental(monkeypatch, tmp_path):
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1
    build_dir = spec.build_dir
    assert build_dir.exists()

    # No source/flag change: build() keeps the dir and run_ninja is still
    # invoked (ninja owns the fine-grained incremental decision), so builds
    # counter increments once per call but the .so/meta are stable.
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    assert build_dir.exists()
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_missing_meta_triggers_rebuild(monkeypatch, tmp_path):
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1
    spec.meta_path.unlink()
    stale_artifact = spec.build_dir / "stale-without-meta"
    stale_artifact.write_text("must be removed")
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    assert not stale_artifact.exists()
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_invalidation_preserves_generated_sources_in_build_dir(monkeypatch, tmp_path):
    """Generated source trees are build inputs, not stale build outputs."""
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    generated_dir = spec.build_dir / "generated"
    generated_dir.mkdir(parents=True)
    generated_source = generated_dir / "kernel.cu"
    generated_source.write_text("__global__ void generated() {}")
    generated_helper = generated_dir / "kernel_config.inc"
    generated_helper.write_text("// generated helper")
    spec.sources = [generated_source]

    stale_artifact = spec.build_dir / "stale-without-meta"
    stale_artifact.write_text("old build output")
    spec.jit_library_path.write_bytes(b"old shared library")

    spec.build(verbose=False, need_lock=False)

    assert calls["builds"] == 1
    assert generated_source.read_text() == "__global__ void generated() {}"
    assert generated_helper.read_text() == "// generated helper"
    assert not stale_artifact.exists()
    assert spec.jit_library_path.read_bytes() == b"\x7fELF"
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_corrupt_meta_triggers_rebuild(monkeypatch, tmp_path):
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1
    spec.meta_path.write_text("{ not valid json !!!")
    stale_artifact = spec.build_dir / "stale-with-corrupt-meta"
    stale_artifact.write_text("must be removed")
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    assert not stale_artifact.exists()
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_rendered_ninja_change_invalidates_existing_build(monkeypatch, tmp_path):
    """Fingerprint the Ninja content the current process would render, not the
    previous build.ninja stored in the cache directory."""
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1
    old_meta = core._read_meta(spec.meta_path)
    stale_artifact = spec.build_dir / "stale-ninja-config"
    stale_artifact.write_text("must be removed")

    old_render = spec._render_ninja
    monkeypatch.setattr(
        spec, "_render_ninja", lambda: old_render() + "generator_input = changed\n"
    )
    assert (
        old_meta["ninja_content_sha256"] != spec.expected_meta["ninja_content_sha256"]
    )

    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    assert not stale_artifact.exists()
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_failed_ninja_does_not_commit_meta(monkeypatch, tmp_path):
    spec, _ = _make_nvcc_spec(monkeypatch, tmp_path)

    def failing_ninja(*_args, **_kwargs):
        raise RuntimeError("nvcc failed")

    monkeypatch.setattr(core, "run_ninja", failing_ninja)
    with pytest.raises(RuntimeError, match="nvcc failed"):
        spec.build(verbose=False, need_lock=False)
    assert not spec.meta_path.exists()


def test_aot_path_ignores_meta(monkeypatch, tmp_path):
    """AOT artifacts are loaded without touching JIT meta.json."""
    spec, _ = _make_nvcc_spec(monkeypatch, tmp_path)
    aot = tmp_path / "aot" / spec.name / f"{spec.name}.so"
    aot.parent.mkdir(parents=True, exist_ok=True)
    aot.write_bytes(b"\x7fELF")
    monkeypatch.setattr(core.jit_env, "FLASHINFER_AOT_DIR", tmp_path / "aot")
    assert spec.is_aot
    # try_load routes to the AOT path and never consults meta.json.
    monkeypatch.setattr(
        spec, "load", lambda so_path=None: SimpleNamespace(path=so_path)
    )
    loaded = spec.try_load()
    assert loaded is not None
    assert loaded.path == aot


def test_meta_is_json_serializable(monkeypatch, tmp_path):
    spec, _ = _make_nvcc_spec(monkeypatch, tmp_path)
    import json as _json

    _json.dumps(spec.expected_meta)
    # numpy scalars or Path objects must not leak into the fingerprint.
    for v in spec.expected_meta.values():
        assert not isinstance(v, Path)


def test_module_source_hash_independent_of_include_tree(monkeypatch, tmp_path):
    """A change to the module's own source file invalidates even when the
    include tree hash is unchanged."""
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1

    # Mutate the module source (in spec.sources, not the include tree).
    src = spec.sources[0]
    old_mtime = src.stat().st_mtime
    src.write_text(src.read_text() + "\n// module-source change")
    os.utime(src, (old_mtime, old_mtime))

    meta_before = core._read_meta(spec.meta_path)
    assert meta_before["sources_sha256"] != spec.expected_meta["sources_sha256"]

    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_include_tree_change_invalidates_without_source_change(monkeypatch, tmp_path):
    """A change under the include tree invalidates even when the module's
    own sources are untouched."""
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    # Redirect the include-tree hash at a controlled directory.
    include_dir = tmp_path / "include_tree"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "helper.cuh").write_text("// v1")
    monkeypatch.setattr(spec, "extra_include_dirs", [include_dir])

    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1

    # Mutate a file inside the include tree only.
    (include_dir / "helper.cuh").write_text("// v2")

    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 2
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_build_jit_specs_applies_fingerprint_and_commits_meta(monkeypatch, tmp_path):
    """The batch precompile path must run the same fingerprint gate and commit
    meta.json for every module it builds (no stale-artifact reuse)."""
    spec1, calls1 = _make_nvcc_spec(
        monkeypatch, tmp_path, name="batch_a", mtime_shift=1_700_000_000
    )
    spec2, calls2 = _make_nvcc_spec(
        monkeypatch, tmp_path, name="batch_b", mtime_shift=1_700_000_000
    )

    # A batch ninja run "compiles" every subninja module under the JIT dir.
    def batch_run_ninja(*_args, **_kwargs):
        jit_root = core.jit_env.FLASHINFER_JIT_DIR
        for child in jit_root.iterdir():
            if not child.is_dir():
                continue
            so = child / f"{child.name}.so"
            so.write_bytes(b"\x7fELF")
            (child / "built.marker").write_text("1")
            for name_, calls_ in ((spec1, calls1), (spec2, calls2)):
                if name_.build_dir == child:
                    calls_["builds"] = calls_.get("builds", 0) + 1

    monkeypatch.setattr(core, "run_ninja", batch_run_ninja)

    # Build both via the batch entry point.
    core.build_jit_specs([spec1, spec2], verbose=False, skip_prebuilt=False)
    assert calls1["builds"] == 1 and calls2["builds"] == 1
    assert spec1.meta_path.exists() and spec2.meta_path.exists()
    assert core._read_meta(spec1.meta_path) == spec1.expected_meta
    assert core._read_meta(spec2.meta_path) == spec2.expected_meta

    # Same-mtime source change to batch_a must invalidate and rebuild it,
    # while batch_b (unchanged) is preserved.
    src = spec1.sources[0]
    old_mtime = src.stat().st_mtime
    src.write_text(src.read_text() + "\n// changed")
    os.utime(src, (old_mtime, old_mtime))

    core.build_jit_specs([spec1, spec2], verbose=False, skip_prebuilt=False)
    assert calls1["builds"] == 2
    assert calls2["builds"] == 2  # run_ninja invoked for both (ninja decides)
    assert core._read_meta(spec1.meta_path) == spec1.expected_meta
    assert core._read_meta(spec2.meta_path) == spec2.expected_meta


def _lock_is_held_by_another_process(lock_path: Path) -> bool:
    """Probe a FileLock from a separate interpreter.

    A single-spec build uses this same lock, so a timeout proves it cannot race
    the batch builder while Ninja runs or metadata is committed.
    """
    script = """
import sys
from filelock import FileLock, Timeout

try:
    with FileLock(sys.argv[1], timeout=0.2, thread_local=False):
        pass
except Timeout:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(lock_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def test_build_jit_specs_holds_module_lock_through_meta_commit(monkeypatch, tmp_path):
    spec, _ = _make_nvcc_spec(monkeypatch, tmp_path, name="batch_locked")
    lock_observations = []

    def batch_run_ninja(*_args, **_kwargs):
        lock_observations.append(_lock_is_held_by_another_process(spec.lock_path))
        spec.jit_library_path.write_bytes(b"\x7fELF")

    original_write_meta = core._write_meta_atomic

    def write_meta_while_observing_lock(meta_path, meta):
        lock_observations.append(_lock_is_held_by_another_process(spec.lock_path))
        original_write_meta(meta_path, meta)

    monkeypatch.setattr(core, "run_ninja", batch_run_ninja)
    monkeypatch.setattr(core, "_write_meta_atomic", write_meta_while_observing_lock)

    core.build_jit_specs([spec], verbose=False, skip_prebuilt=False)

    assert lock_observations == [True, True]
    assert not _lock_is_held_by_another_process(spec.lock_path)
    assert core._read_meta(spec.meta_path) == spec.expected_meta


def test_wipe_failure_aborts_build(monkeypatch, tmp_path):
    """A failed stale-directory wipe must abort instead of blessing residual
    objects with a fresh meta.json."""
    spec, calls = _make_nvcc_spec(monkeypatch, tmp_path)
    spec.build(verbose=False, need_lock=False)
    assert calls["builds"] == 1

    # Make the fingerprint mismatch, then force rmtree to fail.
    spec.extra_cuda_cflags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]

    def failing_rmtree(*_a, **_k):
        raise OSError("cannot remove")

    monkeypatch.setattr(core.shutil, "rmtree", failing_rmtree)
    meta_before = core._read_meta(spec.meta_path)
    with pytest.raises(OSError, match="cannot remove"):
        spec.build(verbose=False, need_lock=False)
    # The failed build must not refresh the fingerprint to bless residual
    # objects compiled from the old source.
    assert core._read_meta(spec.meta_path) == meta_before
    assert not spec.is_compiled


def test_is_compiled_requires_valid_meta(monkeypatch, tmp_path):
    spec, _ = _make_nvcc_spec(monkeypatch, tmp_path)
    assert not spec.is_compiled
    spec.build(verbose=False, need_lock=False)
    assert spec.is_compiled
    # Corrupt the committed meta: .so still exists but the module is not
    # considered compiled anymore.
    spec.meta_path.write_text("{ broken")
    assert not spec.is_compiled
