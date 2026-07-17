"""Packaging and source-snapshot checks for the SM90 push FP8 kernel."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
import zipfile
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest


_PACKAGE_PATH = "flashinfer/moe_ep/kernel_src/sm90_push_megamoe"
_CUDA_RESOURCES = (
    "src/a2a/sm90_push_a2a_ops.cu",
    "src/a2a/sm90_push_a2a.cuh",
    "src/fp8_gemm/fp8_moe_binding.cu",
    "src/fp8_gemm/fp8_moe_fc1_fused.cuh",
    "src/fp8_gemm/fp8_moe_jit.cuh",
    "src/fp8_gemm/fp8_moe_launcher.cuh",
    "src/fp8_gemm/fp8_moe_scheduler.cuh",
)
_PYTHON_RESOURCES = (
    "__init__.py",
    "shim/__init__.py",
    "shim/gemm.py",
    "shim/jit.py",
    "shim/protocol.py",
    "shim/runner.py",
    "shim/weights.py",
)
_DOCUMENT_RESOURCES = ("ACKNOWLEDGEMENT.md",)


def _resource_at(package_root, relative_path: str):
    resource = package_root
    for part in relative_path.split("/"):
        resource = resource.joinpath(part)
    return resource


def test_sm90_push_package_data_contains_cuda_sources():
    project_root = Path(__file__).resolve().parents[2]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    key = '"flashinfer.moe_ep.kernel_src.sm90_push_megamoe" = ['
    package_block = pyproject.split(key, maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert '"*.md"' in package_block
    assert '"src/a2a/*.cu"' in package_block
    assert '"src/a2a/*.cuh"' in package_block
    assert '"src/fp8_gemm/*.cu"' in package_block
    assert '"src/fp8_gemm/*.cuh"' in package_block


def test_sm90_push_runtime_resources_expose_packaged_cuda_sources():
    package_root = resources.files("flashinfer.moe_ep.kernel_src.sm90_push_megamoe")
    for relative_path in (
        *_CUDA_RESOURCES,
        *_PYTHON_RESOURCES,
        *_DOCUMENT_RESOURCES,
    ):
        resource = _resource_at(package_root, relative_path)
        assert resource.is_file()
        assert resource.read_text(encoding="utf-8").strip()


def test_sm90_push_prebuilt_wheel_contains_runtime_package():
    wheel_env = os.environ.get("FLASHINFER_TEST_WHEEL")
    if wheel_env is None:
        pytest.skip("set FLASHINFER_TEST_WHEEL to inspect a prebuilt wheel")

    wheel = Path(wheel_env).expanduser()
    assert wheel.is_file(), f"FLASHINFER_TEST_WHEEL is not a file: {wheel}"
    assert wheel.suffix == ".whl", f"expected a .whl file, got: {wheel}"
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())

    required = {
        f"{_PACKAGE_PATH}/{relative_path}"
        for relative_path in (
            *_CUDA_RESOURCES,
            *_PYTHON_RESOURCES,
            *_DOCUMENT_RESOURCES,
        )
    }
    missing = sorted(required - members)
    assert not missing, "prebuilt wheel is missing SM90 push files:\n" + "\n".join(
        missing
    )


def test_sm90_push_backend_imports_kernel_package_through_public_boundaries():
    project_root = Path(__file__).resolve().parents[2]
    backend_root = (
        project_root
        / "flashinfer"
        / "moe_ep"
        / "backends"
        / "mega"
        / "kernel"
        / "sm90_push_fp8"
    )
    package_marker = "kernel_src.sm90_push_megamoe"

    for path in backend_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert f"{package_marker}.src" not in source
        tree = ast.parse(source, filename=str(path))
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                targets.append(node.module)
        for target in targets:
            if package_marker not in target:
                continue
            suffix = target.split(package_marker, maxsplit=1)[1]
            assert suffix in ("", ".shim") or suffix.startswith(".shim."), (
                f"{path} bypasses the SM90 push package boundary with import {target!r}"
            )


def test_sm90_push_weight_helpers_defer_kernel_package_import():
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_JIT"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import flashinfer.moe_ep.backends.mega.kernel.sm90_push_fp8.weights; "
            "assert 'flashinfer.moe_ep.kernel_src.sm90_push_megamoe' "
            "not in sys.modules",
        ],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_sm90_push_snapshot_is_content_addressed_and_mtime_stable(
    tmp_path, monkeypatch
):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import jit

    captured = []

    def fake_gen_jit_spec(name, sources, **kwargs):
        spec = SimpleNamespace(name=name, sources=list(sources), **kwargs)
        captured.append(spec)
        return spec

    monkeypatch.setattr(jit.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    monkeypatch.setattr(jit, "gen_jit_spec", fake_gen_jit_spec)

    first = jit.gen_sm90_push_a2a_module()
    snapshot_dir = tmp_path / first.name
    source_paths = [
        snapshot_dir / "sm90_push_a2a_ops.cu",
        snapshot_dir / "sm90_push_a2a.cuh",
    ]
    before = {path: path.stat().st_mtime_ns for path in source_paths}
    time.sleep(0.01)
    second = jit.gen_sm90_push_a2a_module()
    after = {path: path.stat().st_mtime_ns for path in source_paths}

    assert first.name == second.name == jit.sm90_push_a2a_uri()
    assert first.sources == second.sources == [source_paths[0]]
    assert first.extra_include_paths == second.extra_include_paths == [snapshot_dir]
    assert before == after
    assert captured == [first, second]
    for source_path in source_paths:
        packaged = jit._SOURCE_DIR / source_path.name
        assert source_path.read_text(encoding="utf-8") == packaged.read_text(
            encoding="utf-8"
        )


def test_sm90_push_private_gemm_snapshot_is_complete_and_mtime_stable(
    tmp_path, monkeypatch
):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import gemm

    captured = []

    def fake_gen_jit_spec(name, sources, **kwargs):
        spec = SimpleNamespace(name=name, sources=list(sources), **kwargs)
        captured.append(spec)
        return spec

    monkeypatch.setattr(gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    monkeypatch.setattr(gemm, "gen_jit_spec", fake_gen_jit_spec)
    monkeypatch.setattr(gemm, "is_cuda_version_at_least", lambda _version: True)

    first = gemm.gen_sm90_push_fp8_moe_gemm_module()
    snapshot_dir = tmp_path / first.name
    source_paths = [snapshot_dir / name for name in gemm._SOURCE_NAMES]
    build_config = snapshot_dir / "fp8_moe_build_config.h"
    before = {path: path.stat().st_mtime_ns for path in [*source_paths, build_config]}
    time.sleep(0.01)
    second = gemm.gen_sm90_push_fp8_moe_gemm_module()
    after = {path: path.stat().st_mtime_ns for path in [*source_paths, build_config]}

    assert first.name == second.name == gemm.sm90_push_fp8_moe_gemm_uri()
    assert first.sources[0] == second.sources[0] == source_paths[0]
    assert first.extra_include_paths[0] == second.extra_include_paths[0] == snapshot_dir
    assert before == after
    assert captured == [first, second]
    assert first.extra_ldflags == second.extra_ldflags == ["-lnvrtc", "-lcuda"]
    assert "-DENABLE_FP8_BLOCK_SCALE" in first.extra_cuda_cflags
    build_config_bytes = build_config.read_bytes()
    assert b"FLASHINFER_SM90_PUSH_FP8_MOE_SOURCE_DIGEST" in build_config_bytes
    assert first.name.rsplit("_", maxsplit=1)[-1].encode() in build_config_bytes
    for source_path in source_paths:
        packaged = gemm._SOURCE_DIR / source_path.name
        assert source_path.read_bytes() == gemm._canonical_source(packaged)


def test_sm90_push_private_gemm_requires_cuda_12_8(tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import gemm

    checked = []

    def unsupported(version):
        checked.append(version)
        return False

    monkeypatch.setattr(gemm, "is_cuda_version_at_least", unsupported)
    monkeypatch.setattr(gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)

    with pytest.raises(RuntimeError, match=r"requires CUDA Toolkit 12\.8"):
        gemm.gen_sm90_push_fp8_moe_gemm_module()

    assert checked == ["12.8"]
    assert not any(tmp_path.iterdir())


def test_sm90_push_aot_keeps_a2a_and_gates_gemm_on_cuda_12_8():
    project_root = Path(__file__).resolve().parents[2]
    source = (project_root / "flashinfer" / "aot.py").read_text(encoding="utf-8")

    a2a = source.index("jit_specs.append(gen_sm90_push_a2a_module())")
    guard = source.index('if get_cuda_version() >= Version("12.8"):', a2a)
    gemm = source.index("jit_specs.append(gen_sm90_push_fp8_moe_gemm_module())", guard)
    skip = source.index("Skipping SM90 push FP8 MoE GEMM AOT module", gemm)

    assert a2a < guard < gemm < skip


def test_sm90_push_private_gemm_cubin_digest_covers_dependencies():
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import gemm

    sources = {name: name.encode() for name in gemm._SOURCE_NAMES}
    dependencies = {name: name.encode() for name in gemm._DEPENDENCY_NAMES}

    flags = ("-arch=sm_90a",)
    first = gemm._digest(sources, dependencies, flags)
    dependencies["nv_internal/include/tensorrt_llm/common/cudaFp8Utils.h"] += (
        b"\nchanged"
    )
    second = gemm._digest(sources, dependencies, flags)
    jit_source = (gemm._SOURCE_DIR / "fp8_moe_jit.cuh").read_text(encoding="utf-8")

    assert first[:20] != second[:20]
    assert "FLASHINFER_SM90_PUSH_FP8_MOE_SOURCE_DIGEST +" in jit_source


def test_sm90_push_private_gemm_module_cache_is_keyed_by_uri(monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import gemm

    built = []
    current_uri = ["private_gemm_a"]

    class FakeModule:
        def set_deepgemm_jit_include_dirs(self, include_dirs):
            self.include_dirs = include_dirs

    class FakeSpec:
        def __init__(self, name):
            self.name = name

        def build_and_load(self):
            built.append(self.name)
            return FakeModule()

    monkeypatch.setattr(gemm, "sm90_push_fp8_moe_gemm_uri", lambda: current_uri[0])
    monkeypatch.setattr(
        gemm,
        "gen_sm90_push_fp8_moe_gemm_module",
        lambda: FakeSpec(current_uri[0]),
    )
    monkeypatch.setattr(gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", Path("generated"))
    gemm._load_sm90_push_fp8_moe_gemm_module_cached.cache_clear()

    first = gemm._load_sm90_push_fp8_moe_gemm_module()
    assert gemm._load_sm90_push_fp8_moe_gemm_module() is first
    current_uri[0] = "private_gemm_b"
    second = gemm._load_sm90_push_fp8_moe_gemm_module()

    assert second is not first
    assert built == ["private_gemm_a", "private_gemm_b"]
    gemm._load_sm90_push_fp8_moe_gemm_module_cached.cache_clear()


def test_sm90_push_snapshot_replaces_from_unique_sibling_files(tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import jit

    target = tmp_path / "snapshot.cu"
    real_replace = jit.os.replace
    replacements = []

    def track_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent == target.parent
        assert source_path != destination_path
        assert source_path.read_text(encoding="utf-8") in {"first\n", "second\n"}
        replacements.append(source_path)
        real_replace(source_path, destination_path)

    monkeypatch.setattr(jit.os, "replace", track_replace)

    jit._write_snapshot_atomic(target, "first\n")
    jit._write_snapshot_atomic(target, "second\n")

    assert target.read_text(encoding="utf-8") == "second\n"
    assert len(replacements) == len(set(replacements)) == 2
    assert list(tmp_path.glob(".snapshot.cu.*.tmp")) == []


@pytest.mark.parametrize("matching_target", [True, False])
def test_sm90_push_snapshot_permission_error_requires_matching_target(
    tmp_path, monkeypatch, matching_target
):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import jit

    target = tmp_path / "snapshot.cuh"
    target.write_text("old\n")
    expected = "new\n"

    def fail_replace(_source, destination):
        if matching_target:
            with Path(destination).open("w", encoding="utf-8", newline="") as current:
                current.write(expected)
        raise PermissionError("injected replace failure")

    monkeypatch.setattr(jit.os, "replace", fail_replace)

    if matching_target:
        jit._write_snapshot_atomic(target, expected)
        assert target.read_text(encoding="utf-8") == expected
    else:
        with pytest.raises(PermissionError, match="injected replace failure"):
            jit._write_snapshot_atomic(target, expected)
        assert target.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob(".snapshot.cuh.*.tmp")) == []


def test_sm90_push_uri_changes_with_cuda_flags():
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import sm90_push_a2a_uri

    assert sm90_push_a2a_uri(("-arch=sm_90a",)) != sm90_push_a2a_uri(
        ("-arch=sm_90a", "-lineinfo")
    )


def test_sm90_push_uri_canonicalizes_crlf_and_lf_sources(tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import jit

    lf_dir = tmp_path / "lf"
    crlf_dir = tmp_path / "crlf"
    lf_dir.mkdir()
    crlf_dir.mkdir()
    content = b"first line\nsecond line\n"
    for name in jit._SOURCE_NAMES:
        (lf_dir / name).write_bytes(content)
        (crlf_dir / name).write_bytes(content.replace(b"\n", b"\r\n"))

    monkeypatch.setattr(jit, "_SOURCE_DIR", lf_dir)
    lf_uri = jit.sm90_push_a2a_uri(("-arch=sm_90a",))
    monkeypatch.setattr(jit, "_SOURCE_DIR", crlf_dir)
    crlf_uri = jit.sm90_push_a2a_uri(("-arch=sm_90a",))

    assert crlf_uri == lf_uri


def test_sm90_push_cuda_source_uses_package_local_header():
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import jit

    source = (jit._SOURCE_DIR / "sm90_push_a2a_ops.cu").read_text(encoding="utf-8")
    assert '#include "sm90_push_a2a.cuh"' in source
    assert "flashinfer/fused_moe/sm90_push_a2a.cuh" not in source


def test_sm90_push_weights_do_not_depend_on_trace_templates():
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe.shim import weights

    source = Path(weights.__file__).read_text(encoding="utf-8")
    assert "trace.templates" not in source
    assert "flashinfer_api" not in source
