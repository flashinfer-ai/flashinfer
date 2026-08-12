"""Packaging and source-snapshot checks for the SM90 push kernels."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
import zipfile
from importlib import resources
from importlib.util import resolve_name
from pathlib import Path
from types import SimpleNamespace

import pytest


_PACKAGE_NAME = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
_BACKEND_PACKAGE_NAME = (
    "flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_push_cuda"
)
_NVFP4_BACKEND_PACKAGE_NAME = (
    "flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_PATH = "flashinfer/moe_ep/kernel_src/sm90/push_style_megamoe"
_CUDA_RESOURCES = (
    "src/a2a/sm90_push_a2a_ops.cu",
    "src/a2a/sm90_push_a2a.cuh",
    "src/fp8_gemm/fp8_moe_binding.cu",
    "src/fp8_gemm/fp8_moe_fc1_fused.cuh",
    "src/fp8_gemm/fp8_moe_jit.cuh",
    "src/fp8_gemm/fp8_moe_launcher.cuh",
    "src/fp8_gemm/fp8_moe_scheduler.cuh",
    "src/nvfp4_w4a8_gemm/binding.cu",
    "src/nvfp4_w4a8_gemm/decode.cuh",
    "src/nvfp4_w4a8_gemm/kernel.cuh",
    "src/nvfp4_w4a8_gemm/kernel_inst_m64_n64.cu",
    "src/nvfp4_w4a8_gemm/kernel_inst_m64_n128.cu",
    "src/nvfp4_w4a8_gemm/kernel_inst_m128_n64.cu",
    "src/nvfp4_w4a8_gemm/kernel_inst_m128_n128.cu",
    "src/nvfp4_w4a8_gemm/kernel_instantiation.cuh",
    "src/nvfp4_w4a8_gemm/kernel_launchers.cuh",
    "src/nvfp4_w4a8_gemm/scheduler.cuh",
    "src/nvfp4_rs_gemm/sm90_nvfp4_rs_binding.cu",
    "src/nvfp4_rs_gemm/decode.cuh",
    "src/nvfp4_rs_gemm/scheduler.cuh",
    "src/nvfp4_rs_gemm/sm90_nvfp4_rs_kernel.cuh",
)
_PYTHON_RESOURCES = (
    "__init__.py",
    "shim/__init__.py",
    "shim/gemm.py",
    "shim/jit.py",
    "shim/protocol.py",
    "shim/runner.py",
    "shim/weights.py",
    "shim/nvfp4_runner.py",
    "shim/nvfp4_rs_gemm.py",
    "shim/nvfp4_w4a8_gemm.py",
    "shim/nvfp4_weights.py",
)
_DOCUMENT_RESOURCES = ("ACKNOWLEDGEMENT.md",)
_NVFP4_BACKEND_PYTHON_RESOURCES = (
    "__init__.py",
    "backend.py",
    "config.py",
    "staging.py",
    "weights.py",
)


def _resource_at(package_root, relative_path: str):
    resource = package_root
    for part in relative_path.split("/"):
        resource = resource.joinpath(part)
    return resource


def _python_resources(resource_root):
    for resource in resource_root.iterdir():
        if resource.is_dir():
            yield from _python_resources(resource)
        elif resource.name.endswith(".py") and resource.is_file():
            yield resource


def _backend_sources(package_name: str = _BACKEND_PACKAGE_NAME):
    source_tree = _PROJECT_ROOT.joinpath(*package_name.split("."))
    if source_tree.is_dir():
        return sorted(source_tree.rglob("*.py"))

    package_root = resources.files(package_name)
    return sorted(_python_resources(package_root), key=str)


def test_sm90_push_package_data_contains_cuda_sources():
    pyproject_path = _PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.is_file():
        pytest.skip("pyproject.toml is only available in source-tree test runs")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    key = '"flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe" = ['
    package_block = pyproject.split(key, maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert '"*.md"' in package_block
    assert '"src/a2a/*.cu"' in package_block
    assert '"src/a2a/*.cuh"' in package_block
    assert '"src/fp8_gemm/*.cu"' in package_block
    assert '"src/fp8_gemm/*.cuh"' in package_block
    assert '"src/nvfp4_w4a8_gemm/*.cu"' in package_block
    assert '"src/nvfp4_w4a8_gemm/*.cuh"' in package_block
    assert '"src/nvfp4_rs_gemm/*.cu"' in package_block
    assert '"src/nvfp4_rs_gemm/*.cuh"' in package_block


def test_sm90_push_runtime_resources_expose_packaged_cuda_sources():
    package_root = resources.files(_PACKAGE_NAME)
    for relative_path in (
        *_CUDA_RESOURCES,
        *_PYTHON_RESOURCES,
        *_DOCUMENT_RESOURCES,
    ):
        resource = _resource_at(package_root, relative_path)
        assert resource.is_file()
        assert resource.read_text(encoding="utf-8").strip()


def test_sm90_push_nvfp4_backend_python_resources_are_packaged():
    package_root = resources.files(
        "flashinfer.moe_ep.backends.mega.kernel.sm90_push_nvfp4"
    )
    for relative_path in _NVFP4_BACKEND_PYTHON_RESOURCES:
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
    backend_path = "flashinfer/moe_ep/backends/mega/kernel/sm90_push_nvfp4"
    required.update(
        f"{backend_path}/{relative_path}"
        for relative_path in _NVFP4_BACKEND_PYTHON_RESOURCES
    )
    missing = sorted(required - members)
    assert not missing, "prebuilt wheel is missing SM90 push files:\n" + "\n".join(
        missing
    )


def test_sm90_push_backend_imports_kernel_package_through_public_boundaries():
    package_marker = "kernel_src.sm90.push_style_megamoe"

    for package_name in (_BACKEND_PACKAGE_NAME, _NVFP4_BACKEND_PACKAGE_NAME):
        sources = _backend_sources(package_name)
        assert sources, f"no Python modules found for {package_name}"
        for path in sources:
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
def test_sm90_push_nvfp4_shim_imports_resolve_to_top_level_packages():
    package_root = resources.files(_PACKAGE_NAME)
    package = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim"

    for filename in (
        "nvfp4_rs_gemm.py",
        "nvfp4_runner.py",
        "nvfp4_w4a8_gemm.py",
        "nvfp4_weights.py",
    ):
        path = _resource_at(package_root, f"shim/{filename}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module is None:
                continue
            root = node.module.split(".", maxsplit=1)[0]
            if root not in {"fused_moe", "jit"}:
                continue
            relative_name = "." * node.level + node.module
            assert resolve_name(relative_name, package) == f"flashinfer.{node.module}"


@pytest.mark.parametrize(
    "backend",
    ("sm90.fp8_fp8_bf16_push_cuda", "sm90_push_nvfp4"),
)
def test_sm90_push_weight_helpers_defer_kernel_package_import(backend):
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_JIT"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            f"import flashinfer.moe_ep.backends.mega.kernel.{backend}.weights; "
            "assert 'flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe' "
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
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import jit

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
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import gemm

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
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import gemm

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


def test_sm90_push_private_gemm_cubin_digest_covers_dependencies():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import gemm

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


def test_sm90_push_private_gemm_module_cache_is_keyed_by_uri(monkeypatch, request):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import gemm

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
    request.addfinalizer(gemm._load_sm90_push_fp8_moe_gemm_module_cached.cache_clear)

    first = gemm._load_sm90_push_fp8_moe_gemm_module()
    assert gemm._load_sm90_push_fp8_moe_gemm_module() is first
    current_uri[0] = "private_gemm_b"
    second = gemm._load_sm90_push_fp8_moe_gemm_module()

    assert second is not first
    assert built == ["private_gemm_a", "private_gemm_b"]


def test_sm90_push_snapshot_replaces_from_unique_sibling_files(tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import jit

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
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import jit

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
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import sm90_push_a2a_uri

    assert sm90_push_a2a_uri(("-arch=sm_90a",)) != sm90_push_a2a_uri(
        ("-arch=sm_90a", "-lineinfo")
    )


def test_sm90_push_uri_canonicalizes_crlf_and_lf_sources(tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import jit

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
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import jit

    source = (jit._SOURCE_DIR / "sm90_push_a2a_ops.cu").read_text(encoding="utf-8")
    assert '#include "sm90_push_a2a.cuh"' in source
    assert "flashinfer/fused_moe/sm90_push_a2a.cuh" not in source


def test_sm90_push_weights_do_not_depend_on_trace_templates():
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import weights

    source = Path(weights.__file__).read_text(encoding="utf-8")
    assert "trace.templates" not in source
    assert "flashinfer_api" not in source


@pytest.mark.parametrize("module_name", ["nvfp4_w4a8_gemm", "nvfp4_rs_gemm"])
def test_sm90_push_nvfp4_gemm_requires_cuda_12_0(module_name, tmp_path, monkeypatch):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import shim

    module = getattr(shim, module_name)
    monkeypatch.setattr(module, "is_cuda_version_at_least", lambda _version: False)
    monkeypatch.setattr(module.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    generator = (
        module.gen_sm90_push_nvfp4_w4a8_gemm_module
        if module_name == "nvfp4_w4a8_gemm"
        else module.gen_sm90_push_nvfp4_rs_gemm_module
    )

    with pytest.raises(RuntimeError, match=r"requires CUDA 12\.0"):
        generator()

    assert not any(tmp_path.iterdir())


def test_sm90_push_nvfp4_launchers_use_direct_runtime_launches():
    project_root = Path(__file__).resolve().parents[2]
    source_root = (
        project_root
        / "flashinfer"
        / "moe_ep"
        / "kernel_src"
        / "sm90"
        / "push_style_megamoe"
        / "src"
    )
    for directory in ("nvfp4_w4a8_gemm", "nvfp4_rs_gemm"):
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((source_root / directory).iterdir())
            if path.suffix in (".cu", ".cuh")
        )
        assert "<<<" in source
        assert "cudaKernel_t" not in source
        assert "cudaLaunchKernelEx" not in source


@pytest.mark.parametrize("module_name", ["nvfp4_w4a8_gemm", "nvfp4_rs_gemm"])
def test_sm90_push_nvfp4_uri_covers_sources_dependencies_and_cuda_flags(
    module_name, monkeypatch
):
    from dataclasses import replace

    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import shim

    module = getattr(shim, module_name)
    snapshot = module._capture_source_snapshot()
    if module_name == "nvfp4_w4a8_gemm":
        digest = lambda value: module._source_digest(value)
        original_flags = module._cuda_flags()
    else:
        knobs = module._experiment_knobs()
        arguments = ("rs_wgmma", 64, 3, 64, knobs)
        digest = lambda value: module._source_digest(*arguments, snapshot=value)
        original_flags = module._cuda_flags(*arguments)

    source_digest = digest(snapshot)
    assert digest(replace(snapshot, layout_cuh=snapshot.layout_cuh + b"\nchanged")) != (
        source_digest
    )
    monkeypatch.setattr(
        module,
        "_cuda_flags",
        lambda *_args: original_flags + ("-lineinfo",),
    )
    assert digest(snapshot) != source_digest


@pytest.mark.parametrize("module_name", ["nvfp4_w4a8_gemm", "nvfp4_rs_gemm"])
def test_sm90_push_nvfp4_uri_canonicalizes_crlf_sources(
    module_name, tmp_path, monkeypatch
):
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import shim

    module = getattr(shim, module_name)
    lf_dir = tmp_path / "lf"
    crlf_dir = tmp_path / "crlf"
    lf_dir.mkdir()
    crlf_dir.mkdir()
    content = b"first line\nsecond line\n"
    for name in module._SOURCE_NAMES:
        (lf_dir / name).write_bytes(content)
        (crlf_dir / name).write_bytes(content.replace(b"\n", b"\r\n"))

    monkeypatch.setattr(module, "_source_directory", lambda: lf_dir)
    if module_name == "nvfp4_w4a8_gemm":
        lf_uri = module.get_sm90_push_nvfp4_w4a8_gemm_uri()
    else:
        lf_uri = module.get_sm90_push_nvfp4_rs_gemm_uri()
    monkeypatch.setattr(module, "_source_directory", lambda: crlf_dir)
    if module_name == "nvfp4_w4a8_gemm":
        crlf_uri = module.get_sm90_push_nvfp4_w4a8_gemm_uri()
    else:
        crlf_uri = module.get_sm90_push_nvfp4_rs_gemm_uri()

    assert crlf_uri == lf_uri
