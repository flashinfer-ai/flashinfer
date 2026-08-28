import importlib.util
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from flashinfer.jit import env as jit_env


class _FakeEntryPoint:
    def __init__(self, name, manifest):
        self.name = name
        self._manifest = manifest

    def load(self):
        return lambda: self._manifest


class _FakeEntryPoints(tuple):
    def select(self, *, group):
        if group == "flashinfer.jit_cache.providers":
            return self
        return ()


@pytest.fixture
def jit_cache_shim_module():
    package_dir = (
        Path(__file__).resolve().parents[2]
        / "flashinfer-jit-cache"
        / "flashinfer_jit_cache"
    )
    module_name = "_test_flashinfer_jit_cache"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture
def provider_package_config_module():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "flashinfer-jit-cache-provider"
        / "package_config.py"
    )
    module_name = "_test_jit_cache_provider_package_config"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture
def wheelhouse_verifier_module():
    verifier_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "verify_jit_cache_provider_wheelhouse.py"
    )
    module_name = "_test_jit_cache_wheelhouse_verifier"
    spec = importlib.util.spec_from_file_location(module_name, verifier_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _create_aot_module(root: Path, module_name: str) -> Path:
    module_dir = root / module_name
    module_dir.mkdir(parents=True)
    module_path = module_dir / f"{module_name}.so"
    module_path.touch()
    return module_path


def test_shim_discovers_and_normalizes_provider(
    monkeypatch, tmp_path, jit_cache_shim_module
):
    manifest = {
        "schema_version": 1,
        "distribution": "flashinfer-jit-cache-sm90a",
        "version": "0.6.16+cu130",
        "jit_cache_dir": str(tmp_path),
        "cuda_architectures": ["9.0a"],
        "modules": ["attention_module"],
    }
    entry_points = _FakeEntryPoints((_FakeEntryPoint("sm90a", manifest),))
    monkeypatch.setattr(
        jit_cache_shim_module.importlib.metadata,
        "entry_points",
        lambda: entry_points,
    )

    providers = jit_cache_shim_module.get_jit_cache_providers()

    assert len(providers) == 1
    assert providers[0].provider_id == "sm90a"
    assert providers[0].cuda_architectures == frozenset({"sm90a"})
    assert providers[0].modules == frozenset({"attention_module"})


def test_shim_ignores_invalid_provider(monkeypatch, caplog, jit_cache_shim_module):
    manifest = {
        "schema_version": 1,
        "distribution": "flashinfer-jit-cache-sm80",
        "version": "0.6.16+cu130",
        "jit_cache_dir": "/tmp/provider",
        "cuda_architectures": ["8.0"],
        "modules": [],
    }
    entry_points = _FakeEntryPoints((_FakeEntryPoint("sm80", manifest),))
    monkeypatch.setattr(
        jit_cache_shim_module.importlib.metadata,
        "entry_points",
        lambda: entry_points,
    )

    assert jit_cache_shim_module.get_jit_cache_providers() == ()
    assert "has no modules" in caplog.text


@pytest.mark.parametrize(
    ("architecture", "expected_architecture", "expected_tag"),
    [
        ("8.0", "8.0", "sm80"),
        ("sm90a", "9.0a", "sm90a"),
        ("10.7a", "10.7a", "sm107a"),
        ("compute_120f", "12.0f", "sm120f"),
        ("12.1a", "12.1a", "sm121a"),
    ],
)
def test_provider_build_config_normalizes_architecture(
    provider_package_config_module,
    architecture,
    expected_architecture,
    expected_tag,
):
    assert provider_package_config_module.normalize_cuda_architecture(architecture) == (
        expected_architecture,
        expected_tag,
    )


def test_provider_build_config_uses_distinct_distribution_name(
    monkeypatch, provider_package_config_module
):
    monkeypatch.setenv("FLASHINFER_JIT_CACHE_PROVIDER_ARCH", "9.0a")
    monkeypatch.setenv("FLASHINFER_LOCAL_VERSION", "cu130")

    config = provider_package_config_module.get_provider_build_config()

    assert config.provider_tag == "sm90a"
    assert config.cuda_major == "13"
    assert config.distribution == "flashinfer-jit-cache-sm90a"
    assert config.package == "flashinfer_jit_cache.providers.sm90a"
    assert config.version.endswith("+cu130")


def test_provider_backend_uses_configured_build_dependencies(monkeypatch, tmp_path):
    backend_path = (
        Path(__file__).resolve().parents[2]
        / "flashinfer-jit-cache-provider"
        / "build_backend.py"
    )
    provider_source = tmp_path / "flashinfer_jit_cache_provider"
    provider_source.mkdir()
    config = SimpleNamespace(
        cuda_architecture="10.7a",
        cuda_major="13",
        provider_tag="sm107a",
        distribution="flashinfer-jit-cache-sm107a",
        package="flashinfer_jit_cache.providers.sm107a",
        version="0.6.18+cu134",
    )

    package_config = ModuleType("package_config")
    package_config.PROJECT_ROOT = tmp_path
    package_config.PROVIDER_SOURCE_DIR = provider_source
    package_config.get_provider_build_config = lambda: config

    build_utils = ModuleType("build_utils")
    build_utils.get_git_version = lambda cwd=None: "deadbeef"
    build_utils.get_build_dependency_requirements = lambda cuda_major: [
        "nvidia-cutlass-dsl[cu13]>=4.6.2a0"
        if cuda_major == "13"
        else "nvidia-cutlass-dsl>=4.6.2a0"
    ]

    monkeypatch.setitem(sys.modules, "package_config", package_config)
    monkeypatch.setitem(sys.modules, "build_utils", build_utils)
    monkeypatch.setattr(sys, "path", list(sys.path))

    module_name = "_test_jit_cache_provider_build_backend"
    spec = importlib.util.spec_from_file_location(module_name, backend_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        monkeypatch.setattr(
            module._orig,
            "get_requires_for_build_wheel",
            lambda _settings: ["setuptools>=77"],
        )
        monkeypatch.setattr(
            module._orig,
            "get_requires_for_build_editable",
            lambda _settings: ["setuptools>=77"],
        )

        expected = ["setuptools>=77", "nvidia-cutlass-dsl[cu13]>=4.6.2a0"]
        assert module.get_requires_for_build_wheel(None) == expected
        assert module.get_requires_for_build_editable(None) == expected

        monkeypatch.delenv(module.PROVIDER_PLATFORM_TAG_ENV, raising=False)
        assert module._provider_platform_tag("linux_x86_64") == "linux_x86_64"

        monkeypatch.setattr(module.platform, "system", lambda: "Linux")
        monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
        monkeypatch.setattr(module.platform, "libc_ver", lambda: ("glibc", "2.28"))
        monkeypatch.setenv(module.PROVIDER_PLATFORM_TAG_ENV, "manylinux_2_28_x86_64")
        assert module._provider_platform_tag("linux_x86_64") == "manylinux_2_28_x86_64"

        monkeypatch.setenv(module.PROVIDER_PLATFORM_TAG_ENV, "manylinux_2_28_aarch64")
        with pytest.raises(RuntimeError, match="Unsupported provider platform tag"):
            module._provider_platform_tag("linux_x86_64")

        monkeypatch.setenv(module.PROVIDER_PLATFORM_TAG_ENV, "manylinux_2_28_x86_64")
        monkeypatch.setattr(module.platform, "libc_ver", lambda: ("glibc", "2.34"))
        with pytest.raises(RuntimeError, match="glibc 2.34 is too new"):
            module._provider_platform_tag("linux_x86_64")
    finally:
        sys.modules.pop(module_name, None)


def test_provider_wheelhouse_resolves_cu134_from_shared_config():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "build_jit_cache_provider_wheelhouse.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "ARCH": "aarch64",
            "FLASHINFER_JIT_CACHE_PROVIDER_ARCH": "12.1a",
            "FLASHINFER_LOCAL_VERSION": "cu134",
        }
    )
    env.pop("CUDA_VERSION", None)
    env.pop("DOCKER_IMAGE", None)
    env.pop("FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG", None)

    result = subprocess.run(
        ["bash", str(script_path), "--print-config"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "provider=sm121a" in result.stdout
    assert "cuda_version=13.4" in result.stdout
    assert "pytorch_index=nightly/cu134" in result.stdout
    assert "container=pytorch/manylinuxaarch64-builder:cuda13.4" in result.stdout
    assert "provider_platform_tag=manylinux_2_28_aarch64" in result.stdout


def test_provider_wheelhouse_custom_container_defaults_to_native_tag():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "build_jit_cache_provider_wheelhouse.sh"
    )
    env = os.environ.copy()
    env.update(
        {
            "ARCH": "x86_64",
            "DOCKER_IMAGE": "example.invalid/custom-builder:latest",
            "FLASHINFER_JIT_CACHE_PROVIDER_ARCH": "10.7a",
            "FLASHINFER_LOCAL_VERSION": "cu134",
        }
    )
    env.pop("CUDA_VERSION", None)
    env.pop("FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG", None)

    result = subprocess.run(
        ["bash", str(script_path), "--print-config"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "container=example.invalid/custom-builder:latest" in result.stdout
    assert "provider_platform_tag=native" in result.stdout


@pytest.mark.parametrize(
    "package_version",
    ("0.6.18", "0.6.18+cu130", "0.6.18+cu134"),
)
def test_jit_cache_version_requires_exact_public_version(monkeypatch, package_version):
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.setattr(jit_env, "flashinfer_version", "0.6.18")

    jit_env._check_jit_cache_version("flashinfer-jit-cache", package_version)


def test_jit_cache_version_rejects_unbounded_prefix(monkeypatch):
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.setattr(jit_env, "flashinfer_version", "0.6.1")

    with pytest.raises(RuntimeError, match="does not match flashinfer version"):
        jit_env._check_jit_cache_version("flashinfer-jit-cache", "0.6.10+cu130")


def test_jit_cache_build_setup_preserves_indexes_and_cleans_constraint(tmp_path):
    common_script = (
        Path(__file__).resolve().parents[2] / "scripts" / "jit_cache_build_common.sh"
    )
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        "  printf '%s\\n' 'torch==2.9.0+cu130'\n"
        "fi\n"
    )
    fake_python.chmod(0o755)
    trap_marker = tmp_path / "trap-ran"
    script = f"""
set -euo pipefail
trap 'touch "{trap_marker}"' EXIT
source "{common_script}"
export PIP_CONSTRAINT=/tmp/original-constraint
export PIP_EXTRA_INDEX_URL=https://mirror.example/simple
setup_jit_cache_python_build "{fake_python}" 13.0 cu130
generated_constraint=${{PIP_CONSTRAINT}}
test -f "${{generated_constraint}}"
test "${{PIP_EXTRA_INDEX_URL}}" = "https://mirror.example/simple https://download.pytorch.org/whl/cu130"
cleanup_jit_cache_python_build
test ! -e "${{generated_constraint}}"
test "${{PIP_CONSTRAINT}}" = /tmp/original-constraint
"""

    subprocess.run(["bash", "-c", script], check=True)

    assert trap_marker.is_file()


def test_native_provider_cuda_inspection_reports_no_ptx(
    monkeypatch, tmp_path, wheelhouse_verifier_module
):
    wheel_path = tmp_path / "provider.whl"
    archive_path = "provider/jit_cache/test_module/test_module.so"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(archive_path, b"test")
    wheel = wheelhouse_verifier_module.Wheel(
        path=wheel_path,
        distribution="flashinfer-jit-cache-sm120f",
        version="0.6.16+cu130",
        requirements=(),
        contents=(archive_path,),
        metadata_path="provider.dist-info/METADATA",
    )
    cuobjdump = tmp_path / "cuobjdump"
    cuobjdump.touch()

    def mock_run(cmd, **_kwargs):
        if cmd[1] == "--list-elf":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="ELF file 1: test.sm120f.cubin\n", stderr=""
            )
        assert cmd[1] == "--list-ptx"
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="",
            stderr="cuobjdump info: No PTX file found to extract\n",
        )

    monkeypatch.setattr(wheelhouse_verifier_module.subprocess, "run", mock_run)

    architectures, ptx_modules = wheelhouse_verifier_module.inspect_cuda_architectures(
        wheel,
        {"test_module": archive_path},
        "sm120f",
        cuobjdump,
        strict=True,
    )

    assert architectures == {"test_module": ["sm120f"]}
    assert ptx_modules == []


def test_native_provider_cuda_inspection_rejects_ptx(
    monkeypatch, tmp_path, wheelhouse_verifier_module
):
    wheel_path = tmp_path / "provider.whl"
    archive_path = "provider/jit_cache/test_module/test_module.so"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr(archive_path, b"test")
    wheel = wheelhouse_verifier_module.Wheel(
        path=wheel_path,
        distribution="flashinfer-jit-cache-sm120f",
        version="0.6.16+cu130",
        requirements=(),
        contents=(archive_path,),
        metadata_path="provider.dist-info/METADATA",
    )
    cuobjdump = tmp_path / "cuobjdump"
    cuobjdump.touch()

    def mock_run(cmd, **_kwargs):
        if cmd[1] == "--list-elf":
            return subprocess.CompletedProcess(
                cmd, 0, stdout="ELF file 1: test.sm120f.cubin\n", stderr=""
            )
        assert cmd[1] == "--list-ptx"
        return subprocess.CompletedProcess(
            cmd, 0, stdout="PTX file 1: test.compute_80.ptx\n", stderr=""
        )

    monkeypatch.setattr(wheelhouse_verifier_module.subprocess, "run", mock_run)

    with pytest.raises(ValueError, match="native-provider modules contain PTX"):
        wheelhouse_verifier_module.inspect_cuda_architectures(
            wheel,
            {"test_module": archive_path},
            "sm120f",
            cuobjdump,
            strict=True,
        )


def test_get_aot_path_selects_provider_for_target_arch(monkeypatch, tmp_path):
    legacy_root = tmp_path / "legacy"
    sm80_root = tmp_path / "sm80"
    sm90_root = tmp_path / "sm90a"
    expected = _create_aot_module(sm90_root, "attention_module")
    _create_aot_module(sm80_root, "attention_module")

    providers = (
        jit_env.AOTProvider(
            provider_id="sm80",
            distribution="flashinfer-jit-cache-sm80",
            version="0.6.16+cu130",
            jit_cache_dir=sm80_root,
            cuda_architectures=frozenset({"sm80"}),
            modules=frozenset({"attention_module"}),
        ),
        jit_env.AOTProvider(
            provider_id="sm90a",
            distribution="flashinfer-jit-cache-sm90a",
            version="0.6.16+cu130",
            jit_cache_dir=sm90_root,
            cuda_architectures=frozenset({"sm90a"}),
            modules=frozenset({"attention_module"}),
        ),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", legacy_root)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", providers)
    monkeypatch.setattr(
        jit_env, "_target_cuda_architectures", lambda: frozenset({"sm90a"})
    )

    assert jit_env.get_aot_path("attention_module") == expected


@pytest.mark.parametrize(
    ("provider_architecture", "target_architecture"),
    [
        ("sm100a", "sm103a"),
        ("sm120f", "sm121a"),
        ("sm80", "sm86"),
    ],
)
def test_get_aot_path_does_not_infer_provider_compatibility(
    monkeypatch,
    tmp_path,
    provider_architecture,
    target_architecture,
):
    legacy_root = tmp_path / "legacy"
    provider_root = tmp_path / provider_architecture
    _create_aot_module(provider_root, "attention_module")
    provider = jit_env.AOTProvider(
        provider_id=provider_architecture,
        distribution=f"flashinfer-jit-cache-{provider_architecture}",
        version="0.6.16+cu130",
        jit_cache_dir=provider_root,
        cuda_architectures=frozenset({provider_architecture}),
        modules=frozenset({"attention_module"}),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", legacy_root)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", (provider,))
    monkeypatch.setattr(
        jit_env,
        "_target_cuda_architectures",
        lambda: frozenset({target_architecture}),
    )

    assert jit_env.get_aot_path("attention_module") == (
        legacy_root / "attention_module" / "attention_module.so"
    )


def test_get_aot_path_requires_provider_to_cover_all_targets(monkeypatch, tmp_path):
    legacy_root = tmp_path / "legacy"
    provider_root = tmp_path / "sm90a"
    _create_aot_module(provider_root, "attention_module")
    provider = jit_env.AOTProvider(
        provider_id="sm90a",
        distribution="flashinfer-jit-cache-sm90a",
        version="0.6.16+cu130",
        jit_cache_dir=provider_root,
        cuda_architectures=frozenset({"sm90a"}),
        modules=frozenset({"attention_module"}),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", legacy_root)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", (provider,))
    monkeypatch.setattr(
        jit_env,
        "_target_cuda_architectures",
        lambda: frozenset({"sm80", "sm90a"}),
    )

    assert jit_env.get_aot_path("attention_module") == (
        legacy_root / "attention_module" / "attention_module.so"
    )


def test_get_aot_path_does_not_guess_when_target_is_unknown(monkeypatch, tmp_path):
    legacy_root = tmp_path / "legacy"
    provider_root = tmp_path / "sm80"
    _create_aot_module(provider_root, "attention_module")
    provider = jit_env.AOTProvider(
        provider_id="sm80",
        distribution="flashinfer-jit-cache-sm80",
        version="0.6.16+cu130",
        jit_cache_dir=provider_root,
        cuda_architectures=frozenset({"sm80"}),
        modules=frozenset({"attention_module"}),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", legacy_root)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", (provider,))
    monkeypatch.setattr(jit_env, "_target_cuda_architectures", lambda: frozenset())

    assert jit_env.get_aot_path("attention_module") == (
        legacy_root / "attention_module" / "attention_module.so"
    )


def test_get_aot_path_prefers_legacy_monolithic_wheel(monkeypatch, tmp_path):
    legacy_root = tmp_path / "legacy"
    provider_root = tmp_path / "sm80"
    expected = _create_aot_module(legacy_root, "attention_module")
    _create_aot_module(provider_root, "attention_module")
    provider = jit_env.AOTProvider(
        provider_id="sm80",
        distribution="flashinfer-jit-cache-sm80",
        version="0.6.16+cu130",
        jit_cache_dir=provider_root,
        cuda_architectures=frozenset({"sm80"}),
        modules=frozenset({"attention_module"}),
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", legacy_root)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", (provider,))
    monkeypatch.setattr(
        jit_env, "_target_cuda_architectures", lambda: frozenset({"sm80"})
    )

    assert jit_env.get_aot_path("attention_module") == expected
