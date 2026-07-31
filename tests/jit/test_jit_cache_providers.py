import importlib.util
import sys
from pathlib import Path

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
    assert config.distribution == "flashinfer-jit-cache-sm90a"
    assert config.package == "flashinfer_jit_cache.providers.sm90a"
    assert config.version.endswith("+cu130")


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
