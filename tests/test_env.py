"""Regression tests for _get_cubin_dir() priority — issue #2976.

env.py imports CompilationContext (CUDA deps), so we load it in isolation
with lightweight stubs to keep tests runnable without a GPU.

    python -m pytest tests/test_env.py -v --noconftest
"""

import importlib.util
import pathlib
import sys
import types

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_env_module():
    """Load flashinfer.jit.env with minimal stubs (no CUDA required)."""
    stubs = {
        "flashinfer": types.ModuleType("flashinfer"),
        "flashinfer.jit": types.ModuleType("flashinfer.jit"),
        "flashinfer.version": types.ModuleType("flashinfer.version"),
        "flashinfer.compilation_context": types.ModuleType(
            "flashinfer.compilation_context"
        ),
    }
    stubs["flashinfer"].__path__ = [str(_REPO_ROOT / "flashinfer")]
    stubs["flashinfer.jit"].__path__ = [str(_REPO_ROOT / "flashinfer" / "jit")]
    stubs["flashinfer.version"].__version__ = "0.0.0+test"
    stubs["flashinfer.version"].__git_version__ = "test"

    class _Stub:
        def __init__(self):
            self.TARGET_CUDA_ARCHS = set()

    stubs["flashinfer.compilation_context"].CompilationContext = _Stub

    saved = {k: sys.modules.get(k) for k in (*stubs, "flashinfer.jit.env")}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "flashinfer.jit.env", str(_REPO_ROOT / "flashinfer" / "jit" / "env.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["flashinfer.jit.env"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


_env = _load_env_module()


def _fake_cubin_pkg(path):
    """Return a stub ``flashinfer_cubin`` module pointing at *path*."""
    m = types.ModuleType("flashinfer_cubin")
    m.__version__ = "0.0.0+test"
    m.get_cubin_dir = lambda: path
    return m


def _create_aot_module(root, module_name):
    module_dir = root / module_name
    module_dir.mkdir(parents=True)
    module_path = module_dir / f"{module_name}.so"
    module_path.touch()
    return module_path


# -- priority tests (regression for #2976) ----------------------------------


def test_env_var_overrides_package(monkeypatch, tmp_path):
    """FLASHINFER_CUBIN_DIR must take priority over the installed package."""
    env_dir = str(tmp_path / "env_cubins")
    pkg_dir = str(tmp_path / "pkg_cubins")
    monkeypatch.setenv("FLASHINFER_CUBIN_DIR", env_dir)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr(_env, "has_flashinfer_cubin", lambda: True)
    monkeypatch.setitem(sys.modules, "flashinfer_cubin", _fake_cubin_pkg(pkg_dir))
    assert _env._get_cubin_dir() == pathlib.Path(env_dir)


def test_package_used_when_no_env_var(monkeypatch, tmp_path):
    """Without the env var, the package path should be returned."""
    pkg_dir = str(tmp_path / "pkg_cubins")
    monkeypatch.delenv("FLASHINFER_CUBIN_DIR", raising=False)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    monkeypatch.setattr(_env, "has_flashinfer_cubin", lambda: True)
    monkeypatch.setitem(sys.modules, "flashinfer_cubin", _fake_cubin_pkg(pkg_dir))
    assert _env._get_cubin_dir() == pathlib.Path(pkg_dir)


def test_env_var_used_when_no_package(monkeypatch, tmp_path):
    """Env var should work even when the package is not installed."""
    env_dir = str(tmp_path / "env_cubins")
    monkeypatch.setenv("FLASHINFER_CUBIN_DIR", env_dir)
    monkeypatch.setattr(_env, "has_flashinfer_cubin", lambda: False)
    assert _env._get_cubin_dir() == pathlib.Path(env_dir)


def test_default_when_nothing_set(monkeypatch):
    """Fall back to the default cache directory."""
    monkeypatch.delenv("FLASHINFER_CUBIN_DIR", raising=False)
    monkeypatch.setattr(_env, "has_flashinfer_cubin", lambda: False)
    assert _env._get_cubin_dir() == _env.FLASHINFER_CACHE_DIR / "cubins"


def test_aot_artifacts_select_each_heterogeneous_target(monkeypatch, tmp_path):
    fallback_root = tmp_path / "package-aot"
    sm103_root = tmp_path / "sm103a"
    sm120_root = tmp_path / "sm120f"
    sm103_path = _create_aot_module(sm103_root, "attention_module")
    sm120_path = _create_aot_module(sm120_root, "attention_module")
    providers = (
        _env.AOTProvider(
            provider_id="sm103a",
            distribution="flashinfer-jit-cache-sm103a",
            version="0.6.16+cu130",
            jit_cache_dir=sm103_root,
            cuda_architectures=frozenset({"sm103a"}),
            modules=frozenset({"attention_module"}),
        ),
        _env.AOTProvider(
            provider_id="sm120f",
            distribution="flashinfer-jit-cache-sm120f",
            version="0.6.16+cu130",
            jit_cache_dir=sm120_root,
            cuda_architectures=frozenset({"sm120f"}),
            modules=frozenset({"attention_module"}),
        ),
    )
    monkeypatch.setattr(_env, "FLASHINFER_AOT_DIR", fallback_root)
    monkeypatch.setattr(_env, "FLASHINFER_AOT_PROVIDERS", providers)
    monkeypatch.setattr(
        _env,
        "_target_cuda_architectures",
        lambda: frozenset({"sm103a", "sm120f"}),
    )

    artifacts = _env.get_aot_artifacts("attention_module")

    assert [(artifact.provider_id, artifact.path) for artifact in artifacts] == [
        ("sm103a", sm103_path),
        ("sm120f", sm120_path),
    ]


def test_cuda_architectures_for_call_collects_nested_tensor_devices(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(current_device=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(
        _env,
        "_cuda_architecture_for_device",
        lambda device_index: {0: "sm103a", 1: "sm120f"}[device_index],
    )

    tensor0 = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=0))
    tensor1 = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=1))

    assert _env._cuda_architectures_for_call(
        (tensor0,), {"nested": [tensor1]}
    ) == frozenset({"sm103a", "sm120f"})
