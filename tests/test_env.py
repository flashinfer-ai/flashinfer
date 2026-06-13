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

    saved = {k: sys.modules.get(k) for k in stubs}
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
