"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# NOTE(lequn): Do not "from .jit.env import xxx".
# Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
# This helps AOT script to override envs.

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import FrozenSet, Tuple

from ..compilation_context import CompilationContext
from ..version import __version__ as flashinfer_version

# NOTE: use stdlib logging (namespaced under flashinfer.jit) instead of the
# FlashInferJITLogger from core.py -- core.py imports this module, so importing
# it here would create a circular import. A plain warning is fine because
# _get_cubin_dir() only runs once, at module import.
logger = logging.getLogger("flashinfer.jit")


def has_flashinfer_jit_cache() -> bool:
    """
    Check if flashinfer_jit_cache module is available.

    Returns:
        True if flashinfer_jit_cache exists, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("flashinfer_jit_cache") is not None


def has_flashinfer_cubin() -> bool:
    """
    Check if flashinfer_cubin module is available.

    Returns:
        True if flashinfer_cubin exists, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("flashinfer_cubin") is not None


FLASHINFER_BASE_DIR: pathlib.Path = pathlib.Path(
    os.getenv("FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

FLASHINFER_CACHE_DIR: pathlib.Path = FLASHINFER_BASE_DIR / ".cache" / "flashinfer"
_package_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]


def _get_cubin_dir():
    """
    Get the cubin directory path with the following priority:
    1. Environment variable FLASHINFER_CUBIN_DIR
    2. flashinfer-cubin package if installed
    3. Default cache directory
    """
    # First check environment variable
    env_dir = os.getenv("FLASHINFER_CUBIN_DIR")
    if env_dir:
        if has_flashinfer_cubin():
            logger.warning(
                "FLASHINFER_CUBIN_DIR=%s overrides the installed flashinfer-cubin "
                "package; cubins will be read from that path instead of the package.",
                env_dir,
            )
        return pathlib.Path(env_dir)

    # Then check if flashinfer-cubin package is installed
    if has_flashinfer_cubin():
        import flashinfer_cubin

        flashinfer_cubin_version = flashinfer_cubin.__version__
        # Allow bypassing version check with environment variable
        # NOTE(yiyang): skip version check for editable/source installs where
        # flashinfer_version falls back to "0.0.0+unknown" (no _build_meta.py).
        if (
            not os.getenv("FLASHINFER_DISABLE_VERSION_CHECK")
            and flashinfer_version != "0.0.0+unknown"
            and flashinfer_version != flashinfer_cubin_version
        ):
            raise RuntimeError(
                f"flashinfer-cubin version ({flashinfer_cubin_version}) does not match "
                f"flashinfer version ({flashinfer_version}). "
                "Please install the same version of both packages. "
                "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
            )

        return pathlib.Path(flashinfer_cubin.get_cubin_dir())

    # Fall back to default cache directory
    return FLASHINFER_CACHE_DIR / "cubins"


FLASHINFER_CUBIN_DIR: pathlib.Path = _get_cubin_dir()


@dataclass(frozen=True)
class AOTProvider:
    """Installed package that owns a set of architecture-specific AOT modules."""

    provider_id: str
    distribution: str
    version: str
    jit_cache_dir: pathlib.Path
    cuda_architectures: FrozenSet[str]
    modules: FrozenSet[str]


def _check_jit_cache_version(distribution: str, package_version: str) -> None:
    # NOTE(Zihao): jit-cache versions contain a CUDA local-version suffix,
    # for example 0.3.1+cu129, so compare the FlashInfer version prefix.
    if (
        not os.getenv("FLASHINFER_DISABLE_VERSION_CHECK")
        and flashinfer_version != "0.0.0+unknown"
        and not package_version.startswith(flashinfer_version)
    ):
        raise RuntimeError(
            f"{distribution} version ({package_version}) does not match "
            f"flashinfer version ({flashinfer_version}). "
            "Please install the same version of both packages. "
            "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
        )


def _get_aot_locations() -> Tuple[pathlib.Path, Tuple[AOTProvider, ...]]:
    """
    Get the legacy AOT directory and any installed binary provider packages.

    ``flashinfer-jit-cache`` historically owned one directory containing every
    module. Newer shim builds discover separately installable providers while
    retaining that directory as a compatibility fallback.
    """
    if has_flashinfer_jit_cache():
        import flashinfer_jit_cache

        flashinfer_jit_cache_version = flashinfer_jit_cache.__version__
        _check_jit_cache_version("flashinfer-jit-cache", flashinfer_jit_cache_version)

        providers = []
        get_providers = getattr(flashinfer_jit_cache, "get_jit_cache_providers", None)
        if get_providers is not None:
            for provider in get_providers():
                _check_jit_cache_version(provider.distribution, provider.version)
                providers.append(
                    AOTProvider(
                        provider_id=provider.provider_id,
                        distribution=provider.distribution,
                        version=provider.version,
                        jit_cache_dir=pathlib.Path(provider.jit_cache_dir),
                        cuda_architectures=frozenset(provider.cuda_architectures),
                        modules=frozenset(provider.modules),
                    )
                )

        return (
            pathlib.Path(flashinfer_jit_cache.get_jit_cache_dir()),
            tuple(providers),
        )

    return _package_root / "data" / "aot", ()


FLASHINFER_AOT_DIR, FLASHINFER_AOT_PROVIDERS = _get_aot_locations()
FLASHINFER_AOT_DIRS: Tuple[pathlib.Path, ...] = (FLASHINFER_AOT_DIR,) + tuple(
    provider.jit_cache_dir for provider in FLASHINFER_AOT_PROVIDERS
)


def _target_cuda_architectures() -> FrozenSet[str]:
    compilation_context = CompilationContext()
    return frozenset(
        f"sm{major}{minor}" for major, minor in compilation_context.TARGET_CUDA_ARCHS
    )


def get_aot_path(module_name: str) -> pathlib.Path:
    """Resolve an AOT module from the legacy wheel or a compatible provider."""
    legacy_path = FLASHINFER_AOT_DIR / module_name / f"{module_name}.so"
    if legacy_path.exists():
        return legacy_path

    target_architectures = _target_cuda_architectures()
    if not target_architectures:
        return legacy_path
    for provider in FLASHINFER_AOT_PROVIDERS:
        if module_name not in provider.modules:
            continue
        if not target_architectures.issubset(provider.cuda_architectures):
            continue
        provider_path = provider.jit_cache_dir / module_name / f"{module_name}.so"
        if provider_path.exists():
            return provider_path

    # JitSpec uses this stable path for existence checks and diagnostics when
    # no compatible prebuilt module is installed.
    return legacy_path


def _get_workspace_dir_name() -> pathlib.Path:
    compilation_context = CompilationContext()
    # NOTE(Zihao): sorted() is crucial here to ensure deterministic directory names.
    # Without it, the same set of CUDA archs could generate different directory names
    # across runs (e.g., "75_80_89" vs "89_75_80"), causing cache fragmentation.
    arch = "_".join(
        f"{major}{minor}"
        for major, minor in sorted(compilation_context.TARGET_CUDA_ARCHS)
    )
    return FLASHINFER_CACHE_DIR / flashinfer_version / arch


# use pathlib
FLASHINFER_WORKSPACE_DIR: pathlib.Path = _get_workspace_dir_name()
FLASHINFER_JIT_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR: pathlib.Path = FLASHINFER_WORKSPACE_DIR / "generated"
FLASHINFER_DATA: pathlib.Path = _package_root / "data"
FLASHINFER_INCLUDE_DIR: pathlib.Path = _package_root / "data" / "include"
FLASHINFER_CSRC_DIR: pathlib.Path = _package_root / "data" / "csrc"
# FLASHINFER_SRC_DIR = _package_root / "data" / "src"
CUTLASS_INCLUDE_DIRS: list[pathlib.Path] = [
    _package_root / "data" / "cutlass" / "include",
    _package_root / "data" / "cutlass" / "tools" / "util" / "include",
]
SPDLOG_INCLUDE_DIR: pathlib.Path = _package_root / "data" / "spdlog" / "include"
CCCL_INCLUDE_DIRS: list[pathlib.Path] = [
    _package_root / "data" / "cccl" / "cub",
    _package_root / "data" / "cccl" / "libcudacxx" / "include",
    _package_root / "data" / "cccl" / "thrust",
]
