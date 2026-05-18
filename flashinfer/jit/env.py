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

import os
import pathlib
from typing import Optional

from build_utils import (
    SM_FAMILY_ORDER,
    jit_cache_sm_family_covers_capabilities,
    jit_cache_sm_family_for_capabilities,
)

from ..compilation_context import CompilationContext
from ..version import __version__ as flashinfer_version
from packaging.version import InvalidVersion, Version


def _public_package_version(version: str) -> str:
    try:
        return Version(version).public
    except InvalidVersion:
        return version.split("+", 1)[0]


def _sm_family_from_package_version(version: str) -> Optional[str]:
    try:
        local_version = Version(version).local
    except InvalidVersion:
        local_version = version.split("+", 1)[1] if "+" in version else None
    if not local_version:
        return None
    for part in local_version.split("."):
        if part in SM_FAMILY_ORDER:
            return part
    return None


def _visible_cuda_device_capabilities() -> list[tuple[int, int]]:
    try:
        import torch
    except Exception:
        return []

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return []
    return [
        torch.cuda.get_device_capability(device)
        for device in range(torch.cuda.device_count())
    ]


def _validate_jit_cache_sm_family(flashinfer_jit_cache_version: str) -> None:
    installed_family = _sm_family_from_package_version(flashinfer_jit_cache_version)
    if installed_family is None:
        return

    capabilities = _visible_cuda_device_capabilities()
    if not capabilities:
        return

    if not jit_cache_sm_family_covers_capabilities(installed_family, capabilities):
        try:
            expected_family = jit_cache_sm_family_for_capabilities(capabilities)
        except ValueError as e:
            raise RuntimeError(
                "Visible CUDA devices cannot be covered by a single "
                f"flashinfer-jit-cache wheel: {e}"
            ) from e
        capability_text = ", ".join(
            f"sm{major}{minor}" for major, minor in capabilities
        )
        raise RuntimeError(
            f"flashinfer-jit-cache SM family ({installed_family}) does not match "
            f"visible CUDA devices ({capability_text}; expected {expected_family}). "
            "Install the matching flashinfer-jit-cache wheel or set "
            "FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
        )


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
    1. flashinfer-cubin package if installed
    2. Environment variable FLASHINFER_CUBIN_DIR
    3. Default cache directory
    """
    # First check if flashinfer-cubin package is installed
    if has_flashinfer_cubin():
        import flashinfer_cubin

        flashinfer_cubin_version = flashinfer_cubin.__version__
        # Allow bypassing version check with environment variable
        # NOTE(yiyang): skip version check for editable/source installs where
        # flashinfer_version falls back to "0.0.0+unknown" (no _build_meta.py).
        if (
            not os.getenv("FLASHINFER_DISABLE_VERSION_CHECK")
            and flashinfer_version != "0.0.0+unknown"
            and _public_package_version(flashinfer_version)
            != _public_package_version(flashinfer_cubin_version)
        ):
            raise RuntimeError(
                f"flashinfer-cubin version ({flashinfer_cubin_version}) does not match "
                f"flashinfer version ({flashinfer_version}). "
                "Please install the same version of both packages. "
                "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
            )

        return pathlib.Path(flashinfer_cubin.get_cubin_dir())

    # Then check environment variable
    env_dir = os.getenv("FLASHINFER_CUBIN_DIR")
    if env_dir:
        return pathlib.Path(env_dir)

    # Fall back to default cache directory
    return FLASHINFER_CACHE_DIR / "cubins"


FLASHINFER_CUBIN_DIR: pathlib.Path = _get_cubin_dir()


def _get_aot_dir():
    """
    Get the AOT directory path with the following priority:
    1. flashinfer-jit-cache package if installed
    2. Default fallback to _package_root / "data" / "aot"
    """
    # First check if flashinfer-jit-cache package is installed
    if has_flashinfer_jit_cache():
        import flashinfer_jit_cache

        flashinfer_jit_cache_version = flashinfer_jit_cache.__version__
        # NOTE(Zihao): we don't use exact version match here because the version of flashinfer-jit-cache
        # contains the CUDA version suffix: e.g. 0.3.1+cu129.
        # Allow bypassing version check with environment variable
        if not os.getenv("FLASHINFER_DISABLE_VERSION_CHECK"):
            if flashinfer_version != "0.0.0+unknown" and _public_package_version(
                flashinfer_version
            ) != _public_package_version(flashinfer_jit_cache_version):
                raise RuntimeError(
                    f"flashinfer-jit-cache version ({flashinfer_jit_cache_version}) does not match "
                    f"flashinfer version ({flashinfer_version}). "
                    "Please install the same version of both packages. "
                    "Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check."
                )
            _validate_jit_cache_sm_family(flashinfer_jit_cache_version)

        return pathlib.Path(flashinfer_jit_cache.get_jit_cache_dir())

    # Fall back to default directory
    return _package_root / "data" / "aot"


FLASHINFER_AOT_DIR: pathlib.Path = _get_aot_dir()


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
