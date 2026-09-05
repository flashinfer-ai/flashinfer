"""PEP 517 backend for architecture-specific jit-cache provider wheels."""

import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as _orig
from wheel.bdist_wheel import bdist_wheel

from package_config import (
    PROJECT_ROOT,
    PROVIDER_SOURCE_DIR,
    get_provider_build_config,
)


sys.path.insert(0, str(PROJECT_ROOT))

from build_utils import get_build_dependency_requirements, get_git_version


os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
config = get_provider_build_config()
os.environ["FLASHINFER_CUDA_ARCH_LIST"] = config.cuda_architecture

PROVIDER_PLATFORM_TAG_ENV = "FLASHINFER_JIT_CACHE_PROVIDER_PLATFORM_TAG"


def _write_build_metadata() -> None:
    metadata_path = PROVIDER_SOURCE_DIR / "_build_meta.py"
    metadata_path.write_text(
        '"""Generated jit-cache provider build metadata."""\n'
        f'__version__ = "{config.version}"\n'
        f'__git_version__ = "{get_git_version(cwd=PROJECT_ROOT)}"\n'
    )


_write_build_metadata()


def _ensure_build_inputs() -> None:
    submodule_paths = [
        PROJECT_ROOT / "3rdparty" / "cutlass" / "include",
        PROJECT_ROOT / "3rdparty" / "spdlog" / "include",
        PROJECT_ROOT / "3rdparty" / "cccl" / "cub",
    ]
    if not all(path.exists() for path in submodule_paths):
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            check=False,
        )
        missing = [str(path) for path in submodule_paths if not path.exists()]
        if result.returncode != 0 and missing:
            raise RuntimeError(
                f"git submodule update failed and submodules are missing: {missing}\n"
                f"git stderr: {result.stderr.decode().strip()}"
            )

    import importlib.util

    backend_spec = importlib.util.spec_from_file_location(
        "main_build_backend", PROJECT_ROOT / "build_backend.py"
    )
    if backend_spec is None or backend_spec.loader is None:
        raise RuntimeError("Could not load the FlashInfer build backend")
    main_build_backend = importlib.util.module_from_spec(backend_spec)
    backend_spec.loader.exec_module(main_build_backend)
    main_build_backend._create_data_dir(use_symlinks=True)


def _write_provider_manifest(jit_cache_dir: Path) -> None:
    modules = sorted(
        module_dir.name
        for module_dir in jit_cache_dir.iterdir()
        if module_dir.is_dir() and (module_dir / f"{module_dir.name}.so").is_file()
    )
    if not modules:
        raise RuntimeError("No .so files were generated for the jit-cache provider")

    manifest = {
        "schema_version": 1,
        "provider_id": config.provider_tag,
        "distribution": config.distribution,
        "version": config.version,
        "cuda_architectures": [config.provider_tag],
        "modules": modules,
    }
    (PROVIDER_SOURCE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _build_aot_modules(verbose: bool = True) -> None:
    _ensure_build_inputs()

    from flashinfer import aot

    jit_cache_dir = PROVIDER_SOURCE_DIR / "jit_cache"
    build_dir = PROJECT_ROOT / "build" / "aot-providers" / config.provider_tag
    aot.compile_and_package_modules(
        out_dir=jit_cache_dir,
        build_dir=build_dir,
        project_root=PROJECT_ROOT,
        config=None,
        verbose=verbose,
        skip_prebuilt=False,
    )
    _write_provider_manifest(jit_cache_dir)


def _provider_platform_tag(default_platform_tag: str) -> str:
    requested_tag = os.environ.get(PROVIDER_PLATFORM_TAG_ENV, "").strip()
    if not requested_tag:
        return default_platform_tag

    machine = platform.machine()
    expected_tag = f"manylinux_2_28_{machine}"
    libc_name, libc_version = platform.libc_ver()
    libc_match = re.fullmatch(r"(\d+)\.(\d+)", libc_version)
    if (
        platform.system() != "Linux"
        or machine not in ("x86_64", "aarch64")
        or requested_tag != expected_tag
        or libc_name != "glibc"
        or libc_match is None
    ):
        raise RuntimeError(
            f"Unsupported provider platform tag {requested_tag!r} for "
            f"{platform.system()} {machine} with {libc_name} {libc_version}; "
            f"expected {expected_tag!r} on glibc 2.28 or older"
        )
    glibc_version = tuple(int(part) for part in libc_match.groups())
    if glibc_version > (2, 28):
        raise RuntimeError(
            f"glibc {libc_version} is too new for provider platform tag "
            f"{requested_tag!r}"
        )
    return requested_tag


class PlatformSpecificBdistWheel(bdist_wheel):
    """Build a native provider wheel with a stable Python ABI tag."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.py_limited_api = "cp39"

    def get_tag(self):
        _, _, default_platform_tag = super().get_tag()
        platform_tag = _provider_platform_tag(default_platform_tag)
        return "cp39", "abi3", platform_tag


class _MonkeyPatchBdistWheel:
    def __enter__(self):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        self.original_bdist_wheel = setuptools_bdist_wheel.bdist_wheel
        setuptools_bdist_wheel.bdist_wheel = PlatformSpecificBdistWheel
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        setuptools_bdist_wheel.bdist_wheel = self.original_bdist_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    print(
        f"Building {config.distribution} {config.version} for "
        f"{config.cuda_architecture}"
    )
    _build_aot_modules()
    with _MonkeyPatchBdistWheel():
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _build_aot_modules()
    build_editable_impl = getattr(_orig, "build_editable", None)
    if build_editable_impl is None:
        raise RuntimeError("build_editable not supported by setuptools backend")
    with _MonkeyPatchBdistWheel():
        return build_editable_impl(wheel_directory, config_settings, metadata_directory)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )


def get_requires_for_build_wheel(config_settings=None):
    """Install configured dependencies in the isolated wheel build env."""
    return [
        *_orig.get_requires_for_build_wheel(config_settings),
        *get_build_dependency_requirements(config.cuda_major),
    ]


def get_requires_for_build_editable(config_settings=None):
    """Install configured dependencies in the isolated editable build env."""
    get_requires = getattr(_orig, "get_requires_for_build_editable", None)
    requirements = [] if get_requires is None else get_requires(config_settings)
    return [*requirements, *get_build_dependency_requirements(config.cuda_major)]
