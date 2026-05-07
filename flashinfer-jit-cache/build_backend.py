"""
Copyright (c) 2025 by FlashInfer team.

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

import sys
import os
import platform
from pathlib import Path
from setuptools import build_meta as _orig
from wheel.bdist_wheel import bdist_wheel

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_utils import get_git_version

# Skip version check when building flashinfer-jit-cache package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


# SM family → arch list filter. Each entry is the predicate `(major, minor_str) -> bool`
# applied to entries in FLASHINFER_CUDA_ARCH_LIST. The local-version suffix encodes the
# family so users (and pip) can resolve the right wheel: e.g. "0.6.11+cu130.sm10x".
#
# Keep this in sync with `_detect_sm_family` in flashinfer/__main__.py.
SM_FAMILIES = {
    "sm9x": lambda major, minor: major < 10,        # 7.5 / 8.0 / 8.9 / 9.0a (Ampere/Ada/Hopper)
    "sm10x": lambda major, minor: 10 <= major < 12, # 10.0a / 10.3a / 11.0a (Datacenter Blackwell)
    "sm12x": lambda major, minor: major >= 12,      # 12.0f / 12.1a       (Consumer Blackwell)
}


def _filter_arch_list_for_family(arch_list: str, family: str) -> str:
    """Filter a space-separated FLASHINFER_CUDA_ARCH_LIST to only entries belonging to `family`."""
    predicate = SM_FAMILIES[family]
    kept = []
    for entry in arch_list.split():
        major_str, minor_str = entry.split(".", 1)
        major = int(major_str)
        # `minor_str` may carry a suffix like 'a' or 'f' — keep the whole string for output,
        # but parse leading digits for the comparison.
        leading_digits = "".join(c for c in minor_str if c.isdigit())
        minor = int(leading_digits) if leading_digits else 0
        if predicate(major, minor):
            kept.append(entry)
    return " ".join(kept)


def _resolve_sm_family() -> str:
    """Return the SM family this build targets, or '' for a legacy multi-family build."""
    family = os.environ.get("FLASHINFER_JIT_CACHE_SM_FAMILY", "").strip().lower()
    if not family:
        return ""
    if family not in SM_FAMILIES:
        raise RuntimeError(
            f"Invalid FLASHINFER_JIT_CACHE_SM_FAMILY={family!r}. "
            f"Expected one of: {sorted(SM_FAMILIES)}"
        )
    return family


def _apply_sm_family_filter() -> str:
    """If a family is selected, narrow FLASHINFER_CUDA_ARCH_LIST in-place and return the family suffix."""
    family = _resolve_sm_family()
    if not family:
        return ""

    arch_list = os.environ.get("FLASHINFER_CUDA_ARCH_LIST")
    if not arch_list:
        # The downstream build will fail with a clear error in compile_and_package_modules;
        # we let it raise there to keep error messages consistent.
        return family

    filtered = _filter_arch_list_for_family(arch_list, family)
    if not filtered:
        raise RuntimeError(
            f"FLASHINFER_JIT_CACHE_SM_FAMILY={family} but FLASHINFER_CUDA_ARCH_LIST="
            f"{arch_list!r} contains no archs in that family. "
            f"Set FLASHINFER_CUDA_ARCH_LIST to include archs matching {family}."
        )
    print(f"SM family {family}: filtering FLASHINFER_CUDA_ARCH_LIST {arch_list!r} -> {filtered!r}")
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = filtered
    return family


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Add dev suffix if specified
    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version
    git_version = get_git_version(cwd=Path(__file__).parent.parent)

    # Append local version suffix if available
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")

    # When this build targets a single SM family, append it to the local-version
    # so users can pin e.g. "flashinfer-jit-cache==0.6.11+cu130.sm10x".
    family = _resolve_sm_family()
    if family:
        local_version = f"{local_version}.{family}" if local_version else family

    if local_version:
        # Use + to create a local version identifier that will appear in wheel name
        version = f"{version}+{local_version}"
    build_meta_file = Path(__file__).parent / "flashinfer_jit_cache" / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent.parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-jit-cache package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def _compile_jit_cache(output_dir: Path, verbose: bool = True):
    """Compile AOT modules using flashinfer.aot functions directly."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Ensure 3rdparty submodules are populated (may be empty in CI Docker images).
    # Skip if submodules are already present or if git metadata is incomplete
    # (e.g., Docker builds where .git points to a parent repo not in the context).
    import subprocess

    submodule_check_paths = [
        project_root / "3rdparty" / "cutlass" / "include",
        project_root / "3rdparty" / "spdlog" / "include",
        project_root / "3rdparty" / "cccl" / "cub",
    ]
    if not all(p.exists() for p in submodule_check_paths):
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(project_root),
            capture_output=True,
        )
        if result.returncode != 0:
            missing = [str(p) for p in submodule_check_paths if not p.exists()]
            if missing:
                raise RuntimeError(
                    f"git submodule update failed and submodules are missing: {missing}\n"
                    f"git stderr: {result.stderr.decode().strip()}"
                )

    # Ensure flashinfer/data/ symlinks exist (normally created by the main
    # package's build_backend, but jit-cache builds may not install the main
    # package first). Use importlib to avoid name collision with this file.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "main_build_backend", project_root / "build_backend.py"
    )
    main_build_backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_build_backend)
    main_build_backend._create_data_dir(use_symlinks=True)

    from flashinfer import aot

    # Set up build directory
    build_dir = project_root / "build" / "aot"

    # Use the centralized compilation function from aot.py
    aot.compile_and_package_modules(
        out_dir=output_dir,
        build_dir=build_dir,
        project_root=project_root,
        config=None,  # Use default config
        verbose=verbose,
        skip_prebuilt=False,
    )


def _build_aot_modules():
    # First, ensure AOT modules are compiled
    aot_package_dir = Path(__file__).parent / "flashinfer_jit_cache" / "jit_cache"
    aot_package_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Compile AOT modules
        _compile_jit_cache(aot_package_dir)

        # Verify that some modules were actually compiled
        so_files = list(aot_package_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError("No .so files were generated during AOT compilation")

        print(f"Successfully compiled {len(so_files)} AOT modules")

    except Exception as e:
        print(f"Failed to compile AOT modules: {e}")
        raise


def _prepare_build():
    """Shared preparation logic for both wheel and editable builds."""
    _apply_sm_family_filter()
    # Re-derive the build metadata so the family suffix is reflected in
    # `_build_meta.py` even when the import-time call ran before the env var
    # was set (e.g. when callers configure FLASHINFER_JIT_CACHE_SM_FAMILY in
    # the same process before invoking build_wheel).
    _create_build_metadata()
    _build_aot_modules()


class PlatformSpecificBdistWheel(bdist_wheel):
    """Custom wheel builder that uses py_limited_api for cp39+."""

    def finalize_options(self):
        super().finalize_options()
        # Force platform-specific wheel (not pure Python)
        self.root_is_pure = False
        # Use py_limited_api for cp39 (Python 3.9+)
        self.py_limited_api = "cp39"

    def get_tag(self):
        # Use py_limited_api tags
        python_tag = "cp39"
        abi_tag = "abi3"  # Stable ABI tag

        # Get platform tag
        machine = platform.machine()
        if platform.system() == "Linux":
            # Use manylinux_2_28 as specified
            if machine == "x86_64":
                plat_tag = "manylinux_2_28_x86_64"
            elif machine == "aarch64":
                plat_tag = "manylinux_2_28_aarch64"
            else:
                plat_tag = f"linux_{machine}"
        else:
            # For non-Linux platforms, use the default
            import distutils.util

            plat_tag = distutils.util.get_platform().replace("-", "_").replace(".", "_")

        return python_tag, abi_tag, plat_tag


class _MonkeyPatchBdistWheel:
    """Context manager to temporarily replace bdist_wheel with our custom class."""

    def __enter__(self):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        self.original_bdist_wheel = setuptools_bdist_wheel.bdist_wheel
        setuptools_bdist_wheel.bdist_wheel = PlatformSpecificBdistWheel

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from setuptools.command import bdist_wheel as setuptools_bdist_wheel

        setuptools_bdist_wheel.bdist_wheel = self.original_bdist_wheel


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with custom AOT module compilation."""
    print("Building flashinfer-jit-cache wheel...")

    _prepare_build()

    with _MonkeyPatchBdistWheel():
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable install with custom AOT module compilation."""
    print("Building flashinfer-jit-cache in editable mode...")

    _prepare_build()

    # Now build the editable install using setuptools
    _orig_build_editable = getattr(_orig, "build_editable", None)
    if _orig_build_editable is None:
        raise RuntimeError("build_editable not supported by setuptools backend")

    result = _orig_build_editable(wheel_directory, config_settings, metadata_directory)

    return result


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata with platform-specific wheel tags."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable install."""
    with _MonkeyPatchBdistWheel():
        return _orig.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )


# Export the required interface
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = getattr(
    _orig, "get_requires_for_build_editable", None
)
