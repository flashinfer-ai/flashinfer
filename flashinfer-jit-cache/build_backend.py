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
    from flashinfer import aot

    # Get the project root directory
    project_root = Path(__file__).parent.parent

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
