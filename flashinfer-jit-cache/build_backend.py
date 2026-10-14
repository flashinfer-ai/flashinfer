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
import re
import shutil
from pathlib import Path
from setuptools import build_meta as _orig
from wheel.bdist_wheel import bdist_wheel

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_utils import get_git_version

# Skip version check when building flashinfer-jit-cache package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


def _provider_tag(architecture: str) -> str:
    normalized = architecture.strip().lower()
    for prefix in ("compute_", "sm_", "sm"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = normalized.replace(".", "").replace("_", "")
    if not re.fullmatch(r"\d{2,3}[af]?", normalized):
        raise RuntimeError(f"Invalid provider CUDA architecture {architecture!r}")
    return f"sm{normalized}"


def _write_provider_requirements(version: str) -> None:
    requirements_path = (
        Path(__file__).parent / "flashinfer_jit_cache" / "_provider_requirements.txt"
    )
    architecture_list = os.environ.get("FLASHINFER_JIT_CACHE_PROVIDER_ARCHS", "")
    if not architecture_list.strip():
        raise RuntimeError("A shim build requires FLASHINFER_JIT_CACHE_PROVIDER_ARCHS")
    provider_tags = sorted(
        {_provider_tag(architecture) for architecture in architecture_list.split()}
    )
    # Dependencies are literal; runtime ranks compatibility from provider targets.
    requirements = [
        f"flashinfer-jit-cache-{provider_tag}=={version}"
        for provider_tag in provider_tags
    ]
    requirements_path.write_text("\n".join(requirements) + "\n")


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
        _write_provider_requirements(version)
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-jit-cache package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    _write_provider_requirements(version)
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def _prepare_build():
    """Generate shim metadata and remove stale monolithic build output."""
    _create_build_metadata()
    aot_package_dir = Path(__file__).parent / "flashinfer_jit_cache" / "jit_cache"
    if aot_package_dir.exists():
        shutil.rmtree(aot_package_dir)


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
    """Build the platform-specific provider shim wheel."""
    print("Building flashinfer-jit-cache provider shim wheel...")

    _prepare_build()

    # Requirements differ by CPU platform, so this metadata-only shim still
    # needs a platform tag.
    with _MonkeyPatchBdistWheel():
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable provider shim install."""
    print("Building flashinfer-jit-cache in editable mode...")

    _prepare_build()

    _orig_build_editable = getattr(_orig, "build_editable", None)
    if _orig_build_editable is None:
        raise RuntimeError("build_editable not supported by setuptools backend")

    with _MonkeyPatchBdistWheel():
        return _orig_build_editable(
            wheel_directory, config_settings, metadata_directory
        )


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


def get_requires_for_build_wheel(config_settings=None):
    """Return the shim's isolated wheel build requirements."""
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings=None):
    """Return the shim's isolated editable build requirements."""
    get_requires = getattr(_orig, "get_requires_for_build_editable", None)
    return [] if get_requires is None else get_requires(config_settings)
