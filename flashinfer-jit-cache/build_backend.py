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
from pathlib import Path
from setuptools import build_meta as _orig

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# add flashinfer._build_meta, always override to ensure version is up-to-date
build_meta_file = Path(__file__).parent.parent / "flashinfer" / "_build_meta.py"
version_file = Path(__file__).parent.parent / "version.txt"
if version_file.exists():
    with open(version_file, "r") as f:
        version = f.read().strip()
with open(build_meta_file, "w") as f:
    f.write('"""Build metadata for flashinfer package."""\n')
    f.write(f'__version__ = "{version}"\n')


def get_version():
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Append CUDA version suffix if available
    cuda_suffix = os.environ.get("CUDA_VERSION_SUFFIX", "")
    if cuda_suffix:
        # Replace + with . for proper version formatting
        if "+" in version:
            base_version, local = version.split("+", 1)
            version = f"{base_version}+{cuda_suffix}.{local}"
        else:
            version = f"{version}+{cuda_suffix}"

    return version


def compile_jit_cache(output_dir: Path, verbose: bool = True):
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


def _prepare_build():
    """Shared preparation logic for both wheel and editable builds."""
    # First, ensure AOT modules are compiled
    aot_package_dir = Path(__file__).parent / "flashinfer_jit_cache" / "jit_cache"
    aot_package_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Compile AOT modules
        compile_jit_cache(aot_package_dir)

        # Verify that some modules were actually compiled
        so_files = list(aot_package_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError("No .so files were generated during AOT compilation")

        print(f"Successfully compiled {len(so_files)} AOT modules")

    except Exception as e:
        print(f"Failed to compile AOT modules: {e}")
        raise

    # Create build metadata file with version information
    package_dir = Path(__file__).parent / "flashinfer_jit_cache"
    build_meta_file = package_dir / "_build_meta.py"
    version = get_version()

    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-jit-cache package."""\n')
        f.write(f'__version__ = "{version}"\n')

    print(f"Created build metadata file with version {version}")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with custom AOT module compilation."""
    print("Building flashinfer-jit-cache wheel...")

    _prepare_build()

    # Now build the wheel using setuptools
    # The setup.py file will handle the platform-specific wheel naming
    result = _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

    return result


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


# Export the required interface
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
get_requires_for_build_editable = getattr(
    _orig, "get_requires_for_build_editable", None
)
