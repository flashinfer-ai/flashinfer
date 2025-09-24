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
from pathlib import Path
from setuptools import build_meta as _orig

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# add flashinfer._build_meta if not there, it should exist in Path(__file__).parent.parent / "flashinfer" / "_build_meta.py"
build_meta_file = Path(__file__).parent.parent / "flashinfer" / "_build_meta.py"
if not build_meta_file.exists():
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
            return f.read().strip()
    return "0.0.0+unknown"


def compile_aot_modules(output_dir: Path, verbose: bool = True):
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


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel with custom AOT module compilation."""
    print("Building flashinfer-aot-modules wheel...")

    # First, ensure AOT modules are compiled
    aot_package_dir = Path(__file__).parent / "flashinfer_aot_modules" / "aot_modules"
    aot_package_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Compile AOT modules
        compile_aot_modules(aot_package_dir)

        # Verify that some modules were actually compiled
        so_files = list(aot_package_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError("No .so files were generated during AOT compilation")

        print(f"Successfully compiled {len(so_files)} AOT modules")

    except Exception as e:
        print(f"Failed to compile AOT modules: {e}")
        raise

    # Create build metadata file with version information
    package_dir = Path(__file__).parent / "flashinfer_aot_modules"
    build_meta_file = package_dir / "_build_meta.py"
    version = get_version()

    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-aot-modules package."""\n')
        f.write(f'__version__ = "{version}"\n')

    print(f"Created build metadata file with version {version}")

    # Now build the wheel using setuptools
    # The setup.py file will handle the platform-specific wheel naming
    result = _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

    return result


def build_sdist(sdist_directory, config_settings=None):
    """Build source distribution with custom handling."""
    print("Building flashinfer-aot-modules source distribution...")

    # Compile AOT modules for source distribution
    aot_package_dir = Path(__file__).parent / "flashinfer_aot_modules" / "aot_modules"
    aot_package_dir.mkdir(parents=True, exist_ok=True)

    try:
        compile_aot_modules(aot_package_dir)
    except Exception as e:
        print(f"Warning: Failed to compile AOT modules for sdist: {e}")
        print("The source distribution will not include pre-compiled modules")

    # Create build metadata file with version information
    package_dir = Path(__file__).parent / "flashinfer_aot_modules"
    build_meta_file = package_dir / "_build_meta.py"
    version = get_version()

    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-aot-modules package."""\n')
        f.write(f'__version__ = "{version}"\n')

    print(f"Created build metadata file with version {version}")

    # Build the sdist using setuptools
    return _orig.build_sdist(sdist_directory, config_settings)


# Export the required interface
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_editable = getattr(_orig, "build_editable", None)
get_requires_for_build_editable = getattr(
    _orig, "get_requires_for_build_editable", None
)
