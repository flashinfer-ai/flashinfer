"""
Custom build backend that downloads cubins before building the package.
"""

import os
import sys
from pathlib import Path
from setuptools import build_meta as _orig
from setuptools.build_meta import *

# Add parent directory to path to import artifacts module
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


def _download_cubins():
    """Download cubins to the source directory before building."""
    from flashinfer.artifacts import download_artifacts

    # Create cubins directory in the source tree
    cubin_dir = Path(__file__).parent / "flashinfer_cubin" / "cubins"
    cubin_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variable to download to our package directory
    original_cubin_dir = os.environ.get("FLASHINFER_CUBIN_DIR")
    os.environ["FLASHINFER_CUBIN_DIR"] = str(cubin_dir)

    try:
        print(f"Downloading cubins to {cubin_dir}...")
        download_artifacts()
        print(f"Successfully downloaded cubins to {cubin_dir}")

        # Count the downloaded files
        cubin_files = list(cubin_dir.rglob("*.cubin"))
        print(f"Downloaded {len(cubin_files)} cubin files")

    finally:
        # Restore original environment variable
        if original_cubin_dir:
            os.environ["FLASHINFER_CUBIN_DIR"] = original_cubin_dir
        else:
            os.environ.pop("FLASHINFER_CUBIN_DIR", None)


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "flashinfer_cubin"
    build_meta_file = package_dir / "_build_meta.py"

    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-cubin package."""\n')
        f.write(f'__version__ = "{version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel, downloading cubins first."""
    _download_cubins()
    _create_build_metadata()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable install, downloading cubins first."""
    _download_cubins()
    _create_build_metadata()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


# Pass through all other hooks
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
