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

import os
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

# Add parent directory to path to import artifacts module
sys.path.insert(0, str(Path(__file__).parent.parent))

from flashinfer.artifacts import download_artifacts


def get_version():
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            return f.read().strip()
    return "0.0.0+unknown"


class DownloadAndBuildPy(build_py):
    """Custom build command that downloads cubins before building."""

    def run(self):
        print("Downloading cubins from artifactory...")

        # Ensure the cubins directory exists in the source tree first
        source_cubin_dir = Path(__file__).parent / "flashinfer_cubin" / "cubins"
        source_cubin_dir.mkdir(parents=True, exist_ok=True)

        # Create a placeholder file to ensure directory is not empty during packaging
        placeholder_file = source_cubin_dir / ".placeholder"
        if not placeholder_file.exists():
            placeholder_file.write_text(
                "# Placeholder file to ensure directory is not empty during packaging\n"
            )

        # Create a temporary directory for cubins within the package
        cubin_package_dir = Path(self.build_lib) / "flashinfer_cubin" / "cubins"
        cubin_package_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variable to download to our package directory
        original_cubin_dir = os.environ.get("FLASHINFER_CUBIN_DIR")
        os.environ["FLASHINFER_CUBIN_DIR"] = str(cubin_package_dir)

        try:
            # Download all cubins using the existing download_artifacts function
            download_artifacts()
            print(f"Cubins downloaded to {cubin_package_dir}")
        except Exception as e:
            raise RuntimeError(f"Error downloading cubins: {e}") from e
        finally:
            # Restore original environment variable
            if original_cubin_dir:
                os.environ["FLASHINFER_CUBIN_DIR"] = original_cubin_dir
            else:
                os.environ.pop("FLASHINFER_CUBIN_DIR", None)

        # Create build metadata file with version information
        package_dir = Path(self.build_lib) / "flashinfer_cubin"
        build_meta_file = package_dir / "_build_meta.py"
        version = get_version()

        with open(build_meta_file, "w") as f:
            f.write('"""Build metadata for flashinfer-cubin package."""\n')
            f.write(f'__version__ = "{version}"\n')

        print(f"Created build metadata file with version {version}")

        # Continue with normal build
        super().run()


class CustomSdist(sdist):
    """Custom sdist command that includes downloaded cubins."""

    def run(self):
        # Ensure the cubins directory exists in the source tree first
        source_cubin_dir = Path(__file__).parent / "flashinfer_cubin" / "cubins"
        source_cubin_dir.mkdir(parents=True, exist_ok=True)

        # Create a placeholder file to ensure directory is not empty during packaging
        placeholder_file = source_cubin_dir / ".placeholder"
        if not placeholder_file.exists():
            placeholder_file.write_text(
                "# Placeholder file to ensure directory is not empty during packaging\n"
            )

        # Download cubins first
        print("Downloading cubins for source distribution...")

        cubin_package_dir = (
            Path(self.distribution.package_dir.get("", "."))
            / "flashinfer_cubin"
            / "cubins"
        )
        cubin_package_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variable to download to our package directory
        original_cubin_dir = os.environ.get("FLASHINFER_CUBIN_DIR")
        os.environ["FLASHINFER_CUBIN_DIR"] = str(cubin_package_dir)

        try:
            download_artifacts()
        finally:
            if original_cubin_dir:
                os.environ["FLASHINFER_CUBIN_DIR"] = original_cubin_dir
            else:
                os.environ.pop("FLASHINFER_CUBIN_DIR", None)

        # Create build metadata file with version information for sdist
        package_dir = (
            Path(self.distribution.package_dir.get("", ".")) / "flashinfer_cubin"
        )
        build_meta_file = package_dir / "_build_meta.py"
        version = get_version()

        with open(build_meta_file, "w") as f:
            f.write('"""Build metadata for flashinfer-cubin package."""\n')
            f.write(f'__version__ = "{version}"\n')

        print(f"Created build metadata file with version {version}")

        # Continue with normal sdist
        super().run()


# Minimal setup() call - configuration is now in pyproject.toml
# This is kept for backward compatibility and to register custom cmdclass
if __name__ == "__main__":
    setup(
        version=get_version(),
        cmdclass={
            "build_py": DownloadAndBuildPy,
            "sdist": CustomSdist,
        },
    )
