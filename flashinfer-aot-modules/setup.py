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
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

# Add parent directory to path to import flashinfer modules
sys.path.insert(0, str(Path(__file__).parent.parent))


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

    # Use the centralized compilation function from aot.py
    aot.compile_and_package_modules(
        output_dir=output_dir,
        config=None,  # Use default config
        project_root=project_root,
        verbose=verbose,
        setup_environment=True,
    )


class CompileAndBuildPy(build_py):
    """Custom build command that compiles AOT modules before building."""

    def run(self):
        print("Building flashinfer-aot-modules package...")

        # Create directory for AOT modules within the package
        aot_package_dir = (
            Path(self.build_lib) / "flashinfer_aot_modules" / "aot_modules"
        )
        aot_package_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Compile AOT modules directly to the package directory
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
        package_dir = Path(self.build_lib) / "flashinfer_aot_modules"
        build_meta_file = package_dir / "_build_meta.py"
        version = get_version()

        with open(build_meta_file, "w") as f:
            f.write('"""Build metadata for flashinfer-aot-modules package."""\n')
            f.write(f'__version__ = "{version}"\n')

        print(f"Created build metadata file with version {version}")

        # Continue with normal build
        super().run()


class CustomSdist(sdist):
    """Custom sdist command that includes compiled AOT modules."""

    def run(self):
        # Compile AOT modules first
        print("Compiling AOT modules for source distribution...")

        aot_package_dir = (
            Path(self.distribution.package_dir.get("", "."))
            / "flashinfer_aot_modules"
            / "aot_modules"
        )
        aot_package_dir.mkdir(parents=True, exist_ok=True)

        try:
            compile_aot_modules(aot_package_dir)
        except Exception as e:
            print(f"Warning: Failed to compile AOT modules for sdist: {e}")
            print("The source distribution will not include pre-compiled modules")

        # Create build metadata file with version information for sdist
        package_dir = (
            Path(self.distribution.package_dir.get("", ".")) / "flashinfer_aot_modules"
        )
        build_meta_file = package_dir / "_build_meta.py"
        version = get_version()

        with open(build_meta_file, "w") as f:
            f.write('"""Build metadata for flashinfer-aot-modules package."""\n')
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
            "build_py": CompileAndBuildPy,
            "sdist": CustomSdist,
        },
    )
