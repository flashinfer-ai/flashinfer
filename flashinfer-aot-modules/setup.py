"""Setup script for flashinfer-aot-modules package."""

import os
import platform
from pathlib import Path
from setuptools import setup, find_packages
from wheel.bdist_wheel import bdist_wheel


def get_version():
    """Get version from version.txt file."""
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0"

    # Append CUDA version suffix if available
    cuda_suffix = os.environ.get("CUDA_VERSION_SUFFIX", "")
    if cuda_suffix:
        # Use + to create a local version identifier that will appear in wheel name
        version = f"{version}+{cuda_suffix}"

    return version


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


if __name__ == "__main__":
    setup(
        name="flashinfer-aot-modules",
        version=get_version(),
        description="Pre-compiled AOT modules for FlashInfer",
        long_description="This package contains pre-compiled AOT modules for FlashInfer. It provides all necessary compiled shared libraries (.so files) for optimized inference operations.",
        long_description_content_type="text/plain",
        author="FlashInfer team",
        maintainer="FlashInfer team",
        url="https://github.com/flashinfer-ai/flashinfer",
        project_urls={
            "Homepage": "https://github.com/flashinfer-ai/flashinfer",
            "Documentation": "https://github.com/flashinfer-ai/flashinfer",
            "Repository": "https://github.com/flashinfer-ai/flashinfer",
            "Issue Tracker": "https://github.com/flashinfer-ai/flashinfer/issues",
        },
        packages=find_packages(),
        package_data={
            "flashinfer_aot_modules": ["aot_modules/**/*.so"],
        },
        include_package_data=True,
        python_requires=">=3.9",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        license="Apache-2.0",
        cmdclass={"bdist_wheel": PlatformSpecificBdistWheel},
    )
