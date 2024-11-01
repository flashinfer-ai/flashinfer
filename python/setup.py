"""
Copyright (c) 2023 by FlashInfer team.

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

import contextlib
import os
import pathlib
import shutil
import sys
from typing import Iterator
import warnings

import setuptools

root = pathlib.Path(__file__).resolve().parents[1]
this_dir = pathlib.Path(__file__).parent


def get_version():
    version = os.getenv("FLASHINFER_BUILD_VERSION")
    if version is None:
        with open(this_dir / "flashinfer" / "data" / "version.txt") as f:
            version = f.read().strip()
    return version


def generate_build_meta() -> None:
    version = get_version()
    with open(this_dir / "flashinfer" / "_build_meta.py", "w") as f:
        f.write(f"__version__ = {version!r}\n")


def clear_aot_config():
    # remove aot_config.py
    aot_config_path = this_dir / "flashinfer" / "jit" / "aot_config.py"
    if os.path.exists(aot_config_path):
        os.remove(aot_config_path)


@contextlib.contextmanager
def link_data_files() -> Iterator[None]:
    this_dir = pathlib.Path(__file__).parent
    data_dir = root / "python" / "flashinfer" / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    def ln(src: str, dst: str, is_dir: bool = False) -> None:
        (data_dir / dst).symlink_to(root / src, target_is_directory=is_dir)

    ln("3rdparty/cutlass", "cutlass", True)
    ln("include", "include", True)
    ln("python/csrc", "csrc", True)
    ln("version.txt", "version.txt")
    (this_dir / "MANIFEST.in").unlink(True)
    (this_dir / "MANIFEST.in").symlink_to("jit_MANIFEST.in")

    yield

    if sys.argv[1] != "develop":
        shutil.rmtree(data_dir)
        (this_dir / "MANIFEST.in").unlink(True)


if __name__ == "__main__":
    clear_aot_config()

    # Suppress warnings complaining that:
    # Package 'flashinfer.data*' is absent from the `packages` configuration.
    warnings.filterwarnings("ignore", r".*flashinfer\.data.*", UserWarning)

    with link_data_files():
        generate_build_meta()

        setuptools.setup(
            name="flashinfer",
            version=get_version(),
            packages=setuptools.find_packages(
                include=["flashinfer*"],
                exclude=["flashinfer.data*"],
            ),
            include_package_data=True,
            author="FlashInfer team",
            license="Apache License 2.0",
            description="FlashInfer: Kernel Library for LLM Serving",
            url="https://github.com/flashinfer-ai/flashinfer",
            python_requires=">=3.8",
        )
