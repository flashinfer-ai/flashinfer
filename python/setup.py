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

from typing import List, Tuple

import pathlib
import os
import setuptools

root = pathlib.Path(__name__).parent


def get_version():
    version = os.getenv("FLASHINFER_BUILD_VERSION")
    if version is None:
        with open(root / "version.txt") as f:
            version = f.read().strip()
    return version


def generate_build_meta() -> None:
    version = get_version()
    with open(root / "flashinfer/_build_meta.py", "w") as f:
        f.write(f"__version__ = {version!r}\n")


def clear_aot_config():
    # remove aot_config.py
    aot_config_path = root / "flashinfer" / "jit" / "aot_config.py"
    if os.path.exists(aot_config_path):
        os.remove(aot_config_path)


if __name__ == "__main__":
    generate_build_meta()
    clear_aot_config()
    setuptools.setup(
        name="flashinfer",
        version=get_version(),
        packages=setuptools.find_packages(),
        author="FlashInfer team",
        license="Apache License 2.0",
        description="FlashInfer: Kernel Library for LLM Serving",
        url="https://github.com/flashinfer-ai/flashinfer",
        python_requires=">=3.8",
    )
