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

from pathlib import Path
from typing import List, Mapping

import setuptools

root = Path(__file__).parent.resolve()


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_version():
    package_version = (root / "version.txt").read_text().strip()
    return f"{package_version}"


def generate_build_meta() -> None:
    build_meta_str = f"__version__ = {get_version()!r}\n"
    write_if_different(root / "flashinfer" / "_build_meta.py", build_meta_str)


ext_modules: List[setuptools.Extension] = []
cmdclass: Mapping[str, type[setuptools.Command]] = {}
install_requires = [
    "numpy",
    "torch",
    "ninja",
    "requests",
    "nvidia-ml-py",
    "einops",
    "click",
    "tqdm",
    "tabulate",
    "apache-tvm-ffi==0.1.0b11",
    "packaging>=24.2",
    "nvidia-cudnn-frontend>=1.13.0",
]
generate_build_meta()


setuptools.setup(
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
)
