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

import os
import platform
import sysconfig
import subprocess
import re
from pathlib import Path
from packaging.version import Version
from typing import List, Mapping

import setuptools
from setuptools.dist import Distribution, strtobool

root = Path(__file__).parent.resolve()
aot_ops_package_dir = root / "build" / "aot-ops-package-dir"
enable_aot = aot_ops_package_dir.is_dir() and any(aot_ops_package_dir.iterdir())


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def get_version():
    package_version = (root / "version.txt").read_text().strip()
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version is None:
        return package_version
    return f"{package_version}+{local_version}"


def get_cuda_path() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    # get output of "which nvcc"
    nvcc_path = subprocess.run(["which", "nvcc"], capture_output=True)
    if nvcc_path.returncode != 0:
        raise RuntimeError("Could not find nvcc")
    cuda_home = os.path.dirname(
        os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
    )
    return cuda_home


def get_cuda_version() -> Version:
    cuda_home = get_cuda_path()
    nvcc = os.path.join(cuda_home, "bin/nvcc")
    txt = subprocess.check_output([nvcc, "--version"], text=True)
    matches = re.findall(r"release (\d+\.\d+),", txt)
    if not matches:
        raise RuntimeError(
            f"Could not parse CUDA version from nvcc --version output: {txt}"
        )
    return Version(matches[0])


def generate_build_meta(aot_build_meta: dict) -> None:
    build_meta_str = f"__version__ = {get_version()!r}\n"
    if len(aot_build_meta) != 0:
        build_meta_str += f"build_meta = {aot_build_meta!r}\n"
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
    "apache-tvm-ffi>=0.1.0a10",
    "packaging>=24.2",
    "nvidia-cudnn-frontend>=1.13.0",
]
generate_build_meta({})

if enable_aot:
    import torch

    cuda_version = get_cuda_version()
    torch_full_version = Version(torch.__version__)
    torch_version = f"{torch_full_version.major}.{torch_full_version.minor}"
    install_requires = [req for req in install_requires if not req.startswith("torch ")]
    install_requires.append(f"torch == {torch_version}.*")

    aot_build_meta = {}
    aot_build_meta["cuda_major"] = cuda_version.major
    aot_build_meta["cuda_minor"] = cuda_version.minor
    aot_build_meta["torch"] = torch_version
    aot_build_meta["python"] = platform.python_version()
    aot_build_meta["FLASHINFER_CUDA_ARCH_LIST"] = os.environ.get(
        "FLASHINFER_CUDA_ARCH_LIST"
    )
    generate_build_meta(aot_build_meta)


class AotDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return enable_aot


bdist_wheel_options = {}
use_limited_api = strtobool(os.getenv("FLASHINFER_AOT_USE_PY_LIMITED_API", "1"))

if use_limited_api and not sysconfig.get_config_var("Py_GIL_DISABLED"):
    bdist_wheel_options["py_limited_api"] = "cp39"

setuptools.setup(
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    options={"bdist_wheel": bdist_wheel_options},
    distclass=AotDistribution,
)
