# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent TVM compilation cache for the optional TIRx KDA backend."""

import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile

import tvm
import tirx_kernels.tirx_lite as txl
from tvm.support.nvcc import get_cuda_version

from ...jit import env as jit_env
from ...jit.core import JitSpec, get_tmpdir, jit_spec_registry, logger


@functools.cache
def _compiler_fingerprint():
    sha = hashlib.sha256()
    import tvm_ffi

    compile_mode = os.environ.get("TVM_CUDA_COMPILE_MODE", "nvrtc").lower()
    nvrtc_version = None
    if compile_mode == "nvrtc":
        from cuda.bindings import nvrtc

        status, major, minor = nvrtc.nvrtcVersion()
        if status != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Unable to query the NVRTC compiler version")
        nvrtc_version = (major, minor)
    compiler_options = tuple(
        (name, os.environ.get(name))
        for name in (
            "TVM_CUDA_PTXAS_REG_LEVEL",
            "TVM_CUDA_PTXAS_EXTRA_OPTS",
            "TVM_CUDA_NVRTC_EXTRA_OPTS",
            "TVM_CUDA_NVCC_NO_FAST_MATH",
            "TVM_KERNEL_DEBUG",
            "TVM_IKET_OFFICIAL_PROFILE",
        )
    )

    sha.update(
        repr(
            (
                tvm.__version__,
                tvm.support.libinfo().get("GIT_COMMIT_HASH"),
                tvm_ffi.__version__,
                get_cuda_version(),
                compile_mode,
                nvrtc_version,
                compiler_options,
            )
        ).encode()
    )
    paths = [
        Path(__file__),
        Path(__file__).with_name("fused.py"),
        Path(__file__).with_name("split.py"),
        *Path(txl.__file__).parent.rglob("*.py"),
    ]
    for path in sorted(paths):
        sha.update(path.name.encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


class _KdaTirxSpec(JitSpec):
    def __init__(self, key, factory):
        digest = hashlib.sha256(
            repr((key, _compiler_fingerprint())).encode()
        ).hexdigest()
        self.name = "kda_tirx_" + digest
        self.path = jit_env.FLASHINFER_JIT_DIR / self.name / "kernel.so"
        self.meta_path = self.path.with_suffix(".json")
        self.factory = factory

    @property
    def lock_path(self):
        return get_tmpdir() / (self.name + ".lock")

    @property
    def is_compiled(self):
        if not self.path.is_file() or not self.meta_path.is_file():
            return False
        try:
            return (
                json.loads(self.meta_path.read_text())["sha256"]
                == hashlib.sha256(self.path.read_bytes()).hexdigest()
            )
        except (OSError, ValueError, KeyError):
            return False

    def get_library_path(self):
        return self.path

    def try_load(self):
        if not self.is_compiled:
            return None
        try:
            return tvm.runtime.executable.Executable(
                tvm.runtime.load_module(str(self.path))
            )
        except Exception as exc:
            logger.warning("Unable to load TIRx KDA cache %s: %s", self.path, exc)
            return None

    def build(self):
        kernel = self.factory()
        with kernel.target():
            executable = tvm.compile(
                kernel.mod, target=kernel.target(), tir_pipeline="tirx"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.path.parent) as tmp:
            library = Path(tmp) / "kernel.so"
            executable.export_library(str(library))
            metadata = Path(tmp) / "kernel.json"
            metadata.write_text(
                json.dumps({"sha256": hashlib.sha256(library.read_bytes()).hexdigest()})
            )
            os.replace(library, self.path)
            os.replace(metadata, self.meta_path)

    def load(self):
        return tvm.runtime.executable.Executable(
            tvm.runtime.load_module(str(self.path))
        )


@functools.lru_cache(maxsize=128)
def get_kernel(kind, arch, heads, tuning=()):
    def factory():
        if kind == "fused":
            from .fused import build_kernel

            (max_items,) = tuning
            return build_kernel(heads, arch, max_items=max_items)
        from .split import make_front, make_chain

        if kind == "front":
            return make_front(heads, arch)
        return make_chain(heads, arch, hpc=tuning[0])

    spec = _KdaTirxSpec((kind, arch, heads, tuning), factory)
    jit_spec_registry.register(spec)
    return spec.build_and_load()
