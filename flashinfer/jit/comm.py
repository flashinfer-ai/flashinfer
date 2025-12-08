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

from .core import JitSpec, gen_jit_spec, current_compilation_context
from . import env as jit_env
import shlex
import os


def gen_comm_alltoall_module() -> JitSpec:
    return gen_jit_spec(
        "comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_alltoall.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_alltoall_prepare.cu",
        ],
    )


def gen_trtllm_mnnvl_comm_module() -> JitSpec:
    return gen_jit_spec(
        "trtllm_mnnvl_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_mnnvl_allreduce.cu",
        ],
    )


def gen_nvshmem_module() -> JitSpec:
    lib_dirs = jit_env.get_nvshmem_lib_dirs()
    ldflags = (
        [f"-L{lib_dir}" for lib_dir in lib_dirs]
        + ["-lnvshmem_device"]
        + shlex.split(os.environ.get("NVSHMEM_LDFLAGS", ""))
    )

    return gen_jit_spec(
        "nvshmem",
        [jit_env.FLASHINFER_CSRC_DIR / "nvshmem_binding.cu"],
        extra_include_paths=[str(p) for p in jit_env.get_nvshmem_include_dirs()],
        extra_ldflags=ldflags,
        needs_device_linking=True,
    )


def gen_trtllm_comm_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10]
    )
    return gen_jit_spec(
        "trtllm_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce_fusion.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_moe_allreduce_fusion.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )


def gen_vllm_comm_module() -> JitSpec:
    return gen_jit_spec(
        "vllm_comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "vllm_custom_all_reduce.cu",
        ],
    )


def gen_moe_alltoall_module() -> JitSpec:
    return gen_jit_spec(
        "mnnvl_moe_alltoall",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_moe_alltoall.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "communicationKernels"
            / "moeAlltoAllKernels.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "cpp"
            / "common"
            / "envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "cpp"
            / "common"
            / "tllmException.cpp",
        ],
        extra_include_paths=[
            str(jit_env.FLASHINFER_CSRC_DIR / "nv_internal"),
            str(jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include"),
        ],
    )
