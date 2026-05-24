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
from .utils import write_if_different
from . import env as jit_env
import os
import pathlib
import jinja2
from itertools import product
from typing import Dict, Tuple, List, Any


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


def gen_mixed_comm_module() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_mixed_comm"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [jit_env.FLASHINFER_CSRC_DIR / "mixed_comm.cu"]

    def is_valid_op(
        op: str,
        use_local_tp: bool,
        use_local_dp: bool,
        use_inter_tp: bool,
        use_inter_dp: bool,
    ) -> bool:
        # Should be aligned with is_valid_op in mixed_comm.cuh
        use_tp = use_local_tp or use_inter_tp
        use_dp = use_local_dp or use_inter_dp
        use_mixed = use_tp and use_dp
        if use_mixed:
            return op in ("ALLREDUCE_ALLGATHER", "REDUCESCATTER_ALLREDUCE")
        if use_tp:
            return op == "ALLREDUCE"
        return op in ("ALLGATHER", "REDUCESCATTER")

    def is_valid_mode(mode: str, use_local_tp: bool, use_inter_tp: bool) -> bool:
        # Should be aligned with is_valid_mode in mixed_comm.cuh
        if mode.startswith("OPT_WAITS_"):
            return True
        if mode.startswith("OPT_BYTES1_"):
            return use_local_tp
        if mode.startswith("OPT_BYTES2_"):
            return use_inter_tp
        return False

    def is_valid_block_y(
        block_size_y: int, local_tp_size: int, local_dp_size: int, mode: str
    ) -> bool:
        # Should be aligned with is_valid_block_y in mixed_comm.cuh
        if not mode.endswith("_MC"):
            return block_size_y == 1
        if mode.startswith("OPT_WAITS_"):
            return local_dp_size % block_size_y == 0
        return (local_tp_size * local_dp_size) % block_size_y == 0

    with open(jit_env.FLASHINFER_CSRC_DIR / "mixed_comm_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
    kernel_dict = {
        "allreduce_kernel": "ALLREDUCE",
        "allgather_kernel": "ALLGATHER",
        "reducescatter_kernel": "REDUCESCATTER",
        "fused_allreduce_allgather_kernel": "ALLREDUCE_ALLGATHER",
        "fused_reducescatter_allreduce_kernel": "REDUCESCATTER_ALLREDUCE",
    }
    mode_list = [
        "OPT_WAITS_MC",
        "OPT_WAITS_UC",
        "OPT_BYTES1_MC",
        "OPT_BYTES1_UC",
        "OPT_BYTES2_MC",
        "OPT_BYTES2_UC",
    ]
    dtype_list = ["nv_half", "nv_bfloat16"]
    local_size_list = [1, 2, 4, 8]
    inter_flag_list = [False, True]
    compile_dict: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
        (kernel_name, mode): []
        for kernel_name, mode in product(kernel_dict.keys(), mode_list)
    }
    for kernel_name, op in kernel_dict.items():
        for (
            block_size_y,
            local_tp_size,
            local_dp_size,
            use_inter_tp,
            use_inter_dp,
            mode,
            dtype,
        ) in product(
            local_size_list,
            local_size_list,
            local_size_list,
            inter_flag_list,
            inter_flag_list,
            mode_list,
            dtype_list,
        ):
            use_local_tp = local_tp_size > 1
            use_local_dp = local_dp_size > 1
            if not use_local_tp and not use_local_dp:
                continue
            if not is_valid_op(
                op, use_local_tp, use_local_dp, use_inter_tp, use_inter_dp
            ):
                continue
            if not is_valid_mode(mode, use_local_tp, use_inter_tp):
                continue
            if not is_valid_block_y(block_size_y, local_tp_size, local_dp_size, mode):
                continue
            compile_dict[(kernel_name, mode)].append(
                {
                    "kernel_name": kernel_name,
                    "block_size_y": block_size_y,
                    "local_tp_size": local_tp_size,
                    "local_dp_size": local_dp_size,
                    "use_inter_tp": str(use_inter_tp).lower(),
                    "use_inter_dp": str(use_inter_dp).lower(),
                    "mode": mode,
                    "dtype": dtype,
                }
            )
    for (kernel_name, mode), instantiations in compile_dict.items():
        dest_path = gen_directory / f"mixed_comm_{kernel_name}_{mode}_inst.cu"
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(instantiations=instantiations)
        write_if_different(dest_path, source)

    import nvidia.nvshmem

    path_base = pathlib.Path(nvidia.nvshmem.__path__[0])

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10]
    ) + ["-rdc=true"]
    ldflags = [f"-L{str(path_base / 'lib')}"] + ["-lnvshmem_device"]

    return gen_jit_spec(
        "mixed_comm",
        source_paths,
        extra_include_paths=[str(path_base / "include")],
        extra_cuda_cflags=nvcc_flags,
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
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "cpp"
            / "common"
            / "logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "cpp"
            / "common"
            / "stringUtils.cpp",
        ],
        extra_include_paths=[
            str(jit_env.FLASHINFER_CSRC_DIR / "nv_internal"),
            str(jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include"),
        ],
    )


def gen_dcp_alltoall_module() -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]
    )
    return gen_jit_spec(
        "dcp_alltoall",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_dcp_alltoall.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "helixAllToAll.cu",
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
        extra_cuda_cflags=nvcc_flags,
    )
