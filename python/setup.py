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

import argparse
import os
import platform
import re
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext

this_dir = Path(__file__).parent.resolve()

sys.path.append(str(this_dir))

package_version = "0.1.6"
enable_aot = os.environ.get("FLASHINFER_ENABLE_AOT", "0") == "1"
enable_bf16 = os.environ.get("FLASHINFER_ENABLE_BF16", "1") == "1"
enable_fp8 = os.environ.get("FLASHINFER_ENABLE_FP8", "1") == "1"

if enable_bf16:
    torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
if enable_fp8:
    torch_cpp_ext.COMMON_NVCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8")


def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def get_instantiation_cu() -> Tuple[List[str], List[str], List[str]]:
    from _aot_build_utils import (
        generate_batch_paged_decode_inst,
        generate_batch_paged_prefill_inst,
        generate_batch_ragged_prefill_inst,
        generate_dispatch_inc,
        generate_single_decode_inst,
        generate_single_prefill_inst,
    )

    path = this_dir / "csrc" / "generated"
    path.mkdir(parents=True, exist_ok=True)

    head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64,128,256").split(",")
    pos_encoding_modes = os.environ.get("FLASHINFER_POS_ENCODING_MODES", "0").split(",")
    allow_fp16_qk_reduction_options = os.environ.get(
        "FLASHINFER_ALLOW_FP16_QK_REDUCTION_OPTIONS", "0"
    ).split(",")
    mask_modes = os.environ.get("FLASHINFER_MASK_MODES", "0,1,2").split(",")

    head_dims = list(map(int, head_dims))
    pos_encoding_modes = list(map(int, pos_encoding_modes))
    allow_fp16_qk_reduction_options = list(map(int, allow_fp16_qk_reduction_options))
    mask_modes = list(map(int, mask_modes))
    # dispatch.inc
    write_if_different(
        path / "dispatch.inc",
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                head_dims=head_dims,
                pos_encoding_modes=pos_encoding_modes,
                allow_fp16_qk_reductions=allow_fp16_qk_reduction_options,
                mask_modes=mask_modes,
            )
        ),
    )

    idtypes = ["i32"]
    prefill_dtypes = ["f16"]
    decode_dtypes = ["f16"]
    fp16_dtypes = ["f16"]
    fp8_dtypes = ["e4m3", "e5m2"]
    if enable_bf16:
        prefill_dtypes.append("bf16")
        decode_dtypes.append("bf16")
        fp16_dtypes.append("bf16")
    if enable_fp8:
        decode_dtypes.extend(fp8_dtypes)

    files_decode = []
    files_prefill = []
    single_decode_uris = []
    # single decode files
    for head_dim, pos_encoding_mode in product(head_dims, pos_encoding_modes):
        for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
            product(fp16_dtypes, fp8_dtypes)
        ):
            dtype_out = dtype_q
            fname = f"single_decode_head_{head_dim}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}.cu"
            files_decode.append(str(path / fname))
            content = generate_single_decode_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                dtype_q,
                dtype_kv,
                dtype_out,
            )
            for use_sliding_window in [True, False]:
                for use_logits_soft_cap in [True, False]:
                    single_decode_uris.append(
                        f"single_decode_with_kv_cache_dtype_q_{dtype_q}_"
                        f"dtype_kv_{dtype_kv}_"
                        f"dtype_o_{dtype_out}_"
                        f"head_dim_{head_dim}_"
                        f"posenc_{pos_encoding_mode}_"
                        f"use_swa_{use_sliding_window}_"
                        f"use_logits_cap_{use_logits_soft_cap}"
                    )
            write_if_different(path / fname, content)

    # batch decode files
    batch_decode_uris = []
    for (
        head_dim,
        pos_encoding_mode,
    ) in product(
        head_dims,
        pos_encoding_modes,
    ):
        for idtype in idtypes:
            for dtype_q, dtype_kv in list(zip(decode_dtypes, decode_dtypes)) + list(
                product(fp16_dtypes, fp8_dtypes)
            ):
                dtype_out = dtype_q
                fname = f"batch_paged_decode_head_{head_dim}_posenc_{pos_encoding_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_out}_idtype_{idtype}.cu"
                files_decode.append(str(path / fname))
                content = generate_batch_paged_decode_inst.get_cu_file_str(
                    head_dim,
                    pos_encoding_mode,
                    dtype_q,
                    dtype_kv,
                    dtype_out,
                    idtype,
                )
                for use_sliding_window in [True, False]:
                    for use_logits_soft_cap in [True, False]:
                        batch_decode_uris.append(
                            f"batch_decode_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_out}_"
                            f"dtype_idx_{idtype}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{use_sliding_window}_"
                            f"use_logits_cap_{use_logits_soft_cap}"
                        )
                write_if_different(path / fname, content)

    # single prefill files
    single_prefill_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
    ) in product(
        head_dims,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        mask_modes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            product(prefill_dtypes, fp8_dtypes)
        ):
            fname = f"single_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}.cu"
            files_prefill.append(str(path / fname))
            content = generate_single_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
            )
            for use_sliding_window in [True, False]:
                for use_logits_soft_cap in [True, False]:
                    if (
                        mask_mode == 0
                    ):  # NOTE(Zihao): uri do not contain mask, avoid duplicate uris
                        single_prefill_uris.append(
                            f"single_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{use_sliding_window}_"
                            f"use_logits_cap_{use_logits_soft_cap}_"
                            f"f16qk_{bool(allow_fp16_qk_reduction)}"
                        )
            write_if_different(path / fname, content)

    # batch prefill files
    batch_prefill_uris = []
    for (
        head_dim,
        pos_encoding_mode,
        allow_fp16_qk_reduction,
        mask_mode,
        idtype,
    ) in product(
        head_dims,
        pos_encoding_modes,
        allow_fp16_qk_reduction_options,
        mask_modes,
        idtypes,
    ):
        for dtype_q, dtype_kv in list(zip(prefill_dtypes, prefill_dtypes)) + list(
            product(prefill_dtypes, fp8_dtypes)
        ):
            fname = f"batch_paged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            files_prefill.append(str(path / fname))
            content = generate_batch_paged_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            fname = f"batch_ragged_prefill_head_{head_dim}_posenc_{pos_encoding_mode}_fp16qkred_{allow_fp16_qk_reduction}_mask_{mask_mode}_dtypeq_{dtype_q}_dtypekv_{dtype_kv}_dtypeout_{dtype_q}_idtype_{idtype}.cu"
            files_prefill.append(str(path / fname))
            content = generate_batch_ragged_prefill_inst.get_cu_file_str(
                head_dim,
                pos_encoding_mode,
                allow_fp16_qk_reduction,
                mask_mode,
                dtype_q,  # dtype_q
                dtype_kv,  # dtype_kv
                dtype_q,  # dtype_out
                idtype,
            )
            write_if_different(path / fname, content)

            for sliding_window in [True, False]:
                for logits_soft_cap in [True, False]:
                    if (
                        mask_mode == 0
                    ):  # NOTE(Zihao): uri do not contain mask, avoid duplicate uris
                        batch_prefill_uris.append(
                            f"batch_prefill_with_kv_cache_dtype_q_{dtype_q}_"
                            f"dtype_kv_{dtype_kv}_"
                            f"dtype_o_{dtype_q}_"
                            f"dtype_idx_{idtype}_"
                            f"head_dim_{head_dim}_"
                            f"posenc_{pos_encoding_mode}_"
                            f"use_swa_{sliding_window}_"
                            f"use_logits_cap_{logits_soft_cap}_"
                            f"f16qk_{bool(allow_fp16_qk_reduction)}"
                        )

    # Change to relative path
    files_prefill = [str(Path(p).relative_to(this_dir)) for p in files_prefill]
    files_decode = [str(Path(p).relative_to(this_dir)) for p in files_decode]

    return (
        files_prefill,
        files_decode,
        single_decode_uris
        + batch_decode_uris
        + single_prefill_uris
        + batch_prefill_uris,
    )


def get_version():
    local_version = os.getenv("FLASHINFER_LOCAL_VERSION")
    if local_version is None:
        return package_version
    return f"{package_version}+{local_version}"


def get_cuda_version() -> Tuple[int, int]:
    if torch_cpp_ext.CUDA_HOME is None:
        nvcc = "nvcc"
    else:
        nvcc = os.path.join(torch_cpp_ext.CUDA_HOME, "bin/nvcc")
    txt = subprocess.check_output([nvcc, "--version"], text=True)
    major, minor = map(int, re.findall(r"release (\d+)\.(\d+),", txt)[0])
    return major, minor


def generate_build_meta(aot_build_meta: Dict) -> None:
    build_meta_str = f"__version__ = {get_version()!r}\n"
    if len(aot_build_meta) != 0:
        build_meta_str += f"build_meta = {aot_build_meta!r}\n"
    (this_dir / "flashinfer" / "_build_meta.py").write_text(build_meta_str)


def generate_aot_config(aot_kernel_uris: List[str]) -> None:
    aot_config_str = f"""prebuilt_ops_uri = set({aot_kernel_uris})"""
    (this_dir / "flashinfer" / "jit" / "aot_config.py").write_text(aot_config_str)


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


class NinjaBuildExtension(torch_cpp_ext.BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            max_num_jobs_cores = max(1, os.cpu_count())
            os.environ["MAX_JOBS"] = str(max_num_jobs_cores)

        super().__init__(*args, **kwargs)


ext_modules = []
generate_build_meta({})
generate_aot_config([])

if enable_aot:
    # cuda arch check for fp8 at the moment.
    for cuda_arch_flags in torch_cpp_ext._get_cuda_arch_flags():
        arch = int(re.search(r"compute_(\d+)", cuda_arch_flags).group(1))
        if arch < 75:
            raise RuntimeError("FlashInfer requires sm75+")

    aot_build_meta = {}
    aot_build_meta["cuda_major"], aot_build_meta["cuda_minor"] = get_cuda_version()
    aot_build_meta["torch"] = torch.__version__
    aot_build_meta["python"] = platform.python_version()
    aot_build_meta["TORCH_CUDA_ARCH_LIST"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
    generate_build_meta(aot_build_meta)

    remove_unwanted_pytorch_nvcc_flags()
    files_prefill, files_decode, aot_kernel_uris = get_instantiation_cu()
    generate_aot_config(aot_kernel_uris)

    root = this_dir.parent
    include_dirs = [
        str(root.resolve() / "include"),
        str(root.resolve() / "3rdparty" / "cutlass" / "include"),  # for group gemm
        str(root.resolve() / "3rdparty" / "cutlass" / "tools" / "util" / "include"),
    ]
    cxx_flags = [
        "-O3",
        "-Wno-switch-bool",
    ]
    nvcc_flags = [
        "-O3",
        "-std=c++17",
        "--threads=1",
        "-Xfatbin",
        "-compress-all",
        "-use_fast_math",
    ]
    sm90a_flags = "-gencode arch=compute_90a,code=sm_90a".split()
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._kernels",
            sources=[
                "csrc/bmm_fp8.cu",
                "csrc/cascade.cu",
                "csrc/group_gemm.cu",
                "csrc/norm.cu",
                "csrc/page.cu",
                "csrc/quantization.cu",
                "csrc/rope.cu",
                "csrc/sampling.cu",
                "csrc/renorm.cu",
                "csrc/activation.cu",
                "csrc/batch_decode.cu",
                "csrc/batch_prefill.cu",
                "csrc/single_decode.cu",
                "csrc/single_prefill.cu",
                "csrc/flashinfer_ops.cu",
            ]
            + files_decode
            + files_prefill,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    )
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._kernels_sm90",
            sources=[
                "csrc/group_gemm_sm90.cu",
                "csrc/flashinfer_gemm_sm90_ops.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags + sm90a_flags,
            },
        )
    )

setuptools.setup(
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
)
