"""
Copyright (c) 2024 by FlashInfer team.

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

import functools
import os
from enum import Enum
from itertools import product
from types import SimpleNamespace
from typing import List, Literal, Optional, Tuple

import jinja2
import torch

from .artifacts import ArtifactPath, MetaInfoHash
from .autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from .fused_moe.utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)
from .jit.cubin_loader import get_cubin
from .utils import (
    is_sm100a_supported,
    is_sm120a_supported,
    is_sm121a_supported,
    LibraryError,
)

CUDNN_AVAILABLE = False
try:
    import cudnn

    CUDNN_AVAILABLE = True
except ImportError:
    pass
except OSError as e:
    error_msg = str(e).lower()
    is_lib_missing = any(ext in error_msg for ext in [".so", ".dll"])
    if not is_lib_missing:
        raise


from .jit import JitSpec
from .jit import env as jit_env
from .jit import (
    gen_jit_spec,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    current_compilation_context,
)
from .jit.cubin_loader import setup_cubin_loader
from .jit.utils import dtype_cutlass_map, filename_safe_dtype_map, write_if_different
from .utils import (
    _get_cache_buf,
    determine_gemm_backend,
    get_indptr,
    is_float8,
    register_custom_op,
    register_fake_op,
    get_compute_capability,
)

DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024


def _match_sm_version(device: torch.device, sm_version: list[str]):
    major, minor = get_compute_capability(device)
    device_arch = f"{major * 10 + minor}"
    return device_arch in sm_version


def gen_gemm_module() -> JitSpec:
    return gen_jit_spec(
        "gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "bmm_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_gemm_ops.cu",
        ],
        extra_ldflags=["-lcublas", "-lcublasLt"],
    )


@functools.cache
def get_gemm_module():
    module = gen_gemm_module().build_and_load()

    # auto-tuned cublas fp8 gemm runner
    def cublas_fp8_gemm_runner():
        class CublasFp8GemmRunner(TunableRunner):
            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                # cublas has heuristic for fp8 gemm, so we only need to use the default tactic
                return [0]

            def forward(
                self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs,
            ) -> torch.Tensor:
                cublas_handle = torch.cuda.current_blas_handle()
                a, b, scale_a, scale_b, out, workspace_buffer = inputs
                module.bmm_fp8.default(
                    a, b, out, scale_a, scale_b, workspace_buffer, cublas_handle
                )
                return out

        return CublasFp8GemmRunner()

    # torch library for cutlass_segment_gemm

    @register_custom_op("flashinfer::cutlass_segment_gemm", mutates_args=("y"))
    def cutlass_segment_gemm(
        workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_ld: torch.Tensor,
        w_ld: torch.Tensor,
        y_ld: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        module.cutlass_segment_gemm.default(
            workspace_buffer,
            all_problems,
            x_data,
            w_data,
            y_data,
            x_ld,
            w_ld,
            y_ld,
            empty_x_data,
            weight_column_major,
        )

    @register_fake_op("flashinfer::cutlass_segment_gemm")
    def _fake_cutlass_segment_gemm(
        workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_ld: torch.Tensor,
        w_ld: torch.Tensor,
        y_ld: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        pass

    # Register the module
    _gemm_module = SimpleNamespace(
        cublas_fp8_gemm_runner=cublas_fp8_gemm_runner,
        cutlass_segment_gemm=cutlass_segment_gemm,
    )

    return _gemm_module


def gen_gemm_sm100_module_cutlass_fp4() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm100_cutlass_fp4"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass.cu",
    ]

    with open(jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
        dtype_list = ["__nv_bfloat16", "half"]
        cta_m_n_k_list = [
            (128, 64, 128),
            (128, 256, 128),
            (128, 128, 256),
            (128, 256, 256),
        ]
        for cta_m, cta_n, cta_k in cta_m_n_k_list:
            for dtype in dtype_list:
                dest_path = (
                    gen_directory
                    / f"fp4_gemm_cutlass_{dtype}_{cta_m}_{cta_n}_{cta_k}.cu"
                )
                source_paths.append(dest_path)
                source = kernel_inst_templ.render(
                    type=dtype,
                    cta_m=cta_m,
                    cta_n=cta_n,
                    cta_k=cta_k,
                )
                write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )
    return gen_jit_spec(
        "fp4_gemm_cutlass",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
        ],
        extra_cflags=[
            "-DFAST_BUILD",
        ],
        extra_ldflags=["-lcuda"],
    )


def gen_gemm_sm120_module_cutlass_fp4() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm120_cutlass_fp4"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass_sm120.cu",
    ]

    with open(jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass_sm120.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
        dtype_list = ["__nv_bfloat16", "half"]
        # SM120/121 uses only 128x128x128 tile configuration with implied 1x1x1 cluster shape
        cta_m_n_k_list = [
            (128, 128, 128),
        ]
        for cta_m, cta_n, cta_k in cta_m_n_k_list:
            for dtype in dtype_list:
                dest_path = (
                    gen_directory
                    / f"fp4_gemm_cutlass_{dtype}_{cta_m}_{cta_n}_{cta_k}.cu"
                )
                source_paths.append(dest_path)
                source = kernel_inst_templ.render(
                    type=dtype,
                    cta_m=cta_m,
                    cta_n=cta_n,
                    cta_k=cta_k,
                )
                write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    return gen_jit_spec(
        "fp4_gemm_cutlass_sm120",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
        ],
        extra_cflags=[
            "-DFAST_BUILD",
        ],
        extra_ldflags=["-lcuda"],
    )


def gen_gemm_sm100_module_cutlass_fp8() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm100_cutlass_fp8"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "fp8_gemm_cutlass.cu",
    ]

    with open(jit_env.FLASHINFER_CSRC_DIR / "fp8_gemm_cutlass.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
        dtype_list = ["__nv_bfloat16", "half"]
        cta_m_n_k_list = [
            (64, 64, 128),
            (64, 128, 128),
            (64, 256, 128),
            (128, 64, 128),
            (128, 128, 128),
            (128, 256, 128),
        ]
        for cta_m, cta_n, cta_k in cta_m_n_k_list:
            for dtype in dtype_list:
                dest_path = (
                    gen_directory
                    / f"fp8_gemm_cutlass_{dtype}_{cta_m}_{cta_n}_{cta_k}.cu"
                )
                source_paths.append(dest_path)
                source = kernel_inst_templ.render(
                    type=dtype,
                    cta_m=cta_m,
                    cta_n=cta_n,
                    cta_k=cta_k,
                )
                write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )

    return gen_jit_spec(
        "fp8_gemm_cutlass",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
        ],
        extra_cflags=[
            "-DFAST_BUILD",
        ],
        extra_ldflags=["-lcuda"],
    )


def gen_gemm_sm100_module() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm100"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = []
    for prefix in ["gemm_groupwise", "group_gemm_fp8_groupwise"]:
        with open(
            jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm100_kernel_inst.jinja"
        ) as f:
            kernel_inst_templ = jinja2.Template(f.read())
        dtype_in_list = [torch.float8_e4m3fn, torch.float8_e5m2]
        dtype_out_list = [torch.float16, torch.bfloat16]
        scale_major_k_list = ["true", "false"]
        mma_sm_list = [1, 2]
        for dtype_in, dtype_out, scale_major_k, mma_sm in product(
            dtype_in_list, dtype_out_list, scale_major_k_list, mma_sm_list
        ):
            name_dtype_in = filename_safe_dtype_map[dtype_in]
            name_dtype_out = filename_safe_dtype_map[dtype_out]
            dest_path = (
                gen_directory
                / f"{prefix}_{name_dtype_in}_{name_dtype_out}_major{scale_major_k}_mma{mma_sm}_sm100.cu"
            )
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                dtype_in=dtype_cutlass_map[dtype_in],
                dtype_out=dtype_cutlass_map[dtype_out],
                scale_major_k=scale_major_k,
                mma_sm=mma_sm,
            )
            write_if_different(dest_path, source)
    prefix = "group_gemm_mxfp4_groupwise"
    with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm100_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
    dtype_a_list = [torch.float8_e4m3fn, torch.float8_e5m2]
    dtype_d_list = [torch.float16, torch.bfloat16]
    mma_sm_list = [1, 2]
    swap_ab_list = ["true", "false"]
    for dtype_a, dtype_d, mma_sm, swap_ab in product(
        dtype_a_list, dtype_d_list, mma_sm_list, swap_ab_list
    ):
        name_dtype_a = filename_safe_dtype_map[dtype_a]
        name_dtype_d = filename_safe_dtype_map[dtype_d]
        dest_path = (
            gen_directory
            / f"{prefix}_{name_dtype_a}_{name_dtype_d}_mma{mma_sm}_swap{swap_ab}_sm100.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_a=dtype_cutlass_map[dtype_a],
            dtype_b="cutlass::float_e2m1_t",
            dtype_d=dtype_cutlass_map[dtype_d],
            mma_sm=mma_sm,
            swap_ab=swap_ab,
        )
        write_if_different(dest_path, source)
    for filename in [
        "gemm_groupwise_sm100.cu",
        "group_gemm_fp8_groupwise_sm100.cu",
        "group_gemm_mxfp4_groupwise_sm100.cu",
        "gemm_sm100_pybind.cu",
        "group_gemm_sm100_pybind.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )
    return gen_jit_spec(
        "gemm_sm100",
        source_paths,
        extra_cuda_cflags=nvcc_flags,
    )


@functools.cache
def get_gemm_sm100_module():
    module = gen_gemm_sm100_module().build_and_load()

    return module


def gen_gemm_sm120_module() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm120"
    gen_directory.mkdir(parents=True, exist_ok=True)
    source_paths = []

    # Generate kernel instantiations following SM100's approach
    prefix = "gemm_groupwise"
    dtype_in_list = [torch.float8_e4m3fn, torch.float8_e5m2]
    dtype_out_list = [torch.float16, torch.bfloat16]
    scale_major_k_list = ["true", "false"]
    # SM120 uses fixed 128x128x128 tiles with Cooperative schedule

    with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm120_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    for dtype_in, dtype_out, scale_major_k in product(
        dtype_in_list,
        dtype_out_list,
        scale_major_k_list,
    ):
        name_dtype_in = filename_safe_dtype_map[dtype_in]
        name_dtype_out = filename_safe_dtype_map[dtype_out]
        dest_path = (
            gen_directory
            / f"{prefix}_{name_dtype_in}_{name_dtype_out}_major{scale_major_k}_sm120.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_in=dtype_cutlass_map[dtype_in],
            dtype_out=dtype_cutlass_map[dtype_out],
            scale_major_k=scale_major_k,
        )
        write_if_different(dest_path, source)

    # Generate group gemm kernel instantiations
    prefix = "group_gemm_fp8_groupwise"
    with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm120_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    for dtype_in, dtype_out, scale_major_k in product(
        dtype_in_list,
        dtype_out_list,
        scale_major_k_list,
    ):
        name_dtype_in = filename_safe_dtype_map[dtype_in]
        name_dtype_out = filename_safe_dtype_map[dtype_out]
        dest_path = (
            gen_directory
            / f"{prefix}_{name_dtype_in}_{name_dtype_out}_major{scale_major_k}_sm120.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_in=dtype_cutlass_map[dtype_in],
            dtype_out=dtype_cutlass_map[dtype_out],
            scale_major_k=scale_major_k,
        )
        write_if_different(dest_path, source)

    # Copy source files
    for filename in [
        "gemm_groupwise_sm120.cu",
        "group_gemm_fp8_groupwise_sm120.cu",
        "gemm_sm120_pybind.cu",
        "group_gemm_sm120_pybind.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[
            12,
        ]
    )

    return gen_jit_spec(
        "gemm_sm120",
        source_paths,
        extra_cuda_cflags=nvcc_flags,
    )


@functools.cache
def get_gemm_sm120_module():
    module = gen_gemm_sm120_module().build_and_load()
    return module


@functools.cache
def get_gemm_sm120_module_cutlass_fp8():
    """Get CUTLASS FP8 runner for SM120/SM121 using the groupwise scaling kernel."""
    module = get_gemm_sm120_module()

    def cutlass_fp8_gemm_runner():
        class CutlassFp8GemmRunner(TunableRunner):
            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                # For now, return a single default tactic
                return [-1]

            def forward(
                self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs,
            ) -> torch.Tensor:
                a, b, scale_a, scale_b, out, workspace_buffer = inputs

                # Handle both 2D (MM) and 3D (BMM) cases
                # SM120 kernel now supports batch operations natively
                if a.dim() == 2:
                    # 2D case: simple matrix multiplication
                    # Make B column-major for the kernel
                    b_col_major = b.transpose(-2, -1)
                else:
                    # 3D case: batch matrix multiplication
                    # B is already in the right format [batch, k, n] (column-major)
                    b_col_major = b

                # Determine dimensions first to know scale granularity
                if a.dim() == 2:
                    n_dim = b_col_major.shape[0]
                    m_dim = a.shape[0]
                    k_dim = a.shape[1]
                    batch_size = 1
                else:
                    n_dim = b_col_major.shape[2]  # BMM case: [batch, k, n]
                    m_dim = a.shape[1]
                    k_dim = a.shape[2]
                    batch_size = a.shape[0]

                # ScaleGranularityK must equal TileK (128)
                if k_dim < 128:
                    raise ValueError(
                        f"SM120/SM121 CUTLASS blockwise scaling requires k >= 128, got k={k_dim}. "
                    )

                scale_gran_m = 1
                scale_gran_n = 128
                scale_gran_k = 128

                # For scalar scales, create compatible shapes for SM120
                # SM120 requires scale tensors with specific shapes based on granularity
                # Scale shape should be [m/scale_gran_m, k/scale_gran_k] for A
                # and [n/scale_gran_n, k/scale_gran_k] for B
                if scale_a.numel() == 1:
                    scale_m_count = (
                        batch_size * m_dim + scale_gran_m - 1
                    ) // scale_gran_m
                    scale_k_count = (
                        k_dim + scale_gran_k - 1
                    ) // scale_gran_k  # k dimension
                    scale_a_expanded = scale_a.view(1, 1).expand(
                        scale_m_count, scale_k_count
                    )
                else:
                    scale_a_expanded = scale_a

                if scale_b.numel() == 1:
                    # Calculate the expected scale dimensions
                    scale_n_count = (
                        batch_size * n_dim + scale_gran_n - 1
                    ) // scale_gran_n
                    scale_k_count = (
                        k_dim + scale_gran_k - 1
                    ) // scale_gran_k  # k dimension
                    scale_b_expanded = scale_b.view(1, 1).expand(
                        scale_n_count, scale_k_count
                    )
                else:
                    scale_b_expanded = scale_b

                # Call SM120 gemm_fp8_nt_groupwise (now handles both 2D and 3D)
                module.gemm_fp8_nt_groupwise.default(
                    workspace_buffer,
                    a,
                    b_col_major,
                    scale_a_expanded,
                    scale_b_expanded,
                    out,
                    scale_gran_m,  # scale_granularity_m
                    scale_gran_n,  # scale_granularity_n
                    scale_gran_k,  # scale_granularity_k (adjusted for small k)
                    "MN",  # scale_major_mode
                )
                return out

        return CutlassFp8GemmRunner()

    # Register the module
    return SimpleNamespace(
        cutlass_fp8_gemm_runner=cutlass_fp8_gemm_runner,
    )


def gen_trtllm_gen_gemm_module() -> JitSpec:
    # Fetch "flashinferMetaInfo.h" from the online kernel cache. This file
    # contains the `tllmGenGemmList` as the list of available kernels online.
    # It is included when compiling `trtllm_gemm_runner.cu`.
    include_path = f"{ArtifactPath.TRTLLM_GEN_GEMM}/include"
    header_name = "flashinferMetaInfo"

    # use `get_cubin` to get "flashinferMetaInfo.h"
    metainfo = get_cubin(
        f"{include_path}/{header_name}",
        MetaInfoHash.TRTLLM_GEN_GEMM,
        ".h",
    )
    # make sure "flashinferMetaInfo.h" is downloaded or cached
    assert metainfo, f"{header_name}.h not found"
    return gen_jit_spec(
        "trtllm_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_ENABLE_CUDA",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_GEMM}\\"',
        ]
        + sm100a_nvcc_flags,
        # link "include" sub-directory in cache
        extra_include_paths=[jit_env.FLASHINFER_CUBIN_DIR / include_path],
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def get_trtllm_gemm_module():
    mod = gen_trtllm_gen_gemm_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


@functools.cache
def get_gemm_sm100_module_cutlass_fp8():
    module = gen_gemm_sm100_module_cutlass_fp8().build_and_load()

    def cutlass_fp8_gemm_runner():
        class CutlassFp8GemmRunner(TunableRunner):
            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                return list(range(module.fp8_gemm_tactic_num()))

            def forward(
                self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs,
            ) -> torch.Tensor:
                a, b, scale_a, scale_b, out, workspace_buffer = inputs
                module.fp8_gemm.default(
                    a,
                    b.transpose(-2, -1),
                    scale_a,
                    scale_b,
                    out,
                    workspace_buffer,
                    tactic,
                )
                return out

        return CutlassFp8GemmRunner()

    # Register the module
    return SimpleNamespace(
        cutlass_fp8_gemm_runner=cutlass_fp8_gemm_runner,
    )


def fp8_gemm_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    runner_names: List[str],
) -> None:
    runners = []
    # No e5m2 for cutlass
    is_e5m2 = a.dtype == torch.float8_e5m2 or b.dtype == torch.float8_e5m2
    is_sm_supported = _match_sm_version(a.device, ["100", "103", "110"])
    is_sm120_supported = _match_sm_version(a.device, ["120", "121"])

    if "cutlass" in runner_names and not is_e5m2:
        if is_sm_supported:
            runners.append(
                get_gemm_sm100_module_cutlass_fp8().cutlass_fp8_gemm_runner()
            )
        elif is_sm120_supported:
            k_dim = a.shape[-1] if a.dim() == 2 else a.shape[2]
            if k_dim >= 128:
                runners.append(
                    get_gemm_sm120_module_cutlass_fp8().cutlass_fp8_gemm_runner()
                )
    if "cublas" in runner_names:
        runners.append(get_gemm_module().cublas_fp8_gemm_runner())
    if CUDNN_AVAILABLE and "cudnn" in runner_names:
        runners.append(_cudnn_gemm_fp8_runner())

    if len(runners) == 0:
        major, minor = get_compute_capability(torch.device("cuda"))
        raise ValueError(f"No valid runner found for current device sm{major}{minor}")

    tuner = AutoTuner.get()
    a_tensor_index = 0
    out_tensor_index = 4
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (a_tensor_index,),
                (-2,),
                get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2,
            ),
        ),
        constraint_specs=(
            ConstraintSpec(
                out_tensor_index, -2, lambda shapes: shapes[a_tensor_index][-2]
            ),
        ),
    )

    inputs = [a, b, scale_a, scale_b, out, workspace_buffer]
    runner, tactic = tuner.choose_one(
        "fp8_gemm",
        runners,
        tuning_config,
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)


def _create_cutlass_fp4_gemm_module(module, op_name: str, tuner_name: str):
    """Helper function to create cutlass FP4 GEMM module."""

    class CutlassFp4GemmRunner(TunableRunner):
        def __init__(self):
            self._fp4_gemm_runner = module.fp4_gemm

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return list(range(module.fp4_gemm_tactic_num()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            a, b, a_descale, b_descale, alpha, out, workspace_buffer = inputs
            module.fp4_gemm.default(
                a, b, a_descale, b_descale, alpha, out, workspace_buffer, tactic
            )
            return out

    @register_custom_op(
        op_name,
        mutates_args=(""),
    )
    def cutlass_fp4_gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        a_descale: torch.Tensor,
        b_descale: torch.Tensor,
        alpha: torch.Tensor,
        out: torch.Tensor,
        workspace_buffer: torch.Tensor,
    ):
        tuner = AutoTuner.get()

        a_tensor_index = 0
        a_scale_tensor_index = 2
        out_tensor_index = 5

        def pad_up(x, y):
            return ((x + y - 1) // y) * y

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (a_tensor_index,),
                    (0,),
                    get_last_power_of_2_num_tokens_buckets,
                    last_positive_power_of_2,
                ),
            ),
            constraint_specs=(
                ConstraintSpec(
                    a_scale_tensor_index,
                    0,
                    lambda shapes: pad_up(shapes[a_tensor_index][0], 128),
                ),
                ConstraintSpec(
                    out_tensor_index, 0, lambda shapes: shapes[a_tensor_index][0]
                ),
            ),
        )

        fp4_runner = CutlassFp4GemmRunner()

        inputs = [a, b, a_descale, b_descale, alpha, out, workspace_buffer]
        _, tactic = tuner.choose_one(
            tuner_name,
            [fp4_runner],
            tuning_config,
            inputs,
        )

        fp4_runner(inputs=inputs, tactic=tactic)

    return SimpleNamespace(
        cutlass_fp4_gemm=cutlass_fp4_gemm,
    )


@functools.cache
def get_gemm_sm100_module_cutlass_fp4():
    """Get the SM100/103/110 FP4 GEMM module."""
    module = gen_gemm_sm100_module_cutlass_fp4().build_and_load()
    return _create_cutlass_fp4_gemm_module(
        module, "flashinfer::cutlass_fp4_gemm", "cutlass_fp4_gemm"
    )


@functools.cache
def get_gemm_sm120_module_cutlass_fp4():
    """Get the SM120/121 FP4 GEMM module."""
    module = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
    return _create_cutlass_fp4_gemm_module(
        module, "flashinfer::cutlass_fp4_gemm_sm120", "cutlass_fp4_gemm_sm120"
    )


def gen_gemm_sm100_module_tgv(dtype: torch.dtype = torch.bfloat16) -> JitSpec:
    """
    Generate TGV GEMM module for SM100 architecture.

    Args:
        dtype: Data type for the GEMM operation (torch.bfloat16 or torch.float16)

    Returns:
        JitSpec for the TGV GEMM module
    """
    if dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported dtype {dtype}. Only bfloat16 and float16 are supported."
        )

    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    module_name = f"tgv_gemm_{dtype_str}"

    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / f"gen_tgv_gemm_{dtype_str}"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "tgv_gemm.cu",
    ]

    # Read the Jinja template
    with open(jit_env.FLASHINFER_CSRC_DIR / "tgv_gemm.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    # Define tile size configurations (cta_m, cta_n, dma_stages)
    cta_m_n_dma_list = [
        (64, 8, 6),
        (64, 8, 8),
        (64, 8, 10),
        (64, 8, 12),
        (64, 16, 6),
        (64, 16, 8),
        (64, 16, 10),
        (64, 32, 6),
        (64, 32, 8),
        (64, 64, 6),
        (128, 16, 6),
    ]

    # Generate instances for the specified dtype
    for cta_m, cta_n, dma_stage in cta_m_n_dma_list:
        dest_path = (
            gen_directory / f"tgv_gemm_{dtype_str}_{cta_m}x{cta_n}_{dma_stage}.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            cta_m=cta_m, cta_n=cta_n, dma_stage=dma_stage, dtype=dtype_str
        )
        write_if_different(dest_path, source)

    return gen_jit_spec(
        module_name,
        source_paths,
        extra_cuda_cflags=sm100a_nvcc_flags,
        extra_include_paths=[
            jit_env.FLASHINFER_INCLUDE_DIR,
            jit_env.FLASHINFER_CSRC_DIR,
        ],
    )


@functools.cache
def get_gemm_sm100_module_tgv(dtype: torch.dtype = torch.bfloat16):
    """
    Get and build the TGV GEMM module for the specified dtype.

    Args:
        dtype: Data type for the GEMM operation (torch.bfloat16 or torch.float16)

    Returns:
        SimpleNamespace with the runner function
    """
    module = gen_gemm_sm100_module_tgv(dtype).build_and_load()

    def tgv_gemm_runner():
        class TGVGemmRunner(TunableRunner):
            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                # Return all available TGV configurations
                # Based on the configurations in tgv_gemm_configs.h
                tactic_fn = module.tgv_gemm_tactic_num
                return list(range(tactic_fn()))

            def forward(
                self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs,
            ) -> torch.Tensor:
                a, b, bias = inputs
                pdl = kwargs.get("pdl", False)
                # swap gemm m and n by swapping b and a
                # tgv_gemm takes mat1 as weights and mat2 as input tensor
                # from [m,k]x[k,n]+[n,] to [n,k]x[k,m]+[n,]
                gemm_fn = module.tgv_gemm
                out = gemm_fn.default(b.t(), a.t(), bias, tactic, pdl)
                return out.t()

        return TGVGemmRunner()

    # Register the module
    return SimpleNamespace(
        tgv_gemm_runner=tgv_gemm_runner,
    )


def tgv_gemm_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    pdl: bool = False,
) -> torch.Tensor:
    """
    Perform TGV GEMM on SM100 architecture with automatic dtype detection.

    Computes: A @ B + bias

    Args:
        a: First input tensor of shape (M, K) in row-major layout
        b: Second input tensor of shape (K, N) in column-major layout
        bias: Bias tensor of shape (N,)
        pdl: Whether to use PDL (persistent data loader), defaults to False

    Returns:
        Output tensor of shape (M, N)

    Supported dtypes:
        - torch.bfloat16
        - torch.float16

    Note:
        - Requires SM100, SM103, or SM110 architecture
        - Input tensors a and b must have the same dtype
        - Tensor b is expected to be in column-major layout (transposed from typical PyTorch row-major)
    """
    # Verify SM100 architecture support
    if not _match_sm_version(a.device, ["100", "103", "110"]):
        raise ValueError("TGV GEMM requires SM100, SM103, or SM110 architecture")

    # Verify dtype support
    if a.dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported dtype {a.dtype}. Only bfloat16 and float16 are supported."
        )

    if a.dtype != b.dtype:
        raise ValueError(
            f"Input tensors must have the same dtype. Got {a.dtype} and {b.dtype}."
        )

    runners = []
    runners.append(get_gemm_sm100_module_tgv(a.dtype).tgv_gemm_runner())

    tuner = AutoTuner.get()
    a_tensor_index = 0
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (a_tensor_index,),
                (-2,),
                get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2,
            ),
        ),
        constraint_specs=(),
    )

    inputs = [a, b, bias]
    dtype_str = "bf16" if a.dtype == torch.bfloat16 else "fp16"
    runner, tactic = tuner.choose_one(
        f"{dtype_str}_tgv_gemm",
        runners,
        tuning_config,
        inputs,
    )

    return runner(inputs=inputs, tactic=tactic, pdl=pdl)


def gen_gemm_sm90_module() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm90"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = []
    with open(jit_env.FLASHINFER_CSRC_DIR / "group_gemm_sm90_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
    for dtype_in, dtype_out in [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
        (torch.float8_e5m2, torch.float16),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e5m2, torch.bfloat16),
    ]:
        name_dtype_in = filename_safe_dtype_map[dtype_in]
        name_dtype_out = filename_safe_dtype_map[dtype_out]
        dest_path = (
            gen_directory / f"group_gemm_{name_dtype_in}_{name_dtype_out}_sm90.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_in=dtype_cutlass_map[dtype_in],
            dtype_out=dtype_cutlass_map[dtype_out],
        )
        write_if_different(dest_path, source)
    for filename in [
        "group_gemm_sm90.cu",
        "flashinfer_gemm_sm90_ops.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)
    return gen_jit_spec(
        "gemm_sm90",
        source_paths,
        extra_cuda_cflags=sm90a_nvcc_flags,
    )


@functools.cache
def get_gemm_sm90_module():
    module = gen_gemm_sm90_module().build_and_load()

    # torch library for cutlass_segment_gemm_sm90

    @register_custom_op(
        "flashinfer::cutlass_segment_gemm_sm90",
        mutates_args=("workspace_buffer", "y"),
    )
    def cutlass_segment_gemm_sm90(
        workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_stride: torch.Tensor,
        w_stride: torch.Tensor,
        y_stride: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        empty_y_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        module.cutlass_segment_gemm_sm90.default(
            workspace_buffer,
            int_workspace_buffer,
            all_problems,
            x_data,
            w_data,
            y_data,
            x_stride,
            w_stride,
            y_stride,
            empty_x_data,
            empty_y_data,
            weight_column_major,
        )

    @register_fake_op("flashinfer::cutlass_segment_gemm_sm90")
    def _fake_cutlass_segment_gemm_sm90(
        workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        all_problems: torch.Tensor,
        x_data: torch.Tensor,
        w_data: torch.Tensor,
        y_data: torch.Tensor,
        x_stride: torch.Tensor,
        w_stride: torch.Tensor,
        y_stride: torch.Tensor,
        y: torch.Tensor,
        empty_x_data: torch.Tensor,
        empty_y_data: torch.Tensor,
        weight_column_major: bool,
    ) -> None:
        pass

    # Register the module
    return SimpleNamespace(
        cutlass_segment_gemm_sm90=cutlass_segment_gemm_sm90,
    )


def launch_compute_sm80_group_gemm_args(
    x: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    w_column_major: bool,
    batch_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: Optional[torch.Tensor] = None,
):
    device = x.device
    prob_type = torch.int32  # problem sizes -> int
    ptr_type = torch.int64  # pointers -> int64_t
    ld_type = torch.int64  # strides -> int64_t

    seg_indptr = seg_indptr.to(ptr_type)
    if weight_indices is not None:
        weight_indices = weight_indices.to(ptr_type)

    d_out = weights.size(1) if w_column_major else weights.size(2)
    d_in = weights.size(2) if w_column_major else weights.size(1)

    all_problems = torch.empty((batch_size, 3), dtype=prob_type, device=device)

    x_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    w_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    y_data = torch.empty(batch_size, dtype=ptr_type, device=device)

    x_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)
    w_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)
    y_stride_data = torch.empty(batch_size, dtype=ld_type, device=device)

    from .triton.gemm import compute_sm80_group_gemm_args

    compute_sm80_group_gemm_args[(batch_size,)](
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
        x,
        weights,
        y,
        seg_indptr,
        weight_indices,
        d_in,
        d_out,
        w_column_major,
    )

    return (
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
    )


def launch_compute_sm90_group_gemm_args(
    x: torch.Tensor,
    weights: torch.Tensor,
    y: torch.Tensor,
    w_column_major: bool,
    batch_size: int,
    seg_indptr: torch.Tensor,
    weight_indices: Optional[torch.Tensor] = None,
):
    device = x.device
    prob_type = torch.int32  # problem sizes -> int
    ptr_type = torch.int64  # pointers -> int64_t
    stride_type = torch.int64  # strides -> int64_t

    seg_indptr = seg_indptr.to(ptr_type)
    if weight_indices is not None:
        weight_indices = weight_indices.to(ptr_type)

    d_out = weights.size(1) if w_column_major else weights.size(2)
    d_in = weights.size(2) if w_column_major else weights.size(1)

    all_problems = torch.empty((batch_size, 3), dtype=prob_type, device=device)

    x_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    w_data = torch.empty(batch_size, dtype=ptr_type, device=device)
    y_data = torch.empty(batch_size, dtype=ptr_type, device=device)

    x_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)
    w_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)
    y_stride_data = torch.empty(batch_size, dtype=stride_type, device=device)

    from .triton.gemm import compute_sm90_group_gemm_args

    compute_sm90_group_gemm_args[(batch_size,)](
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
        x,
        weights,
        y,
        seg_indptr,
        weight_indices,
        d_in,
        d_out,
        w_column_major,
    )

    return (
        all_problems,
        x_data,
        w_data,
        y_data,
        x_stride_data,
        w_stride_data,
        y_stride_data,
    )


class SegmentGEMMWrapper:
    r"""Wrapper for segment GEMM kernels.

    Example
    -------
    >>> import torch
    >>> from flashinfer import SegmentGEMMWrapper
    >>> # create a 1MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    >>> segment_gemm = SegmentGEMMWrapper(workspace_buffer)
    >>> seq_lens = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    >>> # create packed input tensor (10 = 1 + 2 + 3 + 4)
    >>> x = torch.randn(10, 128, device="cuda", dtype=torch.float16)
    >>> # create weight tensor with 4 weights, each with 128 input and 256 output channels, column major
    >>> weights = torch.randn(4, 256, 128, device="cuda", dtype=torch.float16)
    >>> # compute the segment GEMM
    >>> y = segment_gemm.run(x, weights, 4, True, seg_lens=seq_lens)
    >>> y.shape
    torch.Size([10, 256])
    >>> y_ref_0 = torch.matmul(x[:1], weights[0].t())
    >>> torch.allclose(y[:1], y_ref_0)
    True
    >>> y_ref_1 = torch.matmul(x[1:3], weights[1].t())
    >>> torch.allclose(y[1:3], y_ref_1)
    True
    >>> y_ref_2 = torch.matmul(x[3:6], weights[2].t())
    >>> torch.allclose(y[3:6], y_ref_2)
    True
    >>> y_ref_3 = torch.matmul(x[6:], weights[3].t())
    >>> torch.allclose(y[6:], y_ref_3)
    True
    >>>
    >>> # another example with weight indices
    >>> weight_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
    >>> y = segment_gemm.run(x, weights, 4, True, seg_lens=seq_lens, weight_indices=weight_indices)
    >>> y.shape
    torch.Size([10, 256])
    >>> y_ref_0 = torch.matmul(x[:1], weights[0].t())
    >>> torch.allclose(y[:1], y_ref_0)
    True
    >>> y_ref_1 = torch.matmul(x[1:3], weights[1].t())
    >>> torch.allclose(y[1:3], y_ref_1)
    True
    >>> y_ref_2 = torch.matmul(x[3:6], weights[0].t())
    >>> torch.allclose(y[3:6], y_ref_2)
    True
    >>> y_ref_3 = torch.matmul(x[6:], weights[1].t())
    >>> torch.allclose(y[6:], y_ref_3)
    True
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, backend: str = "auto"
    ) -> None:
        r"""Initialize the wrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The workspace buffer for the kernels, we use it for storing intermediate results in cutlass
            segment GEMM kernels. Encouraged size is 128MB.
        """
        self._int_workspace_buffer = torch.empty(
            (1024 * 1024,), dtype=torch.int8, device=float_workspace_buffer.device
        )
        self._float_workspace_buffer = float_workspace_buffer
        self.backend = backend

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer for the kernels.
        int_workspace_buffer : torch.Tensor
            The new int workspace buffer for the kernels.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer

    def run(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        out: Optional[torch.Tensor] = None,
        seg_lens: Optional[torch.Tensor] = None,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the segment GEMM kernel.

        Compute the matrix multiplication between a batch of input tensor (with variable number of rows, but fixed
        number of columns) and a batch of weight tensor with fixed number of rows and columns:

        .. math::

            y[i] = x[i] \times W[i]

        if :attr:`weight_indices` is provided, we will select the weight tensor based on the indices in the
        :attr:`weight_indices` tensor:

        .. math::

            y[i] = x[i] \times W[\text{weight_indices}[i]]

        We use Ragged Tensor to represent the input tensor :attr:`x` and the output tensor :attr:`y`, and each x[i]
        is a segment of the concatenated tensor. Please see :ref:`Ragged Tensor tutorial <kv-layout>` for more details.
        We use a ``seg_len`` or ``seg_indptr`` tensor (either would work) to indicate the start and end of each segment,
        where the ``seg_indptr`` is the cumulative sum of the ``seg_lens`` tensor (with an additional 0 at the beginning):

        .. math::

            \text{seg_indptr}[i] = \sum_{j=0}^{i-1} \text{seg_lens}[j], \quad \text{seg_indptr}[0] = 0

        - If ``seg_lens`` is provided, then :attr:`x` has shape ``(sum(seg_lens), d_in)`` and :attr:`y` has shape
            ``(sum(seg_lens), d_out)``, where ``d_in`` is the number of columns of the input tensor and ``d_out`` is the
            number of columns of the output tensor.
        - If ``seg_indptr`` is provided, then :attr:`x` has shape ``(seg_indptr[-1], d_in)`` and :attr:`y` has shape
            ``(seg_indptr[-1], d_out)``.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape ``(sum(seg_lens), d_in)``.
        weights : torch.Tensor
            The 3D weight tensor with shape ``(num_weights, d_in, d_out)`` if :attr:`weight_column_major` is ``False``,
            or ``(num_weights, d_out, d_in)`` if :attr:`weight_column_major` is ``True``.
        batch_size : int
            The number of segments.
        weight_column_major : bool
            Whether the weight tensor is column major.
        out : Optional[torch.Tensor]
            The output tensor, with shape ``(sum(seg_lens), d_out)``.
            If not provided, a new tensor will be created internally.
        seg_lens : Optional[torch.Tensor]
            The length of each segment, with shape ``(batch_size,)``, expects a 1D tensor of dtype ``torch.int64``.
        seg_indptr : Optional[torch.Tensor]
            The indptr of the segments, with shape ``(batch_size + 1,)``, expects a 1D tensor of dtype ``torch.int64``.
            If this is provided, then :attr:`seg_lens` will be ignored, otherwise ``seg_indptr`` will be computed
            internally from :attr:`seg_lens`.
        weight_indices : Optional[torch.Tensor]
            The indices of the weight tensor to be selected for each segment, with shape ``(batch_size,)``.
            Expects a 1D tensor of dtype ``torch.int64``.
            If this is provided, then the weight tensor will be selected based on the indices in this tensor.

        Returns
        -------
        torch.Tensor
            The output tensor with shape ``(sum(seg_lens), d_out)``.
        """
        if seg_lens is None and seg_indptr is None:
            raise ValueError("Either seg_lens or seg_indptr should be provided.")
        if seg_indptr is None:
            seg_indptr = get_indptr(seg_lens.to(x))
        if weight_indices is None:
            # create an empty CPU tensor as placeholder
            weight_indices = torch.empty(0, dtype=torch.int64)
        cumulative_batch_size = x.size(0)
        d_out = weights.size(1) if weight_column_major else weights.size(2)
        if out is None:
            if is_float8(x):
                out_dtype = torch.bfloat16
            else:
                out_dtype = x.dtype
            out = torch.zeros(
                (cumulative_batch_size, d_out), dtype=out_dtype, device=x.device
            )
        else:
            if out.shape != (cumulative_batch_size, d_out):
                raise ValueError(
                    f"Output tensor shape mismatch, expected {cumulative_batch_size, d_out}, got {out.shape}"
                )
        empty_x_data = torch.empty(0, dtype=x.dtype, device=x.device)
        empty_y_data = torch.empty(0, dtype=out.dtype, device=out.device)

        if self.backend == "auto":
            backend = determine_gemm_backend(x.device)
        else:
            backend = self.backend

        if backend == "sm90":
            (
                all_problems,
                x_data,
                w_data,
                y_data,
                x_stride_data,
                w_stride_data,
                y_stride_data,
            ) = launch_compute_sm90_group_gemm_args(
                x,
                weights,
                out,
                weight_column_major,
                batch_size,
                seg_indptr,
                weight_indices,
            )
            get_gemm_sm90_module().cutlass_segment_gemm_sm90(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_stride_data,
                w_stride_data,
                y_stride_data,
                out,  # for torch compile mutates_args
                empty_x_data,  # for kernel type dispatch
                empty_y_data,
                weight_column_major,
            )
        elif backend == "sm80":
            (
                all_problems,
                x_data,
                w_data,
                y_data,
                x_ld_data,
                w_ld_data,
                y_ld_data,
            ) = launch_compute_sm80_group_gemm_args(
                x,
                weights,
                out,
                weight_column_major,
                batch_size,
                seg_indptr,
                weight_indices,
            )
            get_gemm_module().cutlass_segment_gemm(
                self._int_workspace_buffer,
                all_problems,
                x_data,
                w_data,
                y_data,
                x_ld_data,
                w_ld_data,
                y_ld_data,
                out,
                empty_x_data,
                weight_column_major,
            )
        else:
            raise ValueError(f"Unsupported gemm backend: {backend}")
        return out

    forward = run


class UIDs(Enum):
    """UIDs for CUDNN graph tensors"""

    A_UID = 0
    B_UID = 1
    ALPHA_UID = 2
    BLOCK_DESCALE_A_UID = 3
    BLOCK_DESCALE_B_UID = 4
    A_SCALE_UID = 5
    B_SCALE_UID = 6
    O_UID = 7


def _check_cudnn_availability():
    """Check if cuDNN is available and raise exception if not."""
    if not CUDNN_AVAILABLE:
        raise RuntimeError(
            "cuDNN is not available. Please install cuDNN to use FP8 GEMM functions. "
            "You can install it with: pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend"
        )


def _check_cudnn_fp4_availability():
    """Check if cuDNN FP4 support is available and raise exception if not."""
    _check_cudnn_availability()

    # Check cuDNN version for FP4 support (requires 1.13.* or later)
    try:
        version_str = cudnn.__version__
        major, minor = map(int, version_str.split(".")[:2])

        if (major, minor) < (1, 13):
            raise RuntimeError(
                f"cuDNN FP4 requires version 1.13+, found {version_str}. "
                f"Upgrade: pip install --upgrade nvidia-cudnn-cu12 nvidia-cudnn-frontend"
            )
    except (ImportError, AttributeError, ValueError, IndexError) as e:
        raise RuntimeError(
            "Unable to determine cuDNN version. FP4 requires cuDNN 1.13+."
        ) from e

    # Check cuDNN backend version for FP4 support (requires >= 91002)
    try:
        backend_version = cudnn.backend_version()
        if backend_version < 91002:
            raise RuntimeError(
                f"cuDNN FP4 requires backend version >= 91002, found {backend_version}. "
                f"Please upgrade cuDNN backend."
            )
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            "Unable to determine cuDNN backend version. FP4 requires backend >= 91002."
        ) from e


def _is_cublas_fp4_available_in_cudnn():
    """Check if cuBLAS backend for FP4 GEMM is available in cuDNN."""
    _check_cudnn_availability()

    # Check cuDNN backend version for FP4 support (requires cudnn_version == 9.11.1 or cudnn_version >= 9.13)
    backend_version = cudnn.backend_version()
    CUDNN_VERSION_9_11_1 = 91101
    CUDNN_VERSION_9_13_0 = 91300
    return (
        backend_version == CUDNN_VERSION_9_11_1
        or backend_version >= CUDNN_VERSION_9_13_0
    )


def _get_native_fp4_dtype():
    """get native fp4 datatype if supported in the torch, otherwise return uint8."""
    if hasattr(torch, "float4_e2m1fn_x2"):
        return torch.float4_e2m1fn_x2
    else:
        return torch.uint8


# Global cudnn handle. need to make it per device in future
_cudnn_handle = None


def _get_cudnn_handle(stream: torch.cuda.Stream):
    """Create and return a cached cuDNN handle."""
    global _cudnn_handle
    if _cudnn_handle is None:
        _check_cudnn_availability()
        _cudnn_handle = cudnn.create_handle()
    cudnn.set_stream(_cudnn_handle, stream.cuda_stream)
    return _cudnn_handle


def _validate_fp8_output_dtype(dtype: torch.dtype):
    """Validate that the output dtype is either bf16 or fp16."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Unsupported output dtype: {dtype}. "
            f"Only torch.bfloat16 and torch.float16 are supported for FP8 GEMM operations."
        )


@functools.cache
def build_cudnn_gemm_block_scale_dequantize_graph(
    a_shape,
    a_stride,
    b_shape,
    b_stride,
    a_descale_shape,
    a_descale_stride,
    b_descale_shape,
    b_descale_stride,
    ab_type,
    o_type,
    block_size,
    device,
    alpha,
    use_nvfp4,
):
    _check_cudnn_availability()
    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(stream)) as (graph, _):
        scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

        a_cudnn_tensor = graph.tensor(
            name="a", dim=a_shape, stride=a_stride, data_type=ab_type
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=b_shape, stride=b_stride, data_type=ab_type
        )
        block_descale_a_cudnn_tensor = graph.tensor(
            name="block_descale_a",
            dim=a_descale_shape,
            stride=a_descale_stride,
            data_type=scale_type,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )
        block_descale_b_cudnn_tensor = graph.tensor(
            name="block_descale_b",
            dim=b_descale_shape,
            stride=b_descale_stride,
            data_type=scale_type,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

        dequant_a_tensor = graph.block_scale_dequantize(
            a_cudnn_tensor,
            block_descale_a_cudnn_tensor,
            block_size=[1, block_size],
            name="dequant_a",
        )
        dequant_a_tensor.set_data_type(cudnn.data_type.FLOAT)
        dequant_b_tensor = graph.block_scale_dequantize(
            b_cudnn_tensor,
            block_descale_b_cudnn_tensor,
            block_size=[block_size, 1],
            name="dequant_b",
        )
        dequant_b_tensor.set_data_type(cudnn.data_type.FLOAT)
        c_tensor = graph.matmul(
            dequant_a_tensor,
            dequant_b_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="gemm",
        )
        c_tensor.set_data_type(cudnn.data_type.FLOAT)

        c_final_cudnn_tensor = c_tensor

        if alpha is not None:
            global_scale_cudnn_tensor = graph.tensor(
                name="global_scale",
                dim=(1, 1, 1),
                stride=(1, 1, 1),
                data_type=cudnn.data_type.FLOAT,
            )
            c_final_cudnn_tensor = graph.mul(
                name="scale_mul",
                a=c_tensor,
                b=global_scale_cudnn_tensor,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            global_scale_cudnn_tensor.set_uid(UIDs.ALPHA_UID.value)

        c_final_cudnn_tensor.set_name("c_final").set_output(True).set_data_type(o_type)

        a_cudnn_tensor.set_uid(UIDs.A_UID.value)
        b_cudnn_tensor.set_uid(UIDs.B_UID.value)
        block_descale_a_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_A_UID.value)
        block_descale_b_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_B_UID.value)
        c_final_cudnn_tensor.set_uid(UIDs.O_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])

        # WAR: The alpha (contains the global scale) is not supported by the cuBLAS backend (eng0)
        # in older cuDNN versions, so we deselect it.
        if (alpha is not None) and (not _is_cublas_fp4_available_in_cudnn()):
            graph.deselect_engines(["eng0"])
        graph.check_support()
        graph.build_plans()

        return graph


def execute_cudnn_gemm_fp4_graph(
    graph,
    a,
    b,
    a_descale,
    b_descale,
    alpha,
    c_final,
    workspace_buffer,
):
    variant_pack = {
        UIDs.A_UID.value: a.view(_get_native_fp4_dtype()),
        UIDs.B_UID.value: b.view(_get_native_fp4_dtype()),
        UIDs.BLOCK_DESCALE_A_UID.value: a_descale,
        UIDs.BLOCK_DESCALE_B_UID.value: b_descale,
        UIDs.O_UID.value: c_final,
    }

    if alpha is not None:
        variant_pack[UIDs.ALPHA_UID.value] = alpha.view(torch.float)

    if workspace_buffer.numel() < graph.get_workspace_size():
        workspace_buffer = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    stream = torch.cuda.current_stream(a.device)

    graph.execute(variant_pack, workspace_buffer, handle=_get_cudnn_handle(stream))


@functools.cache
def build_cudnn_gemm_with_per_tensor_q_graph(
    a_shape, a_stride, b_shape, b_stride, a_type, b_type, o_type, device
):
    """Build a cuDNN graph for GEMM with per-tensor quantization.

    This function is cached to avoid rebuilding identical graphs.

    Args:
        a_shape: Shape of tensor A
        a_stride: Stride of tensor A
        b_shape: Shape of tensor B
        b_stride: Stride of tensor B
        a_type: Data type for input tensor A
        b_type: Data type for input tensor B
        o_type: Data type for output tensor

    Returns:
        cuDNN graph object
    """
    _check_cudnn_availability()

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(stream)) as (graph, _):
        a_cudnn_tensor = graph.tensor(
            name="a", dim=a_shape, stride=a_stride, data_type=a_type
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=b_shape, stride=b_stride, data_type=b_type
        )
        a_scale_cudnn_tensor = graph.tensor(
            name="a_scale",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        b_scale_cudnn_tensor = graph.tensor(
            name="b_scale",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        c_cudnn_tensor = graph.matmul(
            name="matmul",
            A=a_cudnn_tensor,
            B=b_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        c_cudnn_tensor.set_name("c").set_data_type(cudnn.data_type.FLOAT)
        c_after_scale_a_cudnn_tensor = graph.mul(
            name="scale_mul_a",
            a=c_cudnn_tensor,
            b=a_scale_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        c_after_scale_b_cudnn_tensor = graph.mul(
            name="scale_mul_b",
            a=c_after_scale_a_cudnn_tensor,
            b=b_scale_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        c_after_scale_b_cudnn_tensor.set_name("c_final").set_output(True).set_data_type(
            o_type
        )

        a_cudnn_tensor.set_uid(UIDs.A_UID.value)
        b_cudnn_tensor.set_uid(UIDs.B_UID.value)
        a_scale_cudnn_tensor.set_uid(UIDs.A_SCALE_UID.value)
        b_scale_cudnn_tensor.set_uid(UIDs.B_SCALE_UID.value)
        c_after_scale_b_cudnn_tensor.set_uid(UIDs.O_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()

        return graph


def execute_cudnn_gemm_with_per_tensor_q_graph(
    graph, a, b, a_scale, b_scale, c_final, workspace
):
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.A_SCALE_UID.value: a_scale,
        UIDs.B_SCALE_UID.value: b_scale,
        UIDs.O_UID.value: c_final,
    }

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(stream)

    if workspace.numel() < graph.get_workspace_size():
        workspace = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    graph.execute(variant_pack, workspace, handle=cudnn_handle)


def _torch_data_type_to_cudnn_data_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif dtype == torch.float16:
        return cudnn.data_type.HALF
    elif dtype == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif dtype == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _cudnn_gemm_fp8(
    workspace: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: Optional[torch.Tensor],
    torch_out_dtype: torch.dtype,
):
    _check_cudnn_availability()

    graph = build_cudnn_gemm_with_per_tensor_q_graph(
        a.shape,
        a.stride(),
        b.shape,
        b.stride(),
        _torch_data_type_to_cudnn_data_type(a.dtype),
        _torch_data_type_to_cudnn_data_type(b.dtype),
        _torch_data_type_to_cudnn_data_type(torch_out_dtype),
        a.device,
    )

    execute_cudnn_gemm_with_per_tensor_q_graph(
        graph, a, b, a_scale, b_scale, out, workspace
    )
    return out


def _cudnn_gemm_fp8_runner():
    class CudnnFp8GemmRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            # cudnn has heuristic for fp8 gemm, so we only need to use the default tactic
            return [0]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, scale_a, scale_b, out, workspace_buffer = inputs
            _cudnn_gemm_fp8(workspace_buffer, a, b, scale_a, scale_b, out, out.dtype)
            return out

    return CudnnFp8GemmRunner()


def _get_real_fp4_shape_from_packed_uint8(packed_fp4_tensor):
    # the FP4 data are packed into uint8, we need to expand the shape and stride information to get the real shape and stride to be used in the cuDNN graph.
    is_column_major = packed_fp4_tensor.stride(-2) == 1
    real_shape = list(packed_fp4_tensor.shape)
    real_stride = list(packed_fp4_tensor.stride())

    # this function will be used for both mm and bmm, so we need to insert batch dimension if the tensor is 2d
    if len(real_shape) == 2:
        real_shape.insert(0, 1)
        real_stride.insert(0, packed_fp4_tensor.numel())

    # each packed uint8 contains 2 fp4 elements
    real_shape[-2 if is_column_major else -1] *= 2
    if is_column_major:
        real_stride[-1] *= 2
        for i in range(len(real_stride) - 2):
            real_stride[i] *= 2
    else:
        for i in range(len(real_stride) - 1):
            real_stride[i] *= 2

    return (tuple(real_shape), tuple(real_stride))


def _expand_block_scale_tensor_shape(block_scale_tensor, batch_size):
    # This function will be shared for both mm and bmm, when 2d block scale tensor is provided, we need unfold the batch dimension. the unfoled dim and stride is returned.
    block_scale_shape = list(block_scale_tensor.shape)
    block_scale_stride = list(block_scale_tensor.stride())

    if len(block_scale_shape) == 2:
        # expand to 3d
        block_scale_shape.insert(0, batch_size)
        block_scale_stride.insert(0, 1)

        # update the stride and shape for the expanded dimension
        is_column_major = block_scale_tensor.stride(-2) == 1
        expand_dim = 2 if is_column_major else 1

        assert block_scale_shape[expand_dim] % batch_size == 0
        block_scale_shape[expand_dim] = block_scale_shape[expand_dim] // batch_size
        block_scale_stride[0] = (
            block_scale_stride[expand_dim] * block_scale_shape[expand_dim]
        )
    elif len(block_scale_shape) == 3:
        pass
    else:
        raise ValueError(
            f"Unsupported block scale tensor shape: {block_scale_shape}, expected 2d or 3d."
        )

    return (tuple(block_scale_shape), tuple(block_scale_stride))


def mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    backend: Literal["cudnn", "trtllm", "cutlass"] = "cudnn",
    use_nvfp4: bool = True,
) -> torch.Tensor:
    r"""MM FP4

    Parameters
    ----------
    a: torch.Tensor
        Input tensor, shape (m, k), fp4 e2m1fn_x2 or uint8.

    b: torch.Tensor
        Mat2 tensor, shape (k, n), should be column major, fp4 e2m1fn_x2 or uint8.

    a_descale: torch.Tensor
        Block scale tensor for A, shape (m, k // block_size), float8_e4m3fn or uint8.

    b_descale: torch.Tensor
        Block scale tensor for B, shape (k, n // block_size), float8_e4m3fn or uint8.

    alpha: Optional[torch.Tensor]
        Global scale tensor, float scalar.

    out_dtype: torch.dtype
        Output dtype, bf16 or fp16.

    out: Optional[torch.Tensor]
        Out tensor, shape (m, n), bf16 or fp16, defaults to ``None``.

    block_size: int
        Block size for FP4 quantization, only 16 and 32 are supported. 16 in case of nvfp4 quantization. 32 in case of mxfp4 quantization.

    use_8x4_sf_layout: bool
        Whether to use 8x4 scale factor layout or 128x4 scale factor layout, defaults to False.

    backend: Literal["cudnn", "trtllm", "cutlass"]
        Backend to use, defaults to "cudnn".

    use_nvfp4: bool
        Whether to use nvfp4 quantization or mxfp4 quantization, defaults to False.

    Notes
    -----
    When cudnn/cutlass backend is used, both a and b should quantized with nvfp4_quantize using the 128x4 scale factor layout and do_shuffle=False.
    When trtllm backend is used, b must be quantized with 128x4 layout and `do_shuffle=True`. a can be quantized with either 128x4 or 8x4 layout (controlled by `use_8x4_sf_layout`) and `do_shuffle=False`.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (m, n), bf16 or fp16.

    Examples
    --------
    >>> import torch
    >>> from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
    >>> a = torch.randn([48, 128], device="cuda", dtype=torch.bfloat16)
    >>> b = torch.randn([256, 128], device="cuda", dtype=torch.bfloat16)
    >>> a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    >>> b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
    >>> a_fp4, a_sf = nvfp4_quantize(a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    >>> b_fp4, b_sf = nvfp4_quantize(b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=True)
    >>> out = mm_fp4(a_fp4, b_fp4.T, a_sf, b_sf.T, 1.0/(a_global_sf * b_global_sf), torch.bfloat16, None, backend="trtllm")
    >>> out.shape
    torch.Size([48, 256])
    """
    # pre-check the input tensor, block scale tensor and alpha tensor
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"mm_fp4 accepts 2d tensors, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"K dimension mismatch in mm_fp4. got a.shape[1] = {a.shape[1]}, b.shape[0] = {b.shape[0]}"
        )
    if a.dtype not in {torch.uint8, _get_native_fp4_dtype()} or b.dtype not in {
        torch.uint8,
        _get_native_fp4_dtype(),
    }:
        raise ValueError(
            f"a and b must have float4_e2m1fn_x2 packed into uint8. "
            f"Got {a.dtype} and {b.dtype}."
        )
    if a_descale.dtype not in {
        torch.float8_e4m3fn,
        torch.uint8,
    } or b_descale.dtype not in {torch.float8_e4m3fn, torch.uint8}:
        raise ValueError(
            f"a_descale and b_descale must have float8_e4m3fnx2 packed into uint8. "
            f"Got {a_descale.dtype} and {b_descale.dtype}."
        )
    if alpha is not None and alpha.dtype != torch.float:
        raise ValueError(f"alpha must be a float tensor, got {alpha.dtype}")
    if alpha is not None and alpha.numel() != 1:
        raise ValueError(f"alpha must be a scalar, got {alpha.numel()}")

    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Unsupported output dtype: {out_dtype}. "
            f"Only torch.bfloat16 and torch.float16 are supported for FP4 GEMM operations."
        )

    if use_nvfp4 and block_size != 16:
        raise ValueError("nvfp4 only supports block_size = 16.")
    if not use_nvfp4 and block_size != 32:
        raise ValueError("mxfp4 supports block_size = 32.")
    if backend != "trtllm" and use_8x4_sf_layout:
        raise ValueError("Only TRTLLM FP4 GEMM supports 8x4 scale factor layout.")
    if backend == "trtllm" and _match_sm_version(a.device, ["110"]):
        raise ValueError("TRTLLM FP4 GEMM is not supported on SM110.")
    if backend != "cudnn" and not use_nvfp4:
        raise ValueError("Only cudnn FP4 GEMM supports mxfp4 quantization.")
    if (
        backend == "cudnn"
        and not use_nvfp4
        and _match_sm_version(a.device, ["120"])
        and cudnn.backend_version() < 91400
    ):
        raise LibraryError(
            "cudnn FP4 GEMM with mxfp4 quantization is not supported on SM120 with cuDNN backend version < 9.14.0."
        )

    # allocate the output tensor if not provided
    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]),
            device=a.device,
            dtype=out_dtype,
        )

    workspace_buffer = _get_cache_buf(
        "mm_fp4_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    if backend == "cudnn":
        _check_cudnn_fp4_availability()

        # the fp4 cudnn graph will be shared for both mm and bmm, so
        # here we need to get the 3d shape and stride including the
        # batch dimension for both input and block scale tensors.
        real_a_shape, real_a_stride = _get_real_fp4_shape_from_packed_uint8(a)
        real_b_shape, real_b_stride = _get_real_fp4_shape_from_packed_uint8(b)
        batch = real_a_shape[0]
        expanded_a_descale_shape, expanded_a_descale_stride = (
            _expand_block_scale_tensor_shape(a_descale, batch)
        )
        expanded_b_descale_shape, expanded_b_descale_stride = (
            _expand_block_scale_tensor_shape(b_descale, batch)
        )

        # build the fp4 cudnn graph
        graph = build_cudnn_gemm_block_scale_dequantize_graph(
            real_a_shape,
            real_a_stride,
            real_b_shape,
            real_b_stride,
            expanded_a_descale_shape,
            expanded_a_descale_stride,
            expanded_b_descale_shape,
            expanded_b_descale_stride,
            cudnn.data_type.FP4_E2M1,
            _torch_data_type_to_cudnn_data_type(out_dtype),
            block_size,
            a.device,
            alpha,
            use_nvfp4,
        )

        # execute the fp4 cudnn graph
        execute_cudnn_gemm_fp4_graph(
            graph, a, b, a_descale, b_descale, alpha, out, workspace_buffer
        )
    elif backend == "trtllm":
        if out_dtype != torch.bfloat16:
            raise ValueError(
                f"Unsupported output dtype: {out_dtype}. "
                f"Only torch.bfloat16 is supported for TRTLLM FP4 GEMM operations."
            )

        get_trtllm_fp4_gemm_module().trtllm_fp4_gemm(
            a,
            b.T,
            a_descale,
            b_descale.T,
            alpha,
            out,
            use_8x4_sf_layout=use_8x4_sf_layout,
            workspace_buffer=workspace_buffer,
        )
    elif backend == "cutlass":
        # cutlass require uint8 scale when a/b is fp4 packed uint8.
        if a.dtype == torch.uint8 and a_descale.dtype == torch.float8_e4m3fn:
            a_descale = a_descale.view(torch.uint8)
        if b.dtype == torch.uint8 and b_descale.dtype == torch.float8_e4m3fn:
            b_descale = b_descale.view(torch.uint8)

        # Dispatch to the correct module based on device architecture
        major, _ = get_compute_capability(a.device)
        if major == 12:
            gemm_module = get_gemm_sm120_module_cutlass_fp4()
        else:
            gemm_module = get_gemm_sm100_module_cutlass_fp4()

        gemm_module.cutlass_fp4_gemm(
            a, b.T, a_descale, b_descale.T, alpha, out, workspace_buffer
        )
    return out


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cublas", "cutlass", "auto"] = "cublas",
) -> torch.Tensor:
    r"""BMM FP8

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (b, m, k), fp8 e4m3 or fp8 e5m2.

    B: torch.Tensor
        Mat2 tensor, shape (b, k, n), should be column major, fp8 e4m3 or fp8 e5m2.

    A_scale: torch.Tensor
        Scale tensor for A, float.

    B_scale: torch.Tensor
        Scale tensor for B, float.

    dtype: torch.dtype
        out dtype, bf16 or fp16.

    out: Optional[torch.Tensor]
        Out tensor, shape (b, m, n), bf16 or fp16, defaults to ``None``.

    backend: Literal["cudnn", "cublas", "cutlass", "auto"]
        The backend to use for the operation. Defaults to ``"cublas"``.
        ``"auto"`` allows selecting the best tactic from all available backends when autotune is enabled.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (b, m, n), bf16 or fp16.

    Examples
    --------
    >>> import torch
    >>> import torch.nn.functional as F
    >>> import flashinfer
    >>> def to_float8(x, dtype=torch.float8_e4m3fn):
    ...     finfo = torch.finfo(dtype)
    ...     min_val, max_val = x.aminmax()
    ...     amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    ...     scale = finfo.max / amax
    ...     x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    ...     return x_scl_sat.to(dtype), scale.float().reciprocal()
    >>>
    >>> input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    >>> input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)
    >>> # column major weight
    >>> weight = torch.randn([16, 80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    >>> weight_fp8, weight_inv_s = to_float8(weight, dtype=torch.float8_e4m3fn)
    >>> out = flashinfer.bmm_fp8(input_fp8, weight_fp8, input_inv_s, weight_inv_s, torch.bfloat16)
    >>> out.shape
    torch.Size([16, 48, 80])
    >>> out.dtype
    torch.bfloat16
    """
    _validate_fp8_output_dtype(dtype)

    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )

    workspace_buffer = _get_cache_buf(
        "bmm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, A.device
    )

    if backend == "cudnn":
        backends = ["cudnn"]
    elif backend == "cublas":
        backends = ["cublas"]
    elif backend == "cutlass":
        if A.dtype == torch.float8_e5m2 or B.dtype == torch.float8_e5m2:
            raise ValueError("e5m2 is not supported for cutlass backend")
        backends = ["cutlass"]
    elif backend == "auto":
        backends = ["cutlass", "cublas", "cudnn"]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    fp8_gemm_sm100(A, B, A_scale, B_scale, out, workspace_buffer, backends)
    return out


def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal["MN", "K"]] = None,
    mma_sm: int = 1,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    backend: Literal["cutlass", "trtllm"] = "cutlass",
) -> torch.Tensor:
    r"""Performs matrix multiplication with FP8 data types using groupwise scaling.

    This function implements a GEMM operation that allows for fine-grained control over
    scale granularity across different dimensions. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape (m, k), fp8 e4m3 or fp8 e5m2.

    b: torch.Tensor
        Column-major input tensor shape (n, k), fp8 e4m3 or fp8 e5m2.

    a_scale: torch.Tensor
        if the backend is ``cutlass``:
            Column-major scale tensor for a, shape ``(m, k // block_size)`` if scale_major_mode is ``K``
            or shape ``(k // block_size, m)`` if scale_major_mode is ``MN``
        if the backend is ``trtllm``:
            scale_major_mode should be None, the scale tensor should be (m, k // block_size),
            contiguous on the first dimension

    b_scale: torch.Tensor
        if the backend is ``cutlass``:
            Row-major scale tensor for b, shape ``(n // block_size, k // block_size)`` if scale_major_k is ``K``
            or shape ``(k // block_size, n // block_size)`` if scale_major_mode is ``MN``
        if the backend is ``trtllm``:
            scale_major_mode should be None, the scale tensor should be (k // block_size, n // block_size),
            contiguous on the first dimension

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, (m_granularity, n_granularity, k_granularity).

    scale_major_mode: Literal["MN", "K"]
        The layout mode of scale tensor, `MN` for MN-major scale with shape of
        ``(k // block_size, *)`` and `K` for K-major scale with shape of
        ``(*, k // block_size)``

    mma_sm: int
        How many SMs to use for the MMA operation, must be 1 or 2.
        2 is faster when number of rows (M) per group is large (>= 256).

    out: Optional[torch.Tensor]
        Output tensor, shape (m, n). If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        If out is not specified, we will create an output tensor with this dtype.
        Defaults to ``torch.bfloat16``.

    backend: Literal["cutlass", "trtllm"]
        The backend to use for the operation. Defaults to ``"cutlass"``.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape (m, n).

    Notes
    -----
    The ``m`` should be padded to a multiple of 4 before calling this function, to accommodate the kernel's requirement.
    """
    if backend == "trtllm" and _match_sm_version(a.device, ["110"]):
        raise ValueError("TRTLLM FP8 GEMM is not supported on SM110.")

    workspace_buffer = _get_cache_buf(
        "gemm_fp8_nt_groupwise_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Shape mismatch. a.shape = {a.shape}, b.shape = {b.shape}")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Shape mismatch. a.shape[1] = {a.shape[1]}, b.shape[1] = {b.shape[1]}"
        )

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
    else:
        out_dtype = out.dtype

    _validate_fp8_output_dtype(out_dtype)

    # NOTE(Zihao): (out_specified, need_padding)
    # (False, False) -> create out_padded tensor explicitly
    # (False, True) -> create out_padded tensor explicitly
    # (True, False) -> use out tensor as out_padded
    # (True, True) -> create out_padded tensor explicitly

    if out is None:
        out = torch.empty(
            a.shape[0],
            b.shape[0],
            device=a.device,
            dtype=out_dtype,
        )

    if backend == "cutlass":
        if not _match_sm_version(a.device, ["100", "103", "110", "120", "121"]):
            raise ValueError(
                "gemm_fp8_nt_groupwise is only supported on SM100, SM103, SM110, SM120, or SM121 in cutlass backend."
            )
    elif backend == "trtllm":
        if not _match_sm_version(a.device, ["100", "103"]):
            raise ValueError(
                "gemm_fp8_nt_groupwise is only supported on SM100, SM103 in trtllm backend."
            )

    if backend == "cutlass":
        assert scale_major_mode is not None
        if is_sm120a_supported(a.device) or is_sm121a_supported(a.device):
            # SM120/121 doesn't use mma_sm parameter
            get_gemm_sm120_module().gemm_fp8_nt_groupwise.default(
                workspace_buffer,
                a,
                b,
                a_scale,
                b_scale,
                out,
                *scale_granularity_mnk,
                scale_major_mode,
            )
        elif is_sm100a_supported(a.device):
            get_gemm_sm100_module().gemm_fp8_nt_groupwise.default(
                workspace_buffer,
                a,
                b,
                a_scale,
                b_scale,
                out,
                *scale_granularity_mnk,
                scale_major_mode,
                mma_sm,
            )
        else:
            raise ValueError(f"Unsupported device for FP8 GEMM: {a.device}")
    elif backend == "trtllm":
        assert scale_granularity_mnk == (1, 128, 128)
        assert a.shape[1] >= 256
        # mma_sm is ignored
        get_trtllm_gemm_module().trtllm_gemm(
            workspace_buffer,
            a,
            b,
            a_scale,
            b_scale,
            None,
            out,
            False,
            -1,
        )

    return out


@functools.cache
def get_trtllm_fp4_gemm_module():
    mod = gen_trtllm_gen_gemm_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())

    class TrtllmFp4GemmRunner(TunableRunner):
        def __init__(self, use_8x4_sf_layout: bool = True):
            self._fp4_gemm_runner = op.trtllm_gemm
            self._use_8x4_sf_layout = use_8x4_sf_layout

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a_tensor_index = 1
            b_tensor_index = 2

            a = profile.get_opt_shapes()[a_tensor_index]
            b = profile.get_opt_shapes()[b_tensor_index]
            m = a[0]
            n = b[0]
            k = a[1] * 2
            (
                workspace_buffer,
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out,
            ) = inputs
            type_e2m1 = 0
            type_bf16 = 2
            return list(
                op.trtllm_gemm_tactics(
                    m, n, k, type_e2m1, type_bf16, self._use_8x4_sf_layout
                )
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            (
                workspace_buffer,
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out,
            ) = inputs
            op.trtllm_gemm.default(
                workspace_buffer,
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out,
                self._use_8x4_sf_layout,
                tactic,
            )
            return out

    @register_custom_op(
        "flashinfer::trtllm_fp4_gemm",
        mutates_args=(""),
    )
    def trtllm_fp4_gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        a_descale: torch.Tensor,
        b_descale: torch.Tensor,
        alpha: torch.Tensor,
        out: torch.Tensor,
        use_8x4_sf_layout: bool,
        workspace_buffer: torch.Tensor,
    ):
        tuner = AutoTuner.get()

        a_tensor_index = 1
        a_scale_tensor_index = 3
        out_tensor_index = 6

        def pad_up(x, y):
            return ((x + y - 1) // y) * y

        tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (a_tensor_index,),
                    (0,),
                    get_last_power_of_2_num_tokens_buckets,
                    last_positive_power_of_2,
                ),
            ),
            constraint_specs=(
                ConstraintSpec(
                    a_scale_tensor_index,
                    0,
                    lambda shapes: pad_up(
                        shapes[a_tensor_index][0], 8 if use_8x4_sf_layout else 128
                    ),
                ),
                ConstraintSpec(
                    out_tensor_index, 0, lambda shapes: shapes[a_tensor_index][0]
                ),
            ),
        )

        fp4_runner = TrtllmFp4GemmRunner(use_8x4_sf_layout)

        inputs = [
            workspace_buffer,
            a,
            b,
            a_descale,
            b_descale,
            alpha,
            out,
        ]
        _, tactic = tuner.choose_one(
            "trtllm_fp4_gemm_8x4" if use_8x4_sf_layout else "trtllm_fp4_gemm_128x4",
            [fp4_runner],
            tuning_config,
            inputs,
        )

        fp4_runner(inputs=inputs, tactic=tactic)

    # Register the module
    return SimpleNamespace(
        trtllm_fp4_gemm=trtllm_fp4_gemm,
    )


def gemm_fp8_nt_blockscaled(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal["MN", "K"]] = "MN",
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Performs matrix multiplication with FP8 data types using block-scaled scaling.

    Block-scaled scaling is a special case of groupwise scaling where the scale granularity
    is (128, 128, 128).
    """
    return gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        scale_granularity_mnk=(128, 128, 128),
        scale_major_mode=scale_major_mode,
        mma_sm=mma_sm,
        out=out,
        out_dtype=out_dtype,
    )


def group_gemm_fp8_nt_groupwise(
    a: torch.Tensor,  # (cum_m, k)
    b: torch.Tensor,  # (batch_size, n, k)
    a_scale: torch.Tensor,  # (k // block_size, cum_m)
    b_scale: torch.Tensor,  # (batch_size, k // block_size, n // block_size)
    m_indptr: torch.Tensor,  # (batch_size + 1, )
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    scale_major_mode: Literal["MN", "K"] = "MN",
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,  # (cum_m, n)
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform group GEMM with FP8 data types using groupwise scaling. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor shape ``(cum_m, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
        ``cum_m`` is the cumulative sum of the segment lengths.

    b: torch.Tensor
        Column-major input tensor shape ``(batch_size, n, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape ``(cum_m, k // block_size)`` if scale_major_mode is ``K``
        or shape ``(k // block_size, cum_m)`` if scale_major_mode is ``MN``, data type is ``torch.float32``.

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape ``(batch_size, n // block_size, k // block_size)`` if scale_major_mode is ``K``
        shape ``(batch_size, k // block_size, n // block_size)`` if scale_major_mode is ``MN``, data type is ``torch.float32``.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(batch_size + 1,)``, data type is ``torch.int32``.
        Element element in ``m_indptr`` must be a multiple of 4.

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, (m_granularity, n_granularity, k_granularity).

    scale_major_mode: Literal["MN", "K"]
        The layout mode of scale tensor, `MN` for MN-major scale with shape of
        ``(k // block_size, *)`` and `K` for K-major scale with shape of
        ``(*, k // block_size)``

    mma_sm: int
        How many SMs to use for the MMA operation, must be 1 or 2.
        2 is faster when number of rows (M) per group is large (>= 256).

    out: Optional[torch.Tensor]
        The output tensor, shape ``(cum_m, n)``. If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        The data type of the output tensor, must be ``torch.bfloat16`` or ``torch.float16``.

    Returns
    -------
    out: torch.Tensor
        The output tensor, shape ``(cum_m, n)``.

    Notes
    -----
    Each value in ``m_indptr`` should be padded to a multiple of 4 before calling this function,
    to accommodate the kernel's requirement.
    """
    if (
        not is_sm100a_supported(a.device)
        and not is_sm120a_supported(a.device)
        and not is_sm121a_supported(a.device)
    ):
        raise ValueError(
            "gemm_fp8_nt_groupwise is only supported on SM100, SM120, and SM121."
        )
    if not (_match_sm_version(a.device, ["100", "103", "110", "120", "121"])):
        raise ValueError(
            "gemm_fp8_nt_groupwise is only supported on SM100, SM103, SM110, SM120, or SM121."
        )

    int_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_int_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_float_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    assert a.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert b.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert a_scale.dtype == torch.float32
    assert b_scale.dtype == torch.float32
    assert m_indptr.dtype == torch.int32
    assert scale_major_mode in ["MN", "K"]
    assert mma_sm in [1, 2]
    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype
    _validate_fp8_output_dtype(out_dtype)

    num_groups = m_indptr.shape[0] - 1
    assert b.shape[0] == num_groups
    n = b.shape[1]
    k = b.shape[2]

    # assert a.shape[0] == m_indptr[-1].item()  # Not enabled in consideration of performance
    assert a.shape[1] == k
    align_n = 8
    align_k = 16
    assert n % align_n == 0
    assert k % align_k == 0

    out_shape = (a.shape[0], n)
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
    else:
        assert out.shape == out_shape
        assert out.dtype == out_dtype

    if is_sm120a_supported(a.device) or is_sm121a_supported(a.device):
        # SM120/121 doesn't use mma_sm parameter
        get_gemm_sm120_module().group_gemm_fp8_nt_groupwise.default(
            int_workspace_buffer,
            float_workspace_buffer,
            a,
            b,
            a_scale,
            b_scale,
            out,
            m_indptr,
            n,
            k,
            *scale_granularity_mnk,
            scale_major_mode,
        )
    elif is_sm100a_supported(a.device):
        get_gemm_sm100_module().group_gemm_fp8_nt_groupwise.default(
            int_workspace_buffer,
            float_workspace_buffer,
            a,
            b,
            a_scale,
            b_scale,
            out,
            m_indptr,
            n,
            k,
            *scale_granularity_mnk,
            scale_major_mode,
            mma_sm,
        )
    else:
        raise ValueError(
            f"group_gemm_fp8_nt_groupwise requires SM100, SM120, or SM121, but got {a.device}"
        )
    return out


def group_gemm_mxfp8_mxfp4_nt_groupwise(
    a: torch.Tensor,  # (cum_m, k)
    b: torch.Tensor,  # (batch_size, n, k // 2)
    a_scale: torch.Tensor,  # (cum_m_padded, k // 32)
    b_scale: torch.Tensor,  # (batch_size, n_padded, k // 32)
    m_indptr: torch.Tensor,  # (batch_size + 1, )
    mma_sm: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    swap_ab: bool = True,
    out: Optional[torch.Tensor] = None,  # (cum_m, n)
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform group GEMM with MXFP4 data types using groupwise scaling. Currently only supported on NVIDIA
    Blackwell architecture.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor, shape ``(cum_m, k)``, data type is ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
        ``cum_m`` is the cumulative sum of the segment lengths.

    b: torch.Tensor
        Column-major input tensor, shape ``(batch_size, n, k // 2)``, data type is ``torch.uint8``.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape ``(cum_m_padded, k // 32)``, data type is ``torch.uint8``.

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape ``(batch_size, n_padded, k // 32)``, data type is ``torch.uint8``.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(batch_size + 1,)``, data type is ``torch.int32``.
        Element element in ``m_indptr`` must be a multiple of 4.

    mma_sm: int
        How many SMs to use for the MMA operation, must be 1 or 2.
        2 is faster when number of rows (M) per group is large (>= 256).

    tile_m: int
        The tile size for the M dimension, must be 128.

    tile_n: int
        The tile size for the N dimension, must be 64, 128, 192, or 256.

    tile_k: int
        The tile size for the K dimension, must be 128 or 256.

    swap_ab: bool
        Whether to swap the A and B tensors.

    out: Optional[torch.Tensor]
        The output tensor, shape ``(cum_m, n)``. If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        The data type of the output tensor, must be ``torch.bfloat16`` or ``torch.float16``.

    Returns
    -------
    out: torch.Tensor
        The output tensor, shape ``(cum_m, n)``.

    Notes
    -----
    Each value in ``m_indptr`` should be padded to a multiple of 4 before calling this function,
    to accommodate the kernel's requirement.
    """
    int_workspace_buffer = _get_cache_buf(
        "group_gemm_mxfp4_nt_groupwise_int_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_mxfp4_nt_groupwise_float_workspace",
        DEFAULT_WORKSPACE_SIZE,
        a.device,
    )

    assert a.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    assert b.dtype == torch.uint8
    assert a_scale.dtype == torch.uint8
    assert b_scale.dtype == torch.uint8
    assert m_indptr.dtype == torch.int32
    assert mma_sm in [1, 2]
    assert tile_m in [128]
    assert tile_n in [64, 128, 192, 256]
    assert tile_k in [128, 256]
    assert swap_ab in [True, False]
    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype
    assert out_dtype in [torch.bfloat16, torch.float16]

    num_groups = m_indptr.shape[0] - 1
    assert b.shape[0] == num_groups
    n = b.shape[1]
    k = b.shape[2] * 2  # Multiply by 2 because b is e2m1 packed as uint8

    # assert a.shape[0] == m_indptr[-1].item()  # Not enabled in consideration of performance
    assert a.shape[1] == k
    align_n = 8
    align_k = 128
    assert n % align_n == 0
    assert k % align_k == 0

    out_shape = (a.shape[0], n)
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)
    else:
        assert out.shape == out_shape
        assert out.dtype == out_dtype

    get_gemm_sm100_module().group_gemm_mxfp4_nt_groupwise.default(
        int_workspace_buffer,
        float_workspace_buffer,
        a,
        b,
        a_scale,
        b_scale,
        out,
        m_indptr,
        n,
        k,
        mma_sm,
        tile_m,
        tile_n,
        tile_k,
        swap_ab,
    )
    return out


# NOTE(Zihao): keep the old name for backward compatibility
group_gemm_mxfp4_nt_groupwise = group_gemm_mxfp8_mxfp4_nt_groupwise


def pad_indptr_to_multiple_of_4(
    m_indptr: torch.Tensor,
):
    from .triton.gemm import compute_padding_mapping

    batch_size = m_indptr.shape[0] - 1
    m = m_indptr[1:] - m_indptr[:-1]
    m = m + 3 - (m + 3) % 4
    padded_m_indptr = torch.cat((torch.zeros((1,), device=m.device, dtype=m.dtype), m))
    padded_m_indptr = padded_m_indptr.cumsum(dim=0, dtype=padded_m_indptr.dtype)

    m_rank = torch.zeros((m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device)
    padded_m_rank = torch.zeros(
        (m_indptr[-1],), dtype=m_indptr.dtype, device=m_indptr.device
    )

    compute_padding_mapping[(batch_size,)](
        m_indptr, padded_m_indptr, m_rank, padded_m_rank
    )

    return padded_m_indptr, padded_m_rank


def gen_deepgemm_sm100_module() -> SimpleNamespace:
    from flashinfer.deep_gemm import load_all

    load_all()
    return SimpleNamespace(
        group_deepgemm_fp8_nt_groupwise=group_deepgemm_fp8_nt_groupwise,
        batch_deepgemm_fp8_nt_groupwise=batch_deepgemm_fp8_nt_groupwise,
    )


@functools.cache
def get_deepgemm_sm100_module():
    module = gen_deepgemm_sm100_module()
    return module


def group_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,  # (m, k)
    b: torch.Tensor,  # (batch_size, n, k)
    a_scale: torch.Tensor,  # (m, k // block_size)
    b_scale: torch.Tensor,  # (batch_size, n // block_size, k // block_size)
    m_indices: torch.Tensor,  # (m, )
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,  # (m, n)
    out_dtype: Optional[torch.dtype] = None,
):
    r"""Perform grouped matrix multiplication with FP8 data types using DeepGEMM backend.

    This function performs a grouped GEMM operation where each group in tensor `b` is multiplied
    with the corresponding rows in tensor `a`. The grouping is determined by the `m_indices` tensor,
    which specifies which group each row belongs to. This is particularly useful for scenarios
    like mixture of experts (MoE) where different tokens are routed to different experts.

    The operation can be conceptualized as:

    >>> for i in range(num_groups):
    >>>    row_slice = slice(i * m_per_group, (i + 1) * m_per_group)
    >>>    output[row_slice] = a[row_slice] @ b[i].T

    Currently only supported on NVIDIA Blackwell (SM100) architecture.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor A of shape ``(m, k)`` with FP8 data type (``torch.float8_e4m3fn``).
        This tensor contains all rows that will be multiplied with different groups in `b`.

    b : torch.Tensor
        Input tensor B of shape ``(batch_size, n, k)`` with FP8 data type (``torch.float8_e4m3fn``).
        Each slice ``b[i]`` represents a different group/expert that will be multiplied with
        the corresponding rows in `a`.

    a_scale : torch.Tensor
        Scaling factors for tensor `a` of shape ``(m, k // block_size)`` with ``torch.float32`` dtype.
        These are typically generated from per-token quantization of the original float32 tensor.

    b_scale : torch.Tensor
        Scaling factors for tensor `b` of shape ``(batch_size, n // block_size, k // block_size)``
        with ``torch.float32`` dtype. These are typically generated from per-block quantization
        of the original float32 tensor for each group.

    m_indices : torch.Tensor
        Group assignment tensor of shape ``(m,)`` with ``torch.int32`` dtype. Each element
        specifies which group (index into `b`) the corresponding row in `a` belongs to.
        For example, if ``m_indices[i] = j``, then row ``i`` in `a` will be multiplied with
        group ``j`` in `b`.

    scale_granularity_mnk : Tuple[int, int, int], optional
        The granularity of the scaling factors as ``(m_granularity, n_granularity, k_granularity)``.
        Default is ``(1, 128, 128)`` which means per-token scaling for `a` and 128x128 block
        scaling for `b`.

    out : Optional[torch.Tensor], optional
        Pre-allocated output tensor of shape ``(m, n)``. If not provided, a new tensor will be
        created.

    out_dtype : Optional[torch.dtype], optional
        Data type of the output tensor. If `out` is provided, this parameter is ignored.
        Default is ``torch.bfloat16``.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(m, n)`` containing the results of the grouped matrix multiplication.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.gemm import group_deepgemm_fp8_nt_groupwise
    >>> from flashinfer.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
    >>>
    >>> # Setup: 2 groups, 128 tokens per group, 4096 hidden size, 2048 expert size
    >>> m_per_group, n, k = 128, 2048, 4096
    >>> group_size = 2
    >>> m = m_per_group * group_size
    >>>
    >>> # Create float32 inputs
    >>> a_f32 = torch.randn(m, k, device="cuda", dtype=torch.float32)
    >>> b_f32 = torch.randn(group_size, n, k, device="cuda", dtype=torch.float32)
    >>>
    >>> # Quantize to FP8 with appropriate scaling
    >>> a_fp8, a_scale = per_token_cast_to_fp8(a_f32)
    >>> b_fp8 = torch.empty_like(b_f32, dtype=torch.float8_e4m3fn)
    >>> b_scale = torch.empty((group_size, n // 128, k // 128), device="cuda", dtype=torch.float32)
    >>> for i in range(group_size):
    ...     b_fp8[i], b_scale[i] = per_block_cast_to_fp8(b_f32[i])
    >>>
    >>> # Create group assignment
    >>> m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    >>> for i in range(group_size):
    ...     row_slice = slice(i * m_per_group, (i + 1) * m_per_group)
    ...     m_indices[row_slice] = i
    >>>
    >>> # Perform grouped GEMM
    >>> result = group_deepgemm_fp8_nt_groupwise(
    ...     a_fp8, b_fp8, a_scale, b_scale, m_indices, out_dtype=torch.bfloat16
    ... )
    >>> print(result.shape)  # torch.Size([256, 2048])

    Notes
    -----
    - This function requires NVIDIA Blackwell (SM100) architecture
    - The scaling factors should be generated using appropriate quantization functions
      like ``per_token_cast_to_fp8`` for `a` and ``per_block_cast_to_fp8`` for `b`
    - The function internally uses the DeepGEMM backend for optimized FP8 computation
    - All input tensors must be on the same CUDA device
    - The block size for scaling is determined by the ``scale_granularity_mnk`` parameter
    """
    from flashinfer.deep_gemm import m_grouped_fp8_gemm_nt_contiguous

    if not _match_sm_version(a.device, ["100", "103"]):
        raise ValueError(
            "m_grouped_fp8_gemm_nt_contiguous is only supported on SM100, SM100, SM103."
        )

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(a.shape[0], b.shape[1], dtype=out_dtype, device=a.device)

    m_grouped_fp8_gemm_nt_contiguous(
        (a, a_scale), (b, b_scale), out, m_indices, scale_granularity_mnk
    )

    return out


def batch_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,  # (batch_size, m, k)
    b: torch.Tensor,  # (batch_size, n, k)
    a_scale: torch.Tensor,  # (batch_size, m, k // block_size)
    b_scale: torch.Tensor,  # (batch_size, n // block_size, k // block_size)
    masked_m: torch.Tensor,  # (batch_size, )
    expected_m: int,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,  # (batch_size, m, n)
    out_dtype: Optional[torch.dtype] = None,
):
    r"""Perform batch matrix multiplication with FP8 data types using DeepGEMM backend.

    This function performs a batch GEMM operation where each group in tensor `b` is multiplied
    with the corresponding group of rows in tensor `a`. The results of each group is masked by
    the `masked_m` tensor, which specifies which group each row belongs to. This is particularly
    useful for scenarios like mixture of experts (MoE) where different tokens are routed to different experts.

    The operation can be conceptualized as:

    >>> for i in range(num_groups):
    >>>     output[i] = a[i][:masked_m[i]] @ b[i][:masked_m[i]].T

    Currently only supported on NVIDIA Blackwell (SM100) architecture.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor A of shape ``(batch_size, m, k)`` with FP8 data type (``torch.float8_e4m3fn``).
        Each slice ``a[i]`` represents a group of rows that will be multiplied with
        the corresponding group/expert in `b`.

    b : torch.Tensor
        Input tensor B of shape ``(batch_size, n, k)`` with FP8 data type (``torch.float8_e4m3fn``).
        Each slice ``b[i]`` represents a different group/expert that will be multiplied with
        the corresponding rows in `a`.

    a_scale : torch.Tensor
        Scaling factors for tensor `a` of shape ``(batch_size, m, k // block_size)`` with ``torch.float32`` dtype.
        These are typically generated from per-token quantization of the original float32 tensor.

    b_scale : torch.Tensor
        Scaling factors for tensor `b` of shape ``(batch_size, n // block_size, k // block_size)``
        with ``torch.float32`` dtype. These are typically generated from per-block quantization
        of the original float32 tensor for each group.

    masked_m : torch.Tensor
        Masking tensor of shape ``(batch_size,)`` with ``torch.int32`` dtype. Each element
        specifies the effective rows to be multiplied in each group.
        For example, if ``masked_m[i] = j``, then first ``j`` rows in `a[i]` will be multiplied with
        group ``i`` in `b`.

    expected_m : int
        A value hint (which is a value on CPU) for the M expectation of each batch, correctly setting
        this value may lead to better performance.

    scale_granularity_mnk : Tuple[int, int, int], optional
        The granularity of the scaling factors as ``(m_granularity, n_granularity, k_granularity)``.
        Default is ``(1, 128, 128)`` which means per-token scaling for `a` and 128x128 block
        scaling for `b`.

    out : Optional[torch.Tensor], optional
        Pre-allocated output tensor of shape ``(batch_size, m, n)``. If not provided, a new tensor will be
        created.

    out_dtype : Optional[torch.dtype], optional
        Data type of the output tensor. If `out` is provided, this parameter is ignored.
        Default is ``torch.bfloat16``.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``(batch_size, m, n)`` containing the results of the batch matrix multiplication.

    Examples
    --------
    >>> import torch
    >>> from flashinfer.gemm import batch_deepgemm_fp8_nt_groupwise
    >>> from flashinfer.utils import per_token_cast_to_fp8, per_block_cast_to_fp8
    >>>
    >>> # Setup: 2 groups, 128 tokens per group, 4096 hidden size, 2048 expert size
    >>> m, n, k = 128, 2048, 4096
    >>> group_size = 2
    >>>
    >>> # Create float32 inputs
    >>> a = torch.rand((group_size, m, k), device="cuda", dtype=torch.float32)
    >>> b = torch.rand((group_size, n, k), device="cuda", dtype=torch.float32)
    >>> masked_m = torch.randint(0, m, (group_size,), device="cuda", dtype=torch.int32)
    >>> a_fp8 = torch.empty_like(a, device="cuda", dtype=torch.float8_e4m3fn)
    >>> a_scale = torch.empty((group_size, m, k // 128), device="cuda", dtype=torch.float32)
    >>> b_fp8 = torch.empty_like(b, device="cuda", dtype=torch.float8_e4m3fn)
    >>> b_scale = torch.empty(
    ...    (group_size, n // 128, k // 128), device="cuda", dtype=torch.float32
    >>> )
    >>> for i in range(group_size):
    ...    a_fp8[i], a_scale[i] = per_token_cast_to_fp8(a[i])
    ...    b_fp8[i], b_scale[i] = per_block_cast_to_fp8(b[i])
    >>>
    >>> expected_m = min(int(masked_m.float().mean()) + 1, m)
    >>>
    >>> # Perform batch GEMM
    >>> result = batch_deepgemm_fp8_nt_groupwise(
    ...     a_fp8, b_fp8, a_scale, b_scale, masked_m, expected_m, out_dtype=torch.bfloat16
    ... )
    >>> print(result.shape)  # torch.Size([2, 128, 2048])

    Notes
    -----
    - This function requires NVIDIA Blackwell (SM100) architecture
    - The scaling factors should be generated using appropriate quantization functions
      like ``per_token_cast_to_fp8`` for `a` and ``per_block_cast_to_fp8`` for `b`
    - The function internally uses the DeepGEMM backend for optimized FP8 computation
    - All input tensors must be on the same CUDA device
    - The block size for scaling is determined by the ``scale_granularity_mnk`` parameter
    """
    from flashinfer.deep_gemm import m_grouped_fp8_gemm_nt_masked

    if not _match_sm_version(a.device, ["100", "103"]):
        raise ValueError(
            "m_grouped_fp8_gemm_nt_masked is only supported on SM100, SM103."
        )

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(
            a.shape[0], a.shape[1], b.shape[1], dtype=out_dtype, device=a.device
        )

    m_grouped_fp8_gemm_nt_masked(
        (a, a_scale), (b, b_scale), out, masked_m, expected_m, scale_granularity_mnk
    )

    return out
