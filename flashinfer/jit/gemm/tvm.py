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

import os
from typing import Tuple

import torch

from .. import env as jit_env
from ..utils import write_if_different


def gen_grouped_gemm_fp8_tvm_binding(
    uri: str,
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    dtype_out: torch.dtype,
    scale_granularity_m: int,
    scale_granularity_n: int,
    scale_granularity_k: int,
    scale_major_mode: str,  # "K" or "MN"
    mma_sm: int,
) -> Tuple[str, list]:
    """Generate TVM binding for FP8 grouped GEMM.

    Parameters
    ----------
    uri : str
        Unique identifier for this kernel configuration
    dtype_a : torch.dtype
        Data type of matrix A
    dtype_b : torch.dtype
        Data type of matrix B
    dtype_out : torch.dtype
        Data type of output matrix
    scale_granularity_m : int
        Scaling granularity in M dimension
    scale_granularity_n : int
        Scaling granularity in N dimension
    scale_granularity_k : int
        Scaling granularity in K dimension
    scale_major_mode : str
        Scale storage mode ("K" or "MN")
    mma_sm : int
        MMA scheduling mode (1 or 2)

    Returns
    -------
    Tuple[str, list]
        URI and list of generated source file paths
    """
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)

    source_paths = []

    # Copy the base implementation file unchanged
    src_path = jit_env.FLASHINFER_TVM_BINDING_DIR / "grouped_gemm_fp8.cu"
    dest_path = gen_directory / "grouped_gemm_fp8.cu"
    source_paths.append(dest_path)
    with open(src_path, "r") as f:
        source = f.read()
    write_if_different(dest_path, source)

    # Read the base TVM binding file and create specialized version
    tvm_binding_src = (
        jit_env.FLASHINFER_TVM_BINDING_DIR / "grouped_gemm_fp8_jit_tvm_binding.cu"
    )
    with open(tvm_binding_src, "r") as f:
        base_content = f.read()

    # Convert scale_major_mode to integer
    scale_major_mode_val = 0 if scale_major_mode == "K" else 1

    # Create specialized version by modifying the function export
    # Replace the direct export with a specialized wrapper
    specialized_content = base_content.replace(
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped_gemm_fp8_run, GroupedGemmFp8Run);",
        f"""// Specialized wrapper for this configuration
int GroupedGemmFp8RunSpecialized(
    DLTensor* int_workspace_buffer,
    DLTensor* float_workspace_buffer,
    DLTensor* A,
    DLTensor* B,
    DLTensor* SFA,
    DLTensor* SFB,
    DLTensor* D,
    DLTensor* m_indptr,
    int64_t n, int64_t k,
    TVMStreamHandle cuda_stream
) {{
    return GroupedGemmFp8Run(
        int_workspace_buffer,
        float_workspace_buffer,
        A, B, SFA, SFB, D, m_indptr,
        n, k,
        {scale_granularity_m},  // scale_granularity_m
        {scale_granularity_n},  // scale_granularity_n
        {scale_granularity_k},  // scale_granularity_k
        {scale_major_mode_val}, // scale_major_mode
        {mma_sm},               // mma_sm
        cuda_stream
    );
}}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped_gemm_fp8_run, GroupedGemmFp8RunSpecialized);""",
    )

    binding_dest_path = gen_directory / "grouped_gemm_fp8_jit_tvm_binding.cu"
    source_paths.append(binding_dest_path)
    write_if_different(binding_dest_path, specialized_content)

    return uri, source_paths


def get_grouped_gemm_fp8_uri(
    dtype_a: torch.dtype,
    dtype_b: torch.dtype,
    dtype_out: torch.dtype,
    scale_granularity_m: int,
    scale_granularity_n: int,
    scale_granularity_k: int,
    scale_major_mode: str,
    mma_sm: int,
) -> str:
    """Generate URI for FP8 grouped GEMM configuration."""
    dtype_a_str = str(dtype_a).split(".")[-1]
    dtype_b_str = str(dtype_b).split(".")[-1]
    dtype_out_str = str(dtype_out).split(".")[-1]

    return (
        f"group_gemm_fp8_{dtype_a_str}_{dtype_b_str}_{dtype_out_str}_"
        f"sg_{scale_granularity_m}_{scale_granularity_n}_{scale_granularity_k}_"
        f"sm_{scale_major_mode}_mma_{mma_sm}"
    )
