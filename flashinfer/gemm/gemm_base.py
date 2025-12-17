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
from enum import Enum
from types import SimpleNamespace
from typing import List, Literal, Optional, Tuple

from flashinfer.trtllm_low_latency_gemm import trtllm_low_latency_gemm
import torch

from ..api_logging import flashinfer_api
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)
from ..utils import (
    get_native_fp4_dtype,
    is_sm100a_supported,
    is_sm100f_supported,
    is_sm120a_supported,
    is_sm121a_supported,
    LibraryError,
    backend_requirement,
    supported_compute_capability,
)
from ..jit.gemm import gen_gemm_sm90_module
from ..jit.gemm import gen_gemm_module
from ..jit.gemm import gen_gemm_sm100_module
from ..jit.gemm import gen_gemm_sm120_module
from ..jit.gemm import gen_gemm_sm120_module_cutlass_fp4
from ..jit.gemm import gen_gemm_sm100_module_cutlass_fp4
from ..jit.gemm import gen_gemm_sm100_module_cutlass_fp8
from ..jit.gemm import gen_trtllm_gen_gemm_module
from ..jit.gemm import gen_tgv_gemm_sm10x_module
from ..jit.gemm import gen_deepgemm_sm100_module
from ..jit.cpp_ext import get_cuda_version
from ..jit.gemm import gen_fp8_blockscale_gemm_sm90_module


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


from ..jit.cubin_loader import setup_cubin_loader
from ..utils import (
    _get_cache_buf,
    determine_gemm_backend,
    get_indptr,
    is_float8,
    register_custom_op,
    register_fake_op,
    get_compute_capability,
)

DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024

# Error messages
CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR = "cudnn FP4 GEMM with mxfp4 quantization is not supported on SM120 with cuDNN backend version < 9.14.0."


def _match_sm_version(device: torch.device, sm_version: list[str]):
    major, minor = get_compute_capability(device)
    device_arch = f"{major * 10 + minor}"
    return device_arch in sm_version


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
                module.bmm_fp8(
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
        module.cutlass_segment_gemm(
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


@functools.cache
def get_gemm_sm100_module():
    module = gen_gemm_sm100_module().build_and_load()

    return module


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
                    scale_a_expanded = (
                        scale_a.view(1, 1)
                        .expand(scale_m_count, scale_k_count)
                        .contiguous()
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
                    scale_b_expanded = (
                        scale_b.view(1, 1)
                        .expand(scale_n_count, scale_k_count)
                        .contiguous()
                    )
                else:
                    scale_b_expanded = scale_b

                # Call SM120 gemm_fp8_nt_groupwise (now handles both 2D and 3D)
                module.gemm_fp8_nt_groupwise(
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
                module.fp8_gemm(
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


_FP8_GEMM_SM100_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (-2,),
            get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            4,  # out_tensor_index
            -2,
            lambda shapes: shapes[0][-2],
        ),
    ),
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
    if "cutlass_sm10x" in runner_names:
        runners.append(get_gemm_sm100_module_cutlass_fp8().cutlass_fp8_gemm_runner())
    if "cutlass_sm12x" in runner_names:
        runners.append(get_gemm_sm120_module_cutlass_fp8().cutlass_fp8_gemm_runner())
    if "cublas" in runner_names:
        runners.append(get_gemm_module().cublas_fp8_gemm_runner())
    if "cudnn" in runner_names:
        runners.append(_cudnn_gemm_fp8_runner())
    assert runners, "No suitable runners found"
    tuner = AutoTuner.get()

    inputs = [a, b, scale_a, scale_b, out, workspace_buffer]
    runner, tactic = tuner.choose_one(
        "fp8_gemm",
        runners,
        _FP8_GEMM_SM100_TUNING_CONFIG,
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)


def _create_cutlass_fp4_gemm_module(module, op_name: str, tuner_name: str):
    """Helper function to create cutlass FP4 GEMM module."""

    def cutlass_fp4_gemm_runner():
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
                (
                    a,
                    b,
                    a_descale,
                    b_descale,
                    alpha,
                    _,
                    out,
                    _,
                    _,
                    workspace_buffer,
                ) = inputs
                if a.dtype == torch.uint8 and a_descale.dtype == torch.float8_e4m3fn:
                    a_descale = a_descale.view(torch.uint8)
                if b.dtype == torch.uint8 and b_descale.dtype == torch.float8_e4m3fn:
                    b_descale = b_descale.view(torch.uint8)
                module.fp4_gemm(
                    a, b.T, a_descale, b_descale.T, alpha, out, workspace_buffer, tactic
                )
                return out

        return CutlassFp4GemmRunner()

    return SimpleNamespace(
        cutlass_fp4_gemm_runner=cutlass_fp4_gemm_runner,
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


def get_cutlass_fp4_gemm_module(
    sm_major: int,
):
    if sm_major in [10, 11]:
        return get_gemm_sm100_module_cutlass_fp4()
    elif sm_major == 12:
        return get_gemm_sm120_module_cutlass_fp4()
    else:
        raise ValueError(f"Unsupported SM major version: {sm_major}")


@functools.cache
def get_tgv_gemm_sm10x_module(
    dtype: torch.dtype = torch.bfloat16, use_sm_100f: bool = False
):
    """
    Get and build the TGV GEMM module for the specified dtype.

    Args:
        dtype: Data type for the GEMM operation (torch.bfloat16 or torch.float16)
        use_sm_100f: Whether to compile with SM100f flags (default: False), which makes the compiled kernel
            compatible with both B200 and B300 GPUs. However, it's only available with CUDA 12.9+.

    Returns:
        SimpleNamespace with the runner function
    """
    module = gen_tgv_gemm_sm10x_module(dtype, use_sm_100f).build_and_load()

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
                c = torch.empty(
                    (a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device
                )
                gemm_fn(b.t(), a.t(), bias, tactic, c, pdl)
                return c

        return TGVGemmRunner()

    # Register the module
    return SimpleNamespace(
        tgv_gemm_runner=tgv_gemm_runner,
    )


@flashinfer_api
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
        Output tensor of shape (M, N) in row-major layout

    Supported dtypes:
        - torch.bfloat16
        - torch.float16

    Note:
        - Requires SM100, SM103, or SM110 architecture
        - Input tensors a and b must have the same dtype
        - Tensor b is expected to be in column-major layout (transposed from typical PyTorch row-major)
    """
    # Verify SM100 architecture support
    if not _match_sm_version(a.device, ["100", "103"]):
        raise ValueError("TGV GEMM requires SM100, SM103 architecture")

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
    use_sm_100f = is_sm100f_supported(a.device)
    runners.append(get_tgv_gemm_sm10x_module(a.dtype, use_sm_100f).tgv_gemm_runner())

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
        module.cutlass_segment_gemm_sm90(
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

    from ..triton.gemm import compute_sm80_group_gemm_args

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

    from ..triton.gemm import compute_sm90_group_gemm_args

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

    @flashinfer_api
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

    # Check cuDNN backend version for FP4 support (requires cudnn_version == 9.11.1 or cudnn_version >= 9.13)
    backend_version = cudnn.backend_version()
    CUDNN_VERSION_9_11_1 = 91101
    CUDNN_VERSION_9_13_0 = 91300
    return (
        backend_version == CUDNN_VERSION_9_11_1
        or backend_version >= CUDNN_VERSION_9_13_0
    )


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
def create_cudnn_execution_plans_fp4_gemm(
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
    alpha_is_not_none,
    use_nvfp4,
):
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

        if alpha_is_not_none:
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
        if (alpha_is_not_none) and (not _is_cublas_fp4_available_in_cudnn()):
            graph.deselect_engines(["eng0"])

        return graph


@functools.cache
def build_plans_cudnn_fp4_gemm_graph(
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
    tactic: int = -1,
):
    # Graph should have been already cached, when we ran _cudnn_gemm_fp4_requirement
    graph = create_cudnn_execution_plans_fp4_gemm(
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
    )

    graph.check_support()
    if tactic != -1:
        graph.build_plan_at_index(tactic)
    else:
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
    tactic: int = -1,
):
    variant_pack = {
        UIDs.A_UID.value: a.view(get_native_fp4_dtype()),
        UIDs.B_UID.value: b.view(get_native_fp4_dtype()),
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

    if tactic == -1:
        graph.execute(variant_pack, workspace_buffer, handle=_get_cudnn_handle(stream))
    else:
        graph.execute_plan_at_index(
            variant_pack, workspace_buffer, tactic, handle=_get_cudnn_handle(stream)
        )


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


@flashinfer_api
def mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    backend: Literal["trtllm_low_latency"] = "trtllm_low_latency",
):
    r"""FP8 matrix multiplication.

    Parameters
    ----------
    a: torch.Tensor
        Input tensor, shape (m, k), fp8 e4m3.

    b: torch.Tensor
        - When using "trtllm_low_latency" backend,
          Weight tensor, shape (k // block_size, n, block_size), fp8 e4m3
          B needs to be pre-processed using `prepare_low_latency_gemm_weights`.
          block_size is 128 for e4m3.

    alpha: Optional[torch.Tensor]
        Scale tensor for the output, float. If None, defaults to 1.0 for no scaling.

    out_dtype: torch.dtype
        Output tensor data type. Default is torch.bfloat16.

    out: Optional[torch.Tensor]
        Output tensor, shape (m, n). If None, a new tensor will be allocated.

    backend: Literal["trtllm_low_latency"]
        Backend to use for computation. Default is "trtllm_low_latency".
        - "trtllm_low_latency": optimized for small M dimension.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (m, n) with dtype `out_dtype`.

    Examples
    --------
    >>> import torch
    >>> from flashinfer import mm_fp8, prepare_low_latency_gemm_weights
    >>> m = 16
    >>> n = 2560
    >>> k = 32768
    >>> a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    >>> a_fp8, a_inv_s = to_float8(a, dtype=torch.float8_e4m3fn)
    >>> b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    >>> b_fp8, b_inv_s = to_float8(b, dtype=torch.float8_e4m3fn)
    >>> prepared_b = prepare_low_latency_gemm_weights(b_fp8)
    >>> alpha = a_inv_s * b_inv_s
    >>> out = mm_fp8(a_fp8, prepared_b, alpha)
    >>> out.shape
    torch.Size([16, 2560])
    """

    supported_out_dtypes = (torch.bfloat16,)
    supported_backends = ("trtllm_low_latency",)

    if backend == "trtllm_low_latency":
        m = a.shape[0]
        n = b.shape[1]
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Only {supported_backends} are supported for FP8 GEMM operations."
        )

    # allocate the output tensor if not provided
    if out is None:
        if out_dtype not in supported_out_dtypes:
            raise ValueError(
                f"Unsupported output dtype: {out_dtype}. "
                f"Only {supported_out_dtypes} are supported for FP8 GEMM operations."
            )
        out = torch.empty(
            (m, n),
            device=a.device,
            dtype=out_dtype,
        )
    else:
        if out.dtype not in supported_out_dtypes:
            raise ValueError(
                f"Unsupported output dtype: {out.dtype}. "
                f"Only {supported_out_dtypes} are supported for FP8 GEMM operations."
            )
        if out.shape != (a.shape[0], b.shape[1]):
            raise ValueError(
                f"Output shape mismatch. Expected {a.shape[0], b.shape[1]}, got {out.shape}."
            )
        if out.device != a.device:
            raise ValueError(
                f"Output device mismatch. Expected {a.device}, got {out.device}."
            )
        if out_dtype is not None and out.dtype != out_dtype:
            raise ValueError(
                f"Output dtype mismatch. Expected {out_dtype}, got {out.dtype}."
            )

    if backend == "trtllm_low_latency":
        trtllm_low_latency_gemm(a, b, alpha, out)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Only {supported_backends} are supported for FP8 GEMM operations."
        )
    return out


def _get_cudnn_fp4_gemm_graph(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_nvfp4: bool = True,
    tactic: int = -1,
):
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
    # Constructed graph is cached, via @functools.cache decorator.
    graph = build_plans_cudnn_fp4_gemm_graph(
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
        alpha is not None,
        use_nvfp4,
        tactic=tactic,
    )
    return graph


def _cudnn_gemm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_nvfp4: bool = True,
    workspace_buffer: torch.Tensor = None,
    tactic: int = -1,
):
    # Graph should have been already cached, when we ran _cudnn_gemm_fp4_requirement
    graph = _get_cudnn_fp4_gemm_graph(
        a=a,
        b=b,
        a_descale=a_descale,
        b_descale=b_descale,
        alpha=alpha,
        out_dtype=out_dtype,
        out=out,
        block_size=block_size,
        use_nvfp4=use_nvfp4,
        tactic=tactic,
    )
    # execute the fp4 cudnn graph
    execute_cudnn_gemm_fp4_graph(
        graph, a, b, a_descale, b_descale, alpha, out, workspace_buffer, tactic=tactic
    )


def _cudnn_gemm_fp4_runner():
    class CudnnFp4GemmRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            # cudnn has heuristic for fp4 gemm, so we only need to use the default tactic
            (
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out_dtype,
                out,
                block_size,
                use_nvfp4,
                workspace_buffer,
            ) = inputs

            # Graph should have been already cached, when we ran _cudnn_gemm_fp4_requirement
            graph = _get_cudnn_fp4_gemm_graph(
                a=a,
                b=b,
                a_descale=a_descale,
                b_descale=b_descale,
                alpha=alpha,
                out_dtype=out_dtype,
                out=out,
                block_size=block_size,
                use_nvfp4=use_nvfp4,
                tactic=-1,
            )

            num_plans = graph.get_execution_plan_count()
            return list(range(num_plans))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            (
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out_dtype,
                out,
                block_size,
                use_nvfp4,
                workspace_buffer,
            ) = inputs
            _cudnn_gemm_fp4(
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out_dtype,
                out,
                block_size,
                use_nvfp4,
                workspace_buffer,
                tactic=tactic,
            )

    return CudnnFp4GemmRunner()


def _check_mm_fp4_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,  # unused
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,  # unused
    backend: Literal["cudnn", "trtllm", "cutlass", "auto"] = "auto",  # unused
    use_nvfp4: bool = True,
):
    # Generic checks
    ## pre-check the input tensor, block scale tensor and alpha tensor
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"mm_fp4 accepts 2d tensors, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"K dimension mismatch in mm_fp4. got a.shape[1] = {a.shape[1]}, b.shape[0] = {b.shape[0]}"
        )
    if a.dtype not in {torch.uint8, get_native_fp4_dtype()} or b.dtype not in {
        torch.uint8,
        get_native_fp4_dtype(),
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
        raise ValueError("mxfp4 only supports block_size = 32.")

    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cudnn_gemm_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,  # unused
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    backend: Literal["cudnn", "trtllm", "cutlass", "auto"] = "auto",  # unused
    use_nvfp4: bool = True,
):
    if use_8x4_sf_layout:
        raise ValueError("Only TRTLLM FP4 GEMM supports 8x4 scale factor layout.")
    if (
        not use_nvfp4
        and _match_sm_version(a.device, ["120"])
        and cudnn.backend_version() < 91400
    ):
        raise LibraryError(CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR)

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

    # build the fp4 cudnn graph. This graph will be cached & reused in mm_fp4()
    # because the graph is constructed with @functools.cache decorator
    graph = create_cudnn_execution_plans_fp4_gemm(
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
    graph.check_support()

    return True


@supported_compute_capability([100, 103])
def _trtllm_gemm_fp4_requirement(
    a: torch.Tensor,  # unused
    b: torch.Tensor,  # unused
    a_descale: torch.Tensor,  # unused
    b_descale: torch.Tensor,  # unused
    alpha: Optional[torch.Tensor] = None,  # unused
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,  # unused
    block_size: int = 16,  # unused
    use_8x4_sf_layout: bool = False,  # unused
    backend: Literal["cudnn", "trtllm", "cutlass", "auto"] = "auto",  # unused
    use_nvfp4: bool = True,
):
    if not use_nvfp4:
        raise ValueError("Only cudnn and auto FP4 GEMM supports mxfp4 quantization.")
    if out_dtype != torch.bfloat16:
        raise ValueError(
            f"Unsupported output dtype: {out_dtype}. "
            f"Only torch.bfloat16 is supported for TRTLLM FP4 GEMM operations."
        )
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cutlass_gemm_fp4_requirement(
    a: torch.Tensor,  # unused
    b: torch.Tensor,  # unused
    a_descale: torch.Tensor,  # unused
    b_descale: torch.Tensor,  # unused
    alpha: Optional[torch.Tensor] = None,  # unused
    out_dtype: torch.dtype = torch.bfloat16,  # unused
    out: Optional[torch.Tensor] = None,  # unused
    block_size: int = 16,  # unused
    use_8x4_sf_layout: bool = False,
    backend: Literal["cudnn", "trtllm", "cutlass", "auto"] = "auto",  # unused
    use_nvfp4: bool = True,
):
    if use_8x4_sf_layout:
        raise ValueError("Only TRTLLM FP4 GEMM supports 8x4 scale factor layout.")
    if not use_nvfp4:
        raise ValueError("Only cudnn and auto FP4 GEMM supports mxfp4 quantization.")
    return True


def _heuristic_func_mm_fp4(
    suitable_backends: List[str],
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    backend: Literal["cudnn", "trtllm", "cutlass", "auto"] = "cudnn",
    use_nvfp4: bool = True,
):
    r"""
    Heuristic function for mm_fp4 backend selection. Routes to either cudnn or cutlass.
    Note: trtllm is not considered in the backend selection because it requires a specific
    input quantization (swizzling/shuffling) that differs from the preparation used
    for cudnn and cutlass backends.

    Logic for which comes first:
    - If cuda version is 12 - use cutlass.
    - If cuda version is 13 and cudnn version is less than 9.15 - use cutlass.
    - If cuda version is 13 and cudnn version is 9.15 or greater - use cudnn.

    """
    cuda_major = get_cuda_version().major
    # If cuda version is 13 or greater:
    # cudnn is more performant if cudnn version is 9.15 or greater.
    if CUDNN_AVAILABLE and cuda_major >= 13 and cudnn.backend_version() >= 91500:
        candidate_backends = ("cudnn", "cutlass")
    # Otherwise, prioritize cutlass
    else:
        candidate_backends = ("cutlass", "cudnn")

    # Filter and return only supported backends
    return [c for c in candidate_backends if c in suitable_backends]


def _pad_up(x, y):
    return ((x + y - 1) // y) * y


_MM_FP4_TUNING_CONFIG_8x4 = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),
            get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            2,  # a_scale_tensor_index
            0,
            lambda shapes: _pad_up(shapes[0][0], 8),
        ),
        ConstraintSpec(
            6,  # out_tensor_index
            0,
            lambda shapes: shapes[0][0],
        ),
    ),
)


_MM_FP4_TUNING_CONFIG_128x4 = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),
            get_last_power_of_2_num_tokens_buckets,
            last_positive_power_of_2,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            2,  # a_scale_tensor_index
            0,
            lambda shapes: _pad_up(shapes[0][0], 128),
        ),
        ConstraintSpec(
            6,  # out_tensor_index
            0,
            lambda shapes: shapes[0][0],
        ),
    ),
)


@backend_requirement(
    {
        "cudnn": _cudnn_gemm_fp4_requirement,
        "trtllm": _trtllm_gemm_fp4_requirement,
        "cutlass": _cutlass_gemm_fp4_requirement,
    },
    common_check=_check_mm_fp4_problem_size,
    heuristic_func=_heuristic_func_mm_fp4,  # result stored in mm_fp4.suitable_auto_backends
)
@flashinfer_api
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
    backend: Literal["cudnn", "trtllm", "cutlass", "auto"] = "auto",
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
        Output dtype, bf16 or fp16. When ``backend="trtllm"``, only ``bf16`` is supported.

    out: Optional[torch.Tensor]
        Out tensor, shape (m, n), bf16 or fp16, defaults to ``None``.

    block_size: int
        Block size for FP4 quantization, only 16 and 32 are supported. 16 in case of nvfp4 quantization. 32 in case of mxfp4 quantization.

    use_8x4_sf_layout: bool
        Whether to use 8x4 scale factor layout or 128x4 scale factor layout, defaults to False.

    backend: Literal["cudnn", "trtllm", "cutlass", "auto"]
        Backend to use, defaults to ``"auto"``, which automatically selects the best
        backend between ``"cudnn"`` and ``"cutlass"`` based on the current CUDA and
        cuDNN versions. The ``"trtllm"`` backend is never selected when
        ``backend="auto"`` because it requires different weight preparation.

    use_nvfp4: bool
        Whether to use nvfp4 quantization or mxfp4 quantization, defaults to ``True``.
        See the ``block_size`` parameter for related constraints.

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

    # Auto-select the best backend
    if backend == "auto":
        backends = mm_fp4.suitable_auto_backends
    else:
        backends = [backend]

    # At this point, backends contains a supported backend if specified, or all supported backends if backend='auto'.
    # Lazy initialization of runners to avoid overhead of creating a new runner that will not be used
    major, _ = get_compute_capability(a.device)

    backend_to_runner_factory = {
        "cudnn": lambda: _cudnn_gemm_fp4_runner(),
        "trtllm": lambda: get_trtllm_fp4_gemm_module().trtllm_fp4_gemm_runner(
            use_8x4_sf_layout
        ),
        "cutlass": lambda: get_cutlass_fp4_gemm_module(major).cutlass_fp4_gemm_runner(),
    }
    runners = [backend_to_runner_factory[cur_backend]() for cur_backend in backends]

    # Now we have a list of runners for desired & supported backends.
    tuner = AutoTuner.get()

    tuning_config = (
        _MM_FP4_TUNING_CONFIG_8x4 if use_8x4_sf_layout else _MM_FP4_TUNING_CONFIG_128x4
    )

    inputs = [
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_nvfp4,
        workspace_buffer,
    ]
    runner, tactic = tuner.choose_one(
        "fp4_gemm",
        runners,
        tuning_config,
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)
    return out


@supported_compute_capability([89, 90, 100, 103, 120, 121])
def _cudnn_bmm_fp8_requirement(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cublas", "cutlass", "auto"] = "cublas",
):
    _check_cudnn_availability()
    return True


@supported_compute_capability([89, 90, 100, 103, 120, 121])
def _cublas_bmm_fp8_requirement(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cublas", "cutlass", "auto"] = "cublas",
):
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cutlass_bmm_fp8_requirement(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cublas", "cutlass", "auto"] = "cublas",
):
    if A.dtype == torch.float8_e5m2 or B.dtype == torch.float8_e5m2:
        raise ValueError("e5m2 is not supported for bmm_fp8 with cutlass backend")
    return True


def _check_bmm_fp8_problem_size(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cublas", "cutlass", "auto"] = "cublas",
):
    _validate_fp8_output_dtype(dtype)
    return True


def _heuristic_func_bmm_fp8(
    suitable_backends: List[str],
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cublas", "cutlass", "auto"] = "cublas",
):
    # No e5m2 for cutlass
    is_e5m2 = A.dtype == torch.float8_e5m2 or B.dtype == torch.float8_e5m2
    is_sm_supported = _match_sm_version(A.device, ["100", "103", "110"])
    is_sm120_supported = _match_sm_version(A.device, ["120", "121"])

    # preserve order of ["cudnn", "cublas", "cutlass"]
    heuristic_backends = []
    if "cutlass" in suitable_backends and not is_e5m2:
        if is_sm_supported:
            heuristic_backends.append("cutlass_sm10x")
        elif is_sm120_supported:
            k_dim = A.shape[-1] if A.dim() == 2 else A.shape[2]
            if k_dim >= 128:
                heuristic_backends.append("cutlass_sm12x")
    if "cublas" in suitable_backends:
        heuristic_backends.append("cublas")
    if CUDNN_AVAILABLE and "cudnn" in suitable_backends:
        heuristic_backends.append("cudnn")
    return heuristic_backends


@backend_requirement(
    {
        "cudnn": _cudnn_bmm_fp8_requirement,
        "cublas": _cublas_bmm_fp8_requirement,
        "cutlass": _cutlass_bmm_fp8_requirement,
    },
    common_check=_check_bmm_fp8_problem_size,
    heuristic_func=_heuristic_func_bmm_fp8,
)
@flashinfer_api
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

    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )

    workspace_buffer = _get_cache_buf(
        "bmm_fp8_workspace", DEFAULT_WORKSPACE_SIZE, A.device
    )

    if backend == "auto":
        backends = bmm_fp8.suitable_auto_backends
    elif backend == "cutlass":
        backends = _heuristic_func_bmm_fp8(
            ["cutlass"], A, B, A_scale, B_scale, dtype, out, backend
        )
    elif backend == "cudnn" and CUDNN_AVAILABLE:
        backends = ["cudnn"]
    else:
        backends = [backend]

    fp8_gemm_sm100(A, B, A_scale, B_scale, out, workspace_buffer, backends)
    return out


@supported_compute_capability([100, 103, 120, 121])
def _cutlass_gemm_fp8_nt_groupwise_requirement(
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
):
    if scale_major_mode is None:
        raise ValueError("scale_major_mode is required in CUTLASS")

    return True


@supported_compute_capability([100, 103])
def _trtllm_gemm_fp8_nt_groupwise_requirement(
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
):
    if scale_granularity_mnk != (1, 128, 128):
        raise ValueError("scale_granularity_mnk must be (1, 128, 128) in TRTLLM")
    if a.shape[1] < 256:
        raise ValueError("a.shape[1] must be >= 256 in TRTLLM")

    return True


def _check_gemm_fp8_nt_groupwise_problem_size(
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
):
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
    return True


@backend_requirement(
    {
        "cutlass": _cutlass_gemm_fp8_nt_groupwise_requirement,
        "trtllm": _trtllm_gemm_fp8_nt_groupwise_requirement,
    },
    common_check=_check_gemm_fp8_nt_groupwise_problem_size,
)
@flashinfer_api
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

    workspace_buffer = _get_cache_buf(
        "gemm_fp8_nt_groupwise_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
    else:
        out_dtype = out.dtype

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
        if is_sm120a_supported(a.device) or is_sm121a_supported(a.device):
            # SM120/121 doesn't use mma_sm parameter
            get_gemm_sm120_module().gemm_fp8_nt_groupwise(
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
            get_gemm_sm100_module().gemm_fp8_nt_groupwise(
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

    def trtllm_fp4_gemm_runner(use_8x4_sf_layout: bool = True):
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
                    a,
                    b,
                    a_descale,
                    b_descale,
                    alpha,
                    _,
                    out,
                    _,
                    _,
                    workspace_buffer,
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
                    a,
                    b,
                    a_descale,
                    b_descale,
                    alpha,
                    _,
                    out,
                    _,
                    _,
                    workspace_buffer,
                ) = inputs
                self._fp4_gemm_runner(
                    workspace_buffer,
                    a,
                    b.T,
                    a_descale,
                    b_descale.T,
                    alpha,
                    out,
                    self._use_8x4_sf_layout,
                    tactic,
                )
                return out

        return TrtllmFp4GemmRunner(use_8x4_sf_layout)

    # Register the module
    return SimpleNamespace(
        trtllm_fp4_gemm_runner=trtllm_fp4_gemm_runner,
    )


@supported_compute_capability([100, 103, 120, 121])
def _check_gemm_fp8_nt_blockscaled_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal["MN", "K"]] = "MN",
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    _check_gemm_fp8_nt_groupwise_problem_size(
        a,
        b,
        a_scale,
        b_scale,
        scale_major_mode,
        mma_sm,
        scale_granularity_mnk=(128, 128, 128),
        out=out,
        out_dtype=out_dtype,
        backend="cutlass",
    )

    _cutlass_gemm_fp8_nt_groupwise_requirement(
        a,
        b,
        a_scale,
        b_scale,
        scale_major_mode,
        mma_sm,
        scale_granularity_mnk=(128, 128, 128),
        out=out,
        out_dtype=out_dtype,
        backend="cutlass",
    )

    return True


@backend_requirement(
    {},
    common_check=_check_gemm_fp8_nt_blockscaled_problem_size,
)
@flashinfer_api
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


@supported_compute_capability([100, 103, 120, 121])
def _check_group_gemm_fp8_nt_groupwise_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    scale_major_mode: Literal["MN", "K"] = "MN",
    mma_sm: int = 1,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    if a.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(f"a must be a float8 tensor, but got {a.dtype}")
    if b.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(f"b must be a float8 tensor, but got {b.dtype}")
    if a_scale.dtype not in [torch.float32]:
        raise ValueError(f"a_scale must be a float32 tensor, but got {a_scale.dtype}")
    if b_scale.dtype not in [torch.float32]:
        raise ValueError(f"b_scale must be a float32 tensor, but got {b_scale.dtype}")
    if m_indptr.dtype not in [torch.int32]:
        raise ValueError(f"m_indptr must be a int32 tensor, but got {m_indptr.dtype}")
    if scale_major_mode not in ["MN", "K"]:
        raise ValueError(
            f"scale_major_mode must be either 'MN' or 'K', but got {scale_major_mode}"
        )
    if mma_sm not in [1, 2]:
        raise ValueError(f"mma_sm must be either 1 or 2, but got {mma_sm}")

    # assert a.shape[0] == m_indptr[-1].item()  # Not enabled in consideration of performance
    n = b.shape[1]
    k = b.shape[2]

    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype
        if out.shape != (a.shape[0], n):
            raise ValueError(
                f"Shape mismatch. out.shape = {out.shape}, (a.shape[0], n) = {(a.shape[0], n)}"
            )
        if out.dtype != out_dtype:
            raise ValueError(
                f"dtype mismatch. out.dtype = {out.dtype}, out_dtype = {out_dtype}"
            )

    _validate_fp8_output_dtype(out_dtype)

    if a.shape[1] != k:
        raise ValueError(f"Shape mismatch. a.shape[1] = {a.shape[1]}, k = {k}")
    if n % 8 != 0:
        raise ValueError(f"n must be a multiple of 8, but got {n}")
    if k % 16 != 0:
        raise ValueError(f"k must be a multiple of 16, but got {k}")

    num_groups = m_indptr.shape[0] - 1

    if is_sm120a_supported(a.device) or is_sm121a_supported(a.device):
        if num_groups > 1:
            raise RuntimeError(
                "group_gemm_fp8_nt_groupwise has correctness issues for num_groups > 1 on SM120/121"
            )

    return True


@backend_requirement(
    {},
    common_check=_check_group_gemm_fp8_nt_groupwise_problem_size,
)
@flashinfer_api
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
    int_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_int_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_float_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype

    n = b.shape[1]
    k = b.shape[2]

    out_shape = (a.shape[0], n)
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)

    if is_sm120a_supported(a.device) or is_sm121a_supported(a.device):
        # SM120/121 doesn't use mma_sm parameter
        get_gemm_sm120_module().group_gemm_fp8_nt_groupwise(
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
        get_gemm_sm100_module().group_gemm_fp8_nt_groupwise(
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
    return out


@supported_compute_capability([100, 103, 110, 120, 121])
def _check_group_gemm_mxfp8_mxfp4_nt_groupwise_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    mma_sm: int = 1,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    swap_ab: bool = True,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    if a.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"a must be a float8_e4m3fn or float8_e5m2 tensor, but got {a.dtype}"
        )
    if b.dtype != torch.uint8:
        raise ValueError(f"b must be a uint8 tensor, but got {b.dtype}")
    if a_scale.dtype != torch.uint8:
        raise ValueError(f"a_scale must be a uint8 tensor, but got {a_scale.dtype}")
    if b_scale.dtype != torch.uint8:
        raise ValueError(f"b_scale must be a uint8 tensor, but got {b_scale.dtype}")
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be a int32 tensor, but got {m_indptr.dtype}")
    if mma_sm not in [1, 2]:
        raise ValueError(f"mma_sm must be either 1 or 2, but got {mma_sm}")
    if tile_m not in [128]:
        raise ValueError(f"tile_m must be 128, but got {tile_m}")
    if tile_n not in [64, 128, 192, 256]:
        raise ValueError(f"tile_n must be one of [64, 128, 192, 256], but got {tile_n}")
    if tile_k not in [128, 256]:
        raise ValueError(f"tile_k must be either 128 or 256, but got {tile_k}")
    if swap_ab not in [True, False]:
        raise ValueError(f"swap_ab must be a boolean value, but got {swap_ab}")

    # Determine out_dtype if not specified
    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype

    if out_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"out_dtype must be either torch.bfloat16 or torch.float16, but got {out_dtype}"
        )

    num_groups = m_indptr.shape[0] - 1
    if b.shape[0] != num_groups:
        raise ValueError(
            f"b.shape[0] must equal num_groups (m_indptr.shape[0] - 1), but got b.shape[0]={b.shape[0]}, num_groups={num_groups}"
        )

    n = b.shape[1]
    k = b.shape[2] * 2  # Multiply by 2 because b is e2m1 packed as uint8

    # assert a.shape[0] == m_indptr[-1].item()  # Not enabled in consideration of performance
    if a.shape[1] != k:
        raise ValueError(
            f"a.shape[1] must equal k, but got a.shape[1]={a.shape[1]}, k={k}"
        )

    align_n = 8
    align_k = 128
    if n % align_n != 0:
        raise ValueError(f"n must be a multiple of {align_n}, but got n={n}")
    if k % align_k != 0:
        raise ValueError(f"k must be a multiple of {align_k}, but got k={k}")

    out_shape = (a.shape[0], n)
    if out is not None:
        if out.shape != out_shape:
            raise ValueError(f"out.shape must be {out_shape}, but got {out.shape}")
        if out.dtype != out_dtype:
            raise ValueError(f"out.dtype must be {out_dtype}, but got {out.dtype}")

    return True


@backend_requirement(
    {},
    common_check=_check_group_gemm_mxfp8_mxfp4_nt_groupwise_problem_size,
)
@flashinfer_api
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
    # Determine out_dtype if not specified
    if out is None:
        if out_dtype is None:
            out_dtype = torch.bfloat16
    else:
        if out_dtype is None:
            out_dtype = out.dtype

    n = b.shape[1]
    k = b.shape[2] * 2  # Multiply by 2 because b is e2m1 packed as uint8

    out_shape = (a.shape[0], n)
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=a.device)

    get_gemm_sm100_module().group_gemm_mxfp4_nt_groupwise(
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
    from ..triton.gemm import compute_padding_mapping

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


@functools.cache
def get_deepgemm_sm100_module():
    module = gen_deepgemm_sm100_module()
    return module


@supported_compute_capability([100, 103])
def _check_group_deepgemm_fp8_nt_groupwise_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> bool:
    from flashinfer.deep_gemm import (
        _check_group_deepgemm_fp8_nt_contiguous_problem_size,
    )

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(a.shape[0], b.shape[1], dtype=out_dtype, device=a.device)

    return _check_group_deepgemm_fp8_nt_contiguous_problem_size(
        (a, a_scale), (b, b_scale), out, m_indices, scale_granularity_mnk
    )


@backend_requirement(
    {},
    common_check=_check_group_deepgemm_fp8_nt_groupwise_problem_size,
)
@flashinfer_api
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

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(a.shape[0], b.shape[1], dtype=out_dtype, device=a.device)
    print("GOT HERE")
    m_grouped_fp8_gemm_nt_contiguous(
        (a, a_scale), (b, b_scale), out, m_indices, scale_granularity_mnk
    )

    return out


@supported_compute_capability([100, 103])
def _check_batch_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> bool:
    from flashinfer.deep_gemm import _check_m_grouped_fp8_gemm_nt_masked_problem_size

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(
            a.shape[0], a.shape[1], b.shape[1], dtype=out_dtype, device=a.device
        )

    return _check_m_grouped_fp8_gemm_nt_masked_problem_size(
        (a, a_scale), (b, b_scale), out, masked_m, expected_m, scale_granularity_mnk
    )


@backend_requirement(
    {},
    common_check=_check_batch_deepgemm_fp8_nt_groupwise,
)
@flashinfer_api
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

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(
            a.shape[0], a.shape[1], b.shape[1], dtype=out_dtype, device=a.device
        )

    m_grouped_fp8_gemm_nt_masked(
        (a, a_scale), (b, b_scale), out, masked_m, expected_m, scale_granularity_mnk
    )

    return out


@functools.cache
def get_fp8_blockscale_gemm_runner_sm90():
    """Get the FP8 block scale GEMM runner module for SM90."""
    module = gen_fp8_blockscale_gemm_sm90_module().build_and_load()
    from ..jit import env as jit_env

    deepgemm_include_dir = str(
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "tensorrt_llm"
    )
    module.set_deepgemm_jit_include_dirs([deepgemm_include_dir])
    return module.init()


@flashinfer_api
def fp8_blockscale_gemm_sm90(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Perform FP8 block-scaled GEMM with automatic swapAB optimization.
    This function automatically selects between normal and swapAB kernel based on
    the M dimension. For small M (< 32), it uses the swapAB kernel for
    better performance.

    Supported Dtype Combinations
    -----------------------------
    - **BF16 + BF16 → BF16**: Both inputs BF16, internal quantization (no scales needed)
    - **BF16 + FP8 → BF16**: BF16 input, FP8 weight
    - **FP8 + FP8 → BF16** (W8A8): Both inputs FP8 with scales required

    Parameters
    ----------
    input : torch.Tensor
        Input activation tensor of shape (M, K).
        - BF16 (torch.bfloat16) with internal quantization
    weight : torch.Tensor
        Weight tensor of shape (N, K). Can be:
        - FP8 (torch.float8_e4m3fn) with weight_scale required
        - BF16 (torch.bfloat16) for internal quantization
    input_scale : torch.Tensor, optional
    weight_scale : torch.Tensor, optional
        Scaling factors for weight. Required if weight is FP8.
    out : torch.Tensor, optional
        Output tensor of shape (M, N). If None, will be allocated.
    out_dtype : torch.dtype, optional
        Output data type. Default is torch.bfloat16.
    Returns
    -------
    torch.Tensor
        Output tensor of shape (M, N) with dtype `out_dtype`.
    Examples
    --------
    >>> import torch
    >>> from flashinfer.gemm import fp8_blockscale_gemm_sm90
    >>>
    >>> M, N, K = 16, 4096, 4096
    >>> device = "cuda"
    >>>
    >>> # BF16 inputs
    >>> input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    >>> weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    >>> output = fp8_blockscale_gemm_sm90(input_bf16, weight_bf16)
    >>> print(output.shape)  # torch.Size([16, 4096])
    >>>
    >>> # Mixed: BF16 input + FP8 weight
    >>> from flashinfer.testing.utils import per_token_cast_to_fp8
    >>> input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    >>> weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    >>> weight_fp8, weight_scale = per_token_cast_to_fp8(weight_bf16)
    >>> output = fp8_blockscale_gemm_sm90(input_bf16, weight_fp8, None, weight_scale)
    >>> print(output.shape)  # torch.Size([16, 4096])
    >>>
    >>> # FP8 weight with 128x128 block scales
    >>> from flashinfer.testing.utils import per_block_cast_to_fp8
    >>> weight_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    >>> weight_fp8, weight_scale = per_block_cast_to_fp8(weight_bf16)
    >>> # weight_scale has shape (N // 128, K // 128)
    >>> input_bf16 = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    >>> output = fp8_blockscale_gemm_sm90(input_bf16, weight_fp8, None, weight_scale)
    >>> print(output.shape)  # torch.Size([16, 4096])
    Notes
    -----
    - This function requires NVIDIA Hopper (SM90) architecture and CUDA 12.8+
    - SwapAB kernel is automatically used when M < 32 (threshold)
    - For FP8 inputs, scaling factors must be provided
    - For BF16 inputs, quantization and scaling happen internally
    - Weight scales support two granularities:
      * Per-token (1x128 blocks): (N, K//128)
      * Per-block (128x128 blocks): (N//128, K//128)
    - Input scales only support per-token format: (M, K//128)
    - The function uses DeepGEMM backend with JIT compilation
    """
    # Validate architecture support
    if not _match_sm_version(input.device, ["90", "90a"]):
        raise ValueError(
            "fp8_blockscale_gemm_sm90 is only supported on SM90 (Hopper) architecture."
        )

    # Validate tensor dimensions
    if input.ndim != 2:
        raise ValueError(f"Input must be 2D (M, K), got shape {input.shape}")
    if weight.ndim != 2:
        raise ValueError(f"Weight must be 2D (N, K), got shape {weight.shape}")

    M, K = input.shape
    N, K_weight = weight.shape

    if K_weight != K:
        raise ValueError(
            f"K dimension mismatch: input has K={K}, weight has K={K_weight}"
        )

    # Validate K is divisible by block size (128)
    BLOCK_SIZE = 128
    if K % BLOCK_SIZE != 0:
        raise ValueError(
            f"K dimension must be divisible by block size ({BLOCK_SIZE}), got K={K}"
        )

    if N % 64 != 0:
        raise ValueError(f"N dimension must be divisible by 64, got N={N}")

    # Validate dtype combinations
    input_is_fp8 = input.dtype == torch.float8_e4m3fn
    weight_is_fp8 = weight.dtype == torch.float8_e4m3fn
    input_is_bf16 = input.dtype == torch.bfloat16
    weight_is_bf16 = weight.dtype == torch.bfloat16

    # Explicitly reject FP8 input + BF16 weight (missing kernel implementation)
    if input_is_fp8 and weight_is_bf16:
        raise ValueError(
            "FP8 input + BF16 weight is not supported (missing kernel implementation). "
        )

    # Validate scale requirements for FP8 inputs
    if input_is_fp8:
        if input_scale is None:
            raise ValueError("input_scale is required when input is FP8. ")
        if input_scale.dtype != torch.float32:
            raise ValueError(f"input_scale must be float32, got {input_scale.dtype}")
        if input_scale.device != input.device:
            raise ValueError(
                f"input_scale device mismatch. Expected {input.device}, "
                f"got {input_scale.device}"
            )
    else:
        if not input_is_bf16:
            raise ValueError(
                f"Input must be either FP8 (torch.float8_e4m3fn) or BF16 (torch.bfloat16), "
                f"got {input.dtype}"
            )
        if input_scale is not None:
            raise ValueError(
                "input_scale should not be provided for BF16 inputs. "
                "Use FP8 inputs if you want to provide external scales."
            )

    if weight_is_fp8:
        if weight_scale is None:
            raise ValueError("weight_scale is required when weight is FP8. ")
        expected_per_token_shape = (N, K // BLOCK_SIZE)
        expected_per_block_shape = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE, K // BLOCK_SIZE)
        is_per_token = weight_scale.shape == expected_per_token_shape
        is_per_block = weight_scale.shape == expected_per_block_shape

        if not (is_per_token or is_per_block):
            raise ValueError(
                f"weight_scale shape mismatch. Expected either {expected_per_token_shape} "
                f"(per-token, 1x128 blocks) or {expected_per_block_shape} "
                f"(per-block, 128x128 blocks), got {weight_scale.shape}"
            )
        if weight_scale.dtype != torch.float32:
            raise ValueError(f"weight_scale must be float32, got {weight_scale.dtype}")
    else:
        if not weight_is_bf16:
            raise ValueError(
                f"Weight must be either FP8 (torch.float8_e4m3fn) or BF16 (torch.bfloat16), "
                f"got {weight.dtype}"
            )
        if weight_scale is not None:
            raise ValueError(
                "weight_scale should not be provided for BF16 weights. "
                "Use FP8 weights if you want to provide external scales."
            )

    # Validate output tensor if provided
    if out is not None:
        if out.shape != (M, N):
            raise ValueError(
                f"Output shape mismatch. Expected ({M}, {N}), got {out.shape}"
            )
        if out.device != input.device:
            raise ValueError(
                f"Output device mismatch. Expected {input.device}, got {out.device}"
            )
        if out.dtype not in [torch.bfloat16, torch.float16]:
            raise ValueError(
                f"Output dtype must be torch.bfloat16 or torch.float16, got {out.dtype}"
            )
        if out_dtype is not None and out.dtype != out_dtype:
            raise ValueError(
                f"Output dtype mismatch. Expected {out_dtype}, got {out.dtype}"
            )
        out_dtype = out.dtype
    else:
        # Allocate output
        out_dtype = out_dtype or torch.bfloat16
        if out_dtype not in [torch.bfloat16, torch.float16]:
            raise ValueError(
                f"Output dtype must be torch.bfloat16 or torch.float16, got {out_dtype}"
            )
        out = torch.empty(M, N, dtype=out_dtype, device=input.device)

    # Get the runner
    runner = get_fp8_blockscale_gemm_runner_sm90()

    # Allocate workspace
    workspace_size = runner.get_workspace_size(M, N, K)
    workspace = None
    if workspace_size > 0:
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=input.device)
        runner.configure_workspace(workspace)

    runner.run_gemm(input, weight, out, input_scale, weight_scale)
    return out
