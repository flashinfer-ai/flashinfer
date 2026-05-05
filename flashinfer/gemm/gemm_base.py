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
from ..trace.templates.gemm import (
    batch_deepgemm_fp8_nt_groupwise_trace,
    bmm_bf16_trace,
    bmm_fp8_trace,
    bmm_mxfp8_trace,
    fp8_blockscale_gemm_sm90_trace,
    mm_bf16_trace,
    mm_fp4_trace,
    mm_fp8_trace,
    mm_mxfp8_trace,
)
from ..trace.templates.attention import segment_gemm_run_trace
from ..trace.templates.page import tgv_gemm_sm100_trace
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from .kernels.utils import _select_sm100_mm_fp4_cute_dsl_tactic
from ..utils import (
    get_device_sm_count,
    get_native_fp4_dtype,
    is_sm100a_supported,
    is_sm100f_supported,
    is_sm12x_supported,
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
from ..jit.gemm import gen_gemm_sm103_module_cutlass_fp4
from ..jit.gemm import gen_gemm_sm100_module_cutlass_fp8
from ..jit.gemm import gen_gemm_sm100_module_cutlass_mxfp8
from ..jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8
from ..jit.gemm import gen_gemm_sm100_module_cutlass_bf16
from ..jit.gemm import gen_mm_bf16_cublaslt_module
from ..jit.gemm import gen_trtllm_gen_gemm_module
from ..jit.gemm import gen_tgv_gemm_sm10x_module
from ..jit.gemm import gen_deepgemm_sm100_module
from ..jit.cpp_ext import get_cuda_version
from ..jit.gemm import gen_fp8_blockscale_gemm_sm90_module
from ..tllm_enums import DtypeTrtllmGen, SfLayout
from .routergemm import get_tinygemm2_module


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

# sizeof(cublasLtMatmulAlgo_t) = uint64_t[8] = 64 bytes.
# Shared by cuBLAS FP8, cuBLASLt BF16, and any other cuBLASLt-based runners.
_CUBLASLT_ALGO_BYTES = 64
_CUBLASLT_MAX_ALGOS = 100

# Error messages
CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR = "cudnn FP4 GEMM with mxfp4 quantization is not supported on SM120/SM121 with cuDNN backend version < 9.14.0."

_TORCH_TO_CUTLASS_DTYPE_ATTR = {
    torch.bfloat16: "BFloat16",
    torch.float16: "Float16",
}


def _match_sm_version(device: torch.device, sm_version: list[str]):
    major, minor = get_compute_capability(device)
    device_arch = f"{major * 10 + minor}"
    return device_arch in sm_version


@functools.cache
def get_gemm_module():
    module = gen_gemm_module().build_and_load()

    def cublas_fp8_gemm_runner():
        class CublasFp8GemmRunner(TunableRunner):
            def __init__(self):
                self._algo_cache: dict = {}

            def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
                a, b, _, _, out, _ = inputs
                return (a.shape, b.shape, a.dtype, b.dtype, out.dtype)

            def _get_algos(self, inputs):
                a, b, scale_a, scale_b, out, workspace_buffer = inputs
                key = self.get_cache_key_extras(inputs)
                cached = self._algo_cache.get(key)
                if cached is not None:
                    return cached
                algo_buf = torch.empty(
                    _CUBLASLT_MAX_ALGOS * _CUBLASLT_ALGO_BYTES,
                    dtype=torch.uint8,
                    device="cpu",
                )
                with torch.cuda.device(a.device):
                    cublas_handle = torch.cuda.current_blas_handle()
                count = module.bmm_fp8_get_algos(
                    a,
                    b,
                    out,
                    scale_a,
                    scale_b,
                    workspace_buffer,
                    cublas_handle,
                    algo_buf,
                )
                result = (algo_buf, count)
                self._algo_cache[key] = result
                return result

            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                _, count = self._get_algos(inputs)
                return list(range(count))

            def forward(
                self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs,
            ) -> torch.Tensor:
                a, b, scale_a, scale_b, out, workspace_buffer = inputs
                with torch.cuda.device(a.device):
                    cublas_handle = torch.cuda.current_blas_handle()
                if tactic >= 0:
                    algo_buf, count = self._get_algos(inputs)
                    if count == 0:
                        raise RuntimeError(
                            "cuBLASLt heuristic returned zero FP8 algorithms for "
                            f"A={tuple(a.shape)}, B={tuple(b.shape)}, out={tuple(out.shape)}."
                        )
                    if tactic >= count:
                        raise ValueError(
                            f"Requested tactic {tactic} but only {count} algorithms "
                            f"available for A={tuple(a.shape)}, B={tuple(b.shape)}."
                        )
                    module.bmm_fp8_run_with_algo(
                        a,
                        b,
                        out,
                        scale_a,
                        scale_b,
                        workspace_buffer,
                        cublas_handle,
                        algo_buf,
                        tactic,
                    )
                else:
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


@supported_compute_capability([100, 103])
def _cutlass_mm_bf16_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    backend: Literal[
        "cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"
    ] = "cudnn",
):
    if bias is not None:
        raise ValueError(
            "You cannot use the CUTLASS backend with a bias. Use the TGV backend instead."
        )
    if pdl:
        raise ValueError(
            "The CUTLASS backend does not support PDL. Use the TGV backend instead."
        )

    _validate_bf16_output_dtype(out_dtype)

    return True


# Gated to Blackwell (SM100/SM103) for the initial scope of this backend.
# cuBLASLt supports BF16 GEMM on SM80+; the gate can be widened in a follow-up.
@supported_compute_capability([100, 103])
def _cublaslt_mm_bf16_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    backend: Literal["cudnn", "cutlass", "tgv", "cublaslt", "auto"] = "cudnn",
):
    if bias is not None:
        raise ValueError(
            "You cannot use the cuBLASLt backend with a bias. Use the TGV backend instead."
        )
    if pdl:
        raise ValueError(
            "The cuBLASLt backend does not support PDL. Use the TGV backend instead."
        )
    _validate_bf16_output_dtype(out_dtype)

    return True


@supported_compute_capability([80, 86, 87, 89, 90, 100, 103, 110, 120, 121])
def _cudnn_mm_bf16_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    backend: Literal[
        "cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"
    ] = "cudnn",
):
    _validate_bf16_output_dtype(out_dtype)
    _check_cudnn_availability()

    return True


@supported_compute_capability([100, 103])
def _tgv_gemm_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    backend: Literal[
        "cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"
    ] = "cudnn",
):
    if out_dtype != torch.bfloat16:
        raise ValueError(
            "You cannot provide an output dtype to the TGV backend. Use the CUTLASS or cuDNN backend instead."
        )
    return True


@supported_compute_capability([90, 100, 103, 110, 120, 121])
def _tinygemm_mm_bf16_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    backend: Literal["cudnn", "cutlass", "tgv", "tinygemm", "auto"] = "cudnn",
):
    if out_dtype != torch.bfloat16:
        raise ValueError("The TinyGEMM backend only supports bfloat16 output.")
    if a.dim() != 2:
        raise ValueError("The TinyGEMM backend requires a 2D input tensor.")
    if b.dim() != 2:
        raise ValueError("The TinyGEMM backend requires a 2D weight tensor.")
    if not a.is_contiguous():
        raise ValueError("The TinyGEMM backend requires a contiguous input tensor.")
    if not b.transpose(-2, -1).is_contiguous():
        raise ValueError(
            "The TinyGEMM backend requires b.T to be contiguous. "
            "Pass b as the transpose of a contiguous row-major weight tensor."
        )
    if b.shape[1] % 16 != 0:
        raise ValueError(
            "The TinyGEMM backend requires the output feature dimension to be a multiple of 16."
        )
    if out is not None and not out.is_contiguous():
        raise ValueError("The TinyGEMM backend requires a contiguous output tensor.")
    if bias is not None and not bias.is_contiguous():
        raise ValueError("The TinyGEMM backend requires a contiguous bias tensor.")
    return True


def _check_mm_bf16_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal[
        "cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"
    ] = "cudnn",
):
    if a.dtype != torch.bfloat16:
        raise ValueError(
            f"First tensor has unsupported dtype {a.dtype}. Only bfloat16 is supported."
        )
    if b.dtype != torch.bfloat16:
        raise ValueError(
            f"Second tensor has unsupported dtype {b.dtype}. Only bfloat16 is supported."
        )

    if bias is not None and bias.dtype != torch.bfloat16:
        raise ValueError(
            f"Bias tensor has unsupported dtype {bias.dtype}. Only bfloat16 is supported."
        )

    if out is not None:
        if out.shape != (a.shape[0], b.shape[1]):
            raise ValueError(
                f"Output shape mismatch. Expected {(a.shape[0], b.shape[1])}, got {out.shape}."
            )
        if out.device != a.device:
            raise ValueError(
                f"Output device mismatch. Expected {a.device}, got {out.device}."
            )
        if out.dtype != out_dtype:
            raise ValueError(
                f"Output dtype mismatch. Expected {out_dtype}, got {out.dtype}."
            )

    return True


def _heuristic_func_mm_bf16(
    suitable_backends: List[str],
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal[
        "cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"
    ] = "cudnn",
):
    heuristic_backends = []
    if bias is not None or pdl:
        # CUTLASS and cuBLASLt doesn't support bias/pdl, only TGV and cuDNN do
        if "tgv" in suitable_backends:
            heuristic_backends.append("tgv")
        if "cudnn" in suitable_backends:
            heuristic_backends.append("cudnn")
    else:
        if "cutlass" in suitable_backends:
            heuristic_backends.append("cutlass")
        if "tgv" in suitable_backends:
            heuristic_backends.append("tgv")
        if "cudnn" in suitable_backends:
            heuristic_backends.append("cudnn")
        if "cublaslt" in suitable_backends:
            heuristic_backends.append("cublaslt")

    if "tinygemm" in suitable_backends and out_dtype == torch.bfloat16:
        heuristic_backends.append("tinygemm")

    return heuristic_backends


@backend_requirement(
    {
        "cudnn": _cudnn_mm_bf16_requirement,
        "cutlass": _cutlass_mm_bf16_requirement,
        "tgv": _tgv_gemm_requirement,
        "cublaslt": _cublaslt_mm_bf16_requirement,
        "tinygemm": _tinygemm_mm_bf16_requirement,
    },
    common_check=_check_mm_bf16_problem_size,
    heuristic_func=_heuristic_func_mm_bf16,
)
@flashinfer_api(trace=mm_bf16_trace)
def mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal[
        "cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"
    ] = "cudnn",
) -> torch.Tensor:
    r"""MM BF16

    Parameters
    ----------
    a: torch.Tensor
        Input tensor, shape (m, k), bf16 in row-major layout.

    b: torch.Tensor
        Weight tensor, shape (k, n), bf16 in column-major layout.

    bias: Optional[torch.Tensor]
        Optional bias tensor, shape (n,). Enabled for TGV and TinyGEMM backends. Defaults to ``None``.

    pdl: bool
        Whether to use Programmatic Dependent Launch. Enabled for TGV and TinyGEMM backends. Defaults to ``False``.

    out: Optional[torch.Tensor]
        Out tensor, shape (m, n), bf16, fp16, or fp32. FP16 and FP32 output are enabled
        for CUTLASS and cuDNN backends; TinyGEMM requires bf16 output.

    out_dtype: torch.dtype
        Output dtype, bf16, fp16, or fp32. Enabled for CUTLASS, cuDNN, and cuBLASLt backends.
        Defaults to ``torch.bfloat16``.

    backend: Literal["cudnn", "cutlass", "tgv", "cublaslt", "tinygemm", "auto"]
        The backend to use for the operation. Defaults to ``"cudnn"``.
        ``"cudnn"`` uses the cuDNN backend.
        ``"cutlass"`` uses the CUTLASS backend.
        ``"tgv"`` uses the TGV backend.
        ``"cublaslt"`` uses the cuBLASLt backend with heuristic algorithm search.
        ``"tinygemm"`` uses the TinyGEMM backend for small-M BF16 GEMM.
        ``"auto"`` allows selecting the best tactic from all available backends when autotune is enabled.

    Returns
    -------
    torch.Tensor
        Out tensor, shape (m, n), bf16, fp16, or fp32 in row-major layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> # Using the TGV backend
    >>> a = torch.randn([48, 64], device="cuda", dtype=torch.bfloat16)
    >>> b = torch.randn([80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    >>> bias = torch.randn([80], device="cuda", dtype=torch.bfloat16)
    >>> out = flashinfer.mm_bf16(a, b, bias=bias, pdl=True, backend="tgv")
    >>> out.shape
    torch.Size([48, 80])
    >>> out.dtype
    torch.bfloat16
    >>> # Using the CUTLASS backend
    >>> fp16_out = torch.empty([48, 80], device="cuda", dtype=torch.float16)
    >>> out = flashinfer.mm_bf16(a, b, out=fp16_out, out_dtype=torch.float16, backend="cutlass")
    >>> out.shape
    torch.Size([48, 80])
    >>> out.dtype
    torch.float16
    >>> # Using the cuDNN backend
    >>> out = flashinfer.mm_bf16(a, b, backend="cudnn")
    >>> out.shape
    torch.Size([48, 80])
    >>> out.dtype
    torch.bfloat16
    >>> # Using the cuBLASLt backend
    >>> out = flashinfer.mm_bf16(a, b, backend="cublaslt")
    >>> out.shape
    torch.Size([48, 80])
    >>> out.dtype
    torch.bfloat16
    """

    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]),
            device=a.device,
            dtype=out_dtype,
        )

    workspace_buffer = _get_cache_buf(
        "mm_bf16_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )
    if backend == "auto":
        backends = mm_bf16.suitable_auto_backends
    elif backend == "cudnn":
        backends = _heuristic_func_mm_bf16(
            ["cudnn"], a, b, None, False, out, out_dtype, backend
        )
    elif backend == "cutlass":
        backends = _heuristic_func_mm_bf16(
            ["cutlass"], a, b, None, False, out, out_dtype, backend
        )
    elif backend == "tgv":
        backends = _heuristic_func_mm_bf16(
            ["tgv"], a, b, bias, pdl, out, out_dtype, backend
        )
    elif backend == "cublaslt":
        backends = _heuristic_func_mm_bf16(
            ["cublaslt"], a, b, None, False, out, out_dtype, backend
        )
    elif backend == "tinygemm":
        backends = _heuristic_func_mm_bf16(
            ["tinygemm"], a, b, bias, pdl, out, out_dtype, backend
        )
    else:
        backends = [backend]

    bf16_gemm_sm100(a, b, bias, pdl, out, workspace_buffer, backends)
    return out


@supported_compute_capability([100, 103])
def _cutlass_bmm_bf16_requirement(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cudnn", "cutlass", "auto"] = "cudnn",
):
    _validate_bf16_output_dtype(out_dtype)

    return True


@supported_compute_capability([100, 103])
def _cudnn_bmm_bf16_requirement(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cudnn", "cutlass", "auto"] = "cudnn",
):
    _validate_bf16_output_dtype(out_dtype)
    _check_cudnn_availability()
    return True


def _check_bmm_bf16_problem_size(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cudnn", "cutlass", "auto"] = "cudnn",
):
    if A.dtype != torch.bfloat16:
        raise ValueError(
            f"First tensor has unsupported dtype {A.dtype}. Only bfloat16 is supported."
        )
    if B.dtype != torch.bfloat16:
        raise ValueError(
            f"Second tensor has unsupported dtype {B.dtype}. Only bfloat16 is supported."
        )

    if out is not None:
        expected_shape = (A.shape[0], A.shape[1], B.shape[2])
        if out.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch. Expected {expected_shape}, got {out.shape}."
            )
        if out.device != A.device:
            raise ValueError(
                f"Output device mismatch. Expected {A.device}, got {out.device}."
            )
        if out.dtype != out_dtype:
            raise ValueError(
                f"Output dtype mismatch. Expected {out_dtype}, got {out.dtype}."
            )

    return True


def _heuristic_func_bmm_bf16(
    suitable_backends: List[str],
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cudnn", "cutlass", "auto"] = "cudnn",
):
    heuristic_backends = []
    if "cudnn" in suitable_backends:
        heuristic_backends.append("cudnn")
    if "cutlass" in suitable_backends:
        heuristic_backends.append("cutlass")
    return heuristic_backends


@backend_requirement(
    {
        "cutlass": _cutlass_bmm_bf16_requirement,
        "cudnn": _cudnn_bmm_bf16_requirement,
    },
    common_check=_check_bmm_bf16_problem_size,
    heuristic_func=_heuristic_func_bmm_bf16,
)
@flashinfer_api(trace=bmm_bf16_trace)
def bmm_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cudnn", "cutlass", "auto"] = "cudnn",
) -> torch.Tensor:
    r"""BMM BF16

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (b, m, k), bf16 in row-major layout.

    B: torch.Tensor
        Weight tensor, shape (b, k, n), bf16 in column-major layout.

    out: Optional[torch.Tensor]
        Out tensor, shape (b, m, n), bf16, fp16, or fp32, defaults to ``None``.

    out_dtype: torch.dtype
        Output dtype, bf16 (default), fp16, or fp32.

    backend: Literal["cudnn", "cutlass", "auto"]
        Backend to use, defaults to "cudnn". ``"auto"`` allows selecting the best tactic from all available backends when autotune is enabled.

    Returns
    -------
    torch.Tensor
        Out tensor, shape (b, m, n), bf16, fp16, or fp32 in row-major layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> # Using the CUTLASS backend
    >>> input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
    >>> weight = torch.randn([16, 80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    >>> fp16_out = torch.empty([16, 48, 80], device="cuda", dtype=torch.float16)
    >>> out = flashinfer.bmm_bf16(input, weight, out=fp16_out, out_dtype=torch.float16, backend="cutlass")
    >>> out.shape
    torch.Size([16, 48, 80])
    >>> out.dtype
    torch.float16
    >>> # using the cuDNN backend
    >>> out = flashinfer.bmm_bf16(input, weight, backend="cudnn")
    >>> out.shape
    torch.Size([16, 48, 80])
    >>> out.dtype
    torch.bfloat16
    """

    expected_shape = (A.shape[0], A.shape[1], B.shape[2])
    if out is None:
        out = torch.empty(
            expected_shape,
            device=A.device,
            dtype=out_dtype,
        )

    workspace_buffer = _get_cache_buf(
        "bmm_bf16_workspace", DEFAULT_WORKSPACE_SIZE, A.device
    )

    if backend == "auto":
        backends = bmm_bf16.suitable_auto_backends
    else:
        backends = [backend]

    bf16_gemm_sm100(A, B, None, False, out, workspace_buffer, backends)
    return out


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
                # TODO: add multi-tactic support when PingPong 64x128x128 schedule
                # is implemented (see gemm_groupwise_sm120.cuh)
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
                # SM120/SM121 kernel now supports batch operations natively
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

                scale_gran_m = 1
                scale_gran_n = 128
                scale_gran_k = 128

                # round up to the next multiple
                def _pad_to_multiple(x, multiple):
                    return ((x + multiple - 1) // multiple) * multiple

                # SM120/SM121 CUTLASS blockwise scaling requires:
                # - N % 128 == 0 (ScaleGranularityN)
                # - K % 128 == 0 (TileK)
                # If not aligned, we pad and then slice the result
                n_padded = _pad_to_multiple(n_dim, scale_gran_n)
                k_padded = _pad_to_multiple(k_dim, scale_gran_k)
                needs_n_padding = n_padded != n_dim
                needs_k_padding = k_padded != k_dim

                if not needs_k_padding and not needs_n_padding:
                    # No padding needed
                    a_padded = a
                    b_col_major_padded = b_col_major
                else:
                    # Padding needed
                    if a.dim() == 2:
                        a_padded = a
                        if needs_k_padding:
                            a_padded = torch.nn.functional.pad(
                                a_padded.contiguous(), (0, k_padded - k_dim)
                            )
                        b_col_major_padded = torch.zeros(
                            (n_padded, k_padded),
                            dtype=b_col_major.dtype,
                            device=b_col_major.device,
                        )
                        b_col_major_padded[:n_dim, :k_dim].copy_(b_col_major)
                    else:
                        a_padded = a
                        if needs_k_padding:
                            a_padded = torch.nn.functional.pad(
                                a_padded.contiguous(), (0, k_padded - k_dim)
                            )

                        b_underlying_padded = torch.zeros(
                            (batch_size, n_padded, k_padded),
                            dtype=b_col_major.dtype,
                            device=b_col_major.device,
                        )
                        b_col_major_padded = b_underlying_padded.transpose(-2, -1)
                        b_col_major_padded[:, :k_dim, :n_dim].copy_(b_col_major)

                # Create padded output if needed
                if needs_n_padding:
                    if a.dim() == 2:
                        out_padded = torch.empty(
                            (m_dim, n_padded), device=out.device, dtype=out.dtype
                        )
                    else:
                        out_padded = torch.empty(
                            (batch_size, m_dim, n_padded),
                            device=out.device,
                            dtype=out.dtype,
                        )
                else:
                    out_padded = out

                # For scalar scales, create compatible shapes for SM120/SM121
                if scale_a.numel() == 1:
                    scale_m_count = (
                        batch_size * m_dim + scale_gran_m - 1
                    ) // scale_gran_m
                    scale_k_count = (k_padded + scale_gran_k - 1) // scale_gran_k
                    scale_a_expanded = (
                        scale_a.view(1, 1)
                        .expand(scale_m_count, scale_k_count)
                        .contiguous()
                    )
                else:
                    scale_a_expanded = scale_a

                if scale_b.numel() == 1:
                    scale_n_count = (
                        batch_size * n_padded + scale_gran_n - 1
                    ) // scale_gran_n
                    scale_k_count = (k_padded + scale_gran_k - 1) // scale_gran_k
                    scale_b_expanded = (
                        scale_b.view(1, 1)
                        .expand(scale_n_count, scale_k_count)
                        .contiguous()
                    )
                else:
                    scale_b_expanded = scale_b

                # Call SM120/SM121 gemm_fp8_nt_groupwise (now handles both 2D and 3D)
                module.gemm_fp8_nt_groupwise(
                    workspace_buffer,
                    a_padded,
                    b_col_major_padded,
                    scale_a_expanded,
                    scale_b_expanded,
                    out_padded,
                    scale_gran_m,  # scale_granularity_m
                    scale_gran_n,  # scale_granularity_n
                    scale_gran_k,  # scale_granularity_k (adjusted for small k)
                    "MN",  # scale_major_mode
                )

                # Slice the result if we padded
                if needs_n_padding:
                    if a.dim() == 2:
                        out.copy_(out_padded[:, :n_dim])
                    else:
                        out.copy_(out_padded[:, :, :n_dim])

                return out

        return CutlassFp8GemmRunner()

    # Register the module
    return SimpleNamespace(
        cutlass_fp8_gemm_runner=cutlass_fp8_gemm_runner,
    )


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
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
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


@functools.cache
def get_gemm_sm100_module_cutlass_bf16():
    module = gen_gemm_sm100_module_cutlass_bf16().build_and_load()

    def cutlass_bf16_gemm_runner():
        class CutlassBf16GemmRunner(TunableRunner):
            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                return list(range(module.bf16_gemm_tactic_num()))

            def forward(
                self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs,
            ) -> torch.Tensor:
                a, b, _, _, out, workspace_buffer = inputs
                module.bf16_gemm(
                    a,
                    b.transpose(-2, -1),
                    out,
                    workspace_buffer,
                    tactic,
                )
                return out

        return CutlassBf16GemmRunner()

    return SimpleNamespace(
        cutlass_bf16_gemm_runner=cutlass_bf16_gemm_runner,
    )


@functools.cache
def get_mm_bf16_cublaslt_module():
    module = gen_mm_bf16_cublaslt_module().build_and_load()

    def cublaslt_bf16_gemm_runner():
        class CublasltBf16GemmRunner(TunableRunner):
            def __init__(self):
                self._algo_cache: dict = {}

            def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
                a, b, _, _, out, _ = inputs
                return (
                    a.shape[0],
                    b.shape[1],
                    a.shape[1],
                    self._compute_dtype(out.dtype),
                )

            @staticmethod
            def _compute_dtype(out_dtype):
                # cuBLASLt with BF16 inputs supports BF16 or FP32 output natively.
                # FP16 output is achieved via FP32 compute + cast (FP32→FP16
                # preserves more precision than BF16→FP16).
                if out_dtype == torch.float16:
                    return torch.float32
                return out_dtype

            def _get_algos(self, inputs):
                a, b, _, _, out, workspace_buffer = inputs
                compute_dt = self._compute_dtype(out.dtype)
                key = self.get_cache_key_extras(inputs)
                cached = self._algo_cache.get(key)
                if cached is not None:
                    return cached
                algo_buf = torch.empty(
                    _CUBLASLT_MAX_ALGOS * _CUBLASLT_ALGO_BYTES,
                    dtype=torch.uint8,
                    device="cpu",
                )
                with torch.cuda.device(a.device):
                    cublas_handle = torch.cuda.current_blas_handle()
                proxy_out = (
                    out
                    if out.dtype == compute_dt
                    else torch.empty_like(out, dtype=compute_dt)
                )
                count = module.mm_bf16_cublaslt_get_algos(
                    a,
                    b.transpose(-2, -1),
                    proxy_out,
                    workspace_buffer,
                    cublas_handle,
                    algo_buf,
                )
                result = (algo_buf, count)
                self._algo_cache[key] = result
                return result

            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                _, count = self._get_algos(inputs)
                return list(range(count))

            def forward(
                self,
                inputs: List[torch.Tensor],
                tactic: int = -1,
                do_preparation: bool = False,
                **kwargs,
            ) -> torch.Tensor:
                a, b, _, _, out, workspace_buffer = inputs
                with torch.cuda.device(a.device):
                    cublas_handle = torch.cuda.current_blas_handle()
                b_t = b.transpose(-2, -1)

                compute_dt = self._compute_dtype(out.dtype)
                need_cast = out.dtype != compute_dt
                if need_cast:
                    compute_out = torch.empty_like(out, dtype=compute_dt)
                else:
                    compute_out = out

                algo_buf, count = self._get_algos(inputs)
                if count == 0:
                    raise RuntimeError(
                        "cuBLASLt heuristic returned zero algorithms for "
                        f"M={a.shape[0]}, N={b.shape[1]}, K={a.shape[1]}, "
                        f"dtype={compute_out.dtype}. "
                        "This shape/dtype combination may not be supported."
                    )
                if tactic >= count:
                    raise ValueError(
                        f"Requested tactic {tactic} but only {count} algorithms "
                        f"available for M={a.shape[0]}, N={b.shape[1]}, K={a.shape[1]}, "
                        f"dtype={compute_out.dtype}."
                    )
                if tactic < 0:
                    tactic = 0
                module.mm_bf16_cublaslt_run_with_algo(
                    a,
                    b_t,
                    compute_out,
                    workspace_buffer,
                    cublas_handle,
                    algo_buf,
                    tactic,
                )
                if need_cast:
                    out.copy_(compute_out)
                return out

        return CublasltBf16GemmRunner()

    return SimpleNamespace(
        cublaslt_bf16_gemm_runner=cublaslt_bf16_gemm_runner,
    )


_BF16_GEMM_SM100_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (-2,),
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
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


def _tinygemm_bf16_gemm_runner():
    module = get_tinygemm2_module()

    class TinyGemmBf16GemmRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            # inputs layout: a, b, bias, pdl, out, workspace_buffer
            _, _, bias, pdl, _, _ = inputs
            return (bias is not None, bool(pdl))

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            return [0]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, bias, pdl, out, _ = inputs
            weight = b.transpose(-2, -1)
            if bias is None:
                module.tinygemm2_nobias_op(a, weight, out, pdl)
            else:
                module.tinygemm2_op(a, weight, bias, out, pdl)
            return out

    return TinyGemmBf16GemmRunner()


def bf16_gemm_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    pdl: bool,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    runner_names: List[str],
) -> None:
    use_sm_100f = is_sm100f_supported(a.device)

    tuner = AutoTuner.get()
    # Effective bucket mapper, accounting for any active
    # ``with autotune(tuning_buckets=..., round_up=...)`` overrides.
    # The cuDNN runner uses this for its override-shape ``cache_m`` so the
    # graph it builds at runtime is the SAME graph the autotuner profiled
    # tactics on.
    effective_m_bucket_mapper = tuner.get_effective_map_to_tuning_buckets(
        _BF16_GEMM_SM100_TUNING_CONFIG, spec_idx=0
    )
    # Capture the real tensors' stride layout so the cuDNN graph built
    # for autotune profiling (where ``a`` is a torch.rand-synthesized
    # contiguous tensor) matches the graph used at runtime.
    is_a_k_major = a.stride(-1) == 1
    is_b_k_major = b.stride(-2) == 1

    runners = []
    if "cudnn" in runner_names:
        runners.append(
            _cudnn_gemm_bf16_runner(
                effective_m_bucket_mapper,
                is_a_k_major=is_a_k_major,
                is_b_k_major=is_b_k_major,
            )
        )
    if "cublaslt" in runner_names:
        runners.append(get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner())
    if "cutlass" in runner_names:
        runners.append(get_gemm_sm100_module_cutlass_bf16().cutlass_bf16_gemm_runner())
    if "tgv" in runner_names:
        runners.append(
            get_tgv_gemm_sm10x_module(a.dtype, use_sm_100f).tgv_gemm_runner()
        )
    if "tinygemm" in runner_names:
        runners.append(_tinygemm_bf16_gemm_runner())
    assert runners, "No suitable runners found"

    inputs = [a, b, bias, pdl, out, workspace_buffer]
    runner, tactic = tuner.choose_one(
        "bf16_gemm",
        runners,
        _BF16_GEMM_SM100_TUNING_CONFIG,
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)


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
    """Get the SM100/110 FP4 GEMM module."""
    module = gen_gemm_sm100_module_cutlass_fp4().build_and_load()
    return _create_cutlass_fp4_gemm_module(
        module, "flashinfer::cutlass_fp4_gemm", "cutlass_fp4_gemm"
    )


@functools.cache
def get_gemm_sm103_module_cutlass_fp4():
    """Get the SM103 FP4 GEMM module."""
    module = gen_gemm_sm103_module_cutlass_fp4().build_and_load()
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
    sm_minor: int,
):
    if sm_major in [10, 11]:
        if sm_minor == 3:
            return get_gemm_sm103_module_cutlass_fp4()
        else:
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
                a, b, bias, pdl, out, *_ = inputs

                # swap gemm m and n by swapping b and a
                # tgv_gemm takes mat1 as weights and mat2 as input tensor
                # from [m,k]x[k,n]+[n,] to [n,k]x[k,m]+[n,]
                gemm_fn = module.tgv_gemm
                gemm_fn(b.t(), a.t(), bias, tactic, out, pdl)
                return out

        return TGVGemmRunner()

    # Register the module
    return SimpleNamespace(
        tgv_gemm_runner=tgv_gemm_runner,
    )


@flashinfer_api(trace=tgv_gemm_sm100_trace)
def tgv_gemm_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    pdl: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perform TGV GEMM on SM100 architecture with automatic dtype detection.

    Computes: A @ B + bias

    Args:
        a: First input tensor of shape (M, K) in row-major layout
        b: Second input tensor of shape (K, N) in column-major layout
        bias: Bias tensor of shape (N,)
        pdl: Whether to use PDL (persistent data loader), defaults to False
        out: Optional output tensor, shape (M, N), defaults to None.

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

    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]),
            device=a.device,
            dtype=a.dtype,
        )
    else:
        if out.shape != (a.shape[0], b.shape[1]):
            raise ValueError(
                f"Output shape mismatch. Expected {(a.shape[0], b.shape[1])}, got {out.shape}."
            )
        if out.device != a.device:
            raise ValueError(
                f"Output device mismatch. Expected {a.device}, got {out.device}."
            )
        if out.dtype != a.dtype:
            raise ValueError(
                f"Output dtype mismatch. Expected {a.dtype}, got {out.dtype}."
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
                get_hybrid_num_tokens_buckets,
                map_to_hybrid_bucket_uncapped,
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

    inputs = [a, b, bias, pdl, out]
    dtype_str = "bf16" if a.dtype == torch.bfloat16 else "fp16"
    runner, tactic = tuner.choose_one(
        f"{dtype_str}_tgv_gemm",
        runners,
        tuning_config,
        inputs,
    )

    return runner(inputs=inputs, tactic=tactic)


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

    @flashinfer_api
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

    @flashinfer_api(trace=segment_gemm_run_trace)
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
    BIAS_UID = 7
    O_UID = 8


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


def _check_cudnn_override_shape_availability():
    """Raise if the installed cuDNN backend does not support is_override_shape_enabled."""
    _check_cudnn_availability()
    backend_version = cudnn.backend_version()
    if backend_version < 92100:
        raise RuntimeError(
            f"cuDNN override-shape GEMM requires backend version >= 92100 (9.21.0), "
            f"found {backend_version}. "
            f"Please upgrade cuDNN: pip install --upgrade nvidia-cudnn-cu12 nvidia-cudnn-frontend"
        )
    try:
        version_str = cudnn.__version__
        major, minor = map(int, version_str.split(".")[:2])
        if (major, minor) < (1, 20):
            raise RuntimeError(
                f"cuDNN override-shape GEMM requires cudnn-frontend version >= 1.20, found {version_str}. "
                f"Please upgrade: pip install --upgrade nvidia-cudnn-frontend"
            )
    except (AttributeError, ValueError, IndexError) as e:
        raise RuntimeError(
            "Unable to determine cudnn-frontend version. "
            "Override-shape GEMM requires cudnn-frontend >= 1.20"
        ) from e


def is_cudnn_override_shape_available() -> bool:
    """Return True if the installed cuDNN backend supports is_override_shape_enabled."""
    if not CUDNN_AVAILABLE:
        return False
    try:
        if cudnn.backend_version() < 92100:
            return False
        version_str = cudnn.__version__
        major, minor = map(int, version_str.split(".")[:2])
        return (major, minor) >= (1, 20)
    except Exception:
        return False


def clear_cudnn_graph_cache() -> None:
    """Invalidate all process-local cuDNN GEMM graph caches **and** the
    AutoTuner profiling cache.

    .. note::
        **Internal / debug-only helper** -- not part of FlashInfer's
        public API.  Production callers should never need this:
        every ``build_cudnn_gemm_*`` helper is wrapped with
        ``functools.lru_cache(maxsize=1024)``, which auto-evicts cold
        shapes and keeps hot shapes resident, capping GPU memory
        growth on its own.

        This function exists for the few cases LRU cannot handle.  In
        every one of them, **the AutoTuner cache must also be cleared**
        because tactic indices (``execute_plan_at_index(..., N)``) are
        offsets into the cuDNN ``policy=ALL`` plan list of a *specific
        graph* -- if we throw the graph away and let it rebuild, the
        rebuilt plan list may enumerate engines differently and the
        stored ``N`` would silently point to a different kernel.  We
        therefore clear both stores atomically so plan indices and the
        graphs they were profiled against stay in lockstep:

            * **Hot-patching during development** -- after editing a
              ``build_*`` helper in-place, LRU will not evict the
              now-stale graph because the dev workload rarely reaches
              1024 distinct shapes; reach for this to invalidate
              without restarting Python.
            * **Deliberate cuDNN library version swap** in the same
              process -- the rebuilt graphs may expose different
              engines, so old tactic indices are no longer valid.
            * **Test fixtures** that want a clean slate per testcase.
            * **Manual GPU memory release** -- safe even if the build
              code is unchanged (the next call will simply re-tune
              from cold and pick equivalent tactics again).

        The function is intentionally *not* re-exported from
        :mod:`flashinfer.gemm` (i.e. not in its ``__all__``) and has
        no API stability guarantee.  Import it directly from this
        module if you really need it::

            from flashinfer.gemm.gemm_base import clear_cudnn_graph_cache

    This does **not** clear:

        * cuDNN handles (``_cudnn_handles``) -- those are cheap to
          re-set the stream on and expensive to recreate.
        * The autotuner's ``_find_nearest_profile`` LRU cache -- it is
          shape-keyed, not graph-keyed, and stays correct across
          rebuilds.
    """
    cached_builders = (
        build_cudnn_gemm_fp4_graph,
        build_cudnn_gemm_fp4_graph_override_shape,
        build_cudnn_gemm_mxfp8_graph_override_shape,
        build_cudnn_gemm_with_per_tensor_q_graph,
        build_cudnn_gemm_with_per_tensor_q_graph_override_shape,
        build_cudnn_gemm_bf16_graph,
        build_cudnn_gemm_bf16_graph_override_shape,
        create_cudnn_execution_plans_mxfp8_gemm,
    )
    for fn in cached_builders:
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()

    # Plan indices stored in AutoTuner.profiling_cache are offsets into
    # the cuDNN plan list of the graph that was profiled.  Once we
    # discard the graph LRU, those indices may no longer refer to the
    # same engine after the rebuilt graph re-enumerates plans, so we
    # clear the autotuner cache too.  See the docstring for details.
    AutoTuner.get().clear_cache()


# One cudnn handle per each GPU
_cudnn_handles: dict[int, int] = {}


def _get_cudnn_handle(device, stream: torch.cuda.Stream):
    """Create and return a cached cuDNN handle."""
    global _cudnn_handles
    device_id = device.index

    if _cudnn_handles.get(device_id) is None:
        _check_cudnn_availability()
        _cudnn_handles[device_id] = cudnn.create_handle()
        print("cudnn_handle created for device_id = {}\n".format(device_id))
    cudnn.set_stream(_cudnn_handles[device_id], stream.cuda_stream)

    return _cudnn_handles[device_id]


def _validate_fp8_output_dtype(dtype: torch.dtype):
    """Validate that the output dtype is either bf16 or fp16."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Unsupported output dtype: {dtype}. "
            f"Only torch.bfloat16 and torch.float16 are supported for FP8 GEMM operations."
        )


def _validate_bf16_output_dtype(dtype: torch.dtype):
    """Validate that the output dtype is bf16, fp16, or fp32."""
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(
            f"Unsupported output dtype: {dtype}. "
            f"Only torch.bfloat16, torch.float16, and torch.float32 are supported for BF16 GEMM operations."
        )


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_fp4_graph(
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
    policy=None,
):
    _check_cudnn_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
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

        graph.check_support()
        graph.build_plans(policy)

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
        graph.execute(
            variant_pack, workspace_buffer, handle=_get_cudnn_handle(a.device, stream)
        )
    else:
        graph.execute_plan_at_index(
            variant_pack,
            workspace_buffer,
            tactic,
            handle=_get_cudnn_handle(a.device, stream),
        )


# ---------------------------------------------------------------------------
# override_shape shared constant
# ---------------------------------------------------------------------------

# Sentinel value used as "cache M" when building override-shape graphs.
# Any M value will work in general.
# 8192 covers typical LLM inference shapes and set as default value.
_OVERRIDE_SHAPE_CACHE_M = 8192


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_fp4_graph_override_shape(
    batch,
    n,
    k,
    ab_type,
    o_type,
    block_size,
    device,
    alpha_is_not_none,
    use_nvfp4,
    cache_m: int = _OVERRIDE_SHAPE_CACHE_M,
    policy=None,
):
    """Build a cuDNN FP4 GEMM graph with override-shape support.

    The graph is compiled once using ``cache_m`` as the M dimension.  Block
    scale dimensions are derived from ``cache_m`` at compile time; at execution
    time the real M (and corresponding scale dims) are passed via
    ``override_shapes`` / ``override_strides``.

    Caching key contains ``(batch, n, k, ...)`` but **not** M.
    """

    _check_cudnn_override_shape_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

    # Build shapes / strides using cache_m
    block_scale_dim_m, block_scale_dim_n, block_scale_dim_k = (
        _calculate_block_scale_dims(cache_m, n, k, block_size)
    )

    a_shape = [batch, cache_m, k]
    a_stride = [cache_m * k, k, 1]

    b_shape = [batch, k, n]
    b_stride = [k * n, 1, k]

    a_descale_shape = [batch, block_scale_dim_m, block_scale_dim_k]
    a_descale_stride = [block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1]
    b_descale_shape = [batch, block_scale_dim_k, block_scale_dim_n]
    b_descale_stride = [block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k]

    stream = torch.cuda.current_stream(device)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(device, stream),
        is_override_shape_enabled=True,
    )

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
            dim=[1, 1, 1],
            stride=[1, 1, 1],
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

    if alpha_is_not_none and not _is_cublas_fp4_available_in_cudnn():
        graph.deselect_engines(["eng0"])

    graph.check_support()
    graph.build_plans(policy)

    return graph


# Internal helper called from mm_fp4; the user-facing mm_fp4 is already
# decorated, so decorating here would double-log the same invocation.
def execute_cudnn_gemm_fp4_graph_override_shape(
    graph,
    a,
    b,
    a_descale,
    b_descale,
    alpha,
    c_final,
    workspace_buffer,
    tactic: int = 0,
):
    """Execute FP4 GEMM cuDNN graph with dynamic-shape overrides."""

    real_a_shape, real_a_stride = _get_real_fp4_shape_from_packed_uint8(a)
    real_b_shape, real_b_stride = _get_real_fp4_shape_from_packed_uint8(b)
    batch = real_a_shape[0]
    expanded_a_descale_shape, expanded_a_descale_stride = (
        _expand_block_scale_tensor_shape(a_descale, batch)
    )
    expanded_b_descale_shape, expanded_b_descale_stride = (
        _expand_block_scale_tensor_shape(b_descale, batch)
    )

    c_shape, c_stride = _get_bf16_3d_shape_stride(c_final)

    if real_a_stride[-1] != 1 or real_b_stride[-2] != 1:
        raise ValueError(
            f"a and b must be k-major (contiguous along the K dimension), "
            f"got a stride={tuple(real_a_stride)}, b stride={tuple(real_b_stride)}"
        )

    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.BLOCK_DESCALE_A_UID.value: a_descale,
        UIDs.BLOCK_DESCALE_B_UID.value: b_descale,
        UIDs.O_UID.value: c_final,
    }

    if alpha is not None:
        variant_pack[UIDs.ALPHA_UID.value] = alpha.view(torch.float)

    override_uids = [
        UIDs.A_UID.value,
        UIDs.B_UID.value,
        UIDs.BLOCK_DESCALE_A_UID.value,
        UIDs.BLOCK_DESCALE_B_UID.value,
        UIDs.O_UID.value,
    ]
    override_shapes = [
        real_a_shape,
        real_b_shape,
        expanded_a_descale_shape,
        expanded_b_descale_shape,
        c_shape,
    ]
    override_strides = [
        real_a_stride,
        real_b_stride,
        expanded_a_descale_stride,
        expanded_b_descale_stride,
        c_stride,
    ]

    if workspace_buffer.numel() < graph.get_workspace_size():
        workspace_buffer = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    stream = torch.cuda.current_stream(a.device)

    graph.execute_plan_at_index(
        variant_pack,
        workspace_buffer,
        tactic,
        handle=_get_cudnn_handle(a.device, stream),
        override_uids=override_uids,
        override_shapes=override_shapes,
        override_strides=override_strides,
    )


def execute_cudnn_gemm_mxfp8_graph(
    graph,
    a,
    b,
    a_descale,
    b_descale,
    c_final,
    workspace_buffer,
    tactic: int = -1,
):
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.BLOCK_DESCALE_A_UID.value: a_descale,
        UIDs.BLOCK_DESCALE_B_UID.value: b_descale,
        UIDs.O_UID.value: c_final,
    }

    workspace_size = graph.get_workspace_size()

    if workspace_buffer.numel() < workspace_size:
        workspace_buffer = torch.empty(
            workspace_size, device=a.device, dtype=torch.uint8
        )

    stream = torch.cuda.current_stream(a.device)

    if tactic == -1:
        graph.execute(
            variant_pack, workspace_buffer, handle=_get_cudnn_handle(a.device, stream)
        )
    else:
        graph.execute_plan_at_index(
            variant_pack,
            workspace_buffer,
            tactic,
            handle=_get_cudnn_handle(a.device, stream),
        )


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_mxfp8_graph_override_shape(
    batch,
    n,
    k,
    a_type,
    b_type,
    o_type,
    block_size,
    device,
    cache_m: int = _OVERRIDE_SHAPE_CACHE_M,
    policy=None,
):
    """Build a cuDNN MXFP8 GEMM graph with override-shape support.

    Compiled once using ``cache_m`` as M; at execution time the actual M is
    provided through ``override_shapes`` / ``override_strides``.
    """
    _check_cudnn_override_shape_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    if a_type not in [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2]:
        raise ValueError(f"A type must be FP8_E4M3 or FP8_E5M2, got {a_type}")
    if b_type not in [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2]:
        raise ValueError(f"B type must be FP8_E4M3 or FP8_E5M2, got {b_type}")
    if o_type not in [cudnn.data_type.BFLOAT16, cudnn.data_type.HALF]:
        raise ValueError(f"Output type must be BF16 or FP16, got {o_type}")

    block_scale_dim_m, block_scale_dim_n, block_scale_dim_k = (
        _calculate_block_scale_dims(cache_m, n, k, block_size)
    )

    scale_type = cudnn.data_type.FP8_E8M0

    a_shape = [batch, cache_m, k]
    a_stride = [cache_m * k, k, 1]
    b_shape = [batch, k, n]
    b_stride = [k * n, 1, k]
    a_descale_shape = [batch, block_scale_dim_m, block_scale_dim_k]
    a_descale_stride = [block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1]
    b_descale_shape = [batch, block_scale_dim_k, block_scale_dim_n]
    b_descale_stride = [block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k]

    stream = torch.cuda.current_stream(device)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(device, stream),
        is_override_shape_enabled=True,
    )

    a_cudnn_tensor = graph.tensor(
        name="a", dim=a_shape, stride=a_stride, data_type=a_type
    )
    b_cudnn_tensor = graph.tensor(
        name="b", dim=b_shape, stride=b_stride, data_type=b_type
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
    c_tensor.set_output(True).set_data_type(o_type)

    a_cudnn_tensor.set_uid(UIDs.A_UID.value)
    b_cudnn_tensor.set_uid(UIDs.B_UID.value)
    block_descale_a_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_A_UID.value)
    block_descale_b_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_B_UID.value)
    c_tensor.set_uid(UIDs.O_UID.value)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])
    graph.check_support()
    graph.build_plans(policy)

    return graph


# Internal helper called from mm_mxfp8; the user-facing mm_mxfp8 is already
# decorated, so decorating here would double-log the same invocation.
def execute_cudnn_gemm_mxfp8_graph_override_shape(
    graph,
    a,
    b,
    a_descale,
    b_descale,
    c_final,
    workspace_buffer,
    tactic: int = 0,
):
    """Execute MXFP8 GEMM cuDNN graph with dynamic-shape overrides."""
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.BLOCK_DESCALE_A_UID.value: a_descale,
        UIDs.BLOCK_DESCALE_B_UID.value: b_descale,
        UIDs.O_UID.value: c_final,
    }

    override_uids = [
        UIDs.A_UID.value,
        UIDs.B_UID.value,
        UIDs.BLOCK_DESCALE_A_UID.value,
        UIDs.BLOCK_DESCALE_B_UID.value,
        UIDs.O_UID.value,
    ]
    override_shapes = [
        list(a.shape),
        list(b.shape),
        list(a_descale.shape),
        list(b_descale.shape),
        list(c_final.shape),
    ]
    override_strides = [
        list(a.stride()),
        list(b.stride()),
        list(a_descale.stride()),
        list(b_descale.stride()),
        list(c_final.stride()),
    ]

    if workspace_buffer.numel() < graph.get_workspace_size():
        workspace_buffer = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    stream = torch.cuda.current_stream(a.device)

    graph.execute_plan_at_index(
        variant_pack,
        workspace_buffer,
        tactic,
        handle=_get_cudnn_handle(a.device, stream),
        override_uids=override_uids,
        override_shapes=override_shapes,
        override_strides=override_strides,
    )


@functools.lru_cache(maxsize=2048)
def build_cudnn_gemm_with_per_tensor_q_graph(
    a_shape,
    a_stride,
    b_shape,
    b_stride,
    a_type,
    b_type,
    o_type,
    device,
    policy=None,
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
        policy: cuDNN build plan policy. None defaults to HEURISTICS_CHOICE.
                Use ALL to enumerate all execution plans for autotuning.

    Returns:
        cuDNN graph object
    """
    _check_cudnn_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
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
        graph.build_plans(policy)

        return graph


def execute_cudnn_gemm_with_per_tensor_q_graph(
    graph,
    a,
    b,
    a_scale,
    b_scale,
    c_final,
    workspace,
    tactic: int = -1,
):
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.A_SCALE_UID.value: a_scale,
        UIDs.B_SCALE_UID.value: b_scale,
        UIDs.O_UID.value: c_final,
    }

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    if workspace.numel() < graph.get_workspace_size():
        workspace = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
    else:
        graph.execute_plan_at_index(
            variant_pack, workspace, tactic, handle=cudnn_handle
        )


# ---------------------------------------------------------------------------
# FP8 per-tensor GEMM with override_shape (dynamic M dimension)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_with_per_tensor_q_graph_override_shape(
    batch, n, k, a_type, b_type, o_type, device, cache_m: int = _OVERRIDE_SHAPE_CACHE_M
):
    """Build an FP8 per-tensor-quantized GEMM cuDNN graph with override-shape.

    Compiled once with ``cache_m`` as M; at execution time the actual M is
    supplied through ``override_shapes`` / ``override_strides``.
    """
    _check_cudnn_override_shape_availability()

    a_shape = [batch, cache_m, k]
    a_stride = [cache_m * k, k, 1]
    b_shape = [batch, k, n]
    b_stride = [k * n, 1, k]

    stream = torch.cuda.current_stream(device)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(device, stream),
        is_override_shape_enabled=True,
    )

    a_cudnn_tensor = graph.tensor(
        name="a", dim=a_shape, stride=a_stride, data_type=a_type
    )
    b_cudnn_tensor = graph.tensor(
        name="b", dim=b_shape, stride=b_stride, data_type=b_type
    )
    a_scale_cudnn_tensor = graph.tensor(
        name="a_scale",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    b_scale_cudnn_tensor = graph.tensor(
        name="b_scale",
        dim=[1, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    c_cudnn_tensor = graph.matmul(
        name="matmul",
        A=a_cudnn_tensor,
        B=b_cudnn_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    c_cudnn_tensor.set_name("c").set_data_type(cudnn.data_type.FLOAT)
    c_after_scale_a = graph.mul(
        name="scale_mul_a",
        a=c_cudnn_tensor,
        b=a_scale_cudnn_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    c_after_scale_b = graph.mul(
        name="scale_mul_b",
        a=c_after_scale_a,
        b=b_scale_cudnn_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    c_after_scale_b.set_name("c_final").set_output(True).set_data_type(o_type)

    a_cudnn_tensor.set_uid(UIDs.A_UID.value)
    b_cudnn_tensor.set_uid(UIDs.B_UID.value)
    a_scale_cudnn_tensor.set_uid(UIDs.A_SCALE_UID.value)
    b_scale_cudnn_tensor.set_uid(UIDs.B_SCALE_UID.value)
    c_after_scale_b.set_uid(UIDs.O_UID.value)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    return graph


# Internal helper called from mm_fp8 per-tensor path; the user-facing mm_fp8
# is already decorated, so decorating here would double-log the same invocation.
def execute_cudnn_gemm_with_per_tensor_q_graph_override_shape(
    graph, a, b, a_scale, b_scale, c_final, workspace, tactic: int = 0
):
    """Execute FP8 per-tensor GEMM graph with dynamic-shape overrides."""
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b,
        UIDs.A_SCALE_UID.value: a_scale,
        UIDs.B_SCALE_UID.value: b_scale,
        UIDs.O_UID.value: c_final,
    }

    override_uids = [UIDs.A_UID.value, UIDs.B_UID.value, UIDs.O_UID.value]
    override_shapes = [list(a.shape), list(b.shape), list(c_final.shape)]
    override_strides = [list(a.stride()), list(b.stride()), list(c_final.stride())]

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    if workspace.numel() < graph.get_workspace_size():
        workspace = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    graph.execute_plan_at_index(
        variant_pack,
        workspace,
        tactic,
        handle=cudnn_handle,
        override_uids=override_uids,
        override_shapes=override_shapes,
        override_strides=override_strides,
    )


def _torch_data_type_to_cudnn_data_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif dtype == torch.float16:
        return cudnn.data_type.HALF
    elif dtype == torch.float32:
        return cudnn.data_type.FLOAT
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
    tactic: int = -1,
):
    _check_cudnn_availability()

    if tactic == -1:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
    else:
        policy = cudnn.build_plan_policy.ALL

    graph = build_cudnn_gemm_with_per_tensor_q_graph(
        a.shape,
        a.stride(),
        b.shape,
        b.stride(),
        _torch_data_type_to_cudnn_data_type(a.dtype),
        _torch_data_type_to_cudnn_data_type(b.dtype),
        _torch_data_type_to_cudnn_data_type(torch_out_dtype),
        a.device,
        policy=policy,
    )

    execute_cudnn_gemm_with_per_tensor_q_graph(
        graph,
        a,
        b,
        a_scale,
        b_scale,
        out,
        workspace,
        tactic=tactic,
    )
    return out


def _cudnn_gemm_fp8_runner():
    class CudnnFp8GemmRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            a, b, _, _, out, _ = inputs
            return (a.dtype, b.dtype, out.dtype)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a, b, _, _, out, _ = inputs
            graph = build_cudnn_gemm_with_per_tensor_q_graph(
                a.shape,
                a.stride(),
                b.shape,
                b.stride(),
                _torch_data_type_to_cudnn_data_type(a.dtype),
                _torch_data_type_to_cudnn_data_type(b.dtype),
                _torch_data_type_to_cudnn_data_type(out.dtype),
                a.device,
                policy=cudnn.build_plan_policy.ALL,
            )
            return list(range(graph.get_execution_plan_count()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, scale_a, scale_b, out, workspace_buffer = inputs
            _cudnn_gemm_fp8(
                workspace_buffer,
                a,
                b,
                scale_a,
                scale_b,
                out,
                out.dtype,
                tactic=tactic,
            )
            return out

    return CudnnFp8GemmRunner()


def _get_3d_shape_stride_from_vector(vector: torch.Tensor, dim: int = 0):
    """Expand 1d vector to 3d tensor for cuDNN"""
    if vector.dim() != 1:
        raise ValueError(f"Expected 1D vector, got {vector.dim()}D tensor")
    n = vector.shape[0]
    p = vector.stride(0)
    shape = [1, 1, 1]
    stride = [1, 1, 1]
    shape[dim] = n
    stride[dim] = p
    return (tuple(shape), tuple(stride))


def _get_bf16_3d_shape_stride(tensor: torch.Tensor):
    """Expand 2d tensor to 3d tensor for cuDNN"""
    if tensor.dim() != 2 and tensor.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {tensor.dim()}D tensor")
    shape = list(tensor.shape)
    stride = list(tensor.stride())

    if len(shape) == 2:
        shape.insert(0, 1)
        stride.insert(0, tensor.numel())

    return (tuple(shape), tuple(stride))


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_bf16_graph(
    a_shape,
    a_stride,
    b_shape,
    b_stride,
    o_type,
    device,
    bias_is_not_none,
    bias_shape,
    bias_stride,
    policy=None,
):
    _check_cudnn_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
        a_cudnn_tensor = graph.tensor(
            name="a", dim=a_shape, stride=a_stride, data_type=cudnn.data_type.BFLOAT16
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.BFLOAT16
        )
        c_cudnn_tensor = graph.matmul(
            name="matmul",
            A=a_cudnn_tensor,
            B=b_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        c_cudnn_tensor.set_data_type(cudnn.data_type.FLOAT)

        if bias_is_not_none:
            bias_cudnn_tensor = graph.tensor(
                name="bias",
                dim=bias_shape,
                stride=bias_stride,
                data_type=cudnn.data_type.BFLOAT16,
            )
            c_final_cudnn_tensor = graph.add(
                name="bias_add",
                a=c_cudnn_tensor,
                b=bias_cudnn_tensor,
            )
            bias_cudnn_tensor.set_uid(UIDs.BIAS_UID.value)
        else:
            c_final_cudnn_tensor = c_cudnn_tensor

        c_final_cudnn_tensor.set_name("c_final").set_output(True).set_data_type(o_type)

        a_cudnn_tensor.set_uid(UIDs.A_UID.value)
        b_cudnn_tensor.set_uid(UIDs.B_UID.value)
        c_final_cudnn_tensor.set_uid(UIDs.O_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(policy)

        return graph


def execute_cudnn_gemm_bf16_graph(
    graph, a, b, bias, c_final, workspace, tactic: int = -1
):
    if bias is not None:
        variant_pack = {
            UIDs.A_UID.value: a,
            UIDs.B_UID.value: b,
            UIDs.BIAS_UID.value: bias,
            UIDs.O_UID.value: c_final,
        }
    else:
        variant_pack = {
            UIDs.A_UID.value: a,
            UIDs.B_UID.value: b,
            UIDs.O_UID.value: c_final,
        }

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    if workspace.numel() < graph.get_workspace_size():
        workspace = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    if tactic == -1:
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
    else:
        graph.execute_plan_at_index(
            variant_pack, workspace, tactic, handle=cudnn_handle
        )


# ---------------------------------------------------------------------------
# BF16 GEMM with override_shape (dynamic M dimension)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1024)
def build_cudnn_gemm_bf16_graph_override_shape(
    batch,
    n,
    k,
    o_type,
    device,
    bias_is_not_none,
    cache_m: int = _OVERRIDE_SHAPE_CACHE_M,
    is_a_k_major: bool = True,
    is_b_k_major: bool = True,
    policy=None,
):
    """Build a cuDNN BF16 GEMM graph with override-shape support.

    Caching key is ``(batch, n, k, o_type, device, cache_m)`` — M is **not**
    part of the key.

    Args:
        is_a_k_major: If True, A has shape (batch, M, K) with row-major strides
            (K is the contiguous dimension).  If False, A has shape (batch, M, K)
            with column-major strides (M is the contiguous dimension).
        is_b_k_major: If True, B has shape (batch, K, N) where K is the leading
            dimension (stride along N is 1, i.e. N-contiguous within each K row).
            If False, B is row-major with K-contiguous layout (stride along K is 1).
    """
    _check_cudnn_override_shape_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    a_shape = (batch, cache_m, k)
    a_stride = (cache_m * k, k, 1) if is_a_k_major else (cache_m * k, 1, cache_m)
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k) if is_b_k_major else (k * n, n, 1)
    bias_shape = (1, 1, n)
    bias_stride = (n, n, 1)

    stream = torch.cuda.current_stream(device)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(device, stream),
        is_override_shape_enabled=True,
    )

    a_cudnn_tensor = graph.tensor(
        name="a",
        dim=list(a_shape),
        stride=list(a_stride),
        data_type=cudnn.data_type.BFLOAT16,
    )
    b_cudnn_tensor = graph.tensor(
        name="b",
        dim=list(b_shape),
        stride=list(b_stride),
        data_type=cudnn.data_type.BFLOAT16,
    )
    c_cudnn_tensor = graph.matmul(
        name="matmul",
        A=a_cudnn_tensor,
        B=b_cudnn_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    c_cudnn_tensor.set_data_type(cudnn.data_type.FLOAT)

    if bias_is_not_none:
        bias_cudnn_tensor = graph.tensor(
            name="bias",
            dim=bias_shape,
            stride=bias_stride,
            data_type=cudnn.data_type.BFLOAT16,
        )
        c_final_cudnn_tensor = graph.add(
            name="bias_add",
            a=c_cudnn_tensor,
            b=bias_cudnn_tensor,
        )
        bias_cudnn_tensor.set_uid(UIDs.BIAS_UID.value)
    else:
        c_final_cudnn_tensor = c_cudnn_tensor

    c_final_cudnn_tensor.set_name("c_final").set_output(True).set_data_type(o_type)

    a_cudnn_tensor.set_uid(UIDs.A_UID.value)
    b_cudnn_tensor.set_uid(UIDs.B_UID.value)
    c_final_cudnn_tensor.set_uid(UIDs.O_UID.value)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(policy)

    return graph


# Internal helper called from mm_bf16; the user-facing mm_bf16 is already
# decorated, so decorating here would double-log the same invocation.
def execute_cudnn_gemm_bf16_graph_override_shape(
    graph, a, b, bias, c_final, workspace, tactic: int = 0
):
    """Execute a BF16 GEMM cuDNN graph built with override-shape enabled.

    Passes the actual shapes/strides of *a*, *b*, and *c_final* as
    ``override_shapes`` / ``override_strides`` so a single compiled plan
    handles any M dimension without rebuilding.
    """
    a_shape, a_stride = _get_bf16_3d_shape_stride(a)
    b_shape, b_stride = _get_bf16_3d_shape_stride(b)
    c_shape, c_stride = _get_bf16_3d_shape_stride(c_final)

    if bias is not None:
        variant_pack = {
            UIDs.A_UID.value: a,
            UIDs.B_UID.value: b,
            UIDs.BIAS_UID.value: bias,
            UIDs.O_UID.value: c_final,
        }

        bias_shape, bias_stride = _get_3d_shape_stride_from_vector(bias, 2)

        override_uids = [
            UIDs.A_UID.value,
            UIDs.B_UID.value,
            UIDs.BIAS_UID.value,
            UIDs.O_UID.value,
        ]
        override_shapes = [
            list(a_shape),
            list(b_shape),
            list(bias_shape),
            list(c_shape),
        ]
        override_strides = [
            list(a_stride),
            list(b_stride),
            list(bias_stride),
            list(c_stride),
        ]
    else:
        variant_pack = {
            UIDs.A_UID.value: a,
            UIDs.B_UID.value: b,
            UIDs.O_UID.value: c_final,
        }

        override_uids = [UIDs.A_UID.value, UIDs.B_UID.value, UIDs.O_UID.value]
        override_shapes = [list(a_shape), list(b_shape), list(c_shape)]
        override_strides = [list(a_stride), list(b_stride), list(c_stride)]

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    if workspace.numel() < graph.get_workspace_size():
        workspace = torch.empty(
            graph.get_workspace_size(), device=a.device, dtype=torch.uint8
        )

    graph.execute_plan_at_index(
        variant_pack,
        workspace,
        tactic,
        handle=cudnn_handle,
        override_uids=override_uids,
        override_shapes=override_shapes,
        override_strides=override_strides,
    )


def _cudnn_gemm_bf16(
    workspace: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    tactic: int = -1,
):
    _check_cudnn_availability()

    # This allows the same graph to work for both mm (2D) and bmm (3D)
    a_shape, a_stride = _get_bf16_3d_shape_stride(a)
    b_shape, b_stride = _get_bf16_3d_shape_stride(b)

    if bias is not None:
        bias_shape, bias_stride = _get_3d_shape_stride_from_vector(bias, 2)
    else:
        bias_shape = (1, 1, 1)
        bias_stride = (1, 1, 1)

    if tactic == -1:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
    else:
        policy = cudnn.build_plan_policy.ALL

    graph = build_cudnn_gemm_bf16_graph(
        a_shape,
        a_stride,
        b_shape,
        b_stride,
        _torch_data_type_to_cudnn_data_type(out.dtype),
        a.device,
        bias is not None,
        bias_shape,
        bias_stride,
        policy=policy,
    )

    execute_cudnn_gemm_bf16_graph(graph, a, b, bias, out, workspace, tactic=tactic)
    return out


def _cudnn_gemm_bf16_runner(
    m_bucket_mapper=None,
    is_a_k_major: Optional[bool] = None,
    is_b_k_major: Optional[bool] = None,
):
    """Build a CudnnBf16GemmRunner.

    See :func:`_cudnn_gemm_fp4_runner` for the ``m_bucket_mapper``
    rationale; this is the BF16 GEMM analog using the same alignment
    scheme.  When ``m_bucket_mapper`` is ``None``, defaults to
    ``map_to_hybrid_bucket_uncapped`` -- the same mapper
    ``_BF16_GEMM_SM100_TUNING_CONFIG`` uses as
    ``map_to_tuning_buckets``.

    ``is_a_k_major`` / ``is_b_k_major`` describe the stride layout of
    the real ``a`` and ``b`` tensors at the call site.  These are baked
    into the cuDNN graph at build time (see
    :func:`build_cudnn_gemm_bf16_graph_override_shape`), so they are
    part of the graph cache key.  The autotuner's profile path
    synthesizes ``a`` via ``torch.rand`` which is always contiguous
    (k-major); if the real tensor is *not* k-major we would otherwise
    profile under one stride flag and execute under another, picking
    a tactic from the wrong graph.  Capturing the flags here at runner
    construction (using the real tensors from the caller) keeps the
    flag identical across profile and runtime paths.

    Callers (``bf16_gemm_sm100``) typically pass the effective mapper
    from ``AutoTuner.get_effective_map_to_tuning_buckets`` and the
    real ``a.stride()[-1] == 1`` / ``b.stride()[-2] == 1``.
    """

    class CudnnBf16GemmRunner(TunableRunner):
        def __init__(
            self,
            m_bucket_mapper,
            is_a_k_major: Optional[bool],
            is_b_k_major: Optional[bool],
        ):
            super().__init__()
            self._m_bucket_mapper = (
                m_bucket_mapper
                if m_bucket_mapper is not None
                else map_to_hybrid_bucket_uncapped
            )
            # Default to k-major (the convention torch.rand-synthesized
            # profile tensors use) when caller didn't specify.
            self._is_a_k_major = True if is_a_k_major is None else is_a_k_major
            self._is_b_k_major = True if is_b_k_major is None else is_b_k_major

        def _get_override_graph(self, a, b, bias, out):
            a_shape, _ = _get_bf16_3d_shape_stride(a)
            b_shape, _ = _get_bf16_3d_shape_stride(b)

            batch = a_shape[0]
            actual_m = a_shape[-2]
            k = a_shape[-1]
            n = b_shape[-1]
            o_type = _torch_data_type_to_cudnn_data_type(out.dtype)

            # See _cudnn_gemm_fp4_runner._get_override_graph for full
            # rationale: cache_m must match the AutoTuner cache key so
            # the runtime graph is the SAME graph the autotuner profiled
            # tactics on.  is_a_k_major / is_b_k_major are taken from
            # ``self`` (captured from the real input tensors at runner
            # construction) instead of from the local ``a``/``b`` here:
            # in autotuner.profile mode ``a`` is a torch.rand-synthesized
            # contiguous tensor regardless of the real layout, so reading
            # its stride would build a different graph than the runtime
            # call, breaking tactic-index alignment.
            cache_m = self._m_bucket_mapper(actual_m)

            graph = build_cudnn_gemm_bf16_graph_override_shape(
                batch=batch,
                n=n,
                k=k,
                o_type=o_type,
                device=a.device,
                bias_is_not_none=bias is not None,
                cache_m=cache_m,
                is_a_k_major=self._is_a_k_major,
                is_b_k_major=self._is_b_k_major,
                policy=cudnn.build_plan_policy.ALL,
            )
            return graph

        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            # inputs layout: a, b, bias, pdl, out, workspace_buffer
            # out.dtype distinguishes bfloat16 / float16 / float32 output graphs
            _, _, bias, _, out, _ = inputs
            return (out.dtype, bias is not None)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a, b, bias, _, out, _ = inputs

            if is_cudnn_override_shape_available():
                graph = self._get_override_graph(a, b, bias, out)
            else:
                a_shape, a_stride = _get_bf16_3d_shape_stride(a)
                b_shape, b_stride = _get_bf16_3d_shape_stride(b)

                if bias is not None:
                    bias_shape, bias_stride = _get_3d_shape_stride_from_vector(bias, 2)
                else:
                    bias_shape = (1, 1, 1)
                    bias_stride = (1, 1, 1)

                graph = build_cudnn_gemm_bf16_graph(
                    a_shape,
                    a_stride,
                    b_shape,
                    b_stride,
                    _torch_data_type_to_cudnn_data_type(out.dtype),
                    a.device,
                    bias is not None,
                    bias_shape,
                    bias_stride,
                    policy=cudnn.build_plan_policy.ALL,
                )

            return list(range(graph.get_execution_plan_count()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, bias, _, out, workspace_buffer = inputs

            if is_cudnn_override_shape_available():
                graph = self._get_override_graph(a, b, bias, out)

                execute_cudnn_gemm_bf16_graph_override_shape(
                    graph,
                    a,
                    b,
                    bias,
                    out,
                    workspace_buffer,
                    tactic=max(tactic, 0),
                )
            else:
                _cudnn_gemm_bf16(workspace_buffer, a, b, bias, out, tactic=tactic)

            return out

    return CudnnBf16GemmRunner(m_bucket_mapper, is_a_k_major, is_b_k_major)


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


@flashinfer_api(trace=mm_fp8_trace)
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


def _create_cutlass_mxfp8_gemm_module(module, op_name: str, tuner_name: str):
    """Helper function to create cutlass MXFP8 GEMM module."""

    def cutlass_mxfp8_gemm_runner():
        class CutlassMxfp8GemmRunner(TunableRunner):
            def get_valid_tactics(
                self,
                inputs: List[torch.Tensor],
                profile: OptimizationProfile,
            ) -> List[int]:
                return list(range(module.mxfp8_gemm_tactic_num()))

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
                    _,
                    out,
                    workspace_buffer,
                ) = inputs

                # CUTLASS expects b_descale in (N, K/32).
                # 2D input is (K/32, N) and must be transposed; 1D swizzled is pass-through.
                if b_descale.ndim == 2:
                    # Input is (K/32, N), transpose to (N, K/32) for CUTLASS
                    b_descale_processed = b_descale.T
                    if not b_descale_processed.is_contiguous():
                        b_descale_processed = b_descale_processed.contiguous()
                else:
                    # 1D swizzled format - pass as-is, just ensure contiguous
                    b_descale_processed = b_descale
                    if not b_descale_processed.is_contiguous():
                        b_descale_processed = b_descale_processed.contiguous()

                module.mxfp8_gemm(
                    a,
                    b.T,
                    a_descale,
                    b_descale_processed,
                    out,
                    workspace_buffer,
                    tactic,
                )
                return out

        return CutlassMxfp8GemmRunner()

    return SimpleNamespace(
        cutlass_mxfp8_gemm_runner=cutlass_mxfp8_gemm_runner,
    )


@functools.cache
def get_gemm_sm100_module_cutlass_mxfp8():
    """Get the SM100/103/110 MXFP8 GEMM module."""
    module = gen_gemm_sm100_module_cutlass_mxfp8().build_and_load()
    return _create_cutlass_mxfp8_gemm_module(
        module, "flashinfer::cutlass_mxfp8_gemm", "cutlass_mxfp8_gemm"
    )


@functools.cache
def _load_gemm_sm120_mxfp8_module():
    """Load the raw TVM-FFI SM120 MXFP8 module (cached)."""
    return gen_gemm_sm120_module_cutlass_mxfp8().build_and_load()


@functools.cache
def get_gemm_sm120_module_cutlass_mxfp8():
    """Get the SM120/121 MXFP8 GEMM module."""
    return _create_cutlass_mxfp8_gemm_module(
        _load_gemm_sm120_mxfp8_module(),
        "flashinfer::cutlass_mxfp8_gemm",
        "cutlass_mxfp8_gemm",
    )


def get_cutlass_mxfp8_gemm_module(
    sm_major: int,
):
    if sm_major in [10, 11]:
        return get_gemm_sm100_module_cutlass_mxfp8()
    elif sm_major in [12]:
        return get_gemm_sm120_module_cutlass_mxfp8()
    else:
        raise ValueError(f"Unsupported SM major version: {sm_major}")


def _check_mm_mxfp8_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_8x4_sf_layout: bool = True,
    backend: Literal["cutlass", "cute-dsl", "trtllm", "auto"] = "auto",  # unused
) -> bool:
    # Generic checks
    ## pre-check the input tensors and block scale tensors
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"mm_mxfp8 accepts 2d tensors, got {a.shape=} and {b.shape=}")

    # b is passed transposed (shape [k, n]), so verify K matches.
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"K dimension mismatch in mm_mxfp8. got {a.shape[1]=}, {b.shape[0]=}"
        )

    # The output may contain NaN/Inf if the dimensions are too small
    min_n = 128
    min_k = 128
    if b.shape[1] < min_n or a.shape[1] < min_k:
        raise ValueError(
            f"MXFP8 requires n >= {min_n} and k >= {min_k} for CUTLASS MXFP8. "
            f"got m={a.shape[0]}, n={b.shape[1]}, k={a.shape[1]}."
        )

    # Input dtype as returned by mxfp8_quantize_sm100
    if a.dtype != torch.float8_e4m3fn:
        raise ValueError(f"a must be a float8_e4m3fn tensor, got {a.dtype=}")

    if b.dtype != torch.float8_e4m3fn:
        raise ValueError(f"b must be a float8_e4m3fn tensor, got {b.dtype=}")

    # Scale dtype as returned by mxfp8_quantize_sm100
    if a_descale.dtype != torch.uint8:
        raise ValueError(f"a_descale must be a uint8 tensor, got {a_descale.dtype=}")

    if b_descale.dtype != torch.uint8:
        raise ValueError(f"b_descale must be a uint8 tensor, got {b_descale.dtype=}")

    # MXFP8 block size
    sf_vec_size = 32
    if a_descale.ndim == 2:
        sf_layout = SfLayout.layout_linear
    else:
        sf_layout = SfLayout.layout_8x4 if use_8x4_sf_layout else SfLayout.layout_128x4

    if a_descale.ndim == 1:
        expected_len = _mxfp8_swizzled_scale_len(a.shape[0], a.shape[1], sf_layout)
        if a_descale.shape[0] != expected_len:
            raise ValueError(
                "a_descale shape mismatch for swizzled layout. "
                f"Expected {(expected_len,)}, got {a_descale.shape}."
            )
    elif a_descale.ndim == 2:
        if a.shape[1] % sf_vec_size != 0:
            raise ValueError(
                "a_descale shape mismatch for non-swizzled layout. "
                f"a.shape[1] must be divisible by {sf_vec_size}, got {a.shape[1]}."
            )
        expected_shape = (a.shape[0], a.shape[1] // sf_vec_size)
        if a_descale.shape != expected_shape:
            raise ValueError(
                "a_descale shape mismatch for non-swizzled layout. "
                f"Expected {expected_shape}, got {a_descale.shape}."
            )
    else:
        raise ValueError(
            f"a_descale must be 1D (swizzled) or 2D (non-swizzled), got {a_descale.shape}."
        )

    if b_descale.ndim == 1:
        expected_len = _mxfp8_swizzled_scale_len(b.shape[1], b.shape[0], sf_layout)
        if b_descale.shape[0] != expected_len:
            raise ValueError(
                "b_descale shape mismatch for swizzled layout. "
                f"Expected {(expected_len,)}, got {b_descale.shape}."
            )
    elif b_descale.ndim == 2:
        if b.shape[0] % sf_vec_size != 0:
            raise ValueError(
                "b_descale shape mismatch for non-swizzled layout. "
                f"b.shape[0] must be divisible by {sf_vec_size}, got {b.shape[0]}."
            )
        expected_shape = (b.shape[0] // sf_vec_size, b.shape[1])
        if b_descale.shape != expected_shape:
            raise ValueError(
                "b_descale shape mismatch for non-swizzled layout. "
                f"Expected {expected_shape}, got {b_descale.shape}."
            )
    else:
        raise ValueError(
            f"b_descale must be 1D (swizzled) or 2D (non-swizzled), got {b_descale.shape}."
        )

    if out is not None:
        expected_shape = (a.shape[0], b.shape[1])
        if out.shape != expected_shape:
            raise ValueError(
                f"Output shape mismatch. Expected {expected_shape}, got {out.shape}."
            )
        if out.device != a.device:
            raise ValueError(
                f"Output device mismatch. Expected {a.device}, got {out.device}."
            )
        if out.dtype != out_dtype:
            raise ValueError(
                f"Output dtype mismatch. Expected {out_dtype}, got {out.dtype}."
            )

    _validate_mxfp8_output_dtype(out_dtype)
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cutlass_gemm_mxfp8_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_8x4_sf_layout: bool = True,
    backend: Literal["cutlass", "cute-dsl", "trtllm", "auto"] = "auto",
):
    if is_sm12x_supported(a.device):
        # SM120/121 CUTLASS MXFP8 only supports 1D swizzled scales (SfLayout.layout_128x4).
        if use_8x4_sf_layout:
            return False
        if a_descale.ndim != 1 or b_descale.ndim != 1:
            return False
        # K and N must be multiples of 32.
        if a.shape[1] % 32 != 0 or b.shape[1] % 32 != 0:
            return False
    return True


@supported_compute_capability([100, 103])
def _trtllm_gemm_mxfp8_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_8x4_sf_layout: bool = True,
    backend: Literal["trtllm", "auto"] = "auto",
):
    if out_dtype != torch.bfloat16:
        return False
    if a.ndim != 2 or b.ndim != 2:  # currently don't support BlockMajorK layout
        return False
    k, n = b.shape
    if k % 256 != 0:
        return False
    return True


@supported_compute_capability([100, 103])
def _cute_dsl_gemm_mxfp8_requirement(
    a: torch.Tensor,  # unused
    b: torch.Tensor,  # unused
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,  # unused
    out_dtype: torch.dtype = torch.bfloat16,  # unused
    use_8x4_sf_layout: bool = True,  # unused
    backend: Literal["cutlass", "cute-dsl", "auto"] = "auto",  # unused
):
    # CuTe DSL MXFP8 path currently expects swizzled 1D block scales
    # in F8_128x4 layout for both A and B.
    if a_descale.ndim != 1 or b_descale.ndim != 1:
        raise ValueError(
            "cute_dsl mm_mxfp8 requires swizzled 1D scale tensors for a_descale and b_descale."
        )
    _check_cute_dsl_availability()
    return True


# Shared helpers for CuTe DSL block-scaled GEMM runners (mxfp8 & mxfp4/nvfp4)
_SM100_MMA_TILER_MN_CANDIDATES = [
    (128, 64),
    (256, 64),
    (128, 128),
    (256, 128),
    (128, 192),
    (256, 192),
    (128, 256),
    (256, 256),
]

_SM100_CLUSTER_SHAPE_MN_CANDIDATES = [
    (1, 1),
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 1),
    (4, 2),
    (4, 4),
]

_SM100_DEFAULT_MMA_TILER_MN = (128, 128)
_SM100_DEFAULT_CLUSTER_SHAPE_MN = (1, 1)


def _select_default_sm120_mma_tiler(m, n, sm_count):
    """Select optimal SM120 tile shape based on problem size and SM count.

    Uses narrower tiles (64x64, 64x128, 128x64) when the default 128x128
    would leave SMs idle on small-M shapes.
    """
    coarse_tile = (128, 128)
    coarse_tiles = ((m + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    if m <= 128 and coarse_tiles < max(1, sm_count // 2):
        if n > 1536:
            return (64, 128)
        medium_tile = (128, 64)
        medium_tiles = ((m + medium_tile[0] - 1) // medium_tile[0]) * (
            (n + medium_tile[1] - 1) // medium_tile[1]
        )
        if medium_tiles < max(1, sm_count // 2):
            return (64, 64)
        return (128, 64)
    return (128, 128)


def _get_approximate_cta_nums(m, n, tile_mn, cluster_shape_mn):
    tile_m, tile_n = tile_mn
    cluster_m, cluster_n = cluster_shape_mn
    ctas_m = ((m + tile_m - 1) // tile_m + cluster_m - 1) // cluster_m * cluster_m
    ctas_n = ((n + tile_n - 1) // tile_n + cluster_n - 1) // cluster_n * cluster_n
    return ctas_m * ctas_n


def _get_sm100_block_scaled_tactics(
    m,
    n,
    real_k,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    c_cutlass_dtype,
    device,
):
    """Enumerate valid SM100 block-scaled GEMM tactics for autotuning.

    Returns list of (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)
    tuples.  Shared by both the mxfp8 and fp4 cute-dsl runners.
    """
    from .kernels.dense_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    batch_size = 1
    m_aligned = m % 8 == 0
    n_aligned = n % 8 == 0

    valid_tactics = []
    for mma_tiler_mn in _SM100_MMA_TILER_MN_CANDIDATES:
        for cluster_shape_mn in _SM100_CLUSTER_SHAPE_MN_CANDIDATES:
            for swap_ab in (False, True):
                if not swap_ab and not n_aligned:
                    continue
                if swap_ab and not m_aligned:
                    continue

                if swap_ab:
                    c_major = "m"
                    kernel_m, kernel_n = n, m
                else:
                    c_major = "n"
                    kernel_m, kernel_n = m, n

                if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
                    ab_dtype,
                    sf_dtype,
                    sf_vec_size,
                    c_cutlass_dtype,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    kernel_m,
                    kernel_n,
                    real_k,
                    batch_size,
                    "k",
                    "k",
                    c_major,
                ):
                    continue

                for use_prefetch in (False, True):
                    if use_prefetch:
                        cta_nums = _get_approximate_cta_nums(
                            kernel_m, kernel_n, mma_tiler_mn, cluster_shape_mn
                        )
                        sm_count = torch.cuda.get_device_properties(
                            device
                        ).multi_processor_count
                        cta_wave_ratio = cta_nums / sm_count
                        if not (0.5 < cta_wave_ratio < 1.0 or real_k >= 8192):
                            continue

                    valid_tactics.append(
                        (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)
                    )
    return valid_tactics


def _compile_block_scaled_gemm(
    cache,
    cache_key,
    make_gemm_kernel,
    ab_cutlass_dtype,
    sf_dtype,
    c_cutlass_dtype,
    ab_assumed_align,
    cluster_shape_mn,
    swap_ab,
    sf_m,
    sf_n,
    sf_k,
    batch_size,
):
    """Compile a block-scaled GEMM kernel via CuTe DSL and cache it.

    ``make_gemm_kernel`` is a zero-arg callable that returns a kernel instance
    (Sm100 or Sm103).  It is only invoked on a cache miss.

    TVM-FFI compilation pattern:
      - A, B, C, alpha: make_fake_compact_tensor -> torch tensors
        passed directly at runtime via TVM-FFI C-level dlpack
      - SF tensors: make_ptr (complex 6D BlockScaledBasicChunk
        layout can't be expressed as torch tensor) -> data_ptr() at runtime
      - Stream: make_fake_stream -> automatic env stream at runtime

    For FP4 runners, ``ab_cutlass_dtype`` is ``Uint8`` because FP4 data is
    stored as uint8 in torch (2 FP4 values per byte); the kernel wrapper
    recasts from Uint8 to Float4E2M1FN internally.
    """
    if cache_key in cache:
        return cache[cache_key]

    import cutlass
    import cutlass.cute as cute

    from cutlass.cute.runtime import make_ptr
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    gemm = make_gemm_kernel()

    sym_m = cute.sym_int()
    sym_k = cute.sym_int()
    sym_n = cute.sym_int()

    a_fake = cute.runtime.make_fake_compact_tensor(
        ab_cutlass_dtype,
        (sym_m, sym_k),
        stride_order=(1, 0),
        assumed_align=ab_assumed_align,
    )
    b_fake = cute.runtime.make_fake_compact_tensor(
        ab_cutlass_dtype,
        (sym_n, sym_k),
        stride_order=(1, 0),
        assumed_align=ab_assumed_align,
    )
    if swap_ab:
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_n, sym_m),
            stride_order=(0, 1),
            assumed_align=16,
        )
    else:
        c_fake = cute.runtime.make_fake_compact_tensor(
            c_cutlass_dtype,
            (sym_m, sym_n),
            stride_order=(1, 0),
            assumed_align=16,
        )

    a_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    b_sf_ptr = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, 16)
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )

    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_gemm = cute.compile(
        gemm.wrapper,
        a_fake,
        b_fake,
        c_fake,
        sf_m,
        sf_n,
        sf_k,
        batch_size,
        a_sf_ptr,
        b_sf_ptr,
        alpha_fake,
        max_active_clusters,
        stream_fake,
        swap_ab,
        options="--opt-level 2 --enable-tvm-ffi",
    )

    result = (compiled_gemm, max_active_clusters)
    cache[cache_key] = result
    return result


_CUTE_DSL_ALPHA_ONE_CACHE: dict = {}


def _prepare_alpha_for_launch(alpha_tensor, device):
    """Prepare alpha as a 1-dim float32 device tensor with shape [1].

    When *alpha_tensor* is ``None``, returns a cached ``tensor([1.0])``
    on *device* (allocated once, reused forever).
    """
    if alpha_tensor is None:
        cached = _CUTE_DSL_ALPHA_ONE_CACHE.get(device)
        if cached is None:
            cached = torch.tensor([1.0], dtype=torch.float32, device=device)
            _CUTE_DSL_ALPHA_ONE_CACHE[device] = cached
        return cached
    if alpha_tensor.dim() == 0:
        return alpha_tensor.unsqueeze(0)
    return alpha_tensor.reshape(1)


_CUTE_DSL_MM_MXFP8_KERNEL_CACHE: dict[tuple, tuple] = {}


def _check_cute_dsl_availability():
    try:
        from flashinfer.cute_dsl.utils import is_cute_dsl_available
    except ImportError as err:
        raise RuntimeError("CuTe DSL is not available.") from err

    if not is_cute_dsl_available():
        raise RuntimeError("CuTe DSL is not available.")


def _cute_dsl_gemm_mxfp8_runner(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
):
    import cutlass

    from .kernels.dense_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"cute_dsl mm_mxfp8 does not support output dtype {out_dtype}. "
            "Supported: torch.bfloat16, torch.float16."
        )

    cutlass_dtype_attr = _TORCH_TO_CUTLASS_DTYPE_ATTR.get(out_dtype)
    if cutlass_dtype_attr is None:
        raise ValueError(
            f"cute_dsl mm_mxfp8 does not support output dtype {out_dtype}. "
            "Supported: torch.bfloat16, torch.float16."
        )
    c_cutlass_dtype = getattr(cutlass, cutlass_dtype_attr)
    _ = sm_major, sm_minor

    class CuteDSLMxfp8GemmRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> list:
            (a, b, a_descale, b_descale, _, out, _) = inputs
            return _get_sm100_block_scaled_tactics(
                m=a.shape[0],
                n=b.shape[1],
                real_k=a.shape[1],
                ab_dtype=cutlass.Float8E4M3FN,
                sf_dtype=cutlass.Float8E8M0FNU,
                sf_vec_size=32,
                c_cutlass_dtype=c_cutlass_dtype,
                device=a.device,
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=None,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_descale, b_descale, _, out, _) = inputs
            m = a.shape[0]
            real_k = a.shape[1]
            n = b.shape[1]

            sf_vec_size = 32
            sf_dtype = cutlass.Float8E8M0FNU
            batch_size = 1

            if tactic is None or tactic == -1:
                tactic = (
                    _SM100_DEFAULT_MMA_TILER_MN,
                    _SM100_DEFAULT_CLUSTER_SHAPE_MN,
                    False,
                    False,
                )

            (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch) = tactic

            if swap_ab:
                kernel_m, kernel_n = n, m
                # Swap A/B: kernel expects both mA and mB with shape (*, K).
                # b is (k, n) col-major → b.T is (n, k) row-major.
                # a is already (m, k) row-major — no transpose needed.
                kernel_a, kernel_b = b.T, a
                kernel_a_sf, kernel_b_sf = b_descale, a_descale
            else:
                kernel_m, kernel_n = m, n
                kernel_a, kernel_b = a, b.T
                kernel_a_sf, kernel_b_sf = a_descale, b_descale

            sf_m = (kernel_m + 127) // 128
            sf_n = (kernel_n + 127) // 128
            sf_k = (real_k // sf_vec_size + 3) // 4

            cache_key = (
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                enable_pdl,
                out_dtype,
            )

            compiled_gemm, _ = _compile_block_scaled_gemm(
                _CUTE_DSL_MM_MXFP8_KERNEL_CACHE,
                cache_key,
                lambda: Sm100BlockScaledPersistentDenseGemmKernel(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    use_prefetch,
                    enable_pdl,
                ),
                ab_cutlass_dtype=cutlass.Float8E4M3FN,
                sf_dtype=sf_dtype,
                c_cutlass_dtype=c_cutlass_dtype,
                ab_assumed_align=16,
                cluster_shape_mn=cluster_shape_mn,
                swap_ab=swap_ab,
                sf_m=sf_m,
                sf_n=sf_n,
                sf_k=sf_k,
                batch_size=batch_size,
            )

            alpha_for_launch = _prepare_alpha_for_launch(None, a.device)

            launch_out = (
                out.as_strided(out.shape, (1, out.shape[0])) if swap_ab else out
            )
            compiled_gemm(
                kernel_a,
                kernel_b,
                launch_out,
                sf_m,
                sf_n,
                sf_k,
                kernel_a_sf.data_ptr(),
                kernel_b_sf.data_ptr(),
                alpha_for_launch,
            )
            return out

    return CuteDSLMxfp8GemmRunner()


def _heuristic_func_mm_mxfp8(
    suitable_backends: List[str],
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_8x4_sf_layout: bool = True,
    backend: Literal["cutlass", "cute-dsl", "trtllm", "auto"] = "auto",
) -> List[str]:
    # don't select trtllm since it requires weight shuffling
    if "cutlass" in suitable_backends:
        return ["cutlass"]
    return []


@backend_requirement(
    {
        "cutlass": _cutlass_gemm_mxfp8_requirement,
        "trtllm": _trtllm_gemm_mxfp8_requirement,
        "cute-dsl": _cute_dsl_gemm_mxfp8_requirement,
    },
    common_check=_check_mm_mxfp8_problem_size,
    heuristic_func=_heuristic_func_mm_mxfp8,  # result stored in mm_mxfp8.suitable_auto_backends
)
@flashinfer_api(trace=mm_mxfp8_trace)
def mm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_8x4_sf_layout: bool = False,
    backend: Literal["cutlass", "cute-dsl", "trtllm", "auto"] = "auto",
) -> torch.Tensor:
    r"""MM MXFP8 (block size 32)

    Parameters
    ----------
    a: torch.Tensor
        Input A tensor, shape (m, k), mxfp8 e4m3.

    b: torch.Tensor
        Input B tensor, shape (k, n), should be column major, mxfp8 e4m3.

    a_descale: torch.Tensor
        Block scale tensor for A. Can be:
        - 2D non-swizzled: shape (m, k // 32)
        - 1D swizzled: shape (M_padded * K_padded,)
          where M_padded = round_up(m, 8 if 8x4 layout else 128), K_padded = round_up(k // 32, 4)
        dtype: uint8.

    b_descale: torch.Tensor
        Block scale tensor for B. Can be:
        - 2D non-swizzled: shape (k // 32, n) - transposed format
        - 1D swizzled: shape (N_padded * K_padded,) where N_padded = round_up(n, 128), K_padded = round_up(k // 32, 4)
        dtype: uint8.
        Note: For 2D format, this is the transposed version (typically passed as scale.t()).
        For 1D swizzled format, it's flattened from (N_padded, K_padded) layout.

    out: Optional[torch.Tensor]
        Out tensor, shape (m, n), bf16 or fp16. If provided, can only be used with the CUTLASS backend. Defaults to ``None``.

    out_dtype: torch.dtype
        Output dtype, bf16 or fp16. Defaults to ``torch.bfloat16``.

    use_8x4_sf_layout: bool
        Whether the scale tensors for a are in 8x4 layout (vs 128x4).

    backend: Literal["cutlass", "cute-dsl", "trtllm", "auto"]
        The backend to use for the operation. Defaults to ``"auto"``.
        ``"auto"`` selects the CUTLASS backend.
        - The ``"cute-dsl"`` backend currently requires swizzled 1D scales
          (``mxfp8_quantize(..., is_sf_swizzled_layout=True)``).
        - The ``"trtllm"`` requires b to be quantized with 128x4 swizzle layout and shuffled.
          a can be quantized with either 128x4 or 8x4 layout (controlled by `use_8x4_sf_layout`).
        - On SM12x GPUs, the ``"cutlass"`` backend only supports
          1D swizzled scales (``SfLayout.layout_128x4``). Passing 2D linear scales will raise
          an error. Use ``mxfp8_quantize(..., sf_swizzle_layout=SfLayout.layout_128x4)``.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (m, n), bf16 or fp16.

    Examples
    --------
    >>> import torch
    >>> from flashinfer import mxfp8_quantize, mm_mxfp8
    >>> m, n, k = 512, 256, 128
    >>> # Create input tensors - note: weight is [n, k] for typical NN layers
    >>> a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    >>> weight = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    >>>
    >>> # Option 1: Use swizzled layout (recommended for accuracy)
    >>> # Quantize input [m, k] - scales are 1D swizzled for (M, K/32) layout
    >>> a_mx, a_sf = mxfp8_quantize(input=a, is_sf_swizzled_layout=True)
    >>> # Quantize weight [n, k] - scales are 1D swizzled for (N, K/32) layout
    >>> w_mx, w_sf = mxfp8_quantize(input=weight, is_sf_swizzled_layout=True)
    >>> # Pass weight.T as [k, n] and 1D swizzled scales directly
    >>> out = mm_mxfp8(a_mx, w_mx.t(), a_sf, w_sf, out_dtype=torch.bfloat16)
    >>> out.shape
    torch.Size([512, 256])
    >>>
    >>> # Option 2: Use non-swizzled layout (for compatibility)
    >>> a_mx, a_sf = mxfp8_quantize(input=a, is_sf_swizzled_layout=False)
    >>> w_mx, w_sf = mxfp8_quantize(input=weight, is_sf_swizzled_layout=False)
    >>> # For non-swizzled: reshape to 2D and transpose weight scale to (k//32, n)
    >>> a_sf_2d = a_sf.view(m, k // 32)
    >>> w_sf_2d = w_sf.view(n, k // 32).t()  # Transpose to (k // 32, n)
    >>> out = mm_mxfp8(a_mx, w_mx.t(), a_sf_2d, w_sf_2d, out_dtype=torch.bfloat16)
    >>> out.shape
    torch.Size([512, 256])
    """

    assert a.ndim == 2, f"mm_mxfp8: a must be 2D, got {a.ndim}D with shape {a.shape}"
    assert b.ndim == 2, f"mm_mxfp8: b must be 2D, got {b.ndim}D with shape {b.shape}"
    assert a.shape[1] == b.shape[0], (
        f"mm_mxfp8: K dimension mismatch: a.shape[1]={a.shape[1]}, b.shape[0]={b.shape[0]}"
    )

    assert a_descale.ndim in (1, 2), (
        f"mm_mxfp8: a_descale must be 1D (swizzled) or 2D (non-swizzled), "
        f"got {a_descale.ndim}D with shape {a_descale.shape}, dtype={a_descale.dtype}"
    )
    assert b_descale.ndim in (1, 2), (
        f"mm_mxfp8: b_descale must be 1D (swizzled) or 2D (non-swizzled), "
        f"got {b_descale.ndim}D with shape {b_descale.shape}, dtype={b_descale.dtype}"
    )

    # NOTE: do NOT reshape swizzled 1D scales to 2D; it breaks the F8_128x4 layout.

    # allocate the output tensor if not provided
    if out is None:
        out = torch.empty(
            (a.shape[0], b.shape[1]),
            device=a.device,
            dtype=out_dtype,
        )

    workspace_buffer = _get_cache_buf(
        "mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    if backend == "auto":
        backends = mm_mxfp8.suitable_auto_backends
    else:
        backends = [backend]

    major, minor = get_compute_capability(a.device)

    backend_to_runner_factory = {
        "cutlass": lambda: get_cutlass_mxfp8_gemm_module(
            major
        ).cutlass_mxfp8_gemm_runner(),
        "trtllm": lambda: get_trtllm_gemm_module().trtllm_mxfp8_gemm_runner(
            use_8x4_sf_layout
        ),
        "cute-dsl": lambda: _cute_dsl_gemm_mxfp8_runner(major, minor, True, out_dtype),
    }

    runners: List[TunableRunner] = [
        backend_to_runner_factory[cur_backend]() for cur_backend in backends
    ]

    tuner = AutoTuner.get()

    tuning_config = _MM_MXFP8_TUNING_CONFIG

    inputs = [
        a,
        b,
        a_descale,
        b_descale,
        out_dtype,
        out,
        workspace_buffer,
    ]

    runner, tactic = tuner.choose_one(
        custom_op="mxfp8_gemm",
        runners=runners,
        tuning_config=tuning_config,
        inputs=inputs,
    )

    runner(inputs=inputs, tactic=tactic)
    return out


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
    _check_cudnn_availability()

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

    if tactic == -1:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
    else:
        policy = cudnn.build_plan_policy.ALL

    # build the fp4 cudnn graph
    # Constructed graph is cached, via @functools.lru_cache decorator.
    graph = build_cudnn_gemm_fp4_graph(
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
        policy=policy,
    )

    # execute the fp4 cudnn graph
    execute_cudnn_gemm_fp4_graph(
        graph, a, b, a_descale, b_descale, alpha, out, workspace_buffer, tactic=tactic
    )

    return out


def _cudnn_gemm_fp4_runner(m_bucket_mapper=None):
    """Build a CudnnFp4GemmRunner.

    Args:
        m_bucket_mapper: A callable ``int -> int`` mapping the runtime M
            (i.e. ``actual_m``) to the build-time ``cache_m`` for the cuDNN
            override-shape graph.  When ``None``, defaults to
            ``map_to_hybrid_bucket_uncapped`` -- the same mapper the
            ``_MM_FP4_TUNING_CONFIG_*`` configs use as
            ``map_to_tuning_buckets``.  Callers (``mm_fp4``) typically
            override this with the *currently active* mapper from
            ``AutoTuner.get_effective_map_to_tuning_buckets`` so any
            ``with autotune(tuning_buckets=..., round_up=...)`` overrides
            propagate into the cuDNN graph.

    Why this matters:
        cuDNN's override-shape feature builds one graph keyed by
        ``cache_m`` and reuses it for any runtime M (the real shape is
        passed through ``override_shapes`` at execute time).  The
        AutoTuner stores per-bucket tactics keyed by
        ``map_to_tuning_buckets(M)``.  If the runner's ``cache_m``
        function and the autotuner's ``map_to_tuning_buckets`` disagree,
        a tactic profiled on graph ``cache_m=A`` is silently applied
        to graph ``cache_m=B`` at runtime -- the plan index has
        different meaning in the two graphs.  Sharing the mapper keeps
        both keyed by the exact same value.

        The mapper need NOT satisfy ``mapper(M) >= M`` -- cuDNN
        override-shape accepts arbitrary ``cache_m`` and ``actual_m``
        in any order.  We only require that profile-time and
        runtime-time produce the same ``cache_m`` for the same input.
    """

    class CudnnFp4GemmRunner(TunableRunner):
        def __init__(self, m_bucket_mapper):
            super().__init__()
            self._m_bucket_mapper = (
                m_bucket_mapper
                if m_bucket_mapper is not None
                else map_to_hybrid_bucket_uncapped
            )

        def _get_override_graph(self, a, b, alpha, out_dtype, block_size, use_nvfp4):
            real_a_shape, _ = _get_real_fp4_shape_from_packed_uint8(a)
            real_b_shape, _ = _get_real_fp4_shape_from_packed_uint8(b)

            batch = real_a_shape[0]
            actual_m = real_a_shape[1]
            k = real_a_shape[2]
            n = real_b_shape[2]

            # cache_m must match the AutoTuner cache key so the runtime
            # graph is the SAME graph the autotuner profiled tactics on.
            # ``self._m_bucket_mapper`` is set by the caller (``mm_fp4``)
            # to the *currently effective* ``map_to_tuning_buckets`` (with
            # any ``autotune(tuning_buckets=..., round_up=...)`` override
            # applied).  Sharing the mapper keeps cache_m and the tactic
            # cache key in lockstep -- otherwise a tactic profiled on
            # graph ``cache_m=A`` is silently applied to graph ``cache_m=B``
            # at runtime, which has a different plan-index meaning.
            cache_m = self._m_bucket_mapper(actual_m)

            graph = build_cudnn_gemm_fp4_graph_override_shape(
                batch=batch,
                n=n,
                k=k,
                ab_type=cudnn.data_type.FP4_E2M1,
                o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                block_size=block_size,
                device=a.device,
                alpha_is_not_none=alpha is not None,
                use_nvfp4=use_nvfp4,
                cache_m=cache_m,
                policy=cudnn.build_plan_policy.ALL,
            )
            return graph

        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            # inputs layout: a, b, a_descale, b_descale, alpha, out_dtype,
            #                out, block_size, use_nvfp4, workspace_buffer
            # All four values affect which cuDNN graph is built.
            _, _, _, _, alpha, out_dtype, out, block_size, use_nvfp4, _ = inputs
            return (out_dtype, block_size, use_nvfp4, alpha is not None)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
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

            # currently cudnn backend does not support alpha for dynamic-shape
            # remove this restriction once cudnn suppport it
            if is_cudnn_override_shape_available():
                graph = self._get_override_graph(
                    a, b, alpha, out_dtype, block_size, use_nvfp4
                )
            else:
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

                graph = build_cudnn_gemm_fp4_graph(
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
                    policy=cudnn.build_plan_policy.ALL,
                )

            return list(range(graph.get_execution_plan_count()))

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

            # currently cudnn backend does not support alpha for dynamic-shape
            # remove this restriction once cudnn suppport it
            if is_cudnn_override_shape_available():
                graph = self._get_override_graph(
                    a, b, alpha, out_dtype, block_size, use_nvfp4
                )

                execute_cudnn_gemm_fp4_graph_override_shape(
                    graph,
                    a,
                    b,
                    a_descale,
                    b_descale,
                    alpha,
                    out,
                    workspace_buffer,
                    tactic=max(tactic, 0),
                )
            else:
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

            return out

    return CudnnFp4GemmRunner(m_bucket_mapper)


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
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"
    ] = "auto",  # unused
    use_nvfp4: bool = True,
    enable_pdl: bool = True,  # unused
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
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"
    ] = "auto",  # unused
    use_nvfp4: bool = True,
    enable_pdl: bool = True,  # unused
):
    if use_8x4_sf_layout:
        raise ValueError("Only TRTLLM FP4 GEMM supports 8x4 scale factor layout.")
    if (
        not use_nvfp4
        and _match_sm_version(a.device, ["120", "121"])
        and cudnn.backend_version() < 91400
    ):
        raise LibraryError(CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR)

    _check_cudnn_fp4_availability()

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
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"
    ] = "auto",  # unused
    use_nvfp4: bool = True,
    enable_pdl: bool = True,  # unused
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
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"
    ] = "auto",  # unused
    use_nvfp4: bool = True,
    enable_pdl: bool = True,  # unused
):
    if use_8x4_sf_layout:
        raise ValueError("Only TRTLLM FP4 GEMM supports 8x4 scale factor layout.")
    if not use_nvfp4:
        raise ValueError("Only cudnn and auto FP4 GEMM supports mxfp4 quantization.")
    return True


@supported_compute_capability([100, 103])
def _cute_dsl_gemm_fp4_requirement(
    a: torch.Tensor,  # unused
    b: torch.Tensor,  # unused
    a_descale: torch.Tensor,  # unused
    b_descale: torch.Tensor,  # unused
    alpha: Optional[torch.Tensor] = None,  # unused
    out_dtype: torch.dtype = torch.bfloat16,  # unused
    out: Optional[torch.Tensor] = None,  # unused
    block_size: int = 16,  # unused
    use_8x4_sf_layout: bool = False,
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"
    ] = "auto",  # unused
    use_nvfp4: bool = True,
    enable_pdl: bool = True,  # unused
):
    # cute_dsl backend requires 128x4 scale factor layout.
    # The kernel internally uses CUTLASS BlockScaledBasicChunk which expects
    # M/N padded to 128, K padded to 4 -- matching FlashInfer's quantization
    # preparation for 128x4 layout.
    if use_8x4_sf_layout:
        raise ValueError("cute_dsl FP4 GEMM only supports 128x4 scale factor layout.")
    _check_cute_dsl_availability()
    return True


@supported_compute_capability([120, 121])
def _b12x_gemm_fp4_requirement(
    a: torch.Tensor,  # unused
    b: torch.Tensor,  # unused
    a_descale: torch.Tensor,  # unused
    b_descale: torch.Tensor,  # unused
    alpha: Optional[torch.Tensor] = None,  # unused
    out_dtype: torch.dtype = torch.bfloat16,  # unused
    out: Optional[torch.Tensor] = None,  # unused
    block_size: int = 16,  # unused
    use_8x4_sf_layout: bool = False,
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"
    ] = "auto",  # unused
    use_nvfp4: bool = True,
    enable_pdl: bool = True,  # unused
):
    # b12x backend requires CUDA 13+, 128x4 scale factor layout, and NVFP4 only.
    if get_cuda_version().major < 13:
        raise ValueError(
            "b12x FP4 GEMM requires CUDA 13 or later. "
            f"Current CUDA version: {get_cuda_version()}."
        )
    if use_8x4_sf_layout:
        raise ValueError("b12x FP4 GEMM only supports 128x4 scale factor layout.")
    if not use_nvfp4:
        raise ValueError("b12x FP4 GEMM only supports NVFP4 (sf_vec_size=16).")
    _check_cute_dsl_availability()
    return True


# Module-level kernel cache for CuTe DSL GEMM, shared across runner instances.
# Keyed by (sf_vec_size, mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch,
#            kernel_type, use_tma_store, enable_pdl, out_dtype).
_CUTE_DSL_MM_FP4_KERNEL_CACHE: dict[tuple, tuple] = {}


def _cute_dsl_gemm_fp4_runner(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
    use_nvfp4: bool,
):
    """Create a CuTe DSL FP4 GEMM runner for the cute_dsl backend.

    On SM100: uses the SM100 kernel only.
    On SM103: uses both SM100 kernel and the SM103-specific 3xFP4 kernel.
    The autotuner selects the best (kernel_type, tile, cluster, swap_ab, prefetch,
    use_tma_store) combination.
    """
    import cutlass

    from .kernels.dense_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    sm_version = sm_major * 10 + sm_minor

    # TODO(yunzheq): Re-enable SM103 kernel once cutlass-dsl package includes
    # SM103MmaMXF4Op and compatible PersistentTileSchedulerParams.
    # To re-enable, remove the `Sm103Kernel = None` line below.
    Sm103Kernel = None
    # if sm_version == 103:
    #     try:
    #         from .kernels.dense_blockscaled_gemm_sm103 import (
    #             Sm103BlockScaledPersistentDenseGemmKernel,
    #         )
    #
    #         Sm103Kernel = Sm103BlockScaledPersistentDenseGemmKernel
    #     except ImportError:
    #         pass

    cutlass_dtype_attr = _TORCH_TO_CUTLASS_DTYPE_ATTR.get(out_dtype)
    c_cutlass_dtype = (
        getattr(cutlass, cutlass_dtype_attr) if cutlass_dtype_attr is not None else None
    )
    if c_cutlass_dtype is None:
        raise ValueError(
            f"cute_dsl backend does not support output dtype {out_dtype}. "
            f"Supported: torch.bfloat16, torch.float16."
        )

    class CuteDSLFp4GemmRunner(TunableRunner):
        """TunableRunner for CuTe DSL block-scaled FP4 dense GEMM.

        Tactics are tuples:
            (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, kernel_type, use_tma_store)
        where:
            - kernel_type: "sm100" or "sm103"
            - use_tma_store: None for sm100, True/False for sm103
        """

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> list:
            (a, b, a_descale, b_descale, alpha, _, out, _, _, _) = inputs
            m = a.shape[0]
            k_packed = a.shape[1]
            n = b.shape[1]
            real_k = k_packed * 2  # FP4 packed as uint8

            sf_vec_size = 16 if use_nvfp4 else 32
            ab_dtype = cutlass.Float4E2M1FN
            sf_dtype = cutlass.Float8E4M3FN if use_nvfp4 else cutlass.Float8E8M0FNU

            # SM100 tactics (shared enumeration with mxfp8)
            sm100_base = _get_sm100_block_scaled_tactics(
                m,
                n,
                real_k,
                ab_dtype,
                sf_dtype,
                sf_vec_size,
                c_cutlass_dtype,
                a.device,
            )
            valid_tactics = [(*t, "sm100", None) for t in sm100_base]

            # --- SM103 tactics (only on SM103) ---
            if sm_version == 103 and Sm103Kernel is not None:
                batch_size = 1
                m_aligned = m % 8 == 0
                n_aligned = n % 8 == 0

                sm103_mma_tiler_candidates = [
                    (128, 128),
                    (256, 128),
                    (128, 256),
                    (256, 256),
                ]

                for mma_tiler_mn in sm103_mma_tiler_candidates:
                    for cluster_shape_mn in _SM100_CLUSTER_SHAPE_MN_CANDIDATES:
                        for swap_ab in (False, True):
                            if not swap_ab and not n_aligned:
                                continue
                            if swap_ab and not m_aligned:
                                continue

                            if swap_ab:
                                c_major = "m"
                                kernel_m, kernel_n = n, m
                            else:
                                c_major = "n"
                                kernel_m, kernel_n = m, n

                            for use_tma_store in (True, False):
                                if not Sm103Kernel.can_implement(
                                    ab_dtype,
                                    sf_dtype,
                                    sf_vec_size,
                                    c_cutlass_dtype,
                                    mma_tiler_mn,
                                    cluster_shape_mn,
                                    kernel_m,
                                    kernel_n,
                                    real_k,
                                    batch_size,
                                    "k",
                                    "k",
                                    c_major,
                                    use_tma_store,
                                ):
                                    continue

                                valid_tactics.append(
                                    (  # type: ignore[arg-type]
                                        mma_tiler_mn,
                                        cluster_shape_mn,
                                        swap_ab,
                                        False,
                                        "sm103",
                                        use_tma_store,
                                    )
                                )

            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=None,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_descale, b_descale, alpha_tensor, _, out, _, _, _) = inputs
            m = a.shape[0]
            k_packed = a.shape[1]
            n = b.shape[1]
            real_k = k_packed * 2

            sf_vec_size = 16 if use_nvfp4 else 32
            sf_dtype = cutlass.Float8E4M3FN if use_nvfp4 else cutlass.Float8E8M0FNU
            batch_size = 1

            if tactic is None or tactic == -1:
                # Use analytical heuristic to pick the best tactic based on
                # tile and wave quantization efficiency.
                tactic = _select_sm100_mm_fp4_cute_dsl_tactic(
                    m, n, real_k, get_device_sm_count(a.device)
                )

            (
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
            ) = tactic

            if swap_ab:
                kernel_m, kernel_n = n, m
                # Swap A/B: kernel expects both mA and mB with shape (*, K).
                # b is (k_packed, n) col-major → b.T is (n, k_packed) row-major.
                # a is already (m, k_packed) row-major — no transpose needed.
                kernel_a, kernel_b = b.T, a
                kernel_a_sf, kernel_b_sf = b_descale.T, a_descale
            else:
                kernel_m, kernel_n = m, n
                # b comes in as (k_packed, n), need (n, k_packed) for the kernel
                kernel_a, kernel_b = a, b.T
                kernel_a_sf, kernel_b_sf = a_descale, b_descale.T

            # Scale factor dimensions (128x4 padded)
            sf_m = (kernel_m + 127) // 128
            sf_n = (kernel_n + 127) // 128
            sf_k = (real_k // sf_vec_size + 3) // 4

            cache_key = (
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
                enable_pdl,
                out_dtype,
            )

            if kernel_type == "sm103" and Sm103Kernel is not None:
                make_kernel = lambda: Sm103Kernel(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    use_tma_store,
                    enable_pdl,
                )
            else:
                make_kernel = lambda: Sm100BlockScaledPersistentDenseGemmKernel(
                    sf_vec_size,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    use_prefetch,
                    enable_pdl,
                )

            compiled_gemm, _ = _compile_block_scaled_gemm(
                _CUTE_DSL_MM_FP4_KERNEL_CACHE,
                cache_key,
                make_kernel,
                ab_cutlass_dtype=cutlass.Uint8,
                sf_dtype=sf_dtype,
                c_cutlass_dtype=c_cutlass_dtype,
                ab_assumed_align=32,
                cluster_shape_mn=cluster_shape_mn,
                swap_ab=swap_ab,
                sf_m=sf_m,
                sf_n=sf_n,
                sf_k=sf_k,
                batch_size=batch_size,
            )

            alpha_for_launch = _prepare_alpha_for_launch(alpha_tensor, a.device)

            # swap_ab compiled kernel expects column-major mC with shape
            # (sym_n, sym_m) = (m, n).  Reinterpret out's storage as
            # column-major so the runtime shape+stride checks pass.
            # The kernel reconstructs C's layout from the raw pointer,
            # so the view's strides don't affect computation.
            launch_out = (
                out.as_strided(out.shape, (1, out.shape[0])) if swap_ab else out
            )
            compiled_gemm(
                kernel_a,
                kernel_b,
                launch_out,
                sf_m,
                sf_n,
                sf_k,
                kernel_a_sf.data_ptr(),
                kernel_b_sf.data_ptr(),
                alpha_for_launch,
            )
            return out

    return CuteDSLFp4GemmRunner()


# Module-level kernel cache for b12x GEMM (separate from CuTe DSL SM100 cache).
_B12X_MM_FP4_KERNEL_CACHE: dict[tuple, tuple] = {}


def _b12x_gemm_fp4_runner(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
    use_nvfp4: bool,
):
    """Create a b12x FP4 GEMM runner for SM120.

    Uses the SM120 warp-level MMA kernel with underfill tile selection.
    """
    import cutlass

    from .kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
    )

    cutlass_dtype_attr = _TORCH_TO_CUTLASS_DTYPE_ATTR.get(out_dtype)
    c_cutlass_dtype = (
        getattr(cutlass, cutlass_dtype_attr) if cutlass_dtype_attr is not None else None
    )
    if c_cutlass_dtype is None:
        raise ValueError(
            f"b12x backend does not support output dtype {out_dtype}. "
            f"Supported: torch.bfloat16, torch.float16."
        )

    class B12xFp4GemmRunner(TunableRunner):
        """TunableRunner for b12x block-scaled FP4 dense GEMM on SM120.

        Tactics are tuples:
            (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, kernel_type, use_tma_store)
        where kernel_type is always "sm120" and use_tma_store is always None.
        """

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> list:
            (a, b, a_descale, b_descale, alpha, _, out, _, _, _) = inputs
            m = a.shape[0]
            k_packed = a.shape[1]
            n = b.shape[1]
            real_k = k_packed * 2

            sf_vec_size = 16
            ab_dtype = cutlass.Float4E2M1FN
            sf_dtype = cutlass.Float8E4M3FN
            batch_size = 1

            valid_tactics = []
            sm120_mma_tiler_candidates = [
                (64, 64),
                (64, 128),
                (128, 64),
                (128, 128),
            ]
            swap_ab = False
            for mma_tiler_mn in sm120_mma_tiler_candidates:
                if not Sm120B12xBlockScaledDenseGemmKernel.can_implement(
                    ab_dtype,
                    sf_dtype,
                    sf_vec_size,
                    c_cutlass_dtype,
                    mma_tiler_mn,
                    (1, 1),
                    m,
                    n,
                    real_k,
                    batch_size,
                    "k",
                    "k",
                    "n",
                ):
                    continue
                for use_prefetch in (False, True):
                    valid_tactics.append(
                        (mma_tiler_mn, (1, 1), swap_ab, use_prefetch, "sm120", None)
                    )
            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=None,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_descale, b_descale, alpha_tensor, _, out, _, _, _) = inputs
            m = a.shape[0]
            k_packed = a.shape[1]
            n = b.shape[1]
            real_k = k_packed * 2

            sf_vec_size = 16
            sf_dtype = cutlass.Float8E4M3FN
            batch_size = 1

            if tactic is None or tactic == -1:
                tactic = (
                    _select_default_sm120_mma_tiler(
                        m, n, get_device_sm_count(a.device)
                    ),
                    (1, 1),
                    False,
                    False,
                    "sm120",
                    None,
                )

            (
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
            ) = tactic

            # b12x SM120 kernel does not support swap_ab
            kernel_m, kernel_n = m, n
            kernel_a, kernel_b = a, b.T
            kernel_a_sf, kernel_b_sf = a_descale, b_descale.T

            sf_m = (kernel_m + 127) // 128
            sf_n = (kernel_n + 127) // 128
            sf_k = (real_k // sf_vec_size + 3) // 4

            cache_key = (
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
                enable_pdl,
                out_dtype,
            )

            make_kernel = lambda: Sm120B12xBlockScaledDenseGemmKernel(
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                use_prefetch,
                enable_pdl,
            )

            compiled_gemm, _ = _compile_block_scaled_gemm(
                _B12X_MM_FP4_KERNEL_CACHE,
                cache_key,
                make_kernel,
                ab_cutlass_dtype=cutlass.Uint8,
                sf_dtype=sf_dtype,
                c_cutlass_dtype=c_cutlass_dtype,
                ab_assumed_align=32,
                cluster_shape_mn=cluster_shape_mn,
                swap_ab=swap_ab,
                sf_m=sf_m,
                sf_n=sf_n,
                sf_k=sf_k,
                batch_size=batch_size,
            )

            alpha_for_launch = _prepare_alpha_for_launch(alpha_tensor, a.device)

            compiled_gemm(
                kernel_a,
                kernel_b,
                out,
                sf_m,
                sf_n,
                sf_k,
                kernel_a_sf.data_ptr(),
                kernel_b_sf.data_ptr(),
                alpha_for_launch,
            )
            return out

    return B12xFp4GemmRunner()


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
    backend: Literal[
        "cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"
    ] = "cudnn",
    use_nvfp4: bool = True,
    enable_pdl: bool = True,  # unused
):
    r"""
    Heuristic function for mm_fp4 backend selection. Routes to either cudnn or cutlass.
    Note: trtllm is not considered in the backend selection because it requires a specific
    input quantization (swizzling/shuffling) that differs from the preparation used
    for cudnn and cutlass backends.

    Logic for which comes first:
    - If cuda version is 12 - use cutlass.
    - If cuda version is 13 and cudnn version is less than 9.15 - use cutlass.
    - If cuda version is 13 and cudnn version is 9.15 or greater:
      - On SM103 (B300) - use cutlass (faster based on benchmarks).
      - On SM100 (B200) - use cudnn (faster based on benchmarks).

    """
    cuda_major = get_cuda_version().major
    # Get compute capability to distinguish between SM100 (10.0) and SM103 (10.3)
    major, minor = get_compute_capability(a.device)
    is_sm103 = major == 10 and minor == 3
    is_sm120 = major == 12 and minor == 0

    # SM120 + CUDA 13: prefer b12x (warp-level MMA, underfill tile selection)
    if is_sm120 and use_nvfp4 and cuda_major >= 13:
        return [c for c in ("b12x", "cutlass", "cudnn") if c in suitable_backends]

    # If cuda version is 13 or greater and cudnn version is 9.15 or greater:
    # On SM103 (B300), cutlass is more performant than cudnn.
    # On SM100 (B200), cudnn is more performant than cutlass.
    if CUDNN_AVAILABLE and cuda_major >= 13 and cudnn.backend_version() >= 91500:
        if is_sm103:
            candidate_backends = ("cutlass", "cudnn")
        else:
            candidate_backends = ("cudnn", "cutlass")
    # Otherwise, prioritize cutlass
    else:
        candidate_backends = ("cutlass", "cudnn")

    # Filter and return only supported backends
    return [c for c in candidate_backends if c in suitable_backends]


def _pad_up(x, y):
    return ((x + y - 1) // y) * y


def _mxfp8_swizzled_scale_len(m: int, k: int, swizzle_layout: SfLayout) -> int:
    """Return the 1D swizzled scale length for MXFP8."""
    if swizzle_layout == SfLayout.layout_128x4:
        m_padded = _pad_up(m, 128)
        num_k_tiles = _pad_up(k, 128) // 128
        return m_padded * num_k_tiles * 4
    elif swizzle_layout == SfLayout.layout_8x4:
        m_padded = _pad_up(m, 8)
        num_k_tiles = _pad_up(k, 128) // 128
        return m_padded * num_k_tiles * 4
    elif swizzle_layout == SfLayout.layout_linear:
        return m * k
    else:
        raise ValueError(f"Unsupported swizzle layout: {swizzle_layout}")


_MM_FP4_TUNING_CONFIG_8x4 = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
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
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
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


_MM_MXFP8_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            2,  # a_descale_tensor_index
            0,
            lambda shapes: (
                _mxfp8_swizzled_scale_len(
                    shapes[0][0], shapes[0][1], SfLayout.layout_128x4
                )
                if len(shapes[2]) == 1
                else shapes[0][0]
            ),
        ),
        ConstraintSpec(
            5,  # out_tensor_index
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
        "cute-dsl": _cute_dsl_gemm_fp4_requirement,
        "b12x": _b12x_gemm_fp4_requirement,
    },
    common_check=_check_mm_fp4_problem_size,
    heuristic_func=_heuristic_func_mm_fp4,  # result stored in mm_fp4.suitable_auto_backends
)
@flashinfer_api(trace=mm_fp4_trace)
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
    backend: Literal["cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"] = "auto",
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
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

    backend: Literal["cudnn", "trtllm", "cutlass", "cute-dsl", "b12x", "auto"]
        Backend to use, defaults to ``"auto"``. On SM120, ``"auto"`` prefers
        ``"b12x"`` (NVFP4 only), then ``"cutlass"``, then ``"cudnn"``. On other
        architectures, ``"auto"`` selects between ``"cudnn"`` and ``"cutlass"``
        based on the current CUDA and cuDNN versions. The ``"trtllm"`` and
        ``"cute-dsl"`` backends are never auto-selected because they require
        different weight preparation.

    use_nvfp4: bool
        Whether to use nvfp4 quantization or mxfp4 quantization, defaults to ``True``.
        See the ``block_size`` parameter for related constraints.

    enable_pdl: bool
        Whether to enable Programmatic Dependent Launch (PDL) for the ``cute_dsl``
        backend, defaults to ``True``. PDL allows overlapping the tail of one kernel
        with the start of the next for reduced launch latency. This parameter is
        only used by the ``cute_dsl`` backend and is ignored by other backends.

    Notes
    -----
    When cudnn/cutlass backend is used, both a and b should quantized with nvfp4_quantize using the 128x4 scale factor layout and do_shuffle=False.
    When trtllm backend is used, b must be quantized with 128x4 layout and `do_shuffle=True`. a can be quantized with either 128x4 or 8x4 layout (controlled by `use_8x4_sf_layout`) and `do_shuffle=False`.
    When cute_dsl backend is used, both a and b should be quantized with 128x4 scale factor layout:
    nvfp4_quantize(..., do_shuffle=False) for NVFP4, or mxfp4_quantize(...) for MXFP4.

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
    major, minor = get_compute_capability(a.device)

    tuner = AutoTuner.get()
    tuning_config = (
        _MM_FP4_TUNING_CONFIG_8x4 if use_8x4_sf_layout else _MM_FP4_TUNING_CONFIG_128x4
    )

    # Effective bucket mapper, accounting for any active
    # ``with autotune(tuning_buckets=..., round_up=...)`` overrides.
    # The cuDNN runner uses this for its override-shape ``cache_m`` so the
    # graph it builds at runtime is the SAME graph the autotuner profiled
    # tactics on.  Without this, custom-bucket overrides cause silent
    # cache_m / autotune-key drift (wrong plan, possible NaN).
    effective_m_bucket_mapper = tuner.get_effective_map_to_tuning_buckets(
        tuning_config, spec_idx=0
    )

    backend_to_runner_factory = {
        "cudnn": lambda: _cudnn_gemm_fp4_runner(effective_m_bucket_mapper),
        "trtllm": lambda: get_trtllm_gemm_module().trtllm_fp4_gemm_runner(
            use_8x4_sf_layout
        ),
        "cutlass": lambda: get_cutlass_fp4_gemm_module(
            major, minor
        ).cutlass_fp4_gemm_runner(),
        "cute-dsl": lambda: _cute_dsl_gemm_fp4_runner(
            major, minor, enable_pdl, out_dtype, use_nvfp4
        ),
        "b12x": lambda: _b12x_gemm_fp4_runner(
            major, minor, enable_pdl, out_dtype, use_nvfp4
        ),
    }
    runners = [backend_to_runner_factory[cur_backend]() for cur_backend in backends]

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


@supported_compute_capability([89, 90, 100, 103, 110, 120, 121])
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


@supported_compute_capability([89, 90, 100, 103, 110, 120, 121])
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
            # supports all K values through padding
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
@flashinfer_api(trace=bmm_fp8_trace)
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
        if is_sm12x_supported(a.device):
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
            DtypeTrtllmGen.E4m3,
            DtypeTrtllmGen.Bfloat16,
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
def get_trtllm_gemm_module():
    mod = gen_trtllm_gen_gemm_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())

    class TrtllmGemmRunner(TunableRunner):
        def __init__(
            self,
            input_dtype: DtypeTrtllmGen,
            output_dtype: DtypeTrtllmGen,
            use_8x4_sf_layout: bool = True,
        ):
            self._gemm_runner = op.trtllm_gemm
            self._use_8x4_sf_layout = use_8x4_sf_layout
            self._input_dtype = input_dtype
            self._output_dtype = output_dtype

        def unpack_inputs(
            self,
            inputs: List[torch.Tensor],
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
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
            return a, b, a_descale, b_descale, alpha, out, workspace_buffer

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a_tensor_index = 0
            b_tensor_index = 1

            a = profile.get_opt_shapes()[a_tensor_index]
            b = profile.get_opt_shapes()[b_tensor_index]
            m = a[0]
            n = b[1]
            assert a[1] == b[0], (
                f"The k dimension is inconsistent between A ({a}) and B ({b})"
            )
            if self._input_dtype == DtypeTrtllmGen.E2m1:
                k = a[1] * 2
            else:
                k = a[1]
            (
                a,
                b,
                a_descale,
                b_descale,
                alpha,
                out,
                workspace_buffer,
            ) = self.unpack_inputs(inputs)
            return list(
                op.trtllm_gemm_tactics(
                    m,
                    n,
                    k,
                    self._input_dtype,
                    self._output_dtype,
                    self._use_8x4_sf_layout,
                )
            )

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            a, b, a_descale, b_descale, alpha, out, workspace_buffer = (
                self.unpack_inputs(inputs)
            )
            self._gemm_runner(
                self._input_dtype,
                self._output_dtype,
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

    def trtllm_gemm_runner(
        input_dtype: DtypeTrtllmGen,
        output_dtype: DtypeTrtllmGen,
        use_8x4_sf_layout: bool = True,
    ):
        return TrtllmGemmRunner(input_dtype, output_dtype, use_8x4_sf_layout)

    def trtllm_fp4_gemm_runner(
        use_8x4_sf_layout: bool = True,
    ):
        return TrtllmGemmRunner(
            DtypeTrtllmGen.E2m1, DtypeTrtllmGen.Bfloat16, use_8x4_sf_layout
        )

    def trtllm_mxfp8_gemm_runner(
        use_8x4_sf_layout: bool = True,
    ):
        # monkey patch to align with cutlass runner's input format
        class TrtllmMxFp8GemmRunner(TrtllmGemmRunner):
            def unpack_inputs(
                self,
                inputs: List[torch.Tensor],
            ) -> Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                (
                    a,
                    b,
                    a_descale,
                    b_descale,
                    out_dtype,
                    out,
                    workspace_buffer,
                ) = inputs
                assert out_dtype == torch.bfloat16
                return a, b, a_descale, b_descale, None, out, workspace_buffer

        return TrtllmMxFp8GemmRunner(
            DtypeTrtllmGen.MxE4m3, DtypeTrtllmGen.Bfloat16, use_8x4_sf_layout
        )

    # Register the module
    return SimpleNamespace(
        trtllm_gemm_runner=trtllm_gemm_runner,
        trtllm_fp4_gemm_runner=trtllm_fp4_gemm_runner,
        trtllm_mxfp8_gemm_runner=trtllm_mxfp8_gemm_runner,
        trtllm_gemm=op.trtllm_gemm,
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

    if is_sm12x_supported(a.device):
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

    if is_sm12x_supported(a.device):
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
    if swap_ab not in [True, False]:
        raise ValueError(f"swap_ab must be a boolean value, but got {swap_ab}")

    if is_sm12x_supported(a.device):
        if mma_sm not in [1]:
            raise ValueError(f"mma_sm must be 1, but got {mma_sm}")
        if tile_m not in [128]:
            raise ValueError(f"tile_m must be 128, but got {tile_m}")
        if tile_n not in [128]:
            raise ValueError(f"tile_n must be 128, but got {tile_n}")
        if tile_k not in [128]:
            raise ValueError(f"tile_k must be 128, but got {tile_k}")
    else:
        if mma_sm not in [1, 2]:
            raise ValueError(f"mma_sm must be either 1 or 2, but got {mma_sm}")
        if tile_m not in [128]:
            raise ValueError(f"tile_m must be 128, but got {tile_m}")
        if tile_n not in [64, 128, 192, 256]:
            raise ValueError(
                f"tile_n must be one of [64, 128, 192, 256], but got {tile_n}"
            )
        if tile_k not in [128, 256]:
            raise ValueError(f"tile_k must be either 128 or 256, but got {tile_k}")

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
    Blackwell, Blackwell Geforce, and DGX Spark architectures.

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
        How many SMs to use for the MMA operation, must be 1 or 2. 2 is not supported on SM120/121.
        2 is faster when number of rows (M) per group is large (>= 256).

    tile_m: int
        The tile size for the M dimension, must be 128.

    tile_n: int
        The tile size for the N dimension, must be 64, 128, 192, or 256. Only 128 is supported on SM120/121.

    tile_k: int
        The tile size for the K dimension, must be 128 or 256. Only 128 is supported on SM120/121.

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

    if is_sm12x_supported(a.device):
        # SM120/121 doesn't use mma_sm parameter or swap_ab
        get_gemm_sm120_module().group_gemm_mxfp4_nt_groupwise(
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
            tile_m,
            tile_n,
            tile_k,
        )
    elif is_sm100a_supported(a.device):
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
    else:
        raise ValueError(f"Unsupported device for MXFP4 group GEMM: {a.device}")
    return out


# NOTE(Zihao): keep the old name for backward compatibility
group_gemm_mxfp4_nt_groupwise = group_gemm_mxfp8_mxfp4_nt_groupwise


# NOTE: Just 120/121 support has been added, but it is trivial to generalize
@supported_compute_capability([120, 121])
def _check_group_gemm_nvfp4_nt_groupwise_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
):
    if a.dtype not in [torch.uint8]:
        raise ValueError(f"a must be a uint8 tensor, but got {a.dtype}")
    if b.dtype != torch.uint8:
        raise ValueError(f"b must be a uint8 tensor, but got {b.dtype}")
    if a_scale.dtype != torch.uint8:
        raise ValueError(f"a_scale must be a uint8 tensor, but got {a_scale.dtype}")
    if b_scale.dtype != torch.uint8:
        raise ValueError(f"b_scale must be a uint8 tensor, but got {b_scale.dtype}")
    if m_indptr.dtype != torch.int32:
        raise ValueError(f"m_indptr must be a int32 tensor, but got {m_indptr.dtype}")
    if alpha is not None and alpha.dtype != torch.float32:
        raise ValueError(
            f"alpha must be a float32 tensor or None, but got {alpha.dtype}"
        )
    if tile_m not in [128]:
        raise ValueError(f"tile_m must be 128, but got {tile_m}")
    if tile_n not in [128]:
        raise ValueError(f"tile_n must be one of 128, but got {tile_n}")
    if tile_k not in [128, 256]:
        raise ValueError(f"tile_k must be either 128 or 256, but got {tile_k}")

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

    if alpha is not None:
        if alpha.dtype != torch.float32:
            raise ValueError(f"alpha must be a float32 tensor, but got {alpha.dtype}")
        if alpha.device != a.device:
            raise ValueError(
                f"alpha must be on the same device as a, but got alpha.device={alpha.device}, a.device={a.device}"
            )
        if alpha.numel() != 0 and alpha.shape != (num_groups,):
            raise ValueError(
                f"alpha must be a empty or have shape {(num_groups,)}, but got alpha.shape={alpha.shape}"
            )

    if b.shape[0] != num_groups:
        raise ValueError(
            f"b.shape[0] must equal num_groups (m_indptr.shape[0] - 1), but got b.shape[0]={b.shape[0]}, num_groups={num_groups}"
        )

    # Check b, a_scale, and b_scale are all on the same device
    if b.device != a.device:
        raise ValueError(
            f"b must be on the same device as a, but got b.device={b.device}, a.device={a.device}"
        )
    if a_scale.device != a.device:
        raise ValueError(
            f"a_scale must be on the same device as a, but got a_scale.device={a_scale.device}, a.device={a.device}"
        )
    if b_scale.device != b.device:
        raise ValueError(
            f"b_scale must be on the same device as b, but got b_scale.device={b_scale.device}, b.device={b.device}"
        )

    n = b.shape[1]
    k = b.shape[2] * 2  # Multiply by 2 because b is e2m1 packed as uint8

    # assert a.shape[0] == m_indptr[-1].item()  # Not enabled in consideration of performance
    if a.shape[1] * 2 != k:
        raise ValueError(
            f"a.shape[1] * 2 must equal k, but got a.shape[1]={a.shape[1]}, k={k}"
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
    common_check=_check_group_gemm_nvfp4_nt_groupwise_problem_size,
)
@flashinfer_api
def group_gemm_nvfp4_nt_groupwise(
    a: torch.Tensor,  # (cum_m, k)
    b: torch.Tensor,  # (batch_size, n, k // 2)
    a_scale: torch.Tensor,  # (cum_m_padded, k // 16)
    b_scale: torch.Tensor,  # (batch_size, n_padded, k // 16)
    m_indptr: torch.Tensor,  # (batch_size + 1, )
    alpha: Optional[torch.Tensor] = None,  # (batch_size, )
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    out: Optional[torch.Tensor] = None,  # (cum_m, n)
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Perform group GEMM with NVFP4 data types using groupwise scaling. Currently only implemented on NVIDIA
    Blackwell Geforce, and DGX Spark architectures.

    Parameters
    ----------
    a: torch.Tensor
        Row-major input tensor, shape ``(cum_m, k // 2)``, data type is ``torch.uint8`` (packed NVFP4).
        ``cum_m`` is the cumulative sum of the segment lengths.

    b: torch.Tensor
        Column-major input tensor, shape ``(batch_size, n, k // 2)``, data type is ``torch.uint8``.

    a_scale: torch.Tensor
        Column-major scale tensor for a, shape ``(cum_m_padded, k // 16)``, data type is ``torch.uint8``.

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape ``(batch_size, n_padded, k // 16)``, data type is ``torch.uint8``.

    m_indptr: torch.Tensor
        The indptr of the segment lengths, shape ``(batch_size + 1,)``, data type is ``torch.int32``.
        Element element in ``m_indptr`` must be a multiple of 4.

    alpha: Optional[torch.Tensor] = None, # (batch_size, )
        The alpha tensor, shape ``(batch_size, )``, data type is ``torch.float32``.

    tile_m: int
        The tile size for the M dimension, must be 128.

    tile_n: int
        The tile size for the N dimension, must be 128.

    tile_k: int
        The tile size for the K dimension, must be 128 or 256.

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
        "group_gemm_nvfp4_nt_groupwise_int_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_nvfp4_nt_groupwise_float_workspace",
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

    if alpha is None:
        # empty torch tensor
        alpha = torch.tensor([], dtype=torch.float32, device=a.device)

    get_gemm_sm120_module().group_gemm_nvfp4_nt_groupwise(
        int_workspace_buffer,
        float_workspace_buffer,
        a,
        b,
        a_scale,
        b_scale,
        out,
        alpha,
        m_indptr,
        n,
        k,
        tile_m,
        tile_n,
        tile_k,
    )

    return out


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
@flashinfer_api(trace=batch_deepgemm_fp8_nt_groupwise_trace)
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


@flashinfer_api(trace=fp8_blockscale_gemm_sm90_trace)
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


def _calculate_block_scale_dims(
    m: int, n: int, k: int, block_size: int
) -> Tuple[int, int, int]:
    """Calculate block scale dimensions using indestructible block formula."""
    INDESTRUCTIBLE_128x4_BLOCK_M_N = 128
    INDESTRUCTIBLE_128x4_BLOCK_K = 4

    def div_up(a, b):
        return (a + b - 1) // b

    block_scale_dim_m = (
        div_up(m, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    )
    block_scale_dim_n = (
        div_up(n, INDESTRUCTIBLE_128x4_BLOCK_M_N) * INDESTRUCTIBLE_128x4_BLOCK_M_N
    )
    block_scale_dim_k = (
        div_up(div_up(k, block_size), INDESTRUCTIBLE_128x4_BLOCK_K)
        * INDESTRUCTIBLE_128x4_BLOCK_K
    )

    return block_scale_dim_m, block_scale_dim_n, block_scale_dim_k


@functools.lru_cache(maxsize=1024)
def create_cudnn_execution_plans_mxfp8_gemm(
    a_shape,
    a_stride,
    a_type,  # cudnn.data_type, FP8_E4M3 or FP8_E5M2
    b_shape,
    b_stride,
    b_type,  # cudnn.data_type, FP8_E4M3 or FP8_E5M2
    block_size,
    o_type,  # cudnn.data_type, BF16 or FP16
    device,
):
    if len(a_shape) != 3:
        raise ValueError(f"A shape must be 3D, got {a_shape}")
    if len(b_shape) != 3:
        raise ValueError(f"B shape must be 3D, got {b_shape}")

    if a_type not in [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2]:
        raise ValueError(f"A type must be FP8_E4M3 or FP8_E5M2, got {a_type}")
    if b_type not in [cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2]:
        raise ValueError(f"B type must be FP8_E4M3 or FP8_E5M2, got {b_type}")
    if o_type not in [cudnn.data_type.BFLOAT16, cudnn.data_type.HALF]:
        raise ValueError(f"Output type must be BF16 or FP16, got {o_type}")

    # Extract batch, m, n, k dimensions
    b_dim = a_shape[0]
    m = a_shape[1]
    k = a_shape[2]
    n = b_shape[2]

    # Calculate block scale dimensions using indestructible block formula
    block_scale_dim_m, block_scale_dim_n, block_scale_dim_k = (
        _calculate_block_scale_dims(m, n, k, block_size)
    )

    # For mxfp8, scale tensors need to be reshaped to 3D with correct strides
    # cuDNN expects K-major layout: stride for K dimension should be 1
    # For block_descale_a: shape [b, block_scale_dim_m, block_scale_dim_k], stride [block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1]
    # For block_descale_b: shape [b, block_scale_dim_k, block_scale_dim_n], stride [block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k]

    a_descale_shape = (b_dim, block_scale_dim_m, block_scale_dim_k)
    a_descale_stride = (
        block_scale_dim_m * block_scale_dim_k,
        block_scale_dim_k,
        1,
    )

    b_descale_shape = (b_dim, block_scale_dim_k, block_scale_dim_n)
    b_descale_stride = (
        block_scale_dim_n * block_scale_dim_k,
        1,
        block_scale_dim_k,
    )

    # MXFP8 uses FP8_E4M3/FP8_E5M2 for quantized data
    # MXFP8 uses FP8_E8M0 for scale data
    scale_type = cudnn.data_type.FP8_E8M0

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
        a_cudnn_tensor = graph.tensor(
            name="a",
            dim=tuple(a_shape),  # [b, m, k]
            stride=tuple(a_stride),  # [m * k, k, 1]
            data_type=a_type,
        )
        b_cudnn_tensor = graph.tensor(
            name="b",
            dim=tuple(b_shape),  # [b, k, n]
            stride=tuple(b_stride),  # [k * n, 1, k]
            data_type=b_type,
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

        # Dequantize the input tensors
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

        # The actual matmul operation
        c_tensor = graph.matmul(
            dequant_a_tensor,
            dequant_b_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
            name="gemm",
        )
        c_tensor.set_data_type(cudnn.data_type.FLOAT)

        # Output the dequantized result with the specified output dtype
        c_tensor.set_output(True).set_data_type(o_type)
        c_final_cudnn_tensor = c_tensor

        a_cudnn_tensor.set_uid(UIDs.A_UID.value)
        b_cudnn_tensor.set_uid(UIDs.B_UID.value)
        block_descale_a_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_A_UID.value)
        block_descale_b_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_B_UID.value)
        c_final_cudnn_tensor.set_uid(UIDs.O_UID.value)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.B])

        return graph


def _get_cudnn_mxfp8_gemm_graph(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    block_size: int = 32,  # mxfp8 block size is 32
    policy=None,
):
    graph = create_cudnn_execution_plans_mxfp8_gemm(
        a_shape=a.shape,
        a_stride=a.stride(),
        b_shape=b.shape,
        b_stride=b.stride(),
        a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
        b_type=_torch_data_type_to_cudnn_data_type(b.dtype),
        o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
        block_size=block_size,
        device=a.device,
    )

    graph.check_support()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
    graph.build_plans(policy)
    return graph


def _cudnn_gemm_mxfp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    workspace_buffer: torch.Tensor = None,
    tactic: int = -1,
):
    # mxfp8 block size is 32
    block_size = 32

    if tactic == -1:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
    else:
        policy = cudnn.build_plan_policy.ALL

    # Graph should have been already cached, when we ran _cudnn_bmm_mxfp8_requirement
    graph = _get_cudnn_mxfp8_gemm_graph(
        a=a,
        b=b,
        out_dtype=out_dtype,
        out=out,
        block_size=block_size,
        policy=policy,
    )
    # execute the mxfp8 cudnn graph
    execute_cudnn_gemm_mxfp8_graph(
        graph=graph,
        a=a,
        b=b,
        a_descale=a_descale,
        b_descale=b_descale,
        c_final=out,
        workspace_buffer=workspace_buffer,
        tactic=tactic,
    )


def _cudnn_gemm_mxfp8_runner():
    class CudnnMxfp8GemmRunner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            a, b, _, _, out, _ = inputs
            return (a.dtype, b.dtype, out.dtype)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a, b, _, _, out, _ = inputs
            graph = _get_cudnn_mxfp8_gemm_graph(
                a=a,
                b=b,
                out_dtype=out.dtype,
                out=out,
                policy=cudnn.build_plan_policy.ALL,
            )
            return list(range(graph.get_execution_plan_count()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, scale_a, scale_b, out, workspace_buffer = inputs
            _cudnn_gemm_mxfp8(
                a=a,
                b=b,
                a_descale=scale_a,
                b_descale=scale_b,
                out=out,
                out_dtype=out.dtype,
                workspace_buffer=workspace_buffer,
                tactic=tactic,
            )
            return out

    return CudnnMxfp8GemmRunner()


def mxfp8_gemm_sm100(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    runner_names: List[str],
) -> None:
    runners = []
    if "cudnn" in runner_names:
        runners.append(_cudnn_gemm_mxfp8_runner())
    assert runners, "No suitable runners found"
    tuner = AutoTuner.get()

    inputs = [a, b, scale_a, scale_b, out, workspace_buffer]
    runner, tactic = tuner.choose_one(
        "mxfp8_gemm",  # TODO: check if this is correct
        runners,
        _FP8_GEMM_SM100_TUNING_CONFIG,  # TODO: check if this is correct
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)


@supported_compute_capability([100, 103])
def _cudnn_bmm_mxfp8_requirement(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn"] = "cudnn",
):
    _check_cudnn_availability()
    return True


def _validate_mxfp8_output_dtype(dtype: torch.dtype):
    """Validate that the output dtype is either bf16 or fp16."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Unsupported output dtype: {dtype}. "
            f"Only torch.bfloat16 and torch.float16 are supported for MXFP8 GEMM operations."
        )


def _check_bmm_mxfp8_problem_size(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn"] = "cudnn",
):
    # Check input tensors
    if A.ndim != 3 or B.ndim != 3:
        # A is [b, m, k], B is [b, k, n]
        raise ValueError(f"bmm_mxfp8 accepts 3d tensors, got {A.shape=} and {B.shape=}")
    if A.shape[2] != B.shape[1]:
        raise ValueError(
            f"K dimension (last dim of A) mismatch in bmm_mxfp8. got {A.shape=}, {B.shape=}"
        )

    _validate_mxfp8_output_dtype(dtype)
    return True


@supported_compute_capability([120, 121])
def _cutlass_bmm_mxfp8_requirement(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cutlass", "auto"] = "auto",
):
    # SM120/121 CUTLASS MXFP8 only supports 1D swizzled scales.
    if A_scale.ndim != 1 or B_scale.ndim != 1:
        return False
    return True


def _heuristic_func_bmm_mxfp8(
    suitable_backends: List[str],
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cutlass", "auto"] = "auto",
):
    heuristic_backends = []
    major, _ = get_compute_capability(A.device)
    if major == 12 and "cutlass" in suitable_backends:
        heuristic_backends.append("cutlass")
    elif CUDNN_AVAILABLE and "cudnn" in suitable_backends:
        heuristic_backends.append("cudnn")
    return heuristic_backends


@backend_requirement(
    {
        "cudnn": _cudnn_bmm_mxfp8_requirement,
        "cutlass": _cutlass_bmm_mxfp8_requirement,
    },
    common_check=_check_bmm_mxfp8_problem_size,
    heuristic_func=_heuristic_func_bmm_mxfp8,
)
@flashinfer_api(trace=bmm_mxfp8_trace)
def bmm_mxfp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: Literal["cudnn", "cutlass", "auto"] = "auto",
) -> torch.Tensor:
    r"""BMM MXFP8

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (b, m, k), fp8 e4m3 or fp8 e5m2.

    B: torch.Tensor
        Mat2 tensor, shape (b, k, n), should be column major, fp8 e4m3 or fp8 e5m2.

    A_scale: torch.Tensor
        Scale tensor for A, uint8 (fp8 e8m0 format).

    B_scale: torch.Tensor
        Scale tensor for B, uint8 (fp8 e8m0 format).

    dtype: torch.dtype
        out dtype, bf16 or fp16.

    out: Optional[torch.Tensor]
        Out tensor, shape (b, m, n), bf16 or fp16, defaults to ``None``.

    backend: Literal["cudnn", "cutlass", "auto"]
        The backend to use for the operation. Defaults to ``"auto"``.
        On SM120/121 GPUs, ``"auto"`` selects the CUTLASS backend; scales must
        be 1D swizzled (``SfLayout.layout_128x4``). Pass ``B`` in the standard
        shape ``[b, k, n]`` (column-major); the CUTLASS path transposes internally.

    Returns
    -------
    out: torch.Tensor
        Out tensor, shape (b, m, n), bf16 or fp16.
    """

    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )

    workspace_buffer = _get_cache_buf(
        "bmm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, A.device
    )

    resolved_backend = backend
    if resolved_backend == "auto":
        if not bmm_mxfp8.suitable_auto_backends:
            raise ValueError("No suitable backend found for bmm_mxfp8")
        resolved_backend = bmm_mxfp8.suitable_auto_backends[0]

    if resolved_backend == "cutlass":
        # SM120/121 CUTLASS path.
        # B is [b, k, n] col-major; CUTLASS expects mat2 as [B, N, K].
        # col-major [b, k, n] with strides (k*n, 1, k) → .transpose(1,2) → [b, n, k]
        # with strides (k*n, k, 1), which is contiguous row-major [B, N, K].
        B_cutlass = B.transpose(1, 2)
        if not B_cutlass.is_contiguous():
            B_cutlass = B_cutlass.contiguous()
        raw_module = _load_gemm_sm120_mxfp8_module()
        raw_module.mxfp8_gemm(A, B_cutlass, A_scale, B_scale, out, workspace_buffer, -1)
        return out

    if resolved_backend == "cudnn":
        if not CUDNN_AVAILABLE:
            raise ValueError("cudnn is not available")
        mxfp8_gemm_sm100(A, B, A_scale, B_scale, out, workspace_buffer, ["cudnn"])
        return out

    raise ValueError(f"Invalid backend: {backend}")
