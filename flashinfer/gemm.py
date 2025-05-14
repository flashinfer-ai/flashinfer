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
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .jit import FLASHINFER_CSRC_DIR, has_prebuilt_ops, load_cuda_ops
from .utils import (
    _get_cache_buf,
    determine_gemm_backend,
    get_indptr,
    is_float8,
    register_custom_op,
    register_fake_op,
)

_gemm_module = None
_gemm_module_sm90 = None


def get_gemm_module():
    global _gemm_module
    if _gemm_module is None:
        if has_prebuilt_ops:
            _kernels = torch.ops.flashinfer_kernels

            module = _kernels
        else:
            module = load_cuda_ops(
                "gemm",
                [
                    FLASHINFER_CSRC_DIR / "bmm_fp8.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm.cu",
                    FLASHINFER_CSRC_DIR / "flashinfer_gemm_ops.cu",
                ],
                extra_ldflags=["-lcublas", "-lcublasLt"],
            )

        # torch library for bmm_fp8

        @register_custom_op(
            "flashinfer::bmm_fp8", mutates_args=("workspace_buffer", "D")
        )
        def bmm_fp8(
            workspace_buffer: torch.Tensor,
            A: torch.Tensor,
            B: torch.Tensor,
            D: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
        ) -> None:
            cublas_handle = torch.cuda.current_blas_handle()
            module.bmm_fp8.default(
                A,
                B,
                D,
                A_scale,
                B_scale,
                workspace_buffer,
                cublas_handle,
            )

        @register_fake_op("flashinfer::bmm_fp8")
        def _fake_bmm_fp8(
            workspace_buffer: torch.Tensor,
            A: torch.Tensor,
            B: torch.Tensor,
            D: torch.Tensor,
            A_scale: torch.Tensor,
            B_scale: torch.Tensor,
        ) -> None:
            pass

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
            bmm_fp8=bmm_fp8,
            cutlass_segment_gemm=cutlass_segment_gemm,
        )

    return _gemm_module


@functools.cache
def get_gemm_sm100_module():
    if has_prebuilt_ops:
        _kernels_sm100 = torch.ops.flashinfer_kernels_sm100
        module = _kernels_sm100
    else:
        module = load_cuda_ops(
            "gemm_sm100",
            [
                FLASHINFER_CSRC_DIR / "gemm_groupwise_sm100.cu",
                FLASHINFER_CSRC_DIR / "group_gemm_groupwise_sm100.cu",
                FLASHINFER_CSRC_DIR / "gemm_sm100_pybind.cu",
                FLASHINFER_CSRC_DIR / "group_gemm_sm100_pybind.cu",
            ],
            extra_cuda_cflags=["-gencode", "arch=compute_100a,code=sm_100a"],
        )

    return module


def get_gemm_sm90_module():
    global _gemm_module_sm90
    if _gemm_module_sm90 is None:
        if has_prebuilt_ops:
            _kernels_sm90 = torch.ops.flashinfer_kernels_sm90

            module = _kernels_sm90
        else:
            module = load_cuda_ops(
                "gemm_sm90",
                [
                    FLASHINFER_CSRC_DIR / "group_gemm_sm90.cu",
                    FLASHINFER_CSRC_DIR / "flashinfer_gemm_sm90_ops.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm_f16_f16_sm90.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm_bf16_bf16_sm90.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm_e4m3_f16_sm90.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm_e5m2_f16_sm90.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm_e4m3_bf16_sm90.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm_e5m2_bf16_sm90.cu",
                ],
                extra_cuda_cflags=["-gencode", "arch=compute_90a,code=sm_90a"],
            )

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
        _gemm_module_sm90 = SimpleNamespace(
            cutlass_segment_gemm_sm90=cutlass_segment_gemm_sm90,
        )

    return _gemm_module_sm90


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


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
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
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    get_gemm_module().bmm_fp8(workspace_buffer, A, B, out, A_scale, B_scale)
    return out


def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int] = (1, 128, 128),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
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
        Column-major scale tensor for a, shape (ceil_div(k, k_granularity), ceil_div(m, m_granularity)).

    b_scale: torch.Tensor
        Row-major scale tensor for b, shape (ceil_div(k, k_granularity), ceil_div(n, n_granularity)).

    scale_granularity_mnk: Tuple[int, int, int]
        The granularity of the scale tensor, (m_granularity, n_granularity, k_granularity).

    out: Optional[torch.Tensor]
        Output tensor, shape (m, n). If not specified, we will create an output tensor explicitly.

    out_dtype: Optional[torch.dtype]
        If out is not specified, we will create an output tensor with this dtype.
        Defaults to ``torch.bfloat16``.

    Returns
    -------
    out: torch.Tensor
        Output tensor, shape (m, n).

    Notes
    -----
    If ``m`` is not a multiple of 4, we will pad ``m`` to the next multiple of 4 to accommodate the kernel's requirement.
    """
    workspace_buffer = _get_cache_buf(
        "gemm_fp8_nt_groupwise_workspace", 32 * 1024 * 1024, a.device
    )
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Shape mismatch. a.shape = {a.shape}, b.shape = {b.shape}")
    m = a.shape[0]
    m_padded = ((m + 3) // 4) * 4
    need_padding = m_padded != m
    if need_padding:
        a_padded = F.pad(a, (0, 0, 0, m_padded - m))
        # a_scale is (k // 128, m)
        a_scale_padded = F.pad(a_scale, (0, m_padded - m))
    else:
        a_padded = a
        a_scale_padded = a_scale

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Shape mismatch. a.shape[1] = {a.shape[1]}, b.shape[1] = {b.shape[1]}"
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

    if need_padding or out is None:
        out_padded = torch.empty(
            m_padded,
            b.shape[0],
            device=a.device,
            dtype=out_dtype,
        )
    else:
        out_padded = out

    get_gemm_sm100_module().gemm_fp8_nt_groupwise.default(
        workspace_buffer,
        a_padded,
        b,
        a_scale_padded,
        b_scale,
        out_padded,
        *scale_granularity_mnk,
    )

    # NOTE(Zihao): if out is specified and need_padding is True, we need to copy
    # the result back to out

    if out is not None and need_padding:
        out.copy_(out_padded[:m])
    else:
        out = out_padded[:m]

    return out


def gemm_fp8_nt_blockscaled(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
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
    out: Optional[torch.Tensor] = None,  # (cum_m, n)
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    from .triton.gemm import compute_padding_mapping

    int_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_int_workspace", 32 * 1024 * 1024, a.device
    )
    float_workspace_buffer = _get_cache_buf(
        "group_gemm_fp8_nt_groupwise_float_workspace", 32 * 1024 * 1024, a.device
    )

    batch_size = m_indptr.shape[0] - 1
    assert b.shape[0] == batch_size
    assert b_scale.shape[0] == batch_size
    n = b.shape[1]
    k = b.shape[2]

    if out is None:
        out_dtype = out_dtype or torch.bfloat16
        out = torch.empty(a.shape[0], n, dtype=out_dtype, device=a.device)

    if (m_indptr % 4 == 0).all():
        get_gemm_sm100_module().group_gemm_fp8_nt_groupwise.default(
            int_workspace_buffer,
            float_workspace_buffer,
            a,
            b,
            a_scale,
            b_scale,
            out,
            m_indptr,
            m_indptr[-1],
            n,
            k,
            *scale_granularity_mnk,
        )
        return out

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

    padded_a = torch.zeros((padded_m_indptr[-1], k), dtype=a.dtype, device=a.device)
    padded_out = torch.zeros(
        (padded_m_indptr[-1], n), dtype=out.dtype, device=out.device
    )
    padded_a_scale = torch.zeros(
        (k // scale_granularity_mnk[2], padded_m_indptr[-1]),
        dtype=a_scale.dtype,
        device=a_scale.device,
    )

    padded_a[padded_m_rank] = a[m_rank]
    padded_a_scale[::, padded_m_rank] = a_scale[::, m_rank]

    get_gemm_sm100_module().group_gemm_fp8_nt_groupwise.default(
        int_workspace_buffer,
        float_workspace_buffer,
        padded_a,
        b,
        padded_a_scale,
        b_scale,
        padded_out,
        padded_m_indptr,
        padded_m_indptr[-1],
        n,
        k,
        *scale_granularity_mnk,
    )

    out[m_rank] = padded_out[padded_m_rank]

    return out
