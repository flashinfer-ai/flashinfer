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

from typing import Optional

import torch

from .utils import get_indptr, get_compute_capability
from .jit import load_cuda_ops, FLASHINFER_CSRC_DIR, has_prebuilt_ops
from typing import Optional


_gemm_module = None
_gemm_module_sm90 = None


def get_gemm_module():
    global _gemm_module
    if _gemm_module is None:
        if has_prebuilt_ops:
            from . import _kernels

            _gemm_module = _kernels
        else:
            _gemm_module = load_cuda_ops(
                "gemm",
                [
                    FLASHINFER_CSRC_DIR / "bmm_fp8.cu",
                    FLASHINFER_CSRC_DIR / "group_gemm.cu",
                    FLASHINFER_CSRC_DIR / "flashinfer_gemm_ops.cu",
                ],
            )
    return _gemm_module


def get_gemm_sm90_module():
    print("get_gemm_sm90_module")
    global _gemm_module_sm90
    if _gemm_module_sm90 is None:
        if has_prebuilt_ops:
            from . import _kernels_sm90

            _gemm_module_sm90 = _kernels_sm90
        else:
            _gemm_module_sm90 = load_cuda_ops(
                "gemm_sm90",
                [
                    FLASHINFER_CSRC_DIR / "group_gemm_sm90.cu",
                    FLASHINFER_CSRC_DIR / "flashinfer_gemm_sm90_ops.cu",
                ],
                extra_cuda_cflags=["-gencode", "arch=compute_90a,code=sm_90a"],
            )
    return _gemm_module_sm90


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

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
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
        is a segment of the concatenated tensor. Please see :ref:`Ragged Tensor tutorial <ragged-layout>` for more details.
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
        major, _ = get_compute_capability(x.device)
        if major >= 9:
            return get_gemm_sm90_module().cutlass_segment_gemm_sm90(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                seg_indptr,
                weight_indices,
                x,
                weights,
                batch_size,
                weight_column_major,
            )
        else:
            return get_gemm_module().cutlass_segment_gemm(
                self._int_workspace_buffer,
                seg_indptr,
                weight_indices,
                x,
                weights,
                batch_size,
                weight_column_major,
            )

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
    get_gemm_module().bmm_fp8(A, B, out, A_scale, B_scale)
    return out
