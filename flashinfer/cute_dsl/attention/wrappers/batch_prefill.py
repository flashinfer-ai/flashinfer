# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BatchPrefillCuteDSLWrapper — PyTorch-facing API for batch prefill attention.

Constructs AttentionConfig + AttentionFusion from user-facing parameters,
creates the kernel, compiles it, and provides the run() interface.
"""

import math
from typing import Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Int32

from ..config import AttentionConfig, AttentionFusion
from ..fusion.mask import MaskType
from ..fusion.variant import AttentionVariant, StandardAttention
from ..prefill import BlackwellFusedMultiHeadAttentionForward


class BatchPrefillCuteDSLWrapper:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        self._use_cuda_graph = use_cuda_graph

        # Data types will be set in plan() method based on input parameters
        self._in_dtype = None
        self._out_dtype = None
        self._qk_acc_dtype = cutlass.Float32
        self._pv_acc_dtype = cutlass.Float32

    def plan(
        self,
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo=None,
        causal=True,
        sm_scale=1.0,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        window_left: int = -1,
        variant: AttentionVariant | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run this example!")

        self._batch_size = qo_indptr.shape[0] - 1
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        assert num_qo_heads % num_kv_heads == 0, (
            "num_qo_heads must be divisible by num_kv_heads"
        )
        self._head_dim = head_dim_qk
        assert head_dim_vo is None or head_dim_vo == head_dim_qk, (
            "head_dim_vo must be None or equal to head_dim_qk"
        )
        self._causal = causal
        self._sm_scale = sm_scale
        self._device = qo_indptr.device
        self._is_persistent = True

        h_r = num_qo_heads // num_kv_heads

        if variant is None:
            variant = StandardAttention()
        self._variant = variant

        # Set data types based on input parameters
        if q_data_type == torch.bfloat16:
            self._in_dtype = cutlass.BFloat16
            self._out_dtype = cutlass.BFloat16
        elif q_data_type == torch.half:
            self._in_dtype = cutlass.Float16
            self._out_dtype = cutlass.Float16
        elif q_data_type == torch.float8_e4m3fn:
            self._in_dtype = cutlass.Float8E4M3FN
            self._out_dtype = cutlass.Float16  # Output is always Float16 for FP8 input
        else:
            raise ValueError(f"Unsupported input data type: {q_data_type}")

        s_cumsum_q_cute_tensor, s_cumsum_q_torch_tensor = (
            cutlass_torch.cute_tensor_like(
                qo_indptr.to(torch.int32),
                Int32,
                is_dynamic_layout=True,
                assumed_align=16,
            )
        )
        s_q = qo_indptr[1:] - qo_indptr[:-1]

        s_cumsum_k_cute_tensor, s_cumsum_k_torch_tensor = (
            cutlass_torch.cute_tensor_like(
                kv_indptr.to(torch.int32),
                Int32,
                is_dynamic_layout=True,
                assumed_align=16,
            )
        )
        s_k = kv_indptr[1:] - kv_indptr[:-1]

        qo_shape = (1, torch.sum(s_q), h_r * self._num_kv_heads, self._head_dim)
        o_padding = (0, torch.max(s_q), 0, 0, 0)
        kv_shape = (1, torch.sum(s_k), self._num_kv_heads, self._head_dim)

        self._o_padding = o_padding[1]
        self._kv_padding = 0

        q_ref, q_cute, q_torch = create_and_pad_tensor(
            qo_shape,
            (0, 0, 0, 0, 0),
            self._in_dtype,
            s_cumsum=s_cumsum_q_torch_tensor,
            is_dynamic_layout=True,
        )
        k_ref, k_cute, k_torch = create_and_pad_tensor(
            kv_shape,
            (0, 0, 0, 0, 0),
            self._in_dtype,
            s_cumsum=s_cumsum_k_torch_tensor,
            is_dynamic_layout=True,
        )
        v_ref, v_cute, v_torch = create_and_pad_tensor(
            kv_shape,
            (0, 0, 0, 0, 0),
            self._in_dtype,
            s_cumsum=s_cumsum_k_torch_tensor,
            is_dynamic_layout=True,
        )

        _, o_cute, o_torch = create_and_pad_tensor(
            qo_shape,
            o_padding,
            self._out_dtype,
            s_cumsum=s_cumsum_q_torch_tensor,
            is_dynamic_layout=True,
        )

        self._has_params = self._variant.extra_params is not None
        if self._has_params:
            ep = self._variant.extra_params.to(torch.float32).to(self._device)
            if not ep.is_contiguous():
                raise ValueError(
                    f"AttentionVariant.extra_params must be contiguous, "
                    f"got strides {ep.stride()} for shape {ep.shape}. "
                    f"Call .contiguous() before returning from extra_params."
                )
            self._params_torch = ep
            params_cute = from_dlpack(ep, assumed_align=16)

        self._mma_tiler_mn = (128, 128)
        self._mma_tiler = (128, 128, self._head_dim)

        # Determine mask type
        self._mask_type = MaskType.NO_MASK
        if self._causal:
            self._mask_type = MaskType.CAUSAL_MASK
        elif window_left > 0:
            self._mask_type = MaskType.SLIDING_WINDOW_MASK
        else:
            if torch.any(s_k % self._mma_tiler_mn[1] != 0).item():
                self._mask_type = MaskType.RESIDUAL_MASK

        # Build AttentionConfig and AttentionFusion, then create the kernel
        config = AttentionConfig(
            qk_acc_dtype=self._qk_acc_dtype,
            pv_acc_dtype=self._pv_acc_dtype,
            mma_tiler=self._mma_tiler,
            is_persistent=self._is_persistent,
            mask_type=self._mask_type,
            num_repeat_kv_heads=h_r,
            window_left=window_left,
        )
        fusion = AttentionFusion(variant=self._variant)
        fmha = BlackwellFusedMultiHeadAttentionForward(config, fusion)

        problem_size = (
            self._batch_size,
            int(torch.max(s_q).item()),
            int(torch.max(s_k).item()),
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
        )

        self._problem_size = problem_size
        self._s_cumsum_q_cute_tensor = s_cumsum_q_cute_tensor
        self._s_cumsum_k_cute_tensor = s_cumsum_k_cute_tensor
        self._s_q_all = s_cumsum_q_torch_tensor[-1].item()
        self._s_k_all = s_cumsum_k_torch_tensor[-1].item()

        log2_e = math.log2(
            math.exp(1.0)
        )  # gpu uses exp2 for perf concerns, we need an extra factor 'log2_e' here
        scale_softmax = self._sm_scale
        self._scale_softmax_log2 = scale_softmax * log2_e
        self._scale_output = 1.0

        # Get current CUDA stream from PyTorch
        torch_stream = torch.cuda.current_stream()
        # Get the raw stream pointer as a CUstream
        stream = cuda.CUstream(torch_stream.cuda_stream)

        # compile fmha kernel
        compiled_fmha = cute.compile(
            fmha,
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            self._problem_size,
            self._s_cumsum_q_cute_tensor,
            self._s_q_all,
            self._s_cumsum_k_cute_tensor,
            self._s_k_all,
            self._scale_softmax_log2,
            self._scale_output,
            params_cute.iterator if self._has_params else None,
            stream,
        )

        self._compiled_fmha = compiled_fmha

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the prefill attention computation.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape [batch_size, seq_len, num_heads, head_dim].
        k : torch.Tensor
            The key tensor with shape [batch_size, seq_len, num_heads, head_dim].
        v : torch.Tensor
            The value tensor with shape [batch_size, seq_len, num_heads, head_dim].
        out : Optional[torch.Tensor], optional
            The output tensor. If None, a new tensor will be created.

        Returns
        -------
        torch.Tensor
            The output tensor with shape [batch_size, seq_len, num_heads, head_dim].
        """

        if self._compiled_fmha is None:
            raise RuntimeError("Plan the prefill attention computation first!")

        if out is None:
            out = torch.empty_like(q, device=q.device)

        q_cute = from_dlpack(q, assumed_align=16)
        k_cute = from_dlpack(k, assumed_align=16)
        v_cute = from_dlpack(v, assumed_align=16)
        o_cute, o_torch = qkv_torch_2_cute(out, self._o_padding, self._out_dtype)

        if self._has_params:
            params_cute = from_dlpack(self._params_torch, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        self._compiled_fmha(
            q_cute.iterator,
            k_cute.iterator,
            v_cute.iterator,
            o_cute.iterator,
            self._problem_size,
            self._s_cumsum_q_cute_tensor,
            self._s_q_all,
            self._s_cumsum_k_cute_tensor,
            self._s_k_all,
            self._scale_softmax_log2,
            self._scale_output,
            params_cute.iterator if self._has_params else None,
            stream,
        )

        return o_torch


def qkv_torch_2_cute(x_torch, padding, dtype, s_cumsum=None, is_dynamic_layout=True):
    # (b, s, h, d)

    # pad tensor in front of the tensor on the second dimension
    x_torch_full = torch.nn.functional.pad(x_torch, (0, 0, 0, 0, padding, 0))

    x_torch = x_torch_full[padding:, :, :].detach()
    x_torch._keep_alive = x_torch_full

    # Create dtype cute tensor with offset (gpu)
    x_cute = from_dlpack(x_torch, assumed_align=16)
    x_cute.element_type = dtype

    return (x_cute, x_torch)


def create_and_pad_tensor(shape, padding, dtype, s_cumsum=None, is_dynamic_layout=True):
    # (b, s, h, d)
    shape_ = tuple(map(lambda x, y: x + y, shape, padding))
    if s_cumsum is not None:
        if shape_[0] != 1 or padding[0] != 0:
            raise ValueError("Invalid tensor creation for variable sequence length")
        # (s_total + padding, h, d)
        shape_ = shape_[1:]
        padding = padding[1:]

    # Create f32 torch tensor (cpu)
    f32_torch_tensor_full = cutlass_torch.create_and_permute_torch_tensor(
        shape_,
        torch.float32,
        permute_order=None,
        init_type=cutlass.torch.TensorInitType.RANDOM,
        init_config=cutlass.torch.RandomInitConfig(
            min_val=-2 if dtype.is_float or dtype.signed else 0, max_val=2
        ),
    )
    # Create dtype cute & torch tensor (gpu)
    _, torch_tensor_full = cutlass_torch.cute_tensor_like(
        f32_torch_tensor_full,
        dtype,
        is_dynamic_layout,
        assumed_align=16,
    )

    # Offset the tensor
    slices = tuple(slice(s, e) for s, e in zip(padding, shape_))
    torch_tensor = torch_tensor_full[slices].detach()
    f32_torch_tensor = f32_torch_tensor_full[slices].detach()
    torch_tensor._keep_alive = torch_tensor_full
    f32_torch_tensor._keep_alive = f32_torch_tensor_full

    # Create dtype cute tensor with offset (gpu)
    cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
    cute_tensor.element_type = dtype

    # From ragged to jagged
    if s_cumsum is not None:
        torch_tensor = torch.nested.nested_tensor_from_jagged(
            values=torch_tensor, offsets=s_cumsum
        )
        f32_torch_tensor = torch.nested.nested_tensor_from_jagged(
            values=f32_torch_tensor, offsets=s_cumsum.cpu()
        )

    return (
        f32_torch_tensor,
        cute_tensor,
        torch_tensor,
    )
