# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BatchPrefillCuteDSLWrapper — PyTorch-facing API for batch prefill attention.

Constructs AttentionConfig + AttentionFusion from user-facing parameters,
creates the kernel, compiles it via TVM-FFI, and provides the run() interface.
Compilation is memoized via @functools.cache with symbolic tensor dimensions,
so kernels are compiled once per (dtype, heads, head_dim, mask, variant) combo
and reused across batches of any size.
"""

import functools
import math
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.attention import cute_dsl_batch_prefill_run_trace

from ..config import AttentionConfig, AttentionFusion
from ..fusion.mask import MaskType
from ..fusion.variant import AttentionVariant, StandardAttention
from ..prefill import BlackwellFusedMultiHeadAttentionForward


@functools.cache
def _get_compiled_prefill_kernel(
    in_dtype,
    out_dtype,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    mask_type,
    window_left,
    is_persistent,
    variant,
    params_shape,
):
    """Compile and cache the prefill kernel.

    Uses symbolic dimensions for sequence lengths and batch size so the same
    compiled kernel can be reused across different batch shapes.  Pass
    ``variant=None`` for standard attention (always cache-hits); pass the
    actual variant instance for custom variants (hashable by identity).

    ``AttentionFusion`` is constructed *inside* this function so it never
    appears in the cache key (it is unhashable).
    """
    if variant is None:
        variant = StandardAttention()
    fusion = AttentionFusion(variant=variant)
    h_r = num_qo_heads // num_kv_heads

    config = AttentionConfig(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=(128, 128, head_dim),
        is_persistent=is_persistent,
        mask_type=mask_type,
        num_repeat_kv_heads=h_r,
        window_left=window_left,
    )
    _dtype_width_map = {
        cutlass.Float16: 16,
        cutlass.BFloat16: 16,
        cutlass.Float8E4M3FN: 8,
    }
    config.can_implement(dtype_width=_dtype_width_map[in_dtype])
    fmha = BlackwellFusedMultiHeadAttentionForward(config, fusion)

    sym_s_q = cute.sym_int()
    sym_s_k = cute.sym_int()
    sym_batch_p1 = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_s_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_s_k, num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_s_k, num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    o_fake = cute.runtime.make_fake_compact_tensor(
        out_dtype,
        (sym_s_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    cum_seqlen_q_fake = cute.runtime.make_fake_compact_tensor(
        Int32,
        (sym_batch_p1,),
        assumed_align=16,
    )
    cum_seqlen_k_fake = cute.runtime.make_fake_compact_tensor(
        Int32,
        (sym_batch_p1,),
        assumed_align=16,
    )

    params_fake = None
    if params_shape is not None:
        ndim = len(params_shape)
        stride_order = tuple(range(ndim - 1, -1, -1))
        params_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            params_shape,
            stride_order=stride_order,
            assumed_align=16,
        )

    problem_size = (1, 1, 1, num_qo_heads, num_kv_heads, head_dim)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        q_fake,
        k_fake,
        v_fake,
        o_fake,
        problem_size,
        cum_seqlen_q_fake,
        1,
        cum_seqlen_k_fake,
        1,
        0.0,
        1.0,
        params_fake,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


class BatchPrefillCuteDSLWrapper:
    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        # Named float_workspace_buffer for compatibility with the parent
        # BatchPrefillWithRaggedKVCacheWrapper API. Callers typically pass
        # torch.uint8; the CuTe DSL kernel does not use this buffer.
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        self._use_cuda_graph = use_cuda_graph

        self._in_dtype = None
        self._out_dtype = None
        self._compiled_fmha = None

    @flashinfer_api
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
        """Compile the FMHA prefill kernel for the given configuration.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            Cumulative query sequence lengths, shape [batch_size + 1].
        kv_indptr : torch.Tensor
            Cumulative KV sequence lengths, shape [batch_size + 1].
        num_qo_heads : int
            Number of query/output heads.
        num_kv_heads : int
            Number of key/value heads (must divide num_qo_heads).
        head_dim_qk : int
            Head dimension for queries and keys.
        head_dim_vo : Optional[int]
            Head dimension for values and output. Must equal head_dim_qk if set.
        causal : bool
            Whether to apply causal masking.
        sm_scale : float
            Softmax scale factor (typically 1/sqrt(head_dim)).
        q_data_type : torch.dtype
            Data type for queries (float16, bfloat16, or float8_e4m3fn).
        kv_data_type : torch.dtype
            Data type for keys/values.
        window_left : int
            Sliding window size. -1 disables sliding window.
        variant : Optional[AttentionVariant]
            Attention variant (ALiBi, RPE, Sigmoid, etc.). None uses standard softmax.
        """
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

        if variant is None:
            variant = StandardAttention()
        self._variant = variant

        self._q_data_type = q_data_type

        # Map torch dtype → cutlass dtype
        _dtype_map = {
            torch.float16: (cutlass.Float16, cutlass.Float16),
            torch.bfloat16: (cutlass.BFloat16, cutlass.BFloat16),
            torch.float8_e4m3fn: (cutlass.Float8E4M3FN, cutlass.Float16),
        }
        if q_data_type not in _dtype_map:
            raise ValueError(f"Unsupported input data type: {q_data_type}")
        self._in_dtype, self._out_dtype = _dtype_map[q_data_type]

        # Sequence lengths from indptr
        s_q = qo_indptr[1:] - qo_indptr[:-1]
        s_k = kv_indptr[1:] - kv_indptr[:-1]
        s_q_all = int(qo_indptr[-1].item())
        s_k_all = int(kv_indptr[-1].item())
        max_s_q = int(torch.max(s_q).item())
        max_s_k = int(torch.max(s_k).item())

        # Store for runtime
        self._qo_indptr = qo_indptr.to(torch.int32)
        self._kv_indptr = kv_indptr.to(torch.int32)
        self._s_q_all = s_q_all
        self._s_k_all = s_k_all
        self._o_padding = max_s_q

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

        mma_tiler_n = 128

        # Determine mask type
        self._mask_type = MaskType.NO_MASK
        if self._causal:
            self._mask_type = MaskType.CAUSAL_MASK
        elif window_left > 0:
            self._mask_type = MaskType.SLIDING_WINDOW_MASK
        else:
            if torch.any(s_k % mma_tiler_n != 0).item():
                self._mask_type = MaskType.RESIDUAL_MASK

        self._problem_size = (
            self._batch_size,
            max_s_q,
            max_s_k,
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
        )

        log2_e = math.log2(math.exp(1.0))
        self._scale_softmax_log2 = self._sm_scale * log2_e
        self._scale_output = 1.0

        cache_variant = (
            self._variant if not isinstance(self._variant, StandardAttention) else None
        )
        params_shape = tuple(self._params_torch.shape) if self._has_params else None

        self._compiled_fmha = _get_compiled_prefill_kernel(
            self._in_dtype,
            self._out_dtype,
            num_qo_heads,
            num_kv_heads,
            self._head_dim,
            self._mask_type,
            window_left,
            self._is_persistent,
            cache_variant,
            params_shape,
        )

        # Pre-allocate padded output scratch buffer.  The kernel uses a
        # negative pointer offset into the output tensor for TMA varlen
        # addressing (see prefill.py __call__, "markus's trick"), so the
        # buffer needs max_s_q extra rows in front.  Allocating once here
        # avoids per-run() allocation overhead across all layers.
        _torch_out_dtype_map = {
            torch.float16: torch.float16,
            torch.bfloat16: torch.bfloat16,
            torch.float8_e4m3fn: torch.float16,
        }
        torch_out_dtype = _torch_out_dtype_map[q_data_type]
        self._o_scratch = torch.empty(
            (self._o_padding + s_q_all, num_qo_heads, self._head_dim),
            dtype=torch_out_dtype,
            device=self._device,
        )
        self._o_scratch_view = self._o_scratch[self._o_padding :]

    def _validate_run_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> None:
        """Check that run() inputs are consistent with the plan() configuration."""
        for name, tensor in [("q", q), ("k", k), ("v", v)]:
            if tensor.dtype != self._q_data_type:
                raise ValueError(
                    f"{name}.dtype={tensor.dtype} does not match the planned "
                    f"q_data_type={self._q_data_type}"
                )
            if tensor.device != self._device:
                raise ValueError(
                    f"{name}.device={tensor.device} does not match the planned "
                    f"device={self._device}"
                )
        if q.shape[-1] != self._head_dim:
            raise ValueError(
                f"q.shape[-1]={q.shape[-1]} does not match the planned "
                f"head_dim={self._head_dim}"
            )
        if q.shape[-2] != self._num_qo_heads:
            raise ValueError(
                f"q.shape[-2]={q.shape[-2]} does not match the planned "
                f"num_qo_heads={self._num_qo_heads}"
            )
        if k.shape[-2] != self._num_kv_heads:
            raise ValueError(
                f"k.shape[-2]={k.shape[-2]} does not match the planned "
                f"num_kv_heads={self._num_kv_heads}"
            )
        if out is not None:
            if out.device != self._device:
                raise ValueError(
                    f"out.device={out.device} does not match the planned "
                    f"device={self._device}"
                )

    @flashinfer_api(trace=cute_dsl_batch_prefill_run_trace)
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
            The query tensor with shape [total_q_len, num_heads, head_dim].
        k : torch.Tensor
            The key tensor with shape [total_kv_len, num_heads, head_dim].
        v : torch.Tensor
            The value tensor with shape [total_kv_len, num_heads, head_dim].
        out : Optional[torch.Tensor], optional
            The output tensor. If None, a new tensor will be created.

        Returns
        -------
        torch.Tensor
            The output tensor with shape [total_q_len, num_heads, head_dim].
        """
        if self._compiled_fmha is None:
            raise RuntimeError("Plan the prefill attention computation first!")

        self._validate_run_inputs(q, k, v, out)

        self._compiled_fmha(
            q,
            k,
            v,
            self._o_scratch_view,
            self._problem_size,
            self._qo_indptr,
            self._s_q_all,
            self._kv_indptr,
            self._s_k_all,
            self._scale_softmax_log2,
            self._scale_output,
            self._params_torch if self._has_params else None,
        )

        if out is not None:
            out.copy_(self._o_scratch_view)
            return out
        return self._o_scratch_view.clone()
