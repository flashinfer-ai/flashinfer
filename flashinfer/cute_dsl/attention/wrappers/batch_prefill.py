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
import os
from typing import Optional

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from flashinfer.api_logging import flashinfer_api

from ...utils import require_cute_dsl_arch as _require_dsl_arch
from flashinfer.trace.templates.attention import cute_dsl_batch_prefill_run_trace

from ..config import AttentionConfig, AttentionFusion
from ..fusion.mask import MaskSpec
from ..fusion.variant import AttentionVariant, StandardAttention
from ..prefill import BlackwellFusedMultiHeadAttentionForward

# V dtypes accepted when v.dtype differs from the planned q/k dtype
# (mixed-dtype PV path: P converts to V's dtype for the PV MMA).
_V_DTYPE_MAP = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


@functools.cache
def _dsl_supports_expected_tx() -> bool:
    """True if the installed DSL supports per-acquire TMA byte-count
    overrides (``PipelineTmaUmma.producer_acquire(expected_tx=...)``,
    added in nvidia-cutlass-dsl 4.6).  Mixed K/V dtypes need it to
    re-arm each V slot of the shared K/V ring with V's byte count.
    """
    import inspect

    from cutlass import pipeline

    return (
        "expected_tx"
        in inspect.signature(pipeline.PipelineTmaUmma.producer_acquire).parameters
    )


@functools.cache
def _get_compiled_prefill_kernel(
    in_dtype,
    out_dtype,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    mask_spec,
    is_persistent,
    variant,
    params_shape,
    with_lse=False,
    v_in_dtype=None,
    page_size=None,
):
    """Compile and cache the prefill kernel.

    Uses symbolic dimensions for sequence lengths and batch size so the same
    compiled kernel can be reused across different batch shapes.  Pass
    ``variant=None`` for standard attention (always cache-hits); pass the
    actual variant instance for custom variants — variants are keyed by
    value via the cache-key protocol on ``AttentionVariant`` (type +
    ``extra_params`` shape/dtype + hashable instance scalars), so fresh
    instances of the same variant config hit the same cache entry.

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
        mask_spec=mask_spec,
        num_repeat_kv_heads=h_r,
        page_size=page_size,
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
    page_table_fake = None
    page_indptr_fake = None
    if page_size is not None:
        # Paged K/V caches: (num_pages, page_size, h_k, d) page-pool views,
        # plus the flat page-id table and its per-item indptr.  Outer
        # strides are fully symbolic (batch_decode.py precedent) so the
        # same kernel handles NHD-compact caches, combined-cache slices
        # (leading-dim stride 2x from kv_cache[:, 0/1]), and
        # HND-transposed views; only head_dim is pinned contiguous (TMA).
        sym_num_pages = cute.sym_int()
        sym_num_page_ids = cute.sym_int()
        k_fake = cute.runtime.make_fake_tensor(
            in_dtype,
            (sym_num_pages, page_size, num_kv_heads, head_dim),
            stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
            assumed_align=16,
        )
        v_fake = cute.runtime.make_fake_tensor(
            v_in_dtype if v_in_dtype is not None else in_dtype,
            (sym_num_pages, page_size, num_kv_heads, head_dim),
            stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
            assumed_align=16,
        )
        page_table_fake = cute.runtime.make_fake_compact_tensor(
            Int32,
            (sym_num_page_ids,),
            assumed_align=4,
        )
        page_indptr_fake = cute.runtime.make_fake_compact_tensor(
            Int32,
            (sym_batch_p1,),
            assumed_align=4,
        )
    else:
        k_fake = cute.runtime.make_fake_compact_tensor(
            in_dtype,
            (sym_s_k, num_kv_heads, head_dim),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        v_fake = cute.runtime.make_fake_compact_tensor(
            v_in_dtype if v_in_dtype is not None else in_dtype,
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

    lse_fake = None
    if with_lse:
        lse_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            (sym_s_q, num_qo_heads),
            stride_order=(1, 0),
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
        0,
        0,
        params_fake,
        lse_fake,
        page_table_fake,
        page_indptr_fake,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


class BatchPrefillCuteDSLWrapper:
    r"""PyTorch-facing wrapper for the CuTe-DSL ragged-KV batch prefill kernel.

    This wrapper exposes a ``plan`` + ``run`` API compatible with
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`, but compiles a
    CuTe-DSL kernel under the hood instead of the C++ FA2/FA3 path.

    Example
    -------

    .. code-block:: python

        wrapper = BatchPrefillCuteDSLWrapper(workspace_buffer)
        wrapper.plan(qo_indptr, kv_indptr,
                     num_qo_heads=32, num_kv_heads=8, head_dim_qk=128)
        out = wrapper.run(q, k, v)
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        r"""Initialise the wrapper and bind it to a workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            Pre-allocated workspace buffer on the target CUDA device.  Named for
            API parity with :class:`BatchPrefillWithRaggedKVCacheWrapper`; callers
            typically pass ``torch.uint8``.  The CuTe-DSL kernel itself does not
            consume this buffer, but it is retained so the wrapper can mirror the
            parent API.
        use_cuda_graph : bool
            Whether the wrapper will be used inside a CUDA graph capture.  Defaults
            to ``False``.
        """
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
        kv_indptr=None,
        num_qo_heads=None,
        num_kv_heads=None,
        head_dim_qk=None,
        head_dim_vo=None,
        causal=True,
        sm_scale=None,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        window_left: int = -1,
        variant: AttentionVariant | None = None,
        window_right: int = -1,
        page_size: Optional[int] = None,
        paged_kv_indptr: Optional[torch.Tensor] = None,
        paged_kv_indices: Optional[torch.Tensor] = None,
        paged_kv_last_page_len: Optional[torch.Tensor] = None,
        kv_layout: str = "NHD",
    ) -> None:
        """Compile the FMHA prefill kernel for the given configuration.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            Cumulative query sequence lengths, shape [batch_size + 1].
        kv_indptr : Optional[torch.Tensor]
            Cumulative KV sequence lengths, shape [batch_size + 1] (ragged
            KV).  Must be None when planning for a paged KV cache — pass
            ``page_size`` + the paged triple instead, and the logical
            per-item KV lengths are derived from them.
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
        sm_scale : Optional[float]
            Softmax scale factor.  Defaults to ``1/sqrt(head_dim_qk)``
            when None (matching the top-level flashinfer wrappers).
        q_data_type : torch.dtype
            Data type for queries (float16, bfloat16, or float8_e4m3fn).
        kv_data_type : torch.dtype
            Data type for keys/values.
        window_left : int
            Max lookback distance: query ``q`` attends to keys
            ``k >= q + (kv_len - qo_len) - window_left``. -1 = unbounded.
            Composes with ``causal`` (causal sliding window attention).
        window_right : int
            Max lookahead distance for non-causal masks: query ``q`` attends
            to keys ``k <= q + (kv_len - qo_len) + window_right``. -1 =
            unbounded. Mutually exclusive with ``causal=True``.
        variant : Optional[AttentionVariant]
            Attention variant (ALiBi, RPE, Sigmoid, etc.). None uses standard softmax.
        page_size : Optional[int]
            Tokens per KV-cache page.  None (default) plans for ragged
            (contiguous varlen) KV; an int plans for a paged KV cache and
            requires the ``paged_kv_*`` triple below.  Must divide the
            128-token KV tile (16/32/64/128; 8 accepted).
        paged_kv_indptr : Optional[torch.Tensor]
            Page-count cumsums per batch item, shape [batch_size + 1], int32.
        paged_kv_indices : Optional[torch.Tensor]
            Flat physical page ids, shape [paged_kv_indptr[-1]], int32.
            Pages referenced here must be finite everywhere (including past
            ``last_page_len`` — the kernel over-reads full pages and relies
            on masking, like every TMA-based paged kernel).  Reclaimed
            out-of-window slots may point at a null block: with
            ``window_left`` set, the kernel never reads pages wholly below
            the attention window.
        paged_kv_last_page_len : Optional[torch.Tensor]
            Valid entries in each item's last page, shape [batch_size].
        kv_layout : str
            ``"NHD"`` (default) or ``"HND"`` page layout.  Paged-only; the
            ragged path remains NHD.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run this example!")
        _require_dsl_arch(qo_indptr.device)

        if num_qo_heads is None or num_kv_heads is None or head_dim_qk is None:
            raise ValueError("num_qo_heads, num_kv_heads and head_dim_qk are required")

        self._page_size = page_size
        self._paged_kv_layout = kv_layout
        self._paged_kv_indptr = None
        self._paged_kv_indices = None
        self._paged_kv_last_page_len = None
        if kv_layout not in ("NHD", "HND"):
            raise ValueError(f"kv_layout must be 'NHD' or 'HND', got {kv_layout!r}")
        if page_size is not None:
            if page_size < 8 or 128 % page_size != 0:
                raise ValueError(
                    f"page_size={page_size} must be >= 8 and divide the "
                    "128-token KV tile (8/16/32/64/128)"
                )
            if (
                paged_kv_indptr is None
                or paged_kv_indices is None
                or paged_kv_last_page_len is None
            ):
                raise ValueError(
                    "paged plan requires paged_kv_indptr, paged_kv_indices "
                    "and paged_kv_last_page_len"
                )
            if kv_indptr is not None:
                raise ValueError(
                    "pass either kv_indptr (ragged) or the paged_kv_* triple, not both"
                )
            from flashinfer.page import get_seq_lens

            device = qo_indptr.device
            seq_lens = get_seq_lens(
                paged_kv_indptr.cpu(), paged_kv_last_page_len.cpu(), page_size
            )
            # Logical token cumsums: the kernel's consumer roles derive
            # per-item seqlen_k from these exactly as on the ragged path.
            kv_indptr = torch.zeros(seq_lens.numel() + 1, dtype=torch.int64)
            kv_indptr[1:] = torch.cumsum(seq_lens, 0)
            kv_indptr = kv_indptr.to(device)
            self._paged_kv_indptr = paged_kv_indptr.to(torch.int32).to(device)
            self._paged_kv_indices = paged_kv_indices.to(torch.int32).to(device)
            self._paged_kv_last_page_len = paged_kv_last_page_len.to(torch.int32).to(
                device
            )
        else:
            if kv_indptr is None:
                raise ValueError("kv_indptr is required for a ragged plan")
            if kv_layout != "NHD":
                raise NotImplementedError("HND layout is only supported with paged KV")

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
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim_qk)
        self._sm_scale = sm_scale
        self._device = qo_indptr.device
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
        # The kernel loads the first KV tile of every valid Q tile
        # unconditionally (all four warp roles assume >= 1 KV tile), so a
        # zero-length KV item would read out of range (paged: page-table
        # index -1) and produce undefined output (empty-row softmax).
        if s_k.numel() > 0 and int(torch.min(s_k).item()) <= 0:
            raise ValueError(
                "cute-dsl prefill requires kv_len >= 1 for every batch item; "
                "zero-length KV items are not supported"
            )

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

        # Mask band: causal and window bounds are independent, composable
        # parameters.
        if self._causal and window_right >= 0:
            raise ValueError(
                "window_right is mutually exclusive with causal "
                "(causal already bounds lookahead at 0)"
            )
        # Causal is a right bound with runtime value 0; window VALUES are
        # runtime kernel arguments (only their presence is compiled in),
        # so one cached kernel serves every window size and the causal /
        # right-window cases share a kernel.
        self._mask_spec = MaskSpec(
            has_window_left=window_left >= 0,
            has_window_right=self._causal or window_right >= 0,
        )
        self._window_left = max(window_left, 0)
        self._window_right = 0 if self._causal else max(window_right, 0)

        # Scheduling: the persistent kernel assigns work items to SMs
        # statically at launch; the non-persistent kernel launches one CTA
        # per item and lets the hardware dispatch dynamically.  Banded
        # masks (causal / windows) run faster non-persistent: their items
        # are short or heterogeneous, and the static assignment's
        # imbalance costs more than the per-CTA prologue it saves.
        # Unmasked items are uniform, so the static schedule is free and
        # the amortized prologue wins.
        # FLASHINFER_CUTE_PREFILL_PERSISTENT=0/1 overrides.
        _persistent_env = os.environ.get("FLASHINFER_CUTE_PREFILL_PERSISTENT")
        if _persistent_env is not None:
            self._is_persistent = _persistent_env == "1"
        else:
            self._is_persistent = not (
                self._mask_spec.has_left_bound or self._mask_spec.has_right_bound
            )

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

        # Stashed so run(return_lse=True) can lazily compile the LSE
        # variant (return_lse is a run-time argument).
        self._compile_key = (
            self._in_dtype,
            self._out_dtype,
            num_qo_heads,
            num_kv_heads,
            self._head_dim,
            self._mask_spec,
            self._is_persistent,
            cache_variant,
            params_shape,
        )
        self._cache_variant = cache_variant
        # page_size joins the compile key only when set, so ragged callers
        # (kwarg-less) and the wrapper share one functools.cache entry.
        self._page_kwargs = (
            {"page_size": self._page_size} if self._page_size is not None else {}
        )
        self._compiled_fmha = _get_compiled_prefill_kernel(
            *self._compile_key, **self._page_kwargs
        )
        # Lazily compiled kernel variants, keyed by (with_lse, v_in_dtype):
        # return_lse and a mixed V dtype are both run()-time properties.
        self._kernel_cache = {(False, None): self._compiled_fmha}

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
        for name, tensor in [("q", q), ("k", k)]:
            if tensor.dtype != self._q_data_type:
                raise ValueError(
                    f"{name}.dtype={tensor.dtype} does not match the planned "
                    f"q_data_type={self._q_data_type}"
                )
        # V may differ from Q/K (mixed-dtype PV path, e.g. fp8 V with
        # bf16 Q/K); the kernel variant is selected per V dtype in run().
        if v.dtype != self._q_data_type and v.dtype not in _V_DTYPE_MAP:
            raise ValueError(
                f"v.dtype={v.dtype} is not supported; expected "
                f"{self._q_data_type} or one of {sorted(map(str, _V_DTYPE_MAP))}"
            )
        for name, tensor in [("q", q), ("k", k), ("v", v)]:
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
        return_lse: bool = False,
        lse: Optional[torch.Tensor] = None,
    ):
        r"""Run the prefill attention computation.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape [total_q_len, num_heads, head_dim].
        k : torch.Tensor
            The key tensor with shape [total_kv_len, num_heads, head_dim].
        v : torch.Tensor
            The value tensor with shape [total_kv_len, num_heads, head_dim].
            May use a different dtype than q/k (e.g. fp8 V with bf16 Q/K);
            the matching kernel variant is JIT-compiled on first use.
        out : Optional[torch.Tensor], optional
            The output tensor. If None, a new tensor will be created.
        return_lse : bool
            Whether to also return the per-row log-sum-exp, shape
            [total_q_len, num_heads], float32, log2 domain (flashinfer
            convention).  Standard attention only.  The LSE kernel variant
            is JIT-compiled on the first such call.
        lse : Optional[torch.Tensor], optional
            Pre-allocated LSE tensor. If None, a new tensor is created.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            The output tensor with shape [total_q_len, num_heads, head_dim],
            plus the LSE tensor when ``return_lse=True``.
        """
        if self._compiled_fmha is None:
            raise RuntimeError("Plan the prefill attention computation first!")
        if self._page_size is not None:
            raise RuntimeError(
                "this wrapper was planned for a paged KV cache; use run_paged()"
            )

        self._validate_run_inputs(q, k, v, out)
        return self._invoke_kernel(q, k, v, out, return_lse, lse, None, None)

    @flashinfer_api
    def run_paged(
        self,
        q: torch.Tensor,
        paged_kv_cache,
        out: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        lse: Optional[torch.Tensor] = None,
    ):
        r"""Run paged-KV prefill attention (requires a paged ``plan()``).

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape ``[total_q_len, num_qo_heads, head_dim]``.
        paged_kv_cache : torch.Tensor or (torch.Tensor, torch.Tensor)
            Either the combined cache ``[num_pages, 2, page_size,
            num_kv_heads, head_dim]`` (NHD) / ``[num_pages, 2, num_kv_heads,
            page_size, head_dim]`` (HND), or a ``(k_cache, v_cache)`` tuple
            of ``[num_pages, page_size, num_kv_heads, head_dim]`` (NHD) /
            ``[num_pages, num_kv_heads, page_size, head_dim]`` (HND).
            Referenced pages must be finite everywhere (see ``plan``).
            The tuple form may carry a V cache whose dtype differs from
            the planned K dtype (mixed-dtype PV path, e.g. an fp8 V cache
            with bf16 Q/K); K must always match the plan.
        out, return_lse, lse
            As in :meth:`run`.
        """
        if self._compiled_fmha is None:
            raise RuntimeError("Plan the prefill attention computation first!")
        if self._page_size is None:
            raise RuntimeError(
                "this wrapper was planned for ragged KV; use run() or re-plan "
                "with page_size + the paged_kv_* triple"
            )

        if isinstance(paged_kv_cache, (tuple, list)):
            k_cache, v_cache = paged_kv_cache
        else:
            if paged_kv_cache.dim() != 5 or paged_kv_cache.shape[1] != 2:
                raise ValueError(
                    "combined paged_kv_cache must be 5D [num_pages, 2, ...]; "
                    f"got shape {tuple(paged_kv_cache.shape)}"
                )
            k_cache, v_cache = paged_kv_cache[:, 0], paged_kv_cache[:, 1]
        if self._paged_kv_layout == "HND":
            # [num_pages, h, page_size, d] -> logical NHD view; the kernel
            # consumes the underlying strides (symbolic-stride compile).
            k_cache = k_cache.transpose(-3, -2)
            v_cache = v_cache.transpose(-3, -2)

        if k_cache.dtype != self._q_data_type:
            raise ValueError(
                f"k_cache.dtype={k_cache.dtype} != planned {self._q_data_type}; "
                "K must match the planned kv dtype (the QK MMA operands share it)"
            )
        # V may differ from Q/K (mixed-dtype PV path, e.g. fp8 V cache with
        # bf16 Q/K); the kernel variant is selected per V dtype in
        # _invoke_kernel, same as the ragged route.  Only the
        # (k_cache, v_cache) tuple form can express a mixed cache.
        if v_cache.dtype != self._q_data_type and v_cache.dtype not in _V_DTYPE_MAP:
            raise ValueError(
                f"v_cache.dtype={v_cache.dtype} is not supported; expected "
                f"{self._q_data_type} or one of {sorted(map(str, _V_DTYPE_MAP))}"
            )
        for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
            if t.shape[-3:] != (
                self._page_size,
                self._num_kv_heads,
                self._head_dim,
            ):
                raise ValueError(
                    f"{name} logical shape {tuple(t.shape)} mismatches plan "
                    f"(page_size={self._page_size}, "
                    f"num_kv_heads={self._num_kv_heads}, "
                    f"head_dim={self._head_dim})"
                )
            if t.stride(-1) != 1:
                raise ValueError(f"{name} head_dim must be contiguous")
        if q.dtype != self._q_data_type or q.device != self._device:
            raise ValueError(
                f"q dtype/device ({q.dtype}, {q.device}) mismatches plan "
                f"({self._q_data_type}, {self._device})"
            )

        if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("", "0"):
            self._validate_paged_cache(k_cache, v_cache)

        return self._invoke_kernel(
            q,
            k_cache,
            v_cache,
            out,
            return_lse,
            lse,
            self._paged_kv_indices,
            self._paged_kv_indptr,
        )

    def _validate_paged_cache(self, k_cache, v_cache) -> None:
        """Debug-mode (FLASHINFER_VALIDATE_INPUTS) scan: every referenced
        page must be finite everywhere.  A NaN in a live page's tail
        corrupts output through the PV MMA (0 x NaN = NaN) — see the paged
        test suite's null-block contract."""
        pages = self._paged_kv_indices.long()
        for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
            ref = t[pages].float()
            if not torch.isfinite(ref).all().item():
                raise ValueError(
                    f"FLASHINFER_VALIDATE_INPUTS: {name} has non-finite values "
                    "in table-referenced pages; live pages must be finite "
                    "everywhere (including past last_page_len)"
                )

    def _invoke_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor],
        return_lse: bool,
        lse: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        page_indptr: Optional[torch.Tensor],
    ):
        if return_lse:
            if self._cache_variant is not None and (
                self._cache_variant.has_logits_transform
                or self._cache_variant.has_vectorized_logits_transform
            ):
                # The transform path has no correction warp (softmax warps
                # own the epilogs) and no softmax normalization — LSE is
                # undefined there.  score_mod and statistics-update
                # variants use the standard path and their finals already
                # reflect the modification (sinks fold into row_sum at
                # tile 0), so LSE is exact for them.
                raise NotImplementedError(
                    "return_lse is not supported with logits-transform variants"
                )
            if lse is None:
                lse = torch.empty(
                    (self._s_q_all, self._num_qo_heads),
                    dtype=torch.float32,
                    device=self._device,
                )
        else:
            lse = None

        v_in_dtype = _V_DTYPE_MAP[v.dtype] if v.dtype != self._q_data_type else None
        if v_in_dtype is not None and not _dsl_supports_expected_tx():
            raise NotImplementedError(
                f"mixed K/V dtypes (V={v.dtype} with planned "
                f"{self._q_data_type}) require nvidia-cutlass-dsl>=4.6 "
                "(producer_acquire lacks per-acquire expected_tx on the "
                "installed version); upgrade nvidia-cutlass-dsl or use a "
                "uniform KV dtype"
            )
        kernel_key = (return_lse, v_in_dtype)
        kernel_fn = self._kernel_cache.get(kernel_key)
        if kernel_fn is None:
            kernel_fn = _get_compiled_prefill_kernel(
                *self._compile_key,
                with_lse=return_lse,
                v_in_dtype=v_in_dtype,
                **self._page_kwargs,
            )
            self._kernel_cache[kernel_key] = kernel_fn

        kernel_fn(
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
            self._window_left,
            self._window_right,
            self._params_torch if self._has_params else None,
            lse,
            page_table,
            page_indptr,
        )

        if out is not None:
            out.copy_(self._o_scratch_view)
        else:
            out = self._o_scratch_view.clone()
        if return_lse:
            return out, lse
        return out
