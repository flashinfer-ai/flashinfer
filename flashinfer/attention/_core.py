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

import functools
import math
from typing import Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit import gen_batch_attention_module
from ..trace.templates.attention import batch_attention_run_trace
from ..utils import (
    MaskMode,
    canonicalize_torch_dtype,
    PosEncodingMode,
    TensorLayout,
    _check_kv_layout,
    _unpack_paged_kv_cache,
    determine_attention_backend,
)
from ..prefill import BatchPrefillWithPagedKVCacheWrapper
from ..jit.attention.variants import attention_sink_decl
from ..jit.attention.modules import (
    batch_prefill_bidirectional_ranges_jit_args,
    get_batch_prefill_bidirectional_ranges_spec,
)
from ..jit.utils import filename_safe_dtype_map


@functools.cache
def get_holistic_attention_module(*args):
    return gen_batch_attention_module(*args).build_and_load()


class BatchAttention:
    r"""Holistic batched attention wrapper that fuses paged-prefill and paged-decode requests
    into a single kernel launch.

    ``BatchAttention`` dispatches between prefill-style and decode-style execution per
    request based on the ``qo_indptr`` / ``kv_indptr`` ranges supplied to :meth:`plan`, so a
    serving stack can submit a mixed batch (e.g. some prompts in prefill, others in decode)
    without splitting it into two separate wrappers.  Workspace buffers are owned by the
    instance and reused across :meth:`plan` / :meth:`run` calls.

    Parameters
    ----------
    kv_layout : str
        Layout of the paged KV-cache tensors, either ``"NHD"`` (token-major) or
        ``"HND"`` (head-major).  Defaults to ``"NHD"``.
    device : str
        CUDA device that owns the internal workspace buffers, e.g. ``"cuda"`` or
        ``"cuda:0"``.  Defaults to ``"cuda"``.
    """

    @flashinfer_api
    def __init__(
        self,
        kv_layout: str = "NHD",
        device: str = "cuda",
    ):
        r"""Allocate workspace buffers and bind the wrapper to a CUDA device.

        See :class:`BatchAttention` for the meaning of each parameter.
        """
        _check_kv_layout(kv_layout)
        self._kv_layout = kv_layout

        self.float_workspace_buffer = torch.empty(
            384 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device(device),
        )
        self.page_locked_int_workspace_buffer = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device=torch.device("cpu"),
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        causal: bool = False,
        sm_scale: float = None,
        logits_soft_cap: Optional[float] = None,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        use_profiler: bool = False,
    ) -> None:
        r"""Plan the holistic attention kernel for a specific batch shape.

        Should be called before any :meth:`run` call.  The plan is cached on the
        instance and reused across subsequent :meth:`run` invocations with the same
        layout.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            CSR-style query offsets, shape ``[batch_size + 1]``, dtype ``int32``.
        kv_indptr : torch.Tensor
            CSR-style page offsets into ``kv_indices``, shape ``[batch_size + 1]``,
            dtype ``int32``.
        kv_indices : torch.Tensor
            Page indices into the paged KV-cache, shape ``[kv_indptr[-1]]``,
            dtype ``int32``.
        kv_len_arr : torch.Tensor
            Per-request KV-cache lengths in tokens, shape ``[batch_size]``,
            dtype ``int32``.
        num_qo_heads : int
            Number of query / output heads.
        num_kv_heads : int
            Number of key / value heads.  Must divide ``num_qo_heads``.
        head_dim_qk : int
            Per-head dimension of the query / key tensors.
        head_dim_vo : int
            Per-head dimension of the value / output tensors.
        page_size : int
            Page size of the paged KV-cache.
        causal : bool
            Whether to apply a causal mask.  Defaults to ``False``.
        sm_scale : float
            Softmax scale.  If ``None``, defaults to ``1/sqrt(head_dim_qk)``.
        logits_soft_cap : Optional[float]
            Logits soft-cap value.  ``None`` or ``0`` disables capping.
        q_data_type : torch.dtype
            Dtype of the query tensor.  Defaults to ``torch.bfloat16``.
        kv_data_type : torch.dtype
            Dtype of the key / value tensors.  Defaults to ``torch.bfloat16``.
        use_profiler : bool
            Whether to compile the profiler-enabled variant of the kernel.  Defaults
            to ``False``.
        """
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        self._logits_soft_cap = logits_soft_cap

        # head_dim > 256 is for holistic (persistent) kernel.
        if head_dim_qk > 256 or head_dim_vo > 256:
            raise ValueError(
                "BatchAttention (holistic persistent kernel) does not support "
                f"head_dim > 256 (got head_dim_qk={head_dim_qk}, "
                f"head_dim_vo={head_dim_vo}). Use "
                "BatchPrefillWithPagedKVCacheWrapper(backend='fa2') or "
                "BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True) instead."
            )

        # get jit module
        get_module_args = (
            q_data_type,
            kv_data_type,
            q_data_type,
            kv_indptr.dtype,
            head_dim_qk,
            head_dim_vo,
            PosEncodingMode["NONE"].value,
            logits_soft_cap > 0.0,
            use_profiler,  # different compiler path
        )
        self.module = get_holistic_attention_module(*get_module_args)

        qo_indptr_host = qo_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_indptr_host = kv_indptr.to(torch.device("cpu"), non_blocking=True)
        kv_len_arr_host = kv_len_arr.to(torch.device("cpu"), non_blocking=True)
        torch.cuda.synchronize()

        batch_size = kv_len_arr.shape[0]
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._page_size = page_size
        self._use_profiler = use_profiler

        # No addtional buf allocated for CUDA graph tensor
        # Allocate outside FlashInfer
        self._kv_indices = kv_indices
        self._plan_info = self.module.plan(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self.page_locked_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim_vo,
            causal,
        )

    @flashinfer_api(trace=batch_attention_run_trace)
    def run(
        self,
        q: torch.Tensor,
        kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        logits_soft_cap: float = 0.0,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_cache_sf: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Run the planned holistic attention kernel.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape ``[total_qo_tokens, num_qo_heads, head_dim_qk]``.
        kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Either a single packed paged KV-cache tensor (when K and V share storage) or a
            ``(k_cache, v_cache)`` pair.  Layout must match the ``kv_layout`` passed to
            :meth:`__init__`.
        out : Optional[torch.Tensor]
            Optional output buffer.  If ``None``, a new tensor is allocated with the same
            shape as ``q``.
        lse : Optional[torch.Tensor]
            Optional log-sum-exp buffer, shape ``[total_qo_tokens, num_qo_heads]``, dtype
            ``float32``.  Allocated if ``None``.
        k_scale : Optional[torch.Tensor]
            FP8 dequantization scale for ``k``.  Pre-multiplied into ``sm_scale``.
        v_scale : Optional[torch.Tensor]
            FP8 dequantization scale for ``v``.  Applied to the output.
        logits_soft_cap : float
            Logits soft-cap value.  Must be consistent with the ``logits_soft_cap``
            passed to :meth:`plan` (a non-zero value here requires a non-zero plan-time
            value too).
        profiler_buffer : Optional[torch.Tensor]
            Profiler buffer.  Required if the wrapper was planned with
            ``use_profiler=True``.
        kv_cache_sf : Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
            Optional scale tensors for NVFP4 KV-cache (one tensor or a ``(k_sf, v_sf)``
            pair, mirroring the structure of ``kv_cache``).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``(out, lse)`` — the attention output and its log-sum-exp.
        """
        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        if logits_soft_cap > 0.0 and self._logits_soft_cap <= 0.0:
            raise ValueError(
                "logits_soft_cap used in kernel run but not provided in plan(). This will cause template deduction error."
            )

        k_cache, v_cache = _unpack_paged_kv_cache(kv_cache, self._kv_layout)
        if out is None:
            out = torch.empty_like(q)
        if lse is None:
            # lse shape: [batch_size, num_qo_heads]
            lse = torch.empty(
                q.shape[0], q.shape[1], device=q.device, dtype=torch.float32
            )
        head_dim_qk = q.shape[2]
        sm_scale = self._sm_scale
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(head_dim_qk)
        if k_scale is not None:
            sm_scale *= k_scale
        if v_scale is None:
            v_scale = 1.0
        # profiler_buffer is optional
        profiler_args = (profiler_buffer,) if self._use_profiler else ()

        # Unpack kv_cache_sf for NVFP4 (maybe_k_cache_sf, maybe_v_cache_sf)
        k_cache_sf, v_cache_sf = (
            _unpack_paged_kv_cache(kv_cache_sf, self._kv_layout)
            if kv_cache_sf is not None
            else (None, None)
        )

        self.module.run(
            self.float_workspace_buffer,
            self.int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._kv_indices,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            self._num_qo_heads,
            self._num_kv_heads,
            self._page_size,
            v_scale,
            sm_scale,
            logits_soft_cap,
            # ADDITIONAL_FUNC_PARAMS (maybe_k_cache_sf, maybe_v_cache_sf)
            k_cache_sf,
            v_cache_sf,
            # PROFILER_FUNC_PARAMS
            *profiler_args,
        )

        return out, lse


class BatchAttentionWithAttentionSinkWrapper(BatchPrefillWithPagedKVCacheWrapper):
    r"""
    Wrapper for prefill and decode attention with paged KV-cache that adds support for
    attention sinks. This class extends `BatchPrefillWithPagedKVCacheWrapper`, providing
    a convenient interface for using attention sinks during prefill or decode attention.
    """

    # No @flashinfer_api here: parent class BatchPrefillWithPagedKVCacheWrapper
    # already decorates __init__, so decorating again produces double log entries.
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        head_dim_qk: int = 128,
        head_dim_vo: int = 128,
        window_left: int = -1,
    ) -> None:
        # trtllm is separate code path
        assert backend in ["fa2", "fa3", "auto"]
        if backend == "auto":
            # dispatch backend before init jit module
            backend = determine_attention_backend(
                float_workspace_buffer.device,
                PosEncodingMode[pos_encoding_mode].value,
                use_fp16_qk_reduction,  # use_fp16_qk_reduction
                custom_mask_buf is not None,  # use_custom_mask
                q_data_type,
                kv_data_type,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
            )

        jit_args = [
            f"batch_prefill_attention_sink_{filename_safe_dtype_map[q_data_type]}_swa_{window_left >= 0}_{backend}",  # uri
            q_data_type,  # dtype_q
            kv_data_type,  # dtype_kv
            q_data_type,  # dtype_o
            torch.int32,  # idtype
            head_dim_qk,  # hidden_dim_qk
            head_dim_vo,  # hidden_dim_vo
            ["sink"],  # additional_tensor_names
            ["float"],  # additional_tensor_dtypes
            ["sm_scale"],  # additional_scalar_names
            ["double"],  # additional_scalar_dtypes
            "AttentionSink",
            attention_sink_decl[backend],
        ]
        jit_kwargs = {
            "use_sliding_window": window_left >= 0,
            "use_fp16_qk_reduction": use_fp16_qk_reduction,
            "pos_encoding_mode": PosEncodingMode[pos_encoding_mode].value,
        }

        super().__init__(
            float_workspace_buffer=float_workspace_buffer,
            kv_layout=kv_layout,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buf,
            paged_kv_indptr_buf=paged_kv_indptr_buf,
            paged_kv_indices_buf=paged_kv_indices_buf,
            paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
            custom_mask_buf=custom_mask_buf,
            mask_indptr_buf=mask_indptr_buf,
            backend=backend,
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )


class BatchPrefillWithCausalBidirectionalRangesWrapper(
    BatchPrefillWithPagedKVCacheWrapper
):
    r"""Paged batch prefill with a causal mask plus per-query bidirectional ranges.

    The mask this wrapper applies is

    .. code-block:: text

        (causal AND causal_window) OR (in_range AND range_window)

    where ``in_range`` is decided per query token from an inclusive
    ``[start, end]`` key span supplied by the caller. Tokens inside their own
    span see each other in both directions; everything else stays causal. This
    is the shape a prefix-style batch produces, but the wrapper knows nothing
    about what the spans mean.

    The mask is never materialized. A custom fa2 attention variant owns the
    whole expression and evaluates it on every KV tile from the compact range
    tensor, so nothing here grows with ``qo_len * kv_len``.

    Ranges are passed to :meth:`run` as a contiguous ``int32`` tensor of shape
    ``[total_q, 2]`` on the query device, one row per scheduled query token in
    the same order as the query tensor. Row ``i`` is the inclusive
    ``[start, end]`` span of query ``i`` in absolute key positions within its
    request; ``(-1, -1)`` marks a query with no span, which then keeps the
    plain causal mask.

    The wrapper is fa2 only, because fa2 is the only backend whose kernels
    evaluate a custom ``LogitsMask`` per KV tile. It is not related to, and
    cannot be combined with, ``prefix_len_ptr`` (multi-item scoring), which
    selects a different mask mode and owns the mask itself.

    The JIT module is specialized in the constructor on the query, KV and
    output dtypes, both head dimensions and the positional-encoding mode.
    :meth:`plan` and :meth:`workspace_size` reject any value that differs from
    what was compiled, and reject the inherited options that would otherwise be
    accepted and silently ignored.

    Example
    -------
    >>> import torch, flashinfer
    >>> workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper(
    ...     workspace, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    ...     head_dim_qk=128, head_dim_vo=128,
    ... )  # doctest: +SKIP
    """

    # No @flashinfer_api on the overrides here: the parent class already
    # decorates __init__, plan, workspace_size and run.
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indptr_buf: Optional[torch.Tensor] = None,
        paged_kv_indices_buf: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buf: Optional[torch.Tensor] = None,
        pos_encoding_mode: str = "NONE",
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: torch.dtype = torch.bfloat16,
        o_data_type: Optional[torch.dtype] = None,
        head_dim_qk: int = 128,
        head_dim_vo: int = 128,
    ) -> None:
        if pos_encoding_mode != "NONE":
            # POS_ENCODING_MODE is a compile-time constant of the customize
            # config, and a rotary mode makes the kernel read rope parameters
            # that this variant does not declare as additional scalars.
            raise NotImplementedError(
                "pos_encoding_mode must be 'NONE' for this wrapper: the variant "
                "declares no rope scalars, so any other mode would not compile. "
                f"Got {pos_encoding_mode!r}."
            )
        q_data_type = canonicalize_torch_dtype(q_data_type)
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        o_data_type = (
            q_data_type
            if o_data_type is None
            else canonicalize_torch_dtype(o_data_type)
        )

        self._spec = get_batch_prefill_bidirectional_ranges_spec(
            q_data_type,
            kv_data_type,
            o_data_type,
            torch.int32,
            head_dim_qk,
            head_dim_vo,
        )

        super().__init__(
            float_workspace_buffer=float_workspace_buffer,
            kv_layout=kv_layout,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buf,
            paged_kv_indptr_buf=paged_kv_indptr_buf,
            paged_kv_indices_buf=paged_kv_indices_buf,
            paged_kv_last_page_len_buf=paged_kv_last_page_len_buf,
            backend=self._spec["backend"],
            jit_args=batch_prefill_bidirectional_ranges_jit_args(self._spec),
            jit_kwargs=self._spec["jit_kwargs"],
            variant_owns_mask=True,
        )

    def _check_specialization(
        self,
        head_dim_qk: int,
        head_dim_vo: Optional[int],
        pos_encoding_mode: str,
        q_data_type: Optional[Union[str, torch.dtype]],
        kv_data_type: Optional[Union[str, torch.dtype]],
        o_data_type: Optional[Union[str, torch.dtype]],
        custom_mask: Optional[torch.Tensor],
        packed_custom_mask: Optional[torch.Tensor],
        causal: bool,
        use_fp16_qk_reduction: bool,
        window_left: int,
        logits_soft_cap: Optional[float],
        prefix_len_ptr: Optional[torch.Tensor],
        token_pos_in_items_ptr: Optional[torch.Tensor],
        token_pos_in_items_len: int,
        max_item_len_ptr: Optional[torch.Tensor],
    ) -> Tuple[torch.dtype, torch.dtype, torch.dtype, int]:
        """Validate one ``plan``/``workspace_size`` call against the built module.

        An omitted dtype or ``head_dim_vo`` is taken from the specialization the
        constructor compiled, not from a generic default, so a caller never has
        to restate what it already said once. Only a value that was actually
        passed is compared, and a mismatch is refused rather than quietly
        planned for a kernel that was never built.

        Returns the ``(q, kv, o)`` dtypes and ``head_dim_vo`` of the built
        module, so both callers hand the parent exactly what it was compiled
        for.
        """
        spec = self._spec
        for name, value, key in (
            ("q_data_type", q_data_type, "dtype_q"),
            ("kv_data_type", kv_data_type, "dtype_kv"),
            ("o_data_type", o_data_type, "dtype_o"),
            ("head_dim_qk", head_dim_qk, "head_dim_qk"),
            ("head_dim_vo", head_dim_vo, "head_dim_vo"),
        ):
            if value is None:
                continue
            got = (
                value if key.startswith("head_dim") else canonicalize_torch_dtype(value)
            )
            if got != spec[key]:
                raise ValueError(
                    f"{name}={got} does not match the value this wrapper was "
                    f"constructed with ({spec[key]}). The JIT module is "
                    "specialized in the constructor; build a second wrapper for "
                    "a second configuration."
                )
        q_data_type = spec["dtype_q"]
        kv_data_type = spec["dtype_kv"]
        o_data_type = spec["dtype_o"]
        head_dim_vo = spec["head_dim_vo"]
        if (
            PosEncodingMode[pos_encoding_mode].value
            != spec["jit_kwargs"]["pos_encoding_mode"]
        ):
            raise ValueError(
                f"pos_encoding_mode={pos_encoding_mode!r} does not match the "
                "value this wrapper was constructed with."
            )

        # Inherited options that this wrapper would otherwise accept and then
        # ignore, because the variant owns the mask or because the module was
        # not compiled for them.
        for name, value in (
            ("custom_mask", custom_mask),
            ("packed_custom_mask", packed_custom_mask),
            ("prefix_len_ptr", prefix_len_ptr),
            ("token_pos_in_items_ptr", token_pos_in_items_ptr),
            ("max_item_len_ptr", max_item_len_ptr),
        ):
            if value is not None:
                raise ValueError(
                    f"{name} is not supported: the attention variant owns the "
                    "whole mask, so a caller-supplied mask would be ignored. "
                    "Pass the spans through run(bidirectional_ranges=...)."
                )
        if token_pos_in_items_len != 0:
            raise ValueError(
                "token_pos_in_items_len is not supported: multi-item scoring "
                "owns the mask itself and cannot be combined with this variant."
            )
        if causal:
            raise ValueError(
                "causal=True is not supported: causality is already part of "
                "this variant's mask and the flag would be ignored. Use "
                "run(causal_window_left=...) to bound the causal part."
            )
        if use_fp16_qk_reduction:
            raise ValueError(
                "use_fp16_qk_reduction=True is not supported: the module was "
                "compiled with fp16 QK reduction off."
            )
        if window_left != -1:
            raise ValueError(
                "window_left is not supported here: the variant applies its own "
                "windows. Use run(causal_window_left=..., range_window_left=...), "
                "which bound the causal and bidirectional parts separately."
            )
        if logits_soft_cap:
            raise ValueError(
                "logits_soft_cap is not supported: the module was compiled with "
                "the soft-cap path off."
            )
        return q_data_type, kv_data_type, o_data_type, head_dim_vo

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = None,
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> None:
        r"""Plan a batch whose mask this variant owns.

        The signature mirrors the parent wrapper so that a caller can swap the
        two, but every argument that the variant makes meaningless is rejected
        instead of ignored, and the dtypes and head dimensions must match what
        the constructor compiled. Backend-specific planning arguments of the
        parent (``seq_lens``, ``block_tables``, ``rope_scale``, ...) are absent
        here: this wrapper is fa2 only and applies no positional encoding.
        """
        q_data_type, kv_data_type, o_data_type, head_dim_vo = (
            self._check_specialization(
                head_dim_qk,
                head_dim_vo,
                pos_encoding_mode,
                q_data_type,
                kv_data_type,
                o_data_type,
                custom_mask,
                packed_custom_mask,
                causal,
                use_fp16_qk_reduction,
                window_left,
                logits_soft_cap,
                prefix_len_ptr,
                token_pos_in_items_ptr,
                token_pos_in_items_len,
                max_item_len_ptr,
            )
        )
        return super().plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            head_dim_vo=head_dim_vo,
            causal=False,
            pos_encoding_mode=pos_encoding_mode,
            sm_scale=sm_scale,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            non_blocking=non_blocking,
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def workspace_size(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        head_dim_vo: Optional[int] = None,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = None,
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> Tuple[int, int]:
        r"""Caller-owned workspace size for a :meth:`plan` with these arguments.

        Validates exactly what :meth:`plan` validates, so a caller cannot size a
        workspace for a configuration that :meth:`plan` would then reject.
        """
        q_data_type, kv_data_type, o_data_type, head_dim_vo = (
            self._check_specialization(
                head_dim_qk,
                head_dim_vo,
                pos_encoding_mode,
                q_data_type,
                kv_data_type,
                o_data_type,
                custom_mask,
                packed_custom_mask,
                causal,
                use_fp16_qk_reduction,
                window_left,
                logits_soft_cap,
                prefix_len_ptr,
                token_pos_in_items_ptr,
                token_pos_in_items_len,
                max_item_len_ptr,
            )
        )
        return super().workspace_size(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            page_size,
            head_dim_vo=head_dim_vo,
            causal=False,
            pos_encoding_mode=pos_encoding_mode,
            sm_scale=sm_scale,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            max_token_per_sequence=max_token_per_sequence,
            max_sequence_kv=max_sequence_kv,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        bidirectional_ranges: torch.Tensor,
        causal_window_left: int = -1,
        range_window_left: int = -1,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        kv_cache_sf: Optional[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run causal + bidirectional-ranges attention.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape ``[total_q, num_qo_heads, head_dim_qk]``.
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV cache, as accepted by the parent wrapper.
        bidirectional_ranges : torch.Tensor
            Contiguous ``int32`` tensor of shape ``[total_q, 2]`` on the query
            device. Row ``i`` is the inclusive ``[start, end]`` key span that
            query ``i`` attends bidirectionally, or ``(-1, -1)`` for none. A
            contiguous row slice of a larger ``[N, 2]`` buffer is accepted; a
            non-contiguous tensor is rejected rather than copied, so the kernel
            argument stays a stable pointer under CUDA graph capture.
        causal_window_left : int
            Sliding window on the causal part. A key is kept when
            ``q_abs - kv < causal_window_left``. ``-1`` (or ``0``) disables the
            window, leaving the causal part unbounded.
        range_window_left : int
            Sliding window on the bidirectional part. ``-1`` (or ``0``) leaves
            the spans unclamped, which is what a span normally wants; set it to
            bound how far back a span may reach.
        q_scale, k_scale : Optional[float]
            Scalar calibration scales folded into ``sm_scale``. Per-head scale
            tensors are rejected: this variant declares a single ``double``
            ``sm_scale``, so a tensor could not be folded into it.
        v_scale : Optional[float]
            Scalar calibration scale of the value cache. It never reaches the
            variant: the parent rescales the output once the kernel has
            returned. Only a scalar is exposed here, so the public contract of
            this wrapper stays a single output rescaling; per-head tensors are
            rejected.
        out : Optional[torch.Tensor]
            Output tensor; allocated internally when omitted. Its dtype must be
            the ``o_data_type`` this wrapper was constructed with.
        lse : Optional[torch.Tensor]
            Log-sum-exp tensor, allocated internally when omitted.
        return_lse : bool
            Whether to return the log-sum-exp alongside the output.
        enable_pdl : Optional[bool]
            Programmatic dependent launch, as in the parent wrapper.
        kv_cache_sf : Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
            Scale factors for a packed NVFP4 KV cache. Required when the
            wrapper was constructed with a packed fp4 ``kv_data_type``, and
            meaningless otherwise.

        Notes
        -----
        ``window_left``, ``sinks`` and the parent's remaining keyword arguments
        are deliberately absent. Each of them either contradicts the mask this
        variant owns or falls outside the contract this specialized wrapper
        offers, so it is not exposed rather than accepted and then ignored.
        """
        if bidirectional_ranges.dtype != torch.int32:
            raise ValueError(
                f"bidirectional_ranges must be int32, got {bidirectional_ranges.dtype}"
            )
        if bidirectional_ranges.dim() != 2 or bidirectional_ranges.size(-1) != 2:
            raise ValueError(
                "bidirectional_ranges must have shape [total_q, 2], got "
                f"{tuple(bidirectional_ranges.shape)}"
            )
        if bidirectional_ranges.size(0) != q.size(0):
            raise ValueError(
                "bidirectional_ranges must have one row per query token: got "
                f"{bidirectional_ranges.size(0)} rows for {q.size(0)} queries"
            )
        if bidirectional_ranges.device != q.device:
            raise ValueError(
                "bidirectional_ranges must live on the query device, got "
                f"{bidirectional_ranges.device} and {q.device}"
            )
        if not bidirectional_ranges.is_contiguous():
            raise ValueError(
                "bidirectional_ranges must be contiguous. Copying it here would "
                "hand the kernel a fresh pointer on every call, which a captured "
                "CUDA graph would then replay against freed memory; call "
                ".contiguous() at the call site if that is what you want."
            )
        for name, value in (("q_scale", q_scale), ("k_scale", k_scale)):
            if isinstance(value, torch.Tensor):
                raise ValueError(
                    f"{name} must be a scalar for this wrapper: the variant "
                    "declares a single sm_scale double, so a per-head tensor "
                    "cannot be folded into it."
                )
        if isinstance(v_scale, torch.Tensor):
            raise ValueError(
                "v_scale must be a scalar for this wrapper: its public contract "
                "is a single scalar rescaling of the output, not a per-head one."
            )
        if q.dtype != self._spec["dtype_q"]:
            raise ValueError(
                f"q has dtype {q.dtype}, but this wrapper was built for "
                f"{self._spec['dtype_q']}."
            )

        # The kernel indexes the rows flat; the 2-D shape is the public
        # contract, and this view keeps the caller's pointer.
        ranges_flat = bidirectional_ranges.view(-1)
        # Window semantics are "keep when distance < N", so a disabled window is
        # 0 rather than -1.
        causal_n = float(max(causal_window_left, 0))
        range_n = float(max(range_window_left, 0))
        return super().run(
            q,
            paged_kv_cache,
            ranges_flat,
            causal_n,
            range_n,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out=out,
            lse=lse,
            return_lse=return_lse,
            enable_pdl=enable_pdl,
            kv_cache_sf=kv_cache_sf,
        )
