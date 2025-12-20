"""
Copyright (c) 2023 by FlashInfer team.

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
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch

from .api_logging import flashinfer_api

## NOTE: MLA functions have been moved to mla.py, but we keep the aliases here for backward compatibility.
from .mla import (
    trtllm_batch_decode_with_kv_cache_mla as trtllm_batch_decode_with_kv_cache_mla,
    xqa_batch_decode_with_kv_cache_mla as xqa_batch_decode_with_kv_cache_mla,
)
from .xqa import xqa, xqa_mla as xqa_mla
from .cudnn import cudnn_batch_decode_with_kv_cache as cudnn_batch_decode_with_kv_cache
from .jit import (
    gen_batch_decode_mla_module,
    gen_batch_decode_module,
    gen_customize_batch_decode_module,
    gen_customize_batch_prefill_module,
    gen_single_decode_module,
    get_batch_decode_uri,
    get_batch_prefill_uri,
    get_single_decode_uri,
    setup_cubin_loader,
    gen_trtllm_gen_fmha_module,
)
from .page import get_seq_lens
from .prefill import (
    get_batch_prefill_jit_module,
    get_batch_prefill_module,
    get_single_prefill_module,
)
from .utils import (
    log2e,
    FP4Tensor,
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _get_range_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    device_support_pdl,
    get_device_sm_count,
    is_float8,
    register_custom_op,
    register_fake_op,
    ceil_div,
    round_up,
    get_compute_capability,
    GPUArchitectureError,
)


@functools.cache
def get_single_decode_module(*args):
    uri = get_single_decode_uri(*args)
    module = gen_single_decode_module(*args).build_and_load()
    run_func = module.run

    # torch library for single_decode_with_kv_cache

    @register_custom_op(f"flashinfer::{uri}_run", mutates_args=("tmp", "o"))
    def run_single_decode(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tmp: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        run_func(
            q,
            k,
            v,
            tmp,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            alibi_slopes,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
        )

    @register_fake_op(f"flashinfer::{uri}_run")
    def _fake_run_single_decode(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tmp: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        alibi_slopes: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        pass

    # Register the module.
    return SimpleNamespace(run=run_single_decode)


@functools.cache
def get_batch_decode_jit_module(module_name: str, jit_module: Any):
    plan_func = jit_module.plan
    run_func = jit_module.run

    @register_custom_op(
        f"flashinfer::{module_name}_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        *args,
    ) -> None:
        run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            enable_pdl,
            *args,
        )

    @register_fake_op(f"flashinfer::{module_name}_run")
    def _fake_run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        *args,
    ) -> None:
        pass

    return SimpleNamespace(
        plan=plan_func,
        run=run_batch_decode,
    )


@functools.cache
def get_batch_decode_module(*args):
    uri = get_batch_decode_uri(*args)
    mod = gen_batch_decode_module(*args).build_and_load()
    plan_func = mod.plan
    run_func = mod.run

    # torch library for batch_decode_with_paged_kv_cache_run

    @register_custom_op(
        f"flashinfer::{uri}_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
            "o",
            "maybe_lse",
        ),
    )
    def run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        run_func(
            float_workspace_buffer,
            int_workspace_buffer,
            plan_info_vec,
            q,
            paged_k_cache,
            paged_v_cache,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            o,
            maybe_lse,
            kv_layout_code,
            window_left,
            enable_pdl,
            alibi_slopes,
            logits_soft_cap,
            sm_scale,
            1.0 / rope_scale,  # rope_rcp_scale
            1.0 / rope_theta,  # rope_rcp_theta
        )

    @register_fake_op(f"flashinfer::{uri}_run")
    def _fake_run_batch_decode(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: Optional[torch.Tensor],
        paged_v_cache: Optional[torch.Tensor],
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        kv_layout_code: int,
        window_left: int,
        enable_pdl: bool,
        alibi_slopes: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=plan_func,
        run=run_batch_decode,
    )


@functools.cache
def get_trtllm_gen_fmha_module():
    mod = gen_trtllm_gen_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


@flashinfer_api
def single_decode_with_kv_cache_with_jit_module(
    jit_module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    kv_layout: str = "NHD",
    window_left: int = -1,
    return_lse: bool = False,
):
    device = q.device
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, device)
    o = torch.empty_like(q)
    if return_lse:
        lse = torch.empty((q.size(0)), dtype=torch.float32, device=device)
    else:
        lse = None
    jit_module.run(
        q,
        k,
        v,
        tmp,
        o,
        lse,
        TensorLayout[kv_layout].value,
        window_left,
        *args,
    )
    return o


@functools.cache
def get_batch_decode_mla_module(*args):
    return gen_batch_decode_mla_module(*args).build_and_load()


@overload
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


@flashinfer_api
def single_decode_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_tensor_cores: bool = False,
    q_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Decode attention with KV Cache for single request, return attention output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[num_qo_heads, head_dim]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The value tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if
        :attr:`kv_layout` is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if
        :attr:`kv_layout` is ``HND``.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Defaults to ``NONE``.
    use_tensor_cores: bool
        Whether to use tensor cores for the computation. Will be faster for large group
        size in grouped query attention. Defaults to ``False``.
    q_scale : Optional[float]
        The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
    k_scale : Optional[float]
        The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
    v_scale : Optional[float]
        The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
    window_left : int
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.
    logits_soft_cap : Optional[float]
        The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
        provided, will be set to ``0``. If greater than 0, the logits will be capped according to
        formula:
        :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
        where :math:`x` is the input logits.
    sm_scale : Optional[float]
        The scale of softmax, if not provided, will be set to ``1 / sqrt(head_dim)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to ``1.0``.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to ``1e4``.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[num_qo_heads, head_dim_vo]``.
        * The log sum exp value, shape: ``[num_qo_heads]``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 32
    >>> head_dim = 128
    >>> q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.single_decode_with_kv_cache(q, k, v)
    >>> o.shape
    torch.Size([32, 128])

    Note
    ----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    _check_pos_encoding_mode(pos_encoding_mode)
    _check_kv_layout(kv_layout)
    tmp = _get_cache_buf("single_decode_with_kv_cache_tmp", 32 * 1024 * 1024, q.device)
    head_dim = q.shape[-1]
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    if q_scale is not None:
        sm_scale *= q_scale
    if k_scale is not None:
        sm_scale *= k_scale
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    num_qo_heads = q.shape[0]

    lse = None
    if return_lse:
        lse = torch.empty((num_qo_heads,), dtype=torch.float32, device=q.device)

    if use_tensor_cores:
        out = torch.empty_like(q.unsqueeze(0))
        get_single_prefill_module(
            "fa2",
            q.dtype,
            k.dtype,
            q.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            False,  # use_fp16_qk_reduction
        ).run(
            q.unsqueeze(0),
            k,
            v,
            tmp,
            out,
            lse.unsqueeze(0) if lse is not None else None,
            MaskMode.NON_CAUSAL.value,
            TensorLayout[kv_layout].value,
            window_left,
            None,  # packed_custom_mask
            _get_cache_alibi_slopes_buf(num_qo_heads, q.device),
            logits_soft_cap,
            sm_scale,
            None,  # scale_q, not supported yet
            None,  # scale_k
            None,  # scale_v
            rope_scale,
            rope_theta,
        )
        out = out.squeeze(0)
        if return_lse:
            lse = lse.squeeze(0)
    else:
        out = torch.empty_like(q)
        get_single_decode_module(
            q.dtype,
            k.dtype,
            q.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
        ).run(
            q,
            k,
            v,
            tmp,
            out,
            lse,
            _get_cache_alibi_slopes_buf(num_qo_heads, q.device),
            TensorLayout[kv_layout].value,
            window_left,
            logits_soft_cap,
            sm_scale,
            rope_scale,
            rope_theta,
        )

    if v_scale is not None:
        # TODO(Zihao): fused into kernel
        if out.itemsize == 1:
            out = (out.to(float) * v_scale).to(out.dtype)
        else:
            out *= v_scale
    if return_lse:
        return out, lse
    else:
        return out


class BatchDecodeWithPagedKVCacheWrapper:
    r"""Wrapper class for decode attention with paged kv-cache (first proposed in
    `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> kv_page_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= kv_last_page_len <= page_size
    >>> kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_cache_at_layer = [
    ...     torch.randn(
    ...         max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ...     ) for _ in range(num_layers)
    ... ]
    >>> # create auxiliary data structures for batch decode attention
    >>> decode_wrapper.plan(
    ...     kv_page_indptr,
    ...     kv_page_indices,
    ...     kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     pos_encoding_mode="NONE",
    ...     data_type=torch.float16
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch decode attention, reuse auxiliary data structures for all layers
    ...     o = decode_wrapper.run(q, kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([7, 64, 128])

    Note
    ----
    To accelerate computation, FlashInfer's batch decode attention creates some
    auxiliary data structures, these data structures can be reused across multiple
    batch decode attention calls (e.g. different Transformer layers). This wrapper class
    manages the lifecycle of these data structures.
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor. Must be initialized to 0 for its first use.
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        paged_kv_indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``trtllm-gen``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            if use_tensor_cores:
                self._jit_module = get_batch_prefill_jit_module(
                    jit_args[0],
                    gen_customize_batch_prefill_module(
                        "fa2", *jit_args
                    ).build_and_load(),
                )
            else:
                self._jit_module = get_batch_decode_jit_module(
                    jit_args[0],
                    gen_customize_batch_decode_module(*jit_args).build_and_load(),
                )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._kv_lens_buffer: Optional[torch.Tensor] = None
        if backend == "trtllm-gen":
            self._kv_lens_buffer = torch.empty(
                (32768,), dtype=torch.int32, device=self.device
            )

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._use_tensor_cores = use_tensor_cores or backend == "trtllm-gen"
        self._use_cuda_graph = use_cuda_graph

        if use_tensor_cores:
            if use_cuda_graph:
                # NOTE(Zihao): if once created, no need to update it in plan/run
                self._qo_indptr_buf = torch.arange(
                    self._fixed_batch_size + 1,
                    dtype=torch.int32,
                    device=float_workspace_buffer.device,
                )
        self._backend = backend

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Optional[Union[str, torch.dtype]] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        data_type: Optional[Union[str, torch.dtype]] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: bool = False,
    ) -> None:
        r"""Plan batch decode for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``, dtype: ``torch.int32``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[kv_indptr[-1]]``, dtype: ``torch.int32``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``, dtype: ``torch.int32``
        num_qo_heads : int
            The number of query/output heads
        num_kv_heads : int
            The number of key/value heads
        head_dim : int
            The dimension of the heads
        page_size : int
            The page size of the paged kv cache
        pos_encoding_mode : str
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Defaults to ``NONE``.
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to
            ``q_data_type``. Defaults to ``None``.
        data_type: Optional[Union[str, torch.dtype]]
            The data type of both the query and key/value tensors. Defaults to torch.float16.
            data_type is deprecated, please use q_data_type and kv_data_type instead.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        seq_lens: Optional[torch.Tensor]
            A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``.
        block_tables: Optional[torch.Tensor]
            A uint32 2D tensor indicating the block table of each prompt. shape: ``[batch_size, max_num_blocks_per_seq]``.
        fixed_split_size : Optional[int],
            The fixed split size for FA2 split-kv decode, in pages. Only supported by tensor core decode for now. Recommend setting to the average sequence length of your workload.
            When enabled, will lead to deterministic softmax score reduction in the merge_states kernel, and therefore
            batch-size invariant outputs. See https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
            Note that compatibility with CUDA graph is NOT guaranteed, as even when bs is fixed, kv seq len can change
            and lead to a varied number of launched CTAs.
        disable_split_kv : bool,
            Whether to disable the split-kv for determinism in CUDA Graph, defaults to ``False``.
        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        self._workspace_size = (
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size()
        )

        batch_size = len(last_page_len)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr, non_blocking=non_blocking)
            self._paged_kv_last_page_len_buf.copy_(
                last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(indices)].copy_(
                indices, non_blocking=(indices.device == self.device) and non_blocking
            )
        else:
            self._paged_kv_indptr_buf = indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

        indptr_host = indptr.to("cpu")
        last_page_len_host = last_page_len.to("cpu")

        if data_type is not None:
            if q_data_type is None:
                q_data_type = data_type
            if kv_data_type is None:
                kv_data_type = data_type

        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        if fixed_split_size is not None and not self.use_tensor_cores:
            raise ValueError(
                "fixed_split_size is only supported by tensor core decode for now."
            )
        if fixed_split_size is None:
            fixed_split_size = -1

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        self._batch_size = batch_size
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._block_tables: Optional[torch.Tensor] = block_tables
        self._max_kv_len: Optional[int] = None

        if seq_lens is None:
            kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)
        else:
            kv_lens_arr_host = seq_lens.cpu()
        if self._backend == "trtllm-gen":
            assert logits_soft_cap == 0.0
            self._max_kv_len = max(kv_lens_arr_host).item()
            self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
                kv_lens_arr_host, non_blocking=non_blocking
            )
            if self._block_tables is None:
                blocks_per_seq = [
                    (seq_len + page_size - 1) // page_size
                    for seq_len in kv_lens_arr_host
                ]
                max_num_blocks_per_seq = max(blocks_per_seq)
                self._block_tables = torch.zeros(
                    (batch_size, max_num_blocks_per_seq),
                    dtype=torch.int,
                    device=self.device,
                )
                block_id = indptr[0]
                for i in range(batch_size):
                    num_blocks_needed = blocks_per_seq[i]
                    self._block_tables[i, :num_blocks_needed] = (
                        self._paged_kv_indices_buf[
                            block_id : block_id + num_blocks_needed
                        ]
                    )
                    block_id += num_blocks_needed
            self._cached_module = get_trtllm_gen_decode_module(
                q_data_type,
                kv_data_type,
                q_data_type,
                indptr.dtype,
                head_dim,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                window_left >= 0,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
            )
            self._plan_info = self._cached_module.plan()  # None
        elif self.use_tensor_cores:
            self._max_kv_len = max(kv_lens_arr_host).item()
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_prefill_module(
                    "fa2",
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                    False,  # use_fp16_qk_reduction
                )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                indptr_host,
                kv_lens_arr_host,
                batch_size,  # total_num_rows
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                head_dim,
                head_dim,
                False,  # causal
                window_left,
                fixed_split_size,
                disable_split_kv,
                0,  # num_colocated_ctas
            )
        else:
            if self._jit_module is not None:
                self._cached_module = self._jit_module
            else:
                self._cached_module = get_batch_decode_module(
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    indptr.dtype,
                    head_dim,  # head_dim_qk
                    head_dim,  # head_dim_vo
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left != -1,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                )
            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                indptr_host,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                window_left,
                logits_soft_cap,
                head_dim,
                head_dim,
                torch.empty(0, dtype=q_data_type),
                torch.empty(0, dtype=kv_data_type),
            )

        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: this function is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q, paged_kv_cache, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
        )

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
        q_len_per_req: Optional[int] = 1,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch decode attention between query and paged kv cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[batch_size, num_qo_heads, head_dim]``
        paged_kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            The paged KV-Cache stored as a tuple of tensors or a single tensor:

            * a tuple ``(k_cache, v_cache)`` of 4-D tensors, each with shape:
              ``[max_num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
              and ``[max_num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.

            * a single 5-D tensor with shape:
              ``[max_num_pages, 2, page_size, num_kv_heads, head_dim]`` if
              :attr:`kv_layout` is ``NHD``, and
              ``[max_num_pages, 2, num_kv_heads, page_size, head_dim]`` if
              :attr:`kv_layout` is ``HND``. Where ``paged_kv_cache[:, 0]`` is the key-cache and
              ``paged_kv_cache[:, 1]`` is the value-cache.
        *args
            Additional arguments for the custom kernel.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention scores, defaults to ``False``.
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        q_len_per_req : int
            The number of query tokens per request, if not provided, will be set to ``1``.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * attention output, shape: ``[batch_size, num_qo_heads, head_dim]``
            * logsumexp of attention scores, shape: ``[batch_size, num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)

        if self._kv_layout == "NHD":
            page_size = k_cache.shape[1]
        else:
            page_size = k_cache.shape[2]
        _check_cached_qkv_data_type(
            q, k_cache, self._cached_q_data_type, self._cached_kv_data_type
        )

        # Convert NHD layout to HND for trtllm-gen backend
        if self._backend == "trtllm-gen" and self._kv_layout == "NHD":
            # For NHD: [..., N, H, D] -> HND: [..., H, N, D]
            k_cache = k_cache.transpose(-3, -2)
            v_cache = v_cache.transpose(-3, -2)

        pos_encoding_mode = self._pos_encoding_mode
        window_left = self._window_left if window_left is None else window_left
        if self._backend != "trtllm-gen":
            # NOTE(Siyuan): since window_left is appeared in the plan function, we need to make sure it is the same as the one in the plan function.
            # Remove this check if the backend supports dynamic window_left.
            assert window_left == self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            head_dim = q.shape[-1]
            sm_scale = 1.0 / math.sqrt(head_dim)
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q)
        else:
            check_shape_dtype_device(out, q.shape, q.dtype, q.device, "out")

        if self._backend == "trtllm-gen":
            q = q.view(q.size(0) // q_len_per_req, q_len_per_req, q.size(1), q.size(2))

        if self.use_tensor_cores:
            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k_cache,
                v_cache,
                self._qo_indptr_buf,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                MaskMode.NON_CAUSAL.value,
                TensorLayout[self._kv_layout].value,
                window_left,
                enable_pdl,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    None,  # packed_custom_mask
                    None,  # mask_indptr_buf
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    None,  # maybe_prefix_len_ptr
                    None,  # maybe_token_pos_in_items_ptr
                    None,  # maybe_max_item_len_ptr
                    logits_soft_cap,
                    sm_scale,
                    None,  # scale_q, not supported yet
                    None,  # scale_k
                    None,  # scale_v
                    rope_scale,
                    rope_theta,
                    0,  # token_pos_in_items_len
                    self._workspace_size,
                    paged_kv_cache,
                    self._num_qo_heads,
                    self._num_kv_heads,
                    self._block_tables,
                    self._kv_lens_buffer,
                    page_size,
                    self._max_kv_len,
                    sinks,
                ]

            self._cached_module.paged_run(*run_args)
        else:
            # trtllm-gen does not need plan info
            if self._backend == "trtllm-gen" and self._plan_info is None:
                plan_info: List[int] = []
            else:
                plan_info = self._plan_info
            assert plan_info is not None, "plan info is not initialized"

            run_args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k_cache,
                v_cache,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len_buf,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                window_left,
                enable_pdl,
            ]

            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    logits_soft_cap,
                    sm_scale,
                    rope_scale,
                    rope_theta,
                ]

            self._cached_module.run(*run_args)
        if v_scale is not None:
            # TODO(Zihao): fused into kernel
            if is_float8(out):
                out = (out.to(torch.float32) * v_scale).to(out.dtype)
            else:
                out *= v_scale

        return (out, lse) if return_lse else out

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: this function is deprecated, please use :meth:`run_return_lse` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(
            q,
            paged_kv_cache,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            return_lse=True,
        )

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


class CUDAGraphBatchDecodeWithPagedKVCacheWrapper(BatchDecodeWithPagedKVCacheWrapper):
    r"""CUDAGraph-compatible Wrapper class for decode attention with paged kv-cache (first
    proposed in `vLLM <https://arxiv.org/abs/2309.06180>`_) for batch of requests.

    Note that this wrapper may not be as efficient as :class:`BatchDecodeWithPagedKVCacheWrapper`
    because we won't dispatch to different kernels for different batch sizes/sequence lengths/etc
    to accommodate the CUDAGraph requirement.

    Check :ref:`our tutorial<kv-layout>` for page table layout.

    Note
    ----
    The :meth:`plan` method could not be captured by CUDAGraph.

    See Also
    --------
    :class:`BatchDecodeWithPagedKVCacheWrapper`
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        indptr_buffer: torch.Tensor,
        indices_buffer: torch.Tensor,
        last_page_len_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_tensor_cores: bool = False,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        workspace_buffer : torch.Tensor
            The user reserved workspace buffer on GPU used to store auxiliary data structures,
            recommended size is 128MB, the device of the workspace buffer should be the
            same as the device of the input tensors.

        indptr_buffer : torch.Tensor
            The user reserved buffer on GPU to store the indptr of the paged kv cache, should
            be large enough to store the indptr of maximum batch size (``[max_batch_size + 1]``)
            during the lifecycle of this wrapper.

        indices_buffer : torch.Tensor
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.

        last_page_len_buffer : torch.Tensor
            The user reserved buffer on GPU to store the number of entries in the last page,
            should be large enough to store the maximum batch size (``[max_batch_size]``)
            during the lifecycle of this wrapper.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        """
        super().__init__(
            workspace_buffer,
            kv_layout,
            use_cuda_graph=True,
            use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=indptr_buffer,
            paged_kv_indices_buffer=indices_buffer,
            paged_kv_last_page_len_buffer=last_page_len_buffer,
        )


class BatchDecodeMlaWithPagedKVCacheWrapper:
    r"""Warning: this class is deprecated and will be removed in a future release.
    Please use :class:`flashinfer.mla.BatchMLAPagedAttentionWrapper` instead, which provides
    a more efficient and general MLA implementation that supports decode and incremental prefill.
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        use_tensor_cores: bool = False,
        paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
        paged_kv_indices_buffer: Optional[torch.Tensor] = None,
        paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Constructor of :class:`BatchDecodeWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        use_cuda_graph : bool
            Whether to enable CUDAGraph for batch decode attention, if enabled, the
            auxiliary data structures will be stored as the provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        use_tensor_cores : bool
            Whether to use tensor cores for the computation. Will be faster for large group
            size in grouped query attention. Defaults to ``False``.

        paged_kv_indptr_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the indptr of the paged kv cache, the size
            of the buffer should be ``[batch_size + 1]``.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the page indices of the paged kv cache,
            should be large enough to store the maximum number of page indices
            (``max_num_pages``) during the lifecycle of this wrapper.
            Only needed when ``use_cuda_graph`` is ``True``.

        paged_kv_last_page_len_buffer : Optional[torch.Tensor]
            The user reserved buffer on GPU to store the number of entries in the last page, the
            size of the buffer should be ``[batch_size]``.
            Only needed when ``use_cuda_graph`` is ``True``.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,),
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )

        if use_cuda_graph:
            if not torch.is_tensor(paged_kv_indptr_buffer):
                raise ValueError(
                    "paged_kv_indptr_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buffer):
                raise ValueError(
                    "paged_kv_indices_buffer should be a torch.Tensor in cudagraph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buffer):
                raise ValueError(
                    "paged_kv_last_page_len_buffer should be a torch.Tensor in cudagraph mode"
                )
            self._fixed_batch_size = len(paged_kv_last_page_len_buffer)
            if len(paged_kv_indptr_buffer) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The size of paged_kv_indptr_buffer should be batch_size + 1"
                )
        else:
            self._fixed_batch_size = 0

        self._use_tensor_cores = use_tensor_cores
        self._paged_kv_indptr_buf = paged_kv_indptr_buffer
        self._paged_kv_indices_buf = paged_kv_indices_buffer
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buffer
        self._use_cuda_graph = use_cuda_graph

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    @property
    def use_tensor_cores(self) -> bool:
        return self._use_tensor_cores

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer: torch.Tensor
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        head_dim_compressed_kv: int,
        page_size: int,
        sm_scale: float,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        data_type: Union[str, torch.dtype] = "float16",
        q_data_type: Optional[Union[str, torch.dtype]] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> None:
        r"""Plan batch decode for given problem specification.

        Parameters
        ----------
        indptr : torch.Tensor
            The indptr of the paged kv cache, shape: ``[batch_size + 1]``, dtype: ``torch.int32``
        indices : torch.Tensor
            The page indices of the paged kv cache, shape: ``[qo_indptr[-1]]``, dtype: ``torch.int32``
        last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged kv
            cache, shape: ``[batch_size]``, dtype: ``torch.int32``
        num_qo_heads : int
            The number of query/output heads
        head_dim_compressed_kv : int
            The dimension of the compressed kv, is also kv_lora_rank
        page_size : int
            The page size of the paged kv cache
        sm_scale : float
            The scale of softmax, should be ``1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)``
        window_left : int
            The left (inclusive) window size for the attention window, when set to ``-1``, the window
            size will be set to the full length of the sequence. Defaults to ``-1``.
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        data_type : Union[str, torch.dtype]
            The data type of the paged kv cache. Defaults to ``float16``.
        q_data_type : Optional[Union[str, torch.dtype]]
            The data type of the query tensor. If None, will be set to
            ``data_type``. Defaults to ``None``.

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple run calls.
        """
        batch_size = len(last_page_len)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        if self.is_cuda_graph_enabled:
            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The size of indices should be less than or equal to the allocated buffer"
                )
            self._paged_kv_indptr_buf.copy_(indptr)
            self._paged_kv_indices_buf[: len(indices)] = indices
            self._paged_kv_last_page_len_buf.copy_(last_page_len)
        else:
            self._paged_kv_indptr_buf = indptr.to(self.device)
            self._paged_kv_indices_buf = indices.to(self.device)
            self._paged_kv_last_page_len_buf = last_page_len.to(self.device)

        data_type = canonicalize_torch_dtype(data_type)
        if not q_data_type:
            q_data_type = data_type
        q_data_type = canonicalize_torch_dtype(q_data_type)

        indptr_host = indptr.to("cpu")

        self._cached_module = get_batch_decode_mla_module(
            q_data_type,
            data_type,
            q_data_type,
            indptr.dtype,
            head_dim_compressed_kv,
            num_qo_heads,
            window_left != -1,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            self._use_tensor_cores,
        )
        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            indptr_host,
            batch_size,
            num_qo_heads,
            page_size,
            self.is_cuda_graph_enabled,
        )

        self._sm_scale = sm_scale
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    @flashinfer_api
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        paged_ckv_cache: torch.Tensor,
        paged_kpe_cache: torch.Tensor,
        q_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: bool = False,  # fake placeholder (sm80)
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch decode attention between query and paged kv cache.

        Parameters
        ----------
        q_nope : torch.Tensor
            The query tensor not related to ROPE, shape: ``[batch_size, num_qo_heads, head_dim_ckv]``
        q_pe : torch.Tensor
            The query tensor related to ROPE, shape: ``[batch_size, num_qo_heads, head_dim_kpe]``
        paged_ckv_cache : torch.Tensor
            The paged compressed-KV-Cache stored as a single tensor:
            * 3-D tensors, each with shape: ``[max_num_pages, page_size, head_dim_ckv]``.
        paged_kpe_cache : torch.Tensor
            The paged k-pe-Cache stored as a single tensor:
            * 3-D tensors, each with shape: ``[max_num_pages, page_size, head_dim_kpe]``.
        q_scale : Optional[float]
            The calibration scale of query for fp8 input, if not provided, will be set to ``1.0``.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention scores, defaults to ``False``.
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[batch_size, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * attention output, shape: ``[batch_size, num_qo_heads, head_dim]``
            * logsumexp of attention scores, shape: ``[batch_size, num_qo_heads]``.
        """

        # MLA decode kernel supports SM80 only
        major, minor = get_compute_capability(q_nope.device)
        device_arch = major * 10 + minor
        if device_arch != 80:
            raise GPUArchitectureError(
                f"MLA decode kernel is not supported on this GPU (SM{device_arch}). "
                "Supported architecture: SM80."
            )
        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if q_scale is not None:
            sm_scale *= q_scale
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        device = self.device
        if out is None:
            out = torch.empty_like(q_nope, device=device)
        else:
            check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q_nope.size(0), q_nope.size(1)),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                check_shape_dtype_device(
                    lse,
                    (q_nope.size(0), q_nope.size(1)),
                    q_nope.dtype,
                    q_nope.device,
                    "lse",
                )
        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            paged_ckv_cache,
            paged_kpe_cache,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len_buf,
            out,
            sm_scale,
            window_left,
            logits_soft_cap,
            rope_scale,
            rope_theta,
            lse,
            enable_pdl,
        )
        out = [out, lse] if return_lse else [out]
        if v_scale is not None:
            out[0] *= v_scale

        return tuple(out) if return_lse else out[0]

    run_return_lse = functools.partialmethod(run, return_lse=True)


class TrtllmGenDecodeModule:
    def __init__(self) -> None:
        self._sm_count: Optional[int] = None
        self._mod = gen_trtllm_gen_fmha_module()
        self._op = self._mod.build_and_load()
        from flashinfer.jit.cubin_loader import setup_cubin_loader

        setup_cubin_loader(self._mod.get_library_path())

    def _paged_run(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        bmm1_scale: Union[float, torch.Tensor],
        bmm2_scale: Union[float, torch.Tensor],
        workspace_size: int,
        window_left: int = -1,
        enable_pdl: bool = None,
        out: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(query)
        if self._sm_count is None:
            self._sm_count = get_device_sm_count(query.device)

        if isinstance(bmm1_scale, torch.Tensor):
            assert bmm1_scale.dtype == torch.float32
            bmm1_scale = bmm1_scale * log2e
        if isinstance(bmm2_scale, torch.Tensor):
            assert bmm2_scale.dtype == torch.float32

        assert len(query.size()) == 4
        batch_size = query.size(0)
        max_q_len = query.size(1)
        query = query.flatten(0, 1)  # [B*S, H, D]

        self._op.trtllm_paged_attention_decode(
            out,
            None,  # fp4 output not supported in wrapper api yet.
            query,  # [B * S, H, D], w/ MTP here so S dim is > 1
            k_cache,
            v_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            max_seq_len,
            bmm1_scale,
            bmm2_scale,
            -1,  # o_sf_scale
            -1,  # o_sf_vec_size
            0,  # o_sf_start_index
            batch_size,
            window_left,
            0,  # sparse_mla_top_k
            self._sm_count,
            enable_pdl,
            workspace_size,
            sinks,
            None,  # cum_seq_lens_q
        )
        return out

    def _plan(self, *args, **kwargs):
        pass


@functools.cache
def get_trtllm_gen_decode_module(*args):
    uri = get_batch_prefill_uri("trtllm-gen", *args)
    module = TrtllmGenDecodeModule()

    @register_custom_op(
        f"flashinfer::{uri}_ragged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "o",
            "maybe_lse",
        ),
    )
    def paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        scale_q: Optional[torch.Tensor],
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
        workspace_size: int,
        paged_kv_cache: Optional[torch.Tensor] = None,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_kv_len: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        assert maybe_lse is None
        assert paged_kv_cache is not None
        assert num_qo_heads is not None
        assert num_kv_heads is not None
        assert block_tables is not None
        assert kv_lens_buffer is not None
        assert page_size is not None
        assert max_kv_len is not None
        assert enable_pdl is not None
        assert workspace_size > 0, "workspace_size must be greater than 0"
        o = module._paged_run(
            q.contiguous(),  # NOTE(Siyuan): without contiguous, the result is incorrect
            paged_k_cache,
            paged_v_cache,
            float_workspace_buffer,
            block_tables,
            kv_lens_buffer,
            max_kv_len,
            sm_scale,
            1.0,  # NOTE(Siyuan): update this to expose bmm2 scale
            workspace_size,
            window_left,
            enable_pdl,
            out=o,
            sinks=sinks,
        )

    @register_fake_op(f"flashinfer::{uri}_paged_run")
    def _fake_paged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        enable_pdl: bool,
        maybe_custom_mask: Optional[torch.Tensor],
        maybe_mask_indptr: Optional[torch.Tensor],
        maybe_alibi_slopes: Optional[torch.Tensor],
        maybe_prefix_len_ptr: Optional[torch.Tensor],
        maybe_token_pos_in_items_ptr: Optional[torch.Tensor],
        maybe_max_item_len_ptr: Optional[torch.Tensor],
        logits_soft_cap: float,
        sm_scale: float,
        rope_scale: float,
        rope_theta: float,
        token_pos_in_items_len: int,
        paged_kv_cache: Optional[torch.Tensor] = None,
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_kv_len: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=module._plan,
        paged_run=paged_run,
    )


@flashinfer_api
def trtllm_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    window_left: int = -1,
    out: Optional[Union[torch.Tensor, FP4Tensor]] = None,
    out_dtype: Optional[Union[torch.dtype, str]] = None,
    o_sf_scale: Optional[float] = None,
    o_sf_vec_size: Optional[int] = None,
    sinks: Optional[List[torch.Tensor]] = None,
    kv_layout: str = "HND",
    enable_pdl: Optional[bool] = None,
    backend: str = "auto",
    q_len_per_req: Optional[int] = 1,
    o_scale: Optional[float] = 1.0,
    mask: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, FP4Tensor]:
    """
    Parameters
    ----------
    query : torch.Tensor
        query tensor with shape [num_tokens, num_heads, head_dim], num_tokens = total query tokens in the batch.

    kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If kv_cache is a single tensor, it should be a tensor with shape [num_pages, 1 or 2, num_kv_heads, page_size, head_dim] if :attr:`kv_layout` is ``HND``,
        or [num_pages, 1 or 2, page_size, num_kv_heads, head_dim] if :attr:`kv_layout` is ``NHD``.
        If kv_cache is a tuple of two tensors, it should be a tuple of two tensors with shape [num_pages, num_kv_heads, page_size, head_dim] if :attr:`kv_layout` is ``HND``,
        or [num_pages, page_size, num_kv_heads, head_dim] if :attr:`kv_layout` is ``NHD``.
        The first tensor is the key cache, and the second tensor is the value cache.

    workspace_buffer : torch.Tensor. Must be initialized to 0 for its first use.
        workspace

    block_tables : torch.Tensor
        page_table of kv cache, [batch_size, num_pages]

    seq_lens : torch.Tensor
        A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``

    max_seq_len : int
        max sequence length for kv_cache

    bmm1_scale : Union[float, torch.Tensor]
        fused scale for bmm1 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.

    bmm2_scale : Union[float, torch.Tensor]
        fused scale for bmm2 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.

    window_left : int = -1
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.

    out :  Optional[Union[torch.Tensor, FP4Tensor]] = None
        output tensor, if not provided, will be allocated with ``out_dtype``, if ``out_dtype`` is not provided, will use the type of ``query``.

    out_dtype : Optional[Union[torch.dtype, str]] = None
        output dtype, if not provided, will use the type of ``out``. For nvfp4, use string ``nvfp4``.

    o_sf_scale : Optional[float] = None
        scale for nvfp4 output tensor scale factor.

    o_sf_vec_size : Optional[int] = None
        vector size for nvfp4 output tensor scale factor.

    sinks : Optional[List[torch.Tensor]] = None
        additional value per head in the denominator of the softmax.

    kv_layout : str = "HND"
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
        Defaults to ``HND``.

    enable_pdl : Optional[bool] = None
        Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
        When set to ``None``, the backend will be chosen based on the device architecture and kernel availability.

    backend : str = "auto"
        The implementation backend, could be ``auto``/``xqa`` or ``trtllm-gen``. Defaults to ``auto``.
        When set to ``auto``, the backend will be chosen based on the device architecture and kernel availability.
        For sm_100 and sm_103 (blackwell architecture), ``auto`` will choose ``trtllm-gen`` backend.
        For sm_90 (hopper architecture) and sm_120 (blackwell architecture), ``auto`` will choose ``xqa`` backend.

    o_scale : Optional[float] = 1.0
        output scale factor for xqa fp8 output.

    mask : Optional[torch.Tensor] = None
        causal attention mask for xqa speculative decoding.

    max_q_len: Optional[int] = None
        The maximum query sequence length across all requests when using variable-length queries.
        Only supported by trtllm-gen backend. Must be provided together with ``cum_seq_lens_q``.
        When None, all requests use uniform query length specified by ``q_len_per_req``.

    cum_seq_lens_q : Optional[torch.Tensor] = None
        Cumulative query sequence lengths for variable-length query support, shape: ``[batch_size + 1]``, dtype: ``torch.int32``.
        Only supported by trtllm-gen backend. Must be provided together with ``max_q_len``.
        When None, all requests use uniform query length specified by ``q_len_per_req``.

    Returns
    -------
    out : Union[torch.Tensor, FP4Tensor]
        output torch.Tensor or FP4Tensor.
    """
    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl

    if isinstance(kv_cache, tuple):
        k_cache, v_cache = kv_cache
    else:
        if kv_cache.shape[1] == 1:
            k_cache, v_cache = kv_cache, kv_cache
        else:
            assert kv_cache.shape[1] == 2, (
                "When kv_cache is a single tensor, the second dimension must be 1 or 2"
            )
            # NOTE(Zihao): unbind transforms [num_pages, 2, ...] to ([num_pages, ...], [num_pages, ...])
            # it doesn't change underlying storage
            k_cache, v_cache = kv_cache.unbind(dim=1)

    if backend == "auto":
        backend = (
            "trtllm-gen" if get_compute_capability(query.device)[0] == 10 else "xqa"
        )

    if backend == "xqa":
        # xqa backend doesn't support nvfp4 output
        if out_dtype == "nvfp4" or (out_dtype is None and isinstance(out, FP4Tensor)):
            raise ValueError("xqa backend does not support nvfp4 output")
        if o_sf_scale is not None or o_sf_vec_size is not None:
            raise ValueError("xqa backend does not support o_sf_scale or o_sf_vec_size")
        if max_q_len is not None or cum_seq_lens_q is not None:
            raise ValueError("xqa backend does not support cum_seq_lens_q")

        # Handle out and out_dtype
        if out_dtype is None:
            out_dtype = out.dtype if out is not None else query.dtype
        if out is None:
            out = torch.empty_like(query, dtype=out_dtype)

        # Call xqa_batch_decode_with_kv_cache
        return xqa_batch_decode_with_kv_cache(
            query=query,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=workspace_buffer,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=window_left,
            out=out,
            sinks=sinks,
            kv_layout=kv_layout,
            enable_pdl=enable_pdl,
            q_len_per_req=q_len_per_req,
            o_scale=o_scale,
            mask=mask,
        )
    elif backend == "trtllm-gen":
        # Convert NHD layout to HND if necessary (transpose only changes stride, not data)
        if kv_layout == "NHD":
            # For NHD: [..., N, H, D] -> HND: [..., H, N, D]
            k_cache = k_cache.transpose(-3, -2)
            v_cache = v_cache.transpose(-3, -2)

        run_func = get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
        sm_count = get_device_sm_count(query.device)

        if out_dtype == "nvfp4" or (out_dtype is None and isinstance(out, FP4Tensor)):
            assert query.dtype == torch.float8_e4m3fn, (
                "query must be fp8 when out_dtype is nvfp4."
            )
            assert o_sf_scale is not None
            assert o_sf_vec_size in [None, 16], "only o_sf_vec_size = 16 is supported"
            o_sf_vec_size = o_sf_vec_size or 16

            fp4_out_shape = query.shape[:-1] + (ceil_div(query.shape[-1], 2),)

            if isinstance(out, FP4Tensor):
                fp4_out_scale_shape = (
                    out.scale.shape[0],
                    round_up(query.shape[1] * query.shape[2] // o_sf_vec_size, 4),
                )
                out_scale_factor = out.scale
                o_sf_start_index = out.scale_start_index
                out = out.data
                # out_dtype may be None
                out_dtype = out_dtype or "nvfp4"
            elif out is None:
                fp4_out_scale_shape = (
                    round_up(query.shape[0], 128),
                    round_up(query.shape[1] * query.shape[2] // o_sf_vec_size, 4),
                )
                out_scale_factor = torch.empty(
                    fp4_out_scale_shape, dtype=torch.float8_e4m3fn, device=query.device
                )
                o_sf_start_index = 0
                out = torch.empty(fp4_out_shape, dtype=torch.uint8, device=query.device)
            else:
                raise ValueError(f"Invalid out: {out}")

            assert out_dtype == "nvfp4"
            assert isinstance(out, torch.Tensor)

            # Use uint8 as the container dtype to compliant with next fp4 gemm.
            check_shape_dtype_device(
                out, fp4_out_shape, torch.uint8, query.device, "out"
            )

            check_shape_dtype_device(
                out_scale_factor,
                fp4_out_scale_shape,
                torch.float8_e4m3fn,
                query.device,
                "out_scale_factor",
            )

            # Check o_sf_start_index is valid
            if (
                o_sf_start_index < 0
                or o_sf_start_index + out.shape[0] > out_scale_factor.shape[0]
            ):
                raise ValueError(
                    f"o_sf_start_index is out of the valid range of out_scale_factor. "
                    f"o_sf_start_index={o_sf_start_index}, out.shape[0]={out.shape[0]}, "
                    f"out_scale_factor.shape[0]={out_scale_factor.shape[0]}"
                )

        elif isinstance(out_dtype, torch.dtype) or out_dtype is None:
            assert o_sf_scale is None
            assert o_sf_vec_size is None
            out_scale_factor = None
            o_sf_start_index = 0
            if out_dtype is None:
                out_dtype = out.dtype if out is not None else query.dtype
            out = out if out is not None else torch.empty_like(query, dtype=out_dtype)
            if out_dtype not in (query.dtype, torch.float16, torch.bfloat16):
                raise ValueError(f"Unsupported out_dtype: {out_dtype}")
            check_shape_dtype_device(out, query.shape, out_dtype, query.device, "out")
        else:
            raise ValueError(f"Invalid out_dtype: {out_dtype}")

        if isinstance(bmm1_scale, torch.Tensor):
            assert bmm1_scale.dtype == torch.float32
            bmm1_scale = bmm1_scale * log2e
        if isinstance(bmm2_scale, torch.Tensor):
            assert bmm2_scale.dtype == torch.float32

        if q_len_per_req is not None:
            max_q_len = q_len_per_req
            batch_size = query.size(0) // q_len_per_req
        else:
            assert max_q_len is not None
            batch_size = cum_seq_lens_q.size(0) - 1

        run_func(
            out,
            out_scale_factor,
            query,
            k_cache,
            v_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            max_seq_len,
            bmm1_scale,
            bmm2_scale,
            o_sf_scale or -1.0,
            o_sf_vec_size or -1,
            o_sf_start_index,
            batch_size,
            window_left,
            0,  # sparse_mla_top_k
            sm_count,
            enable_pdl,
            workspace_buffer.numel() * workspace_buffer.element_size(),
            sinks,
            cum_seq_lens_q,
        )

        return (
            out
            if out_dtype != "nvfp4"
            else FP4Tensor(out, out_scale_factor, o_sf_start_index, query.shape)
        )
    else:
        raise KeyError(f"Backend {backend} not supported")


# xqa uses NHD layout
@flashinfer_api
def xqa_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    window_left: int = -1,
    out: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    kv_layout: str = "NHD",
    enable_pdl: bool = None,
    q_len_per_req: Optional[int] = 1,
    o_scale: Optional[float] = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Parameters
    ----------
    query : torch.Tensor
        query tensor with shape [num_tokens, num_heads, head_dim], num_tokens = batch_size * q_len_per_request

    kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If kv_cache is a single tensor, it should be a tensor with shape [num_pages, 1 or 2, page_size, num_kv_heads, head_dim] if :attr:`kv_layout` is ``NHD``,
        or [num_pages, 1 or 2, num_kv_heads, page_size, head_dim] if :attr:`kv_layout` is ``HND``.
        If kv_cache is a tuple of two tensors, it should be a tuple of two tensors with shape [num_pages, page_size, num_kv_heads, head_dim] if :attr:`kv_layout` is ``NHD``,
        or [num_pages, num_kv_heads, page_size, head_dim] if :attr:`kv_layout` is ``HND``.

    workspace_buffer : torch.Tensor. Must be initialized to 0 for its first use.
        workspace

    block_tables : torch.Tensor
        page_table of kv cache, [batch_size, num_pages]

    seq_lens : torch.Tensor
        A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``

    max_seq_len : int
        max sequence length for kv_cache

    bmm1_scale : Union[float, torch.Tensor]
        fused scale for bmm1 input.

    bmm2_scale : Union[float, torch.Tensor]
        fused scale for bmm2 input.

    window_left : int = -1
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.

    out :  Optional[torch.Tensor] = None
        output tensor, if not provided, will be allocated with ``query.dtype``.

    sinks : Optional[torch.Tensor] = None
        additional value per head in the denominator of the softmax.

    kv_layout : str
        The layout of the kv cache. Can be either ``NHD`` or ``HND``. Defaults to ``NHD``.

    enable_pdl : bool
        Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
        Only supported for >= sm90, and currently only for FA2, CUDA core, and trtllm-gen decode.

    o_scale : Optional[float] = 1.0
        output scale factor for fp8 output.

    mask : Optional[torch.Tensor] = None
        causal attention mask for xqa speculative decoding.

    Returns
    -------
    out : torch.Tensor
        output torch.Tensor.
    """
    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl

    if isinstance(kv_cache, tuple):
        k_cache, v_cache = kv_cache
    else:
        if kv_cache.shape[1] == 1:
            k_cache, v_cache = kv_cache, kv_cache
        else:
            assert kv_cache.shape[1] == 2, (
                "When kv_cache is a single tensor, the second dimension must be 1 or 2"
            )
            # NOTE(Zihao): unbind transforms [num_pages, 2, ...] to ([num_pages, ...], [num_pages, ...])
            # it doesn't change underlying storage
            k_cache, v_cache = kv_cache.unbind(dim=1)

    sm_count = get_device_sm_count(query.device)

    # Extract shape parameters based on layout
    if kv_layout == "NHD":
        # NHD: [num_pages, page_size, num_kv_heads, head_dim]
        page_size = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
    else:  # HND
        # HND: [num_pages, num_kv_heads, page_size, head_dim]
        num_kv_heads = k_cache.shape[1]
        page_size = k_cache.shape[2]
        head_dim = k_cache.shape[3]

    workspace_u8 = workspace_buffer.view(torch.uint8)
    semaphore = workspace_u8[: 8 * 1024 * 1024]  # reserve 8MB for semaphore
    scratch = workspace_u8[8 * 1024 * 1024 :]
    kv_scale_value = bmm2_scale * o_scale
    q_scale_value = bmm1_scale / kv_scale_value * (head_dim**0.5)

    if q_len_per_req > 1:
        batch_size = query.shape[0] // q_len_per_req
        query = query.view(batch_size, q_len_per_req, query.shape[1], query.shape[2])
    query_new = query.unsqueeze(1)
    seq_lens_new = seq_lens.unsqueeze(1)
    sinks_new = sinks.reshape(num_kv_heads, -1) if sinks is not None else None

    # Ensure 4D output for xqa
    if out is None:
        out = torch.empty_like(query)
    out_4d = out.unsqueeze(1)

    xqa(
        query_new,
        k_cache,
        v_cache,
        block_tables,
        seq_lens_new,
        out_4d,
        scratch,
        semaphore,
        num_kv_heads,
        page_size,
        sinks=sinks_new,
        q_scale=q_scale_value,
        kv_scale=kv_scale_value,
        sliding_win_size=window_left + 1 if window_left >= 0 else 0,
        kv_layout=kv_layout,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
        rcp_out_scale=1.0 / o_scale,
        q_seq_len=q_len_per_req,
        mask=mask,
    )

    return out


def fast_decode_plan(
    self,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    q_data_type: Optional[Union[str, torch.dtype]] = None,
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    data_type: Optional[Union[str, torch.dtype]] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    disable_split_kv: bool = False,
    global_override_indptr_cpu: Optional[torch.Tensor] = None,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for FlashInferMultiStepDraftBackend.
    Modifications:
    - Remove unnecessary device-to-device copy for the cuda graph buffers.
    - Remove unnecessary host-to-device copy for the metadata buffers.
    """
    batch_size = len(last_page_len)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0

    # Handle data types consistently
    if data_type is not None:
        if q_data_type is None:
            q_data_type = data_type
        if kv_data_type is None:
            kv_data_type = data_type
    elif q_data_type is None:
        q_data_type = "float16"

    if kv_data_type is None:
        kv_data_type = q_data_type

    if self.use_tensor_cores:
        qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")
        # Here we set fixed_split_size to -1 to avoid the assertion error in flashinfer's plan function
        if fixed_split_size is None:
            fixed_split_size = -1

    if self.is_cuda_graph_enabled:
        if batch_size != self._fixed_batch_size:
            raise ValueError(
                "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                " mismatches the batch size set during initialization {}".format(
                    batch_size, self._fixed_batch_size
                )
            )
        if len(indices) > len(self._paged_kv_indices_buf):
            raise ValueError(
                "The size of indices should be less than or equal to the allocated buffer"
            )
    else:
        self._paged_kv_indptr_buf = indptr
        self._paged_kv_indices_buf = indices
        self._paged_kv_last_page_len_buf = last_page_len
        if self.use_tensor_cores:
            self._qo_indptr_buf = qo_indptr_host.to(
                self.device, non_blocking=non_blocking
            )

    # Create empty tensors for dtype info if needed
    empty_q_data = torch.empty(
        0,
        dtype=(
            getattr(torch, q_data_type) if isinstance(q_data_type, str) else q_data_type
        ),
        device=self.device,
    )

    empty_kv_cache = torch.empty(
        0,
        dtype=(
            getattr(torch, kv_data_type)
            if isinstance(kv_data_type, str)
            else kv_data_type
        ),
        device=self.device,
    )

    indptr_host = (
        global_override_indptr_cpu
        if global_override_indptr_cpu is not None
        else indptr.cpu()
    )

    with torch.cuda.device(self.device):
        if self.use_tensor_cores:
            # ALSO convert last_page_len to CPU
            if page_size == 1:
                # When page size is 1, last_page_len is always 1.
                # Directly construct the host tensor rather than executing a device-to-host copy.
                last_page_len_host = torch.ones(
                    (batch_size,), dtype=torch.int32, device="cpu"
                )
            else:
                last_page_len_host = last_page_len.cpu()

            kv_lens_arr_host = get_seq_lens(indptr_host, last_page_len_host, page_size)

            try:
                # Make sure we pass exactly 16 arguments for tensor core version
                self._plan_info = self._cached_module.plan(
                    self._float_workspace_buffer,
                    self._int_workspace_buffer,
                    self._pin_memory_int_workspace_buffer,
                    qo_indptr_host,
                    indptr_host,
                    kv_lens_arr_host,
                    batch_size,  # total_num_rows
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    page_size,
                    self.is_cuda_graph_enabled,
                    head_dim,
                    head_dim,
                    False,  # causal
                    window_left,
                    fixed_split_size,
                    disable_split_kv,
                    0,  # num_colocated_ctas
                )
            except Exception as e:
                raise RuntimeError(f"Error in standard plan: {e}") from e
        else:
            try:
                # Make sure we pass exactly 15 arguments for standard version
                self._plan_info = self._cached_module.plan(
                    self._float_workspace_buffer,
                    self._int_workspace_buffer,
                    self._pin_memory_int_workspace_buffer,
                    indptr_host,
                    batch_size,
                    num_qo_heads,
                    num_kv_heads,
                    page_size,
                    self.is_cuda_graph_enabled,
                    window_left,
                    logits_soft_cap,
                    head_dim,
                    head_dim,
                    empty_q_data,
                    empty_kv_cache,
                )
            except Exception as e:
                raise RuntimeError(f"Error in standard plan: {e}") from e

    self._pos_encoding_mode = pos_encoding_mode
    self._window_left = window_left
    self._logits_soft_cap = logits_soft_cap
    self._sm_scale = sm_scale
    self._rope_scale = rope_scale
    self._rope_theta = rope_theta
