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
import logging
import math
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import torch
from .jit import (
    gen_batch_prefill_module,
    gen_customize_batch_prefill_module,
    gen_single_prefill_module,
    get_pod_uri,
    has_prebuilt_ops,
    prebuilt_ops_uri,
)
from .page import block_sparse_indices_to_vector_sparse_offsets, get_seq_lens
from .quantization import packbits, segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    determine_attention_backend,
    get_cuda_stream,
    is_float8,
    register_custom_op,
    register_fake_op,
)

_pod_modules = {}

def get_pod_module(backend):
    def backend_module(*args):
        global _pod_modules
        modules_dict = (
            _pod_modules
        )
        if args not in modules_dict:
            uri = get_pod_uri(backend, *args)
            
            from . import _kernels

            run_func = _kernels.pod_with_kv_cache
            # torch library for pod_with_kv_cache

            @register_custom_op(
                f"flashinfer::{uri}_run", mutates_args=("tmp", "o", "maybe_lse")
            )
            def run_pod(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                tmp: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_packed_custom_mask: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
            ) -> None:
                with q.device as device:  # device guard
                    if backend == "fa3":
                        run_func(
                            q,
                            k,
                            v,
                            tmp,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            logits_soft_cap,
                            sm_scale,
                            get_cuda_stream(device),
                        )
                    else:
                        run_func(
                            q,
                            k,
                            v,
                            tmp,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            maybe_packed_custom_mask,
                            maybe_alibi_slopes,
                            logits_soft_cap,
                            sm_scale,
                            1.0 / rope_scale,  # rope_rcp_scale
                            1.0 / rope_theta,  # rope_rcp_theta
                            get_cuda_stream(device),
                        )
                    return o

            @register_fake_op(f"flashinfer::{uri}_run")
            def _fake_run_pod(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                tmp: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_packed_custom_mask: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
            ) -> None:
                pass

            # Register the module
            modules_dict[args] = SimpleNamespace(run=run_pod)

        return modules_dict[args]

    return backend_module

def pod_with_kv_cache_with_jit_module(
    jit_module: Any,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *args,
    kv_layout: str = "NHD",
    mask_mode: int = MaskMode.NON_CAUSAL.value,
    window_left: int = -1,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    with q.device as device:  # device guard
        tmp = _get_cache_buf(
            "pod_with_kv_cache_tmp", 32 * 1024 * 1024, device=device
        )
        o = torch.empty_like(q)
        lse = None
        if return_lse:
            lse = torch.empty(
                (q.size(0), q.size(1)), dtype=torch.float32, device=device
            )
        jit_module.run(
            q,
            k,
            v,
            tmp,
            o,
            lse,
            mask_mode,
            TensorLayout[kv_layout].value,
            window_left,
            *args,
            get_cuda_stream(device),
        )
        return (o, lse) if return_lse else o


@overload
def pod_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def pod_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def pod_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    custom_mask: Optional[torch.Tensor] = None,
    packed_custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    kv_layout: str = "NHD",
    pos_encoding_mode: str = "NONE",
    use_fp16_qk_reduction: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    return_lse: bool = False,
    backend: str = "auto",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Prefill/Append attention with KV cache for single request, return the attention
    output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[qo_len, num_qo_heads, head_dim]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim]`` if :attr:`kv_layout`
        is ``NHD``, ``[num_kv_heads, kv_len, head_dim]`` if :attr:`kv_layout` is
        ``HND``.
    custom_mask : Optional[torch.Tensor]
        The custom boolean mask tensor, shape: ``[qo_len, kv_len]``.
        The elements in the mask tensor should be either ``True`` or ``False``,
        where ``False`` means the corresponding element in the attention matrix will be
        masked out.

        When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
        function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
        additional overhead.
    packed_custom_mask : Optional[torch.Tensor]
        The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
        The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
    causal : bool
        Whether to apply causal mask to the attention matrix.
        This is only effective when :attr:`custom_mask` is not provided.
    kv_layout : str
        The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.
    pos_encoding_mode : str
        The position encoding applied inside attention kernels, could be
        ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
        Default is ``NONE``.
    use_fp16_qk_reduction : bool
        Whether to use f16 for qk reduction (faster at the cost of slight precision
        loss).
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
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.
    backend : str
        The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
        If set to ``auto``, the function will automatically choose the backend based on the
        device architecture and kernel availability.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[qo_len, num_qo_heads, head_dim]``.
        * The log sum exp value, shape: ``[qo_len, num_qo_heads]``.

    Examples
    --------

    >>> import torch
    >>> import flashinfer
    >>> qo_len = 128
    >>> kv_len = 4096
    >>> num_qo_heads = 32
    >>> num_kv_heads = 4
    >>> head_dim = 128
    >>> q = torch.randn(qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    >>> o = flashinfer.pod_with_kv_cache(q, k, v, causal=True,
            use_fp16_qk_reduction=True)
    >>> o.shape
    torch.Size([128, 32, 128])
    >>> mask = torch.tril(
    >>>     torch.full((qo_len, kv_len), True, device="cuda:0"),
    >>>     diagonal=(kv_len - qo_len),
    >>> )
    >>> mask
    tensor([[ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            [ True,  True,  True,  ..., False, False, False],
            ...,
            [ True,  True,  True,  ...,  True, False, False],
            [ True,  True,  True,  ...,  True,  True, False],
            [ True,  True,  True,  ...,  True,  True,  True]], device='cuda:0')
    >>> o_custom = flashinfer.pod_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_custom, rtol=1e-3, atol=1e-3)
    True

    Note
    ----
    The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads`` is
    not equal to ``num_kv_heads``, the function will use
    `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
    """
    _check_pos_encoding_mode(pos_encoding_mode)
    _check_kv_layout(kv_layout)
    tmp = _get_cache_buf("pod_with_kv_cache_tmp", 32 * 1024 * 1024, q.device)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    if rope_scale is None:
        rope_scale = 1.0
    if rope_theta is None:
        rope_theta = 1e4
    if custom_mask is not None and packed_custom_mask is None:
        # create packed custom mask from custom mask
        packed_custom_mask = packbits(
            custom_mask.contiguous().view(-1), bitorder="little"
        )

    if packed_custom_mask is not None:
        mask_mode = MaskMode.CUSTOM.value
    else:
        if causal:
            mask_mode = MaskMode.CAUSAL.value
        else:
            mask_mode = MaskMode.NON_CAUSAL.value

    lse = None
    if return_lse:
        lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=q.device)

    if backend == "auto":
        backend = determine_attention_backend(
            q.device,
            PosEncodingMode[pos_encoding_mode].value,
            use_fp16_qk_reduction,
            packed_custom_mask is not None,  # use_custom_mask
            q.dtype,
            k.dtype,
        )
    module_getter = get_pod_module(backend)

    out = torch.empty_like(q)
    module_getter(
        q.dtype,
        k.dtype,
        q.dtype,
        q.shape[-1],
        PosEncodingMode[pos_encoding_mode].value,
        window_left >= 0,  # use_sliding_window
        logits_soft_cap > 0,  # use_logits_soft_cap
        use_fp16_qk_reduction,
    ).run(
        q,
        k,
        v,
        tmp,
        out,
        lse,
        mask_mode,
        TensorLayout[kv_layout].value,
        window_left,
        packed_custom_mask,
        _get_cache_alibi_slopes_buf(q.shape[1], q.device),
        logits_soft_cap,
        sm_scale,
        rope_scale,
        rope_theta,
    )

    return (out, lse) if return_lse else out


pod_with_kv_cache_return_lse = functools.partial(
    pod_with_kv_cache, return_lse=True
)


def _compute_page_mask_indptr(
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    if len(qo_indptr) != len(paged_kv_indptr):
        raise ValueError(
            "The length of qo_indptr and paged_kv_indptr should be the same."
        )
    mask_indptr = torch.empty_like(qo_indptr)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1])
        * (
            (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) * page_size
            + paged_kv_last_page_len
        ),
        0,
    )
    return mask_indptr