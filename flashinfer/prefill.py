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
    get_batch_prefill_uri,
    get_single_prefill_uri,
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

_single_prefill_modules = {}
_single_prefill_sm90_modules = {}
_batch_prefill_modules = {}
_batch_prefill_sm90_modules = {}
_batch_prefill_jit_modules = {}


def get_single_prefill_module(backend):
    def backend_module(*args):
        global _single_prefill_modules, _single_prefill_sm90_modules
        modules_dict = (
            _single_prefill_modules
            if backend == "fa2"
            else _single_prefill_sm90_modules
        )
        if args not in modules_dict:
            uri = get_single_prefill_uri(backend, *args)
            if has_prebuilt_ops and uri in prebuilt_ops_uri:
                if backend == "fa2":
                    from . import _kernels

                    run_func = _kernels.single_prefill_with_kv_cache
                else:
                    from . import _kernels_sm90

                    run_func = _kernels_sm90.single_prefill_with_kv_cache_sm90
            else:
                run_func = gen_single_prefill_module(backend, *args).run

            # torch library for single_prefill_with_kv_cache

            @register_custom_op(
                f"flashinfer::{uri}_run", mutates_args=("tmp", "o", "maybe_lse")
            )
            def run_single_prefill(
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
            def _fake_run_single_prefill(
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
            modules_dict[args] = SimpleNamespace(run=run_single_prefill)

        return modules_dict[args]

    return backend_module


def get_batch_prefill_module(backend):
    def backend_module(*args):
        global _batch_prefill_modules, _batch_prefill_sm90_modules
        modules_dict = (
            _batch_prefill_modules if backend == "fa2" else _batch_prefill_sm90_modules
        )
        if args not in modules_dict:
            uri = get_batch_prefill_uri(backend, *args)
            if has_prebuilt_ops and uri in prebuilt_ops_uri:
                if backend == "fa2":
                    from . import _kernels

                    plan_func = _kernels.batch_prefill_with_kv_cache_plan
                    ragged_run_func = _kernels.batch_prefill_with_ragged_kv_cache_run
                    paged_run_func = _kernels.batch_prefill_with_paged_kv_cache_run
                else:
                    from . import _kernels_sm90

                    plan_func = _kernels_sm90.batch_prefill_with_kv_cache_sm90_plan
                    ragged_run_func = (
                        _kernels_sm90.batch_prefill_with_ragged_kv_cache_sm90_run
                    )
                    paged_run_func = (
                        _kernels_sm90.batch_prefill_with_paged_kv_cache_sm90_run
                    )
            else:
                module = gen_batch_prefill_module(backend, *args)
                plan_func = module.plan
                ragged_run_func = module.ragged_run
                paged_run_func = module.paged_run

            # torch library for ragged_run

            @register_custom_op(
                f"flashinfer::{uri}_ragged_run",
                mutates_args=(
                    "float_workspace_buffer",
                    "int_workspace_buffer",
                    "o",
                    "maybe_lse",
                ),
            )
            def ragged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                qo_indptr: torch.Tensor,
                kv_indptr: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
            ) -> None:
                with q.device as device:  # device guard
                    if backend == "fa2":
                        ragged_run_func(
                            float_workspace_buffer,
                            int_workspace_buffer,
                            plan_info_vec,
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            maybe_custom_mask,
                            maybe_mask_indptr,
                            maybe_alibi_slopes,
                            logits_soft_cap,
                            sm_scale,
                            1.0 / rope_scale,  # rope_rcp_scale
                            1.0 / rope_theta,  # rope_rcp_theta
                            get_cuda_stream(device),
                        )
                    else:
                        ragged_run_func(
                            float_workspace_buffer,
                            int_workspace_buffer,
                            plan_info_vec,
                            q,
                            k,
                            v,
                            qo_indptr,
                            kv_indptr,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            logits_soft_cap,
                            sm_scale,
                            get_cuda_stream(device),
                        )

                    return o

            @register_fake_op(f"flashinfer::{uri}_ragged_run")
            def _fake_ragged_run(
                float_workspace_buffer: torch.Tensor,
                int_workspace_buffer: torch.Tensor,
                plan_info_vec: List[int],
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                qo_indptr: torch.Tensor,
                kv_indptr: torch.Tensor,
                o: torch.Tensor,
                maybe_lse: Optional[torch.Tensor],
                mask_mode: int,
                layout: int,
                window_left: int,
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
            ) -> None:
                pass

            # torch library for paged_run

            @register_custom_op(
                f"flashinfer::{uri}_paged_run",
                mutates_args=(
                    "float_workspace_buffer",
                    "int_workspace_buffer",
                    "paged_k_cache",
                    "paged_v_cache",
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
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
                logits_soft_cap: float,
                sm_scale: float,
                rope_scale: float,
                rope_theta: float,
            ) -> None:
                with q.device as device:  # device guard
                    if backend == "fa2":
                        paged_run_func(
                            float_workspace_buffer,
                            int_workspace_buffer,
                            plan_info_vec,
                            q,
                            paged_k_cache,
                            paged_v_cache,
                            qo_indptr,
                            paged_kv_indptr,
                            paged_kv_indices,
                            paged_kv_last_page_len,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            maybe_custom_mask,
                            maybe_mask_indptr,
                            maybe_alibi_slopes,
                            logits_soft_cap,
                            sm_scale,
                            1.0 / rope_scale,  # rope_rcp_scale
                            1.0 / rope_theta,  # rope_rcp_theta
                            get_cuda_stream(device),
                        )
                    else:
                        paged_run_func(
                            float_workspace_buffer,
                            int_workspace_buffer,
                            plan_info_vec,
                            q,
                            paged_k_cache,
                            paged_v_cache,
                            qo_indptr,
                            paged_kv_indptr,
                            paged_kv_indices,
                            paged_kv_last_page_len,
                            o,
                            maybe_lse,
                            mask_mode,
                            layout,
                            window_left,
                            logits_soft_cap,
                            sm_scale,
                            get_cuda_stream(device),
                        )
                    return o

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
                maybe_custom_mask: Optional[torch.Tensor],
                maybe_mask_indptr: Optional[torch.Tensor],
                maybe_alibi_slopes: Optional[torch.Tensor],
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
            modules_dict[args] = SimpleNamespace(
                plan=plan_func,
                ragged_run=ragged_run,
                paged_run=paged_run,
            )
        return modules_dict[args]

    return backend_module


def get_batch_prefill_jit_module(module_name: str, jit_module: Any):
    global _batch_prefill_jit_modules
    if module_name in _batch_prefill_jit_modules:
        return _batch_prefill_jit_modules[module_name]

    plan_func = jit_module.plan
    ragged_run_func = jit_module.ragged_run
    paged_run_func = jit_module.paged_run

    # torch library for ragged_run
    @register_custom_op(
        f"flashinfer::{module_name}_ragged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "o",
            "maybe_lse",
        ),
    )
    def ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        with q.device as device:  # device guard
            ragged_run_func(
                float_workspace_buffer,
                int_workspace_buffer,
                plan_info_vec,
                q,
                k,
                v,
                qo_indptr,
                kv_indptr,
                o,
                maybe_lse,
                mask_mode,
                layout,
                window_left,
                *args,
                get_cuda_stream(device),
            )

    @register_fake_op(f"flashinfer::{module_name}_ragged_run")
    def _fake_ragged_run(
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        plan_info_vec: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        o: torch.Tensor,
        maybe_lse: Optional[torch.Tensor],
        mask_mode: int,
        layout: int,
        window_left: int,
        *args,
    ) -> None:
        pass

    # torch library for paged_run
    @register_custom_op(
        f"flashinfer::{module_name}_paged_run",
        mutates_args=(
            "float_workspace_buffer",
            "int_workspace_buffer",
            "paged_k_cache",
            "paged_v_cache",
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
        *args,
    ) -> None:
        with q.device as device:  # device guard
            paged_run_func(
                float_workspace_buffer,
                int_workspace_buffer,
                plan_info_vec,
                q,
                paged_k_cache,
                paged_v_cache,
                qo_indptr,
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
                o,
                maybe_lse,
                mask_mode,
                layout,
                window_left,
                *args,
                get_cuda_stream(device),
            )

    @register_fake_op(f"flashinfer::{module_name}_paged_run")
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
        *args,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    _batch_prefill_jit_modules[module_name] = SimpleNamespace(
        plan=plan_func,
        ragged_run=ragged_run,
        paged_run=paged_run,
    )

    return _batch_prefill_jit_modules[module_name]


def single_prefill_with_kv_cache_with_jit_module(
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
            "single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, device=device
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
def single_prefill_with_kv_cache(
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
def single_prefill_with_kv_cache(
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


def single_prefill_with_kv_cache(
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
    >>> o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True,
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
    >>> o_custom = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
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
    tmp = _get_cache_buf("single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, q.device)
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
    module_getter = get_single_prefill_module(backend)

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


single_prefill_with_kv_cache_return_lse = functools.partial(
    single_prefill_with_kv_cache, return_lse=True
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


class BatchPrefillWithPagedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with paged kv-cache for batch of
    requests.

    Check :ref:`our tutorial<page-layout>` for page table layout.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> max_num_pages = 128
    >>> page_size = 16
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    >>> paged_kv_indptr = torch.tensor(
    ...     [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    ... )
    >>> # 1 <= paged_kv_last_page_len <= page_size
    >>> paged_kv_last_page_len = torch.tensor(
    ...     [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    ... )
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> kv_cache_at_layer = torch.randn(
    ...     num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    ... )
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     causal=True,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.run(q, kv_cache)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.tril(
    ...         torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i]),
    ...     )
    ...     mask_arr.append(mask_i.flatten())
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     paged_kv_indptr,
    ...     paged_kv_indices,
    ...     paged_kv_last_page_len,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     page_size,
    ...     custom_mask=mask,
    ... )
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     kv_cache = kv_cache_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.run(q, kv_cache)
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...



    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    creates some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

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
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchPrefillWithPagedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        paged_kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indptr`` array, the size of this
            buffer should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        paged_kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_indices`` array, should be large
            enough to store the maximum possible size of the ``paged_kv_indices`` array during
            the lifetime of the wrapper. This argument is only effective when ``use_cuda_graph``
            is ``True``.

        paged_kv_last_page_len_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``paged_kv_last_page_len`` array, the size of
            the buffer should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved buffer to store the custom mask tensor, should be large enough to
            store the maximum possible size of the packed custom mask tensor during the lifetime of
            the wrapper. This argument is only effective when ``use_cuda_graph`` is set to ``True``
            and the custom mask will be used in attention computation.

        mask_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``mask_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True`` and the custom
            mask will be used in attention computation.

        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                gen_customize_batch_prefill_module(backend, *jit_args),
            )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        if backend in ["fa3", "auto"]:
            # NOTE(Zihao): assume maximum accumulate kv length is 16M
            self._vector_sparse_indices_buffer = torch.empty(
                (16 * 1024 * 1024,), dtype=torch.int32, device=self.device
            )
            # NOTE(Zihao): assume maximum batch size is 32768
            self._vector_sparse_indptr_buffer = torch.empty(
                (32768,), dtype=torch.int32, device=self.device
            )

        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            device="cpu",
            pin_memory=True,
        )
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indptr_buf):
                raise ValueError(
                    "paged_kv_indptr_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_indices_buf):
                raise ValueError(
                    "paged_kv_indices_buf should be a torch.Tensor in CUDA graph mode"
                )
            if not torch.is_tensor(paged_kv_last_page_len_buf):
                raise ValueError(
                    "paged_kv_last_page_len_buf should be a torch.Tensor in CUDA graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(paged_kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of paged_kv_indptr_buf should be batch_size + 1."
                )
            if len(paged_kv_last_page_len_buf) != self._fixed_batch_size:
                raise ValueError(
                    "The length of paged_kv_last_page_len_buf should be batch_size."
                )
            # NOTE(Zihao): do not check custom_mask_buf and mask_indptr_buf here, as they are optional
        else:
            self._fixed_batch_size = 0

        self._qo_indptr_buf = qo_indptr_buf
        self._paged_kv_indptr_buf = paged_kv_indptr_buf
        self._paged_kv_indices_buf = paged_kv_indices_buf
        self._paged_kv_last_page_len_buf = paged_kv_last_page_len_buf
        self._custom_mask_buf = custom_mask_buf
        self._mask_indptr_buf = mask_indptr_buf
        self._max_total_num_rows = None
        self._backend = backend

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

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = False,
    ) -> None:
        r"""Plan batch prefill/append attention on Paged KV-Cache for given problem specification.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        paged_kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        paged_kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[qo_indptr[-1]]``.
        paged_kv_last_page_len : torch.Tensor
            The number of entries in the last page of each request in the paged
            kv-cache, shape: ``[batch_size]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim : int
            The dimension of the heads.
        page_size : int
            The size of each page in the paged kv-cache.
        custom_mask : Optional[torch.Tensor]
            The flattened boolean mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            The elements in the mask tensor should be either ``True`` or ``False``,
            where ``False`` means the corresponding element in the attention matrix will be
            masked out.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

            When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
            function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
            additional overhead.
        packed_custom_mask : Optional[torch.Tensor]
            The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
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
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Union[str, torch.dtype]
            The data type of the query tensor, defaults torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``False``.
            If ``True``, user should synchronize before calling :meth:`run` or cuda graph replay.

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        batch_size = len(qo_indptr) - 1
        if custom_mask is not None or packed_custom_mask is not None:
            mask_indptr = _compute_page_mask_indptr(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                page_size,
            )
        if packed_custom_mask is None and custom_mask is not None:
            # create packed custom mask from custom mask
            packed_custom_mask, mask_indptr = segment_packbits(
                custom_mask.contiguous().view(-1),
                mask_indptr,
                bitorder="little",
            )

        # NOTE(Zihao): only required if qo_indptr/paged_kv_indptr are device tensors
        qo_indptr_host = qo_indptr.to("cpu")
        paged_kv_indptr_host = paged_kv_indptr.to("cpu")
        paged_kv_last_page_len_host = paged_kv_last_page_len.to("cpu")
        kv_lens_arr_host = get_seq_lens(
            paged_kv_indptr_host, paged_kv_last_page_len_host, page_size
        )
        self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
            kv_lens_arr_host, non_blocking=non_blocking
        )

        total_num_rows = qo_indptr_host[-1]

        if self.is_cuda_graph_enabled:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_num_rows
            elif total_num_rows > self._max_total_num_rows:
                raise ValueError(
                    "The total number of rows in qo_indptr {} in cuda graph mode cannot "
                    "exceed the number of rows set during initialization {}.".format(
                        total_num_rows, self._max_total_num_rows
                    )
                )

            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed during the lifecycle of the wrapper in "
                    "cuda graph mode, the runtime batch size {} mismatches the batch size {} "
                    " set during initialization.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            if len(paged_kv_indices) > len(self._paged_kv_indices_buf):
                raise ValueError(
                    "The length of paged_kv_indices exceeds the allocated buffer size."
                )

            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
            self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
            self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
                paged_kv_indices, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf.copy_(
                paged_kv_last_page_len, non_blocking=non_blocking
            )

            if packed_custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._mask_indptr_buf):
                    raise ValueError(
                        "mask_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                self._custom_mask_buf[: len(packed_custom_mask)].copy_(
                    packed_custom_mask, non_blocking=non_blocking
                )
                # NOTE(Zihao): mask_indptr has the same length as qo_indptr
                self._mask_indptr_buf.copy_(mask_indptr, non_blocking=non_blocking)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=non_blocking)
            self._paged_kv_indptr_buf = paged_kv_indptr.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf = paged_kv_indices.to(
                self.device, non_blocking=non_blocking
            )
            self._paged_kv_last_page_len_buf = paged_kv_last_page_len.to(
                self.device, non_blocking=non_blocking
            )
            if packed_custom_mask is not None:
                self._custom_mask_buf = packed_custom_mask.to(
                    self.device, non_blocking=non_blocking
                )
                self._mask_indptr_buf = mask_indptr.to(
                    self.device, non_blocking=non_blocking
                )

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type

        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    self._custom_mask_buf is not None,  # use_custom_mask
                    q_data_type,
                    kv_data_type,
                )

            get_module_args = (
                q_data_type,
                kv_data_type,
                q_data_type,
                paged_kv_indptr.dtype,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                window_left >= 0,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )

            self._cached_module = get_batch_prefill_module(self._backend)(
                *get_module_args
            )

        if self._backend == "fa3":
            if page_size != 1:
                vector_sparse_indptr_host = torch.cat(
                    [
                        torch.tensor([0], dtype=torch.int32, device=kv_lens_arr_host.device),
                        torch.cumsum(kv_lens_arr_host, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                self._vector_sparse_indptr_buffer[
                    : len(vector_sparse_indptr_host)
                ].copy_(vector_sparse_indptr_host, non_blocking=non_blocking)
                paged_kv_indptr_host = vector_sparse_indptr_host

        with self.device as device:
            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                paged_kv_indptr_host,
                kv_lens_arr_host,
                self._max_total_num_rows or total_num_rows,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                self.is_cuda_graph_enabled,
                head_dim,
                causal,
                get_cuda_stream(device),
            )

        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
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
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This function is deprecated, please use :meth:`run` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *args,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch prefill/append attention between query and paged kv-cache.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
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

        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        return_lse : bool
            Whether to return the logsumexp of attention output

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        _check_cached_qkv_data_type(
            q, k_cache, self._cached_q_data_type, self._cached_kv_data_type
        )
        stride_block = k_cache.stride(0)
        if self._kv_layout == "NHD":
            page_size = k_cache.shape[1]
            stride_n = k_cache.stride(1)
        else:
            page_size = k_cache.shape[2]
            stride_n = k_cache.stride(2)
        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if k_scale is not None:
            sm_scale *= k_scale
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        lse = None
        if return_lse:
            lse = torch.empty(
                (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
            )

        out = torch.empty_like(q)

        if self._custom_mask_buf is not None:
            mask_mode = MaskMode.CUSTOM.value
        else:
            if self._causal:
                mask_mode = MaskMode.CAUSAL.value
            else:
                mask_mode = MaskMode.NON_CAUSAL.value

        if self._backend == "fa3":
            # NOTE(Zihao): we divide both stride_block and stride_n by stride_n
            # because we will multiply stride_n back in the kernel
            sparse_indices = block_sparse_indices_to_vector_sparse_offsets(
                self._paged_kv_indices_buf,
                self._paged_kv_indptr_buf,
                self._vector_sparse_indices_buffer,  # output
                self._vector_sparse_indptr_buffer,
                self._kv_lens_buffer,
                stride_block // stride_n,
                1,  # stride_n // stride_n
                page_size,
            )
            sparse_indptr = self._vector_sparse_indptr_buffer
        else:
            sparse_indices = self._paged_kv_indices_buf
            sparse_indptr = self._paged_kv_indptr_buf

        run_args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k_cache,
            v_cache,
            self._qo_indptr_buf,
            sparse_indptr,
            sparse_indices,
            self._paged_kv_last_page_len_buf,
            out,
            lse,
            mask_mode,
            TensorLayout[self._kv_layout].value,
            window_left,
        ]

        if self._jit_module is not None:
            run_args.extend(list(args))
        else:
            run_args += [
                self._custom_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
            ]

        self._cached_module.paged_run(*run_args)
        if v_scale is not None:
            out *= v_scale

        return (out, lse) if return_lse else out

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def forward_return_lse(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: This function is deprecated, please use :meth:`run_return_lse` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run_return_lse(q, paged_kv_cache, k_scale, v_scale)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass


def _compute_mask_indptr(
    qo_indptr: torch.Tensor, kv_indptr: torch.Tensor
) -> torch.Tensor:
    if len(qo_indptr) != len(kv_indptr):
        raise ValueError("The length of qo_indptr and kv_indptr should be the same.")
    mask_indptr = torch.empty_like(qo_indptr)
    mask_indptr[0] = 0
    mask_indptr[1:] = torch.cumsum(
        (qo_indptr[1:] - qo_indptr[:-1]) * (kv_indptr[1:] - kv_indptr[:-1]),
        0,
    )
    return mask_indptr


class BatchPrefillWithRaggedKVCacheWrapper:
    r"""Wrapper class for prefill/append attention with ragged (tensor) kv-cache for
    batch of requests.

    Check :ref:`our tutorial<ragged-layout>` for ragged kv-cache layout.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_layers = 32
    >>> num_qo_heads = 64
    >>> num_kv_heads = 16
    >>> head_dim = 128
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    ...     workspace_buffer, "NHD"
    ... )
    >>> batch_size = 7
    >>> nnz_kv = 100
    >>> nnz_qo = 100
    >>> qo_indptr = torch.tensor(
    ...     [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
    ... )
    >>> kv_indptr = qo_indptr.clone()
    >>> q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    >>> k_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> v_at_layer = torch.randn(num_layers, nnz_kv, num_kv_heads, head_dim).half().to("cuda:0")
    >>> # create auxiliary data structures for batch prefill attention
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     causal=True,
    ... )
    >>> outputs = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o = prefill_wrapper.run(q, k, v)
    ...     outputs.append(o)
    ...
    >>> outputs[0].shape
    torch.Size([100, 64, 128])
    >>>
    >>> # below is another example of creating custom mask for batch prefill attention
    >>> mask_arr = []
    >>> qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
    >>> kv_len = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
    >>> for i in range(batch_size):
    ...     mask_i = torch.tril(
    ...         torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
    ...         diagonal=(kv_len[i] - qo_len[i]),
    ...     )
    ...     mask_arr.append(mask_i.flatten())
    ...
    >>> mask = torch.cat(mask_arr, dim=0)
    >>> prefill_wrapper.plan(
    ...     qo_indptr,
    ...     kv_indptr,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ...     custom_mask=mask
    ... )
    >>> outputs_custom_mask = []
    >>> for i in range(num_layers):
    ...     q = q_at_layer[i]
    ...     k = k_at_layer[i]
    ...     v = v_at_layer[i]
    ...     # compute batch prefill attention, reuse auxiliary data structures
    ...     o_custom = prefill_wrapper.run(q, k, v)
    ...     assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
    ...
    >>> outputs_custom_mask[0].shape
    torch.Size([100, 64, 128])


    Note
    ----
    To accelerate computation, FlashInfer's batch prefill/append attention operators
    creates some auxiliary data structures, these data structures can be reused across
    multiple prefill/append attention calls (e.g. different Transformer layers). This
    wrapper class manages the lifecycle of these data structures.
    """

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: Optional[torch.Tensor] = None,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        custom_mask_buf: Optional[torch.Tensor] = None,
        mask_indptr_buf: Optional[torch.Tensor] = None,
        backend: str = "auto",
        jit_args: Optional[List[Any]] = None,
    ) -> None:
        r"""Constructor of :class:`BatchPrefillWithRaggedKVCacheWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.

        kv_layout : str
            The layout of the input k/v tensors, could be either ``NHD`` or ``HND``.

        use_cuda_graph : bool
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored as the provided buffers.

        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.

        custom_mask_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the custom mask tensor, should be large
            enough to store the maximum possible size of the packed custom mask tensor during the
            lifetime of the wrapper. This argument is only effective when ``use_cuda_graph``
            is ``True`` and custom mask will be used in attention computation.

        mask_indptr_buf : Optional[torch.Tensor]
            The user reserved GPU buffer to store the ``mask_indptr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True`` and custom mask
            will be used in attention computation.

        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.
        """
        _check_kv_layout(kv_layout)
        if jit_args is not None:
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                gen_customize_batch_prefill_module(backend, *jit_args),
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
            self._int_workspace_buffer.shape, dtype=torch.uint8,
            pin_memory=True, device="cpu",
        )
        self._use_cuda_graph = use_cuda_graph
        if use_cuda_graph:
            if not torch.is_tensor(qo_indptr_buf):
                raise ValueError(
                    "qo_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            if not torch.is_tensor(kv_indptr_buf):
                raise ValueError(
                    "kv_indptr_buf should be a torch.Tensor in cuda graph mode"
                )
            self._fixed_batch_size = len(qo_indptr_buf) - 1
            if len(kv_indptr_buf) != self._fixed_batch_size + 1:
                raise ValueError(
                    "The length of kv_indptr_buf ({}) should be the same as qo_indptr_buf ({}).".format(
                        len(kv_indptr_buf), self._fixed_batch_size
                    )
                )
            # NOTE(Zihao): do not check custom_mask_buf and mask_indptr_buf here,
            # as they may not be used.

        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._custom_mask_buf = custom_mask_buf
        self._mask_indptr_buf = mask_indptr_buf
        self._max_total_num_rows = None
        self._backend = backend

    @property
    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer
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

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        custom_mask: Optional[torch.Tensor] = None,
        packed_custom_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: str = "float16",
        kv_data_type: Optional[str] = None,
    ) -> None:
        r"""Plan batch prefill/append attention on Ragged KV-Cache for given problem specification.

        Parameters
        ----------
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        kv_indptr : torch.Tensor
            The indptr of the key/value tensor, shape: ``[batch_size + 1]``.
        num_qo_heads : int
            The number of query/output heads.
        num_kv_heads : int
            The number of key/value heads.
        head_dim : int
            The dimension of the heads.
        custom_mask : Optional[torch.Tensor]
            The flattened boolean mask tensor, shape: ``(sum(q_len[i] * k_len[i] for i in range(batch_size))``.
            The elements in the mask tensor should be either ``True`` or ``False``,
            where ``False`` means the corresponding element in the attention matrix will be
            masked out.

            Please refer to the :ref:`mask layout <mask-layout>` for more details about flattened
            layout of mask tensor.

            When :attr:`custom_mask` is provided, and :attr:`packed_custom_mask` is not, the
            function will pack the custom mask tensor into a 1D packed mask tensor, which introduces
            additional overhead.
        packed_custom_mask : Optional[torch.Tensor]
            The 1D packed uint8 mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.

            If provided, the custom mask will be added to the attention matrix before softmax
            and after scaling. The mask tensor should be in the same device as the input tensors.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This argument is ignored if ``mask`` is provided in :meth:`plan`.
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
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Union[str, torch.dtype]
            The data type of the query tensor, defaults to torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.

        Note
        ----
        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this plan call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        The :meth:`plan` method cannot be used in Cuda Graph or in ``torch.compile``.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        batch_size = len(qo_indptr) - 1
        if len(kv_indptr) != batch_size + 1:
            raise ValueError(
                "The kv_indptr length should be equal to mask_indptr length."
            )
        if custom_mask is not None or packed_custom_mask is not None:
            mask_indptr = _compute_mask_indptr(qo_indptr, kv_indptr)
        if packed_custom_mask is None and custom_mask is not None:
            # create packed custom mask from custom mask
            packed_custom_mask, mask_indptr = segment_packbits(
                custom_mask.contiguous().view(-1),
                mask_indptr,
                bitorder="little",
            )

        # NOTE(Zihao): only required if qo_indptr/paged_kv_indptr are device tensors
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")

        total_num_rows = qo_indptr_host[-1]

        if self.is_cuda_graph_enabled:
            if self._max_total_num_rows is None:
                self._max_total_num_rows = total_num_rows
            elif total_num_rows > self._max_total_num_rows:
                raise ValueError(
                    "The total number of rows in qo_indptr {} in cuda graph mode cannot "
                    "exceed the number of rows set during initialization {}.".format(
                        total_num_rows, self._max_total_num_rows
                    )
                )

            if batch_size != self._fixed_batch_size:
                raise ValueError(
                    "The batch size should be fixed in cudagraph mode, the runtime batch size {} "
                    " mismatches the batch size set during initialization {}.".format(
                        batch_size, self._fixed_batch_size
                    )
                )
            self._qo_indptr_buf.copy_(qo_indptr)
            self._kv_indptr_buf.copy_(kv_indptr)
            if packed_custom_mask is not None:
                if not torch.is_tensor(self._custom_mask_buf):
                    raise ValueError(
                        "custom_mask_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in attention computation."
                    )
                if not torch.is_tensor(self._mask_indptr_buf):
                    raise ValueError(
                        "mask_indptr_buf must be initialized with a torch.Tensor in cuda graph mode if we use custom mask in the attention computation."
                    )
                self._custom_mask_buf[: len(packed_custom_mask)] = packed_custom_mask
                self._mask_indptr_buf.copy_(mask_indptr)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device)
            self._kv_indptr_buf = kv_indptr.to(self.device)
            if packed_custom_mask is not None:
                self._custom_mask_buf = packed_custom_mask.to(self.device)
                self._mask_indptr_buf = mask_indptr.to(self.device)

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        kv_len_arr = kv_indptr_host[1:] - kv_indptr_host[:-1]

        if self._jit_module is not None:
            self._cached_module = self._jit_module
        else:
            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    self._custom_mask_buf is not None,  # use_custom_mask
                    q_data_type,
                    kv_data_type,
                )

            get_module_args = (
                q_data_type,
                kv_data_type,
                q_data_type,
                kv_indptr.dtype,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                window_left >= 0,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )
            self._cached_module = get_batch_prefill_module(self._backend)(
                *get_module_args
            )

        with self.device as device:
            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                kv_indptr_host,
                kv_len_arr,
                self._max_total_num_rows or total_num_rows,
                batch_size,
                num_qo_heads,
                num_kv_heads,
                1,  # page_size
                self.is_cuda_graph_enabled,
                head_dim,
                causal,
                get_cuda_stream(device),
            )

        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This function is deprecated, please use :meth:`run` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v)

    @overload
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        return_lse: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        return_lse: Literal[True] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch prefill/append attention between query and kv-cache stored as
        ragged tensor.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``
        k : torch.Tensor
            The key tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim]``
        v : torch.Tensor
            The value tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim]``
        return_lse : bool
            Whether to return the logsumexp of attention output

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        _check_cached_qkv_data_type(
            q, k, self._cached_q_data_type, self._cached_kv_data_type
        )

        window_left = self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        lse = None
        if return_lse:
            lse = torch.empty(
                (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
            )
        out = torch.empty_like(q)

        if is_float8(q):
            logging.warning(
                "Our current prefill kernel implementation needs f16 input, the f8 inputs "
                " are casted to f16, which could result in performance degradation."
            )
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        if self._custom_mask_buf is not None:
            mask_mode = MaskMode.CUSTOM.value
        else:
            if self._causal:
                mask_mode = MaskMode.CAUSAL.value
            else:
                mask_mode = MaskMode.NON_CAUSAL.value

        run_args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr_buf,
            self._kv_indptr_buf,
            out,
            lse,
            mask_mode,
            TensorLayout[self._kv_layout].value,
            window_left,
        ]
        if self._jit_module is not None:
            run_args.extend(list(args))
        else:
            run_args += [
                self._custom_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
            ]

        self._cached_module.ragged_run(*run_args)
        return (out, lse) if return_lse else out

    run_return_lse = functools.partialmethod(run, return_lse=True)

    def forward_return_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Warning: This function is deprecated, please use :meth:`run_return_lse` instead."""
        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run_return_lse(q, k, v)

    def end_forward(self) -> None:
        r"""Warning: this function is deprecated and has no effect."""
        pass
