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
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, overload

import torch

from .jit import (
    gen_batch_prefill_module,
    gen_customize_batch_prefill_module,
    gen_fmha_cutlass_sm100a_module,
    gen_single_prefill_module,
    get_batch_prefill_uri,
    get_single_prefill_uri,
    setup_cubin_loader,
    setup_metainfo_loader,
    trtllm_gen_fmha_module,
)
from .cudnn import cudnn_batch_prefill_with_kv_cache
from .page import block_sparse_indices_to_vector_sparse_offsets, get_seq_lens
from .quantization import packbits, segment_packbits
from .utils import (
    FP4Tensor,
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_cached_qkv_data_type,
    _check_kv_layout,
    _check_pos_encoding_mode,
    _check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    _get_cache_buf,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    determine_attention_backend,
    device_support_pdl,
    get_device_sm_count,
    is_float8,
    is_sm100a_supported,
    register_custom_op,
    register_fake_op,
    ceil_div,
    round_up,
)


@functools.cache
def get_fmha_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
    use_fp16_qk_reduction: bool = False,
):
    if is_sm100a_supported(torch.device("cuda")):
        return gen_fmha_cutlass_sm100a_module(
            dtype_q,
            dtype_kv,
            dtype_o,
            dtype_idx,
            head_dim_qk,
            head_dim_vo,
            pos_encoding_mode,
            use_sliding_window,
            use_logits_soft_cap,
        ).build_and_load()
    else:
        raise ValueError("SM100A is not supported on this device")


def make_hashable_cache(func):
    """
    Decorator that converts unhashable arguments (like lists) to hashable ones (tuples)
    before applying functools.cache.
    """

    @functools.cache
    def cached_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Convert unhashable arguments to hashable ones
        hashable_args = []
        for arg in args:
            if isinstance(arg, list):
                hashable_args.append(tuple(arg))
            else:
                hashable_args.append(arg)

        hashable_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list):
                hashable_kwargs[key] = tuple(value)
            else:
                hashable_kwargs[key] = value

        return cached_wrapper(*hashable_args, **hashable_kwargs)

    return wrapper


@make_hashable_cache
def get_customize_batch_prefill_module(
    backend: str,
    uri: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    idtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    additional_tensor_names: List[str],
    additional_tensor_dtypes: List[str],
    additional_scalar_names: List[str],
    additional_scalar_dtypes: List[str],
    variant_name: str,
    variant_decl: str,
    pos_encoding_mode: int = 0,
    use_sliding_window: bool = False,
    use_logits_soft_cap: bool = False,
    use_fp16_qk_reduction: bool = False,
    fp8_enabled: bool = False,
):
    return gen_customize_batch_prefill_module(
        backend,
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        idtype,
        head_dim_qk,
        head_dim_vo,
        additional_tensor_names,
        additional_tensor_dtypes,
        additional_scalar_names,
        additional_scalar_dtypes,
        variant_name,
        variant_decl,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
        use_fp16_qk_reduction,
        fp8_enabled,
    ).build_and_load()


@functools.cache
def get_trtllm_gen_prefill_module():
    mod = trtllm_gen_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    setup_metainfo_loader(mod.get_library_path())

    def _paged_run(
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        batch_size: int,
        cum_seq_lens_q: torch.Tensor,
        cum_seq_lens_kv: torch.Tensor,
        enable_pdl: bool,
        window_left: int = -1,
        out: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sm_count = get_device_sm_count(query.device)
        if out is None:
            out = torch.empty_like(query)
        op.trtllm_paged_attention_context(
            out,
            None,  # fp4 output not supported in wrapper api yet.
            query,
            k_cache,
            v_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            max_kv_len,
            bmm1_scale,
            bmm2_scale,
            -1,  # o_sf_scale
            -1,  # o_sf_vec_size
            0,  # o_sf_start_index
            batch_size,
            window_left,
            cum_seq_lens_q,
            cum_seq_lens_kv,
            sm_count,
            enable_pdl,
            sinks,
        )
        return out

    def _ragged_run(*args, **kwargs):
        # TODO(Zihao): trtllm-gen backend already supports variable length attention,
        # but not integrated into flashinfer yet.
        raise NotImplementedError(
            "Variable length is not implemented for trtllm-gen backend yet."
        )

    def _plan(*args, **kwargs):
        pass

    return SimpleNamespace(
        paged_run=_paged_run,
        ragged_run=_ragged_run,
        plan=_plan,
    )


@functools.cache
def get_single_prefill_module(backend, *args):
    uri = get_single_prefill_uri(backend, *args)
    module = gen_single_prefill_module(backend, *args).build_and_load()
    run_func = module.run.default

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
        scale_q: Optional[torch.Tensor],
        scale_k: Optional[torch.Tensor],
        scale_v: Optional[torch.Tensor],
        rope_scale: float,
        rope_theta: float,
    ) -> None:
        if backend == "fa3":
            if not is_float8(q):
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
                )
            else:
                # FP8 enabled
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
                    scale_q,
                    scale_k,
                    scale_v,
                    sm_scale,
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
    return SimpleNamespace(run=run_single_prefill)


@functools.cache
def get_batch_prefill_module(backend, *args):
    if backend == "trtllm-gen":
        uri = "trtllm_gen_context"
        module = get_trtllm_gen_prefill_module()
        plan_func = module.plan
        ragged_run_func = module.ragged_run
        paged_run_func = module.paged_run
    else:
        uri = get_batch_prefill_uri(backend, *args)
        module = gen_batch_prefill_module(backend, *args).build_and_load()
        plan_func = module.plan.default
        ragged_run_func = module.ragged_run.default
        paged_run_func = module.paged_run.default

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
    ) -> None:
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
                enable_pdl,
                maybe_custom_mask,
                maybe_mask_indptr,
                maybe_alibi_slopes,
                maybe_prefix_len_ptr,
                maybe_token_pos_in_items_ptr,
                maybe_max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                1.0 / rope_scale,  # rope_rcp_scale
                1.0 / rope_theta,  # rope_rcp_theta
                token_pos_in_items_len,
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
                enable_pdl,
                maybe_prefix_len_ptr,
                maybe_token_pos_in_items_ptr,
                maybe_max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                token_pos_in_items_len,
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
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_q_len: Optional[int] = None,
        max_kv_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        cum_seq_lens_kv: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        if backend == "trtllm-gen":
            assert maybe_lse is None
            assert num_qo_heads is not None
            assert num_kv_heads is not None
            assert block_tables is not None
            assert kv_lens_buffer is not None
            assert page_size is not None
            assert max_kv_len is not None
            assert batch_size is not None
            assert cum_seq_lens_q is not None
            assert cum_seq_lens_kv is not None
            assert enable_pdl is not None
            o = paged_run_func(
                q.contiguous(),  # NOTE(Siyuan): without contiguous, the result is incorrect
                paged_k_cache,
                paged_v_cache,
                int_workspace_buffer,
                block_tables,
                kv_lens_buffer,
                max_q_len,
                max_kv_len,
                sm_scale,
                1.0,  # NOTE(Siyuan): update this to expose bmm2 scale
                batch_size,
                cum_seq_lens_q,
                cum_seq_lens_kv,
                enable_pdl,
                window_left,
                out=o,
                sinks=sinks,
            )
        elif backend == "fa2":
            assert not is_float8(q)
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
                enable_pdl,
                maybe_custom_mask,
                maybe_mask_indptr,
                maybe_alibi_slopes,
                maybe_prefix_len_ptr,
                maybe_token_pos_in_items_ptr,
                maybe_max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                1.0 / rope_scale,  # rope_rcp_scale
                1.0 / rope_theta,  # rope_rcp_theta
                token_pos_in_items_len,
            )
        else:
            if not is_float8(q):
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
                    enable_pdl,
                    maybe_prefix_len_ptr,
                    maybe_token_pos_in_items_ptr,
                    maybe_max_item_len_ptr,
                    logits_soft_cap,
                    sm_scale,
                    token_pos_in_items_len,
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
                    enable_pdl,
                    scale_q,
                    scale_k,
                    scale_v,
                    sm_scale,
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
        num_qo_heads: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        block_tables: Optional[torch.Tensor] = None,
        kv_lens_buffer: Optional[torch.Tensor] = None,
        page_size: Optional[int] = None,
        max_q_len: Optional[int] = None,
        max_kv_len: Optional[int] = None,
        batch_size: Optional[int] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        cum_seq_lens_kv: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    # Register the module.
    #
    # Note that plan is not part of model logic. It should not be included in
    # Cuda Graph or torch.compile. So, we don't provide a torch library for plan.
    return SimpleNamespace(
        plan=plan_func,
        ragged_run=ragged_run,
        paged_run=paged_run,
    )


@functools.cache
def get_batch_prefill_jit_module(module_name: str, jit_module: Any):
    plan_func = jit_module.plan.default
    ragged_run_func = jit_module.ragged_run.default
    paged_run_func = jit_module.paged_run.default

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
    return SimpleNamespace(
        plan=plan_func,
        ragged_run=ragged_run,
        paged_run=paged_run,
    )


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
    device = q.device
    tmp = _get_cache_buf(
        "single_prefill_with_kv_cache_tmp", 32 * 1024 * 1024, device=device
    )
    o = torch.empty(q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=device)
    lse = None
    if return_lse:
        lse = torch.empty((q.size(0), q.size(1)), dtype=torch.float32, device=device)
    jit_module.run.default(
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
    )
    return (o, lse) if return_lse else o


@overload
def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
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
    backend: str = "auto",
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
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
    backend: str = "auto",
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def single_prefill_with_kv_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_q: Optional[torch.Tensor] = None,
    scale_k: Optional[torch.Tensor] = None,
    scale_v: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = None,
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
    backend: str = "auto",
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Prefill/Append attention with KV cache for single request, return the attention
    output.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[qo_len, num_qo_heads, head_dim_qk]``.
    k : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_qk]`` if :attr:`kv_layout`
        is ``NHD``, or ``[num_kv_heads, kv_len, head_dim_qk]`` if :attr:`kv_layout` is
        ``HND``.
    v : torch.Tensor
        The key tensor, shape: ``[kv_len, num_kv_heads, head_dim_vo]`` if :attr:`kv_layout`
        is ``NHD``, ``[num_kv_heads, kv_len, head_dim_vo]`` if :attr:`kv_layout` is
        ``HND``.
    scale_q : Optional[torch.Tensor]
        The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_k : Optional[torch.Tensor]
        The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    scale_v : Optional[torch.Tensor]
        The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
        Used with FP8 Quantization. If not provided, will be set to ``1.0``.
    o_dtype : Optional[torch.dtype]
        The output tensor data type, if not provided, will be set to the same as the q.
        This is necessary as output dtype cannot be automatically inferred in quant.
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
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim_qk)``.
    rope_scale : Optional[float]
        The scale used in RoPE interpolation, if not provided, will be set to 1.0.
    rope_theta : Optional[float]
        The theta used in RoPE, if not provided, will be set to 1e4.
    backend : str
        The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
        If set to ``auto``, the function will automatically choose the backend based on the
        device architecture and kernel availability.
    return_lse : bool
        Whether to return the log sum exp value of the attention logits.

    Returns
    -------
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
        If :attr:`return_lse` is ``True``, a tuple of two tensors:

        * The attention output, shape: ``[qo_len, num_qo_heads, head_dim_vo]``.
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

    if is_float8(q):
        # FP8 quant enabled, do sanity check:
        #   1. unsupported feature
        #   2. dtype check
        assert window_left == -1
        assert q.dtype == k.dtype == v.dtype
        assert q.shape[-1] == k.shape[-1] == v.shape[-1]
        if scale_q is None:
            scale_q = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
        if scale_k is None:
            scale_k = torch.ones(k.shape[1], dtype=torch.float32, device=q.device)
        if scale_v is None:
            scale_v = torch.ones(v.shape[1], dtype=torch.float32, device=q.device)

    if backend == "auto":
        backend = determine_attention_backend(
            q.device,
            PosEncodingMode[pos_encoding_mode].value,
            use_fp16_qk_reduction,
            packed_custom_mask is not None,  # use_custom_mask
            q.dtype,
            k.dtype,
        )

    # o_dtype should be provided for FP8 attention
    if o_dtype is None:
        o_dtype = q.dtype
    out = torch.empty(q.shape[:-1] + v.shape[-1:], dtype=o_dtype, device=q.device)

    module = get_single_prefill_module(
        backend,
        q.dtype,
        k.dtype,
        out.dtype,
        q.shape[-1],  # head_dim_qk
        v.shape[-1],  # head_dim_vo
        PosEncodingMode[pos_encoding_mode].value,
        window_left >= 0,  # use_sliding_window
        logits_soft_cap > 0,  # use_logits_soft_cap
        use_fp16_qk_reduction,
    )

    module.run(
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
        scale_q,
        scale_k,
        scale_v,
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

    Check :ref:`our tutorial <kv-layout>` for page table layout.

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
    >>> workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
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
    create some auxiliary data structures, these data structures can be reused across
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
        jit_kwargs: Optional[Dict[str, Any]] = None,
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
            The implementation backend, could be ``auto``/``fa2``,``fa3`` or ``cudnn``. Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.

        jit_kwargs : Optional[Dict[str, Any]]
            The keyword arguments to create the JIT module, defaults to None.
        """
        _check_kv_layout(kv_layout)

        if jit_args is not None:
            if jit_kwargs is None:
                jit_kwargs = {}
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                get_customize_batch_prefill_module(backend, *jit_args, **jit_kwargs),
            )
        else:
            self._jit_module = None

        self._kv_layout = kv_layout
        if backend == "cudnn":
            assert kv_layout == "NHD", "CUDNN backend only supports NHD layout"

        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._vector_sparse_indptr_buffer: Optional[torch.Tensor] = None
        if backend in ["fa3", "auto", "trtllm-gen"]:
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
        self._plan_info = None
        self._cached_module = None
        self._seq_lens_kv = None
        self._seq_lens_q = None
        self._block_tables = None

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
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_q: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
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
        head_dim_qk : int
            The dimension of the query/key heads.
        page_size : int
            The size of each page in the paged kv-cache.
        head_dim_vo : Optional[int]
            The dimension of the value/output heads, if not provided, will be set to
            ``head_dim_qk``.
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
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        prefix_len_ptr :Optional[torch.Tensor]
            prefix length. A uint32 1D tensor indicating the prefix length of each prompt. The tensor size is equal to the batch size.
        token_pos_in_items_ptr : Optional[float]
            A uint16 1D tensor (it will be converted to uint16 in flashinfer) indicating the token position of each item and started from 0 (delimiter)
            for each item. E.g., if we have 3 items of length 3, 2, 4 respectively for this member. This vector will be looking like
            `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]` with 4 delimiters indexed as 0. For batch size > 1,
            we will concat them as 1D with zero paddings to make sure each has the same length, the padding length is defined by
            `token_pos_in_items_len` - length of the raw `token_pos_in_items_ptr` for each prompt.
        token_pos_in_items_len : int
            zero padding length for `token_pos_in_items_ptr` to better handle the bsz > 1 case. Still using the above 3,2,4 example.
            If we set `token_pos_in_items_len` to be 20, it will be  `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]`
            with 7 padded zeros. (note there're 8 zeros in the end where the first one is the delimiter token 0 in the end of the prompt)
        max_item_len_ptr : Optional[float]
            a uint16 vector contains the max token length of all items for each prompt
        seq_lens: Optional[torch.Tensor]
            A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``.
        seq_lens_q: Optional[torch.Tensor]
            A uint32 1D tensor indicating the q sequence length of each prompt. shape: ``[batch_size]``.
            If not provided, will be set to the same value as ``seq_lens``.
        block_tables: Optional[torch.Tensor]
            A uint32 2D tensor indicating the block table of each prompt. shape: ``[batch_size, max_num_blocks_per_seq]``.
        max_token_per_sequence: Optional[int],
            Required for cudnn backend. This is the scalar max token length of each sequence.
        max_sequence_kv: Optional[int],
            Required for cudnn backend. This is the scalar max sequence length of each sequence in kv cache.

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
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk

        batch_size = len(qo_indptr) - 1
        self._batch_size = batch_size
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
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

        self._prefix_len_ptr = prefix_len_ptr
        self._token_pos_in_items_ptr = token_pos_in_items_ptr
        self._token_pos_in_items_len = token_pos_in_items_len
        self._max_item_len_ptr = max_item_len_ptr

        # NOTE(Zihao): only required if qo_indptr/paged_kv_indptr are device tensors
        if max_token_per_sequence is not None:
            self._max_q_len = max_token_per_sequence
        else:
            qo_indptr_host = qo_indptr.to("cpu")
            self._max_q_len = max(qo_indptr_host).item()
            total_num_rows = qo_indptr_host[-1]

        if max_sequence_kv is not None:
            self._max_kv_len = max_sequence_kv
        else:
            paged_kv_indptr_host = paged_kv_indptr.to("cpu")
            paged_kv_last_page_len_host = paged_kv_last_page_len.to("cpu")
            if seq_lens is None:
                kv_lens_arr_host = get_seq_lens(
                    paged_kv_indptr_host, paged_kv_last_page_len_host, page_size
                )
            else:
                kv_lens_arr_host = seq_lens.cpu().flatten()
            self._kv_lens_buffer[: len(kv_lens_arr_host)].copy_(
                kv_lens_arr_host, non_blocking=non_blocking
            )
            self._max_kv_len = max(kv_lens_arr_host).item()

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
            self._paged_kv_last_page_len_buf.copy_(
                paged_kv_last_page_len, non_blocking=non_blocking
            )
            self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
                paged_kv_indices,
                non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
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
                    packed_custom_mask,
                    non_blocking=(packed_custom_mask.device == self.device)
                    and non_blocking,
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
            else:
                self._custom_mask_buf = None
                self._mask_indptr_buf = None

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
            if self._backend != "cudnn":
                get_module_args = (
                    q_data_type,
                    kv_data_type,
                    q_data_type,
                    paged_kv_indptr.dtype,
                    head_dim_qk,
                    head_dim_vo,
                    PosEncodingMode[pos_encoding_mode].value,
                    window_left >= 0,  # use_sliding_window
                    logits_soft_cap > 0,  # use_logits_soft_cap
                    use_fp16_qk_reduction,
                )

                self._cached_module = get_batch_prefill_module(
                    self._backend, *get_module_args
                )

        if self._backend == "fa3" or self._backend == "trtllm-gen":
            if page_size != 1:
                vector_sparse_indptr_host = torch.cat(
                    [
                        torch.tensor(
                            [0], dtype=torch.int32, device=kv_lens_arr_host.device
                        ),
                        torch.cumsum(kv_lens_arr_host, dim=0, dtype=torch.int32),
                    ],
                    dim=0,
                )
                self._vector_sparse_indptr_buffer[
                    : len(vector_sparse_indptr_host)
                ].copy_(vector_sparse_indptr_host, non_blocking=non_blocking)
                paged_kv_indptr_host = vector_sparse_indptr_host

        self._block_tables = block_tables
        if self._backend == "trtllm-gen":
            assert self._kv_layout == "HND"
            assert logits_soft_cap == 0.0
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
                block_id = paged_kv_indptr_host[0]
                for i in range(batch_size):
                    num_blocks_needed = blocks_per_seq[i]
                    assert self._block_tables is not None, (
                        "block_tables is not initialized"
                    )
                    self._block_tables[i, :num_blocks_needed] = paged_kv_indices[
                        block_id : block_id + num_blocks_needed
                    ]
                    block_id += num_blocks_needed

        if self._cached_module is not None:
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
                head_dim_qk,
                head_dim_vo,
                causal,
            )

        self._causal = causal
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._seq_lens_kv = seq_lens
        self._seq_lens_q = seq_lens_q if seq_lens_q is not None else seq_lens

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
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        enable_pdl: Optional[bool] = None,
        window_left: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

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

        *args
            Additional arguments for custom kernels.
        k_scale : Optional[float]
            The calibration scale of key for fp8 input, if not provided, will be set to ``1.0``.
        v_scale : Optional[float]
            The calibration scale of value for fp8 input, if not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention output
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)
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
        window_left = self._window_left if window_left is None else window_left
        if self._backend != "trtllm-gen":
            # NOTE(Siyuan): since window_left is appeared in the plan function, we need to make sure it is the same as the one in the plan function.
            # Remove this check if the backend supports dynamic window_left.
            assert window_left == self._window_left
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
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
                _check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty(
                q.shape[:-1] + v_cache.shape[-1:], dtype=q.dtype, device=q.device
            )
        else:
            _check_shape_dtype_device(
                out, q.shape[:-1] + v_cache.shape[-1:], q.dtype, q.device, "out"
            )

        if self._custom_mask_buf is not None:
            mask_mode = MaskMode.CUSTOM.value
        else:
            if self._causal:
                mask_mode = MaskMode.CAUSAL.value
            else:
                mask_mode = MaskMode.NON_CAUSAL.value

        if self._prefix_len_ptr is not None:
            mask_mode = MaskMode.MULTIITEMSCORING.value

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

        if self._backend == "cudnn":
            if self._seq_lens_q is not None and self._seq_lens_q.dim() == 1:
                self._seq_lens_q = self._seq_lens_q.reshape(self._batch_size, 1, 1, 1)

            if self._seq_lens_kv is not None and self._seq_lens_kv.dim() == 1:
                self._seq_lens_kv = self._seq_lens_kv.reshape(self._batch_size, 1, 1, 1)

            cudnn_batch_prefill_with_kv_cache(
                q,
                k_cache,  # Need to be changed
                v_cache,  # Need to be changed
                self._sm_scale,
                self._float_workspace_buffer,
                actual_seq_lens_q=self._seq_lens_q,
                actual_seq_lens_kv=self._seq_lens_kv,
                max_token_per_sequence=self._max_q_len,
                max_sequence_kv=self._max_kv_len,
                block_tables=self._block_tables,
                causal=self._causal,
                return_lse=return_lse,
                batch_offsets_q=self._qo_indptr_buf,
                batch_offsets_o=self._qo_indptr_buf,
                out=out,
                lse=lse,
            )
        else:
            if self._backend != "trtllm-gen":
                assert self._plan_info is not None, "plan info is not initialized"
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
                enable_pdl,
            ]
            if self._jit_module is not None:
                run_args.extend(list(args))
            else:
                run_args += [
                    self._custom_mask_buf,
                    self._mask_indptr_buf,
                    _get_cache_alibi_slopes_buf(q.shape[1], q.device),
                    self._prefix_len_ptr,
                    self._token_pos_in_items_ptr,
                    self._max_item_len_ptr,
                    logits_soft_cap,
                    sm_scale,
                    None,  # scale_q, not supported yet
                    None,  # scale_k
                    None,  # scale_v
                    rope_scale,
                    rope_theta,
                    self._token_pos_in_items_len,
                    self._num_qo_heads,
                    self._num_kv_heads,
                    self._block_tables,
                    self._kv_lens_buffer,
                    page_size,
                    self._max_q_len,
                    self._max_kv_len,
                    self._batch_size,
                    self._qo_indptr_buf,
                    self._vector_sparse_indptr_buffer,
                    sinks,
                ]

            assert self._cached_module is not None, "cached module is not initialized"
            self._cached_module.paged_run(*run_args)
            if v_scale is not None:
                # TODO(Zihao): fused into kernel
                if is_float8(out):
                    out = (out.to(torch.float32) * v_scale).to(out.dtype)
                else:
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
        return self.run_return_lse(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)

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

    Check :ref:`our tutorial <kv-layout>` for ragged kv-cache layout.

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
    create some auxiliary data structures, these data structures can be reused across
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
        jit_kwargs: Optional[Dict[str, Any]] = None,
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
            The implementation backend, could be ``auto``/``fa2``/``fa3`` or ``trtllm-gen``.
            Defaults to ``auto``.
            If set to ``auto``, the wrapper will automatically choose the backend based on the
            device architecture and kernel availability.

        jit_args : Optional[List[Any]]
            If provided, the wrapper will use the provided arguments to create the JIT module,
            otherwise, the wrapper will use default attention implementation.

        jit_kwargs : Optional[Dict[str, Any]]
            The keyword arguments to create the JIT module, defaults to None.
        """
        _check_kv_layout(kv_layout)
        if jit_args is not None:
            if jit_kwargs is None:
                jit_kwargs = {}
            self._jit_module = get_batch_prefill_jit_module(
                jit_args[0],
                get_customize_batch_prefill_module(backend, *jit_args, **jit_kwargs),
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
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
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
        self._cached_module = None

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
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
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
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        non_blocking: bool = True,
        prefix_len_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_ptr: Optional[torch.Tensor] = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: Optional[torch.Tensor] = None,
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
        head_dim_qk : int
            The dimension of the heads on query/key tensor.
        head_dim_vo : Optional[int]
            The dimension of the heads on value/output tensor.
            If not provided, will be set to ``head_dim_vo``.
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
            ``1.0 / sqrt(head_dim_qk)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : Union[str, torch.dtype]
            The data type of the query tensor, defaults to torch.float16.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        prefix_len_ptr :Optional[torch.Tensor]
            prefix length. A uint32 1D tensor indicating the prefix length of each prompt. The tensor size is equal to the batch size.
        token_pos_in_items_ptr : Optional[float]
            A uint16 1D tensor (it will be converted to uint16 in flashinfer) indicating the token position of each item and started from 0 (delimiter)
            for each item. E.g., if we have 3 items of length 3, 2, 4 respectively for this member. This vector will be looking like
            `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]` with 4 delimiters indexed as 0. For batch size > 1,
            we will concat them as 1D with zero paddings to make sure each has the same length, the padding length is defined by
            `token_pos_in_items_len` - length of the raw `token_pos_in_items_ptr` for each prompt.
        token_pos_in_items_len : int
            zero padding length for `token_pos_in_items_ptr` to better handle the bsz > 1 case. Still using the above 3,2,4 example.
            If we set `token_pos_in_items_len` to be 20, it will be  `[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]`
            with 7 padded zeros. (note there're 8 zeros in the end where the first one is the delimiter token 0 in the end of the prompt)
        max_item_len_ptr : Optional[float]
            a uint16 vector contains the max token length of all items for each prompt

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
        if head_dim_vo is None:
            head_dim_vo = head_dim_qk

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
            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
            self._kv_indptr_buf.copy_(kv_indptr, non_blocking=non_blocking)
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
                self._mask_indptr_buf.copy_(mask_indptr, non_blocking=non_blocking)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=non_blocking)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
            if packed_custom_mask is not None:
                self._custom_mask_buf = packed_custom_mask.to(
                    self.device, non_blocking=non_blocking
                )
                self._mask_indptr_buf = mask_indptr.to(
                    self.device, non_blocking=non_blocking
                )

        self._cached_q_data_type = q_data_type
        self._cached_kv_data_type = kv_data_type
        kv_len_arr = kv_indptr_host[1:] - kv_indptr_host[:-1]

        self._prefix_len_ptr = prefix_len_ptr
        self._token_pos_in_items_ptr = token_pos_in_items_ptr
        self._token_pos_in_items_len = token_pos_in_items_len
        self._max_item_len_ptr = max_item_len_ptr

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
                head_dim_qk,
                head_dim_vo,
                PosEncodingMode[pos_encoding_mode].value,
                window_left >= 0,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )
            if self._backend == "cutlass":
                self._cached_module = get_fmha_module(*get_module_args)
            else:
                self._cached_module = get_batch_prefill_module(
                    self._backend, *get_module_args
                )

        if self._backend == "cutlass":
            self._plan_info = fmha_varlen_plan(
                self._cached_module, qo_indptr, kv_indptr, num_qo_heads, causal
            )
            self._max_qo_len = torch.max(qo_indptr[1:] - qo_indptr[:-1]).item()
        else:
            assert self._cached_module is not None, "cached module is not initialized"
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
                head_dim_qk,
                head_dim_vo,
                causal,
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
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        enable_pdl: Optional[bool] = None,
    ) -> torch.Tensor: ...

    @overload
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        enable_pdl: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute batch prefill/append attention between query and kv-cache stored as
        ragged tensor.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_qk]``
        k : torch.Tensor
            The key tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim_qk]``
        v : torch.Tensor
            The value tensor, shape: ``[kv_indptr[-1], num_kv_heads, head_dim_vo]``
        *args
            Additional arguments for the custom kernel.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the logsumexp of attention output
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_vo]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[qo_indptr[-1], num_qo_heads, head_dim_vo]``.
            * The logsumexp of attention output, shape: ``[qo_indptr[-1], num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)
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
        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                _check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )
        if out is None:
            out = torch.empty(
                q.shape[:-1] + v.shape[-1:], dtype=q.dtype, device=q.device
            )
        else:
            _check_shape_dtype_device(
                out, q.shape[:-1] + v.shape[-1:], q.dtype, q.device, "out"
            )
        if self._backend == "cutlass":
            out, lse = fmha_varlen(
                q,
                k,
                v,
                self._qo_indptr_buf,
                self._kv_indptr_buf,
                plan_info=self._plan_info,
                causal=self._causal,
                sm_scale=sm_scale,
                max_qo_len=self._max_qo_len,
                out=out,
                lse=lse,
            )
            return (out, lse) if return_lse else out

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
            enable_pdl,
        ]
        if self._jit_module is not None:
            run_args.extend(list(args))
        else:
            run_args += [
                self._custom_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                self._prefix_len_ptr,
                self._token_pos_in_items_ptr,
                self._max_item_len_ptr,
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
                self._token_pos_in_items_len,
            ]

        assert self._cached_module is not None, "cached module is not initialized"
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


def fmha_varlen_plan(
    module,
    qo_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor,
    num_qo_heads: int,
    causal: bool,
):
    num_ctas = torch.cuda.get_device_properties(
        qo_segment_offsets.device
    ).multi_processor_count
    work_indptr = torch.empty(
        num_ctas + 1, device=qo_segment_offsets.device, dtype=torch.int32
    )
    qo_tile_indices = torch.empty(
        131072, device=qo_segment_offsets.device, dtype=torch.int32
    )
    head_indices = torch.empty(
        131072, device=qo_segment_offsets.device, dtype=torch.int32
    )
    batch_indices = torch.empty(
        131072, device=qo_segment_offsets.device, dtype=torch.int32
    )
    module.plan(
        qo_segment_offsets,
        kv_segment_offsets,
        work_indptr,
        qo_tile_indices,
        head_indices,
        batch_indices,
        256,  # qo_tile_size
        num_qo_heads,
        num_ctas,
        causal,
    )
    return (
        work_indptr,
        qo_tile_indices,
        head_indices,
        batch_indices,
    )


@overload
def fmha_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor,
    plan_info: Optional[List[torch.Tensor]] = None,
    max_qo_len: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def fmha_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor,
    plan_info: Optional[List[torch.Tensor]] = None,
    max_qo_len: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: Literal[True] = True,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def fmha_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor,
    plan_info: Optional[List[torch.Tensor]] = None,
    max_qo_len: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    workspace_buffer = _get_cache_buf(
        "fmha_varlen_cutlass_workspace", 32 * 1024 * 1024, q.device
    )
    module = get_fmha_module(
        q.dtype,
        k.dtype,
        v.dtype,
        torch.int32,
        q.shape[2],
        v.shape[2],
        PosEncodingMode.NONE.value,
        False,  # use_sliding_window
        False,  # use_logits_soft_cap
    )

    nnz_qo, num_qo_heads, head_dim_qk = q.shape
    nnz_kv, num_kv_heads, head_dim_vo = v.shape

    mask_mode_code = 1 if causal else 0
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_qk)

    qo_total_len = nnz_qo
    if max_qo_len is None:
        max_qo_len = torch.max(qo_segment_offsets[1:] - qo_segment_offsets[:-1]).item()

    if plan_info is None:
        plan_info = fmha_varlen_plan(
            module, qo_segment_offsets, kv_segment_offsets, num_qo_heads, causal
        )

    (
        work_indptr,
        qo_tile_indices,
        head_indices,
        batch_indices,
    ) = plan_info

    if out is None:
        out = torch.empty(
            qo_total_len + max(max_qo_len, 128),
            num_qo_heads,
            head_dim_vo,
            device=q.device,
            dtype=q.dtype,
        )[max(max_qo_len, 128) :]

    if lse is None and return_lse:
        lse = torch.empty(
            qo_total_len, num_qo_heads, device=q.device, dtype=torch.float32
        )

    module.run(
        workspace_buffer,
        q,
        k,
        v,
        qo_segment_offsets,
        kv_segment_offsets,
        work_indptr,
        qo_tile_indices,
        head_indices,
        batch_indices,
        out,
        lse,
        mask_mode_code,
        sm_scale,
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        max_qo_len,
    )

    return out, lse


@functools.cache
def get_trtllm_gen_fmha_module():
    mod = trtllm_gen_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    setup_metainfo_loader(mod.get_library_path())
    return op


def trtllm_ragged_attention_deepseek(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    workspace_buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: float,
    bmm2_scale: float,
    o_sf_scale: float,
    batch_size: int,
    window_left: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    enable_pdl: bool,
    is_causal: bool,
    return_lse: bool,
    attention_sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Parameters
    ----------
    query : torch.Tensor
        query tensor with shape [num_tokens, num_heads, head_dim]
    key : torch.Tensor
        key tensor with shape [num_tokens, num_heads, head_dim]
    value : torch.Tensor
        value tensor with shape [num_tokens, num_heads, head_dim]
    workspace_buffer : torch.Tensor
        workspace buffer
    seq_lens : torch.Tensor
        sequence lengths
    max_q_len : int
        max query length
    max_kv_len : int
        max key/value length
    bmm1_scale : float
        scale for bmm1, scale_q * scale_k * 1.0 / (head_dim_qk ** 0.5)
    bmm2_scale : float
        scale for bmm2, scale_v
    o_sf_scale : float
        scale for output
    batch_size : int
        batch size
    window_left : int
        window left
    cum_seq_lens_q : torch.Tensor
        cumulative sequence lengths for query
    cum_seq_lens_kv : torch.Tensor
        cumulative sequence lengths for key/value
    enable_pdl : bool
        enable pdl
    is_causal : bool
        is causal
    attention_sinks : Optional[torch.Tensor]
        attention sinks
    out : Optional[torch.Tensor]
        output tensor, if not provided, will be allocated with shape [query.shape[0], query.shape[1], value.shape[2]]
    lse : Optional[torch.Tensor]
        lse tensor, if not provided, will be allocated with shape [query.shape[0], query.shape[1]]

    Returns
    -------
    out: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        output torch.Tensor or Tuple[torch.Tensor, torch.Tensor].
        If return_lse is True, the output will be a tuple of two tensors, the first is the output tensor, the second is the lse tensor.
        If return_lse is False, the output will be a single tensor.
    """
    assert query.shape[2] == 192 and key.shape[2] == 192 and value.shape[2] == 128, (
        "currently only support deepseek r1 192 query and 128 value"
    )

    if enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)

    run_func = get_trtllm_gen_fmha_module().trtllm_ragged_attention
    sm_count = get_device_sm_count(query.device)
    if out is None:
        out = torch.empty(
            query.shape[0],
            query.shape[1],
            value.shape[2],
            device=query.device,
            dtype=query.dtype,
        )
    if return_lse and lse is None:
        lse = torch.empty(
            query.shape[0],
            query.shape[1],
            device=query.device,
            dtype=torch.float32,
        )

    run_func(
        out,
        query,
        key,
        value,
        workspace_buffer,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        o_sf_scale,
        batch_size,
        window_left,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        sm_count,
        enable_pdl,
        is_causal,
        attention_sinks,
        lse,
    )
    if return_lse:
        return out, lse
    else:
        return out


def trtllm_batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: float,
    bmm2_scale: float,
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    window_left: int = -1,
    out: Optional[Union[torch.Tensor, FP4Tensor]] = None,
    out_dtype: Optional[Union[torch.dtype, str]] = None,
    o_sf_scale: Optional[float] = None,
    o_sf_vec_size: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
    sinks: Optional[List[torch.Tensor]] = None,
) -> Union[torch.Tensor, FP4Tensor]:
    """
    Parameters
    ----------
    query : torch.Tensor
        query tensor with shape [num_tokens, num_heads, head_dim]
    kv_cache : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        If kv_cache is a single tensor, it should be a tensor with shape [num_pages, 1 or 2, num_kv_heads, page_size, head_dim]
        If kv_cache is a tuple of two tensors, it should be a tuple of two tensors with shape [num_pages, num_kv_heads, page_size, head_dim]
    workspace_buffer : torch.Tensor. Must be initialized to 0 for its first use.
        workspace
    block_tables : torch.Tensor
        page_table of kv cache, [batch_size, num_pages]
    seq_lens : torch.Tensor
        A uint32 1D tensor indicating the kv sequence length of each prompt. shape: ``[batch_size]``
    max_q_len : int
        max sequence length for query
    max_kv_len : int
        max sequence length for kv_cache
    bmm1_scale : float
        fused scale for bmm1 input.
    bmm2_scale : float
        fused scale for bmm2 input.
    batch_size : int
        batch size
    cum_seq_lens_q : torch.Tensor
        cumulative sequence length for query. shape: ``[batch_size + 1]``
    cum_seq_lens_kv : torch.Tensor
        cumulative sequence length for kv_cache. shape: ``[batch_size + 1]``
    window_left : int = -1
        The left (inclusive) window size for the attention window, when set to ``-1``, the window
        size will be set to the full length of the sequence. Defaults to ``-1``.
    out : Optional[Union[torch.Tensor, FP4Tensor]] = None
        output tensor, if not provided, will be allocated with ``out_dtype``, if ``out_dtype`` is not provided, will use the type of ``query``.
    out_dtype : Optional[Union[torch.dtype, str]] = None
        output dtype, if not provided, will use the type of ``out``. For nvfp4, use string ``nvfp4``.
    o_sf_scale : Optional[float] = None
        scale for nvfp4 output tensor scale factor.
    o_sf_vec_size : Optional[int] = None
        vector size for nvfp4 output tensor scale factor.
    sinks : Optional[List[torch.Tensor]] = None
        additional value per head in the denominator of the softmax.

    Returns
    -------
    out: Union[torch.Tensor, FP4Tensor]
        output torch.Tensor or FP4Tensor.
    """

    if enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)

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

    run_func = get_trtllm_gen_fmha_module().trtllm_paged_attention_context
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

        assert isinstance(out, torch.Tensor)

        # Use uint8 as the container dtype to compliant with next fp4 gemm.
        _check_shape_dtype_device(out, fp4_out_shape, torch.uint8, query.device, "out")

        _check_shape_dtype_device(
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
        out_dtype = out_dtype or query.dtype
        out = out if out is not None else torch.empty_like(query, dtype=out_dtype)
        _check_shape_dtype_device(out, query.shape, query.dtype, query.device, "out")
    else:
        raise ValueError(f"Invalid out_dtype: {out_dtype}")

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
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        o_sf_scale or -1.0,
        o_sf_vec_size or -1,
        o_sf_start_index,
        batch_size,
        window_left,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        sm_count,
        enable_pdl,
        sinks,
    )
    return (
        out
        if out_dtype != "nvfp4"
        else FP4Tensor(out, out_scale_factor, o_sf_start_index, query.shape)
    )
