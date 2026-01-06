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
from typing import Optional, Union
import torch

from .api_logging import flashinfer_api
from .jit.xqa import gen_xqa_module, gen_xqa_module_mla
from .jit.utils import filename_safe_dtype_map
from .utils import (
    get_device_sm_count,
    register_custom_op,
    register_fake_op,
    get_compute_capability,
    device_support_pdl,
)


@functools.cache
def get_xqa_module(
    input_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool,
    output_dtype: torch.dtype,
    q_seq_len: int,
):
    module = gen_xqa_module(
        input_dtype,
        kv_cache_dtype,
        page_size,
        head_dim,
        head_group_ratio,
        use_sliding_window,
        output_dtype,
        q_seq_len,
    ).build_and_load()

    if q_seq_len > 1:
        use_spec_dec = True
    else:
        use_spec_dec = False

    @register_custom_op(
        f"flashinfer::xqa_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_output_{filename_safe_dtype_map[output_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}_use_spec_dec_{use_spec_dec}_spec_q_seq_len_{q_seq_len}",
        mutates_args=("output", "workspace_buffer"),
    )
    def xqa(
        run_sm90_fp8_mha: bool,
        sm_count: int,
        num_kv_heads: int,
        sliding_win_size: int,
        q_scale: Union[float, torch.Tensor],
        output: torch.Tensor,
        rcp_out_scale: float,
        q: torch.Tensor,
        sinks: Optional[torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        batch_size: int,
        kv_scale: Union[float, torch.Tensor],
        semaphores: torch.Tensor,
        workspace_buffer: torch.Tensor,
        enable_pdl: bool,
        q_seq_len: int,
        mask: Optional[torch.Tensor],
    ) -> None:
        module.xqa_wrapper(
            run_sm90_fp8_mha,
            sm_count,
            num_kv_heads,
            sliding_win_size,
            1.0 if isinstance(q_scale, torch.Tensor) else q_scale,
            None if isinstance(q_scale, float) else q_scale,
            output,
            rcp_out_scale,
            q,
            sinks,
            k_cache,
            v_cache,
            page_table,
            max_seq_len,
            seq_lens,
            batch_size,
            1.0 if isinstance(kv_scale, torch.Tensor) else kv_scale,
            None if isinstance(kv_scale, float) else kv_scale,
            q_seq_len,
            mask,
            semaphores,
            workspace_buffer,
            enable_pdl,
        )

    @register_fake_op(
        f"flashinfer::xqa_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_output_{filename_safe_dtype_map[output_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}_use_spec_dec_{use_spec_dec}_spec_q_seq_len_{q_seq_len}"
    )
    def _fake_xqa(
        run_sm90_fp8_mha: bool,
        sm_count: int,
        num_kv_heads: int,
        sliding_win_size: int,
        q_scale: Union[float, torch.Tensor],
        output: torch.Tensor,
        rcp_out_scale: float,
        q: torch.Tensor,
        sinks: Optional[torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        batch_size: int,
        kv_scale: Union[float, torch.Tensor],
        semaphores: torch.Tensor,
        workspace_buffer: torch.Tensor,
        enable_pdl: bool,
        q_seq_len: int,
        mask: Optional[torch.Tensor],
    ) -> None:
        pass

    return SimpleNamespace(
        xqa=xqa,
    )


@flashinfer_api
def xqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
    semaphores: torch.Tensor,
    num_kv_heads: int,
    page_size: int,
    sinks: Optional[torch.Tensor] = None,
    q_scale: Union[float, torch.Tensor] = 1.0,
    kv_scale: Union[float, torch.Tensor] = 1.0,
    sliding_win_size: int = 0,
    kv_layout: str = "NHD",
    sm_count: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
    rcp_out_scale: float = 1.0,
    q_seq_len: int = 1,
    mask: Optional[torch.Tensor] = None,
) -> None:
    r"""Apply attention with paged KV cache using XQA kernel.
    Parameters
    ----------
    q : torch.Tensor
        Query tensor with shape ``[batch_size, beam_width, num_q_heads, head_dim]`` if not using speculative decoding,
        or ``[batch_size, beam_width, q_seq_len, num_q_heads, head_dim]`` if using speculative decoding. ``q_seq_len`` is the number of speculative decoding tokens.
        Data type should be torch.float16 or torch.bfloat16.
        Now only beam_width 1 is supported.
    k_cache: torch.Tensor
        Paged K cache tensor with shape ``[num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
        or ``[num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.
        Data type should match query tensor or be torch.float8_e4m3fn, in which case xqa will run fp8 calculation.
        Should be the same data type as v_cache.
    v_cache: torch.Tensor
        Paged V cache tensor with shape ``[num_pages, page_size, num_kv_heads, head_dim]`` if :attr:`kv_layout` is ``NHD``,
        or ``[num_pages, num_kv_heads, page_size, head_dim]`` if :attr:`kv_layout` is ``HND``.
        Data type should match query tensor or be torch.float8_e4m3fn, in which case xqa will run fp8 calculation.
        Should be the same data type as k_cache.
    page_table : torch.Tensor
        Page table tensor with shape ``batch_size, nb_pages_per_seq``.
        Data type should be torch.int32.
        K and V share the same table.
    seq_lens : torch.Tensor
        Sequence lengths tensor with shape ``[batch_size, beam_width]``.
        Data type should be torch.uint32.
    output : torch.Tensor
        Output tensor with shape that matches the query tensor.
        Data type should match query tensor or kv tensor. This tensor will be modified in-place.
    workspace_buffer : torch.Tensor
        Workspace buffer for temporary computations.
        Data type should be torch.uint8.
    semaphores : torch.Tensor
        Semaphore buffer for synchronization.
        Data type should be torch.uint32.
    num_kv_heads : int
        Number of key-value heads in the attention mechanism.
    page_size : int
        Size of each page in the paged KV cache. Must be one of [16, 32, 64, 128].
    sinks : Optional[torch.Tensor], default=None
        Attention sink values with shape ``[num_kv_heads, head_group_ratio]``.
        Data type should be torch.float32.
        If None, no attention sinks are used.
    q_scale : Union[float, torch.Tensor], default=1.0
        Scale factor for query tensor.
    kv_scale : Union[float, torch.Tensor], default=1.0
        Scale factor for KV cache.
    sliding_win_size : int, default=0
        Sliding window size for attention. If 0, no sliding window is used.
    kv_layout : str, default="NHD"
        The layout of the KV cache. Can be either ``NHD`` or ``HND``.
    sm_count : Optional[int], default=None
        Number of streaming multiprocessors to use.
        If None, will be inferred from the device.
    enable_pdl : Optional[bool], default=None
        Whether to enable PDL (Persistent Data Loader) optimization.
        If None, will be set to True if hardware supports it.
    rcp_out_scale : float, default=1.0
        Reciprocal of output scale factor.
    q_seq_len : int, default=1
        Query sequence length. When > 1, enables speculative decoding mode.
    mask : Optional[torch.Tensor], default=None
        Causal attention mask for speculative decoding mode (when ``q_seq_len > 1``).
        Shape: ``[batch_size, q_seq_len, mask_size_per_row]`` where
        ``mask_size_per_row = ((q_seq_len + 31) // 32) * 2``.
        Data type should be torch.uint16 (bit-packed format, aligned to 32 bits).

    Note
    ----
    The function automatically infers several parameters from tensor shapes:
    - batch_size from q.shape[0]
    - num_q_heads from q.shape[-2]
    - head_dim from q.shape[-1]
    - input_dtype from q.dtype
    - kv_cache_dtype from k.dtype
    - head_group_ratio from num_q_heads // num_kv_heads
    - max_seq_len from page_table.shape[-1] * page_size
    """
    # Handle optional parameters
    if sm_count is None:
        sm_count = get_device_sm_count(q.device)

    enable_pdl = enable_pdl if enable_pdl is not None else device_support_pdl(q.device)

    # Infer parameters from tensors
    batch_size = q.shape[0]
    num_q_heads = q.shape[-2]
    head_dim = q.shape[-1]

    # Calculate head_group_ratio
    head_group_ratio = num_q_heads // num_kv_heads

    # Calculate max_seq_len from page_table and page_size
    num_pages_per_seq = page_table.shape[-1]
    max_seq_len = num_pages_per_seq * page_size

    # Determine if sliding window is used
    use_sliding_window = sliding_win_size > 0

    assert k_cache.dtype == v_cache.dtype, "K and V cache must have the same dtype"

    if output.dtype == torch.float8_e4m3fn:
        assert k_cache.dtype == torch.float8_e4m3fn, (
            "KV cache must be fp8 when output is fp8"
        )
    else:
        assert output.dtype == q.dtype, "Output and query must have the same dtype"

    # Convert HND layout to NHD if necessary (transpose only changes stride, not data)
    if kv_layout == "HND":
        # For HND: [..., H, N, D] -> NHD: [..., N, H, D]
        k_cache = k_cache.transpose(-3, -2)
        v_cache = v_cache.transpose(-3, -2)

    if (
        k_cache.dtype == torch.float8_e4m3fn
        and get_compute_capability(torch.device(device="cuda"))[0] == 9
    ):
        run_sm90_fp8_mha = True
    else:
        run_sm90_fp8_mha = False

    if get_compute_capability(torch.device(device="cuda"))[0] not in [9, 10, 12]:
        raise RuntimeError("XQA is only supported on SM90, SM100, SM120 GPUs")

    xqa_module = get_xqa_module(
        q.dtype,
        k_cache.dtype,
        page_size,
        head_dim,
        head_group_ratio,
        use_sliding_window,
        output.dtype,
        q_seq_len,
    )

    if q_seq_len > 1:
        assert mask is not None, "Mask is required for speculative decoding"
        if sinks is not None:
            run_sm90_fp8_mha = False  # TODO: mha_sm90.cu has precision issue if sinks and speculative decoding are used simultaneously

    xqa_module.xqa(
        run_sm90_fp8_mha,
        sm_count,
        num_kv_heads,
        sliding_win_size if use_sliding_window else 0,
        q_scale,
        output,
        rcp_out_scale,
        q,
        sinks,
        k_cache,
        v_cache,
        page_table,
        max_seq_len,
        seq_lens,
        batch_size,
        kv_scale,
        semaphores,
        workspace_buffer,
        enable_pdl,
        q_seq_len,
        mask,
    )


@functools.cache
def get_xqa_module_mla(
    input_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool = False,
):
    module = gen_xqa_module_mla(
        input_dtype,
        kv_cache_dtype,
        page_size,
        head_dim,
        head_group_ratio,
        use_sliding_window,
    ).build_and_load()

    @register_custom_op(
        f"flashinfer::xqa_mla_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}",
        mutates_args=("output", "workspace_buffer"),
    )
    def xqa_mla(
        sm_count: int,
        q_scale: Union[float, torch.Tensor],
        output: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        batch_size: int,
        kv_scale: Union[float, torch.Tensor],
        semaphores: torch.Tensor,
        workspace_buffer: torch.Tensor,
        enable_pdl: bool,
    ) -> None:
        module.xqa_wrapper_mla(
            sm_count,
            1.0 if isinstance(q_scale, torch.Tensor) else q_scale,
            None if isinstance(q_scale, float) else q_scale,
            output,
            q,
            k_cache,
            v_cache,
            page_table,
            max_seq_len,
            seq_lens,
            batch_size,
            1.0 if isinstance(kv_scale, torch.Tensor) else kv_scale,
            None if isinstance(kv_scale, float) else kv_scale,
            semaphores,
            workspace_buffer,
            enable_pdl,
        )

    @register_fake_op(
        f"flashinfer::xqa_mla_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}"
    )
    def _fake_xqa_mla(
        sm_count: int,
        q_scale: Union[float, torch.Tensor],
        output: torch.Tensor,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        batch_size: int,
        kv_scale: Union[float, torch.Tensor],
        semaphores: torch.Tensor,
        workspace_buffer: torch.Tensor,
        enable_pdl: bool,
    ) -> None:
        pass

    return SimpleNamespace(
        xqa_mla=xqa_mla,
    )


@flashinfer_api
def xqa_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
    semaphores: torch.Tensor,
    page_size: int,
    q_scale: Union[float, torch.Tensor] = 1.0,
    kv_scale: Union[float, torch.Tensor] = 1.0,
    sm_count: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Apply attention with paged KV cache using XQA MLA (Multi-Head Latent Attention) kernel.
    Parameters
    ----------
    q : torch.Tensor
        Query tensor with shape ``[batch_size, beam_width, num_q_heads, head_dim]``.
        Data type should be torch.float8_e4m3fn.
        Now only beam_width 1 is supported.
    k_cache: torch.Tensor
        Paged K cache tensor with shape ``[total_num_cache_heads, head_dim]``.
        Data type should be torch.float8_e4m3fn
    v_cache: torch.Tensor
        Paged V cache tensor with shape ``[total_num_cache_heads, head_dim]``.
        Data type should be torch.float8_e4m3fn
    page_table : torch.Tensor
        Page table tensor with shape ``batch_size, nb_pages_per_seq``.
        Data type should be torch.int32.
        K and V share the same table.
    seq_lens : torch.Tensor
        Sequence lengths tensor with shape ``[batch_size, beam_width]``.
        Data type should be torch.uint32.
    output : torch.Tensor
        Output tensor with shape ``[batch_size, beam_width, num_q_heads, head_dim]``.
        Data type should be torch.bfloat16. This tensor will be modified in-place.
    workspace_buffer : torch.Tensor
        Workspace buffer for temporary computations.
        Data type should be torch.uint8.
    semaphores : torch.Tensor
        Semaphore buffer for synchronization.
        Data type should be torch.uint32.
    page_size : int
        Size of each page in the paged KV cache. Must be one of [16, 32, 64, 128].
    q_scale : Union[float, torch.Tensor], default=1.0
        Scale factor for query tensor.
    kv_scale : Union[float, torch.Tensor], default=1.0
        Scale factor for KV cache.
    sm_count : Optional[int], default=None
        Number of streaming multiprocessors to use.
        If None, will be inferred from the device.
    enable_pdl : Optional[bool], default=None
        Whether to enable PDL (Persistent Data Loader) optimization.
        If None, will be set to True if hardware supports it.

    Note
    ----
    The function automatically infers several parameters from tensor shapes:
    - batch_size from q.shape[0]
    - head_dim from q.shape[-1]
    - head_group_ratio is fixed to 128 for MLA
    - max_seq_len from page_table.shape[-1] * page_size
    """
    # Handle optional parameters
    if sm_count is None:
        sm_count = get_device_sm_count(q.device)

    enable_pdl = enable_pdl if enable_pdl is not None else device_support_pdl(q.device)

    # Infer parameters from tensors
    batch_size = q.shape[0]
    head_dim = q.shape[-1]

    # Calculate head_group_ratio
    head_group_ratio = 128

    # Calculate max_seq_len from page_table and page_size
    num_pages_per_seq = page_table.shape[-1]
    max_seq_len = num_pages_per_seq * page_size

    assert k_cache.dtype == v_cache.dtype, "K and V cache must have the same dtype"

    if get_compute_capability(torch.device(device="cuda"))[0] not in [12]:
        raise RuntimeError("XQA MLA is only supported on SM120 GPUs")

    xqa_module = get_xqa_module_mla(
        q.dtype,
        k_cache.dtype,
        page_size,
        head_dim,
        head_group_ratio,
        False,
    )
    xqa_module.xqa_mla(
        sm_count,
        q_scale,
        output,
        q,
        k_cache,
        v_cache,
        page_table,
        max_seq_len,
        seq_lens,
        batch_size,
        kv_scale,
        semaphores,
        workspace_buffer,
        enable_pdl,
    )
