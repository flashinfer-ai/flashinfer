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
from typing import Optional

import torch

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm90a_nvcc_flags
from .jit.utils import filename_safe_dtype_map
from .utils import (
    get_device_sm_count,
    register_custom_op,
    register_fake_op,
)

xqa_nvcc_flags = [
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DCACHE_ELEM_ENUM=0",
    "-DUSE_CUSTOM_BARRIER=1",
    "-DLOW_PREC_OUTPUT=0",
    "-DSPEC_DEC=0",
]


def gen_xqa_module(
    dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool,
) -> JitSpec:
    if dtype == torch.float16:
        flag_dtype = ["-DINPUT_FP16=1", "-DDTYPE=__half"]
    elif dtype == torch.bfloat16:
        flag_dtype = ["-DINPUT_FP16=0", "-DDTYPE=__nv_bfloat16"]
    else:
        raise ValueError(
            f"Invalid dtype: {dtype} for XQA, only float16 and bfloat16 are supported"
        )

    if page_size not in [16, 32, 64, 128]:
        raise ValueError(
            f"Invalid page_size: {page_size}, only 16, 32, 64, 128 are supported"
        )
    flag_tokens_per_page = [f"-DTOKENS_PER_PAGE={page_size}"]

    if head_dim % 16 != 0 or head_dim > 256 or head_dim < 16:
        raise ValueError(
            f"Invalid head_dim: {head_dim}, must be divisible by 16 and in range [16, 256]"
        )
    flag_head_dim = [f"-DHEAD_ELEMS={head_dim}"]

    flag_head_group_ratio = [f"-DHEAD_GRP_SIZE={head_group_ratio}"]

    if use_sliding_window:
        flag_sliding_window = ["-DSLIDING_WINDOW=1"]
    else:
        flag_sliding_window = ["-DSLIDING_WINDOW=0"]

    return gen_jit_spec(
        f"xqa_dtype_{filename_safe_dtype_map[dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}",
        [
            jit_env.FLASHINFER_CSRC_DIR / "xqa/mha.cu",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/xqa_wrapper.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_xqa_ops.cu",
        ],
        extra_cuda_cflags=xqa_nvcc_flags
        + sm90a_nvcc_flags
        + flag_tokens_per_page
        + flag_head_dim
        + flag_dtype
        + flag_head_group_ratio
        + flag_sliding_window,
    )


@functools.cache
def get_xqa_module(
    dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool,
):
    module = gen_xqa_module(
        dtype, page_size, head_dim, head_group_ratio, use_sliding_window
    ).build_and_load()

    @register_custom_op(
        f"flashinfer::xqa_dtype_{filename_safe_dtype_map[dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}",
        mutates_args=("output", "workspace_buffer"),
    )
    def xqa(
        sm_count: int,
        num_kv_heads: int,
        slidingWinSize: int,
        q_scale: float,
        output: torch.Tensor,
        q: torch.Tensor,
        sinks: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        batch_size: int,
        kv_scale: torch.Tensor,
        semaphores: torch.Tensor,
        workspace_buffer: torch.Tensor,
    ) -> None:
        module.xqa_wrapper.default(
            sm_count,
            num_kv_heads,
            slidingWinSize,
            q_scale,
            output,
            q,
            sinks,
            kv_cache,
            page_table,
            max_seq_len,
            seq_lens,
            batch_size,
            kv_scale,
            semaphores,
            workspace_buffer,
        )

    @register_fake_op(
        f"flashinfer::xqa_dtype_{filename_safe_dtype_map[dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}"
    )
    def _fake_xqa(
        sm_count: int,
        num_kv_heads: int,
        slidingWinSize: int,
        q_scale: float,
        output: torch.Tensor,
        q: torch.Tensor,
        sinks: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table: torch.Tensor,
        max_seq_len: int,
        seq_lens: torch.Tensor,
        batch_size: int,
        kv_scale: torch.Tensor,
        semaphores: torch.Tensor,
        workspace_buffer: torch.Tensor,
    ) -> None:
        pass

    return SimpleNamespace(
        xqa=xqa,
    )


def xqa(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
    semaphores: torch.Tensor,
    num_kv_heads: int,
    page_size: int,
    sinks: Optional[torch.Tensor] = None,
    q_scale: float = 1.0,
    kv_scale: Optional[torch.Tensor] = None,
    sliding_win_size: int = 0,
    sm_count: Optional[int] = None,
) -> None:
    r"""Apply attention with paged KV cache using XQA kernel.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor with shape ``[batch_size, beam_width, num_q_heads, head_dim]``.
        Data type should be torch.float16 or torch.bfloat16.

    kv_cache : torch.Tensor
        Paged KV cache tensor with shape ``[total_num_cache_heads, head_dim]``.
        Contains both K and V cache data interleaved.
        Data type should match query tensor.

    page_table : torch.Tensor
        Page table tensor with shape ``[batch_size, beam_width, 2, num_pages_per_seq]``.
        Data type should be torch.uint32.
        The third dimension represents K and V cache (0 for K, 1 for V).

    seq_lens : torch.Tensor
        Sequence lengths tensor with shape ``[batch_size, beam_width]``.
        Data type should be torch.uint32.

    output : torch.Tensor
        Output tensor with shape ``[batch_size, beam_width, num_q_heads, head_dim]``.
        Data type should match query tensor. This tensor will be modified in-place.

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

    q_scale : float, default=1.0
        Scale factor for query tensor.

    kv_scale : Optional[torch.Tensor], default=None
        Scale factor for KV cache with shape ``[1]``.
        Data type should be torch.float32.
        If None, defaults to 1.0.

    sliding_win_size : int, default=0
        Sliding window size for attention. If 0, no sliding window is used.

    sm_count : Optional[int], default=None
        Number of streaming multiprocessors to use.
        If None, will be inferred from the device.

    Note
    ----
    The function automatically infers several parameters from tensor shapes:
    - batch_size from q.shape[0]
    - num_q_heads from q.shape[2]
    - head_dim from q.shape[-1]
    - use_fp16 from q.dtype
    - head_group_ratio from num_q_heads // num_kv_heads
    """
    # Handle optional parameters
    if sm_count is None:
        sm_count = get_device_sm_count(q.device)

    if kv_scale is None:
        kv_scale = torch.ones(1, dtype=torch.float32, device=q.device)

    # Infer parameters from tensors
    batch_size = q.shape[0]
    num_q_heads = q.shape[2]
    head_dim = q.shape[-1]

    # Calculate head_group_ratio
    head_group_ratio = num_q_heads // num_kv_heads

    # Calculate max_seq_len from page_table and page_size
    num_pages_per_seq = page_table.shape[-1]
    max_seq_len = num_pages_per_seq * page_size

    # Determine if sliding window is used
    use_sliding_window = sliding_win_size >= 0

    xqa_module = get_xqa_module(
        q.dtype, page_size, head_dim, head_group_ratio, use_sliding_window
    )
    xqa_module.xqa(
        sm_count,
        num_kv_heads,
        sliding_win_size if use_sliding_window else 0,
        q_scale,
        output,
        q,
        sinks,
        kv_cache,
        page_table,
        max_seq_len,
        seq_lens,
        batch_size,
        kv_scale,
        semaphores,
        workspace_buffer,
    )
