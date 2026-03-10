# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CuTe DSL MLA Decode Kernel Integration
=======================================

Wraps NVIDIA's CuTe DSL MLA decode kernels (FP16/FP8) for Blackwell SM100
and exposes them via a PyTorch API compatible with FlashInfer's MLA backend.
"""

import functools
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .mla_decode_fp16 import BlackwellMultiHeadLatentAttentionForwardFP16
from .mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8
from .utils import get_num_sm


# Default kernel configuration — matches DeepSeek-V2/V3 MLA dimensions
_LATENT_DIM = 512
_ROPE_DIM = 64
_MMA_QK_TILER_MN = (128, 128)
_MMA_PV_TILER_MN = (128, 256)
_MAX_ACTIVE_CLUSTERS = 2
_SKIP_CORRECTION_THRESHOLD = 0.0


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    max_seq_len: int,
    H: int,
    max_active_blocks: int,
) -> Tuple[int, int]:
    """Cache split_kv and workspace_size since they are deterministic for the same params."""
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv(
        B, q_len, max_seq_len, _MMA_QK_TILER_MN, max_active_blocks
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
        H, q_len, _LATENT_DIM, B, split_kv, cutlass.Float32
    )
    return split_kv, workspace_size


@functools.cache
def _get_compiled_mla_kernel(
    is_fp8: bool,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
) -> Callable:
    """Compile and cache an MLA decode kernel.

    Returns a callable that accepts (q_latent, q_rope, c_latent, c_rope,
    page_table, o, lse, workspace, split_kv_scalar, cache_seqs,
    block_split_kvs, softmax_scale_scalar, output_scale_scalar).

    All scalar arguments must be pre-wrapped as Int32/Float32.
    """
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )

    kernel_obj = KernelClass(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=_MMA_QK_TILER_MN,
        mma_pv_tiler_mn=_MMA_PV_TILER_MN,
        max_active_clusters=_MAX_ACTIVE_CLUSTERS,
        page_size=page_size,
        skip_correction_threshold=_SKIP_CORRECTION_THRESHOLD,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
    )

    cutlass_dtype = cutlass.Float8E4M3FN if is_fp8 else cutlass.Float16

    # All dimensions as sym_int — this matches the original kernel's use of
    # mark_compact_shape_dynamic, which makes ALL shapes dynamic CuTe Integers.
    # Static Python ints would cause cute.assume() to fail with AttributeError
    # inside initialize_workspace() since it expects DSL Integer types.
    sym_heads = cute.sym_int()
    sym_latent = cute.sym_int()
    sym_seq_q = cute.sym_int()
    sym_rope = cute.sym_int()
    sym_batch = cute.sym_int()  # query/output batch dimension
    sym_kv_batch = cute.sym_int()  # KV cache batch dim (flat pool, =1 in paged mode)
    sym_seq_kv = cute.sym_int()
    sym_page_count = cute.sym_int()
    sym_workspace_size = cute.sym_int()

    # q_latent: [num_heads, latent_dim, seq_len_q, batch_size] — stride[1]==1
    q_latent_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_heads, sym_latent, sym_seq_q, sym_batch),
        stride_order=(3, 0, 2, 1),
        assumed_align=128,
    )
    # q_rope: [num_heads, rope_dim, seq_len_q, batch_size] — stride[1]==1
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_heads, sym_rope, sym_seq_q, sym_batch),
        stride_order=(3, 0, 2, 1),
        assumed_align=128,
    )
    # c_latent: [seq_len_k, latent_dim, kv_batch] — stride[1]==1
    # kv_batch is a separate sym_int from query batch: paged KV cache uses a flat
    # pool so kv_batch=1 at runtime, while query batch can be any value.
    c_latent_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_seq_kv, sym_latent, sym_kv_batch),
        stride_order=(2, 0, 1),
        assumed_align=128,
    )
    # c_rope: [seq_len_k, rope_dim, kv_batch] — stride[1]==1
    c_rope_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_seq_kv, sym_rope, sym_kv_batch),
        stride_order=(2, 0, 1),
        assumed_align=128,
    )
    # page_table: [page_count, batch_size] with stride[0]==1
    # Matches the original kernel's convention: page_table_ref.permute(1, 0) gives
    # strides (1, page_count), so dim0(page_count) is the contiguous dimension.
    # This allows passing block_tables.t() directly without .contiguous().
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_page_count, sym_batch),
        stride_order=(0, 1),
        assumed_align=128,
    )
    # o: [num_heads, latent_dim, seq_len_q, batch_size] — stride[1]==1
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_heads, sym_latent, sym_seq_q, sym_batch),
        stride_order=(3, 0, 2, 1),
        assumed_align=128,
    )
    # lse: [num_heads, seq_len_q, batch_size] — stride[0]==1 (num_heads dim is contiguous)
    # stride_order[d]=rank: dim0 rank=0 means dim0 is fastest → stride[0]=1 compile-time constant
    lse_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_heads, sym_seq_q, sym_batch),
        stride_order=(0, 1, 2),
        assumed_align=128,
    )
    # workspace: 1-D
    workspace_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_workspace_size,),
        assumed_align=128,
    )
    # cache_seqs: [batch_size] — int32
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=128,
    )
    # block_split_kvs: [batch_size] — int32
    block_split_kvs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=128,
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        workspace_fake,
        Int32(1),  # split_kv placeholder
        cache_seqs_fake,
        block_split_kvs_fake,
        Float32(1.0),  # softmax_scale placeholder
        Float32(1.0),  # output_scale placeholder
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


def cute_dsl_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CuTe DSL MLA decode kernel for Blackwell SM100.

    Parameters
    ----------
    query : torch.Tensor
        [B, q_len, H, D_qk] where D_qk = kv_lora_rank + qk_rope_head_dim
    kv_cache : torch.Tensor
        [num_pages, page_size, D_ckv + D_kpe] (3D) or [num_pages, 1, page_size, D_ckv + D_kpe] (4D)
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer.
    kv_lora_rank : int
        Latent dimension (e.g. 512).
    qk_rope_head_dim : int
        RoPE dimension (e.g. 64).
    block_tables : torch.Tensor
        [B, max_pages] — page table indices.
    seq_lens : torch.Tensor
        [B] — per-request KV sequence lengths.
    max_seq_len : int
        Maximum sequence length across the batch.
    softmax_scale : float
        Scale factor for QK^T before softmax.
    output_scale : float
        Scale factor applied to the output.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor [B, H, kv_lora_rank].

    Returns
    -------
    torch.Tensor
        Output tensor [B, H, kv_lora_rank].
    """
    assert query.dtype in (
        torch.float16,
        torch.float8_e4m3fn,
    ), f"cute_dsl_mla_decode only supports float16 and float8_e4m3fn, got {query.dtype}"
    assert kv_cache.dtype == query.dtype, (
        f"kv_cache dtype {kv_cache.dtype} must match query dtype {query.dtype}"
    )
    B, q_len, H, D_qk = query.shape
    assert D_qk == kv_lora_rank + qk_rope_head_dim
    assert kv_lora_rank == _LATENT_DIM
    assert qk_rope_head_dim == _ROPE_DIM

    is_fp8 = query.dtype == torch.float8_e4m3fn

    # Handle 3D vs 4D kv_cache: normalize to 3D [num_pages, page_size, D_total]
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]

    # Split query into latent and rope components and reshape to kernel layout.
    # [B, q_len, H, D] -> slice -> permute -> [H, D, q_len, B] with stride[1]=1.
    # Do NOT call .contiguous() — permute gives stride[1]=1 which the kernel requires.
    q_latent_k = query[..., :kv_lora_rank].permute(2, 3, 1, 0)
    q_rope_k = query[..., kv_lora_rank:].permute(2, 3, 1, 0)

    # Reshape KV cache to kernel layout [page_size, D, num_pages].
    # The kernel indexes via page_table: c_latent[intra_page_offset, d, physical_page_idx].
    # After permute: strides = (D_total, 1, page_size*D_total) → stride[1]=1 ✓
    c_latent_k = kv_cache[:, :, :kv_lora_rank].permute(1, 2, 0)
    c_rope_k = kv_cache[:, :, kv_lora_rank:].permute(1, 2, 0)

    # Page table: [B, max_pages] -> [max_pages, B] (view only, no copy needed).
    # The kernel accepts non-contiguous strides via CuTe layout, matching the original
    # kernel's convention of page_table_ref.permute(1, 0) without .contiguous().
    page_table_k = block_tables.permute(1, 0)

    # Cached split_kv and workspace_size computation
    max_active_blocks = get_num_sm(query.device)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B, q_len, max_seq_len, H, max_active_blocks
    )

    # Prepare workspace — slice of contiguous 1D buffer is already contiguous
    workspace_bytes = workspace_buffer[: max(workspace_size, 1)]

    # Output buffer setup: kernel needs [H, D, q_len, B] with stride[1]==1.
    # If caller provides `out`, reuse it directly via permute to avoid allocation + copy_.
    #   q_len==1: out [B, H, D] → permute(1,2,0) → [H, D, B] → unsqueeze(2) → [H, D, 1, B]
    #   q_len >1: out [B, q_len, H, D] → permute(2,3,1,0) → [H, D, q_len, B]
    # Both give stride[1]=1 ✓, kernel writes directly into out's memory.
    out_dtype = torch.float8_e4m3fn if is_fp8 else torch.float16
    if out is not None:
        if q_len == 1:
            o_k = out.permute(1, 2, 0).unsqueeze(2)
        else:
            o_k = out.permute(2, 3, 1, 0)
    else:
        # Allocate as [B, q_len, H, D] so that permute back is already contiguous.
        # permute(2, 3, 1, 0) → [H, D, q_len, B] with stride[1]=1 ✓
        o_k = torch.empty(
            (B, q_len, H, _LATENT_DIM), dtype=out_dtype, device=query.device
        ).permute(2, 3, 1, 0)

    # LSE: [H, q_len, B] with stride[0]==1 (H dim is contiguous)
    lse_k = torch.empty(
        (B, q_len, H), dtype=torch.float32, device=query.device
    ).permute(2, 1, 0)

    # cache_seqs: per-batch sequence lengths (skip .to() if already int32)
    cache_seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    
    # TOOD: this will trigger a kernel.
    # block_split_kvs: uniform split_kv for all batches
    block_split_kvs = torch.full((B,), split_kv, dtype=torch.int32, device=query.device)

    # Get compiled kernel (cached after first compile)
    compiled_kernel = _get_compiled_mla_kernel(
        is_fp8=is_fp8,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        is_persistent=True,
        is_var_seq=True,
        is_var_split_kv=True,
    )

    # Call the kernel
    compiled_kernel(
        q_latent_k,
        q_rope_k,
        c_latent_k,
        c_rope_k,
        page_table_k,
        o_k,
        lse_k,
        workspace_bytes,
        Int32(split_kv),
        cache_seqs,
        block_split_kvs,
        Float32(softmax_scale),
        Float32(output_scale),
    )

    # If out was provided, kernel already wrote into it — return directly.
    if out is not None:
        return out

    # No out provided: reshape kernel output [H, D, q_len, B] -> [B, (q_len,) H, D]
    # The permute back is always contiguous because we allocated as [B, q_len, H, D].
    result = o_k.permute(3, 2, 0, 1)
    if q_len == 1:
        result = result.squeeze(1)

    return result
