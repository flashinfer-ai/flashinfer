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
def _get_compiled_mla_kernel(
    is_fp8: bool,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
) -> Tuple[Callable, object]:
    """Compile and cache an MLA decode kernel.

    Returns (compiled_kernel_closure, kernel_class_instance).
    The kernel_class_instance is needed for get_split_kv() and get_workspace_size().
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
    # page_table: [page_count, batch_size]
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_page_count, sym_batch),
        stride_order=(1, 0),
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

    def tensor_api(
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        c_latent: torch.Tensor,
        c_rope: torch.Tensor,
        page_table: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        workspace: torch.Tensor,
        split_kv: int,
        cache_seqs: torch.Tensor,
        block_split_kvs: torch.Tensor,
        softmax_scale: float,
        output_scale: float,
    ) -> None:
        nonlocal compiled_kernel
        compiled_kernel(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            o,
            lse,
            workspace,
            Int32(split_kv),
            cache_seqs,
            block_split_kvs,
            Float32(softmax_scale),
            Float32(output_scale),
        )

    return tensor_api, kernel_obj


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
        # [num_pages, 1, page_size, D_total] -> [num_pages, page_size, D_total]
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]
    D_total = kv_cache.shape[2]
    assert D_total == kv_lora_rank + qk_rope_head_dim

    # Split query into latent and rope components
    q_nope = query[..., :kv_lora_rank]  # [B, q_len, H, latent_dim]
    q_rope = query[..., kv_lora_rank:]  # [B, q_len, H, rope_dim]

    # Reshape to kernel layout: [B, q_len, H, D] -> [H, D, q_len, B]
    # Do NOT call .contiguous() — permute gives stride[1]=1 which the kernel requires.
    # .contiguous() would rearrange to row-major making stride[3]=1 instead.
    q_latent_k = q_nope.permute(2, 3, 1, 0)  # [H, latent_dim, q_len, B], stride[1]=1
    q_rope_k = q_rope.permute(2, 3, 1, 0)  # [H, rope_dim, q_len, B], stride[1]=1

    # Reshape KV cache to kernel layout [page_size, D, num_pages].
    # The kernel indexes via page_table: for batch b, page p, offset t:
    #   c_latent[t, d, page_table[p, b]] = token (page_table[p,b]*page_size + t)'s latent[d]
    # kv_cache: [num_pages, page_size, D_total] with strides (page_size*D_total, D_total, 1)
    # After permute(1, 2, 0) on latent slice: [page_size, latent_dim, num_pages]
    #   strides = (D_total, 1, page_size*D_total) → stride[1]=1 ✓
    c_latent_k = kv_cache[:, :, :kv_lora_rank].permute(
        1, 2, 0
    )  # [page_size, latent_dim, num_pages]
    c_rope_k = kv_cache[:, :, kv_lora_rank:].permute(
        1, 2, 0
    )  # [page_size, rope_dim, num_pages]

    # Page table: [B, max_pages] -> [max_pages, B]
    page_table_k = block_tables.t().contiguous().to(torch.int32)

    # Determine split_kv and workspace
    is_persistent = True
    is_var_seq = True
    is_var_split_kv = True
    max_active_blocks = get_num_sm(query.device)

    split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv(
        B, q_len, max_seq_len, _MMA_QK_TILER_MN, max_active_blocks
    )

    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
        H, q_len, _LATENT_DIM, B, split_kv, cutlass.Float32
    )

    # Prepare workspace tensor
    if workspace_size > 0:
        workspace_bytes = workspace_buffer[:workspace_size].contiguous()
    else:
        workspace_bytes = workspace_buffer[:1].contiguous()

    # Allocate output: [H, latent_dim, q_len, B] with stride[1]==1
    # torch.empty(B, H, q_len, D) has row-major strides (H*q_len*D, q_len*D, D, 1).
    # After permute(1, 3, 2, 0) → shape [H, D, q_len, B] with strides (q_len*D, 1, D, H*q_len*D).
    # Do NOT call .contiguous() — that would collapse to row-major making stride[3]=1.
    out_dtype = torch.float8_e4m3fn if is_fp8 else torch.float16
    o_k = torch.empty(
        (B, H, q_len, _LATENT_DIM), dtype=out_dtype, device=query.device
    ).permute(1, 3, 2, 0)  # [H, latent_dim, q_len, B], stride[1]=1

    # LSE: [H, q_len, B] with stride[0]==1 (H dim is contiguous).
    # torch.empty(B, q_len, H) has row-major strides (q_len*H, H, 1).
    # After permute(2, 1, 0) → shape [H, q_len, B] with strides (1, H, q_len*H).
    # Do NOT call .contiguous() — that would make stride[2]=1 instead of stride[0]=1.
    lse_k = torch.empty(
        (B, q_len, H), dtype=torch.float32, device=query.device
    ).permute(2, 1, 0)  # [H, q_len, B], stride[0]=1

    # cache_seqs: per-batch sequence lengths
    cache_seqs = seq_lens.to(torch.int32).contiguous()

    # block_split_kvs: per-batch split_kv values
    # Compute per-batch split_kv based on actual sequence lengths
    block_split_kvs = torch.ones(B, dtype=torch.int32, device=query.device) * split_kv

    # Get compiled kernel
    tensor_api, kernel_cls = _get_compiled_mla_kernel(
        is_fp8=is_fp8,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
    )

    # Call the kernel
    tensor_api(
        q_latent_k,
        q_rope_k,
        c_latent_k,
        c_rope_k,
        page_table_k,
        o_k,
        lse_k,
        workspace_bytes.view(torch.uint8),
        split_kv,
        cache_seqs,
        block_split_kvs,
        softmax_scale,
        output_scale,
    )

    # Reshape output: [H, latent_dim, q_len, B] -> [B, q_len, H, latent_dim]
    result = o_k.permute(3, 2, 0, 1).contiguous()

    # Squeeze q_len dimension if it's 1: [B, 1, H, D] -> [B, H, D]
    if q_len == 1:
        result = result.squeeze(1)

    if out is not None:
        out.copy_(result)
        return out

    return result
