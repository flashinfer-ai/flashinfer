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

Wraps NVIDIA's CuTe DSL MLA decode kernels (FP16/BF16/FP8) for Blackwell SM100
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
from .utils import get_num_sm, torch_to_cutlass_dtype


# Default kernel configuration — matches DeepSeek-V2/V3 MLA dimensions
_LATENT_DIM = 512
_ROPE_DIM = 64
_MMA_QK_TILER_MN = (128, 128)
_MMA_PV_TILER_MN = (128, 256)
_MAX_ACTIVE_CLUSTERS = 2
_SKIP_CORRECTION_THRESHOLD = 0.0

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    H: int,
    max_active_blocks: int,
) -> Tuple[int, int]:
    """Cache split_kv and workspace_size since they are deterministic for the same params."""
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv_simplified(
        B, q_len, max_active_blocks
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
        H, q_len, _LATENT_DIM, B, split_kv, cutlass.Float32
    )
    return split_kv, workspace_size


@functools.cache
def _get_compiled_mla_kernel(
    torch_dtype: torch.dtype,
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
    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )

    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    if not KernelClass.can_implement(
        1,  # B (runtime, use placeholder)
        seq_len_q,
        1,  # K (runtime, use placeholder)
        num_heads,
        _LATENT_DIM,
        _ROPE_DIM,
        cutlass_dtype,
        cutlass_dtype,
        cutlass.Float32,
        cutlass.Float32,
        _MMA_QK_TILER_MN,
        _MMA_PV_TILER_MN,
        1,  # split_kv (runtime, use 1 to pass the H<128 check)
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        page_size,
    ):
        raise ValueError(
            f"cute_dsl_mla_decode: unsupported configuration "
            f"(q_len={seq_len_q}, num_heads={num_heads}, page_size={page_size}, "
            f"dtype={torch_dtype})"
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

    # All tensors use contiguous row-major layout (stride_order descending).
    # The kernel's __call__ reinterprets them to the required layout via
    # cute.make_tensor zero-cost metadata shuffle.

    # q_latent: [batch_size, seq_len_q, num_heads, latent_dim] — contiguous
    # make_fake_compact_tensor stride_order: value 0 = fastest (stride=1)
    q_latent_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    # q_rope: [batch_size, seq_len_q, num_heads, rope_dim] — contiguous
    q_rope_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_rope),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    # c_latent: [kv_batch, seq_len_k, latent_dim] — contiguous
    # kv_batch is a separate sym_int from query batch: paged KV cache uses a flat
    # pool so kv_batch=num_pages at runtime, while query batch can be any value.
    c_latent_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_latent),
        stride_order=(2, 1, 0),
        assumed_align=128,
    )
    # c_rope: [kv_batch, seq_len_k, rope_dim] — contiguous
    c_rope_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_rope),
        stride_order=(2, 1, 0),
        assumed_align=128,
    )
    # page_table: [batch_size, page_count] — contiguous
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=128,
    )
    # o: [batch_size, seq_len_q, num_heads, latent_dim] — contiguous
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride_order=(3, 2, 1, 0),
        assumed_align=128,
    )
    # lse: [batch_size, seq_len_q, num_heads] — contiguous
    lse_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_batch, sym_seq_q, sym_heads),
        stride_order=(2, 1, 0),
        assumed_align=128,
    )
    # workspace: 1-D (int8 to match typical torch workspace buffers)
    workspace_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int8,
        (sym_workspace_size,),
        assumed_align=128,
    )
    # cache_seqs: [batch_size] — int32
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=128,
    )
    # block_split_kvs: [batch_size] — int32 (only needed for is_var_split_kv=True)
    if is_var_split_kv:
        block_split_kvs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_batch,),
            assumed_align=128,
        )
    else:
        block_split_kvs_fake = None

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


# TODO: query[..., :kv_lora_rank], do we need to remove such kind of slice and move the logic to call routine in the kernel file.
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
        Pre-allocated workspace buffer (uint8). Required size depends on batch size
        and split_kv (auto-computed from B, q_len, and number of SMs):

        - Formula: ``B * H * q_len * split_kv * (kv_lora_rank + 1) * 4`` bytes
          (0 when split_kv == 1, which happens when B >= num_SMs / 2)
        - Typical max: ~18 MB on a 148-SM GPU (e.g. B=4..8, H=128, D=512)
        - Safe default: 128 MB covers all realistic configurations
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
    assert query.dtype in _SUPPORTED_DTYPES, (
        f"cute_dsl_mla_decode only supports {_SUPPORTED_DTYPES}, got {query.dtype}"
    )
    assert kv_cache.dtype == query.dtype, (
        f"kv_cache dtype {kv_cache.dtype} must match query dtype {query.dtype}"
    )
    B, q_len, H, D_qk = query.shape
    assert D_qk == kv_lora_rank + qk_rope_head_dim
    assert kv_lora_rank == _LATENT_DIM
    assert qk_rope_head_dim == _ROPE_DIM

    q_dtype = query.dtype
    is_fp8 = q_dtype == torch.float8_e4m3fn

    # Handle 3D vs 4D kv_cache: normalize to 3D [num_pages, page_size, D_total]
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]

    # Split query into latent and rope components — keep contiguous [B, q_len, H, D].
    # The kernel's __call__ reinterprets to [H, D, q_len, B] via zero-cost make_tensor.
    q_latent_k = query[..., :kv_lora_rank]
    q_rope_k = query[..., kv_lora_rank:]

    # KV cache slices — keep contiguous [num_pages, page_size, D].
    # The kernel reinterprets to [page_size, D, num_pages] internally.
    c_latent_k = kv_cache[:, :, :kv_lora_rank]
    c_rope_k = kv_cache[:, :, kv_lora_rank:]

    # Page table: [B, max_pages]: passed directly, kernel reinterprets.
    page_table_k = block_tables

    # Runtime validation (int comparisons only, negligible overhead)
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if H < 128 and H != 1:
        raise ValueError(
            f"cute_dsl_mla_decode requires num_heads == 128 (or 1), got {H}"
        )

    # Cached split_kv and workspace_size computation
    max_active_blocks = get_num_sm(query.device)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B, q_len, H, max_active_blocks
    )

    if H < 128 and split_kv != 1:
        raise ValueError(
            f"cute_dsl_mla_decode: num_heads={H} < 128 requires split_kv==1, "
            f"got split_kv={split_kv}"
        )

    # Prepare workspace: slice of contiguous 1D buffer is already contiguous
    assert workspace_buffer.numel() >= workspace_size, (
        f"workspace_buffer too small: {workspace_buffer.numel()} bytes, "
        f"need {workspace_size} bytes"
    )
    workspace_bytes = workspace_buffer[: max(workspace_size, 1)]

    # Output buffer: contiguous [B, q_len, H, D].
    # Kernel reinterprets to [H, D, q_len, B] internally via zero-cost make_tensor.
    out_dtype = q_dtype
    if out is not None:
        if q_len == 1:
            o_k = out.unsqueeze(1)  # [B, H, D] → [B, 1, H, D]
        else:
            o_k = out
    else:
        o_k = torch.empty(
            (B, q_len, H, _LATENT_DIM), dtype=out_dtype, device=query.device
        )

    # LSE: contiguous [B, q_len, H]. Kernel reinterprets to [H, q_len, B].
    lse_k = torch.empty((B, q_len, H), dtype=torch.float32, device=query.device)

    # cache_seqs: per-batch sequence lengths (skip .to() if already int32)
    cache_seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)

    is_var_split_kv = False
    block_split_kvs = None

    # Get compiled kernel (cached after first compile)
    compiled_kernel = _get_compiled_mla_kernel(
        torch_dtype=q_dtype,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        is_persistent=True,
        is_var_seq=True,
        is_var_split_kv=is_var_split_kv,
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

    # o_k is already [B, q_len, H, D] contiguous — just squeeze for q_len==1.
    if q_len == 1:
        return o_k.squeeze(1)
    return o_k
