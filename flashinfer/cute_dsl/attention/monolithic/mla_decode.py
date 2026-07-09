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
from typing import Callable, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flashinfer.utils import device_support_pdl

from .mla_decode_fp16 import BlackwellMultiHeadLatentAttentionForwardFP16
from .mla_decode_fp8 import BlackwellMultiHeadLatentAttentionForwardFP8
from flashinfer.cute_dsl.utils import (
    _as_cute_dsl_workspace_i8,
    get_max_active_clusters,
    get_num_sm,
    torch_to_cutlass_dtype,
)


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    H: int,
    kv_lora_rank: int,
    max_active_blocks: int,
) -> Tuple[int, int]:
    """Cache split_kv and workspace_size since they are deterministic for the same params."""
    # When folding S_q into heads, the workspace dims are the effective dims
    # (num_heads * F, q_len // F). get_workspace_size already pads H<128 to
    # 128, so passing num_heads_eff and seq_len_q_eff yields the right size.
    mma_qk_tile_m = 128
    fold_sq_ratio = BlackwellMultiHeadLatentAttentionForwardFP16.compute_fold_sq_ratio(
        H, q_len, mma_qk_tile_m
    )
    num_heads_eff = H * fold_sq_ratio
    seq_len_q_eff = q_len // fold_sq_ratio
    split_kv = BlackwellMultiHeadLatentAttentionForwardFP16.get_split_kv_simplified(
        B, seq_len_q_eff, max_active_blocks
    )
    workspace_size = BlackwellMultiHeadLatentAttentionForwardFP16.get_workspace_size(
        num_heads_eff, seq_len_q_eff, kv_lora_rank, B, split_kv, cutlass.Float32
    )
    return split_kv, workspace_size


@functools.cache
def _check_can_implement(
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
) -> None:
    """Check if the kernel supports the given configuration (cached)."""
    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_in_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = torch_to_cutlass_dtype(torch_out_dtype)
    if not KernelClass.can_implement(
        1,  # B (runtime, use placeholder)
        seq_len_q,
        1,  # K (runtime, use placeholder)
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        cutlass_in_dtype,
        cutlass_out_dtype,
        cutlass.Float32,
        cutlass.Float32,
        mma_qk_tiler_mn,
        mma_pv_tiler_mn,
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        page_size,
    ):
        raise ValueError(
            f"cute_dsl_mla_decode: unsupported configuration "
            f"(q_len={seq_len_q}, num_heads={num_heads}, page_size={page_size}, "
            f"in_dtype={torch_dtype}, out_dtype={torch_out_dtype})"
        )


@functools.cache
def _get_compiled_mla_kernel(
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_heads: int,
    seq_len_q: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    skip_correction_threshold: float = 0.0,
    is_workspace_size_zero: bool = False,
    enable_pdl: bool = False,
) -> Callable:
    """Compile and cache an MLA decode kernel.

    Returns a callable that accepts (q_latent, q_rope, c_latent, c_rope,
    page_table, o, lse, workspace, split_kv_scalar, cache_seqs,
    block_split_kvs, softmax_scale_scalar, output_scale_scalar).

    All scalar arguments must be pre-wrapped as Int32/Float32.
    """
    # Tile sizes for Blackwell mma.
    # (128, 128) for QK and (128, 256) for PV.
    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)
    # 2 CTAs along M (num_heads)
    cluster_shape_mnk = (2, 1, 1)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = torch_to_cutlass_dtype(torch_out_dtype)

    # Derive the seq_len_q-into-heads fold factor.  F > 1 means the kernel
    # repacks the [H, S_q] tile to [H*F, S_q/F] internally so MTP / spec-decoding
    # with H < 128 fully populates the 128-wide MMA-M tile.
    fold_sq_ratio = KernelClass.compute_fold_sq_ratio(
        num_heads, seq_len_q, mma_qk_tiler_mn[0]
    )
    fold_sq = fold_sq_ratio > 1

    kernel_obj = KernelClass(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=mma_qk_tiler_mn,
        mma_pv_tiler_mn=mma_pv_tiler_mn,
        max_active_clusters=get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        ),
        page_size=page_size,
        skip_correction_threshold=skip_correction_threshold,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        enable_pdl=enable_pdl,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        fold_sq=fold_sq,
    )

    # All dimensions as sym_int — this matches the original kernel's use of
    # mark_compact_shape_dynamic, which makes ALL shapes dynamic CuTe Integers.
    # Static Python ints would cause cute.assume() to fail with AttributeError
    # inside initialize_workspace() since it expects DSL Integer types.
    sym_heads = cute.sym_int()
    sym_latent = cute.sym_int(divisibility=16)
    sym_seq_q = cute.sym_int()
    sym_rope = cute.sym_int(divisibility=16)
    sym_batch = cute.sym_int()  # query/output batch dimension
    sym_kv_batch = cute.sym_int()  # KV cache batch dim (flat pool, =1 in paged mode)
    sym_seq_kv = cute.sym_int()
    sym_page_count = cute.sym_int()
    sym_workspace_size = cute.sym_int()

    # q_latent, q_rope, c_latent, c_rope are slices of contiguous tensors on
    # the last dim (e.g. query[..., :kv_lora_rank]), so they are NOT contiguous:
    #   stride[-2] = D_qk (original full last dim), not the sliced shape.
    # Use make_fake_tensor with fully dynamic strides so the compiled kernel
    # reads actual strides from the runtime tensor.  Last-dim stride is always 1.

    # q_latent: [batch_size, seq_len_q, num_heads, latent_dim] — non-contiguous slice
    q_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # q_rope: [batch_size, seq_len_q, num_heads, rope_dim] — non-contiguous slice
    q_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # c_latent: [kv_batch, seq_len_k, latent_dim] — non-contiguous slice
    # kv_batch is a separate sym_int from query batch: paged KV cache uses a flat
    # pool so kv_batch=num_pages at runtime, while query batch can be any value.
    c_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # c_rope: [kv_batch, seq_len_k, rope_dim] — non-contiguous slice
    c_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    # page_table: [batch_size, page_count] — contiguous
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=16,
    )
    # o: [batch_size, seq_len_q, num_heads, latent_dim] — contiguous
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_out_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    # lse: [batch_size, seq_len_q, num_heads] — contiguous
    lse_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_batch, sym_seq_q, sym_heads),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    if is_workspace_size_zero:
        workspace_fake = None
    else:
        # workspace: 1-D int8 buffer. 32-byte alignment because workspace stores
        # fp32 partial sums internally, requiring stricter alignment than tensors.
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (sym_workspace_size,),
            assumed_align=32,
        )
    # cache_seqs: [batch_size] — int32
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=16,
    )
    # block_split_kvs: [batch_size] — int32 (only needed for is_var_split_kv=True)
    if is_var_split_kv:
        block_split_kvs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_batch,),
            assumed_align=16,
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
        options="--enable-tvm-ffi --opt-level 2",
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
    out_dtype: Optional[torch.dtype] = None,
    is_var_seq: bool = True,
    enable_pdl: Optional[bool] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """CuTe DSL MLA decode kernel for Blackwell SM100.

    Parameters
    ----------
    query : torch.Tensor
        [B, q_len, H, D_qk] where D_qk = kv_lora_rank + qk_rope_head_dim
    kv_cache : torch.Tensor
        [num_pages, page_size, D_ckv + D_kpe] (3D) or [num_pages, 1, page_size, D_ckv + D_kpe] (4D)
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer (int8 or uint8). Required size depends on batch size
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
        Pre-allocated output tensor [B, q_len, H, kv_lora_rank].
    out_dtype : Optional[torch.dtype]
        Output data type. If None, defaults to torch.bfloat16 (matching trtllm-gen backend).
        Supported values: torch.bfloat16, torch.float8_e4m3fn (FP8 input only),
        torch.float16, torch.bfloat16 (FP16/BF16 input).
    is_var_seq : bool
        Whether the sequence length is variable.
        If True, the sequence length is variable.
        Otherwise,the sequence length is fixed for all the requests in the batch.
    enable_pdl : Optional[bool], default=None
        Whether to enable Programmatic Dependent Launch (PDL).
        If None, auto-detects based on device capability.
    lse : Optional[torch.Tensor]
        Pre-allocated Log-Sum-Exp buffer.  Accepted shapes (dtype must be
        ``torch.float32``):

        * ``[B, q_len, H]`` (native kernel layout, no reshape), or
        * ``[B * q_len, H]`` (matches ``trtllm-gen`` shape; the wrapper
          reshapes via ``.view`` to the native layout).

        If ``return_lse`` is True and this is None, a buffer of the native
        ``[B, q_len, H]`` shape is allocated internally.
    return_lse : bool
        Whether to return LSE values.  When True, the function returns
        ``(out, lse)`` (the ``lse`` tensor returned is in whatever shape
        the caller supplied; if no ``lse`` was supplied, ``[B, q_len, H]``).

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Output tensor [B, q_len, H, kv_lora_rank] when ``return_lse=False``;
        otherwise ``(output, lse)``.
    """
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    assert query.dtype in supported_dtypes, (
        f"cute_dsl_mla_decode only supports {supported_dtypes}, got {query.dtype}"
    )
    assert kv_cache.dtype == query.dtype, (
        f"kv_cache dtype {kv_cache.dtype} must match query dtype {query.dtype}"
    )
    B, q_len, H, D_qk = query.shape
    assert D_qk == kv_lora_rank + qk_rope_head_dim

    q_dtype = query.dtype
    # Resolve output dtype: for FP8 input, default to bfloat16 (matching trtllm-gen backend);
    # for FP16/BF16 input, default to same as input. Allow override via out_dtype or out tensor.
    if out is not None:
        o_dtype = out.dtype
    elif out_dtype is not None:
        o_dtype = out_dtype
    elif q_dtype == torch.float8_e4m3fn:
        o_dtype = torch.bfloat16
    else:
        o_dtype = q_dtype

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

    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")

    # Cached split_kv and workspace_size computation
    max_active_blocks = get_num_sm(query.device)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B, q_len, H, kv_lora_rank, max_active_blocks
    )

    # Prepare workspace: the CUTE signature uses int8, while public FlashInfer
    # workspace buffers are byte storage and may be uint8.
    workspace_buffer = _as_cute_dsl_workspace_i8(workspace_buffer)
    if workspace_buffer.numel() < workspace_size:
        raise ValueError(
            f"workspace_buffer too small: {workspace_buffer.numel()} bytes, "
            f"need {workspace_size} bytes"
        )
    is_workspace_size_zero = workspace_size == 0
    if is_workspace_size_zero:
        workspace_bytes = None
    else:
        workspace_bytes = workspace_buffer[:workspace_size]
    # Output buffer: contiguous [B, q_len, H, D].
    # Kernel reinterprets to [H, D, q_len, B] internally via zero-cost make_tensor.
    if out is not None:
        o_k = out
    else:
        o_k = torch.empty(
            (B, q_len, H, kv_lora_rank), dtype=o_dtype, device=query.device
        )

    # LSE: contiguous [B, q_len, H]. Kernel reinterprets to [H, q_len, B].
    # If caller supplied an `lse` buffer in either the native 3D shape or the
    # 2D trtllm-gen shape [B*q_len, H], reshape to the 3D native layout for
    # the kernel call.
    if lse is not None:
        if lse.dtype != torch.float32:
            raise ValueError(f"lse must be torch.float32, got {lse.dtype}")
        if lse.shape == (B, q_len, H):
            lse_k = lse
        elif lse.shape == (B * q_len, H):
            # Native kernel layout is 3D; .view is zero-cost when contiguous.
            lse_k = lse.view(B, q_len, H)
        else:
            raise ValueError(
                f"lse must have shape (B, q_len, H)=({B}, {q_len}, {H}) "
                f"or (B*q_len, H)=({B * q_len}, {H}); got {tuple(lse.shape)}"
            )
    else:
        lse_k = torch.empty((B, q_len, H), dtype=torch.float32, device=query.device)

    # cache_seqs: per-batch sequence lengths (skip .to() if already int32)
    cache_seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)

    is_var_split_kv = False
    block_split_kvs = None
    skip_correction_threshold = 0.0

    # for fix-length, set is_persistent to True; otherwise, set to False.
    is_persistent = not is_var_seq

    # Validate configuration (cached, negligible overhead after first call)
    _check_can_implement(
        torch_dtype=q_dtype,
        torch_out_dtype=o_dtype,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
    )

    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl

    # Get compiled kernel (cached after first compile)
    # Note: when is_workspace_size_zero is True, workspace_bytes is None and it will launch one kernel without workspace.
    # Otherwise, workspace_bytes is not None and it will launch two kernels.
    compiled_kernel = _get_compiled_mla_kernel(
        torch_dtype=q_dtype,
        torch_out_dtype=o_dtype,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_heads=H,
        seq_len_q=q_len,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        skip_correction_threshold=skip_correction_threshold,
        is_workspace_size_zero=is_workspace_size_zero,
        enable_pdl=enable_pdl,
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

    # Pick the output to return: caller-provided buffer (already written
    # in-place) or the freshly allocated o_k.  o_k is [B, q_len, H, D],
    # matching trtllm-gen output shape.
    out_tensor = out if out is not None else o_k

    if return_lse:
        # Return the lse tensor in the shape the caller supplied (or 3D when
        # we allocated it).  When caller passed 2D, lse_k is a .view into
        # that same memory, so returning the original `lse` keeps the
        # caller's expected shape.
        return out_tensor, (lse if lse is not None else lse_k)

    return out_tensor
