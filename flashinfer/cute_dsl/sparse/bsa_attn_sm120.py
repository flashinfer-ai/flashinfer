# Copyright (c) 2025 by FlashInfer team.
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

from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from flashinfer.api_logging import flashinfer_api

from .bsa_utils.cache_utils import get_jit_cache
from .bsa_utils.testing import is_fake_mode
from .bsa_utils.cute_tensor_utils import to_cute_tensor

_BLOCK_SIZE = 64

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _prepare_sm120_sparse_metadata(
    q2k_block_index: torch.Tensor,
    q2k_block_nums: Optional[torch.Tensor],
    block_sizes: Optional[torch.Tensor],
    block_sparse_num: int,
    *,
    batch_size: int,
    num_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    device: torch.device,
):
    """Validate and transpose SM120 sparse metadata into kernel ABI layout."""
    assert q2k_block_index.dtype == torch.int32
    assert q2k_block_index.device == device
    assert q2k_block_index.shape[:3] == (batch_size, num_heads, num_q_blocks)

    has_block_nums = q2k_block_nums is not None and q2k_block_nums.numel() > 0
    if has_block_nums:
        assert q2k_block_nums.dtype == torch.int32
        assert q2k_block_nums.device == device
        assert q2k_block_nums.shape == (batch_size, num_heads, num_q_blocks)
        q2k_block_nums = q2k_block_nums.contiguous()
    else:
        assert 1 <= block_sparse_num <= q2k_block_index.shape[-1]

    has_block_sizes = block_sizes is not None and block_sizes.numel() > 0
    block_sizes_mode = 0
    if has_block_sizes:
        assert block_sizes.dtype == torch.int32
        assert block_sizes.device == device
        if block_sizes.ndim == 1:
            assert block_sizes.shape == (num_kv_blocks,)
            block_sizes_t = block_sizes.contiguous()
            block_sizes_mode = 1
        elif block_sizes.ndim == 2:
            assert block_sizes.shape == (batch_size, num_kv_blocks)
            block_sizes_t = block_sizes.contiguous().permute(1, 0)
            block_sizes_mode = 2
        else:
            assert block_sizes.ndim == 3
            assert block_sizes.shape == (batch_size, num_heads, num_kv_blocks)
            block_sizes_t = block_sizes.contiguous().permute(2, 1, 0)
            block_sizes_mode = 3

    # (B, H, Q, K) -> (K, Q, H, B)
    q2k_t = q2k_block_index.contiguous().permute(3, 2, 1, 0)
    q2k_nums_t = q2k_block_nums.permute(2, 1, 0) if has_block_nums else q2k_t
    if not has_block_sizes:
        block_sizes_t = q2k_nums_t

    return (
        has_block_nums,
        has_block_sizes,
        block_sizes_mode,
        q2k_t,
        q2k_nums_t,
        block_sizes_t,
    )


_sm120_compile_cache = get_jit_cache("bsa_fwd_sm120")


@flashinfer_api
def bsa_attn_sm120_blk64_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass for BSA block-sparse attention using the sm120_blk64 CuTe-DSL kernel (SM120/SM121 only).

    Args:
        q: Query tensor (batch, seqlen_q, num_heads, head_dim), fp16/bf16.
        k: Key tensor (batch, seqlen_k, num_kv_heads, head_dim).
        v: Value tensor (batch, seqlen_k, num_kv_heads, head_dim).
        q2k_block_index: (batch, num_heads, num_q_blocks, max_kv_blocks) int32.
        block_sparse_num: Number of KV blocks per Q block. Ignored when q2k_block_nums is provided.
        block_sizes: Actual token count per KV block, int32. Shape: (num_kv_blocks,) or
            (batch, num_kv_blocks) or (batch, num_heads, num_kv_blocks). Pass None to
            skip per-block padding masking.
        q2k_block_nums: Per-(batch, head, q_block) KV block count,
            (batch, num_heads, num_q_blocks) int32. Optional.
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim)).
        return_lse: Whether to return log-sum-exp.
        out: Pre-allocated output tensor (batch, seqlen_q, num_heads, head_dim).
        lse: Pre-allocated LSE tensor (batch, num_heads, seqlen_q).
    """
    from .sm120_blk64.flash_fwd_sm120 import BlockSparseAttnForwardSm120Blk64  # noqa: PLC0415

    assert q.dtype in (torch.float16, torch.bfloat16), (
        "bsa_attn_sm120_blk64_fwd only supports fp16/bf16"
    )
    assert q.dtype == k.dtype == v.dtype, "q, k, v must have the same dtype"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "inputs must be on CUDA device"
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4

    major, minor = torch.cuda.get_device_capability(q.device)
    arch = major * 10 + minor
    if arch // 10 != 12:
        raise RuntimeError(
            f"bsa_attn_sm120_blk64_fwd (sm120_blk64) only supports SM120/SM121, current device is SM{arch}"
        )

    batch, seqlen_q, num_heads, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_kv_heads = k.shape[2]
    head_dim_v = v.shape[3]

    assert head_dim == 128, f"sm120_blk64 requires head_dim=128, got {head_dim}"
    assert head_dim_v == 128, f"sm120_blk64 requires head_dim_v=128, got {head_dim_v}"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1

    gqa_ratio = num_heads // num_kv_heads
    num_q_blocks = _ceil_div(seqlen_q, _BLOCK_SIZE)
    num_kv_blocks = _ceil_div(seqlen_k, _BLOCK_SIZE)

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if out is not None:
        assert out.dtype == q.dtype, (
            f"out.dtype ({out.dtype}) must match q.dtype ({q.dtype}): "
            "the kernel reuses Q shared memory for the O epilogue and requires identical dtypes"
        )
        assert out.shape == (batch, seqlen_q, num_heads, head_dim_v), (
            f"out.shape {tuple(out.shape)} must be "
            f"(batch={batch}, seqlen_q={seqlen_q}, num_heads={num_heads}, head_dim_v={head_dim_v})"
        )
    if out is None:
        out = torch.empty(
            (batch, seqlen_q, num_heads, head_dim_v), dtype=q.dtype, device=q.device
        )
    if lse is not None:
        assert lse.dtype == torch.float32, (
            f"lse.dtype ({lse.dtype}) must be float32: the kernel always writes LSE in float32"
        )
        assert lse.shape == (batch, num_heads, seqlen_q), (
            f"lse.shape {tuple(lse.shape)} must be "
            f"(batch={batch}, num_heads={num_heads}, seqlen_q={seqlen_q})"
        )
    if lse is None:
        lse = torch.empty(
            (batch, num_heads, seqlen_q), dtype=torch.float32, device=q.device
        )

    (
        has_block_nums,
        has_block_sizes,
        block_sizes_mode,
        q2k_t,
        q2k_nums_t,
        block_sizes_t,
    ) = _prepare_sm120_sparse_metadata(
        q2k_block_index,
        q2k_block_nums,
        block_sizes,
        block_sparse_num,
        batch_size=batch,
        num_heads=num_heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        device=q.device,
    )

    # Transpose from BSHD to the layout expected by the sm120 kernel:
    # Q/K/O: (B,S,H,D) -> (S,D,H,B), leading_dim=1 (D stride=1 preserved as view)
    # V:     (B,S,H,D) -> (D,S,H,B), leading_dim=0 (D stride=1 preserved as view)
    # LSE:   (B,H,S)   -> (S,H,B)
    # NOTE: no .contiguous() — we need to preserve the original stride[leading_dim]=1
    # so that mark_layout_dynamic works. The kernel writes directly into the view,
    # which shares memory with the user-provided out/lse tensors.
    q_t = q.permute(1, 3, 2, 0)
    k_t = k.permute(1, 3, 2, 0)
    v_t = v.permute(3, 1, 2, 0)
    out_t = out.permute(1, 3, 2, 0)
    lse_t = lse.permute(2, 1, 0)

    q_cute = to_cute_tensor(q_t, assumed_align=128, leading_dim=1, enable_tvm_ffi=False)
    k_cute = to_cute_tensor(k_t, assumed_align=128, leading_dim=1, enable_tvm_ffi=False)
    v_cute = to_cute_tensor(v_t, assumed_align=128, leading_dim=0, enable_tvm_ffi=False)
    out_cute = to_cute_tensor(
        out_t, assumed_align=128, leading_dim=1, enable_tvm_ffi=False
    )
    lse_cute = to_cute_tensor(
        lse_t, assumed_align=4, leading_dim=0, enable_tvm_ffi=False
    )
    q2k_cute = to_cute_tensor(
        q2k_t, assumed_align=None, leading_dim=0, enable_tvm_ffi=False
    )
    q2k_nums_cute = to_cute_tensor(
        q2k_nums_t, assumed_align=None, leading_dim=0, enable_tvm_ffi=False
    )
    block_sizes_cute = to_cute_tensor(
        block_sizes_t, assumed_align=None, leading_dim=0, enable_tvm_ffi=False
    )

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    fwd_kernel = BlockSparseAttnForwardSm120Blk64(
        gqa_ratio=gqa_ratio,
        head_dim=head_dim,
        value_dim=head_dim_v,
        blocksparse_blocksize_q=_BLOCK_SIZE,
        blocksparse_blocksize_k=_BLOCK_SIZE,
        dtype=torch2cute_dtype_map[q.dtype],
        acc_dtype=cutlass.Float32,
        has_block_sizes=has_block_sizes,
        has_block_nums=has_block_nums,
        block_sizes_mode=block_sizes_mode,
    )

    compile_key = (
        "sm120_blk64_fwd",
        int(arch),
        q.dtype,
        int(head_dim),
        int(head_dim_v),
        int(gqa_ratio),
        bool(has_block_nums),
        bool(has_block_sizes),
        int(block_sizes_mode),
        q_t.stride(),
        k_t.stride(),
        v_t.stride(),
        out_t.stride(),
        lse_t.stride(),
        q2k_t.stride(),
        q2k_nums_t.stride(),
        block_sizes_t.stride(),
    )

    args = (
        q_cute,
        k_cute,
        v_cute,
        out_cute,
        lse_cute,
        q2k_cute,
        q2k_nums_cute,
        cutlass.Int32(block_sparse_num),
        block_sizes_cute,
        cutlass.Float32(softmax_scale),
        current_stream,
    )

    if compile_key not in _sm120_compile_cache:
        _sm120_compile_cache[compile_key] = cute.compile(fwd_kernel, *args)

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_sm120_blk64_fwd_kernel"):
            _sm120_compile_cache[compile_key](*args)

    # out_t and lse_t are views of out/lse — kernel already wrote results in-place.
    return out, lse if return_lse else None


_sm120_sage_compile_cache = get_jit_cache("bsa_fwd_sm120_sage")


@flashinfer_api
def bsa_attn_sm120_blk64_sage_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass for BSA block-sparse attention using the native SM120 Sage
    QK-INT8 / PV-FP8 CuTe-DSL kernel (SM120/SM121 only, MHA only).

    Unlike :func:`bsa_attn_sm120_blk64_fwd`, this function uses BHSD tensor
    layout (batch, heads, seqlen, head_dim) throughout to match the layout
    produced by :func:`flashinfer.quantize_sage_qkv_sm120` and upstream
    Block-Sparse-Attention's native contract. It is a standalone function,
    not wired into :class:`flashinfer.BlockSparseAttentionWrapper`: it does
    not compute LSE and only supports MHA (``num_kv_heads == num_heads``).

    Args:
        q_int8: Query tensor (batch, num_heads, seqlen_q, 128), int8. Produced
            by :func:`flashinfer.quantize_sage_q_sm120`.
        k_int8: Key tensor (batch, num_heads, seqlen_k, 128), int8, channel
            mean-centered. Produced by :func:`flashinfer.quantize_sage_kv_sm120`.
        v_fp8: Value tensor (batch, num_heads, 128, ceil(seqlen_k/64)*64),
            float8_e4m3fn, in Sage's HDS layout with the 16-token physical
            permutation baked in. Produced by
            :func:`flashinfer.quantize_sage_kv_sm120`.
        q_scale: Per-32-token-group Q descale, (batch, num_heads,
            ceil(seqlen_q/128)*4), float32.
        k_scale: Per-K64-tile K descale, (batch, num_heads,
            ceil(seqlen_k/64)), float32.
        v_scale: Per-channel V descale, (batch, num_heads, 128), float32.
        q2k_block_index: (batch, num_heads, num_q_blocks, max_kv_blocks) int32.
        block_sparse_num: Number of KV blocks per Q block. Ignored when
            q2k_block_nums is provided.
        block_sizes: Actual token count per KV block, int32. Shape:
            (num_kv_blocks,) or (batch, num_kv_blocks) or
            (batch, num_heads, num_kv_blocks). Pass None to skip per-block
            padding masking.
        q2k_block_nums: Per-(batch, head, q_block) KV block count,
            (batch, num_heads, num_q_blocks) int32. Optional.
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim)).
        out: Pre-allocated output tensor (batch, num_heads, seqlen_q, 128),
            bfloat16.
    """
    from .sm120_blk64.flash_fwd_sm120_sage import (  # noqa: PLC0415
        BlockSparseAttnForwardSageSm120Blk64,
    )

    def _require_contiguous_cuda(name: str, tensor: torch.Tensor, device) -> None:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    assert q_int8.dtype == torch.int8 and k_int8.dtype == torch.int8, (
        "bsa_attn_sm120_blk64_sage_fwd requires int8 Q/K"
    )
    assert v_fp8.dtype == torch.float8_e4m3fn, (
        "bsa_attn_sm120_blk64_sage_fwd requires float8_e4m3fn V"
    )
    assert q_int8.dim() == 4 and k_int8.dim() == 4 and v_fp8.dim() == 4

    _require_contiguous_cuda("q_int8", q_int8, q_int8.device)
    _require_contiguous_cuda("k_int8", k_int8, q_int8.device)
    _require_contiguous_cuda("v_fp8", v_fp8, q_int8.device)
    _require_contiguous_cuda("q_scale", q_scale, q_int8.device)
    _require_contiguous_cuda("k_scale", k_scale, q_int8.device)
    _require_contiguous_cuda("v_scale", v_scale, q_int8.device)
    for name, tensor in (
        ("q_scale", q_scale),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use torch.float32, got {tensor.dtype}")

    major, minor = torch.cuda.get_device_capability(q_int8.device)
    arch = major * 10 + minor
    if arch // 10 != 12:
        raise RuntimeError(
            f"bsa_attn_sm120_blk64_sage_fwd (sm120_blk64 Sage) only supports "
            f"SM120/SM121, current device is SM{arch}"
        )

    batch, num_heads, seqlen_q, head_dim = q_int8.shape
    seqlen_k = k_int8.shape[2]

    assert head_dim == 128, f"sm120_blk64 Sage requires head_dim=128, got {head_dim}"
    if k_int8.shape != (batch, num_heads, seqlen_k, head_dim):
        raise ValueError(
            f"k_int8.shape {tuple(k_int8.shape)} must be "
            f"(batch={batch}, num_heads={num_heads}, seqlen_k={seqlen_k}, head_dim={head_dim}) "
            "(bsa_attn_sm120_blk64_sage_fwd only supports MHA: num_kv_heads must equal num_heads)"
        )

    padded_k = _ceil_div(seqlen_k, _BLOCK_SIZE) * _BLOCK_SIZE
    if v_fp8.shape != (batch, num_heads, head_dim, padded_k):
        raise ValueError(
            f"v_fp8.shape {tuple(v_fp8.shape)} must be "
            f"(batch={batch}, num_heads={num_heads}, head_dim={head_dim}, "
            f"padded_seqlen_k={padded_k})"
        )

    num_q_blocks = _ceil_div(seqlen_q, _BLOCK_SIZE)
    num_kv_blocks = _ceil_div(seqlen_k, _BLOCK_SIZE)

    # Q scale granularity is 32-token groups padded to 4 groups per 128-token
    # tile (independent of the 64-token block-sparse compute tile above).
    num_q_scale_groups = _ceil_div(seqlen_q, 128) * 4
    expected_scale_shapes = (
        ("q_scale", q_scale, (batch, num_heads, num_q_scale_groups)),
        ("k_scale", k_scale, (batch, num_heads, num_kv_blocks)),
        ("v_scale", v_scale, (batch, num_heads, head_dim)),
    )
    for name, tensor, expected_shape in expected_scale_shapes:
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name}.shape {tuple(tensor.shape)} must be {expected_shape}"
            )

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    if out is not None:
        assert out.dtype == torch.bfloat16, (
            f"out.dtype ({out.dtype}) must be torch.bfloat16"
        )
        assert out.shape == (batch, num_heads, seqlen_q, head_dim), (
            f"out.shape {tuple(out.shape)} must be "
            f"(batch={batch}, num_heads={num_heads}, seqlen_q={seqlen_q}, head_dim={head_dim})"
        )
    if out is None:
        out = torch.empty(
            (batch, num_heads, seqlen_q, head_dim),
            dtype=torch.bfloat16,
            device=q_int8.device,
        )

    (
        has_block_nums,
        has_block_sizes,
        block_sizes_mode,
        q2k_t,
        q2k_nums_t,
        block_sizes_t,
    ) = _prepare_sm120_sparse_metadata(
        q2k_block_index,
        q2k_block_nums,
        block_sizes,
        block_sparse_num,
        batch_size=batch,
        num_heads=num_heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        device=q_int8.device,
    )

    # BHSD -> kernel layout: Q/K/O (B,H,S,D) -> (S,D,H,B), leading_dim=1.
    # V is already Sage's (B,H,D,padded_S) -> (D,S,H,B), leading_dim=1.
    # Scales (B,H,G) -> (G,H,B), leading_dim=0.
    q_t = q_int8.permute(2, 3, 1, 0)
    k_t = k_int8.permute(2, 3, 1, 0)
    v_t = v_fp8.permute(2, 3, 1, 0)
    out_t = out.permute(2, 3, 1, 0)
    q_scale_t = q_scale.permute(2, 1, 0)
    k_scale_t = k_scale.permute(2, 1, 0)
    v_scale_t = v_scale.permute(2, 1, 0)

    q_cute = to_cute_tensor(q_t, assumed_align=128, leading_dim=1, enable_tvm_ffi=False)
    k_cute = to_cute_tensor(k_t, assumed_align=128, leading_dim=1, enable_tvm_ffi=False)
    v_cute = to_cute_tensor(v_t, assumed_align=128, leading_dim=1, enable_tvm_ffi=False)
    out_cute = to_cute_tensor(
        out_t, assumed_align=128, leading_dim=1, enable_tvm_ffi=False
    )
    q_scale_cute = to_cute_tensor(
        q_scale_t, assumed_align=4, leading_dim=0, enable_tvm_ffi=False
    )
    k_scale_cute = to_cute_tensor(
        k_scale_t, assumed_align=4, leading_dim=0, enable_tvm_ffi=False
    )
    v_scale_cute = to_cute_tensor(
        v_scale_t, assumed_align=4, leading_dim=0, enable_tvm_ffi=False
    )
    q2k_cute = to_cute_tensor(
        q2k_t, assumed_align=None, leading_dim=0, enable_tvm_ffi=False
    )
    q2k_nums_cute = to_cute_tensor(
        q2k_nums_t, assumed_align=None, leading_dim=0, enable_tvm_ffi=False
    )
    block_sizes_cute = to_cute_tensor(
        block_sizes_t, assumed_align=None, leading_dim=0, enable_tvm_ffi=False
    )

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    fwd_kernel = BlockSparseAttnForwardSageSm120Blk64(
        gqa_ratio=1,
        head_dim=head_dim,
        value_dim=head_dim,
        blocksparse_blocksize_q=_BLOCK_SIZE,
        blocksparse_blocksize_k=_BLOCK_SIZE,
        has_block_sizes=has_block_sizes,
        has_block_nums=has_block_nums,
        block_sizes_mode=block_sizes_mode,
    )

    compile_key = (
        "sm120_blk64_sage_fwd",
        int(arch),
        int(head_dim),
        bool(has_block_nums),
        bool(has_block_sizes),
        int(block_sizes_mode),
        q_t.stride(),
        k_t.stride(),
        v_t.stride(),
        out_t.stride(),
        q_scale_t.stride(),
        k_scale_t.stride(),
        v_scale_t.stride(),
        q2k_t.stride(),
        q2k_nums_t.stride(),
        block_sizes_t.stride(),
    )

    args = (
        q_cute,
        k_cute,
        v_cute,
        out_cute,
        q_scale_cute,
        k_scale_cute,
        v_scale_cute,
        q2k_cute,
        q2k_nums_cute,
        cutlass.Int32(block_sparse_num),
        block_sizes_cute,
        cutlass.Float32(softmax_scale),
        current_stream,
    )

    if compile_key not in _sm120_sage_compile_cache:
        _sm120_sage_compile_cache[compile_key] = cute.compile(fwd_kernel, *args)

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_sm120_blk64_sage_fwd_kernel"):
            _sm120_sage_compile_cache[compile_key](*args)

    return out
