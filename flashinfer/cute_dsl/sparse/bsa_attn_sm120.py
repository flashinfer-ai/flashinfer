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

    if out is None:
        out = torch.empty(
            (batch, seqlen_q, num_heads, head_dim_v), dtype=q.dtype, device=q.device
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
