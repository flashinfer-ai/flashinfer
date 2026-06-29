# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Adapted from Block-Sparse-Attention/bsa_attn_interface.py for FlashInfer integration.

import os
import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

from flashinfer.api_logging import flashinfer_api

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

from .bsa_utils.cache_utils import get_jit_cache
from .bsa_utils.testing import is_fake_mode
from .bsa_utils import fa_logging
from .blk128.cute_dsl_utils import to_cute_tensor
from .blk128.flash_fwd_sm100 import FlashAttentionForwardSm100

_bsa_clc_enabled: bool = os.environ.get("BSA_CLC", "1") == "1"


def _get_use_clc_scheduler() -> bool:
    return _bsa_clc_enabled


def _parse_arch_str(arch_str):
    import re

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@flashinfer_api
def bsa_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    allow_empty_block_nums: bool = True,
    softmax_scale: Optional[float] = None,
    pack_gqa: Optional[bool] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for BSA block-sparse attention (SM100 only).

    Args:
        q: Query tensor (batch, seqlen_q, num_heads, head_dim)
        k: Key tensor (batch, seqlen_k, num_heads_kv, head_dim)
        v: Value tensor (batch, seqlen_k, num_heads_kv, head_dim_v)
        q2k_block_index: (batch, num_heads, num_q_blocks, max_kv_blocks) int32
        block_sparse_num: Number of KV blocks per Q block (even, >= 2).
            Ignored when q2k_block_nums is provided.
        block_sizes: Actual token count per KV block (num_kv_blocks,) int32.
            Pass None to skip block-size masking (assumes full blocks).
        q2k_block_nums: Per-(batch,head,q_block) number of KV blocks,
            (batch, num_heads, num_q_blocks) int32.  Optional.
        allow_empty_block_nums: Allow q2k_block_nums to contain 0.
        softmax_scale: Softmax scale (default: 1/sqrt(head_dim)).
        pack_gqa: Whether to pack GQA heads.
        return_lse: Whether to return log-sum-exp.
        out: Pre-allocated output tensor.
        lse: Pre-allocated LSE tensor.
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    batch_size, seqlen_q, num_head, head_dim = q.shape
    k.shape[1]
    num_head_kv = k.shape[2]
    head_dim_v = v.shape[-1]

    assert q.dtype in [torch.float16, torch.bfloat16], (
        "inputs must be float16 or bfloat16"
    )
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"

    if not is_fake_mode():
        assert all(t.is_cuda for t in (q, k, v)), "inputs must be on CUDA device"

    arch = _get_device_arch()
    assert arch // 10 in [10, 11], "BSA only supports SM100/SM110"
    assert num_head % num_head_kv == 0

    assert q2k_block_index.dtype == torch.int32, "q2k_block_index must be int32"
    has_block_sizes = block_sizes is not None
    if has_block_sizes:
        assert block_sizes.dtype == torch.int32, "block_sizes must be int32"
    if q2k_block_nums is not None:
        q2k_block_nums = maybe_contiguous(q2k_block_nums)
        assert q2k_block_nums.dtype == torch.int32, "q2k_block_nums must be int32"
    else:
        assert block_sparse_num >= 2 and block_sparse_num % 2 == 0, (
            f"block_sparse_num={block_sparse_num} must be even and >= 2"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    qhead_per_kvhead = num_head // num_head_kv

    tile_n = 128
    use_2cta_instrs = False
    tile_m = 128

    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if pack_gqa and (tile_m % qhead_per_kvhead != 0):
        pack_gqa = False

    lse_shape = (batch_size, num_head, seqlen_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            batch_size, seqlen_q, num_head, head_dim_v, dtype=q.dtype, device=q.device
        )

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=q.device)
            if requires_grad or return_lse
            else None
        )

    dtype = torch2cute_dtype_map[q.dtype]
    use_clc_scheduler = _get_use_clc_scheduler()

    current_stream = (
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if is_fake_mode()
        else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )

    has_variable_block_nums = q2k_block_nums is not None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        lse is None,
        tile_m,
        tile_n,
        pack_gqa,
        arch,
        use_2cta_instrs,
        use_clc_scheduler,
        fa_logging.get_fa_log_level(),
        has_variable_block_nums,
        allow_empty_block_nums and has_variable_block_nums,
        has_block_sizes,
    )

    if compile_key not in bsa_attn_fwd.compile_cache:  # type: ignore[attr-defined]
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4) if lse is not None else None
        block_index_tensor = to_cute_tensor(q2k_block_index)
        block_sizes_tensor = to_cute_tensor(block_sizes) if has_block_sizes else None
        block_nums_tensor = (
            to_cute_tensor(q2k_block_nums) if has_variable_block_nums else None
        )

        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            is_persistent=True,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
            allow_empty_block_nums=allow_empty_block_nums and has_variable_block_nums,
            has_block_sizes=has_block_sizes,
        )

        bsa_attn_fwd.compile_cache[compile_key] = cute.compile(  # type: ignore[attr-defined]
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            block_index_tensor,
            block_sizes_tensor,
            block_sparse_num,
            block_nums_tensor,
            current_stream,
            options="--enable-tvm-ffi",
        )

    if not is_fake_mode():
        with torch.cuda.nvtx.range("bsa_attn_fwd_kernel"):
            bsa_attn_fwd.compile_cache[compile_key](  # type: ignore[attr-defined]
                q.detach(),
                k.detach(),
                v.detach(),
                out.detach(),
                lse,
                softmax_scale,
                q2k_block_index.detach(),
                block_sizes.detach() if has_block_sizes else None,
                block_sparse_num,
                q2k_block_nums.detach() if has_variable_block_nums else None,
                current_stream,
            )

    return out, lse


bsa_attn_fwd.compile_cache = get_jit_cache("bsa_fwd")  # type: ignore[attr-defined]
