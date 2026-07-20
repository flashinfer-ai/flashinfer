# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fused RoPE + FP8 quantization (MLA latent key/value) using the public
# cuda.tile API.

import math
import os
from types import SimpleNamespace
from typing import Optional
from typing import Tuple
from typing import TypeAlias

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from ....cutile.cutile_common import cached_replace_hints

ConstInt: TypeAlias = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Set FLASHINFER_CUTILE_AUTOTUNE_DISABLED=1 to skip exhaustive search (faster but suboptimal)
_AUTOTUNE_DISABLED = os.getenv("FLASHINFER_CUTILE_AUTOTUNE_DISABLED", "0") != "0"

# Module-level tune cache: (num_tokens, num_qo_heads, num_kv_heads, rope_dim, no_rope_dim, q_dtype, out_dtype, device) -> (best_cfg, tuned_kernel)
_rope_quantize_fp8_tune_cache: dict = {}


def _default_tokens_per_block(num_tokens: int) -> int:
    if num_tokens <= 4:
        return 1
    if num_tokens <= 32:
        return 4
    if num_tokens <= 128:
        return 8
    return 32


def _rope_quantize_fp8_cutile_configs(num_tokens: int):
    candidates = [tpb for tpb in (1, 4, 8, 16, 32) if tpb <= max(32, num_tokens)]
    if _AUTOTUNE_DISABLED:
        default_tpb = _default_tokens_per_block(num_tokens)
        return [SimpleNamespace(TOKENS_PER_BLOCK=default_tpb, occupancy=4)]

    return [
        SimpleNamespace(TOKENS_PER_BLOCK=tokens_per_block, occupancy=occupancy)
        for tokens_per_block in candidates
        for occupancy in (1, 2, 4)
    ]


def _cutile_autotune_rope_quantize_fp8(
    stream,
    q_rope,
    k_rope,
    q_nope,
    k_nope,
    cos_sin_cache,
    pos_ids,
    q_rope_out,
    k_rope_out,
    q_nope_out,
    k_nope_out,
    quant_scale_q,
    quant_scale_kv,
    num_tokens,
    num_qo_heads,
    num_kv_heads,
    rope_dim,
    no_rope_dim,
    total_blocks_y,
):
    cache_key = (
        num_tokens,
        num_qo_heads,
        num_kv_heads,
        rope_dim,
        no_rope_dim,
        q_rope.dtype,
        q_rope_out.dtype,
        str(q_rope.device),
    )
    if cache_key not in _rope_quantize_fp8_tune_cache:
        result = exhaustive_search(
            list(_rope_quantize_fp8_cutile_configs(num_tokens)),
            stream,
            lambda cfg: (
                math.ceil(num_tokens / cfg.TOKENS_PER_BLOCK),
                total_blocks_y,
                1,
            ),
            _rope_quantize_fp8_kernel,
            lambda cfg: (
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                cos_sin_cache,
                pos_ids,
                q_rope_out,
                k_rope_out,
                q_nope_out,
                k_nope_out,
                quant_scale_q,
                quant_scale_kv,
                num_tokens,
                num_qo_heads,
                num_kv_heads,
                rope_dim,
                no_rope_dim,
                cfg.TOKENS_PER_BLOCK,
            ),
            lambda cfg: {"occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _rope_quantize_fp8_tune_cache[cache_key] = (
            best_cfg,
            _rope_quantize_fp8_kernel.replace_hints(occupancy=best_cfg.occupancy),
        )
    best_cfg, tuned_kernel = _rope_quantize_fp8_tune_cache[cache_key]
    ct.launch(
        stream,
        (math.ceil(num_tokens / best_cfg.TOKENS_PER_BLOCK), total_blocks_y, 1),
        tuned_kernel,
        (
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            cos_sin_cache,
            pos_ids,
            q_rope_out,
            k_rope_out,
            q_nope_out,
            k_nope_out,
            quant_scale_q,
            quant_scale_kv,
            num_tokens,
            num_qo_heads,
            num_kv_heads,
            rope_dim,
            no_rope_dim,
            best_cfg.TOKENS_PER_BLOCK,
        ),
    )


def _load_rope_factors(pos_ids, cos_sin_cache, token_block, TOKENS_PER_BLOCK, HALF_DIM):
    pos_tile = ct.load(
        pos_ids, index=token_block, shape=TOKENS_PER_BLOCK, padding_mode=PAD_ZERO
    )
    pos_rows = ct.reshape(pos_tile, (TOKENS_PER_BLOCK, 1))
    half_cols = ct.reshape(ct.arange(HALF_DIM, dtype=ct.int32), (1, HALF_DIM))
    cos = ct.gather(cos_sin_cache, (pos_rows, half_cols), check_bounds=False)
    sin = ct.gather(cos_sin_cache, (pos_rows, half_cols + HALF_DIM), check_bounds=False)
    return cos, sin


def _apply_rope_interleave_batched(
    x_tile, cos, sin, out_dtype, quant_scale, TOKENS_PER_BLOCK, HALF_DIM
):
    x_3d = ct.reshape(ct.astype(x_tile, ct.float32), (TOKENS_PER_BLOCK, HALF_DIM, 2))
    x_even = ct.reshape(
        ct.extract(x_3d, index=(0, 0, 0), shape=(TOKENS_PER_BLOCK, HALF_DIM, 1)),
        (TOKENS_PER_BLOCK, HALF_DIM),
    )
    x_odd = ct.reshape(
        ct.extract(x_3d, index=(0, 0, 1), shape=(TOKENS_PER_BLOCK, HALF_DIM, 1)),
        (TOKENS_PER_BLOCK, HALF_DIM),
    )

    out_even = (x_even * cos - x_odd * sin) * quant_scale
    out_odd = (x_odd * cos + x_even * sin) * quant_scale

    return ct.cat(
        (
            ct.reshape(ct.astype(out_even, out_dtype), (TOKENS_PER_BLOCK, HALF_DIM, 1)),
            ct.reshape(ct.astype(out_odd, out_dtype), (TOKENS_PER_BLOCK, HALF_DIM, 1)),
        ),
        2,
    )


def _quantize_batched_tile(x_tile, out_dtype, quant_scale):
    return ct.astype(ct.astype(x_tile, ct.float32) * quant_scale, out_dtype)


@ct.kernel
def _rope_quantize_fp8_kernel(
    q_rope,
    k_rope,
    q_nope,
    k_nope,
    cos_sin_cache,
    pos_ids,
    q_rope_out,
    k_rope_out,
    q_nope_out,
    k_nope_out,
    quant_scale_q,
    quant_scale_kv,
    NUM_TOKENS: ConstInt,
    NUM_QO_HEADS: ConstInt,
    NUM_KV_HEADS: ConstInt,
    ROPE_DIM: ConstInt,
    NO_ROPE_DIM: ConstInt,
    TOKENS_PER_BLOCK: ConstInt,
):
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)

    HALF_DIM: ConstInt = ROPE_DIM // 2
    no_rope_chunks: ConstInt = (NO_ROPE_DIM + ROPE_DIM - 1) // ROPE_DIM

    q_rope_end = NUM_QO_HEADS
    k_rope_end = q_rope_end + NUM_KV_HEADS
    k_nope_end = k_rope_end + NUM_KV_HEADS * no_rope_chunks

    if pid_y < q_rope_end:
        cos, sin = _load_rope_factors(
            pos_ids, cos_sin_cache, pid_x, TOKENS_PER_BLOCK, HALF_DIM
        )
        head_idx = pid_y
        q_tile = ct.load(
            q_rope,
            index=(pid_x, head_idx, 0),
            shape=(TOKENS_PER_BLOCK, 1, ROPE_DIM),
            padding_mode=PAD_ZERO,
        )
        q_rot = _apply_rope_interleave_batched(
            q_tile,
            cos,
            sin,
            q_rope_out.dtype,
            quant_scale_q,
            TOKENS_PER_BLOCK,
            HALF_DIM,
        )
        ct.store(
            q_rope_out,
            index=(pid_x, head_idx, 0),
            tile=ct.reshape(q_rot, (TOKENS_PER_BLOCK, 1, ROPE_DIM)),
        )
    elif pid_y < k_rope_end:
        cos, sin = _load_rope_factors(
            pos_ids, cos_sin_cache, pid_x, TOKENS_PER_BLOCK, HALF_DIM
        )
        k_tile = ct.load(
            k_rope,
            index=(pid_x, 0),
            shape=(TOKENS_PER_BLOCK, ROPE_DIM),
            padding_mode=PAD_ZERO,
        )
        k_rot = _apply_rope_interleave_batched(
            k_tile,
            cos,
            sin,
            k_rope_out.dtype,
            quant_scale_kv,
            TOKENS_PER_BLOCK,
            HALF_DIM,
        )
        ct.store(
            k_rope_out,
            index=(pid_x, 0),
            tile=ct.reshape(k_rot, (TOKENS_PER_BLOCK, ROPE_DIM)),
        )
    elif pid_y < k_nope_end:
        chunk_idx = pid_y - k_rope_end
        k_tile = ct.load(
            k_nope,
            index=(pid_x, chunk_idx),
            shape=(TOKENS_PER_BLOCK, ROPE_DIM),
            padding_mode=PAD_ZERO,
        )
        ct.store(
            k_nope_out,
            index=(pid_x, chunk_idx),
            tile=_quantize_batched_tile(k_tile, k_nope_out.dtype, quant_scale_kv),
        )
    else:
        task_idx = pid_y - k_nope_end
        head_idx = task_idx // no_rope_chunks
        chunk_idx = task_idx % no_rope_chunks
        q_tile = ct.load(
            q_nope,
            index=(pid_x, head_idx, chunk_idx),
            shape=(TOKENS_PER_BLOCK, 1, ROPE_DIM),
            padding_mode=PAD_ZERO,
        )
        ct.store(
            q_nope_out,
            index=(pid_x, head_idx, chunk_idx),
            tile=_quantize_batched_tile(q_tile, q_nope_out.dtype, quant_scale_q),
        )


def rope_quantize_fp8_cutile(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE + FP8 quantization — cuTile backend.

    Applies Rotary Position Embedding and quantizes q/k tensors to FP8 in a single
    fused pass, using the public cuda.tile API.

    Args:
        q_rope: Query RoPE portion [num_tokens, num_qo_heads, rope_dim].
        k_rope: Key RoPE portion [num_tokens, rope_dim] (MLA) or [num_tokens, num_kv_heads, rope_dim].
        q_nope: Query non-RoPE portion [num_tokens, num_qo_heads, no_rope_dim] or None.
        k_nope: Key non-RoPE portion, shape depends on MLA vs MHA.
        cos_sin_cache: Float32 [max_seq_len, rope_dim] cosine/sine cache.
        pos_ids: Int32 position IDs [num_tokens].
        is_neox: Must be False (interleaved layout only).
        quantize_dtype: Output dtype (default: torch.float8_e4m3fn).
        quant_scale_q: Quantization scale for Q.
        quant_scale_kv: Quantization scale for KV.
        q_rope_out / k_rope_out / q_nope_out / k_nope_out: Optional pre-allocated outputs.

    Returns:
        (q_rope_out, k_rope_out, q_nope_out, k_nope_out)
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")
    if k_rope.ndim != 2:
        # The kernel loads k_rope as a 2D [tokens, rope_dim] latent (the MLA
        # single-shared-K-head case). A 3D GQA/MHA key would compile to an
        # opaque TileTypeError, so reject it up front.
        raise NotImplementedError(
            "rope_quantize_fp8_cutile supports MLA-style 2D key tensors "
            f"(single shared K head) only; got {k_rope.ndim}D key."
        )

    nnz = q_rope.shape[0]
    num_qo_heads = q_rope.shape[1]
    is_mla = k_rope.ndim == 2
    num_kv_heads = 1 if is_mla else k_rope.shape[1]

    if q_nope is None:
        q_nope = torch.empty(
            nnz, num_qo_heads, 0, dtype=q_rope.dtype, device=q_rope.device
        )
    if k_nope is None:
        if is_mla:
            k_nope = torch.empty(nnz, 0, dtype=k_rope.dtype, device=k_rope.device)
        else:
            k_nope = torch.empty(
                nnz, num_kv_heads, 0, dtype=k_rope.dtype, device=k_rope.device
            )

    if quantize_dtype is None:
        for out in (q_rope_out, k_rope_out, q_nope_out, k_nope_out):
            if out is not None:
                quantize_dtype = out.dtype
                break
        else:
            quantize_dtype = torch.float8_e4m3fn

    if quantize_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(
            f"quantize_dtype must be float8_e4m3fn or float8_e5m2; got {quantize_dtype}"
        )

    # The kernel tiles the full (token, head, rope/nope-chunk) space and writes
    # every in-bounds output element (tail token-blocks are bounds-clipped on
    # store, mirroring the mm_bf16 idiom); empty nope buffers are 0-byte. So the
    # outputs need no pre-zeroing — empty_like avoids a redundant memset on this
    # fused hot path (unlike the beta=0 GEMM epilogue the #3426 pattern guards).
    q_rope_out = (
        q_rope_out
        if q_rope_out is not None
        else torch.empty_like(q_rope, dtype=quantize_dtype)
    )
    k_rope_out = (
        k_rope_out
        if k_rope_out is not None
        else torch.empty_like(k_rope, dtype=quantize_dtype)
    )
    q_nope_out = (
        q_nope_out
        if q_nope_out is not None
        else torch.empty_like(q_nope, dtype=quantize_dtype)
    )
    k_nope_out = (
        k_nope_out
        if k_nope_out is not None
        else torch.empty_like(k_nope, dtype=quantize_dtype)
    )

    num_tokens = q_rope.shape[0]
    rope_dim = q_rope.shape[2]
    num_kv_heads = 1 if k_rope.ndim == 2 else k_rope.shape[1]
    no_rope_dim = q_nope.shape[2] if q_nope is not None else 0

    no_rope_chunks = (no_rope_dim + rope_dim - 1) // rope_dim
    total_blocks_y = (
        num_qo_heads
        + num_kv_heads
        + num_kv_heads * no_rope_chunks
        + num_qo_heads * no_rope_chunks
    )

    if is_neox:
        raise ValueError(
            "rope_quantize_fp8_cutile supports is_neox=False (interleaved) only."
        )

    stream = torch.cuda.current_stream()
    configs = _rope_quantize_fp8_cutile_configs(num_tokens)
    if len(configs) == 1:
        cfg = configs[0]
        grid = (math.ceil(num_tokens / cfg.TOKENS_PER_BLOCK), total_blocks_y, 1)
        # Memoize the hinted kernel so cuTile's per-shape JIT compile cache
        # survives across launches on this single-config (non-autotune) path.
        kernel = cached_replace_hints(
            _rope_quantize_fp8_kernel, occupancy=cfg.occupancy
        )
        args = (
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            cos_sin_cache,
            pos_ids,
            q_rope_out,
            k_rope_out,
            q_nope_out,
            k_nope_out,
            quant_scale_q,
            quant_scale_kv,
            num_tokens,
            num_qo_heads,
            num_kv_heads,
            rope_dim,
            no_rope_dim,
            cfg.TOKENS_PER_BLOCK,
        )
        ct.launch(stream, grid, kernel, args)
    else:
        _cutile_autotune_rope_quantize_fp8(
            stream,
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            cos_sin_cache,
            pos_ids,
            q_rope_out,
            k_rope_out,
            q_nope_out,
            k_nope_out,
            quant_scale_q,
            quant_scale_kv,
            num_tokens,
            num_qo_heads,
            num_kv_heads,
            rope_dim,
            no_rope_dim,
            total_blocks_y,
        )
    return q_rope_out, k_rope_out, q_nope_out, k_nope_out
