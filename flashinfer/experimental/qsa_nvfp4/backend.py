"""
Copyright (c) 2026 by FlashInfer team.

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

# QSA sparse paged attention over an NVFP4 KV cache, for SM80 through SM90.

import math
from typing import Optional

import torch

from ...utils import get_compute_capability
from .jit import get_qsa_nvfp4_module

_HEAD_DIM = 256
_TILE_TOKENS = 64
_MAX_GROUP_SIZE = 16
# Splits are sized so rows * kv_heads * splits reaches about this many CTAs. One value per
# architecture family, never per device.
_TARGET_CTAS = {8: 128, 9: 128}


def _num_splits(
    num_rows: int, num_kv_heads: int, selection_width: int, major: int
) -> int:
    num_tiles = math.ceil(selection_width / _TILE_TOKENS)
    return min(
        num_tiles, max(1, math.ceil(_TARGET_CTAS[major] / (num_rows * num_kv_heads)))
    )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_aligned(tensor: torch.Tensor, name: str, num_bytes: int) -> None:
    # The kernel moves rows in num_bytes vectors: base and every outer stride must be aligned.
    element_size = tensor.element_size()
    _check(
        tensor.stride(-1) == 1
        and tensor.data_ptr() % num_bytes == 0
        and all(s * element_size % num_bytes == 0 for s in tensor.stride()[:-1]),
        f"{name} needs a contiguous last dim and {num_bytes}-byte aligned base and strides",
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: Optional[torch.Tensor] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    *,
    output_gate: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
) -> torch.Tensor:
    del (
        use_prefill_config
    )  # kept for the signature; the split rule does not depend on it
    major = get_compute_capability(q.device)[0]
    if major not in _TARGET_CTAS:
        raise RuntimeError(
            f"QSA NVFP4 attention supports SM80 through SM90, got SM{major}x"
        )
    _check(q.ndim == 3 and k_cache.ndim == 4, "q must be 3D and k_cache 4D")
    num_rows, num_qo_heads, head_dim = q.shape
    num_blocks, page_size, num_kv_heads, packed_dim = k_cache.shape
    _check(q.dtype == torch.bfloat16, "q must be bfloat16")
    _check(head_dim == _HEAD_DIM, f"head_dim must be {_HEAD_DIM}")
    _check(packed_dim * 2 == head_dim, "k_cache must hold head_dim / 2 packed bytes")
    _check(
        k_cache.dtype == torch.uint8
        and v_cache.shape == k_cache.shape
        and v_cache.dtype == k_cache.dtype
        and v_cache.stride() == k_cache.stride(),
        "k_cache and v_cache must be uint8 views with the same shape and strides",
    )
    sf_shape = (num_blocks, page_size, num_kv_heads, head_dim // 16)
    _check(
        k_sf.dtype == torch.float8_e4m3fn
        and tuple(k_sf.shape) == sf_shape
        and v_sf.shape == k_sf.shape
        and v_sf.dtype == k_sf.dtype
        and v_sf.stride() == k_sf.stride(),
        f"k_sf and v_sf must be float8_e4m3fn views of shape {sf_shape} with the same strides",
    )
    _check(num_blocks > 0 and page_size > 0, "k_cache must not be empty")
    _check(
        num_qo_heads % num_kv_heads == 0
        and num_qo_heads // num_kv_heads <= _MAX_GROUP_SIZE,
        f"num_qo_heads must be a multiple of num_kv_heads, at most {_MAX_GROUP_SIZE}x",
    )
    _check(
        logical_indices.ndim == 2
        and logical_indices.shape[0] == num_rows
        and logical_indices.shape[1] >= 2
        and logical_indices.dtype == torch.int32
        and logical_indices.stride(1) == 1,
        "logical_indices must be int32 [rows, selection_width + 1] with a contiguous last dim",
    )
    _check(
        block_table.ndim == 2
        and block_table.numel() > 0
        and block_table.dtype == torch.int32
        and block_table.stride(1) == 1,
        "block_table must be a non-empty int32 2D tensor with a contiguous last dim",
    )
    _check(
        token_to_req.shape == (num_rows,)
        and token_to_req.dtype == torch.int32
        and token_to_req.is_contiguous(),
        "token_to_req must be contiguous int32 [rows]",
    )
    _check(
        k_scale is not None and v_scale is not None, "k_scale and v_scale are required"
    )
    _check(
        output_gate.shape == q.shape
        and output_gate.dtype == torch.bfloat16
        and output_gate.is_contiguous(),
        "output_gate must be contiguous bfloat16 with the shape of q",
    )
    if out is None:
        out = torch.empty_like(q)
    _check(
        out.shape == q.shape and out.dtype == q.dtype,
        "out must match q in shape and dtype",
    )
    tensors = (q, k_cache, v_cache, k_sf, v_sf, logical_indices, block_table)
    _check(
        all(x.device == q.device for x in (*tensors, token_to_req, output_gate, out)),
        "all tensors must be on the same device",
    )
    _check_aligned(q, "q", 8)
    for name, tensor in (
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("k_sf", k_sf),
        ("v_sf", v_sf),
        ("output_gate", output_gate),
        ("out", out),
    ):
        _check_aligned(tensor, name, 16)
    if num_rows == 0:
        return out

    num_splits = _num_splits(
        num_rows, num_kv_heads, logical_indices.shape[1] - 1, major
    )
    if num_splits > 1:
        partial_out = q.new_empty(
            (num_splits, num_rows, num_qo_heads, head_dim), dtype=torch.float32
        )
        partial_lse = q.new_empty(
            (num_splits, num_rows, num_qo_heads), dtype=torch.float32
        )
    else:
        partial_out = partial_lse = out  # not read with a single split
    get_qsa_nvfp4_module().run(
        q,
        k_cache,
        v_cache,
        k_sf,
        v_sf,
        logical_indices,
        block_table,
        token_to_req,
        output_gate,
        out,
        partial_out,
        partial_lse,
        float(k_scale) * math.log2(math.e) / math.sqrt(head_dim),
        float(v_scale),
        num_splits,
    )
    return out
