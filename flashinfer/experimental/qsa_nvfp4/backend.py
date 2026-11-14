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

# QSA sparse paged attention over an NVFP4, FP8-e4m3 or BF16 KV cache, for SM80 through SM90.

import functools
import math
from typing import Optional

import torch

from ...utils import get_compute_capability
from .jit import get_qsa_nvfp4_module

_HEAD_DIM = 256
_MAX_GROUP_SIZE = 16
# kernel format code and tile tokens by cache dtype
_FORMATS = {
    torch.uint8: (0, 64),
    torch.float8_e4m3fn: (1, 64),
    torch.bfloat16: (2, 32),
}


@functools.cache
def _resident_ctas(device_index: int, kv_format: int) -> tuple[int, int]:
    """SM count and resident CTAs per SM."""
    with torch.cuda.device(device_index):
        ctas_per_sm = get_qsa_nvfp4_module().ctas_per_sm(kv_format)
    num_sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    return num_sms, ctas_per_sm


def _num_splits(
    num_rows: int,
    num_kv_heads: int,
    num_tiles: int,
    kv_format: int,
    resident: tuple[int, int],
) -> int:
    num_sms, ctas_per_sm = resident
    work = num_rows * num_kv_heads
    if kv_format == 0:  # NVFP4 runs faster at two CTAs per SM, every split kept
        return min(num_tiles, max(1, 2 * num_sms // work))
    # Fill one wave of resident CTAs. BF16 runs faster at one CTA per SM until that
    # leaves eight or more tiles per split.
    splits = min(num_tiles, max(1, ctas_per_sm * num_sms // work))
    if kv_format == 2:
        one_per_sm = min(num_tiles, max(1, num_sms // work))
        if math.ceil(num_tiles / one_per_sm) < 8:
            splits = one_per_sm
    # Then use the fewest splits with the same tiles per split.
    return math.ceil(num_tiles / math.ceil(num_tiles / splits))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_aligned(
    tensor: torch.Tensor, name: str, num_bytes: int, stride: tuple[int, ...]
) -> None:
    # The kernel moves rows in num_bytes vectors: base and every outer stride must be aligned,
    # i.e. num_bytes divides the gcd of the base and the outer byte strides. stride is the
    # tensor's strides, read once by the caller.
    *outer, last = stride
    if (
        last != 1
        or math.gcd(tensor.data_ptr(), tensor.element_size() * math.gcd(*outer))
        % num_bytes
    ):
        raise ValueError(
            f"{name} needs a contiguous last dim and {num_bytes}-byte aligned base and strides"
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
    k_sf: Optional[torch.Tensor] = None,
    v_sf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del (
        use_prefill_config
    )  # kept for the signature; the split rule does not depend on it
    device = q.device
    major = get_compute_capability(device)[0]
    if major not in (8, 9):
        raise RuntimeError(
            f"QSA sparse paged attention supports SM80 through SM90, got SM{major}x"
        )
    _check(q.ndim == 3 and k_cache.ndim == 4, "q must be 3D and k_cache 4D")
    num_rows, num_qo_heads, head_dim = q.shape
    num_blocks, page_size, num_kv_heads, cache_dim = k_cache.shape
    _check(q.dtype == torch.bfloat16, "q must be bfloat16")
    _check(head_dim == _HEAD_DIM, f"head_dim must be {_HEAD_DIM}")
    _check(
        k_cache.dtype in _FORMATS,
        "k_cache must be uint8 (NVFP4), float8_e4m3fn or bfloat16",
    )
    kv_format, tile_tokens = _FORMATS[k_cache.dtype]
    nvfp4 = k_cache.dtype == torch.uint8
    _check(
        cache_dim == (head_dim // 2 if nvfp4 else head_dim),
        "k_cache must hold head_dim / 2 packed bytes (uint8 is NVFP4; view an FP8 cache "
        "as float8_e4m3fn) or head_dim values",
    )
    k_stride = k_cache.stride()
    _check(
        v_cache.shape == k_cache.shape
        and v_cache.dtype == k_cache.dtype
        and v_cache.stride() == k_stride,
        "k_cache and v_cache must have the same dtype, shape and strides",
    )
    if nvfp4:
        sf_shape = (num_blocks, page_size, num_kv_heads, head_dim // 16)
        if not (
            k_sf is not None
            and v_sf is not None
            and k_sf.dtype == torch.float8_e4m3fn
            and k_sf.shape == sf_shape
            and v_sf.shape == sf_shape
            and v_sf.dtype == k_sf.dtype
            and v_sf.stride() == k_sf.stride()
        ):
            raise ValueError(
                f"k_sf and v_sf must be float8_e4m3fn views of shape {sf_shape} with the same strides"
            )
    else:
        _check(k_sf is None and v_sf is None, "k_sf and v_sf are only taken with NVFP4")
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
    if k_cache.dtype == torch.bfloat16:
        k_scale = v_scale = 1.0
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
        all(
            x is None or x.device == device
            for x in (*tensors, token_to_req, output_gate, out)
        ),
        "all tensors must be on the same device",
    )
    # v_cache and v_sf share the dtype and strides of k_cache and k_sf.
    _check_aligned(q, "q", 8, q.stride())
    _check_aligned(k_cache, "k_cache", 16, k_stride)
    _check_aligned(v_cache, "v_cache", 16, k_stride)
    if nvfp4:
        sf_stride = k_sf.stride()
        _check_aligned(k_sf, "k_sf", 16, sf_stride)
        _check_aligned(v_sf, "v_sf", 16, sf_stride)
    _check_aligned(output_gate, "output_gate", 16, output_gate.stride())
    _check_aligned(out, "out", 16, out.stride())
    if num_rows == 0:
        return out

    num_tiles = math.ceil((logical_indices.shape[1] - 1) / tile_tokens)
    num_splits = _num_splits(
        num_rows,
        num_kv_heads,
        num_tiles,
        kv_format,
        _resident_ctas(device.index, kv_format),
    )
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
        float(k_scale) * math.log2(math.e) / math.sqrt(head_dim),
        float(v_scale),
        kv_format,
        num_splits,
    )
    return out
