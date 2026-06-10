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

"""Shared helpers used by ``TraceTemplate.init`` functions.

This module contains the small set of input-construction patterns that
recur across many templates (paged-KV cache index arrays, ragged indptr,
RoPE pos_ids and cos/sin caches, sampling probs). Each helper is short and
documented; init functions in ``templates/<category>.py`` call into here so
the per-template init bodies stay focused on shape/dtype, not boilerplate.

The full source of this module is **inlined into every dumped JSON's
``"init"`` field** by ``flashinfer/trace/template.py:_render_init_source``,
so downstream consumers don't need flashinfer installed to re-run the init
snippets.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def make_paged_kv_indices(
    batch_size: int,
    num_pages_per_seq: int,
    page_size: int,
    *,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(kv_indptr, kv_indices, kv_last_page_len)`` for a uniform batch.

    Every sequence is assigned exactly ``num_pages_per_seq`` pages, fully
    populated (last-page length == page_size).

    Invariants
    ----------
    - ``kv_indptr.shape == (batch_size + 1,)``, dtype int32, monotonic, [0]=0.
    - ``kv_indices == arange(0, batch_size * num_pages_per_seq)``, int32.
    - ``kv_last_page_len == full(batch_size, page_size)``, int32.
    """
    total_pages = batch_size * num_pages_per_seq
    kv_indptr = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), page_size, dtype=torch.int32, device=device
    )
    return kv_indptr, kv_indices, kv_last_page_len


def make_ragged_indptr(
    seg_lens,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    """Return cumulative-sum ``indptr`` of length ``len(seg_lens)+1``.

    ``seg_lens`` may be a list / tuple / 1-D tensor of segment lengths.
    """
    if isinstance(seg_lens, torch.Tensor):
        lens = seg_lens.to(device=device, dtype=dtype)
    else:
        lens = torch.tensor(list(seg_lens), dtype=dtype, device=device)
    indptr = torch.zeros(lens.numel() + 1, dtype=dtype, device=device)
    indptr[1:] = torch.cumsum(lens, dim=0).to(dtype)
    return indptr


def make_uniform_qo_indptr(
    batch_size: int,
    qo_len: int,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    """Return ``[0, qo_len, 2*qo_len, ..., batch_size*qo_len]`` int32."""
    return torch.arange(batch_size + 1, dtype=torch.int32, device=device) * qo_len


def make_pos_ids(
    nnz: int,
    max_seq_len: Optional[int] = None,
    *,
    device: str = "cuda",
) -> torch.Tensor:
    """Return ``[0, 1, ..., nnz-1] (% max_seq_len)`` as int32 on ``device``.

    If ``max_seq_len`` is None, no wrapping is applied.
    """
    pos = torch.arange(nnz, dtype=torch.int32, device=device)
    if max_seq_len is not None:
        pos = pos % max_seq_len
    return pos


def make_rope_cos_sin_cache(
    max_seq_len: int,
    rope_dim: int,
    *,
    base: float = 1e4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return concatenated ``[cos | sin]`` cache of shape ``[max_seq_len, rope_dim]``."""
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    inv = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    freqs = t.unsqueeze(-1) * inv.unsqueeze(0)
    cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return cache.to(dtype)


def make_probs(
    batch_size: int,
    vocab_size: int,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a ``[batch_size, vocab_size]`` probability distribution.

    Uses ``softmax(randn(...))`` so each row sums to 1.0. This mirrors the
    pattern used throughout ``tests/utils/test_sampling.py``.
    """
    return torch.softmax(
        torch.randn(batch_size, vocab_size, dtype=torch.float32, device=device),
        dim=-1,
    ).to(dtype)


def make_logits(
    batch_size: int,
    vocab_size: int,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``randn(batch_size, vocab_size)`` logits."""
    return torch.randn(batch_size, vocab_size, dtype=dtype, device=device)


def fp8_safe_randn(
    *shape: int,
    scale: float = 0.1,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``randn(*shape) * scale`` — keeps values in the FP8/FP4 representable range.

    Tests for fp8/fp4 paths typically multiply ``randn`` by 0.1 to avoid
    saturation when quantizing. Use this helper to mirror that convention.
    """
    return (torch.randn(*shape, dtype=dtype, device=device) * scale).to(dtype)


def per_tensor_fp8_quantize(
    x: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor FP8 quantization, mirroring ``tests/utils_fp8.py:to_float8``.

    Returns ``(x_fp8, inv_scale)`` where ``inv_scale`` is the dequant
    multiplier (``float ≈ fp8 * inv_scale``).
    """
    finfo = torch.finfo(fp8_dtype)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax
    x_q = (x.float() * scale).clamp(min=finfo.min, max=finfo.max).to(fp8_dtype)
    return x_q, scale.float().reciprocal()


def fp8_block_quant_1d(
    x_bf16: torch.Tensor,
    block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[T, H]`` activations into FP8 with per-``(token, block)``
    column-block scales. Returns ``(x_fp8, scales)`` where
    ``scales`` has shape ``[T, H // block]``.

    Mirrors ``_fp8_block_quant_1d`` in
    ``tests/moe/test_dpsk_fused_moe_fp8.py``.
    """
    assert x_bf16.dim() == 2
    T, H = x_bf16.shape
    assert H % block == 0
    nb = H // block
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max
    x_f32 = x_bf16.to(torch.float32)
    x_fp8 = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=x_bf16.device)
    scales = torch.empty((T, nb), dtype=torch.float32, device=x_bf16.device)
    for j in range(nb):
        sl = slice(j * block, (j + 1) * block)
        blk = x_f32[:, sl]
        amax = torch.amax(torch.abs(blk), dim=1)
        s = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
        x_fp8[:, sl] = (blk / s.unsqueeze(1)).to(torch.float8_e4m3fn)
        scales[:, j] = s
    return x_fp8, scales


def fp8_block_quant_2d(
    w_bf16: torch.Tensor,
    block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights ``[..., R, C]`` with 2-D ``block × block`` scales.

    Returns ``(w_fp8, scales)`` where ``scales`` has shape
    ``[..., R // block, C // block]``. Mirrors ``_fp8_block_quant_2d`` in
    ``tests/moe/test_dpsk_fused_moe_fp8.py``.
    """
    assert w_bf16.dim() >= 2
    *prefix, R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nb_r, nb_c = R // block, C // block
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max
    w_f32 = w_bf16.to(torch.float32).contiguous()
    prefix_ndim = len(prefix)
    reshaped = w_f32.reshape(*prefix, nb_r, block, nb_c, block)
    permute_dims = tuple(range(prefix_ndim)) + (
        prefix_ndim,
        prefix_ndim + 2,
        prefix_ndim + 1,
        prefix_ndim + 3,
    )
    blocks = reshaped.permute(permute_dims).contiguous()
    amax = torch.amax(torch.abs(blocks), dim=(-1, -2))
    scales = torch.where(
        amax > 0, amax / max_fp8, torch.ones_like(amax, dtype=torch.float32)
    )
    q_blocks = (blocks / scales.unsqueeze(-1).unsqueeze(-1)).to(torch.float8_e4m3fn)
    inv_permute = [0] * (prefix_ndim + 4)
    for i, p in enumerate(permute_dims):
        inv_permute[p] = i
    w_fp8 = q_blocks.permute(*inv_permute).reshape(*prefix, R, C).contiguous()
    return w_fp8, scales


__all__ = [
    "make_paged_kv_indices",
    "make_ragged_indptr",
    "make_uniform_qo_indptr",
    "make_pos_ids",
    "make_rope_cos_sin_cache",
    "make_probs",
    "make_logits",
    "fp8_safe_randn",
    "per_tensor_fp8_quantize",
    "fp8_block_quant_1d",
    "fp8_block_quant_2d",
]
