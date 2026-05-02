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


__all__ = [
    "make_paged_kv_indices",
    "make_ragged_indptr",
    "make_uniform_qo_indptr",
    "make_pos_ids",
    "make_rope_cos_sin_cache",
    "make_probs",
    "make_logits",
    "fp8_safe_randn",
]
