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

Helpers shared by the MSA op wrappers.
"""

import torch

# MSA KV-block size in tokens; also the KV-cache page size on the paged path.
_BLK_KV = 128

# Process-wide cache of cute.compile results, shared by all MSA ops; each op
# keys it with a tuple starting with its own tag (e.g. "sparse_prefill", ...).
_compile_cache: dict = {}


def _q_offset_tensor(
    q_offset,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    device,
) -> torch.Tensor:
    """Build the per-batch causal offset tensor the kernels consume.

    A query's global position is ``q_offset[b] + local_index`` (MSA q_offset
    semantics). When ``q_offset`` is None, queries are right-aligned to the end
    of the KV sequence: ``q_offset[b] = seqlen_k[b] - seqlen_q[b]``."""
    if q_offset is None:
        return (
            (cu_seqlens_k[1:] - cu_seqlens_k[:-1])
            - (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        ).to(torch.int32)
    return _q_offset_explicit(q_offset, cu_seqlens_q.numel() - 1, device)


def _q_offset_explicit(q_offset, batch_size: int, device) -> torch.Tensor:
    """Normalize a caller-provided ``q_offset`` (int or int32 tensor) to a
    ``(batch_size,)`` int32 tensor on ``device``."""
    if isinstance(q_offset, int):
        return torch.full((batch_size,), q_offset, dtype=torch.int32, device=device)
    if q_offset.dtype != torch.int32:
        raise ValueError("q_offset must be int32")
    return q_offset.to(device)


def _cutlass_dtype(torch_dtype: torch.dtype):
    """Map a torch dtype to the corresponding cutlass dtype."""
    import cutlass

    return {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.uint8: cutlass.Uint8,
        torch.int32: cutlass.Int32,
    }[torch_dtype]


def _fake(dtype, shape, align=16, stride_order=None):
    """Make a fake compact tensor for ``cute.compile`` (row-major by default).

    Compiling against symbolic shapes (``cute.sym_int()``) yields one kernel
    that accepts torch tensors of any matching-rank shape at runtime."""
    import cutlass.cute as cute

    if stride_order is None:
        stride_order = tuple(reversed(range(len(shape))))
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        assumed_align=align,
    )


def _resolve_packed_kv(k, v, head_dim, *, paged, kv_nvfp4):
    """Shared wrapper plumbing: detect packed K/V split views (paged,
    non-NVFP4 only) via :func:`_packed_kv_view`, enforcing contiguity for
    every other input."""
    packed = None if kv_nvfp4 or not paged else _packed_kv_view_cached(k, v, head_dim)
    if packed is None and not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("k/v must be contiguous")
    return packed


# Cached views pin their storage, hence the small cap.
_PACKED_VIEW_CACHE: dict = {}


def _packed_kv_view_cached(k, v, head_dim):
    """Memoized :func:`_packed_kv_view`: callers pass the same cache views
    every step, and the per-call detection cost lands on the decode hot path."""
    if k.is_contiguous() and v.is_contiguous():
        return None
    key = (k.data_ptr(), v.data_ptr(), k.shape, k.stride(), v.stride(), k.dtype)
    hit = _PACKED_VIEW_CACHE.get(key)
    if hit is None:
        if len(_PACKED_VIEW_CACHE) >= 32:
            _PACKED_VIEW_CACHE.clear()
        hit = _packed_kv_view(k, v, head_dim)
        _PACKED_VIEW_CACHE[key] = hit
    return hit


def _packed_kv_view(k, v, head_dim):
    """Recover the packed cache behind K/V views split from a paged KV cache
    that stores K and V in one ``2 * head_dim`` content dim per token.

    Returns ``None`` when both tensors are plain contiguous, else ``(packed,
    stride_order)``: the storage re-viewed as a 5-D cache with K at plane 0
    and V at plane 1 of dim 3, plus its dim ranks for ``cute.compile``. The
    view must be compact, meaning its strides telescope in some dim order (a
    permuted-contiguous NHD cache qualifies); any other layout raises."""
    if k.is_contiguous() and v.is_contiguous():
        return None
    if (
        k.ndim != 4
        or k.stride() != v.stride()
        or k.stride(-1) != 1
        or v.data_ptr() - k.data_ptr() != head_dim * k.element_size()
    ):
        raise ValueError(
            "k/v must be contiguous, or K/V views split from a paged KV cache "
            "that packs K and V in one 2*head_dim content dim per token"
        )
    num_pages, num_kv_heads, page, _ = k.shape
    packed = k.as_strided(
        (num_pages, num_kv_heads, page, 2, head_dim),
        (k.stride(0), k.stride(1), k.stride(2), head_dim, 1),
    )
    # Sized strides must telescope: the kernel derives addresses from the
    # compact layout. Size-1 dims carry arbitrary strides, so they rank
    # outermost (inner they would break the kernel's static alignment proof)
    # and get rebuilt strides.
    ones = [i for i in range(5) if packed.shape[i] == 1]
    sized = sorted(
        (i for i in range(5) if packed.shape[i] > 1), key=lambda i: packed.stride(i)
    )
    strides = [0] * 5
    expected = 1
    for i in sized:
        if packed.stride(i) != expected:
            raise ValueError(
                "packed K/V views must cover a compact KV cache, got strides "
                f"{tuple(packed.stride())} for shape {tuple(packed.shape)}"
            )
        strides[i] = expected
        expected *= packed.shape[i]
    for i in ones:
        strides[i] = expected
    packed = k.as_strided(packed.shape, strides)
    rank = [0] * 5
    for pos, dim in enumerate(sized + ones):
        rank[dim] = pos
    return packed, tuple(rank)
