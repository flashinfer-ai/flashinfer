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

Synthetic NVFP4 paged-KV MSA decode problems in the canonical page layout.

Builds one planar page pool exactly as ``docs/design_docs/nvfp4_msa_paged_kv_layout.md``
describes it (K data, linear K scales, V data, swizzled V scales per page), a
permuted page table, right-aligned decode tokens and a head-major top-k
selection, and returns the strided views every SM100/SM103 MSA reader
consumes.  Shared by the experimental Cake decode tests and its benchmark.
"""

from __future__ import annotations

import torch

HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
SCALE_VEC = 16
DATA_DIM = HEAD_DIM // 2
SCALE_DIM = HEAD_DIM // SCALE_VEC
K_GLOBAL_SCALE = 0.75
V_GLOBAL_SCALE = 0.85


def page_layout(num_kv_heads: int) -> dict:
    """Byte offsets of the four regions of one page (a function of Hkv only)."""
    k_scale = num_kv_heads * PAGE_SIZE * DATA_DIM
    v_data = k_scale + num_kv_heads * PAGE_SIZE * SCALE_DIM
    v_scale = v_data + k_scale
    return {
        "k_data": 0,
        "k_scale": k_scale,
        "v_data": v_data,
        "v_scale": v_scale,
        "page_bytes": v_scale + num_kv_heads * PAGE_SIZE * SCALE_DIM,
    }


def page_views(pool: torch.Tensor, num_pages: int, num_kv_heads: int):
    """``(k, k_scale, v, v_scale)`` strided uint8 views of a planar pool."""
    layout = page_layout(num_kv_heads)
    page_bytes = layout["page_bytes"]
    data_shape = (num_pages, num_kv_heads, PAGE_SIZE, DATA_DIM)
    scale_shape = (num_pages, num_kv_heads, PAGE_SIZE, SCALE_DIM)
    data_stride = (page_bytes, PAGE_SIZE * DATA_DIM, DATA_DIM, 1)
    scale_stride = (page_bytes, PAGE_SIZE * SCALE_DIM, SCALE_DIM, 1)
    return (
        torch.as_strided(pool, data_shape, data_stride, layout["k_data"]),
        torch.as_strided(pool, scale_shape, scale_stride, layout["k_scale"]),
        torch.as_strided(pool, data_shape, data_stride, layout["v_data"]),
        torch.as_strided(pool, scale_shape, scale_stride, layout["v_scale"]),
    )


def unit_rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)


def quantize_nvfp4(x: torch.Tensor, global_scale: float):
    """``(..., 128)`` floats -> packed E2M1 bytes ``(..., 64)`` and E4M3 scale bytes ``(..., 8)``.

    ``sf = e4m3(amax16 / (6 * global_scale))`` and ``q = e2m1(x / (float(sf) *
    global_scale))`` with round-to-nearest-even at the E2M1 midpoints, i.e. the
    dequant is ``e2m1 * float(sf) * global_scale``.
    """
    grouped = x.float().reshape(*x.shape[:-1], HEAD_DIM // SCALE_VEC, SCALE_VEC)
    amax = grouped.abs().amax(dim=-1)
    sf = (amax / (6.0 * float(global_scale))).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    inv = torch.where(
        sf_f > 0,
        1.0 / (sf_f.clamp(min=1e-30) * float(global_scale)),
        torch.zeros_like(sf_f),
    )
    y = grouped * inv.unsqueeze(-1)
    magnitude = y.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for bound, value, inclusive in (
        (0.25, 1, False),
        (0.75, 2, True),
        (1.25, 3, False),
        (1.75, 4, True),
        (2.5, 5, False),
        (3.5, 6, True),
        (5.0, 7, False),
    ):
        hit = magnitude >= bound if inclusive else magnitude > bound
        code = torch.where(hit, torch.full_like(code, value), code)
    codes = (code | ((y < 0).to(torch.uint8) << 3)).reshape(*x.shape[:-1], HEAD_DIM)
    packed = (codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)
    return packed.to(torch.uint8), sf.view(torch.uint8)


def v_scale_swizzle_index(device) -> torch.Tensor:
    """``idx`` such that ``swizzled_flat[idx[t * 8 + s]]`` holds linear scale ``(t, s)``."""
    t = torch.arange(PAGE_SIZE, device=device).unsqueeze(1)
    s = torch.arange(SCALE_DIM, device=device).unsqueeze(0)
    groups = SCALE_DIM // 4
    swizzled_t = (t // 4) * 4 + s // groups
    swizzled_s = (s % groups) * 4 + t % 4
    return (swizzled_t * SCALE_DIM + swizzled_s).reshape(-1)


def swizzle_v_scale(linear: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``(..., 128, 8)`` linear scales -> the cache writer's swizzled order."""
    flat = linear.reshape(*linear.shape[:-2], PAGE_SIZE * SCALE_DIM)
    out = torch.empty_like(flat)
    out[..., idx] = flat
    return out.reshape_as(linear)


def build_decode_inputs(
    seq_lens,
    *,
    num_kv_heads: int,
    group_size: int = 16,
    seqlen_q: int = 1,
    device,
    seed: int = 0,
    k_global_scale: float = K_GLOBAL_SCALE,
    v_global_scale: float = V_GLOBAL_SCALE,
    topk: int = TOPK,
    pages_per_chunk: int = 512,
) -> dict:
    """One decode step of ``len(seq_lens)`` requests against a fresh page pool.

    Token ``i`` of a request sits at position ``seq_len - seqlen_q + i``; its
    selection holds ``min(topk, own_page + 1)`` distinct pages at or before its
    own page (always including the own page, so the causal edge is exercised),
    ascending and ``-1`` padded, one independent selection per KV head.
    """
    device = torch.device(device)
    seq_lens = torch.as_tensor(list(seq_lens), dtype=torch.int32)
    batch = int(seq_lens.numel())
    if batch == 0 or int(seq_lens.min()) < seqlen_q:
        raise ValueError("every request needs at least seqlen_q KV tokens")
    num_q_heads = num_kv_heads * group_size
    blocks = (seq_lens.long() + PAGE_SIZE - 1) // PAGE_SIZE
    max_pages = int(blocks.max())
    num_pages = int(blocks.sum())
    cpu = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_pages, generator=cpu).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32)
    cursor = 0
    for request, count in enumerate(blocks.tolist()):
        page_table[request, :count] = permutation[cursor : cursor + count]
        cursor += count

    layout = page_layout(num_kv_heads)
    pool = torch.zeros(
        num_pages * layout["page_bytes"], dtype=torch.uint8, device=device
    )
    k, k_scale, v, v_scale = page_views(pool, num_pages, num_kv_heads)
    swizzle = v_scale_swizzle_index(device)
    gen = torch.Generator(device=device).manual_seed(seed + 7)
    for lo in range(0, num_pages, pages_per_chunk):
        hi = min(num_pages, lo + pages_per_chunk)
        keys = unit_rms(
            torch.randn(
                hi - lo, num_kv_heads, PAGE_SIZE, HEAD_DIM, generator=gen, device=device
            )
        )
        values = torch.randn(
            hi - lo, num_kv_heads, PAGE_SIZE, HEAD_DIM, generator=gen, device=device
        )
        packed_k, sf_k = quantize_nvfp4(keys, k_global_scale)
        packed_v, sf_v = quantize_nvfp4(values, v_global_scale)
        k[lo:hi].copy_(packed_k)
        k_scale[lo:hi].copy_(sf_k)
        v[lo:hi].copy_(packed_v)
        v_scale[lo:hi].copy_(swizzle_v_scale(sf_v, swizzle))

    total_q = batch * seqlen_q
    q = torch.randn(total_q, num_q_heads, HEAD_DIM, generator=gen, device=device).to(
        torch.bfloat16
    )

    token = torch.arange(seqlen_q)
    positions = (seq_lens.long().unsqueeze(1) - seqlen_q + token.unsqueeze(0)).reshape(
        -1
    )
    own_page = positions // PAGE_SIZE
    q2k = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32)
    select = torch.Generator().manual_seed(seed + 99)
    for row in range(total_q):
        own = int(own_page[row])
        count = min(topk, own + 1)
        for head in range(num_kv_heads):
            if count > 1:
                history = torch.randperm(own, generator=select)[: count - 1]
                chosen = torch.cat([history, torch.tensor([own])])
            else:
                chosen = torch.tensor([own])
            q2k[head, row, :count] = torch.sort(chosen).values.to(torch.int32)

    return dict(
        q=q,
        k=k,
        v=v,
        k_scale=k_scale,
        v_scale=v_scale,
        q2k_indices=q2k.to(device).contiguous(),
        page_table=page_table.to(device).contiguous(),
        seqused_k=seq_lens.to(device),
        seqlen_q=seqlen_q,
        softmax_scale=HEAD_DIM**-0.5,
        k_global_scale=float(k_global_scale),
        v_global_scale=float(v_global_scale),
        num_pages=num_pages,
        max_pages=max_pages,
        pool=pool,
    )


def upstream_route_kwargs(inputs: dict) -> dict:
    """Keyword arguments of ``flashinfer.msa_ops.msa_sparse_decode_attention`` for ``inputs``."""
    return dict(
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        seqlen_q=inputs["seqlen_q"],
        causal=True,
        softmax_scale=inputs["softmax_scale"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
    )
