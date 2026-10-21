from __future__ import annotations

import torch


def make_paged_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim: int = 256,
    head_dim_qk: int | None = None,
    head_dim_vo: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    page_table_width: int | None = None,
    num_pages: int | None = None,
    vllm_combined_kv: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    if len(q_seqlens) != len(cache_seqlens):
        raise ValueError("q_seqlens and cache_seqlens must have the same length")
    torch.manual_seed(seed)
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    head_dim_qk = head_dim if head_dim_qk is None else head_dim_qk
    head_dim_vo = head_dim if head_dim_vo is None else head_dim_vo
    q = torch.randn(total_q, q_heads, head_dim_qk, device=device, dtype=dtype) / 4

    pages_per_request = [
        (cache_len + page_size - 1) // page_size for cache_len in cache_seqlens
    ]
    max_pages = max(pages_per_request, default=0)
    if page_table_width is not None:
        if page_table_width < max_pages:
            raise ValueError(
                f"page_table_width={page_table_width} is smaller than max_pages={max_pages}"
            )
        max_pages = page_table_width
    total_pages_needed = sum(pages_per_request)
    if num_pages is None:
        num_pages = max(1, total_pages_needed * 2)
    if num_pages < total_pages_needed:
        raise ValueError(
            f"num_pages={num_pages} is smaller than required total {total_pages_needed}"
        )

    if vllm_combined_kv:
        # vLLM MiniMax-M3 layout: one combined cache [num_blocks, 2, page, H, D]
        # with K = cache[:, 0] and V = cache[:, 1] as STRIDED slices.
        if head_dim_qk != head_dim_vo:
            raise ValueError("vllm_combined_kv requires head_dim_qk == head_dim_vo")
        combined_kv_cache = (
            torch.randn(
                num_pages,
                2,
                page_size,
                kv_heads,
                head_dim_qk,
                device=device,
                dtype=dtype,
            )
            / 4
        )
        k_cache = combined_kv_cache[:, 0]
        v_cache = combined_kv_cache[:, 1]
    else:
        k_cache = (
            torch.randn(
                num_pages,
                page_size,
                kv_heads,
                head_dim_qk,
                device=device,
                dtype=dtype,
            )
            / 4
        )
        v_cache = (
            torch.randn(
                num_pages,
                page_size,
                kv_heads,
                head_dim_vo,
                device=device,
                dtype=dtype,
            )
            / 4
        )
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    page_order = torch.randperm(num_pages, device=device)
    cursor = 0
    for request_idx, num_req_pages in enumerate(pages_per_request):
        if num_req_pages == 0:
            continue
        page_ids = page_order[cursor : cursor + num_req_pages].to(torch.int32)
        cursor += num_req_pages
        page_table[request_idx, :num_req_pages] = page_ids
        page_table[request_idx, num_req_pages:] = page_ids[-1]

    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    q_offsets = [0]
    for q_len in q_seqlens:
        q_offsets.append(q_offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(q_offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def quantize_paged_kv_cache_e4m3(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _max_pages = page_table.shape
    _, page_size, kv_heads, _head_dim = k_cache.shape
    finfo = torch.finfo(torch.float8_e4m3fn)
    k_quant = torch.empty_like(k_cache, dtype=torch.float8_e4m3fn)
    v_quant = torch.empty_like(v_cache, dtype=torch.float8_e4m3fn)
    k_descale = torch.ones(
        (batch, kv_heads), dtype=torch.float32, device=k_cache.device
    )
    v_descale = torch.ones(
        (batch, kv_heads), dtype=torch.float32, device=v_cache.device
    )
    for request_idx in range(batch):
        cache_len = int(cache_seqlens[request_idx].item())
        num_pages = (cache_len + page_size - 1) // page_size
        if num_pages == 0:
            continue
        page_ids = page_table[request_idx, :num_pages].to(torch.long)
        k_pages = k_cache.index_select(0, page_ids).to(torch.float32)
        v_pages = v_cache.index_select(0, page_ids).to(torch.float32)
        k_scale = k_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        v_scale = v_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        k_scale = torch.where(k_scale > 0, k_scale, torch.ones_like(k_scale))
        v_scale = torch.where(v_scale > 0, v_scale, torch.ones_like(v_scale))
        k_descale[request_idx] = k_scale
        v_descale[request_idx] = v_scale
        k_quant[page_ids] = (
            (k_pages / k_scale.view(1, 1, kv_heads, 1))
            .clamp(
                min=finfo.min,
                max=finfo.max,
            )
            .to(torch.float8_e4m3fn)
        )
        v_quant[page_ids] = (
            (v_pages / v_scale.view(1, 1, kv_heads, 1))
            .clamp(
                min=finfo.min,
                max=finfo.max,
            )
            .to(torch.float8_e4m3fn)
        )
    return (
        k_quant.contiguous(),
        v_quant.contiguous(),
        k_descale.contiguous(),
        v_descale.contiguous(),
    )


def make_msa_q2k_indices(
    *,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_kv_heads: int,
    total_q_capacity: int | None = None,
    topk: int = 16,
    block_tokens: int = 128,
    seed: int = 0,
    force_block0: bool = False,
    poison_padding: bool = False,
) -> torch.Tensor:
    """Build sorted MiniMax-MSA q2k block lists for tests.

    The returned tensor has shape `[num_kv_heads, total_q_capacity, topk]`.
    Lists are batch-local block ids, sorted ascending, with the local block
    forced present and tail entries padded with -1 by default.
    """
    if cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
        raise ValueError("cache_seqlens and cu_seqlens_q must be rank-1 tensors")
    if int(block_tokens) <= 0 or int(topk) <= 0:
        raise ValueError("block_tokens and topk must be positive")
    if int(num_kv_heads) <= 0:
        raise ValueError("num_kv_heads must be positive")
    device = cache_seqlens.device
    q_offsets = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    cache_lengths = [int(v) for v in cache_seqlens.detach().cpu().tolist()]
    total_q = q_offsets[-1] if q_offsets else 0
    if total_q_capacity is None:
        total_q_capacity = total_q
    total_q_capacity = int(total_q_capacity)
    if total_q_capacity < total_q:
        raise ValueError("total_q_capacity must cover cu_seqlens_q[-1]")

    q2k = torch.full(
        (int(num_kv_heads), total_q_capacity, int(topk)),
        -1,
        dtype=torch.int32,
        device=device,
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    poison_base = 1_000_000
    for request_idx, (q_start, q_end) in enumerate(
        zip(q_offsets[:-1], q_offsets[1:], strict=False)
    ):
        qo_len = q_end - q_start
        cache_len = cache_lengths[request_idx]
        for q_row in range(q_start, q_end):
            token_local = q_row - q_start
            visible = max(token_local + cache_len - qo_len + 1, 1)
            num_visible_blocks = (visible + int(block_tokens) - 1) // int(block_tokens)
            count = min(int(topk), num_visible_blocks)
            local_block = max((visible - 1) // int(block_tokens), 0)
            candidates = [
                block for block in range(num_visible_blocks) if block != local_block
            ]
            for kv_head in range(int(num_kv_heads)):
                selected = {local_block}
                if force_block0 and local_block != 0 and len(selected) < count:
                    selected.add(0)
                remaining = count - len(selected)
                if remaining > 0 and candidates:
                    perm = torch.randperm(len(candidates), generator=gen).tolist()
                    for idx in perm:
                        block = candidates[idx]
                        if block in selected:
                            continue
                        selected.add(block)
                        if len(selected) == count:
                            break
                values = sorted(selected)
                q2k[kv_head, q_row, : len(values)] = torch.tensor(
                    values, dtype=torch.int32, device=device
                )
                if poison_padding and len(values) < int(topk):
                    pad_count = int(topk) - len(values)
                    poison = torch.arange(
                        poison_base,
                        poison_base + pad_count,
                        dtype=torch.int32,
                        device=device,
                    )
                    q2k[kv_head, q_row, len(values) :] = poison
                    poison_base += pad_count
    return q2k.contiguous()


# Harvested from b12x tests/test_attention_paged_planner.py
def _make_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int = 64,
    q_heads: int = 8,
    kv_heads: int = 1,
    head_dim_qk: int = 256,
    head_dim_vo: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    device = "cuda"
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, q_heads, head_dim_qk, dtype=dtype, device=device)
    max_pages = max(
        (cache_len + page_size - 1) // page_size for cache_len in cache_seqlens
    )
    num_pages = (
        sum((cache_len + page_size - 1) // page_size for cache_len in cache_seqlens) + 8
    )
    k_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_qk, dtype=torch.float32, device=device
    ).to(kv_dtype)
    v_cache = torch.randn(
        num_pages, page_size, kv_heads, head_dim_vo, dtype=torch.float32, device=device
    ).to(kv_dtype)
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    cursor = 0
    for request_idx, cache_len in enumerate(cache_seqlens):
        req_pages = (cache_len + page_size - 1) // page_size
        page_ids = torch.arange(
            cursor, cursor + req_pages, dtype=torch.int32, device=device
        )
        cursor += req_pages
        page_table[request_idx, :req_pages] = page_ids
        page_table[request_idx, req_pages:] = page_ids[-1]
    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    offsets = [0]
    for q_len in q_seqlens:
        offsets.append(offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(offsets, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q
