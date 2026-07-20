from __future__ import annotations

import torch

from flashinfer.experimental.sm12x.attention.nsa_indexer.reference import (
    pack_index_k_cache_reference,
    contiguous_logits_reference,
    paged_decode_logits_reference,
    unpack_index_k_cache_reference,
)


_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _make_real_page_table(
    *,
    page_starts: list[int],
    seqlens: list[int],
    width_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    real_page_table = torch.full(
        (len(seqlens), width_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for row_idx, (page_start, seq_len) in enumerate(
        zip(page_starts, seqlens, strict=True)
    ):
        block_count = (int(seq_len) + 63) // 64
        if block_count:
            real_page_table[row_idx, :block_count] = torch.arange(
                page_start,
                page_start + block_count,
                dtype=torch.int32,
                device=device,
            )
    return real_page_table


def _manual_paged_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_matrix: torch.Tensor,
    real_page_table: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
) -> torch.Tensor:
    num_queries = q_fp8.shape[0]
    width_tokens = real_page_table.shape[1] * 64
    out = torch.full(
        (num_queries, width_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    q_fp32 = q_fp8.to(torch.float32)
    weights_f = weights.to(torch.float32)
    for query_row in range(int(query_row_to_batch.numel())):
        batch_row = int(query_row_to_batch[query_row].item())
        seq_len = min(int(seqlens_per_query[query_row].item()), width_tokens)
        for token_pos in range(seq_len):
            page_col = token_pos // 64
            page_id = int(real_page_table[batch_row, page_col].item())
            if page_id < 0:
                continue
            token_idx = page_id * 64 + (token_pos % 64)
            score = torch.matmul(q_fp32[query_row], k_matrix[token_idx]).relu_()
            out[query_row, token_pos] = (score * weights_f[query_row]).sum()
    return out


def _quantize_rows_to_kv_fp8(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = k.abs().amax(dim=1) / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quant = (
        (k / scale.unsqueeze(1))
        .clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    return quant, scale.to(torch.float32)


def _manual_contiguous_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_matrix: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
) -> torch.Tensor:
    num_queries = q_fp8.shape[0]
    out = torch.full(
        (num_queries, k_matrix.shape[0]),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    q_fp32 = q_fp8.to(torch.float32)
    weights_f = weights.to(torch.float32)
    for query_row in range(int(k_start.numel())):
        ks = max(0, int(k_start[query_row].item()))
        ke = min(int(k_end[query_row].item()), k_matrix.shape[0])
        if ke <= ks:
            continue
        score = torch.matmul(q_fp32[query_row], k_matrix[ks:ke].transpose(0, 1)).relu_()
        out[query_row, ks:ke] = (score * weights_f[query_row].unsqueeze(1)).sum(dim=0)
    return out


def _scalar_paged_logits_oracle(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_dequant: torch.Tensor,
    real_page_table: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
) -> torch.Tensor:
    num_queries = q_fp8.shape[0]
    width_tokens = real_page_table.shape[1] * 64
    out = torch.full(
        (num_queries, width_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    for query_row in range(int(query_row_to_batch.numel())):
        batch_row = int(query_row_to_batch[query_row].item())
        seq_len = min(max(0, int(seqlens_per_query[query_row].item())), width_tokens)
        for logical_pos in range(seq_len):
            page_id = int(real_page_table[batch_row, logical_pos // 64].item())
            if page_id < 0:
                continue
            physical = page_id * 64 + (logical_pos % 64)
            if physical < 0 or physical >= k_dequant.shape[0]:
                continue
            acc = torch.tensor(0.0, dtype=torch.float32, device=q_fp8.device)
            for head in range(q_fp8.shape[1]):
                score = torch.dot(
                    q_fp8[query_row, head].to(torch.float32),
                    k_dequant[physical].to(torch.float32),
                )
                acc = acc + torch.relu(score) * weights[query_row, head].to(
                    torch.float32
                )
            out[query_row, logical_pos] = acc
    return out


def _scalar_contiguous_logits_oracle(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_dequant: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
) -> torch.Tensor:
    out = torch.full(
        (q_fp8.shape[0], k_dequant.shape[0]),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    for query_row in range(int(k_start.numel())):
        ks = max(0, int(k_start[query_row].item()))
        ke = min(int(k_end[query_row].item()), k_dequant.shape[0])
        for key_row in range(ks, ke):
            acc = torch.tensor(0.0, dtype=torch.float32, device=q_fp8.device)
            for head in range(q_fp8.shape[1]):
                score = torch.dot(
                    q_fp8[query_row, head].to(torch.float32),
                    k_dequant[key_row].to(torch.float32),
                )
                acc = acc + torch.relu(score) * weights[query_row, head].to(
                    torch.float32
                )
            out[query_row, key_row] = acc
    return out


@torch.inference_mode()
def test_pack_nsa_index_k_cache_roundtrip_matches_input_for_odd_lengths() -> None:
    device = torch.device("cpu")
    for num_tokens in (63, 64, 65, 127, 128, 129):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(70_000 + num_tokens)
        k = (
            torch.randn(
                (num_tokens, 128), generator=gen, dtype=torch.float32, device=device
            )
            / 4
        )
        packed = pack_index_k_cache_reference(k)
        unpacked = unpack_index_k_cache_reference(packed, num_tokens=num_tokens)
        max_abs = (unpacked - k).abs().max().item()
        rmse = torch.sqrt(((unpacked - k) ** 2).mean()).item()
        assert packed.shape == (((num_tokens + 63) // 64), 64 * (128 + 4))
        assert max_abs <= 0.08, f"num_tokens={num_tokens}: max_abs={max_abs:.6f}"
        assert rmse <= 0.008, f"num_tokens={num_tokens}: rmse={rmse:.6f}"


def test_paged_decode_logits_reference_matches_manual() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(71_001)

    page_starts = [2, 6, 10]
    width_blocks = 3
    num_tokens = (max(page_starts) + width_blocks) * 64
    q_rows = 3
    num_heads = 4
    seqlens = torch.tensor([65, 96, 130], dtype=torch.int32, device=device)
    real_page_table = _make_real_page_table(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )
    k = (
        torch.randn(
            (num_tokens, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 3
    )
    index_k_cache = pack_index_k_cache_reference(k)
    unpacked = unpack_index_k_cache_reference(index_k_cache, num_tokens=num_tokens)
    q_fp8 = (
        torch.randn(
            (q_rows, num_heads, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (q_rows, num_heads), generator=gen, dtype=torch.float32, device=device
    )

    actual = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )
    expected = _manual_paged_logits(
        q_fp8=q_fp8,
        weights=weights,
        k_matrix=unpacked,
        real_page_table=real_page_table,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_contiguous_logits_reference_matches_manual() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(71_002)

    q_rows = 5
    num_heads = 3
    k_rows = 80
    q_fp8 = (
        torch.randn(
            (q_rows, num_heads, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (q_rows, num_heads), generator=gen, dtype=torch.float32, device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device)
        / 3
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_start = torch.tensor([0, 4, 12, 40, 40], dtype=torch.int32, device=device)
    k_end = torch.tensor([8, 20, 20, 56, 40], dtype=torch.int32, device=device)

    actual = contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )
    expected = _manual_contiguous_logits(
        q_fp8=q_fp8,
        weights=weights,
        k_matrix=kv_fp8[0].to(torch.float32) * kv_fp8[1].unsqueeze(1),
        k_start=k_start,
        k_end=k_end,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_paged_decode_logits_reference_matches_scalar_oracle_for_expanded_queries() -> (
    None
):
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(71_003)

    real_page_table = torch.tensor(
        [
            [4, -1, 1],
            [7, 2, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    q_rows = 4
    num_heads = 5
    num_tokens = 8 * 64
    k = (
        torch.randn(
            (num_tokens, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 5
    )
    index_k_cache = pack_index_k_cache_reference(k)
    k_dequant = unpack_index_k_cache_reference(index_k_cache, num_tokens=num_tokens)
    q_fp8 = (
        torch.randn(
            (q_rows, num_heads, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 3
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (q_rows, num_heads), generator=gen, dtype=torch.float32, device=device
    )
    query_row_to_batch = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device=device)
    seqlens_per_query = torch.tensor([65, 9, 130, 0], dtype=torch.int32, device=device)

    actual = paged_decode_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        real_page_table=real_page_table,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
    )
    expected = _scalar_paged_logits_oracle(
        q_fp8=q_fp8,
        weights=weights,
        k_dequant=k_dequant,
        real_page_table=real_page_table,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_contiguous_logits_reference_matches_scalar_oracle_for_clamped_ranges() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(71_004)

    q_rows = 4
    num_heads = 4
    k_rows = 19
    q_fp8 = (
        torch.randn(
            (q_rows, num_heads, 128), generator=gen, dtype=torch.float32, device=device
        )
        / 4
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (q_rows, num_heads), generator=gen, dtype=torch.float32, device=device
    )
    k = (
        torch.randn((k_rows, 128), generator=gen, dtype=torch.float32, device=device)
        / 4
    )
    kv_fp8 = _quantize_rows_to_kv_fp8(k)
    k_dequant = kv_fp8[0].to(torch.float32) * kv_fp8[1].unsqueeze(1)
    k_start = torch.tensor([-3, 0, 7, 18], dtype=torch.int32, device=device)
    k_end = torch.tensor([2, 0, 99, 18], dtype=torch.int32, device=device)

    actual = contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )
    expected = _scalar_contiguous_logits_oracle(
        q_fp8=q_fp8,
        weights=weights,
        k_dequant=k_dequant,
        k_start=k_start,
        k_end=k_end,
    )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
