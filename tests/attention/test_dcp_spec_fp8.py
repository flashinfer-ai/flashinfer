"""GPU correctness and graph tests for the production Cake FMHA DCP profile."""

from __future__ import annotations

import math

import pytest
import torch

from flashinfer.dcp import (
    get_dcp_spec_counter_bytes,
    get_dcp_spec_workspace_size_bytes,
)
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.utils import is_sm100a_supported


_HEAD_DIM = 128
_PAGE_SIZE = 64
_LOG2_E = math.log2(math.e)


def _require_blackwell_dcp() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Cake FMHA DCP requires SM100 or SM103")


def _rank_positions(rank: int, global_len: int, cp_world: int) -> torch.Tensor:
    return torch.arange(rank, global_len, cp_world, dtype=torch.long, device="cuda")


def _local_visible_count(prefix: int, row: int, rank: int, cp_world: int) -> int:
    return max(0, (prefix + row - rank) // cp_world + 1)


def _quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = max(float(x.abs().amax().item()) / fp8_max, 1.0e-8)
    storage = (x.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return storage, scale


def _pack_rank_cache(
    global_k: torch.Tensor,
    global_v: torch.Tensor,
    global_lens: list[int],
    *,
    rank: int,
    cp_world: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    batch_size, _, num_kv_heads, head_dim = global_k.shape
    local_k = []
    local_v = []
    local_lens = []
    for batch_idx, global_len in enumerate(global_lens):
        positions = _rank_positions(rank, global_len, cp_world)
        local_k.append(global_k[batch_idx, positions])
        local_v.append(global_v[batch_idx, positions])
        local_lens.append(int(positions.numel()))

    data_pages_per_seq = [
        max(1, math.ceil(length / _PAGE_SIZE)) for length in local_lens
    ]
    total_data_pages = sum(data_pages_per_seq)
    dummy_page = total_data_pages
    k_cache = torch.zeros(
        (total_data_pages + 1, num_kv_heads, _PAGE_SIZE, head_dim),
        dtype=global_k.dtype,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    max_local_seq_len = max(local_lens)
    loop_blocks = max(1, math.ceil(max_local_seq_len / 128))
    loop_blocks += loop_blocks % 2
    max_pages_per_seq = loop_blocks * 2
    block_tables = torch.full(
        (batch_size, max_pages_per_seq),
        dummy_page,
        dtype=torch.int32,
        device="cuda",
    )

    next_page = 0
    for batch_idx, (length, page_count) in enumerate(
        zip(local_lens, data_pages_per_seq, strict=True)
    ):
        pages = torch.arange(
            next_page,
            next_page + page_count,
            dtype=torch.int32,
            device="cuda",
        )
        block_tables[batch_idx, :page_count] = pages
        if length:
            padded_k = torch.zeros(
                (page_count * _PAGE_SIZE, num_kv_heads, head_dim),
                dtype=global_k.dtype,
                device="cuda",
            )
            padded_v = torch.zeros_like(padded_k)
            padded_k[:length].copy_(local_k[batch_idx])
            padded_v[:length].copy_(local_v[batch_idx])
            k_cache[pages.long()] = padded_k.reshape(
                page_count, _PAGE_SIZE, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
            v_cache[pages.long()] = padded_v.reshape(
                page_count, _PAGE_SIZE, num_kv_heads, head_dim
            ).permute(0, 2, 1, 3)
        next_page += page_count

    seq_lens = torch.tensor(local_lens, dtype=torch.int32, device="cuda")
    return k_cache, v_cache, block_tables, seq_lens, max_local_seq_len


def _rank_reference(
    query: torch.Tensor,
    global_k: torch.Tensor,
    global_v: torch.Tensor,
    prefix_lens: list[int],
    *,
    rank: int,
    cp_world: int,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, q_len, num_q_heads, _head_dim = query.shape
    num_kv_heads = global_k.shape[2]
    group_size = num_q_heads // num_kv_heads
    output = torch.zeros_like(query)
    lse = torch.full(
        (batch_size, q_len, num_q_heads),
        -float("inf"),
        dtype=torch.float32,
        device="cuda",
    )
    for batch_idx, prefix in enumerate(prefix_lens):
        positions = _rank_positions(rank, prefix + q_len, cp_world)
        keys = global_k[batch_idx, positions].repeat_interleave(group_size, dim=1)
        values = global_v[batch_idx, positions].repeat_interleave(group_size, dim=1)
        for row in range(q_len):
            visible = _local_visible_count(prefix, row, rank, cp_world)
            if visible == 0:
                continue
            scores = (
                torch.einsum(
                    "hd,khd->hk", query[batch_idx, row].float(), keys[:visible].float()
                )
                * sm_scale
            )
            output[batch_idx, row] = torch.einsum(
                "hk,khd->hd", torch.softmax(scores, dim=-1), values[:visible].float()
            ).to(torch.bfloat16)
            lse[batch_idx, row] = torch.logsumexp(scores, dim=-1) * _LOG2_E
    return output, lse


def _dense_reference(
    query: torch.Tensor,
    global_k: torch.Tensor,
    global_v: torch.Tensor,
    prefix_lens: list[int],
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, q_len, num_q_heads, _ = query.shape
    group_size = num_q_heads // global_k.shape[2]
    output = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, q_len, num_q_heads), dtype=torch.float32, device="cuda"
    )
    for batch_idx, prefix in enumerate(prefix_lens):
        keys = global_k[batch_idx, : prefix + q_len].repeat_interleave(
            group_size, dim=1
        )
        values = global_v[batch_idx, : prefix + q_len].repeat_interleave(
            group_size, dim=1
        )
        for row in range(q_len):
            visible = prefix + row + 1
            scores = (
                torch.einsum(
                    "hd,khd->hk", query[batch_idx, row].float(), keys[:visible].float()
                )
                * sm_scale
            )
            output[batch_idx, row] = torch.einsum(
                "hk,khd->hd", torch.softmax(scores, dim=-1), values[:visible].float()
            ).to(torch.bfloat16)
            lse[batch_idx, row] = torch.logsumexp(scores, dim=-1) * _LOG2_E
    return output, lse


def _merge_partials(
    partials: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    stacked_o = torch.stack([output.float() for output, _ in partials], dim=0)
    stacked_lse = torch.stack([lse for _, lse in partials], dim=0)
    merged_lse = torch.logsumexp(stacked_lse * math.log(2.0), dim=0) * _LOG2_E
    weights = torch.exp2(stacked_lse - merged_lse.unsqueeze(0))
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    merged_o = (stacked_o * weights.unsqueeze(-1)).sum(dim=0).to(torch.bfloat16)
    return merged_o, merged_lse


def _run_public_rank(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    workspace: torch.Tensor,
    counter: torch.Tensor | None = None,
    *,
    max_local_seq_len: int,
    q_len: int,
    cp_world: int,
    cp_rank: int,
    bmm1_scale: float,
    bmm2_scale: float,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return trtllm_batch_decode_with_kv_cache(
        query.flatten(0, 1),
        (k_cache, v_cache),
        workspace,
        block_tables,
        seq_lens,
        max_local_seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out.flatten(0, 1),
        kv_layout="HND",
        backend="cake",
        q_len_per_req=q_len,
        lse=lse.flatten(0, 1),
        return_lse=True,
        multi_ctas_kv_counter_buffer=counter,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=prefix_lens,
    )


def test_fp8_page64_q3_public_api_rank_correctness_and_merge() -> None:
    _require_blackwell_dcp()
    torch.manual_seed(7)
    batch_size, q_len = 2, 3
    num_q_heads, num_kv_heads = 8, 2
    cp_world = 4
    prefix_lens = [0, 65]
    max_global_len = max(prefix_lens) + q_len
    sm_scale = _HEAD_DIM**-0.5
    query = (
        torch.randn(
            batch_size,
            q_len,
            num_q_heads,
            _HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.2
    ).to(torch.bfloat16)
    k_source = (
        torch.randn(
            batch_size,
            max_global_len,
            num_kv_heads,
            _HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.2
    ).to(torch.bfloat16)
    v_source = torch.randn_like(k_source) * 0.2
    k_storage, k_scale = _quantize_fp8(k_source)
    v_storage, v_scale = _quantize_fp8(v_source)
    represented_k = k_storage.float() * k_scale
    represented_v = v_storage.float() * v_scale
    prefix_tensor = torch.tensor(prefix_lens, dtype=torch.int32, device="cuda")

    partials = []
    # This short-shard fixture is intentionally split1, so no partial buffer
    # or completion counter is required.
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    for rank in range(cp_world):
        k_cache, v_cache, block_tables, seq_lens, max_local = _pack_rank_cache(
            k_storage,
            v_storage,
            [prefix + q_len for prefix in prefix_lens],
            rank=rank,
            cp_world=cp_world,
        )
        out = torch.full_like(query, float("nan"))
        lse = torch.full(
            (batch_size, q_len, num_q_heads),
            float("nan"),
            dtype=torch.float32,
            device="cuda",
        )
        actual_o, actual_lse = _run_public_rank(
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            prefix_tensor,
            workspace,
            max_local_seq_len=max_local,
            q_len=q_len,
            cp_world=cp_world,
            cp_rank=rank,
            bmm1_scale=sm_scale * k_scale,
            bmm2_scale=v_scale,
            out=out,
            lse=lse,
        )
        actual_o = actual_o.view_as(query)
        actual_lse = actual_lse.view(batch_size, q_len, num_q_heads)
        expected_o, expected_lse = _rank_reference(
            query,
            represented_k,
            represented_v,
            prefix_lens,
            rank=rank,
            cp_world=cp_world,
            sm_scale=sm_scale,
        )
        empty_mask = torch.tensor(
            [
                [
                    _local_visible_count(prefix, row, rank, cp_world) == 0
                    for row in range(q_len)
                ]
                for prefix in prefix_lens
            ],
            dtype=torch.bool,
            device="cuda",
        )
        if empty_mask.any():
            assert torch.count_nonzero(actual_o[empty_mask]) == 0
            assert torch.isneginf(actual_lse[empty_mask]).all()
        torch.testing.assert_close(actual_o, expected_o, atol=0.1, rtol=0.1)
        torch.testing.assert_close(actual_lse, expected_lse, atol=0.1, rtol=0.1)
        partials.append((actual_o, actual_lse))

    merged_o, merged_lse = _merge_partials(partials)
    dense_o, dense_lse = _dense_reference(
        query, represented_k, represented_v, prefix_lens, sm_scale
    )
    torch.testing.assert_close(merged_o, dense_o, atol=0.1, rtol=0.1)
    torch.testing.assert_close(merged_lse, dense_lse, atol=0.1, rtol=0.1)


def test_fp8_page64_b1_split3_public_api_correctness_and_graph_replay() -> None:
    _require_blackwell_dcp()
    torch.manual_seed(11)
    batch_size, q_len = 1, 4
    num_q_heads, num_kv_heads = 64, 8
    cp_world, cp_rank = 4, 0
    prefix_len = 8192
    global_len = prefix_len + q_len
    sm_scale = _HEAD_DIM**-0.5
    query = (
        torch.randn(
            batch_size,
            q_len,
            num_q_heads,
            _HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.2
    ).to(torch.bfloat16)
    k_source = (
        torch.randn(
            batch_size,
            global_len,
            num_kv_heads,
            _HEAD_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.2
    ).to(torch.bfloat16)
    v_source = torch.randn_like(k_source) * 0.2
    k_storage, k_scale = _quantize_fp8(k_source)
    v_storage, v_scale = _quantize_fp8(v_source)
    represented_k = k_storage.float() * k_scale
    represented_v = v_storage.float() * v_scale
    k_cache, v_cache, block_tables, seq_lens, max_local = _pack_rank_cache(
        k_storage,
        v_storage,
        [global_len],
        rank=cp_rank,
        cp_world=cp_world,
    )
    prefix_lens = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
    )
    out = torch.empty_like(query)
    lse = torch.empty(
        (batch_size, q_len, num_q_heads), dtype=torch.float32, device="cuda"
    )
    workspace = torch.empty(
        get_dcp_spec_workspace_size_bytes(batch_size, q_len, num_q_heads, 3),
        dtype=torch.uint8,
        device="cuda",
    )
    counter = torch.zeros(
        get_dcp_spec_counter_bytes(batch_size, q_len, num_kv_heads),
        dtype=torch.uint8,
        device="cuda",
    )

    def run():
        return _run_public_rank(
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            prefix_lens,
            workspace,
            counter,
            max_local_seq_len=max_local,
            q_len=q_len,
            cp_world=cp_world,
            cp_rank=cp_rank,
            bmm1_scale=sm_scale * k_scale,
            bmm2_scale=v_scale,
            out=out,
            lse=lse,
        )

    expected_o, expected_lse = _rank_reference(
        query,
        represented_k,
        represented_v,
        [prefix_len],
        rank=cp_rank,
        cp_world=cp_world,
        sm_scale=sm_scale,
    )
    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected_o, atol=0.1, rtol=0.1)
    torch.testing.assert_close(lse, expected_lse, atol=0.1, rtol=0.1)
    assert torch.count_nonzero(counter) == 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    captured_o = out.clone()
    captured_lse = lse.clone()
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, captured_o, atol=0, rtol=0)
    torch.testing.assert_close(lse, captured_lse, atol=0, rtol=0)
    assert torch.count_nonzero(counter) == 0


def test_fp8_page64_b256_public_api_cuda_graph_capture_replay() -> None:
    _require_blackwell_dcp()
    batch_size, q_len = 256, 4
    num_q_heads, num_kv_heads = 64, 8
    cp_world, cp_rank = 4, 0
    prefix_len = 8192
    global_len = prefix_len + q_len
    local_len = int(_rank_positions(cp_rank, global_len, cp_world).numel())
    pages_per_seq = math.ceil(local_len / _PAGE_SIZE)
    total_data_pages = batch_size * pages_per_seq
    dummy_page = total_data_pages
    k_cache = torch.zeros(
        (total_data_pages + 1, num_kv_heads, _PAGE_SIZE, _HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    loop_blocks = math.ceil(local_len / 128)
    loop_blocks += loop_blocks % 2
    max_pages_per_seq = loop_blocks * 2
    block_tables = torch.full(
        (batch_size, max_pages_per_seq),
        dummy_page,
        dtype=torch.int32,
        device="cuda",
    )
    page_ids = torch.arange(total_data_pages, dtype=torch.int32, device="cuda").view(
        batch_size, pages_per_seq
    )
    block_tables[:, :pages_per_seq] = page_ids
    seq_lens = torch.full((batch_size,), local_len, dtype=torch.int32, device="cuda")
    prefix_lens = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
    )
    query = torch.randn(
        batch_size,
        q_len,
        num_q_heads,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = torch.empty_like(query)
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    lse = torch.empty(
        (batch_size, q_len, num_q_heads), dtype=torch.float32, device="cuda"
    )

    def run():
        return _run_public_rank(
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            prefix_lens,
            workspace,
            max_local_seq_len=local_len,
            q_len=q_len,
            cp_world=cp_world,
            cp_rank=cp_rank,
            bmm1_scale=_HEAD_DIM**-0.5,
            bmm2_scale=1.0,
            out=out,
            lse=lse,
        )

    # Prewarm this exact tensor/layout binding so its immutable TMA descriptor
    # slots exist before capture.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    expected_o = out.clone()
    expected_lse = lse.clone()
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected_o, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()


def test_fp8_page64_d256_ratio16_all_ranks_and_graph_replay() -> None:
    _require_blackwell_dcp()
    torch.manual_seed(17)
    batch_size, q_len = 1, 4
    num_q_heads, num_kv_heads, head_dim = 16, 1, 256
    cp_world = 4
    prefix_len = 32764
    global_len = prefix_len + q_len
    sm_scale = head_dim**-0.5
    query = (
        torch.randn(
            batch_size,
            q_len,
            num_q_heads,
            head_dim,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.2
    ).to(torch.bfloat16)
    k_source = (
        torch.randn(
            batch_size,
            global_len,
            num_kv_heads,
            head_dim,
            dtype=torch.float32,
            device="cuda",
        )
        * 0.2
    ).to(torch.bfloat16)
    v_source = torch.randn_like(k_source) * 0.2
    k_storage, k_scale = _quantize_fp8(k_source)
    v_storage, v_scale = _quantize_fp8(v_source)
    represented_k = k_storage.float() * k_scale
    represented_v = v_storage.float() * v_scale
    prefix_lens = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
    )
    workspace = torch.empty(
        get_dcp_spec_workspace_size_bytes(
            batch_size, q_len, num_q_heads, 8, head_dim=head_dim
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    counter = torch.zeros(
        get_dcp_spec_counter_bytes(batch_size, q_len, num_kv_heads),
        dtype=torch.uint8,
        device="cuda",
    )

    partials = []
    for rank in range(cp_world):
        k_cache, v_cache, block_tables, seq_lens, max_local = _pack_rank_cache(
            k_storage,
            v_storage,
            [global_len],
            rank=rank,
            cp_world=cp_world,
        )
        assert max_local == 8192
        out = torch.empty_like(query)
        lse = torch.empty(
            (batch_size, q_len, num_q_heads),
            dtype=torch.float32,
            device="cuda",
        )

        def run(
            query=query,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            workspace=workspace,
            counter=counter,
            max_local=max_local,
            q_len=q_len,
            cp_world=cp_world,
            rank=rank,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out=out,
            lse=lse,
        ):
            return _run_public_rank(
                query,
                k_cache,
                v_cache,
                block_tables,
                seq_lens,
                prefix_lens,
                workspace,
                counter,
                max_local_seq_len=max_local,
                q_len=q_len,
                cp_world=cp_world,
                cp_rank=rank,
                bmm1_scale=sm_scale * k_scale,
                bmm2_scale=v_scale,
                out=out,
                lse=lse,
            )

        expected_o, expected_lse = _rank_reference(
            query,
            represented_k,
            represented_v,
            [prefix_len],
            rank=rank,
            cp_world=cp_world,
            sm_scale=sm_scale,
        )
        run()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected_o, atol=0.1, rtol=0.1)
        torch.testing.assert_close(lse, expected_lse, atol=0.1, rtol=0.1)
        assert torch.count_nonzero(counter) == 0

        if rank == 0:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            captured_o = out.clone()
            captured_lse = lse.clone()
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(out, captured_o, atol=0, rtol=0)
            torch.testing.assert_close(lse, captured_lse, atol=0, rtol=0)
            assert torch.count_nonzero(counter) == 0

        partials.append((out.clone(), lse.clone()))

    merged_o, merged_lse = _merge_partials(partials)
    dense_o, dense_lse = _dense_reference(
        query, represented_k, represented_v, [prefix_len], sm_scale
    )
    torch.testing.assert_close(merged_o, dense_o, atol=0.1, rtol=0.1)
    torch.testing.assert_close(merged_lse, dense_lse, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("q_len", (1, 2, 3, 4, 5, 6, 7, 8))
def test_fp8_page64_d256_b128_reviewer_all_ranks_and_graph_replay(
    q_len: int,
) -> None:
    """Validate every admitted B128 D256 CP4 split1 CUDA Graph route."""

    _require_blackwell_dcp()
    batch_size = 128
    num_q_heads, num_kv_heads, head_dim = 16, 1, 256
    cp_world = 4
    prefix_len = 32768 - q_len
    local_len = 8192
    pages_per_seq = local_len // _PAGE_SIZE
    query = torch.zeros(
        (batch_size, q_len, num_q_heads, head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.zeros(
        (pages_per_seq, num_kv_heads, _PAGE_SIZE, head_dim),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    block_tables = torch.arange(pages_per_seq, dtype=torch.int32, device="cuda").repeat(
        batch_size, 1
    )
    seq_lens = torch.full((batch_size,), local_len, dtype=torch.int32, device="cuda")
    prefix_lens = torch.full(
        (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
    )
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")

    partials = []
    for rank in range(cp_world):
        out = torch.empty_like(query)
        lse = torch.empty(
            (batch_size, q_len, num_q_heads),
            dtype=torch.float32,
            device="cuda",
        )
        out_ptr = out.data_ptr()
        lse_ptr = lse.data_ptr()

        def run(
            rank=rank,
            out=out,
            lse=lse,
        ):
            return _run_public_rank(
                query,
                k_cache,
                v_cache,
                block_tables,
                seq_lens,
                prefix_lens,
                workspace,
                max_local_seq_len=local_len,
                q_len=q_len,
                cp_world=cp_world,
                cp_rank=rank,
                bmm1_scale=head_dim**-0.5,
                bmm2_scale=1.0,
                out=out,
                lse=lse,
            )

        expected_rows = torch.tensor(
            [
                math.log2(_local_visible_count(prefix_len, row, rank, cp_world))
                for row in range(q_len)
            ],
            dtype=torch.float32,
            device="cuda",
        )
        expected_lse = expected_rows.view(1, q_len, 1).expand_as(lse)

        run()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
        torch.testing.assert_close(lse, expected_lse, atol=0.1, rtol=0.1)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
        torch.testing.assert_close(lse, expected_lse, atol=0.1, rtol=0.1)
        assert out.data_ptr() == out_ptr
        assert lse.data_ptr() == lse_ptr
        partials.append((out.clone(), lse.clone()))

    merged_o, merged_lse = _merge_partials(partials)
    dense_rows = torch.tensor(
        [math.log2(prefix_len + row + 1) for row in range(q_len)],
        dtype=torch.float32,
        device="cuda",
    )
    expected_dense_lse = dense_rows.view(1, q_len, 1).expand_as(merged_lse)
    torch.testing.assert_close(merged_o, torch.zeros_like(merged_o), atol=0, rtol=0)
    torch.testing.assert_close(merged_lse, expected_dense_lse, atol=0.1, rtol=0.1)


def test_fp8_page64_d256_initializes_dynamic_smem_on_each_device() -> None:
    _require_blackwell_dcp()
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for per-device host-shim coverage")

    batch_size, q_len = 1, 4
    num_q_heads, num_kv_heads, head_dim = 16, 1, 256
    prefix_len = 60
    global_len = prefix_len + q_len
    sm_scale = head_dim**-0.5

    for device_index in (0, 1):
        with torch.cuda.device(device_index):
            torch.manual_seed(41 + device_index)
            query = (
                torch.randn(
                    batch_size,
                    q_len,
                    num_q_heads,
                    head_dim,
                    dtype=torch.float32,
                    device="cuda",
                )
                * 0.2
            ).to(torch.bfloat16)
            k_source = (
                torch.randn(
                    batch_size,
                    global_len,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float32,
                    device="cuda",
                )
                * 0.2
            ).to(torch.bfloat16)
            v_source = torch.randn_like(k_source) * 0.2
            k_storage, k_scale = _quantize_fp8(k_source)
            v_storage, v_scale = _quantize_fp8(v_source)
            represented_k = k_storage.float() * k_scale
            represented_v = v_storage.float() * v_scale
            prefix_lens = torch.full(
                (batch_size,), prefix_len, dtype=torch.int32, device="cuda"
            )
            k_cache, v_cache, block_tables, seq_lens, max_local = _pack_rank_cache(
                k_storage,
                v_storage,
                [global_len],
                rank=0,
                cp_world=1,
            )
            out = torch.empty_like(query)
            lse = torch.empty(
                (batch_size, q_len, num_q_heads),
                dtype=torch.float32,
                device="cuda",
            )
            workspace = torch.empty(1, dtype=torch.uint8, device="cuda")

            _run_public_rank(
                query,
                k_cache,
                v_cache,
                block_tables,
                seq_lens,
                prefix_lens,
                workspace,
                max_local_seq_len=max_local,
                q_len=q_len,
                cp_world=1,
                cp_rank=0,
                bmm1_scale=sm_scale * k_scale,
                bmm2_scale=v_scale,
                out=out,
                lse=lse,
            )
            torch.cuda.synchronize(device_index)
            expected_o, expected_lse = _dense_reference(
                query,
                represented_k,
                represented_v,
                [prefix_len],
                sm_scale,
            )
            torch.testing.assert_close(out, expected_o, atol=0.1, rtol=0.1)
            torch.testing.assert_close(lse, expected_lse, atol=0.1, rtol=0.1)
