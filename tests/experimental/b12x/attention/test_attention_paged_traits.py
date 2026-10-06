from __future__ import annotations

from types import SimpleNamespace

import torch

from b12x.attention.paged.planner import create_paged_plan
from b12x.attention.paged.traits import (
    select_paged_forward_traits,
    select_paged_forward_traits_from_plan,
)

from .test_attention_paged_planner import _make_inputs


def test_paged_decode_fp8_traits_match_flashinfer_shape_rules() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[2048, 4096],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        fixed_split_size=8,
    )
    traits = select_paged_forward_traits_from_plan(plan)

    assert traits.cta_tile_q == 16
    assert traits.num_warps_q == 1
    assert traits.num_warps_kv == 4
    assert traits.num_mma_q == 1
    assert traits.num_mma_kv == 1
    assert traits.cta_tile_kv == 64
    assert traits.q_smem_bytes == 16 * 256 * 2
    assert traits.shared_storage_bytes == 66048


def test_paged_extend_fp8_traits_expand_kv_tile_without_duplicate_bf16_cache() -> None:
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = _make_inputs(
        q_seqlens=[6, 5, 7, 4],
        cache_seqlens=[97, 81, 113, 68],
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
    )
    traits = select_paged_forward_traits_from_plan(plan)

    assert traits.cta_tile_q == 64
    assert traits.num_warps_q == 4
    assert traits.num_warps_kv == 1
    assert traits.num_mma_q == 1
    assert traits.num_mma_kv == 2
    assert traits.cta_tile_kv == 32
    assert traits.shared_storage_bytes == 49152


def test_paged_fp8_traits_use_more_kv_mmas_than_bf16_traits() -> None:
    fp8_traits = select_paged_forward_traits(
        cta_tile_q=32,
        head_dim_qk=256,
        head_dim_vo=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    bf16_traits = select_paged_forward_traits(
        cta_tile_q=64,
        head_dim_qk=256,
        head_dim_vo=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        device="cuda",
    )

    assert fp8_traits.num_mma_kv == 4
    assert bf16_traits.num_mma_kv == 1
    assert fp8_traits.cta_tile_kv > bf16_traits.cta_tile_kv


def test_low_sm_long_bf16_extend_amortizes_wider_kv_tile(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            shared_memory_per_multiprocessor=102400,
            shared_memory_per_block_optin=101376,
            multi_processor_count=48,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 1),
    )

    def make_plan(*, total_q: int, kv_chunk_size: int) -> SimpleNamespace:
        return SimpleNamespace(
            mode="extend",
            enable_cuda_graph=True,
            split_kv=False,
            msa_block_sparse=False,
            window_left=-1,
            page_size=64,
            cta_tile_q=64,
            head_dim_qk=256,
            head_dim_vo=256,
            num_q_heads=24,
            num_kv_heads=4,
            gqa_group_size=6,
            dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            total_q=total_q,
            num_qo_tiles=(total_q + 63) // 64,
            kv_chunk_size=kv_chunk_size,
            page_table_shape=(1, 512),
            device=torch.device("cuda", 0),
        )

    short_traits = select_paged_forward_traits_from_plan(
        make_plan(total_q=4096, kv_chunk_size=12288)
    )
    long_traits = select_paged_forward_traits_from_plan(
        make_plan(total_q=4096, kv_chunk_size=13312)
    )

    assert short_traits.num_mma_kv == 1
    assert short_traits.cta_tile_kv == 16
    assert short_traits.num_ctas_per_sm == 2
    assert long_traits.num_mma_kv == 2
    assert long_traits.cta_tile_kv == 32
    assert long_traits.num_ctas_per_sm == 1


def test_high_sm_bf16_extend_uses_grid_coverage_and_loop_count(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            shared_memory_per_multiprocessor=102400,
            shared_memory_per_block_optin=101376,
            multi_processor_count=188,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )

    def make_plan(*, batch: int, kv_chunk_size: int) -> SimpleNamespace:
        gqa_group_size = 6
        return SimpleNamespace(
            mode="extend",
            enable_cuda_graph=True,
            split_kv=False,
            msa_block_sparse=False,
            window_left=-1,
            page_size=64,
            cta_tile_q=64,
            head_dim_qk=256,
            head_dim_vo=256,
            num_q_heads=24,
            num_kv_heads=4,
            gqa_group_size=gqa_group_size,
            dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            total_q=batch * 64,
            num_qo_tiles=batch * gqa_group_size + batch - 1,
            kv_chunk_size=kv_chunk_size,
            page_table_shape=(batch, 2),
            device=torch.device("cuda", 0),
        )

    underfilled_traits = select_paged_forward_traits_from_plan(
        make_plan(batch=1, kv_chunk_size=64)
    )
    short_traits = select_paged_forward_traits_from_plan(
        make_plan(batch=8, kv_chunk_size=64)
    )
    longer_traits = select_paged_forward_traits_from_plan(
        make_plan(batch=8, kv_chunk_size=128)
    )

    assert underfilled_traits.cta_tile_kv == 32
    assert short_traits.cta_tile_kv == 16
    assert longer_traits.cta_tile_kv == 32


def test_bf16_extend_can_force_narrow_tile(monkeypatch) -> None:
    monkeypatch.setenv("B12X_PAGED_EXTEND_BF16_N16", "1")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            shared_memory_per_multiprocessor=102400,
            shared_memory_per_block_optin=101376,
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 1))
    plan = SimpleNamespace(
        mode="extend",
        enable_cuda_graph=True,
        split_kv=False,
        msa_block_sparse=False,
        window_left=-1,
        page_size=64,
        cta_tile_q=64,
        head_dim_qk=256,
        head_dim_vo=256,
        num_q_heads=24,
        num_kv_heads=4,
        gqa_group_size=6,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        total_q=2048,
        num_qo_tiles=32,
        kv_chunk_size=32768,
        page_table_shape=(1, 512),
        device=torch.device("cuda", 0),
    )

    traits = select_paged_forward_traits_from_plan(plan)

    assert traits.num_mma_kv == 1
    assert traits.cta_tile_kv == 16
    assert traits.num_ctas_per_sm == 2


def test_paged_bf16_decode_traits_cover_padded_sync_storage_for_128_vo() -> None:
    traits = select_paged_forward_traits(
        cta_tile_q=16,
        head_dim_qk=128,
        head_dim_vo=128,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        device="cuda",
    )

    padded_sync_bytes = (
        traits.num_warps_kv
        * traits.cta_tile_q
        * (traits.head_dim_vo + 24)
        * 4
        + traits.num_warps_kv * traits.cta_tile_q * 8
    )
    assert traits.shared_storage_bytes >= padded_sync_bytes


def test_paged_fp8_decode_traits_keep_minimax_head128_tile_within_page() -> None:
    traits = select_paged_forward_traits(
        cta_tile_q=16,
        head_dim_qk=128,
        head_dim_vo=128,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        device="cuda",
    )

    assert traits.num_mma_kv == 1
    assert traits.cta_tile_kv == 64


def test_sm121_fp8_gqa6_h256_decode_compacts_sync_storage(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            shared_memory_per_multiprocessor=102400,
            shared_memory_per_block_optin=101376,
        ),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 1))
    plan = SimpleNamespace(
        mode="decode",
        enable_cuda_graph=True,
        split_kv=True,
        msa_block_sparse=False,
        window_left=-1,
        page_size=64,
        cta_tile_q=16,
        head_dim_qk=256,
        head_dim_vo=256,
        num_q_heads=24,
        num_kv_heads=4,
        gqa_group_size=6,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        device=torch.device("cuda", 0),
    )

    traits = select_paged_forward_traits_from_plan(plan)

    assert traits.cta_tile_kv == 64
    assert traits.shared_storage_bytes == 40960
    assert traits.launch_smem_bytes == 41984
    assert traits.num_ctas_per_sm == 2

    monkeypatch.setenv("B12X_PAGED_GQA6_COMPACT_SYNC", "0")
    baseline_traits = select_paged_forward_traits_from_plan(plan)
    assert baseline_traits.shared_storage_bytes == 66048
    assert baseline_traits.launch_smem_bytes == 67584
    assert baseline_traits.num_ctas_per_sm == 1


def test_laguna_page128_split_graph_traits_consume_one_physical_page(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            shared_memory_per_multiprocessor=102400,
            shared_memory_per_block_optin=101376,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )
    plan = SimpleNamespace(
        mode="decode",
        enable_cuda_graph=True,
        split_kv=True,
        msa_block_sparse=False,
        window_left=-1,
        page_size=128,
        cta_tile_q=16,
        head_dim_qk=128,
        head_dim_vo=128,
        num_q_heads=36,
        num_kv_heads=4,
        gqa_group_size=9,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        device=torch.device("cuda", 0),
    )

    traits = select_paged_forward_traits_from_plan(plan)

    assert traits.num_mma_kv == 2
    assert traits.cta_tile_kv == 128
    assert traits.shared_storage_bytes == 36864
    # The canonical typed storage keeps its 1,024-byte TMA payload alignment:
    # two 16-byte barrier arrays precede a 36,864-byte payload at offset 1,024.
    assert traits.launch_smem_bytes == 37888
    assert traits.num_ctas_per_sm == 2


def test_laguna_page128_traits_reject_same_gqa_nonproduction_head_counts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            shared_memory_per_multiprocessor=102400,
            shared_memory_per_block_optin=101376,
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (12, 0),
    )
    for num_q_heads, num_kv_heads in ((18, 2), (72, 8)):
        plan = SimpleNamespace(
            mode="decode",
            enable_cuda_graph=True,
            split_kv=True,
            msa_block_sparse=False,
            window_left=-1,
            page_size=128,
            cta_tile_q=16,
            head_dim_qk=128,
            head_dim_vo=128,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            gqa_group_size=9,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            device=torch.device("cuda", 0),
        )

        traits = select_paged_forward_traits_from_plan(plan)

        assert traits.num_mma_kv == 1
        assert traits.cta_tile_kv == 64
