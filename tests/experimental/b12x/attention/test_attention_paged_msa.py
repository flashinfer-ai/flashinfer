from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.paged.reference import (
    materialize_paged_kv_cache,
    msa_attention_reference,
)
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import (
    B12XPagedAttentionScratchCaps,
    plan_paged_attention_scratch,
)
from b12x.attention.paged.planner import create_paged_plan

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import (
    make_msa_q2k_indices,
    make_paged_inputs,
    quantize_paged_kv_cache_e4m3,
)


def _msa_dense_mask_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, q_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    q_per_kv = q_heads // kv_heads
    out = torch.empty(
        (total_q, q_heads, v_cache.shape[-1]), dtype=q.dtype, device=q.device
    )
    lse = torch.empty((total_q, q_heads), dtype=torch.float32, device=q.device)
    q_offsets = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    scale = head_dim**-0.5

    for request_idx, (q_start, q_end) in enumerate(
        zip(q_offsets[:-1], q_offsets[1:], strict=False)
    ):
        cache_len = int(cache_seqlens[request_idx].item())
        qo_len = q_end - q_start
        k_full, v_full = materialize_paged_kv_cache(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            request_idx=request_idx,
        )
        for q_row in range(q_start, q_end):
            token_local = q_row - q_start
            causal_limit = token_local + cache_len - qo_len
            for q_head in range(q_heads):
                kv_head = q_head // q_per_kv
                scores = (
                    torch.matmul(
                        k_full[:, kv_head].to(torch.float32),
                        q[q_row, q_head].to(torch.float32),
                    )
                    * scale
                )
                mask = torch.ones((cache_len,), dtype=torch.bool, device=q.device)
                for block_id_raw in q2k_indices[kv_head, q_row].detach().cpu().tolist():
                    block_id = int(block_id_raw)
                    if block_id < 0:
                        continue
                    start = block_id * 128
                    end = min(start + 128, causal_limit + 1, cache_len)
                    if end > start:
                        mask[start:end] = False
                scores = scores.masked_fill(mask, float("-inf"))
                probs = torch.softmax(scores, dim=0)
                out[q_row, q_head].copy_(
                    torch.matmul(probs, v_full[:, kv_head].to(torch.float32)).to(
                        q.dtype
                    )
                )
                lse[q_row, q_head] = torch.logsumexp(scores, dim=0)
    return out, lse


def _assert_close(
    out: torch.Tensor, ref: torch.Tensor, lse: torch.Tensor, ref_lse: torch.Tensor
) -> None:
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=1e-5)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.99999


def _run_msa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    fixed_split_size: int | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="decode",
        msa_block_sparse=True,
        fixed_split_size=-1 if fixed_split_size is None else fixed_split_size,
    )
    page_size = int(k_cache.shape[1])
    assert plan.split_kv is True
    assert plan.kv_chunk_size % 64 == 0
    assert plan.kv_chunk_size % page_size == 0
    assert page_size <= plan.kv_chunk_size <= 32 * page_size
    assert plan.total_num_partial_rows >= plan.new_batch_size
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.padded_batch_size, 1),
            max_partial_rows=plan.total_num_partial_rows,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty(
        (q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        fixed_split_size=fixed_split_size,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    return out, _lse_base2_to_natural(lse_base2)


def _run_msa_extend(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    msa_union_tile: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        msa_block_sparse=True,
        msa_union_tile=msa_union_tile,
    )
    assert plan.split_kv is False
    assert plan.cta_tile_q in (16, 128)
    assert plan.new_batch_size > 0
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="extend",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=0,
            num_cache_pages=k_cache.shape[0],
            msa_block_sparse=True,
            msa_union_tile=msa_union_tile,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty(
        (q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse_base2 = paged_attention_forward(binding=binding)
    return out, _lse_base2_to_natural(lse_base2)


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_attention_reference_matches_dense_mask_small_decode(
    page_size: int,
) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[129, 513],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1201,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=7,
        force_block0=True,
    )

    out, lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    ref, ref_lse = _msa_dense_mask_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close(out, ref, lse, ref_lse)


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_attention_reference_ignores_poisoned_padding(page_size: int) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[127],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1202,
    )
    clean = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=11,
    )
    poisoned = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=11,
        poison_padding=True,
    )

    out, lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, clean
    )
    poison_out, poison_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, poisoned
    )
    _assert_close(poison_out, out, poison_lse, lse)


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_attention_reference_handles_varlen_extend_causality(
    page_size: int,
) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[5, 3],
        cache_seqlens=[200, 384],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1203,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        total_q_capacity=16,
        seed=19,
        force_block0=True,
    )

    out, lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    ref, ref_lse = _msa_dense_mask_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    _assert_close(out, ref, lse, ref_lse)


def _lse_base2_to_natural(lse: torch.Tensor) -> torch.Tensor:
    return lse * math.log(2.0)


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_decode_eager_bf16_matches_reference_tail_cases(page_size: int) -> None:
    require_b12x()
    cache_lens = [1, 64, 127, 128, 129, 200, 2047, 2048, 5000]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * len(cache_lens),
        cache_seqlens=cache_lens,
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1301,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=31,
        force_block0=True,
        poison_padding=True,
    )

    out, lse = _run_msa_decode(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    ref, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.999


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_decode_eager_bf16_split_chunk_invariance(page_size: int) -> None:
    require_b12x()
    cache_lens = [1, 64, 127, 128, 129, 200, 2047, 2048, 5000]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * len(cache_lens),
        cache_seqlens=cache_lens,
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1302,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=37,
        force_block0=True,
        poison_padding=True,
    )
    ref, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )

    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    # Same token-domain chunk sweep at both page sizes (chunks are in native pages).
    fixed_split_sizes = (1, 2, 4, 32) if page_size == 64 else (1, 2, 4, 16)
    for fixed_split_size in fixed_split_sizes:
        out, lse = _run_msa_decode(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q2k_indices,
            fixed_split_size=fixed_split_size,
        )
        torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
        cosine = torch.nn.functional.cosine_similarity(
            out.to(torch.float32).reshape(-1),
            ref.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        assert cosine >= 0.999
        outputs.append(out.detach().clone())
        lses.append(lse.detach().clone())

    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0], rtol=3e-3, atol=3e-3)
    for lse in lses[1:]:
        torch.testing.assert_close(lse, lses[0], rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    "page_size,vllm_combined,fixed_split_size",
    [
        (64, False, None),
        (128, False, None),
        (128, False, 2),
        (128, True, None),
        (128, True, 1),
        (128, True, 4),
    ],
)
def test_msa_decode_eager_fp8_kv_matches_reference(
    page_size: int, vllm_combined: bool, fixed_split_size: int | None
) -> None:
    require_b12x()
    cache_lens = [129, 2048, 5000]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * len(cache_lens),
        cache_seqlens=cache_lens,
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1303,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=41,
        force_block0=True,
        poison_padding=True,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    if vllm_combined:
        # vLLM combined [N, 2, page, H, D] fp8 cache; K/V are strided slices.
        combined = torch.stack([k_fp8, v_fp8], dim=1)
        k_fp8 = combined[:, 0]
        v_fp8 = combined[:, 1]
        assert not k_fp8.is_contiguous() and not v_fp8.is_contiguous()

    out, lse = _run_msa_decode(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        fixed_split_size=fixed_split_size,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=5e-2, atol=5e-2)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.995


@pytest.mark.parametrize(
    "page_size,msa_union_tile",
    [(64, None), (64, False), (128, None), (128, False)],
)
def test_msa_extend_eager_bf16_matches_reference_varlen(
    page_size: int, msa_union_tile: bool | None
) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 5, 300],
        cache_seqlens=[129, 384, 640],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1304,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=43,
        force_block0=True,
        poison_padding=True,
    )

    out, lse = _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        msa_union_tile=msa_union_tile,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.999


@pytest.mark.parametrize(
    "page_size,msa_union_tile,vllm_combined",
    [
        (64, None, False),
        (128, None, False),
        (128, None, True),
    ],
)
def test_msa_extend_eager_fp8_kv_matches_reference_varlen(
    page_size: int, msa_union_tile: bool | None, vllm_combined: bool
) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 5, 300],
        cache_seqlens=[129, 384, 640],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1306,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=47,
        force_block0=True,
        poison_padding=True,
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    if vllm_combined:
        combined = torch.stack([k_fp8, v_fp8], dim=1)
        k_fp8 = combined[:, 0]
        v_fp8 = combined[:, 1]
        assert not k_fp8.is_contiguous()

    out, lse = _run_msa_extend(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
        msa_union_tile=msa_union_tile,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=5e-2, atol=5e-2)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.995


def test_msa_extend_eager_fp8_page128_minimax_vllm_shape_matches_reference() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[396],
        cache_seqlens=[396],
        page_size=128,
        q_heads=16,
        kv_heads=1,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=2301,
        page_table_width=1024,
        num_pages=64,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=1,
        total_q_capacity=4096,
        seed=2302,
        force_block0=True,
        poison_padding=True,
    )

    k_fp8 = k_cache.to(torch.float8_e4m3fn)
    v_fp8 = v_cache.to(torch.float8_e4m3fn)
    combined = torch.stack([k_fp8, v_fp8], dim=1)
    k_fp8 = combined[:, 0]
    v_fp8 = combined[:, 1]
    assert not k_fp8.is_contiguous()

    unit_descale = torch.ones((), dtype=torch.float32, device=q.device).expand(1)
    assert int(unit_descale.stride(0)) == 0
    out, lse = _run_msa_extend(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=unit_descale,
        v_descale=unit_descale,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=unit_descale,
        v_descale=unit_descale,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=5e-2, atol=5e-2)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.995


def test_msa_extend_fp8_worklist_capacity_is_compile_keyed() -> None:
    require_b12x()
    clear_attention_caches()

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[396],
        cache_seqlens=[396],
        page_size=128,
        q_heads=16,
        kv_heads=1,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=2401,
        page_table_width=1024,
        num_pages=64,
    )
    k_fp8 = k_cache.to(torch.float8_e4m3fn)
    v_fp8 = v_cache.to(torch.float8_e4m3fn)
    combined = torch.stack([k_fp8, v_fp8], dim=1)
    k_fp8 = combined[:, 0]
    v_fp8 = combined[:, 1]
    unit_descale = torch.ones((), dtype=torch.float32, device=q.device).expand(1)

    q_small = q[:208]
    cache_seqlens_small = torch.tensor([208], dtype=torch.int32, device=q.device)
    cu_seqlens_small = torch.tensor([0, 208], dtype=torch.int32, device=q.device)
    q2k_small = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens_small,
        cu_seqlens_q=cu_seqlens_small,
        num_kv_heads=1,
        total_q_capacity=4096,
        seed=2402,
        force_block0=True,
        poison_padding=True,
    )
    _run_msa_extend(
        q_small,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens_small,
        cu_seqlens_small,
        q2k_small,
        k_descale=unit_descale,
        v_descale=unit_descale,
    )

    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=1,
        total_q_capacity=4096,
        seed=2403,
        force_block0=True,
        poison_padding=True,
    )
    out, lse = _run_msa_extend(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=unit_descale,
        v_descale=unit_descale,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=unit_descale,
        v_descale=unit_descale,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=5e-2, atol=5e-2)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.995


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_extend_rejects_per_token_fp8(page_size: int) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[5],
        cache_seqlens=[384],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1307,
    )
    k_fp8 = k_cache.to(torch.float8_e4m3fn)
    v_fp8 = v_cache.to(torch.float8_e4m3fn)
    # Per-token (non-union) fp8 extend hits a known-broken dequant path and is
    # rejected at plan time; union-tile fp8 extend is the supported path.
    with pytest.raises(TypeError, match="union-tile"):
        create_paged_plan(
            q,
            k_fp8,
            v_fp8,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode="extend",
            msa_block_sparse=True,
            msa_union_tile=False,
        )


@pytest.mark.parametrize("page_size", [64, 128])
def test_msa_extend_qo_len_one_matches_decode(page_size: int) -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[129, 2048],
        page_size=page_size,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1305,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=47,
        force_block0=True,
    )

    extend_out, extend_lse = _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    decode_out, decode_lse = _run_msa_decode(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(extend_lse, decode_lse, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(extend_out, decode_out, rtol=3e-3, atol=3e-3)


@torch.inference_mode()
@pytest.mark.parametrize(
    "page_size,vllm_combined_kv,kv_dtype",
    [
        (64, False, "bf16"),
        (128, False, "bf16"),
        (128, True, "bf16"),
        (128, True, "fp8"),
    ],
)
def test_msa_decode_cuda_graph_replays_with_mutating_metadata_and_q2k(
    page_size: int, vllm_combined_kv: bool, kv_dtype: str
) -> None:
    require_b12x()
    clear_attention_caches()

    batch = 2
    page_table_width = 80
    num_pages = 512
    kv_is_fp8 = kv_dtype == "fp8"
    lse_tol = 5e-2 if kv_is_fp8 else 2e-3
    cos_min = 0.995 if kv_is_fp8 else 0.999

    def make_case(cache_lens: list[int], *, seed: int):
        q_c, k_c, v_c, table_c, seqlens_c, cu_c = make_paged_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=cache_lens,
            page_size=page_size,
            q_heads=64,
            kv_heads=4,
            head_dim=128,
            dtype=torch.bfloat16,
            seed=seed,
            page_table_width=page_table_width,
            num_pages=num_pages,
            vllm_combined_kv=vllm_combined_kv and not kv_is_fp8,
        )
        if not kv_is_fp8:
            return q_c, k_c, v_c, table_c, seqlens_c, cu_c, None, None
        k_q, v_q, k_ds, v_ds = quantize_paged_kv_cache_e4m3(
            k_c, v_c, table_c, seqlens_c
        )
        if vllm_combined_kv:
            combined = torch.stack([k_q, v_q], dim=1)
            k_q = combined[:, 0]
            v_q = combined[:, 1]
            assert not k_q.is_contiguous()
        return q_c, k_q, v_q, table_c, seqlens_c, cu_c, k_ds, v_ds

    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale,
        v_descale,
    ) = make_case(
        [2048, 5000],
        seed=1401,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        total_q_capacity=batch,
        seed=1402,
        force_block0=True,
    )
    q2k_data_ptr = int(q2k_indices.data_ptr())

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=batch * 32,
            max_partial_rows=batch * 32,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    out, lse = paged_attention_forward(binding=binding)
    torch.cuda.synchronize()
    ref_out, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.testing.assert_close(lse * math.log(2.0), ref_lse, rtol=lse_tol, atol=lse_tol)
    assert (
        torch.nn.functional.cosine_similarity(
            out.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= cos_min
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)

    graph.replay()
    torch.cuda.synchronize()
    lse_view = binding.scratch.current_lse_view() * math.log(2.0)
    torch.testing.assert_close(lse_view, ref_lse, rtol=lse_tol, atol=lse_tol)
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= cos_min
    )

    (
        q_next,
        k_next,
        v_next,
        page_table_next,
        cache_seqlens_next,
        cu_seqlens_q_next,
        k_descale_next,
        v_descale_next,
    ) = make_case([1, 129], seed=1403)
    q2k_next = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        num_kv_heads=4,
        total_q_capacity=batch,
        seed=1404,
        force_block0=True,
    )
    q.copy_(q_next)
    k_cache.copy_(k_next)
    v_cache.copy_(v_next)
    if k_descale is not None:
        k_descale.copy_(k_descale_next)
        v_descale.copy_(v_descale_next)
    q2k_indices.copy_(q2k_next)
    assert int(q2k_indices.data_ptr()) == q2k_data_ptr
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table_next,
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        q2k_indices=q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    graph.replay()
    torch.cuda.synchronize()
    ref_out_next, ref_lse_next = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table_next,
        cache_seqlens_next,
        cu_seqlens_q_next,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    lse_next = binding.scratch.current_lse_view() * math.log(2.0)
    torch.testing.assert_close(lse_next, ref_lse_next, rtol=lse_tol, atol=lse_tol)
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out_next.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= cos_min
    )


@torch.inference_mode()
def test_msa_decode_cuda_graph_captures_minimax_metadata_update_contract() -> None:
    require_b12x()
    clear_attention_caches()

    batch = 4
    page_size = 128
    page_table_width = 80
    num_pages = 512
    lse_tol = 5e-2
    cos_min = 0.995

    def make_case(cache_lens: list[int], *, seed: int):
        q_c, k_c, v_c, table_c, seqlens_c, cu_c = make_paged_inputs(
            q_seqlens=[1] * batch,
            cache_seqlens=cache_lens,
            page_size=page_size,
            q_heads=16,
            kv_heads=1,
            head_dim=128,
            dtype=torch.bfloat16,
            seed=seed,
            page_table_width=page_table_width,
            num_pages=num_pages,
        )
        k_q, v_q, k_ds, v_ds = quantize_paged_kv_cache_e4m3(
            k_c, v_c, table_c, seqlens_c
        )
        combined = torch.stack([k_q, v_q], dim=1)
        k_q = combined[:, 0]
        v_q = combined[:, 1]
        assert not k_q.is_contiguous()
        return (
            q_c,
            k_q,
            v_q,
            table_c,
            seqlens_c,
            cu_c,
            k_ds,
            v_ds,
        )

    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale,
        v_descale,
    ) = make_case([2048, 5000, 129, 4096], seed=1601)
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=1,
        total_q_capacity=batch,
        seed=1602,
        force_block0=True,
    )
    q2k_data_ptr = int(q2k_indices.data_ptr())
    page_table_data_ptr = int(page_table.data_ptr())
    cache_seqlens_data_ptr = int(cache_seqlens.data_ptr())
    cu_seqlens_q_data_ptr = int(cu_seqlens_q.data_ptr())

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=page_size,
            max_total_q=batch,
            max_batch=batch,
            max_page_table_width=page_table_width,
            max_work_items=batch * 32,
            max_partial_rows=batch * 32,
            num_cache_pages=num_pages,
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)

    def bind_current():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            q2k_indices=q2k_indices,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    binding = bind_current()
    out, lse = paged_attention_forward(binding=binding)
    torch.cuda.synchronize()
    ref_out, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    torch.testing.assert_close(lse * math.log(2.0), ref_lse, rtol=lse_tol, atol=lse_tol)
    assert (
        torch.nn.functional.cosine_similarity(
            out.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= cos_min
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        binding = bind_current()
        paged_attention_forward(binding=binding)

    graph.replay()
    torch.cuda.synchronize()
    lse_view = binding.scratch.current_lse_view() * math.log(2.0)
    torch.testing.assert_close(lse_view, ref_lse, rtol=lse_tol, atol=lse_tol)
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= cos_min
    )

    (
        q_next,
        k_next,
        v_next,
        page_table_next,
        cache_seqlens_next,
        cu_seqlens_q_next,
        k_descale_next,
        v_descale_next,
    ) = make_case([1, 129, 2048, 5000], seed=1603)
    q2k_next = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        num_kv_heads=1,
        total_q_capacity=batch,
        seed=1604,
        force_block0=True,
    )
    q.copy_(q_next)
    k_cache.copy_(k_next)
    v_cache.copy_(v_next)
    page_table.copy_(page_table_next)
    cache_seqlens.copy_(cache_seqlens_next)
    cu_seqlens_q.copy_(cu_seqlens_q_next)
    if k_descale is not None:
        assert k_descale_next is not None
        k_descale.copy_(k_descale_next)
    if v_descale is not None:
        assert v_descale_next is not None
        v_descale.copy_(v_descale_next)
    q2k_indices.copy_(q2k_next)
    assert int(q2k_indices.data_ptr()) == q2k_data_ptr
    assert int(page_table.data_ptr()) == page_table_data_ptr
    assert int(cache_seqlens.data_ptr()) == cache_seqlens_data_ptr
    assert int(cu_seqlens_q.data_ptr()) == cu_seqlens_q_data_ptr

    graph.replay()
    torch.cuda.synchronize()
    ref_out_next, ref_lse_next = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    lse_next = binding.scratch.current_lse_view() * math.log(2.0)
    torch.testing.assert_close(lse_next, ref_lse_next, rtol=lse_tol, atol=lse_tol)
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out_next.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= cos_min
    )


@torch.inference_mode()
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_msa_decode_cuda_graph_replays_minimax_bucket1_after_prefill(
    kv_dtype: str,
) -> None:
    require_b12x()
    clear_attention_caches()

    page_size = 128
    page_table_width = 1024
    num_pages = 2048
    lse_tol = 5e-2
    cos_min = 0.995

    def make_case(cache_len: int, *, seed: int):
        q_c, k_c, v_c, table_c, seqlens_c, cu_c = make_paged_inputs(
            q_seqlens=[1],
            cache_seqlens=[cache_len],
            page_size=page_size,
            q_heads=16,
            kv_heads=1,
            head_dim=128,
            dtype=torch.bfloat16,
            seed=seed,
            page_table_width=page_table_width,
            num_pages=num_pages,
            vllm_combined_kv=kv_dtype == "bf16",
        )
        if kv_dtype == "bf16":
            return (
                q_c,
                k_c,
                v_c,
                table_c,
                seqlens_c,
                cu_c,
                None,
                None,
            )
        k_q, v_q, k_ds, v_ds = quantize_paged_kv_cache_e4m3(
            k_c, v_c, table_c, seqlens_c
        )
        combined = torch.stack([k_q, v_q], dim=1)
        k_q = combined[:, 0]
        v_q = combined[:, 1]
        assert not k_q.is_contiguous()
        return (
            q_c,
            k_q,
            v_q,
            table_c,
            seqlens_c,
            cu_c,
            k_ds[:, 0].contiguous(),
            v_ds[:, 0].contiguous(),
        )

    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale,
        v_descale,
    ) = make_case(1, seed=2001)
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=1,
        total_q_capacity=1,
        seed=2002,
        force_block0=True,
    )
    q2k_source = q2k_indices.clone()
    q2k_data_ptr = int(q2k_indices.data_ptr())
    q2k_source_data_ptr = int(q2k_source.data_ptr())
    page_table_data_ptr = int(page_table.data_ptr())
    cache_seqlens_data_ptr = int(cache_seqlens.data_ptr())
    cu_seqlens_q_data_ptr = int(cu_seqlens_q.data_ptr())

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=page_size,
            max_total_q=1,
            max_batch=1,
            max_page_table_width=page_table_width,
            max_work_items=32,
            max_partial_rows=32,
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=1,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)

    def bind_current():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            q2k_indices=q2k_indices,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    paged_attention_forward(binding=bind_current())
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q2k_indices.copy_(q2k_source)
        binding = bind_current()
        paged_attention_forward(binding=binding)

    with pytest.raises(RuntimeError, match="cannot replace decode graph replay state"):
        scratch_plan.prepare_decode_graph_replay_state(
            batch=1,
            max_page_table_width=page_table_width,
            max_cache_page_count=page_table_width,
        )
    graph.replay()
    torch.cuda.synchronize()

    (
        q_next,
        k_next,
        v_next,
        page_table_next,
        cache_seqlens_next,
        cu_seqlens_q_next,
        k_descale_next,
        v_descale_next,
    ) = make_case(19, seed=2003)
    q2k_next = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        num_kv_heads=1,
        total_q_capacity=1,
        seed=2004,
        force_block0=True,
    )
    q.copy_(q_next)
    k_cache.copy_(k_next)
    v_cache.copy_(v_next)
    page_table.copy_(page_table_next)
    cache_seqlens.copy_(cache_seqlens_next)
    cu_seqlens_q.copy_(cu_seqlens_q_next)
    if k_descale is not None:
        assert k_descale_next is not None
        k_descale.copy_(k_descale_next)
    if v_descale is not None:
        assert v_descale_next is not None
        v_descale.copy_(v_descale_next)
    q2k_source.copy_(q2k_next)
    assert int(q2k_indices.data_ptr()) == q2k_data_ptr
    assert int(q2k_source.data_ptr()) == q2k_source_data_ptr
    assert int(page_table.data_ptr()) == page_table_data_ptr
    assert int(cache_seqlens.data_ptr()) == cache_seqlens_data_ptr
    assert int(cu_seqlens_q.data_ptr()) == cu_seqlens_q_data_ptr

    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    lse = binding.scratch.current_lse_view() * math.log(2.0)
    torch.testing.assert_close(lse, ref_lse, rtol=lse_tol, atol=lse_tol)
    assert (
        torch.nn.functional.cosine_similarity(
            output.to(torch.float32).reshape(-1),
            ref_out.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        >= cos_min
    )


@torch.inference_mode()
@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_msa_decode_cuda_graph_replays_minimax_padded_bucket_after_prefill(
    kv_dtype: str,
) -> None:
    require_b12x()
    clear_attention_caches()

    bucket = 16
    page_size = 128
    page_table_width = 1024
    num_pages = 2048
    lse_tol = 5e-2
    cos_min = 0.995

    def make_case(cache_len: int, *, seed: int):
        q_live, k_c, v_c, table_live, seqlens_live, _cu_live = make_paged_inputs(
            q_seqlens=[1],
            cache_seqlens=[cache_len],
            page_size=page_size,
            q_heads=16,
            kv_heads=1,
            head_dim=128,
            dtype=torch.bfloat16,
            seed=seed,
            page_table_width=page_table_width,
            num_pages=num_pages,
            vllm_combined_kv=kv_dtype == "bf16",
        )
        device = q_live.device
        q_c = torch.zeros(
            (bucket, q_live.shape[1], q_live.shape[2]),
            dtype=q_live.dtype,
            device=device,
        )
        q_c[0].copy_(q_live[0])
        table_c = torch.zeros(
            (bucket, page_table_width), dtype=torch.int32, device=device
        )
        table_c[0].copy_(table_live[0])
        seqlens_c = torch.zeros((bucket,), dtype=torch.int32, device=device)
        seqlens_c[0].copy_(seqlens_live[0])
        cu_c = torch.ones((bucket + 1,), dtype=torch.int32, device=device)
        cu_c[0] = 0

        if kv_dtype == "bf16":
            return q_c, k_c, v_c, table_c, seqlens_c, cu_c, None, None

        k_q, v_q, k_ds, v_ds = quantize_paged_kv_cache_e4m3(
            k_c, v_c, table_live, seqlens_live
        )
        combined = torch.stack([k_q, v_q], dim=1)
        k_q = combined[:, 0]
        v_q = combined[:, 1]
        assert not k_q.is_contiguous()
        k_ds_c = torch.ones((bucket, k_q.shape[2]), dtype=torch.float32, device=device)
        v_ds_c = torch.ones_like(k_ds_c)
        k_ds_c[0].copy_(k_ds[0])
        v_ds_c[0].copy_(v_ds[0])
        return q_c, k_q, v_q, table_c, seqlens_c, cu_c, k_ds_c, v_ds_c

    (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale,
        v_descale,
    ) = make_case(1, seed=2101)
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=1,
        total_q_capacity=bucket,
        seed=2102,
        force_block0=True,
    )
    q2k_source = q2k_indices.clone()
    live_cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=q.device)

    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode="decode",
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=page_size,
            max_total_q=bucket,
            max_batch=bucket,
            max_page_table_width=page_table_width,
            max_work_items=bucket * 32,
            max_partial_rows=bucket * 32,
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=True,
            msa_block_sparse=True,
        )
    )
    scratch_plan.prepare_decode_graph_replay_state(
        batch=bucket,
        max_page_table_width=page_table_width,
        max_cache_page_count=page_table_width,
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    output = torch.empty_like(q)

    def bind_current():
        return scratch_plan.bind(
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            q2k_indices=q2k_indices,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    def assert_live_row_matches_reference(binding) -> None:
        ref_out, ref_lse = msa_attention_reference(
            q[:1],
            k_cache,
            v_cache,
            page_table[:1],
            cache_seqlens[:1],
            live_cu_seqlens_q,
            q2k_indices,
            k_descale=None if k_descale is None else k_descale[:1],
            v_descale=None if v_descale is None else v_descale[:1],
        )
        lse = binding.scratch.current_lse_view()[:1] * math.log(2.0)
        torch.testing.assert_close(lse, ref_lse, rtol=lse_tol, atol=lse_tol)
        assert (
            torch.nn.functional.cosine_similarity(
                output[:1].to(torch.float32).reshape(-1),
                ref_out.to(torch.float32).reshape(-1),
                dim=0,
            ).item()
            >= cos_min
        )
        assert int(binding.scratch.total_num_rows_ptr.item()) == 1

    paged_attention_forward(binding=bind_current())
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q2k_indices.copy_(q2k_source)
        binding = bind_current()
        paged_attention_forward(binding=binding)

    graph.replay()
    torch.cuda.synchronize()
    assert_live_row_matches_reference(binding)

    (
        q_next,
        k_next,
        v_next,
        page_table_next,
        cache_seqlens_next,
        cu_seqlens_q_next,
        k_descale_next,
        v_descale_next,
    ) = make_case(19, seed=2103)
    q2k_next = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens_next,
        cu_seqlens_q=cu_seqlens_q_next,
        num_kv_heads=1,
        total_q_capacity=bucket,
        seed=2104,
        force_block0=True,
    )
    q.copy_(q_next)
    k_cache.copy_(k_next)
    v_cache.copy_(v_next)
    page_table.copy_(page_table_next)
    cache_seqlens.copy_(cache_seqlens_next)
    cu_seqlens_q.copy_(cu_seqlens_q_next)
    if k_descale is not None:
        assert k_descale_next is not None
        k_descale.copy_(k_descale_next)
    if v_descale is not None:
        assert v_descale_next is not None
        v_descale.copy_(v_descale_next)
    q2k_source.copy_(q2k_next)

    graph.replay()
    torch.cuda.synchronize()
    assert_live_row_matches_reference(binding)


def test_msa_decode_graph_metadata_skips_zero_length_padded_rows() -> None:
    require_b12x()

    from b12x.attention.paged.graph_replay import (
        update_msa_decode_graph_chunk_metadata,
    )

    batch = 4
    max_chunks_per_req = 32
    capacity = batch * max_chunks_per_req
    cache_seqlens = torch.tensor([19, 0, 0, 0], dtype=torch.int32, device="cuda")
    request_indices = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    qo_tile_indices = torch.full_like(request_indices, -1)
    kv_tile_indices = torch.full_like(request_indices, -1)
    block_valid_mask = torch.full_like(request_indices, 7)
    merge_indptr = torch.full((batch + 1,), -1, dtype=torch.int32, device="cuda")
    o_indptr = torch.full_like(merge_indptr, -1)
    kv_chunk_size_ptr = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    kv_window_start_tokens = torch.full((batch,), -1, dtype=torch.int32, device="cuda")

    update_msa_decode_graph_chunk_metadata(
        cache_seqlens=cache_seqlens,
        request_indices=request_indices,
        qo_tile_indices=qo_tile_indices,
        kv_tile_indices=kv_tile_indices,
        merge_indptr=merge_indptr,
        o_indptr=o_indptr,
        block_valid_mask=block_valid_mask,
        kv_chunk_size_ptr=kv_chunk_size_ptr,
        kv_window_start_tokens=kv_window_start_tokens,
        kv_chunk_size=128,
        page_size=128,
    )
    torch.cuda.synchronize()

    assert merge_indptr.cpu().tolist() == [0, 1, 1, 1, 1]
    assert o_indptr.cpu().tolist() == [0, 1, 1, 1, 1]
    assert int(kv_chunk_size_ptr.item()) == 128
    assert kv_window_start_tokens.cpu().tolist() == [0, 0, 0, 0]
    assert int(block_valid_mask[0].item()) == 1
    assert int(block_valid_mask[1:].sum().item()) == 0
    assert int(request_indices[0].item()) == 0
    assert int(qo_tile_indices[0].item()) == 0
    assert int(kv_tile_indices[0].item()) == 0


def test_msa_decode_graph_compile_key_is_stable_within_minimax_batch_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    clear_attention_caches()

    import b12x.attention.paged._forward as paged_api

    forward_specs: list[str] = []

    def fake_launch(
        _func,
        *,
        compile_spec,
        compile_args,
        runtime_args,
        compile_kwargs=None,
    ):
        if compile_spec.kernel_id == "attention.paged.forward":
            forward_specs.append(repr(compile_spec))

    monkeypatch.setattr(paged_api, "b12x_launch", fake_launch)

    spec_by_batch: dict[int, str] = {}
    for batch in (1, 2, 4, 8, 16):
        page_table_width = 80
        scratch_plan = plan_paged_attention_scratch(
            B12XPagedAttentionScratchCaps(
                device=torch.device("cuda"),
                mode="decode",
                dtype=torch.bfloat16,
                kv_dtype=torch.bfloat16,
                num_q_heads=16,
                num_kv_heads=1,
                head_dim_qk=128,
                head_dim_vo=128,
                page_size=128,
                max_total_q=batch,
                max_batch=batch,
                max_page_table_width=page_table_width,
                max_work_items=batch * 32,
                max_partial_rows=batch * 32,
                num_cache_pages=2048,
                use_cuda_graph=True,
                msa_block_sparse=True,
            )
        )
        scratch_plan.prepare_decode_graph_replay_state(
            batch=batch,
            max_page_table_width=page_table_width,
            max_cache_page_count=page_table_width,
        )
        q2k_graph = torch.empty(
            (1, batch, 16), dtype=torch.int32, device=torch.device("cuda")
        )
        batch_specs: list[str] = []
        for cache_len, seed in ((129, 1700 + batch), (5000, 1800 + batch)):
            q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
                make_paged_inputs(
                    q_seqlens=[1] * batch,
                    cache_seqlens=[cache_len] * batch,
                    page_size=128,
                    q_heads=16,
                    kv_heads=1,
                    head_dim=128,
                    dtype=torch.bfloat16,
                    seed=seed,
                    page_table_width=page_table_width,
                    num_pages=2048,
                    vllm_combined_kv=True,
                )
            )
            q2k_indices = make_msa_q2k_indices(
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                num_kv_heads=1,
                total_q_capacity=batch,
                seed=seed + 1000,
                force_block0=True,
            )
            q2k_graph.copy_(q2k_indices)
            scratch = tuple(
                torch.empty(shape, dtype=dtype, device=q.device)
                for shape, dtype in scratch_plan.shapes_and_dtypes()
            )
            output = torch.empty_like(q)
            before = len(forward_specs)
            binding = scratch_plan.bind(
                scratch=scratch,
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                output=output,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                q2k_indices=q2k_graph,
            )
            paged_attention_forward(binding=binding)
            assert len(forward_specs) == before + 1
            batch_specs.append(forward_specs[-1])

        assert len(set(batch_specs)) == 1
        spec_by_batch[batch] = batch_specs[0]

    assert len(set(spec_by_batch.values())) == len(spec_by_batch)


def test_msa_extend_compile_key_does_not_require_decode_graph_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    require_b12x()
    clear_attention_caches()

    import b12x.attention.paged._forward as paged_api

    forward_specs: list[str] = []

    def fake_launch(
        _func,
        *,
        compile_spec,
        compile_args,
        runtime_args,
        compile_kwargs=None,
    ):
        if compile_spec.kernel_id == "attention.paged.forward":
            forward_specs.append(repr(compile_spec))

    monkeypatch.setattr(paged_api, "b12x_launch", fake_launch)

    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[8, 4],
        cache_seqlens=[128, 256],
        page_size=128,
        q_heads=16,
        kv_heads=1,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1901,
        page_table_width=8,
        num_pages=32,
        vllm_combined_kv=True,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=1,
        seed=1902,
        force_block0=True,
    )

    _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
        msa_union_tile=False,
    )

    assert len(forward_specs) == 1


def test_msa_decode_eager_bf16_page128_vllm_combined_cache_matches_reference() -> None:
    """vLLM MiniMax-M3 cache shape: combined [N, 2, 128, H, D] with K/V strided slices."""
    require_b12x()
    cache_lens = [1, 64, 127, 128, 129, 200, 2047, 2048, 5000]
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1] * len(cache_lens),
        cache_seqlens=cache_lens,
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1501,
        vllm_combined_kv=True,
    )
    assert k_cache.stride(0) == 2 * 128 * 4 * 128
    assert v_cache.stride(0) == 2 * 128 * 4 * 128
    assert not k_cache.is_contiguous()
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=53,
        force_block0=True,
        poison_padding=True,
    )
    ref, ref_lse = msa_attention_reference(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )

    # Eager default split plus an explicit chunk sweep (split invariance).
    outputs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    for fixed_split_size in (None, 1, 2, 4, 16):
        out, lse = _run_msa_decode(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q2k_indices,
            fixed_split_size=fixed_split_size,
        )
        torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
        cosine = torch.nn.functional.cosine_similarity(
            out.to(torch.float32).reshape(-1),
            ref.to(torch.float32).reshape(-1),
            dim=0,
        ).item()
        assert cosine >= 0.999
        outputs.append(out.detach().clone())
        lses.append(lse.detach().clone())

    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0], rtol=3e-3, atol=3e-3)
    for lse in lses[1:]:
        torch.testing.assert_close(lse, lses[0], rtol=2e-3, atol=2e-3)


def test_msa_extend_eager_bf16_page128_vllm_combined_cache_matches_reference() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 5, 300],
        cache_seqlens=[129, 384, 640],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1502,
        vllm_combined_kv=True,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=59,
        force_block0=True,
        poison_padding=True,
    )

    out, lse = _run_msa_extend(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    ref, ref_lse = msa_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        q2k_indices,
    )
    torch.testing.assert_close(lse, ref_lse, rtol=2e-3, atol=2e-3)
    cosine = torch.nn.functional.cosine_similarity(
        out.to(torch.float32).reshape(-1),
        ref.to(torch.float32).reshape(-1),
        dim=0,
    ).item()
    assert cosine >= 0.999


def test_msa_extend_qo_len_one_page128_vllm_combined_cache_matches_decode() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1, 1],
        cache_seqlens=[129, 2048],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1503,
        vllm_combined_kv=True,
    )
    q2k_indices = make_msa_q2k_indices(
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        num_kv_heads=4,
        seed=61,
        force_block0=True,
    )

    extend_out, extend_lse = _run_msa_extend(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    decode_out, decode_lse = _run_msa_decode(
        q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, q2k_indices
    )
    torch.testing.assert_close(extend_lse, decode_lse, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(extend_out, decode_out, rtol=3e-3, atol=3e-3)


def test_paged_plan_accepts_dense_page128() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[200],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1504,
    )
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="decode",
    )
    assert plan.page_size == 128
    assert plan.msa_block_sparse is False


def test_paged_plan_accepts_page128_fp8_kv() -> None:
    require_b12x()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = make_paged_inputs(
        q_seqlens=[1],
        cache_seqlens=[200],
        page_size=128,
        q_heads=64,
        kv_heads=4,
        head_dim=128,
        dtype=torch.bfloat16,
        seed=1505,
    )
    k_fp8 = k_cache.to(torch.float8_e4m3fn)
    v_fp8 = v_cache.to(torch.float8_e4m3fn)
    plan = create_paged_plan(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="decode",
        msa_block_sparse=True,
    )
    assert plan.page_size == 128
    assert plan.split_kv is True
