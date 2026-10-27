"""Query-paired Q128 plans of the PrimTS decode kernel.

A query-paired plan runs two consecutive Q128 tiles of one sequence and KV
head on one CTA, each K/V instance owning a tile over the whole K/V stream.
Dense block-sparse wrapper plans pair contiguous K/V; paged decode plans pair
when ``query_paired=True`` and the launch qualifies, or when the automatic
selection pairs a ``None`` plan. Sage recipes are covered in
``test_attention_ts_sage.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flashinfer.attention.prims_ts._block_sparse import config as sparse_config
from flashinfer.attention.prims_ts.block_sparse import BlockSparseTSWrapper
from tests.attention.test_attention_ts_block_sparse import (
    _HEAD_DIM,
    _REQUIRES_PRIMTS_GPU,
)
from tests.attention.test_attention_ts_decode import (
    _FP8,
    _exercise_public_paths,
    _make_decode_case,
    _pack_decode_case,
    _plan_case,
    _seq_lens_from_csr,
)


@dataclass(frozen=True)
class _PairedCase:
    """One dense Q128 problem: Q tiles of 16 tokens from eight heads per KV head."""

    name: str
    seq_len_q: int
    seq_len_kv: int
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = 2
    num_kv_heads: int = 2
    persistent: bool = False
    mask_type: str = "dense"

    @property
    def num_qo_heads(self) -> int:
        return 8 * self.num_kv_heads


_CASES = (
    # One Q tile leaves the second instance's tile past the end of Q.
    _PairedCase("one_tile_one_kv_tile", seq_len_q=16, seq_len_kv=128),
    _PairedCase("two_tiles_ragged_kv", seq_len_q=32, seq_len_kv=1000),
    _PairedCase(
        "causal_three_tiles_fp16",
        seq_len_q=48,
        seq_len_kv=640,
        dtype=torch.float16,
        mask_type="causal",
    ),
    # Only the first of the two CTAs streams a K/V tile its first Q tile
    # cannot see.
    _PairedCase(
        "causal_two_ctas_persistent",
        seq_len_q=64,
        seq_len_kv=1072,
        mask_type="causal",
        persistent=True,
    ),
)


def _plan(case: _PairedCase, *, query_paired: bool) -> BlockSparseTSWrapper:
    wrapper = BlockSparseTSWrapper()
    select_scheduler = sparse_config._select_block_sparse_scheduler
    sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            if case.persistent:
                # The test shapes are too small for the planner to pick the
                # persistent scheduler; keep its Q tile and force the choice.
                monkeypatch.setattr(
                    sparse_config,
                    "_select_block_sparse_scheduler",
                    lambda **kwargs: (select_scheduler(**kwargs)[0], True),
                )
            wrapper.plan(
                case.batch_size,
                case.seq_len_q,
                case.seq_len_kv,
                case.num_qo_heads,
                case.num_kv_heads,
                _HEAD_DIM,
                16,
                64,
                device=torch.device("cuda", 0),
                use_block_sparse=False,
                mask_type=case.mask_type,
                q_data_type=case.dtype,
                query_paired=query_paired,
            )
    finally:
        sparse_config._resolve_block_sparse_launch_spec.cache_clear()
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == 128
    assert policy["tile_size_kv"] == 128
    assert policy["scheduler"] == ("persistent" if case.persistent else "static")
    assert policy["query_paired"] is query_paired
    return wrapper


def _reference(q, k, v, sm_scale: float, mask_type: str) -> torch.Tensor:
    group = q.shape[2] // k.shape[2]
    k_heads = k.float().repeat_interleave(group, dim=2)
    v_heads = v.float().repeat_interleave(group, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k_heads) * sm_scale
    if mask_type == "causal":
        # Query i sees the keys up to its own position at the end of K/V.
        seq_len_q, seq_len_kv = q.shape[1], k.shape[1]
        last_keys = torch.arange(seq_len_q, device=q.device) + seq_len_kv - seq_len_q
        hidden = torch.arange(seq_len_kv, device=q.device) > last_keys[:, None]
        scores = scores.masked_fill(hidden, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v_heads)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("query_paired", (False, True), ids=("unpaired", "paired"))
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_dense_plan_matches_fp32(case: _PairedCase, query_paired: bool) -> None:
    """Paired and unpaired dense Q128 plans match FP32 attention."""

    torch.manual_seed(20261018)
    shape_q = (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM)
    shape_kv = (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM)
    q, k, v = (
        torch.randn(shape, device="cuda").to(case.dtype)
        for shape in (shape_q, shape_kv, shape_kv)
    )
    sm_scale = _HEAD_DIM**-0.5
    out = _plan(case, query_paired=query_paired).run(q, k, v, sm_scale=sm_scale)
    assert out.dtype == case.dtype
    expected = _reference(q, k, v, sm_scale, case.mask_type)
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@dataclass(frozen=True)
class _PagedPairedCase:
    """One paged decode problem whose Q tiles fill a wave of CTAs when paired."""

    name: str
    heads_q_per_kv: int
    seq_len_q: int
    num_kv_heads: int = 8
    batch_size: int = 16
    page_size: int = 32
    mask_type: str = "dense"
    dtype: torch.dtype = torch.bfloat16
    output_dtype: torch.dtype | None = None
    head_dim: int = _HEAD_DIM
    # Per-request Q lengths of a packed batch, cycled over the batch.
    q_lens: tuple[int, ...] | None = None
    window_left: int = -1

    @property
    def kv_lens(self) -> tuple[int, ...]:
        # Ragged lengths; the longest makes the first CTA of a 64-token causal
        # request stream one K/V tile its first Q tile cannot see.
        return tuple(1072 - 97 * (i % 7) for i in range(self.batch_size))


_PAGED_CASES = (
    _PagedPairedCase("bf16_group8_three_tiles", heads_q_per_kv=8, seq_len_q=48),
    # A CTA's K/V stream starts at its first Q tile's window, 16 tokens
    # before its second tile's.
    _PagedPairedCase(
        "bf16_group8_causal_window",
        heads_q_per_kv=8,
        seq_len_q=64,
        mask_type="causal",
        window_left=127,
    ),
    # One token per Q tile: five tiles leave the last CTA's second tile past
    # the sequence, and each tile masks its own window, one key after the
    # previous tile's.
    _PagedPairedCase(
        "bf16_group128_causal_window_page64",
        heads_q_per_kv=128,
        seq_len_q=5,
        num_kv_heads=4,
        page_size=64,
        mask_type="causal",
        window_left=31,
    ),
    _PagedPairedCase(
        "bf16_group8_packed_causal_window",
        heads_q_per_kv=8,
        seq_len_q=64,
        mask_type="causal",
        q_lens=(64, 5, 33, 48, 17),
        window_left=300,
    ),
    # Paired FP8 CTAs beyond one wave run on the persistent scheduler.
    _PagedPairedCase(
        "fp8_group8_persistent", heads_q_per_kv=8, seq_len_q=48, dtype=_FP8
    ),
    _PagedPairedCase(
        "fp8_head_dim64_group128_bf16_output",
        heads_q_per_kv=128,
        seq_len_q=5,
        num_kv_heads=4,
        dtype=_FP8,
        output_dtype=torch.bfloat16,
        head_dim=64,
    ),
)


def _paged_decode_case(case: _PagedPairedCase):
    """Build ``case``'s inputs and reference; packed cases also return Q offsets."""

    decode_case = _make_decode_case(
        kv_lens=case.kv_lens,
        num_qo_heads=case.heads_q_per_kv * case.num_kv_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=case.head_dim,
        seq_len_q=case.seq_len_q,
        page_size=case.page_size,
        qkv_dtype=case.dtype,
        output_dtype=case.output_dtype or case.dtype,
        cache_form="tuple",
        mask_type=case.mask_type,
        window_left=case.window_left,
        device="cuda",
        seed=20261020,
    )
    if case.q_lens is None:
        return decode_case, None
    q_lens = [case.q_lens[i % len(case.q_lens)] for i in range(case.batch_size)]
    return _pack_decode_case(decode_case, q_lens)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _PAGED_CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_paged_query_paired_public_paths(case: _PagedPairedCase) -> None:
    """A paired paged plan matches FP32 attention eagerly, standalone and in a graph."""

    decode_case, qo_indptr = _paged_decode_case(case)
    seq_lens = _seq_lens_from_csr(
        decode_case.paged_kv_indptr,
        decode_case.paged_kv_last_page_len,
        case.page_size,
    )
    wrapper = _plan_case(
        decode_case,
        max_kv_len=max(case.kv_lens),
        qo_indptr=qo_indptr,
        max_seq_len_q=case.seq_len_q,
        query_paired=True,
    )
    policy = dict(wrapper._policy)
    assert policy["query_paired"] is True
    assert policy["tile_size_q"] == 128
    _exercise_public_paths(
        wrapper,
        decode_case,
        seq_lens,
        max_kv_len=max(case.kv_lens),
        exercise_all_paths=True,
        qo_indptr=qo_indptr,
        max_seq_len_q=case.seq_len_q,
    )
