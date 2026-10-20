"""Query-paired dense Q128 plans of the PrimTS decode kernel.

A query-paired plan runs two consecutive Q128 tiles of one sequence and KV
head on one CTA, each K/V instance owning a tile over the whole K/V stream.
Sage recipes are covered in ``test_attention_ts_sage.py``.
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
