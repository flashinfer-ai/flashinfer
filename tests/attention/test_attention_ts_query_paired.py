"""Query-paired dense Q128 plans of the PrimTS decode kernel.

A query-paired plan runs two consecutive Q128 tiles of one sequence and KV
head on one CTA, each K/V instance owning a tile over the whole K/V stream.
Sage recipes are covered in ``test_attention_ts_sage.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flashinfer.attention.prims_ts import _block_sparse
from flashinfer.attention.prims_ts.block_sparse import BlockSparseTSWrapper
from tests.attention.test_attention_ts_block_sparse import (
    _HEAD_DIM,
    _REQUIRES_PRIMTS_GPU,
)

sparse_config = _block_sparse.config


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
    _PairedCase("three_tiles_fp16", seq_len_q=48, seq_len_kv=640, dtype=torch.float16),
    _PairedCase(
        "three_tiles_persistent", seq_len_q=48, seq_len_kv=1000, persistent=True
    ),
    # The first Q tile sees K/V up to 1024, the second one more K/V tile.
    _PairedCase(
        "causal_one_cta_extra_kv_tile",
        seq_len_q=32,
        seq_len_kv=1040,
        mask_type="causal",
    ),
    # Only the first of the two CTAs streams a K/V tile its first Q tile
    # cannot see.
    _PairedCase(
        "causal_two_ctas_extra_kv_tile",
        seq_len_q=64,
        seq_len_kv=1072,
        mask_type="causal",
    ),
    _PairedCase(
        "causal_three_tiles_fp16",
        seq_len_q=48,
        seq_len_kv=640,
        dtype=torch.float16,
        mask_type="causal",
    ),
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
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_query_paired_matches_unpaired(case: _PairedCase) -> None:
    """A paired plan reproduces the unpaired kernel and FP32 attention."""

    torch.manual_seed(20261018)
    q = torch.randn(
        (case.batch_size, case.seq_len_q, case.num_qo_heads, _HEAD_DIM),
        device="cuda",
    ).to(case.dtype)
    k = torch.randn(
        (case.batch_size, case.seq_len_kv, case.num_kv_heads, _HEAD_DIM),
        device="cuda",
    ).to(case.dtype)
    v = torch.randn_like(k)
    sm_scale = _HEAD_DIM**-0.5
    outputs = {}
    for query_paired in (False, True):
        wrapper = _plan(case, query_paired=query_paired)
        outputs[query_paired] = wrapper.run(q, k, v, sm_scale=sm_scale)
        torch.cuda.synchronize()
    paired, unpaired = outputs[True], outputs[False]
    assert paired.dtype == case.dtype
    assert torch.isfinite(paired).all()
    torch.testing.assert_close(paired.float(), unpaired.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        paired.float(),
        _reference(q, k, v, sm_scale, case.mask_type),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        pytest.param(
            {"use_block_sparse": True, "max_blocks_per_row": 4},
            "dense, contiguous",
            id="sparse",
        ),
        pytest.param({"num_qo_heads": 2, "q_block_size": 32}, "Q128", id="q64-tile"),
    ),
)
def test_query_paired_plan_rejects_other_profiles(
    overrides: dict[str, object], match: str
) -> None:
    """Pairing is a dense Q128 plan option only."""

    arguments = dict(
        batch_size=1,
        seq_len_q=32,
        seq_len_kv=256,
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        q_block_size=16,
        kv_block_size=64,
        device=torch.device("cuda", 0),
        use_block_sparse=False,
        q_data_type=torch.bfloat16,
        query_paired=True,
    )
    with pytest.raises(NotImplementedError, match=match):
        BlockSparseTSWrapper().plan(**{**arguments, **overrides})
