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

from flashinfer.attention.prims_ts import (
    _block_sparse,
    get_prims_ts_batch_decode_workspace_size,
    prepare_prims_ts_batch_decode_with_kv_cache,
)
from flashinfer.attention.prims_ts import decode as decode_module
from flashinfer.attention.prims_ts.block_sparse import BlockSparseTSWrapper
from tests.attention.test_attention_ts_block_sparse import (
    _HEAD_DIM,
    _REQUIRES_PRIMTS_GPU,
)
from tests.attention.test_attention_ts_decode import (
    _FP8,
    _assert_case_correct,
    _exercise_public_paths,
    _make_decode_case,
    _pack_decode_case,
    _plan_case,
    _run_case,
    _seq_lens_from_csr,
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

    @property
    def out_dtype(self) -> torch.dtype:
        return self.dtype if self.output_dtype is None else self.output_dtype

    @property
    def kv_lens(self) -> tuple[int, ...]:
        # Ragged lengths; the longest makes the first CTA of a 64-token causal
        # request stream one K/V tile its first Q tile cannot see.
        return tuple(1072 - 97 * (i % 7) for i in range(self.batch_size))


_PAGED_CASES = (
    _PagedPairedCase("bf16_group8_three_tiles", heads_q_per_kv=8, seq_len_q=48),
    _PagedPairedCase(
        "fp16_group8_causal_page16",
        heads_q_per_kv=8,
        seq_len_q=64,
        page_size=16,
        mask_type="causal",
        dtype=torch.float16,
    ),
    # One token per Q tile: five tiles leave the last CTA's second tile past
    # the sequence.
    _PagedPairedCase(
        "bf16_group128_causal_page64",
        heads_q_per_kv=128,
        seq_len_q=5,
        num_kv_heads=4,
        page_size=64,
        mask_type="causal",
    ),
    # 96 heads leave 32 padding rows in each one-token Q tile.
    _PagedPairedCase(
        "bf16_group96_page128",
        heads_q_per_kv=96,
        seq_len_q=4,
        num_kv_heads=4,
        page_size=128,
    ),
    _PagedPairedCase(
        "bf16_group8_packed_causal",
        heads_q_per_kv=8,
        seq_len_q=64,
        mask_type="causal",
        q_lens=(64, 5, 33, 48, 17),
    ),
    # Paged FP8 pairs at page size 32. Paired CTAs beyond one wave run on the
    # persistent scheduler; eight requests keep them within one static wave.
    _PagedPairedCase(
        "fp8_group8_persistent", heads_q_per_kv=8, seq_len_q=48, dtype=_FP8
    ),
    _PagedPairedCase(
        "fp8_group8_static_bf16_output",
        heads_q_per_kv=8,
        seq_len_q=48,
        batch_size=8,
        dtype=_FP8,
        output_dtype=torch.bfloat16,
    ),
    _PagedPairedCase(
        "fp8_group32_causal_fp16_output",
        heads_q_per_kv=32,
        seq_len_q=12,
        mask_type="causal",
        dtype=_FP8,
        output_dtype=torch.float16,
    ),
    _PagedPairedCase(
        "fp8_group128_causal",
        heads_q_per_kv=128,
        seq_len_q=5,
        num_kv_heads=4,
        mask_type="causal",
        dtype=_FP8,
    ),
    _PagedPairedCase(
        "fp8_head_dim64_causal",
        heads_q_per_kv=8,
        seq_len_q=48,
        mask_type="causal",
        dtype=_FP8,
        head_dim=64,
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
        output_dtype=case.out_dtype,
        cache_form="tuple",
        mask_type=case.mask_type,
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


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_prepared_paged_query_paired_matches_wrapper() -> None:
    """A prepared paired launch reproduces the paired wrapper, eagerly and in a graph."""

    case = _PagedPairedCase(
        "bf16_group8_causal", heads_q_per_kv=8, seq_len_q=48, mask_type="causal"
    )
    decode_case, _ = _paged_decode_case(case)
    seq_lens = _seq_lens_from_csr(
        decode_case.paged_kv_indptr,
        decode_case.paged_kv_last_page_len,
        case.page_size,
    )
    max_kv_len = max(case.kv_lens)
    wrapper = _plan_case(decode_case, max_kv_len=max_kv_len, query_paired=True)
    assert dict(wrapper._policy)["query_paired"] is True
    expected = _run_case(wrapper, decode_case)

    workspace = torch.zeros(
        get_prims_ts_batch_decode_workspace_size(
            case.batch_size,
            case.heads_q_per_kv * case.num_kv_heads,
            case.num_kv_heads,
            case.head_dim,
            case.page_size,
            max_kv_len,
            seq_len_q=case.seq_len_q,
            q_dtype=case.dtype,
            out_dtype=case.out_dtype,
            mask_type=case.mask_type,
            device="cuda",
            query_paired=True,
        ),
        dtype=torch.int8,
        device="cuda",
    )
    out = torch.empty_like(expected)
    prepared = prepare_prims_ts_batch_decode_with_kv_cache(
        decode_case.q,
        decode_case.paged_kv_cache,
        workspace,
        decode_case.block_tables,
        seq_lens,
        max_kv_len,
        out=out,
        seq_len_q=case.seq_len_q,
        out_dtype=case.out_dtype,
        mask_type=case.mask_type,
        page_size=case.page_size,
        query_paired=True,
    )
    run_args = dict(
        bmm1_scale=decode_case.bmm1_scale, bmm2_scale=decode_case.bmm2_scale
    )
    assert prepared.run(decode_case.q, out=out, **run_args) is out
    torch.cuda.synchronize()
    _assert_case_correct(out, decode_case)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    graph_out = torch.full_like(expected, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.run(decode_case.q, out=graph_out, **run_args)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, expected, rtol=0, atol=0)


_PAIRED_DECODE_SHAPE = dict(
    batch_size=16,
    num_qo_heads=64,
    num_kv_heads=8,
    head_dim=_HEAD_DIM,
    page_size=32,
    max_kv_len=4096,
    seq_len_q=48,
    q_dtype_key="bfloat16",
    k_dtype_key="bfloat16",
    v_dtype_key="bfloat16",
    output_dtype_key="bfloat16",
    kv_layout="HND",
    mask_type="causal",
    use_packed_q=False,
    window_left=-1,
)


def _paged_policy(query_paired: bool | None, **changes: object) -> dict[str, object]:
    shape = {**_PAIRED_DECODE_SHAPE, **changes}
    spec = decode_module._resolve_decode_launch_spec(
        torch.cuda.current_device(), **shape, query_paired=query_paired
    )
    return dict(spec.policy)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    "changes",
    (
        pytest.param({"seq_len_q": 16}, id="one-q-tile"),
        pytest.param({"batch_size": 1, "num_kv_heads": 1}, id="small-grid"),
        pytest.param({"window_left": 511}, id="sliding-window"),
        pytest.param({"head_dim": 256}, id="head-dim-256"),
        # Head dim 64 pairs only the paged FP8 recipe.
        pytest.param({"head_dim": 64}, id="bf16-head-dim-64"),
    ),
)
def test_paged_query_paired_plans_unpaired_when_unsupported(
    changes: dict[str, object],
) -> None:
    """Pairing falls back to an unpaired plan and records that in the policy."""

    assert _paged_policy(True)["query_paired"] is True
    assert _paged_policy(True, **changes)["query_paired"] is False


def test_paged_query_paired_must_be_a_bool() -> None:
    with pytest.raises(TypeError, match="query_paired"):
        decode_module._resolve_decode_launch_spec(
            0, **_PAIRED_DECODE_SHAPE, query_paired="yes"
        )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("changes", "model_pairs"),
    (
        # 8192 unpaired CTAs take twice the waves of the 4096 paired ones.
        pytest.param({"batch_size": 64, "seq_len_q": 256}, True, id="pairs"),
        # 1024 unpaired CTAs take seven waves; 512 paired ones still take four.
        pytest.param({"seq_len_q": 128}, False, id="unpaired"),
    ),
)
def test_paged_query_paired_none_follows_the_cost_model(
    changes: dict[str, object], model_pairs: bool
) -> None:
    """None pairs when the grouped-Q cost model scores the paired tile cheapest."""

    shape = {"max_kv_len": 8192, **changes}
    assert _paged_policy(None, **shape)["query_paired"] is model_pairs
    assert _paged_policy(True, **shape)["query_paired"] is True
    assert _paged_policy(False, **shape)["query_paired"] is False


def test_grouped_q_candidates_pair_keeps_q128_tiles() -> None:
    """The paired Keeps Q128 candidate covers two Q tiles per CTA."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        enumerate_grouped_q_mma_candidates,
    )

    candidates = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=8, seq_len_q=48, query_paired=True
    )
    paired = [candidate for candidate in candidates if candidate.query_paired]
    assert [(c.variant, c.tile_size_q, c.q_tiles) for c in paired] == [
        ("keeps_mma_ab", 128, 3)
    ]
    assert paired[0].ctas_per_kv_head == 2
    # One Q tile leaves nothing to pair.
    single = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=8, seq_len_q=16, query_paired=True
    )
    assert not any(candidate.query_paired for candidate in single)
