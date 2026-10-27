"""Sage attention in paged PrimTS decode plans.

A paged Sage plan reads its scale tensors in the flat layout of a dense Sage
plan whose K/V and Q lengths are the plan's capacities. Each request of a
paged launch must reproduce a dense contiguous Sage plan over the same data,
which runs the same Keeps Q128 tiles; sliding windows, which the dense plan
lacks, are checked against FP32 attention of the dequantized inputs. Scale
slots past a request's length hold NaN, which the plan must never read.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    SageAttentionConfig,
    SageAttentionParams,
    batch_decode_with_paged_kv_cache,
    get_prims_ts_batch_decode_workspace_size,
    prepare_prims_ts_batch_decode_with_kv_cache,
)
from flashinfer.attention.prims_ts.block_sparse import BlockSparseTSWrapper
from flashinfer.attention.prims_ts.sage.config import sage_flat_scale_numel
from tests.attention.sage_test_utils import (
    FP8,
    INT8_STD,
    REQUIRES_INT8_QK_GPU,
    ieee_float32_references,  # noqa: F401
    random_qk,
)
from tests.attention.test_attention_ts_block_sparse import (
    _HEAD_DIM,
    _REQUIRES_PRIMTS_GPU,
)

_SM_SCALE = _HEAD_DIM**-0.5


@dataclass(frozen=True)
class _PagedSageCase:
    """One paged Sage launch: per-request K/V lengths, Q tokens and recipe."""

    name: str
    kv_lens: tuple[int, ...]
    seq_len_q: int
    qk_dtype: torch.dtype = FP8
    output_dtype: torch.dtype = torch.bfloat16
    mask_type: str = "causal"
    page_size: int = 32
    k_block_size: int = 16
    v_mean: bool = False
    window_left: int = -1
    # Per-request Q tokens of a packed launch; None runs fixed Q.
    q_lens: tuple[int, ...] | None = None
    query_paired: bool = False
    num_kv_heads: int = 2
    heads_q_per_kv: int = 8

    @property
    def batch_size(self) -> int:
        return len(self.kv_lens)

    @property
    def max_kv_len(self) -> int:
        return 1024

    @property
    def num_qo_heads(self) -> int:
        return self.num_kv_heads * self.heads_q_per_kv

    @property
    def request_q_lens(self) -> tuple[int, ...]:
        return self.q_lens or (self.seq_len_q,) * self.batch_size

    @property
    def config(self) -> SageAttentionConfig:
        return SageAttentionConfig(k_block_size=self.k_block_size, v_mean=self.v_mean)


_CASES = (
    _PagedSageCase(
        "fp8_sq1_page64_k64_v_mean",
        kv_lens=(1000, 33, 640),
        seq_len_q=1,
        mask_type="dense",
        page_size=64,
        k_block_size=64,
        v_mean=True,
        output_dtype=torch.float16,
    ),
    # Eight K/V heads and eight Q tiles a request fill a wave of paired CTAs.
    _PagedSageCase(
        "fp8_causal_paired_k4",
        kv_lens=(1000, 515, 300),
        seq_len_q=128,
        k_block_size=4,
        query_paired=True,
        num_kv_heads=8,
    ),
    _PagedSageCase(
        "fp8_packed_causal_v_mean",
        kv_lens=(1000, 129, 515),
        seq_len_q=64,
        q_lens=(64, 5, 33),
        v_mean=True,
    ),
    pytest.param(
        _PagedSageCase(
            "int8_causal_page64",
            kv_lens=(1000, 129, 515),
            seq_len_q=33,
            qk_dtype=torch.int8,
            page_size=64,
        ),
        marks=REQUIRES_INT8_QK_GPU,
    ),
)


@dataclass
class _PagedSageInputs:
    """A paged launch's tensors and each request's logical data."""

    q: torch.Tensor
    kv_cache: tuple[torch.Tensor, torch.Tensor]
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    qo_indptr: torch.Tensor | None
    params: SageAttentionParams
    # Per request: Q [Sq_b, Hq, D], K and V [S_b, Hkv, D] in their storage
    # dtypes, Q scales [Hq, Sq_b], K scales [Hkv, ceil(S_b / block)].
    requests: list[dict[str, torch.Tensor]]


def _paged_sage_inputs(case: _PagedSageCase, seed: int = 20261027) -> _PagedSageInputs:
    """Draw 8-bit data and positive scales; padding scale slots hold NaN."""

    torch.manual_seed(seed)
    batch_size, num_kv_heads = case.batch_size, case.num_kv_heads
    num_qo_heads, page_size = case.num_qo_heads, case.page_size
    k_block = case.k_block_size
    seq_len_q, max_kv_len = case.seq_len_q, case.max_kv_len
    pages = [-(-length // page_size) for length in case.kv_lens]
    page_ids = torch.randperm(sum(pages) + 3)
    k_cache = torch.zeros(
        (sum(pages) + 3, num_kv_heads, page_size, _HEAD_DIM),
        dtype=case.qk_dtype,
        device="cuda",
    )
    v_cache = torch.zeros(k_cache.shape, dtype=FP8, device="cuda")
    # Unused entries hold -1, so a read past a request's pages faults.
    block_table = torch.full(
        (batch_size, -(-max_kv_len // page_size)), -1, dtype=torch.int32
    )
    q_scale = torch.full(
        (num_qo_heads, sage_flat_scale_numel(batch_size, seq_len_q, 1)),
        float("nan"),
        device="cuda",
    )
    k_scale = torch.full(
        (num_kv_heads, sage_flat_scale_numel(batch_size, max_kv_len, k_block)),
        float("nan"),
        device="cuda",
    )
    v_scale = torch.empty(
        (batch_size, num_kv_heads, _HEAD_DIM), device="cuda"
    ).uniform_(0.5, 2.0)
    v_mean = torch.empty_like(v_scale).uniform_(-0.5, 0.5) if case.v_mean else None
    # INT8 data has a wider range; its scales shrink to match E4M3's.
    unit = 1.0 / INT8_STD if case.qk_dtype == torch.int8 else 1.0
    requests = []
    first_page = 0
    for b, (kv_len, q_len) in enumerate(
        zip(case.kv_lens, case.request_q_lens, strict=True)
    ):
        k = random_qk((kv_len, num_kv_heads, _HEAD_DIM), case.qk_dtype)
        v = torch.randn((kv_len, num_kv_heads, _HEAD_DIM), device="cuda").to(FP8)
        q = random_qk((q_len, num_qo_heads, _HEAD_DIM), case.qk_dtype)
        q_scales = torch.empty((num_qo_heads, q_len), device="cuda").uniform_(
            0.1 * unit, 0.4 * unit
        )
        k_scales = torch.empty(
            (num_kv_heads, -(-kv_len // k_block)), device="cuda"
        ).uniform_(0.5 * unit, 2.0 * unit)
        q_first = b * seq_len_q + b
        q_scale[:, q_first : q_first + q_len] = q_scales
        k_first = b * max_kv_len // k_block + b
        k_scale[:, k_first : k_first + k_scales.shape[1]] = k_scales
        request_pages = page_ids[first_page : first_page + pages[b]]
        first_page += pages[b]
        block_table[b, : pages[b]] = request_pages.to(torch.int32)
        for page, page_id in enumerate(request_pages.tolist()):
            tokens = slice(page * page_size, min((page + 1) * page_size, kv_len))
            used = tokens.stop - tokens.start
            k_cache[page_id, :, :used] = k[tokens].transpose(0, 1)
            v_cache[page_id, :, :used] = v[tokens].transpose(0, 1)
        requests.append(
            {"q": q, "k": k, "v": v, "q_scale": q_scales, "k_scale": k_scales}
        )
    if case.q_lens is None:
        q = torch.stack([request["q"] for request in requests])
        q = q[:, 0] if seq_len_q == 1 else q
        qo_indptr = None
    else:
        q = torch.cat([request["q"] for request in requests])
        qo_indptr = torch.tensor(
            [0, *torch.tensor(case.q_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
    return _PagedSageInputs(
        q=q,
        kv_cache=(k_cache, v_cache),
        block_table=block_table.cuda(),
        seq_lens=torch.tensor(case.kv_lens, dtype=torch.int32, device="cuda"),
        qo_indptr=qo_indptr,
        params=SageAttentionParams(
            q_scale=q_scale, k_scale=k_scale, v_scale=v_scale, v_mean=v_mean
        ),
        requests=requests,
    )


def _plan(case: _PagedSageCase) -> BatchDecodePagedTSWrapper:
    wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
    wrapper.plan(
        torch.device("cuda", 0),
        case.batch_size,
        case.num_qo_heads,
        case.num_kv_heads,
        _HEAD_DIM,
        case.page_size,
        case.max_kv_len,
        max_seq_len_q=case.seq_len_q,
        packed_query=case.q_lens is not None,
        q_data_type=case.qk_dtype,
        v_data_type=FP8,
        o_data_type=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
        query_paired=case.query_paired,
        sage_config=case.config,
    )
    assert dict(wrapper._policy)["query_paired"] is case.query_paired
    return wrapper


def _run(wrapper, inputs: _PagedSageInputs, **kwargs) -> torch.Tensor:
    return wrapper.run(
        inputs.q,
        inputs.kv_cache,
        inputs.seq_lens,
        inputs.block_table,
        qo_indptr=inputs.qo_indptr,
        sage=inputs.params,
        **kwargs,
    )


def _request_outputs(case: _PagedSageCase, out: torch.Tensor) -> list[torch.Tensor]:
    """Split a launch's output into per-request ``[Sq_b, Hq, D]`` tensors."""

    if case.q_lens is not None:
        return list(out.split(list(case.q_lens)))
    if case.seq_len_q == 1:
        out = out[:, None]
    return list(out.unbind(0))


def _contiguous_request(
    case: _PagedSageCase, inputs: _PagedSageInputs, request_idx: int
) -> torch.Tensor:
    """Run one request through a dense contiguous Sage plan."""

    request = inputs.requests[request_idx]
    q_len, kv_len = request["q"].shape[0], request["k"].shape[0]
    wrapper = BlockSparseTSWrapper()
    wrapper.plan(
        1,
        q_len,
        kv_len,
        case.num_qo_heads,
        case.num_kv_heads,
        _HEAD_DIM,
        device=torch.device("cuda", 0),
        use_block_sparse=False,
        mask_type=case.mask_type,
        q_data_type=case.qk_dtype,
        v_data_type=FP8,
        o_data_type=case.output_dtype,
        sage_config=case.config,
        query_paired=case.query_paired,
    )
    params = inputs.params
    return wrapper.run(
        request["q"][None],
        request["k"][None],
        request["v"][None],
        sm_scale=_SM_SCALE,
        sage=SageAttentionParams(
            q_scale=request["q_scale"].contiguous(),
            k_scale=request["k_scale"].contiguous(),
            v_scale=params.v_scale[request_idx : request_idx + 1].contiguous(),
            v_mean=(
                None
                if params.v_mean is None
                else params.v_mean[request_idx : request_idx + 1].contiguous()
            ),
        ),
    )[0]


def _reference(case: _PagedSageCase, inputs: _PagedSageInputs) -> list[torch.Tensor]:
    """FP32 attention of each request's dequantized Q, K and V."""

    outputs = []
    params = inputs.params
    for b, request in enumerate(inputs.requests):
        q_len, kv_len = request["q"].shape[0], request["k"].shape[0]
        q = request["q"].float() * request["q_scale"].t()[..., None]
        k_scales = request["k_scale"].repeat_interleave(case.k_block_size, dim=1)
        k = request["k"].float() * k_scales[:, :kv_len].t()[..., None]
        v = request["v"].float() * params.v_scale[b]
        group = case.heads_q_per_kv
        scores = (
            torch.einsum("qhd,khd->hqk", q, k.repeat_interleave(group, dim=1))
            * _SM_SCALE
        )
        if case.mask_type == "causal":
            last_key = torch.arange(q_len, device="cuda")[:, None] + kv_len - q_len
            key = torch.arange(kv_len, device="cuda")[None]
            hidden = key > last_key
            if case.window_left >= 0:
                hidden |= key < last_key - case.window_left
            scores = scores.masked_fill(hidden, float("-inf"))
        out = torch.einsum(
            "hqk,khd->qhd", scores.softmax(dim=-1), v.repeat_interleave(group, dim=1)
        )
        if params.v_mean is not None:
            out = out + params.v_mean[b].repeat_interleave(group, dim=0)
        outputs.append(out)
    return outputs


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_paged_sage_matches_contiguous_sage(case: _PagedSageCase) -> None:
    """Every request of a paged launch reproduces a dense contiguous Sage plan."""

    inputs = _paged_sage_inputs(case)
    wrapper = _plan(case)
    policy = dict(wrapper._policy)
    assert policy["use_sage_attention"] is True
    assert policy["tile_size_q"] == 128 and not policy["use_split_kv"]
    out = _run(wrapper, inputs, bmm1_scale=_SM_SCALE)
    assert torch.isfinite(out.float()).all()
    for request_idx, actual in enumerate(_request_outputs(case, out)):
        expected = _contiguous_request(case, inputs, request_idx)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


_WINDOW_CASES = (
    _PagedSageCase(
        "fp8_window31", kv_lens=(1000, 129, 515), seq_len_q=64, window_left=31
    ),
    # Eight K/V heads and eight Q tiles a request fill a wave of paired CTAs.
    _PagedSageCase(
        "fp8_packed_window127_v_mean_paired",
        kv_lens=(1000, 777, 515),
        seq_len_q=128,
        q_lens=(128, 5, 77),
        window_left=127,
        v_mean=True,
        query_paired=True,
        num_kv_heads=8,
    ),
)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _WINDOW_CASES, ids=lambda case: case.name)
@torch.no_grad()
def test_paged_sage_window_matches_fp32(case: _PagedSageCase) -> None:
    """A sliding window masks each row's keys before its window.

    The tolerance covers the E4M3 probabilities of the PV product, whose
    error grows as a window shrinks: a 32-key window reaches 0.042 against
    FP32. A window one key wider or narrower is off by 0.7 or more.
    """

    inputs = _paged_sage_inputs(case)
    out = _run(_plan(case), inputs, bmm1_scale=_SM_SCALE)
    for actual, expected in zip(
        _request_outputs(case, out), _reference(case, inputs), strict=True
    ):
        torch.testing.assert_close(actual.float(), expected, rtol=5e-2, atol=5e-2)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@torch.no_grad()
def test_paged_sage_launch_paths_agree() -> None:
    """Eager, CUDA graph, one-shot and prepared launches return the same output."""

    case = _WINDOW_CASES[1]
    inputs = _paged_sage_inputs(case)
    wrapper = _plan(case)
    eager = _run(wrapper, inputs)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(wrapper, inputs, out=graph_out, validate=False)
    graph.replay()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)

    one_shot = batch_decode_with_paged_kv_cache(
        inputs.q,
        inputs.kv_cache,
        inputs.block_table,
        inputs.seq_lens,
        qo_indptr=inputs.qo_indptr,
        max_seq_len_q=case.seq_len_q,
        mask_type=case.mask_type,
        window_left=case.window_left,
        out_dtype=case.output_dtype,
        query_paired=case.query_paired,
        sage=inputs.params,
        sage_config=case.config,
        max_kv_len=case.max_kv_len,
    )
    torch.testing.assert_close(one_shot, eager, rtol=0, atol=0)

    workspace = torch.zeros(
        get_prims_ts_batch_decode_workspace_size(
            case.batch_size,
            case.num_qo_heads,
            case.num_kv_heads,
            _HEAD_DIM,
            case.page_size,
            case.max_kv_len,
            qo_indptr=inputs.qo_indptr,
            max_seq_len_q=case.seq_len_q,
            q_dtype=case.qk_dtype,
            v_dtype=FP8,
            out_dtype=case.output_dtype,
            mask_type=case.mask_type,
            window_left=case.window_left,
            query_paired=case.query_paired,
            sage_config=case.config,
        ),
        dtype=torch.int8,
        device="cuda",
    )
    prepared_out = torch.empty_like(eager)
    prepared = prepare_prims_ts_batch_decode_with_kv_cache(
        inputs.q,
        inputs.kv_cache,
        workspace,
        inputs.block_table,
        inputs.seq_lens,
        case.max_kv_len,
        out=prepared_out,
        qo_indptr=inputs.qo_indptr,
        max_seq_len_q=case.seq_len_q,
        mask_type=case.mask_type,
        window_left=case.window_left,
        query_paired=case.query_paired,
        sage=inputs.params,
        sage_config=case.config,
    )
    prepared.run(inputs.q, out=prepared_out)
    torch.testing.assert_close(prepared_out, eager, rtol=0, atol=0)
