# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness coverage for the PrimTS 1CTA DSA (token-sparse) MLA decode kernel.

GLM-5.3 / DeepSeek-V3.2 decode: every packed query token attends to the
``sparse_mla_top_k`` rows of an E4M3 ``[N, 576]`` KV pool its indexer routed
to it.  The reference (``tests/test_helpers/dsa_sparse_mla_reference.py``) is
an FP32 softmax over the same E4M3 rows.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

cutlass = pytest.importorskip("cutlass", minversion="4.7.0")

from flashinfer.attention.prims_ts import (
    dsa as dsa_module,
    prims_ts_dsa_sparse_mla,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.token_sparse_launch import (
    compile_dsa_sparse_kernel,
)
from tests.test_helpers import dsa_sparse_mla_reference as ref

_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS DSA sparse MLA requires SM100 or SM103",
)
_FP8 = torch.float8_e4m3fn
_HEAD_DIM = 576
_V_DIM = 512
_SM_SCALE = 1.0 / 16.0
# The kernel's E4M3 P / FP32-accumulated output against the FP32 reference over
# the same E4M3 rows (2.2e-2 .. 2.8e-2 on B200).
_REL_L2_TOL = 4e-2
# Base-2 LSE: the kernel's E4M3 P and FP32 accumulation order against the FP32
# reference; observed differences stay below 0.03.
_LSE_ATOL = 0.1


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    total_q: int = 8
    heads: int = 64
    top_k: int = 2048
    rows: int = 1 << 15
    ragged: bool = False  # random active prefix per token (sparse_mla_top_k_lens)
    neg: float = 0.0  # fraction of -1 slots inside the active width
    sequential: bool = False  # token t uses rows t*K .. t*K+K-1 (no gather scatter)
    q_zero: bool = False  # Q = 0: the output is the mean of the V rows (V path only)
    zero_len_token: bool = False  # token 1 gets sparse_mla_top_k_lens == 0
    duplicates: bool = False  # every token routes to each of its rows twice
    masked_first_tile: bool = False  # slots 0..127 are -1, later tiles are valid
    seed: int = 0


_FP8_CASES = [
    Case("t8"),
    Case("t1", total_q=1),
    Case("ragged", total_q=5, ragged=True),
    Case("neg30", neg=0.3),
    Case("t16-ragged-neg50", total_q=16, ragged=True, neg=0.5),
    Case("topk128", total_q=3, top_k=128),
    Case("sequential", sequential=True),
    Case("qzero", q_zero=True),
    Case("h128", total_q=4, heads=128),
    Case("h128-t1", total_q=1, heads=128),
    Case("h128-t2", total_q=2, heads=128),
    Case("two-waves", total_q=300),
    Case("lens-zero", total_q=4, ragged=True, zero_len_token=True),
    Case("topk136", total_q=3, top_k=136),
    Case("topk256-ragged", total_q=3, top_k=256, ragged=True),
    Case("topk136-unsplit", total_q=40, top_k=136),
    Case("dup-rows", total_q=3, duplicates=True),
    Case("topk4", total_q=3, top_k=4),
    Case("topk64", total_q=3, top_k=64),
    Case("topk64-ragged", total_q=3, top_k=64, ragged=True),
    Case("topk4-lens-zero", total_q=4, top_k=4, ragged=True, zero_len_token=True),
    Case("masked-first-tile", total_q=3, masked_first_tile=True),
]


@dataclasses.dataclass
class Fixture:
    query: torch.Tensor  # [T, H, 576] e4m3
    block_tables: torch.Tensor  # [T, K] int32
    top_k_lens: torch.Tensor  # [T] int32 (always materialized)
    top_k_lens_arg: torch.Tensor | None  # what the API receives
    fp8_pool: torch.Tensor  # [N, 576] e4m3


def build_fixture(case: Case, device: torch.device) -> Fixture:
    gen = torch.Generator(device=device).manual_seed(case.seed)
    T, H, K, N = case.total_q, case.heads, case.top_k, case.rows
    x = torch.randn(N, _HEAD_DIM, device=device, generator=gen) * 0.5
    q = (torch.randn(T, H, _HEAD_DIM, device=device, generator=gen) * 0.5).to(_FP8)
    if case.q_zero:
        q = torch.zeros_like(q)
    if case.sequential:
        idx = (
            torch.arange(T, device=device)[:, None] * K
            + torch.arange(K, device=device)[None, :]
        ).to(torch.int32) % N
    else:
        idx = torch.stack(
            [torch.randperm(N, device=device, generator=gen)[:K] for _ in range(T)]
        ).to(torch.int32)
    if case.duplicates:
        idx[:, 1::2] = idx[:, 0::2]
    lens = torch.full((T,), K, dtype=torch.int32, device=device)
    lens_arg = None
    if case.ragged:
        lens = torch.randint(1, K + 1, (T,), device=device, generator=gen).to(
            torch.int32
        )
        lens[0] = K
        if case.zero_len_token:
            lens[1] = 0
        lens_arg = lens
    if case.neg > 0:
        idx[torch.rand(T, K, device=device, generator=gen) < case.neg] = -1
    if case.masked_first_tile:
        idx[:, :128] = -1
    return Fixture(
        query=q,
        block_tables=idx,
        top_k_lens=lens,
        top_k_lens_arg=lens_arg,
        fp8_pool=x.to(_FP8),
    )


def _rel_l2(out: torch.Tensor, expected: torch.Tensor) -> float:
    d = out.float() - expected.float()
    return (d.norm() / expected.float().norm().clamp(min=1e-20)).item()


def _check_against(
    out: torch.Tensor,
    lse: torch.Tensor,
    expected: torch.Tensor,
    expected_lse: torch.Tensor,
    *,
    tol: float = _REL_L2_TOL,
) -> None:
    assert not torch.isnan(out).any()
    rel = _rel_l2(out, expected)
    assert rel < tol, f"rel_l2 {rel:.3e} >= {tol:.1e}"
    for t in range(out.shape[0]):
        assert _rel_l2(out[t], expected[t]) < tol or expected[t].abs().sum() == 0, t
    finite = torch.isfinite(expected_lse)
    assert torch.equal(torch.isfinite(lse), finite)
    assert torch.allclose(lse[finite], expected_lse[finite], atol=_LSE_ATOL, rtol=0)


def _run_fp8(fx: Fixture, *, skip_corr_threshold: float = 0.0, **kwargs):
    return prims_ts_dsa_sparse_mla(
        fx.query,
        fx.fp8_pool,
        fx.block_tables,
        None,
        None,
        sparse_mla_top_k=fx.block_tables.shape[1],
        sparse_mla_top_k_lens=fx.top_k_lens_arg,
        bmm1_scale=_SM_SCALE,
        bmm2_scale=1.0,
        skip_corr_threshold=skip_corr_threshold,
        **kwargs,
    )


def _fp8_reference(fx: Fixture):
    return ref.sparse_mla_reference(
        fx.query,
        fx.fp8_pool.float(),
        fx.block_tables,
        bmm1_scale=_SM_SCALE,
        bmm2_scale=1.0,
        top_k_lens=fx.top_k_lens,
    )


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("case", _FP8_CASES, ids=lambda c: c.name)
def test_prims_ts_dsa_fp8_matches_reference(case: Case) -> None:
    device = torch.device("cuda")
    fx = build_fixture(case, device)
    lse = torch.empty((case.total_q, case.heads), dtype=torch.float32, device=device)
    out = _run_fp8(fx, lse=lse)
    torch.cuda.synchronize()
    expected, expected_lse = _fp8_reference(fx)
    _check_against(out, lse, expected, expected_lse)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_prims_ts_dsa_fully_masked_token() -> None:
    """A token whose every slot is -1 yields zeros and an LSE of -inf; its neighbours are unaffected."""
    device = torch.device("cuda")
    fx = build_fixture(Case("masked", total_q=4, neg=0.2), device)
    fx.block_tables[1] = -1
    lse = torch.empty((4, 64), dtype=torch.float32, device=device)
    run, reference = _run_fp8, _fp8_reference
    out = run(fx, lse=lse)
    torch.cuda.synchronize()
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.isneginf(lse[1]).all()
    expected, expected_lse = reference(fx)
    _check_against(out, lse, expected, expected_lse)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "case",
    [
        Case("skipcorr", total_q=6, ragged=True, neg=0.1),
        Case("skipcorr-masked-first-tile", total_q=3, masked_first_tile=True),
    ],
    ids=lambda c: c.name,
)
def test_prims_ts_dsa_skip_correction(case: Case) -> None:
    """The lazy rescale (threshold 8 = the E4M3 P headroom) stays inside the reference tolerance."""
    device = torch.device("cuda")
    fx = build_fixture(case, device)
    lse = torch.empty((case.total_q, 64), dtype=torch.float32, device=device)
    run, reference = _run_fp8, _fp8_reference
    exact = run(fx).clone()
    out = run(fx, skip_corr_threshold=8.0, lse=lse)
    torch.cuda.synchronize()
    expected, expected_lse = reference(fx)
    _check_against(out, lse, expected, expected_lse)
    assert _rel_l2(out, exact) < _REL_L2_TOL


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_prims_ts_dsa_cuda_graph_replay() -> None:
    """Captured launches read the live buffers: new payloads and routes replay bitwise like eager."""
    device = torch.device("cuda")
    case = Case("graph", total_q=5, ragged=True, neg=0.1)
    fx = build_fixture(case, device)
    run, reference = _run_fp8, _fp8_reference
    out = torch.empty((case.total_q, 64, _V_DIM), dtype=torch.bfloat16, device=device)
    lse = torch.empty((case.total_q, 64), dtype=torch.float32, device=device)
    # The first eager call compiles and caches the Gather4 descriptors of these
    # pool buffers; capture must not allocate or encode on the host.
    run(fx, out=out, lse=lse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        run(fx, out=out, lse=lse)
    torch.cuda.synchronize()
    for seed in (1, 2):
        fresh = build_fixture(dataclasses.replace(case, seed=seed), device)
        fx.query.copy_(fresh.query)
        fx.fp8_pool.copy_(fresh.fp8_pool)
        fx.block_tables.copy_(fresh.block_tables)
        fx.top_k_lens.copy_(fresh.top_k_lens)
        out.zero_()
        lse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        replayed, replayed_lse = out.clone(), lse.clone()
        eager = run(fresh)
        torch.cuda.synchronize()
        assert torch.equal(replayed, eager)
        expected, expected_lse = reference(fresh)
        _check_against(replayed, replayed_lse, expected, expected_lse)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_prims_ts_dsa_validate_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FLASHINFER_VALIDATE_INPUTS=1`` rejects routes outside the pool and bad active widths."""
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    device = torch.device("cuda")
    fx = build_fixture(Case("validate", total_q=3, ragged=True, neg=0.2), device)
    run = _run_fp8
    baseline = run(fx).clone()
    torch.cuda.synchronize()
    rows = fx.fp8_pool.shape[0]
    good = fx.block_tables.clone()
    for bad_value in (rows, -2):
        fx.block_tables.copy_(good)
        fx.block_tables[0, 3] = bad_value
        with pytest.raises(ValueError, match="block_tables entries"):
            run(fx)
    fx.block_tables.copy_(good)
    good_lens = fx.top_k_lens.clone()
    fx.top_k_lens[2] = fx.block_tables.shape[1] + 1
    with pytest.raises(ValueError, match="sparse_mla_top_k_lens"):
        run(fx)
    fx.top_k_lens.copy_(good_lens)
    assert torch.equal(run(fx), baseline)
    # Slots past a token's active length are never read, so they are not
    # checked (incl. a zero-length token).
    fx.top_k_lens[2] = fx.block_tables.shape[1] // 2
    fx.top_k_lens[1] = 0
    expected = run(fx).clone()
    fx.block_tables[2, fx.block_tables.shape[1] // 2 :] = rows + 7
    fx.block_tables[1] = rows + 7
    assert torch.equal(run(fx), expected)


def _valid_args(device: torch.device) -> dict:
    T, K = 2, 64
    q = torch.zeros((T, 64, _HEAD_DIM), dtype=_FP8, device=device)
    idx = torch.zeros((T, K), dtype=torch.int32, device=device)
    args = dict(
        query=q,
        block_tables=idx,
        seq_lens=None,
        cum_seq_lens_q=None,
        sparse_mla_top_k=K,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        kv_cache=torch.zeros((128, _HEAD_DIM), dtype=_FP8, device=device),
    )
    return args


_COMMON_INVALID = [
    ("query-dtype", lambda a, d: a.update(query=a["query"].to(torch.bfloat16))),
    ("query-rank", lambda a, d: a.update(query=a["query"][:, 0])),
    ("query-heads", lambda a, d: a.update(query=a["query"][:, :48].contiguous())),
    ("query-head-dim", lambda a, d: a.update(query=a["query"][..., :512].contiguous())),
    (
        "query-strided",
        lambda a, d: a.update(
            query=torch.zeros((2, 64, 2 * _HEAD_DIM), dtype=_FP8, device=d)[
                ..., :_HEAD_DIM
            ]
        ),
    ),
    (
        "query-misaligned",
        lambda a, d: a.update(
            query=torch.zeros(2 * 64 * _HEAD_DIM + 1, dtype=_FP8, device=d)[1:].view(
                2, 64, _HEAD_DIM
            )
        ),
    ),
    ("tables-dtype", lambda a, d: a.update(block_tables=a["block_tables"].long())),
    ("tables-shape", lambda a, d: a.update(block_tables=a["block_tables"][:, :32])),
    ("tables-rank", lambda a, d: a.update(block_tables=a["block_tables"][0])),
    ("topk-not-multiple-of-4", lambda a, d: a.update(sparse_mla_top_k=66)),
    ("topk-zero", lambda a, d: a.update(sparse_mla_top_k=0)),
    (
        "lens-shape",
        lambda a, d: a.update(
            sparse_mla_top_k_lens=torch.ones(3, dtype=torch.int32, device=d)
        ),
    ),
    (
        "lens-dtype",
        lambda a, d: a.update(
            sparse_mla_top_k_lens=torch.ones(2, dtype=torch.int64, device=d)
        ),
    ),
    (
        "seq-lens-dtype",
        lambda a, d: a.update(seq_lens=torch.ones(2, dtype=torch.int64, device=d)),
    ),
    ("scale-nan", lambda a, d: a.update(bmm1_scale=float("nan"))),
    ("scale-bool", lambda a, d: a.update(bmm2_scale=True)),
    ("skip-corr-too-large", lambda a, d: a.update(skip_corr_threshold=8.5)),
    ("skip-corr-negative", lambda a, d: a.update(skip_corr_threshold=-1.0)),
    (
        "out-dtype",
        lambda a, d: a.update(
            out=torch.empty((2, 64, _V_DIM), dtype=torch.float16, device=d)
        ),
    ),
    (
        "out-shape",
        lambda a, d: a.update(
            out=torch.empty((2, 64, 576), dtype=torch.bfloat16, device=d)
        ),
    ),
    (
        "lse-shape",
        lambda a, d: a.update(lse=torch.empty((2, 32), dtype=torch.float32, device=d)),
    ),
]
_FP8_INVALID = [
    ("cache-dtype", lambda a, d: a.update(kv_cache=a["kv_cache"].to(torch.bfloat16))),
    ("cache-row", lambda a, d: a.update(kv_cache=a["kv_cache"][:, :512].contiguous())),
    (
        "cache-misaligned",
        lambda a, d: a.update(
            kv_cache=torch.zeros(128 * _HEAD_DIM + 1, dtype=_FP8, device=d)[1:].view(
                128, _HEAD_DIM
            )
        ),
    ),
]
# The validation message each case must raise (re.search patterns).
_INVALID_MATCH = {
    "query-dtype": "query must be a CUDA E4M3 tensor",
    "query-rank": "query must be a CUDA E4M3 tensor",
    "query-heads": r"query must be \[T, 64\|128",
    "query-head-dim": r"query must be \[T, 64\|128",
    "query-strided": "query must be contiguous",
    "query-misaligned": "query must start at a 16-byte-aligned address",
    "tables-dtype": "block_tables must be contiguous INT32",
    "tables-shape": "block_tables must be contiguous INT32",
    "tables-rank": "block_tables must be contiguous INT32",
    "topk-not-multiple-of-4": "sparse_mla_top_k must be a positive multiple of 4",
    "topk-zero": "sparse_mla_top_k must be a positive multiple of 4",
    "lens-shape": "sparse_mla_top_k_lens must be contiguous INT32",
    "lens-dtype": "sparse_mla_top_k_lens must be contiguous INT32",
    "seq-lens-dtype": "seq_lens must be a 1-D CUDA INT32",
    "scale-nan": "bmm1_scale must be a finite number",
    "scale-bool": "bmm2_scale must be a finite number",
    "skip-corr-too-large": r"skip_corr_threshold must be in \[0, 8\]",
    "skip-corr-negative": r"skip_corr_threshold must be in \[0, 8\]",
    "out-dtype": "out must be contiguous BF16",
    "out-shape": "out must be contiguous BF16",
    "lse-shape": "lse must be contiguous FP32",
    "cache-dtype": "kv_cache dtype .* is not supported",
    "cache-row": "kv_cache must be a contiguous CUDA E4M3 tensor",
    "cache-misaligned": "kv_cache must start at a 16-byte-aligned address",
}


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "name,mutate",
    _COMMON_INVALID + _FP8_INVALID,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_prims_ts_dsa_rejects_invalid_arguments(name: str, mutate) -> None:
    device = torch.device("cuda")
    args = _valid_args(device)
    mutate(args, device)
    with pytest.raises(ValueError, match=_INVALID_MATCH[name]):
        prims_ts_dsa_sparse_mla(**args)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_prims_ts_dsa_zero_tokens() -> None:
    """A step without query tokens returns the empty output without a launch."""
    device = torch.device("cuda")
    args = _valid_args(device)
    args["query"] = args["query"][:0]
    args["block_tables"] = args["block_tables"][:0]
    out = prims_ts_dsa_sparse_mla(**args)
    assert tuple(out.shape) == (0, 64, _V_DIM) and out.dtype == torch.bfloat16


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
def test_prims_ts_dsa_compiled_kernel_is_shared_across_token_counts() -> None:
    """The JIT key is (heads, top-k, skip-correction, split-KV count, V head-dim
    split); T is a runtime extent, so token counts that select the same split
    share one kernel."""
    device = torch.device("cuda")
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    top_k = 256
    totals = (1, 7, 64, 65, 150)
    variants = {dsa_module._select_fp8_split(t, top_k, sm_count) for t in totals}
    before = compile_dsa_sparse_kernel.cache_info()
    for total_q in totals:
        _run_fp8(build_fixture(Case("shared", total_q=total_q, top_k=top_k), device))
    torch.cuda.synchronize()
    after = compile_dsa_sparse_kernel.cache_info()
    assert after.misses - before.misses <= len(variants)
    assert after.hits - before.hits >= len(totals) - len(variants)


def test_prims_ts_dsa_fp8_split_policy() -> None:
    """E4M3 follows trtllm-gen: at least two tiles per split CTA within one wave
    (one tile for a top-k of one tile pair), then a V head-dim split while the
    grid fills at most a quarter / half of the SMs."""
    select = dsa_module._select_fp8_split
    ctas = (1, 4, 5, 8, 9, 16, 18, 19, 37, 38, 74, 75, 96)
    assert [select(t, 2048, 148) for t in ctas] == [
        (8, 4),
        (8, 4),
        (8, 2),
        (8, 2),
        (8, 2),
        (8, 1),
        (8, 1),
        (4, 1),
        (4, 1),
        (2, 1),
        (2, 1),
        (1, 1),
        (1, 1),
    ]
    assert select(1, 512, 148) == (2, 4)  # two tile pairs
    # One tile pair: one tile per CTA while the V split applies, else no split.
    assert [select(t, 256, 148) for t in (1, 18, 32, 37, 38, 64)] == [
        (2, 4),
        (2, 4),
        (2, 2),
        (2, 2),
        (1, 1),
        (1, 1),
    ]
    assert select(1, 128, 148) == (1, 1)  # a single tile
    assert select(48, 2048, 148) == (2, 1)  # e.g. 24 tokens with 128 heads


_SPLIT_CASES = [
    Case("ragged-neg30", total_q=6, ragged=True, neg=0.3),
    Case("lens-zero", total_q=4, ragged=True, zero_len_token=True),
    Case("topk1028", total_q=3, top_k=1028),
]


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    ("split", "head_dim_split"),
    ((1, 1), (2, 1), (2, 4), (4, 2), (16, 4)),
)
@pytest.mark.parametrize("case", _SPLIT_CASES, ids=lambda c: c.name)
def test_prims_ts_dsa_split_kv_matches_reference(
    split: int,
    head_dim_split: int,
    case: Case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced splits (incl. empty and one-tile splits, and V head-dim splits)
    match the reference."""
    monkeypatch.setattr(
        dsa_module,
        "_select_fp8_split",
        lambda num_ctas, top_k, sm_count: (split, head_dim_split),
    )
    device = torch.device("cuda")
    fx = build_fixture(case, device)
    lse = torch.empty((case.total_q, case.heads), dtype=torch.float32, device=device)
    out = _run_fp8(fx, lse=lse)
    torch.cuda.synchronize()
    expected, expected_lse = _fp8_reference(fx)
    _check_against(out, lse, expected, expected_lse)


@_REQUIRES_PRIMTS_GPU
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("shape", [(-1, 64, _HEAD_DIM), (-1, 1, 64, _HEAD_DIM)])
def test_prims_ts_dsa_paged_cache_layout(shape) -> None:
    """``[num_pages, page_size, 576]`` / ``[num_pages, 1, page_size, 576]`` caches
    are addressed as flat rows: entry ``page * page_size + offset``."""
    device = torch.device("cuda")
    fx = build_fixture(Case("paged", total_q=4, ragged=True, neg=0.1), device)
    flat = _run_fp8(fx).clone()
    paged = prims_ts_dsa_sparse_mla(
        fx.query,
        fx.fp8_pool.view(*shape),
        fx.block_tables,
        None,
        None,
        sparse_mla_top_k=fx.block_tables.shape[1],
        sparse_mla_top_k_lens=fx.top_k_lens_arg,
        bmm1_scale=_SM_SCALE,
        bmm2_scale=1.0,
    )
    torch.cuda.synchronize()
    assert torch.equal(paged, flat)
