# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake DeepSeek-V4 NVFP4 sparse-MLA prefill on SM100 / SM103.

``trtllm_batch_decode_sparse_mla_dsv4(..., backend="cake", kv_cache_format="nvfp4")``
runs the NVFP4 prefill bodies from 128 metadata rows on: ``nvfp4_h64_prefill``
(8 to 64 heads, one CTA per query token) and ``nvfp4_h128_prefill`` (128 heads,
one two-CTA cluster per token). Every case checks the BF16 output and the
base-2 LSE against an FP32 oracle over the dequantized NVFP4 pools and the
query as the route packs it (the NVFP4 cache row format), with the kernel-error
gate of the NVFP4 route (O ``atol=rtol=5e-2``, LSE ``atol=2e-2``).
"""

import math

import pytest
import torch

from flashinfer.jit.cpp_ext import get_cuda_version, is_cuda_version_at_least
from flashinfer.mla import (
    get_cake_dsv4_workspace_bytes,
    nvfp4_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)
from flashinfer.mla import cake_dsv4 as cake
from flashinfer.utils import get_compute_capability
from tests.attention.sparse_mla_test_utils import _dequantize_nvfp4_cache

_HEAD_DIM = 512
_SCALE = _HEAD_DIM**-0.55
_LOG2E = math.log2(math.e)
_CAKE_CCS = ((10, 0), (10, 3))
_CAKE_MIN_CUDA = "13.4"
_O_TOL = dict(atol=5e-2, rtol=5e-2)
_LSE_TOL = dict(atol=2e-2, rtol=0.0)


def _require_cake_arch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    cc = get_compute_capability(torch.device("cuda"))
    if tuple(cc) not in _CAKE_CCS:
        pytest.skip(
            f"backend='cake' NVFP4 prefill needs SM100/SM103, got SM{cc[0]}{cc[1]}"
        )
    if not is_cuda_version_at_least(_CAKE_MIN_CUDA):
        pytest.skip(
            f"backend='cake' NVFP4 prefill needs CUDA {_CAKE_MIN_CUDA} or newer, "
            f"got {get_cuda_version()}"
        )


def _pool(gen, pages: int, page_size: int, layout: str, offset: float):
    """A packed NVFP4 pool and its dequantized rows [pages * page_size, 512]."""
    latent = (
        (
            torch.randn((pages, page_size, _HEAD_DIM), generator=gen, device="cuda")
            * 0.6
            + offset
        )
        .clamp(-1, 1)
        .to(torch.bfloat16)
    )
    cache = nvfp4_quantize_pack_sparse_mla_cache(latent, kv_layout=layout)
    return cache, _dequantize_nvfp4_cache(cache).reshape(-1, _HEAD_DIM)


def _selection(gen, rows: int, width: int, pool_tokens: int, lens_rule: str):
    """Random tokens per row inside the active prefix, -1 past it and inside it."""
    if lens_rule == "full":
        lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    elif lens_rule == "random":
        lens = torch.randint(0, width + 1, (rows,), generator=gen, device="cuda").to(
            torch.int32
        )
        lens[0] = width
    else:
        raise ValueError(lens_rule)
    table = torch.randint(
        0, pool_tokens, (rows, width), generator=gen, device="cuda"
    ).to(torch.int32)
    positions = torch.arange(width, device="cuda")[None, :]
    table = table.masked_fill(positions >= lens[:, None], -1)
    # interior padding inside the active prefix is masked too
    odd = torch.arange(rows, device="cuda") % 2 == 1
    table[odd & (lens >= 8), 3] = -1
    return table, lens


def _case(
    *,
    heads: int,
    topk: int,
    q_lens=(67, 70),
    extra_topk: int = 0,
    page_size: int = 64,
    extra_page_size: int = 64,
    layout: str = "HND",
    sink: bool = True,
    lens_rule: str = "random",
    pool_pages: int = 48,
    seed: int = 0,
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = sum(q_lens)
    main_cache, main_rows = _pool(gen, pool_pages, page_size, layout, -0.05)
    main_idx, main_lens = _selection(gen, rows, topk, main_rows.shape[0], lens_rule)
    extra_cache = extra_idx = extra_lens = extra_rows = None
    if extra_topk:
        extra_pages = max(pool_pages * page_size // extra_page_size, 2)
        extra_cache, extra_rows = _pool(gen, extra_pages, extra_page_size, layout, 0.05)
        extra_idx, extra_lens = _selection(
            gen, rows, extra_topk, extra_rows.shape[0], lens_rule
        )
    # Each head aims at one valid key of its row so a wrong key set changes O by O(1).
    query = torch.randn((rows, heads, _HEAD_DIM), generator=gen, device="cuda") * 0.3
    first = main_idx[:, 0].clamp_min(0).long()
    target = main_rows[first]  # [rows, 512]
    query = query + 6.0 / _SCALE * target[:, None, :] / target.norm(dim=-1).pow(2)[
        :, None, None
    ].clamp_min(1e-6)
    query = query.clamp(-8, 8).to(torch.bfloat16)
    sinks = (
        (torch.randn((heads,), generator=gen, device="cuda") * 0.3).float()
        if sink
        else None
    )
    cum = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        dtype=torch.int32,
        device="cuda",
    )
    workspace = torch.zeros(
        get_cake_dsv4_workspace_bytes(
            rows,
            heads,
            topk,
            torch.bfloat16,
            kv_cache_format="nvfp4",
            extra_topk=extra_topk,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    return dict(
        heads=heads,
        rows=rows,
        q_lens=q_lens,
        query=query,
        main_cache=main_cache,
        main_rows=main_rows,
        main_idx=main_idx,
        main_lens=main_lens,
        extra_cache=extra_cache,
        extra_rows=extra_rows,
        extra_idx=extra_idx,
        extra_lens=extra_lens,
        sinks=sinks,
        layout=layout,
        cum_seq_lens_q=cum,
        max_q_len=max(q_lens),
        seq_lens=torch.full(
            (len(q_lens),), pool_pages * page_size, dtype=torch.int32, device="cuda"
        ),
        workspace=workspace,
        topk=topk,
        extra_topk=extra_topk,
    )


def _packed_query_rows(query: torch.Tensor) -> torch.Tensor:
    """The query as the route packs it: one NVFP4 cache row per (token, head), dequantized."""
    rows, heads = query.shape[:2]
    cache = nvfp4_quantize_pack_sparse_mla_cache(
        query.reshape(rows * heads, 1, _HEAD_DIM)
    )
    return _dequantize_nvfp4_cache(cache).reshape(rows, heads, _HEAD_DIM)


def _valid(indices: torch.Tensor, lens) -> torch.Tensor:
    positions = torch.arange(indices.shape[1], device="cuda")[None, :]
    valid = indices >= 0
    if lens is not None:
        valid &= positions < lens[:, None]
    return valid


def _oracle(case, *, main_lens="given", extra_lens="given", chunk: int = 32):
    q = _packed_query_rows(case["query"])
    main_lens = case["main_lens"] if main_lens == "given" else main_lens
    extra_lens = case["extra_lens"] if extra_lens == "given" else extra_lens
    rows, heads = q.shape[:2]
    out = torch.zeros((rows, heads, _HEAD_DIM), dtype=torch.float32, device="cuda")
    lse = torch.empty((rows, heads), dtype=torch.float32, device="cuda")
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        idx = case["main_idx"][start:stop]
        valid = _valid(idx, None if main_lens is None else main_lens[start:stop])
        kv = case["main_rows"][idx.clamp_min(0).long()]
        if case["extra_idx"] is not None:
            eidx = case["extra_idx"][start:stop]
            kv = torch.cat((kv, case["extra_rows"][eidx.clamp_min(0).long()]), 1)
            valid = torch.cat(
                (
                    valid,
                    _valid(
                        eidx, None if extra_lens is None else extra_lens[start:stop]
                    ),
                ),
                1,
            )
        scores = torch.einsum("rhd,rkd->rhk", q[start:stop], kv) * _SCALE
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        row_lse = torch.logsumexp(scores, dim=-1)
        safe = torch.where(torch.isinf(row_lse), torch.zeros_like(row_lse), row_lse)
        probs = torch.exp(scores - safe[..., None]).masked_fill(~valid[:, None, :], 0.0)
        o = torch.einsum("rhk,rkd->rhd", probs, kv)
        if case["sinks"] is not None:
            s = case["sinks"][None, :]
            o = o * torch.sigmoid(row_lse - s)[..., None]
            row_lse = torch.logaddexp(row_lse, s.expand_as(row_lse))
        out[start:stop] = o
        lse[start:stop] = row_lse * _LOG2E
    return out, lse


def _run(case, *, out=None, main_lens="given", extra_lens="given", dense: bool = False):
    query = case["query"]
    kwargs = dict(
        compressed_kv_cache=case["extra_cache"],
        sparse_topk_lens=case["main_lens"] if main_lens == "given" else main_lens,
        extra_sparse_indices=case["extra_idx"],
        extra_sparse_topk_lens=case["extra_lens"]
        if extra_lens == "given"
        else extra_lens,
        seq_lens=case["seq_lens"],
        out=out,
        bmm1_scale=_SCALE,
        bmm2_scale=1.0,
        sinks=case["sinks"],
        kv_layout=case["layout"],
        backend="cake",
        kv_cache_format="nvfp4",
    )
    if dense:
        batch = len(case["q_lens"])
        q_in = query.reshape(batch, -1, case["heads"], _HEAD_DIM)
        if out is not None:
            kwargs["out"] = out.reshape(q_in.shape)
    else:
        q_in = query
        kwargs.update(
            cum_seq_lens_q=case["cum_seq_lens_q"], max_q_len=case["max_q_len"]
        )
    result = trtllm_batch_decode_sparse_mla_dsv4(
        q_in, case["main_cache"], case["workspace"], case["main_idx"], **kwargs
    )
    return result.reshape(case["rows"], case["heads"], _HEAD_DIM)


def _lse(case) -> torch.Tensor:
    plan = cake._nvfp4_plan(
        num_query_tokens=case["rows"],
        num_heads=case["heads"],
        sparse_topk=case["topk"],
        extra_topk=case["extra_topk"],
        sm_count=torch.cuda.get_device_properties(
            torch.device("cuda")
        ).multi_processor_count,
    )
    layout = cake.cake_dsv4_workspace_layout(
        case["rows"], case["heads"], plan.num_splits, with_lse=True, with_prefill=True
    )
    offset, _ = layout.lse
    return (
        cake._workspace_bytes(case["workspace"])[
            offset : offset + case["rows"] * case["heads"] * 4
        ]
        .view(torch.float32)
        .view(case["rows"], case["heads"])
        .clone()
    )


def _check(case, out, **oracle_kwargs):
    torch.cuda.synchronize()
    ref_out, ref_lse = _oracle(case, **oracle_kwargs)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref_out, **_O_TOL)
    torch.testing.assert_close(_lse(case), ref_lse, **_LSE_TOL)


_GRID = [
    pytest.param(dict(heads=128, topk=512), id="h128-k512-single"),
    pytest.param(dict(heads=128, topk=128, extra_topk=128), id="h128-k128-dual"),
    pytest.param(
        dict(heads=128, topk=128, extra_topk=512, extra_page_size=2),
        id="h128-k128-dual-page2",
    ),
    pytest.param(
        dict(heads=64, topk=512, extra_topk=512, layout="NHD"), id="h64-k512-dual-nhd"
    ),
    pytest.param(dict(heads=64, topk=128, sink=False), id="h64-k128-single-nosink"),
    pytest.param(
        dict(heads=32, topk=128, extra_topk=128, extra_page_size=32),
        id="h32-k128-dual-page32",
    ),
    pytest.param(dict(heads=16, topk=512), id="h16-k512-single"),
    pytest.param(
        dict(heads=16, topk=128, extra_topk=128, layout="NHD"), id="h16-k128-dual-nhd"
    ),
    pytest.param(dict(heads=8, topk=256, page_size=128), id="h8-k256-single-page128"),
    pytest.param(
        dict(heads=64, topk=128, q_lens=(128,)), id="h64-k128-one-request-128"
    ),
    pytest.param(
        dict(heads=128, topk=128, q_lens=(300, 213)), id="h128-k128-513-tokens"
    ),
]


@pytest.mark.parametrize("spec", _GRID)
def test_nvfp4_prefill_matches_dequantized_oracle(spec):
    """Two ragged requests with independent random lengths and -1 padding."""
    _require_cake_arch()
    case = _case(seed=1, **spec)
    out = _run(case)
    _check(case, out)


def test_nvfp4_prefill_dense_query_and_full_lengths():
    """A dense [batch, q_len] query, no length vectors (every table entry active)."""
    _require_cake_arch()
    case = _case(
        seed=2, heads=64, topk=128, extra_topk=128, q_lens=(72, 72), lens_rule="full"
    )
    out = _run(case, dense=True, main_lens=None, extra_lens=None)
    _check(case, out, main_lens=None, extra_lens=None)


def test_nvfp4_prefill_rows_without_keys_are_zero():
    """Rows whose selection is all -1 give O = 0 (and a -inf LSE without a sink)."""
    _require_cake_arch()
    case = _case(seed=3, heads=32, topk=128, sink=False, lens_rule="full")
    case["main_idx"][5::17] = -1
    out = _run(case)
    torch.cuda.synchronize()
    assert torch.count_nonzero(out[5::17]) == 0
    _check(case, out)


@pytest.mark.parametrize("heads", [16, 128])
def test_nvfp4_prefill_graph_replay_is_bitwise(heads):
    """The route is capturable (no allocation, the packed query and lengths live in the workspace)."""
    _require_cake_arch()
    case = _case(seed=4, heads=heads, topk=128, extra_topk=128, lens_rule="full")
    out = torch.empty_like(case["query"])
    eager = _run(case, out=out).clone()
    eager_lse = _lse(case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(case, out=out)  # warm the modules outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _run(case, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    out.zero_()
    case["workspace"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    assert torch.equal(_lse(case), eager_lse)
    _check(case, out)


def test_nvfp4_route_switches_to_the_prefill_bodies_at_128_rows(monkeypatch):
    _require_cake_arch()
    calls = []
    original = cake._nvfp4_prefill

    def recording(**kwargs):
        calls.append(kwargs["query_rows"].shape[0])
        return original(**kwargs)

    monkeypatch.setattr(cake, "_nvfp4_prefill", recording)
    small = _case(seed=5, heads=16, topk=128, q_lens=(64, 63))
    _run(small)
    assert calls == []
    large = _case(seed=5, heads=16, topk=128, q_lens=(64, 64))
    out = _run(large)
    assert calls == [128]
    _check(large, out)
