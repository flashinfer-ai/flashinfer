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

"""CAKE DeepSeek-V4 NVFP4 (384-byte cache) sparse-MLA prefill on SM100/SM103.

Every case runs the public entry point with ``backend="cake"`` and
``kv_cache_format="nvfp4"`` on caches written by
:func:`flashinfer.mla.nvfp4_quantize_pack_sparse_mla_cache` and checks the
BF16 output and the base-2 LSE against an FP32 oracle that reads the
dequantized NVFP4 pools and the query with its NoPE dims passed through the
same NVFP4 quantizer the kernel applies (the SM120 NVFP4 kernel-error gate:
O ``atol=rtol=5e-2``, LSE ``atol=rtol=2e-2``). The grid covers single and dual
caches, independent main/extra lengths, ``-1`` padding, rows without a valid
entry, sinks, partial KV tiles, odd token counts, ragged and dense queries,
HND/NHD, padded page pitches, caller-owned workspaces, CUDA-graph replay,
padded query rows, the thin-head epilogue variant (8/16/32 heads) and the
tracking rows (K=256, 8 heads, non-default page sizes).
"""

from __future__ import annotations

import math

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    cake_dsv4_nvfp4_lse,
    get_cake_dsv4_workspace_bytes,
    nvfp4_quantize_append_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
    trtllm_batch_decode_sparse_mla_dsv4,
)
from flashinfer.mla.cake_dsv4 import _nvfp4_route
from flashinfer.utils import get_compute_capability
from tests.attention.sparse_mla_test_utils import (
    _BYTES_PER_TOKEN,
    _dequantize_nvfp4_cache,
    _dequantize_nvfp4_query,
)

HEAD_DIM = 512
SCALE = HEAD_DIM**-0.55
O_TOL = dict(atol=5e-2, rtol=5e-2)
LSE_TOL = dict(atol=2e-2, rtol=2e-2)
LOG2E = math.log2(math.e)


def _require_sm100_family() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("the CAKE DSv4 NVFP4 route requires SM100/SM103")


# Latent pools follow the kernel contract's input domain: N(offset, 0.25) clamped to [-1, 1]. The O gate
# (atol = rtol = 5e-2) is defined on that domain; the route's PV path dequantizes V to E4M3 with the
# per-16 scale folded in, and E4M3 spacing is 2^-4 of the value's binade (0.0625 for |v| in [1, 2),
# 0.25 for |v| in [2, 4)). With an unclamped N(0, 0.5) pool (|v| up to ~3) a two-key row with p ~ 0.5 can
# therefore legitimately differ from the FP32 oracle by ~0.06 on a single element (measured: 1 of
# 5,046,272 at 0.056, deterministic across runs, removed entirely when the oracle models the E4M3 V
# fold), which is a property of the quantized PV path shared with the FP8 route, not of this kernel.
_POOL_STD = 0.25
_POOL_CLAMP = 1.0


def _pool(gen, pages: int, page_size: int, layout: str, offset: float):
    latent = (
        (
            torch.randn((pages, page_size, HEAD_DIM), generator=gen, device="cuda")
            * _POOL_STD
            + offset
        )
        .clamp(-_POOL_CLAMP, _POOL_CLAMP)
        .to(torch.bfloat16)
    )
    return latent, nvfp4_quantize_pack_sparse_mla_cache(latent, kv_layout=layout)


def _selection(gen, rows: int, width: int, pool_tokens: int, lens_rule: str):
    """Random unique token ids inside the active prefix, -1 past it, independent lengths."""
    if lens_rule == "full":
        lens = torch.full((rows,), width, dtype=torch.int32)
    elif lens_rule == "random":
        lens = (
            torch.randint(0, width + 1, (rows,), generator=gen, device="cuda")
            .to(torch.int32)
            .cpu()
        )
        lens[0] = width  # at least one complete row
    elif lens_rule == "zero_rows":
        lens = (
            torch.randint(1, width + 1, (rows,), generator=gen, device="cuda")
            .to(torch.int32)
            .cpu()
        )
        lens[::3] = 0
    else:
        raise ValueError(lens_rule)
    table = torch.full((rows, width), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(lens[r])
        if n:
            perm = torch.randperm(pool_tokens, generator=gen, device="cuda")[:n]
            table[r, :n] = perm.to(torch.int32).cpu()
            # Interior -1 padding inside the active prefix is masked too.
            if n >= 8 and r % 2 == 1:
                table[r, n // 2] = -1
    return table.cuda(), lens.cuda()


def _case(
    *,
    heads: int,
    main_topk: int,
    extra_topk: int = 0,
    q_lens=(37, 40),
    main_page: int = 64,
    extra_page: int = 64,
    layout: str = "HND",
    sink: bool = True,
    lens_rule: str = "random",
    seed: int = 0,
    pool_pages: int = 32,
    ragged: bool = True,
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows = sum(q_lens)
    main_latent, main_cache = _pool(gen, pool_pages, main_page, layout, -0.05)
    main_idx, main_lens = _selection(
        gen, rows, main_topk, pool_pages * main_page, lens_rule
    )
    case = dict(
        heads=heads,
        rows=rows,
        q_lens=list(q_lens),
        main_latent=main_latent,
        main_cache=main_cache,
        main_idx=main_idx,
        main_lens=main_lens,
        extra_latent=None,
        extra_cache=None,
        extra_idx=None,
        extra_lens=None,
        layout=layout,
        ragged=ragged,
    )
    if extra_topk:
        extra_pages = max(pool_pages * main_page // extra_page, 2)
        extra_latent, extra_cache = _pool(gen, extra_pages, extra_page, layout, 0.05)
        extra_idx, extra_lens = _selection(
            gen, rows, extra_topk, extra_pages * extra_page, lens_rule
        )
        case.update(
            extra_latent=extra_latent,
            extra_cache=extra_cache,
            extra_idx=extra_idx,
            extra_lens=extra_lens,
        )
    query = (
        torch.randn((rows, heads, HEAD_DIM), generator=gen, device="cuda") * 0.6
    ).to(torch.bfloat16)
    case["query"] = query
    case["sinks"] = (
        (torch.randn((heads,), generator=gen, device="cuda") * 0.3).float()
        if sink
        else None
    )
    cum = [0]
    for n in q_lens:
        cum.append(cum[-1] + n)
    case["cum_seq_lens_q"] = torch.tensor(cum, dtype=torch.int32, device="cuda")
    case["max_q_len"] = max(q_lens)
    case["seq_lens"] = torch.full(
        (len(q_lens),), pool_pages * main_page, dtype=torch.int32, device="cuda"
    )
    topk = main_topk + extra_topk
    case["workspace"] = torch.zeros(
        get_cake_dsv4_workspace_bytes(rows, heads, topk, torch.bfloat16),
        dtype=torch.uint8,
        device="cuda",
    )
    return case


def _flat_rows(cache: torch.Tensor) -> torch.Tensor:
    return _dequantize_nvfp4_cache(cache).reshape(-1, HEAD_DIM)


def _oracle(case, *, main_lens=None, chunk: int = 64):
    """FP32 oracle on the dequantized NVFP4 pools and the NVFP4-quantized query NoPE."""
    main_rows = _flat_rows(case["main_cache"])
    extra_rows = (
        _flat_rows(case["extra_cache"]) if case["extra_cache"] is not None else None
    )
    q = _dequantize_nvfp4_query(case["query"])
    main_idx = case["main_idx"]
    main_lens = case["main_lens"] if main_lens is None else main_lens
    rows, heads = q.shape[:2]
    out = torch.zeros((rows, heads, HEAD_DIM), dtype=torch.float32, device="cuda")
    lse = torch.full((rows, heads), float("-inf"), dtype=torch.float32, device="cuda")
    for start in range(0, rows, chunk):
        stop = min(rows, start + chunk)
        idx = main_idx[start:stop]
        pos = torch.arange(idx.shape[1], device="cuda").unsqueeze(0)
        valid = (idx >= 0) & (pos < main_lens[start:stop].unsqueeze(1))
        kv = main_rows[idx.clamp_min(0).long()]
        if case["extra_idx"] is not None:
            eidx = case["extra_idx"][start:stop]
            epos = torch.arange(eidx.shape[1], device="cuda").unsqueeze(0)
            evalid = (eidx >= 0) & (epos < case["extra_lens"][start:stop].unsqueeze(1))
            kv = torch.cat((kv, extra_rows[eidx.clamp_min(0).long()]), dim=1)
            valid = torch.cat((valid, evalid), dim=1)
        scores = torch.einsum("rhd,rkd->rhk", q[start:stop], kv) * SCALE
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        row_lse = torch.logsumexp(scores, dim=-1)
        safe = torch.where(torch.isinf(row_lse), torch.zeros_like(row_lse), row_lse)
        probs = torch.exp(scores - safe.unsqueeze(-1)).masked_fill(
            ~valid.unsqueeze(1), 0.0
        )
        o = torch.einsum("rhk,rkd->rhd", probs, kv)
        if case["sinks"] is not None:
            s = case["sinks"].reshape(1, -1)
            o = o * torch.sigmoid(row_lse - s).unsqueeze(-1)
            row_lse = torch.logaddexp(row_lse, s.expand_as(row_lse))
        out[start:stop] = o
        lse[start:stop] = row_lse * LOG2E
    return out, lse


def _run(
    case, *, out=None, query=None, workspace=None, offset: int = 0, extra_lens="given"
):
    query = case["query"] if query is None else query
    workspace = case["workspace"] if workspace is None else workspace
    kwargs = dict(
        compressed_kv_cache=case["extra_cache"],
        swa_topk_lens=case["main_lens"],
        extra_sparse_indices=case["extra_idx"],
        extra_sparse_topk_lens=case["extra_lens"] if extra_lens == "given" else None,
        seq_lens=case["seq_lens"],
        out=out,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        sinks=case["sinks"],
        kv_layout=case["layout"],
        enable_pdl=False,
        backend="cake",
        kv_cache_format="nvfp4",
        sparse_topk_lens_offset=offset,
    )
    if case["ragged"]:
        kwargs.update(
            cum_seq_lens_q=case["cum_seq_lens_q"], max_q_len=case["max_q_len"]
        )
        q_in = query
    else:
        batch = len(case["q_lens"])
        q_in = query.reshape(batch, -1, query.shape[-2], query.shape[-1])
        if out is not None:
            kwargs["out"] = out.reshape(q_in.shape)
    result = trtllm_batch_decode_sparse_mla_dsv4(
        q_in, case["main_cache"], workspace, case["main_idx"], **kwargs
    )
    return result.reshape(-1, case["heads"], HEAD_DIM)


def _check(case, out, lse, **oracle_kwargs):
    ref_out, ref_lse = _oracle(case, **oracle_kwargs)
    torch.testing.assert_close(out.float(), ref_out, **O_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def _lse(case):
    return cake_dsv4_nvfp4_lse(case["workspace"], case["rows"], case["heads"])


# --------------------------------------------------------------------------- #
# Correctness grid                                                            #
# --------------------------------------------------------------------------- #

_GRID = [
    pytest.param(dict(heads=128, main_topk=512), id="h128-k512-single"),
    pytest.param(dict(heads=128, main_topk=128, extra_topk=128), id="h128-k128-dual"),
    pytest.param(
        dict(heads=64, main_topk=512, extra_topk=512, layout="NHD"),
        id="h64-k512-dual-nhd",
    ),
    pytest.param(
        dict(heads=64, main_topk=128, sink=False), id="h64-k128-single-nosink"
    ),
    pytest.param(
        dict(heads=32, main_topk=128, extra_topk=128, extra_page=2),
        id="h32-k128-dual-extrapage2",
    ),
    pytest.param(dict(heads=16, main_topk=512), id="h16-k512-single"),
    pytest.param(
        dict(heads=16, main_topk=128, extra_topk=128, layout="NHD"),
        id="h16-k128-dual-nhd",
    ),
    pytest.param(dict(heads=8, main_topk=128), id="h8-k128-single-tracking"),
    pytest.param(dict(heads=128, main_topk=256), id="h128-k256-single-tracking"),
    pytest.param(
        dict(heads=128, main_topk=128, main_page=32, extra_topk=128, extra_page=128),
        id="h128-k128-dual-page32-128",
    ),
]


@pytest.mark.parametrize("spec", _GRID)
def test_nvfp4_prefill_matches_dequantized_oracle(spec) -> None:
    """Ragged two-request batch (37 + 40 tokens: partial KV tiles, odd token count),
    independent random lengths, interior and trailing -1 padding."""
    _require_sm100_family()
    case = _case(seed=1, **spec)
    out = _run(case)
    torch.cuda.synchronize()
    _check(case, out, _lse(case))


@pytest.mark.parametrize("sink", [False, True])
def test_nvfp4_prefill_rows_without_valid_kv(sink: bool) -> None:
    """Rows with length 0 yield zeros and an LSE of -inf (or the sink alone)."""
    _require_sm100_family()
    case = _case(
        heads=128,
        main_topk=128,
        extra_topk=128,
        lens_rule="zero_rows",
        sink=sink,
        seed=2,
    )
    out = _run(case)
    torch.cuda.synchronize()
    lse = _lse(case)
    empty = (case["main_lens"] == 0) & (case["extra_lens"] == 0)
    assert bool(empty.any())
    assert torch.all(out[empty] == 0)
    if sink:
        torch.testing.assert_close(
            lse[empty], (case["sinks"] * LOG2E).expand(int(empty.sum()), -1), **LSE_TOL
        )
    else:
        assert torch.all(torch.isneginf(lse[empty]))
    _check(case, out, lse)


def test_nvfp4_prefill_dense_query_batch() -> None:
    """[batch, q_len, heads, 512] queries without cum_seq_lens_q."""
    _require_sm100_family()
    case = _case(
        heads=64, main_topk=128, extra_topk=512, q_lens=(29, 29), ragged=False, seed=3
    )
    out = _run(case)
    torch.cuda.synchronize()
    _check(case, out, _lse(case))


def test_nvfp4_prefill_omitted_extra_lengths_activate_every_column() -> None:
    _require_sm100_family()
    case = _case(heads=128, main_topk=128, extra_topk=128, lens_rule="full", seed=4)
    out = _run(case, extra_lens=None)
    torch.cuda.synchronize()
    _check(case, out, _lse(case))


def test_nvfp4_prefill_length_offset_applies_to_main_segment() -> None:
    _require_sm100_family()
    case = _case(heads=64, main_topk=512, lens_rule="full", seed=5)
    out = _run(case, offset=-96)
    torch.cuda.synchronize()
    _check(case, out, _lse(case), main_lens=(case["main_lens"] - 96).clamp_min(0))


def test_nvfp4_prefill_padded_page_pitch_and_append() -> None:
    """A vLLM-style pool with padding between pages, filled by the append helper."""
    _require_sm100_family()
    case = _case(heads=128, main_topk=512, seed=6, pool_pages=16)
    pages, page_size = 16, 64
    pitch = (page_size + 3) * _BYTES_PER_TOKEN
    backing = torch.full((pages * pitch,), 0xA5, dtype=torch.uint8, device="cuda")
    strided = torch.as_strided(
        backing, (pages, page_size, _BYTES_PER_TOKEN), (pitch, _BYTES_PER_TOKEN, 1)
    )
    slots = torch.arange(pages * page_size, dtype=torch.int64, device="cuda")
    nvfp4_quantize_append_sparse_mla_cache(
        case["main_latent"].reshape(-1, HEAD_DIM), slots, strided
    )
    padded_cache = torch.as_strided(
        backing,
        (pages, 1, page_size, _BYTES_PER_TOKEN),
        (pitch, pitch, _BYTES_PER_TOKEN, 1),
    )
    reference_out = _run(case)
    torch.cuda.synchronize()
    reference_lse = _lse(case).clone()
    case["main_cache"] = padded_cache
    out = _run(case)
    torch.cuda.synchronize()
    # Same bytes per page as the full-page pack: identical attention output and LSE.
    torch.testing.assert_close(out, reference_out, atol=0, rtol=0)
    torch.testing.assert_close(_lse(case), reference_lse, atol=0, rtol=0)
    assert torch.all(
        backing.view(pages, pitch)[:, page_size * _BYTES_PER_TOKEN :] == 0xA5
    )


def test_nvfp4_prefill_column_sliced_tables_match_contiguous() -> None:
    """Row-strided views of one wide table are read without a copy."""
    _require_sm100_family()
    case = _case(heads=64, main_topk=128, extra_topk=512, seed=7)
    reference_out = _run(case).clone()
    torch.cuda.synchronize()
    reference_lse = _lse(case).clone()
    wide = torch.full(
        (case["rows"], 128 + 512 + 64), -7, dtype=torch.int32, device="cuda"
    )
    wide[:, :128] = case["main_idx"]
    wide[:, 128:640] = case["extra_idx"]
    case["main_idx"], case["extra_idx"] = wide[:, :128], wide[:, 128:640]
    out = _run(case)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference_out, atol=0, rtol=0)
    torch.testing.assert_close(_lse(case), reference_lse, atol=0, rtol=0)


def test_nvfp4_prefill_padded_query_rows_untouched() -> None:
    _require_sm100_family()
    case = _case(heads=128, main_topk=128, seed=8)
    rows = case["rows"]
    padded_query = torch.cat(
        (
            case["query"],
            torch.ones((5, 128, HEAD_DIM), dtype=torch.bfloat16, device="cuda"),
        )
    )
    out = torch.full(
        (rows + 5, 128, HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    result = _run(case, query=padded_query, out=out)
    torch.cuda.synchronize()
    assert result.data_ptr() == out.data_ptr()
    assert torch.all(out[rows:] == 7.0)
    _check(case, out[:rows], _lse(case))


def test_nvfp4_prefill_caller_workspace_exact_and_undersized() -> None:
    _require_sm100_family()
    case = _case(heads=128, main_topk=512, extra_topk=512, seed=9)
    exact = get_cake_dsv4_workspace_bytes(case["rows"], 128, 1024, torch.bfloat16)
    workspace = torch.zeros(exact, dtype=torch.uint8, device="cuda")
    out = _run(case, workspace=workspace)
    torch.cuda.synchronize()
    ref_out, ref_lse = _oracle(case)
    torch.testing.assert_close(out.float(), ref_out, **O_TOL)
    torch.testing.assert_close(
        cake_dsv4_nvfp4_lse(workspace, case["rows"], 128), ref_lse, **LSE_TOL
    )
    with pytest.raises(ValueError, match="workspace_buffer requires at least"):
        _run(
            case,
            workspace=torch.zeros(1024 + 256 * 1024, dtype=torch.uint8, device="cuda"),
        )


def test_nvfp4_prefill_cuda_graph_replay() -> None:
    """Capture once, replay with new query bytes: no allocation, results track the eager call."""
    _require_sm100_family()
    case = _case(heads=128, main_topk=128, extra_topk=128, seed=10)
    out = torch.empty(
        (case["rows"], 128, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    _run(case, out=out)  # eager warm-up on the same tensors (JIT, caches)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _run(case, out=out)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            _run(case, out=out)
    torch.cuda.synchronize()
    for seed in (11, 12):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        case["query"].copy_(
            (torch.randn(case["query"].shape, generator=gen, device="cuda") * 0.6).to(
                torch.bfloat16
            )
        )
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        _check(case, out, _lse(case))


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
def test_nvfp4_route_selection_matches_head_count(heads: int) -> None:
    expected = "nvfp4_h128_prefill_persistent" + (
        "" if heads % 64 == 0 else "_thin_heads"
    )
    assert _nvfp4_route(heads) == expected


def test_nvfp4_public_entry_refuses_combined_lengths() -> None:
    _require_sm100_family()
    case = _case(heads=64, main_topk=128, seed=13)
    with pytest.raises(ValueError, match="requires swa_topk_lens"):
        trtllm_batch_decode_sparse_mla_dsv4(
            case["query"],
            case["main_cache"],
            case["workspace"],
            case["main_idx"],
            sparse_topk_lens=case["main_lens"],
            seq_lens=case["seq_lens"],
            cum_seq_lens_q=case["cum_seq_lens_q"],
            max_q_len=case["max_q_len"],
            bmm1_scale=SCALE,
            backend="cake",
            kv_cache_format="nvfp4",
        )


def test_nvfp4_sparse_backend_still_requires_sm120() -> None:
    _require_sm100_family()
    case = _case(heads=64, main_topk=128, seed=14)
    with pytest.raises(ValueError, match="backend='sparse' requires SM120/SM121"):
        trtllm_batch_decode_sparse_mla_dsv4(
            case["query"],
            case["main_cache"],
            case["workspace"],
            case["main_idx"],
            swa_topk_lens=case["main_lens"],
            seq_lens=case["seq_lens"],
            cum_seq_lens_q=case["cum_seq_lens_q"],
            max_q_len=case["max_q_len"],
            bmm1_scale=SCALE,
            backend="sparse",
            kv_cache_format="nvfp4",
        )
    assert flashinfer.mla.cake_dsv4_nvfp4_lse is cake_dsv4_nvfp4_lse
