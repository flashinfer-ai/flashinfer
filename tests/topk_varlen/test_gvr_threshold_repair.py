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
"""GVR (V1) non-converged threshold-search repair regressions.

Port of TRT-LLM PR #18094's regression suite (plus the FlashInfer-found
short-row case) onto ``top_k_varlen(backend="gvr")``. Before the repair,
these inputs made the Phase-2/3 threshold search terminate without a
threshold whose count lands in ``[K, kC]``, and the kernel then shipped a
silently wrong top-K: identity indices ``row[0:K]`` (degenerate hints),
or an underfilled row whose untouched output slots keep stale/-1 garbage
(hostile hints, tie plateaus wider than kC, rows with ``N_eff = K + 1``).

The correctness contract checked here is tie-interchangeable EXACT top-K:
unique in-range indices whose gathered-value multiset bitwise-equals the
``torch.topk`` value multiset.
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.utils import get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE, reason="flashinfer not importable"
)


def _cute_dsl_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


def _gvr_hw_supported() -> bool:
    if not (_FLASHINFER_AVAILABLE and torch.cuda.is_available()):
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    return (
        flashinfer.top_k_varlen.is_backend_supported("gvr", major * 10 + minor)
        and _cute_dsl_available()
    )


requires_gvr = pytest.mark.skipif(
    not _gvr_hw_supported(),
    reason="gvr requires Blackwell (sm_100/103) + nvidia-cutlass-dsl",
)

_DEV = "cuda"


def _assert_exact_topk(indices, logits, seq_lens, top_k, next_n=1, compress_ratio=1):
    """Per-row tie-aware exactness for rows with N_eff > top_k."""
    sl = seq_lens.cpu().tolist()
    for r in range(indices.shape[0]):
        n_eff = min(
            (sl[r // next_n] - next_n + (r % next_n) + 1) // compress_ratio,
            logits.shape[1],
        )
        assert n_eff > top_k, "test rows must be non-degenerate"
        idx = indices[r].to(torch.int64)
        assert int(idx.min()) >= 0, f"row {r}: negative index (underfilled row)"
        assert int(idx.max()) < n_eff, f"row {r}: index past N_eff"
        assert int(torch.unique(idx).numel()) == top_k, f"row {r}: duplicate indices"
        got = torch.sort(logits[r].float().gather(0, idx) + 0.0, descending=True).values
        ref = torch.sort(
            torch.topk(logits[r, :n_eff].float(), top_k).values + 0.0,
            descending=True,
        ).values
        assert torch.equal(got, ref), f"row {r}: top-K value multiset mismatch"


def _make_hint(kind, flat_row, n_idx_space, top_k):
    """Hint tensors in COMPRESSED index space [0, n_idx_space)."""
    if kind == "bottom_k":
        # bracket entirely below the answer
        return torch.topk(flat_row, top_k, largest=False).indices.int()
    if kind == "uniform":
        # all-identical hint -> degenerate bracket (the old row[0:K] emit)
        return torch.full((top_k,), n_idx_space // 2, dtype=torch.int32, device=_DEV)
    if kind == "random":
        return torch.randint(0, n_idx_space, (top_k,), dtype=torch.int32, device=_DEV)
    raise ValueError(kind)


@requires_gvr
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("hint", ["bottom_k", "uniform", "random"])
@pytest.mark.parametrize("load_balance", [True, False])
def test_gvr_hostile_hint(top_k, hint, load_balance):
    """Hints that don't straddle the true K-th value must not affect
    exactness (they may only slow the search)."""
    N, cr = 65536, 4
    torch.manual_seed(1234)
    logits = torch.randn(1, N, dtype=torch.float32, device=_DEV)
    pre_idx = _make_hint(hint, logits[0], N, top_k).unsqueeze(0)
    seq_lens = torch.full((1,), N * cr, dtype=torch.int32, device=_DEV)
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        compress_ratio=cr,
        backend="gvr",
        load_balance=load_balance,
    )
    _assert_exact_topk(indices, logits, seq_lens, top_k, compress_ratio=cr)


def _run_gvr_filled(logits, seq_lens, top_k, load_balance):
    """Run gvr into a -7-filled buffer and check every slot was written with a
    unique in-range index (the only contract an all-tie row can have)."""
    rows, n = logits.shape
    gen = torch.Generator(device=_DEV).manual_seed(rows * n + top_k)
    hint = torch.randint(
        0, n, (rows, top_k), generator=gen, device=_DEV, dtype=torch.int32
    )
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=hint,
        out_indices=out,
        backend="gvr",
        load_balance=load_balance,
    )
    torch.cuda.synchronize()
    assert int((out == -7).sum()) == 0, "unwritten output slots"
    assert bool(((out >= 0) & (out < n)).all()), "index out of range"
    for r in range(rows):
        assert int(torch.unique(out[r]).numel()) == top_k, f"row {r}: duplicate indices"
    return out


@requires_gvr
@pytest.mark.parametrize("load_balance", [True, False])
def test_gvr_rows_below_neg_flt_max(load_balance):
    """Rows with fewer than K values above -FLT_MAX (all -inf, or a few finite
    values followed by a -inf tail) used to come back with unwritten slots:
    the two-sided repair anchored the lower bracket end at -FLT_MAX, so
    count(>= val_lo) < K contradicted the collapse invariant and the plateau
    fill never ran. The anchor is now -inf. Any K unique in-range indices are
    a valid answer for an all-tie row; every finite value must be selected in
    the partial row."""
    for rows, n, k in [(1, 8192, 1024), (4, 8192, 512), (2, 32768, 1024)]:
        logits = torch.full((rows, n), float("-inf"), device=_DEV)
        seq_lens = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
        _run_gvr_filled(logits, seq_lens, k, load_balance)
    # one all -inf row inside a normal batch: the other rows stay exact
    rows, n, k = 4, 8192, 1024
    torch.manual_seed(5)
    logits = torch.randn(rows, n, device=_DEV)
    logits[2] = float("-inf")
    seq_lens = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = _run_gvr_filled(logits, seq_lens, k, load_balance)
    for r in (0, 1, 3):
        got = torch.sort(logits[r][out[r].long()], descending=True).values
        ref = torch.sort(torch.topk(logits[r], k).values, descending=True).values
        assert torch.equal(got, ref), f"row {r}: inexact"
    # 10 finite values + -inf tail, N > K: the 10 must all be selected
    logits = torch.full((1, n), float("-inf"), device=_DEV)
    logits[0, :10] = torch.arange(10, device=_DEV).float()
    seq_lens = torch.full((1,), n, dtype=torch.int32, device=_DEV)
    out = _run_gvr_filled(logits, seq_lens, k, load_balance)
    assert set(range(10)) <= set(out[0].tolist()), "finite values dropped"


@requires_gvr
@pytest.mark.parametrize("n_pos", [3, 100, 1000])
def test_gvr_relu_sparse_plateau(n_pos):
    """ReLU-sparse rows: n_pos < K positives over an exact-0.0 plateau wider
    than the candidate buffer. No threshold has count in [K, kC]; the old
    fail-soft returned (top_k - n_pos) trailing -1 slots."""
    top_k, N = 2048, 32768
    torch.manual_seed(61)
    row = torch.zeros(N, dtype=torch.float32, device=_DEV)
    pos = torch.randperm(N, device=_DEV)[:n_pos]
    row[pos] = torch.rand(n_pos, device=_DEV) + 1.0
    logits = row.unsqueeze(0)
    pre_idx = torch.topk(row, top_k).indices.int().unsqueeze(0)
    seq_lens = torch.full((1,), N, dtype=torch.int32, device=_DEV)
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr"
    )
    _assert_exact_topk(indices, logits, seq_lens, top_k)


@requires_gvr
@pytest.mark.parametrize("next_n", [2, 4])
@pytest.mark.parametrize("hint", ["bottom_k", "uniform"])
def test_gvr_mtp_hostile_hint(next_n, hint):
    """MTP geometry (per-row N_eff arithmetic) with the repair firing on
    every row; per-request kv_len differs by one token to exercise the
    mod-cr boundary."""
    top_k, N, cr, n_req = 512, 65536, 4, 2
    torch.manual_seed(7)
    num_rows = n_req * next_n
    logits = torch.randn(num_rows, N, dtype=torch.float32, device=_DEV)
    hint_row = _make_hint(hint, logits[0], N, top_k)
    pre_idx = hint_row.unsqueeze(0).expand(n_req, top_k).contiguous()
    seq_lens = torch.tensor([N * cr, N * cr - 1], dtype=torch.int32, device=_DEV)
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        next_n=next_n,
        compress_ratio=cr,
        backend="gvr",
    )
    _assert_exact_topk(
        indices, logits, seq_lens, top_k, next_n=next_n, compress_ratio=cr
    )


@requires_gvr
@pytest.mark.parametrize("batch", [16, 64, 256])
@pytest.mark.parametrize("load_balance", [True, False])
def test_gvr_neff_kplus1_rows(batch, load_balance):
    """FlashInfer-found case: batches where most rows have N_eff = K + 1
    (the acceptance window is a knife-edge) shipped out-of-range indices.
    Mirrors the failing sweep config (short scenario, N=8192, K=1024)."""
    N, top_k = 8192, 1024
    seed = batch * 7 + N // 64
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    logits = torch.randn(batch, N, generator=gen, device=_DEV, dtype=torch.float32)
    seq_lens = torch.randint(
        top_k + 1, max(N // 8, top_k + 2), (batch,), generator=gen, device=_DEV
    ).int()
    seq_lens[: max(batch // 4, 1)] = N
    pre_idx = torch.zeros(batch, top_k, dtype=torch.int32, device=_DEV)
    sl = seq_lens.cpu().tolist()
    for b in range(batch):
        n0 = min(sl[b], N)
        true_top = torch.topk(logits[b, :n0], min(top_k, n0)).indices.int()
        pre_idx[b] = torch.randint(
            0, n0, (top_k,), generator=gen, device=_DEV, dtype=torch.int32
        )
        n_hit = int(min(top_k, n0) * 0.6)
        pre_idx[b, :n_hit] = true_top[:n_hit]
    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        pre_idx=pre_idx,
        backend="gvr",
        load_balance=load_balance,
    )
    _assert_exact_topk(indices, logits, seq_lens, top_k)
