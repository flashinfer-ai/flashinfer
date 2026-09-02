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
"""Tests for the ``gvr_2`` (self-sampling GVR V2) top_k_varlen backend.

Ported from TRT-LLM's ``test_gvr_selfsampling_topk.py`` (PR #17821) onto the
FlashInfer ``top_k_varlen(backend="gvr_2")`` API. The correctness contract is
tie-interchangeable EXACT top-K:

* selected indices unique and inside the row's valid prefix;
* the sorted multiset of gathered values bitwise-equals the sorted
  ``torch.topk`` values (signed zeros normalized via ``+ 0.0``);
* short rows (``n <= top_k``): identity indices ``{0..n-1}`` + ``-1`` tail
  (values tail = ``-FLT_MAX``);
* per-row tails beyond each row's valid prefix are poisoned with ``+3e38`` so
  any out-of-bounds read corrupts the multiset and fails loudly.

The route-dispatch fuzz tests are CPU-only (pure-Python host dispatch).
"""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.topk_varlen.kernels import gvr2_topk_host as _host
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


def _gvr2_hw_supported() -> bool:
    if not (_FLASHINFER_AVAILABLE and torch.cuda.is_available()):
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    return (
        flashinfer.top_k_varlen.is_backend_supported("gvr_2", major * 10 + minor)
        and _cute_dsl_available()
    )


_IS_GVR2_CAPABLE = _gvr2_hw_supported()
requires_gvr2 = pytest.mark.skipif(
    not _IS_GVR2_CAPABLE,
    reason="gvr_2 requires datacenter Blackwell (sm_100/103, Rubin sm_107) "
    "+ nvidia-cutlass-dsl",
)

_DEV = "cuda"
_FMIN = -3.4028234663852886e38  # torch.finfo(torch.float32).min


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _round64(n: int) -> int:
    return (n + 63) // 64 * 64


def _row_valid_lens(seq_lens_host, rows, next_n, cr, n_cap):
    """Mirror of the per-row device formula (uncompressed seq_lens in)."""
    out = []
    for r in range(rows):
        n = (seq_lens_host[r // next_n] - next_n + (r % next_n) + 1) // cr
        out.append(min(max(n, 0), n_cap))
    return out


def _make_case(batch_size, n_valid, top_k, seed, hit_ratio=0.6):
    """Batch-uniform case: fp32 randn-2.0 logits, 64-rounded width, poisoned
    tail, hint = 60% true top-K prefix + random valid indices."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    npad = _round64(n_valid)
    logits = torch.randn(batch_size, npad, generator=gen, device=_DEV) - 2.0
    logits[:, n_valid:] = 3e38  # poison: an OOB read corrupts the multiset
    k_eff = min(top_k, n_valid)
    ref = torch.topk(logits[:, :n_valid], k_eff, dim=1)
    n_hit = int(k_eff * hit_ratio)
    pre_idx = torch.randint(
        0, n_valid, (batch_size, top_k), generator=gen, device=_DEV, dtype=torch.int32
    )
    pre_idx[:, :n_hit] = ref.indices[:, :n_hit].int()
    seq_lens = torch.full((batch_size,), n_valid, dtype=torch.int32, device=_DEV)
    return logits, seq_lens, pre_idx, ref.values


def _check_exact(logits, indices, n_valid, ref_vals):
    """Tie-aware exactness: unique in-range indices + bitwise value-multiset
    equality (signed zeros normalized)."""
    top_k = indices.shape[1]
    idx64 = indices.to(torch.int64)
    assert int(idx64.min()) >= 0, "negative output index"
    assert int(idx64.max()) < n_valid, "output index past n_valid"
    for row in range(indices.shape[0]):
        assert int(torch.unique(idx64[row]).numel()) == top_k, (
            f"row {row}: duplicate indices"
        )
    got = torch.gather(logits, 1, idx64)
    got_sorted = torch.sort(got + 0.0, dim=1, descending=True).values
    ref_sorted = torch.sort(ref_vals + 0.0, dim=1, descending=True).values
    assert torch.equal(got_sorted, ref_sorted), (
        "top-K value multiset mismatch (inexact or padding read)"
    )


def _check_varlen_rows(logits, indices, n_r, top_k, values=None):
    """Per-row varlen oracle: short rows = identity + -1 tail; long rows =
    tie-aware exact."""
    for r in range(indices.shape[0]):
        n = n_r[r]
        if n <= top_k:
            head = indices[r, :n].to(torch.int64) if n > 0 else None
            if n > 0:
                assert torch.equal(
                    torch.sort(head).values, torch.arange(n, device=_DEV)
                ), f"row {r}: short-path head not a permutation of arange({n})"
            assert bool((indices[r, n:] == -1).all()), f"row {r}: tail not -1"
            if values is not None:
                if n > 0:
                    assert torch.equal(values[r, :n], logits[r, :n]), (
                        f"row {r}: short-path values head"
                    )
                assert bool((values[r, n:] == _FMIN).all()), (
                    f"row {r}: values tail not -FLT_MAX"
                )
        else:
            idx = indices[r].to(torch.int64)
            assert int(idx.min()) >= 0 and int(idx.max()) < n, f"row {r}: range"
            assert int(torch.unique(idx).numel()) == top_k, f"row {r}: dups"
            ref = torch.topk(logits[r, :n], top_k).values
            got = torch.sort(
                torch.gather(logits[r], 0, idx) + 0.0, descending=True
            ).values
            assert torch.equal(got, torch.sort(ref + 0.0, descending=True).values), (
                f"row {r} inexact"
            )
            if values is not None:
                assert torch.equal(values[r], torch.gather(logits[r], 0, idx)), (
                    f"row {r}: values mismatch"
                )


def _make_varlen_case(kv, next_n, cr, top_k, seed, msl_c=None):
    """Ragged case in the FlashInfer contract: uncompressed per-request
    seq_lens, compressed-space logits of width msl_c (4-aligned)."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    batch = len(kv)
    rows = batch * next_n
    if msl_c is None:
        msl_c = _round64(max(max(kv) // cr, top_k + 1))
    logits = torch.randn(rows, msl_c, generator=gen, device=_DEV) - 2.0
    seq_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    n_r = _row_valid_lens(kv, rows, next_n, cr, msl_c)
    for r in range(rows):
        logits[r, n_r[r] :] = 3e38  # per-row poison
    pre_idx = torch.zeros(batch, top_k, dtype=torch.int32, device=_DEV)
    for b in range(batch):
        lo = max(min(n_r[b * next_n : (b + 1) * next_n]), 1)
        pre_idx[b] = torch.randint(
            0, lo, (top_k,), generator=gen, device=_DEV, dtype=torch.int32
        )
    return logits, seq_lens, pre_idx, n_r, msl_c


def _run_gvr2(logits, seq_lens, top_k, pre_idx, **kw):
    return flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr_2", **kw
    )


# ---------------------------------------------------------------------------
# batch-uniform exactness (via uniform seq_lens)
# ---------------------------------------------------------------------------

# gate-edge (131075/131076 straddle the K=2048 hint-band gate), small-N floor,
# mid band, deployment-envelope top — from the upstream suite.
_CASES = [
    (512, 4099),
    (512, 65536),
    (512, 262143),
    (1024, 16387),
    (1024, 131072),
    (2048, 4111),
    (2048, 131075),
    (2048, 131076),
    (2048, 262144),
]


@requires_gvr2
@pytest.mark.parametrize("top_k,n_valid", _CASES)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_gvr2_exactness(top_k, n_valid, batch_size):
    seed = n_valid * 31 + top_k + batch_size
    logits, seq_lens, pre_idx, ref_vals = _make_case(batch_size, n_valid, top_k, seed)
    indices, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
    _check_exact(logits, indices, n_valid, ref_vals)


_SHORT_CASES = [
    (512, 256),
    (512, 511),
    (512, 512),
    (1024, 64),
    (1024, 1024),
    (2048, 1000),
    (2048, 2047),
    (2048, 2048),
]


@requires_gvr2
@pytest.mark.parametrize("top_k,n_valid", _SHORT_CASES)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_gvr2_short_path(top_k, n_valid, batch_size):
    seed = top_k + n_valid
    logits, seq_lens, pre_idx, _ = _make_case(batch_size, n_valid, top_k, seed)
    indices, values = _run_gvr2(logits, seq_lens, top_k, pre_idx, return_values=True)
    n_r = [n_valid] * batch_size
    _check_varlen_rows(logits, indices, n_r, top_k, values=values)


@requires_gvr2
@pytest.mark.parametrize("top_k,n_valid", [(512, 8192), (1024, 32768)])
def test_gvr2_values_output(top_k, n_valid):
    logits, seq_lens, pre_idx, ref_vals = _make_case(4, n_valid, top_k, seed=42)
    indices, values = _run_gvr2(logits, seq_lens, top_k, pre_idx, return_values=True)
    _check_exact(logits, indices, n_valid, ref_vals)
    assert values is not None and values.dtype == torch.float32
    idx64 = indices.to(torch.int64)
    assert torch.equal(values, torch.gather(logits, 1, idx64))


@requires_gvr2
@pytest.mark.parametrize(
    "hint_kind",
    ["all_zero", "all_same", "all_max", "half_dup", "minus_one_tail", "all_minus_one"],
)
@pytest.mark.parametrize("top_k,n_valid", [(512, 8192), (2048, 131075)])
def test_gvr2_degenerate_hints(hint_kind, top_k, n_valid):
    """Hints steer the sampling ladder but never exactness — degenerate hints
    must not affect the output contract."""
    logits, seq_lens, pre_idx, ref_vals = _make_case(2, n_valid, top_k, seed=7)
    if hint_kind == "all_zero":
        pre_idx.zero_()
    elif hint_kind == "all_same":
        pre_idx.fill_(1234)
    elif hint_kind == "all_max":
        pre_idx.fill_(n_valid - 1)
    elif hint_kind == "half_dup":
        pre_idx[:, top_k // 2 :] = pre_idx[:, : top_k // 2]
    elif hint_kind == "minus_one_tail":
        pre_idx[:, top_k // 2 :] = -1
    elif hint_kind == "all_minus_one":
        pre_idx.fill_(-1)
    indices, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
    _check_exact(logits, indices, n_valid, ref_vals)


# ---------------------------------------------------------------------------
# varlen production contract
# ---------------------------------------------------------------------------

_VARLEN_CASES = [
    # (kv, next_n, cr, top_k)
    ([33000, 8200, 300], 1, 1, 512),  # cr1 hetero + short row
    ([131075, 32800, 2000], 1, 4, 512),  # cr4 hetero (compressed index space)
    ([9000, 5001], 2, 1, 512),  # cr1 MTP-2
    ([65540], 4, 4, 1024),  # cr4 MTP-4 (boundary-crossing rows)
    ([40000, 7003], 3, 4, 512),  # cr4 next_n=3 (formula must generalize)
]


@requires_gvr2
@pytest.mark.parametrize("kv,next_n,cr,top_k", _VARLEN_CASES)
@pytest.mark.parametrize("with_values", [False, True])
def test_gvr2_varlen(kv, next_n, cr, top_k, with_values):
    seed = sum(kv) + next_n + cr
    logits, seq_lens, pre_idx, n_r, _ = _make_varlen_case(kv, next_n, cr, top_k, seed)
    indices, values = _run_gvr2(
        logits,
        seq_lens,
        top_k,
        pre_idx,
        next_n=next_n,
        compress_ratio=cr,
        return_values=with_values,
    )
    _check_varlen_rows(logits, indices, n_r, top_k, values=values)


@requires_gvr2
def test_gvr2_zero_kv_slot():
    """kv_len = 0 (evicted CUDA-graph slot) and kv_len < next_n (zero-window)
    rows emit all -1; live rows in the same launch stay exact."""
    for kv, next_n in ([[0, 40000], 2], [[1, 33000], 2], [[3, 65540, 0], 4]):
        top_k = 512
        logits, seq_lens, pre_idx, n_r, _ = _make_varlen_case(
            kv, next_n, 1, top_k, seed=13
        )
        indices, _ = _run_gvr2(
            logits, seq_lens, top_k, pre_idx, next_n=next_n, compress_ratio=1
        )
        _check_varlen_rows(logits, indices, n_r, top_k)


@requires_gvr2
def test_gvr2_launch_modes():
    """Streaming-main SPLIT + TSH-floor domain (16 rows) and BLK=512 wide
    batch (200 rows)."""
    for rows, base, step, top_k in ((16, 40000, 977, 1024), (200, 6000, 64, 512)):
        kv = [base + step * i for i in range(rows)]
        logits, seq_lens, pre_idx, n_r, _ = _make_varlen_case(
            kv, 1, 1, top_k, seed=rows
        )
        indices, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
        _check_varlen_rows(logits, indices, n_r, top_k)


@requires_gvr2
def test_gvr2_heterogeneous_lengths_main():
    """304 rows (BLK=256 non-split main) with random per-request lengths
    spanning long/mid/short/n<=k/zero-window; each row checked against its
    own prefix topk. Strongest single portable test from upstream."""
    rows, top_k, msl_c, cr, next_n = 304, 1024, 65536, 4, 1
    gen = torch.Generator(device=_DEV).manual_seed(1000 + rows)
    kv = []
    for i in range(rows):
        m = i % 5
        if m == 0:
            kv.append(
                int(
                    torch.randint(
                        msl_c * cr // 2, msl_c * cr, (1,), generator=gen, device=_DEV
                    )
                )
            )
        elif m == 1:
            kv.append(
                int(
                    torch.randint(
                        top_k * cr + 1,
                        msl_c * cr // 2,
                        (1,),
                        generator=gen,
                        device=_DEV,
                    )
                )
            )
        elif m == 2:
            kv.append(
                int(torch.randint(1, top_k * cr, (1,), generator=gen, device=_DEV))
            )
        elif m == 3:
            kv.append(top_k * cr)  # n == k boundary
        else:
            kv.append(0)  # zero-window
    logits, seq_lens, pre_idx, n_r, _ = _make_varlen_case(
        kv, next_n, cr, top_k, seed=rows, msl_c=msl_c
    )
    indices, _ = _run_gvr2(
        logits, seq_lens, top_k, pre_idx, next_n=next_n, compress_ratio=cr
    )
    _check_varlen_rows(logits, indices, n_r, top_k)


@requires_gvr2
def test_gvr2_noncontiguous_arena_view():
    """Column-sliced logits view (row stride > width, 256-rounded arena — the
    DSL paged-MQA layout) must stay exact; only stride(1)==1 is required."""
    rows, msl, top_k = 8, 8300, 512
    stride = (msl + 255) // 256 * 256
    gen = torch.Generator(device=_DEV).manual_seed(5)
    arena = torch.randn(rows, stride, generator=gen, device=_DEV) - 2.0
    logits = arena[:, :msl]
    assert not logits.is_contiguous()
    kv = [8300, 8000, 6000, 4000, 2000, 513, 512, 100]
    seq_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    n_r = _row_valid_lens(kv, rows, 1, 1, msl)
    for r in range(rows):
        logits[r, n_r[r] :] = 3e38
    pre_idx = torch.randint(
        0, 100, (rows, top_k), generator=gen, device=_DEV, dtype=torch.int32
    )
    indices, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
    _check_varlen_rows(logits, indices, n_r, top_k)


@requires_gvr2
@pytest.mark.parametrize(
    "rows,msl",
    [
        (16, 131008),  # streaming main, R=9 split
        (256, 8128),  # streaming main, R=1
        (16, 8128),  # register-resident
        (16, 65472),  # clustered register-resident
        (64, 131008),  # cluster split
    ],
)
def test_gvr2_arena_view_oversized_seq_lens(rows, msl):
    """A kv length beyond the logical row width must clamp to the WIDTH, not
    to the arena row stride: on a 256-rounded arena view the columns in
    [width, stride) belong to other data and must never be classified (radix
    parity — ``seq_lens`` is clamped to ``logits.shape[1]``). The arena tail
    is poisoned with +3e38 so a stride-clamped family fails the multiset."""
    top_k = 1024
    stride = (msl + 255) // 256 * 256
    assert stride > msl
    gen = torch.Generator(device=_DEV).manual_seed(11)
    arena = torch.randn(rows, stride, generator=gen, device=_DEV) - 2.0
    arena[:, msl:] = 3e38  # arena tail: outside every row's logical width
    logits = arena[:, :msl]
    assert not logits.is_contiguous()
    kv = [stride + 777 if r % 2 == 0 else msl - 64 * (r % 5) for r in range(rows)]
    seq_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    n_r = _row_valid_lens(kv, rows, 1, 1, msl)  # clamps to the width
    for r in range(rows):
        logits[r, n_r[r] :] = 3e38
    pre_idx = torch.randint(
        0, top_k, (rows, top_k), generator=gen, device=_DEV, dtype=torch.int32
    )
    indices, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
    _check_varlen_rows(logits, indices, n_r, top_k)


@pytest.mark.skipif(
    not (
        _FLASHINFER_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
    ),
    reason="needs two CUDA devices",
)
def test_gvr2_logits_on_non_current_device():
    """Launch with logits on a device that is NOT torch.cuda.current_device():
    the launcher cache key, the DSL compile target and the launch itself must
    all follow the tensor's device — a heterogeneous multi-GPU process must
    never reuse or run an engine compiled for another architecture."""
    if not _cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl not installed")
    cur = torch.cuda.current_device()
    capable = []
    for i in range(torch.cuda.device_count()):
        if i == cur:
            continue
        major, minor = get_compute_capability(torch.device(f"cuda:{i}"))
        if flashinfer.top_k_varlen.is_backend_supported("gvr_2", major * 10 + minor):
            capable.append(i)
    if not capable:
        pytest.skip("no gvr_2-capable device other than the current one")
    dev = torch.device(f"cuda:{capable[0]}")
    rows, n_valid, top_k = 4, 16384, 1024
    gen = torch.Generator(device=dev).manual_seed(3)
    logits = torch.randn(rows, n_valid, generator=gen, device=dev) - 2.0
    seq_lens = torch.full((rows,), n_valid, dtype=torch.int32, device=dev)
    pre_idx = torch.randint(
        0, n_valid, (rows, top_k), generator=gen, device=dev, dtype=torch.int32
    )
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, top_k, pre_idx=pre_idx, backend="gvr_2"
    )
    torch.cuda.synchronize(dev)
    assert indices.device == logits.device
    assert torch.cuda.current_device() == cur, "entry must restore the current device"
    _check_exact(logits, indices, n_valid, torch.topk(logits, top_k, dim=1).values)


@requires_gvr2
def test_gvr2_workspace_override():
    """Caller-provided workspace (multi-stream escape hatch) must agree with
    the default per-device slab."""
    kv, top_k = [131075, 32800, 2000], 512
    logits, seq_lens, pre_idx, n_r, _ = _make_varlen_case(kv, 1, 1, top_k, seed=17)
    idx_default, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
    ws_bytes = _host.workspace_bytes()
    ws = torch.zeros(ws_bytes, dtype=torch.uint8, device=_DEV)
    idx_ws, _ = _run_gvr2(
        logits, seq_lens, top_k, pre_idx, workspace={"gvr2_workspace": ws}
    )
    _check_varlen_rows(logits, idx_ws, n_r, top_k)
    got_d = torch.sort(
        torch.gather(logits, 1, idx_default.long().clamp_min(0)), dim=1
    ).values
    got_w = torch.sort(
        torch.gather(logits, 1, idx_ws.long().clamp_min(0)), dim=1
    ).values
    assert torch.equal(got_d, got_w)


# ---------------------------------------------------------------------------
# kernel-family admission parity (dispatch tiers) + oracle
# ---------------------------------------------------------------------------


def _assert_family(rows, msl_c, top_k, next_n, cr, want):
    """The varlen launcher must admit the same family the free route picks."""
    key = (rows, msl_c, top_k, msl_c, next_n, cr, _host._arch_token())
    lc = _host._VARLEN_CACHE.get(key)
    assert lc is not None, f"launcher not cached for {key}"
    assert lc[0] == want, f"family {lc[0]} != {want} for {key}"


@requires_gvr2
@pytest.mark.parametrize(
    "rows,msl_c,top_k,next_n,cr,family",
    [
        (8, 131072, 1024, 4, 4, "reg_clus"),  # clustered register-resident
        (8, 6144, 512, 1, 4, "reg"),  # register-resident
        (8, 3072, 512, 1, 4, "reg"),  # regimg flavor (cached as "reg")
        (64, 131072, 1024, 1, 1, "clus"),  # CS=2 cluster split
        (32, 131072, 1024, 1, 1, "clus"),  # CS=4 cluster split
        (304, 65536, 1024, 1, 1, "main"),  # BLK=256 wide-batch main
    ],
)
def test_gvr2_family_parity_and_oracle(rows, msl_c, top_k, next_n, cr, family):
    assert rows % next_n == 0
    batch = rows // next_n
    gen = torch.Generator(device=_DEV).manual_seed(rows + msl_c)
    kv = [
        int(torch.randint(1, msl_c * cr + 1, (1,), generator=gen, device=_DEV))
        for _ in range(batch)
    ]
    kv[0] = msl_c * cr  # pin the envelope so route sees the full width
    logits, seq_lens, pre_idx, n_r, _ = _make_varlen_case(
        kv, next_n, cr, top_k, seed=rows, msl_c=msl_c
    )
    indices, _ = _run_gvr2(
        logits, seq_lens, top_k, pre_idx, next_n=next_n, compress_ratio=cr
    )
    _assert_family(rows, msl_c, top_k, next_n, cr, family)
    _check_varlen_rows(logits, indices, n_r, top_k)


# ---------------------------------------------------------------------------
# CUDA graph capture / replay
# ---------------------------------------------------------------------------


@requires_gvr2
def test_gvr2_cuda_graph():
    """Warmup → capture → replay with in-place-growing seq_lens, including a
    row crossing the n<=k short-path boundary inside the graph."""
    batch, msl, top_k = 8, 131072, 1024
    gen = torch.Generator(device=_DEV).manual_seed(23)
    logits = torch.randn(batch, msl, generator=gen, device=_DEV) - 2.0
    kv0 = [100000, 65536, 40000, 8000, 2000, 1023, 1024, 131072]
    seq_lens = torch.tensor(kv0, dtype=torch.int32, device=_DEV)
    pre_idx = torch.randint(
        0, 1000, (batch, top_k), generator=gen, device=_DEV, dtype=torch.int32
    )
    out_i = torch.full((batch, top_k), -7, dtype=torch.int32, device=_DEV)

    # warmup on a side stream: JIT + workspace/launcher caches populate
    # outside capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run_gvr2(logits, seq_lens, top_k, pre_idx, out_indices=out_i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _run_gvr2(logits, seq_lens, top_k, pre_idx, out_indices=out_i)

    # Replays must survive BOTH directions of in-place length change (a
    # schedule valid only for growth is a classic replay hazard): grow for a
    # few steps, then shrink back below the short-path boundary.
    deltas = [0, 517, 1034, 1551, -400, -1200]
    for d in deltas:
        kv = [min(max(v + d, 0), msl) for v in kv0]
        seq_lens.copy_(torch.tensor(kv, dtype=torch.int32, device=_DEV))
        logits.copy_(torch.randn(batch, msl, generator=gen, device=_DEV) - 2.0)
        out_i.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        n_r = _row_valid_lens(kv, batch, 1, 1, msl)
        _check_varlen_rows(logits, out_i, n_r, top_k)


@requires_gvr2
def test_gvr2_warmup_then_capture():
    """warmup_varlen must pre-compile the launcher so capture works without an
    eager call at the captured geometry (row_stride pinned to the logits
    width, since FlashInfer's envelope is the logits row width)."""
    batch, msl, top_k = 4, 32768, 512
    _host.warmup_varlen(
        top_k, msl, compress_ratio=1, next_n=1, num_rows_list=(batch,), row_stride=msl
    )
    gen = torch.Generator(device=_DEV).manual_seed(29)
    logits = torch.randn(batch, msl, generator=gen, device=_DEV) - 2.0
    kv = [msl, 20000, 5000, 600]
    seq_lens = torch.tensor(kv, dtype=torch.int32, device=_DEV)
    pre_idx = torch.zeros(batch, top_k, dtype=torch.int32, device=_DEV)
    out_i = torch.empty(batch, top_k, dtype=torch.int32, device=_DEV)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _run_gvr2(logits, seq_lens, top_k, pre_idx, out_indices=out_i)
    g.replay()
    torch.cuda.synchronize()
    _check_varlen_rows(logits, out_i, _row_valid_lens(kv, batch, 1, 1, msl), top_k)


# ---------------------------------------------------------------------------
# backend registration / guards / cross-backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_next_n_row_relationship_validated_up_front():
    """The grouped next_n ABI (rows == batch * next_n, next_n >= 1) must be
    rejected by the public API's own validation for EVERY backend — kernels
    index seq_lens[row // next_n], so a mismatch reads the wrong request's
    length (or out of bounds) rather than failing loudly."""
    logits = torch.randn(4, 4096, dtype=torch.float32, device=_DEV)
    seq_lens = torch.full((4,), 4096, dtype=torch.int32, device=_DEV)  # != 4/2
    with pytest.raises(ValueError, match=r"seq_lens\.shape"):
        flashinfer.top_k_varlen(logits, seq_lens, 512, next_n=2)
    with pytest.raises(ValueError, match="next_n"):
        flashinfer.top_k_varlen(logits, seq_lens, 512, next_n=0)


@requires_gvr2
def test_gvr2_empty_batch():
    """B=0 must be a well-formed no-op through the public API (both the
    explicit backend and auto), returning a (0, top_k) result — not a crash
    in scheduling, compilation, or the heuristic."""
    logits = torch.empty(0, 8192, dtype=torch.float32, device=_DEV)
    seq_lens = torch.empty(0, dtype=torch.int32, device=_DEV)
    pre_idx = torch.empty(0, 512, dtype=torch.int32, device=_DEV)
    for backend in ("gvr_2", "auto"):
        indices, values = flashinfer.top_k_varlen(
            logits, seq_lens, 512, pre_idx=pre_idx, backend=backend
        )
        assert tuple(indices.shape) == (0, 512)
        assert values is None


@requires_gvr2
def test_gvr2_rejects_non_fp32():
    logits = torch.randn(4, 8192, dtype=torch.bfloat16, device=_DEV)
    seq_lens = torch.full((4,), 8192, dtype=torch.int32, device=_DEV)
    pre_idx = torch.zeros(4, 512, dtype=torch.int32, device=_DEV)
    with pytest.raises(Exception, match=r"not supported|Problem size"):
        flashinfer.top_k_varlen(logits, seq_lens, 512, pre_idx=pre_idx, backend="gvr_2")


@requires_gvr2
def test_gvr2_rejects_bad_top_k():
    logits = torch.randn(4, 8192, dtype=torch.float32, device=_DEV)
    seq_lens = torch.full((4,), 8192, dtype=torch.int32, device=_DEV)
    pre_idx = torch.zeros(4, 768, dtype=torch.int32, device=_DEV)
    with pytest.raises(Exception, match=r"not supported|Problem size"):
        flashinfer.top_k_varlen(logits, seq_lens, 768, pre_idx=pre_idx, backend="gvr_2")


@requires_gvr2
def test_gvr2_auto_selects_gvr2():
    """With fp32 + pre_idx, auto must pick gvr_2 first (it won 211/213
    measured cells; see the heuristic's docstring)."""
    logits = torch.randn(4, 8192, dtype=torch.float32, device=_DEV)
    seq_lens = torch.full((4,), 8192, dtype=torch.int32, device=_DEV)
    pre_idx = torch.zeros(4, 512, dtype=torch.int32, device=_DEV)
    pre_idx[:, 0] = logits.argmax(dim=-1).int()
    indices, _ = flashinfer.top_k_varlen(
        logits, seq_lens, 512, pre_idx=pre_idx, backend="auto"
    )
    order = flashinfer.top_k_varlen.suitable_auto_backends
    assert order[0] == "gvr_2"
    assert "gvr" in order
    _check_varlen_rows(logits, indices, [8192] * 4, 512)


@requires_gvr2
def test_gvr2_cross_backend_value_consistency():
    """Sorted selected-value multisets must agree across gvr_2, gvr, radix,
    and radix_cutlass on the same fp32 inputs."""
    top_k, n_valid = 512, 32768
    logits, seq_lens, pre_idx, _ = _make_case(4, n_valid, top_k, seed=3)
    logits = logits[:, :n_valid].contiguous()  # drop the poison pad
    seq_lens = torch.full((4,), n_valid, dtype=torch.int32, device=_DEV)
    vals = {}
    for backend in ("gvr_2", "gvr", "radix", "radix_cutlass"):
        idx, _ = flashinfer.top_k_varlen(
            logits, seq_lens, top_k, pre_idx=pre_idx, backend=backend
        )
        vals[backend] = torch.sort(
            torch.gather(logits, 1, idx.long()), dim=1, descending=True
        ).values
    for backend in ("gvr", "radix", "radix_cutlass"):
        assert torch.equal(vals["gvr_2"], vals[backend]), f"gvr_2 vs {backend}"


# ---------------------------------------------------------------------------
# adversarial value patterns (mass ties / inf floods / denormals)
# ---------------------------------------------------------------------------
# Random-logits tests cannot see tie-handling or extreme-value bugs. The
# upstream contract: finite inputs INCLUDING +/-inf and denormals are
# tie-aware exact; only NaN ordering is implementation-specific (untested).


def _adversarial_logits(pattern, rows, N, top_k, gen):
    if pattern == "constant":
        return torch.full((rows, N), 1.5, device=_DEV)
    if pattern == "two_values":
        x = torch.full((rows, N), -1.0, device=_DEV)
        pick = torch.rand(rows, N, generator=gen, device=_DEV) < 0.3
        x[pick] = 2.0
        return x
    if pattern == "huge_flood_gt_k":
        # near-FLT_MAX flood (3.1e38 > the kernel's 3e38 pad sentinel):
        # more huge values than slots. Literal +inf is NOT used here — the
        # upstream kernel drops +inf (see test_gvr2_plus_inf_upstream_caveat).
        x = torch.randn(rows, N, generator=gen, device=_DEV)
        n_huge = top_k + top_k // 2
        for r in range(rows):
            pos = torch.randperm(N, generator=gen, device=_DEV)[:n_huge]
            x[r, pos] = 3.1e38
        return x
    if pattern == "huge_flood_lt_k":
        x = torch.randn(rows, N, generator=gen, device=_DEV)
        for r in range(rows):
            pos = torch.randperm(N, generator=gen, device=_DEV)[: top_k // 4]
            x[r, pos] = 3.1e38
        x[:, 0] = float("-inf")  # a -inf below everything (supported)
        return x
    if pattern == "quantized4":
        lv = torch.tensor([-2.0, -0.5, 0.5, 2.0], device=_DEV)
        return lv[torch.randint(0, 4, (rows, N), generator=gen, device=_DEV)]
    if pattern == "denormal":
        # magnitudes straddling the subnormal range
        x = torch.randn(rows, N, generator=gen, device=_DEV) * 1e-40
        x[:, : top_k // 2] = 1e-38  # some normals on top
        return x
    raise ValueError(pattern)


@requires_gvr2
@pytest.mark.parametrize(
    "pattern",
    [
        "constant",
        "two_values",
        "huge_flood_gt_k",
        "huge_flood_lt_k",
        "quantized4",
        "denormal",
    ],
)
@pytest.mark.parametrize("n_valid,batch", [(32768, 4), (131072, 1)])
def test_gvr2_adversarial_patterns(pattern, n_valid, batch):
    top_k = 1024
    # Fixed per-pattern seeds (hash() is randomized per process; using it made
    # this test nondeterministic).
    seeds = {
        "constant": 101,
        "two_values": 202,
        "huge_flood_gt_k": 303,
        "huge_flood_lt_k": 404,
        "quantized4": 505,
        "denormal": 606,
    }
    gen = torch.Generator(device=_DEV).manual_seed(seeds[pattern])
    logits = _adversarial_logits(pattern, batch, n_valid, top_k, gen)
    seq_lens = torch.full((batch,), n_valid, dtype=torch.int32, device=_DEV)
    pre_idx = torch.randint(
        0, n_valid, (batch, top_k), generator=gen, device=_DEV, dtype=torch.int32
    )
    ref_vals = torch.topk(logits, top_k, dim=1).values
    indices, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
    _check_exact(logits, indices, n_valid, ref_vals)


@requires_gvr2
@pytest.mark.xfail(
    reason="UPSTREAM CAVEAT (TRT-LLM #17821 kernels, reproduced bit-identically "
    "by the upstream implementation on the same inputs): literal +inf logits "
    "are not selected in at least the clustered-register family. All finite "
    "values — including 3.1e38 > the 3e38 pad sentinel — and -inf are "
    "tie-aware exact (covered by test_gvr2_adversarial_patterns). Same class "
    "as the documented NaN implementation-specific ordering.",
    strict=False,  # family-dependent: some shapes handle +inf correctly
)
def test_gvr2_plus_inf_upstream_caveat():
    n_valid, batch, top_k = 32768, 4, 1024
    gen = torch.Generator(device=_DEV).manual_seed(1000)
    logits = torch.randn(batch, n_valid, generator=gen, device=_DEV)
    for r in range(batch):
        pos = torch.randperm(n_valid, generator=gen, device=_DEV)[: top_k // 4]
        logits[r, pos] = float("inf")
    seq_lens = torch.full((batch,), n_valid, dtype=torch.int32, device=_DEV)
    pre_idx = torch.randint(
        0, n_valid, (batch, top_k), generator=gen, device=_DEV, dtype=torch.int32
    )
    indices, _ = _run_gvr2(logits, seq_lens, top_k, pre_idx)
    got_inf = torch.isinf(logits.gather(1, indices.long().clamp_min(0)))
    assert int(got_inf.sum()) == batch * (top_k // 4), "+inf logits were dropped"


# ---------------------------------------------------------------------------
# route dispatch: pure-Python fuzz (CPU-only, no GPU required)
# ---------------------------------------------------------------------------


def test_gvr2_route_factorization_fuzz():
    """route_split(b,n,npad,k) must reproduce route() exactly — the lossless
    static/dynamic factorization the varlen engine's per-row re-derivation
    relies on."""
    checked = 0
    npad = 1 << 20
    for k in (512, 1024, 2048):
        bases = (
            2 * k,
            3 * k,
            4 * k + 64,
            2560,
            4096,
            8192,
            16384,
            4 * 1024,
            4 * 4096,
            4 * 32768,
            65536,
            131072,
            262144,
        )
        for b in (1, 8, 16, 64, 148, 296, 1024):
            ns = set()
            for base in bases:
                for d in range(-4, 5):
                    n = base + d
                    if k < n <= npad:
                        ns.add(n)
            ns.update(range(k + 1, npad, 4999))
            s = (b * 2654435761 + k) % (1 << 31)
            for _ in range(400):
                s = (s * 1103515245 + 12345) % (1 << 31)
                ns.add(k + 1 + s % (npad - k - 1))
            for n in sorted(ns):
                assert _host.route_split(b, n, npad, k) == _host.route(b, n, npad, k), (
                    b,
                    n,
                    npad,
                    k,
                )
                checked += 1
    assert checked > 10_000


def test_gvr2_route_bands_contiguous():
    bands = _host.route_bands(8, 262144, 1024)
    lo_expect = 1024 + 1
    for lo, hi, st in bands:
        assert lo == lo_expect, (lo, lo_expect)
        assert hi >= lo
        lo_expect = hi + 1
        for f in _host._DYN_RT[st["kernel"]]:
            assert f not in st["rt"], (st["kernel"], f)
    assert lo_expect == 262144 + 1


def test_gvr2_route_pure_and_total():
    for k in (512, 1024, 2048):
        for b in (1, 4, 16, 64, 148, 296, 512, 1024, 4096, 8192):
            for n in (k + 1, 4096, 65536, 262144, 1 << 20):
                if n <= k:
                    continue
                plan = _host.route(b, n, (n + 63) // 64 * 64, k)
                assert plan["kernel"] in ("main", "reg", "regimg", "clus", "reg_clus")
                assert plan["block"] >= 128
                assert plan["grid"][0] >= 1
