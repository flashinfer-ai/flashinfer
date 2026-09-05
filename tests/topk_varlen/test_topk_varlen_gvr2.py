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
        (8, 6144, 512, 1, 4, "reg"),  # register-resident (local VPT=2 rung)
        (8, 8192, 512, 1, 1, "reg"),  # local VPT=2 rung, top of its band
        (256, 8192, 512, 1, 1, "reg"),  # local b > 148 BLK=512 rung (upstream: main)
        (8, 12288, 512, 1, 1, "reg"),  # upstream VPT=4 rung, unchanged
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
def test_gvr2_auto_selects_gvr2_hint_free():
    """fp32 WITHOUT pre_idx: auto still picks gvr_2 (hint-free it beats every
    radix backend 1.5-5x on the measured grid) and never admits gvr (V1
    needs the hint); the one carved-out cell (K >= 2048, N <= 4096, one row)
    ranks radix_filter first where it is available, gvr_2 right behind, and a
    real hint puts gvr_2 back in front."""
    logits = torch.randn(4, 8192, dtype=torch.float32, device=_DEV)
    seq_lens = torch.full((4,), 8192, dtype=torch.int32, device=_DEV)
    indices, _ = flashinfer.top_k_varlen(logits, seq_lens, 512, backend="auto")
    order = flashinfer.top_k_varlen.suitable_auto_backends
    assert order[0] == "gvr_2" and "gvr" not in order
    _check_varlen_rows(logits, indices, [8192] * 4, 512)
    tiny = torch.randn(1, 4096, dtype=torch.float32, device=_DEV)
    one = torch.full((1,), 4096, dtype=torch.int32, device=_DEV)
    indices, _ = flashinfer.top_k_varlen(tiny, one, 2048, backend="auto")
    order = flashinfer.top_k_varlen.suitable_auto_backends
    if "radix_filter" in order:
        assert order[:2] == ["radix_filter", "gvr_2"]
    else:
        assert order[0] == "gvr_2"
    _check_varlen_rows(tiny, indices, [4096], 2048)
    hint = torch.zeros(1, 2048, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(tiny, one, 2048, pre_idx=hint, backend="auto")
    assert flashinfer.top_k_varlen.suitable_auto_backends[0] == "gvr_2"


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
        # more huge values than slots. Literal +inf is covered separately by
        # test_gvr2_posinf_completeness / test_gvr2_posinf_reg_clus.
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


def _check_complete_exact(out, logits, n_valid, ref_vals, sentinel):
    """Every output slot written (no sentinel left), then tie-aware exact."""
    torch.cuda.synchronize()
    assert int((out == sentinel).sum()) == 0, "unwritten output slots"
    _check_exact(logits, out, n_valid, ref_vals)


@requires_gvr2
@pytest.mark.parametrize("n_valid", [3072, 4096], ids=["n3072", "n4096"])
def test_gvr2_high_anchor_hint_completeness(n_valid):
    """Port of TRT-LLM PR #18501's first regression (register family): anchor-only
    hints whose gathered values all sit ABOVE the true k-th value -- an argmax
    anchor over the all-zero cold-start hint buffer, with row[0] = second-max so
    the hint-derived bracket holds exactly two entries -- make the classify
    histogram total fall short of k. The kernel must then escape to the
    key-space ranking instead of stopping at the histogram total (pre-fix:
    out[tot:k) left unwritten, 130,304 of 131,072 slots per cell here). The
    shapes pin the vulnerable non-BRL reg variant (BRL classify clamps
    out-of-bracket values into bin 0 and cannot under-count)."""
    top_k, bs = 512, 256
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    logits = torch.randn((bs, n_valid), generator=gen, dtype=torch.float32, device=_DEV)
    logits[:, 0] = torch.topk(logits, 2, dim=1).values[:, 1]  # bracket = [2nd max, max]
    ref_vals = torch.topk(logits, top_k, dim=1).values
    pre_idx = torch.zeros((bs, top_k), dtype=torch.int32, device=_DEV)
    pre_idx[:, 0] = logits.argmax(dim=1).to(torch.int32)
    seq_lens = torch.full((bs,), n_valid, dtype=torch.int32, device=_DEV)
    out = torch.full((bs, top_k), -7, dtype=torch.int32, device=_DEV)
    _run_gvr2(logits, seq_lens, top_k, pre_idx, out_indices=out)
    _check_complete_exact(out, logits, n_valid, ref_vals, -7)


@requires_gvr2
def test_gvr2_neginf_tail_completeness():
    """Port of TRT-LLM PR #18501's second regression: an in-window -inf in the
    row's tail column (n_valid % 4 == 1, so that column lives outside the
    float4 register batch) drags the hint-free DEG bracket to -inf, every
    classify product becomes NaN and the histogram total is zero (pre-fix:
    whole rows unwritten). Odd rows keep fewer than top_k finite entries so the
    -inf tie class exercises the escape's fill-lane bound (pre-fix: duplicate
    indices from the -inf fill lanes of the last partial float4)."""
    top_k, bs, npad, n_valid = 1024, 256, 4096, 4093
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    logits = torch.randn((bs, npad), generator=gen, dtype=torch.float32, device=_DEV)
    logits[:, n_valid:] = 3e38  # poison past the window
    logits[:, n_valid - 1] = float("-inf")  # in-window -inf in the tail column
    logits[1::2, 500:n_valid] = float("-inf")  # odd rows: n_finite < top_k
    masked = logits.clone()
    masked[:, n_valid:] = float("-inf")
    ref_vals = torch.topk(masked, top_k, dim=1).values
    pre_idx = torch.zeros((bs, top_k), dtype=torch.int32, device=_DEV)
    seq_lens = torch.full((bs,), n_valid, dtype=torch.int32, device=_DEV)
    out = torch.full((bs, top_k), -7, dtype=torch.int32, device=_DEV)
    _run_gvr2(logits, seq_lens, top_k, pre_idx, out_indices=out)
    _check_complete_exact(out, logits, n_valid, ref_vals, -7)


@requires_gvr2
@pytest.mark.parametrize(
    "n_valid,top_k,positions",
    [
        (4096, 1024, [1000]),  # in the register fold window (DKG issue #58 case A)
        (4096, 1024, [3000]),  # outside the window
        (4096, 512, [1000]),  # regimg flavor
        (4096, 2048, [1000]),  # plain reg (no img)
        (
            4096,
            1024,
            list(range(0, 4096, 4))[:1029],
        ),  # more +inf than K: every slot +inf
    ],
    ids=["in_window", "out_of_window", "k512", "k2048", "ties_gt_k"],
)
def test_gvr2_posinf_completeness(n_valid, top_k, positions):
    """Port of TRT-LLM PR #18625's regression (register family, DKG issue #58):
    a +inf inside the row drives the hint-free bracket maximum to +inf, the
    bracket width becomes infinite, the classify scale rcp(width) becomes 0
    and every value folds into bin 0, so the whole-bin emit dropped the +inf.
    The infinite width must now fail the collapse guard and take the
    key-space escape, where fkey(+inf) is the maximum key. In-window and
    out-of-window +inf, the regimg / plain-reg flavors and a more-than-K-ties
    row are pinned; N=4096 keeps the register families."""
    gen = torch.Generator(device=_DEV).manual_seed(top_k + n_valid)
    logits = torch.randn((1, n_valid), generator=gen, device=_DEV) * 2.0
    logits[0, positions] = float("inf")
    seq_lens = torch.full((1,), n_valid, dtype=torch.int32, device=_DEV)
    ref_vals = torch.topk(logits, top_k, dim=1).values
    for pre_idx in (
        torch.topk(logits, top_k, dim=1).indices.int().contiguous(),
        torch.full((1, top_k), -1, dtype=torch.int32, device=_DEV),
    ):
        out = torch.full((1, top_k), -7, dtype=torch.int32, device=_DEV)
        _run_gvr2(logits, seq_lens, top_k, pre_idx, out_indices=out)
        torch.cuda.synchronize()
        assert int((out == -7).sum()) == 0, "unwritten output slots"
        n_inf = int(torch.isinf(logits[0][out[0].long()]).sum())
        assert n_inf == min(top_k, len(positions)), "+inf dropped from the top-k"
        _check_exact(logits, out, n_valid, ref_vals)


@requires_gvr2
@pytest.mark.parametrize(
    "batch,n_valid,mode",
    [
        (4, 32768, "in_sample"),  # +inf among the first K columns (hint-free sample)
        (4, 32768, "out_of_sample"),  # +inf at column 30000
        (4, 65536, "in_sample"),  # crossing bin > CS*CMPC: degen path already exact
        (4, 32768, "quarter_random"),  # K/4 random +inf per row, random hint
    ],
    ids=["in_sample_32k", "out_of_sample_32k", "in_sample_64k", "quarter_random_32k"],
)
def test_gvr2_posinf_reg_clus(batch, n_valid, mode):
    """Port of TRT-LLM PR #18625's second commit (clustered-register family,
    DKG issue #58): a sampled +inf made the reg_clus bracket width infinite,
    the pre-fix guard accepted it, the classify scale became 0, every finite
    value landed in bin 1 and the +inf became NaN in the trash bin and was
    never staged, so the output was the true top-K with the +inf replaced by
    the (K+1)-th value (all slots valid). Rows above CS*CMPC = 32K columns took
    the whole-row ``degen`` path and were already exact; the fix routes an
    infinite-width bracket there too. With a hint the +inf is in the sample
    (oracle) or the sentinel bracket is installed (-1), so every variant must
    now be exact for both hints. 4 x 32K, K=1024 is the reg_clus route."""
    top_k = 1024
    gen = torch.Generator(device=_DEV).manual_seed(n_valid + batch)
    logits = torch.randn(batch, n_valid, generator=gen, device=_DEV) * 2.0
    if mode == "quarter_random":
        for r in range(batch):
            pos = torch.randperm(n_valid, generator=gen, device=_DEV)[: top_k // 4]
            logits[r, pos] = float("inf")
    else:
        logits[:, 1000 if mode == "in_sample" else 30000] = float("inf")
    n_inf = int(torch.isinf(logits[0]).sum())
    seq_lens = torch.full((batch,), n_valid, dtype=torch.int32, device=_DEV)
    ref = torch.sort(
        torch.topk(logits, top_k, dim=1).values, dim=1, descending=True
    ).values
    hints = [
        torch.topk(logits, top_k, dim=1).indices.int().contiguous(),
        torch.full((batch, top_k), -1, dtype=torch.int32, device=_DEV),
    ]
    if mode == "quarter_random":
        hints.append(
            torch.randint(
                0,
                n_valid,
                (batch, top_k),
                generator=gen,
                device=_DEV,
                dtype=torch.int32,
            )
        )
    for pre_idx in hints:
        out = torch.full((batch, top_k), -7, dtype=torch.int32, device=_DEV)
        _run_gvr2(logits, seq_lens, top_k, pre_idx, out_indices=out)
        torch.cuda.synchronize()
        assert int((out == -7).sum()) == 0, "unwritten output slots"
        assert bool(((out >= 0) & (out < n_valid)).all()), "index out of range"
        for r in range(batch):
            assert torch.unique(out[r]).numel() == top_k, (
                f"duplicate indices in row {r}"
            )
        sel = logits.gather(1, out.long())
        assert torch.equal(
            torch.isinf(sel).sum(dim=1),
            torch.full((batch,), min(top_k, n_inf), device=_DEV),
        ), "+inf dropped from the top-k"
        got = torch.sort(sel, dim=1, descending=True).values
        assert torch.equal(got, ref), "selected value multiset differs from torch.topk"


@requires_gvr2
@pytest.mark.parametrize(
    "kv", [200, 256, 100000], ids=["in_range", "equal_width", "oversized"]
)
def test_gvr2_width_below_k(kv):
    """Logits narrower than top_k + 1: every row is short and must come back
    as identity + -1 tail whatever seq_lens says. The launcher used to hand
    the register families max(N, top_k + 1) as the row-read bound, so an
    oversized kv length made them read top_k + 1 elements at a 256-element
    row stride (into the next row, then past the tensor) and rank the
    garbage; row 2 is poisoned so a cross-row read from row 1 shows up."""
    rows, n, top_k = 3, 256, 512
    gen = torch.Generator(device=_DEV).manual_seed(7)
    logits = torch.randn(rows, n, generator=gen, device=_DEV)
    logits[2] = 3e38
    seq_lens = torch.full((rows,), kv, dtype=torch.int32, device=_DEV)
    pre_idx = torch.full((rows, top_k), -1, dtype=torch.int32, device=_DEV)
    indices, values = _run_gvr2(logits, seq_lens, top_k, pre_idx, return_values=True)
    _check_varlen_rows(logits, indices, [min(kv, n)] * rows, top_k, values=values)


@requires_gvr2
def test_gvr2_sequential_warmup_populates_exact_launchers():
    """warmup_varlen keyed its completion on the per-engine representative
    rows, so a second call adding a row count that maps to an already-warmed
    engine returned before that row count's launcher existed, and a CUDA-graph
    capture at that row count failed with the launcher-cache-miss error."""
    top_k, msl = 512, 8192
    npad = _round64(msl)
    rows = 61  # a row count no other test warms, so the precondition below holds

    def has_launcher(r):
        return any(
            k[0] == r and k[1] == npad and k[2] == top_k for k in _host._VARLEN_CACHE
        )

    _host.warmup_varlen(top_k, msl, num_rows_list=[64])
    assert not has_launcher(rows), (
        "precondition: the 61-row launcher must not exist yet"
    )
    _host.warmup_varlen(top_k, msl, num_rows_list=[rows, 64])
    assert has_launcher(rows), "61-row launcher missing after sequential warmup"
    logits = torch.zeros(rows, npad, device=_DEV)
    kv_lens = torch.full((rows,), msl, dtype=torch.int32, device=_DEV)
    pre_idx = torch.zeros(rows, top_k, dtype=torch.int32, device=_DEV)
    out = torch.empty(rows, top_k, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            _host.run_varlen(logits, pre_idx, kv_lens, out, max_seq_len=msl)
    torch.cuda.synchronize()


@requires_gvr2
def test_gvr2_rejects_overlapping_rows():
    """stride(0) < shape[1] (an expand view) is not a row-major layout the
    kernel can address; the checker used to derive npad = 0 from it and
    auto ranked gvr_2 first for the call."""
    from flashinfer.topk_varlen import topk_varlen as tv
    from flashinfer.utils import BackendSupportedError

    base = torch.randn(1, 8192, device=_DEV)
    x = base.expand(2, -1)
    seq_lens = torch.full((2,), 8192, dtype=torch.int32, device=_DEV)
    pre_idx = torch.full((2, 1024), -1, dtype=torch.int32, device=_DEV)
    assert not tv._gvr2_top_k_varlen_check(x, seq_lens, 1024, pre_idx=pre_idx)
    # a failed explicit-backend check surfaces as the decorator's
    # ValueError("Problem size is not supported ...")
    with pytest.raises((BackendSupportedError, ValueError), match="not supported"):
        _run_gvr2(x, seq_lens, 1024, pre_idx)
    out = torch.empty(2, 1024, dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="overlap"):
        _host.run_varlen(x, pre_idx, seq_lens, out, max_seq_len=8192)
    # auto skips gvr_2 and radix (both gated) and lands on a backend that
    # handles the view (gvr or radix_filter): the result is exact
    indices, _ = flashinfer.top_k_varlen(x, seq_lens, 1024, pre_idx=pre_idx)
    ref = torch.sort(torch.topk(base[0], 1024).values).values
    for r in range(2):
        assert torch.equal(torch.sort(base[0][indices[r].long()]).values, ref)


@requires_gvr2
def test_gvr2_workspace_must_be_16_byte_aligned():
    """The compiled workspace ABI assumes 16-byte alignment; the host check
    used to accept 8 and let the DSL refuse the launch."""
    logits = torch.zeros(1, 4096, device=_DEV)
    buf = torch.zeros(_host.WS_BYTES + 16, dtype=torch.uint8, device=_DEV)
    _host.validate_run_ws(buf[16 : 16 + _host.WS_BYTES], logits)
    with pytest.raises(RuntimeError, match="16-byte"):
        _host.validate_run_ws(buf[8 : 8 + _host.WS_BYTES], logits)
    # pre_idx / indices declare the same 16-byte ABI: a shifted contiguous view
    # is refused by the host with a FlashInfer message (was a DSL error)
    seq_lens = torch.full((1,), 4096, dtype=torch.int32, device=_DEV)
    good = torch.zeros(1, 1024, dtype=torch.int32, device=_DEV)
    shifted = torch.zeros(1024 + 4, dtype=torch.int32, device=_DEV)[1:1025].view(
        1, 1024
    )
    with pytest.raises(RuntimeError, match="16-byte"):
        _host.run_varlen(logits, shifted, seq_lens, good.clone(), max_seq_len=4096)
    with pytest.raises(RuntimeError, match="16-byte"):
        _host.run_varlen(logits, good, seq_lens, shifted, max_seq_len=4096)


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


# ---------------------------------------------------------------------------
# hint-free gvr_2 (FlashInfer-local: pre_idx=None runs on a cached arange
# anchor) + the FlashInfer-local route() rungs in the 4K < n <= 8K band
# ---------------------------------------------------------------------------


@requires_gvr2
@pytest.mark.parametrize(
    "kv,next_n,cr,top_k",
    [
        ([8192, 5000, 300, 1], 1, 1, 512),  # reg VPT=2 rung + short rows
        ([8192 * 4, 6000 * 4, 2048 * 4], 2, 4, 512),  # MTP rows, cr=4
        ([65536, 40000, 1000], 1, 1, 1024),  # reg_clus band
        ([262144, 100000, 2047], 1, 1, 2048),  # main/clus band, K=2048
        ([4096] * 200 + [100] * 56, 1, 1, 512),  # b > 148 BLK=512 rung
    ],
)
def test_gvr2_hint_free_exact(kv, next_n, cr, top_k):
    """pre_idx=None is exact on every family (the hint only steers sampling);
    short rows take the identity path, tails are -1, poisoned padding is
    never read."""
    logits, seq_lens, _, n_r, _ = _make_varlen_case(kv, next_n, cr, top_k, seed=7)
    indices, values = _run_gvr2(
        logits,
        seq_lens,
        top_k,
        None,
        next_n=next_n,
        compress_ratio=cr,
        return_values=True,
    )
    _check_varlen_rows(logits, indices, n_r, top_k, values)


@requires_gvr2
def test_gvr2_hint_free_cuda_graph_replay():
    """Hint-free calls replay under CUDA graphs: the arange table is a stable
    per-(device, k) address, growth happens only eagerly (refused under
    capture), and seq_lens contents may change between replays."""
    from flashinfer.topk_varlen.kernels import gvr2_topk_host as _host

    top_k, n = 512, 8192
    kv_a = [8192, 4096, 700, 8192]
    logits, seq_lens, _, n_r_a, _ = _make_varlen_case(
        kv_a, 1, 1, top_k, seed=3, msl_c=n
    )
    out = torch.empty(len(kv_a), top_k, dtype=torch.int32, device=_DEV)
    # eager call: compiles the launcher and sizes the table
    _run_gvr2(logits, seq_lens, top_k, None, out_indices=out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        _run_gvr2(logits, seq_lens, top_k, None, out_indices=out)
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    _check_varlen_rows(logits, out, n_r_a, top_k)
    # grow / shrink rows between replays
    kv_b = [3000, 8192, 8192, 100]
    seq_lens.copy_(torch.tensor(kv_b, dtype=torch.int32, device=_DEV))
    n_r_b = _row_valid_lens(kv_b, len(kv_b), 1, 1, n)
    for r in range(len(kv_b)):
        logits[r, n_r_b[r] :] = 3e38
        logits[r, : n_r_b[r]] = torch.randn(n_r_b[r], device=_DEV) - 2.0
    g.replay()
    torch.cuda.synchronize()
    _check_varlen_rows(logits, out, n_r_b, top_k)
    # table growth under capture is refused: warm the launcher at a larger
    # batch WITH a hint (table untouched), then capture hint-free
    big = (
        4 * len(kv_a) + _host._HINT_FREE[(torch.cuda.current_device(), top_k)].shape[0]
    )
    logits_b = torch.randn(big, n, device=_DEV)
    seq_b = torch.full((big,), n, dtype=torch.int32, device=_DEV)
    hint = torch.zeros(big, top_k, dtype=torch.int32, device=_DEV)
    out_b = torch.empty(big, top_k, dtype=torch.int32, device=_DEV)
    _run_gvr2(logits_b, seq_b, top_k, hint, out_indices=out_b)
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with (
        pytest.raises(RuntimeError, match="hint-free gvr_2"),
        torch.cuda.stream(s),
        torch.cuda.graph(g2, stream=s),
    ):
        _run_gvr2(logits_b, seq_b, top_k, None, out_indices=out_b)
    torch.cuda.synchronize()
    # warmup_varlen sizes the table for its largest batch, so the same
    # capture then succeeds
    _host.warmup_varlen(top_k, n, num_rows_list=(big,))
    g3 = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g3, stream=s):
        _run_gvr2(logits_b, seq_b, top_k, None, out_indices=out_b)
    torch.cuda.current_stream().wait_stream(s)
    g3.replay()
    torch.cuda.synchronize()
    _check_varlen_rows(logits_b, out_b, [n] * big, top_k)


@requires_gvr2
def test_gvr2_hint_free_host_requires_top_k():
    from flashinfer.topk_varlen.kernels import gvr2_topk_host as _host

    logits = torch.randn(2, 4096, device=_DEV)
    kv = torch.full((2,), 4096, dtype=torch.int32, device=_DEV)
    out = torch.empty(2, 512, dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="top_k is required"):
        _host.run_varlen(logits, None, kv, out)
    hint = torch.zeros(2, 1024, dtype=torch.int32, device=_DEV)
    with pytest.raises(RuntimeError, match="top_k=512 != pre_idx"):
        _host.run_varlen(logits, hint, kv, out, top_k=512)


def test_gvr2_route_local_rungs():
    """The two FlashInfer-local rungs (4K < n <= 8K): VPT=2 for b <= 148,
    BLK=512/VPT=4/MINB=2 for 148 < b <= 296 (one wave) — with NBH == IMGOFF
    (asserted at launch) — the main slab beyond one wave, and the upstream
    VPT=4 rung untouched above 8K."""
    from flashinfer.topk_varlen.kernels import gvr2_topk_host as _host

    for k in (512, 1024, 2048):
        for n in (4100, 6144, 8192):
            p = _host.route(8, n, _round64(n), k)
            assert p["kernel"] == "reg" and p["tpl"][:3] == (1024, 2, 1), (
                n,
                k,
                p["tpl"],
            )
            assert p["rt"]["IMGOFF"] == p["tpl"][7]
            p = _host.route(256, n, _round64(n), k)
            assert p["kernel"] == "reg" and p["tpl"][:3] == (512, 4, 2), (
                n,
                k,
                p["tpl"],
            )
            assert p["rt"]["IMGOFF"] == p["tpl"][7] == _host.NB
            assert _host.route(296, n, _round64(n), k)["tpl"][:3] == (512, 4, 2)
            assert _host.route(297, n, _round64(n), k)["kernel"] == "main"
            assert _host.route(512, n, _round64(n), k)["kernel"] == "main"
        p = _host.route(8, 12288, 12288, k)
        assert p["kernel"] == "reg" and p["tpl"][:3] == (1024, 4, 1)
        p = _host.route(256, 12288, 12288, k)
        assert p["kernel"] == "main"
    # the lossless static/dynamic factorization must still hold on the new rungs
    for b, n in ((8, 8192), (256, 8192), (149, 5000), (148, 5000)):
        assert _host.route_split(b, n, _round64(n), 512) == _host.route(
            b, n, _round64(n), 512
        )
