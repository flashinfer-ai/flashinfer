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
"""Windowed (prefill) ``top_k_varlen``: per-row ``[row_starts[r], row_starts[r] +
seq_lens[r])`` candidate windows with window-local output indices, served by the
gvr_2 prefill engines (TRT-LLM #18702 port).

Contract under test: exact top-k of the window (tie-aware), ``-1`` pad, identity
for windows shorter than K, NO read outside the window (out-of-window cells are
poisoned with NaN / +-inf / 3e38 and the <= 3 lead lanes before a misaligned
``ks`` hold +inf so a leak becomes top-1), ``-inf`` masks inside the window,
CUDA-graph replay after warm-up, and the API/backends' admission rules.
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
    not (_FLASHINFER_AVAILABLE and torch.cuda.is_available()),
    reason="flashinfer + CUDA required",
)
_DEV = "cuda"


def _cc() -> int:
    major, minor = get_compute_capability(torch.device(_DEV))
    return major * 10 + minor


requires_gvr2 = pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE
    or not torch.cuda.is_available()
    or not flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()),
    reason="gvr_2 unsupported on this device",
)


def _check_windowed(logits, got, ks, lens, top_k):
    """Tie-aware exactness per row: trailing -1 pad from the window length,
    unique in-window head, identity for short windows, exact index set when
    the K-th value is unique else strictly-above set + boundary-tie count."""
    assert got.shape == (logits.shape[0], top_k) and got.dtype == torch.int32
    ksl, lel = ks.tolist(), lens.tolist()
    got64 = got.to(torch.int64)
    for r in range(logits.shape[0]):
        nv = max(min(ksl[r] + lel[r], logits.shape[1]) - ksl[r], 0)
        m = min(nv, top_k)
        row = got64[r]
        assert bool((row[m:] == -1).all()), (
            f"row {r}: pad must be trailing -1 x{top_k - m}"
        )
        head = row[:m]
        if m == 0:
            continue
        assert bool((head != -1).all()), f"row {r}: -1 inside the valid head"
        assert int(head.min()) >= 0 and int(head.max()) < nv, (
            f"row {r}: index outside [0,{nv})"
        )
        assert int(torch.unique(head).numel()) == m, f"row {r}: duplicate indices"
        win = logits[r, ksl[r] : ksl[r] + nv]
        if nv <= top_k:
            assert torch.equal(
                torch.sort(head).values, torch.arange(nv, device=logits.device)
            ), f"row {r}: short window must be identity"
            continue
        vals = torch.sort(win, descending=True).values
        v_k, v_next = vals[top_k - 1], vals[top_k]
        got_vals = win[head]
        if bool(v_k != v_next):
            ref = (win >= v_k).nonzero(as_tuple=True)[0]
            assert ref.numel() == top_k
            assert torch.equal(torch.sort(head).values, ref), (
                f"row {r}: index set mismatch"
            )
        else:
            above = (win > v_k).nonzero(as_tuple=True)[0]
            got_above = head[got_vals > v_k]
            assert torch.equal(torch.sort(got_above).values, above), (
                f"row {r}: strictly-above set mismatch"
            )
            assert int((got_vals == v_k).sum()) == top_k - above.numel(), (
                f"row {r}: wrong number of boundary-tied picks"
            )
            assert bool((got_vals >= v_k).all()), f"row {r}: value below k-th selected"


def _make_case(rows, ncols, ks_list, len_list, *, seed, dist="randn"):
    """DeepGEMM-like storage (stride = align(ncols + 256, 256), column slice
    [:, :ncols]); every out-of-window cell poisoned (rotating NaN/inf/3e38/-inf)
    and the <= 3 lead lanes before each ks set to +inf: an over-read or a frame
    bug would surface as a wrong top-1 or a negative index."""
    gen = torch.Generator(device=_DEV).manual_seed(seed)
    stride = ((ncols + 256 + 255) // 256) * 256
    if dist == "randn":
        full = torch.randn(
            (rows, stride), generator=gen, dtype=torch.float32, device=_DEV
        )
    elif dist == "equal":
        full = torch.ones((rows, stride), dtype=torch.float32, device=_DEV)
    elif dist == "twoval":
        full = torch.randint(0, 2, (rows, stride), generator=gen, device=_DEV).float()
    else:
        raise ValueError(dist)
    logits = full[:, :ncols]
    ks = torch.tensor(ks_list, dtype=torch.int32, device=_DEV)
    lens = torch.tensor(len_list, dtype=torch.int32, device=_DEV)
    cols = torch.arange(stride, device=_DEV).unsqueeze(0)
    outside = (cols < ks.unsqueeze(1)) | (cols >= (ks + lens).unsqueeze(1))
    pat = torch.tensor([float("nan"), float("inf"), 3e38, float("-inf")], device=_DEV)[
        cols % 4
    ].expand(rows, -1)
    full.masked_scatter_(outside, pat[outside])
    for r, s in enumerate(ks_list):
        full[r, max(s - 3, 0) : s] = float("inf")
    return logits, ks, lens


def _run(logits, ks, lens, top_k, **kw):
    out, _ = flashinfer.top_k_varlen(
        logits, lens, top_k, row_starts=ks, backend=kw.pop("backend", "gvr_2"), **kw
    )
    torch.cuda.synchronize()
    return out


@requires_gvr2
@pytest.mark.parametrize("top_k", [512, 1024, 2048], ids=lambda k: f"k{k}")
def test_prefill_causal_ramp_single_request(top_k):
    """One request (ks = 0), causal windows 1..rows: rows shorter than K are
    identity, the rest exact."""
    rows = 148 + top_k // 4
    ks = [0] * rows
    lens = list(range(1, rows + 1))
    lg, rs, le = _make_case(rows, rows, ks, lens, seed=top_k)
    _check_windowed(lg, _run(lg, rs, le, top_k), rs, le, top_k)


@requires_gvr2
@pytest.mark.parametrize("lead", [1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_prefill_packed_misaligned_ks(top_k, lead):
    """Two packed requests; the second starts at ks % 4 == lead with +inf
    poison at [ks-lead, ks): a leaked lead lane would become top-1 and a missed
    frame correction a negative index."""
    a = 300
    ks1 = ((a + 3) // 4) * 4 + lead
    n1 = 4096 + 17
    rows = a + n1
    ncols = ks1 + n1
    ks = [0] * a + [ks1] * n1
    lens = list(range(1, a + 1)) + list(range(1, n1 + 1))
    lg, rs, le = _make_case(rows, ncols, ks, lens, seed=top_k * 100 + lead)
    out = _run(lg, rs, le, top_k)
    assert int(out.min()) >= -1, (
        "negative index leaked (missed -lead correction / guard)"
    )
    _check_windowed(lg, out, rs, le, top_k)


@requires_gvr2
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_short_windows(top_k):
    """Window lengths {0, 1, k-1, k, k+1}: identity 0..nv-1 + trailing -1;
    an empty window is all -1."""
    for nv in (0, 1, top_k - 1, top_k, top_k + 1):
        rows = 4
        ncols = max(nv, 1) + 8
        lg, rs, le = _make_case(rows, ncols, [0] * rows, [nv] * rows, seed=nv + top_k)
        _check_windowed(lg, _run(lg, rs, le, top_k), rs, le, top_k)


@requires_gvr2
@pytest.mark.parametrize("dist", ["equal", "twoval"], ids=lambda d: d)
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_ties_degenerate(top_k, dist):
    """All-equal and two-valued windows drive the degenerate narrowing paths."""
    rows, n = 16, 4096
    lg, rs, le = _make_case(rows, n, [0] * rows, [n] * rows, seed=top_k, dist=dist)
    _check_windowed(lg, _run(lg, rs, le, top_k), rs, le, top_k)


@requires_gvr2
@pytest.mark.parametrize("lead", [1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_neginf_masks_in_window(top_k, lead):
    """Fewer than K finite values in the window (the rest -inf, as SGLang's
    init/local-token masks): the K-th boundary lies in the -inf tie class,
    crossed with a misaligned lead; no negative index may leak."""
    n_finite = top_k - 100
    nv = top_k + 400
    ks1 = ((37 + 3) // 4) * 4 + lead
    ncols = ks1 + nv
    rows = 5
    gen = torch.Generator(device=_DEV).manual_seed(top_k * 10 + lead)
    stride = ((ncols + 256 + 255) // 256) * 256
    full = torch.full((rows, stride), float("-inf"), dtype=torch.float32, device=_DEV)
    for r in range(rows):
        full[r, ks1 : ks1 + n_finite] = torch.randn(
            n_finite, generator=gen, device=_DEV
        )
        full[r, :ks1] = float("inf")
        full[r, ks1 + nv :] = 3e38
    lg = full[:, :ncols]
    rs = torch.tensor([ks1] * rows, dtype=torch.int32, device=_DEV)
    le = torch.tensor([nv] * rows, dtype=torch.int32, device=_DEV)
    out = _run(lg, rs, le, top_k)
    assert int(out.min()) >= -1, "negative index leaked in a -inf tie class"
    _check_windowed(lg, out, rs, le, top_k)


@requires_gvr2
def test_prefill_window_end_clamps_to_width():
    """ks + len beyond the logits width clamps to the width (arena rule), and
    a window that starts at or past the width is empty (all -1)."""
    top_k, n = 512, 3000
    rows = 3
    lg, rs, le = _make_case(rows, n, [0, 1000, n], [n, n, 5], seed=3)
    # row 1: window [1000, 4000) clamps to [1000, 3000); row 2: empty
    lg[1, 1000:n] = torch.randn(n - 1000, device=_DEV)
    out = _run(lg, rs, le, top_k)
    le_eff = torch.tensor([n, n - 1000, 0], dtype=torch.int32, device=_DEV)
    _check_windowed(lg, out, rs, le_eff, top_k)


@requires_gvr2
def test_prefill_return_values_are_window_values():
    top_k = 512
    rows, ncols = 64, 4096
    ks = [(r * 37) % 2048 for r in range(rows)]
    lens = [min(ncols - s, 700 + r * 20) for r, s in enumerate(ks)]
    lg, rs, le = _make_case(rows, ncols, ks, lens, seed=5)
    out, vals = flashinfer.top_k_varlen(
        lg, le, top_k, row_starts=rs, return_values=True, backend="gvr_2"
    )
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)
    for r in range(rows):
        m = min(lens[r], top_k)
        win = lg[r, ks[r] : ks[r] + lens[r]]
        assert torch.equal(vals[r, :m], win[out[r, :m].long()])
        if m < top_k:
            assert bool((vals[r, m:] == torch.finfo(torch.float32).min).all())


@requires_gvr2
def test_prefill_reference_engine_parity():
    """The b=1 host-loop reference engine (window copied to an aligned row)
    agrees with the in-kernel windowed engine on value multisets."""
    top_k = 512
    rows, ncols = 40, 4096
    ks = [(r * 53) % 1500 for r in range(rows)]
    lens = [min(ncols - s, 300 + r * 90) for r, s in enumerate(ks)]
    lg, rs, le = _make_case(rows, ncols, ks, lens, seed=9)
    o1 = torch.empty(rows, top_k, dtype=torch.int32, device=_DEV)
    o2 = torch.empty_like(o1)
    _host.run_varlen(lg, None, le, o1, top_k=top_k, row_starts=rs, max_seq_len=ncols)
    _host.run_varlen(
        lg,
        None,
        le,
        o2,
        top_k=top_k,
        row_starts=rs,
        max_seq_len=ncols,
        engine="reference",
    )
    torch.cuda.synchronize()
    _check_windowed(lg, o1, rs, le, top_k)
    _check_windowed(lg, o2, rs, le, top_k)
    for r in range(rows):
        m = min(lens[r], top_k)
        win = lg[r, ks[r] : ks[r] + lens[r]]
        assert torch.equal(
            torch.sort(win[o1[r, :m].long()]).values,
            torch.sort(win[o2[r, :m].long()]).values,
        )


@requires_gvr2
def test_prefill_cuda_graph_replay_after_warmup():
    """warmup_prefill compiles the engines; a windowed call then captures and
    replays exactly with changed window contents (row_starts and lengths)."""
    top_k, n = 512, 8192
    rows = 64
    _host.warmup_prefill(top_k, n)
    stride = ((n + 256 + 255) // 256) * 256
    full = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)
    lg = full[:, :n]
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    with torch.cuda.stream(
        s
    ):  # eager warm-up ON the capturing stream (slab + launcher)
        flashinfer.top_k_varlen(
            lg, le, top_k, row_starts=rs, out_indices=out, backend="gvr_2"
        )
    torch.cuda.synchronize()
    assert _host.prefill_ready(rows, top_k, n)
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(
            lg, le, top_k, row_starts=rs, out_indices=out, backend="gvr_2"
        )
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)
    # new windows between replays: staircase of 4 requests, misaligned starts
    ks2 = [(r // 16) * 2001 for r in range(rows)]
    le2 = [min(n - ks2[r], (r % 16 + 1) * 300) for r in range(rows)]
    rs.copy_(torch.tensor(ks2, dtype=torch.int32))
    le.copy_(torch.tensor(le2, dtype=torch.int32))
    full.normal_()
    g.replay()
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)


@requires_gvr2
def test_prefill_unwarmed_capture_raises_loudly():
    top_k, n = 1024, 4096
    rows = 8
    stride = ((n + 256 + 255) // 256) * 256
    lg = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)[:, :n]
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = torch.empty((rows, top_k), dtype=torch.int32, device=_DEV)
    key = _host._prefill_cache_key(
        _host._prefill_tier(rows, n, top_k), top_k, _host._prefill_bucket(n)
    )
    _host._PREFILL_CACHE.pop(key, None)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        _host.default_workspace(lg)  # the slab exists; only the launcher is missing
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with (
        pytest.raises(RuntimeError, match="prefill launcher not compiled"),
        torch.cuda.stream(s),
        torch.cuda.graph(g, stream=s),
    ):
        flashinfer.top_k_varlen(
            lg, le, top_k, row_starts=rs, out_indices=out, backend="gvr_2"
        )
    torch.cuda.synchronize()


@requires_gvr2
def test_prefill_slab_over_gridy_limit():
    """> 65535 rows in one call are slabbed (gridDim.y <= 65535)."""
    k, rows, n = 512, 70000, 2048
    stride = ((n + 256 + 255) // 256) * 256
    lg = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)[:, :n]
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = _run(lg, rs, le, k)
    idx = torch.tensor([0, 1, 32767, 32768, 65535, 65536, 69999], device=_DEV)
    _check_windowed(lg[idx], out[idx].contiguous(), rs[idx], le[idx], k)


@requires_gvr2
def test_prefill_engine_key_distinct_from_decode():
    from flashinfer.topk_varlen.kernels import gvr2_topk_decode as dev

    tpl = (256, 8, 4, 256, 2, False, False, 1, 0, 1)
    a = dev.get_compiled(tpl, hint_free=True)
    b = dev.get_compiled(tpl, hint_free=True, prefill=True)
    assert a is not b
    with pytest.raises(RuntimeError, match="varlen tuple"):
        dev.get_compiled(tpl[:7], hint_free=True, prefill=True)


def test_prefill_api_validation_and_admission():
    """row_starts contract errors are ValueErrors at the API; only gvr_2 admits
    windowed calls (auto never picks another backend for them)."""
    from flashinfer.topk_varlen import topk_varlen as api
    from flashinfer.utils import BackendSupportedError

    n, rows, k = 4096, 8, 512
    lg = torch.randn((rows, n), dtype=torch.float32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    gvr2_here = flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc())
    # tensor-contract errors are ValueErrors raised by the API body (reached
    # only where a checker admitted the call, i.e. where gvr_2 exists)
    if gvr2_here:
        with pytest.raises(ValueError, match="row_starts"):
            flashinfer.top_k_varlen(lg, le, k, row_starts=rs.to(torch.int64))
        with pytest.raises(ValueError, match="row_starts"):
            flashinfer.top_k_varlen(lg, le, k, row_starts=rs[:4])
    else:
        with pytest.raises(BackendSupportedError):
            flashinfer.top_k_varlen(lg, le, k, row_starts=rs)
    # mode conflicts (hint, cr) are refused by every checker: the decorator
    # reports "no suitable backend" for auto and "problem size not supported"
    # (ValueError) for an explicitly requested backend
    hint = torch.zeros(rows, k, dtype=torch.int32, device=_DEV)
    for kw in ({"pre_idx": hint}, {"compress_ratio": 4}):
        with pytest.raises(BackendSupportedError):
            flashinfer.top_k_varlen(lg, le, k, row_starts=rs, **kw)
        with pytest.raises((ValueError, BackendSupportedError)):
            flashinfer.top_k_varlen(lg, le, k, row_starts=rs, backend="gvr_2", **kw)
    # explicit non-gvr_2 backends refuse windowed calls
    for b in ("radix", "radix_cutlass", "radix_filter", "gvr"):
        with pytest.raises((ValueError, BackendSupportedError)):
            flashinfer.top_k_varlen(lg, le, k, row_starts=rs, backend=b)
    # admission: only gvr_2's checker admits row_starts
    common = dict(pre_idx=None, compress_ratio=1, next_n=1, row_starts=rs)
    for chk in (
        api._radix_top_k_varlen_check,
        api._radix_cutlass_top_k_varlen_check,
        api._radix_filter_top_k_varlen_check,
        api._gvr_top_k_varlen_check,
    ):
        assert not chk(lg, le, k, **common), chk.__name__
    if gvr2_here:
        assert api._gvr2_top_k_varlen_check(lg, le, k, **common)
        order = api._top_k_varlen_heuristic(
            ["radix", "gvr_2", "radix_cutlass", "radix_filter"], lg, le, k, **common
        )
        assert order[0] == "gvr_2"
