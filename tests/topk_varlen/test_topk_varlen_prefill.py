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
seq_lens[r])`` candidate windows with window-local output indices (absolute
logits columns with ``absolute_indices=True``), served by the
gvr_2 prefill engines (TRT-LLM #18702 port).

Contract under test: exact top-k of the window (tie-aware), ``-1`` pad, identity
for windows shorter than K, NO read outside the window (out-of-window cells are
poisoned with NaN / +-inf / 3e38 and the <= 3 lead lanes before a misaligned
``ks`` hold +inf so a leak becomes top-1), ``-inf`` masks inside the window,
CUDA-graph replay after warm-up, and the API/backends' admission rules.
"""

import subprocess
import sys

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
    # torch.where, not masked_scatter_: the latter's int64 prefix sum over the
    # mask doubled the peak memory of the large fixtures
    full.copy_(torch.where(outside, pat, full))
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
@pytest.mark.parametrize("hinted", [False, True], ids=["slab", "rung"])
@pytest.mark.parametrize("lead", [1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_prefill_packed_misaligned_ks(top_k, lead, hinted):
    """Two packed requests; the second starts at ks % 4 == lead with +inf
    poison at [ks-lead, ks): a leaked lead lane would become top-1 and a missed
    frame correction a negative index. slab = no bound and a logits width past
    every register capacity (the width is the bound then); rung = the register
    rung picked for the ``max_window_len`` bound."""
    a = 300
    ks1 = ((a + 3) // 4) * 4 + lead
    n1 = 4096 + 17
    rows = a + n1
    ncols = ks1 + n1 + (0 if hinted else 8192)
    ks = [0] * a + [ks1] * n1
    lens = list(range(1, a + 1)) + list(range(1, n1 + 1))
    lg, rs, le = _make_case(rows, ncols, ks, lens, seed=top_k * 100 + lead)
    out = _run(lg, rs, le, top_k, **({"max_window_len": n1} if hinted else {}))
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
@pytest.mark.parametrize("hinted", [False, True], ids=["slab", "rung"])
@pytest.mark.parametrize("lead", [1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_neginf_masks_in_window(top_k, lead, hinted):
    """Fewer than K finite values in the window (the rest -inf, as SGLang's
    init/local-token masks): the K-th boundary lies in the -inf tie class,
    crossed with a misaligned lead; no negative index may leak. slab = width
    past every register capacity, no bound; rung = ``max_window_len`` bound."""
    n_finite = top_k - 100
    nv = top_k + 400
    ks1 = ((37 + 3) // 4) * 4 + lead
    ncols = ks1 + nv + (0 if hinted else 8192)
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
    out = _run(lg, rs, le, top_k, **({"max_window_len": nv} if hinted else {}))
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
    s.wait_stream(torch.cuda.current_stream())
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
    # drop every windowed launcher (slab tiers and register rungs alike)
    _host._PREFILL_CACHE.clear()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
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


@requires_gvr2
@pytest.mark.parametrize(
    "bound", [2045, 2048, 2049, 4096, 4097, 8192], ids=lambda b: f"L{b}"
)
@pytest.mark.parametrize("top_k", [512, 1024, 2048], ids=lambda k: f"k{k}")
def test_prefill_register_rung_hinted(top_k, bound):
    """With ``max_window_len`` the launch runs on the register rung the per-part
    table picks for that bound (or the slab where the table says so): exact on
    every lead with +inf poison, and the launcher the dispatcher resolves for
    this geometry is exactly the one the table names -- its register key is
    dropped first, so the launch itself must have (re)created it."""
    L = bound
    if bound <= 4097:
        R = 3  # three packed requests: every lead class at the request edges
        ks = [(r * L) for r in range(R) for _ in range(L)]
        ncols = R * L
    else:
        # one request at the 8K class, misaligned by 3 (the packing is covered
        # by the classes above): 3 x 8192 rows x 24832 cols fp32 plus the
        # poison scatter peaked near 10 GB and ran a 24576-row check loop
        R = 1
        ks = [3] * L
        ncols = L + 4
    rows = R * L
    lens = [i + 1 for _ in range(R) for i in range(L)]
    lg, rs, le = _make_case(rows, ncols, ks, lens, seed=bound + top_k)
    plan = _host._prefill_reg_route(rows, top_k, L)
    bucket = _host._prefill_bucket(L)
    key = None
    if plan is not None:
        assert plan["tpl"][0] == 512 and plan["tpl"][0] * plan["tpl"][1] * 4 >= L
        key = _host._prefill_cache_key(
            "reg",
            (tuple(plan["tpl"]), plan["rt"]["CMP"], plan["rt"]["QC"], plan["smem"]),
            top_k,
            0,
            False,
        )
        _host._PREFILL_CACHE.pop(key, None)  # the launch below must recreate it
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(
        lg, le, top_k, row_starts=rs, out_indices=out, backend="gvr_2", max_window_len=L
    )
    torch.cuda.synchronize()
    # the bound + 3 lead lanes must fit the rung: no row may be skipped
    assert not bool((out == -7).any()), "a row was left unwritten by the hinted launch"
    _check_windowed(lg, out, rs, le, top_k)
    lc = _host._prefill_get(rows, top_k, L, bucket, compile_ok=False)
    assert lc is not None, "the launch left no launcher for its own geometry"
    if plan is not None:
        assert lc[0] == "reg", "the table names a register rung but the slab ran"
        assert key in _host._PREFILL_CACHE, "exact register launcher key missing"
    else:
        assert lc[0] == "main", "the table sends this bound to the slab"


@requires_gvr2
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_prefill_bound_violation_reports_minus_one(top_k):
    """``max_window_len`` is a contract: on a register rung, a row whose scan
    extent (window + lead lanes) exceeds the rung's capacity + 3-lane tail
    comes out as all -1 — never a truncated ranking — while every conforming
    row, including those exactly at that limit, stays exact at every lead."""
    bound = 4096
    plan = _host._prefill_reg_route(40, top_k, bound)
    if plan is None:
        pytest.skip("slab for this bound on this part: no capacity to violate")
    cap = (
        plan["tpl"][0] * plan["tpl"][1] * 4 + 3
    )  # register batch + the 3-lane scalar tail
    lens_pat = [
        cap - 6,
        cap - 4,
        cap - 3,
        cap - 2,
        cap - 1,
        cap,
        cap + 1,
        cap + 5,
        top_k,
        300,
    ]
    ks_list, len_list = [], []
    for lead in range(4):
        for ln in lens_pat:
            ks_list.append(1024 + lead)
            len_list.append(ln)
    rows = len(ks_list)
    ncols = 1024 + 4 + max(len_list)
    lg, rs, le = _make_case(rows, ncols, ks_list, len_list, seed=top_k)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    flashinfer.top_k_varlen(
        lg,
        le,
        top_k,
        row_starts=rs,
        out_indices=out,
        backend="gvr_2",
        max_window_len=bound,
    )
    torch.cuda.synchronize()
    assert not bool((out == -7).any()), "a row was left unwritten"
    fits = [(ln + (ks & 3)) <= cap for ks, ln in zip(ks_list, len_list, strict=True)]
    for r in range(rows):
        if fits[r]:
            continue
        assert bool((out[r] == -1).all()), f"row {r}: violating row must be all -1"
    keep = [r for r in range(rows) if fits[r]]
    idx = torch.tensor(keep, device=_DEV)
    _check_windowed(lg[idx], out[idx].contiguous(), rs[idx], le[idx], top_k)


def test_prefill_route_table_pure():
    """The per-part rung table (pure host logic) — pins the measured choices."""
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(_host, "_arch_token", lambda: "sm100")
        mp.setattr(_host, "_sm_count", lambda: 148)
        r = _host._prefill_reg_route
        assert r(8192, 512, 2045)["tpl"][:3] == (512, 1, 4)
        assert r(8192, 512, 2048)["tpl"][:3] == (512, 1, 4)  # lead lanes ride the tail
        assert r(8192, 512, 2049)["tpl"][:3] == (512, 2, 4)
        assert r(8192, 1024, 2048)["tpl"][:3] == (512, 1, 4)
        assert r(8192, 2048, 2048) is None  # bound <= k: every row identity, slab
        assert r(8192, 2048, 4096)["tpl"][:3] == (512, 2, 4)
        assert r(8192, 512, 4096)["tpl"][:3] == (512, 2, 4)
        assert r(8192, 512, 4097) is None  # K <= 1024 above 4K: slab
        assert r(8192, 512, 8192) is None
        assert r(8192, 2048, 8192)["tpl"][:3] == (512, 4, 2)
        assert r(8192, 2048, 8193) is None
        assert r(8192, 512, 400) is None  # bound <= k: identity rows, slab
        mp.setattr(_host, "_arch_token", lambda: "sm107")
        mp.setattr(_host, "_sm_count", lambda: 208)
        assert r(8192, 512, 2048)["tpl"][:3] == (512, 1, 4)
        assert r(32768, 512, 2048) is None  # short windows, high rows: slab
        assert r(8192, 1024, 2048)["tpl"][:3] == (512, 1, 4)
        assert r(8192, 512, 2100)["tpl"][:3] == (512, 2, 2)  # VPT=2 class, small launch
        assert r(32768, 512, 2100) is None  # mostly-empty VPT=2 at high rows: slab
        assert r(8192, 2048, 2100) is None
        assert r(32768, 2048, 4096)["tpl"][:3] == (512, 2, 2)  # full 4K windows: rung
        assert r(8192, 512, 4096)["tpl"][:3] == (512, 2, 2)
        assert r(8192, 2048, 8192)["tpl"][:3] == (512, 4, 2)
        assert r(8192, 512, 8192) is None
    finally:
        mp.undo()


@requires_gvr2
def test_prefill_hinted_cuda_graph_replay():
    """warmup_prefill compiles the register rungs too; a hinted capture replays
    exactly with changed windows."""
    top_k, L, R = 512, 2045, 4
    rows = R * L
    _host.warmup_prefill(top_k, R * L)
    # finite everywhere (the second replay grows windows past the first ones)
    stride = (R * L + 256 + 255) // 256 * 256
    full = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)
    lg = full[:, : R * L]
    rs = torch.tensor(
        [(r * L) for r in range(R) for _ in range(L)], dtype=torch.int32, device=_DEV
    )
    le = torch.tensor(
        [i + 1 for _ in range(R) for i in range(L)], dtype=torch.int32, device=_DEV
    )
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            out_indices=out,
            backend="gvr_2",
            max_window_len=L,
        )
    torch.cuda.synchronize()
    assert _host.prefill_ready(rows, top_k, L, width=R * L)
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            out_indices=out,
            backend="gvr_2",
            max_window_len=L,
        )
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)
    le.copy_(
        torch.tensor(
            [max(1, (i * 7) % L) for _ in range(R) for i in range(L)], dtype=torch.int32
        )
    )
    g.replay()
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)


def test_prefill_max_window_len_validation():
    n, rows, k = 4096, 8, 512
    lg = torch.randn((rows, n), dtype=torch.float32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    if flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()):
        for bad in (0, -1, "8", 2.5, True):
            with pytest.raises(ValueError, match="max_window_len"):
                flashinfer.top_k_varlen(lg, le, k, row_starts=rs, max_window_len=bad)
    # decode mode takes the bound too (on seq_lens, in tokens) and refuses a
    # bad value the same way, any backend
    for bad in (0, "8", True):
        with pytest.raises(ValueError, match="max_window_len"):
            flashinfer.top_k_varlen(lg, le, k, max_window_len=bad)


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
    # decode-mode arguments (hint, next_n, compress_ratio) are refused by the
    # API body, even at their decode defaults: passing them at all is the
    # error. The checkers admit the call so that auto and an explicit gvr_2
    # both report that ValueError (where gvr_2 exists; elsewhere auto has no
    # backend that admits a windowed call at all)
    hint = torch.zeros(rows, k, dtype=torch.int32, device=_DEV)
    for kw in (
        {"pre_idx": hint},
        {"compress_ratio": 4},
        {"compress_ratio": 1},
        {"compress_ratio": 2},
        {"next_n": 2},
        {"next_n": 1},
        {"next_n": 1, "compress_ratio": 1},
    ):
        for extra in ({}, {"backend": "gvr_2"}):
            if gvr2_here:
                with pytest.raises(ValueError, match="decode-mode argument"):
                    flashinfer.top_k_varlen(lg, le, k, row_starts=rs, **kw, **extra)
            else:
                with pytest.raises((ValueError, BackendSupportedError)):
                    flashinfer.top_k_varlen(lg, le, k, row_starts=rs, **kw, **extra)

    # the sentinel defaults: omitted, None and 1 are the same decode call
    # (radix_cutlass exists on every GPU; full rows, so every slot is a hit and
    # the sorted selected values are a tie-safe identity of the selection)
    def _vals(idx):
        return lg.gather(1, idx.long()).sort(dim=1).values

    plain = flashinfer.top_k_varlen(lg, le, k, backend="radix_cutlass")[0]
    for kw in (
        {"next_n": None, "compress_ratio": None},
        {"next_n": 1, "compress_ratio": 1},
    ):
        got = flashinfer.top_k_varlen(lg, le, k, backend="radix_cutlass", **kw)[0]
        assert torch.equal(_vals(got), _vals(plain)), kw
    if gvr2_here:
        # an explicit None is the omitted form in windowed mode too (rs == 0 and
        # full windows here, so window positions are the columns)
        plain_w = flashinfer.top_k_varlen(lg, le, k, row_starts=rs)[0]
        none_w = flashinfer.top_k_varlen(
            lg, le, k, row_starts=rs, next_n=None, compress_ratio=None, pre_idx=None
        )[0]
        assert torch.equal(_vals(none_w), _vals(plain_w))
    # explicit non-gvr_2 backends refuse windowed calls
    for b in ("radix", "radix_cutlass", "radix_filter", "gvr"):
        with pytest.raises((ValueError, BackendSupportedError)):
            flashinfer.top_k_varlen(lg, le, k, row_starts=rs, backend=b)
    # admission: only gvr_2's checker admits row_starts
    common = dict(row_starts=rs)
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


def _abs_lens(rows, ncols, ks, top_k):
    """Window-length menu per row: empty, one, K-1, K, K+3, mid, the largest
    register-class bounds; every row fits its request's slice."""
    menu = [0, 1, top_k - 1, top_k, top_k + 3, 1500, 2045, 2048]
    return [max(0, min(ncols - s, menu[r % len(menu)])) for r, s in enumerate(ks)]


def _same_selection(logits, a, b, starts=None):
    """Tie-safe row-wise equality of two selections (the engines' emit order is
    not fixed, and two launches may break a K-th-boundary tie differently, so
    index sets are not the contract): the -1 pad pattern must match and the
    sorted multisets of the SELECTED VALUES must match. ``a`` and ``b`` hold
    absolute logits columns, or window-local indices when ``starts`` (clamped
    like the kernel) is given."""
    assert a.shape == b.shape
    if starts is not None:
        off = starts.clamp(min=0, max=logits.shape[1]).unsqueeze(1).to(a.dtype)
        a = torch.where(a >= 0, a + off, a)
        b = torch.where(b >= 0, b + off, b)
    pa, pb = a < 0, b < 0
    assert torch.equal(pa.sum(1), pb.sum(1)), "pad counts differ"
    nan = torch.tensor(float("nan"), device=logits.device)
    va = torch.where(pa, nan, logits.gather(1, a.clamp_min(0).long()))
    vb = torch.where(pb, nan, logits.gather(1, b.clamp_min(0).long()))
    sa, sb = torch.sort(va, dim=1).values, torch.sort(vb, dim=1).values
    same = (sa == sb) | (sa.isnan() & sb.isnan())
    assert bool(same.all()), "selected value multisets differ"
    return True


@requires_gvr2
@pytest.mark.parametrize("hinted", [False, True], ids=["slab", "rung"])
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_prefill_absolute_indices(top_k, hinted):
    """``absolute_indices=True`` returns logits columns: every hit is the
    window-local index plus ``row_starts[r]``, ``-1`` pads are untouched and
    identity (short) windows are shifted too. Both engine families: the
    ``max_window_len`` bound admits the register rung; without a bound the width
    (past every register capacity here) routes to the slab. Values are the
    logits at the absolute columns."""
    rows, ncols = 96, 6144 if hinted else 6144 + 8192
    ks = [(r * 61) % 3000 + (r % 4) for r in range(rows)]  # every lead 0..3
    lens = _abs_lens(rows, ncols, ks, top_k)
    lg, rs, le = _make_case(rows, ncols, ks, lens, seed=11 + top_k)
    kw = {"max_window_len": 2048} if hinted else {}
    loc = _run(lg, rs, le, top_k, **kw)
    _check_windowed(lg, loc, rs, le, top_k)
    ab, vals = flashinfer.top_k_varlen(
        lg,
        le,
        top_k,
        row_starts=rs,
        backend="gvr_2",
        absolute_indices=True,
        return_values=True,
        **kw,
    )
    torch.cuda.synchronize()
    _same_selection(lg, ab, torch.where(loc >= 0, loc + rs.unsqueeze(1), loc))
    hit = ab >= 0
    assert torch.equal(vals[hit], lg.gather(1, ab.clamp_min(0).long())[hit])
    assert bool((vals[~hit] == torch.finfo(torch.float32).min).all())


@requires_gvr2
def test_prefill_absolute_indices_reference_engine():
    """The host-loop reference engine honours ``absolute_indices`` (value
    multisets agree with the in-kernel engine; pads coincide)."""
    top_k = 512
    rows, ncols = 24, 4096
    ks = [(r * 53) % 1500 + (r % 4) for r in range(rows)]
    lens = _abs_lens(rows, ncols, ks, top_k)
    lg, rs, le = _make_case(lg_rows := rows, ncols, ks, lens, seed=21)
    o1 = torch.empty(lg_rows, top_k, dtype=torch.int32, device=_DEV)
    o2 = torch.empty_like(o1)
    _host.run_varlen(
        lg,
        None,
        le,
        o1,
        top_k=top_k,
        row_starts=rs,
        max_seq_len=ncols,
        absolute_indices=True,
    )
    _host.run_varlen(
        lg,
        None,
        le,
        o2,
        top_k=top_k,
        row_starts=rs,
        max_seq_len=ncols,
        engine="reference",
        absolute_indices=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(o1 < 0, o2 < 0)
    for r in range(rows):
        m = min(lens[r], top_k)
        assert bool((o1[r, :m] >= ks[r]).all()) and bool(
            (o1[r, :m] < ks[r] + lens[r]).all()
        )
        assert torch.equal(
            torch.sort(lg[r, o1[r, :m].long()]).values,
            torch.sort(lg[r, o2[r, :m].long()]).values,
        )


@requires_gvr2
def test_prefill_absolute_indices_warmup_and_graph_replay():
    """``warmup_prefill(absolute_indices=True)`` compiles the absolute variants
    (a distinct set: the local one is not implied), ``prefill_ready`` sees
    them, and a captured absolute call replays exactly on new windows."""
    top_k, n, rows = 512, 4096, 48
    _host._PREFILL_CACHE.clear()
    _host._PREFILL_WARMUP_DONE.clear()
    _host.warmup_prefill(top_k, n, absolute_indices=True)
    assert _host.prefill_ready(rows, top_k, n, absolute_indices=True)
    assert _host.prefill_ready(rows, top_k, 2048, width=n, absolute_indices=True)
    assert not _host.prefill_ready(rows, top_k, n)
    stride = ((n + 256 + 255) // 256) * 256
    full = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)
    lg = full[:, :n]
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            out_indices=out,
            backend="gvr_2",
            absolute_indices=True,
        )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            out_indices=out,
            backend="gvr_2",
            absolute_indices=True,
        )
    torch.cuda.current_stream().wait_stream(s)
    ks2 = [(r // 12) * 1001 + (r % 4) for r in range(rows)]
    le2 = [min(n - ks2[r], (r % 12 + 1) * 250) for r in range(rows)]
    rs.copy_(torch.tensor(ks2, dtype=torch.int32))
    le.copy_(torch.tensor(le2, dtype=torch.int32))
    full.normal_()
    g.replay()
    torch.cuda.synchronize()
    loc = _run(lg, rs, le, top_k)
    _check_windowed(lg, loc, rs, le, top_k)
    _same_selection(lg, out, torch.where(loc >= 0, loc + rs.unsqueeze(1), loc))


def test_prefill_absolute_indices_validation():
    n, rows, k = 4096, 8, 512
    lg = torch.randn((rows, n), dtype=torch.float32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    if flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()):
        for bad in (1, "yes", None):
            with pytest.raises(ValueError, match="absolute_indices"):
                flashinfer.top_k_varlen(lg, le, k, row_starts=rs, absolute_indices=bad)
    # decode mode has no window frame to shift: refused on every backend
    with pytest.raises(ValueError, match="absolute_indices"):
        flashinfer.top_k_varlen(lg, le, k, absolute_indices=True)


def test_prefill_skip_check_refuses_other_backends():
    """``skip_check=True`` bypasses the checkers; the dispatcher must still
    refuse ``row_starts`` on every backend but gvr_2 BEFORE launching anything
    (a radix / gvr run would silently rank the row prefix, not the window)."""
    from flashinfer.utils import BackendSupportedError

    n, rows, k = 4096, 8, 512
    lg = torch.randn((rows, n), dtype=torch.float32, device=_DEV)
    le = torch.full((rows,), 1000, dtype=torch.int32, device=_DEV)
    rs = torch.full((rows,), 100, dtype=torch.int32, device=_DEV)
    for b in ("radix", "radix_cutlass", "radix_filter", "gvr"):
        with pytest.raises(BackendSupportedError, match="row_starts"):
            flashinfer.top_k_varlen(
                lg, le, k, row_starts=rs, backend=b, skip_check=True
            )
    # decode-mode arguments are refused by the API body before any dispatch;
    # with the checkers bypassed this holds on every GPU, whatever the backend
    hint = torch.zeros(rows, k, dtype=torch.int32, device=_DEV)
    for kw in ({"next_n": 1}, {"compress_ratio": 1}, {"pre_idx": hint}):
        for b in ("gvr_2", "auto"):
            with pytest.raises(ValueError, match="decode-mode argument"):
                flashinfer.top_k_varlen(
                    lg, le, k, row_starts=rs, backend=b, skip_check=True, **kw
                )
    if flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()):
        out, _ = flashinfer.top_k_varlen(
            lg, le, k, row_starts=rs, backend="gvr_2", skip_check=True
        )
        torch.cuda.synchronize()
        _check_windowed(lg, out, rs, le, k)


@requires_gvr2
def test_prefill_zero_rows():
    """An empty batch returns without a launch; a caller buffer is untouched."""
    n, k = 4096, 512
    lg = torch.empty((0, n), dtype=torch.float32, device=_DEV)
    le = torch.empty((0,), dtype=torch.int32, device=_DEV)
    rs = torch.empty((0,), dtype=torch.int32, device=_DEV)
    out, _ = flashinfer.top_k_varlen(lg, le, k, row_starts=rs, backend="gvr_2")
    assert out.shape == (0, k)
    buf = torch.empty((0, k), dtype=torch.int32, device=_DEV)
    out2, _ = flashinfer.top_k_varlen(
        lg, le, k, row_starts=rs, out_indices=buf, backend="gvr_2"
    )
    assert out2.shape == (0, k)
    torch.cuda.synchronize()


@requires_gvr2
@pytest.mark.parametrize("hinted", [False, True], ids=["slab", "rung"])
def test_prefill_out_values_through_api(hinted):
    """``return_values`` with caller buffers (2-D and flat): the values are the
    window scores at the selected columns and the pads hold the fp32 sentinel,
    on both engines (slab = no bound, width past the register capacity; rung =
    ``max_window_len`` bound)."""
    top_k, R, L = 512, 2, 1500
    rows = R * L
    ks = [(r * L) for r in range(R) for _ in range(L)]
    lens = [i + 1 for _ in range(R) for i in range(L)]
    lg, rs, le = _make_case(rows, R * L + (0 if hinted else 8192), ks, lens, seed=11)
    for shape in ((rows, top_k), (rows * top_k,)):
        out_i = torch.full(shape, -7, dtype=torch.int32, device=_DEV)
        out_v = torch.full(shape, 123.0, dtype=torch.float32, device=_DEV)
        idx, vals = flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            return_values=True,
            out_indices=out_i,
            out_values=out_v,
            backend="gvr_2",
            max_window_len=L if hinted else None,
        )
        torch.cuda.synchronize()
        idx = idx.view(rows, top_k)
        vals = vals.view(rows, top_k)
        _check_windowed(lg, idx, rs, le, top_k)
        valid = idx >= 0
        cols = (idx.to(torch.int64) + rs.unsqueeze(1).to(torch.int64)).clamp_min(0)
        gathered = lg.gather(1, cols)
        assert torch.equal(vals[valid], gathered[valid])
        assert bool((vals[~valid] == torch.finfo(torch.float32).min).all())


@requires_gvr2
def test_prefill_row_starts_out_of_range():
    """Out-of-range starts: a negative ks is clamped to 0 and the window keeps
    its length; a ks at or past the width is an empty window (all -1); a window
    past the width is clamped. Never a read outside the row (poisoned cells)."""
    n, k, rows = 4096, 512, 4
    full = torch.full((rows, ((n + 256 + 255) // 256) * 256), float("nan"), device=_DEV)
    lg = full[:, :n]
    lg.normal_()  # finite inside the row, NaN in the stride slack beyond it
    le = torch.tensor([1000, 1000, 1000, 1000], dtype=torch.int32, device=_DEV)
    rs = torch.tensor([-1, n + 5, 0, n - 700], dtype=torch.int32, device=_DEV)
    out = _run(lg, rs, le, k)
    ref = _run(
        lg, torch.tensor([0, 0, 0, n - 700], dtype=torch.int32, device=_DEV), le, k
    )
    _same_selection(lg[0:1], out[0:1], ref[0:1])  # ks=-1 behaves as ks=0
    assert bool((out[1] == -1).all())  # ks past the width: empty window
    _same_selection(lg[2:3], out[2:3], ref[2:3])
    # window of 1000 at n-700 is clamped to the 700 columns that exist
    _check_windowed(lg[3:4], out[3:4], rs[3:4], le[3:4], k)
    ref_eng = torch.empty((rows, k), dtype=torch.int32, device=_DEV)
    _host.run_varlen(lg, None, le, ref_eng, top_k=k, row_starts=rs, engine="reference")
    torch.cuda.synchronize()
    _same_selection(lg, out, ref_eng, starts=rs)
    # returned values are read at the column the kernel ranked: the clamped
    # start plus the window-local index (a negative start acts as 0)
    idx_v, vals = flashinfer.top_k_varlen(lg, le, k, row_starts=rs, return_values=True)
    torch.cuda.synchronize()
    starts = rs.to(torch.int64).clamp(min=0)
    for r in (0, 2, 3):
        hit = idx_v[r] >= 0
        cols = idx_v[r][hit].to(torch.int64) + starts[r]
        assert torch.equal(vals[r][hit], lg[r, cols]), r
    assert bool(
        (vals[1] == torch.finfo(torch.float32).min).all()
    )  # empty window: pads only


@requires_gvr2
def test_prefill_warmup_covers_identity_bounds():
    """A bound at or below top_k (every row an identity row) shares the smallest
    non-identity bucket's launcher, so warmup_prefill covers it, prefill_ready says
    so, and a capture of such a call succeeds without an eager call of that
    geometry (reviewer's reproducer: warm up K=2048 at 4096 columns, then a
    one-row chunk with bound 512)."""
    k, n = 2048, 4096
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _host.warmup_prefill(k, n)
        assert _host.prefill_ready(1, k, 512, width=n)
        lg = torch.randn((1, n), dtype=torch.float32, device=_DEV)
        rs = torch.zeros((1,), dtype=torch.int32, device=_DEV)
        le = torch.tensor([300], dtype=torch.int32, device=_DEV)
        out = torch.full((1, k), -7, dtype=torch.int32, device=_DEV)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            flashinfer.top_k_varlen(
                lg, le, k, row_starts=rs, max_window_len=512, out_indices=out
            )
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(
        out[0, :300].sort().values, torch.arange(300, dtype=torch.int32, device=_DEV)
    )
    assert bool((out[0, 300:] == -1).all())


@requires_gvr2
def test_prefill_cuda_graph_replay_grows_windows():
    """Capture with every window short (identity rows at ks=0); replay with
    misaligned starts of every lead class and full-width windows: the slab
    engine reads starts and lengths on device, so the grown windows rank
    exactly (the opposite direction of the shrink replay above)."""
    top_k, n, rows = 512, 8192, 64
    _host.warmup_prefill(top_k, n)
    stride = ((n + 256 + 255) // 256) * 256
    full = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)
    lg = full[:, :n]
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    le = torch.full((rows,), 300, dtype=torch.int32, device=_DEV)  # 300 < k
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flashinfer.top_k_varlen(
            lg, le, top_k, row_starts=rs, out_indices=out, backend="gvr_2"
        )
    torch.cuda.synchronize()
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
    ks2 = [(r % 4) + (r // 16) * 1001 for r in range(rows)]  # every lead class
    le2 = [n - ks2[r] for r in range(rows)]  # full-width windows
    rs.copy_(torch.tensor(ks2, dtype=torch.int32))
    le.copy_(torch.tensor(le2, dtype=torch.int32))
    full.normal_()
    g.replay()
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)


@requires_gvr2
def test_prefill_hinted_cuda_graph_replay_shifts_starts():
    """Hinted (register rung) capture at aligned starts; each replay shifts
    every request's start by 1, 2 and 3 columns (every lead class) with the
    windows at the bound."""
    top_k, L, R = 512, 2045, 4
    rows = R * L
    ncols = R * L + 4
    _host.warmup_prefill(top_k, ncols)
    stride = (ncols + 256 + 255) // 256 * 256
    full = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)
    lg = full[:, :ncols]
    base = [(r * L) for r in range(R) for _ in range(L)]
    rs = torch.tensor(base, dtype=torch.int32, device=_DEV)
    le = torch.tensor(
        [i + 1 for _ in range(R) for i in range(L)], dtype=torch.int32, device=_DEV
    )
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            out_indices=out,
            backend="gvr_2",
            max_window_len=L,
        )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            out_indices=out,
            backend="gvr_2",
            max_window_len=L,
        )
    torch.cuda.current_stream().wait_stream(s)
    for shift in (1, 2, 3):
        rs.copy_(torch.tensor([b + shift for b in base], dtype=torch.int32))
        full.normal_()
        g.replay()
        torch.cuda.synchronize()
        _check_windowed(lg, out, rs, le, top_k)


@requires_gvr2
def test_prefill_hinted_capture_on_fresh_stream_needs_no_slab():
    """A hinted register-rung call never needs the per-stream workspace slab:
    after warm-up it captures on a stream that ran no eager gvr_2 call, and
    ``prefill_ready`` says so; a slab-routed geometry (width past the register
    capacity) on that stream is not ready."""
    top_k, L, R = 512, 2045, 2
    rows = R * L
    wide = 8192 + 4  # a width every register rung refuses -> slab engine
    _host.warmup_prefill(top_k, wide)  # on the current (default) stream
    stride = (R * L + 256 + 255) // 256 * 256
    full = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)
    lg = full[:, : R * L]
    rs = torch.tensor(
        [(r * L) for r in range(R) for _ in range(L)], dtype=torch.int32, device=_DEV
    )
    le = torch.tensor(
        [i + 1 for _ in range(R) for i in range(L)], dtype=torch.int32, device=_DEV
    )
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    # torch.cuda.Stream() hands out pooled handles: an earlier test may have
    # left a slab on this very handle, so establish "no slab on s" instead of
    # assuming it (retire every launch first, then drop the entry under the lock)
    torch.cuda.synchronize()
    with torch.cuda.stream(s), _host._mu:
        _host._ws_keep.pop(_host._ws_key(torch.cuda.current_device()), None)
    with torch.cuda.stream(s):
        assert _host.prefill_ready(rows, top_k, L, width=R * L)
        assert not _host.prefill_ready(rows, top_k, wide)  # slab engine: no slab on s
    g = torch.cuda.CUDAGraph()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(
            lg,
            le,
            top_k,
            row_starts=rs,
            out_indices=out,
            backend="gvr_2",
            max_window_len=L,
        )
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)


@requires_gvr2
def test_prefill_warmup_custom_rows_cover_slabs_and_rungs():
    """A custom ``num_rows_list`` is decomposed into launch slabs and warmed for
    the key the serving call looks up (not a substituted row count): a count
    with a small remainder slab and a count above every Rubin gate both report
    ready for the hinted and the unhinted call."""
    top_k, n = 512, 2048
    _host._PREFILL_CACHE.clear()
    _host._PREFILL_WARMUP_DONE.clear()
    rows = _host._PREFILL_ROW_SLAB + 100  # tail slab of 100 rows
    _host.warmup_prefill(top_k, n, num_rows_list=(rows, 17000))
    for r in (rows, 17000):
        assert _host.prefill_ready(r, top_k, n)
        assert _host.prefill_ready(r, top_k, n, width=n)
        assert _host.prefill_ready(r, top_k, n // 2 + 1, width=n)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_prefill_absolute_indices_on_non_current_device():
    """logits on a device that is not the current one: the host re-enters
    itself under that device and must forward every windowed kwarg, so the
    absolute frame survives (it was silently dropped before)."""
    other = torch.device("cuda:1")
    major, minor = get_compute_capability(other)
    if not flashinfer.top_k_varlen.is_backend_supported("gvr_2", major * 10 + minor):
        pytest.skip("gvr_2 unsupported on cuda:1")
    torch.cuda.set_device(0)
    top_k, R, L = 512, 2, 1500
    rows = R * L
    gen = torch.Generator(device=other).manual_seed(5)
    lg = torch.randn((rows, R * L), generator=gen, dtype=torch.float32, device=other)
    rs = torch.tensor(
        [(r * L) for r in range(R) for _ in range(L)], dtype=torch.int32, device=other
    )
    le = torch.tensor(
        [i + 1 for _ in range(R) for i in range(L)], dtype=torch.int32, device=other
    )
    out_abs, _ = flashinfer.top_k_varlen(
        lg, le, top_k, row_starts=rs, absolute_indices=True, backend="gvr_2"
    )
    out_loc, _ = flashinfer.top_k_varlen(lg, le, top_k, row_starts=rs, backend="gvr_2")
    torch.cuda.synchronize(other)
    assert torch.cuda.current_device() == 0
    _check_windowed(lg, out_loc, rs, le, top_k)
    _same_selection(
        lg, out_abs, torch.where(out_loc >= 0, out_loc + rs.unsqueeze(1), out_loc)
    )


@requires_gvr2
def test_prefill_warmup_recreates_slab_after_release():
    """``release_gvr2_resources()`` drops the per-stream workspace slabs but
    not the warm-up done set, so a second ``warmup_prefill`` with the same key
    used to short-circuit and leave the stream slab-less: a slab-routed capture
    then raised. The re-warm must recreate THIS stream's slab (the engines are
    still compiled) and the capture must work."""
    top_k, n, rows = 512, 8192, 64  # width 8192 at K=512 routes to the slab engine
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _host.warmup_prefill(top_k, n)
        assert _host.prefill_ready(rows, top_k, n)
        torch.cuda.synchronize()
        assert flashinfer.topk_varlen.release_gvr2_resources() > 0
        assert not _host.prefill_ready(rows, top_k, n), "slab released: not ready"
        _host.warmup_prefill(top_k, n)  # same done key: must still recreate the slab
        assert _host.prefill_ready(rows, top_k, n)
    torch.cuda.synchronize()
    stride = ((n + 256 + 255) // 256) * 256
    full = torch.randn((rows, stride), dtype=torch.float32, device=_DEV)
    lg = full[:, :n]
    rs = torch.zeros((rows,), dtype=torch.int32, device=_DEV)
    le = torch.full((rows,), n, dtype=torch.int32, device=_DEV)
    out = torch.full((rows, top_k), -7, dtype=torch.int32, device=_DEV)
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        flashinfer.top_k_varlen(
            lg, le, top_k, row_starts=rs, out_indices=out, backend="gvr_2"
        )
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)


def test_prefill_env_probe_mirrors_gvr2_cc_list():
    """The host's environment probe admits exactly the compute capabilities the
    public gvr_2 checker declares (every CC-10 part), so the two enumerations
    cannot drift apart silently."""
    from flashinfer.topk_varlen import topk_varlen as _api

    assert set(_api._GVR_CCS) == {cc for cc in _api._ALL_CCS if cc // 10 == 10}


@pytest.mark.skipif(
    not _FLASHINFER_AVAILABLE
    or not torch.cuda.is_available()
    or flashinfer.top_k_varlen.is_backend_supported("gvr_2", _cc()),
    reason="needs a device outside the gvr_2 support set",
)
def test_prefill_helpers_refuse_unsupported_device():
    """On a part outside the gvr_2 support set ``prefill_ready`` is False and
    ``warmup_prefill`` raises one sentence naming the reason, instead of an
    opaque compile or launch error."""
    assert _host.prefill_ready(8, 512, 4096) is False
    with pytest.raises(RuntimeError, match="outside the gvr_2 support set"):
        _host.warmup_prefill(512, 4096)


def test_prefill_helpers_without_cute_dsl_subprocess():
    """With the CuTe DSL made unavailable (``sys.modules['cutlass'] = None``
    before any flashinfer import) the package still imports, the lazy
    ``flashinfer.topk_varlen`` exports resolve, ``release_gvr2_resources`` is a
    no-op, ``prefill_ready`` is False and ``warmup_prefill`` raises a sentence
    naming the missing dependency (never a NameError or a compile error)."""
    code = r"""
import sys
for name in ("cutlass", "cutlass.cute", "cutlass.cutlass_dsl", "cutlass._mlir"):
    sys.modules[name] = None
import flashinfer
from flashinfer.topk_varlen import release_gvr2_resources
assert release_gvr2_resources() == 0  # the host module was never loaded
from flashinfer.topk_varlen import prefill_ready, warmup_prefill
assert prefill_ready(8, 512, 4096) is False
try:
    warmup_prefill(512, 4096)
except RuntimeError as e:
    assert "nvidia-cutlass-dsl" in str(e), e
else:
    raise SystemExit("warmup_prefill did not raise")
print("NO_DSL_OK")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
    )
    assert r.returncode == 0 and "NO_DSL_OK" in r.stdout, (
        r.stdout[-2000:] + r.stderr[-4000:]
    )


@requires_gvr2
@pytest.mark.parametrize("hinted", [False, True], ids=["slab", "rung"])
@pytest.mark.parametrize("lead", [0, 1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 2048], ids=lambda k: f"k{k}")
def test_prefill_plus_inf_inside_window(top_k, lead, hinted):
    """A literal +inf INSIDE the window is a valid candidate and must be
    selected (the decode engine once omitted +inf; the windowed engines rank it
    like any value): planted at the window's first and last lane, a middle
    lane, around the K-th boundary and on a stride, under every lead class, on
    both engine families, with the usual +inf poison in the lead lanes."""
    nv = top_k + 700
    ks1 = 40 + lead
    ncols = ks1 + nv + (0 if hinted else 8192)
    rows = 6
    lg, rs, le = _make_case(rows, ncols, [ks1] * rows, [nv] * rows, seed=top_k + lead)
    planted = {
        0: [0],
        1: [nv - 1],
        2: [nv // 2],
        3: [0, nv - 1, 7, nv // 3],
        4: [top_k - 1, top_k],
        5: list(range(0, nv, 97)),
    }
    for r, cols in planted.items():
        for p in cols:
            lg[r, ks1 + p] = float("inf")
    out = _run(lg, rs, le, top_k, **({"max_window_len": nv} if hinted else {}))
    _check_windowed(lg, out, rs, le, top_k)
    for r, cols in planted.items():
        sel = set(out[r][out[r] >= 0].tolist())
        assert set(cols) <= sel, (
            f"row {r}: +inf candidates omitted {sorted(set(cols) - sel)[:4]}"
        )


@requires_gvr2
@pytest.mark.parametrize("hinted", [False, True], ids=["slab", "rung"])
@pytest.mark.parametrize("lead", [0, 1, 2, 3], ids=lambda x: f"lead{x}")
@pytest.mark.parametrize("top_k", [512, 1024], ids=lambda k: f"k{k}")
def test_prefill_all_neginf_window(top_k, lead, hinted):
    """A window holding only -inf (every candidate masked, as SGLang's masks can
    produce) with more candidates than K: K distinct in-window indices, values
    all -inf, no -1 pad and nothing from outside the window -- a -1 pad and a
    valid -inf must stay distinguishable on both engine families."""
    nv = top_k + 300
    ks1 = 24 + lead
    ncols = ks1 + nv + (0 if hinted else 8192)
    rows = 4
    stride = ((ncols + 256 + 255) // 256) * 256
    full = torch.full((rows, stride), 3e38, dtype=torch.float32, device=_DEV)
    full[:, :ks1] = float("inf")  # lead lanes / previous request: a leak is top-1
    full[:, ks1 : ks1 + nv] = float("-inf")
    lg = full[:, :ncols]
    rs = torch.tensor([ks1] * rows, dtype=torch.int32, device=_DEV)
    le = torch.tensor([nv] * rows, dtype=torch.int32, device=_DEV)
    out, vals = flashinfer.top_k_varlen(
        lg,
        le,
        top_k,
        row_starts=rs,
        backend="gvr_2",
        return_values=True,
        **({"max_window_len": nv} if hinted else {}),
    )
    torch.cuda.synchronize()
    _check_windowed(lg, out, rs, le, top_k)
    assert bool((vals == float("-inf")).all())
