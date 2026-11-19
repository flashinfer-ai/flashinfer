"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the SM103 "specialized" DeepSeek-V4-family sparse MLA decode
backend of trtllm_batch_decode_sparse_mla_dsv4: correctness vs an fp32
reference at allowlisted shape classes, backend-dispatch semantics (auto /
explicit / kill switch / off-allowlist raise), and CUDA-graph replay ==
eager.
"""

import pytest
import torch

import flashinfer.sparse_mla_decode_dsv4_cuda as spec_mod
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized sparse-MLA dsv4 decode kernel is SM103-only",
)

H, D, W = 16, 512, 640
SWA_COLS, CKV_COLS = 128, 512
PS_SWA, PS_CKV, COMPRESS = 256, 64, 8
SCALE = 0.044194173824159216  # 512 ** -0.5
# Allowlisted (B, SEQ-bucket) classes use these per-request kv lens (the
# shape classes the kernel's allowlist keys on).
PINNED = {(1, 32768): [41333], (2, 65536): [41675, 82822]}


def _make_inputs(B, Q, SEQ, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    gc = torch.Generator().manual_seed(seed + 2)
    lens = PINNED.get((B, SEQ))
    if lens is None:
        gl = torch.Generator().manual_seed(B * 1000003 + SEQ)
        lens = [
            int(torch.randint(max(SEQ // 2, 256), SEQ + 1, (1,), generator=gl).item())
            for _ in range(B)
        ]
    T = B * Q
    swa_pages = sum((s + PS_SWA - 1) // PS_SWA for s in lens)
    ckv_pages = sum(
        ((s + COMPRESS - 1) // COMPRESS + PS_CKV - 1) // PS_CKV for s in lens
    )
    swa_perm = torch.randperm(swa_pages, generator=gc)
    ckv_perm = torch.randperm(ckv_pages, generator=gc)

    def pool(pages, ps):
        out = torch.empty(pages, ps, D, dtype=torch.bfloat16, device=dev)
        flat = out.reshape(pages * ps, D)
        step = 262144
        for r0 in range(0, pages * ps, step):
            n = min(step, pages * ps - r0)
            flat[r0 : r0 + n] = (torch.randn(n, D, generator=g, device=dev) * 0.5).to(
                torch.bfloat16
            )
        return out

    swa_kv = pool(swa_pages, PS_SWA)
    ckv_kv = pool(ckv_pages, PS_CKV)
    q = torch.randn(T, H, D, generator=g, device=dev).to(torch.bfloat16)
    sinks = torch.randn(H, generator=g, device=dev, dtype=torch.float32)
    si = torch.full((T, W), -1, dtype=torch.int32, device=dev)
    tl = torch.empty(T, dtype=torch.int32, device=dev)
    swa_base = ckv_base = 0
    for b, s in enumerate(lens):
        n_swa_pg = (s + PS_SWA - 1) // PS_SWA
        n_ckv_rows = (s + COMPRESS - 1) // COMPRESS
        n_ckv_pg = (n_ckv_rows + PS_CKV - 1) // PS_CKV
        swa_map = swa_perm[swa_base : swa_base + n_swa_pg].to(dev)
        ckv_map = ckv_perm[ckv_base : ckv_base + n_ckv_pg].to(dev)
        n_val = min(CKV_COLS, n_ckv_rows)
        picks = torch.sort(torch.randperm(n_ckv_rows, generator=gc)[:n_val]).values.to(
            dev
        )
        ckv_off = (
            ckv_map[torch.div(picks, PS_CKV, rounding_mode="floor")] * PS_CKV
            + picks % PS_CKV
        ).to(torch.int32)
        for j in range(Q):
            t = b * Q + j
            pe = s - 1 - (Q - 1 - j)
            pos = torch.arange(pe - SWA_COLS + 1, pe + 1, device=dev).clamp_min(0)
            si[t, :SWA_COLS] = (
                swa_map[torch.div(pos, PS_SWA, rounding_mode="floor")] * PS_SWA
                + pos % PS_SWA
            ).to(torch.int32)
            si[t, SWA_COLS : SWA_COLS + n_val] = ckv_off
            tl[t] = SWA_COLS + n_val
        swa_base += n_swa_pg
        ckv_base += n_ckv_pg
    return dict(
        q=q,
        swa_kv_cache=swa_kv,
        compressed_kv_cache=ckv_kv,
        sparse_indices=si,
        sparse_topk_lens=tl,
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=dev),
        cum_seq_lens_q=torch.arange(0, T + 1, Q, dtype=torch.int32, device=dev),
        max_q_len=Q,
        sinks=sinks,
    )


def _swa_window_ends(inp):
    """Per-token SWA visibility bound from the public entry's contract:
    clamp(seq_lens[b] - (q_len_b - 1 - q_off), 0, 128)."""
    cum = inp["cum_seq_lens_q"].to("cpu", torch.int64)
    seq = inp["seq_lens"].to("cpu", torch.int64)
    T = inp["q"].shape[0]
    ends = torch.empty(T, dtype=torch.int64)
    for b in range(seq.shape[0]):
        for t in range(int(cum[b]), int(cum[b + 1])):
            q_off = t - int(cum[b])
            q_len_b = int(cum[b + 1]) - int(cum[b])
            ends[t] = max(0, min(128, int(seq[b]) - (q_len_b - 1 - q_off)))
    return ends.to(inp["q"].device)


def _reference(inp):
    """fp32 table-driven sparse-MLA decode: token t attends the valid
    storage offsets of its sparse_indices row (SWA cols into the SWA pool,
    bounded by the seq_lens-derived visibility window, the rest into the
    compressed pool), softmax at 512**-0.5 with the per-head sink folded
    into the denominator."""
    dev = inp["q"].device
    T = inp["q"].shape[0]
    swa_flat = inp["swa_kv_cache"].reshape(-1, D)
    ckv_flat = inp["compressed_kv_cache"].reshape(-1, D)
    qf = inp["q"].to(torch.float32)
    sk = inp["sinks"].to(torch.float32)
    out = torch.empty(T, H, D, dtype=torch.float32, device=dev)
    cols = torch.arange(W, device=dev)
    swa_ends = _swa_window_ends(inp)
    for t0 in range(0, T, 8):
        tab = inp["sparse_indices"][t0 : t0 + 8].to(torch.int64)
        ln = inp["sparse_topk_lens"][t0 : t0 + 8].to(torch.int64)
        c = tab.shape[0]
        valid = (cols.unsqueeze(0) < ln.unsqueeze(1)) & (tab >= 0)
        valid &= (cols.unsqueeze(0) >= SWA_COLS) | (
            cols.unsqueeze(0) < swa_ends[t0 : t0 + c].unsqueeze(1)
        )
        idx = tab.clamp_min(0)
        kv = torch.cat(
            [
                swa_flat[idx[:, :SWA_COLS].reshape(-1)]
                .to(torch.float32)
                .reshape(c, SWA_COLS, D),
                ckv_flat[idx[:, SWA_COLS:].reshape(-1)]
                .to(torch.float32)
                .reshape(c, CKV_COLS, D),
            ],
            dim=1,
        )
        logits = torch.einsum("thd,twd->thw", qf[t0 : t0 + c], kv) * SCALE
        logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
        m = torch.maximum(logits.max(dim=-1).values, sk.view(1, H))
        p = torch.nan_to_num(torch.exp(logits - m.unsqueeze(-1)), nan=0.0)
        denom = p.sum(dim=-1) + torch.exp(sk.view(1, H) - m)
        out[t0 : t0 + c] = torch.einsum("thw,twd->thd", p, kv) / denom.unsqueeze(-1)
    return out.to(torch.bfloat16)


_WS = {}


def _ws():
    if "b" not in _WS:
        _WS["b"] = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return _WS["b"]


def _api(inp, out=None, backend="auto", **kwargs):
    return trtllm_batch_decode_sparse_mla_dsv4(
        query=inp["q"],
        swa_kv_cache=inp["swa_kv_cache"],
        workspace_buffer=_ws(),
        sparse_indices=inp["sparse_indices"],
        compressed_kv_cache=inp["compressed_kv_cache"],
        sparse_topk_lens=inp["sparse_topk_lens"],
        seq_lens=inp["seq_lens"],
        out=out,
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        sinks=inp["sinks"],
        cum_seq_lens_q=inp["cum_seq_lens_q"],
        max_q_len=int(inp["max_q_len"]),
        backend=backend,
        sparse_indices_are_storage_offsets=True,
        **kwargs,
    )


def _assert_close(out, gold):
    diff = (out.float() - gold).abs()
    tol = 0.004 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) >= 0.99, float(diff.max())
    assert float(diff.max()) <= 0.024


@requires_sm103
@pytest.mark.parametrize("B,Q,SEQ", [(1, 6, 32768), (2, 6, 65536)])
def test_specialized_correct_and_routed(B, Q, SEQ):
    inp = _make_inputs(B, Q, SEQ, seed=17)
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    _assert_close(out, gold)
    out2 = _api(inp, backend="specialized")
    assert torch.equal(out, out2)


@requires_sm103
def test_off_allowlist_stays_stock_and_explicit_raises():
    # q_len 1 is not an allowlisted shape class: auto must fall back.
    inp = _make_inputs(2, 1, 4096, seed=3)
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert "shape_class_not_allowlisted" in spec_mod.LAST_FALLBACK_REASONS
    _assert_close(out, _reference(inp).float())
    with pytest.raises(ValueError, match="fallback_reasons"):
        _api(inp, backend="specialized")


@requires_sm103
def test_kill_switch_restores_stock(monkeypatch):
    inp = _make_inputs(1, 6, 32768, seed=23)
    # monkeypatch restores the pre-test value on teardown (os.environ.pop
    # would delete a kill switch the invoker had set for the whole run).
    monkeypatch.setenv(spec_mod.KILL_SWITCH_ENV, "1")
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert spec_mod.LAST_FALLBACK_REASONS[0].startswith("kill_switch")
    _assert_close(out, _reference(inp).float())


@requires_sm103
def test_cuda_graph_replay_matches_eager(monkeypatch):
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    inp = _make_inputs(1, 6, 32768, seed=29)
    out = torch.empty(6, H, D, dtype=torch.bfloat16, device="cuda")
    _api(inp, out=out, backend="specialized")
    eager = out.clone()
    _assert_close(eager, _reference(inp).float())
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    # The eager call and out.clone() were submitted to the current stream;
    # the side-stream warmup writes the same `out`, so it must order after
    # them (the reverse wait below only orders capture after the warmup).
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _api(inp, out=out, backend="specialized")
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        _api(inp, out=out, backend="specialized")
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)


@requires_sm103
def test_invalid_table_entries_never_reach_a_gather():
    """Inactive tail columns (beyond sparse_topk_lens) carry no
    offset-validity contract, and -1 holes may point anywhere: neither may
    produce a global read. Tail columns hold out-of-range garbage
    (INT_MAX-class), pool row 0 holds NaN, and the output must still match
    the fp32 reference exactly as the length/hole mask defines it."""
    inp = _make_inputs(2, 6, 32768, seed=43)
    # poison pool row 0 (both pools): a hole clamped-and-gathered would
    # propagate NaN through the PV MMA even at zero probability
    inp["swa_kv_cache"].reshape(-1, D)[0] = float("nan")
    inp["compressed_kv_cache"].reshape(-1, D)[0] = float("nan")
    # punch -1 holes into the active region of a few rows
    inp["sparse_indices"][3, 5] = -1
    inp["sparse_indices"][7, 60] = -1
    # garbage in the inactive tail: shorten some rows' lens and poison tails
    lens = inp["sparse_topk_lens"]
    lens[1] = 128
    lens[9] = 160
    tbl = inp["sparse_indices"]
    tbl[1, 128:] = 2**31 - 1024  # far out of range
    tbl[9, 160:] = -(2**30)
    # the fp32 reference clamps -1 holes to pool row 0 and gathers it (its
    # 0-probability times a poisoned NaN row is still NaN), so the
    # reference gets CLEAN pools and a sanitized table; the defined output
    # (holes and tails masked) is identical, while the kernel gets the
    # poisoned pools and table
    clean_tbl = tbl.clone()
    clean_tbl[1, 128:] = -1
    clean_tbl[9, 160:] = -1
    clean_swa = inp["swa_kv_cache"].clone()
    clean_swa.reshape(-1, D)[0] = 0
    clean_ckv = inp["compressed_kv_cache"].clone()
    clean_ckv.reshape(-1, D)[0] = 0
    gold = _reference(
        {
            **inp,
            "sparse_indices": clean_tbl,
            "swa_kv_cache": clean_swa,
            "compressed_kv_cache": clean_ckv,
        }
    ).float()
    assert torch.isfinite(gold).all()
    out = _api(inp, backend="specialized")
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    _assert_close(out, gold)


def _direct_inputs(B, seq_lens, n_ckv=0, seed=None, zero_q=True):
    """Hand-built inputs on identity-mapped pools: SWA column c of every row
    is storage offset c (one 256-row SWA page), compressed column 128+k is
    offset k (eight 64-row pages, 512 rows)."""
    dev = torch.device("cuda")
    Q = 6
    T = B * Q
    swa_kv = torch.zeros(1, PS_SWA, D, dtype=torch.bfloat16, device=dev)
    ckv_kv = torch.zeros(8, PS_CKV, D, dtype=torch.bfloat16, device=dev)
    if seed is None:
        q = torch.zeros(T, H, D, dtype=torch.bfloat16, device=dev)
        sinks = torch.zeros(H, dtype=torch.float32, device=dev)
    else:
        g = torch.Generator(device=dev).manual_seed(seed)
        swa_kv.copy_(
            (torch.randn(1, PS_SWA, D, generator=g, device=dev) * 0.5).to(
                torch.bfloat16
            )
        )
        ckv_kv.copy_(
            (torch.randn(8, PS_CKV, D, generator=g, device=dev) * 0.5).to(
                torch.bfloat16
            )
        )
        q = torch.randn(T, H, D, generator=g, device=dev).to(torch.bfloat16)
        sinks = torch.randn(H, generator=g, device=dev, dtype=torch.float32)
    if zero_q and seed is not None:
        q.zero_()
    si = torch.full((T, W), -1, dtype=torch.int32, device=dev)
    si[:, :SWA_COLS] = torch.arange(SWA_COLS, dtype=torch.int32, device=dev)
    if n_ckv:
        si[:, SWA_COLS : SWA_COLS + n_ckv] = torch.arange(
            n_ckv, dtype=torch.int32, device=dev
        )
    tl = torch.full((T,), SWA_COLS + n_ckv, dtype=torch.int32, device=dev)
    return dict(
        q=q,
        swa_kv_cache=swa_kv,
        compressed_kv_cache=ckv_kv,
        sparse_indices=si,
        sparse_topk_lens=tl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        cum_seq_lens_q=torch.arange(0, T + 1, Q, dtype=torch.int32, device=dev),
        max_q_len=Q,
        sinks=sinks,
    )


@requires_sm103
def test_swa_visibility_window_enforced():
    """Reviewer repro: B=1, Q=6, seq_lens=[6], all 128 SWA columns active,
    zero queries/sinks, SWA pool row i filled with i. The public entry
    assigns seq_lens the SWA-visibility-window role, so token j attends
    only SWA columns 0..j (window end = seq_lens[0] - (6 - 1 - j)): the
    output must be sum(0..j) / (j + 2), not mean over all 128 columns."""
    inp = _direct_inputs(1, [6])
    rows = torch.arange(PS_SWA, dtype=torch.float32, device="cuda")
    inp["swa_kv_cache"][0] = rows.unsqueeze(1).expand(PS_SWA, D).to(torch.bfloat16)
    expected = torch.tensor(
        [sum(range(j + 1)) / (j + 2) for j in range(6)], device="cuda"
    )
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)  # auto
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    got = out.float()
    assert torch.allclose(
        got, expected.view(6, 1, 1).expand_as(got), atol=0.01, rtol=0.01
    ), got[:, 0, 0]
    _assert_close(out, _reference(inp).float())
    out2 = _api(inp, backend="specialized")
    assert torch.equal(out, out2)


@requires_sm103
@pytest.mark.parametrize("scale_div", [1, 4])
def test_split_partial_cancellation_preserved(scale_div):
    """Reviewer repro at T=60 (the 4-split route): 640 valid keys, zero
    queries/sinks; keys 0-79 hold 32.25, 80-159 hold 32, 160-319 hold -32,
    320-639 hold 0 (each value across all 512 coords). The analytic result
    is 20/641; a merge that rounds split partials to BF16 cancels to 0."""
    B = 10
    inp = _direct_inputs(B, [32768] * B, n_ckv=CKV_COLS)
    vals = torch.zeros(W, dtype=torch.float32)
    vals[:80] = 32.25 / scale_div
    vals[80:160] = 32.0 / scale_div
    vals[160:320] = -32.0 / scale_div
    swa_flat = inp["swa_kv_cache"].reshape(-1, D)
    ckv_flat = inp["compressed_kv_cache"].reshape(-1, D)
    for c in range(SWA_COLS):
        swa_flat[c] = vals[c]
    for k in range(CKV_COLS):
        ckv_flat[k] = vals[SWA_COLS + k]
    expected = float(vals.sum()) / (W + 1)  # 20/641 (scaled)
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)  # auto
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    got = out.float()
    atol = 3e-4 / scale_div + 1e-5
    assert torch.allclose(got, torch.full_like(got, expected), atol=atol), (
        float(got.min()),
        float(got.max()),
        expected,
    )
    _assert_close(out, _reference(inp).float())


@requires_sm103
@pytest.mark.parametrize("B", [1, 2, 3, 4, 5, 6, 10, 13])
def test_swa_window_sweep_all_routes(B):
    """Mixed open/narrow SWA windows with random data across the batch sizes
    that exercise each kernel route (h8 / fused2 / 4-split main / b13):
    even requests get seq_lens < 128 (per-token narrowing windows), odd
    requests a fully open window; auto must match the windowed reference."""
    seq_lens = [70 if b % 2 == 0 else 4096 for b in range(B)]
    inp = _direct_inputs(B, seq_lens, n_ckv=256, seed=101 + B, zero_q=False)
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)  # auto
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    _assert_close(out, gold)


@requires_sm103
@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_nondefault_hca_flag_is_not_served(backend):
    inp = _direct_inputs(1, [6])
    count = spec_mod.DISPATCH_STATS["specialized"]
    message = "does not accept hca_is_causal" if backend == "auto" else "no_hca_args"
    with pytest.raises(ValueError, match=message):
        _api(inp, backend=backend, hca_is_causal=False)
    assert spec_mod.LAST_FALLBACK_REASONS == ["no_hca_args"]
    assert spec_mod.DISPATCH_STATS["specialized"] == count


@requires_sm103
@pytest.mark.parametrize("backend", ["auto", "specialized"])
@pytest.mark.parametrize(
    "name,value,message",
    [
        ("sparse_topk_lens", 127, "fixed 128 SWA entries"),
        ("sparse_topk_lens", 641, "cannot exceed sparse_indices capacity"),
        ("seq_lens", 5, "per-request query lengths"),
    ],
)
def test_eager_length_validation(monkeypatch, backend, name, value, message):
    inp = _direct_inputs(1, [6])
    inp[name][0] = value
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    before = dict(spec_mod.DISPATCH_STATS)
    with pytest.raises(ValueError, match=message):
        _api(inp, backend=backend)
    assert before == spec_mod.DISPATCH_STATS


@requires_sm103
@pytest.mark.parametrize(
    "operand,reason",
    [
        ("q", "q_not_16B_aligned"),
        ("swa_kv_cache", "swa_pool_not_16B_aligned"),
        ("compressed_kv_cache", "compressed_pool_not_16B_aligned"),
        ("out", "out_not_bf16_q_shaped_contiguous"),
    ],
)
def test_offset_view_alignment(monkeypatch, operand, reason):
    from flashinfer.mla import _core

    inp = _direct_inputs(1, [6])
    inp["out"] = torch.empty_like(inp["q"])
    t = inp[operand]
    inp[operand] = t.new_empty(t.numel() + 1)[1:].view_as(t).copy_(t)
    assert inp[operand].is_contiguous() and inp[operand].data_ptr() % 16 == 2
    mod = spec_mod.get_sparse_mla_decode_dsv4_module()
    with pytest.raises(ValueError, match="16-byte aligned"):
        mod.sparse_mla_decode_dsv4_run(
            inp["q"],
            inp["swa_kv_cache"],
            inp["compressed_kv_cache"],
            inp["sparse_indices"],
            inp["sparse_topk_lens"],
            inp["seq_lens"],
            inp["cum_seq_lens_q"],
            6,
            inp["sinks"],
            inp["out"],
        )
    with pytest.raises(ValueError, match=reason):
        _api(inp, out=inp["out"], backend="specialized")

    # Verify handoff, without assuming a stock cubin supports offset views.
    class ReachedStock(Exception):
        pass

    def stock(device, backend):
        assert device == t.device and backend == "auto"
        raise ReachedStock

    monkeypatch.setattr(_core, "_resolve_dsv4_sparse_mla_backend", stock)
    with pytest.raises(ReachedStock):
        _api(inp, out=inp["out"])
    assert [reason] == spec_mod.LAST_FALLBACK_REASONS


@requires_sm103
@pytest.mark.parametrize("error", [FileNotFoundError, RuntimeError])
def test_unavailable_specialization_continues_on_stock(monkeypatch, error):
    from types import SimpleNamespace

    inp = _make_inputs(1, 6, 1024, seed=71)
    gold = _reference(inp).float()
    attempts = []

    def build():
        attempts.append(1)
        raise error(
            "artifact missing with JIT disabled"
            if error is FileNotFoundError
            else "failed build"
        )

    monkeypatch.setattr(spec_mod, "_MODULE", None)
    monkeypatch.setattr(spec_mod, "_MODULE_ERROR", None)
    monkeypatch.setattr(
        spec_mod,
        "gen_sparse_mla_decode_dsv4_module",
        lambda: SimpleNamespace(build_and_load=build),
    )
    before = dict(spec_mod.DISPATCH_STATS)
    for i in (1, 2):
        _assert_close(_api(inp), gold)
        assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
        assert spec_mod.DISPATCH_STATS["fallback"] == before["fallback"] + i
        assert spec_mod.LAST_FALLBACK_REASONS[0].startswith(
            "specialized_module_unavailable:" + error.__name__
        )
    with pytest.raises(RuntimeError, match=error.__name__):
        _api(inp, backend="specialized")
    assert attempts == [1]


@pytest.mark.parametrize("fail", [False, True])
def test_concurrent_cold_module_load(monkeypatch, fail):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace

    start = Barrier(4)
    attempts = []
    module = object()

    def build():
        attempts.append(1)
        if fail:
            raise RuntimeError("cold failure")
        return module

    def load(_):
        start.wait()
        return spec_mod.get_sparse_mla_decode_dsv4_module(required=False)

    monkeypatch.setattr(spec_mod, "_MODULE", None)
    monkeypatch.setattr(spec_mod, "_MODULE_ERROR", None)
    monkeypatch.setattr(spec_mod, "_BUILT", {"ready": False})
    monkeypatch.setattr(
        spec_mod,
        "gen_sparse_mla_decode_dsv4_module",
        lambda: SimpleNamespace(build_and_load=build),
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(load, range(4)))
    assert attempts == [1]
    assert all(result is (None if fail else module) for result in results)


def test_module_build_failure_is_cached(monkeypatch):
    """A failed build_and_load must be attempted once: later calls re-raise
    the remembered error instead of re-running the JIT build, so repeated
    backend="auto" fallbacks stay cheap."""
    calls = {"n": 0}

    class _FailingSpec:
        def build_and_load(self):
            calls["n"] += 1
            raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(spec_mod, "_MODULE", None)
    monkeypatch.setattr(spec_mod, "_MODULE_ERROR", None)
    monkeypatch.setattr(
        spec_mod, "gen_sparse_mla_decode_dsv4_module", lambda: _FailingSpec()
    )
    for _ in range(3):
        with pytest.raises(RuntimeError, match="synthetic build failure"):
            spec_mod.get_sparse_mla_decode_dsv4_module()
    assert calls["n"] == 1
