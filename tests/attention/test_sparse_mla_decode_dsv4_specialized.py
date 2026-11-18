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

import os

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


def _reference(inp):
    """fp32 table-driven sparse-MLA decode: token t attends the valid
    storage offsets of its sparse_indices row (SWA cols into the SWA pool,
    the rest into the compressed pool), softmax at 512**-0.5 with the
    per-head sink folded into the denominator."""
    dev = inp["q"].device
    T = inp["q"].shape[0]
    swa_flat = inp["swa_kv_cache"].reshape(-1, D)
    ckv_flat = inp["compressed_kv_cache"].reshape(-1, D)
    qf = inp["q"].to(torch.float32)
    sk = inp["sinks"].to(torch.float32)
    out = torch.empty(T, H, D, dtype=torch.float32, device=dev)
    cols = torch.arange(W, device=dev)
    for t0 in range(0, T, 8):
        tab = inp["sparse_indices"][t0 : t0 + 8].to(torch.int64)
        ln = inp["sparse_topk_lens"][t0 : t0 + 8].to(torch.int64)
        c = tab.shape[0]
        valid = (cols.unsqueeze(0) < ln.unsqueeze(1)) & (tab >= 0)
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


def _api(inp, out=None, backend="auto"):
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
def test_kill_switch_restores_stock():
    inp = _make_inputs(1, 6, 32768, seed=23)
    os.environ[spec_mod.KILL_SWITCH_ENV] = "1"
    try:
        before = dict(spec_mod.DISPATCH_STATS)
        out = _api(inp)
        assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
        assert spec_mod.LAST_FALLBACK_REASONS[0].startswith("kill_switch")
        _assert_close(out, _reference(inp).float())
    finally:
        os.environ.pop(spec_mod.KILL_SWITCH_ENV, None)


@requires_sm103
def test_cuda_graph_replay_matches_eager():
    inp = _make_inputs(1, 6, 32768, seed=29)
    out = torch.empty(6, H, D, dtype=torch.bfloat16, device="cuda")
    _api(inp, out=out)  # eager warmup builds the module before capture
    eager = out.clone()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        _api(inp, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        _api(inp, out=out)
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
    lens[1] = 40
    lens[9] = 100
    tbl = inp["sparse_indices"]
    tbl[1, 40:] = 2**31 - 1024  # far out of range
    tbl[9, 100:] = -(2**30)
    # the fp32 reference gathers unmasked entries clamped only at 0, so give
    # it a SANITIZED table (masked entries -> -1; they cannot affect the
    # defined output) while the kernel gets the poisoned one
    clean_tbl = tbl.clone()
    clean_tbl[1, 40:] = -1
    clean_tbl[9, 100:] = -1
    gold = _reference({**inp, "sparse_indices": clean_tbl}).float()
    assert torch.isfinite(gold).all()
    out = _api(inp, backend="specialized")
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    _assert_close(out, gold)
