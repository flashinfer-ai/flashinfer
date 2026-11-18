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
"""

import pytest
import torch

from flashinfer.mla._sparse_mla_sm90 import (
    BACKENDS,
    _backend_order,
    _dispatch_table,
    _reset_sparse_mla_sm90_stats,
    _sparse_mla_sm90_stats,
)
from flashinfer.mla import (
    get_sparse_mla_sm90_workspace_size_bytes,
    sparse_mla_decode_sm90,
)

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISABLE_ENV_LEGACY = "FLASHINFER_DISABLE_SPARSE_MLA_SM90"
H, CKV, KPE, PAGE, BPT, TOPK = 8, 512, 64, 64, 656, 2048
D = CKV + KPE
SM_SCALE = 256**-0.5


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV_LEGACY, raising=False)
    _reset_sparse_mla_sm90_stats()
    yield
    _reset_sparse_mla_sm90_stats()


def _sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")


# --------------------------------------------------------------------------
# Backend selection surface (#5667 pattern, GPU-free)
# --------------------------------------------------------------------------


def test_backends_tuple_is_plain_strings_with_specialized():
    assert BACKENDS == ("auto", "specialized", "reference")
    assert all(isinstance(b, str) for b in BACKENDS)


def test_dispatch_table_loads_and_matches_guard_constants():
    table = _dispatch_table()
    assert table is not None
    assert table["kill_switch"] == DISABLE_ENV
    assert "specialized" in table["backends"]
    op = table["ops"]["sparse_mla_decode"]
    assert op["num_heads"] == H
    assert op["qk_dim"] == D
    assert op["page_size"] == PAGE
    assert op["bytes_per_token"] == BPT
    assert op["topk"] == TOPK
    assert op["envelope"]["max_token_rows"] == 64


def test_backend_order_never_swaps_an_explicit_backend():
    # Explicit backends never fall back: "specialized" runs the fused kernel
    # or raises; only "auto" routes around a refusal.
    assert _backend_order("specialized", False) == ("specialized",)
    assert _backend_order("reference", True) == ("reference",)
    assert _backend_order("auto", True) == ("specialized", "reference")
    assert _backend_order("auto", False) == ("reference",)
    assert _backend_order("fa2", True) == ()


def test_stats_structure():
    stats = _sparse_mla_sm90_stats()
    assert set(stats["backend_dispatches"]) == {"specialized", "reference"}
    assert stats["backends"] == BACKENDS
    assert "fallback_reasons" in stats and "dispatch_table_loaded" in stats


def test_unknown_backend_raises_gpu_free():
    q = torch.zeros((1, 1, H, D), dtype=torch.bfloat16)
    kv = torch.zeros((1, PAGE, BPT), dtype=torch.uint8)
    idx = torch.full((1, 1, TOPK), -1, dtype=torch.int32)
    sl = torch.ones((1,), dtype=torch.int32)
    ws = torch.zeros((0,), dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="never silently swapped"):
        sparse_mla_decode_sm90(q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="fa2")


# --------------------------------------------------------------------------
# Device tests (SM90)
# --------------------------------------------------------------------------


def _make_case(B=2, QL=3, kv_lens=(500, 130), seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    pprs = [(ln + PAGE - 1) // PAGE for ln in kv_lens]
    num_pages = sum(pprs)
    total = num_pages * PAGE
    ntiles = CKV // 128
    master = (
        torch.randn((total, D), generator=g, device=device, dtype=torch.float32) * 0.5
    )
    nope = master[:, :CKV].reshape(total, ntiles, 128)
    scales = (nope.abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(1e-8)
    qn = (nope / scales).to(torch.float8_e4m3fn)
    rows = torch.empty((total, BPT), dtype=torch.uint8, device=device)
    rows[:, :CKV] = qn.reshape(total, CKV).view(torch.uint8)
    rows[:, CKV : CKV + 4 * ntiles] = (
        scales.to(torch.float32)
        .reshape(total, ntiles)
        .contiguous()
        .view(torch.uint8)
        .reshape(total, 4 * ntiles)
    )
    rows[:, CKV + 4 * ntiles :] = (
        master[:, CKV:]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(total, 2 * KPE)
    )
    kv_cache = rows.reshape(num_pages, PAGE, BPT)

    starts = [0]
    for p in pprs[:-1]:
        starts.append(starts[-1] + p)
    idx = torch.full((B, QL, TOPK), -1, dtype=torch.int32, device=device)
    for b in range(B):
        n = min(kv_lens[b], TOPK)
        idx[b, :, :n] = starts[b] * PAGE + torch.arange(
            n, dtype=torch.int32, device=device
        )
    q = (
        torch.randn((B, QL, H, D), generator=g, device=device, dtype=torch.float32)
        * 0.5
    ).to(torch.bfloat16)
    sl = torch.tensor([min(k, TOPK) for k in kv_lens], dtype=torch.int32, device=device)
    nq = B * QL
    ws_bytes = get_sparse_mla_sm90_workspace_size_bytes(B, QL) if nq <= 64 else 0
    ws = torch.zeros((ws_bytes or 1,), dtype=torch.uint8, device=device)
    return q, kv_cache, idx, sl, ws


def _ref(q, kv_cache, idx):
    flat = kv_cache.reshape(-1, BPT)
    B, QL = q.shape[:2]
    ntiles = CKV // 128
    out = torch.zeros((B, QL, H, CKV), dtype=torch.float32, device=q.device)
    qf = q.float()
    for b in range(B):
        for j in range(QL):
            sel = idx[b, j].long()
            sel = sel[sel >= 0]
            if sel.numel() == 0:
                continue
            rows = flat[sel]
            nope = (
                rows[:, :CKV]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .float()
                .reshape(-1, ntiles, 128)
            )
            sc = (
                rows[:, CKV : CKV + 4 * ntiles]
                .contiguous()
                .view(torch.float32)
                .reshape(-1, ntiles, 1)
            )
            ckv = (nope * sc).reshape(-1, CKV)
            kpe = rows[:, CKV + 4 * ntiles :].contiguous().view(torch.bfloat16).float()
            logit = (qf[b, j, :, :CKV] @ ckv.T + qf[b, j, :, CKV:] @ kpe.T) * SM_SCALE
            out[b, j] = torch.softmax(logit, -1) @ ckv
    return out.to(torch.bfloat16)


def _matched(y, ref, rtol=0.04, atol=0.02):
    yf, rf = y.float(), ref.float()
    return float(((yf - rf).abs() <= (atol + rtol * rf.abs())).float().mean().item())


def test_specialized_matches_reference_backend():
    _sm90_or_skip()
    q, kv, idx, sl, ws = _make_case()
    y_spec = sparse_mla_decode_sm90(
        q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="specialized"
    )
    stats = _sparse_mla_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    y_ref = sparse_mla_decode_sm90(
        q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="reference"
    )
    stats = _sparse_mla_sm90_stats()
    assert stats["backend_dispatches"]["reference"] == 1, stats
    ref = _ref(q, kv, idx)
    assert _matched(y_spec, ref) >= 0.98
    assert _matched(y_ref, ref) >= 0.999  # same math as the reference
    # determinism
    y2 = sparse_mla_decode_sm90(
        q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="specialized"
    )
    assert torch.equal(y_spec, y2)


def test_auto_routes_on_allowlist_and_reference_never_fused():
    _sm90_or_skip()
    q, kv, idx, sl, ws = _make_case()
    sparse_mla_decode_sm90(q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="auto")
    stats = _sparse_mla_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    # explicit "reference" never runs the fused kernel
    _reset_sparse_mla_sm90_stats()
    sparse_mla_decode_sm90(q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="reference")
    stats = _sparse_mla_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 1, stats


@pytest.mark.parametrize("env", [DISABLE_ENV, DISABLE_ENV_LEGACY])
def test_kill_switch_forces_the_composable_path(env, monkeypatch):
    _sm90_or_skip()
    q, kv, idx, sl, ws = _make_case()
    monkeypatch.setenv(env, "1")
    y = sparse_mla_decode_sm90(q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="auto")
    stats = _sparse_mla_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 1, stats
    assert _matched(y, _ref(q, kv, idx)) >= 0.999
    # explicit "specialized" under the kill switch raises (never falls back)
    with pytest.raises(NotImplementedError, match="never falls back"):
        sparse_mla_decode_sm90(
            q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="specialized"
        )


def test_specialized_off_envelope_raises():
    _sm90_or_skip()
    # token rows above the qualified envelope (65 > 64): explicit
    # "specialized" raises with the refusal reason — never falls back.
    q, kv, idx, sl, ws = _make_case(B=13, QL=5, kv_lens=tuple([200] * 13))
    with pytest.raises(NotImplementedError, match="never falls back"):
        sparse_mla_decode_sm90(
            q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="specialized"
        )
    stats = _sparse_mla_sm90_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 0, stats
    assert any("specialized" in r for r in stats["fallback_reasons"]), stats
    # "auto" routes the same call to the composable reference path instead.
    y = sparse_mla_decode_sm90(q, kv, idx, sl, ws, sm_scale=SM_SCALE, backend="auto")
    assert _matched(y, _ref(q, kv, idx)) >= 0.999
