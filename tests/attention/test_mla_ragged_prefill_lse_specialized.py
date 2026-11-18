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

Tests for the SM103 "specialized" LSE-emitting ragged MLA prefill backend
of trtllm_ragged_attention_deepseek: out + base-2 LSE correctness vs an
fp32 reference at measured serving classes (incl. the causal class),
backend-dispatch semantics (auto routes return_lse=True on-allowlist;
return_lse=False keeps stock; explicit never swapped; kill switch), and the
uniform CUDA-graph capture bail.
"""

import os
from unittest import mock

import pytest
import torch

import flashinfer.mla_ragged_prefill_dsr1_cuda as spec_mod
from flashinfer.prefill import trtllm_ragged_attention_deepseek

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized LSE ragged MLA prefill kernel is SM103-only",
)

HQ, D_QK, D_V = 12, 192, 128
SCALE = float(D_QK) ** -0.5
LOG2E = 1.4426950408889634


def _make_lens(total, n, mx):
    if n == 1:
        return [total]
    rest = total - mx
    base, rem = divmod(rest, n - 1)
    return [mx] + [base + (1 if i < rem else 0) for i in range(n - 1)]


def _cu(lens, dev):
    cu = [0]
    for ln in lens:
        cu.append(cu[-1] + ln)
    return torch.tensor(cu, dtype=torch.int32, device=dev)


def _randn_fp8(shape, g, dev, scale=0.5):
    out = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    flat = out.view(-1, shape[-1])
    step = max(1, (1 << 22) // max(shape[-1], 1))
    for i in range(0, flat.shape[0], step):
        n = min(step, flat.shape[0] - i)
        flat[i : i + n] = (
            torch.randn((n, shape[-1]), generator=g, device=dev) * scale
        ).to(torch.float8_e4m3fn)
    return out


def _make_inputs(B, TQ, TKV, MAXQ, MAXKV, causal, seed=0):
    dev = torch.device("cuda")
    q_lens = _make_lens(TQ, B, MAXQ)
    kv_lens = q_lens if causal else _make_lens(TKV, B, MAXKV)
    g = torch.Generator(device=dev).manual_seed(seed)
    return dict(
        q=_randn_fp8((TQ, HQ, D_QK), g, dev),
        k=_randn_fp8((TKV, HQ, D_QK), g, dev),
        v=_randn_fp8((TKV, HQ, D_V), g, dev),
        q_lens=torch.tensor(q_lens, dtype=torch.int32, device=dev),
        kv_lens=torch.tensor(kv_lens, dtype=torch.int32, device=dev),
        cu_q=_cu(q_lens, dev),
        cu_kv=_cu(kv_lens, dev),
        max_q_len=MAXQ,
        max_kv_len=MAXKV,
        causal=bool(causal),
        B=B,
    )


def _reference(inp):
    """fp32 ragged MLA prefill; returns (out bf16, lse fp32 base-2 with the
    bmm1 scale in the exponent — the return_lse=True convention)."""
    dev = inp["q"].device
    causal = inp["causal"]
    cuq = inp["cu_q"].to(torch.int64).tolist()
    cuk = cuq if causal else inp["cu_kv"].to(torch.int64).tolist()
    outs, lses = [], []
    for r in range(inp["B"]):
        q0, q1 = cuq[r], cuq[r + 1]
        k0, k1 = cuk[r], cuk[r + 1]
        n, L = q1 - q0, k1 - k0
        kf = inp["k"][k0:k1].to(torch.float32)
        vf = inp["v"][k0:k1].to(torch.float32)
        step = max(1, (1 << 22) // max(L, 1))
        for i in range(0, n, step):
            j = min(n, i + step)
            qf = inp["q"][q0 + i : q0 + j].to(torch.float32)
            sc = torch.einsum("qhd,lhd->qhl", qf, kf) * SCALE
            if causal:
                pos = torch.arange(i, j, device=dev)
                l_idx = torch.arange(L, device=dev)
                sc = sc.masked_fill(
                    l_idx.view(1, 1, -1) > pos.view(-1, 1, 1), float("-inf")
                )
            lses.append(torch.logsumexp(sc, dim=-1) * LOG2E)
            outs.append(torch.einsum("qhl,lhd->qhd", torch.softmax(sc, dim=-1), vf))
    return (torch.cat(outs, 0).to(torch.bfloat16), torch.cat(lses, 0).to(torch.float32))


_WS = {}


def _ws():
    if "b" not in _WS:
        _WS["b"] = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return _WS["b"]


def _api(inp, backend="auto", return_lse=True, out=None):
    return trtllm_ragged_attention_deepseek(
        query=inp["q"],
        key=inp["k"],
        value=inp["v"],
        workspace_buffer=_ws(),
        seq_lens=inp["kv_lens"],
        max_q_len=inp["max_q_len"],
        max_kv_len=inp["max_kv_len"],
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=inp["B"],
        window_left=-1,
        cum_seq_lens_q=inp["cu_q"],
        cum_seq_lens_kv=inp["cu_kv"],
        enable_pdl=False,
        is_causal=inp["causal"],
        return_lse=return_lse,
        out=out,
        backend=backend,
    )


def _check(o, l, gold_o, gold_l, out_atol, out_cap, lse_atol, lse_cap):
    do = (o.float() - gold_o.float()).abs()
    assert (
        float((do <= out_atol + 0.05 * gold_o.float().abs()).float().mean()) >= 0.99
    ), float(do.max())
    assert float(do.max()) <= out_cap
    dl = (l - gold_l).abs()
    assert float((dl <= lse_atol).float().mean()) >= 0.99, float(dl.max())
    assert float(dl.max()) <= lse_cap


# (B, TQ, TKV, MAXQ, MAXKV, causal, out_atol, out_cap, lse_atol, lse_cap) —
# tolerances are the measured-class authored bounds (std-audited).
CASES = [
    (1, 128, 12416, 128, 12416, 0, 0.0016, 0.0048, 0.0019, 0.0057),
    (1, 256, 15872, 256, 15872, 0, 0.0014, 0.0042, 0.0018, 0.0054),
    # causal lse cap: measured max |err| is 1.5e-3 on this class (per-row
    # re-time + 4-fresh-seed battery on an internal GB300 node); 0.02 keeps
    # >10x margin without admitting a broken LSE.
    (1, 16128, 16128, 16128, 16128, 1, 0.002, 0.037, 0.0047, 0.02),
]


@requires_sm103
@pytest.mark.parametrize("B,TQ,TKV,MAXQ,MAXKV,causal,oa,oc,la,lc", CASES)
def test_specialized_correct_and_routed(
    B, TQ, TKV, MAXQ, MAXKV, causal, oa, oc, la, lc
):
    inp = _make_inputs(B, TQ, TKV, MAXQ, MAXKV, causal, seed=21)
    gold_o, gold_l = _reference(inp)
    before = dict(spec_mod.DISPATCH_STATS)
    o, l = _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    _check(o, l, gold_o, gold_l, oa, oc, la, lc)
    o2, l2 = _api(inp, backend="specialized")
    assert torch.equal(o, o2) and torch.equal(l, l2)


@requires_sm103
def test_dispatch_semantics():
    inp = _make_inputs(1, 128, 12416, 128, 12416, 0, seed=3)
    # return_lse=False keeps the stock path (per-measured-row routing)
    before = dict(spec_mod.DISPATCH_STATS)
    _api(inp, return_lse=False)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert "lse_not_requested" in spec_mod.LAST_FALLBACK_REASONS
    with pytest.raises(ValueError, match="fallback_reasons"):
        _api(inp, backend="specialized", return_lse=False)
    # off-allowlist shape stays stock; explicit raises
    off = _make_inputs(1, 128, 8192, 128, 8192, 0, seed=5)
    before = dict(spec_mod.DISPATCH_STATS)
    _api(off)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert "shape_not_in_measured_allowlist" in spec_mod.LAST_FALLBACK_REASONS
    with pytest.raises(ValueError, match="fallback_reasons"):
        _api(off, backend="specialized")
    # kill switch restores stock on an allowlisted call
    os.environ[spec_mod.KILL_SWITCH_ENV] = "1"
    try:
        before = dict(spec_mod.DISPATCH_STATS)
        o, l = _api(inp)
        assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
        assert spec_mod.LAST_FALLBACK_REASONS[0].startswith("kill_switch")
        gold_o, gold_l = _reference(inp)
        _check(o, l, gold_o, gold_l, 0.0048, 0.02, 0.01, 0.05)  # sanity bound
        with pytest.raises(ValueError, match="fallback_reasons"):
            _api(inp, backend="specialized")
    finally:
        os.environ.pop(spec_mod.KILL_SWITCH_ENV, None)


@requires_sm103
def test_deterministic():
    inp = _make_inputs(1, 256, 15872, 256, 15872, 0, seed=7)
    o1, l1 = _api(inp)
    o2, l2 = _api(inp)
    o3, l3 = _api(inp)
    assert torch.equal(o1, o2) and torch.equal(o2, o3)
    assert torch.equal(l1, l2) and torch.equal(l2, l3)


@requires_sm103
def test_graph_capture_bails_to_stock():
    # The specialized route uniformly bails under CUDA-graph capture (the
    # stock path serves capture); verify the guard reports the bail reason.
    inp = _make_inputs(1, 128, 12416, 128, 12416, 0, seed=9)
    with mock.patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=True
    ):
        reasons = spec_mod.supported_with_reasons(
            inp["q"],
            inp["k"],
            inp["v"],
            _ws(),
            inp["kv_lens"],
            inp["max_q_len"],
            inp["max_kv_len"],
            inp["cu_q"],
            inp["cu_kv"],
            inp["B"],
            None,
            extras=dict(
                is_causal=False,
                return_lse=True,
                lse=None,
                window_left=-1,
                attention_sinks=None,
                skip_softmax_threshold_scale_factor=None,
                sage_attn_sfs=(),
                num_elts_per_sage_attn_blk=(),
                use_fp16_softmax=None,
                uses_spcompress=None,
                q_seq_lens_cpu=None,
                kv_seq_lens_cpu=None,
                bmm1_scale=SCALE,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
            ),
        )
    assert reasons == ["cuda_graph_capture_bail"]


@requires_sm103
def test_generalized_entry_parity_and_dispatch():
    """ragged_attention_deepseek is the vendor-neutral front: same enum,
    same dispatch semantics, bitwise parity with the direct entry,
    return_lse semantics preserved."""
    from flashinfer.prefill import (
        RAGGED_DEEPSEEK_BACKENDS,
        TRTLLM_RAGGED_DEEPSEEK_BACKENDS,
        ragged_attention_deepseek,
    )

    assert RAGGED_DEEPSEEK_BACKENDS == TRTLLM_RAGGED_DEEPSEEK_BACKENDS
    inp = _make_inputs(1, 128, 12416, 128, 12416, 0, seed=17)
    kwargs = dict(
        query=inp["q"],
        key=inp["k"],
        value=inp["v"],
        workspace_buffer=_ws(),
        seq_lens=inp["kv_lens"],
        max_q_len=inp["max_q_len"],
        max_kv_len=inp["max_kv_len"],
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=inp["B"],
        window_left=-1,
        cum_seq_lens_q=inp["cu_q"],
        cum_seq_lens_kv=inp["cu_kv"],
        enable_pdl=False,
        is_causal=inp["causal"],
        return_lse=True,
    )
    # auto on the allowlist routes to specialized through the front
    before = dict(spec_mod.DISPATCH_STATS)
    gen_o, gen_l = ragged_attention_deepseek(**kwargs)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1
    direct_o, direct_l = _api(inp)
    assert torch.equal(gen_o, direct_o) and torch.equal(gen_l, direct_l)
    # explicit backends behave identically through the front
    spec_o, spec_l = ragged_attention_deepseek(**{**kwargs, "backend": "specialized"})
    assert torch.equal(spec_o, gen_o) and torch.equal(spec_l, gen_l)
    stock_o, stock_l = ragged_attention_deepseek(**{**kwargs, "backend": "trtllm-gen"})
    ds_o, ds_l = _api(inp, backend="trtllm-gen")
    assert torch.equal(stock_o, ds_o) and torch.equal(stock_l, ds_l)
    # return_lse=False parity through the front: this call form leaves the
    # specialized allowlist (it keeps the stock path), so compare the front
    # against the direct entry at the SAME arguments, not against gen_o
    gen_only = ragged_attention_deepseek(**{**kwargs, "return_lse": False})
    direct_only = _api(inp, return_lse=False)
    assert torch.equal(gen_only, direct_only)
    # invalid backend raises through the front too
    with pytest.raises(ValueError, match="backend must be one of"):
        ragged_attention_deepseek(**{**kwargs, "backend": "bogus"})
