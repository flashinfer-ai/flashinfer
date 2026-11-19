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
from flashinfer.prefill import (
    ragged_attention_deepseek,
    trtllm_ragged_attention_deepseek,
)

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
    # the generic front is the ONLY surface exposing the specialized
    # backend (the vendor-named entry stays stock)
    return ragged_attention_deepseek(
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
@pytest.mark.parametrize(
    "TQ,TKV",
    [(128, 12416), (256, 15872), (2101, 66304)],
)
@pytest.mark.parametrize("frac", ["half", "17"])
def test_runtime_kv_extent_shorter_than_buffer(TQ, TKV, frac):
    # Capacity-reuse form of an allowlisted class: K/V sized to the class's
    # full capacity while cu_kv/seq_lens describe a shorter active prefix.
    # The specialized route must honor the runtime extent; the inactive tail
    # (poisoned with V=1 here) must not contribute to the output or the LSE.
    dev = torch.device("cuda")
    active = 17 if frac == "17" else TKV // 2 + 17  # off the BN tile grid
    q = torch.zeros((TQ, HQ, D_QK), dtype=torch.float8_e4m3fn, device=dev)
    k = torch.zeros((TKV, HQ, D_QK), dtype=torch.float8_e4m3fn, device=dev)
    v = torch.ones((TKV, HQ, D_V), device=dev).to(torch.float8_e4m3fn)
    v[:active] = 0.0
    inp = dict(
        q=q,
        k=k,
        v=v,
        kv_lens=torch.tensor([active], dtype=torch.int32, device=dev),
        cu_q=_cu([TQ], dev),
        cu_kv=_cu([active], dev),
        max_q_len=TQ,
        max_kv_len=TKV,
        B=1,
        causal=0,
    )
    before = dict(spec_mod.DISPATCH_STATS)
    o, l = _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    # All scores are 0, so attention is uniform over the ACTIVE prefix:
    # O = mean(V[:active]) = 0 and LSE = log2(active) exactly.  With the full
    # buffer attended instead, O = (TKV - active) / TKV and LSE = log2(TKV).
    assert float(o.float().abs().max()) <= 1e-3
    gold_lse = torch.log2(torch.tensor(float(active))).item()
    assert float((l - gold_lse).abs().max()) <= 0.02, float(l.max())


@requires_sm103
def test_auto_falls_back_when_module_unavailable():
    # A failed specialized-module build/load (e.g. a cache-only deployment
    # without the compiled artifact) must be a fallback reason under "auto",
    # not an exception: the stock path serves the same call.  The failure is
    # cached, so the build is attempted once, not per call.  An explicit
    # backend="specialized" call still raises, with the reason included.
    inp = _make_inputs(1, 128, 12416, 128, 12416, 0, seed=11)
    gold_o, gold_l = _reference(inp)

    def _boom():
        raise RuntimeError("specialized module not in the JIT cache")

    before = dict(spec_mod.DISPATCH_STATS)
    saved_failure = list(spec_mod._MODULE_LOAD_FAILURE)
    spec_mod._MODULE_LOAD_FAILURE.clear()
    try:
        with mock.patch.object(
            spec_mod, "get_mla_ragged_prefill_dsr1_module", side_effect=_boom
        ) as loader:
            o, l = _api(inp)
            assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
            assert any(
                r.startswith("specialized_module_unavailable:")
                for r in spec_mod.LAST_FALLBACK_REASONS
            ), spec_mod.LAST_FALLBACK_REASONS
            _check(o, l, gold_o, gold_l, 0.0048, 0.02, 0.01, 0.05)  # stock served
            _api(inp)
            assert loader.call_count == 1  # cached failure: one build attempt
            with pytest.raises(ValueError, match="specialized_module_unavailable"):
                _api(inp, backend="specialized")
    finally:
        spec_mod._MODULE_LOAD_FAILURE.clear()
        spec_mod._MODULE_LOAD_FAILURE.extend(saved_failure)
    # with the real module restored, auto serves specialized again
    before = dict(spec_mod.DISPATCH_STATS)
    _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1


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
    """ragged_attention_deepseek is a front over NAMED real backends:
    "specialized" links directly to the agent-generated kernel, "trtllm"/
    "cute-dsl" are the trtllm-named entry's paths, "auto" only selects
    among names; return_lse semantics preserved."""
    from flashinfer.prefill import (
        RAGGED_DEEPSEEK_BACKENDS,
        ragged_attention_deepseek,
    )

    assert RAGGED_DEEPSEEK_BACKENDS == ("auto", "specialized", "trtllm", "cute-dsl")
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
    # "auto" selects "specialized" on its allowlist (dispatch stats move)
    before = dict(spec_mod.DISPATCH_STATS)
    gen_o, gen_l = ragged_attention_deepseek(**kwargs)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1
    # the direct specialized name produces the same bits
    spec_o, spec_l = ragged_attention_deepseek(**{**kwargs, "backend": "specialized"})
    assert torch.equal(spec_o, gen_o) and torch.equal(spec_l, gen_l)
    # the named "trtllm" backend equals the entry's trtllm-gen path
    stock_o, stock_l = ragged_attention_deepseek(**{**kwargs, "backend": "trtllm"})
    ds_o, ds_l = _api(inp, backend="trtllm")
    assert torch.equal(stock_o, ds_o) and torch.equal(stock_l, ds_l)
    # the vendor-named entry stays STOCK: it never routes to the
    # specialized kernel. (Stock behavior treats any non-"cute-dsl" name
    # as the trtllm-gen path — unchanged by this PR.)
    before = dict(spec_mod.DISPATCH_STATS)
    ev_o, ev_l = trtllm_ragged_attention_deepseek(
        **{**kwargs, "backend": "specialized"}
    )
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert torch.equal(ev_o, ds_o) and torch.equal(ev_l, ds_l)
    # return_lse=False leaves the specialized allowlist: "auto" must select
    # "trtllm" (no behavior of its own) and match the entry at the SAME args
    before = dict(spec_mod.DISPATCH_STATS)
    gen_only = ragged_attention_deepseek(**{**kwargs, "return_lse": False})
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    direct_only = _api(inp, return_lse=False)
    assert torch.equal(gen_only, direct_only)
    # explicit "specialized" off-allowlist raises; invalid name raises
    with pytest.raises(ValueError, match="fallback_reasons"):
        ragged_attention_deepseek(
            **{**kwargs, "backend": "specialized", "return_lse": False}
        )
    with pytest.raises(ValueError, match="backend must be one of"):
        ragged_attention_deepseek(**{**kwargs, "backend": "bogus"})


@requires_sm103
def test_uniform_attention_unbiased():
    """Adversarial uniform-attention construction (the former disclosure):
    constant k, v == 1 exactly, constant q scanned across three octaves of
    the internal weight scale, so every attention weight is identical and
    the exact output is v's row-mean (== 1).  Before the non-causal routes
    normalized by the quantized-weight sum this construction showed a
    systematic PV bias up to ~4.7% (the e4m3 numerator against the exact
    normalizer); with the output normalizer built from the SAME e4m3
    weights the numerator uses, the ratio is exact for uniform rows and the
    worst bias over the scan must collapse to output-rounding level.  The
    LSE keeps the exact sum and must stay within the class bound."""
    dev = torch.device("cuda")
    B, TQ, TKV, MAXQ, MAXKV = 1, 128, 12416, 128, 12416
    inp = _make_inputs(B, TQ, TKV, MAXQ, MAXKV, 0, seed=5)
    worst_bias, worst_lse = 0.0, 0.0
    for i in range(25):
        cq = 0.30 + 0.0125 * i
        inp["q"] = torch.full((TQ, HQ, D_QK), cq, device=dev).to(torch.float8_e4m3fn)
        inp["k"] = torch.full((TKV, HQ, D_QK), 0.5, device=dev).to(torch.float8_e4m3fn)
        inp["v"] = torch.full((TKV, HQ, D_V), 1.0, device=dev).to(torch.float8_e4m3fn)
        gold_o, gold_l = _reference(inp)
        o, l = _api(inp, backend="specialized")
        rel = float(((o.float() - gold_o.float()).abs() / gold_o.float().abs()).max())
        worst_bias = max(worst_bias, rel)
        worst_lse = max(worst_lse, float((l - gold_l).abs().max()))
    assert worst_bias <= 0.01, worst_bias  # was ~4.7% before the fix
    assert worst_lse <= 0.0057, worst_lse  # the class lse cap (unchanged)
