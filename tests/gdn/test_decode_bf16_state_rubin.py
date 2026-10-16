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

"""Rubin-tuned T=1 GDN decode kernel with BF16 state (gdn_decode_bf16_state_rubin), SM107 only.

Where the generic dispatcher picks its wide_vec T=1 kernel (B * HV >= 512) the math and FMA order
are identical, so outputs and the updated state must be bitwise equal. At smaller work sizes the
generic dispatcher runs a different kernel (mtp_ilp4) with another reduction order, so there the
comparison uses a bf16 tolerance. The instruction cuts (on by default) change only address math,
load/store width, register placement and scheduling, so turning them all off must give the same
bits. The fp32-reference check is test_decode_delta_rule.py::test_gdn_decode_bf16_state_t1_kernel.
"""

import math

import pytest
import torch

import flashinfer.gdn_decode as gdn_decode
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule as generic
from flashinfer.gdn_kernels.gdn_decode_bf16_state_rubin import gated_delta_rule_t1_rubin

pytestmark = pytest.mark.arch_rubin

D = 128
NO_CUTS = dict(
    pair_rows=False,
    vn_lane=False,
    o_hoist=False,
    st_part=False,
    unpack_bits=False,
    unpack_qk=False,
    vec_qk=False,
    static_io=False,
    pf_lite=False,
    pf_u32=False,
    pf_addr=False,
)


def _inputs(B, HQ, HK, HV, pool, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*s, dt=dtype):
        return torch.randn(*s, device="cuda", dtype=dt, generator=g)

    return dict(
        q=r(B, 1, HQ, D),
        k=r(B, 1, HK, D),
        v=r(B, 1, HV, D),
        A_log=r(HV, dt=torch.float32),
        a=r(B, 1, HV),
        dt_bias=r(HV, dt=torch.bfloat16),  # the API takes bf16 / fp32 dt_bias
        b=r(B, 1, HV),
        state=r(pool, HV, D, D, dt=torch.bfloat16) * 0.1,
    )


def _call(fn, inp, state, idx, out_idx, l2norm, **extra):
    out = torch.empty(
        inp["q"].shape[0], 1, inp["v"].shape[2], D, device="cuda", dtype=torch.bfloat16
    )
    fn(
        A_log=inp["A_log"],
        a=inp["a"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        b=inp["b"],
        initial_state_source=state,
        initial_state_indices=idx,
        output_state_indices=out_idx,
        use_qk_l2norm_in_kernel=l2norm,
        scale=1.0 / math.sqrt(D),
        output=out,
        **extra,
    )
    return out


def _api(inp, state, idx, out_idx, backend):
    return gated_delta_rule_decode_pretranspose(
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        state=None,
        A_log=inp["A_log"],
        a=inp["a"],
        dt_bias=inp["dt_bias"],
        b=inp["b"],
        scale=1.0 / math.sqrt(D),
        use_qk_l2norm=True,
        initial_state=state,
        initial_state_indices=idx,
        output_state_indices=out_idx,
        backend=backend,
    )[0]


def _pool(B, mode):
    if mode == "split":
        pool = 2 * B
        idx = torch.arange(B, dtype=torch.int32, device="cuda")
        out_idx = torch.arange(B, 2 * B, dtype=torch.int32, device="cuda")
    elif mode == "shuffled":
        pool = B + 7
        idx = torch.randperm(pool, device="cuda")[:B].to(torch.int32)
        out_idx = None
    else:
        pool = B
        idx = torch.arange(B, dtype=torch.int32, device="cuda")
        out_idx = None
    return pool, idx, out_idx


@pytest.mark.parametrize("B", [1, 4, 8, 16, 32, 64, 128, 512])
@pytest.mark.parametrize("HQ,HK,HV", [(16, 16, 64), (16, 16, 32)])
@pytest.mark.parametrize("l2norm", [True, False])
@pytest.mark.parametrize("pool_mode", ["single", "split", "shuffled"])
@pytest.mark.parametrize("cuts", [True, False])
def test_rubin_t1_matches_generic(B, HQ, HK, HV, l2norm, pool_mode, cuts):
    pool, idx, out_idx = _pool(B, pool_mode)
    inp = _inputs(B, HQ, HK, HV, pool, seed=B * 7 + HV)
    st_ref = inp["state"].clone()
    st_new = inp["state"].clone()
    o_ref = _call(generic, inp, st_ref, idx, out_idx, l2norm)
    o_new = _call(
        gated_delta_rule_t1_rubin,
        inp,
        st_new,
        idx,
        out_idx,
        l2norm,
        **({} if cuts else NO_CUTS),
    )
    torch.cuda.synchronize()

    if B * HV >= 512:  # the generic dispatcher runs the same wide_vec T=1 math
        assert torch.equal(o_new, o_ref)
        assert torch.equal(st_new, st_ref)
    else:
        torch.testing.assert_close(o_new.float(), o_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(st_new.float(), st_ref.float(), atol=2e-2, rtol=2e-2)
    # pool slots that are not written must stay untouched
    touched = torch.zeros(pool, dtype=torch.bool, device="cuda")
    touched[(out_idx if out_idx is not None else idx).long()] = True
    assert torch.equal(st_new[~touched], inp["state"][~touched])


@pytest.mark.parametrize("tile_v", [32, 64, 128])
@pytest.mark.parametrize("stage_smem", [True, False])
def test_rubin_t1_variants_bitwise(tile_v, stage_smem):
    """Every compile-time variant computes the same thing."""
    B, HQ, HK, HV = 64, 16, 16, 64
    inp = _inputs(B, HQ, HK, HV, B, seed=3)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    st_ref = inp["state"].clone()
    o_ref = _call(generic, inp, st_ref, idx, None, True)
    st = inp["state"].clone()
    o = _call(
        gated_delta_rule_t1_rubin,
        inp,
        st,
        idx,
        None,
        True,
        tile_v=tile_v,
        stage_smem=stage_smem,
    )
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(st, st_ref)


@pytest.mark.parametrize("cuts", [True, False])
def test_rubin_t1_negative_index_matches_generic(cuts):
    """Padding rows (index -1) follow the generic kernel's redirect-to-slot-0 semantics."""
    B, HQ, HK, HV = 16, 16, 16, 64
    inp = _inputs(B, HQ, HK, HV, B, seed=11)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    idx[3] = -1
    st_ref = inp["state"].clone()
    st_new = inp["state"].clone()
    o_ref = _call(generic, inp, st_ref, idx, None, True)
    o_new = _call(
        gated_delta_rule_t1_rubin,
        inp,
        st_new,
        idx,
        None,
        True,
        **({} if cuts else NO_CUTS),
    )
    torch.cuda.synchronize()
    good = torch.ones(B, dtype=torch.bool, device="cuda")
    good[3] = False
    good[0] = (
        False  # slot 0 is also written by the redirected padding row (race in both)
    )
    assert torch.equal(o_new[good], o_ref[good])
    assert torch.equal(st_new[good], st_ref[good])


@pytest.mark.parametrize("B", [64, 512])
def test_rubin_t1_fused_qkv_views(B):
    """q / k / v as views of one fused QKV buffer (static_io compiles their strides): the
    same bits as contiguous q / k / v."""
    HQ, HK, HV = 16, 16, 64
    inp = _inputs(B, HQ, HK, HV, B, seed=B + 5)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    st_ref = inp["state"].clone()
    o_ref = _call(gated_delta_rule_t1_rubin, inp, st_ref, idx, None, True)
    qkv = torch.cat([inp["q"], inp["k"], inp["v"]], dim=2)  # [B, 1, HQ + HK + HV, D]
    views = dict(
        inp, q=qkv[:, :, :HQ], k=qkv[:, :, HQ : HQ + HK], v=qkv[:, :, HQ + HK :]
    )
    assert not views["q"].is_contiguous()
    st = inp["state"].clone()
    o = _call(gated_delta_rule_t1_rubin, views, st, idx, None, True)
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(st, st_ref)


@pytest.mark.parametrize("B", [2048, 4096])
def test_rubin_t1_bitwise_large_batch(B):
    HQ, HK, HV = 16, 16, 64
    inp = _inputs(B, HQ, HK, HV, B, seed=B)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    st_ref = inp["state"].clone()
    o_ref = _call(generic, inp, st_ref, idx, None, True)
    st = inp["state"].clone()
    o = _call(gated_delta_rule_t1_rubin, inp, st, idx, None, True)
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(st, st_ref)


@pytest.mark.parametrize("B,HV", [(4, 64), (8, 64), (16, 32), (64, 64), (512, 64)])
@pytest.mark.parametrize("pool_mode", ["single", "split"])
def test_rubin_api_routing(B, HV, pool_mode, monkeypatch):
    """backend="auto" on SM107 uses the Rubin kernel exactly when B * HV >= 512, and
    "auto", "rubin" and "flashinfer" (generic) agree bitwise there."""
    calls = []
    real = gdn_decode._gated_delta_rule_t1_rubin

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gdn_decode, "_gated_delta_rule_t1_rubin", counting)
    pool, idx, out_idx = _pool(B, pool_mode)
    inp = _inputs(B, 16, 16, HV, pool, seed=B + HV)
    res = {}
    for backend in ("flashinfer", "auto", "rubin"):
        st = inp["state"].clone()
        n = len(calls)
        res[backend] = (_api(inp, st, idx, out_idx, backend), st)
        res[backend + "_routed"] = len(calls) > n
    torch.cuda.synchronize()
    assert not res["flashinfer_routed"]
    assert res["rubin_routed"]
    assert res["auto_routed"] == (B * HV >= 512)
    if B * HV >= 512:
        for backend in ("auto", "rubin"):
            assert torch.equal(res[backend][0], res["flashinfer"][0])
            assert torch.equal(res[backend][1], res["flashinfer"][1])
    else:
        torch.testing.assert_close(
            res["rubin"][0].float(), res["flashinfer"][0].float(), atol=2e-2, rtol=2e-2
        )


def test_rubin_api_int64_indices_fp16_inputs():
    """int64 read / write indices and fp16 q / k / v / a / b: same bits as the generic path."""
    B, HV = 64, 64
    inp = _inputs(B, 16, 16, HV, 2 * B, seed=21, dtype=torch.float16)
    idx = torch.arange(B, dtype=torch.int64, device="cuda")
    out_idx = torch.arange(B, 2 * B, dtype=torch.int64, device="cuda")
    st_ref = inp["state"].clone()
    st = inp["state"].clone()
    o_ref = _api(inp, st_ref, idx, out_idx, "flashinfer")
    o = _api(inp, st, idx, out_idx, "rubin")
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(st, st_ref)
