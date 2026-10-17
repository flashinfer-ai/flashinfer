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
comparison uses a bf16 tolerance. The build specializes on a few input properties (16 B aligned
q/k rows, unit-stride output rows, dense state rows); the layout test runs each one's other side.
The fp32-reference check is test_decode_delta_rule.py::test_gdn_decode_bf16_state_t1_kernel.
"""

import math

import pytest
import torch

import flashinfer.gdn_decode as gdn_decode
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose, ugpu_state_pool
from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule as generic
from flashinfer.gdn_kernels.gdn_decode_bf16_state_rubin import gated_delta_rule_t1_rubin

pytestmark = pytest.mark.arch_rubin

D = 128


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


def _api(inp, state, idx, out_idx, backend, **extra):
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
        **extra,
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
def test_rubin_t1_matches_generic(B, HQ, HK, HV, l2norm, pool_mode):
    pool, idx, out_idx = _pool(B, pool_mode)
    inp = _inputs(B, HQ, HK, HV, pool, seed=B * 7 + HV)
    st_ref = inp["state"].clone()
    st_new = inp["state"].clone()
    o_ref = _call(generic, inp, st_ref, idx, out_idx, l2norm)
    o_new = _call(gated_delta_rule_t1_rubin, inp, st_new, idx, out_idx, l2norm)
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


@pytest.mark.parametrize(
    "tile_v,pf_dist", [(32, None), (64, None), (128, None), (64, 0)]
)
def test_rubin_t1_variants_bitwise(tile_v, pf_dist):
    """Every tile size, with and without the L2 prefetch, computes the same thing."""
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
        pf_dist=pf_dist,
    )
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(st, st_ref)


def test_rubin_t1_negative_index_matches_generic():
    """Padding rows (index -1) follow the generic kernel's redirect-to-slot-0 semantics."""
    B, HQ, HK, HV = 16, 16, 16, 64
    inp = _inputs(B, HQ, HK, HV, B, seed=11)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    idx[3] = -1
    st_ref = inp["state"].clone()
    st_new = inp["state"].clone()
    o_ref = _call(generic, inp, st_ref, idx, None, True)
    o_new = _call(gated_delta_rule_t1_rubin, inp, st_new, idx, None, True)
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
    """q / k / v as views of one fused QKV buffer (each build compiles their strides): the
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


def test_rubin_t1_strided_pool_sizes():
    """vLLM-style padded pool (slot stride > HV * V * K) at two pool sizes in a row: each
    size gets its own build and the same bits as the generic kernel."""
    HQ, HK, HV = 16, 16, 32
    for B in (16, 32):
        pool = 2 * B
        inp = _inputs(B, HQ, HK, HV, pool, seed=B + 9)
        idx = (torch.arange(B, dtype=torch.int32, device="cuda") * 2 + 1) % pool
        states = []
        for _ in range(2):
            big = torch.empty(pool, HV + 1, D, D, dtype=torch.bfloat16, device="cuda")
            st = big[:, :HV]
            st.copy_(inp["state"])
            assert not st.is_contiguous()
            states.append(st)
        o_ref = _call(generic, inp, states[0], idx, None, True)
        o = _call(gated_delta_rule_t1_rubin, inp, states[1], idx, None, True)
        torch.cuda.synchronize()
        assert torch.equal(o, o_ref)
        assert torch.equal(states[1], states[0])


@pytest.mark.parametrize("layout", ["unaligned_qk", "strided_output", "padded_rows"])
def test_rubin_t1_layouts(layout):
    """The other side of each input property the build specializes on: q / k rows that are
    not 16 B aligned (no 16 B q / k loads), an output with a non-unit last stride, state rows
    padded past K (no dense-row prefetch addressing). The same bits as the generic kernel."""
    B, HQ, HK, HV = 64, 16, 16, 64
    inp = _inputs(B, HQ, HK, HV, B, seed=17)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    st_ref = inp["state"].clone()
    o_ref = _call(generic, inp, st_ref, idx, None, True)
    st = inp["state"].clone()
    if layout == "unaligned_qk":
        for name, H in (("q", HQ), ("k", HK)):
            # rows 264 B apart: only every other row starts on a 16 B boundary
            view = torch.empty(B, 1, H, D + 4, dtype=torch.bfloat16, device="cuda")
            view = view[..., :D]
            view.copy_(inp[name])
            inp = dict(inp, **{name: view})
    if layout == "padded_rows":
        big = torch.empty(B, HV, D, D + 8, dtype=torch.bfloat16, device="cuda")
        st = big[..., :D]
        st.copy_(inp["state"])
    out = torch.empty(B, 1, HV, D, device="cuda", dtype=torch.bfloat16)
    if layout == "strided_output":
        out = torch.empty(B, 1, HV, 2 * D, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    gated_delta_rule_t1_rubin(
        A_log=inp["A_log"],
        a=inp["a"],
        dt_bias=inp["dt_bias"],
        q=inp["q"],
        k=inp["k"],
        v=inp["v"],
        b=inp["b"],
        initial_state_source=st,
        initial_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
        scale=1.0 / math.sqrt(D),
        output=out,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, o_ref)
    assert torch.equal(st, st_ref)


def test_rubin_t1_disk_cache_round_trip(monkeypatch, tmp_path):
    """A second process reloads the exported kernel instead of recompiling (simulated
    in-process: clear the in-process caches, forbid cute.compile, call again)."""
    import flashinfer.gdn_kernels.gdn_decode_bf16_state_rubin as rubin_mod
    import flashinfer.jit.env as jit_env

    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.setattr(rubin_mod, "_compiled_t1_rubin", {})
    monkeypatch.setattr(rubin_mod, "_launch_t1_rubin", {})
    B, HV = 8, 64
    inp = _inputs(B, 16, 16, HV, B, seed=23)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    st1 = inp["state"].clone()
    o1 = _call(gated_delta_rule_t1_rubin, inp, st1, idx, None, True)
    artifacts = list(tmp_path.glob("gdn_decode_bf16_state_rubin_*_cute_dsl/*.o"))
    assert len(artifacts) == 1, f"expected one exported artifact, got {artifacts}"

    monkeypatch.setattr(rubin_mod, "_compiled_t1_rubin", {})
    monkeypatch.setattr(rubin_mod, "_launch_t1_rubin", {})

    def _no_recompile(*args, **kwargs):
        raise AssertionError("cute.compile ran despite a valid disk artifact")

    monkeypatch.setattr(rubin_mod.cute, "compile", _no_recompile)
    st2 = inp["state"].clone()
    o2 = _call(gated_delta_rule_t1_rubin, inp, st2, idx, None, True)
    torch.cuda.synchronize()
    assert torch.equal(o1, o2)
    assert torch.equal(st1, st2)


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


@pytest.mark.parametrize("pool_mode", ["single", "split"])
def test_rubin_api_int64_indices_fp16_inputs(pool_mode):
    """int64 read / write indices and fp16 q / k / v / a / b: the same bits as the generic
    path with int32 indices."""
    B, HV = 64, 64
    pool, idx, out_idx = _pool(B, pool_mode)
    inp = _inputs(B, 16, 16, HV, pool, seed=21, dtype=torch.float16)
    idx64 = idx.to(torch.int64)
    out_idx64 = None if out_idx is None else out_idx.to(torch.int64)
    st_ref = inp["state"].clone()
    st = inp["state"].clone()
    o_ref = _api(inp, st_ref, idx, out_idx, "flashinfer")
    o = _api(inp, st, idx64, out_idx64, "rubin")
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(st, st_ref)


def _ugpu_pool(state):
    """A uGPU pool [pool, HV, 2, V/2, K] holding the contents of a [pool, HV, V, K] pool."""
    P, HV = state.shape[:2]
    pool = ugpu_state_pool(P, HV, D, state.device)
    pool.copy_(state.view(P, HV, 2, D // 2, D))
    return pool


def _as_4d(pool):
    P, HV = pool.shape[:2]
    return pool.reshape(P, HV, D, D)


@pytest.mark.parametrize("B,HV", [(1, 64), (8, 64), (64, 32), (512, 64)])
@pytest.mark.parametrize("pool_mode", ["single", "split", "shuffled"])
@pytest.mark.parametrize("eager_per_die", [False, True])
def test_rubin_ugpu_matches_one_launch(B, HV, pool_mode, eager_per_die, monkeypatch):
    """A ugpu_state_pool makes the default call run each half of every state on its own die
    (eager calls below _UGPU_EAGER_MIN_STATES: back to back on the current stream; both
    forms checked): the same bits as one launch over an ordinary pool."""
    if eager_per_die:
        monkeypatch.setattr(gdn_decode, "_UGPU_EAGER_MIN_STATES", 0)
    pool, idx, out_idx = _pool(B, pool_mode)
    inp = _inputs(B, 16, 16, HV, pool, seed=B + HV + 3)
    st_ref = inp["state"].clone()
    o_ref = _api(inp, st_ref, idx, out_idx, "rubin")
    st = _ugpu_pool(inp["state"])
    o = _api(inp, st, idx, out_idx, "auto")
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref)
    assert torch.equal(_as_4d(st), st_ref)


def test_rubin_ugpu_negative_index():
    B, HV = 16, 64
    inp = _inputs(B, 16, 16, HV, B, seed=13)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    idx[5] = -1
    st_ref = inp["state"].clone()
    o_ref = _api(inp, st_ref, idx, None, "rubin")
    st = _ugpu_pool(inp["state"])
    o = _api(inp, st, idx, None, "auto")
    torch.cuda.synchronize()
    good = torch.ones(B, dtype=torch.bool, device="cuda")
    good[5] = False
    good[0] = False
    assert torch.equal(o[good], o_ref[good])
    assert torch.equal(_as_4d(st)[good], st_ref[good])


def test_rubin_ugpu_cuda_graph():
    """The per-die fork / join is captured into a CUDA graph and replays correctly."""
    B, HV = 64, 64
    inp = _inputs(B, 16, 16, HV, B, seed=5)
    idx = torch.arange(B, dtype=torch.int32, device="cuda")
    st_ref = inp["state"].clone()
    for _ in range(3):
        o_ref = _api(inp, st_ref, idx, None, "rubin")
    st = _ugpu_pool(inp["state"])
    out = torch.empty(B, 1, HV, D, device="cuda", dtype=torch.bfloat16)

    def step():
        gated_delta_rule_decode_pretranspose(
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            state=None,
            A_log=inp["A_log"],
            a=inp["a"],
            dt_bias=inp["dt_bias"],
            b=inp["b"],
            scale=1.0 / math.sqrt(D),
            output=out,
            use_qk_l2norm=True,
            initial_state=st,
            initial_state_indices=idx,
        )

    step()  # eager call: creates the per-die green contexts outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, o_ref)
    assert torch.equal(_as_4d(st), st_ref)


@pytest.mark.parametrize("eager_per_die", [False, True])
def test_rubin_ugpu_on_non_current_device(eager_per_die, monkeypatch):
    """A uGPU decode on cuda:1 while cuda:0 is current: per-die streams and events of cuda:1,
    the same bits as one launch there."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    if eager_per_die:
        monkeypatch.setattr(gdn_decode, "_UGPU_EAGER_MIN_STATES", 0)
    B, HV = 32, 64
    with torch.cuda.device(1):
        inp = _inputs(B, 16, 16, HV, B, seed=29)
        idx = torch.arange(B, dtype=torch.int32, device="cuda")
        st_ref = inp["state"].clone()
        o_ref = _api(inp, st_ref, idx, None, "rubin")
        st = _ugpu_pool(inp["state"])
    assert st.device == torch.device("cuda:1")
    with torch.cuda.device(0):
        o = _api(inp, st, idx, None, "auto")
    torch.cuda.synchronize(1)
    assert torch.equal(o, o_ref)
    assert torch.equal(_as_4d(st), st_ref)
