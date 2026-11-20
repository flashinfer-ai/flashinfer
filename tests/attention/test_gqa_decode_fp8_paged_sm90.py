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

import contextlib
import math

import pytest
import torch

import flashinfer
from flashinfer.prefill import (
    GQA_DEC_FP8Q_SM90_BACKENDS,
    _gqa_dec_fp8q_sm90_auto_admits,
    _gqa_dec_fp8q_sm90_auto_routes,
    _gqa_dec_fp8q_sm90_backend_order,
    _gqa_dec_fp8q_sm90_plan_eligibility,
    _gqa_dec_fp8q_sm90_stats,
    _gqa_fp8q_sm90_dispatch_table,
    _reset_gqa_dec_fp8q_sm90_stats,
)

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISABLE_ENV_LEGACY = "FLASHINFER_DISABLE_GQA_DECODE_FP8_PAGED_SM90"
H_QO, H_KV, HEAD_DIM, PAGE, QL = 8, 1, 128, 16, 6
RTOL, ATOL, MATCHED = 0.04, 0.02, 0.98


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV_LEGACY, raising=False)
    _reset_gqa_dec_fp8q_sm90_stats()
    yield
    _reset_gqa_dec_fp8q_sm90_stats()


def _sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # the GPU tests allocate and capture on cuda:0, so gate on THAT device's
    # capability, not on torch.cuda.current_device()
    if torch.cuda.get_device_capability(0) != (9, 0):
        pytest.skip("SM90 required")


# --------------------------------------------------------------------------
# Plan-time geometry guard (GPU-free)
# --------------------------------------------------------------------------


def _plan_kwargs(**overrides):
    kwargs = dict(
        kv_layout="NHD",
        causal=True,
        has_custom_mask=False,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        window_left=-1,
        logits_soft_cap=0.0,
        sm_scale=None,
        num_qo_heads=H_QO,
        num_kv_heads=H_KV,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE,
        batch_size=7,
        total_q=7 * QL,
    )
    kwargs.update(overrides)
    return kwargs


def test_plan_guard_accepts_the_measured_contract():
    assert _gqa_dec_fp8q_sm90_plan_eligibility(**_plan_kwargs()) is None
    assert (
        _gqa_dec_fp8q_sm90_plan_eligibility(
            **_plan_kwargs(sm_scale=1.0 / math.sqrt(HEAD_DIM))
        )
        is None
    )
    assert (
        _gqa_dec_fp8q_sm90_plan_eligibility(**_plan_kwargs(batch_size=1, total_q=QL))
        is None
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"kv_layout": "HND"},
        {"causal": False},
        {"has_custom_mask": True},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"use_fp16_qk_reduction": True},
        {"window_left": 1024},
        {"logits_soft_cap": 30.0},
        {"sm_scale": 1.0},
        {"num_qo_heads": 4},
        {"num_kv_heads": 2},
        {"head_dim_qk": 256},
        {"head_dim_vo": 64},
        {"page_size": 1},
        {"batch_size": 0},
        {"batch_size": 129, "total_q": 129 * QL},
        {"total_q": 7 * QL - 1},  # not a uniform q_len == 6 batch
        {"total_q": 7},  # single-token decode is served by the stock path
    ],
)
def test_plan_guard_rejects_other_geometries(overrides):
    assert _gqa_dec_fp8q_sm90_plan_eligibility(**_plan_kwargs(**overrides)) is not None


def test_stats_report_build_state_and_counters():
    stats = _gqa_dec_fp8q_sm90_stats()
    for key in (
        "plans_seen",
        "runs_dispatched",
        "module_loaded",
        "module_error",
        "backends",
        "backend_dispatches",
        "fallback_reasons",
        "dispatch_table_loaded",
    ):
        assert key in stats
    assert stats["backends"] == GQA_DEC_FP8Q_SM90_BACKENDS
    assert set(stats["backend_dispatches"]) == {"specialized", "composable"}


# --------------------------------------------------------------------------
# Backend selection surface (#5667 pattern, GPU-free)
# --------------------------------------------------------------------------


def test_backends_tuple_is_plain_strings_with_specialized():
    assert "specialized" in GQA_DEC_FP8Q_SM90_BACKENDS
    assert "auto" in GQA_DEC_FP8Q_SM90_BACKENDS
    assert all(isinstance(b, str) for b in GQA_DEC_FP8Q_SM90_BACKENDS)


def test_dispatch_table_loads_and_matches_guard_constants():
    table = _gqa_fp8q_sm90_dispatch_table()
    assert table is not None
    assert table["kill_switch"] == DISABLE_ENV
    assert "specialized" in table["backends"]
    op = table["ops"]["decode"]
    assert op["num_qo_heads"] == H_QO
    assert op["num_kv_heads"] == H_KV
    assert op["head_dim"] == HEAD_DIM
    assert op["page_size"] == PAGE
    assert op["uniform_q_len"] == QL
    assert op["kv_layout"] == "NHD"
    assert op["causal"] is True
    assert op["envelope"]["max_batch_size"] == 128
    assert op["envelope"]["max_kv_len"] == 262144


def test_auto_routes_only_on_the_allowlist():
    ok, reason = _gqa_dec_fp8q_sm90_auto_routes(
        H_QO, H_KV, HEAD_DIM, HEAD_DIM, PAGE, "NHD", True
    )
    assert ok and reason is None
    for bad in (
        dict(num_qo_heads=4),
        dict(num_kv_heads=2),
        dict(head_dim_qk=256),
        dict(head_dim_vo=64),
        dict(page_size=1),
        dict(kv_layout="HND"),
        dict(causal=False),
    ):
        kwargs = dict(
            num_qo_heads=H_QO,
            num_kv_heads=H_KV,
            head_dim_qk=HEAD_DIM,
            head_dim_vo=HEAD_DIM,
            page_size=PAGE,
            kv_layout="NHD",
            causal=True,
        )
        kwargs.update(bad)
        ok, reason = _gqa_dec_fp8q_sm90_auto_routes(**kwargs)
        assert not ok and reason, bad


def test_backend_order_never_swaps_an_explicit_stock_backend():
    # explicit "specialized" never falls back: it runs the kernel or raises
    assert _gqa_dec_fp8q_sm90_backend_order("specialized", False) == ("specialized",)
    assert _gqa_dec_fp8q_sm90_backend_order("auto", True) == (
        "specialized",
        "composable",
    )
    assert _gqa_dec_fp8q_sm90_backend_order("auto", False) == ("composable",)
    for stock in ("fa2", "fa3", "cudnn", "trtllm-gen", "cute-dsl"):
        assert "specialized" not in _gqa_dec_fp8q_sm90_backend_order(stock, True)


# --------------------------------------------------------------------------
# Device tests (SM90): reference equivalence, kill-switch, CUDA graph
# --------------------------------------------------------------------------


def _make_batch(kv_lens, device, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    B = len(kv_lens)
    q_lens = [QL] * B
    pprs = [(k + PAGE - 1) // PAGE for k in kv_lens]
    P = sum(pprs)
    nb = max(pprs)
    bt = torch.zeros((B, nb), dtype=torch.int32, device=device)
    s = 0
    indptr = [0]
    indices = []
    for i, ppr in enumerate(pprs):
        bt[i, :ppr] = torch.arange(s, s + ppr, dtype=torch.int32, device=device)
        indices.extend(range(s, s + ppr))
        indptr.append(indptr[-1] + ppr)
        s += ppr

    def rnd(shape):
        x = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
        return (x * 0.5).to(torch.float8_e4m3fn)

    TQ = sum(q_lens)
    cu = [0]
    for ln in q_lens:
        cu.append(cu[-1] + ln)
    return dict(
        q=rnd((TQ, H_QO, HEAD_DIM)),
        k_cache=rnd((P, PAGE, H_KV, HEAD_DIM)),
        v_cache=rnd((P, PAGE, H_KV, HEAD_DIM)),
        qo_indptr=torch.tensor(cu, dtype=torch.int32, device=device),
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=torch.tensor(indices, dtype=torch.int32, device=device),
        last_page_len=torch.tensor(
            [(k - 1) % PAGE + 1 for k in kv_lens], dtype=torch.int32, device=device
        ),
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        block_table=bt,
    )


def _reference(batch, q_descale=None, k_descale=None, v_descale=None):
    """fp32 dequant-then-attend verify-decode reference (row pos of request r
    sees kv_len_r - 6 + pos + 1 keys), with optional per-sequence descales."""
    device = batch["q"].device
    q = batch["q"].float()
    kf = batch["k_cache"].float().reshape(-1, HEAD_DIM)
    vf = batch["v_cache"].float().reshape(-1, HEAD_DIM)
    cu = batch["qo_indptr"].tolist()
    kv_lens = batch["seq_lens"].tolist()
    bt = batch["block_table"]
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    scale = HEAD_DIM**-0.5
    off = torch.arange(PAGE, device=device)
    for r in range(len(kv_lens)):
        q0, q1 = cu[r], cu[r + 1]
        ql = q1 - q0
        kvl = kv_lens[r]
        npages = (kvl + PAGE - 1) // PAGE
        flat = (bt[r, :npages].long().unsqueeze(1) * PAGE + off).reshape(-1)[:kvl]
        k = kf[flat]
        v = vf[flat]
        qr = q[q0:q1]
        if q_descale is not None:
            qr = qr * float(q_descale.reshape(-1)[r])
        if k_descale is not None:
            k = k * float(k_descale.reshape(-1)[r])
        if v_descale is not None:
            v = v * float(v_descale.reshape(-1)[r])
        sc = torch.einsum("nhd,ld->nhl", qr, k) * scale
        vis = (kvl - ql) + torch.arange(ql, device=device) + 1
        mask = torch.arange(kvl, device=device).view(1, 1, kvl) >= vis.view(ql, 1, 1)
        sc = sc.masked_fill(mask, float("-inf"))
        out[q0:q1] = torch.einsum("nhl,ld->nhd", torch.softmax(sc, -1), v).to(
            torch.bfloat16
        )
    return out


def _plan_wrapper(
    batch,
    device,
    use_cuda_graph=False,
    backend="specialized",
    q_data_type=torch.float8_e4m3fn,
    num_qo_heads=H_QO,
):
    ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    kwargs = {}
    if use_cuda_graph:
        kwargs = dict(
            use_cuda_graph=True,
            qo_indptr_buf=batch["qo_indptr"],
            paged_kv_indptr_buf=batch["kv_indptr"],
            paged_kv_indices_buf=batch["kv_indices"],
            paged_kv_last_page_len_buf=batch["last_page_len"],
        )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend=backend, **kwargs
    )
    wrapper.plan(
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
        num_qo_heads,
        H_KV,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=q_data_type,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    return wrapper


def _matched_ratio(y, ref):
    yf, rf = y.float(), ref.float()
    ok = (yf - rf).abs() <= (ATOL + RTOL * rf.abs())
    return float(ok.float().mean().item())


@pytest.mark.parametrize(
    "kv_lens",
    [
        [1500],
        [1500, 658, 76, 341, 341, 863, 2030],
        [584, 727],
        [64, 640, 6400, 1280, 320, 960, 2048, 128, 333, 77],
        # FP32 carry boundary: 128 x 5761 gives ~4-5 CTAs per request, so at
        # least one split exceeds 24 tiles and takes the carry path; 128 x
        # 4608 (96 tiles, at most 24 per split) is the no-carry control.
        [5761] * 128,
        [4608] * 128,
    ],
)
def test_specialized_matches_reference_and_stock(kv_lens, monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch(kv_lens, device)
    wrapper = _plan_wrapper(batch, device)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 1, stats

    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    ref = _reference(batch)
    assert _matched_ratio(out, ref) >= MATCHED

    # determinism: the kernel is atomics-free and bit-stable
    out2 = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    assert torch.equal(out, out2)

    # kill-switch: the explicit plan raises (never falls back); an auto-armed
    # plan serves the same call by dequant + stock fa2
    monkeypatch.setenv(DISABLE_ENV, "1")
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_disabled"] == 1, stats
    monkeypatch.delenv(DISABLE_ENV)
    wrapper_auto = _plan_wrapper(batch, device, backend="auto")
    monkeypatch.setenv(DISABLE_ENV, "1")
    out_stock = wrapper_auto.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dequant_fallback"] == 1, stats
    assert _matched_ratio(out_stock, ref) >= MATCHED
    assert _matched_ratio(out, out_stock) >= MATCHED


def test_non_uniform_q_len_takes_the_stock_path():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    # break uniformity: 5 + 7 + 6 rows instead of 6 + 6 + 6
    batch["qo_indptr"] = torch.tensor([0, 5, 12, 18], dtype=torch.int32, device=device)
    # explicit "specialized" refuses the ineligible plan loudly
    with pytest.raises(NotImplementedError, match="never falls back"):
        _plan_wrapper(batch, device)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0 and stats["plans_ineligible"] == 1, stats
    # auto never arms off the q_len domain: the stock auto resolution serves
    # the plan untouched (never the slower composable bf16-dequant fa2 pin)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats
    assert not getattr(wrapper, "_gqa_dec_fp8q_sm90_requested", False)
    assert wrapper._cached_q_data_type == torch.float8_e4m3fn  # dtype not flipped
    assert any(
        r.startswith("auto:") and "q_len" in r for r in stats["fallback_reasons"]
    ), stats
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_dequant_fallback"] == 0  # stock-served, not composable
    assert out.shape == batch["q"].shape


def test_legacy_kill_switch_alias_still_disables(monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    monkeypatch.setenv(DISABLE_ENV_LEGACY, "1")
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_disabled"] == 1 and stats["runs_dequant_fallback"] == 1, stats
    assert stats["runs_dispatched"] == 0
    assert _matched_ratio(out, _reference(batch)) >= MATCHED
    # explicit "specialized" raises under the alias too
    monkeypatch.delenv(DISABLE_ENV_LEGACY)
    wrapper_exp = _plan_wrapper(batch, device)
    monkeypatch.setenv(DISABLE_ENV_LEGACY, "1")
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper_exp.run(batch["q"], (batch["k_cache"], batch["v_cache"]))


def test_explicit_specialized_raises_for_contracts_it_has_no_kernel_for():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    batch = _make_batch([584, 75], device)
    with pytest.raises(ValueError, match="specialized"):
        _plan_wrapper(batch, device, backend="specialized", q_data_type=torch.bfloat16)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0 and stats["runs_dispatched"] == 0


def test_explicit_stock_backend_is_never_swapped():
    """A caller who names a stock backend must never be served by the
    specialized kernel — even on an allowlisted shape."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    try:
        wrapper = _plan_wrapper(batch, device, backend="fa2")
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    except Exception:
        pass  # stock fa2 has no fp8-Q contract on SM90; failing loudly is stock behavior
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats
    assert stats["runs_dispatched"] == 0, stats
    assert stats["backend_dispatches"]["specialized"] == 0, stats


def test_auto_routes_to_specialized_on_the_allowlist_only():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 1, stats
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert _matched_ratio(out, _reference(batch)) >= MATCHED

    # off-allowlist geometry: auto never routes to "specialized"
    _reset_gqa_dec_fp8q_sm90_stats()
    batch4 = _make_batch([584, 75], device)
    batch4["q"] = batch4["q"][:, :4, :].contiguous()
    # stock auto may reject fp8-Q; it must simply never arm us
    with contextlib.suppress(Exception):
        _plan_wrapper(batch4, device, backend="auto", num_qo_heads=4)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats
    assert any(r.startswith("auto:") for r in stats["fallback_reasons"]), stats


def test_run_guard_declines_on_mismatched_arguments():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    # return_lse is outside the contract: explicit "specialized" raises
    wrapper = _plan_wrapper(batch, device)
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]), return_lse=True)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0 and stats["runs_ineligible"] == 1
    # an auto-armed plan is served by the dequant fallback
    wrapper_auto = _plan_wrapper(batch, device, backend="auto")
    out, lse = wrapper_auto.run(
        batch["q"], (batch["k_cache"], batch["v_cache"]), return_lse=True
    )
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_ineligible"] == 2 and stats["runs_dequant_fallback"] == 1
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


@pytest.mark.parametrize("entry", ["forward", "forward_return_lse"])
def test_deprecated_forward_requires_soft_cap_replan(entry):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([80], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    assert _gqa_dec_fp8q_sm90_stats()["plans_prepared"] == 1
    with pytest.raises(ValueError, match="Re-plan with logits_soft_cap"):
        getattr(wrapper, entry)(
            batch["q"],
            (batch["k_cache"], batch["v_cache"]),
            causal=True,
            logits_soft_cap=30,
        )
    assert _gqa_dec_fp8q_sm90_stats()["backend_dispatches"] == {
        "specialized": 0,
        "composable": 0,
    }


@pytest.fixture
def other_gpu_current():
    _sm90_or_skip()
    with torch.cuda.device(1 if torch.cuda.device_count() > 1 else 0):
        yield


def test_graph_capture_replays_bitwise_equal_to_eager(other_gpu_current):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([700, 76, 341, 1363], device)
    wrapper = _plan_wrapper(batch, device, use_cuda_graph=True)
    q = batch["q"]
    kv = (batch["k_cache"], batch["v_cache"])
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    eager = wrapper.run(q, kv).clone()
    assert _gqa_dec_fp8q_sm90_stats()["runs_dispatched"] == 1

    graph = _capture(wrapper, q, kv, out)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched_capturing"] == 1, stats

    _poison_and_replay(graph, out)
    assert torch.equal(out, eager)
    assert _matched_ratio(out, _reference(batch)) >= MATCHED

    # new content through the same buffers replays correctly
    q.copy_(_make_batch([700, 76, 341, 1363], device, seed=7)["q"])
    _poison_and_replay(graph, out)
    eager2 = wrapper.run(q, kv)
    assert torch.equal(out, eager2)


# --------------------------------------------------------------------------
# Review regressions: descale placement, fallback robustness, replan safety
# --------------------------------------------------------------------------


def test_per_sequence_v_descale_is_applied_in_fp32():
    """A large V descale must not overflow the fp16 split partials (448 * 256
    exceeds fp16 max): the descale lands in the fp32 merge, not the partials."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([6, 640], device)
    # drive EVERY V page to the e4m3 max: request 0 (one page) exercises the
    # large-descale overflow, and request 1's constant V makes its expected
    # output exactly 448 * 2^-12 — random V scaled by 2^-12 would sit inside
    # ATOL of zero, so a kernel that drops the request would still pass
    batch["v_cache"][:] = torch.full_like(batch["v_cache"].float(), 448.0).to(
        torch.float8_e4m3fn
    )
    qd = torch.tensor([1.25, 0.5], dtype=torch.float32, device=device)
    kd = torch.tensor([0.75, 1.5], dtype=torch.float32, device=device)
    vd = torch.tensor([256.0, 2.0**-12], dtype=torch.float32, device=device)
    wrapper = _plan_wrapper(batch, device)
    out = wrapper.run(
        batch["q"],
        (batch["k_cache"], batch["v_cache"]),
        q_scale=qd,
        k_scale=kd,
        v_scale=vd,
    )
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    assert torch.isfinite(out.float()).all()
    ref = _reference(batch, q_descale=qd, k_descale=kd, v_descale=vd)
    assert _matched_ratio(out, ref) >= MATCHED
    # the small-descale request asserts its known nonzero value separately
    # (attention over constant V returns V): 448 * 2^-12 with a tolerance far
    # below that value, so a zeroed second request cannot pass
    expected_small = 448.0 * 2.0**-12
    small = out[6:12].float()
    assert torch.all((small - expected_small).abs() <= 0.05 * expected_small), small


def test_kill_switch_still_serves_per_sequence_q_scale(monkeypatch):
    """On an auto-armed plan the composable path must serve a per-sequence
    q_scale tensor even when the specialized kernel never prepared (kill
    switch set at plan time); an explicit plan raises at plan time."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    monkeypatch.setenv(DISABLE_ENV, "1")
    batch = _make_batch([584, 75], device)
    with pytest.raises(NotImplementedError, match="never falls back"):
        _plan_wrapper(batch, device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_disabled"] == 2 and stats["plans_prepared"] == 0, stats
    qd = torch.tensor([1.5, 0.25], dtype=torch.float32, device=device)
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]), q_scale=qd)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_dequant_fallback"] == 1, stats
    assert _matched_ratio(out, _reference(batch, q_descale=qd)) >= MATCHED


def test_positional_scales_on_default_auto_match_native_per_head_abi():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 75], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    s = torch.ones(1, dtype=torch.float32, device=device)
    sq = torch.arange(1, H_QO + 1, dtype=torch.float32, device=device)
    kv = batch["k_cache"], batch["v_cache"]
    assert _matched_ratio(wrapper.run(batch["q"], kv), _reference(batch)) >= MATCHED
    assert _gqa_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    stock = _plan_wrapper(batch, device, backend="fa3")
    expected = stock.run(batch["q"], kv, sq, 2 * s, 3 * s)
    assert torch.equal(wrapper.run(batch["q"], kv, sq, 2 * s, 3 * s), expected)
    assert not torch.equal(
        expected, stock.run(batch["q"], kv, sq.flip([0]), 2 * s, 3 * s)
    )
    wrapper_exp = _plan_wrapper(batch, device)
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper_exp.run(batch["q"], (batch["k_cache"], batch["v_cache"]), s, s, s)


def test_workspace_size_sizes_the_composable_fa2_plan():
    """workspace_size on an fp8-Q specialized wrapper must size the composable
    fa2/bf16 plan exactly as plan() builds it (not hit a generator assert)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_args = (
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
    )
    spec = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend="specialized"
    )
    sized = spec.workspace_size(
        *ws_args,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    stock = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend="fa2"
    ).workspace_size(
        *ws_args,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    assert sized == stock
    # a contract the specialized backend has no kernel for raises plan()'s
    # error instead of reaching the stock module generators
    with pytest.raises(ValueError, match="specialized"):
        spec.workspace_size(*ws_args, causal=True, q_data_type=torch.bfloat16)


def test_graph_mode_replan_with_grown_kv_keeps_buffer_pointers():
    """A CUDA-graph wrapper re-planned with grown sequences must refresh the
    same block-table/workspace buffers in place: a retained capture keeps
    their pointers (and the page-table width) baked into the launch."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    small = _make_batch([20, 35], device)  # 2 + 3 pages
    grown = _make_batch([100, 85], device)  # 7 + 6 pages
    ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws,
        kv_layout="NHD",
        backend="specialized",
        use_cuda_graph=True,
        qo_indptr_buf=grown["qo_indptr"].clone(),
        paged_kv_indptr_buf=grown["kv_indptr"].clone(),
        paged_kv_indices_buf=grown["kv_indices"].clone(),
        paged_kv_last_page_len_buf=grown["last_page_len"].clone(),
    )

    def _plan(b):
        wrapper.plan(
            b["qo_indptr"],
            b["kv_indptr"],
            b["kv_indices"],
            b["last_page_len"],
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )

    _plan(small)
    st1 = wrapper._gqa_dec_fp8q_sm90_state
    assert st1 is not None
    out1 = wrapper.run(small["q"], (small["k_cache"], small["v_cache"]))
    assert _matched_ratio(out1, _reference(small)) >= MATCHED

    _plan(grown)
    st2 = wrapper._gqa_dec_fp8q_sm90_state
    assert st2 is not None
    assert st2.block_table.data_ptr() == st1.block_table.data_ptr()
    assert st2.block_table.shape == st1.block_table.shape
    assert st2.workspace.data_ptr() == st1.workspace.data_ptr()
    assert st2.workspace.numel() == st1.workspace.numel()
    assert st2.seq_lens.data_ptr() == st1.seq_lens.data_ptr()
    out2 = wrapper.run(grown["q"], (grown["k_cache"], grown["v_cache"]))
    assert _matched_ratio(out2, _reference(grown)) >= MATCHED


def test_auto_replan_never_arms_over_a_retained_fa3_graph(monkeypatch):
    """An allowlisted fp8-Q auto re-plan must not swap a retained FA3
    CUDA-graph capture onto the composable fa2 route: FA2 planning would
    overwrite the integer workspace the captured FA3 kernels point into.
    GPU-free: capability and stock resolution are mocked."""
    import flashinfer.prefill as fp

    class _Sentinel(Exception):
        pass

    class _Observed(fp.BatchPrefillWithPagedKVCacheWrapper):
        def __setattr__(self, name, value):
            if self.__dict__.get("_observe", False):
                raise _Sentinel(name)
            super().__setattr__(name, value)

    w = _Observed.__new__(_Observed)
    w.__dict__.update(
        _float_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _batch_invariant=None,
        _requested_backend="auto",
        _backend="fa3",
        _cuda_graph_backend="fa3",
        _jit_module=None,
        _use_cuda_graph=True,
        _custom_mask_buf=None,
        _plan_info=object(),
        _variant_owns_mask=False,
        _kv_layout="NHD",
        device=torch.device("cuda"),  # never touched: capability is mocked
        _qo_indptr_buf=torch.tensor([0, QL], dtype=torch.int32),
        _observe=True,
    )
    monkeypatch.setattr(fp, "get_compute_capability", lambda d: (9, 0))
    monkeypatch.setattr(fp, "determine_attention_backend", lambda *a, **k: "fa2")
    with pytest.raises(ValueError, match="CUDA graph backend"):
        w.plan(
            torch.tensor([0, QL], dtype=torch.int32),
            torch.tensor([0, 5], dtype=torch.int32),
            torch.arange(5, dtype=torch.int32),
            torch.tensor([16], dtype=torch.int32),
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
    stats = _gqa_dec_fp8q_sm90_stats()
    assert any("retained CUDA-graph backend" in r for r in stats["fallback_reasons"]), (
        stats
    )


# --------------------------------------------------------------------------
# Codex review regressions: complete auto admission, PV operand range,
# retained-graph replan contract
# --------------------------------------------------------------------------


def _admit_kwargs(batch_size, qo_indptr, kv_lens, **overrides):
    kwargs = dict(
        num_qo_heads=H_QO,
        num_kv_heads=H_KV,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE,
        kv_layout="NHD",
        causal=True,
        has_custom_mask=False,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        window_left=-1,
        logits_soft_cap=0.0,
        sm_scale=None,
        batch_size=batch_size,
        qo_indptr_host=torch.tensor(qo_indptr, dtype=torch.int32),
        prefix_len_ptr=None,
        token_pos_in_items_len=0,
        has_jit_module=False,
        kv_meta_fn=lambda: (
            (
                None,
                None,
                torch.tensor(kv_lens, dtype=torch.int32),
            )
            if kv_lens is not None
            else None
        ),
    )
    kwargs.update(overrides)
    return kwargs


def test_auto_admission_resolves_the_full_domain_gpu_free():
    """backend='auto' admission must check the whole measured domain (q_len,
    batch, kv envelope, plan options) from host metadata, not just geometry."""
    ok, reason, kv_meta = _gqa_dec_fp8q_sm90_auto_admits(
        **_admit_kwargs(2, [0, 6, 12], [64, 65])
    )
    assert ok and reason is None and kv_meta is not None
    declined = [
        _admit_kwargs(2, [0, 4, 12], [64, 64]),  # non-uniform q_len
        _admit_kwargs(1, [0, 128], [1024]),  # ordinary prefill q_len
        _admit_kwargs(1, [0, 1], [64]),  # single-token decode
        _admit_kwargs(129, [i * QL for i in range(130)], [64] * 129),  # batch
        _admit_kwargs(1, [0, 6], [262145]),  # kv above measured envelope
        _admit_kwargs(1, [0, 6], [4]),  # kv_len < q_len
        _admit_kwargs(1, [0, 6], None),  # kv lengths unavailable
        _admit_kwargs(1, [0, 6], [64], window_left=1024),
        _admit_kwargs(1, [0, 6], [64], logits_soft_cap=30.0),
        _admit_kwargs(1, [0, 6], [64], sm_scale=1.0),
        _admit_kwargs(1, [0, 6], [64], has_custom_mask=True),
        _admit_kwargs(1, [0, 6], [64], has_jit_module=True),
    ]
    for kwargs in declined:
        ok, reason, _ = _gqa_dec_fp8q_sm90_auto_admits(**kwargs)
        assert not ok and reason, kwargs


def test_auto_keeps_stock_resolution_for_ordinary_prefill():
    """An fp8-Q prefill outside the six-token verify domain must keep the
    stock auto resolution: no bf16 dtype flip, no fa2 pin, no composable
    dequant fallback (the measured 2.6x ordinary-prefill regression)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    for q_len, kv_len in [(128, 1024), (1, 64)]:
        _reset_gqa_dec_fp8q_sm90_stats()
        g = torch.Generator(device=device)
        g.manual_seed(3)
        npages = (kv_len + PAGE - 1) // PAGE

        def rnd(shape):
            x = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
            return (x * 0.5).to(torch.float8_e4m3fn)

        batch = dict(
            q=rnd((q_len, H_QO, HEAD_DIM)),
            k_cache=rnd((npages, PAGE, H_KV, HEAD_DIM)),
            v_cache=rnd((npages, PAGE, H_KV, HEAD_DIM)),
            qo_indptr=torch.tensor([0, q_len], dtype=torch.int32, device=device),
            kv_indptr=torch.tensor([0, npages], dtype=torch.int32, device=device),
            kv_indices=torch.arange(npages, dtype=torch.int32, device=device),
            last_page_len=torch.tensor(
                [(kv_len - 1) % PAGE + 1], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([kv_len], dtype=torch.int32, device=device),
            block_table=torch.arange(npages, dtype=torch.int32, device=device).view(
                1, npages
            ),
        )
        wrapper = _plan_wrapper(batch, device, backend="auto")
        stats = _gqa_dec_fp8q_sm90_stats()
        assert stats["plans_prepared"] == 0, (q_len, stats)
        assert not getattr(wrapper, "_gqa_dec_fp8q_sm90_requested", False)
        assert wrapper._cached_q_data_type == torch.float8_e4m3fn
        assert wrapper._backend != "fa2" or q_len == 1  # never the dequant pin
        out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
        stats = _gqa_dec_fp8q_sm90_stats()
        assert stats["runs_dequant_fallback"] == 0, (q_len, stats)
        assert stats["runs_dispatched"] == 0
        assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_workspace_size_off_domain_auto_sizes_the_stock_plan(monkeypatch):
    """workspace_size must apply the same complete auto admission as plan():
    an off-domain fp8-Q plan is sized for the stock resolution, not for the
    composable fa2/bf16 pin."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    # off-domain: non-uniform q_len
    qo = torch.tensor([0, 5, 12, 18], dtype=torch.int32, device=device)
    ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_args = (
        qo,
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
    )
    kwargs = dict(
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )

    def size_with(auto_routes):
        import flashinfer.prefill as fp

        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", backend="auto"
        )
        if auto_routes is not None:
            monkeypatch.setattr(fp, "_gqa_dec_fp8q_sm90_auto_routes", auto_routes)
        try:
            return w.workspace_size(*ws_args, **kwargs)
        except Exception as exc:  # stock behavior may be an exception too
            return (type(exc), str(exc))
        finally:
            monkeypatch.undo()

    # the stock resolution's behavior for this plan (value, or its exception)
    # must be exactly what the off-domain auto wrapper reports
    sized = size_with(None)
    stock = size_with(lambda *a, **k: (False, "bypassed"))
    assert sized == stock
    assert not (isinstance(sized, tuple) and sized and sized[0] is ValueError), sized
    # in-domain sizing still matches the composable fa2/bf16 plan
    in_domain = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend="auto"
    ).workspace_size(
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
        **kwargs,
    )
    fa2_bf16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend="fa2"
    ).workspace_size(
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    assert in_domain == fa2_bf16


def _gap_batch(device, single_key=False):
    kv_len = 48
    npages = (kv_len + PAGE - 1) // PAGE
    q = torch.zeros((QL, H_QO, HEAD_DIM), dtype=torch.float32, device=device)
    q[:, :, 0] = 4.0
    k = torch.zeros((npages, PAGE, H_KV, HEAD_DIM), dtype=torch.float32, device=device)
    k[0, 0, 0, 0] = 64.0
    v = torch.full(
        (npages, PAGE, H_KV, HEAD_DIM), 448.0, dtype=torch.float32, device=device
    )
    v[0, 0, 0, :] = 0.0
    if single_key:
        v.zero_()
        v[0, 1, 0, :] = 448.0  # visible on every query row
    batch = dict(
        q=q.to(torch.float8_e4m3fn),
        k_cache=k.to(torch.float8_e4m3fn),
        v_cache=v.to(torch.float8_e4m3fn),
        qo_indptr=torch.tensor([0, QL], dtype=torch.int32, device=device),
        kv_indptr=torch.tensor([0, npages], dtype=torch.int32, device=device),
        kv_indices=torch.arange(npages, dtype=torch.int32, device=device),
        last_page_len=torch.tensor(
            [(kv_len - 1) % PAGE + 1], dtype=torch.int32, device=device
        ),
        seq_lens=torch.tensor([kv_len], dtype=torch.int32, device=device),
        block_table=torch.arange(npages, dtype=torch.int32, device=device).view(
            1, npages
        ),
    )
    return batch


@pytest.mark.parametrize(
    "gap,single_key", [(17, False), (18, False), (22, False), (22, True)]
)
def test_small_probabilities_survive_pv_staging(gap, single_key):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _gap_batch(device, single_key)
    # key 0 scores `gap` nats; every other key's relative weight is exp(-gap)
    qd = torch.tensor(
        [gap * math.sqrt(HEAD_DIM) / 256.0], dtype=torch.float32, device=device
    )
    vd = torch.tensor(
        [2**18 if single_key else 256.0], dtype=torch.float32, device=device
    )
    wrapper = _plan_wrapper(batch, device, backend="specialized")
    out = wrapper.run(
        batch["q"], (batch["k_cache"], batch["v_cache"]), q_scale=qd, v_scale=vd
    )
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    ref = _reference(batch, q_descale=qd, v_descale=vd)
    # At least one visible low-probability key carries positive V, so every row, head and
    # channel must retain a strictly positive contribution: a tolerance check
    # alone would accept zeros for the ~1e-3 reference values at gap 22
    assert bool((out.float() > 0).all())
    assert _matched_ratio(out, ref) >= MATCHED
    # the recovered contributions are small; hold them to a tight absolute bound
    assert float((out.float() - ref.float()).abs().max()) <= 2e-3


def test_eight_auto_plans_retain_only_one_float_scratch_allocation():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([80, 80], device)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    wrappers = [flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws) for _ in range(8)]
    float_storage_bytes = {ws.data_ptr(): ws.nbytes}
    schedules = set()
    for wrapper in wrappers:
        _graph_plan(
            wrapper,
            *(
                batch[k]
                for k in ("qo_indptr", "kv_indptr", "kv_indices", "last_page_len")
            ),
        )
        _, fbuf, ibuf, _ = wrapper._verify_native
        storage = fbuf.untyped_storage()
        float_storage_bytes[storage.data_ptr()] = storage.nbytes()
        schedules.add(ibuf.data_ptr())
    assert sum(float_storage_bytes.values()) == ws.nbytes
    assert len(schedules) == len(wrappers)


def _graph_batch_and_wrapper(device, kv_lens=(80, 80)):
    batch = _make_batch(list(kv_lens), device)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws,
        kv_layout="NHD",
        backend="auto",
        use_cuda_graph=True,
        qo_indptr_buf=batch["qo_indptr"].clone(),
        paged_kv_indptr_buf=batch["kv_indptr"].clone(),
        paged_kv_indices_buf=batch["kv_indices"].clone(),
        paged_kv_last_page_len_buf=batch["last_page_len"].clone(),
    )
    return batch, wrapper


def _graph_plan(wrapper, qo_indptr, kv_indptr, kv_indices, last_page_len):
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )


def _capture(wrapper, q, kv, out, *args, **kwargs):
    with torch.cuda.device(q.device):
        stream = torch.cuda.Stream(device=q.device)
        current = torch.cuda.current_stream(q.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            wrapper.run(q, kv, *args, out=out, **kwargs)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            wrapper.run(q, kv, *args, out=out, **kwargs)
        return graph


def _poison_and_replay(graph, out):
    with torch.cuda.device(out.device):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.current_stream(out.device).synchronize()


def _ref_with_indices(batch, kv_indices, kv_lens):
    b = dict(batch)
    device = batch["q"].device
    b["kv_indices"] = kv_indices
    b["seq_lens"] = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    # rebuild the per-request block table the reference walks
    pprs = [(k + PAGE - 1) // PAGE for k in kv_lens]
    nb = max(pprs)
    bt = torch.zeros((len(kv_lens), nb), dtype=torch.int32, device=device)
    s = 0
    for i, ppr in enumerate(pprs):
        bt[i, :ppr] = kv_indices[s : s + ppr]
        s += ppr
    b["block_table"] = bt
    return _reference(b)


def test_native_positional_graph_survives_failed_then_successful_replan(
    monkeypatch, other_gpu_current
):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch, wrapper = _graph_batch_and_wrapper(device)
    qo, ki, ids, last = (
        batch[n] for n in ("qo_indptr", "kv_indptr", "kv_indices", "last_page_len")
    )
    _graph_plan(wrapper, qo, ki, ids, last)
    native, fbuf, ibuf, _ = wrapper._verify_native
    assert fbuf is wrapper._float_workspace_buffer
    assert ibuf.data_ptr() != wrapper._int_workspace_buffer.data_ptr()
    q = batch["q"]
    kv = batch["k_cache"], batch["v_cache"]
    sq = torch.arange(1, H_QO + 1, dtype=torch.float32, device=device)
    sk = torch.ones(H_KV, dtype=torch.float32, device=device)
    sv = 3 * sk
    out = wrapper.run(q, kv, sq, sk, sv)
    before = _plan_wrapper(batch, device, backend="fa3").run(q, kv, sq, sk, sv)
    assert torch.equal(out, before)
    graph = _capture(wrapper, q, kv, out, sq, sk, sv)

    def fail(f, i, pin, *args):
        assert i.data_ptr() != ibuf.data_ptr()
        i.zero_()
        raise ValueError("injected FA3 plan failure")

    changed_ids = ids.flip([0])
    changed_last = last - 1
    with monkeypatch.context() as m:
        m.setattr(native, "plan", fail)
        with pytest.raises(ValueError, match="injected FA3 plan failure"):
            _graph_plan(wrapper, qo, ki, changed_ids, changed_last)
    _poison_and_replay(graph, out)
    assert torch.equal(out, before)
    assert torch.equal(wrapper.run(q, kv, sq, sk, sv), before)

    _graph_plan(wrapper, qo, ki, changed_ids, changed_last)
    stock = _plan_wrapper(
        dict(batch, kv_indices=changed_ids, last_page_len=changed_last),
        device,
        backend="fa3",
    )
    expected = stock.run(q, kv, sq, sk, sv)
    assert not torch.equal(expected, before)
    assert torch.equal(wrapper.run(q, kv, sq, sk, sv), expected)
    _poison_and_replay(graph, out)
    assert torch.equal(out, expected)


def test_retained_graph_replay_follows_an_in_contract_replan():
    """A replan the specialized backend serves must refresh the captured
    plan-time metadata in place: the retained graph's replay computes the
    NEW plan (reviewer probe: real replay, not pointer checks)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch, wrapper = _graph_batch_and_wrapper(device)
    _graph_plan(
        wrapper,
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
    )
    kv = (batch["k_cache"], batch["v_cache"])
    out = torch.empty(batch["q"].shape, dtype=torch.bfloat16, device=device)
    graph = _capture(wrapper, batch["q"], kv, out)
    assert _gqa_dec_fp8q_sm90_stats()["runs_dispatched_capturing"] == 1
    # in-contract replan: same geometry, permuted page ids and a shorter kv
    # (request 1 drops to 3 pages / 48 keys)
    new_indices = batch["kv_indices"].flip([0])[:8].contiguous()
    new_lens = [80, 48]
    _graph_plan(
        wrapper,
        batch["qo_indptr"],
        torch.tensor([0, 5, 8], dtype=torch.int32, device=device),
        new_indices,
        torch.tensor(
            [(k - 1) % PAGE + 1 for k in new_lens], dtype=torch.int32, device=device
        ),
    )
    eager = wrapper.run(batch["q"], kv)
    _poison_and_replay(graph, out)
    assert torch.equal(out, eager)
    assert _matched_ratio(out, _ref_with_indices(batch, new_indices, new_lens)) >= (
        MATCHED
    )


@pytest.mark.parametrize(
    "change", ["disabled", "dtype", "mask", "prepare", "query_lengths", "short_auto"]
)
def test_refused_replan_preserves_graph_and_eager(monkeypatch, change):
    import flashinfer.prefill as fp

    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch, wrapper = _graph_batch_and_wrapper(device)
    _graph_plan(
        wrapper,
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
    )
    kv = (batch["k_cache"], batch["v_cache"])
    out = torch.empty(batch["q"].shape, dtype=torch.bfloat16, device=device)
    graph = _capture(wrapper, batch["q"], kv, out)
    pre_eager = wrapper.run(batch["q"], kv).clone()
    assert _gqa_dec_fp8q_sm90_stats()["runs_dispatched_capturing"] == 1
    opts = dict(
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    ki = batch["kv_indptr"]
    ids = batch["kv_indices"].flip([0])
    last = batch["last_page_len"]
    with monkeypatch.context() as m:
        if change == "short_auto":
            ki = ki.new_tensor([0, 4, 8])
            ids = ids[:8]
            last = last.new_tensor([16, 16])
        elif change == "disabled":
            m.setenv(DISABLE_ENV, "1")
        elif change == "dtype":
            opts["q_data_type"] = torch.bfloat16
        elif change == "mask":
            opts["custom_mask"] = torch.ones(
                2 * QL * 80, device=device, dtype=torch.bool
            )
        elif change == "query_lengths":
            opts["seq_lens_q"] = torch.tensor([5, 7])
        else:

            def fail(*args, **kwargs):
                raise ValueError("injected block table failure")

            m.setattr(fp, "_build_block_tables_from_paged_kv_indices", fail)
        with pytest.raises(ValueError):
            wrapper.plan(
                batch["qo_indptr"],
                ki,
                ids,
                last,
                H_QO,
                H_KV,
                HEAD_DIM,
                PAGE,
                **opts,
            )
    assert torch.equal(wrapper._paged_kv_indices_buf, batch["kv_indices"])
    assert torch.equal(wrapper._paged_kv_indptr_buf, batch["kv_indptr"])
    assert torch.equal(wrapper._paged_kv_last_page_len_buf, batch["last_page_len"])
    assert torch.equal(wrapper.run(batch["q"], kv), pre_eager)
    assert _matched_ratio(pre_eager, _reference(batch)) >= MATCHED
    _poison_and_replay(graph, out)
    # never a mix: the invalidated capture still computes the previous plan
    assert torch.equal(out, pre_eager)


def test_masked_eager_to_unmasked_verify_and_graph_retained_mask_size():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([80], device)
    ws = torch.empty(8 << 20, dtype=torch.uint8, device=device)
    eager = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws)
    args = (
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
    )
    eager.plan(
        *args,
        custom_mask=torch.ones(QL * 80, dtype=torch.bool, device=device),
        q_data_type=torch.bfloat16,
    )
    _graph_plan(eager, *args[:4])
    assert eager._gqa_dec_fp8q_sm90_state is not None
    assert (
        _matched_ratio(
            eager.run(batch["q"], (batch["k_cache"], batch["v_cache"])),
            _reference(batch),
        )
        >= MATCHED
    )

    qo = batch["qo_indptr"].new_tensor([0, 65])
    graph_args = (qo, *args[1:])
    graph = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws,
        "NHD",
        True,
        qo.clone(),
        args[1].clone(),
        args[2].clone(),
        args[3].clone(),
        torch.full((650,), 255, dtype=torch.uint8, device=device),
        qo.new_tensor([0, 650]),
    )
    sizes = graph.workspace_size(*graph_args, q_data_type=torch.bfloat16)
    graph.reset_workspace_buffer(
        *(torch.empty(n, dtype=torch.uint8, device=device) for n in sizes)
    )
    graph.plan(*graph_args, q_data_type=torch.bfloat16)
    assert graph._backend == "fa2"
    assert graph._workspace_size == sizes[0]


def test_graph_q_partition_replan_is_rejected_before_state_changes():
    """A replan with a non-verify Q partition on a wrapper whose retained
    fa2 capture cannot serve an fp8 query must be rejected by the stock
    resolver before any accepted state changes (the capture stays valid)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch, wrapper = _graph_batch_and_wrapper(device)
    _graph_plan(
        wrapper,
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
    )
    kv = (batch["k_cache"], batch["v_cache"])
    out = torch.empty(batch["q"].shape, dtype=torch.bfloat16, device=device)
    graph = _capture(wrapper, batch["q"], kv, out)
    eager = wrapper.run(batch["q"], kv).clone()
    qo2 = torch.tensor([0, 4, 12], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="CUDA graph backend"):
        _graph_plan(
            wrapper,
            qo2,
            batch["kv_indptr"],
            batch["kv_indices"],
            batch["last_page_len"],
        )
    # the rejected replan changed nothing: the capture still replays the
    # accepted plan bitwise
    _poison_and_replay(graph, out)
    assert torch.equal(out, eager)


@pytest.mark.parametrize(
    "lens,admitted",
    [
        ([16], False),
        ([64], False),
        ([16, 64], False),
        ([2048], True),
        ([16, 8192, 48, 128], True),
    ],
)
def test_auto_whole_batch_short_preference(lens, admitted):
    cu = [i * QL for i in range(len(lens) + 1)]
    ok, reason, meta = _gqa_dec_fp8q_sm90_auto_admits(
        **_admit_kwargs(len(lens), cu, lens)
    )
    assert ok is admitted
    if admitted:
        assert reason is None and meta is not None
    else:
        assert reason == "all KV lengths <= 64 (prefer stock auto)" and meta is None


def test_gap_analytic_and_mutated_descale_replay():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _gap_batch(device)
    q, kv = b["q"], (b["k_cache"], b["v_cache"])
    wrapper = _plan_wrapper(b, device, use_cuda_graph=True, backend="specialized")
    qd = torch.tensor(
        [22 * math.sqrt(HEAD_DIM) / 256], device=device, dtype=torch.float32
    )
    vd = torch.tensor([2**18], device=device, dtype=torch.float32)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    graph = _capture(wrapper, q, kv, out, q_scale=qd, v_scale=vd)
    assert _gqa_dec_fp8q_sm90_stats()["runs_dispatched_capturing"] == 1
    n = torch.arange(42, 48, dtype=torch.float64, device=device).view(QL, 1, 1)
    previous = None
    for gap in (22, 24, 25):
        qd.fill_(gap * math.sqrt(HEAD_DIM) / 256)
        expected = (n * (448 * 2**18) / (math.exp(gap) + n)).to(torch.bfloat16)
        expected = expected.expand_as(out)
        assert torch.equal(_reference(b, q_descale=qd, v_descale=vd), expected)
        if gap >= 24:
            # Isolate the old operand rounding: shifted sums did NOT round to fp16.
            p = 2048 * math.exp(-gap)
            half_p = torch.tensor(p, dtype=torch.float16).item()
            assert half_p == (2**-24 if gap == 24 else 0.0)
            old = (n * half_p * (448 * 2**18) / (2048 + n * p)).to(torch.bfloat16)
            assert _matched_ratio(old.expand_as(out), expected) == 0.0
        eager = wrapper.run(q, kv, q_scale=qd, v_scale=vd)
        _poison_and_replay(graph, out)
        assert torch.equal(out, eager)
        assert bool((out.float() > 0).all())
        torch.testing.assert_close(out.float(), expected.float(), rtol=RTOL, atol=ATOL)
        if previous is not None:
            assert not torch.equal(out, previous)
        previous = out.clone()


def test_signed_packed_v_lanes_keep_order():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make_batch([48], device)
    b["q"].zero_()
    # 128 distinct signed e4m3 values, including subnormals, across both V halves.
    raw = torch.arange(1, 65, dtype=torch.uint8, device=device)
    lanes = torch.stack((raw, raw | 128), dim=1).reshape(-1).view(torch.float8_e4m3fn)
    b["v_cache"].copy_(lanes.expand_as(b["v_cache"]))
    out = _plan_wrapper(b, device).run(b["q"], (b["k_cache"], b["v_cache"]))
    assert torch.equal(out, lanes.to(torch.bfloat16).expand_as(out))


@pytest.mark.parametrize("lens", [[16], [64], [16, 64]])
def test_all_short_auto_is_native_fa3(lens, monkeypatch):
    import flashinfer.prefill as fp

    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make_batch(lens, device)
    a = _plan_wrapper(b, device, backend="auto")
    f = _plan_wrapper(b, device, backend="fa3")
    assert a._backend == f._backend == "fa3"
    assert a._cached_q_data_type == torch.float8_e4m3fn
    assert not getattr(a, "_gqa_dec_fp8q_sm90_requested", False)
    assert getattr(a, "_verify_native", None) is None
    q, kv = b["q"], (b["k_cache"], b["v_cache"])
    sq = torch.arange(1, H_QO + 1, dtype=torch.float32, device=device)
    sk = torch.ones(H_KV, dtype=torch.float32, device=device)
    for args, kw in [
        ((), {}),
        ((sq, 2 * sk, 3 * sk), {}),
        ((), dict(q_scale=0.5, k_scale=1.5, v_scale=3.0)),
    ]:
        assert torch.equal(a.run(q, kv, *args, **kw), f.run(q, kv, *args, **kw))
    s = _gqa_dec_fp8q_sm90_stats()
    assert s["plans_prepared"] == s["workspace_allocs"] == 0
    assert s["runs_dispatched"] == s["runs_dequant_fallback"] == 0

    def size():
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(a._float_workspace_buffer)
        return w.workspace_size(
            b["qo_indptr"],
            b["kv_indptr"],
            b["kv_indices"],
            b["last_page_len"],
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )

    sized = size()
    with monkeypatch.context() as m:
        m.setattr(
            fp,
            "_gqa_dec_fp8q_sm90_auto_routes",
            lambda *a, **kw: (False, "stock control"),
        )
        assert sized == size()


@pytest.mark.parametrize("lens", [[2048], [16, 8192, 48, 128]])
def test_long_and_mixed_auto_still_dispatch(lens):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make_batch(lens, device)
    wrapper = _plan_wrapper(b, device, backend="auto")
    out = wrapper.run(b["q"], (b["k_cache"], b["v_cache"]))
    assert _gqa_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out, _reference(b)) >= MATCHED


def test_short_stock_graph_does_not_switch_backend_on_growth():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b, wrapper = _graph_batch_and_wrapper(device, (80,))
    qo, ids, last = b["qo_indptr"], b["kv_indices"], b["last_page_len"]
    short = dict(b, kv_indptr=qo.new_tensor([0, 4]), kv_indices=ids[:4])
    _graph_plan(wrapper, qo, short["kv_indptr"], short["kv_indices"], last)
    assert wrapper._backend == "fa3"
    q, kv = b["q"], (b["k_cache"], b["v_cache"])
    before = _plan_wrapper(short, device, backend="fa3").run(q, kv)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    graph = _capture(wrapper, q, kv, out)
    _poison_and_replay(graph, out)
    assert torch.equal(out, before)
    try:
        _graph_plan(wrapper, qo, b["kv_indptr"], ids, last)
    except ValueError as exc:
        assert "CUDA graph backend" in str(exc)
        expected = before
    else:
        expected = _plan_wrapper(b, device, backend="fa3").run(q, kv)
        assert not torch.equal(expected, before)
    assert wrapper._backend == "fa3"
    assert not getattr(wrapper, "_gqa_dec_fp8q_sm90_requested", False)
    assert torch.equal(wrapper.run(q, kv), expected)
    _poison_and_replay(graph, out)
    assert torch.equal(out, expected)
