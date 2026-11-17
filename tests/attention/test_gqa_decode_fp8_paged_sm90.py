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
    if torch.cuda.get_device_capability() != (9, 0):
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
    assert _gqa_dec_fp8q_sm90_backend_order("specialized", False) == (
        "specialized",
        "composable",
    )
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


def _reference(batch):
    """fp32 dequant-then-attend verify-decode reference (row pos of request r
    sees kv_len_r - 6 + pos + 1 keys)."""
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
        sc = torch.einsum("nhd,ld->nhl", q[q0:q1], k) * scale
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

    # kill-switch: the same call is served by dequant + stock fa2
    monkeypatch.setenv(DISABLE_ENV, "1")
    out_stock = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_disabled"] == 1 and stats["runs_dequant_fallback"] == 1, stats
    assert _matched_ratio(out_stock, ref) >= MATCHED
    assert _matched_ratio(out, out_stock) >= MATCHED


def test_non_uniform_q_len_takes_the_stock_path():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    # break uniformity: 5 + 7 + 6 rows instead of 6 + 6 + 6
    batch["qo_indptr"] = torch.tensor([0, 5, 12, 18], dtype=torch.int32, device=device)
    wrapper = _plan_wrapper(batch, device)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0 and stats["plans_ineligible"] == 1, stats
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_dequant_fallback"] == 1
    assert out.shape == batch["q"].shape


def test_legacy_kill_switch_alias_still_disables(monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device)
    monkeypatch.setenv(DISABLE_ENV_LEGACY, "1")
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_disabled"] == 1 and stats["runs_dequant_fallback"] == 1, stats
    assert stats["runs_dispatched"] == 0
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


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


def test_run_guard_falls_back_on_mismatched_arguments():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device)
    # return_lse is outside the contract: dequant fallback serves it
    out, lse = wrapper.run(
        batch["q"], (batch["k_cache"], batch["v_cache"]), return_lse=True
    )
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_ineligible"] == 1 and stats["runs_dequant_fallback"] == 1
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_graph_capture_replays_bitwise_equal_to_eager():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([700, 76, 341, 1363], device)
    wrapper = _plan_wrapper(batch, device, use_cuda_graph=True)
    q = batch["q"]
    kv = (batch["k_cache"], batch["v_cache"])
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    eager = wrapper.run(q, kv).clone()
    assert _gqa_dec_fp8q_sm90_stats()["runs_dispatched"] == 1

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, kv, out=out)  # warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        wrapper.run(q, kv, out=out)
    stats = _gqa_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched_capturing"] == 1, stats

    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)

    # new content through the same buffers replays correctly
    q.copy_(_make_batch([700, 76, 341, 1363], device, seed=7)["q"])
    graph.replay()
    torch.cuda.synchronize()
    eager2 = wrapper.run(q, kv)
    assert torch.equal(out, eager2)
