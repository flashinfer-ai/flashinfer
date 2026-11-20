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
    GQA_CTX_FP8Q_SM90_BACKENDS,
    _gqa_ctx_fp8q_sm90_auto_routes,
    _gqa_ctx_fp8q_sm90_backend_order,
    _gqa_ctx_fp8q_sm90_plan_eligibility,
    _gqa_ctx_fp8q_sm90_stats,
    _gqa_fp8q_sm90_dispatch_table,
    _reset_gqa_ctx_fp8q_sm90_stats,
)

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISABLE_ENV_LEGACY = "FLASHINFER_DISABLE_GQA_CONTEXT_FP8_PAGED_SM90"
H_QO, H_KV, HEAD_DIM, PAGE = 8, 1, 128, 16
RTOL, ATOL, MATCHED = 0.04, 0.02, 0.98


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV_LEGACY, raising=False)
    _reset_gqa_ctx_fp8q_sm90_stats()
    yield
    _reset_gqa_ctx_fp8q_sm90_stats()


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
        total_q=5379,
    )
    kwargs.update(overrides)
    return kwargs


def test_plan_guard_accepts_the_measured_contract():
    assert _gqa_ctx_fp8q_sm90_plan_eligibility(**_plan_kwargs()) is None
    assert (
        _gqa_ctx_fp8q_sm90_plan_eligibility(
            **_plan_kwargs(sm_scale=1.0 / math.sqrt(HEAD_DIM))
        )
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
        {"batch_size": 129},
        {"total_q": 0},
        {"total_q": 32769},
    ],
)
def test_plan_guard_rejects_other_geometries(overrides):
    assert _gqa_ctx_fp8q_sm90_plan_eligibility(**_plan_kwargs(**overrides)) is not None


def test_stats_report_build_state_and_counters():
    stats = _gqa_ctx_fp8q_sm90_stats()
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
    assert stats["backends"] == GQA_CTX_FP8Q_SM90_BACKENDS
    assert set(stats["backend_dispatches"]) == {"specialized", "composable"}


# --------------------------------------------------------------------------
# Backend selection surface (#5667 pattern, GPU-free)
# --------------------------------------------------------------------------


def test_backends_tuple_is_plain_strings_with_specialized():
    assert "specialized" in GQA_CTX_FP8Q_SM90_BACKENDS
    assert "auto" in GQA_CTX_FP8Q_SM90_BACKENDS
    assert all(isinstance(b, str) for b in GQA_CTX_FP8Q_SM90_BACKENDS)


def test_dispatch_table_loads_and_matches_guard_constants():
    table = _gqa_fp8q_sm90_dispatch_table()
    assert table is not None
    assert table["kill_switch"] == DISABLE_ENV
    assert "specialized" in table["backends"]
    op = table["ops"]["context"]
    assert op["num_qo_heads"] == H_QO
    assert op["num_kv_heads"] == H_KV
    assert op["head_dim"] == HEAD_DIM
    assert op["page_size"] == PAGE
    assert op["kv_layout"] == "NHD"
    assert op["causal"] is True
    env = op["envelope"]
    assert env["max_batch_size"] == 128
    assert env["max_total_q"] == 32768
    assert env["max_kv_len"] == 262144
    assert env["max_referenced_pages"] == 131072


def test_auto_routes_only_on_the_allowlist():
    ok, reason = _gqa_ctx_fp8q_sm90_auto_routes(
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
        ok, reason = _gqa_ctx_fp8q_sm90_auto_routes(**kwargs)
        assert not ok and reason, bad


def test_backend_order_never_swaps_an_explicit_stock_backend():
    # explicit "specialized" never falls back: it runs the kernel or raises
    assert _gqa_ctx_fp8q_sm90_backend_order("specialized", False) == ("specialized",)
    # auto: "specialized" only on the allowlist
    assert _gqa_ctx_fp8q_sm90_backend_order("auto", True) == (
        "specialized",
        "composable",
    )
    assert _gqa_ctx_fp8q_sm90_backend_order("auto", False) == ("composable",)
    # an explicitly named stock backend never sees "specialized"
    for stock in ("fa2", "fa3", "cudnn", "trtllm-gen", "cute-dsl"):
        assert "specialized" not in _gqa_ctx_fp8q_sm90_backend_order(stock, True)


# --------------------------------------------------------------------------
# Device tests (SM90): reference equivalence, kill-switch, CUDA graph
# --------------------------------------------------------------------------


def _make_batch(q_lens, kv_lens, device, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    B = len(q_lens)
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


def _reference(batch, qk_descale=1.0, sm_scale=None):
    """fp32 dequant-then-attend chunked-prefill reference (row pos of request
    r sees kv_len_r - q_len_r + pos + 1 keys). ``qk_descale`` folds a
    per-sequence q_descale * k_descale product into the logits: a scalar
    applies uniformly, a sequence applies its r-th value to request r."""
    device = batch["q"].device
    q = batch["q"].float()
    k_flat = batch["k_cache"].float().reshape(-1, HEAD_DIM)
    v_flat = batch["v_cache"].float().reshape(-1, HEAD_DIM)
    cu = batch["qo_indptr"].tolist()
    kv_lens = batch["seq_lens"].tolist()
    bt = batch["block_table"]
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    base_scale = HEAD_DIM**-0.5 if sm_scale is None else sm_scale
    off = torch.arange(PAGE, device=device)
    for r in range(len(kv_lens)):
        q0, q1 = cu[r], cu[r + 1]
        ql = q1 - q0
        if ql == 0:
            continue
        descale_r = (
            qk_descale[r]
            if isinstance(qk_descale, (list, tuple))
            else (qk_descale[r].item() if torch.is_tensor(qk_descale) else qk_descale)
        )
        scale = base_scale * descale_r
        kvl = kv_lens[r]
        npages = (kvl + PAGE - 1) // PAGE
        flat = (bt[r, :npages].long().unsqueeze(1) * PAGE + off).reshape(-1)[:kvl]
        k = k_flat[flat]
        v = v_flat[flat]
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
    "q_lens,kv_lens",
    [
        ([6, 6, 6, 5, 1, 5, 350], [1500, 658, 76, 341, 341, 863, 2030]),
        ([256], [4096]),
        ([5, 90, 120, 7, 61], [584, 727, 120, 75, 1468]),
    ],
)
def test_specialized_matches_reference_and_stock(q_lens, kv_lens, monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch(q_lens, kv_lens, device)
    wrapper = _plan_wrapper(batch, device)
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 1, stats

    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    ref = _reference(batch)
    assert _matched_ratio(out, ref) >= MATCHED

    # determinism: the kernel is atomics-free and bit-stable
    out2 = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    assert torch.equal(out, out2)

    # kill-switch: the explicit plan raises (never falls back), an auto-armed
    # plan serves the same call by dequant + stock fa2
    monkeypatch.setenv(DISABLE_ENV, "1")
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_disabled"] == 1, stats
    monkeypatch.delenv(DISABLE_ENV)
    wrapper_auto = _plan_wrapper(batch, device, backend="auto")
    monkeypatch.setenv(DISABLE_ENV, "1")
    out_stock = wrapper_auto.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dequant_fallback"] == 1, stats
    assert _matched_ratio(out_stock, ref) >= MATCHED
    assert _matched_ratio(out, out_stock) >= MATCHED


def test_legacy_kill_switch_alias_still_disables(monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    monkeypatch.setenv(DISABLE_ENV_LEGACY, "1")
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_disabled"] == 1 and stats["runs_dequant_fallback"] == 1, stats
    assert stats["runs_dispatched"] == 0
    assert _matched_ratio(out, _reference(batch)) >= MATCHED
    # explicit "specialized" raises under the alias too — at plan time: the
    # kill switch is read at plan and run, so the explicit plan itself refuses
    with pytest.raises(NotImplementedError, match="never falls back"):
        _plan_wrapper(batch, device)


def test_explicit_specialized_raises_for_contracts_it_has_no_kernel_for():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    batch = _make_batch([5, 7], [584, 75], device)
    with pytest.raises(ValueError, match="specialized"):
        _plan_wrapper(batch, device, backend="specialized", q_data_type=torch.bfloat16)
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0 and stats["runs_dispatched"] == 0


def test_explicit_stock_backend_is_never_swapped():
    """A caller who names a stock backend must never be served by the
    specialized kernel — even on an allowlisted shape."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    # fp8-Q with backend="fa2": stock behavior, whatever it is, is never this
    # kernel (stock fa2 has no fp8-Q contract on SM90, so the plan or run is
    # expected to fail loudly in stock code).
    try:
        wrapper = _plan_wrapper(batch, device, backend="fa2")
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    except Exception:
        pass
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats
    assert stats["runs_dispatched"] == 0, stats
    assert stats["backend_dispatches"]["specialized"] == 0, stats


def test_auto_routes_to_specialized_on_the_allowlist_only():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 1, stats
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert _matched_ratio(out, _reference(batch)) >= MATCHED

    # off-allowlist geometry: auto never routes to "specialized"
    _reset_gqa_ctx_fp8q_sm90_stats()
    batch4 = _make_batch([5, 7], [584, 75], device)
    batch4["q"] = batch4["q"][:, :4, :].contiguous()
    # stock auto may reject fp8-Q; it must simply never arm us
    with contextlib.suppress(Exception):
        _plan_wrapper(batch4, device, backend="auto", num_qo_heads=4)
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats
    assert any(r.startswith("auto:") for r in stats["fallback_reasons"]), stats


def test_run_guard_declines_on_mismatched_arguments():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    # return_lse is outside the contract: explicit "specialized" raises
    wrapper = _plan_wrapper(batch, device)
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]), return_lse=True)
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0 and stats["runs_ineligible"] == 1
    # an auto-armed plan is served by the dequant fallback
    wrapper_auto = _plan_wrapper(batch, device, backend="auto")
    out, lse = wrapper_auto.run(
        batch["q"], (batch["k_cache"], batch["v_cache"]), return_lse=True
    )
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_ineligible"] == 2 and stats["runs_dequant_fallback"] == 1
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_graph_capture_replays_bitwise_equal_to_eager():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([6, 6, 120, 70], [700, 76, 341, 1363], device)
    wrapper = _plan_wrapper(batch, device, use_cuda_graph=True)
    q = batch["q"]
    kv = (batch["k_cache"], batch["v_cache"])
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
    eager = wrapper.run(q, kv).clone()
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        wrapper.run(q, kv, out=out)  # warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        wrapper.run(q, kv, out=out)
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched_capturing"] == 1, stats

    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)

    # new content through the same buffers replays correctly
    q.copy_(_make_batch([6, 6, 120, 70], [700, 76, 341, 1363], device, seed=7)["q"])
    graph.replay()
    torch.cuda.synchronize()
    eager2 = wrapper.run(q, kv)
    assert torch.equal(out, eager2)


# --------------------------------------------------------------------------
# Review regressions: kernel numerics clamps, replan capacity, guard surface
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_len,kv_len", [(4, 4096), (12, 4096), (32, 4096), (8192, 32768)]
)
def test_raw_score_range_and_unit_pv(q_len, kv_len):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make_batch([q_len], [kv_len], device)
    for name in ("q", "k_cache"):
        b[name].view(torch.uint8).fill_(0x60)  # 32; raw QK = 131072
    b["v_cache"].view(torch.uint8).fill_(0x38)  # constant 1
    w = _plan_wrapper(b, device, backend="auto")
    if q_len == 8192:
        assert w._gqa_ctx_fp8q_sm90_state.psplit == 1
    scale = torch.tensor([[1 / 32]], device=device)
    out = w.run(b["q"], (b["k_cache"], b["v_cache"]), q_scale=scale, k_scale=scale)
    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
    assert _gqa_ctx_fp8q_sm90_stats()["backend_dispatches"] == {
        "specialized": 1,
        "composable": 0,
    }


@pytest.mark.parametrize("backend", ["auto", "specialized"])
def test_refused_captured_replan_preserves_last_plan(backend, monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    b = _make_batch([32], [64], device)
    w = _plan_wrapper(b, device, use_cuda_graph=True, backend=backend)
    q = b["q"]
    kv = (b["k_cache"], b["v_cache"])
    eager = w.run(q, kv)
    out = torch.empty_like(eager)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        w.run(q, kv, out=out)

    # kv_indices is also the wrapper's writable graph buffer. Freeze the
    # requested order before even the injected common-planner refusal.
    replan_indices = b["kv_indices"].clone().flip(0)

    def plan(rows, **opts):
        w.plan(
            b["qo_indptr"].new_tensor([0, rows]),
            b["kv_indptr"],
            replan_indices,
            b["last_page_len"],
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
            **opts,
        )

    state = w._gqa_ctx_fp8q_sm90_state
    for rows, disabled, opts, reason in (
        (16, False, {}, "launch schedule"),
        (24, True, {}, "kill switch"),
        (24, False, {"disable_split_kv": True}, "disable_split_kv"),
    ):
        with monkeypatch.context() as patch:
            if disabled:
                patch.setenv(DISABLE_ENV, "1")
            with pytest.raises(NotImplementedError, match=reason):
                plan(rows, **opts)
        assert w._gqa_ctx_fp8q_sm90_state is state
        assert torch.equal(w.run(q, kv), eager)
        out.fill_(17)
        g.replay()
        assert torch.equal(out, eager)

    # Admission succeeds, then the common FA2 planner refuses. Private
    # captured metadata and the last accepted eager plan must both survive.
    from types import SimpleNamespace

    import flashinfer.prefill as prefill

    accepted_module = w._cached_module
    accepted_plan = w._plan_info
    old_ref = _reference(b)
    old_cu = b["qo_indptr"].clone()
    old_indices = b["kv_indices"].clone()
    workspaces = (
        w._int_workspace_buffer,
        w._pin_memory_int_workspace_buffer,
    )
    torch.cuda.current_stream(device).synchronize()
    markers = [buf[:16].clone() for buf in workspaces]

    def refuse_common_plan(*args):
        accepted_module.plan(*args)
        # Dirty explicit workspace bytes even if both geometries happened
        # to generate the same native schedule. Queue the DMA before raising.
        args[1][:16].copy_(args[2][:16], non_blocking=True)
        args[1][:16].bitwise_xor_(0x5A)
        args[2][16:32].bitwise_not_()
        raise RuntimeError("injected common planner refusal")

    pin_tail = workspaces[1][16:32].clone()
    with monkeypatch.context() as patch:
        patch.setattr(
            prefill,
            "get_batch_prefill_module",
            lambda *args: SimpleNamespace(plan=refuse_common_plan),
        )
        with pytest.raises(RuntimeError, match="injected common planner refusal"):
            plan(24)
    assert w._cached_module is accepted_module and w._plan_info is accepted_plan
    assert torch.equal(b["qo_indptr"], old_cu)
    assert torch.equal(b["kv_indices"], old_indices)
    assert all(
        torch.equal(buf[:16], mark)
        for buf, mark in zip(workspaces, markers, strict=True)
    )
    assert torch.equal(workspaces[1][16:32], pin_tail)

    # Exercise a failure *after* native plan_info was produced and the first
    # retained specialized metadata copy was enqueued.
    build = prefill._build_block_tables_from_paged_kv_indices

    def refuse_refresh(*args, **kwargs):
        build(*args, **kwargs)
        raise RuntimeError("injected late metadata refusal")

    with monkeypatch.context() as patch:
        patch.setattr(
            prefill, "_build_block_tables_from_paged_kv_indices", refuse_refresh
        )
        with pytest.raises(RuntimeError, match="injected late metadata refusal"):
            plan(24)
    assert w._cached_module is accepted_module and w._plan_info is accepted_plan
    assert w._gqa_ctx_fp8q_sm90_state is state
    assert torch.equal(w.run(q, kv), eager)
    out.fill_(17)
    g.replay()
    assert torch.equal(out, eager)
    if backend == "auto":
        before = _gqa_ctx_fp8q_sm90_stats()["backend_dispatches"]
        with monkeypatch.context() as patch:
            patch.setenv(DISABLE_ENV, "1")
            stock = w.run(q, kv)
        assert _matched_ratio(stock, old_ref) >= MATCHED
        assert _gqa_ctx_fp8q_sm90_stats()["backend_dispatches"] == {
            "specialized": before["specialized"],
            "composable": before["composable"] + 1,
        }

    plan(24)
    pool = w._fp8_replan_backup_pool
    assert sum(t.nbytes for t in pool if t.device.type == "cpu") == workspaces[1].nbytes
    assert not any(t.is_pinned() for t in pool if t.device.type == "cpu")
    assert sum(t.nbytes for t in pool if t.is_cuda) == workspaces[0].nbytes + 72
    plan(24)
    assert len(w._fp8_replan_backup_pool) == len(pool)
    assert all(a is b for a, b in zip(pool, w._fp8_replan_backup_pool, strict=True))
    ref = dict(
        b,
        q=q[:24],
        qo_indptr=b["qo_indptr"].new_tensor([0, 24]),
        block_table=replan_indices.view(1, -1),
    )
    out.fill_(17)
    g.replay()
    assert torch.equal(out[:24], w.run(q[:24], kv))
    assert _matched_ratio(out[:24], _reference(ref)) >= MATCHED


def test_uniform_large_negative_scores_keep_softmax_weight():
    """Rows whose scores are a constant -16384 must return the value row
    (softmax is uniform), not 0: the renormaliser clamps in scaled units, not
    at a raw -1e4."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    # covers packed decode (q_len 4), regular decode (q_len 12) and the
    # (possibly split) prefill tiles (q_len 60)
    batch = _make_batch([4, 12, 60], [640, 320, 700], device)
    batch["q"] = torch.full_like(batch["q"].float(), 16.0).to(torch.float8_e4m3fn)
    batch["k_cache"] = torch.full_like(batch["k_cache"].float(), -8.0).to(
        torch.float8_e4m3fn
    )
    batch["v_cache"] = torch.ones_like(batch["v_cache"].float()).to(torch.float8_e4m3fn)
    # every score is 16 * -8 * 128 = -16384 (all partial sums inside f16)
    wrapper = _plan_wrapper(batch, device)
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert torch.isfinite(out.float()).all()
    # uniform softmax over V == 1 gives exactly 1; allow the disclosed e4m3
    # P-quantisation error
    assert (out.float() - 1.0).abs().max().item() < 0.15


def test_tiny_descales_stay_finite_and_uniform():
    """A pathologically small q/k descale product must not round the packed
    softmax scale to zero (masked -inf * 0 = NaN); the kernel clamps the scale
    at the f16 minimum, the correct uniform-softmax limit."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device)
    descale = torch.full((3,), 1e-4, dtype=torch.float32, device=device)
    out = wrapper.run(
        batch["q"],
        (batch["k_cache"], batch["v_cache"]),
        q_scale=descale,
        k_scale=descale,
    )
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert torch.isfinite(out.float()).all()
    ref = _reference(batch, qk_descale=1e-8)
    assert _matched_ratio(out, ref) >= MATCHED


def test_replan_to_fewer_pages_keeps_dispatching():
    """After a large plan, a smaller replan must only require the pages it
    references: the V^T scratch is sliced to the plan's coverage, so a small
    v_cache is accepted."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    big = _make_batch([6, 6, 120], [2030, 1500, 863], device)
    wrapper = _plan_wrapper(big, device)
    out = wrapper.run(big["q"], (big["k_cache"], big["v_cache"]))
    assert _matched_ratio(out, _reference(big)) >= MATCHED

    small = _make_batch([5, 7, 9], [75, 84, 60], device)
    wrapper.plan(
        small["qo_indptr"],
        small["kv_indptr"],
        small["kv_indices"],
        small["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    # the small caches hold exactly the pages the small plan references
    out = wrapper.run(small["q"], (small["k_cache"], small["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 2, stats
    assert _matched_ratio(out, _reference(small)) >= MATCHED


def test_graph_mode_replan_never_swaps_buffer_pointers():
    """In CUDA-graph mode a replan must refresh the plan-owned buffers in
    place (a captured graph holds their pointers); a replan that would need
    bigger buffers declines the specialized kernel instead."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    wide = _make_batch([6, 6, 120], [2030, 1500, 863], device)
    narrow = _make_batch([5, 7, 9], [75, 84, 60], device)

    # wide first: the narrow replan reuses every buffer in place
    wrapper = _plan_wrapper(wide, device, use_cuda_graph=True)
    wrapper.run(wide["q"], (wide["k_cache"], wide["v_cache"]))
    bufs = wrapper._gqa_ctx_fp8q_sm90_bufs
    ptrs = {name: t.data_ptr() for name, t in bufs.items()}
    wrapper.plan(
        narrow["qo_indptr"],
        narrow["kv_indptr"],
        narrow["kv_indices"],
        narrow["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    assert wrapper._gqa_ctx_fp8q_sm90_state is not None
    # every retained buffer keeps its pointer (new names may be added: a
    # buffer that never existed has no captured pointer to preserve)
    assert {
        name: wrapper._gqa_ctx_fp8q_sm90_bufs[name].data_ptr() for name in ptrs
    } == ptrs
    out = wrapper.run(narrow["q"], (wide["k_cache"], wide["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 2
    small_ref = dict(narrow)
    small_ref["k_cache"], small_ref["v_cache"] = wide["k_cache"], wide["v_cache"]
    assert _matched_ratio(out, _reference(small_ref)) >= MATCHED

    # narrow first: a wider replan would need a pointer swap -> decline
    # (fresh batches: the replan above wrote through the graph-mode metadata
    # buffers, which _plan_wrapper aliases to the batch tensors)
    wide = _make_batch([6, 6, 120], [2030, 1500, 863], device)
    narrow = _make_batch([5, 7, 9], [75, 84, 60], device)
    _reset_gqa_ctx_fp8q_sm90_stats()
    ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws,
        kv_layout="NHD",
        backend="specialized",
        use_cuda_graph=True,
        max_total_num_rows=int(wide["qo_indptr"][-1]),
        qo_indptr_buf=wide["qo_indptr"].clone(),
        paged_kv_indptr_buf=wide["kv_indptr"].clone(),
        paged_kv_indices_buf=wide["kv_indices"].clone(),
        paged_kv_last_page_len_buf=wide["last_page_len"].clone(),
    )

    def _plan2(b):
        wrapper2.plan(
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

    _plan2(narrow)
    assert _gqa_ctx_fp8q_sm90_stats()["plans_prepared"] == 1
    bufs2 = wrapper2._gqa_ctx_fp8q_sm90_bufs
    ptrs2 = {name: t.data_ptr() for name, t in bufs2.items()}
    seq_lens_before = bufs2["seq_lens"].clone()
    block_table_before = bufs2["block_table"].clone()
    # fits the stock graph buffers, not the retained plan buffers: the
    # explicit plan refuses loudly without refreshing the retained buffers
    with pytest.raises(NotImplementedError, match="never falls back"):
        _plan2(wide)
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 1, stats  # only the narrow plan armed
    # No run was captured on wrapper2: refusal leaves it unarmed.
    assert wrapper2._gqa_ctx_fp8q_sm90_state is None
    assert "plan: graph-retained workspace capacity" in stats["fallback_reasons"]
    assert {
        name: wrapper2._gqa_ctx_fp8q_sm90_bufs[name].data_ptr() for name in ptrs2
    } == ptrs2
    # the refusal happened before any in-place refresh: a capture from the
    # accepted plan still reads a consistent seq_lens / block_table pair
    assert torch.equal(bufs2["seq_lens"], seq_lens_before)
    assert torch.equal(bufs2["block_table"], block_table_before)
    # the narrow contract replans and keeps dispatching
    _plan2(narrow)
    assert _gqa_ctx_fp8q_sm90_stats()["plans_prepared"] == 2
    out2 = wrapper2.run(narrow["q"], (narrow["k_cache"], narrow["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out2, _reference(narrow)) >= MATCHED


def test_kill_switch_before_plan_still_serves_tensor_q_scale(monkeypatch):
    """The row->request map must exist for every specialized-requested plan,
    so on an auto-armed plan a per-sequence q_scale tensor is served by the
    dequant fallback even when the kill switch was set before plan(); an
    explicit backend='specialized' plan raises at plan time instead."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    monkeypatch.setenv(DISABLE_ENV, "1")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    with pytest.raises(NotImplementedError, match="never falls back"):
        _plan_wrapper(batch, device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    # distinct non-unit per-request descales (exact powers of two): a scale
    # applied to the wrong request, or not at all, fails the comparison
    q_scale = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32, device=device)
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]), q_scale=q_scale)
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_dequant_fallback"] == 1, stats
    assert _matched_ratio(out, _reference(batch, qk_descale=[0.5, 1.0, 2.0])) >= MATCHED


def test_mutated_sm_scale_declines_specialization():
    """forward()-style mutators can rewrite _sm_scale after an armed plan;
    the fixed-scale kernel must decline: an auto-armed plan is served by the
    stock path, an explicit plan raises."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    wrapper._sm_scale = 0.5
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_ineligible"] == 1 and stats["runs_dequant_fallback"] == 1
    assert _matched_ratio(out, _reference(batch, sm_scale=0.5)) >= MATCHED
    wrapper._sm_scale = None
    out2 = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out2, _reference(batch)) >= MATCHED
    # the explicit form raises on the same decline
    wrapper_exp = _plan_wrapper(batch, device)
    wrapper_exp._sm_scale = 0.5
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper_exp.run(batch["q"], (batch["k_cache"], batch["v_cache"]))


def test_workspace_size_matches_the_armed_plan():
    """workspace_size(backend='specialized' or allowlisted fp8 'auto') must
    size the fa2 route the armed plan is actually planned on (bf16 query),
    and raise plan()'s contract error for contracts plan() would reject."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)

    def _ws(backend, q_dtype):
        ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", backend=backend
        )
        return wrapper.workspace_size(
            batch["qo_indptr"],
            batch["kv_indptr"],
            batch["kv_indices"],
            batch["last_page_len"],
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=q_dtype,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )

    stock = _ws("fa2", torch.bfloat16)
    specialized = _ws("specialized", torch.float8_e4m3fn)
    auto = _ws("auto", torch.float8_e4m3fn)
    assert specialized == stock
    assert auto == stock
    with pytest.raises(ValueError, match="specialized"):
        _ws("specialized", torch.bfloat16)


def test_batch_invariant_workspace_size_and_plan_keep_original_dtype():
    from flashinfer.batch_invariant import BatchInvariantUnsupportedError

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("batch-invariant attention requires SM100/SM103/SM107")
    device = torch.device("cuda:0")
    b = _make_batch([4], [16], device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(16 << 20, device=device, dtype=torch.uint8),
        backend="auto",
        batch_invariant=True,
    )
    args = (
        b["qo_indptr"],
        b["kv_indptr"],
        b["kv_indices"],
        b["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
    )
    opts = dict(causal=True, q_data_type=torch.bfloat16, o_data_type=torch.bfloat16)
    size, int_size = w.workspace_size(*args, **opts)
    assert size >= 0 and int_size == 0
    w.plan(*args, **opts)
    # This geometry would arm ordinary auto. Both public calls must refuse
    # rather than quoting bytes for a plan the invariant backend cannot run.
    opts.update(q_data_type=torch.float8_e4m3fn, kv_data_type=torch.float8_e4m3fn)
    for call in (w.workspace_size, w.plan):
        with pytest.raises(BatchInvariantUnsupportedError):
            call(*args, **opts)
    assert not w._gqa_ctx_fp8q_sm90_requested
    assert w._cached_q_data_type == torch.bfloat16


def test_auto_arming_declines_on_a_captured_fa3_graph():
    """An allowlisted fp8 auto plan must not rewrite the backend to fa2 when
    the wrapper's captured CUDA-graph backend is fa3 (the fa2 schedule would
    clobber the captured graph's integer workspace)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, use_cuda_graph=True, backend="auto")
    assert _gqa_ctx_fp8q_sm90_stats()["plans_prepared"] == 1
    _reset_gqa_ctx_fp8q_sm90_stats()
    # simulate a wrapper whose captured graphs hold an fa3 workspace layout
    wrapper._cuda_graph_backend = "fa3"
    with contextlib.suppress(Exception):
        wrapper.plan(
            batch["qo_indptr"],
            batch["kv_indptr"],
            batch["kv_indices"],
            batch["last_page_len"],
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats
    assert any("captured CUDA-graph backend" in r for r in stats["fallback_reasons"]), (
        stats
    )


def test_positional_fp8_scale_args_are_rejected_loudly():
    """The stock positional (q_scale, k_scale, v_scale) *args contract is
    consumed by fp8 stock kernels; the specialized-armed route dequantises Q
    to bf16, so silent dropping must be a loud error instead."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device)
    ones = torch.ones(3, dtype=torch.float32, device=device)
    with pytest.raises(NotImplementedError, match="positional"):
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]), ones, ones, ones)
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 0


def test_rejected_replan_does_not_arm_the_wrapper():
    """A replan that would arm the specialized kernel but is rejected by the
    seq_lens_q validation must leave the previously accepted (unarmed) plan's
    state and run() behavior unchanged: the arming decision is committed to
    the wrapper only after validation."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend="auto"
    )
    plan_args = (
        batch["qo_indptr"],
        batch["kv_indptr"],
        batch["kv_indices"],
        batch["last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
    )
    # accepted plan: bf16 query and KV on the stock route — never armed
    wrapper.plan(
        *plan_args,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )
    assert not wrapper._gqa_ctx_fp8q_sm90_requested
    # an fp8-Q replan that would arm, rejected by the seq_lens_q validation
    bad_lens = torch.tensor([6, 90, 7], dtype=torch.int32)  # != qo_indptr diffs
    with pytest.raises(ValueError, match="seq_lens_q"):
        wrapper.plan(
            *plan_args,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
            seq_lens_q=bad_lens,
        )
    # the rejected replan left the accepted plan unarmed: a bf16 run still
    # takes the stock path (a leaked armed flag would send it down the fp8-Q
    # branch and fail the cached-dtype check after dequantisation)
    assert not wrapper._gqa_ctx_fp8q_sm90_requested
    q_bf16 = batch["q"].float().to(torch.bfloat16)
    k_bf16 = batch["k_cache"].float().to(torch.bfloat16)
    v_bf16 = batch["v_cache"].float().to(torch.bfloat16)
    out = wrapper.run(q_bf16, (k_bf16, v_bf16))
    assert out.shape == batch["q"].shape
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 0


@pytest.mark.parametrize("backend", ["specialized", "auto"])
def test_armed_plan_rejects_non_fp8_query_at_run(backend):
    """The fallback's internal bf16 ABI is not the caller's query contract."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, backend=backend)
    q_bf16 = batch["q"].float().to(torch.bfloat16)
    error, reason = (
        (NotImplementedError, "never falls back")
        if backend == "specialized"
        else (ValueError, "q_data_type")
    )
    with pytest.raises(error, match=reason):
        wrapper.run(q_bf16, (batch["k_cache"], batch["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["backend_dispatches"] == {
        "specialized": 0,
        "composable": 0,
    }
    # valid e4m3 input still dispatches the specialized kernel
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_run_guard_requires_k_cache_page_coverage():
    """The kernel reads K and V at the same physical page ids: a K cache with
    fewer pages than the planned coverage is declined (explicit raises), the
    same rule the V^T coverage check applies to v_cache."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device)
    short_k = batch["k_cache"][:-1].clone()
    with pytest.raises(NotImplementedError, match="never falls back"):
        wrapper.run(batch["q"], (short_k, batch["v_cache"]))
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    assert any("k_cache pages" in r for r in stats["fallback_reasons"]), stats


def test_extreme_descales_match_reference():
    """Per-sequence descale products large enough that |max_score * scale|
    crosses any f16-budget clamp still return the exact softmax value: the
    renormaliser subtracts the raw-domain maximum before the scale folds in.
    (Formerly returned 0: the clamped scaled reference left the exponent
    centered thousands of log2 units above zero.)"""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([16], [16], device)
    for name in ("q", "k_cache", "v_cache"):
        batch[name].view(torch.uint8).fill_(0x38)  # e4m3 1.0
    wrapper = _plan_wrapper(batch, device)
    q_scale = torch.full((1,), 256.0, dtype=torch.float32, device=device)
    k_scale = torch.full((1,), 8.0, dtype=torch.float32, device=device)
    out = wrapper.run(
        batch["q"],
        (batch["k_cache"], batch["v_cache"]),
        q_scale=q_scale,
        k_scale=k_scale,
    )
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out, _reference(batch, qk_descale=[2048.0])) >= MATCHED
    # a descale product beyond f16 range saturates to the one-hot argmax
    # limit of softmax: finite output, never NaN/inf
    huge = torch.full((1,), 1e6, dtype=torch.float32, device=device)
    out2 = wrapper.run(
        batch["q"], (batch["k_cache"], batch["v_cache"]), q_scale=huge, k_scale=huge
    )
    assert bool(out2.float().isfinite().all())


def test_seq_lens_past_page_span_declines():
    """An explicit seq_lens that implies more pages than the request's listed
    span is refused at plan time: the kernel derives its page count from the
    kv length and would otherwise walk the block-table row's zero-padded tail
    (attending to page 0's data)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([16], [16], device)  # one listed page
    too_long = torch.tensor([32], dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError, match="page span"):
        ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", backend="specialized"
        )
        wrapper.plan(
            batch["qo_indptr"],
            batch["kv_indptr"],
            batch["kv_indices"],
            batch["last_page_len"],
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
            seq_lens=too_long,
        )
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert any("page span" in r for r in stats["fallback_reasons"]), stats


def test_disable_split_kv_declines_specialized():
    """disable_split_kv asks for split-free execution; the specialized kernel
    splits KV for short-query rows (and psplit prefill tiles), so it must not
    run: auto falls back to the FA2 plan (which honors the flag), explicit
    raises."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    ws = torch.empty(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend="specialized"
    )
    with pytest.raises(NotImplementedError, match="disable_split_kv"):
        wrapper.plan(
            batch["qo_indptr"],
            batch["kv_indptr"],
            batch["kv_indices"],
            batch["last_page_len"],
            H_QO,
            H_KV,
            HEAD_DIM,
            PAGE,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
            disable_split_kv=True,
        )
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats
    assert any("disable_split_kv" in r for r in stats["fallback_reasons"]), stats


def test_graph_replan_across_prefill_boundary_declines():
    """A CUDA graph bakes its plan's kernel LAUNCH SET, not only buffer
    pointers: the launcher skips the split-KV decode kernels on a pure
    single-request prefill step (batch 1, q_len > 16). A graph-mode replan
    that crosses that boundary is declined (the capture could not cover the
    new rows); a same-class replan still arms."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    big = _make_batch([32], [64], device)
    wrapper = _plan_wrapper(big, device, use_cuda_graph=True)
    out = wrapper.run(big["q"], (big["k_cache"], big["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out, _reference(big)) >= MATCHED

    def _replan(b):
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

    # same class (single-request prefill, fits retained buffers): arms
    mid = _make_batch([24], [64], device)
    _replan(mid)
    assert wrapper._gqa_ctx_fp8q_sm90_state is not None
    out = wrapper.run(mid["q"], (big["k_cache"], big["v_cache"]))
    assert _matched_ratio(out, _reference(mid)) >= MATCHED

    # crossing to decode-class rows: declined (explicit backend raises; an
    # auto-armed plan would fall back to the stock path), and the wrapper is
    # left unarmed
    short = _make_batch([16], [64], device)
    with pytest.raises(NotImplementedError, match="launch schedule"):
        _replan(short)
    assert wrapper._gqa_ctx_fp8q_sm90_state is None
    stats = _gqa_ctx_fp8q_sm90_stats()
    assert any("launch schedule" in r for r in stats["fallback_reasons"]), stats


def test_fp8_query_flavor_mismatch_raises():
    """An e5m2 query on an e4m3-planned wrapper must not be served silently:
    the dequant fallback would cast it to bf16 before the stock cached-dtype
    check could see it."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device)
    q_e5m2 = batch["q"].view(torch.float8_e5m2)
    with pytest.raises(ValueError, match="q_data_type"):
        wrapper.run(q_e5m2, (batch["k_cache"], batch["v_cache"]))
    assert _gqa_ctx_fp8q_sm90_stats()["runs_dispatched"] == 0


def test_one_element_q_scale_broadcasts_in_dequant_fallback(monkeypatch):
    """The dequant fallback treats a 1-element q_scale tensor as the
    per-tensor convention and broadcasts it; any other size that is not the
    planned batch size raises instead of gathering out of bounds. The planned
    batch size counts zero-row requests the row map cannot recover."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    monkeypatch.setenv(DISABLE_ENV, "1")
    batch = _make_batch([5, 90, 7], [584, 727, 75], device)
    wrapper = _plan_wrapper(batch, device, backend="auto")
    one = torch.tensor([2.0], dtype=torch.float32, device=device)
    out = wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]), q_scale=one)
    assert _matched_ratio(out, _reference(batch, qk_descale=[2.0, 2.0, 2.0])) >= MATCHED
    bad = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="batch_size=3"):
        wrapper.run(batch["q"], (batch["k_cache"], batch["v_cache"]), q_scale=bad)
