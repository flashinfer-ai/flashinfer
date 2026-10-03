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

from types import SimpleNamespace

import pytest
import torch

import flashinfer
import flashinfer.prefill as prefill_mod
from flashinfer.prefill import (
    _eagle_verify_fp8kv_sm90_plan_eligibility,
    _eagle_verify_fp8kv_sm90_prepare,
    _eagle_verify_fp8kv_sm90_run_eligibility,
    _eagle_verify_fp8kv_sm90_stats,
    _reset_eagle_verify_fp8kv_sm90_stats,
)
from flashinfer.utils import MaskMode

DISABLE_ENV = "FLASHINFER_DISABLE_EAGLE_VERIFY_FP8KV_SM90"
PDL_ENV = "FLASHINFER_EAGLE_VERIFY_FP8KV_SM90_PDL"
QO_LEN, H_QO, H_KV, HEAD_DIM, PAGE = 4, 4, 1, 256, 1
MASK_TAIL_SLACK = 8  # bytes the kernel may read past the packed mask


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(PDL_ENV, raising=False)
    _reset_eagle_verify_fp8kv_sm90_stats()
    yield
    _reset_eagle_verify_fp8kv_sm90_stats()


# --------------------------------------------------------------------------
# Plan-time guard (GPU-free up to the device check, which comes last)
# --------------------------------------------------------------------------


def _plan_kwargs(**overrides):
    kwargs = dict(
        backend="fa2",
        forwarded_uniform_q_len=QO_LEN,
        has_custom_mask=True,
        num_qo_heads=H_QO,
        num_kv_heads=H_KV,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        batch_size=4,
        device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return kwargs


def test_plan_guard_rejects_non_verify_plans_silently():
    # ordinary prefill plans: the hook must stay silent (stock path)
    for overrides in (
        dict(forwarded_uniform_q_len=0),
        dict(forwarded_uniform_q_len=5),
        dict(has_custom_mask=False),
        dict(backend="fa3"),
    ):
        reason = _eagle_verify_fp8kv_sm90_plan_eligibility(**_plan_kwargs(**overrides))
        assert reason is not None and reason.startswith("not_verify_route:"), overrides


@pytest.mark.parametrize(
    "overrides",
    [
        dict(num_qo_heads=8),
        dict(num_kv_heads=2),
        dict(head_dim_qk=128),
        dict(head_dim_vo=128),
        dict(page_size=16),
        dict(q_data_type=torch.float16),
        dict(kv_data_type=torch.bfloat16),
        dict(o_data_type=torch.float16),
        dict(kv_layout="HND"),
        dict(pos_encoding_mode="ROPE_LLAMA"),
        dict(use_fp16_qk_reduction=True),
        dict(batch_size=0),
    ],
)
def test_plan_guard_rejects_other_geometries_loudly(overrides):
    reason = _eagle_verify_fp8kv_sm90_plan_eligibility(**_plan_kwargs(**overrides))
    assert reason is not None and not reason.startswith("not_verify_route:"), overrides


def test_plan_guard_device_check_is_last():
    # every semantic/shape check passes; only the (CPU) device rejects
    reason = _eagle_verify_fp8kv_sm90_plan_eligibility(**_plan_kwargs())
    assert reason == "device=cpu"


# --------------------------------------------------------------------------
# Run-time guard (GPU-free: CPU tensors with the production strides)
# --------------------------------------------------------------------------


def _fake_wrapper(
    batch_size=2, kv_len=64, device="cpu", mask_slack=MASK_TAIL_SLACK, **overrides
):
    rows = batch_size * QO_LEN
    total_kv = batch_size * kv_len
    mask_bytes_per_req = (QO_LEN * kv_len + 7) // 8
    mask_bytes = batch_size * mask_bytes_per_req
    workspace_floats = batch_size * 198 * 16 * (HEAD_DIM + 2)
    wrapper = SimpleNamespace(
        _logits_soft_cap=0.0,
        _pos_encoding_mode="NONE",
        _use_fp16_qk_reduction=False,
        _sm_scale=0.0625,
        _prefix_len_ptr=None,
        _qo_indptr_buf=torch.arange(
            0, (batch_size + 1) * QO_LEN, QO_LEN, dtype=torch.int32, device=device
        ),
        _paged_kv_indptr_buf=torch.arange(
            0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32, device=device
        ),
        _paged_kv_indices_buf=torch.arange(total_kv, dtype=torch.int32, device=device),
        # packed bits followed by the tail slack the kernel's 64-bit mask
        # window may read into
        _custom_mask_buf=torch.full(
            (mask_bytes + mask_slack,), 255, dtype=torch.uint8, device=device
        ),
        _mask_indptr_buf=torch.arange(
            0,
            (batch_size + 1) * mask_bytes_per_req,
            mask_bytes_per_req,
            dtype=torch.int32,
            device=device,
        ),
        is_cuda_graph_enabled=True,
        _eagle_verify_fp8kv_sm90_state=SimpleNamespace(
            batch_size=batch_size,
            num_splits=198,
            workspace=torch.empty(workspace_floats, dtype=torch.float32, device=device),
            workspace_floats=workspace_floats,
            mask_bytes_required=mask_bytes + MASK_TAIL_SLACK,
            device_index=0,
            pdl=False,
        ),
    )
    for key, value in overrides.items():
        setattr(wrapper, key, value)
    tensors = dict(
        q=torch.zeros(rows, H_QO, HEAD_DIM, dtype=torch.bfloat16, device=device),
        k_cache=torch.zeros(
            total_kv + 1, PAGE, H_KV, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
        ),
        v_cache=torch.zeros(
            total_kv + 1, PAGE, H_KV, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
        ),
        out=torch.zeros(rows, H_QO, HEAD_DIM, dtype=torch.bfloat16, device=device),
    )
    return wrapper, tensors


def _run_reason(wrapper, tensors, **overrides):
    args = dict(
        mask_mode=MaskMode.CUSTOM.value,
        window_left=-1,
        q_scale=None,
        k_scale=1.0,
        v_scale=1.0,
        return_lse=False,
        sinks=None,
        kv_cache_sf=None,
    )
    args.update(overrides)
    return _eagle_verify_fp8kv_sm90_run_eligibility(
        wrapper,
        tensors["q"],
        tensors["k_cache"],
        tensors["v_cache"],
        tensors["out"],
        **args,
    )


def test_run_guard_accepts_the_production_call_up_to_the_device():
    wrapper, tensors = _fake_wrapper()
    # CPU tensors: everything but the device passes
    assert _run_reason(wrapper, tensors) == "device cpu"
    assert _run_reason(wrapper, tensors, k_scale=None, v_scale=None) == "device cpu"
    assert _run_reason(wrapper, tensors, k_scale=1, v_scale=1) == "device cpu"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(return_lse=True), "return_lse"),
        (dict(sinks=torch.zeros(4)), "sinks"),
        (dict(kv_cache_sf=torch.zeros(1)), "kv_cache_sf"),
        (dict(q_scale=1.0), "q_scale"),
        (dict(k_scale=0.5), "k_scale"),
        (dict(k_scale=torch.ones(1)), "k_scale"),  # tensors need a sync to read
        (dict(v_scale=2.0), "v_scale"),
        (dict(mask_mode=MaskMode.CAUSAL.value), "mask_mode=1"),
        (dict(window_left=1024), "window_left=1024"),
    ],
)
def test_run_guard_rejects_unsupported_call_arguments(overrides, expected):
    wrapper, tensors = _fake_wrapper()
    assert _run_reason(wrapper, tensors, **overrides) == expected


def test_run_guard_reads_the_wrapper_state_set_by_plan():
    wrapper, tensors = _fake_wrapper(_sm_scale=0.125)
    assert _run_reason(wrapper, tensors) == "sm_scale=0.125"
    wrapper, tensors = _fake_wrapper(_logits_soft_cap=30.0)
    assert _run_reason(wrapper, tensors) == "logits_soft_cap=30.0"
    wrapper, tensors = _fake_wrapper(_pos_encoding_mode="ROPE_LLAMA")
    assert _run_reason(wrapper, tensors) == "pos_encoding_mode=ROPE_LLAMA"
    wrapper, tensors = _fake_wrapper(_sm_scale=None)  # run() defaults to 1/sqrt(256)
    assert _run_reason(wrapper, tensors) == "device cpu"


def test_run_guard_asserts_the_strides_the_kernel_derives():
    wrapper, tensors = _fake_wrapper()
    # a KV pool with a padded head dim: 256 B rows are no longer 256 B apart
    padded = torch.zeros(
        tensors["k_cache"].shape[0],
        PAGE,
        H_KV,
        HEAD_DIM + 64,
        dtype=torch.float8_e4m3fn,
    )
    tensors["k_cache"] = padded[..., :HEAD_DIM]
    assert _run_reason(wrapper, tensors).startswith("k_cache strides")
    wrapper, tensors = _fake_wrapper()
    tensors["q"] = tensors["q"].transpose(0, 1).contiguous().transpose(0, 1)
    assert _run_reason(wrapper, tensors) == "q not contiguous"
    wrapper, tensors = _fake_wrapper()
    tensors["q"] = tensors["q"][: QO_LEN * 1]  # prefix slice: fewer rows than planned
    assert _run_reason(wrapper, tensors).startswith("q shape/dtype")
    wrapper, tensors = _fake_wrapper(_custom_mask_buf=None)
    assert _run_reason(wrapper, tensors) == "custom_mask buffer"
    # exact-size buffer: the tail window would over-read
    wrapper, tensors = _fake_wrapper(mask_slack=0)
    assert _run_reason(wrapper, tensors).startswith("custom_mask buffer 64 < 72")
    wrapper, tensors = _fake_wrapper()
    wrapper._eagle_verify_fp8kv_sm90_state.workspace = torch.empty(
        16, dtype=torch.float32
    )
    assert _run_reason(wrapper, tensors) == "workspace"


def test_stats_report_build_state_and_counters():
    stats = _eagle_verify_fp8kv_sm90_stats()
    for key in (
        "runs_dispatched",
        "plans_prepared",
        "plans_ineligible",
        "mask_buf_padded",
        "module_loaded",
        "module_error",
        "compiled_variants",
        "distinct_kernels",
        "graph_nodes_per_launch",
        "specialized_dispatches",
    ):
        assert key in stats
    assert stats["distinct_kernels"] == 2
    assert stats["graph_nodes_per_launch"] == 2


# --------------------------------------------------------------------------
# Plan-time host contract of the device code (GPU-free: the checks run before
# the module is touched; the eligibility guard is stubbed because a non-SM90
# host would stop at the device check)
# --------------------------------------------------------------------------


def _prepare_kwargs(wrapper, batch_size, qo_indptr_host, packed_mask_bytes):
    return dict(
        wrapper=wrapper,
        forwarded_uniform_q_len=QO_LEN,
        has_custom_mask=True,
        num_qo_heads=H_QO,
        num_kv_heads=H_KV,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        pos_encoding_mode="NONE",
        use_fp16_qk_reduction=False,
        batch_size=batch_size,
        qo_indptr_host=qo_indptr_host,
        packed_mask_bytes=packed_mask_bytes,
    )


def _contract_wrapper(batch_size, mask_bytes, cuda_graph):
    wrapper, _ = _fake_wrapper(batch_size=batch_size)
    wrapper._backend = "fa2"
    wrapper._kv_layout = "NHD"
    wrapper.device = torch.device("cpu")
    wrapper.is_cuda_graph_enabled = cuda_graph
    wrapper._custom_mask_buf = torch.full((mask_bytes,), 255, dtype=torch.uint8)
    return wrapper


@pytest.fixture
def eligible_without_module(monkeypatch):
    monkeypatch.setattr(
        prefill_mod, "_eagle_verify_fp8kv_sm90_plan_eligibility", lambda *a, **k: None
    )
    monkeypatch.setattr(
        prefill_mod, "_get_eagle_verify_fp8kv_sm90_module", lambda: None
    )


def test_prepare_rejects_a_qo_indptr_that_is_not_stride_4(eligible_without_module):
    wrapper = _contract_wrapper(2, 64, cuda_graph=True)
    bad = torch.tensor([0, 4, 9], dtype=torch.int32)
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, bad, 8))
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["plans_ineligible"] == 1
    assert wrapper._eagle_verify_fp8kv_sm90_state is None
    good = torch.tensor([0, 4, 8], dtype=torch.int32)
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 8))
    # passed the contract checks and reached the (stubbed, failing) module load
    assert _eagle_verify_fp8kv_sm90_stats()["plans_ineligible"] == 1


def test_prepare_requires_mask_tail_slack_in_cuda_graph_mode(eligible_without_module):
    good = torch.tensor([0, 4, 8], dtype=torch.int32)
    # user-provided graph buffer with exactly the packed bytes: fail closed,
    # buffer untouched
    wrapper = _contract_wrapper(2, 64, cuda_graph=True)
    before = wrapper._custom_mask_buf
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["plans_ineligible"] == 1 and stats["mask_buf_padded"] == 0
    assert wrapper._custom_mask_buf is before
    # the same buffer with >= 8 spare bytes is accepted
    wrapper = _contract_wrapper(2, 64 + MASK_TAIL_SLACK, cuda_graph=True)
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    assert _eagle_verify_fp8kv_sm90_stats()["plans_ineligible"] == 1


def test_prepare_pads_the_eager_mode_mask_buffer(eligible_without_module):
    good = torch.tensor([0, 4, 8], dtype=torch.int32)
    wrapper = _contract_wrapper(2, 64, cuda_graph=False)
    wrapper._custom_mask_buf = torch.arange(64, dtype=torch.uint8)
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["plans_ineligible"] == 0 and stats["mask_buf_padded"] == 1
    padded = wrapper._custom_mask_buf
    assert padded.numel() == 64 + MASK_TAIL_SLACK
    assert torch.equal(padded[:64], torch.arange(64, dtype=torch.uint8))
    assert int(padded[64:].sum()) == 0


def test_disable_switch_is_read_at_plan_time(monkeypatch):
    monkeypatch.setattr(
        prefill_mod, "_eagle_verify_fp8kv_sm90_plan_eligibility", lambda *a, **k: None
    )
    loads = []
    monkeypatch.setattr(
        prefill_mod,
        "_get_eagle_verify_fp8kv_sm90_module",
        lambda: loads.append(1) or None,
    )
    good = torch.tensor([0, 4, 8], dtype=torch.int32)
    wrapper = _contract_wrapper(2, 64 + MASK_TAIL_SLACK, cuda_graph=True)
    monkeypatch.setenv(DISABLE_ENV, "1")
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["plans_disabled"] == 1 and loads == []
    assert wrapper._eagle_verify_fp8kv_sm90_state is None
    monkeypatch.setenv(DISABLE_ENV, "0")
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    assert _eagle_verify_fp8kv_sm90_stats()["plans_disabled"] == 1 and loads == [1]


def test_pdl_switch_is_read_at_plan_time_and_defaults_off(monkeypatch):
    monkeypatch.setattr(
        prefill_mod, "_eagle_verify_fp8kv_sm90_plan_eligibility", lambda *a, **k: None
    )
    calls = []

    class _Module:
        def init(self, d):
            calls.append(("init", d))

        def workspace_floats(self, b, s):
            return b * s * 16 * 258

    monkeypatch.setattr(
        prefill_mod, "_get_eagle_verify_fp8kv_sm90_module", lambda: _Module()
    )
    monkeypatch.setattr(prefill_mod, "get_device_sm_count", lambda d: 132)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    good = torch.tensor([0, 4, 8], dtype=torch.int32)
    wrapper = _contract_wrapper(2, 64 + MASK_TAIL_SLACK, cuda_graph=True)
    wrapper._eagle_verify_fp8kv_sm90_workspace = None
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    state = wrapper._eagle_verify_fp8kv_sm90_state
    assert state is not None and state.pdl is False  # default: plain merge launch
    assert state.num_splits == 198  # ceil(3 * 132 / 2)
    assert _eagle_verify_fp8kv_sm90_stats()["workspace_allocs"] == 1
    monkeypatch.setenv(PDL_ENV, "1")
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    assert wrapper._eagle_verify_fp8kv_sm90_state.pdl is True
    assert _eagle_verify_fp8kv_sm90_stats()["workspace_allocs"] == 1  # reused
    monkeypatch.setenv(PDL_ENV, "0")
    _eagle_verify_fp8kv_sm90_prepare(**_prepare_kwargs(wrapper, 2, good, 64))
    assert wrapper._eagle_verify_fp8kv_sm90_state.pdl is False


# --------------------------------------------------------------------------
# On-device (SM90 only): specialized vs stock FA2 on verify-shaped problems,
# disable switch, CUDA-graph capture with and without a prepared module, and
# the split counts that exercise the merge's sub-warp loop bounds.
# --------------------------------------------------------------------------


def _sm90():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (9, 0)


def _verify_problem(batch_size, kv_len, mask_kind, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = batch_size * QO_LEN
    q = (torch.randn(rows, H_QO, HEAD_DIM, generator=g) * 0.5).to(torch.bfloat16)
    num_slots = batch_size * kv_len + 64
    k = (torch.randn(num_slots, H_KV, HEAD_DIM, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn(num_slots, H_KV, HEAD_DIM, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    qo_indptr = torch.arange(0, (batch_size + 1) * QO_LEN, QO_LEN, dtype=torch.int32)
    kv_indptr = torch.arange(0, (batch_size + 1) * kv_len, kv_len, dtype=torch.int32)
    kv_indices = (
        torch.randperm(num_slots - 1, generator=g)[: batch_size * kv_len].to(
            torch.int32
        )
        + 1
    )
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32)
    blocks = []
    for _ in range(batch_size):
        m = torch.ones(QO_LEN, kv_len, dtype=torch.bool)
        if mask_kind == "chain":
            m[:, kv_len - QO_LEN :] = torch.tril(
                torch.ones(QO_LEN, QO_LEN, dtype=torch.bool)
            )
        elif mask_kind == "tree":
            m[:, kv_len - QO_LEN :] = torch.tril(
                torch.ones(QO_LEN, QO_LEN, dtype=torch.bool)
            )
            m[2, kv_len - QO_LEN + 1] = False  # branch 2 does not see draft 1
        elif mask_kind == "random":
            m = torch.rand(QO_LEN, kv_len, generator=g) < 0.5
            m[:, 0] = True
        blocks.append(m.flatten())
    custom_mask = torch.cat(blocks).to(device)
    return dict(
        q=q.to(device),
        k=k.to(device),
        v=v.to(device),
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=kv_last_page_len,
        custom_mask=custom_mask,
        batch_size=batch_size,
        kv_len=kv_len,
    )


def _wrapper(problem, device, use_cuda_graph=True):
    bs, kv_len = problem["batch_size"], problem["kv_len"]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    if not use_cuda_graph:
        return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
    return flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty(bs + 1, dtype=torch.int32, device=device),
        paged_kv_indptr_buf=torch.empty(bs + 1, dtype=torch.int32, device=device),
        paged_kv_indices_buf=torch.empty(bs * kv_len, dtype=torch.int32, device=device),
        paged_kv_last_page_len_buf=torch.empty(bs, dtype=torch.int32, device=device),
        # + the tail slack the kernel's two-word mask window may read into
        # (serving frameworks size this buffer at max_tokens x context_len
        # bytes, far above the packed bits)
        custom_mask_buf=torch.empty(
            (bs * QO_LEN * kv_len + 7) // 8 + MASK_TAIL_SLACK,
            dtype=torch.uint8,
            device=device,
        ),
        mask_indptr_buf=torch.empty(bs + 1, dtype=torch.int32, device=device),
    )


def _plan(wrapper, problem, uniform_q_len=QO_LEN):
    wrapper.plan(
        problem["qo_indptr"],
        problem["kv_indptr"],
        problem["kv_indices"],
        problem["kv_last_page_len"],
        H_QO,
        H_KV,
        HEAD_DIM,
        PAGE,
        custom_mask=problem["custom_mask"],
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
        uniform_q_len=uniform_q_len,
    )


def _run(wrapper, problem):
    return wrapper.run(
        problem["q"], (problem["k"], problem["v"]), k_scale=1.0, v_scale=1.0
    )


def _reference(problem):
    # fp32 end to end, per request and per draft row, same mask semantics
    q = problem["q"].float()
    k = problem["k"].float()[:, 0]
    v = problem["v"].float()[:, 0]
    bs, kv_len = problem["batch_size"], problem["kv_len"]
    mask = problem["custom_mask"].view(bs, QO_LEN, kv_len)
    out = torch.empty_like(q)
    for r in range(bs):
        idx = problem["kv_indices"][r * kv_len : (r + 1) * kv_len].long().to(q.device)
        kk, vv = k[idx], v[idx]
        for t in range(QO_LEN):
            s = (q[r * QO_LEN + t] @ kk.T) * 0.0625  # [H_QO, kv_len]
            s = s.masked_fill(~mask[r, t][None, :], float("-inf"))
            out[r * QO_LEN + t] = torch.softmax(s, dim=-1) @ vv
    return out


def _assert_close_to_reference(o, ref):
    rel_rms = ((o.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rel_rms <= 0.005, rel_rms
    assert torch.isfinite(o.float()).all()


@pytest.mark.skipif(not _sm90(), reason="needs an SM90 GPU")
@pytest.mark.parametrize(
    "batch_size,kv_len", [(1, 777), (2, 4096), (4, 1023), (5, 2048)]
)
@pytest.mark.parametrize("mask_kind", ["chain", "tree", "random"])
def test_specialized_matches_stock_and_reference_on_device(
    batch_size, kv_len, mask_kind, monkeypatch
):
    device = torch.device("cuda:0")
    problem = _verify_problem(
        batch_size, kv_len, mask_kind, device, seed=batch_size * 7 + kv_len
    )
    ref = _reference(problem)

    wrapper = _wrapper(problem, device)
    _plan(wrapper, problem)
    assert _eagle_verify_fp8kv_sm90_stats()["plans_prepared"] == 1, (
        _eagle_verify_fp8kv_sm90_stats()
    )
    o_spec = _run(wrapper, problem)
    torch.cuda.synchronize()
    assert _eagle_verify_fp8kv_sm90_stats()["runs_dispatched"] == 1
    _assert_close_to_reference(o_spec, ref)

    # the disable switch keeps the plan hint (16-row FA2 tile) and runs the
    # stock kernel
    monkeypatch.setenv(DISABLE_ENV, "1")
    stock = _wrapper(problem, device)
    _plan(stock, problem)
    o_stock = _run(stock, problem)
    torch.cuda.synchronize()
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["runs_dispatched"] == 1 and stats["plans_disabled"] == 1
    _assert_close_to_reference(o_stock, ref)
    cos = torch.nn.functional.cosine_similarity(
        o_spec.float().flatten(), o_stock.float().flatten(), dim=0
    )
    assert cos.item() >= 0.999


@pytest.mark.skipif(not _sm90(), reason="needs an SM90 GPU")
def test_eager_plan_dispatches_with_a_padded_mask_buffer():
    device = torch.device("cuda:0")
    problem = _verify_problem(2, 1500, "chain", device)
    wrapper = _wrapper(problem, device, use_cuda_graph=False)
    _plan(wrapper, problem)
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["plans_prepared"] == 1 and stats["mask_buf_padded"] == 1
    o = _run(wrapper, problem)
    torch.cuda.synchronize()
    assert _eagle_verify_fp8kv_sm90_stats()["runs_dispatched"] == 1
    _assert_close_to_reference(o, _reference(problem))


@pytest.mark.skipif(not _sm90(), reason="needs an SM90 GPU")
def test_capture_after_plan_records_the_specialized_kernel():
    device = torch.device("cuda:0")
    problem = _verify_problem(4, 3000, "tree", device)
    wrapper = _wrapper(problem, device)
    _plan(wrapper, problem)
    o_eager = _run(wrapper, problem)
    torch.cuda.synchronize()
    before = _eagle_verify_fp8kv_sm90_stats()["runs_dispatched"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(wrapper, problem)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o_graph = _run(wrapper, problem)
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["runs_dispatched"] == before + 2
    assert stats["runs_dispatched_capturing"] == 1
    assert stats["workspace_allocs"] == 1  # nothing allocated during capture
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(o_graph, o_eager, rtol=0, atol=0)  # deterministic
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(o_graph, o_eager, rtol=0, atol=0)


@pytest.mark.skipif(not _sm90(), reason="needs an SM90 GPU")
def test_capture_without_a_prepared_module_falls_back_cleanly(monkeypatch):
    device = torch.device("cuda:0")
    problem = _verify_problem(2, 2000, "chain", device)
    wrapper = _wrapper(problem, device)
    _plan(wrapper, problem)
    # simulate a process whose module is not available at capture time
    monkeypatch.setattr(prefill_mod, "_eagle_verify_fp8kv_sm90_module", None)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(wrapper, problem)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o_graph = _run(wrapper, problem)
    g.replay()
    torch.cuda.synchronize()
    stats = _eagle_verify_fp8kv_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_capture_unready"] >= 1
    _assert_close_to_reference(o_graph, _reference(problem))


@pytest.mark.skipif(not _sm90(), reason="needs an SM90 GPU")
@pytest.mark.parametrize(
    "batch_size,kv_len", [(64, 2048), (14, 4096), (132, 1024), (7, 777)]
)
def test_odd_split_counts_do_not_deadlock_the_merge(batch_size, kv_len):
    # num_splits = ceil(3 * SMs / B) is odd and < 32 for the first three batch
    # sizes on a 132-SM part (7, 29, 3; the fourth is a uniform control at 57):
    # the two 16-lane subgroups of a merge warp then run the split loop a
    # different number of times, which a full-warp shuffle inside it deadlocks
    device = torch.device("cuda:0")
    problem = _verify_problem(batch_size, kv_len, "chain", device, seed=batch_size)
    wrapper = _wrapper(problem, device)
    _plan(wrapper, problem)
    o = _run(wrapper, problem)
    torch.cuda.synchronize()
    assert _eagle_verify_fp8kv_sm90_stats()["runs_dispatched"] == 1
    _assert_close_to_reference(o, _reference(problem))
