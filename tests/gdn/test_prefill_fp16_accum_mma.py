"""
Copyright (c) 2025 by FlashInfer team.

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

# Tests for the opt-in FP16-accumulate MMA mode of the SM12x GDN prefill kernels.
#
# The mode (``FLASHINFER_GDN_FP16_ACCUM_MMA=1`` or the private
# ``_fp16_accum_mma`` argument of ``chunk_gated_delta_rule``) replaces the
# FP32-accumulate tensor-core MMAs of the SM120 delta-rule kernels by
# FP16-accumulate MMAs whose partial sums are carried into FP32 registers.
#
# * ``test_switch_*`` and ``test_default_off_*`` need no GPU: they check that
#   the mode only reaches the SM120 entry points when it is requested.
# * The numerical tests run on SM12x only and compare against a token-by-token
#   FP64 recurrence of the delta rule. Every distinct (dtype, path, gates,
#   initial state) combination recompiles the CuTe-DSL kernels, so the cases
#   below are a covering set rather than the full cross product.

from __future__ import annotations

import inspect
import itertools
import random

import pytest
import torch

import flashinfer.gdn_prefill as gdn_prefill
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import is_sm12x_supported

HEAD_SIZE = 128
_FROM_INPUTS = object()  # "use the initial state stored in the inputs"
# (num_q_heads, num_k_heads, num_v_heads): equal heads, GQA and GVA (ratio 3).
HEADS = {"mha": (2, 2, 2), "gqa": (4, 1, 1), "gva": (2, 2, 6)}
SEQ_LENS = {
    "64": [64],
    "multi": [64, 128, 512],
    "8192": [8192],
    "ragged": [61, 130, 197],  # none a multiple of the 64-token chunk
}

# BF16 tier of tests/gdn/test_prefill_delta_rule.py, used for both dtypes.
ATOL_O, RTOL_O = 1e-2, 1e-2
ATOL_KV, RTOL_KV = 5e-3, 1e-3

# (dtype, use_cp, alpha+beta given, initial state given, heads, lens, cp_chunk_len).
# Every pair of values of the six axes occurs together at least once and all
# eight (dtype, path, gates) triples are present, in 8 distinct kernel
# specializations (the CP kernels always receive alpha and beta, so for the CP
# path "gates" only changes the data).
_NUMERIC_CASES = [
    ("bfloat16", True, True, True, "gva", "8192", None),
    ("bfloat16", True, True, True, "mha", "64", None),
    ("bfloat16", True, False, False, "mha", "multi", 128),
    ("float16", True, True, False, "gqa", "ragged", None),
    ("float16", True, False, False, "gqa", "8192", None),
    ("float16", True, False, False, "gqa", "multi", 128),
    ("float16", True, False, False, "gva", "64", None),
    ("bfloat16", False, True, True, "gqa", "64", None),
    ("bfloat16", False, True, True, "gva", "multi", None),
    ("bfloat16", False, False, True, "mha", "ragged", None),
    ("float16", False, True, False, "mha", "8192", None),
    ("float16", False, False, True, "gva", "ragged", None),
]


def _case_id(case) -> str:
    dtype, use_cp, gates, init_state, heads, lens, cp_chunk_len = case
    return "-".join(
        (
            "bf16" if dtype == "bfloat16" else "fp16",
            "cp" if use_cp else "noncp",
            "gates" if gates else "nogates",
            "init" if init_state else "noinit",
            heads,
            lens if cp_chunk_len is None else f"{lens}-chunk{cp_chunk_len}",
        )
    )


def _skip_unless_sm12x():
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("the FP16-accumulate MMA mode is SM12x only")


def _make_inputs(qkv_factory, dtype, heads, seq_lens, gates, init_state, seed=0):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_q_heads, num_k_heads, num_v_heads = HEADS[heads]
    num_heads = max(num_q_heads, num_v_heads)
    total = sum(seq_lens)
    device = torch.device("cuda")
    with device:
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, HEAD_SIZE, dtype
        )
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        cu_seqlens = torch.tensor([0, *itertools.accumulate(seq_lens)])
        # Slowly decaying gates keep the recurrence long-memory, so rounding
        # errors of the carried state show up over many chunks.
        alpha = 0.9 + 0.1 * torch.rand(total, num_heads) if gates else None
        beta = torch.rand(total, num_heads) if gates else None
        state0 = (
            0.1 * torch.randn(len(seq_lens), num_heads, HEAD_SIZE, HEAD_SIZE)
            if init_state
            else None
        )
    return {
        "q": q,
        "k": k,
        "v": v,
        "alpha": alpha,
        "beta": beta,
        "state0": state0,  # [N, H, V, K], the kernel layout
        "cu_seqlens": cu_seqlens,
        "seq_lens": list(seq_lens),
        "dtype": dtype,
    }


def _run_kernel(
    inp,
    scale,
    use_cp,
    cp_chunk_len,
    fp16_accum_mma,
    *,
    init_state=_FROM_INPUTS,
    output_state=None,
    **extra,
):
    """One chunk_gated_delta_rule call; returns (output, final state)."""
    q, v = inp["q"], inp["v"]
    num_heads = max(q.shape[1], v.shape[1])
    out = torch.full(
        (q.shape[0], num_heads, HEAD_SIZE), float("nan"), dtype=q.dtype, device=q.device
    )
    if output_state is None:
        output_state = torch.full(
            (len(inp["seq_lens"]), num_heads, HEAD_SIZE, HEAD_SIZE),
            float("nan"),
            dtype=torch.float32,
            device=q.device,
        )
    chunk_gated_delta_rule(
        q,
        inp["k"],
        v,
        inp["alpha"],
        inp["beta"],
        scale,
        inp["state0"] if init_state is _FROM_INPUTS else init_state,
        True,
        inp["cu_seqlens"],
        use_qk_l2norm_in_kernel=False,
        output=out,
        output_state=output_state,
        use_cp=use_cp,
        max_seqlen=max(inp["seq_lens"]),
        _cp_chunk_len=cp_chunk_len if use_cp else None,
        _fp16_accum_mma=fp16_accum_mma,
        **extra,
    )
    torch.cuda.synchronize()
    return out, output_state


def _fp64_reference(inp, scale, state0=_FROM_INPUTS, checkpoint_every=0):
    """Token-by-token FP64 delta rule on the dtype-rounded kernel inputs (CPU).

    S <- alpha S;  u = beta (v - S k);  S <- S + u k^T;  o = scale S q
    with S in the kernel layout [H, V, K]. q/k/v heads are repeat-interleaved to
    the number of value/output heads (GQA and GVA). Returns (output [T, H, D],
    final states [N, H, V, K], checkpoints [n, H, V, K]).
    """
    if state0 is _FROM_INPUTS:
        state0 = inp["state0"]
    num_heads = max(inp["q"].shape[1], inp["v"].shape[1])
    total = inp["q"].shape[0]

    def heads_first(x):  # [T, h, D] -> [H, T, D] in FP64
        x = x.double().cpu().repeat_interleave(num_heads // x.shape[1], dim=1)
        return x.transpose(0, 1).contiguous()

    q, k, v = (heads_first(inp[name]) for name in ("q", "k", "v"))
    ones = torch.ones(num_heads, total, dtype=torch.float64)
    alpha = ones if inp["alpha"] is None else inp["alpha"].double().cpu().t()
    beta = ones if inp["beta"] is None else inp["beta"].double().cpu().t()
    out = torch.empty(total, num_heads, HEAD_SIZE, dtype=torch.float64)
    states, checkpoints = [], []
    start = 0
    for i, seq_len in enumerate(inp["seq_lens"]):
        if state0 is None:
            s = torch.zeros(num_heads, HEAD_SIZE, HEAD_SIZE, dtype=torch.float64)
        else:
            s = state0[i].double().cpu().clone()
        for t in range(start, start + seq_len):
            s *= alpha[:, t, None, None]
            u = beta[:, t, None] * (v[:, t] - (s @ k[:, t, :, None])[..., 0])
            s += u[:, :, None] * k[:, t, None, :]
            out[t] = scale * (s @ q[:, t, :, None])[..., 0]
            if checkpoint_every and (t - start + 1) % checkpoint_every == 0:
                checkpoints.append(s.clone())
        states.append(s)
        start += seq_len
    return out, torch.stack(states), checkpoints


def _rel_l2(x, ref):
    x, ref = x.double().cpu(), ref.double().cpu()
    return ((x - ref).norm() / ref.norm()).item()


def _check_against_reference(inp, out, state, ref_out, ref_state):
    dtype = inp["dtype"]
    torch.testing.assert_close(
        out.float().cpu(), ref_out.to(dtype).float(), atol=ATOL_O, rtol=RTOL_O
    )
    torch.testing.assert_close(
        state.cpu(), ref_state.float(), atol=ATOL_KV, rtol=RTOL_KV
    )
    return {
        "out_rel_l2": _rel_l2(out, ref_out),
        "state_rel_l2": _rel_l2(state, ref_state),
    }


def _check_numeric_case(qkv_factory, case, fp16_accum_mma=True):
    """Kernel vs FP64 reference for one row of _NUMERIC_CASES."""
    _skip_unless_sm12x()
    dtype_name, use_cp, gates, init_state, heads, lens, cp_chunk_len = case
    inp = _make_inputs(
        qkv_factory,
        getattr(torch, dtype_name),
        heads,
        SEQ_LENS[lens],
        gates,
        init_state,
    )
    scale = HEAD_SIZE**-0.5
    out, state = _run_kernel(inp, scale, use_cp, cp_chunk_len, fp16_accum_mma)
    ref_out, ref_state, _ = _fp64_reference(inp, scale)
    return _check_against_reference(inp, out, state, ref_out, ref_state)


def _check_checkpoint_case(qkv_factory, fp16_accum_mma=True):
    """CP path, bf16: state checkpoints every 128 tokens vs the FP64 states."""
    _skip_unless_sm12x()
    seq_lens, every = [128, 256, 512], 128
    inp = _make_inputs(
        qkv_factory, torch.bfloat16, "gva", seq_lens, gates=True, init_state=False
    )
    counts = [n // every for n in seq_lens]
    num_heads = max(inp["q"].shape[1], inp["v"].shape[1])
    checkpoints = torch.full(
        (sum(counts), num_heads, HEAD_SIZE, HEAD_SIZE),
        float("nan"),
        dtype=torch.float32,
        device=inp["q"].device,
    )
    cu_starts = torch.tensor(
        [0, *itertools.accumulate(counts)], dtype=torch.int64, device=inp["q"].device
    )
    scale = HEAD_SIZE**-0.5
    out, state = _run_kernel(
        inp,
        scale,
        True,
        every,
        fp16_accum_mma,
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=cu_starts,
        checkpoint_every_n_tokens=every,
    )
    ref_out, ref_state, ref_ckpts = _fp64_reference(inp, scale, checkpoint_every=every)
    metrics = _check_against_reference(inp, out, state, ref_out, ref_state)
    assert len(ref_ckpts) == checkpoints.shape[0]
    torch.testing.assert_close(
        checkpoints.cpu(),
        torch.stack(ref_ckpts).float(),
        atol=ATOL_KV,
        rtol=RTOL_KV,
    )
    return metrics


def _check_state_indices_case(qkv_factory, fp16_accum_mma=True):
    """Non-CP path, fp16: initial states gathered from / final states scattered
    into a pool through state_indices; unrelated pool rows stay untouched."""
    _skip_unless_sm12x()
    seq_lens = [100, 256]
    inp = _make_inputs(
        qkv_factory, torch.float16, "gva", seq_lens, gates=True, init_state=False
    )
    num_heads = max(inp["q"].shape[1], inp["v"].shape[1])
    pool_size = 5
    slots = torch.tensor([3, 0], dtype=torch.int32, device=inp["q"].device)
    pool0 = 0.1 * torch.randn(
        pool_size,
        num_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        dtype=torch.float32,
        device=inp["q"].device,
    )
    pool = pool0.clone()
    scale = HEAD_SIZE**-0.5
    out, _ = _run_kernel(
        inp,
        scale,
        False,
        None,
        fp16_accum_mma,
        init_state=pool,
        output_state=pool,
        state_indices=slots,
    )
    ref_out, ref_state, _ = _fp64_reference(inp, scale, state0=pool0[slots.long()])
    metrics = _check_against_reference(inp, out, pool[slots.long()], ref_out, ref_state)
    untouched = [i for i in range(pool_size) if i not in slots.tolist()]
    assert torch.equal(pool[untouched], pool0[untouched])
    return metrics


def _check_bf16_error_not_worse(qkv_factory, use_cp, fp16_accum_mma=True):
    """bf16 inputs: the relative L2 error against FP64 of the final state and of
    the output is not larger with the mode than with FP32 accumulation."""
    _skip_unless_sm12x()
    inp = _make_inputs(
        qkv_factory, torch.bfloat16, "gva", [4096], gates=True, init_state=True
    )
    scale = HEAD_SIZE**-0.5
    ref_out, ref_state, _ = _fp64_reference(inp, scale)
    errors = {}
    for mode in (False, fp16_accum_mma):
        out, state = _run_kernel(inp, scale, use_cp, None, mode)
        errors[mode] = (_rel_l2(state, ref_state), _rel_l2(out, ref_out))
    state_err, out_err = errors[fp16_accum_mma]
    base_state_err, base_out_err = errors[False]
    assert state_err <= base_state_err, (state_err, base_state_err)
    assert out_err <= base_out_err, (out_err, base_out_err)
    return {
        "state_rel_l2_on": state_err,
        "state_rel_l2_off": base_state_err,
        "out_rel_l2_on": out_err,
        "out_rel_l2_off": base_out_err,
    }


@pytest.mark.parametrize(
    "case", _NUMERIC_CASES, ids=[_case_id(case) for case in _NUMERIC_CASES]
)
def test_fp16_accum_mma_matches_fp64(qkv_factory, case):
    _check_numeric_case(qkv_factory, case)


def test_fp16_accum_mma_state_checkpoints(qkv_factory):
    _check_checkpoint_case(qkv_factory)


def test_fp16_accum_mma_state_indices(qkv_factory):
    _check_state_indices_case(qkv_factory)


@pytest.mark.parametrize("use_cp", [False, True], ids=["noncp", "cp"])
def test_fp16_accum_mma_bf16_error_not_worse(qkv_factory, use_cp):
    _check_bf16_error_not_worse(qkv_factory, use_cp)


# ---------------------------------------------------------------------------
# Mode switch (no GPU needed: the kernel entry points are replaced by recorders)
# ---------------------------------------------------------------------------

_ENTRY_POINTS = (
    "chunk_gated_delta_rule_sm90",
    "chunk_gated_delta_rule_sm100",
    "chunk_gated_delta_rule_sm120",
    "cp_delta_rule_dsl_sm90",
    "cp_delta_rule_dsl_sm100",
    "cp_delta_rule_dsl_sm120",
)
_ENV = "FLASHINFER_GDN_FP16_ACCUM_MMA"


def _recorded_calls(monkeypatch, capability, use_cp, backend, override):
    calls = []

    def recorder(name):
        def entry_point(*args, **kwargs):
            calls.append((name, kwargs))

        return entry_point

    for name in _ENTRY_POINTS:
        monkeypatch.setattr(gdn_prefill, name, recorder(name))
    monkeypatch.setattr(gdn_prefill, "get_compute_capability", lambda _: capability)
    monkeypatch.setattr(gdn_prefill, "get_device_sm_count", lambda _: 170)
    monkeypatch.setattr(gdn_prefill, "get_device_name", lambda _: "recorded device")
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    total = 128
    q = torch.zeros(total, 2, HEAD_SIZE, dtype=torch.bfloat16)
    chunk_gated_delta_rule(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        cu_seqlens=torch.tensor([0, total]),
        use_cp=use_cp,
        backend=backend,
        _fp16_accum_mma=override,
    )
    assert len(calls) == 1
    return calls[0]


# (arch major, backend, environment variable, override, mode reaches the kernels)
_SWITCH_CASES = [
    (12, "auto", None, None, False),
    (12, "auto", "1", None, True),
    (12, "flashinfer", "1", None, True),
    (12, "auto", "0", None, False),
    (12, "auto", "", None, False),
    (12, "auto", "true", None, False),
    (12, "auto", None, True, True),
    (12, "flashinfer", None, True, True),
    (12, "auto", "1", False, False),
    (12, "auto", "0", True, True),
    (12, "auto", None, False, False),
    (9, "auto", "1", None, False),
    (9, "auto", None, True, False),
    (10, "auto", "1", None, False),
    (10, "flashinfer", None, True, False),
]


@pytest.mark.parametrize("use_cp", [False, True], ids=["noncp", "cp"])
@pytest.mark.parametrize(
    "arch, backend, env, override, expected",
    _SWITCH_CASES,
    ids=[f"sm{c[0]}-{c[1]}-env={c[2]}-override={c[3]}" for c in _SWITCH_CASES],
)
def test_switch_truth_table(
    monkeypatch, arch, backend, env, override, expected, use_cp
):
    if env is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, env)
    name, kwargs = _recorded_calls(monkeypatch, (arch, 0), use_cp, backend, override)
    kind = "cp_delta_rule_dsl" if use_cp else "chunk_gated_delta_rule"
    assert name == f"{kind}_sm{arch}0"
    if expected:
        assert kwargs["fp16_accum_mma"] is True
    else:
        # Every default call keeps exactly its previous arguments.
        assert "fp16_accum_mma" not in kwargs


@pytest.mark.parametrize(
    "env, override, expected",
    [
        (None, None, False),
        ("1", None, True),
        ("0", None, False),
        ("true", None, False),
        (" 1", None, False),
        ("1", False, False),
        ("0", True, True),
        (None, True, True),
        (None, False, False),
    ],
)
def test_switch_resolution(monkeypatch, env, override, expected):
    if env is None:
        monkeypatch.delenv(_ENV, raising=False)
    else:
        monkeypatch.setenv(_ENV, env)
    assert gdn_prefill._fp16_accum_mma_requested(override) is expected


def test_default_off_in_every_layer():
    """Every kernel class and host wrapper that takes the mode defaults to off."""
    pytest.importorskip("cutlass")
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_cp_sm120 as cp
    from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm120 as non_cp

    targets = [
        non_cp._FullyFusedDeltaRuleSm120.__init__,
        non_cp._get_prefill_kernel,
        non_cp.delta_rule_prefill_dsl,
        cp.CPDeltaRuleTPrecomputeSm120.__init__,
        cp._get_t_precompute_kernel,
        cp.cp_delta_rule_t_precompute_dsl_sm120,
        cp.CPDeltaRuleMNPrecomputeSm120.__init__,
        cp._get_mn_precompute_kernel,
        cp.cp_delta_rule_mn_precompute_dsl_sm120,
        cp.CPDeltaRulePrefillSm120.__init__,
        cp._get_prefill_kernel,
        cp.cp_delta_rule_prefill_dsl_sm120,
        cp.cp_delta_rule_dsl_sm120,
    ]
    for target in targets:
        parameter = inspect.signature(target).parameters["fp16_accum_mma"]
        assert parameter.default is False, target.__qualname__
