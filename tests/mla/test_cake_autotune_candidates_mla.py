"""Cake in the second selection stage of uniform-Q MLA decode ``backend="auto"``.

``trtllm_batch_decode_with_kv_cache_mla`` selects in two stages: the TRTLLM-GEN /
CuTe DSL selection, unchanged, then, only for the units the device's offer table
lists, a comparison of that winner with Cake's generated Blackwell route. These
tests check that the first stage keeps its op, runners, config and cache entries;
that Cake joins only the second stage, only for listed units, requests it serves,
while the tuner may select it and never while an autotune process group tunes;
that the committed offer table is well formed and lists only measured units; that a tuning bucket
offers Cake only when every batch the bucket serves is listed and launches the
programs of the profiled batch; that Cake is profiled only against a first-stage
winner and a Cake choice stays valid only while that winner does; that both
stages time a single cold replay and the second keeps Cake unless the first
stage's winner measures faster by more than the absolute tie of each bucket;
that the memo state and the tuning buckets are what the tuner's
accessors and the unmemoized computation give; that an
untuned call never selects Cake and every Cake miss runs the first stage's
selection; that a repeated lookup-only call of a listed batch reuses the
decision of both stages exactly while the tuner and its settings are unchanged,
also when the caller changes max_seq_len on every call (one selection per
bucket and KV tile count, the same runner, tactic, Cake launch and output as
without the memo); that a tuned call is correct whichever candidate wins; that a forced
Cake selection matches TRTLLM-GEN (bmm2_scale, q_len > 1) on live rows while
padded rows output exact zeros; that it reads no host state and launches only
the kernels of explicit ``backend="cake"``; and that it stays correct after input
updates that bypass torch's version counter and under CUDA-graph replay with
updated inputs (also two graphs replayed on two streams at once).

Tests that need Cake offered replace the offer table with a synthetic one: the
committed table is derived from measurements on B200 and GB300 only.
"""

import contextlib
import itertools
import math
import os
import statistics
import threading
from dataclasses import replace

import pytest
import torch

import flashinfer
from flashinfer.api_logging import cake_auto_candidates_enabled
from flashinfer.autotuner import (
    AutoTuner,
    DynamicDim,
    OptimizationProfile,
    ProfilingCacheKey,
    TunableRunner,
    autotune,
    cake_auto_candidate_open,
)
from flashinfer.autotune_cache import ReplayCoverage, autotune_v2
from flashinfer.autotuner import autotuner as autotuner_module
from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets
from flashinfer.mla import _core as core
from flashinfer.mla import cake_trtllm_mla_blackwell as cake_module
from flashinfer.mla._utils import _round_to_seq_len_bucket
from flashinfer.utils import (
    get_compute_capability,
    get_device_sm_count,
    get_trtllm_gen_multi_ctas_kv_counter_bytes,
)
from tests.attention.test_trtllm_gen_mla import trtllm_mla_blackwell_reference

DISABLE_ENV = "FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES"
EXPERIMENTAL_ENV = "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS"
QK_DIM, LATENT, NUM_HEADS = 576, 512, 128
TOLERANCE = {torch.bfloat16: 1e-2, torch.float8_e4m3fn: 1e-1}
TUNING_BUCKETS = (1, 2, 4, 8, 64)
# The first-stage winner a stubbed Cake choice beat.
WINNER = ("_FakeCuteDsl", -1)

_workspace_buffer = None


def _compute_capability():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return get_compute_capability(torch.device("cuda"))


def _require_cake_arch():
    if _compute_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Cake Blackwell MLA decode requires SM100a or SM103a")


def _require_sm10x():
    if _compute_capability()[0] != 10:
        pytest.skip("uniform MLA decode autotuning runs on SM100/SM103/SM107")


def _workspace() -> torch.Tensor:
    global _workspace_buffer
    if _workspace_buffer is None:
        _workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
    return _workspace_buffer


def _make_case(
    *,
    dtype=torch.bfloat16,
    batch_size=4,
    q_len=1,
    page_size=32,
    num_heads=NUM_HEADS,
    max_seq_len=1024,
    seed=0,
):
    generator = torch.Generator().manual_seed(seed)
    pages = max_seq_len // page_size
    # Inputs of magnitude 0.5 give a peaked softmax, so reading wrong keys moves
    # the output well past the tolerance.
    query = torch.randn(batch_size, q_len, num_heads, QK_DIM, generator=generator) * 0.5
    kv_cache = torch.randn(batch_size * pages, page_size, QK_DIM, generator=generator)
    block_tables = torch.randperm(batch_size * pages, generator=generator)
    seq_lens = torch.randint(
        max(q_len, max_seq_len // 2),
        max_seq_len + 1,
        (batch_size,),
        generator=generator,
    )
    return dict(
        query=query.to("cuda", dtype),
        kv_cache=(kv_cache * 0.5).to("cuda", dtype),
        workspace_buffer=_workspace(),
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=QK_DIM - LATENT,
        block_tables=block_tables.reshape(batch_size, pages).to("cuda", torch.int32),
        seq_lens=seq_lens.to("cuda", torch.int32),
        max_seq_len=max_seq_len,
        bmm1_scale=1.0 / (192**0.5),
        bmm2_scale=1.0,
        backend="auto",
    )


def _decode(case):
    return flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(**case)


def _reference(case):
    # FP32 copies hold the FP8 values exactly and avoid FP8 gathers.
    sinks = case.get("sinks")
    return (
        trtllm_mla_blackwell_reference(
            case["query"].float(),
            case["kv_cache"].float(),
            case["block_tables"],
            case["seq_lens"],
            case["bmm1_scale"],
            case["kv_cache"].shape[-2],
            sinks[0] if sinks is not None else None,
        )
        * case["bmm2_scale"]
    )


def _assert_matches_reference(output, case, label=""):
    tolerance = TOLERANCE[case["query"].dtype]
    torch.testing.assert_close(
        output,
        _reference(case),
        rtol=tolerance,
        atol=tolerance,
        msg=lambda message: f"{label}: {message}",
    )


def _name(runner) -> str:
    if isinstance(runner, core._CakeTrtllmMlaBlackwellRunner):
        return "cake"
    if isinstance(runner, core._MlaStageOneWinnerRunner):
        return "incumbent"
    if isinstance(runner, core._BatchMLAPagedAttentionTrtllmGenBackend):
        return "trtllm-gen"
    return "cute-dsl"


class _NoLaunch:
    """Stands in for a selected runner: the call launches nothing."""

    def __call__(self, *args, **kwargs):
        return None


def _candidate_calls(monkeypatch, case, *, tune=False):
    """``(op, candidate names, tuning config)`` of every selection one call
    makes (a second stage that only looks up is recorded from its lookup);
    the call launches nothing."""
    calls = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        calls.append((op, [_name(runner) for runner in runners], config))
        return _NoLaunch(), -1

    original_search_cache = core.AutoTuner.search_cache

    def search_cache(self, op, runners, input_shapes, config, *args, **kwargs):
        if op != core._MLA_CAKE_MARGIN_OP:
            return original_search_cache(
                self, op, runners, input_shapes, config, *args, **kwargs
            )
        calls.append((op, [_name(runner) for runner in runners], config))
        return False, 0, -1, None

    with monkeypatch.context() as patch:
        patch.setattr(core.AutoTuner, "choose_one", choose_one)
        patch.setattr(core.AutoTuner, "search_cache", search_cache)
        with autotune(tune):
            _decode(case)
    return calls


def _names(calls):
    return [name for _, names, _ in calls for name in names]


def _ops(calls):
    return [op for op, _, _ in calls]


def _spy_choose_one(monkeypatch):
    """Records ``(op, runners, (runner, tactic))`` of every selection."""
    calls = []
    original = core.AutoTuner.choose_one

    def choose_one(self, op, runners, config, inputs, **kwargs):
        selected = original(self, op, runners, config, inputs, **kwargs)
        calls.append((op, list(runners), selected))
        return selected

    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    return calls


def _bucket_range(batch_size):
    """The batches of the tuning bucket of ``batch_size`` (hybrid buckets)."""
    buckets = get_hybrid_num_tokens_buckets(core._TRTLLM_GEN_MLA_MAX_BATCH)
    first = max(bucket for bucket in buckets if bucket <= batch_size)
    larger = [bucket for bucket in buckets if bucket > first]
    return first, (larger[0] - 1 if larger else batch_size)


def _offer_case(monkeypatch, case, domain=None):
    """Replace the offer table with a synthetic one listing the unit of this
    case on this device: its domain (or ``domain``) over its tuning bucket."""
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    query = case["query"]
    page_size = case["kv_cache"].shape[-2]
    if domain is None:
        runner = _cpu_runner(
            query.dtype,
            page_size,
            case["max_seq_len"],
            bmm2_scale=case["bmm2_scale"],
        )
        inputs = [query, case["block_tables"], case["seq_lens"], _nan_out(case)]
        domain = (
            runner._domain(inputs, num_sms, arch=arch)
            or cake_module.DOMAIN_BF16_VQUARTER
        )
    first, last = _bucket_range(query.shape[0])
    key = (
        arch,
        num_sms,
        core._cake_mla_dtype_name(query.dtype),
        page_size,
        query.shape[1],
        False,
    )
    unit = (domain, _round_to_seq_len_bucket(case["max_seq_len"]), first, last)
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {key: (unit,)})


def _clear_all_cake_mla_memos():
    """Drop every memoized Cake host fact and selection decision."""
    core._CAKE_MLA_AUTO_MEMO.clear()
    core._CAKE_MLA_STAGE_TWO_MEMO.clear()
    core._CAKE_MLA_TABLE_MEMO.clear()
    core._CAKE_MLA_TABLE_MEMO_OWNER[0] = None
    core._CAKE_MLA_RANGES_MEMO[0] = None
    core._CAKE_MLA_GATE_MEMO[0] = None
    core._CAKE_MLA_NO_PROGRAM_DEVICES.clear()
    core._cake_mla_auto_static_reason.cache_clear()


@pytest.fixture(autouse=True)
def _clear_cake_mla_memos():
    """Every test starts and ends without memoized Cake host facts or
    selection decisions (they are process-global)."""
    _clear_all_cake_mla_memos()
    yield
    _clear_all_cake_mla_memos()


@pytest.fixture
def clean_tuner(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    yield tuner
    tuner.clear_cache()


@pytest.fixture
def fresh_tuner(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    return tuner


def _hold_cake_entry(tuner) -> None:
    """Give ``tuner`` a Cake MLA winner for an op no call uses, as after a
    tuning run Cake won elsewhere: Cake is then listed outside tuning."""
    key = ProfilingCacheKey(
        custom_op="cake_mla_candidate_gate_test",
        runner_class_name=core._CAKE_MLA_RUNNER,
        runner_hash=0,
        nearest_profile=((1,),),
        extras=(),
    )
    tuner.profiling_cache[key] = (-1, None)


@pytest.mark.parametrize(
    "dtype,page_size",
    [(torch.bfloat16, 32), (torch.bfloat16, 64), (torch.float8_e4m3fn, 64)],
)
def test_first_stage_is_unchanged_and_cake_joins_the_second(
    monkeypatch, fresh_tuner, dtype, page_size
):
    """The first selection has the op, runners and tuning config of a call
    without Cake; Cake joins only the second selection, after the first-stage
    winner, with the first stage's batch mapping."""
    _require_cake_arch()
    # The Cake Blackwell route is a stable backend: no experimental opt-in.
    monkeypatch.delenv(EXPERIMENTAL_ENV, raising=False)
    case = _make_case(dtype=dtype, page_size=page_size)
    _offer_case(monkeypatch, case)

    enabled = _candidate_calls(monkeypatch, case, tune=True)
    monkeypatch.setenv(DISABLE_ENV, "1")
    disabled = _candidate_calls(monkeypatch, case, tune=True)

    assert _ops(disabled) == [core._MLA_DECODE_OP]
    assert disabled[0][1] and "cake" not in disabled[0][1]
    assert enabled[0][:2] == disabled[0][:2] and enabled[0][2] is disabled[0][2]
    assert [call[:2] for call in enabled[1:]] == [
        (core._MLA_CAKE_MARGIN_OP, ["cake", "incumbent"])
    ]
    first, second = enabled[0][2], enabled[1][2]
    assert (
        second.dynamic_tensor_specs[0].map_to_tuning_buckets
        is first.dynamic_tensor_specs[0].map_to_tuning_buckets
    )
    assert set(second.dynamic_tensor_specs[0].gen_tuning_buckets) <= set(
        first.dynamic_tensor_specs[0].gen_tuning_buckets
    )


def test_unlisted_unit_runs_the_first_stage_only(monkeypatch, fresh_tuner):
    """With no unit listed for the device a tuned call makes only the first
    selection."""
    _require_cake_arch()
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {})
    calls = _candidate_calls(monkeypatch, _make_case(), tune=True)
    assert _ops(calls) == [core._MLA_DECODE_OP] and "cake" not in _names(calls)


def _sparse(case):
    batch_size, q_len = case["query"].shape[:2]
    tokens = case["kv_cache"].shape[0] * case["kv_cache"].shape[1]
    indices = torch.randint(0, tokens, (batch_size, q_len, 128), device="cuda")
    return dict(case, sparse_mla_top_k=128, block_tables=indices.to(torch.int32))


def _counter_buffer(case):
    batch_size, _, num_heads, _ = case["query"].shape
    size = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        batch_size, num_heads, get_device_sm_count(torch.device("cuda"))
    )
    buffer = torch.zeros(size, dtype=torch.uint8, device="cuda")
    return dict(case, multi_ctas_kv_counter_buffer=buffer)


def _fp8_sink(case, monkeypatch):
    fp8 = _make_case(dtype=torch.float8_e4m3fn, page_size=64)
    return dict(fp8, sinks=[torch.zeros(NUM_HEADS, dtype=torch.float32, device="cuda")])


def _switch_off(case, monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "1")
    return case


def _unsupported_arch(case, monkeypatch):
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 7))
    return case


def _tensor_scales(case, monkeypatch):
    return dict(
        case,
        bmm1_scale=torch.tensor([case["bmm1_scale"]], device="cuda"),
        bmm2_scale=torch.tensor([1.0], device="cuda"),
    )


def _separate_kv_page_indices(case, monkeypatch):
    block_tables = torch.stack([case["block_tables"]] * 2, dim=1)
    return dict(case, uses_shared_paged_kv_idx=False, block_tables=block_tables)


_ABSENT = {
    "switch_off": _switch_off,
    "unsupported_arch": _unsupported_arch,
    "unsupported_dimension_tuple": lambda case, mp: _make_case(num_heads=16),
    "tensor_scales": _tensor_scales,
    "lse_requested": lambda case, mp: dict(case, return_lse=True),
    "skip_softmax": lambda case, mp: dict(
        case, skip_softmax_threshold_scale_factor=1e-30
    ),
    "sparse": lambda case, mp: _sparse(case),
    "counter_buffer": lambda case, mp: _counter_buffer(case),
    "separate_kv_page_indices": _separate_kv_page_indices,
    "explicit_trtllm_gen": lambda case, mp: dict(case, backend="trtllm-gen"),
    # The stateless route would take the generic tail program.
    "fp8_page32_tail": lambda case, mp: _make_case(
        dtype=torch.float8_e4m3fn, page_size=32
    ),
    "fp8_sink": _fp8_sink,
}


@pytest.mark.parametrize("variant", sorted(_ABSENT))
def test_cake_auto_candidate_absent(monkeypatch, fresh_tuner, variant):
    _require_sm10x()
    base = _make_case()
    if _compute_capability() in ((10, 0), (10, 3)):
        # Offer the unit of the base request, so only the variant decides.
        _offer_case(monkeypatch, base)
    case = _ABSENT[variant](base, monkeypatch)
    # Tuning opens the tuner gate, so only the request decides.
    try:
        calls = _candidate_calls(monkeypatch, case, tune=True)
    except ValueError as error:
        # No other candidate serves the request either; Cake never joins
        # without a first stage.
        assert "no backend supports" in str(error)
        return
    assert _ops(calls) == [core._MLA_DECODE_OP]
    assert _names(calls) and "cake" not in _names(calls)


def test_mla_cake_gate_follows_tuner_and_process_group(monkeypatch, fresh_tuner):
    """The listing gate: closed for an untuned call without a Cake entry,
    open while tuning or holding a Cake entry, closed while an autotune
    process group tunes or the switch is off. No GPU needed."""
    runner = core._CAKE_MLA_RUNNER
    assert runner == core._CakeTrtllmMlaBlackwellRunner.__name__

    assert not cake_auto_candidate_open(runner)
    monkeypatch.setattr(fresh_tuner, "is_tuning_mode", True)
    assert cake_auto_candidate_open(runner)
    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    assert not cake_auto_candidate_open(runner)

    # Serving with the group still set lists Cake as without a group.
    monkeypatch.setattr(fresh_tuner, "is_tuning_mode", False)
    assert not cake_auto_candidate_open(runner)
    _hold_cake_entry(fresh_tuner)
    assert cake_auto_candidate_open(runner)
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert not cake_auto_candidate_open(runner)


def test_cake_gate_memo_follows_the_tuner_state(monkeypatch, fresh_tuner):
    """The dispatcher's memoized Cake candidate gate is reused only for the
    same tuner and memo state: a Cake winner added elsewhere, the switch
    turned off and on again and another tuner evaluate the gate again, and a
    result whose switch changed after the state was read is not stored. No GPU
    needed."""
    gates = []
    original = core.cake_auto_candidate_open

    def gate(runner_class_name):
        gates.append(runner_class_name)
        return original(runner_class_name)

    monkeypatch.setattr(core, "cake_auto_candidate_open", gate)

    def candidate_open(tuner):
        return core._mla_cake_candidate_open(tuner, core._mla_cake_memo_state(tuner))

    assert [candidate_open(fresh_tuner) for _ in range(2)] == [False] * 2
    assert len(gates) == 1
    _hold_cake_entry(fresh_tuner)  # a new winner-source generation
    assert [candidate_open(fresh_tuner) for _ in range(2)] == [True] * 2
    assert len(gates) == 2
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert [candidate_open(fresh_tuner) for _ in range(2)] == [False] * 2
    assert len(gates) == 3
    monkeypatch.delenv(DISABLE_ENV)
    assert [candidate_open(fresh_tuner) for _ in range(2)] == [True] * 2
    assert len(gates) == 4
    # Another tuner with the same state: the gate reads that tuner.
    other = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", other)
    assert core._mla_cake_memo_state(other) == core._mla_cake_memo_state(fresh_tuner)
    assert not candidate_open(other)
    assert len(gates) == 5
    monkeypatch.setattr(AutoTuner, "_instance", fresh_tuner)
    state = core._mla_cake_memo_state(fresh_tuner)
    # Switched off after the state was read: the result is not stored.
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert not core._mla_cake_candidate_open(fresh_tuner, state)
    monkeypatch.delenv(DISABLE_ENV)
    assert core._mla_cake_candidate_open(fresh_tuner, state)
    assert core._mla_cake_candidate_open(fresh_tuner, state)
    assert len(gates) == 7


def _memo_state_through_accessors(tuner):
    """``_mla_cake_memo_state`` read through the tuner's accessors and
    ``cake_auto_candidates_enabled`` (the reference of its inlined reads)."""
    if tuner.is_tuning_mode or tuner._managed_cache is not None:
        return None
    skip_ops = tuner._effective_skip_ops
    if core._MLA_DECODE_OP in skip_ops or core._MLA_CAKE_MARGIN_OP in skip_ops:
        return None
    local = tuner._v2_local
    if (
        local.active
        or local.replay_report is not None
        or local.strict_replay
        or local.measure is not None
    ):
        return None
    stack = tuner._get_override_stack()
    return (
        autotuner_module._winner_sources_generation,
        tuple(stack[-1][:3]) if stack else None,
        (
            os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0"),
            cake_auto_candidates_enabled(),
        ),
    )


def test_memo_state_reads_the_switch_and_stacks_as_the_tuner(monkeypatch, fresh_tuner):
    """The memo state reads the Cake switch on every call, compared as
    ``cake_auto_candidates_enabled`` compares it (also flipped between two
    calls), and the thread's skip_ops and override stacks as the tuner's
    accessors read them, also in a thread that never created them. No GPU
    needed."""
    tuner = fresh_tuner
    states = []
    for value in (None, "1", "0", "", "true", " 1", "1", None):
        if value is None:
            monkeypatch.delenv(DISABLE_ENV, raising=False)
        else:
            monkeypatch.setenv(DISABLE_ENV, value)
        state = core._mla_cake_memo_state(tuner)
        assert state == _memo_state_through_accessors(tuner), value
        assert state[2] == core._mla_cake_memo_settings(), value
        assert state[2][1] == cake_auto_candidates_enabled(), value
        states.append(state)
    enabled = [True, False, True, True, True, True, False, True]
    assert [state[2][1] for state in states] == enabled
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    bundled = os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    with monkeypatch.context() as patch:
        patch.setenv(
            "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0" if bundled == "1" else "1"
        )
        state = core._mla_cake_memo_state(tuner)
        assert state == _memo_state_through_accessors(tuner)
        assert state != states[0]

    contexts = (
        lambda: autotune(False),
        lambda: autotune(False, tuning_buckets=(1, 2, 4)),
        lambda: autotune(False, round_up=True),
        lambda: autotune(False, skip_ops="another_op"),
        lambda: autotune(False, skip_ops=core._MLA_DECODE_OP),
        lambda: autotune(False, skip_ops=core._MLA_CAKE_MARGIN_OP),
    )
    for index, context in enumerate(contexts):
        with context():
            state = core._mla_cake_memo_state(tuner)
            assert state == _memo_state_through_accessors(tuner), index
            with autotune(False, skip_ops="nested_op"):
                state = core._mla_cake_memo_state(tuner)
                assert state == _memo_state_through_accessors(tuner), index

    results = []

    def fresh_thread():
        # No stack was created in this thread before the first read, and the
        # read creates none (the accessors of the reference do).
        first = core._mla_cake_memo_state(tuner)
        created = (
            hasattr(tuner._skip_ops_local, "stack"),
            hasattr(tuner._override_local, "stack"),
        )
        results.append((first, _memo_state_through_accessors(tuner), created))

    thread = threading.Thread(target=fresh_thread)
    thread.start()
    thread.join()
    assert results and results[0][0] == results[0][1] is not None
    assert results[0][2] == (False, False)


class _Workspace:
    """A workspace buffer of ``nbytes`` bytes, for the bucket computation."""

    def __init__(self, nbytes):
        self.nbytes = nbytes

    def numel(self):
        return self.nbytes

    def element_size(self):
        return 1


def test_decode_buckets_skip_the_cute_dsl_search_with_trtllm_gen(monkeypatch):
    """With TRTLLM-GEN listed the buckets are the hybrid buckets of its cap
    (8192, at least the CuTe DSL cap), without the CuTe DSL workspace search,
    and the memoized tuple is returned on every call. No GPU needed."""

    def no_search(*args, **kwargs):
        raise AssertionError("the CuTe DSL workspace search ran")

    monkeypatch.setattr(core, "_resolve_cute_dsl_workspace_sizer", no_search)
    expected = get_hybrid_num_tokens_buckets(core._TRTLLM_GEN_MLA_MAX_BATCH)
    results = [
        core._compute_mla_decode_buckets(
            _Workspace(nbytes),
            runner_names,
            q_len,
            NUM_HEADS,
            LATENT,
            max_seq_len,
            torch.device("cpu"),
            "auto",
            None,
        )
        for runner_names in (("trtllm-gen",), ("trtllm-gen", "cute-dsl"))
        for q_len in (1, 2, 4)
        for max_seq_len in (128, 16384)
        for nbytes in (0, 128 << 20)
    ]
    assert all(result == expected for result in results)
    assert all(result is results[0] for result in results)


def _reference_cute_dsl_cap(nbytes, q_len, max_seq_len, num_sms, impl, sinks):
    """The CuTe DSL batch cap before memoization: the workspace search."""
    sizer, resolved = core._resolve_cute_dsl_workspace_sizer(impl, sinks, False)
    lo, hi = 1, core._TRTLLM_GEN_MLA_MAX_BATCH
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, workspace_bytes = core._call_cute_dsl_workspace_sizer(
            sizer, resolved, mid, q_len, NUM_HEADS, LATENT, num_sms, max_seq_len
        )
        if workspace_bytes <= nbytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _reference_decode_buckets(
    nbytes, runner_names, q_len, max_seq_len, num_sms, impl, sinks
):
    """``_compute_mla_decode_buckets`` before memoization: the CuTe DSL
    workspace search runs whenever CuTe DSL is listed."""
    cap = 0
    if "trtllm-gen" in runner_names:
        cap = core._TRTLLM_GEN_MLA_MAX_BATCH
    if "cute-dsl" in runner_names:
        cap = max(
            cap,
            _reference_cute_dsl_cap(nbytes, q_len, max_seq_len, num_sms, impl, sinks),
        )
    return get_hybrid_num_tokens_buckets(max(1, cap))


def test_decode_buckets_match_the_unmemoized_computation(monkeypatch):
    """The buckets equal those of the unmemoized computation over runner
    sets, q_len, max_seq_len, workspace sizes, SM counts, CuTe DSL
    implementations and sinks (also on a repeated, memoized call); where the
    unmemoized computation raises for CuTe DSL alone, so does the memoized
    one. No GPU needed."""
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    from flashinfer.cute_dsl import utils as cute_dsl_utils

    num_sms = {"value": 148}
    monkeypatch.setattr(cute_dsl_utils, "get_num_sm", lambda device: num_sms["value"])
    trtllm_gen_buckets = get_hybrid_num_tokens_buckets(core._TRTLLM_GEN_MLA_MAX_BATCH)
    sinks_tensor = torch.zeros(NUM_HEADS, dtype=torch.float32)
    checked = 0
    for case in itertools.product(
        (148, 160),
        (("trtllm-gen",), ("cute-dsl",), ("trtllm-gen", "cute-dsl")),
        (1, 2, 4),
        (128, 1024, 4096, 16384),
        (0, 1 << 20, 16 << 20, 128 << 20),
        ("auto", "modular", "monolithic"),
        (None, [sinks_tensor]),
    ):
        sms, runner_names, q_len, max_seq_len, nbytes, impl, sinks = case
        num_sms["value"] = sms
        args = (
            _Workspace(nbytes),
            runner_names,
            q_len,
            NUM_HEADS,
            LATENT,
            max_seq_len,
            torch.device("cpu"),
            impl,
            sinks,
        )
        try:
            expected = _reference_decode_buckets(
                nbytes, runner_names, q_len, max_seq_len, sms, impl, sinks
            )
        except ValueError:
            if "trtllm-gen" in runner_names:
                # No CuTe DSL search, and so no error, with TRTLLM-GEN listed.
                actual = core._compute_mla_decode_buckets(*args)
                assert actual == trtllm_gen_buckets, case
            else:
                with pytest.raises(ValueError):
                    core._compute_mla_decode_buckets(*args)
            continue
        for _ in range(2):
            assert core._compute_mla_decode_buckets(*args) == expected, case
        checked += 1
    assert checked > 0


def test_cute_dsl_batch_search_shares_entries_across_unread_max_seq_len(
    monkeypatch,
):
    """The memoized CuTe DSL workspace search keys ``max_seq_len`` only for
    the policy that reads it (monolithic): with the modular policy a call
    with another ``max_seq_len`` hits the entry; with the monolithic policy
    it searches again. No GPU needed."""
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    sizer_calls = []
    sizer = core._call_cute_dsl_workspace_sizer

    def counting_sizer(*args, **kwargs):
        sizer_calls.append(args)
        return sizer(*args, **kwargs)

    monkeypatch.setattr(core, "_call_cute_dsl_workspace_sizer", counting_sizer)
    core._cute_dsl_max_batch_search.cache_clear()

    def cap(impl, max_seq_len):
        return core._cute_dsl_max_supported_batch(
            workspace_bytes=16 << 20,
            q_len=1,
            num_heads=NUM_HEADS,
            kv_lora_rank=LATENT,
            max_active_blocks=148,
            max_seq_len=max_seq_len,
            candidate_max=core._TRTLLM_GEN_MLA_MAX_BATCH,
            cute_dsl_impl=impl,
            sinks=None,
        )

    try:
        first = cap("modular", 1024)
        assert sizer_calls
        sizer_calls.clear()
        assert [cap("modular", msl) for msl in (1025, 4096, 16384)] == [first] * 3
        assert not sizer_calls
        cap("monolithic", 1024)
        sizer_calls.clear()
        cap("monolithic", 1025)
        assert sizer_calls
    finally:
        core._cute_dsl_max_batch_search.cache_clear()


def test_untuned_auto_builds_no_cake_runner(monkeypatch, fresh_tuner):
    """Without a Cake entry an untuned call skips the Cake eligibility check,
    builds no Cake runner and makes the first selection only; holding one adds
    the second selection."""
    _require_cake_arch()
    built = []
    original_init = core._CakeTrtllmMlaBlackwellRunner.__init__

    def init(self, **kwargs):
        built.append(self)
        original_init(self, **kwargs)

    def reason(**kwargs):
        raise AssertionError("the Cake eligibility check ran")

    case = _make_case()
    _offer_case(monkeypatch, case)
    with monkeypatch.context() as patch:
        patch.setattr(core._CakeTrtllmMlaBlackwellRunner, "__init__", init)
        patch.setattr(core, "_cake_mla_auto_candidate_reason", reason)
        calls = _candidate_calls(monkeypatch, case)
    assert _ops(calls) == [core._MLA_DECODE_OP] and not built
    assert calls[0][1] and "cake" not in calls[0][1]

    _hold_cake_entry(fresh_tuner)
    held = _candidate_calls(monkeypatch, case)
    assert held[0][:2] == calls[0][:2]
    assert [call[:2] for call in held[1:]] == [
        (core._MLA_CAKE_MARGIN_OP, ["cake", "incumbent"])
    ]


def test_group_tuning_lists_no_cake(monkeypatch, fresh_tuner):
    """Every rank of an autotune process group must list the same runners, so
    group tuning makes the first selection only; tuning without the group adds
    the second."""
    _require_cake_arch()
    case = _make_case()
    _offer_case(monkeypatch, case)
    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    grouped = _candidate_calls(monkeypatch, case, tune=True)
    monkeypatch.setattr(autotuner_module, "_tune_process_group", None)
    alone = _candidate_calls(monkeypatch, case, tune=True)
    assert _ops(grouped) == [core._MLA_DECODE_OP]
    assert grouped[0][1] and "cake" not in grouped[0][1]
    assert alone[0][:2] == grouped[0][:2]
    assert _ops(alone) == [core._MLA_DECODE_OP, core._MLA_CAKE_MARGIN_OP]


def test_cake_auto_candidate_reason_filters_inputs(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 0))
    case = _make_case(batch_size=2)
    base = dict(
        query=case["query"],
        kv_cache=case["kv_cache"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        out=torch.empty(2, 1, NUM_HEADS, LATENT, dtype=torch.bfloat16, device="cuda"),
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=QK_DIM - LATENT,
        sparse_mla_top_k=0,
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=1.0,
        sinks=None,
        skip_softmax_threshold_scale_factor=None,
        uses_shared_paged_kv_idx=True,
        lse_requested=False,
        use_fp16_softmax=None,
        multi_ctas_kv_counter_buffer=None,
        sparse_mla_top_k_lens=None,
        enable_dcp=False,
    )
    reason = core._cake_mla_auto_candidate_reason
    assert reason(**base) is None
    sink = torch.zeros(NUM_HEADS, dtype=torch.float32, device="cuda")
    assert reason(**dict(base, sinks=[sink])) is None
    fp8_query = case["query"].to(torch.float8_e4m3fn)
    fp8_page64 = case["kv_cache"].reshape(-1, 64, QK_DIM).to(torch.float8_e4m3fn)
    fp8 = dict(base, query=fp8_query, kv_cache=fp8_page64)
    assert reason(**fp8) is None

    rejected = {
        "page_size": dict(kv_cache=case["kv_cache"].reshape(-1, 16, QK_DIM)),
        "sink_dtype": dict(sinks=[sink.to(torch.bfloat16)]),
        "seq_lens_dtype": dict(seq_lens=case["seq_lens"].long()),
        "strided_out": dict(
            out=torch.empty(2, 1, NUM_HEADS, 2 * LATENT, device="cuda").to(
                torch.bfloat16
            )[..., :LATENT]
        ),
        "nonfinite_scale": dict(bmm1_scale=float("inf")),
        "fp16_softmax": dict(use_fp16_softmax=True),
        "dcp": dict(enable_dcp=True),
        # The generic tail program writes NaN for a zero-length request.
        "fp8_page32_tail": dict(
            query=fp8_query, kv_cache=case["kv_cache"].to(torch.float8_e4m3fn)
        ),
        "fp8_sink_tail": dict(query=fp8_query, kv_cache=fp8_page64, sinks=[sink]),
        "heads64_tail": dict(
            query=case["query"][:, :, :64].contiguous(),
            sinks=None,
        ),
    }
    for name, override in rejected.items():
        assert reason(**dict(base, **override)) is not None, name

    monkeypatch.setattr(core, "get_compute_capability", lambda device: (10, 7))
    assert reason(**base) is not None


def _cpu_runner(dtype=torch.bfloat16, page_size=64, max_seq_len=1024, **kwargs):
    kv_cache = torch.empty(8, page_size, QK_DIM, dtype=dtype)
    options = dict(
        kv_cache=kv_cache,
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=QK_DIM - LATENT,
        max_seq_len=max_seq_len,
        bmm1_scale=0.1,
        bmm2_scale=1.0,
        sinks=None,
        enable_pdl=True,
    )
    options.update(kwargs)
    return core._CakeTrtllmMlaBlackwellRunner(**options)


def _cpu_inputs(dtype, batch_size, q_len=1, num_heads=NUM_HEADS):
    return [
        torch.empty(batch_size, q_len, num_heads, QK_DIM, dtype=dtype),
        torch.zeros(batch_size, 32, dtype=torch.int32),
        torch.ones(batch_size, dtype=torch.int32),
        torch.empty(batch_size, q_len, num_heads, LATENT, dtype=torch.bfloat16),
    ]


def _bucket(batch_size, next_bucket=None):
    """The tuner's profile of the bucket ``batch_size``: its batch dimension's max
    is the next bucket, or infinity for the largest bucket."""
    upper = float("inf") if next_bucket is None else next_bucket
    return OptimizationProfile([[DynamicDim(batch_size, batch_size, upper)]], [None])


@pytest.fixture
def cake_families(monkeypatch):
    """Simulate a 148-SM SM100a device whose Cake programs build without a
    toolchain: returns the families built and the capture switch."""
    built = []
    state = {"capturing": False}
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: ("sm_100a", 148))
    monkeypatch.setattr(
        cake_module, "get_cake_trtllm_mla_blackwell_family", built.append
    )
    monkeypatch.setattr(cake_module, "_FP8_LUTS", {})
    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state["capturing"]
    )
    return built, state


def _family(domain, rows=None):
    """The family of ``domain`` on the simulated 148-SM SM100a device."""
    if rows is None:
        return cake_module.family_for(domain, "sm_100a", 148)
    return cake_module.launch_family(domain, "sm_100a", 148, rows)


_DOMAINS = (
    cake_module.DOMAIN_BF16_VQUARTER,
    cake_module.DOMAIN_BF16_VHALF,
    cake_module.DOMAIN_BF16_CLC,
    cake_module.DOMAIN_FP8_PAGE64,
)


@pytest.fixture
def offer_everything(monkeypatch):
    """A synthetic offer table listing every domain, batch and configuration
    (max_seq_len 1024, no sinks) on both simulated devices, so the bucket rule
    alone decides."""
    table = {
        (arch, num_sms, dtype, page_size, q_len, False): tuple(
            (domain, 1024, 1, core._CAKE_MLA_MAX_ROWS) for domain in _DOMAINS
        )
        for arch, num_sms in (("sm_100a", 148), ("sm_103a", 152))
        for dtype in ("bfloat16", "float8_e4m3fn")
        for page_size in (32, 64)
        for q_len in (1, 2)
    }
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", table)
    return table


# A unit as the derived table lists one: B200, BF16, page 64, one query token,
# max_seq_len bucket 1024, batches 1-15 (tuning buckets 1, 2, 4 and 8), all on
# the quarter-value split program (at most 18 rows on 148 SMs).
_SYNTHETIC_OFFERS = {
    ("sm_100a", 148, "bfloat16", 64, 1, False): (
        (cake_module.DOMAIN_BF16_VQUARTER, 1024, 1, 15),
    ),
}


@pytest.fixture
def synthetic_offers(monkeypatch):
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", dict(_SYNTHETIC_OFFERS))


class _StubIncumbent:
    """Fixed first-stage winner lookups in place of ``_MlaStageOneWinnerRunner``."""

    def __init__(self, winner=WINNER):
        self.winner = winner

    def profile_winner(self, profile):
        return self.winner

    def call_winner(self, inputs):
        return self.winner

    def times_profile(self, profile):
        return True


def _staged(runner, winner=WINNER):
    """``runner`` with a first-stage winner, as the second stage sets it."""
    runner._incumbent = _StubIncumbent(winner)
    return runner


def _serves(runner, inputs, profile) -> bool:
    return runner._serves_bucket(inputs, inputs[0].size(0), profile)


def test_cake_bucket_rule_covers_every_batch_of_the_bucket(
    cake_families, offer_everything
):
    """A bucket offers Cake only when every batch mapped to the bucket (up to the
    next bucket) launches the programs of the profiled batch, within the row cap,
    and never the largest bucket. No GPU needed."""
    built, _ = cake_families
    bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn
    page64 = _cpu_runner(bf16, 64)
    # 148 SMs (74 resident clusters): aligned BF16 takes VQUARTER while
    # 4 * rows <= 74 and VHALF while 2 * rows <= 74; page 64 has nothing above.
    # Bucket 4 of two-token rows serves 8-14 VQUARTER rows.
    assert _serves(page64, _cpu_inputs(bf16, 4, 2), _bucket(4, 8))
    assert _family(cake_module.DOMAIN_BF16_VQUARTER) in built
    # Bucket 8 spans VQUARTER (16-18 rows) and VHALF (20-30 rows): the profile
    # measures one of them only.
    assert not _serves(page64, _cpu_inputs(bf16, 8, 2), _bucket(8, 16))
    # Batches 38-63 of bucket 32 leave its domains.
    assert not _serves(page64, _cpu_inputs(bf16, 32), _bucket(32, 64))
    # The largest bucket also serves every larger batch.
    assert not _serves(page64, _cpu_inputs(bf16, 4, 2), _bucket(4))
    assert _serves(page64, _cpu_inputs(bf16, 4, 2), None)
    # Requests with sinks are never offered, although the table lists the unit
    # without sinks.
    sink = [torch.zeros(NUM_HEADS, dtype=torch.float32)]
    page64_sink = _cpu_runner(bf16, 64, sinks=sink)
    assert not _serves(page64_sink, _cpu_inputs(bf16, 16), _bucket(16, 32))

    # Page 32 reaches the generic CLC program above 37 rows. Bucket 32 spans
    # VHALF and the one-wave program, bucket 64 the one-wave (64-74 rows) and
    # the ordered program (75-127 rows): neither is offered.
    page32 = _cpu_runner(bf16, 32)
    assert not _serves(page32, _cpu_inputs(bf16, 32), _bucket(32, 64))
    assert not _serves(page32, _cpu_inputs(bf16, 64), _bucket(64, 128))
    # Bucket 128 launches the longest-first order and the generic CLC program
    # for every batch; both are built.
    assert _serves(page32, _cpu_inputs(bf16, 128), _bucket(128, 256))
    window = {
        _family(cake_module.DOMAIN_BF16_CLC_ORDER),
        _family(cake_module.DOMAIN_BF16_CLC, 128),
    }
    assert len(window) == 2 and window <= core._CAKE_MLA_LOADED_FAMILIES
    assert _family(cake_module.DOMAIN_BF16_CLC, 64) != _family(
        cake_module.DOMAIN_BF16_CLC, 128
    )
    # The row cap bounds the bucket's last batch.
    cap = core._CAKE_MLA_MAX_ROWS
    assert _serves(page32, _cpu_inputs(bf16, 256), _bucket(256, cap))
    assert not _serves(page32, _cpu_inputs(bf16, 256), _bucket(256, 1024))
    assert _serves(page32, _cpu_inputs(bf16, 128, 2), _bucket(128, 256))
    assert not _serves(page32, _cpu_inputs(bf16, 256, 2), _bucket(256, 512))
    assert page32.supports(_cpu_inputs(bf16, cap))
    assert not page32.supports(_cpu_inputs(bf16, cap + 1))

    # FP8 page 64 always launches its one program, up to its own row cap; the
    # tail program (FP8 page 32, other head counts) never joins.
    fp8_cap = core._CAKE_MLA_FP8_MAX_ROWS
    fp8_page64 = _cpu_runner(fp8, 64)
    assert _serves(fp8_page64, _cpu_inputs(fp8, 4), _bucket(4, 8))
    assert _serves(
        fp8_page64, _cpu_inputs(fp8, fp8_cap // 2), _bucket(fp8_cap // 2, fp8_cap)
    )
    assert not _serves(
        fp8_page64, _cpu_inputs(fp8, fp8_cap), _bucket(fp8_cap, 2 * fp8_cap)
    )
    assert fp8_page64.supports(_cpu_inputs(fp8, fp8_cap))
    assert not fp8_page64.supports(_cpu_inputs(fp8, fp8_cap + 1))
    fp8_page32 = _staged(_cpu_runner(fp8, 32))
    assert fp8_page32.get_valid_tactics(_cpu_inputs(fp8, 4), _bucket(4, 8)) == []
    # A cached Cake choice for an FP8 page-32 shape (e.g. from an older cache
    # file) never launches the tail program.
    assert not fp8_page32.validate_tactic(_cpu_inputs(fp8, 4), WINNER)
    heads64 = _staged(_cpu_runner(bf16, 64))
    assert (
        heads64.get_valid_tactics(_cpu_inputs(bf16, 4, num_heads=64), _bucket(4, 8))
        == []
    )
    assert not {
        _family(cake_module.DOMAIN_FP8_TAIL),
        _family(cake_module.DOMAIN_BF16_TAIL),
    } & set(built)

    # Revalidating a reused bucket during tuning checks the bucket's batches,
    # not the caller's.
    staged = _staged(_cpu_runner(bf16, 64))
    caller = _cpu_inputs(bf16, 300, 2)
    assert staged.validate_tactic_for_profile(caller, WINNER, _bucket(4, 8))
    assert not staged.validate_tactic_for_profile(caller, WINNER, _bucket(8, 16))
    assert not staged.validate_tactic(caller, WINNER)


def test_cake_bucket_rule_on_generated_profiles(cake_families, offer_everything):
    """The bucket rule holds on the profiles the tuner generates from the MLA
    tuning config: their batch dimension is a DynamicDim whose max is the next
    bucket. Two-token rows. No GPU needed."""
    bf16 = torch.bfloat16
    config = core._mla_decode_tuning_config(
        buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256), num_pages=64, profile_seq_len=1024
    )
    profiles = AutoTuner()._generate_optimization_profiles(
        config, _cpu_inputs(bf16, 4, 2)
    )
    page64 = _cpu_runner(bf16, 64)
    page32 = _cpu_runner(bf16, 32)
    served = {}
    for profile in profiles:
        dim = profile.shapes[0][0]
        assert isinstance(dim, DynamicDim)
        inputs = _cpu_inputs(bf16, dim.opt, 2)
        served[dim.opt] = (
            _serves(page64, inputs, profile),
            _serves(page32, inputs, profile),
        )
    # 148 SMs: VQUARTER up to 18 rows (batch 9), VHALF up to 37 (batch 18), then
    # (page 32 only) the one-wave CLC program up to 74 rows (batch 37) and the
    # ordered CLC program above.
    assert served == {
        1: (True, True),
        2: (True, True),
        4: (True, True),
        8: (False, False),
        16: (False, False),
        32: (False, False),
        64: (False, True),
        128: (False, True),
        256: (False, False),
    }


def test_cake_validate_tactic_contains_builds_and_capture(
    cake_families, offer_everything
):
    """A cached Cake choice is a miss when the batch leaves its domains or row
    cap, or when a stream capture would have to build its program or allocate
    its FP8 decode table; outside capture it builds both. No GPU needed."""
    built, state = cake_families
    bf16 = torch.bfloat16
    runner = _staged(_cpu_runner(bf16, 64))
    inputs = _cpu_inputs(bf16, 4, 2)
    device = inputs[0].device

    assert not runner.validate_tactic(inputs, -1)
    assert not runner.validate_tactic(_cpu_inputs(bf16, 64), WINNER)

    state["capturing"] = True
    assert not runner.validate_tactic(inputs, WINNER)
    assert not built and device not in cake_module._FP8_LUTS

    state["capturing"] = False
    assert runner.validate_tactic(inputs, WINNER)
    assert len(built) == 1 and device in cake_module._FP8_LUTS

    state["capturing"] = True
    assert runner.validate_tactic(inputs, WINNER)
    cake_module._FP8_LUTS.clear()
    assert not runner.validate_tactic(inputs, WINNER)
    # A batch of another family is not built during capture either.
    assert not runner.validate_tactic(_cpu_inputs(bf16, 32), WINNER)
    assert len(built) == 1

    # An ordered generic CLC launch also needs the longest-first order program:
    # with only the attention program loaded it is a miss during capture.
    page32 = _staged(_cpu_runner(bf16, 32))
    window = _cpu_inputs(bf16, 128)
    order = _family(cake_module.DOMAIN_BF16_CLC_ORDER)
    state["capturing"] = False
    cake_module._fp8_lut(device)
    state["capturing"] = True
    core._CAKE_MLA_LOADED_FAMILIES.add(_family(cake_module.DOMAIN_BF16_CLC, 128))
    assert not page32.validate_tactic(window, WINNER) and order not in built
    state["capturing"] = False
    assert page32.validate_tactic(window, WINNER) and order in built
    state["capturing"] = True
    assert page32.validate_tactic(window, WINNER)


def test_cake_build_failure_is_recorded_once(
    monkeypatch, cake_families, offer_everything
):
    """A family that fails to build is unavailable for the process: Cake is a
    miss for it and the build is not retried. No GPU needed."""
    attempts = []

    def unavailable(name):
        attempts.append(name)
        raise RuntimeError(f"no toolchain for {name}")

    monkeypatch.setattr(
        cake_module, "get_cake_trtllm_mla_blackwell_family", unavailable
    )
    runner = _staged(_cpu_runner(torch.bfloat16, 64))
    inputs = _cpu_inputs(torch.bfloat16, 4, 2)
    assert not runner.validate_tactic(inputs, WINNER)
    assert not runner.validate_tactic(inputs, WINNER)
    assert runner.get_valid_tactics(inputs, _bucket(4, 8)) == []
    assert len(attempts) == 1 and set(attempts) == core._CAKE_MLA_UNAVAILABLE_FAMILIES


def test_cake_profiled_only_against_a_first_stage_winner(
    cake_families, offer_everything
):
    """Cake offers a profile only when the first stage holds a winner for it,
    and names that winner as its tactic. No GPU needed."""
    built, _ = cake_families
    inputs = _cpu_inputs(torch.bfloat16, 4, 2)
    alone = _cpu_runner(torch.bfloat16, 64)
    assert alone.get_valid_tactics(inputs, _bucket(4, 8)) == []
    assert _staged(_cpu_runner(), None).get_valid_tactics(inputs, _bucket(4, 8)) == []
    assert not built, "Cake built a program without a first-stage winner"
    assert _staged(_cpu_runner()).get_valid_tactics(inputs, _bucket(4, 8)) == [WINNER]


def test_cake_choice_valid_only_while_the_first_stage_keeps_its_winner(
    cake_families, offer_everything
):
    """A cached Cake choice names the first-stage winner it beat: it is a miss
    once the first stage selects another winner or none. No GPU needed."""
    inputs = _cpu_inputs(torch.bfloat16, 4, 2)
    runner = _staged(_cpu_runner())
    assert runner.validate_tactic(inputs, WINNER)
    assert runner.validate_tactic_for_profile(inputs, WINNER, _bucket(4, 8))
    for winner in (("_FakeTrtllmGen", -1), None):
        runner._incumbent.winner = winner
        assert not runner.validate_tactic(inputs, WINNER)
        assert not runner.validate_tactic_for_profile(inputs, WINNER, _bucket(4, 8))


def _check_offer_table(table) -> None:
    """Every unit of an offer table: a supported configuration without sinks;
    a power-of-two max_seq_len bucket; whole tuning buckets within the row cap;
    and every batch on the listed domain of the listed device, at both edges of
    the max_seq_len bucket (a device-specific domain such as split-KV starts at a
    bucket edge, so one bucket never spans two domains)."""
    buckets = get_hybrid_num_tokens_buckets(core._TRTLLM_GEN_MLA_MAX_BATCH)
    for key, units in table.items():
        arch, num_sms, dtype_name, page_size, q_len, has_sinks = key
        assert arch in ("sm_100a", "sm_103a") and num_sms > 0, key
        assert dtype_name in ("bfloat16", "float8_e4m3fn"), key
        assert page_size in (32, 64) and q_len >= 1 and not has_sinks, key
        assert units, key
        dtype = getattr(torch, dtype_name)
        max_rows = (
            core._CAKE_MLA_FP8_MAX_ROWS
            if dtype == torch.float8_e4m3fn
            else core._CAKE_MLA_MAX_ROWS
        )
        for domain, seq_bucket, first, last in units:
            unit = (key, domain, seq_bucket, first, last)
            assert seq_bucket == _round_to_seq_len_bucket(seq_bucket), unit
            assert first in buckets and first <= last, unit
            assert last * q_len <= max_rows, unit
            assert last + 1 in buckets or last == max_rows // q_len, unit
            for max_seq_len in sorted({seq_bucket // 2 + 1, seq_bucket}):
                runner = _cpu_runner(dtype, page_size, max_seq_len)
                for batch_size in range(first, last + 1):
                    inputs = _cpu_inputs(dtype, batch_size, q_len)
                    assert runner._domain(inputs, num_sms, arch=arch) == domain, (
                        unit,
                        max_seq_len,
                        batch_size,
                    )


def test_offer_table_well_formed():
    """The committed offer table and the synthetic one of these tests are well
    formed. No GPU needed."""
    _check_offer_table(core._CAKE_MLA_AUTO_OFFERS)
    _check_offer_table(_SYNTHETIC_OFFERS)
    # The measurements the committed table derives from timed no split-KV
    # unit, so it lists none; _check_offer_table holds every listed unit on
    # its measured domain under the committed _SPLITKV_RANGE.
    assert not any(
        unit[0] == cake_module.DOMAIN_BF16_SPLITKV
        for units in core._CAKE_MLA_AUTO_OFFERS.values()
        for unit in units
    )


def test_committed_offer_table_unlisted_units_offer_nothing(monkeypatch, cake_families):
    """Units the measured table does not list offer nothing: devices without
    measurements, requests with sinks and FP8 with 32-token pages. No Cake
    program or FP8 decode table is built for them. No GPU needed."""
    built, _ = cake_families
    table = core._CAKE_MLA_AUTO_OFFERS
    assert set(table) <= {
        (arch, num_sms, dtype, page_size, q_len, False)
        for arch, num_sms in (("sm_100a", 148), ("sm_103a", 152))
        for dtype in ("bfloat16", "float8_e4m3fn")
        for page_size in (32, 64)
        for q_len in (1, 2)
    }
    assert not any(key[2] == "float8_e4m3fn" and key[3] == 32 for key in table)
    for arch, num_sms in (("sm_100a", 132), ("sm_103a", 148), ("sm_100a", 152)):
        monkeypatch.setattr(
            cake_module, "_device_facts", lambda i, a=arch, n=num_sms: (a, n)
        )
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for page_size in (32, 64):
                for q_len in (1, 2):
                    for has_sinks in (False, True):
                        for max_seq_len in (512, 1024, 4096, 16384):
                            assert not core._cake_mla_auto_offer_ranges(
                                arch,
                                num_sms,
                                dtype,
                                page_size,
                                q_len,
                                has_sinks,
                                max_seq_len,
                            )
        for page_size in (32, 64):
            runner = _staged(_cpu_runner(torch.bfloat16, page_size))
            inputs = _cpu_inputs(torch.bfloat16, 4)
            assert runner.get_valid_tactics(inputs, _bucket(4, 8)) == []
            assert not runner.validate_tactic(inputs, WINNER)
            assert not runner.supports(inputs)
    for arch, num_sms in (("sm_100a", 148), ("sm_103a", 152)):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for page_size in (32, 64):
                for q_len in (1, 2):
                    for max_seq_len in (512, 1024, 4096, 16384):
                        assert not core._cake_mla_auto_offer_ranges(
                            arch, num_sms, dtype, page_size, q_len, True, max_seq_len
                        )
                        if dtype == torch.float8_e4m3fn and page_size == 32:
                            assert not core._cake_mla_auto_offer_ranges(
                                arch,
                                num_sms,
                                dtype,
                                page_size,
                                q_len,
                                False,
                                max_seq_len,
                            )
    assert not built, "an unlisted unit built a Cake program"
    assert not cake_module._FP8_LUTS, "an unlisted unit created the FP8 decode table"


def test_offer_table_lookup(monkeypatch, cake_families, synthetic_offers):
    """A unit is offered only on its exact device, dtype, page size, q_len,
    max_seq_len bucket, domain and batch range, and never with sinks. No GPU
    needed."""
    vquarter = cake_module.DOMAIN_BF16_VQUARTER
    bf16 = torch.bfloat16

    def offered(
        arch="sm_100a",
        num_sms=148,
        dtype=bf16,
        page_size=64,
        domain=vquarter,
        q_len=1,
        has_sinks=False,
        max_seq_len=1024,
        batch_size=4,
    ):
        return core._cake_mla_auto_offered(
            arch,
            num_sms,
            dtype,
            page_size,
            domain,
            q_len,
            has_sinks,
            max_seq_len,
            batch_size,
        )

    assert offered() and offered(batch_size=1) and offered(batch_size=15)
    assert offered(max_seq_len=513)  # bucket 1024
    for change in (
        dict(batch_size=16),
        dict(max_seq_len=512),
        dict(max_seq_len=1025),
        dict(domain=cake_module.DOMAIN_BF16_VHALF),
        dict(arch="sm_103a"),
        dict(num_sms=152),
        dict(arch="sm_103a", num_sms=152),
        dict(dtype=torch.float16),
        dict(dtype=torch.float8_e4m3fn),
        dict(page_size=32),
        dict(q_len=2),
        dict(has_sinks=True),
    ):
        assert not offered(**change), change
    assert core._cake_mla_auto_offer_ranges(
        "sm_100a", 148, bf16, 64, 1, False, 1024
    ) == ((vquarter, 1, 15),)

    # Sinks are never offered, even for a listed unit with sinks.
    with_sinks = {
        ("sm_100a", 148, "bfloat16", 64, 1, True): ((vquarter, 1024, 1, 15),),
    }
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", with_sinks)
    assert not offered(has_sinks=True)
    assert not core._cake_mla_auto_offer_ranges("sm_100a", 148, bf16, 64, 1, True, 1024)


# --- The two selection stages on the real tuner, with first-stage stand-ins ---


class _FakeRunner(TunableRunner):
    """A first-stage candidate that launches nothing."""

    def get_valid_tactics(self, inputs, profile):
        return [-1]

    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[3]

    def __hash__(self):
        return hash(type(self))


class _FakeTrtllmGen(_FakeRunner):
    pass


class _FakeCuteDsl(_FakeRunner):
    pass


class _Stages:
    """Runs the first and second selection stages on a fresh tuner with fake
    first-stage runners and the Cake runner of a simulated 148-SM SM100a device.
    Profiling returns ``times`` (ms) per runner class (a number, or a function
    of the profiled batch), instead of launching."""

    BUCKETS = (1, 2, 4, 8, 16, 32)

    def __init__(self, monkeypatch, tuner):
        self.tuner = tuner
        self.runners = [_FakeTrtllmGen(), _FakeCuteDsl()]
        self.config = core._mla_decode_tuning_config(
            buckets=self.BUCKETS, num_pages=8, profile_seq_len=1024
        )
        self.times = {
            "_FakeTrtllmGen": 2.0,
            "_FakeCuteDsl": 1.0,
            core._CAKE_MLA_RUNNER: 0.5,
        }
        # (profiled class, timed class, batch) of every measurement.
        self.profiled = []
        # (profiled class, batch, the tuner's tie tolerance) of every
        # measurement.
        self.tolerances = []
        # A factor per profiled class of the second stage (incumbent, Cake)
        # on its times, so the second stage can measure the first stage's
        # winner differently from the first stage.
        self.stage_two_scale = {}
        self.ranges = ((1, 15),)

        def profile(tuner, runner, inputs, tactic, tuning_config, **kwargs):
            timed = type(runner).__name__
            if isinstance(runner, core._MlaStageOneWinnerRunner):
                timed = tactic[0]
            batch = inputs[0].size(0)
            self.profiled.append((type(runner).__name__, timed, batch))
            self.tolerances.append((type(runner).__name__, batch, tuner._tie_tolerance))
            time = self.times[timed]
            time = time(batch) if callable(time) else time
            return time * self.stage_two_scale.get(type(runner).__name__, 1.0)

        monkeypatch.setattr(AutoTuner, "_profile_single_kernel", profile)
        monkeypatch.setattr(
            AutoTuner,
            "_prepare_input_tensors_with_batches",
            lambda tuner, inputs, config: [inputs],
        )

    def stage_one(self, inputs, tune=False):
        self.tuner.is_tuning_mode = tune
        try:
            return self.tuner.choose_one(
                core._MLA_DECODE_OP, self.runners, self.config, inputs
            )
        finally:
            self.tuner.is_tuning_mode = False

    def select(
        self, batch_size=4, *, inputs=None, tune=False, selected=None, memo=False
    ):
        """The two stages for a call; ``selected`` skips the first. ``memo``
        passes the dispatcher's memo key (here the input sizes, which with the
        fixed runners and config decide the lookups), so a lookup-only decision
        is memoized and reused."""
        if inputs is None:
            inputs = _cpu_inputs(torch.bfloat16, batch_size)
        if selected is None:
            selected = self.stage_one(inputs, tune)
        self.tuner.is_tuning_mode = tune
        memo_key = None
        if memo and not tune:
            memo_key = ("stages", self.tuner._get_input_sizes(inputs))
        try:
            return core._mla_cake_stage_two(
                self.tuner,
                self.runners,
                self.config,
                inputs,
                selected,
                lambda: _cpu_runner(torch.bfloat16, 64),
                self.ranges,
                memo_key,
            )
        finally:
            self.tuner.is_tuning_mode = False

    def entries(self, op):
        """``{bucket: (runner class, tactic)}`` of the op's winners."""
        return {
            key.nearest_profile[0][0]: (key.runner_class_name, tactic)
            for key, (tactic, _) in self.tuner._winner_cache().items()
            if key.custom_op == op
        }

    def cake_profiled(self, runner=core._CAKE_MLA_RUNNER):
        return sorted(
            batch for profiled, _, batch in self.profiled if profiled == runner
        )

    def incumbent_profiled(self):
        return self.cake_profiled("_MlaStageOneWinnerRunner")


@pytest.fixture
def stages(monkeypatch, fresh_tuner, cake_families, synthetic_offers):
    staged = _Stages(monkeypatch, fresh_tuner)
    staged.families = cake_families
    return staged


# Bucket 16 bounds bucket 8; Cake is not offered there, so it is not timed.
_CAKE_WINS = {bucket: (core._CAKE_MLA_RUNNER, WINNER) for bucket in (1, 2, 4, 8)}


def test_two_stages_keep_first_stage_entries(monkeypatch, stages):
    """The first stage tunes and stores exactly what it stores alone; the
    second stage times and stores entries of its own op only, for the offered
    buckets. No GPU needed."""
    stages.stage_one(_cpu_inputs(torch.bfloat16, 4), tune=True)
    alone = stages.entries(core._MLA_DECODE_OP)
    alone_profiled = list(stages.profiled)
    assert alone == {bucket: ("_FakeCuteDsl", -1) for bucket in stages.BUCKETS}

    stages.tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", stages.tuner)
    stages.profiled.clear()
    runners = list(stages.runners)
    assert stages.select(4, tune=True)[0] is not None
    assert stages.runners == runners
    assert stages.entries(core._MLA_DECODE_OP) == alone
    assert [
        entry
        for entry in stages.profiled
        if entry[0] in ("_FakeTrtllmGen", "_FakeCuteDsl")
    ] == alone_profiled
    assert stages.entries(core._MLA_CAKE_MARGIN_OP) == _CAKE_WINS
    assert stages.cake_profiled() == [1, 2, 4, 8]
    assert stages.incumbent_profiled() == [1, 2, 4, 8]


def test_second_stage_times_only_offered_buckets_under_overrides(monkeypatch, stages):
    """``autotune(tuning_buckets=...)`` replaces the second stage's buckets with
    its own; still only the offered buckets are timed. No GPU needed."""
    monkeypatch.setattr(
        AutoTuner, "_override_tuning_buckets", property(lambda self: stages.BUCKETS)
    )
    stages.select(4, tune=True)
    assert stages.cake_profiled() == [1, 2, 4, 8]
    assert stages.incumbent_profiled() == [1, 2, 4, 8]
    assert stages.entries(core._MLA_CAKE_MARGIN_OP) == _CAKE_WINS


def test_cake_selected_where_it_beat_the_first_stage(stages):
    """A tuned call selects Cake, with the first-stage winner it beat, for an
    offered bucket and the first stage's winner elsewhere. No GPU needed."""
    tuned = stages.select(4, tune=True)
    assert isinstance(tuned[0], core._CakeTrtllmMlaBlackwellRunner)
    for batch_size in (1, 4, 9, 15):
        runner, tactic = stages.select(batch_size)
        assert _name(runner) == "cake" and tactic == WINNER, batch_size
    for batch_size in (16, 20, 31):
        assert stages.select(batch_size) == (stages.runners[1], -1), batch_size


_TIE_MS = core._MLA_CAKE_STAGE_TWO_TIE_MS


def _tie_times(stages, outcome):
    """Sets Cake's stubbed time against the first stage's winner (CuTe DSL)
    and returns the expected second-stage winner entry and decision."""
    first_stage = stages.times["_FakeCuteDsl"]
    stages.times[core._CAKE_MLA_RUNNER] = {
        "tie": first_stage,
        # Within the absolute tie: Cake is re-timed and kept.
        "first_stage_barely_faster": math.nextafter(first_stage, math.inf),
        "first_stage_faster_within_the_tie": first_stage + 0.5 * _TIE_MS,
        # Beyond it, by the smallest margin the tie resolves, and by far.
        "first_stage_faster_beyond_the_tie": first_stage + 1.5 * _TIE_MS,
        "first_stage_faster": first_stage * 1.5,
        "cake_faster": first_stage * 0.5,
    }[outcome]
    if outcome in ("first_stage_faster_beyond_the_tie", "first_stage_faster"):
        return ("_MlaStageOneWinnerRunner", WINNER), (stages.runners[1], -1)
    return (core._CAKE_MLA_RUNNER, WINNER), ("cake", WINNER)


_TIE_OUTCOMES = [
    "tie",
    "first_stage_barely_faster",
    "first_stage_faster_within_the_tie",
    "first_stage_faster_beyond_the_tie",
    "first_stage_faster",
    "cake_faster",
]

# Outcomes in which the tuner re-times Cake against the first stage's winner
# (twice each) before it keeps Cake.
_RETIMED_OUTCOMES = ("first_stage_barely_faster", "first_stage_faster_within_the_tie")


@pytest.mark.parametrize("outcome", _TIE_OUTCOMES)
def test_second_stage_tie_selects_cake(monkeypatch, stages, outcome):
    """An exact tie between Cake and the first stage's winner selects Cake, and
    so does a first-stage winner faster by at most the absolute tie
    (``_MLA_CAKE_STAGE_TWO_TIE_MS``), after the tuner re-times both: Cake is
    stored as the second stage's winner of every offered bucket, selected by
    the tuned call and by lookups, and memoized for lookup-only calls. A
    first-stage winner faster by more than the tie still wins, and a strictly
    faster Cake wins. No GPU needed."""
    winner, expected = _tie_times(stages, outcome)
    assert _decision(stages.select(4, tune=True)) == expected
    timed = 3 if outcome in _RETIMED_OUTCOMES else 1
    assert stages.cake_profiled() == [b for b in (1, 2, 4, 8) for _ in range(timed)]
    assert stages.incumbent_profiled() == stages.cake_profiled()
    assert stages.entries(core._MLA_CAKE_MARGIN_OP) == {
        bucket: winner for bucket in (1, 2, 4, 8)
    }
    lookups = _count_margin_lookups(monkeypatch)
    for batch_size in (1, 4, 9, 15):
        inputs = _cpu_inputs(torch.bfloat16, batch_size)
        assert _decision(stages.select(inputs=inputs)) == expected, batch_size
        before = lookups[0]
        for _ in range(2):
            assert _decision(stages.select(inputs=inputs, memo=True)) == expected
        # Looked up once, then reused from the memo.
        assert lookups[0] - before == 1, batch_size
    assert any(
        entry[0]() is stages.tuner and entry[3] is not None
        for entry in core._CAKE_MLA_STAGE_TWO_MEMO.values()
    ) == (expected[0] == "cake")
    for batch_size in (16, 31):
        assert stages.select(batch_size) == (stages.runners[1], -1), batch_size


@pytest.mark.parametrize("outcome", _TIE_OUTCOMES)
def test_round_up_second_stage_tie_selects_cake(monkeypatch, stages, outcome):
    """Under ``autotune(round_up=True)`` the second stage uses the first
    stage's config; there too an exact tie and a first-stage winner faster by
    at most the absolute tie select Cake, and a first-stage winner faster by
    more wins. No GPU needed."""
    winner, expected = _tie_times(stages, outcome)
    with autotune(False, round_up=True):
        assert _decision(stages.select(4, tune=True)) == expected
        assert 4 in stages.cake_profiled()
        entries = stages.entries(core._MLA_CAKE_MARGIN_OP)
        assert entries and set(entries.values()) == {winner}
        assert _decision(stages.select(4)) == expected


@pytest.mark.parametrize(
    "cake_ratio,cake_kept", [(1.0, True), (1.02, True), (1.1, False), (0.9, True)]
)
def test_second_stage_tie_tolerance_keeps_cake_within_it(stages, cake_ratio, cake_kept):
    """``autotune(tie_tolerance=0.05)``, wider here than the absolute tie,
    widens the second stage's tie to 5 %: Cake is kept against a first-stage
    winner faster by at most 5 % (its re-timing confirms it), and a
    first-stage winner faster by more wins. No GPU needed."""
    stages.times[core._CAKE_MLA_RUNNER] = stages.times["_FakeCuteDsl"] * cake_ratio
    if cake_kept:
        winner, expected = (core._CAKE_MLA_RUNNER, WINNER), ("cake", WINNER)
    else:
        winner = ("_MlaStageOneWinnerRunner", WINNER)
        expected = (stages.runners[1], -1)
    with autotune(False, tie_tolerance=0.05):
        assert _decision(stages.select(4, tune=True)) == expected
        assert stages.entries(core._MLA_CAKE_MARGIN_OP) == {
            bucket: winner for bucket in (1, 2, 4, 8)
        }
        assert _decision(stages.select(4)) == expected
    # The first stage keeps its own winner (CuTe DSL, 2x faster than TRTLLM-GEN).
    assert set(stages.entries(core._MLA_DECODE_OP).values()) == {("_FakeCuteDsl", -1)}


def _first_stage_ms(batch):
    """A first-stage winner time (ms) that grows with the batch, so every
    offered bucket's absolute tie is a different relative tolerance."""
    return 0.010 * batch


@pytest.mark.parametrize(
    "cake_gap_ms,tie_tolerance,kept",
    [
        # One timer step (2.048 us): 20 % of bucket 1, 2.5 % of bucket 8.
        ({1: 0.002048, 2: 0.002048, 4: 0.002048, 8: 0.002048}, None, (1, 2, 4, 8)),
        # One and a half steps: beyond the tie of every bucket.
        ({1: 0.003072, 2: 0.003072, 4: 0.003072, 8: 0.003072}, None, ()),
        # Per bucket: within the tie at 1 and 4, beyond it at 2 and 8.
        ({1: 0.001, 2: 0.003, 4: 0.0025, 8: 0.004}, None, (1, 4)),
        # A wider autotune(tie_tolerance=0.1) holds where it is wider (8: 5 %).
        ({1: 0.001, 2: 0.003, 4: 0.0025, 8: 0.004}, 0.1, (1, 4, 8)),
        # A narrower one leaves each bucket's absolute tie.
        ({1: 0.001, 2: 0.003, 4: 0.0025, 8: 0.004}, 0.001, (1, 4)),
    ],
)
def test_second_stage_keeps_cake_within_each_buckets_absolute_tie(
    stages, cake_gap_ms, tie_tolerance, kept
):
    """The second stage keeps Cake in a bucket unless the first stage's winner
    measures faster by more than ``_MLA_CAKE_STAGE_TWO_TIE_MS``, whatever the
    bucket's time: each offered bucket is timed under the relative tolerance
    that is this tie at the time the first stage measured for its winner
    there, or under a wider ``autotune(tie_tolerance=...)``. Every offered
    bucket is timed in its own pass, Cake against the first-stage winner,
    once or (when Cake is within the tolerance) three times; no other bucket
    is timed, the first stage's entries are unchanged, the tolerance stack is
    restored, and lookups follow the stored winners. No GPU needed."""
    stages.times["_FakeCuteDsl"] = _first_stage_ms
    stages.times["_FakeTrtllmGen"] = lambda batch: 2 * _first_stage_ms(batch)
    stages.times[core._CAKE_MLA_RUNNER] = lambda batch: (
        _first_stage_ms(batch) + cake_gap_ms.get(batch, 0.0)
    )
    context = (
        contextlib.nullcontext()
        if tie_tolerance is None
        else autotune(False, tie_tolerance=tie_tolerance)
    )
    with context:
        stack = list(stages.tuner._get_override_stack())
        # Bucket 4 is timed in a pass before the last; the last pass's lookup
        # still returns the call's selection.
        assert _decision(stages.select(4, tune=True)) == (
            ("cake", WINNER) if 4 in kept else (stages.runners[1], -1)
        )
        assert stages.tuner._get_override_stack() == stack
    assert stages.entries(core._MLA_DECODE_OP) == {
        bucket: ("_FakeCuteDsl", -1) for bucket in stages.BUCKETS
    }
    assert stages.entries(core._MLA_CAKE_MARGIN_OP) == {
        bucket: (
            (core._CAKE_MLA_RUNNER if bucket in kept else "_MlaStageOneWinnerRunner"),
            WINNER,
        )
        for bucket in (1, 2, 4, 8)
    }
    margin = ("_MlaStageOneWinnerRunner", core._CAKE_MLA_RUNNER)
    for bucket in (1, 2, 4, 8):
        tie = max(tie_tolerance or 0.0, _TIE_MS / _first_stage_ms(bucket))
        timed = [
            tolerance
            for name, batch, tolerance in stages.tolerances
            if name in margin and batch == bucket
        ]
        within = _first_stage_ms(bucket) + cake_gap_ms[bucket] <= (
            _first_stage_ms(bucket) * (1 + tie)
        )
        assert len(timed) == (6 if within else 2), bucket
        assert timed == pytest.approx([tie] * len(timed)), bucket
    assert {batch for name, batch, _ in stages.tolerances if name in margin} == {
        1,
        2,
        4,
        8,
    }
    for batch_size in (1, 2, 4, 9, 15):
        bucket = 1 << (batch_size.bit_length() - 1)
        expected = ("cake", WINNER) if bucket in kept else (stages.runners[1], -1)
        assert _decision(stages.select(batch_size)) == expected, batch_size


@pytest.mark.parametrize(
    "scale,cake_scaled,gap_ties,kept",
    [
        # The second stage measures the first stage's winner 30 % slower than
        # the first stage did: the tolerance spans 1.3 ties there.
        (1.3, False, 1.2, True),
        (1.3, False, 1.4, False),
        # 20 % faster: it spans 0.8 ties.
        (0.8, False, 0.7, True),
        (0.8, False, 0.9, False),
        # Both candidates measured 30 % slower: one tie at the first stage's
        # scale.
        (1.3, True, 0.9, True),
        (1.3, True, 1.1, False),
    ],
)
def test_second_stage_tie_follows_the_second_stages_time_of_the_winner(
    stages, scale, cake_scaled, gap_ties, kept
):
    """The tolerance is relative: ``_MLA_CAKE_STAGE_TWO_TIE_MS`` divided by
    the first stage's time of its winner (``I1``), which the tuner applies to
    its second-stage measurement of that winner (``I2``). Cake is therefore
    kept within ``_MLA_CAKE_STAGE_TWO_TIE_MS * I2 / I1`` of ``I2``; where both
    candidates' second-stage times scale alike, that is the tie at the first
    stage's scale. No GPU needed."""
    cake = core._CAKE_MLA_RUNNER
    stages.times["_FakeCuteDsl"] = _first_stage_ms
    stages.times["_FakeTrtllmGen"] = lambda batch: 2 * _first_stage_ms(batch)
    stages.stage_two_scale = {"_MlaStageOneWinnerRunner": scale}
    if cake_scaled:
        stages.stage_two_scale[cake] = scale
        stages.times[cake] = lambda batch: _first_stage_ms(batch) + gap_ties * _TIE_MS
    else:
        stages.times[cake] = lambda batch: (
            scale * _first_stage_ms(batch) + gap_ties * _TIE_MS
        )
    expected = ("cake", WINNER) if kept else (stages.runners[1], -1)
    assert _decision(stages.select(4, tune=True)) == expected
    winner = cake if kept else "_MlaStageOneWinnerRunner"
    assert stages.entries(core._MLA_CAKE_MARGIN_OP) == {
        bucket: (winner, WINNER) for bucket in (1, 2, 4, 8)
    }
    margin = ("_MlaStageOneWinnerRunner", cake)
    for bucket in (1, 2, 4, 8):
        timed = [
            tolerance
            for name, batch, tolerance in stages.tolerances
            if name in margin and batch == bucket
        ]
        assert len(timed) == (6 if kept else 2), bucket
        assert timed == pytest.approx(
            [_TIE_MS / _first_stage_ms(bucket)] * len(timed)
        ), bucket


def test_second_stage_without_a_first_stage_time_keeps_the_tuners_tolerance(
    monkeypatch, stages
):
    """Without the first stage's time of its winner (no selection record) the
    second stage times every bucket in one pass under the tuner's own
    tolerance: an exact tie keeps Cake, a first-stage winner faster by any
    margin wins. No GPU needed."""
    monkeypatch.setattr(
        core._MlaStageOneWinnerRunner, "profile_time", lambda self, profile: None
    )
    calls = _spy_choose_one(monkeypatch)
    # Within the absolute tie, which needs the first stage's time.
    _tie_times(stages, "first_stage_faster_within_the_tie")
    assert _decision(stages.select(4, tune=True)) == (stages.runners[1], -1)
    assert [op for op, _, _ in calls].count(core._MLA_CAKE_MARGIN_OP) == 1
    assert {tolerance for _, _, tolerance in stages.tolerances} == {0.0}
    assert stages.entries(core._MLA_CAKE_MARGIN_OP) == {
        bucket: ("_MlaStageOneWinnerRunner", WINNER) for bucket in (1, 2, 4, 8)
    }


def test_first_stage_time_reads_the_winners_selection_record(
    monkeypatch, stages, tmp_path
):
    """``_MlaStageOneWinnerRunner.profile_time`` is the time the first stage
    measured for its winner of a profile: from this process's tuning, and from
    a cache file loaded into another tuner; ``None`` without a winner. No GPU
    needed."""
    stages.times["_FakeCuteDsl"] = _first_stage_ms
    inputs = _cpu_inputs(torch.bfloat16, 4)
    profiles = stages.tuner._generate_optimization_profiles(stages.config, inputs)
    incumbent = core._MlaStageOneWinnerRunner(stages.runners, stages.config, inputs)
    assert all(incumbent.profile_time(profile) is None for profile in profiles)
    stages.stage_one(inputs, tune=True)
    expected = {
        profile.get_opt_shapes()[0][0]: _first_stage_ms(profile.get_opt_shapes()[0][0])
        for profile in profiles
    }
    measured = {
        profile.get_opt_shapes()[0][0]: incumbent.profile_time(profile)
        for profile in profiles
    }
    assert measured == pytest.approx(expected)

    path = str(tmp_path / "configs.json")
    stages.tuner.save_configs(path)
    loaded = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", loaded)
    assert loaded.load_configs(path)
    measured = {
        profile.get_opt_shapes()[0][0]: incumbent.profile_time(profile)
        for profile in profiles
    }
    assert measured == pytest.approx(expected)


def test_second_stage_lists_cake_first_outside_autotune_v2(fresh_tuner):
    """Cake is the second stage's first candidate, so the tuner's strict
    minimum keeps it at a tie; inside an ``autotune_v2`` context, where the
    first candidate also races the default tactic (the first stage's
    selection), the first-stage winner stays first. No GPU needed."""
    cake = _cpu_runner(torch.bfloat16, 64)
    incumbent = core._MlaStageOneWinnerRunner(
        [_FakeTrtllmGen(), _FakeCuteDsl()], None, [], cake
    )
    order = core._mla_cake_stage_two_runners
    assert order(fresh_tuner, incumbent, cake) == [cake, incumbent]
    with autotune_v2(mode="replay", persistent_cache=False):
        assert fresh_tuner._in_v2_context
        assert order(fresh_tuner, incumbent, cake) == [incumbent, cake]
    assert order(fresh_tuner, incumbent, cake) == [cake, incumbent]


def _record_policies(monkeypatch):
    """Records ``(profiled runner class, L2 policy, cold chain launches)`` of
    every measurement of the stubbed profiling of ``_Stages``."""
    recorded = []
    profile = AutoTuner._profile_single_kernel

    def record(tuner, runner, inputs, tactic, tuning_config, **kwargs):
        recorded.append(
            (
                type(runner).__name__,
                tuning_config.l2_cache_policy,
                tuning_config.cold_chain_launches,
            )
        )
        return profile(tuner, runner, inputs, tactic, tuning_config, **kwargs)

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", record)
    return recorded


def _margin_policies(tuner):
    """The stored L2 policies of the second stage's winners."""
    return {
        policy[policy.index("l2_cache_policy") + 1]
        for (_, key), policy in tuner._profiling_cache_policies.items()
        if key.custom_op == core._MLA_CAKE_MARGIN_OP
    }


@pytest.mark.parametrize("round_up", [False, True])
def test_second_stage_times_a_single_cold_replay(monkeypatch, stages, round_up):
    """Both stages time a single cold replay per sample: the second stage's
    config differs from the first stage's only in its buckets, and both ops
    store the ``cold`` policy (also under ``autotune(round_up=True)``, where
    the second stage uses the first stage's config). No GPU needed."""
    first = stages.config
    second = core._mla_cake_margin_tuning_config(first, stages.ranges)
    assert (second.cold_chain_launches, second.profile_arena_input_indices) == (1, ())
    assert replace(second, dynamic_tensor_specs=first.dynamic_tensor_specs) == first
    assert second.l2_cache_policy == first.l2_cache_policy == "cold"

    recorded = _record_policies(monkeypatch)
    context = autotune(False, round_up=True) if round_up else contextlib.nullcontext()
    with context:
        assert _decision(stages.select(4, tune=True)) == ("cake", WINNER)
    assert {name for name, _, _ in recorded} == {
        "_FakeTrtllmGen",
        "_FakeCuteDsl",
        "_MlaStageOneWinnerRunner",
        core._CAKE_MLA_RUNNER,
    }
    assert {(policy, chain) for _, policy, chain in recorded} == {("cold", 1)}
    stored = {
        (key.custom_op, policy[policy.index("l2_cache_policy") + 1])
        for (_, key), policy in stages.tuner._profiling_cache_policies.items()
    }
    assert stored == {
        (core._MLA_DECODE_OP, "cold"),
        (core._MLA_CAKE_MARGIN_OP, "cold"),
    }
    assert _margin_policies(stages.tuner) == {"cold"}


def test_untuned_call_never_selects_cake(stages):
    """Without tuning the call runs the first stage's default; a Cake entry of
    the second stage without a first-stage winner is a miss. No GPU needed."""
    assert stages.select(4) == (stages.runners[0], -1)
    inputs = _cpu_inputs(torch.bfloat16, 4)
    cake = _cpu_runner(torch.bfloat16, 64)
    key = ProfilingCacheKey(
        custom_op=core._MLA_CAKE_MARGIN_OP,
        runner_class_name=core._CAKE_MLA_RUNNER,
        runner_hash=hash(cake),
        nearest_profile=AutoTuner._find_nearest_profile(
            stages.tuner._get_input_sizes(inputs), stages.config
        ),
        extras=cake.get_cache_key_extras(inputs),
    )
    stages.tuner.profiling_cache[key] = (WINNER, None)
    assert stages.select(4, inputs=inputs) == (stages.runners[0], -1)
    assert not stages.profiled


def test_cake_profiled_only_where_the_first_stage_has_a_winner(stages):
    """S1: the second stage never profiles Cake for a profile without a
    first-stage winner, where Cake would win unopposed. No GPU needed."""
    selected = (stages.runners[0], -1)
    assert stages.select(4, tune=True, selected=selected) == selected
    assert not stages.profiled and not stages.entries(core._MLA_CAKE_MARGIN_OP)

    stages.stage_one(_cpu_inputs(torch.bfloat16, 4), tune=True)
    winners = stages.tuner._winner_cache()
    (bucket4,) = [
        key
        for key in winners
        if key.custom_op == core._MLA_DECODE_OP and key.nearest_profile[0][0] == 4
    ]
    del winners[bucket4]
    stages.profiled.clear()
    stages.select(2, tune=True, selected=(stages.runners[1], -1))
    assert stages.cake_profiled() == [1, 2, 8]
    entries = stages.entries(core._MLA_CAKE_MARGIN_OP)
    assert 4 not in entries and entries[8] == (core._CAKE_MLA_RUNNER, WINNER)


def test_cake_choice_follows_the_first_stage_winner(stages):
    """H1: a Cake entry names the first-stage winner it beat. Once the first
    stage selects another winner the entry is a miss (the call runs the new
    winner) until the second stage measures Cake against it. No GPU needed."""
    stages.select(4, tune=True)
    assert stages.select(4)[1] == WINNER

    winners = stages.tuner._winner_cache()
    for key in [key for key in winners if key.custom_op == core._MLA_DECODE_OP]:
        del winners[key]
    stages.times["_FakeTrtllmGen"] = 0.8
    stages.stage_one(_cpu_inputs(torch.bfloat16, 4), tune=True)
    trtllm_gen = ("_FakeTrtllmGen", -1)
    assert set(stages.entries(core._MLA_DECODE_OP).values()) == {trtllm_gen}
    assert stages.select(4) == (stages.runners[0], -1)

    stages.profiled.clear()
    stages.select(4, tune=True)
    assert stages.cake_profiled() == [1, 2, 4, 8]
    runner, tactic = stages.select(4)
    assert _name(runner) == "cake" and tactic == trtllm_gen


@pytest.mark.parametrize(
    "miss",
    [
        "validate_miss",
        "capture_without_family",
        "unavailable_family",
        "noncontiguous_out",
        "outside_offered_batches",
    ],
)
def test_cake_miss_runs_the_first_stage_winner(
    monkeypatch, stages, cake_families, miss
):
    """Every Cake miss runs the first stage's tuned winner (here CuTe DSL, the
    second candidate), never the first candidate's default. No GPU needed."""
    built, state = cake_families
    stages.select(4, tune=True)
    assert _name(stages.select(4)[0]) == "cake"
    first_stage = (stages.runners[1], -1)
    inputs = _cpu_inputs(torch.bfloat16, 4)
    if miss == "validate_miss":
        monkeypatch.setattr(
            core._CakeTrtllmMlaBlackwellRunner,
            "validate_tactic",
            lambda self, inputs, tactic: False,
        )
    elif miss == "capture_without_family":
        monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
        state["capturing"] = True
    elif miss == "unavailable_family":
        family = cake_module.launch_family(
            cake_module.DOMAIN_BF16_VQUARTER, "sm_100a", 148, 4
        )
        monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", {family})
    elif miss == "noncontiguous_out":
        out = torch.empty(4, 1, NUM_HEADS, 2 * LATENT, dtype=torch.bfloat16)
        inputs[3] = out[..., :LATENT]
    else:
        inputs = _cpu_inputs(torch.bfloat16, 20)
    built_before = list(built)
    assert stages.select(inputs=inputs) == first_stage
    assert built == built_before
    # A second-stage miss is no untuned shape: no fallback warning.
    assert not any(
        op == core._MLA_CAKE_MARGIN_OP for op, _ in stages.tuner._logged_cache_miss_oor
    )


def test_second_stage_only_looks_up_during_capture(monkeypatch, stages, cake_families):
    """Under tuning, a stream capture never profiles the second stage. No GPU
    needed."""
    _, state = cake_families
    selected = (stages.runners[1], -1)
    state["capturing"] = True
    assert stages.select(4, tune=True, selected=selected) == selected
    assert not stages.profiled
    state["capturing"] = False
    stages.select(4, tune=True)
    stages.profiled.clear()
    state["capturing"] = True
    runner, tactic = stages.select(4, tune=True, selected=selected)
    assert _name(runner) == "cake" and tactic == WINNER and not stages.profiled


def test_second_stage_is_outside_replay_diagnostics(monkeypatch, stages):
    """Under strict ``autotune_v2`` replay the first stage records the call; the
    second stage records nothing, and its miss runs the first stage's winner
    instead of raising. No GPU needed."""
    stages.select(4, tune=True)
    local = stages.tuner._v2_local
    report = ReplayCoverage()
    monkeypatch.setattr(local, "replay_report", report)
    monkeypatch.setattr(local, "strict_replay", True)
    runner, tactic = stages.select(4)
    assert _name(runner) == "cake" and tactic == WINNER

    winners = stages.tuner._winner_cache()
    for key in [key for key in winners if key.custom_op == core._MLA_CAKE_MARGIN_OP]:
        del winners[key]
    assert stages.select(4) == (stages.runners[1], -1)
    assert [record["op"] for record in report.records] == [core._MLA_DECODE_OP] * 2
    assert (report.hits, report.misses) == (2, 0)
    assert (local.replay_report, local.strict_replay) == (report, True)


def test_failed_second_stage_tuning_runs_the_first_stage_winner(monkeypatch, stages):
    """An error while tuning the second stage (here the FP8 decode table)
    runs the first stage's winner instead of failing the call. No GPU
    needed."""

    def fail(device):
        raise RuntimeError("no FP8 decode table")

    monkeypatch.setattr(cake_module, "_fp8_lut", fail)
    assert stages.select(4, tune=True) == (stages.runners[1], -1)
    assert not stages.cake_profiled()


# --- The memoized second stage (the eager host path of a tuned call) ---


def _count_margin_lookups(monkeypatch):
    """Counts the second stage's cache lookups."""
    counts = [0]
    original = AutoTuner.search_cache

    def search_cache(self, op, *args, **kwargs):
        if op == core._MLA_CAKE_MARGIN_OP:
            counts[0] += 1
        return original(self, op, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "search_cache", search_cache)
    return counts


def _decision(result):
    """``(runner, tactic)`` with a Cake runner (one per call) named ``cake``."""
    runner, tactic = result
    if isinstance(runner, core._CakeTrtllmMlaBlackwellRunner):
        return "cake", tactic
    return runner, tactic


@pytest.mark.parametrize(
    "scenario",
    ["cake_won", "cake_tied", "first_stage_kept", "untuned", "other_first_stage"],
)
def test_memoized_second_stage_matches_the_lookup(monkeypatch, stages, scenario):
    """With the dispatcher's memo key the second stage looks up once per request
    sizes and first-stage selection, then returns the memoized decision without
    a lookup: the (runner, tactic) the lookup returns, for every batch of the
    offered buckets, and the first stage's selection without a lookup outside
    them. No GPU needed."""
    if scenario == "cake_tied":
        stages.times[core._CAKE_MLA_RUNNER] = stages.times["_FakeCuteDsl"]
    elif scenario == "first_stage_kept":
        stages.times[core._CAKE_MLA_RUNNER] = 1.5 * stages.times["_FakeCuteDsl"]
    if scenario != "untuned":
        stages.select(4, tune=True)
    selected = (stages.runners[0], -1) if scenario == "other_first_stage" else None
    expected = {
        "cake_won": ("cake", WINNER),
        "cake_tied": ("cake", WINNER),
        "first_stage_kept": (stages.runners[1], -1),
        "untuned": (stages.runners[0], -1),
        "other_first_stage": (stages.runners[0], -1),
    }[scenario]
    lookups = _count_margin_lookups(monkeypatch)
    for batch_size in (1, 4, 9, 15, 16, 31):
        inputs = _cpu_inputs(torch.bfloat16, batch_size)
        offered = batch_size <= 15
        looked_up = _decision(stages.select(inputs=inputs, selected=selected))
        before = lookups[0]
        first = _decision(stages.select(inputs=inputs, selected=selected, memo=True))
        assert lookups[0] - before == int(offered), (scenario, batch_size)
        again = _decision(stages.select(inputs=inputs, selected=selected, memo=True))
        assert lookups[0] - before == int(offered), (scenario, batch_size)
        assert first == again == looked_up, (scenario, batch_size)
        if offered:
            assert looked_up == expected, (scenario, batch_size)
        elif scenario != "untuned":
            assert looked_up == (selected or (stages.runners[1], -1))


def test_memoized_cake_decision_under_capture(monkeypatch, stages):
    """A memoized Cake decision is reused during stream capture while its
    families are loaded; a capture without them looks up, misses, runs the
    first stage's winner and memoizes nothing, so the next eager call loads
    them and runs Cake again. No GPU needed."""
    _, state = stages.families
    stages.select(4, tune=True)
    lookups = _count_margin_lookups(monkeypatch)
    assert _decision(stages.select(4, memo=True)) == ("cake", WINNER)
    assert lookups[0] == 1
    state["capturing"] = True
    assert _decision(stages.select(4, memo=True)) == ("cake", WINNER)
    assert lookups[0] == 1
    assert _decision(stages.select(4)) == ("cake", WINNER)
    assert lookups[0] == 2

    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
    for _ in range(2):
        assert stages.select(4, memo=True) == (stages.runners[1], -1)
    assert lookups[0] == 4
    state["capturing"] = False
    assert _decision(stages.select(4, memo=True)) == ("cake", WINNER)
    assert lookups[0] == 5
    assert _decision(stages.select(4, memo=True)) == ("cake", WINNER)
    assert lookups[0] == 5


def test_second_stage_memo_skipped_under_replay_diagnostics(monkeypatch, stages):
    """Under strict ``autotune_v2`` replay every call looks up (and records
    nothing in the second stage), as without the memo. No GPU needed."""
    stages.select(4, tune=True)
    lookups = _count_margin_lookups(monkeypatch)
    local = stages.tuner._v2_local
    report = ReplayCoverage()
    monkeypatch.setattr(local, "replay_report", report)
    monkeypatch.setattr(local, "strict_replay", True)
    for _ in range(3):
        assert _decision(stages.select(4, memo=True)) == ("cake", WINNER)
    assert lookups[0] == 3
    assert not any(
        entry[0]() is stages.tuner for entry in core._CAKE_MLA_STAGE_TWO_MEMO.values()
    )
    assert [record["op"] for record in report.records] == [core._MLA_DECODE_OP] * 3


@pytest.mark.parametrize(
    "change",
    [
        "second_stage_entries",
        "first_stage_winner",
        "unavailable_family",
        "new_tuner",
        "unrelated_entry",
        "offer_table",
        "splitkv_range",
        "splitkv_range_in_place",
    ],
)
def test_memoized_decision_follows_tuner_and_family_changes(
    monkeypatch, stages, change
):
    """A memoized decision is dropped when any tuner's winners or loaded
    configs change, the tuner or the offer table is replaced, a split-KV
    device range changes (replaced or in place), or a family it launches
    becomes unavailable: the next call looks up and returns what the lookup
    returns. No GPU needed."""
    stages.select(4, tune=True)
    assert _decision(stages.select(4, memo=True)) == ("cake", WINNER)
    lookups = _count_margin_lookups(monkeypatch)
    expected = ("cake", WINNER)
    if change == "second_stage_entries":
        winners = stages.tuner._winner_cache()
        for key in [
            key for key in winners if key.custom_op == core._MLA_CAKE_MARGIN_OP
        ]:
            del winners[key]
        expected = (stages.runners[1], -1)
    elif change == "first_stage_winner":
        winners = stages.tuner._winner_cache()
        for key in [key for key in winners if key.custom_op == core._MLA_DECODE_OP]:
            del winners[key]
        stages.times["_FakeTrtllmGen"] = 0.8
        stages.stage_one(_cpu_inputs(torch.bfloat16, 4), tune=True)
        # The Cake entry names CuTe DSL, which the first stage no longer selects.
        expected = (stages.runners[0], -1)
    elif change == "unavailable_family":
        family = cake_module.launch_family(
            cake_module.DOMAIN_BF16_VQUARTER, "sm_100a", 148, 4
        )
        monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", {family})
        expected = (stages.runners[1], -1)
    elif change == "new_tuner":
        tuner = AutoTuner()
        # The same winners, copied without advancing the winner-source
        # generation: only the tuner differs.
        dict.update(tuner.profiling_cache, stages.tuner.profiling_cache)
        stages.tuner = tuner
        monkeypatch.setattr(AutoTuner, "_instance", tuner)
    elif change == "unrelated_entry":
        _hold_cake_entry(stages.tuner)
    elif change == "splitkv_range":
        # New ranges in a new object (max_seq_len 1024 keeps its domain).
        monkeypatch.setattr(
            cake_module,
            "_SPLITKV_RANGE",
            {**cake_module._SPLITKV_RANGE, ("sm_100a", 148): (4097, 18)},
        )
    elif change == "splitkv_range_in_place":
        # The same ranges in a new object keep the memoized decision.
        monkeypatch.setattr(
            cake_module, "_SPLITKV_RANGE", dict(cake_module._SPLITKV_RANGE)
        )
        assert _decision(stages.select(4, memo=True)) == expected
        assert lookups[0] == 0
        cake_module._SPLITKV_RANGE[("sm_100a", 148)] = (4097, 18)
    else:
        monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", dict(_SYNTHETIC_OFFERS))
    assert _decision(stages.select(4, memo=True)) == expected
    assert lookups[0] == 1
    assert _decision(stages.select(4)) == expected
    assert lookups[0] == 2
    # A decision no readiness check took part in is memoized again.
    if change != "unavailable_family":
        assert _decision(stages.select(4, memo=True)) == expected
        assert lookups[0] == 2


def test_auto_memo_follows_the_tuner_state(
    monkeypatch, fresh_tuner, cake_families, synthetic_offers
):
    """``_CAKE_MLA_AUTO_MEMO`` (the decisions of both stages) stores a decision
    only while the offer table is the one the call read before both stages,
    for a first-stage runner of the call, and returns it only for the tuner,
    offer table, split-KV device range values and state it was stored under; a Cake decision only while its
    families are loaded with the same FP8 decode table, and as the first
    stage's selection for tensors that are not all contiguous. The state
    holds the Cake switch, and a decision is not stored when the switch
    changed during the call. The state is ``None`` (nothing is memoized or
    reused) while tuning, with a managed store, inside an ``autotune_v2``
    context, under replay diagnostics, with a measurement policy, and while
    ``skip_ops`` names either stage's op. No GPU needed."""
    tuner = fresh_tuner
    query, block_tables, seq_lens, out = _cpu_inputs(torch.bfloat16, 4)
    kv_cache = torch.empty(8, 64, QK_DIM, dtype=torch.bfloat16)
    runners = [_FakeTrtllmGen(), _FakeCuteDsl()]
    selected = (runners[1], -1)
    state = core._mla_cake_memo_state(tuner)
    assert state is not None
    family = _family(cake_module.DOMAIN_BF16_VQUARTER, 4)
    lut = torch.empty(1)
    cake = (WINNER, frozenset({family}), None, lut)
    put = core._mla_cake_auto_memo_put
    offers = core._CAKE_MLA_AUTO_OFFERS
    put(tuner, ("first",), state, offers, runners, selected, None)
    put(tuner, ("cake",), state, offers, runners, selected, cake)
    # The offer table was replaced during the call.
    put(tuner, ("moved",), state, dict(offers), runners, selected, None)
    put(tuner, ("stranger",), state, offers, runners, (_FakeCuteDsl(), -1), None)
    # The Cake switch was turned off during the call (the gate read it again).
    with monkeypatch.context() as patch:
        patch.setenv(DISABLE_ENV, "1")
        put(tuner, ("toggled",), state, offers, runners, selected, None)
    assert set(core._CAKE_MLA_AUTO_MEMO) == {("first",), ("cake",)}

    def get(key, tensor=out, owner=tuner):
        # As the dispatcher: no memo without a state.
        current = core._mla_cake_memo_state(owner)
        if current is None:
            return None
        return core._mla_cake_auto_memo_get(
            owner, key, current, query, kv_cache, block_tables, seq_lens, tensor
        )

    first_stage = (1, -1, None)
    assert get(("first",)) == first_stage
    assert get(("cake",)) is None  # its family is not loaded
    core._CAKE_MLA_LOADED_FAMILIES.add(family)
    cake_module._FP8_LUTS[query.device] = lut
    assert get(("cake",)) == (1, -1, cake)
    strided = torch.empty(4, 1, NUM_HEADS, 2 * LATENT, dtype=torch.bfloat16)
    assert get(("cake",), strided[..., :LATENT]) == first_stage
    cake_module._FP8_LUTS[query.device] = torch.empty(1)
    assert get(("cake",)) is None
    cake_module._FP8_LUTS[query.device] = lut
    core._CAKE_MLA_UNAVAILABLE_FAMILIES.add(family)
    assert get(("cake",)) is None
    core._CAKE_MLA_UNAVAILABLE_FAMILIES.clear()
    assert get(("cake",)) == (1, -1, cake)

    local = tuner._v2_local
    for target, name, value in (
        (tuner, "is_tuning_mode", True),
        (tuner, "_managed_cache", object()),
        (local, "active", True),
        (local, "replay_report", ReplayCoverage()),
        (local, "strict_replay", True),
        (local, "measure", object()),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(target, name, value)
            assert core._mla_cake_memo_state(tuner) is None, name
            assert get(("first",)) is None, name
    for op in (core._MLA_DECODE_OP, core._MLA_CAKE_MARGIN_OP):
        with autotune(False, skip_ops=op):
            assert core._mla_cake_memo_state(tuner) is None, op
            assert get(("first",)) is None, op
    with autotune(False, skip_ops="another_op"):
        assert get(("first",)) == first_stage
    with autotune(False, tuning_buckets=(1, 2, 4)):
        assert get(("first",)) is None
    bundled = os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    with monkeypatch.context() as patch:
        patch.setenv(
            "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0" if bundled == "1" else "1"
        )
        assert get(("first",)) is None
    assert get(("first",)) == first_stage
    # The state holds the Cake switch, which decides whether Cake joins.
    with monkeypatch.context() as patch:
        patch.setenv(DISABLE_ENV, "1")
        assert get(("first",)) is None
        assert get(("cake",)) is None
    assert get(("first",)) == first_stage
    assert get(("first",), owner=AutoTuner()) is None
    with monkeypatch.context() as patch:
        patch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", dict(_SYNTHETIC_OFFERS))
        assert get(("first",)) is None
    assert get(("first",)) == first_stage
    # The split-KV device ranges are compared by value: equal ranges in a new
    # object keep the decision, a changed range (also in place) drops it.
    with monkeypatch.context() as patch:
        ranges = dict(cake_module._SPLITKV_RANGE)
        patch.setattr(cake_module, "_SPLITKV_RANGE", ranges)
        assert get(("first",)) == first_stage
        ranges[("sm_100a", 148)] = (4097, 18)
        assert get(("first",)) is None
    assert get(("first",)) == first_stage
    _hold_cake_entry(tuner)  # any change of any tuner's winners
    assert get(("first",)) is None
    assert get(("cake",)) is None


def test_offered_batch_ranges_follow_the_offer_table(monkeypatch, cake_families):
    """The memoized offered batch ranges equal the offer table's ranges for
    every committed unit, follow a replaced table, and are empty with sinks,
    for unlisted units and on devices other than SM100a/SM103a. No GPU
    needed."""
    devices = {(arch, num_sms) for arch, num_sms, *_ in core._CAKE_MLA_AUTO_OFFERS}
    for arch, num_sms in sorted(devices):
        monkeypatch.setattr(
            cake_module, "_device_facts", lambda index, facts=(arch, num_sms): facts
        )
        for key, units in core._CAKE_MLA_AUTO_OFFERS.items():
            if key[:2] != (arch, num_sms):
                continue
            _, _, dtype_name, page_size, q_len, _ = key
            dtype = getattr(torch, dtype_name)
            for bucket in {unit[1] for unit in units}:
                for max_seq_len in (bucket // 2 + 1, bucket):
                    expected = tuple(
                        (first, last)
                        for _, first, last in core._cake_mla_auto_offer_ranges(
                            arch, num_sms, dtype, page_size, q_len, False, max_seq_len
                        )
                    )
                    assert expected
                    for _ in range(2):
                        # Past the per-configuration memo: the second
                        # lookup reads the table memo.
                        core._CAKE_MLA_RANGES_MEMO[0] = None
                        assert (
                            core._cake_mla_auto_batch_ranges(
                                0, dtype, page_size, q_len, False, max_seq_len
                            )
                            == expected
                        ), (key, max_seq_len)
                    assert not core._cake_mla_auto_batch_ranges(
                        0, dtype, page_size, q_len, True, max_seq_len
                    )

    bf16 = torch.bfloat16
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: ("sm_100a", 148))
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", dict(_SYNTHETIC_OFFERS))
    assert core._cake_mla_auto_batch_ranges(0, bf16, 64, 1, False, 1024) == ((1, 15),)
    assert core._cake_mla_auto_batch_ranges(0, bf16, 64, 1, False, 600) == ((1, 15),)
    assert not core._cake_mla_auto_batch_ranges(0, bf16, 64, 1, False, 2048)
    assert not core._cake_mla_auto_batch_ranges(0, bf16, 32, 1, False, 1024)
    assert not core._cake_mla_auto_batch_ranges(0, bf16, 64, 2, False, 1024)
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {})
    assert not core._cake_mla_auto_batch_ranges(0, bf16, 64, 1, False, 1024)

    queries = []

    def not_blackwell(index):
        queries.append(index)
        raise RuntimeError("requires compute capability 10.0 or 10.3")

    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", dict(_SYNTHETIC_OFFERS))
    monkeypatch.setattr(cake_module, "_device_facts", not_blackwell)
    for _ in range(3):
        # Past the per-configuration memo, so each call reaches the device.
        core._CAKE_MLA_RANGES_MEMO[0] = None
        assert not core._cake_mla_auto_batch_ranges(0, bf16, 64, 1, False, 1024)
    # A device without Cake programs is queried once, not on every call.
    assert queries == [0]


def test_offered_batch_ranges_are_memoized_per_configuration(
    monkeypatch, cake_families
):
    """Each request configuration's offered batch ranges are looked up once,
    also when configurations alternate, while the offer table and the Cake
    module's ``_device_facts`` are the same objects; a replaced table and a
    replaced ``_device_facts`` (as tests patch it) are looked up again. No GPU
    needed."""
    lookups = []
    original = core._cake_mla_lookup_batch_ranges

    def lookup(*args):
        lookups.append(args)
        return original(*args)

    monkeypatch.setattr(core, "_cake_mla_lookup_batch_ranges", lookup)
    bf16 = torch.bfloat16

    def ranges(max_seq_len=1024):
        return core._cake_mla_auto_batch_ranges(0, bf16, 64, 1, False, max_seq_len)

    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", dict(_SYNTHETIC_OFFERS))
    assert [ranges() for _ in range(3)] == [((1, 15),)] * 3
    assert len(lookups) == 1
    # Alternating configurations each hit after their first lookup.
    assert [ranges(2048), ranges(), ranges(2048), ranges()] == [
        (),
        ((1, 15),),
        (),
        ((1, 15),),
    ]
    assert len(lookups) == 2
    key = ("sm_100a", 148, "bfloat16", 64, 1, False)
    unit = (cake_module.DOMAIN_BF16_VQUARTER, 1024, 2, 3)
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {key: (unit,)})
    assert [ranges() for _ in range(2)] == [((2, 3),)] * 2
    assert len(lookups) == 3
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: ("sm_103a", 152))
    assert [ranges() for _ in range(2)] == [()] * 2
    assert len(lookups) == 4
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: ("sm_100a", 148))
    assert ranges() == ((2, 3),)
    assert len(lookups) == 5


def test_cake_families_memo_follows_the_offer_table(
    monkeypatch, cake_families, offer_everything
):
    """The memoized families of a batch equal their derivation from the
    domain, the offer table and the routes, for batches on both sides of every
    program boundary and the row cap, and follow a replaced table. No GPU
    needed."""
    for dtype, page_size in ((torch.bfloat16, 32), (torch.bfloat16, 64)):
        runner = _cpu_runner(dtype, page_size)
        for batch_size in (1, 4, 18, 19, 37, 38, 64, 74, 75, 128, 300, 512, 513):
            inputs = _cpu_inputs(dtype, batch_size)
            expected = runner._launched_families(
                dtype, 1, NUM_HEADS, QK_DIM, 32, batch_size, "sm_100a", 148
            )
            for _ in range(2):
                assert runner._families(inputs, (batch_size,)) == expected, (
                    page_size,
                    batch_size,
                )
            if page_size == 32:
                # Every page-32 batch within the row cap launches a program.
                assert (expected is None) == (batch_size > core._CAKE_MLA_MAX_ROWS)
    runner = _cpu_runner(torch.bfloat16, 32)
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {})
    assert runner._families(_cpu_inputs(torch.bfloat16, 4), (4,)) is None


@pytest.mark.parametrize("facts", [("sm_100a", 148), ("sm_103a", 152)])
def test_cake_families_memo_follows_the_splitkv_range(
    monkeypatch, cake_families, facts
):
    """The memoized families of a long-KV batch follow the device's split-KV
    range: a unit listed on the split-KV domain launches the split-KV family
    while the range covers its max_seq_len, and nothing once the range is
    replaced or changed in place (the domain is then the value-split one, which
    that unit does not list); a unit listed on the value-split domain only
    then. No GPU needed."""
    arch, num_sms = facts
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: facts)
    monkeypatch.setattr(cake_module, "_SPLITKV_RANGE", dict(cake_module._SPLITKV_RANGE))
    key = (arch, num_sms, "bfloat16", 32, 1, False)
    splitkv = (cake_module.DOMAIN_BF16_SPLITKV, 4096, 1, 15)
    vquarter = (cake_module.DOMAIN_BF16_VQUARTER, 4096, 1, 15)
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {key: (splitkv,)})
    runner = _cpu_runner(torch.bfloat16, 32, 4096)
    inputs = _cpu_inputs(torch.bfloat16, 4)
    expected = frozenset(
        cake_module.launch_families(cake_module.DOMAIN_BF16_SPLITKV, arch, num_sms, 4)
    )
    for _ in range(2):
        assert runner._families(inputs, (4,)) == expected
    # Changed in place: the range no longer reaches max_seq_len 4096.
    cake_module._SPLITKV_RANGE[facts] = (4097, 18)
    assert runner._families(inputs, (4,)) is None
    # Replaced: the device is no longer listed.
    monkeypatch.setattr(cake_module, "_SPLITKV_RANGE", {})
    assert runner._families(inputs, (4,)) is None
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {key: (vquarter,)})
    assert runner._families(inputs, (4,)) == frozenset(
        cake_module.launch_families(cake_module.DOMAIN_BF16_VQUARTER, arch, num_sms, 4)
    )
    monkeypatch.setattr(cake_module, "_SPLITKV_RANGE", {facts: (2049, 18)})
    assert runner._families(inputs, (4,)) is None


# max_seq_len values across the bucket edges 128 to 4096 and the KV tile edges
# of their neighbourhoods.
_SWEPT_MAX_SEQ_LENS = sorted(
    {
        *range(1, 300),
        *range(480, 560),
        *range(880, 1160),
        *range(1900, 2200),
        *range(2940, 3060),
        *range(3960, 4240),
    }
)

# (dtype, page size, batch) on every domain the candidate launches on the
# simulated 148-SM SM100a device: BF16 VQUARTER, VHALF, the CLC program in one
# wave and with the work order, FP8 page 64 (its split count reads max_seq_len),
# and BF16 split-KV from the device's split-KV range on (VQUARTER below it; its
# split count reads max_seq_len).
_SWEPT_GEOMETRIES = (
    (torch.bfloat16, 32, 4),
    (torch.bfloat16, 32, 32),
    (torch.bfloat16, 32, 64),
    (torch.bfloat16, 32, 128),
    (torch.bfloat16, 64, 4),
    (torch.float8_e4m3fn, 64, 16),
    (torch.float8_e4m3fn, 64, 128),
)


def _offer_every_bucket(monkeypatch, max_seq_lens, arch="sm_100a", num_sms=148):
    """A synthetic offer table listing every candidate domain for every batch
    and every max_seq_len bucket of ``max_seq_lens`` (no sinks, q_len 1),
    except VQUARTER in every second bucket: the offered ranges, and the
    families of a batch VQUARTER serves, then differ between adjacent
    buckets."""
    buckets = sorted({_round_to_seq_len_bucket(value) for value in max_seq_lens})
    table = {
        (arch, num_sms, dtype_name, page_size, 1, False): tuple(
            (domain, bucket, 1, core._CAKE_MLA_MAX_ROWS)
            for domain in (*_DOMAINS, cake_module.DOMAIN_BF16_SPLITKV)
            for position, bucket in enumerate(buckets)
            if domain != cake_module.DOMAIN_BF16_VQUARTER or position % 2 == 0
        )
        for dtype_name in ("bfloat16", "float8_e4m3fn")
        for page_size in (32, 64)
    }
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", table)
    # A placeholder FP8 decode table for the CPU tensors: _launch_plan resolves.
    monkeypatch.setattr(cake_module, "_FP8_LUTS", {torch.device("cpu"): torch.zeros(1)})


@pytest.mark.parametrize("splitkv_start", [2049, 3000])
def test_auto_memo_seq_len_holds_what_the_decision_reads(
    monkeypatch, cake_families, splitkv_start
):
    """Every ``max_seq_len`` with the same memo field
    (``_mla_auto_memo_seq_len``) has the same offered batch ranges, Cake
    cache-key extras, launched families and resolved Cake launch (domain and
    split counts included), derived without the memos, for every candidate
    domain, across max_seq_len bucket and KV tile edges and the start of the
    device's split-KV range, also one that is not a bucket edge; the memoized
    ranges and families equal that derivation for every value, also where
    the offer table differs between adjacent buckets. A caller
    varying max_seq_len in the last 121 values of a bucket has one field, and
    the values of a bucket have one per KV tile count. No GPU needed."""
    arch, num_sms = "sm_100a", 148
    monkeypatch.setattr(
        cake_module, "_SPLITKV_RANGE", {(arch, num_sms): (splitkv_start, 18)}
    )
    _offer_every_bucket(monkeypatch, _SWEPT_MAX_SEQ_LENS)
    for dtype, page_size, batch_size in _SWEPT_GEOMETRIES:
        inputs = _cpu_inputs(dtype, batch_size)
        reads = {}
        for max_seq_len in _SWEPT_MAX_SEQ_LENS:
            runner = _cpu_runner(dtype, page_size, max_seq_len)
            ranges = tuple(
                (first, last)
                for _, first, last in core._cake_mla_auto_offer_ranges(
                    arch, num_sms, dtype, page_size, 1, False, max_seq_len
                )
            )
            families = runner._launched_families(
                dtype, 1, NUM_HEADS, QK_DIM, 32, batch_size, arch, num_sms
            )
            plan = runner._launch_plan(inputs)
            label = (dtype, page_size, batch_size, max_seq_len)
            if (dtype, batch_size) != (torch.bfloat16, 4):  # VQUARTER
                assert families is not None, label
            assert plan is not None, label
            assert (
                core._cake_mla_auto_batch_ranges(
                    0, dtype, page_size, 1, False, max_seq_len
                )
                == ranges
            ), label
            assert runner._families(inputs, (batch_size,)) == families, label
            read = (ranges, runner.get_cache_key_extras(inputs), families, plan[0])
            field = core._mla_auto_memo_seq_len(max_seq_len, 0)
            assert reads.setdefault(field, read) == read, label
        if dtype == torch.float8_e4m3fn and batch_size == 16:
            # The FP8 page-64 split count follows the KV tile count (2 to 4
            # here), which the field therefore keeps.
            assert len({read[3] for read in reads.values()}) > 1
        if dtype == torch.bfloat16 and batch_size == 4:
            # Split-KV from the range's start on, VQUARTER below it, which is
            # offered in every second bucket only.
            families = {read[2] for read in reads.values()}
            assert None in families and len(families) == 3
    for top in (256, 1024, 2048):
        fields = {
            core._mla_auto_memo_seq_len(value, 0) for value in range(top - 120, top + 1)
        }
        assert len(fields) == 1
        fields = {
            core._mla_auto_memo_seq_len(value, 0)
            for value in range(top // 2 + 1, top + 1)
        }
        assert len(fields) == max(1, top // 256)
    # A range start inside a bucket's KV tile splits that tile's values.
    fields = {core._mla_auto_memo_seq_len(value, 0) for value in range(2049, 4097)}
    assert len(fields) == 16 + (splitkv_start % 128 != 1)


def test_cute_dsl_batch_cap_reads_max_seq_len_only_through_the_memo_field(
    monkeypatch, cake_families
):
    """The first stage's CuTe DSL batch cap (its tuning buckets without
    TRTLLM-GEN) reads max_seq_len only through the memo field
    (``_mla_auto_memo_seq_len``): for the monolithic workspace policy, which
    reads the KV tile count, and the modular one, which reads none of it,
    every max_seq_len of one field gives the same split counts and workspace
    sizes, the same unmemoized cap and the same memoized buckets. No GPU
    needed."""
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    from flashinfer.cute_dsl import utils as cute_dsl_utils

    num_sms = 148
    monkeypatch.setattr(cute_dsl_utils, "get_num_sm", lambda device: num_sms)
    max_seq_lens = sorted(
        {
            *range(1, 260),
            *(tile * 128 + offset for tile in range(2, 34) for offset in (-1, 0, 1, 2)),
        }
    )

    def outcome(function, *args):
        try:
            return function(*args)
        except ValueError as error:
            return type(error)

    for impl in ("monolithic", "modular"):
        sizer, resolved = core._resolve_cute_dsl_workspace_sizer(impl, None, False)
        reads = {}
        for max_seq_len in max_seq_lens:
            sizes = tuple(
                outcome(
                    core._call_cute_dsl_workspace_sizer,
                    sizer,
                    resolved,
                    batch_size,
                    1,
                    NUM_HEADS,
                    LATENT,
                    num_sms,
                    max_seq_len,
                )
                for batch_size in (1, 4, 64)
            )
            caps = tuple(
                outcome(
                    _reference_cute_dsl_cap, nbytes, 1, max_seq_len, num_sms, impl, None
                )
                for nbytes in (1 << 20, 16 << 20)
            )
            buckets = tuple(
                outcome(
                    core._compute_mla_decode_buckets,
                    _Workspace(nbytes),
                    ("cute-dsl",),
                    1,
                    NUM_HEADS,
                    LATENT,
                    max_seq_len,
                    torch.device("cpu"),
                    impl,
                    None,
                )
                for nbytes in (1 << 20, 16 << 20)
            )
            read = (sizes, caps, buckets)
            field = core._mla_auto_memo_seq_len(max_seq_len, 0)
            assert reads.setdefault(field, read) == read, (impl, max_seq_len)
        sizes = {read[0] for read in reads.values()}
        if impl == "monolithic":
            # The split count follows the KV tile count, so the field must
            # keep it.
            assert len(sizes) > 1
        else:
            assert len(sizes) == 1


def test_auto_memo_stays_bounded_when_max_seq_len_varies(
    monkeypatch, fresh_tuner, cake_families, synthetic_offers
):
    """A caller varying max_seq_len adds at most one memoized decision per
    value of the memo field (one per KV tile count, 519 for max_seq_len 1 to
    65536), one offered-ranges entry per bucket, and the decision memo is
    cleared past ``_CAKE_MLA_MEMO_LIMIT`` entries. No GPU needed."""
    # A split-KV range starting on a KV tile edge adds no field.
    monkeypatch.setattr(cake_module, "_SPLITKV_RANGE", {("sm_100a", 148): (2049, 18)})
    max_seq_lens = range(1, 65537)
    fields = {core._mla_auto_memo_seq_len(value, 0) for value in max_seq_lens}
    assert len(fields) == 65536 // 128 + 7
    for max_seq_len in max_seq_lens:
        core._cake_mla_auto_batch_ranges(0, torch.bfloat16, 64, 1, False, max_seq_len)
    assert len(core._CAKE_MLA_RANGES_MEMO[0][2]) == 17
    assert sum(key[0] == "ranges" for key in core._CAKE_MLA_TABLE_MEMO) == 17

    monkeypatch.setattr(core, "_CAKE_MLA_MEMO_LIMIT", 8)
    state = core._mla_cake_memo_state(fresh_tuner)
    assert state is not None
    runners = [_FakeTrtllmGen()]
    for field in sorted(fields):
        core._mla_cake_auto_memo_put(
            fresh_tuner,
            ("call", field),
            state,
            core._CAKE_MLA_AUTO_OFFERS,
            runners,
            (runners[0], -1),
            None,
        )
        assert 0 < len(core._CAKE_MLA_AUTO_MEMO) <= 9


def test_cake_preparation_launches_nothing():
    """The tuner's preparation call does not launch the decode (``_ready``
    built the programs). No GPU needed."""
    inputs = _cpu_inputs(torch.bfloat16, 4)
    assert _cpu_runner().forward(inputs, do_preparation=True) is None


def test_untuned_auto_keeps_first_candidate(monkeypatch, clean_tuner):
    _require_cake_arch()
    calls = _spy_choose_one(monkeypatch)
    case = _make_case()
    _offer_case(monkeypatch, case)

    with autotune(False):
        output = _decode(case)

    # Without a Cake entry the untuned call makes the first selection only.
    assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP]
    _, runners, (runner, tactic) = calls[0]
    assert "cake" not in [_name(r) for r in runners]
    assert runner is runners[0] and tactic == -1
    assert _name(runner) == "trtllm-gen"
    _assert_matches_reference(output, case, "untuned")


@pytest.mark.parametrize(
    "dtype,batch_size,q_len,page_size",
    [
        (torch.bfloat16, 4, 1, 32),
        (torch.bfloat16, 2, 2, 64),
        (torch.float8_e4m3fn, 4, 1, 64),
        (torch.float8_e4m3fn, 2, 2, 64),
    ],
)
def test_autotuned_auto_matches_reference(
    monkeypatch, clean_tuner, dtype, batch_size, q_len, page_size
):
    """Real tuning of both stages for an offered unit (synthetic table): the
    first stage never lists Cake, the second profiles it for the unit's bucket,
    and the tuned and cached calls are correct whichever candidate wins."""
    _require_cake_arch()
    profiled = []
    original = core._CakeTrtllmMlaBlackwellRunner.get_valid_tactics

    def get_valid_tactics(self, inputs, profile):
        tactics = original(self, inputs, profile)
        profiled.append(tactics)
        return tactics

    monkeypatch.setattr(
        core._CakeTrtllmMlaBlackwellRunner, "get_valid_tactics", get_valid_tactics
    )
    case = _make_case(
        dtype=dtype, batch_size=batch_size, q_len=q_len, page_size=page_size
    )
    _offer_case(monkeypatch, case)
    calls = _spy_choose_one(monkeypatch)
    results = _spy_stage_two(monkeypatch)

    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        tuned = _decode(case)
    tuned_calls = len(calls)
    tuned_results = len(results)
    with autotune(False):
        cached = _decode(case)

    ops = [op for op, _, _ in calls]
    assert ops[:tuned_calls] == [core._MLA_DECODE_OP, core._MLA_CAKE_MARGIN_OP]
    for op, runners, _ in calls:
        names = [_name(r) for r in runners]
        if op == core._MLA_DECODE_OP:
            assert names and "cake" not in names
        else:
            assert names == ["cake", "incumbent"]
    assert any(profiled), "the Cake candidate was never profiled"
    # The cached call makes the first selection, and the second (a lookup)
    # while the tuner holds a Cake winner.
    assert ops[tuned_calls:] == [core._MLA_DECODE_OP]
    assert (len(results) > tuned_results) == (
        clean_tuner.may_select_runner_class(core._CAKE_MLA_RUNNER)
    )
    for label, output in (("tuned", tuned), ("cached", cached)):
        _assert_matches_reference(output, case, label)


@pytest.mark.parametrize(
    "dtype,batch_size,q_len,page_size,max_seq_len",
    [
        (torch.bfloat16, 1, 1, 64, 128),
        (torch.bfloat16, 2, 1, 64, 256),
        (torch.float8_e4m3fn, 2, 2, 64, 1024),
    ],
)
def test_second_stage_decision_follows_its_measurements(
    monkeypatch, clean_tuner, dtype, batch_size, q_len, page_size, max_seq_len
):
    """Real tuning of a small offered unit: the second stage times the
    incumbent and Cake per offered bucket with the single cold replay, finite
    and without an error, and stores Cake exactly where Cake's time is at most
    the incumbent's plus the bucket's absolute tie (the relative tolerance
    ``_MLA_CAKE_STAGE_TWO_TIE_MS`` at the first stage's recorded time of its
    winner), confirmed by the tuner's re-timing of both. The tuned call and a
    later lookup-only call run what the stored winner of their bucket says,
    and the output matches the reference. (Whether the two measure within the
    tie is up to the timer, so the test checks the decision against the
    recorded measurements rather than forcing one.)"""
    _require_cake_arch()
    margin_runners = {"_MlaStageOneWinnerRunner", core._CAKE_MLA_RUNNER}
    # {(runner class, profiled batch): [time or exception]}
    measured: dict = {}
    # (L2 policy, cold chain launches) of every second-stage measurement.
    policies = set()
    profile = AutoTuner._profile_single_kernel

    def record(tuner, runner, inputs, tactic, tuning_config, *args, **kwargs):
        name = type(runner).__name__
        times = measured.setdefault((name, inputs[0].size(0)), [])
        if name in margin_runners:
            policies.add(
                (tuning_config.l2_cache_policy, tuning_config.cold_chain_launches)
            )
        try:
            result = profile(
                tuner, runner, inputs, tactic, tuning_config, *args, **kwargs
            )
        except Exception as exc:
            times.append(exc)
            raise
        times.append(result)
        return result

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", record)
    case = _make_case(
        dtype=dtype,
        batch_size=batch_size,
        q_len=q_len,
        page_size=page_size,
        max_seq_len=max_seq_len,
    )
    _offer_case(monkeypatch, case)
    failed = clean_tuner.stats.failed_profiling_count
    failed_before = set(failed.get(core._MLA_CAKE_MARGIN_OP, ()))
    results = _spy_stage_two(monkeypatch)

    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        tuned = _decode(case)
    tuned_decision = results[-1]

    buckets = {batch for name, batch in measured if name == core._CAKE_MLA_RUNNER}
    assert batch_size in buckets
    assert buckets == {
        batch for name, batch in measured if name == "_MlaStageOneWinnerRunner"
    }
    stored = {
        key.nearest_profile[0][0]: key.runner_class_name
        for key in clean_tuner._winner_cache()
        if key.custom_op == core._MLA_CAKE_MARGIN_OP
    }
    # The first stage's recorded time of its winner per bucket.
    first_stage_ms = {
        key.nearest_profile[0][0]: selection["time_ms"]
        for (_, key), selection in clean_tuner._profiling_cache_selections.items()
        if key.custom_op == core._MLA_DECODE_OP
    }
    for bucket in buckets:
        cake_times = measured[(core._CAKE_MLA_RUNNER, bucket)]
        incumbent_times = measured[("_MlaStageOneWinnerRunner", bucket)]
        for result in cake_times + incumbent_times:
            assert not isinstance(result, BaseException), (bucket, result)
            assert math.isfinite(result) and result > 0, (bucket, result)
        tie = min(core._MLA_CAKE_STAGE_TWO_TIE_MS / first_stage_ms[bucket], 0.5)
        cake_time, incumbent_time = cake_times[0], incumbent_times[0]
        if cake_time <= incumbent_time:
            # Cake measured fastest: no re-timing.
            rounds, cake_kept = 1, True
        elif cake_time <= incumbent_time * (1 + tie):
            # Within the tie: both re-timed twice, Cake kept on the means.
            rounds = 3
            cake_kept = statistics.fmean(cake_times[1:]) <= statistics.fmean(
                incumbent_times[1:]
            ) * (1 + tie)
        else:
            rounds, cake_kept = 1, False
        assert (len(cake_times), len(incumbent_times)) == (rounds, rounds), bucket
        expected = core._CAKE_MLA_RUNNER if cake_kept else "_MlaStageOneWinnerRunner"
        assert stored.get(bucket) == expected, (bucket, cake_times, incumbent_times)
    assert policies == {("cold", 1)}
    assert set(failed.get(core._MLA_CAKE_MARGIN_OP, ())) == failed_before
    assert _margin_policies(clean_tuner) == {"cold"}

    cake_selected = stored[batch_size] == core._CAKE_MLA_RUNNER
    assert (_name(tuned_decision[0]) == "cake") == cake_selected
    before = len(results)
    with autotune(False):
        cached = _decode(dict(case, out=_nan_out(case)))
    # The lookup-only call reaches the second stage while the tuner holds a
    # Cake winner; then it follows the stored winner of its bucket.
    if clean_tuner.may_select_runner_class(core._CAKE_MLA_RUNNER):
        assert len(results) == before + 1
        assert (_name(results[-1][0]) == "cake") == cake_selected
    else:
        assert len(results) == before and not cake_selected
    _assert_matches_reference(tuned, case, "tuned")
    _assert_matches_reference(cached, case, "cached")


@pytest.mark.parametrize(
    "dtype,batch_size,q_len,page_size",
    [
        (torch.bfloat16, 2, 2, 32),
        (torch.bfloat16, 4, 1, 64),
        (torch.float8_e4m3fn, 4, 1, 64),
    ],
)
def test_forced_cake_selection_matches_reference(
    monkeypatch, clean_tuner, dtype, batch_size, q_len, page_size
):
    _require_cake_arch()
    case = _make_case(
        dtype=dtype, batch_size=batch_size, q_len=q_len, page_size=page_size
    )
    seen = _force_cake(monkeypatch, case)
    launches = _spy_cake_forward(monkeypatch)

    forced = _decode(case)
    _assert_cake_launched(seen, launches)
    cake, inputs = seen[0]
    direct = torch.zeros_like(forced)
    cake(inputs=[*inputs[:3], direct], tactic=_forced_tactic(cake))
    explicit = _decode(dict(case, backend="cake"))

    for label, output in (
        ("forced", forced),
        ("direct", direct),
        ("explicit", explicit),
    ):
        _assert_matches_reference(output, case, label)


def _spy_stage_two(monkeypatch):
    """Records the ``(runner, tactic)`` every second selection stage returns."""
    results = []
    original = core._mla_cake_stage_two

    def stage_two(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(core, "_mla_cake_stage_two", stage_two)
    return results


def test_unlaunchable_cake_choice_falls_back(monkeypatch, clean_tuner):
    _require_cake_arch()
    # Aligned BF16 page 64 has no Cake domain once rows exceed a quarter of the SMs.
    batch_size = get_device_sm_count(torch.device("cuda"))
    case = _make_case(page_size=64, batch_size=batch_size, max_seq_len=256)
    selected = _force_cake(monkeypatch, case)
    launches = _spy_cake_forward(monkeypatch)
    results = _spy_stage_two(monkeypatch)

    output = _decode(case)

    cake, inputs = selected[0]
    assert _name(cake) == "cake" and not cake.supports(inputs)
    assert not launches and results[-1] == selected.first_stage
    _assert_matches_reference(output, case, "fallback")


def _spy_cake_forward(monkeypatch):
    launches = []
    original_forward = core._CakeTrtllmMlaBlackwellRunner.forward

    def forward(self, *args, **kwargs):
        launches.append(self)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(core._CakeTrtllmMlaBlackwellRunner, "forward", forward)
    return launches


def test_unavailable_cake_family_falls_back(monkeypatch, clean_tuner):
    """A Cake choice whose family this process recorded as unavailable is not
    launched although the batch is in its domains (supports() holds)."""
    _require_cake_arch()
    case = _make_case(batch_size=2, q_len=2, page_size=32)
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    family = cake_module.launch_family(
        cake_module.DOMAIN_BF16_VQUARTER, arch, num_sms, 4
    )
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", {family})
    selected = _force_cake(monkeypatch, case)
    launches = _spy_cake_forward(monkeypatch)
    results = _spy_stage_two(monkeypatch)

    output = _decode(case)

    cake, inputs = selected[0]
    assert _name(cake) == "cake" and cake.supports(inputs)
    assert cake._domain(inputs, num_sms, arch=arch) == cake_module.DOMAIN_BF16_VQUARTER
    assert not cake.validate_tactic(inputs, _forced_tactic(cake))
    assert not launches and results[-1] == selected.first_stage
    _assert_matches_reference(output, case, "unavailable family fallback")


def test_cake_choice_not_built_during_capture(monkeypatch, clean_tuner):
    """A Cake choice whose program is not loaded when a CUDA graph is captured
    runs the first stage's selection in the graph: no build, no Cake launch."""
    _require_cake_arch()
    case = _make_case(batch_size=2, q_len=2, page_size=32)
    batch_size, q_len = case["query"].shape[:2]
    case["out"] = torch.zeros(
        batch_size, q_len, NUM_HEADS, LATENT, dtype=torch.bfloat16, device="cuda"
    )
    selected = _force_cake(monkeypatch, case)
    # Eager warm-up of the first stage's selection only: switched off, Cake is
    # skipped.
    monkeypatch.setenv(DISABLE_ENV, "1")
    _decode(case)
    monkeypatch.delenv(DISABLE_ENV)
    torch.cuda.synchronize()
    assert not selected

    builds = []

    def build(name):
        builds.append(name)
        raise RuntimeError(f"{name} built during capture")

    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())
    monkeypatch.setattr(cake_module, "get_cake_trtllm_mla_blackwell_family", build)
    launches = _spy_cake_forward(monkeypatch)
    results = _spy_stage_two(monkeypatch)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode(case)
    torch.cuda.synchronize()

    cake, inputs = selected[-1]
    assert _name(cake) == "cake" and cake.supports(inputs)
    assert not builds and not launches and results[-1] == selected.first_stage
    assert not core._CAKE_MLA_UNAVAILABLE_FAMILIES
    case["out"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(case["out"], case, "capture fallback replay")


class _Forced(list):
    """``(runner, inputs)`` of every forced Cake choice, and the first-stage
    selection of the last call."""

    first_stage = None


def _force_cake(monkeypatch, case):
    """Offer the unit of ``case``, then make the second stage find, without
    tuning, a Cake entry that beat the first-stage selection; the first stage
    selects its last candidate (CuTe DSL where it serves the request, not the
    default TRTLLM-GEN) with the default tactic."""
    _offer_case(monkeypatch, case)
    forced = _Forced()
    original_search_cache = core.AutoTuner.search_cache

    def beaten():
        runner, tactic = forced.first_stage
        return type(runner).__name__, tactic

    def choose_one(self, op, runners, config, inputs, **kwargs):
        assert op == core._MLA_DECODE_OP, op
        forced.first_stage = (runners[-1], -1)
        return forced.first_stage

    def search_cache(self, op, runners, *args, **kwargs):
        if op != core._MLA_CAKE_MARGIN_OP:
            return original_search_cache(self, op, runners, *args, **kwargs)
        [index] = [
            index
            for index, runner in enumerate(runners)
            if isinstance(runner, core._CakeTrtllmMlaBlackwellRunner)
        ]
        forced.append((runners[index], list(kwargs["inputs"])))
        return True, index, beaten(), None

    monkeypatch.setattr(
        core.AutoTuner, "may_select_runner_class", lambda self, name: True
    )
    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    monkeypatch.setattr(core.AutoTuner, "search_cache", search_cache)
    monkeypatch.setattr(
        core._MlaStageOneWinnerRunner, "call_winner", lambda self, inputs: beaten()
    )
    return forced


def _forced_tactic(cake):
    """The Cake tactic of a forced choice: the first-stage selection it beat."""
    return cake._incumbent.call_winner(None)


def _assert_cake_launched(selected, launches):
    """The last forced Cake choice was validated and its forward ran (eagerly
    or recorded into a graph), not the first stage's selection."""
    cake, inputs = selected[-1]
    assert _name(cake) == "cake"
    assert cake.validate_tactic(inputs, _forced_tactic(cake))
    assert launches and all(_name(runner) == "cake" for runner in launches)
    launches.clear()


def _profile_times(times):
    """A stand-in for kernel profiling: ``times`` (ms) per runner class (2.0
    when absent); the second-stage incumbent takes its winner's time."""

    def profile(tuner, runner, inputs, tactic, tuning_config, **kwargs):
        name = type(runner).__name__
        if isinstance(runner, core._MlaStageOneWinnerRunner):
            name = tactic[0]
        return times.get(name, 2.0)

    return profile


@pytest.mark.parametrize(
    "miss", ["validate_miss", "switch_off", "tensor_scales", "capture_without_family"]
)
def test_cake_miss_runs_the_tuned_first_stage_winner(monkeypatch, fresh_tuner, miss):
    """After tuning with Cake measured fastest and TRTLLM-GEN slowest, a cached
    call runs Cake; each Cake miss runs exactly the first stage's selection,
    which is the tuned CuTe DSL winner where CuTe DSL serves the request."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=64)
    _offer_case(monkeypatch, case)
    monkeypatch.setattr(
        core.AutoTuner,
        "_profile_single_kernel",
        _profile_times(
            {
                "_BatchMLAPagedAttentionTrtllmGenBackend": 3.0,
                core._CAKE_MLA_RUNNER: 1.0,
            }
        ),
    )
    calls = _spy_choose_one(monkeypatch)
    results = _spy_stage_two(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)
    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        _decode(case)
    tuned_winner = [sel for op, _, sel in calls if op == core._MLA_DECODE_OP][-1]

    with autotune(False):
        output = _decode(dict(case, out=_nan_out(case)))
    assert _name(results[-1][0]) == "cake" and launches
    _assert_matches_reference(output, case, "cake")

    calls.clear()
    results.clear()
    launches.clear()
    variant = dict(case, out=_nan_out(case))
    if miss == "validate_miss":
        monkeypatch.setattr(
            core._CakeTrtllmMlaBlackwellRunner,
            "validate_tactic",
            lambda self, inputs, tactic: False,
        )
        # The memoized decisions (of both stages, and of the second) were
        # derived from the validator replaced here; in a process the validator
        # never changes.
        core._CAKE_MLA_AUTO_MEMO.clear()
        core._CAKE_MLA_STAGE_TWO_MEMO.clear()
    elif miss == "switch_off":
        monkeypatch.setenv(DISABLE_ENV, "1")
        assert fresh_tuner.may_select_runner_class(core._CAKE_MLA_RUNNER)
    elif miss == "tensor_scales":
        variant.update(_tensor_scales(case, monkeypatch))
        variant["out"] = _nan_out(case)
    if miss == "capture_without_family":
        # Warm the first stage's winner eagerly: switched off, Cake is skipped.
        monkeypatch.setenv(DISABLE_ENV, "1")
        _decode(variant)
        monkeypatch.delenv(DISABLE_ENV)
        torch.cuda.synchronize()
        calls.clear()

        def build(name):
            raise RuntimeError(f"{name} built during capture")

        monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
        monkeypatch.setattr(cake_module, "get_cake_trtllm_mla_blackwell_family", build)
        graph = torch.cuda.CUDAGraph()
        with autotune(False), torch.cuda.graph(graph):
            _decode(variant)
        variant["out"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
    else:
        with autotune(False):
            _decode(variant)
    output = variant["out"]

    (first_stage,) = [sel for op, _, sel in calls if op == core._MLA_DECODE_OP]
    executed = results[-1] if results else first_stage
    assert executed[0] is first_stage[0] and executed[1] == first_stage[1]
    assert not launches
    if miss != "tensor_scales":
        # The same request: the first stage's tuned winner, not its default.
        assert type(first_stage[0]) is type(tuned_winner[0])
        assert first_stage[1] == tuned_winner[1]
    _assert_matches_reference(output, case, miss)


# (dtype, page size, query rows[, max_seq_len], default 1024). With 74 or 76
# resident clusters (148 or 152 SMs), the BF16 rows reach VQUARTER (4), VHALF
# (32), the CLC program in one wave (64; the one-wave program where one is
# generated for the device, see ``_one_wave_route``) and with the longest-first
# order program (128, and 300 with a partial last order CTA); FP8 page 64 has one
# domain. Sinks make every BF16 row VQUARTER, with several work items per cluster
# above 18 rows. The split-KV geometries (max_seq_len bucket 4096, few rows) reach
# the split-KV program on the devices whose ``_SPLITKV_RANGE`` covers them (B200
# and GB300 from 2049). The tail domains never join backend="auto".
_PARITY_GEOMETRIES = {
    "bf16_p32_vquarter": (torch.bfloat16, 32, 4),
    "bf16_p32_vhalf": (torch.bfloat16, 32, 32),
    "bf16_p32_clc_one_wave": (torch.bfloat16, 32, 64),
    "bf16_p32_clc": (torch.bfloat16, 32, 128),
    "bf16_p32_clc_window300": (torch.bfloat16, 32, 300),
    "bf16_p64_vquarter": (torch.bfloat16, 64, 4),
    "bf16_p64_vhalf": (torch.bfloat16, 64, 32),
    "fp8_p64_small": (torch.float8_e4m3fn, 64, 4),
    "fp8_p64_large": (torch.float8_e4m3fn, 64, 128),
    "bf16_p32_splitkv": (torch.bfloat16, 32, 4, 4096),
    "bf16_p64_splitkv": (torch.bfloat16, 64, 4, 3072),
}

# Variants: padded requests have seq_lens 0 (every third request); with q_len 2
# the next request has seq_lens 1, so its first query row has no causal KV token.
# "all_padded": every request has seq_lens 0.
_PARITY_VARIANTS = (
    "plain",
    "sinks",
    "bmm2_scale",
    "padded",
    "sinks_padded",
    "q_len2",
    "padded_q_len2",
    "all_padded",
)


def _one_wave_route(arch: str, num_sms: int) -> bool:
    key = cake_module.route_key(cake_module.DOMAIN_BF16_CLC_ONE_WAVE, arch, num_sms)
    return key in cake_module.ROUTES


# Variants whose forced Cake selection backend="auto" never reaches, so their
# cells are expected skips: requests with sinks are not offered (explicit
# backend="cake" sink parity is checked in tests/attention/test_trtllm_gen_mla.py).
_AUTO_SKIPPED_VARIANTS = {
    "sinks": "backend='auto' does not offer Cake for requests with sinks",
    "sinks_padded": "backend='auto' does not offer Cake for requests with sinks",
}


def _skip_unserved_geometry(geometry, variant, arch, num_sms) -> None:
    """Skip the expected cells: variants backend="auto" never offers, FP8 with
    sinks (the tail program), CLC with bmm2_scale != 1, the one-wave CLC
    geometry on a device without that program and the split-KV geometries
    outside the device's ``_SPLITKV_RANGE``."""
    dtype = _PARITY_GEOMETRIES[geometry][0]
    if "sinks" in variant and dtype != torch.bfloat16:
        pytest.skip("FP8 with sinks takes the tail program, which never joins")
    if variant in _AUTO_SKIPPED_VARIANTS:
        pytest.skip(_AUTO_SKIPPED_VARIANTS[variant])
    if variant == "bmm2_scale" and "clc" in geometry:
        pytest.skip("the CLC domain requires bmm2_scale == 1")
    if geometry == "bf16_p32_clc_one_wave" and not _one_wave_route(arch, num_sms):
        pytest.skip(f"no one-wave CLC program for {arch} with {num_sms} SMs")
    if "splitkv" in geometry:
        _, _, rows, max_seq_len = _PARITY_GEOMETRIES[geometry]
        limits = cake_module._SPLITKV_RANGE.get((arch, num_sms))
        if limits is None or max_seq_len < limits[0] or rows > limits[1]:
            pytest.skip(
                f"no split-KV domain for {geometry} on {arch} with {num_sms} SMs"
            )


def _parity_case(geometry, q_len=1, seed=0):
    dtype, page_size, rows, *max_seq_len = _PARITY_GEOMETRIES[geometry]
    return _make_case(
        dtype=dtype,
        page_size=page_size,
        batch_size=rows // q_len,
        q_len=q_len,
        max_seq_len=max_seq_len[0] if max_seq_len else 1024,
        seed=seed,
    )


def _pad(seq_lens, q_len) -> None:
    """Every third request padded (seq_lens 0); with q_len > 1 the next request's
    first query row has no causal KV token (seq_lens 1)."""
    seq_lens[::3] = 0
    if q_len > 1:
        seq_lens[1::3] = 1


def _live_rows(seq_lens, q_len):
    """[batch, q_len] mask of the query rows with at least one causal KV token."""
    offsets = torch.arange(1 - q_len, 1, dtype=torch.int32, device=seq_lens.device)
    return seq_lens[:, None] + offsets[None, :] >= 1


def _nan_out(case):
    shape = (*case["query"].shape[:-1], LATENT)
    return torch.full(shape, float("nan"), dtype=torch.bfloat16, device="cuda")


def _assert_live_and_padded(output, case, label):
    """Live rows match the FP32 reference; padded rows are exact zeros."""
    live = _live_rows(case["seq_lens"], case["query"].shape[1])
    if bool(live.any()):
        tolerance = TOLERANCE[case["query"].dtype]
        torch.testing.assert_close(
            output[live],
            _reference(case)[live],
            rtol=tolerance,
            atol=tolerance,
            msg=lambda message: f"{label} vs reference: {message}",
        )
    padded = output[~live]
    assert torch.equal(padded, torch.zeros_like(padded)), f"{label}: padded rows"


# A desynchronized KV ring hangs in torch.cuda.synchronize(), which the signal
# method cannot interrupt; the limit also covers a cold JIT build of a family.
@pytest.mark.timeout(600, method="thread")
@pytest.mark.parametrize("variant", _PARITY_VARIANTS)
@pytest.mark.parametrize("geometry", sorted(_PARITY_GEOMETRIES))
def test_forced_cake_matches_trtllm_gen(monkeypatch, clean_tuner, geometry, variant):
    """The auto Cake candidate matches the reference and TRTLLM-GEN on every
    domain it reaches, with sinks, bmm2_scale != 1 and q_len > 1. Padded rows
    (no causal KV token) output exact zeros (the output is NaN-filled first) and
    do not disturb the other rows; a batch of padded requests only completes.
    TRTLLM-GEN leaves padded rows unwritten, so they are compared on live rows."""
    _require_cake_arch()
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    _skip_unserved_geometry(geometry, variant, arch, num_sms)
    dtype, _, rows = _PARITY_GEOMETRIES[geometry][:3]
    q_len = 2 if variant.endswith("q_len2") else 1
    case = _parity_case(geometry, q_len)
    if "sinks" in variant:
        generator = torch.Generator().manual_seed(1)
        sink = torch.randn(NUM_HEADS, generator=generator) * 2.0
        case["sinks"] = [sink.to("cuda")]
    elif variant == "bmm2_scale":
        case["bmm2_scale"] = 0.5
    if variant == "all_padded":
        case["seq_lens"].zero_()
    elif "padded" in variant:
        _pad(case["seq_lens"], q_len)

    selected = _force_cake(monkeypatch, case)
    launches = _spy_cake_forward(monkeypatch)
    cake_out = _decode(dict(case, out=_nan_out(case)))
    torch.cuda.synchronize()
    _assert_cake_launched(selected, launches)
    cake, inputs = selected[-1]
    domain = cake._domain(inputs, num_sms, arch=arch)
    families = cake_module.launch_families(domain, arch, num_sms, rows)
    label = f"{geometry}/{variant} (Cake {domain}, families {families})"
    if "splitkv" in geometry:
        assert domain == cake_module.DOMAIN_BF16_SPLITKV, label
    if geometry == "bf16_p32_clc_one_wave" and domain == cake_module.DOMAIN_BF16_CLC:
        one_wave = cake_module.route_key(
            cake_module.DOMAIN_BF16_CLC_ONE_WAVE, arch, num_sms
        )
        assert families == (cake_module.ROUTES[one_wave],), label
    _assert_live_and_padded(cake_out, case, label)

    # Each side is within the tolerance of the reference on live rows.
    # The explicit TRTLLM-GEN entry takes the sink as one tensor, not a list.
    live = _live_rows(case["seq_lens"], q_len)
    if not bool(live.any()):
        return
    trtllm_gen_case = dict(case, backend="trtllm-gen", out=_nan_out(case))
    if case.get("sinks") is not None:
        trtllm_gen_case["sinks"] = case["sinks"][0]
    trtllm_gen_out = _decode(trtllm_gen_case)
    tolerance = TOLERANCE[dtype]
    torch.testing.assert_close(
        cake_out[live],
        trtllm_gen_out[live],
        rtol=2 * tolerance,
        atol=2 * tolerance,
        msg=lambda message: f"{label} vs trtllm-gen: {message}",
    )


# Geometries of the in-place update and graph-replay tests: every program the
# candidate launches (value-split counts, one-wave and ordered CLC, FP8 page 64).
_STATELESS_GEOMETRIES = (
    "bf16_p32_vquarter",
    "bf16_p32_vhalf",
    "bf16_p64_vhalf",
    "bf16_p32_clc_one_wave",
    "bf16_p32_clc",
    "fp8_p64_small",
    "bf16_p32_splitkv",
    "bf16_p64_splitkv",
)


def _explicit_cake(case, seq_lens, block_tables):
    """Explicit ``backend="cake"`` on fresh tensors holding the given values."""
    return _decode(
        dict(
            case,
            backend="cake",
            seq_lens=seq_lens.clone(),
            block_tables=block_tables.clone(),
            out=_nan_out(case),
        )
    )


@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("geometry", _STATELESS_GEOMETRIES)
def test_forced_cake_reads_updated_inputs(monkeypatch, clean_tuner, geometry, q_len):
    _require_cake_arch()
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    _skip_unserved_geometry(geometry, "plain", arch, num_sms)
    case = _parity_case(geometry, q_len)
    selected = _force_cake(monkeypatch, case)
    launches = _spy_cake_forward(monkeypatch)
    _assert_live_and_padded(_decode(dict(case, out=_nan_out(case))), case, "first call")
    _assert_cake_launched(selected, launches)

    # Rewrite seq_lens / block_tables without bumping their version counters,
    # as a non-torch writer (Triton, CUDA) does; then pad some requests the same way.
    update = _parity_case(geometry, q_len, seed=1)
    assert not torch.equal(update["seq_lens"], case["seq_lens"])
    versions = (case["seq_lens"]._version, case["block_tables"]._version)
    case["seq_lens"].data.copy_(update["seq_lens"])
    case["block_tables"].data.copy_(update["block_tables"])
    assert (case["seq_lens"]._version, case["block_tables"]._version) == versions
    _assert_live_and_padded(
        _decode(dict(case, out=_nan_out(case))), case, "updated inputs"
    )
    _assert_cake_launched(selected, launches)

    padded = update["seq_lens"].clone()
    _pad(padded, q_len)
    case["seq_lens"].data.copy_(padded)
    assert case["seq_lens"]._version == versions[0]
    _assert_live_and_padded(
        _decode(dict(case, out=_nan_out(case))), case, "padded update"
    )
    _assert_cake_launched(selected, launches)


@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("geometry", _STATELESS_GEOMETRIES)
def test_forced_cake_cuda_graph_replay_with_updated_inputs(
    monkeypatch, clean_tuner, geometry, q_len
):
    """A graph captured through the auto Cake candidate follows later in-place
    changes of seq_lens, block_tables and query: every replay equals explicit
    ``backend="cake"`` on fresh tensors holding the same values, matches the
    reference on live rows and outputs zeros on padded rows."""
    _require_cake_arch()
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    _skip_unserved_geometry(geometry, "plain", arch, num_sms)
    case = _parity_case(geometry, q_len)
    selected = _force_cake(monkeypatch, case)
    batch_size, max_len = case["query"].shape[0], case["max_seq_len"]
    case["out"] = _nan_out(case)

    # Eager warm-up builds the programs; the graph must not depend on its state.
    launches = _spy_cake_forward(monkeypatch)
    _decode(case)
    _decode(case)
    torch.cuda.synchronize()
    _assert_cake_launched(selected, launches)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode(case)
    torch.cuda.synchronize()
    # The Cake forward was recorded into the graph, not the fallback.
    _assert_cake_launched(selected, launches)

    padded = case["seq_lens"].clone()
    _pad(padded, q_len)
    generator = torch.Generator(device="cuda").manual_seed(1)
    width = case["block_tables"].shape[1]
    for label, values in (
        ("captured", case["seq_lens"].clone()),
        ("all_zero", torch.zeros_like(padded)),
        ("all_max", torch.full_like(padded, max_len)),
        ("padded", padded),
    ):
        case["seq_lens"].copy_(values)
        permutation = torch.randperm(
            batch_size * width, device="cuda", generator=generator
        )
        case["block_tables"].copy_(
            case["block_tables"].reshape(-1)[permutation].reshape(batch_size, width)
        )
        case["out"].fill_(float("nan"))
        graph.replay()
        eager = _explicit_cake(case, case["seq_lens"], case["block_tables"])
        torch.cuda.synchronize()
        assert torch.equal(case["out"], eager), label
        _assert_live_and_padded(case["out"], case, f"replay {label}")

    update = _parity_case(geometry, q_len, seed=1)
    case["query"].copy_(update["query"])
    case["out"].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_live_and_padded(case["out"], case, "replay after query update")


@pytest.mark.parametrize("geometry", ["bf16_p32_clc", "bf16_p32_clc_window300"])
def test_forced_cake_graphs_replay_concurrently(monkeypatch, clean_tuner, geometry):
    """Two graphs captured on the default capture stream through the auto Cake
    candidate, replayed on two streams at once, each equal their eager result:
    each captured ordered CLC launch owns its work order. Stateless calls add no
    per-stream work-order buffer."""
    _require_cake_arch()
    cases = [_parity_case(geometry, seed=seed) for seed in (0, 1)]
    selected = _force_cake(monkeypatch, cases[0])
    launches = _spy_cake_forward(monkeypatch)
    _pad(cases[1]["seq_lens"], 1)
    # Explicit backend="cake" (default mode) uses the per-stream buffer.
    expected = [
        _explicit_cake(case, case["seq_lens"], case["block_tables"]) for case in cases
    ]
    torch.cuda.synchronize()
    buffers = len(cake_module._WORK_ORDER_BUFFERS)
    graphs = []
    for case in cases:
        case["out"] = _nan_out(case)
        _decode(case)
        torch.cuda.synchronize()
        _assert_cake_launched(selected, launches)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _decode(case)
        _assert_cake_launched(selected, launches)
        graphs.append(graph)
    torch.cuda.synchronize()
    assert len(cake_module._WORK_ORDER_BUFFERS) == buffers

    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(20):
        for graph, stream in zip(graphs, streams, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for index, case in enumerate(cases):
        assert torch.equal(case["out"], expected[index]), index
        _assert_live_and_padded(case["out"], case, f"graph {index}")


@pytest.mark.parametrize(
    "geometry",
    [
        "bf16_p32_vquarter",
        "bf16_p64_vhalf",
        "bf16_p32_clc_one_wave",
        "bf16_p32_clc",
        "fp8_p64_small",
        "bf16_p32_splitkv",
    ],
)
def test_stateless_cake_reads_no_host_state(monkeypatch, clean_tuner, geometry):
    """A warm stateless Cake call reads nothing on the host, builds no torch row
    plan, caches nothing, and launches exactly the families ``launch_families``
    names: the attention program, preceded by the longest-first order program
    for an ordered CLC launch. Padded requests included."""
    _require_cake_arch()
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    _skip_unserved_geometry(geometry, "plain", arch, num_sms)
    case = _parity_case(geometry, q_len=2)
    selected = _force_cake(monkeypatch, case)
    _pad(case["seq_lens"], 2)
    case["out"] = _nan_out(case)
    _decode(case)  # warm-up: builds the programs and the FP8 decode table
    torch.cuda.synchronize()
    cake, inputs = selected[-1]
    domain = cake._domain(inputs, num_sms, arch=arch)
    rows = case["query"].shape[0] * case["query"].shape[1]

    def forbidden(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"stateless Cake called {name}")

        return fail

    for name in ("_host_kv_lens", "_row_plan", "row_metadata", "_work_order_buffer"):
        monkeypatch.setattr(cake_module, name, forbidden(name))
    launched = []
    get_family = cake_module.get_cake_trtllm_mla_blackwell_family

    class _Recorded:
        def __init__(self, name):
            self.name, self.module = name, get_family(name)

        def run(self, *args):
            launched.append(self.name)
            return self.module.run(*args)

    monkeypatch.setattr(cake_module, "get_cake_trtllm_mla_blackwell_family", _Recorded)
    workspace_keys = set(cake_module._WORKSPACE_CACHE)
    buffers = len(cake_module._WORK_ORDER_BUFFERS)
    case["out"].fill_(float("nan"))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cake(inputs=inputs, tactic=-1)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert tuple(launched) == cake_module.launch_families(domain, arch, num_sms, rows)
    assert set(cake_module._WORKSPACE_CACHE) == workspace_keys
    assert len(cake_module._WORK_ORDER_BUFFERS) == buffers
    _assert_live_and_padded(case["out"], case, f"{geometry} stateless")


def test_explicit_cake_first_use_inside_capture_raises(monkeypatch):
    """The FP8 decode table (also the placeholder sink pointer) is created by a
    host copy, which a stream capture cannot record: a first use inside capture
    raises a clear error instead of an invalid capture."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=32)

    def decode():
        return cake_module.trtllm_mla_blackwell_decode(
            case["query"],
            case["kv_cache"],
            case["block_tables"],
            case["seq_lens"],
            case["max_seq_len"],
            qk_nope_head_dim=128,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=QK_DIM - LATENT,
            sparse_mla_top_k=0,
            out=None,
            bmm1_scale=case["bmm1_scale"],
            bmm2_scale=1.0,
            sinks=None,
            skip_softmax_threshold_scale_factor=None,
            enable_pdl=None,
            uses_shared_paged_kv_idx=True,
            lse=None,
            return_lse=False,
            cum_seq_lens_q=None,
            max_q_len=None,
            multi_ctas_kv_counter_buffer=None,
            sparse_mla_top_k_lens=None,
            enable_dcp=False,
        )

    decode()  # builds the program
    torch.cuda.synchronize()
    monkeypatch.setattr(cake_module, "_FP8_LUTS", {})
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="before CUDA graph capture"),
        torch.cuda.graph(graph),
    ):
        decode()
    torch.cuda.synchronize()


def test_cake_not_listed_for_tail_shapes_with_cake_entries(monkeypatch, clean_tuner):
    """With Cake entries in the tuner (e.g. loaded from a cache file), an FP8
    page-32 request, which only the multi-millisecond tail program would serve,
    makes the first selection only and launches no Cake."""
    _require_cake_arch()
    monkeypatch.setattr(
        core.AutoTuner, "may_select_runner_class", lambda self, name: True
    )
    case = _make_case(dtype=torch.float8_e4m3fn, page_size=32)
    _offer_case(monkeypatch, case)
    calls = _spy_choose_one(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)
    with autotune(False):
        output = _decode(case)
    assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP]
    _, runners, (runner, tactic) = calls[-1]
    assert "cake" not in [_name(r) for r in runners]
    assert runner is runners[0] and tactic == -1 and not launches
    _assert_matches_reference(output, case, "fp8 page 32")


def test_cake_build_failure_drops_cake_from_tuning(monkeypatch, clean_tuner):
    _require_cake_arch()

    def unavailable(name):
        raise RuntimeError(f"no toolchain for {name}")

    monkeypatch.setattr(
        cake_module, "get_cake_trtllm_mla_blackwell_family", unavailable
    )
    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())
    profiled = []
    original = core._CakeTrtllmMlaBlackwellRunner.get_valid_tactics

    def get_valid_tactics(self, inputs, profile):
        tactics = original(self, inputs, profile)
        profiled.append(tactics)
        return tactics

    monkeypatch.setattr(
        core._CakeTrtllmMlaBlackwellRunner, "get_valid_tactics", get_valid_tactics
    )
    case = _make_case(batch_size=2, q_len=2)
    _offer_case(monkeypatch, case)
    calls = _spy_choose_one(monkeypatch)
    results = _spy_stage_two(monkeypatch)

    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        output = _decode(case)

    assert [op for op, _, _ in calls] == [
        core._MLA_DECODE_OP,
        core._MLA_CAKE_MARGIN_OP,
    ]
    assert [_name(r) for r in calls[1][1]] == ["cake", "incumbent"]
    (runner, _), first_stage = results[-1], calls[0][2]
    assert runner is first_stage[0]
    assert profiled and all(tactics == [] for tactics in profiled)
    assert core._CAKE_MLA_UNAVAILABLE_FAMILIES
    _assert_matches_reference(output, case, f"winner {_name(runner)}")


# --- The resolved Cake launch and the memoized second stage on a device ---


def _record_runs(monkeypatch):
    """Replace every family by a recorder of its ``run`` calls (nothing is
    built or launched)."""
    runs = []

    class _Recorder:
        def __init__(self, name):
            self.name = name

        def run(self, *args):
            runs.append((self.name, args))

    monkeypatch.setattr(cake_module, "get_cake_trtllm_mla_blackwell_family", _Recorder)
    return runs


def _normalized_runs(runs, roles):
    """Each run with its tensors named by role (a caller tensor or the FP8
    decode table, by address) or as per-call scratch, with shape, strides and
    dtype; the other arguments as they are."""
    normalized = []
    for name, args in runs:
        values = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                values.append(
                    (
                        roles.get(arg.data_ptr(), "scratch"),
                        tuple(arg.shape),
                        tuple(arg.stride()),
                        arg.dtype,
                    )
                )
            else:
                values.append(arg)
        normalized.append((name, tuple(values)))
    return normalized


# (dtype, page size, query rows, max_seq_len) of every program the candidate
# launches. The last three reach the split-KV program on both simulated devices
# (``_SPLITKV_RANGE``: B200 and GB300 from 2049, up to 18 rows).
_LAUNCH_GEOMETRIES = (
    (torch.bfloat16, 32, 4, 1024),
    (torch.bfloat16, 32, 32, 1024),
    (torch.bfloat16, 32, 64, 1024),
    (torch.bfloat16, 32, 128, 1024),
    (torch.bfloat16, 32, 300, 1024),
    (torch.bfloat16, 64, 4, 1024),
    (torch.bfloat16, 64, 32, 1024),
    (torch.float8_e4m3fn, 64, 4, 1024),
    (torch.float8_e4m3fn, 64, 64, 1024),
    (torch.float8_e4m3fn, 64, 128, 1024),
    (torch.bfloat16, 32, 4, 4096),
    (torch.bfloat16, 32, 18, 2049),
    (torch.bfloat16, 64, 2, 16384),
)


@pytest.mark.parametrize("bmm2_scale", [1.0, 0.5])
@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("facts", [("sm_100a", 148), ("sm_103a", 152)])
def test_resolved_cake_launch_matches_the_dispatcher(
    monkeypatch, facts, q_len, bmm2_scale
):
    """The launch the second stage resolves once (``_launch_plan``) makes the
    same family calls with the same arguments as the stateless dispatcher on
    every program the candidate reaches (split-KV included), both simulated
    devices, q_len 1 and 2 and bmm2_scale != 1: the same caller tensors
    (address, shape, strides, dtype), the same scalars and stream, per-call
    scratch of the same sizes. The resolved launch takes the device's domain,
    so a split-KV geometry resolves the split-KV program. Nothing is built or
    launched; any CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setattr(cake_module, "_device_facts", lambda index: facts)
    runs = _record_runs(monkeypatch)
    device = torch.device("cuda", torch.cuda.current_device())
    lut = cake_module._fp8_lut(device)
    checked = splitkv = 0
    for dtype, page_size, rows, max_seq_len in _LAUNCH_GEOMETRIES:
        batch_size = rows // q_len
        width = -(-max_seq_len // page_size)
        query = torch.empty(
            batch_size, q_len, NUM_HEADS, QK_DIM, dtype=dtype, device=device
        )
        # Every page-table entry names page 0: a small pool suffices.
        kv_cache = torch.empty(4, page_size, QK_DIM, dtype=dtype, device=device)
        block_tables = torch.zeros(batch_size, width, dtype=torch.int32, device=device)
        seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
        out = torch.empty(
            batch_size, q_len, NUM_HEADS, LATENT, dtype=torch.bfloat16, device=device
        )
        inputs = [query, block_tables, seq_lens, out]
        runner = _cpu_runner(
            dtype,
            page_size,
            max_seq_len,
            kv_cache=kv_cache,
            bmm1_scale=1.0 / (192**0.5),
            bmm2_scale=bmm2_scale,
        )
        label = (facts, dtype, page_size, rows, q_len, max_seq_len, bmm2_scale)
        domain = runner._domain(inputs, facts[1], arch=facts[0])
        is_splitkv = domain == cake_module.DOMAIN_BF16_SPLITKV
        assert is_splitkv == (max_seq_len > 1024), label
        if domain is None:
            # BF16 page 64 beyond a half-value split, CLC with bmm2_scale != 1.
            assert page_size == 64 or bmm2_scale != 1.0, label
            assert runner._launch_plan(inputs) is None, label
            continue
        plan = runner._launch_plan(inputs)
        assert plan is not None, label
        roles = {
            query.data_ptr(): "query",
            kv_cache.data_ptr(): "kv_cache",
            block_tables.data_ptr(): "block_tables",
            seq_lens.data_ptr(): "seq_lens",
            out.data_ptr(): "out",
            lut.data_ptr(): "fp8_table",
        }
        runs.clear()
        runner._plan = plan
        assert runner.forward(inputs) is out
        resolved = _normalized_runs(runs, roles)
        runs.clear()
        runner._plan = None
        runner.forward(inputs)
        dispatched = _normalized_runs(runs, roles)
        assert resolved == dispatched, label
        assert [name for name, _ in resolved] == list(
            cake_module.launch_families(domain, facts[0], facts[1], rows)
        ), label
        checked += 1
        splitkv += is_splitkv
    assert checked >= 6 and splitkv >= 2


@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("geometry", _STATELESS_GEOMETRIES)
def test_memoized_cake_call_matches_explicit_cake_bitwise(
    monkeypatch, clean_tuner, geometry, q_len
):
    """A repeated call reuses the memoized decision and its resolved launch
    without a second lookup; its output equals, bit for bit, the first call's,
    explicit ``backend="cake"`` on the same values and the stateless
    dispatcher, with padded requests."""
    _require_cake_arch()
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    _skip_unserved_geometry(geometry, "plain", arch, num_sms)
    case = _parity_case(geometry, q_len)
    _pad(case["seq_lens"], q_len)
    selected = _force_cake(monkeypatch, case)
    launches = _spy_cake_forward(monkeypatch)
    results = _spy_stage_two(monkeypatch)
    first = _decode(dict(case, out=_nan_out(case)))
    looked_up = len(selected)
    second = _decode(dict(case, out=_nan_out(case)))
    third = _decode(dict(case, out=_nan_out(case)))
    assert len(selected) == looked_up == 1
    # Only the first call runs the second stage: the repeated calls reuse the
    # decision of both stages (``_CAKE_MLA_AUTO_MEMO``) and run neither.
    assert [_name(runner) for runner, _ in results] == ["cake"]
    assert len(launches) == 3 and all(runner._plan is not None for runner in launches)
    explicit = _explicit_cake(case, case["seq_lens"], case["block_tables"])
    stateless = cake_module.trtllm_mla_blackwell_decode(
        case["query"],
        case["kv_cache"],
        case["block_tables"],
        case["seq_lens"],
        case["max_seq_len"],
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=QK_DIM - LATENT,
        sparse_mla_top_k=0,
        out=_nan_out(case),
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=case["bmm2_scale"],
        sinks=None,
        skip_softmax_threshold_scale_factor=None,
        enable_pdl=None,
        uses_shared_paged_kv_idx=True,
        lse=None,
        return_lse=False,
        cum_seq_lens_q=None,
        max_q_len=None,
        multi_ctas_kv_counter_buffer=None,
        sparse_mla_top_k_lens=None,
        enable_dcp=False,
        stateless=True,
    )
    torch.cuda.synchronize()
    for label, output in (("second", second), ("third", third)):
        assert torch.equal(output, first), label
    assert torch.equal(first, explicit)
    assert torch.equal(first, stateless)
    _assert_live_and_padded(first, case, f"{geometry} memoized")


def _count_tuning_configs(monkeypatch):
    """Counts the first stage's tuning-config builds."""
    counts = [0]
    original = core._build_mla_decode_tuning_config

    def build(*args, **kwargs):
        counts[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "_build_mla_decode_tuning_config", build)
    return counts


def _spy_runner_builds(monkeypatch):
    """Records the class of every TRTLLM-GEN, CuTe DSL and Cake runner a call
    builds."""
    built = []
    for cls in (
        core._BatchMLAPagedAttentionTrtllmGenBackend,
        core._BatchMLAPagedAttentionCuteDslModularBackend,
        core._BatchMLAPagedAttentionCuteDslMonolithicBackend,
    ):

        def build(*args, _cls=cls, _original=cls.from_functional, **kwargs):
            built.append(_cls)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cls, "from_functional", build)
    original_init = core._CakeTrtllmMlaBlackwellRunner.__init__

    def init(self, **kwargs):
        built.append(core._CakeTrtllmMlaBlackwellRunner)
        original_init(self, **kwargs)

    monkeypatch.setattr(core._CakeTrtllmMlaBlackwellRunner, "__init__", init)
    return built


def _count_eligibility_checks(monkeypatch):
    """Counts the Cake candidate gate (``cake_auto_candidate_open``) and
    eligibility checks (``_cake_mla_auto_candidate_reason``) and the memo
    lookups of the dispatcher."""
    counts = {"gate": 0, "reason": 0, "memo": 0}
    original_gate = core.cake_auto_candidate_open
    original_reason = core._cake_mla_auto_candidate_reason
    original_get = core._mla_cake_auto_memo_get

    def gate(runner_class_name):
        counts["gate"] += 1
        return original_gate(runner_class_name)

    def reason(**kwargs):
        counts["reason"] += 1
        return original_reason(**kwargs)

    def get(*args, **kwargs):
        counts["memo"] += 1
        return original_get(*args, **kwargs)

    monkeypatch.setattr(core, "cake_auto_candidate_open", gate)
    monkeypatch.setattr(core, "_cake_mla_auto_candidate_reason", reason)
    monkeypatch.setattr(core, "_mla_cake_auto_memo_get", get)
    return counts


def _spy_runner_calls(monkeypatch):
    """Records ``(runner class, tactic)`` of every runner a call runs."""
    executed = []
    original = TunableRunner.__call__

    def call(self, inputs, **kwargs):
        executed.append((type(self), kwargs.get("tactic")))
        return original(self, inputs, **kwargs)

    monkeypatch.setattr(TunableRunner, "__call__", call)
    return executed


@pytest.mark.parametrize("cake_wins", [True, False])
def test_tuned_auto_selects_once_then_reuses_the_decision(
    monkeypatch, fresh_tuner, cake_wins
):
    """After tuning, the first call of a request builds the tuning config and
    looks up both stages once; the repeated calls reuse the decision of both
    stages (no tuning config, no lookup, no Cake gate or eligibility check),
    build only the runner it selected and run exactly that runner and tactic:
    Cake where it measured faster, equal to explicit ``backend="cake"`` bit for
    bit, and the first stage's winner where Cake lost this bucket but won
    another one."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=64)
    _offer_case(monkeypatch, case)
    monkeypatch.setattr(
        core.AutoTuner,
        "_profile_single_kernel",
        _profile_times(
            {
                "_BatchMLAPagedAttentionTrtllmGenBackend": 3.0,
                core._CAKE_MLA_RUNNER: 1.0 if cake_wins else 9.0,
            }
        ),
    )
    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        _decode(case)
    if not cake_wins:
        # Cake won elsewhere, so the second stage runs for this unit too.
        _hold_cake_entry(fresh_tuner)
    lookups = _count_margin_lookups(monkeypatch)
    calls = _spy_choose_one(monkeypatch)
    configs = _count_tuning_configs(monkeypatch)
    results = _spy_stage_two(monkeypatch)
    executed = _spy_runner_calls(monkeypatch)
    checks = _count_eligibility_checks(monkeypatch)
    built = _spy_runner_builds(monkeypatch)
    outputs, builds = [], []
    with autotune(False):
        for _ in range(3):
            built.clear()
            outputs.append(_decode(dict(case, out=_nan_out(case))))
            builds.append(list(built))
    torch.cuda.synchronize()
    assert lookups[0] == 1 and configs[0] == 1
    assert checks == {"gate": 1, "reason": 1, "memo": 3}
    assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP]
    ((runner, tactic),) = results
    assert executed == [(type(runner), tactic)] * 3
    assert builds[1] == builds[2] == [type(runner)]
    if cake_wins:
        assert _name(runner) == "cake"
        explicit = _explicit_cake(case, case["seq_lens"], case["block_tables"])
        torch.cuda.synchronize()
        for output in outputs:
            assert torch.equal(output, explicit)
    else:
        first_stage = calls[0][2]
        assert runner is first_stage[0] and tactic == first_stage[1]
        assert _name(runner) != "cake"
    for output in outputs:
        _assert_matches_reference(output, case, f"memoized {_name(runner)}")


def test_auto_without_a_cake_result_memoizes_the_first_stage(monkeypatch, fresh_tuner):
    """Tuned where Cake lost, with no Cake result in any winner source, the
    Cake candidate gate is closed. The first call of a listed batch evaluates
    it once and memoizes the first stage's selection; the repeated calls make
    no selection, build no tuning config, evaluate neither the gate nor the
    eligibility check, build only the selected runner and run its tactic. A
    call with a new key under the same state selects without the gate. A
    Cake result added elsewhere (a new winner-source generation) opens the
    gate: the next call evaluates the gate and the check and selects through
    both stages (Cake lost here: the first stage's winner again)."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=64)
    _offer_case(monkeypatch, case)
    monkeypatch.setattr(
        core.AutoTuner,
        "_profile_single_kernel",
        _profile_times(
            {
                "_BatchMLAPagedAttentionTrtllmGenBackend": 3.0,
                core._CAKE_MLA_RUNNER: 9.0,
            }
        ),
    )
    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        _decode(case)
    assert not fresh_tuner.may_select_runner_class(core._CAKE_MLA_RUNNER)
    lookups = _count_margin_lookups(monkeypatch)
    calls = _spy_choose_one(monkeypatch)
    configs = _count_tuning_configs(monkeypatch)
    results = _spy_stage_two(monkeypatch)
    executed = _spy_runner_calls(monkeypatch)
    checks = _count_eligibility_checks(monkeypatch)
    built = _spy_runner_builds(monkeypatch)
    outputs, builds = [], []
    with autotune(False):
        for _ in range(3):
            built.clear()
            outputs.append(_decode(dict(case, out=_nan_out(case))))
            builds.append(list(built))
    torch.cuda.synchronize()
    assert configs[0] == 1 and lookups[0] == 0 and not results
    assert checks == {"gate": 1, "reason": 0, "memo": 3}
    ((op, _, (runner, tactic)),) = calls
    assert op == core._MLA_DECODE_OP and _name(runner) != "cake"
    assert executed == [(type(runner), tactic)] * 3
    assert builds[1] == builds[2] == [type(runner)]
    ((entry),) = core._CAKE_MLA_AUTO_MEMO.values()
    assert entry[3][2] is None
    for output in outputs:
        _assert_matches_reference(output, case, "memoized first stage")

    # A new key under the same tuner state (here another bmm1_scale) selects,
    # but reuses the last gate result: no gate.
    scaled = dict(case, bmm1_scale=case["bmm1_scale"] / 2)
    checks.update(gate=0, reason=0, memo=0)
    executed.clear()
    with autotune(False):
        output = _decode(dict(scaled, out=_nan_out(case)))
    torch.cuda.synchronize()
    assert checks == {"gate": 0, "reason": 0, "memo": 1}
    assert len(calls) == 2 and configs[0] == 2 and not results
    assert len(core._CAKE_MLA_AUTO_MEMO) == 2
    assert executed == [(type(runner), tactic)]
    _assert_matches_reference(output, scaled, "new key, same state")

    _hold_cake_entry(fresh_tuner)
    calls.clear()
    executed.clear()
    checks.update(gate=0, reason=0, memo=0)
    with autotune(False):
        output = _decode(dict(case, out=_nan_out(case)))
    torch.cuda.synchronize()
    assert checks == {"gate": 1, "reason": 1, "memo": 1}
    assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP]
    assert lookups[0] == 1 and len(results) == 1
    assert executed == [(type(runner), tactic)]
    _assert_matches_reference(output, case, "gate opened")


# --- A caller changing max_seq_len on every call (seq_lens clamped to it) ---

# (dtype, page size, batch, top of the max_seq_len bucket): the BF16 CLC
# program with the work order, the FP8 page-64 program, whose split count reads
# the KV tile count (16 rows, bucket 1024: 2 to 4 splits on 148 or 152 SMs;
# above KV tile 8 the SM count caps it at 4), and BF16 split-KV (4 rows,
# bucket 4096) on the devices whose split-KV range covers that bucket, whose
# split count reads it too (the "edges" values cross into bucket 2048,
# VQUARTER there).
_VARYING_GEOMETRIES = {
    "bf16_p32_clc": (torch.bfloat16, 32, 128, 1024),
    "fp8_p64": (torch.float8_e4m3fn, 64, 16, 1024),
    "bf16_p32_splitkv": (torch.bfloat16, 32, 4, 4096),
}


def _varying_max_seq_lens(pattern, top):
    """``inc``: the last 121 values of the bucket of ``top``, in order;
    ``rand``: 40 seeded draws from that bucket; ``edges``: values on both
    sides of the bucket's lower edge and of KV tile edges, revisited."""
    if pattern == "inc":
        return list(range(top - 120, top + 1))
    if pattern == "rand":
        generator = torch.Generator().manual_seed(top)
        draws = torch.randint(top // 2 + 1, top + 1, (40,), generator=generator)
        return draws.tolist()
    low = top // 2
    return [top, low + 1, low, low + 128, low + 129, top - 128, top - 127, low, top]


def _offer_max_seq_lens(monkeypatch, case, max_seq_lens):
    """Offer the unit of ``case`` (its tuning bucket of batches) for the
    bucket of every value of ``max_seq_lens``, on each domain Cake launches
    there (two where the device's split-KV range starts inside the bucket)."""
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    query = case["query"]
    page_size = case["kv_cache"].shape[-2]
    inputs = [query, case["block_tables"], case["seq_lens"], _nan_out(case)]
    units = set()
    first, last = _bucket_range(query.shape[0])
    for max_seq_len in max_seq_lens:
        runner = _cpu_runner(
            query.dtype, page_size, max_seq_len, bmm2_scale=case["bmm2_scale"]
        )
        domain = runner._domain(inputs, num_sms, arch=arch)
        assert domain is not None, max_seq_len
        units.add((domain, _round_to_seq_len_bucket(max_seq_len), first, last))
    key = (
        arch,
        num_sms,
        core._cake_mla_dtype_name(query.dtype),
        page_size,
        query.shape[1],
        False,
    )
    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {key: tuple(sorted(units))})


def _varying_call(case, max_seq_len):
    """``case`` with ``max_seq_len`` and its seq_lens clamped to it."""
    return dict(
        case, max_seq_len=max_seq_len, seq_lens=case["seq_lens"].clamp(max=max_seq_len)
    )


def _unrecorded(spies, function, *args):
    """``function(*args)``, dropping what the list ``spies`` recorded meanwhile."""
    marks = [len(spy) for spy in spies]
    result = function(*args)
    for spy, mark in zip(spies, marks, strict=True):
        del spy[mark:]
    return result


def _decode_selections(calls):
    return sum(op == core._MLA_DECODE_OP for op, _, _ in calls)


def _memo_fields(max_seq_lens):
    device_index = torch.cuda.current_device()
    return {core._mla_auto_memo_seq_len(value, device_index) for value in max_seq_lens}


def _check_varying_max_seq_lens(monkeypatch, case, max_seq_lens):
    """Force Cake for ``case``, offered for every bucket of ``max_seq_lens``,
    and call it with each of them (seq_lens clamped), first with every Cake
    memo cleared before each call, then with the memos: the second pass
    resolves the first's Cake launches, selects once per memo field, and every
    output equals explicit ``backend="cake"`` bit for bit. The first stage is
    stubbed to one selection here, so the runner check shows only that every
    call runs Cake; the launch and output checks carry the evidence. Returns
    the resolved launches."""
    _force_cake(monkeypatch, case)
    _offer_max_seq_lens(monkeypatch, case, max_seq_lens)
    calls = _spy_choose_one(monkeypatch)
    configs = _count_tuning_configs(monkeypatch)
    results = _spy_stage_two(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)
    executed = _spy_runner_calls(monkeypatch)
    spies = (calls, results, launches, executed)
    variants = [_varying_call(case, value) for value in max_seq_lens]

    def run(memoized):
        _clear_all_cake_mla_memos()
        for spy in spies:
            spy.clear()
        configs[0] = 0
        with autotune(False):
            for variant in variants:
                if not memoized:
                    _clear_all_cake_mla_memos()
                output = _decode(dict(variant, out=_nan_out(case)))
                explicit = _unrecorded(
                    spies,
                    _explicit_cake,
                    variant,
                    variant["seq_lens"],
                    variant["block_tables"],
                )
                torch.cuda.synchronize()
                label = (memoized, variant["max_seq_len"])
                assert torch.equal(output, explicit), label
        assert len(launches) == len(variants)
        assert all(runner._plan is not None for runner in launches)
        plans = [(runner._plan[0], runner._plan[3]) for runner in launches]
        return list(executed), plans

    selected, plans = run(memoized=False)
    assert configs[0] == _decode_selections(calls) == len(results) == len(variants)
    fields = _memo_fields(max_seq_lens)
    assert run(memoized=True) == (selected, plans)
    assert len(core._CAKE_MLA_AUTO_MEMO) == len(fields)
    assert configs[0] == _decode_selections(calls) == len(results) == len(fields)
    assert all(_name(runner) == "cake" for runner, _ in results)
    assert len(selected) == len(variants)
    assert all(cls is core._CakeTrtllmMlaBlackwellRunner for cls, _ in selected)
    return [launch for launch, _ in plans]


def _check_fp8_split_counts(launches, max_seq_lens, pattern):
    """The FP8 page-64 launches split each row as ``plan_num_split`` gives for
    its max_seq_len, and more than one split count occurs except in ``inc``
    (one KV tile count): a memo field without the KV tile count would reuse
    one launch for all."""
    _, num_sms = cake_module._device_facts(torch.cuda.current_device())
    splits = []
    for value, launch in zip(max_seq_lens, launches, strict=True):
        assert launch.kind == cake_module._LAUNCH_FP8_PAGE64, value
        work_items = launch.scratch[0]
        expected = cake_module.plan_num_split(work_items, value, num_sms)
        assert launch.scratch == (work_items, expected), value
        splits.append(expected)
    if pattern != "inc":
        assert len(set(splits)) > 1, splits


def _check_splitkv_launches(launches, max_seq_lens):
    """4 query rows launch split-KV exactly from the device's split-KV range
    start on, with the split count ``plan_kv_split`` gives for their
    max_seq_len, and VQUARTER below it."""
    arch, num_sms = cake_module._device_facts(torch.cuda.current_device())
    limits = cake_module._SPLITKV_RANGE.get((arch, num_sms))
    for value, launch in zip(max_seq_lens, launches, strict=True):
        splitkv = limits is not None and value >= limits[0] and limits[1] >= 4
        assert (launch.kind == cake_module._LAUNCH_SPLITKV) == splitkv, value
        if splitkv:
            split = cake_module.plan_kv_split(4, value, num_sms)
            assert launch.scratch == (4, split), value


@pytest.mark.parametrize("pattern", ["inc", "rand", "edges"])
@pytest.mark.parametrize("geometry", sorted(_VARYING_GEOMETRIES))
def test_auto_reuses_the_decision_across_max_seq_len_of_a_bucket(
    monkeypatch, clean_tuner, geometry, pattern
):
    """A caller that changes max_seq_len on every call (seq_lens clamped to
    it) selects once per value of the memo field (``_mla_auto_memo_seq_len``):
    one memoized decision, one tuning config, one first-stage selection and
    one second stage per value, none on the other calls. Every call runs Cake
    with the resolved launch the unmemoized path (every Cake memo cleared
    before each call) resolves for its max_seq_len, and its output equals,
    bit for bit, explicit ``backend="cake"`` with the same max_seq_len. The
    first stage is stubbed (one selection); the tuned test below checks the
    first stage's selection."""
    _require_cake_arch()
    dtype, page_size, batch_size, top = _VARYING_GEOMETRIES[geometry]
    max_seq_lens = _varying_max_seq_lens(pattern, top)
    case = _make_case(
        dtype=dtype, page_size=page_size, batch_size=batch_size, max_seq_len=top
    )
    launches = _check_varying_max_seq_lens(monkeypatch, case, max_seq_lens)
    if geometry == "fp8_p64":
        _check_fp8_split_counts(launches, max_seq_lens, pattern)
    if geometry == "bf16_p32_splitkv":
        _check_splitkv_launches(launches, max_seq_lens)


@pytest.mark.parametrize("splitkv_start", [None, 3000])
def test_auto_decision_follows_the_splitkv_range_across_max_seq_len(
    monkeypatch, clean_tuner, splitkv_start
):
    """A caller changing max_seq_len across the start of the device's split-KV
    range (the committed start, a bucket edge, or 3000, inside a bucket and a
    KV tile) gets one decision on each side: the calls launch the split-KV
    program from the start on and the value-split one below it, each with the
    launch and output of the unmemoized path and of explicit
    ``backend="cake"`` for its max_seq_len."""
    _require_cake_arch()
    facts = cake_module._device_facts(torch.cuda.current_device())
    if splitkv_start is not None:
        monkeypatch.setitem(cake_module._SPLITKV_RANGE, facts, (splitkv_start, 18))
    limits = cake_module._SPLITKV_RANGE.get(facts)
    if limits is None:
        pytest.skip(f"no split-KV range on {facts}")
    start = limits[0]
    max_seq_lens = [start - 129, start - 1, start, start + 1, start + 127]
    max_seq_lens += [start - 1, start, start + 128, start - 128, start + 1]
    case = _make_case(
        batch_size=4,
        page_size=32,
        max_seq_len=_round_to_seq_len_bucket(max(max_seq_lens)),
    )
    launches = _check_varying_max_seq_lens(monkeypatch, case, max_seq_lens)
    for value, launch in zip(max_seq_lens, launches, strict=True):
        splitkv = launch.kind == cake_module._LAUNCH_SPLITKV
        assert splitkv == (value >= start), value


@pytest.mark.parametrize("cake_wins", [True, False])
def test_tuned_auto_across_max_seq_len_matches_the_unmemoized_selection(
    monkeypatch, fresh_tuner, cake_wins
):
    """Tuned at max_seq_len 1024 only, a caller changing max_seq_len across KV
    tile counts and buckets (seq_lens clamped to it) runs on every call the
    runner and tactic the unmemoized path (every Cake memo cleared before
    each call) selects for that max_seq_len: a first-stage winner keyed by the
    KV tile count (monolithic CuTe DSL) or the bucket, and a Cake choice that
    beat it, are not reused for another one. It selects once per value of the
    memo field. Each output equals the unmemoized call's bit for bit where two
    unmemoized calls agree, a Cake output equals explicit ``backend="cake"``
    bit for bit, and every output matches the reference. The tuned selection
    (Cake, or the first stage's winner) differs from the one at 896, one KV
    tile fewer in the same bucket; skipped where the first stage's tuned
    winner is not monolithic CuTe DSL, whose key reads the KV tile count."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=64)
    max_seq_lens = [1024, 1000, 897, 896, 800, 769, 768, 1024, 640, 513, 512, 897]
    max_seq_lens += [300, 1024]
    _offer_max_seq_lens(monkeypatch, case, max_seq_lens)
    monkeypatch.setattr(
        core.AutoTuner,
        "_profile_single_kernel",
        _profile_times(
            {
                "_BatchMLAPagedAttentionTrtllmGenBackend": 3.0,
                core._CAKE_MLA_RUNNER: 1.0 if cake_wins else 9.0,
            }
        ),
    )
    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        _decode(case)
    if not cake_wins:
        # Cake won elsewhere, so the second stage runs for this unit too.
        _hold_cake_entry(fresh_tuner)
    calls = _spy_choose_one(monkeypatch)
    configs = _count_tuning_configs(monkeypatch)
    executed = _spy_runner_calls(monkeypatch)
    variants = [_varying_call(case, value) for value in max_seq_lens]

    def run(memoized):
        _clear_all_cake_mla_memos()
        calls.clear()
        executed.clear()
        configs[0] = 0
        outputs = []
        with autotune(False):
            for variant in variants:
                if not memoized:
                    _clear_all_cake_mla_memos()
                outputs.append(_decode(dict(variant, out=_nan_out(case))))
        torch.cuda.synchronize()
        assert len(executed) == len(variants)
        return list(executed), outputs

    selected, expected = run(memoized=False)
    assert configs[0] == _decode_selections(calls) == len(variants)
    first_stage = [
        (type(runner), tactic)
        for op, _, (runner, tactic) in calls
        if op == core._MLA_DECODE_OP
    ]
    repeated = run(memoized=False)[1]
    memoized, outputs = run(memoized=True)
    fields = _memo_fields(max_seq_lens)
    assert len(core._CAKE_MLA_AUTO_MEMO) == len(fields)
    assert configs[0] == _decode_selections(calls) == len(fields)
    assert memoized == selected
    for variant, (runner_class, _), output, first, second in zip(
        variants, selected, outputs, expected, repeated, strict=True
    ):
        label = f"max_seq_len {variant['max_seq_len']} ({runner_class.__name__})"
        if torch.equal(first, second):
            assert torch.equal(output, first), label
        if runner_class is core._CakeTrtllmMlaBlackwellRunner:
            explicit = _explicit_cake(
                variant, variant["seq_lens"], variant["block_tables"]
            )
            torch.cuda.synchronize()
            assert torch.equal(output, explicit), label
        _assert_matches_reference(output, variant, label)
    # The selection changes inside the bucket of 1024 (896 has one KV tile
    # fewer), so a memo field without the KV tile count would fail above.
    tuned, fewer_tiles = max_seq_lens.index(1024), max_seq_lens.index(896)
    if cake_wins:
        assert selected[tuned][0] is core._CakeTrtllmMlaBlackwellRunner
    else:
        assert selected[tuned] == first_stage[tuned]
    winner = core._BatchMLAPagedAttentionCuteDslMonolithicBackend
    if first_stage[tuned][0] is not winner:
        pytest.skip(
            f"the first stage's tuned winner is {first_stage[tuned][0].__name__}, "
            "not monolithic CuTe DSL: the KV tile count does not change it"
        )
    assert first_stage[fewer_tiles] != first_stage[tuned]
    assert selected[fewer_tiles] != selected[tuned]


def test_dispatcher_keeps_no_closure_cells():
    """The dispatcher closes over none of its locals (the second stage's Cake
    factory is a ``functools.partial``), so no call of any backend pays for
    closure cells. No GPU needed."""
    code = core._mla_with_kv_cache_impl.__code__
    assert code.co_cellvars == () and code.co_freevars == ()


@pytest.mark.parametrize(
    "change",
    [
        "new_tuner",
        "autotune_override",
        "offer_table",
        "tuning_again",
        "skip_ops",
        "switch_off",
        "replay_diagnostics",
        "autotune_v2_context",
    ],
)
def test_memoized_auto_decision_follows_the_tuner_and_settings(
    monkeypatch, fresh_tuner, change
):
    """A memoized Cake decision is not reused for another tuner, inside an
    ``autotune()`` override, for another offer table or after tuning changed
    the winners: the next call selects through both stages, and the call after
    it reuses the new decision (after tuning again with Cake slower: the first
    stage's winner). While ``skip_ops`` names the op, under replay
    diagnostics (which record every call) or inside an ``autotune_v2``
    context, every call selects and nothing is reused, with the selection of
    a call without the memo (no Cake when skipped); once ``skip_ops`` is
    reverted, the memoized decision is reused again. The Cake switch is part
    of the memo state: switched off at runtime, the first call selects without
    Cake and the next reuses that decision; switched on again, the first call
    selects Cake again and the next reuses it."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=64)
    _offer_case(monkeypatch, case)
    times = {
        "_BatchMLAPagedAttentionTrtllmGenBackend": 3.0,
        core._CAKE_MLA_RUNNER: 1.0,
    }
    monkeypatch.setattr(core.AutoTuner, "_profile_single_kernel", _profile_times(times))
    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        _decode(case)
    with autotune(False):
        _decode(dict(case, out=_nan_out(case)))
    assert len(core._CAKE_MLA_AUTO_MEMO) == 1
    context: contextlib.AbstractContextManager = autotune(False)
    report = None
    # First-stage selections of the two calls below, and whether they run
    # Cake (None: not asserted).
    selections, cake = 1, True
    if change == "new_tuner":
        tuner = AutoTuner()
        # The same winners, copied without advancing the winner-source
        # generation: only the tuner differs.
        dict.update(tuner.profiling_cache, fresh_tuner.profiling_cache)
        monkeypatch.setattr(AutoTuner, "_instance", tuner)
    elif change == "autotune_override":
        context = autotune(False, tuning_buckets=TUNING_BUCKETS)
    elif change == "offer_table":
        monkeypatch.setattr(
            core, "_CAKE_MLA_AUTO_OFFERS", dict(core._CAKE_MLA_AUTO_OFFERS)
        )
    elif change == "tuning_again":
        winners = fresh_tuner._winner_cache()
        for key in [
            key for key in winners if key.custom_op == core._MLA_CAKE_MARGIN_OP
        ]:
            del winners[key]
        times[core._CAKE_MLA_RUNNER] = 9.0
        with autotune(True, tuning_buckets=TUNING_BUCKETS):
            _decode(case)
        # Cake won elsewhere, so the second stage runs for this unit too.
        _hold_cake_entry(fresh_tuner)
        cake = False
    elif change == "skip_ops":
        context = autotune(False, skip_ops=core._MLA_DECODE_OP)
        selections, cake = 2, False
    elif change == "switch_off":
        # The switch is part of the memo state: the first call selects
        # without Cake and memoizes that, the second reuses it.
        monkeypatch.setenv(DISABLE_ENV, "1")
        selections, cake = 1, False
    elif change == "replay_diagnostics":
        report = ReplayCoverage()
        context = autotune_v2(
            mode="replay", persistent_cache=False, replay_report=report
        )
        selections, cake = 2, None
    else:
        context = autotune_v2(mode="replay", persistent_cache=False)
        selections, cake = 2, None
    calls = _spy_choose_one(monkeypatch)
    launches = _spy_cake_forward(monkeypatch)
    with context:
        outputs = [_decode(dict(case, out=_nan_out(case))) for _ in range(2)]
    torch.cuda.synchronize()
    assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP] * selections
    if cake is not None:
        assert len(launches) == (2 if cake else 0), change
    if report is not None:
        assert [record["op"] for record in report.records] == [core._MLA_DECODE_OP] * 2
    for output in outputs:
        _assert_matches_reference(output, case, change)
    if change in ("skip_ops", "switch_off"):
        monkeypatch.delenv(DISABLE_ENV, raising=False)
        calls.clear()
        launches.clear()
        with autotune(False):
            for _ in range(2):
                _decode(dict(case, out=_nan_out(case)))
        # Switched off, the memoized first-stage decision replaced the Cake
        # decision (one entry per key): switched on again, the first call
        # selects Cake again and the second reuses it.
        assert len(calls) == (1 if change == "switch_off" else 0), change
        assert len(launches) == 2, change


def _record_runner_calls(monkeypatch):
    """Records ``(runner class, tactic)`` of every runner a call runs, and
    launches nothing."""
    executed = []

    def call(self, inputs, **kwargs):
        executed.append((type(self), kwargs.get("tactic")))

    monkeypatch.setattr(TunableRunner, "__call__", call)
    return executed


def _strided_out(case):
    shape = (*case["query"].shape[:-1], 2 * LATENT)
    return torch.empty(shape, dtype=torch.bfloat16, device="cuda")[..., :LATENT]


def test_memoized_cake_decision_runs_the_first_stage_for_strided_out(
    monkeypatch, fresh_tuner
):
    """A memoized Cake decision reused for a call whose ``out`` is not
    contiguous runs the first stage's runner and tactic of that decision, as
    the eligibility check makes such a call do: no selection, no eligibility
    check, no Cake runner, and the memoized decision stays."""
    _require_cake_arch()
    case = _make_case(batch_size=4, page_size=64)
    _offer_case(monkeypatch, case)
    monkeypatch.setattr(
        core.AutoTuner,
        "_profile_single_kernel",
        _profile_times(
            {
                "_BatchMLAPagedAttentionTrtllmGenBackend": 3.0,
                core._CAKE_MLA_RUNNER: 1.0,
            }
        ),
    )
    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        _decode(case)
    calls = _spy_choose_one(monkeypatch)
    results = _spy_stage_two(monkeypatch)
    with autotune(False):
        _decode(dict(case, out=_nan_out(case)))
    ((runner, _),) = results
    assert _name(runner) == "cake"
    ((_, _, first_stage),) = calls
    calls.clear()
    checks = _count_eligibility_checks(monkeypatch)
    built = _spy_runner_builds(monkeypatch)
    executed = _record_runner_calls(monkeypatch)
    with autotune(False):
        _decode(dict(case, out=_strided_out(case)))
    assert not calls and len(results) == 1
    assert checks == {"gate": 0, "reason": 0, "memo": 1}
    assert built == [type(first_stage[0])]
    assert executed == [(type(first_stage[0]), first_stage[1])]
    ((entry),) = core._CAKE_MLA_AUTO_MEMO.values()
    assert entry[3][2] is not None


def test_rejected_listed_calls_memoize_only_what_the_key_decides(
    monkeypatch, clean_tuner
):
    """A listed batch the eligibility check rejects for a reason its memo key
    decides (here a static check) memoizes the first stage's selection: the
    repeated calls neither check nor select, and a bmm2_scale of the other
    sign is another key. Rejected for a strided ``out``, nothing is memoized
    and every call checks and selects (the gate result of the state is
    reused). A call the scalar checks reject (LSE)
    builds no memo key and does not evaluate the Cake candidate gate."""
    _require_cake_arch()
    case = _make_case(batch_size=4)
    _offer_case(monkeypatch, case)
    monkeypatch.setattr(
        core.AutoTuner, "may_select_runner_class", lambda self, name: True
    )
    calls = _spy_choose_one(monkeypatch)
    results = _spy_stage_two(monkeypatch)
    checks = _count_eligibility_checks(monkeypatch)
    executed = _record_runner_calls(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(core, "_cake_mla_auto_static_reason", lambda *args: "static")
        with autotune(False):
            for _ in range(3):
                _decode(dict(case, bmm2_scale=0.0))
        assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP]
        assert checks == {"gate": 1, "reason": 1, "memo": 3} and not results
        ((entry),) = core._CAKE_MLA_AUTO_MEMO.values()
        assert entry[3][2] is None
        ((_, _, (runner, tactic)),) = calls
        assert executed == [(type(runner), tactic)] * 3
        with autotune(False):
            _decode(dict(case, bmm2_scale=-0.0))
        # Another key under the same state: the last gate result is reused.
        assert len(calls) == 2 and checks["reason"] == 2 and checks["gate"] == 1
        assert len(core._CAKE_MLA_AUTO_MEMO) == 2

    core._CAKE_MLA_AUTO_MEMO.clear()
    calls.clear()
    checks.update(gate=0, reason=0, memo=0)
    with autotune(False):
        for _ in range(2):
            _decode(dict(case, out=_strided_out(case)))
    assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP] * 2
    # The gate result of the same state is memoized (gate 0, was 2).
    assert checks == {"gate": 0, "reason": 2, "memo": 2} and not results
    assert not core._CAKE_MLA_AUTO_MEMO

    checks.update(gate=0, reason=0, memo=0)
    with autotune(False):
        _decode(dict(case, return_lse=True))
    # The scalar checks run before the gate: LSE calls pay for neither.
    assert checks == {"gate": 0, "reason": 0, "memo": 0} and not results
    assert not core._CAKE_MLA_AUTO_MEMO


def test_unoffered_calls_build_no_cake_runner(monkeypatch, clean_tuner):
    """With Cake entries held, an untuned call whose batch is outside every
    listed range, or whose unit is not listed, makes the first selection only:
    no Cake eligibility check, no second stage, no Cake runner."""
    _require_cake_arch()
    case = _make_case(batch_size=4)
    _offer_case(monkeypatch, case)
    monkeypatch.setattr(
        core.AutoTuner, "may_select_runner_class", lambda self, name: True
    )
    built, reasons = [], []
    original_init = core._CakeTrtllmMlaBlackwellRunner.__init__
    original_reason = core._cake_mla_auto_candidate_reason

    def init(self, **kwargs):
        built.append(self)
        original_init(self, **kwargs)

    def reason(**kwargs):
        reasons.append(kwargs)
        return original_reason(**kwargs)

    monkeypatch.setattr(core._CakeTrtllmMlaBlackwellRunner, "__init__", init)
    monkeypatch.setattr(core, "_cake_mla_auto_candidate_reason", reason)
    results = _spy_stage_two(monkeypatch)
    calls = _spy_choose_one(monkeypatch)
    beyond = _make_case(batch_size=_bucket_range(4)[1] + 1)
    with autotune(False):
        output = _decode(beyond)
    assert not built and not reasons and not results
    assert not core._CAKE_MLA_AUTO_MEMO
    assert [op for op, _, _ in calls] == [core._MLA_DECODE_OP]
    _assert_matches_reference(output, beyond, "outside the listed batches")

    monkeypatch.setattr(core, "_CAKE_MLA_AUTO_OFFERS", {})
    with autotune(False):
        output = _decode(case)
    assert not built and not reasons and not results
    assert not core._CAKE_MLA_AUTO_MEMO
    _assert_matches_reference(output, case, "unlisted unit")

    # The listed batch joins the second stage (a lookup; nothing is tuned),
    # and its decision is memoized.
    _offer_case(monkeypatch, case)
    with autotune(False):
        _decode(case)
    assert reasons and results
    assert len(core._CAKE_MLA_AUTO_MEMO) == 1
