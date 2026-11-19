"""Cake in the second selection stage of uniform-Q MLA decode ``backend="auto"``.

``trtllm_batch_decode_with_kv_cache_mla`` selects in two stages: the TRTLLM-GEN /
CuTe DSL selection, unchanged, then, only for the units the device's offer table
lists, a comparison of that winner with Cake's generated Blackwell route. These
tests check that the first stage keeps its op, runners, config and cache entries;
that Cake joins only the second stage, only for listed units, requests it serves,
while the tuner may select it and never while an autotune process group tunes;
that the committed offer table is empty and well formed; that a tuning bucket
offers Cake only when every batch the bucket serves is listed and launches the
programs of the profiled batch; that Cake is profiled only against a first-stage
winner and a Cake choice stays valid only while that winner does; that an
untuned call never selects Cake and every Cake miss runs the first stage's
selection; that a tuned call is correct whichever candidate wins; that a forced
Cake selection matches TRTLLM-GEN (bmm2_scale, q_len > 1) on live rows while
padded rows output exact zeros; that it reads no host state and launches only
the kernels of explicit ``backend="cake"``; and that it stays correct after input
updates that bypass torch's version counter and under CUDA-graph replay with
updated inputs (also two graphs replayed on two streams at once).

Tests that need Cake offered replace the offer table with a synthetic one: the
committed table is derived from measurements and offers nothing until then.
"""

import pytest
import torch

import flashinfer
from flashinfer.autotuner import (
    AutoTuner,
    DynamicDim,
    OptimizationProfile,
    ProfilingCacheKey,
    TunableRunner,
    autotune,
    cake_auto_candidate_open,
)
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
# The first-stage winner a forced or stubbed Cake choice beat.
FORCED = ("forced first-stage winner", -1)
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
    makes; the call launches nothing."""
    calls = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        calls.append((op, [_name(runner) for runner in runners], config))
        return _NoLaunch(), -1

    with monkeypatch.context() as patch:
        patch.setattr(core.AutoTuner, "choose_one", choose_one)
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
        domain = runner._domain(inputs, num_sms) or cake_module.DOMAIN_BF16_VQUARTER
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
        (core._MLA_CAKE_MARGIN_OP, ["incumbent", "cake"])
    ]
    first, second = enabled[0][2], enabled[1][2]
    assert (
        second.dynamic_tensor_specs[0].map_to_tuning_buckets
        is first.dynamic_tensor_specs[0].map_to_tuning_buckets
    )
    assert set(second.dynamic_tensor_specs[0].gen_tuning_buckets) <= set(
        first.dynamic_tensor_specs[0].gen_tuning_buckets
    )


def test_committed_offer_table_runs_the_first_stage_only(monkeypatch, fresh_tuner):
    """With the committed (empty) offer table a tuned call makes only the
    first selection."""
    _require_cake_arch()
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
        (core._MLA_CAKE_MARGIN_OP, ["incumbent", "cake"])
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
    and every batch on the listed domain of the listed device."""
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
            runner = _cpu_runner(dtype, page_size, seq_bucket)
            for batch_size in range(first, last + 1):
                inputs = _cpu_inputs(dtype, batch_size, q_len)
                assert runner._domain(inputs, num_sms) == domain, (unit, batch_size)


def test_offer_table_well_formed():
    """The committed offer table and the synthetic one of these tests are well
    formed. No GPU needed."""
    _check_offer_table(core._CAKE_MLA_AUTO_OFFERS)
    _check_offer_table(_SYNTHETIC_OFFERS)


def test_committed_offer_table_offers_nothing(monkeypatch, cake_families):
    """Until measurements derive it, the committed table is empty: no device
    offers Cake, and no Cake program or FP8 decode table is built. No GPU
    needed."""
    built, _ = cake_families
    assert core._CAKE_MLA_AUTO_OFFERS == {}
    for arch, num_sms in (("sm_100a", 148), ("sm_103a", 152)):
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
                            for domain in _DOMAINS:
                                assert not core._cake_mla_auto_offered(
                                    arch,
                                    num_sms,
                                    dtype,
                                    page_size,
                                    domain,
                                    q_len,
                                    has_sinks,
                                    max_seq_len,
                                    4,
                                )
        for page_size in (32, 64):
            runner = _staged(_cpu_runner(torch.bfloat16, page_size))
            inputs = _cpu_inputs(torch.bfloat16, 4)
            assert runner.get_valid_tactics(inputs, _bucket(4, 8)) == []
            assert not runner.validate_tactic(inputs, WINNER)
            assert not runner.supports(inputs)
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
    Profiling returns ``times`` (ms) per runner class, instead of launching."""

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
        self.ranges = ((1, 15),)

        def profile(tuner, runner, inputs, tactic, tuning_config, **kwargs):
            timed = type(runner).__name__
            if isinstance(runner, core._MlaStageOneWinnerRunner):
                timed = tactic[0]
            self.profiled.append((type(runner).__name__, timed, inputs[0].size(0)))
            return self.times[timed]

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

    def select(self, batch_size=4, *, inputs=None, tune=False, selected=None):
        """The two stages for a call; ``selected`` skips the first."""
        if inputs is None:
            inputs = _cpu_inputs(torch.bfloat16, batch_size)
        if selected is None:
            selected = self.stage_one(inputs, tune)
        self.tuner.is_tuning_mode = tune
        try:
            return core._mla_cake_stage_two(
                self.tuner,
                self.runners,
                self.config,
                inputs,
                selected,
                _cpu_runner(torch.bfloat16, 64),
                self.ranges,
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

    def cake_profiled(self):
        return sorted(
            batch
            for profiled, _, batch in self.profiled
            if profiled == core._CAKE_MLA_RUNNER
        )


@pytest.fixture
def stages(monkeypatch, fresh_tuner, cake_families, synthetic_offers):
    return _Stages(monkeypatch, fresh_tuner)


_CAKE_WINS = {
    1: (core._CAKE_MLA_RUNNER, WINNER),
    2: (core._CAKE_MLA_RUNNER, WINNER),
    4: (core._CAKE_MLA_RUNNER, WINNER),
    8: (core._CAKE_MLA_RUNNER, WINNER),
    # Bucket 16 bounds bucket 8; Cake is not offered there.
    16: ("_MlaStageOneWinnerRunner", WINNER),
}


def test_two_stages_keep_first_stage_entries(monkeypatch, stages):
    """The first stage tunes and stores exactly what it stores alone; the
    second stage adds entries of its own op only, for the offered buckets and
    the bucket after each. No GPU needed."""
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


def test_cake_not_selected_at_a_tie(stages):
    """Cake must measure faster than the first-stage winner. No GPU needed."""
    stages.times[core._CAKE_MLA_RUNNER] = stages.times["_FakeCuteDsl"]
    stages.select(4, tune=True)
    assert stages.cake_profiled() == [1, 2, 4, 8]
    assert set(stages.entries(core._MLA_CAKE_MARGIN_OP).values()) == {
        ("_MlaStageOneWinnerRunner", WINNER)
    }
    assert stages.select(4) == (stages.runners[1], -1)


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


def test_switched_off_cake_runs_the_first_stage_winner(monkeypatch, stages):
    """``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES=1`` with cached Cake entries:
    the dispatcher skips the second stage, and the first stage still holds its
    tuned winner for the buckets Cake won. No GPU needed."""
    stages.select(4, tune=True)
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert stages.tuner.may_select_runner_class(core._CAKE_MLA_RUNNER)
    assert not cake_auto_candidate_open(core._CAKE_MLA_RUNNER)
    for batch_size in (1, 4, 8):
        inputs = _cpu_inputs(torch.bfloat16, batch_size)
        assert stages.stage_one(inputs) == (stages.runners[1], -1)


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

    with autotune(True, tuning_buckets=TUNING_BUCKETS):
        tuned = _decode(case)
    tuned_calls = len(calls)
    with autotune(False):
        cached = _decode(case)

    ops = [op for op, _, _ in calls]
    assert ops[:tuned_calls] == [core._MLA_DECODE_OP, core._MLA_CAKE_MARGIN_OP]
    for op, runners, _ in calls:
        names = [_name(r) for r in runners]
        if op == core._MLA_DECODE_OP:
            assert names and "cake" not in names
        else:
            assert names == ["incumbent", "cake"]
    assert any(profiled), "the Cake candidate was never profiled"
    # The cached call makes the second selection while the tuner holds a Cake
    # winner.
    assert (core._MLA_CAKE_MARGIN_OP in ops[tuned_calls:]) == (
        clean_tuner.may_select_runner_class(core._CAKE_MLA_RUNNER)
    )
    for label, output in (("tuned", tuned), ("cached", cached)):
        _assert_matches_reference(output, case, label)


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
    cake(inputs=[*inputs[:3], direct], tactic=FORCED)
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
    assert not launches and _name(results[-1][0]) == "trtllm-gen"
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

    output = _decode(case)

    cake, inputs = selected[0]
    assert _name(cake) == "cake" and cake.supports(inputs)
    assert cake._domain(inputs, num_sms) == cake_module.DOMAIN_BF16_VQUARTER
    assert not cake.validate_tactic(inputs, FORCED)
    assert not launches
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
    # Eager warm-up of the untuned default (TRTLLM-GEN) only.
    _decode(case)
    torch.cuda.synchronize()

    builds = []

    def build(name):
        builds.append(name)
        raise RuntimeError(f"{name} built during capture")

    monkeypatch.setattr(core, "_CAKE_MLA_LOADED_FAMILIES", set())
    monkeypatch.setattr(core, "_CAKE_MLA_UNAVAILABLE_FAMILIES", set())
    monkeypatch.setattr(cake_module, "get_cake_trtllm_mla_blackwell_family", build)
    selected = _force_cake(monkeypatch, case)
    launches = _spy_cake_forward(monkeypatch)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _decode(case)
    torch.cuda.synchronize()

    cake, inputs = selected[-1]
    assert _name(cake) == "cake" and cake.supports(inputs)
    assert not builds and not launches
    assert not core._CAKE_MLA_UNAVAILABLE_FAMILIES
    case["out"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(case["out"], case, "capture fallback replay")


def _force_cake(monkeypatch, case):
    """Offer the unit of ``case``, then make the second stage pick Cake without
    tuning, as the winner over the first-stage winner ``FORCED``; the first
    stage selects its default. Records ``(runner, inputs)`` of every forced
    choice."""
    _offer_case(monkeypatch, case)
    selected = []

    def choose_one(self, op, runners, config, inputs, **kwargs):
        if op == core._MLA_CAKE_MARGIN_OP:
            selected.append((runners[-1], list(inputs)))
            return runners[-1], FORCED
        return runners[0], -1

    monkeypatch.setattr(
        core.AutoTuner, "may_select_runner_class", lambda self, name: True
    )
    monkeypatch.setattr(core.AutoTuner, "choose_one", choose_one)
    monkeypatch.setattr(
        core._MlaStageOneWinnerRunner, "call_winner", lambda self, inputs: FORCED
    )
    return selected


def _assert_cake_launched(selected, launches):
    """The last forced Cake choice was validated and its forward ran (eagerly
    or recorded into a graph), not the first stage's selection."""
    cake, inputs = selected[-1]
    assert _name(cake) == "cake" and cake.validate_tactic(inputs, FORCED)
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


# (dtype, page size, query rows). With 74 or 76 resident clusters (148 or 152
# SMs), the BF16 rows reach VQUARTER (4), VHALF (32), the CLC program in one
# wave (64; the one-wave program where one is generated for the device, see
# ``_one_wave_route``) and with the longest-first order program (128, and 300
# with a partial last order CTA); FP8 page 64 has one domain. Sinks make every
# BF16 row VQUARTER, with several work items per cluster above 18 rows. The tail
# domains never join backend="auto".
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
    sinks (the tail program), CLC with bmm2_scale != 1 and the one-wave CLC
    geometry on a device without that program."""
    dtype = _PARITY_GEOMETRIES[geometry][0]
    if "sinks" in variant and dtype != torch.bfloat16:
        pytest.skip("FP8 with sinks takes the tail program, which never joins")
    if variant in _AUTO_SKIPPED_VARIANTS:
        pytest.skip(_AUTO_SKIPPED_VARIANTS[variant])
    if variant == "bmm2_scale" and "clc" in geometry:
        pytest.skip("the CLC domain requires bmm2_scale == 1")
    if geometry == "bf16_p32_clc_one_wave" and not _one_wave_route(arch, num_sms):
        pytest.skip(f"no one-wave CLC program for {arch} with {num_sms} SMs")


def _parity_case(geometry, q_len=1, seed=0):
    dtype, page_size, rows = _PARITY_GEOMETRIES[geometry]
    return _make_case(
        dtype=dtype,
        page_size=page_size,
        batch_size=rows // q_len,
        q_len=q_len,
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
    dtype, _, rows = _PARITY_GEOMETRIES[geometry]
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
    domain = cake._domain(inputs, num_sms)
    families = cake_module.launch_families(domain, arch, num_sms, rows)
    label = f"{geometry}/{variant} (Cake {domain}, families {families})"
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
    domain = cake._domain(inputs, num_sms)
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
    assert [_name(r) for r in calls[1][1]] == ["incumbent", "cake"]
    (runner, _), first_stage = results[-1], calls[0][2]
    assert runner is first_stage[0]
    assert profiled and all(tactics == [] for tactics in profiled)
    assert core._CAKE_MLA_UNAVAILABLE_FAMILIES
    _assert_matches_reference(output, case, f"winner {_name(runner)}")
