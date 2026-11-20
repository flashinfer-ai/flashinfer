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

The Cake NVFP4 SVDQuant programs as the last automatic candidate of
``mm_nvfp4_svdquant(..., backend="auto")`` on SM100/SM103: CUTLASS stays the
first runner (the untuned and cache-miss default), Cake is listed after it
only for problems it routes, and ``FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES=1``
removes it. Outside tuning Cake is listed only while the tuner holds a Cake
entry, so an untuned call does no Cake work and lists CUTLASS only. The Cake
kernels' numerics are covered by ``test_cake_nvfp4_svdquant_gemm_backend.py``.
"""

import json

import pytest
import torch
from packaging.version import Version

from flashinfer import autotune, mm_nvfp4_svdquant
from flashinfer.autotuner import (
    AutoTuner,
    OptimizationProfile,
    ProfilingCacheKey,
    StaticDim,
    make_bucket_mapper,
)
from flashinfer import cake_nvfp4_svdquant_gemm_backend as cake_backend
from flashinfer.cake_nvfp4_svdquant_gemm_backend import cake_jit
from flashinfer.fused_moe.utils import map_to_hybrid_bucket_uncapped
from flashinfer.gemm import gemm_svdquant
from flashinfer.jit.cpp_ext import get_cuda_version
from flashinfer.utils import get_compute_capability
from tests.gemm.test_nvfp4_svdquant_gemm import (
    _make_gemm_problem,
    _skip_unless_cake_backend,
    _sqnr_db,
)

_OP = "nvfp4_svdquant_gemm"
_CUTLASS_RUNNER = "Nvfp4SvdquantGemmRunner"
_CAKE_RUNNER = "CakeNvfp4SvdquantGemmRunner"
_DISABLE_ENV = "FLASHINFER_DISABLE_CAKE_AUTO_CANDIDATES"

# (m, n, k, rank, use_bias) problems the Cake route table serves on SM100 and
# SM103 at every M of their tuning cache key: decode and prefill M, both bias
# forms, several ranks.
_ROUTED = [
    (129, 3072, 3072, 32, True),
    (256, 3072, 3072, 32, False),
    (200, 3072, 3072, 96, True),
    (6912, 3072, 3072, 32, True),
    (129, 3072, 12288, 128, False),
]
# Problems CUTLASS serves but Cake does not route: M below the 128-row
# tensor-map box. A tuning bucket of such an M may still have a Cake route.
_UNROUTED = [
    (65, 3072, 3072, 32, True),
    (100, 3072, 12288, 128, False),
]


@pytest.fixture(autouse=True)
def _fresh_tuner_cache(monkeypatch):
    monkeypatch.delenv(_DISABLE_ENV, raising=False)
    # clear_cache() keeps an autotune_v2 store that an earlier test attached;
    # while attached it opens the Cake gate. Detach it for this test only.
    monkeypatch.setattr(AutoTuner.get(), "_managed_cache", None)
    AutoTuner.get().clear_cache()
    AutoTuner.get().reset_statistics()
    yield
    AutoTuner.get().clear_cache()


class _Choices(list):
    """``(runners, selection)`` of each recorded call; ``lookups`` holds the
    ``(tuning_config, inputs)`` of the same calls."""

    def __init__(self):
        super().__init__()
        self.lookups = []


@pytest.fixture
def svdquant_choices(monkeypatch):
    """Record the runner list and selection of each ``nvfp4_svdquant_gemm`` call."""
    tuner = AutoTuner.get()
    choose_one = tuner.choose_one
    calls = _Choices()

    def recording_choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
        selection = choose_one(custom_op, runners, tuning_config, inputs, **kwargs)
        if custom_op == _OP:
            calls.append((list(runners), selection))
            calls.lookups.append((tuning_config, list(inputs)))
        return selection

    monkeypatch.setattr(tuner, "choose_one", recording_choose_one)
    return calls


def _hold_cake_entry():
    """Give the tuner a Cake winner for an op no call uses, as after a tuning
    run that Cake won elsewhere: Cake is then listed outside tuning, while
    every call of the test still misses the cache."""
    key = ProfilingCacheKey(
        custom_op="cake_candidate_gate_test",
        runner_class_name=_CAKE_RUNNER,
        runner_hash=0,
        nearest_profile=((1,),),
        extras=(),
    )
    AutoTuner.get().profiling_cache[key] = (0, None)


def _holds_cake_winner(tuner):
    return any(key.runner_class_name == _CAKE_RUNNER for key in tuner.profiling_cache)


def _exact_key_winner(runner, lookup):
    """The tactic that the tuner holds for ``runner`` under the exact cache key
    of a recorded call (its own bucket, no fallback to another bucket), or
    ``None`` when that key holds no entry."""
    tuning_config, inputs = lookup
    tuner = AutoTuner.get()
    key = AutoTuner._get_cache_key(
        _OP,
        runner,
        tuner._get_input_sizes(inputs),
        tuning_config,
        runner.get_cache_key_extras(inputs),
    )
    entry = tuner._winner_cache().get(key)
    return None if entry is None else entry[0]


def _run(p, use_bias, backend="auto", **kwargs):
    return mm_nvfp4_svdquant(
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        bias=p["bias"] if use_bias else None,
        backend=backend,
        **kwargs,
    )


def _runner_inputs(p, use_bias, out):
    """The ``nvfp4_svdquant_gemm`` runner inputs (no workspace) of ``p``."""
    return [
        p["xq"],
        p["wq"],
        p["x_sf_flat"],
        p["w_sf_flat"],
        p["alpha"],
        p["d"],
        p["l1_scaled"],
        p["bias"] if use_bias else None,
        out,
        None,
    ]


def _profile(m):
    """A tuning profile whose M bucket is ``m`` (all the Cake runner reads)."""
    return OptimizationProfile(shapes=[[StaticDim(m)]], tensor_initializers=[])


def _runner_names(runners):
    return [type(runner).__name__ for runner in runners]


def _expected(p, use_bias):
    return p["ref_bias"] if use_bias else p["ref"]


def _cache_key_ms(m):
    """Every M that the effective bucket mapping sends to the cache key of ``m``
    (a plain scan, independent of the module's binary search)."""
    mapper = AutoTuner.get().get_effective_map_to_tuning_buckets(
        gemm_svdquant._NVFP4_SVDQUANT_GEMM_TUNING_CONFIG
    )
    key = mapper(m)
    return [x for x in range(1, key + 1) if mapper(x) == key]


def _key_admits(m, problem, enable_pdl):
    """Reference: whether every M of the cache key of ``m`` has a Cake route
    whose PDL setting admits ``enable_pdl``, checked at every M."""
    for x in _cache_key_ms(m):
        route_pdl = gemm_svdquant._cake_nvfp4_svdquant_m_pdl(x, problem)
        if route_pdl is None or (enable_pdl is not None and enable_pdl != route_pdl):
            return False
    return True


def _problem_inputs(m, n, k, rank, use_bias):
    p = _make_gemm_problem(m, n, k, rank=rank)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    return p, _runner_inputs(p, use_bias, out)


@pytest.mark.parametrize("m,n,k,rank,use_bias", _ROUTED)
def test_cake_is_last_candidate_and_untuned_call_keeps_cutlass(
    svdquant_choices, m, n, k, rank, use_bias
):
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    p = _make_gemm_problem(m, n, k, rank=rank)

    _hold_cake_entry()
    out = _run(p, use_bias)

    assert mm_nvfp4_svdquant.suitable_auto_backends == ["cutlass", "cake"]
    ((runners, (runner, tactic)),) = svdquant_choices
    assert _runner_names(runners) == [_CUTLASS_RUNNER, _CAKE_RUNNER]
    # Untuned with an empty cache: the first runner's default, as before.
    assert runner is runners[0] and tactic == -1
    assert torch.equal(out, _run(p, use_bias, backend="cutlass"))


def test_disable_env_removes_cake_candidate(svdquant_choices, monkeypatch):
    _skip_unless_cake_backend()
    monkeypatch.setenv(_DISABLE_ENV, "1")
    torch.manual_seed(0)
    p = _make_gemm_problem(129, 3072, 3072, rank=32)

    _hold_cake_entry()
    out = _run(p, True)

    assert mm_nvfp4_svdquant.suitable_auto_backends == ["cutlass"]
    ((runners, _),) = svdquant_choices
    assert _runner_names(runners) == [_CUTLASS_RUNNER]
    assert _sqnr_db(p["ref_bias"], out.float()) > 40.0
    # Explicit selection does not depend on the switch.
    cake_out = _run(p, True, backend="cake")
    assert _sqnr_db(p["ref_bias"], cake_out.float()) > 40.0


def test_cake_backend_is_not_experimental():
    # The experimental opt-in gate (FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS)
    # does not apply to this backend; only the Cake switch gates it.
    assert "cake" not in mm_nvfp4_svdquant.experimental_backends


@pytest.mark.parametrize("m,n,k,rank,use_bias", _UNROUTED)
def test_unrouted_problem_has_no_cake_candidate(
    svdquant_choices, m, n, k, rank, use_bias
):
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    p = _make_gemm_problem(m, n, k, rank=rank)

    with autotune(True):
        out = _run(p, use_bias)

    assert "cake" not in mm_nvfp4_svdquant.suitable_auto_backends
    for runners, _ in svdquant_choices:
        assert _runner_names(runners) == [_CUTLASS_RUNNER]
    assert _sqnr_db(_expected(p, use_bias), out.float()) > 40.0

    # The runner itself offers no tactic for a shape without a route, nor for
    # the bucket (128) that holds it.
    cake_runner = gemm_svdquant._cake_nvfp4_svdquant_gemm_runner(None)
    inputs = _runner_inputs(p, use_bias, out)
    assert cake_runner.get_valid_tactics(inputs, _profile(m)) == []
    assert cake_runner.get_valid_tactics(inputs, _profile(128)) == []


def test_cake_never_wins_a_bucket_holding_unrouted_m(svdquant_choices):
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    n, k, rank = 3072, 3072, 32
    tuned = _make_gemm_problem(129, n, k, rank=rank)
    out = torch.empty(129, n, dtype=torch.bfloat16, device="cuda")
    inputs = _runner_inputs(tuned, True, out)
    cake_runner = gemm_svdquant._cake_nvfp4_svdquant_gemm_runner(None)
    problem = gemm_svdquant._cake_nvfp4_svdquant_problem(inputs)
    # M=128 routes, but its bucket also holds M=65..127, which do not.
    assert gemm_svdquant._cake_nvfp4_svdquant_m_pdl(128, problem) is not None
    assert cake_runner.get_valid_tactics(inputs, _profile(128)) == []
    # M=129 maps to the cache key 256 (M 129..256).
    assert _cache_key_ms(129) == list(range(129, 257))
    expected = [0] if _key_admits(129, problem, None) else []
    assert cake_runner.get_valid_tactics(inputs, _profile(129)) == expected

    # Tuning at M=129 also tunes the M=128 bucket; an untuned M=100 call,
    # which lists CUTLASS only, must find that bucket's tuned CUTLASS tactic.
    with autotune(True):
        _run(tuned, True)
    # Cake is listed outside tuning from here on, whichever runner won, so only
    # the bucket check leaves it out of the M=100 call.
    _hold_cake_entry()
    p = _make_gemm_problem(100, n, k, rank=rank)
    replay = _run(p, True)

    runners, (runner, tactic) = svdquant_choices[-1]
    assert _runner_names(runners) == [_CUTLASS_RUNNER]
    assert runner is runners[0] and tactic != -1
    # The M=128 key itself holds that CUTLASS tactic, not a fallback.
    assert _exact_key_winner(runner, svdquant_choices.lookups[-1]) == tactic
    assert _sqnr_db(p["ref_bias"], replay.float()) > 40.0


def test_unsupported_arch_or_cuda_has_no_cake_candidate(svdquant_choices):
    capability = get_compute_capability(torch.device("cuda"))
    if capability[0] != 10:
        pytest.skip("CUTLASS NVFP4 SVDQuant requires an SM100-class GPU")
    if capability in ((10, 0), (10, 3)) and get_cuda_version() >= Version("13.0"):
        pytest.skip("Cake NVFP4 SVDQuant is supported on this GPU and CUDA")
    torch.manual_seed(0)
    p = _make_gemm_problem(129, 3072, 3072, rank=32)

    _hold_cake_entry()
    out = _run(p, True)

    assert mm_nvfp4_svdquant.suitable_auto_backends == ["cutlass"]
    ((runners, _),) = svdquant_choices
    assert _runner_names(runners) == [_CUTLASS_RUNNER]
    assert _sqnr_db(p["ref_bias"], out.float()) > 40.0


def test_explicit_pdl_admits_only_matching_cake_program(svdquant_choices):
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    m, n, k, rank = 129, 3072, 3072, 32
    p = _make_gemm_problem(m, n, k, rank=rank)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    inputs = _runner_inputs(p, True, out)
    route_pdl = gemm_svdquant._cake_nvfp4_svdquant_route_pdl(inputs)
    assert route_pdl is not None

    problem = gemm_svdquant._cake_nvfp4_svdquant_problem(inputs)

    _hold_cake_entry()
    _run(p, True, enable_pdl=route_pdl)
    _run(p, True, enable_pdl=not route_pdl)

    (matching, _), (mismatching, _) = svdquant_choices
    # The matching setting lists Cake only when every M of the cache key of
    # M=129 (129..256) routes to a program with that setting.
    expected = [_CUTLASS_RUNNER]
    if _key_admits(m, problem, route_pdl):
        expected.append(_CAKE_RUNNER)
    assert _runner_names(matching) == expected
    assert _runner_names(mismatching) == [_CUTLASS_RUNNER]


@pytest.mark.parametrize("m,n,k,rank,use_bias", _ROUTED)
def test_autotuned_auto_is_accurate(
    monkeypatch, svdquant_choices, m, n, k, rank, use_bias
):
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    p = _make_gemm_problem(m, n, k, rank=rank)
    expected = _expected(p, use_bias)
    # (runner class, profiled M) of every measurement, timed as usual.
    profiled = []
    profile_single_kernel = AutoTuner._profile_single_kernel

    def recording(self, runner, inputs, *args, **kwargs):
        profiled.append((type(runner).__name__, int(inputs[0].shape[0])))
        return profile_single_kernel(self, runner, inputs, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", recording)

    with autotune(True):
        tuned = _run(p, use_bias)
    assert _sqnr_db(expected, tuned.float()) > 40.0
    # The tuning call's own M is its largest bucket, stored under the cache
    # key of M. Cake is measured there only when every M of that key has a
    # route (the key 128 holds M=65..127, which do not).
    cake_runner = gemm_svdquant._cake_nvfp4_svdquant_gemm_runner(None)
    inputs = _runner_inputs(p, use_bias, tuned)
    problem = gemm_svdquant._cake_nvfp4_svdquant_problem(inputs)
    listed = _key_admits(m, problem, None)
    if m <= 128:
        assert not listed
    offered = cake_runner.get_valid_tactics(inputs, _profile(m))
    assert offered == ([0] if listed else [])
    # Tuning measured Cake at the bucket of M exactly when its key lists Cake;
    # either runner may win.
    assert ((_CAKE_RUNNER, m) in profiled) == listed
    assert (_CUTLASS_RUNNER, m) in profiled
    failed = AutoTuner.get().stats.failed_tactics
    assert f"{_OP}::{_CAKE_RUNNER}" not in failed

    # Replay outside the tuning context with whichever runner won. A failed
    # Cake measurement would leave the comparison incomplete and uncached.
    # Outside tuning Cake is listed only while the tuner holds a Cake winner.
    replay = _run(p, use_bias)
    assert _sqnr_db(expected, replay.float()) > 40.0
    runners, (winner, tactic) = svdquant_choices[-1]
    cake_held = _holds_cake_winner(AutoTuner.get())
    assert cake_held or type(winner).__name__ != _CAKE_RUNNER
    expected = (
        [_CUTLASS_RUNNER, _CAKE_RUNNER] if listed and cake_held else [_CUTLASS_RUNNER]
    )
    assert _runner_names(runners) == expected
    assert any(winner is runner for runner in runners)
    assert not (winner is runners[0] and tactic == -1)


@pytest.mark.parametrize("m,n,k,rank,use_bias", _ROUTED)
def test_forced_cake_selection_matches_explicit_cake(
    monkeypatch, m, n, k, rank, use_bias
):
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    p = _make_gemm_problem(m, n, k, rank=rank)
    explicit = _run(p, use_bias, backend="cake")

    _hold_cake_entry()
    tuner = AutoTuner.get()
    choose_one = tuner.choose_one
    forced_runners = []

    def choose_cake(custom_op, runners, tuning_config, inputs, **kwargs):
        if custom_op != _OP:
            return choose_one(custom_op, runners, tuning_config, inputs, **kwargs)
        assert _runner_names(runners) == [_CUTLASS_RUNNER, _CAKE_RUNNER]
        forced_runners.append(runners[-1])
        # The Cake runner's only tactic, as a tuned Cake winner returns it.
        return runners[-1], 0

    monkeypatch.setattr(tuner, "choose_one", choose_cake)
    forced = _run(p, use_bias)

    assert len(forced_runners) == 1
    # Same routed program with the same PDL setting as backend="cake".
    assert torch.equal(forced, explicit)
    assert _sqnr_db(_expected(p, use_bias), forced.float()) > 40.0


def test_tuning_bucket_low_matches_a_scan():
    low_of = gemm_svdquant._tuning_bucket_low
    mapper = map_to_hybrid_bucket_uncapped
    for key in (1, 2, 128, 256, 512, 2048, 2560, 4096, 8192, 16384):
        assert mapper(key) == key
        expected = min(x for x in range(1, key + 1) if mapper(x) == key)
        assert low_of(mapper, key) == expected
    # The defect case: M=6889 is stored under 8192, which holds M=4097..8192.
    assert mapper(6889) == 8192 and low_of(mapper, 8192) == 4097
    # One key per M (the round_up override with a generated bucket list).
    assert low_of(lambda x: x, 6889) == 6889
    # A rounding-down or clamping mapper has no bounded [low, key]: refused.
    assert low_of(make_bucket_mapper((128, 256), round_map=False), 128) is None
    assert low_of(make_bucket_mapper((128, 256), round_map=True), 256) is None


# (n, k, rank, has_bias, arch, sm_count) problems that exercise the exact-M
# rows, the M bounds of the general rows and, on two SM counts, the wide-tail
# rows (N 3072, K 12288).
_ROUTE_PROBLEMS = [
    (n, k, rank, has_bias, arch, sm_count)
    for arch in cake_jit.ARCHES
    for (n, k, rank, has_bias, sm_counts) in (
        (3072, 3072, 32, True, (148,)),
        (3072, 12288, 32, True, (148, 160)),
        (12288, 3072, 32, True, (148,)),
        (3072, 3072, 128, False, (148,)),
    )
    for sm_count in sm_counts
]


@pytest.mark.parametrize("problem", _ROUTE_PROBLEMS)
def test_route_m_starts_cover_every_route_change(problem):
    # The bucket check reads only the first M of each run; the route must
    # not change inside a run. Pure Python: no GPU needed.
    n, k, rank, has_bias, arch, sm_count = problem
    high = 16400

    def record(m):
        try:
            return cake_jit.select_record(
                m=m,
                n=n,
                k=k,
                rank=rank,
                has_bias=has_bias,
                arch=arch,
                sm_count=sm_count,
            )
        except (RuntimeError, ValueError):
            return None

    starts = set(gemm_svdquant._cake_nvfp4_svdquant_route_m_starts(1, high))
    previous = record(1)
    for m in range(2, high + 1):
        current = record(m)
        if current != previous:
            assert m in starts, (problem, m, previous, current)
        previous = current
    # A sub-range starts at its own low end.
    assert gemm_svdquant._cake_nvfp4_svdquant_route_m_starts(4097, 8192)[0] == 4097


def test_cake_offer_is_one_verdict_per_cache_key():
    # Every M of one cache key gets the same verdict, and it equals the
    # all-M reference, for each PDL request.
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    n, k, rank = 3072, 3072, 32
    _, inputs = _problem_inputs(4000, n, k, rank, True)
    problem = gemm_svdquant._cake_nvfp4_svdquant_problem(inputs)
    assert problem is not None
    for m in (100, 128, 129, 200, 256, 3585, 4000, 4096, 5000, 6889, 8192):
        key_ms = _cache_key_ms(m)
        for enable_pdl in (None, True, False):
            expected = _key_admits(m, problem, enable_pdl)
            for x in (key_ms[0], m, key_ms[-1]):
                offered = gemm_svdquant._cake_nvfp4_svdquant_offered(
                    x, problem, enable_pdl
                )
                assert offered == expected, (enable_pdl, m, x)


def test_cake_absent_from_a_key_holding_an_m_with_other_pdl(svdquant_choices):
    # M=4000 is not a hybrid bucket: tuning there stores the winner under the
    # cache key 4096, which holds M=3585..4096. For (N, K, rank) = (3072,
    # 3072, 32) with a bias, M=3585..4095 route to the non-PDL fallback and
    # M=4096 to an exact-row PDL program on sm_100a and sm_103a. With an
    # explicit PDL setting Cake must therefore be absent for every M of the
    # key; a later M=4096 call then finds the key's tuned CUTLASS tactic
    # instead of the untuned default.
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    n, k, rank, tuned_m, other_m = 3072, 3072, 32, 4000, 4096
    tuned, inputs = _problem_inputs(tuned_m, n, k, rank, True)
    other, other_inputs = _problem_inputs(other_m, n, k, rank, True)
    problem = gemm_svdquant._cake_nvfp4_svdquant_problem(inputs)
    assert _cache_key_ms(tuned_m) == list(range(3585, 4097))
    assert gemm_svdquant._cake_nvfp4_svdquant_m_pdl(3585, problem) is False
    assert gemm_svdquant._cake_nvfp4_svdquant_m_pdl(tuned_m, problem) is False
    assert gemm_svdquant._cake_nvfp4_svdquant_m_pdl(other_m, problem) is True

    for pdl in (False, True):
        cake_runner = gemm_svdquant._cake_nvfp4_svdquant_gemm_runner(pdl)
        for m, m_inputs in ((tuned_m, inputs), (other_m, other_inputs)):
            assert not cake_runner.supports_inputs(m_inputs), (pdl, m)
            assert cake_runner.get_valid_tactics(m_inputs, _profile(m)) == []
    # With enable_pdl=None every routed program is admitted.
    any_pdl = gemm_svdquant._cake_nvfp4_svdquant_gemm_runner(None)
    assert any_pdl.supports_inputs(inputs) == _key_admits(tuned_m, problem, None)
    assert any_pdl.supports_inputs(other_inputs) == any_pdl.supports_inputs(inputs)

    with autotune(True):
        _run(tuned, True, enable_pdl=False)
    for runners, _ in svdquant_choices:
        assert _runner_names(runners) == [_CUTLASS_RUNNER]

    # Cake is listed outside tuning from here on, so only the key check leaves
    # it out of the replayed calls.
    _hold_cake_entry()
    for m, p in ((other_m, other), (tuned_m, tuned)):
        out = _run(p, True, enable_pdl=False)
        runners, (runner, tactic) = svdquant_choices[-1]
        assert _runner_names(runners) == [_CUTLASS_RUNNER]
        # The tuned CUTLASS tactic of the key, not the untuned default.
        assert runner is runners[0] and tactic != -1, m
        # Held under this M's own key (4096), not served from another bucket.
        assert _exact_key_winner(runner, svdquant_choices.lookups[-1]) == tactic, m
        assert _sqnr_db(p["ref_bias"], out.float()) > 40.0


# Listing gate: no Cake work outside tuning until the tuner holds a Cake entry.


def test_cake_runner_class_name_matches_the_gate():
    for enable_pdl in (None, True, False):
        runner = gemm_svdquant._cake_nvfp4_svdquant_gemm_runner(enable_pdl)
        assert type(runner).__name__ == _CAKE_RUNNER
        # One cached runner per PDL request; no per-call construction.
        assert gemm_svdquant._cake_nvfp4_svdquant_gemm_runner(enable_pdl) is runner
    assert gemm_svdquant._CAKE_SVDQUANT_RUNNER == _CAKE_RUNNER


def test_group_tuning_lists_no_cake_candidate(monkeypatch):
    # Cake is offered per tuning cache key of the rank-local M, so tuning under
    # an autotune process group leaves it out: every rank must list the same
    # runners. Without tuning the group changes nothing. No GPU needed.
    from flashinfer.autotuner import autotuner as autotuner_module

    tuner = AutoTuner.get()
    heuristic = gemm_svdquant._heuristic_func_nvfp4_svdquant
    monkeypatch.setattr(tuner, "is_tuning_mode", True)
    assert heuristic(["cutlass", "cake"]) == ["cutlass", "cake"]

    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    assert tuner.may_select_runner_class(_CAKE_RUNNER)
    assert heuristic(["cutlass", "cake"]) == ["cutlass"]

    # Serving with the group still set: listed as without a group.
    monkeypatch.setattr(tuner, "is_tuning_mode", False)
    assert heuristic(["cutlass", "cake"]) == ["cutlass"]
    _hold_cake_entry()
    assert heuristic(["cutlass", "cake"]) == ["cutlass", "cake"]


class _GroupPlanChecked(Exception):
    """Raised by the stubbed autotune-group plan check once it has recorded."""


def test_group_tuning_plans_the_same_runners_for_every_m(monkeypatch):
    # Under an autotune process group every rank must list the same runners:
    # AutoTuner.choose_one compares their class names across ranks before
    # timing. A rank at M=129 (routed: Cake listed while tuning) and a rank at
    # M=65 (unrouted) plan CUTLASS only. Without a group M=129 still lists
    # Cake. The group check is stubbed: it records the compared runner names
    # and stops the call, so no collective runs and nothing is timed.
    from flashinfer.autotuner import autotuner as autotuner_module

    _skip_unless_cake_backend()
    torch.manual_seed(0)
    routed, unrouted = (129, 3072, 3072, 32, True), (65, 3072, 3072, 32, True)
    assert routed in _ROUTED and unrouted in _UNROUTED
    problems = [
        _make_gemm_problem(m, n, k, rank=rank)
        for m, n, k, rank, _ in (routed, unrouted)
    ]
    offered, planned = [], []
    choose_one = AutoTuner.choose_one

    def check(phase, selection=(), error=None):
        assert phase == "operation" and selection[:2] == ("choose_one", _OP), phase
        planned.append(list(selection[2]))
        raise _GroupPlanChecked

    def recording(self, custom_op, runners, tuning_config, inputs, **kwargs):
        assert custom_op == _OP
        offered.append(_runner_names(runners))
        if autotuner_module._tune_process_group is None:
            return runners[0], -1
        return choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs)

    monkeypatch.setattr(autotuner_module, "_check_tuning_group_phase", check)
    monkeypatch.setattr(AutoTuner, "choose_one", recording)
    monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)

    for p in problems:
        out = _run(p, True)
        assert _sqnr_db(p["ref_bias"], out.float()) > 40.0
    assert not planned
    assert offered == [[_CUTLASS_RUNNER, _CAKE_RUNNER], [_CUTLASS_RUNNER]]

    monkeypatch.setattr(autotuner_module, "_tune_process_group", object())
    for p in problems:
        with pytest.raises(_GroupPlanChecked):
            _run(p, True)
    assert offered[2:] == planned == [[_CUTLASS_RUNNER], [_CUTLASS_RUNNER]]


def test_untuned_call_does_no_cake_work(monkeypatch, svdquant_choices):
    # Without tuning and without a Cake entry the call runs no Cake route or
    # bucket check and builds no Cake runner; a held Cake entry makes it reach
    # the Cake checks. backend_requirement still runs the "cake" support
    # checker on every automatic call, as before Cake joined the candidates:
    # one route lookup for M through the package's select_record.
    _skip_unless_cake_backend()
    torch.manual_seed(0)
    m, n, k, rank, use_bias = _ROUTED[0]
    p = _make_gemm_problem(m, n, k, rank=rank)
    support_probes, route_lookups = [], []

    def spy(calls, select_record):
        def recording(*args, **kwargs):
            calls.append(kwargs.get("m"))
            return select_record(*args, **kwargs)

        return recording

    # is_supported (the support checker) reads the package binding; the
    # candidate's route and bucket checks read cake_jit.select_record.
    monkeypatch.setattr(
        cake_backend, "select_record", spy(support_probes, cake_backend.select_record)
    )
    monkeypatch.setattr(
        cake_jit, "select_record", spy(route_lookups, cake_jit.select_record)
    )
    for name in (
        "_cake_nvfp4_svdquant_gemm_runner",
        "_cake_nvfp4_svdquant_problem",
        "_cake_nvfp4_svdquant_offered",
    ):

        def fail(*args, _name=name, **kwargs):
            raise AssertionError(f"{_name} was called")

        monkeypatch.setattr(gemm_svdquant, name, fail)

    out = _run(p, use_bias)

    assert support_probes == [m]
    assert route_lookups == []
    assert mm_nvfp4_svdquant.suitable_auto_backends == ["cutlass"]
    ((runners, (runner, tactic)),) = svdquant_choices
    assert _runner_names(runners) == [_CUTLASS_RUNNER]
    assert runner is runners[0] and tactic == -1
    assert torch.equal(out, _run(p, use_bias, backend="cutlass"))

    _hold_cake_entry()
    with pytest.raises(AssertionError, match="was called"):
        _run(p, use_bias)


def test_cake_winner_served_after_tuning_and_from_cache_file(
    monkeypatch, request, tmp_path
):
    # A cache key that Cake won is served by Cake outside tuning, in the tuning
    # process and in a later tuner that loads the saved cache file; clearing
    # the caches closes the gate again.
    _skip_unless_cake_backend()
    request.addfinalizer(AutoTuner.get().clear_cache)
    torch.manual_seed(0)
    for m, n, k, rank, use_bias in _ROUTED:
        p, inputs = _problem_inputs(m, n, k, rank, use_bias)
        problem = gemm_svdquant._cake_nvfp4_svdquant_problem(inputs)
        if problem is not None and _key_admits(m, problem, None):
            break
    else:
        pytest.fail(
            "no _ROUTED problem has a Cake route at every M of its tuning cache "
            "key on this GPU; Cake NVFP4 SVDQuant is supported here, so a "
            "declared routed problem must be admitted"
        )
    expected = _expected(p, use_bias)

    calls = []
    choose_one = AutoTuner.choose_one

    def recording(self, custom_op, runners, tuning_config, inputs, **kwargs):
        runner, tactic = choose_one(
            self, custom_op, runners, tuning_config, inputs, **kwargs
        )
        if custom_op == _OP:
            calls.append((_runner_names(runners), type(runner).__name__))
        return runner, tactic

    monkeypatch.setattr(AutoTuner, "choose_one", recording)
    timed = []

    def timing(self, runner, *args, **kwargs):
        # Cake measures faster than every CUTLASS tactic, without a launch.
        timed.append(type(runner).__name__)
        return 1.0 if type(runner).__name__ == _CAKE_RUNNER else 2.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", timing)
    cache = tmp_path / "svdquant_cake.json"

    with autotune(True, tuning_buckets=(m,), cache=str(cache)):
        _run(p, use_bias)
    assert _CAKE_RUNNER in timed and _CUTLASS_RUNNER in timed
    assert calls[-1] == ([_CUTLASS_RUNNER, _CAKE_RUNNER], _CAKE_RUNNER)
    saved = json.loads(cache.read_text())
    assert any(
        isinstance(value, list) and value[:1] == [_CAKE_RUNNER]
        for key, value in saved.items()
        if not key.startswith("_")
    )

    # The tuning process, outside tuning.
    out = _run(p, use_bias)
    assert calls[-1] == ([_CUTLASS_RUNNER, _CAKE_RUNNER], _CAKE_RUNNER)
    assert _sqnr_db(expected, out.float()) > 40.0

    # A later tuner: CUTLASS only until the cache file is loaded.
    later = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", later)
    _run(p, use_bias)
    assert calls[-1] == ([_CUTLASS_RUNNER], _CUTLASS_RUNNER)
    with autotune(False, cache=str(cache)):
        out = _run(p, use_bias)
    assert calls[-1] == ([_CUTLASS_RUNNER, _CAKE_RUNNER], _CAKE_RUNNER)
    assert _sqnr_db(expected, out.float()) > 40.0

    later.clear_cache()
    out = _run(p, use_bias)
    assert calls[-1] == ([_CUTLASS_RUNNER], _CUTLASS_RUNNER)
    assert _sqnr_db(expected, out.float()) > 40.0
