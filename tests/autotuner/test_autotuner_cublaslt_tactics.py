"""
Tests for the cuBLASLt GEMM runners' algorithm-descriptor tactics.

The cuBLAS FP8 (``bmm_fp8(backend="cublas")``) and cuBLASLt BF16
(``mm_bf16(backend="cublaslt")`` and the cuBLASLt fallback of
``backend="cute-dsl"``) runners use a ``cublasLtMatmulAlgo_t`` descriptor as
their tactic. Candidates are enumerated at the tuning bucket's M, the cache
key carries no exact M, and a descriptor that does not apply to the actual
problem runs the heuristic default instead.

The serialization tests run on CPU; the rest need a GPU.
"""

import json

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, bmm_fp8, mm_bf16
from flashinfer.autotune_cache import autotune_v2
from flashinfer.autotuner import (
    AutoTuner,
    TuningConfig,
    _json_to_tactic,
    _tactic_to_json,
)
from flashinfer.autotuner.autotuner import _profile_measurement_scope
from flashinfer.gemm import gemm_base
from flashinfer.gemm.gemm_base import (
    _BF16_GEMM_SM100_TUNING_CONFIG,
    _cublaslt_algo_tensor,
    _cublaslt_algos_to_tactics,
    _is_cublaslt_algo_tactic,
    get_gemm_module,
    get_mm_bf16_cublaslt_module,
)
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8

from .utils import DummyRunner, reset_autotuner

# Words above 2**63 must survive every round-trip unchanged.
_DESCRIPTORS = [
    (0, 1, 2, 3, 4, 5, 6, 7),
    (2**64 - 1, 2**63, 2**63 - 1, 0, 17, 2**40 + 3, 1, 2**64 - 2),
]


# ---------------------------------------------------------------------------
# CPU: descriptor tactic serialization
# ---------------------------------------------------------------------------


def test_descriptor_bytes_roundtrip():
    algo_buf = torch.empty(100 * 64, dtype=torch.uint8)
    for i, descriptor in enumerate(_DESCRIPTORS):
        algo_buf[i * 64 : (i + 1) * 64] = _cublaslt_algo_tensor(descriptor)
    tactics = _cublaslt_algos_to_tactics(algo_buf, len(_DESCRIPTORS))
    assert tactics == _DESCRIPTORS
    assert all(_is_cublaslt_algo_tactic(t) for t in tactics)
    assert len(set(tactics)) == len(tactics)


def test_descriptor_json_roundtrip():
    for descriptor in _DESCRIPTORS:
        restored = _json_to_tactic(json.loads(json.dumps(_tactic_to_json(descriptor))))
        assert restored == descriptor
        assert _is_cublaslt_algo_tactic(restored)


@pytest.mark.parametrize(
    "tactic",
    [-1, 3, (1, 2, 3), (0,) * 9, (-1,) * 8, (2**64,) + (0,) * 7, [0] * 8, (0.0,) * 8],
)
def test_non_descriptor_tactics_rejected(tactic):
    assert not _is_cublaslt_algo_tactic(tactic)


_OP = "test::cublaslt_descriptor_persistence"
_CONFIG = TuningConfig()


def _tune_descriptors(monkeypatch, winner):
    times = {t: (0.5 if t == winner else 1.0) for t in _DESCRIPTORS}
    monkeypatch.setattr(
        AutoTuner,
        "_profile_single_kernel",
        lambda self, runner, inputs, tactic, tuning_config, **kw: times[tactic],
    )
    return AutoTuner.get().choose_one(
        _OP, [DummyRunner(_DESCRIPTORS)], _CONFIG, [torch.zeros(4, 8)]
    )


def _fresh_process():
    tuner = reset_autotuner()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner._managed_decoded.clear()
    return tuner


@pytest.mark.parametrize("winner", _DESCRIPTORS)
def test_descriptor_config_file_roundtrip(tmp_path, monkeypatch, winner):
    path = str(tmp_path / "configs.json")
    _fresh_process()
    try:
        with autotune(True, cache=path):
            _, tactic = _tune_descriptors(monkeypatch, winner)
        assert tactic == winner

        tuner = _fresh_process()
        with autotune(False, cache=path):
            hit, _, tactic, _ = tuner.search_cache(
                _OP,
                [DummyRunner(_DESCRIPTORS)],
                ((4, 8),),
                _CONFIG,
                inputs=[torch.zeros(4, 8)],
            )
        assert hit
        assert tactic == winner
    finally:
        _fresh_process()


@pytest.mark.parametrize("winner", _DESCRIPTORS)
def test_descriptor_managed_cache_roundtrip(tmp_path, monkeypatch, winner):
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    _fresh_process()
    try:
        with autotune_v2():
            _, tactic = _tune_descriptors(monkeypatch, winner)
        assert tactic == winner
        entries = list(tmp_path.rglob("entries/*.json"))
        assert len(entries) == 1
        assert json.loads(entries[0].read_text())["tactic"] == list(winner)

        _fresh_process()
        with autotune_v2(mode="replay"):
            runner, tactic = AutoTuner.get().choose_one(
                _OP, [DummyRunner(_DESCRIPTORS)], _CONFIG, [torch.zeros(4, 8)]
            )
        assert tactic == winner
    finally:
        _fresh_process()


# ---------------------------------------------------------------------------
# GPU: bucketing, graph capture, fallback
# ---------------------------------------------------------------------------

_N, _K = 1024, 2048


def _compute_capability():
    major, minor = get_compute_capability(torch.device("cuda"))
    return major * 10 + minor


def _skip_unless_backend(backend):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    cc = _compute_capability()
    if backend == "cublas":
        if not bmm_fp8.is_backend_supported("cublas", cc):
            pytest.skip(f"bmm_fp8 cublas backend not supported on sm{cc}.")
        return
    if not mm_bf16.is_backend_supported(backend, cc):
        pytest.skip(f"mm_bf16 {backend} backend not supported on sm{cc}.")
    if backend == "cute-dsl":
        from flashinfer.cute_dsl.utils import is_cute_dsl_available

        if not is_cute_dsl_available():
            pytest.skip("nvidia-cutlass-dsl is not available.")


class _Problem:
    """One GEMM of the backend under test, with static buffers for graph capture."""

    def __init__(self, backend, m, seed=0):
        self.backend = backend
        gen = torch.Generator(device="cuda").manual_seed(seed)
        if backend == "cublas":
            a = torch.randn(
                [1, m, _K], device="cuda", dtype=torch.bfloat16, generator=gen
            )
            b = torch.randn(
                [1, _N, _K], device="cuda", dtype=torch.bfloat16, generator=gen
            ).transpose(-2, -1)
            self.reference = torch.bmm(a, b)
            self.a, self.a_scale = to_float8(a, dtype=torch.float8_e4m3fn)
            self.b, self.b_scale = to_float8(b, dtype=torch.float8_e4m3fn)
            self.out = torch.empty([1, m, _N], device="cuda", dtype=torch.bfloat16)
        else:
            self.a = torch.randn(
                [m, _K], device="cuda", dtype=torch.bfloat16, generator=gen
            )
            self.b = torch.randn(
                [_N, _K], device="cuda", dtype=torch.bfloat16, generator=gen
            ).T
            self.reference = self.a @ self.b
            self.out = torch.empty([m, _N], device="cuda", dtype=torch.bfloat16)

    def __call__(self):
        if self.backend == "cublas":
            bmm_fp8(
                self.a,
                self.b,
                self.a_scale,
                self.b_scale,
                torch.bfloat16,
                self.out,
                backend="cublas",
            )
        else:
            mm_bf16(self.a, self.b, out=self.out, backend=self.backend)
        return self.out

    def check(self):
        cos_sim = F.cosine_similarity(
            self.reference.float().reshape(-1), self.out.float().reshape(-1), dim=0
        )
        assert cos_sim > 0.99


@pytest.fixture
def tuner():
    tuner = reset_autotuner()
    yield tuner
    reset_autotuner()


@pytest.fixture
def profiled(monkeypatch):
    """Runner class names of every profiled tactic."""
    calls = []
    original = AutoTuner._profile_single_kernel

    def counting(self, runner, *args, **kwargs):
        calls.append(runner.__class__.__name__)
        return original(self, runner, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", counting)
    return calls


@pytest.fixture
def descriptor_runs(monkeypatch):
    """(used_descriptor, tactic) of every descriptor run."""
    runs = []
    original = gemm_base._check_cublaslt_descriptor_ran

    def recording(used_descriptor, tactic):
        runs.append((used_descriptor, tactic))
        return original(used_descriptor, tactic)

    monkeypatch.setattr(gemm_base, "_check_cublaslt_descriptor_ran", recording)
    return runs


_CUBLAS_RUNNERS = {
    "cublas": "CublasFp8GemmRunner",
    "cublaslt": "CublasltBf16GemmRunner",
    "cute-dsl": "CuteDSLCublasltFallbackBf16Runner",
}


def _bucket_winner(tuner, runner_name, bucket_m):
    winners = [
        tactic
        for key, (tactic, _) in tuner._winner_cache().items()
        if key.runner_class_name == runner_name
        and key.nearest_profile[0][-2] == bucket_m
    ]
    assert len(winners) == 1
    return winners[0]


@pytest.mark.parametrize("backend", ["cublas", "cublaslt", "cute-dsl"])
def test_tuned_descriptor_reused_across_bucket(
    backend, tuner, profiled, descriptor_runs
):
    _skip_unless_backend(backend)
    runner_name = _CUBLAS_RUNNERS[backend]
    bucket_m, run_m = 512, 500
    mapper = AutoTuner.get().get_effective_map_to_tuning_buckets(
        _BF16_GEMM_SM100_TUNING_CONFIG
    )
    assert mapper(run_m) == bucket_m

    with autotune(True):
        _Problem(backend, bucket_m)()
    assert runner_name in profiled
    winner = _bucket_winner(tuner, runner_name, bucket_m)
    assert _is_cublaslt_algo_tactic(winner)

    for tune_mode in (True, False):
        profiled.clear()
        descriptor_runs.clear()
        problem = _Problem(backend, run_m, seed=1)
        with autotune(tune_mode):
            problem()
        problem.check()
        assert profiled == []
        assert descriptor_runs == [(1, winner)]


@pytest.mark.parametrize("backend", ["cublas", "cublaslt", "cute-dsl"])
def test_graph_capture_inside_outer_autotune(backend, tuner, profiled):
    _skip_unless_backend(backend)
    with autotune(True):
        _Problem(backend, 600)()
        profiled.clear()
        # 513 shares 600's bucket and runs here for the first time, inside
        # the capture.
        for m in (600, 513):
            problem = _Problem(backend, m, seed=m)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                problem()
            problem.out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            problem.check()
    assert profiled == []


def _runner_and_inputs(backend, m):
    problem = _Problem(backend, m)
    workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    if backend == "cublas":
        runner = get_gemm_module().cublas_fp8_gemm_runner()
        inputs = [
            problem.a,
            problem.b,
            problem.a_scale,
            problem.b_scale,
            problem.out,
            workspace,
        ]
    else:
        runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
        inputs = [problem.a, problem.b, None, False, problem.out, workspace]
    return problem, runner, inputs


@pytest.mark.parametrize("backend", ["cublas", "cublaslt"])
def test_invalid_descriptor_runs_heuristic_default(backend, descriptor_runs):
    _skip_unless_backend(backend)
    problem, runner, inputs = _runner_and_inputs(backend, 300)
    corrupted = [
        (2**64 - 1,) * 8,
        (0x0123456789ABCDEF, 2**63 + 5, 42, 7, 2**64 - 3, 1, 0, 99),
    ]
    for tactic in corrupted:
        assert runner.validate_tactic(inputs, tactic)
        problem.out.zero_()
        runner.forward(inputs, tactic=tactic)
        torch.cuda.synchronize()
        problem.check()
        with (
            _profile_measurement_scope(),
            pytest.raises(RuntimeError, match="does not apply"),
        ):
            runner.forward(inputs, tactic=tactic)
    assert [used for used, _ in descriptor_runs] == [0] * (2 * len(corrupted))

    # Descriptors enumerated with a workspace, run without one: those that
    # need workspace fall back, the others run as given.
    descriptors = runner.get_valid_tactics(inputs, None)
    assert descriptors
    no_workspace = inputs[:-1] + [inputs[-1][:0]]
    descriptor_runs.clear()
    for tactic in descriptors:
        problem.out.zero_()
        runner.forward(no_workspace, tactic=tactic)
        torch.cuda.synchronize()
        problem.check()
    assert [tactic for _, tactic in descriptor_runs] == descriptors
    assert any(used for used, _ in descriptor_runs)


@pytest.mark.parametrize("backend", ["cublas", "cublaslt"])
def test_index_tactics_rejected(backend):
    _skip_unless_backend(backend)
    _, runner, inputs = _runner_and_inputs(backend, 64)
    assert runner.validate_tactic(inputs, -1)
    assert not runner.validate_tactic(inputs, 3)
    with pytest.raises(ValueError, match="descriptor"):
        runner.forward(inputs, tactic=3)


@pytest.mark.parametrize("backend", ["cublas", "cublaslt"])
def test_tuned_descriptor_persists(backend, tmp_path, tuner, descriptor_runs):
    _skip_unless_backend(backend)
    runner_name = _CUBLAS_RUNNERS[backend]
    path = str(tmp_path / "configs.json")
    with autotune(True, cache=path):
        _Problem(backend, 256)()
    winner = _bucket_winner(tuner, runner_name, 256)

    with open(path) as f:
        saved = [
            _json_to_tactic(value[1])
            for key, value in json.load(f).items()
            if not key.startswith("_") and value[0] == runner_name
        ]
    assert winner in saved

    reset_autotuner()
    descriptor_runs.clear()
    problem = _Problem(backend, 200, seed=2)
    with autotune(False, cache=path):
        problem()
    problem.check()
    assert descriptor_runs == [(1, winner)]
