"""Tuning decision versions and source-based compiled artifacts are independent."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import importlib
import json

import pytest
import torch

from b12x._lib import compiler
from b12x._lib.compile_plan import ProgramKey, record_program
from b12x._lib.compile_pool import CompilationPlan, CompileJob
from b12x.preparation import (
    DetectedDevice,
    PreparationSession,
    PreparedCall,
    TuningCacheRequirement,
)
from b12x.preparation._cache import SelectionCache, cache_identity
from .test_defaults import contract
from .test_session import _deterministic_timer, declaration


# Ordinals 0 and 2 are the same part sold under two product labels, whose
# reported total memory differs by a few MiB; ordinal 3 is different silicon.
_TEST_GPU_A = SimpleNamespace(
    name="Test GPU A",
    major=12,
    minor=0,
    multi_processor_count=170,
    total_memory=32 * 1024**3,
)
_TEST_GPUS = {
    0: _TEST_GPU_A,
    1: _TEST_GPU_A,
    2: SimpleNamespace(
        name="Test GPU B",
        major=12,
        minor=0,
        multi_processor_count=170,
        total_memory=32 * 1024**3 - 10 * 1024**2,
    ),
    3: SimpleNamespace(
        name="Test GPU C",
        major=12,
        minor=1,
        multi_processor_count=148,
        total_memory=120 * 1024**3,
    ),
}


@pytest.fixture
def cache_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *_args: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda ordinal: _TEST_GPUS[ordinal].name
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda ordinal: _TEST_GPUS[ordinal]
    )
    monkeypatch.setattr(
        compiler, "_device_arch_key", lambda ordinal: ("cuda", (12, 0), 170)
    )
    monkeypatch.delenv("B12X_TUNING_CACHE_VERSION", raising=False)


def _save_choice(cache):
    cache.save(
        "shape",
        assignment={"width": 2},
        config={"width": 2},
        coverage={
            "cartesian_count": 3,
            "legal_count": 3,
            "effective_count": 3,
            "measured_count": 3,
        },
        programs=(ProgramKey("cute", "selected-source-key"),),
    )


def test_manual_version_selects_a_distinct_decision_cache_without_deleting_choices(
    cache_device, tmp_path, monkeypatch
):
    original = SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0))
    _save_choice(original)
    assert original.identity["tuning_cache_version"] == 1
    assert (
        SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0)).get("shape")
        is not None
    )
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "2")
    retune = SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0))
    assert retune.path != original.path
    assert retune.get("shape") is None
    assert original.path.is_file()
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "1")
    assert (
        SelectionCache(tmp_path, cache_identity({"model": "model-a"}, 0)).get("shape")
        is not None
    )


@pytest.mark.parametrize("version", ["0", "-1", "", "1.5", "invalid"])
def test_invalid_tuning_version_fails_before_cache_access(
    cache_device, monkeypatch, version
):
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", version)
    with pytest.raises(
        ValueError, match="B12X_TUNING_CACHE_VERSION must be a positive integer"
    ):
        cache_identity({}, 0)


def test_decisions_track_silicon_rather_than_product_label(cache_device):
    assert cache_identity({"model": "a"}, 0) != cache_identity({"model": "b"}, 0)
    assert cache_identity({"model": "a"}, 0) == cache_identity({"model": "a"}, 1)
    # Ordinal 2 is ordinal 0's part under a different product label, and its
    # reported total memory even differs by a few MiB.
    assert cache_identity({"model": "a"}, 0) == cache_identity({"model": "a"}, 2)
    # Different silicon still gets its own decisions.
    assert cache_identity({"model": "a"}, 0) != cache_identity({"model": "a"}, 3)


def test_stream_gated_measurement_keeps_prior_choices_in_a_separate_cache(
    cache_device, tmp_path
):
    identity = cache_identity({}, 0)
    assert identity["measurement"] == "stream_gated_events_v1"
    previous = {key: value for key, value in identity.items() if key != "measurement"}
    original = SelectionCache(tmp_path, previous)
    _save_choice(original)
    measured = SelectionCache(tmp_path, identity)
    assert measured.path != original.path
    assert measured.get("shape") is None
    assert original.path.is_file()


def test_cached_choices_survive_uuid_and_ordinal_changes(
    cache_device, tmp_path, monkeypatch
):
    original = SelectionCache(tmp_path, cache_identity({"model": "a"}, 0))
    _save_choice(original)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda ordinal: SimpleNamespace(
            **vars(_TEST_GPUS[ordinal]), uuid="replacement-gpu"
        ),
    )
    moved = SelectionCache(tmp_path, cache_identity({"model": "a"}, 1))
    assert moved.path == original.path
    assert moved.get("shape") == original.get("shape")


@pytest.mark.parametrize("name", ["", "   "])
def test_missing_device_name_fails_before_cache_access(cache_device, monkeypatch, name):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda ordinal: name)
    with pytest.raises(RuntimeError, match="CUDA device name is unavailable"):
        cache_identity({}, 0)


def test_source_and_toolchain_changes_rekey_kernels_without_rekeying_decisions(
    cache_device, tmp_path, monkeypatch
):
    source = tmp_path / "kernel.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(compiler, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(
        compiler,
        "_b12x_package_fingerprint",
        compiler._compute_b12x_package_fingerprint,
    )
    monkeypatch.setattr(compiler, "_runtime_toolchain_key", lambda: ("compiler-a",))

    def compile_context():
        return compiler._static_compile_cache_context.__wrapped__(object())

    compiler._compile_environment_key.cache_clear()
    try:
        decision = cache_identity({}, 0)
        kernel = compile_context()
        source.write_text("VALUE = 2\n")
        changed_source = compile_context()
        assert changed_source != kernel
        assert cache_identity({}, 0) == decision
        monkeypatch.setattr(compiler, "_runtime_toolchain_key", lambda: ("compiler-b",))
        changed_toolchain = compile_context()
        assert changed_toolchain != changed_source
        assert cache_identity({}, 0) == decision
        monkeypatch.setenv("NVCC_APPEND_FLAGS", "-lineinfo")
        compiler._compile_environment_key.cache_clear()
        assert compile_context() != changed_toolchain
        assert cache_identity({}, 0) == decision
    finally:
        compiler._compile_environment_key.cache_clear()


def test_manual_tuning_version_does_not_change_kernel_compilation_context(
    cache_device, monkeypatch
):
    monkeypatch.setattr(
        compiler, "_b12x_package_fingerprint", lambda: "same-kernel-source"
    )
    monkeypatch.setattr(compiler, "_runtime_toolchain_key", lambda: ("same-compiler",))
    compiler._compile_environment_key.cache_clear()
    try:
        before = compiler._static_compile_cache_context.__wrapped__(object())
        decision = cache_identity({}, 0)
        monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "2")
        compiler._compile_environment_key.cache_clear()
        assert compiler._static_compile_cache_context.__wrapped__(object()) == before
        assert cache_identity({}, 0) != decision
    finally:
        compiler._compile_environment_key.cache_clear()


def test_matching_malformed_decision_still_fails_closed(cache_device, tmp_path):
    identity = cache_identity({}, 0)
    cache = SelectionCache(tmp_path, identity)
    _save_choice(cache)
    payload = json.loads(cache.path.read_text())
    payload["records"]["shape"]["coverage"]["measured_count"] = 1
    cache.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="only completed exhaustive"):
        SelectionCache(tmp_path, identity)


@pytest.mark.parametrize("failure", ("identity", "incomplete"))
def test_cache_agreement_rejects_incompatible_or_incomplete_peer_results(
    cache_device,
    tmp_path,
    failure,
):
    identity = cache_identity({}, 0)
    cache = SelectionCache(tmp_path, identity)
    _save_choice(cache)
    peer = json.loads(json.dumps(cache.records))
    if failure == "identity":
        identity = {**identity, "sm_count": identity["sm_count"] + 1}
        message = "identities differ"
    else:
        peer["shape"]["coverage"]["measured_count"] = 1
        message = "only completed exhaustive"
    with pytest.raises(ValueError, match=message):
        cache.reconcile((TuningCacheRequirement((0, 1), identity, peer),))
    assert cache.get("shape")["assignment"]["width"] == 2


def test_ranks_share_decisions_when_one_part_carries_two_product_labels(
    cache_device,
    tmp_path,
):
    """A tensor-parallel group may mix product labels for the same part."""
    cache = SelectionCache(tmp_path, cache_identity({"model": "a"}, 0))
    _save_choice(cache)
    peer = SelectionCache(tmp_path, cache_identity({"model": "a"}, 2))
    assert peer.path == cache.path
    assert peer.get("shape") is not None
    cache.reconcile((TuningCacheRequirement((0, 2), peer.identity, peer.records),))
    assert cache.get("shape")["assignment"]["width"] == 2


def test_cached_choice_rebuilds_changed_program_without_racing(
    cache_device, tmp_path, monkeypatch
):
    session_module = importlib.import_module("b12x.preparation.session")
    _deterministic_timer(monkeypatch)
    available, built = set(), []
    source_version = "a"

    def describe(job):
        return CompilationPlan(
            job, (ProgramKey("cute", f"{source_version}-{job.args[0]}"),)
        )

    def build(plans):
        for plan in plans:
            for program in plan.programs:
                built.append(program.key)
                available.add(program)

    monkeypatch.setattr(session_module, "describe_compilation", describe)
    monkeypatch.setattr(
        session_module,
        "compiled_program_available",
        lambda program: program in available,
    )
    monkeypatch.setattr(session_module, "compile_in_process", build)

    def prepare(ordinal=0):
        def materialize(selection, device):
            program = ProgramKey("cute", f"{source_version}-{selection.config.width}")
            return SimpleNamespace(value=selection.config.width * 3, program=program)

        def factory(state):
            def run():
                assert state.program in available
                record_program(state.program)
                return state.value

            return PreparedCall(run=run, produce=lambda: None)

        plan = replace(
            declaration(tuning=contract(default=2)),
            _compile_jobs=lambda config, device: (
                CompileJob.create("test.compiler:compile", config.width),
            ),
            _materialize=materialize,
        )
        with PreparationSession(
            device=DetectedDevice(None, None), compile_workers=0
        ) as session:
            session._cache = SelectionCache(tmp_path, cache_identity({}, ordinal))
            result = session.prepare(
                (
                    plan.request(
                        name="query", prepare_call=factory, benchmark_call=factory
                    ),
                )
            )
            return result.selections["query"].source, result.benchmarked_candidates

    assert prepare() == ("tuned", 3)
    assert set(built) == {"a-1", "a-2", "a-4"}
    source_version = "b"
    built.clear()
    assert prepare(ordinal=1) == ("cached", 0)
    assert built == ["b-2"]
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "2")
    built.clear()
    assert prepare() == ("tuned", 3)
    assert set(built) == {"b-1", "b-4"}
    monkeypatch.setenv("B12X_TUNING_CACHE_VERSION", "3")
    built.clear()
    assert prepare() == ("tuned", 3)
    assert built == []
