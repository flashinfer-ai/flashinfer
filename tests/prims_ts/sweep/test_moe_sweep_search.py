# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace
import io
import json
import math
import sys
from types import SimpleNamespace

import pytest

from flashinfer.prims_ts.moe.tools.sweep import engine
from flashinfer.prims_ts.moe.tools.sweep import utils as sweep_utils
from flashinfer.prims_ts.moe.tools.sweep.engine import (
    SweepOptions,
    SweepEngine,
    _best_final_time,
    _prune_failed_finalists,
    _shape_label,
    cap_and_prune_candidates,
    cap_candidates_round_robin,
    code_fingerprint,
    describe_worker_failure,
    format_speedup_summary,
    manifest_identity,
    manifest_identity_mismatches,
    menu_speedup_curve,
    reference_consistency,
    retryable_failed_ids,
    selection_portfolio_shapes,
    speedup_vs_pre_sweep,
    summarize_gpu_clocks,
    summarize_worker_activity,
)
from flashinfer.prims_ts.moe.tools.sweep.workers import (
    FailureKind,
    ProcessResult,
    WorkerActivity,
    WorkerStatus,
)
from flashinfer.prims_ts.moe.tools.sweep.measure import (
    TimingProtocol,
    final_timing_protocol,
)
from flashinfer.prims_ts.moe.tools.sweep.portfolio import (
    FAILED,
    NOT_APPLICABLE,
    PortfolioCandidate as Candidate,
    Shape,
    TimingKind,
    TimingValue,
    build_final_retime_pool,
    select_greedy_menu,
)
from flashinfer.prims_ts.moe.tools.sweep.search import (
    AdaptiveMoveSelector,
    BetaArm,
    CandidateOutcome,
    GroupBudgetAllocator,
    GroupProgress,
    NoiseAwareThresholds,
    ParentMoveTracker,
    RejectionReason,
    SearchState,
    contested_group_weights,
    group_best_objective,
    rank_parents_by_mean_regret,
    select_parent_pool,
)
from flashinfer.prims_ts.moe.tools.sweep.shapes import model_moe_shape
from flashinfer.prims_ts.moe.tools.sweep.space import Candidate as SpaceCandidate


def test_shape_timeout_defaults_to_sixty_seconds() -> None:
    assert SweepOptions.__dataclass_fields__["shape_timeout_seconds"].default == 60.0


def test_timing_protocol_and_sweep_options_share_iteration_defaults() -> None:
    protocol = TimingProtocol()
    assert (
        protocol.warmup_iters
        == SweepOptions.__dataclass_fields__["warmup_iters"].default
    )
    assert (
        protocol.repeat_iters
        == SweepOptions.__dataclass_fields__["repeat_iters"].default
    )


def test_code_fingerprint_reports_unknown_git_state_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine,
        "git_output",
        lambda *_args, **_kwargs: None,
    )

    fingerprint = code_fingerprint()

    assert fingerprint["git_dirty"] is None
    assert fingerprint["git_dirty_files"] == -1


def test_git_output_preserves_porcelain_leading_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(stdout=" M benchmarks/x.py\n")

    monkeypatch.setattr(sweep_utils.subprocess, "run", fake_run)

    assert sweep_utils.git_output("status", repository="/repo") == " M benchmarks/x.py"
    assert observed["command"] == ("git", "-C", "/repo", "status")
    assert observed["timeout"] == 30


def test_code_fingerprint_preserves_complete_porcelain_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine,
        "git_output",
        lambda *args, **_kwargs: (
            " M benchmarks/x.py" if "status" in args else "deadbeef"
        ),
    )

    fingerprint = code_fingerprint()

    assert fingerprint["git_dirty"] is True
    assert fingerprint["git_dirty_files"] == ["benchmarks/x.py"]


def test_nvml_driver_version(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake_nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlSystemGetDriverVersion=lambda: b"620.43",
        nvmlShutdown=lambda: calls.append("shutdown"),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    assert sweep_utils.nvml_driver_version() == "620.43"
    assert calls == ["init", "shutdown"]


def test_search_loop_profile_reports_utilization_and_phase_fractions() -> None:
    sweep = object.__new__(SweepEngine)
    # Historical capacity stays valid when a resumed run changes worker count.
    sweep.options = SimpleNamespace(worker_count=4)
    sweep._search_profile_lock = engine.threading.Lock()
    sweep._search_loop_wall_seconds = 10.0
    sweep._search_worker_busy_seconds = 12.0
    sweep._search_worker_capacity_seconds = 20.0
    sweep._search_loop_active_since = None
    sweep._search_loop_seconds = {"select_menu": 2.0, "proposing": 3.0}
    sweep._search_loop_calls = {"select_menu": 4, "proposing": 5}

    profile = sweep._search_loop_profile()

    assert profile["worker_capacity_seconds"] == 20.0
    assert profile["worker_utilization"] == pytest.approx(0.6)
    assert profile["profiled_main_thread_seconds"] == 5.0
    assert profile["unprofiled_main_thread_seconds"] == 5.0
    assert profile["phases"]["select_menu"] == {
        "seconds": 2.0,
        "calls": 4,
        "wall_fraction": 0.2,
    }


def test_lineage_contributions_distinguish_own_origin_from_lineage() -> None:
    incumbent = SpaceCandidate(options={"value": 0}, tile_n=64)
    seed = SpaceCandidate(
        options={"value": 1},
        tile_n=64,
        parent_ids=(incumbent.candidate_id,),
        move_kind="seed_stages",
    )
    crossover = SpaceCandidate(
        options={"value": 2},
        tile_n=64,
        parent_ids=(seed.candidate_id,),
        move_kind="crossover",
    )
    selected_child = SpaceCandidate(
        options={"value": 3},
        tile_n=64,
        parent_ids=(crossover.candidate_id,),
        move_kind="flags",
    )
    sweep = object.__new__(SweepEngine)
    sweep.candidates = {
        candidate.candidate_id: candidate
        for candidate in (incumbent, seed, crossover, selected_child)
    }
    sweep.incumbent_ids = {incumbent.candidate_id}
    sweep._seed_candidates = [seed]
    sweep.search = SimpleNamespace(two_step_depth={crossover.candidate_id: 0})

    contributions = sweep._lineage_contributions(
        (
            incumbent.candidate_id,
            seed.candidate_id,
            selected_child.candidate_id,
        )
    )

    assert contributions == {
        "incumbent": {"selected_entries": 1},
        "seed": {"selected_entries": 1},
        "crossover_lineage": {"selected_entries": 1},
        "two_step_lineage": {"selected_entries": 1},
    }


def test_clock_summary_and_low_clock_fraction() -> None:
    summary = summarize_gpu_clocks(
        [
            {
                "phase": "search",
                "measurements": [
                    {
                        "sm_clock_mhz": 2424,
                        "sm_clock_max_mhz": 2424,
                        "clock_event_reasons": [],
                    },
                    {
                        "sm_clock_mhz": 2100,
                        "sm_clock_max_mhz": 2424,
                        "clock_event_reasons": ["power"],
                    },
                ],
            }
        ]
    )

    assert summary["search"]["median_sm_clock_mhz"] == 2262
    assert summary["search"]["low_clock_fraction"] == 0.5
    assert summary["search"]["clock_event_reasons"] == {"power": 1}


def test_best_final_time_prefers_retime_and_falls_back_to_search() -> None:
    search = {"a": {"shape": TimingValue.measured(8.0, 0.1)}}
    retime = {"a": {"shape": TimingValue.measured(10.0, 0.1)}}

    assert _best_final_time(retime, search, "shape", None) == 10.0
    assert _best_final_time({}, search, "shape", None) == 8.0


def test_reference_failed_finalist_is_pruned_from_timings() -> None:
    candidates = [Candidate("good", 64), Candidate("failed", 64)]
    timings = {
        "good": {"shape": TimingValue.measured(10.0)},
        "failed": {"shape": TimingValue.measured(1.0)},
    }

    retained, retained_timings = _prune_failed_finalists(
        candidates, timings, {"failed"}
    )

    assert [candidate.candidate_id for candidate in retained] == ["good"]
    assert set(retained_timings) == {"good"}


def test_shape_label_includes_k_and_parallelism() -> None:
    shape = SimpleNamespace(
        timing_num_tokens=64,
        local_num_experts=112,
        fc2_n=3584,
        fc2_k=7168,
        tp=2,
        ep=4,
        routing="uniform",
    )

    assert _shape_label(shape, "fc2") == "T=64 E=112 N=3584 K=7168 TP/EP=2/4 R=uniform"


def test_search_failure_summary_includes_errors_ooms_and_cuda_faults() -> None:
    sweep = SweepEngine.__new__(SweepEngine)
    sweep.records = [
        {
            "phase": "search",
            "status": "error",
            "failure": "run_error",
            "error": "CUDA out of memory",
            "tile_n": 64,
            "move_kind": "stages_one",
            "candidate_id": "oom",
        },
        {
            "phase": "search",
            "status": "error",
            "failure": "cuda_fault",
            "error": "misaligned address",
            "tile_n": 16,
            "move_kind": "tile_k",
            "candidate_id": "fault",
        },
    ]

    failures = sweep._search_failure_summary()

    assert {row["failure"] for row in failures} == {"cuda_oom", "cuda_fault"}
    assert sum(row["count"] for row in failures) == 2


def test_reference_cuda_fault_terminates_worker_but_oom_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = SpaceCandidate(options={}, tile_n=64)
    materialized = SimpleNamespace()
    fake_cuda = SimpleNamespace(
        empty_cache=lambda: None,
        memory_allocated=lambda: 0,
        memory_reserved=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setattr(engine, "_candidate_from_payload", lambda _value: candidate)
    monkeypatch.setattr(
        engine,
        "ConfigContext",
        lambda **values: SimpleNamespace(**values),
    )
    monkeypatch.setattr(
        engine,
        "materialize_candidate",
        lambda *_args, **_kwargs: materialized,
    )
    monkeypatch.setattr(
        engine,
        "is_applicable",
        lambda *_args, **_kwargs: (True, None),
    )
    monkeypatch.setattr(engine, "_record_gpu_memory", lambda _record: None)
    payload = {
        "action": "reference",
        "candidate": {},
        "context": {"fc": "fc2"},
        "shapes": [{"shape_id": "shape", "fc2_k": 64}],
    }

    def illegal_access(*_args, **_kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(engine, "reference_check_candidate", illegal_access)
    fault = engine.run_worker_payload(payload)
    assert fault["failure"] == FailureKind.CUDA_FAULT.value
    assert fault["_terminate_worker"] is True

    def cuda_oom(*_args, **_kwargs):
        raise RuntimeError("CUDA error: out of memory")

    monkeypatch.setattr(engine, "reference_check_candidate", cuda_oom)
    oom = engine.run_worker_payload(payload)
    assert oom["status"] == "oom"
    assert "failure" not in oom
    assert "_terminate_worker" not in oom


def test_worker_sync_probe_turns_sticky_cuda_fault_into_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "output.json"
    progress_path = tmp_path / "progress.json"
    payload_path.write_text("{}", encoding="utf-8")
    request = {
        "payload": str(payload_path),
        "output": str(output_path),
        "progress": str(progress_path),
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    monkeypatch.setattr(
        engine,
        "run_worker_payload",
        lambda *_args, **_kwargs: {"status": "ok"},
    )

    def synchronize() -> None:
        raise RuntimeError("misaligned address")

    fake_cuda = SimpleNamespace(synchronize=synchronize, empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    engine.run_worker_loop()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "worker_error"
    assert result["failure"] == FailureKind.CUDA_FAULT.value


def test_retime_incomplete_reason_distinguishes_oom_timeout_and_deadline() -> None:
    assert (
        SweepEngine._retime_incomplete_reason(None)
        == "candidate was not re-timed before the deadline"
    )
    assert (
        SweepEngine._retime_incomplete_reason(
            {"status": "error", "error": "CUDA out of memory"}
        )
        == "CUDA out of memory during final re-timing"
    )
    assert (
        SweepEngine._retime_incomplete_reason(
            {"status": "timeout", "failure": "run_timeout", "error": "timed out"}
        )
        == "timed out"
    )


def _portfolio_fixture():
    shapes = [Shape("heavy", 3.0), Shape("small"), Shape("other", 2.0)]
    candidates = [
        Candidate("a", 64),
        Candidate("b", 64),
        Candidate("c", 128),
        Candidate("d", 128),
    ]
    timings = {
        "a": {"heavy": 10.0, "small": 30.0, "other": NOT_APPLICABLE},
        "b": {"heavy": 11.0, "small": 10.0, "other": NOT_APPLICABLE},
        "c": {"heavy": FAILED, "small": 11.0, "other": 5.0},
        "d": {"heavy": FAILED, "small": NOT_APPLICABLE, "other": 5.5},
    }
    return candidates, shapes, timings


def test_seed_cap_round_robins_tile_groups() -> None:
    candidates = [
        SpaceCandidate({"value": index}, tile_n=tile_n)
        for tile_n in (8, 8, 8, 16, 16, 32)
        for index in range(1)
    ]
    capped = cap_candidates_round_robin(candidates, limit=5)
    assert [candidate.tile_n for candidate in capped] == [8, 16, 32, 8, 16]


def test_seed_cap_prunes_stale_incumbent_ids() -> None:
    pending = [
        SpaceCandidate({"value": index}, tile_n=tile_n)
        for index, tile_n in enumerate((8, 8, 16, 16, 32))
    ]
    candidates = {candidate.candidate_id: candidate for candidate in pending}
    incumbents = set(candidates)

    retained, dropped = cap_and_prune_candidates(
        candidates, incumbents, pending, limit=3
    )

    assert [candidate.tile_n for candidate in retained] == [8, 16, 32]
    assert len(dropped) == 2
    assert incumbents == set(candidates)
    assert all(candidate_id in candidates for candidate_id in incumbents)


def test_resume_identity_ignores_operational_fields() -> None:
    base = {
        "target": {"fc": "fc1", "tokens": [1, 2]},
        "timing": {
            "warmup_iters": 10,
            "repeat_iters": 30,
            "protocol_version": 1,
            "compile_timeout_seconds": 600,
        },
        "search": {
            "seed": 42,
            "menu_size": 4,
            "max_input_stages": 16,
            "max_accumulator_stages": 10,
            "max_c_smem_stages": 4,
            "max_workid_stages": 8,
            "max_sf_ring_ratio": 4,
            "worker_count": 16,
        },
    }
    resumed = json.loads(json.dumps(base))
    resumed["timing"]["compile_timeout_seconds"] = 120
    resumed["search"]["worker_count"] = 4
    resumed["search"]["max_seeds"] = 64
    assert manifest_identity(base) == manifest_identity(resumed)
    resumed["timing"]["repeat_iters"] = 31
    assert manifest_identity(base) != manifest_identity(resumed)


def test_resume_identity_matches_manifest_reloaded_from_json() -> None:
    current = {
        "target": {
            "fc": "fc1",
            "shape_launch_keys": [{"selected_tile_ns": (64, 128), "fc1_n": 512}],
        },
        "timing": {"warmup_iters": 10, "repeat_iters": 30, "protocol_version": 1},
        "search": {
            "seed": 42,
            "menu_size": 4,
            "max_input_stages": 16,
            "max_accumulator_stages": 10,
            "max_c_smem_stages": 4,
            "max_workid_stages": 8,
        },
    }
    reloaded = json.loads(json.dumps(current))
    assert manifest_identity_mismatches(reloaded, current) == []
    reloaded["target"]["shape_launch_keys"][0]["fc1_n"] = 1024
    reloaded["search"]["seed"] = 7
    assert manifest_identity_mismatches(reloaded, current) == [
        "manifest.search.seed: resumed=7 current=42",
        "manifest.target.shape_launch_keys[0].fc1_n: resumed=1024 current=512",
    ]


def test_hotfix_resume_opens_segment_tags_records_and_strict_resume_rejects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    shape = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        global_num_tokens=1,
        tp=1,
        ep=1,
        routing="uniform",
    )
    context = engine.ConfigContext(
        sm_arch=107,
        fc="fc2",
        activation_type=0,
        dtype_a=0,
        dtype_b=0,
        dtype_c=0,
        top_k=1,
        num_local_experts=1,
    )
    incumbent = SpaceCandidate(options={}, tile_n=64)
    monkeypatch.setattr(engine, "build_sweep_shapes", lambda **_kwargs: (shape,))
    monkeypatch.setattr(engine, "context_for_shapes", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        engine,
        "enumerate_incumbents",
        lambda *_args, **_kwargs: (incumbent,),
    )
    monkeypatch.setattr(engine, "resolve_timing_backend", lambda: "cuda_event")
    monkeypatch.setattr(
        engine,
        "gpu_identity",
        lambda _gpu: {"gpu_name": "test-gpu", "gpu_uuid": "GPU-test"},
    )
    fingerprints = iter(
        (
            {"prims_ts_source_digest": "base"},
            {"prims_ts_source_digest": "hotfix"},
            {"prims_ts_source_digest": "hotfix"},
            {"prims_ts_source_digest": "strict-change"},
        )
    )
    monkeypatch.setattr(engine, "code_fingerprint", lambda: next(fingerprints))
    options = SweepOptions(
        out_dir=tmp_path,
        fc="fc2",
        model="deepseek-v4-pro-nvfp4",
        sm_arch=107,
        tp=(1,),
        ep=(1,),
        tokens=(1,),
        final_tokens=(1,),
        budget_minutes=1.0,
        tail_minutes=0.0,
        menu_size=1,
        worker_count=1,
        gpu="0",
    )

    SweepEngine(options)
    resumed = SweepEngine(replace(options, resume=True))

    assert resumed.resume_segment == 1
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["resume_history"][-1]["segment"] == 1
    assert manifest["resume_history"][-1]["code"]["prims_ts_source_digest"] == "hotfix"

    appended = []
    record = resumed._record(
        SimpleNamespace(append=appended.append),
        incumbent,
        {"status": "ok", "measurements": []},
        phase="search",
    )
    assert record["resume_segment"] == 1
    assert appended == [record]

    same_code = SweepEngine(replace(options, resume=True))
    same_manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert same_code.resume_segment == 1
    assert len(same_manifest["resume_history"]) == 1

    with pytest.raises(ValueError, match="code fingerprint changed"):
        SweepEngine(replace(options, resume=True, strict_resume=True))


def test_retry_selection_uses_latest_transient_failure_only() -> None:
    records = [
        {"phase": "search", "candidate_id": "old", "status": "timeout"},
        {"phase": "search", "candidate_id": "old", "status": "ok"},
        {"phase": "search", "candidate_id": "timeout", "status": "timeout"},
        {"phase": "search", "candidate_id": "worker", "status": "worker_error"},
        {
            "phase": "search",
            "candidate_id": "oom",
            "status": "error",
            "error": "OutOfMemoryError: CUDA out of memory",
        },
        {"phase": "retime", "candidate_id": "retime", "status": "timeout"},
    ]
    assert retryable_failed_ids(records) == ["timeout", "worker", "oom"]


def test_speedup_vs_pre_sweep_uses_best_incumbent_and_usable_candidates() -> None:
    def record(candidate_id, status="ok", **times):
        return {
            "phase": "search",
            "candidate_id": candidate_id,
            "status": status,
            "measurements": [
                {"shape_id": key, "status": "ok", "median_us": value}
                for key, value in times.items()
            ],
        }

    records = [
        record("inc_a", s1=10.0, s2=20.0),
        record("inc_b", s1=12.0, s2=16.0),
        record("new_fast", s1=5.0, s2=16.0),
        record("new_failed", status="timeout", s2=1.0),
        {
            "phase": "retime",
            "candidate_id": "retime_only",
            "status": "ok",
            "measurements": [{"shape_id": "s2", "status": "ok", "median_us": 1.0}],
        },
        record("new_only", s3=4.0),
    ]
    rows = speedup_vs_pre_sweep(records, {"inc_a", "inc_b"}, ["s1", "s2", "s3"])
    assert rows["s1"]["speedup"] == pytest.approx(2.0)
    assert rows["s1"]["best_candidate_id"] == "new_fast"
    assert rows["s2"]["speedup"] == pytest.approx(1.0)
    assert rows["s3"]["pre_sweep_us"] is None and rows["s3"]["speedup"] is None

    labels = {"s1": "T=8 E=32 N=512", "s2": "T=64 E=32 N=512"}
    assert format_speedup_summary(rows, labels) == (
        "speedup vs pre-sweep best, per shape: min 1.00x (T=64 E=32 N=512), "
        "max 2.00x (T=8 E=32 N=512); 1 without a pre-sweep timing"
    )


def test_engine_speedup_rows_only_include_search_shapes() -> None:
    search_shape = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        global_num_tokens=32,
        tp=4,
        ep=4,
        routing="uniform",
    )
    search_shape_id = engine.shape_id(search_shape, "fc2")
    sweep = object.__new__(SweepEngine)
    sweep.options = SimpleNamespace(fc="fc2")
    sweep.shapes = (search_shape,)
    sweep.incumbent_ids = {"incumbent"}
    sweep.records = [
        {
            "phase": "search",
            "candidate_id": "incumbent",
            "status": "ok",
            "measurements": [
                {
                    "shape_id": search_shape_id,
                    "status": "ok",
                    "median_us": 10.0,
                }
            ],
        }
    ]

    assert list(sweep._speedup_rows()) == [search_shape_id]


def test_evaluate_batch_does_not_dispatch_candidate_with_no_live_shapes() -> None:
    sweep = object.__new__(SweepEngine)
    sweep.options = SimpleNamespace(fc="fc1")
    sweep.shapes = ()
    candidate = SpaceCandidate(options={}, tile_n=64)
    sweep.candidates = {candidate.candidate_id: candidate}
    sweep.records = []
    sweep.search = SearchState()

    records = sweep._evaluate_batch(
        None,
        [candidate],
        protocol=TimingProtocol(warmup_iters=0, repeat_iters=1),
        phase="search",
        shapes_by_candidate={candidate.candidate_id: []},
        record_results=False,
    )

    assert records == [
        {
            "status": "validation_error",
            "error": "no_live_shapes",
            "measurements": [],
            "candidate_id": candidate.candidate_id,
        }
    ]


def test_reference_consistency_flags_error_growth_but_only_counts_rel_mismatch() -> (
    None
):
    def record(phase, candidate_id, *, mean_abs, max_abs, max_rel):
        return {
            "phase": phase,
            "candidate_id": candidate_id,
            "checks": [
                {
                    "shape_id": "shape",
                    "passed": True,
                    "mean_abs": mean_abs,
                    "max_abs": max_abs,
                    "max_rel": max_rel,
                }
            ],
        }

    result = reference_consistency(
        [
            record(
                "reference_sanity",
                "incumbent",
                mean_abs=1.0,
                max_abs=2.0,
                max_rel=0.1,
            ),
            record(
                "reference",
                "same-errors",
                mean_abs=1.0,
                max_abs=2.0,
                max_rel=0.2,
            ),
            record(
                "reference",
                "larger-errors",
                mean_abs=2.0,
                max_abs=2.0,
                max_rel=0.1,
            ),
        ],
        selected_ids={"larger-errors"},
    )

    assert result["checked_count"] == 3
    assert result["flagged_count"] == 1
    assert result["max_rel_mismatch_count"] == 1
    assert result["flagged_menu_rows"][0]["candidate_id"] == "larger-errors"
    assert result["flagged_menu_rows"][0]["mean_abs_ratio"] == pytest.approx(2.0)


def test_worker_activity_summary_counts_phases_and_ignores_ages() -> None:
    activities = [
        WorkerActivity("compile", "K=7168", phase_started=10.0, request_started=10.0),
        WorkerActivity("compile", "K=2048", phase_started=50.0, request_started=40.0),
        WorkerActivity("lock_wait", "K=7168", phase_started=70.0, request_started=5.0),
        WorkerActivity("run", "14/27 abc", phase_started=99.0, request_started=20.0),
    ]
    key, text = summarize_worker_activity(activities, now=100.0)
    assert key == (("compile", 2), ("lock_wait", 1), ("run", 1))
    assert text.startswith("compile=2 lock_wait=1 run=1")
    assert "oldest compile 90s" in text
    assert "oldest lock_wait 30s" in text
    assert "timing shape 14/27 (request age 80s)" in text
    later_key, _ = summarize_worker_activity(activities, now=500.0)
    assert later_key == key
    assert summarize_worker_activity([], now=1.0) == ((), "no worker requests active")


def test_worker_failure_description_names_phase_and_limit() -> None:
    def result(phase: str, phase_seconds: float, failure: FailureKind) -> ProcessResult:
        return ProcessResult(
            status=WorkerStatus.TIMED_OUT,
            failure=failure,
            returncode=None,
            stdout="",
            stderr="",
            elapsed_seconds=130.0,
            phase=phase,
            phase_unit="K=7168",
            phase_elapsed_seconds=phase_seconds,
        )

    compile_kill = describe_worker_failure(
        result("compile", 120.5, FailureKind.COMPILE_TIMEOUT),
        compile_timeout_seconds=120.0,
        run_timeout_seconds=600.0,
    )
    assert compile_kill.startswith("killed in compile (K=7168) after 120.5s")
    assert "(compile limit 120s)" in compile_kill
    lock_kill = describe_worker_failure(
        result("lock_wait", 125.0, FailureKind.LOCK_TIMEOUT),
        compile_timeout_seconds=120.0,
        run_timeout_seconds=600.0,
    )
    assert "(sweep deadline)" in lock_kill


def test_timing_values_preserve_applicability_and_failure() -> None:
    assert NOT_APPLICABLE.kind is TimingKind.NOT_APPLICABLE
    assert FAILED.kind is TimingKind.FAILED
    assert TimingValue.measured(2.0, 0.1).is_measured
    with pytest.raises(ValueError, match="positive"):
        TimingValue.measured(0.0)


def test_greedy_menu_is_weighted_cap_aware_and_reports_regret() -> None:
    candidates, shapes, timings = _portfolio_fixture()
    result = select_greedy_menu(
        candidates,
        shapes,
        timings,
        menu_size={64: 1, 128: 1},
        min_gain=0.0,
    )

    assert result.selected == ("b", "c")
    assert result.by_tile_n == {64: ("b",), 128: ("c",)}
    assert result.average_regret == pytest.approx(0.05)
    assert result.worst_regret == pytest.approx(0.10)
    assert result.weighted_log_regret == pytest.approx(3 * math.log(1.1) / 6)
    assert result.marginal_contributions["b"] > 0
    assert result.marginal_contributions["c"] > 0


def test_greedy_menu_default_stops_when_improvement_is_within_noise() -> None:
    candidates = [Candidate("base", 64), Candidate("noise", 64), Candidate("real", 64)]
    shapes = [Shape("s")]
    timings = {
        "base": {"s": TimingValue.measured(10.0, 0.1)},
        "noise": {"s": TimingValue.measured(9.95, 0.1)},
        "real": {"s": TimingValue.measured(9.5, 0.1)},
    }

    result = select_greedy_menu(candidates, shapes, timings, menu_size=3)

    assert result.selected == ("real",)


def test_final_selection_can_use_all_retime_routings() -> None:
    uniform = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        global_num_tokens=32,
        tp=4,
        ep=4,
        routing="uniform",
    )
    zipf = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        global_num_tokens=32,
        tp=4,
        ep=4,
        routing="zipf:0.8",
    )
    uniform_id = engine.shape_id(uniform, "fc2")
    zipf_id = engine.shape_id(zipf, "fc2")
    candidates = [Candidate("uniform-specialist", 32), Candidate("robust", 32)]
    timings = {
        "uniform-specialist": {uniform_id: 1.0, zipf_id: 100.0},
        "robust": {uniform_id: 1.1, zipf_id: 1.0},
    }

    uniform_only = select_greedy_menu(
        candidates,
        selection_portfolio_shapes([uniform, zipf], fc="fc2", routings=("uniform",)),
        timings,
        menu_size=1,
    )
    all_routings = select_greedy_menu(
        candidates,
        selection_portfolio_shapes(
            [uniform, zipf], fc="fc2", routings=("uniform", "zipf:0.8")
        ),
        timings,
        menu_size=1,
    )

    assert uniform_only.selected == ("uniform-specialist",)
    assert all_routings.selected == ("robust",)


def test_greedy_menu_reports_uncovered_shape() -> None:
    result = select_greedy_menu(
        [Candidate("a", 64)],
        [Shape("s")],
        {"a": {"s": 1.0}},
        menu_size=0,
        min_gain=0.0,
    )
    assert result.selected == ()
    assert math.isinf(result.average_regret)
    assert math.isinf(result.worst_regret)


def test_final_retime_pool_is_the_required_union() -> None:
    candidates, shapes, timings = _portfolio_fixture()
    pool = build_final_retime_pool(
        candidates,
        shapes,
        timings,
        top_k=1,
        near_best_fraction=0.05,
        interim_menu=["d"],
        contributors={"a": 0.2},
        incumbents=["old"],
    )

    assert pool.candidates == ("d", "a", "old", "b", "c")
    assert "top_k:heavy" in pool.reasons["a"]
    assert "top_k:small" in pool.reasons["b"]
    assert "top_k:other" in pool.reasons["c"]
    assert pool.reasons["d"] == ("interim_menu",)
    assert pool.reasons["old"] == ("incumbent",)


def test_final_retime_pool_caps_and_round_robins_near_best_shapes() -> None:
    candidates = [Candidate(name, 64) for name in ("a", "b", "c", "d")]
    shapes = [Shape("left"), Shape("right")]
    timings = {
        "a": {"left": 1.0, "right": 1.03},
        "b": {"left": 1.01, "right": 1.02},
        "c": {"left": 1.02, "right": 1.01},
        "d": {"left": 1.03, "right": 1.0},
    }

    pool = build_final_retime_pool(
        candidates,
        shapes,
        timings,
        top_k=1,
        near_best_fraction=0.05,
        max_near_best_per_shape=2,
    )

    assert pool.candidates == ("a", "d", "b", "c")
    assert "near_best:left" in pool.reasons["b"]
    assert "near_best:right" in pool.reasons["c"]


def test_noise_aware_success_and_two_step_bands() -> None:
    thresholds = NoiseAwareThresholds(
        success_fraction=0.03,
        two_step_fraction=0.10,
        noise_multiplier=1.0,
    )
    quiet = thresholds.classify_one(1.04, 1.0)
    noisy = thresholds.classify_one(
        TimingValue.measured(1.04, 0.02),
        TimingValue.measured(1.0, 0.02),
    )
    two_step = thresholds.classify_one(1.08, 1.0)

    assert quiet is CandidateOutcome.TWO_STEP
    assert noisy is CandidateOutcome.SUCCESS
    assert two_step is CandidateOutcome.TWO_STEP
    assert thresholds.classify({"a": FAILED}, {"a": 1.0}) is CandidateOutcome.FAILED


def test_parent_aware_success_requires_a_material_near_best_improvement() -> None:
    thresholds = NoiseAwareThresholds(
        success_fraction=0.03,
        two_step_fraction=0.10,
        noise_multiplier=1.0,
    )
    best = {"s": TimingValue.measured(10.0, 0.01)}
    parent = {"s": TimingValue.measured(10.2, 0.01)}

    assert (
        thresholds.classify_against_parent(
            {"s": TimingValue.measured(10.0, 0.01)},
            parent,
            best,
        )
        is CandidateOutcome.SUCCESS
    )
    assert (
        thresholds.classify_against_parent(
            {"s": TimingValue.measured(10.19, 0.01)},
            parent,
            best,
        )
        is CandidateOutcome.TWO_STEP
    )


def test_beta_arm_accounts_for_all_attempts() -> None:
    arm = BetaArm()
    arm.record_outcome(CandidateOutcome.SUCCESS)
    arm.record_outcome(CandidateOutcome.TWO_STEP)
    arm.record_outcome(CandidateOutcome.FAILED)
    arm.record_rejected(RejectionReason.ILLEGAL)
    arm.record_duplicate()
    assert (arm.attempts, arm.successes, arm.failures) == (3, 1, 2)
    assert (arm.rejected, arm.duplicates, arm.proposed) == (1, 1, 5)
    assert arm.evaluation_yield == pytest.approx(0.6)
    assert arm.duplicate_rate == pytest.approx(0.2)
    assert arm.duplicate_penalty == pytest.approx(5 / 6)
    assert arm.rejection_reasons == {"illegal": 1}
    assert BetaArm.from_dict(arm.to_dict()) == arm
    assert BetaArm.from_dict({"attempts": 1, "successes": 1, "failures": 0}) == BetaArm(
        attempts=1, successes=1
    )


def test_parent_move_tracker_exhausts_duplicate_heavy_pairs() -> None:
    tracker = ParentMoveTracker(duplicate_limit=2)
    assert tracker.available_moves("parent", ("stages", "flags")) == (
        "stages",
        "flags",
    )
    assert not tracker.record_duplicate("parent", "stages")
    assert tracker.record_duplicate("parent", "stages")
    assert tracker.available_moves("parent", ("stages", "flags")) == ("flags",)
    tracker.mark_exhausted("parent", "flags")
    assert tracker.parent_is_exhausted("parent", ("stages", "flags"))
    restored = ParentMoveTracker(
        duplicate_limit=2,
        duplicate_counts={("restored", "cluster"): 2},
    )
    assert restored.is_exhausted("restored", "cluster")


def test_thompson_selector_penalizes_duplicate_heavy_arms(monkeypatch) -> None:
    selector = AdaptiveMoveSelector(seed=1, exploration_floor=0.0)
    for _ in range(3):
        selector.record_duplicate("group", "duplicate_heavy")
    monkeypatch.setattr(BetaArm, "sample", lambda self, rng, **kwargs: 1.0)

    assert selector.choose("group", ["duplicate_heavy", "novel"]) == "novel"


def test_thompson_selector_is_deterministic_across_serialization() -> None:
    selector = AdaptiveMoveSelector(seed=17, exploration_floor=0.2)
    for _ in range(5):
        move = selector.choose("fc1:64", ["stages", "cluster", "flags"])
        selector.record("fc1:64", move, move == "cluster")

    restored = AdaptiveMoveSelector.from_dict(
        json.loads(json.dumps(selector.to_dict()))
    )
    expected = [
        selector.choose("fc1:64", ["stages", "cluster", "flags"]) for _ in range(20)
    ]
    actual = [
        restored.choose("fc1:64", ["stages", "cluster", "flags"]) for _ in range(20)
    ]
    assert actual == expected


def test_parent_pool_uses_weighted_coverage_and_contribution() -> None:
    shapes = [Shape("important", 9.0), Shape("minor", 1.0)]
    timings = {
        "wide": {"important": 1.02, "minor": 2.0},
        "narrow": {"important": 1.2, "minor": 1.0},
        "menu": {"important": 1.4, "minor": 1.4},
    }
    pool = select_parent_pool(
        ["wide", "narrow", "menu"],
        shapes,
        timings,
        {"important": 1.0, "minor": 1.0},
        menu=["menu"],
        contributions={"narrow": 0.2},
        max_parents=2,
    )

    assert pool.parents == ("menu", "wide")
    assert pool.scores["wide"].coverage_weight == 9.0
    assert pool.scores["wide"].score > pool.scores["menu"].score


def test_empty_parent_pool_falls_back_to_top_mean_regret() -> None:
    shapes = [Shape("heavy", 3.0), Shape("light", 1.0)]
    timings = {
        "middle": {"heavy": 1.2, "light": 1.4},
        "best_mean": {"heavy": 1.1, "light": 1.6},
        "worst": {"heavy": 1.5, "light": 1.1},
    }
    ranked = rank_parents_by_mean_regret(
        ["middle", "best_mean", "worst"],
        shapes,
        timings,
        {"heavy": 1.0, "light": 1.0},
        top_k=2,
    )
    assert [candidate_id for candidate_id, _regret in ranked] == [
        "best_mean",
        "middle",
    ]

    pool = select_parent_pool(
        ["middle", "best_mean", "worst"],
        shapes,
        timings,
        {"heavy": 1.0, "light": 1.0},
        near_best_fraction=0.03,
        max_parents=2,
        fallback_top_k=2,
    )
    assert pool.parents == ("best_mean", "middle")
    assert all(pool.scores[parent].is_fallback for parent in pool.parents)


def test_parent_pool_tops_up_sparse_qualified_pool() -> None:
    shapes = [Shape("shape")]
    timings = {
        "qualified": {"shape": 1.0},
        "fallback-a": {"shape": 1.2},
        "fallback-b": {"shape": 1.3},
    }

    pool = select_parent_pool(
        tuple(timings),
        shapes,
        timings,
        {"shape": 1.0},
        min_parents=3,
    )

    assert pool.parents == ("qualified", "fallback-a", "fallback-b")
    assert pool.scores["fallback-a"].is_fallback


def test_group_allocator_preserves_floor_and_reduces_stagnant_share() -> None:
    progress = {
        "a": GroupProgress(since_improvement=40),
        "b": GroupProgress(),
    }
    allocator = GroupBudgetAllocator(
        minimum_share=0.1,
        stagnation_window=40,
        stagnation_reduction=0.25,
    )
    shares = allocator.shares({"a": 1.0, "b": 1.0}, progress)
    allocation = allocator.allocate(100, {"a": 1.0, "b": 1.0}, progress)

    assert shares["a"] >= 0.1
    assert shares["a"] < shares["b"]
    assert sum(allocation.values()) == 100
    assert allocation["a"] >= 10
    assert allocation["a"] < allocation["b"]


def test_group_allocator_repeatedly_decays_then_freezes_at_floor() -> None:
    allocator = GroupBudgetAllocator(
        minimum_share=0.05,
        stagnation_window=10,
        stagnation_reduction=0.5,
        freeze_after_windows=2,
    )
    fresh = allocator.shares(
        {"stagnant": 1.0, "active": 1.0},
        {"stagnant": GroupProgress(since_improvement=0)},
    )
    decayed = allocator.shares(
        {"stagnant": 1.0, "active": 1.0},
        {"stagnant": GroupProgress(since_improvement=10)},
    )
    frozen = allocator.shares(
        {"stagnant": 1.0, "active": 1.0},
        {"stagnant": GroupProgress(since_improvement=20)},
    )

    assert fresh["stagnant"] > decayed["stagnant"] > frozen["stagnant"]
    assert frozen["stagnant"] == pytest.approx(0.05)
    assert GroupProgress(since_improvement=20).is_frozen(10)
    all_frozen = allocator.shares(
        {"left": 1.0, "right": 3.0},
        {
            "left": GroupProgress(since_improvement=20),
            "right": GroupProgress(since_improvement=20),
        },
    )
    assert sum(all_frozen.values()) == pytest.approx(1.0)
    assert all_frozen["left"] == pytest.approx(all_frozen["right"])


def test_group_allocator_prefers_recently_improving_group() -> None:
    allocator = GroupBudgetAllocator(minimum_share=0.05, stagnation_window=4)
    progress = {
        "improving": GroupProgress(improvement_history=[True, False, True, True]),
        "flat": GroupProgress(improvement_history=[False, False, False, False]),
    }

    shares = allocator.shares({"improving": 1.0, "flat": 1.0}, progress)

    assert shares["improving"] > shares["flat"]
    assert shares["flat"] >= 0.05


def test_contested_group_weights_count_only_well_sampled_contenders() -> None:
    candidates = [
        Candidate("a1", 8),
        Candidate("a2", 8),
        Candidate("b1", 16),
        Candidate("b2", 16),
    ]
    shapes = [Shape("heavy", 2.0), Shape("light", 1.0)]
    timings = {
        "a1": {"heavy": 1.0, "light": 0.99},
        "a2": {"heavy": 1.1},
        "b1": {"heavy": 1.2, "light": 1.0},
        "b2": {"heavy": 1.3, "light": 1.01},
    }

    competition = contested_group_weights(
        candidates,
        shapes,
        timings,
        min_samples=2,
    )

    assert competition.weights["8"] > competition.weights["16"] > 0


def test_contested_group_weights_ignore_ineligible_shapes() -> None:
    competition = contested_group_weights(
        [Candidate("a", 8), Candidate("b", 16)],
        [Shape("only-a", 2.0), Shape("only-b", 1.0)],
        {"a": {"only-a": 1.0}, "b": {"only-b": 1.0}},
        eligible={"8": {"only-a"}, "16": {"only-b"}},
    )

    assert set(competition.weights) == {"8", "16"}
    assert all(weight > 0 for weight in competition.weights.values())


def test_group_allocator_gives_zero_weight_groups_no_floor() -> None:
    shares = GroupBudgetAllocator(minimum_share=0.05).shares(
        {"active": 1.0, "retired": 0.0}
    )

    assert shares["active"] == pytest.approx(1.0)
    assert shares["retired"] == 0.0


def test_group_allocator_does_not_revive_all_zero_groups() -> None:
    shares = GroupBudgetAllocator(minimum_share=0.05).shares(
        {"retired-a": 0.0, "retired-b": 0.0}
    )

    assert not any(shares.values())


def test_group_progress_counts_cumulative_improvement() -> None:
    progress = GroupProgress()
    assert progress.record(1.0)
    assert not progress.record(0.996)
    # This is >0.5% from the last significant point, despite two small steps.
    assert progress.record(0.992)
    assert progress.since_improvement == 0


def test_group_objective_uses_only_the_groups_per_shape_best() -> None:
    objective = group_best_objective(
        ("a", "b"),
        [Shape("heavy", 3.0), Shape("light", 1.0)],
        {
            "a": {"heavy": 10.0, "light": 40.0},
            "b": {"heavy": 20.0, "light": 20.0},
            "other": {"heavy": 1.0, "light": 1.0},
        },
    )

    assert objective == pytest.approx(math.exp((3 * math.log(10) + math.log(20)) / 4))


def test_menu_speedup_curve_reports_each_prefix() -> None:
    curve = menu_speedup_curve(
        ["first", "second"],
        [Shape("a"), Shape("b")],
        {
            "inc": {
                "a": TimingValue.measured(10.0),
                "b": TimingValue.measured(20.0),
            },
            "first": {
                "a": TimingValue.measured(5.0),
                "b": TimingValue.measured(20.0),
            },
            "second": {
                "a": TimingValue.measured(10.0),
                "b": TimingValue.measured(10.0),
            },
        },
        {"inc"},
    )

    assert [row["entries"] for row in curve] == [1, 2]
    assert curve[0]["geomean_speedup_vs_incumbent"] == pytest.approx(math.sqrt(2))
    assert curve[1]["geomean_speedup_vs_incumbent"] == pytest.approx(2.0)


def test_search_state_json_round_trip_preserves_rng_and_counters() -> None:
    selector = AdaptiveMoveSelector(seed=123, exploration_floor=0.1)
    selected = selector.choose("fc2:128", ["tile_k", "warps"])
    selector.record("fc2:128", selected, CandidateOutcome.SUCCESS)
    state = SearchState(
        selector=selector,
        groups={"fc2:128": GroupProgress(3, 2, 0.4)},
        evaluated=["sig-a"],
        two_step_remaining={"sig-c": 4},
        two_step_depth={"sig-c": 1},
        pending_confirmations={"sig-d": ["shape-a"]},
        group_dispatch_stats={
            "64": {
                "draws": 3,
                "proposals": 0,
                "drops_by_reason": {"no_move": 3},
            }
        },
        dispatched=5,
        completed=4,
    )

    restored = SearchState.from_json(state.to_json())
    assert restored.to_dict() == state.to_dict()
    assert restored.selector.choose("fc2:128", ["tile_k", "warps"]) == (
        state.selector.choose("fc2:128", ["tile_k", "warps"])
    )


def test_search_state_drops_exhausted_two_step_budgets_on_resume() -> None:
    value = SearchState().to_dict()
    value["two_step_remaining"] = {"negative": -1, "zero": 0, "live": 2}

    restored = SearchState.from_dict(value)

    assert restored.two_step_remaining == {"live": 2}


def test_resume_keeps_legacy_invalidated_confirmation_excluded() -> None:
    value = SearchState().to_dict()
    value["invalidated_pairs"] = [["candidate", "shape"]]

    restored = SearchState.from_dict(value)

    assert restored.pending_confirmations == {"candidate": ["shape"]}


def test_two_step_child_budget_is_bounded_by_lineage_depth() -> None:
    sweep = object.__new__(SweepEngine)
    sweep.search = SearchState(two_step_depth={"parent": 1})

    sweep._record_two_step_child(
        "at-limit",
        "parent",
        CandidateOutcome.TWO_STEP,
        status="ok",
    )
    sweep._record_two_step_child(
        "first",
        "root",
        CandidateOutcome.TWO_STEP,
        status="ok",
    )

    assert "at-limit" not in sweep.search.two_step_remaining
    assert sweep.search.two_step_remaining == {"first": 4}
    assert sweep.search.two_step_depth == {"parent": 1, "first": 0}


def test_adaptive_confirms_only_after_inflight_results_are_drained() -> None:
    sweep = object.__new__(SweepEngine)
    sweep.search_deadline = 0.0
    sweep.deadline = 0.0
    sweep.options = SimpleNamespace(reference_reserve_minutes=0.0)
    sweep._search_stop_minute = None
    sweep.candidates = {}
    events = []
    sweep._progress = lambda *_args, **_kwargs: None
    sweep._timing_inputs = lambda **_kwargs: ([], [], {})
    sweep._start_search_profile = lambda: None
    sweep._stop_search_profile = lambda: None
    sweep._record_search_phase = lambda *_args, **_kwargs: None
    sweep._report_search_profile = lambda **_kwargs: None

    def drain(_writer, *, wait_for_all=False, **_kwargs):
        assert wait_for_all
        events.append("drain")
        return [{"candidate_id": "child"}]

    sweep._harvest_search_inflight = drain
    sweep._process_adaptive_records = lambda *_args: events.append("process")
    sweep._confirm_pending = lambda *_args, **_kwargs: events.append("confirm")
    sweep._checkpoint = lambda **_kwargs: events.append("checkpoint")
    sweep._search_elapsed_minutes = lambda: 1.0

    sweep._adaptive(
        SimpleNamespace(),
        TimingProtocol(warmup_iters=0, repeat_iters=1),
    )

    assert events == ["drain", "process", "confirm", "checkpoint"]


def test_confirmed_timing_uses_conservative_maximum() -> None:
    value = engine._conservative_confirmed_timing(
        TimingValue.measured(10.0, 0.2),
        TimingValue.measured(15.0, 0.1),
    )

    assert value == TimingValue.measured(15.0, 0.2)


def test_suspicious_search_improvement_requires_confirmation() -> None:
    sweep = object.__new__(SweepEngine)
    sweep.options = SimpleNamespace(confirm_fraction=0.10)
    sweep.search = SearchState()

    sweep._mark_suspicious_records(
        (
            {
                "candidate_id": "candidate",
                "measurements": [
                    {"shape_id": "large", "status": "ok", "median_us": 8.0},
                    {"shape_id": "small", "status": "ok", "median_us": 9.5},
                ],
            },
        ),
        {
            "large": TimingValue.measured(10.0),
            "small": TimingValue.measured(10.0),
        },
    )

    assert sweep.search.pending_confirmations == {"candidate": ["large"]}


def test_failed_confirmation_remains_reportable_as_unconfirmed() -> None:
    shape = model_moe_shape(
        model="deepseek-v4-pro-nvfp4",
        global_num_tokens=1,
        tp=1,
        ep=1,
        routing="uniform",
    )
    shape_key = engine.shape_id(shape, "fc2")
    candidate = SpaceCandidate(options={}, tile_n=shape.selected_tile_ns[0])
    sweep = object.__new__(SweepEngine)
    sweep.options = SimpleNamespace(fc="fc2")
    sweep.shapes = (shape,)
    sweep.candidates = {candidate.candidate_id: candidate}
    sweep.search = SearchState(
        pending_confirmations={candidate.candidate_id: [shape_key]}
    )
    sweep._evaluate_batch = lambda *_args, **_kwargs: [
        {
            "candidate_id": candidate.candidate_id,
            "measurements": [{"shape_id": shape_key, "status": "run_error"}],
        }
    ]

    sweep._confirm_pending(
        SimpleNamespace(),
        TimingProtocol(warmup_iters=0, repeat_iters=1),
    )

    assert sweep.search.pending_confirmations == {candidate.candidate_id: [shape_key]}


def test_exhausted_parent_cannot_survive_on_impossible_crossover() -> None:
    tracker = ParentMoveTracker()
    for move in ("flags", "cluster"):
        tracker.mark_exhausted("parent", move)

    assert not engine._parent_has_unexhausted_move(
        tracker,
        "parent",
        ("flags", "cluster"),
        has_crossover_alternative=False,
    )
    assert engine._parent_has_unexhausted_move(
        tracker,
        "parent",
        ("flags", "cluster"),
        has_crossover_alternative=True,
    )


def test_final_timing_protocol_doubles_search_samples() -> None:
    search = TimingProtocol(warmup_iters=10, repeat_iters=30, cold_l2_cache=True)
    final = final_timing_protocol(search)
    assert final.warmup_iters == 10
    assert final.repeat_iters == 60
    assert final.cold_l2_cache is True

    longer = final_timing_protocol(TimingProtocol(warmup_iters=5, repeat_iters=80))
    assert longer.warmup_iters == 5
    assert longer.repeat_iters == 160
