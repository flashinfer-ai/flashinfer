# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from flashinfer.prims_ts.moe.tools.sweep import engine, menu_retime
from flashinfer.prims_ts.moe.tools.sweep.menu_config import load_menu_target
from flashinfer.prims_ts.moe.tools.sweep.menu_retime import (
    RetimeEngine,
    run_menu_retime,
)

MENU_DIR = Path(__file__).parents[3] / "flashinfer/prims_ts/moe/tools/sweep/menus"


@pytest.mark.parametrize(
    "target_name", ["sm107_nvfp4", "sm107_mxfp4_mxfp8", "sm107_mxfp8"]
)
def test_retime_engine_constructs_for_checked_in_targets(
    tmp_path, monkeypatch, target_name
):
    monkeypatch.setattr(engine, "enumerate_incumbents", lambda *_args: ())
    monkeypatch.setattr(engine, "resolve_timing_backend", lambda: "cuda_event")
    monkeypatch.setattr(
        engine,
        "gpu_identity",
        lambda _gpu: {"gpu_name": "test-gpu", "gpu_uuid": "GPU-test"},
    )
    monkeypatch.setattr(
        engine, "code_fingerprint", lambda: {"prims_ts_source_digest": "test"}
    )
    target = load_menu_target(MENU_DIR / f"{target_name}.json")

    for deployment in target.deployments:
        for fc in ("fc1", "fc2"):
            out_dir = tmp_path / deployment.name / fc
            retime = RetimeEngine(
                target=target,
                deployment=deployment,
                fc=fc,
                candidates=(),
                out_dir=out_dir,
                worker_count=1,
                gpu="0",
                resume=False,
            )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            assert manifest["target"]["sm_arch"] == target.sm_arch
            assert manifest["menu_retime"]["deployment"] == deployment.name
            assert retime.shapes == retime.final_shapes
            assert retime.shapes


def test_menu_resume_accepts_hotfix_and_starts_unstarted_sub_runs_fresh(
    tmp_path, monkeypatch
):
    resumed = {}

    class FakeRetimeEngine:
        def __init__(self, *, deployment, fc, out_dir, resume, **_kwargs):
            resumed[(deployment.name, fc)] = resume
            out_dir.mkdir(parents=True, exist_ok=True)

        def run_retime(self):
            pass

        def close(self):
            pass

    fingerprints = iter(
        ({"prims_ts_source_digest": "base"}, {"prims_ts_source_digest": "hotfix"})
    )
    monkeypatch.setattr(menu_retime, "RetimeEngine", FakeRetimeEngine)
    monkeypatch.setattr(menu_retime, "_matrix_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(menu_retime, "_target_quant_contexts", lambda _target: {})
    monkeypatch.setattr(menu_retime, "code_fingerprint", lambda: next(fingerprints))
    target = load_menu_target(MENU_DIR / "sm107_mxfp8.json")
    run_menu_retime(target, (), out_dir=tmp_path, worker_count=1)
    started = target.deployments[0].name
    (tmp_path / started / "fc1" / "run_manifest.json").write_text("{}")

    run_menu_retime(target, (), out_dir=tmp_path, worker_count=1, resume=True)

    assert resumed.pop((started, "fc1")) is True
    assert resumed and not any(resumed.values())


def _stub_engine(tmp_path, candidate_ids, records=(), unplannable=()):
    engine = object.__new__(RetimeEngine)
    engine.options = SimpleNamespace(warmup_iters=1, repeat_iters=2)
    engine.timing_backend = "test"
    engine.candidates = {
        candidate_id: SimpleNamespace(candidate_id=candidate_id)
        for candidate_id in candidate_ids
    }
    engine.menu_candidates = dict(engine.candidates)
    engine.records = list(records)
    engine.journal_path = tmp_path / "results.jsonl"
    engine.menu_label = "deployment/fc1"
    calls = []

    def progress(self, _message, *, force=False):
        pass

    def plan(self, candidate):
        if candidate.candidate_id in unplannable:
            raise ValueError("K=256: unsupported")
        return [f"shape-{candidate.candidate_id}"]

    def record(self, _writer, candidate, result, *, phase):
        calls.append(("record", candidate.candidate_id, phase, result["status"]))
        self.records.append(
            {**result, "phase": phase, "candidate_id": candidate.candidate_id}
        )

    def evaluate(self, _writer, candidates, **_kwargs):
        calls.append(("evaluate", [value.candidate_id for value in candidates]))
        for candidate in candidates:
            self.records.append(
                {
                    "phase": "menu_retime",
                    "candidate_id": candidate.candidate_id,
                    "status": "ok",
                }
            )

    def reference(self, _writer, candidates, *, shapes_by_candidate, **_kwargs):
        ids = [value.candidate_id for value in candidates]
        calls.append(("reference", ids))
        assert set(shapes_by_candidate) == set(ids)
        for candidate_id in ids:
            self.records.append(
                {
                    "phase": "menu_reference",
                    "candidate_id": candidate_id,
                    "status": "failed" if candidate_id == "wrong" else "ok",
                }
            )

    engine._progress = MethodType(progress, engine)
    engine._reference_shapes_for_candidate = MethodType(plan, engine)
    engine._record = MethodType(record, engine)
    engine._evaluate_batch = MethodType(evaluate, engine)
    engine._reference_batch = MethodType(reference, engine)
    return engine, calls


def test_retime_references_fresh_candidates_before_timing(tmp_path):
    engine, calls = _stub_engine(tmp_path, ["candidate"])
    engine.run_retime()

    assert calls == [("reference", ["candidate"]), ("evaluate", ["candidate"])]


def test_retime_resume_skips_completed_timing_and_reference(tmp_path):
    engine, calls = _stub_engine(
        tmp_path,
        ["candidate"],
        records=[
            {"phase": "menu_retime", "candidate_id": "candidate", "status": "ok"},
            {"phase": "menu_reference", "candidate_id": "candidate", "status": "ok"},
        ],
    )
    engine.run_retime()

    assert calls == []


def test_retime_resume_retries_transient_timing_failures(tmp_path):
    engine, calls = _stub_engine(
        tmp_path,
        ["timeout", "invalid"],
        records=[
            {"phase": "menu_reference", "candidate_id": "timeout", "status": "ok"},
            {"phase": "menu_reference", "candidate_id": "invalid", "status": "ok"},
            {"phase": "menu_retime", "candidate_id": "timeout", "status": "timeout"},
            {
                "phase": "menu_retime",
                "candidate_id": "invalid",
                "status": "validation_error",
            },
        ],
    )
    engine.run_retime()

    assert calls == [("evaluate", ["timeout"])]


def test_retime_does_not_time_reference_failure(tmp_path):
    engine, calls = _stub_engine(tmp_path, ["wrong"])
    engine.run_retime()

    assert calls == [("reference", ["wrong"])]


def test_retime_journals_unplannable_candidates_without_dispatch(tmp_path):
    engine, calls = _stub_engine(tmp_path, ["good", "bad"], unplannable={"bad"})
    engine.run_retime()

    assert calls == [
        ("record", "bad", "menu_reference", "not_applicable"),
        ("reference", ["good"]),
        ("evaluate", ["good"]),
    ]

    calls.clear()
    engine.run_retime()

    assert calls == []
