"""CPU tests for the cuDNN plan-build worker pool, driven by fake workers."""

import pathlib
import sys
import textwrap
import time

import pytest

import flashinfer
import flashinfer.gemm._cudnn_plan_pool as pool_mod
from flashinfer.gemm._cudnn_plan_build import Bf16OverrideGraphSpec

BUILD_MODULE = (
    pathlib.Path(flashinfer.__file__).parent / "gemm" / "_cudnn_plan_build.py"
).resolve()

SPEC = Bf16OverrideGraphSpec(
    batch=1,
    n=64,
    k=64,
    o_type=9,
    bias=False,
    cache_m=8,
    is_a_k_major=True,
    is_b_k_major=True,
    uid_a=0,
    uid_b=1,
    uid_bias=2,
    uid_o=3,
)
CANDIDATES = [(7, ((1, 2),)), (0, ()), (7, ((1, 3),))]

FAKE_WORKER = textwrap.dedent(
    """
    import importlib.util, os, sys, time
    spec = importlib.util.spec_from_file_location("b", {module!r})
    b = importlib.util.module_from_spec(spec)
    sys.modules["b"] = b
    spec.loader.exec_module(b)
    mode = sys.argv[1]
    candidates = {candidates!r}
    while True:
        req = b.read_frame(sys.stdin.buffer)
        if req is None:
            break
        if mode == "exit":
            sys.exit(3)
        if mode == "sleep":
            time.sleep(30)
        if mode == "garbage":
            sys.stdout.buffer.write(b"\\x05" + b"\\x00" * 7 + b"junk!")
            sys.stdout.buffer.flush()
            continue
        i, w = req["shard"]
        mine = list(candidates)
        if mode == "mismatch" and i == 1:
            mine = mine[:-1]
        plans = [
            (idx, tac, f"blob{{idx}}".encode())
            for idx, tac in enumerate(mine)
            if idx % w == i
        ]
        if mode == "partial" and i == 0:
            plans = plans[1:]
        if mode == "badindex":
            plans = [(99, mine[0], b"x")]
        b.write_frame(sys.stdout.buffer, {{
            "device_index": req["device_index"],
            "candidates": mine,
            "plans": plans,
            "error": None,
            "seen": {{"device_index": req["device_index"],
                      "visible": os.environ.get("CUDA_VISIBLE_DEVICES")}},
        }})
    """
)


@pytest.fixture
def fake_cmd(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(
        FAKE_WORKER.format(module=str(BUILD_MODULE), candidates=CANDIDATES)
    )
    return lambda mode: [sys.executable, "-I", str(script), mode]


def _pool(fake_cmd, mode, workers=2, timeout_s=10.0):
    pool = pool_mod.CudnnPlanBuildPool(
        workers, timeout_s=timeout_s, worker_cmd=fake_cmd(mode)
    )
    return pool


def test_results_are_reassembled_in_candidate_order(fake_cmd):
    pool = _pool(fake_cmd, "ok")
    try:
        result = pool.build(SPEC, device_index=0, heur_modes=[0, 2])
    finally:
        pool.shutdown()
    assert result == [(tac, f"blob{i}".encode()) for i, tac in enumerate(CANDIDATES)]


def test_mismatched_candidates_fall_back(fake_cmd):
    pool = _pool(fake_cmd, "mismatch")
    try:
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is None
        assert pool.broken
    finally:
        pool.shutdown()


def test_worker_exit_falls_back_and_stays_off(fake_cmd):
    pool = _pool(fake_cmd, "exit")
    try:
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is None
        start = time.monotonic()
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is None
        assert time.monotonic() - start < 0.5
    finally:
        pool.shutdown()


def test_worker_timeout_kills_process(fake_cmd):
    pool = _pool(fake_cmd, "sleep", workers=1, timeout_s=0.5)
    try:
        start = time.monotonic()
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is None
        assert time.monotonic() - start < 5
        assert all(p.poll() is not None for p in pool._all_processes())
    finally:
        pool.shutdown()


def test_request_carries_device_index_and_pins_the_device(fake_cmd, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    pool = _pool(fake_cmd, "ok", workers=1)
    try:
        assert pool.build(SPEC, device_index=1, heur_modes=[0]) is not None
        seen = pool.last_responses[0]["seen"]
    finally:
        pool.shutdown()
    assert seen == {"device_index": 1, "visible": "6"}


def test_shutdown_terminates_workers(fake_cmd):
    pool = _pool(fake_cmd, "ok")
    assert pool.build(SPEC, device_index=0, heur_modes=[0]) is not None
    procs = pool._all_processes()
    assert procs and all(p.poll() is None for p in procs)
    pool.shutdown()
    assert all(p.poll() is not None for p in procs)


@pytest.mark.parametrize(
    "value, workers, warns",
    [
        ("0", None, False),
        ("", None, False),
        ("two", None, True),
        ("-1", None, True),
        ("2", 2, False),
    ],
)
def test_env_parsing(monkeypatch, caplog, value, workers, warns):
    monkeypatch.setenv("FLASHINFER_CUDNN_PLAN_BUILD_WORKERS", value)
    monkeypatch.setattr(pool_mod, "_pool", None)
    monkeypatch.setattr(pool_mod, "_pool_resolved", False)
    # The CPU cap has its own test; keep it out of parsing on small runners.
    monkeypatch.setattr(pool_mod.os, "cpu_count", lambda: 8)
    with caplog.at_level("WARNING"):
        pool = pool_mod.get_cudnn_plan_build_pool()
    try:
        assert (pool.num_workers if pool else None) == workers
        assert ("FLASHINFER_CUDNN_PLAN_BUILD_WORKERS" in caplog.text) == warns
    finally:
        if pool:
            pool.shutdown()


@pytest.mark.parametrize("mode", ["garbage", "badindex", "partial"])
def test_malformed_or_incomplete_results_fall_back(fake_cmd, mode):
    pool = _pool(fake_cmd, mode)
    try:
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is None
        assert pool.broken
    finally:
        pool.shutdown()


def test_disable_turns_the_pool_off(fake_cmd):
    pool = _pool(fake_cmd, "ok")
    try:
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is not None
        procs = pool._all_processes()
        pool.disable("deserialize failed")
        assert pool.broken
        assert all(p.poll() is not None for p in procs)
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is None
    finally:
        pool.shutdown()


def test_idle_workers_exit_and_respawn(fake_cmd):
    pool = pool_mod.CudnnPlanBuildPool(
        2, timeout_s=10.0, worker_cmd=fake_cmd("ok"), idle_s=0.3
    )
    try:
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is not None
        first = pool._all_processes()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(p.poll() is None for p in first):
            time.sleep(0.05)
        assert all(p.poll() is not None for p in first)
        assert not pool.broken
        assert pool.build(SPEC, device_index=0, heur_modes=[0]) is not None
        assert {p.pid for p in pool._all_processes()}.isdisjoint(p.pid for p in first)
    finally:
        pool.shutdown()


def test_worker_count_is_capped_at_cpu_count(monkeypatch, caplog):
    monkeypatch.setenv("FLASHINFER_CUDNN_PLAN_BUILD_WORKERS", "100000")
    monkeypatch.setattr(pool_mod, "_pool", None)
    monkeypatch.setattr(pool_mod, "_pool_resolved", False)
    monkeypatch.setattr(pool_mod.os, "cpu_count", lambda: 6)
    with caplog.at_level("WARNING"):
        pool = pool_mod.get_cudnn_plan_build_pool()
    try:
        assert pool.num_workers == 6
        assert "FLASHINFER_CUDNN_PLAN_BUILD_WORKERS" in caplog.text
    finally:
        pool.shutdown()
