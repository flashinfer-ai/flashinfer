# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

from flashinfer.prims_ts.moe.tools.sweep.calibration import (
    CalibrationData,
    CompileScalingPoint,
    EnvironmentFingerprint,
    MAX_CALIBRATION_WORKERS,
    ShapeGPUCost,
    WorkerGPUFootprint,
    derive_noise_band,
    derive_safe_worker_cap,
    measure_worker_gpu_footprint,
)


def test_calibration_worker_limit_applies_to_probes_and_saved_recommendations():
    from flashinfer.prims_ts.moe.tools.sweep.__main__ import (
        _calibration_worker_counts,
        _fit_calibration_worker_counts,
    )

    assert _calibration_worker_counts((1, 8, 16, 32), cpu_count=64) == (
        1,
        8,
        16,
    )
    assert _fit_calibration_worker_counts((1, 2, 4, 8, 16), available_jobs=45) == (
        1,
        2,
        4,
        8,
        16,
    )
    assert _fit_calibration_worker_counts((1, 2, 4, 8, 16), available_jobs=15) == (
        1,
        2,
        4,
    )
    noise = derive_noise_band([10.0, 10.1, 9.9])
    calibration = CalibrationData(
        environment=_environment(),
        compile_scaling=[CompileScalingPoint(32, 100.0)],
        worker_count=32,
        noise=noise,
    )
    assert MAX_CALIBRATION_WORKERS == 16
    assert calibration.recommended_worker_count() == 16


def test_noise_band_trims_single_small_sample_outlier():
    band = derive_noise_band(
        [99.9, 100.0, 100.0, 100.1, 1_000.0],
        relative_floor=0.01,
    )

    assert band.median == 100.0
    assert band.absolute == 1.0
    assert band.lower == 99.0
    assert band.upper == 101.0


def _environment() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        gpu="Test GPU",
        driver="600.1",
        cuda="13.0",
        cute_dsl="4.8.0",
        cpu="Test CPU",
        flashinfer_sha="abc123",
    )


def test_worker_gpu_footprint_measurement_and_safe_cap():
    class FakeCuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def memory_allocated(device):
            assert device == 0
            return 3_000

        @staticmethod
        def memory_reserved(device):
            assert device == 0
            return 4_000

        @staticmethod
        def get_device_properties(device):
            assert device == 0
            return SimpleNamespace(total_memory=20_000, uuid=None)

    class FakeNvml:
        NVML_CLOCK_SM = 0

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            assert index == 0
            return object()

        @staticmethod
        def nvmlDeviceGetComputeRunningProcesses(handle):
            assert handle is not None
            return [SimpleNamespace(pid=7, usedGpuMemory=5_000)]

    footprint = measure_worker_gpu_footprint(
        torch_module=SimpleNamespace(cuda=FakeCuda()),
        nvml_module=FakeNvml(),
        pid=7,
    )

    assert footprint == WorkerGPUFootprint(3_000, 4_000, 5_000, 20_000)
    assert footprint.effective_bytes == 5_000
    assert (
        derive_safe_worker_cap(
            20_000,
            [footprint],
            reference_reserve_bytes=4_000,
            margin_bytes=1_000,
        )
        == 3
    )
    assert (
        derive_safe_worker_cap(
            20_000,
            [footprint, WorkerGPUFootprint(2_000, 6_000)],
            reference_reserve_bytes=4_000,
            margin_bytes=1_000,
            maximum_workers=2,
        )
        == 2
    )


def test_calibration_roundtrip_keeps_operational_worker_data():
    shape_cost = ShapeGPUCost("fc1:shape", 0.75)
    noise = derive_noise_band([10.0, 10.1, 9.9])
    footprint = WorkerGPUFootprint(3_000, 4_000, 5_000, 20_000)
    calibration = CalibrationData(
        environment=_environment(),
        compile_scaling=[
            CompileScalingPoint(
                workers=4,
                throughput_per_minute=40.0,
                worker_gpu_footprints=(footprint,),
            )
        ],
        worker_count=4,
        noise=noise,
        shape_costs=[shape_cost],
        total_gpu_memory_bytes=20_000,
        reference_reserve_bytes=4_000,
        gpu_memory_margin_bytes=1_000,
    )

    loaded = CalibrationData.from_dict(calibration.to_dict())
    assert loaded.to_dict() == calibration.to_dict()
    assert loaded.shape_costs[0].full_gpu_lock_seconds == 0.75
    assert loaded.safe_worker_cap() == 3
    assert loaded.recommended_worker_count() == 3


def test_calibration_stops_after_failed_worker_point_and_uses_evaluate_memory(
    monkeypatch,
):
    from flashinfer.prims_ts.moe.tools.sweep import __main__ as sweep_main

    calls = []
    compile_calls = 0
    compile_footprint = WorkerGPUFootprint(500, 1_000, None, 20_000).to_dict()
    evaluate_footprint = WorkerGPUFootprint(2_000, 4_000, None, 20_000).to_dict()

    class FakeEngine:
        def __init__(self, options):
            self.options = options
            self.shapes = [SimpleNamespace(selected_tile_ns=(128,))]
            self.candidates = {
                f"candidate-{index}": SimpleNamespace(tile_n=128) for index in range(4)
            }

        def dispatch(
            self,
            candidate,
            shapes,
            *,
            protocol,
            action="evaluate",
            cache_tag=None,
        ):
            nonlocal compile_calls
            del candidate, shapes, protocol
            calls.append((action, cache_tag))
            if action == "compile":
                compile_calls += 1
                if compile_calls >= 5:
                    return {"status": "worker_error", "error": "capacity"}
                return {
                    "status": "ok",
                    "peak_rss_bytes": 100,
                    "peak_gpu_memory_bytes": 1_000,
                    "gpu_memory_footprint": compile_footprint,
                }
            return {
                "status": "ok",
                "peak_gpu_memory_bytes": 4_000,
                "gpu_memory_footprint": evaluate_footprint,
                "lock_hold_ms": 10.0,
                "measurements": [
                    {
                        "status": "ok",
                        "shape_id": "shape",
                        "samples_us": [99.9, 100.0, 100.1],
                    }
                ],
            }

        def close(self):
            return None

    monkeypatch.setattr(sweep_main, "SweepEngine", FakeEngine)
    monkeypatch.setattr(
        sweep_main, "_calibration_worker_counts", lambda _counts: (1, 2)
    )
    monkeypatch.setattr(
        sweep_main, "_environment_fingerprint", lambda gpu: _environment()
    )
    args = SimpleNamespace(
        model=None,
        dtype="nvfp4",
        hidden=128,
        intermediate=128,
        num_experts=8,
        top_k=2,
        activation="swiglu",
        bias=False,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        sm_arch=107,
        tp=(1,),
        ep=(1,),
        tp_ep=None,
        tokens=(32,),
        gpu="0",
    )

    calibration = sweep_main._run_calibration(args)

    assert compile_calls == 6
    assert [point.failed for point in calibration.compile_scaling] == [False, True]
    assert calibration.worker_count == 1
    assert calibration.compile_scaling[0].worker_gpu_footprints == (
        WorkerGPUFootprint.from_dict(evaluate_footprint),
        WorkerGPUFootprint.from_dict(evaluate_footprint),
    )
    assert calibration.safe_worker_cap() == 2
    assert calls[-2:] == [
        ("evaluate", "timing-fc1"),
        ("evaluate", "timing-fc2"),
    ]


def test_calibration_reuses_reserved_warmup_candidates(monkeypatch):
    from flashinfer.prims_ts.moe.tools.sweep import __main__ as sweep_main

    calls = []
    footprint = WorkerGPUFootprint(500, 1_000, None, 20_000).to_dict()

    class FakeEngine:
        def __init__(self, options):
            self.options = options
            self.shapes = [SimpleNamespace(selected_tile_ns=(128,))]
            count = 23 if options.fc == "fc1" else 24
            self.candidates = {
                f"{options.fc}-candidate-{index}": SimpleNamespace(
                    tile_n=128,
                    name=f"{options.fc}-candidate-{index}",
                )
                for index in range(count)
            }

        def dispatch(
            self,
            candidate,
            shapes,
            *,
            protocol,
            action="evaluate",
            cache_tag=None,
        ):
            del shapes, protocol
            calls.append((self.options.fc, action, cache_tag, candidate.name))
            if action == "compile":
                return {
                    "status": "ok",
                    "peak_rss_bytes": 100,
                    "peak_gpu_memory_bytes": 1_000,
                    "gpu_memory_footprint": footprint,
                }
            return {
                "status": "ok",
                "peak_gpu_memory_bytes": 1_000,
                "gpu_memory_footprint": footprint,
                "lock_hold_ms": 10.0,
                "measurements": [
                    {
                        "status": "ok",
                        "shape_id": "shape",
                        "samples_us": [99.9, 100.0, 100.1],
                    }
                ],
            }

        def close(self):
            return None

    monkeypatch.setattr(sweep_main, "SweepEngine", FakeEngine)
    monkeypatch.setattr(
        sweep_main,
        "_calibration_worker_counts",
        lambda _counts: (1, 2, 4, 8, 16),
    )
    monkeypatch.setattr(
        sweep_main, "_environment_fingerprint", lambda gpu: _environment()
    )
    args = SimpleNamespace(
        model=None,
        dtype="nvfp4",
        hidden=128,
        intermediate=128,
        num_experts=8,
        top_k=2,
        activation="swiglu",
        bias=False,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        sm_arch=107,
        tp=(1,),
        ep=(1,),
        tp_ep=None,
        tokens=(32,),
        gpu="0",
    )

    calibration = sweep_main._run_calibration(args)

    compile_calls = [call for call in calls if call[1] == "compile"]
    assert len(compile_calls) == 48
    warmup_ids = [call[3] for call in compile_calls[:16]]
    measured_ids = [call[3] for call in compile_calls[16:]]
    assert len(set(warmup_ids)) == 2
    assert all(warmup_ids.count(candidate_id) == 8 for candidate_id in set(warmup_ids))
    assert len(set(measured_ids)) == 32
    assert set(warmup_ids).isdisjoint(measured_ids)
    assert [point.workers for point in calibration.compile_scaling] == [
        1,
        2,
        4,
        8,
        16,
    ]
