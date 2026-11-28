# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Measurement policies for collective MegaMoE trials."""

from __future__ import annotations

import warnings


class CaptureUnsupported(RuntimeError):
    """A prepared candidate could not be captured on every EP rank."""


def _is_capture_rejection(error: BaseException | None) -> bool:
    while error is not None:
        if isinstance(error, CaptureUnsupported):
            return True
        code = getattr(error, "error_code", None)
        if code is None and isinstance(error, RuntimeError):
            message = str(error).partition("\n")[0].removeprefix("CUDA error: ")
            # Some PyTorch versions expose only the CUDA runtime error string.
            code = {
                "operation not permitted when stream is capturing": 900,
                "operation not permitted when stream is capturing (900)": 900,
                "operation failed due to a previous error during capture": 901,
                "operation failed due to a previous error during capture (901)": 901,
            }.get(message)
        if code == 900:
            return True
        if code != 901:
            return False
        # capture_end can report invalidation instead of the original exception.
        error = error.__cause__ or error.__context__
    return False


def measurement_settings(context, *, native_graph_calls=None, native_warmups=0):
    policy = context.measurement_policy
    native = policy is None or (
        policy.execution_mode == "auto" and policy._timer == "auto"
    )
    cold_l2 = bool(policy.cold_l2) if policy is not None else False
    graph = (policy is not None and policy.use_cuda_graph is True) or (
        native and native_graph_calls is not None
    )
    return {
        "cuda_graph_profile_replays": (context.cuda_graph_profile_replays or 1)
        if graph
        else None,
        **(
            {
                # One flush per operator call, including native graph timing.
                "calls_per_sample": 1 if cold_l2 else native_graph_calls,
                "warmup_iters": native_warmups,
            }
            if native and native_graph_calls is not None
            else {}
        ),
        "cold_l2": cold_l2,
        # Auto retains the backend's native event timing.
        "timer": "native_events"
        if policy is None
        or (policy.execution_mode == "auto" and policy._timer == "auto")
        else policy.timer,
    }


def measure_samples(trial, weights, repeats, settings, process_group):
    """Return matched rank-local samples in microseconds for ``trial.launch``."""
    import torch
    import torch.distributed as dist

    from flashinfer.autotuner import AutoTuner
    from flashinfer.autotuner.autotuner import (
        _CuptiInfraError,
        _cupti_measure_spans,
        _load_cupti,
        _profile_measurement_scope,
    )
    from flashinfer.tllm_utils import delay_kernel

    tuner = AutoTuner.get()
    world = trial.config.world_size

    def gather(value):
        if world == 1:
            return [value]
        values = [None] * world
        dist.all_gather_object(values, value, group=process_group)
        return values

    def barrier():
        if world > 1:
            dist.barrier(group=process_group)

    graph = None
    flush = None
    timer = settings["timer"]
    graph_replays = settings["cuda_graph_profile_replays"]
    calls = settings.get("calls_per_sample", 1)

    def launch():
        trial.launch(*weights)

    with _profile_measurement_scope():
        if graph_replays is not None:
            graph = torch.cuda.CUDAGraph()
            barrier()
            error = status = None
            try:
                # Keep the caller's stream if capture_end raises.
                with (
                    torch.cuda.stream(torch.cuda.current_stream()),
                    torch.cuda.graph(graph),
                ):
                    for _ in range(calls):
                        launch()
            except Exception as exc:
                error = exc
                status = (_is_capture_rejection(exc), f"{type(exc).__name__}: {exc}")
            failures = [failure for failure in gather(status) if failure is not None]
            if any(not recoverable for recoverable, _ in failures):
                if status is not None and not status[0]:
                    assert error is not None
                    raise error
                raise RuntimeError(f"Candidate graph capture failed: {failures}")
            if failures:
                graph.reset()
                raise CaptureUnsupported(f"Candidate graph capture failed: {failures}")
            for _ in range(max(1, settings.get("warmup_iters", 0))):
                graph.replay()
            torch.cuda.synchronize()
        run = graph.replay if graph is not None else launch
        if settings["cold_l2"]:
            flush = torch.empty(
                2 * tuner._get_l2_cache_size_in_bytes(trial.device.index),
                dtype=torch.uint8,
                device=trial.device,
            )

        def prologue():
            barrier()
            if flush is not None:
                flush.zero_()

        def check_ready(error):
            errors = gather(error)
            if any(errors):
                raise _CuptiInfraError(
                    f"CUPTI initialization failed on EP ranks: {errors}"
                )

        try:
            if timer == "cupti":
                available = _load_cupti() is not None and not tuner._cupti_disabled
                if all(gather(available)):
                    error = None
                    try:
                        samples = _cupti_measure_spans(
                            run,
                            repeats * (graph_replays or 1),
                            prologue=prologue,
                            check_ready=check_ready,
                        )
                    except _CuptiInfraError as exc:
                        error = str(exc)
                    if not any(gather(error)):
                        return [sample * 1e3 / calls for sample in samples]
                tuner._cupti_disabled = True
                warnings.warn(
                    "MegaMoE CUPTI timing is unavailable on an EP rank; using "
                    "CUDA events on every rank. Retune without the conflicting "
                    "profiler for CUPTI measurements.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                timer = "events"

            samples = []
            # Match iterations across ranks. With cold L2, every replay gets its
            # own flush and event pair; flush kernels are outside the interval.
            cold_replays = (graph_replays or 1) if flush is not None else 1
            hot_replays = (graph_replays or 1) if flush is None else 1
            for _ in range(repeats * cold_replays):
                prologue()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                if timer == "events":
                    delay = (
                        tuner._CUDA_GRAPH_DELAY_MICRO_SECS
                        if graph is not None
                        else tuner.stream_delay_micro_secs
                    )
                    if delay > 0:
                        delay_kernel(delay)
                start.record()
                for _ in range(hot_replays):
                    run()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1e3 / (hot_replays * calls))
            return samples
        finally:
            if graph is not None:
                graph.reset()
