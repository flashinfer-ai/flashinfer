# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict CUPTI measurements for the standalone MXFP4 SiTU MoE benchmark.

Uses the activity callback/correlation approach in flashinfer.testing.utils.
Kernel durations come from CUPTI GPU activity; host submission and synchronized
end-to-end durations use cupti.get_timestamp(), on the same time base. There is
no CUDA-event estimation, wall-clock GPU timing, or fallback backend.

This helper owns the process's CUPTI activity session while measuring; run the
benchmark without another profiler or concurrent workload in the same process.
"""

from collections import Counter
import ctypes
import importlib.metadata
import math
import statistics
import sys


def _require_cupti():
    try:
        from cupti import cupti

        version = importlib.metadata.version("cupti-python")
        if int(version.split(".")[0]) < 13:
            raise RuntimeError("cupti-python >= 13 is required")
    except Exception as error:
        raise RuntimeError(
            "This benchmark requires usable cupti-python >= 13; no timing fallback is available"
        ) from error
    return cupti


def _dropped_records(cupti):
    query = getattr(cupti, "activity_get_num_dropped_records", None)
    if query is None:
        return None
    # The Python binding takes an intptr_t pointing to the size_t output.
    # CUPTI activity buffers are global, so context and stream ID are zero.
    dropped = ctypes.c_size_t()
    query(0, 0, ctypes.addressof(dropped))
    return dropped.value


def measure_moe_cupti(fn, *, mode, warmup, repeats):
    """Measure an already prepared callable, with exactly ``repeats`` samples.

    Both modes use a dedicated caller stream and the same cold-L2 policy. Each
    graph contains exactly one fn call. Preparation, capture, warmup, L2 flush,
    and pre-sample synchronization are excluded from all four reported metrics.
    End-to-end timing includes Python submission and device synchronization,
    with CUPTI tracing enabled; it is not an uninstrumented serving measurement.
    """
    if mode not in ("eager", "graph") or warmup < 1 or repeats < 1:
        raise ValueError("mode must be eager/graph and warmup/repeats must be positive")
    cupti = _require_cupti()
    import torch

    device = torch.cuda.current_device()
    l2_bytes = int(torch.cuda.get_device_properties(device).L2_cache_size)
    if l2_bytes <= 0:
        raise RuntimeError("The device did not report a usable L2 cache size")
    flush_buffer = torch.empty(2 * l2_bytes, device=device, dtype=torch.int8)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = None
    runner = fn
    with torch.cuda.stream(stream):
        if mode == "graph":
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                fn()
            runner = graph.replay
        for _ in range(warmup):
            flush_buffer.zero_()
            runner()
        torch.cuda.synchronize(device)

    kind = cupti.ActivityKind
    gpu_kinds = (kind.CONCURRENT_KERNEL, kind.MEMCPY, kind.MEMSET)
    api_kinds = (kind.RUNTIME, kind.DRIVER)
    apis = []
    activities = []
    callback_errors = []

    def buffer_requested():
        return 8 * 1024 * 1024, 0

    def buffer_completed(records):
        # Copy fields while the callback owns the underlying activity buffer.
        try:
            for record in records:
                if record.kind in api_kinds:
                    apis.append(
                        (int(record.start), int(record.end), int(record.correlation_id))
                    )
                elif record.kind in gpu_kinds:
                    is_kernel = record.kind == kind.CONCURRENT_KERNEL
                    name = str(record.name) if is_kernel else kind(record.kind).name
                    activities.append(
                        {
                            "kind": int(record.kind),
                            "name": name,
                            "start": int(record.start),
                            "end": int(record.end),
                            "correlation_id": int(record.correlation_id),
                            "context_id": int(record.context_id),
                            "stream_id": int(record.stream_id),
                            "bytes": 0 if is_kernel else int(record.bytes),
                        }
                    )
        except Exception as error:
            callback_errors.append(f"{type(error).__name__}: {error}")

    enabled = []
    timestamps = []
    dropped_before = None
    dropped_after = None
    registered = False
    try:
        cupti.activity_register_callbacks(buffer_requested, buffer_completed)
        registered = True
        for activity_kind in (*api_kinds, *gpu_kinds):
            cupti.activity_enable(activity_kind)
            enabled.append(activity_kind)
        # Reset the global counter before this measurement session.
        dropped_before = _dropped_records(cupti)
        with torch.cuda.stream(stream):
            for _ in range(repeats):
                flush_buffer.zero_()
                torch.cuda.synchronize(device)
                start = int(cupti.get_timestamp())
                runner()
                submitted = int(cupti.get_timestamp())
                torch.cuda.synchronize(device)
                synchronized = int(cupti.get_timestamp())
                timestamps.append((start, submitted, synchronized))
        cupti.activity_flush_all(0)
        dropped_after = _dropped_records(cupti)
    finally:
        # Attempt every cleanup operation. Preserve an original failure in the
        # exception chain if cleanup itself also fails.
        original_error = sys.exc_info()[1]
        cleanup_errors = []
        cleanup_operations = []
        if registered:
            cleanup_operations.extend(
                (
                    lambda: cupti.activity_flush_all(0),
                    *(lambda k=k: cupti.activity_disable(k) for k in reversed(enabled)),
                )
            )
        cleanup_operations.append(cupti.finalize)
        for operation in cleanup_operations:
            try:
                operation()
            except Exception as error:
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        if cleanup_errors:
            message = "CUPTI cleanup failed: " + "; ".join(cleanup_errors)
            raise RuntimeError(message) from original_error

    if callback_errors:
        raise RuntimeError(
            "CUPTI activity callback failed: " + "; ".join(callback_errors)
        )
    if dropped_after:
        raise RuntimeError(f"CUPTI dropped {dropped_after} activity records")
    if len(timestamps) != repeats:
        raise RuntimeError("CUPTI did not collect every requested iteration")

    samples = {
        name: []
        for name in (
            "gpu_span_ms",
            "kernel_sum_ms",
            "host_enqueue_ms",
            "synchronized_e2e_ms",
        )
    }
    expected_signature = None
    for iteration, (start, submitted, synchronized) in enumerate(timestamps):
        if not 0 < start < submitted < synchronized:
            raise RuntimeError(
                f"Invalid CUPTI host timestamps in iteration {iteration}"
            )
        correlation_ids = {
            correlation
            for api_start, api_end, correlation in apis
            if start <= api_start <= api_end <= submitted
        }
        selected = [a for a in activities if a["correlation_id"] in correlation_ids]
        kernels = [a for a in selected if a["kind"] == kind.CONCURRENT_KERNEL]
        if not correlation_ids or not kernels:
            raise RuntimeError(
                f"Missing correlated kernel records in iteration {iteration}"
            )
        for activity in activities:
            if start <= activity["start"] < synchronized and (
                activity["correlation_id"] not in correlation_ids
            ):
                raise RuntimeError(
                    f"Uncorrelated GPU activity in iteration {iteration}: {activity['name']}"
                )
        if any(not start <= a["start"] < a["end"] <= synchronized for a in selected):
            raise RuntimeError(
                f"Invalid or incomplete GPU timestamps in iteration {iteration}"
            )
        signature = Counter(
            (a["kind"], a["name"], a["context_id"], a["stream_id"], a["bytes"])
            for a in selected
        )
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise RuntimeError(
                f"Inconsistent kernel/memory activity multiplicities in iteration {iteration}"
            )
        samples["gpu_span_ms"].append(
            (max(a["end"] for a in selected) - min(a["start"] for a in selected)) / 1e6
        )
        samples["kernel_sum_ms"].append(
            sum(a["end"] - a["start"] for a in kernels) / 1e6
        )
        samples["host_enqueue_ms"].append((submitted - start) / 1e6)
        samples["synchronized_e2e_ms"].append((synchronized - start) / 1e6)
    if any(
        not math.isfinite(value) or value <= 0
        for values in samples.values()
        for value in values
    ):
        raise RuntimeError("CUPTI returned invalid measurement samples")
    inventory = [
        {
            "kind": kind(activity_kind).name,
            "name": name,
            "context_id": context_id,
            "stream_id": stream_id,
            "bytes": size,
            "count_per_iteration": count,
        }
        for (activity_kind, name, context_id, stream_id, size), count in sorted(
            expected_signature.items()
        )
    ]
    return {
        "backend": "cupti",
        "mode": mode,
        "warmup_iters": warmup,
        "repeat_iters": repeats,
        "cold_l2_flush_bytes": flush_buffer.numel(),
        "samples_ms": samples,
        **{name: statistics.median(values) for name, values in samples.items()},
        "activity_inventory": inventory,
        "kernel_launches_per_iteration": sum(
            item["count_per_iteration"]
            for item in inventory
            if item["kind"] == kind.CONCURRENT_KERNEL.name
        ),
        "dropped_record_check": "available"
        if dropped_after is not None
        else "unavailable",
        "dropped_records_before_session": dropped_before,
        "dropped_records": dropped_after,
    }
