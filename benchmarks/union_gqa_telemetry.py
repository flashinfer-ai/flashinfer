# Copyright (c) 2026 by NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");

"""External GPU sampling and wall-time phase markers for Union-GQA benchmarks."""

import csv
import datetime
import json
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


class GPUTelemetry:
    """Sample in a separate process so polling does not hold Python's GIL.

    Samples describe a benchmark phase, not individual sub-millisecond kernels.
    NVML's power/clock update cadence can be slower than the requested interval.
    """

    fields = [
        "timestamp",
        "uuid",
        "clocks.current.sm",
        "clocks.current.memory",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "clocks_event_reasons.active",
        "clocks_event_reasons.sw_power_cap",
        "clocks_event_reasons.sw_thermal_slowdown",
        "clocks_event_reasons.hw_thermal_slowdown",
    ]

    def __init__(self, prefix, gpu_uuid, interval_ms):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        if not gpu_uuid.startswith("GPU-"):
            gpu_uuid = "GPU-" + gpu_uuid
        command = [
            "nvidia-smi",
            f"--id={gpu_uuid}",
            f"--query-gpu={','.join(self.fields)}",
            "--format=csv,noheader,nounits",
        ]
        # Check support before starting any measurements; retain stderr on failure.
        probe = subprocess.run(command, capture_output=True, text=True)
        if probe.returncode:
            raise RuntimeError(f"GPU sampler failed: {probe.stdout}{probe.stderr}")
        self.prefix.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "benchmark_argv": sys.argv,
                    "sampler_command": command + [f"--loop-ms={interval_ms}"],
                    "sample_fields": self.fields,
                    "gpu_uuid": gpu_uuid,
                    "timezone": time.tzname,
                    "started_epoch_s": time.time(),
                },
                indent=2,
            )
            + "\n"
        )
        self.raw = self.prefix.with_suffix(".samples.csv").open("w")
        self.errors = self.prefix.with_suffix(".stderr").open("w")
        self.markers = self.prefix.with_suffix(".phases.jsonl").open("w")
        self.process = subprocess.Popen(
            command + [f"--loop-ms={interval_ms}"],
            stdout=self.raw,
            stderr=self.errors,
        )
        self.phases = []

    @contextmanager
    def phase(self, **metadata):
        if self.process.poll() is not None:
            raise RuntimeError("GPU sampler exited; inspect telemetry .stderr")
        record = dict(metadata, start_epoch_s=time.time())
        try:
            yield record
        finally:
            record["end_epoch_s"] = time.time()
            self.phases.append(record)
            self.markers.write(json.dumps(record) + "\n")
            self.markers.flush()

    def close(self):
        failed = self.process.poll() is not None
        if not failed:
            self.process.terminate()
        self.process.wait(timeout=10)
        self.raw.close()
        self.errors.close()
        self.markers.close()
        if failed:
            raise RuntimeError("GPU sampler exited; inspect telemetry .stderr")
        samples = []
        with self.prefix.with_suffix(".samples.csv").open() as source:
            for row in csv.reader(source):
                if len(row) != len(self.fields):
                    continue
                row = [value.strip() for value in row]
                timestamp = datetime.datetime.strptime(
                    row[0], "%Y/%m/%d %H:%M:%S.%f"
                ).timestamp()
                samples.append((timestamp, row))
        summary = []
        numeric = {
            "sm_mhz": 2,
            "memory_mhz": 3,
            "power_w": 4,
            "power_limit_w": 5,
            "temperature_c": 6,
        }
        for phase in self.phases:
            rows = [
                row
                for timestamp, row in samples
                if phase["start_epoch_s"] <= timestamp < phase["end_epoch_s"]
            ]
            result = dict(phase, samples=len(rows))
            for name, column in numeric.items():
                values = [
                    float(row[column])
                    for row in rows
                    if row[column] not in ("N/A", "[N/A]", "[Not Supported]")
                ]
                if values:
                    result[name + "_mean"] = statistics.mean(values)
                    result[name + "_min"] = min(values)
                    result[name + "_max"] = max(values)
            for name, column in (
                ("sw_power_cap", 8),
                ("sw_thermal_slowdown", 9),
                ("hw_thermal_slowdown", 10),
            ):
                supported = [
                    row[column]
                    for row in rows
                    if row[column] in ("Active", "Not Active")
                ]
                result[name + "_fraction"] = (
                    sum(value == "Active" for value in supported) / len(supported)
                    if supported
                    else None
                )
            summary.append(result)
        self.prefix.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
