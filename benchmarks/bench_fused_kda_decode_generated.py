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

"""Strict paired benchmark for generated fused KDA decode kernels on B200."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import torch

from flashinfer.jit.fused_kda_decode_generated import (
    _MANIFEST_FILENAME,
    fused_kda_decode_generated_is_available,
    get_kda_csrc_dir,
)
from flashinfer.kda_decode import fused_kda_decode
from flashinfer.testing import utils as testing_utils
from flashinfer.utils import get_compute_capability


_impl = importlib.import_module("flashinfer.kda_kernels.fused_kda_decode")
_HEAD_DIM = 128
_OFFICIAL_SHAPES = (
    (96, 1),
    (96, 4),
    (96, 8),
    (96, 32),
    (96, 128),
    (48, 1),
    (48, 4),
    (48, 32),
    (48, 128),
    (32, 1),
    (32, 4),
    (32, 32),
    (32, 128),
    (24, 1),
    (24, 4),
    (24, 32),
    (24, 64),
    (12, 1),
    (12, 4),
    (12, 32),
    (12, 256),
)
_ABBA_ORDER = ("baseline", "candidate", "candidate", "baseline")
_ORIGINAL_SHAPES = tuple(shape for shape in _OFFICIAL_SHAPES if shape[0] != 32)


def _page_strides(num_heads):
    hidden_size = num_heads * _HEAD_DIM
    conv_slot_bytes = 3 * hidden_size * 3 * torch.bfloat16.itemsize
    state_slot_bytes = num_heads * _HEAD_DIM * _HEAD_DIM * torch.float32.itemsize
    page_bytes = conv_slot_bytes + state_slot_bytes
    return page_bytes // torch.bfloat16.itemsize, page_bytes // torch.float32.itemsize


def _make_inputs(num_heads, num_rows, seed=42):
    device = torch.device("cuda")
    hidden_size = num_heads * _HEAD_DIM
    num_slots = num_rows + 1
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, dtype=torch.float32):
        return torch.randn(
            shape, device=device, dtype=torch.float32, generator=generator
        ).to(dtype)

    x_storage = randn((num_rows, 3 * hidden_size + 17), torch.bfloat16)
    conv_slot_stride, state_slot_stride = _page_strides(num_heads)
    conv_state = torch.empty_strided(
        (num_slots, 3 * hidden_size, 3),
        (conv_slot_stride, 1, 3 * hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    conv_state.copy_(0.1 * randn((num_slots, 3 * hidden_size, 3), torch.bfloat16))
    state = torch.empty_strided(
        (num_slots, num_heads, _HEAD_DIM, _HEAD_DIM),
        (state_slot_stride, _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1),
        dtype=torch.float32,
        device=device,
    )
    state.copy_(
        0.01 * randn((num_slots, num_heads, _HEAD_DIM, _HEAD_DIM), torch.float32)
    )
    beta_storage = randn((1, num_rows, num_heads + 1), torch.bfloat16)
    output_gate_storage = randn((num_rows, hidden_size + 7), torch.bfloat16)
    return {
        "x": x_storage[:, : 3 * hidden_size],
        "weight": 0.1 * randn((3, 4, hidden_size)),
        "conv_state": conv_state,
        "raw_gate": randn((1, num_rows, num_heads, _HEAD_DIM), torch.bfloat16),
        "raw_beta": beta_storage[:, :, :num_heads],
        "A_log": 0.5 * randn((num_heads,)),
        "dt_bias": 0.1 * randn((hidden_size,)),
        "state_indices": torch.arange(
            num_rows, 0, -1, dtype=torch.int32, device=device
        ),
        "state": state,
        "output_gate": output_gate_storage.as_strided(
            (num_rows, num_heads, _HEAD_DIM),
            (hidden_size + 7, _HEAD_DIM, 1),
        ),
        "norm_weight": randn((_HEAD_DIM,)),
        "lower_bound": -5.0,
        "norm_eps": 1e-5,
        "output": torch.empty(
            (1, num_rows, num_heads, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
    }


def _require_b200_and_cupti():
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    device = torch.device("cuda")
    if get_compute_capability(device) != (10, 0):
        raise RuntimeError("this benchmark requires an NVIDIA B200 (SM100a)")
    if "B200" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError("this benchmark requires an NVIDIA B200")
    try:
        importlib.import_module("cupti")
        cupti_version = importlib.metadata.version("cupti-python")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeError("cupti-python >= 13 is required") from error
    if int(cupti_version.split(".", maxsplit=1)[0]) < 13:
        raise RuntimeError(f"cupti-python >= 13 is required, found {cupti_version}")
    if not fused_kda_decode_generated_is_available():
        raise RuntimeError("the generated fused KDA manifest is not complete")
    return cupti_version


def _forbid_timing_fallback(*args, **kwargs):
    raise RuntimeError("CUPTI timing fallback is forbidden for this benchmark")


def _query_single_visible_gpu_identity():
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 2 or not fields[0] or not fields[1]:
        raise RuntimeError("nvidia-smi returned an invalid GPU identity")
    return fields[0], fields[1]


def _run_worker(args):
    _require_b200_and_cupti()
    inputs = _make_inputs(args.worker_heads, args.worker_rows)
    variant_name = None

    if args.worker_backend == "baseline":
        route_guard = mock.patch.object(
            _impl, "_select_generated_variant", return_value=None
        )
        fallback_guard = mock.patch.object(
            _impl, "_get_compiled_kernel", wraps=_impl._get_compiled_kernel
        )
    else:
        output_gate = inputs["output_gate"]
        variant = _impl._select_generated_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state_indices=inputs["state_indices"],
            state=inputs["state"],
            output_gate=output_gate,
            lower_bound=inputs["lower_bound"],
            norm_eps=inputs["norm_eps"],
        )
        if variant is None:
            raise RuntimeError("candidate inputs did not select a generated variant")
        variant_name = variant.name
        route_guard = mock.patch.object(
            _impl,
            "_get_compiled_kernel",
            side_effect=RuntimeError("candidate fell back to the CuTe DSL kernel"),
        )
        fallback_guard = mock.patch.object(
            _impl, "_select_generated_variant", wraps=_impl._select_generated_variant
        )

    with route_guard, fallback_guard:
        fused_kda_decode(**inputs)
        torch.cuda.synchronize()
        with (
            mock.patch.object(
                testing_utils,
                "bench_gpu_time_with_cuda_event",
                side_effect=_forbid_timing_fallback,
            ),
            mock.patch.object(
                testing_utils,
                "bench_gpu_time_with_cudagraph",
                side_effect=_forbid_timing_fallback,
            ),
        ):
            samples = testing_utils.bench_gpu_time(
                fused_kda_decode,
                dry_run_iters=args.dry_run_iters,
                repeat_iters=args.repeat_iters,
                enable_cupti=True,
                use_cuda_graph=True,
                input_kwargs=inputs,
                cold_l2_cache=True,
            )

    payload = {
        "backend": args.worker_backend,
        "num_heads": args.worker_heads,
        "num_rows": args.worker_rows,
        "variant_name": variant_name,
        "samples_ms": [float(value) for value in samples],
    }
    Path(args.worker_json).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _worker_command(args, backend, num_heads, num_rows, output_path):
    return (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-backend",
        backend,
        "--worker-heads",
        str(num_heads),
        "--worker-rows",
        str(num_rows),
        "--worker-json",
        str(output_path),
        "--dry-run-iters",
        str(args.dry_run_iters),
        "--repeat-iters",
        str(args.repeat_iters),
    )


def _write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _require_close(observed, expected, description):
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or not math.isfinite(observed)
        or not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise RuntimeError(f"{description} does not match its recomputed value")


def _validate_samples(samples, repeat_iters, description):
    if not isinstance(samples, list) or len(samples) != repeat_iters:
        raise RuntimeError(f"{description} must contain exactly {repeat_iters} samples")
    for sample in samples:
        if (
            isinstance(sample, bool)
            or not isinstance(sample, (int, float))
            or not math.isfinite(sample)
            or sample <= 0
        ):
            raise RuntimeError(f"{description} contains an invalid sample")
    return samples


def _validate_row(row, index, repeat_iters):
    if not isinstance(row, dict):
        raise RuntimeError("checkpoint row must be an object")
    expected_row_fields = {
        "shape",
        "num_heads",
        "num_rows",
        "baseline_ms",
        "candidate_ms",
        "speedup",
        "measurements",
    }
    if set(row) != expected_row_fields:
        raise RuntimeError("checkpoint row schema is invalid")
    num_heads, num_rows = _OFFICIAL_SHAPES[index]
    shape = f"h{num_heads}_rows{num_rows}"
    if (
        row.get("shape") != shape
        or type(row.get("num_heads")) is not int
        or row.get("num_heads") != num_heads
        or type(row.get("num_rows")) is not int
        or row.get("num_rows") != num_rows
    ):
        raise RuntimeError("checkpoint rows are not an official-shape prefix")
    measurements = row.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(_ABBA_ORDER):
        raise RuntimeError(f"{shape} does not contain the four ABBA cells")
    backend_samples = {"baseline": [], "candidate": []}
    candidate_variants = set()
    for order_index, expected_backend in enumerate(_ABBA_ORDER):
        measurement = measurements[order_index]
        if not isinstance(measurement, dict):
            raise RuntimeError(f"{shape} measurement must be an object")
        if set(measurement) != {
            "order_index",
            "backend",
            "variant_name",
            "median_ms",
            "samples_ms",
        }:
            raise RuntimeError(f"{shape} measurement schema is invalid")
        if (
            type(measurement.get("order_index")) is not int
            or measurement.get("order_index") != order_index
            or measurement.get("backend") != expected_backend
        ):
            raise RuntimeError(f"{shape} measurement order is not exact ABBA")
        variant_name = measurement.get("variant_name")
        if expected_backend == "baseline":
            if variant_name is not None:
                raise RuntimeError(f"{shape} baseline unexpectedly used a variant")
        elif not isinstance(variant_name, str) or not variant_name:
            raise RuntimeError(f"{shape} candidate variant identity is missing")
        else:
            candidate_variants.add(variant_name)
        samples = _validate_samples(
            measurement.get("samples_ms"),
            repeat_iters,
            f"{shape} {expected_backend} cell {order_index}",
        )
        cell_median = statistics.median(samples)
        _require_close(
            measurement.get("median_ms"),
            cell_median,
            f"{shape} cell {order_index} median",
        )
        backend_samples[expected_backend].extend(samples)
    if len(candidate_variants) != 1:
        raise RuntimeError(f"{shape} candidate cells selected different variants")
    baseline_ms = statistics.median(backend_samples["baseline"])
    candidate_ms = statistics.median(backend_samples["candidate"])
    speedup = baseline_ms / candidate_ms
    _require_close(row.get("baseline_ms"), baseline_ms, f"{shape} baseline")
    _require_close(row.get("candidate_ms"), candidate_ms, f"{shape} candidate")
    _require_close(row.get("speedup"), speedup, f"{shape} speedup")


def _geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _summarize(rows):
    speedups = [row["speedup"] for row in rows]
    original_shapes = set(_ORIGINAL_SHAPES)
    original_speedups = [
        row["speedup"]
        for row in rows
        if (row["num_heads"], row["num_rows"]) in original_shapes
    ]
    if len(original_speedups) != len(_ORIGINAL_SHAPES):
        raise RuntimeError("rows do not cover the original 17 shapes")
    return {
        "shape_count": len(rows),
        "baseline_geomean_ms": _geometric_mean([row["baseline_ms"] for row in rows]),
        "candidate_geomean_ms": _geometric_mean([row["candidate_ms"] for row in rows]),
        "official21_geomean_speedup": _geometric_mean(speedups),
        "original17_geomean_speedup": _geometric_mean(original_speedups),
        "minimum_speedup": min(speedups),
        "every_shape_faster": all(speedup > 1.0 for speedup in speedups),
    }


def _validate_summary(observed, expected):
    if not isinstance(observed, dict) or set(observed) != set(expected):
        raise RuntimeError("complete checkpoint summary schema is invalid")
    for field, value in expected.items():
        if isinstance(value, bool):
            if type(observed.get(field)) is not bool or observed[field] != value:
                raise RuntimeError(f"checkpoint summary {field} is invalid")
        elif isinstance(value, int):
            if type(observed.get(field)) is not int or observed[field] != value:
                raise RuntimeError(f"checkpoint summary {field} is invalid")
        else:
            _require_close(observed.get(field), value, f"checkpoint summary {field}")


def _load_checkpoint(path, *, identity, measurement_config):
    if not path.is_file():
        return [], False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint top level must be an object")
    expected_fields = {"status", "identity", "measurement", "rows"}
    if payload.get("status") == "complete":
        expected_fields.add("summary")
    if set(payload) != expected_fields:
        raise RuntimeError("checkpoint top-level schema is invalid")
    if payload.get("identity") != identity:
        raise RuntimeError(
            "checkpoint identity does not match this commit, GPU, or job"
        )
    if payload.get("measurement") != measurement_config:
        raise RuntimeError("checkpoint measurement settings do not match this run")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) > len(_OFFICIAL_SHAPES):
        raise RuntimeError("checkpoint rows are invalid")
    for index, row in enumerate(rows):
        _validate_row(row, index, measurement_config["repeat_iters_per_cell"])
    status = payload.get("status")
    if status not in ("in_progress", "complete"):
        raise RuntimeError(f"unsupported checkpoint status {status!r}")
    if status == "complete" and len(rows) != len(_OFFICIAL_SHAPES):
        raise RuntimeError("complete checkpoint does not contain every official shape")
    if status == "complete":
        _validate_summary(payload.get("summary"), _summarize(rows))
    elif "summary" in payload:
        raise RuntimeError("in-progress checkpoint unexpectedly contains a summary")
    return rows, status == "complete"


def _benchmark_payload(*, status, identity, measurement_config, rows, summary=None):
    payload = {
        "status": status,
        "identity": identity,
        "measurement": measurement_config,
        "rows": rows,
    }
    if summary is not None:
        payload["summary"] = summary
    return payload


def _run_paired_benchmark(args):
    cupti_version = _require_b200_and_cupti()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output_json).resolve()
    try:
        output_path.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("--output-json must be outside the source repository")
    manifest_path = (get_kda_csrc_dir() / _MANIFEST_FILENAME).resolve()
    for description, source_path in (
        ("benchmark", Path(__file__).resolve()),
        ("fused KDA implementation", Path(_impl.__file__).resolve()),
        ("generated manifest", manifest_path),
    ):
        try:
            source_path.relative_to(repo_root)
        except ValueError as error:
            raise RuntimeError(
                f"{description} is not loaded from the benchmark repository"
            ) from error
    git_status = subprocess.run(
        ("git", "-C", str(repo_root), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if git_status:
        raise RuntimeError("benchmark repository must be completely clean")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu_uuid, pci_bus_id = _query_single_visible_gpu_identity()
    source_commit = subprocess.run(
        ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(source_commit) != 40:
        raise RuntimeError("benchmark source commit is not a full Git object ID")
    gpu = {
        "name": properties.name,
        "compute_capability": list(get_compute_capability(torch.device("cuda"))),
        "sm_count": properties.multi_processor_count,
        "uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
    }
    identity = {
        "source_commit": source_commit,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "gpu": gpu,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    measurement_config = {
        "timer": "bench_gpu_time",
        "backend": "cupti",
        "cupti_python_version": cupti_version,
        "cuda_graph": True,
        "cold_l2": True,
        "interleaving": "abba",
        "order": list(_ABBA_ORDER),
        "dry_run_iters": args.dry_run_iters,
        "repeat_iters_per_cell": args.repeat_iters,
    }
    rows, complete = _load_checkpoint(
        output_path, identity=identity, measurement_config=measurement_config
    )
    if complete:
        print(f"checkpoint is already complete: {output_path}", flush=True)
        return

    with tempfile.TemporaryDirectory(
        prefix="flashinfer-fused-kda-paired-"
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        for shape_index, (num_heads, num_rows) in enumerate(
            _OFFICIAL_SHAPES[len(rows) :], start=len(rows)
        ):
            backend_samples = {"baseline": [], "candidate": []}
            measurements = []
            for order_index, backend in enumerate(_ABBA_ORDER):
                worker_json = temporary_path / (
                    f"shape-{shape_index:02d}-{order_index}-{backend}.json"
                )
                subprocess.run(
                    _worker_command(args, backend, num_heads, num_rows, worker_json),
                    check=True,
                )
                worker_measurement = json.loads(worker_json.read_text(encoding="utf-8"))
                if (
                    worker_measurement.get("backend") != backend
                    or worker_measurement.get("num_heads") != num_heads
                    or worker_measurement.get("num_rows") != num_rows
                ):
                    raise RuntimeError("worker result does not match its request")
                variant_name = worker_measurement.get("variant_name")
                if backend == "baseline" and variant_name is not None:
                    raise RuntimeError("baseline worker selected a generated variant")
                if backend == "candidate" and (
                    not isinstance(variant_name, str) or not variant_name
                ):
                    raise RuntimeError("candidate worker omitted its variant identity")
                samples = _validate_samples(
                    worker_measurement.get("samples_ms"),
                    args.repeat_iters,
                    f"h{num_heads}_rows{num_rows} {backend} worker",
                )
                backend_samples[backend].extend(samples)
                measurements.append(
                    {
                        "order_index": order_index,
                        "backend": backend,
                        "variant_name": variant_name,
                        "median_ms": statistics.median(samples),
                        "samples_ms": samples,
                    }
                )

            baseline_ms = statistics.median(backend_samples["baseline"])
            candidate_ms = statistics.median(backend_samples["candidate"])
            rows.append(
                {
                    "shape": f"h{num_heads}_rows{num_rows}",
                    "num_heads": num_heads,
                    "num_rows": num_rows,
                    "baseline_ms": baseline_ms,
                    "candidate_ms": candidate_ms,
                    "speedup": baseline_ms / candidate_ms,
                    "measurements": measurements,
                }
            )
            _validate_row(rows[-1], shape_index, args.repeat_iters)
            _write_json_atomic(
                output_path,
                _benchmark_payload(
                    status="in_progress",
                    identity=identity,
                    measurement_config=measurement_config,
                    rows=rows,
                ),
            )
            print(
                f"h{num_heads} rows{num_rows}: baseline={baseline_ms:.6f} ms "
                f"candidate={candidate_ms:.6f} ms "
                f"speedup={baseline_ms / candidate_ms:.6f}x",
                flush=True,
            )

    summary = _summarize(rows)
    _write_json_atomic(
        output_path,
        _benchmark_payload(
            status="complete",
            identity=identity,
            measurement_config=measurement_config,
            rows=rows,
            summary=summary,
        ),
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json")
    parser.add_argument("--shapes", choices=("official21",), default="official21")
    parser.add_argument(
        "--baseline", choices=("public-fallback",), default="public-fallback"
    )
    parser.add_argument("--candidate", choices=("generated",), default="generated")
    parser.add_argument("--timing", choices=("cupti",), default="cupti")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--interleave", choices=("abba",), default="abba")
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--worker-backend", choices=("baseline", "candidate"))
    parser.add_argument("--worker-heads", type=int)
    parser.add_argument("--worker-rows", type=int)
    parser.add_argument("--worker-json")
    args = parser.parse_args()
    if args.dry_run_iters < 1 or args.repeat_iters < 1:
        parser.error("iteration counts must be positive")
    if args.worker_backend is not None:
        if None in (args.worker_heads, args.worker_rows, args.worker_json):
            parser.error("worker mode requires heads, rows, and output JSON")
    else:
        if args.output_json is None:
            parser.error("--output-json is required")
        if not args.cuda_graph or not args.cold_l2:
            parser.error("--cuda-graph and --cold-l2 are required")
    return args


def main():
    args = _parse_args()
    if args.worker_backend is not None:
        _run_worker(args)
    else:
        _run_paired_benchmark(args)


if __name__ == "__main__":
    main()
