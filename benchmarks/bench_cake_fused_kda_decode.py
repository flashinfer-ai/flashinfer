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

"""Strict paired benchmark for the Cake fused KDA decode backend on B200."""

import argparse
import ast
import gc
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
import time
from pathlib import Path
from unittest import mock

import torch

from flashinfer.jit.cake_fused_kda_decode import (
    CAKE_FUSED_KDA_DECODE_ABIS,
    cake_fused_kda_decode_is_available,
    get_cake_fused_kda_decode_program_identity,
    get_cake_fused_kda_decode_variants,
    select_cake_fused_kda_decode_variant,
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
_FULL_DOMAIN_HEADS = (12, 24, 32, 48, 96)
_FULL_DOMAIN_TAIL_ROWS = (384, 512, 768, 1024, 1536, 2048, 4096)
_FULL_DOMAIN_SHAPES = tuple(
    (num_heads, num_rows)
    for num_heads in _FULL_DOMAIN_HEADS
    for num_rows in range(1, 257)
) + tuple(
    (num_heads, num_rows)
    for num_heads in _FULL_DOMAIN_HEADS
    for num_rows in _FULL_DOMAIN_TAIL_ROWS
)
_FULL_DOMAIN_SCHEMA = "cake-fused-kda-full-domain-benchmark-v2"
_FULL_DOMAIN_ROW_SCHEMA = "cake-fused-kda-full-domain-row-v2"
_FULL_DOMAIN_INHERITANCE_SCHEMA = "cake-fused-kda-full-domain-inheritance-v2"
_EQUIVALENCE_SCHEMA = "cake-fused-kda-decode-equivalence-v1"
_EQUIVALENCE_VERIFIER = "tools/verify_cake_fused_kda_decode_equivalence.py"
_LEGACY_FULL_DOMAIN_SCHEMA = "fused-kda-generated-full-domain-benchmark-v1"
_LEGACY_FULL_DOMAIN_ROW_SCHEMA = "fused-kda-generated-full-domain-row-v1"
_MEASURED_PREDECESSOR_COMMIT = "00a9d35a9d2ec1870a2068d9f970a7c15a9fa92a"
_EXACT_PR_BASELINE_COMMIT = "fad4af96fac0714feb197044a7226d382cb58a31"
_EXACT_PR_MERGE_COMMIT = "180f0d660aa05892fdaf77d2e4333dc1bb29d3ae"
_EXACT_PR_BASELINE_SOURCE_SHA256 = (
    "dd6cb13f54a823012fe57a12bfadf24c7b3177566a1832ae8eb7f5cfc7977f84"
)
_EXACT_PR_FALLBACK_SYMBOLS = (
    "_aligned_tensor",
    "_sigmoid",
    "_fused_kda_decode_kernel",
    "_fused_kda_decode_launch",
    "_make_compile_inputs",
    "_get_compiled_kernel",
    "_check_cuda_tensor",
)


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
    state_values = randn((num_slots, num_heads, _HEAD_DIM, _HEAD_DIM), torch.float32)
    state_values.mul_(0.01)
    state.copy_(state_values)
    del state_values
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
    if not cake_fused_kda_decode_is_available():
        raise RuntimeError("the Cake fused KDA source registry is not complete")
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


def _cake_program_record(repo_root):
    variants = get_cake_fused_kda_decode_variants()
    records = []
    for variant in variants:
        body_path = variant.body_path.resolve()
        try:
            relative_body = body_path.relative_to(repo_root)
        except ValueError as error:
            raise RuntimeError(
                f"Cake source {variant.name!r} is not loaded from the benchmark repository"
            ) from error
        records.append(
            {
                "name": variant.name,
                "target": variant.target,
                "body": str(relative_body),
                "source_sha256": variant.source_sha256,
                "kernel_symbol": variant.kernel_symbol,
                "abi_kind": variant.abi_kind,
                "abi": [
                    list(argument)
                    for argument in CAKE_FUSED_KDA_DECODE_ABIS[variant.abi_kind]
                ],
                "state_dtype": variant.state_dtype,
                "slot_offset_bits": variant.slot_offset_bits,
                "extra_cuda_cflags": list(variant.extra_cuda_cflags),
                "threads": variant.threads,
                "dynamic_smem_bytes": variant.dynamic_smem_bytes,
                "eligibility": [
                    {
                        "heads": list(rule.heads),
                        "minimum_rows": rule.minimum_rows,
                        "maximum_rows": rule.maximum_rows,
                        "state_indices_modes": list(rule.state_indices_modes),
                        "lower_bound_values": (
                            "any"
                            if rule.lower_bound_values is None
                            else list(rule.lower_bound_values)
                        ),
                        "norm_eps_values": (
                            "any"
                            if rule.norm_eps_values is None
                            else list(rule.norm_eps_values)
                        ),
                        "strides": dict(rule.strides),
                    }
                    for rule in variant.eligibility
                ],
            }
        )
    return {
        "identity_sha256": get_cake_fused_kda_decode_program_identity(),
        "variants": records,
    }


def _run_worker(args):
    _require_b200_and_cupti()
    inputs = _make_inputs(args.worker_heads, args.worker_rows)
    variant_name = None

    if args.worker_backend == "baseline":
        inputs["backend"] = "cute-dsl"
        route_guard = mock.patch.object(
            _impl, "_select_cake_variant", return_value=None
        )
        fallback_guard = mock.patch.object(
            _impl, "_get_compiled_kernel", wraps=_impl._get_compiled_kernel
        )
    else:
        inputs["backend"] = "cake"
        inputs["state_indices_mode"] = "positive_unique"
        output_gate = inputs["output_gate"]
        variant = _impl._select_cake_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state=inputs["state"],
            output_gate=output_gate,
            output=inputs["output"],
            state_indices_mode=inputs["state_indices_mode"],
            lower_bound=inputs["lower_bound"],
            norm_eps=inputs["norm_eps"],
        )
        if variant is None:
            raise RuntimeError("candidate inputs did not select a Cake variant")
        variant_name = variant.name
        route_guard = mock.patch.object(
            _impl,
            "_get_compiled_kernel",
            side_effect=RuntimeError("candidate fell back to the CuTe DSL kernel"),
        )
        fallback_guard = mock.patch.object(
            _impl, "_select_cake_variant", wraps=_impl._select_cake_variant
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
    program = _cake_program_record(repo_root)
    for description, source_path in (
        ("benchmark", Path(__file__).resolve()),
        ("fused KDA implementation", Path(_impl.__file__).resolve()),
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
        "program": program,
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
                    raise RuntimeError("baseline worker selected a Cake variant")
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


def _canonical_json_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_symbol_sha256(source_text, symbols):
    tree = ast.parse(source_text)
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = sorted(set(symbols) - set(nodes))
    if missing:
        raise RuntimeError(f"baseline source is missing symbols: {missing}")
    payload = "\n".join(
        ast.dump(nodes[name], include_attributes=False) for name in symbols
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _attest_exact_pr_fallback(repo_root, implementation_path):
    exact_source = subprocess.run(
        (
            "git",
            "-C",
            str(repo_root),
            "show",
            f"{_EXACT_PR_MERGE_COMMIT}:flashinfer/kda_kernels/fused_kda_decode.py",
        ),
        check=True,
        capture_output=True,
    ).stdout
    exact_source_sha256 = hashlib.sha256(exact_source).hexdigest()
    if exact_source_sha256 != _EXACT_PR_BASELINE_SOURCE_SHA256:
        raise RuntimeError("exact PR baseline source identity is unavailable")
    exact_symbol_sha256 = _source_symbol_sha256(
        exact_source.decode("utf-8"), _EXACT_PR_FALLBACK_SYMBOLS
    )
    current_symbol_sha256 = _source_symbol_sha256(
        implementation_path.read_text(encoding="utf-8"), _EXACT_PR_FALLBACK_SYMBOLS
    )
    if current_symbol_sha256 != exact_symbol_sha256:
        raise RuntimeError("public fallback kernel symbols drifted from the exact PR")
    return {
        "commit": _EXACT_PR_BASELINE_COMMIT,
        "merge_commit": _EXACT_PR_MERGE_COMMIT,
        "source_sha256": exact_source_sha256,
        "fallback_symbol_ast_sha256": exact_symbol_sha256,
        "route": "generated selector forced to return None",
    }


def _full_domain_measurement_config(args, cupti_version):
    return {
        "timer": "bench_gpu_time",
        "backend": "cupti",
        "cupti_python_version": cupti_version,
        "cuda_graph": True,
        "cold_l2": True,
        "interleaving": "abba",
        "order": list(_ABBA_ORDER),
        "graph_warmup_replays_per_cell": args.dry_run_iters,
        "graph_warmup_replays_per_implementation": 2 * args.dry_run_iters,
        "timed_replays_per_sample": args.repeat_iters,
        "paired_samples": 2,
        "timed_replays_per_implementation": 2 * args.repeat_iters,
        "cell_reducer": "median",
        "paired_speedup_reducer": "median",
        "aggregate_reducer": "geometric_mean",
    }


def _full_domain_run_identity(repo_root):
    benchmark_path = Path(__file__).resolve()
    implementation_path = Path(_impl.__file__).resolve()
    for description, source_path in (
        ("benchmark", benchmark_path),
        ("fused KDA implementation", implementation_path),
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
    source_commit = subprocess.run(
        ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(source_commit) != 40:
        raise RuntimeError("benchmark source commit is not a full Git object ID")
    shape_inventory = [
        {"num_heads": heads, "num_rows": rows} for heads, rows in _FULL_DOMAIN_SHAPES
    ]
    return {
        "source_commit": source_commit,
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            implementation_path.read_bytes()
        ).hexdigest(),
        "program": _cake_program_record(repo_root),
        "shape_inventory": "dense rows 1..256 plus tail rows, for H=12,24,32,48,96",
        "shape_count": len(_FULL_DOMAIN_SHAPES),
        "shape_inventory_sha256": _canonical_json_sha256(shape_inventory),
        "candidate_route": "public fused_kda_decode Cake dispatcher",
        "baseline": _attest_exact_pr_fallback(repo_root, implementation_path),
    }


def _full_domain_session_identity():
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu_uuid, pci_bus_id = _query_single_visible_gpu_identity()
    return {
        "gpu": {
            "name": properties.name,
            "compute_capability": list(get_compute_capability(torch.device("cuda"))),
            "sm_count": properties.multi_processor_count,
            "uuid": gpu_uuid,
            "pci_bus_id": pci_bus_id,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "managed_step_id": os.environ.get("CODESLACK_MANAGED_STEP_ID"),
        "managed_step_attempt": os.environ.get("CODESLACK_STEP_ATTEMPT"),
        "process_started_unix_seconds": time.time(),
    }


def _run_full_domain_cell(
    args,
    backend,
    num_heads,
    num_rows,
    *,
    capture_reference=False,
    correctness_reference=None,
):
    inputs = _make_inputs(num_heads, num_rows)
    variant_name = None
    if backend == "baseline":
        inputs["backend"] = "cute-dsl"
        route_guard = mock.patch.object(
            _impl, "_select_cake_variant", return_value=None
        )
        fallback_guard = mock.patch.object(
            _impl, "_get_compiled_kernel", wraps=_impl._get_compiled_kernel
        )
    else:
        inputs["backend"] = "cake"
        inputs["state_indices_mode"] = "positive_unique"
        variant = _impl._select_cake_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state=inputs["state"],
            output_gate=inputs["output_gate"],
            output=inputs["output"],
            state_indices_mode=inputs["state_indices_mode"],
            lower_bound=inputs["lower_bound"],
            norm_eps=inputs["norm_eps"],
        )
        if variant is None:
            raise RuntimeError("candidate inputs did not select a Cake variant")
        variant_name = variant.name
        route_guard = mock.patch.object(
            _impl, "_select_cake_variant", wraps=_impl._select_cake_variant
        )
        fallback_guard = mock.patch.object(
            _impl,
            "_get_compiled_kernel",
            side_effect=RuntimeError("candidate fell back to the exact PR kernel"),
        )

    with route_guard as route_mock, fallback_guard as fallback_mock:
        fused_kda_decode(**inputs)
        torch.cuda.synchronize()
        correctness_snapshot = None
        correctness = None
        if capture_reference:
            correctness_snapshot = {
                name: inputs[name].detach().clone()
                for name in ("output", "conv_state", "state")
            }
        if correctness_reference is not None:
            torch.testing.assert_close(
                inputs["output"], correctness_reference["output"], rtol=3e-2, atol=2e-2
            )
            torch.testing.assert_close(
                inputs["conv_state"],
                correctness_reference["conv_state"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                inputs["state"], correctness_reference["state"], rtol=3e-2, atol=2e-3
            )
            correctness = {
                "checked": True,
                "candidate": "public Cake dispatcher",
                "reference": "exact PR fallback",
                "output": {"rtol": 3e-2, "atol": 2e-2, "passed": True},
                "conv_state": {"rtol": 0.0, "atol": 0.0, "passed": True},
                "state": {"rtol": 3e-2, "atol": 2e-3, "passed": True},
            }
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
        torch.cuda.synchronize()
        route_call_count = route_mock.call_count
        fallback_call_count = fallback_mock.call_count

    samples = _validate_samples(
        [float(value) for value in samples],
        args.repeat_iters,
        f"h{num_heads}_rows{num_rows} {backend} cell",
    )
    if backend == "baseline":
        if route_call_count != 0 or fallback_call_count < 1:
            raise RuntimeError("baseline cell did not isolate the exact PR fallback")
    elif route_call_count < 1 or fallback_call_count != 0:
        raise RuntimeError("candidate cell did not isolate the Cake backend")
    del inputs
    measurement = {
        "backend": backend,
        "variant_name": variant_name,
        "route_call_count": route_call_count,
        "fallback_call_count": fallback_call_count,
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
    }
    return measurement, correctness_snapshot, correctness


def _validate_full_domain_row(row, index, repeat_iters, *, legacy=False):
    if not isinstance(row, dict) or set(row) != {
        "shape",
        "num_heads",
        "num_rows",
        "baseline_ms",
        "candidate_ms",
        "paired_speedups",
        "speedup",
        "correctness",
        "measurements",
    }:
        raise RuntimeError("full-domain row schema is invalid")
    num_heads, num_rows = _FULL_DOMAIN_SHAPES[index]
    shape = f"h{num_heads}_rows{num_rows}"
    if (
        row["shape"] != shape
        or row["num_heads"] != num_heads
        or type(row["num_heads"]) is not int
        or row["num_rows"] != num_rows
        or type(row["num_rows"]) is not int
    ):
        raise RuntimeError("full-domain rows are not a contiguous shape prefix")
    measurements = row["measurements"]
    if not isinstance(measurements, list) or len(measurements) != 4:
        raise RuntimeError(f"{shape} does not contain four ABBA cells")
    candidate_variants = set()
    for order_index, expected_backend in enumerate(_ABBA_ORDER):
        cell = measurements[order_index]
        if not isinstance(cell, dict) or set(cell) != {
            "order_index",
            "backend",
            "variant_name",
            "route_call_count",
            "fallback_call_count",
            "median_ms",
            "samples_ms",
        }:
            raise RuntimeError(f"{shape} measurement schema is invalid")
        if cell["order_index"] != order_index or cell["backend"] != expected_backend:
            raise RuntimeError(f"{shape} measurement order is not exact ABBA")
        samples = _validate_samples(
            cell["samples_ms"], repeat_iters, f"{shape} cell {order_index}"
        )
        _require_close(
            cell["median_ms"],
            statistics.median(samples),
            f"{shape} cell {order_index} median",
        )
        if type(cell["route_call_count"]) is not int:
            raise RuntimeError(f"{shape} route count is invalid")
        if expected_backend == "baseline":
            route_count_is_valid = (
                cell["route_call_count"] >= 1
                if legacy
                else cell["route_call_count"] == 0
            )
            if (
                cell["variant_name"] is not None
                or not route_count_is_valid
                or cell["fallback_call_count"] < 1
            ):
                raise RuntimeError(f"{shape} baseline route proof is invalid")
        else:
            if (
                not isinstance(cell["variant_name"], str)
                or not cell["variant_name"]
                or cell["route_call_count"] < 1
                or cell["fallback_call_count"] != 0
            ):
                raise RuntimeError(f"{shape} Cake route proof is invalid")
            candidate_variants.add(cell["variant_name"])
    if len(candidate_variants) != 1:
        raise RuntimeError(f"{shape} candidate cells selected different variants")
    correctness = row["correctness"]
    expected_candidate = (
        "public generated dispatcher" if legacy else "public Cake dispatcher"
    )
    if (
        not isinstance(correctness, dict)
        or correctness.get("checked") is not True
        or correctness.get("candidate") != expected_candidate
        or correctness.get("reference") != "exact PR fallback"
        or any(
            correctness.get(name, {}).get("passed") is not True
            for name in ("output", "conv_state", "state")
        )
    ):
        raise RuntimeError(f"{shape} correctness evidence is invalid")
    baseline_medians = [measurements[0]["median_ms"], measurements[3]["median_ms"]]
    candidate_medians = [measurements[1]["median_ms"], measurements[2]["median_ms"]]
    paired_speedups = [
        baseline_medians[0] / candidate_medians[0],
        baseline_medians[1] / candidate_medians[1],
    ]
    if not isinstance(row["paired_speedups"], list) or len(row["paired_speedups"]) != 2:
        raise RuntimeError(f"{shape} paired speedups are invalid")
    for pair_index, expected in enumerate(paired_speedups):
        _require_close(
            row["paired_speedups"][pair_index], expected, f"{shape} pair {pair_index}"
        )
    _require_close(
        row["baseline_ms"], statistics.median(baseline_medians), f"{shape} baseline"
    )
    _require_close(
        row["candidate_ms"],
        statistics.median(candidate_medians),
        f"{shape} candidate",
    )
    _require_close(
        row["speedup"], statistics.median(paired_speedups), f"{shape} speedup"
    )


def _full_domain_rows_root(output_path):
    return output_path.with_name(f"{output_path.name}.rows")


def _full_domain_row_path(rows_root, index):
    return rows_root / f"row-{index:04d}.json"


def _load_full_domain_rows(
    rows_root,
    identity_sha256,
    repeat_iters,
    *,
    inherited_identity_sha256_by_row=None,
):
    if not rows_root.is_dir():
        return [], []
    row_files = sorted(rows_root.glob("row-*.json"))
    expected_names = {
        _full_domain_row_path(rows_root, index).name
        for index in range(len(_FULL_DOMAIN_SHAPES))
    }
    unexpected = [path.name for path in row_files if path.name not in expected_names]
    if unexpected:
        raise RuntimeError(f"unexpected full-domain row receipts: {unexpected[:3]}")
    rows = []
    row_receipts = []
    missing_seen = False
    for index in range(len(_FULL_DOMAIN_SHAPES)):
        path = _full_domain_row_path(rows_root, index)
        if not path.is_file():
            missing_seen = True
            continue
        if missing_seen:
            raise RuntimeError("full-domain row receipts contain a gap")
        receipt_bytes = path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema",
            "identity_sha256",
            "shape_index",
            "session",
            "row",
        }:
            raise RuntimeError(f"full-domain receipt {path.name} schema is invalid")
        if inherited_identity_sha256_by_row is not None and index < len(
            inherited_identity_sha256_by_row
        ):
            expected_identity_sha256 = inherited_identity_sha256_by_row[index]
        else:
            expected_identity_sha256 = identity_sha256
        inherited = inherited_identity_sha256_by_row is not None and index < len(
            inherited_identity_sha256_by_row
        )
        valid_schemas = (
            {_FULL_DOMAIN_ROW_SCHEMA, _LEGACY_FULL_DOMAIN_ROW_SCHEMA}
            if inherited
            else {_FULL_DOMAIN_ROW_SCHEMA}
        )
        if (
            receipt["schema"] not in valid_schemas
            or receipt["identity_sha256"] != expected_identity_sha256
            or receipt["shape_index"] != index
        ):
            raise RuntimeError(f"full-domain receipt {path.name} identity is invalid")
        session = receipt["session"]
        gpu = session.get("gpu") if isinstance(session, dict) else None
        if (
            not isinstance(gpu, dict)
            or gpu.get("compute_capability") != [10, 0]
            or "B200" not in str(gpu.get("name", "")).upper()
        ):
            raise RuntimeError(f"full-domain receipt {path.name} is not B200 evidence")
        _validate_full_domain_row(
            receipt["row"],
            index,
            repeat_iters,
            legacy=receipt["schema"] == _LEGACY_FULL_DOMAIN_ROW_SCHEMA,
        )
        rows.append(receipt["row"])
        row_receipts.append(
            {"path": path.name, "sha256": hashlib.sha256(receipt_bytes).hexdigest()}
        )
    return rows, row_receipts


def _checkpoint_receipt_identity_sha256_by_row(
    checkpoint_path,
    checkpoint,
    *,
    seen_paths=None,
):
    """Resolve immutable receipt identities through an inheritance chain."""

    seen = set() if seen_paths is None else set(seen_paths)
    resolved_path = checkpoint_path.resolve()
    if resolved_path in seen:
        raise RuntimeError("full-domain checkpoint inheritance contains a cycle")
    seen.add(resolved_path)
    identity = checkpoint.get("identity")
    progress = checkpoint.get("progress")
    if not isinstance(identity, dict) or not isinstance(progress, dict):
        raise RuntimeError("inherited checkpoint identity or progress is invalid")
    completed_rows = progress.get("completed_rows")
    if (
        type(completed_rows) is not int
        or completed_rows < 1
        or completed_rows > len(_FULL_DOMAIN_SHAPES)
    ):
        raise RuntimeError("inherited checkpoint completed row count is invalid")
    identity_sha256 = _canonical_json_sha256(identity)
    identities = [identity_sha256] * completed_rows
    inheritance = checkpoint.get("inheritance")
    if inheritance is None:
        return identities
    legacy_keys = {
        "manifest_path",
        "manifest_sha256",
        "predecessor_checkpoint",
        "predecessor_identity_sha256",
        "completed_rows",
        "inherited_route_names",
    }
    current_keys = {
        "equivalence_path",
        "equivalence_sha256",
        "predecessor_checkpoint",
        "predecessor_identity_sha256",
        "completed_rows",
        "inherited_route_names",
    }
    if not isinstance(inheritance, dict) or set(inheritance) not in (
        legacy_keys,
        current_keys,
    ):
        raise RuntimeError("checkpoint inheritance attestation is invalid")
    inherited_count = inheritance["completed_rows"]
    if (
        type(inherited_count) is not int
        or inherited_count < 1
        or inherited_count > completed_rows
    ):
        raise RuntimeError("checkpoint inherited row count is invalid")
    predecessor_record = inheritance["predecessor_checkpoint"]
    if not isinstance(predecessor_record, dict) or set(predecessor_record) != {
        "path",
        "sha256",
    }:
        raise RuntimeError("checkpoint predecessor record is invalid")
    predecessor_path = Path(predecessor_record["path"]).resolve()
    if not predecessor_path.is_file() or predecessor_path.is_symlink():
        raise RuntimeError("checkpoint predecessor is not a regular file")
    predecessor_bytes = predecessor_path.read_bytes()
    if hashlib.sha256(predecessor_bytes).hexdigest() != predecessor_record["sha256"]:
        raise RuntimeError("checkpoint predecessor SHA-256 mismatch")
    predecessor = json.loads(predecessor_bytes)
    predecessor_identity = predecessor.get("identity")
    if (
        predecessor.get("schema")
        not in (
            _FULL_DOMAIN_SCHEMA,
            _LEGACY_FULL_DOMAIN_SCHEMA,
        )
        or predecessor.get("status") not in ("in_progress", "complete")
        or predecessor.get("measurement") != checkpoint.get("measurement")
        or predecessor.get("progress", {}).get("completed_rows") != inherited_count
        or not isinstance(predecessor_identity, dict)
        or _canonical_json_sha256(predecessor_identity)
        != inheritance["predecessor_identity_sha256"]
        or predecessor_identity.get("baseline") != identity.get("baseline")
        or predecessor_identity.get("shape_inventory_sha256")
        != identity.get("shape_inventory_sha256")
    ):
        raise RuntimeError("checkpoint predecessor identity or protocol is invalid")
    predecessor_identities = _checkpoint_receipt_identity_sha256_by_row(
        predecessor_path,
        predecessor,
        seen_paths=seen,
    )
    if len(predecessor_identities) != inherited_count:
        raise RuntimeError("checkpoint predecessor receipt inventory is invalid")
    identities[:inherited_count] = predecessor_identities
    return identities


def _require_sha256(value, description):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{description} is not a SHA-256 digest")


def _normalize_kernel_symbol(source, symbol, description):
    encoded_symbol = symbol.encode()
    occurrences = source.count(encoded_symbol)
    if occurrences != 1:
        raise RuntimeError(
            f"{description} contains {occurrences} occurrences of {symbol!r}, expected one"
        )
    return source.replace(encoded_symbol, b"CAKE_FUSED_KDA_KERNEL_SYMBOL")


def _normalize_predecessor_binding(source):
    body_guard = (
        b"#ifndef FLASHINFER_FUSED_KDA_DECODE_BODY_FILE\n"
        b'#error "FLASHINFER_FUSED_KDA_DECODE_BODY_FILE must name one frozen CUDA body"\n'
        b"#endif\n"
    )
    if source.count(body_guard) != 1:
        raise RuntimeError("predecessor binding body guard is invalid")
    source = source.replace(body_guard, b"", 1)
    preamble_start = source.index(b"#include <cstdint>\n")
    preamble_end = source.index(b"#undef int8_t\n", preamble_start) + len(
        b"#undef int8_t\n"
    )
    source = source[:preamble_start] + source[preamble_end:]
    replacements = (
        (
            b"FLASHINFER_FUSED_KDA_DECODE",
            b"FLASHINFER_CAKE_FUSED_KDA_DECODE",
        ),
        (b"fused_kda_decode_generated", b"cake_fused_kda_decode"),
        (
            b"fused KDA decode generated kernel ABI changed",
            b"Cake fused KDA decode kernel ABI changed",
        ),
        (
            b"fused KDA decode argument-plan identity",
            b"Cake fused KDA decode argument-plan identity",
        ),
        (b"this fused KDA decode module", b"this Cake fused KDA decode module"),
        (
            b"fused KDA decode dynamic shared memory",
            b"Cake fused KDA decode dynamic shared memory",
        ),
        (
            b"cudaFuncSetAttribute(fused KDA decode)",
            b"cudaFuncSetAttribute(Cake fused KDA decode)",
        ),
        (
            b"fused KDA decode repeated-row launch",
            b"Cake fused KDA decode repeated-row launch",
        ),
        (b"fused KDA decode launch", b"Cake fused KDA decode launch"),
    )
    for predecessor, current in replacements:
        source = source.replace(predecessor, current)
    return source.replace(
        b'identity must be a full SHA-256");\n\n\n#include',
        b'identity must be a full SHA-256");\n\n#include',
        1,
    )


def _legacy_eligibility(variant):
    result = []
    for rule in variant.get("eligibility", []):
        if not isinstance(rule, dict) or "slot_classes" not in rule:
            raise RuntimeError("predecessor eligibility schema is invalid")
        converted = dict(rule)
        converted["state_indices_modes"] = converted.pop("slot_classes")
        result.append(converted)
    return result


def _validate_equivalence_receipt(
    receipt,
    *,
    repo_root,
    predecessor_identity,
    predecessor_manifest,
    predecessor_manifest_sha256,
    current_identity,
    predecessor_rows,
):
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "verifier",
        "predecessor",
        "current",
        "variants",
        "route_equivalence",
        "result",
    }:
        raise RuntimeError("equivalence receipt schema is invalid")
    if receipt["schema"] != _EQUIVALENCE_SCHEMA:
        raise RuntimeError("unsupported equivalence receipt")

    verifier = receipt["verifier"]
    if not isinstance(verifier, dict) or set(verifier) != {
        "script_sha256",
        "slurm_job_id",
        "node",
        "gpu_uuid",
        "gpu_name",
        "compute_capability",
        "cuda_version",
        "nvcc_version",
        "cuobjdump_version",
        "python_version",
        "torch_version",
        "tvm_ffi_version",
        "toolchain_sha256",
    }:
        raise RuntimeError("equivalence verifier identity is invalid")
    _require_sha256(verifier["script_sha256"], "equivalence verifier script")
    _require_sha256(verifier["toolchain_sha256"], "equivalence toolchain")
    if (
        not all(
            isinstance(verifier[name], str) and verifier[name]
            for name in (
                "slurm_job_id",
                "node",
                "gpu_uuid",
                "gpu_name",
                "cuda_version",
                "nvcc_version",
                "cuobjdump_version",
                "python_version",
                "torch_version",
                "tvm_ffi_version",
            )
        )
        or "B200" not in verifier["gpu_name"].upper()
        or verifier["compute_capability"] != [10, 0]
    ):
        raise RuntimeError("equivalence receipt is not complete B200 evidence")
    verifier_path = repo_root / _EQUIVALENCE_VERIFIER
    toolchain = {
        name: verifier[name]
        for name in (
            "cuda_version",
            "nvcc_version",
            "cuobjdump_version",
            "python_version",
            "torch_version",
            "tvm_ffi_version",
        )
    }
    if verifier["script_sha256"] != hashlib.sha256(
        verifier_path.read_bytes()
    ).hexdigest() or verifier["toolchain_sha256"] != _canonical_json_sha256(toolchain):
        raise RuntimeError("equivalence verifier or toolchain identity is invalid")

    predecessor = receipt["predecessor"]
    current = receipt["current"]
    if not isinstance(predecessor, dict) or set(predecessor) != {
        "commit",
        "manifest_sha256",
    }:
        raise RuntimeError("equivalence predecessor identity is invalid")
    if not isinstance(current, dict) or set(current) != {
        "commit",
        "program_identity_sha256",
        "registry_sha256",
        "binding_sha256",
    }:
        raise RuntimeError("equivalence current identity is invalid")
    registry_path = Path(
        importlib.import_module("flashinfer.jit.cake_fused_kda_decode").__file__
    ).resolve()
    binding_path = repo_root / "csrc/kda/cake_fused_kda_decode_binding.cuh"
    predecessor_binding = subprocess.run(
        (
            "git",
            "-C",
            str(repo_root),
            "show",
            f"{predecessor['commit']}:csrc/kda/fused_kda_decode_generated_binding.cuh",
        ),
        check=True,
        capture_output=True,
    ).stdout
    if (
        predecessor["commit"] != _MEASURED_PREDECESSOR_COMMIT
        or predecessor["manifest_sha256"] != predecessor_manifest_sha256
        or current["commit"] != current_identity.get("source_commit")
        or current["program_identity_sha256"]
        != current_identity.get("program", {}).get("identity_sha256")
        or current["registry_sha256"]
        != hashlib.sha256(registry_path.read_bytes()).hexdigest()
        or current["binding_sha256"]
        != hashlib.sha256(binding_path.read_bytes()).hexdigest()
        or _normalize_predecessor_binding(predecessor_binding)
        != binding_path.read_bytes()
    ):
        raise RuntimeError("equivalence source identity does not match this run")

    predecessor_variants = predecessor_manifest.get("variants")
    current_variants = current_identity.get("program", {}).get("variants")
    proof_variants = receipt["variants"]
    if (
        not isinstance(predecessor_variants, list)
        or not isinstance(current_variants, list)
        or not isinstance(proof_variants, list)
        or len(predecessor_variants) != 44
        or len(current_variants) != 44
        or len(proof_variants) != 44
    ):
        raise RuntimeError("equivalence receipt must cover exactly 44 variants")

    proof_variant_keys = {
        "name",
        "target",
        "predecessor_source_sha256",
        "current_source_sha256",
        "predecessor_kernel_symbol",
        "current_kernel_symbol",
        "normalized_source_sha256",
        "symbol_rename_occurrences",
        "abi_sha256",
        "compile_flags_sha256",
        "predecessor_sass_sha256",
        "current_sass_sha256",
        "predecessor_resource_usage_sha256",
        "current_resource_usage_sha256",
        "execution_input_sha256",
        "predecessor_execution_sha256",
        "current_execution_sha256",
        "source_transform_exact",
        "abi_equal",
        "compile_flags_equal",
        "launch_equal",
        "sass_equal",
        "resource_usage_equal",
        "execution_bitwise_equal",
    }
    for predecessor_variant, current_variant, proof in zip(
        predecessor_variants, current_variants, proof_variants, strict=True
    ):
        if not isinstance(proof, dict) or set(proof) != proof_variant_keys:
            raise RuntimeError("equivalence variant record schema is invalid")
        name = current_variant.get("name")
        if predecessor_variant.get("name") != name or proof["name"] != name:
            raise RuntimeError("equivalence variant order or name changed")
        predecessor_abi = (
            predecessor_manifest.get("contract", {})
            .get("kernel_abis", {})
            .get(predecessor_variant.get("abi_kind"))
        )
        metadata_equal = (
            predecessor_variant.get("target") == current_variant.get("target")
            and predecessor_variant.get("abi_kind") == current_variant.get("abi_kind")
            and predecessor_abi == current_variant.get("abi")
            and predecessor_variant.get("state_dtype")
            == current_variant.get("state_dtype")
            and predecessor_variant.get("slot_offset_bits", 32)
            == current_variant.get("slot_offset_bits")
            and predecessor_variant.get("extra_cuda_cflags")
            == current_variant.get("extra_cuda_cflags")
            and predecessor_variant.get("launch", {}).get("threads")
            == current_variant.get("threads")
            and predecessor_variant.get("launch", {}).get("dynamic_smem_bytes")
            == current_variant.get("dynamic_smem_bytes")
            and _legacy_eligibility(predecessor_variant)
            == current_variant.get("eligibility")
        )
        if not metadata_equal:
            raise RuntimeError(f"static launch contract changed for {name!r}")

        predecessor_body = f"csrc/kda/{predecessor_variant['body']}"
        predecessor_source = subprocess.run(
            (
                "git",
                "-C",
                str(repo_root),
                "show",
                f"{predecessor['commit']}:{predecessor_body}",
            ),
            check=True,
            capture_output=True,
        ).stdout
        current_source = (repo_root / current_variant["body"]).read_bytes()
        if hashlib.sha256(predecessor_source).hexdigest() != predecessor_variant.get(
            "source_sha256"
        ) or hashlib.sha256(current_source).hexdigest() != current_variant.get(
            "source_sha256"
        ):
            raise RuntimeError(f"source identity changed for {name!r}")
        predecessor_normalized = _normalize_kernel_symbol(
            predecessor_source,
            predecessor_variant["kernel_symbol"],
            f"predecessor {name}",
        )
        current_normalized = _normalize_kernel_symbol(
            current_source,
            current_variant["kernel_symbol"],
            f"current {name}",
        )
        normalized_sha256 = hashlib.sha256(current_normalized).hexdigest()
        abi_sha256 = _canonical_json_sha256(current_variant["abi"])
        for digest_name in (
            "compile_flags_sha256",
            "predecessor_sass_sha256",
            "current_sass_sha256",
            "predecessor_resource_usage_sha256",
            "current_resource_usage_sha256",
            "execution_input_sha256",
            "predecessor_execution_sha256",
            "current_execution_sha256",
        ):
            _require_sha256(proof[digest_name], f"{name} {digest_name}")
        if (
            predecessor_normalized != current_normalized
            or proof["target"] != current_variant["target"]
            or proof["predecessor_source_sha256"]
            != predecessor_variant["source_sha256"]
            or proof["current_source_sha256"] != current_variant["source_sha256"]
            or proof["predecessor_kernel_symbol"]
            != predecessor_variant["kernel_symbol"]
            or proof["current_kernel_symbol"] != current_variant["kernel_symbol"]
            or proof["normalized_source_sha256"] != normalized_sha256
            or proof["symbol_rename_occurrences"] != 1
            or proof["abi_sha256"] != abi_sha256
            or proof["predecessor_sass_sha256"] != proof["current_sass_sha256"]
            or proof["predecessor_resource_usage_sha256"]
            != proof["current_resource_usage_sha256"]
            or proof["predecessor_execution_sha256"]
            != proof["current_execution_sha256"]
            or any(
                proof[name] is not True
                for name in (
                    "source_transform_exact",
                    "abi_equal",
                    "compile_flags_equal",
                    "launch_equal",
                    "sass_equal",
                    "resource_usage_equal",
                    "execution_bitwise_equal",
                )
            )
        ):
            raise RuntimeError(f"execution equivalence failed for {name!r}")

    variants = get_cake_fused_kda_decode_variants()
    route_records = []
    used_variants = set()
    for index, row in enumerate(predecessor_rows):
        candidate_cells = [
            cell for cell in row["measurements"] if cell["backend"] == "candidate"
        ]
        names = {cell["variant_name"] for cell in candidate_cells}
        if len(names) != 1:
            raise RuntimeError(f"predecessor row {index} has inconsistent routes")
        predecessor_name = names.pop()
        heads, num_rows = _FULL_DOMAIN_SHAPES[index]
        conv_stride, state_stride = _page_strides(heads)
        selected = select_cake_fused_kda_decode_variant(
            target="sm100a",
            num_heads=heads,
            num_rows=num_rows,
            num_slots=num_rows + 1,
            state_dtype="float32",
            state_indices_mode="positive_unique",
            lower_bound=-5.0,
            norm_eps=1.0e-5,
            x_row_stride=3 * heads * _HEAD_DIM + 17,
            conv_slot_stride=conv_stride,
            beta_row_stride=heads + 1,
            state_slot_stride=state_stride,
            output_gate_row_stride=heads * _HEAD_DIM + 7,
            variants=variants,
        )
        if selected is None or selected.name != predecessor_name:
            raise RuntimeError(
                f"current selector does not preserve inherited row {index} route"
            )
        used_variants.add(predecessor_name)
        route_records.append(
            {
                "shape_index": index,
                "num_heads": heads,
                "num_rows": num_rows,
                "variant_name": predecessor_name,
            }
        )
    route_sha256 = _canonical_json_sha256(route_records)
    route_equivalence = receipt["route_equivalence"]
    if not isinstance(route_equivalence, dict) or set(route_equivalence) != {
        "shape_count",
        "shape_inventory_sha256",
        "predecessor_routes_sha256",
        "current_routes_sha256",
        "missing",
        "mismatches",
        "used_variants",
    }:
        raise RuntimeError("route equivalence schema is invalid")
    if (
        route_equivalence["shape_count"] != len(predecessor_rows)
        or route_equivalence["shape_inventory_sha256"]
        != current_identity["shape_inventory_sha256"]
        or route_equivalence["predecessor_routes_sha256"] != route_sha256
        or route_equivalence["current_routes_sha256"] != route_sha256
        or route_equivalence["missing"] != 0
        or route_equivalence["mismatches"] != []
        or route_equivalence["used_variants"] != sorted(used_variants)
    ):
        raise RuntimeError("route equivalence does not match the inherited rows")

    result = receipt["result"]
    if not isinstance(result, dict) or set(result) != {
        "variant_count",
        "route_count",
        "all_variants_passed",
        "all_routes_passed",
        "eligible_for_timing_inheritance",
    }:
        raise RuntimeError("equivalence result schema is invalid")
    if result != {
        "variant_count": 44,
        "route_count": len(predecessor_rows),
        "all_variants_passed": True,
        "all_routes_passed": True,
        "eligible_for_timing_inheritance": True,
    }:
        raise RuntimeError("equivalence receipt did not pass every gate")
    return sorted(used_variants)


def _load_full_domain_inheritance(
    inheritance_path,
    *,
    current_identity,
    current_identity_sha256,
    measurement,
    rows_root,
    repeat_iters,
):
    inheritance_bytes = inheritance_path.read_bytes()
    inheritance = json.loads(inheritance_bytes)
    if not isinstance(inheritance, dict) or set(inheritance) != {
        "schema",
        "predecessor_checkpoint",
        "predecessor_manifest",
        "predecessor_rows_root",
        "predecessor_identity_sha256",
        "completed_rows",
        "current_identity_sha256",
        "equivalence_receipt",
    }:
        raise RuntimeError("full-domain inheritance schema is invalid")
    if inheritance["schema"] != _FULL_DOMAIN_INHERITANCE_SCHEMA:
        raise RuntimeError("unsupported full-domain inheritance record")
    if inheritance["current_identity_sha256"] != current_identity_sha256:
        raise RuntimeError("inheritance record does not name the current identity")
    completed_rows = inheritance["completed_rows"]
    if (
        type(completed_rows) is not int
        or completed_rows < 1
        or completed_rows > len(_FULL_DOMAIN_SHAPES)
    ):
        raise RuntimeError("inheritance completed row count is invalid")

    predecessor_checkpoint_record = inheritance["predecessor_checkpoint"]
    if not isinstance(predecessor_checkpoint_record, dict) or set(
        predecessor_checkpoint_record
    ) != {"path", "sha256"}:
        raise RuntimeError("predecessor checkpoint record is invalid")
    predecessor_checkpoint_path = Path(predecessor_checkpoint_record["path"]).resolve()
    if (
        not predecessor_checkpoint_path.is_file()
        or predecessor_checkpoint_path.is_symlink()
    ):
        raise RuntimeError("predecessor checkpoint is not a regular file")
    predecessor_checkpoint_bytes = predecessor_checkpoint_path.read_bytes()
    if (
        hashlib.sha256(predecessor_checkpoint_bytes).hexdigest()
        != predecessor_checkpoint_record["sha256"]
    ):
        raise RuntimeError("predecessor checkpoint SHA-256 mismatch")
    predecessor_checkpoint = json.loads(predecessor_checkpoint_bytes)
    predecessor_identity = predecessor_checkpoint.get("identity")
    if not isinstance(predecessor_identity, dict):
        raise RuntimeError("predecessor identity is invalid")
    predecessor_identity_sha256 = _canonical_json_sha256(predecessor_identity)
    if (
        predecessor_checkpoint.get("schema")
        not in (_FULL_DOMAIN_SCHEMA, _LEGACY_FULL_DOMAIN_SCHEMA)
        or predecessor_checkpoint.get("status") not in ("in_progress", "complete")
        or predecessor_checkpoint.get("measurement") != measurement
        or predecessor_checkpoint.get("progress", {}).get("completed_rows")
        != completed_rows
        or predecessor_identity_sha256 != inheritance["predecessor_identity_sha256"]
        or predecessor_identity.get("baseline") != current_identity.get("baseline")
        or predecessor_identity.get("shape_inventory_sha256")
        != current_identity.get("shape_inventory_sha256")
    ):
        raise RuntimeError("predecessor checkpoint identity or protocol is invalid")

    predecessor_manifest_record = inheritance["predecessor_manifest"]
    if not isinstance(predecessor_manifest_record, dict) or set(
        predecessor_manifest_record
    ) != {"path", "sha256"}:
        raise RuntimeError("predecessor manifest record is invalid")
    predecessor_manifest_path = Path(predecessor_manifest_record["path"]).resolve()
    if (
        not predecessor_manifest_path.is_file()
        or predecessor_manifest_path.is_symlink()
    ):
        raise RuntimeError("predecessor manifest is not a regular file")
    predecessor_manifest_bytes = predecessor_manifest_path.read_bytes()
    predecessor_manifest_sha256 = hashlib.sha256(predecessor_manifest_bytes).hexdigest()
    if predecessor_manifest_sha256 != predecessor_manifest_record[
        "sha256"
    ] or predecessor_manifest_sha256 != predecessor_identity.get("manifest_sha256"):
        raise RuntimeError("predecessor manifest SHA-256 mismatch")
    predecessor_manifest = json.loads(predecessor_manifest_bytes)

    predecessor_rows_root = Path(inheritance["predecessor_rows_root"]).resolve()
    predecessor_identity_sha256_by_row = _checkpoint_receipt_identity_sha256_by_row(
        predecessor_checkpoint_path,
        predecessor_checkpoint,
    )
    if len(predecessor_identity_sha256_by_row) != completed_rows:
        raise RuntimeError("predecessor checkpoint receipt identity count is invalid")
    predecessor_rows, predecessor_receipts = _load_full_domain_rows(
        predecessor_rows_root,
        predecessor_identity_sha256,
        repeat_iters,
        inherited_identity_sha256_by_row=predecessor_identity_sha256_by_row,
    )
    if len(predecessor_rows) != completed_rows:
        raise RuntimeError("predecessor row receipts do not match completed rows")
    if predecessor_checkpoint.get("row_receipts") != predecessor_receipts:
        raise RuntimeError("predecessor checkpoint row inventory is invalid")

    equivalence_record = inheritance["equivalence_receipt"]
    if not isinstance(equivalence_record, dict) or set(equivalence_record) != {
        "path",
        "sha256",
    }:
        raise RuntimeError("equivalence receipt record is invalid")
    equivalence_path = Path(equivalence_record["path"]).resolve()
    if not equivalence_path.is_file() or equivalence_path.is_symlink():
        raise RuntimeError("equivalence receipt is not a regular file")
    equivalence_bytes = equivalence_path.read_bytes()
    if hashlib.sha256(equivalence_bytes).hexdigest() != equivalence_record["sha256"]:
        raise RuntimeError("equivalence receipt SHA-256 mismatch")
    used_variants = _validate_equivalence_receipt(
        json.loads(equivalence_bytes),
        repo_root=Path(__file__).resolve().parents[1],
        predecessor_identity=predecessor_identity,
        predecessor_manifest=predecessor_manifest,
        predecessor_manifest_sha256=predecessor_manifest_sha256,
        current_identity=current_identity,
        predecessor_rows=predecessor_rows,
    )

    rows_root.mkdir(parents=True, exist_ok=True)
    for index in range(completed_rows):
        source = _full_domain_row_path(predecessor_rows_root, index)
        destination = _full_domain_row_path(rows_root, index)
        source_bytes = source.read_bytes()
        if destination.exists():
            if destination.read_bytes() != source_bytes:
                raise RuntimeError(f"inherited row {index} copy differs")
            continue
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(source_bytes)
        os.replace(temporary, destination)

    return {
        "equivalence_path": str(equivalence_path),
        "equivalence_sha256": hashlib.sha256(equivalence_bytes).hexdigest(),
        "predecessor_checkpoint": predecessor_checkpoint_record,
        "predecessor_identity_sha256": predecessor_identity_sha256,
        "completed_rows": completed_rows,
        "inherited_route_names": used_variants,
    }, predecessor_identity_sha256_by_row


def _summarize_full_domain(rows):
    if len(rows) != len(_FULL_DOMAIN_SHAPES):
        raise RuntimeError("full-domain summary requires every shape")
    official_shapes = set(_OFFICIAL_SHAPES)
    original_shapes = set(_ORIGINAL_SHAPES)
    official_speedups = [
        row["speedup"]
        for row in rows
        if (row["num_heads"], row["num_rows"]) in official_shapes
    ]
    original_speedups = [
        row["speedup"]
        for row in rows
        if (row["num_heads"], row["num_rows"]) in original_shapes
    ]
    if len(official_speedups) != 21 or len(original_speedups) != 17:
        raise RuntimeError("full-domain rows do not cover the official subsets")
    speedups = [row["speedup"] for row in rows]
    original17 = _geometric_mean(original_speedups)
    minimum = min(speedups)
    return {
        "shape_count": len(rows),
        "baseline_geomean_ms": _geometric_mean([row["baseline_ms"] for row in rows]),
        "candidate_geomean_ms": _geometric_mean([row["candidate_ms"] for row in rows]),
        "full_domain_geomean_speedup": _geometric_mean(speedups),
        "official21_geomean_speedup": _geometric_mean(official_speedups),
        "original17_geomean_speedup": original17,
        "minimum_speedup": minimum,
        "every_shape_faster": all(speedup > 1.0 for speedup in speedups),
        "original17_at_least_1_10": original17 >= 1.10,
        "minimum_at_least_1_01": minimum >= 1.01,
    }


def _write_full_domain_checkpoint(
    output_path,
    *,
    status,
    identity,
    measurement,
    rows,
    row_receipts,
    inheritance=None,
    summary=None,
):
    payload = {
        "schema": _FULL_DOMAIN_SCHEMA,
        "status": status,
        "identity": identity,
        "measurement": measurement,
        "progress": {
            "completed_rows": len(rows),
            "total_rows": len(_FULL_DOMAIN_SHAPES),
            "next_shape": (
                None
                if len(rows) == len(_FULL_DOMAIN_SHAPES)
                else {
                    "num_heads": _FULL_DOMAIN_SHAPES[len(rows)][0],
                    "num_rows": _FULL_DOMAIN_SHAPES[len(rows)][1],
                }
            ),
        },
        "row_receipts": row_receipts,
    }
    if inheritance is not None:
        payload["inheritance"] = inheritance
    if summary is not None:
        payload["summary"] = summary
        payload["rows"] = rows
    _write_json_atomic(output_path, payload)


def _run_full_domain_benchmark(args):
    cupti_version = _require_b200_and_cupti()
    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.output_json).resolve()
    try:
        output_path.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise RuntimeError("--output-json must be outside the source repository")
    identity = _full_domain_run_identity(repo_root)
    identity_sha256 = _canonical_json_sha256(identity)
    measurement = _full_domain_measurement_config(args, cupti_version)
    rows_root = _full_domain_rows_root(output_path)
    rows_root.mkdir(parents=True, exist_ok=True)
    inheritance = None
    inherited_identity_sha256_by_row = None
    if args.inheritance_json is not None:
        inheritance, inherited_identity_sha256_by_row = _load_full_domain_inheritance(
            Path(args.inheritance_json).resolve(),
            current_identity=identity,
            current_identity_sha256=identity_sha256,
            measurement=measurement,
            rows_root=rows_root,
            repeat_iters=args.repeat_iters,
        )
    rows, row_receipts = _load_full_domain_rows(
        rows_root,
        identity_sha256,
        args.repeat_iters,
        inherited_identity_sha256_by_row=inherited_identity_sha256_by_row,
    )
    if output_path.is_file():
        checkpoint = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema") != _FULL_DOMAIN_SCHEMA
            or checkpoint.get("identity") != identity
            or checkpoint.get("measurement") != measurement
            or checkpoint.get("inheritance") != inheritance
        ):
            raise RuntimeError("full-domain checkpoint identity or protocol drifted")
    if len(rows) == len(_FULL_DOMAIN_SHAPES):
        summary = _summarize_full_domain(rows)
        _write_full_domain_checkpoint(
            output_path,
            status="complete",
            identity=identity,
            measurement=measurement,
            rows=rows,
            row_receipts=row_receipts,
            inheritance=inheritance,
            summary=summary,
        )
        print(f"checkpoint is already complete: {output_path}", flush=True)
        return

    session = _full_domain_session_identity()
    started = time.monotonic()
    starting_row_count = len(rows)
    for shape_index in range(len(rows), len(_FULL_DOMAIN_SHAPES)):
        num_heads, num_rows = _FULL_DOMAIN_SHAPES[shape_index]
        measurements = []
        correctness_reference = None
        correctness = None
        for order_index, backend in enumerate(_ABBA_ORDER):
            cell, captured_reference, cell_correctness = _run_full_domain_cell(
                args,
                backend,
                num_heads,
                num_rows,
                capture_reference=order_index == 0,
                correctness_reference=(
                    correctness_reference if order_index == 1 else None
                ),
            )
            if captured_reference is not None:
                if correctness_reference is not None:
                    raise RuntimeError("correctness reference was captured twice")
                correctness_reference = captured_reference
            if cell_correctness is not None:
                correctness = cell_correctness
                del correctness_reference
                correctness_reference = None
            cell["order_index"] = order_index
            measurements.append(cell)
        if correctness_reference is not None or correctness is None:
            raise RuntimeError("full-domain correctness comparison was not completed")
        baseline_medians = [measurements[0]["median_ms"], measurements[3]["median_ms"]]
        candidate_medians = [measurements[1]["median_ms"], measurements[2]["median_ms"]]
        paired_speedups = [
            baseline_medians[0] / candidate_medians[0],
            baseline_medians[1] / candidate_medians[1],
        ]
        row = {
            "shape": f"h{num_heads}_rows{num_rows}",
            "num_heads": num_heads,
            "num_rows": num_rows,
            "baseline_ms": statistics.median(baseline_medians),
            "candidate_ms": statistics.median(candidate_medians),
            "paired_speedups": paired_speedups,
            "speedup": statistics.median(paired_speedups),
            "correctness": correctness,
            "measurements": measurements,
        }
        _validate_full_domain_row(row, shape_index, args.repeat_iters)
        row_path = _full_domain_row_path(rows_root, shape_index)
        if row_path.exists():
            raise RuntimeError(f"refusing to overwrite row receipt {row_path}")
        receipt = {
            "schema": _FULL_DOMAIN_ROW_SCHEMA,
            "identity_sha256": identity_sha256,
            "shape_index": shape_index,
            "session": session,
            "row": row,
        }
        _write_json_atomic(row_path, receipt)
        receipt_bytes = row_path.read_bytes()
        rows.append(row)
        row_receipts.append(
            {"path": row_path.name, "sha256": hashlib.sha256(receipt_bytes).hexdigest()}
        )
        _write_full_domain_checkpoint(
            output_path,
            status="in_progress",
            identity=identity,
            measurement=measurement,
            rows=rows,
            row_receipts=row_receipts,
            inheritance=inheritance,
        )
        new_rows = len(rows) - starting_row_count
        print(
            f"{row['shape']}: baseline={row['baseline_ms']:.6f} ms "
            f"candidate={row['candidate_ms']:.6f} ms speedup={row['speedup']:.6f}x "
            f"({len(rows)}/{len(_FULL_DOMAIN_SHAPES)})",
            flush=True,
        )
        # CUDA Graph timing can leave cyclic Python objects holding graph-owned
        # allocations until collection.  Release them after sealing each row so
        # the largest tail shapes do not inherit every earlier row's live graph.
        gc.collect()
        torch.cuda.empty_cache()
        if args.max_new_rows is not None and new_rows >= args.max_new_rows:
            break
        if (
            args.max_runtime_seconds is not None
            and time.monotonic() - started >= args.max_runtime_seconds
        ):
            break

    if len(rows) == len(_FULL_DOMAIN_SHAPES):
        summary = _summarize_full_domain(rows)
        _write_full_domain_checkpoint(
            output_path,
            status="complete",
            identity=identity,
            measurement=measurement,
            rows=rows,
            row_receipts=row_receipts,
            inheritance=inheritance,
            summary=summary,
        )
        print(json.dumps(summary, indent=2), flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json")
    parser.add_argument(
        "--shapes", choices=("official21", "full-domain"), default="official21"
    )
    parser.add_argument(
        "--baseline", choices=("public-fallback",), default="public-fallback"
    )
    parser.add_argument("--candidate", choices=("cake",), default="cake")
    parser.add_argument("--timing", choices=("cupti",), default="cupti")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--interleave", choices=("abba",), default="abba")
    parser.add_argument("--dry-run-iters", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--max-new-rows", type=int)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--inheritance-json")
    parser.add_argument("--worker-backend", choices=("baseline", "candidate"))
    parser.add_argument("--worker-heads", type=int)
    parser.add_argument("--worker-rows", type=int)
    parser.add_argument("--worker-json")
    args = parser.parse_args()
    if args.dry_run_iters < 1 or args.repeat_iters < 1:
        parser.error("iteration counts must be positive")
    if args.max_new_rows is not None and args.max_new_rows < 1:
        parser.error("--max-new-rows must be positive")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.worker_backend is not None:
        if None in (args.worker_heads, args.worker_rows, args.worker_json):
            parser.error("worker mode requires heads, rows, and output JSON")
    else:
        if args.output_json is None:
            parser.error("--output-json is required")
        if not args.cuda_graph or not args.cold_l2:
            parser.error("--cuda-graph and --cold-l2 are required")
        if args.shapes == "full-domain" and (
            args.dry_run_iters != 5 or args.repeat_iters != 16
        ):
            parser.error(
                "full-domain requires 5 warmups and 16 timed replays per ABBA cell"
            )
        if args.inheritance_json is not None and args.shapes != "full-domain":
            parser.error("--inheritance-json requires --shapes full-domain")
    return args


def main():
    args = _parse_args()
    if args.worker_backend is not None:
        _run_worker(args)
    elif args.shapes == "full-domain":
        _run_full_domain_benchmark(args)
    else:
        _run_paired_benchmark(args)


if __name__ == "__main__":
    main()
