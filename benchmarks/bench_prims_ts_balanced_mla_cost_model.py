#!/usr/bin/env python3
"""Measure and fit balanced PrimsTS MLA scheduler cost models.

The benchmark forces a deterministic set of piece-size targets through the
same replay-stable descriptor ABI used in production, times captured decode
graphs, fits the six scheduler coefficients independently for each kernel
family and input dtype, and validates the fitted model through the optimized
CUDA scheduler. Results are append-only JSONL and can be resumed safely.

After a completed measurement run, use ``--resume --refit-generation N`` to
reuse that exact measurement cohort with a new fitting implementation. The
generation must increase monotonically; the operation appends a new fit and
fresh paired validations without collecting forced-target curves again.

This is the only path allowed to bootstrap an uncalibrated device. It passes
an internal seed model solely to compile the balanced kernels and collect
forced-target measurements. Normal balanced APIs reject the device until the
resulting models and exact hardware identity are checked into
``_balanced_scheduler.py``.

The default timing protocol is 10 warmups followed by the minimum of five
trials of 200 CUDA-graph replays. Use ``--quick`` only for harness smoke tests;
it reduces workload and target coverage, not the per-target timing protocol.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
from typing import Any, Sequence

import numpy as np
import torch

from flashinfer.attention.prims_ts import BatchMLADecodePagedTSWrapper
from flashinfer.attention.prims_ts._balanced_plan import BalancedMLADecodePlan
from flashinfer.attention.prims_ts._balanced_scheduler import (
    B200_BF16_2CTA_COST,
    BalancedCostModel,
    balanced_cost_bucket,
)


KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
PAGE_SIZE = 32
FAMILY_HEADS = {"1cta": 16, "2cta": 128}
FAMILY_KERNEL = {
    "1cta": "throughput_latency_1cta",
    "2cta": "throughput_2cta",
}
DTYPES = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
FIT_METHOD = "tail_aware_selection_regret_v8_schedule_identity"
VALIDATION_METHOD = "paired_best_candidate_v9_schedule_identity"
BUCKET_FIT_METHOD = {
    "sparse_uniform_small": "tail_aware_selection_regret_v9_schedule_identity",
}
BUCKET_VALIDATION_METHOD = {
    "sparse_uniform_small": "paired_best_candidate_v10_schedule_identity",
}
COST_BUCKETS = (
    "single",
    "sparse_small",
    "sparse_uniform_small",
    "sparse",
    "dense_small",
    "dense_large",
)
ARTIFACT_SCHEMA_VERSION = 4
MEASUREMENT_METHOD_VERSION = "optimized-cuda-forced-schedule-graph-v3"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_SUFFIXES = frozenset((".cc", ".cu", ".cuh", ".h", ".py"))
_MEASUREMENT_SOURCE_SYMBOLS = frozenset(
    {
        "CalibrationCase",
        "Runtime",
        "attention_reference",
        "append_record",
        "calibration_cases",
        "capture_graph",
        "ceil_div",
        "check_output",
        "collect_measurements",
        "distribution_sample",
        "forced_target_split_counts",
        "make_runtime",
        "make_values",
        "measurement_key",
        "schedule_sha256",
        "schedule_snapshot",
        "schedule_statistics",
        "stage_forced_target",
        "target_candidates",
        "time_graph",
        "load_records",
    }
)
_MEASUREMENT_SOURCE_CONSTANTS = frozenset(
    {
        "DTYPES",
        "FAMILY_HEADS",
        "FAMILY_KERNEL",
        "KV_LORA_RANK",
        "MEASUREMENT_METHOD_VERSION",
        "PAGE_SIZE",
        "QK_HEAD_DIM",
        "QK_ROPE_HEAD_DIM",
    }
)
_FIT_SOURCE_SYMBOLS = frozenset(
    {
        "completed_fit_records",
        "completed_validation_keys",
        "fit_latency_cost_model",
        "fit_method",
        "main",
        "normalized_cost",
        "observed_schedule_latency",
        "predicted_schedule_latency",
        "select_fit_generation",
        "stage_recorded_schedule",
        "tune_selection_cost_model",
        "validation_method",
    }
)
_FIT_SOURCE_CONSTANTS = frozenset(
    {
        "BUCKET_FIT_METHOD",
        "BUCKET_VALIDATION_METHOD",
        "COST_BUCKETS",
        "FIT_METHOD",
        "VALIDATION_METHOD",
    }
)


def fit_method(cost_bucket_name: str) -> str:
    return BUCKET_FIT_METHOD.get(cost_bucket_name, FIT_METHOD)


def validation_method(cost_bucket_name: str) -> str:
    return BUCKET_VALIDATION_METHOD.get(cost_bucket_name, VALIDATION_METHOD)


@dataclass(frozen=True)
class CalibrationCase:
    name: str
    seq_lens: tuple[int, ...]
    max_kv_len: int


@dataclass
class Runtime:
    case: CalibrationCase
    family: str
    dtype_name: str
    wrapper: BatchMLADecodePagedTSWrapper
    query: torch.Tensor
    kv_cache: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    output: torch.Tensor
    reference: torch.Tensor
    graph: torch.cuda.CUDAGraph
    bmm1_scale: float
    bmm2_scale: float
    policy: dict[str, Any]

    def run(self) -> None:
        self.wrapper.run(
            self.query,
            self.kv_cache,
            self.block_tables,
            self.seq_lens,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            out=self.output,
            validate=False,
        )


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def cost_bucket(
    seq_lens: Sequence[int], num_partitions: int, k_tile_tokens: int = 128
) -> str:
    return balanced_cost_bucket(
        seq_lens,
        num_partitions=num_partitions,
        k_tile_tokens=k_tile_tokens,
    )


def parse_csv(value: str, choices: set[str]) -> tuple[str, ...]:
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(values) - choices)
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values from {sorted(choices)}, got {value!r}"
        )
    return values


def distribution_sample(name: str, batch_size: int, seed: int) -> tuple[int, ...]:
    if name == "rl":
        values = [100, 1000, 2000, 3000, 10000, 30000, 50000, 100000, 110000]
        weights = [7, 18, 200, 200, 400, 780, 600, 80, 65]
        stream_id = 1
    elif name == "prod":
        values = [4096, 8192, 16384, 32768, 65536, 131072]
        weights = [35, 20, 15, 12, 10, 8]
        stream_id = 2
    else:
        raise ValueError(name)
    rng = random.Random(seed + stream_id * 1_000_000 + batch_size * 1000)
    return tuple(
        ceil_div(value, 128) * 128
        for value in rng.choices(values, weights=weights, k=batch_size)
    )


def calibration_cases(seed: int, quick: bool) -> tuple[CalibrationCase, ...]:
    cases = (
        CalibrationCase("single_32k", (32768,), 32768),
        CalibrationCase("single_131k", (131072,), 131072),
        CalibrationCase("uniform_b8_8k", (8192,) * 8, 8192),
        CalibrationCase("ragged_b8_131k", (512,) * 7 + (131072,), 131072),
        CalibrationCase("ragged_b32_131k", (512,) * 31 + (131072,), 131072),
        CalibrationCase("ragged_b128_131k", (512,) * 127 + (131072,), 131072),
        CalibrationCase("uniform_b32_32k", (32768,) * 32, 32768),
        CalibrationCase("uniform_b128_8k", (8192,) * 128, 8192),
        *(
            CalibrationCase(
                f"{distribution}_b{batch_size}",
                distribution_sample(distribution, batch_size, seed),
                110080 if distribution == "rl" else 131072,
            )
            for distribution in ("rl", "prod")
            for batch_size in (32, 64, 128)
        ),
    )
    if quick:
        quick_names = {"single_32k", "uniform_b8_8k", "ragged_b8_131k", "rl_b32"}
        return tuple(case for case in cases if case.name in quick_names)
    return cases


def _hash_source_files(root: Path, source_files: Sequence[Path]) -> str:
    """Hash path names and contents so dirty source is identified exactly."""

    root = root.resolve()
    digest = hashlib.sha256()
    for source_path in sorted(path.resolve() for path in source_files):
        relative_path = source_path.relative_to(root)
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_python_symbols(
    source_path: Path,
    *,
    symbols: frozenset[str],
    constants: frozenset[str],
) -> str:
    """Hash selected top-level definitions without coupling fit-only edits."""

    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    selected: dict[str, str] = {}
    for node in module.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name if node.name in symbols else None
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [target.id for target in targets if isinstance(target, ast.Name)]
            name = next((value for value in names if value in constants), None)
        if name is not None:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise RuntimeError(f"cannot fingerprint {name} in {source_path}")
            selected[name] = segment
    missing = sorted((symbols | constants) - selected.keys())
    if missing:
        raise RuntimeError(
            f"measurement source definitions are missing from {source_path}: {missing}"
        )
    digest = hashlib.sha256()
    for name, segment in sorted(selected.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(segment.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def measurement_source_identity(root: Path = _REPO_ROOT) -> dict[str, Any]:
    """Fingerprint the benchmark, CUDA scheduler, and PrimsTS MLA sources."""

    roots = (
        root / "flashinfer/attention/prims_ts",
        root / "flashinfer/jit/prims_balanced_mla.py",
        root / "csrc/prims_balanced_mla_plan.cu",
        root / "csrc/prims_balanced_mla_scheduler_device.cu",
        root / "csrc/prims_balanced_mla_scheduler.cuh",
    )
    source_files = set()
    for source_root in roots:
        if source_root.is_dir():
            source_files.update(
                path
                for path in source_root.rglob("*")
                if path.is_file()
                and path.suffix in _SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            )
        elif source_root.is_file():
            source_files.add(source_root)
        else:
            raise RuntimeError(f"measurement source path is missing: {source_root}")
    benchmark_path = root / "benchmarks/bench_prims_ts_balanced_mla_cost_model.py"
    return {
        "sha256": _hash_source_files(root, tuple(source_files)),
        "file_count": len(source_files),
        "measurement_harness_sha256": _hash_python_symbols(
            benchmark_path,
            symbols=_MEASUREMENT_SOURCE_SYMBOLS,
            constants=_MEASUREMENT_SOURCE_CONSTANTS,
        ),
    }


def fit_source_identity(root: Path = _REPO_ROOT) -> dict[str, Any]:
    """Fingerprint fitting policy independently of reusable measurements."""

    benchmark_path = root / "benchmarks/bench_prims_ts_balanced_mla_cost_model.py"
    return {
        "sha256": _hash_python_symbols(
            benchmark_path,
            symbols=_FIT_SOURCE_SYMBOLS,
            constants=_FIT_SOURCE_CONSTANTS,
        ),
        "fit_methods": {bucket: fit_method(bucket) for bucket in COST_BUCKETS},
        "validation_methods": {
            bucket: validation_method(bucket) for bucket in COST_BUCKETS
        },
    }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cuda_driver_version() -> int | None:
    try:
        from cuda.bindings import driver as cuda_driver

        init_status = cuda_driver.cuInit(0)[0]
        if init_status != cuda_driver.CUresult.CUDA_SUCCESS:
            return None
        status, version = cuda_driver.cuDriverGetVersion()
        return int(version) if status == cuda_driver.CUresult.CUDA_SUCCESS else None
    except (ImportError, RuntimeError):
        return None


def measurement_hardware_identity(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "name": properties.name,
        "uuid": str(getattr(properties, "uuid", "unavailable")),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_memory": properties.total_memory,
        "multi_processor_count": properties.multi_processor_count,
        "l2_cache_size": getattr(properties, "L2_cache_size", None),
    }


def measurement_software_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_driver": _cuda_driver_version(),
        "numpy": np.__version__,
        "nvidia_cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
        "apache_tvm_ffi": _package_version("apache-tvm-ffi"),
    }


def build_resume_signature(
    *,
    families: Sequence[str],
    dtype_names: Sequence[str],
    seed: int,
    warmups: int,
    iterations: int,
    trials: int,
    quick: bool,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "measurement_method": MEASUREMENT_METHOD_VERSION,
        "workload": {
            "families": list(families),
            "dtypes": list(dtype_names),
            "seed": seed,
            "warmups": warmups,
            "iterations": iterations,
            "trials": trials,
            "quick": quick,
        },
        "hardware": measurement_hardware_identity(device),
        "software": measurement_software_identity(),
        "source": measurement_source_identity(),
    }


def validate_resume_signature(
    records: Sequence[dict[str, Any]], expected: dict[str, Any]
) -> bool:
    """Return whether metadata exists, rejecting mixed or stale cohorts."""

    metadata_records = [
        record for record in records if record.get("record") == "metadata"
    ]
    if not metadata_records:
        if records:
            raise ValueError(
                "resume artifact identity does not match: metadata is missing"
            )
        return False
    if len(metadata_records) != 1 or metadata_records[0].get("signature") != expected:
        raise ValueError("resume artifact identity does not match the current run")
    return True


def select_fit_generation(
    records: Sequence[dict[str, Any]],
    *,
    requested_generation: int | None,
    current_fit_identity: dict[str, Any],
) -> int:
    """Select a resumable fit generation or start an explicit new one."""

    fit_records = [
        record for record in records if record.get("record") in {"fit", "validation"}
    ]
    generations = [int(record.get("fit_generation", 0)) for record in fit_records]
    latest_generation = max(generations, default=0)
    if requested_generation is not None:
        if requested_generation <= latest_generation:
            raise ValueError(
                "--refit-generation must be newer than every existing fit generation "
                f"({latest_generation})"
            )
        return requested_generation

    active_records = [
        record
        for record, generation in zip(fit_records, generations, strict=True)
        if generation == latest_generation
    ]
    if any(
        record.get("fit_identity") != current_fit_identity for record in active_records
    ):
        raise ValueError(
            "fit identity changed; use --refit-generation "
            f"{latest_generation + 1} to preserve measurements and append a new fit"
        )
    return latest_generation


def completed_validation_keys(
    records: Sequence[dict[str, Any]], fit_generation: int
) -> set[tuple[str, str, str]]:
    """Return validations completed in exactly one fit generation."""

    return {
        validation_key(record["family"], record["dtype"], record["case"])
        for record in records
        if record.get("record") == "validation"
        and int(record.get("fit_generation", 0)) == fit_generation
        and record.get("method") == validation_method(str(record.get("cost_bucket")))
    }


def completed_fit_records(
    records: Sequence[dict[str, Any]], fit_generation: int
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return fits completed in exactly one fit generation."""

    completed = {}
    for record in records:
        if (
            record.get("record") == "fit"
            and int(record.get("fit_generation", 0)) == fit_generation
            and record.get("method") == fit_method(str(record.get("cost_bucket")))
        ):
            completed[(record["family"], record["dtype"], record["cost_bucket"])] = (
                record
            )
    return completed


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
    return records


def capture_graph(fn) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    warmups: int,
    iterations: int,
    trials: int,
) -> tuple[float, list[float]]:
    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(trials):
        begin.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / iterations)
    return min(samples), samples


def make_values(
    shape: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if dtype == torch.bfloat16:
        return torch.empty(shape, dtype=dtype, device=device).normal_(
            mean=0.0, std=0.2, generator=generator
        )
    # Integer source data keeps peak allocation bounded and gives every forced
    # schedule the same deterministic, nontrivial E4M3 workload.
    return torch.randint(
        -3,
        4,
        shape,
        dtype=torch.int8,
        device=device,
        generator=generator,
    ).to(dtype)


@torch.no_grad()
def attention_reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Sequence[int],
    *,
    bmm1_scale: float,
    bmm2_scale: float,
) -> torch.Tensor:
    """Compute the SQ1 paged MLA result directly in FP32.

    This deliberately shares neither the balanced scheduler nor a PrimsTS
    attention kernel with the measured path. The reference is computed once
    per calibration case; forced targets reuse it because BF16 attention is
    split-invariant and FP8 split-local probability quantization is covered by
    the benchmark's FP8 tolerance.
    """

    outputs = []
    for batch_idx, seq_len in enumerate(seq_lens):
        page_count = ceil_div(int(seq_len), PAGE_SIZE)
        page_ids = block_tables[batch_idx, :page_count].long()
        request_cache = (
            kv_cache[page_ids].reshape(-1, QK_HEAD_DIM)[: int(seq_len)].float()
        )
        request_query = query[batch_idx, 0].float()
        scores = (
            request_query[:, :KV_LORA_RANK] @ request_cache[:, :KV_LORA_RANK].T
            + request_query[:, KV_LORA_RANK:] @ request_cache[:, KV_LORA_RANK:].T
        )
        probabilities = torch.softmax(scores * bmm1_scale, dim=-1)
        outputs.append(probabilities @ request_cache[:, :KV_LORA_RANK] * bmm2_scale)
    return torch.stack(outputs)[:, None]


def make_runtime(
    case: CalibrationCase,
    family: str,
    dtype_name: str,
    device: torch.device,
    seed: int,
) -> Runtime:
    batch_size = len(case.seq_lens)
    num_heads = FAMILY_HEADS[family]
    qkv_dtype = DTYPES[dtype_name]
    pages_per_row = ceil_div(case.max_kv_len, PAGE_SIZE)
    num_pages = batch_size * pages_per_row
    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).reshape(
        batch_size, pages_per_row
    )
    seq_lens = torch.tensor(case.seq_lens, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device).manual_seed(
        seed + batch_size * 1009 + num_heads * 9176 + (1 if dtype_name == "fp8" else 0)
    )
    query = make_values(
        (batch_size, 1, num_heads, QK_HEAD_DIM),
        dtype=qkv_dtype,
        device=device,
        generator=generator,
    )
    kv_cache = make_values(
        (num_pages, PAGE_SIZE, QK_HEAD_DIM),
        dtype=qkv_dtype,
        device=device,
        generator=generator,
    )
    output = torch.empty(
        (batch_size, 1, num_heads, KV_LORA_RANK),
        dtype=torch.bfloat16,
        device=device,
    )
    q_scale, kv_scale = (0.0625, 0.125) if dtype_name == "fp8" else (1.0, 1.0)
    bmm1_scale = q_scale * kv_scale / math.sqrt(QK_HEAD_DIM)
    bmm2_scale = kv_scale
    wrapper = BatchMLADecodePagedTSWrapper()
    # Deliberately use the private calibration bootstrap. Public balanced
    # planning must fail closed until measurements from this script have been
    # reviewed and added to the device/model registry.
    wrapper._plan(
        device,
        batch_size,
        num_heads,
        KV_LORA_RANK,
        QK_ROPE_HEAD_DIM,
        PAGE_SIZE,
        case.max_kv_len,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=qkv_dtype,
        kv_data_type=qkv_dtype,
        o_data_type=torch.bfloat16,
        balanced=True,
        balanced_seq_lens=case.seq_lens,
        balanced_bootstrap_cost=B200_BF16_2CTA_COST,
        mask_type="causal",
    )
    assert wrapper._plan_state is not None
    policy = dict(wrapper._plan_state.policy)
    if policy["kernel"] != FAMILY_KERNEL[family]:
        raise RuntimeError(
            f"{family}/{dtype_name}/{case.name} selected {policy['kernel']}, "
            f"expected {FAMILY_KERNEL[family]}"
        )

    def run(*, validate: bool) -> None:
        wrapper.run(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            out=output,
            validate=validate,
        )

    # Perform all public validation before capture. The graph closure must not
    # trigger the validation path's device-to-host metadata reads.
    run(validate=True)
    torch.cuda.synchronize()
    reference = attention_reference(
        query,
        kv_cache,
        block_tables,
        case.seq_lens,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    if not torch.isfinite(reference).all() or not torch.count_nonzero(reference):
        raise RuntimeError(f"non-finite or trivial baseline for {family}/{dtype_name}")
    graph = capture_graph(lambda: run(validate=False))
    return Runtime(
        case=case,
        family=family,
        dtype_name=dtype_name,
        wrapper=wrapper,
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        output=output,
        reference=reference,
        graph=graph,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        policy=policy,
    )


def forced_target_split_counts(plan, seq_lens: Sequence[int], target: int):
    """Return forced piece counts, or ``None`` when packed ABI capacity fails."""

    if target <= 0:
        raise ValueError("target must be positive")
    k_tiles = [ceil_div(int(value), plan.k_tile_tokens) for value in seq_lens]
    split_counts = [
        min(ceil_div(value, target), 127, plan.num_partitions) if value else 0
        for value in k_tiles
    ]
    if sum(split_counts) > plan.descriptor_capacity:
        return None
    if sum(count for count in split_counts if count > 1) > 0x1000:
        return None
    return split_counts


def schedule_signature(plan) -> tuple[Any, ...]:
    """Snapshot every replay-visible schedule descriptor for exact comparison."""

    descriptor_count = plan.last_descriptor_count
    combine_count = plan.last_combine_request_count
    return (
        tuple(
            tuple(row)
            for row in plan.work_descriptors[:descriptor_count].cpu().tolist()
        ),
        tuple(plan.partition_offsets.cpu().tolist()),
        tuple(
            tuple(row)
            for row in plan.combine_descriptors[:combine_count].cpu().tolist()
        ),
        int(plan.num_combine_descriptors.cpu().item()),
    )


def schedule_snapshot(plan) -> dict[str, Any]:
    """Return a JSON-stable snapshot of every replay-visible descriptor."""

    descriptor_count = plan.last_descriptor_count
    combine_count = plan.last_combine_request_count
    return {
        "work_descriptors": plan.work_descriptors[:descriptor_count].cpu().tolist(),
        "partition_offsets": plan.partition_offsets.cpu().tolist(),
        "combine_descriptors": plan.combine_descriptors[:combine_count].cpu().tolist(),
        "num_combine_descriptors": int(plan.num_combine_descriptors.cpu().item()),
    }


def schedule_sha256(snapshot: dict[str, Any]) -> str:
    """Return a stable identity for one exact emitted schedule."""

    serialized = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def schedule_statistics(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Derive latency-fit features from an exact descriptor snapshot."""

    descriptors = snapshot["work_descriptors"]
    partition_offsets = snapshot["partition_offsets"]
    partition_features = []
    max_split_count = 0
    for begin, end in zip(partition_offsets, partition_offsets[1:], strict=False):
        tile_count = 0
        split_inverse_count = 0.0
        split_descriptor_count = 0
        for _, tile_begin, tile_end, split_info in descriptors[begin:end]:
            tile_count += tile_end - tile_begin
            if split_info & 1:
                split_count = (split_info >> 13) & 0x7F
                max_split_count = max(max_split_count, split_count)
                split_inverse_count += 1.0 / split_count
                split_descriptor_count += 1
        if end > begin:
            partition_features.append(
                [
                    tile_count,
                    end - begin,
                    split_inverse_count,
                    split_descriptor_count,
                ]
            )
    return {
        "descriptor_count": len(descriptors),
        "combine_request_count": int(snapshot["num_combine_descriptors"]),
        "max_split_count": max_split_count,
        "partition_features": partition_features,
    }


def stage_forced_target(
    plan, seq_lens: Sequence[int], target: int
) -> dict[str, Any] | None:
    """Stage a forced target through the production optimized CUDA scheduler."""

    split_counts = forced_target_split_counts(plan, seq_lens, target)
    if split_counts is None:
        return None
    device_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=plan.device)
    plan.schedule_device(
        device_seq_lens,
        scheduler="optimized",
        forced_target_piece_tiles=target,
    )
    snapshot = schedule_snapshot(plan)
    return {
        **schedule_statistics(snapshot),
        "max_split_count": max(split_counts, default=0),
        "placement_cost_model": asdict(plan.cost),
        "schedule_snapshot": snapshot,
        "schedule_sha256": schedule_sha256(snapshot),
    }


def stage_recorded_schedule(
    plan, seq_lens: Sequence[int], measurement: dict[str, Any]
) -> dict[str, Any]:
    """Restore and verify the exact schedule used by a measurement record."""

    try:
        cost = BalancedCostModel(**measurement["placement_cost_model"])
        expected_snapshot = measurement["schedule_snapshot"]
        expected_sha256 = str(measurement["schedule_sha256"])
        target = int(measurement["target_tiles"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("measurement is missing exact schedule identity") from error
    if schedule_sha256(expected_snapshot) != expected_sha256:
        raise RuntimeError("measurement schedule snapshot does not match its identity")
    plan.cost = cost
    stats = stage_forced_target(plan, seq_lens, target)
    if stats is None:
        raise RuntimeError("recorded schedule is no longer capacity-feasible")
    if (
        stats["schedule_sha256"] != expected_sha256
        or stats["schedule_snapshot"] != expected_snapshot
    ):
        raise RuntimeError("recorded schedule cannot be reproduced exactly")
    return stats


def target_candidates(seq_lens: Sequence[int], quick: bool) -> tuple[int, ...]:
    max_tiles = max(ceil_div(int(value), 128) for value in seq_lens)
    piece_counts = (
        (1, 4, 16, 64, 127)
        if quick
        else (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 127)
    )
    targets = {max(ceil_div(max_tiles, pieces), 1) for pieces in piece_counts}
    targets.add(max_tiles)
    return tuple(sorted(targets))


def check_output(runtime: Runtime) -> float:
    rtol, atol = (5e-2, 2e-3) if runtime.dtype_name == "fp8" else (2e-2, 1e-3)
    actual = runtime.output.float()
    torch.testing.assert_close(
        actual,
        runtime.reference.float(),
        rtol=rtol,
        atol=atol,
    )
    relative_l2 = torch.linalg.vector_norm(
        actual - runtime.reference
    ) / torch.linalg.vector_norm(runtime.reference)
    if float(relative_l2) > (0.1 if runtime.dtype_name == "fp8" else 0.02):
        raise AssertionError(f"relative L2 error is {float(relative_l2):.6f}")
    return float((actual - runtime.reference).abs().max())


def fit_latency_cost_model(records: list[dict[str, Any]], family: str, dtype_name: str):
    rows = [
        record
        for record in records
        if record.get("record") == "measurement"
        and record.get("status") == "ok"
        and record.get("family") == family
        and record.get("dtype") == dtype_name
    ]
    if not rows:
        raise RuntimeError(f"no measurements for {family}/{dtype_name}")
    case_names = sorted({str(row["case"]) for row in rows})
    case_index = {name: idx for idx, name in enumerate(case_names)}
    # [tile, fixed-piece, split-fixed/piece-count, split-piece,
    #  reducer-fixed, reducer-piece], all in microseconds while fitting.
    coefficients = np.array([3.046, 7.3, 0.0, 6.0, 2.8, 0.17])
    intercepts = np.zeros(len(case_names))
    for _ in range(40):
        matrix = []
        observed = []
        for row in rows:
            partition_features = np.asarray(row["partition_features"], dtype=float)
            producer_costs = partition_features @ coefficients[:4]
            critical = partition_features[int(np.argmax(producer_costs))]
            reducer = [
                1.0 if int(row["max_split_count"]) > 1 else 0.0,
                float(row["max_split_count"])
                if int(row["max_split_count"]) > 1
                else 0.0,
            ]
            vector = np.zeros(len(case_names) + 6)
            vector[case_index[str(row["case"])]] = 1.0
            vector[len(case_names) :] = np.concatenate((critical, reducer))
            matrix.append(vector)
            observed.append(float(row["latency_us"]))
        design = np.asarray(matrix)
        fitted, *_ = np.linalg.lstsq(design, np.asarray(observed), rcond=None)
        fitted = np.maximum(fitted, 0.0)
        new_intercepts = fitted[: len(case_names)]
        new_coefficients = fitted[len(case_names) :]
        if np.allclose(coefficients, new_coefficients, rtol=1e-5, atol=1e-7):
            intercepts = new_intercepts
            coefficients = new_coefficients
            break
        intercepts = new_intercepts
        coefficients = new_coefficients

    predictions = []
    for row in rows:
        features = np.asarray(row["partition_features"], dtype=float)
        producer = float(np.max(features @ coefficients[:4]))
        max_split = int(row["max_split_count"])
        reducer = (
            coefficients[4] + coefficients[5] * max_split if max_split > 1 else 0.0
        )
        predictions.append(
            intercepts[case_index[str(row["case"])]] + producer + reducer
        )
    observed = np.asarray([float(row["latency_us"]) for row in rows])
    rmse = float(np.sqrt(np.mean((np.asarray(predictions) - observed) ** 2)))
    # Cost models are integer nanoseconds. Preserve relative fitted scale and
    # keep a positive tile cost so planner break-even divisions stay defined.
    values_ns = [max(int(round(value * 1000.0)), 0) for value in coefficients]
    values_ns[0] = max(values_ns[0], 1)
    cost = BalancedCostModel(*values_ns)
    return (
        cost,
        rmse,
        {name: float(intercepts[index]) for name, index in case_index.items()},
    )


def normalized_cost(cost: BalancedCostModel) -> BalancedCostModel:
    """Normalize the scale-invariant planner model to 1000 tile-cost units."""

    scale = 1000.0 / max(cost.cost_per_k_tile, 1)
    values = [
        1000,
        round(cost.fixed_piece_cost * scale),
        round(cost.split_fixed_cost * scale),
        round(cost.split_piece_cost * scale),
        round(cost.reducer_fixed_cost * scale),
        round(cost.reducer_piece_cost * scale),
    ]
    return BalancedCostModel(*(max(int(value), 0) for value in values))


def predicted_schedule_latency(
    snapshot: dict[str, Any],
    *,
    latency_fit_cost: BalancedCostModel,
    case_intercept_us: float,
) -> float:
    """Predict latency from the candidate's actual placement features."""

    stats = schedule_statistics(snapshot)
    features = np.asarray(stats["partition_features"], dtype=float)
    coefficients = np.asarray(
        [
            latency_fit_cost.cost_per_k_tile,
            latency_fit_cost.fixed_piece_cost,
            latency_fit_cost.split_fixed_cost,
            latency_fit_cost.split_piece_cost,
        ],
        dtype=float,
    )
    producer_us = float(np.max(features @ coefficients)) / 1000.0
    max_split = int(stats["max_split_count"])
    reducer_us = (
        (
            latency_fit_cost.reducer_fixed_cost
            + latency_fit_cost.reducer_piece_cost * max_split
        )
        / 1000.0
        if max_split > 1
        else 0.0
    )
    return max(case_intercept_us + producer_us + reducer_us, 1e-9)


def observed_schedule_latency(
    rows: Sequence[dict[str, Any]], schedule_id: str
) -> float | None:
    """Return a direct latency only for the same emitted schedule."""

    matching = [
        float(row["latency_us"])
        for row in rows
        if row.get("schedule_sha256") == schedule_id
    ]
    return min(matching) if matching else None


def tune_selection_cost_model(
    records: list[dict[str, Any]],
    family: str,
    dtype_name: str,
    cases: Sequence[CalibrationCase],
    latency_fit_cost: BalancedCostModel,
    latency_fit_case_intercepts_us: dict[str, float],
    device: torch.device,
) -> tuple[BalancedCostModel, dict[str, Any]]:
    """Tune directly for measured planner selection regret.

    A latency least-squares fit is useful as a seed, but is not by itself a
    safe scheduler objective: fitted split overhead also enters hard target
    floors. This deterministic derivative-free search evaluates the optimized
    production CUDA scheduler. Exact schedule matches use direct timings; unseen
    placements use the latency fit evaluated on that candidate's own
    partition features. This deliberately does not treat target size as a
    complete schedule identity.
    """

    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if (
            record.get("record") == "measurement"
            and record.get("status") == "ok"
            and record.get("family") == family
            and record.get("dtype") == dtype_name
        ):
            rows_by_case.setdefault(str(record["case"]), []).append(record)
    # Sparse forced-target curves can hide a narrow cliff at the exact target
    # selected by the planner.  A paired validation measures that target and
    # the best sampled target back-to-back.  Re-anchor its ratio to the
    # original best-sample latency so later fit versions can use the direct
    # observation without importing cross-run clock or thermal drift.
    measurement_rows_by_case = {name: list(rows) for name, rows in rows_by_case.items()}
    for record in records:
        if (
            record.get("record") != "validation"
            or not str(record.get("method", "")).startswith("paired_best_candidate_")
            or record.get("family") != family
            or record.get("dtype") != dtype_name
        ):
            continue
        case_name = str(record["case"])
        selected_target = int(record["selected_target_tiles"])
        selected_schedule_id = str(record["selected_schedule_sha256"])
        anchor_schedule_id = str(record["best_measured_schedule_sha256"])
        anchor_rows = [
            row
            for row in measurement_rows_by_case.get(case_name, ())
            if row.get("schedule_sha256") == anchor_schedule_id
        ]
        if not anchor_rows:
            raise RuntimeError(
                "paired validation is missing its calibration anchor: "
                f"{family}/{dtype_name}/{case_name}/{anchor_schedule_id}"
            )
        anchor_latency = min(float(row["latency_us"]) for row in anchor_rows)
        rows_by_case.setdefault(case_name, []).append(
            {
                "target_tiles": selected_target,
                "latency_us": anchor_latency * float(record["selected_over_best"]),
                "num_partitions": int(anchor_rows[0]["num_partitions"]),
                "schedule_sha256": selected_schedule_id,
                "schedule_snapshot": record["selected_schedule_snapshot"],
            }
        )
    selected_cases = [case for case in cases if case.name in rows_by_case]
    if len(selected_cases) != len(cases):
        missing = sorted({case.name for case in cases} - rows_by_case.keys())
        raise RuntimeError(f"missing calibration curves: {missing}")

    best_latency = {
        name: min(float(row["latency_us"]) for row in rows)
        for name, rows in rows_by_case.items()
    }
    num_partitions = {
        name: int(rows[0]["num_partitions"]) for name, rows in rows_by_case.items()
    }
    cache: dict[
        tuple[int, ...],
        tuple[float, dict[str, float], dict[str, int], dict[str, str]],
    ] = {}
    selector_plans = {
        case.name: BalancedMLADecodePlan(
            batch_size=len(case.seq_lens),
            num_partitions=num_partitions[case.name],
            device=device,
            cost=latency_fit_cost,
            kernel_family=family,
            dtype_name=dtype_name,
            max_seq_len=case.max_kv_len,
        )
        for case in selected_cases
    }
    selector_seq_lens = {
        case.name: torch.tensor(case.seq_lens, dtype=torch.int32, device=device)
        for case in selected_cases
    }

    def evaluate(cost: BalancedCostModel):
        key = tuple(asdict(cost).values())
        if key in cache:
            return cache[key]
        ratios: dict[str, float] = {}
        targets: dict[str, int] = {}
        schedule_ids: dict[str, str] = {}
        for case in selected_cases:
            plan = selector_plans[case.name]
            plan.cost = cost
            plan.schedule_device(
                selector_seq_lens[case.name],
                scheduler="optimized",
            )
            targets[case.name] = plan.last_target_piece_tiles
            snapshot = schedule_snapshot(plan)
            schedule_id = schedule_sha256(snapshot)
            schedule_ids[case.name] = schedule_id
            latency = observed_schedule_latency(rows_by_case[case.name], schedule_id)
            if latency is None:
                latency = predicted_schedule_latency(
                    snapshot,
                    latency_fit_cost=latency_fit_cost,
                    case_intercept_us=latency_fit_case_intercepts_us[case.name],
                )
            ratios[case.name] = latency / best_latency[case.name]
        log_ratios = np.log(np.asarray(list(ratios.values())))
        # Optimize the geometric mean while explicitly penalizing the p90 and
        # worst case.  A model that wins on average but creates a large ragged
        # cliff is not suitable for a production scheduler.
        score = float(
            np.mean(log_ratios)
            + 0.5 * np.quantile(log_ratios, 0.9)
            + 0.5 * np.max(log_ratios)
        )
        cache[key] = score, ratios, targets, schedule_ids
        return cache[key]

    candidates: set[BalancedCostModel] = {
        normalized_cost(latency_fit_cost),
        normalized_cost(B200_BF16_2CTA_COST),
    }
    for fixed in (0, 1000, 2000, 3000, 5000, 8000, 16000, 32000, 64000):
        for reducer_fixed in (0, 1000, 4000):
            for reducer_piece in (0, 50, 150):
                candidates.add(
                    BalancedCostModel(
                        1000,
                        fixed,
                        0,
                        0,
                        reducer_fixed,
                        reducer_piece,
                    )
                )

    rng = random.Random(
        20260902
        + (1 if family == "2cta" else 0) * 100_000
        + (1 if dtype_name == "fp8" else 0) * 10_000
    )
    for _ in range(1200):
        candidates.add(
            BalancedCostModel(
                1000,
                0 if rng.random() < 0.1 else round(rng.uniform(0.0, 64.0) * 1000),
                0 if rng.random() < 0.7 else round(rng.uniform(0.0, 128.0) * 1000),
                0 if rng.random() < 0.5 else round(rng.uniform(0.0, 64.0) * 1000),
                0 if rng.random() < 0.2 else round(rng.uniform(0.0, 48.0) * 1000),
                0 if rng.random() < 0.2 else round(rng.uniform(0.0, 0.5) * 1000),
            )
        )

    ranked = sorted(
        ((evaluate(cost)[0], cost) for cost in candidates),
        key=lambda item: (item[0], tuple(asdict(item[1]).values())),
    )
    for _ in range(4):
        refinements: set[BalancedCostModel] = set()
        for _, base in ranked[:8]:
            values = list(asdict(base).values())
            for index in range(1, 6):
                for factor in (0.0, 0.5, 0.75, 1.25, 1.5, 2.0):
                    updated = values.copy()
                    updated[index] = round(updated[index] * factor)
                    if updated[index] == values[index] == 0 and factor > 0:
                        updated[index] = round(250 * factor)
                    refinements.add(BalancedCostModel(*updated))
        ranked = sorted(
            ranked + [(evaluate(cost)[0], cost) for cost in refinements],
            key=lambda item: (item[0], tuple(asdict(item[1]).values())),
        )

    score, cost = ranked[0]
    _, ratios, targets, schedule_ids = evaluate(cost)
    return cost, {
        "objective": score,
        "surrogate_ratio_geomean": statistics.geometric_mean(ratios.values()),
        "surrogate_ratio_p90": float(np.quantile(list(ratios.values()), 0.9)),
        "surrogate_ratio_max": max(ratios.values()),
        "surrogate_ratios": ratios,
        "selected_targets": targets,
        "selected_schedule_sha256": schedule_ids,
        "evaluated_models": len(cache),
    }


def measurement_key(family: str, dtype_name: str, case: str, target: int):
    return family, dtype_name, case, int(target)


def validation_key(family: str, dtype_name: str, case: str):
    return family, dtype_name, case


def collect_measurements(
    *,
    output_path: Path,
    records: list[dict[str, Any]],
    families: Sequence[str],
    dtype_names: Sequence[str],
    cases: Sequence[CalibrationCase],
    device: torch.device,
    seed: int,
    warmups: int,
    iterations: int,
    trials: int,
    quick: bool,
    refit_only: bool,
) -> list[dict[str, Any]]:
    """Collect or resume the immutable forced-schedule measurement cohort."""

    completed_measurements = {
        measurement_key(r["family"], r["dtype"], r["case"], r["target_tiles"])
        for r in records
        if r.get("record") == "measurement"
    }
    for family in families:
        for dtype_name in dtype_names:
            for case in cases:
                candidates = target_candidates(case.seq_lens, quick)
                missing = [
                    target
                    for target in candidates
                    if measurement_key(family, dtype_name, case.name, target)
                    not in completed_measurements
                ]
                if not missing:
                    continue
                if refit_only:
                    raise ValueError(
                        "--refit-generation requires a complete measurement cohort; "
                        f"missing {family}/{dtype_name}/{case.name} targets {missing}"
                    )
                print(
                    f"START {family}/{dtype_name}/{case.name}: {len(missing)} targets",
                    flush=True,
                )
                runtime = make_runtime(case, family, dtype_name, device, seed)
                assert runtime.wrapper._plan_state is not None
                plan = runtime.wrapper._plan_state.balanced_plan
                assert plan is not None
                for target in missing:
                    schedule_stats = stage_forced_target(plan, case.seq_lens, target)
                    if schedule_stats is None:
                        append_record(
                            output_path,
                            {
                                "record": "measurement",
                                "status": "capacity",
                                "family": family,
                                "dtype": dtype_name,
                                "case": case.name,
                                "batch_size": len(case.seq_lens),
                                "max_kv_len": case.max_kv_len,
                                "target_tiles": target,
                                "placement_cost_model": asdict(plan.cost),
                            },
                        )
                        completed_measurements.add(
                            measurement_key(family, dtype_name, case.name, target)
                        )
                        continue
                    runtime.graph.replay()
                    torch.cuda.synchronize()
                    max_abs_diff = check_output(runtime)
                    latency_us, trials_us = time_graph(
                        runtime.graph,
                        warmups=warmups,
                        iterations=iterations,
                        trials=trials,
                    )
                    record = {
                        "record": "measurement",
                        "status": "ok",
                        "family": family,
                        "kernel": runtime.policy["kernel"],
                        "dtype": dtype_name,
                        "case": case.name,
                        "batch_size": len(case.seq_lens),
                        "seq_mean": statistics.fmean(case.seq_lens),
                        "seq_max": max(case.seq_lens),
                        "max_kv_len": case.max_kv_len,
                        "num_partitions": plan.num_partitions,
                        "target_tiles": target,
                        "latency_us": latency_us,
                        "trials_us": trials_us,
                        "max_abs_diff": max_abs_diff,
                        **schedule_stats,
                    }
                    append_record(output_path, record)
                    records.append(record)
                    completed_measurements.add(
                        measurement_key(family, dtype_name, case.name, target)
                    )
                    print(
                        f"RESULT {family}/{dtype_name}/{case.name} "
                        f"target={target} pieces={schedule_stats['descriptor_count']} "
                        f"latency={latency_us:.3f} us",
                        flush=True,
                    )
                del runtime
                torch.cuda.empty_cache()
    return load_records(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("balanced_mla_cost_model.jsonl"),
        help="append-only JSONL artifact (refuses overwrite unless --resume)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--families", default="1cta,2cta")
    parser.add_argument("--dtypes", default="bf16,fp8")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refit-generation",
        type=int,
        help=(
            "reuse the exact measurement cohort, skip measurement collection, and "
            "append this newer fit plus fresh paired validations"
        ),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    families = parse_csv(args.families, set(FAMILY_HEADS))
    dtype_names = parse_csv(args.dtypes, set(DTYPES))
    if min(args.warmups, args.iterations, args.trials) <= 0:
        parser.error("--warmups, --iterations, and --trials must be positive")
    if args.refit_generation is not None and not args.resume:
        parser.error("--refit-generation requires --resume")
    if args.refit_generation is not None and args.refit_generation <= 0:
        parser.error("--refit-generation must be positive")

    output_path = args.output.resolve()
    output_exists = output_path.exists()
    if output_exists and not args.resume:
        parser.error(f"refusing to overwrite existing output: {output_path}")
    if args.refit_generation is not None and not output_exists:
        parser.error("--refit-generation requires an existing measurement artifact")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(output_path)

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("cost-model measurements require a CUDA device")
    torch.cuda.set_device(device)
    signature = build_resume_signature(
        families=families,
        dtype_names=dtype_names,
        seed=args.seed,
        warmups=args.warmups,
        iterations=args.iterations,
        trials=args.trials,
        quick=args.quick,
        device=device,
    )
    try:
        has_metadata = validate_resume_signature(records, signature)
    except ValueError as error:
        parser.error(f"--resume {error}")
    if not has_metadata:
        append_record(
            output_path,
            {
                "record": "metadata",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "signature": signature,
            },
        )
        records = load_records(output_path)

    current_fit_identity = fit_source_identity()
    try:
        fit_generation = select_fit_generation(
            records,
            requested_generation=args.refit_generation,
            current_fit_identity=current_fit_identity,
        )
    except ValueError as error:
        parser.error(str(error))

    cases = calibration_cases(args.seed, args.quick)
    try:
        records = collect_measurements(
            output_path=output_path,
            records=records,
            families=families,
            dtype_names=dtype_names,
            cases=cases,
            device=device,
            seed=args.seed,
            warmups=args.warmups,
            iterations=args.iterations,
            trials=args.trials,
            quick=args.quick,
            refit_only=args.refit_generation is not None,
        )
    except ValueError as error:
        parser.error(str(error))
    completed_validations = completed_validation_keys(records, fit_generation)
    completed_fits = completed_fit_records(records, fit_generation)
    for family in families:
        for dtype_name in dtype_names:
            latency_fit_cost, fit_rmse_us, intercepts = fit_latency_cost_model(
                records, family, dtype_name
            )
            case_partitions = {}
            for case in cases:
                partitions = {
                    int(record["num_partitions"])
                    for record in records
                    if record.get("record") == "measurement"
                    and record.get("status") == "ok"
                    and record.get("family") == family
                    and record.get("dtype") == dtype_name
                    and record.get("case") == case.name
                }
                if len(partitions) != 1:
                    raise RuntimeError(
                        "calibration case has inconsistent partition counts: "
                        f"{family}/{dtype_name}/{case.name}/{sorted(partitions)}"
                    )
                case_partitions[case.name] = partitions.pop()
            for bucket in COST_BUCKETS:
                bucket_cases = tuple(
                    case
                    for case in cases
                    if cost_bucket(
                        case.seq_lens,
                        case_partitions[case.name],
                    )
                    == bucket
                )
                if not bucket_cases:
                    continue
                fit_key = (family, dtype_name, bucket)
                fit_record = completed_fits.get(fit_key)
                if fit_record is None:
                    cost, selection_fit = tune_selection_cost_model(
                        records,
                        family,
                        dtype_name,
                        bucket_cases,
                        latency_fit_cost,
                        intercepts,
                        device,
                    )
                    fit_record = {
                        "record": "fit",
                        "fit_generation": fit_generation,
                        "fit_identity": current_fit_identity,
                        "family": family,
                        "dtype": dtype_name,
                        "cost_bucket": bucket,
                        "method": fit_method(bucket),
                        "cost_model": asdict(cost),
                        "latency_fit_cost_model": asdict(latency_fit_cost),
                        "latency_fit_rmse_us": fit_rmse_us,
                        "latency_fit_case_intercepts_us": intercepts,
                        **selection_fit,
                    }
                    append_record(output_path, fit_record)
                    completed_fits[fit_key] = fit_record
                    print(
                        f"FIT {family}/{dtype_name}/{bucket}: {asdict(cost)} "
                        f"surrogate_max={selection_fit['surrogate_ratio_max']:.4f} "
                        f"latency_rmse={fit_rmse_us:.3f} us",
                        flush=True,
                    )
                else:
                    cost = BalancedCostModel(**fit_record["cost_model"])
                    print(
                        f"RESUME FIT {family}/{dtype_name}/{bucket}: {asdict(cost)}",
                        flush=True,
                    )
                for case in bucket_cases:
                    key = validation_key(family, dtype_name, case.name)
                    if key in completed_validations:
                        continue
                    runtime = make_runtime(case, family, dtype_name, device, args.seed)
                    assert runtime.wrapper._plan_state is not None
                    plan = runtime.wrapper._plan_state.balanced_plan
                    assert plan is not None
                    runtime_bucket = cost_bucket(
                        case.seq_lens,
                        plan.num_partitions,
                        plan.k_tile_tokens,
                    )
                    if runtime_bucket != bucket:
                        raise RuntimeError(
                            "runtime/calibration cost-bucket mismatch: "
                            f"{runtime_bucket} != {bucket}"
                        )
                    plan.cost = cost
                    plan.schedule_device(runtime.seq_lens, scheduler="optimized")
                    selected_target = plan.last_target_piece_tiles
                    selected_snapshot = schedule_snapshot(plan)
                    selected_schedule_id = schedule_sha256(selected_snapshot)
                    selected_stats = schedule_statistics(selected_snapshot)
                    runtime.graph.replay()
                    torch.cuda.synchronize()
                    max_abs_diff = check_output(runtime)
                    selected_us, trials_us = time_graph(
                        runtime.graph,
                        warmups=args.warmups,
                        iterations=args.iterations,
                        trials=args.trials,
                    )
                    candidates = [
                        r
                        for r in records
                        if r.get("record") == "measurement"
                        and r.get("status") == "ok"
                        and r.get("family") == family
                        and r.get("dtype") == dtype_name
                        and r.get("case") == case.name
                    ]
                    best = min(candidates, key=lambda r: float(r["latency_us"]))
                    best_target = int(best["target_tiles"])
                    best_schedule_id = str(best["schedule_sha256"])
                    best_snapshot = best["schedule_snapshot"]
                    if schedule_sha256(best_snapshot) != best_schedule_id:
                        raise RuntimeError(
                            "recorded best schedule has an invalid identity: "
                            f"{family}/{dtype_name}/{case.name}/{best_target}"
                        )
                    schedules_match = (
                        selected_schedule_id == best_schedule_id
                        and selected_snapshot == best_snapshot
                    )
                    if schedules_match:
                        paired_best_us = selected_us
                        paired_best_trials_us = trials_us
                        paired_best_abs_diff = max_abs_diff
                    else:
                        stage_recorded_schedule(plan, case.seq_lens, best)
                        runtime.graph.replay()
                        torch.cuda.synchronize()
                        paired_best_abs_diff = check_output(runtime)
                        paired_best_us, paired_best_trials_us = time_graph(
                            runtime.graph,
                            warmups=args.warmups,
                            iterations=args.iterations,
                            trials=args.trials,
                        )
                    validation = {
                        "record": "validation",
                        "fit_generation": fit_generation,
                        "fit_identity": current_fit_identity,
                        "method": validation_method(bucket),
                        "family": family,
                        "dtype": dtype_name,
                        "cost_bucket": bucket,
                        "case": case.name,
                        "selected_target_tiles": selected_target,
                        "selected_descriptor_count": selected_stats["descriptor_count"],
                        "selected_placement_cost_model": asdict(cost),
                        "selected_schedule_snapshot": selected_snapshot,
                        "selected_schedule_sha256": selected_schedule_id,
                        "selected_latency_us": selected_us,
                        "selected_trials_us": trials_us,
                        "best_measured_target_tiles": best_target,
                        "best_measured_schedule_snapshot": best_snapshot,
                        "best_measured_schedule_sha256": best_schedule_id,
                        "best_measured_latency_us": float(best["latency_us"]),
                        "paired_best_latency_us": paired_best_us,
                        "paired_best_trials_us": paired_best_trials_us,
                        "paired_schedule_matches_selected": schedules_match,
                        "selected_over_best": selected_us / paired_best_us,
                        "max_abs_diff": max(max_abs_diff, paired_best_abs_diff),
                    }
                    append_record(output_path, validation)
                    print(
                        f"VALIDATE {family}/{dtype_name}/{bucket}/{case.name}: "
                        f"target={selected_target} "
                        f"regret={validation['selected_over_best']:.4f}",
                        flush=True,
                    )
                    del runtime
                    torch.cuda.empty_cache()

    print(f"DONE {output_path}", flush=True)


if __name__ == "__main__":
    main()
