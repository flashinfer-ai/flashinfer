#!/usr/bin/env python3
"""Benchmark graph-replayed paged attention on Qwen-like GQA serving shapes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Mapping

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import torch

from benchmarks.experimental.b12x.common import make_l2_flush_fn, resolve_l2_flush_bytes
from b12x.attention import paged
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention.paged.workspace import PagedAttentionWorkspace
from b12x.attention.paged.planner import create_paged_plan
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._scratch import build_paged_attention_binding
from b12x.attention.paged.traits import select_paged_forward_traits_from_plan
from b12x._lib import b12x_package_fingerprint


_REFERENCE_MINIMUM_COSINE = 0.999
_REFERENCE_MAXIMUM_RELATIVE_L2 = 0.02
_REFERENCE_RELATIVE_TOLERANCE = 0.05
_REFERENCE_ABSOLUTE_TOLERANCE = 0.02
_OUTPUT_GUARD_BYTES = 4 * 1024
_TENSOR_HASH_CHUNK_BYTES = 64 * 1024 * 1024
_RUNTIME_ENVIRONMENT_PREFIXES = (
    "B12X_",
    "CUTE_",
    "CUTLASS_",
    "CUDA_",
    "TORCH_",
    "PYTORCH_",
    "TRITON_",
    "NVCC_",
    "PTXAS_",
    "NCCL_",
)
_RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_PATH",
    "CUDA_CACHE_MAXSIZE",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_DISABLE_PTX_JIT",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_TF32_OVERRIDE",
)


@dataclass(frozen=True)
class _ReadOnlyInputSnapshot:
    clones: dict[str, torch.Tensor]
    tensor_sha256: dict[str, str]
    aggregate_sha256: str


@dataclass(frozen=True)
class _GuardedOutput:
    storage: torch.Tensor
    output: torch.Tensor
    prefix: torch.Tensor
    suffix: torch.Tensor
    prefix_value: float
    suffix_value: float

    def poison(self) -> None:
        self.prefix.fill_(self.prefix_value)
        self.output.fill_(float("nan"))
        self.suffix.fill_(self.suffix_value)

    def assert_fully_overwritten(self, *, backend: str) -> None:
        prefix_intact = bool(torch.all(self.prefix == self.prefix_value).item())
        suffix_intact = bool(torch.all(self.suffix == self.suffix_value).item())
        if not prefix_intact or not suffix_intact:
            raise AssertionError(
                f"{backend} output padding canary was modified: "
                f"prefix_intact={prefix_intact}, suffix_intact={suffix_intact}"
            )
        finite = bool(torch.isfinite(self.output).all().item())
        if not finite:
            remaining_poison = int(torch.isnan(self.output).count_nonzero().item())
            raise AssertionError(
                f"{backend} logical output was not fully overwritten with finite values: "
                f"remaining_poison={remaining_poison}"
            )


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")


def _capture_graph(fn: Callable[[], None], *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    graph._b12x_benchmark_call = fn
    return graph


def _bench_graph(graph, *, replays, l2_flush=None):
    from b12x.testing.benchmark import graph_samples
    return [value / 1000.0 for value in graph_samples(
        graph, replays=replays, l2_flush=l2_flush)["replay_us"]]


def _export_graph_trace(
    graph: torch.cuda.CUDAGraph,
    *,
    path: pathlib.Path,
    replays: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    print(f"CUDA graph trace: {path}")


_PAIRED_AB_BA_TIMING_METHOD = "paired-interleaved-ab-ba"






def _replay_graph_for_correctness(
    graph: torch.cuda.CUDAGraph,
    *,
    l2_flush=None,
) -> None:
    """Replay once in the benchmark cache state before collecting timings."""
    if l2_flush is not None:
        l2_flush()
        torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()


def _reference_gate(
    *,
    backend: str,
    output: torch.Tensor,
    reference: torch.Tensor,
    minimum_cosine: float = _REFERENCE_MINIMUM_COSINE,
    maximum_relative_l2: float = _REFERENCE_MAXIMUM_RELATIVE_L2,
    relative_tolerance: float = _REFERENCE_RELATIVE_TOLERANCE,
    absolute_tolerance: float = _REFERENCE_ABSOLUTE_TOLERANCE,
) -> tuple[float, float, float, int]:
    nonzero = int(torch.count_nonzero(output).item())
    finite = bool(torch.isfinite(output).all().item())
    max_abs = (output - reference).abs().max().item()
    relative_l2 = _relative_l2_error(output, reference)
    cosine = _cosine_similarity(output, reference)
    allclose = bool(
        torch.allclose(
            output.float(),
            reference.float(),
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
    )
    if (
        nonzero == 0
        or not finite
        or not math.isfinite(cosine)
        or cosine < minimum_cosine
        or not math.isfinite(relative_l2)
        or relative_l2 > maximum_relative_l2
        or not allclose
    ):
        row_diagnostics = ""
        if output.shape == reference.shape and output.ndim >= 2:
            output_rows = output.float().reshape(-1, output.shape[-1])
            reference_rows = reference.float().reshape(-1, reference.shape[-1])
            row_diff_norm = torch.linalg.vector_norm(
                output_rows - reference_rows, dim=-1
            )
            row_ref_norm = torch.linalg.vector_norm(reference_rows, dim=-1)
            row_relative_l2 = row_diff_norm / row_ref_norm.clamp_min(1.0e-12)
            worst_rows = torch.topk(
                row_relative_l2, k=min(24, row_relative_l2.numel())
            )
            row_diagnostics = (
                ", worst_row_rel_l2="
                + repr(
                    [
                        (int(row), float(error))
                        for error, row in zip(
                            worst_rows.values.tolist(),
                            worst_rows.indices.tolist(),
                            strict=True,
                        )
                    ]
                )
            )
        raise AssertionError(
            f"{backend} paged attention failed the Torch reference gate: "
            f"nonzero={nonzero}, finite={finite}, cos={cosine:.8f}, "
            f"minimum_cosine={minimum_cosine:.8f}, rel_l2={relative_l2:.8f}, "
            f"maximum_relative_l2={maximum_relative_l2:.8f}, "
            f"allclose={allclose} (rtol={relative_tolerance}, "
            f"atol={absolute_tolerance}){row_diagnostics}"
        )
    return max_abs, relative_l2, cosine, nonzero


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=pathlib.Path(__file__).resolve().parents[3],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "missing"


def _append_jsonl(path: pathlib.Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output:
        json.dump(payload, output, sort_keys=True, allow_nan=False)
        output.write("\n")


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_graph_replay_policy_metadata(
) -> dict[str, object]:
    """Describe the persistent device metadata consumed by the b12x call."""
    backends: dict[str, object] = {
        "b12x": {
            "strict_live_length_graph_safe": True,
            "live_length_dependent_host_planning": False,
            "runtime_metadata_binding": (
                "stable-address device buffers bound by reference; no page-table, "
                "cache-length, or cu-seqlens D2D copies are captured"
            ),
            "live_length_staging_before_timing": [
                "fill persistent device cache_seqlens input",
            ],
            "planning_timed": [
                "captured device replay-metadata updater selects schedule, chunking, and valid work",
            ],
            "planning_excluded_from_timing": [],
            "timed_graph_work": [
                "device replay-metadata updater",
                "paged-attention execution",
                "captured split-kv merge when selected",
            ],
        }
    }
    payload: dict[str, object] = {
        "schema": "b12x-decode-graph-replay-policy-v2",
        "measurement_scope": "prepared-call-stream-gated-events",
        "backends": backends,

    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def _decode_graph_timing_metadata(
    base_timing: Mapping[str, object] | None,
) -> dict[str, object]:
    timing = (
        dict(base_timing)
        if base_timing is not None
        else {
            "method": "stream_gated_events_v1",
            "sample_index": "sample-index",
        }
    )
    timing["replay_policy"] = _decode_graph_replay_policy_metadata(

    )
    return timing


def _runtime_environment_provenance() -> dict[str, object]:
    """Return the complete benchmark-control environment without broad NVIDIA_ scans."""
    set_variables = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(_RUNTIME_ENVIRONMENT_PREFIXES)
    }
    explicit_controls = {
        name: (
            {"status": "set", "value": os.environ[name]}
            if name in os.environ
            else {"status": "missing"}
        )
        for name in _RUNTIME_ENVIRONMENT_EXPLICIT_CONTROLS
    }
    payload: dict[str, object] = {
        "schema": "b12x-runtime-environment-v1",
        "complete_set_variable_prefixes": list(_RUNTIME_ENVIRONMENT_PREFIXES),
        "set_variables": set_variables,
        "explicit_controls": explicit_controls,
        "nvidia_enumeration": {
            "policy": "explicit-only",
            "included": [
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "NVIDIA_TF32_OVERRIDE",
            ],
            "reason": "avoid collecting unrelated NVIDIA_ variables that may contain secrets",
        },
    }
    payload["sha256"] = _json_sha256(payload)
    return payload


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata plus logical content in bounded host-side chunks."""
    metadata = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    byte_view = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    for offset in range(0, byte_view.numel(), _TENSOR_HASH_CHUNK_BYTES):
        chunk = byte_view.narrow(
            0,
            offset,
            min(_TENSOR_HASH_CHUNK_BYTES, byte_view.numel() - offset),
        ).cpu()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _extend_read_only_input_snapshot(
    base: _ReadOnlyInputSnapshot | None = None,
    **inputs: torch.Tensor | None,
) -> _ReadOnlyInputSnapshot:
    clones = {} if base is None else dict(base.clones)
    tensor_hashes = {} if base is None else dict(base.tensor_sha256)
    duplicate_names = set(clones).intersection(inputs)
    if duplicate_names:
        raise ValueError(f"read-only input snapshot names must be unique: {sorted(duplicate_names)}")
    for name, tensor in inputs.items():
        if tensor is None:
            continue
        clone = tensor.detach().clone()
        actual_hash = _tensor_content_sha256(tensor)
        clone_hash = _tensor_content_sha256(clone)
        if clone_hash != actual_hash:
            raise AssertionError(f"failed to clone read-only input {name} exactly")
        clones[name] = clone
        tensor_hashes[name] = actual_hash
    aggregate = _json_sha256(
        {"schema": "b12x-read-only-inputs-v1", "tensor_sha256": tensor_hashes}
    )
    return _ReadOnlyInputSnapshot(
        clones=clones,
        tensor_sha256=tensor_hashes,
        aggregate_sha256=aggregate,
    )


def _assert_read_only_inputs_unchanged(
    snapshot: _ReadOnlyInputSnapshot,
    actual_inputs: Mapping[str, torch.Tensor],
) -> None:
    missing = set(snapshot.tensor_sha256) - set(actual_inputs)
    unexpected = set(actual_inputs) - set(snapshot.tensor_sha256)
    if missing or unexpected:
        raise AssertionError(
            "read-only input set changed: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    mutated: list[str] = []
    for name, expected_hash in snapshot.tensor_sha256.items():
        if _tensor_content_sha256(actual_inputs[name]) != expected_hash:
            mutated.append(name)
    if mutated:
        raise AssertionError(f"read-only paged-attention inputs were mutated: {mutated}")


def _read_only_input_provenance(snapshot: _ReadOnlyInputSnapshot) -> dict[str, object]:
    return {
        "schema": "b12x-read-only-inputs-v1",
        "tensor_sha256": dict(snapshot.tensor_sha256),
        "aggregate_sha256": snapshot.aggregate_sha256,
    }


def _allocate_guarded_output(like: torch.Tensor) -> _GuardedOutput:
    guard_elements = max(
        (_OUTPUT_GUARD_BYTES + like.element_size() - 1) // like.element_size(),
        1,
    )
    storage = torch.empty(
        guard_elements + like.numel() + guard_elements,
        dtype=like.dtype,
        device=like.device,
    )
    prefix = storage[:guard_elements]
    output = storage[guard_elements : guard_elements + like.numel()].view_as(like)
    suffix = storage[guard_elements + like.numel() :]
    guarded = _GuardedOutput(
        storage=storage,
        output=output,
        prefix=prefix,
        suffix=suffix,
        prefix_value=91.0,
        suffix_value=-73.0,
    )
    guarded.poison()
    return guarded


def _benchmark_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, value in vars(args).items():
        if key == "raw_samples_jsonl":
            continue
        config[key] = str(value) if isinstance(value, pathlib.Path) else value
    return config


def _case_contract(
    fields: dict[str, object],
    *,
    input_seed: int,
    input_generator: str,
) -> dict[str, object]:
    input_generation = {
        "generator": input_generator,
        "seed": input_seed,
        "case": fields,
    }
    case = {
        **fields,
        "input_seed": input_seed,
        "input_generation_sha256": _json_sha256(input_generation),
    }
    case["case_contract_sha256"] = _json_sha256(case)
    return case


def _expected_case_contract(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "legacy-matrix":
        identity_fields = [
            "mode",
            "phase",
            "batch",
            "q_seqlen",
            "cache_seqlen",
            "window_left",
        ]
        expected = [
            {
                "mode": args.mode,
                "phase": case.phase,
                "batch": case.batch,
                "q_seqlen": case.q_seqlen,
                "cache_seqlen": case.cache_seqlen,
                "window_left": args.window_left,
            }
            for case in _build_shape_cases(
                batch=args.batch,
                q_seqlens=_parse_csv_ints(args.q_seqlens),
                cache_seqlens=_parse_csv_ints(args.cache_seqlens),
            )
        ]
    else:
        identity_fields = [
            "mode",
            "batch",
            "context_tokens",
            "effective_cache_tokens",
        ]
        expected = [
            {
                "mode": args.mode,
                "batch": case.batch,
                "context_tokens": case.context_tokens,
                "effective_cache_tokens": case.effective_cache_tokens,
            }
            for case in _build_decode_replay_cases(
                batch_buckets=_parse_csv_ints(args.batch_buckets),
                context_tokens=_parse_csv_ints(args.decode_contexts),
            )
        ]
    contract: dict[str, object] = {
        "identity_fields": identity_fields,
        "expected": expected,
    }
    contract["sha256"] = _json_sha256(contract)
    return contract


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _record_samples(
    path: pathlib.Path | None,
    *,
    backend: str,
    case: dict[str, object],
    samples_ms: list[float],
    correctness: dict[str, object] | None = None,
    timing: dict[str, object] | None = None,
) -> None:
    samples_us = [sample * 1000.0 for sample in samples_ms]
    _append_jsonl(
        path,
        {
            "type": "graph-replay-samples",
            "backend": backend,
            "case": case,
            "unit": "us",
            "samples": samples_us,
            "count": len(samples_us),
            "mean": statistics.fmean(samples_us),
            "median": statistics.median(samples_us),
            "p95": _percentile(samples_us, 0.95),
            "sample_stdev": statistics.stdev(samples_us)
            if len(samples_us) > 1
            else 0.0,
            "minimum": min(samples_us),
            "maximum": max(samples_us),
            "correctness": correctness,
            "timing": timing,
        },
    )


def _initialize_raw_sample_log(
    path: pathlib.Path | None,
    *,
    args: argparse.Namespace,
    argv: list[str],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    repo = pathlib.Path(__file__).resolve().parents[3]
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logical_device = torch.cuda.current_device()
    visible = [item.strip() for item in visible_devices.split(",") if item.strip()]
    physical_device = visible[logical_device] if logical_device < len(visible) else None
    gpu_properties = torch.cuda.get_device_properties(logical_device)
    benchmark_path = pathlib.Path(__file__).resolve()
    benchmark_dependencies = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (benchmark_path, repo / "benchmarks" / "common.py")
    }
    benchmark_config = _benchmark_config(args)
    expected_case_contract = _expected_case_contract(args)
    runtime_environment = _runtime_environment_provenance()
    _append_jsonl(
        path,
        {
            "type": "provenance",
            "command": shlex.join(
                [sys.executable, str(pathlib.Path(__file__).resolve()), *argv]
            ),
            "argv": argv,
            "worktree": str(repo),
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "dirty_paths": _git_value("status", "--short").splitlines(),
            "b12x_package_fingerprint": b12x_package_fingerprint(),
            "benchmark_sha256": benchmark_dependencies[
                str(benchmark_path.relative_to(repo))
            ],
            "benchmark_dependencies_sha256": benchmark_dependencies,
            "benchmark_config": benchmark_config,
            "benchmark_config_sha256": _json_sha256(benchmark_config),
            "benchmark_case_contract": expected_case_contract,
            "runtime_environment": runtime_environment,
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cutlass": {
                package: _package_version(package)
                for package in (
                    "nvidia-cutlass-dsl",
                    "nvidia-cutlass-dsl-libs-base",
                    "nvidia-cutlass-dsl-libs-core",
                    "nvidia-cutlass-dsl-libs-cu12",
                    "nvidia-cutlass-dsl-libs-cu13",
                )
            },
            "runtime_packages": {
                package: _package_version(package)
                for package in ("cuda-python", "cuda-bindings")
            },
            "gpu": {
                "cuda_visible_devices": visible_devices,
                "logical_index": logical_device,
                "physical_index": physical_device,
                "name": torch.cuda.get_device_name(logical_device),
                "uuid": str(gpu_properties.uuid),
                "capability": list(torch.cuda.get_device_capability(logical_device)),
                "l2_cache_bytes": int(gpu_properties.L2_cache_size),
            },
            "serving_mode": {
                "cuda_graph_replay": True,
                "stable_allocations": True,
                "fixed_workspace_capacity": True,
                "decode_graph_replay_policy": (
                    _decode_graph_replay_policy_metadata(

                    )
                    if args.mode == "decode-graph-buckets"
                    else {"status": "not-applicable-to-legacy-matrix"}
                ),
                "requested_kv_cache_layout": _kv_cache_layout_name(
                    combined_kv_cache=args.combined_kv_cache
                ),
                "warmup": args.warmup,
                "replays": args.replays,
                "l2_flush": args.flush_l2,
                "l2_flush_bytes": resolve_l2_flush_bytes(args.l2_flush_bytes),
                "correctness": "torch-reference" if args.check else "not-requested",
            },
        },
    )
    print(f"raw replay samples: {path}")


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported dtype {name}")


def _resolve_kv_dtype(name: str, q_dtype: torch.dtype) -> torch.dtype:
    if name == "same":
        return q_dtype
    return _dtype_from_name(name)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _relative_l2_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.to(torch.float32)
    b_f = b.to(torch.float32)
    diff_norm = (a_f - b_f).norm().item()
    ref_norm = max(b_f.norm().item(), 1e-12)
    return diff_norm / ref_norm


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


BENCHMARK_PROFILES: dict[str, dict[str, object]] = {
    "qwen3.8-27b": {
        "mode": "decode-graph-buckets",
        "batch": 8,
        "batch_buckets": "1,2,4,8,12,16",
        "decode_contexts": "128,16384,32768,65536,131072",
        "capture_context": 0,
        "q_seqlens": "1",
        "cache_seqlens": "64,512,2048,8192",
        "page_size": 64,
        "q_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "dtype": "bf16",
        "kv_dtype": "same",
    },
    "qwen-gqa": {
        "mode": "decode-graph-buckets",
        "batch": 8,
        "batch_buckets": "1,2,4,8,12,16",
        "decode_contexts": "128,16384,32768,65536,131072",
        "capture_context": 0,
        "q_seqlens": "1",
        "cache_seqlens": "64,512,2048,8192",
        "page_size": 64,
        "q_heads": 8,
        "kv_heads": 1,
        "head_dim": 256,
        "dtype": "bf16",
        "kv_dtype": "same",
    },
    "minimax-m2.7": {
        "mode": "decode-graph-buckets",
        "batch": 8,
        "batch_buckets": "1,2,4,8,12,16",
        "decode_contexts": "128,16384,32768,65536,131072",
        "capture_context": 0,
        "q_seqlens": "1",
        "cache_seqlens": "64,512,2048,8192",
        "page_size": 64,
        "q_heads": 24,
        "kv_heads": 4,
        "head_dim": 128,
        "dtype": "bf16",
        "kv_dtype": "same",
    },
}
BENCHMARK_PROFILE_ALIASES = {
    "minimax-m2": "minimax-m2.7",
    "mimimax-m2.7": "minimax-m2.7",
}


def _canonical_profile_name(name: str) -> str:
    return BENCHMARK_PROFILE_ALIASES.get(name, name)


def _profile_choices() -> list[str]:
    return sorted((*BENCHMARK_PROFILES.keys(), *BENCHMARK_PROFILE_ALIASES.keys()))


def _preparse_profile(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", choices=_profile_choices(), default="qwen-gqa")
    args, _ = parser.parse_known_args(argv)
    return _canonical_profile_name(args.profile)


def _gqa_group_size(*, q_heads: int, kv_heads: int) -> int:
    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("q_heads and kv_heads must be positive")
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    return q_heads // kv_heads






@dataclass(frozen=True)
class ShapeCase:
    phase: str
    batch: int
    q_seqlen: int
    cache_seqlen: int

    @property
    def total_q(self) -> int:
        return self.batch * self.q_seqlen


@dataclass(frozen=True)
class CaseMetrics:
    backend: str
    mean_us: float


@dataclass(frozen=True)
class BackendCapture:
    graph: torch.cuda.CUDAGraph
    workspace: PagedAttentionWorkspace
    output: torch.Tensor
    guarded_output: _GuardedOutput
    plan_desc: str
    read_only_snapshot: _ReadOnlyInputSnapshot | None
    read_only_inputs: dict[str, torch.Tensor] | None




def _build_shape_cases(
    *,
    batch: int,
    q_seqlens: list[int],
    cache_seqlens: list[int],
) -> list[ShapeCase]:
    cases: list[ShapeCase] = []
    for q_seqlen in q_seqlens:
        phase = "decode" if q_seqlen == 1 else "extend"
        for cache_seqlen in cache_seqlens:
            if phase == "extend" and q_seqlen > cache_seqlen:
                continue
            cases.append(
                ShapeCase(
                    phase=phase,
                    batch=batch,
                    q_seqlen=q_seqlen,
                    cache_seqlen=cache_seqlen,
                )
            )
    if not cases:
        raise ValueError(
            "no valid causal attention shapes: prefill q_seqlen must not exceed cache_seqlen"
        )
    return cases


def _make_uniform_page_metadata(
    *,
    batch: int,
    cache_seqlen: int,
    page_size: int,
    num_pages: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = "cuda"
    pages_per_request = (cache_seqlen + page_size - 1) // page_size
    total_pages_needed = batch * pages_per_request
    if num_pages < total_pages_needed:
        raise ValueError(
            f"num_pages={num_pages} is too small for batch={batch}, cache_seqlen={cache_seqlen}, "
            f"page_size={page_size}; need at least {total_pages_needed}"
        )
    page_table = torch.zeros(batch, pages_per_request, dtype=torch.int32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    page_order = torch.randperm(num_pages, generator=generator, device=device)
    for request_idx in range(batch):
        start = request_idx * pages_per_request
        page_ids = page_order[start : start + pages_per_request].to(torch.int32)
        page_table[request_idx] = page_ids
    cache_seqlens = torch.full((batch,), cache_seqlen, dtype=torch.int32, device=device)
    return page_table, cache_seqlens


def _kv_cache_layout_name(*, combined_kv_cache: bool) -> str:
    return (
        "combined-pages-2-nhd-strided-views"
        if combined_kv_cache
        else "separate-contiguous-nhd"
    )


def _kv_cache_layout_contract(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> dict[str, object]:
    same_storage = (
        k_cache.untyped_storage().data_ptr()
        == v_cache.untyped_storage().data_ptr()
    )
    return {
        "schema": "b12x-paged-kv-layout-v1",
        "kind": _kv_cache_layout_name(combined_kv_cache=same_storage),
        "shared_storage": same_storage,
        "k_stride": list(k_cache.stride()),
        "v_stride": list(v_cache.stride()),
        "k_storage_offset_elements": int(k_cache.storage_offset()),
        "v_storage_offset_elements": int(v_cache.storage_offset()),
    }


def _make_uniform_paged_inputs(
    *,
    batch: int,
    q_seqlen: int,
    cache_seqlen: int,
    capture_cache_seqlen: int | None,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
    combined_kv_cache: bool = False,
    num_cache_pages: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(seed)
    device = "cuda"
    total_q = batch * q_seqlen
    q = torch.randn(total_q, q_heads, head_dim, device=device, dtype=dtype) / 4
    capture_cache_seqlen = max(cache_seqlen, capture_cache_seqlen or cache_seqlen)
    capture_pages_per_request = (capture_cache_seqlen + page_size - 1) // page_size
    num_pages = batch * capture_pages_per_request
    if num_cache_pages is not None:
        if num_cache_pages < num_pages:
            raise ValueError("declared KV pool cannot hold the benchmark sequences")
        num_pages = num_cache_pages
    if combined_kv_cache:
        combined_cache = (
            torch.randn(
                num_pages,
                2,
                page_size,
                kv_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            / 4
        )
        k_cache = combined_cache[:, 0]
        v_cache = combined_cache[:, 1]
    else:
        k_cache = (
            torch.randn(
                num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype
            )
            / 4
        )
        v_cache = (
            torch.randn(
                num_pages, page_size, kv_heads, head_dim, device=device, dtype=dtype
            )
            / 4
        )
    page_table, cache_seqlens = _make_uniform_page_metadata(
        batch=batch,
        cache_seqlen=cache_seqlen,
        page_size=page_size,
        num_pages=num_pages,
        seed=seed,
    )
    capture_page_table, capture_cache_seqlens = _make_uniform_page_metadata(
        batch=batch,
        cache_seqlen=capture_cache_seqlen,
        page_size=page_size,
        num_pages=num_pages,
        seed=seed + 10_000,
    )
    cu_seqlens_q = torch.arange(
        0, total_q + 1, q_seqlen, dtype=torch.int32, device=device
    )
    return (
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        capture_page_table,
        capture_cache_seqlens,
        cu_seqlens_q,
    )


def _quantize_paged_kv_cache_global_e4m3(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    batch: int,
    kv_heads: int,
    combined_kv_cache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    k_scale = k_cache.abs().amax().to(torch.float32) / finfo.max
    v_scale = v_cache.abs().amax().to(torch.float32) / finfo.max
    if float(k_scale.item()) == 0.0:
        k_scale = torch.ones_like(k_scale)
    if float(v_scale.item()) == 0.0:
        v_scale = torch.ones_like(v_scale)
    k_fp8 = (
        (k_cache.to(torch.float32) / k_scale)
        .clamp(min=finfo.min, max=finfo.max)
        .to(torch.float8_e4m3fn)
    )
    v_fp8 = (
        (v_cache.to(torch.float32) / v_scale)
        .clamp(min=finfo.min, max=finfo.max)
        .to(torch.float8_e4m3fn)
    )
    k_descale = torch.full(
        (batch, kv_heads),
        float(k_scale.item()),
        dtype=torch.float32,
        device=k_cache.device,
    )
    v_descale = torch.full(
        (batch, kv_heads),
        float(v_scale.item()),
        dtype=torch.float32,
        device=v_cache.device,
    )
    if combined_kv_cache:
        combined_cache = torch.stack((k_fp8, v_fp8), dim=1)
        k_fp8 = combined_cache[:, 0]
        v_fp8 = combined_cache[:, 1]
    else:
        k_fp8 = k_fp8.contiguous()
        v_fp8 = v_fp8.contiguous()
    return (
        k_fp8,
        v_fp8,
        k_descale,
        v_descale,
        float(k_scale.item()),
        float(v_scale.item()),
    )




def _format_plan_desc(*, kv_chunk_size: int, split_kv: bool) -> str:
    desc = f"chunk={int(kv_chunk_size)}"
    return f"{desc},split" if split_kv else f"{desc},nosplit"


def _format_decode_graph_replay_plan_desc(
    workspace: object,
) -> str:
    plan = workspace.plan
    assert workspace.request_indices is not None
    split_desc = "split" if plan.split_kv else "nosplit"
    return (
        f"chunk=device-lut,grid={int(workspace.request_indices.numel())},"
        f"{split_desc}"
    )


def _paged_forward_traits_contract(plan: object) -> dict[str, object]:
    traits = select_paged_forward_traits_from_plan(plan)
    return {
        "schema": "b12x-paged-forward-traits-v1",
        "cta_tile_q": int(traits.cta_tile_q),
        "cta_tile_kv": int(traits.cta_tile_kv),
        "num_mma_q": int(traits.num_mma_q),
        "num_mma_kv": int(traits.num_mma_kv),
        "num_threads": int(traits.num_threads),
        "launch_smem_bytes": int(traits.launch_smem_bytes),
        "num_ctas_per_sm": int(traits.num_ctas_per_sm),
    }


def _observe_decode_graph_replay_topology(
    workspace: object,
    *,
    batch: int,
) -> dict[str, object]:
    """Read the schedule produced by the captured device updater after replay."""
    plan = workspace.plan
    required = {
        name: getattr(workspace, name, None)
        for name in (
            "request_indices",
            "block_valid_mask",
            "merge_indptr",
            "o_indptr",
            "kv_chunk_size_ptr",
        )
    }
    missing = [name for name, tensor in required.items() if not isinstance(tensor, torch.Tensor)]
    if missing:
        raise RuntimeError(
            "decode graph replay topology is missing workspace tensors: "
            f"{missing}"
        )
    if batch <= 0 or batch > int(plan.total_q):
        raise ValueError(
            f"observed decode batch {batch} is outside plan total_q={plan.total_q}"
        )

    chunk_tokens = int(required["kv_chunk_size_ptr"][0].item())
    page_size = int(plan.page_size)
    if chunk_tokens <= 0 or chunk_tokens % page_size != 0:
        raise RuntimeError(
            "captured decode updater produced an invalid chunk size: "
            f"chunk_tokens={chunk_tokens}, page_size={page_size}"
        )
    partial_rows = int(required["merge_indptr"][batch].item())
    output_rows = int(required["o_indptr"][batch].item())
    if output_rows != partial_rows:
        raise RuntimeError(
            "captured decode updater produced inconsistent merge/output topology: "
            f"merge_rows={partial_rows}, output_rows={output_rows}"
        )
    work_item_capacity = int(required["request_indices"].numel())
    regularized = bool(
        getattr(workspace, "_use_regular_decode_graph_replay", False)
    )
    useful_work_items = (
        partial_rows
        if regularized
        else int(required["block_valid_mask"].count_nonzero().item())
    )
    if useful_work_items < 0 or useful_work_items > work_item_capacity:
        raise RuntimeError(
            "captured decode updater produced an invalid useful-work count: "
            f"useful={useful_work_items}, capacity={work_item_capacity}"
        )
    num_kv_heads = int(plan.num_kv_heads)
    return {
        "schema": "b12x-decode-graph-observed-topology-v2",
        "source": "captured-device-lut-updater",
        "scheduling_mode": (
            "regularized-fixed-grid" if regularized else "compact-valid-mask"
        ),
        "kv_chunk_size_tokens": chunk_tokens,
        "kv_chunk_size_pages": chunk_tokens // page_size,
        "useful_work_items": useful_work_items,
        "work_item_capacity": work_item_capacity,
        "padded_work_items": work_item_capacity - useful_work_items,
        "forward_grid_ctas": work_item_capacity * num_kv_heads,
        "useful_forward_ctas": useful_work_items * num_kv_heads,
        "early_exit_forward_ctas": (
            work_item_capacity - useful_work_items
        )
        * num_kv_heads,
        "partial_rows": partial_rows,
    }


def _run_backend_forward(
    *,
    workspace: PagedAttentionWorkspace,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
) -> None:
    binding = build_paged_attention_binding(
        scratch=workspace,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    paged_attention_forward(binding=binding)


def _build_backend_graph_plan(
    *,
    workspace: PagedAttentionWorkspace,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    fixed_split_pages: int | None,
    graph_ctas_per_sm: int | None,
) -> object:
    assert workspace._plan_q is not None
    assert workspace._plan_k_cache is not None
    assert workspace._plan_v_cache is not None
    active_total_q = int(cu_seqlens_q[-1].item())
    plan = create_paged_plan(
        workspace._plan_q[:active_total_q],
        workspace._plan_k_cache,
        workspace._plan_v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode=workspace.mode,
        fixed_split_size=-1 if fixed_split_pages is None else int(fixed_split_pages),
        disable_split_kv=False,
        # A fixed split override is an explicit benchmark contract, not merely
        # chunk-size metadata.  Force the split path so the CLI can never
        # silently benchmark the direct kernel while printing a split value.
        force_split_kv=True if fixed_split_pages is not None else None,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    if fixed_split_pages is not None and not plan.split_kv:
        raise RuntimeError(
            "fixed split benchmark control produced a no-split paged plan"
        )
    return plan


def _load_backend_graph_plan(
    *,
    workspace: PagedAttentionWorkspace,
    plan: object,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> str:
    workspace._ensure_capacity(plan)
    workspace._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
    workspace._copy_plan_metadata(plan)
    workspace._plan = plan
    return _format_plan_desc(kv_chunk_size=plan.kv_chunk_size, split_kv=plan.split_kv)


_WORKSPACE_SCHEDULE_FIELDS = (
    "request_indices",
    "qo_tile_indices",
    "kv_tile_indices",
    "merge_indptr",
    "o_indptr",
    "kv_chunk_size_ptr",
    "kv_window_start_tokens",
    "total_num_rows_ptr",
    "block_valid_mask",
)

_WORKSPACE_RUNTIME_METADATA_FIELDS = (
    "page_table",
    "cache_seqlens",
    "cu_seqlens_q",
)

_WORKSPACE_POLICY_READ_ONLY_FIELDS = (
    "_decode_graph_chunk_pages_lut",
)


def _workspace_read_only_inputs(
    workspace: object,
    *,
    include_schedule: bool = True,
    include_runtime_metadata: bool = True,
) -> dict[str, torch.Tensor]:
    fields = _WORKSPACE_POLICY_READ_ONLY_FIELDS
    if include_runtime_metadata:
        fields = _WORKSPACE_RUNTIME_METADATA_FIELDS + fields
    if include_schedule:
        fields = _WORKSPACE_SCHEDULE_FIELDS + fields
    return {
        f"workspace.{name}": tensor
        for name in fields
        if isinstance((tensor := getattr(workspace, name, None)), torch.Tensor)
    }


def _snapshot_backend_replay_inputs(
    *,
    base_snapshot: _ReadOnlyInputSnapshot,
    base_inputs: Mapping[str, torch.Tensor],
    workspace: object,
    include_workspace_schedule: bool = True,
    include_workspace_runtime_metadata: bool = True,
) -> tuple[_ReadOnlyInputSnapshot, dict[str, torch.Tensor]]:
    workspace_inputs = _workspace_read_only_inputs(
        workspace,
        include_schedule=include_workspace_schedule,
        include_runtime_metadata=include_workspace_runtime_metadata,
    )
    snapshot = _extend_read_only_input_snapshot(base_snapshot, **workspace_inputs)
    return snapshot, {**base_inputs, **workspace_inputs}


def _poison_backend_result_regions(capture: BackendCapture | B12XDecodeGraphBucket) -> None:
    capture.guarded_output.poison()
    workspace = capture.workspace
    for tensor in (workspace.lse, workspace.tmp_output, workspace.tmp_lse):
        if tensor is not None:
            tensor.fill_(float("nan"))


def _active_split_kv_temporary_results(
    *,
    tmp_output: torch.Tensor,
    tmp_lse: torch.Tensor,
    o_indptr: torch.Tensor,
    batch: int,
    regular_decode_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select only temporary rows written by the current correctness replay."""
    if batch <= 0:
        raise AssertionError(f"decode graph replay produced an invalid batch: {batch}")
    if int(o_indptr.shape[0]) < batch + 1:
        raise AssertionError("decode graph replay o_indptr is smaller than the graph batch")

    active_partial_rows = int(o_indptr[batch].item())
    if not 0 < active_partial_rows <= int(tmp_output.shape[0]):
        raise AssertionError(
            "decode graph replay produced an invalid active partial-row count: "
            f"{active_partial_rows}"
        )
    if not regular_decode_graph:
        return tmp_output[:active_partial_rows], tmp_lse[:active_partial_rows]

    # The regularized fixed grid reserves max_chunks_per_request rows for each
    # request.  Its forward and merge kernels address row
    # req * max_chunks_per_request + chunk, so shorter replays leave deliberate
    # holes between requests.  This is a post-replay correctness observation;
    # it never feeds launch policy or timing.
    if int(tmp_output.shape[0]) % batch != 0:
        raise AssertionError(
            "regular decode graph temporary-row capacity is not divisible by batch"
        )
    max_chunks_per_request = int(tmp_output.shape[0]) // batch
    chunks_per_request = o_indptr[1 : batch + 1] - o_indptr[:batch]
    valid_chunk_counts = (chunks_per_request > 0) & (
        chunks_per_request <= max_chunks_per_request
    )
    if not bool(valid_chunk_counts.all().item()):
        raise AssertionError(
            "regular decode graph replay produced an invalid per-request chunk count"
        )
    chunk_slots = torch.arange(max_chunks_per_request, device=o_indptr.device)
    active_mask = chunk_slots.unsqueeze(0) < chunks_per_request.unsqueeze(1)
    return (
        tmp_output.reshape(batch, max_chunks_per_request, *tmp_output.shape[1:])[
            active_mask
        ],
        tmp_lse.reshape(batch, max_chunks_per_request, *tmp_lse.shape[1:])[
            active_mask
        ],
    )


def _assert_backend_result_regions_overwritten(
    capture: BackendCapture | B12XDecodeGraphBucket,
) -> None:
    capture.guarded_output.assert_fully_overwritten(backend="b12x")
    workspace = capture.workspace
    plan = workspace.plan
    lse = workspace.current_lse_view()
    if not bool(torch.isfinite(lse).all().item()):
        raise AssertionError("b12x logical LSE result was not fully overwritten with finite values")
    if plan.split_kv:
        assert workspace.tmp_output is not None
        assert workspace.tmp_lse is not None
        tmp_output = workspace.tmp_output[: int(plan.total_num_partial_rows)]
        tmp_lse = workspace.tmp_lse[: int(plan.total_num_partial_rows)]
        if (
            isinstance(capture, B12XDecodeGraphBucket)
            and workspace._decode_graph_chunk_pages_lut is not None
        ):
            assert workspace.o_indptr is not None
            tmp_output, tmp_lse = _active_split_kv_temporary_results(
                tmp_output=tmp_output,
                tmp_lse=tmp_lse,
                o_indptr=workspace.o_indptr,
                batch=capture.batch,
                regular_decode_graph=bool(
                    workspace._use_regular_decode_graph_replay
                ),
            )
        elif (
            workspace.use_cuda_graph
            and workspace.mode in ("extend", "verify")
        ):
            assert workspace.o_indptr is not None
            tmp_output, tmp_lse = _active_split_kv_temporary_results(
                tmp_output=tmp_output,
                tmp_lse=tmp_lse,
                o_indptr=workspace.o_indptr,
                batch=int(plan.page_table_shape[0]),
                regular_decode_graph=False,
            )
        # Causally empty split rows are represented by LSE=-inf.  Their
        # temporary output is semantically ignored by the merge and need not
        # be written.  Require every contributing partial to be finite and
        # reject NaN/+inf LSE without imposing writes on neutral slots.
        if bool(torch.isnan(tmp_lse).any().item()) or bool(
            torch.isposinf(tmp_lse).any().item()
        ):
            raise AssertionError(
                "b12x active split-KV temporary LSE contains NaN or +inf"
            )
        contributing = torch.isfinite(tmp_lse)
        if not bool(contributing.any().item()):
            raise AssertionError(
                "b12x split-KV replay produced no contributing partials"
            )
        if not bool(torch.isfinite(tmp_output[contributing]).all().item()):
            raise AssertionError(
                "b12x contributing split-KV temporary output was not fully overwritten"
            )


def _strict_backend_replay_for_correctness(
    capture: BackendCapture | B12XDecodeGraphBucket,
    *,
    l2_flush=None,
) -> None:
    if capture.read_only_snapshot is None or capture.read_only_inputs is None:
        raise RuntimeError("strict correctness replay requires pre-launch input snapshots")
    if l2_flush is not None:
        l2_flush()
        torch.cuda.synchronize()
    _poison_backend_result_regions(capture)
    torch.cuda.synchronize()
    capture.graph.replay()
    torch.cuda.synchronize()
    _assert_backend_result_regions_overwritten(capture)
    _assert_read_only_inputs_unchanged(
        capture.read_only_snapshot,
        capture.read_only_inputs,
    )


def _strict_guarded_replay_for_correctness(
    *,
    backend: str,
    graph: torch.cuda.CUDAGraph,
    guarded_output: _GuardedOutput,
    read_only_snapshot: _ReadOnlyInputSnapshot | None,
    read_only_inputs: Mapping[str, torch.Tensor] | None,
    l2_flush=None,
) -> None:
    if read_only_snapshot is None or read_only_inputs is None:
        raise RuntimeError("strict correctness replay requires pre-launch input snapshots")
    if l2_flush is not None:
        l2_flush()
        torch.cuda.synchronize()
    guarded_output.poison()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    guarded_output.assert_fully_overwritten(backend=backend)
    _assert_read_only_inputs_unchanged(read_only_snapshot, read_only_inputs)


def _decode_effective_cache_tokens(
    *,
    context_tokens: int,
    q_seqlen: int = 1,
) -> int:
    if context_tokens < 0:
        raise ValueError("decode context_tokens must be non-negative")
    if q_seqlen <= 0:
        raise ValueError("decode q_seqlen must be positive")
    return int(context_tokens + q_seqlen)


@dataclass(frozen=True)
class DecodeReplayCase:
    batch: int
    context_tokens: int

    @property
    def effective_cache_tokens(self) -> int:
        return _decode_effective_cache_tokens(context_tokens=self.context_tokens)


def _build_decode_replay_cases(
    *,
    batch_buckets: list[int],
    context_tokens: list[int],
) -> list[DecodeReplayCase]:
    if not batch_buckets:
        raise ValueError("expected at least one batch bucket")
    if not context_tokens:
        raise ValueError("expected at least one decode context")
    if any(batch <= 0 for batch in batch_buckets):
        raise ValueError("decode batch buckets must be positive")
    if any(context <= 0 for context in context_tokens):
        raise ValueError("decode graph bucket contexts must be positive")
    return [
        DecodeReplayCase(
            batch=int(batch),
            context_tokens=int(context),
        )
        for batch in sorted(dict.fromkeys(batch_buckets))
        for context in sorted(dict.fromkeys(context_tokens))
    ]


def _next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class DecodeGraphBucketPolicy:
    batch: int
    capture_context_tokens: int
    capture_page_count: int
    graph_ctas_per_sm: int
    query_tiles_per_request: int
    architecture_max_chunks_per_request: int
    max_chunks_per_request: int
    max_work_items: int
    max_partial_rows: int
    worst_page_count: int
    source: str

    @property
    def effective_capture_tokens(self) -> int:
        return _decode_effective_cache_tokens(
            context_tokens=self.capture_context_tokens
        )


def _resolve_decode_graph_bucket_policy(
    *,
    batch: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    decode_contexts: list[int],
    capture_context_override: int,
    fixed_split_pages_override: int,
    graph_ctas_per_sm_override: int,
    max_chunks_per_request_override: int = 0,
) -> DecodeGraphBucketPolicy:
    if fixed_split_pages_override > 0:
        raise ValueError(
            "decode-graph-buckets uses the production device-LUT replay policy; "
            "--fixed-split-pages is only supported by legacy-matrix"
        )
    if capture_context_override > 0:
        capture_context_tokens = int(capture_context_override)
        source = "manual"
    else:
        requested_capture_pages = (
            int(_decode_effective_cache_tokens(context_tokens=max(decode_contexts)))
            + page_size
            - 1
        ) // page_size
        capture_context_tokens = int(
            _next_power_of_two(requested_capture_pages) * page_size - 1
        )
        source = "heuristic"

    if capture_context_tokens < max(decode_contexts):
        raise ValueError(
            "decode graph capture context must cover the largest replay context"
        )
    capture_page_count = (
        int(_decode_effective_cache_tokens(context_tokens=capture_context_tokens))
        + page_size
        - 1
    ) // page_size
    def resolve_capacity(
        max_work_items: int | None = None,
        max_partial_rows: int | None = None,
    ) -> object:
        return paged.decode_graph_capacity(
            device=torch.device("cuda", torch.cuda.current_device()),
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page_size,
            batch=batch,
            max_cache_page_count=capture_page_count,
            graph_ctas_per_sm=(
                int(graph_ctas_per_sm_override)
                if graph_ctas_per_sm_override > 0
                else None
            ),
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
        )

    capacity = resolve_capacity()
    if max_chunks_per_request_override > 0:
        capacity = resolve_capacity(
            max_work_items=(
                batch
                * capacity.query_tiles_per_request
                * max_chunks_per_request_override
            ),
            max_partial_rows=batch * max_chunks_per_request_override,
        )
        source = "manual-chunks"

    return DecodeGraphBucketPolicy(
        batch=int(batch),
        capture_context_tokens=int(capture_context_tokens),
        capture_page_count=capture_page_count,
        graph_ctas_per_sm=capacity.graph_ctas_per_sm,
        query_tiles_per_request=capacity.query_tiles_per_request,
        architecture_max_chunks_per_request=(
            capacity.architecture_max_chunks_per_request
        ),
        max_chunks_per_request=capacity.max_chunks_per_request,
        max_work_items=capacity.max_work_items,
        max_partial_rows=capacity.max_partial_rows,
        worst_page_count=capacity.worst_page_count,
        source=source,
    )


@dataclass(frozen=True)
class DecodeBucketSharedInputs:
    batch: int
    capture_context_tokens: int
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    capture_page_table: torch.Tensor
    capture_cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    k_descale: torch.Tensor | None
    v_descale: torch.Tensor | None
    k_scale: float | None
    v_scale: float | None
    read_only_snapshot: _ReadOnlyInputSnapshot | None

    @property
    def read_only_inputs(self) -> dict[str, torch.Tensor]:
        inputs = {
            "q": self.q,
            "k_cache": self.k_cache,
            "v_cache": self.v_cache,
            "capture_page_table": self.capture_page_table,
            "capture_cache_seqlens": self.capture_cache_seqlens,
            "cu_seqlens_q": self.cu_seqlens_q,
        }
        if self.k_descale is not None:
            inputs["k_descale"] = self.k_descale
        if self.v_descale is not None:
            inputs["v_descale"] = self.v_descale
        return inputs


def _make_decode_bucket_shared_inputs(
    *,
    batch: int,
    capture_context_tokens: int,
    page_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    seed: int,
    strict_check: bool = True,
    combined_kv_cache: bool = False,
    num_cache_pages: int | None = None,
    q_strides: tuple[int, ...] | None = None,
) -> DecodeBucketSharedInputs:
    (
        q,
        k_cache,
        v_cache,
        capture_page_table,
        capture_cache_seqlens,
        _capture_page_table_dup,
        _capture_cache_seqlens_dup,
        cu_seqlens_q,
    ) = _make_uniform_paged_inputs(
        batch=batch,
        q_seqlen=1,
        cache_seqlen=_decode_effective_cache_tokens(
            context_tokens=capture_context_tokens
        ),
        capture_cache_seqlen=_decode_effective_cache_tokens(
            context_tokens=capture_context_tokens
        ),
        page_size=page_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        seed=seed,
        combined_kv_cache=combined_kv_cache,
        num_cache_pages=num_cache_pages,
    )
    if q_strides is not None:
        q = torch.empty_strided(q.shape, q_strides, dtype=q.dtype, device=q.device).copy_(q)
    if os.environ.get("B12X_PAGED_DEBUG_IDENTICAL_KV_HEADS", "0") == "1":
        k_cache.copy_(k_cache[:, :, :1, :].expand_as(k_cache))
        v_cache.copy_(v_cache[:, :, :1, :].expand_as(v_cache))
    k_descale = None
    v_descale = None
    k_scale = None
    v_scale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_cache, v_cache, k_descale, v_descale, k_scale, v_scale = (
            _quantize_paged_kv_cache_global_e4m3(
                k_cache,
                v_cache,
                batch=batch,
                kv_heads=kv_heads,
                combined_kv_cache=combined_kv_cache,
            )
        )
    read_only_snapshot = (
        _extend_read_only_input_snapshot(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            capture_page_table=capture_page_table,
            capture_cache_seqlens=capture_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        if strict_check
        else None
    )
    return DecodeBucketSharedInputs(
        batch=batch,
        capture_context_tokens=int(capture_context_tokens),
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        capture_page_table=capture_page_table,
        capture_cache_seqlens=capture_cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        k_scale=k_scale,
        v_scale=v_scale,
        read_only_snapshot=read_only_snapshot,
    )


@dataclass
class B12XDecodeGraphBucket:
    shared: DecodeBucketSharedInputs
    preparation: object
    scratch_plan: object
    scratch_storage: torch.Tensor
    binding: object
    workspace: object
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    guarded_output: _GuardedOutput
    current_page_table: torch.Tensor
    current_cache_seqlens: torch.Tensor
    current_plan_desc: str
    forward_traits_contract: dict[str, object]
    read_only_snapshot: _ReadOnlyInputSnapshot | None
    read_only_inputs: dict[str, torch.Tensor] | None

    @property
    def batch(self) -> int:
        return self.shared.batch

    @property
    def q(self) -> torch.Tensor:
        return self.shared.q

    @property
    def k_cache(self) -> torch.Tensor:
        return self.shared.k_cache

    @property
    def v_cache(self) -> torch.Tensor:
        return self.shared.v_cache

    @property
    def cu_seqlens_q(self) -> torch.Tensor:
        return self.shared.cu_seqlens_q

    @property
    def k_descale(self) -> torch.Tensor | None:
        return self.shared.k_descale

    @property
    def v_descale(self) -> torch.Tensor | None:
        return self.shared.v_descale

    def prepare_replay(self, *, context_tokens: int) -> None:
        if context_tokens > self.shared.capture_context_tokens:
            raise ValueError(
                "decode graph replay context exceeds the captured bucket: "
                f"{context_tokens} > {self.shared.capture_context_tokens}"
            )
        effective_cache_tokens = _decode_effective_cache_tokens(
            context_tokens=context_tokens
        )
        # The graph owns a fixed page table and fixed schedule topology.  A
        # replay changes only this persistent device length tensor; the
        # captured device updater selects chunking and valid work from the LUT.
        self.current_cache_seqlens.fill_(effective_cache_tokens)
        if self.shared.read_only_snapshot is not None:
            source_snapshot = _extend_read_only_input_snapshot(
                self.shared.read_only_snapshot,
                page_table=self.current_page_table,
                cache_seqlens=self.current_cache_seqlens,
            )
            source_inputs = {
                **self.shared.read_only_inputs,
                "page_table": self.current_page_table,
                "cache_seqlens": self.current_cache_seqlens,
            }
            self.read_only_snapshot, self.read_only_inputs = (
                _snapshot_backend_replay_inputs(
                    base_snapshot=source_snapshot,
                    base_inputs=source_inputs,
                    workspace=self.workspace,
                    include_workspace_schedule=False,
                    include_workspace_runtime_metadata=False,
                )
            )
        else:
            self.read_only_snapshot = None
            self.read_only_inputs = None




def _capture_backend_graph(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    capture_page_table: torch.Tensor,
    capture_cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    fixed_split_pages: int | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    warmup: int,
    graph_ctas_per_sm: int | None,
    window_left: int,
    paged_mode: str = "auto",
    strict_check: bool = True,
) -> BackendCapture:
    base_snapshot = (
        _extend_read_only_input_snapshot(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            capture_page_table=capture_page_table,
            capture_cache_seqlens=capture_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        if strict_check
        else None
    )
    base_inputs = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "capture_page_table": capture_page_table,
        "capture_cache_seqlens": capture_cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
    }
    if k_descale is not None:
        base_inputs["k_descale"] = k_descale
    if v_descale is not None:
        base_inputs["v_descale"] = v_descale
    guarded_output = _allocate_guarded_output(q)
    output = guarded_output.output
    inferred_mode = (
        "decode" if int(q.shape[0]) == int(page_table.shape[0]) else "extend"
    )
    mode = inferred_mode if paged_mode == "auto" else paged_mode
    if mode not in ("decode", "extend", "verify"):
        raise ValueError(f"unsupported paged attention mode {mode!r}")
    if mode == "decode" and inferred_mode != "decode":
        raise ValueError("decode mode requires exactly one query row per request")
    if mode == "verify" and inferred_mode == "decode":
        raise ValueError("verify mode requires more than one query row per request")
    workspace = PagedAttentionWorkspace.for_tensors(
        mode=mode,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        use_cuda_graph=True,
    )
    if mode in ("extend", "verify"):
        if fixed_split_pages is not None or graph_ctas_per_sm is not None:
            raise ValueError(
                "extend/verify graph replay uses the workspace planner policy; "
                "fixed split and CTA overrides are decode-only benchmark options"
            )
        workspace.prepare_prefill_graph_replay_state(
            batch=int(capture_page_table.shape[0]),
            total_q_capacity=int(q.shape[0]),
            max_page_table_width=int(capture_page_table.shape[1]),
            max_cache_seqlen=int(capture_cache_seqlens.max().item()),
            cu_seqlens_q=cu_seqlens_q,
            window_left=window_left,
        )
        capture_plan = workspace.plan
        replay_plan = None
    elif fixed_split_pages is None and graph_ctas_per_sm is None:
        workspace.prepare_decode_graph_replay_state(
            batch=int(capture_page_table.shape[0]),
            total_q_capacity=int(q.shape[0]),
            max_page_table_width=int(capture_page_table.shape[1]),
            max_cache_page_count=int(capture_page_table.shape[1]),
            window_left=window_left,
        )
        workspace._copy_runtime_metadata(
            capture_page_table,
            capture_cache_seqlens,
            cu_seqlens_q,
        )
        workspace.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
        capture_plan = workspace.plan
        replay_plan = None
    else:
        replay_plan = _build_backend_graph_plan(
            workspace=workspace,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            fixed_split_pages=fixed_split_pages,
            graph_ctas_per_sm=graph_ctas_per_sm,
        )
        capture_plan = _build_backend_graph_plan(
            workspace=workspace,
            page_table=capture_page_table,
            cache_seqlens=capture_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            fixed_split_pages=fixed_split_pages,
            graph_ctas_per_sm=graph_ctas_per_sm,
        )
        workspace._ensure_capacity(capture_plan)
        _load_backend_graph_plan(
            workspace=workspace,
            plan=capture_plan,
            page_table=capture_page_table,
            cache_seqlens=capture_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )

    def run() -> None:
        _run_backend_forward(
            workspace=workspace,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    if base_snapshot is not None:
        capture_snapshot, capture_inputs = _snapshot_backend_replay_inputs(
            base_snapshot=base_snapshot,
            base_inputs=base_inputs,
            workspace=workspace,
        )
    else:
        capture_snapshot = None
        capture_inputs = None
    graph = _capture_graph(run, warmup=warmup)
    guarded_output.assert_fully_overwritten(backend="b12x-capture")
    if capture_snapshot is not None and capture_inputs is not None:
        _assert_read_only_inputs_unchanged(capture_snapshot, capture_inputs)
    if mode in ("extend", "verify"):
        workspace.update_prefill_graph_replay_metadata(
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            window_left=window_left,
        )
        chunk_desc = _format_plan_desc(
            kv_chunk_size=capture_plan.kv_chunk_size,
            split_kv=capture_plan.split_kv,
        )
    elif replay_plan is None:
        workspace._copy_runtime_metadata(
            page_table,
            cache_seqlens,
            cu_seqlens_q,
        )
        workspace.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
        chunk_desc = _format_plan_desc(
            kv_chunk_size=capture_plan.kv_chunk_size,
            split_kv=capture_plan.split_kv,
        )
    else:
        chunk_desc = _load_backend_graph_plan(
            workspace=workspace,
            plan=replay_plan,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )
    if base_snapshot is not None:
        read_only_snapshot, read_only_inputs = _snapshot_backend_replay_inputs(
            base_snapshot=base_snapshot,
            base_inputs=base_inputs,
            workspace=workspace,
        )
    else:
        read_only_snapshot = None
        read_only_inputs = None
    return BackendCapture(
        graph=graph,
        workspace=workspace,
        output=output,
        guarded_output=guarded_output,
        plan_desc=chunk_desc,
        read_only_snapshot=read_only_snapshot,
        read_only_inputs=read_only_inputs,
    )








def _capture_b12x_decode_graph_bucket(
    *,
    shared: DecodeBucketSharedInputs,
    policy: DecodeGraphBucketPolicy,
    warmup: int,
    serving_declaration: dict | None = None,
) -> B12XDecodeGraphBucket:
    if policy.batch != shared.batch:
        raise ValueError("decode graph policy batch does not match shared inputs")
    if policy.capture_page_count != int(shared.capture_page_table.shape[1]):
        raise ValueError(
            "decode graph policy page capacity does not match shared inputs"
        )
    caps = paged.Caps(
        device=shared.q.device,
        mode="decode",
        dtype=shared.q.dtype,
        kv_dtype=shared.k_cache.dtype,
        num_q_heads=int(shared.q.shape[1]),
        num_kv_heads=int(shared.k_cache.shape[2]),
        head_dim_qk=int(shared.q.shape[2]),
        head_dim_vo=int(shared.v_cache.shape[3]),
        page_size=int(shared.k_cache.shape[1]),
        max_total_q=shared.batch,
        max_batch=shared.batch,
        max_page_table_width=policy.capture_page_count,
        max_work_items=policy.max_work_items,
        max_partial_rows=policy.max_partial_rows,
        num_cache_pages=int(shared.k_cache.shape[0]),
        use_cuda_graph=True,
        # The serving graph owns stable, full-bucket device metadata buffers.
        # Bind those addresses directly so replay measures the device policy
        # updater and attention kernels, not redundant full page-table D2D
        # copies introduced solely by the benchmark harness.
        copy_runtime_metadata=False,
    )
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    if serving_declaration is not None:
        payload = dict(serving_declaration["invocation"]["caps"])
        payload["dtype"] = getattr(torch, payload["dtype"])
        payload["kv_dtype"] = getattr(torch, payload["kv_dtype"])
        caps = paged.Caps(device=shared.q.device, **payload)

    replay_cache_seqlens = shared.capture_cache_seqlens.clone()
    guarded_output = _allocate_guarded_output(shared.q)
    output = guarded_output.output
    operands = dict(
        q=shared.q, k_cache=shared.k_cache, v_cache=shared.v_cache, output=output,
        page_table=shared.capture_page_table, cache_seqlens=replay_cache_seqlens,
        cu_seqlens_q=shared.cu_seqlens_q, k_descale=shared.k_descale, v_descale=shared.v_descale,
    )
    invocation = paged.invocation_from_tensors(caps, **operands)
    if serving_declaration is not None:
        actual = json.loads(json.dumps(invocation["operands"].to_dict()))
        expected = serving_declaration["invocation"]["operands"]
        if actual != expected:
            differences = {name: {"actual": actual.get(name), "expected": expected.get(name)}
                           for name in actual.keys() | expected.keys()
                           if actual.get(name) != expected.get(name)}
            raise ValueError(f"benchmark operand metadata differs from the serving declaration: {differences}")
    scratch_plan = paged.plan(caps, invocation=invocation)
    source_values = shared.q.clone()

    def prepare(state):
        spec = state.scratch_plan.scratch_specs()[0]
        storage = torch.empty(spec.shape, dtype=spec.dtype, device=caps.device)
        binding = state.bind(scratch=storage, active_total_q=shared.batch, **operands)
        return PreparedCall(run=lambda: state.run(binding),
                            produce=lambda: shared.q.copy_(source_values),
                            owners=(storage, binding), capture_safe=False)

    preparation = PreparationSession(device=caps.device, autotune=True, compile_workers=1)
    preparation.prepare((scratch_plan.request(name="paged-decode", prepare_call=prepare,
                                              benchmark_call=prepare),))
    preparation.freeze()
    state = require_prepared(scratch_plan, "attention.gqa", caps.device)
    (scratch_spec,) = state.scratch_plan.scratch_specs()
    scratch_storage = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=caps.device)
    captured_binding: object | None = None

    def run() -> None:
        nonlocal captured_binding
        captured_binding = paged.bind(
            scratch_plan,
            scratch=scratch_storage,
            q=shared.q,
            k_cache=shared.k_cache,
            v_cache=shared.v_cache,
            output=output,
            page_table=shared.capture_page_table,
            cache_seqlens=replay_cache_seqlens,
            cu_seqlens_q=shared.cu_seqlens_q,
            active_total_q=shared.batch,
            k_descale=shared.k_descale,
            v_descale=shared.v_descale,
        )
        paged.run(binding=captured_binding)

    graph = _capture_graph(run, warmup=warmup)
    if captured_binding is None:
        raise RuntimeError("decode graph capture did not retain its production binding")
    workspace = captured_binding.scratch
    capture_plan_desc = _format_decode_graph_replay_plan_desc(workspace)
    guarded_output.assert_fully_overwritten(backend="b12x-capture")
    if shared.read_only_snapshot is not None:
        _assert_read_only_inputs_unchanged(
            shared.read_only_snapshot,
            shared.read_only_inputs,
        )
        source_snapshot = _extend_read_only_input_snapshot(
            shared.read_only_snapshot,
            page_table=shared.capture_page_table,
            cache_seqlens=replay_cache_seqlens,
        )
        source_inputs = {
            **shared.read_only_inputs,
            "page_table": shared.capture_page_table,
            "cache_seqlens": replay_cache_seqlens,
        }
        read_only_snapshot, read_only_inputs = _snapshot_backend_replay_inputs(
            base_snapshot=source_snapshot,
            base_inputs=source_inputs,
            workspace=workspace,
            include_workspace_schedule=False,
            include_workspace_runtime_metadata=False,
        )
    else:
        read_only_snapshot = None
        read_only_inputs = None
    return B12XDecodeGraphBucket(
        shared=shared,
        preparation=preparation,
        scratch_plan=scratch_plan,
        scratch_storage=scratch_storage,
        binding=captured_binding,
        workspace=workspace,
        graph=graph,
        output=output,
        guarded_output=guarded_output,
        current_page_table=shared.capture_page_table,
        current_cache_seqlens=replay_cache_seqlens,
        current_plan_desc=capture_plan_desc,
        forward_traits_contract=_paged_forward_traits_contract(workspace.plan),
        read_only_snapshot=read_only_snapshot,
        read_only_inputs=read_only_inputs,
    )




def _reference_output_from_snapshot(
    snapshot: _ReadOnlyInputSnapshot,
    *,
    window_left: int = -1,
) -> torch.Tensor:
    cloned = snapshot.clones
    ref_out, _ = paged_attention_reference(
        cloned["q"],
        cloned["k_cache"],
        cloned["v_cache"],
        cloned["page_table"],
        cloned["cache_seqlens"],
        cloned["cu_seqlens_q"],
        k_descale=cloned.get("k_descale"),
        v_descale=cloned.get("v_descale"),
        causal=True,
        window_left=window_left,
    )
    return ref_out


def _decode_reference_output(
    *,
    read_only_snapshot: _ReadOnlyInputSnapshot,
) -> torch.Tensor:
    return _reference_output_from_snapshot(read_only_snapshot)


def _run_legacy_matrix(args: argparse.Namespace) -> None:
    dtype = _dtype_from_name(args.dtype)
    kv_dtype = _resolve_kv_dtype(args.kv_dtype, dtype)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    q_seqlens = _parse_csv_ints(args.q_seqlens)
    cache_seqlens = _parse_csv_ints(args.cache_seqlens)
    cases = _build_shape_cases(
        batch=args.batch,
        q_seqlens=q_seqlens,
        cache_seqlens=cache_seqlens,
    )
    if args.torch_profile_trace is not None and len(cases) != 1:
        raise ValueError("--torch-profile-trace requires exactly one shape case")

    print(
        "shape matrix:",
        {
            "mode": args.mode,
            "batch": args.batch,
            "q_seqlens": q_seqlens,
            "cache_seqlens": cache_seqlens,
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "q_dtype": str(dtype),
            "kv_dtype": str(kv_dtype),
            "kv_cache_layout": _kv_cache_layout_name(
                combined_kv_cache=args.combined_kv_cache
            ),
            "fixed_split_pages": args.fixed_split_pages,
            "capture_cache_seqlen": args.capture_cache_seqlen,
            "graph_ctas_per_sm": args.graph_ctas_per_sm,
            "paged_mode": args.paged_mode,
            "window_left": args.window_left,
            "replays": args.replays,

            "l2_flush": args.flush_l2,
        },
    )

    for case_idx, case in enumerate(cases):
        (
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens_tensor,
            capture_page_table,
            capture_cache_seqlens,
            cu_seqlens_q,
        ) = _make_uniform_paged_inputs(
            batch=case.batch,
            q_seqlen=case.q_seqlen,
            cache_seqlen=case.cache_seqlen,
            capture_cache_seqlen=args.capture_cache_seqlen
            if args.capture_cache_seqlen > 0
            else None,
            page_size=args.page_size,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            seed=1 + case_idx,
            combined_kv_cache=args.combined_kv_cache,
        )
        k_descale = None
        v_descale = None
        k_scale = None
        v_scale = None
        if kv_dtype == torch.float8_e4m3fn:
            k_cache, v_cache, k_descale, v_descale, k_scale, v_scale = (
                _quantize_paged_kv_cache_global_e4m3(
                    k_cache,
                    v_cache,
                    batch=case.batch,
                    kv_heads=args.kv_heads,
                    combined_kv_cache=args.combined_kv_cache,
                )
            )
        backend_capture = _capture_backend_graph(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens_tensor,
            capture_page_table=capture_page_table,
            capture_cache_seqlens=capture_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            fixed_split_pages=args.fixed_split_pages
            if args.fixed_split_pages > 0
            else None,
            k_descale=k_descale,
            v_descale=v_descale,
            warmup=args.warmup,
            graph_ctas_per_sm=args.graph_ctas_per_sm
            if args.graph_ctas_per_sm > 0
            else None,
            window_left=args.window_left,
            paged_mode=args.paged_mode,
            strict_check=args.check,
        )
        b12x_forward_traits = _paged_forward_traits_contract(
            backend_capture.workspace.plan
        )
        check_suffix = ""
        reference_output: torch.Tensor | None = None
        b12x_correctness: dict[str, object] | None = None
        if args.check:
            _strict_backend_replay_for_correctness(
                backend_capture,
                l2_flush=l2_flush,
            )
            reference_output = _reference_output_from_snapshot(
                backend_capture.read_only_snapshot,
                window_left=args.window_left,
            )
            reference_max_abs, reference_rel_l2, reference_cos, nonzero = (
                _reference_gate(
                    backend="b12x",
                    output=backend_capture.output,
                    reference=reference_output,
                )
            )
            b12x_correctness = {
                "oracle": "b12x.attention.paged.reference",
                "passed": True,
                "finite": True,
                "nonzero": nonzero,
                "max_abs": reference_max_abs,
                "relative_l2": reference_rel_l2,
                "cosine": reference_cos,
                "allclose": True,
                "minimum_cosine": _REFERENCE_MINIMUM_COSINE,
                "maximum_relative_l2": _REFERENCE_MAXIMUM_RELATIVE_L2,
                "relative_tolerance": _REFERENCE_RELATIVE_TOLERANCE,
                "absolute_tolerance": _REFERENCE_ABSOLUTE_TOLERANCE,
                "read_only_inputs": _read_only_input_provenance(
                    backend_capture.read_only_snapshot
                ),
            }
            check_suffix = (
                f" torch_max_abs={reference_max_abs:.5f}"
                f" torch_cos={reference_cos:.8f}"
                f" nonzero={nonzero}"
            )
        if args.torch_profile_trace is not None:
            _export_graph_trace(
                backend_capture.graph,
                path=args.torch_profile_trace,
            )
        backend_times_ms = _bench_graph(
            backend_capture.graph,
            replays=args.replays,
            l2_flush=l2_flush,
        )
        sample_case = _case_contract(
            {
                "profile": args.profile,
                "mode": args.mode,
                "paged_mode": args.paged_mode,
                "phase": case.phase,
                "batch": case.batch,
                "q_seqlen": case.q_seqlen,
                "cache_seqlen": case.cache_seqlen,
                "window_left": args.window_left,
                "page_size": args.page_size,
                "q_heads": args.q_heads,
                "kv_heads": args.kv_heads,
                "head_dim": args.head_dim,
                "q_dtype": str(dtype),
                "kv_dtype": str(kv_dtype),
                "kv_cache_layout": _kv_cache_layout_contract(k_cache, v_cache),
                "plan": backend_capture.plan_desc,
                "b12x_forward_traits": b12x_forward_traits,
            },
            input_seed=1 + case_idx,
            input_generator="uniform-paged-inputs-v1",
        )
        _record_samples(
            args.raw_samples_jsonl,
            backend="b12x",
            case=sample_case,
            samples_ms=backend_times_ms,
            correctness=b12x_correctness,
        )
        backend_metrics = CaseMetrics(
            backend="b12x",
            mean_us=statistics.fmean(backend_times_ms) * 1000.0,
        )


        line = (
            f"{case.phase:>6s} "
            f"bs={case.batch:2d} "
            f"q={case.q_seqlen:2d} "
            f"k={case.cache_seqlen:5d} "
            f"{backend_capture.plan_desc:>17s} "
            f"| {backend_metrics.backend} mean={backend_metrics.mean_us:8.1f} us"
        )
        print(line + check_suffix)



def _run_decode_graph_buckets(args: argparse.Namespace) -> None:
    if args.q_seqlens != "1":
        raise ValueError("decode-graph-buckets mode only supports --q-seqlens 1")
    dtype = _dtype_from_name(args.dtype)
    kv_dtype = _resolve_kv_dtype(args.kv_dtype, dtype)
    l2_flush = make_l2_flush_fn(args.flush_l2, args.l2_flush_bytes)
    batch_buckets = _parse_csv_ints(args.batch_buckets)
    decode_contexts = _parse_csv_ints(args.decode_contexts)
    cases = _build_decode_replay_cases(
        batch_buckets=batch_buckets,
        context_tokens=decode_contexts,
    )
    if args.torch_profile_trace is not None and len(cases) != 1:
        raise ValueError(
            "--torch-profile-trace requires exactly one decode graph case"
        )

    print(
        "decode graph buckets:",
        {
            "profile": args.profile,
            "mode": args.mode,
            "batch_buckets": sorted(dict.fromkeys(batch_buckets)),
            "decode_context_tokens": sorted(dict.fromkeys(decode_contexts)),
            "capture_context_override": None
            if args.capture_context <= 0
            else int(args.capture_context),
            "page_size": args.page_size,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "q_dtype": str(dtype),
            "kv_dtype": str(kv_dtype),
            "kv_cache_layout": _kv_cache_layout_name(
                combined_kv_cache=args.combined_kv_cache
            ),
            "fixed_split_pages": args.fixed_split_pages,
            "graph_ctas_per_sm": args.graph_ctas_per_sm,
            "replays": args.replays,

            "l2_flush": args.flush_l2,
        },
    )

    for bucket_idx, batch in enumerate(sorted(dict.fromkeys(batch_buckets))):
        serving_declaration = None
        if args.preparation_queries is not None:
            for line in args.preparation_queries.read_text().splitlines():
                record = json.loads(line)
                if (record.get("component") == "attention.gqa"
                        and record["invocation"]["caps"]["mode"] == "decode"
                        and record["invocation"]["caps"]["max_batch"] == batch):
                    serving_declaration = record
            if serving_declaration is None:
                raise ValueError(f"no serving decode declaration for batch={batch}")
            declared = serving_declaration["invocation"]["caps"]
            geometry = (declared["num_q_heads"], declared["num_kv_heads"], declared["head_dim_qk"], declared["page_size"])
            if geometry != (args.q_heads, args.kv_heads, args.head_dim, args.page_size):
                raise ValueError("benchmark attention geometry differs from serving declaration")
            capture_context = declared["max_page_table_width"] * args.page_size - 1
        else:
            capture_context = int(args.capture_context)
        bucket_policy = _resolve_decode_graph_bucket_policy(
            batch=batch,
            q_dtype=dtype,
            kv_dtype=kv_dtype,
            page_size=args.page_size,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            decode_contexts=decode_contexts,
            capture_context_override=capture_context,
            fixed_split_pages_override=int(args.fixed_split_pages),
            graph_ctas_per_sm_override=int(args.graph_ctas_per_sm),
            max_chunks_per_request_override=int(args.max_chunks_per_request),
        )
        if serving_declaration is not None:
            bucket_policy = replace(
                bucket_policy, max_work_items=declared["max_work_items"],
                max_partial_rows=declared["max_partial_rows"],
                max_chunks_per_request=declared["max_partial_rows"] // batch,
                source="serving-declaration",
            )
        shared = _make_decode_bucket_shared_inputs(
            batch=batch,
            capture_context_tokens=bucket_policy.capture_context_tokens,
            page_size=args.page_size,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            kv_dtype=kv_dtype,
            seed=1 + bucket_idx,
            strict_check=args.check,
            combined_kv_cache=args.combined_kv_cache,
            num_cache_pages=None if serving_declaration is None else declared["num_cache_pages"],
            q_strides=(None if serving_declaration is None else
                       tuple(serving_declaration["invocation"]["operands"]["q"]["strides"])),
        )
        b12x_bucket = _capture_b12x_decode_graph_bucket(
            shared=shared,
            policy=bucket_policy,
            warmup=args.warmup,
            serving_declaration=serving_declaration,
        )
        print(
            f"decode-graph-bucket "
            f"bs={batch:2d} "
            f"source={bucket_policy.source:>20s} "
            f"capture_ctx={bucket_policy.capture_context_tokens:6d} "
            f"capture_kv={bucket_policy.effective_capture_tokens:6d} "
            f"capture_pages={bucket_policy.capture_page_count:4d} "
            f"graph_ctas_per_sm={bucket_policy.graph_ctas_per_sm:2d} "
            f"qtiles_per_req={bucket_policy.query_tiles_per_request:2d} "
            f"max_chunks_per_req={bucket_policy.max_chunks_per_request:4d} "
            f"max_work_items={bucket_policy.max_work_items:5d} "
            f"max_partial_rows={bucket_policy.max_partial_rows:5d}"
        )

        for case in (case for case in cases if case.batch == batch):
            try:
                b12x_bucket.prepare_replay(context_tokens=case.context_tokens)
            except Exception as exc:
                raise RuntimeError(
                    f"requested decode-graph case was blocked: "
                    f"bs={case.batch}, ctx={case.context_tokens}, "
                    f"kv={case.effective_cache_tokens}, "
                    f"cap={bucket_policy.capture_context_tokens}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if args.torch_profile_trace is not None:
                _export_graph_trace(
                    b12x_bucket.graph,
                    path=args.torch_profile_trace,
                )

            check_suffix = ""
            b12x_correctness: dict[str, object] | None = None
            if args.check:
                tma_debug_dump = os.environ.get(
                    "B12X_PAGED_KV_TMA_DEBUG_DUMP", ""
                )
                if tma_debug_dump in ("K", "Q", "S", "V"):
                    if l2_flush is not None:
                        l2_flush()
                    _poison_backend_result_regions(b12x_bucket)
                    torch.cuda.synchronize()
                    b12x_bucket.graph.replay()
                    torch.cuda.synchronize()
                    tmp_output = b12x_bucket.workspace.tmp_output
                    assert tmp_output is not None
                    if tma_debug_dump == "Q":
                        expected_q = b12x_bucket.q[0, :6, :].float()
                        dumped_q = tmp_output.flatten()[
                            : expected_q.numel()
                        ].view_as(expected_q).float()
                        delta = dumped_q - expected_q
                        q_candidates = b12x_bucket.q[0].float()
                        nearest_q = torch.argmin(
                            torch.cdist(dumped_q, q_candidates), dim=1
                        ).tolist()
                        print(
                            f"Q stage dump mismatch_count="
                            f"{int((dumped_q != expected_q).sum().item())} "
                            f"max_abs={float(delta.abs().max().item())} "
                            f"rel_l2={float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected_q).clamp_min(1e-12))} "
                            f"nearest_q_heads={nearest_q}"
                        )
                        return
                    if tma_debug_dump == "S":
                        live_tokens = int(
                            b12x_bucket.current_cache_seqlens[0].item()
                        )
                        page_count = (live_tokens + args.page_size - 1) // args.page_size
                        page_ids = b12x_bucket.current_page_table[
                            0, :page_count
                        ].long()
                        logical_k = (
                            b12x_bucket.k_cache[page_ids]
                            .reshape(-1, args.kv_heads, args.head_dim)[
                                :live_tokens, 0, :
                            ]
                            .float()
                        )
                        k_scale = 1.0
                        if b12x_bucket.k_descale is not None:
                            if b12x_bucket.k_descale.ndim == 1:
                                k_scale = float(
                                    b12x_bucket.k_descale[0].item()
                                )
                            else:
                                k_scale = float(
                                    b12x_bucket.k_descale[0, 0].item()
                                )
                        expected_scores = (
                            b12x_bucket.q[0, :6, :].float()
                            @ logical_k.transpose(0, 1)
                        ) * k_scale
                        dumped_scores = tmp_output.flatten()[
                            : 6 * live_tokens
                        ].view(6, live_tokens).float()
                        delta = dumped_scores - expected_scores
                        rel_l2 = float(
                            torch.linalg.vector_norm(delta)
                            / torch.linalg.vector_norm(expected_scores).clamp_min(1e-12)
                        )
                        candidate_k = (
                            b12x_bucket.k_cache[page_ids]
                            .reshape(-1, args.kv_heads, args.head_dim)[:64]
                            .float()
                        )
                        candidate_scores = torch.einsum(
                            "qd,thd->thq",
                            b12x_bucket.q[0, :6, :].float(),
                            candidate_k,
                        ) * k_scale
                        nearest = []
                        for key_col in range(live_tokens):
                            candidate_error = torch.linalg.vector_norm(
                                candidate_scores
                                - dumped_scores[:, key_col].view(1, 1, 6),
                                dim=2,
                            )
                            nearest_flat = torch.topk(
                                candidate_error.flatten(),
                                k=min(4, candidate_error.numel()),
                                largest=False,
                            )
                            nearest.append(
                                [
                                    (
                                        int(idx.item()) // args.kv_heads,
                                        int(idx.item()) % args.kv_heads,
                                        float(err.item()),
                                    )
                                    for err, idx in zip(
                                        nearest_flat.values, nearest_flat.indices, strict=True
                                    )
                                ]
                            )
                        print(
                            f"S QK dump max_abs={float(delta.abs().max().item())} "
                            f"rel_l2={rel_l2} "
                            f"nearest_keys={nearest} "
                            f"dumped={dumped_scores.tolist()} "
                            f"expected={expected_scores.tolist()}"
                        )
                        return
                    page_id = int(b12x_bucket.current_page_table[0, 0].item())
                    cache = (
                        b12x_bucket.k_cache
                        if tma_debug_dump == "K"
                        else b12x_bucket.v_cache
                    )
                    expected_codes = (
                        cache[page_id, :24, 0, :]
                        .view(torch.uint8)
                        .float()
                    )
                    dumped_codes = tmp_output[0, :24, :].float()
                    mismatch = (dumped_codes != expected_codes)
                    candidate_codes = (
                        cache[page_id, :64, :, :]
                        .view(torch.uint8)
                        .reshape(64, args.kv_heads, args.head_dim)
                        .reshape(64, args.kv_heads, args.head_dim // 16, 16)
                    )
                    vector_mapping = []
                    dumped_vectors = dumped_codes[:4].reshape(
                        4, args.head_dim // 16, 16
                    )
                    for dump_row in range(dumped_vectors.shape[0]):
                        row_mapping = []
                        for dump_vec in range(dumped_vectors.shape[1]):
                            matches = (
                                candidate_codes
                                == dumped_vectors[dump_row, dump_vec]
                            ).all(dim=-1)
                            match_idx = matches.nonzero()
                            row_mapping.append(
                                [
                                    tuple(int(v) for v in idx.tolist())
                                    for idx in match_idx[:4]
                                ]
                            )
                        vector_mapping.append(row_mapping)
                    print(
                        f"{tma_debug_dump} TMA dump mismatch_count="
                        f"{int(mismatch.sum().item())} "
                        f"max_abs={float((dumped_codes - expected_codes).abs().max().item())} "
                        f"vector_mapping={vector_mapping}"
                    )
                    return
                _strict_backend_replay_for_correctness(
                    b12x_bucket,
                    l2_flush=l2_flush,
                )
                ref_out = _decode_reference_output(
                    read_only_snapshot=b12x_bucket.read_only_snapshot,
                )
                (
                    b12x_ref_max_abs,
                    b12x_ref_rel_l2,
                    b12x_ref_cos,
                    b12x_nonzero,
                ) = _reference_gate(
                    backend="b12x", output=b12x_bucket.output, reference=ref_out
                )
                b12x_correctness = {
                    "oracle": "b12x.attention.paged.reference",
                    "passed": True,
                    "finite": True,
                    "nonzero": b12x_nonzero,
                    "max_abs": b12x_ref_max_abs,
                    "relative_l2": b12x_ref_rel_l2,
                    "cosine": b12x_ref_cos,
                    "allclose": True,
                    "minimum_cosine": _REFERENCE_MINIMUM_COSINE,
                    "maximum_relative_l2": _REFERENCE_MAXIMUM_RELATIVE_L2,
                    "relative_tolerance": _REFERENCE_RELATIVE_TOLERANCE,
                    "absolute_tolerance": _REFERENCE_ABSOLUTE_TOLERANCE,
                    "read_only_inputs": _read_only_input_provenance(
                        b12x_bucket.read_only_snapshot
                    ),
                }
                check_suffix = (
                    f" | b12x/ref rel_l2={b12x_ref_rel_l2:.6f} cos={b12x_ref_cos:.8f}"
                )
            timing_metadata: dict[str, object] | None = None
            backend_times_ms = _bench_graph(
                b12x_bucket.graph,
                replays=args.replays,
                l2_flush=l2_flush,
            )
            timing_metadata = _decode_graph_timing_metadata(
                timing_metadata,

            )
            observed_replay_topology = _observe_decode_graph_replay_topology(
                b12x_bucket.workspace,
                batch=case.batch,
            )
            replay_policy = timing_metadata["replay_policy"]
            sample_case = _case_contract(
                {
                    "profile": args.profile,
                    "mode": args.mode,
                    "batch": case.batch,
                    "context_tokens": case.context_tokens,
                    "effective_cache_tokens": case.effective_cache_tokens,
                    "capture_context_tokens": bucket_policy.capture_context_tokens,
                    "capture_page_count": bucket_policy.capture_page_count,
                    "graph_ctas_per_sm": bucket_policy.graph_ctas_per_sm,
                    "query_tiles_per_request": (
                        bucket_policy.query_tiles_per_request
                    ),
                    "architecture_max_chunks_per_request": (
                        bucket_policy.architecture_max_chunks_per_request
                    ),
                    "max_chunks_per_request": (
                        bucket_policy.max_chunks_per_request
                    ),
                    "max_work_items": bucket_policy.max_work_items,
                    "max_partial_rows": bucket_policy.max_partial_rows,
                    "worst_page_count": bucket_policy.worst_page_count,
                    "page_size": args.page_size,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "q_dtype": str(dtype),
                    "kv_dtype": str(kv_dtype),
                    "kv_cache_layout": _kv_cache_layout_contract(
                        shared.k_cache,
                        shared.v_cache,
                    ),
                    "timing_method": "stream_gated_events_v1",
                    "replay_policy": replay_policy,
                    "plan": b12x_bucket.current_plan_desc,
                    "b12x_forward_traits": (
                        b12x_bucket.forward_traits_contract
                    ),
                    "b12x_observed_replay_topology": (
                        observed_replay_topology
                    ),
                },
                input_seed=1 + bucket_idx,
                input_generator="decode-bucket-shared-inputs-v1",
            )
            _record_samples(
                args.raw_samples_jsonl,
                backend="b12x",
                case=sample_case,
                samples_ms=backend_times_ms,
                correctness=b12x_correctness,
                timing=timing_metadata,
            )
            backend_metrics = CaseMetrics(
                backend="b12x",
                mean_us=statistics.fmean(backend_times_ms) * 1000.0,
            )


            line = (
                f"decode-graph "
                f"bs={case.batch:2d} "
                f"ctx={case.context_tokens:6d} "
                f"kv={case.effective_cache_tokens:6d} "
                f"cap={bucket_policy.capture_context_tokens:6d} "
                f"{b12x_bucket.current_plan_desc:>17s} "
                f"observed={observed_replay_topology['kv_chunk_size_pages']:3d}p/"
                f"{observed_replay_topology['useful_work_items']:3d}w/"
                f"{observed_replay_topology['work_item_capacity']:3d}c "
                f"| {backend_metrics.backend} mean={backend_metrics.mean_us:8.1f} us"
            )
            print(line + check_suffix)

        b12x_bucket.graph.reset()
        b12x_bucket.preparation.close()
        del b12x_bucket
        del shared
        torch.cuda.empty_cache()



def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    profile_name = _preparse_profile(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=_profile_choices(),
        default=profile_name,
        help="Shape preset. Explicit CLI shape flags override preset defaults.",
    )
    parser.add_argument(
        "--mode",
        choices=["legacy-matrix", "decode-graph-buckets"],
        default="decode-graph-buckets",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--batch-buckets", type=str, default="1,2,4,8,12,16")
    parser.add_argument(
        "--decode-contexts", type=str, default="128,16384,32768,65536,131072"
    )
    parser.add_argument("--capture-context", type=int, default=0)
    parser.add_argument("--preparation-queries", type=pathlib.Path,
                        help="Replay the last recorded decode declaration per batch from a serving query log")
    parser.add_argument("--q-seqlens", type=str, default="1")
    parser.add_argument("--cache-seqlens", type=str, default="64,512,2048,8192")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--kv-dtype", choices=["same", "bf16", "fp16", "fp8_e4m3fn"], default="same"
    )
    parser.add_argument(
        "--combined-kv-cache",
        action="store_true",
        help=(
            "benchmark K/V as strided views into one "
            "[pages, 2, page_size, kv_heads, head_dim] allocation"
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--fixed-split-pages", type=int, default=0)
    parser.add_argument("--capture-cache-seqlen", type=int, default=0)
    parser.add_argument("--graph-ctas-per-sm", type=int, default=0)
    parser.add_argument(
        "--max-chunks-per-request",
        type=int,
        default=0,
        help=(
            "cap each decode-graph request to this many device-LUT chunks; "
            "zero preserves the production planner"
        ),
    )
    parser.add_argument(
        "--paged-mode",
        choices=["auto", "decode", "extend", "verify"],
        default="auto",
        help=(
            "B12X kernel family for legacy-matrix cases. Use verify for uniform "
            "multi-token speculative decode; auto preserves shape inference."
        ),
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=-1,
        help="causal sliding-window size for legacy-matrix cases; -1 is full attention",
    )
    parser.add_argument("--ci-level", type=float, default=0.95, help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", action="store_false", dest="flush_l2")
    parser.add_argument(
        "--l2-flush-bytes",
        type=int,
        default=0,
        help="L2 eviction size in bytes; default is 2x detected L2 capacity.",
    )
    parser.add_argument(
        "--raw-samples-jsonl",
        type=pathlib.Path,
        help=(
            "write provenance and every graph-replay timing sample as JSONL; "
            "the file is replaced at benchmark start"
        ),
    )
    parser.add_argument(
        "--torch-profile-trace",
        type=pathlib.Path,
        help=(
            "export child-kernel CUDA graph traces for one benchmark case; "
            "the trace records the b12x graph replay"
        ),
    )
    parser.set_defaults(**BENCHMARK_PROFILES[profile_name])
    args = parser.parse_args(argv)
    args.profile = _canonical_profile_name(args.profile)

    require_cuda()
    if args.window_left < -1:
        raise ValueError("--window-left must be -1 or a non-negative token count")
    if args.mode != "legacy-matrix" and args.window_left != -1:
        raise ValueError("--window-left is currently supported only in legacy-matrix mode")
    if args.mode != "legacy-matrix" and args.paged_mode != "auto":
        raise ValueError("--paged-mode is currently supported only in legacy-matrix mode")
    if args.replays <= 0:
        raise ValueError("--replays must be positive for graph-replay benchmarking")
    if args.max_chunks_per_request < 0:
        raise ValueError("--max-chunks-per-request must be nonnegative")
    _gqa_group_size(q_heads=args.q_heads, kv_heads=args.kv_heads)
    _initialize_raw_sample_log(args.raw_samples_jsonl, args=args, argv=argv)
    l2_flush_bytes = resolve_l2_flush_bytes(args.l2_flush_bytes)
    flush_desc = (
        f"on ({l2_flush_bytes / (1 << 20):.1f} MiB per launch)"
        if args.flush_l2
        else "off"
    )
    print(f"L2 flush: {flush_desc}")
    clear_attention_caches()
    if args.mode == "legacy-matrix":
        _run_legacy_matrix(args)
        return
    _run_decode_graph_buckets(args)


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
