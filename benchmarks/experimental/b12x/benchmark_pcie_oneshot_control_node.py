"""Immutable one-run worker for the PCIe oneshot control-node A/B harness.

Use ``run_pcie_oneshot_control_node_ab.py`` rather than comparing two ad-hoc
JSON files. That external controller invokes this exact file for both source
trees in an alternating AB/BA sequence and validates every recorded contract.

This harness intentionally measures only the plain staged one-shot all-reduce
at 64 KiB by default. A public-head comparison is a net algorithm comparison;
it must not be described as isolated control-node launch overhead or generalized
to fused RMSNorm / two-shot collectives without their own measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.distributed as dist
from cuda.bindings import runtime as cudart

SCHEMA_NAME = "b12x.pcie_oneshot_control_node.run"
SCHEMA_VERSION = 3
DTYPE_NAME = "bfloat16"
TOP_LEVEL_KEYS = {
    "schema",
    "contract",
    "scope",
    "correctness",
    "identity",
    "provenance",
    "software",
    "hardware",
    "environment",
    "per_rank",
    "distributed_critical_path",
}
CORRECTNESS_KEYS = {"verdict"}
CONTRACT_KEYS = {
    "world_size",
    "numel",
    "dtype",
    "warmup",
    "iters",
    "preflight_replays",
}
SCOPE_KEYS = {
    "operation",
    "staging",
    "size_bytes",
    "comparison_semantics",
    "excluded_paths",
}
IDENTITY_KEYS = {
    "label",
    "variant",
    "run_index",
    "implementation_root",
    "implementation_git_sha",
    "implementation_git_dirty",
    "harness_sha256",
    "controller_sha256",
    "controller_argv_sha256",
    "worker_argv",
    "worker_argv_sha256",
    "started_at_utc",
}
PROVENANCE_KEYS = {
    "module_path",
    "module_sha256",
    "cuda_source_path",
    "cuda_source_sha256",
    "extension_path",
    "extension_sha256",
    "extension_build_root",
    "extension_build_manifest",
    "extension_build_manifest_sha256",
    "fresh_extension_cache",
    "post_run_verified",
}
METRIC_KEYS = {"cold_us", "mean_us", "p50_us", "p95_us", "min_us", "max_us"}
RANK_MODE_KEYS = METRIC_KEYS | {"samples_us"}
CRITICAL_MODE_KEYS = METRIC_KEYS | {"samples_us"}
SOFTWARE_KEYS = {
    "python",
    "platform",
    "torch",
    "torch_cuda",
    "cuda_runtime_version",
    "cuda_driver_version",
    "nvcc_version",
}
HARDWARE_KEYS = {
    "hostname",
    "devices",
    "nvidia_smi_identity",
    "nvidia_smi_sample_fields",
    "nvidia_smi_samples",
    "nvidia_smi_topology",
}
DEVICE_KEYS = {
    "index",
    "name",
    "uuid",
    "compute_capability",
    "total_memory_bytes",
    "multi_processor_count",
}
NVIDIA_QUERY_KEYS = {"fields", "returncode", "rows", "stderr"}
NVIDIA_SAMPLE_KEYS = {"monotonic_ns", "phase", "query"}
NVIDIA_TOPOLOGY_KEYS = {"returncode", "stdout", "stderr"}
GRAPH_STRUCTURE_KEYS = {
    "kernel_nodes",
    "direct_control_worker_edge",
    "control_grid",
    "control_block",
    "worker_grid",
    "worker_block",
}
NVIDIA_IDENTITY_FIELDS = ("index", "uuid", "name", "driver_version")
NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "name",
    "driver_version",
    "pstate",
    "power.limit",
    "power.draw",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.sm",
    "clocks.max.memory",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "clocks_event_reasons.active",
)
EXPLICIT_ENV_KEYS = {
    "CUDA_DEVICE_ORDER",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_EXTENSIONS_DIR",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-graph-kernel-nodes", type=int, required=True)
    parser.add_argument("--numel", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--sampler-interval", type=float, default=0.2)
    parser.add_argument("--preflight-replays", type=int, default=4)
    parser.add_argument("--distributed-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--controller-sha256", required=True)
    parser.add_argument("--controller-argv-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return result.stdout.strip()


def _git_identity(root: Path) -> tuple[str, bool]:
    sha = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    dirty = bool(_run_command(["git", "status", "--porcelain=v1"], cwd=root))
    return sha, dirty


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _extension_build_manifest(build_root: Path) -> list[dict[str, object]]:
    """Hash every durable input/output in one fresh JIT extension directory."""

    entries = []
    for path in sorted(build_root.rglob("*")):
        if not path.is_file() or path.name in {".ninja_deps", ".ninja_log", "lock"}:
            continue
        entries.append(
            {
                "path": str(path.relative_to(build_root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not entries:
        raise RuntimeError(f"extension build manifest is empty: {build_root}")
    return entries


def _verified_implementation(
    implementation_root: Path,
) -> tuple[object, type, dict[str, object]]:
    """Import, build, and bind the implementation to exact source/binary hashes."""

    module = importlib.import_module("b12x.comm.pcie.pcie_oneshot")
    module_path = Path(module.__file__).resolve()
    expected_module = (
        implementation_root / "flashinfer/experimental/b12x/comm/pcie/pcie_oneshot.py"
    ).resolve()
    if module_path != expected_module:
        raise RuntimeError(
            f"imported PCIe oneshot module {module_path} != {expected_module}"
        )
    cuda_source = module_path.with_suffix(".cu")
    if not cuda_source.is_file():
        raise RuntimeError(f"PCIe oneshot CUDA source is missing: {cuda_source}")

    extension = module._load_extension()
    extension_path = Path(extension.__file__).resolve()
    build_root = extension_path.parent
    cache_value = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not cache_value:
        raise RuntimeError(
            "TORCH_EXTENSIONS_DIR must name a fresh per-run extension cache"
        )
    expected_cache = Path(cache_value).resolve()
    if not extension_path.is_relative_to(expected_cache):
        raise RuntimeError(
            f"loaded extension {extension_path} is outside fresh cache {expected_cache}"
        )
    manifest = _extension_build_manifest(build_root)
    provenance = {
        "module_path": str(module_path),
        "module_sha256": _sha256(module_path),
        "cuda_source_path": str(cuda_source),
        "cuda_source_sha256": _sha256(cuda_source),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
        "extension_build_root": str(build_root),
        "extension_build_manifest": manifest,
        "extension_build_manifest_sha256": _json_sha256(manifest),
        "fresh_extension_cache": True,
        "post_run_verified": False,
    }
    return module, module.PCIeOneshotAllReducePool, provenance


def _verify_post_run_provenance(
    provenance: dict[str, object],
    *,
    implementation_root: Path,
    expected_git_sha: str,
) -> None:
    git_sha, git_dirty = _git_identity(implementation_root)
    if git_sha != expected_git_sha or git_dirty:
        raise RuntimeError("implementation Git identity changed during benchmark")
    checks = (
        ("module_path", "module_sha256"),
        ("cuda_source_path", "cuda_source_sha256"),
        ("extension_path", "extension_sha256"),
    )
    for path_key, digest_key in checks:
        if _sha256(Path(provenance[path_key])) != provenance[digest_key]:
            raise RuntimeError(f"{path_key} changed during benchmark")
    manifest = _extension_build_manifest(Path(provenance["extension_build_root"]))
    if manifest != provenance["extension_build_manifest"]:
        raise RuntimeError("extension build manifest changed during benchmark")
    if _json_sha256(manifest) != provenance["extension_build_manifest_sha256"]:
        raise RuntimeError("extension build manifest digest changed during benchmark")
    provenance["post_run_verified"] = True


def _cuda_version(call: Callable[[], tuple[object, int]]) -> int:
    result, version = call()
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA version query failed: {result}")
    return int(version)


def _nvidia_smi_query(fields: tuple[str, ...]) -> dict[str, object]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "fields": list(fields),
            "returncode": -1,
            "rows": [],
            "stderr": f"nvidia-smi timed out after {exc.timeout}s",
        }
    return {
        "fields": list(fields),
        "returncode": result.returncode,
        "rows": [
            [value.strip() for value in line.split(",")]
            for line in result.stdout.splitlines()
            if line.strip()
        ],
        "stderr": result.stderr.strip(),
    }


def _nvidia_smi_topology() -> dict[str, object]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"nvidia-smi topology timed out after {exc.timeout}s",
        }
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.rstrip(),
        "stderr": result.stderr.strip(),
    }


class _NvidiaSmiSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._phase = "unmarked"
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        sample = {
            "monotonic_ns": time.monotonic_ns(),
            "phase": self._phase,
            "query": _nvidia_smi_query(NVIDIA_SMI_FIELDS),
        }
        with self._lock:
            self.samples.append(sample)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread.start()

    def mark_phase(self, phase: str) -> None:
        self._phase = phase
        # A short timing phase must still contribute telemetry.
        self._sample()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            raise RuntimeError("nvidia-smi sampler thread did not stop within 15s")
        self._phase = "final"
        self._sample()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()


def _environment_toggles() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in EXPLICIT_ENV_KEYS
        or key.startswith("B12X_")
        or key.startswith("NCCL_")
        or key.startswith("TORCH_NCCL_")
    }


def _dim3_list(value: object) -> list[int]:
    return [int(value.x), int(value.y), int(value.z)]


def _graph_kernel_structure(
    graph: torch.cuda.CUDAGraph,
    *,
    expected_kernel_nodes: int,
) -> dict[str, object]:
    """Prove the candidate's 1x1 control node directly orders its worker."""

    graph_handle = graph.raw_cuda_graph()
    result, _, num_nodes = cudart.cudaGraphGetNodes(graph_handle)
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetNodes(size) failed: {result}")
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        num_nodes,
    )
    if result != cudart.cudaError_t.cudaSuccess or returned_nodes != num_nodes:
        raise RuntimeError(
            f"cudaGraphGetNodes(data) failed: {result}, {returned_nodes=}, {num_nodes=}"
        )
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    kernel_nodes = []
    for node in nodes[:num_nodes]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphNodeGetType failed: {result}")
        if node_type == kernel_type:
            kernel_nodes.append(node)
    if len(kernel_nodes) != expected_kernel_nodes:
        raise AssertionError(
            f"graph has {len(kernel_nodes)} kernel nodes, expected "
            f"{expected_kernel_nodes}"
        )

    edge_query = cudart.cudaGraphGetEdges(graph_handle)
    if edge_query[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetEdges(size) failed: {edge_query[0]}")
    num_edges = int(edge_query[-1])
    edges = cudart.cudaGraphGetEdges(graph_handle, num_edges)
    result, from_nodes, to_nodes, returned_edges = (
        edges[0],
        edges[1],
        edges[2],
        edges[-1],
    )
    if result != cudart.cudaError_t.cudaSuccess or returned_edges != num_edges:
        raise RuntimeError(
            f"cudaGraphGetEdges(data) failed: {result}, {returned_edges=}, {num_edges=}"
        )
    kernel_edges = []
    for source, destination in zip(
        from_nodes[:num_edges],
        to_nodes[:num_edges],
        strict=True,
    ):
        result, source_type = cudart.cudaGraphNodeGetType(source)
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"source cudaGraphNodeGetType failed: {result}")
        result, destination_type = cudart.cudaGraphNodeGetType(destination)
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"destination cudaGraphNodeGetType failed: {result}")
        if source_type == kernel_type and destination_type == kernel_type:
            kernel_edges.append((source, destination))

    def geometry(node: object) -> tuple[list[int], list[int]]:
        result, params = cudart.cudaGraphKernelNodeGetParams(node)
        if result != cudart.cudaError_t.cudaSuccess and hasattr(
            cudart, "cudaGraphNodeGetParams"
        ):
            fallback_result, node_params = cudart.cudaGraphNodeGetParams(node)
            if fallback_result != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(
                    "cudaGraphKernelNodeGetParams failed: "
                    f"{result}; cudaGraphNodeGetParams failed: {fallback_result}"
                )
            result, params = fallback_result, node_params.kernel
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphKernelNodeGetParams failed: {result}")
        return _dim3_list(params.gridDim), _dim3_list(params.blockDim)

    if expected_kernel_nodes == 2:
        if len(kernel_edges) != 1:
            raise AssertionError(
                "staged candidate graph must contain one direct control-to-worker "
                f"kernel edge, found {len(kernel_edges)}"
            )
        control, worker = map(geometry, kernel_edges[0])
        if control != ([1, 1, 1], [1, 1, 1]):
            raise AssertionError(f"control kernel is not <<<1,1>>>: {control}")
        if worker[0][0] <= 1 or worker[1][0] < 64:
            raise AssertionError(f"worker launch geometry is implausible: {worker}")
        direct_edge = True
    elif expected_kernel_nodes == 1:
        if kernel_edges:
            raise AssertionError("one-kernel baseline graph has a kernel edge")
        control = (None, None)
        worker = geometry(kernel_nodes[0])
        direct_edge = False
    else:
        raise AssertionError(
            "plain staged A/B contract supports only one-node baseline and "
            "two-node candidate graphs"
        )
    return {
        "kernel_nodes": len(kernel_nodes),
        "direct_control_worker_edge": direct_edge,
        "control_grid": control[0],
        "control_block": control[1],
        "worker_grid": worker[0],
        "worker_block": worker[1],
    }


def _measure(
    operation: Callable[[], None],
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    iters: int,
) -> tuple[float, list[float]]:
    cold_start = torch.cuda.Event(enable_timing=True)
    cold_end = torch.cuda.Event(enable_timing=True)
    cold_start.record(stream)
    operation()
    cold_end.record(stream)
    stream.synchronize()
    cold_us = float(cold_start.elapsed_time(cold_end) * 1000)

    for _ in range(warmup):
        operation()
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        operation()
        end.record(stream)
    stream.synchronize()
    samples = [
        float(start.elapsed_time(end) * 1000)
        for start, end in zip(starts, ends, strict=True)
    ]
    return cold_us, samples


def _mutating_replay_preflight(
    operation: Callable[[], None],
    *,
    inp: torch.Tensor,
    out: torch.Tensor,
    stream: torch.cuda.Stream,
    rank: int,
    world_size: int,
    replays: int,
) -> None:
    """Expose stale-slot reuse by validating changing data on both slot parities."""

    if replays < 4:
        raise ValueError("mutating preflight requires at least four replays")
    rank_sum = world_size * (world_size + 1) // 2
    for iteration in range(replays):
        multiplier = iteration + 1
        with torch.cuda.stream(stream):
            inp.fill_(multiplier * (rank + 1))
            operation()
        stream.synchronize()
        torch.testing.assert_close(
            out,
            torch.full_like(out, multiplier * rank_sum),
            rtol=0,
            atol=0,
        )


def _summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot summarize an empty latency vector")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "mean_us": sum(ordered) / len(ordered),
        "p50_us": percentile(0.50),
        "p95_us": percentile(0.95),
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _gather_rank_metrics(
    eager_cold_us: float,
    eager_samples: list[float],
    graph_cold_us: float,
    graph_samples: list[float],
    graph_structure: dict[str, object],
    device: torch.device,
) -> list[dict[str, object]]:
    if len(eager_samples) != len(graph_samples) or not eager_samples:
        raise ValueError("eager/graph latency vectors must be non-empty and aligned")
    control_grid = graph_structure["control_grid"] or [0, 0, 0]
    control_block = graph_structure["control_block"] or [0, 0, 0]
    structure_values = [
        float(graph_structure["kernel_nodes"]),
        float(bool(graph_structure["direct_control_worker_edge"])),
        *map(float, control_grid),
        *map(float, control_block),
        *map(float, graph_structure["worker_grid"]),
        *map(float, graph_structure["worker_block"]),
    ]
    if len(structure_values) != 14:
        raise AssertionError("graph structure encoding must contain 14 values")
    local = torch.tensor(
        [
            eager_cold_us,
            *eager_samples,
            graph_cold_us,
            *graph_samples,
            *structure_values,
        ],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)

    result = []
    for rank, values_tensor in enumerate(gathered):
        values = values_tensor.cpu().tolist()
        iters = len(eager_samples)
        rank_eager_samples = values[1 : 1 + iters]
        graph_cold_index = 1 + iters
        rank_graph_samples = values[graph_cold_index + 1 : graph_cold_index + 1 + iters]
        structure = values[graph_cold_index + 1 + iters :]
        eager_summary = _summary(rank_eager_samples)
        graph_summary = _summary(rank_graph_samples)
        has_control = bool(int(structure[1]))
        result.append(
            {
                "rank": rank,
                "eager": {
                    "cold_us": values[0],
                    **eager_summary,
                    "samples_us": rank_eager_samples,
                },
                "graph": {
                    "cold_us": values[graph_cold_index],
                    **graph_summary,
                    "samples_us": rank_graph_samples,
                },
                "graph_structure": {
                    "kernel_nodes": int(structure[0]),
                    "direct_control_worker_edge": has_control,
                    "control_grid": [int(v) for v in structure[2:5]]
                    if has_control
                    else None,
                    "control_block": [int(v) for v in structure[5:8]]
                    if has_control
                    else None,
                    "worker_grid": [int(v) for v in structure[8:11]],
                    "worker_block": [int(v) for v in structure[11:14]],
                },
            }
        )
    return result


def _distributed_critical_path(
    per_rank: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Summarize max-rank latency at each aligned timed iteration."""

    if not per_rank:
        raise ValueError("distributed critical path requires at least one rank")
    result: dict[str, object] = {}
    for mode in ("eager", "graph"):
        rank_vectors = [list(row[mode]["samples_us"]) for row in per_rank]
        lengths = {len(samples) for samples in rank_vectors}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"{mode} rank latency vectors are empty or misaligned")
        critical_samples = [
            max(rank_samples[index] for rank_samples in rank_vectors)
            for index in range(next(iter(lengths)))
        ]
        result[mode] = {
            "cold_us": max(float(row[mode]["cold_us"]) for row in per_rank),
            **_summary(critical_samples),
            "samples_us": critical_samples,
        }
    return result


def _device_metadata() -> list[dict[str, object]]:
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "uuid": str(getattr(props, "uuid", "unavailable")),
                "compute_capability": [props.major, props.minor],
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return devices


def _validate_correctness(correctness: object) -> None:
    if not isinstance(correctness, dict) or set(correctness) != CORRECTNESS_KEYS:
        raise ValueError("benchmark correctness schema mismatch")
    if correctness["verdict"] != "pass":
        raise ValueError(f"benchmark correctness did not pass: {correctness}")


def _validate_record(
    record: dict[str, object],
    contract: dict[str, object],
    *,
    expected_graph_kernel_nodes: int,
) -> None:
    if set(record) != TOP_LEVEL_KEYS:
        raise ValueError(
            f"benchmark schema keys differ: {set(record) ^ TOP_LEVEL_KEYS}"
        )
    if record["schema"] != {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}:
        raise ValueError(f"unsupported benchmark schema: {record['schema']}")
    _validate_correctness(record["correctness"])
    if set(record["contract"]) != CONTRACT_KEYS or record["contract"] != contract:
        raise ValueError(
            f"benchmark contract mismatch: {record['contract']} != {contract}"
        )
    scope = record["scope"]
    if set(scope) != SCOPE_KEYS:
        raise ValueError("benchmark scope schema mismatch")
    if (
        scope["operation"] != "plain_oneshot_all_reduce"
        or scope["staging"] != "double_buffered_eager_ipc"
        or int(scope["size_bytes"]) != int(contract["numel"]) * torch.bfloat16.itemsize
        or "net algorithm" not in str(scope["comparison_semantics"]).lower()
        or scope["excluded_paths"]
        != ["fused_add_rms_norm", "twoshot_reduce_scatter", "twoshot_all_gather"]
    ):
        raise ValueError(
            f"benchmark scope is not the narrow staged-path contract: {scope}"
        )
    if set(record["identity"]) != IDENTITY_KEYS:
        raise ValueError("benchmark identity schema mismatch")
    if set(record["provenance"]) != PROVENANCE_KEYS:
        raise ValueError("benchmark provenance schema mismatch")
    provenance = record["provenance"]
    if not provenance["fresh_extension_cache"] or not provenance["post_run_verified"]:
        raise ValueError("benchmark extension provenance was not freshly verified")
    manifest = provenance["extension_build_manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("benchmark extension build manifest is empty")
    for entry in manifest:
        if set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("benchmark extension manifest entry schema mismatch")
    if _json_sha256(manifest) != provenance["extension_build_manifest_sha256"]:
        raise ValueError("benchmark extension manifest digest mismatch")
    if set(record["software"]) != SOFTWARE_KEYS:
        raise ValueError("benchmark software schema mismatch")
    if set(record["hardware"]) != HARDWARE_KEYS:
        raise ValueError("benchmark hardware schema mismatch")
    if not isinstance(record["environment"], dict):
        raise ValueError("benchmark environment must be an object")
    if not isinstance(record["per_rank"], list) or len(record["per_rank"]) != int(
        contract["world_size"]
    ):
        raise ValueError("benchmark per-rank cardinality mismatch")
    hardware = record["hardware"]
    if not isinstance(hardware["devices"], list) or not hardware["devices"]:
        raise ValueError("benchmark device inventory must be a non-empty list")
    for device in hardware["devices"]:
        if set(device) != DEVICE_KEYS:
            raise ValueError("benchmark device schema mismatch")
    identity_query = hardware["nvidia_smi_identity"]
    if set(identity_query) != NVIDIA_QUERY_KEYS:
        raise ValueError("benchmark NVIDIA identity schema mismatch")
    if identity_query["fields"] != list(NVIDIA_IDENTITY_FIELDS):
        raise ValueError("benchmark NVIDIA identity field contract mismatch")
    if identity_query["returncode"] != 0:
        raise ValueError(f"benchmark NVIDIA identity query failed: {identity_query}")
    if len(identity_query["rows"]) != len(hardware["devices"]):
        raise ValueError("benchmark NVIDIA identity/device cardinality mismatch")
    if any(len(row) != len(NVIDIA_IDENTITY_FIELDS) for row in identity_query["rows"]):
        raise ValueError("benchmark NVIDIA identity row schema mismatch")
    if hardware["nvidia_smi_sample_fields"] != list(NVIDIA_SMI_FIELDS):
        raise ValueError("benchmark NVIDIA sample field contract mismatch")
    if (
        not isinstance(hardware["nvidia_smi_samples"], list)
        or not hardware["nvidia_smi_samples"]
    ):
        raise ValueError("benchmark NVIDIA samples must be a non-empty list")
    successful_samples = 0
    successful_phases: set[str] = set()
    for sample in hardware["nvidia_smi_samples"]:
        if set(sample) != NVIDIA_SAMPLE_KEYS:
            raise ValueError("benchmark NVIDIA sample schema mismatch")
        if set(sample["query"]) != NVIDIA_QUERY_KEYS:
            raise ValueError("benchmark NVIDIA sample query schema mismatch")
        if sample["query"]["fields"] != list(NVIDIA_SMI_FIELDS):
            raise ValueError("benchmark NVIDIA sample query field mismatch")
        if sample["query"]["returncode"] == 0:
            successful_samples += 1
            if not sample["query"]["rows"]:
                raise ValueError("successful NVIDIA telemetry sample has zero rows")
            successful_phases.add(str(sample["phase"]))
            if any(
                len(row) != len(NVIDIA_SMI_FIELDS) for row in sample["query"]["rows"]
            ):
                raise ValueError("benchmark NVIDIA sample row schema mismatch")
            if len(sample["query"]["rows"]) != len(hardware["devices"]):
                raise ValueError("benchmark NVIDIA sample/device cardinality mismatch")
    if successful_samples == 0:
        raise ValueError("benchmark collected no successful NVIDIA telemetry samples")
    required_phases = {"mutating_preflight", "eager_timing", "graph_timing"}
    if not required_phases <= successful_phases:
        raise ValueError(
            "benchmark lacks successful phase telemetry: "
            f"{required_phases - successful_phases}"
        )
    topology = hardware["nvidia_smi_topology"]
    if set(topology) != NVIDIA_TOPOLOGY_KEYS:
        raise ValueError("benchmark NVIDIA topology schema mismatch")
    if topology["returncode"] != 0 or not topology["stdout"]:
        raise ValueError(f"benchmark NVIDIA topology query failed: {topology}")
    for row in record["per_rank"]:
        if set(row) != {"rank", "eager", "graph", "graph_structure"}:
            raise ValueError("per-rank schema mismatch")
        if set(row["eager"]) != RANK_MODE_KEYS:
            raise ValueError("eager metric schema mismatch")
        if set(row["graph"]) != RANK_MODE_KEYS:
            raise ValueError("graph metric schema mismatch")
        for mode in ("eager", "graph"):
            samples = row[mode]["samples_us"]
            if not isinstance(samples, list) or len(samples) != int(contract["iters"]):
                raise ValueError(f"per-rank {mode} sample vector length mismatch")
            if any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in samples
            ):
                raise ValueError(f"per-rank {mode} samples must be finite and positive")
            expected_summary = _summary([float(value) for value in samples])
            for key, expected in expected_summary.items():
                if float(row[mode][key]) != expected:
                    raise ValueError(f"per-rank {mode} {key} summary mismatch")
        structure = row["graph_structure"]
        if set(structure) != GRAPH_STRUCTURE_KEYS:
            raise ValueError("graph structure schema mismatch")
        if int(structure["kernel_nodes"]) != expected_graph_kernel_nodes:
            raise ValueError("graph kernel-node count mismatch")
        if expected_graph_kernel_nodes == 2:
            if (
                structure["direct_control_worker_edge"] is not True
                or structure["control_grid"] != [1, 1, 1]
                or structure["control_block"] != [1, 1, 1]
            ):
                raise ValueError("candidate graph lacks direct <<<1,1>>> control edge")
        elif (
            structure["direct_control_worker_edge"] is not False
            or structure["control_grid"] is not None
            or structure["control_block"] is not None
        ):
            raise ValueError("baseline graph unexpectedly records a control node")
    if {int(row["rank"]) for row in record["per_rank"]} != set(
        range(int(contract["world_size"]))
    ):
        raise ValueError("per-rank identities do not match world size")
    critical_path = record["distributed_critical_path"]
    if set(critical_path) != {"eager", "graph"}:
        raise ValueError("distributed critical-path mode schema mismatch")
    for mode in ("eager", "graph"):
        if set(critical_path[mode]) != CRITICAL_MODE_KEYS:
            raise ValueError(f"distributed {mode} critical-path schema mismatch")
    expected_critical_path = _distributed_critical_path(record["per_rank"])
    if critical_path != expected_critical_path:
        raise ValueError(
            "distributed critical path was not derived from aligned rank maxima"
        )


def main() -> None:
    args = _parse_args()
    if (
        args.iters <= 0
        or args.warmup < 0
        or args.numel != 32768
        or args.run_index < 0
        or args.expected_graph_kernel_nodes not in (1, 2)
        or args.preflight_replays < 4
        or args.distributed_timeout_seconds <= 0
    ):
        raise ValueError(
            "this narrow harness requires 32,768 BF16 elements (64 KiB), "
            "at least four preflight replays, one/two graph nodes, positive "
            "iterations/timeout, and non-negative warmup/run-index"
        )
    if args.sampler_interval <= 0:
        raise ValueError("sampler interval must be positive")

    implementation_root = args.implementation_root.resolve()
    git_sha, git_dirty = _git_identity(implementation_root)
    if git_sha != args.expected_git_sha:
        raise ValueError(f"implementation SHA {git_sha} != {args.expected_git_sha}")
    if git_dirty and not args.allow_dirty:
        raise ValueError(f"implementation worktree is dirty: {implementation_root}")

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=args.distributed_timeout_seconds),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    _, pool_type, provenance = _verified_implementation(implementation_root)
    contract = {
        "world_size": world_size,
        "numel": args.numel,
        "dtype": DTYPE_NAME,
        "warmup": args.warmup,
        "iters": args.iters,
        "preflight_replays": args.preflight_replays,
    }
    started_at = datetime.now(timezone.utc).isoformat()
    sampler = _NvidiaSmiSampler(args.sampler_interval) if rank == 0 else None

    pool = None
    try:
        nbytes = args.numel * torch.bfloat16.itemsize
        pool = pool_type.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_input_bytes=nbytes,
            max_size=nbytes,
        )
        eager_channel_id = "eager:control-node-benchmark"
        graph_channel_id = "graph:control-node-benchmark"
        pool.prepare_channels((eager_channel_id, graph_channel_id))
        stream = torch.cuda.Stream(device=device)
        channel = pool.for_stream(stream, channel_id=eager_channel_id)
        eager_inp = torch.full(
            (args.numel,),
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        eager_out = torch.empty_like(eager_inp)

        def eager_operation() -> None:
            with torch.cuda.stream(stream):
                channel.all_reduce(eager_inp, out=eager_out)

        graph_inp = torch.full_like(eager_inp, rank + 1)
        graph_out = torch.empty_like(graph_inp)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with (
            pool.capture(stream, channel_id=graph_channel_id) as graph_channel,
            torch.cuda.graph(
                graph,
                stream=stream,
            ),
        ):
            graph_channel.all_reduce(graph_inp, out=graph_out)
        graph_structure = _graph_kernel_structure(
            graph,
            expected_kernel_nodes=args.expected_graph_kernel_nodes,
        )

        def graph_operation() -> None:
            with torch.cuda.stream(stream):
                graph.replay()

        # JIT compilation and graph construction are complete. Telemetry starts
        # here and every correctness/timing phase receives an explicit marker.
        if sampler is not None:
            sampler.start()
            sampler.mark_phase("mutating_preflight")
        _mutating_replay_preflight(
            eager_operation,
            inp=eager_inp,
            out=eager_out,
            stream=stream,
            rank=rank,
            world_size=world_size,
            replays=args.preflight_replays,
        )
        _mutating_replay_preflight(
            graph_operation,
            inp=graph_inp,
            out=graph_out,
            stream=stream,
            rank=rank,
            world_size=world_size,
            replays=args.preflight_replays,
        )

        with torch.cuda.stream(stream):
            eager_inp.fill_(rank + 1)
        stream.synchronize()
        dist.barrier()
        if sampler is not None:
            sampler.mark_phase("eager_timing")
        eager_cold_us, eager_samples = _measure(
            eager_operation,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
        )
        expected = world_size * (world_size + 1) // 2
        torch.testing.assert_close(
            eager_out,
            torch.full_like(eager_out, expected),
            rtol=0,
            atol=0,
        )

        with torch.cuda.stream(stream):
            graph_inp.fill_(rank + 1)
        stream.synchronize()
        dist.barrier()
        if sampler is not None:
            sampler.mark_phase("graph_timing")
        graph_cold_us, graph_samples = _measure(
            graph_operation,
            stream=stream,
            warmup=args.warmup,
            iters=args.iters,
        )
        torch.testing.assert_close(
            graph_out,
            torch.full_like(graph_out, expected),
            rtol=0,
            atol=0,
        )

        per_rank = _gather_rank_metrics(
            eager_cold_us,
            eager_samples,
            graph_cold_us,
            graph_samples,
            graph_structure,
            device,
        )
        observed_nodes = {
            int(row["graph_structure"]["kernel_nodes"]) for row in per_rank
        }
        if observed_nodes != {args.expected_graph_kernel_nodes}:
            raise AssertionError(
                f"rank graph-node counts {observed_nodes} do not match "
                f"{args.expected_graph_kernel_nodes}"
            )
        torch.cuda.synchronize(device)
        dist.barrier()
        _verify_post_run_provenance(
            provenance,
            implementation_root=implementation_root,
            expected_git_sha=args.expected_git_sha,
        )

        if sampler is not None:
            sampler.mark_phase("provenance_and_finalize")
            sampler.stop()
        if rank == 0:
            harness_path = Path(__file__).resolve()
            worker_argv = list(sys.argv)
            record = {
                "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
                "contract": contract,
                "scope": {
                    "operation": "plain_oneshot_all_reduce",
                    "staging": "double_buffered_eager_ipc",
                    "size_bytes": nbytes,
                    "comparison_semantics": (
                        "net algorithm comparison between public baseline and "
                        "candidate; not isolated launch overhead"
                    ),
                    "excluded_paths": [
                        "fused_add_rms_norm",
                        "twoshot_reduce_scatter",
                        "twoshot_all_gather",
                    ],
                },
                "correctness": {"verdict": "pass"},
                "identity": {
                    "label": args.label,
                    "variant": args.variant,
                    "run_index": args.run_index,
                    "implementation_root": str(implementation_root),
                    "implementation_git_sha": git_sha,
                    "implementation_git_dirty": git_dirty,
                    "harness_sha256": _sha256(harness_path),
                    "controller_sha256": args.controller_sha256,
                    "controller_argv_sha256": args.controller_argv_sha256,
                    "worker_argv": worker_argv,
                    "worker_argv_sha256": _json_sha256(worker_argv),
                    "started_at_utc": started_at,
                },
                "provenance": provenance,
                "software": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cuda_runtime_version": _cuda_version(cudart.cudaRuntimeGetVersion),
                    "cuda_driver_version": _cuda_version(cudart.cudaDriverGetVersion),
                    "nvcc_version": _run_command(["nvcc", "--version"]),
                },
                "hardware": {
                    "hostname": socket.gethostname(),
                    "devices": _device_metadata(),
                    "nvidia_smi_identity": _nvidia_smi_query(NVIDIA_IDENTITY_FIELDS),
                    "nvidia_smi_sample_fields": list(NVIDIA_SMI_FIELDS),
                    "nvidia_smi_samples": sampler.samples,
                    "nvidia_smi_topology": _nvidia_smi_topology(),
                },
                "environment": _environment_toggles(),
                "per_rank": per_rank,
                "distributed_critical_path": _distributed_critical_path(per_rank),
            }
            _validate_record(
                record,
                contract,
                expected_graph_kernel_nodes=args.expected_graph_kernel_nodes,
            )
            rendered = json.dumps(record, indent=2, sort_keys=True)
            print(rendered, flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        if sampler is not None and sampler.is_alive:
            sampler.stop()
        if pool is not None:
            pool.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
