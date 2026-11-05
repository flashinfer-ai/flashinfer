"""Run repeated immutable AB/BA measurements with one external harness.

The measurement script lives outside both implementation worktrees and is
hashed into every record. Each pair reverses execution order (AB, then BA) to
reduce monotonic thermal/clock drift. Raw records are schema- and
contract-validated before the aggregate report is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
from contextlib import suppress
from pathlib import Path


RUN_SCHEMA = {"name": "b12x.pcie_oneshot_control_node.run", "version": 3}
SUITE_SCHEMA = {"name": "b12x.pcie_oneshot_control_node.ab_suite", "version": 3}
RUN_TOP_LEVEL_KEYS = {
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
METRICS = ("mean_us", "p50_us", "p95_us")
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
METRIC_KEYS = {"cold_us", "mean_us", "p50_us", "p95_us", "min_us", "max_us"}
RANK_MODE_KEYS = METRIC_KEYS | {"samples_us"}
CRITICAL_MODE_KEYS = METRIC_KEYS | {"samples_us"}
GRAPH_STRUCTURE_KEYS = {
    "kernel_nodes",
    "direct_control_worker_edge",
    "control_grid",
    "control_block",
    "worker_grid",
    "worker_block",
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
NVIDIA_IDENTITY_FIELDS = ["index", "uuid", "name", "driver_version"]
NVIDIA_SMI_FIELDS = [
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
]
SUITE_TOP_LEVEL_KEYS = {
    "schema",
    "contract",
    "controller",
    "harness",
    "implementations",
    "sequence",
    "records",
    "paired_deltas",
    "paired_summary",
    "aggregate",
}
CONTROLLER_KEYS = {"source_path", "sha256", "argv", "argv_sha256"}
HARNESS_KEYS = {
    "source_path",
    "frozen_path",
    "sha256",
    "identical_for_all_runs",
}
IMPLEMENTATION_KEYS = {"root", "git_sha"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-worktree", type=Path, required=True)
    parser.add_argument("--candidate-worktree", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--numel",
        type=int,
        default=32768,
        help="fixed at 32768 BF16 elements (64 KiB); other values are rejected",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--sampler-interval", type=float, default=0.2)
    parser.add_argument("--preflight-replays", type=int, default=4)
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--distributed-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--harness",
        type=Path,
        default=Path(__file__).with_name("benchmark_pcie_oneshot_control_node.py"),
    )
    return parser.parse_args()


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> str:
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
        raise TimeoutError(
            f"command exceeded {timeout:.1f}s and its process group was terminated: "
            f"{' '.join(args)}\n{stdout}"
        ) from None
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(args)}\n{stdout}"
        )
    return stdout.strip()


def _git_sha(worktree: Path) -> str:
    sha = _run(["git", "rev-parse", "HEAD"], cwd=worktree)
    dirty = _run(["git", "status", "--porcelain=v1"], cwd=worktree)
    if dirty:
        raise ValueError(f"benchmark worktree is dirty: {worktree}\n{dirty}")
    return sha


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


def _expected_contract(args: argparse.Namespace) -> dict[str, object]:
    return {
        "world_size": args.world_size,
        "numel": args.numel,
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "iters": args.iters,
        "preflight_replays": args.preflight_replays,
    }


def _summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot summarize an empty latency vector")
    ordered = sorted(float(value) for value in samples)

    def percentile(fraction: float) -> float:
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "mean_us": sum(ordered) / len(ordered),
        "p50_us": percentile(0.50),
        "p95_us": percentile(0.95),
        "min_us": ordered[0],
        "max_us": ordered[-1],
    }


def _distributed_critical_path(
    per_rank: list[dict[str, object]],
) -> dict[str, object]:
    result = {}
    for mode in ("eager", "graph"):
        vectors = [list(row[mode]["samples_us"]) for row in per_rank]
        lengths = {len(vector) for vector in vectors}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError(f"{mode} rank vectors are empty or misaligned")
        critical = [
            max(float(vector[index]) for vector in vectors)
            for index in range(next(iter(lengths)))
        ]
        result[mode] = {
            "cold_us": max(float(row[mode]["cold_us"]) for row in per_rank),
            **_summary(critical),
            "samples_us": critical,
        }
    return result


def _validate_correctness(correctness: object) -> None:
    if not isinstance(correctness, dict) or set(correctness) != CORRECTNESS_KEYS:
        raise ValueError("run correctness schema mismatch")
    if correctness["verdict"] != "pass":
        raise ValueError(f"run correctness did not pass: {correctness}")


def _validate_run(
    record: dict[str, object],
    *,
    contract: dict[str, object],
    variant: str,
    sha: str,
    run_index: int,
    harness_sha256: str,
    expected_nodes: int,
    implementation_root: Path,
    controller_sha256: str,
    controller_argv_sha256: str,
    extension_dir: Path,
) -> None:
    if set(record) != RUN_TOP_LEVEL_KEYS:
        raise ValueError(f"run schema keys differ: {set(record) ^ RUN_TOP_LEVEL_KEYS}")
    if record["schema"] != RUN_SCHEMA:
        raise ValueError(f"run schema mismatch: {record['schema']}")
    _validate_correctness(record["correctness"])
    if set(record["contract"]) != CONTRACT_KEYS or record["contract"] != contract:
        raise ValueError(f"run contract mismatch: {record['contract']} != {contract}")
    scope = record["scope"]
    if set(scope) != SCOPE_KEYS:
        raise ValueError("run scope schema mismatch")
    if (
        scope["operation"] != "plain_oneshot_all_reduce"
        or scope["staging"] != "double_buffered_eager_ipc"
        or int(scope["size_bytes"]) != int(contract["numel"]) * 2
        or "net algorithm" not in str(scope["comparison_semantics"]).lower()
        or scope["excluded_paths"]
        != ["fused_add_rms_norm", "twoshot_reduce_scatter", "twoshot_all_gather"]
    ):
        raise ValueError(f"run scope is too broad or malformed: {scope}")
    identity = record["identity"]
    if set(identity) != IDENTITY_KEYS:
        raise ValueError("run identity schema mismatch")
    expected_identity = {
        "label": f"{variant}-{sha[:12]}-{run_index}",
        "variant": variant,
        "run_index": run_index,
        "implementation_root": str(implementation_root),
        "implementation_git_sha": sha,
        "implementation_git_dirty": False,
        "harness_sha256": harness_sha256,
        "controller_sha256": controller_sha256,
        "controller_argv_sha256": controller_argv_sha256,
    }
    for key, expected in expected_identity.items():
        if identity[key] != expected:
            raise ValueError(f"identity {key}={identity[key]!r}, expected {expected!r}")
    if _json_sha256(identity["worker_argv"]) != identity["worker_argv_sha256"]:
        raise ValueError("worker argv digest mismatch")
    provenance = record["provenance"]
    if set(provenance) != PROVENANCE_KEYS:
        raise ValueError("run provenance schema mismatch")
    expected_module = (
        implementation_root / "flashinfer/experimental/b12x/comm/pcie/pcie_oneshot.py"
    ).resolve()
    expected_source = expected_module.with_suffix(".cu")
    if Path(provenance["module_path"]).resolve() != expected_module:
        raise ValueError("run imported module does not belong to implementation root")
    if Path(provenance["cuda_source_path"]).resolve() != expected_source:
        raise ValueError("run CUDA source does not belong to implementation root")
    if (
        not Path(provenance["extension_path"])
        .resolve()
        .is_relative_to(extension_dir.resolve())
    ):
        raise ValueError("run extension binary is outside its fresh cache")
    if not provenance["fresh_extension_cache"] or not provenance["post_run_verified"]:
        raise ValueError("run extension provenance was not freshly post-verified")
    for path_key, digest_key in (
        ("module_path", "module_sha256"),
        ("cuda_source_path", "cuda_source_sha256"),
        ("extension_path", "extension_sha256"),
    ):
        if _sha256(Path(provenance[path_key])) != provenance[digest_key]:
            raise ValueError(f"run {path_key} digest mismatch")
    manifest = provenance["extension_build_manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("run extension build manifest is empty")
    for entry in manifest:
        if set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("run extension manifest entry schema mismatch")
    if _json_sha256(manifest) != provenance["extension_build_manifest_sha256"]:
        raise ValueError("run extension build manifest digest mismatch")
    if set(record["software"]) != SOFTWARE_KEYS:
        raise ValueError("run software schema mismatch")
    if set(record["hardware"]) != HARDWARE_KEYS:
        raise ValueError("run hardware schema mismatch")
    if not isinstance(record["environment"], dict):
        raise ValueError("run environment must be an object")
    if not isinstance(record["per_rank"], list) or len(record["per_rank"]) != int(
        contract["world_size"]
    ):
        raise ValueError("run per-rank cardinality mismatch")
    hardware = record["hardware"]
    if not isinstance(hardware["devices"], list) or not hardware["devices"]:
        raise ValueError("run device inventory must be a non-empty list")
    for device in hardware["devices"]:
        if set(device) != DEVICE_KEYS:
            raise ValueError("run device schema mismatch")
    identity_query = hardware["nvidia_smi_identity"]
    if set(identity_query) != NVIDIA_QUERY_KEYS:
        raise ValueError("run NVIDIA identity schema mismatch")
    if identity_query["fields"] != NVIDIA_IDENTITY_FIELDS:
        raise ValueError("run NVIDIA identity field contract mismatch")
    if identity_query["returncode"] != 0:
        raise ValueError(f"run NVIDIA identity query failed: {identity_query}")
    if len(identity_query["rows"]) != len(hardware["devices"]):
        raise ValueError("run NVIDIA identity/device cardinality mismatch")
    if any(len(row) != len(NVIDIA_IDENTITY_FIELDS) for row in identity_query["rows"]):
        raise ValueError("run NVIDIA identity row schema mismatch")
    if hardware["nvidia_smi_sample_fields"] != NVIDIA_SMI_FIELDS:
        raise ValueError("run NVIDIA sample field contract mismatch")
    if (
        not isinstance(hardware["nvidia_smi_samples"], list)
        or not hardware["nvidia_smi_samples"]
    ):
        raise ValueError("run NVIDIA samples must be a non-empty list")
    successful_samples = 0
    successful_phases: set[str] = set()
    for sample in hardware["nvidia_smi_samples"]:
        if set(sample) != NVIDIA_SAMPLE_KEYS:
            raise ValueError("run NVIDIA sample schema mismatch")
        if set(sample["query"]) != NVIDIA_QUERY_KEYS:
            raise ValueError("run NVIDIA sample query schema mismatch")
        if sample["query"]["fields"] != NVIDIA_SMI_FIELDS:
            raise ValueError("run NVIDIA sample query field mismatch")
        if sample["query"]["returncode"] == 0:
            successful_samples += 1
            if not sample["query"]["rows"]:
                raise ValueError("successful NVIDIA sample has zero rows")
            successful_phases.add(str(sample["phase"]))
            if any(
                len(row) != len(NVIDIA_SMI_FIELDS) for row in sample["query"]["rows"]
            ):
                raise ValueError("run NVIDIA sample row schema mismatch")
            if len(sample["query"]["rows"]) != len(hardware["devices"]):
                raise ValueError("run NVIDIA sample/device cardinality mismatch")
    if successful_samples == 0:
        raise ValueError("run collected no successful NVIDIA telemetry samples")
    required_phases = {"mutating_preflight", "eager_timing", "graph_timing"}
    if not required_phases <= successful_phases:
        raise ValueError(
            f"run lacks successful phase telemetry: {required_phases - successful_phases}"
        )
    topology = hardware["nvidia_smi_topology"]
    if set(topology) != NVIDIA_TOPOLOGY_KEYS:
        raise ValueError("run NVIDIA topology schema mismatch")
    if topology["returncode"] != 0 or not topology["stdout"]:
        raise ValueError(f"run NVIDIA topology query failed: {topology}")
    for row in record["per_rank"]:
        if set(row) != {"rank", "eager", "graph", "graph_structure"}:
            raise ValueError("run per-rank schema mismatch")
        if set(row["eager"]) != RANK_MODE_KEYS:
            raise ValueError("run per-rank eager metric schema mismatch")
        if set(row["graph"]) != RANK_MODE_KEYS:
            raise ValueError("run per-rank graph metric schema mismatch")
        for mode in ("eager", "graph"):
            samples = row[mode]["samples_us"]
            if not isinstance(samples, list) or len(samples) != int(contract["iters"]):
                raise ValueError(f"run per-rank {mode} sample length mismatch")
            if any(
                not math.isfinite(float(value)) or float(value) <= 0
                for value in samples
            ):
                raise ValueError(f"run per-rank {mode} samples are invalid")
            expected_summary = _summary(samples)
            for key, expected in expected_summary.items():
                if float(row[mode][key]) != expected:
                    raise ValueError(f"run per-rank {mode} {key} mismatch")
        structure = row["graph_structure"]
        if set(structure) != GRAPH_STRUCTURE_KEYS:
            raise ValueError("run graph structure schema mismatch")
        if int(structure["kernel_nodes"]) != expected_nodes:
            raise ValueError("observed graph-node count differs across ranks")
        if expected_nodes == 2:
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
        raise ValueError("run rank identities do not match world size")
    critical_path = record["distributed_critical_path"]
    if set(critical_path) != {"eager", "graph"}:
        raise ValueError("run distributed critical-path mode schema mismatch")
    for mode in ("eager", "graph"):
        if set(critical_path[mode]) != CRITICAL_MODE_KEYS:
            raise ValueError(f"run distributed {mode} schema mismatch")
    if critical_path != _distributed_critical_path(record["per_rank"]):
        raise ValueError("run critical path is not the aligned per-iteration rank MAX")


def _outside_worktrees(path: Path, roots: tuple[Path, ...], label: str) -> Path:
    resolved = path.resolve()
    for root in roots:
        if resolved == root or resolved.is_relative_to(root):
            raise ValueError(f"{label} must be outside measured worktree {root}")
    return resolved


def _stable_environment(record: dict[str, object]) -> dict[str, object]:
    environment = dict(record["environment"])
    environment.pop("TORCH_EXTENSIONS_DIR", None)
    return environment


def _stable_hardware(record: dict[str, object]) -> dict[str, object]:
    hardware = record["hardware"]
    return {
        "hostname": hardware["hostname"],
        "devices": hardware["devices"],
        "nvidia_smi_identity": hardware["nvidia_smi_identity"],
        "nvidia_smi_sample_fields": hardware["nvidia_smi_sample_fields"],
        "nvidia_smi_topology": hardware["nvidia_smi_topology"],
    }


def _paired_deltas(records: list[dict[str, object]]) -> list[dict[str, object]]:
    deltas = []
    for pair_index in range(len(records) // 2):
        pair = records[2 * pair_index : 2 * pair_index + 2]
        by_variant = {record["identity"]["variant"]: record for record in pair}
        if set(by_variant) != {"baseline", "candidate"}:
            raise ValueError(f"pair {pair_index} does not contain one A and one B")
        delta = {"pair_index": pair_index, "metrics": {}}
        for mode in ("eager", "graph"):
            delta["metrics"][mode] = {}
            for metric in METRICS:
                before = float(
                    by_variant["baseline"]["distributed_critical_path"][mode][metric]
                )
                after = float(
                    by_variant["candidate"]["distributed_critical_path"][mode][metric]
                )
                delta["metrics"][mode][metric] = {
                    "baseline_us": before,
                    "candidate_us": after,
                    "delta_us": after - before,
                    "delta_percent": (after / before - 1) * 100,
                }
        deltas.append(delta)
    return deltas


def _paired_summary(deltas: list[dict[str, object]]) -> dict[str, object]:
    """Descriptive paired uncertainty; this harness makes no significance claim."""

    result = {}
    for mode in ("eager", "graph"):
        result[mode] = {}
        for metric in METRICS:
            values = [
                float(delta["metrics"][mode][metric]["delta_us"]) for delta in deltas
            ]
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            margin = 1.96 * stdev / math.sqrt(len(values)) if len(values) > 1 else None
            result[mode][metric] = {
                "pairs": len(values),
                "mean_delta_us": mean,
                "sample_stdev_us": stdev,
                "normal_approx_95ci_us": [mean - margin, mean + margin]
                if margin is not None
                else None,
                "interpretation": "descriptive_only",
            }
    return result


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for variant in ("baseline", "candidate"):
        variant_records = [
            record for record in records if record["identity"]["variant"] == variant
        ]
        result[variant] = {}
        for mode in ("eager", "graph"):
            result[variant][mode] = {}
            for metric in METRICS:
                values = [
                    float(record["distributed_critical_path"][mode][metric])
                    for record in variant_records
                ]
                result[variant][mode][metric] = {
                    "mean_us": statistics.mean(values),
                    "median_us": statistics.median(values),
                    "min_us": min(values),
                    "max_us": max(values),
                }
    return result


def main() -> None:
    args = _parse_args()
    if (
        args.pairs <= 0
        or args.world_size <= 0
        or args.numel != 32768
        or args.warmup < 0
        or args.iters <= 0
        or args.sampler_interval <= 0
        or args.preflight_replays < 4
        or args.run_timeout_seconds <= 0
        or args.distributed_timeout_seconds <= 0
    ):
        raise ValueError(
            "invalid A/B contract: this harness is scoped to 32,768 BF16 "
            "elements (64 KiB) and requires positive timeouts/iterations plus "
            "at least four mutating preflight replays"
        )
    controller_path = Path(__file__).resolve()
    controller_sha256 = _sha256(controller_path)
    controller_argv = list(sys.argv)
    controller_argv_sha256 = _json_sha256(controller_argv)
    baseline_root = args.baseline_worktree.resolve()
    candidate_root = args.candidate_worktree.resolve()
    roots = (baseline_root, candidate_root)
    if baseline_root == candidate_root:
        raise ValueError("baseline and candidate worktrees must differ")
    output_dir = _outside_worktrees(args.output_dir, roots, "output directory")
    output = _outside_worktrees(args.output, roots, "suite output")
    harness_source = _outside_worktrees(args.harness, roots, "harness")
    if not harness_source.is_file():
        raise ValueError(f"harness does not exist: {harness_source}")
    baseline_sha = _git_sha(baseline_root)
    candidate_sha = _git_sha(candidate_root)
    if baseline_sha == candidate_sha:
        raise ValueError("baseline and candidate Git SHAs must differ")
    harness_sha256 = _sha256(harness_source)
    contract = _expected_contract(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    harness = output_dir / f"harness-{harness_sha256[:16]}.py"
    harness_bytes = harness_source.read_bytes()
    if harness.exists() and harness.read_bytes() != harness_bytes:
        raise ValueError(f"frozen harness collision at {harness}")
    if not harness.exists():
        harness.write_bytes(harness_bytes)
    if _sha256(harness) != harness_sha256:
        raise ValueError("frozen harness hash differs from source harness")

    variants = {
        "baseline": {
            "root": baseline_root,
            "sha": baseline_sha,
            "expected_nodes": 1,
        },
        "candidate": {
            "root": candidate_root,
            "sha": candidate_sha,
            "expected_nodes": 2,
        },
    }
    records = []
    sequence = []
    for pair_index in range(args.pairs):
        order = (
            ("baseline", "candidate")
            if pair_index % 2 == 0
            else ("candidate", "baseline")
        )
        for variant in order:
            run_index = len(sequence)
            sequence.append(variant)
            config = variants[variant]
            raw_path = output_dir / f"{run_index:02d}-{variant}.json"
            extension_dir = (
                output_dir
                / "torch-extensions"
                / f"{run_index:02d}-{variant}-{config['sha'][:12]}"
            )
            extension_dir.parent.mkdir(parents=True, exist_ok=True)
            extension_dir.mkdir(exist_ok=False)
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(config["root"]) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            env["TORCH_EXTENSIONS_DIR"] = str(extension_dir)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={args.world_size}",
                str(harness),
                "--label",
                f"{variant}-{config['sha'][:12]}-{run_index}",
                "--variant",
                variant,
                "--run-index",
                str(run_index),
                "--implementation-root",
                str(config["root"]),
                "--expected-git-sha",
                config["sha"],
                "--expected-graph-kernel-nodes",
                str(config["expected_nodes"]),
                "--numel",
                str(args.numel),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--sampler-interval",
                str(args.sampler_interval),
                "--preflight-replays",
                str(args.preflight_replays),
                "--distributed-timeout-seconds",
                str(args.distributed_timeout_seconds),
                "--controller-sha256",
                controller_sha256,
                "--controller-argv-sha256",
                controller_argv_sha256,
                "--output",
                str(raw_path),
            ]
            print(
                f"[{run_index + 1}/{2 * args.pairs}] {variant} {config['sha'][:12]}",
                flush=True,
            )
            _run(
                command,
                cwd=config["root"],
                env=env,
                timeout=args.run_timeout_seconds,
            )
            record = json.loads(raw_path.read_text(encoding="utf-8"))
            _validate_run(
                record,
                contract=contract,
                variant=variant,
                sha=config["sha"],
                run_index=run_index,
                harness_sha256=harness_sha256,
                expected_nodes=config["expected_nodes"],
                implementation_root=config["root"],
                controller_sha256=controller_sha256,
                controller_argv_sha256=controller_argv_sha256,
                extension_dir=extension_dir,
            )
            records.append(record)

    software_fingerprints = {
        json.dumps(record["software"], sort_keys=True) for record in records
    }
    hardware_fingerprints = {
        json.dumps(_stable_hardware(record), sort_keys=True) for record in records
    }
    environment_fingerprints = {
        json.dumps(_stable_environment(record), sort_keys=True) for record in records
    }
    if len(software_fingerprints) != 1:
        raise ValueError("software stack changed during alternating A/B suite")
    if len(hardware_fingerprints) != 1:
        raise ValueError(
            "hardware/driver/topology changed during alternating A/B suite"
        )
    if len(environment_fingerprints) != 1:
        raise ValueError("environment toggles changed during alternating A/B suite")

    paired_deltas = _paired_deltas(records)
    suite = {
        "schema": SUITE_SCHEMA,
        "contract": contract,
        "controller": {
            "source_path": str(controller_path),
            "sha256": controller_sha256,
            "argv": controller_argv,
            "argv_sha256": controller_argv_sha256,
        },
        "harness": {
            "source_path": str(harness_source),
            "frozen_path": str(harness),
            "sha256": harness_sha256,
            "identical_for_all_runs": True,
        },
        "implementations": {
            "baseline": {"root": str(baseline_root), "git_sha": baseline_sha},
            "candidate": {"root": str(candidate_root), "git_sha": candidate_sha},
        },
        "sequence": sequence,
        "records": records,
        "paired_deltas": paired_deltas,
        "paired_summary": _paired_summary(paired_deltas),
        "aggregate": _aggregate(records),
    }
    if set(suite) != SUITE_TOP_LEVEL_KEYS:
        raise ValueError("suite top-level schema mismatch")
    if suite["schema"] != SUITE_SCHEMA or suite["contract"] != contract:
        raise ValueError("suite identity/contract mismatch")
    if set(suite["controller"]) != CONTROLLER_KEYS:
        raise ValueError("suite controller schema mismatch")
    if (
        _sha256(Path(suite["controller"]["source_path"]))
        != suite["controller"]["sha256"]
    ):
        raise ValueError("suite controller source digest mismatch")
    if _json_sha256(suite["controller"]["argv"]) != suite["controller"]["argv_sha256"]:
        raise ValueError("suite controller argv digest mismatch")
    if set(suite["harness"]) != HARNESS_KEYS:
        raise ValueError("suite harness schema mismatch")
    if set(suite["implementations"]) != {"baseline", "candidate"}:
        raise ValueError("suite implementation variants mismatch")
    for implementation in suite["implementations"].values():
        if set(implementation) != IMPLEMENTATION_KEYS:
            raise ValueError("suite implementation schema mismatch")
    if len(suite["sequence"]) != args.pairs * 2:
        raise ValueError("suite sequence cardinality mismatch")
    if len(suite["records"]) != args.pairs * 2:
        raise ValueError("suite record cardinality mismatch")
    if len(suite["paired_deltas"]) != args.pairs:
        raise ValueError("suite paired-delta cardinality mismatch")
    rendered = json.dumps(suite, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
