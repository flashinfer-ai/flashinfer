#!/usr/bin/env python3
"""Orchestrate controlled SM100/SM103 CuTe-DSL dense W4A16 benchmarks.

The published PR #4466 matrix and the TP8 serving projection shapes are named
suites.  Every ``(shape, graph, PDL, repeat)`` measurement runs in a fresh
worker process, allowing the worker to make exactly one CUPTI timing call.  The
worker always times a preallocated output and an explicitly resolved tactic:
either the autotuned winner (persisted in a shared cache) or one stable
canonical tactic.

Example:

    python benchmarks/bench_dense_w4a16_sm100.py \
      --suite pr4466 --label baseline --repeats 3 --iters 100 \
      --arms graph_pdl_on --tactic-mode canonical \
      --output-dir /results/dense-w4a16-baseline
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PR_4466_M_VALUES = (1, 8, 32, 128, 512, 1024, 2048, 4096)
CANONICAL_TACTIC_SPEC = "256,64,256:2,1:true"


@dataclass(frozen=True)
class Projection:
    suite: str
    name: str
    n: int
    k: int
    provenance: str


@dataclass(frozen=True)
class Case:
    suite: str
    projection: str
    m: int
    n: int
    k: int
    provenance: str

    @property
    def name(self) -> str:
        return f"{self.projection}_m{self.m}"


@dataclass(frozen=True)
class Arm:
    name: str
    cuda_graph: bool
    enable_pdl: bool


PROJECTIONS = (
    Projection(
        "pr4466",
        "ffn_down_full",
        6656,
        19968,
        "FlashInfer PR #4466 published N=6656, K=19968 matrix",
    ),
    Projection(
        "pr4466",
        "ffn_up_full",
        19968,
        6656,
        "FlashInfer PR #4466 published N=19968, K=6656 matrix",
    ),
    # Muse/GLM dense layers at TP=8.  hidden=6656, intermediate=19968,
    # 32 query heads, 2 KV heads, head_dim=128.  KV heads are replicated
    # four ways when TP exceeds the KV-head count.
    Projection(
        "serving_tp8",
        "attention_qkv",
        768,
        6656,
        "TP8 QKV: 4096/8 query channels + 128 key + 128 value",
    ),
    Projection(
        "serving_tp8",
        "attention_output",
        6656,
        512,
        "TP8 row-parallel attention output: K=(32*128)/8",
    ),
    Projection(
        "serving_tp8",
        "ffn_gate_up",
        4992,
        6656,
        "TP8 merged gate/up: N=2*19968/8",
    ),
    Projection(
        "serving_tp8",
        "ffn_down",
        6656,
        2496,
        "TP8 row-parallel down projection: K=19968/8",
    ),
)

ARMS = {
    "eager_pdl_off": Arm("eager_pdl_off", False, False),
    "eager_pdl_on": Arm("eager_pdl_on", False, True),
    "graph_pdl_off": Arm("graph_pdl_off", True, False),
    "graph_pdl_on": Arm("graph_pdl_on", True, True),
}
DEFAULT_ARMS = ("graph_pdl_on",)

PIPELINE_COLUMNS = (
    "num_load2trans_stage",
    "num_trans2mma_stage",
    "num_acc_stage",
    "num_c_stage",
    "num_tile_info_stage",
    "num_acc_tmem_cols",
    "num_a_tmem_cols",
    "num_tmem_alloc_cols",
    "configured_transform_fragment_size",
    "num_transform_warpgroups",
    "num_transform_warps",
    "threads_per_cta",
    "num_regs_epilogue_warps",
    "num_regs_generic_warps",
    "max_active_clusters",
)


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("expected unique comma-separated values")
    return parsed


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in _parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from error
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive")
    return parsed


def _cases(suite: str, m_values: Iterable[int]) -> list[Case]:
    selected_suites = {"pr4466", "serving_tp8"} if suite == "all" else {suite}
    return [
        Case(
            projection.suite,
            projection.name,
            m,
            projection.n,
            projection.k,
            projection.provenance,
        )
        for projection in PROJECTIONS
        if projection.suite in selected_suites
        for m in m_values
    ]


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _worker_result_path(
    output_dir: Path,
    label: str,
    case: Case,
    arm: Arm,
    repeat: int,
    compile_opt_level: int,
    transform_fragment_size: int | None,
    tactic_tag: str,
) -> Path:
    filename = "__".join(
        (
            _safe_name(label),
            case.suite,
            case.name,
            arm.name,
            f"o{compile_opt_level}",
            f"f{transform_fragment_size or 'production'}",
            _safe_name(tactic_tag),
            f"r{repeat}",
        )
    )
    return output_dir / "workers" / f"{filename}.json"


def _worker_log_path(result_path: Path) -> Path:
    return result_path.parents[1] / "logs" / f"{result_path.stem}.log"


def _repo_source_identity(repo: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    diff = git("diff", "--binary", "HEAD")
    return {
        "revision": git("rev-parse", "HEAD"),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _preflight(python: str, repo: Path) -> dict[str, Any]:
    script = r"""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import torch
import flashinfer
from cupti import cupti

version = importlib.metadata.version("cupti-python")
if int(version.split(".", 1)[0]) < 13:
    raise RuntimeError(f"cupti-python >= 13 required, found {version}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
props = torch.cuda.get_device_properties(0)
if (props.major, props.minor) not in ((10, 0), (10, 3)):
    raise RuntimeError(f"SM100/SM103 required, got SM{props.major}{props.minor}")
timestamp = int(cupti.get_timestamp())
if timestamp <= 0:
    raise RuntimeError(f"invalid CUPTI timestamp {timestamp}")
print(json.dumps({
    "python": __import__("sys").version.split()[0],
    "python_executable": __import__("sys").executable,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "flashinfer_file": str(Path(flashinfer.__file__).resolve()),
    "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
    "cupti_python": version,
    "nvidia_cuda_cupti": importlib.metadata.version("nvidia-cuda-cupti"),
    "cuda_bindings": importlib.metadata.version("cuda-bindings"),
    "cupti_timestamp": timestamp,
    "gpu_name": props.name,
    "compute_capability": [props.major, props.minor],
    "nvidia_smi": subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,power.limit,clocks.max.sm",
        "--format=csv,noheader,nounits",
    ], text=True).strip(),
}))
"""
    result = subprocess.run(
        [python, "-c", script],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "CUPTI/GPU preflight failed (event fallback is not allowed):\n"
            + result.stdout
            + result.stderr
        )
    try:
        metadata = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid preflight output: {result.stdout!r}") from error
    imported = Path(metadata["flashinfer_file"])
    if not imported.is_relative_to(repo):
        raise RuntimeError(
            f"preflight imported flashinfer outside checkout: {imported} != {repo}"
        )
    metadata["repo_source"] = _repo_source_identity(repo)
    return metadata


def _worker_command(
    args: argparse.Namespace,
    worker: Path,
    case: Case,
    arm: Arm,
    repeat: int,
    result_path: Path,
    autotune_cache: Path,
    input_cache_dir: Path,
) -> list[str]:
    command = [
        args.python,
        str(worker),
        "--result-json",
        str(result_path),
        "--suite",
        case.suite,
        "--case",
        case.name,
        "--label",
        args.label,
        "--repeat",
        str(repeat),
        "--m",
        str(case.m),
        "--n",
        str(case.n),
        "--k",
        str(case.k),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--tactic-mode",
        args.tactic_mode,
        "--forced-tactic",
        args.forced_tactic,
        "--compile-opt-level",
        str(args.compile_opt_level),
        "--input-cache-dir",
        str(input_cache_dir),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
    ]
    if args.transform_fragment_size is not None:
        command.extend(("--transform-fragment-size", str(args.transform_fragment_size)))
    if args.allow_experimental_tactic:
        command.append("--allow-experimental-tactic")
    if args.tactic_mode == "auto":
        command.extend(("--autotune-cache", str(autotune_cache)))
    if arm.enable_pdl:
        command.append("--enable-pdl")
    if arm.cuda_graph:
        command.append("--cuda-graph")
    return command


def _read_worker_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid worker result {path}: {error}") from error
    if result.get("schema_version") != 1:
        raise RuntimeError(f"unexpected worker schema in {path}")
    result["worker_result_file"] = str(path)
    return result


def _load_worker_results(output_dir: Path) -> list[dict[str, Any]]:
    return [
        _read_worker_result(path)
        for path in sorted((output_dir / "workers").glob("*.json"))
    ]


def _validate_result_invocation(
    path: Path,
    worker_command: list[str],
    expected_repo_source: dict[str, Any],
) -> None:
    result = _read_worker_result(path)
    actual = result.get("invocation")
    expected = worker_command
    # The worker records its resolved sys.executable, while the orchestrator
    # may have been given an equivalent command name. Every other argument,
    # including the output path and all benchmark controls, must match exactly.
    if not isinstance(actual, list) or actual[1:] != expected[1:]:
        raise RuntimeError(
            f"worker result does not match the current plan: {path}\n"
            f"expected={expected!r}\nactual={actual!r}"
        )
    expected_worker_sha256 = hashlib.sha256(Path(expected[1]).read_bytes()).hexdigest()
    if result.get("worker_sha256") != expected_worker_sha256:
        raise RuntimeError(
            f"worker result was produced by a different worker: {path}\n"
            f"expected_sha256={expected_worker_sha256}\n"
            f"actual_sha256={result.get('worker_sha256')!r}"
        )
    actual_repo = result.get("environment", {}).get("repo", {})
    actual_repo_source = {
        field: actual_repo.get(field) for field in ("revision", "diff_sha256")
    }
    if actual_repo_source != expected_repo_source:
        raise RuntimeError(
            f"worker result source does not match the current checkout: {path}\n"
            f"expected={expected_repo_source!r}\nactual={actual_repo_source!r}"
        )


def _validate_result_set(
    results: list[dict[str, Any]], preflight: dict[str, Any]
) -> None:
    if not results:
        return
    baseline = results[0]
    invariant_fields = ("worker_sha256", "seed", "warmup", "iters")
    environment_fields = (
        "python",
        "python_executable",
        "torch",
        "torch_cuda",
        "flashinfer_file",
        "cutlass_dsl",
        "cupti_python",
        "nvidia_cuda_cupti",
        "cuda_bindings",
        "gpu_name",
        "compute_capability",
        "nvidia_smi",
    )
    source_by_label: dict[str, tuple[Any, ...]] = {}
    for result in results:
        mismatches = {
            field: (baseline[field], result[field])
            for field in invariant_fields
            if result.get(field) != baseline.get(field)
        }
        for field in ("reference_rtol", "reference_atol"):
            expected = baseline["correctness"][field]
            actual = result["correctness"][field]
            if actual != expected:
                mismatches[f"correctness.{field}"] = (expected, actual)
        for field in environment_fields:
            expected = preflight[field]
            actual = result["environment"].get(field)
            if actual != expected:
                mismatches[f"environment.{field}"] = (expected, actual)
        repo = result["environment"]["repo"]
        source = (
            repo.get("revision"),
            repo.get("diff_sha256"),
        )
        label = result["label"]
        if label in source_by_label and source_by_label[label] != source:
            mismatches["environment.repo_source_for_label"] = (
                source_by_label[label],
                source,
            )
        source_by_label[label] = source
        if mismatches:
            raise RuntimeError(
                "mixed benchmark controls or hardware in output directory "
                f"for {result['worker_result_file']}: {mismatches!r}"
            )


def _validate_fixed_tactics(results: list[dict[str, Any]]) -> None:
    expected_tactic: dict[tuple[Any, ...], Any] = {}
    expected_pipeline: dict[tuple[Any, ...], Any] = {}
    for result in results:
        requested_tactic_key = (
            json.dumps(result["forced_tactic_requested"], sort_keys=True)
            if result["tactic_mode"] == "canonical"
            else "auto"
        )
        tactic_key = (
            result["suite"],
            result["case"],
            result["m"],
            result["n"],
            result["k"],
            result["enable_pdl"],
            result["compile_opt_level"],
            result["environment"]["repo"]["revision"],
            result["environment"]["repo"]["diff_sha256"],
            result["tactic_mode"],
            requested_tactic_key,
        )
        tactic = result["tactic"]
        if tactic_key in expected_tactic and expected_tactic[tactic_key] != tactic:
            raise RuntimeError(
                "tactic changed across labels/graph/repeats for "
                f"{tactic_key}: {expected_tactic[tactic_key]!r} != {tactic!r}"
            )
        expected_tactic[tactic_key] = tactic

        # Pipeline-depth changes are valid experiments across source labels,
        # but one source must derive the same stages across graph/repeats.
        pipeline_key = (
            result["label"],
            result["pipeline"]["configured_transform_fragment_size"],
            *tactic_key,
        )
        pipeline = result["pipeline"]
        if (
            pipeline_key in expected_pipeline
            and expected_pipeline[pipeline_key] != pipeline
        ):
            raise RuntimeError(
                "pipeline changed across graph/repeats for "
                f"{pipeline_key}: {expected_pipeline[pipeline_key]!r} != {pipeline!r}"
            )
        expected_pipeline[pipeline_key] = pipeline


def _summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result["label"],
            result["suite"],
            result["case"],
            result["m"],
            result["n"],
            result["k"],
            result["cuda_graph"],
            result["enable_pdl"],
            result["tactic_mode"],
            result["compile_opt_level"],
            result["pipeline"]["configured_transform_fragment_size"],
            json.dumps(result["tactic"], separators=(",", ":")),
        )
        groups.setdefault(key, []).append(result)

    rows = []
    for key, repetitions in sorted(groups.items()):
        repetitions.sort(key=lambda result: result["repeat"])
        medians = [float(result["median_ms"]) for result in repetitions]
        representative = repetitions[0]
        median_ms = float(statistics.median(medians))
        flops = 2 * representative["m"] * representative["n"] * representative["k"]
        row = {
            "label": key[0],
            "suite": key[1],
            "case": key[2],
            "m": key[3],
            "n": key[4],
            "k": key[5],
            "cuda_graph": key[6],
            "enable_pdl": key[7],
            "tactic_mode": key[8],
            "compile_opt_level": key[9],
            "configured_transform_fragment_size": key[10],
            "timing_backend": "cupti",
            "cold_l2": True,
            "preallocated_output": True,
            "repeats": len(repetitions),
            "repeat_medians_ms": medians,
            "median_of_repeat_medians_ms": median_ms,
            "min_repeat_median_ms": min(medians),
            "max_repeat_median_ms": max(medians),
            "tflops_from_median": flops / median_ms / 1e9,
            "tactic_index": representative["tactic_index"],
            "tactic_in_production_search_space": representative[
                "tactic_in_production_search_space"
            ],
            "tactic": representative["tactic"],
            "pipeline": representative["pipeline"],
            "correctness_all_passed": all(
                repetition["correctness"]["output_all_finite"]
                and repetition["correctness"]["eager_bitwise_repeatable"]
                and repetition["correctness"]["graph_bitwise_repeatable"] is not False
                and repetition["correctness"]["graph_matches_eager_bitwise"]
                is not False
                for repetition in repetitions
            ),
            "reference_max_abs": max(
                repetition["correctness"]["reference_max_abs"]
                for repetition in repetitions
            ),
            "worker_result_files": [
                repetition["worker_result_file"] for repetition in repetitions
            ],
        }
        rows.append(row)
    return rows


def _csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"pipeline", "worker_result_files"}
    }
    row["repeat_medians_ms"] = json.dumps(
        row["repeat_medians_ms"], separators=(",", ":")
    )
    row["tactic"] = json.dumps(row["tactic"], separators=(",", ":"))
    for column in PIPELINE_COLUMNS:
        row[column] = summary["pipeline"][column]
    return row


def _write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    *,
    preflight: dict[str, Any],
    command: list[str],
    planned_result_paths: list[Path],
    current_label: str,
) -> None:
    _validate_result_set(results, preflight)
    _validate_fixed_tactics(results)
    planned_result_files = {str(path) for path in planned_result_paths}
    unexpected_same_label = [
        result["worker_result_file"]
        for result in results
        if result["label"] == current_label
        and result["worker_result_file"] not in planned_result_files
    ]
    if unexpected_same_label:
        raise RuntimeError(
            "output directory contains same-label worker results outside the "
            f"current plan for {current_label!r}: {unexpected_same_label!r}"
        )
    summaries = _summaries(results)
    completed_result_paths = [path for path in planned_result_paths if path.is_file()]
    raw_path = output_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results)
    )
    summary_document = {
        "schema_version": 1,
        "preflight": preflight,
        "orchestrator_command": command,
        "shape_contract": {
            "pr4466_m_values": PR_4466_M_VALUES,
            "canonical_tactic": CANONICAL_TACTIC_SPEC,
            "projections": [projection.__dict__ for projection in PROJECTIONS],
            "arms": [arm.__dict__ for arm in ARMS.values()],
        },
        "raw_result_count": len(results),
        "current_plan": {
            "label": current_label,
            "planned_count": len(planned_result_paths),
            "completed_count": len(completed_result_paths),
            "complete": len(completed_result_paths) == len(planned_result_paths),
            "result_files": [str(path) for path in planned_result_paths],
        },
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_document, indent=2) + "\n"
    )
    csv_rows = [_csv_row(summary) for summary in summaries]
    if csv_rows:
        with (output_dir / "summary.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    sample_rows = [
        {
            "label": result["label"],
            "suite": result["suite"],
            "case": result["case"],
            "repeat": result["repeat"],
            "m": result["m"],
            "n": result["n"],
            "k": result["k"],
            "cuda_graph": result["cuda_graph"],
            "enable_pdl": result["enable_pdl"],
            "tactic_mode": result["tactic_mode"],
            "compile_opt_level": result["compile_opt_level"],
            "configured_transform_fragment_size": result["pipeline"][
                "configured_transform_fragment_size"
            ],
            "tactic_index": result["tactic_index"],
            "tactic": json.dumps(result["tactic"], separators=(",", ":")),
            "sample_index": sample_index,
            "sample_ms": sample_ms,
            "worker_result_file": result["worker_result_file"],
        }
        for result in results
        for sample_index, sample_ms in enumerate(result["samples_ms"], start=1)
    ]
    if sample_rows:
        with (output_dir / "samples.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("pr4466", "serving_tp8", "all"),
        default="pr4466",
    )
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--m-values",
        type=_parse_positive_ints,
        default=PR_4466_M_VALUES,
        help="Override the published M grid for targeted probes.",
    )
    parser.add_argument(
        "--arms",
        type=_parse_csv,
        default=DEFAULT_ARMS,
        help="Comma-separated graph/PDL arms: " + ",".join(ARMS),
    )
    parser.add_argument(
        "--cases",
        type=_parse_csv,
        help="Optional comma-separated projection names to retain.",
    )
    parser.add_argument(
        "--tactic-mode", choices=("auto", "canonical"), default="canonical"
    )
    parser.add_argument(
        "--forced-tactic",
        default=CANONICAL_TACTIC_SPEC,
        help=(
            "Tactic for canonical mode as "
            "TILE_M,TILE_N,TILE_K:CLUSTER_M,CLUSTER_N:true|false."
        ),
    )
    parser.add_argument(
        "--allow-experimental-tactic",
        action="store_true",
        help=(
            "Allow a fixed canonical tactic outside the production search space; "
            "the worker still requires the kernel can_implement check to pass."
        ),
    )
    parser.add_argument("--compile-opt-level", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--transform-fragment-size",
        type=int,
        choices=(32, 64, 128),
        help="Override the production 32/128 transform-fragment heuristic.",
    )
    parser.add_argument(
        "--autotune-cache",
        type=Path,
        help=(
            "Base path for the tactic cache; the benchmark appends the compile "
            "optimization level, transform-fragment configuration, and source "
            "identity."
        ),
    )
    parser.add_argument("--input-cache-dir", type=Path)
    parser.add_argument("--rtol", type=float, default=1.5e-2)
    parser.add_argument("--atol", type=float, default=1.5e-2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--worker-timeout-seconds", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    unknown_arms = sorted(set(args.arms) - set(ARMS))
    if unknown_arms:
        parser.error(f"unknown --arms values: {','.join(unknown_arms)}")
    if args.repeats <= 0 or args.iters <= 0 or args.warmup < 0:
        parser.error("repeats/iters must be positive and warmup nonnegative")
    if args.worker_timeout_seconds <= 0:
        parser.error("--worker-timeout-seconds must be positive")
    if args.rtol < 0 or args.atol < 0:
        parser.error("correctness tolerances must be nonnegative")
    if args.transform_fragment_size is not None and args.tactic_mode != "canonical":
        parser.error(
            "--transform-fragment-size requires --tactic-mode canonical so "
            "autotuning cannot rank the production fragment instead"
        )
    if args.allow_experimental_tactic and args.tactic_mode != "canonical":
        parser.error("--allow-experimental-tactic requires --tactic-mode canonical")

    selected_cases = _cases(args.suite, args.m_values)
    if args.cases:
        requested = set(args.cases)
        known = {case.projection for case in selected_cases}
        unknown = requested - known
        if unknown:
            parser.error(
                f"unknown --cases values for this suite: {','.join(sorted(unknown))}"
            )
        selected_cases = [
            case for case in selected_cases if case.projection in requested
        ]
    if args.list_cases:
        for case in selected_cases:
            print(
                f"{case.suite:<12} {case.name:<24} "
                f"M={case.m:<4} N={case.n:<6} K={case.k:<6} {case.provenance}"
            )
        return 0

    repo = Path(__file__).resolve().parents[1]
    worker = Path(__file__).with_name("bench_dense_w4a16_sm100_worker.py")
    if not worker.is_file():
        raise RuntimeError(f"missing worker: {worker}")
    output_dir = args.output_dir.resolve()
    repo_source = _repo_source_identity(repo)
    fragment_tag = args.transform_fragment_size or "production"
    autotune_cache_base = (args.autotune_cache or output_dir / "tactics.json").resolve()
    cache_suffix = autotune_cache_base.suffix or ".json"
    cache_stem = (
        autotune_cache_base.stem
        if autotune_cache_base.suffix
        else autotune_cache_base.name
    )
    source_tag = f"{repo_source['revision'][:12]}-{repo_source['diff_sha256'][:12]}"
    autotune_cache = autotune_cache_base.with_name(
        f"{cache_stem}_o{args.compile_opt_level}_f{fragment_tag}_s{source_tag}"
        f"{cache_suffix}"
    )
    input_cache_dir = (args.input_cache_dir or output_dir / "input_cache").resolve()
    selected_arms = [ARMS[name] for name in args.arms]

    plan = [
        (
            case,
            arm,
            repeat,
            _worker_result_path(
                output_dir,
                args.label,
                case,
                arm,
                repeat,
                args.compile_opt_level,
                args.transform_fragment_size,
                args.forced_tactic if args.tactic_mode == "canonical" else "auto",
            ),
        )
        for case in selected_cases
        for arm in selected_arms
        for repeat in range(1, args.repeats + 1)
    ]
    planned_result_paths = [entry[3] for entry in plan]
    if args.dry_run:
        print(f"# {len(plan)} fresh worker processes")
        for case, arm, repeat, result_path in plan:
            print(
                subprocess.list2cmdline(
                    _worker_command(
                        args,
                        worker,
                        case,
                        arm,
                        repeat,
                        result_path,
                        autotune_cache,
                        input_cache_dir,
                    )
                )
            )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "workers").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    preflight = _preflight(args.python, repo)
    command = [sys.executable, *sys.argv]
    (output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(
        f"Preflight: {preflight['gpu_name']} SM"
        f"{preflight['compute_capability'][0]}{preflight['compute_capability'][1]}, "
        f"cupti-python {preflight['cupti_python']}"
    )
    print(f"Plan: {len(plan)} fresh worker processes -> {output_dir}")
    for case, arm, repeat, result_path in plan:
        if not result_path.is_file():
            continue
        if not args.resume:
            raise RuntimeError(
                f"worker result already exists: {result_path}; pass --resume or "
                "use a new label/output"
            )
        _validate_result_invocation(
            result_path,
            _worker_command(
                args,
                worker,
                case,
                arm,
                repeat,
                result_path,
                autotune_cache,
                input_cache_dir,
            ),
            preflight["repo_source"],
        )
    all_results = _load_worker_results(output_dir)
    _write_outputs(
        output_dir,
        all_results,
        preflight=preflight,
        command=command,
        planned_result_paths=planned_result_paths,
        current_label=args.label,
    )

    for ordinal, (case, arm, repeat, result_path) in enumerate(plan, start=1):
        worker_command = _worker_command(
            args,
            worker,
            case,
            arm,
            repeat,
            result_path,
            autotune_cache,
            input_cache_dir,
        )
        if result_path.is_file():
            if args.resume:
                _validate_result_invocation(
                    result_path, worker_command, preflight["repo_source"]
                )
                print(f"[{ordinal}/{len(plan)}] RESUME {result_path.stem}")
                continue
            raise RuntimeError(
                f"worker result already exists: {result_path}; pass --resume or use a new label/output"
            )
        print(f"[{ordinal}/{len(plan)}] START {result_path.stem}", flush=True)
        try:
            completed = subprocess.run(
                worker_command,
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=args.worker_timeout_seconds,
                check=False,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"worker timed out after {args.worker_timeout_seconds}s: {result_path.stem}"
            ) from error
        log_path = _worker_log_path(result_path)
        log_path.write_text(
            "COMMAND "
            + json.dumps(worker_command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\nSTDERR\n"
            + completed.stderr
        )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"worker failed with exit {completed.returncode}; see {log_path}"
            )
        _validate_result_invocation(
            result_path, worker_command, preflight["repo_source"]
        )
        result = json.loads(result_path.read_text())
        print(
            f"[{ordinal}/{len(plan)}] DONE  {result_path.stem}: "
            f"{result['median_ms'] * 1000.0:.3f} us, "
            f"tactic={result['tactic_index']}"
        )
        all_results = _load_worker_results(output_dir)
        _write_outputs(
            output_dir,
            all_results,
            preflight=preflight,
            command=command,
            planned_result_paths=planned_result_paths,
            current_label=args.label,
        )

    all_results = _load_worker_results(output_dir)
    _write_outputs(
        output_dir,
        all_results,
        preflight=preflight,
        command=command,
        planned_result_paths=planned_result_paths,
        current_label=args.label,
    )
    print(
        f"Complete: {len(all_results)} raw results; "
        f"{output_dir / 'raw.jsonl'}, {output_dir / 'samples.csv'}, "
        f"{output_dir / 'summary.csv'}, "
        f"{output_dir / 'summary.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
