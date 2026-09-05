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

"""Benchmark the VibeCUDA MSA backend against the pinned PR4355 fast baseline.

This benchmark reproduces the protocol of
``benchmarks/bench_cake_msa_sm100.py`` (imported as a sibling module, so the
frozen shape manifest, deterministic input construction, tolerances,
independent FP32 masked-attention reference, and strict CUPTI measurement
protocol stay identical) and swaps in the two backends under comparison:

* the configured fast baseline: the public
  ``flashinfer.msa_ops.msa_sparse_attention`` /
  ``flashinfer.msa_ops.msa_sparse_decode_attention`` API of the pinned
  FlashInfer PR4355 checkout, whose default dispatch routes to the fused
  CUDA C++ CAKE SM100/SM103 kernels (``flashinfer/msa_ops/_cake_sm100.py``,
  ``csrc/cake_msa``), and
* the candidate: the same public API in this checkout with
  ``backend="vibecuda"``.

The two revisions cannot coexist in one Python process (both provide the
``flashinfer`` package), so each measured backend/shape pair already runs in
a fresh process; the baseline workers simply import flashinfer from the
pinned checkout instead of this one.  Each row is first checked in two more
isolated processes that validate the baseline and the candidate
*independently* against the authoritative independent FP32 masked-attention
reference from the imported module (plus an exact-zero-row full-write
audit), since on compute capability 10.0/10.3 the in-tree ``backend="auto"``
successor path is SM120/SM121-only and raises, which the ``probe-auto``
worker proves once per run on the candidate checkout.

Example
-------
Clone the pinned PR4355 source next to this checkout, then run from a clean
FlashInfer checkout::

    git worktree add /tmp/flashinfer-pr4355 a312d1c3b99b4f4983cba734268c10de60df75e8
    python benchmarks/bench_vibecuda_msa_sm100.py \
      --candidate-root "$PWD" --candidate-sha "$(git rev-parse HEAD)" \
      --baseline-root /tmp/flashinfer-pr4355 \
      --json /tmp/msa-vibecuda-sm100.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cake_msa_sm100 as cake_bench


PR4355_SOURCE_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer.git"
PR4355_PULL_REQUEST = "https://github.com/flashinfer-ai/flashinfer/pull/4355"
PR4355_SOURCE_SHA = "a312d1c3b99b4f4983cba734268c10de60df75e8"

BACKEND_CANDIDATE = "vibecuda"
BACKEND_BASELINE = "pr4355"
BACKENDS = (BACKEND_BASELINE, BACKEND_CANDIDATE)
CANDIDATE_PUBLIC_NAME = "flashinfer.msa_ops backend=vibecuda"
BASELINE_PUBLIC_NAME = (
    "flashinfer.msa_ops default CAKE SM100/SM103 dispatch at "
    f"{PR4355_PULL_REQUEST} ({PR4355_SOURCE_SHA})"
)
ZERO_ROW_FULL_WRITE_CONTRACT = (
    "every (query row, head) slice that is exactly zero in the FP32 reference "
    "(a row with no valid selected token) must be exactly zero in the backend "
    "output, so stale or partially written empty tensors cannot pass"
)

MSAShape = cake_bench.MSAShape
SHAPE_MANIFEST = cake_bench.SHAPE_MANIFEST
SHAPES_BY_ID = cake_bench.SHAPES_BY_ID
ACTIVITY_SCOPE = cake_bench.ACTIVITY_SCOPE
CORRECTNESS_TOLERANCES = cake_bench.CORRECTNESS_TOLERANCES


def _reference_tolerance(shape: MSAShape) -> dict[str, float]:
    """Declared precision contract for the independent FP32 reference.

    The authoritative benchmark compares against its independent FP32
    reference at the Q-dtype tolerance (its FP16 rows are FP16/FP16).  Rows
    whose K/V storage is FP8 E4M3 carry the FP8 quantization error through
    any implementation, so those rows use the FP8 entry, matching the
    kv-dtype tolerance the authoritative benchmark applies to FP8 rows.
    """

    if shape.kv_dtype == "float8_e4m3fn":
        return CORRECTNESS_TOLERANCES["float8_e4m3fn"]
    return CORRECTNESS_TOLERANCES[shape.q_dtype]


def _candidate_call(
    shape: MSAShape, inputs: dict[str, Any]
) -> tuple[Callable[[], Any], str, dict[str, Any]]:
    msa_ops = importlib.import_module("flashinfer.msa_ops")
    if shape.operation == "sparse_prefill":
        public_api = "flashinfer.msa_ops.msa_sparse_attention"

        def call():
            return msa_ops.msa_sparse_attention(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["q2k"],
                inputs["cu_q"],
                inputs["cu_k"],
                causal=shape.causal,
                page_table=inputs["page_table"],
                seqused_k=inputs["seqused_k"],
                return_softmax_lse=False,
                backend=BACKEND_CANDIDATE,
            )

    else:
        public_api = "flashinfer.msa_ops.msa_sparse_decode_attention"

        def call():
            return msa_ops.msa_sparse_decode_attention(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["q2k"],
                page_table=inputs["page_table"],
                seqused_k=inputs["seqused_k"],
                cu_seqlens_k=inputs["cu_k"],
                seqlen_q=shape.seqlen_q,
                causal=shape.causal,
                return_softmax_lse=False,
                force_fused=shape.force_fused,
                backend=BACKEND_CANDIDATE,
            )

    return call, public_api, {"excluded_setup": ["deterministic_input_construction"]}


def _verify_backend_reference(
    torch,
    shape: MSAShape,
    inputs: dict[str, Any],
    backend_call: Callable[[], Any],
    *,
    backend_api: str,
    backend_name: str,
) -> dict[str, Any]:
    """Validate one backend independently against the FP32 reference.

    Additionally audits the exact-zero-row full-write contract so a stale,
    empty, or partially written output tensor cannot pass.
    """

    backend_output = cake_bench._primary_output(backend_call())
    reference_output = cake_bench._candidate_reference_output(torch, shape, inputs)
    torch.cuda.synchronize()
    expected_shape = (
        shape.batch_size * shape.seqlen_q,
        shape.num_q_heads,
        shape.head_dim,
    )
    tolerance = _reference_tolerance(shape)
    shape_matches = (
        tuple(backend_output.shape) == expected_shape
        and tuple(reference_output.shape) == expected_shape
    )
    dtype_matches = backend_output.dtype == reference_output.dtype
    result: dict[str, Any] = {
        "reference": "independent_torch_fp32_masked_attention",
        "backend": backend_name,
        "backend_public_api": backend_api,
        "same_q_k_v_tensor_objects": True,
        "same_sequence_metadata_tensor_objects": True,
        "same_page_table_argument": True,
        "expected_shape": list(expected_shape),
        "backend_dtype": str(backend_output.dtype),
        "reference_dtype": str(reference_output.dtype),
        **tolerance,
    }
    if not shape_matches or not dtype_matches:
        result.update(
            {
                "status": "failed",
                "passed": False,
                "backend_shape": list(backend_output.shape),
                "reference_shape": list(reference_output.shape),
                "max_abs_error": None,
                "mismatch_count": None,
                "zero_row_violation_count": None,
            }
        )
        return result

    backend_float = backend_output.float()
    reference_float = reference_output.float()
    close = torch.isclose(
        backend_float,
        reference_float,
        atol=float(tolerance["atol"]),
        rtol=float(tolerance["rtol"]),
        equal_nan=False,
    )
    backend_nonfinite_count = int((~torch.isfinite(backend_float)).sum().item())
    reference_nonfinite_count = int((~torch.isfinite(reference_float)).sum().item())
    reference_zero_slices = (reference_float == 0).all(dim=-1)
    zero_row_violation_count = int(
        ((backend_float != 0).any(dim=-1) & reference_zero_slices).sum().item()
    )
    passed = (
        bool(close.all().item())
        and backend_nonfinite_count == 0
        and reference_nonfinite_count == 0
        and zero_row_violation_count == 0
    )
    finite = torch.isfinite(backend_float) & torch.isfinite(reference_float)
    max_abs_error = None
    if bool(finite.any().item()):
        max_abs_error = float(
            (backend_float[finite] - reference_float[finite]).abs().max().item()
        )
    result.update(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "max_abs_error": max_abs_error,
            "mismatch_count": int((~close).sum().item()),
            "backend_nonfinite_count": backend_nonfinite_count,
            "reference_nonfinite_count": reference_nonfinite_count,
            "reference_zero_row_count": int(reference_zero_slices.sum().item()),
            "zero_row_violation_count": zero_row_violation_count,
            "zero_row_full_write_contract": ZERO_ROW_FULL_WRITE_CONTRACT,
        }
    )
    return result


def _configure_imports(source_root: Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(source_root))
    torch = importlib.import_module("torch")
    flashinfer = importlib.import_module("flashinfer")
    imported_source = Path(flashinfer.__file__).resolve().parents[1]
    if imported_source != source_root:
        raise RuntimeError(
            f"expected flashinfer from {source_root}, imported {imported_source}"
        )
    return torch, flashinfer


def _baseline_revision_proof(source_root: Path, torch) -> dict[str, Any]:
    """Executable proof that the baseline worker runs the pinned CAKE backend."""

    return {
        "pull_request": PR4355_PULL_REQUEST,
        "source_sha": PR4355_SOURCE_SHA,
        "cake_backend_module": (
            source_root / "flashinfer" / "msa_ops" / "_cake_sm100.py"
        ).is_file(),
        "cake_cuda_sources": (source_root / "csrc" / "cake_msa").is_dir(),
        "public_default_dispatch": (
            "msa_sparse_attention/msa_sparse_decode_attention route to "
            "cake_msa_sparse_attention/cake_msa_sparse_decode_attention on "
            "compute capability 10.0/10.3 at the pinned revision"
        ),
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def _run_probe_auto(source_root: Path, shape: MSAShape) -> dict[str, Any]:
    """Prove backend="auto" cannot stand in for the CAKE baseline on CC10."""

    torch, _ = _configure_imports(source_root)
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = cake_bench._make_inputs(torch, shape, device)
    msa_ops = importlib.import_module("flashinfer.msa_ops")
    try:
        msa_ops.msa_sparse_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k"],
            inputs["cu_q"],
            inputs["cu_k"],
            causal=shape.causal,
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            return_softmax_lse=False,
            backend="auto",
        )
    except Exception as error:  # noqa: BLE001 - the message is the evidence
        return {
            "status": "unavailable",
            "backend": "auto",
            "public_api": "flashinfer.msa_ops.msa_sparse_attention",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    return {
        "status": "available",
        "backend": "auto",
        "public_api": "flashinfer.msa_ops.msa_sparse_attention",
    }


def _run_worker(args: argparse.Namespace) -> None:
    source_root = args.worker_source_root.resolve()
    source_sha = cake_bench._validate_checkout(
        source_root, args.worker_source_sha, "FlashInfer worker source"
    )
    cupti_python_version = cake_bench._require_cupti()
    torch, flashinfer = _configure_imports(source_root)
    hardware = cake_bench._hardware(torch)
    device = torch.device("cuda", torch.cuda.current_device())
    shape = SHAPES_BY_ID[args.worker_shape]
    inputs = cake_bench._make_inputs(torch, shape, device)
    software = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "flashinfer_version": getattr(flashinfer, "__version__", None),
        "cupti_python_version": cupti_python_version,
    }
    if args.worker_backend == "probe-auto":
        probe = _run_probe_auto(source_root, shape)
        result = {
            **probe,
            "backend": args.worker_backend,
            "probed_backend": probe["backend"],
            "shape": shape.stable_id,
            "source_sha": source_sha,
            "hardware": hardware,
            "software": software,
        }
        args.worker_json.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return
    if args.worker_backend in ("verify-candidate", "verify-baseline"):
        if args.worker_backend == "verify-candidate":
            call, public_api, _ = _candidate_call(shape, inputs)
            backend_name = BACKEND_CANDIDATE
        else:
            call, public_api, _ = cake_bench._candidate_call(shape, inputs)
            backend_name = BACKEND_BASELINE
        correctness = _verify_backend_reference(
            torch,
            shape,
            inputs,
            call,
            backend_api=public_api,
            backend_name=backend_name,
        )
        if backend_name == BACKEND_BASELINE:
            correctness["baseline_revision_proof"] = _baseline_revision_proof(
                source_root, torch
            )
        result = {
            "status": "verified" if correctness["passed"] else "failed",
            "backend": args.worker_backend,
            "shape": shape.stable_id,
            "correctness": correctness,
            "source_sha": source_sha,
            "hardware": hardware,
            "software": software,
        }
        args.worker_json.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return
    if args.worker_backend == BACKEND_CANDIDATE:
        call, public_api, setup = _candidate_call(shape, inputs)
    elif args.worker_backend == BACKEND_BASELINE:
        call, public_api, setup = cake_bench._candidate_call(shape, inputs)
    else:
        raise RuntimeError(f"unknown worker backend {args.worker_backend!r}")
    torch.cuda.synchronize()
    timing_utils = importlib.import_module("flashinfer.testing.utils")
    timing = cake_bench._measure_strict_cupti(
        timing_utils,
        call,
        samples=args.samples,
        warmup=args.warmup,
    )
    result = {
        "status": "measured",
        "backend": args.worker_backend,
        "public_api": public_api,
        **setup,
        **timing,
        "shape": shape.stable_id,
        "source_sha": source_sha,
        "hardware": hardware,
        "software": software,
    }
    if args.worker_backend == BACKEND_BASELINE:
        result["baseline_revision_proof"] = _baseline_revision_proof(source_root, torch)
    args.worker_json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_isolated(
    args: argparse.Namespace,
    *,
    backend: str,
    shape: MSAShape,
    output: Path,
) -> dict[str, Any]:
    if backend in (BACKEND_BASELINE, "verify-baseline"):
        worker_root = args.baseline_root.resolve()
        worker_sha = args.baseline_sha
    else:
        worker_root = args.candidate_root.resolve()
        worker_sha = args.candidate_sha
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--candidate-root",
        str(args.candidate_root.resolve()),
        "--candidate-sha",
        args.candidate_sha,
        "--baseline-root",
        str(args.baseline_root.resolve()),
        "--baseline-sha",
        args.baseline_sha,
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
        "--worker-source-root",
        str(worker_root),
        "--worker-source-sha",
        worker_sha,
        "--worker-backend",
        backend,
        "--worker-shape",
        shape.stable_id,
        "--worker-json",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(
            f"isolated {backend}/{shape.stable_id} worker failed "
            f"with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not output.is_file():
        raise RuntimeError(f"worker did not write {output}")
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("backend") != backend or result.get("shape") != shape.stable_id:
        raise RuntimeError(f"worker returned mismatched result: {result}")
    return result


def _selected_shapes(args: argparse.Namespace) -> tuple[MSAShape, ...]:
    if args.shapes is None:
        return SHAPE_MANIFEST
    if len(set(args.shapes)) != len(args.shapes):
        raise ValueError("--shapes must not contain duplicate stable IDs")
    requested = set(args.shapes)
    # Preserve manifest order even when the CLI list is reordered, keeping
    # backend alternation and output JSON deterministic.
    return tuple(shape for shape in SHAPE_MANIFEST if shape.stable_id in requested)


def _validate_cross_worker_metadata(results: list[dict[str, Any]]) -> None:
    if not results:
        raise RuntimeError("no measurements were collected")
    expected_hardware = results[0]["hardware"]
    expected_software = {
        key: value
        for key, value in results[0]["software"].items()
        if key != "flashinfer_version"
    }
    for result in results:
        if result["hardware"] != expected_hardware:
            raise RuntimeError("workers ran on different hardware")
        software = {
            key: value
            for key, value in result["software"].items()
            if key != "flashinfer_version"
        }
        if software != expected_software:
            raise RuntimeError("workers used different software environments")
        if result["activity_scope"] != ACTIVITY_SCOPE:
            raise RuntimeError("worker reported an unexpected activity scope")
        if result["timing_backend"] != "CUPTI":
            raise RuntimeError("worker did not use CUPTI")


def _require_correctness(result: dict[str, Any], shape: MSAShape) -> None:
    if result["status"] != "verified" or not result["correctness"]["passed"]:
        raise RuntimeError(
            f"independent-reference correctness failed for "
            f"{result['backend']}/{shape.stable_id}: {result['correctness']}"
        )


def _run_parent(args: argparse.Namespace) -> None:
    candidate_root = args.candidate_root.resolve()
    baseline_root = args.baseline_root.resolve()
    script_root = Path(__file__).resolve().parents[1]
    if script_root != candidate_root:
        raise RuntimeError(
            f"benchmark script must come from {candidate_root}, got {script_root}"
        )
    candidate_sha = cake_bench._validate_checkout(
        candidate_root, args.candidate_sha, "FlashInfer candidate source"
    )
    baseline_sha = cake_bench._validate_checkout(
        baseline_root, args.baseline_sha, "FlashInfer pinned PR4355 baseline"
    )

    selected_shapes = _selected_shapes(args)
    rows = []
    measured_results: list[dict[str, Any]] = []
    correctness_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="flashinfer-msa-vibecuda-bench-") as td:
        temp_root = Path(td)
        probe = _run_isolated(
            args,
            backend="probe-auto",
            shape=selected_shapes[0],
            output=temp_root / "probe-auto.json",
        )
        print(json.dumps(probe, sort_keys=True, allow_nan=False), flush=True)
        if probe["status"] not in ("available", "unavailable"):
            raise RuntimeError(f"unexpected probe-auto result: {probe}")
        for index, shape in enumerate(selected_shapes):
            print(
                f"Verifying {shape.stable_id} (each backend vs the independent "
                "FP32 reference)",
                flush=True,
            )
            by_correctness = {}
            for verify_backend in ("verify-baseline", "verify-candidate"):
                correctness_worker = _run_isolated(
                    args,
                    backend=verify_backend,
                    shape=shape,
                    output=temp_root / f"{index}-{verify_backend}.json",
                )
                correctness_results.append(correctness_worker)
                _require_correctness(correctness_worker, shape)
                by_correctness[verify_backend] = correctness_worker["correctness"]
            process_order = BACKENDS if index % 2 == 0 else tuple(reversed(BACKENDS))
            print(
                f"Measuring {shape.stable_id} ({', '.join(process_order)})",
                flush=True,
            )
            by_backend = {}
            for backend in process_order:
                result = _run_isolated(
                    args,
                    backend=backend,
                    shape=shape,
                    output=temp_root / f"{index}-{backend}.json",
                )
                by_backend[backend] = result
                measured_results.append(result)

            candidate = by_backend[BACKEND_CANDIDATE]
            baseline = by_backend[BACKEND_BASELINE]
            speedup = baseline["median_ms"] / candidate["median_ms"]
            if not math.isfinite(speedup) or speedup <= 0.0:
                raise RuntimeError(f"invalid speedup for {shape.stable_id}: {speedup}")
            row = {
                "shape": shape.as_public_dict(),
                "comparison_status": "measured",
                "correctness": {
                    "baseline_vs_independent_fp32_reference": by_correctness[
                        "verify-baseline"
                    ],
                    "candidate_vs_independent_fp32_reference": by_correctness[
                        "verify-candidate"
                    ],
                    "correctness_process": (
                        "one separate untimed process per backend per shape"
                    ),
                },
                "process_order": list(process_order),
                "baseline": baseline,
                "candidate": candidate,
                "speedup_baseline_over_candidate": speedup,
                "candidate_sha": candidate_sha,
                "baseline_sha": baseline_sha,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    cake_bench._validate_checkout(candidate_root, candidate_sha, "FlashInfer candidate")
    cake_bench._validate_checkout(baseline_root, baseline_sha, "FlashInfer baseline")
    _validate_cross_worker_metadata(measured_results)

    speedups = [row["speedup_baseline_over_candidate"] for row in rows]
    arithmetic_mean = statistics.fmean(speedups) if speedups else None
    geometric_mean = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        if speedups
        else None
    )
    first = measured_results[0]
    result = {
        "schema_version": 2,
        "manifest": {
            "version": cake_bench.MANIFEST_VERSION,
            "source": "benchmarks/bench_cake_msa_sm100.py:SHAPE_MANIFEST",
            "shape_count": len(SHAPE_MANIFEST),
            "selected_shape_ids": [shape.stable_id for shape in selected_shapes],
        },
        "comparison": {
            "candidate_backend": BACKEND_CANDIDATE,
            "candidate_public_api": CANDIDATE_PUBLIC_NAME,
            "candidate_sha": candidate_sha,
            "baseline_backend": (
                "FlashInfer PR4355 CAKE SM100/SM103 "
                "(flashinfer/msa_ops/_cake_sm100.py, csrc/cake_msa)"
            ),
            "baseline_public_api": BASELINE_PUBLIC_NAME,
            "baseline_repository": PR4355_SOURCE_REPOSITORY,
            "baseline_sha": baseline_sha,
            "baseline_expected_sha": PR4355_SOURCE_SHA,
            "auto_backend_probe": probe,
        },
        "hardware": first["hardware"],
        "software": first["software"],
        "protocol": {
            "timing_backend": "CUPTI",
            "cold_l2": True,
            "cuda_graph": False,
            "activity_scope": ACTIVITY_SCOPE,
            "included_gpu_activities": [
                "concurrent_kernel",
                "memcpy",
                "memset",
            ],
            "one_public_api_call_per_sample": True,
            "worker_isolation": "one_process_per_measured_backend_shape_pair",
            "correctness_worker_isolation": (
                "one_separate_untimed_process_per_backend_per_shape"
            ),
            "correctness_reference": (
                "independent torch FP32 masked-attention reference from "
                "benchmarks/bench_cake_msa_sm100.py, applied independently to "
                "the baseline and the candidate at the declared per-dtype "
                "tolerances (FP8-KV rows use the FP8 entry)"
            ),
            "zero_row_full_write_contract": ZERO_ROW_FULL_WRITE_CONTRACT,
            "fallback_policy": "reject",
            "samples_per_pair": args.samples,
            "additional_warmup_calls_per_pair": args.warmup,
            "speedup_formula": "baseline_median_ms / candidate_median_ms",
            "input_identity": (
                "Both backends reconstruct identical tensors, sparse block "
                "selections, sequence metadata, and page tables from each "
                "row's recorded seed and shape in the frozen manifest."
            ),
            "timed_region_parity": {
                "candidate": (
                    "exactly one public msa_sparse_attention/"
                    "msa_sparse_decode_attention call (backend='vibecuda') per "
                    "sample; input construction excluded; no plan/setup phase; "
                    "output allocated inside the timed call"
                ),
                "baseline": (
                    "exactly one public msa_sparse_attention/"
                    "msa_sparse_decode_attention call (PR4355 default CAKE "
                    "dispatch, no backend kwarg at the pinned revision) per "
                    "sample; input construction excluded; no CSR/schedule "
                    "preprocessing at this revision; output allocated inside "
                    "the timed call"
                ),
            },
        },
        "rows": rows,
        "summary": {
            "all_required_measurements_valid": True,
            "all_backends_independently_verified": True,
            "measured_comparisons": len(speedups),
            "arithmetic_mean_speedup": arithmetic_mean,
            "geometric_mean_speedup": geometric_mean,
            "minimum_speedup": min(speedups) if speedups else None,
            "maximum_speedup": max(speedups) if speedups else None,
        },
    }
    args.json.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.json}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-sha", default=PR4355_SOURCE_SHA)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=tuple(SHAPES_BY_ID),
        metavar="STABLE_ID",
        help=(
            "run only these frozen manifest rows (default: all rows); output "
            "records both the full manifest count and the selected stable IDs"
        ),
    )
    parser.add_argument(
        "--worker-source-root",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-source-sha", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-backend",
        choices=BACKENDS + ("verify-candidate", "verify-baseline", "probe-auto"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-shape",
        choices=tuple(SHAPES_BY_ID),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup <= 0:
        parser.error("--samples and --warmup must be positive")
    worker_values = (
        args.worker_source_root,
        args.worker_source_sha,
        args.worker_backend,
        args.worker_shape,
        args.worker_json,
    )
    if any(value is not None for value in worker_values):
        if not all(value is not None for value in worker_values):
            parser.error("all internal worker options must be supplied together")
        if args.json is not None:
            parser.error("--json is not valid in worker mode")
        if args.shapes is not None:
            parser.error("--shapes is not valid in worker mode")
    elif args.json is None:
        parser.error("--json is required")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker_backend is not None:
        _run_worker(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
