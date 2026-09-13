"""Synthetic low-M BF16 GEMM: frozen default graphs versus autotuned graphs.

Run each shape in a fresh process on an exclusively allocated SM100/SM103 GPU:
    python benchmarks/bench_bf16_low_m.py --m 1 --n 43008 --k 5376 --output wide.json
    python benchmarks/bench_bf16_low_m.py --m 1 --n 5376 --k 21504 --output deep.json

Requires CuTe DSL >=4.7 and cupti-python >=13. No private model data is needed.
No dispatch changes are made. Timings include all kernels in each GEMM call,
exclude L2 eviction, and are not end-to-end serving throughput measurements.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


class BenchmarkGateError(RuntimeError):
    """A fixed, benchmark-authored message safe to include in a shared receipt."""


def positive_int(value):
    """Parse a strictly positive CLI dimension or iteration count."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args(argv=None):
    """Parse one low-M shape without importing or initializing CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    for axis in ("m", "n", "k"):
        parser.add_argument(f"--{axis}", type=positive_int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.m > 32:
        parser.error("this reproduction is limited to M <= 32")
    return args


def check_rows(actual, reference):
    """Validate every row against FP32, including zero-reference edge cases."""
    import torch

    if actual.ndim != 2 or actual.shape != reference.shape:
        raise ValueError("expected matching two-dimensional outputs")
    x, y = actual.float(), reference.float()
    if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
        raise AssertionError("non-finite output or reference")
    xnorm = torch.linalg.vector_norm(x, dim=1)
    ynorm = torch.linalg.vector_norm(y, dim=1)
    errnorm = torch.linalg.vector_norm(x - y, dim=1)
    zero = ynorm == 0
    exact_zero = zero & (xnorm == 0)
    nrmse = torch.where(
        zero,
        torch.where(exact_zero, 0.0, float("inf")),
        errnorm / ynorm.clamp_min(1e-30),
    )
    cosine = torch.where(
        zero, exact_zero.float(), (x * y).sum(1) / (xnorm * ynorm).clamp_min(1e-30)
    )
    if not bool(((nrmse <= 1 / 256) & (cosine >= 0.99999)).all()):
        raise AssertionError("FP32 per-row NRMSE/cosine gate failed")
    return {"max_nrmse": nrmse.max().item(), "min_cosine": cosine.min().item()}


def bracket(before, candidate, after):
    """Summarize one A/B/A trial using the mean of the two control medians."""
    medians = [statistics.median(v) for v in (before, candidate, after)]
    if not all(
        math.isfinite(x) and x > 0
        for values in (before, candidate, after)
        for x in values
    ):
        raise ValueError("timings must be finite and positive")
    a1, b, a2 = medians
    control = (a1 + a2) / 2
    return {
        "before_us": a1,
        "candidate_us": b,
        "after_us": a2,
        "speedup": control / b,
        "control_drift": abs(a1 - a2) / control,
    }


def run(args, result):
    """Validate and time frozen default/tuned graphs, retaining partial results."""
    if os.environ.get("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0") != "0":
        raise BenchmarkGateError(
            "disable inherited file-based tuning for a default comparison"
        )
    import torch
    import flashinfer
    from flashinfer import autotune, mm_bf16
    from flashinfer.autotuner import AutoTuner
    from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
    from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
        default_tactic,
        run_direct_dense,
    )
    from flashinfer.testing.utils import bench_gpu_time_with_cupti
    from cupti import cupti  # noqa: F401 -- require CUPTI, never silently time events

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise BenchmarkGateError("requires SM100 or SM103")
    if not is_cute_dsl_experimental_available():
        raise BenchmarkGateError(
            "requires the current warp-capable CuTe DSL >=4.7 runtime"
        )
    if int(importlib.metadata.version("cupti-python").split(".")[0]) < 13:
        raise BenchmarkGateError("requires cupti-python >=13")
    root = Path(flashinfer.__file__).resolve().parent.parent
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--exit-code",
            "--quiet",
            "HEAD",
            "--",
            "flashinfer",
        ],
        check=True,
    )
    result["environment"] = {
        "source_sha": sha,
        "flashinfer": flashinfer.__version__,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "cupti": importlib.metadata.version("cupti-python"),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "timer": "CUPTI, cold-L2, complete CUDA-graph replay, microseconds",
    }
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # Only this fresh process's in-memory tuning state is cleared; no disk cache.
    AutoTuner.get().clear_cache()
    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16).T
    initial_a = a.clone()
    b_fp32 = b.float()
    reference = a.float() @ b_fp32
    graphs, outputs = {}, {}
    checks = result.setdefault("correctness", {})

    def capture(name, fn, out, tune=False):
        """Check eager output and freeze the selected kernels before cache changes."""
        print(json.dumps({"event": "prepare", "arm": name}), flush=True)
        with autotune(tune):
            fn()
        checks[name] = [check_rows(out, reference)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name], outputs[name] = graph, out
        print(json.dumps({"event": "captured", "arm": name}), flush=True)

    # Freeze both default graphs before ANY autotuning; later cache choices
    # cannot change kernels already captured in these graphs.
    for tuned in (False, True):
        for backend in ("cublaslt", "cute-dsl"):
            name = backend + ("_tuned" if tuned else "_default")
            out = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)

            def call(backend=backend, out=out):
                """Bind each backend and output before the surrounding loop advances."""
                return mm_bf16(a, b, out=out, backend=backend, pdl=False)

            capture(name, call, out, tuned)
    tactic = default_tactic(args.m, args.n, args.k)
    out = torch.empty_like(outputs["cute-dsl_default"])
    capture("direct_default", lambda: run_direct_dense(a, b, out, False, tactic), out)

    for mutate in (False, True):
        for _ in range(10):
            if mutate:
                a.normal_()
                reference = a.float() @ b_fp32
            for name, graph in graphs.items():
                graph.replay()
                checks[name].append(check_rows(outputs[name], reference))
    a.copy_(initial_a)
    torch.cuda.synchronize()
    print(json.dumps({"event": "correctness_passed", "arms": list(graphs)}), flush=True)
    result["timings"] = {}

    def measure(name):
        """Collect complete cold-L2 graph replay durations in microseconds."""
        times = bench_gpu_time_with_cupti(
            graphs[name].replay,
            dry_run_iters=20,
            repeat_iters=args.iterations,
            use_cuda_graph=False,
            cold_l2_cache=True,
        )
        if len(times) != args.iterations:
            raise BenchmarkGateError("CUPTI did not report every requested replay")
        return [float(t) * 1000 for t in times]

    for name in graphs:
        if name == "cublaslt_tuned":
            continue
        trials = result["timings"].setdefault(name, [])
        for _ in range(3):
            a1, candidate, a2 = (
                measure("cublaslt_tuned"),
                measure(name),
                measure("cublaslt_tuned"),
            )
            summary = bracket(a1, candidate, a2)
            trials.append(
                {
                    **summary,
                    "samples_us": {"before": a1, "candidate": candidate, "after": a2},
                }
            )
            print(
                json.dumps(
                    {
                        "event": "bracket",
                        "arm": name,
                        "trial": len(trials),
                        **summary,
                    }
                ),
                flush=True,
            )
    result["status"] = "complete"


def main(argv=None):
    """Write a non-overwriting receipt while preserving any failure traceback."""
    args = parse_args(argv)
    result = {
        "shape": {a: getattr(args, a) for a in ("m", "n", "k")},
        "seed": args.seed,
        "pdl": False,
        "bias": False,
        "status": "incomplete",
    }
    # Refuse overwriting a previous run; retain partial evidence if a gate fails.
    with args.output.open("x") as output:
        try:
            run(args, result)
        except Exception as exc:
            result["error_type"] = type(exc).__name__
            # Third-party exception text can contain paths or command arguments.
            result["error_message"] = (
                str(exc)
                if isinstance(exc, BenchmarkGateError)
                else "Failure details are available in stderr."
            )
            raise
        finally:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")


if __name__ == "__main__":
    main()
