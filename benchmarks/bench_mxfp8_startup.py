"""Measure MXFP8 cold compilation, disk reuse, and cached-winner startup.

Each case runs in a fresh process. The baseline disables disk persistence and
precompilation but keeps the current kernel/candidate set. It is not a replay
of an older release or a full vLLM model. Use --output-dir with a new directory.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def child(args):
    import statistics
    import torch
    import torch.nn.functional as F
    import flashinfer
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fp8_quantization import mxfp8_quantize
    from flashinfer.gemm import gemm_mm_fp4_cute_dsl as compiler
    from flashinfer.gemm import gemm_mm_mxfp8_precompile as precompile
    from flashinfer.testing import bench_gpu_time

    counts = {"parent_compile_calls": 0, "profile_calls": 0, "pool_tasks": 0}
    phase_wall = {"parent_compile_s": 0.0, "profile_s": 0.0, "pool_s": 0.0}
    make_compile = compiler._make_blockscaled_gemm_compile_fn

    def traced_make(*a, **kw):
        fn = make_compile(*a, **kw)

        def compile_fn():
            counts["parent_compile_calls"] += 1
            if args.forbid_compile:
                raise RuntimeError("warm process unexpectedly compiled a GEMM")
            start = time.perf_counter()
            try:
                return fn()
            finally:
                phase_wall["parent_compile_s"] += time.perf_counter() - start

        return compile_fn

    compiler._make_blockscaled_gemm_compile_fn = traced_make
    profile = AutoTuner._profile_single_kernel

    def traced_profile(self, *a, **kw):
        counts["profile_calls"] += 1
        start = time.perf_counter()
        try:
            return profile(self, *a, **kw)
        finally:
            phase_wall["profile_s"] += time.perf_counter() - start

    AutoTuner._profile_single_kernel = traced_profile
    pool = precompile._run_mm_mxfp8_precompile_pool

    def traced_pool(payloads, workers):
        counts["pool_tasks"] += len(payloads)
        start = time.perf_counter()
        try:
            return pool(payloads, workers)
        finally:
            phase_wall["pool_s"] += time.perf_counter() - start

    precompile._run_mm_mxfp8_precompile_pool = traced_pool
    if args.forbid_compile:

        def unexpected_pool(*a, **kw):
            counts["unexpected_pool"] = True
            raise RuntimeError("warm process unexpectedly started a compile pool")

        precompile._run_mm_mxfp8_precompile_pool = unexpected_pool

    torch.manual_seed(1234)
    buckets = tuple(int(x) for x in args.buckets.split(","))
    m = max(buckets)
    weight = torch.randn((args.n, args.k), dtype=torch.bfloat16, device="cuda")
    w, ws = mxfp8_quantize(weight, is_sf_swizzled_layout=True)
    operands = {}
    for rows in buckets:
        x = torch.randn((rows, args.k), dtype=torch.bfloat16, device="cuda")
        a, scale = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        operands[rows] = (
            x,
            a,
            scale,
            torch.empty((rows, args.n), device="cuda", dtype=torch.bfloat16),
        )

    def run(rows):
        _, a, scale, out = operands[rows]
        return flashinfer.mm_mxfp8(
            a, w.T, scale, ws, out=out, out_dtype=torch.bfloat16, backend="cute-dsl"
        )

    tuner = AutoTuner.get()
    configs = Path(os.environ["FLASHINFER_WORKSPACE_BASE"]) / "winners.json"
    if args.load_winners:
        assert tuner.load_configs(str(configs))
    torch.cuda.synchronize()
    start = time.perf_counter()
    with flashinfer.autotune(tuning_buckets=buckets):
        run(m)
    torch.cuda.synchronize()
    tune_wall = time.perf_counter() - start
    if args.load_winners:
        assert counts["profile_calls"] == 0, "saved winners unexpectedly retuned"
    tuner.save_configs(str(configs))
    # Keep the override active during replay so bucket key identities match.
    validations = []
    with flashinfer.autotune(False, tuning_buckets=buckets):
        for rows in buckets:
            output = run(rows)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(rows)
            graph.replay()
            torch.cuda.synchronize()
            reference = operands[rows][0].float() @ weight.float().T
            cosine = F.cosine_similarity(
                reference.flatten(), output.float().flatten(), dim=0
            ).item()
            assert torch.isfinite(output).all() and cosine > 0.98, (rows, cosine)
            times = bench_gpu_time(
                lambda: run(rows),
                enable_cupti=False,
                use_cuda_graph=True,
                cold_l2_cache=False,
                dry_run_iters=5,
                repeat_iters=20,
            )
            validations.append(
                {
                    "m": rows,
                    "cosine_after_graph": cosine,
                    "warm_graph_median_ms": statistics.median(times),
                }
            )
    if args.forbid_compile:
        assert counts["parent_compile_calls"] == 0 and not counts.get("unexpected_pool")
    result = dict(
        case=args.child_case,
        gpu=torch.cuda.get_device_name(),
        cc=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        flashinfer_version_metadata=flashinfer.__version__,
        flashinfer_source=str(Path(flashinfer.__file__).resolve()),
        n=args.n,
        k=args.k,
        buckets=buckets,
        tune_wall_s=tune_wall,
        warm_graph_median_ms=next(
            item["warm_graph_median_ms"] for item in validations if item["m"] == m
        ),
        **counts,
        # Profile wall includes on-demand compilation: these spans overlap.
        phase_wall=phase_wall,
        validations=validations,
        object_count=len(
            list(Path(os.environ["FLASHINFER_WORKSPACE_BASE"]).rglob("*mm_mxfp8*/*.o"))
        ),
    )
    (Path(args.output_dir) / f"{args.child_case}.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--buckets", default="8,64,256")
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--child-case")
    parser.add_argument("--forbid-compile", action="store_true")
    parser.add_argument("--load-winners", action="store_true")
    args = parser.parse_args()
    if args.child_case:
        child(args)
        return
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = [
        ("baseline", "baseline", 0, True, False, False),
        ("disk_cold", "serial", 1, False, False, False),
        ("disk_warm", "serial", 1, False, True, False),
        ("cached_winners", "serial", 1, False, True, True),
        ("parallel_cold", "parallel", 4, False, False, False),
        ("parallel_warm", "parallel", 4, False, True, False),
    ]
    summary = []
    for name, cache, workers, disabled, forbid, winners in cases:
        env = dict(
            os.environ,
            FLASHINFER_WORKSPACE_BASE=str(root / cache),
            FLASHINFER_MM_MXFP8_CUTE_DSL_COMPILE_WORKERS=str(workers),
            FLASHINFER_CUTE_DSL_DISABLE_CACHE=str(int(disabled)),
            FLASHINFER_MM_MXFP8_CUTE_DSL_MAX_TACTICS=str(args.top_n),
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-dir",
            str(root),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
            "--buckets",
            args.buckets,
            "--child-case",
            name,
        ]
        if forbid:
            command.append("--forbid-compile")
        if winners:
            command.append("--load-winners")
        start = time.perf_counter()
        with (root / f"{name}.log").open("w") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        result = json.loads((root / f"{name}.json").read_text())
        result["process_wall_s"] = time.perf_counter() - start
        result["top_n"] = args.top_n
        summary.append(result)
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
