# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Compare weight representations using one retained Rubin performance profile.

FC1 compares all eight shuffle/group/layout combinations. Plain BF16 compares
the four shuffle/layout combinations because it has no U/G grouping. Each
candidate passes a small numerical check and full-output equivalence before
timing. Timing order rotates across candidates; input/routing generation and
weight preprocessing stay outside the measurement.
"""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from benchmarks.bench_rubin_gemm import (
    ROOT,
    build_candidate,
    check_launch,
    configuration,
    measure,
    physical_gpu,
)


def representations(mode):
    groups = (64, 1) if mode == "fc1" else (1,)
    return {
        f"s{shuffle}-g{group}-{layout}": dict(
            use_shuffled_matrix=shuffle,
            gated_weight_group_size=group,
            weight_layout=layout,
        )
        for shuffle, group, layout in itertools.product(
            (0, 1), groups, ("block_major_k", "major_k")
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("e32", "e256", "e384"), required=True)
    parser.add_argument(
        "--mode", choices=("fc1", "routed-bf16", "dense-bf16"), default="fc1"
    )
    parser.add_argument("--inputs", choices=("normal", "matched"), default="normal")
    parser.add_argument("--fixture-dir", type=Path, help="Required for matched inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timer", choices=("cupti", "events"), default="cupti")
    parser.add_argument("--epilogue-regs", type=int)
    parser.add_argument("--non-epilogue-regs", type=int)
    parser.add_argument("--load-a-regs", type=int)
    parser.add_argument("--load-sfa-regs", type=int)
    parser.add_argument(
        "--compare-registers",
        action="store_true",
        help="Alternate retained and requested register budgets for every representation",
    )
    parser.add_argument(
        "--representation",
        action="append",
        help="Limit this comparison to named representations; may be repeated",
    )
    args = parser.parse_args()
    if args.inputs == "matched" and args.fixture_dir is None:
        parser.error("--fixture-dir is required for matched inputs")
    physical_gpu()
    if args.output.exists() or args.output.with_suffix(".partial.json").exists():
        parser.error(f"Preserving existing result: {args.output}")
    if min(args.windows, args.iterations) <= 0 or args.warmups < 0:
        parser.error("windows/iterations must be positive; warmups nonnegative")
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    os.environ.setdefault("FLASHINFER_PRIMS_TS_COMPILE_OPTIONS", "--opt-level 2")
    os.environ.setdefault("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    os.environ.setdefault("CUTE_DSL_KEEP", "ptx,cubin")

    import torch
    import cuda.bindings.driver as cuda
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner

    if torch.cuda.get_device_capability() != (10, 7):
        parser.error("These profiles require Rubin SM107")
    if args.timer == "cupti":
        from cutlass.testing import CuptiProfiler

        CuptiProfiler()
    variants = representations(args.mode)
    if args.representation:
        unknown = set(args.representation) - variants.keys()
        if unknown:
            parser.error(f"Unknown representations: {sorted(unknown)}")
        variants = {k: v for k, v in variants.items() if k in args.representation}
    register_options = {
        k: getattr(args, k)
        for k in ("epilogue_regs", "non_epilogue_regs", "load_a_regs", "load_sfa_regs")
        if getattr(args, k) is not None
    }
    if args.compare_registers:
        if not register_options:
            parser.error("--compare-registers requires a register override")
        variants = {
            name + suffix: {**opts, **regs}
            for name, opts in variants.items()
            for suffix, regs in (("-retained", {}), ("-registers", register_options))
        }
    else:
        variants = {
            name: {**opts, **register_options} for name, opts in variants.items()
        }
    shape, _ = configuration(args.case, args.mode)
    matched = None
    if args.inputs == "matched":
        from benchmarks.rubin_gemm_inputs import MatchedInputs

        matched = MatchedInputs(
            args.fixture_dir,
            shape,
            fc1=args.mode == "fc1",
        )
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    candidates = {}
    baseline_hashes = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, overrides in variants.items():
        _, options = configuration(args.case, args.mode, **overrides)
        assert runner.reference_check(
            num_experts=8,
            num_tokens=1031,
            top_k=8,
            problem_n=1280,
            problem_k=2304,
            routing="uniform",
            seed=args.seed,
            routing_seed=args.seed,
            repeat_launches=3,
            output_guard_elements=256,
            require_mixed_cga_runtime=True,
            **options,
        )
        settings = SimpleNamespace(
            **{
                **vars(args),
                "zero": False,
                "output": args.output.with_suffix(f".{name}.json"),
            }
        )
        candidate = build_candidate(
            args.inputs,
            settings,
            shape,
            options,
            stream,
            torch_stream,
            matched_inputs=matched,
        )
        if baseline_hashes is None:
            baseline_hashes = candidate["expected"]
        assert candidate["expected"] == baseline_hashes, (
            name,
            "Weight representations changed full output bytes",
        )
        candidate["metadata"].update(
            representation=name,
            full_output_matches_baseline=True,
            command=shlex.join(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.bench_rubin_weight_layouts",
                    *sys.argv[1:],
                ]
            ),
        )
        candidate["metadata"]["source_sha256"][
            str(Path(__file__).relative_to(ROOT))
        ] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        candidates[name] = candidate
        print("READY", args.case, args.mode, args.inputs, name, flush=True)

    names = list(candidates)
    result = dict(
        case=args.case,
        mode=args.mode,
        inputs=args.inputs,
        host=socket.gethostname(),
        physical_gpu=physical_gpu(),
        baseline=names[0],
        candidates={k: c["metadata"] for k, c in candidates.items()},
        sequence=[],
        complete=False,
    )
    telemetry = []
    try:
        for suffix, query in (
            (
                "clocks",
                "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
            ),
            (
                "processes",
                "--query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory",
            ),
        ):
            log = args.output.with_suffix(f".{suffix}.csv").open("w")
            proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    "-i",
                    str(physical_gpu()),
                    query,
                    "--format=csv",
                    "--loop-ms=100",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            telemetry.append((proc, log))
        for index in range(args.windows):
            offset = index % len(names)
            order = names[offset:] + names[:offset]
            if index // len(names) % 2:
                order.reverse()
            for name in order:
                candidate = candidates[name]
                before = time.time()
                us, durations = measure(
                    candidate["compiled"],
                    candidate["launch_args"],
                    torch_stream,
                    timer=args.timer,
                    warmups=args.warmups,
                    iterations=args.iterations,
                )
                window = dict(
                    index=index,
                    representation=name,
                    us=us,
                    durations_us=durations,
                    start=before,
                    end=time.time(),
                )
                candidate["metadata"]["windows"].append(window)
                result["sequence"].append(window)
                args.output.with_suffix(".partial.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print("WINDOW", index, name, us, flush=True)
    finally:
        for proc, log in telemetry:
            proc.terminate()
            proc.wait(timeout=10)
            log.close()
    for name, candidate in candidates.items():
        check_launch(
            candidate["compiled"],
            candidate["launch_args"],
            candidate["buffers"],
            candidate["guards"],
            candidate["expected"],
            zero=False,
        )
        metadata = candidate["metadata"]
        if matched is not None:
            metadata["post_matched_reference"] = matched.check(candidate["buffers"])
        us = statistics.median(w["us"] for w in metadata["windows"])
        metadata.update(
            us=us,
            tflops=2
            * candidate["valid_rows"]
            * shape["problem_n"]
            * shape["problem_k"]
            / (us * 1e6),
            complete=True,
            post_benchmark_guards_and_hashes_passed=True,
        )
        print(
            "RESULT",
            args.case,
            args.mode,
            args.inputs,
            name,
            us,
            "us",
            metadata["tflops"],
            "TFLOPS",
            flush=True,
        )
    baseline_us = candidates[names[0]]["metadata"]["us"]
    for candidate in candidates.values():
        candidate["metadata"]["baseline_over_candidate"] = (
            baseline_us / candidate["metadata"]["us"]
        )
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
