# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Compare the VibeCUDA AlphaMoE router with pinned CAKE PR 4339.

The candidate and CAKE checkouts both provide the ``flashinfer`` package, so
the benchmark executes them in isolated Python processes and combines their
matched CUPTI measurements. Prepare the baseline checkout as documented in
``benchmarks/README.md``, then run::

    python3 benchmarks/bench_alphamoe_router.py \
      --candidate-python /tmp/flashinfer-vibecuda-venv/bin/python \
      --baseline-root /tmp/flashinfer-pr4339-baseline \
      --baseline-python /tmp/flashinfer-pr4339-venv/bin/python

Torch is only an independent correctness reference, never the denominator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

CAKE_PR = "https://github.com/flashinfer-ai/flashinfer/pull/4339"
CAKE_SHA = "0725744e58a9e338e8d315d82891878b07decd8f"
DRY_RUN_ITERS = 5
REPEAT_ITERS = 10


@dataclass(frozen=True)
class RouterConfig:
    name: str
    num_tokens: int
    num_experts: int
    top_k: int
    block_m: int
    has_shared_expert: bool


CONFIGS = (
    RouterConfig("single-1tok-e512-shared", 1, 512, 2, 16, True),
    RouterConfig("decode-8tok-e257-shared", 8, 257, 9, 8, True),
    RouterConfig("batch-32tok-e512", 32, 512, 8, 16, False),
    RouterConfig("batch-128tok-e512", 128, 512, 8, 16, False),
)


def _checkout_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _validate_baseline(root: Path) -> None:
    actual = _checkout_sha(root)
    if actual != CAKE_SHA:
        raise RuntimeError(
            f"CAKE baseline must be {CAKE_SHA}, got {actual} at {root}"
        )


def _clean_pythonpath(root: Path) -> str:
    candidate_root = Path(__file__).resolve().parents[1]
    entries = [str(root)]
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        resolved = Path(entry).resolve()
        if resolved != candidate_root:
            entries.append(str(resolved))
    return os.pathsep.join(entries)


def _run_worker(*, backend: str, root: Path, python: Path, output: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _clean_pythonpath(root)
    subprocess.run(
        [
            str(python),
            str(Path(__file__).resolve()),
            "--worker",
            backend,
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        check=True,
    )


def _worker(backend: str, output: Path) -> None:
    import numpy as np
    import torch

    from flashinfer.fused_moe import (
        allocate_alphamoe_route_plan,
        alphamoe_fused_router,
    )
    try:
        from flashinfer.testing import bench_gpu_time
    except ImportError:
        from flashinfer.testing.utils import bench_gpu_time

    capability = torch.cuda.get_device_capability()
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError(f"CC 10.0 or 10.3 required, got {capability}")

    rows: list[dict[str, object]] = []
    for case_index, config in enumerate(CONFIGS):
        generator = torch.Generator(device="cuda").manual_seed(29001 + case_index)
        logits = torch.randn(
            config.num_tokens,
            config.num_experts,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        plan = allocate_alphamoe_route_plan(
            logits,
            top_k=config.top_k,
            block_m=config.block_m,
            has_shared_expert=config.has_shared_expert,
        )

        if backend == "cake":

            def run() -> None:
                alphamoe_fused_router(
                    logits,
                    top_k=config.top_k,
                    block_m=config.block_m,
                    has_shared_expert=config.has_shared_expert,
                    plan=plan,
                )

        else:

            def run() -> None:
                alphamoe_fused_router(logits, plan=plan, backend="vibecuda")

        run()
        torch.cuda.synchronize()
        samples = bench_gpu_time(
            run,
            enable_cupti=True,
            dry_run_iters=DRY_RUN_ITERS,
            repeat_iters=REPEAT_ITERS,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
        rows.append(
            {
                "config": asdict(config),
                "median_us": float(np.median(samples)) * 1e3,
                "samples": len(samples),
            }
        )

    output.write_text(
        json.dumps(
            {
                "backend": backend,
                "device": torch.cuda.get_device_name(),
                "compute_capability": list(capability),
                "timing": {
                    "method": "CUPTI GPU activity",
                    "cold_l2": True,
                    "cuda_graph": False,
                    "dry_run_iters": DRY_RUN_ITERS,
                    "repeat_iters": REPEAT_ITERS,
                    "aggregation": "per-workload median",
                },
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


def _aggregate(candidate: dict, cake: dict) -> dict:
    if candidate["device"] != cake["device"]:
        raise RuntimeError("candidate and CAKE measurements used different devices")
    if candidate["timing"] != cake["timing"]:
        raise RuntimeError("candidate and CAKE timing protocols differ")
    rows = []
    for candidate_row, cake_row in zip(candidate["rows"], cake["rows"], strict=True):
        if candidate_row["config"] != cake_row["config"]:
            raise RuntimeError("candidate and CAKE workload manifests differ")
        speedup = cake_row["median_us"] / candidate_row["median_us"]
        rows.append(
            {
                "config": candidate_row["config"],
                "cake_us": cake_row["median_us"],
                "vibecuda_us": candidate_row["median_us"],
                "speedup": speedup,
            }
        )
    speedups = [row["speedup"] for row in rows]
    return {
        "baseline": {"name": "CAKE AlphaMoE router", "pr": CAKE_PR, "sha": CAKE_SHA},
        "device": candidate["device"],
        "timing": candidate["timing"],
        "rows": rows,
        "arithmetic_mean_speedup": statistics.fmean(speedups),
        "geometric_mean_speedup": math.exp(statistics.fmean(map(math.log, speedups))),
    }


def _print_result(result: dict) -> None:
    print(f"VibeCUDA AlphaMoE router vs CAKE PR 4339 ({CAKE_SHA[:12]})")
    print(
        "Protocol: CUPTI, cold L2, no CUDA Graph, "
        f"dry_run={DRY_RUN_ITERS}, repeats={REPEAT_ITERS}, median"
    )
    for row in result["rows"]:
        name = row["config"]["name"]
        print(
            f"{name:28s} CAKE {row['cake_us']:8.2f} us  "
            f"VibeCUDA {row['vibecuda_us']:8.2f} us  {row['speedup']:6.2f}x"
        )
    print(f"arithmetic mean: {result['arithmetic_mean_speedup']:.4f}x")
    print(f"geometric mean: {result['geometric_mean_speedup']:.4f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--worker", choices=("cake", "vibecuda"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        _worker(args.worker, args.output)
        return
    if args.baseline_root is None or args.baseline_python is None:
        parser.error("--baseline-root and --baseline-python are required")

    baseline_root = args.baseline_root.resolve()
    _validate_baseline(baseline_root)
    candidate_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="alphamoe-router-bench-") as tmp:
        tmp_path = Path(tmp)
        candidate_json = tmp_path / "candidate.json"
        cake_json = tmp_path / "cake.json"
        _run_worker(
            backend="vibecuda",
            root=candidate_root,
            python=args.candidate_python.resolve(),
            output=candidate_json,
        )
        _run_worker(
            backend="cake",
            root=baseline_root,
            python=args.baseline_python.resolve(),
            output=cake_json,
        )
        result = _aggregate(
            json.loads(candidate_json.read_text()), json.loads(cake_json.read_text())
        )
    _print_result(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
