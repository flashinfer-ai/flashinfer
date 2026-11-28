"""Model-health instrumentation for the FlashInfer LLM example.

Measures the lifecycle stages a serving deployment cares about and emits
them as both a human summary and machine-readable JSON — groundwork for
performance gates (each metric is a natural threshold candidate):

- ``import_seconds``   — ``import flashinfer`` in a fresh interpreter
- ``load_seconds``     — checkpoint -> device (weights + sharding)
- ``cold_prefill_seconds`` / ``jit_builds_cold`` — first forward, including
  JIT compilation / cubin loading (startup-preparation cost)
- ``warm_prefill_tokens_per_s`` — prefill throughput after warmup
- ``decode_tokens_per_s`` — steady-state decode throughput
- ``jit_builds_steady`` — must be 0 (no recompiles once serving)
- with ``--autotune``: decode/prefill re-measured after a
  ``flashinfer.autotune(True)`` warmup pass, reported as ``*_autotuned``

Timing is host-side (``perf_counter`` around synchronized calls): this is
health instrumentation for trend/gate purposes, not a kernel benchmark —
use ``flashinfer.testing.bench_gpu_time`` / CUPTI for kernel numbers.
CUDA-graph capture of the decode step is a planned addition (the wrappers
support ``use_cuda_graph=True`` with preallocated page-table buffers).

Usage:
    python health_check.py --tiny moe --json health.json
    python health_check.py --model-id Qwen/Qwen3-0.6B --batch 4 --prompt-len 512
    torchrun --nproc_per_node=2 health_check.py --tiny moe    # TEP (TP=EP)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import torch

from generate import GenerationEngine, install_jit_build_counter, maybe_init_distributed
from modeling import FlashInferLLM, resolve_checkpoint_dir


def measure_import_seconds() -> float:
    code = (
        "import time; t = time.perf_counter(); import flashinfer; "
        "print(time.perf_counter() - t)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    return float(out.stdout.strip().splitlines()[-1])


def timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", help="HF model id or local checkpoint path")
    parser.add_argument(
        "--tiny",
        choices=["dense", "moe"],
        help="use a tiny random-weight checkpoint (no download) instead of --model-id",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--prefill-iters", type=int, default=5)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--json", dest="json_path", help="write metrics JSON here")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not args.model_id and not args.tiny:
        parser.error("pass --model-id or --tiny")

    jit_counter = install_jit_build_counter()
    world_size, rank = maybe_init_distributed()
    if rank != 0:  # keep stdout single-writer; stderr stays visible
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 — process-lifetime sink
    device = torch.device("cuda", torch.cuda.current_device())

    metrics = {
        "gpu": torch.cuda.get_device_name(device),
        "world_size": world_size,
        "batch": args.batch,
        "prompt_len": args.prompt_len,
        "decode_tokens": args.decode_tokens,
    }

    # --- stage 0: import cost (fresh interpreter, rank 0 only) ---
    if rank == 0:
        metrics["import_seconds"] = round(measure_import_seconds(), 3)

    # --- resolve checkpoint ---
    tmpdir = None
    if args.tiny:
        from reference_check import build_tiny_checkpoint

        ckpt = None
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            ckpt = build_tiny_checkpoint(args.tiny, tmpdir, args.seed)
        if world_size > 1:
            obj = [ckpt]
            torch.distributed.broadcast_object_list(obj, src=0)
            ckpt = obj[0]
        metrics["model"] = f"tiny-{args.tiny}"
    else:
        ckpt = str(resolve_checkpoint_dir(args.model_id))
        metrics["model"] = args.model_id

    try:
        # --- stage 1: load ---
        t0 = time.perf_counter()
        model = FlashInferLLM.from_pretrained(
            ckpt, device, tp_size=world_size, tp_rank=rank
        )
        engine = GenerationEngine(model)
        metrics["load_seconds"] = round(time.perf_counter() - t0, 3)

        g = torch.Generator().manual_seed(args.seed + 1)
        vocab = model.config.vocab_size
        prompts = [
            torch.randint(0, vocab, (args.prompt_len,), generator=g).tolist()
            for _ in range(args.batch)
        ]
        prefill_tokens = args.batch * args.prompt_len

        # --- stage 2: cold prefill (JIT warmup / cubin loads) ---
        builds_before = jit_counter["count"]
        metrics["cold_prefill_seconds"] = round(
            timed(lambda: engine.prefill_logits(prompts)), 3
        )
        metrics["jit_builds_cold"] = jit_counter["count"] - builds_before

        def measure_throughput(tag: str):
            times = [
                timed(lambda: engine.prefill_logits(prompts))
                for _ in range(args.prefill_iters)
            ]
            metrics[f"warm_prefill_seconds{tag}"] = round(statistics.median(times), 4)
            metrics[f"warm_prefill_tokens_per_s{tag}"] = round(
                prefill_tokens / statistics.median(times), 1
            )
            result = engine.generate(
                prompts, max_new_tokens=args.decode_tokens, jit_counter=jit_counter
            )
            metrics[f"decode_tokens_per_s{tag}"] = round(
                result["decode_tokens"] / max(result["decode_seconds"], 1e-9), 1
            )
            return result

        # --- stage 3: warm throughput ---
        result = measure_throughput("")
        metrics["jit_builds_steady"] = result["jit_builds_steady"]
        metrics["jit_builds_total"] = jit_counter["count"]

        # --- stage 4 (optional): autotuned throughput ---
        if args.autotune:
            import flashinfer

            with flashinfer.autotune(True):
                engine.generate(prompts, max_new_tokens=4)
            metrics["autotune_warmup"] = True
            measure_throughput("_autotuned")

        print(
            f"\nhealth check: {metrics['model']} "
            f"({'TEP tp=ep=' + str(world_size) if world_size > 1 else '1 GPU'})"
        )
        for key, value in metrics.items():
            print(f"  {key:>32}: {value}")
        for key in ("jit_builds_steady",):
            if metrics.get(key, 0) != 0:
                print(f"UNHEALTHY: {key}={metrics[key]} (expected 0)")
                sys.exit(1)
        if args.json_path and rank == 0:
            with open(args.json_path, "w") as f:
                json.dump(metrics, f, indent=1)
            print(f"metrics written to {args.json_path}")
        print("[smoke] health=pass")
    finally:
        if rank == 0 and tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if world_size > 1:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
