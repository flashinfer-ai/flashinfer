"""Deterministic decode-throughput benchmark against a running vLLM server.

Sends one /v1/completions request with temperature=0 and ignore_eos so the
model decodes exactly --tokens tokens, then reports wall-clock tok/s from the
server-reported usage. Stdlib only.

Usage:
    python scripts/bench_serving_tps.py --api-key KEY [--url http://localhost:8001] \
        [--model Qwen3.6-27B-FP6-W6A6] [--tokens 512] [--runs 3]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def run_once(args: argparse.Namespace) -> tuple[int, float]:
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.tokens,
        "temperature": 0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{args.url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = json.load(resp)
    dt = time.perf_counter() - t0
    completion_tokens = int(body["usage"]["completion_tokens"])
    return completion_tokens, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model", default="Qwen3.6-27B-FP6-W6A6")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--prompt",
        default="Explain how a CPU pipeline works, in detail, step by step.",
    )
    args = ap.parse_args()

    # Warmup run: absorbs Triton/CUTE JIT latency spikes and prefix-cache fill.
    run_once(args)

    rates = []
    for i in range(args.runs):
        n, dt = run_once(args)
        rate = n / dt
        rates.append(rate)
        print(f"run {i + 1}: {n} tokens in {dt:7.2f}s -> {rate:6.2f} tok/s")
    print(f"best: {max(rates):.2f} tok/s   mean: {sum(rates) / len(rates):.2f} tok/s")


if __name__ == "__main__":
    main()
