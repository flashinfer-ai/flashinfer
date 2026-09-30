#!/usr/bin/env python3
"""Measure engine-side prefill latency with exact token-ID prompts.

Requests use deterministic token sequences and an invocation-specific vLLM
cache salt. Observed cache hits are rejected before throughput is summarized.
Prometheus counter deltas separate queue time from model prefill time; wall
time remains useful for engines that do not expose those counters.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.request
import uuid
from pathlib import Path


METRICS = (
    "vllm:e2e_request_latency_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
)
CACHE_METRICS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)


def request_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def read_metrics(base_url: str) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
            text = response.read().decode()
    except Exception:
        return {}

    result: dict[str, float] = {}
    names = [metric + suffix for metric in METRICS for suffix in ("_sum", "_count")]
    for name in [*names, *CACHE_METRICS]:
        values = [
            float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.startswith(name + "{") or line.startswith(name + " ")
        ]
        if values:
            result[name] = sum(values)
    return result


def check_uncached(before, after, usage, *, sglang_reports_only_hits=False) -> float | None:
    details = usage.get("prompt_tokens_details") or {}
    reported = details.get("cached_tokens")
    # SGLang with enable_cache_report omits this object specifically for zero
    # hits. This interpretation requires an explicit, verified serving contract.
    if reported is None and sglang_reports_only_hits:
        reported = 0
    metric = "vllm:prefix_cache_hits_total"
    observed = after[metric] - before[metric] if metric in before and metric in after else None
    if reported is None and observed is None:
        raise RuntimeError("Cannot qualify cold prefill without cache-hit evidence")
    for value in (reported, observed):
        if value is not None and value != 0:
            raise RuntimeError(
                f"Refusing an uncached-prefill measurement with {value} cached tokens"
            )
    return reported if reported is not None else observed


def metric_delta(
    before: dict[str, float], after: dict[str, float], metric: str
) -> float | None:
    count_name = metric + "_count"
    sum_name = metric + "_sum"
    if count_name not in before or count_name not in after:
        return None
    if after[count_name] - before[count_name] != 1:
        return None
    return after[sum_name] - before[sum_name]


def main() -> None:
    from qwen_sampling import DEFAULT_SAMPLING, temperature

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--temperature", type=temperature, default=1.0)
    parser.add_argument("--cache-salt", default=None,
                        help="vLLM cache namespace; default is unique per invocation")
    parser.add_argument("--sglang-flush-cache", action="store_true",
                        help="Verify cache reporting and flush an idle SGLang cache before each request")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    if args.sglang_flush_cache:
        with urllib.request.urlopen(base_url + "/server_info", timeout=30) as response:
            server_info = json.load(response)
        if server_info.get("enable_cache_report") is not True:
            raise RuntimeError("SGLang must expose cache reporting for an uncached qualification")
    samples: list[dict[str, object]] = []
    cache_salt = args.cache_salt or uuid.uuid4().hex
    total_runs = args.warmups + args.samples
    for run in range(total_runs):
        if args.sglang_flush_cache:
            from measure_decode_corpus import wait_for_idle

            wait_for_idle(base_url)
            # Response delivery can precede retirement of an overlapped batch.
            # SGLang's deferred flush waits for its full internal idle condition;
            # zero scheduler gauges alone do not imply that condition is met.
            request = urllib.request.Request(base_url + "/flush_cache?timeout=30", method="POST")
            with urllib.request.urlopen(request, timeout=45) as response:
                if response.status != 200 or not response.read().startswith(b"Cache flushed."):
                    raise RuntimeError("SGLang did not confirm cache flush")
        rng = random.Random(args.seed + run)
        token_ids = [rng.randrange(1000, 150000) for _ in range(args.tokens)]
        payload: dict[str, object] = {
            "model": args.model,
            "prompt": token_ids,
            "max_tokens": 1,
            **DEFAULT_SAMPLING,
            "temperature": args.temperature,
            "ignore_eos": True,
        }
        before = read_metrics(base_url)
        if not args.sglang_flush_cache:
            payload["cache_salt"] = cache_salt
        started = time.perf_counter()
        response = request_json(f"{base_url}/v1/completions", payload)
        wall_seconds = time.perf_counter() - started
        after = read_metrics(base_url)
        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", args.tokens))
        if prompt_tokens != args.tokens:
            raise RuntimeError(f"Expected {args.tokens} input tokens, got {prompt_tokens}")
        cached_tokens = check_uncached(before, after, usage,
                                       sglang_reports_only_hits=args.sglang_flush_cache)
        sample: dict[str, object] = {
            "run": run,
            "warmup": run < args.warmups,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "wall_seconds": wall_seconds,
            "wall_tokens_per_second": prompt_tokens / wall_seconds,
        }
        for metric in METRICS:
            sample[metric.removeprefix("vllm:")] = metric_delta(
                before, after, metric
            )
        prefill_seconds = sample["request_prefill_time_seconds"]
        sample["engine_prefill_tokens_per_second"] = (
            prompt_tokens / prefill_seconds if prefill_seconds else None
        )
        samples.append(sample)
        print(json.dumps(sample, sort_keys=True))

    measured = [sample for sample in samples if not sample["warmup"]]
    summary = {
        "base_url": base_url,
        "model": args.model,
        "tokens": args.tokens,
        "warmups": args.warmups,
        "samples": args.samples,
        "seed": args.seed,
        "sampling": {**DEFAULT_SAMPLING, "temperature": args.temperature},
        "cache_salt": cache_salt,
        "cache_control": "sglang_flush" if args.sglang_flush_cache else "vllm_salt_and_observed_zero_hits",
        "median_wall_tokens_per_second": statistics.median(
            float(sample["wall_tokens_per_second"]) for sample in measured
        ),
        "median_engine_prefill_tokens_per_second": statistics.median(
            float(sample["engine_prefill_tokens_per_second"])
            for sample in measured
            if sample["engine_prefill_tokens_per_second"] is not None
        )
        if any(
            sample["engine_prefill_tokens_per_second"] is not None
            for sample in measured
        )
        else None,
        "runs": samples,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
