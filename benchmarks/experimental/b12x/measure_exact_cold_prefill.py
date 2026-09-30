"""Measure exact-length cold token-ID prefills and verify zero prefix reuse."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import httpx
from benchmarks.experimental.b12x.prompt_source_metrics import delta, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Duration must be positive and finite")
    if args.output.exists() or args.tokens < 8:
        parser.error("Use a fresh output path and at least eight tokens")
    if not (0 <= args.temperature <= 2 and 0 < args.top_p <= 1):
        parser.error("Temperature must be in [0, 2] and top-p in (0, 1]")
    samples = []
    result = {"conditions": {"prompt_tokens": args.tokens, "max_tokens": 1,
                             "duration_seconds": args.duration,
                             "temperature": args.temperature, "top_p": args.top_p,
                             "prompt_type": "deterministic cyclic token IDs with unique leading nonce"},
              "warmup": None, "samples": samples}
    with httpx.Client(timeout=600) as client:
        measured_started = None
        while (measured_started is None or not samples
               or time.monotonic() - measured_started < args.duration):
            nonce = time.time_ns()
            prompt = [1000 + byte for byte in nonce.to_bytes(8, "little")]
            prompt.extend(1400 + i % 127 for i in range(args.tokens - 8))
            before = metrics(args.base_url)
            started = time.monotonic()
            first = None
            usage = {}
            with client.stream("POST", f"{args.base_url}/v1/completions", json={
                "model": args.model, "prompt": prompt, "max_tokens": 1,
                "temperature": args.temperature, "top_p": args.top_p,
                "ignore_eos": True, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            }) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if first is None and (usage.get("completion_tokens", 0) > 0 or
                                          any(c.get("text") for c in chunk.get("choices", []))):
                        first = time.monotonic()
            assert first is not None and usage["prompt_tokens"] == args.tokens
            sources = delta(before, metrics(args.base_url))
            assert sources == {"external_kv_transfer": 0, "local_compute": args.tokens, "local_cache_hit": 0}, sources
            sample = {"nonce": nonce, "prompt_tokens": args.tokens,
                      "ttft_seconds": first - started,
                      "tok_per_sec": args.tokens / (first - started),
                      "prompt_sources": sources}
            if measured_started is None:
                result["warmup"] = sample
                measured_started = time.monotonic()
            else:
                samples.append(sample)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(sample), flush=True)
    result["median_tok_per_sec"] = statistics.median(s["tok_per_sec"] for s in samples)
    result["status"] = "qualified"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
