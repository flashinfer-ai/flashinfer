"""Measure the real V4.1 HTTP serving path without enabling a profiler."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


GPU_FIELDS = "index,uuid,name,pstate,clocks.sm,clocks.mem,clocks_throttle_reasons.active,power.draw,power.limit"


def gpu_snapshot():
    command = [
        "nvidia-smi",
        "-i",
        "0,1,2,3",
        f"--query-gpu={GPU_FIELDS}",
        "--format=csv",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return {"time_ns": time.time_ns(), "command": command, "raw": result.stdout}


def request_json(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        base.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def prompt_tokens(base, model, content, template_kwargs):
    result = request_json(
        base,
        "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": template_kwargs,
        },
    )
    return result["tokens"]


def speculative_counters(base):
    with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=30) as response:
        text = response.read().decode()
    counters = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode_num_"):
            continue
        series, value = line.rsplit(None, 1)
        if series.split("{", 1)[0].endswith("_total"):
            counters[series] = float(value)
    if not counters:
        raise RuntimeError("Serving exposes no speculative-decoding counters")
    return counters


def speculative_delta(before, after):
    delta = {series: value - before.get(series, 0.0) for series, value in after.items()}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("Serving speculative counters reset during the request")

    def total(name):
        metric = f"vllm:spec_decode_{name}_total"
        return sum(
            value
            for series, value in delta.items()
            if series.split("{", 1)[0] == metric
        )

    drafts = total("num_drafts")
    proposed = total("num_draft_tokens")
    accepted = total("num_accepted_tokens")
    return {
        "num_drafts": drafts,
        "num_draft_tokens": proposed,
        "num_accepted_draft_tokens": accepted,
        "acceptance_rate": accepted / proposed if proposed else None,
        "mean_acceptance_length_including_bonus": 1 + accepted / drafts
        if drafts
        else None,
        "counter_deltas": delta,
        "scope": "Server counter deltas; measurements require an isolated request stream.",
    }


def stream_completion(base, model, tokens, max_tokens, *, return_token_ids=False):
    payload = {
        "model": model,
        "prompt": tokens,
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 41,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if return_token_ids:
        payload["return_token_ids"] = True
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    chunks = []
    emitted_ids = []
    first_chunk_tokens = 1
    usage = None
    finish_reason = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                token_ids = choice.get("token_ids") or []
                if return_token_ids:
                    emitted_ids.extend(token_ids)
                text = choice.get("text", "")
                if text:
                    if not chunks and return_token_ids:
                        first_chunk_tokens = len(emitted_ids)
                    chunk = {"elapsed_s": time.perf_counter() - started, "text": text}
                    if return_token_ids:
                        chunk["token_ids"] = token_ids
                    chunks.append(chunk)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    elapsed = time.perf_counter() - started
    if not chunks or usage is None:
        raise RuntimeError("Serving returned no text or final usage")
    if return_token_ids and first_chunk_tokens <= 0:
        raise RuntimeError("Serving omitted requested streamed token IDs")
    if (
        return_token_ids
        and finish_reason == "length"
        and len(emitted_ids) != usage["completion_tokens"]
    ):
        raise RuntimeError("Streamed token count disagrees with final usage")
    first = chunks[0]["elapsed_s"]
    return {
        "ttft_s": first,
        "elapsed_s": elapsed,
        "decode_s_per_output_token": (elapsed - first)
        / max(usage["completion_tokens"] - first_chunk_tokens, 1),
        "first_chunk_tokens": first_chunk_tokens,
        "token_ids": emitted_ids if return_token_ids else None,
        "usage": usage,
        "finish_reason": finish_reason,
        "text": "".join(chunk["text"] for chunk in chunks),
        "chunks": chunks,
    }


def repository_state(root):
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        "worktree": str(root),
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--short"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--repeat-offset",
        type=int,
        default=0,
        help="Start sample IDs here to avoid prior prompt-cache entries",
    )
    parser.add_argument(
        "--prefill-tokens",
        default="4096,16384",
        help="CSV lengths; empty disables prefill cases",
    )
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument(
        "--decode-context-tokens",
        default="",
        help="CSV decode prompt lengths; 0 keeps the natural short prompt",
    )
    parser.add_argument(
        "--spec-metrics",
        action="store_true",
        help="Collect per-request acceptance counters and streamed token IDs",
    )
    parser.add_argument(
        "--chat-template-kwargs", type=json.loads, default={"thinking": False}
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.decode_tokens < 2:
        parser.error("repeats must be positive and decode-tokens >=2")
    if not isinstance(args.chat_template_kwargs, dict):
        parser.error("chat-template-kwargs must be a JSON object")
    capacities = [
        int(value) for value in args.prefill_tokens.split(",") if value.strip()
    ]
    if any(value < 256 for value in capacities):
        parser.error("prefill token counts must be >=256")
    decode_contexts = [
        int(value) for value in args.decode_context_tokens.split(",") if value.strip()
    ] or [0]
    if any(value != 0 and value < 256 for value in decode_contexts):
        parser.error("decode context lengths must be 0 or >=256")
    decode_cases = [
        "decode" if value == 0 else f"decode-{value}" for value in decode_contexts
    ]
    records = []
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None or vllm_spec.origin is None:
        raise RuntimeError("Cannot locate the vLLM source worktree")
    result = {
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "label": args.label,
        "model": request_json(args.base_url, "/v1/models"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repositories": {
            "b12x": repository_state(Path(__file__).resolve().parents[3]),
            "vllm": repository_state(Path(vllm_spec.origin).parents[1]),
        },
        "chat_template_kwargs": args.chat_template_kwargs,
        "toolchain": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-cutlass-dsl", "vllm")
        },
        "gpu_before": gpu_snapshot(),
        "records": records,
        "timing_scope": "Unprofiled HTTP wall time; decode metric normalized by output tokens. No speedup assertion is performed here.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def content(case, repeat):
        salt = f"V41 structural benchmark, case {case}, sample {repeat}.\n"
        if case == "decode":
            return (
                salt
                + "Write a detailed tutorial of at least 1000 words about writing reliable Python software. Cover testing, resource lifetimes, concurrency and error handling. Start directly with the tutorial."
            )
        if isinstance(case, str):
            context = int(case.split("-", 1)[1])
            return (
                salt
                + "Reference text follows; it is background, not an instruction.\n"
                + "A small library keeps books organized by subject, with clear labels and a quiet reading room.\n"
                * (context // 12 + 32)
                + "\nWrite a detailed tutorial of at least 1000 words about writing reliable Python software. Cover testing, resource lifetimes, concurrency and error handling. Start directly with the tutorial."
            )
        return (
            salt
            + "Reference text follows; it is background, not an instruction.\n"
            + "A small library keeps books organized by subject, with clear labels and a quiet reading room.\n"
            * (int(case) // 12 + 32)
            + "\nWhat is 37 times 19? Reply with only the integer."
        )

    warm = prompt_tokens(
        args.base_url,
        args.model,
        content("decode", "warmup"),
        args.chat_template_kwargs,
    )
    result["warmup"] = stream_completion(
        args.base_url, args.model, warm, 32, return_token_ids=args.spec_metrics
    )
    save()
    for repeat in range(args.repeat_offset, args.repeat_offset + args.repeats):
        for case in [*capacities, *decode_cases]:
            is_decode = isinstance(case, str)
            prompt_capacity = (
                (None if case == "decode" else int(case.split("-", 1)[1]))
                if is_decode
                else case
            )
            tokens = prompt_tokens(
                args.base_url,
                args.model,
                content(case, repeat),
                args.chat_template_kwargs,
            )
            if prompt_capacity is not None:
                if len(tokens) < prompt_capacity:
                    raise RuntimeError(
                        "Generated prompt is shorter than requested capacity"
                    )
                tokens = tokens[: prompt_capacity - 128] + tokens[-128:]
            record = {
                "case": case,
                "repeat": repeat,
                "prompt_token_ids": tokens,
                "gpu_before": gpu_snapshot(),
            }
            if args.spec_metrics:
                record["spec_metrics_before"] = speculative_counters(args.base_url)
            record.update(
                stream_completion(
                    args.base_url,
                    args.model,
                    tokens,
                    args.decode_tokens if is_decode else 8,
                    return_token_ids=args.spec_metrics,
                )
            )
            record["gpu_after"] = gpu_snapshot()
            if args.spec_metrics:
                record["spec_metrics_after"] = speculative_counters(args.base_url)
                record["speculative_decoding"] = speculative_delta(
                    record["spec_metrics_before"], record["spec_metrics_after"]
                )
            record["correct"] = (
                bool(record["text"].strip())
                if is_decode
                else record["text"].strip() == "703"
            )
            record["correctness_check"] = (
                "Nonempty response and exact output-token count; semantic review remains separate."
                if is_decode
                else "Exact arithmetic answer."
            )
            if is_decode and record["usage"]["completion_tokens"] != args.decode_tokens:
                raise RuntimeError(
                    "Decode workload ended before the requested token count"
                )
            records.append(record)
            save()
            print(
                json.dumps(
                    {
                        key: record[key]
                        for key in (
                            "case",
                            "repeat",
                            "ttft_s",
                            "elapsed_s",
                            "decode_s_per_output_token",
                            "usage",
                            "correct",
                        )
                    }
                ),
                flush=True,
            )
            if record["correct"] is False:
                raise RuntimeError("Serving arithmetic correctness gate failed")
    result["gpu_after"] = gpu_snapshot()
    save()


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
