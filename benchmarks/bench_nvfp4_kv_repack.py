"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

DESCRIPTION = """Reproduce FlashInfer #4775 with signed FP4 data and a dequantized reference.

Run one KV dtype per process. Select the source checkout with PYTHONPATH and
give each checkout its own FLASHINFER_WORKSPACE_BASE. Timings exclude plan,
quantization, compilation, and reference checking. The selected checkout must
be clean, including untracked files. Keep benchmark outputs outside that checkout.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

import torch
import flashinfer


CASES = {
    "b1-2048-d128": (1, 2048, 2048, 128),
    "b1-4096-d128": (1, 4096, 4096, 128),
    "b4-2048-d128": (4, 2048, 2048, 128),
    "b8-1024-d128": (8, 1024, 1024, 128),
    "b2-2048-d256": (2, 2048, 2048, 256),
    "tail-d128": (2, 113, 257, 128),
    "tail-d256": (2, 113, 257, 256),
    "fallback-d512": (2, 113, 257, 512),
    "short-q-d128": (2, 1, 257, 128),
    "b1-128-d256": (1, 128, 128, 256),
    "b1-256-d256": (1, 256, 256, 256),
    "b1-512-d256": (1, 512, 512, 256),
    "b1-1024-d256": (1, 1024, 1024, 256),
    "b1-4096-d256": (1, 4096, 4096, 256),
}
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "nvfp4": torch.uint8,
}


def positive_int(value):
    """Parse a strictly positive integer for benchmark dimensions and sampling."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def source_commit(source):
    """Require a clean imported checkout before attributing results to its HEAD."""
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise RuntimeError(
            f"Benchmark requires a clean source checkout: {source}\n{status}"
        )
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def write_results(output, result):
    """Preserve completed results if writing the next JSON snapshot fails."""
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as tmp:
        temporary = Path(tmp) / output.name
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)


def make_kv(shape, dtype):
    packed = torch.randint(0, 256, shape, device="cuda", dtype=torch.uint8)
    choices = torch.tensor([0, 1, 16, 32, 40, 48, 56], device="cuda", dtype=torch.uint8)
    scales = choices[
        torch.randint(0, len(choices), (*shape[:-1], shape[-1] // 8), device="cuda")
    ]
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = lut[codes].reshape(*shape[:-1], shape[-1] * 2)
    dequant = (
        values * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    ).to(dtype)
    return packed, scales, dequant


def graph_times(run, rounds, repeats):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    calls_per_graph = 10
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            run()
    # Stabilize clocks after JIT compilation and reference work.
    for _ in range(30):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (repeats * calls_per_graph))
    return samples


def run_case(name, args):
    batch, qo_len, kv_len, dim = CASES[name]
    torch.manual_seed(4775)
    qdtype = DTYPES[args.q_dtype]
    q = torch.randn(batch * qo_len, args.q_heads, dim, dtype=qdtype, device="cuda")
    pages_per_seq = (kv_len + args.page_size - 1) // args.page_size
    pages = batch * pages_per_seq
    shape = (pages, 2, args.page_size, args.kv_heads, dim // 2)
    packed, scales, dequant = make_kv(shape, qdtype)
    if args.layout == "HND" and args.mode == "paged":
        packed, scales, dequant = [
            x.transpose(2, 3).contiguous() for x in (packed, scales, dequant)
        ]
    kv = packed if args.kv_dtype == "nvfp4" else dequant.to(DTYPES[args.kv_dtype])
    ref_kv = dequant if args.kv_dtype == "nvfp4" else kv.to(qdtype)
    if args.mode == "ragged":

        def as_ragged(x):
            x = x.transpose(1, 2).reshape(
                batch, pages_per_seq * args.page_size, 2, args.kv_heads, x.shape[-1]
            )
            x = x[:, :kv_len].reshape(batch * kv_len, 2, args.kv_heads, x.shape[-1])
            tensors = x.unbind(1)
            if args.layout == "HND":
                tensors = [t.transpose(0, 1) for t in tensors]
            return tuple(t.contiguous() for t in tensors)

        kv, ref_kv, scales = [as_ragged(x) for x in (kv, ref_kv, scales)]
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32) * pages_per_seq
    indices = torch.arange(pages, dtype=torch.int32)
    last_page_len = torch.full(
        (batch,), (kv_len - 1) % args.page_size + 1, dtype=torch.int32
    )
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def plan(dtype):
        if args.mode == "ragged":
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, args.layout, backend="fa2"
            )
            wrapper.plan(
                qo_indptr,
                torch.arange(batch + 1, dtype=torch.int32) * kv_len,
                args.q_heads,
                args.kv_heads,
                dim,
                causal=args.causal,
                pos_encoding_mode=args.pos_encoding_mode,
                q_data_type=qdtype,
                kv_data_type=dtype,
            )
            return wrapper
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, args.layout, backend="fa2"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            indices,
            last_page_len,
            args.q_heads,
            args.kv_heads,
            dim,
            args.page_size,
            causal=args.causal,
            pos_encoding_mode=args.pos_encoding_mode,
            q_data_type=qdtype,
            kv_data_type=dtype,
        )
        return wrapper

    print(
        f"Preparing {name} {args.q_dtype}/{args.kv_dtype} {args.mode}/{args.layout}",
        flush=True,
    )
    ref_args = ref_kv if args.mode == "ragged" else (ref_kv,)
    reference = plan(qdtype).run(q, *ref_args)
    torch.cuda.synchronize()
    wrapper = plan(DTYPES[args.kv_dtype])
    out = torch.empty_like(q)
    kwargs = {"kv_cache_sf": scales} if args.kv_dtype == "nvfp4" else {}

    def run():
        kv_args = kv if args.mode == "ragged" else (kv,)
        return wrapper.run(q, *kv_args, out=out, **kwargs)

    actual = run()
    delta = (actual.float() - reference.float()).abs()
    max_abs, mean_abs = delta.max().item(), delta.mean().item()
    atol = 0.002 if qdtype == torch.float16 else 0.02
    torch.testing.assert_close(actual, reference, atol=atol, rtol=atol)
    times = graph_times(run, args.rounds, args.repeats) if not args.check_only else []
    return {
        "case": name,
        "batch": batch,
        "qo_len": qo_len,
        "kv_len": kv_len,
        "head_dim": dim,
        "q_heads": args.q_heads,
        "kv_heads": args.kv_heads,
        "q_dtype": args.q_dtype,
        "kv_dtype": args.kv_dtype,
        "layout": args.layout,
        "page_size": args.page_size,
        "causal": args.causal,
        "pos_encoding_mode": args.pos_encoding_mode,
        "mode": args.mode,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "atol": atol,
        "rtol": atol,
        "samples_ms": times,
        "median_ms": statistics.median(times) if times else None,
    }


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--kv-dtype", choices=DTYPES, default="nvfp4")
    parser.add_argument("--q-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES)[:5])
    parser.add_argument("--q-heads", type=positive_int, default=32)
    parser.add_argument("--kv-heads", type=positive_int, default=8)
    parser.add_argument("--page-size", type=positive_int, default=16)
    parser.add_argument("--layout", choices=["NHD", "HND"], default="NHD")
    parser.add_argument("--mode", choices=["paged", "ragged"], default="paged")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--pos-encoding-mode", choices=["NONE", "ROPE_LLAMA"], default="NONE"
    )
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=30)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(flashinfer.__file__).resolve().parents[1]
    result = {
        "git_commit": source_commit(source),
        "source": str(source),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cache": os.environ.get("FLASHINFER_WORKSPACE_BASE"),
        "timing": "CUDA events around graph replays; 10 calls per graph; 30 warmup replays",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        row = run_case(name, args)
        result["results"].append(row)
        write_results(args.output, result)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
