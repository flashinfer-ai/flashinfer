"""Per-call HOST (CPU) overhead of the GEMM dispatch path.

Host cost only matters when it exceeds the kernel, which is the eager
LLM-serving regime: small M, weights already resident.  This measures enqueue
rate -- N calls with no per-call sync, on a shape whose kernel is far cheaper
than the dispatch -- so wall/N is the host cost.

    python benchmarks/bench_host_overhead.py [--json out.json] [--compare base.json]

Use --compare to print a before/after table when changing the dispatch path.
"""

import argparse
import json
import time

import torch

import flashinfer

DEFAULT_SHAPE = (8, 2048, 2048)  # M, N, K -- kernel ~2 us, dispatch dominates


def bench(fn, reps=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / reps * 1e6)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shape",
        type=int,
        nargs=3,
        default=list(DEFAULT_SHAPE),
        metavar=("M", "N", "K"),
    )
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--compare", type=str, default="")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    m, n, k = args.shape
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=dev, dtype=torch.bfloat16).t().contiguous().t()
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)

    props = torch.cuda.get_device_properties(0)
    print(
        f"{props.name} sm{props.major}{props.minor} | mm_bf16 M={m} N={n} K={k} "
        f"| torch {torch.__version__} | flashinfer {flashinfer.__version__}"
    )

    results = {"torch.mm": bench(lambda: torch.mm(a, b, out=out), args.reps)}
    for backend in ("auto", "cudnn", "cublaslt", "cutlass", "tgv", "tinygemm"):
        try:
            flashinfer.mm_bf16(a, b, out=out, backend=backend)
        except Exception as exc:
            print(f"  skip {backend}: {str(exc)[:60]}")
            continue
        results[f"mm_bf16({backend})"] = bench(
            lambda x=backend: flashinfer.mm_bf16(a, b, out=out, backend=x), args.reps
        )

    base = None
    if args.compare:
        with open(args.compare) as f:
            base = json.load(f)["results"]
    width = max(len(x) for x in results)
    print(
        f"\n{'':<{width}}  {'us/call':>9s}"
        + (f"{'baseline':>10s}{'delta':>10s}" if base else "")
    )
    for name, us in results.items():
        line = f"{name:<{width}}  {us:9.1f}"
        if base and name in base:
            d = us - base[name]
            line += f"{base[name]:10.1f}{d:+10.1f}"
        print(line)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"shape": [m, n, k], "gpu": props.name, "results": results}, f, indent=1
            )
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
