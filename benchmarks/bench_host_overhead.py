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
        # Stop the clock before draining: GPU completion is not host cost.
        best = min(best, (time.perf_counter() - t0) / reps * 1e6)
        torch.cuda.synchronize()
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

    def add(name, fn):
        # A backend can pass its first call and still fail later (warmup, re-plan);
        # skip it rather than abort the whole table.
        try:
            fn()
            results[name] = bench(fn, args.reps)
        except Exception as exc:
            print(f"  skip {name}: {str(exc)[:60]}")

    for backend in ("auto", "cudnn", "cublaslt", "cutlass", "tgv", "tinygemm"):
        add(
            f"mm_bf16({backend})",
            lambda x=backend: flashinfer.mm_bf16(a, b, out=out, backend=x),
        )

    if props.major in (10, 12):
        w = b.t().contiguous()
        ga = (448 * 6) / a.float().abs().max()
        gw = (448 * 6) / w.float().abs().max()
        layout = flashinfer.SfLayout.layout_128x4
        a4, a_sf = flashinfer.nvfp4_quantize(a, ga, sfLayout=layout, do_shuffle=False)
        w4, w_sf = flashinfer.nvfp4_quantize(w, gw, sfLayout=layout, do_shuffle=False)
        alpha = (1.0 / (ga * gw)).reshape(1).float()
        for backend in ("auto", "cudnn", "cutlass", "cute-dsl"):
            add(
                f"mm_fp4({backend})",
                lambda x=backend: flashinfer.mm_fp4(
                    a4,
                    w4.T,
                    a_sf,
                    w_sf.T,
                    alpha,
                    torch.bfloat16,
                    out=out,
                    block_size=16,
                    use_8x4_sf_layout=False,
                    backend=x,
                ),
            )

    base = None
    if args.compare:
        with open(args.compare) as f:
            saved = json.load(f)
        if saved.get("shape") != [m, n, k] or saved.get("gpu") != props.name:
            raise SystemExit(
                f"--compare baseline was measured on {saved.get('gpu')} shape "
                f"{saved.get('shape')}; this run is {props.name} shape {[m, n, k]}"
            )
        base = saved["results"]
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
