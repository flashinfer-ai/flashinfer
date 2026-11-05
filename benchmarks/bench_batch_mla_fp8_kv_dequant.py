"""Benchmark the SM90 batch MLA FP8 KV dequant path.

This measures the per-tensor FP8 KV path used by the GLM-5.3-Flash NoPE
sparse MLA shape. On the PR head, the per-tensor path uses the folded scale
and cooperative dequant. On the PR's base commit, the same script measures
the original per-element dequant.

Example:
    python benchmarks/bench_batch_mla_fp8_kv_dequant.py \
        --rows 8192 --topk 2048 --head-dim 512 --heads 8 --kv-dtype fp8
"""

import argparse
import json
import statistics

import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=512)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--pages", type=int, default=1 << 20)
    ap.add_argument("--kv-dtype", choices=["bf16", "fp8"], default="fp8")
    ap.add_argument("--backend", default="fa3")
    ap.add_argument(
        "--scale-mode",
        choices=["per-tensor", "array"],
        default="per-tensor",
        help="per-tensor is the production path; array is the lab scale-array path",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--qscale", type=float, default=0.1)
    ap.add_argument("--smscale", type=float, default=0.0625)
    ap.add_argument("--kvscale", type=float, default=0.5)
    ap.add_argument("--cluster", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    kv_dtype = torch.bfloat16 if args.kv_dtype == "bf16" else torch.float8_e4m3fn

    kv = torch.randn(args.pages, 1, args.head_dim, dtype=torch.bfloat16, device=dev)
    if args.kv_dtype == "fp8":
        kv = (kv.float() * args.kvscale).to(torch.float8_e4m3fn)
    q_nope = (
        torch.randn(
            args.rows, args.heads, args.head_dim, dtype=torch.bfloat16, device=dev
        )
        * args.qscale
    )
    q_pe = torch.empty(args.rows, args.heads, 0, dtype=torch.bfloat16, device=dev)
    kpe = torch.empty(args.pages, 1, 0, dtype=kv_dtype, device=dev)

    gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    idx = torch.randint(
        0,
        args.pages,
        (args.rows, args.topk),
        device=dev,
        generator=gen,
        dtype=torch.int32,
    )
    if args.cluster > 0:
        n = int(args.topk * args.cluster)
        idx[:, :n] = torch.randint(
            0,
            min(65536, args.pages),
            (args.rows, n),
            device=dev,
            generator=gen,
            dtype=torch.int32,
        )
    kv_indices = idx.reshape(-1).contiguous()
    kv_len_arr = torch.full((args.rows,), args.topk, dtype=torch.int32, device=dev)
    kv_indptr = torch.zeros(args.rows + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = (
        torch.arange(1, args.rows + 1, dtype=torch.int32, device=dev) * args.topk
    )
    qo_indptr = torch.arange(args.rows + 1, dtype=torch.int32, device=dev)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend=args.backend)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=args.heads,
        head_dim_ckv=args.head_dim,
        head_dim_kpe=0,
        page_size=1,
        causal=False,
        sm_scale=args.smscale,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
    )

    if args.kv_dtype == "bf16":
        scale = {}
    elif args.scale_mode == "array":
        scale = {
            "ckv_scale_arr": torch.full(
                (args.pages, 1, 4), 1.0, dtype=torch.float32, device=dev
            ),
            "kpe_scale": 1.0,
        }
    else:
        scale = {"ckv_scale": 1.0, "kpe_scale": 1.0}

    def run():
        return wrapper.run(q_nope, q_pe, kv, kpe, **scale)

    for _ in range(args.warmup):
        out = run()
    torch.cuda.synchronize()

    run_medians = []
    for _ in range(args.runs):
        timings = []
        for _ in range(args.iters):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            out = run()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        timings.sort()
        run_medians.append(statistics.median(timings))
    run_medians.sort()

    result = {
        "rows": args.rows,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "topk": args.topk,
        "pages": args.pages,
        "kv_dtype": args.kv_dtype,
        "backend": args.backend,
        "scale_mode": args.scale_mode,
        "ms_median": round(statistics.median(run_medians), 3),
        "ms_min": round(run_medians[0], 3),
        "ms_max": round(run_medians[-1], 3),
        "run_medians": [round(x, 3) for x in run_medians],
        "out_shape": list(out.shape),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
