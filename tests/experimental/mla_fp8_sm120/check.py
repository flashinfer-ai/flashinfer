"""Numerical checks of the native FP8 branch; wait for GPU idle before CUDA work."""

import argparse
import json
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parents[3] / "benchmarks/mla_fp8_sm120"
sys.path.insert(0, str(STUDY))

import torch
from bench_fp8_kv import wait_idle
from kv_bench_kernels import quantize_rows
from native_fp8.wrapper import NativeMLA, build
from run_mla import reference

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm", type=int, default=32)
    parser.add_argument("--bn", type=int, default=32)
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/mla_fp8_sm120_correctness.json")
    )
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    config = {k: getattr(args, k) for k in ("bm", "bn", "stages", "groups")}
    print("BUILD", str(build(**config)), flush=True)
    wait_idle(700 * 1024**2)
    torch.manual_seed(490)
    torch.set_num_threads(4)
    workspace = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    scenarios = [
        ("decode", 20, [1], [129], 16, False),
        ("decode_ragged", 20, [1, 1, 1], [1, 63, 2053], 16, False),
        ("extend_causal", 20, [3, 5], [129, 257], 16, True),
        ("prefill_causal", 20, [17, 65], [17, 65], 16, True),
        ("prefill_noncausal", 20, [7, 13], [49, 129], 16, False),
        ("tp2_heads", 10, [1, 3], [1025, 129], 1, True),
        ("tp4_heads", 5, [9], [79], 32, True),
        ("empty_kv", 20, [1, 1], [0, 33], 16, False),
    ]
    if args.quick:
        scenarios = scenarios[:1]
    if args.only:
        scenarios = [s for s in scenarios if s[0] in args.only]
    results = []
    for name, heads, qlens, lengths, page, causal in scenarios:
        wait_idle()
        qind = torch.tensor(
            [0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32
        )
        npages = [(n + page - 1) // page for n in lengths]
        kind = torch.tensor(
            [0] + list(torch.tensor(npages).cumsum(0)), dtype=torch.int32
        )
        indices = torch.randperm(sum(npages), dtype=torch.int32)
        q = torch.randn(sum(qlens), heads, 576, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(sum(npages), page, 576, device="cuda", dtype=torch.bfloat16)
        # Unequal per-token scales exercise both their QK and PV application.
        kv.mul_(torch.linspace(0.5, 2, kv.shape[0], device="cuda")[:, None, None])
        q8, qs = quantize_rows(q)
        kv8, ks = quantize_rows(kv)
        deq_q = q8.cpu().double() * qs.cpu().double()[..., None]
        deq_kv = kv8.cpu().double() * ks.cpu().double()[..., None]
        expected, expected_lse = reference(
            deq_q[..., :512],
            deq_q[..., 512:],
            deq_kv,
            qind,
            kind,
            indices,
            lengths,
            causal,
        )
        for fused in [True] if args.quick else [False, True]:
            run = NativeMLA(
                workspace,
                qind,
                kind,
                indices,
                torch.tensor(lengths, dtype=torch.int32),
                heads=heads,
                page_size=page,
                causal=causal,
                fused=fused,
                **config,
            )
            out = run.run_prequantized(q8, kv8, qs, ks).cpu().double()
            lse = run.lse.cpu().double()
            error = (out - expected).norm() / expected.norm().clamp_min(1e-30)
            finite = torch.isfinite(expected_lse)
            lse_error = (lse[finite] - expected_lse[finite]).abs().max().item()
            row = dict(
                name=name,
                fused=fused,
                config=run.config,
                attributes=run.attributes,
                binary=str(build(**config)),
                relative_l2_vs_dequant_fp64=error.item(),
                max_abs=(out - expected).abs().max().item(),
                lse_max_abs=lse_error,
                finite=bool(torch.isfinite(out).all()),
            )
            print(json.dumps(row), flush=True)
            assert row["finite"] and error < 0.04, row
            assert lse_error < 0.001, row
            assert bool(torch.isneginf(lse[~finite]).all())
            # Full BF16-Q API uses the identical quantization and supports graph replay.
            run.run(q, kv8, ks)
            torch.testing.assert_close(run.out.cpu().double(), out, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run.run(q, kv8, ks)
            graph.replay()
            torch.testing.assert_close(run.out.cpu().double(), out, atol=0, rtol=0)
            results.append(row)
            del graph, run
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        "PASS",
        len(results),
        "cases including fused/separate merge and graph replay",
        flush=True,
    )


if __name__ == "__main__":
    main()
