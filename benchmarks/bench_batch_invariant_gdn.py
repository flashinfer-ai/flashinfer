"""Decode and prefill cost of ``chunk_gated_delta_rule(batch_invariant=True)`` against the regular GDN paths.

Decode: one token per request over an fp32 state pool, replayed as a CUDA graph, against
``gated_delta_rule_decode_pretranspose`` over a bf16 pool. The class's chunk phase advances with every replay, so its
figure is the mean over whole 64-step chunks (the chunk-completing step included); the median is the mid-chunk step.
Prefill: sequences of whole chunks against today's dispatch. CUPTI GPU time with a cold L2.

    python benchmarks/bench_batch_invariant_gdn.py --heads 16x48 --batches 8 16 64 128 256
"""

import argparse
import statistics

import torch

import flashinfer
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.testing import bench_gpu_time

D, CHUNK = 128, 64


def _tokens(n, HQ, HV, gen):
    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    q, k, v = (randn(n, h, D).to(torch.bfloat16) for h in (HQ, HQ, HV))
    g = -torch.nn.functional.softplus(randn(n, HV) - 6.0)
    return q, k, v, g, torch.sigmoid(randn(n, HV))


def bench_decode(HQ, HV, B, chunks):
    gen = torch.Generator(device="cuda").manual_seed(B)
    q, k, v, g, beta = _tokens(B, HQ, HV, gen)
    pool = 0.05 * torch.randn(B, HV, D, D, device="cuda", generator=gen)
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    cu = torch.arange(B + 1, dtype=torch.int32, device="cuda")
    out = torch.empty(B, HV, D, dtype=torch.bfloat16, device="cuda")

    def step(prefix_lens=None):
        flashinfer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            state_indices=slots,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
            output=out,
            prefix_lens=prefix_lens,
            batch_invariant=True,
        )

    step(torch.zeros(B, dtype=torch.int32, device="cuda"))
    for _ in range(CHUNK - 1):
        step()
    times = bench_gpu_time(
        step,
        dry_run_iters=CHUNK,
        repeat_iters=CHUNK * chunks,
        enable_cupti=True,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    bf16_pool = pool.to(torch.bfloat16)
    A_log, dt_bias = (
        0.1 * torch.randn(HV, device="cuda", generator=gen) for _ in range(2)
    )
    a = (0.1 * torch.randn(B, 1, HV, device="cuda", generator=gen)).to(torch.bfloat16)
    b = torch.randn(B, 1, HV, device="cuda", generator=gen).to(torch.bfloat16)
    q4, k4, v4 = (
        q[:, None].contiguous(),
        k[:, None].contiguous(),
        v[:, None].contiguous(),
    )

    def base():
        gated_delta_rule_decode_pretranspose(
            q4,
            k4,
            v4,
            None,
            A_log,
            a,
            dt_bias,
            b,
            use_qk_l2norm=True,
            initial_state=bf16_pool,
            initial_state_indices=slots,
            backend="auto",
        )

    base()
    base_times = bench_gpu_time(
        base,
        dry_run_iters=5,
        repeat_iters=30,
        enable_cupti=True,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    return (
        statistics.mean(times),
        statistics.median(times),
        statistics.median(base_times),
    )


def bench_prefill(HQ, HV, L, nseq):
    gen = torch.Generator(device="cuda").manual_seed(L)
    q, k, v, g, beta = _tokens(L * nseq, HQ, HV, gen)
    s0 = 0.05 * torch.randn(nseq, HV, D, D, device="cuda", generator=gen)
    pool = s0.clone()
    cu_cpu = [L * i for i in range(nseq + 1)]
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device="cuda")
    slots = torch.arange(nseq, dtype=torch.int32, device="cuda")
    common = dict(
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        use_qk_l2norm_in_kernel=True,
        gate_domain="log",
    )

    def ours():
        flashinfer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            state_indices=slots,
            batch_invariant=True,
            **common,
        )

    def auto():
        flashinfer.chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state=s0, output_final_state=True, **common
        )

    res = []
    for fn in (ours, auto):
        fn()
        res.append(
            statistics.median(
                bench_gpu_time(
                    fn,
                    dry_run_iters=3,
                    repeat_iters=20,
                    enable_cupti=True,
                    cold_l2_cache=True,
                )
            )
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="16x48")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 64, 128, 256])
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 32768])
    ap.add_argument("--chunks", type=int, default=5)
    args = ap.parse_args()
    HQ, HV = (int(x) for x in args.heads.split("x"))
    print(
        f"{torch.cuda.get_device_name(0)}, heads {args.heads}, torch {torch.__version__}"
    )
    print(
        "decode, microseconds per step: batch | batch_invariant mean over a chunk | mid-chunk | regular | ratio"
    )
    for B in args.batches:
        mean, mid, base = bench_decode(HQ, HV, B, args.chunks)
        print(
            f"{B:5d} | {mean * 1e3:8.1f} | {mid * 1e3:8.1f} | {base * 1e3:8.1f} | {base / mean:.2f}"
        )
    print("prefill, microseconds: sequences | L | batch_invariant | auto | ratio")
    for nseq in (1, 4):
        for L in args.lengths:
            ours, auto = bench_prefill(HQ, HV, L, nseq)
            print(
                f"{nseq:3d} | {L:6d} | {ours * 1e3:9.1f} | {auto * 1e3:9.1f} | {auto / ours:.2f}"
            )


if __name__ == "__main__":
    main()
