"""Decode and prefill cost of ``chunk_gated_delta_rule(batch_invariant=True)`` (class ``gdn.bf16.s16``) against the
regular GDN paths.

Decode: one token per request over a bf16 state pool, replayed as a CUDA graph, against
``gated_delta_rule_decode_pretranspose`` over a bf16 pool. The class's chunk phase advances with every replay, so its
figure is the mean over whole 16-step chunks; the maximum of the per-phase medians is the fold step.
Prefill: equal-length sequences against today's dispatch with its log gates precomputed. CUPTI GPU time, cold L2.

    python benchmarks/bench_batch_invariant_gdn_class_s.py --heads 16x48 --batches 8 64 --lengths 1024 8192
"""

import argparse
import statistics

import torch

import flashinfer
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.testing import bench_gpu_time

D, CHUNK = 128, 16
BF = torch.bfloat16


def _tokens(n, HQ, HV, gen):
    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    q, k, v = (randn(n, h, D).to(BF) for h in (HQ, HQ, HV))
    return q, k, v, (randn(n, HV) - 6.0).to(BF), randn(n, HV).to(BF)


def _fused(A_log, dt_bias):
    return dict(
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        A_log=A_log,
        dt_bias=dt_bias,
        batch_invariant=True,
    )


def bench_decode(HQ, HV, B, chunks):
    gen = torch.Generator(device="cuda").manual_seed(B)
    q, k, v, a, b = _tokens(B, HQ, HV, gen)
    A_log, dt_bias = (
        0.1 * torch.randn(HV, device="cuda", generator=gen) for _ in range(2)
    )
    pool = (0.05 * torch.randn(B, HV, D, D, device="cuda", generator=gen)).to(BF)
    base_pool = pool.clone()
    slots = torch.arange(B, dtype=torch.int32, device="cuda")
    out = torch.empty(B, HV, D, dtype=BF, device="cuda")

    def step():
        flashinfer.chunk_gated_delta_rule(
            q,
            k,
            v,
            a,
            b,
            initial_state=pool,
            state_indices=slots,
            output=out,
            **_fused(A_log, dt_bias),
        )

    for _ in range(CHUNK):
        step()
    times = bench_gpu_time(
        step,
        dry_run_iters=CHUNK,
        repeat_iters=CHUNK * chunks,
        enable_cupti=True,
        use_cuda_graph=True,
        cold_l2_cache=True,
    )
    phases = [statistics.median(times[p::CHUNK]) for p in range(CHUNK)]
    q4, k4, v4, a4, b4 = (t[:, None].contiguous() for t in (q, k, v, a, b))

    def base():
        gated_delta_rule_decode_pretranspose(
            q4,
            k4,
            v4,
            None,
            A_log,
            a4,
            dt_bias,
            b4,
            use_qk_l2norm=True,
            initial_state=base_pool,
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
    return statistics.mean(phases), max(phases), statistics.median(base_times)


def bench_prefill(HQ, HV, L, nseq):
    gen = torch.Generator(device="cuda").manual_seed(L)
    q, k, v, a, b = _tokens(L * nseq, HQ, HV, gen)
    A_log, dt_bias = (
        0.1 * torch.randn(HV, device="cuda", generator=gen) for _ in range(2)
    )
    g = -torch.exp(A_log) * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b.float())
    s0 = 0.05 * torch.randn(nseq, HV, D, D, device="cuda", generator=gen)
    pool = s0.to(BF)
    cu_cpu = [L * i for i in range(nseq + 1)]
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device="cuda")
    slots = torch.arange(nseq, dtype=torch.int32, device="cuda")

    def ours():
        flashinfer.chunk_gated_delta_rule(
            q,
            k,
            v,
            a,
            b,
            initial_state=pool,
            state_indices=slots,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            **_fused(A_log, dt_bias),
        )

    def auto():
        flashinfer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=s0,
            output_final_state=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
        )

    res = []
    for fn in (ours, auto):
        fn()
        times = bench_gpu_time(
            fn, dry_run_iters=3, repeat_iters=20, enable_cupti=True, cold_l2_cache=True
        )
        res.append(statistics.median(times))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="16x48")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 64, 128, 256])
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 32768])
    ap.add_argument("--sequences", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--chunks", type=int, default=10)
    args = ap.parse_args()
    HQ, HV = (int(x) for x in args.heads.split("x"))
    print(
        f"{torch.cuda.get_device_name(0)}, heads {args.heads}, torch {torch.__version__}, "
        f"class {flashinfer.batch_invariant.fingerprint('chunk_gated_delta_rule')}"
    )
    print(
        "decode, microseconds per step: batch | batch_invariant mean over a chunk | fold step | regular | ratio"
    )
    for B in args.batches:
        mean, fold, base = bench_decode(HQ, HV, B, args.chunks)
        print(
            f"{B:5d} | {mean * 1e3:8.1f} | {fold * 1e3:8.1f} | {base * 1e3:8.1f} | {base / mean:.2f}"
        )
    print("prefill, microseconds: sequences | L | batch_invariant | auto | ratio")
    for nseq in args.sequences:
        for L in args.lengths:
            ours, auto = bench_prefill(HQ, HV, L, nseq)
            print(
                f"{nseq:3d} | {L:6d} | {ours * 1e3:9.1f} | {auto * 1e3:9.1f} | {auto / ours:.2f}"
            )


if __name__ == "__main__":
    main()
