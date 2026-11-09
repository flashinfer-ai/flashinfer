"""Decode step cost of the exact (prefill-bitwise) KDA decode next to today's recurrent KDA decode.

Per (heads, batch) point, CUPTI GPU time with a cold L2 (median of ``--iters``) of: the pooled exact step replayed
as a CUDA graph (``KdaExactDecodePool.capture_step``), the pinned engine call alone (the kernel floor of the step),
and ``recurrent_kda`` at ``T = 1`` on a bf16 state pool addressed by ``ssm_state_indices`` (the call vLLM makes),
captured the same way. The pool's chunk phase advances with every replay, so its figure averages the sixteen phases,
the steady-state serving condition. ``ratio`` is recurrent / exact (above 1 means the exact step is faster)."""

import argparse
import statistics

import torch

import flashinfer
from flashinfer.kda import recurrent_kda
from flashinfer.kda_exact_decode import (
    KdaExactDecodePool,
    KdaLayerParams,
    kda_exact_prefill,
)
from flashinfer.testing import bench_gpu_time

D = 128
LB = -5.0


def _median_us(fn, iters, use_cuda_graph):
    return (
        statistics.median(
            bench_gpu_time(
                fn,
                enable_cupti=True,
                dry_run_iters=3,
                repeat_iters=iters,
                cold_l2_cache=True,
                use_cuda_graph=use_cuda_graph,
            )
        )
        * 1e3
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64, 256])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda")
    print(
        f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}, "
        f"flashinfer {flashinfer.__version__}",
        flush=True,
    )
    print(
        "| heads | B | exact pool step, graph [us] | engine alone [us] | recurrent_kda T=1 bf16 pool, graph [us] | ratio |"
    )
    print("|---|---|---|---|---|---|")
    for h in args.heads:
        gen = torch.Generator(device=dev).manual_seed(1)
        params = KdaLayerParams(
            A_log=(0.1 * torch.randn(h, device=dev, generator=gen)).float(),
            dt_bias=(0.1 * torch.randn(h, D, device=dev, generator=gen)).float(),
            lower_bound=LB,
        )
        for B in args.batch:
            pool = KdaExactDecodePool(B, h, h, params, dev)
            for b in range(B):
                Lp = 256 + (7 * b) % 16
                q = torch.randn(Lp, h, D, device=dev, generator=gen).to(torch.bfloat16)
                k = torch.randn(Lp, h, D, device=dev, generator=gen).to(torch.bfloat16)
                v = torch.randn(Lp, h, D, device=dev, generator=gen).to(torch.bfloat16)
                g = (torch.randn(Lp, h, D, device=dev, generator=gen) - 6.0).to(
                    torch.bfloat16
                )
                beta = torch.randn(Lp, h, device=dev, generator=gen).to(torch.bfloat16)
                s0 = 0.05 * torch.randn(h, D, D, device=dev, generator=gen)
                _, st = kda_exact_prefill(q, k, v, g, beta, s0, params)
                pool.load(b, st)
            q_t = torch.randn(B, h, D, device=dev, generator=gen).to(torch.bfloat16)
            k_t = torch.randn(B, h, D, device=dev, generator=gen).to(torch.bfloat16)
            v_t = torch.randn(B, h, D, device=dev, generator=gen).to(torch.bfloat16)
            g_t = (torch.randn(B, h, D, device=dev, generator=gen) - 6.0).to(
                torch.bfloat16
            )
            b_t = torch.randn(B, h, device=dev, generator=gen).to(torch.bfloat16)
            ids = torch.arange(B, dtype=torch.int32, device=dev)

            graph = pool.capture_step(ids, q_t, k_t, v_t, g_t, b_t)
            t_step = _median_us(graph.replay, args.iters, use_cuda_graph=False)
            t_eng = _median_us(pool.kernel_only(ids), args.iters, use_cuda_graph=True)

            state_pool = pool.state[:B].to(torch.bfloat16)

            def recurrent():
                recurrent_kda(
                    q_t[:, None],
                    k_t[:, None],
                    v_t[:, None],
                    g_t[:, None],
                    b_t[:, None],
                    A_log=params.A_log,
                    dt_bias=params.dt_bias,
                    use_gate_in_kernel=True,
                    beta_is_logit=True,
                    lower_bound=LB,
                    initial_state=state_pool,
                    ssm_state_indices=ids,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )

            t_rec = _median_us(recurrent, args.iters, use_cuda_graph=True)
            print(
                f"| {h} | {B} | {t_step:.1f} | {t_eng:.1f} | {t_rec:.1f} | {t_rec / t_step:.2f} |",
                flush=True,
            )
            del pool, graph
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
