"""End-to-end benchmark: megamoe (mono_moe) vs cutlass_fused_moe (block-FP8).

Both run the same block-wise-FP8 MoE math on Hopper (SM90a) at the Qwen3.5-35B
shape (E=256, N=512, K=2048).  Timing is end-to-end via CUDA graph replay
(`use_cuda_graph=True`), which amortizes launch overhead and reflects the
serving regime where the MoE is replayed from a captured graph.

For a fair comparison both paths include the SAME phases — routing + the two
block-FP8 GEMMs (with internal bf16->fp8 activation quant) + SiLU:

  * mono_moe          — routing (top-K + renormalize) is FUSED inside the
                        single kernel, from `router_logits`.
  * cutlass_fused_moe — takes pre-computed routing, so the timed region runs a
                        FUSED routing kernel (`fused_topk_deepseek`, ~3us) then
                        `cutlass_fused_moe`.  We deliberately do NOT use a torch
                        softmax+topk for routing here: that multi-kernel
                        sequence costs ~21us in-graph (more than the whole mono
                        kernel) and would unfairly penalize cutlass — it
                        measures torch, not the MoE.

Note the routing kernels differ in scoring (mono = softmax, fused_topk_deepseek
= sigmoid+grouped), so they pick different experts.  That is fine for TIMING
(both are one fast fused routing kernel).  The CORRECTNESS cross-check is done
separately: cutlass is fed the SAME softmax top-K selection mono uses (computed
once, outside the timed region) and compared by cosine similarity.

Run:  python benchmarks/bench_megamoe.py
"""

import argparse

import torch
import torch.nn.functional as F

from flashinfer import fused_moe
from flashinfer.fused_moe import mono_moe, has_megamoe, fused_topk_deepseek
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm90a_supported

# Fixed geometry of the compiled megamoe variant.
E = 256
N = 512  # N_half; up-projection emits 2*N rows (gate || up)
K = 2048
BLOCK = 128


def ceil_div(a, b):
    return (a + b - 1) // b


def quant_fp8_block_wise(w, block_row=128, block_col=128):
    """Block-wise (128x128) FP8 quant. w:[E,rows,cols] -> (fp8, scales[E,rb,cb])."""
    Ee, rows, cols = w.shape
    rb, cb = ceil_div(rows, block_row), ceil_div(cols, block_col)
    wf = w.float()
    scales = torch.zeros(Ee, rb, cb, device=w.device, dtype=torch.float32)
    w_fp8 = torch.zeros_like(wf)
    for ri in range(rb):
        r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
        for ci in range(cb):
            c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
            blk = wf[:, r0:r1, c0:c1]
            amax = blk.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
            scales[:, ri, ci] = (amax / 448.0).reshape(Ee)
            w_fp8[:, r0:r1, c0:c1] = (blk / (amax / 448.0)).clamp(-448, 448)
    return w_fp8.to(torch.float8_e4m3fn), scales


def routing_softmax_topk(logits, top_k):
    scores = torch.softmax(logits.float(), dim=-1)
    wts, ids = torch.topk(scores, top_k, dim=-1)
    wts = wts / wts.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return wts.float(), ids.to(torch.int32)


def summarize(times_ms):
    t = torch.tensor(times_ms)
    return t.median().item(), t.std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--top-k", type=int, nargs="+", default=[8])
    ap.add_argument("--cupti", action="store_true", help="use CUPTI timing if available")
    args = ap.parse_args()

    dev = torch.device("cuda")
    if not is_sm90a_supported(dev):
        print("megamoe requires SM90a (Hopper) — skipping.")
        return
    if not has_megamoe():
        print("megamoe extension unavailable (failed to build/load) — skipping.")
        return

    torch.manual_seed(0)
    # Shared quantized weights (same tensors fed to both kernels).
    w13 = torch.randn(E, 2 * N, K, device=dev) * 0.1
    w2 = torch.randn(E, K, N, device=dev) * 0.1
    w13_fp8, s13 = quant_fp8_block_wise(w13)
    w2_fp8, s2 = quant_fp8_block_wise(w2)

    timing = "CUPTI" if args.cupti else "CUDA-graph"
    print(
        f"End-to-end MoE, block-FP8, SM90 — {timing} timing.\n"
        f"Both paths = fused routing + 2 block-FP8 GEMMs + SiLU.\n"
        f"  mono_moe         : routing fused in-kernel (softmax)\n"
        f"  cutlass+routing  : fused_topk_deepseek (sigmoid) + cutlass_fused_moe\n"
        f"cos = correctness check with MATCHED softmax routing fed to cutlass.\n"
    )
    print(f"{'shape':>16} | {'mono_moe':>22} | {'cutlass + routing':>22} | {'speedup':>8} | {'cos':>6}")
    print("-" * 92)

    for m in args.tokens:
        for top_k in args.top_k:
            x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
            logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)
            topk_w, topk_ids = routing_softmax_topk(logits, top_k)

            # ── mono_moe: routing happens inside the kernel ──
            scratch = fused_moe.alloc_scratchpad(dev)
            out_mono = torch.empty(m, K, dtype=torch.bfloat16, device=dev)

            def run_mono():
                mono_moe(
                    x, logits, w13_fp8, s13, w2_fp8, s2,
                    top_k=top_k, scoring_func="softmax", renormalize=True,
                    out=out_mono, scratchpad=scratch,
                )

            # ── cutlass_fused_moe: pre-computed routing, deepseek block-FP8 ──
            out_cutlass = torch.zeros(m, K, dtype=torch.bfloat16, device=dev)

            # cutlass_fused_moe uses the opposite SwiGLU half ordering from
            # mono_moe: it reads fc1 weights as [up || gate] whereas megamoe
            # uses [gate || up].  Swap the two N-row halves (and their scale
            # rows) so both kernels compute the SAME math from the SAME
            # logical weights — verified to bring both to cos≈0.999 vs the
            # fp32 reference.  This is one-time weight prep, not per-call work.
            w13_swapped = torch.cat([w13_fp8[:, N:, :], w13_fp8[:, :N, :]], dim=1)
            s13_rows = s13.shape[1] // 2
            s13_swapped = torch.cat([s13[:, s13_rows:, :], s13[:, :s13_rows, :]], dim=1)
            w13_c = w13_swapped.contiguous()
            w2_c = w2_fp8.contiguous()
            s13_c = s13_swapped.contiguous()
            s2_c = s2.contiguous()

            # Pre-allocated routing outputs (graph-capture-safe: same buffers
            # every call).  Filled by fused_topk_deepseek inside the timed
            # region so cutlass pays a routing cost comparable to mono's
            # in-kernel routing — without the ~21us torch-softmax tax.
            route_bias = torch.zeros(E, device=dev, dtype=torch.bfloat16)
            route_vals = torch.empty(m, top_k, device=dev, dtype=torch.bfloat16)
            route_ids = torch.empty(m, top_k, device=dev, dtype=torch.int32)
            n_group, topk_group = 8, 4  # 256 experts / 8 groups, DeepSeek-V3 style

            def run_cutlass():
                fused_topk_deepseek(
                    logits, route_bias, n_group, topk_group, top_k,
                    1.0, route_vals, route_ids,
                )
                fused_moe.cutlass_fused_moe(
                    x,
                    route_ids,
                    route_vals.float(),
                    w13_c,
                    w2_c,
                    torch.bfloat16,
                    use_deepseek_fp8_block_scale=True,
                    quant_scales=[s13_c, s2_c],
                    output=out_cutlass,
                )

            # Warm up.
            run_mono()
            try:
                run_cutlass()
                have_cutlass = True
            except (NotImplementedError, RuntimeError) as e:
                have_cutlass = False
                cutlass_err = str(e).splitlines()[0][:60]

            # Correctness cross-check: feed cutlass the SAME softmax top-K
            # selection mono uses (outside any timed region) and compare.
            cos = float("nan")
            if have_cutlass:
                out_cut_ref = torch.zeros(m, K, dtype=torch.bfloat16, device=dev)
                fused_moe.cutlass_fused_moe(
                    x, topk_ids, topk_w, w13_c, w2_c, torch.bfloat16,
                    use_deepseek_fp8_block_scale=True,
                    quant_scales=[s13_c, s2_c], output=out_cut_ref,
                )
                torch.cuda.synchronize()
                cos = F.cosine_similarity(
                    out_mono.float().reshape(-1), out_cut_ref.float().reshape(-1), dim=0
                ).item()

            mono_med, mono_std = summarize(
                bench_gpu_time(run_mono, use_cuda_graph=True, enable_cupti=args.cupti)
            )
            if have_cutlass:
                cut_med, cut_std = summarize(
                    bench_gpu_time(run_cutlass, use_cuda_graph=True, enable_cupti=args.cupti)
                )
                cut_str = f"{cut_med:7.4f} ± {cut_std:6.4f} ms"
                speedup = f"{cut_med / mono_med:6.2f}x"
            else:
                cut_str = f"unavailable ({cutlass_err})"
                speedup = "    n/a"

            shape = f"m={m},k={top_k}"
            mono_str = f"{mono_med:7.4f} ± {mono_std:6.4f} ms"
            print(f"{shape:>16} | {mono_str:>22} | {cut_str:>22} | {speedup:>8} | {cos:6.3f}")


if __name__ == "__main__":
    main()
