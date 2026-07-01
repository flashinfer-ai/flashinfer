"""End-to-end benchmark: monomoe (mono_moe) vs cutlass_fused_moe (block-FP8).

Both run the same block-wise-FP8 MoE math on Hopper (SM90a) at the Qwen3.5-35B
shape (E=256, N=512, K=2048).  Timing is end-to-end via CUDA graph replay
(`use_cuda_graph=True`), which amortizes launch overhead and reflects the
serving regime where the MoE is replayed from a captured graph.

Accuracy and speed are measured against different baselines, on purpose:

  * Accuracy  — both kernels are compared (cosine similarity) against an
                fp32 reference computed from the UNQUANTIZED weights and the
                same expert selection.  The fp32 reference is the ground
                truth; each kernel's `cos` is its own accuracy, independent of
                the other.  (Comparing the two kernels to each other is
                misleading: cutlass is inaccurate at small token counts, which
                would wrongly drag mono's number down.)
  * Speed     — cutlass_fused_moe is the speedup baseline (`speedup` =
                cutlass_time / mono_time).

The one deliberate asymmetry is routing:

  * mono_moe          — routing (top-K + renormalize) is FUSED inside the
                        single kernel, from `router_logits`, so it is part of
                        the timed region (it cannot be separated out).
  * cutlass_fused_moe — routing is computed ONCE outside the timed region (the
                        softmax top-K selection mono uses) and fed in as
                        pre-computed `topk_ids` / `topk_w`.  The timed region is
                        therefore JUST `cutlass_fused_moe` — the two GEMMs.

Both kernels are fed the SAME expert selection.  Block-wise WEIGHT quant is
one-time prep outside the loop for both paths; activation bf16->fp8 quant is
internal to both kernels (inside the timed region for both).

Run:  python benchmarks/bench_monomoe.py
"""

import argparse

import torch
import torch.nn.functional as F

from flashinfer import fused_moe
from flashinfer.fused_moe import mono_moe, has_monomoe
from flashinfer.testing import bench_gpu_time
from flashinfer.utils import is_sm90a_supported

# Fixed geometry of the compiled monomoe variant.
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


def moe_reference_fp32(x, w13, w2, topk_w, topk_ids):
    """Ground-truth MoE in fp32 from the UNQUANTIZED weights.

    Uses mono_moe's `[gate || up]` fc1 half-ordering and the given (already
    renormalized) top-K selection.  No FP8 quantization anywhere — this is the
    accuracy reference both kernels are scored against.
    """
    m = x.shape[0]
    out = torch.zeros(m, K, device=x.device, dtype=torch.float32)
    xf, w13f, w2f = x.float(), w13.float(), w2.float()
    for t in range(m):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            wgt = float(topk_w[t, j])
            gate = xf[t] @ w13f[e, :N, :].T
            up = xf[t] @ w13f[e, N:, :].T
            h = F.silu(gate) * up
            out[t] += wgt * (h @ w2f[e].T)
    return out


def cosine(a, b):
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def summarize(times_ms):
    t = torch.tensor(times_ms)
    return t.median().item(), t.std().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--top-k", type=int, nargs="+", default=[8])
    ap.add_argument(
        "--cupti", action="store_true", help="use CUPTI timing if available"
    )
    args = ap.parse_args()

    dev = torch.device("cuda")
    if not is_sm90a_supported(dev):
        print("monomoe requires SM90a (Hopper) — skipping.")
        return
    if not has_monomoe():
        print("monomoe extension unavailable (failed to build/load) — skipping.")
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
        f"Both paths = 2 block-FP8 GEMMs + SiLU (activation quant internal).\n"
        f"  mono_moe         : routing fused in-kernel (softmax), timed\n"
        f"  cutlass          : pre-computed softmax routing (NOT timed) + cutlass_fused_moe\n"
        f"Accuracy: cos vs an fp32 reference (ground truth, unquantized weights).\n"
        f"Speed:    speedup = cutlass_time / mono_time.\n"
    )
    print(
        f"{'shape':>12} | {'mono_moe':>22} | {'cutlass':>22} | {'speedup':>8} | "
        f"{'mono cos':>8} | {'cutlass cos':>11}"
    )
    print("-" * 100)

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
                    x,
                    logits,
                    w13_fp8,
                    s13,
                    w2_fp8,
                    s2,
                    top_k=top_k,
                    scoring_func="softmax",
                    renormalize=True,
                    out=out_mono,
                    scratchpad=scratch,
                )

            # ── cutlass_fused_moe: pre-computed routing, deepseek block-FP8 ──
            out_cutlass = torch.zeros(m, K, dtype=torch.bfloat16, device=dev)

            # cutlass_fused_moe uses the opposite SwiGLU half ordering from
            # mono_moe: it reads fc1 weights as [up || gate] whereas monomoe
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

            # Routing is computed ONCE here, OUTSIDE the timed region, using the
            # same softmax top-K selection mono uses (topk_ids / topk_w from
            # above).  These fixed buffers are graph-capture-safe, and the timed
            # region below is JUST the two block-FP8 GEMMs — no routing kernel.
            route_ids = topk_ids.contiguous()
            route_vals = topk_w.float().contiguous()

            def run_cutlass():
                fused_moe.cutlass_fused_moe(
                    x,
                    route_ids,
                    route_vals,
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

            # Accuracy vs the fp32 ground truth (warm-up outputs are valid).
            # Each kernel is scored independently against the reference, NOT
            # against each other, so cutlass's small-m inaccuracy can't drag
            # mono's number down.
            torch.cuda.synchronize()
            ref = moe_reference_fp32(x, w13, w2, route_vals, route_ids)
            mono_cos = cosine(out_mono, ref)
            cutlass_cos = cosine(out_cutlass, ref) if have_cutlass else float("nan")

            mono_med, mono_std = summarize(
                bench_gpu_time(run_mono, use_cuda_graph=True, enable_cupti=args.cupti)
            )
            if have_cutlass:
                cut_med, cut_std = summarize(
                    bench_gpu_time(
                        run_cutlass, use_cuda_graph=True, enable_cupti=args.cupti
                    )
                )
                cut_str = f"{cut_med:7.4f} ± {cut_std:6.4f} ms"
                speedup = f"{cut_med / mono_med:6.2f}x"
                cutlass_cos_str = f"{cutlass_cos:11.3f}"
            else:
                cut_str = f"unavailable ({cutlass_err})"
                speedup = "    n/a"
                cutlass_cos_str = f"{'n/a':>11}"

            shape = f"m={m},k={top_k}"
            mono_str = f"{mono_med:7.4f} ± {mono_std:6.4f} ms"
            print(
                f"{shape:>12} | {mono_str:>22} | {cut_str:>22} | {speedup:>8} | "
                f"{mono_cos:8.3f} | {cutlass_cos_str}"
            )


if __name__ == "__main__":
    main()
