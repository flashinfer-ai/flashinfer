"""SM90 NVFP4 MegaMoE: correctness against the reference implementation.

Runs the FlashInfer kernel and the DeepGEMM reference on identical inputs and
weights and requires bit-identical output -- both drive the same CUDA kernel, so
any difference is a host-side (layout / descriptor / plan) bug.

Requires 8 GPUs. The script spawns its own worker per GPU, so run it with plain
python, not torchrun:  python tests/moe/test_megamoe_nvfp4_sm90.py
"""

import argparse
import sys

import torch
import torch.distributed as dist

SHAPES = {
    "flash": dict(hidden=4096, intermediate_hidden=2048, num_experts=256, num_topk=6),
    "pro": dict(hidden=7168, intermediate_hidden=3072, num_experts=384, num_topk=6),
    "mimo": dict(hidden=6144, intermediate_hidden=2048, num_experts=384, num_topk=8),
}


def make_routing(m, num_experts, num_topk, rank, seed):
    torch.manual_seed(rank + 101 + seed * 1000)
    scores = torch.randn((m, num_experts), dtype=torch.float, device="cuda")
    w, idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    return idx, w.float()


def worker(local_rank, world_size, args):
    import deep_gemm  # reference
    from deep_gemm.quantization_nvfp4 import quantize_to_nvfp4
    from deep_gemm.utils import per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist

    from flashinfer.megamoe import (
        FAMILY_THRESHOLD,
        MegaMoESymmBuffer,
        nvfp4_mega_moe,
    )

    rank, num_ranks, group = init_dist(local_rank, world_size)
    failures = 0

    for shape_name in args.shapes:
        s = SHAPES[shape_name]
        hidden, ih = s["hidden"], s["intermediate_hidden"]
        num_experts, num_topk = s["num_experts"], s["num_topk"]
        experts_per_rank = num_experts // num_ranks

        g = torch.Generator(device="cuda")
        g.manual_seed(7919 + rank)
        l1_bf = (
            torch.randn(
                (experts_per_rank, ih * 2, hidden),
                dtype=torch.bfloat16,
                device="cuda",
                generator=g,
            )
            * 0.05
        )
        l2_bf = (
            torch.randn(
                (experts_per_rank, hidden, ih),
                dtype=torch.bfloat16,
                device="cuda",
                generator=g,
            )
            * 0.05
        )
        l1_packed, l1_scale = quantize_to_nvfp4(l1_bf, group_size=16)
        l2_packed, l2_scale = quantize_to_nvfp4(l2_bf, group_size=16)
        del l1_bf, l2_bf
        torch.cuda.empty_cache()

        # The prepacked scales are tile-major (E, N/block_n, K/128, block_n, 8),
        # so the weight layout is part of the family: block_n=256 for the fused
        # kernel, block_n=128 for split L1/L2. Build each on first use.
        transformed = {}

        ref_buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group, num_experts, args.cap, num_topk, hidden, ih
        )
        fi_buf = MegaMoESymmBuffer(
            group, num_experts, args.cap, num_topk, hidden, ih, fast_math=args.fast_math
        )

        for m in args.m:
            x_bf = torch.randn(
                (m, hidden), dtype=torch.bfloat16, device="cuda", generator=g
            )
            x_fp8, x_sf = per_token_cast_to_fp8(
                x_bf, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
            )
            topk_idx, topk_w = make_routing(m, num_experts, num_topk, rank, 0)

            # Pin the family on both sides rather than letting each one's
            # own "auto" rule decide -- a silent disagreement would compare
            # two different kernels and still look like a pass.
            family = "split" if m > FAMILY_THRESHOLD else "fused"
            block_n = 128 if family == "split" else 256
            if block_n not in transformed:
                transformed[block_n] = (
                    deep_gemm.transform_nvfp4_weights_for_mega_moe_sm90(
                        (l1_packed, l1_scale), (l2_packed, l2_scale), block_n=block_n
                    )
                )
            l1w, l2w = transformed[block_n]

            y_ref = torch.zeros((m, hidden), dtype=torch.bfloat16, device="cuda")
            ref_buf.x[:m].copy_(x_fp8)
            ref_buf.x_sf[:m].copy_(x_sf)
            ref_buf.topk_idx[:m].copy_(topk_idx)
            ref_buf.topk_weights[:m].copy_(topk_w)
            deep_gemm.nvfp4_mega_moe(
                y_ref,
                l1w,
                l2w,
                ref_buf,
                recipe=(128, 128, 128),
                activation="swiglu",
                activation_clamp=10.0,
                fast_math=args.fast_math,
                kernel_family=family,
                family_threshold=FAMILY_THRESHOLD,
            )
            torch.cuda.synchronize()
            dist.barrier(group=group)

            y_fi = torch.zeros((m, hidden), dtype=torch.bfloat16, device="cuda")
            fi_buf.stage(x_fp8, x_sf, topk_idx, topk_w)
            nvfp4_mega_moe(
                y_fi,
                l1w[0],
                l2w[0],
                fi_buf,
                kernel_family=family,
                family_threshold=FAMILY_THRESHOLD,
            )
            torch.cuda.synchronize()
            dist.barrier(group=group)

            # Reduce every reported field across ranks, so the line rank 0
            # prints describes all of them rather than its own slice.
            d = (y_fi.float() - y_ref.float()).abs()
            flags = torch.tensor(
                [
                    int(torch.equal(y_fi.view(torch.int16), y_ref.view(torch.int16))),
                    int(torch.isfinite(y_fi.float()).all().item()),
                ],
                device="cuda",
                dtype=torch.int32,
            )
            max_abs = d.max().reshape(1)
            num_diff = (d > 0).sum().reshape(1)
            dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(num_diff, op=dist.ReduceOp.SUM, group=group)
            same, finite = bool(flags[0].item()), bool(flags[1].item())
            ok = same and finite
            if rank == 0:
                tag = "PASS" if ok else "FAIL"
                extra = (
                    ""
                    if same
                    else f" max_abs={max_abs.item():.3e} n={int(num_diff.item())}"
                )
                print(
                    f"{tag} {shape_name} M={m} {family} fm={args.fast_math} "
                    f"bit-identical={same} finite={finite}{extra}",
                    flush=True,
                )
            failures += 0 if ok else 1

        ref_buf.destroy()
        fi_buf.destroy()
        del transformed, l1_packed, l2_packed, l1_scale, l2_scale
        torch.cuda.empty_cache()

    dist.barrier(group=group)
    if rank == 0:
        print(f"DONE failures={failures}", flush=True)
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="+", default=["flash"], choices=sorted(SHAPES))
    p.add_argument("--m", nargs="+", type=int, default=[1, 2, 8, 16, 32, 64])
    p.add_argument("--cap", type=int, default=8448)
    p.add_argument("--num-processes", type=int, default=8)
    # kFastMath is a template parameter of the kernel, so it must match on both
    # sides. The False path additionally needs the JIT spec to override the
    # -use_fast_math that gen_jit_spec() applies to every module.
    p.add_argument(
        "--no-fast-math", dest="fast_math", action="store_false", default=True
    )
    args = p.parse_args()
    torch.multiprocessing.spawn(
        worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
