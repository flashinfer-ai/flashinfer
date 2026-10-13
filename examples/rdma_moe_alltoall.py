# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""NVFP4 or planned BF16 identity experts over SM120 RDMA.

Run on an eight-GPU SM120 host:
  NCCL_GIN_TYPE=3 torchrun --standalone --nproc-per-node=8 \
      examples/rdma_moe_alltoall.py --dtype nvfp4

Four or eight ranks are supported. For planned BF16, add:
  --dtype bf16 --combine-reduction bf16_local_partial_planned

Use --dtype bf16 for a lossless communication reference. The NVFP4 example
checks against an independent Torch quantization reference and reports the
separate quantization error relative to the original BF16 tokens.

The example's setup group uses Gloo. GIN additionally needs matching NCCL
2.30.7 headers/runtime and a working GDAKI provider; see the feature README.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from flashinfer.comm import RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig

GLOBAL_SCALE = 256.0  # Shared by all ranks; dispatch does not transmit it.


def nvfp4_roundtrip_reference(x):
    """Independent E4M3 block scaling and nearest-even E2M1 quantization."""
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    scales = (blocks.abs().amax(dim=-1) * (GLOBAL_SCALE / 6)).to(torch.float8_e4m3fn)
    effective = scales.float() / GLOBAL_SCALE
    normalized = torch.where(effective[..., None] > 0, blocks / effective[..., None], 0)
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    magnitude = normalized.abs().clamp_max(6).contiguous()
    codes = torch.bucketize(magnitude, thresholds)
    ties = (codes < 7) & (magnitude == thresholds[codes.clamp_max(6)])
    codes += (ties & ((codes & 1) != 0)).to(codes.dtype)
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)[codes]
    restored = values * normalized.sign() * effective[..., None]
    return restored.reshape_as(x).to(torch.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("nvfp4", "bf16"), default="nvfp4")
    parser.add_argument("--hidden", type=int, choices=(4096, 6144, 7168), default=4096)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument(
        "--combine-reduction",
        choices=("fp32", "bf16_local_partial_planned"),
        default="fp32",
    )
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("--tokens must be nonnegative")
    planned = args.combine_reduction == "bf16_local_partial_planned"
    if planned and args.dtype != "bf16":
        parser.error("planned reduction requires --dtype bf16")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (4, 8):
        raise ValueError("This example requires four or eight ranks")
    options = {}
    if planned:
        # Select the rank-fanout dispatch and compact registered arena.
        # These choices remain fixed for the buffer's lifetime.
        options.update(
            communication_mode="coalesced",
            dispatch_mode="coalesced",
            combine_mode="coalesced",
        )
    buffer = RdmaMoeAlltoAll(
        dist.group.WORLD,
        max(1, args.tokens),
        args.hidden,
        288,
        config=RdmaMoeAlltoAllConfig(
            communication_dtype=args.dtype,
            combine_reduction=args.combine_reduction,
            **options,
        ),
    )
    try:
        # Bounded, structured smoke input: amax=3 in every block gives an
        # exactly representable block scale. Odd multiples of 1/32 avoid
        # quantization thresholds, so approximate device reciprocals cannot
        # turn this strict check into a tolerance-dependent comparison.
        indices = torch.arange(args.tokens * args.hidden, device=device)
        x = (((indices + 17 * rank) % 95) * 2 - 95).float() / 32
        x = x.reshape(args.tokens, args.hidden).to(torch.bfloat16)
        x.reshape(args.tokens, args.hidden // 16, 16)[..., 0] = 3.0
        # One distinct expert per source rank, on the next GPU.
        routes = torch.full(
            (args.tokens, 1),
            ((rank + 1) % world) * (288 // world),
            dtype=torch.int32,
            device=device,
        )
        weights = torch.ones((args.tokens, 1), dtype=torch.float32, device=device)
        dispatch_options = {"topk_weights": weights} if planned else {}
        recv_x, recv_scales, counts, handle, hook = buffer.dispatch(
            x,
            routes,
            nvfp4_global_scale=GLOBAL_SCALE,
            return_recv_hook=True,
            **dispatch_options,
        )
        hook()
        if args.dtype == "nvfp4":
            # Real experts would compute on these BF16 activations. Only
            # counts[e] rows are valid; leave all padding initialized to zero.
            expert_x = torch.zeros(
                (*recv_x.shape[:-1], args.hidden), dtype=torch.bfloat16, device=device
            )
            global_scales = torch.full(
                (*recv_x.shape[:-1], 1),
                GLOBAL_SCALE,
                dtype=torch.float32,
                device=device,
            )
            for expert, count in enumerate(counts.tolist()):
                if count:
                    expert_x[expert, :count] = buffer.dequantize_nvfp4_to_bf16(
                        recv_x[expert, :count],
                        global_scales[expert, :count],
                        recv_scales[expert, :count],
                    )
            # Model both quantization steps independently of the transport:
            # dispatch -> BF16 expert input -> quantized combine output.
            expected = nvfp4_roundtrip_reference(nvfp4_roundtrip_reference(x))
        else:
            expert_x, global_scales, expected = recv_x, None, x
        if planned:
            # The handle owns the dispatch-time route/weight snapshot. Changes
            # to these caller tensors do not change this combine operation.
            routes.fill_(-1)
            weights.zero_()
            output, hook = buffer.combine(expert_x, handle, return_recv_hook=True)
        else:
            output, hook = buffer.combine(
                expert_x,
                handle,
                topk_idx=routes,
                topk_weights=weights,
                global_scale=global_scales,
                return_recv_hook=True,
            )
        hook()
        torch.cuda.synchronize(device)
        # No broad atol hides transport errors or treats NVFP4 as lossless.
        passed = torch.tensor(int(torch.equal(output, expected)), dtype=torch.int32)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        assert passed.item(), "The round trip differs from its quantized reference"
        error = output.float() - x.float()
        max_abs = error.abs().max().item() if error.numel() else 0.0
        relative_rmse = (error.norm() / x.float().norm().clamp_min(1e-20)).item()
        metrics = torch.tensor([max_abs, relative_rmse], dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"RDMA/{args.dtype}/{args.combine_reduction}: "
                f"exact reference check passed; "
                f"hidden={args.hidden}, tokens/rank={args.tokens}, "
                f"max-rank BF16-input error: max_abs={metrics[0].item():.6g}, "
                f"relative_RMSE={metrics[1].item():.6g}",
                flush=True,
            )
    finally:
        torch.cuda.synchronize(device)
        buffer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
