"""Time sparse_paged_scores over the shapes a sparse-attention selector asks for.

The kernel scores every visible entry of a paged KV cache against a multi-head
query, which is the input a top-k ranks. Rows is how many query tokens a step
scores at once, and columns is how far back the selector looks.
"""

import argparse
import math

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

# rows, num_heads, head_dim, columns, num_requests
SHAPES = [
    (1, 4, 128, 1024, 1),
    (1, 16, 128, 4096, 1),
    (8, 16, 128, 4096, 1),
    (16, 16, 128, 4096, 2),
    (40, 8, 256, 2048, 5),
    (48, 8, 256, 2048, 6),
    (64, 4, 128, 1024, 64),
    (64, 16, 128, 4096, 8),
    (512, 8, 256, 2048, 2),
    (2048, 4, 128, 1024, 2),
    (2048, 16, 128, 4096, 2),
]


def build(rows, num_heads, head_dim, columns, num_requests, page_size, ratio, dtype):
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(0)
    pages_per_request = (columns + page_size - 1) // page_size
    pages = pages_per_request * num_requests

    q = torch.randn(
        rows, num_heads, head_dim, dtype=dtype, device=device, generator=gen
    )
    k_cache = torch.randn(
        pages, page_size, head_dim, dtype=dtype, device=device, generator=gen
    )
    # Shuffled so the gather chases pages rather than walking them in order, and
    # every eleventh entry unmapped so the masked path is exercised too.
    page_table = (
        torch.randperm(pages, device=device, generator=gen)
        .reshape(num_requests, pages_per_request)
        .contiguous()
        .to(torch.int32)
    )
    page_table[:, ::11] = -1

    per_request = max(1, rows // num_requests)
    token_to_req = (
        (torch.arange(rows, device=device, dtype=torch.int32) // per_request)
        .clamp(max=num_requests - 1)
        .contiguous()
    )
    seqlen = columns * ratio
    # Spread the rows of a request over its sequence, with the last one seeing
    # all of it: a row at position zero scores nothing and would time the launch
    # rather than the kernel.
    step = max(1, seqlen // max(1, per_request))
    positions = (
        (
            ((torch.arange(rows, device=device, dtype=torch.int32) % per_request) + 1)
            * step
            - 1
        )
        .clamp(min=0, max=seqlen - 1)
        .contiguous()
    )
    seq_lens = torch.full((num_requests,), seqlen, dtype=torch.int32, device=device)

    logits = torch.empty(rows, columns, dtype=torch.float32, device=device)
    visible = torch.empty(rows, dtype=torch.int32, device=device)

    def run():
        flashinfer.sparse_paged_scores(
            q,
            k_cache,
            page_table,
            token_to_req,
            positions,
            seq_lens,
            ratio,
            math.sqrt(head_dim),
            num_columns=columns,
            logits=logits,
            visible_blocks=visible,
        )

    return run


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--iters-within-graph", type=int, default=50)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    # Replayed from a graph. The smallest shapes here run in a few microseconds,
    # which is the same order as a launch, so timing them one launch at a time
    # measures the launch as much as the kernel. Warm cache: a step scores
    # several rows against the same pages, which is what the kernel is shaped
    # for, and the replays leave the cache alone.
    print(f"{'rows':>6} {'heads':>6} {'head_dim':>9} {'columns':>8} {'warm us':>9}")
    for rows, num_heads, head_dim, columns, num_requests in SHAPES:
        run = build(
            rows,
            num_heads,
            head_dim,
            columns,
            num_requests,
            args.page_size,
            args.compress_ratio,
            dtype,
        )
        times = bench_gpu_time(
            run,
            use_cuda_graph=True,
            num_iters_within_graph=args.iters_within_graph,
            # The closure holds its own tensors, so the helper has nothing to
            # rotate and cannot cold the cache between replays. Said here
            # rather than left to the default, which would disable itself with
            # a warning and read as a cold measurement.
            cold_l2_cache=False,
        )
        print(
            f"{rows:>6} {num_heads:>6} {head_dim:>9} {columns:>8} "
            f"{np.median(times) * 1e3:>9.2f}"
        )


if __name__ == "__main__":
    main()
