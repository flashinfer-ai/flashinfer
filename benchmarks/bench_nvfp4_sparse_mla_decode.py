"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

NVFP4 sparse MLA decode (flashinfer.mla.nvfp4_sparse_mla_decode, SM100) against FP8 TRTLLM-gen sparse MLA
(flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla) at DSA decode shapes: 16 heads per rank, top-k 2048,
speculative decoding with q_len query tokens per request.

Both kernels attend to the same selected rows, NVFP4 from 352-byte nvfp4_ds_mla rows and FP8 from 576-byte e4m3
rows. The indices are request-shaped: a request's q_len tokens pick their keys from one pool of that request's
context (recent rows over-represented), as a sparse indexer does. Uniformly random indices over the whole cache
make TRTLLM-gen up to ~20x slower than it runs in an engine. Times are CUDA-graph replays with warm L2.

    python benchmarks/bench_nvfp4_sparse_mla_decode.py --requests 1 2 3 4 5 6 7 8 9
"""

import argparse
import math
import statistics
import warnings

import torch

import flashinfer
from flashinfer.testing import bench_gpu_time

BLOCK = 1024  # rows per KV block
NUM_HEADS, HEAD_DIM, V_HEAD_DIM, ROW_BYTES = 16, 576, 512, 352


def request_shaped_indices(num_requests, q_len, topk, num_blocks, context, g):
    dev = g.device
    blocks = torch.randperm(num_blocks, device=dev, generator=g)
    per_request = context // BLOCK
    rows = []
    for r in range(num_requests):
        pages = blocks[r * per_request : (r + 1) * per_request]
        ctx = (
            pages[:, None] * BLOCK + torch.arange(BLOCK, device=dev)[None, :]
        ).reshape(-1)
        recent, old = ctx[-4096:], ctx[:-4096]
        pool = torch.cat(
            [
                recent[torch.randperm(recent.numel(), device=dev, generator=g)[:1100]],
                old[
                    torch.randperm(old.numel(), device=dev, generator=g)[
                        : max(topk - 1100, 1900)
                    ]
                ],
            ]
        )
        for _ in range(q_len):
            rows.append(
                pool[torch.randperm(pool.numel(), device=dev, generator=g)[:topk]]
            )
    return torch.stack(rows).to(torch.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument(
        "--requests", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    parser.add_argument(
        "--q-len",
        type=int,
        default=5,
        help="query tokens per request (1 + draft tokens)",
    )
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument(
        "--context", type=int, default=50 * 1024, help="context tokens per request"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=1400,
        help=f"KV blocks of {BLOCK} rows in the cache",
    )
    args = parser.parse_args()
    dev = torch.device("cuda")
    if torch.cuda.get_device_capability(dev) not in ((10, 0), (10, 3)):
        raise SystemExit(
            "NVFP4 sparse MLA decode needs compute capability 10.0 or 10.3"
        )
    if max(args.requests) * (args.context // BLOCK) > args.blocks:
        raise SystemExit("not enough KV blocks for the requested contexts")
    g = torch.Generator(device=dev).manual_seed(0)
    rows = args.blocks * BLOCK
    kv = torch.randint(
        0, 256, (rows, ROW_BYTES), dtype=torch.uint8, device=dev, generator=g
    )
    rope = torch.randn(rows, 64, device=dev, generator=g) * 0.5
    kv[:, 256:320] = rope.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.rand(rows, 32, device=dev, generator=g) * 0.99 + 0.01
    kv[:, 320:] = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    fp8 = (torch.randn(rows, HEAD_DIM, device=dev, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    fp8_pages = fp8.view(-1, 64, HEAD_DIM).unsqueeze(1)
    workspace = torch.zeros(512 << 20, dtype=torch.int8, device=dev)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    print(f"{'requests':>8} {'T':>4} {'NVFP4 us':>9} {'FP8 us':>8} {'NVFP4/FP8':>10}")
    for num_requests in args.requests:
        idx = request_shaped_indices(
            num_requests, args.q_len, args.topk, args.blocks, args.context, g
        )
        T = idx.shape[0]
        q = (torch.randn(T, NUM_HEADS, HEAD_DIM, device=dev, generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )
        out = torch.empty(T, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device=dev)
        seq_lens = torch.full((T,), args.topk, dtype=torch.int32, device=dev)

        def run_nvfp4():
            flashinfer.mla.nvfp4_sparse_mla_decode(q, kv, idx, sm_scale, out=out)

        def run_fp8():
            flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
                query=q.unsqueeze(1),
                kv_cache=fp8_pages,
                workspace_buffer=workspace,
                qk_nope_head_dim=192,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                block_tables=idx.unsqueeze(1),
                seq_lens=seq_lens,
                max_seq_len=args.topk,
                sparse_mla_top_k=args.topk,
                bmm1_scale=sm_scale,
                bmm2_scale=1.0,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ExperimentalWarning
            run_nvfp4()  # compile and query occupancy outside the graph
        run_fp8()
        t_nvfp4 = statistics.median(
            bench_gpu_time(run_nvfp4, use_cuda_graph=True, cold_l2_cache=False)
        )
        t_fp8 = statistics.median(
            bench_gpu_time(run_fp8, use_cuda_graph=True, cold_l2_cache=False)
        )
        print(
            f"{num_requests:8d} {T:4d} {t_nvfp4 * 1e3:9.1f} {t_fp8 * 1e3:8.1f} {t_nvfp4 / t_fp8:10.2f}"
        )


if __name__ == "__main__":
    main()
