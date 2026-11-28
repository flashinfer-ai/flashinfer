# SPDX-License-Identifier: Apache-2.0
"""GPU-only cached MRoPE timing; warm buffers, no packing, startup excluded."""

import argparse
import statistics
import torch
from flashinfer import apply_mrope_with_cos_sin_cache_inplace as mrope
from flashinfer.testing import bench_gpu_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 117, 213, 2048])
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--rotary-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--fp32-cache", action="store_true")
    args = parser.parse_args()
    print(torch.cuda.get_device_name())
    for n in args.tokens:
        qkv = torch.zeros(
            n,
            (args.q_heads + 2 * args.kv_heads) * 128,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = qkv[:, : args.q_heads * 128]
        k = qkv[:, args.q_heads * 128 : (args.q_heads + args.kv_heads) * 128]
        p = torch.randint(2048, (3, n), dtype=torch.int64, device="cuda")
        c = (
            torch.rand(
                2048,
                args.rotary_dim,
                device="cuda",
                dtype=torch.float32 if args.fp32_cache else q.dtype,
            )
            * 2
            - 1
        )
        half = args.rotary_dim // 2
        sections = (half - 2 * (half // 3), half // 3, half // 3)

        def run():
            mrope(p, q, k, 128, c, sections, True)

        run()
        preferred = bench_gpu_time(run, enable_cupti=True)
        # Explicit event crosscheck: CUPTI's fallback may use a shorter graph.
        events = bench_gpu_time(
            run, enable_cupti=False, use_cuda_graph=True, num_iters_within_graph=128
        )
        print(
            f"T={n} Hq={args.q_heads} Hk={args.kv_heads} R={args.rotary_dim} "
            f"preferred_us={statistics.median(preferred) * 1000:.3f} "
            f"event128_us={statistics.median(events) * 1000:.3f}"
        )


if __name__ == "__main__":
    main()
