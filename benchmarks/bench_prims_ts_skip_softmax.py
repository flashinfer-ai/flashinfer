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
"""

"""Benchmark skip softmax in the PrimTS context (prefill) kernel.

Times the dense PrimTS context kernel against its skip-softmax specialization
at several thresholds on one fixed-shape problem per configuration. Random
N(0, 0.2) inputs keep every K/V tile's scaled score gap to the running maximum
inside roughly exp(gap) in [0.7, 1.4], so ``threshold=0`` measures the cost of
the skip machinery without any skipping, ``threshold=1`` skips a data-dependent
subset of tiles, and ``threshold=2`` skips every tile after a softmax warp's
first one, which bounds the attainable speedup.

Example:
    python benchmarks/bench_prims_ts_skip_softmax.py --iters 30
"""

import argparse
import math
import statistics

import torch

from flashinfer.attention.prims_ts import BatchPrefillTSWrapper
from flashinfer.testing import bench_gpu_time


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8": torch.float8_e4m3fn,
    }[name]


def bench_config(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    mask_type: str,
    thresholds: tuple[float, ...],
    iters: int,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    q = (
        0.2 * torch.randn(batch_size, seq_len_q, num_qo_heads, head_dim, device=device)
    ).to(dtype)
    k = (
        0.2 * torch.randn(batch_size, seq_len_kv, num_kv_heads, head_dim, device=device)
    ).to(dtype)
    v = (0.2 * torch.randn_like(k.float())).to(dtype)
    sm_scale = 1.0 / math.sqrt(head_dim)
    out_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
    out = torch.empty(q.shape, device=device, dtype=out_dtype)

    dense = BatchPrefillTSWrapper()
    dense.plan(q, k, v, mask_type=mask_type, sm_scale=sm_scale, out_dtype=out_dtype)
    dense_out = dense.run(q, k, v)

    causal_fraction = 0.5 if mask_type == "causal" else 1.0
    flops = (
        4.0
        * batch_size
        * num_qo_heads
        * seq_len_q
        * seq_len_kv
        * head_dim
        * causal_fraction
    )
    print(
        f"\nB={batch_size} Sq={seq_len_q} Sk={seq_len_kv} Hq={num_qo_heads} "
        f"Hkv={num_kv_heads} D={head_dim} {str(dtype).split('.')[-1]} {mask_type}"
    )
    print(
        f"{'threshold':>10} {'median_us':>10} {'TFLOP/s':>9} {'speedup':>8} {'max|out-dense|':>15}"
    )
    baseline_us = None
    for threshold in (None, *thresholds):
        wrapper = BatchPrefillTSWrapper()
        wrapper.plan(
            q,
            k,
            v,
            mask_type=mask_type,
            sm_scale=sm_scale,
            out_dtype=out_dtype,
            skip_softmax_threshold=threshold,
        )
        wrapper.run(q, k, v, out=out)
        torch.cuda.synchronize()
        times_ms = bench_gpu_time(
            lambda: wrapper.run(q, k, v, out=out),
            dry_run_iters=5,
            repeat_iters=iters,
            enable_cupti=True,
        )
        median_us = statistics.median(times_ms) * 1e3
        if baseline_us is None:
            baseline_us = median_us
        max_abs = (out.float() - dense_out.float()).abs().max().item()
        label = "dense" if threshold is None else f"{threshold:g}"
        print(
            f"{label:>10} {median_us:10.1f} {flops / median_us / 1e6:9.1f} "
            f"{baseline_us / median_us:8.2f} {max_abs:15.4g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.0, 1.0, 2.0),
        help="Skip-softmax thresholds in the ordinary softmax domain.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        choices=(128, 256),
        default=None,
        help="Only run the configurations with this head dimension.",
    )
    args = parser.parse_args()
    thresholds = tuple(args.thresholds)
    configs = (
        dict(
            batch_size=1,
            seq_len_q=8192,
            seq_len_kv=8192,
            num_qo_heads=24,
            num_kv_heads=24,
            head_dim=128,
            dtype="bf16",
            mask_type="dense",
        ),
        dict(
            batch_size=1,
            seq_len_q=8192,
            seq_len_kv=8192,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim=128,
            dtype="bf16",
            mask_type="causal",
        ),
        dict(
            batch_size=1,
            seq_len_q=8192,
            seq_len_kv=8192,
            num_qo_heads=24,
            num_kv_heads=24,
            head_dim=128,
            dtype="fp8",
            mask_type="dense",
        ),
        dict(
            batch_size=1,
            seq_len_q=4096,
            seq_len_kv=4096,
            num_qo_heads=16,
            num_kv_heads=16,
            head_dim=256,
            dtype="bf16",
            mask_type="dense",
        ),
        dict(
            batch_size=1,
            seq_len_q=4096,
            seq_len_kv=4096,
            num_qo_heads=16,
            num_kv_heads=16,
            head_dim=256,
            dtype="fp8",
            mask_type="dense",
        ),
        dict(
            batch_size=1,
            seq_len_q=4096,
            seq_len_kv=4096,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            dtype="bf16",
            mask_type="causal",
        ),
    )
    for config in configs:
        if args.head_dim is not None and config["head_dim"] != args.head_dim:
            continue
        bench_config(
            **{**config, "dtype": _dtype(config["dtype"])},
            thresholds=thresholds,
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
