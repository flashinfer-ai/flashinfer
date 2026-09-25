# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark the experimental Kimi-K3 fused MoE router on SM100/SM103.

The 28 routed shapes (``num_tokens in {1, 2, ..., 8192}`` x ``block_m in
{8, 16}``) are timed with CUPTI (``--cupti``, cupti-python >= 13) or CUDA
events and a cold L2 between iterations.  When ``sglang`` is importable the
same tensors are also routed by SGLang's ``moe_route_radix.route_radix`` plus
``moe_align.moe_align_block_size`` (the two-launch equivalent of this one
launch); the two arms are measured in interleaved rounds and the per-row and
geometric-mean speedups are printed.

Usage::

    python benchmarks/bench_cake_kimi_k3_fused_router.py [--cupti] [--rows m1024_bm8 ...]
"""

import argparse
import json
import math

import torch

from flashinfer.experimental.kimi_k3_fused_router.cake_backend import (
    BLOCK_M_VALUES,
    NUM_EXPERTS,
    SUPPORTED_NUM_TOKENS,
    TOP_K,
    allocate_kimi_k3_route_plan,
    max_route_blocks,
)
from flashinfer.fused_moe import prepare_kimi_k3_fused_router
from flashinfer.testing import bench_gpu_time

ROWS = {
    f"m{rows}_bm{bm}": (rows, bm)
    for bm in BLOCK_M_VALUES
    for rows in SUPPORTED_NUM_TOKENS
}


def make_inputs(num_tokens, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    expert = torch.arange(NUM_EXPERTS, dtype=torch.float32, device=device)
    token = torch.arange(num_tokens, dtype=torch.float32, device=device).reshape(-1, 1)
    logits = torch.randn((num_tokens, NUM_EXPERTS), generator=gen, device=device)
    logits = logits + expert.reshape(1, -1) * 1.0e-5 + token * 1.0e-7
    bias = torch.randn((NUM_EXPERTS,), generator=gen, device=device) * 0.05
    return logits.contiguous(), bias.contiguous()


def _sglang_route(logits, bias, block_m, device):
    """SGLang ``route_radix`` + ``moe_align_block_size`` closure, or None.

    Mirrors the two-launch baseline of the source benchmark: the route kernel
    is driven through the JIT module's ``run`` entry with caller-owned
    ``topk_weights`` / ``topk_ids`` (no per-call allocation) when that entry is
    available, otherwise through the allocating public ``route_radix`` wrapper;
    the align kernel always goes through ``moe_align_block_size``.
    """
    try:
        from sglang.kernels.ops.moe import moe_align, moe_route_radix
    except Exception:  # noqa: BLE001 - any import failure disables the baseline
        return None
    num_tokens = int(logits.shape[0])
    blocks = max_route_blocks(num_tokens, block_m)
    topk_weights = torch.empty((num_tokens, TOP_K), dtype=torch.float32, device=device)
    topk_ids = torch.empty((num_tokens, TOP_K), dtype=torch.int32, device=device)
    sorted_token_ids = torch.empty(blocks * block_m, dtype=torch.int32, device=device)
    expert_ids = torch.empty(blocks, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
    # SGLang shifts every expert id by one (expert -1 owns bucket zero), so the
    # align kernel receives E + 1 experts and E + 2 prefix entries.
    cumsum_buffer = torch.empty(NUM_EXPERTS + 2, dtype=torch.int32, device=device)

    route_raw = None
    builder = getattr(moe_route_radix, "_jit_route_radix_module", None)
    if callable(builder):
        try:
            route_raw = getattr(builder(), "run", None)
        except Exception:  # noqa: BLE001 - fall back to the public wrapper
            route_raw = None

    if callable(route_raw):

        def route():
            # (scores, bias, weights, ids, topk, routed_scaling_factor,
            #  renormalize, apply_scale, sorted)
            route_raw(
                logits, bias, topk_weights, topk_ids, TOP_K, 1.0, True, False, False
            )
            return topk_ids

    else:

        def route():
            _, ids = moe_route_radix.route_radix(
                logits, bias, TOP_K, True, 1.0, False, False
            )
            return ids

    def run():
        ids = route()
        moe_align.moe_align_block_size(
            topk_ids=ids,
            num_experts=NUM_EXPERTS + 1,
            block_size=block_m,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_pad=num_tokens_post_pad,
            cumsum_buffer=cumsum_buffer,
            pad_sorted_token_ids=True,
        )

    run()
    torch.cuda.synchronize()
    return run


def _median_us(fn, *, cupti):
    times = bench_gpu_time(fn, enable_cupti=cupti, cold_l2_cache=True)
    times = sorted(times)
    return 1000.0 * float(times[len(times) // 2])


def _paired_us(fused, baseline, *, cupti, rounds):
    """Interleaved A/B rounds; returns the median over rounds of each arm."""
    fused_times, baseline_times = [], []
    for round_index in range(rounds):
        order = (fused, baseline) if round_index % 2 == 0 else (baseline, fused)
        for fn in order:
            value = _median_us(fn, cupti=cupti)
            (fused_times if fn is fused else baseline_times).append(value)
    return (
        sorted(fused_times)[len(fused_times) // 2],
        sorted(baseline_times)[len(baseline_times) // 2],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="*", default=list(ROWS))
    parser.add_argument("--cupti", action="store_true", help="CUPTI kernel time")
    parser.add_argument("--rounds", type=int, default=5, help="interleaved A/B rounds")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    results = []
    print(f"{torch.cuda.get_device_name(device)}, {len(args.rows)} rows")
    header = f"{'row':<14}{'arm':>5}{'grid':>6}{'fused us':>10}"
    have_sglang = None
    for name in args.rows:
        num_tokens, block_m = ROWS[name]
        logits, bias = make_inputs(
            num_tokens, device, seed=4568000 + num_tokens + block_m
        )
        plan = allocate_kimi_k3_route_plan(num_tokens, block_m, device)
        runner = prepare_kimi_k3_fused_router(logits, bias, block_m=block_m, plan=plan)
        runner()
        torch.cuda.synchronize()
        baseline = _sglang_route(logits, bias, block_m, device)
        if have_sglang is None:
            have_sglang = baseline is not None
            if have_sglang:
                header += f"{'sglang us':>11}{'speedup':>9}"
            else:
                print("sglang not importable: reporting fused timings only")
            print(header)
        row = dict(
            row=name,
            num_tokens=num_tokens,
            block_m=block_m,
            arm=runner.arm,
            grid=runner.grid_x,
        )
        if baseline is None:
            fused_us = _median_us(runner, cupti=args.cupti)
            row["fused_us"] = fused_us
            print(f"{name:<14}{runner.arm:>5}{runner.grid_x:>6}{fused_us:>10.2f}")
        else:
            fused_us, baseline_us = _paired_us(
                runner, baseline, cupti=args.cupti, rounds=args.rounds
            )
            row.update(
                fused_us=fused_us, sglang_us=baseline_us, speedup=baseline_us / fused_us
            )
            print(
                f"{name:<14}{runner.arm:>5}{runner.grid_x:>6}{fused_us:>10.2f}"
                f"{baseline_us:>11.2f}{baseline_us / fused_us:>9.3f}"
            )
        results.append(row)
    if have_sglang:
        speedups = [r["speedup"] for r in results]
        print(
            "geomean speedup vs sglang route_radix + moe_align_block_size: "
            f"{math.exp(sum(map(math.log, speedups)) / len(speedups)):.3f}"
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                dict(device=torch.cuda.get_device_name(device), rows=results),
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
