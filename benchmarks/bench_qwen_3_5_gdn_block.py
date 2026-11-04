# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Time the backends of the experimental Qwen 3.5 GDN block at the Qwen3.8-27B geometry.

Every registered backend runs the same plan-bound call (decode rows, MTP verify requests x T) under CUDA-graph replay
with a cold L2 (CUDA-event timing of graph replays), in two passes, the second in reverse backend order; each cell is
the median of both passes. A backend registered outside this package is loaded with ``--register module`` (a module that calls
``flashinfer.experimental.qwen_3_5_gdn_block.api.register_backend`` on import). There is no unfused baseline here: the
serving chain the block replaces lives in the serving framework.

    python benchmarks/bench_qwen_3_5_gdn_block.py --backends frost
    python benchmarks/bench_qwen_3_5_gdn_block.py --register my_pkg.gdn_backend --backends frost mine
"""

import argparse
import importlib
import itertools
import statistics

import torch

from flashinfer.qwen_3_5_gdn_block import (
    Qwen35GdnGeometry,
    prepare_qwen_3_5_gdn_block,
    qwen_3_5_gdn_block_support,
)
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph

GEOMETRY = Qwen35GdnGeometry(
    d_model=5120,
    num_v_heads=48,
    num_k_heads=16,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel=4,
)


def make_inputs(form, requests, tokens, device):
    g = GEOMETRY
    hv, hk, dk, dv, k = (
        g.num_v_heads,
        g.num_k_heads,
        g.head_k_dim,
        g.head_v_dim,
        g.conv_kernel,
    )
    c = 2 * hk * dk + hv * dv
    rows = requests * tokens
    slots = rows + 8
    qkvz = torch.randn(rows, c + hv * dv, device=device).bfloat16() * 0.1
    state_len = tokens + k - 2
    conv = torch.randn(slots, state_len, c, device=device).bfloat16().transpose(1, 2)
    idx = (torch.randperm(slots - 1, device=device)[:rows] + 1).int()
    x = dict(
        qkv=qkvz[:, :c],
        z=qkvz[:, c:].view(rows, hv, dv),
        hidden=torch.randn(rows, g.d_model, device=device).bfloat16(),
        ba_weight=(torch.randn(2 * hv, g.d_model, device=device) * 0.02).bfloat16(),
        A_log=torch.randn(hv, device=device) * 0.5,
        dt_bias=(torch.randn(hv, device=device) * 0.5).bfloat16(),
        ssm_state=torch.randn(slots, hv, dv, dk, device=device) * 0.1,
        state_indices=idx.view(requests, tokens) if form == "verify" else idx,
        norm_weight=(1 + 0.2 * torch.randn(dv, device=device)).bfloat16(),
        out_scale=torch.tensor([0.05], device=device),
        conv_state=conv,
        conv_weight=(torch.randn(c, k, device=device) * 0.3).bfloat16(),
        out=torch.empty(rows, hv, dv, dtype=torch.float8_e4m3fn, device=device),
    )
    if form == "verify":
        x["cu_seqlens"] = torch.arange(
            0, rows + 1, tokens, device=device, dtype=torch.int32
        )
        x["num_accepted_tokens"] = torch.full(
            (requests,), tokens, device=device, dtype=torch.int32
        )
    return x


def touched_bytes(rows):
    g = GEOMETRY
    c = 2 * g.num_k_heads * g.head_k_dim + g.num_v_heads * g.head_v_dim
    state = 2 * rows * g.num_v_heads * g.head_v_dim * g.head_k_dim * 4
    weights = 2 * g.num_v_heads * g.d_model * 2 + c * g.conv_kernel * 2
    activations = rows * (c + g.num_v_heads * g.head_v_dim + g.d_model) * 2
    return state + weights + activations + rows * c * g.conv_kernel * 2


def time_plan(plan, form, requests, tokens, device, props):
    # Cold L2 by rotating whole input sets inside the graph: the utility's
    # clone-based rotation would make the strided qkv / z views contiguous, a
    # different binding than serving passes.
    n_sets = min(
        64,
        max(4, -(-3 * props.L2_cache_size // touched_bytes(requests * tokens))),
    )
    sets = [make_inputs(form, requests, tokens, device) for _ in range(n_sets)]
    for x in sets:
        plan.run(**x)
    rotation = itertools.cycle(sets)
    return bench_gpu_time_with_cudagraph(
        lambda: plan.run(**next(rotation)),
        dry_run_iters=10,
        repeat_iters=100,
        num_iters_within_graph=n_sets,
        cold_l2_cache=False,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backends", nargs="+", default=["frost"])
    p.add_argument("--register", nargs="*", default=[])
    p.add_argument("--decode-rows", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--verify-requests", nargs="*", type=int, default=[1, 2, 4])
    p.add_argument("--verify-tokens", type=int, default=6)
    args = p.parse_args()
    for m in args.register:
        importlib.import_module(m)

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    print(
        f"# {props.name} (sm_{props.major}{props.minor}, {props.multi_processor_count} SMs)"
    )
    shapes = [("decode", r, 1) for r in args.decode_rows] + [
        ("verify", r, args.verify_tokens) for r in args.verify_requests
    ]
    print(
        f"{'form':<7} {'req':>4} {'T':>3} "
        + " ".join(f"{b:>12}" for b in args.backends)
    )
    for form, requests, tokens in shapes:
        plans = {}
        for backend in args.backends:
            sup = qwen_3_5_gdn_block_support(
                device,
                GEOMETRY,
                form,
                backend,
                requests=requests,
                tokens_per_request=tokens,
            )
            if sup.served:
                plans[backend] = prepare_qwen_3_5_gdn_block(
                    form,
                    geometry=GEOMETRY,
                    device=device,
                    requests=requests,
                    tokens_per_request=tokens,
                    backend=backend,
                )
        # The second pass runs the backends in reverse order, so clock or
        # thermal drift does not favor the one that runs first.
        times = {b: [] for b in plans}
        for order in (list(plans), list(plans)[::-1]):
            for backend in order:
                times[backend] += time_plan(
                    plans[backend], form, requests, tokens, device, props
                )
        cells = [
            f"{statistics.median(times[b]) * 1e3:>9.2f} us"
            if b in plans
            else f"{'-':>12}"
            for b in args.backends
        ]
        print(f"{form:<7} {requests:>4} {tokens:>3} " + " ".join(cells))


if __name__ == "__main__":
    main()
