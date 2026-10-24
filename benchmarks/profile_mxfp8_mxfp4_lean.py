# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""N64/N128 single-GPU proxy for Nsight Compute: lean FC12 at synthetic per-expert geometry.

The multi-rank mega kernel waits on live peers, so Nsight Compute's kernel replay
cannot profile it in place. This builds the communication-free lean FC12 path
(the core shared with mega) for one rank's 32 local experts with uniform
per-expert token counts, checks it against the independent reference, then
runs one NVTX-marked launch for ``ncu --nvtx --nvtx-include ncu_target/``.
This is a PROXY: no dispatch, combine, or peer traffic is included.

``--replay-iters K`` also times K back-to-back CUDA-graph replays of one
launch. Sustained load is what sets the SM clock under the power limit,
as in a distributed benchmark; a single launch or Nsight Compute's replay passes
can run at a different clock. With ``--wait-stats`` the last replay's
counters are reported too (the launcher zeroes them inside the graph).

Inputs mirror ``tests/moe_ep/test_mxfp8_mxfp4_lean_gpu.py::_case``, except
that weight quantization and the reference run one expert at a time: the
reference MXFP4 quantizer broadcasts against its 16-entry LUT, so processing
all experts together can require a large temporary allocation.
"""

import argparse
import json
import os
import statistics
import time

import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_abi import (
    LeanFc12Config,
    LeanFc12Inputs,
    build_lean_row_plan,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_kernel import (
    LeanFc12Launcher,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.reference import (
    compute_reference,
    interleave_gate_up_8,
    quantize_mxfp4,
    quantize_mxfp8,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
    to_atom_swizzled,
)

parser = argparse.ArgumentParser()
parser.add_argument("--intermediate", type=int, required=True)
parser.add_argument("--tokens-per-expert", type=int, required=True)
parser.add_argument("--experts", type=int, default=32)
parser.add_argument("--hidden", type=int, default=6400)
parser.add_argument(
    "--num-stages", choices=("auto", "2", "3", "4", "5", "6", "7", "8"), default="auto"
)
parser.add_argument("--tile-tokens", type=int, choices=(64, 128, 192), default=64)
parser.add_argument(
    "--cluster",
    default=None,
    help="Cluster shape 'CM,CN' for the 1-CTA tile: CM>1 multicasts each token "
    "tile to CM weight tiles, CN>1 multicasts each weight tile to CN token "
    "tiles (default 1,1; --two-cta implies 2,1).",
)
parser.add_argument(
    "--two-cta",
    action="store_true",
    help="2-CTA pair MMA (M256, cluster (2, 1, 1)): each CTA stages its own "
    "128 weight rows and half of the token tile.",
)
parser.add_argument(
    "--group-hint",
    type=int,
    default=None,
    help="Scheduler FC1-tile budget per group. The lean launcher defaults to the "
    "SM count (small groups); the shipped EP16 default is 4096 below 1024 "
    "tokens/rank.",
)
parser.add_argument("--seed", type=int, default=20260924)
parser.add_argument("--warmup-iters", type=int, default=3)
parser.add_argument(
    "--mismatch-budget",
    type=int,
    default=0,
    help="Elements allowed outside the reference tolerance (recorded in the JSON).",
)
parser.add_argument(
    "--wait-stats",
    action="store_true",
    help="Record per-role %%globaltimer wait accounting (adds global atomics; "
    "do not run under Nsight Compute or use for timing).",
)
parser.add_argument(
    "--replay-iters",
    type=int,
    default=0,
    help="Time this many back-to-back CUDA-graph replays of one launch "
    "(sustained load; 0 skips).",
)
args = parser.parse_args()
if args.wait_stats:
    # Read when the kernel is traced and the launcher allocates its counters.
    os.environ["FLASHINFER_MXFP8_MXFP4_WAIT_STATS"] = "1"
if (
    min(
        args.tokens_per_expert,
        args.experts,
        args.hidden,
        args.intermediate,
        args.warmup_iters,
    )
    <= 0
    or args.replay_iters < 0
):
    parser.error("counts, dimensions, and warmup iterations must be positive")

hidden, intermediate, experts = args.hidden, args.intermediate, args.experts
counts = [args.tokens_per_expert] * experts
scale_alignment = 128
data_capacity = sum((c + 63) // 64 * 64 for c in counts)
scale_capacity = sum(
    (c + scale_alignment - 1) // scale_alignment * scale_alignment for c in counts
)
config = LeanFc12Config(
    experts,
    hidden,
    intermediate,
    data_capacity,
    scale_capacity,
    data_row_alignment=64,
    scale_row_alignment=scale_alignment,
)
plan = build_lean_row_plan(tuple(counts), config)
generator = torch.Generator(device="cuda").manual_seed(args.seed)


def random(shape, scale):
    return torch.randn(shape, generator=generator, device="cuda") * scale


magnitudes = (
    torch.tensor([0.125, 0.25, 0.5, 1.0, 2.0, 0.5, 0.25, 1.0], device="cuda")
    .repeat_interleave(32)
    .repeat((hidden + 255) // 256)[:hidden]
)
xq, xs = quantize_mxfp8(random((sum(counts), hidden), 0.5) * magnitudes)

w13q, w13s, w2q, w2s, expected = [], [], [], [], []
offset = 0
for count in counts:
    q13, s13 = quantize_mxfp4(
        interleave_gate_up_8(random((1, 2 * intermediate, hidden), 0.125))
    )
    q2, s2 = quantize_mxfp4(random((1, hidden, intermediate), 0.125))
    w13q.append(q13)
    w13s.append(s13)
    w2q.append(q2)
    w2s.append(s2)
    if count:
        ids = torch.zeros((1, count, 1), device="cuda", dtype=torch.int64)
        expected.append(
            compute_reference(
                input_q=xq[offset : offset + count].unsqueeze(0),
                input_scale=xs[offset : offset + count].unsqueeze(0),
                topk_ids=ids,
                topk_weights=torch.ones(ids.shape, device="cuda"),
                w13_packed=q13,
                w13_scale=s13,
                w2_packed=q2,
                w2_scale=s2,
            ).route_terms[0, :, 0]
        )
    offset += count
w13q, w13s = torch.cat(w13q), torch.cat(w13s)
w2q, w2s = torch.cat(w2q), torch.cat(w2s)
expected = torch.cat(expected)

activation = torch.zeros(
    (data_capacity, hidden), device="cuda", dtype=torch.float8_e4m3fn
)
scales = torch.full(
    (scale_capacity, hidden // 32), 127, device="cuda", dtype=torch.uint8
)
valid_rows = []
offset = 0
for expert, count in enumerate(counts):
    data_start = plan.expert_data_row_offsets[expert]
    scale_start = plan.expert_scale_row_offsets[expert]
    activation[data_start : data_start + count].copy_(xq[offset : offset + count])
    scales[scale_start : scale_start + count].copy_(xs[offset : offset + count])
    valid_rows.extend(range(data_start, data_start + count))
    offset += count


def metadata(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


inputs = LeanFc12Inputs(
    activation=activation,
    activation_scales=scales,
    fc1_weight=w13q.transpose(1, 2),
    fc1_weight_scales=torch.stack([to_atom_swizzled(s) for s in w13s]),
    fc2_weight=w2q.transpose(1, 2),
    fc2_weight_scales=torch.stack([to_atom_swizzled(s) for s in w2s]),
    output=torch.empty((data_capacity, hidden), device="cuda", dtype=torch.bfloat16),
    expert_token_sizes=metadata(counts),
    expert_data_row_offsets=metadata(plan.expert_data_row_offsets),
    expert_scale_row_offsets=metadata(plan.expert_scale_row_offsets),
)
rows = torch.tensor(valid_rows, device="cuda", dtype=torch.int64)
launcher = LeanFc12Launcher(config)
launcher.tactic["mma_tiler_mnk"] = (256 if args.two_cta else 128, args.tile_tokens, 128)
launcher.tactic["cluster_shape_mnk"] = (2, 1, 1) if args.two_cta else (1, 1, 1)
if args.cluster is not None:
    if args.two_cta:
        parser.error("--cluster applies to the 1-CTA tile")
    cm, cn = (int(v) for v in args.cluster.split(","))
    launcher.tactic["cluster_shape_mnk"] = (cm, cn, 1)
if args.group_hint is not None:
    launcher.tactic["group_hint"] = args.group_hint
launcher.tactic["num_stages"] = (
    "auto" if args.num_stages == "auto" else int(args.num_stages)
)

for _ in range(args.warmup_iters):
    actual = launcher.run(inputs)
torch.cuda.synchronize()
# Larger per-expert counts can leave a few elements outside the tolerance
# where FP8 re-quantization of the FC1 output rounds differently from the
# reference. The count is identical across tile widths and tile orders, so a
# budget only unblocks profiling; it is recorded and never a default.
diff = (actual[rows].float() - expected.float()).abs()
outside = diff > 0.015 + 0.05 * expected.float().abs()
mismatches = int(outside.sum())
max_abs_diff = float(diff.max())
if mismatches > args.mismatch_budget:
    torch.testing.assert_close(actual[rows], expected, atol=0.015, rtol=0.05)
print(
    f"lean proxy reference OK: counts={args.tokens_per_expert}x{experts} "
    f"H={hidden} I={intermediate} num_stages={launcher.resolved_num_stages}",
    flush=True,
)

wait_summary = None
if args.wait_stats:
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.wait_stats import (
        MAX_CTAS,
        SLOTS,
        Slot,
        summarize,
    )

    # Stats from the last warmup run; CTAs that ran the MMA loop are active.
    stats = launcher.wait_stats()
    active = int((stats.reshape(MAX_CTAS, SLOTS)[:, Slot.MMA_LOOP_NS] > 0).sum())
    wait_summary = {"active_ctas": active, **summarize(stats.cpu(), active)}

print(
    json.dumps(
        {
            "proxy": "lean-fc12-no-communication",
            "tile_tokens": args.tile_tokens,
            "requested_stages": args.num_stages,
            "resolved_stages": launcher.resolved_num_stages,
            "tactic": launcher.tactic,
            "seed": args.seed,
            "hidden": hidden,
            "intermediate": intermediate,
            "experts": experts,
            "tokens_per_expert": args.tokens_per_expert,
            "mismatches": mismatches,
            "mismatch_budget": args.mismatch_budget,
            "max_abs_diff": max_abs_diff,
            "wait_stats": wait_summary,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        }
    ),
    flush=True,
)

replay = None
if args.replay_iters:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launcher.run(inputs)
    events = [
        torch.cuda.Event(enable_timing=True) for _ in range(args.replay_iters + 1)
    ]
    graph.replay()  # settle before the timed window
    torch.cuda.synchronize()
    window_start = time.time()
    events[0].record()
    for event in events[1:]:
        graph.replay()
        event.record()
    torch.cuda.synchronize()
    window_end = time.time()
    times_us = [
        1000.0 * start.elapsed_time(stop)
        for start, stop in zip(events[:-1], events[1:], strict=True)
    ]
    times_us.sort()
    replay = {
        "iters": args.replay_iters,
        "median_us": statistics.median(times_us),
        "p10_us": times_us[len(times_us) // 10],
        "p90_us": times_us[(9 * len(times_us)) // 10],
        # Wall-clock window, to align external clock/power samples.
        "window_start_s": window_start,
        "window_end_s": window_end,
    }
    if args.wait_stats:
        stats = launcher.wait_stats()
        active = int((stats.reshape(MAX_CTAS, SLOTS)[:, Slot.MMA_LOOP_NS] > 0).sum())
        replay["wait_stats"] = {"active_ctas": active, **summarize(stats.cpu(), active)}
    print(json.dumps({"replay": replay}), flush=True)

torch.cuda.nvtx.range_push("ncu_target")
launcher.run(inputs)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
