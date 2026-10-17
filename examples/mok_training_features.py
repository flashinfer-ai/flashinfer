#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Distributed MoK training features: clamped SwiGLU, context-only recompute,
FP32 weight-gradient accumulation and native MXFP8 routed experts.

Run: torchrun --standalone --nproc-per-node=4 examples/mok_training_features.py
EP4 uses 256 routed experts (64 per rank) and top-8; EP8 and EP32 use the
GLM-5.3-Flash layout of 288 routed experts (36 or 9 per rank). Ranks hold
unequal source counts, including an empty rank, and every case runs several
macrobatches so backward replays the forward ring. Requires peer-accessible
GPUs and PyTorch symmetric-memory multicast.

Checks per case:
- BF16 cases: every element of all nine outputs against the independent BF16
  autograd reference (atol = rtol = 1e-2). FP32 accumulators start from zero
  and are checked the same way.
- MXFP8 cases (BF16 or FP32 weight-gradient accumulation): global relative
  L2 error of the routed-dependent outputs against an FP32 autograd reference
  (quantization error), the BF16 shared expert weight gradients elementwise
  against the BF16 reference and within a tight relative L2 of FP32, and a
  distance from the BF16 result showing the routed GEMMs ran in MXFP8.
- Recompute cases: the checkpointed iteration (forward, context dropped,
  ``recompute_forward_context``, backward) is bitwise identical to the
  saved-context iteration on the same inputs.
- Every case: two eager runs and two CUDA Graph replays are bitwise equal.
"""

import datetime
import json
import os

import torch
import torch.distributed as dist

from flashinfer.mok import (
    create_mok_bf16_workspace,
    prepare_mok_bf16,
    quantize_mok_mxfp8_weights,
)
from mok_bf16_toy import RESULT_NAMES, TrainingIteration, error_report, reference

LAYOUTS = {1: (4, 2), 4: (64, 8), 8: (36, 8), 16: (16, 8), 32: (9, 8), 64: (4, 8)}
LIMIT = 0.125  # Clamps a substantial fraction of gate and up at this input scale.
# MXFP8 shifts gate/up by its quantization error; with most values near the
# clamp edge, clamp masks flip against the FP32 reference and dominate the
# error. A 3-sigma limit still clamps thousands of values per rank while keeping
# the comparison a measure of MXFP8 error (the exact clamped MXFP8 recipe is
# checked bitwise-close on one GPU against a quantization-aware reference).
MXFP8_LIMIT = 0.375
CASES = (
    dict(name="bf16-clamped", swiglu_limit=LIMIT),
    dict(name="bf16-recompute", recompute=True),
    dict(name="bf16-fp32-wgrad", fp32_wgrad=True),
    dict(name="mxfp8", mxfp8=True),
    dict(name="mxfp8-fp32-wgrad", mxfp8=True, fp32_wgrad=True),
    dict(
        name="mxfp8-clamped-recompute",
        mxfp8=True,
        swiglu_limit=MXFP8_LIMIT,
        recompute=True,
    ),
)
# MXFP8 quantization error bounds (global relative L2 against FP32). The BF16
# shared expert's weight gradients carry only BF16 rounding (and clamp-mask
# flips), so they get a much tighter bound.
MXFP8_REL_L2 = 0.15
SHARED_REL_L2 = 0.02
MXFP8_MIN_DISTANCE = 1e-3


def rel_l2(actual, expected):
    stats = torch.zeros(2, dtype=torch.float64, device=actual.device)
    stats[0] = (actual.double() - expected.double()).square().sum()
    stats[1] = expected.double().square().sum()
    dist.all_reduce(stats)
    return (stats[0] / stats[1]).sqrt().item() if stats[1] > 0 else 0.0


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(seconds=300)
        )
    try:
        rank, ep = dist.get_rank(), dist.get_world_size()
        if ep not in LAYOUTS:
            raise ValueError(f"Launch with one of {sorted(LAYOUTS)} ranks")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        local_experts, topk = LAYOUTS[ep]
        total_experts = ep * local_experts
        hidden, intermediate = 512, 256
        counts = ([513, 0, 1031, 256] * ep)[:ep] if ep > 1 else [1031]
        options = dict(device=device, dtype=torch.bfloat16)
        torch.manual_seed(32471)
        shared = [
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(hidden, intermediate, **options) / intermediate**0.5,
        ]
        global_routed = [
            torch.randn(total_experts, intermediate, hidden, **options) / hidden**0.5,
            torch.randn(total_experts, intermediate, hidden, **options) / hidden**0.5,
            torch.randn(total_experts, hidden, intermediate, **options)
            / intermediate**0.5,
        ]
        routed = [
            w[rank * local_experts : (rank + 1) * local_experts].contiguous()
            for w in global_routed
        ]
        weights = (*shared, *routed)
        mxfp8_weights = quantize_mok_mxfp8_weights(*routed)
        adapters = {
            False: prepare_mok_bf16(ep_size=ep, local_experts=local_experts, topk=topk),
            True: prepare_mok_bf16(
                ep_size=ep, local_experts=local_experts, topk=topk, fp32_wgrad=True
            ),
        }
        config, workspace = create_mok_bf16_workspace(
            group=dist.group.WORLD,
            device=device,
            num_local_tokens=counts[rank],
            hidden_size=hidden,
            topk=topk,
            fwd_num_comm_sms=8,
            bwd_num_comm_sms=8,
            minibatch_size=256,
            macrobatch_size=768,
            schedule_capacity_multiplier=1.0,
        )
        assert workspace.initial_source_counts == tuple(counts)
        torch.manual_seed(8291)
        total = sum(counts)
        data = dict(
            x=torch.randn(total, hidden, **options) * 0.125,
            d_output=torch.randn(total, hidden, **options) * 0.125,
            expert_ids=torch.rand(total, total_experts, device=device)
            .argsort(-1)[:, :topk]
            .contiguous(),
        )
        scores = torch.rand(total, topk, device=device) + 0.1
        data["scores"] = scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
        start = sum(counts[:rank])
        section = slice(start, start + counts[rank])
        x, ids, scores, dy = (
            data[k][section].clone() for k in ("x", "expert_ids", "scores", "d_output")
        )
        gate = dict(atol=1e-2, rtol=1e-2)
        bf16_outputs, records = None, []
        for case in CASES:
            limit = case.get("swiglu_limit")
            mxfp8 = case.get("mxfp8", False)
            fp32_wgrad = case.get("fp32_wgrad", False)
            functional = adapters[fp32_wgrad]
            functional.prepare(limit, mxfp8)
            functional.prepare_recompute(limit, mxfp8)
            accumulators = (
                [torch.zeros(w.shape, device=device) for w in weights]
                if fp32_wgrad
                else None
            )

            def iteration(recompute):
                return TrainingIteration(
                    config,
                    workspace,
                    x,
                    ids,
                    scores,
                    dy,
                    weights,
                    functional=functional,
                    swiglu_limit=limit,
                    mxfp8_weights=mxfp8_weights if mxfp8 else None,
                    recompute=recompute,
                    accumulators=accumulators,
                )

            def snapshot(values):
                torch.cuda.synchronize()
                return tuple(v.detach().clone() for v in values)

            saved_context = iteration(False)
            outputs = snapshot(saved_context.run())
            again = snapshot(saved_context.run())
            record = dict(case=case["name"], counts=counts)
            if limit is not None:
                # Fraction of shared gate pre-activations above the limit.
                above = (x.float() @ shared[0].float().T > limit).float()
                stats = torch.tensor([above.sum(), above.numel()], device=device)
                dist.all_reduce(stats)
                record["shared_gate_clamped_fraction"] = (stats[0] / stats[1]).item()
            record["eager_repeat_bitwise"] = all(
                torch.equal(a, b) for a, b in zip(outputs, again, strict=True)
            )
            dist.barrier()
            saved_context.capture()
            replays = [snapshot(saved_context.run()) for _ in range(2)]
            record["graph_replays_bitwise"] = all(
                torch.equal(a, b)
                for replay in replays
                for a, b in zip(outputs, replay, strict=True)
            )
            dist.barrier()
            if case.get("recompute"):
                checkpointed = snapshot(iteration(True).run())
                record["recompute_bitwise"] = all(
                    torch.equal(a, b)
                    for a, b in zip(outputs, checkpointed, strict=True)
                )
                differs = [
                    n
                    for n, a, b in zip(RESULT_NAMES, outputs, checkpointed, strict=True)
                    if not torch.equal(a, b)
                ]
                assert record["recompute_bitwise"], (rank, case["name"], differs)
            dist.barrier()
            expected = reference(
                data, weights, source_counts=counts, swiglu_limit=limit
            )
            if not mxfp8:
                errors = error_report(outputs, expected, gate)
                record["strict_pass"] = all(v["pass"] for v in errors.values())
                record["worst_ratio"] = {
                    k: v["global_max_error_ratio"] for k, v in errors.items()
                }
                assert record["strict_pass"], (rank, case["name"], errors)
                if limit is None and not fp32_wgrad and bf16_outputs is None:
                    bf16_outputs = outputs
            else:
                oracle = reference(
                    data, weights, fp32=True, source_counts=counts, swiglu_limit=limit
                )
                record["rel_l2_vs_fp32"] = {
                    n: rel_l2(a, b)
                    for n, a, b in zip(RESULT_NAMES, outputs, oracle, strict=True)
                }
                # The shared expert stays BF16: its weight gradients are exact
                # BF16-path results; routed outputs carry MXFP8 error.
                errors = error_report(outputs, expected, gate)
                shared_errors = {n: errors[n] for n in RESULT_NAMES[6:]}
                record["shared_strict_pass"] = all(
                    v["pass"] for v in shared_errors.values()
                )
                assert record["shared_strict_pass"], (rank, case["name"], shared_errors)
                bound = {n: MXFP8_REL_L2 for n in RESULT_NAMES[:6]}
                bound.update({n: SHARED_REL_L2 for n in RESULT_NAMES[6:]})
                assert all(record["rel_l2_vs_fp32"][n] < b for n, b in bound.items()), (
                    case["name"],
                    record["rel_l2_vs_fp32"],
                )
                if limit is None and bf16_outputs is not None:
                    record["rel_l2_vs_bf16_run"] = rel_l2(outputs[0], bf16_outputs[0])
                    assert record["rel_l2_vs_bf16_run"] > MXFP8_MIN_DISTANCE
            for key in ("eager_repeat_bitwise", "graph_replays_bitwise"):
                assert record[key], (rank, case["name"], key)
            if counts[rank] == 0:
                assert all(torch.count_nonzero(v).item() == 0 for v in outputs[6:])
            records.append(record)
            if rank == 0:
                print(json.dumps(record), flush=True)
            del saved_context
            torch.cuda.synchronize()
            dist.barrier()
        if rank == 0:
            print(
                json.dumps(
                    dict(status="PASS", ep=ep, experts=total_experts, topk=topk)
                ),
                flush=True,
            )
        return records
    finally:
        if owns_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
