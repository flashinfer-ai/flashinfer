#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic MoK training example with independent BF16 autograd checks.

Run: torchrun --standalone --nproc-per-node=4 examples/mok_bf16_toy.py
Requires peer-accessible GPUs and PyTorch symmetric-memory multicast.
"""

import datetime
import json
import os

import torch
import torch.distributed as dist
from flashinfer.mok import prepare_mok_bf16, create_mok_bf16_workspace


RESULT_NAMES = (
    "y",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)


def expert(x, gate, up, down):
    a, b = x @ gate.T, x @ up.T
    # Native SwiGLU promotes BF16 inputs to FP32, then rounds the product once.
    hidden = (torch.nn.functional.silu(a.float()) * b.float()).to(x.dtype)
    return hidden @ down.T


def reference(global_data, weights, *, fp32=False, source_counts=None):
    rank, ep = dist.get_rank(), dist.get_world_size()
    device = weights[0].device
    dtype = torch.float32 if fp32 else torch.bfloat16
    x = global_data["x"].to(device=device, dtype=dtype)
    dy = global_data["d_output"].to(device=device, dtype=dtype)
    ids = global_data["expert_ids"].to(device)
    scores = global_data["scores"].to(device)
    t, h = x.shape
    local = t // ep
    if source_counts is None:
        source_counts = [local] * ep
    if len(source_counts) != ep or sum(source_counts) != t:
        raise ValueError("Source counts must cover the global input exactly")
    start = sum(source_counts[:rank])
    own_rows = slice(start, start + source_counts[rank])
    local_experts = weights[3].shape[0]
    y_sum = torch.zeros((t, h), dtype=torch.float32, device=device)
    dx_sum = torch.zeros_like(y_sum)
    dscores = torch.zeros_like(scores)
    dw = [torch.zeros_like(w, dtype=dtype) for w in weights[3:]]
    for e in range(local_experts):
        rows, slots = (ids == rank * local_experts + e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        xe = x[rows].detach().requires_grad_()
        we = [w[e].detach().to(dtype).requires_grad_() for w in weights[3:]]
        out = expert(xe, *we)
        # Differentiating already-scaled score inputs; scaling is not repeated.
        se = scores[rows, slots].detach().requires_grad_()
        weighted = out.float() * se[:, None]
        grads = torch.autograd.grad(weighted, (xe, *we, se), dy[rows].float())
        y_sum.index_add_(0, rows, weighted.detach())
        dx_sum.index_add_(0, rows, grads[0].float())
        dscores[rows, slots] = grads[-1]
        for dest, grad in zip(dw, grads[1:4], strict=True):
            dest[e].copy_(grad)
    for tensor in (y_sum, dx_sum, dscores):
        dist.all_reduce(tensor)
    xs = x[own_rows].detach().requires_grad_()
    ws = [w.detach().to(dtype).requires_grad_() for w in weights[:3]]
    ys = expert(xs, *ws)
    shared_grads = torch.autograd.grad(ys, (xs, *ws), dy[own_rows])
    y = (y_sum[own_rows] + ys.detach().float()).to(dtype)
    dx = (dx_sum[own_rows] + shared_grads[0].float()).to(dtype)
    return (y, dx, dscores[own_rows], *dw, *shared_grads[1:])


def error_report(actual, expected, gate):
    """Apply elementwise atol/rtol on every rank; retain aggregate diagnostics."""
    atol, rtol = float(gate["atol"]), float(gate["rtol"])
    if not (0 < atol < float("inf") and 0 <= rtol < float("inf")):
        raise ValueError("Require finite atol > 0 and rtol >= 0")
    reports = {}
    for name, a, b in zip(RESULT_NAMES, actual, expected, strict=True):
        assert a.shape == b.shape, (name, a.shape, b.shape)
        # First three fields use MAX; the remaining five use SUM across ranks.
        # Chunking bounds temporary storage for full expert weight gradients.
        stats = torch.zeros(8, dtype=torch.float64, device=a.device)
        worst = torch.zeros(4, dtype=torch.float64, device=a.device)
        offset = 0
        for aa, bb in zip(
            a.flatten().split(1048576), b.flatten().split(1048576), strict=True
        ):
            if aa.numel() == 0:
                continue
            af, bf = aa.float(), bb.float()
            diff = (af - bf).abs()
            allowed = atol + rtol * bf.abs()
            finite = torch.isfinite(af) & torch.isfinite(bf)
            mismatch = (~finite) | (diff > allowed)
            ratio = torch.nan_to_num(
                diff / allowed, nan=float("inf"), posinf=float("inf")
            )
            peak, index = ratio.max(dim=0)
            example = torch.stack(
                (
                    index.double() + offset,
                    af[index].double(),
                    bf[index].double(),
                    allowed[index].double(),
                )
            )
            worst = torch.where(peak > stats[1], example, worst)
            stats[0] = torch.maximum(
                stats[0], torch.nan_to_num(diff, nan=float("inf")).max().double()
            )
            stats[1] = torch.maximum(stats[1], peak.double())
            stats[2] = torch.maximum(
                stats[2],
                torch.nan_to_num(diff - allowed, nan=float("inf")).max().double(),
            )
            stats[3] += diff.double().sum()
            stats[4] += bf.abs().double().sum()
            stats[5] += (~torch.isfinite(af)).sum() + (~torch.isfinite(bf)).sum()
            stats[6] += mismatch.sum()
            stats[7] += aa.numel()
            offset += aa.numel()
        local = stats.tolist()
        local_worst = worst.tolist()
        dist.all_reduce(stats[:3], op=dist.ReduceOp.MAX)
        dist.all_reduce(stats[3:], op=dist.ReduceOp.SUM)
        maximum, ratio, excess, error, norm, nonfinite, mismatched, elements = (
            stats.tolist()
        )
        relative = error / norm if norm else (0.0 if error == 0 else float("inf"))
        local_relative = (
            local[3] / local[4]
            if local[4]
            else (0.0 if local[3] == 0 else float("inf"))
        )
        reports[name] = {
            "shape": list(a.shape),
            "actual_dtype": str(a.dtype),
            "atol": atol,
            "rtol": rtol,
            "rank_max_absolute": local[0],
            "rank_max_error_ratio": local[1],
            "rank_max_tolerance_excess": local[2],
            "rank_abs_error_sum": local[3],
            "rank_reference_l1": local[4],
            "rank_relative_l1": local_relative,
            "rank_nonfinite": int(local[5]),
            "rank_mismatched": int(local[6]),
            "rank_elements": int(local[7]),
            "rank_worst_error": {
                "flat_index": int(local_worst[0]),
                "actual": local_worst[1],
                "reference": local_worst[2],
                "allowed_error": local_worst[3],
            },
            "global_max_absolute": maximum,
            "global_relative_l1": relative,
            "global_max_error_ratio": ratio,
            "global_max_tolerance_excess": excess,
            "global_nonfinite": int(nonfinite),
            "global_mismatched": int(mismatched),
            "global_elements": int(elements),
            "pass": nonfinite == 0 and mismatched == 0,
        }
    return reports


class TrainingIteration:
    def __init__(self, config, workspace, x, ids, scores, dy, weights, *, functional):
        if (
            x.dtype != torch.bfloat16
            or dy.dtype != torch.bfloat16
            or scores.dtype != torch.float32
            or ids.dtype != torch.int64
            or len(weights) != 6
            or any(
                not isinstance(v, torch.Tensor) or v.dtype != torch.bfloat16
                for v in weights
            )
        ):
            raise ValueError(
                "This workload requires BF16 x/dY/expert tensors, FP32 scores and int64 IDs"
            )
        if (
            x.shape != dy.shape
            or ids.shape != scores.shape
            or ids.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "Source input, routing and upstream-gradient shapes must agree"
            )
        self.config, self.workspace = config, workspace
        self.functional = functional
        self.x, self.ids, self.scores, self.dy = x, ids, scores, dy
        self.weights = weights
        self.graph = None
        self.outputs = None

    def _execute(self):
        self.schedule = self.functional.build_schedule(
            self.workspace,
            self.config,
            self.ids,
            num_local_experts=self.weights[3].shape[0],
        )
        y, self.context = self.functional.forward(
            self.config,
            self.workspace,
            self.schedule,
            self.x,
            self.scores,
            *self.weights,
        )
        gradients = self.functional.backward(
            self.config,
            self.workspace,
            self.schedule,
            self.context,
            self.dy,
            self.x,
            self.scores,
            *self.weights,
        )
        self.outputs = (y, *gradients)
        if any(
            v.dtype != (torch.float32 if idx == 2 else torch.bfloat16)
            for idx, v in enumerate(self.outputs)
        ):
            raise RuntimeError("BF16 output/gradient formats changed")
        return self.outputs

    def capture(self):
        if self.graph is not None:
            raise RuntimeError("Iteration is already captured")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._execute()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._execute()

    def run(self):
        if self.graph is None:
            return self._execute()
        self.graph.replay()
        return self.outputs


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(seconds=180)
        )
    rank, ep = dist.get_rank(), dist.get_world_size()
    assert ep in (1, 4, 16, 64), "Launch with 1, 4, 16 or 64 ranks"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tokens = 512
    topk, local_experts = (2, 4) if ep in (1, 4) else (8, 256 // ep)
    total_experts = ep * local_experts
    functional = prepare_mok_bf16(ep_size=ep, local_experts=local_experts, topk=topk)
    gate = dict(atol=1e-2, rtol=1e-2)
    cases = []
    for hidden, intermediate in ((256, 256), (512, 512)):
        config, workspace = create_mok_bf16_workspace(
            group=dist.group.WORLD,
            device=device,
            num_local_tokens=tokens,
            hidden_size=hidden,
            topk=topk,
            fwd_num_comm_sms=8,
            bwd_num_comm_sms=8,
            minibatch_size=256,
            macrobatch_size=768,
            schedule_capacity_multiplier=2.0,
        )
        options = dict(dtype=torch.bfloat16, device=device)
        torch.manual_seed(32471)
        shared = [
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(hidden, intermediate, **options) / intermediate**0.5,
        ]
        global_weights = [
            torch.randn(total_experts, intermediate, hidden, **options) / hidden**0.5,
            torch.randn(total_experts, intermediate, hidden, **options) / hidden**0.5,
            torch.randn(total_experts, hidden, intermediate, **options)
            / intermediate**0.5,
        ]
        weights = (
            *shared,
            *(
                w[rank * local_experts : (rank + 1) * local_experts]
                for w in global_weights
            ),
        )
        x, dy = (
            torch.empty(tokens, hidden, **options),
            torch.empty(tokens, hidden, **options),
        )
        ids = torch.empty(tokens, topk, dtype=torch.int64, device=device)
        scores = torch.empty(tokens, topk, device=device)
        iteration = TrainingIteration(
            config, workspace, x, ids, scores, dy, weights, functional=functional
        )

        def fill(generation, empty):
            torch.manual_seed(8291 + generation)
            all_x = torch.randn(tokens * ep, hidden, **options) * 0.125
            all_dy = torch.randn_like(all_x) * 0.125
            all_scores = torch.rand(tokens * ep, topk, device=device) + 0.1
            all_scores.div_(all_scores.sum(-1, keepdim=True)).mul_(2.5)
            rows = torch.arange(tokens * ep, device=device) + generation
            all_ids = torch.stack(
                [
                    (rows + slot) % (topk if empty else total_experts)
                    for slot in range(topk)
                ],
                -1,
            )
            section = slice(rank * tokens, (rank + 1) * tokens)
            for destination, value in zip(
                (x, dy, ids, scores), (all_x, all_dy, all_ids, all_scores), strict=True
            ):
                destination.copy_(value[section])
            torch.cuda.synchronize()
            dist.barrier()
            return dict(x=all_x, d_output=all_dy, expert_ids=all_ids, scores=all_scores)

        reports, loads = [], []

        def check(actual, expected, generation, empty):
            torch.cuda.synchronize()
            errors = error_report(actual, expected, gate)
            assert all(value["pass"] for value in errors.values()), (
                rank,
                hidden,
                generation,
                errors,
            )
            loads.append(iteration.schedule.num_tokens.item())
            report = dict(generation=generation, empty=empty, errors=errors)
            reports.append(report)
            print(
                f"rank {rank}: H{hidden} generation={generation} empty={empty} all nine PASS",
                flush=True,
            )

        for generation, empty in ((0, False), (1, True), (2, False)):
            data = fill(generation, empty)
            expected = reference(data, weights)
            check(iteration.run(), expected, generation, empty)
        iteration.capture()
        saved = []
        for generation, empty in (
            (11, False),
            (11, False),
            (11, False),
            (17, True),
            (23, False),
        ):
            data = fill(generation, empty)
            expected = reference(data, weights)
            actual = iteration.run()
            check(actual, expected, generation, empty)
            if generation == 11:
                saved.append(tuple(value.clone() for value in actual))
        assert all(
            all(torch.equal(a, b) for a, b in zip(saved[0], other, strict=True))
            for other in saved[1:]
        )
        cases.append(
            dict(
                hidden=hidden,
                intermediate=intermediate,
                source_tokens_per_rank=tokens,
                global_source_tokens=tokens * ep,
                topk=topk,
                global_experts=total_experts,
                macro=config.macrobatch_size,
                mini=config.minibatch_size,
                schedule_capacity=workspace.storage.schedule_capacity,
                routed_loads=loads,
                reports=reports,
                three_graph_replays_bitwise_equal=True,
            )
        )
    if rank == 0:
        print(json.dumps(dict(status="PASS", ep=ep, cases=cases), indent=2))
    if owns_group:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
