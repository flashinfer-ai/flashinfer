#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Unequal source inputs, empty ranks and reusable MoK training graphs.

Run: torchrun --standalone --nproc-per-node=4 examples/mok_bf16_unequal.py
Requires peer-accessible GPUs and symmetric-memory multicast.
"""

import datetime
import json
import os

import torch
import torch.distributed as dist

from flashinfer.mok import create_mok_bf16_workspace, prepare_mok_bf16
from mok_bf16_toy import TrainingIteration, error_report, reference


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(seconds=180)
        )
    try:
        rank, ep = dist.get_rank(), dist.get_world_size()
        if ep != 4:
            raise ValueError("Launch this example with four ranks")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        hidden, intermediate, topk, local_experts = 256, 256, 2, 4
        options = dict(device=device, dtype=torch.bfloat16)
        functional = prepare_mok_bf16(
            ep_size=ep, local_experts=local_experts, topk=topk
        )
        torch.manual_seed(32471)
        shared = [
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(hidden, intermediate, **options) / intermediate**0.5,
        ]
        global_weights = [
            torch.randn(ep * local_experts, intermediate, hidden, **options)
            / hidden**0.5,
            torch.randn(ep * local_experts, intermediate, hidden, **options)
            / hidden**0.5,
            torch.randn(ep * local_experts, hidden, intermediate, **options)
            / intermediate**0.5,
        ]
        weights = (
            *shared,
            *(
                a[rank * local_experts : (rank + 1) * local_experts]
                for a in global_weights
            ),
        )
        gate = dict(max_absolute_error=0.01, relative_l1_error=0.01)
        cases = []
        for counts in ([0, 1, 255, 513], [0, 513, 769, 8193], [0, 0, 0, 0]):
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
                schedule_capacity_multiplier=2.0,
            )
            assert workspace.initial_source_counts == tuple(counts)

            def data_for(lengths, seed, empty_experts):
                torch.manual_seed(seed)
                total = sum(lengths)
                x = torch.randn(total, hidden, **options) * 0.125
                dy = torch.randn_like(x) * 0.125
                scores = torch.rand(total, topk, device=device) + 0.1
                scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
                row = torch.arange(total, device=device) + seed
                expert_count = topk if empty_experts else ep * local_experts
                ids = torch.stack(
                    [(row + slot) % expert_count for slot in range(topk)], -1
                )
                return dict(x=x, d_output=dy, expert_ids=ids, scores=scores)

            def iteration_for(lengths, data):
                start = sum(lengths[:rank])
                section = slice(start, start + lengths[rank])
                values = [
                    data[key][section].clone()
                    for key in ("x", "expert_ids", "scores", "d_output")
                ]
                return TrainingIteration(
                    config, workspace, *values, weights, functional=functional
                )

            reports = []

            def check(iteration, data, lengths):
                expected = reference(data, weights, source_counts=lengths)
                actual = iteration.run()
                torch.cuda.synchronize()
                errors = error_report(actual, expected, gate)
                assert all(a["pass"] for a in errors.values()), errors
                assert actual[0].shape == actual[1].shape == (lengths[rank], hidden)
                assert actual[2].shape == (lengths[rank], topk)
                if lengths[rank] == 0:
                    assert all(torch.count_nonzero(a).item() == 0 for a in actual[6:])
                reports.append(errors)
                saved = tuple(a.detach().cpu().clone() for a in actual)
                # All peers finish reading before another graph mutates symmetric storage.
                dist.barrier()
                return saved

            data = data_for(counts, 8291, False)
            first = iteration_for(counts, data)
            first.capture()
            saved = [check(first, data, counts) for _ in range(3)]
            assert all(
                all(torch.equal(a, b) for a, b in zip(saved[0], other, strict=True))
                for other in saved[1:]
            )
            changed_counts = counts[1:] + counts[:1]
            changed_data = data_for(changed_counts, 8299, True)
            second = iteration_for(changed_counts, changed_data)
            second.capture()
            check(second, changed_data, changed_counts)
            replayed = check(first, data, counts)
            assert all(
                torch.equal(a, b) for a, b in zip(saved[0], replayed, strict=True)
            )
            fresh_data = data_for(counts, 8311, True)
            start = sum(counts[:rank])
            section = slice(start, start + counts[rank])
            for destination, key in zip(
                (first.x, first.ids, first.scores, first.dy),
                ("x", "expert_ids", "scores", "d_output"),
                strict=True,
            ):
                destination.copy_(fresh_data[key][section])
            torch.cuda.synchronize()
            dist.barrier()
            check(first, fresh_data, counts)
            cases.append(
                dict(
                    source_counts=counts,
                    changed_source_counts=changed_counts,
                    source_capacity=workspace.source_capacity,
                    checks=len(reports),
                    three_replays_identical=True,
                    earlier_graph_replay=True,
                    same_shape_input_update=True,
                    worst_max_absolute=max(
                        e["global_max_absolute"]
                        for report in reports
                        for e in report.values()
                    ),
                    worst_relative_l1=max(
                        e["global_relative_l1"]
                        for report in reports
                        for e in report.values()
                    ),
                )
            )
        print(json.dumps(dict(status="PASS", rank=rank, cases=cases)), flush=True)
    finally:
        if owns_group:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
