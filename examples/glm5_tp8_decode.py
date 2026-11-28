# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Minimal GLM-5 TP8 MoE decode example on one node with eight B300 GPUs.

Requires FlashInfer installed from source or a nightly wheel from
https://flashinfer.ai/whl; release wheels omit the experimental CUDA sources
that this operator compiles on first use. From the repository root:

    torchrun --standalone --nproc-per-node=8 examples/glm5_tp8_decode.py

This specialized operator supports TP=8, hidden size 6144, 256 routed experts
plus one shared expert, local intermediate size 256, and q_len=1 or q_len=4.
Its qualified target is B300 with 148 SMs; it is not a general-purpose MoE API.
The example uses zero expert weights, so the correct output equals the residual
input. Real callers pass one layer of their FP8 checkpoint, keyed by the
checkpoint tensor names (for example memory-mapped safetensors on the CPU);
every rank passes the same layer and the packer copies only its own slices to
the GPU. Weight packing happens once, outside CUDA graph capture.
"""

from datetime import timedelta
import os

import torch
import torch.distributed as dist

from flashinfer.glm5_tp8_decode import (
    create_glm5_tp8_decode_workspace,
    pack_glm5_tp8_moe_weights,
    prepare_glm5_tp8_moe,
)


def make_zero_expert_layer():
    """One sparse layer's MoE tensors under their checkpoint names, on the CPU.

    Every expert aliases the same zero codes and unit scales; the packer only
    reads its inputs. Byte views avoid FP8 fill kernels on the CPU.
    """
    gate_up = torch.zeros((2048, 6144), dtype=torch.uint8).view(torch.float8_e4m3fn)
    down = torch.zeros((6144, 2048), dtype=torch.uint8).view(torch.float8_e4m3fn)
    gate_up_scale = torch.ones((16, 48), dtype=torch.float32)
    down_scale = torch.ones((48, 16), dtype=torch.float32)
    layer = {
        "post_attention_layernorm.weight": torch.ones(6144, dtype=torch.bfloat16),
        "mlp.gate.weight": torch.zeros((256, 6144), dtype=torch.bfloat16),
        "mlp.gate.e_score_correction_bias": torch.zeros(256, dtype=torch.float32),
    }
    experts = ["mlp.shared_experts"] + [f"mlp.experts.{e}" for e in range(256)]
    for prefix in experts:
        for name in ("gate_proj", "up_proj"):
            layer[f"{prefix}.{name}.weight"] = gate_up
            layer[f"{prefix}.{name}.weight_scale_inv"] = gate_up_scale
        layer[f"{prefix}.down_proj.weight"] = down
        layer[f"{prefix}.down_proj.weight_scale_inv"] = down_scale
    return layer


def check_residual(actual, expected, group):
    # A collective check ensures all workers take the same cleanup path.
    torch.cuda.synchronize(actual.device)
    local_ok = torch.equal(actual, expected)
    accepted = [None] * dist.get_world_size(group)
    dist.all_gather_object(accepted, local_ok, group=group)
    if not all(accepted):
        raise AssertionError(f"zero-expert residual check failed on ranks {accepted}")


def main():
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        raise RuntimeError("launch this example with torchrun --nproc-per-node=8")
    local_rank = int(os.environ["LOCAL_RANK"])
    device_index = 0 if torch.cuda.device_count() == 1 else local_rank
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    workspace = None
    graphs = {}
    try:
        # Collective creation compiles the kernels and initializes all their
        # entry points before the first peer exchange, including under
        # CUDA_MODULE_LOADING=LAZY.
        workspace = create_glm5_tp8_decode_workspace(
            rank=rank,
            group=group,
            device=device,
            max_seq_len=4096,
            operators=("moe",),
        )
        weights = pack_glm5_tp8_moe_weights(
            make_zero_expert_layer(), rank=rank, device=device
        )
        stream = torch.cuda.Stream(device=device)
        inputs, outputs, plans = {}, {}, {}
        for q_len in (1, 4):
            inputs[q_len] = torch.ones(
                (q_len, 6144), dtype=torch.bfloat16, device=device
            )
            outputs[q_len] = torch.empty_like(inputs[q_len])
            plans[q_len] = prepare_glm5_tp8_moe(
                inputs[q_len],
                weights,
                out=outputs[q_len],
                workspace=workspace,
            )
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

        # Every rank issues the same sequence. Keep calls on one stream so the
        # persistent collective grids and their shared scratch are serialized.
        for q_len in (1, 4):
            with torch.cuda.stream(stream):
                plans[q_len]()
            check_residual(outputs[q_len], inputs[q_len], group)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                plans[q_len]()
            graphs[q_len] = graph
            del graph

        # Addresses stay fixed while values change in place; no recapture or
        # host-side counter updates are required when alternating q1 and q4.
        for q_len, value in ((4, -0.75), (1, 0.5), (4, 1.25), (1, -1.0)):
            with torch.cuda.stream(stream):
                inputs[q_len].fill_(value)
                graphs[q_len].replay()
            check_residual(outputs[q_len], inputs[q_len], group)
        if rank == 0:
            print("PASS: TP8 q1/q4 eager calls and graph replay with changed inputs")
    finally:
        if workspace is not None:
            # Retain workspace, plans, weights and bound tensors until every
            # replay is complete. Captured graphs cannot outlive their workspace.
            torch.cuda.synchronize(device)
            graphs.clear()
            workspace.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
