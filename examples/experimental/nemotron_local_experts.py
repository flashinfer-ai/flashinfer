# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic SM103 example of native Nemotron plan lifetime and graph capture.

This is local computation only, with two live experts; it is not a representative
MoE performance benchmark. No checkpoint, process group or NVLS is required.
"""

import torch

from flashinfer.fused_moe.nemotron_mxfp8 import (
    pack_nemotron_mxfp8_weights,
    prepare_nemotron_mxfp8_local_experts,
)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(17)
    w1 = torch.zeros((128, 4096, 768), dtype=torch.float8_e4m3fn, device="cuda")
    w2 = torch.zeros((128, 768, 4096), dtype=torch.float8_e4m3fn, device="cuda")
    w1[:2].copy_((torch.randn((2, 4096, 768), device="cuda") / 32).to(w1.dtype))
    w2[:2].copy_((torch.randn((2, 768, 4096), device="cuda") / 64).to(w2.dtype))
    s1 = torch.full((128, 4096, 24), 127, dtype=torch.uint8, device="cuda")
    s2 = torch.full((128, 768, 128), 127, dtype=torch.uint8, device="cuda")
    weights = pack_nemotron_mxfp8_weights(w1, s1, w2, s2)
    plan = prepare_nemotron_mxfp8_local_experts(
        weights, max_tokens=128, local_expert_offset=0
    )
    x = torch.randn((128, 768), dtype=torch.bfloat16, device="cuda")
    ids = torch.full((128, 10), -1, dtype=torch.int64, device="cuda")
    ids[:, 0], ids[:, 1] = 0, 1
    probabilities = torch.full((128, 10), 0.5, device="cuda")
    output = torch.empty((128, 768), device="cuda")
    plan.run(x, ids, probabilities, out=output)
    expected = output.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(x, ids, probabilities, out=output)
    graph.replay()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print(f"Exact eager/graph replay: {tuple(output.shape)}, {output.dtype}")
    torch.cuda.synchronize()
    graph.reset()
    plan.close()


if __name__ == "__main__":
    main()
