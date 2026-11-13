# SPDX-License-Identifier: Apache-2.0
"""Runnable example of the experimental GC MoE API on one rank.

Stages a tiny MXFP4 checkpoint into a GC MoE workspace, runs the fused
dispatch -> per-expert compute -> combine kernel, and checks it against a
Torch reference. Every architecture stages all-ones scales, so the decode
below is shared by SM90, SM100 and SM120.

    python examples/experimental/gc_moe.py
"""

import os
import socket

import torch
import torch.distributed as dist

import flashinfer

NUM_EXPERTS = 4
HIDDEN = 512
INTERMEDIATE = 256
NUM_TOPK = 2
CAPACITY = 64
NUM_TOKENS = 8

FP4_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def decode_fp4(payload):
    """[E, N, K/2] int8 MXFP4 -> [E, N, K] float32 (even K in the low nibble)."""
    table = torch.tensor(FP4_TABLE, device=payload.device)
    raw = payload.view(torch.uint8).to(torch.int32)
    out = torch.empty(
        payload.shape[:-1] + (2 * payload.shape[-1],), device=payload.device
    )
    out[..., 0::2] = table[raw & 0x0F]
    out[..., 1::2] = table[(raw >> 4) & 0x0F]
    return out


def checkpoint_weights(device, generator):
    def payload(rows, k):
        return torch.randint(
            -128,
            128,
            (NUM_EXPERTS, rows, k // 2),
            dtype=torch.int8,
            generator=generator,
        ).to(device)

    capability = torch.cuda.get_device_capability()

    def scales(rows, k):
        if capability == (9, 0):
            return torch.ones((NUM_EXPERTS, rows, k // 32), device=device)
        if capability in ((10, 0), (10, 3)):
            # SM100 reads MN-major packed scales: [E, N, K/128] with stride(-2) == 1.
            return torch.full(
                (NUM_EXPERTS, k // 128, rows),
                0x7F7F7F7F,
                dtype=torch.int32,
                device=device,
            ).transpose(-1, -2)
        return torch.full(
            (NUM_EXPERTS, rows, k // 128), 0x7F7F7F7F, dtype=torch.int32, device=device
        )

    return (
        (payload(2 * INTERMEDIATE, HIDDEN), scales(2 * INTERMEDIATE, HIDDEN)),
        (payload(HIDDEN, INTERMEDIATE), scales(HIDDEN, INTERMEDIATE)),
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("requires a CUDA device")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{free_port()}",
        world_size=1,
        rank=0,
        device_id=device,
    )
    generator = torch.Generator(device="cpu").manual_seed(0)

    l1_checkpoint, l2_checkpoint = checkpoint_weights(device, generator)
    x = (torch.randn(NUM_TOKENS, HIDDEN, generator=generator) * 0.5).to(device)
    x = x.to(torch.float8_e4m3fn)
    topk_weights, topk_idx = torch.topk(
        torch.randn(NUM_TOKENS, NUM_EXPERTS, generator=generator), NUM_TOPK, dim=-1
    )
    topk_weights = torch.softmax(topk_weights, dim=-1).to(device)
    topk_idx = topk_idx.to(torch.int64).to(device)

    with flashinfer.GCMoEWorkspace(
        dist.group.WORLD,
        NUM_EXPERTS,
        CAPACITY,
        NUM_TOPK,
        HIDDEN,
        INTERMEDIATE,
    ) as workspace:
        l1_weights, l2_weights = flashinfer.transform_gc_moe_weights(
            l1_checkpoint, l2_checkpoint
        )
        workspace.x[:NUM_TOKENS].copy_(x)
        if workspace.x_sf.dtype == torch.float32:
            workspace.x_sf[:NUM_TOKENS].fill_(1.0)
        else:
            workspace.x_sf[:NUM_TOKENS].fill_(0x7F7F7F7F)
        workspace.topk_idx[:NUM_TOKENS].copy_(topk_idx)
        workspace.topk_weights[:NUM_TOKENS].copy_(topk_weights)

        y = torch.empty(NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        flashinfer.gc_moe(y, l1_weights, l2_weights, workspace)
        torch.cuda.synchronize()

        w1 = decode_fp4(l1_checkpoint[0])
        w2 = decode_fp4(l2_checkpoint[0])
        expected = torch.zeros(NUM_TOKENS, HIDDEN, device=device)
        for token in range(NUM_TOKENS):
            for slot in range(NUM_TOPK):
                expert = int(topk_idx[token, slot])
                linear = x[token].float() @ w1[expert].T
                gate, up = linear[:INTERMEDIATE], linear[INTERMEDIATE:]
                mid = gate * torch.sigmoid(gate) * up
                expected[token] += float(topk_weights[token, slot]) * (
                    mid @ w2[expert].T
                )
        cosine = torch.nn.functional.cosine_similarity(
            expected.flatten(), y.float().flatten(), dim=0
        )
        print(f"backend={workspace.backend} cosine={float(cosine):.6f}")
        print(f"max|y|={float(y.abs().max()):.5g}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
