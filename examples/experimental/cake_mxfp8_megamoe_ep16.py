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

# Run with: torchrun --nproc-per-node=16 <this-file> --tokens 16 --backend cute_dsl.

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from datetime import timedelta

import torch
import torch.distributed as dist
from flashinfer.moe_ep import (
    CakeMxfp8MegaMoeEp16,
    preprocess_cake_mxfp8_megamoe_ep16_weights,
)


def _select_device_index(environ: Mapping[str, str], visible_devices: int) -> int:
    """Validate torchrun metadata and select a process-local CUDA ordinal."""
    try:
        world_size = int(environ["WORLD_SIZE"])
        local_world_size = int(environ["LOCAL_WORLD_SIZE"])
        local_rank = int(environ["LOCAL_RANK"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Launch with torchrun and integer WORLD_SIZE, LOCAL_WORLD_SIZE, LOCAL_RANK"
        ) from error
    if world_size != 16:
        raise ValueError("This example requires WORLD_SIZE=16")
    if not 1 <= local_world_size <= world_size:
        raise ValueError("LOCAL_WORLD_SIZE must be between 1 and WORLD_SIZE")
    if not 0 <= local_rank < local_world_size:
        raise ValueError("LOCAL_RANK must be between 0 and LOCAL_WORLD_SIZE - 1")
    if visible_devices < 1:
        raise ValueError("Each rank must have a visible CUDA device")
    # A launcher may give each process a private one-device CUDA visibility mask.
    if visible_devices == 1:
        return 0
    if local_world_size > visible_devices:
        raise ValueError("LOCAL_WORLD_SIZE exceeds the visible CUDA device count")
    return local_rank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--backend", choices=("cuda", "cute_dsl"), default="cuda")
    args = parser.parse_args()

    device_index = _select_device_index(os.environ, torch.cuda.device_count())
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise RuntimeError("This example requires SM103 GPUs")
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=10))
    try:
        if dist.get_world_size() != 16:
            raise RuntimeError("This example requires a 16-rank process group")
        rank = dist.get_rank()
        torch.manual_seed(1000 + rank)
        w13 = torch.randn((32, 10240, 3072), dtype=torch.bfloat16, device=device)
        w2 = torch.randn((32, 3072, 5120), dtype=torch.bfloat16, device=device)
        weights = preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)

        local_tokens = torch.arange(args.tokens, dtype=torch.int64, device=device)
        global_tokens = rank * args.tokens + local_tokens
        route_slots = (
            global_tokens[:, None] * 8 + torch.arange(8, device=device)[None, :]
        )
        # The affine permutation spreads routes across owners while giving every
        # expert the same global load for each supported token count.
        topk_ids = (route_slots * 73 + 19) % 512
        topk_weights = torch.full(
            (args.tokens, 8),
            1.0 / 8.0,
            dtype=torch.float32,
            device=device,
        )
        hidden_states = torch.randn(
            (args.tokens, 3072),
            dtype=torch.bfloat16,
            device=device,
        )

        session = CakeMxfp8MegaMoeEp16(weights, topk_ids, backend=args.backend)
        output = session.run(
            hidden_states,
            topk_ids,
            topk_weights,
            out=session.workspace_output,
        )
        torch.cuda.synchronize()
        if rank == 0:
            print(f"backend={args.backend}, output shape={tuple(output.shape)}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
