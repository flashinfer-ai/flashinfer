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

# Run with: torchrun --nproc-per-node=16 <this-file> --tokens 16.

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    CakeMxfp8MegaMoeEp16,
    preprocess_cake_mxfp8_megamoe_ep16_weights,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(16, 32, 64), default=16)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1000 + rank)
    w13 = torch.randn((32, 10240, 3072), dtype=torch.bfloat16, device=device)
    w2 = torch.randn((32, 3072, 5120), dtype=torch.bfloat16, device=device)
    weights = preprocess_cake_mxfp8_megamoe_ep16_weights(w13, w2)

    local_tokens = torch.arange(args.tokens, dtype=torch.int64, device=device)
    global_tokens = rank * args.tokens + local_tokens
    owners = global_tokens % 16
    groups = (global_tokens // 16) % 4
    first_experts = owners * 32 + groups * 8
    topk_ids = first_experts[:, None] + torch.arange(8, device=device)[None, :]
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

    session = CakeMxfp8MegaMoeEp16(weights, topk_ids)
    output = session.run(
        hidden_states,
        topk_ids,
        topk_weights,
        out=session.workspace_output,
    )
    torch.cuda.synchronize()
    if rank == 0:
        print(output.shape)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
