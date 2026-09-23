# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""torchrun example: 16 SM103a/152-SM GPUs in one NVL72, four ranks per node."""

import os

import torch
import torch.distributed as dist

from flashinfer.moe_ep.cake_w4a8_megamoe_ep16 import (
    CakeW4A8MegaMoeEp16,
    preprocess_cake_w4a8_megamoe_ep16_weights,
)
from flashinfer.moe_ep.weights import MoEWeightPack


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    rank = dist.get_rank()
    generator = torch.Generator(device="cuda").manual_seed(17 + rank)
    weights = preprocess_cake_w4a8_megamoe_ep16_weights(
        MoEWeightPack(
            torch.randint(
                256,
                (32, 10240, 1536),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            ),
            torch.randint(
                256,
                (32, 3072, 2560),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            ),
            torch.full((32, 10240, 96), 119, dtype=torch.uint8, device="cuda"),
            torch.full((32, 3072, 160), 119, dtype=torch.uint8, device="cuda"),
        )
    )
    ids = torch.randint(
        512, (16, 8), dtype=torch.int64, device="cuda", generator=generator
    )
    x = torch.randn(16, 3072, dtype=torch.bfloat16, device="cuda", generator=generator)
    router_weights = torch.randn(16, 8, device="cuda", generator=generator).softmax(-1)
    output = torch.empty_like(x)
    session = CakeW4A8MegaMoeEp16(weights, ids)
    session.forward(x, router_weights, out=output)
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    print(f"rank {rank}: output {tuple(output.shape)} {output.dtype}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
