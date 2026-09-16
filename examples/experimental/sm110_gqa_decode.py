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

# Run the experimental exact-SM110 FP16 GQA decode kernel.

import argparse

import torch

from flashinfer import sm110_gqa_decode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=1024)
    args = parser.parse_args()

    q = torch.randn(args.batch, 32, 128, dtype=torch.float16, device="cuda")
    kv = torch.randn(
        args.batch,
        2,
        8,
        args.capacity,
        128,
        dtype=torch.float16,
        device="cuda",
    )
    sequence_lengths = torch.full(
        (args.batch,),
        args.capacity,
        dtype=torch.int32,
        device="cuda",
    )
    out = sm110_gqa_decode(q, kv, sequence_lengths)
    torch.cuda.synchronize()
    print(f"output shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}")


if __name__ == "__main__":
    main()
