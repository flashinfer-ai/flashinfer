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

# Run the experimental exact-SM100 BF16 union-GQA prefill kernel.

import argparse

import torch

from flashinfer import union_gqa_prefill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--topk-blocks", type=int, default=8)
    args = parser.parse_args()

    q = torch.randn(args.seq_len, 16, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(args.seq_len, 1, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    block_indices = torch.arange(args.topk_blocks, dtype=torch.int32, device="cuda")
    block_indices = (
        block_indices.view(1, 1, -1).expand(args.seq_len, 1, -1).contiguous()
    )
    cu_seqlens_q = torch.tensor([0, args.seq_len], dtype=torch.int32, device="cuda")

    out = union_gqa_prefill(
        q,
        k,
        v,
        block_indices,
        cu_seqlens_q,
        max_seqlen_q=args.seq_len,
    )
    torch.cuda.synchronize()
    print(f"output shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}")


if __name__ == "__main__":
    main()
