# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Run both KPage block sizes: python examples/vsa_kpage.py."""

import torch

from flashinfer import BlockSparseAttentionWrapper


def main():
    for block in (64, 128):
        seq, heads = 4 * block, 2
        q, k, v = [
            torch.randn(seq, heads, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        ]
        wrapper = BlockSparseAttentionWrapper(
            torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
            backend=f"vsa_sm100_blk{block}",
        )
        wrapper.plan(
            None,
            None,
            seq,
            seq,
            block,
            block,
            heads,
            heads,
            128,
            q_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            block_mask=torch.ones(heads, 4, 4, dtype=torch.bool, device="cuda"),
            kv_block_lens=torch.tensor(
                [block, 17, block, 7], dtype=torch.int32, device="cuda"
            ),
            kpage=True,
        )
        out, lse = wrapper.run(q, k, v, return_lse=True)
        torch.cuda.synchronize()
        print(f"blk{block}: output={tuple(out.shape)}, LSE={tuple(lse.shape)}")


if __name__ == "__main__":
    main()
