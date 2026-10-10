"""Minimal adaptive sparse block-mask example."""

import torch

from flashinfer.sparse import adaptive_sparse_block_mask


def main() -> None:
    torch.manual_seed(0)
    block_size = 128
    blocks = 192
    scores = torch.randn(1, 8, blocks, blocks, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([blocks * block_size], device="cuda", dtype=torch.int32)
    mask = adaptive_sparse_block_mask(
        scores,
        lengths,
        lengths,
        lengths,
        block_size=block_size,
        alpha=0.5,
    )
    print(mask.shape, mask.dtype, int(mask.sum()))


if __name__ == "__main__":
    main()
