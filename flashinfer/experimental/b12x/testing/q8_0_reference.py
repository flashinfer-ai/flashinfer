"""Independent GGML Q8_0 raw-block decoder for correctness qualification."""

import torch


def dequantize_blocks(blocks: torch.Tensor) -> torch.Tensor:
    if blocks.dtype != torch.uint8 or blocks.ndim < 2 or blocks.shape[-1] != 34:
        raise ValueError("expected uint8 Q8_0 blocks ending in 34 bytes")
    raw = blocks.cpu().contiguous()
    scale = raw[..., :2].contiguous().view(torch.float16).float()
    values = raw[..., 2:].contiguous().view(torch.int8).float()
    return (scale * values).flatten(-2)
