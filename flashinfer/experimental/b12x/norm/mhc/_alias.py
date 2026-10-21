"""Byte-range alias checks for mHC views into caller-owned storage."""

import torch


def overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    if not left.numel() or not right.numel() or not torch._C._overlaps(left, right):
        return False

    def interval(tensor):
        start = tensor.storage_offset() * tensor.element_size()
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        return start, start + span * tensor.element_size()

    left_start, left_end = interval(left)
    right_start, right_end = interval(right)
    return left_start < right_end and right_start < left_end
