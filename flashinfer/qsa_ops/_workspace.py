"""Carving caller-owned buffers into views, and the tensor checks made before a launch."""

import torch

from ..topk import WORKSPACE_ALIGNMENT
from ..utils import round_up


def walk():
    """A cursor that hands out aligned spans and remembers where it got to."""
    offset = 0

    def take(nbytes):
        nonlocal offset
        begin = offset
        offset += round_up(nbytes, WORKSPACE_ALIGNMENT)
        return (begin, nbytes)

    def total():
        return offset

    return take, total


def check_buffer(buffer: torch.Tensor, what: str) -> None:
    if buffer.dtype != torch.uint8:
        raise ValueError(f"{what} is raw bytes, got {buffer.dtype}")
    if not buffer.is_cuda:
        raise ValueError(f"{what} has to be on a CUDA device")
    if not buffer.is_contiguous():
        raise ValueError(f"{what} has to be contiguous")
    if buffer.data_ptr() % WORKSPACE_ALIGNMENT:
        raise ValueError(f"{what} has to be {WORKSPACE_ALIGNMENT}-byte aligned")


def cut(buffer: torch.Tensor, span, dtype: torch.dtype, shape):
    begin, size = span
    view = buffer[begin : begin + size].view(dtype)
    return view if shape is None else view.view(shape)


def check_tensor(
    tensor,
    name,
    *,
    shape=None,
    dtype=None,
    device=None,
    contiguous=False,
    innermost=False,
):
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must be {tuple(shape)}, got {tuple(tensor.shape)}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if innermost and tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} must be contiguous in its innermost dimension, got stride {tensor.stride(-1)}"
        )
