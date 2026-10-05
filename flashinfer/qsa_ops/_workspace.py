"""Carving one caller-owned byte buffer into the views a QSA object runs from."""

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
