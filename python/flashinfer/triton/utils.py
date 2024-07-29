from typing import List

import torch


def check_input(x: torch.Tensor):
    assert x.is_cuda, f"{str(x)} must be a CUDA Tensor"
    assert x.is_contiguous(), f"{str(x)} must be contiguous"


def check_dim(d, x: torch.Tensor):
    assert x.dim() == d, f"{str(x)} must be a {d}D tensor"


def check_shape(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == b.dim(), f"tensors should have same dim"
    for i in range(a.dim()):
        assert a.size(i) == b.size(
            i
        ), f"tensors shape mismatch, {a.size()} and {b.size()}"


def check_device(tensors: List[torch.Tensor]):
    device = tensors[0].device
    for t in tensors:
        assert (
            t.device == device
        ), f"All tensors should be on the same device, but got {device} and {t.device}"
