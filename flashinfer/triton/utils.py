from typing import List

import torch
from flashinfer.utils import get_compute_capability


def check_input(x: torch.Tensor):
    assert x.is_cuda, f"{str(x)} must be a CUDA Tensor"
    assert x.is_contiguous(), f"{str(x)} must be contiguous"


def check_dim(d, x: torch.Tensor):
    assert x.dim() == d, f"{str(x)} must be a {d}D tensor"


def check_shape(a: torch.Tensor, b: torch.Tensor):
    assert a.dim() == b.dim(), "tensors should have same dim"
    for i in range(a.dim()):
        assert a.size(i) == b.size(i), (
            f"tensors shape mismatch, {a.size()} and {b.size()}"
        )


def check_device(tensors: List[torch.Tensor], major: int = None, minor: int = None):
    device = tensors[0].device
    for t in tensors:
        assert t.device == device, (
            f"All tensors should be on the same device, but got {device} and {t.device}"
        )

    if major is not None or minor is not None:
        capability = get_compute_capability(device)
        if major is not None:
            assert capability[0] == major, (
                f"Device major should be {major}, but got {capability[0]}"
            )
        if minor is not None:
            assert capability[1] == minor, (
                f"Device minor should be {minor}, but got {capability[1]}"
            )
