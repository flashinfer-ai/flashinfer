from typing import List

import torch
from flashinfer.utils import get_compute_capability, GPUArchitectureError


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


def check_device(
    tensors: List[torch.Tensor], major: List[int] = None, minor: List[int] = None
):
    device = tensors[0].device
    for t in tensors:
        assert t.device == device, (
            f"All tensors should be on the same device, but got {device} and {t.device}"
        )
    if major is not None or minor is not None:
        actual_major, actual_minor = get_compute_capability(device)
        if major is not None and actual_major not in major:
            raise GPUArchitectureError(
                f"Device major should be in {major}, but got {actual_major}"
            )
        if minor is not None and actual_minor not in minor:
            raise GPUArchitectureError(
                f"Device minor should be in {minor}, but got {actual_minor}"
            )
