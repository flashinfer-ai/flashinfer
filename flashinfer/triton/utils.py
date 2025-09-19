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


def check_device(tensors: List[torch.Tensor], major:List[int] = [], minor:List[int] = []):
    device = tensors[0].device 
    for t in tensors:
        assert t.device == device, (
            f"All tensors should be on the same device, but got {device} and {t.device}"
        )

    if len(major) > 0:
        assert get_compute_capability(device)[0] in major, f"Device major should be {major}, but got {get_compute_capability(device)[0]}"
    if len(minor) > 0:
        assert get_compute_capability(device)[1] in minor, f"Device minor should be {minor}, but got {get_compute_capability(device)[1]}"