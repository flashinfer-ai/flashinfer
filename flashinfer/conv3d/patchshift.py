"""
Copyright (c) 2026 by the PatchShift Conv3d contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.patchshift_conv3d import gen_patchshift_conv3d_module
from ..trace.templates.conv3d import patchshift_conv3d_trace
from ..utils import (
    backend_requirement,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)


@functools.cache
def get_patchshift_conv3d_module():
    return gen_patchshift_conv3d_module().build_and_load()


def _require_sm100a(tensor: torch.Tensor) -> None:
    if tensor.is_cuda and torch.cuda.get_device_capability(tensor.device) != (10, 0):
        raise ValueError("PatchShift Conv3d currently requires SM100a/B200")


def pack_patchshift_conv3d_weight(weight: torch.Tensor) -> torch.Tensor:
    """Prepack a BF16 ``[K, C, 3, 3, 3]`` weight for PatchShift Conv3d."""
    if weight.ndim != 5 or tuple(weight.shape[2:]) != (3, 3, 3):
        raise ValueError("weight must have shape [K, C, 3, 3, 3]")
    if not weight.is_cuda or weight.dtype is not torch.bfloat16:
        raise ValueError("weight must be a CUDA BF16 tensor")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0 or weight.shape[1] % 8 != 0:
        raise ValueError(
            "weight output channels must be positive and input channels must "
            "be positive and divisible by 8"
        )
    _require_sm100a(weight)
    module = get_patchshift_conv3d_module()
    numel = module.packed_weight_numel(weight.shape[1], weight.shape[0])
    packed_weight = torch.empty(numel, dtype=weight.dtype, device=weight.device)
    module.pack_weight(weight, packed_weight)
    return packed_weight


def prepare_patchshift_conv3d(
    input: torch.Tensor, packed_weight: torch.Tensor, out_channels: int
) -> torch.Tensor:
    """Prepare pointer-dependent TMA descriptors outside CUDA graph capture."""
    if not input.is_cuda or input.dtype is not torch.bfloat16 or input.ndim != 5:
        raise ValueError("input must be a contiguous CUDA BF16 NDHWC tensor")
    if not input.is_contiguous() or any(size <= 0 for size in input.shape):
        raise ValueError("input must have positive extents and be contiguous NDHWC")
    if input.shape[-1] % 8 != 0:
        raise ValueError("input channels must be divisible by 8")
    if out_channels <= 0:
        raise ValueError("out_channels must be positive")
    if (
        not packed_weight.is_cuda
        or packed_weight.device != input.device
        or packed_weight.dtype != input.dtype
        or packed_weight.ndim != 1
        or not packed_weight.is_contiguous()
    ):
        raise ValueError(
            "packed_weight must be a contiguous 1D BF16 tensor on the input device"
        )
    _require_sm100a(input)
    module = get_patchshift_conv3d_module()
    workspace = torch.empty(
        module.descriptor_workspace_size(), dtype=torch.uint8, device=input.device
    )
    module.prepare(workspace, input, packed_weight, out_channels)
    return workspace


@supported_compute_capability([100])
def _check_patchshift_conv3d(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    workspace: torch.Tensor,
    out_channels: int,
    out: Optional[torch.Tensor] = None,
) -> bool:
    if not input.is_cuda or input.dtype is not torch.bfloat16 or input.ndim != 5:
        raise ValueError("input must be a contiguous CUDA BF16 NDHWC tensor")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous NDHWC")
    if input.shape[-1] % 8 != 0:
        raise ValueError("input channels must be divisible by 8")
    if any(size <= 0 for size in input.shape):
        raise ValueError("input extents must be positive")
    if out_channels <= 0:
        raise ValueError("out_channels must be positive")
    if packed_weight.device != input.device or packed_weight.dtype != input.dtype:
        raise ValueError("packed_weight must be BF16 on the input device")
    if packed_weight.ndim != 1 or not packed_weight.is_contiguous():
        raise ValueError("packed_weight must be a contiguous 1D tensor")
    if workspace.device != input.device or workspace.dtype != torch.uint8:
        raise ValueError("workspace must be uint8 on the input device")
    if workspace.ndim != 1 or not workspace.is_contiguous():
        raise ValueError("workspace must be a contiguous 1D tensor")
    if out is not None:
        expected = (*input.shape[:-1], out_channels)
        if tuple(out.shape) != expected:
            raise ValueError(f"out must have shape {expected}")
        if out.device != input.device or out.dtype != input.dtype:
            raise ValueError("out must be BF16 on the input device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous NDHWK")
        if out.data_ptr() == input.data_ptr():
            raise ValueError("out must not alias input")
    return True


@flashinfer_api(trace=patchshift_conv3d_trace)
@backend_requirement(backend_checks={}, common_check=_check_patchshift_conv3d)
def patchshift_conv3d(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    workspace: torch.Tensor,
    out_channels: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run fixed BF16 3x3x3, pad-1, stride-1 Conv3d on SM100a.

    ``input`` and the returned tensor use NDHWC layout. Call
    :func:`pack_patchshift_conv3d_weight` once per weight and
    :func:`prepare_patchshift_conv3d` once per input shape before capture.
    """
    if out is None:
        out = torch.empty(
            (*input.shape[:-1], out_channels),
            dtype=input.dtype,
            device=input.device,
        )
    _patchshift_conv3d_impl(out, input, packed_weight, workspace)
    return out


@register_custom_op("flashinfer::patchshift_conv3d", mutates_args=("out", "workspace"))
def _patchshift_conv3d_impl(
    out: torch.Tensor,
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    get_patchshift_conv3d_module().run(workspace, input, packed_weight, out)


@register_fake_op("flashinfer::patchshift_conv3d")
def _patchshift_conv3d_impl_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    pass
