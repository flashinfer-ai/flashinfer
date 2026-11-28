# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Caller-owned byte workspaces carved into typed, aligned sections."""

import math
from dataclasses import dataclass

import torch

WORKSPACE_ALIGNMENT = 256
WORKSPACE_DTYPES = (torch.int8, torch.uint8)


@dataclass(frozen=True)
class WorkspaceSection:
    """One typed tensor view owned by a caller-provided byte workspace."""

    byte_offset: int
    byte_size: int
    shape: tuple[int, ...]
    dtype: torch.dtype


def align_up(value: int, alignment: int = WORKSPACE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def append_workspace_section(
    byte_end: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    reserve_elems: int = 0,
) -> tuple[WorkspaceSection, int]:
    byte_offset = align_up(byte_end)
    byte_size = math.prod(shape) * dtype_itemsize(dtype)
    reserved = max(byte_size, reserve_elems * dtype_itemsize(dtype))
    return (
        WorkspaceSection(byte_offset, byte_size, shape, dtype),
        byte_offset + reserved,
    )


def dtype_itemsize(dtype: torch.dtype) -> int:
    itemsize = {
        torch.int8: 1,
        torch.uint8: 1,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int32: 4,
        torch.float32: 4,
    }
    try:
        return itemsize[dtype]
    except KeyError as error:
        raise TypeError(f"unsupported workspace section dtype {dtype}") from error


def validate_workspace_buffer(
    workspace_buffer: torch.Tensor,
    *,
    device: torch.device,
    required_bytes: int,
) -> None:
    if not isinstance(workspace_buffer, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if workspace_buffer.dtype not in WORKSPACE_DTYPES:
        raise TypeError("workspace_buffer must have dtype torch.int8 or torch.uint8")
    if workspace_buffer.device != device:
        raise ValueError(
            f"workspace_buffer must be on {device}, got {workspace_buffer.device}"
        )
    if not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    if workspace_buffer.data_ptr() % 32 != 0:
        raise ValueError("workspace_buffer data pointer must be 32-byte aligned")
    available_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if available_bytes < required_bytes:
        raise ValueError(
            "workspace_buffer is too small: requires at least "
            f"{required_bytes} bytes, got {available_bytes}"
        )


def workspace_section_view(
    workspace_buffer: torch.Tensor, section: WorkspaceSection
) -> torch.Tensor:
    workspace_bytes = workspace_buffer.reshape(-1).view(torch.uint8)
    section_bytes = workspace_bytes[
        section.byte_offset : section.byte_offset + section.byte_size
    ]
    return section_bytes.view(section.dtype).view(section.shape)
