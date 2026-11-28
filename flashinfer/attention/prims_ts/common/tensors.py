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

"""Tensor, dtype and device checks shared by the PrimTS host frontends."""

from typing import Optional, Union

import torch

MAX_INT32 = 2**31 - 1
SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))


def attention_dtype_key(dtype: torch.dtype) -> str:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("attention-ts dtypes must be torch.dtype values")
    keys = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.uint8: "float4_e2m1fn",
    }
    try:
        return keys[dtype]
    except KeyError as error:
        raise NotImplementedError(
            "attention-ts decode supports torch.float16, torch.bfloat16, "
            "torch.float8_e4m3fn, and packed NVFP4 K/V in torch.uint8; "
            f"got {dtype}"
        ) from error


def compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= int(extent)
    return tuple(reversed(strides))


def cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def cutlass_dtype(dtype_key: str):
    """Return the cutlass numeric type named by a dtype key."""

    import cutlass

    return {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
        "int8": cutlass.Int8,
    }[dtype_key]


def resolve_cuda_device(
    device: Optional[Union[int, str, torch.device]],
) -> tuple[torch.device, int]:
    """Normalize a device specifier without checking CUDA support."""

    if device is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int) and not isinstance(device, bool):
        resolved = torch.device("cuda", device)
    else:
        try:
            resolved = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise TypeError("device must identify one CUDA device") from error
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
    # Preserve non-CUDA device types for the caller's explicit validation.
    device_index = (
        cuda_device_index(resolved)
        if resolved.type == "cuda"
        else (resolved.index or 0)
    )
    return torch.device(resolved.type, device_index), device_index


def validate_16byte_alignment(tensor: torch.Tensor, name: str) -> None:
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")


def validate_exact_compact_strides(
    tensor: torch.Tensor,
    name: str,
    layout: str,
) -> None:
    expected_strides = compact_strides(tuple(tensor.shape))
    if tensor.stride() != expected_strides:
        raise ValueError(
            f"{name} must have compact {layout} strides "
            f"{expected_strides}, but has {tensor.stride()}"
        )


def validate_runtime_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("attention-ts decode tensors must be CUDA tensors")
    device_index = device.index
    with torch.cuda.device(device_index):
        capability = torch.cuda.get_device_capability(device_index)
    if capability not in SUPPORTED_COMPUTE_CAPABILITIES:
        raise NotImplementedError(
            "attention-ts decode requires an SM100a/B200, SM103a/B300 or "
            "SM107a/Rubin GPU; "
            f"device cuda:{device_index} has compute capability {capability}"
        )
    # Rubin runs through the sm_100f family target; a CuTe DSL older than 4.8
    # cannot emit for it unless CUTE_DSL_ARCH=sm_100f is set before import.
    if capability == (10, 7):
        from ....cute_dsl.utils import require_cute_dsl_arch

        require_cute_dsl_arch(device_index)
