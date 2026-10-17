# Copyright (c) 2025 by the b12x authors.
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

"""Torch and CUTLASS dtype conversion."""

import cutlass
import importlib.util
import torch


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def is_cute_dsl_available() -> bool:
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


def get_cutlass_dtype(dtype: str) -> cutlass.dtype:
    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
        "float8_e5m2": cutlass.Float8E5M2,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float8_e8m0fnu": cutlass.Float8E8M0FNU,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
        "float6_e3m2fn": cutlass.Float6E3M2FN,
        "float6_e2m3fn": cutlass.Float6E2M3FN,
    }
    try:
        return dtype_map[dtype]
    except KeyError as exc:
        raise KeyError(f"unsupported cutlass dtype string: {dtype!r}") from exc


def cutlass_to_torch_dtype(cutlass_dtype):
    """Return the Torch dtype used to store tensors for a CUTLASS element type.

    MX-FP6 element types (``Float6E3M2FN`` / ``Float6E2M3FN``) map to ``torch.uint8``
    because PyTorch has no packed FP6 view. Kernels receive the logical element type via
    compile-time ``_gptr`` / pointer element-type injection (same pattern as NVFP4/dlpack).
    """
    torch_dtype = getattr(torch, cutlass_dtype.__name__.lower(), None)

    torch_type_map = {
        cutlass.TFloat32: torch.float32,
        cutlass.Float32: torch.float32,
        cutlass.Float16: torch.float16,
        cutlass.BFloat16: torch.bfloat16,
        cutlass.Float8E5M2: torch.float8_e5m2,
        cutlass.Float8E4M3FN: torch.float8_e4m3fn,
        cutlass.Float8E8M0FNU: torch.float8_e8m0fnu,
        cutlass.Float8E4M3B11FNUZ: torch.float8_e4m3fnuz,
        cutlass.Float4E2M1FN: torch.float4_e2m1fn_x2,  # FP4 packed (2 values per byte)
        cutlass.Float6E3M2FN: torch.uint8,
        cutlass.Float6E2M3FN: torch.uint8,
    }
    if torch_dtype is None:
        torch_dtype = torch_type_map.get(cutlass_dtype)

    if torch_dtype is None:
        raise TypeError(f"{cutlass_dtype} is not supported by torch")
    return torch_dtype


_num_sm_cache: dict[int, int] = {}

# Cache for HardwareInfo - it's expensive to create on every call
_hardware_info_cache: "cutlass.utils.HardwareInfo | None" = None
