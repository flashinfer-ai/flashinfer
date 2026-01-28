"""
Copyright (c) 2025 by FlashInfer team.

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

import ctypes
import functools
import importlib.util
from typing import Union, Tuple

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import torch
from cutlass._mlir import ir
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type


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
    }
    return dtype_map[dtype]


def cutlass_to_torch_dtype(cutlass_dtype):
    """
    Return the corresponding torch.dtype per the given DSL type
    """
    torch_dtype = getattr(torch, cutlass_dtype.__name__.lower(), None)

    torch_type_map = {
        cutlass.TFloat32: torch.float32,
        cutlass.Float32: torch.float32,
        cutlass.Float16: torch.float16,
        cutlass.BFloat16: torch.bfloat16,
        cutlass.Float8E5M2: torch.float8_e5m2,
        cutlass.Float8E4M3FN: torch.float8_e4m3fn,
        cutlass.Float8E4M3B11FNUZ: torch.float8_e4m3fnuz,
        cutlass.Float4E2M1FN: torch.float4_e2m1fn_x2,  # FP4 packed (2 values per byte)
    }
    if torch_dtype is None:
        torch_dtype = torch_type_map.get(cutlass_dtype)

    if torch_dtype is None:
        raise TypeError(f"{cutlass_dtype} is not supported by torch")
    return torch_dtype


@functools.cache
def get_num_sm(device: torch.device) -> int:
    # get the compute capability of the device, which would be cached
    return torch.cuda.get_device_properties(device).multi_processor_count


# Cache for HardwareInfo - it's expensive to create on every call
_hardware_info_cache: "cutlass.utils.HardwareInfo | None" = None


def get_hardware_info() -> "cutlass.utils.HardwareInfo":
    """Get cached HardwareInfo singleton.

    HardwareInfo queries CUDA device capabilities, which can be expensive.
    This function caches the singleton to avoid repeated queries.
    """
    global _hardware_info_cache
    if _hardware_info_cache is None:
        _hardware_info_cache = cutlass.utils.HardwareInfo()
    return _hardware_info_cache


@functools.cache
def get_max_active_clusters(cluster_size: int) -> int:
    """Get max active clusters for a given cluster size (cached).

    Args:
        cluster_size: Product of cluster_shape_mn dimensions.

    Returns:
        Maximum number of active clusters supported by hardware.
    """
    return get_hardware_info().get_max_active_clusters(cluster_size)


# WAR for CuTeDSL make_ptr implementation for flashinfer
class _Pointer(Pointer):
    """Runtime representation of a pointer that can inter-operate with
    various data structures, including numpy arrays and device memory.

    :param pointer: The pointer to the data
    :type pointer: int or pointer-like object
    :param dtype: Data type of the elements pointed to
    :type dtype: Type
    :param mem_space: Memory space where the pointer resides, defaults generic
    :type mem_space: _cute_ir.AddressSpace, optional
    :param assumed_align: Alignment of input pointer in bytes, defaults None
    :type assumed_align: int, optional

    :ivar _pointer: The underlying pointer
    :ivar _dtype: Data type of the elements
    :ivar _addr_space: Memory space of the pointer
    :ivar _assumed_align: Alignment of the pointer in bytes
    :ivar _desc: C-type descriptor for the pointer
    :ivar _c_pointer: C-compatible pointer representation
    """

    def __init__(
        self,
        pointer,
        dtype,
        mem_space: _cute_ir.AddressSpace = _cute_ir.AddressSpace.generic,
        assumed_align=None,
    ):
        self._pointer = pointer
        self._dtype = dtype
        self._addr_space = mem_space

        if assumed_align is None:
            self._assumed_align = dtype.width // 8
        else:
            self._assumed_align = assumed_align

        self._desc = None
        self._c_pointer = None
        assert int(self._pointer) % self._assumed_align == 0, (
            f"pointer must be {self._assumed_align} bytes aligned"
        )

    def size_in_bytes(self) -> int:
        return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))

    def __get_mlir_types__(self):
        return [self.mlir_type]

    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer))
            self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]

    def __new_from_mlir_values__(self, values):
        assert len(values) == 1
        return values[0]

    # Move mlir Type out of __init__ to decouple with mlir Context
    @property
    def mlir_type(self) -> ir.Type:
        return _cute_ir.PtrType.get(
            self._dtype.mlir_type, self._addr_space, self._assumed_align
        )

    @property
    def dtype(self) -> Type[Numeric]:
        return self._dtype

    @property
    def memspace(self):
        return self._addr_space

    def align(self, min_align: int, *, loc=None, ip=None) -> Pointer:
        raise NotImplementedError("align is not supported in runtime")

    def verify(self, expected_py_type):
        # if expected_py_type is Pointer:
        #     return True
        # elif isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer:
        #     return True
        if expected_py_type is Pointer or (
            isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer
        ):
            return True

        return False

    def __str__(self) -> str:
        return f"Ptr<0x{int(self._pointer):016x}@{self._addr_space}>"

    def __repr__(self):
        return self.__str__()


def make_ptr(
    dtype: Type[Numeric],
    value: Union[int, ctypes._Pointer],
    mem_space: AddressSpace = AddressSpace.generic,
    assumed_align=None,
) -> Pointer:
    """Create a pointer from a memory address

    :param dtype: Data type of the pointer elements
    :type dtype: Type[Numeric]
    :param value: Memory address as integer or ctypes pointer
    :type value: Union[int, ctypes._Pointer]
    :param mem_space: Memory address space, defaults to AddressSpace.generic
    :type mem_space: AddressSpace, optional
    :param assumed_align: Alignment in bytes, defaults to None
    :type assumed_align: int, optional
    :return: A pointer object
    :rtype: Pointer

    .. code-block:: python

        import numpy as np
        import ctypes

        from cutlass import Float32
        from cutlass.cute.runtime import make_ptr

        # Create a numpy array
        a = np.random.randn(16, 32).astype(np.float32)

        # Get pointer address as integer
        ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Create pointer from address
        y = make_ptr(cutlass.Float32, ptr_address)
    """
    # check if value is int or ctypes.POINTER
    if isinstance(value, int):
        address_value = value
    elif isinstance(value, ctypes._Pointer):
        # get address value
        address_value = ctypes.cast(value, ctypes.c_void_p).value
        assert address_value is not None, "Pointer address is None"
    else:
        raise TypeError(
            f"Expect int or ctypes.POINTER for value but got {type(value)=}"
        )

    return _Pointer(address_value, dtype, mem_space, assumed_align=assumed_align)


def convert_sf_to_mma_layout(
    sf: torch.Tensor,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """Convert scale factors from swizzled 2D layout to 6D MMA-compatible layout.

    This function converts scale factors produced by `fp4_quantize(..., is_sf_swizzled_layout=True)`
    to the 6D layout expected by CuteDSL grouped GEMM kernels.

    The swizzled scale factors from `fp4_quantize` have shape `(M, K/sf_vec_size)` but are
    stored in a swizzled pattern internally. This function reshapes them to the explicit
    6D MMA-compatible layout: `(32, 4, m_tiles, 4, k_tiles, num_groups)` with the
    physical storage order `(num_groups, m_tiles, k_tiles, 32, 4, 4)`.

    Layout mapping (from linear (m, k) position):
        - m_tile = m // 128
        - outer_m = m % 32
        - inner_m = (m % 128) // 32
        - k_tile = k // 4
        - inner_k = k % 4
        - 6D position: (outer_m, inner_m, m_tile, inner_k, k_tile, group)

    Args:
        sf: Scale factor tensor from `fp4_quantize(..., is_sf_swizzled_layout=True)`.
            Shape: `(M, K/sf_vec_size)` or `(num_groups * M, K/sf_vec_size)`.
        m: The M dimension (rows) of the original matrix before quantization.
        k: The K dimension (columns) of the original matrix before quantization.
        num_groups: Number of groups (e.g., experts). Default: 1.
        sf_vec_size: Scale factor vector size. Default: 16.

    Returns:
        Scale factors in 6D MMA layout: `(32, 4, m_tiles, 4, k_tiles, num_groups)`.
        This is a strided view (not contiguous) with physical storage order
        `(num_groups, m_tiles, k_tiles, 32, 4, 4)`.

    Example:
        >>> # Quantize weight tensor
        >>> w_q, w_sf = fp4_quantize(weight, global_scale=gs, is_sf_swizzled_layout=True)
        >>> # Convert scale factors to MMA layout
        >>> w_sf_mma = convert_sf_to_mma_layout(w_sf, m=weight.shape[0], k=weight.shape[1])

    Note:
        - The input `sf` must be produced with `is_sf_swizzled_layout=True`.
        - M and K dimensions must be multiples of 128 and 64 respectively for proper alignment.
        - For grouped tensors (e.g., expert weights), reshape to `(num_groups * M, K)`
          before quantization, then use this function with the appropriate `num_groups`.
        - The returned tensor is a strided view, NOT contiguous. This is intentional as
          the CuteDSL kernel expects the specific physical memory layout.
    """
    sf_k = ceil_div(k, sf_vec_size)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(sf_k, 4)

    # Verify input shape
    expected_elements = num_groups * m_tiles * k_tiles * 32 * 4 * 4
    actual_elements = sf.numel()
    if actual_elements != expected_elements:
        raise ValueError(
            f"Scale factor tensor has {actual_elements} elements, "
            f"expected {expected_elements} for m={m}, k={k}, num_groups={num_groups}"
        )

    # Reshape from flat 2D to 6D physical storage order
    # Physical storage: (num_groups, m_tiles, k_tiles, 32, 4, 4)
    sf_6d = sf.view(num_groups, m_tiles, k_tiles, 32, 4, 4)

    # Permute to MMA logical order: (32, 4, m_tiles, 4, k_tiles, num_groups)
    # This creates a strided view (non-contiguous), which is what the kernel expects
    sf_6d = sf_6d.permute(3, 4, 1, 5, 2, 0)

    return sf_6d  # Return strided view, NOT contiguous


def convert_sf_from_mma_layout(
    sf_6d: torch.Tensor,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """Convert scale factors from 6D MMA layout back to 2D swizzled layout.

    This is the inverse of `convert_sf_to_mma_layout`.

    Args:
        sf_6d: Scale factors in 6D MMA layout: `(32, 4, m_tiles, 4, k_tiles, num_groups)`.
               Can be either a strided view or contiguous.
        m: The M dimension (rows) of the original matrix.
        k: The K dimension (columns) of the original matrix.
        num_groups: Number of groups. Default: 1.
        sf_vec_size: Scale factor vector size. Default: 16.

    Returns:
        Scale factors in 2D swizzled layout: `(num_groups * M_padded, K_padded/sf_vec_size)`.
    """
    sf_k = ceil_div(k, sf_vec_size)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(sf_k, 4)

    # Permute from MMA logical order back to storage order
    # From: (32, 4, m_tiles, 4, k_tiles, num_groups)
    # To: (num_groups, m_tiles, k_tiles, 32, 4, 4)
    sf_storage = sf_6d.permute(5, 2, 4, 0, 1, 3).contiguous()

    # Reshape to 2D
    padded_m = m_tiles * 128
    padded_sf_k = k_tiles * 4
    sf_2d = sf_storage.reshape(num_groups * padded_m, padded_sf_k)

    return sf_2d


def get_mma_sf_shape(
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = 16,
) -> Tuple[int, int, int, int, int, int]:
    """Get the 6D MMA-compatible scale factor shape.

    Args:
        m: The M dimension (rows) of the matrix.
        k: The K dimension (columns) of the matrix.
        num_groups: Number of groups. Default: 1.
        sf_vec_size: Scale factor vector size. Default: 16.

    Returns:
        Shape tuple: (32, 4, m_tiles, 4, k_tiles, num_groups)
    """
    sf_k = ceil_div(k, sf_vec_size)
    m_tiles = ceil_div(m, 128)
    k_tiles = ceil_div(sf_k, 4)
    return (32, 4, m_tiles, 4, k_tiles, num_groups)
