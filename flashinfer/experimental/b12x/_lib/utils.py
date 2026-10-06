"""
Copyright (c) 2025 by the b12x authors.

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

import cuda.bindings.driver as cuda
import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.typing import AddressSpace, Numeric, Pointer, Type


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def is_cute_dsl_available() -> bool:
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


# MX-FP6 (W6A6/W6A8) uses sf_vec_size=32 and m16n8k32 MMA (see b12x._lib.fp6).
MXFP6_SF_VEC_SIZE = 32
MXFP6_MMA_K = 32
MXFP6_SF_DTYPE_STR = "float8_e8m0fnu"
MXFP6_AB_DTYPE_STRINGS = frozenset({"float6_e3m2fn", "float6_e2m3fn"})


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


def is_mxfp6_ab_dtype(ab_dtype) -> bool:
    """True when ``ab_dtype`` is an MX-FP6 operand element type."""
    return ab_dtype in (cutlass.Float6E3M2FN, cutlass.Float6E2M3FN)


def is_mxfp6_ab_dtype_string(dtype: str) -> bool:
    return dtype in MXFP6_AB_DTYPE_STRINGS


def mxfp6_tile_k(sf_vec_size: int = MXFP6_SF_VEC_SIZE) -> int:
    """K tile size for one MX-FP6 pipeline stage (four ``m16n8k32`` MMA slices)."""
    if sf_vec_size != MXFP6_SF_VEC_SIZE:
        raise ValueError(f"MX-FP6 expects sf_vec_size={MXFP6_SF_VEC_SIZE}, got {sf_vec_size}")
    return sf_vec_size * 4


def mxfp6_num_k_blocks(tile_k: int, mma_k: int = MXFP6_MMA_K) -> int:
    """Number of ``m16n8k32`` MMA K-blocks covered by ``tile_k``."""
    if tile_k % mma_k != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by mma_k={mma_k}")
    return tile_k // mma_k


def mxfp6_packed_k_bytes(k: int) -> int:
    """Packed storage bytes along K (4 FP6 values per 3 bytes)."""
    if k % 4 != 0:
        raise ValueError(f"k must be divisible by 4 for MX-FP6 packing, got {k}")
    return (3 * k) // 4


def mxfp6_logical_k_from_packed_bytes(packed_k_bytes: int) -> int:
    """Logical K element count from packed byte width along K."""
    if packed_k_bytes % 3 != 0:
        raise ValueError(
            f"packed_k_bytes must be divisible by 3, got {packed_k_bytes}"
        )
    return (packed_k_bytes * 4) // 3


def mxfp6_tile_shape_mnk(
    tile_m: int,
    tile_n: int,
    sf_vec_size: int = MXFP6_SF_VEC_SIZE,
) -> Tuple[int, int, int]:
    """Return ``(tile_m, tile_n, tile_k)`` for MX-FP6 block-scaled kernels."""
    return tile_m, tile_n, mxfp6_tile_k(sf_vec_size)


def verify_mxfp6_smem_tile_k(
    tile_k: int,
    sf_vec_size: int = MXFP6_SF_VEC_SIZE,
) -> int:
    """Check ``tile_k`` satisfies ``sm120_make_smem_layout_sfa/sfb`` divisibility for MX-FP6.

    Returns ``mma_nsf = tile_k // sf_vec_size`` (4 when ``tile_k=128``).
    """
    if sf_vec_size != MXFP6_SF_VEC_SIZE:
        raise ValueError(f"MX-FP6 expects sf_vec_size={MXFP6_SF_VEC_SIZE}, got {sf_vec_size}")
    if tile_k % sf_vec_size != 0:
        raise ValueError(f"tile_k={tile_k} must be divisible by sf_vec_size={sf_vec_size}")
    mma_nsf = tile_k // sf_vec_size
    blk_sf = 4
    if tile_k % (blk_sf * mma_nsf) != 0:
        raise ValueError(
            f"tile_k={tile_k} must be divisible by blk_sf*mma_nsf={blk_sf * mma_nsf} "
            f"for sm120 SF smem (mma_nsf={mma_nsf})"
        )
    return mma_nsf


_num_sm_cache: dict[int, int] = {}


def get_num_sm(device: torch.device) -> int:
    """Return the device SM count without exposing lru_cache to Dynamo."""
    if device.type != "cuda":
        return 1
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    sm_count = _num_sm_cache.get(index)
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(index).multi_processor_count
        _num_sm_cache[index] = sm_count
    return sm_count


@torch._dynamo.disable
def current_cuda_stream() -> cuda.CUstream:
    """Return the current Torch CUDA stream as a CUDA driver stream handle."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def cuda_stream_to_int(stream: object | None) -> int | None:
    """Return a raw CUDA stream handle, preserving None as a caller fallback."""
    if stream is None:
        return None
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is not None:
        return int(cuda_stream)
    return int(stream)


def cuda_stream_from_int_or_current(stream_int: int | None) -> cuda.CUstream:
    """Use a supplied raw CUDA stream handle, or fall back to Torch's current stream."""
    if stream_int is None:
        return current_cuda_stream()
    return cuda.CUstream(int(stream_int))


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
    if cluster_size == 1:
        # A one-CTA cluster is bounded exactly by the SM count. Avoid the
        # CUTLASS occupancy probe, which requires a live CUDA context only to
        # recover that same value.
        return get_num_sm(torch.device("cuda"))
    return get_hardware_info().get_max_active_clusters(cluster_size)


# Compatibility wrapper around CuTe DSL runtime pointers.
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
        return expected_py_type is Pointer or (
            isinstance(expected_py_type, ir.Value) and expected_py_type.ty is Pointer
        )

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (self._dtype, self._addr_space, self._assumed_align)

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


@dsl_user_op
def sm120_make_smem_layout_sfa(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFA based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    blk_mn = 128
    blk_sf = min(4, tile_shape_mnk[2] // sf_vec_size)
    # The global scale atom always has four K groups.  A BK64 stage consumes
    # two groups but keeps the same row pitch inside shared memory.
    blk_elems = blk_mn * 4
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    assert tile_shape_mnk[0] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[0] must be divisible by 16"
    )

    # Scale-factor tiles are quantized in 128-row blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfa_tile_m = max(blk_mn, ceil_div(tile_shape_mnk[0], blk_mn) * blk_mn)

    sSFA_shapeM = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sSF_strideM = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * blk_elems,
    )

    sSFA_shape = (sSFA_shapeM, sSFA_shapeK)
    sSFA_stride = (sSF_strideM, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    sfa_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfa_smem_layout_staged


@dsl_user_op
def sm120_make_smem_layout_sfb(
    tiled_mma: cute.TiledMma,
    tile_shape_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    """
    Make smem layout for SFB based on:
    1. BlockScaledBasicChunk
    2. MMA tiler shape
    3. Scale factor vector size
    4. Number of stages

    :param tiled_mma: The tiled MMA
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The mma tiler shape
    :type mma_tiler_mnk: cute.Tile
    :param sf_vec_size: The scale factor vector size
    :type sf_vec_size: int
    :param num_stages: The number of stages
    :type num_stages: int

    :return: Smem layout for SFA
    :rtype: cute.Layout
    """

    blk_mn = 128
    blk_sf = min(4, tile_shape_mnk[2] // sf_vec_size)
    # Keep the four-group atom pitch when a BK64 stage exposes two groups.
    blk_elems = blk_mn * 4

    assert sf_vec_size == 16 or sf_vec_size == 32, "sf_vec_size must be 16 or 32"

    assert tile_shape_mnk[1] % (blk_mn // 8) == 0, (
        "tile_shape_mnk[1] must be divisible by 16"
    )

    assert tile_shape_mnk[2] % sf_vec_size == 0, (
        "tile_shape_mnk[2] must be divisible by sf_vec_size"
    )

    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size

    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)

    # Scale-factor tiles are quantized in 128-column blocks, so narrower MMA
    # tiles still allocate one full SF block and consume only the live subset.
    sfb_tile_n = max(blk_mn, ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)

    sSFA_shapeN = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sSF_strideN = (mn_basic_block_stride, blk_elems)

    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0, (
        "tile_shape_mnk[2] must be divisible by blk_sf * mma_nsf"
    )

    sSFA_shapeK = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sSF_strideK = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * blk_elems,
    )

    sSFA_shape = (sSFA_shapeN, sSFA_shapeK)
    sSFA_stride = (sSF_strideN, sSF_strideK)

    smem_layout = cute.make_layout(sSFA_shape, stride=sSFA_stride)

    sfb_smem_layout_staged = cute.append(
        smem_layout,
        cute.make_layout(
            num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))
        ),
    )

    return sfb_smem_layout_staged
