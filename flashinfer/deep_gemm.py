"""
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Imported and adapted from DeepGEMM.

import ctypes
import enum
import functools
import hashlib
import json
from typing import Any, Dict, Optional, Tuple

try:
    import cuda.bindings.driver as cbd
except ImportError as e:
    raise ImportError(
        "Could not import the 'cuda' module. "
        "Please install cuda-python that matches your CUDA version."
    ) from e

import torch

from .artifacts import ArtifactPath
from .cuda_utils import checkCudaErrors
from .jit.cubin_loader import get_cubin
from .jit.env import FLASHINFER_CUBIN_DIR
from .utils import (
    ceil_div,
    round_up,
    supported_compute_capability,
    backend_requirement,
)


class GemmType(enum.Enum):
    Normal = 0
    GroupedContiguous = 1
    GroupedMasked = 2

    def __str__(self) -> str:
        return {
            0: "GemmType::Normal",
            1: "GemmType::GroupedContiguous",
            2: "GemmType::GroupedMasked",
        }[self.value]


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def shape_direction(self):
        return 1 if self.value == 0 else -1

    def non_contiguous_dim(self):
        return -2 if self.value == 0 else -1

    def __str__(self) -> str:
        return {0: "cute::UMMA::Major::K", 1: "cute::UMMA::Major::MN"}[self.value]


class MajorTypeCD(enum.Enum):
    NMajor = 0
    MMajor = 1

    def non_contiguous_dim(self):
        return -2 if self.value == 0 else -1


def major_check(t: torch.Tensor):
    assert t.dim() in (2, 3)
    if t.dim() == 3:
        assert t.stride(0) == t.size(-2) * t.size(-1), (
            "Grouped dimension cannot have abnormal stride"
        )
    assert t.stride(-2) == 1 or t.stride(-1) == 1


def get_major_type_ab(t: torch.Tensor):
    major_check(t)
    return MajorTypeAB.KMajor if t.stride(-1) == 1 else MajorTypeAB.MNMajor


def get_major_type_cd(t: torch.Tensor):
    major_check(t)
    return MajorTypeCD.NMajor if t.stride(-1) == 1 else MajorTypeCD.MMajor


def get_element_size(dtype: torch.dtype):
    return {
        torch.float8_e4m3fn: 1,
        torch.bfloat16: 2,
        torch.float: 4,
    }[dtype]


def get_m_alignment_for_contiguous_layout():
    return 128


def get_tma_aligned_size(x: int, element_size: int) -> int:
    tma_alignment_bytes = 16
    assert tma_alignment_bytes % element_size == 0
    alignment = tma_alignment_bytes // element_size
    return round_up(x, alignment)


def get_col_major_tma_aligned_packed_tensor(x: torch.Tensor) -> torch.Tensor:
    # NOTES: for the extreme performance, you may rewrite/fuse this function in CUDA
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = round_up(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)

    # Finally, transpose
    transposed = torch.transpose(
        torch.empty((b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int),
        1,
        2,
    )
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def check_sf_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    gran: Tuple[int, int],
    num_groups: Optional[int],
    tma_stride_check: bool = False,
    type_check: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Type check
    if type_check is not None:
        assert sf.dtype == type_check

    # Always do shape checks
    assert sf.dtype in (torch.float, torch.int)
    assert sf.dim() == int(num_groups is not None) + 2
    if num_groups is not None:
        assert sf.size(-3) == num_groups
    assert sf.size(-2) == ceil_div(mn, gran[0])
    assert sf.size(-1) == ceil_div(k, gran[1] * (1 if sf.dtype == torch.float else 4))

    # TMA stride checks: TMA aligned and MN-major
    if tma_stride_check:
        if num_groups is not None:
            assert sf.stride(-3) == sf.stride(-1) * sf.size(-1)
        assert sf.stride(-2) == 1
        assert sf.stride(-1) == get_tma_aligned_size(mn, sf.element_size())

    return sf


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Tuple[int, int, int],
    num_groups: Optional[int] = None,
    is_sfa: bool = False,
):
    gran = (recipe[0 if is_sfa else 1], recipe[2])

    should_skip_transform = (
        sf.dtype == torch.int
        and gran == (1, 128)
        and get_device_arch() in ("100a", "103a")
    ) or (
        sf.dtype == torch.int
        and gran == (128, 128)
        and get_device_arch() in ("100a", "103a")
    )

    if not should_skip_transform:
        # Pre-transform checks
        check_sf_layout(sf, mn=mn, k=k, gran=gran, num_groups=num_groups)

    # (FP32, 1, 128) on Hopper: transform to TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (1, 128) and get_device_arch() == "90a":
        raise NotImplementedError

    # (FP32, 1, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if (
        sf.dtype == torch.float
        and gran == (1, 128)
        and get_device_arch() in ("100a", "103a")
    ):
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(
            sf,
            mn=mn,
            k=k,
            gran=(1, 128),
            num_groups=num_groups,
            tma_stride_check=True,
            type_check=torch.int,
        )

    # (FP32, 128, 128) on Hopper: no need to transform, check shape and whatever-major
    if sf.dtype == torch.float and gran == (128, 128) and get_device_arch() == "90a":
        raise NotImplementedError

    # (FP32, 128, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if (
        sf.dtype == torch.float
        and gran == (128, 128)
        and get_device_arch() in ("100a", "103a")
    ):
        sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
        sf = get_col_major_tma_aligned_packed_tensor(sf)
        return check_sf_layout(
            sf,
            mn=mn,
            k=k,
            gran=(1, 128),
            num_groups=num_groups,
            tma_stride_check=True,
            type_check=torch.int,
        )

    if should_skip_transform:
        # TODO: add transpose kernel if SF layout is not satisfied
        return check_sf_layout(
            sf,
            mn=mn,
            k=k,
            gran=(1, 128),
            num_groups=num_groups,
            tma_stride_check=True,
            type_check=torch.int,
        )

    raise AssertionError(
        f"Unknown cases: {sf.dtype=}, {gran=}, arch={get_device_arch()}"
    )


@functools.lru_cache(maxsize=None)
def get_device_arch():
    major, minor = torch.cuda.get_device_capability()
    suffix = "a" if major >= 9 else ""
    return f"{major * 10 + minor}{suffix}"


def hash_to_hex(s: str) -> str:
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()[0:12]


@functools.lru_cache(maxsize=None)
def must_be_k_major() -> bool:
    return {
        "90a": True,
        "100a": False,
        "103a": False,
    }[get_device_arch()]


@functools.lru_cache(maxsize=None)
def get_default_recipe(
    sfa_dtype: torch.dtype, sfb_dtype: torch.dtype
) -> Tuple[int, int, int]:
    assert sfa_dtype in (torch.float, torch.int)
    return {
        ("90a", torch.float): (1, 128, 128),
        ("100a", torch.float): (1, 128, 128),
        ("100a", torch.int): (1, 1, 128),
        ("103a", torch.float): (1, 128, 128),
        ("103a", torch.int): (1, 1, 128),
    }[(get_device_arch(), sfb_dtype)]


class MulticastConfig:
    def __init__(self, num_multicast: int, is_multicast_on_a: bool):
        self.num_multicast = num_multicast
        self.is_multicast_on_a = is_multicast_on_a

    def get_ab_load_block_m(self, block_m: int):
        # NOTES: this for >= SM100 only
        assert get_device_arch() != "90a"
        return block_m // (self.num_multicast if self.is_multicast_on_a else 1)

    def get_ab_load_block_n(self, block_n: int):
        # NOTES: this for >= SM100 only
        assert get_device_arch() != "90a"
        return block_n // (1 if self.is_multicast_on_a else self.num_multicast)


class SharedMemoryConfig:
    def __init__(
        self,
        smem_size: int,
        swizzle_a_mode: int,
        swizzle_b_mode: int,
        swizzle_cd_mode: int,
    ):
        self.smem_size = smem_size
        self.swizzle_a_mode = swizzle_a_mode
        self.swizzle_b_mode = swizzle_b_mode
        # NOTES: sometimes the default swizzling pattern maybe not compatible (e.g., FP32 output)
        self.swizzle_cd_mode = swizzle_cd_mode
        # TODO: swizzle SF as well
        self.swizzle_sf_mode = 0

        assert self.swizzle_a_mode != 0
        assert self.swizzle_b_mode != 0
        assert self.swizzle_cd_mode > 16
        assert self.swizzle_sf_mode == 0


def is_multicast_legal(
    shape_dim: int,
    block_dim: int,
    num_multicast: int,
    num_sms: int,
    require_divisible: bool = False,
) -> bool:
    divisible = (
        ceil_div(shape_dim, block_dim) % num_multicast == 0 or not require_divisible
    )
    return divisible and num_sms % num_multicast == 0


def get_swizzle_mode(block_size: int, elem_size: int) -> int:
    # `> 0` means interleaving
    # 16B actually means non-swizzling (but interleaving)
    for mode_bytes in (128, 64, 32, 16):
        if (block_size * elem_size) % mode_bytes == 0:
            return mode_bytes
    AssertionError("Invalid mode")
    return 0


def get_sf_aligned_block_sizes(block_m: int, block_n: int, ab_dtype: torch.dtype):
    num_utccp_aligned_elems = 128
    assert block_m % num_utccp_aligned_elems == 0
    return {
        torch.bfloat16: (0, 0),
        torch.float8_e4m3fn: (
            round_up(block_m, num_utccp_aligned_elems),
            round_up(block_n, num_utccp_aligned_elems),
        ),
    }[ab_dtype]


def is_tmem_size_legal(block_m: int, block_n: int, ab_dtype: torch.float):
    # M waves or epilogue stages (* 2), SFA and SFB
    sf_block_m, sf_block_n = get_sf_aligned_block_sizes(block_m, block_n, ab_dtype)
    return ((2 * block_n) + (sf_block_m // 32) + (sf_block_n // 32)) <= 512


def get_smem_config(
    block_m: int,
    block_n: int,
    block_k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    major_d: MajorTypeCD,
    ab_dtype: torch.dtype,
    cd_dtype: torch.dtype,
    num_stages: int,
    multicast_config: MulticastConfig,
) -> SharedMemoryConfig:
    assert major_d == MajorTypeCD.NMajor

    ab_elem_size = get_element_size(ab_dtype)
    cd_elem_size = get_element_size(cd_dtype)

    load_block_m = multicast_config.get_ab_load_block_m(block_m)
    load_block_n = multicast_config.get_ab_load_block_n(block_n)
    swizzle_a_mode = get_swizzle_mode(
        block_k if major_a == MajorTypeAB.KMajor else load_block_m, ab_elem_size
    )
    swizzle_b_mode = get_swizzle_mode(
        block_k if major_b == MajorTypeAB.KMajor else load_block_n, ab_elem_size
    )
    swizzle_cd_mode = get_swizzle_mode(
        block_n if major_d == MajorTypeCD.NMajor else block_m, cd_elem_size
    )

    # 2 stages of STSM and TMA store
    # TODO: consider other layouts
    layout_ad_m = 128
    smem_d = min(block_m, layout_ad_m) * swizzle_cd_mode * 2

    # A/B shared memory
    smem_a_per_stage = load_block_m * block_k * ab_elem_size
    smem_b_per_stage = load_block_n * block_k * ab_elem_size

    # SF shared memory must be aligned to UTCCP
    # Each stage must prefetch next 4 stages' SF (including the current)
    sf_block_m, sf_block_n = get_sf_aligned_block_sizes(block_m, block_n, ab_dtype)
    smem_scales_a_per_stage = sf_block_m * 4
    smem_scales_b_per_stage = sf_block_n * 4

    # TODO: remove SF barriers for BF16 GEMMs
    # TMA full/empty barriers, with-SF full barriers, tensor memory full/empty barriers, accumulation full barrier
    # NOTES: some shapes may only have 1 epilogue stage, but we still allocate space for 2 stages
    # NOTES: cases without accumulation will not use the accumulation full barrier
    smem_barrier = num_stages * 8 * 3 + 2 * 8 * 2 + 8
    smem_tmem_ptr = 4

    # Sum them up
    smem_size = 0
    smem_size += smem_d
    smem_size += num_stages * smem_a_per_stage
    smem_size += num_stages * smem_b_per_stage
    smem_size += num_stages * smem_scales_a_per_stage
    smem_size += num_stages * smem_scales_b_per_stage
    smem_size += smem_barrier
    smem_size += smem_tmem_ptr

    return SharedMemoryConfig(
        smem_size, swizzle_a_mode, swizzle_b_mode, swizzle_cd_mode
    )


@functools.lru_cache(maxsize=None)
def get_best_configs(
    gemm_type: GemmType,
    m: int,
    n: int,
    k: int,
    num_groups: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    major_d: MajorTypeCD,
    ab_dtype: torch.dtype,
    cd_dtype: torch.dtype,
    num_sms: int,
) -> Tuple[int, int, int, int, int, MulticastConfig, SharedMemoryConfig]:
    assert ab_dtype == torch.float8_e4m3fn
    assert cd_dtype in (torch.bfloat16, torch.float)

    # `BLOCK_M` and `BLOCK_N` are selected according to MMA instructions
    block_ms: Tuple[int, ...] = None
    if gemm_type == GemmType.GroupedContiguous:
        block_ms = (get_m_alignment_for_contiguous_layout(),)
    else:
        block_ms = (128,) if major_b == MajorTypeAB.KMajor else (128, 256)
    # NOTES: some `% 32 == 16` cases are not compatible with 2-CTA TMA swizzling
    block_ns = (
        tuple(range(16, 257, 16))
        if major_b == MajorTypeAB.KMajor
        else tuple(range(32, 257, 32))
    )

    # `BLOCK_K` is selected in a fixed manner
    block_k = 128 // get_element_size(ab_dtype)

    fix_wave_saturate = lambda x: num_sms if x == 0 else x
    get_num_waves = lambda bm, bn: (
        ceil_div(ceil_div(m, bm) * ceil_div(n, bn) * num_groups, num_sms)
        if bm
        else None
    )
    get_last_wave_util = lambda bm, bn: fix_wave_saturate(
        (ceil_div(m, bm) * ceil_div(n, bn) * num_groups) % num_sms
    )

    # Decide block sizes by waves
    # TODO: move block size search into `common.py`
    best_block_m, best_block_n = None, None
    for block_m in block_ms:
        for block_n in block_ns:
            success = False
            num_waves, best_num_waves = (
                get_num_waves(block_m, block_n),
                get_num_waves(best_block_m, best_block_n),
            )
            if (
                best_block_m is None
                or best_block_n is None
                or num_waves < best_num_waves
            ):
                success = True
            elif num_waves == best_num_waves:
                # Check last wave utilization
                util = get_last_wave_util(block_m, block_n)
                best_util = get_last_wave_util(best_block_m, best_block_n)
                success = util > best_util
                if util == best_util:
                    # Case 1: same `block_m`, smaller `block_n` (wasted)
                    success |= block_m == best_block_m and block_n < best_block_n
                    # Case 2: same `block_n`, smaller `block_m` (wasted)
                    success |= block_n == best_block_n and block_m < best_block_m
                    # Case 3: different for both `block_m` and `block_n`, larger `block_n` is better
                    success |= block_m != best_block_m and block_n > best_block_n
            success &= is_tmem_size_legal(block_m, block_n, ab_dtype)
            best_block_m, best_block_n = (
                (block_m, block_n) if success else (best_block_m, best_block_n)
            )
    assert best_block_m is not None and best_block_n is not None

    # Decide the number of TMA multicasts and whether broadcast on A
    best_multicast_config = MulticastConfig(1, True)

    # Try to multicast on the larger block side first
    is_legal = {
        # TODO: support other `tcgen05` layouts
        "A": False,
        "B": is_multicast_legal(m, best_block_m, 2, num_sms, True)
        and gemm_type == GemmType.Normal,
    }
    for i in ("A", "B") if best_block_m > best_block_n else ("B", "A"):
        if m >= 512 and is_legal[i]:
            best_multicast_config = MulticastConfig(2, i == "A")
            break

    # Always pick the longest one
    # NOTES: for double B scales, the best number of stages may be reduced
    # TODO: move stage search into `common.py`
    best_num_stages, best_smem_config, sm100_capacity = None, None, 232448
    stage_candidates = tuple(
        filter(lambda s: s <= max(k // 128, 1), (8, 7, 6, 5, 4, 3, 2, 1))
    )
    for num_stages in stage_candidates:
        best_smem_config = get_smem_config(
            best_block_m,
            best_block_n,
            block_k,
            major_a,
            major_b,
            major_d,
            ab_dtype,
            cd_dtype,
            num_stages,
            best_multicast_config,
        )
        if best_smem_config.smem_size <= sm100_capacity:
            best_num_stages = num_stages
            break
    assert best_smem_config is not None
    assert best_num_stages is not None

    # Recompute the minimal number of SMs required
    # NOTES: less L2 cache usage and less GPU frequency drop
    # TODO: move min SM fix into `common.py`
    num_waves = get_num_waves(best_block_m, best_block_n)
    num_min_sms = ceil_div(
        ceil_div(m, best_block_m) * ceil_div(n, best_block_n) * num_groups, num_waves
    )
    num_min_sms = (
        ceil_div(num_min_sms, best_multicast_config.num_multicast)
        * best_multicast_config.num_multicast
    )
    assert num_min_sms <= num_sms

    return (
        num_min_sms,
        best_block_m,
        best_block_n,
        block_k,
        best_num_stages,
        best_multicast_config,
        best_smem_config,
    )


tmap_type_map: Dict[Any, str] = {
    torch.int8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.int32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.uint8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.uint32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    torch.uint64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    torch.float32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e4m3fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_type_map = {
    0: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    16: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


def make_tma_xd_desc(
    t: torch.Tensor,
    gmem_dims: Tuple[cbd.cuuint64_t, ...],
    gmem_strides: Tuple[cbd.cuuint64_t, ...],
    smem_dims: Tuple[cbd.cuuint32_t, ...],
    swizzle_type: cbd.CUtensorMapSwizzle,
) -> cbd.CUtensorMap:
    num_dims = len(gmem_dims)
    assert len(gmem_strides) == num_dims - 1
    assert len(smem_dims) == num_dims

    tensor_dtype = tmap_type_map[t.dtype]
    tensor_map = checkCudaErrors(
        cbd.cuTensorMapEncodeTiled(
            tensor_dtype,
            num_dims,
            t.data_ptr(),
            gmem_dims,
            gmem_strides,
            smem_dims,
            (cbd.cuuint32_t(1),) * num_dims,
            cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle_type,
            cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    )
    return tensor_map


def make_tma_2d_desc(
    t: torch.Tensor,
    gmem_inner_dim: int,
    gmem_outer_dim: int,
    smem_inner_dim: int,
    smem_outer_dim: int,
    gmem_outer_stride: int,
    swizzle_mode: int,
) -> cbd.CUtensorMap:
    # For swizzling pattern, multiple TMAs are required
    if swizzle_mode != 0:
        assert swizzle_mode % t.element_size() == 0
        smem_inner_dim = swizzle_mode // t.element_size()

    gmem_dims = (cbd.cuuint64_t(gmem_inner_dim), cbd.cuuint64_t(gmem_outer_dim))
    gmem_strides = (cbd.cuuint64_t(gmem_outer_stride * t.element_size()),)
    smem_dims = (cbd.cuuint32_t(smem_inner_dim), cbd.cuuint32_t(smem_outer_dim))
    return make_tma_xd_desc(
        t, gmem_dims, gmem_strides, smem_dims, swizzle_type_map[swizzle_mode]
    )


def make_tma_a_desc(
    major_type: MajorTypeAB,
    t: torch.Tensor,
    shape_m: int,
    shape_k: int,
    block_m: int,
    block_k: int,
    outer_stride: int,
    num_groups: int,
    swizzle_mode: int,
) -> cbd.CUtensorMap:
    if num_groups > 1:
        assert major_type == MajorTypeAB.KMajor

    gmem_inner_dim, gmem_outer_dim = (shape_k, shape_m * num_groups)[
        :: major_type.shape_direction()
    ]
    smem_inner_dim, smem_outer_dim = (block_k, block_m)[:: major_type.shape_direction()]
    return make_tma_2d_desc(
        t,
        gmem_inner_dim,
        gmem_outer_dim,
        smem_inner_dim,
        smem_outer_dim,
        outer_stride,
        swizzle_mode,
    )


def make_tma_b_desc(
    major_type: MajorTypeAB,
    t: torch.Tensor,
    shape_n: int,
    shape_k: int,
    block_n: int,
    block_k: int,
    outer_stride: int,
    num_groups: int,
    swizzle_mode: int,
) -> cbd.CUtensorMap:
    # `num_groups` is always applied into the outer dimensions
    io_shapes = (shape_k, shape_n)[:: major_type.shape_direction()]
    gmem_inner_dim, gmem_outer_dim = (io_shapes[0], io_shapes[1] * num_groups)
    smem_inner_dim, smem_outer_dim = (block_k, block_n)[:: major_type.shape_direction()]

    return make_tma_2d_desc(
        t,
        gmem_inner_dim,
        gmem_outer_dim,
        smem_inner_dim,
        smem_outer_dim,
        outer_stride,
        swizzle_mode,
    )


def make_tma_cd_desc(
    major_type: MajorTypeCD,
    t: torch.Tensor,
    shape_m: int,
    shape_n: int,
    block_m: int,
    block_n: int,
    outer_stride: int,
    num_groups: int,
    swizzle_mode: int,
) -> cbd.CUtensorMap:
    assert major_type == MajorTypeCD.NMajor

    # Swizzling requires the inner box dim to be less or equal than `kSwizzleCDMode`
    # bytes, so `BLOCK_N * sizeof(T) / kSwizzleCDMode` TMA stores are required
    layout_ad_m = 128
    return make_tma_2d_desc(
        t,
        shape_n,
        shape_m * num_groups,
        block_n,
        min(block_m, layout_ad_m),
        outer_stride,
        swizzle_mode,
    )


def make_tma_sf_desc(
    major_type: MajorTypeAB,
    t: torch.Tensor,
    shape_mn: int,
    shape_k: int,
    block_mn: int,
    block_k: int,
    num_groups: int,
    swizzle_mode: int,
) -> cbd.CUtensorMap:
    assert major_type == MajorTypeAB.MNMajor

    # TODO: maybe swizzle SF as well
    assert swizzle_mode == 0

    # Make TMA aligned to 16 bytes
    shape_mn = get_tma_aligned_size(shape_mn, t.element_size())
    return make_tma_2d_desc(
        t,
        shape_mn,
        ceil_div(shape_k, block_k * 4) * num_groups,
        block_mn,
        1,
        shape_mn,
        swizzle_mode,
    )


# Map some common Python types into C types
pytypes_to_ctypes = {
    True: "true",
    False: "false",
    torch.bfloat16: "cutlass::bfloat16_t",
    torch.float: "float",
}


RUNTIME_CACHE = {}


class SM100FP8GemmRuntime:
    def __init__(self, path: str, symbol: str) -> None:
        self.path = path
        self.lib = None
        self.kernel = None
        self.symbol = symbol
        # Store a reference to the cleanup function to avoid import issues during shutdown
        self._cleanup_func = cbd.cuLibraryUnload

    def __call__(self, **kwargs) -> cbd.CUresult:
        # Load CUBIN
        if self.kernel is None:
            # Load CUBIN
            path = bytes(self.path, encoding="utf-8")
            self.lib = checkCudaErrors(
                cbd.cuLibraryLoadFromFile(path, [], [], 0, [], [], 0)
            )
            self.kernel = checkCudaErrors(
                cbd.cuLibraryGetKernel(self.lib, bytes(self.symbol, encoding="utf-8"))
            )

        # noinspection PyArgumentList
        return self.launch(self.kernel, kwargs)

    def __del__(self) -> None:
        if self.lib is not None:
            cleanup = getattr(self, "_cleanup_func", None)
            if callable(cleanup):
                try:
                    cleanup(self.lib)
                except Exception as e:
                    # Ignore any errors during shutdown
                    print(f"Failed to delete SM100FP8GemmRuntime with exception: {e}")

    @staticmethod
    def generate(kwargs: Dict[str, Any]) -> str:
        assert kwargs["CD_DTYPE_T"] in (torch.bfloat16, torch.float)
        code = f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_1d1d_impl<
        {kwargs["MAJOR_A"]},
        {kwargs["MAJOR_B"]},
        {kwargs["M"] if "m" in kwargs["COMPILED_DIMS"] else 0},
        {kwargs["N"] if "n" in kwargs["COMPILED_DIMS"] else 0},
        {kwargs["K"] if "k" in kwargs["COMPILED_DIMS"] else 0},
        {kwargs["BLOCK_M"]},
        {kwargs["BLOCK_N"]},
        {kwargs["BLOCK_K"]},
        {kwargs["NUM_GROUPS"]},
        {kwargs["SWIZZLE_A_MODE"]},
        {kwargs["SWIZZLE_B_MODE"]},
        {kwargs["SWIZZLE_CD_MODE"]},
        {kwargs["NUM_STAGES"]},
        {kwargs["NUM_LAST_STAGES"]},
        {kwargs["NUM_NON_EPILOGUE_THREADS"]},
        {kwargs["NUM_EPILOGUE_THREADS"]},
        {kwargs["NUM_MULTICAST"]},
        {pytypes_to_ctypes[kwargs["IS_MULTICAST_ON_A"]]},
        {kwargs["GEMM_TYPE"]},
        {pytypes_to_ctypes[kwargs["WITH_ACCUMULATION"]]},
        {pytypes_to_ctypes[kwargs["CD_DTYPE_T"]]}
      >);
}};
"""
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel: cbd.CUkernel, kwargs: Dict[str, Any]) -> cbd.CUresult:
        checkCudaErrors(
            cbd.cuKernelSetAttribute(
                cbd.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kwargs["SMEM_SIZE"],
                kernel,
                cbd.CUdevice(kwargs["DEVICE_INDEX"]),
            )
        )

        attr_val = cbd.CUlaunchAttributeValue()
        attr_val.clusterDim.x = kwargs["NUM_MULTICAST"]
        attr_val.clusterDim.y = 1
        attr_val.clusterDim.z = 1
        attr = cbd.CUlaunchAttribute()
        attr.id = cbd.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value = attr_val

        config = cbd.CUlaunchConfig()
        config.numAttrs = 1
        config.attrs = [attr]
        config.gridDimX = kwargs["NUM_SMS"]
        config.gridDimY = 1
        config.gridDimZ = 1
        config.blockDimX = (
            kwargs["NUM_NON_EPILOGUE_THREADS"] + kwargs["NUM_EPILOGUE_THREADS"]
        )
        config.blockDimY = 1
        config.blockDimZ = 1
        config.sharedMemBytes = kwargs["SMEM_SIZE"]
        config.hStream = kwargs["STREAM"]

        arg_values = (
            kwargs["GROUPED_LAYOUT"].data_ptr(),
            kwargs["M"],
            kwargs["N"],
            kwargs["K"],
            kwargs["TENSOR_MAP_A"],
            kwargs["TENSOR_MAP_B"],
            kwargs["TENSOR_MAP_SFA"],
            kwargs["TENSOR_MAP_SFB"],
            kwargs["TENSOR_MAP_C"],
            kwargs["TENSOR_MAP_D"],
        )
        arg_types = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        return cbd.cuLaunchKernelEx(config, kernel, (arg_values, arg_types), 0)


def load_all():
    for cubin_name in KERNEL_MAP:
        if cubin_name in RUNTIME_CACHE:
            continue
        symbol, sha256 = KERNEL_MAP[cubin_name]
        cubin_name = cubin_name + ".cubin"
        get_cubin(ArtifactPath.DEEPGEMM + "/" + cubin_name, sha256)
        path = FLASHINFER_CUBIN_DIR / ArtifactPath.DEEPGEMM / cubin_name
        assert path.exists()
        RUNTIME_CACHE[cubin_name] = SM100FP8GemmRuntime(str(path), symbol)


def load(name: str, code: str) -> SM100FP8GemmRuntime:
    signature = f"{name}$${code}"
    cubin_name = f"kernel.{name}.{hash_to_hex(signature)}"
    if cubin_name not in KERNEL_MAP:
        raise ValueError(f"cubin not registered: {cubin_name}")
    if cubin_name in RUNTIME_CACHE:
        return RUNTIME_CACHE[cubin_name]
    symbol, sha256 = KERNEL_MAP[cubin_name]
    cubin_name = cubin_name + ".cubin"
    get_cubin(ArtifactPath.DEEPGEMM + "/" + cubin_name, sha256)
    path = FLASHINFER_CUBIN_DIR / ArtifactPath.DEEPGEMM / cubin_name
    assert path.exists()
    RUNTIME_CACHE[cubin_name] = SM100FP8GemmRuntime(str(path), symbol)
    return RUNTIME_CACHE[cubin_name]


def m_grouped_fp8_gemm_nt_contiguous_static_kwargs_gen(
    m: int,
    n: int,
    k: int,
    aligned_k: int,
    num_groups: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    major_d: MajorTypeCD,
    compiled_dims: str,
    output_dtype: torch.dtype,
):
    num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    num_sms, block_m, block_n, block_k, num_stages, multicast_config, smem_config = (
        get_best_configs(
            GemmType.GroupedContiguous,
            m,
            n,
            k,
            num_groups,
            major_a,
            major_b,
            major_d,
            torch.float8_e4m3fn,
            output_dtype,
            num_sms,
        )
    )
    kwargs = {
        # Templated or runtime arguments according to the `COMPILED_DIMS`
        "COMPILED_DIMS": compiled_dims,
        "M": m,
        "N": n,
        "K": aligned_k,
        # Templated arguments
        "GEMM_TYPE": GemmType.GroupedContiguous,
        "NUM_NON_EPILOGUE_THREADS": 128,
        "NUM_EPILOGUE_THREADS": 128,
        "MAJOR_A": major_a,
        "MAJOR_B": major_b,
        "NUM_GROUPS": num_groups,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "NUM_STAGES": num_stages,
        "NUM_LAST_STAGES": ceil_div(k, block_k) % num_stages,
        "SWIZZLE_A_MODE": smem_config.swizzle_a_mode,
        "SWIZZLE_B_MODE": smem_config.swizzle_b_mode,
        "SWIZZLE_CD_MODE": smem_config.swizzle_cd_mode,
        "NUM_MULTICAST": multicast_config.num_multicast,
        "IS_MULTICAST_ON_A": multicast_config.is_multicast_on_a,
        "WITH_ACCUMULATION": False,
        "CD_DTYPE_T": output_dtype,
    }
    return (
        num_sms,
        block_m,
        block_n,
        block_k,
        num_stages,
        multicast_config,
        smem_config,
    ), kwargs


def m_grouped_fp8_gemm_nt_contiguous_kwargs_gen(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    compiled_dims: str,
):
    m, k = a.shape
    num_groups, n, _ = b.shape
    major_d = MajorTypeCD.NMajor

    # K must be aligned to 128
    aligned_k = round_up(k, 128)
    (
        (
            num_sms,
            block_m,
            block_n,
            block_k,
            num_stages,
            multicast_config,
            smem_config,
        ),
        static_kwargs,
    ) = m_grouped_fp8_gemm_nt_contiguous_static_kwargs_gen(
        m,
        n,
        k,
        aligned_k,
        num_groups,
        major_a,
        major_b,
        major_d,
        compiled_dims,
        d.dtype,
    )
    # NOTES: you cannot distinguish groups for A, SFA, and D
    tensor_map_a = make_tma_a_desc(
        major_a,
        a,
        m,
        k,
        multicast_config.get_ab_load_block_m(block_m),
        block_k,
        a.stride(major_a.non_contiguous_dim()),
        num_groups=1,
        swizzle_mode=smem_config.swizzle_a_mode,
    )
    tensor_map_b = make_tma_b_desc(
        major_b,
        b,
        n,
        k,
        multicast_config.get_ab_load_block_n(block_n),
        block_k,
        b.stride(major_b.non_contiguous_dim()),
        num_groups=num_groups,
        swizzle_mode=smem_config.swizzle_b_mode,
    )
    tensor_map_d = make_tma_cd_desc(
        major_d,
        d,
        m,
        n,
        block_m,
        block_n,
        d.stride(major_d.non_contiguous_dim()),
        num_groups=1,
        swizzle_mode=smem_config.swizzle_cd_mode,
    )
    tensor_map_sfa = make_tma_sf_desc(
        MajorTypeAB.MNMajor,
        sfa,
        m,
        k,
        block_m,
        block_k,
        num_groups=1,
        swizzle_mode=smem_config.swizzle_sf_mode,
    )
    tensor_map_sfb = make_tma_sf_desc(
        MajorTypeAB.MNMajor,
        sfb,
        n,
        k,
        block_n,
        block_k,
        num_groups=num_groups,
        swizzle_mode=smem_config.swizzle_sf_mode,
    )
    all_kwargs = {
        **static_kwargs,
        # Runtime arguments
        "GROUPED_LAYOUT": m_indices,
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config.smem_size,
        "TENSOR_MAP_A": tensor_map_a,
        "TENSOR_MAP_B": tensor_map_b,
        "TENSOR_MAP_SFA": tensor_map_sfa,
        "TENSOR_MAP_SFB": tensor_map_sfb,
        "TENSOR_MAP_C": tensor_map_d,
        "TENSOR_MAP_D": tensor_map_d,
        "STREAM": torch.cuda.current_stream().cuda_stream,
        "DEVICE_INDEX": d.device.index,
    }
    return static_kwargs, all_kwargs


def m_grouped_fp8_gemm_nt_contiguous_sm10x(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    d: torch.Tensor,
    m_indices: torch.Tensor,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    compiled_dims: str,
) -> None:
    static_kwargs, all_kwargs = m_grouped_fp8_gemm_nt_contiguous_kwargs_gen(
        a, sfa, b, sfb, d, m_indices, major_a, major_b, compiled_dims
    )
    # Generate, build and run the kernel
    code = SM100FP8GemmRuntime.generate(static_kwargs)
    runtime = load("fp8_m_grouped_gemm", code)
    runtime(**all_kwargs)


def m_grouped_fp8_gemm_nt_masked_static_kwargs_gen(
    m: int,
    n: int,
    k: int,
    expected_m: int,
    aligned_k: int,
    num_groups: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    major_d: MajorTypeCD,
    compiled_dims: str,
    output_dtype: torch.dtype,
):
    num_sms = torch.cuda.get_device_properties(device="cuda").multi_processor_count
    num_sms, block_m, block_n, block_k, num_stages, multicast_config, smem_config = (
        get_best_configs(
            GemmType.GroupedMasked,
            expected_m,
            n,
            k,
            num_groups,
            major_a,
            major_b,
            major_d,
            torch.float8_e4m3fn,
            output_dtype,
            num_sms,
        )
    )
    if num_groups > 1:
        assert m % block_m == 0

    kwargs = {
        # Templated or runtime arguments according to the `COMPILED_DIMS`
        "COMPILED_DIMS": compiled_dims,
        "M": m,
        "N": n,
        "K": aligned_k,
        # Templated arguments
        "GEMM_TYPE": GemmType.GroupedMasked,
        "NUM_NON_EPILOGUE_THREADS": 128,
        "NUM_EPILOGUE_THREADS": 128,
        "MAJOR_A": major_a,
        "MAJOR_B": major_b,
        "NUM_GROUPS": num_groups,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "NUM_STAGES": num_stages,
        "NUM_LAST_STAGES": ceil_div(k, block_k) % num_stages,
        "SWIZZLE_A_MODE": smem_config.swizzle_a_mode,
        "SWIZZLE_B_MODE": smem_config.swizzle_b_mode,
        "SWIZZLE_CD_MODE": smem_config.swizzle_cd_mode,
        "NUM_MULTICAST": multicast_config.num_multicast,
        "IS_MULTICAST_ON_A": multicast_config.is_multicast_on_a,
        "WITH_ACCUMULATION": False,
        "CD_DTYPE_T": output_dtype,
    }
    return (
        num_sms,
        block_m,
        block_n,
        block_k,
        num_stages,
        multicast_config,
        smem_config,
    ), kwargs


def m_grouped_fp8_gemm_nt_masked_kwargs_gen(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    compiled_dims: str,
):
    num_groups, m, k = a.shape
    _, n, _ = b.shape
    major_d = MajorTypeCD.NMajor

    # K must be aligned to 128
    aligned_k = round_up(k, 128)
    (
        (
            num_sms,
            block_m,
            block_n,
            block_k,
            num_stages,
            multicast_config,
            smem_config,
        ),
        static_kwargs,
    ) = m_grouped_fp8_gemm_nt_masked_static_kwargs_gen(
        m,
        n,
        k,
        expected_m,
        aligned_k,
        num_groups,
        major_a,
        major_b,
        major_d,
        compiled_dims,
        d.dtype,
    )

    tensor_map_a = make_tma_a_desc(
        major_a,
        a,
        m,
        k,
        multicast_config.get_ab_load_block_m(block_m),
        block_k,
        a.stride(major_a.non_contiguous_dim()),
        num_groups,
        smem_config.swizzle_a_mode,
    )
    tensor_map_b = make_tma_b_desc(
        major_b,
        b,
        n,
        k,
        multicast_config.get_ab_load_block_n(block_n),
        block_k,
        b.stride(major_b.non_contiguous_dim()),
        num_groups,
        smem_config.swizzle_b_mode,
    )
    tensor_map_d = make_tma_cd_desc(
        major_d,
        d,
        m,
        n,
        block_m,
        block_n,
        d.stride(major_d.non_contiguous_dim()),
        num_groups,
        smem_config.swizzle_cd_mode,
    )
    tensor_map_sfa = make_tma_sf_desc(
        MajorTypeAB.MNMajor,
        sfa,
        m,
        k,
        block_m,
        block_k,
        num_groups,
        smem_config.swizzle_sf_mode,
    )
    tensor_map_sfb = make_tma_sf_desc(
        MajorTypeAB.MNMajor,
        sfb,
        n,
        k,
        block_n,
        block_k,
        num_groups,
        smem_config.swizzle_sf_mode,
    )
    all_kwargs = {
        **static_kwargs,
        # Runtime arguments
        "GROUPED_LAYOUT": masked_m,
        "NUM_SMS": num_sms,
        "SMEM_SIZE": smem_config.smem_size,
        "TENSOR_MAP_A": tensor_map_a,
        "TENSOR_MAP_B": tensor_map_b,
        "TENSOR_MAP_SFA": tensor_map_sfa,
        "TENSOR_MAP_SFB": tensor_map_sfb,
        "TENSOR_MAP_C": tensor_map_d,
        "TENSOR_MAP_D": tensor_map_d,
        "STREAM": torch.cuda.current_stream().cuda_stream,
        "DEVICE_INDEX": d.device.index,
    }
    return static_kwargs, all_kwargs


def m_grouped_fp8_gemm_nt_masked_sm10x(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    compiled_dims: str,
) -> None:
    static_kwargs, all_kwargs = m_grouped_fp8_gemm_nt_masked_kwargs_gen(
        a, sfa, b, sfb, d, masked_m, expected_m, major_a, major_b, compiled_dims
    )
    # Generate, build and run the kernel
    code = SM100FP8GemmRuntime.generate(static_kwargs)
    runtime = load("fp8_m_grouped_gemm", code)
    runtime(**all_kwargs)


@supported_compute_capability([100, 103])
def _check_group_deepgemm_fp8_nt_contiguous_problem_size(
    a_fp8: Tuple[torch.Tensor, torch.Tensor],
    b_fp8: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
) -> bool:
    # NOTES: shape must be `[M, K] @ [G, N, K].mT`
    major_a = get_major_type_ab(a_fp8[0])
    major_b = get_major_type_ab(b_fp8[0])
    if major_a != MajorTypeAB.KMajor:
        raise ValueError(f"major_a must be KMajor, but got {major_a}")
    if must_be_k_major() and (major_b != MajorTypeAB.KMajor):
        raise ValueError(f"major_b must be KMajor, but got {major_b}")

    if not m_indices.is_contiguous():
        raise ValueError(
            f"m_indices must be contiguous, but got {m_indices.is_contiguous()}"
        )

    a, sfa = a_fp8
    b, sfb = b_fp8
    m, k = a.shape
    num_groups, n, k_ = b.shape
    m_, n_ = d.shape
    m__ = m_indices.numel()

    # Type and shape checks
    if m != m_ or k != k_ or n != n_ or m__ != m_:
        raise ValueError(
            f"Shape mismatch. m = {m}, m_ = {m_}, k = {k}, k_ = {k_}, n = {n}, n_ = {n_}, m__ = {m__}"
        )
    if a.dtype != torch.float8_e4m3fn:
        raise ValueError(f"a must be float8_e4m3fn, but got {a.dtype}")
    if b.dtype != torch.float8_e4m3fn:
        raise ValueError(f"b must be float8_e4m3fn, but got {b.dtype}")
    if d.dtype != torch.bfloat16:
        raise ValueError(f"d must be bfloat16, but got {d.dtype}")
    if m_indices.dtype != torch.int32:
        raise ValueError(f"m_indices must be int32, but got {m_indices.dtype}")

    # D must be N-major
    if get_major_type_cd(d) != MajorTypeCD.NMajor:
        raise ValueError(f"d must be N-major, but got {get_major_type_cd(d)}")

    return True


@backend_requirement(
    {},
    common_check=_check_group_deepgemm_fp8_nt_contiguous_problem_size,
)
def m_grouped_fp8_gemm_nt_contiguous(
    a_fp8: Tuple[torch.Tensor, torch.Tensor],
    b_fp8: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
) -> None:
    # Compiled dims can be upper cases
    compiled_dims = compiled_dims.lower()

    major_a = get_major_type_ab(a_fp8[0])
    major_b = get_major_type_ab(b_fp8[0])

    a, sfa = a_fp8
    b, sfb = b_fp8
    m, k = a.shape
    num_groups, n, k_ = b.shape

    # Do nothing if the problem is empty
    if m == 0:
        return

    # Transform SFA and SFB into compute-required layout
    recipe = get_default_recipe(sfa.dtype, sfb.dtype) if recipe is None else recipe
    sfa = transform_sf_into_required_layout(sfa, mn=m, k=k, recipe=recipe, is_sfa=True)
    sfb = transform_sf_into_required_layout(
        sfb, mn=n, k=k, recipe=recipe, num_groups=num_groups, is_sfa=False
    )

    impl = {
        "100a": functools.partial(
            m_grouped_fp8_gemm_nt_contiguous_sm10x,
            major_a=major_a,
            major_b=major_b,
            compiled_dims=compiled_dims,
        ),
        "103a": functools.partial(
            m_grouped_fp8_gemm_nt_contiguous_sm10x,
            major_a=major_a,
            major_b=major_b,
            compiled_dims=compiled_dims,
        ),
    }[get_device_arch()]
    impl(a, sfa, b, sfb, d, m_indices)


@supported_compute_capability([100, 103])
def _check_m_grouped_fp8_gemm_nt_masked_problem_size(
    a_fp8: Tuple[torch.Tensor, torch.Tensor],
    b_fp8: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
) -> bool:
    major_a = get_major_type_ab(a_fp8[0])
    major_b = get_major_type_ab(b_fp8[0])
    if major_a != MajorTypeAB.KMajor:
        raise ValueError(f"major_a must be KMajor, but got {major_a}")
    if major_b != MajorTypeAB.KMajor:
        raise ValueError(f"major_b must be KMajor, but got {major_b}")

    if not masked_m.is_contiguous():
        raise ValueError(
            f"masked_m must be contiguous, but got {masked_m.is_contiguous()}"
        )

    a, sfa = a_fp8
    b, sfb = b_fp8
    num_groups, m, k = a.shape
    num_groups_, n, k_ = b.shape
    num_groups__, m_, n_ = d.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    if (
        num_groups != num_groups_
        or num_groups != num_groups__
        or num_groups != num_groups___
    ):
        raise ValueError(
            f"num_groups mismatch. num_groups = {num_groups}, num_groups_ = {num_groups_}, num_groups__ = {num_groups__}, num_groups___ = {num_groups___}"
        )
    if m != m_ or n != n_ or k != k_:
        raise ValueError(
            f"m, n, k mismatch. m = {m}, m_ = {m_}, n = {n}, n_ = {n_}, k = {k}, k_ = {k_}"
        )
    if expected_m <= 0 or m <= 0 or n <= 0 or k <= 0 or num_groups <= 0:
        raise ValueError(
            f"expected_m, m, n, k, num_groups must be greater than 0, but got expected_m = {expected_m}, m = {m}, n = {n}, k = {k}, num_groups = {num_groups}"
        )
    if a.dtype != torch.float8_e4m3fn:
        raise ValueError(f"a must be float8_e4m3fn, but got {a.dtype}")
    if b.dtype != torch.float8_e4m3fn:
        raise ValueError(f"b must be float8_e4m3fn, but got {b.dtype}")
    if d.dtype != torch.bfloat16:
        raise ValueError(f"d must be bfloat16, but got {d.dtype}")
    if masked_m.dtype != torch.int32:
        raise ValueError(f"masked_m must be int32, but got {masked_m.dtype}")

    # D must be N-major
    if get_major_type_cd(d) != MajorTypeCD.NMajor:
        raise ValueError(f"d must be N-major, but got {get_major_type_cd(d)}")

    return True


@backend_requirement(
    {},
    common_check=_check_m_grouped_fp8_gemm_nt_masked_problem_size,
)
def m_grouped_fp8_gemm_nt_masked(
    a_fp8: Tuple[torch.Tensor, torch.Tensor],
    b_fp8: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
) -> None:
    # Compiled dims can be upper cases
    compiled_dims = compiled_dims.lower()

    # NOTES: shape must be `[G, M, K] @ [G, N, K].mT`
    major_a = get_major_type_ab(a_fp8[0])
    major_b = get_major_type_ab(b_fp8[0])
    assert major_a == major_b == MajorTypeAB.KMajor
    assert masked_m.is_contiguous()

    a, sfa = a_fp8
    b, sfb = b_fp8
    num_groups, m, k = a.shape
    num_groups_, n, k_ = b.shape

    # Transform SFA and SFB into compute-required layout
    recipe = get_default_recipe(sfa.dtype, sfb.dtype) if recipe is None else recipe
    sfa = transform_sf_into_required_layout(
        sfa, mn=m, k=k, recipe=recipe, num_groups=num_groups, is_sfa=True
    )
    sfb = transform_sf_into_required_layout(
        sfb, mn=n, k=k, recipe=recipe, num_groups=num_groups, is_sfa=False
    )

    impl = {
        "100a": functools.partial(
            m_grouped_fp8_gemm_nt_masked_sm10x,
            major_a=major_a,
            major_b=major_b,
            compiled_dims=compiled_dims,
        ),
        "103a": functools.partial(
            m_grouped_fp8_gemm_nt_masked_sm10x,
            major_a=major_a,
            major_b=major_b,
            compiled_dims=compiled_dims,
        ),
    }[get_device_arch()]
    impl(a, sfa, b, sfb, d, masked_m, expected_m)


class KernelMap:
    # Hash for kernel_map.json, updated when deepgemm cubins are republished
    KERNEL_MAP_HASH = "f161e031826adb8c4f0d31ddbd2ed77e4909e4e43cdfc9728918162a62fcccfb"

    def __init__(self):
        self.indice = None

    def init_indices(self):
        indice_path = ArtifactPath.DEEPGEMM + "/" + "kernel_map.json"
        assert get_cubin(indice_path, self.KERNEL_MAP_HASH), (
            "cubin kernel map file not found, nor downloaded with matched sha256"
        )
        path = FLASHINFER_CUBIN_DIR / indice_path
        assert path.exists()
        with open(path, "r") as f:
            self.indice = json.load(f)

    def __iter__(self):
        if self.indice is None:
            self.init_indices()
        for name in self.indice:
            yield name

    def __getitem__(self, key):
        if self.indice is None:
            self.init_indices()
        return self.indice[key]


KERNEL_MAP = KernelMap()
