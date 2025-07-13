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
from typing import Any, Dict, Optional, Tuple

import cuda.bindings.driver as cbd
import torch

from .cuda_utils import checkCudaErrors
from .jit.cubin_loader import get_cubin
from .jit.env import FLASHINFER_CACHE_DIR
from .utils import ceil_div, round_up


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
        assert t.stride(0) == t.size(-2) * t.size(
            -1
        ), "Grouped dimension cannot have abnormal stride"
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
        sf.dtype == torch.int and gran == (1, 128) and get_device_arch() == "100a"
    ) or (sf.dtype == torch.int and gran == (128, 128) and get_device_arch() == "100a")

    if not should_skip_transform:
        # Pre-transform checks
        check_sf_layout(sf, mn=mn, k=k, gran=gran, num_groups=num_groups)

    # (FP32, 1, 128) on Hopper: transform to TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (1, 128) and get_device_arch() == "90a":
        raise NotImplemented

    # (FP32, 1, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (1, 128) and get_device_arch() == "100a":
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
        raise NotImplemented

    # (FP32, 128, 128) on SM100: transform to (INT, 1, 128), TMA-aligned and MN-major
    if sf.dtype == torch.float and gran == (128, 128) and get_device_arch() == "100a":
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

    assert False, f"Unknown cases: {sf.dtype=}, {gran=}, arch={get_device_arch()}"


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
    assert False, "Invalid mode"


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
            num_waves, best_num_waves = get_num_waves(block_m, block_n), get_num_waves(
                best_block_m, best_block_n
            )
            if best_block_m is None or best_block_n is None:
                success = True
            elif num_waves < best_num_waves:
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
    return make_tma_2d_desc(
        t,
        *(shape_k, shape_m * num_groups)[:: major_type.shape_direction()],
        *(block_k, block_m)[:: major_type.shape_direction()],
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
    io_shapes = (io_shapes[0], io_shapes[1] * num_groups)

    return make_tma_2d_desc(
        t,
        *io_shapes,
        *(block_k, block_n)[:: major_type.shape_direction()],
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
            checkCudaErrors(cbd.cuLibraryUnload(self.lib))

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
        {kwargs['MAJOR_A']},
        {kwargs['MAJOR_B']},
        {kwargs['M'] if 'm' in kwargs['COMPILED_DIMS'] else 0},
        {kwargs['N'] if 'n' in kwargs['COMPILED_DIMS'] else 0},
        {kwargs['K'] if 'k' in kwargs['COMPILED_DIMS'] else 0},
        {kwargs['BLOCK_M']},
        {kwargs['BLOCK_N']},
        {kwargs['BLOCK_K']},
        {kwargs['NUM_GROUPS']},
        {kwargs['SWIZZLE_A_MODE']},
        {kwargs['SWIZZLE_B_MODE']},
        {kwargs['SWIZZLE_CD_MODE']},
        {kwargs['NUM_STAGES']},
        {kwargs['NUM_LAST_STAGES']},
        {kwargs['NUM_NON_EPILOGUE_THREADS']},
        {kwargs['NUM_EPILOGUE_THREADS']},
        {kwargs['NUM_MULTICAST']},
        {pytypes_to_ctypes[kwargs['IS_MULTICAST_ON_A']]},
        {kwargs['GEMM_TYPE']},
        {pytypes_to_ctypes[kwargs['WITH_ACCUMULATION']]},
        {pytypes_to_ctypes[kwargs['CD_DTYPE_T']]}
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
        get_cubin(cubin_name, sha256)
        path = FLASHINFER_CACHE_DIR / "cubins" / f"{cubin_name}.cubin"
        assert path.exists()
        RUNTIME_CACHE[cubin_name] = SM100FP8GemmRuntime(str(path), symbol)


def load(name: str, code: str) -> SM100FP8GemmRuntime:
    signature = f"{name}$${code}"
    cubin_name = f"kernel.{name}.{hash_to_hex(signature)}"
    if cubin_name not in KERNEL_MAP:
        raise ValueError("cubin not registered")
    if cubin_name in RUNTIME_CACHE:
        return RUNTIME_CACHE[cubin_name]
    symbol, sha256 = KERNEL_MAP[cubin_name]
    get_cubin(cubin_name, sha256)
    path = FLASHINFER_CACHE_DIR / "cubins" / f"{cubin_name}.cubin"
    print(path)
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
        num_sms,
        block_m,
        block_n,
        block_k,
        num_stages,
        multicast_config,
        smem_config,
    ), static_kwargs = m_grouped_fp8_gemm_nt_contiguous_static_kwargs_gen(
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


def m_grouped_fp8_gemm_nt_contiguous_sm100(
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


def m_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    m_indices: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
) -> None:
    # Compiled dims can be upper cases
    compiled_dims = compiled_dims.lower()

    # NOTES: shape must be `[M, K] @ [G, N, K].mT`
    major_a = get_major_type_ab(a[0])
    major_b = get_major_type_ab(b[0])
    assert major_a == MajorTypeAB.KMajor
    if must_be_k_major():
        assert major_b == MajorTypeAB.KMajor
    assert m_indices.is_contiguous()

    a, sfa = a
    b, sfb = b
    m, k = a.shape
    num_groups, n, k_ = b.shape
    m_, n_ = d.shape
    m__ = m_indices.numel()

    # Type and shape checks
    assert m == m_ == m__ and n == n_ and k == k_
    assert n > 0 and k > 0 and num_groups > 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    assert d.dtype == torch.bfloat16
    assert m_indices.dtype == torch.int32

    # D must be N-major
    assert get_major_type_cd(d) == MajorTypeCD.NMajor

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
            m_grouped_fp8_gemm_nt_contiguous_sm100,
            major_a=major_a,
            major_b=major_b,
            compiled_dims=compiled_dims,
        )
    }[get_device_arch()]
    impl(a, sfa, b, sfb, d, m_indices)


# fmt: off
KERNEL_MAP = {
    'kernel.fp8_m_grouped_gemm.4caa3b87e72c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '974430846f83807e9ac33ed485715dfad5e980356ba165fcf7d9298b95740132'),
    'kernel.fp8_m_grouped_gemm.7953ee0a470b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '927ea49d4c71dce096fec7cd112d05cde5a7d375ceb0509bf9d1c416b9b18102'),
    'kernel.fp8_m_grouped_gemm.43a66a194af2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8884aedfd483cadc2dd39a42beec6755e9490a9257e4bafd57d0d176bf7eedec'),
    'kernel.fp8_m_grouped_gemm.03104cf927df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '79c8e72af840326d32380ef676bd694a6a2d6ea79ffc10089261f49247797031'),
    'kernel.fp8_m_grouped_gemm.d8caec6f111b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd540d880f3724a00d4e25a9842d228e2aa67305e606fbc7d93ff80256262c213'),
    'kernel.fp8_m_grouped_gemm.9e474c44622e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a6381638c9bff6a735308e5acc87bbff42df9054bb038f9a545b8f94bd623e77'),
    'kernel.fp8_m_grouped_gemm.e2987a4d4273': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3b3b25cfe0ee2611d766948a701b0575f562f2fac8496459b0c158a58b999864'),
    'kernel.fp8_m_grouped_gemm.9b9742c0b9b2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '475b30cd5ab01af715e936500414cc1e35ac1820fd63f1c848dea2a5ff192ada'),
    'kernel.fp8_m_grouped_gemm.116f46018c23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f2d6c6d845d9c0ed324ffca6a96418604b0cb9bff99303b9b3b5be72d284ddaf'),
    'kernel.fp8_m_grouped_gemm.9d265f01caa7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd0549a424abe77f5464ff66ae67c5e63f644137b2fa5948d0ef1ce9136f50c7e'),
    'kernel.fp8_m_grouped_gemm.a918127d1773': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8482c4bac2130fdcb953b394b046b12e38e065374f479a810f1e2ee0f4bf3506'),
    'kernel.fp8_m_grouped_gemm.d709904c7313': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '037cba9091e6ddfeff3fa4fd1a618b058ff1bbb2d208ece3caf0c5aebdcaabb3'),
    'kernel.fp8_m_grouped_gemm.c4b0089929a7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f3eb3921d40711d50eddc8a72590db2d7ef80314084c1d0571192e5586cc6336'),
    'kernel.fp8_m_grouped_gemm.65dad77228df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '74359a56c9fd1e5825f881ceb1251dfc7f9e4075252426d766c222129728ff1d'),
    'kernel.fp8_m_grouped_gemm.3d80bde34f4d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7c48ba84eead3c0cac4774d72a0c0459632450b6da3904450787b5b0b6cd6689'),
    'kernel.fp8_m_grouped_gemm.13405f117b95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cd2751843c5d6b9a565afa99a8456eb1d1f53f161b7a3241da9c17ecbf161993'),
    'kernel.fp8_m_grouped_gemm.df8b3af58001': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '31cc6aca19ae1021d1dba491aaa63ee8c98ea0f075aca80b9f921979b65ebf8c'),
    'kernel.fp8_m_grouped_gemm.b6ea4e9c1ee9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b39f2ab3e40c155de44d3acb04ee918626a56d976041197e9a76e114f63735aa'),
    'kernel.fp8_m_grouped_gemm.49778ac9c667': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'caeadd19c083a2b67ede039a26607848cbd7a3e645e16871b6a528058ade24ce'),
    'kernel.fp8_m_grouped_gemm.01abe8ac39f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '05061cdd734fbc363d4f0263129e5b588bed3672cdd702d10341a0c5d898c0c0'),
    'kernel.fp8_m_grouped_gemm.67a38a8a7b2b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c0fe2145a1f2bb78329c3d257a624d792b05c986c9fadc37c02a54b037ba8fdf'),
    'kernel.fp8_m_grouped_gemm.f83010d0a78d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5c63040a083656bcb37ff4e11a2f5bfe39c12a77097fb5e14f6a2f942195a942'),
    'kernel.fp8_m_grouped_gemm.d3ef2db30620': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6335583575b2725cb5a2b144019e39580e5a1e5de6b33aadae6ac0c831433c9f'),
    'kernel.fp8_m_grouped_gemm.7ebbe8c3fc4d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4e02faee4070e2d70fffc904976773ec63348801b6156bb94c646ad0abde8bc8'),
    'kernel.fp8_m_grouped_gemm.4db0a0684aa2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8b35a3ff7fb5bce8c8873ec7fe1d6035bab246d222d6804093e288461feb2845'),
    'kernel.fp8_m_grouped_gemm.08051e88e138': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bd4718fa07d2e76eddb55e12040269c878fad546fb8b31a57d23df146a9020ab'),
    'kernel.fp8_m_grouped_gemm.93f36856973c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c50cbf9458fc973da6389043cea556d8cab91157d1aa020e28692f671a983b01'),
    'kernel.fp8_m_grouped_gemm.9ebed43a6023': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '74c313ca0421bb9fc4482f3eaa5c348d5aa8a2ea66e336cdbf64ac5f2c01faaf'),
    'kernel.fp8_m_grouped_gemm.34e2d7ee46b8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bbd30a978bb421bbf3dc6d70a24f5375f3ec0c41845b1cee4bba65ea5fc0ad0b'),
    'kernel.fp8_m_grouped_gemm.ddd5007b9957': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ab5e4759ca9b6a937f578d69a4b2bd4214078e9c0485ae31ff59e55eba899a42'),
    'kernel.fp8_m_grouped_gemm.6cb0320d6d23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '381b36783dfb05cde441461084b9a2a15df414021f8b30cf24c532f165c235ba'),
    'kernel.fp8_m_grouped_gemm.22c8d7120f84': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '25ce7aa06db2a087535e3e558d3b40be2397957c45bbe4d654f0cd50a29eb939'),
    'kernel.fp8_m_grouped_gemm.14831ab6238e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1621bc847239631ddbaedfeac4826760c639f92cb02c87deebdc7579de1c197a'),
    'kernel.fp8_m_grouped_gemm.2b3721bf294c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2126572f900440434c503e108d958e52e4f575d00fdf03d9da60fb4bddef8caf'),
    'kernel.fp8_m_grouped_gemm.640e14f816cf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b50e1ab25246725e4aaf5c9f553c8f8772f6668057f18facb947959011c65ffd'),
    'kernel.fp8_m_grouped_gemm.84f38656586a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '59ebbc12d5f8a002b10443f809f55573d2675f234474cb72e6781821bd129460'),
    'kernel.fp8_m_grouped_gemm.f9851e3c94f2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3b08f62c422de13fdc9ecddecde567ec057130af6432fc815fde0b6c30497812'),
    'kernel.fp8_m_grouped_gemm.75ac829a6832': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '81a923bf85cb23fc378aa1eb05100513c7b8087722780d603fdec8b4ce24e790'),
    'kernel.fp8_m_grouped_gemm.bd9ef159ca1f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7260a2fa0a455fc79c32ebc34bf44c0270ceae5fa09d819acbe98ea1e42817a3'),
    'kernel.fp8_m_grouped_gemm.76383f629fc4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ef235d7e15188b6716748b45f7580beca3ce287343f7692b6d4c1a929399ed9b'),
    'kernel.fp8_m_grouped_gemm.d5046aeaf1a6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '332e7790232338ed438322601a31782180876f0c382515b1b2d2fda6a7f3fa57'),
    'kernel.fp8_m_grouped_gemm.166e3fa97b55': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '79b21460009f9b0dcb2327830a0d085df8b6f978949d8b1b314c38f540724989'),
    'kernel.fp8_m_grouped_gemm.89f63ec7b8ed': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'de25d0e47d66b0556d43debb8cda3c0587c312cb2d1f114a8e3f2b9699cb22ad'),
    'kernel.fp8_m_grouped_gemm.933e8ff9de4f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '781d63f7aa32e5c04736e9023b94ced9edc9550dde1fecc8a55ad3dd7224d395'),
    'kernel.fp8_m_grouped_gemm.d84c0323cd7a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a1660424ebe7da6f93cfd2b76aeafc12f886e3f2ce9fe48ae39bf8a63670088c'),
    'kernel.fp8_m_grouped_gemm.ca96166e620d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ffa51697a5f4c2dac63b6f25a496124ca08bf322b8419e1100389be9f9bf0355'),
    'kernel.fp8_m_grouped_gemm.b53f23fe8b35': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ee244d1f5a6e926b124a5d5cc1a7cf420cb8458c92991691472cc966a7e8c35f'),
    'kernel.fp8_m_grouped_gemm.b631d84ad228': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '159a0cfc8f55e95f5924dc5e1f69abc15f19e3efe2fe37602a6b0da1f31bb678'),
    'kernel.fp8_m_grouped_gemm.9d32a431cf70': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f8234f2b208eedd357feb2b4fa21ba7c3fbcc85fcfcb3cdd955d57e4b9195421'),
    'kernel.fp8_m_grouped_gemm.12ff4c6e55ba': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fef41e024cd964b2a57a1b1a519de759c98c84e3c9f519ba518514c1e5ee3fff'),
    'kernel.fp8_m_grouped_gemm.adb182190e42': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ccdb0b3c3bbd2e45ebb9fa3259543d913f952d3fa7bbe12921ceb23235fd78d0'),
    'kernel.fp8_m_grouped_gemm.65c649fb73ca': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '935b0ff4828e216ec677cb9dabe0d1d060865a82719019b521924af56c19dce5'),
    'kernel.fp8_m_grouped_gemm.e11d8197ff5a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f0d797c00b9c75d32f32d529391e459b87c9543799a0f15fd4ed345914ac46d5'),
    'kernel.fp8_m_grouped_gemm.812550444fe4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'dbb951e40f65eb9d6b7339773ff830be14462f868e5e3f02c305cfacbd4a98be'),
    'kernel.fp8_m_grouped_gemm.4caa3b87e72c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3cad3a50c3e441e99b0f0443458b3253b23d8af22c4dceb89d327882c8132f2c'),
    'kernel.fp8_m_grouped_gemm.7953ee0a470b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1376628654c3ce22fb03a4428a395ee7a190b486d567813d0391163cc556fc2f'),
    'kernel.fp8_m_grouped_gemm.c57637209e85': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fce716516c9f48abfdc199009eeefa2d8cfb8297959e17bef8c6821cdd12ab5c'),
    'kernel.fp8_m_grouped_gemm.03104cf927df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fccc915d3d4604a7f57747b8336433186427e919688562ad6a94b39744a3c9c9'),
    'kernel.fp8_m_grouped_gemm.be55ed92eabd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '140771f10750b0e66f0ff1c0e708ee33732f169e149c06224e7a64996eefb5d5'),
    'kernel.fp8_m_grouped_gemm.f0800daaacd4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'dfc89ed76f5009e8975fa6e4ceb4292c75972b171aced8c9bc3cea39202ba282'),
    'kernel.fp8_m_grouped_gemm.e2987a4d4273': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '84a92b7e3af5cf3e37b57656159362e779b89c6f94c5ce8855344e0282949178'),
    'kernel.fp8_m_grouped_gemm.9b9742c0b9b2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9ce162b63d49de2cc7271095511f8aefb763b57e3c5446777f656e39f1750099'),
    'kernel.fp8_m_grouped_gemm.5d83408ad0a0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1dbafe8f7d4d6ff1414910e20607534adfbb083c8b6d4203c2831d3bbb3a06c2'),
    'kernel.fp8_m_grouped_gemm.9d265f01caa7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7783fa8e00f557a95f066e88611cf76a26983432e827064a27be42a49dd61deb'),
    'kernel.fp8_m_grouped_gemm.a3687807b5db': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cea1e9e73c148d3329ce32d0597c7362013f37f4283894d95e1dd2628ffd986d'),
    'kernel.fp8_m_grouped_gemm.75ef51830e08': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f13ae2339ebc8446fff0d42833dd5210851372d514b5d244e6645b2e466fd929'),
    'kernel.fp8_m_grouped_gemm.c4b0089929a7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd224aab02334802c3cd78e51c7681610cf873f64ffb2b2ced21239fb61b83b04'),
    'kernel.fp8_m_grouped_gemm.65dad77228df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1541c340dece3006451849e0aa96596cc25fa7fc6776804d57c4311a48139625'),
    'kernel.fp8_m_grouped_gemm.f61444ba7046': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f82ad41ec4f0076f9488586d53b53d362b52085806e171814ecf8623c04d39f6'),
    'kernel.fp8_m_grouped_gemm.13405f117b95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '128615e8be2021e3eb0a8b41342e9ec4619c04be4b2c251421eb2a776fbc388c'),
    'kernel.fp8_m_grouped_gemm.5273eaff13b8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0ef1bd806e5bb0fd38f16a93a0e8954978ad997cd574b3f9f6e7ee2511eab3f6'),
    'kernel.fp8_m_grouped_gemm.eabd2207ac17': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9077bd619db85d403efd66a2f8f4a7a98f711cf6b8411d81870d96cfb838494b'),
    'kernel.fp8_m_grouped_gemm.49778ac9c667': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '71f9a3fa7d0903cb279077234d517de28f841cdcb4fda791b62dcabe6a6cd126'),
    'kernel.fp8_m_grouped_gemm.01abe8ac39f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f6fe3a1065675a569a72d7ca73cb91568e917511006b74e748c4fd6ea1760bf3'),
    'kernel.fp8_m_grouped_gemm.c619d155cb8f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b19c0d1e33d3c3978e80a5ceaf99e9afd018972b8e6e79221bb17f862c66bd9c'),
    'kernel.fp8_m_grouped_gemm.c2419a16cb48': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f1062de88bdb0e6ba37f9e881bb4cc3872e615c2610e519786197ef577202193'),
    'kernel.fp8_m_grouped_gemm.9f06bdc5289e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '55d009705dd111feee5cb31c8e5ec536e75864ad132f553811b0fcaf89c53e84'),
    'kernel.fp8_m_grouped_gemm.fc15347f796e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ff75a9e2ab39b764cc0e4873e8ed9efd8ef88590ce7bb66f196afd9eb29963d0'),
    'kernel.fp8_m_grouped_gemm.4db0a0684aa2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3b891dc14fea1368c2e255ed3545d1f261fbef40dc684a4a56b1358493c6d785'),
    'kernel.fp8_m_grouped_gemm.08051e88e138': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f183a4cad37423f3a80a14d0cc3102e06691efd8fa4f0ed54e9fb652c5e0ba88'),
    'kernel.fp8_m_grouped_gemm.6c3a266346e7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3992ead13bccb5a1fbaa1abf0b7ff4a9fcb84607ed70b3b6a1504ed4909f59da'),
    'kernel.fp8_m_grouped_gemm.f70058b01f81': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6d42930df06d940cbfe9e16e079a2db64a905a84eaaef379ae4c251250bd2ad8'),
    'kernel.fp8_m_grouped_gemm.4166ff1d78fc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ef11f9df164879506f0d5de233da45de0fcd419aa52c30b7f1a5e0b08fbe91a4'),
    'kernel.fp8_m_grouped_gemm.223efdcae644': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e933718119dac120939ec4fb5bb8f8c40c624ade0f2cd1caece2075da1950012'),
    'kernel.fp8_m_grouped_gemm.6cb0320d6d23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'dacc1e4ac6580bb2835e0183f0e0de00ef8daad12cac307a8874c2b1d5e9e3b3'),
    'kernel.fp8_m_grouped_gemm.22c8d7120f84': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e5dfd781d964ea57c856da960fcc68d6cecbbb893eb56c60dd321b45f777ed65'),
    'kernel.fp8_m_grouped_gemm.27791478ab00': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0e1a075146fcccab99295614bcb2f9980b77170daaae24bebf460ba912020778'),
    'kernel.fp8_m_grouped_gemm.41dea87fd6be': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8181071953b34f59ccae4a90dc4d99badf912430242366b7675bb7842ed3a7a4'),
    'kernel.fp8_m_grouped_gemm.6e76205d15c3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2b74a1b0debd54fe2cb6683ed628201106118b8e20462b72f9b678fe3b26f973'),
    'kernel.fp8_m_grouped_gemm.de75d095d244': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd053dd6d27472ad700fb28dd1a800d28d9f078359c7a61c3a0ac21700f7f643f'),
    'kernel.fp8_m_grouped_gemm.ff60a177be62': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0cec5b99f0fff866a1f124f082ad9d122e652eb356048abdffe633104809fa77'),
    'kernel.fp8_m_grouped_gemm.27edf09cd64c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9d62c749a051b3be770bfe1d7c7e2940a34b5b8d5387d77e0dd18741e9b012ab'),
    'kernel.fp8_m_grouped_gemm.bd9ef159ca1f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f56dbd198db6cc66d07197af41cc7ff1a04f5b259bdabbf0033524cceb75bd3a'),
    'kernel.fp8_m_grouped_gemm.15037c26bff2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1f0d63673febc607edc3f709229578d13a6c1ce041ea0e1bc7d9d3bbfc7dbde1'),
    'kernel.fp8_m_grouped_gemm.d5046aeaf1a6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '25d32004050b4de768c1d8c280586eb831a9ab3fa1bffa716a99e13a0a67723b'),
    'kernel.fp8_m_grouped_gemm.166e3fa97b55': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '84e34ed4ae9a4d48f9988a9ec4a9e30ae51777f57e72b768439daa95efeccf95'),
    'kernel.fp8_m_grouped_gemm.3bf99a5d7f8d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0353a378804ab7e180185cb0bbd5727ccaa70c6e25d2019769bca587665c9f5e'),
    'kernel.fp8_m_grouped_gemm.468c865ef8be': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '73501ebe05599d4e4e541ac62ee935a5cb081c7176252f5809b0ac2b6c6e86b9'),
    'kernel.fp8_m_grouped_gemm.d84c0323cd7a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '83dc74c02d74b74b4e416433f13333f9dd2bf478372b13416c6983bea3e5f1d6'),
    'kernel.fp8_m_grouped_gemm.0d364e3996d8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f0237959a99d449204f4431ac04d1342f19cff635427b31ff962ddb785f009a2'),
    'kernel.fp8_m_grouped_gemm.b53f23fe8b35': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b7adde69ac1c99225a2dd8270c0b400121cac10427171a745464b169c5aa9467'),
    'kernel.fp8_m_grouped_gemm.b631d84ad228': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8e4b539a765d4e0b05b37cfd94335ea711ad65daf61a7b2d08dffac2c46aca1a'),
    'kernel.fp8_m_grouped_gemm.75319cb78bd1': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4487185d92187285f77515ad1133504410601eeea9b38a46c99eb2298501b503'),
    'kernel.fp8_m_grouped_gemm.dc984489a357': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '34fd4b1e89a97629a3cee4975ca9b6d301fa5c89dd56b06639c7f48914bad3bc'),
    'kernel.fp8_m_grouped_gemm.adb182190e42': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5b3e81172bdb5e93d8f5dab94b59a9fbe402b5a28085564edb1e139966e72cec'),
    'kernel.fp8_m_grouped_gemm.a23db73abcec': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a6cb18bd817d90f91844b214cf7e27ef61c6e4859a892db5474ad53d9564ee9d'),
    'kernel.fp8_m_grouped_gemm.e11d8197ff5a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7178b563f4add9bb49bfdb702effae99814ee24df23772877e2c7cefd9a7f5ab'),
    'kernel.fp8_m_grouped_gemm.812550444fe4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3bcebc2425800e987ad97db535741abb5e12ccc99258bb4ce6c4ee771b99a5f7'),
}
# fmt: on
