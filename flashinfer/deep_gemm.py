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
    'kernel.fp8_m_grouped_gemm.4caa3b87e72c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c43980becee2933157be3970da33f9086dc3357406665ab86c62c66fdac51bc6'),
    'kernel.fp8_m_grouped_gemm.7953ee0a470b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '910fe4d73e6285be839bd9ae760815ea1caa0912e937ad9b1a75fca6bb77d47f'),
    'kernel.fp8_m_grouped_gemm.43a66a194af2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3e159e3d742cf6ce51db209bd115c3f3128fde632fc5b2866caf9e6c2d926050'),
    'kernel.fp8_m_grouped_gemm.03104cf927df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1bcee8e16f9ac0095c70948bd424aa021aa16472d66792fcc81e842474a82d7d'),
    'kernel.fp8_m_grouped_gemm.d8caec6f111b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd0373ee4c6c6481620df57f903d3b4f2c082c4a020110ae542e2291f7c448361'),
    'kernel.fp8_m_grouped_gemm.9e474c44622e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '09a11fd4f34cae2628a6a840e2117fa2f21675c2ba091802111a2800e4e5cb37'),
    'kernel.fp8_m_grouped_gemm.e2987a4d4273': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3639765fc6e0565bdd6f4fe3107c575069d0f7955c2feffac2b25cd2c289d7c3'),
    'kernel.fp8_m_grouped_gemm.9b9742c0b9b2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0299bebf7c98619ed02564ced80a135c0a39f390b36b5a5dd88f64a49feb8885'),
    'kernel.fp8_m_grouped_gemm.116f46018c23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7f162b1d42e2b37a4ceaf771c7bb5afaf8b7530ac45a1001b6b5e471b64506e2'),
    'kernel.fp8_m_grouped_gemm.9d265f01caa7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '27462e1b438623e6e5a97baabcbe5513a88691fddd831da8b128d6e29b8f1499'),
    'kernel.fp8_m_grouped_gemm.a918127d1773': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ebc0d2b1c8141e91ad2a576fecf9536d15957e80db8f3e300fea118dc20e3561'),
    'kernel.fp8_m_grouped_gemm.d709904c7313': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a8e560bc0bd63fcb40a8b78323910fd416c246e25a456e47c0b2e9e22bd57b52'),
    'kernel.fp8_m_grouped_gemm.c4b0089929a7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ac61ddedc2b7fedbdf8711b3473deefc96eec330807d0f500c9c564e20297cea'),
    'kernel.fp8_m_grouped_gemm.65dad77228df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3298bb071f6f84f1fae40e2294f7a7d525e0abc9e68ed4ba65984174698a4e03'),
    'kernel.fp8_m_grouped_gemm.3d80bde34f4d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8f8f6fefe39344da5bb4d85a9758015246b71887324eecd00efece259932ced5'),
    'kernel.fp8_m_grouped_gemm.13405f117b95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6ba7cf0aa4eb2aec6424a9da4df01a64d1a974e95a0adb6986e316022bd7ad05'),
    'kernel.fp8_m_grouped_gemm.df8b3af58001': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '03bda6cf88ee3ae8bc25f41c193f4bb09277333fc0b7a480bd73667d215a026f'),
    'kernel.fp8_m_grouped_gemm.b6ea4e9c1ee9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '82b52d0ca02ab4c1db6f85e22fef7fbd08ff4dee664fa6bad18c06e55565e639'),
    'kernel.fp8_m_grouped_gemm.49778ac9c667': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '10b7142e51e5493c81b4c48bc920462246a8720f339709414e75a50b3aea3d60'),
    'kernel.fp8_m_grouped_gemm.01abe8ac39f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4194d1d1f313112cadefea31a9517f7d08862880de6a6a9cfa9759b4970e40ea'),
    'kernel.fp8_m_grouped_gemm.67a38a8a7b2b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a44b5112b3ca5e81cfb4fe4e125e1236729629f86712d144ccad605458cf72bc'),
    'kernel.fp8_m_grouped_gemm.f83010d0a78d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '78ebbeeba75439983fa143076429537975be53552b29ee5fdf64c53b88dffa36'),
    'kernel.fp8_m_grouped_gemm.d3ef2db30620': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '74a592b15358f1a4f3f80f9abd09c56b6dd7120acddd36a9a56431cae1fb2b7f'),
    'kernel.fp8_m_grouped_gemm.7ebbe8c3fc4d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd256ae47bee4d0ee26b91598fc1e3680ed84fffaabae111591be6db00fd09604'),
    'kernel.fp8_m_grouped_gemm.4db0a0684aa2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '37aa095a3ab9a714cddfbec50615e536f658f0cf39f9521e49772deef214b067'),
    'kernel.fp8_m_grouped_gemm.08051e88e138': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7da587174c24b0ea089de98c2b5855786c504a995c63f365c3cec2c651ce499a'),
    'kernel.fp8_m_grouped_gemm.93f36856973c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fc7cf572d1ec99515617a8ff473ccd8051b4928377c06840b2ff631fc78eeabc'),
    'kernel.fp8_m_grouped_gemm.9ebed43a6023': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8e8be6b6f536fac662c26e1226b235b22c8210ba329d6e717e098d4c6dd9c78f'),
    'kernel.fp8_m_grouped_gemm.34e2d7ee46b8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '21506afcb761b13ac3427974dfc90bdae547da79db8925d12ec5b0a915a628f8'),
    'kernel.fp8_m_grouped_gemm.ddd5007b9957': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9423e1b9a6118cd5b6735134cabf03529af9575552c2118565f34a19c47d159f'),
    'kernel.fp8_m_grouped_gemm.6cb0320d6d23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '92ba8f15343c1e56029f8b0fcfc5f7f781ca6abfd404b54eb5d673c958dde8a8'),
    'kernel.fp8_m_grouped_gemm.22c8d7120f84': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c4a1b3b76871d2080eb523a3ebca10487b827b6bc7a818b2c95889b44cc99e8b'),
    'kernel.fp8_m_grouped_gemm.14831ab6238e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '174b447fabf2191627124d33a45117365ae5d5da02ebc321d045f76cfad6f053'),
    'kernel.fp8_m_grouped_gemm.2b3721bf294c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '04c09d99e8acd956f5ad03404604f942ec4c0887b95338862e4c5eb872d43020'),
    'kernel.fp8_m_grouped_gemm.640e14f816cf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bbb958a2da6a40df345fe15d59c09a31ed4d463bce090d41300448fa36e1d001'),
    'kernel.fp8_m_grouped_gemm.84f38656586a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd71e5af3745a9c038eb1925345ee358ac1bc952afd8deab31aa38cc298051b0b'),
    'kernel.fp8_m_grouped_gemm.f9851e3c94f2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b61353cb15bb46783a9394ba476fe73d806e397717e5f820597fb919a5c1b9ee'),
    'kernel.fp8_m_grouped_gemm.75ac829a6832': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '14fcfef388d4c18482d10ab39e5087447e96959586cd073e8c3482206949544b'),
    'kernel.fp8_m_grouped_gemm.bd9ef159ca1f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f4cc0501ef26e11227f9e9727cf6f7a245600abd7d6bc59a822decc2fe158de9'),
    'kernel.fp8_m_grouped_gemm.76383f629fc4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5f2e82c05544bda7b2c4714f43254cfb6af67bc1dd1128255c2f956d7e31423d'),
    'kernel.fp8_m_grouped_gemm.d5046aeaf1a6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd6b6b8bc9a8a8a7f080916e1d4c2070060d98da39f5af84676293119d947832f'),
    'kernel.fp8_m_grouped_gemm.166e3fa97b55': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd8a86d15e6cfddb15b4a983ff089ac5b100111d69c151f0790fafb44854ec474'),
    'kernel.fp8_m_grouped_gemm.89f63ec7b8ed': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6adf70303b6c942597b1a70e191e4c56488a6f516803f72abad1da31033feb84'),
    'kernel.fp8_m_grouped_gemm.933e8ff9de4f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f217b507f8e50019192bcd9832865584713bf1384b98fba25cb234c05324344c'),
    'kernel.fp8_m_grouped_gemm.d84c0323cd7a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3a3163be606a15a85da7ed6e27a29e54e260a0ff988ffe40ecae6dff5fd06915'),
    'kernel.fp8_m_grouped_gemm.ca96166e620d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0a7178c17264a461aa60f7df4b45f2cd36259216b53b36db10c934b0aa71618d'),
    'kernel.fp8_m_grouped_gemm.b53f23fe8b35': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7025ee85fcea01f093a49face271b7763e128af9abbf56c45817fddd7d121cf7'),
    'kernel.fp8_m_grouped_gemm.b631d84ad228': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd24848f8f456961e2e43d59918c881e67f875edf31cc3dc02fbc52e4c9c021e5'),
    'kernel.fp8_m_grouped_gemm.9d32a431cf70': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'de15347ab257b7356971ab0a9aba7bcd4a07fda13725a3053bded7af136ded69'),
    'kernel.fp8_m_grouped_gemm.12ff4c6e55ba': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3445ca0f192278c22c0a171fe4a6c224d8999bcde7e65a11c4ce5ab796a11810'),
    'kernel.fp8_m_grouped_gemm.adb182190e42': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '648ff8fd06b04d0cd99432bc7405ffa7c24ac02f2fe3fd59c1a08f379150488e'),
    'kernel.fp8_m_grouped_gemm.65c649fb73ca': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0d3649f5808813d9cc1a1cbda3f9990c75de50cbbfb2ae820355fdf19042ecb9'),
    'kernel.fp8_m_grouped_gemm.e11d8197ff5a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bd9be68163bedfb1303b1302b652da246d79cdcc491a7b506427be6cfd45a0f1'),
    'kernel.fp8_m_grouped_gemm.812550444fe4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fa5af65fa20737bbc6aeb49b0dc1609f25780a70cc1e6ffcd5dd48721c79c64d'),
    'kernel.fp8_m_grouped_gemm.4caa3b87e72c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f94f6a7e58f5694f7f19d3c7573f98bb4eb3352569341f2a3a9242a294a2f6c1'),
    'kernel.fp8_m_grouped_gemm.7953ee0a470b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '35267ec4049cf286e5f035b42dbbf90642395ca017b538d5f4f369f4c784da6e'),
    'kernel.fp8_m_grouped_gemm.c57637209e85': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ed41f47889c74662e2c3477d63a75d0dcf0113879627eb56651ab856a967c67d'),
    'kernel.fp8_m_grouped_gemm.03104cf927df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '50b565f01e0f3e0cfbd2b53be97d86516c8e8a27bed1c2c71f67f45b08256633'),
    'kernel.fp8_m_grouped_gemm.be55ed92eabd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '96cbb73f0d787e30da13e0ece0b5175b9184c1ea5c5cc6968571766692abce3f'),
    'kernel.fp8_m_grouped_gemm.f0800daaacd4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f6692e377e3bd1f416118a04c12a6404c35f78845b57578564f69307d9599386'),
    'kernel.fp8_m_grouped_gemm.e2987a4d4273': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3f48765b1eedb2989e0f75e36edd34e25af6bfb14bf807fe407fa0dca40eb16b'),
    'kernel.fp8_m_grouped_gemm.9b9742c0b9b2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4a6a81b7bff59303e176897790639cb7aa22f319b3e0b84db0d2f9172bb7fd48'),
    'kernel.fp8_m_grouped_gemm.5d83408ad0a0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '459eb640b2ab0cd64eaf4a2cbd3977a0c151c2dbdfa8eabcb0759a1c205ed9b2'),
    'kernel.fp8_m_grouped_gemm.9d265f01caa7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3e0bc7fb7852cc20a951becfc7e8a2cc11bf7cf2c8f6b1abdb91dc6259891395'),
    'kernel.fp8_m_grouped_gemm.a3687807b5db': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '134e0ef3289d5ff22e0e1b3a15270796a3edf7918bf3fae7a925c9a1167df35b'),
    'kernel.fp8_m_grouped_gemm.75ef51830e08': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f3acc78df88ac8fd7e95b3e51e14e2781337820fa30bf85fd26878af9706ac85'),
    'kernel.fp8_m_grouped_gemm.c4b0089929a7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7eebbd97345e894d93ed6abfabe4d1356ffb30f7a2ac444ff2299c59a2d9eefe'),
    'kernel.fp8_m_grouped_gemm.65dad77228df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '55e991a1c67e2c76c98b95e735b67815518f06b8b852cf865ce25e1771252c75'),
    'kernel.fp8_m_grouped_gemm.f61444ba7046': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '573a4cb9016ef5a7855d7e7ed535092737a97fd62ea6cfe38b3286f8d853de60'),
    'kernel.fp8_m_grouped_gemm.13405f117b95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '36ce7ed8a0aa7c8345c7301272a5a5871ae51c4012eebc9608ea1f34f6cf59b6'),
    'kernel.fp8_m_grouped_gemm.5273eaff13b8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd85bb4c8680ebea5cd572379816066877ef019e36fc6bd802fb73ee65209f34d'),
    'kernel.fp8_m_grouped_gemm.eabd2207ac17': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e6b3a657126ec14e1438b547da8eabbf038837f19ee36a38bbaecea392d3419c'),
    'kernel.fp8_m_grouped_gemm.49778ac9c667': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8e83fcbe6859544e927142b5d1ebe02f154dd41a0d1dbf0ec8a7344f6a13cdc7'),
    'kernel.fp8_m_grouped_gemm.01abe8ac39f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '127152dc2889e91c3b87ff4462811a101ad6a75cc84cccac9c2face383e73fe2'),
    'kernel.fp8_m_grouped_gemm.c619d155cb8f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '23dd46296191b2dabe0b8c167b8b85e8c88904d86e8f8070e88da644b32be660'),
    'kernel.fp8_m_grouped_gemm.c2419a16cb48': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e37b816d2c8fd2796c0480b5674271127ed30caaac7d1dae53f6cea86e32b6e4'),
    'kernel.fp8_m_grouped_gemm.9f06bdc5289e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2278f4507d92dc0f2cb973728b9aa32d5881f373eef23a0634cce2a374e9de9c'),
    'kernel.fp8_m_grouped_gemm.fc15347f796e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'aba2c66e9656c2c8a01173413344e3131cd8c6d5acd7500ab8e753793a471685'),
    'kernel.fp8_m_grouped_gemm.4db0a0684aa2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a84a8864c5b9c719e323e842d45c3c6620a1155944961859e74c8a72a24ebdc9'),
    'kernel.fp8_m_grouped_gemm.08051e88e138': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a09d0ea7a4ac7d7193c92d1280d0a066afeccd5449c7c6bd56b561b3c609c3b3'),
    'kernel.fp8_m_grouped_gemm.6c3a266346e7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0918729d0c8700bd50cae4aac7a640340e4daf35f699eebe74e5023a071f31d2'),
    'kernel.fp8_m_grouped_gemm.f70058b01f81': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a48d9a46b60135691eecedaf3f4d0326deac87d2802729df0078c2819ab289af'),
    'kernel.fp8_m_grouped_gemm.4166ff1d78fc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd19bfd9ebb65e70a48c30dce2b62f98a9ac938465deb102bf30aa63f5d2d4c15'),
    'kernel.fp8_m_grouped_gemm.223efdcae644': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f0030a67de452f90a1ca4c93fbb9c865cfff1b0209c76b23abb3a83168747067'),
    'kernel.fp8_m_grouped_gemm.6cb0320d6d23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6851a769b91d86bd0ed617a2bfbc832e94f38ee67682ce031ef0ea5bbbd4e468'),
    'kernel.fp8_m_grouped_gemm.22c8d7120f84': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'db6a25dc49d5b0d1a6d024e08dcd02a2f6cec1485a4414123a2e41998d806ce1'),
    'kernel.fp8_m_grouped_gemm.27791478ab00': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '369d3e0a73bc3bd88ab1bb8403c2370c2b36dc81cb49c6905aecfd5ba2fdaead'),
    'kernel.fp8_m_grouped_gemm.41dea87fd6be': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7e8dcd9e2ad7159b35216e28177e3310e1974f7533f301db7ec03d7e017cd7bc'),
    'kernel.fp8_m_grouped_gemm.6e76205d15c3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9429daabfc96d1d5aeffa1296e16a5cc979862a00bf04f498183ea3310ccad74'),
    'kernel.fp8_m_grouped_gemm.de75d095d244': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a19a5460f715baf4a9dfda59be9572b135697d52eb2189104bc1dcf9454b9def'),
    'kernel.fp8_m_grouped_gemm.ff60a177be62': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5f824b612ab2c9a32c6a5ff70cbad373f092d604866a1c7f8e98c540d77893fa'),
    'kernel.fp8_m_grouped_gemm.27edf09cd64c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a97467e4ccd01eab7b10a7f7b8b6a52530c9b200d3fd0d813a7cda899bea70de'),
    'kernel.fp8_m_grouped_gemm.bd9ef159ca1f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2502e6c81ed6970a3b7b0f2616379df6828e58f6533483312a22b3c8f970cf7a'),
    'kernel.fp8_m_grouped_gemm.15037c26bff2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8cb0a2d5569fe32ffd43997eeadb38cead4b46e90e2ea6a16aac569371cf44d8'),
    'kernel.fp8_m_grouped_gemm.d5046aeaf1a6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '81864f4d9f6a6491ab7f6740e85bbc292b94f32aa89f8ff550c41ccae445cdd0'),
    'kernel.fp8_m_grouped_gemm.166e3fa97b55': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3b6f692ae54fef0d1a91f518f47e36464ac77c9e570e2d300742bb33cf7b2dc3'),
    'kernel.fp8_m_grouped_gemm.3bf99a5d7f8d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'eb690b8115b134f6dd5bff8e53f5a7324f9fdc7e764c71da53a90a1183ce80d0'),
    'kernel.fp8_m_grouped_gemm.468c865ef8be': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a9c682cdce0a7b6d4d0709b67874b8c25f3d9fd2efa2f535b245afac88515ef8'),
    'kernel.fp8_m_grouped_gemm.d84c0323cd7a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2a8bc66d6ed309d34d1c590861da00ec6e1873e883db9ad3336b1fa110406f05'),
    'kernel.fp8_m_grouped_gemm.0d364e3996d8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '540c92617d5747cd0f0b3a8c1e1a485440696a3278db492133582cc2d22bd99d'),
    'kernel.fp8_m_grouped_gemm.b53f23fe8b35': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'dae4cfe2ba2a60e35c44b6fd29260d7e091b5569cdf1010a71011f048f0150d4'),
    'kernel.fp8_m_grouped_gemm.b631d84ad228': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '03487ef48fb1a567cfb05ae7a158e03df788d685bebffc2885f3993661dc7f04'),
    'kernel.fp8_m_grouped_gemm.75319cb78bd1': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '85564cdba21bfb616cec3d62122082a07bf437c118385be6b3f125a2f6c10784'),
    'kernel.fp8_m_grouped_gemm.dc984489a357': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7a8516d86013fe319a82b35667811f8f3e27bac407b65523eed7e7501b56d961'),
    'kernel.fp8_m_grouped_gemm.adb182190e42': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4c91e1932a3d58e5b203b4e86da35ac80f18655712834599f13cc87d723ff099'),
    'kernel.fp8_m_grouped_gemm.a23db73abcec': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '54bdf4f1c45778ec760a0d1d632fcf4eddd79d34a5284f6c95a14d14e45169a5'),
    'kernel.fp8_m_grouped_gemm.e11d8197ff5a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '41cfac351e0bb36b3a94c19777c91099985f0f23ca2d1bcc5a8e865c2b97cd72'),
    'kernel.fp8_m_grouped_gemm.812550444fe4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'aa561777aacb4e97a64acdddbe4d98a1f94e8d6a50643a468ef276738f9e383d'),
}
# fmt: on
