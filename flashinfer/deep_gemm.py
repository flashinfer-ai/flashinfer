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
            try:
                checkCudaErrors(self._cleanup_func(self.lib))
            except:
                # Ignore any errors during shutdown
                pass

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


_artifact_hash = "3e5c4fb4cedaa757da61afcf5e3b94ebec33c08f"


def load_all():
    for cubin_name in KERNEL_MAP:
        if cubin_name in RUNTIME_CACHE:
            continue
        symbol, sha256 = KERNEL_MAP[cubin_name]
        cubin_prefix = f"{_artifact_hash}/deep-gemm/"
        get_cubin(cubin_prefix + cubin_name, sha256)
        path = FLASHINFER_CACHE_DIR / "cubins" / f"{cubin_prefix + cubin_name}.cubin"
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
    cubin_prefix = f"{_artifact_hash}/deep-gemm/"
    get_cubin(cubin_prefix + cubin_name, sha256)
    path = FLASHINFER_CACHE_DIR / "cubins" / f"{cubin_prefix + cubin_name}.cubin"
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
        num_sms,
        block_m,
        block_n,
        block_k,
        num_stages,
        multicast_config,
        smem_config,
    ), static_kwargs = m_grouped_fp8_gemm_nt_masked_static_kwargs_gen(
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


def m_grouped_fp8_gemm_nt_masked_sm100(
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


def m_grouped_fp8_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
) -> None:
    # Compiled dims can be upper cases
    compiled_dims = compiled_dims.lower()

    # NOTES: shape must be `[G, M, K] @ [G, N, K].mT`
    major_a = get_major_type_ab(a[0])
    major_b = get_major_type_ab(b[0])
    assert major_a == major_b == MajorTypeAB.KMajor
    assert masked_m.is_contiguous()

    a, sfa = a
    b, sfb = b
    num_groups, m, k = a.shape
    num_groups_, n, k_ = b.shape
    num_groups__, m_, n_ = d.shape
    num_groups___ = masked_m.numel()

    # Type and shape checks
    assert num_groups == num_groups_ == num_groups__ == num_groups___
    assert m == m_ and n == n_ and k == k_
    assert expected_m > 0 and m > 0 and n > 0 and k > 0 and num_groups > 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    assert d.dtype == torch.bfloat16
    assert masked_m.dtype == torch.int32

    # D must be N-major
    assert get_major_type_cd(d) == MajorTypeCD.NMajor

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
            m_grouped_fp8_gemm_nt_masked_sm100,
            major_a=major_a,
            major_b=major_b,
            compiled_dims=compiled_dims,
        )
    }[get_device_arch()]
    impl(a, sfa, b, sfb, d, masked_m, expected_m)


# fmt: off
KERNEL_MAP = {
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '26dc7248d209d055ff6a1d9d64c08bb050501a2373fd3d5384d0fe0d4444b0b9'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd54b50c2dde248023395d3b60c66256e1ca42e29bb11a40a4e493bddce1b47b3'),
    'kernel.fp8_m_grouped_gemm.b6fa3fbb084c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6ab61ca3aa83de8c427821d436156f57a9f4d8e0b7a43b95879a2dbf9afc120e'),
    'kernel.fp8_m_grouped_gemm.3f09cff85f48': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ed7c22bf04d445dfd3f21fb182004bba304bf90bee28d7159b6227c361a61122'),
    'kernel.fp8_m_grouped_gemm.61a7bb28ad05': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f5c5dc3ff985ad4b346ec4354ae218212089ddf6dcdd86963c80eef3b9b2d80a'),
    'kernel.fp8_m_grouped_gemm.1df8bbc10ccf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj32ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '09b9bbc4021404fe8d9227cba932879ad1fce263094cf05f91ef21f572a9751e'),
    'kernel.fp8_m_grouped_gemm.541eb768d574': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f0714b2b2e1cf8a5332091d37c751b5364b202b962b160131e45bb1f25dc72b0'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3e60c53303a5973a766b24dc9b86b0ee0fb608ba90ac49e9bc2a2aac47d2f9dc'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '38fc792380f99727c658d3a53d07fef012cf09cfea8f24421bf285d44380f486'),
    'kernel.fp8_m_grouped_gemm.089e5ba0ea94': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd085514328f811e47c7b5ab1c9d97a4c159467da4280d3a65e4c8708d889c7f6'),
    'kernel.fp8_m_grouped_gemm.3189defb0c2a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6f8c5140041b63f8599890f5cd2a4f37c5974991c8e7c781b99d3580ce869283'),
    'kernel.fp8_m_grouped_gemm.a35831e5db9f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6cc14b1446f1d637acf800fa9651f9e07cae9dce523c288e200d0cc0c2f92f16'),
    'kernel.fp8_m_grouped_gemm.627fa85cc8f7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj8ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c00514fc3f58eff133d181dcf0e3d4801428f7391d03112ea211625651c85cb1'),
    'kernel.fp8_m_grouped_gemm.6dc3a8170777': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj16ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9d459a1fee76e8250bd8b31296f70a9cb2bed728204fe31c459e9b58fd2c0e2d'),
    'kernel.fp8_m_grouped_gemm.e1168abfb62d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6a0d18b7c24b2c6bea0e5d9ad3618b44b94ab1110dd19b12887195ca0aa53c06'),
    'kernel.fp8_m_grouped_gemm.6ea7c04cd002': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '57eae41bbdb2b749ee74a30cf46a17e7a74d390189bd3c8b7395ab615e1ef0f3'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0bb4eb95b2ca1c389e6c0d0f404ab72e439f0525c4ae33378cf865c6b17ca87b'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c0e792874f4e22bde5293cc898dcfd70a1de8d9b0b0e3d0770e24e75c729d976'),
    'kernel.fp8_m_grouped_gemm.bbff044cc33b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9f2604d59935b718a8fb076ec71a58e2271ecc72b99c28466e41c39a87d00a1c'),
    'kernel.fp8_m_grouped_gemm.2f388090c392': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj64ELj128ELj2ELj128ELj128ELj128ELj7ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3bada86aac46e2ed872ff67895d17fc48234c9f5993b8ca02541850125c21a59'),
    'kernel.fp8_m_grouped_gemm.d5a31ec9e69b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj112ELj128ELj4ELj128ELj128ELj32ELj7ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '14ff0da70ed40de7b88a72221c94ef0375265147db4e5bbcc220a4e7fdf67fe4'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '74a5b87a7c5184ee1a95589f4fefcb3e8cdef386fc593571052d4184c9a386bc'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e6d733092a1ed8be855864e94a70b31eb27376ea8ce55fdfbc3d357181ee925a'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ebd7e50194e59da47b68ba1c5989c12f31ba857122b141fa6630495245c02c2c'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7533d50d2ae62bed2f4c2f766a25b42cb8e428d8d006fe5bde03ac0318317cf2'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a9514d719d23ae6bcf54c1629fcd091632c3fb53341270fadcfb20a08d1f5c20'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'daaed11c0d66e11878f85c982337199a512bbd38206018e2d6e58260c6f21644'),
    'kernel.fp8_m_grouped_gemm.341ee4c63447': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj7ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c161dc17cf7f62ca71c51f7931fe76281f72ce6235135b2bb5b9242488c82ffb'),
    'kernel.fp8_m_grouped_gemm.9cec5c8f75e8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj112ELj128ELj2ELj128ELj128ELj32ELj7ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f64c5158d988de54dfe27765c89d107a2c23c239b7edff02d3230e0e093b9193'),
    'kernel.fp8_m_grouped_gemm.02acb2ba71fd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj4ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '377fcb21f59c78554a8eb1da4a3f2e21a40a7f479b0d1d796e3558e769d936f9'),
    'kernel.fp8_m_grouped_gemm.85fa83498202': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj8ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9326f4b823f9f7735de4f9bfd275bfc87d123543b05ce968146be6b144812e28'),
    'kernel.fp8_m_grouped_gemm.99e2fce99455': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj16ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c0e6f65ae2000dd760e4f4f0a1f514d498e2cf93a0d9dedbcf61338d0032d45c'),
    'kernel.fp8_m_grouped_gemm.dee36a8db939': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj224ELj128ELj32ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6400d5ee38ace1d0da77d914d2a0f317c5df679aa0368d59d2905a03a8d13c66'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f0000f979fa383e04a1aa9a7db912b14136bee87cd976db0aaa67e826e630195'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3df6ff5dc617f726e9e32f81aecdd109bc0499affa68ad61bd9c27b919871231'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9d765045666428321c2d6a4c6b420bb82b08fbb45bd98533748bb101baf1baa4'),
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'dcd9ce82c2a5657de2aaeb824ef60869a859610a48ec46726dfb06806f541604'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '998719a88a8cd7bc8d93e8f70c9ddbd012082183dde88bfea6f748215e2d09fc'),
    'kernel.fp8_m_grouped_gemm.b6fa3fbb084c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '40315741d196732645102948b2f5f5fb63eccf04a4144f6bdcc8ecb5b9c125e9'),
    'kernel.fp8_m_grouped_gemm.3f09cff85f48': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a66352ab39f3ab4339af16220d7e991b4c1bb0e3919e37e5432dd87a3dca2267'),
    'kernel.fp8_m_grouped_gemm.8e0322236f28': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj16ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2c5cc4fd85cbf564314ce3f7b63039469204056718bf77b2d6d1a58f2b4e0a51'),
    'kernel.fp8_m_grouped_gemm.b9069133bf1d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj32ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c8d1d1efe579ff126ae51480f8e6f4ca1e4d249c0c630f1d64fd0b01484b7f15'),
    'kernel.fp8_m_grouped_gemm.ef52dedc695f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '99fed09baf450755878ed9c1ec9be80bb687f88b33c6b1415542979287964376'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e23f17f9dcf2fac2ea48037dc97d2c18dafee8f00fb48164ae591ed715411380'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2255766ffaf6c8a864cd860b3637478a705161cc06f2ee0e9808dcb46ae1349f'),
    'kernel.fp8_m_grouped_gemm.089e5ba0ea94': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '59b85b07bf99664bf250f03fa2f141a04c31a2ff78aab437647f1480a16c8fb7'),
    'kernel.fp8_m_grouped_gemm.3189defb0c2a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ef2cfec091aa9944cacfbe92e84f37ec699235dc04344be2193b5f10f803e876'),
    'kernel.fp8_m_grouped_gemm.4c245858ba8f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2838c78baa5e2cc7ce3ad6161826e6458a600f955ca5ed1ecbc7238b2a496fda'),
    'kernel.fp8_m_grouped_gemm.5c8cfd221abc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a6a3b630e304d8a74ee7b4be2ab3cb0ea05072e9444216db487a50333e0d2b93'),
    'kernel.fp8_m_grouped_gemm.9ee71b05329e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj16ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '343e40ef5cc90f6f521ff859544783394ccd57d27078f29a5ec9f54349dd2bf5'),
    'kernel.fp8_m_grouped_gemm.e1168abfb62d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cd9f47ecb92b76ca8e3bbcfa288fe21f246b7543b0e36a2ba3278e07718f4472'),
    'kernel.fp8_m_grouped_gemm.7236199d5dbc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj64ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0d873f9693da710fc44a8759816e57fa6430b61da3c9ae0ad691202c41cc26e1'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6b9a83d1041cd37d8d2034948aa27816e950a812d4e44cabaf5e689c6623da19'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bd7a3c3e8b000ec59458d2d4d6764f60bd3abd98394748caa31bba2cfbdbab41'),
    'kernel.fp8_m_grouped_gemm.b663e7805eb8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj7ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c344887fa7937d512e325601e5ebcce593e5b75e674b5ea0a205449e1609aa3f'),
    'kernel.fp8_m_grouped_gemm.a1fbc0ab688c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj112ELj128ELj2ELj128ELj128ELj32ELj7ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd9a374499b89a78f7506843d71d332778db46f3f654c1d4ffb23861ed6132829'),
    'kernel.fp8_m_grouped_gemm.1b42e3f6b2cb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cc92cc04d03f18650c32c40904dda5a8b4e50aa32a232d2da79068dc30ee8e30'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e475427a00e940c5f6be7ae3805fa339d46f7e7e8636b5b0041f96a1c33f6522'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '746e54979636c64f2423a13e62975b5089f3423d7370745eb97d5eb362211149'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b1c01347a4cbc1f7fb939048f231f22223b2fd79d0a93449948d564f6f0f4983'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c1e6e748de542eb204021cd78c0311b7baf303f29af268d2aab9adeadd60de33'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '25866b01e89cbd3267601a258c142fb212567559313a68d24be03e773546ab3f'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '18ed68443934624eb6582140cc3b742e98e7491abf5281125ca3370641cb18d7'),
    'kernel.fp8_m_grouped_gemm.226bbeb3c3ed': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj112ELj128ELj1ELj128ELj128ELj32ELj7ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '05e05643ef6cd424c391ca2a14438bf970a08a2f18129138ad51ecc45721e2a3'),
    'kernel.fp8_m_grouped_gemm.0bdfecc9ac6c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9452081116fd63f9112434f822a47fec104b285b9e37d8afe6517620ba519feb'),
    'kernel.fp8_m_grouped_gemm.02acb2ba71fd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj4ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '97abcefb782b1e4142811dff5cd73488bd8b65b662e0b49c4c09da20b84d7e82'),
    'kernel.fp8_m_grouped_gemm.85fa83498202': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj8ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c3eaa2e18172cf5e87d80a7fe04f599a8ce891e126ca59a0de3e29045be36f88'),
    'kernel.fp8_m_grouped_gemm.4382a80fa2fd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj224ELj128ELj16ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd6c75674ac9f67a3f037d89a5b8e314393155a115589a4383c6709e1b9d75acc'),
    'kernel.fp8_m_grouped_gemm.a0469fb341ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '01d9415420d3d7d5c1145862eddc4315521645135261cf5913ca9c0768cc3231'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0d5bcdf5700e7935d6b27a52758e8bd25fb644c9c01702d4a5f5385728060363'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '53d4693e3ddd0eec232d5d73ee85dba2c7eceef0a5dd70cc8e3191b6befd15a3'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cd59bda01c1a34a763fb01bdfaf9e610851570ae3191586bd3817a527b9a441c'),
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ea0dba0c8d5ed4d70ebe5a0e03a93c803f88ffb9a808cdfe676b58ee323ae8eb'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd5a571067cebe933929d0395d07cd17116da8e1fb20826b5c0be5e016ff3f19b'),
    'kernel.fp8_m_grouped_gemm.b6fa3fbb084c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9cd80de0a62cb781d64670d702f3cb83a40329014178c76f0f716e079fde6736'),
    'kernel.fp8_m_grouped_gemm.37b0b4d92f98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj8ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6d5fd79854aec5932d9b36723d29473a635cb7c6a9bd95e6acb4f96bbf83d13c'),
    'kernel.fp8_m_grouped_gemm.29862960604e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj48ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c2a127694a422a2b31a9014b1f1d1467388afca437b1c2ebd52e24c4e15f0100'),
    'kernel.fp8_m_grouped_gemm.d006e1878bcb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c0a8b1729c17dffb063ff890bb0658027a6f33a796d08ad9925e129a612d740e'),
    'kernel.fp8_m_grouped_gemm.ef52dedc695f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fe9a7843776a907ccee9d5ce0132a946a7054647bc918cb3aae838a2b78b66f8'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0714c9df5f9558b8a68fd2407b92657fc64f6780318297b546cffa0e7146576b'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1b728303575d39da6915f1d4b058194b8893139d4c4c5055d0d43c54e8a5ab4c'),
    'kernel.fp8_m_grouped_gemm.089e5ba0ea94': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'af67df6af8a6c1aeaf71f0b6c7aa691f938299f83a9fa81dacd15d258f2d9cff'),
    'kernel.fp8_m_grouped_gemm.c8cf6538b89f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj2ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '88cbf0f7362c089160962cb2248d934053f96e3918e23723f68a1490b9bbc78e'),
    'kernel.fp8_m_grouped_gemm.60960d1f59a8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj48ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '35a2a446764fdfea69c2b86f4b239bdd1133430efdd9e758348482e5354a2e20'),
    'kernel.fp8_m_grouped_gemm.3b740f7db9e3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj96ELj128ELj8ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '67a4c4ebcd6f3324327f3da75bc35e0b621f97db61fa98d7630445cd3dd6d3ef'),
    'kernel.fp8_m_grouped_gemm.a773484fcf60': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj16ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'dd9308de22563343a7638f3a77c93d6a8ba2a3ed49d087d7e0a4ce5a9dc01623'),
    'kernel.fp8_m_grouped_gemm.1a8d7d79fb4e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj32ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '951de1101a3d2997a1b31068687ef23c45f35fd4630f2ea68e68e02c5f85a998'),
    'kernel.fp8_m_grouped_gemm.7236199d5dbc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj64ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '090982813d1ce93e72e777651d7bb3e6ebdd03a84562606e98f7971f8bc802c4'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3f830c999d266b87129ef2a799250c398c54266ddd4abed24f4ffeda19451021'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '921349d9459271f15665ef9a07869e326686ad6d772460f47992cc455fa518e7'),
    'kernel.fp8_m_grouped_gemm.aaa5cd23dbcc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj96ELj128ELj1ELj128ELj128ELj64ELj7ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f8949758a45637ec9a420202033e29bdc27d8fb63be5dfb7dfc08bb6815aee5c'),
    'kernel.fp8_m_grouped_gemm.1b6e1649e62f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj176ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '814d2951fb5e22dd8269a6104b2a24fb2ccaa33c3e880f37da1227d1ce9fae9a'),
    'kernel.fp8_m_grouped_gemm.809465bd0729': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj176ELj128ELj4ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd4a27fb32cb5ed09b575957930c5a4262425ef5c599dc9acbd11c1cfa06c3042'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9b3d26953e1fe83fb947fce7699dbf3af9d9cdd377109b93ace786b843e0a351'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f9614269a33b5015fc15fa5257e4bce6def90248c0c9da467d19b23f449d8386'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c2bad37c203ce2593ec32e5824257002707fb541c776c1e972334e746f3f7472'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e4f053f18f6f0896cf51f74119cbeebff3a66ce1cbb6f855ee69cde5c2d58b27'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '13f03c2784bafb95f9eada58c7efc51196e9fa4a30244b337fb2d1efb78082d1'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2d8bd157a381af38729fc604711ba1c87705ae7acb060b6f1ede349d650f8f71'),
    'kernel.fp8_m_grouped_gemm.962f0d605e0d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj160ELj128ELj1ELj128ELj128ELj64ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8ee820b8aca2d6584b10017ac4a5bcc2da9f4b80e3549e4722a30d0835699dd9'),
    'kernel.fp8_m_grouped_gemm.c634a9f8f00c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj160ELj128ELj2ELj128ELj128ELj64ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ab875d4e36ce6dee6f06b91fb5f1cd72143b7533c67a56470f932e63528e4506'),
    'kernel.fp8_m_grouped_gemm.02acb2ba71fd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj4ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '78f9b18ea5f8566aecd48fc1182bfe0206cfa2b57598af1577cf1888d5d8d7b0'),
    'kernel.fp8_m_grouped_gemm.34fa13596ab4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cb4cae2dbf4d4b32cb51eb332b0d8f8af5188e9ae2e288a8d0f531af7a564ec1'),
    'kernel.fp8_m_grouped_gemm.469c50e9bf98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '576d52f8ff41e5316be1af19fa2611b0184590d7f4d4981737ebb4404ba9c1fa'),
    'kernel.fp8_m_grouped_gemm.a0469fb341ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '16f9de215314fc497ebd2ae80c11565760f49910ac1b45ce4f852ce5082b5805'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1fae7050f8f6fdf3b19e0db23f4639b65e5c12d2a4b50d1f23b34a2a1855033e'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '419369d474e03235f46b14406079cf3e99f999f3e9dd864711356931d032100c'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '40d99dc7f8eaa0ceeeb65f5a644a9530139fab7975f731c2ee56688b52ef8c14'),
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6750cbaf1d59ac8c56a555faca3013968b7f55b4be1aa2ce790f051361d09e28'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1404a1747cd38c886d9399da4d648f4b9aaa879778a466c08bd9d56d1e5eff3f'),
    'kernel.fp8_m_grouped_gemm.b6fa3fbb084c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2a18ca92c53949149511a698813bd6b29e4d8908d64c8da3502a7085a2604857'),
    'kernel.fp8_m_grouped_gemm.37b0b4d92f98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj8ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '11e23b0d70aea8b305671deb21964cd727ab2688c096cb5d36d6164f677d28e6'),
    'kernel.fp8_m_grouped_gemm.afc4c55a70c7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj16ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fdb00cbb8cd99e90325252246f9ed6afd72cf3f701ea1c558251a0ef0a5405c6'),
    'kernel.fp8_m_grouped_gemm.d006e1878bcb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e0ddb6c2c2325901f17c3622bcb16c178950831f812c5fc649368cc61d62daae'),
    'kernel.fp8_m_grouped_gemm.ef52dedc695f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '41a90667f2ced5c38d402e4ff9ce5462fda724022e114abe9411dd681d32e09e'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '49d1663204d7ffbf9e029fc6005da5a56dbf4dc606f483cf7edf76751d6ed0ad'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7215876c29ba6706b1b6321eb97e7adc3e376070fe36aec51421e610abd253a3'),
    'kernel.fp8_m_grouped_gemm.089e5ba0ea94': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b7b0c70d541be5dd01d2107d19143f4024445a0c0043478d2d4a905a3326ec66'),
    'kernel.fp8_m_grouped_gemm.c8cf6538b89f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj2ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '182c272377a3f69ed166ca6114bd6c80f1ed00ef9728463b91dad0b3b8c235c1'),
    'kernel.fp8_m_grouped_gemm.68592cb8be00': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd2f0625b5fe8409218102b89af663932d7b3b0510ecfd8372fe1bfc8d2b04d3d'),
    'kernel.fp8_m_grouped_gemm.cc6b872e063c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5d0e54b6a58012fc4a90f949ed1dfde8cda8d6ff869448f128df60f7822f54d7'),
    'kernel.fp8_m_grouped_gemm.9ee71b05329e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj16ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '593fd4e79cb012fa82b05a03e6f7fdd202de6f00c0cfecbc93cfa80632b8d105'),
    'kernel.fp8_m_grouped_gemm.1a8d7d79fb4e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj32ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '821407a65a1f4c340a3f177526609f6922d7b313015bb692cad86d5d17259127'),
    'kernel.fp8_m_grouped_gemm.7236199d5dbc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj64ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '71d1d8576a036992e59551d5b7658dc4fd766c2a13395c1143362ca70b8234e0'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2c28feecf2f6495dfae8d959b9b01532ea755b50afdbd648b10e3b49cd4b4996'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c9c35077b8bce014b838cce0240b810647e5003d4778c25a1edae747ab416c29'),
    'kernel.fp8_m_grouped_gemm.0cf1c87228a3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj112ELj128ELj1ELj128ELj128ELj32ELj7ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a42e3cba06894576bd9b668a6d83c38ea6eb40133a04915410986f0a2e91b40d'),
    'kernel.fp8_m_grouped_gemm.60bcd3fe66d4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5169c080a6a718da1571eefd19170b2ca3370091f9cea50ff57240a523e90a1f'),
    'kernel.fp8_m_grouped_gemm.1b42e3f6b2cb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c42bf04120d7a873a368c72780eae1e606c8ce30233ab9df4585e9ac6d559a8d'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f966bd10fa19d2fcef210281d8f5d4a0618aeffe5b29e35a1b6cf9504f201180'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '00b69948bee8dd2ad66dd70ba98e98fb9b989e5f3a0f266f3ac5da1974c37d06'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9801e137027c48dd72cb387f57f57bd5b12f478c5a3646e18b21cca028610b35'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '100ebd697b87ff9137dc1e4f63cb8e0a80ee320741d4cdb8dd1120950896c3bd'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a714e7cb1ae0ee5573f7dc64da96f12e7453823587d7910ac5a72149c6be9ce7'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e5de47351c171b7d08089a6c2a4b969d98b2cc90e9e48c5c589592ca1a51e69d'),
    'kernel.fp8_m_grouped_gemm.ca30719f403f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj1ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8c9a30dfbcedd5c197234baba0d516325ee68d2efde0e5616e3cf62074b3ac37'),
    'kernel.fp8_m_grouped_gemm.0bdfecc9ac6c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '99248e6f9ec8c00f7b6edf53c18a6805e26fc866e078daf494b07eab670ceee5'),
    'kernel.fp8_m_grouped_gemm.02acb2ba71fd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj4ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f8f4444bc8d8205d14e5ca143577f4d47bfc120e7b4d0e7352ab53f40cd125b2'),
    'kernel.fp8_m_grouped_gemm.bc9f6b291a79': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj224ELj128ELj8ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6a2a7ca9afd1e509f54b070a1c091b801b845097df2f9513e250a1e69b458fa7'),
    'kernel.fp8_m_grouped_gemm.469c50e9bf98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cf30bc0f0c8fcebcebb7d19ac02eab986b9b1a7c053533ac7e1f02ebc159d6fb'),
    'kernel.fp8_m_grouped_gemm.a0469fb341ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'daeb9119aa4b9d6b951d6c80fd103da4e49ebd9c03ff0416194d15436dfc404e'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e78855c97a64aada493c5a1bfba4002d33c657ba012344d73e82fe38a0f33f09'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5637e13f272b09bc77313adf5000cfde2750c15f3cfcddee949b2fb0183595de'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ba9c73880551ad318fa4702cdf9a7e1f853b884377a75bbcd5570bd9863932a2'),
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bb9b08c504c8e72e0a76ef9ba75e83bd33e4060da4e469200bbbaa7270bbc0f2'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '72740a96e91b29a440e71d4683c73fbf1ad6c70525f4ef2cae683d5c9b7b1e0d'),
    'kernel.fp8_m_grouped_gemm.07cc80ec5722': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8c28f895f38bfd34b51ea354f6d7da32bd2c3d0d0b1ce51c5528496e2ca97675'),
    'kernel.fp8_m_grouped_gemm.43c7d5560e95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj48ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ab4c3c4c3039a0497db467c8482df963adb46573b2aa18dd9dbd68c1ff164ede'),
    'kernel.fp8_m_grouped_gemm.0457375eb02f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj16ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '36074ab2b32583cfcd1a4f2f73a765335046ff2407dd7354af20654376188c2e'),
    'kernel.fp8_m_grouped_gemm.d006e1878bcb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f73359e0dce8c3ef4fdb555b8f67d2929eef9f9cd35b298eca05d8d56cdb6101'),
    'kernel.fp8_m_grouped_gemm.ef52dedc695f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '895a77938bc39320b366aff53a3d200c934423363cf81c00c3c75e9dcb63c222'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c0f5194663254545576447a4ddde33ad0dd897788b9170785cf0b89dc1c5077d'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4fe98d1c9b165a01ba1b0416813aacd6f5fd2c2d6904135b9e597534d7ba9be0'),
    'kernel.fp8_m_grouped_gemm.a146ac29f6ce': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f2f8f5b89c4ea2dfc3c3171d8a7b42b9380a50e4b0ebb0c8c3286558bcc4476b'),
    'kernel.fp8_m_grouped_gemm.295cb928b778': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj48ELj128ELj2ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e4011b64de6fda5476ff5ee99e74247f792966f837525bf8ad9d95f3b1e6616f'),
    'kernel.fp8_m_grouped_gemm.0c13b0be1687': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj80ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9fc0fba9095dcdd84db71518f52b46569ee773f3314949d5535567c25690a9f3'),
    'kernel.fp8_m_grouped_gemm.c750c1dfcaa4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '58aa4cfe2a153a1e176cbca7159680aaf0cbb6ee53c227b7094378f0885e1afe'),
    'kernel.fp8_m_grouped_gemm.a773484fcf60': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj16ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a0dd0406202ace33097aabba9b49cad6b7f2683aee9e19d6173b7ba3c3fa09ec'),
    'kernel.fp8_m_grouped_gemm.1a8d7d79fb4e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj32ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f5a65362c33ca5d375dd7d5542c4b8d5891804decc62acee1cdfdf9f065ac739'),
    'kernel.fp8_m_grouped_gemm.7236199d5dbc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj64ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '28cce9dc4d66b61f7670e81feb15f4b8f3e1db8690c28a6f5622fdaba8c885e3'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f84b93c6a73f1be6555d0e4849db303909862374c43d72b13e5b2aeb6201a19e'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a4881e1816c7bf247d8467fad3c63b5d31784a51128802805d3492baa9dca456'),
    'kernel.fp8_m_grouped_gemm.56d4003f047c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj144ELj128ELj1ELj128ELj128ELj32ELj6ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3903207872536e39c08a05764917d4c63d70a8d3194184549f37d8c206e0671b'),
    'kernel.fp8_m_grouped_gemm.65d48b9936ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj144ELj128ELj2ELj128ELj128ELj32ELj6ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '267eb276497c87d5be11357a32b763a098936e91a4ddfcfef1d644652791f5fd'),
    'kernel.fp8_m_grouped_gemm.c5aa84324e6d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj192ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4c490bb9870e7e1fc563ff9be2df02e39050bbe4bc9d08780c06bcce2107e8b4'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2cfa2181bf6e4540d08dde70565095d73f3181d7e7075ad1eb6e28d56e4f5ad6'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4ce9b1d1e0fa869b2e4f5c2a2c2da80be609708c93b4f0f1341a7de655e4da4c'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e920792c2ce1abb1870fe365322ec58bdbc0b5d8ecf1fba10177b16a017fe35c'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ca67702fc38a635dfdeb2e2112285789800fdccb0e30f9ef70e03e2aa3b8edc0'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c7853b22c6c1ed9b94694845c358c5678f9794d6b66a163f88cdb9b4479c57a3'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5761d8f97f9baaed0cb50eed1ab32f81ac87ffe7d226e08e08b04ce9a3556573'),
    'kernel.fp8_m_grouped_gemm.afb89542e7d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj128ELj128ELj1ELj128ELj128ELj128ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f00a521092ebb8043e10652b0178d91ad591113cc36a0a46a7f2388c640df9f5'),
    'kernel.fp8_m_grouped_gemm.bd573a553e2e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj176ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '05abfb50be4b0ec0415e5358502afc61f088fd61cb0eb7005cbf80b8112d7676'),
    'kernel.fp8_m_grouped_gemm.02acb2ba71fd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj4ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a7e2d10e4b8b49017dd3373e7e998eea31a4edbba3b821324ea7a3ecdabb0552'),
    'kernel.fp8_m_grouped_gemm.bc9f6b291a79': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj224ELj128ELj8ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9375f99fe4211407c880436eb3a49992b2e0ec15f4472748ec6fd598a3e63c25'),
    'kernel.fp8_m_grouped_gemm.469c50e9bf98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f4ed0f7c789bf0a6fdacc2f7b584b7a3dd1ffde7ad252f2e4401ebbbde11aac6'),
    'kernel.fp8_m_grouped_gemm.a0469fb341ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6da892763bacc8ca99d045db7751acdb22f47bfb58e27b5f03db7ffaffc8fcdf'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c722136466f58b407fc928d9549bc6141d9ca4003600a5a28b03b70e50d23fa3'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '14a93ea6d64237b52ef36d971145632c423c13225b5fe40700c0186029650973'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7ceac593cdf79685cd83d14b05a606bbfd408069b8b267c63e58ce980178c8a4'),
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b587e64edb9c991958d58049eaa4e56392fe7b06a7b09ee0fd7708c5c9cd05ab'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0614cc2dd2f291f12007b24d3efcb08d1bf04aa2a983d528a1a9151121193e15'),
    'kernel.fp8_m_grouped_gemm.07cc80ec5722': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'dfa27109572f9242a9663d23b4a7e4a946b901240a4dbfc1d3003616d19a04dd'),
    'kernel.fp8_m_grouped_gemm.43c7d5560e95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj48ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '87d2de4358f87ef3563afd90c0d85dff599ac0a22bb214155a97f2a0b5f01449'),
    'kernel.fp8_m_grouped_gemm.0457375eb02f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj16ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd6503bdd679b42675f366db025d8e57f276a270214b62de6be7668c10e060a7d'),
    'kernel.fp8_m_grouped_gemm.d006e1878bcb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bb24113af44360a73252b03dcc0f958007ec16815c218a8da188189a68f640da'),
    'kernel.fp8_m_grouped_gemm.ef52dedc695f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '64eec69b2568d74b1cecb0e2e4f3e76507f66b0ba328ff88b9d020a09e28db0e'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd85c982394233af62f4cabbb53b730415694a48bbec4fddc76f0fd9ddf1c5f9c'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '03f377bb7f28273d06ba5f277390b5f461e7c1bd3379e0e554685611a21c90f8'),
    'kernel.fp8_m_grouped_gemm.a146ac29f6ce': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8436cb1f0fc8dfc15a5c8023af1d5c534caca372160a8fd2d41b12dc25e032da'),
    'kernel.fp8_m_grouped_gemm.295cb928b778': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj48ELj128ELj2ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '75a203f846b4fc736c8e8ab41aa2b3df5de9de831d6ec4395ffb0781fcfc3e7b'),
    'kernel.fp8_m_grouped_gemm.5ca140544980': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj96ELj128ELj4ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '936c4a94304f4b489de04c5da40920c786ae2c6ad5c03c6ef30304654fd3e97e'),
    'kernel.fp8_m_grouped_gemm.c750c1dfcaa4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'eaf31d328e93bae1104197fdad038cf254c0f36b3ea6d9b34f6872faf9264fbe'),
    'kernel.fp8_m_grouped_gemm.a773484fcf60': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj16ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5556f70023dbdfec647eec701a707673243726588474c4a768382e3e82d0ab46'),
    'kernel.fp8_m_grouped_gemm.1a8d7d79fb4e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj32ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6d521f2560568dd32d4df65b42e2b7794913556aad8ec89f09b0c8516b0ab1fb'),
    'kernel.fp8_m_grouped_gemm.7236199d5dbc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj64ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8b0c24b95cd94370cc9677c80a7de958ced872eba783afacb9d18ea5da6ec34b'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'eb174c4dfb9c6fc3b68b14b5cffc20991ae67a9c8a5662ab279bf246a5bcb711'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0bd3dc2c8cbab04318fa4661cede5a2f103dfe1711c126404e4f37224539097b'),
    'kernel.fp8_m_grouped_gemm.8c40008c2518': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj176ELj128ELj1ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cc1840613ce000b0dc25b40708381dfadfb2bce3525c5401dec46cf6730d0574'),
    'kernel.fp8_m_grouped_gemm.1b6e1649e62f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj176ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '19e5aaacd1e46c9cdd2f8bcf378441be261eb737361772e548519e1540eea836'),
    'kernel.fp8_m_grouped_gemm.1b42e3f6b2cb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0cab54641bc43ab2730b8b34e67eefd121cac81e67cacdfa5d027f17c97eb415'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1e0de9a6832b857d196ff2ef731884c40101b7d0e4b5f430bdf87576d15f29c1'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e83cad9aaac18d982ff175b04028490cf18e8300104b12334e23f16cadd38b41'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a70a385c654c2d0cf5052f373ba2c8c5202911688e8c3f1e2828cac0bc5627dd'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '99062aea2df566651910d8e8daa28c3a73ece8e180bae193a51aaf6178297a7f'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0ff22ac855295c38cbc6dfdc0f154e45e39804b1738cceeea535ba2fe07d2a54'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'bde70c71c7b4d5516a0ae040fbfce3ef14289aacd2c099a03d8642d117119120'),
    'kernel.fp8_m_grouped_gemm.962f0d605e0d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj160ELj128ELj1ELj128ELj128ELj64ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd34e87ecd8733fa609eed75e14a5a2a9e86ec00af8ff5a918b11538840011b4f'),
    'kernel.fp8_m_grouped_gemm.0bdfecc9ac6c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1ec7aeacd4508a3f8a210fdebe6fafe8cc034048b27490ce37fba743cad0591b'),
    'kernel.fp8_m_grouped_gemm.a80a7bd4c0e6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b5efa305ee330676cd5f329f4f461b4dd264731de4f9ba1179cde0dd5b647766'),
    'kernel.fp8_m_grouped_gemm.34fa13596ab4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ff43b00e9aa08cb9f72572ac470c470df3eac9154300e628bfab5a4a6dc46270'),
    'kernel.fp8_m_grouped_gemm.469c50e9bf98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '915547c2e3cd32adbece42b5df198aeb231a817950522754f91ea6bc812accd6'),
    'kernel.fp8_m_grouped_gemm.a0469fb341ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '651369a6dba352b0e5a4afed304b852b5920dbe1312a3430186a82115a458abf'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '224683e7e7028881c40b6085447d1a9875484a449dbc2bc9b76f76493707c568'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c2dd6c519d655e54639e66ccf65831cda990ee346cd5808dee605114d55c4950'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ddf7ad0b7adcaf7f85d793cb020f1e5a4f07e3abb2b8f22cb9b6b1eb8015b4dc'),
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '84fce94ae5fa7e1dfe70e5aaad89b42d1666b10deece354b1660f8b165df0589'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd26f2473396f8ec10208d018b330864161c90cc24e7c163d1795dae04c5c2095'),
    'kernel.fp8_m_grouped_gemm.07cc80ec5722': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'afc9b550f16a95592b85fded9bef916710ac4ca399655c3292a04fd22024e9b4'),
    'kernel.fp8_m_grouped_gemm.dbeceaef99e4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ee88cfdb68da30c0106d6088619a310acd61a45d97d15373c6ef56cd0feb27d8'),
    'kernel.fp8_m_grouped_gemm.0457375eb02f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj16ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '61486ce6ae60bacffbcb6b7b63000b2aefabd696aaca375a66546845e7829d27'),
    'kernel.fp8_m_grouped_gemm.d006e1878bcb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c9d91206fde2c369a4e286064e986e4eaf8e58c27d8661d5e3dc0a1aabb43df3'),
    'kernel.fp8_m_grouped_gemm.ef52dedc695f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '81e146244df25a23d886a079d81117e04101dff829ebb4188f9bb44dbc7ebdaf'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f21c453a014715eaec4883cdb9344e299e3748fbf7652b40850ea4cefe288359'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '29c979d47e493d15c3fd5873ea892e4b7f413765a33d20f021466b873b4f68c8'),
    'kernel.fp8_m_grouped_gemm.a146ac29f6ce': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ff9064b4cc42c8c8b30c57bb58942aff690e4dd0a52f71337e6bd2c379e3c18c'),
    'kernel.fp8_m_grouped_gemm.175e594df0a3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj2ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '624a9273d45033d84fbc22ae272d6454a597b075049c1219161115165dd51352'),
    'kernel.fp8_m_grouped_gemm.86957a8cc468': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj112ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6895ada0442bc00caca1c864ecf902b1280e195b5377647738b43d5052fb46e5'),
    'kernel.fp8_m_grouped_gemm.929c84df298c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj112ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3d3bac4efe4ba9a14cb3c83daa4d2bfc1b9b42bcd5fb8b25c1a9e6cabf1fea5c'),
    'kernel.fp8_m_grouped_gemm.a773484fcf60': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj16ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '270cd558f147eaee5e0004e50613462f085e735281b5db33c06c73cbfa4e0809'),
    'kernel.fp8_m_grouped_gemm.1a8d7d79fb4e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj32ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '39c7c732020df27b72c8fd8076e4f949b62df32e632d1405343c9228219e9836'),
    'kernel.fp8_m_grouped_gemm.7236199d5dbc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj64ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f5f744a749e79c59585a5ac70df8e641b8025d974d914d9fe650a2ab69c7bc09'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4a659bb0651a012faa1c8706384329e0c0b001853bf925fb024499b8063c7c65'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fe0f1db2754b82724945e497ba086b7bfc097816d66fb2c80a52927a0367cce8'),
    'kernel.fp8_m_grouped_gemm.431ef2fad71b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj208ELj128ELj1ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c2d676e0216bc5095e805a7d6b05d3eaa82c01d1cc54bfae2c66ff2ee4c24dd3'),
    'kernel.fp8_m_grouped_gemm.9076699b66bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj208ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3c4bd2706f04045be2b1e89755d3ec8ce853c865276f75fc4d87594b94364b06'),
    'kernel.fp8_m_grouped_gemm.78b9f7ada3df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj208ELj128ELj4ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2f255562dba953b992bf2cd48bcec0e4a11347b78eca47b3405c9fa9951a181b'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '681844ae1f8290a42c4133b9b78bd2eaf5b100369e28dccca639f0bf4c3509b4'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'eaaa576382ed5ec90fece10c10fa36c235fd84dec01063138bd8206b1d8b63ac'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'eb577cb08ffe8ba326d538f9e427cdcc881db100e8b9cd04bca7349c1e5b2105'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '64d30b07f73325bb72acfd7e784f5f93a054f2e700f6b9fdb14291070f7c210c'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6a401b0f3ff045fef8024a1cd42d26dc78d314c57409e29ccc8a3be77cad021c'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6ba642de53680775f8bd309b5d12b6b2c6e2dd0aeedacf4cbc2d064cd112cd24'),
    'kernel.fp8_m_grouped_gemm.fed6ecde864f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj176ELj128ELj1ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f1166f9730e2c8b8c6b6611cb362da1027d09be9c2c6255d8bdf7a38e28ece38'),
    'kernel.fp8_m_grouped_gemm.0b5b0ae9a83a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a59231bab7644dad5a9b69ba41b3092e2cab400e6ad145a56d20933c5771cba6'),
    'kernel.fp8_m_grouped_gemm.a80a7bd4c0e6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '48ee034af6240b22bcae9ce2e58b93e90cd2ad36cfa13edce5c84e88af28586e'),
    'kernel.fp8_m_grouped_gemm.34fa13596ab4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1fa1ed98b53b74e57f2932b00dc79355ab814b676ba7b98564018e7bc47237e5'),
    'kernel.fp8_m_grouped_gemm.469c50e9bf98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd450a83e6d30d4a7c7323dbd670b72c2e978eafe280db708226914878c6cb26a'),
    'kernel.fp8_m_grouped_gemm.a0469fb341ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6ac0d6069b2ea36d43dc5916e3edc7eb8cada9b21a7b1f7c437f6801c8e50022'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '06b562c7e8f92152b9e9fed9386fd761ec948e5bdb8242ac1d4a7813b3f29efd'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '49af2735cb25a126b361d84766253f61ae975b5c5c09ec81787b574ca8870095'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7fa979cd597c32ab93623eb3076f2bd61eeb1c89ae7ef3c1d9ba0ce8ba963022'),
    'kernel.fp8_m_grouped_gemm.b39536c0aec7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'cffbbfba4f31788981764f22d6c84726a3ab62392cf2eba44f25fc4e836f2ab0'),
    'kernel.fp8_m_grouped_gemm.a4ee3555cd88': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5374c1f803969964c7d19c09655392e52fa75eab9d58d4f1037435396b150d72'),
    'kernel.fp8_m_grouped_gemm.07cc80ec5722': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b8c190ac6468b2f49e20e863e1a573b8ce8e8b5f2e10221da14f2377bc638e87'),
    'kernel.fp8_m_grouped_gemm.dbeceaef99e4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9864344fe9840fabf10b1daba6fa0899295f7d01727f5f04de3cecc8ac7fc2ba'),
    'kernel.fp8_m_grouped_gemm.0457375eb02f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj16ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '86b321da171976bb33127e406961e8d088b4b48ac3f2631d76c778e91b85f1bc'),
    'kernel.fp8_m_grouped_gemm.d006e1878bcb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj32ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '70d9c26299f32e781baf65d91fcb0139c3bce89478f36d9301b91400f8cacbd0'),
    'kernel.fp8_m_grouped_gemm.ef52dedc695f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj64ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '674d86f50e3633ae5f086d78ac56ca824b960d5ab8ff8ba1c2ee8da857119009'),
    'kernel.fp8_m_grouped_gemm.dcc1fbc106f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj128ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f8ae926fc2583007c0109834b058b1367d8bc8e547da65ef7eedbe3bbcab7a2e'),
    'kernel.fp8_m_grouped_gemm.1203d4cf5d78': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj256ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '822bf1caeaadab7bf1672c797ba580997690e7d7a500794b0588a0b59804d5d7'),
    'kernel.fp8_m_grouped_gemm.a146ac29f6ce': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '74961d4de31f87660b50990f70a4ab3022387c2e0c3436b9924313b0ef00ec99'),
    'kernel.fp8_m_grouped_gemm.175e594df0a3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj2ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c2f60a93e0b0f16dedc99fbe7fbf8f36a45b2e4a499f2d614d091d931891c162'),
    'kernel.fp8_m_grouped_gemm.e1580e822cb7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9cf27e9019106fdc38825e6ceea76e0f8c82c158e7bced0c96b82ee9e4c67cf2'),
    'kernel.fp8_m_grouped_gemm.cc6b872e063c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2a3896144ab16454f4f3441cdcc6bf5c380c25203ca17d88f9353ff8ba7afaf7'),
    'kernel.fp8_m_grouped_gemm.a773484fcf60': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj16ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '855518e5cfe4682970e09e75febe80a8453dc307636cd9b5a5be71dd66308da5'),
    'kernel.fp8_m_grouped_gemm.1a8d7d79fb4e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj32ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '74e1f6ce75992aa07d261e046bc0408704f72e2c4228cc7999b226e5116789e5'),
    'kernel.fp8_m_grouped_gemm.7236199d5dbc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj64ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '69d3eded3c4fbaf663b7a1969696c2acd34c0d5bf628f587ce642675646f696c'),
    'kernel.fp8_m_grouped_gemm.425f53a7f135': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj128ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd273c4fae20b550fa197c819974f374789fce662c0ce586e02989e9915788768'),
    'kernel.fp8_m_grouped_gemm.b9cadb5253e0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj256ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e29d4a727a2d8a9a69550d4bbc62c6b4cd42cfa62f9226d2a85d68b072ae5f99'),
    'kernel.fp8_m_grouped_gemm.abfccf73a058': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4bc8d21c0f14b1da4ec4a5c620839df55927c34d3db91223c87d85cc6d1c78e6'),
    'kernel.fp8_m_grouped_gemm.60bcd3fe66d4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj2ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '0ca3e886111594ea8181c41fbb8ff3bab1cc47eabe346cacf108cf5375385bb0'),
    'kernel.fp8_m_grouped_gemm.1b42e3f6b2cb': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fbc3ef558106d45902a6adc12c4d9dfa44e3b3f4c7396e65df9310a7bfd12116'),
    'kernel.fp8_m_grouped_gemm.9e527d8a5626': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e51a832ad66294e899a2b650c7f49d7e2a301ffd7ea852d0efee7146f34977a2'),
    'kernel.fp8_m_grouped_gemm.42e2fe55cae9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '62942c39566725e25bc29dc56d60c4560790a1953dcb5f4a046c02469014fd5a'),
    'kernel.fp8_m_grouped_gemm.2eb1522add3f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4118fa271cc37c425352eba7130e31e590b8e331d067e8504e541c78ee76cee5'),
    'kernel.fp8_m_grouped_gemm.6a8eb460ee80': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7bf2cc6bbf96c56b776c59545cf5d49bba46e345e31d333db6f6dd92c5e51c16'),
    'kernel.fp8_m_grouped_gemm.b7408f34f837': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '27b3007d8af1c0f036102123cd8c7c6b284784842ad7254375ebb6819c0c8173'),
    'kernel.fp8_m_grouped_gemm.313307927576': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj7168ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1feb71c61aa65fde1d32bffda09f707c749bbbc380c7530a1bf7684d5fb4834d'),
    'kernel.fp8_m_grouped_gemm.ca30719f403f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj1ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4010adc48de9b46d5d819f486b324255cabbc48ab7e9a7963b41d12d4b6d3a14'),
    'kernel.fp8_m_grouped_gemm.0bdfecc9ac6c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj208ELj128ELj2ELj128ELj128ELj32ELj5ELj1ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '21d81c53c19c881365ceec95d3961e790a239fc3e1cf7d1248d45c483ae14c8c'),
    'kernel.fp8_m_grouped_gemm.5f10d28a2af8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj224ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '525f10cace4c49da9de74f95cdffff6122e41771f3297230a668c0fe227514c0'),
    'kernel.fp8_m_grouped_gemm.34fa13596ab4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a14d27fae8bb83c92af6f47d4a0fe00b9de41f59322ff751a66e7ae7cf4dcbc8'),
    'kernel.fp8_m_grouped_gemm.469c50e9bf98': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj16ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '085d393adb9368dae4835b3eab386ae00c511838ba5de00af7a22fdb2c5a7a56'),
    'kernel.fp8_m_grouped_gemm.a0469fb341ea': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj32ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8506fd3bcb616f05d82cc3fddee34d25125004a8fbf317447876fe83856c25dd'),
    'kernel.fp8_m_grouped_gemm.3361eb2f28bf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj64ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '13a7d2d846424742f63c7d98351c504ca5a82939b96c7fe8ef74ba6099cabc54'),
    'kernel.fp8_m_grouped_gemm.30e2c463f2d3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj128ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'daaf10fdeed46972c5c7108f2372178f78eadb3cc7559de9c396d370b0b3f9a0'),
    'kernel.fp8_m_grouped_gemm.5cd1e10a0405': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj7168ELj2048ELj128ELj240ELj128ELj256ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE2ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9cbc64169cb3c8bd2c2a4af0a43d1ad713f3cf0ac9f69d2ef04ed54105275ad9'),
}
# fmt: on
