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


_artifact_hash = "d25901733420c7cddc1adf799b0d4639ed1e162f"


def load_all():
    for cubin_name in KERNEL_MAP:
        if cubin_name in RUNTIME_CACHE:
            continue
        symbol, sha256 = KERNEL_MAP[cubin_name]
        cubin_prefix = f"0ffe2769c7eea90c44894abadf3eecf38801a143/deep-gemm/"
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
    cubin_prefix = f"0ffe2769c7eea90c44894abadf3eecf38801a143/deep-gemm/"
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
    'kernel.fp8_m_grouped_gemm.4caa3b87e72c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '84f2b2a43c225d836291bce3908c32477d95be63994f57ae9d5f8a1d215dec3e'),
    'kernel.fp8_m_grouped_gemm.7953ee0a470b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f6c74341f140f72eb2ddf7502f6292bc40e6168ad1a817499912959003911119'),
    'kernel.fp8_m_grouped_gemm.43a66a194af2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1828ee51f57cee5929d1f7ae55ff4048262c697db8150f85bfc29880d2705465'),
    'kernel.fp8_m_grouped_gemm.03104cf927df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '663d75060506f847f0df46d42806ee6433ca501b85b8cfda5e0760f8c20bd1ea'),
    'kernel.fp8_m_grouped_gemm.d8caec6f111b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a5f673b6edda67b32f963229444598f6093bed20895e8b74e22e2e5a81cccb7c'),
    'kernel.fp8_m_grouped_gemm.9e474c44622e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '88e5c2f504d206b9abf191c6331c6c00ad7340727a6e08c2a190c907d0222022'),
    'kernel.fp8_m_grouped_gemm.e2987a4d4273': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3f73568413f347743222e736db64d6dac292558bd1aaf6700a36cdf05328c7de'),
    'kernel.fp8_m_grouped_gemm.9b9742c0b9b2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9337018bfb7d248116b2d7152060b7583911e70c4b702a6f9377087234a44862'),
    'kernel.fp8_m_grouped_gemm.116f46018c23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd33432ab446cbe0a11f2cb40777a8dc0d682b127dd4d7e41fcea81e8860a9541'),
    'kernel.fp8_m_grouped_gemm.9d265f01caa7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a396bb35f894eb6215b5c4d4d6c1f4175a957f7073822afd2a9aad1497c78668'),
    'kernel.fp8_m_grouped_gemm.a918127d1773': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '65e9794f01c3df191a1facb87dd98bdb660bf279cfd2bba00d74750a30da6d36'),
    'kernel.fp8_m_grouped_gemm.d709904c7313': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b413a636e090435c6d758dd5bb00db7f666720150b3deb194ee5d2cc66ebb849'),
    'kernel.fp8_m_grouped_gemm.c4b0089929a7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '139994663124f5f524fa3cda25b14cb5e1ceb14e947cd4dffa203d75d83a9156'),
    'kernel.fp8_m_grouped_gemm.65dad77228df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fa22370a8a3c15603f780a5a23582921952fbee13c6bdd7518ed3a19a2ed5375'),
    'kernel.fp8_m_grouped_gemm.3d80bde34f4d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj4ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2e8f4e166da428e08e8a490cdace53bf0b9ed74135d5b1eb51c625141f9ce0ff'),
    'kernel.fp8_m_grouped_gemm.13405f117b95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '791a3ac1902646b70f5d3f6fae9774141bc7b187689407cb22b47ef5d0de53ef'),
    'kernel.fp8_m_grouped_gemm.df8b3af58001': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj64ELj128ELj8ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ce2548dfffbca0188ce44e99a2916a58eccaffb4be151d72fc50f83cc82142e0'),
    'kernel.fp8_m_grouped_gemm.b6ea4e9c1ee9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj64ELj128ELj8ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '16686622b2973547e18bb0f72b2285e7d2898e5164cfb0baefdf218334153ab3'),
    'kernel.fp8_m_grouped_gemm.49778ac9c667': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '90babf6e3c6b757c4f1027424860d894e72474409573b2d3ed4472009a0ec19e'),
    'kernel.fp8_m_grouped_gemm.01abe8ac39f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '97a9b3ee5d2493adf4e4b97b43dec8bdf9bd434f7d810e6d3b9e04a48f146e1d'),
    'kernel.fp8_m_grouped_gemm.67a38a8a7b2b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a7327e44d7b7d6d7910c0893f05ed5a2289c5b477cb1b324ed49dc7533f66372'),
    'kernel.fp8_m_grouped_gemm.f83010d0a78d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '87d95098198b417f0da38f44fe0fe12686ca85779618dfc0ae0e9a5193335377'),
    'kernel.fp8_m_grouped_gemm.d3ef2db30620': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e6aba67d40ffe3311bb5dbc1801e9820d693c42672ff14e8dd32b2a77ba93a1d'),
    'kernel.fp8_m_grouped_gemm.7ebbe8c3fc4d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'aeb609aff64f2ba043a76e69f95a8455fc9d5dac0befa6b8526508b543bfa104'),
    'kernel.fp8_m_grouped_gemm.4db0a0684aa2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '24d75c84e5f346628b7dae9dc504beb3aa436c32a5afa36a6cd1def1a371159a'),
    'kernel.fp8_m_grouped_gemm.08051e88e138': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4aa148ad1ea1c64cc38ef3fe89b45e410dfbd8b4c0218eeae171c775343b5752'),
    'kernel.fp8_m_grouped_gemm.93f36856973c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b702567ad0402dd3c6f62f44132785fac2871a407c14b85806ac77e24bbaf071'),
    'kernel.fp8_m_grouped_gemm.9ebed43a6023': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '180dd2f05ab448f09d4848b1c98ffe7670f0a66da37e81ba184f74227ab4c10c'),
    'kernel.fp8_m_grouped_gemm.34e2d7ee46b8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd492cff39e4754feaa1ecd34406c9664f0653727f2a1785f73fbc6708b755f33'),
    'kernel.fp8_m_grouped_gemm.ddd5007b9957': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'aa0d88f7b10488a61e8608ffc5cc085b8e8cc93a493eb20a3fd61629e6872e04'),
    'kernel.fp8_m_grouped_gemm.6cb0320d6d23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b1010c021b2462590ea2ae8a02705907200f5b4f310d16c009851fb953997873'),
    'kernel.fp8_m_grouped_gemm.22c8d7120f84': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1657a41c7aa1a73c4e9310a45d036a261bcb095ee928cc14d211b440871cfb6d'),
    'kernel.fp8_m_grouped_gemm.14831ab6238e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj64ELj128ELj4ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '15781b9f6ba741d076cc3c3d7fb449a5942da2b801f7e333cf1588c31376b88c'),
    'kernel.fp8_m_grouped_gemm.2b3721bf294c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj64ELj128ELj4ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ada960f519d2a856349cb81ca47c2685924b2262ef01bb5c67a3108454978316'),
    'kernel.fp8_m_grouped_gemm.640e14f816cf': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c5660c0f72d0193bd9a2fa93db01ade61daee2877a3f8bee855e69b0961315eb'),
    'kernel.fp8_m_grouped_gemm.84f38656586a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '16f6c91b20d50609072d0bfd44da95c794ceb5ed1fd2c2ac4a98ec3db5d79054'),
    'kernel.fp8_m_grouped_gemm.f9851e3c94f2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b1b712c0792574d4f181d3db1f7c78caf5e5fd85e1174bf0b7f795c0b4997a9c'),
    'kernel.fp8_m_grouped_gemm.75ac829a6832': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '45890ae41738ae451f0ba54e57a19450e8590190d036ff9201edf7f70712af1f'),
    'kernel.fp8_m_grouped_gemm.bd9ef159ca1f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '40bf5457f1c9e3b167e9ea416667b9c7b91b50109ae5c4a446f6d45c4b560e7a'),
    'kernel.fp8_m_grouped_gemm.76383f629fc4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3f2888d599396e68c3242cf6bd97d479e5d74325709e4b95ffa2fc876f16f0aa'),
    'kernel.fp8_m_grouped_gemm.d5046aeaf1a6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ea60dc7190ac9ad9185cf10af0d6e4337c206223f96671d05fe94e907b6d7b16'),
    'kernel.fp8_m_grouped_gemm.166e3fa97b55': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6a2a07cc422e6de060de5af7d7437208b69eec5ad10ab940a1ae224d9a94f06d'),
    'kernel.fp8_m_grouped_gemm.89f63ec7b8ed': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a2d1d9a69010bdb37d02cb2c339f4bde693b2a9c171d4c1c80c84cc7a3d9770f'),
    'kernel.fp8_m_grouped_gemm.933e8ff9de4f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1697fa2b7f0305e9ce79821b227781679e0fd7a665ec8a762ba3deaf8f5b7970'),
    'kernel.fp8_m_grouped_gemm.d84c0323cd7a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd8d5eb64b4e11856335c1451fdc5ea38e8b5e28c00d78d38d7043129ce72c9e4'),
    'kernel.fp8_m_grouped_gemm.ca96166e620d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9c501c3f3f6831db7a7f0e49adcb169e1d6ccd49bcec47ce3bf309f4d00bd7cc'),
    'kernel.fp8_m_grouped_gemm.b53f23fe8b35': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1b295ed818e41764074f0271efa858adfad273e367370deceb18d6781d87fd0b'),
    'kernel.fp8_m_grouped_gemm.b631d84ad228': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '41d07a6dfa0c9440caa98582a879fce81cbdfa8046e91de866e894aa748c6558'),
    'kernel.fp8_m_grouped_gemm.9d32a431cf70': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj32ELj128ELj1ELj128ELj128ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e69d0dfd07c5eddee1ab4e4f21811303b078d2c24403b230de67b449dd20482c'),
    'kernel.fp8_m_grouped_gemm.12ff4c6e55ba': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9a1937c0a98f53910bfa92c589adb2e45302add5ba681694827af084acf36f04'),
    'kernel.fp8_m_grouped_gemm.adb182190e42': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e7baf08d462f19f4b6ca4a9869d9ed42fe28dc5735c93150fd8dc8e7a6565e68'),
    'kernel.fp8_m_grouped_gemm.65c649fb73ca': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj160ELj128ELj4ELj128ELj32ELj64ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9b11eb4968644348e4ee724c76f6bee422e2c30bcff21777131ab5387b82cd38'),
    'kernel.fp8_m_grouped_gemm.e11d8197ff5a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'fa876a7977d5e84487513e4524c7f1b915bb33487ab3d3c790d36f27a1644e99'),
    'kernel.fp8_m_grouped_gemm.812550444fe4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4383e5feb58e2a42e2fdb04612f83c3b333e834f2e075ac3647dcd393154cb2d'),
    'kernel.fp8_m_grouped_gemm.4caa3b87e72c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3e8107b32d59a6392cf0d8be19d5d34f4b39638d6a969c025755eb154dd549b9'),
    'kernel.fp8_m_grouped_gemm.7953ee0a470b': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '931e3d00ff3d9616f2a9bccbe34b999adf6eed2f43c7e9613454f833e86cef17'),
    'kernel.fp8_m_grouped_gemm.c57637209e85': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '98a6bd38eda1c545f0d85ca9d1ea8def2a5eee814aea56b55ca9cb4bb80e980e'),
    'kernel.fp8_m_grouped_gemm.03104cf927df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e40fd72897cae479fbcee957f918112de225d37b6c89a55a7cb9177d98d3a7bf'),
    'kernel.fp8_m_grouped_gemm.be55ed92eabd': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8b26850278b1f2f41bbbb58c10d764344b587ee1e2056f169afbc5a92400050b'),
    'kernel.fp8_m_grouped_gemm.f0800daaacd4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj128ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f733a86e24d6d1194afe5276677193f0118173f82d3ea2fda3d0c7c1815bbc67'),
    'kernel.fp8_m_grouped_gemm.e2987a4d4273': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e2b906cbb888df46a2fac208d6794d683305b570f1080fa7a6fac74fef68ec4b'),
    'kernel.fp8_m_grouped_gemm.9b9742c0b9b2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '423f3452f5599bd40c5c913d22abd9ef45b89597c3f259cb93975d5fd4447890'),
    'kernel.fp8_m_grouped_gemm.5d83408ad0a0': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '16c2d8a245868d837faab47b330dc90505f200e6d97e5da20683e114e01a1ec3'),
    'kernel.fp8_m_grouped_gemm.9d265f01caa7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c2f8226a39e114e9badc4ec2163d2aa22096f3742f8332e36171bf6e1d3c05eb'),
    'kernel.fp8_m_grouped_gemm.a3687807b5db': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'a0bb6b19aec41605597bc4218dbdb43699dae3ace87d4d28509297c98a989dfe'),
    'kernel.fp8_m_grouped_gemm.75ef51830e08': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj512ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd960156c0491b87bf40e9bb64bb894be49183dc492154654e57b43a9e30e2d91'),
    'kernel.fp8_m_grouped_gemm.c4b0089929a7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3bf6f0e599af93c06d26d23d24f1cc9c1beb40f918c07ab60ab14ce8368dd6bc'),
    'kernel.fp8_m_grouped_gemm.65dad77228df': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '6dc94c844f436060b87d1465121402982091a067c98846c1b351eceb8f881be3'),
    'kernel.fp8_m_grouped_gemm.f61444ba7046': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj128ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7dbb351b3a5afe2bcad1f53ed256fc9eff900cb084d6476d23504fd128b9ca9e'),
    'kernel.fp8_m_grouped_gemm.13405f117b95': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj32ELj128ELj4ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ea7d3ecd28a69b5d3896986d50b739c6f97d24df9587b990955ad252126241d9'),
    'kernel.fp8_m_grouped_gemm.5273eaff13b8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj128ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '81bccf1231c3914a14af3b979e8e740e150a8d2925783e96c6ed78b975b7cd85'),
    'kernel.fp8_m_grouped_gemm.eabd2207ac17': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj128ELj4096ELj128ELj128ELj128ELj8ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3ed0b785eb42cbff0649996257c53465ca87c6dbf6ebc8c97777dad10cb313c4'),
    'kernel.fp8_m_grouped_gemm.49778ac9c667': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd57c8a972c8dc61888dd148087e4f936b73bd6c98355945e821d37bb83b7b5ff'),
    'kernel.fp8_m_grouped_gemm.01abe8ac39f9': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '7e8a47efb2d9029305f1539d1161be87425402647a4c2ab6e1ade6571a36b0d6'),
    'kernel.fp8_m_grouped_gemm.c619d155cb8f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '96129ee7c412d9084dab12655030a5f9df62e7f2bbd169b464017173b5d3b271'),
    'kernel.fp8_m_grouped_gemm.c2419a16cb48': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4f2cfb5d02cc994e8e72691acbc1589d95116c6abd5927d56dee33f979c5a13c'),
    'kernel.fp8_m_grouped_gemm.9f06bdc5289e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj128ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c7453802ec628beb86f57c776a50b112afe5965891e3b4b4b65b3c6216cedd5e'),
    'kernel.fp8_m_grouped_gemm.fc15347f796e': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj128ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '9e89d9cc9e7e1aff09d4a621c963c1f9b2262b8d38b0b2198e84a45dc5f90199'),
    'kernel.fp8_m_grouped_gemm.4db0a0684aa2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '855238b5d75746f908565a92cbc7ff221c42ede8c5fcd414ad376eb616be3e93'),
    'kernel.fp8_m_grouped_gemm.08051e88e138': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '8671f73e3624f18d23fe141ee189bd2417ce15ec91a6baad708cd6f71e5ca77d'),
    'kernel.fp8_m_grouped_gemm.6c3a266346e7': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '1e52f12ca56bc81624b83d77a98d4c6f2a4458b33b4603111fc0fb4e05493f78'),
    'kernel.fp8_m_grouped_gemm.f70058b01f81': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'eea2c17102324a7a02bcd4eda9ed096be0eec8667a7dff2e2e69c6efe3984843'),
    'kernel.fp8_m_grouped_gemm.4166ff1d78fc': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj512ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ea1084c2fbcbf1538d5cc958a71720568dc94f1648604c756afbace939b79179'),
    'kernel.fp8_m_grouped_gemm.223efdcae644': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj512ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '15c3dbdab9a32fc47f93436376877cd802443ab48b34aba5b694f2a2d1da8f57'),
    'kernel.fp8_m_grouped_gemm.6cb0320d6d23': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj16ELj128ELj1ELj128ELj128ELj32ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '621cdc47568241235b7e99039469854f56d606a7cff27aa19823819551880a55'),
    'kernel.fp8_m_grouped_gemm.22c8d7120f84': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj32ELj128ELj1ELj128ELj32ELj64ELj8ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '52b06ef9fe0707aba87ddbaa5e4266c6eb52e2252d7993b939ae2d8e1e3b43ad'),
    'kernel.fp8_m_grouped_gemm.27791478ab00': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '2834735e7b527380da1ddced7d4f15af45c7fb82801683b0dd261a45010f7536'),
    'kernel.fp8_m_grouped_gemm.41dea87fd6be': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj128ELj128ELj4ELj128ELj128ELj128ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5a0faec2e2851ecd544b2ea4f0168c4ab1c7482018268900af062c23daff4b00'),
    'kernel.fp8_m_grouped_gemm.6e76205d15c3': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj512ELj4096ELj128ELj176ELj128ELj8ELj128ELj128ELj32ELj5ELj2ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3145ec29e013480d286c5b8328f418f09da6b92d82c95fe3232aa5464b2580ce'),
    'kernel.fp8_m_grouped_gemm.de75d095d244': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj512ELj4096ELj128ELj192ELj128ELj8ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3edb2bdd5729c6c5c3d05163cb1f08d98ccc5ee2c5252f9da2823a4f5a40c607'),
    'kernel.fp8_m_grouped_gemm.ff60a177be62': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f6b9df9f025c095e61c1026ca3f2ec8b42dc32379e35c2d1dd78f98589051fb4'),
    'kernel.fp8_m_grouped_gemm.27edf09cd64c': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'e5ff18f9f4a61749c12d3a68dfd856b43389a581ca1c21512bc14d8a428f9de7'),
    'kernel.fp8_m_grouped_gemm.bd9ef159ca1f': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '54c069577bcec45b2065a03b318af311cf6c63f25d61e66c1768f2b9dc2e6cd7'),
    'kernel.fp8_m_grouped_gemm.15037c26bff2': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b6f60f84d8b594d6896483979884c0035948c2a675e5f47a50444797d3b47728'),
    'kernel.fp8_m_grouped_gemm.d5046aeaf1a6': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj128ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b6621f96fdf729fef1bc6cf8b47cae3a5baee7015b77e0a7c2696c75fd6d38b0'),
    'kernel.fp8_m_grouped_gemm.166e3fa97b55': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj128ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj1ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ad54fdf452e1743decac3ac1d302c220bc77391ec3e39a52a5eea33d1d68b199'),
    'kernel.fp8_m_grouped_gemm.3bf99a5d7f8d': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'd20131f0adee0778246f199a01806f039ecebe235ae48b396095d0aace8693fc'),
    'kernel.fp8_m_grouped_gemm.468c865ef8be': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'c67bbe98689ee6b44d32998ff5d5b7d6a814100657d80abedaf038fbf56f28b9'),
    'kernel.fp8_m_grouped_gemm.d84c0323cd7a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'b17d7b94aaf0aca4783ca2bb45f8ccdb558f6024b9a44b230e0bc3823931335b'),
    'kernel.fp8_m_grouped_gemm.0d364e3996d8': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '3e06abcfca223f927df362b2a468acdf3b94f615451240c1ed60fe40b8e193e1'),
    'kernel.fp8_m_grouped_gemm.b53f23fe8b35': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj512ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'f3cbf706ac0cbf68054b2c8fcf57c2cd0a538aab05696fe4b3c01141009fae1e'),
    'kernel.fp8_m_grouped_gemm.b631d84ad228': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj512ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'ff9d903828b211b50bed13b02cdffd05c16b1aaf83bd5732d2f7ac38cc8c90cf'),
    'kernel.fp8_m_grouped_gemm.75319cb78bd1': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj64ELj128ELj1ELj128ELj128ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '064542553ff3207136259ddf8fe2303f6c51a87ced9fcc5bf94d6811d0bc6193'),
    'kernel.fp8_m_grouped_gemm.dc984489a357': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj64ELj128ELj1ELj128ELj64ELj128ELj7ELj4ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '5df0b619c0f00a6262533cd7a575b577f2ccd39e409393a57220e4ae4a6c4b1d'),
    'kernel.fp8_m_grouped_gemm.adb182190e42': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj4ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '4a5ce70d40eaff0d371d7d91176b2bf4e16a2c5048df0ef78b3624bae01534ce'),
    'kernel.fp8_m_grouped_gemm.a23db73abcec': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj192ELj128ELj4ELj128ELj64ELj128ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', 'df60be2a4b6fc3afe3516b232330d3d7b863fa0d4290fd0cbed648d50bb7c0c9'),
    'kernel.fp8_m_grouped_gemm.e11d8197ff5a': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj4096ELj4096ELj128ELj240ELj128ELj8ELj128ELj128ELj32ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '16108227afb1bf80e1c426be98fbd2116da68be87f1fbc6cb26daa32bb1e6fa3'),
    'kernel.fp8_m_grouped_gemm.812550444fe4': ('_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_1ELj0ELj4096ELj4096ELj128ELj224ELj128ELj8ELj128ELj32ELj64ELj4ELj0ELj128ELj128ELj1ELb1ELNS_8GemmTypeE1ELb0EN7cutlass10bfloat16_tEEEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_', '429a1c97c70597a713e7e2e179d7fc2d9885467e29ce74c83b78aaac9454a6b8'),
}
# fmt: on
