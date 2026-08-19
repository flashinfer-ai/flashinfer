"""
Copyright (c) 2026 by FlashInfer team.

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

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import Literal, NamedTuple

import cuda.bindings.driver as cbd
import torch

from ...cuda_utils import checkCudaErrors
from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from ..utils import write_if_different

CakeBatchDeepGemmShape = Literal[
    "n128_k512",
    "n512_k128",
    "n4096_k7168",
    "large_nk",
    "short_m_n6144_k7168",
    "pack_scales_m224",
    "swap_m224",
    "tail128",
    "m32_n4096_k7168_s6e1_g1",
    "m32_n4096_k7168_s5e2_g8",
    "m32_n7168_k2048_s5e3_g8",
    "m32_n7168_k2048_s5e2_g8",
]
CakeBatchDeepGemmTarget = Literal["sm100a", "sm103a"]


class CakeBatchDeepGemmMetadata(NamedTuple):
    n: int
    k: int
    variant: int
    symbol: str
    source: str
    smem_bytes: int
    use_fast_math: bool


CAKE_BATCH_DEEPGEMM_METADATA: dict[
    CakeBatchDeepGemmShape, CakeBatchDeepGemmMetadata
] = {
    "n128_k512": CakeBatchDeepGemmMetadata(
        n=128,
        k=512,
        variant=0,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n128_k512",
        source="cake_batch_deepgemm_fp8_n128_k512.cu",
        smem_bytes=103424,
        use_fast_math=True,
    ),
    "n512_k128": CakeBatchDeepGemmMetadata(
        n=512,
        k=128,
        variant=1,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n512_k128",
        source="cake_batch_deepgemm_fp8_n512_k128.cu",
        smem_bytes=50176,
        use_fast_math=True,
    ),
    "n4096_k7168": CakeBatchDeepGemmMetadata(
        n=4096,
        k=7168,
        variant=2,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n4096_k7168",
        source="cake_batch_deepgemm_fp8_n4096_k7168.cu",
        smem_bytes=203776,
        use_fast_math=False,
    ),
    "large_nk": CakeBatchDeepGemmMetadata(
        n=7168,
        k=2048,
        variant=3,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_n7168_k2048",
        source="cake_batch_deepgemm_fp8_large_nk.cu",
        smem_bytes=205824,
        use_fast_math=False,
    ),
    "short_m_n6144_k7168": CakeBatchDeepGemmMetadata(
        n=0,
        k=0,
        variant=4,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_large_nk_cta1",
        source="cake_batch_deepgemm_fp8_short_m_n6144_k7168.cu",
        smem_bytes=203776,
        use_fast_math=False,
    ),
    "pack_scales_m224": CakeBatchDeepGemmMetadata(
        n=0,
        k=0,
        variant=5,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_pack_scales_m224",
        source="cake_batch_deepgemm_fp8_pack_scales_m224.cu",
        smem_bytes=10752,
        use_fast_math=False,
    ),
    "swap_m224": CakeBatchDeepGemmMetadata(
        n=0,
        k=0,
        variant=6,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_swap_m224",
        source="cake_batch_deepgemm_fp8_swap_m224.cu",
        smem_bytes=202752,
        use_fast_math=False,
    ),
    "tail128": CakeBatchDeepGemmMetadata(
        n=0,
        k=0,
        variant=7,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_large_nk_cta1_tail128",
        source="cake_batch_deepgemm_fp8_tail128.cu",
        smem_bytes=203776,
        use_fast_math=False,
    ),
    "m32_n4096_k7168_s6e1_g1": CakeBatchDeepGemmMetadata(
        n=4096,
        k=7168,
        variant=8,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_swap_m32",
        source="cake_batch_deepgemm_fp8_m32_n4096_k7168_s6e1_g1.cu",
        smem_bytes=232448,
        use_fast_math=False,
    ),
    "m32_n4096_k7168_s5e2_g8": CakeBatchDeepGemmMetadata(
        n=4096,
        k=7168,
        variant=9,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_swap_m32",
        source="cake_batch_deepgemm_fp8_m32_n4096_k7168_s5e2_g8.cu",
        smem_bytes=198656,
        use_fast_math=False,
    ),
    "m32_n7168_k2048_s5e3_g8": CakeBatchDeepGemmMetadata(
        n=7168,
        k=2048,
        variant=10,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_swap_m32",
        source="cake_batch_deepgemm_fp8_m32_n7168_k2048_s5e3_g8.cu",
        smem_bytes=202752,
        use_fast_math=False,
    ),
    "m32_n7168_k2048_s5e2_g8": CakeBatchDeepGemmMetadata(
        n=7168,
        k=2048,
        variant=11,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_swap_m32",
        source="cake_batch_deepgemm_fp8_m32_n7168_k2048_s5e2_g8.cu",
        smem_bytes=198656,
        use_fast_math=False,
    ),
}

_GENERIC_NK = frozenset(
    {
        (7168, 2048),
        (6144, 7168),
        (7168, 3072),
        (4096, 4096),
        (4096, 2048),
    }
)
_SWAP_M224_PROFILE_NK = frozenset(
    {
        (6144, 7168),
        (7168, 3072),
        (4096, 4096),
        (4096, 2048),
    }
)

_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_KIND = {"sm100a": 1000, "sm103a": 1003}
_CAPABILITY_TARGETS: dict[tuple[int, int], CakeBatchDeepGemmTarget] = {
    (10, 0): "sm100a",
    (10, 3): "sm103a",
}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[3] / "csrc" / "cake"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("Cake batch DeepGEMM FP8 sources were not found")


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[3] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _binding_source(metadata: CakeBatchDeepGemmMetadata, target: str) -> str:
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#define FLASHINFER_CAKE_BATCH_DEEPGEMM_BODY_FILE \"{metadata.source}\"
#define FLASHINFER_CAKE_BATCH_DEEPGEMM_KERNEL {metadata.symbol}
#define FLASHINFER_CAKE_BATCH_DEEPGEMM_VARIANT {metadata.variant}
#define FLASHINFER_CAKE_BATCH_DEEPGEMM_N {metadata.n}
#define FLASHINFER_CAKE_BATCH_DEEPGEMM_K {metadata.k}
#define FLASHINFER_CAKE_BATCH_DEEPGEMM_SMEM_BYTES {metadata.smem_bytes}
#define FLASHINFER_CAKE_BATCH_DEEPGEMM_TARGET_KIND {_TARGET_KIND[target]}
#include \"cake_batch_deepgemm_fp8_binding.cuh\"
"""


@functools.cache
def gen_cake_batch_deepgemm_module(
    shape: CakeBatchDeepGemmShape,
    target: CakeBatchDeepGemmTarget,
) -> JitSpec:
    if shape not in CAKE_BATCH_DEEPGEMM_METADATA:
        raise ValueError(f"unsupported Cake batch DeepGEMM shape: {shape}")
    if target not in _NVCC_FLAGS:
        raise ValueError(f"unsupported Cake batch DeepGEMM target: {target}")
    metadata = CAKE_BATCH_DEEPGEMM_METADATA[shape]
    csrc_dir = _get_csrc_dir()
    uri = f"cake_batch_deepgemm_fp8_{shape}_{target}"
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / f"{uri}_binding.cu"
    write_if_different(binding, _binding_source(metadata, target))
    extra_flags = [*_NVCC_FLAGS[target]]
    if metadata.use_fast_math:
        extra_flags.append("--use_fast_math")
    return gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=extra_flags,
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
    )


@functools.cache
def get_cake_batch_deepgemm_module(
    shape: CakeBatchDeepGemmShape,
    target: CakeBatchDeepGemmTarget,
):
    module = gen_cake_batch_deepgemm_module(shape, target).build_and_load()
    logger.info("Loaded Cake batch DeepGEMM FP8 %s %s module", shape, target)
    return module


_TENSOR_MAP_BYTES = 128
_TENSOR_MAP_CACHE: dict[tuple, torch.Tensor] = {}


def _tensor_map_device(
    tensor: torch.Tensor,
    *,
    data_type,
    global_dims: tuple[int, ...],
    global_strides: tuple[int, ...],
    box_dims: tuple[int, ...],
    swizzle,
    l2_promotion=cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
) -> torch.Tensor:
    key = (
        tensor.device,
        tensor.data_ptr(),
        data_type,
        global_dims,
        global_strides,
        box_dims,
        swizzle,
        l2_promotion,
    )
    cached = _TENSOR_MAP_CACHE.get(key)
    if cached is not None:
        return cached
    rank = len(global_dims)
    tensor_map = checkCudaErrors(
        cbd.cuTensorMapEncodeTiled(
            data_type,
            rank,
            tensor.data_ptr(),
            tuple(cbd.cuuint64_t(value) for value in global_dims),
            tuple(cbd.cuuint64_t(value) for value in global_strides),
            tuple(cbd.cuuint32_t(value) for value in box_dims),
            (cbd.cuuint32_t(1),) * rank,
            cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            l2_promotion,
            cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    )
    host_ptr = tensor_map.getPtr()
    raw = bytes((ctypes.c_ubyte * _TENSOR_MAP_BYTES).from_address(host_ptr))
    host = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    with torch.cuda.device(tensor.device):
        device_desc = host.to(device=tensor.device)
        torch.cuda.current_stream(tensor.device).synchronize()
    _TENSOR_MAP_CACHE[key] = device_desc
    return device_desc


def _tensor_maps(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    shape: CakeBatchDeepGemmShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, m, k = a.shape
    n = b.shape[1]
    swizzle_128 = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    uint8 = cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    if shape == "n4096_k7168":
        a_desc = _tensor_map_device(
            a,
            data_type=uint8,
            global_dims=(128, batch * m, k // 128),
            global_strides=(k, 128),
            box_dims=(128, 128, 2),
            swizzle=swizzle_128,
        )
        b_desc = _tensor_map_device(
            b,
            data_type=uint8,
            global_dims=(128, batch * n, k // 128),
            global_strides=(k, 128),
            box_dims=(128, 128, 2),
            swizzle=swizzle_128,
        )
    else:
        k_box = 1 if shape == "n512_k128" else 2
        a_rows = 112 if shape == "swap_m224" else 128
        b_rows = (
            128
            if shape in {"n512_k128", "short_m_n6144_k7168", "swap_m224", "tail128"}
            else 64
        )
        a_desc = _tensor_map_device(
            a,
            data_type=uint8,
            global_dims=(128, m, k // 128, batch),
            global_strides=(k, 128, m * k),
            box_dims=(128, a_rows, k_box, 1),
            swizzle=swizzle_128,
        )
        b_desc = _tensor_map_device(
            b,
            data_type=uint8,
            global_dims=(128, n, k // 128, batch),
            global_strides=(k, 128, n * k),
            box_dims=(128, b_rows, k_box, 1),
            swizzle=swizzle_128,
        )
    if (n, k) == (512, 128):
        c_desc = _tensor_map_device(
            out,
            data_type=cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            global_dims=(n, m, batch),
            global_strides=(n * 2, m * n * 2),
            box_dims=(64, 128, 1),
            swizzle=cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
        )
    elif shape == "swap_m224":
        c_desc = _tensor_map_device(
            out,
            data_type=cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            global_dims=(n, batch * m),
            global_strides=(n * 2,),
            box_dims=(64, 16),
            swizzle=cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
            l2_promotion=cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        )
    else:
        c_desc = a_desc
    return a_desc, b_desc, c_desc


def _packed_m32_tensor_maps(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the five L2-promoted TMA surfaces consumed by the packed ABI."""

    batch, m, k = a.shape
    n = b.shape[1]
    packed_cols = k // (128 * 4)
    swizzle_128 = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    no_swizzle = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
    l2_256b = cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    uint8 = cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    uint32 = cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32
    a_desc = _tensor_map_device(
        a,
        data_type=uint8,
        global_dims=(128, m, k // 128, batch),
        global_strides=(k, 128, m * k),
        box_dims=(128, 16, 2, 1),
        swizzle=swizzle_128,
        l2_promotion=l2_256b,
    )
    b_desc = _tensor_map_device(
        b,
        data_type=uint8,
        global_dims=(128, n, k // 128, batch),
        global_strides=(k, 128, n * k),
        box_dims=(128, 128, 2, 1),
        swizzle=swizzle_128,
        l2_promotion=l2_256b,
    )
    c_desc = _tensor_map_device(
        out,
        data_type=cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        global_dims=(n, batch * m),
        global_strides=(n * 2,),
        box_dims=(64, 16),
        swizzle=swizzle_128,
        l2_promotion=l2_256b,
    )
    sfa_desc = _tensor_map_device(
        a_scale,
        data_type=uint32,
        global_dims=(m, batch * packed_cols),
        global_strides=(m * 4,),
        box_dims=(32, 1),
        swizzle=no_swizzle,
        l2_promotion=l2_256b,
    )
    sfb_desc = _tensor_map_device(
        b_scale,
        data_type=uint32,
        global_dims=(n, batch * packed_cols),
        global_strides=(n * 4,),
        box_dims=(128, 1),
        swizzle=no_swizzle,
        l2_promotion=l2_256b,
    )
    return a_desc, b_desc, c_desc, sfa_desc, sfb_desc


def _select_packed_m32_route(
    n: int, k: int, expected_m: int
) -> CakeBatchDeepGemmShape:
    if (n, k) == (4096, 7168):
        if expected_m <= 1:
            return "m32_n4096_k7168_s6e1_g1"
        return "m32_n4096_k7168_s5e2_g8"
    if (n, k) == (7168, 2048):
        if expected_m <= 1:
            return "m32_n7168_k2048_s5e3_g8"
        return "m32_n7168_k2048_s5e2_g8"
    raise ValueError(f"unsupported Cake packed batch DeepGEMM shape: N={n}, K={k}")


def _packed_m32_cta_reserve(m: int, k: int, expected_m: int) -> int:
    if k == 7168:
        if expected_m <= 1:
            return 0
        if expected_m <= 3:
            return 16
        if expected_m == 4:
            return 24
        if expected_m < 8 or m >= 512:
            return 16
        return 24
    if expected_m <= 1:
        return 0
    if expected_m == 2:
        return 20
    return 24


def _select_route(
    batch: int, m: int, n: int, k: int, expected_m: int
) -> CakeBatchDeepGemmShape | Literal["m224_tail128"]:
    if (n, k) == (128, 512):
        return "n128_k512"
    if (n, k) == (512, 128):
        return "n512_k128"
    if (batch, m, n, k) == (64, 256, 7168, 2048):
        return "m224_tail128"
    if (
        (expected_m in {230, 1228} and (n, k) in _SWAP_M224_PROFILE_NK)
        or (batch, m, n, k) == (1, 8192, 7168, 2048)
        or (batch, m, n, k) == (1, 16384, 4096, 7168)
        or (batch, m, n, k) == (8, 1024, 4096, 7168)
    ):
        return "swap_m224"
    if (n, k) == (4096, 7168):
        return "n4096_k7168"
    if ((n, k) == (7168, 2048) and m <= 256) or (
        expected_m == 24 and (n, k) in _GENERIC_NK
    ):
        return "short_m_n6144_k7168"
    if (n, k) in _GENERIC_NK:
        return "large_nk"
    raise ValueError(f"unsupported Cake batch DeepGEMM shape: N={n}, K={k}")


_PACKED_SCALE_WORKSPACE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _packed_scale_workspaces(
    a: torch.Tensor, batch: int, m: int, n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    stream = torch.cuda.current_stream(a.device)
    key = (a.device, stream.cuda_stream, batch, m, n, k)
    cached = _PACKED_SCALE_WORKSPACE_CACHE.get(key)
    if cached is not None:
        return cached
    packed_cols = k // (128 * 4)
    with torch.cuda.device(a.device):
        cached = (
            torch.empty((batch, packed_cols, m), dtype=torch.int32, device=a.device),
            torch.empty(
                (batch, packed_cols, n // 128), dtype=torch.int32, device=a.device
            ),
        )
    _PACKED_SCALE_WORKSPACE_CACHE[key] = cached
    return cached


def _run_module(
    shape: CakeBatchDeepGemmShape,
    target: CakeBatchDeepGemmTarget,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    sfa_packed: torch.Tensor,
    sfb_packed: torch.Tensor,
    expected_m: int,
    compute_m_cap: int,
    *,
    descriptor_shape: CakeBatchDeepGemmShape | None = None,
) -> None:
    a_desc, b_desc, c_desc = _tensor_maps(a, b, out, descriptor_shape or shape)
    get_cake_batch_deepgemm_module(shape, target).run(
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        out,
        a_desc,
        b_desc,
        c_desc,
        sfa_packed,
        sfb_packed,
        expected_m,
        compute_m_cap,
    )


def _run_packed_m32_module(
    shape: CakeBatchDeepGemmShape,
    target: CakeBatchDeepGemmTarget,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    expected_m: int,
    cta_reserve: int,
) -> None:
    a_desc, b_desc, c_desc, sfa_desc, sfb_desc = _packed_m32_tensor_maps(
        a, b, a_scale, b_scale, out
    )
    get_cake_batch_deepgemm_module(shape, target).run(
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        out,
        a_desc,
        b_desc,
        c_desc,
        sfa_desc,
        sfb_desc,
        expected_m,
        cta_reserve,
    )


def run_cake_batch_deepgemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    expected_m: int,
) -> None:
    batch, m, k = a.shape
    n = b.shape[1]
    route = _select_route(batch, m, n, k, expected_m)
    capability = torch.cuda.get_device_capability(a.device)
    target = _CAPABILITY_TARGETS.get(capability)
    if target is None:
        raise ValueError(
            f"Cake batch DeepGEMM requires SM100 or SM103, got {capability}"
        )
    if a_scale.dtype == torch.int32 and b_scale.dtype == torch.int32:
        packed_route = _select_packed_m32_route(n, k, expected_m)
        _run_packed_m32_module(
            packed_route,
            target,
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            out,
            expected_m,
            _packed_m32_cta_reserve(m, k, expected_m),
        )
        return
    if route in {"swap_m224", "m224_tail128"}:
        sfa_packed, sfb_packed = _packed_scale_workspaces(a, batch, m, n, k)
        _run_module(
            "pack_scales_m224",
            target,
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            out,
            sfa_packed,
            sfb_packed,
            expected_m,
            224 if route == "m224_tail128" else m,
            descriptor_shape="swap_m224",
        )
        _run_module(
            "swap_m224",
            target,
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            out,
            sfa_packed,
            sfb_packed,
            expected_m,
            224 if route == "m224_tail128" else m,
        )
        if route == "m224_tail128":
            _run_module(
                "tail128",
                target,
                a,
                b,
                a_scale,
                b_scale,
                masked_m,
                out,
                sfa_packed,
                sfb_packed,
                expected_m,
                224,
            )
        return

    assert route != "m224_tail128"
    _run_module(
        route,
        target,
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        out,
        a_scale.view(torch.int32),
        b_scale.view(torch.int32),
        expected_m,
        m,
    )


__all__ = [
    "CAKE_BATCH_DEEPGEMM_METADATA",
    "gen_cake_batch_deepgemm_module",
    "get_cake_batch_deepgemm_module",
    "run_cake_batch_deepgemm_fp8",
]
