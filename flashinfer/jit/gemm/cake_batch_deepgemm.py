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
        n=6144,
        k=7168,
        variant=4,
        symbol="kernel_flashinfer_blackwell_batch_deepgemm_fp8_seed_large_nk_cta1",
        source="cake_batch_deepgemm_fp8_short_m_n6144_k7168.cu",
        smem_bytes=203776,
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

_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_KIND = {"sm100a": 1000, "sm103a": 1003}


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
) -> torch.Tensor:
    key = (
        tensor.device,
        tensor.data_ptr(),
        data_type,
        global_dims,
        global_strides,
        box_dims,
        swizzle,
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
            cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, m, k = a.shape
    n = b.shape[1]
    swizzle_128 = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    uint8 = cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8
    if (n, k) == (4096, 7168):
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
        k_box = 1 if (n, k) == (512, 128) else 2
        b_rows = 128 if (n, k) == (512, 128) else 64
        a_desc = _tensor_map_device(
            a,
            data_type=uint8,
            global_dims=(128, m, k // 128, batch),
            global_strides=(k, 128, m * k),
            box_dims=(128, 128, k_box, 1),
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
    else:
        c_desc = a_desc
    return a_desc, b_desc, c_desc


def _select_route(n: int, k: int, expected_m: int) -> CakeBatchDeepGemmShape:
    if (n, k) == (128, 512):
        return "n128_k512"
    if (n, k) == (512, 128):
        return "n512_k128"
    if (n, k) == (4096, 7168):
        return "n4096_k7168"
    if (n, k) == (6144, 7168) and expected_m == 24:
        return "short_m_n6144_k7168"
    if (n, k) in _GENERIC_NK:
        return "large_nk"
    raise ValueError(f"unsupported Cake batch DeepGEMM shape: N={n}, K={k}")


def run_cake_batch_deepgemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    expected_m: int,
) -> None:
    shape = _select_route(b.shape[1], a.shape[2], expected_m)
    capability = torch.cuda.get_device_capability(a.device)
    target = {(10, 0): "sm100a", (10, 3): "sm103a"}.get(capability)
    if target is None:
        raise ValueError(
            f"Cake batch DeepGEMM requires SM100 or SM103, got {capability}"
        )
    a_desc, b_desc, c_desc = _tensor_maps(a, b, out)
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
        expected_m,
    )


__all__ = [
    "CAKE_BATCH_DEEPGEMM_METADATA",
    "gen_cake_batch_deepgemm_module",
    "get_cake_batch_deepgemm_module",
    "run_cake_batch_deepgemm_fp8",
]
