# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Source-built Blackwell backends for the BF16 x FP4 GEMM.

Route selection, grid arithmetic and the split-K workspace live here; the
generated launchers in ``csrc/blackwell_bf16_fp4`` validate tensor metadata,
encode the TMA descriptors by value and launch exactly one kernel.
"""

from __future__ import annotations

import collections
import functools
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import torch

from ..utils import get_compute_capability, get_device_sm_count
from .gemm_bf16_fp4 import _unswizzle_sf_128x4
from .gemm_bf16_fp4_cute_dsl import (
    _cute_dsl_pack_fp4_weight,
    _e4m3_to_s0e5m3,
)

BlackwellBf16Fp4Backend = Literal["blackwell-native", "blackwell-tiled"]

TILE_M = 16
TILE_N = 64
NATIVE_TMA_K = 256
CUDA_MAX_GRID_Y = 65535
WARP_K = 128
SPLIT_K = 2
SPLIT_REDUCE_THREADS = 128
GROUP_M = 128
SPLIT_K_WORKSPACE_LIMIT = 16
# Bucket-selected seed kernels: the M16 K1024 kernel (every alpha/PDL variant) and the two
# exact-shape kernels (alpha and PDL enabled only); faster than the generic persistent kernels there.
SEED_M16_K = 1024
SEED_EXACT_M17_SHAPE = (17, 3072, 3072)
SEED_EXACT_M768_SHAPE = (768, 2112, 2048)
SEED_EXACT_M17_SM_MULTIPLIER = 2

GRID_COMPONENTS = frozenset(
    {
        "cudnn_tma_bf16",
        "cudnn_tma_f16",
        "cudnn_cp_async_bf16",
        "cudnn_cp_async_f16",
        "cute_bf16",
    }
)
WARP_TILE_M = {
    "cute_warp_mma_m16_k16_bf16": 16,
    "cute_warp_mma_m16_k32_bf16": 16,
    "cute_warp_mma_m16_k48_bf16": 16,
    "cute_warp_mma_m16_bf16": 16,
    "cute_warp_mma_m32_bf16": 32,
    "cute_warp_mma_m64_bf16": 64,
}


def _require_blackwell_source_arch(device: torch.device) -> None:
    major, minor = get_compute_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            "the source-built BF16 x FP4 backends require SM100 or SM103; "
            f"got SM{major}{minor}"
        )


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_arch(device_index: int) -> str:
    from ..jit.blackwell_bf16_fp4 import supported_arch

    capability = get_compute_capability(torch.device("cuda", device_index))
    arch = supported_arch(capability)
    if arch is None:
        raise NotImplementedError(
            f"cuda:{device_index} (compute capability {capability[0]}.{capability[1]}) is not a "
            "Blackwell BF16 x FP4 build target; set FLASHINFER_CUDA_ARCH_LIST to include it"
        )
    return arch


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(get_device_sm_count(torch.device("cuda", device_index)))


@functools.cache
def _unit_alpha(device_index: int) -> torch.Tensor:
    """The alpha operand of programs compiled without alpha; never read by them."""

    return torch.ones(
        (1,), dtype=torch.float32, device=torch.device("cuda", device_index)
    )


# Split-K partial sums, one buffer per (device, stream, M, N).  The buffer is private to the
# stream that launches the partial and reduce kernels, so launches on different streams never
# share it, and a buffer is allocated once per key instead of on every call (steady-state calls
# allocate nothing).  During CUDA Graph capture a missing buffer comes from the graph's private
# memory pool and is owned by the graph, never cached: replays reuse the captured address.
_SPLIT_K_WORKSPACES: "collections.OrderedDict[Tuple[int, int, int, int], torch.Tensor]" = collections.OrderedDict()


def _split_k_workspace(a: torch.Tensor, m: int, n: int) -> torch.Tensor:
    device_index = _device_index(a.device)
    stream = torch.cuda.current_stream(a.device)
    key = (device_index, int(stream.cuda_stream), m, n)
    workspace = _SPLIT_K_WORKSPACES.get(key)
    if workspace is not None:
        _SPLIT_K_WORKSPACES.move_to_end(key)
        return workspace
    workspace = torch.empty((SPLIT_K, m, n), dtype=torch.float32, device=a.device)
    if torch.cuda.is_current_stream_capturing():
        return workspace
    _SPLIT_K_WORKSPACES[key] = workspace
    while len(_SPLIT_K_WORKSPACES) > SPLIT_K_WORKSPACE_LIMIT:
        (old_device, old_stream, _m, _n), old = _SPLIT_K_WORKSPACES.popitem(last=False)
        # The evicted buffer may still be read by kernels queued on its stream; let the
        # caching allocator reuse it only after that work has completed.
        old.record_stream(
            torch.cuda.ExternalStream(
                old_stream, device=torch.device("cuda", old_device)
            )
        )
    return workspace


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _cell_key(
    component: str, has_alpha: bool, enable_pdl: bool, flat_grid: bool
) -> str:
    key = f"{component}:alpha{int(has_alpha)}:pdl{int(enable_pdl)}"
    if component in GRID_COMPONENTS:
        key += ":flat" if flat_grid else ":grid2d"
    return key


@dataclass(frozen=True)
class _Stage:
    cell: str
    module: str
    grid: Tuple[int, int, int]
    bindings: dict[str, Any]


@dataclass(frozen=True)
class _Plan:
    stages: Tuple[_Stage, ...]
    arch: str


def _persistent_grid(
    component: str, m: int, n: int, sm_count: int
) -> Tuple[int, int, int]:
    """Launch grid of a persistent or seed program."""

    grid_n = _ceil_div(n, TILE_N)
    if component == "cute_seed_m16_k1024_bf16":
        # Two N32 halves per M16 x N64 tile, one CTA each, no SM clamp.
        return (_ceil_div(m, 16) * grid_n * 2, 1, 1)
    if component == "cute_seed_exact_m17_n3072_k3072_bf16":
        tiles = _ceil_div(m, 16) * grid_n
        return (min(tiles, SEED_EXACT_M17_SM_MULTIPLIER * sm_count), 1, 1)
    if component == "cute_seed_exact_m768_n2112_k2048_bf16":
        return (_ceil_div(m, 64) * grid_n, 1, 1)
    return (min(_ceil_div(m, WARP_TILE_M[component]) * grid_n, sm_count), 1, 1)


def _select_route(
    m: int,
    n: int,
    k: int,
    *,
    tiled: bool,
    output_bf16: bool,
    has_alpha: bool,
    enable_pdl: bool,
    sm_count: int,
) -> Tuple[Tuple[str, bool, bool, bool, Tuple[int, int, int]], ...]:
    """Stages ``(component, has_alpha, enable_pdl, flat_grid, grid)`` for one problem."""

    cp_async = not tiled and k % NATIVE_TMA_K != 0
    if not tiled and output_bf16 and not cp_async and (m, n, k) == (768, 2112, 2048):
        grid = (_ceil_div(n, TILE_N), m // GROUP_M, 1)
        return (("cudnn_group_m128_bf16", has_alpha, enable_pdl, False, grid),)
    if not tiled and output_bf16 and not cp_async and (m, n, k) == (1, 4096, 4096):
        return (
            (
                "cudnn_split_k2_partial_f32",
                has_alpha,
                enable_pdl,
                False,
                (_ceil_div(n, TILE_N), 1, SPLIT_K),
            ),
            (
                "cudnn_split_k2_reduce_bf16",
                False,
                enable_pdl,
                False,
                (_ceil_div(m * n, SPLIT_REDUCE_THREADS), 1, 1),
            ),
        )
    if tiled:
        component = None
        if k == 16:
            component = "cute_warp_mma_m16_k16_bf16"
        elif k == 32:
            component = "cute_warp_mma_m16_k32_bf16"
        elif k == 48:
            component = "cute_warp_mma_m16_k48_bf16"
        elif m <= 16 and k == SEED_M16_K:
            component = "cute_seed_m16_k1024_bf16"
        elif m <= 16 and k >= WARP_K and k % 64 == 0:
            component = "cute_warp_mma_m16_bf16"
        elif m <= 32 and k >= WARP_K and k % 64 == 0:
            component = "cute_warp_mma_m32_bf16"
            if (m, n, k) == SEED_EXACT_M17_SHAPE and has_alpha and enable_pdl:
                component = "cute_seed_exact_m17_n3072_k3072_bf16"
        elif m >= 33 and k % WARP_K == 0:
            component = "cute_warp_mma_m64_bf16"
            if (m, n, k) == SEED_EXACT_M768_SHAPE and has_alpha and enable_pdl:
                component = "cute_seed_exact_m768_n2112_k2048_bf16"
        if component is not None:
            return (
                (
                    component,
                    has_alpha,
                    enable_pdl,
                    False,
                    _persistent_grid(component, m, n, sm_count),
                ),
            )
    grid_m = _ceil_div(m, TILE_M)
    grid_n = _ceil_div(n, TILE_N)
    flat_grid = grid_m > CUDA_MAX_GRID_Y
    if tiled:
        component = "cute_bf16"
    elif cp_async:
        component = "cudnn_cp_async_bf16" if output_bf16 else "cudnn_cp_async_f16"
    else:
        component = "cudnn_tma_bf16" if output_bf16 else "cudnn_tma_f16"
    grid = (grid_m * grid_n, 1, 1) if flat_grid else (grid_n, grid_m, 1)
    return ((component, has_alpha, enable_pdl, flat_grid, grid),)


def _plan(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    tiled: bool,
    enable_pdl: bool,
) -> _Plan:
    """Select the programs and launch grids for validated operands."""

    from ..jit.blackwell_bf16_fp4 import ROUTES

    index = _device_index(a.device)
    arch = _device_arch(index)
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(out.shape[1])
    stages_spec = _select_route(
        m,
        n,
        k,
        tiled=tiled,
        output_bf16=out.dtype == torch.bfloat16,
        has_alpha=alpha is not None,
        enable_pdl=enable_pdl,
        sm_count=_sm_count(index),
    )
    scale = b_descale if tiled else b_descale.view(torch.uint8)
    alpha_operand = alpha if alpha is not None else _unit_alpha(index)
    stages = []
    partials = None
    for component, has_alpha, pdl, flat_grid, grid in stages_spec:
        cell = _cell_key(component, has_alpha, pdl, flat_grid)
        if component == "cudnn_split_k2_partial_f32":
            partials = _split_k_workspace(a, m, n)
            bindings = dict(
                A=a,
                B=b,
                B_descale=scale,
                alpha=alpha_operand,
                C=partials,
                M=m,
                N=n,
                K=k,
            )
        elif component == "cudnn_split_k2_reduce_bf16":
            bindings = dict(partials=partials, C=out, elements=m * n)
        else:
            bindings = dict(
                A=a, B=b, B_descale=scale, alpha=alpha_operand, C=out, M=m, N=n, K=k
            )
        stages.append(
            _Stage(cell=cell, module=ROUTES[cell], grid=grid, bindings=bindings)
        )
    return _Plan(stages=tuple(stages), arch=arch)


def _launch(plan: _Plan) -> None:
    from ..jit.blackwell_bf16_fp4 import MODULES, load_module

    for stage in plan.stages:
        record = MODULES[stage.module]
        grid = dict(zip(("grid_x", "grid_y", "grid_z"), stage.grid, strict=True))
        arguments = []
        for kind, name in record["arg_plan"]:
            if kind == "grid":
                arguments.append(grid[name])
            else:
                arguments.append(stage.bindings[name])
        getattr(load_module(stage.module, plan.arch), record["ffi_entry"])(*arguments)


def _prepare_blackwell_bf16_fp4(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
    backend: BlackwellBf16Fp4Backend,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare one of the two explicit source-kernel tensor layouts (one-time weight transform)."""

    if block_size != 16:
        raise ValueError(
            f"source-built BF16 x FP4 preparation requires block_size=16; got {block_size}"
        )
    _require_blackwell_source_arch(b.device)
    if b.device != b_descale.device:
        raise ValueError(
            "b and b_descale must be on the same device; got "
            f"{b.device} and {b_descale.device}"
        )
    if alpha is not None and alpha.device != b.device:
        raise ValueError(
            f"alpha must be on the same device as b ({b.device}); got {alpha.device}"
        )

    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size
    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf).contiguous()

    if backend == "blackwell-native":
        return b.contiguous(), linear_sf.view(torch.float8_e4m3fn), alpha

    if backend == "blackwell-tiled":
        if n % 64 != 0:
            raise ValueError(
                f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
            )
        b_kn = b.t().contiguous()
        b_packed = _cute_dsl_pack_fp4_weight(b_kn)
        scale_s0e5m3 = _e4m3_to_s0e5m3(linear_sf.t().contiguous())
        return b_packed, scale_s0e5m3, alpha

    raise ValueError(f"unknown source-built BF16 x FP4 backend {backend!r}")


def _validate_alpha(alpha: Optional[torch.Tensor], a: torch.Tensor) -> None:
    if alpha is None:
        return
    if alpha.device != a.device:
        raise ValueError(
            f"alpha must be on the same device as the GEMM inputs ({a.device}); "
            f"got {alpha.device}"
        )
    if (
        alpha.dtype != torch.float32
        or tuple(alpha.shape) != (1,)
        or not alpha.is_contiguous()
    ):
        raise ValueError(
            "alpha must be a contiguous float32 tensor with shape (1,); "
            f"got dtype={alpha.dtype}, shape={tuple(alpha.shape)}, "
            f"contiguous={alpha.is_contiguous()}"
        )


def _validate_blackwell_native_layout(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
) -> int:
    if b.dim() != 2 or b.dtype != torch.uint8:
        raise ValueError(
            "blackwell-native expects b as contiguous uint8 [N, K/2]; "
            f"got dtype={b.dtype}, shape={tuple(b.shape)}"
        )
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    expected_scale_shape = (n, k // 16)
    if (
        b_descale.dtype != torch.float8_e4m3fn
        or tuple(b_descale.shape) != expected_scale_shape
    ):
        raise ValueError(
            "blackwell-native expects linear float8_e4m3fn scales "
            f"with shape {expected_scale_shape}; got dtype={b_descale.dtype}, "
            f"shape={tuple(b_descale.shape)}"
        )
    if int(a.shape[1]) != k:
        raise ValueError(
            f"a.shape[1]={int(a.shape[1])} but b.shape={tuple(b.shape)} encodes K={k}"
        )
    return n


def _validate_blackwell_tiled_layout(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
) -> int:
    if b.dim() != 2 or b.dtype != torch.int32 or int(b.shape[1]) % 2 != 0:
        raise ValueError(
            "blackwell-tiled expects b as contiguous int32 [K/16, N*2]; "
            f"got dtype={b.dtype}, shape={tuple(b.shape)}"
        )
    k_tiles = int(b.shape[0])
    n = int(b.shape[1]) // 2
    k = k_tiles * 16
    if n % 64 != 0:
        raise ValueError(
            f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
        )
    expected_scale_shape = (k_tiles, n)
    if b_descale.dtype != torch.uint8 or tuple(b_descale.shape) != expected_scale_shape:
        raise ValueError(
            "blackwell-tiled expects S0E5M3 uint8 scales "
            f"with shape {expected_scale_shape}; got dtype={b_descale.dtype}, "
            f"shape={tuple(b_descale.shape)}"
        )
    if int(a.shape[1]) != k:
        raise ValueError(
            f"a.shape[1]={int(a.shape[1])} but b.shape={tuple(b.shape)} encodes K={k}"
        )
    return n


def _compute_blackwell_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    enable_pdl: bool,
    backend: BlackwellBf16Fp4Backend,
) -> torch.Tensor:
    """Validate the prepared ABI, select the route and launch the generated programs."""

    _require_blackwell_source_arch(a.device)
    if a.dim() != 2 or a.dtype != torch.bfloat16:
        raise ValueError(
            "source-built BF16 x FP4 GEMM expects a as bfloat16 [M, K]; "
            f"got dtype={a.dtype}, shape={tuple(a.shape)}"
        )
    if block_size != 16 or int(a.shape[1]) % block_size != 0:
        raise ValueError(
            f"source-built BF16 x FP4 GEMM requires block_size=16 and K%16=0; "
            f"got block_size={block_size}, K={int(a.shape[1])}"
        )
    if not a.is_contiguous() or not b.is_contiguous() or not b_descale.is_contiguous():
        raise ValueError("a, b, and b_descale must be contiguous")
    if b.device != a.device or b_descale.device != a.device:
        raise ValueError(
            "a, b, and b_descale must be on the same device; got "
            f"{a.device}, {b.device}, and {b_descale.device}"
        )

    if backend == "blackwell-native":
        tiled = False
        n = _validate_blackwell_native_layout(a, b, b_descale)
        if out_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"blackwell-native supports bfloat16 or float16 output; got {out_dtype}"
            )
    elif backend == "blackwell-tiled":
        tiled = True
        n = _validate_blackwell_tiled_layout(a, b, b_descale)
        if out_dtype != torch.bfloat16:
            raise ValueError(
                f"blackwell-tiled requires bfloat16 output; got {out_dtype}"
            )
    else:
        raise ValueError(f"unknown source-built BF16 x FP4 backend {backend!r}")

    m = int(a.shape[0])
    if m <= 0 or n <= 0:
        raise ValueError(f"M and N must be positive; got M={m}, N={n}")
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    elif tuple(out.shape) != (m, n):
        raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
    elif out.dtype != out_dtype:
        raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")
    elif out.device != a.device or not out.is_contiguous():
        raise ValueError(
            "out must be contiguous and on the same device as a; got "
            f"device={out.device}, contiguous={out.is_contiguous()}"
        )
    _validate_alpha(alpha, a)

    _launch(
        _plan(a, b, b_descale, alpha, out, tiled=tiled, enable_pdl=bool(enable_pdl))
    )
    return out


__all__: list[str] = []
