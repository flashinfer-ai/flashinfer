# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared contiguous grouped FP8 GEMM launches on Blackwell SM100a.

This is the generated Weave-IR program family for the same mathematical
contract as :func:`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous`:
``out[r, :] = sum_q (a[r, q-block] . b[g_r, :, q-block]) * a_scale[r, q] *
b_scale[g_r, :, q]`` with FP8 E4M3 operands, per-row 128-wide K-block A scales,
128x128 B block scales, ordered FP32 K-block accumulation and a BF16 output.
Rows are routed to experts by the sorted ``m_indices`` vector.  Unlike the
CuTe-DSL kernel, internal expert boundaries need not be aligned to 128 rows.

The host side selects one exact kernel route per problem shape and output
alignment, resolves the launch grid from the device SM count, and binds the
positional argument plan of the registered program.  Descriptor storage for
the pointer TMA ABI is private to each prepared launch: the first ``launch()``
initializes it synchronously and must run outside CUDA Graph capture; later
launches only submit work on the current stream and may be captured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from ..jit.gemm.cake_grouped_fp8_gemm import (
    MODULES,
    ROUTE_GEOMETRY,
    SUPPORTED_COMPUTE_CAPABILITIES,
    generated_program_available,
    load_cake_grouped_fp8_gemm_module,
    select_module,
)

B_SCALE_BLOCK_N = 128
K_BLOCK = 128

# Route names double as the generated-program template names.
K128_ROUTE = "k128_exact"
K384_ROUTE = "k384_exact_three_partial"
DEEPK_C2_ROUTE = "deepk_c2_k_multiple_512"
DEEPK_C2_AB6_SCALE1_ROUTE = "deepk_c2_ab6_scale1_n256_or_m4096"
DEEPK_C2_KG1_ROUTE = "deepk_c2_kg1_non_kg4_k_blocks"
DEEPK_C2_SCALAR_OUTPUT_ROUTE = "deepk_c2_scalar_output"
DEEPK_N256_ROUTE = "deepk_n256_ab4_three_panel_kg4"
DEEPK_CG2_RECURRENCE_ROUTE = "deepk_cg2_ab5_x16_pair_wait_kg4"
DEEPK_CG2_FOUR_LOAD_ROUTE = "deepk_cg2_ab5_x16_four_load_wait_kg4"
DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE = "deepk_cg2_ab5_x16_four_load_wait_grid128_kg4"
DEEPK_CG2_RECURRENCE_EARLY4_ROUTE = "deepk_cg2_ab6_early4_output_alias_kg4"
DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE = "deepk_cg2_ab7_bscale_prefetch_kg4"

_CG2_EARLY4_SHAPES = frozenset({(16384, 2048, 4096)})
_CG2_SELECTED_SHAPES = frozenset(
    {
        (16384, 2048, 4096),
        (16384, 4096, 1024),
        (4096, 2048, 4096),
        (4096, 4096, 1024),
    }
)
# Measured CTA cap: these two shapes run their two-CTA routes on 128 CTAs.
_CG2_GRID128_SHAPES = frozenset({(4096, 2048, 4096), (4096, 4096, 1024)})


def route_for_shape(m: int, n: int, k: int) -> str:
    """Select the exact kernel route for a 16-byte-aligned BF16 output."""
    if (m, n, k) == (4096, 2048, 4096):
        return DEEPK_CG2_AB7_BSCALE_PREFETCH_ROUTE
    if (m, n, k) == (4096, 4096, 1024):
        return DEEPK_CG2_FOUR_LOAD_ROUTE
    if (m, n, k) in _CG2_EARLY4_SHAPES:
        return DEEPK_CG2_RECURRENCE_EARLY4_ROUTE
    if (m, n, k) in _CG2_SELECTED_SHAPES:
        return DEEPK_CG2_RECURRENCE_ROUTE
    if k == 128:
        return K128_ROUTE
    if k == 384:
        return K384_ROUTE
    if k > 0 and k % K_BLOCK == 0:
        if (k // K_BLOCK) % 4:
            return DEEPK_C2_KG1_ROUTE
        if (m == 16384 and n in {2048, 4096}) or (m, n, k) == (4096, 2048, 4096):
            return DEEPK_N256_ROUTE
        if n == 256 or m == 4096:
            return DEEPK_C2_AB6_SCALE1_ROUTE
        return DEEPK_C2_ROUTE
    raise ValueError(
        f"contiguous grouped FP8 GEMM requires positive K divisible by {K_BLOCK}; got K={k}"
    )


def resolve_route_for_grid(route: str, n: int, grid: tuple[int, int, int]) -> str:
    if route == DEEPK_CG2_FOUR_LOAD_ROUTE and n == 4096 and grid == (128, 1, 1):
        return DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE
    return route


def route_geometry(route: str) -> tuple[int, int, int]:
    """``(tile_m, tile_n, cluster_ctas)`` of one route from the export registry."""
    geometry = ROUTE_GEOMETRY.get(route)
    if geometry is None:
        raise NotImplementedError(
            f"route {route!r} has no registered tile geometry in this checkout"
        )
    return (
        int(geometry["tile_m"]),
        int(geometry["tile_n"]),
        int(geometry["cluster_ctas"]),
    )


def launch_plan(
    m: int, n: int, k: int, *, sm_count: int, scalar_output: bool
) -> tuple[str, tuple[int, int, int]]:
    """Resolve ``(route, grid)`` for a problem shape on a device with ``sm_count`` SMs."""
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    route = DEEPK_C2_SCALAR_OUTPUT_ROUTE if scalar_output else route_for_shape(m, n, k)
    tile_m, tile_n, cluster_ctas = route_geometry(route)
    total_tiles = math.ceil(m / tile_m) * math.ceil(n / tile_n)
    cta_limit = sm_count
    if (m, n, k) in _CG2_GRID128_SHAPES:
        cta_limit = min(cta_limit, 128)
    clusters = min(total_tiles, cta_limit // cluster_ctas)
    if clusters <= 0:
        raise ValueError("device cannot schedule one complete kernel cluster")
    grid = (clusters * cluster_ctas, 1, 1)
    return resolve_route_for_grid(route, n, grid), grid


def _require_tensor(
    tensor: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    alignment: int = 16,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if int(tensor.data_ptr()) % alignment:
        raise ValueError(f"{name} storage must be at least {alignment}-byte aligned")
    return tensor


def _validate_indices(m_indices: torch.Tensor, groups: int) -> None:
    """Synchronizing check of the routing contract (sorted, in range)."""
    indices = m_indices.to(torch.int64)
    lowest = int(indices.min())
    highest = int(indices.max())
    if lowest < 0 or highest >= groups:
        raise ValueError(
            "m_indices must satisfy 0 <= index < num_groups; -1 padding is unsupported"
        )
    if indices.numel() > 1 and not bool((indices[1:] >= indices[:-1]).all()):
        raise ValueError("m_indices must be sorted in nondecreasing order")


@dataclass
class PreparedGroupGemmFp8NtGroupwiseContiguous:
    """One prepared contiguous grouped FP8 GEMM launch.

    Tensor storage, shapes and dtypes are bound at preparation; tensor
    *contents* may change between launches.  ``launch()`` submits exactly one
    kernel on PyTorch's current stream for the bound device and returns the
    output tensor.  The first launch initializes private TMA descriptor storage
    and must run outside CUDA Graph capture.
    """

    route: str
    module_name: str
    grid: tuple[int, int, int]
    out: torch.Tensor
    _entry: Callable[..., Any]
    _arguments: tuple[Any, ...]
    _descriptor_storage: torch.Tensor

    def launch(self) -> torch.Tensor:
        with torch.cuda.device(self.out.device), tvm_ffi.use_torch_stream():
            self._entry(*self._arguments)
        return self.out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.grid[0])


def is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available(
    device: torch.device,
) -> bool:
    """True when a generated program family is registered for ``device``."""
    return device.type == "cuda" and generated_program_available(device)


def prepare_group_gemm_fp8_nt_groupwise_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
) -> PreparedGroupGemmFp8NtGroupwiseContiguous:
    r"""Prepare a contiguous grouped FP8 GEMM launch on SM100a.

    Parameters
    ----------
    a : torch.Tensor
        Contiguous FP8 E4M3 input of shape ``(M, K)``; ``M > 0`` and ``K`` a
        positive multiple of 128.
    b : torch.Tensor
        Contiguous FP8 E4M3 expert weights of shape ``(G, N, K)``; ``N`` a
        positive multiple of 128.
    a_scale : torch.Tensor
        Contiguous float32 scales of shape ``(M, K // 128)``.
    b_scale : torch.Tensor
        Contiguous float32 scales of shape ``(G, N // 128, K // 128)``.
    m_indices : torch.Tensor
        Contiguous int32 expert indices of shape ``(M,)``, sorted in
        nondecreasing order with ``0 <= index < G``.  Expert boundaries may
        fall at any row; empty experts are allowed.
    out : Optional[torch.Tensor]
        Contiguous bfloat16 output of shape ``(M, N)``.  Allocated if omitted.
        A 2-byte-aligned output whose address is not a multiple of 16 selects
        the scalar-store route.
    validate_indices : bool
        Check index range and sortedness with a device synchronization.

    Returns
    -------
    PreparedGroupGemmFp8NtGroupwiseContiguous
        Call ``.launch()`` to run the GEMM on the current stream.

    Notes
    -----
    Requires an SM100a (compute capability 10.0) device and a registered
    generated program for the selected route.  All tensors must live on the
    same CUDA device and, except ``out``, be 16-byte aligned.  Index values are
    unchecked unless ``validate_indices=True``; violating the routing contract
    is undefined behavior.
    """
    if not isinstance(a, torch.Tensor) or a.device.type != "cuda":
        raise ValueError("a must be a CUDA torch.Tensor")
    device = a.device
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise NotImplementedError(
            "prepared contiguous grouped FP8 GEMM requires an SM100a device "
            f"(compute capability 10.0); got {capability}"
        )
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("a must have shape (M, K) and b must have shape (G, N, K)")
    m, k = (int(v) for v in a.shape)
    groups, n = (int(v) for v in b.shape[:2])
    if m <= 0 or groups <= 0:
        raise ValueError("contiguous grouped FP8 GEMM requires positive M and G")
    if n <= 0 or n % B_SCALE_BLOCK_N:
        raise ValueError(f"N must be a positive multiple of {B_SCALE_BLOCK_N}, got {n}")
    if k <= 0 or k % K_BLOCK:
        raise ValueError(f"K must be a positive multiple of {K_BLOCK}, got {k}")
    a = _require_tensor(a, "a", shape=(m, k), dtype=torch.float8_e4m3fn, device=device)
    b = _require_tensor(
        b, "b", shape=(groups, n, k), dtype=torch.float8_e4m3fn, device=device
    )
    a_scale = _require_tensor(
        a_scale, "a_scale", shape=(m, k // K_BLOCK), dtype=torch.float32, device=device
    )
    b_scale = _require_tensor(
        b_scale,
        "b_scale",
        shape=(groups, n // B_SCALE_BLOCK_N, k // K_BLOCK),
        dtype=torch.float32,
        device=device,
    )
    m_indices = _require_tensor(
        m_indices, "m_indices", shape=(m,), dtype=torch.int32, device=device
    )
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    out = _require_tensor(
        out, "out", shape=(m, n), dtype=torch.bfloat16, device=device, alignment=2
    )
    if validate_indices:
        _validate_indices(m_indices, groups)

    scalar_output = bool(int(out.data_ptr()) % 16)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    route, grid = launch_plan(m, n, k, sm_count=sm_count, scalar_output=scalar_output)
    module_name = select_module(arch, route)
    record = MODULES[module_name]
    module = load_cake_grouped_fp8_gemm_module(module_name)
    entry = getattr(module, record["ffi_entry"])

    # The scalar-store route never reads its output tensor map; the validated
    # A tensor provides an aligned BF16 view with a compatible descriptor.
    c_tma = a.view(torch.bfloat16) if route == DEEPK_C2_SCALAR_OUTPUT_ROUTE else out
    bindings: dict[str, Any] = {
        "A": a.view(torch.uint8),
        "B": b.view(torch.uint8),
        "C_tma": c_tma,
        "C": out,
        "a_scale": a_scale,
        "b_scale": b_scale,
        "m_indices": m_indices,
        "M": m,
        "N": n,
        "K": k,
        "G": groups,
    }
    workspace_bytes = int(record["tma_workspace_bytes"])
    descriptor_storage = torch.empty(
        max(workspace_bytes, 128), dtype=torch.uint8, device=device
    )
    grid_by_axis = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid_by_axis[name])
        elif kind == "workspace":
            arguments.append(descriptor_storage)
        elif name in bindings:
            arguments.append(bindings[name])
        else:
            raise RuntimeError(
                f"generated program {module_name} binds {name!r}, which this host plan does not declare"
            )
    return PreparedGroupGemmFp8NtGroupwiseContiguous(
        route=route,
        module_name=module_name,
        grid=grid,
        out=out,
        _entry=entry,
        _arguments=tuple(arguments),
        _descriptor_storage=descriptor_storage,
    )


__all__ = [
    "PreparedGroupGemmFp8NtGroupwiseContiguous",
    "is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available",
    "launch_plan",
    "prepare_group_gemm_fp8_nt_groupwise_contiguous",
    "route_for_shape",
]
