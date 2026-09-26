# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared fused grouped FP8 gate_up GEMM + SwiGLU + per-token-group FP8 quantization on Blackwell SM100a.

This is the generated Cake program for the MoE gate_up chain that FlashInfer
otherwise runs as three kernels: :func:`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous`
(BF16 ``[M, 2H]``), :func:`flashinfer.activation.silu_and_mul` and
:func:`flashinfer.quantization.per_token_group_quant_8bit` (group size 128,
FP8 E4M3).  The fused kernel keeps the GEMM's ordered FP32 K-block accumulation
and reproduces the chain's rounding points exactly: ``g = BF16(gate)``,
``u = BF16(up)``, ``h = BF16(silu(g) * u)``, ``scale = max(absmax_128(h), eps) / 448``,
``q = E4M3(clamp(h / scale, -448, 448))``.  The BF16 intermediate, the
activation launch and the quantization launch disappear.

Routing contract (the same as the CuTe-DSL contiguous grouped GEMM): rows are
sorted by expert through ``m_indices``; every *internal* expert boundary is a
multiple of 128 rows (``moe_align_block_size`` with block 128), only the final
expert may end in a partial block, empty experts are allowed, and
``M <= 8192``.  Each kernel tile is a pair of consecutive same-expert 128-row
blocks; an expert with an odd block count computes one dummy block.

Descriptor storage for the pointer TMA ABI is private to each prepared launch:
the first ``launch()`` initializes it synchronously and must run outside CUDA
Graph capture; later launches only submit work on the current stream and may
be captured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from ..jit.gemm.cake_grouped_fp8_fused_silu_quant import (
    MODULES,
    ROUTE_GEOMETRY,
    SUPPORTED_COMPUTE_CAPABILITIES,
    generated_program_available,
    load_cake_grouped_fp8_fused_silu_quant_module,
    select_module,
)

GROUP_SIZE = 128
K_BLOCK = 128
K_MULTIPLE = 512
N2_MULTIPLE = 256
ROW_BLOCK = 128
MAX_M = 8192
CTA_CAP = 128
QUANT_EPS = 1e-10
FP8_E4M3_MAX = 448.0

# Route name doubles as the generated-program template name.
FUSED_ROUTE = "fused_cg2_ab7_pairsched_kg4"


def route_geometry(route: str) -> tuple[int, int, int]:
    """``(tile_m, tile_n, cluster_ctas)`` of one route from the export registry.

    ``tile_n`` counts B rows (gate + up), i.e. twice the SwiGLU columns.
    """
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


def launch_plan(m: int, n2: int, *, sm_count: int) -> tuple[str, tuple[int, int, int]]:
    """Resolve ``(route, grid)`` for ``A[M, K]`` and ``B[G, 2H, K]`` on ``sm_count`` SMs.

    Complete CTA pairs; the tile upper bound counts every 128-row block as a
    lone block.  The 128-CTA cap is the measured B200 policy of the parent
    gate_up route (148 CTAs measured slower on every routing).
    """
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    route = FUSED_ROUTE
    _, tile_n, cluster_ctas = route_geometry(route)
    max_tiles = math.ceil(m / ROW_BLOCK) * (n2 // tile_n)
    clusters = min(max_tiles, min(sm_count, CTA_CAP) // cluster_ctas)
    if clusters <= 0:
        raise ValueError("device cannot schedule one complete CTA pair")
    return route, (clusters * cluster_ctas, 1, 1)


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
    """Synchronizing check of the routing contract (sorted, in range, 128-aligned boundaries)."""
    indices = m_indices.to(torch.int64)
    lowest = int(indices.min())
    highest = int(indices.max())
    if lowest < 0 or highest >= groups:
        raise ValueError(
            "m_indices must satisfy 0 <= index < num_groups; -1 padding is unsupported"
        )
    if indices.numel() > 1:
        step = indices[1:] - indices[:-1]
        if bool((step < 0).any()):
            raise ValueError("m_indices must be sorted in nondecreasing order")
        boundaries = torch.nonzero(step > 0).flatten() + 1
        if bool((boundaries % ROW_BLOCK != 0).any()):
            raise ValueError(
                "m_indices must place every internal expert boundary at a multiple "
                "of 128 rows (only the final expert may end in a partial block)"
            )


@dataclass
class PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant:
    """One prepared fused gate_up GEMM + SwiGLU + FP8 quantization launch.

    Tensor storage, shapes and dtypes are bound at preparation; tensor
    *contents* may change between launches.  ``launch()`` submits exactly one
    kernel on PyTorch's current stream for the bound device and returns
    ``(out_q, out_s)``.  The first launch initializes private TMA descriptor
    storage and must run outside CUDA Graph capture.
    """

    route: str
    module_name: str
    grid: tuple[int, int, int]
    out_q: torch.Tensor
    out_s: torch.Tensor
    _entry: Callable[..., Any]
    _arguments: tuple[Any, ...]
    _descriptor_storage: torch.Tensor

    def launch(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.device(self.out_q.device), tvm_ffi.use_torch_stream():
            self._entry(*self._arguments)
        return self.out_q, self.out_s

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.grid[0])


def is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available(
    device: torch.device,
) -> bool:
    """True when a generated fused program is registered for ``device``."""
    return device.type == "cuda" and generated_program_available(device)


def prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out_q: Optional[torch.Tensor] = None,
    out_s: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
) -> PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant:
    r"""Prepare a fused contiguous grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization launch on SM100a.

    Parameters
    ----------
    a : torch.Tensor
        Contiguous FP8 E4M3 input of shape ``(M, K)``; ``0 < M <= 8192`` and
        ``K`` a positive multiple of 512.
    b : torch.Tensor
        Contiguous FP8 E4M3 expert weights of shape ``(G, 2H, K)``: gate rows
        ``[0, H)`` followed by up rows ``[H, 2H)``; ``2H`` a positive multiple
        of 256.
    a_scale : torch.Tensor
        Contiguous float32 scales of shape ``(M, K // 128)``.
    b_scale : torch.Tensor
        Contiguous float32 scales of shape ``(G, 2H // 128, K // 128)``.
    m_indices : torch.Tensor
        Contiguous int32 expert indices of shape ``(M,)``, sorted in
        nondecreasing order with ``0 <= index < G``; every internal expert
        boundary is a multiple of 128 rows (the final expert may be partial);
        empty experts are allowed.
    out_q : Optional[torch.Tensor]
        Contiguous FP8 E4M3 output ``(M, H)`` = ``silu(gate) * up`` quantized
        per token in groups of 128 columns.  Allocated if omitted.
    out_s : Optional[torch.Tensor]
        Contiguous float32 scales ``(M, H // 128)`` (row-major, one per
        128-column group, ``max(absmax, 1e-10) / 448``).  Allocated if omitted.
    validate_indices : bool
        Check the routing contract with a device synchronization.

    Returns
    -------
    PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant
        Call ``.launch()`` to run the fused kernel on the current stream.

    Notes
    -----
    Requires an SM100a (compute capability 10.0) device and a registered
    generated program.  All tensors must live on the same CUDA device and be
    16-byte aligned (``out_s`` 4-byte).  The outputs match the three-kernel
    FlashInfer chain (CuTe-DSL grouped GEMM, ``silu_and_mul``,
    ``per_token_group_quant_8bit``) element for element on the same inputs.
    Index values are unchecked unless ``validate_indices=True``; violating the
    routing contract is undefined behavior.
    """
    if not isinstance(a, torch.Tensor) or a.device.type != "cuda":
        raise ValueError("a must be a CUDA torch.Tensor")
    device = a.device
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise NotImplementedError(
            "prepared fused grouped FP8 gate_up+SwiGLU+quant requires an SM100a device "
            f"(compute capability 10.0); got {capability}"
        )
    if a.ndim != 2 or b.ndim != 3:
        raise ValueError("a must have shape (M, K) and b must have shape (G, 2H, K)")
    m, k = (int(v) for v in a.shape)
    groups, n2 = (int(v) for v in b.shape[:2])
    if m <= 0 or groups <= 0:
        raise ValueError("fused grouped FP8 gate_up requires positive M and G")
    if m > MAX_M:
        raise ValueError(f"M must be at most {MAX_M} rows (64 blocks of 128), got {m}")
    if n2 <= 0 or n2 % N2_MULTIPLE:
        raise ValueError(f"2H must be a positive multiple of {N2_MULTIPLE}, got {n2}")
    if k <= 0 or k % K_MULTIPLE:
        raise ValueError(f"K must be a positive multiple of {K_MULTIPLE}, got {k}")
    h = n2 // 2
    a = _require_tensor(a, "a", shape=(m, k), dtype=torch.float8_e4m3fn, device=device)
    b = _require_tensor(
        b, "b", shape=(groups, n2, k), dtype=torch.float8_e4m3fn, device=device
    )
    a_scale = _require_tensor(
        a_scale, "a_scale", shape=(m, k // K_BLOCK), dtype=torch.float32, device=device
    )
    b_scale = _require_tensor(
        b_scale,
        "b_scale",
        shape=(groups, n2 // GROUP_SIZE, k // K_BLOCK),
        dtype=torch.float32,
        device=device,
    )
    m_indices = _require_tensor(
        m_indices, "m_indices", shape=(m,), dtype=torch.int32, device=device
    )
    if out_q is None:
        out_q = torch.empty((m, h), dtype=torch.float8_e4m3fn, device=device)
    out_q = _require_tensor(
        out_q, "out_q", shape=(m, h), dtype=torch.float8_e4m3fn, device=device
    )
    if out_s is None:
        out_s = torch.empty((m, h // GROUP_SIZE), dtype=torch.float32, device=device)
    out_s = _require_tensor(
        out_s,
        "out_s",
        shape=(m, h // GROUP_SIZE),
        dtype=torch.float32,
        device=device,
        alignment=4,
    )
    if validate_indices:
        _validate_indices(m_indices, groups)

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    route, grid = launch_plan(m, n2, sm_count=sm_count)
    module_name = select_module(arch, route)
    record = MODULES[module_name]
    module = load_cake_grouped_fp8_fused_silu_quant_module(module_name)
    entry = getattr(module, record["ffi_entry"])

    bindings: dict[str, Any] = {
        "A": a.view(torch.uint8),
        "B": b.view(torch.uint8),
        "out_q": out_q,
        "out_s": out_s,
        "a_scale": a_scale,
        "b_scale": b_scale,
        "m_indices": m_indices,
        "M": m,
        "N": n2,
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
    return PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant(
        route=route,
        module_name=module_name,
        grid=grid,
        out_q=out_q,
        out_s=out_s,
        _entry=entry,
        _arguments=tuple(arguments),
        _descriptor_storage=descriptor_storage,
    )


__all__ = [
    "FUSED_ROUTE",
    "PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant",
    "is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available",
    "launch_plan",
    "prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant",
]
