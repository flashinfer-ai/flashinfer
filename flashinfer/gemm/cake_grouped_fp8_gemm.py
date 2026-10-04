# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared contiguous grouped FP8 GEMM launches on Blackwell (SM100a, SM103a).

This is the generated Cake program family for the same mathematical
contract as :func:`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous`:
``out[r, :] = sum_q (a[r, q-block] . b[g_r, :, q-block]) * a_scale[r, q] *
b_scale[g_r, :, q]`` with FP8 E4M3 operands, per-row 128-wide K-block A scales,
128x128 B block scales, ordered FP32 K-block accumulation and a BF16 output.
Rows are routed to experts by the sorted ``m_indices`` vector.  Unlike the
CuTe-DSL kernel, internal expert boundaries need not be aligned to 128 rows.

Two kernel families serve the contract.  FP32 scales (any values) run the
promotion family: plain E4M3 MMA with FP32 register scaling of every K block.
Packed UE8M0 int32 scales (DeepGEMM's SM100 layout: one int32 holds four
consecutive K-block exponents, activations ``(M, ceil(K/512))`` with any
strides, weights ``(G, N//128, cols)`` or the row-repeated ``(G, N, cols)``)
run the block-scaled family: ``tcgen05.mma.kind::mxf8f6f4`` applies both
scales in the tensor core, ``-1`` entries of ``m_indices`` mark padding rows
of the compact MoE layout (skipped, output rows untouched) and expert runs
start on multiples of ``alignment`` rows (128-multiples select the single-run
schedule, other multiples of 32 the slower multi-run schedule).

The host side selects one kernel route per problem shape and output
alignment, resolves the launch grid from the device SM count (queried once per
device), and binds the positional argument plan of the registered program.
Tensor maps are encoded by the binding and passed to the kernel by value, so a
prepared launch owns no descriptor storage: every ``launch()`` submits exactly
one kernel on the current stream and may be captured into a CUDA graph,
including the first one.  Preparation binds the problem shape and the expert
weights; the per-token operands (``a``, ``a_scale``, ``m_indices``, ``out``)
may be rebound at every ``launch`` so callers need no staging copies.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import tvm_ffi

from ..jit.gemm.cake_grouped_fp8_gemm import (
    ARG_PLANS,
    MODULES,
    PROGRAMS,
    ROUTE_GEOMETRY,
    SUPPORTED_COMPUTE_CAPABILITIES,
    device_arch,
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
# Block-scaled (packed UE8M0 scale) family.
BS_N128_ROUTE = "bs_ue8m0_n128"
BS_N256_ROUTE = "bs_ue8m0_n256"
BS_N128_RUN32_ROUTE = "bs_ue8m0_n128_run32"
BS_N256_RUN32_ROUTE = "bs_ue8m0_n256_run32"
BS_ROUTES = frozenset(
    {BS_N128_ROUTE, BS_N256_ROUTE, BS_N128_RUN32_ROUTE, BS_N256_RUN32_ROUTE}
)
BS_BLOCK_M = 128
BS_BLOCK_K = 128
BS_SUB_ROWS = 32

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
    """Select the kernel route for a 16-byte-aligned BF16 output."""
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


def bs_block_n(m: int, n: int, k: int, sm_count: int) -> int:
    """``BLOCK_N`` of the block-scaled family (mirror of the source dispatcher's measured rule).

    256-column tiles move the fewest bytes per output and win for short K loops or at
    least three waves of M blocks; long-K problems with fewer M blocks run 128.
    """
    if n % B_SCALE_BLOCK_N:
        raise ValueError(f"N must be a multiple of {B_SCALE_BLOCK_N}, got {n}")
    m_blocks = (m + BS_BLOCK_M - 1) // BS_BLOCK_M
    if n % 256 == 0 and not (k // K_BLOCK >= 16 and m_blocks < 3 * sm_count):
        return 256
    return 128


def bs_route_for_shape(m: int, n: int, k: int, *, alignment: int, sm_count: int) -> str:
    """Select the block-scaled route for packed UE8M0 scales."""
    if alignment <= 0 or alignment % BS_SUB_ROWS:
        raise ValueError(
            f"alignment must be a positive multiple of {BS_SUB_ROWS}, got {alignment}"
        )
    if k <= 0 or k % BS_BLOCK_K:
        raise ValueError(
            f"block-scaled grouped FP8 GEMM requires positive K divisible by {BS_BLOCK_K}; got K={k}"
        )
    block_n = bs_block_n(m, n, k, sm_count)
    single_run = alignment % BS_BLOCK_M == 0
    return {
        (128, True): BS_N128_ROUTE,
        (256, True): BS_N256_ROUTE,
        (128, False): BS_N128_RUN32_ROUTE,
        (256, False): BS_N256_RUN32_ROUTE,
    }[(block_n, single_run)]


def resolve_route_for_grid(route: str, n: int, grid: tuple[int, int, int]) -> str:
    """Pick the exported program of a route for the grid it will launch with.

    The four-load schedule exists only as its 128-CTA instantiation; a device
    with fewer than 128 SMs (grid below 128 CTAs) runs the recurrence schedule
    instead, which has the same tile geometry and an exported program at every
    grid.
    """
    if route == DEEPK_CG2_FOUR_LOAD_ROUTE and n == 4096:
        if grid == (128, 1, 1):
            return DEEPK_CG2_FOUR_LOAD_GRID128_ROUTE
        if grid[0] < 128:
            return DEEPK_CG2_RECURRENCE_ROUTE
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
    m: int,
    n: int,
    k: int,
    *,
    sm_count: int,
    scalar_output: bool,
    block_scaled: bool = False,
    alignment: int = 128,
) -> tuple[str, tuple[int, int, int]]:
    """Resolve ``(route, grid)`` for a problem shape on a device with ``sm_count`` SMs."""
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if block_scaled:
        if scalar_output:
            raise ValueError(
                "block-scaled grouped FP8 GEMM requires a 16-byte-aligned output"
            )
        route = bs_route_for_shape(m, n, k, alignment=alignment, sm_count=sm_count)
    elif scalar_output:
        route = DEEPK_C2_SCALAR_OUTPUT_ROUTE
    else:
        route = route_for_shape(m, n, k)
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


@functools.cache
def device_sm_count(device_index: int) -> int:
    """Streaming-multiprocessor count of CUDA device ``device_index`` (queried once)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


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


def _validate_indices(
    m_indices: torch.Tensor, groups: int, *, allow_padding: bool = False
) -> None:
    """Check the routing contract (in range, sorted) with one device-to-host transfer.

    With ``allow_padding`` the value ``-1`` marks padding rows; the remaining
    entries must still be nondecreasing.
    """
    indices = m_indices.to(torch.int64)
    routed = indices[indices >= 0] if allow_padding else indices
    unsorted = (
        (routed[1:] < routed[:-1]).sum()
        if routed.numel() > 1
        else torch.zeros((), dtype=torch.int64, device=indices.device)
    )
    lowest, highest, unsorted = torch.stack(
        (indices.min(), indices.max(), unsorted)
    ).tolist()
    if lowest < (-1 if allow_padding else 0) or highest >= groups:
        if allow_padding:
            raise ValueError("m_indices must satisfy -1 <= index < num_groups")
        raise ValueError(
            "m_indices must satisfy 0 <= index < num_groups; -1 padding needs fill_padding=True"
        )
    if unsorted:
        raise ValueError("m_indices must be sorted in nondecreasing order")


def _packed_scale_strides(
    scale: torch.Tensor, name: str, *, rows: int, k_blocks: int, device: torch.device
) -> tuple[torch.Tensor, int, int]:
    """``(ffi tensor, row stride, col stride)`` of a packed UE8M0 activation-scale tensor.

    Any strides are accepted (the kernel reads ``base[row * rs + col * cs]``); the FFI
    binding needs a contiguous tensor sharing the storage pointer, so the MN-major
    layout is passed as its contiguous transpose.
    """
    if not isinstance(scale, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if scale.device != device or scale.dtype != torch.int32:
        raise ValueError(f"{name} must be an int32 tensor on {device}")
    if (
        scale.ndim != 2
        or int(scale.shape[0]) != rows
        or 4 * int(scale.shape[1]) < k_blocks
    ):
        raise ValueError(
            f"{name} must be packed UE8M0 of shape ({rows}, >= {-(-k_blocks // 4)}), got {tuple(scale.shape)}"
        )
    if int(scale.data_ptr()) % 16:
        raise ValueError(f"{name} storage must be at least 16-byte aligned")
    rs, cs = (int(v) for v in scale.stride())
    if scale.is_contiguous():
        return scale, rs, cs
    if scale.mT.is_contiguous():
        return scale.mT, rs, cs
    raise ValueError(
        f"{name} must be contiguous or the transpose of a contiguous tensor"
    )


def _packed_weight_scale_strides(
    scale: torch.Tensor, *, groups: int, n: int, k: int, device: torch.device
) -> tuple[torch.Tensor, int, int, int]:
    """``(ffi tensor, group stride, 128-row-block stride, col stride)`` of packed UE8M0 weight scales."""
    if not isinstance(scale, torch.Tensor):
        raise ValueError("b_scale must be a torch.Tensor")
    if scale.device != device or scale.dtype != torch.int32:
        raise ValueError(f"b_scale must be an int32 tensor on {device}")
    n_blocks, k_blocks = n // B_SCALE_BLOCK_N, k // K_BLOCK
    if (
        scale.ndim != 3
        or int(scale.shape[0]) != groups
        or 4 * int(scale.shape[2]) < k_blocks
    ):
        raise ValueError(
            f"b_scale must be packed UE8M0 of shape ({groups}, {n_blocks} or {n}, >= {-(-k_blocks // 4)}), "
            f"got {tuple(scale.shape)}"
        )
    if int(scale.data_ptr()) % 16:
        raise ValueError("b_scale storage must be at least 16-byte aligned")
    rows = int(scale.shape[1])
    gs, rs, cs = (int(v) for v in scale.stride())
    if rows == n:
        rs *= B_SCALE_BLOCK_N
    elif rows != n_blocks:
        raise ValueError(
            f"b_scale must have {n_blocks} or {n} rows per group, got {rows}"
        )
    if scale.is_contiguous():
        return scale, gs, rs, cs
    if scale.mT.is_contiguous():
        return scale.mT, gs, rs, cs
    raise ValueError(
        "b_scale must be contiguous or the transpose of a contiguous tensor"
    )


def argument_slots(arg_plan: Sequence[Sequence[str]]) -> dict[str, tuple[int, ...]]:
    """Positions of every named binding in a generated program's argument plan."""
    slots: dict[str, list[int]] = {}
    for index, (kind, name) in enumerate(arg_plan):
        if kind != "grid":
            slots.setdefault(name, []).append(index)
    return {name: tuple(positions) for name, positions in slots.items()}


def _forward_fill_padding(
    m_indices: torch.Tensor, filled: torch.Tensor, positions: torch.Tensor
) -> None:
    """``filled[r] = max(0, max(m_indices[:r + 1]))``: padding rows follow the preceding expert.

    Two small allocation-free kernels (``cummax`` into prepared scratch, then
    ``clamp``); both are graph-capturable.  Leading padding maps onto expert 0.
    """
    torch.cummax(m_indices, 0, out=(filled, positions))
    filled.clamp_(min=0)


@dataclass
class PreparedGroupGemmFp8NtGroupwiseContiguous:
    """One prepared contiguous grouped FP8 GEMM launch.

    Shapes, dtypes and the expert weights (``b``, ``b_scale``) are bound at
    preparation; tensor *contents* may change between launches, and the
    per-token operands ``a``, ``a_scale``, ``m_indices`` and ``out`` may be
    replaced by same-shape tensors at every ``launch`` (``launch(a=..., ...)``),
    so a caller can run one prepared object per layer on whatever buffers the
    dispatcher produced.  ``launch()`` submits exactly one kernel (plus two
    tiny index kernels when ``fill_padding`` is on) on PyTorch's current stream
    for the bound device and returns the output tensor.  Every launch,
    including the first, may be captured into a CUDA graph.  The prepared
    object keeps the operands given at preparation alive (so a bare
    ``launch()`` works); a long-lived plan that always launches on the caller's
    buffers calls ``release_prepared_operands()`` once, after which it holds no
    per-token device memory and every ``launch`` must supply ``a``,
    ``a_scale``, ``m_indices`` and ``out``.
    """

    route: str
    module_name: str
    grid: tuple[int, int, int]
    device: torch.device
    _out: Optional[torch.Tensor]
    m: int
    n: int
    k: int
    groups: int
    fill_padding: bool
    _entry: Callable[..., Any]
    _arguments: tuple[Any, ...]
    _slots: dict[str, tuple[int, ...]]
    _padding_source: Optional[torch.Tensor]
    _padding_scratch: Optional[tuple[torch.Tensor, torch.Tensor]]
    # Block-scaled family: the prepared activation-scale layout (shape, strides); a
    # replacement a_scale must match it because the strides are bound as parameters.
    _a_scale_layout: Optional[tuple[tuple[int, ...], tuple[int, ...]]] = None
    _released: bool = False

    @property
    def block_scaled(self) -> bool:
        return self.route in BS_ROUTES

    @property
    def out(self) -> torch.Tensor:
        """The output tensor bound at preparation (unavailable after ``release_prepared_operands``)."""
        if self._out is None:
            raise ValueError(
                "the prepared operands were released; launch(a=..., a_scale=..., "
                "m_indices=..., out=...) returns the caller's output"
            )
        return self._out

    def release_prepared_operands(self) -> None:
        """Drop the references to the per-token operands bound at preparation.

        ``a``, ``a_scale``, ``out`` and (unless ``fill_padding`` keeps the
        prepared ``m_indices`` as its forward-fill source) ``m_indices`` are
        released so that a plan kept per layer does not pin a dispatcher's
        buffers.  Afterwards every ``launch`` must supply all of them.
        """
        arguments = list(self._arguments)
        names = ("A", "A64", "A32", "C", "C_tma", "a_scale", "SFA")
        if not self.fill_padding:
            names += ("m_indices",)
        for name in names:
            self._rebind(arguments, name, None)
        self._arguments = tuple(arguments)
        self._out = None
        self._released = True

    def _rebind(self, arguments: list[Any], name: str, value: Any) -> None:
        for position in self._slots.get(name, ()):
            arguments[position] = value

    def launch(
        self,
        a: Optional[torch.Tensor] = None,
        a_scale: Optional[torch.Tensor] = None,
        m_indices: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the GEMM; any operand given here replaces the prepared one for this call.

        Replacement tensors must match the prepared shape, dtype, device and
        alignment class (an output that is 16-byte aligned cannot replace a
        2-byte-aligned one and vice versa, because the store route is fixed at
        preparation).  Rebinding allocates nothing on the device.
        """
        device = self.device
        arguments = self._arguments
        scalar = self.route == DEEPK_C2_SCALAR_OUTPUT_ROUTE
        if self._released and (
            a is None
            or a_scale is None
            or out is None
            or (m_indices is None and not self.fill_padding)
        ):
            raise ValueError(
                "the prepared operands were released; launch must receive a, "
                "a_scale, m_indices and out"
            )
        if (
            a is not None
            or a_scale is not None
            or m_indices is not None
            or out is not None
        ):
            rebound = list(arguments)
            if a is not None:
                a = _require_tensor(
                    a,
                    "a",
                    shape=(self.m, self.k),
                    dtype=torch.float8_e4m3fn,
                    device=device,
                )
                a_u8 = a.view(torch.uint8)
                for name in ("A", "A64", "A32"):
                    self._rebind(rebound, name, a_u8)
                if scalar:
                    self._rebind(rebound, "C_tma", a.view(torch.bfloat16))
            if a_scale is not None:
                if self.block_scaled:
                    ffi_scale, rs, cs = _packed_scale_strides(
                        a_scale,
                        "a_scale",
                        rows=self.m,
                        k_blocks=self.k // K_BLOCK,
                        device=device,
                    )
                    layout = (tuple(int(v) for v in a_scale.shape), (rs, cs))
                    if layout != self._a_scale_layout:
                        raise ValueError(
                            f"a_scale must keep the prepared packed layout {self._a_scale_layout}, got {layout}"
                        )
                    self._rebind(rebound, "SFA", ffi_scale)
                else:
                    a_scale = _require_tensor(
                        a_scale,
                        "a_scale",
                        shape=(self.m, self.k // K_BLOCK),
                        dtype=torch.float32,
                        device=device,
                    )
                    self._rebind(rebound, "a_scale", a_scale)
            if m_indices is not None:
                m_indices = _require_tensor(
                    m_indices,
                    "m_indices",
                    shape=(self.m,),
                    dtype=torch.int32,
                    device=device,
                )
                if not self.fill_padding:
                    self._rebind(rebound, "m_indices", m_indices)
            if out is not None:
                out = _require_tensor(
                    out,
                    "out",
                    shape=(self.m, self.n),
                    dtype=torch.bfloat16,
                    device=device,
                    alignment=2,
                )
                if bool(int(out.data_ptr()) % 16) != scalar:
                    raise ValueError(
                        "out must keep the alignment class of the prepared output "
                        "(16-byte aligned or not); the store route is fixed at preparation"
                    )
                self._rebind(rebound, "C", out)
                if not scalar:
                    self._rebind(rebound, "C_tma", out)
                if self.block_scaled and int(out.data_ptr()) % 16:
                    raise ValueError(
                        "block-scaled grouped FP8 GEMM requires a 16-byte-aligned output"
                    )
            arguments = tuple(rebound)
        with torch.cuda.device(device):
            if self.fill_padding:
                assert self._padding_scratch is not None
                source = m_indices if m_indices is not None else self._padding_source
                assert source is not None
                _forward_fill_padding(source, *self._padding_scratch)
            with tvm_ffi.use_torch_stream():
                self._entry(*arguments)
        return self._out if out is None else out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.grid[0])


def is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available(
    device: torch.device,
) -> bool:
    """True when a generated program family is registered for ``device``."""
    return device.type == "cuda" and generated_program_available(device)


def bind_arguments(
    arg_plan: list[list[str]],
    bindings: dict[str, Any],
    grid: tuple[int, int, int],
    *,
    module_name: str,
) -> tuple[Any, ...]:
    """Order ``bindings`` by the generated program's positional argument plan."""
    grid_by_axis = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in arg_plan:
        if kind == "grid":
            arguments.append(grid_by_axis[name])
        elif kind in ("tma_buffer", "buffer", "parameter") and name in bindings:
            arguments.append(bindings[name])
        else:
            raise RuntimeError(
                f"generated program {module_name} binds {kind} {name!r}, which this host plan does not declare"
            )
    return tuple(arguments)


def prepare_group_gemm_fp8_nt_groupwise_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
    fill_padding: bool = False,
    alignment: int = 128,
) -> PreparedGroupGemmFp8NtGroupwiseContiguous:
    r"""Prepare a contiguous grouped FP8 GEMM launch on SM100a / SM103a.

    Parameters
    ----------
    a : torch.Tensor
        Contiguous FP8 E4M3 input of shape ``(M, K)``; ``M > 0`` and ``K`` a
        positive multiple of 128.
    b : torch.Tensor
        Contiguous FP8 E4M3 expert weights of shape ``(G, N, K)``; ``N`` a
        positive multiple of 128.
    a_scale : torch.Tensor
        Contiguous float32 scales of shape ``(M, K // 128)`` (promotion family),
        or packed UE8M0 int32 words of shape ``(M, >= ceil(K / 512))`` with any
        strides (block-scaled family; DeepGEMM's MN-major layout is read in
        place).
    b_scale : torch.Tensor
        Contiguous float32 scales of shape ``(G, N // 128, K // 128)``, or
        packed UE8M0 int32 words of shape ``(G, N // 128, cols)`` or the
        row-repeated ``(G, N, cols)`` (``4 * cols >= K // 128``) when
        ``a_scale`` is packed.
    m_indices : torch.Tensor
        Contiguous int32 expert indices of shape ``(M,)``, sorted in
        nondecreasing order with ``0 <= index < G``.  Expert boundaries may
        fall at any row; empty experts are allowed.  With ``fill_padding`` the
        value ``-1`` marks padding rows (the compact MoE layout): such a row is
        computed with the preceding expert's weights (expert 0 for leading
        padding) and its output row is unspecified.  The block-scaled family
        reads ``-1`` natively per 32-row sub-block: a sub-block whose leading
        entry is ``-1`` is skipped and its output rows are left untouched;
        ``-1`` rows sharing a sub-block with routed rows receive finite values
        of no meaning (as DeepGEMM's 128-row blocks do).  Every expert run must
        start on a multiple of ``alignment`` rows (an expert's rows followed by
        its padding).
    out : Optional[torch.Tensor]
        Contiguous bfloat16 output of shape ``(M, N)``.  Allocated if omitted.
        A 2-byte-aligned output whose address is not a multiple of 16 selects
        the scalar-store route.
    validate_indices : bool
        Check index range and sortedness (one device-to-host transfer).
    fill_padding : bool
        Accept ``-1`` padding rows in ``m_indices`` (promotion family).  The
        prepared object owns an ``(M,)`` int32 plus ``(M,)`` int64 scratch and
        every launch runs two small forward-fill kernels before the GEMM
        (graph-capturable).  Must be ``False`` with packed scales.
    alignment : int
        Block-scaled family only: the row alignment of expert runs in the
        compact layout, a positive multiple of 32.  Multiples of 128 run the
        single-run schedule; other values the slower 32-row multi-run schedule.

    Returns
    -------
    PreparedGroupGemmFp8NtGroupwiseContiguous
        Call ``.launch()`` to run the GEMM on the current stream.

    Notes
    -----
    Requires an SM100a (compute capability 10.0) or SM103a (10.3) device and a
    registered generated program for the selected route.  All tensors must
    live on the same CUDA device and, except ``out``, be 16-byte aligned.
    Preparation performs no device work: the SM count of the device is read
    once per process, and the first launch is as graph-capturable as every
    later one.  Index values are unchecked unless ``validate_indices=True``;
    violating the routing contract is undefined behavior.  The per-token
    operands may be swapped per call: ``prepared.launch(a=a2, a_scale=s2,
    m_indices=i2, out=o2)`` (same shapes, dtypes, device and alignment class).
    """
    if not isinstance(a, torch.Tensor) or a.device.type != "cuda":
        raise ValueError("a must be a CUDA torch.Tensor")
    device = a.device
    device_index = _device_index(device)
    arch = device_arch(device_index)
    if arch is None:
        raise NotImplementedError(
            "prepared contiguous grouped FP8 GEMM requires an SM100a or SM103a device "
            f"(compute capability {sorted(SUPPORTED_COMPUTE_CAPABILITIES)}); "
            f"got {torch.cuda.get_device_capability(device_index)}"
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
    block_scaled = isinstance(a_scale, torch.Tensor) and a_scale.dtype == torch.int32
    if block_scaled:
        if fill_padding:
            raise ValueError(
                "fill_padding is not needed with packed UE8M0 scales: -1 padding rows are skipped natively"
            )
        if (
            isinstance(alignment, bool)
            or not isinstance(alignment, int)
            or alignment <= 0
            or alignment % BS_SUB_ROWS
        ):
            raise ValueError(
                f"alignment must be a positive multiple of {BS_SUB_ROWS}, got {alignment!r}"
            )
        sfa, sfa_rs, sfa_cs = _packed_scale_strides(
            a_scale, "a_scale", rows=m, k_blocks=k // K_BLOCK, device=device
        )
        sfb, sfb_gs, sfb_rs, sfb_cs = _packed_weight_scale_strides(
            b_scale, groups=groups, n=n, k=k, device=device
        )
    else:
        a_scale = _require_tensor(
            a_scale,
            "a_scale",
            shape=(m, k // K_BLOCK),
            dtype=torch.float32,
            device=device,
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
        _validate_indices(m_indices, groups, allow_padding=fill_padding or block_scaled)
    padding_scratch = None
    kernel_indices = m_indices
    if fill_padding:
        kernel_indices = torch.empty((m,), dtype=torch.int32, device=device)
        padding_scratch = (
            kernel_indices,
            torch.empty((m,), dtype=torch.int64, device=device),
        )

    scalar_output = bool(int(out.data_ptr()) % 16)
    if block_scaled and scalar_output:
        raise ValueError(
            "block-scaled grouped FP8 GEMM requires a 16-byte-aligned output"
        )
    route, grid = launch_plan(
        m,
        n,
        k,
        sm_count=device_sm_count(device_index),
        scalar_output=scalar_output,
        block_scaled=block_scaled,
        alignment=alignment,
    )
    module_name = select_module(arch, route)
    program = PROGRAMS[MODULES[module_name]["program"]]
    module = load_cake_grouped_fp8_gemm_module(module_name)
    entry = getattr(module, program["ffi_entry"])

    # The scalar-store route never reads its output tensor map; the validated
    # A tensor provides an aligned BF16 view with a compatible descriptor.
    c_tma = a.view(torch.bfloat16) if route == DEEPK_C2_SCALAR_OUTPUT_ROUTE else out
    a_scale_layout = None
    if block_scaled:
        tile_n = route_geometry(route)[1]
        a_u8 = a.view(torch.uint8)
        bindings: dict[str, Any] = {
            "A": a_u8,
            "A64": a_u8,
            "A32": a_u8,
            "B": b.view(torch.uint8),
            "SFA": sfa,
            "SFB": sfb,
            "m_indices": m_indices,
            "C_tma": out,
            "shape_m": m,
            "shape_n": n,
            "grid_n": math.ceil(n / tile_n),
            "k_tiles": k // BS_BLOCK_K,
            "sfa_row_stride": sfa_rs,
            "sfa_col_stride": sfa_cs,
            "sfb_group_stride": sfb_gs,
            "sfb_row_stride": sfb_rs,
            "sfb_col_stride": sfb_cs,
        }
        a_scale_layout = (tuple(int(v) for v in a_scale.shape), (sfa_rs, sfa_cs))
    else:
        bindings = {
            "A": a.view(torch.uint8),
            "B": b.view(torch.uint8),
            "C_tma": c_tma,
            "C": out,
            "a_scale": a_scale,
            "b_scale": b_scale,
            "m_indices": kernel_indices,
            "M": m,
            "N": n,
            "K": k,
            "G": groups,
        }
    arg_plan = ARG_PLANS[program["arg_plan"]]
    return PreparedGroupGemmFp8NtGroupwiseContiguous(
        route=route,
        module_name=module_name,
        grid=grid,
        device=out.device,
        _out=out,
        m=m,
        n=n,
        k=k,
        groups=groups,
        fill_padding=fill_padding,
        _entry=entry,
        _arguments=bind_arguments(arg_plan, bindings, grid, module_name=module_name),
        _slots=argument_slots(arg_plan),
        _padding_source=m_indices if fill_padding else None,
        _padding_scratch=padding_scratch,
        _a_scale_layout=a_scale_layout,
    )


__all__ = [
    "BS_ROUTES",
    "PreparedGroupGemmFp8NtGroupwiseContiguous",
    "argument_slots",
    "bind_arguments",
    "bs_block_n",
    "bs_route_for_shape",
    "device_sm_count",
    "is_group_gemm_fp8_nt_groupwise_contiguous_prepared_available",
    "launch_plan",
    "prepare_group_gemm_fp8_nt_groupwise_contiguous",
    "route_for_shape",
]
