# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared fused grouped FP8 gate_up GEMM + SwiGLU + per-token-group FP8 quantization on Blackwell SM100a.

These are the generated Cake programs for the MoE gate_up chain that FlashInfer
otherwise runs as three kernels: :func:`flashinfer.gemm.group_gemm_fp8_nt_groupwise_contiguous`
(BF16 ``[M, 2H]``), :func:`flashinfer.activation.silu_and_mul` and
:func:`flashinfer.quantization.per_token_group_quant_8bit` (group size 128,
FP8 E4M3).  Both routes reproduce the chain's rounding points exactly:
``g = BF16(gate)``, ``u = BF16(up)``, ``h = BF16(silu(g) * u)``,
``scale = max(absmax_128(h), eps) / 448``, ``q = E4M3(clamp(h / scale, -448, 448))``.

* ``M >= SMALL_M_MAX`` (2048 rows): two persistent fused kernels keep the
  GEMM's ordered FP32 K-block accumulation in tensor memory and apply
  SwiGLU + quantization in their epilogue.  The BF16 intermediate, the
  activation launch and the quantization launch disappear.
* ``M < SMALL_M_MAX``: the fused kernel's one-SM-per-tile epilogue does not
  amortize on a few tiles, so the prepared launch runs the FlashInfer grouped
  GEMM into a private BF16 workspace followed by one generated elementwise
  kernel that fuses SwiGLU with the group quantization (two launches instead
  of three).  The GEMM kernel is chosen by :func:`small_m_gemm_backend`.
* Routing-aware rule (:func:`select_route`): when the routing is known —
  ``validate_indices=True`` derives the 128-row block count of every expert
  from ``m_indices`` inside its synchronizing check — a large problem whose
  odd-tail blocks (experts with an odd block count) x N256 tiles outnumber the
  SMs the 128-CTA pair grid leaves free also takes the GEMM + act route: an
  odd tail runs as a single-CTA tile that ingests a whole B slab and costs
  ~1.5x a pair tile, so beyond the free SMs the CuTe GEMM (insensitive to odd
  tails) + act is faster.  Without the routing the plan is shape-only.  A
  diverted problem with at least ``ACT_WIDE_MIN_ITEMS`` (row, 128-column group)
  items runs the wide act kernel (eight lanes per group, four groups per warp,
  64 B per lane in flight: 1.09-1.43x the one-warp-per-group kernel from
  M = 2048, H = 1024 on, bitwise-identical outputs); smaller problems keep the
  one-warp-per-group kernel, which is faster on latency-bound problems.

Routing contract (the same as the CuTe-DSL contiguous grouped GEMM): rows are
sorted by expert through ``m_indices``; every *internal* expert boundary is a
multiple of 128 rows (``moe_align_block_size`` with block 128), only the final
expert may end in a partial block, empty experts are allowed, and
``M <= 8192``.  The pair kernel processes consecutive same-expert 128-row
blocks as 256-row cluster tiles (two CTAs, ``cta_group::2``); the tail kernel
processes the odd tail block of every odd-count expert as single-CTA 128-row
tiles (``cta_group::1``), so no dummy block is computed on non-uniform
routings.  The tail kernel is launched with the programmatic-dependent-launch
attribute right after the pair kernel (which signals its dependents at start),
so its CTAs fill the SMs the pair grid leaves free or retires from; it reads
only the operator's inputs.

Descriptor storage for the pointer TMA ABI (fused route, and the Cake GEMM of
the small-M route) is private to each prepared launch: the first ``launch()``
initializes it synchronously and must run outside CUDA Graph capture; later
launches only submit work on the current stream and may be captured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import tvm_ffi

from ..jit.gemm.cake_grouped_fp8_fused_silu_quant import (
    MODULES,
    ROUTE_GEOMETRY,
    SUPPORTED_COMPUTE_CAPABILITIES,
    generated_program_available,
    load_cake_grouped_fp8_fused_silu_quant_module,
    select_stage_module,
)
from .cake_grouped_fp8_gemm import prepare_group_gemm_fp8_nt_groupwise_contiguous

GROUP_SIZE = 128
K_BLOCK = 128
K_MULTIPLE = 512
N2_MULTIPLE = 256
ROW_BLOCK = 128
MAX_M = 8192
CTA_CAP = 128
QUANT_EPS = 1e-10
FP8_E4M3_MAX = 448.0

# Route names double as the generated-program template names of their first kernel.
FUSED_ROUTE = "fused_cg2_ab7_pairsched_solotail_kg4"
ACT_ROUTE = "gemm_then_silu_mul_group_quant_fp8"
ACT_WIDE_ROUTE = "gemm_then_silu_mul_group_quant_fp8_wide"
ACT_ROUTES = frozenset({ACT_ROUTE, ACT_WIDE_ROUTE})
# Generated kernel stages per route in launch order; the fused route's tail kernel has its own template geometry.
FUSED_PAIR_STAGE = "pair"
FUSED_TAIL_STAGE = "tail"
ACT_STAGE = "main"
ROUTE_STAGES = {
    FUSED_ROUTE: (FUSED_PAIR_STAGE, FUSED_TAIL_STAGE),
    ACT_ROUTE: (ACT_STAGE,),
    ACT_WIDE_ROUTE: (ACT_STAGE,),
}
TAIL_GEOMETRY_KEY = f"{FUSED_ROUTE}_{FUSED_TAIL_STAGE}"
# Rows below this take ACT_ROUTE (measured B200 crossover of the fused kernel against grouped GEMM + the fused
# activation kernel; every problem with at most 9 row blocks lost to the chain in the fused kernel).
SMALL_M_MAX = 2048
# Elementwise kernel launch: one warp per (row, 128-column group), four warps per CTA, at most sixteen resident
# CTAs per SM (32 registers x 128 threads, full occupancy); grid-stride beyond that.
ACT_WARPS_PER_CTA = 4
ACT_CTAS_PER_SM = 16
# Wide act kernel launch: four (row, 128-column group) items per warp (eight lanes each), four warps per CTA, at
# most ten resident CTAs per SM (48 registers x 128 threads); grid-stride beyond that.
ACT_WIDE_GROUPS_PER_WARP = 4
ACT_WIDE_CTAS_PER_SM = 10
# A diverted large problem takes the wide act kernel from this many items on (measured B200 crossover envelope:
# 1.085x at 16384 items = M 2048 x H 1024, 1.32-1.43x from 32768; 0.92-0.97x below ~2400 items).
ACT_WIDE_MIN_ITEMS = 16384
# GEMM kernels of the small-M route.
GEMM_BACKEND_CUTE = "cute_dsl"  # group_gemm_fp8_nt_groupwise_contiguous (CuTe-DSL)
GEMM_BACKEND_CAKE = (
    "cake_prepared"  # prepare_group_gemm_fp8_nt_groupwise_contiguous (Cake)
)


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


# A partial final block hands the small-M GEMM to the Cake kernel from this K on (see small_m_gemm_backend).
PARTIAL_TAIL_CAKE_MIN_K = 1024


def small_m_gemm_backend(m: int, n2: int, k: int) -> str:
    """GEMM kernel of the small-M route (measured B200 policy).

    The CuTe-DSL grouped GEMM specializes on ``M % 128`` (tail predication);
    on 128-aligned M it is the faster kernel for every small-M problem
    measured.  Its predicated-tail build pays a penalty that grows with K
    while the Cake prepared GEMM is insensitive to the tail, so a partial
    final block hands the GEMM to the Cake kernel from ``K = 1024`` on.
    """
    if m <= 0 or n2 <= 0 or k <= 0:
        raise ValueError("small_m_gemm_backend requires positive M, 2H and K")
    if m % ROW_BLOCK and k >= PARTIAL_TAIL_CAKE_MIN_K:
        return GEMM_BACKEND_CAKE
    return GEMM_BACKEND_CUTE


def routing_blocks(group_counts: Sequence[int]) -> tuple[int, ...]:
    """128-row block count of every expert (``ceil(rows / 128)``)."""
    counts = [int(c) for c in group_counts]
    if any(c < 0 for c in counts):
        raise ValueError("group_counts must be non-negative")
    return tuple(-(-c // ROW_BLOCK) for c in counts)


def fused_tile_counts(n2: int, group_blocks: Sequence[int]) -> tuple[int, int]:
    """``(pair_tiles, odd_tail_units)`` the fused route schedules for a routing.

    Complete 128-row block pairs x N256 tiles run on the pair kernel, each
    expert's odd tail block x N256 tiles on the tail kernel.
    """
    _, tile_n, _ = route_geometry(FUSED_ROUTE)
    n_tiles = n2 // tile_n
    blocks = [int(b) for b in group_blocks]
    return sum(b // 2 for b in blocks) * n_tiles, sum(b % 2 for b in blocks) * n_tiles


# Routing-aware route rule (measured on B200 at the wide_ep32 shape).  An odd
# tail block runs on the tail kernel as a single-CTA unit that ingests the whole
# B slab (1.5 MiB per 128 rows against a pair CTA's 1.0 MiB) and takes ~1.5x a
# pair tile; the units that start on the SMs the 128-CTA pair grid leaves free
# finish inside the pair phase, every further unit waits for a retiring cluster
# and adds its full latency.  With at most that many odd-tail units the fused
# route wins, beyond it the CuTe GEMM + act route is faster on every routing
# measured.  ``False`` keeps the shape-only rule.
ROUTING_AWARE_RULE = True


def act_items(m: int, n2: int) -> int:
    """(row, 128-column group) items of the act kernels for ``[M, 2H]``."""
    return m * (n2 // 2 // GROUP_SIZE)


def _diverted_act_route(m: int, n2: int) -> str:
    return ACT_WIDE_ROUTE if act_items(m, n2) >= ACT_WIDE_MIN_ITEMS else ACT_ROUTE


def select_route(
    m: int, n2: int, *, sm_count: int, group_blocks: Optional[Sequence[int]] = None
) -> str:
    """Route of a problem: ``M < SMALL_M_MAX`` takes the GEMM + act route; larger
    problems take the fused route unless the routing is known (``group_blocks``,
    128-row blocks per expert) and its odd-tail units outnumber the SMs the pair
    grid leaves free, in which case they take the GEMM + wide act route (the
    small-M act route below ``ACT_WIDE_MIN_ITEMS`` items).
    """
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if m <= 0 or n2 <= 0:
        raise ValueError("select_route requires positive M and 2H")
    if m < SMALL_M_MAX:
        return ACT_ROUTE
    if group_blocks is None or not ROUTING_AWARE_RULE:
        return FUSED_ROUTE
    if sum(int(b) for b in group_blocks) != -(-m // ROW_BLOCK):
        raise ValueError(
            f"group_blocks {list(group_blocks)} do not cover M={m} ({-(-m // ROW_BLOCK)} blocks)"
        )
    _, _, cluster_ctas = route_geometry(FUSED_ROUTE)
    pair_ctas = (min(sm_count, CTA_CAP) // cluster_ctas) * cluster_ctas
    _, odd_units = fused_tile_counts(n2, group_blocks)
    return (
        FUSED_ROUTE if odd_units <= sm_count - pair_ctas else _diverted_act_route(m, n2)
    )


def launch_plan(
    m: int, n2: int, *, sm_count: int, group_blocks: Optional[Sequence[int]] = None
) -> tuple[str, tuple[int, int, int]]:
    """Resolve ``(route, grid)`` for ``A[M, K]`` and ``B[G, 2H, K]`` on ``sm_count`` SMs.

    The route follows :func:`select_route` (``group_blocks`` = 128-row blocks
    per expert enables the routing-aware rule; ``prepare_...`` derives them
    when ``validate_indices=True``).  The GEMM + act route launches one warp
    per (row, 128-column group) item, ``ACT_WARPS_PER_CTA`` warps per CTA, at
    most ``ACT_CTAS_PER_SM`` CTAs per SM; the wide act route one warp per
    ``ACT_WIDE_GROUPS_PER_WARP`` items, at most ``ACT_WIDE_CTAS_PER_SM`` CTAs
    per SM.  The fused route launches complete
    CTA pairs; its tile upper bound counts every 128-row block as a lone block,
    and the 128-CTA cap is the measured B200 policy of the parent gate_up route
    (148 CTAs measured slower on every routing).
    """
    route = select_route(m, n2, sm_count=sm_count, group_blocks=group_blocks)
    if route in ACT_ROUTES:
        route_geometry(route)  # the registry must carry the route
        items = act_items(m, n2)
        if route == ACT_WIDE_ROUTE:
            warps = -(-items // ACT_WIDE_GROUPS_PER_WARP)
            ctas_cap = ACT_WIDE_CTAS_PER_SM * sm_count
        else:
            warps = items
            ctas_cap = ACT_CTAS_PER_SM * sm_count
        ctas = max(1, min(-(-warps // ACT_WARPS_PER_CTA), ctas_cap))
        return route, (ctas, 1, 1)
    _, tile_n, cluster_ctas = route_geometry(route)
    max_tiles = math.ceil(m / ROW_BLOCK) * (n2 // tile_n)
    clusters = min(max_tiles, min(sm_count, CTA_CAP) // cluster_ctas)
    if clusters <= 0:
        raise ValueError("device cannot schedule one complete CTA pair")
    return route, (clusters * cluster_ctas, 1, 1)


def tail_launch_grid(m: int, n2: int, *, sm_count: int) -> tuple[int, int, int]:
    """Grid of the fused route's tail kernel for ``A[M, K]`` and ``B[G, 2H, K]`` on ``sm_count`` SMs.

    One single-CTA 128-row tile per odd-tail block and N256 tile; the grid is
    one CTA per SM up to the unit upper bound (every block an odd tail) and the
    CTAs grid-stride over the actual units, so the grid does not depend on the
    routing.  A routing without odd tails runs an empty grid, overlapped
    with the pair kernel by programmatic dependent launch.
    """
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    tile_m, tile_n, _ = route_geometry(TAIL_GEOMETRY_KEY)
    return (max(1, min(math.ceil(m / tile_m) * (n2 // tile_n), sm_count)), 1, 1)


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


def _validate_indices(m_indices: torch.Tensor, groups: int) -> tuple[int, ...]:
    """Synchronizing check of the routing contract (sorted, in range, 128-aligned boundaries).

    Returns the 128-row block count of every expert (the routing statistics of
    :func:`select_route`); every block is homogeneous under the contract, so the
    expert of a block is the index of its first row.
    """
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
    block_experts = indices[::ROW_BLOCK]
    return tuple(
        int(v) for v in torch.bincount(block_experts, minlength=groups).tolist()
    )


@dataclass
class PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant:
    """One prepared gate_up GEMM + SwiGLU + FP8 quantization launch.

    Tensor storage, shapes and dtypes are bound at preparation; tensor
    *contents* may change between launches.  ``launch()`` submits exactly
    ``num_kernels`` kernels (two for the fused route: the pair kernel, then
    the tail kernel with the programmatic-dependent-launch attribute; two for
    the GEMM + act routes: the grouped GEMM into the private BF16 workspace,
    then the generated activation kernel) on PyTorch's current stream for the bound
    device and returns ``(out_q, out_s)``.  The first launch initializes
    private TMA descriptor storage and must run outside CUDA Graph capture.
    ``grid`` is the grid of the route's first generated kernel;
    ``stage_grids`` holds every generated kernel's grid by stage name.
    """

    route: str
    module_name: str
    grid: tuple[int, int, int]
    out_q: torch.Tensor
    out_s: torch.Tensor
    gemm_backend: Optional[str]
    stage_module_names: dict[str, str]
    stage_grids: dict[str, tuple[int, int, int]]
    _entries: tuple[tuple[Callable[..., Any], tuple[Any, ...]], ...]
    _descriptor_storages: tuple[torch.Tensor, ...]
    _gemm: Optional[Callable[[], Any]]
    _gemm_out: Optional[torch.Tensor]

    def launch(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.device(self.out_q.device):
            if self._gemm is not None:
                self._gemm()
            with tvm_ffi.use_torch_stream():
                for entry, arguments in self._entries:
                    entry(*arguments)
        return self.out_q, self.out_s

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        """CTAs of the route's first generated kernel (the pair kernel or the activation kernel)."""
        return int(self.grid[0])

    @property
    def tail_grid(self) -> Optional[tuple[int, int, int]]:
        """Grid of the fused route's tail kernel (``None`` on the GEMM + act routes)."""
        return self.stage_grids.get(FUSED_TAIL_STAGE)

    @property
    def num_kernels(self) -> int:
        return len(self._entries) + (0 if self._gemm is None else 1)


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
    r"""Prepare a contiguous grouped FP8 gate_up GEMM + SwiGLU + FP8 quantization launch on SM100a.

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
        Call ``.launch()`` to run the route's kernel(s) on the current stream.

    Notes
    -----
    Requires an SM100a (compute capability 10.0) device and a registered
    generated program for the resolved route (:func:`launch_plan`).  All
    tensors must live on the same CUDA device and be 16-byte aligned
    (``out_s`` 4-byte).  The outputs match the three-kernel FlashInfer chain
    (CuTe-DSL grouped GEMM, ``silu_and_mul``, ``per_token_group_quant_8bit``)
    element for element on the same inputs.  ``M < SMALL_M_MAX`` rows run the
    FlashInfer grouped GEMM (:func:`small_m_gemm_backend`) into a BF16
    ``(M, 2H)`` workspace owned by the prepared object, then one generated
    SwiGLU + group-quantization kernel.  Index values are unchecked unless
    ``validate_indices=True``; violating the routing contract is undefined
    behavior.
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
    group_blocks: Optional[tuple[int, ...]] = None
    if validate_indices:
        group_blocks = _validate_indices(m_indices, groups)

    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    route, grid = launch_plan(m, n2, sm_count=sm_count, group_blocks=group_blocks)
    stage_grids: dict[str, tuple[int, int, int]] = {ROUTE_STAGES[route][0]: grid}
    if route == FUSED_ROUTE:
        stage_grids[FUSED_TAIL_STAGE] = tail_launch_grid(m, n2, sm_count=sm_count)
    stage_module_names = {
        stage: select_stage_module(arch, route, stage) for stage in ROUTE_STAGES[route]
    }
    module_name = stage_module_names[ROUTE_STAGES[route][0]]

    gemm_backend: Optional[str] = None
    gemm: Optional[Callable[[], Any]] = None
    gemm_out: Optional[torch.Tensor] = None
    if route == FUSED_ROUTE:
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
    else:
        gemm_backend = small_m_gemm_backend(m, n2, k)
        gemm_out = torch.empty((m, n2), dtype=torch.bfloat16, device=device)
        if gemm_backend == GEMM_BACKEND_CAKE:
            gemm = prepare_group_gemm_fp8_nt_groupwise_contiguous(
                a, b, a_scale, b_scale, m_indices, out=gemm_out
            ).launch
        else:
            from .gemm_base import group_gemm_fp8_nt_groupwise_contiguous

            def gemm(y: torch.Tensor = gemm_out) -> None:
                group_gemm_fp8_nt_groupwise_contiguous(
                    a, b, a_scale, b_scale, m_indices, out=y
                )

        bindings = {"y": gemm_out, "out_q": out_q, "out_s": out_s, "M": m, "H": h}

    entries = []
    descriptor_storages = []
    for stage in ROUTE_STAGES[route]:
        stage_module_name = stage_module_names[stage]
        record = MODULES[stage_module_name]
        module = load_cake_grouped_fp8_fused_silu_quant_module(stage_module_name)
        entry = getattr(module, record["ffi_entry"])
        descriptor_storage: Optional[torch.Tensor] = None
        workspace_bytes = int(record["tma_workspace_bytes"])
        if route == FUSED_ROUTE:
            descriptor_storage = torch.empty(
                max(workspace_bytes, 128), dtype=torch.uint8, device=device
            )
            descriptor_storages.append(descriptor_storage)
        elif workspace_bytes:
            raise RuntimeError(
                f"generated program {stage_module_name} of route {route!r} unexpectedly needs TMA descriptor storage"
            )
        grid_by_axis = dict(
            zip(("grid_x", "grid_y", "grid_z"), stage_grids[stage], strict=True)
        )
        arguments = []
        for kind, name in record["arg_plan"]:
            if kind == "grid":
                arguments.append(grid_by_axis[name])
            elif kind == "workspace":
                if descriptor_storage is None:
                    raise RuntimeError(
                        f"generated program {stage_module_name} binds descriptor storage this host plan does not own"
                    )
                arguments.append(descriptor_storage)
            elif name in bindings:
                arguments.append(bindings[name])
            else:
                raise RuntimeError(
                    f"generated program {stage_module_name} binds {name!r}, which this host plan does not declare"
                )
        entries.append((entry, tuple(arguments)))
    return PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant(
        route=route,
        module_name=module_name,
        grid=grid,
        out_q=out_q,
        out_s=out_s,
        gemm_backend=gemm_backend,
        stage_module_names=stage_module_names,
        stage_grids=stage_grids,
        _entries=tuple(entries),
        _descriptor_storages=tuple(descriptor_storages),
        _gemm=gemm,
        _gemm_out=gemm_out,
    )


__all__ = [
    "ACT_ROUTE",
    "ACT_ROUTES",
    "ACT_WIDE_MIN_ITEMS",
    "ACT_WIDE_ROUTE",
    "act_items",
    "FUSED_ROUTE",
    "GEMM_BACKEND_CAKE",
    "GEMM_BACKEND_CUTE",
    "SMALL_M_MAX",
    "PreparedGroupGemmFp8NtGroupwiseContiguousSiluQuant",
    "is_group_gemm_fp8_nt_groupwise_contiguous_silu_quant_prepared_available",
    "fused_tile_counts",
    "launch_plan",
    "prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant",
    "routing_blocks",
    "select_route",
    "small_m_gemm_backend",
    "tail_launch_grid",
]
