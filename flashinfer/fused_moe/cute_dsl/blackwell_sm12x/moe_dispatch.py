"""SM120/SM121 MoE dispatch layer — workspace, compilation, and launch.

Ported from b12x's integration/tp_moe.py. Supports micro (tiny decode),
static (decode), and dynamic (prefill) backends with token-count-based
selection.
"""

from __future__ import annotations

import hashlib
import os
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import (
    convert_sf_from_mma_layout,
    convert_sf_to_mma_layout,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from .moe_activation import SWIGLUOAI_UNINTERLEAVE, is_gated_activation
from .moe_direct_micro_kernel import (
    MoEDirectMicroKernel,
    build_direct_micro_kernel,
    compile_direct_micro_kernel,
    compiled_direct_micro_accepts_block_dim,
)
from .moe_dynamic_kernel import (
    _MAX_SHARED_INPUT_TOPK,
    _TASK_SLICE_CHUNK,
    MoEDynamicKernel,
    _can_use_gated_optimized_kernel,
)
from .moe_micro_kernel import MoEMicroKernel
from .moe_static_kernel import MoEStaticKernel
from .moe_w4a16_fp4_helpers import swizzle_block_scale
from .moe_w4a16_host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    route_pack_numel_capacity,
    unswizzle_block_scale,
    validate_activation,
)
from .moe_w4a16_kernel import run_w4a16_moe
from .moe_w4a16_prepare import (
    W4A16PackedWeights,
    _normalize_source_format,
    prepare_w4a16_packed_weights,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NVFP4_BLOCK_SIZE = 16
_MXFP4_BLOCK_SIZE = 32
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
# Virtual experts hold at most 32 rows; M128 would double padded MMA work.
_COMPACT_MMA_TILER_MN = (64, 128)
_STATIC_RETAINED_GROUP_N = 2 * _LEVEL_TILE_N
# Rows stored per compact (virtual) expert slot in the static workspace.  The
# static route kernel splits every expert into 32-row chunks (chunk =
# alloc_row >> 5), each with its own slot, so no slot ever holds more rows;
# the kernels read the slot stride from token_map.shape[1].  Sizing slots at
# 32 rows instead of the routed-row capacity keeps the packed-activation
# planes at a few MB for any capacity (E=512, 2560 rows: 2.2 GB -> ~40 MB).
_STATIC_SLOT_ROWS = 32
# Must equal the kernel's task materialization granularity or the task
# queue is mis-sized.
_DYNAMIC_SLICE_CHUNK = _TASK_SLICE_CHUNK
SF_VEC_SIZE = 16
_FORCE_MOE_W4A16_ENV = "FLASHINFER_B12X_FORCE_MOE_W4A16"
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = (
    os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
)

# Micro kernel cutover thresholds (routed pairs)
_MICRO_COMPACT_CUTOVER_PAIRS = 20
_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK = 40
# The micro kernel's per-token staging assumes decode-sized batches.
_MICRO_MAX_TOKENS = 8
# Direct micro takes the smallest decode batches ahead of the MMA micro
# kernel, and only at small intermediate sizes where its CUDA-core dots
# keep up with per-token work. Measured on GB10, pending other GPUs.
_DIRECT_MICRO_CUTOVER_PAIRS = 32
_DIRECT_MICRO_MAX_N = 512
# Test/bench hook: force one backend ("direct_micro", "micro", "static",
# "dynamic"). Deliberately module-level (a monkeypatch target), not an env var.
_FORCED_BACKEND: str | None = None
# Production static band: pairs <= 640 (M <= 80 at topk=8). The kernel's
# 32-row virtual-expert split keeps every scheduled tile inside the tile64
# kernel's single computed M step, making the full compact band correct at
# any per-expert row count.
_STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT = 640
# The retained NVFP4 path uses the tile64 schedule through 1024 routed pairs.
# MXFP4 keeps the generic 640-pair compact boundary.
_STATIC_COMPACT_CUTOVER_PAIRS_NVFP4_DEFAULT = 1024
# The static/dynamic crossover sits at a routed-row density (rows per weight
# expert), so the NVFP4 band scales with E above the 1024-pair floor.  These
# two densities are the legacy fallback for keys without a measured
# registry entry (below): 8 rows per expert when the intermediate extent is a
# 128 multiple, 16 otherwise (the padded extents keep a wider band because
# their static path streams the true extent while the generic dynamic path
# still consumes 128-padded weights; the branch-major gated dynamic kernel
# streams the true extent on branch-major views, so the 16-row band is a measured legacy
# of the shapes it was derived on, not a byte argument). Eligible merged-three
# schedules use the separately calibrated density below.
_STATIC_COMPACT_CUTOVER_ROWS_PER_EXPERT = 8
_STATIC_COMPACT_CUTOVER_ROWS_PER_EXPERT_PADDED = 16


# Measured static/dynamic cutover registry.  Key =
# (quant_mode, activation, E, H, I_true, top_k, sm_count, wrapper capacity in
# tokens); value = routed-pair cutover (static while routed pairs <= value).
# Match every key part: measurements do not transfer to another geometry,
# device SM count, or wrapper capacity. Unknown keys use the density rule.
def _static_cutover_capacity_key(capacity_tokens: int | None) -> int | None:
    """Capacity part of the cutover key: the wrapper's token capacity itself
    (None when the caller does not know it, which disqualifies the registry)."""
    if capacity_tokens is None:
        return None
    capacity_tokens = int(capacity_tokens)
    return capacity_tokens if capacity_tokens > 0 else None


_STATIC_CUTOVER_REGISTRY: Dict[Tuple[str, str, int, int, int, int, int, int], int] = {
    # Seven rows/expert balances random and skewed routes; eight regresses Zipf.
    ("nvfp4", "silu", 256, 2048, 512, 8, 110, 8192): 1792,
}


def _static_cutover_registry_lookup(
    quant_mode: str | None,
    activation: str | None,
    num_experts: int | None,
    hidden_size: int | None,
    intermediate_size: int | None,
    num_topk: int | None,
    sm_count: int | None,
    capacity_tokens: int | None,
) -> int | None:
    """Registry cutover for a fully specified key; None when any key part
    (including the wrapper capacity) is missing or the key is unmeasured (the
    caller then applies the density rule)."""
    capacity_key = _static_cutover_capacity_key(capacity_tokens)
    if None in (
        quant_mode,
        activation,
        num_experts,
        hidden_size,
        intermediate_size,
        num_topk,
        sm_count,
        capacity_key,
    ):
        return None
    key = (
        str(quant_mode),
        str(activation),
        int(num_experts),
        int(hidden_size),
        int(intermediate_size),
        int(num_topk),
        int(sm_count),
        int(capacity_key),
    )
    return _STATIC_CUTOVER_REGISTRY.get(key)


_STATIC_COMPACT_CUTOVER_PAIRS = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
_STATIC_COMPACT_CUTOVER_PAIRS_CACHE: Dict[Tuple, int] = {}

# MAC (max active clusters) tuning ladders from b12x decode profiling.
# Each entry is (max_routed_rows, optimal_mac).
_MICRO_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (2, 84),
    (4, 127),
    (8, 107),
    (10, 84),
    (16, 63),
    (20, 84),
)
# Max-active-cluster policy for the retained2 tile64 static band, keyed by
# routed-row range after the 32-row virtual-expert split.
_STATIC_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (64, 110),
    (256, 92),
    (448, 110),
    (512, 92),
    (640, 110),
    (768, 92),
    (1024, 110),
)
# Workloads at or below the static cutover (640 pairs MXFP4, 1024 NVFP4)
# take the static kernel, so for NVFP4 these entries are reachable only
# when the dynamic backend is forced; MXFP4 still reaches the 1024 entry.
_DYNAMIC_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (640, 188),
    (1024, 147),
)


def _lookup_mac_ladder(
    ladder: Tuple[Tuple[int, int], ...], routed_rows: int
) -> int | None:
    """Look up optimal MAC from a tuning ladder. Returns None if no match."""
    for end_rows, mac in ladder:
        if routed_rows <= end_rows:
            return mac
    return None


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _static_retained_groups(n: int) -> int:
    """Retained2 groups for intermediate size ``n``: pairs of N128 slices,
    rounded up.  An odd slice count keeps a phantom partner slice that TMA
    zero-fills; the finalize sums exactly this many route-scratch slots."""
    return max(1, _align_up(n, _STATIC_RETAINED_GROUP_N) // _STATIC_RETAINED_GROUP_N)


# Below this measured floor, merging tasks underfills the resident grid.
_STATIC_MERGED_GROUPS_MIN_PAIRS = 1920
# Three retained slices remove the phantom fourth slice and one FC2
# epilogue. Boundary A/B supports a conservative 17 rows/expert for this
# schedule; 20 helped random routes but regressed skewed routes. This is a
# calibrated heuristic, not a universal optimum across routing distributions.
_STATIC_MERGED_CUTOVER_ROWS_PER_EXPERT = 17


def _static_merged_groups(n: int, routed_pairs: int) -> bool:
    """Merged-groups dispatch rule: run the static kernel's merged schedule (one task
    per expert, every N128 slice retained, FC2 once, route scratch ``[route, K]``)
    for the three-slice intermediate extents (256 < n <= 384, the retained-three
    specialization) at or above the calibrated routed-pair floor.  Two-slice
    extents already form one group and keep the retained2 schedule; four or more
    slices do not fit the three A/SFA stages.  The selected schedule is part of
    the static cache key and artifact name."""
    slices = max(1, _align_up(n, _LEVEL_TILE_N) // _LEVEL_TILE_N)
    return slices == 3 and routed_pairs >= _STATIC_MERGED_GROUPS_MIN_PAIRS


# Amortize finalize's counter restore before removing the prologue clear.
_STATIC_DEFERRED_INIT_MIN_PAIRS = 256


def _static_deferred_init(routed_pairs: int) -> bool:
    """Defer counter clearing when the launch amortizes the finalize restore."""
    return int(routed_pairs) >= _STATIC_DEFERRED_INIT_MIN_PAIRS


# The kernels index the packed activation and scale planes with 32-bit
# offsets, so reject any workspace plane large enough to overflow them.
_RUNTIME_MEMREF_LIMIT = (1 << 31) - 1


def _check_memref_limit(name: str, elements: int) -> None:
    if elements > _RUNTIME_MEMREF_LIMIT:
        raise ValueError(
            f"{name} needs {elements} elements, which exceeds the 2^31-1 "
            "runtime memref limit. Reduce the token chunk or expert count "
            "for this launch."
        )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return None


def _normalize_activation_precision(activation_precision: str) -> str:
    """Normalize public activation-precision names to internal modes."""
    if os.environ.get(_FORCE_MOE_W4A16_ENV, "0") == "1":
        return "bf16"

    normalized = str(activation_precision).lower()
    aliases = {
        "fp4": "fp4",
        "nvfp4": "fp4",
        "w4a4": "fp4",
        "bf16": "bf16",
        "w4a16": "bf16",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "activation_precision must be 'fp4' or 'bf16' "
            f"(got {activation_precision!r})."
        ) from exc


def _normalize_quant_mode(
    quant_mode: str | None = None,
    activation_precision: str | None = None,
) -> str:
    """Normalize public quantization names to the dispatch mode."""
    if os.environ.get(_FORCE_MOE_W4A16_ENV, "0") == "1":
        return "w4a16"
    if quant_mode is None:
        activation_precision = _normalize_activation_precision(
            activation_precision or "fp4"
        )
        return "w4a16" if activation_precision == "bf16" else "nvfp4"

    normalized = str(quant_mode).lower()
    aliases = {
        "fp4": "nvfp4",
        "nvfp4": "nvfp4",
        "w4a4": "nvfp4",
        "mxfp4": "mxfp4",
        "bf16": "w4a16",
        "w4a16": "w4a16",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "quant_mode must be 'nvfp4'/'w4a4', 'mxfp4', or 'w4a16' "
            f"(got {quant_mode!r})."
        ) from exc


def _sf_params_for_quant_mode(quant_mode: str):
    """Return (vector size, CuTe scale dtype) for a W4A4 mode."""
    mode = _normalize_quant_mode(quant_mode)
    if mode == "mxfp4":
        return _MXFP4_BLOCK_SIZE, cutlass.Float8E8M0FNU
    return _NVFP4_BLOCK_SIZE, cutlass.Float8E4M3FN


def _activation_precision_from_quant_mode(quant_mode: str) -> str:
    return "bf16" if _normalize_quant_mode(quant_mode) == "w4a16" else "fp4"


def _normalize_source_format_for_quant_mode(source_format: str, quant_mode: str) -> str:
    normalized = _normalize_source_format(source_format)
    if quant_mode != "w4a16" and normalized == "compressed_tensors":
        raise ValueError(
            "source_format='compressed_tensors' requires quant_mode='w4a16'."
        )
    return normalized


def _is_w4a16(activation_precision: str) -> bool:
    return _normalize_activation_precision(activation_precision) == "bf16"


def _level_tile_n(activation_precision: str = "fp4") -> int:
    if _is_w4a16(activation_precision):
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 tile selector"
        )
    return _LEVEL_TILE_N


def _select_dynamic_tile_m(
    routed_rows: int,
    num_experts: int,
    activation: str = "silu",
) -> int:
    """Pick the dynamic kernel's M-tile from routed rows per expert.

    Small tiles cut per-expert tail padding for sparse routing; 128 amortizes
    best for dense prefill (crossovers measured on gated NVFP4). Workspace
    sizing and the kernel build must both derive the tile from this function,
    or the scratch is mis-sized for what the kernel indexes.
    """
    if not is_gated_activation(activation):
        return _LEVEL_TILE_M
    routed_rows = max(1, int(routed_rows))
    num_experts = max(1, int(num_experts))
    if routed_rows < 15 * num_experts:
        return 16
    if routed_rows < 48 * num_experts:
        return 32
    if routed_rows < 96 * num_experts:
        return 64
    return _LEVEL_TILE_M


def _get_static_compact_cutover_pairs(
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    num_experts: int | None = None,
    intermediate_size: int | None = None,
    *,
    hidden_size: int | None = None,
    activation: str | None = None,
    num_topk: int | None = None,
    capacity_tokens: int | None = None,
    sm_count: int | None = None,
) -> int:
    """Routed-pair cutover of the static band: environment override, else the
    measured registry entry of the full key (quant mode, activation, E, H,
    I_true, top-k, SM count, wrapper capacity), else the density rule."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if sm_count is None and None not in (
        hidden_size,
        activation,
        num_topk,
        num_experts,
        intermediate_size,
    ):
        sm_count = (
            get_num_sm(torch.device("cuda")) if torch.cuda.is_available() else None
        )
    capacity_key = _static_cutover_capacity_key(capacity_tokens)
    registry_key_known = None not in (
        quant_mode,
        activation,
        num_experts,
        hidden_size,
        intermediate_size,
        num_topk,
        sm_count,
        capacity_key,
    )
    padded_intermediate = (
        intermediate_size is not None and int(intermediate_size) % _LEVEL_TILE_N != 0
    )
    # A single N128 slice makes the static family stream a 256-aligned
    # extent (twice the true weights), so it keeps the flat boundary instead
    # of the density-widened band.
    single_slice = intermediate_size is not None and static_needs_256_extent(
        int(intermediate_size)
    )
    # Partial keys also affect the fallback's activation/top-k/slice policy.
    cache_key = (
        activation_precision,
        quant_mode,
        num_experts,
        intermediate_size,
        hidden_size,
        activation,
        num_topk,
        sm_count,
        capacity_key,
    )
    cached = _STATIC_COMPACT_CUTOVER_PAIRS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cutover_names: tuple[str, ...] = (
        "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_DYNAMIC_STATIC_CUTOVER_PAIRS",
        "B12X_LEVEL10_STATIC_CUTOVER_PAIRS",
    )
    cutover = _first_env(*cutover_names)
    registry_pairs = (
        _static_cutover_registry_lookup(
            quant_mode,
            activation,
            num_experts,
            hidden_size,
            intermediate_size,
            num_topk,
            sm_count,
            capacity_tokens,
        )
        if registry_key_known
        else None
    )
    if cutover is None and registry_pairs is not None:
        cached = int(registry_pairs)
    elif cutover is None:
        # Only the retained NVFP4 implementation earns the wider band; the
        # generic implementation (MXFP4, or unspecified quant mode) keeps
        # the original boundary.
        if quant_mode == "nvfp4":
            cached = _STATIC_COMPACT_CUTOVER_PAIRS_NVFP4_DEFAULT
            if num_experts is not None and not single_slice:
                rows_per_expert = (
                    _STATIC_COMPACT_CUTOVER_ROWS_PER_EXPERT_PADDED
                    if padded_intermediate
                    else _STATIC_COMPACT_CUTOVER_ROWS_PER_EXPERT
                )
                if (
                    activation is not None
                    and is_gated_activation(activation)
                    and num_topk is not None
                    and num_topk > 1
                    and intermediate_size is not None
                    and 256 < intermediate_size <= 384
                    and _static_merged_groups(
                        intermediate_size, rows_per_expert * int(num_experts)
                    )
                ):
                    rows_per_expert = _STATIC_MERGED_CUTOVER_ROWS_PER_EXPERT
                cached = max(cached, rows_per_expert * int(num_experts))
        else:
            cached = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
    else:
        cached = max(0, int(cutover))
    _STATIC_COMPACT_CUTOVER_PAIRS_CACHE[cache_key] = cached
    return cached


def _as_grouped_scale_view(
    scale_storage: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Create 6D MMA-compatible scale factor view from swizzled storage."""
    batch = scale_storage.shape[0]
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols // SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e4m3fn)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class Sm120StaticMoEWorkspace:
    """Scratch buffers for one SM120 static MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    activation_precision: str
    quant_mode: str

    # Buffers
    row_counts: torch.Tensor  # [state_E] int32
    token_map: torch.Tensor  # [virt_E, _STATIC_SLOT_ROWS] int32
    token_weights: torch.Tensor  # [virt_E, _STATIC_SLOT_ROWS] float32
    packed_input: torch.Tensor  # [virt_E, _STATIC_SLOT_ROWS, k//2] uint8
    packed_input_scale: torch.Tensor  # [virt_E, 128, cols_pad_k] uint8
    barrier_count: torch.Tensor  # [1] int32
    barrier_epoch: torch.Tensor  # [1] int32
    active_expert_count: torch.Tensor  # [1] int32
    weight_expert_ids: torch.Tensor  # [virt_E] int32
    global_to_local_expert: torch.Tensor  # [weight_E] int32
    compact_topk_ids: torch.Tensor  # [state_E] int32, for micro kernel pre-pass
    # 32-row virtual-expert split scratch: [weight_E] monotone row allocators
    # followed by [weight_E, max_chunks] chunk -> local expert id map. The
    # static kernel clears it when the workspace carries no clean marker and
    # its finalize kernel restores the clean state afterwards (graph-replay safe).
    virt_route_scratch: torch.Tensor  # [weight_E*(1+max_chunks)] int32
    # [0] clean marker published by the static kernel's finalize (rest reserved).
    route_state: torch.Tensor  # [8] int32
    # The tiny-decode micro kernel keeps private copies of the two counters it
    # shares with the static kernel, so it can never disturb the clean state.
    micro_row_counts: torch.Tensor  # [state_E] int32
    micro_active_expert_count: torch.Tensor  # [1] int32
    route_output_scratch: torch.Tensor  # [max_rows, ceil(n/256), k] bf16

    # Views (set after allocation)
    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None

    # Direct micro planes (allocated only when the shape can take that path).
    dm_barrier_count: torch.Tensor | None = None
    dm_barrier_epoch: torch.Tensor | None = None
    dm_intermediate: torch.Tensor | None = None
    dm_input_gs: torch.Tensor | None = None
    dm_down_input_scale: torch.Tensor | None = None


def _direct_micro_candidate(k: int, n: int, num_topk: int, weight_E: int) -> bool:
    """Whether any m in the tiny-decode band can run the direct micro kernel."""
    return any(
        MoEDirectMicroKernel.is_supported(m, k, n, num_topk, weight_E)
        for m in range(1, _MICRO_MAX_TOKENS + 1)
    )


def allocate_sm120_static_workspace(
    *,
    state_E: int,
    weight_E: int,
    max_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
) -> Sm120StaticMoEWorkspace:
    """Allocate workspace buffers for the SM120 static MoE kernel."""
    n = _align_up(n, _LEVEL_TILE_N)
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "allocate_sm120_static_workspace only supports quant_mode='nvfp4'; "
            "use allocate_sm120_moe_workspace(..., quant_mode='w4a16') for W4A16."
        )

    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    # Per-slot activation planes: 32 routed rows (the chunk size) and one
    # 128-row block-scale atom.  max_rows stays the routed-row capacity.
    slot_rows = _STATIC_SLOT_ROWS
    rows_pad_k = _align_up(slot_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)
    # The 32-row virtual-expert split may open extra compact expert slots for
    # experts with >32 routed rows. Sum(ceil(rows_e/32)) <= actives +
    # total_rows/32, so ceil(max_rows/32) extra slots always suffice.
    max_chunks = (max_rows + 31) // 32
    virt_E = state_E + max_chunks
    retained_groups = _static_retained_groups(n)
    _check_memref_limit("static packed_input", virt_E * slot_rows * (k // 2))
    _check_memref_limit("static packed_input_scale", virt_E * rows_pad_k * cols_pad_k)
    _check_memref_limit("static route_output_scratch", max_rows * retained_groups * k)
    packed_input = torch.empty(
        virt_E, slot_rows, k // 2, dtype=torch.uint8, device=device
    )

    workspace = Sm120StaticMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        row_counts=torch.zeros(virt_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(virt_E, slot_rows, dtype=torch.int32, device=device),
        token_weights=torch.zeros(
            virt_E, slot_rows, dtype=torch.float32, device=device
        ),
        packed_input=packed_input,
        packed_input_scale=torch.empty(
            virt_E, rows_pad_k, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        weight_expert_ids=torch.arange(virt_E, dtype=torch.int32, device=device),
        global_to_local_expert=torch.empty(weight_E, dtype=torch.int32, device=device),
        compact_topk_ids=torch.empty(
            max(state_E, max_rows), dtype=torch.int32, device=device
        ),
        virt_route_scratch=torch.empty(
            # Row allocator + chunk map + work-claim counter (8-slot pad);
            # shared by the retained NVFP4 and MXFP4 implementation.
            weight_E * (1 + max_chunks) + 8,
            dtype=torch.int32,
            device=device,
        ),
        route_state=torch.zeros(8, dtype=torch.int32, device=device),
        micro_row_counts=torch.zeros(virt_E, dtype=torch.int32, device=device),
        micro_active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        route_output_scratch=torch.empty(
            max_rows,
            retained_groups,
            k,
            dtype=torch.bfloat16,
            device=device,
        ),
    )

    # Finalize views
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    # Direct micro reads weights by global expert id, so its planes are only
    # useful without EP remapping.
    if (
        quant_mode == "nvfp4"
        and state_E == weight_E
        and _direct_micro_candidate(k, n, num_topk, weight_E)
    ):
        dm_rows = min(max_rows, _MICRO_MAX_TOKENS * num_topk)
        # The epoch-based barriers restore their slots after each launch, so
        # the zeroed allocation is the only reset needed (graph-replay safe).
        dm_slots = dm_rows + _MICRO_MAX_TOKENS * 16
        fc2_n_chunks = (n // 2 + 127) // 128
        # The fused kernel binds the intermediate as m * num_topk *
        # fc2_n_chunks * 128 u32 words; size for the largest supported m.
        dm_inter = _MICRO_MAX_TOKENS * num_topk * fc2_n_chunks * 128
        workspace.dm_barrier_count = torch.zeros(
            dm_slots, dtype=torch.int32, device=device
        )
        workspace.dm_barrier_epoch = torch.zeros(
            dm_slots, dtype=torch.int32, device=device
        )
        workspace.dm_intermediate = torch.empty(
            dm_inter, dtype=torch.float32, device=device
        )
        workspace.dm_input_gs = torch.empty(
            weight_E, dtype=torch.float32, device=device
        )
        workspace.dm_down_input_scale = torch.empty(
            weight_E, dtype=torch.float32, device=device
        )
    return workspace


# ---------------------------------------------------------------------------
# Weight views
# ---------------------------------------------------------------------------
@dataclass
class _WeightViews:
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None
    w1_alpha: torch.Tensor | None = None
    w2_alpha: torch.Tensor | None = None
    w1_storage: torch.Tensor | None = None
    w1_scale_storage: torch.Tensor | None = None
    w2_storage: torch.Tensor | None = None
    w2_scale_storage: torch.Tensor | None = None
    _w13_sf_storage: torch.Tensor | None = None
    _down_sf_storage: torch.Tensor | None = None
    # Static-kernel views at the true intermediate extent: gated w13 as
    # [n, k//2, 2E] with the up branch at batch 2e and gate at 2e+1, down as
    # [k, n//2, E].  TMA zero-fills past the extent, so a non-tile-multiple
    # intermediate size costs no padded-weight traffic.
    static_w13_fp4: object = None
    static_down_fp4: object = None
    intermediate_size: int | None = None
    # Extent the static views span and, for single-slice shapes, the static
    # kernel's own 256-aligned scale storages (see _get_weight_views).
    static_intermediate_size: int | None = None
    static_w13_sf_storage: torch.Tensor | None = None
    static_down_sf_storage: torch.Tensor | None = None
    # Branch-major views for the gated dynamic kernel at ``intermediate_size``
    # (the static views may span a larger extent, see static_override).
    branch_major_w13_fp4: object = None
    branch_major_down_fp4: object = None
    # Single-slice shapes: the whole static family (static, MMA micro, direct
    # micro) consumes the 256-aligned copies; keep their backing storages.
    static_family_w1_storage: torch.Tensor | None = None
    static_family_w2_storage: torch.Tensor | None = None
    # Lazy legacy views: when the caller hands over unpadded weights, the
    # tile-padded [2n, k//2, E] / [k, n//2, E] copies the dynamic and micro
    # kernels consume are built by this callable on first use only.
    _legacy_builder: Optional[Callable[[], Tuple[torch.Tensor, torch.Tensor]]] = None
    # Source scale layout for the static kernel: the scale storages above are the
    # caller's own (no copy); the tile-padded scales the dynamic / micro kernels
    # index are built by this callable on their first use only.
    source_scales: bool = False
    _padded_scale_builder: Optional[Callable[[], Tuple[torch.Tensor, torch.Tensor]]] = (
        None
    )
    _padded_w13_sf_storage: torch.Tensor | None = None
    _padded_down_sf_storage: torch.Tensor | None = None

    def padded_scales(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """The tile-padded block-scale storages (dynamic / micro / direct-micro
        operands).  Without the source-scale mode these are the storages the
        static kernel uses too; with it they are materialized on first use and
        refused during CUDA graph capture (run one eager call on that backend)."""
        if not self.source_scales:
            return self._w13_sf_storage, self._down_sf_storage
        if self._padded_w13_sf_storage is None:
            if self._padded_scale_builder is None:
                raise ValueError(
                    "source-scale weight views carry no padded-scale builder"
                )
            _refuse_during_capture(
                "the tile-padded block scales (dynamic / micro kernels)"
            )
            self._padded_w13_sf_storage, self._padded_down_sf_storage = (
                self._padded_scale_builder()
            )
        return self._padded_w13_sf_storage, self._padded_down_sf_storage

    @property
    def legacy_materialized(self) -> bool:
        return self.w13_fp4 is not None and self.down_fp4 is not None

    def ensure_legacy(self) -> None:
        """Materialize the tile-padded legacy weight views on first dynamic /
        micro use.  Static-only callers never reach this.  Refused during CUDA
        graph capture: run one eager call that selects the dynamic or micro
        backend (the natural warm-up) before capturing.
        """
        if self.legacy_materialized:
            return
        if self._legacy_builder is None:
            raise ValueError(
                "weight views carry neither tile-padded legacy views nor a builder"
            )
        if torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "the tile-padded legacy weight views (dynamic / micro kernels) are "
                "materialized on first use and cannot be created during CUDA graph "
                "capture; run one eager warm-up call that selects the dynamic or "
                "micro backend before capturing the graph"
            )
        w1_padded, w2_padded = self._legacy_builder()
        self.w13_fp4 = w1_padded.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        self.down_fp4 = w2_padded.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        self.w1_storage = w1_padded
        self.w2_storage = w2_padded


def _register_cache_eviction(cache: Dict, key: Tuple, *source_tensors) -> None:
    """Evict ``key`` when a source weight tensor is collected, so the cache
    follows the weights' lifetime instead of growing for the whole process.
    """
    for tensor in source_tensors:
        if tensor is not None:
            weakref.finalize(tensor, cache.pop, key, None)


_WEIGHT_CACHE: Dict[Tuple, Tuple] = {}


def _refuse_during_capture(what: str) -> None:
    """Every first-use preparation (padded scale / FP4 copies, converted scale
    views, static-family operands, kernel compilation, workspace allocation)
    allocates or compiles.  Inside a CUDA-graph capture that would be a silent
    allocation, so it is refused: run one eager warm-up call on the same
    wrapper / workspace with the same shapes and backend before capturing.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{what} cannot be prepared during CUDA graph capture; run one eager "
            "warm-up call (same wrapper, shapes and backend) before capturing the graph"
        )


# The retained2 static kernel pairs two N128 slices per group; a shape with a
# single slice (intermediate size <= 128) must present it a 256-aligned
# extent (the earlier padded static family padded to 256 for the same reason).
_STATIC_MIN_EXTENT = 256
# TMA needs 16-byte global strides: a true-extent down view [K, I, E] packs
# I/2 bytes per row, so the true-extent views need I % 32 == 0.
_TRUE_EXTENT_ALIGN = 32


def static_needs_256_extent(intermediate_size: int) -> bool:
    """Whether the static family must see the 256-aligned extent (one N128 slice only)."""
    return _align_up(int(intermediate_size), _LEVEL_TILE_N) < _STATIC_MIN_EXTENT


def static_source_scales(
    intermediate_size: int, is_gated: bool, quant_mode: str = "nvfp4"
) -> bool:
    """Whether the static kernel reads the block scales in the source (modelopt) layout.

    The source stores each expert's w13 scales as 128-row atoms over the 2*I rows
    (up branch first, gate branch from row I).  When I is a multiple of 64 with
    I % 128 == 64 the gate branch starts half an atom in: every gate N128 tile is
    the upper half of one source atom followed by the lower half of the next,
    which the kernel's DMA warp assembles with plain 8-byte loads and shared
    stores (TMA cannot address a half atom), and the down scales already hold
    complete K atoms.  Such a wrapper never materializes padded scale copies
    for the static kernel (the dynamic / micro kernels pad lazily on first use).
    """
    n = int(intermediate_size)
    return (
        bool(is_gated)
        and _normalize_quant_mode(quant_mode) == "nvfp4"
        and n > 128
        and n % 64 == 0
        and n % 128 == 64
        and true_extent_views_supported(n)
    )


def true_extent_views_supported(intermediate_size: int) -> bool:
    """Whether TMA views at the true intermediate extent are legal for this size."""
    return int(intermediate_size) % _TRUE_EXTENT_ALIGN == 0


def _validate_w4a4_dimensions(k: int, n: int, quant_mode: str) -> None:
    if quant_mode not in ("nvfp4", "mxfp4"):
        return
    # Q0/A staging and the FC2 epilogue require full hidden-dimension tiles.
    # Intermediate tails are supported separately by the operand views.
    if k <= 0 or k % _LEVEL_TILE_N:
        raise ValueError(
            f"{quant_mode.upper()} b12x hidden_size ({k}) must be a multiple of 128 "
            "and positive; hidden-dimension tails are not supported."
        )
    sf_vec_size, _ = _sf_params_for_quant_mode(quant_mode)
    if n <= 0 or n % sf_vec_size:
        raise ValueError(
            f"{quant_mode.upper()} b12x intermediate_size ({n}) must be a positive "
            f"multiple of the quantization block size ({sf_vec_size})."
        )


def _prepare_weight_views(
    *,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    quant_mode: str,
    activation: str,
) -> _WeightViews:
    """Shared wrapper/functional preparation from logical, unpadded operands.

    Per-expert activation scales are runtime inputs, not channel data: they
    never participate in intermediate padding or its weight-only caches.
    """
    is_gated = is_gated_activation(activation)
    n_pad = _align_up(n, _LEVEL_TILE_N)
    weight_E = w1_fp4.shape[0]
    pad_args = (w1_fp4, w1_blockscale, w2_fp4, w2_blockscale, None)
    static_override = None
    if static_needs_256_extent(n):
        padded = _pad_intermediate_to_tile(
            *pad_args, n, _STATIC_MIN_EXTENT, k, weight_E, is_gated, quant_mode
        )
        static_override = (padded[0], padded[1], padded[2], padded[3], padded[5])

    source_scales = static_source_scales(n, is_gated, quant_mode)
    legacy_builder = padded_scale_builder = None
    if n_pad != n:

        def legacy_builder():
            padded = _pad_intermediate_to_tile(
                *pad_args, n, _LEVEL_TILE_N, k, weight_E, is_gated, quant_mode
            )
            return padded[0], padded[2]

        if source_scales:

            def padded_scale_builder():
                padded = _pad_intermediate_to_tile(
                    *pad_args,
                    n,
                    _LEVEL_TILE_N,
                    k,
                    weight_E,
                    is_gated,
                    quant_mode,
                    pad_fp4=False,
                )
                sf_vec, _ = _sf_params_for_quant_mode(quant_mode)
                return (
                    convert_sf_from_mma_layout(
                        padded[1],
                        m=2 * n_pad,
                        k=k,
                        num_groups=weight_E,
                        sf_vec_size=sf_vec,
                    ).contiguous(),
                    convert_sf_from_mma_layout(
                        padded[3],
                        m=k,
                        k=n_pad,
                        num_groups=weight_E,
                        sf_vec_size=sf_vec,
                    ).contiguous(),
                )
        else:
            padded = _pad_intermediate_to_tile(
                *pad_args,
                n,
                _LEVEL_TILE_N,
                k,
                weight_E,
                is_gated,
                quant_mode,
                pad_fp4=False,
            )
            w1_blockscale, w2_blockscale = padded[1], padded[3]

    return _get_weight_views(
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w1_alphas=w1_alphas,
        w2_alphas=w2_alphas,
        n=n_pad,
        k=k,
        quant_mode=quant_mode,
        intermediate_size=n,
        branches=2 if is_gated else 1,
        legacy_builder=legacy_builder,
        static_override=static_override,
        source_scales=source_scales,
        padded_scale_builder=padded_scale_builder,
    )


def _get_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
    intermediate_size: int | None = None,
    branches: int | None = None,
    legacy_builder=None,
    static_override: Tuple | None = None,
    source_scales: bool = False,
    padded_scale_builder: Optional[
        Callable[[], Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
) -> _WeightViews:
    """Create permuted weight views for the MoE kernels.

    ``w1_fp4`` / ``w2_fp4`` may be the tile-padded copies (legacy views built
    eagerly, as before) or the unpadded weights: then ``branches`` (2 for gated
    activations, 1 otherwise) fixes the padded row count the block scales
    describe and ``legacy_builder`` produces the padded copies lazily
    (``_WeightViews.ensure_legacy``), so a static-only caller never pins them.

    ``n`` is the tile-aligned intermediate size that ``w1_fp4``/``w2_fp4``
    carry ([2*n, k//2, E] concatenated w13 for dynamic/micro).  The static
    kernel also gets branch-major views ([n_true, k//2, 2E] gated w13,
    [k, n_true//2, E] down) built from the unpadded weights when the true
    ``intermediate_size`` differs, so TMA zero-fills a partial tail tile
    instead of streaming physically padded zeros.

    Two shape classes cannot use the true-extent views and fall back to
    padded operands for the static kernel (the dynamic branch-major extent
    follows ``intermediate_size``):

    * ``intermediate_size % 32 != 0``: the packed down row is not a 16-byte
      TMA stride, so the static and dynamic branch-major views are built from
      the 128-padded legacy copies (materialized here) at the aligned extent.
    * a single N128 slice (``intermediate_size <= 128``): the retained2 static
      kernel needs two slices, so the caller passes ``static_override`` =
      ``(w1_fp4_256, w1_sf_256, w2_fp4_256, w2_sf_256, 256)`` (the 256-padded
      copies from ``_pad_intermediate_to_tile``) and the static kernel gets its
      own scale storages; dynamic and micro keep the 128-aligned operands.
    """
    activation_precision = _normalize_activation_precision(activation_precision)
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    tile_n = _level_tile_n(activation_precision)
    # The kernel splits w13 into gate/up halves by tile index. This only works
    # when the boundary between halves lands on a tile-aligned column.
    if n % tile_n != 0:
        raise ValueError(
            f"intermediate_size ({n}) must be a multiple of {tile_n} "
            f"for the SM120 MoE kernel's gate/up tile split."
        )

    key = (
        activation_precision,
        quant_mode,
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
        bool(source_scales),
    )
    n_true = n if intermediate_size is None else int(intermediate_size)
    if branches is None:
        branches = w1_fp4.shape[1] // (n if w1_fp4.shape[1] % n == 0 else n_true)
    if branches not in (1, 2):
        raise ValueError(f"w13 must carry one or two branches, got {branches}")
    padded_fp4_present = w1_fp4.shape[1] == branches * n and w2_fp4.shape[2] == n // 2
    if not padded_fp4_present and n != n_true and legacy_builder is None:
        raise ValueError(
            "unpadded w13/down weights need a legacy_builder for the tile-padded "
            "views the dynamic and micro kernels consume"
        )
    cached = _WEIGHT_CACHE.get(key)
    if cached is None:
        _refuse_during_capture("the converted block-scale views")
        # Cache the fresh buffers (scale factors + fp32 alphas).
        # Rows / reduction extent the scale layout describes: the padded n (2*n
        # gated, n otherwise), or the true extent when the static kernel reads
        # the source layout (the 6D MMA view is a permutation of the caller's
        # 2D storage, so this is a view, not a copy).
        scale_n = n_true if source_scales else n
        w1_rows = branches * scale_n
        cached = (
            convert_sf_from_mma_layout(
                w1_blockscale,
                m=w1_rows,
                k=k,
                num_groups=w1_fp4.shape[0],
                sf_vec_size=sf_vec_size,
            ).contiguous(),
            convert_sf_from_mma_layout(
                w2_blockscale,
                m=k,
                k=scale_n,
                num_groups=w2_fp4.shape[0],
                sf_vec_size=sf_vec_size,
            ).contiguous(),
            w1_alphas.contiguous().to(torch.float32),
            w2_alphas.contiguous().to(torch.float32),
        )
        _WEIGHT_CACHE[key] = cached
        _register_cache_eviction(
            _WEIGHT_CACHE,
            key,
            w1_fp4,
            w1_blockscale,
            w1_alphas,
            w2_fp4,
            w2_blockscale,
            w2_alphas,
        )
    w13_sf_contiguous, down_sf_contiguous, w1_alpha, w2_alpha = cached

    # Permute [E, w1_rows, k//2] -> [w1_rows, k//2, E] (view, no copy).
    if padded_fp4_present:
        w13 = w1_fp4.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
        down = w2_fp4.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
    else:
        w13 = down = None
    # Static kernel: one branch per batch index at the true extent (views).
    if w1_fp4.shape[1] != branches * n_true or w2_fp4.shape[2] != n_true // 2:
        raise ValueError(
            "intermediate_size does not match the unpadded weights: "
            f"n_true={n_true}, w1 rows={w1_fp4.shape[1]}, "
            f"w2 packed cols={w2_fp4.shape[2]}"
        )
    static_w13 = w1_fp4.reshape(
        w1_fp4.shape[0] * branches, n_true, w1_fp4.shape[2]
    ).permute(1, 2, 0)
    static_down = w2_fp4.permute(1, 2, 0)
    static_sf: Tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
    family: Tuple = (None, None)
    # Branch-major views for the gated dynamic kernel (and the static kernel
    # unless overridden): true extent when TMA-legal, else the 128-padded
    # copies at the aligned extent (I % 32 != 0 -> non-16-byte down stride).
    if not true_extent_views_supported(n_true):
        if w13 is None:
            if legacy_builder is None:
                raise ValueError(
                    f"intermediate_size {n_true} is not a multiple of "
                    f"{_TRUE_EXTENT_ALIGN}; the tile-padded weights are required"
                )
            if torch.cuda.is_current_stream_capturing():
                raise ValueError(
                    "the tile-padded weight copies are materialized on first "
                    "use and cannot be created during CUDA graph capture; run "
                    "one eager warm-up call before capturing the graph"
                )
            w1_p, w2_p = legacy_builder()
            w13 = w1_p.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
            down = w2_p.permute(1, 2, 0).view(torch.float4_e2m1fn_x2)
            w1_fp4, w2_fp4 = w1_p, w2_p
            padded_fp4_present = True
        static_w13 = w1_fp4.reshape(
            w1_fp4.shape[0] * branches, n, w1_fp4.shape[2]
        ).permute(1, 2, 0)
        static_down = w2_fp4.permute(1, 2, 0)
        n_true = n
    branch_major_w13, branch_major_down = static_w13, static_down
    static_extent = n_true
    if static_override is not None:
        # Single-slice shape: 256-aligned copies for the whole static family.
        w1_s, w1_sf_s, w2_s, w2_sf_s, n_s = static_override
        static_w13 = w1_s.reshape(
            w1_s.shape[0] * branches, int(n_s), w1_s.shape[2]
        ).permute(1, 2, 0)
        static_down = w2_s.permute(1, 2, 0)
        static_extent = int(n_s)
        sf_key = (
            "static_sf",
            key,
            w1_sf_s.data_ptr(),
            w2_sf_s.data_ptr(),
            static_extent,
        )
        cached_sf = _WEIGHT_CACHE.get(sf_key)
        if cached_sf is None:
            _refuse_during_capture("the static family's 256-aligned scale views")
            cached_sf = (
                convert_sf_from_mma_layout(
                    w1_sf_s,
                    m=branches * static_extent,
                    k=k,
                    num_groups=w1_s.shape[0],
                    sf_vec_size=sf_vec_size,
                ).contiguous(),
                convert_sf_from_mma_layout(
                    w2_sf_s,
                    m=k,
                    k=static_extent,
                    num_groups=w2_s.shape[0],
                    sf_vec_size=sf_vec_size,
                ).contiguous(),
            )
            _WEIGHT_CACHE[sf_key] = cached_sf
            _register_cache_eviction(_WEIGHT_CACHE, sf_key, w1_sf_s, w2_sf_s)
        static_sf = cached_sf
        family = (w1_s, w2_s)
    elif static_needs_256_extent(n_true):
        raise ValueError(
            f"intermediate_size {n_true} spans a single N128 slice; the static "
            "kernel needs the 256-aligned operands (static_override)"
        )
    return _WeightViews(
        w13_fp4=w13,
        down_fp4=down,
        _legacy_builder=None if padded_fp4_present else legacy_builder,
        static_w13_fp4=static_w13.view(torch.float4_e2m1fn_x2),
        static_down_fp4=static_down.view(torch.float4_e2m1fn_x2),
        intermediate_size=n_true,
        static_intermediate_size=static_extent,
        static_w13_sf_storage=static_sf[0],
        static_down_sf_storage=static_sf[1],
        branch_major_w13_fp4=branch_major_w13.view(torch.float4_e2m1fn_x2),
        branch_major_down_fp4=branch_major_down.view(torch.float4_e2m1fn_x2),
        static_family_w1_storage=family[0],
        static_family_w2_storage=family[1],
        sfb_w13_ptr=make_ptr(
            sf_dtype,
            w13_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        sfb_down_ptr=make_ptr(
            sf_dtype,
            down_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        w1_alpha=w1_alpha,
        w2_alpha=w2_alpha,
        w1_storage=w1_fp4,
        w1_scale_storage=w13_sf_contiguous,
        w2_storage=w2_fp4,
        w2_scale_storage=down_sf_contiguous,
        _w13_sf_storage=w13_sf_contiguous,
        _down_sf_storage=down_sf_contiguous,
        source_scales=bool(source_scales),
        _padded_scale_builder=padded_scale_builder,
    )


# ---------------------------------------------------------------------------
# Kernel compilation cache
#
# The three kernels below are compiled through the shared on-disk CuTe-DSL
# cache (#3874, #4029; docs/design_docs/cute_dsl_kernel_cache.md), so a fresh
# process JITLinks an exported ``.o`` instead of re-running the MLIR pipeline.
# The in-process dicts stay as the level-1 memoization the design describes.
# ---------------------------------------------------------------------------
_CUTE_DSL_MODULE = "b12x_moe"


def _kernel_source_files() -> Tuple[str, ...]:
    """Source files whose content invalidates the on-disk kernel cache.

    Every module contributing device code to the three kernels compiled here:
    the kernel bodies, the shared activation and FP4 device helpers, and the
    SM120 layout builders and block-scaled mainloop they are built from.
    """
    from flashinfer.cute_dsl import fp4_common
    from flashinfer.cute_dsl import utils as cute_dsl_utils
    from flashinfer.gemm.kernels import dense_blockscaled_gemm_sm120_b12x

    from ._moe_dynamic import gated as moe_dynamic_gated
    from ._moe_dynamic import generic as moe_dynamic_generic
    from . import (
        moe_activation,
        moe_dynamic_kernel,
        moe_micro_kernel,
        moe_static_kernel,
    )

    return (
        __file__,
        moe_activation.__file__,
        moe_static_kernel.__file__,
        moe_micro_kernel.__file__,
        moe_dynamic_kernel.__file__,
        moe_dynamic_gated.__file__,
        moe_dynamic_generic.__file__,
        cute_dsl_utils.__file__,
        fp4_common.__file__,
        dense_blockscaled_gemm_sm120_b12x.__file__,
    )


def _disk_kernel_name(prefix: str, cache_key: Tuple) -> str:
    """On-disk specialization name for an in-process kernel cache key.

    The name is the *sole* per-kernel cache key — the module ``meta.json``
    guards only module-wide facts (arch, DSL stack, source hashes) — so it has
    to be injective in every codegen parameter. It is therefore derived from
    the very tuple that keys the in-process cache: a readable shape prefix for
    humans browsing ``cached_ops/``, plus a digest of the exact tuple.

    The digest, rather than a formatted field list, is what makes the mapping
    injective: the keys contain floats and ``None`` (``swiglu_alpha`` /
    ``swiglu_beta`` / ``swiglu_limit``) whose textual forms would collide once
    sanitized into a filename (``1.5`` and ``-1.5`` both sanitize to ``1_5``).
    """
    digest = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _static_kernel_cache_key(
    *,
    activation_precision: str,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    weight_n: int,
    route_rows: int,
    num_topk: int,
    max_rows: int,
    mac: int,
    mma_tiler_mn: Tuple[int, int],
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    activation: str,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float | None,
    merged_groups: bool = False,
    deferred_init: bool = False,
    source_scales: bool = False,
) -> Tuple:
    """The static kernel's cache key: every parameter affecting its codegen.

    Single source of truth for both cache levels — the in-process dict and,
    through :func:`_disk_kernel_name`, the on-disk artifact name.
    """
    return (
        "static",
        activation_precision,
        quant_mode,
        state_E,
        weight_E,
        m,
        k,
        n,
        weight_n,
        route_rows,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        merged_groups,
        deferred_init,
        source_scales,
    )


def _micro_kernel_cache_key(
    *,
    quant_mode: str,
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    mac: int,
    mma_tiler_mn: Tuple[int, int],
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    share_input_across_experts: bool,
    share_expert_scales: bool,
    single_token: bool,
    activation: str,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float | None,
) -> Tuple:
    """The micro kernel's cache key (see :func:`_static_kernel_cache_key`)."""
    return (
        "micro",
        quant_mode,
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        share_input_across_experts,
        share_expert_scales,
        single_token,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
    )


def _dynamic_kernel_cache_key(
    *,
    activation_precision: str,
    quant_mode: str,
    E: int,
    k: int,
    n: int,
    num_topk: int,
    mac: int,
    mma_tiler_mn: Tuple[int, int],
    topk_ids_dtype: torch.dtype,
    input_scales_are_reciprocal: bool,
    fast_math: bool,
    activation: str,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float | None,
    share_input_across_experts: bool,
    branch_major_extent: int | None = None,
) -> Tuple:
    """The dynamic kernel's cache key (see :func:`_static_kernel_cache_key`).

    ``branch_major_extent`` is the true intermediate extent when the
    branch-paired gated kernel streams the branch-major weight views (its
    artifact differs from the legacy concatenated one), else ``None``.

    Deliberately free of ``m`` / ``max_rows``: the dynamic kernel takes its
    runtime-shaped operands as pointers, so one artifact serves every batch
    size.
    """
    return (
        "dynamic",
        activation_precision,
        quant_mode,
        E,
        k,
        n,
        num_topk,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        share_input_across_experts,
        branch_major_extent,
    )


_STATIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_static_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    mac_override: int | None = None,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
    weight_n: int | None = None,
    route_rows: int | None = None,
    merged_groups: bool = False,
    deferred_init: bool = False,
    source_scales: bool = False,
):
    """Compile (or retrieve cached) the SM120 static MoE kernel.

    ``n`` is the tile-aligned intermediate size (scratch geometry, tile
    counts); ``weight_n`` is the true weight extent the branch-major TMA
    views carry (defaults to ``n``).  ``max_rows`` is the per-slot row stride
    of the packed activation planes (token_map.shape[1]); ``route_rows`` is
    the routed-row capacity that sizes the route-output scratch (defaults to
    ``max_rows``).
    """
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 static compiler"
        )
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    sm_count = get_num_sm(torch.device("cuda"))
    mac = (
        mac_override
        if mac_override is not None
        else min(get_max_active_clusters(1), sm_count)
    )

    weight_n = n if weight_n is None else int(weight_n)
    route_rows = max_rows if route_rows is None else int(route_rows)

    mma_tiler_mn = (128, 128)
    if activation_precision == "fp4" and num_topk > 1:
        mma_tiler_mn = _COMPACT_MMA_TILER_MN

    cache_key = _static_kernel_cache_key(
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        state_E=state_E,
        weight_E=weight_E,
        m=m,
        k=k,
        n=n,
        weight_n=weight_n,
        route_rows=route_rows,
        num_topk=num_topk,
        max_rows=max_rows,
        mac=mac,
        mma_tiler_mn=mma_tiler_mn,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        merged_groups=merged_groups,
        deferred_init=deferred_init,
        source_scales=source_scales,
    )
    cached = _STATIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _refuse_during_capture("the static MoE kernel (compile / load)")

    ab_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    output_tile_count_n = max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1])
    kernel: Any = MoEStaticKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=output_tile_count_n,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        merged_groups=merged_groups,
        deferred_init=deferred_init,
        source_scales=source_scales,
    )

    is_gated = is_gated_activation(activation)
    # Branch-major weights: N spans one branch at the true extent and the
    # batch index spans (branches * E).
    w13_batch = (2 if is_gated else 1) * weight_E

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)

    # Build fake tensors for compilation
    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (m * num_topk,),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (m * num_topk,),
        assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (max_rows, k, state_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * max_rows * (k // 2),),
        assumed_align=16,
    )
    route_output_scratch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (route_rows * (1 if merged_groups else _static_retained_groups(n)) * k,),
        assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * rows_pad_k * cols_pad_k,),
        assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    route_state_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (8,),
        assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (weight_n, k, w13_batch),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (k, weight_n, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    # Chunk map + claim counter scale with the routed-row capacity, not the
    # per-slot stride.
    virt_route_scratch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E * (1 + (route_rows + 31) // 32) + 8,),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        _disk_kernel_name(
            f"static_m{m}_k{k}_n{n}_t{num_topk}_r{max_rows}_c{route_rows}"
            f"{'_mg' if merged_groups else ''}{'_di' if deferred_init else ''}{'_ss' if source_scales else ''}",
            cache_key,
        ),
        lambda: cute.compile(
            kernel,
            a_input_fake,
            topk_ids_fake,
            topk_weights_fake,
            packed_a_fake,
            sfa_fake,
            packed_a_storage_fake,
            route_output_scratch_fake,
            scale_storage_fake,
            barrier_count_fake,
            barrier_epoch_fake,
            route_state_fake,
            b_w13_fake,
            sfb_w13_fake,
            b_down_fake,
            sfb_down_fake,
            row_counts_fake,
            active_expert_count_fake,
            weight_expert_ids_fake,
            global_to_local_expert_fake,
            virt_route_scratch_fake,
            input_gs_fake,
            alpha_fake,
            down_alpha_fake,
            global_scale_fake,
            scatter_fake,
            token_map_fake,
            token_weights_fake,
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=_kernel_source_files(),
    )

    result = (compiled, mac)
    _STATIC_KERNEL_CACHE[cache_key] = result
    return result


_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_micro_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    single_token: bool = False,
    mac_override: int | None = None,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    quant_mode: str = "nvfp4",
):
    """Compile (or retrieve cached) the SM120 micro MoE kernel."""
    quant_mode = _normalize_quant_mode(quant_mode)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    sm_count = get_num_sm(torch.device("cuda"))
    mac = (
        mac_override
        if mac_override is not None
        else min(get_max_active_clusters(1), sm_count)
    )

    mma_tiler_mn = _COMPACT_MMA_TILER_MN

    cache_key = _micro_kernel_cache_key(
        quant_mode=quant_mode,
        state_E=state_E,
        weight_E=weight_E,
        m=m,
        k=k,
        n=n,
        num_topk=num_topk,
        max_rows=max_rows,
        mac=mac,
        mma_tiler_mn=mma_tiler_mn,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )
    cached = _MICRO_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _refuse_during_capture("the micro MoE kernel (compile / load)")

    ab_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel = MoEMicroKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1]),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
    )

    is_gated = is_gated_activation(activation)
    w1_rows = (2 if is_gated else 1) * n

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)

    # Build fake tensors for compilation (identical to static kernel)
    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (m * num_topk,),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (m * num_topk,),
        assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (max_rows, k, state_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * max_rows * (k // 2),),
        assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * rows_pad_k * cols_pad_k,),
        assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (w1_rows, k, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (k, n, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        _disk_kernel_name(f"micro_m{m}_k{k}_n{n}_t{num_topk}_r{max_rows}", cache_key),
        lambda: cute.compile(
            kernel,
            a_input_fake,
            topk_ids_fake,
            topk_weights_fake,
            packed_a_fake,
            sfa_fake,
            packed_a_storage_fake,
            scale_storage_fake,
            barrier_count_fake,
            barrier_epoch_fake,
            b_w13_fake,
            sfb_w13_fake,
            b_down_fake,
            sfb_down_fake,
            row_counts_fake,
            active_expert_count_fake,
            weight_expert_ids_fake,
            global_to_local_expert_fake,
            input_gs_fake,
            alpha_fake,
            down_alpha_fake,
            global_scale_fake,
            scatter_fake,
            token_map_fake,
            token_weights_fake,
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=_kernel_source_files(),
    )

    result = (compiled, mac)
    _MICRO_KERNEL_CACHE[cache_key] = result
    return result


# The launch cache skips the per-launch build/configure; the kernel cache
# dedupes compiles across keys that configure to the same artifact
# (m=2..8 differ only in grid_x).
_DIRECT_MICRO_LAUNCH_CACHE: Dict[Tuple, Tuple] = {}
_DIRECT_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_direct_micro_kernel(
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    fast_math: bool = True,
    share_input_across_experts: bool = False,
    share_expert_scales: bool = False,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    device: torch.device | None = None,
):
    """Compile (or retrieve cached) the SM120 direct micro MoE kernel.

    Returns (compiled, grid_x, accepts_block_dim).
    """
    if activation != SWIGLUOAI_UNINTERLEAVE:
        # The kernel constructor only accepts configurable swiglu parameters
        # for swigluoai; other activations use its normalized defaults
        # (accept-and-ignore, matching the MMA kernels).
        swiglu_alpha = None
        swiglu_beta = None
        swiglu_limit = None
    launch_key = (
        weight_E,
        m,
        k,
        n,
        num_topk,
        topk_ids_dtype,
        fast_math,
        share_input_across_experts,
        share_expert_scales,
        activation,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        str(_canonical_cuda_device(device)) if device is not None else None,
    )
    cached = _DIRECT_MICRO_LAUNCH_CACHE.get(launch_key)
    if cached is not None:
        return cached
    _refuse_during_capture("the direct micro MoE kernel (compile / load)")
    kernel = build_direct_micro_kernel(
        weight_E,
        m,
        k,
        n,
        num_topk,
        activation=activation,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=m == 1,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        device=device,
    )
    compile_key = ("direct_micro", kernel.__cache_key__, topk_ids_dtype)
    entry = _DIRECT_MICRO_KERNEL_CACHE.get(compile_key)
    if entry is None:
        compiled = compile_direct_micro_kernel(kernel, topk_ids_dtype=topk_ids_dtype)
        # Register pressure can cap the launchable CTA below the fused body's
        # 512 threads; probe once per compiled kernel.
        accepts = compiled_direct_micro_accepts_block_dim(
            compiled, kernel.launch_block_dim
        )
        entry = (compiled, accepts)
        _DIRECT_MICRO_KERNEL_CACHE[compile_key] = entry
    compiled, accepts = entry
    cached = (compiled, kernel.grid_x, accepts)
    _DIRECT_MICRO_LAUNCH_CACHE[launch_key] = cached
    return cached


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def _expand_to_experts(t: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Broadcast a scalar or [1] tensor to [num_experts], always fp32.

    Both branches must cast: the kernels are compiled against fp32 fake
    tensors for every per-expert scale.
    """
    if t.numel() == 1:
        return t.to(torch.float32).expand(num_experts).contiguous()
    if t.ndim != 1 or t.numel() != num_experts:
        raise ValueError(
            f"per-expert scale must be scalar or [{num_experts}], got {tuple(t.shape)}"
        )
    return t.contiguous().to(torch.float32)


def launch_sm120_static_moe(
    *,
    workspace: Sm120StaticMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
) -> torch.Tensor:
    """Launch the SM120 static, micro, or direct micro MoE kernel.

    The direct micro kernel takes tiny decode batches (m <= 8, routed_rows
    < 64) when it supports the shape, the MMA micro kernel takes the rest of
    its band (routed_rows <= 20-40), and the static kernel takes the rest.
    The MMA micro path runs a Triton pre-pass to compact routing IDs before
    launching; direct micro routes on global expert ids directly.
    """
    _check_memref_limit("scatter_output", scatter_output.numel())
    activation_precision = _normalize_activation_precision(activation_precision)
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 static launcher"
        )

    # Flatten routing tensors
    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)
    routed_rows = num_tokens * top_k

    # Capture whether input_gs was a single shared scalar BEFORE expansion:
    # the m=1 relu2 shared-input micro optimization only applies when every
    # expert sees the same FC1-input global scale.
    input_gs_is_shared = input_gs.numel() == 1
    down_input_scale_is_shared = down_input_scale.numel() == 1

    # Broadcast scalar scales to per-expert [E] tensors
    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    # Shared-scale flags let compact W4A4 micro match the ReLU2 single-token
    # specialization.
    share_input_across_experts = (
        activation == "relu2"
        and num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
    share_expert_scales = (
        activation == "relu2" and input_gs_is_shared and down_input_scale_is_shared
    )

    # Single-slice shapes: the static kernel and the direct CUDA-core micro
    # kernel consume the 256-aligned copies (_get_weight_views static_override);
    # the MMA micro kernel is correct and faster on the 128-aligned operands
    # and keeps them.
    family_n = n
    if weights.static_family_w1_storage is not None:
        family_n = int(weights.static_intermediate_size or n)
    # Direct micro takes its band before the MMA micro decision. It reads
    # weights by global expert id, so EP shapes keep the compact path.
    use_direct_micro = (
        quant_mode == "nvfp4"
        and workspace.state_E == num_experts
        and workspace.dm_barrier_count is not None
        and workspace.dm_barrier_count.numel() >= routed_rows + num_tokens * 16
        and num_tokens <= _MICRO_MAX_TOKENS
        and routed_rows < _DIRECT_MICRO_CUTOVER_PAIRS
        and family_n <= _DIRECT_MICRO_MAX_N
        and MoEDirectMicroKernel.is_supported(
            num_tokens, k, family_n, top_k, num_experts
        )
    )
    if _FORCED_BACKEND is not None:
        if _FORCED_BACKEND == "direct_micro":
            if quant_mode != "nvfp4":
                raise ValueError(
                    "forced direct_micro backend only supports quant_mode=nvfp4"
                )
            if workspace.dm_barrier_count is None or not (
                MoEDirectMicroKernel.is_supported(
                    num_tokens, k, family_n, top_k, num_experts
                )
            ):
                raise ValueError(
                    "forced direct_micro backend cannot run this shape "
                    f"(m={num_tokens}, k={k}, n={family_n}, top_k={top_k})"
                )
            if workspace.dm_barrier_count.numel() < routed_rows + num_tokens * 16:
                raise ValueError(
                    "forced direct_micro backend exceeds the workspace barrier "
                    f"capacity ({workspace.dm_barrier_count.numel()} slots < "
                    f"{routed_rows} routed rows + {num_tokens * 16})"
                )
            use_direct_micro = True
        else:
            use_direct_micro = False
    if use_direct_micro:
        compiled, grid_x, block_ok = _get_direct_micro_kernel(
            num_experts,
            num_tokens,
            k,
            family_n,
            top_k,
            topk_ids_dtype=flat_ids.dtype,
            fast_math=fast_math,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            device=a.device,
        )
        if not block_ok:
            if _FORCED_BACKEND == "direct_micro":
                raise RuntimeError("compiled direct micro MoE kernel cannot launch")
            use_direct_micro = False
    if use_direct_micro:
        # The kernel takes multiplier-form scales only; invert reciprocal
        # inputs into the persistent workspace planes (zeros stay zero,
        # matching the MMA kernels).
        if input_scales_are_reciprocal:
            workspace.dm_input_gs.copy_(
                torch.where(input_gs != 0, 1.0 / input_gs, input_gs)
            )
            workspace.dm_down_input_scale.copy_(
                torch.where(
                    down_input_scale != 0, 1.0 / down_input_scale, down_input_scale
                )
            )
            launch_gs = workspace.dm_input_gs
            launch_down = workspace.dm_down_input_scale
        else:
            launch_gs = input_gs
            launch_down = down_input_scale
        # The direct CUDA-core micro kernel indexes the tile-padded weight
        # storages (n aligned to the tile): materialize the legacy copies, or
        # take the static family's 256-aligned copies for single-slice shapes.
        if weights.static_family_w1_storage is not None:
            dm_w1, dm_w1_sf = (
                weights.static_family_w1_storage,
                weights.static_w13_sf_storage,
            )
            dm_w2, dm_w2_sf = (
                weights.static_family_w2_storage,
                weights.static_down_sf_storage,
            )
        else:
            weights.ensure_legacy()
            dm_w1_sf, dm_w2_sf = weights.padded_scales()
            dm_w1, dm_w2 = weights.w1_storage, weights.w2_storage
        MoEDirectMicroKernel.launch(
            compiled,
            x=a,
            w1_fp4=dm_w1,
            w1_blockscale=dm_w1_sf,
            w1_alphas=weights.w1_alpha,
            a1_gscale=launch_gs,
            a2_gscale=launch_down,
            inter_fp32=workspace.dm_intermediate,
            w2_fp4=dm_w2,
            w2_blockscale=dm_w2_sf,
            w2_alphas=weights.w2_alpha,
            topk_ids=flat_ids,
            topk_weights=flat_weights,
            out=scatter_output,
            barrier_count=workspace.dm_barrier_count,
            barrier_epoch=workspace.dm_barrier_epoch,
            m=num_tokens,
            grid_x=grid_x,
        )
        return scatter_output

    # Decide micro vs static
    micro_cutover = _MICRO_COMPACT_CUTOVER_PAIRS
    if top_k > 1:
        micro_cutover = _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK
    use_micro = (
        activation_precision == "fp4"
        and num_tokens <= _MICRO_MAX_TOKENS
        and routed_rows <= micro_cutover
    )
    if _FORCED_BACKEND is not None:
        if _FORCED_BACKEND == "micro":
            # Forced mode raises on correctness violations, never falls back.
            if num_tokens > _MICRO_MAX_TOKENS:
                raise ValueError(
                    f"forced micro backend supports at most {_MICRO_MAX_TOKENS} "
                    f"tokens (got {num_tokens})"
                )
            if flat_ids.numel() > workspace.compact_topk_ids.numel():
                raise ValueError(
                    "forced micro backend exceeds the workspace compact-id "
                    f"capacity ({workspace.compact_topk_ids.numel()} < "
                    f"{flat_ids.numel()})"
                )
            use_micro = True
        else:
            use_micro = False

    sm_count = get_num_sm(torch.device("cuda"))
    base_mac = min(get_max_active_clusters(1), sm_count)
    tuned_static_mac = _lookup_mac_ladder(_STATIC_MAC_LADDER, routed_rows)
    static_mac = min(tuned_static_mac or base_mac, base_mac)
    if activation_precision == "fp4" and not use_micro and routed_rows < 40:
        static_mac = min(static_mac, 64)

    if use_micro:
        assert flat_ids.numel() <= workspace.compact_topk_ids.numel(), (
            f"compact_topk_ids buffer too small: "
            f"{workspace.compact_topk_ids.numel()} < {flat_ids.numel()}"
        )
        # Single-token ReLU2 is non-gated, so the micro kernel can launch on
        # the routed expert ids directly. Gated SiLU still goes through the
        # compact id buffer so the kernel can map compact launch ids back to
        # the physical gate/up weight experts.
        if num_tokens == 1 and activation == "relu2":
            launch_ids = flat_ids
        elif num_tokens == 1:
            compact_ids = workspace.compact_topk_ids[: flat_ids.numel()]
            compact_ids.copy_(
                torch.arange(
                    flat_ids.numel(),
                    device=flat_ids.device,
                    dtype=torch.int32,
                )
            )
            workspace.weight_expert_ids[: flat_ids.numel()].copy_(
                flat_ids.to(torch.int32)
            )
            workspace.micro_active_expert_count.fill_(flat_ids.numel())
            launch_ids = compact_ids
        else:
            compact_ids = workspace.compact_topk_ids[: flat_ids.numel()]
            from .triton_compact import compact_topk_ids as _triton_compact_topk_ids

            _triton_compact_topk_ids(
                flat_ids,
                compact_ids,
                workspace.weight_expert_ids,
                workspace.micro_active_expert_count,
            )
            launch_ids = compact_ids
        # Select micro MAC: min of tuned ladder, work tiles, and hardware limit.
        micro_work_tiles = max(1, routed_rows * max(1, (n + 128 - 1) // 128))
        tuned_mac = _lookup_mac_ladder(_MICRO_MAC_LADDER, routed_rows)
        micro_mac = min(tuned_mac or base_mac, micro_work_tiles, base_mac)
        compiled, mac = _get_micro_kernel(
            # Physical compact-expert slot count (state_E + virtual-split
            # slots); fake tensor extents must match the allocated arrays.
            int(workspace.row_counts.shape[0]),
            num_experts,
            num_tokens,
            k,
            n,
            top_k,
            # Kernels take the per-slot row stride (token_map.shape[1]), not
            # the routed-row capacity.
            int(workspace.token_map.shape[1]),
            topk_ids_dtype=launch_ids.dtype,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=num_tokens == 1,
            mac_override=micro_mac,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            quant_mode=quant_mode,
        )
        route_output_scratch_args: Tuple[Any, ...] = ()
        virt_scratch_args: Tuple[Any, ...] = ()
    else:
        static_n = weights.static_intermediate_size or weights.intermediate_size or n
        static_n = max(n, _align_up(int(static_n), _LEVEL_TILE_N))
        # top-k=1 uses M128 in _get_static_kernel: no unused A/SFA half
        # remains for the retained slices. Only the M64 path can merge.
        merged_groups = top_k > 1 and _static_merged_groups(
            static_n, num_tokens * top_k
        )
        deferred_init = _static_deferred_init(num_tokens * top_k)
        compiled, mac = _get_static_kernel(
            int(workspace.row_counts.shape[0]),
            num_experts,
            num_tokens,
            k,
            static_n,
            top_k,
            int(workspace.token_map.shape[1]),
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            mac_override=static_mac,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
            weight_n=weights.static_intermediate_size or weights.intermediate_size,
            route_rows=workspace.max_rows,
            merged_groups=merged_groups,
            deferred_init=deferred_init,
            source_scales=bool(weights.source_scales),
        )
        launch_ids = flat_ids
        route_scratch_flat = workspace.route_output_scratch.view(-1)
        if merged_groups:
            # The merged schedule writes one [route, K] slot per route; the
            # workspace keeps the retained2 [route, groups, K] allocation.
            route_scratch_flat = route_scratch_flat[: workspace.max_rows * k]
        route_output_scratch_args = (route_scratch_flat,)
        virt_scratch_args = (workspace.virt_route_scratch,)

    # Pointer arguments must be passed as raw ints (data_ptr()) at runtime.
    # No stream argument: the kernels compile against
    # ``make_fake_stream(use_tvm_ffi_env_stream=True)``, so TVM-FFI supplies
    # the caller's current stream and the parameter is absent from the
    # compiled signature.
    # Micro kernels index the tile-aligned [2n, k//2, E] view; the static
    # kernel streams the branch-major views at the true intermediate extent.
    w13_sf_arg, down_sf_arg = weights._w13_sf_storage, weights._down_sf_storage
    if use_micro:
        w13_sf_arg, down_sf_arg = weights.padded_scales()
    if use_micro or weights.static_w13_fp4 is None:
        weights.ensure_legacy()
        w13_arg, down_arg = weights.w13_fp4, weights.down_fp4
    else:
        w13_arg, down_arg = weights.static_w13_fp4, weights.static_down_fp4
        if weights.static_w13_sf_storage is not None:
            w13_sf_arg = weights.static_w13_sf_storage
            down_sf_arg = weights.static_down_sf_storage
    runtime_args: Tuple[Any, ...] = (
        a,
        launch_ids,
        flat_weights,
        workspace.packed_a_view,
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat,
        *route_output_scratch_args,
        workspace.scale_flat,
        workspace.barrier_count,
        workspace.barrier_epoch,
        # The static kernel and its finalize share the clean-state marker; the
        # micro kernel keeps its own counter copies (its signature has no marker).
        *(() if use_micro else (workspace.route_state,)),
        w13_arg,
        w13_sf_arg.data_ptr(),
        down_arg,
        down_sf_arg.data_ptr(),
        workspace.micro_row_counts if use_micro else workspace.row_counts,
        workspace.micro_active_expert_count
        if use_micro
        else workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        *virt_scratch_args,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output,
        workspace.token_map,
        workspace.token_weights,
    )
    compiled(*runtime_args)

    return scatter_output


# ==========================================================================
# Dynamic backend
# ==========================================================================


def select_sm120_moe_backend(
    *,
    num_tokens: int,
    num_topk: int,
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    num_experts: int | None = None,
    intermediate_size: int | None = None,
    hidden_size: int | None = None,
    activation: str | None = None,
    capacity_tokens: int | None = None,
) -> str:
    """Pick static or dynamic backend based on routed-pair count (the cutover
    comes from the measured registry when the full key is known)."""
    mode = _normalize_quant_mode(quant_mode, activation_precision)
    if mode == "w4a16":
        return "w4a16"
    if _FORCED_BACKEND == "dynamic":
        return "dynamic"
    if _FORCED_BACKEND in ("static", "micro", "direct_micro"):
        # Both micro variants launch through the static workspace path.
        return "static"
    routed_rows = num_tokens * num_topk
    if routed_rows <= _get_static_compact_cutover_pairs(
        "fp4",
        quant_mode=mode,
        num_experts=num_experts,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        activation=activation,
        num_topk=num_topk,
        capacity_tokens=capacity_tokens,
    ):
        return "static"
    return "dynamic"


# ---------------------------------------------------------------------------
# Dynamic workspace
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class Sm120DynamicMoEWorkspace:
    """Scratch buffers for one SM120 dynamic MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    activation_precision: str
    quant_mode: str

    # Core buffers
    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor

    # Dynamic-specific
    routed_rows_capacity: int
    physical_tiles_capacity: int
    task_capacity: int
    # The M-tile the geometry above was sized for; launches must build the
    # kernel with the same tile.
    tile_m: int = _LEVEL_TILE_M
    expert_write_rows: torch.Tensor
    expert_tile_base: torch.Tensor
    pair_head: torch.Tensor
    task_head: torch.Tensor
    task_tail: torch.Tensor
    task_expert: torch.Tensor
    task_valid_rows: torch.Tensor

    # Views
    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None


def _dynamic_task_geometry(
    state_E: int,
    n: int,
    routed_rows: int,
    *,
    tile_m: int = _LEVEL_TILE_M,
    tile_n: int = _LEVEL_TILE_N,
):
    """Compute task queue dimensions from problem geometry.

    Each active expert can introduce at most one additional physical tile
    beyond the base count (due to per-expert tail padding). The task queue
    holds one entry per (m_tile, slice_group) pair — NOT multiplied by E.
    """
    routed_rows = max(1, routed_rows)
    base_m_tiles = _align_up(routed_rows, tile_m) // tile_m
    active_expert_upper_bound = min(state_E, routed_rows)
    max_m_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_cnt = max(1, (n + tile_n - 1) // tile_n)
    slice_groups = max(
        1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK
    )
    max_tasks = max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


def allocate_sm120_dynamic_workspace(
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation_precision: str = "fp4",
    activation: str = "silu",
    quant_mode: str = "nvfp4",
) -> Sm120DynamicMoEWorkspace:
    """Allocate workspace buffers for the SM120 dynamic MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "allocate_sm120_dynamic_workspace only supports quant_mode='nvfp4'; "
            "use allocate_sm120_moe_workspace(..., quant_mode='w4a16') for W4A16."
        )
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    tile_m = _select_dynamic_tile_m(routed_rows, state_E, activation)
    physical_tiles, _, max_tasks = _dynamic_task_geometry(
        state_E,
        n,
        routed_rows,
        tile_m=tile_m,
        tile_n=_level_tile_n(activation_precision),
    )
    rows_padded = physical_tiles * tile_m
    # The kernel addresses activation scales in 128-row SF atoms regardless of
    # tile_m, so the scale plane must cover the last partial atom.
    scale_rows = _align_up(rows_padded, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)
    _check_memref_limit("dynamic packed_input", rows_padded * (k // 2))
    _check_memref_limit("dynamic packed_input_scale", scale_rows * cols_pad_k)
    packed_input = torch.empty(1, rows_padded, k // 2, dtype=torch.uint8, device=device)

    workspace = Sm120DynamicMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=rows_padded,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        routed_rows_capacity=routed_rows,
        physical_tiles_capacity=physical_tiles,
        task_capacity=max_tasks,
        tile_m=tile_m,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(rows_padded, dtype=torch.int32, device=device),
        token_weights=torch.zeros(rows_padded, dtype=torch.float32, device=device),
        packed_input=packed_input,
        packed_input_scale=torch.empty(
            scale_rows, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        expert_write_rows=torch.zeros(state_E, dtype=torch.int32, device=device),
        expert_tile_base=torch.zeros(state_E + 1, dtype=torch.int32, device=device),
        pair_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_tail=torch.zeros(1, dtype=torch.int32, device=device),
        task_expert=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_valid_rows=torch.zeros(max_tasks, dtype=torch.int32, device=device),
    )

    # Finalize views
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return workspace


# ---------------------------------------------------------------------------
# Dynamic kernel compilation
# ---------------------------------------------------------------------------


class _DynamicMoELaunch:
    """Thin JIT wrapper that makes num_tokens and max_rows runtime Int32."""

    def __init__(
        self,
        kernel,
        k,
        num_topk,
        activation_precision: str = "fp4",
        sf_vec_size: int = _NVFP4_BLOCK_SIZE,
    ):
        activation_precision = _normalize_activation_precision(activation_precision)
        if activation_precision == "bf16":
            raise ValueError(
                "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic launcher wrapper"
            )
        self._kernel = kernel
        self._k = k
        self._packed_storage_cols = k // 2
        self._num_topk = num_topk
        self._cols_pad_k = _align_up(k // sf_vec_size, 4)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_expert_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1))
        )
        topk_ids = cute.make_tensor(
            topk_ids_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        topk_weights_t = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1)),
        )
        packed_a = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, self._k, 1), stride=(self._k, 1, rows_padded * self._k)
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout(
                (rows_padded * self._packed_storage_cols,), stride=(1,)
            ),
        )
        # Activation scales live in 128-row SF atoms; the plane is allocated
        # through the last partial atom even when rows_padded is not aligned.
        scale_rows = ((rows_padded + 127) // 128) * 128
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((scale_rows * self._cols_pad_k,), stride=(1,)),
        )
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        self._kernel(
            a_input,
            topk_ids,
            topk_weights_t,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            task_head,
            task_tail,
            task_expert,
            task_valid_rows,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights_t,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )


_DYNAMIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _dynamic_branch_major_extent(
    *,
    activation: str,
    activation_precision: str,
    quant_mode: str,
    tile_m: int,
    k: int,
    intermediate_size: int | None,
    num_topk: int,
    share_input_across_experts: bool,
) -> int | None:
    """True intermediate extent when the dynamic launch streams the
    branch-major weight views (``_WeightViews.static_*``): only the
    branch-paired gated NVFP4 kernel consumes them (up at batch 2e, gate at
    2e+1, N and the FC2 reduction at the true extent).  The generic dynamic
    kernel (relu2, MXFP4, oversize shapes) keeps the tile-padded concatenated
    legacy views, so ``None`` is returned for it.  Mirrors the eligibility
    decision of :class:`MoEDynamicKernel` exactly.
    """
    if intermediate_size is None:
        return None
    activation_precision = _normalize_activation_precision(activation_precision)
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, _ = _sf_params_for_quant_mode(quant_mode)
    share_input_across_experts = bool(
        share_input_across_experts
        and activation_precision == "fp4"
        and num_topk <= _MAX_SHARED_INPUT_TOPK
    )
    eligible = _can_use_gated_optimized_kernel(
        activation=activation,
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(tile_m, _level_tile_n(activation_precision)),
        hidden_size=k,
        intermediate_size=int(intermediate_size),
        num_topk=num_topk,
        share_input_across_experts=share_input_across_experts,
    )
    return int(intermediate_size) if eligible else None


def _get_dynamic_kernel(
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    share_input_across_experts: bool = False,
    tile_m: int = _LEVEL_TILE_M,
    quant_mode: str = "nvfp4",
    intermediate_size: int | None = None,
):
    """Compile (or retrieve cached) the SM120 dynamic MoE kernel.

    ``intermediate_size`` is the true (unpadded) intermediate extent; when the
    branch-paired gated kernel applies, the artifact is compiled against the
    branch-major ``[I_true, K, 2E]`` / ``[K, I_true, E]`` weight views.
    """
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic compiler"
        )
    # Both dynamic implementations reserve 32 route slots per token for the
    # shared-input fast path. Larger top-k values remain correct by using the
    # generic per-route producer instead.
    share_input_across_experts = bool(
        share_input_across_experts
        and activation_precision == "fp4"
        and num_topk <= _MAX_SHARED_INPUT_TOPK
    )
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    sf_vec_size, sf_dtype = _sf_params_for_quant_mode(quant_mode)
    sm_count = get_num_sm(torch.device("cuda"))
    base_mac = min(get_max_active_clusters(1), sm_count)
    tuned_mac = _lookup_mac_ladder(_DYNAMIC_MAC_LADDER, m * num_topk)
    mac = min(tuned_mac or base_mac, base_mac)
    # tile_m comes from the workspace's shared selection so the kernel's task
    # and scale indexing matches the allocated scratch geometry.
    mma_tiler_mn = (tile_m, _level_tile_n(activation_precision))
    n_true = n if intermediate_size is None else int(intermediate_size)
    branch_major_extent = _dynamic_branch_major_extent(
        activation=activation,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        tile_m=tile_m,
        k=k,
        intermediate_size=n_true,
        num_topk=num_topk,
        share_input_across_experts=share_input_across_experts,
    )

    cache_key = _dynamic_kernel_cache_key(
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        E=E,
        k=k,
        n=n,
        num_topk=num_topk,
        mac=mac,
        mma_tiler_mn=mma_tiler_mn,
        topk_ids_dtype=topk_ids_dtype,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        share_input_across_experts=share_input_across_experts,
        branch_major_extent=branch_major_extent,
    )
    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    _refuse_during_capture("the dynamic MoE kernel (compile / load)")

    is_gated = is_gated_activation(activation)
    w1_rows = (2 if is_gated else 1) * n

    scratch_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel: Any = MoEDynamicKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        share_input_across_experts=share_input_across_experts,
        hidden_size=k,
        intermediate_size=n_true,
        num_topk=num_topk,
    )
    launch = _DynamicMoELaunch(
        kernel,
        k=k,
        num_topk=num_topk,
        activation_precision=activation_precision,
        sf_vec_size=sf_vec_size,
    )

    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    # Runtime-shaped tensors passed as pointers
    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(
        topk_ids_cutlass_dtype,
        topk_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    packed_a_fake = make_ptr(
        scratch_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    scale_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )

    task_expert_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_valid_rows_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )

    if branch_major_extent is not None:
        # Branch-paired gated kernel: branch-major views at the true extent
        # (up at batch 2e, gate at 2e+1; FC2 reduces over I_true).
        w13_fake_shape = (branch_major_extent, k, 2 * E)
        down_fake_shape = (k, branch_major_extent, E)
    else:
        # Generic kernel: tile-padded concatenated legacy views.
        w13_fake_shape = (w1_rows, k, E)
        down_fake_shape = (k, n, E)
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        w13_fake_shape,
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        down_fake_shape,
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E + 1,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        _disk_kernel_name(
            f"dynamic_e{E}_k{k}_n{n}_t{num_topk}"
            + ("" if branch_major_extent is None else f"_i{branch_major_extent}"),
            cache_key,
        ),
        lambda: cute.compile(
            launch,
            a_input_fake,
            topk_ids_fake,
            topk_weights_fake,
            packed_a_fake,
            sfa_fake,
            packed_a_storage_fake,
            scale_storage_fake,
            barrier_count_fake,
            barrier_epoch_fake,
            pair_head_fake,
            task_head_fake,
            task_tail_fake,
            task_expert_fake,
            task_valid_rows_fake,
            b_w13_fake,
            sfb_w13_fake,
            b_down_fake,
            sfb_down_fake,
            row_counts_fake,
            expert_write_rows_fake,
            expert_tile_base_fake,
            input_gs_fake,
            alpha_fake,
            down_alpha_fake,
            global_scale_fake,
            scatter_fake,
            token_map_fake,
            token_weights_fake,
            1,
            1,
            1,
            1,  # runtime Int32 placeholders
            mac,
            stream_fake,
            options="--opt-level 2 --enable-tvm-ffi",
        ),
        extra_key_files=_kernel_source_files(),
    )

    result = (compiled, mac)
    _DYNAMIC_KERNEL_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Dynamic launch
# ---------------------------------------------------------------------------
def launch_sm120_dynamic_moe(
    *,
    workspace: Sm120DynamicMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str = "nvfp4",
) -> torch.Tensor:
    """Launch the SM120 dynamic MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic launcher"
        )
    _check_memref_limit("scatter_output", scatter_output.numel())
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)
    input_gs_is_shared = input_gs.numel() == 1

    # Broadcast scalar scales to per-expert [E] tensors
    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    compiled, mac = _get_dynamic_kernel(
        num_experts,
        num_tokens,
        k,
        n,
        top_k,
        workspace.max_rows,
        topk_ids_dtype=torch.int32,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        activation_precision=activation_precision,
        share_input_across_experts=input_gs_is_shared,
        tile_m=workspace.tile_m,
        quant_mode=quant_mode,
        intermediate_size=weights.intermediate_size,
    )
    branch_major_extent = _dynamic_branch_major_extent(
        activation=activation,
        activation_precision=activation_precision,
        quant_mode=quant_mode,
        tile_m=workspace.tile_m,
        k=k,
        intermediate_size=weights.intermediate_size,
        num_topk=top_k,
        share_input_across_experts=input_gs_is_shared,
    )

    # Dynamic kernel: runtime-shaped args are DataPointer (pass data_ptr()),
    # fixed-shape args are Tensor (pass torch tensor directly).  No stream
    # argument -- see the note in launch_sm120_static_moe.
    # The branch-paired gated kernel streams the branch-major views at the
    # true intermediate extent (no padded FP4 copies); the generic kernel
    # indexes the tile-padded concatenated legacy views.
    if branch_major_extent is not None and weights.branch_major_w13_fp4 is not None:
        w13_arg, down_arg = weights.branch_major_w13_fp4, weights.branch_major_down_fp4
    else:
        weights.ensure_legacy()
        w13_arg, down_arg = weights.w13_fp4, weights.down_fp4
    dynamic_w13_sf, dynamic_down_sf = weights.padded_scales()
    runtime_args: Tuple[Any, ...] = (
        a.data_ptr(),
        flat_ids.data_ptr(),
        flat_weights.data_ptr(),
        workspace.packed_a_view.data_ptr(),
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat.data_ptr(),
        workspace.scale_flat.data_ptr(),
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.pair_head,
        workspace.task_head,
        workspace.task_tail,
        workspace.task_expert.data_ptr(),
        workspace.task_valid_rows.data_ptr(),
        w13_arg,
        dynamic_w13_sf.data_ptr(),
        down_arg,
        dynamic_down_sf.data_ptr(),
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output.data_ptr(),
        workspace.token_map.data_ptr(),
        workspace.token_weights.data_ptr(),
        num_tokens,
        workspace.max_rows,
        workspace.physical_tiles_capacity * workspace.tile_m,
        workspace.task_capacity,
    )
    compiled(*runtime_args)

    return scatter_output


# ==========================================================================
# W4A16 route-packing implementation
# ==========================================================================
@dataclass(kw_only=True)
class Sm120W4A16MoEWorkspace:
    """Scratch buffers for the SM120 W4A16 MoE path."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device
    activation: str
    activation_precision: str
    quant_mode: str
    routed_rows_capacity: int
    route_num_experts: int

    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    fc1_c_tmp: torch.Tensor
    fc2_c_tmp: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_map: torch.Tensor | None = None


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _canonical_cuda_device(device: torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _w4a16_workspace_geometry(
    *,
    routed_rows: int,
    route_num_experts: int,
    num_topk: int,
    k: int,
    n: int,
    is_gated: bool,
    device: torch.device,
) -> tuple[int, int, int, int, int]:
    route_slots = 1
    route_blocks = 1
    fc1_c_tmp_elements = 1
    fc2_c_tmp_elements = 1
    fc1_cols = (2 if is_gated else 1) * int(n)
    sms = get_num_sm(device)
    # Size the route buffers for the power-of-2 capacity so route packing keeps
    # a single triton specialization across token counts.
    routed_rows_capacity = route_pack_numel_capacity(
        int(routed_rows), topk=int(num_topk)
    )
    for block_size in _W4A16_ALLOWED_ROUTED_SIZES:
        slots = max_packed_route_slots(
            routed_rows_capacity,
            int(block_size),
            int(route_num_experts),
        )
        blocks = (slots + int(block_size) - 1) // int(block_size)
        route_slots = max(route_slots, slots)
        route_blocks = max(route_blocks, blocks)
        fc1_c_tmp_elements = max(
            fc1_c_tmp_elements,
            packed_gemm_scratch_elements(
                size_n=fc1_cols,
                route_slots=slots,
                moe_block_size=int(block_size),
                sms=sms,
            ),
        )
        fc2_c_tmp_elements = max(
            fc2_c_tmp_elements,
            packed_gemm_scratch_elements(
                size_n=int(k),
                route_slots=slots,
                moe_block_size=int(block_size),
                sms=sms,
            ),
        )
    return (
        route_slots,
        route_blocks,
        fc1_c_tmp_elements,
        fc2_c_tmp_elements,
        fc1_cols,
    )


def _make_w4a16_expert_map(
    *,
    state_E: int,
    weight_E: int,
    device: torch.device,
) -> torch.Tensor | None:
    if int(state_E) == int(weight_E):
        return None
    if int(state_E) > int(weight_E):
        raise ValueError("num_local_experts cannot exceed num_experts")
    expert_map = torch.empty((int(weight_E),), dtype=torch.int32, device=device)
    expert_map.fill_(-1)
    expert_map[: int(state_E)].copy_(
        torch.arange(int(state_E), dtype=torch.int32, device=device)
    )
    return expert_map


def _allocate_sm120_w4a16_workspace(
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation: str = "silu",
) -> Sm120W4A16MoEWorkspace:
    is_gated = validate_activation(activation)
    routed_rows = max(1, int(routed_rows))
    route_num_experts = int(weight_E) if int(state_E) != int(weight_E) else int(state_E)
    (
        route_slots,
        route_blocks,
        fc1_c_tmp_elements,
        fc2_c_tmp_elements,
        fc1_cols,
    ) = _w4a16_workspace_geometry(
        routed_rows=routed_rows,
        route_num_experts=route_num_experts,
        num_topk=num_topk,
        k=k,
        n=n,
        is_gated=is_gated,
        device=device,
    )
    return Sm120W4A16MoEWorkspace(
        state_E=int(state_E),
        weight_E=int(weight_E),
        max_rows=routed_rows,
        k=int(k),
        n=int(n),
        num_topk=int(num_topk),
        device=device,
        activation=activation,
        activation_precision="bf16",
        quant_mode="w4a16",
        routed_rows_capacity=routed_rows,
        route_num_experts=route_num_experts,
        intermediate_cache13=torch.empty(
            (routed_rows * max(fc1_cols, int(k)),),
            dtype=torch.bfloat16,
            device=device,
        ),
        intermediate_cache2=torch.empty(
            (routed_rows, int(n)),
            dtype=torch.bfloat16,
            device=device,
        ),
        fc1_c_tmp=torch.empty(
            (fc1_c_tmp_elements,),
            dtype=torch.float32,
            device=device,
        ),
        fc2_c_tmp=torch.empty(
            (fc2_c_tmp_elements,),
            dtype=torch.float32,
            device=device,
        ),
        packed_route_indices=torch.empty(
            (route_slots,),
            dtype=torch.int32,
            device=device,
        ),
        block_expert_ids=torch.empty(
            (route_blocks,),
            dtype=torch.int32,
            device=device,
        ),
        packed_route_count=torch.empty((1,), dtype=torch.int32, device=device),
        expert_offsets=torch.empty(
            (route_num_experts + 1,),
            dtype=torch.int32,
            device=device,
        ),
        expert_map=_make_w4a16_expert_map(
            state_E=state_E,
            weight_E=weight_E,
            device=device,
        ),
    )


_W4A16_WEIGHT_CACHE: Dict[Tuple, W4A16PackedWeights] = {}


def _get_w4a16_packed_weights(
    *,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    activation: str,
    params_dtype: torch.dtype,
    source_format: str = "modelopt",
) -> W4A16PackedWeights:
    key = (
        activation,
        params_dtype,
        source_format,
        tuple(w1_weight.shape),
        tuple(w1_weight_sf.shape),
        tuple(w1_alpha.shape),
        tuple(w2_weight.shape),
        tuple(w2_weight_sf.shape),
        tuple(w2_alpha.shape),
        w1_weight.data_ptr(),
        w1_weight_sf.data_ptr(),
        w1_alpha.data_ptr(),
        w2_weight.data_ptr(),
        w2_weight_sf.data_ptr(),
        w2_alpha.data_ptr(),
    )
    cached = _W4A16_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_cuda_graph_capturing():
        raise RuntimeError(
            "W4A16 packed weights are not initialized for CUDA graph capture; "
            "run once before capture so the prepared weights are cached."
        )
    prepared = prepare_w4a16_packed_weights(
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        activation=activation,
        params_dtype=params_dtype,
        source_format=source_format,
    )
    _W4A16_WEIGHT_CACHE[key] = prepared
    _register_cache_eviction(
        _W4A16_WEIGHT_CACHE,
        key,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
    )
    return prepared


def _validate_w4a16_workspace(
    workspace: Sm120W4A16MoEWorkspace,
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation: str,
) -> None:
    validate_activation(activation)
    if workspace.state_E != int(state_E) or workspace.weight_E != int(weight_E):
        raise ValueError("pre-allocated W4A16 workspace expert geometry mismatch")
    if workspace.k != int(k) or workspace.n != int(n):
        raise ValueError("pre-allocated W4A16 workspace hidden geometry mismatch")
    if workspace.num_topk != int(num_topk):
        raise ValueError("pre-allocated W4A16 workspace top-k mismatch")
    if getattr(workspace, "activation", None) != activation:
        raise ValueError("pre-allocated W4A16 workspace activation mismatch")
    if _canonical_cuda_device(workspace.device) != _canonical_cuda_device(device):
        raise ValueError(
            f"pre-allocated W4A16 workspace is on {workspace.device}, expected {device}"
        )
    if workspace.routed_rows_capacity < max(1, int(routed_rows)):
        raise ValueError(
            "pre-allocated W4A16 workspace is too small for the requested "
            f"routed rows ({workspace.routed_rows_capacity} < {routed_rows})"
        )


def _launch_sm120_w4a16_moe(
    *,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    scatter_output: torch.Tensor,
    fast_math: bool = True,
    activation: str = "silu",
    source_format: str = "modelopt",
    _workspace=None,
    _prepared_weights=None,
) -> torch.Tensor:
    prepared = (
        _prepared_weights
        if isinstance(_prepared_weights, W4A16PackedWeights)
        else _get_w4a16_packed_weights(
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            activation=activation,
            params_dtype=a.dtype,
            source_format=source_format,
        )
    )
    if int(prepared.num_experts) != int(num_local_experts):
        raise ValueError("num_local_experts must match w1_weight.shape[0] for W4A16.")
    num_tokens = int(topk_ids.size(0))
    routed_rows = num_tokens * int(top_k)
    k = int(a.size(1))
    n = int(prepared.intermediate_size)

    if _workspace is None:
        workspace = _get_cached_workspace(
            backend="w4a16",
            state_E=num_local_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=top_k,
            device=a.device,
            quant_mode="w4a16",
            activation=activation,
        )
    else:
        workspace = _workspace
    if not isinstance(workspace, Sm120W4A16MoEWorkspace):
        raise TypeError("expected a W4A16 workspace for quant_mode='w4a16'")
    _validate_w4a16_workspace(
        workspace,
        state_E=num_local_experts,
        weight_E=num_experts,
        routed_rows=routed_rows,
        k=k,
        n=n,
        num_topk=top_k,
        device=a.device,
        activation=activation,
    )

    return run_w4a16_moe(
        a,
        prepared,
        topk_weights,
        topk_ids,
        activation=activation,
        intermediate_cache13=workspace.intermediate_cache13,
        intermediate_cache2=workspace.intermediate_cache2,
        output=scatter_output,
        fc1_c_tmp=workspace.fc1_c_tmp,
        fc2_c_tmp=workspace.fc2_c_tmp,
        packed_route_indices=workspace.packed_route_indices,
        block_expert_ids=workspace.block_expert_ids,
        packed_route_count=workspace.packed_route_count,
        expert_offsets=workspace.expert_offsets,
        expert_map=workspace.expert_map,
        fast_math=fast_math,
    )


# ==========================================================================
# Workspace cache (for functional API path)
# ==========================================================================

_Sm120Workspace = Union[
    Sm120StaticMoEWorkspace,
    Sm120DynamicMoEWorkspace,
    Sm120W4A16MoEWorkspace,
]

# Stores the workspace with the largest capacity seen per key and never
# shrinks within a process. clear_sm120_moe_caches() releases everything.
_WORKSPACE_CACHE: Dict[Tuple, _Sm120Workspace] = {}


def clear_sm120_moe_caches() -> None:
    """Release every module-level SM12x MoE cache.

    References held by callers are unaffected.
    """
    _WORKSPACE_CACHE.clear()
    _WEIGHT_CACHE.clear()
    _W4A16_WEIGHT_CACHE.clear()
    _PADDED_SCALE_CACHE.clear()
    _PADDED_FP4_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()
    _MICRO_KERNEL_CACHE.clear()
    _DIRECT_MICRO_LAUNCH_CACHE.clear()
    _DIRECT_MICRO_KERNEL_CACHE.clear()
    _DYNAMIC_KERNEL_CACHE.clear()


def allocate_sm120_moe_workspace(
    *,
    state_E: int,
    weight_E: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    max_rows: int | None = None,
    routed_rows: int | None = None,
    quant_mode: str | None = None,
    activation_precision: str | None = None,
    backend: str | None = None,
    activation: str = "silu",
) -> _Sm120Workspace:
    """Allocate the right SM120 MoE workspace from a quantization mode."""
    mode = _normalize_quant_mode(quant_mode, activation_precision)
    capacity_rows = routed_rows if routed_rows is not None else max_rows
    if capacity_rows is None:
        raise ValueError("routed_rows or max_rows is required")
    capacity_rows = max(1, int(capacity_rows))
    device = torch.device(device)

    if mode == "w4a16":
        if backend not in (None, "w4a16"):
            raise ValueError("quant_mode='w4a16' does not use static/dynamic backend")
        return _allocate_sm120_w4a16_workspace(
            state_E=state_E,
            weight_E=weight_E,
            routed_rows=capacity_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            activation=activation,
        )

    activation_precision = "fp4"
    _validate_w4a4_dimensions(k, n, mode)
    if backend is None:
        backend = select_sm120_moe_backend(
            num_tokens=max(
                1, (capacity_rows + max(1, int(num_topk)) - 1) // max(1, int(num_topk))
            ),
            num_topk=int(num_topk),
            activation_precision=activation_precision,
            quant_mode=mode,
            num_experts=weight_E,
            intermediate_size=n,
            hidden_size=k,
            activation=activation,
            capacity_tokens=max(
                1, (capacity_rows + max(1, int(num_topk)) - 1) // max(1, int(num_topk))
            ),
        )
    if backend == "dynamic":
        return allocate_sm120_dynamic_workspace(
            state_E=state_E,
            weight_E=weight_E,
            routed_rows=capacity_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            activation_precision=activation_precision,
            activation=activation,
            quant_mode=mode,
        )
    if backend == "static":
        return allocate_sm120_static_workspace(
            state_E=state_E,
            weight_E=weight_E,
            max_rows=capacity_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
            activation_precision=activation_precision,
            quant_mode=mode,
        )
    raise ValueError(f"unsupported SM120 MoE backend {backend!r}")


def _get_cached_workspace(
    *,
    backend: str,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    activation: str = "silu",
) -> _Sm120Workspace:
    """Get or allocate a cached workspace for the given problem shape.

    Reuses the cached workspace if it has enough capacity for the requested
    routed_rows. For static workspaces, max_rows is the direct capacity.
    For dynamic workspaces, routed_rows_capacity is used because the dynamic
    geometry (physical tiles, task queue slots) depends on the original
    routed_rows, not just max_rows.
    """
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    activation_precision = _activation_precision_from_quant_mode(quant_mode)
    # Key dynamic workspaces on the tile band of this call's routed_rows; a
    # larger cached workspace must not pin small calls to its 128 tile.
    tile_m = (
        _select_dynamic_tile_m(max(1, routed_rows), state_E, activation)
        if backend == "dynamic" and quant_mode != "w4a16"
        else None
    )
    cache_key = (
        state_E,
        weight_E,
        k,
        n,
        num_topk,
        str(device),
        backend,
        quant_mode,
        activation,
        tile_m,
    )
    cached = _WORKSPACE_CACHE.get(cache_key)

    if cached is not None:
        if isinstance(cached, Sm120DynamicMoEWorkspace):
            if cached.routed_rows_capacity >= max(1, routed_rows):
                assert tile_m is None or cached.tile_m == tile_m
                return cached
        elif isinstance(cached, Sm120W4A16MoEWorkspace):
            if cached.routed_rows_capacity >= max(1, routed_rows):
                return cached
        else:
            if cached.max_rows >= max(1, routed_rows):
                return cached

    if quant_mode == "w4a16" and _is_cuda_graph_capturing():
        raise RuntimeError(
            "W4A16 workspace is not initialized for CUDA graph capture; "
            "provide a preallocated workspace from "
            "allocate_sm120_moe_workspace(..., quant_mode='w4a16') or warm the "
            "functional path before capture."
        )
    _refuse_during_capture("a per-call MoE workspace")
    workspace = allocate_sm120_moe_workspace(
        state_E=state_E,
        weight_E=weight_E,
        routed_rows=routed_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        quant_mode=quant_mode,
        activation_precision=activation_precision,
        backend=backend,
        activation=activation,
    )

    _WORKSPACE_CACHE[cache_key] = workspace
    return workspace


# ==========================================================================
# Unified dispatch
# ==========================================================================
# Tile-padded block scales (needed by every W4A4 kernel for I % 128 != 0) and tile-padded FP4 copies (dynamic /
# micro kernels only, built lazily) are cached separately: one scale bundle per weight set, never duplicated.
_PADDED_SCALE_CACHE: Dict[Tuple, Tuple] = {}
_PADDED_FP4_CACHE: Dict[Tuple, Tuple] = {}


def _pad_intermediate_to_tile(
    w1_weight,
    w1_weight_sf,
    w2_weight,
    w2_weight_sf,
    fc2_input_scale,
    n,
    tile,
    h,
    num_experts,
    is_gated,
    quant_mode="nvfp4",
    pad_fp4=True,
):
    """Zero-pad W4A4 weights + scale factors so the intermediate size is a
    multiple of ``tile`` (gate/up tile-split requirement); padded channels are
    zero, so the result is numerically identical.

    With ``pad_fp4=False`` only the block-scale tensors are padded: the static kernel streams the FP4 weights at
    the true extent through TMA views and needs just the 128-aligned scale
    layout, so a static-only caller never materializes the padded FP4 copies
    (the returned FP4 tensors are the originals).
    """
    quant_mode = _normalize_quant_mode(quant_mode)
    sf_vec_size, _ = _sf_params_for_quant_mode(quant_mode)
    n_pad = ((n + tile - 1) // tile) * tile
    if n_pad == n:
        return w1_weight, w1_weight_sf, w2_weight, w2_weight_sf, fc2_input_scale, n
    E = int(num_experts)

    def mma_to_logical(sf, m, k):
        sw = convert_sf_from_mma_layout(
            sf,
            m=m,
            k=k,
            num_groups=E,
            sf_vec_size=sf_vec_size,
        )
        m_pad = ((m + 127) // 128) * 128
        sw = sw.reshape(E, m_pad, -1)
        cb = (k + sf_vec_size - 1) // sf_vec_size
        # MXFP4 logical scales remain raw UE8M0 bytes.
        if quant_mode == "mxfp4":
            cols_padded = ((cb + 3) // 4) * 4
            return torch.stack(
                [
                    sw[e]
                    .reshape(m_pad // 128, cols_padded // 4, 32, 4, 4)
                    .permute(0, 3, 2, 1, 4)
                    .contiguous()
                    .reshape(m_pad, cols_padded)[:m, :cb]
                    for e in range(E)
                ],
                0,
            )
        # NVFP4 logical scales are decoded float32 magnitudes.
        return torch.stack(
            [unswizzle_block_scale(sw[e], rows=m, cols_blocks=cb) for e in range(E)], 0
        )

    def logical_to_mma(log, m, k):
        sw = torch.stack([swizzle_block_scale(log[e]) for e in range(E)], 0)
        scale_dtype = torch.uint8 if quant_mode == "mxfp4" else torch.float8_e4m3fn
        sw2d = sw.reshape(E * sw.shape[1], sw.shape[2]).to(scale_dtype)
        return convert_sf_to_mma_layout(
            sw2d,
            m=m,
            k=k,
            num_groups=E,
            sf_vec_size=sf_vec_size,
        )

    def pad_dim(t, dim, old, new):
        if new == old:
            return t
        shp = list(t.shape)
        shp[dim] = new - old
        return torch.cat([t, t.new_zeros(shp)], dim=dim)

    # The block scales are padded once per weight set (the static kernel needs the 128-aligned per-branch scale
    # layout on its first call); the FP4 copies are a separate, later cache entry so a static-only caller never
    # holds them and a dynamic / micro caller never pads the scales twice.
    scale_key = (
        n,
        tile,
        h,
        E,
        bool(is_gated),
        quant_mode,
        w1_weight_sf.data_ptr(),
        w2_weight_sf.data_ptr(),
    )
    scales = _PADDED_SCALE_CACHE.get(scale_key)
    if scales is None:
        _refuse_during_capture("the tile-padded block scales")
        if is_gated:
            log1 = mma_to_logical(w1_weight_sf, m=2 * n, k=h)
            up_sf, gate_sf = log1[:, :n, :], log1[:, n : 2 * n, :]
            log1p = torch.cat(
                [pad_dim(up_sf, 1, n, n_pad), pad_dim(gate_sf, 1, n, n_pad)], dim=1
            )
            w1_sf_p = logical_to_mma(log1p, m=2 * n_pad, k=h)
        else:
            log1 = mma_to_logical(w1_weight_sf, m=n, k=h)
            w1_sf_p = logical_to_mma(pad_dim(log1, 1, n, n_pad), m=n_pad, k=h)
        log2 = mma_to_logical(w2_weight_sf, m=h, k=n)
        cb_n = (n + sf_vec_size - 1) // sf_vec_size
        cb_np = (n_pad + sf_vec_size - 1) // sf_vec_size
        w2_sf_p = logical_to_mma(pad_dim(log2, 2, cb_n, cb_np), m=h, k=n_pad)
        scales = (w1_sf_p, w2_sf_p)
        _PADDED_SCALE_CACHE[scale_key] = scales
        _register_cache_eviction(
            _PADDED_SCALE_CACHE,
            scale_key,
            w1_weight_sf,
            w2_weight_sf,
        )
    w1_sf_p, w2_sf_p = scales
    # The scale is scalar/[E], not [I]. Return the caller's live tensor even
    # on a weight-cache hit so in-place updates and graph replay stay valid.
    if not pad_fp4:
        return w1_weight, w1_sf_p, w2_weight, w2_sf_p, fc2_input_scale, n_pad

    fp4_key = (n, tile, E, bool(is_gated), w1_weight.data_ptr(), w2_weight.data_ptr())
    fp4 = _PADDED_FP4_CACHE.get(fp4_key)
    if fp4 is None:
        _refuse_during_capture("the tile-padded FP4 weight copies")
        if is_gated:
            # w1 packs [up(0:n), gate(n:2n)] rows; pad each half so the split stays
            # tile-aligned, then re-concat.
            up, gate = w1_weight[:, :n, :], w1_weight[:, n : 2 * n, :]
            w1p = torch.cat(
                [pad_dim(up, 1, n, n_pad), pad_dim(gate, 1, n, n_pad)], dim=1
            )
        else:
            w1p = pad_dim(w1_weight, 1, n, n_pad)
        # w2 reduces over the intermediate dim: pad its packed columns.
        w2p = pad_dim(w2_weight, 2, n // 2, n_pad // 2)
        fp4 = (w1p, w2p)
        _PADDED_FP4_CACHE[fp4_key] = fp4
        _register_cache_eviction(_PADDED_FP4_CACHE, fp4_key, w1_weight, w2_weight)
    w1p, w2p = fp4
    return w1p, w1_sf_p, w2p, w2_sf_p, fc2_input_scale, n_pad


def _validate_static_workspace_for_launch(
    workspace,
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
    activation_precision: str,
    quant_mode: str,
) -> None:
    """Reject stale or undersized shared static workspaces before launch."""
    if type(workspace) is not Sm120StaticMoEWorkspace:
        raise ValueError(
            "pre-allocated static workspace must be Sm120StaticMoEWorkspace"
        )
    expected_device = _canonical_cuda_device(device)
    try:
        workspace_device = _canonical_cuda_device(workspace.device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"pre-allocated static workspace device is invalid: {workspace.device!r}."
        ) from error
    expected_metadata = {
        "state_E": state_E,
        "weight_E": weight_E,
        "k": k,
        "n": n,
        "num_topk": num_topk,
        "activation_precision": activation_precision,
        "quant_mode": quant_mode,
    }
    for name, expected in expected_metadata.items():
        if getattr(workspace, name, None) != expected:
            raise ValueError(
                f"pre-allocated static workspace {name} mismatch: "
                f"expected {expected!r}, got {getattr(workspace, name, None)!r}."
            )
    if workspace_device != expected_device:
        raise ValueError(
            f"pre-allocated static workspace device mismatch: expected "
            f"{expected_device}, got {workspace_device}."
        )
    if not isinstance(workspace.max_rows, int) or workspace.max_rows < routed_rows:
        raise ValueError(
            "pre-allocated static workspace capacity is too small: "
            f"max_rows={getattr(workspace, 'max_rows', None)!r}, "
            f"routed_rows={routed_rows}."
        )

    max_rows = workspace.max_rows
    max_chunks = (max_rows + 31) // 32
    virt_E = state_E + max_chunks
    sf_vec_size, _ = _sf_params_for_quant_mode(quant_mode)
    slot_rows = _STATIC_SLOT_ROWS
    rows_pad_k = _align_up(slot_rows, 128)
    cols_pad_k = _align_up(k // sf_vec_size, 4)
    retained_groups = _static_retained_groups(n)
    tensors = {
        "row_counts": ((virt_E,), torch.int32),
        "token_map": ((virt_E, slot_rows), torch.int32),
        "token_weights": ((virt_E, slot_rows), torch.float32),
        "packed_input": ((virt_E, slot_rows, k // 2), torch.uint8),
        "packed_input_scale": (
            (virt_E, rows_pad_k, cols_pad_k),
            torch.uint8,
        ),
        "barrier_count": ((1,), torch.int32),
        "barrier_epoch": ((1,), torch.int32),
        "active_expert_count": ((1,), torch.int32),
        "weight_expert_ids": ((virt_E,), torch.int32),
        "global_to_local_expert": ((weight_E,), torch.int32),
        "compact_topk_ids": ((max(state_E, max_rows),), torch.int32),
        "route_state": ((8,), torch.int32),
        "micro_row_counts": ((virt_E,), torch.int32),
        "micro_active_expert_count": ((1,), torch.int32),
        "virt_route_scratch": (
            (weight_E * (1 + max_chunks) + 8,),
            torch.int32,
        ),
        "route_output_scratch": (
            (max_rows, retained_groups, k),
            torch.bfloat16,
        ),
        "packed_a_view": (
            (slot_rows, k // 2, virt_E),
            torch.float4_e2m1fn_x2,
        ),
        "packed_a_flat": ((virt_E * slot_rows * (k // 2),), torch.uint8),
        "scale_flat": (
            (virt_E * rows_pad_k * cols_pad_k,),
            torch.uint8,
        ),
    }
    if (
        quant_mode == "nvfp4"
        and state_E == weight_E
        and _direct_micro_candidate(k, n, num_topk, weight_E)
    ):
        dm_rows = min(max_rows, _MICRO_MAX_TOKENS * num_topk)
        dm_slots = dm_rows + _MICRO_MAX_TOKENS * 16
        fc2_n_chunks = (n // 2 + 127) // 128
        dm_intermediate = _MICRO_MAX_TOKENS * num_topk * fc2_n_chunks * 128
        tensors.update(
            {
                "dm_barrier_count": ((dm_slots,), torch.int32),
                "dm_barrier_epoch": ((dm_slots,), torch.int32),
                "dm_intermediate": ((dm_intermediate,), torch.float32),
                "dm_input_gs": ((weight_E,), torch.float32),
                "dm_down_input_scale": ((weight_E,), torch.float32),
            }
        )
    for name, (shape, dtype) in tensors.items():
        value = getattr(workspace, name, None)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or _canonical_cuda_device(value.device) != expected_device
            or value.data_ptr() % 16 != 0
        ):
            raise ValueError(
                f"pre-allocated static workspace {name} mismatch: expected "
                f"shape={shape}, dtype={dtype}, device={expected_device}, "
                "and 16-byte alignment."
            )

    for view_name, storage_name in (
        ("packed_a_view", "packed_input"),
        ("packed_a_flat", "packed_input"),
        ("scale_flat", "packed_input_scale"),
    ):
        if (
            getattr(workspace, view_name).data_ptr()
            != getattr(workspace, storage_name).data_ptr()
        ):
            raise ValueError(
                f"pre-allocated static workspace {view_name} must alias {storage_name}"
            )

    packed = workspace.packed_input
    route = workspace.route_output_scratch
    packed_range = (
        packed.data_ptr(),
        packed.data_ptr() + packed.numel() * packed.element_size(),
    )
    route_range = (
        route.data_ptr(),
        route.data_ptr() + route.numel() * route.element_size(),
    )
    if max(packed_range[0], route_range[0]) < min(packed_range[1], route_range[1]):
        raise ValueError(
            "pre-allocated static workspace packed_input and "
            "route_output_scratch storage overlap"
        )


def launch_sm120_moe(
    *,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor] = None,
    input_global_scale: Optional[torch.Tensor] = None,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    scatter_output: torch.Tensor,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    swiglu_alpha: float = 1.702,
    swiglu_beta: float = 1.0,
    swiglu_limit: float | None = None,
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    source_format: str = "modelopt",
    _workspace=None,
    _weight_views=None,
    _prepared_weights=None,
) -> torch.Tensor:
    """Unified SM120 MoE dispatch — selects static or dynamic by token count.

    input_global_scale overrides w1_alpha as the FC1 input-quant scale and
    is folded into the multiplier internally.  With _weight_views supplied,
    w1_alpha must already contain the fold.

    Optional _workspace and _weight_views can be pre-allocated and reused
    across calls to avoid per-call allocation overhead (wrapper path).
    When not provided (functional API path), a module-level workspace cache
    is used to avoid re-allocating on every call.
    """
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)
    activation_precision = _activation_precision_from_quant_mode(quant_mode)

    num_tokens = topk_ids.size(0)
    k = a.size(1)  # hidden_size
    is_gated = is_gated_activation(activation)
    # w1_weight.size(1) is 2*n for gated or n for non-gated
    intermediate_size = w1_weight.size(1) // 2 if is_gated else w1_weight.size(1)
    n = intermediate_size
    _validate_w4a4_dimensions(k, n, quant_mode)
    n_unpadded = n
    if quant_mode != "w4a16":
        n = _align_up(n, _LEVEL_TILE_N)

    routed_rows = num_tokens * top_k

    if quant_mode == "w4a16":
        return _launch_sm120_w4a16_moe(
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            scatter_output=scatter_output,
            fast_math=fast_math,
            activation=activation,
            source_format=source_format,
            _workspace=_workspace,
            _prepared_weights=_prepared_weights,
        )

    if quant_mode == "nvfp4" and input_global_scale is not None:
        input_gs = input_global_scale
        if _weight_views is None:
            # Alpha must carry input_gs back or the output magnitude is wrong.
            # The wrapper folds before building _weight_views; don't fold twice.
            w1_alpha = (
                w1_alpha.to(torch.float32) * input_global_scale.to(torch.float32)
            ).contiguous()
    else:
        input_gs = w1_alpha

    # Resolve the backend before applying its private physical geometry.
    if _workspace is not None:
        workspace = _workspace
        workspace_activation_precision = getattr(
            workspace, "activation_precision", activation_precision
        )
        if workspace_activation_precision != activation_precision:
            raise ValueError(
                "pre-allocated workspace activation_precision does not match "
                f"requested activation_precision={activation_precision!r}."
            )
        workspace_quant_mode = getattr(workspace, "quant_mode", quant_mode)
        if workspace_quant_mode != quant_mode:
            raise ValueError(
                "pre-allocated workspace quant_mode does not match "
                f"requested quant_mode={quant_mode!r}."
            )
        if isinstance(workspace, Sm120DynamicMoEWorkspace):
            if num_local_experts != num_experts:
                raise ValueError(
                    "pre-allocated dynamic SM120 MoE workspace requires "
                    "num_local_experts == num_experts because dynamic expert "
                    "buffers are indexed by global topk ids."
                )
            # A pre-allocated dynamic workspace keeps its stored tile_m even
            # for smaller calls; its geometry was sized for that tile.
            backend = "dynamic"
        else:
            backend = "static"
    else:
        workspace = None
        backend = select_sm120_moe_backend(
            num_tokens=num_tokens,
            num_topk=top_k,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
            num_experts=num_experts,
            intermediate_size=n_unpadded,
            hidden_size=k,
            activation=activation,
            capacity_tokens=num_tokens,
        )
        # The dynamic kernel indexes row_counts/expert_write_rows directly with
        # topk_ids but those buffers are sized with num_local_experts. Unless
        # num_local_experts == num_experts, fall back to the static backend which
        # has global-to-local expert remapping.
        if backend == "dynamic" and num_local_experts != num_experts:
            backend = "static"

    # retained2 consumes two adjacent N128 slices; an odd slice count keeps a
    # phantom partner that TMA zero-fills, so no 256-aligned padding is needed.
    weight_views = _weight_views

    if fc2_input_scale is None:
        if quant_mode == "nvfp4":
            raise ValueError("fc2_input_scale is required when quant_mode='nvfp4'.")
        # MXFP4 has no tensor-wide FC2 input scale. Reuse an existing
        # per-expert tensor because the shared kernel signature still carries
        # the argument; the MXFP4 quantizer ignores it.
        down_input_scale = w2_alpha
    else:
        down_input_scale = fc2_input_scale
    weights = (
        weight_views
        if weight_views is not None
        else _prepare_weight_views(
            w1_fp4=w1_weight,
            w1_blockscale=w1_weight_sf,
            w2_fp4=w2_weight,
            w2_blockscale=w2_weight_sf,
            w1_alphas=w1_alpha,
            w2_alphas=w2_alpha,
            n=n_unpadded,
            k=k,
            quant_mode=quant_mode,
            activation=activation,
        )
    )

    if workspace is None:
        workspace = _get_cached_workspace(
            backend=backend,
            state_E=num_local_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=top_k,
            device=a.device,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
            activation=activation,
        )

    if backend == "static":
        _validate_static_workspace_for_launch(
            workspace,
            state_E=num_local_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=top_k,
            device=a.device,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
        )

    if backend == "dynamic":
        return launch_sm120_dynamic_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=input_gs,
            down_input_scale=down_input_scale,
            scatter_output=scatter_output,
            num_experts=num_experts,
            num_tokens=num_tokens,
            k=k,
            n=n,
            top_k=top_k,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
        )
    else:
        return launch_sm120_static_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=input_gs,
            down_input_scale=down_input_scale,
            scatter_output=scatter_output,
            num_experts=num_experts,
            num_tokens=num_tokens,
            k=k,
            n=n,
            top_k=top_k,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            activation_precision=activation_precision,
            quant_mode=quant_mode,
        )
