"""SM120/SM121 MoE dispatch layer — workspace, compilation, and launch.

Ported from b12x's integration/tp_moe.py. Supports micro (tiny decode),
static (decode), and dynamic (prefill) backends with token-count-based
selection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import sym_int

from flashinfer.cute_dsl.utils import (
    convert_sf_from_mma_layout,
    current_cuda_stream,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from .moe_activations import (
    SWIGLUOAI_UNINTERLEAVE,
    is_gated_moe_activation,
    normalize_moe_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)
from .moe_silu import (
    MoEDynamicKernelSilu,
    MoEDynamicKernelSwiGLUOAI,
    MoEMicroKernelSilu,
    MoEMicroKernelSwiGLUOAI,
)
from .moe_relu2 import MoEDynamicKernelRelu2, MoEMicroKernelRelu2
from .moe_w4a16_host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
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
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
_DYNAMIC_SLICE_CHUNK = 1
SF_VEC_SIZE = 16
_FORCE_MOE_W4A16_ENV = "FLASHINFER_B12X_FORCE_MOE_W4A16"
_MICRO_SHARE_INPUT_ACROSS_EXPERTS = (
    os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
)

# Micro kernel cutover thresholds (routed pairs)
_MICRO_COMPACT_CUTOVER_PAIRS = 20
_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK = 40
_STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT = 640
_STATIC_COMPACT_CUTOVER_PAIRS = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
_STATIC_COMPACT_CUTOVER_PAIRS_CACHE: Dict[str, int] = {}

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
_STATIC_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (24, 148),
    (32, 169),
    (40, 132),
    (48, 149),
    (64, 134),
    (80, 175),
    (96, 171),
    (120, 125),
    (128, 130),
    (160, 171),
    (192, 166),
    (256, 141),
    (320, 158),
    (512, 175),
    (640, 188),
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


# Quantization modes understood by the dispatch layer. Rebased onto upstream
# b12x's explicit 4-mode set (HEAD 5af873a):
#   nvfp4       — FP4 weights + FP4 activations (NVFP4, E4M3 K/16 block scales)
#   w4a16       — FP4 weights + BF16 activations (weight-only)
#   w4a8_mx     — FP4 weights + MXFP8 (E4M3) activations, UE8M0 K/32 scales
#   w4a8_nvfp4  — FP4 weights + FP8 (E4M3) activations, NVFP4 scales decomposed
_W4A8_QUANT_MODES = frozenset({"w4a8_mx", "w4a8_nvfp4"})

# FP4-source formats for the nvfp4 / w4a8 paths. "modelopt" is the legacy
# FlashInfer alias for upstream "modelopt_nvfp4".
_FP4_SOURCE_FORMATS = {
    "modelopt": "modelopt_nvfp4",
    "modelopt_nvfp4": "modelopt_nvfp4",
    "fp4_e8m0_k32": "fp4_e8m0_k32",
    "compressed_tensors": "compressed_tensors",
}


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
        "bf16": "w4a16",
        "w4a16": "w4a16",
        "w4a8_mx": "w4a8_mx",
        "w4a8_nvfp4": "w4a8_nvfp4",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "quant_mode must be one of 'nvfp4'/'w4a4', 'w4a16', "
            f"'w4a8_mx', or 'w4a8_nvfp4' (got {quant_mode!r})."
        ) from exc


def _is_w4a8(quant_mode: str) -> bool:
    """True for the FP8-activation x FP4-weight throughput tiers."""
    return _normalize_quant_mode(quant_mode) in _W4A8_QUANT_MODES


def _activation_precision_from_quant_mode(quant_mode: str) -> str:
    mode = _normalize_quant_mode(quant_mode)
    if mode == "w4a16":
        return "bf16"
    if mode in _W4A8_QUANT_MODES:
        return "fp8"
    return "fp4"


def _normalize_fp4_source_format(source_format: str) -> str:
    """Normalize an FP4-path source format to the upstream canonical name."""
    lowered = str(source_format).lower()
    if lowered == "mxfp4_native":
        raise ValueError(
            "source_format='mxfp4_native' has been removed; use "
            "source_format='fp4_e8m0_k32' for byte-preserved E8M0 K/32 scales."
        )
    try:
        return _FP4_SOURCE_FORMATS[lowered]
    except KeyError as exc:
        raise ValueError(
            "source_format must be one of 'modelopt'/'modelopt_nvfp4', "
            "'fp4_e8m0_k32', or 'compressed_tensors', "
            f"got {source_format!r}"
        ) from exc


def _validate_fp4_source_format_for_quant_mode(
    *, source_format: str, quant_mode: str
) -> None:
    """Upstream (mode, source) compatibility matrix for the FP4 paths."""
    normalized = _normalize_fp4_source_format(source_format)
    mode = _normalize_quant_mode(quant_mode)
    if mode == "w4a16":
        return
    if normalized == "modelopt_nvfp4":
        return
    if normalized == "fp4_e8m0_k32" and mode == "w4a8_mx":
        return
    raise ValueError(
        f"source_format={normalized!r} with quant_mode={mode!r} is "
        "unsupported; use quant_mode='w4a16' for non-NVFP4 sources, "
        "quant_mode='w4a8_mx' for source_format='fp4_e8m0_k32', or "
        "source_format='modelopt_nvfp4' for NVFP4/W4A8-NVFP4 kernels"
    )


def _normalize_source_format_for_quant_mode(source_format: str, quant_mode: str) -> str:
    """Validate (mode, source) and return the format name for that path.

    w4a16 keeps the FlashInfer-native 'modelopt'/'compressed_tensors' names
    (consumed by the W4A16 weight-prep). The nvfp4/w4a8 paths use the upstream
    canonical FP4-source names.
    """
    mode = _normalize_quant_mode(quant_mode)
    if mode == "w4a16":
        return _normalize_source_format(source_format)
    # nvfp4 / w4a8: validate against the upstream matrix.
    _validate_fp4_source_format_for_quant_mode(
        source_format=source_format, quant_mode=mode
    )
    return _normalize_fp4_source_format(source_format)


def _is_w4a16(activation_precision: str) -> bool:
    return _normalize_activation_precision(activation_precision) == "bf16"


def _level_tile_m(activation_precision: str = "fp4") -> int:
    if _is_w4a16(activation_precision):
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 tile selector"
        )
    return _LEVEL_TILE_M


def _level_tile_n(activation_precision: str = "fp4") -> int:
    if _is_w4a16(activation_precision):
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 tile selector"
        )
    return _LEVEL_TILE_N


def _get_static_compact_cutover_pairs(activation_precision: str = "fp4") -> int:
    activation_precision = _normalize_activation_precision(activation_precision)
    cached = _STATIC_COMPACT_CUTOVER_PAIRS_CACHE.get(activation_precision)
    if cached is not None:
        return cached

    cutover_names: tuple[str, ...] = (
        "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
        "B12X_DYNAMIC_STATIC_CUTOVER_PAIRS",
        "B12X_LEVEL10_STATIC_CUTOVER_PAIRS",
    )
    cutover = _first_env(*cutover_names)
    if cutover is None:
        cached = _STATIC_COMPACT_CUTOVER_PAIRS_DEFAULT
    else:
        cached = max(0, int(cutover))
    _STATIC_COMPACT_CUTOVER_PAIRS_CACHE[activation_precision] = cached
    return cached


def _select_moe_mma_tiler_mn(routed_rows: int, n: int) -> Tuple[int, int]:
    """Select optimal MoE tile shape based on routed rows and N dimension.

    Uses narrower 64x128 tiles when routed_rows <= 128 and default 128x128
    would leave SMs idle.
    """
    sm_count = get_num_sm(torch.device("cuda"))
    coarse_tile = (128, 128)
    coarse_tiles = ((routed_rows + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # Single-token decode often lands exactly on the "half the machine"
    # boundary. Keeping the coarse 128x128 tile there leaves the M dimension
    # badly underfilled, so take the narrow 64x128 tile inclusive of equality.
    if routed_rows <= 128 and coarse_tiles <= max(1, sm_count // 2):
        return (64, 128)
    return (128, 128)


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

    # Buffers
    row_counts: torch.Tensor  # [state_E] int32
    token_map: torch.Tensor  # [state_E, max_rows] int32
    token_weights: torch.Tensor  # [state_E, max_rows] float32
    packed_input: torch.Tensor  # [state_E, max_rows, k//2] uint8
    packed_input_scale: torch.Tensor  # [state_E, rows_pad_k, cols_pad_k] uint8
    barrier_count: torch.Tensor  # [1] int32
    barrier_epoch: torch.Tensor  # [1] int32
    active_expert_count: torch.Tensor  # [1] int32
    weight_expert_ids: torch.Tensor  # [state_E] int32
    global_to_local_expert: torch.Tensor  # [weight_E] int32
    compact_topk_ids: torch.Tensor  # [state_E] int32, for micro kernel pre-pass

    # Views (set after allocation)
    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None


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
) -> Sm120StaticMoEWorkspace:
    """Allocate workspace buffers for the SM120 static MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "allocate_sm120_static_workspace only supports quant_mode='nvfp4'; "
            "use allocate_sm120_moe_workspace(..., quant_mode='w4a16') for W4A16."
        )

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)
    packed_input = torch.empty(
        state_E, max_rows, k // 2, dtype=torch.uint8, device=device
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
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(state_E, max_rows, dtype=torch.int32, device=device),
        token_weights=torch.zeros(
            state_E, max_rows, dtype=torch.float32, device=device
        ),
        packed_input=packed_input,
        packed_input_scale=torch.empty(
            state_E, rows_pad_k, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        weight_expert_ids=torch.arange(state_E, dtype=torch.int32, device=device),
        global_to_local_expert=torch.empty(weight_E, dtype=torch.int32, device=device),
        compact_topk_ids=torch.empty(
            max(state_E, max_rows), dtype=torch.int32, device=device
        ),
    )

    # Finalize views
    sf_dtype = cutlass.Float8E4M3FN
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


_WEIGHT_CACHE: Dict[Tuple, _WeightViews] = {}


# ---------------------------------------------------------------------------
# Activation -> backend class resolvers
# ---------------------------------------------------------------------------
def _resolve_micro_cls(activation: str) -> type:
    """Map a normalized activation name to its micro-kernel backend class."""
    activation = normalize_moe_activation(activation)
    if activation == "relu2":
        return MoEMicroKernelRelu2
    if activation == SWIGLUOAI_UNINTERLEAVE:
        return MoEMicroKernelSwiGLUOAI
    return MoEMicroKernelSilu


def _resolve_dynamic_cls(activation: str) -> type:
    """Map a normalized activation name to its dynamic-kernel backend class."""
    activation = normalize_moe_activation(activation)
    if activation == "relu2":
        return MoEDynamicKernelRelu2
    if activation == SWIGLUOAI_UNINTERLEAVE:
        return MoEDynamicKernelSwiGLUOAI
    return MoEDynamicKernelSilu


def _resolve_swiglu_params(
    activation: str,
    swiglu_limit: float | None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
) -> Tuple[float | None, float, float]:
    """Normalize (limit, alpha, beta) for the given activation.

    Defaults: swigluoai -> (7.0, 1.702, 1.0); silu/relu2 -> (None, 1.0, 0.0).
    """
    activation = normalize_moe_activation(activation)
    return (
        normalize_swiglu_limit_for_activation(activation, swiglu_limit),
        normalize_swiglu_alpha_for_activation(activation, swiglu_alpha),
        normalize_swiglu_beta_for_activation(activation, swiglu_beta),
    )


# Storages already flipped to the kernel-native [up; gate] order, keyed by
# (w1_fp4 data_ptr, w1_blockscale data_ptr). The gated micro/dynamic NVFP4
# kernels consume FC1 weights as [up; gate] ("w13"); swigluoai sources arrive
# as [gate; up] ("w31"), so flip the FP4 bytes and the swizzled block-scale
# halves once per storage (the load-time mirror of upstream's W4A16 row
# rotation). Mirrors b12x integration/tp_moe.py::_ensure_w13_kernel_order_inplace.
# Values hold the tensors so the allocator cannot recycle a registered data_ptr
# into a stale repack.
_W13_NORMALIZED_STORAGES: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _swap_w13_fp4_halves_inplace(w1_fp4: torch.Tensor, n: int) -> None:
    """Swap the two FC1 halves of the packed FP4 weights in place ([E, 2n, k//2])."""
    w1_u8 = w1_fp4.view(torch.uint8)
    if w1_u8.dim() != 3 or int(w1_u8.shape[1]) != 2 * n:
        raise ValueError(
            f"w31 FC1 weights must be [E, 2n, k//2]; got {tuple(w1_u8.shape)} for n={n}"
        )
    E = int(w1_u8.shape[0])
    per_expert = int(w1_u8.shape[1]) * int(w1_u8.shape[2])
    # Swap halves a few experts at a time to bound the temporary.
    chunk = max(1, min(E, (32 << 20) // max(1, per_expert)))
    for e0 in range(0, E, chunk):
        sl = w1_u8[e0 : e0 + chunk]
        tmp = sl[:, :n].clone()
        sl[:, :n] = sl[:, n:]
        sl[:, n:] = tmp


def _ensure_w13_kernel_order_inplace(
    w1_fp4: torch.Tensor,
    w13_sf_contiguous: torch.Tensor,
    *,
    n: int,
    num_experts: int,
    rows_pad: int,
) -> None:
    """Flip gate-first ("w31") FC1 storage to kernel-native [up; gate] order.

    Flips the packed FP4 weight bytes in place (once per storage, tracked by
    data_ptr) and swaps the gate/up halves of the *un-swizzled contiguous*
    block-scale buffer ``w13_sf_contiguous`` (shape [num_experts*rows_pad,
    cols]). The scale buffer is freshly produced per call by
    ``convert_sf_from_mma_layout(...).contiguous()`` so it is always swapped to
    match the (idempotently) flipped FP4 storage.
    """
    if not w1_fp4.is_contiguous():
        raise ValueError("w31 FC1 normalization requires contiguous w1 storage")

    # Flip the FP4 weight bytes once per physical storage.
    reg_key = (w1_fp4.data_ptr(), w13_sf_contiguous.data_ptr())
    if reg_key not in _W13_NORMALIZED_STORAGES:
        _swap_w13_fp4_halves_inplace(w1_fp4, n)
        _W13_NORMALIZED_STORAGES[reg_key] = (w1_fp4, w13_sf_contiguous)

    # Swap the gate/up scale halves on the un-swizzled contiguous buffer. Rows
    # are physically [E, rows_pad] (padded to 128); within each expert block the
    # first 2n rows hold [gate(0:n); up(n:2n)] and must become [up; gate].
    sf2d = w13_sf_contiguous.view(num_experts, rows_pad, -1)
    for e in range(num_experts):
        tmp = sf2d[e, :n].clone()
        sf2d[e, :n] = sf2d[e, n : 2 * n]
        sf2d[e, n : 2 * n] = tmp


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
    activation: str = "silu",
) -> _WeightViews:
    """Create permuted weight views for the static kernel.

    The kernel expects concatenated w13 data with shape [2*n, k//2, E]
    via a single TMA descriptor.

    For ``swigluoai_uninterleave`` the FC1 weights arrive gate-first ("w31");
    the gated micro/dynamic NVFP4 kernels consume [up; gate] ("w13"), so the
    FP4 bytes and block-scale halves are flipped once per storage (see
    ``_ensure_w13_kernel_order_inplace``). silu/relu2 are already kernel-native.
    """
    activation = normalize_moe_activation(activation)
    activation_precision = _normalize_activation_precision(activation_precision)
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
        activation,
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    needs_w31_flip = activation == SWIGLUOAI_UNINTERLEAVE and is_gated_moe_activation(
        activation
    )

    # Permute [E, w1_rows, k//2] -> [w1_rows, k//2, E] (view, no copy)
    # w1_rows is 2*n for gated (SiLU) or n for non-gated (ReLU2)
    w13 = w1_fp4.permute(1, 2, 0)
    down = w2_fp4.permute(1, 2, 0)

    # The kernel's TMA descriptors read scale factors in the physical storage
    # order produced by swizzle_block_scale: (batch, rows_padded, cols_padded).
    # convert_sf_to_mma_layout returns a strided 6D view over this storage.
    # We need the ORIGINAL physical storage, not .contiguous() of the view
    # (which would write in permuted logical order).
    # convert_sf_from_mma_layout reverses the permutation back to 2D swizzled.
    sf_dtype = cutlass.Float8E4M3FN
    w1_rows = w1_fp4.shape[1]  # 2*n for gated, n for non-gated
    w13_sf_contiguous = convert_sf_from_mma_layout(
        w1_blockscale,
        m=w1_rows,
        k=k,
        num_groups=w1_fp4.shape[0],  # num_local_experts
    ).contiguous()
    down_sf_contiguous = convert_sf_from_mma_layout(
        w2_blockscale,
        m=k,
        k=n,
        num_groups=w2_fp4.shape[0],
    ).contiguous()

    if needs_w31_flip:
        # rows_pad = ceil(w1_rows/128)*128; convert_sf_from_mma_layout pads M.
        rows_pad = ((w1_rows + 127) // 128) * 128
        _ensure_w13_kernel_order_inplace(
            w1_fp4,
            w13_sf_contiguous,
            n=n,
            num_experts=int(w1_fp4.shape[0]),
            rows_pad=rows_pad,
        )

    views = _WeightViews(
        w13_fp4=w13.view(torch.float4_e2m1fn_x2),
        down_fp4=down.view(torch.float4_e2m1fn_x2),
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
        w1_alpha=w1_alphas.contiguous().to(torch.float32),
        w2_alpha=w2_alphas.contiguous().to(torch.float32),
        w1_storage=w1_fp4,
        w1_scale_storage=w13_sf_contiguous,
        w2_storage=w2_fp4,
        w2_scale_storage=down_sf_contiguous,
    )
    # Keep references to prevent GC of contiguous buffers
    views._w13_sf_storage = w13_sf_contiguous
    views._down_sf_storage = down_sf_contiguous
    _WEIGHT_CACHE[key] = views
    return views


# ---------------------------------------------------------------------------
# Micro kernel compilation cache
# ---------------------------------------------------------------------------
_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_micro_kernel(
    state_E: int,
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
    single_token: bool = False,
    mac_override: int | None = None,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    device: torch.device | None = None,
):
    """Compile (or retrieve cached) the SM120 micro MoE kernel.

    Uses the upstream MoEMicroKernelBackend harness: construct the
    activation-specialized backend, run ``configure(...)`` to bind the problem
    shape (and compute ``grid_x``), then ``cute.compile`` the pointer-taking
    ``__call__``. Returns ``(compiled, kernel)``; the kernel instance carries
    ``grid_x`` for the launch path.
    """
    sf_vec_size = 16
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _resolve_swiglu_params(
        activation, swiglu_limit, swiglu_alpha, swiglu_beta
    )

    cache_key = (
        "micro",
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        topk_ids_dtype,
        fast_math,
        share_input_across_experts,
        share_expert_scales,
        single_token,
        mac_override,
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
    )
    cached = _MICRO_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    micro_cls = _resolve_micro_cls(activation)
    mma_tiler_mn = (128, 128)
    output_tile_count_n = max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1])
    # Gated subclasses (silu/swigluoai) accept swiglu params; relu2 does not.
    extra_kwargs = (
        dict(
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
        if is_gated_moe_activation(activation)
        else {}
    )
    kernel: Any = micro_cls(
        sf_vec_size,
        mma_tiler_mn,
        output_tile_count_n,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=single_token,
        **extra_kwargs,
    )
    # configure() must run before cute.compile so the shape config and grid_x
    # are bound into the kernel instance.
    kernel.configure(
        m,
        k,
        n,
        num_topk,
        weight_E,
        max_active_ctas=mac_override,
        device=device,
    )

    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    # The micro __call__ takes raw pointers for every operand plus two int32
    # barrier tensors. Build fake pointers/tensors matching its signature.
    x_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    w1_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    w1s_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    w1a_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    a1_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    a2_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    inter_fake = make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)
    w2_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    w2s_fake = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    w2a_fake = make_ptr(alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    tid_fake = make_ptr(
        topk_ids_cutlass_dtype,
        topk_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=topk_ids_align,
    )
    tw_fake = make_ptr(alpha_dtype, 4, cute.AddressSpace.gmem, assumed_align=4)
    out_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )

    compiled = cute.compile(
        kernel,
        x_fake,
        w1_fake,
        w1s_fake,
        w1a_fake,
        a1_fake,
        a2_fake,
        inter_fake,
        w2_fake,
        w2s_fake,
        w2a_fake,
        tid_fake,
        tw_fake,
        out_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        m,
        kernel.grid_x,
        current_cuda_stream(),
        options="--opt-level 2 --enable-tvm-ffi",
    )

    result = (compiled, kernel)
    _MICRO_KERNEL_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def _expand_to_experts(t: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Broadcast a scalar or [1] tensor to [num_experts]."""
    if t.numel() == 1:
        return t.expand(num_experts).contiguous()
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
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    activation_precision: str = "fp4",
) -> torch.Tensor:
    """Launch the SM120 "static" band via the MoEMicroKernelBackend harness.

    Upstream b12x folded the legacy tensor-core static backend into the
    compact micro kernel, which now handles the whole small/medium decode
    band (the dispatcher only routes here for routed-pair counts the micro
    kernel reports as supported). The kernel reads the routed activations,
    packed FP4 weights, and swizzled block scales directly and scatters the
    weighted result into ``scatter_output``.
    """
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 static launcher"
        )

    # Flatten routing tensors. The micro kernel consumes the routed expert ids
    # and per-pair weights directly (no compaction pre-pass).
    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)

    # Capture whether the FC1-input / FC2-input scales were single shared
    # scalars BEFORE expansion: the ReLU2 single-token shared-scale micro
    # specializations only apply when every expert sees the same scale.
    input_gs_is_shared = input_gs.numel() == 1
    down_input_scale_is_shared = down_input_scale.numel() == 1

    # Broadcast scalar scales to per-expert [E] tensors (the kernel indexes
    # input_gs / down_input_scale by physical expert id).
    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    share_input_across_experts = (
        activation == "relu2"
        and num_tokens == 1
        and input_gs_is_shared
        and _MICRO_SHARE_INPUT_ACROSS_EXPERTS
    )
    share_expert_scales = (
        activation == "relu2" and input_gs_is_shared and down_input_scale_is_shared
    )

    compiled, kernel = _get_micro_kernel(
        workspace.state_E,
        num_experts,
        num_tokens,
        k,
        n,
        top_k,
        topk_ids_dtype=flat_ids.dtype,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        share_expert_scales=share_expert_scales,
        single_token=num_tokens == 1,
        activation=activation,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        device=a.device,
    )

    # FC1->FC2 intermediate scratch (uint32 packed FP4). Sized from the
    # kernel's bound shape config and cached on the workspace so repeated
    # launches (and CUDA-graph replay) reuse the same buffer.
    inter_elems = int(num_tokens) * int(kernel._cfg.inter_u32)
    inter_fp32 = getattr(workspace, "_micro_inter", None)
    if inter_fp32 is None or inter_fp32.numel() < inter_elems:
        inter_fp32 = torch.empty(inter_elems, dtype=torch.uint32, device=a.device)
        workspace._micro_inter = inter_fp32  # type: ignore[attr-defined]

    # The compiled __call__ takes the operands as DataPointer params, which the
    # tvm-ffi binding accepts as raw int addresses (matching the dynamic launch
    # path). MoEMicroKernelBackend.launch wraps the same arguments in cute
    # Pointer objects, which this tvm-ffi build rejects, so call the compiled
    # function directly with .data_ptr() addresses in the launch() arg order.
    _stream = current_cuda_stream()
    compiled(
        a.data_ptr(),
        weights.w1_storage.data_ptr(),
        weights._w13_sf_storage.data_ptr(),
        weights.w1_alpha.data_ptr(),
        input_gs.data_ptr(),
        down_input_scale.data_ptr(),
        inter_fp32.data_ptr(),
        weights.w2_storage.data_ptr(),
        weights._down_sf_storage.data_ptr(),
        weights.w2_alpha.data_ptr(),
        flat_ids.data_ptr(),
        flat_weights.data_ptr(),
        scatter_output.data_ptr(),
        workspace.barrier_count,
        workspace.barrier_epoch,
        cutlass.Int32(int(num_tokens)),
        cutlass.Int32(int(kernel.grid_x)),
        _stream,
    )

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
) -> str:
    """Pick static or dynamic backend based on routed-pair count."""
    mode = _normalize_quant_mode(quant_mode, activation_precision)
    if mode == "w4a16":
        return "w4a16"
    routed_rows = num_tokens * num_topk
    if routed_rows <= _get_static_compact_cutover_pairs("fp4"):
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
    expert_write_rows: torch.Tensor
    expert_tile_base: torch.Tensor
    pair_head: torch.Tensor
    producers_done_count: torch.Tensor
    all_work_published: torch.Tensor
    task_head: torch.Tensor
    task_tail: torch.Tensor
    task_ready: torch.Tensor
    task_expert: torch.Tensor
    task_m_tile: torch.Tensor
    task_slice_begin: torch.Tensor
    task_slice_count: torch.Tensor
    task_valid_rows: torch.Tensor
    tile_write_count: torch.Tensor

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
) -> Sm120DynamicMoEWorkspace:
    """Allocate workspace buffers for the SM120 dynamic MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "allocate_sm120_dynamic_workspace only supports quant_mode='nvfp4'; "
            "use allocate_sm120_moe_workspace(..., quant_mode='w4a16') for W4A16."
        )
    tile_m = _level_tile_m(activation_precision)
    physical_tiles, _, max_tasks = _dynamic_task_geometry(
        state_E,
        n,
        routed_rows,
        tile_m=tile_m,
        tile_n=_level_tile_n(activation_precision),
    )
    rows_padded = physical_tiles * tile_m
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)
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
        routed_rows_capacity=routed_rows,
        physical_tiles_capacity=physical_tiles,
        task_capacity=max_tasks,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(rows_padded, dtype=torch.int32, device=device),
        token_weights=torch.zeros(rows_padded, dtype=torch.float32, device=device),
        packed_input=packed_input,
        packed_input_scale=torch.empty(
            rows_padded, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        expert_write_rows=torch.zeros(state_E, dtype=torch.int32, device=device),
        expert_tile_base=torch.zeros(state_E + 1, dtype=torch.int32, device=device),
        pair_head=torch.zeros(1, dtype=torch.int32, device=device),
        producers_done_count=torch.zeros(1, dtype=torch.int32, device=device),
        all_work_published=torch.zeros(1, dtype=torch.int32, device=device),
        task_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_tail=torch.zeros(1, dtype=torch.int32, device=device),
        task_ready=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_expert=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_m_tile=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_begin=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_count=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_valid_rows=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),
    )

    # Finalize views
    sf_dtype = cutlass.Float8E4M3FN
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

    def __init__(self, kernel, k, num_topk, activation_precision: str = "fp4"):
        activation_precision = _normalize_activation_precision(activation_precision)
        if activation_precision == "bf16":
            raise ValueError(
                "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic launcher wrapper"
            )
        self._kernel = kernel
        self._k = k
        self._packed_storage_cols = k // 2
        self._num_topk = num_topk
        self._cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

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
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready_ptr: cute.Pointer,
        task_expert_ptr: cute.Pointer,
        task_m_tile_ptr: cute.Pointer,
        task_slice_begin_ptr: cute.Pointer,
        task_slice_count_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        tile_write_count_ptr: cute.Pointer,
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
        max_phys_tiles: cutlass.Int32,
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
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((rows_padded * self._cols_pad_k,), stride=(1,)),
        )
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_ready = cute.make_tensor(
            task_ready_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_m_tile = cute.make_tensor(
            task_m_tile_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_begin = cute.make_tensor(
            task_slice_begin_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_count = cute.make_tensor(
            task_slice_count_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        tile_write_count = cute.make_tensor(
            tile_write_count_ptr,
            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),
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
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
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
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    activation_precision: str = "fp4",
    share_input_across_experts: bool = False,
):
    """Compile (or retrieve cached) the SM120 dynamic MoE kernel."""
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _resolve_swiglu_params(
        activation, swiglu_limit, swiglu_alpha, swiglu_beta
    )
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic compiler"
        )
    share_input_across_experts = bool(
        share_input_across_experts and activation_precision == "fp4"
    )
    sf_vec_size = 16
    sm_count = get_num_sm(torch.device("cuda"))
    mac = min(get_max_active_clusters(1), sm_count)
    mma_tiler_mn = (
        _level_tile_m(activation_precision),
        _level_tile_n(activation_precision),
    )

    cache_key = (
        "dynamic",
        activation_precision,
        # E omitted: num_experts is a dynamic (SymInt) operand dim, so one
        # compiled dynamic kernel serves any expert count (see note below).
        k,
        n,
        num_topk,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
        swiglu_limit,
        swiglu_alpha,
        swiglu_beta,
        share_input_across_experts,
    )
    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    is_gated = is_gated_moe_activation(activation)
    w1_rows = (2 if is_gated else 1) * n

    scratch_dtype = cutlass.Float4E2M1FN
    weight_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    dynamic_cls = _resolve_dynamic_cls(activation)
    # Gated subclasses (silu/swigluoai) accept swiglu params; relu2 does not.
    extra_kwargs = (
        dict(
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
        if is_gated
        else {}
    )
    kernel: Any = dynamic_cls(
        sf_vec_size,
        mma_tiler_mn,
        fast_math=fast_math,
        share_input_across_experts=share_input_across_experts,
        **extra_kwargs,
    )
    launch = _DynamicMoELaunch(
        kernel,
        k=k,
        num_topk=num_topk,
        activation_precision=activation_precision,
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
    producers_done_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    all_work_published_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )

    task_ready_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_expert_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_m_tile_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_begin_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_valid_rows_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    tile_write_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )

    # num_experts is passed as a dynamic (SymInt) operand dim so a single
    # compiled dynamic kernel serves any expert count — the kernel reads
    # num_experts = Int32(row_counts.shape[0]) at runtime and loops dynamically,
    # so E never needs to be a compile-time constant. This avoids recompiling per
    # expert count (e.g. 256 vs 384) with no measurable kernel-perf cost
    # (validated on GB10: min/p10 latency unchanged vs a static-E build).
    #
    # NOTE: k and n are deliberately left static. Marking them SymInt compiles and
    # binds fine (smem is tile-shaped, not k/n-shaped), but the kernel rebuilds
    # the block-scale (SF) gmem layouts via tile_atom_to_shape_SF(packed_a.shape /
    # b_w13.shape) from pointer operands, and those freeze at the traced k/n -- so
    # a kernel traced at one (k,n) reads scales from wrong offsets at another.
    # Verified in isolation: same-n/different-k reuse passes at the traced k but
    # drops to ~50% within tolerance at a different k (and k+n reuse NaNs).
    # Genericizing k/n would require making the SF layouts SymInt-correct (kernel
    # surgery + upstream divergence); not worth it. num_experts (E) is purely a
    # batch/stride/loop dim with no such dependency, so it stays dynamic.
    E_sym = sym_int(32, divisibility=1, symbol="moe_E")
    Ep1_sym = sym_int(32, divisibility=1, symbol="moe_Ep1")
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (w1_rows, k, E_sym),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        weight_dtype,
        (k, n, E_sym),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E_sym,), assumed_align=4
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E_sym,), assumed_align=4
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (Ep1_sym,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E_sym,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E_sym,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E_sym,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E_sym,), assumed_align=16
    )
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    compiled = cute.compile(
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
        producers_done_count_fake,
        all_work_published_fake,
        task_head_fake,
        task_tail_fake,
        task_ready_fake,
        task_expert_fake,
        task_m_tile_fake,
        task_slice_begin_fake,
        task_slice_count_fake,
        task_valid_rows_fake,
        tile_write_count_fake,
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
        1,
        1,  # runtime Int32 placeholders
        mac,
        current_cuda_stream(),
        options="--opt-level 2 --enable-tvm-ffi",
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
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    activation_precision: str = "fp4",
) -> torch.Tensor:
    """Launch the SM120 dynamic MoE kernel."""
    activation_precision = _normalize_activation_precision(activation_precision)
    if activation_precision == "bf16":
        raise ValueError(
            "internal routing error: quant_mode='w4a16' reached the NVFP4 dynamic launcher"
        )
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
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        activation_precision=activation_precision,
        share_input_across_experts=input_gs_is_shared,
    )

    # Dynamic kernel: runtime-shaped args are DataPointer (pass data_ptr()),
    # fixed-shape args are Tensor (pass torch tensor directly).
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
        workspace.producers_done_count,
        workspace.all_work_published,
        workspace.task_head,
        workspace.task_tail,
        workspace.task_ready.data_ptr(),
        workspace.task_expert.data_ptr(),
        workspace.task_m_tile.data_ptr(),
        workspace.task_slice_begin.data_ptr(),
        workspace.task_slice_count.data_ptr(),
        workspace.task_valid_rows.data_ptr(),
        workspace.tile_write_count.data_ptr(),
        weights.w13_fp4,
        weights._w13_sf_storage.data_ptr(),
        weights.down_fp4,
        weights._down_sf_storage.data_ptr(),
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
        workspace.physical_tiles_capacity * _level_tile_m(activation_precision),
        workspace.task_capacity,
        workspace.physical_tiles_capacity,
    )
    compiled(*runtime_args, current_cuda_stream())

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
    for block_size in _W4A16_ALLOWED_ROUTED_SIZES:
        slots = max_packed_route_slots(
            int(routed_rows),
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
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
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
        activation_amax=activation_amax,
        layer_idx=layer_idx,
        fast_math=fast_math,
    )


# ==========================================================================
# W4A8 throughput tier (mx) dispatch
# ==========================================================================
# Routed-rows floor above which dynamic-band w4a8_mx calls dispatch to the
# dedicated W4A8 throughput-tier pipeline (w4a8/pipeline.py). Ported from
# b12x integration/tp_moe.py: the tier's 48-row expert-run group padding loses
# to the dynamic kernel at few routes/expert, but wins by >5% above ~3840 and
# by ~36-46% at 6144-12288 routed rows. Floor rounded up to 4096. The env flags
# accept both the upstream B12X_* names and the FlashInfer FLASHINFER_B12X_*
# alias (matching _first_env handling elsewhere in this module).
_W4A8_TIER_MIN_ROUTED_ROWS_DEFAULT = 4096
_W4A8_TIER_MIN_ROUTED_ROWS_ENVS = (
    "FLASHINFER_B12X_W4A8_TIER_MIN_ROUTED_ROWS",
    "B12X_W4A8_TIER_MIN_ROUTED_ROWS",
)
_W4A8_TIER_FORCE_ENVS = (
    "FLASHINFER_B12X_MOE_FORCE_W4A8_TIER",
    "B12X_MOE_FORCE_W4A8_TIER",
)


def _w4a8_tier_min_routed_rows() -> int:
    raw = _first_env(*_W4A8_TIER_MIN_ROUTED_ROWS_ENVS)
    if raw is None or not raw.strip():
        return _W4A8_TIER_MIN_ROUTED_ROWS_DEFAULT
    return int(raw)


def _w4a8_tier_forced() -> bool:
    raw = _first_env(*_W4A8_TIER_FORCE_ENVS)
    if raw is None:
        return False
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


# Prepared (repacked) W4A8 tier weights, keyed on source data_ptrs + shapes,
# mirroring _W4A16_WEIGHT_CACHE / _WEIGHT_CACHE. Stores (sources, prepared)
# where prepared is None when the tier cannot serve these weights (non-unit
# alphas) so the refusal decision is cached without re-checking on the GPU.
_W4A8_TIER_WEIGHT_CACHE: Dict[Tuple, Tuple[Tuple, Optional[dict]]] = {}
# Capacity-sized pipeline workspaces, keyed on (m, k, n, weight_E, topk, device).
_W4A8_TIER_WORKSPACE_CACHE: Dict[Tuple, dict] = {}


def _get_w4a8_tier_prepared(
    w13_fp4: torch.Tensor,
    w13_mx: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_mx: torch.Tensor,
    w2_alpha: torch.Tensor,
) -> Optional[dict]:
    """Repack (once) kernel-order FC1/FC2 storage for the W4A8 tier GEMM.

    ``w13_fp4`` is [E, 2n, K/2] u8 (kernel order [up; gate]); ``w13_mx`` is
    [E, 2n, K/32] u8; ``w2_fp4`` is [E, K, n/2] u8; ``w2_mx`` is [E, K, n/32]
    u8 — the exact fp4_e8m0_k32 source layout the pipeline consumes. Returns
    None when the tier cannot serve these weights (non-unit alphas: the tier
    GEMM has no per-expert alpha operand in v1).
    """
    from .w4a8.pipeline import prepare_w4a8_tier_weights

    key = (
        w13_fp4.data_ptr(),
        w13_mx.data_ptr(),
        w2_fp4.data_ptr(),
        w2_mx.data_ptr(),
        w1_alpha.data_ptr(),
        w2_alpha.data_ptr(),
        tuple(w13_fp4.shape),
        tuple(w2_fp4.shape),
    )
    cached = _W4A8_TIER_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached[1]
    sources = (w13_fp4, w13_mx, w2_fp4, w2_mx, w1_alpha, w2_alpha)
    # One-time host check (syncs): v1 consumes MXFP4 sources exactly, whose
    # per-expert weight global dequant scales are ones by contract.
    unit_alphas = bool(
        torch.all(w1_alpha == 1.0).item() and torch.all(w2_alpha == 1.0).item()
    )
    if not unit_alphas:
        _W4A8_TIER_WEIGHT_CACHE[key] = (sources, None)
        return None
    prepared = prepare_w4a8_tier_weights(
        w13_fp4.view(torch.uint8),
        w13_mx.view(torch.uint8),
        w2_fp4.view(torch.uint8),
        w2_mx.view(torch.uint8),
    )
    _W4A8_TIER_WEIGHT_CACHE[key] = (sources, prepared)
    return prepared


def _maybe_launch_w4a8_tier(
    *,
    a: torch.Tensor,
    w13_fp4: torch.Tensor,
    w13_mx: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_mx: torch.Tensor,
    w2_alpha: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: Optional[torch.Tensor],
    m: int,
    k: int,
    n: int,
    weight_E: int,
    num_topk: int,
    routed_rows: int,
    activation: str,
    quant_mode: str,
    apply_router_weight_on_input: bool,
    device: torch.device,
    prepared_w4a8: object | None = None,
    swiglu_limit: float | None = None,
) -> Optional[torch.Tensor]:
    """Route a dynamic-band w4a8_mx call to the W4A8 throughput-tier pipeline.

    Returns the output tensor when the tier handled the call, or None when the
    call should fall through to the dynamic kernel instead. When an explicit
    ``prepared_w4a8`` object is supplied the refusal conditions become hard
    errors (``RuntimeError("B12X prepared W4A8 MoE cannot run: ...")``).

    Ported from b12x integration/tp_moe.py ``_maybe_launch_w4a8_tier``. v1
    refusal conditions: non-"w4a8_mx" mode; tier disabled; routed_rows below
    the floor (unless forced or a prepared object was supplied);
    activation != "silu"; apply_router_weight_on_input; k % 256 or n % 128;
    non-rank-2 routing tensors; non-unit w1/w2 alphas.
    """
    if quant_mode != "w4a8_mx":
        return None
    prepared_required = prepared_w4a8 is not None

    def reject_prepared_w4a8(reason: str) -> None:
        raise RuntimeError(f"B12X prepared W4A8 MoE cannot run: {reason}")

    min_routed = _w4a8_tier_min_routed_rows()
    if min_routed == 0:
        if prepared_required:
            reject_prepared_w4a8(
                "B12X_W4A8_TIER_MIN_ROUTED_ROWS=0 disables the required W4A8 tier"
            )
        return None
    if routed_rows < min_routed and not prepared_required and not _w4a8_tier_forced():
        return None
    if activation != "silu":
        if prepared_required:
            reject_prepared_w4a8(
                f"W4A8 tier supports activation='silu', got {activation!r}"
            )
        return None
    if apply_router_weight_on_input:
        if prepared_required:
            reject_prepared_w4a8(
                "apply_router_weight_on_input is unsupported by the W4A8 tier"
            )
        return None
    if k % 256 != 0 or n % 128 != 0:
        if prepared_required:
            reject_prepared_w4a8(
                f"W4A8 tier requires k % 256 == 0 and n % 128 == 0; got k={k}, n={n}"
            )
        return None
    if topk_ids.dim() != 2 or topk_weights.dim() != 2:
        if prepared_required:
            reject_prepared_w4a8(
                "topk_ids and topk_weights must both be rank-2 tensors"
            )
        return None

    from .w4a8.pipeline import build_w4a8_tier_workspace, w4a8_tier_forward

    capturing = _is_cuda_graph_capturing()
    if prepared_w4a8 is not None:
        prepared = {
            "w13_rp": prepared_w4a8.w13_rp,  # type: ignore[attr-defined]
            "w13_sfb": prepared_w4a8.w13_sfb,  # type: ignore[attr-defined]
            "w2_rp": prepared_w4a8.w2_rp,  # type: ignore[attr-defined]
            "w2_sfb": prepared_w4a8.w2_sfb,  # type: ignore[attr-defined]
        }
    else:
        prepared = _get_w4a8_tier_prepared(
            w13_fp4,
            w13_mx,
            w1_alpha,
            w2_fp4,
            w2_mx,
            w2_alpha,
        )
    if prepared is None:
        # Non-unit alphas (only reachable on the non-prepared path; the
        # prepared path folds alphas at prepare time). Fall through.
        return None

    ws_key = (m, k, n, weight_E, num_topk, str(device))
    ws = _W4A8_TIER_WORKSPACE_CACHE.get(ws_key)
    if capturing and (ws is None or "fc2" not in ws["_compiled"]):
        raise RuntimeError(
            "CUDA graph capture requires a warmed w4a8 tier workspace; "
            "run one eager call for this shape before capture"
        )
    if ws is None:
        ws = build_w4a8_tier_workspace(
            m=m,
            hidden_size=k,
            intermediate_size=n,
            num_experts=weight_E,
            topk=num_topk,
            device=device,
        )
        _W4A8_TIER_WORKSPACE_CACHE[ws_key] = ws

    if output is None:
        if capturing:
            raise ValueError("CUDA graph capture requires a caller-owned output buffer")
        output = torch.empty(m, k, dtype=a.dtype, device=device)
    if tuple(output.shape) != (m, k):
        raise ValueError(f"output must have shape {(m, k)}, got {tuple(output.shape)}")
    if output.dtype != a.dtype:
        raise ValueError(f"output must have dtype {a.dtype}, got {output.dtype}")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")

    if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
        if capturing:
            raise ValueError(
                "CUDA graph capture requires contiguous int32 topk_ids on the "
                "w4a8 tier path"
            )
        topk_ids = topk_ids.to(torch.int32).contiguous()
    if topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous():
        if capturing:
            raise ValueError(
                "CUDA graph capture requires contiguous float32 topk_weights "
                "on the w4a8 tier path"
            )
        topk_weights = topk_weights.to(torch.float32).contiguous()

    return w4a8_tier_forward(
        a,
        prepared["w13_rp"],
        prepared["w13_sfb"],
        prepared["w2_rp"],
        prepared["w2_sfb"],
        topk_ids,
        topk_weights,
        ws,
        out=output,
        swiglu_limit=swiglu_limit,
    )


# ==========================================================================
# Workspace cache (for functional API path)
# ==========================================================================

_Sm120Workspace = Union[
    Sm120StaticMoEWorkspace,
    Sm120DynamicMoEWorkspace,
    Sm120W4A16MoEWorkspace,
]

# Keyed by (state_E, weight_E, k, n, top_k, device, backend).
# Stores the workspace with the largest max_rows seen for each key.
# Grows monotonically — never shrinks within a process.
_WORKSPACE_CACHE: Dict[Tuple, _Sm120Workspace] = {}


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
    if backend is None:
        backend = select_sm120_moe_backend(
            num_tokens=max(
                1, (capacity_rows + max(1, int(num_topk)) - 1) // max(1, int(num_topk))
            ),
            num_topk=int(num_topk),
            activation_precision=activation_precision,
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
    )
    cached = _WORKSPACE_CACHE.get(cache_key)

    if cached is not None:
        if isinstance(cached, (Sm120DynamicMoEWorkspace, Sm120W4A16MoEWorkspace)):
            if cached.routed_rows_capacity >= max(1, routed_rows):  # type: ignore[union-attr]
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
def launch_sm120_moe(
    *,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor] = None,
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
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
    source_format: str = "modelopt",
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    _workspace=None,
    _weight_views=None,
    _prepared_weights=None,
    _prepared_w4a8=None,
) -> torch.Tensor:
    """Unified SM120 MoE dispatch — selects static or dynamic by token count.

    Optional _workspace and _weight_views can be pre-allocated and reused
    across calls to avoid per-call allocation overhead (wrapper path).
    When not provided (functional API path), a module-level workspace cache
    is used to avoid re-allocating on every call.
    """
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)
    activation_precision = _activation_precision_from_quant_mode(quant_mode)

    # Validate the activation name (unknown activations raise) and normalize
    # the swiglu params (swigluoai defaults: limit=7.0, alpha=1.702, beta=1.0;
    # silu/relu2: None/1.0/0.0). swiglu_alpha/beta are not exposed by this
    # public entry point; they ride the activation defaults.
    activation = normalize_moe_activation(activation)
    swiglu_limit, swiglu_alpha, swiglu_beta = _resolve_swiglu_params(
        activation, swiglu_limit
    )

    num_tokens = topk_ids.size(0)
    k = a.size(1)  # hidden_size
    is_gated = is_gated_moe_activation(activation)
    # w1_weight.size(1) is 2*n for gated or n for non-gated
    intermediate_size = w1_weight.size(1) // 2 if is_gated else w1_weight.size(1)
    n = intermediate_size
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

    if quant_mode == "w4a8_nvfp4":
        # w4a8_nvfp4 needs residual scale grids and per-expert alphas the tier
        # GEMM does not implement, and the non-tier w4a8_nvfp4 dynamic path is
        # not yet wired through this dispatcher. Fail loudly rather than run a
        # silently-wrong NVFP4 kernel.
        raise NotImplementedError("quant_mode='w4a8_nvfp4' dispatch not yet wired")

    if quant_mode == "w4a8_mx":
        # The W4A8 throughput tier serves the large-routed ("dynamic") band.
        # Weights arrive in the fp4_e8m0_k32 source layout: w1_weight
        # [E, 2n, K/2] u8, w1_weight_sf [E, 2n, K/32] u8, w2_weight [E, K, n/2]
        # u8, w2_weight_sf [E, K, n/32] u8.
        tier_out = _maybe_launch_w4a8_tier(
            a=a,
            w13_fp4=w1_weight,
            w13_mx=w1_weight_sf,
            w1_alpha=w1_alpha,
            w2_fp4=w2_weight,
            w2_mx=w2_weight_sf,
            w2_alpha=w2_alpha,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=scatter_output,
            m=num_tokens,
            k=k,
            n=n,
            weight_E=num_experts,
            num_topk=top_k,
            routed_rows=routed_rows,
            activation=activation,
            quant_mode=quant_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            device=a.device,
            prepared_w4a8=_prepared_w4a8,
            swiglu_limit=swiglu_limit,
        )
        if tier_out is not None:
            return tier_out
        # Below the tier floor (micro-eligible / small-routed w4a8_mx) the
        # non-tier mxfp8-activation kernel path is not yet wired through this
        # dispatcher (owned by the broader b12x rebase). Fail loudly.
        raise NotImplementedError(
            "quant_mode='w4a8_mx' below the throughput-tier floor "
            f"(routed_rows={routed_rows} < {_w4a8_tier_min_routed_rows()}); "
            "the non-tier w4a8_mx dispatch path is not yet wired. Set "
            "B12X_MOE_FORCE_W4A8_TIER=1 to force the tier, or raise the "
            "routed-row count above the floor."
        )

    if fc2_input_scale is None:
        raise ValueError("fc2_input_scale is required when quant_mode='nvfp4'.")
    down_input_scale = fc2_input_scale

    weights = (
        _weight_views
        if _weight_views is not None
        else _get_weight_views(
            w1_fp4=w1_weight,
            w1_blockscale=w1_weight_sf,
            w2_fp4=w2_weight,
            w2_blockscale=w2_weight_sf,
            w1_alphas=w1_alpha,
            w2_alphas=w2_alpha,
            n=n,
            k=k,
            activation_precision=activation_precision,
            activation=activation,
        )
    )

    # Resolve workspace and backend selection.
    # When a pre-allocated workspace is provided (CUDA graph wrapper path),
    # infer the backend from the workspace type so they stay in sync —
    # the caller already committed to a backend at allocation time.
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
        if isinstance(workspace, Sm120DynamicMoEWorkspace):
            if num_local_experts != num_experts:
                raise ValueError(
                    "pre-allocated dynamic SM120 MoE workspace requires "
                    "num_local_experts == num_experts because dynamic expert "
                    "buffers are indexed by global topk ids."
                )
            backend = "dynamic"
        else:
            backend = "static"
    else:
        backend = select_sm120_moe_backend(
            num_tokens=num_tokens,
            num_topk=top_k,
            activation_precision=activation_precision,
        )
        # The "static" band is now served by the compact micro kernel, which
        # only handles 1 <= num_tokens <= 8 (it keeps the tokens' activations
        # resident). For shapes it cannot support, fall back to the dynamic
        # grouped-GEMM backend.
        micro_cls = _resolve_micro_cls(activation)
        if backend == "static" and not micro_cls.is_supported(  # type: ignore[attr-defined]
            num_tokens, k, n, top_k, num_experts
        ):
            backend = "dynamic"
        # The dynamic kernel indexes row_counts/expert_write_rows directly with
        # topk_ids but those buffers are sized with num_local_experts. Unless
        # num_local_experts == num_experts the per-expert buffers do not line
        # up; the micro path indexes weights by physical expert id and is the
        # only backend here that tolerates num_local_experts != num_experts.
        if backend == "dynamic" and num_local_experts != num_experts:
            backend = "static"
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

    if backend == "dynamic":
        return launch_sm120_dynamic_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=w1_alpha,
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
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            activation_precision=activation_precision,
        )
    else:
        return launch_sm120_static_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=w1_alpha,
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
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            activation_precision=activation_precision,
        )
