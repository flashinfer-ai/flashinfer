"""Public Cake FMHA product entrypoints.

The conventional decode/context APIs are complete-domain Cake routes.  DCP
metadata selects an authenticated additive profile through the same
decode entrypoint and does not change ordinary-call behavior.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import torch
import tvm_ffi

from .jit.cake_fmha import (
    CakeFmhaTarget,
    get_cake_fmha_manifest,
    _is_cake_fmha_decode_native_bf16_available,
    load_cake_fmha_context_bf16_module,
    load_cake_fmha_context_fp8_module,
    load_cake_fmha_context_nvfp4_module,
    load_cake_fmha_context_fp16_hd256_module,
    load_cake_fmha_context_fp8_hd256_module,
    load_cake_fmha_compat_module,
    load_cake_fmha_decode_native_bf16_hd256_smallm_module,
    load_cake_fmha_decode_balanced_bf16_module,
    load_cake_fmha_decode_balanced_fp16_module,
    load_cake_fmha_decode_balanced_fp8_module,
    load_cake_fmha_decode_balanced_hd256_module,
    load_cake_fmha_decode_balanced_hd64_module,
    cake_fmha_balanced_hd64_max_group,
    load_cake_fmha_decode_native_bf16_module,
    load_cake_fmha_decode_native_fp16_hd512_module,
    load_cake_fmha_decode_native_fp16_nhd_module,
    load_cake_fmha_decode_quant_bf16q_module,
    load_cake_fmha_decode_quant_fp8_module,
    load_cake_fmha_decode_quant_nvfp4_module,
)
from .cake_fmha_request_ordered import (
    CakeFmhaRequestOrderedDecodePlan,
    plan_cake_fmha_request_ordered_paged_decode,
)
from .cake_fmha_request_ordered import (
    _fallback_cake_fmha_request_ordered_plan as _fallback_cake_fmha_request_ordered_plan,
)
from .cake_fmha_request_ordered import (
    _run_cake_fmha_request_ordered_paged_decode as _run_cake_fmha_request_ordered_paged_decode,
)
from .utils import get_compute_capability, log2e


@dataclass(frozen=True)
class CakeFmhaDecodeRoute:
    """One exact manifest-backed optimized decode specialization."""

    target: CakeFmhaTarget
    batch_size: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    has_sink: bool
    has_window: bool
    use_scale_ptr: bool
    retain_kv_l2: bool
    component: Literal[
        "decode_aligned_bf16",
        "decode_aligned_fp16",
        "decode_balanced_bf16",
        "decode_balanced_fp16",
        "decode_balanced_fp8",
        "decode_balanced_bf16q",
        "decode_balanced_fp16q",
        "decode_balanced_bf16_hd64",
        "decode_balanced_bf16_hd256",
        "decode_native_bf16",
        "decode_native_bf16_hd256_smallm",
        "decode_native_fp16_hd512",
        "decode_native_fp16_nhd",
        "decode_quant_bf16q",
        "decode_quant_fp8_reduce",
        "decode_quant_fp8",
        "decode_quant_nvfp4",
    ] = "decode_native_bf16"
    page_size: int = 16
    # Small-M hd256 speculative decode: one resident wave of Split-KV CTAs.
    num_split: int = 0
    # The caller passed an ``lse`` output.  The exact-shape decode-native BF16
    # members never write it, so such a route is served by the generic member.
    return_lse: bool = False
    # Aligned hd128 q1 decode: the identifier of the generated program
    # (``flashinfer.jit.cake_fmha_aligned.ALIGNED_C<cluster>_<DTYPE>``) whose
    # cluster size ``aligned_cluster_size`` selected; ``None`` on every other route.
    aligned_program: str | None = None


CakeFmhaContextComponent = Literal[
    "context_bf16",
    "context_fp16_hd256",
    "context_fp8",
    "context_fp8_hd256",
    "context_nvfp4",
]


@dataclass(frozen=True)
class CakeFmhaContextRoute:
    """One exact manifest-backed optimized context specialization."""

    target: CakeFmhaTarget
    component: CakeFmhaContextComponent
    num_m_blocks: int
    num_q_heads: int
    num_kv_heads: int
    pack_g: int
    page_size: int
    l2_swizzle: int
    is_causal: bool
    return_lse: bool
    enable_sink: bool
    exact_profile: Literal["q511", "q257"] | None = None


# These are the exact optimized routes in the pinned 57,280-cell matrix.  Keep
# the component sequences explicit: hd256 needs its staging/scatter support,
# and split-KV NVFP4 decode may also need the shared reduction component.
_PRODUCT_ROUTE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "ctx_bf16_hnd_hd128_hgpack_03df_v2": ("context_bf16",),
    "ctx_fp16_nhd_hd256_stage16_v1": (
        "context_hd256_support",
        "context_fp16_hd256",
    ),
    "ctx_fp8_bf16_nhd_hd256_stage16_v1": (
        "context_hd256_support",
        "context_fp8_hd256",
    ),
    "ctx_fp8_hnd_hd128_hgpack_48b5_v1": ("context_fp8",),
    "ctx_nvfp4_hnd_hd128_dequant_fp8_hg_v1": ("context_nvfp4",),
    "decode_balanced_bf16_v1": (
        "decode_balanced_bf16",
        "decode_balanced_bf16_mtp_n32",
        "decode_balanced_bf16_mtp_n64",
    ),
    "decode_balanced_fp16_v1": (
        "decode_balanced_fp16",
        "decode_balanced_fp16_mtp_n32",
        "decode_balanced_fp16_mtp_n64",
    ),
    "decode_balanced_fp8_v1": ("decode_balanced_fp8",),
    "decode_balanced_bf16q_v1": ("decode_balanced_bf16q",),
    "decode_balanced_fp16q_v1": ("decode_balanced_fp16q",),
    "decode_balanced_bf16_hd64_v1": (
        "decode_balanced_bf16_hd64",
        "decode_balanced_bf16_hd64_g16",
    ),
    "decode_balanced_bf16_hd256_v1": (
        "decode_balanced_bf16_hd256_p16",
        "decode_balanced_bf16_hd256_p32",
        "decode_balanced_bf16_hd256_p64",
    ),
    "decode_native_bf16_v1_bece": ("decode_native_bf16",),
    "decode_native_bf16_hd256_smallm_v1": (
        "decode_native_bf16_hd256_smallm_n32_p16",
        "decode_native_bf16_hd256_smallm_n32_p32",
        "decode_native_bf16_hd256_smallm_n32_p64",
        "decode_native_bf16_hd256_smallm_n64_p16",
        "decode_native_bf16_hd256_smallm_n64_p32",
        "decode_native_bf16_hd256_smallm_n64_p64",
    ),
    "decode_native_fp16_hd512_v1_66b1": ("decode_native_fp16_hd512",),
    "decode_native_fp16_nhd_v1_f32d": ("decode_native_fp16_nhd",),
    "decode_quantized_bf16q_9d8b_v1": ("decode_quant_bf16q",),
    "decode_quantized_fp8_8e5b_v1": (
        "decode_quant_fp8",
        "decode_quant_fp8_reduce",
    ),
    "decode_quantized_nvfp4_8e5b_v1": (
        "decode_quant_nvfp4",
        "decode_quant_fp8_reduce",
    ),
}

# These components have an authenticated FlashInfer TVM-FFI adapter in the
# checked-in package.  A route remains fail-closed until every component in its
# declared chain is present here and covered by the adapter digest.
_AUTHENTICATED_JIT_COMPONENTS = frozenset(
    {
        "compat_v1",
        "context_bf16",
        "context_fp16_hd256",
        "context_fp8",
        "context_fp8_hd256",
        "context_hd256_support",
        "context_nvfp4",
        "decode_balanced_bf16",
        "decode_balanced_bf16_mtp_n32",
        "decode_balanced_bf16_mtp_n64",
        "decode_balanced_fp16",
        "decode_balanced_fp16_mtp_n32",
        "decode_balanced_fp16_mtp_n64",
        "decode_balanced_fp8",
        "decode_balanced_bf16q",
        "decode_balanced_fp16q",
        "decode_balanced_bf16_hd64",
        "decode_balanced_bf16_hd64_g16",
        "decode_balanced_bf16_hd256_p16",
        "decode_balanced_bf16_hd256_p32",
        "decode_balanced_bf16_hd256_p64",
        "decode_native_bf16",
        "decode_native_bf16_hd256_smallm_n32_p16",
        "decode_native_bf16_hd256_smallm_n32_p32",
        "decode_native_bf16_hd256_smallm_n32_p64",
        "decode_native_bf16_hd256_smallm_n64_p16",
        "decode_native_bf16_hd256_smallm_n64_p32",
        "decode_native_bf16_hd256_smallm_n64_p64",
        "decode_native_fp16_hd512",
        "decode_native_fp16_nhd",
        "decode_quant_bf16q",
        "decode_quant_fp8",
        "decode_quant_fp8_reduce",
        "decode_quant_nvfp4",
    }
)

# The aligned hd128 q1 decode programs are not registry components: they ship
# as the generated in-tree unit ``csrc/cake_fmha_aligned/`` whose loader
# ``flashinfer.jit.cake_fmha_aligned`` publishes one ``CakeProgram`` constant
# per cluster size and dtype, authenticated by the program's content digest.
# A route's component chain is its one program identifier, and the route is
# optimized exactly when that identifier is one of these.  The table pins the
# delivered record of each identifier: ``(cluster_size, dtype, program name)``;
# a loader whose record differs fails closed at load.
_ALIGNED_PROGRAMS: dict[str, tuple[int, str, str]] = {
    "ALIGNED_C8_BF16": (8, "bf16", "cake_fmha_aligned_a9cc814b948505a2cc1b"),
    "ALIGNED_C4_BF16": (4, "bf16", "cake_fmha_aligned_2bd88c52bb7a16a1814f"),
    "ALIGNED_C2_BF16": (2, "bf16", "cake_fmha_aligned_1052b6ae93d516e78ed6"),
    "ALIGNED_C1_BF16": (1, "bf16", "cake_fmha_aligned_6f88390dacdd946afa74"),
    "ALIGNED_C8_FP16": (8, "fp16", "cake_fmha_aligned_8d67475a5eaad8c4abc2"),
    "ALIGNED_C4_FP16": (4, "fp16", "cake_fmha_aligned_b100f226ebd9ce6884cc"),
    "ALIGNED_C2_FP16": (2, "fp16", "cake_fmha_aligned_6b02ec3e27916ce5eee5"),
    "ALIGNED_C1_FP16": (1, "fp16", "cake_fmha_aligned_300f2001dc3a9f642eeb"),
}
_ALIGNED_COMPONENTS = ("decode_aligned_bf16", "decode_aligned_fp16")
# ``CakeFmhaTarget`` -> the exact ``CakeProgram`` target of the same device.
_ALIGNED_PROGRAM_TARGET: dict[str, str] = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
# The binding's host parameters, in the kernel's order (then the grid).
_ALIGNED_HOST_PARAMS = (
    "Qt",
    "K",
    "V",
    "O",
    "page_table",
    "seq_lens_kv",
    "uniform_seq_len",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "batch_size",
)


def _aligned_program_id(cluster_size: int, dtype: str) -> str:
    return f"ALIGNED_C{cluster_size}_{dtype.upper()}"


def _aligned_route_program(
    route: CakeFmhaDecodeRoute,
) -> tuple[int, str, str] | None:
    """``(cluster_size, dtype, program name)`` of an aligned route's program, or
    ``None`` when the route's program identifier is not one of the module's or
    not of its dtype."""

    if route.component not in _ALIGNED_COMPONENTS or route.aligned_program is None:
        return None
    program = _ALIGNED_PROGRAMS.get(route.aligned_program)
    if program is None or route.component != f"decode_aligned_{program[1]}":
        return None
    return program


def cake_fmha_aligned_launch_grid(
    program, batch_size: int, num_kv_heads: int
) -> tuple[int, int, int]:
    """The launch grid of one aligned hd128 q1 decode: ``(batch_size *
    num_kv_heads, 1, cluster_size)``, one cluster of the program's CTAs along z
    per (request, KV head) tile.  ``program`` is the generated loader's
    ``CakeProgram`` record; its ``cluster`` is ``(1, 1, cluster_size)``."""

    return int(batch_size) * int(num_kv_heads), 1, int(program.cluster[2])


def _route_components(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute,
) -> tuple[str, ...]:
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component in _ALIGNED_COMPONENTS
    ):
        return (route.aligned_program,) if route.aligned_program is not None else ()
    if isinstance(route, CakeFmhaDecodeRoute):
        route_name = {
            "decode_balanced_bf16": "decode_balanced_bf16_v1",
            "decode_balanced_fp16": "decode_balanced_fp16_v1",
            "decode_balanced_fp8": "decode_balanced_fp8_v1",
            "decode_balanced_bf16q": "decode_balanced_bf16q_v1",
            "decode_balanced_fp16q": "decode_balanced_fp16q_v1",
            "decode_balanced_bf16_hd64": "decode_balanced_bf16_hd64_v1",
            "decode_balanced_bf16_hd256": "decode_balanced_bf16_hd256_v1",
            "decode_native_bf16": "decode_native_bf16_v1_bece",
            "decode_native_bf16_hd256_smallm": "decode_native_bf16_hd256_smallm_v1",
            "decode_native_fp16_hd512": "decode_native_fp16_hd512_v1_66b1",
            "decode_native_fp16_nhd": "decode_native_fp16_nhd_v1_f32d",
            "decode_quant_bf16q": "decode_quantized_bf16q_9d8b_v1",
            "decode_quant_fp8": "decode_quantized_fp8_8e5b_v1",
            "decode_quant_nvfp4": "decode_quantized_nvfp4_8e5b_v1",
        }[route.component]
    else:
        route_name = {
            "context_bf16": "ctx_bf16_hnd_hd128_hgpack_03df_v2",
            "context_fp16_hd256": "ctx_fp16_nhd_hd256_stage16_v1",
            "context_fp8": "ctx_fp8_hnd_hd128_hgpack_48b5_v1",
            "context_fp8_hd256": "ctx_fp8_bf16_nhd_hd256_stage16_v1",
            "context_nvfp4": "ctx_nvfp4_hnd_hd128_dequant_fp8_hg_v1",
        }[route.component]
    return _PRODUCT_ROUTE_COMPONENTS[route_name]


def cake_fmha_route_is_optimized(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute | None,
) -> bool:
    """Return whether ``route`` has a fully authenticated runnable adapter."""

    if route is None:
        return False
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component in _ALIGNED_COMPONENTS
    ):
        return _aligned_route_program(route) is not None
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == "decode_native_bf16"
        and not _is_cake_fmha_decode_native_bf16_available(
            route.target,
            route.batch_size,
            route.q_len,
            route.num_q_heads,
            route.num_kv_heads,
            has_sink=route.has_sink,
            has_window=route.has_window,
            use_scale_ptr=route.use_scale_ptr,
            retain_kv_l2=route.retain_kv_l2,
            return_lse=route.return_lse,
        )
    ):
        return False
    return all(
        component in _AUTHENTICATED_JIT_COMPONENTS
        for component in _route_components(route)
    )


def _tma_paged_kv_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether the paged-KV view can be represented by our TMA maps."""

    if tensor.stride(3) != 1:
        return False
    element_size = tensor.element_size()
    return all(
        stride > 0 and stride * element_size % 16 == 0 for stride in tensor.stride()[:3]
    )


def _tma_nvfp4_paged_kv_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether packed HND NVFP4 KV is exactly TMA-encodable."""

    return tensor.stride(3) == 1 and all(
        stride > 0 and stride % 16 == 0 for stride in tensor.stride()[:3]
    )


def _tma_nvfp4_scale_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether HND E4M3 block scales are exactly TMA-encodable."""

    return (
        tensor.stride(3) == 1
        and tensor.stride(2) == 8
        and tensor.stride(1) > 0
        and tensor.stride(0) > 0
        and tensor.stride(1) % 16 == 0
        and tensor.stride(0) % 16 == 0
    )


_SMALLM_MIN_SUPPORTED_SM_COUNT = 148  # B200; used only for non-CUDA capability replays


def _smallm_sm_count(device: torch.device) -> int:
    if device.type != "cuda":
        return _SMALLM_MIN_SUPPORTED_SM_COUNT
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


# On-device load-balanced split-KV decode (Cake routes ``decode_balanced_bf16_v1``
# and ``decode_balanced_fp16_v1``).  The band is host metadata only: the caller's
# KV length bound (``max_seq_len``) and the number of (request, KV head) work
# tiles, per dtype and query-length class.  BF16: for ``q_len == 1`` the static
# grid-stride route keeps every tile whole below the band; the packed MTP tiles
# (``q_len`` 3..8) take the balanced kernel at every shape (1.6-11.8x faster
# than the static route on every measured shape, B200 + GB300).  FP16 (HND,
# page 16, GQA-8) has no other optimized Cake route -- the previous contract was
# the compat fallback -- so its band admits every shape.  The round-2 families
# (E4M3 cache with E4M3 / BF16 / FP16 query, BF16 head_dim 64, BF16 head_dim
# 256 outside the small-M route) likewise admit every shape: their incumbents
# were the quantized full-block routes, the compat kernel, or nothing.  Values
# mirror the Cake dispatcher's per-arch ``ROUTE_BAND`` and were set from paired
# same-input cold-L2 CUPTI measurements on B200 / GB300.
CAKE_FMHA_BALANCED_ROUTE_BAND: dict[str, dict[str, dict[str, tuple[int, int]]]] = {
    "sm100a": {
        "bf16": {"q1": (32768, 9), "mtp": (1, 1)},
        "fp16": {"q1": (1, 1), "mtp": (1, 1)},
        "fp8": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16q": {"q1": (1, 1), "mtp": (1, 1)},
        "fp16q": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd64": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd256": {"q1": (1, 1), "mtp": (1, 1)},
    },  # (min KV length bound, min work tiles)
    "sm103a": {
        "bf16": {"q1": (32768, 9), "mtp": (1, 1)},
        "fp16": {"q1": (1, 1), "mtp": (1, 1)},
        "fp8": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16q": {"q1": (1, 1), "mtp": (1, 1)},
        "fp16q": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd64": {"q1": (1, 1), "mtp": (1, 1)},
        "bf16_hd256": {"q1": (1, 1), "mtp": (1, 1)},
    },
}
CAKE_FMHA_BALANCED_MAX_REQUESTS = 1024
CAKE_FMHA_BALANCED_MTP_Q_LENS = (3, 4, 5, 6, 7, 8)
# head_dim 256: every query row of a request is its own work tile (no packed
# MTP tile), uniform q_len 1..8.
CAKE_FMHA_BALANCED_HD256_MAX_Q_LEN = 8
CAKE_FMHA_BALANCED_HD64_MAX_GROUP = 16
_BALANCED_MAX_BALANCE_FACTOR = 8
_BALANCED_ROW_PARTIAL_O_PER_SLOT = 8 * 128
_BALANCED_ROW_STATS_PER_SLOT = 16
_BALANCED_MTP_PARTIAL_O_PER_SLOT = 64 * 128
_BALANCED_MTP_STATS_PER_SLOT = 128
_BALANCED_HD64_PARTIAL_O_PER_SLOT = 16 * 64  # 16-row tile x head_dim 64
_BALANCED_HD64_STATS_PER_SLOT = 32  # max[16] then sum[16]
_BALANCED_HD256_PARTIAL_O_PER_SLOT = 8 * 256  # 8-row tile x head_dim 256
_BALANCED_HD256_STATS_PER_SLOT = 16  # max[8] then sum[8]


def _balanced_workspace_bounds(sm_count: int) -> tuple[int, int]:
    """``(max_split_items, max_split_tiles)`` of the persistent balanced grid."""

    return (
        2 * _BALANCED_MAX_BALANCE_FACTOR * sm_count,
        _BALANCED_MAX_BALANCE_FACTOR * sm_count,
    )


def _balanced_slot_words(q_len: int, head_dim: int) -> tuple[int, int]:
    """``(partial O floats, statistics floats)`` per split item of one balanced kernel family."""

    if head_dim == 64:
        return _BALANCED_HD64_PARTIAL_O_PER_SLOT, _BALANCED_HD64_STATS_PER_SLOT
    if head_dim == 256:
        return _BALANCED_HD256_PARTIAL_O_PER_SLOT, _BALANCED_HD256_STATS_PER_SLOT
    if head_dim != 128:
        raise ValueError(
            f"balanced decode serves head_dim 64, 128 or 256, got {head_dim}"
        )
    if q_len == 1:
        return _BALANCED_ROW_PARTIAL_O_PER_SLOT, _BALANCED_ROW_STATS_PER_SLOT
    return _BALANCED_MTP_PARTIAL_O_PER_SLOT, _BALANCED_MTP_STATS_PER_SLOT


def _balanced_counters_per_tile(q_len: int, head_dim: int) -> int:
    """Counter words per split tile: four for the packed head_dim-128 MTP tile, else one."""

    return 4 if (head_dim == 128 and q_len != 1) else 1


def _balanced_q_len_supported(q_len: int, head_dim: int) -> bool:
    """Query lengths a balanced kernel family serves (host metadata only)."""

    if head_dim == 128:
        return q_len == 1 or q_len in CAKE_FMHA_BALANCED_MTP_Q_LENS
    if head_dim == 256:
        return 1 <= q_len <= CAKE_FMHA_BALANCED_HD256_MAX_Q_LEN
    return q_len == 1


def cake_fmha_balanced_counter_bytes(
    sm_count: int, q_len: int = 8, head_dim: int = 128
) -> int:
    """Zero-initialized counter bytes the balanced route needs.

    Tile counters (one word per split tile for the row kernels; four for the
    packed head_dim-128 MTP kernel: arrivals, the two reduce-queue words and the
    two-chunk published flag) are followed by the four 16-byte-aligned queue
    counters.  The kernel resets every counter it touched before it exits, so a
    buffer zeroed once at allocation can be reused across launches (the
    trtllm-gen ``multi_ctas_kv_counter_buffer`` contract).  The default
    arguments size the largest family.
    """

    _, max_split_tiles = _balanced_workspace_bounds(sm_count)
    tile_bytes = max_split_tiles * _balanced_counters_per_tile(q_len, head_dim) * 4
    return (tile_bytes + 15) // 16 * 16 + 4 * 4


def cake_fmha_balanced_workspace_bytes(
    sm_count: int, q_len: int, head_dim: int = 128
) -> int:
    """Partial-output/statistics bytes carved from ``workspace_buffer``.

    Per split item one FP32 partial tile (8 x 128 for the head_dim-128 row
    kernels and the E4M3-cache kernels, 64 x 128 for the packed MTP kernel,
    16 x 64 for head_dim 64, 8 x 256 for head_dim 256) and one statistics slot
    (row maxima then row sums), plus the reserved slot that receives the
    device planner's plan facts (chunk pairs, total items).
    """

    max_split_items, _ = _balanced_workspace_bounds(sm_count)
    partial_o, stats = _balanced_slot_words(q_len, head_dim)
    partial_o_bytes = max_split_items * partial_o * 4
    stats_offset = (partial_o_bytes + 255) // 256 * 256
    return stats_offset + (max_split_items + 1) * stats * 4


def _balanced_route_supported(
    target: CakeFmhaTarget,
    *,
    dtype: str,
    batch_size: int,
    q_len: int,
    num_kv_heads: int,
    max_seq_len: int,
    sm_count: int,
    workspace_buffer: torch.Tensor,
    counter_buffer: torch.Tensor | None,
    head_dim: int = 128,
) -> bool:
    """Host-only admission of the on-device load-balanced decode routes.

    ``dtype`` names the band family (``bf16`` / ``fp16`` for head_dim 128,
    ``fp8`` / ``bf16q`` / ``fp16q`` for the E4M3 cache, ``bf16_hd64`` and
    ``bf16_hd256``); ``head_dim`` selects the workspace geometry.
    """

    if not _balanced_q_len_supported(q_len, head_dim):
        return False
    if batch_size > CAKE_FMHA_BALANCED_MAX_REQUESTS:
        return False
    arch_band = CAKE_FMHA_BALANCED_ROUTE_BAND.get(target, {}).get(dtype)
    if arch_band is None:
        return False
    min_kv_len, min_work_tiles = arch_band["q1" if q_len == 1 else "mtp"]
    if max_seq_len < min_kv_len or batch_size * num_kv_heads < min_work_tiles:
        return False
    if not workspace_buffer.is_contiguous():
        return False
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if workspace_bytes < cake_fmha_balanced_workspace_bytes(sm_count, q_len, head_dim):
        return False
    if counter_buffer is not None:
        counter_bytes = counter_buffer.numel() * counter_buffer.element_size()
        if counter_bytes < cake_fmha_balanced_counter_bytes(sm_count, q_len, head_dim):
            return False
    return True


# ---------------------------------------------------------------------------
# Aligned hd128 q1 decode selector: a Python mirror of the source selector.
#
# Source of truth: the kernel source's selector (module
# ``trtllm_fmha_tgv_bf16`` of the ``attention_decode_bf16_paged_static_g``
# family, source revision ``ea3a718663acd2818814c2444aa00537f6b604d6``:
# ``SEL_BAND_*``, ``_c1_balanced`` .. ``_c8_balanced``, ``aligned_cluster_size``).  The
# constants are copied verbatim (names and values) and the function keeps the
# source's signature and semantics; change the source first, then re-copy.
# ``tests/attention/test_cake_fmha_aligned.py`` holds the source selector
# test's expectations.  In this module ``None`` keeps the module's present
# route (``decode_native_bf16`` below the balanced band for BF16,
# ``decode_balanced_fp16`` for FP16); in the source it is the persistent
# balanced program.
# ---------------------------------------------------------------------------
TILE_Q = 8
QKV_DTYPES = ("bf16", "fp16")
# The bands of aligned_cluster_size, on tiles = batch_size * num_kv_heads and max_kv = max_seq_len_kv.
SEL_BAND_C8_TILES_MAX = 8
SEL_BAND_C4_TILES_MAX = 32
SEL_BAND_C2_TILES_MAX = 64
SEL_BAND_MID_TILES_MAX = 128
SEL_BAND_WIDE_TILES_MIN = 512
SEL_BAND_LONG_KV_MIN = 32768
SEL_BAND_LONG_KV_MAX = 65536
# Carve-outs: the shapes of a band on which the persistent balanced program
# measured faster than the band's aligned program on both targets (the rule: no
# regression at any probed point of the carve-out, >= 2 % faster at two thirds
# of them).  ``None`` keeps the aligned program on the whole band.  Lengths are
# ``max_kv``.
# c1: the balanced program for SEL_BAND_C1_BALANCED_KV_MIN <= max_kv <= SEL_BAND_C1_BALANCED_KV_MAX.
SEL_BAND_C1_BALANCED_KV_MAX: int | None = 2048
SEL_BAND_C1_BALANCED_KV_MIN: int | None = 512
# c2: the balanced program for lo <= max_kv <= hi; the wide form (t, lo, hi) adds tiles >= t with lo <= max_kv <= hi.
SEL_BAND_C2_BALANCED_KV: tuple[int, int] | None = (6144, 8192)
SEL_BAND_C2_BALANCED_WIDE: tuple[int, int, int] | None = (56, 4096, 6143)
# c4: the balanced program for max_kv >= the bound (the band's 32768 upper bound stays).
SEL_BAND_C4_BALANCED_KV_MIN: int | None = 16384
# c8: the balanced program for max_kv >= the bound.
SEL_BAND_C8_BALANCED_KV_MIN: int | None = 65536


def _c1_balanced(max_kv: int) -> bool:
    hi, lo = SEL_BAND_C1_BALANCED_KV_MAX, SEL_BAND_C1_BALANCED_KV_MIN
    return hi is not None and (lo or 0) <= max_kv <= hi


def _c2_balanced(tiles: int, max_kv: int) -> bool:
    win, wide = SEL_BAND_C2_BALANCED_KV, SEL_BAND_C2_BALANCED_WIDE
    if win is not None and win[0] <= max_kv <= win[1]:
        return True
    return wide is not None and tiles >= wide[0] and wide[1] <= max_kv <= wide[2]


def _c4_balanced(max_kv: int) -> bool:
    lo = SEL_BAND_C4_BALANCED_KV_MIN
    return lo is not None and max_kv >= lo


def _c8_balanced(max_kv: int) -> bool:
    lo = SEL_BAND_C8_BALANCED_KV_MIN
    return lo is not None and max_kv >= lo


def aligned_cluster_size(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len_kv: int,
    *,
    q_len: int,
    num_q_heads: int,
    dtype: str,
) -> int | None:
    """The cluster size (8, 4, 2 or 1) of the aligned program for one launch, or
    ``None`` for the persistent balanced program (this module's present route),
    from host metadata only.

    A shape that is not the aligned programs' (``q_len != 1``, Q heads not exactly
    ``TILE_Q`` per KV head, or a dtype not in ``QKV_DTYPES``) takes the persistent
    program.  Otherwise the bands on ``tiles = batch_size * num_kv_heads`` and
    ``max_kv = max_seq_len_kv`` select:

    * 8: ``tiles <= 8`` below ``SEL_BAND_C8_BALANCED_KV_MIN``;
    * 4: ``8 < tiles <= 32`` with ``max_kv <= 32768`` below
      ``SEL_BAND_C4_BALANCED_KV_MIN``, or ``tiles > 64`` with
      ``32768 <= max_kv <= 65536`` except ``tiles >= 512`` with ``max_kv > 32768``;
    * 2: ``32 < tiles <= 64`` with ``max_kv <= 32768`` outside the c2 carve-outs
      (``SEL_BAND_C2_BALANCED_KV``, ``SEL_BAND_C2_BALANCED_WIDE``), or
      ``tiles >= 512`` with ``32768 < max_kv <= 65536``;
    * ``None``: ``8 < tiles <= 64`` with ``max_kv > 32768``, ``tiles > 64`` with
      ``max_kv > 65536``, ``64 < tiles <= 128`` with ``max_kv < 32768``, and the
      carve-outs of the c8 / c4 / c2 / c1 bands where the balanced program
      measured faster;
    * 1: ``tiles > 128`` with ``max_kv < 32768`` outside
      ``SEL_BAND_C1_BALANCED_KV_MIN .. SEL_BAND_C1_BALANCED_KV_MAX``.

    The bands are complete and disjoint: exactly one holds, or the function
    raises.  Selection is a performance choice only: every program handles any
    length up to ``max_pages_per_seq * PAGE_SIZE``.
    """
    if not (
        q_len == 1 and num_q_heads == TILE_Q * num_kv_heads and dtype in QKV_DTYPES
    ):
        return None
    tiles = batch_size * num_kv_heads
    max_kv = max_seq_len_kv
    sel_c8 = tiles <= SEL_BAND_C8_TILES_MAX and not _c8_balanced(max_kv)
    sel_c4 = (
        SEL_BAND_C8_TILES_MAX < tiles <= SEL_BAND_C4_TILES_MAX
        and max_kv <= SEL_BAND_LONG_KV_MIN
        and not _c4_balanced(max_kv)
    ) or (
        tiles > SEL_BAND_C2_TILES_MAX
        and SEL_BAND_LONG_KV_MIN <= max_kv <= SEL_BAND_LONG_KV_MAX
        and not (tiles >= SEL_BAND_WIDE_TILES_MIN and max_kv > SEL_BAND_LONG_KV_MIN)
    )
    sel_c2 = (
        SEL_BAND_C4_TILES_MAX < tiles <= SEL_BAND_C2_TILES_MAX
        and max_kv <= SEL_BAND_LONG_KV_MIN
        and not _c2_balanced(tiles, max_kv)
    ) or (
        tiles >= SEL_BAND_WIDE_TILES_MIN
        and SEL_BAND_LONG_KV_MIN < max_kv <= SEL_BAND_LONG_KV_MAX
    )
    sel_p = (
        (tiles > SEL_BAND_C2_TILES_MAX and max_kv > SEL_BAND_LONG_KV_MAX)
        or (
            SEL_BAND_C2_TILES_MAX < tiles <= SEL_BAND_MID_TILES_MAX
            and max_kv < SEL_BAND_LONG_KV_MIN
        )
        or (
            SEL_BAND_C8_TILES_MAX < tiles <= SEL_BAND_C2_TILES_MAX
            and max_kv > SEL_BAND_LONG_KV_MIN
        )
        or (tiles <= SEL_BAND_C8_TILES_MAX and _c8_balanced(max_kv))
        or (
            SEL_BAND_C8_TILES_MAX < tiles <= SEL_BAND_C4_TILES_MAX
            and max_kv <= SEL_BAND_LONG_KV_MIN
            and _c4_balanced(max_kv)
        )
        or (
            SEL_BAND_C4_TILES_MAX < tiles <= SEL_BAND_C2_TILES_MAX
            and max_kv <= SEL_BAND_LONG_KV_MIN
            and _c2_balanced(tiles, max_kv)
        )
        or (
            tiles > SEL_BAND_MID_TILES_MAX
            and max_kv < SEL_BAND_LONG_KV_MIN
            and _c1_balanced(max_kv)
        )
    )
    sel_c1 = (
        tiles > SEL_BAND_MID_TILES_MAX
        and max_kv < SEL_BAND_LONG_KV_MIN
        and not _c1_balanced(max_kv)
    )
    hits = [
        size
        for size, hit in (
            (8, sel_c8),
            (4, sel_c4),
            (2, sel_c2),
            (None, sel_p),
            (1, sel_c1),
        )
        if hit
    ]
    if len(hits) != 1:
        raise ValueError(
            f"the selector bands must locate exactly one program for tiles={tiles} max_kv={max_kv}, got {hits}"
        )
    return hits[0]


def _smallm_tile_rows(packed_rows: int) -> int | None:
    """Instance tile (32 or 64 rows) holding ``q_len * group`` packed rows.

    Rows in (16, 32] use the 32-row instance and rows in (32, 64] the 64-row
    instance; rows above the packed count are tile padding the kernel never
    stores.  Fewer than 17 rows are left to other routes.
    """

    if packed_rows <= 16 or packed_rows > 64:
        return None
    return 32 if packed_rows <= 32 else 64


def _smallm_tile_rows_for(q_len: int, group: int) -> int | None:
    """Tile rows for ``q_len * group`` packed rows when ``group`` divides the tile."""

    n_rows = _smallm_tile_rows(q_len * group)
    if n_rows is None or group <= 0 or n_rows % group:
        return None
    return n_rows


def _smallm_num_split(
    *, tiles: int, sm_count: int, max_seq_len: int, n_rows: int
) -> int:
    """One-wave Split-KV factor mirroring the Cake route (0 when not resident)."""

    if tiles <= 0 or tiles > sm_count:
        return 0
    blocks = max(1, (max_seq_len + 127) // 128)
    return max(1, min(sm_count // tiles, max(blocks, n_rows), 256))


def _smallm_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    *,
    tiles: int,
    num_split: int,
    n_rows: int,
    lse: torch.Tensor | None,
) -> bool:
    """Mirror the small-M binding's partial/counter/LSE workspace layout."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False

    def align(value: int) -> int:
        return (value + 255) // 256 * 256

    slots = tiles * num_split
    cursor = align(slots * n_rows * 256 * 2)
    cursor = align(cursor + slots * n_rows * 4)
    cursor = align(cursor + tiles * 2 * 4)
    if lse is None:
        cursor += query.shape[0] * query.shape[1] * 4
    return workspace_buffer.numel() * workspace_buffer.element_size() >= cursor


def _decode_native_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    max_seq_len: int,
    pages_per_block: int,
    page_table_rows: int,
    lse: torch.Tensor | None,
) -> bool:
    """Mirror one native binding's resolved metadata/LSE workspace layout."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    even_kv_blocks = max(4, even_kv_blocks)
    required_pages = even_kv_blocks * pages_per_block
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(required_pages, source_pages)
    if pages_per_block == 4:
        padded_pages = (padded_pages + 3) // 4 * 4
    needs_page_padding = source_pages != padded_pages

    cursor = (batch_size * 4 + 15) // 16 * 16
    if needs_page_padding:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    if lse is None:
        cursor += query.shape[0] * query.shape[1] * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    page_size: int,
    max_seq_len: int,
    page_table_rows: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Mirror the BF16Q adapter's padding, scale, and partial workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    group_size = query.shape[1] // num_kv_heads
    if group_size != 8 or not query.is_contiguous():
        cursor += batch_size * num_kv_heads * 8 * 128 * 2
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    cursor += batch_size * query.shape[1] * 128 * 4
    cursor += 2 * batch_size * query.shape[1] * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_scale_supported(
    scale: float | torch.Tensor, query: torch.Tensor
) -> bool:
    return not isinstance(scale, torch.Tensor) or (
        scale.device == query.device
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and scale.is_contiguous()
    )


def _pinned_noop_skip_softmax_supported(scale_factor: float | None) -> bool:
    """Accept the pinned matrix's numerically inert skip-softmax probe."""

    return scale_factor in (None, 0.0, 1e-30)


def _decode_quant_fp8_seq_lens_supported(
    seq_lens: torch.Tensor, max_seq_len: int
) -> bool:
    """Resolve the exact full-block/even-bucket route domain, or fail closed."""

    if seq_lens.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return False
    try:
        lengths = [int(value) for value in seq_lens.detach().cpu().tolist()]
    except (RuntimeError, TypeError, ValueError):
        return False
    if not lengths or max(lengths) != max_seq_len:
        return False
    if any(length < 512 or length % 128 for length in lengths):
        return False
    evened_buckets = {((length + 127) // 128 + 1) // 2 * 2 for length in lengths}
    return len(evened_buckets) == 1


def _decode_quant_fp8_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    page_size: int,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Check a conservative upper bound for runtime split-KV workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    max_splits = max(1, (even_kv_blocks + 3) // 4)
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    if not query.is_contiguous():
        cursor += batch_size * query.shape[1] * 128
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    partial_rows = batch_size * query.shape[1] * max_splits
    cursor += partial_rows * 128 * 4
    cursor += 2 * partial_rows * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_nvfp4_seq_lens_supported(
    seq_lens: torch.Tensor, max_seq_len: int
) -> bool:
    """Resolve NVFP4 runtime split inputs on the host, or fail closed."""

    if seq_lens.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return False
    try:
        lengths = [int(value) for value in seq_lens.detach().cpu().tolist()]
    except (RuntimeError, TypeError, ValueError):
        return False
    return bool(lengths) and min(lengths) > 0 and max(lengths) == max_seq_len


def _decode_quant_nvfp4_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    page_size: int,
    max_seq_len: int,
    page_table_rows: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Conservatively bound NVFP4 GQA padding and split-KV workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    max_splits = max(1, (even_kv_blocks + 3) // 4)
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    group_size = query.shape[1] // num_kv_heads
    if group_size != 8 or not query.is_contiguous():
        cursor += batch_size * num_kv_heads * 8 * 128
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    partial_rows = batch_size * query.shape[1] * max_splits
    cursor += partial_rows * 128 * 4
    cursor += 2 * partial_rows * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _manifest_optimized_route_accounting() -> tuple[int, int]:
    """Return registered/total optimized counts recorded by the manifest.

    This is a route-name drift check, not selector parity.  Exact selector
    coverage requires replaying the independent pinned capability corpus.
    """

    route_counts = get_cake_fmha_manifest()["capability"]["route_counts"]
    total = sum(
        count for name, count in route_counts.items() if name != "cake_fmha_compat_v1"
    )
    registered = sum(route_counts.get(name, 0) for name in _PRODUCT_ROUTE_COMPONENTS)
    return registered, total


def _manifest_authenticated_route_accounting() -> tuple[int, int]:
    """Return runnable/total optimized counts recorded by the manifest."""

    route_counts = get_cake_fmha_manifest()["capability"]["route_counts"]
    total = sum(
        count for name, count in route_counts.items() if name != "cake_fmha_compat_v1"
    )
    authenticated = sum(
        route_counts.get(route_name, 0)
        for route_name, components in _PRODUCT_ROUTE_COMPONENTS.items()
        if all(component in _AUTHENTICATED_JIT_COMPONENTS for component in components)
    )
    return authenticated, total


def _cake_fmha_target(device: torch.device) -> CakeFmhaTarget:
    """Resolve the exact Blackwell cubin target without cross-arch fallback."""

    from .jit.cpp_ext import is_cuda_version_at_least

    capability = get_compute_capability(device)
    if capability == (10, 0):
        if not is_cuda_version_at_least("12.8"):
            raise RuntimeError("Cake FMHA on B200 requires CUDA 12.8 or newer")
        return "sm100a"
    if capability == (10, 3):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError("Cake FMHA on B300 requires CUDA 12.9 or newer")
        return "sm103a"
    raise RuntimeError(
        "Cake FMHA requires compute capability 10.0 (B200/GB200) or "
        f"10.3 (B300/GB300), got {capability[0]}.{capability[1]}"
    )


def get_cake_fmha_module(device: torch.device):
    """Load the authenticated complete-domain compatibility module."""

    target = _cake_fmha_target(device)
    return load_cake_fmha_compat_module(target)


def select_cake_fmha_decode_route(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    q_len: int | None,
    max_seq_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    o_scale: float | None,
    sinks: torch.Tensor | None,
    kv_layout: str,
    uses_shared_paged_kv_idx: bool,
    cum_seq_lens_q: torch.Tensor | None,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    enable_block_sparse_attention: bool,
    lse: torch.Tensor | None = None,
    multi_ctas_kv_counter_buffer: torch.Tensor | None = None,
) -> CakeFmhaDecodeRoute | None:
    """Select an exact product decode route without broadening its contract."""

    if q_len is None or q_len <= 0 or batch_size <= 0 or max_seq_len <= 0:
        return None
    if cum_seq_lens_q is not None or enable_block_sparse_attention:
        return None
    if not _pinned_noop_skip_softmax_supported(skip_softmax_threshold_scale_factor):
        return None
    if query.ndim != 3 or query.stride(2) != 1:
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if any(
        tensor.device != query.device
        for tensor in (
            key_cache,
            value_cache,
            out,
            workspace_buffer,
            block_tables,
            seq_lens,
        )
    ):
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    if query.shape[0] != batch_size * q_len:
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if (
        query.stride(0) <= 0
        or query.stride(1) <= 0
        or query.stride(0) != num_q_heads * query.stride(1)
    ):
        return None
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    # The balanced head_dim-64 route serves 1..16 query heads per KV head (its
    # 16-row query tile); every other decode route serves 1..8.
    max_group = CAKE_FMHA_BALANCED_HD64_MAX_GROUP if int(query.shape[2]) == 64 else 8
    if not 1 <= num_q_heads // num_kv_heads <= max_group:
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if not block_tables.is_contiguous():
        return None
    if uses_shared_paged_kv_idx:
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            return None
    elif block_tables.ndim != 3 or block_tables.shape[:2] != (batch_size, 2):
        return None
    if (
        seq_lens.ndim != 1
        or seq_lens.shape[0] != batch_size
        or not seq_lens.is_contiguous()
    ):
        return None
    if seq_lens.dtype not in (torch.int32, torch.uint32):
        return None
    if sinks is not None and (
        not isinstance(sinks, torch.Tensor)
        or sinks.device != query.device
        or sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
        or not sinks.is_contiguous()
    ):
        return None
    if lse is not None and (
        lse.device != query.device
        or lse.dtype != torch.float32
        or lse.shape != (query.shape[0], num_q_heads)
        or not lse.is_contiguous()
        or lse.stride() != (num_q_heads, 1)
    ):
        return None

    page_size = int(key_cache.shape[2])
    local_blocks = max(1, (max_seq_len + 127) // 128)

    def route(
        component,
        *,
        selected_page_size: int = page_size,
        aligned_program: str | None = None,
    ):
        candidate = CakeFmhaDecodeRoute(
            target=_cake_fmha_target(device),
            batch_size=batch_size,
            q_len=q_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            has_sink=sinks is not None,
            has_window=window_left >= 0,
            use_scale_ptr=isinstance(bmm1_scale, torch.Tensor),
            retain_kv_l2=local_blocks <= 9,
            component=component,
            page_size=selected_page_size,
            return_lse=lse is not None,
            aligned_program=aligned_program,
        )
        if component == "decode_native_bf16" and not cake_fmha_route_is_optimized(
            candidate
        ):
            return None
        return candidate

    dtypes = (query.dtype, key_cache.dtype, value_cache.dtype, out.dtype)
    no_block_scales = key_block_scales is None and value_block_scales is None

    def aligned_route(dtype: str):
        """The aligned hd128 q1 route of ``dtype`` (``"bf16"`` / ``"fp16"``), or
        ``None`` when the launch is outside the programs' contract or
        ``aligned_cluster_size`` keeps the module's present route.

        The programs read the query as ``[batch, num_q_heads, 128]`` rows and
        K / V as separate contiguous ``[pages, num_kv_heads, 16, 128]`` pools
        (their TMA maps derive every stride from the shape), the int32 page table
        as ``[batch, max_pages_per_seq]`` and the int32 lengths on device; they
        write ``out`` in the query dtype and take no workspace, counters, LSE,
        sinks, window, device scales or output scale.
        """

        if not (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and key_cache.is_contiguous()
            and value_cache.is_contiguous()
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and q_len == 1
            and num_q_heads == 8 * num_kv_heads
            and block_tables.dtype == torch.int32
            and seq_lens.dtype == torch.int32
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return None
        cluster_size = aligned_cluster_size(
            batch_size,
            num_kv_heads,
            max_seq_len,
            q_len=q_len,
            num_q_heads=num_q_heads,
            dtype=dtype,
        )
        if cluster_size is None:
            return None
        return route(
            f"decode_aligned_{dtype}",
            selected_page_size=16,
            aligned_program=_aligned_program_id(cluster_size, dtype),
        )

    if dtypes == (torch.bfloat16,) * 4:
        group = num_q_heads // num_kv_heads
        smallm_rows = _smallm_tile_rows_for(q_len, group)
        smallm_tiles = batch_size * num_kv_heads
        smallm_sm_count = _smallm_sm_count(device)
        smallm_num_split = _smallm_num_split(
            tiles=smallm_tiles,
            sm_count=smallm_sm_count,
            max_seq_len=max_seq_len,
            n_rows=smallm_rows or 64,
        )
        if (
            query.is_contiguous()
            and query.shape[2] == 64
            and key_cache.shape[2:] == (16, 64)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and q_len == 1
            and 1 <= group <= CAKE_FMHA_BALANCED_HD64_MAX_GROUP
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="bf16_hd64",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=smallm_sm_count,
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
                head_dim=64,
            )
        ):
            # head_dim 64 (Q16Kv128 ForGen body, 1..16 query heads per KV head)
            # on the on-device balanced scheduler; no other product route
            # serves head_dim 64.
            return route("decode_balanced_bf16_hd64", selected_page_size=16)
        if (
            query.is_contiguous()
            and query.shape[2] == 256
            and smallm_rows is not None
            and page_size in (16, 32, 64)
            and key_cache.shape[3] == 256
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and sinks is None
            and window_left < 0
            and not isinstance(bmm1_scale, torch.Tensor)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and smallm_num_split >= 1
            and _smallm_workspace_supported(
                workspace_buffer,
                query,
                tiles=smallm_tiles,
                num_split=smallm_num_split,
                n_rows=smallm_rows,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            candidate = CakeFmhaDecodeRoute(
                target=_cake_fmha_target(device),
                batch_size=batch_size,
                q_len=q_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                has_sink=False,
                has_window=False,
                use_scale_ptr=False,
                retain_kv_l2=local_blocks <= 9,
                component="decode_native_bf16_hd256_smallm",
                page_size=page_size,
                num_split=smallm_num_split,
            )
            return candidate if cake_fmha_route_is_optimized(candidate) else None
        if (
            query.is_contiguous()
            and query.shape[2] == 256
            and key_cache.shape[3] == 256
            and page_size in (16, 32, 64)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and 1 <= q_len <= CAKE_FMHA_BALANCED_HD256_MAX_Q_LEN
            and 1 <= group <= 8
            and max_seq_len >= q_len
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="bf16_hd256",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=smallm_sm_count,
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
                head_dim=256,
            )
        ):
            # head_dim 256 outside the small-M route's packed tiles: every
            # query row is its own tile (uniform q_len 1..8) on the on-device
            # balanced scheduler, one program per page size (16 / 32 / 64).
            return route("decode_balanced_bf16_hd256")
        # Aligned hd128 q1 decode: one 8/4/2/1-CTA cluster per (request, KV
        # head) tile, selected by ``aligned_cluster_size`` ahead of the present
        # hd128 routes; ``None`` keeps them.
        aligned = aligned_route("bf16")
        if aligned is not None:
            return aligned
        if (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and group == 8
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="bf16",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=smallm_sm_count,
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
            )
        ):
            # Long-context / ragged decode: the on-device balanced scheduler
            # splits long requests into chunks and merges them on device.
            return route("decode_balanced_bf16", selected_page_size=16)
        if (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=8,
                page_table_rows=1,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_bf16", selected_page_size=16)
        return None

    if dtypes == (torch.float16,) * 4:
        # Aligned hd128 q1 decode in FP16: the same programs traced with FP16
        # Q/K/V/O, ahead of the balanced FP16 route; ``None`` keeps it.
        aligned = aligned_route("fp16")
        if aligned is not None:
            return aligned
        if (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and num_q_heads // num_kv_heads == 8
            and sinks is None
            and window_left < 0
            and lse is None
            and not isinstance(bmm1_scale, torch.Tensor)
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _balanced_route_supported(
                _cake_fmha_target(device),
                dtype="fp16",
                batch_size=batch_size,
                q_len=q_len,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                sm_count=_smallm_sm_count(device),
                workspace_buffer=workspace_buffer,
                counter_buffer=multi_ctas_kv_counter_buffer,
            )
        ):
            # Long-context / ragged FP16 decode: the same on-device balanced
            # scheduler and physical schedule, traced with FP16 Q/K/V/O.
            return route("decode_balanced_fp16", selected_page_size=16)
        if (
            query.shape[2] == 128
            and key_cache.shape[2:] == (32, 128)
            and kv_layout == "NHD"
            and not uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=4,
                page_table_rows=2,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_nhd", selected_page_size=32)
        if (
            query.shape[2] == 512
            and query.is_contiguous()
            and key_cache.shape[2:] == (64, 512)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=2,
                page_table_rows=1,
                lse=lse,
            )
            and no_block_scales
            and sinks is None
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_hd512", selected_page_size=64)
        return None

    quant_extensions_absent = (
        sinks is None
        and lse is None
        and window_left == -1
        and skip_softmax_threshold_scale_factor in (None, 0.0)
    )
    # E4M3 cache on the on-device balanced scheduler (E4M3, BF16 or FP16 query
    # with the output in the query dtype; HND, page 16, GQA-8, q_len 1).  Host
    # scales only: ``bmm1_scale = q_scale * k_scale / sqrt(128)`` becomes the
    # kernel's log2 softmax scale and ``bmm2_scale = v_scale / o_scale`` its
    # output scale; device-tensor scales stay on the quantized routes / compat
    # (a per-launch host read would break graph-replay safety).
    balanced_fp8_kv_component = {
        (torch.float8_e4m3fn,) * 4: "decode_balanced_fp8",
        (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        ): "decode_balanced_bf16q",
        (
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float16,
        ): "decode_balanced_fp16q",
    }.get(dtypes)
    if (
        balanced_fp8_kv_component is not None
        and q_len == 1
        and query.shape[2] == 128
        and key_cache.shape[2:] == (16, 128)
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and num_q_heads // num_kv_heads == 8
        and no_block_scales
        and quant_extensions_absent
        and (o_scale is None or float(o_scale) == 1.0)
        and not isinstance(bmm1_scale, torch.Tensor)
        and not isinstance(bmm2_scale, torch.Tensor)
        and float(bmm1_scale) > 0.0
        and float(bmm2_scale) > 0.0
        and query.is_contiguous()
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and _balanced_route_supported(
            _cake_fmha_target(device),
            dtype=balanced_fp8_kv_component.removeprefix("decode_balanced_"),
            batch_size=batch_size,
            q_len=q_len,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            sm_count=_smallm_sm_count(device),
            workspace_buffer=workspace_buffer,
            counter_buffer=multi_ctas_kv_counter_buffer,
        )
    ):
        return route(balanced_fp8_kv_component, selected_page_size=16)

    if (
        dtypes == (torch.float8_e4m3fn,) * 4
        and q_len == 1
        and query.shape[2] == 128
        and page_size in (16, 32)
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and num_q_heads // num_kv_heads == 8
        and no_block_scales
        and quant_extensions_absent
        and max_seq_len >= 512
        and max_seq_len % 128 == 0
        and _decode_quant_fp8_seq_lens_supported(seq_lens, max_seq_len)
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_fp8_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            page_size=page_size,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        return route("decode_quant_fp8")

    if (
        dtypes
        == (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )
        and q_len == 1
        and query.shape[2] == 128
        and page_size in (16, 32)
        and kv_layout in ("HND", "NHD")
        and num_q_heads % num_kv_heads == 0
        and 1 <= num_q_heads // num_kv_heads < 8
        and no_block_scales
        and quant_extensions_absent
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            page_table_rows=1 if uses_shared_paged_kv_idx else 2,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        return route("decode_quant_bf16q")

    if (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        and q_len == 1
        and query.shape[2] == 128
        and key_cache.shape[3] == 64
        and page_size in (16, 32)
        and kv_layout == "HND"
        and quant_extensions_absent
        and key_block_scales is not None
        and value_block_scales is not None
        and key_block_scales.dtype == torch.float8_e4m3fn
        and value_block_scales.dtype == torch.float8_e4m3fn
        and key_block_scales.ndim == 4
        and value_block_scales.shape == key_block_scales.shape
        and key_block_scales.shape[:3] == key_cache.shape[:3]
        and key_block_scales.shape[3] == 8
        and key_block_scales.stride(3) == 1
        and value_block_scales.stride(3) == 1
        and key_block_scales.device == query.device
        and value_block_scales.device == query.device
        and num_q_heads // num_kv_heads <= 8
        and _decode_quant_nvfp4_seq_lens_supported(seq_lens, max_seq_len)
        and _tma_nvfp4_paged_kv_strides_supported(key_cache)
        and _tma_nvfp4_paged_kv_strides_supported(value_cache)
        and _tma_nvfp4_scale_strides_supported(key_block_scales)
        and _tma_nvfp4_scale_strides_supported(value_block_scales)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and key_block_scales.data_ptr() % 16 == 0
        and value_block_scales.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_nvfp4_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            page_table_rows=1 if uses_shared_paged_kv_idx else 2,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        return route("decode_quant_nvfp4")
    return None


@dataclass(frozen=True)
class CakeFmhaAlignedDecodeModule:
    """One loaded aligned hd128 q1 decode program: the route it serves, the
    ``CakeProgram`` record, its exact target, the cluster size the launch grid
    spans along z and the binding's ``run`` entry."""

    route: CakeFmhaDecodeRoute
    program: Any
    target: str
    cluster_size: int
    entry: Any


def _load_cake_fmha_aligned_decode_module(
    route: CakeFmhaDecodeRoute,
) -> CakeFmhaAlignedDecodeModule:
    """Build and load the aligned program of ``route`` on the route's target.

    The loader ``flashinfer.jit.cake_fmha_aligned`` is generated by the Cake
    export beside ``csrc/cake_fmha_aligned/``; it is imported here, lazily, so
    route selection never needs it.  The record's launch facts and argument
    names are checked against the host launch once, at load.
    """

    from .jit import cake_fmha_aligned
    from .jit.cake_program import GRID_ARGS, CakeProgram

    program_facts = _aligned_route_program(route)
    if program_facts is None:
        raise RuntimeError(
            f"Cake FMHA aligned decode route names no program: {route.aligned_program!r}"
        )
    cluster_size, _, program_name = program_facts
    program = getattr(cake_fmha_aligned, route.aligned_program, None)
    if not isinstance(program, CakeProgram):
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned publishes no CakeProgram {route.aligned_program!r}"
        )
    if program.name != program_name:
        raise RuntimeError(
            f"flashinfer.jit.cake_fmha_aligned.{route.aligned_program} is "
            f"{program.name}; this module pins {program_name}"
        )
    target = _ALIGNED_PROGRAM_TARGET[route.target]
    if target not in program.targets:
        raise RuntimeError(
            f"{program.name} ships no {target!r} build; its targets are {list(program.targets)}"
        )
    if tuple(program.args) != (*_ALIGNED_HOST_PARAMS, *GRID_ARGS):
        raise RuntimeError(
            f"the generated {program.name} binding takes {list(program.args)}; "
            f"the host spells {[*_ALIGNED_HOST_PARAMS, *GRID_ARGS]}"
        )
    if tuple(program.cluster) != (1, 1, cluster_size):
        raise RuntimeError(
            f"{program.name} launches cluster {tuple(program.cluster)}; the route "
            f"expects (1, 1, {cluster_size})"
        )
    return CakeFmhaAlignedDecodeModule(
        route=route,
        program=program,
        target=target,
        cluster_size=cluster_size,
        entry=program.entry(target),
    )


def _run_cake_fmha_aligned_decode(
    module: CakeFmhaAlignedDecodeModule,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    bmm1_scale: float,
) -> None:
    """Launch one aligned hd128 q1 decode on the current torch stream.

    The binding's host parameters, in the kernel's order: the query rows
    (``Qt``, TMA source, ``[batch x num_kv_heads x 8, 128]``), the K and V pools
    (TMA sources), the output, the page table, the device lengths,
    ``uniform_seq_len`` (0: every CTA reads its request's length from
    ``seq_lens_kv``, so a captured launch follows new lengths),
    ``max_pages_per_seq`` (the page table's row width; the kernel clamps its
    page-id reads to the row), the log2-domain softmax scale, the head counts
    and the batch size; then the grid of :func:`cake_fmha_aligned_launch_grid`.
    """

    route = module.route
    if (
        tuple(query.shape) != (route.batch_size, route.num_q_heads, 128)
        or key_cache.shape[1] != route.num_kv_heads
        or tuple(block_tables.shape[:1]) != (route.batch_size,)
    ):
        raise RuntimeError(
            "Cake FMHA aligned decode launch does not match its selected route"
        )
    grid = cake_fmha_aligned_launch_grid(
        module.program, route.batch_size, route.num_kv_heads
    )
    values = {
        "Qt": query.view(-1, 128),
        "K": key_cache,
        "V": value_cache,
        "O": out,
        "page_table": block_tables,
        "seq_lens_kv": seq_lens,
        "uniform_seq_len": 0,
        "max_pages_per_seq": int(block_tables.shape[1]),
        "softmax_scale_log2": float(bmm1_scale) * log2e,
        "num_q_heads": route.num_q_heads,
        "num_kv_heads": route.num_kv_heads,
        "batch_size": route.batch_size,
        "grid_x": grid[0],
        "grid_y": grid[1],
        "grid_z": grid[2],
    }
    arguments = [values[name] for name in module.program.args]
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        module.entry(*arguments)


def _resolve_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
) -> tuple[Any, bool]:
    """Resolve a decode module and whether its optimized ABI remains active.

    An aligned hd128 q1 route resolves to a :class:`CakeFmhaAlignedDecodeModule`
    (a generated ``CakeProgram`` binding launched by
    :func:`_run_cake_fmha_aligned_decode`), every other optimized route to a
    module exposing ``cake_paged_attention_decode``.
    """

    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device)), False
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA decode route target does not match the device")
    if route.component in _ALIGNED_COMPONENTS:
        return _load_cake_fmha_aligned_decode_module(route), True
    loader = {
        "decode_balanced_bf16": load_cake_fmha_decode_balanced_bf16_module,
        "decode_balanced_fp16": load_cake_fmha_decode_balanced_fp16_module,
        "decode_balanced_fp8": load_cake_fmha_decode_balanced_fp8_module,
        "decode_balanced_bf16q": load_cake_fmha_decode_balanced_fp8_module,
        "decode_balanced_fp16q": load_cake_fmha_decode_balanced_fp8_module,
        "decode_balanced_bf16_hd64": load_cake_fmha_decode_balanced_hd64_module,
        "decode_balanced_bf16_hd256": load_cake_fmha_decode_balanced_hd256_module,
        "decode_native_bf16": load_cake_fmha_decode_native_bf16_module,
        "decode_native_bf16_hd256_smallm": load_cake_fmha_decode_native_bf16_hd256_smallm_module,
        "decode_native_fp16_hd512": load_cake_fmha_decode_native_fp16_hd512_module,
        "decode_native_fp16_nhd": load_cake_fmha_decode_native_fp16_nhd_module,
        "decode_quant_bf16q": load_cake_fmha_decode_quant_bf16q_module,
        "decode_quant_fp8": load_cake_fmha_decode_quant_fp8_module,
        "decode_quant_nvfp4": load_cake_fmha_decode_quant_nvfp4_module,
    }.get(route.component)
    if loader is None:
        raise RuntimeError(
            f"Cake FMHA decode route has no authenticated loader: {route.component}"
        )
    common_args = (
        route.target,
        route.batch_size,
        route.q_len,
        route.num_q_heads,
        route.num_kv_heads,
    )
    if route.component in ("decode_balanced_bf16", "decode_balanced_fp16"):
        return loader(route.target, route.q_len), True
    if route.component in (
        "decode_balanced_fp8",
        "decode_balanced_bf16q",
        "decode_balanced_fp16q",
    ):
        return loader(
            route.target, route.component.removeprefix("decode_balanced_")
        ), True
    if route.component == "decode_balanced_bf16_hd64":
        # Structural instance by head group: eight softmax columns for 1..8
        # query heads per KV head (every product row), sixteen for 9..16.
        return loader(
            route.target,
            cake_fmha_balanced_hd64_max_group(route.num_q_heads, route.num_kv_heads),
        ), True
    if route.component == "decode_balanced_bf16_hd256":
        return loader(route.target, route.page_size), True
    if route.component == "decode_native_bf16_hd256_smallm":
        group = route.num_q_heads // route.num_kv_heads
        return (
            loader(
                route.target,
                _smallm_tile_rows_for(route.q_len, group),
                route.page_size,
                route.q_len,
                group,
                route.num_split,
            ),
            True,
        )
    if route.component == "decode_native_fp16_hd512":
        return (
            loader(
                *common_args,
                has_window=route.has_window,
                use_scale_ptr=route.use_scale_ptr,
                retain_kv_l2=route.retain_kv_l2,
            ),
            True,
        )
    if route.component == "decode_quant_bf16q":
        return loader(*common_args, route.page_size), True
    if route.component == "decode_quant_fp8":
        return loader(*common_args, route.page_size, full_blocks=True), True
    if route.component == "decode_quant_nvfp4":
        try:
            return loader(*common_args, route.page_size), True
        except (OSError, RuntimeError) as error:
            warnings.warn(
                f"Cake FMHA portable NVFP4 loading failed closed to compat_v1: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return load_cake_fmha_compat_module(route.target), False
    if route.component == "decode_native_bf16":
        return (
            loader(
                *common_args,
                has_sink=route.has_sink,
                has_window=route.has_window,
                use_scale_ptr=route.use_scale_ptr,
                retain_kv_l2=route.retain_kv_l2,
                return_lse=route.return_lse,
            ),
            True,
        )
    return (
        loader(
            *common_args,
            has_sink=route.has_sink,
            has_window=route.has_window,
            use_scale_ptr=route.use_scale_ptr,
            retain_kv_l2=route.retain_kv_l2,
        ),
        True,
    )


def get_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
):
    """Load an optimized decode module, or the authenticated portable fallback."""

    module, _ = _resolve_cake_fmha_decode_module(device, route)
    return module


def _context_tile_mma_work(q_len: int, kv_len: int, tokens_per_tile: int) -> int:
    """Mirror the standalone route's bottom-right causal tile-work model."""

    total = 0
    full_n_blocks = (kv_len + 127) // 128
    shift = kv_len - q_len
    for m_block in range((q_len + tokens_per_tile - 1) // tokens_per_tile):
        max_n = (m_block + 1) * tokens_per_tile + shift
        total += (max_n + 127) // 128 if max_n < kv_len else full_n_blocks
    return total


def _context_pack_g(
    max_q_len: int, max_kv_len: int, num_q_heads: int, num_kv_heads: int
) -> int:
    """Choose the canonical packed-GQA axis from host-visible maxima."""

    group = num_q_heads // num_kv_heads
    if group <= 1 or group > 128:
        return 1
    unpacked = num_q_heads * _context_tile_mma_work(max_q_len, max_kv_len, 256)
    packed = num_kv_heads * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // group)
    )
    return group if packed < unpacked else 1


def _context_hd256_workspace_supported(
    workspace_buffer: torch.Tensor | None,
    query: torch.Tensor,
    *,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    fp8: bool,
) -> bool:
    """Mirror the HD256 binding's packed Q/K/V/O and metadata arena."""

    if (
        workspace_buffer is None
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.device != query.device
        or workspace_buffer.data_ptr() % 16
    ):
        return False

    padded_q = ((max_q_len + 127) // 128) * 128
    max_micro_pages = (max_kv_len + 15) // 16
    total_q_rows = batch_size * num_q_heads * padded_q
    total_micro_pages = batch_size * num_kv_heads * max_micro_pages
    input_bytes = 256 if fp8 else 512

    cursor = total_q_rows * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_micro_pages * 16 * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_micro_pages * 16 * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_q_rows * 512
    cursor = (cursor + 15) // 16 * 16
    cursor += 3 * batch_size * num_q_heads * 4
    cursor += total_micro_pages * 4
    return workspace_buffer.numel() * workspace_buffer.element_size() >= cursor


def _context_nvfp4_workspace_supported(
    workspace_buffer: torch.Tensor | None,
    key_cache: torch.Tensor,
    *,
    batch_size: int,
    num_q_heads: int,
    pack_g: int,
) -> bool:
    """Bound the fused kernel's expanded-metadata workspace exactly."""

    if (
        workspace_buffer is None
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.device != key_cache.device
        or workspace_buffer.data_ptr() % 16
    ):
        return False
    total_bh = batch_size * (num_q_heads // pack_g)
    seq_kv_offset = ((total_bh * 4 + 15) // 16) * 16
    required = ((seq_kv_offset + 2 * total_bh * 4 + 15) // 16) * 16
    return workspace_buffer.numel() * workspace_buffer.element_size() >= required


def _context_bf16_exact_profile(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    *,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    uses_shared_paged_kv_idx: bool,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
    kv_layout: str,
) -> Literal["q511", "q257"] | None:
    """Resolve the two measured mask-loop bodies from exact runtime lengths."""

    common = (
        batch_size == 4
        and num_q_heads == 10
        and num_kv_heads == 2
        and is_causal
        and not return_lse
        and not enable_sink
        and kv_layout == "HND"
        and seq_lens.device == query.device
        and cum_seq_lens_q.device == query.device
    )
    if not common:
        return None
    profile: Literal["q511", "q257"]
    if (
        max_q_len == 511
        and max_kv_len == 2047
        and page_size == 32
        and uses_shared_paged_kv_idx
        and query.shape[0] == 4 * 511
    ):
        profile = "q511"
        expected_q_len = 511
        expected_kv_len = 2047
    elif (
        max_q_len == 257
        and max_kv_len == 1024
        and page_size == 1024
        and not uses_shared_paged_kv_idx
        and query.shape[0] == 4 * 257
    ):
        profile = "q257"
        expected_q_len = 257
        expected_kv_len = 1024
    else:
        return None
    if query.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return None
    try:
        kv_lengths = [int(value) for value in seq_lens.detach().cpu().tolist()]
        q_indptr = [int(value) for value in cum_seq_lens_q.detach().cpu().tolist()]
    except RuntimeError:
        return None
    if kv_lengths != [expected_kv_len] * batch_size:
        return None
    if q_indptr != [index * expected_q_len for index in range(batch_size + 1)]:
        return None
    return profile


def select_cake_fmha_context_route(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    uses_shared_paged_kv_idx: bool,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    is_causal: bool,
    lse: torch.Tensor | None,
    kv_layout: str = "HND",
    workspace_buffer: torch.Tensor | None = None,
) -> CakeFmhaContextRoute | None:
    """Select an exact product context route without broadening its contract."""

    if batch_size <= 0 or max_q_len <= 0 or max_kv_len <= 0 or window_left != -1:
        return None
    if not _decode_quant_scale_supported(
        bmm1_scale, query
    ) or not _decode_quant_scale_supported(bmm2_scale, query):
        return None
    if not _pinned_noop_skip_softmax_supported(skip_softmax_threshold_scale_factor):
        return None
    if query.ndim != 3 or query.stride(2) != 1:
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    page_size = int(key_cache.shape[2])
    if page_size not in (16, 32, 64, 128, 256, 512, 1024):
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    if seq_lens.ndim != 1 or seq_lens.shape[0] != batch_size:
        return None
    if seq_lens.dtype != torch.int32:
        return None
    if not seq_lens.is_contiguous():
        return None
    for indptr in (cum_seq_lens_q, cum_seq_lens_kv):
        if (
            indptr.ndim != 1
            or indptr.shape[0] != batch_size + 1
            or indptr.dtype != torch.int32
            or not indptr.is_contiguous()
        ):
            return None
    if sinks is not None and (
        sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
        or not sinks.is_contiguous()
    ):
        return None
    if sinks is not None and lse is not None:
        return None
    if lse is not None and (
        lse.dtype != torch.float32
        or lse.shape != (query.shape[0], num_q_heads)
        or not lse.is_contiguous()
    ):
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if uses_shared_paged_kv_idx:
        if (
            block_tables.ndim != 2
            or block_tables.shape[0] != batch_size
            or block_tables.stride(1) != 1
        ):
            return None
    elif (
        block_tables.ndim != 3
        or block_tables.shape[:2] != (batch_size, 2)
        or block_tables.stride(2) != 1
    ):
        return None

    dtypes = (query.dtype, key_cache.dtype, value_cache.dtype, out.dtype)
    no_block_scales = key_block_scales is None and value_block_scales is None
    host_scalar_scales = not isinstance(bmm1_scale, torch.Tensor) and not isinstance(
        bmm2_scale, torch.Tensor
    )
    component: CakeFmhaContextComponent | None = None
    if (
        dtypes == (torch.bfloat16,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout in ("HND", "NHD")
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and no_block_scales
        and host_scalar_scales
        and float(bmm2_scale) == 1.0
    ):
        component = "context_bf16"
    elif (
        dtypes == (torch.float8_e4m3fn,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout in ("HND", "NHD")
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and no_block_scales
    ):
        component = "context_fp8"
    elif (
        dtypes == (torch.float16,) * 4
        and query.shape[2] == 256
        and key_cache.shape[3] == 256
        and kv_layout == "NHD"
        and not uses_shared_paged_kv_idx
        and no_block_scales
        and host_scalar_scales
        and not is_causal
        and sinks is None
        and lse is None
        and float(bmm2_scale) == 1.0
    ):
        component = "context_fp16_hd256"
    elif (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )
        and query.shape[2] == 256
        and key_cache.shape[3] == 256
        and kv_layout == "NHD"
        and not uses_shared_paged_kv_idx
        and no_block_scales
        and is_causal
        and sinks is None
        and lse is None
    ):
        component = "context_fp8_hd256"
    elif (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        and query.shape[2] == 128
        and key_cache.shape[2:] == (16, 64)
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and is_causal
        and sinks is None
        and lse is None
        and skip_softmax_threshold_scale_factor in (None, 0.0)
        and key_block_scales is not None
        and value_block_scales is not None
        and key_block_scales.dtype == torch.float8_e4m3fn
        and value_block_scales.dtype == torch.float8_e4m3fn
        and key_block_scales.shape == value_block_scales.shape
        and key_block_scales.shape[:3] == key_cache.shape[:3]
        and key_block_scales.shape[3] == 8
        and key_block_scales.is_contiguous()
        and value_block_scales.is_contiguous()
    ):
        component = "context_nvfp4"
    if component is None:
        return None

    exact_profile = None
    if component == "context_bf16":
        exact_profile = _context_bf16_exact_profile(
            query,
            seq_lens,
            cum_seq_lens_q,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            is_causal=is_causal,
            return_lse=lse is not None,
            enable_sink=sinks is not None,
            kv_layout=kv_layout,
        )
    if exact_profile == "q511":
        pack_g = 5
        num_m_blocks = 11
        l2_swizzle = 1
    elif exact_profile == "q257":
        pack_g = 5
        num_m_blocks = 6
        l2_swizzle = 8
    elif component in ("context_fp16_hd256", "context_fp8_hd256"):
        pack_g = 1
        num_m_blocks = (max_q_len + 127) // 128
        l2_swizzle = 1
    else:
        pack_g = _context_pack_g(max_q_len, max_kv_len, num_q_heads, num_kv_heads)
        tok_per_stage = 128 // pack_g
        num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (2 * tok_per_stage)
        total_bh = batch_size * (num_q_heads // pack_g)
        l2_swizzle = 8 if total_bh % 8 == 0 else 1
    if component == "context_nvfp4" and not _context_nvfp4_workspace_supported(
        workspace_buffer,
        key_cache,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        pack_g=pack_g,
    ):
        return None
    if component in ("context_fp16_hd256", "context_fp8_hd256") and not (
        _context_hd256_workspace_supported(
            workspace_buffer,
            query,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            fp8=component == "context_fp8_hd256",
        )
    ):
        return None
    return CakeFmhaContextRoute(
        target=_cake_fmha_target(device),
        component=component,
        num_m_blocks=num_m_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        pack_g=pack_g,
        page_size=page_size,
        l2_swizzle=l2_swizzle,
        is_causal=is_causal,
        return_lse=lse is not None,
        enable_sink=sinks is not None,
        exact_profile=exact_profile,
    )


def get_cake_fmha_context_module(
    device: torch.device, route: CakeFmhaContextRoute | None
):
    """Load an optimized context module, or the authenticated portable fallback."""

    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device))
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA context route target does not match the device")
    if route.component == "context_fp16_hd256":
        return load_cake_fmha_context_fp16_hd256_module(
            route.target,
            route.num_m_blocks,
            route.num_q_heads,
            route.num_kv_heads,
            route.page_size,
        )
    if route.component == "context_fp8_hd256":
        return load_cake_fmha_context_fp8_hd256_module(
            route.target,
            route.num_m_blocks,
            route.num_q_heads,
            route.num_kv_heads,
            route.page_size,
        )
    if route.component == "context_nvfp4":
        try:
            return load_cake_fmha_context_nvfp4_module(
                route.target,
                route.num_m_blocks,
                route.num_q_heads,
                route.num_kv_heads,
                route.pack_g,
                route.page_size,
                route.l2_swizzle,
            )
        except (OSError, RuntimeError) as error:
            warnings.warn(
                f"Cake FMHA NVFP4 context loading failed closed to compat_v1: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return load_cake_fmha_compat_module(route.target)
    common_args = (
        route.target,
        route.num_m_blocks,
        route.num_q_heads,
        route.num_kv_heads,
        route.pack_g,
        route.page_size,
        route.l2_swizzle,
    )
    common_kwargs = {
        "is_causal": route.is_causal,
        "return_lse": route.return_lse,
        "enable_sink": route.enable_sink,
    }
    if route.component == "context_bf16":
        return load_cake_fmha_context_bf16_module(
            *common_args,
            **common_kwargs,
            exact_profile=route.exact_profile,
        )
    return load_cake_fmha_context_fp8_module(*common_args, **common_kwargs)


def cake_fmha_manifest() -> dict[str, Any]:
    """Return a copy of the authenticated product/capability manifest."""

    return copy.deepcopy(get_cake_fmha_manifest())


def cake_batch_decode_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-decode ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_decode_with_kv_cache`.  The selection is
    explicit and never replaces FlashInfer's default backend selection.
    """

    from .decode import trtllm_batch_decode_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_decode_with_kv_cache requires backend='cake'")
    return trtllm_batch_decode_with_kv_cache(*args, backend="cake", **kwargs)


def cake_batch_context_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-context ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_context_with_kv_cache`.  The selection is
    explicit and never replaces FlashInfer's conventional default.
    """

    from .prefill import trtllm_batch_context_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_context_with_kv_cache requires backend='cake'")
    return trtllm_batch_context_with_kv_cache(*args, backend="cake", **kwargs)


__all__ = [
    "CakeFmhaContextRoute",
    "CakeFmhaDecodeRoute",
    "CakeFmhaRequestOrderedDecodePlan",
    "cake_batch_context_with_kv_cache",
    "cake_batch_decode_with_kv_cache",
    "cake_fmha_manifest",
    "cake_fmha_route_is_optimized",
    "get_cake_fmha_context_module",
    "get_cake_fmha_decode_module",
    "get_cake_fmha_module",
    "plan_cake_fmha_request_ordered_paged_decode",
    "select_cake_fmha_context_route",
    "select_cake_fmha_decode_route",
]
