"""Public Cake FMHA product entrypoints.

The conventional decode/context APIs are complete-domain Cake routes.  Issue
#4323 DCP metadata selects an authenticated additive profile through the same
decode entrypoint and does not change ordinary-call behavior.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .jit.cake_fmha import (
    CakeFmhaTarget,
    get_cake_fmha_manifest,
    load_cake_fmha_context_bf16_module,
    load_cake_fmha_context_fp8_module,
    load_cake_fmha_compat_module,
    load_cake_fmha_decode_native_bf16_module,
)
from .utils import get_compute_capability


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
        "decode_native_bf16",
        "decode_native_fp16_hd512",
        "decode_native_fp16_nhd",
        "decode_quant_bf16q",
        "decode_quant_fp8",
        "decode_quant_nvfp4",
    ] = "decode_native_bf16"
    page_size: int = 16


@dataclass(frozen=True)
class CakeFmhaContextRoute:
    """One exact manifest-backed optimized context specialization."""

    target: CakeFmhaTarget
    component: Literal[
        "context_bf16",
        "context_fp16_hd256",
        "context_fp8",
        "context_fp8_hd256",
        "context_nvfp4",
    ]
    num_m_blocks: int
    num_q_heads: int
    num_kv_heads: int
    pack_g: int
    page_size: int
    l2_swizzle: int
    is_causal: bool
    return_lse: bool
    enable_sink: bool


# These are the exact optimized routes in the pinned 57,280-cell matrix.  Keep
# the component sequences explicit: hd256 needs its staging/scatter support,
# and split-KV NVFP4 decode may also need the shared reduction component.
_OPTIMIZED_ROUTE_COMPONENTS: dict[str, tuple[str, ...]] = {
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
    "decode_native_bf16_v1_bece": ("decode_native_bf16",),
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

# Only these components have an authenticated FlashInfer TVM-FFI adapter in
# the checked-in package.  Candidate routes whose adapter is part of the final
# export remain fail-closed to compat_v1 until that export and binding digest
# are updated together.
_AUTHENTICATED_JIT_COMPONENTS = frozenset(
    {"compat_v1", "context_bf16", "context_fp8", "decode_native_bf16"}
)


def _route_components(route: CakeFmhaDecodeRoute | CakeFmhaContextRoute) -> tuple[str, ...]:
    if isinstance(route, CakeFmhaDecodeRoute):
        route_name = {
            "decode_native_bf16": "decode_native_bf16_v1_bece",
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
    return _OPTIMIZED_ROUTE_COMPONENTS[route_name]


def cake_fmha_route_is_optimized(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute | None,
) -> bool:
    """Return whether ``route`` has a fully authenticated runnable adapter."""

    return route is not None and all(
        component in _AUTHENTICATED_JIT_COMPONENTS
        for component in _route_components(route)
    )


def _optimized_route_coverage() -> tuple[int, int]:
    """Return (routed, total) optimized cells from the authenticated manifest."""

    route_counts = get_cake_fmha_manifest()["capability"]["route_counts"]
    total = sum(count for name, count in route_counts.items() if name != "cake_fmha_compat_v1")
    routed = sum(route_counts.get(name, 0) for name in _OPTIMIZED_ROUTE_COMPONENTS)
    return routed, total


def _cake_fmha_target(device: torch.device) -> CakeFmhaTarget:
    """Resolve the exact Blackwell cubin target without cross-arch fallback."""

    from .jit.cpp_ext import is_cuda_version_at_least

    capability = get_compute_capability(device)
    if capability == (10, 0):
        if not is_cuda_version_at_least("12.8"):
            raise RuntimeError("Cake FMHA on B200 requires CUDA 12.8 or newer")
        target = "sm100a"
    elif capability == (10, 3):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError("Cake FMHA on B300 requires CUDA 12.9 or newer")
        target = "sm103a"
    else:
        raise RuntimeError(
            "Cake FMHA requires compute capability 10.0 (B200/GB200) or "
            f"10.3 (B300/GB300), got {capability[0]}.{capability[1]}"
        )
    return target


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
) -> CakeFmhaDecodeRoute | None:
    """Select an exact product decode route without broadening its contract."""

    if q_len is None or q_len <= 0 or batch_size <= 0 or max_seq_len <= 0:
        return None
    if cum_seq_lens_q is not None or enable_block_sparse_attention:
        return None
    if skip_softmax_threshold_scale_factor not in (None, 0.0):
        return None
    if query.ndim != 3 or not query.is_contiguous():
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    if query.shape[0] != batch_size * q_len:
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    if not 1 <= num_q_heads // num_kv_heads <= 8:
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if uses_shared_paged_kv_idx:
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            return None
    elif block_tables.ndim != 3 or block_tables.shape[:2] != (batch_size, 2):
        return None
    if seq_lens.ndim != 1 or seq_lens.shape[0] != batch_size or not seq_lens.is_contiguous():
        return None
    if seq_lens.dtype not in (torch.int32, torch.uint32):
        return None
    if sinks is not None and (
        not isinstance(sinks, torch.Tensor)
        or sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
        or not sinks.is_contiguous()
    ):
        return None

    page_size = int(key_cache.shape[2])
    local_blocks = max(1, (max_seq_len + 127) // 128)

    def route(component, *, selected_page_size: int = page_size):
        return CakeFmhaDecodeRoute(
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
        )

    dtypes = (query.dtype, key_cache.dtype, value_cache.dtype, out.dtype)
    no_block_scales = key_block_scales is None and value_block_scales is None
    if dtypes == (torch.bfloat16,) * 4:
        if (
            query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_bf16", selected_page_size=16)
        return None

    if dtypes == (torch.float16,) * 4:
        if (
            query.shape[2] == 128
            and key_cache.shape[2:] == (32, 128)
            and kv_layout == "NHD"
            and not uses_shared_paged_kv_idx
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_nhd", selected_page_size=32)
        if (
            query.shape[2] == 512
            and key_cache.shape[2:] == (64, 512)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and no_block_scales
            and sinks is None
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_hd512", selected_page_size=64)
        return None

    quant_extensions_absent = sinks is None and lse is None and window_left == -1
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
        and no_block_scales
        and quant_extensions_absent
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
    ):
        return route("decode_quant_nvfp4")
    return None


def get_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
):
    """Load an optimized decode module, or the authenticated portable fallback."""

    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device))
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA decode route target does not match the device")
    return load_cake_fmha_decode_native_bf16_module(
        route.target,
        route.batch_size,
        route.q_len,
        route.num_q_heads,
        route.num_kv_heads,
        has_sink=route.has_sink,
        has_window=route.has_window,
        use_scale_ptr=route.use_scale_ptr,
        retain_kv_l2=route.retain_kv_l2,
    )


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
) -> CakeFmhaContextRoute | None:
    """Select an exact product context route without broadening its contract."""

    if batch_size <= 0 or max_q_len <= 0 or max_kv_len <= 0 or window_left != -1:
        return None
    if isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor):
        return None
    if skip_softmax_threshold_scale_factor not in (None, 0.0):
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
    if seq_lens.dtype not in (torch.int32, torch.uint32):
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
    component = None
    if (
        dtypes == (torch.bfloat16,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout == "HND"
        and no_block_scales
        and float(bmm2_scale) == 1.0
    ):
        component = "context_bf16"
    elif (
        dtypes == (torch.float8_e4m3fn,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout == "HND"
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
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and is_causal
        and sinks is None
        and lse is None
        and key_block_scales is not None
        and value_block_scales is not None
        and key_block_scales.dtype == torch.float8_e4m3fn
        and value_block_scales.dtype == torch.float8_e4m3fn
        and key_block_scales.shape == value_block_scales.shape
        and key_block_scales.shape[:3] == key_cache.shape[:3]
        and key_block_scales.shape[3] == 8
        and key_block_scales.stride(3) == 1
        and value_block_scales.stride(3) == 1
    ):
        component = "context_nvfp4"
    if component is None:
        return None

    if component in ("context_fp16_hd256", "context_fp8_hd256"):
        pack_g = 1
        num_m_blocks = (max_q_len + 127) // 128
        l2_swizzle = 1
    else:
        pack_g = _context_pack_g(max_q_len, max_kv_len, num_q_heads, num_kv_heads)
        tok_per_stage = 128 // pack_g
        num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (2 * tok_per_stage)
        total_bh = batch_size * (num_q_heads // pack_g)
        l2_swizzle = 8 if total_bh % 8 == 0 else 1
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
    )


def get_cake_fmha_context_module(
    device: torch.device, route: CakeFmhaContextRoute | None
):
    """Load an optimized context module, or the authenticated portable fallback."""

    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device))
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA context route target does not match the device")
    loader = (
        load_cake_fmha_context_bf16_module
        if route.component == "context_bf16"
        else load_cake_fmha_context_fp8_module
    )
    return loader(
        route.target,
        route.num_m_blocks,
        route.num_q_heads,
        route.num_kv_heads,
        route.pack_g,
        route.page_size,
        route.l2_swizzle,
        is_causal=route.is_causal,
        return_lse=route.return_lse,
        enable_sink=route.enable_sink,
    )


def cake_fmha_manifest() -> dict[str, Any]:
    """Return a copy of the authenticated product/capability manifest."""

    return copy.deepcopy(get_cake_fmha_manifest())


def cake_batch_decode_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-decode ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_decode_with_kv_cache`.  The Cake backend is
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
    :func:`flashinfer.trtllm_batch_context_with_kv_cache`.  The Cake backend is
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
    "cake_batch_context_with_kv_cache",
    "cake_batch_decode_with_kv_cache",
    "cake_fmha_manifest",
    "cake_fmha_route_is_optimized",
    "get_cake_fmha_context_module",
    "get_cake_fmha_decode_module",
    "get_cake_fmha_module",
    "select_cake_fmha_context_route",
    "select_cake_fmha_decode_route",
]
