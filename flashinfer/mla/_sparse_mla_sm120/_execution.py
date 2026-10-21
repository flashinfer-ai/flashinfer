from tvm_ffi import Module as ExecutionPlan
from typing import NamedTuple, Optional, TypeVar, TypedDict, cast
from collections.abc import Iterator, Mapping
from types import MappingProxyType
import functools

import torch

from ...jit.mla import (
    gen_sparse_mla_dsv4_nvfp4_cache_ops_module,
    gen_sparse_mla_sm120_module,
)

K = TypeVar("K")


class Dsv4Nvfp4FormatInfo(TypedDict):
    query_dim: int
    value_dim: int
    bytes_per_token: int
    chunk_width: int
    page_size: int
    runtime_page: bool
    heads: tuple[int, ...]
    topks: tuple[int, ...]
    extra_page_sizes: tuple[int, ...]


MODEL_FAMILIES = {
    0: "dsv3_2",
    1: "dsv4",
    2: "glm_nsa",
    3: "glm53_nope",
    4: "dots3_swa",
    5: "dsv4_1",
    6: "dsv4_1_inline",
}


@functools.cache
def get_sparse_mla_sm120_module() -> ExecutionPlan:
    return gen_sparse_mla_sm120_module().build_and_load()


def get_sparse_mla_dsv4_nvfp4_module() -> ExecutionPlan:
    """Compat alias: the NVFP4 route lives in the unified SM120 module."""
    return get_sparse_mla_sm120_module()


@functools.cache
def get_sparse_mla_dsv4_nvfp4_cache_ops_module() -> ExecutionPlan:
    """Cache pack / append kernels, built for the current device (SM100 / SM103 / SM120 / SM121)."""
    return gen_sparse_mla_dsv4_nvfp4_cache_ops_module().build_and_load()


# NVFP4 exports carry a dsv4_nvfp4_ prefix in the unified module to avoid
# colliding with the FP8-family entry points.
_NVFP4_PREFIXED = frozenset(
    {"inspect_metadata", "execute_attention", "format_info", "resolve_attention"}
)


def query(name: str, *args, is_dsv4_nvfp4: bool = False):
    if is_dsv4_nvfp4 and name in _NVFP4_PREFIXED:
        name = "dsv4_nvfp4_" + name
    return getattr(get_sparse_mla_sm120_module(), name)(*args)


KV_SCALE_FORMATS = frozenset(
    {"auto", "pow2_fp32", "arbitrary_fp32", "ue8m0_g32", "ue8m0_g32_inline"}
)

# DeepSeek-V4.1 inline-scale records: each token's data bytes are followed by
# its own scale bytes, so the records are rows of the whole cache (FlashMLA's
# V4.1 ABI as of its 2026.09.30 release). The footer form keeps
# ``page_size * data`` bytes ahead of ``page_size * scales`` bytes per page.
# The attention kernels and the cache writers address these records natively.
_INLINE_SCALE_SUFFIX = "_inline"
DSV41_INLINE_ROW_BYTES = 528
DSV41_INLINE_FP4_ROW_BYTES = 288

_PRECISION_CODES = {"fp8": 0, "default": 1, "bf16": 2}


def _precision_code(precision: str) -> int:
    try:
        return _PRECISION_CODES[precision]
    except KeyError:
        raise ValueError(f"unsupported compute precision {precision!r}") from None


def normalize_kv_scale_format(kv_scale_format: str) -> str:
    fmt = str(kv_scale_format).lower().replace("-", "_")
    if fmt not in KV_SCALE_FORMATS:
        raise ValueError(
            "kv_scale_format must be one of "
            f"{sorted(KV_SCALE_FORMATS)}, got {kv_scale_format!r}"
        )
    return fmt


def split_inline_scale_format(kv_scale_format: str) -> tuple[str, bool]:
    """Return ``(footer_scale_format, inline)`` for a scale format."""
    fmt = normalize_kv_scale_format(kv_scale_format)
    if fmt.endswith(_INLINE_SCALE_SUFFIX):
        return fmt[: -len(_INLINE_SCALE_SUFFIX)], True
    return fmt, False


@functools.cache
def resolve_model_type(d_qk: int, kv_scale_format: str) -> int:
    return query("resolve_format", d_qk, normalize_kv_scale_format(kv_scale_format))


def check_inline_scale_cache(cache: torch.Tensor, row_bytes: int, name: str) -> None:
    """Check that an inline-scale DSv4.1 cache holds rows of the whole cache.

    Accepts a 2D ``[num_rows, row_bytes]`` cache (one-row pages), a 3D
    ``[num_pages, page_size, row_bytes]`` cache, or the 4D HND / NHD forms with
    a singleton head axis. Records sit at a uniform row pitch: the token-axis
    stride of multi-row pages, the page stride of one-row pages. The pitch is a
    16-byte multiple of at least ``row_bytes`` and the page stride a whole
    number of rows, so row ``page * (page_stride // pitch) + slot`` is slot
    ``slot`` of page ``page``; the kernels and writers address records as
    ``row * pitch``.
    """
    if cache.dtype != torch.uint8:
        raise ValueError(f"{name} must be a uint8 cache, got {cache.dtype}")
    if cache.size(-1) != row_bytes or cache.stride(-1) != 1:
        raise ValueError(
            f"{name} must end in a contiguous {row_bytes}-byte record axis, got "
            f"shape {tuple(cache.shape)} and strides {tuple(cache.stride())}"
        )
    if cache.ndim == 2:
        num_pages, page_size = cache.shape[0], 1
        page_stride = row_stride = cache.stride(0)
    elif cache.ndim == 3:
        num_pages, page_size = cache.shape[0], cache.shape[1]
        page_stride, row_stride = cache.stride(0), cache.stride(1)
    elif cache.ndim == 4 and cache.shape[1] == 1:
        num_pages, page_size = cache.shape[0], cache.shape[2]
        page_stride, row_stride = cache.stride(0), cache.stride(2)
    elif cache.ndim == 4 and cache.shape[2] == 1:
        num_pages, page_size = cache.shape[0], cache.shape[1]
        page_stride, row_stride = cache.stride(0), cache.stride(1)
    else:
        raise ValueError(
            f"{name} must be [num_rows, {row_bytes}], [num_pages, page_size, "
            f"{row_bytes}] or a 4D HND/NHD form, got shape {tuple(cache.shape)}"
        )
    if num_pages * page_size <= 1:
        return  # a single record has no pitch
    pitch = page_stride if page_size == 1 else row_stride
    if pitch < row_bytes or pitch % 16:
        raise ValueError(
            f"{name}: the inline-scale row pitch must be a 16-byte multiple of at "
            f"least {row_bytes} bytes, got {pitch}"
        )
    if page_size != 1 and num_pages > 1 and page_stride % pitch:
        raise ValueError(
            f"{name}: inline-scale records are rows of the whole cache, so the page "
            f"stride must be a whole number of rows; got row pitch {pitch} and "
            f"page stride {page_stride}"
        )


def check_inline_scale_caches(
    kv_scale_format: str,
    cache: torch.Tensor,
    extra: Optional[torch.Tensor],
    extra_fp4: bool,
) -> None:
    """Check the caches of an inline-scale call; footer formats pass."""
    if not normalize_kv_scale_format(kv_scale_format).endswith(_INLINE_SCALE_SUFFIX):
        return
    check_inline_scale_cache(cache, DSV41_INLINE_ROW_BYTES, "kv_cache")
    if extra is not None:
        check_inline_scale_cache(
            extra,
            DSV41_INLINE_FP4_ROW_BYTES if extra_fp4 else DSV41_INLINE_ROW_BYTES,
            "extra_kv_cache",
        )


@functools.cache
def format_info(model: int) -> Mapping[str, int]:
    return MappingProxyType(dict(query("format_info", model)))


@functools.cache
def main_page_sizes(model: int) -> tuple[int, ...]:
    """Instantiated main-cache page sizes; empty for runtime-page families."""
    return tuple(query("main_page_sizes", model))


@functools.cache
def dsv4_nvfp4_format_info() -> Dsv4Nvfp4FormatInfo:
    facts = query("format_info", is_dsv4_nvfp4=True)
    return cast(
        Dsv4Nvfp4FormatInfo,
        MappingProxyType(
            {
                key: tuple(value)
                if key in ("heads", "topks", "extra_page_sizes")
                else value
                for key, value in facts.items()
            }
        ),
    )


class FormatValues(Mapping[K, int]):
    def __init__(self, models: Mapping[K, int], field: str) -> None:
        self.models, self.field = models, field

    def __getitem__(self, key: K) -> int:
        return format_info(self.models[key])[self.field]

    def __iter__(self) -> Iterator[K]:
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)


class AttentionMetadata(NamedTuple):
    model: int
    tokens: int
    heads: int
    topk: int
    extra_topk: int
    page_size: int
    extra_page_size: int
    page_stride_bytes: int
    extra_page_stride_bytes: int
    row_stride_bytes: int
    indices_stride: int
    extra_indices_stride: int
    lse_stride: int
    has_lengths: bool
    has_extra_lengths: bool
    has_sink: bool
    extra_fp4: bool
    variant: int
    # Extra-cache row advance (the inline-scale row pitch); 0 without one.
    extra_row_stride_bytes: int = 0


@functools.cache
def metadata_candidates(
    metadata: AttentionMetadata, precision: str, sm_count: int, max_shared_bytes: int
) -> Mapping[int, int]:
    """Legal variants and resolver-owned chunk capacities for actual metadata."""
    numeric = _precision_code(precision)
    return MappingProxyType(
        dict(
            query("metadata_candidates", metadata, numeric, sm_count, max_shared_bytes)
        )
    )


def resolve_dsv4_nvfp4(
    *,
    tokens: int,
    heads: int,
    topk: int,
    extra_topk: int,
    page_size: int,
    extra_page_size: int,
    page_stride_bytes: int,
    extra_page_stride_bytes: int,
    cpb: int,
    sm_count: int,
    max_shared_bytes: int,
    prefill: bool = False,
    stage1_only: bool = False,
    has_lengths: bool = False,
    has_extra_lengths: bool = False,
    has_sink: bool = False,
) -> ExecutionPlan:
    module = get_sparse_mla_dsv4_nvfp4_module()
    return module.resolve_dsv4_nvfp4(
        tokens,
        heads,
        topk,
        extra_topk,
        page_size,
        extra_page_size,
        page_stride_bytes,
        extra_page_stride_bytes,
        cpb,
        sm_count,
        max_shared_bytes,
        prefill,
        stage1_only,
        has_lengths,
        has_extra_lengths,
        has_sink,
    )


def resolve_attention(
    *,
    model: int,
    tokens: int,
    heads: int,
    topk: int,
    extra_topk: int,
    page_size: int,
    extra_page_size: int,
    page_stride_bytes: int,
    extra_page_stride_bytes: int,
    row_stride_bytes: int,
    indices_stride: int,
    extra_indices_stride: int,
    lse_stride: int,
    has_lengths: bool,
    has_extra_lengths: bool,
    has_sink: bool,
    extra_fp4: bool,
    variant: int,
    precision: str,
    cpb: int,
    sm_count: int,
    max_shared_bytes: int,
    extra_row_stride_bytes: int = 0,
) -> ExecutionPlan:
    metadata = (
        model,
        tokens,
        heads,
        topk,
        extra_topk,
        page_size,
        extra_page_size,
        page_stride_bytes,
        extra_page_stride_bytes,
        row_stride_bytes,
        indices_stride,
        extra_indices_stride,
        lse_stride,
        int(has_lengths),
        int(has_extra_lengths),
        int(has_sink),
        int(extra_fp4),
        variant,
        extra_row_stride_bytes,
    )
    requested = _precision_code(precision)
    return get_sparse_mla_sm120_module().resolve_attention(
        metadata, requested, cpb, sm_count, max_shared_bytes
    )
