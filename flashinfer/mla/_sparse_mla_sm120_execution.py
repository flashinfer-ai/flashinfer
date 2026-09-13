from tvm_ffi import Module as ExecutionPlan
from typing import NamedTuple, TypeVar, TypedDict, cast
from collections.abc import Iterator, Mapping
from types import MappingProxyType
import functools

from ..jit.mla import gen_sparse_mla_sm120_module

K = TypeVar("K")


class Dsv4Nvfp4FormatInfo(TypedDict):
    query_dim: int
    value_dim: int
    bytes_per_token: int
    chunk_width: int
    page_size: int
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
}


@functools.cache
def get_sparse_mla_sm120_module() -> ExecutionPlan:
    return gen_sparse_mla_sm120_module().build_and_load()


def get_sparse_mla_dsv4_nvfp4_module() -> ExecutionPlan:
    """Compat alias: the NVFP4 route lives in the unified SM120 module."""
    return get_sparse_mla_sm120_module()


# NVFP4 exports carry a dsv4_nvfp4_ prefix in the unified module to avoid
# colliding with the FP8-family entry points.
_NVFP4_PREFIXED = frozenset(
    {"inspect_metadata", "execute_attention", "format_info", "resolve_attention"}
)


def query(name: str, *args, is_dsv4_nvfp4: bool = False):
    if is_dsv4_nvfp4 and name in _NVFP4_PREFIXED:
        name = "dsv4_nvfp4_" + name
    return getattr(get_sparse_mla_sm120_module(), name)(*args)


KV_SCALE_FORMATS = frozenset({"auto", "pow2_fp32", "arbitrary_fp32", "ue8m0_g32"})


def normalize_kv_scale_format(kv_scale_format: str) -> str:
    fmt = str(kv_scale_format).lower().replace("-", "_")
    if fmt not in KV_SCALE_FORMATS:
        raise ValueError(
            "kv_scale_format must be one of "
            f"{sorted(KV_SCALE_FORMATS)}, got {kv_scale_format!r}"
        )
    return fmt


@functools.cache
def resolve_model_type(d_qk: int, kv_scale_format: str) -> int:
    return query("resolve_format", d_qk, normalize_kv_scale_format(kv_scale_format))


@functools.cache
def format_info(model: int) -> Mapping[str, int]:
    return MappingProxyType(dict(query("format_info", model)))


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


def metadata_candidates(
    metadata: AttentionMetadata, precision: str, sm_count: int, max_shared_bytes: int
) -> Mapping[int, int]:
    """Legal variants and resolver-owned chunk capacities for actual metadata."""
    if precision not in ("default", "fp8", "bf16"):
        raise ValueError(f"unsupported compute precision {precision!r}")
    numeric = {"fp8": 0, "default": 1, "bf16": 2}[precision]
    return dict(
        query("metadata_candidates", metadata, numeric, sm_count, max_shared_bytes)
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
    )
    requested = {"fp8": 0, "default": 1, "bf16": 2}[precision]
    return get_sparse_mla_sm120_module().resolve_attention(
        metadata, requested, cpb, sm_count, max_shared_bytes
    )
