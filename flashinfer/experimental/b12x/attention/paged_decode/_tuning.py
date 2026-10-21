"""Configuration contract for paged decode/verify attention."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from b12x.preparation import (
    FrozenMapping,
    Knob,
    ParameterBinding,
    ParameterSpace,
    TuningContract,
)

from ._kernel import SMEM_LIMIT, default_max_splits

HEAD_DIMS = ((192, 128), (128, 128))
KV_DTYPES = ("bfloat16", "float8_e4m3fn")
PAGE_SIZES = (64, 128)
MAX_Q_PER_REQ = 8
MAX_SPLITS = 128
DESCALE_LAYOUTS = ("tensor", "request", "head")
_CONSUMER_WARPS = 8


@dataclass(frozen=True, kw_only=True)
class PagedDecodeQuery:
    """Geometry and masking contract of one layer family."""

    kv_dtype: str
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    max_batch: int
    max_q_per_req: int
    window_left: int
    causal: bool
    has_sinks: bool
    descale_layout: str
    sm_count: int

    @property
    def group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def max_total_q(self) -> int:
        return self.max_batch * self.max_q_per_req


@dataclass(frozen=True, kw_only=True)
class PagedDecodeConfig:
    """Kernel knobs: split-KV partition and CTA organization."""

    split_kv: bool
    max_splits: int
    head_splits: int
    stage_rows: int
    num_stages: int
    min_tiles_per_split: int

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "PagedDecodeConfig":
        expected = {f.name for f in fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                f"paged_decode config fields must be {sorted(expected)}, got {sorted(payload)}"
            )
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_query(query: PagedDecodeQuery) -> None:
    if not isinstance(query, PagedDecodeQuery):
        raise TypeError("paged_decode query must be PagedDecodeQuery")
    if query.kv_dtype not in KV_DTYPES:
        raise ValueError(f"KV dtype must be one of {KV_DTYPES}, got {query.kv_dtype!r}")
    if (query.head_dim_qk, query.head_dim_vo) not in HEAD_DIMS:
        raise ValueError(f"head dims must be one of {HEAD_DIMS}")
    if query.page_size not in PAGE_SIZES:
        raise ValueError(f"page size must be one of {PAGE_SIZES}")
    if (
        query.num_kv_heads <= 0
        or query.num_q_heads <= 0
        or query.num_q_heads % query.num_kv_heads
    ):
        raise ValueError("query heads must be a positive multiple of KV heads")
    if not 1 <= query.max_q_per_req <= MAX_Q_PER_REQ:
        raise ValueError(f"max_q_per_req must be in [1, {MAX_Q_PER_REQ}]")
    if query.max_batch <= 0 or query.sm_count <= 0:
        raise ValueError("max_batch and sm_count must be positive")
    if query.window_left < -1:
        raise ValueError("window_left must be -1 (none) or nonnegative")
    if query.descale_layout not in DESCALE_LAYOUTS:
        raise ValueError(f"descale_layout must be one of {DESCALE_LAYOUTS}")
    if not any(_rows_fit(query, hs) for hs in _head_split_choices(query)):
        raise ValueError("no head split fits the query rows into one CTA")


def _head_split_choices(query: PagedDecodeQuery) -> tuple[int, ...]:
    return tuple(hs for hs in (1, 2, 4, 8, 16) if query.group_size % hs == 0)


def _rows_fit(query: PagedDecodeQuery, head_splits: int) -> bool:
    rows = (query.group_size // head_splits) * query.max_q_per_req
    return (rows + 15) // 16 <= _CONSUMER_WARPS


def _stage_bytes(query: PagedDecodeQuery, stage_rows: int, num_stages: int) -> int:
    """Shared payload: the K/V stage ring, reused afterwards by the combine."""
    plane = stage_rows * 128  # one 128-byte TMA plane row per token
    if query.kv_dtype == "float8_e4m3fn":
        planes = (query.head_dim_qk + 127) // 128 + 1
    else:
        planes = (query.head_dim_qk + query.head_dim_vo) // 64
    combine = (
        _CONSUMER_WARPS * (query.head_dim_vo // 16) * 8 * 32
        + 2 * _CONSUMER_WARPS * 2 * 32
    ) * 4
    return max(planes * num_stages * plane, combine)


def default_config(query: PagedDecodeQuery, device=None) -> PagedDecodeConfig:
    """Measured defaults on SM120 (RTX PRO 6000, 188 SMs).

    Full-context layers split the KV range across CTAs.  Sliding-window layers
    scan one window per request: no split, with the GQA group sliced across
    CTAs instead.  Non-causal (speculative-drafter) layers pick by batch size.
    """
    hs_choices = [hs for hs in _head_split_choices(query) if _rows_fit(query, hs)]
    if not query.causal:
        if query.max_batch <= 2:
            split, hs = True, 4
        elif query.max_batch <= 12:
            split, hs = True, 1
        else:
            split, hs = False, 2
    elif query.window_left < 0:
        split, hs = True, 1
    else:
        split, hs = False, 4
    if hs not in hs_choices:
        hs = min(hs_choices, key=lambda value: abs(value - hs))
    # FP8 stages are half the bytes, so the ring runs deeper.
    stages = (
        (16 if query.head_dim_qk <= 128 else 12)
        if query.kv_dtype == "float8_e4m3fn"
        else 9
    )
    while stages > 2 and _stage_bytes(query, 16, stages) + 2048 > SMEM_LIMIT:
        stages -= 1
    splits = (
        default_max_splits(
            query.max_batch, query.num_kv_heads, query.sm_count, MAX_SPLITS
        )
        if split
        else 1
    )
    return PagedDecodeConfig(
        split_kv=split,
        max_splits=splits,
        head_splits=hs,
        stage_rows=16,
        num_stages=stages,
        min_tiles_per_split=4,
    )


def validate_config(
    query: PagedDecodeQuery, config: PagedDecodeConfig, device=None
) -> None:
    if not isinstance(config, PagedDecodeConfig):
        raise TypeError("paged_decode config must be PagedDecodeConfig")
    if config.head_splits not in _head_split_choices(query) or not _rows_fit(
        query, config.head_splits
    ):
        raise ValueError(
            f"head_splits={config.head_splits} does not fit the query rows"
        )
    if config.stage_rows not in (16, 32) or query.page_size % config.stage_rows:
        raise ValueError("stage_rows must be 16 or 32 and divide the page size")
    if config.num_stages < 2 or (
        _stage_bytes(query, config.stage_rows, config.num_stages) + 2048 > SMEM_LIMIT
    ):
        raise ValueError("stage ring does not fit shared memory")
    if config.min_tiles_per_split < 1:
        raise ValueError("min_tiles_per_split must be positive")
    if config.split_kv:
        if not 1 <= config.max_splits <= MAX_SPLITS:
            raise ValueError(f"max_splits must be in [1, {MAX_SPLITS}]")
    elif config.max_splits != 1:
        raise ValueError("unsplit configs use max_splits=1")


def _validate_query(query, device):
    validate_query(query)


def _parameters(query, device):
    # The best CTA geometry depends on the live context length, which a
    # plan-time benchmark never sees (preparation runs short synthetic
    # sequences, where more head slices always win while at long context each
    # slice re-streams the whole KV range).  Offer only the measured default;
    # explicit configs can still select any fitting head split.
    hs = default_config(query, device).head_splits
    return ParameterSpace.create(TUNING.knobs, values={"head_splits": (hs,)})


def _materialize(query, device, choice):
    base = default_config(query, device)
    return PagedDecodeConfig(**{**base.to_dict(), **dict(choice)})


TUNING = TuningContract(
    component_id="attention.paged_decode",
    query_schema_version=22,
    config_schema_version=22,
    query_fields=frozenset(field.name for field in fields(PagedDecodeQuery)),
    config_fields=frozenset(field.name for field in fields(PagedDecodeConfig)),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: PagedDecodeConfig(**dict(payload)),
    validate_query=_validate_query,
    validate_config=validate_config,
    default_config=default_config,
    knobs=(Knob(name="head_splits", values=None, binding=ParameterBinding.COMPILE),),
    candidate_contract_version=1,
    parameters=_parameters,
    materialize=_materialize,
)
