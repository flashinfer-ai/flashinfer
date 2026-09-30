"""Tuning contract for prepared paged GQA graph capacities."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from b12x.preparation._efficiency import capture_exhaustive_search, powers_of_two

from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation.tuning import Knob, ParameterBinding, ParameterSpace, TuningContract


def _compress_lut(values: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    for page_count, chunk_pages in enumerate(values, start=1):
        if runs and runs[-1][1] == int(chunk_pages):
            runs[-1] = (page_count, int(chunk_pages))
        else:
            runs.append((page_count, int(chunk_pages)))
    return tuple(runs)


def _expand_lut(
    runs: tuple[tuple[int, int], ...],
    *,
    page_count: int,
) -> tuple[int, ...]:
    values: list[int] = []
    previous_end = 0
    for end, chunk_pages in runs:
        if end <= previous_end or end > page_count or chunk_pages <= 0:
            raise ValueError("invalid paged GQA chunk-pages run table")
        values.extend([chunk_pages] * (end - previous_end))
        previous_end = end
    if previous_end != page_count:
        raise ValueError("paged GQA chunk-pages runs do not cover the capacity")
    return tuple(values)


def _factor_chunk_pages_lut(
    values: tuple[int, ...],
    *,
    max_chunks_per_request: int,
) -> tuple[int, ...]:
    max_chunks_per_request = int(max_chunks_per_request)
    if max_chunks_per_request <= 0:
        raise ValueError("paged GQA max_chunks_per_request must be positive")
    return tuple(
        chunk_pages
        if chunk_pages
        > (page_count + max_chunks_per_request - 1) // max_chunks_per_request
        else 1
        for page_count, chunk_pages in enumerate(values, start=1)
    )


@dataclass(frozen=True, kw_only=True)
class GqaQuery:
    device: object
    mode: str
    q_dtype: str
    kv_dtype: str
    q_heads: int
    kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    kv_cache_layout: str
    batch_size: int
    query_len: int
    cache_tokens: int
    window_left: int
    requested_graph_ctas_per_sm: int | None
    requested_max_work_items: int | None
    requested_max_partial_rows: int | None
    force_split_kv: bool | None
    abi: FrozenMapping
    controls: FrozenMapping
    exhaustive: bool = field(default_factory=capture_exhaustive_search)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "q_dtype": self.q_dtype,
            "kv_dtype": self.kv_dtype,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim_qk": self.head_dim_qk,
            "head_dim_vo": self.head_dim_vo,
            "page_size": self.page_size,
            "kv_cache_layout": self.kv_cache_layout,
            "batch_size": self.batch_size,
            "query_len": self.query_len,
            "cache_tokens": self.cache_tokens,
            "window_left": self.window_left,
            "requested_graph_ctas_per_sm": self.requested_graph_ctas_per_sm,
            "requested_max_work_items": self.requested_max_work_items,
            "requested_max_partial_rows": self.requested_max_partial_rows,
            "force_split_kv": self.force_split_kv,
            "abi": self.abi,
            "controls": self.controls,
            "exhaustive": self.exhaustive,
        }


@dataclass(frozen=True, kw_only=True)
class GqaConfig:
    graph_ctas_per_sm: int
    cta_tile_q: int
    query_tiles_per_request: int
    architecture_max_chunks_per_request: int
    max_chunks_per_request: int
    max_work_items: int
    max_partial_rows: int
    max_effective_kv_pages: int
    worst_page_count: int
    base_chunk_pages_runs: tuple[tuple[int, int], ...]

    @classmethod
    def from_capacity(cls, capacity: object) -> "GqaConfig":
        max_chunks_per_request = int(capacity.max_chunks_per_request)
        factored_lut = _factor_chunk_pages_lut(
            tuple(int(value) for value in capacity.chunk_pages_lut),
            max_chunks_per_request=max_chunks_per_request,
        )
        return cls(
            graph_ctas_per_sm=int(capacity.graph_ctas_per_sm),
            cta_tile_q=int(capacity.cta_tile_q),
            query_tiles_per_request=int(capacity.query_tiles_per_request),
            architecture_max_chunks_per_request=int(
                capacity.architecture_max_chunks_per_request
            ),
            max_chunks_per_request=max_chunks_per_request,
            max_work_items=int(capacity.max_work_items),
            max_partial_rows=int(capacity.max_partial_rows),
            max_effective_kv_pages=int(capacity.max_effective_kv_pages),
            worst_page_count=int(capacity.worst_page_count),
            base_chunk_pages_runs=_compress_lut(factored_lut),
        )

    @classmethod
    def from_config(cls, payload: FrozenMapping) -> "GqaConfig":
        fields = {
            "graph_ctas_per_sm",
            "cta_tile_q",
            "query_tiles_per_request",
            "architecture_max_chunks_per_request",
            "max_chunks_per_request",
            "max_work_items",
            "max_partial_rows",
            "max_effective_kv_pages",
            "worst_page_count",
            "base_chunk_pages_runs",
        }
        if set(payload) != fields:
            raise ValueError("paged GQA config fields do not match schema")
        raw_runs = payload["base_chunk_pages_runs"]
        if not isinstance(raw_runs, tuple):
            raise TypeError("paged GQA base_chunk_pages_runs must be an array")
        runs: list[tuple[int, int]] = []
        for raw_run in raw_runs:
            if not isinstance(raw_run, tuple) or len(raw_run) != 2:
                raise TypeError("paged GQA chunk-pages runs must be pairs")
            end, chunk_pages = raw_run
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (end, chunk_pages)
            ):
                raise TypeError("paged GQA chunk-pages runs must contain integers")
            runs.append((end, chunk_pages))
        scalar_values = {
            field: payload[field] for field in fields - {"base_chunk_pages_runs"}
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in scalar_values.values()
        ):
            raise TypeError("paged GQA scalar config fields must be integers")
        return cls(
            **scalar_values,
            base_chunk_pages_runs=tuple(runs),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "graph_ctas_per_sm": self.graph_ctas_per_sm,
            "cta_tile_q": self.cta_tile_q,
            "query_tiles_per_request": self.query_tiles_per_request,
            "architecture_max_chunks_per_request": (
                self.architecture_max_chunks_per_request
            ),
            "max_chunks_per_request": self.max_chunks_per_request,
            "max_work_items": self.max_work_items,
            "max_partial_rows": self.max_partial_rows,
            "max_effective_kv_pages": self.max_effective_kv_pages,
            "worst_page_count": self.worst_page_count,
            "base_chunk_pages_runs": [list(run) for run in self.base_chunk_pages_runs],
        }

    def chunk_pages_lut(self) -> tuple[int, ...]:
        base_lut = _expand_lut(
            self.base_chunk_pages_runs,
            page_count=self.max_effective_kv_pages,
        )
        return tuple(
            max(
                base_chunk_pages,
                (page_count + self.max_chunks_per_request - 1)
                // self.max_chunks_per_request,
            )
            for page_count, base_chunk_pages in enumerate(base_lut, start=1)
        )


def _heuristic(
    query: GqaQuery,
    _device: DeviceIdentity | None,
) -> GqaConfig:
    import torch

    from .planner import _plan_decode_graph_capacity_heuristic
    from ._controls import paged_controls

    with paged_controls(query.controls):
        capacity = _plan_decode_graph_capacity_heuristic(
            device=query.device,
            q_dtype=getattr(torch, query.q_dtype),
            kv_dtype=getattr(torch, query.kv_dtype),
            num_q_heads=query.q_heads,
            num_kv_heads=query.kv_heads,
            head_dim_qk=query.head_dim_qk,
            head_dim_vo=query.head_dim_vo,
            page_size=query.page_size,
            batch=query.batch_size,
            max_cache_page_count=(query.cache_tokens + query.page_size - 1) // query.page_size,
            window_left=query.window_left,
            graph_ctas_per_sm=query.requested_graph_ctas_per_sm,
            max_work_items=query.requested_max_work_items,
            max_partial_rows=query.requested_max_partial_rows,
            force_split_kv=query.force_split_kv,
        )
    return GqaConfig.from_capacity(capacity)


def _validate(
    query: GqaQuery,
    config: GqaConfig,
    _device: DeviceIdentity | None,
) -> None:
    if query.mode not in ("decode", "verify", "extend") or query.query_len <= 0:
        raise ValueError("unsupported paged GQA preparation mode")
    if query.kv_cache_layout not in ("separate", "combined"):
        raise ValueError("unsupported paged GQA KV-cache layout")
    if query.q_heads <= 0 or query.kv_heads <= 0:
        raise ValueError("paged GQA head counts must be positive")
    if query.q_heads % query.kv_heads:
        raise ValueError("paged GQA query heads must be divisible by KV heads")
    if query.page_size <= 0 or query.cache_tokens <= 0:
        raise ValueError("paged GQA page and cache capacities must be positive")
    positive = (
        config.graph_ctas_per_sm,
        config.cta_tile_q,
        config.query_tiles_per_request,
        config.architecture_max_chunks_per_request,
        config.max_chunks_per_request,
        config.max_work_items,
        config.max_effective_kv_pages,
        config.worst_page_count,
    )
    if any(value <= 0 for value in positive) or config.max_partial_rows < 0:
        raise ValueError("paged GQA profile capacities must be non-negative")
    expected_pages = (query.cache_tokens + query.page_size - 1) // query.page_size
    if query.window_left >= 0:
        expected_pages = min(
            expected_pages,
            max(1, (query.window_left + 2 * query.page_size - 1) // query.page_size),
        )
    if config.max_effective_kv_pages != expected_pages:
        raise ValueError(
            "paged GQA profile effective-page capacity does not match the query"
        )
    group_size = query.q_heads // query.kv_heads
    expected_query_tiles = (group_size + config.cta_tile_q - 1) // config.cta_tile_q
    if config.query_tiles_per_request != expected_query_tiles:
        raise ValueError("paged GQA profile query tile count is inconsistent")
    if config.max_chunks_per_request > (config.architecture_max_chunks_per_request):
        raise ValueError("paged GQA profile exceeds the architecture chunk limit")
    if not 1 <= config.worst_page_count <= config.max_effective_kv_pages:
        raise ValueError("paged GQA worst page count is outside the capacity")
    if (
        query.requested_graph_ctas_per_sm is not None
        and config.graph_ctas_per_sm != query.requested_graph_ctas_per_sm
    ):
        raise ValueError("paged GQA profile ignores requested graph CTA residency")
    if (
        query.requested_max_work_items is not None
        and config.max_work_items > query.requested_max_work_items
    ):
        raise ValueError("paged GQA profile exceeds requested work-item capacity")
    if (
        query.requested_max_partial_rows is not None
        and config.max_partial_rows > query.requested_max_partial_rows
    ):
        raise ValueError("paged GQA profile exceeds requested partial-row capacity")
    if query.force_split_kv is False and config.max_chunks_per_request != 1:
        raise ValueError("paged GQA direct-only query selected a split profile")
    lut = config.chunk_pages_lut()
    maximum_chunks = max(
        (page_count + chunk_pages - 1) // chunk_pages
        for page_count, chunk_pages in enumerate(lut, start=1)
    )
    if maximum_chunks > config.max_chunks_per_request:
        raise ValueError("paged GQA LUT exceeds max_chunks_per_request")
    direct_work = query.batch_size * config.query_tiles_per_request
    if config.max_work_items < direct_work:
        raise ValueError("paged GQA profile cannot hold the direct schedule")


def _validate_query(query: GqaQuery, _device: DeviceIdentity | None) -> None:
    if query.mode not in ("decode", "verify", "extend") or query.query_len <= 0:
        raise ValueError("unsupported paged GQA preparation mode")
    if query.kv_cache_layout not in ("separate", "combined"):
        raise ValueError("unsupported paged GQA KV-cache layout")
    if query.q_heads <= 0 or query.kv_heads <= 0 or query.q_heads % query.kv_heads:
        raise ValueError("paged GQA requires positive, divisible head counts")
    if query.page_size <= 0 or query.cache_tokens <= 0 or query.batch_size <= 0:
        raise ValueError("paged GQA capacities must be positive")


def _tuning_parameters(
    query: GqaQuery,
    device: DeviceIdentity | None,
):
    if query.requested_graph_ctas_per_sm is not None:
        graph_options = (query.requested_graph_ctas_per_sm,)
    else:
        if device is None or device.sm_count <= 0:
            raise ValueError("GQA tuning requires a device identity with an SM count")
        if min(query.batch_size, query.q_heads, query.kv_heads, query.page_size) <= 0:
            raise ValueError("GQA tuning geometry must be positive")
        if query.q_heads % query.kv_heads or query.cache_tokens <= 0:
            raise ValueError("GQA tuning requires divisible heads and positive cache capacity")
        pages = (query.cache_tokens + query.page_size - 1) // query.page_size
        if query.window_left >= 0:
            pages = min(
                pages,
                max(1, (query.window_left + 2 * query.page_size - 1) // query.page_size),
            )
        query_tiles = (query.q_heads // query.kv_heads + 15) // 16
        # Decode is M16. Ordinary schedules cannot split below one page;
        # the analytic FP8 route can split each page into 64-token stages
        # (planner._plan_decode_graph_capacity_heuristic). Cover both through
        # enough slots for every query tile and stage, not a residency favorite.
        # Higher positive residencies are accepted by production, but cannot
        # expose more nonempty work than this captured-capacity saturation.
        stages_per_page = max(query.page_size // 64, 1)
        work_ctas = (
            query.batch_size * query_tiles * pages * stages_per_page * query.kv_heads
        )
        maximum = max(1, (work_ctas + device.sm_count - 1) // device.sm_count)
        graph_options = range(1, maximum + 1)
    if query.requested_graph_ctas_per_sm is not None:
        residencies = frozenset(graph_options)
    else:
        import torch
        from ._controls import paged_controls
        from .planner import resolve_decode_graph_ctas_per_sm

        with paged_controls(query.controls):
            default_residency = resolve_decode_graph_ctas_per_sm(
                kv_dtype=getattr(torch, query.kv_dtype), batch=query.batch_size,
                page_size=query.page_size, head_dim_qk=query.head_dim_qk,
                head_dim_vo=query.head_dim_vo,
                gqa_group_size=query.q_heads // query.kv_heads,
            )
        residencies = powers_of_two(maximum) | {3, 6, maximum, default_residency}
    return ParameterSpace.create(TUNING.knobs, values={
        "graph_ctas_per_sm": graph_options,
        "force_split_kv": (
            (query.force_split_kv,)
            if query.force_split_kv is not None
            else (False, True)
        ),
    }, exhaustive=query.exhaustive,
        efficiency_predicates=(lambda p: p["graph_ctas_per_sm"] in residencies,),
    )


def _materialize_tuning(
    query: GqaQuery,
    device: DeviceIdentity | None,
    choice: FrozenMapping,
) -> GqaConfig:
    graph_ctas = choice["graph_ctas_per_sm"]
    if type(graph_ctas) is not int or graph_ctas <= 0:
        raise ValueError("GQA CTA residency must be a positive integer")
    if (
        query.requested_graph_ctas_per_sm is not None
        and graph_ctas != query.requested_graph_ctas_per_sm
    ):
        raise ValueError("GQA tuning choice conflicts with requested CTA residency")
    if (
        query.force_split_kv is not None
        and choice["force_split_kv"] is not query.force_split_kv
    ):
        raise ValueError("GQA tuning choice conflicts with requested split policy")
    # Only independent planner inputs are choices. Tile geometry, LUTs and
    # workspace capacities remain derived by the unchanged production planner.
    return _heuristic(
        replace(
            query,
            requested_graph_ctas_per_sm=graph_ctas,
            force_split_kv=choice["force_split_kv"],
        ),
        device,
    )


TUNING = TuningContract(
    component_id="attention.gqa",
    query_schema_version=7,
    config_schema_version=3,
    semantic_version=1,
    candidate_contract_version=3,
    query_fields=frozenset(GqaQuery.__dataclass_fields__) - {"device"},
    config_fields=frozenset(GqaConfig.__dataclass_fields__),
    encode_query=GqaQuery.to_dict,
    encode_config=GqaConfig.to_dict,
    decode_config=GqaConfig.from_config,
    validate_query=_validate_query,
    validate_config=_validate,
    default_config=_heuristic,
    knobs=(
        Knob(
            name="graph_ctas_per_sm",
            values=None,
            binding=ParameterBinding.COMPILE,
        ),
        Knob(
            name="force_split_kv",
            values=(False, True),
            binding=ParameterBinding.COMPILE,
        ),
    ),
    parameters=_tuning_parameters,
    materialize=_materialize_tuning,
)


__all__ = ["GqaConfig", "GqaQuery", "TUNING"]
