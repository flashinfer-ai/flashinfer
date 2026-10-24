"""Prepared native DeepSeek V4.1 compressed-cache writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)
from b12x.preparation.types import require_prepared


@dataclass(frozen=True, kw_only=True)
class CacheWriterQuery:
    max_rows: int
    page_size: int
    cache_kind: str
    cache_format: str = "deepseek_v41"
    slot_dtype: str = "int64"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_query(query: CacheWriterQuery, _device) -> None:
    if not isinstance(query, CacheWriterQuery):
        raise TypeError("query must be CacheWriterQuery")
    if type(query.max_rows) is not int or query.max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    if type(query.page_size) is not int or query.page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if query.cache_kind not in ("swa", "indexed"):
        raise ValueError("cache_kind must be 'swa' or 'indexed'")
    if query.cache_format != "deepseek_v41":
        raise ValueError(
            "prepared compressed writer requires cache_format='deepseek_v41'"
        )
    if query.slot_dtype not in ("int32", "int64"):
        raise ValueError("slot_dtype must be 'int32' or 'int64'")


TUNING = replace(
    make_fixed_contract(
        component_id="attention.compressed_sparse_mla.cache_writer",
        query_type=CacheWriterQuery,
        backend="cute",
    ),
    query_schema_version=22,
    validate_query=_validate_query,
)


def page_nbytes(
    page_size: int,
    *,
    cache_kind: str,
    cache_format: str = "deepseek_v41",
) -> int:
    from b12x.attention._shared.mla.kv_cache import page_nbytes as _page_nbytes

    return _page_nbytes(page_size, cache_kind=cache_kind, cache_format=cache_format)


def _slot_dtype(name: str) -> torch.dtype:
    return torch.int64 if name == "int64" else torch.int32


def _representative(query: CacheWriterQuery, ordinal: int):
    from b12x.attention._shared.mla import kv_cache as writer

    device = torch.device("cuda", ordinal)
    record_bytes = 528 if query.cache_kind == "swa" else 288
    return (
        writer,
        torch.empty((query.max_rows, 512), dtype=torch.bfloat16, device=device),
        torch.empty(
            (1, query.page_size * record_bytes), dtype=torch.uint8, device=device
        ),
        torch.empty(
            (query.max_rows,), dtype=_slot_dtype(query.slot_dtype), device=device
        ),
    )


@program_cache(scope="preparation")
def compile_cache_writer(query_payload, ordinal):
    """Compiler-pool factory; intentionally private to the public API."""
    query = CacheWriterQuery(**dict(query_payload))
    _validate_query(query, None)
    with torch.cuda.device(ordinal), FakeTensorMode():
        writer, kv, cache, slots = _representative(query, ordinal)
        kernel, args, spec = writer._compressed_cache_writer_launch(
            kv, cache, slots, query.page_size, query.cache_kind
        )
        return writer.compile_cute(kernel, *args, compile_spec=spec)


@dataclass(frozen=True)
class _CacheWriterState:
    query: CacheWriterQuery
    compiled: object

    def run(
        self,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Prime or execute this exact prepared writer specialization."""
        self.write(
            kv,
            cache,
            slot_mapping,
            page_size=None,
            cache_kind=None,
            cache_format=None,
        )

    def write(
        self,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        page_size: int | None,
        cache_kind: str | None,
        cache_format: str | None,
    ) -> None:
        from b12x.attention._shared.mla import kv_cache as writer

        if page_size is not None and page_size != self.query.page_size:
            raise ValueError("page_size contradicts prepared cache-writer query")
        if cache_kind is not None and cache_kind != self.query.cache_kind:
            raise ValueError("cache_kind contradicts prepared cache-writer query")
        if cache_format is not None and cache_format != self.query.cache_format:
            raise ValueError("cache_format contradicts prepared cache-writer query")
        writer._validate_compressed_cache_writer(
            kv,
            cache,
            slot_mapping,
            self.query.page_size,
            self.query.cache_kind,
            self.query.cache_format,
        )
        if str(slot_mapping.dtype).removeprefix("torch.") != self.query.slot_dtype:
            raise ValueError(
                "slot_mapping dtype contradicts prepared cache-writer query"
            )
        if int(slot_mapping.shape[0]) > self.query.max_rows:
            raise ValueError("cache-writer rows exceed prepared capacity")
        if int(slot_mapping.shape[0]) == 0:
            return
        args = writer._compressed_cache_writer_args(
            kv, cache, slot_mapping, self.query.page_size, self.query.cache_kind
        )
        writer.run_compiled(self.compiled, args)


def plan(
    query: CacheWriterQuery,
    *,
    device: object,
    invocation: FrozenMapping = FrozenMapping(),
    override=None,
) -> Plan:
    """Declare one cache-writer specialization without allocating or compiling."""
    _validate_query(query, None)
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("cache-writer invocation metadata is not supported")
    device = torch.device(device)

    def jobs(_config, detected):
        return (
            CompileJob.create(
                "b12x.attention.compressed_sparse_mla.cache_writer:compile_cache_writer",
                query.to_dict(),
                detected.ordinal,
            ),
        )

    def materialize(_selection, detected):
        compiled = compile_cache_writer(query.to_dict(), detected.ordinal)
        load_programs(compiled)
        return attach_programs(_CacheWriterState(query, compiled), compiled)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=jobs,
        _memory_requirements=lambda _config, _device: MemoryRequirements(),
        _materialize=materialize,
        _device=device,
    )


def write_cache(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan: Plan,
    page_size: int | None = None,
    cache_kind: str | None = None,
    cache_format: str | None = None,
) -> None:
    state = require_prepared(
        plan, "attention.compressed_sparse_mla.cache_writer", kv.device
    )
    state.write(
        kv,
        cache,
        slot_mapping,
        page_size=page_size,
        cache_kind=cache_kind,
        cache_format=cache_format,
    )


__all__ = ["CacheWriterQuery", "page_nbytes", "plan", "write_cache"]
