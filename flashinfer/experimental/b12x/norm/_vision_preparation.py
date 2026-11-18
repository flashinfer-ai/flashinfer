"""Prepared declarations for static native V4.1 vision operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)


@dataclass(frozen=True, kw_only=True)
class VisionQuery:
    operation: str
    channels: int
    heads: int
    ratio: int

    def __post_init__(self):
        if self.operation not in {"gelu", "merge", "rope"}:
            raise ValueError("unknown native vision operation")
        if self.channels <= 0 or self.heads <= 0 or self.ratio <= 0:
            raise ValueError("vision geometry must be positive")
        if self.operation != "rope" and self.heads != 1:
            raise ValueError("only rope has a head specialization")


TUNING = make_fixed_contract(
    component_id="norm.vision", query_type=VisionQuery, backend="cute"
)


@program_cache(scope="preparation")
def compile_vision(query_payload, ordinal):
    from .vision import _compile

    query = VisionQuery(**dict(query_payload))
    return _compile(query.operation, query.channels, query.heads, query.ratio, ordinal)[
        0
    ]


@dataclass(frozen=True)
class VisionState:
    query: VisionQuery
    compiled: object
    types: tuple[object, ...]


def plan(
    query: VisionQuery, *, device, invocation=FrozenMapping(), override=None
) -> Plan:
    if not isinstance(query, VisionQuery):
        raise TypeError("vision plan requires VisionQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("vision live rows and image grids are runtime inputs")
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("native vision requires CUDA")

    def materialize(selection, detected):
        del selection, detected
        from .vision import _compile

        compiled, types = _compile(
            query.operation, query.channels, query.heads, query.ratio, target.index
        )
        return VisionState(query, compiled, types)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=lambda config, detected: (
            CompileJob.create(
                "b12x.norm._vision_preparation:compile_vision",
                TUNING.encode_query(query),
                detected.ordinal,
            ),
        ),
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize,
        _device=target,
    )


__all__ = ["VisionQuery", "VisionState", "TUNING", "plan"]
