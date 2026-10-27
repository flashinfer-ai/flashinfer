"""Metadata compilation and retained execution for unquantized embedding lookup."""

from b12x._lib.program_cache import program_cache
from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._tuning import EmbeddingQuery, TUNING


def query_from_call(weight, ids, *, out, num_rows=None):
    from .api import _check_tensors, _weight_width

    _check_tensors(weight, ids, out, num_rows)
    return EmbeddingQuery(
        max_rows=ids.numel(),
        table_rows=weight.shape[0],
        width=_weight_width(weight),
        row_stride=weight.stride(0),
        weight_dtype=str(weight.dtype).removeprefix("torch."),
        id_dtype=str(ids.dtype).removeprefix("torch."),
        device_count=num_rows is not None,
    )


@program_cache(scope="preparation")
def compile_lookup(payload, ordinal):
    from ._kernel import compile_embedding

    query = EmbeddingQuery(**dict(payload))
    compiled = compile_embedding(
        query.width,
        getattr(torch, query.weight_dtype),
        getattr(torch, query.id_dtype),
        ordinal,
    )
    return attach_programs(
        _EmbeddingState(
            query, torch.device("cuda", ordinal), compiled.raw, compiled.types
        ),
        compiled.raw,
    )


@dataclass(frozen=True)
class _EmbeddingState:
    query: EmbeddingQuery
    device: torch.device
    program: object
    types: tuple

    def run(self, weight, ids, *, out, num_rows=None):
        from .api import _check_tensors, _weight_width
        from ._kernel import launch

        _check_tensors(weight, ids, out, num_rows)
        q = self.query
        if (
            weight.device != self.device
            or (weight.shape[0] > q.table_rows or _weight_width(weight) != q.width)
            or weight.stride(0) != q.row_stride
            or ids.numel() > q.max_rows
            or weight.dtype != getattr(torch, q.weight_dtype)
            or ids.dtype != getattr(torch, q.id_dtype)
            or (num_rows is not None) != q.device_count
        ):
            raise ValueError(
                f"embedding tensors differ from prepared query {q}: "
                f"weight={tuple(weight.shape)}/{weight.stride()}/{weight.dtype}/{weight.device}, "
                f"ids={tuple(ids.shape)}/{ids.dtype}, device_count={num_rows is not None}"
            )
        launch(weight, ids, out, num_rows, prepared=(self.program, self.types))
        return out


def plan(query: EmbeddingQuery, *, device, invocation=FrozenMapping(), override=None):
    if invocation:
        raise ValueError("embedding invocation is fully described by EmbeddingQuery")

    def jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.sequence.embedding._preparation:compile_lookup",
                TUNING.encode_query(query),
                detected.ordinal,
            ),
        )

    def materialize(selection, detected):
        state = compile_lookup(TUNING.encode_query(query), detected.ordinal)
        load_programs(state)
        return state

    return Plan(
        contract=TUNING,
        query=query,
        override=override,
        _device=device,
        shared=True,
        _compile_jobs=jobs,
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize,
    )
