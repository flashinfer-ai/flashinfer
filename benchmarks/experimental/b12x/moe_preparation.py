"""Prepared-call adapters for native fused-MoE benchmarks."""
from __future__ import annotations

from collections.abc import Callable

import torch

from b12x.preparation import PreparedCall
from b12x.preparation.types import require_prepared


def prepared_call(*, output: object, bind: Callable[[object, object], object]) -> Callable[[object], PreparedCall]:
    """Bind one benchmark invocation with the session-owned exact-M scratch."""
    def factory(state: object) -> PreparedCall:
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in state.scratch.scratch_specs()
        )
        binding = bind(state, scratch)
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    return factory

def scratch_for(plan: object) -> tuple[torch.Tensor, ...]:
    """Allocate caller-owned scratch for one prepared exact-M plan."""
    state = require_prepared(plan, plan.component_id)
    return tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in state.scratch.scratch_specs()
    )

def request_for_capacity(declaration: object, *, name: str,
                         calls: dict[int, Callable[[object], PreparedCall]],
                         benchmark_calls: dict[int, Callable[[object], PreparedCall]] | None = None):
    """Make the scalar or exact-M composite request without a legacy warmup path."""
    counts = getattr(declaration, "token_counts", None)
    if benchmark_calls is None:
        benchmark_calls = calls
    if counts is None:
        if len(calls) != 1:
            raise ValueError("scalar MoE declaration requires one prepared call")
        return declaration.request(
            name=name, prepare_call=next(iter(calls.values())),
            benchmark_call=next(iter(benchmark_calls.values())),
        )
    return declaration.request(name=name, prepare_calls=calls, benchmark_calls=benchmark_calls)
