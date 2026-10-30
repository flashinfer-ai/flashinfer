"""Declarative preparation for the fixed rowwise-MXFP8 BMM."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._tuning import BmmQuery, TUNING


@program_cache(scope="preparation")
def compile_bmm(query_payload, device_ordinal):
    """Resolve the exact production launch used by a prepared BMM."""
    from b12x.gemm._shared import mxfp8_bmm as kernels

    query = BmmQuery(**query_payload)
    return kernels._compile(
        b_major=kernels._coerce_b_major(query.b_major),
        groups=query.batch,
        m=query.max_rows,
        n=query.out_features,
        k=query.in_features,
        device=torch.device("cuda", device_ordinal),
    )


@dataclass(frozen=True)
class _BmmExecutionState:
    query: BmmQuery
    launch: object
    device: torch.device

    def run(self, lhs, rhs, out, *, b_major, sf_axis, stream=None):
        from b12x.gemm._shared import mxfp8_bmm as kernels

        if b_major != self.query.b_major or sf_axis != self.query.sf_axis:
            raise ValueError("BMM layout differs from the prepared plan")
        if lhs.device != self.device:
            raise ValueError("BMM operands differ from the prepared device")
        if int(lhs.shape[1]) != self.query.max_rows:
            raise ValueError("BMM plan requires its exact planned M")
        return kernels._run_prepared(
            lhs,
            *kernels._rhs_tensors(rhs),
            out,
            b_major=b_major,
            sf_axis=sf_axis,
            launch=self.launch,
            stream=stream,
        )


def plan(query: BmmQuery, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(query, BmmQuery):
        raise TypeError("BMM plan requires BmmQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("BMM invocation semantics belong in BmmQuery")

    def compile_jobs(config, device):
        del config
        return (
            CompileJob.create(
                "b12x.gemm._bmm._preparation:compile_bmm",
                TUNING.encode_query(query),
                device.ordinal,
            ),
        )

    def memory(config, device):
        del config, device
        return MemoryRequirements()

    def materialize(selection, device):
        return _BmmExecutionState(
            query,
            compile_bmm(TUNING.encode_query(query), device.ordinal),
            torch.device("cuda", device.ordinal),
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=memory,
        _materialize=materialize,
    )
