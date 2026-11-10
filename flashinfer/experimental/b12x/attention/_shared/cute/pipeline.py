from dataclasses import dataclass
from typing import Optional

import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import dsl_user_op, if_generate
from cutlass.pipeline import NamedBarrier as NamedBarrierOg
from cutlass.pipeline import PipelineAsync as PipelineAsyncOg
from cutlass.pipeline import PipelineState
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg
from cutlass.pipeline import PipelineUserType


class PipelineStateSimple:
    def __init__(self, stages: int, phase_index: Int32):
        self._stages = stages
        self._phase_index = phase_index

    def clone(self, *, loc=None, ip=None) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def index(self) -> Int32:
        return (
            Int32(0)
            if const_expr(self._stages == 1)
            else self._phase_index % self._stages
        )

    @property
    def phase(self) -> Int32:
        return (
            self._phase_index
            if const_expr(self._stages == 1)
            else self._phase_index // self._stages
        )

    def advance(self, *, loc=None, ip=None):
        if const_expr(self._stages == 1):
            self._phase_index ^= 1
        else:
            self._phase_index += 1

    def __extract_mlir_values__(self):
        return [self._phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    if type is PipelineUserType.Producer:
        return PipelineStateSimple(stages, Int32(stages))
    if type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    assert False, "invalid PipelineUserType"


@dataclass(frozen=True)
class PipelineTmaAsync(PipelineTmaAsyncOg):
    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineTmaAsyncOg.create(*args, **kwargs)
        object.__setattr__(obj, "__class__", PipelineTmaAsync)
        return obj

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
        *,
        loc=None,
        ip=None,
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(
                state.index, state.phase, loc=loc, ip=ip
            ),
            loc=loc,
            ip=ip,
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(
                state.index, self.producer_mask, loc=loc, ip=ip
            )
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(
                state.index, tx_count, loc=loc, ip=ip
            )
