"""Metadata declaration and resolved launch ownership for the NVFP4 quantizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._impl import compile_bf16_to_fp4_tma
from ._tuning import TUNING, Nvfp4QuantizationConfig, Nvfp4QuantizationQuery


@program_cache(scope="preparation")
def compile_quantizer(query_payload, config_payload, ordinal):
    query = Nvfp4QuantizationQuery(**dict(query_payload))
    config = Nvfp4QuantizationConfig.from_config(config_payload)
    with torch.cuda.device(ordinal):
        return compile_bf16_to_fp4_tma(
            query.rows,
            query.columns,
            liveness_strategy=config.liveness_strategy,
        )


@dataclass(frozen=True)
class _Nvfp4ExecutionState:
    query: Nvfp4QuantizationQuery
    device: torch.device
    launch: object

    def run(self, x, global_scale, outputs):
        if x.device != self.device:
            raise ValueError("NVFP4 source and prepared plan devices differ")
        self.launch(x, global_scale, outputs.packed_a_flat, outputs.scale_flat)


def plan(
    m: int,
    k: int,
    *,
    invocation: FrozenMapping = FrozenMapping(),
    override: Nvfp4QuantizationConfig | None = None,
) -> Plan:
    query = Nvfp4QuantizationQuery(dtype="bfloat16", rows=int(m), columns=int(k))

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.quantization.nvfp4._preparation:compile_quantizer",
                TUNING.encode_query(query),
                config.to_dict(),
                device.ordinal,
            ),
        )

    def materialize(selection, device):
        launch = compile_quantizer(
            TUNING.encode_query(query), selection.config.to_dict(), device.ordinal
        )
        return _Nvfp4ExecutionState(query, torch.device("cuda", device.ordinal), launch)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )
