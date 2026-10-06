"""Prepared BF16 vocabulary projection declaration and prepared state."""
from __future__ import annotations

from collections.abc import Callable

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._contracts import Caps, Binding, _bind
from ._tuning import Bf16VocabProjectionConfig, Bf16VocabProjectionQuery, TUNING


@program_cache(scope="preparation")
def compile_vocab_projection(query_payload, config_payload, ordinal):
    """Compile exactly the selected native vocabulary launcher."""
    from . import _kernel

    query = Bf16VocabProjectionQuery(**dict(query_payload))
    config = Bf16VocabProjectionConfig.from_config(FrozenMapping(config_payload))
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    if config.backend == "torch":
        return {}
    kernel = _kernel._row_kernel if config.algorithm == "row" else _kernel._row_loop_kernel
    with torch.cuda.device(ordinal):
        return {"projection": kernel.warmup(
            torch.bfloat16, torch.bfloat16, torch.bfloat16,
            K=query.in_features, BLOCK_K=config.block_k, N=query.out_features,
            num_warps=config.num_warps, grid=(query.out_features, 1, 1),
        )}


@dataclass(frozen=True)
class _PreparedVocabProjection:
    caps: Caps
    query: Bf16VocabProjectionQuery
    config: Bf16VocabProjectionConfig
    runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def bind(self, *, plan, source, weight) -> Binding:
        return _bind(self.caps, plan=plan, source=source, weight=weight)

    def run(self, source: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.runner(source, weight)


def make_plan(caps: Caps, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(caps, Caps):
        raise TypeError("caps must be bf16_vocab_projection.Caps")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("BF16 vocabulary projection has no invocation metadata")
    query = Bf16VocabProjectionQuery(
        dtype="bfloat16", max_tokens=caps.max_tokens,
        in_features=caps.in_features, out_features=caps.out_features,
    )

    def compile_jobs(config, device):
        if config.backend == "torch":
            return ()
        return (CompileJob.create(
            "b12x.gemm.bf16_vocab_projection._preparation:compile_vocab_projection",
            TUNING.encode_query(query), config.to_dict(), device.ordinal,
        ),)

    def materialize(selection, device):
        config = selection.config
        if config.backend == "torch":
            runner = torch.nn.functional.linear
        else:
            programs = compile_vocab_projection(
                TUNING.encode_query(query), config.to_dict(), device.ordinal,
            )
            launcher = programs["projection"]

            def runner(source, weight):
                rows = int(source.shape[0])
                if rows < 1 or rows > query.max_tokens:
                    raise ValueError(
                        "vocabulary projection source rows differ from preparation"
                    )
                output = torch.empty(
                    (rows, query.out_features), dtype=torch.bfloat16,
                    device=source.device,
                )
                launcher[(query.out_features, rows, 1)](
                    source, weight, output, query.in_features, config.block_k,
                    query.out_features,
                )
                return output

        return _PreparedVocabProjection(caps, query, config, runner)

    return Plan(
        contract=TUNING, query=query, invocation=invocation, override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize, _device=caps.device,
    )


__all__ = ["make_plan", "compile_vocab_projection"]
