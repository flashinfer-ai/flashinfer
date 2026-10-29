"""Declarative MLA query-projection preparation."""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
from dataclasses import dataclass
import torch
from b12x._lib.compile_pool import CompileJob
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._tuning import ProjectionQuery, TUNING


@program_cache(scope="preparation")
def compile_bf16(query_payload, device_ordinal):
    from . import _bf16

    query = ProjectionQuery(**query_payload)
    output_fp8 = query.output_dtype == "float8_e4m3fn"
    with torch.cuda.device(device_ordinal):
        return _bf16._mla_query_projection_bf16_kernel.warmup(
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.float32 if output_fp8 else torch.bfloat16,
            torch.float8_e4m3fn if output_fp8 else torch.bfloat16,
            query.max_rows,
            query.max_rows * 192,
            192,
            192 * 512,
            512,
            query.heads * 64,
            64,
            query.heads * 576,
            576,
            OUTPUT_FP8=output_fp8,
            NOPE_DIM=192,
            LATENT_DIM=512,
            ROPE_DIM=64,
            BLOCK_M=16 if query.max_rows <= 16 else 32,
            BLOCK_N=32,
            BLOCK_K=64,
            num_warps=4,
            num_stages=2,
            grid=(16, query.heads, 1),
        )


@program_cache(scope="preparation")
def compile_mxfp8(query_payload, device_ordinal):
    from b12x.gemm._shared import mxfp8_bmm as kernels

    query = ProjectionQuery(**query_payload)
    return kernels._compile_mla_query_projection(
        b_major=kernels._coerce_b_major(query.b_major),
        groups=query.heads,
        m=query.max_rows,
        n=512,
        k=192,
        output_fp8=query.output_dtype == "float8_e4m3fn",
        device=torch.device("cuda", device_ordinal),
    )


@dataclass(frozen=True)
class _ProjectionExecutionState:
    query: ProjectionQuery
    device: torch.device
    launch: object

    def run(self, q_nope, weight, q_pe, out, *, q_scale=None, stream=None):
        if q_nope.device != self.device or int(q_nope.shape[1]) != self.query.max_rows:
            raise ValueError(
                "MLA query plan differs from its prepared device or exact M"
            )
        if self.query.weight_format == "mxfp8":
            from b12x.gemm._shared import mxfp8_bmm as kernels

            if not isinstance(weight, tuple):
                raise ValueError("prepared MXFP8 MLA plan requires native MXFP8 weight")
            return kernels._run_mla_query_prepared(
                q_nope,
                *kernels._rhs_tensors(weight),
                q_pe,
                q_scale,
                out,
                b_major=self.query.b_major,
                sf_axis=self.query.sf_axis,
                launch=self.launch,
                stream=stream,
            )
        if not isinstance(weight, torch.Tensor):
            raise ValueError("prepared BF16 MLA plan requires BF16 weight")
        from . import _bf16

        return _bf16._run_prepared(
            q_nope,
            weight,
            q_pe,
            q_scale,
            out,
            launcher=self.launch,
            block_m=16 if self.query.max_rows <= 16 else 32,
            output_fp8=self.query.output_dtype == "float8_e4m3fn",
            stream=stream,
        )


def plan(query: ProjectionQuery, *, invocation=FrozenMapping(), override=None) -> Plan:
    if not isinstance(query, ProjectionQuery):
        raise TypeError("MLA query plan requires ProjectionQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("MLA invocation semantics belong in ProjectionQuery")

    def compile_jobs(config, device):
        del config
        factory = (
            "b12x.gemm.mla_query_projection._preparation:compile_bf16"
            if query.weight_format == "bf16"
            else "b12x.gemm.mla_query_projection._preparation:compile_mxfp8"
        )
        return (CompileJob.create(factory, TUNING.encode_query(query), device.ordinal),)

    def materialize(selection, device):
        compiler = compile_bf16 if query.weight_format == "bf16" else compile_mxfp8
        return _ProjectionExecutionState(
            query,
            torch.device("cuda", device.ordinal),
            compiler(TUNING.encode_query(query), device.ordinal),
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=compile_jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )
