"""Compile and retain a fused or separate hyperconnection gate invocation."""

from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from ._tuning import Config, Query, TUNING, projection_query, projection_config


@program_cache(scope="preparation")
def compile_gate(query_payload, config_payload, ordinal):
    import triton
    from ._kernel import projection_gate
    from ...norm.hyperconnection import _kernels as hc
    from ..bf16_gemv._preparation import compile_gemv
    from ..bf16_gemv._tuning import TUNING as GEMV_TUNING

    q, c = Query(**dict(query_payload)), Config(**dict(config_payload))
    with torch.cuda.device(ordinal):
        if c.implementation == "fused":
            return {
                "fused": projection_gate.warmup(
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                    torch.bfloat16,
                    q.max_rows,
                    H=q.hidden_size,
                    S=q.streams,
                    R=q.rank,
                    BM=c.tile_m,
                    BH=c.tile_h,
                    BK=c.tile_k,
                    num_warps=c.num_warps,
                    num_stages=c.num_stages,
                    grid=(
                        triton.cdiv(q.max_rows, c.tile_m),
                        triton.cdiv(q.hidden_size, c.tile_h),
                    ),
                )
            }
        programs = compile_gemv(
            GEMV_TUNING.encode_query(projection_query(q)),
            GEMV_TUNING.encode_config(projection_config(c)),
            ordinal,
        )
        return {
            **programs,
            "gate": hc._gate_mean_kernel.warmup(
                torch.bfloat16,
                torch.bfloat16,
                torch.bfloat16,
                HIDDEN_SIZE=q.hidden_size,
                STREAMS=q.streams,
                BLOCK_H=c.gate_block,
                num_warps=4,
                grid=(q.max_rows, triton.cdiv(q.hidden_size, c.gate_block)),
            ),
        }


@dataclass(frozen=True)
class State:
    query: Query
    config: Config
    device: torch.device
    programs: dict

    def run(self, source, weight, normalized, out, workspace):
        import triton
        from ._kernel import projection_gate
        from ...norm.hyperconnection import _kernels as hc
        from ...norm.hyperconnection._impl import _overlaps

        q, c = self.query, self.config
        rows = source.shape[0]
        if not 0 <= rows <= q.max_rows:
            raise ValueError("hyperconnection gate rows exceed prepared capacity")
        expected = (
            (rows, q.rank),
            (q.streams * q.hidden_size, q.rank),
            (rows, q.streams * q.hidden_size),
            (rows, q.hidden_size),
        )
        tensors = (source, weight, normalized, out)
        for tensor, shape in zip(tensors, expected):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "hyperconnection gate requires aligned contiguous BF16 operands of the prepared shapes"
                )
        if (
            workspace.ndim != 2
            or workspace.shape[0] < rows
            or workspace.shape[1] != q.streams * q.hidden_size
            or workspace.dtype != torch.bfloat16
            or workspace.device != self.device
            or not workspace.is_contiguous()
            or workspace.data_ptr() % 16
        ):
            raise ValueError(
                "hyperconnection gate workspace must cover the BF16 logit capacity"
            )
        if any(
            _overlaps(target, tensor)
            for target in (out, workspace)
            for tensor in tensors[:3]
        ) or _overlaps(out, workspace):
            raise ValueError(
                "hyperconnection gate output and workspace must not overlap inputs or each other"
            )
        if not rows:
            return
        if c.implementation == "fused":
            projection_gate[
                (triton.cdiv(rows, c.tile_m), triton.cdiv(q.hidden_size, c.tile_h))
            ](
                source,
                weight,
                normalized,
                out,
                rows,
                H=q.hidden_size,
                S=q.streams,
                R=q.rank,
                BM=c.tile_m,
                BH=c.tile_h,
                BK=c.tile_k,
                num_warps=c.num_warps,
                num_stages=c.num_stages,
            )
        else:
            logits = workspace[:rows]
            if c.implementation == "torch":
                torch.mm(source, weight.T, out=logits)
            else:
                self.programs["gemv"](source, weight, logits)
            hc._gate_mean_launch(
                normalized, logits, out, q.streams, q.hidden_size, c.gate_block
            )


def plan(query, *, invocation=FrozenMapping(), override=None):
    if invocation:
        raise ValueError("hyperconnection gate invocation belongs in Query")

    def materialize(selection, device):
        programs = compile_gate(
            TUNING.encode_query(query),
            TUNING.encode_config(selection.config),
            device.ordinal,
        )
        load_programs(programs)
        return attach_programs(
            State(
                query, selection.config, torch.device("cuda", device.ordinal), programs
            ),
            programs,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(),
        override=override,
        shared=True,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.gemm.hyperconnection_gate._preparation:compile_gate",
                TUNING.encode_query(query),
                TUNING.encode_config(config),
                device.ordinal,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )
