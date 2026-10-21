"""Metadata-only compilation and retained Mamba scan launches."""

from dataclasses import dataclass

import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._tuning import MambaQuery, TUNING


@dataclass(frozen=True)
class _State:
    query: MambaQuery
    device: torch.device
    program: object

    @property
    def required_workspace(self):
        from ._chunked import workspace_elements

        return workspace_elements(self.query) * 4 if self.query.max_tokens >= 128 else 0

    def run(
        self,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        state,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        state_is_fresh,
        num_seqs,
        num_tokens,
        *,
        out,
        workspace=None,
    ):
        from .api import query_from_call, _overlap, _validate_aliases
        from ._kernel import launch

        tensors = (
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias,
            state,
            cu_seqlens,
            initial_state_indices,
            final_state_indices,
            state_is_fresh,
            num_seqs,
            num_tokens,
        )
        query = query_from_call(*tensors, out=out, dt_softplus=self.query.dt_softplus)
        if query != self.query or x.device != self.device:
            raise ValueError(
                "Mamba operands must preserve the prepared capacities, dtypes, and strides"
            )
        operands = (*tensors, out)
        _validate_aliases(operands)
        if self.required_workspace:
            if (
                workspace is None
                or workspace.device != self.device
                or workspace.dtype != torch.uint8
                or not workspace.is_contiguous()
                or workspace.numel() < self.required_workspace
                or workspace.data_ptr() % 4
            ):
                raise ValueError(
                    f"Mamba requires {self.required_workspace} bytes of aligned caller-owned CUDA scratch"
                )
            if any(_overlap(workspace, t) for t in operands):
                raise ValueError("Mamba scratch must not alias operands")
            operands = (*operands, workspace)
        launch(self.program, query, operands)
        return out


@program_cache(scope="preparation")
def compile_pipeline(payload, config_payload, ordinal):
    from ._kernel import compile_scan
    from ._chunked import compile_chunked

    query = MambaQuery(**dict(payload))
    geometry = tuple(sorted(payload.items()))
    program = (
        compile_chunked(geometry, tuple(sorted(config_payload.items())), ordinal)
        if query.max_tokens >= 128
        else compile_scan(geometry, ordinal)
    )
    return attach_programs(
        _State(query, torch.device("cuda", ordinal), program), program.raw
    )


def plan(query: MambaQuery, *, device, invocation=FrozenMapping(), override=None):
    """Declare a scan; PreparationSession owns compilation and warmup."""
    if invocation:
        raise ValueError("Mamba invocation is fully described by MambaQuery")

    def jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.sequence.mamba_ssd._preparation:compile_pipeline",
                TUNING.encode_query(query),
                TUNING.encode_config(config),
                detected.ordinal,
            ),
        )

    def materialize(selection, detected):
        state = compile_pipeline(
            TUNING.encode_query(query),
            TUNING.encode_config(selection.config),
            detected.ordinal,
        )
        load_programs(state)
        return state

    def memory(config, detected):
        from ._chunked import workspace_elements

        return MemoryRequirements(
            scratch=(
                ScratchBufferSpec(
                    name="workspace",
                    shape=(workspace_elements(query) * 4,),
                    dtype=torch.uint8,
                    device=torch.device("cuda", detected.ordinal),
                ),
            )
            if query.max_tokens >= 128
            else ()
        )

    return Plan(
        contract=TUNING,
        query=query,
        override=override,
        _device=device,
        shared=True,
        _compile_jobs=jobs,
        _memory_requirements=memory,
        _materialize=materialize,
    )
