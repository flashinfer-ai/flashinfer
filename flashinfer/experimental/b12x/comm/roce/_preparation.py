"""Prepared launch ownership for a caller-created RoCE runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan, PreparedCall

from ._tuning import RoceQuery, TUNING


def _dtypes(query: RoceQuery) -> tuple[torch.dtype, ...]:
    values = tuple(query.call.get("dtypes", ()))
    if not values:
        raise ValueError("RoCE declaration requires call.dtypes")
    return tuple(
        getattr(torch, value) if isinstance(value, str) else value for value in values
    )


@program_cache(scope="preparation")
def compile_roce(payload, ordinal: int):
    """Resolve exactly the all-reduce/all-gather launchers in a declaration."""
    from . import _allgather_cute
    from ._oneshot_cute import get_launcher
    from .roce_oneshot import _DTYPE_NAMES

    query = RoceQuery(**dict(payload))
    setup = query.setup
    common = (
        query.world_size,
        query.rank,
        int(setup["threads"]),
        int(setup["slots"]),
        int(setup["flag_stride"]),
        int(setup["hca_count"]),
        ordinal,
    )
    with torch.cuda.device(ordinal):
        programs = {
            dtype: get_launcher(_DTYPE_NAMES[dtype], *common)
            for dtype in _dtypes(query)
        }
        programs["gather"] = _allgather_cute.get_launcher(*common)
        return programs


@dataclass(frozen=True)
class _PreparedRoce:
    runtime: object
    reduce_launchers: dict[torch.dtype, object]
    gather_launcher: object

    def all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor | None = None):
        return self.runtime._run_prepared_all_reduce(inp, prepared=self, out=out)

    def all_gather(self, inp: torch.Tensor, *, out: torch.Tensor | None = None):
        return self.runtime._run_prepared_all_gather(inp, prepared=self, dim=0, out=out)


def prepared_call(
    state: object, *, inp: torch.Tensor, out: torch.Tensor | None = None
) -> PreparedCall:
    """Prime an actual caller-owned collective before publication.

    This deliberately uses the materialized state directly instead of
    resolving it through the plan early.  The session installs the plan's
    prepared state only after this invocation has successfully loaded and
    launched the native program.
    """
    if not isinstance(state, _PreparedRoce):
        raise TypeError("RoCE preparation callback requires materialized runtime state")
    if out is None:
        out = torch.empty_like(inp)
    return PreparedCall(run=lambda: state.all_reduce(inp, out=out), output=out)


def prepared_gather_call(state: object, *, inp: torch.Tensor) -> PreparedCall:
    """Prime the actual RDMA all-gather launcher before publication."""
    if not isinstance(state, _PreparedRoce):
        raise TypeError("RoCE preparation callback requires materialized runtime state")
    return PreparedCall(run=lambda: state.all_gather(inp))


def plan(
    query: RoceQuery,
    *,
    runtime,
    invocation: FrozenMapping = FrozenMapping(),
    override=None,
) -> Plan:
    from .roce_oneshot import RoceOneshotAllReduce

    if not isinstance(query, RoceQuery) or not isinstance(
        runtime, RoceOneshotAllReduce
    ):
        raise TypeError("RoCE plan requires query and caller runtime")
    if (
        query.world_size != runtime.world_size
        or query.rank != runtime.rank
        or tuple(query.hca_names) != tuple(runtime.hca_names)
    ):
        raise ValueError("RoCE query differs from established runtime")
    if invocation:
        raise ValueError("RoCE invocation belongs in query")
    payload = TUNING.encode_query(query)

    def jobs(config, device):
        return (
            CompileJob.create(
                "b12x.comm.roce._preparation:compile_roce", payload, device.ordinal
            ),
        )

    def materialize(selection, device):
        programs = compile_roce(payload, device.ordinal)
        gather = programs["gather"]
        runtime._prepare_resources(_dtypes(query), padded_gather=True)
        return _PreparedRoce(
            runtime,
            {key: value for key, value in programs.items() if key != "gather"},
            gather,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(),
        override=override,
        _compile_jobs=jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
        _device=runtime.device,
    )


def query_from_runtime(
    runtime,
    *,
    surface,
    call,
    topology,
    peer_hosts,
    hca_names=None,
):
    """Build the RoCE query for an established runtime; ``hca_count`` is the runtime's rail count."""
    return RoceQuery(
        surface=surface,
        world_size=runtime.world_size,
        rank=runtime.rank,
        topology=topology,
        peer_hosts=tuple(peer_hosts),
        hca_names=tuple(runtime.hca_names if hca_names is None else hca_names),
        call=FrozenMapping(call),
        setup=FrozenMapping(
            {
                "threads": runtime._threads,
                "slots": runtime._layout.slots,
                "flag_stride": runtime._layout.flag_stride,
                "hca_count": runtime.rail_count,
            }
        ),
    )


__all__ = ["plan", "prepared_call", "prepared_gather_call", "query_from_runtime"]
