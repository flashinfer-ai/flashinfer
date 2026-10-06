"""Prepared ownership for fixed PCIe collective runtimes.

The declarations deliberately retain the caller-created IPC runtime and process
 group.  They only compile/load its already-selected native specialization;
there is no topology discovery or transport selection here.
"""
from __future__ import annotations

from dataclasses import dataclass

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    CollectiveRequirement,
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    PreparedCall,
    current_plan,
)

from ._tuning import PcieConfig, PcieQuery, TUNING

_COMPONENT = "comm.pcie"


def _runtime_device(runtime):
    device = getattr(runtime, "device", None)
    if device is None:
        raise ValueError("PCIe runtime must expose its CUDA device")
    return device


def _runtime_slab_bytes(runtime) -> int:
    if hasattr(runtime, "slab_bytes"):
        return int(runtime.slab_bytes)
    layout = getattr(runtime, "_layout", None)
    if layout is not None and hasattr(layout, "bytes"):
        return int(layout.bytes)
    if runtime.__class__.__name__ == "PCIeVocabParallelArgmax":
        from ._vocab_argmax_cute import SLAB_BYTES
        return int(SLAB_BYTES)
    if runtime.__class__.__name__ == "PCIeDCPTopKOwnerExchange":
        from .pcie_dcp_topk import _SIGNAL_BYTES, _candidate_staging_layout
        layout = _candidate_staging_layout(
            signal_bytes=_SIGNAL_BYTES,
            max_rows=runtime.max_rows,
            topk=runtime.topk,
            world_size=runtime.world_size,
        )
        return int(layout.slab_bytes)
    raise ValueError("PCIe runtime does not expose its IPC slab capacity")


def _surface_for_runtime(runtime) -> str:
    name = runtime.__class__.__name__
    return {
        "PCIeDCPTopKOwnerExchange": "DcpTopKOwnerExchange.stage_candidates",
        "PCIeVocabParallelArgmax": "VocabParallelArgmax.fused_add_argmax",
        "PCIeHierarchicalAllReduce": "PCIeHierarchicalAllReduce.all_reduce",
        "PCIeIslandRSAllReduce": "PCIeIslandRSAllReduce.all_reduce",
    }.get(name, "")



def _alignment(tensor):
    pointer = tensor.data_ptr()
    return min(256, pointer & -pointer) if pointer else 256


def _tensor_call(tensor):
    return FrozenMapping(
        {
            "dtype": str(tensor.dtype),
            "shape": tuple(int(size) for size in tensor.shape),
            "stride": tuple(int(stride) for stride in tensor.stride()),
            "alignment": _alignment(tensor),
        }
    )
def query_from_runtime(runtime, *, surface: str | None = None, call=FrozenMapping()) -> PcieQuery:
    """Snapshot an existing runtime's immutable native specialization."""
    surface = surface or _surface_for_runtime(runtime)
    if not surface:
        raise TypeError("unsupported PCIe runtime owner")
    raw = dict(call)
    controls = {}
    if surface == "DcpTopKOwnerExchange.stage_candidates":
        indices, scores = raw["local_indices"], raw["local_scores"]
        controls = {
            "topk": int(runtime.topk),
            "threads": int(raw.get("threads", 512)),
            "block_limit": int(raw.get("block_limit", 128)),
            "indices": _tensor_call(indices),
            "scores": _tensor_call(scores),
            "rows": int(indices.shape[0]),
        }
        setup = {"max_rows": int(runtime.max_rows), "max_owner_rows": int(runtime.max_owner_rows), "candidate_plane_elems": int(runtime._candidate_plane_elems), "slab_bytes": _runtime_slab_bytes(runtime), "slots": 2}
    elif surface == "VocabParallelArgmax.fused_add_argmax":
        base, bias = raw["base"], raw["bias"]
        controls = {
            "wait_nanosleep_cycles": int(runtime.wait_nanosleep_cycles),
            "base": _tensor_call(base),
            "bias": _tensor_call(bias),
            "batch": int(base.shape[0]),
            "output": None if raw.get("out") is None else _tensor_call(raw["out"]),
        }
        setup = {"local_vocab_size": int(runtime.local_vocab_size), "max_batch_size": int(runtime.max_batch_size), "slab_bytes": _runtime_slab_bytes(runtime), "peer_ranks": tuple(_selected_peers(runtime))}
    elif surface == "PCIeHierarchicalAllReduce.all_reduce":
        inp, out = raw["inp"], raw.get("out")
        from .pcie_hierarchical import _pick_blocks
        blocks = int(raw["blocks"]) if raw.get("blocks") is not None else (runtime.blocks if runtime.blocks is not None else _pick_blocks(inp.numel()))
        vectorized = bool(runtime.vectorized_bf16x2 and out is not None and inp.numel() <= runtime.vectorized_bf16x2_max_elements and _alignment(inp) >= 4 and _alignment(out) >= 4)
        controls = {"threads": int(runtime.threads), "wait_nanosleep_cycles": int(runtime.wait_nanosleep_cycles), "double_buffered": bool(runtime.double_buffered), "deferred_consumption": bool(runtime.deferred_consumption), "vectorized": vectorized, "blocks": blocks, "inp": _tensor_call(inp), "output": None if out is None else _tensor_call(out)}
        setup = {"max_elements": int(runtime.max_elements), "slab_bytes": _runtime_slab_bytes(runtime), "peer_ranks": tuple(_selected_peers(runtime))}
    elif surface == "PCIeIslandRSAllReduce.all_reduce":
        inp, out = raw["inp"], raw.get("out")
        from .pcie_island_rs import _pick_blocks
        blocks = int(raw["blocks"]) if raw.get("blocks") is not None else (runtime.blocks if runtime.blocks is not None else _pick_blocks(inp.numel()))
        controls = {"threads": int(runtime.threads), "wait_nanosleep_cycles": int(runtime.wait_nanosleep_cycles), "blocks": blocks, "inp": _tensor_call(inp), "output": None if out is None else _tensor_call(out)}
        setup = {"max_elements": int(runtime.max_elements), "quarter_capacity": int(runtime.quarter_capacity), "stage_offset": int(runtime.stage_offset), "part_offset": int(runtime.part_offset), "final_offset": int(runtime.final_offset), "slab_bytes": _runtime_slab_bytes(runtime), "peer_ranks": tuple(runtime.mapped_peers)}
    else:
        raise ValueError(f"unsupported PCIe owner surface {surface!r}")
    return PcieQuery(surface=surface, world_size=int(runtime.world_size), rank=int(runtime.rank), topology="pcie_ipc", call=FrozenMapping(controls), setup=FrozenMapping(setup))


def _selected_peers(runtime):
    if hasattr(runtime, "mapped_peers"):
        return runtime.mapped_peers
    if runtime.__class__.__name__ == "PCIeHierarchicalAllReduce":
        from .pcie_hierarchical import _selected_peers
    else:
        from .pcie_vocab_argmax import _selected_peers
    return _selected_peers(runtime.rank, runtime.world_size)


@program_cache(scope="preparation")
def compile_owner_surface(query_payload, ordinal):
    """Resolve the exact existing CuTe getter from metadata-only query data."""
    import torch

    query = PcieQuery(**dict(query_payload))
    call = query.call
    with torch.cuda.device(ordinal):
        if query.surface == "DcpTopKOwnerExchange.stage_candidates":
            from ._dcp_topk_cute import _get_compiled_topk_stage
            return _get_compiled_topk_stage(query.world_size, query.rank, call["topk"], call["threads"])
        if query.surface == "VocabParallelArgmax.fused_add_argmax":
            from ._vocab_argmax_cute import get_vocab_argmax_launcher
            return get_vocab_argmax_launcher(query.world_size, query.rank, ordinal, wait_nanosleep_cycles=call["wait_nanosleep_cycles"])
        if query.surface == "PCIeHierarchicalAllReduce.all_reduce":
            from ._hierarchical_cute import get_hierarchical_launcher
            vectorized = bool(call["vectorized"])
            launcher = get_hierarchical_launcher(query.world_size, query.rank, ordinal, threads=(112 if vectorized else call["threads"]), wait_nanosleep_cycles=call["wait_nanosleep_cycles"], double_buffered=call["double_buffered"], deferred_consumption=call["deferred_consumption"], vectorized_bf16x2=vectorized)
            return (None, launcher) if vectorized else (launcher, None)
        if query.surface == "PCIeIslandRSAllReduce.all_reduce":
            from ._island_rs_cute import get_island_rs_launcher
            return get_island_rs_launcher(query.world_size, query.rank, ordinal, threads=call["threads"], wait_nanosleep_cycles=call["wait_nanosleep_cycles"])
    raise ValueError(f"unsupported PCIe owner surface {query.surface!r}")


@dataclass(frozen=True)
class _OwnerState:
    query: PcieQuery
    runtime: object
    launchers: object

    def require_runtime(self, runtime):
        if runtime is not self.runtime:
            raise ValueError("PCIe plan belongs to a different native runtime owner")
        if (runtime.rank, runtime.world_size, _runtime_device(runtime)) != (self.query.rank, self.query.world_size, _runtime_device(self.runtime)):
            raise ValueError("PCIe runtime identity differs from preparation")

    def launcher(self, *, vectorized=False):
        if self.query.surface == "PCIeHierarchicalAllReduce.all_reduce":
            launcher = self.launchers[1] if vectorized else self.launchers[0]
            if launcher is None:
                raise ValueError("prepared hierarchy does not include the BF16x2 variant")
            return launcher
        return self.launchers


def plan(query: PcieQuery, *, runtime, invocation=FrozenMapping(), override: PcieConfig | None = None) -> Plan:
    """Declare one fixed existing PCIe runtime specialization."""
    if not isinstance(query, PcieQuery):
        raise TypeError("query must be PcieQuery")
    if invocation:
        raise ValueError("PCIe invocation semantics belong in PcieQuery")
    if _surface_for_runtime(runtime) != query.surface:
        raise ValueError("query surface does not match the existing PCIe runtime owner")
    if (int(runtime.rank), int(runtime.world_size)) != (query.rank, query.world_size):
        raise ValueError("query does not match the existing PCIe runtime owner")
    def memory(_config, _device):
        slab = int(query.setup["slab_bytes"])
        # The runtime allocated this slab before declaration. Account its real
        # residency without claiming an allocation owned by the session.
        return MemoryRequirements(
            persistent=(
                PersistentMemory(
                    ("comm.pcie", query.surface, current_plan(), "slab"),
                    slab,
                    slab,
                ),
            )
        )

    def materialize(selection, device):
        return _OwnerState(query, runtime, compile_owner_surface(TUNING.encode_query(query), device.ordinal))

    return Plan(contract=TUNING, query=query, invocation=FrozenMapping(), override=override, _device=_runtime_device(runtime), _compile_jobs=lambda _config, device: (CompileJob.create("b12x.comm.pcie._owner_preparation:compile_owner_surface", TUNING.encode_query(query), device.ordinal),), _memory_requirements=memory, _materialize=materialize)


def collective(runtime, *, key: str) -> CollectiveRequirement:
    import torch.distributed as dist

    group = getattr(runtime, "group", None) or getattr(runtime, "exchange_group", None)
    if group is None:
        raise ValueError("PCIe runtime has no established process group")
    return CollectiveRequirement(
        key=key,
        ranks=tuple(int(rank) for rank in dist.get_process_group_ranks(group)),
    )


def prepared_call(state, **actual):
    """Build a priming call that invokes the materialized native owner state."""
    runtime, query = state.runtime, state.query
    if query.surface == "DcpTopKOwnerExchange.stage_candidates":
        def run():
            return runtime._stage_candidates_on_device(
                actual["local_indices"], actual["local_scores"],
                launcher=state.launcher(),
                threads=query.call["threads"],
                block_limit=query.call["block_limit"],
            )
        return PreparedCall(run=run, capture_safe=True)
    if query.surface == "VocabParallelArgmax.fused_add_argmax":
        out = actual.get("out")
        if out is None:
            import torch
            out = torch.empty(actual["base"].shape[0], dtype=torch.int64, device=runtime.device)
        return PreparedCall(
            run=lambda: runtime._run_prepared(state.launcher(), actual["base"], actual["bias"], out),
            output=out,
            capture_safe=True,
        )
    if query.surface == "PCIeHierarchicalAllReduce.all_reduce":
        out = actual.get("out")
        if out is None:
            import torch
            out = torch.empty_like(actual["inp"])
        return PreparedCall(
            run=lambda: runtime._run_prepared(
                state.launcher(vectorized=query.call["vectorized"]),
                actual["inp"], out, query.call["blocks"],
            ),
            output=out,
            capture_safe=True,
        )
    if query.surface == "PCIeIslandRSAllReduce.all_reduce":
        out = actual.get("out")
        if out is None:
            import torch
            out = torch.empty_like(actual["inp"])
        return PreparedCall(
            run=lambda: runtime._run_prepared(
                state.launcher(), actual["inp"], out, query.call["blocks"],
                actual.get("stream"),
            ),
            output=out,
            capture_safe=True,
        )
    raise ValueError(f"unsupported PCIe owner surface {query.surface!r}")

__all__ = ["collective", "plan", "prepared_call", "query_from_runtime"]
