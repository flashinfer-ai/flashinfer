"""Declarative preparation for native PCIe DCP A2A channels.

The declaration captures launch metadata only.  IPC slabs and process groups remain
owned by the already-created channel; selected launchers are retained by the
materialized state and no runtime path consults a compiler cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    PreparedCall,
)

from ._tuning import PcieConfig, PcieQuery, TUNING

_DCP_SURFACES = frozenset({
    "DcpAllToAll.lse_reduce_scatter",
    "DcpAllToAll.all_gather_heads",
    "DcpAllToAll.all_gather_pair",
    "DcpAllToAll.all_gather_pair_kimi_topk",
    "DcpAllToAll.kimi_topk16",
    "DcpAllToAllPool.lse_reduce_scatter",
    "DcpAllToAllPool.all_gather_heads",
    "DcpAllToAllPool.all_gather_pair",
    "DcpAllToAllPool.all_gather_pair_kimi_topk",
    "DcpAllToAllPool.kimi_topk16",
    "kimi_topk16",
})


def _dtype_name(dtype: torch.dtype) -> str:
    return {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]


def _runtime(runtime):
    if runtime is None or not hasattr(runtime, "world_size"):
        raise TypeError("DCP preparation requires a live native DCP channel")
    return runtime


def _call_value(call: Mapping[str, object], name: str):
    try:
        return call[name]
    except KeyError:
        raise ValueError(f"DCP preparation call metadata misses {name!r}") from None


def query_from_runtime(runtime, *, surface, call) -> PcieQuery:
    """Extract immutable exact-shape/native-launch metadata from a DCP owner."""
    if surface not in _DCP_SURFACES:
        raise ValueError(f"unsupported DCP preparation surface {surface!r}")
    call = dict(call)
    if surface == "kimi_topk16":
        logits = _call_value(call, "router_logits")
        return PcieQuery(surface=surface, world_size=1, rank=0, topology="local",
                         call=FrozenMapping({"threads": int(call.get("threads", 256))}),
                         setup=FrozenMapping({"device_index": logits.device.index, "max_rows": int(logits.shape[0])}))
    runtime = _runtime(runtime)
    setup = FrozenMapping({
        "device_index": runtime.device.index,
        "max_batch_size": int(runtime.max_batch_size),
        "total_heads": int(runtime.total_heads),
        "head_dim": int(runtime.head_dim),
        "query_head_dim": int(runtime.query_head_dim),
        "slot_bytes": int(runtime._slot_bytes),
        "signal_words": len(runtime._signal_ptrs),
    })
    requested_threads = int(call.get("threads", 256))
    requested_blocks = int(call.get("block_limit", 16))
    effective_threads, effective_blocks = runtime._resolve_launch_config(
        threads=requested_threads, block_limit=requested_blocks)
    metadata = {"threads": effective_threads, "block_limit": effective_blocks}
    if surface.endswith("lse_reduce_scatter"):
        value = _call_value(call, "partial_output")
        lse = _call_value(call, "partial_lse")
        metadata.update(dtype=_dtype_name(value.dtype), shape=tuple(value.shape), stride=tuple(value.stride()), lse_shape=tuple(lse.shape), natural_log=bool(call.get("is_lse_base_on_e", True)))
    elif surface.endswith("all_gather_heads"):
        value = _call_value(call, "local_input")
        metadata.update(dtype=str(value.dtype), shape=tuple(value.shape), stride=tuple(value.stride()))
    elif surface.endswith("all_gather_pair"):
        first, second = _call_value(call, "local_first"), _call_value(call, "local_second")
        metadata.update(first_dtype=str(first.dtype), second_dtype=str(second.dtype), first_shape=tuple(first.shape), second_shape=tuple(second.shape))
    elif surface.endswith("all_gather_pair_kimi_topk"):
        metadata["threads"] = 512
    elif surface.endswith("kimi_topk16"):
        metadata["threads"] = int(call.get("threads", 256))
    return PcieQuery(surface=surface, world_size=int(runtime.world_size), rank=int(runtime.rank), topology="pcie_ipc", call=FrozenMapping(metadata), setup=setup)


@program_cache(scope="preparation")
def compile_dcp_surface(query_payload, ordinal):
    """Compile every eager/graph native variant represented by a declaration."""
    from . import _dcp_a2a_cute as cute
    query = PcieQuery(**dict(query_payload))
    call = query.call
    surface = query.surface
    with torch.cuda.device(ordinal):
        if surface == "kimi_topk16" or surface.endswith("kimi_topk16"):
            return {False: cute._get_compiled_kimi_topk16(int(call["threads"]))}
        if surface.endswith("lse_reduce_scatter"):
            return {slot: cute._get_compiled_lse_reduce_scatter(query.world_size, query.rank, call["dtype"], int(call["threads"]), slot) for slot in (False, True)}
        if surface.endswith("all_gather_heads"):
            return {slot: cute._get_compiled_all_gather_heads(query.world_size, query.rank, int(call["threads"]), slot) for slot in (False, True)}
        if surface.endswith("all_gather_pair_kimi_topk"):
            return {slot: cute._get_compiled_all_gather_pair(query.world_size, query.rank, 512, slot, True) for slot in (False, True)}
        if surface.endswith("all_gather_pair"):
            return {slot: cute._get_compiled_all_gather_pair(query.world_size, query.rank, int(call["threads"]), slot, False) for slot in (False, True)}
    raise ValueError(f"unsupported DCP preparation surface {surface!r}")


@dataclass(frozen=True)
class _DcpExecutionState:
    query: PcieQuery
    runtime: object
    launchers: Mapping[bool, object]

    def require_runtime(self, runtime):
        if runtime is not self.runtime:
            raise ValueError("DCP plan belongs to a different channel owner")
        if (runtime.rank, runtime.world_size) != (self.query.rank, self.query.world_size):
            raise ValueError("DCP runtime identity differs from preparation")

    def launcher(self, device_slot_selection: bool):
        try:
            return self.launchers[bool(device_slot_selection)]
        except KeyError:
            raise RuntimeError("DCP plan lacks its required eager/graph launcher") from None

    def run(self, **call):
        """Run a bound call without reentering preparation or compiler caches."""
        surface = self.query.surface
        if surface.endswith("lse_reduce_scatter"):
            return self.runtime._lse_reduce_scatter_on_device(call["partial_output"], call["partial_lse"], call.get("out"), state=self, is_lse_base_on_e=call.get("is_lse_base_on_e", True), threads=call.get("threads", 256), block_limit=call.get("block_limit", 16))
        if surface.endswith("all_gather_heads"):
            return self.runtime._all_gather_heads_on_device(call["local_input"], call.get("out"), state=self, threads=call.get("threads", 256), block_limit=call.get("block_limit", 16))
        if surface.endswith("all_gather_pair_kimi_topk"):
            return self.runtime._all_gather_pair_kimi_topk_on_device(call["local_down"], call["local_router"], call["correction_bias"], call.get("out_down"), call.get("topk_weights"), call.get("topk_ids"), state=self)
        if surface.endswith("all_gather_pair"):
            return self.runtime._all_gather_pair_on_device(call["local_first"], call["local_second"], call.get("out_first"), call.get("out_second"), state=self, threads=call.get("threads", 512))
        if surface == "kimi_topk16":
            from ._dcp_a2a_cute import kimi_topk16
            logits, bias = call["router_logits"], call["correction_bias"]
            weights = call.get("output_weights")
            ids = call.get("output_ids")
            if weights is None or ids is None:
                raise ValueError("Kimi preparation requires caller-owned outputs")
            kimi_topk16(router_logits_ptr=logits.data_ptr(), correction_bias_ptr=bias.data_ptr(),
                        output_weights_ptr=weights.data_ptr(), output_ids_ptr=ids.data_ptr(),
                        rows=int(logits.shape[0]), threads=call.get("threads", 256),
                        launcher=self.launcher(False))
            return weights, ids
        if surface.endswith("kimi_topk16"):
            return self.runtime._kimi_topk16_on_device(call["router_logits"], call["correction_bias"], call.get("output_weights"), call.get("output_ids"), state=self, threads=call.get("threads", 256))
        raise ValueError(f"unsupported DCP execution surface {surface!r}")


def plan(query: PcieQuery, *, runtime=None, invocation=FrozenMapping(), override: PcieConfig | None = None) -> Plan:
    """Declare one fixed DCP collective or local Kimi top-k execution."""
    if not isinstance(query, PcieQuery):
        raise TypeError("query must be PcieQuery")
    if query.surface not in _DCP_SURFACES:
        raise ValueError(f"unsupported DCP preparation surface {query.surface!r}")
    if query.surface == "kimi_topk16":
        if runtime is not None:
            runtime = _runtime(runtime)
    else:
        runtime = _runtime(runtime)
        if (runtime.rank, runtime.world_size) != (query.rank, query.world_size):
            raise ValueError("DCP query does not match its runtime owner")
    if invocation:
        raise ValueError("DCP invocation semantics belong in PcieQuery")

    def materialize(selection, device):
        launchers = compile_dcp_surface(TUNING.encode_query(query), device.ordinal)
        return _DcpExecutionState(query, runtime, MappingProxyType(dict(launchers)))

    resident = int(getattr(runtime, "_slot_bytes", 0)) * 2
    return Plan(
        contract=TUNING, query=query, invocation=FrozenMapping(invocation), override=override,
        _compile_jobs=lambda config, device: (CompileJob.create("b12x.comm.pcie._dcp_preparation:compile_dcp_surface", TUNING.encode_query(query), device.ordinal),),
        _memory_requirements=lambda config, device: MemoryRequirements(persistent=(() if query.surface == "kimi_topk16" else (PersistentMemory(("pcie-dcp-runtime", id(runtime)), resident, resident),))),
        _materialize=materialize, _device=getattr(runtime, "device", None),
    )


def prepare_call(state: _DcpExecutionState, **call):
    """Bind caller-owned DCP tensors and prime the retained native launcher."""
    if not isinstance(state, _DcpExecutionState):
        raise TypeError("DCP callback requires its materialized state")
    surface = state.query.surface
    if surface.endswith("all_gather_pair_kimi_topk"):
        output = (call.get("out_down"), call.get("topk_weights"), call.get("topk_ids"))
    elif surface.endswith("all_gather_pair"):
        output = (call.get("out_first"), call.get("out_second"))
    elif surface.endswith("kimi_topk16"):
        output = (call.get("output_weights"), call.get("output_ids"))
    else:
        output = call.get("out")
    return PreparedCall(run=lambda: state.run(**call), output=output)


__all__ = ["query_from_runtime", "plan", "prepare_call"]
