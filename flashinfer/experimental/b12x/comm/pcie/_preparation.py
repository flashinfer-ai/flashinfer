"""One declaration entry point for established native PCIe resource owners."""
from __future__ import annotations

from importlib import import_module

from b12x.preparation import FrozenMapping, Plan
from ._tuning import PcieConfig, PcieQuery, TUNING


_OWNER_SURFACES = frozenset((
    "DcpTopKOwnerExchange.stage_candidates",
    "VocabParallelArgmax.fused_add_argmax",
    "PCIeHierarchicalAllReduce.all_reduce",
    "PCIeIslandRSAllReduce.all_reduce",
))


def _provider(surface: str):
    if surface.startswith(("OneshotAllReduce.", "OneshotAllReducePool.")):
        module = "_oneshot_preparation"
    elif surface == "DmaAllReduce.all_reduce":
        module = "_dma_preparation"
    elif surface.startswith(("TwoShotReduceScatter.", "PCIeTwoShotBF16.")):
        module = "_twoshot_preparation"
    elif surface == "kimi_topk16" or surface.startswith(("DcpAllToAll.", "DcpAllToAllPool.")):
        module = "_dcp_preparation"
    elif surface in _OWNER_SURFACES:
        module = "_owner_preparation"
    else:
        raise ValueError(f"unknown native PCIe execution surface {surface!r}")
    return import_module(f"{__package__}.{module}")


def query_from_runtime(runtime, *, surface: str, call) -> PcieQuery:
    """Normalize call metadata for the caller's already-established channel."""
    return _provider(surface).query_from_runtime(runtime, surface=surface, call=call)


def plan(query: PcieQuery, *, runtime=None, invocation=FrozenMapping(),
         override: PcieConfig | None = None) -> Plan:
    """Declare native work; neither topology nor a communicator is constructed.

    The all-reduce manager's ``plan`` method first selects its existing native
    channel using the original host routing rule, then calls this entry point.
    """
    if not isinstance(query, PcieQuery):
        raise TypeError("query must be PcieQuery")
    TUNING.validate_query(query, None)
    return _provider(query.surface).plan(
        query, runtime=runtime, invocation=invocation, override=override,
    )


__all__ = ["plan", "query_from_runtime"]
