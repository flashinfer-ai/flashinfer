"""Measured anchor table: complete launch configurations per (model, topology).

``tuned_table.json`` holds, for each tuned signature (hidden, intermediate,
top-k, experts) and EP topology, the configuration that
``tools/sm120_nvfp4_tuning/tune_table.py`` selected at token anchors. A token
count between anchors uses the anchor nearest in log space (the boundary is the
geometric midpoint), so every token count is covered; counts outside the anchor
range use the closest end anchor.

Topologies (110-SM SM120, DP1/TP1, 8-SM partition granularity):

* ``ep4_1numa``: EP4 inside one NUMA node (p2p_direct).
* ``ep8_2numa``: EP8 as 2 x 4 GPUs on two NUMA nodes. An entry's ``backend``
  is ``p2p_split`` (LSA + cross-NUMA shadow transport) or ``gin_hybrid``;
  its knobs are valid only for that backend.

Knobs are the runner flags the tuner measured on top of the built-in
selection. The heuristic applies them as overrides (explicit caller overrides
win) with the geometry pinned to the configuration measured at the anchor.
Shapes, topologies or backends the table does not cover use the built-in
selector unchanged.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

_PATH = Path(__file__).with_name("tuned_table.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    if not _PATH.exists():
        return dict(schema=1, tables=[])
    data = json.loads(_PATH.read_text())
    if data.get("schema") != 1:
        raise ValueError(f"{_PATH.name}: unsupported schema {data.get('schema')}")
    return data


def topology_of(shape) -> Optional[str]:
    """Topology key of a heuristic input, or None if no table can apply."""
    if (shape.data_parallel_size, shape.tensor_parallel_size, shape.num_sms,
            shape.sm_min_partition, shape.sm_partition_alignment) != (1, 1, 110, 8, 8):
        return None
    ep, same, cross = (shape.expert_parallel_size, shape.ep_same_numa_peer_count,
                       shape.ep_cross_numa_peer_count)
    # p2p_split folds the cross-NUMA peers into one P2P group (the runner
    # passes 7 same-NUMA / 0 cross peers for EP8); that view is the split mode.
    split = getattr(shape, "ep_numa_split", None)
    if split is None:
        split = ep == 8 and (same, cross) == (7, 0)
    if ep == 4 and (same, cross) == (3, 0) and not split:
        return "ep4_1numa"
    if ep == 8 and (same, cross) == (3, 4) and not split:
        return "ep8_2numa"
    if ep == 8 and (same, cross) == (7, 0) and split:
        return "ep8_2numa"
    return None


def nearest_entry(signature: Tuple[int, int, int, int], topology: str, tokens: int) -> Optional[dict]:
    for table in _load().get("tables", []):
        if tuple(table["signature"]) == tuple(signature) and table["topology"] == topology:
            entries = table["entries"]
            if not entries:
                return None
            return min(entries, key=lambda e: (abs(math.log(tokens) - math.log(e["tokens"])), e["tokens"]))
    return None


def lookup(shape) -> Optional[dict]:
    """The anchor entry for this shape and its effective backend, or None."""
    topology = topology_of(shape)
    if topology is None or shape.tokens_per_rank <= 0:
        return None
    signature = (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
    entry = nearest_entry(signature, topology, shape.tokens_per_rank)
    if entry is None:
        return None
    if topology == "ep8_2numa":
        folded = (shape.ep_same_numa_peer_count, shape.ep_cross_numa_peer_count) == (7, 0)
        want = "p2p_split" if getattr(shape, "ep_numa_split", folded) else "gin_hybrid"
        if entry["knobs"].get("backend", "gin_hybrid") != want:
            return None  # knobs were tuned for the other backend
    return entry


# Rule for untabulated signatures, from the default-configuration measurements of
# both tuned NVFP4 signatures: p2p_split is 28-40% faster up to 128 tokens/rank
# and 1.3-2.9x faster at 2048-4096 (where gin_hybrid degrades); 512-1024 are
# within about 20% either way; gin_hybrid leads from 8192 on (one exception).
SPLIT_RULE_MAX_TOKENS = 4096
SPLIT_MAX_EXPERTS_PER_RANK = 64  # forwarder completion mask is 64-bit


def preferred_ep8_backend(shape) -> Optional[str]:
    """``p2p_split`` / ``gin_hybrid`` for a real 2 x 4 NUMA EP8 shape (cross-NUMA
    peers not yet folded): the nearest table anchor's backend, else the rule
    above; None if the topology is not ``ep8_2numa``."""
    if topology_of(shape) != "ep8_2numa" or (
            shape.ep_same_numa_peer_count, shape.ep_cross_numa_peer_count) != (3, 4):
        return None
    signature = (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
    entry = nearest_entry(signature, "ep8_2numa", shape.tokens_per_rank)
    if entry is not None:
        return entry["knobs"].get("backend", "gin_hybrid")
    experts_per_rank = shape.num_total_experts // shape.expert_parallel_size
    if (shape.tokens_per_rank <= SPLIT_RULE_MAX_TOKENS
            and experts_per_rank <= SPLIT_MAX_EXPERTS_PER_RANK):
        return "p2p_split"
    return "gin_hybrid"


def overrides_for(knobs: dict) -> dict:
    """Map table knobs to ``MegaMoEHeuristicOverrides`` field values."""
    out = {}
    if knobs.get("backend") == "gin_hybrid":
        out["comm_backend"] = "gin_hybrid"
    if knobs.get("k1_n") is not None:
        out["k1_tile"] = (64, int(knobs["k1_n"]), 128)
    if knobs.get("k2_n") is not None:
        out["k2_tile"] = (64, int(knobs["k2_n"]), 128)
    for key in ("k1_ctas_per_sm", "k2_launch_factor", "k1_stages", "k2_stages", "k2_warps",
                "dispatch_warps"):
        if knobs.get(key) is not None:
            out[key] = int(knobs[key])
    if knobs.get("k2_ready_queue_bundle") is not None:
        out["ready_queue_bundle"] = int(knobs["k2_ready_queue_bundle"])
    for key in ("k2_tail_reclaim", "dispatch_rank_cache", "dispatch_compute_overlap"):
        if knobs.get(key) is not None:
            out[key] = knobs[key] == "on"
    sms = knobs.get("sms")
    if sms is not None:
        if len(sms) == 2:
            sms = (sms[0], 0, 0, sms[1])
        out.update(zip(("k1_sms", "tx_sms", "rx_sms", "k2_sms"), (int(v) for v in sms)))
    return out


def utility_for(shape) -> Optional[bool]:
    """FC1 utility postprocess measured with the anchor's configuration, or None."""
    entry = lookup(shape)
    if entry is None or entry["knobs"].get("fc1_utility_postprocess") is None:
        return None
    return entry["knobs"]["fc1_utility_postprocess"] == "on"


__all__ = ["lookup", "nearest_entry", "overrides_for", "preferred_ep8_backend", "topology_of", "utility_for"]
