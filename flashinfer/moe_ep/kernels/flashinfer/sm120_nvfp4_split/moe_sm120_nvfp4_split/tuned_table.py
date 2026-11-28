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
* ``ep16_2x2numa``: EP16 over two hosts of 2 x 4 GPUs (P2P only inside a
  NUMA node, every other peer through GIN). Not tabulated yet; the backend
  rule below applies.

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


# (EP size, same-domain peers, network peers) of the real NUMA-split layouts.
_SPLIT_LAYOUTS = {8: ("ep8_2numa", 3, 4), 16: ("ep16_2x2numa", 3, 12)}


# The tables were measured on 110-SM RTX PRO 5000 GPUs (8-SM Green Context
# partition granularity); other devices use the rules only.
TABLE_DEVICE = (110, 8, 8)


def device_of(shape) -> Tuple[int, int, int]:
    return (shape.num_sms, shape.sm_min_partition, shape.sm_partition_alignment)


def topology_of(shape) -> Optional[str]:
    """Topology key of a heuristic input, or None if no table can apply."""
    if device_of(shape) != TABLE_DEVICE:
        return None
    return layout_of(shape)


def layout_of(shape) -> Optional[str]:
    """Topology key of a heuristic input on any device (DP1/TP1), or None."""
    if (shape.data_parallel_size, shape.tensor_parallel_size) != (1, 1):
        return None
    ep, same, cross = (shape.expert_parallel_size, shape.ep_same_numa_peer_count,
                       shape.ep_cross_numa_peer_count)
    if ep == 4 and (same, cross) == (3, 0) and not getattr(shape, "ep_numa_split", False):
        return "ep4_1numa"
    if ep in _SPLIT_LAYOUTS:
        name, real_same, real_cross = _SPLIT_LAYOUTS[ep]
        # p2p_split folds the network peers into one P2P group (the runner
        # passes ep-1 same-domain / 0 network peers); that view is the split mode.
        split = getattr(shape, "ep_numa_split", None)
        if split is None:
            split = (same, cross) == (ep - 1, 0)
        if (same, cross) == ((ep - 1, 0) if split else (real_same, real_cross)):
            return name
    return None


def _folded(shape) -> bool:
    ep = shape.expert_parallel_size
    folded = (shape.ep_same_numa_peer_count, shape.ep_cross_numa_peer_count) == (ep - 1, 0)
    return getattr(shape, "ep_numa_split", folded)


def nearest_entry(signature: Tuple[int, int, int, int], topology: str, tokens: int) -> Optional[dict]:
    for table in _load().get("tables", []):
        if tuple(table["signature"]) == tuple(signature) and table["topology"] == topology:
            entries = table["entries"]
            if not entries:
                return None
            return min(entries, key=lambda e: (abs(math.log(tokens) - math.log(e["tokens"])), e["tokens"]))
    return None


def has_table(signature: Tuple[int, int, int, int], topology: str) -> bool:
    """Whether (signature, topology) has a table with anchors."""
    return nearest_entry(signature, topology, 1) is not None


def lookup(shape) -> Optional[dict]:
    """The anchor entry for this shape and its effective backend, or None."""
    topology = topology_of(shape)
    if topology is None or shape.tokens_per_rank <= 0:
        return None
    signature = (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
    entry = nearest_entry(signature, topology, shape.tokens_per_rank)
    if entry is None:
        return None
    if topology != "ep4_1numa":
        want = "p2p_split" if _folded(shape) else "gin_hybrid"
        if entry["knobs"].get("backend", "gin_hybrid") != want:
            return None  # knobs were tuned for the other backend
    return entry


SPLIT_MAX_EXPERTS_PER_RANK = 64  # forwarder completion mask is 64-bit

# Backend rule for a cross-NUMA EP world without an explicit backend and
# without a table anchor (any device): p2p_split (LSA + cross-NUMA shadow
# transport) up to SPLIT_RULE_MAX_TOKENS tokens per rank, gin_hybrid above
# when the driver can run it (``runtime.driver_caps``), else p2p_split. The
# hybrid rings are validated on one host only (EP <= HYBRID_MAX_EP); EP16 is
# always p2p_split. Measured on 110-SM EP8 (default knobs, EP8 tables
# bypassed, cold L2, ABBA p50, gin_hybrid vs p2p_split): DSv4.1 T512/1024/
# 2048/4096/8192/16384/24576 +22/-9/+206/+77/-24/-25/-27 %, GLM5.3
# -4/-14/+121/+187/-6/-20/-27 %, DSv4-Flash +4/-11/+3/+167/+44/-7/-23 %:
# the hybrid collapses (and jitters) at 2048-4096 and wins near 1024 and above
# 8192 for all three; at 8192 it wins DSv4.1/GLM5.3 (-24/-6 %) but loses
# DSv4-Flash (+44 %), so the rule switches above 8192. RESULTS.md has the table.
SPLIT_RULE_MAX_TOKENS = 8192
HYBRID_MAX_EP = 8


def hybrid_available(shape) -> bool:
    """Whether the GIN-ring hybrid can run for this world on this driver."""
    if shape.expert_parallel_size > HYBRID_MAX_EP:
        return False
    from .runtime.driver_caps import hybrid_sm_split_available

    return hybrid_sm_split_available()


def rule_backend(shape) -> Optional[str]:
    """The untabulated cross-NUMA rule: ``p2p_split`` up to
    ``SPLIT_RULE_MAX_TOKENS`` tokens per rank (every size at EP16),
    ``gin_hybrid`` above when the hybrid can run, else ``p2p_split``; None
    when neither can serve the world (more than 64 experts per rank and no
    hybrid)."""
    split_ok = shape.num_total_experts // shape.expert_parallel_size <= SPLIT_MAX_EXPERTS_PER_RANK
    if split_ok and (shape.tokens_per_rank <= SPLIT_RULE_MAX_TOKENS
                     or shape.expert_parallel_size > HYBRID_MAX_EP):
        return "p2p_split"
    if hybrid_available(shape):
        return "gin_hybrid"
    return "p2p_split" if split_ok else None


def preferred_ep8_backend(shape) -> Optional[str]:
    """``p2p_split`` / ``gin_hybrid`` for a real NUMA-split EP shape (EP8 on
    2 x 4 GPUs or EP16 on two such hosts, network peers not yet folded) on any
    device: the nearest 110-SM table anchor's
    backend (a ``gin_hybrid`` anchor reads ``p2p_split`` where the hybrid
    cannot run), else ``rule_backend``. None when the shape is not a
    cross-NUMA world or no backend can serve it."""
    if shape.ep_cross_numa_peer_count <= 0 or _folded(shape):
        return None
    if layout_of(shape) not in ("ep8_2numa", "ep16_2x2numa"):
        return None  # only the validated NUMA-split layouts
    topology = topology_of(shape)
    if topology is not None:
        signature = (shape.hidden, shape.intermediate, shape.num_topk, shape.num_total_experts)
        entry = nearest_entry(signature, topology, shape.tokens_per_rank)
        if entry is not None:
            backend = entry["knobs"].get("backend", "gin_hybrid")
            if backend == "gin_hybrid" and not hybrid_available(shape):
                split_ok = (shape.num_total_experts // shape.expert_parallel_size
                            <= SPLIT_MAX_EXPERTS_PER_RANK)
                return "p2p_split" if split_ok else None
            return backend
    return rule_backend(shape)


def overrides_for(knobs: dict) -> dict:
    """Map table knobs to ``MegaMoEHeuristicOverrides`` field values."""
    out = {}
    if knobs.get("backend") == "gin_hybrid":
        out["comm_backend"] = "gin_hybrid"
    if knobs.get("k1_n") is not None:
        out["k1_tile"] = (64, int(knobs["k1_n"]), 128)
    if knobs.get("k2_n") is not None:
        out["k2_tile"] = (64, int(knobs["k2_n"]), 128)
    for key in ("k1_ctas_per_sm", "k2_launch_factor", "weight_prefetch_ahead", "xnuma_fsplit", "k1_stages", "k2_stages", "k2_warps",
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


def utility_for(shape, generic_decode_preset: bool = True, config=None) -> Optional[bool]:
    """FC1 utility postprocess measured with the anchor's configuration (or
    set by the generic decode preset, where the resolved ``config`` has an
    N16/N32 K1 tile), or None."""
    entry = lookup(shape)
    knobs = entry["knobs"] if entry is not None else None
    if knobs is None and generic_decode_preset:
        from .heuristic import generic_decode_knobs

        knobs = generic_decode_knobs(shape)
        if knobs is not None and (config is None or config.k1_tile[1] not in (16, 32)
                                  or config.k1_compact_tokens):
            return None
    if knobs is None or knobs.get("fc1_utility_postprocess") is None:
        return None
    return knobs["fc1_utility_postprocess"] == "on"


__all__ = ["SPLIT_RULE_MAX_TOKENS", "TABLE_DEVICE", "device_of", "hybrid_available", "layout_of", "lookup",
           "nearest_entry", "overrides_for", "preferred_ep8_backend", "rule_backend", "topology_of", "utility_for"]
