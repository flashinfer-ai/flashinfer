"""Shadow-segment layout for the p2p_split transport (EP spanning NUMA nodes).

Each rank's symmetric P2P arena is split into ``1 + n_cross`` equal segments:

* segment 0 holds the rank's own buffers (inputs, combine output, shared
  workspace) plus a reserved control region at ``reserved_offset``;
* segment k >= 1 is a *shadow* of the k-th cross-NUMA peer: it has the same
  layout as that peer's segment 0, so the unmodified p2p_direct code that
  "reads a peer's inputs" or "writes a peer's combine rows" through peer
  offsets touches the local shadow instead. GIN moves data between a shadow
  and the real peer.

Segment numbering is derived from the topology so that every rank can compute
it for any (owner, source) pair: the owner's cross-NUMA peers in ascending rank
order map to segments 1..n_cross. On a 2x4 NUMA EP8 this equals the
``peer % 4 + 1`` rule used by the original experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class ShadowLayout:
    numa_nodes: Tuple[int, ...]  # P2P domain (host, NUMA node) index of every EP rank, in EP rank order
    rank: int

    @classmethod
    def from_topology(cls, topology, rank: int) -> "ShadowLayout":
        # One integer per P2P domain (host, NUMA node): GPUs on another host
        # are cross peers exactly like GPUs on the other NUMA node.
        domains = {}
        for peer in topology.peers:
            domains.setdefault(peer.p2p_domain, len(domains))
        return cls(tuple(domains[peer.p2p_domain] for peer in topology.peers), int(rank))

    def __post_init__(self):
        world = len(self.numa_nodes)
        if not 0 <= self.rank < world:
            raise ValueError(f"rank {self.rank} outside EP world {world}")
        counts = {len(self.cross_peers(r)) for r in range(world)}
        if len(counts) != 1:
            raise ValueError(f"shadow segments need the same cross-NUMA peer count on every rank, got {sorted(counts)}")

    @property
    def world(self) -> int:
        return len(self.numa_nodes)

    def cross_peers(self, rank: int) -> Tuple[int, ...]:
        own = self.numa_nodes[rank]
        return tuple(p for p, node in enumerate(self.numa_nodes) if node != own)

    @property
    def num_cross(self) -> int:
        return len(self.cross_peers(self.rank))

    @property
    def num_segments(self) -> int:
        return 1 + self.num_cross

    def segment_of(self, owner: int, source: int) -> int:
        """Segment index, in ``owner``'s arena, that shadows ``source``."""
        peers = self.cross_peers(owner)
        if source not in peers:
            raise ValueError(f"rank {source} is not a cross-NUMA peer of rank {owner}")
        return peers.index(source) + 1

    def local_segments(self) -> Dict[int, int]:
        """cross peer -> local segment index (this rank as owner)."""
        return {p: self.segment_of(self.rank, p) for p in self.cross_peers(self.rank)}

    def remote_segments(self) -> Dict[int, int]:
        """cross peer -> segment index of *this* rank inside that peer's arena."""
        return {p: self.segment_of(p, self.rank) for p in self.cross_peers(self.rank)}

    @property
    def cross_mask(self) -> int:
        mask = 0
        for p in self.cross_peers(self.rank):
            mask |= 1 << p
        return mask


def segment_bytes(p2p_bytes: int, reserved_bytes: int, align: int = 2 * 1024 * 1024) -> int:
    """Per-segment stride: own buffers + reserved control region, aligned."""
    return (p2p_bytes + reserved_bytes + align - 1) // align * align


__all__ = ["ShadowLayout", "segment_bytes"]
