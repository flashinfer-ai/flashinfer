"""Pure tests for EP peer transport masks under reordered CUDA devices."""

from __future__ import annotations

import unittest

try:
    from .runtime.gpu_topology import GpuTopologyEntry, derive_ep_transport_topology
except ImportError:
    from runtime.gpu_topology import GpuTopologyEntry, derive_ep_transport_topology


def _peer(rank: int, numa: int) -> GpuTopologyEntry:
    return GpuTopologyEntry(
        global_rank=rank,
        local_rank=rank,
        pci_bus_id=f"0000:{rank + 1:02x}:00.0",
        numa_node=numa,
    )


class GpuTopologyTest(unittest.TestCase):
    def test_contiguous_ep8_masks_cross_numa_half(self):
        peers = [_peer(rank, 0 if rank < 4 else 1) for rank in range(8)]
        rank0 = derive_ep_transport_topology(peers, ep_rank=0)
        rank6 = derive_ep_transport_topology(peers, ep_rank=6)
        self.assertEqual(rank0.force_ibgda_peer_mask, 0b11110000)
        self.assertEqual(rank6.force_ibgda_peer_mask, 0b00001111)
        self.assertEqual((rank0.same_numa_peer_count, rank0.cross_numa_peer_count), (3, 4))

    def test_reordered_devices_follow_numa_not_rank_division(self):
        numa_order = (0, 0, 1, 1, 0, 0, 1, 1)
        peers = [_peer(rank, numa) for rank, numa in enumerate(numa_order)]
        rank0 = derive_ep_transport_topology(peers, ep_rank=0)
        rank2 = derive_ep_transport_topology(peers, ep_rank=2)
        self.assertEqual(rank0.force_ibgda_peer_mask, 0b11001100)
        self.assertEqual(rank2.force_ibgda_peer_mask, 0b00110011)

    def test_two_hosts_split_p2p_domains_by_host_and_numa(self):
        peers = [GpuTopologyEntry(rank, rank % 8, f"0000:{rank % 8 + 1:02x}:00.0",
                                  0 if rank % 8 < 4 else 1, host="h9" if rank < 8 else "h16")
                 for rank in range(16)]
        rank0 = derive_ep_transport_topology(peers, ep_rank=0)
        rank13 = derive_ep_transport_topology(peers, ep_rank=13)
        self.assertEqual((rank0.same_numa_peer_count, rank0.cross_numa_peer_count), (3, 12))
        self.assertEqual(rank0.force_ibgda_peer_mask, 0xFFF0)
        self.assertEqual(rank13.force_ibgda_peer_mask, 0xFFFF & ~0xF000)

    def test_shadow_layout_treats_the_other_host_as_cross(self):
        try:
            from .transport.shadow import ShadowLayout
        except ImportError:
            from transport.shadow import ShadowLayout
        peers = [GpuTopologyEntry(rank, rank % 8, f"0000:{rank % 8 + 1:02x}:00.0",
                                  0 if rank % 8 < 4 else 1, host="h9" if rank < 8 else "h16")
                 for rank in range(16)]
        layout = ShadowLayout.from_topology(derive_ep_transport_topology(peers, ep_rank=9), 9)
        self.assertEqual(layout.num_cross, 12)
        self.assertEqual(set(layout.cross_peers(9)), set(range(16)) - {8, 9, 10, 11})

    def test_same_bus_ids_on_two_hosts_are_distinct_gpus(self):
        peers = [GpuTopologyEntry(r, 0, "0000:01:00.0", 0, host=h) for r, h in enumerate(("a", "b"))]
        self.assertEqual(derive_ep_transport_topology(peers, ep_rank=0).cross_numa_peer_count, 1)

    def test_duplicate_gpu_mapping_is_rejected(self):
        peers = [_peer(0, 0), _peer(1, 0)]
        peers[1] = GpuTopologyEntry(1, 1, peers[0].pci_bus_id, 0)
        with self.assertRaisesRegex(RuntimeError, "unique GPUs"):
            derive_ep_transport_topology(peers, ep_rank=0)


if __name__ == "__main__":
    unittest.main()
